"""Exact electrical-input reuse with separate semantic and paid-point ledgers.

The existing snapshot runtime remains the only physical solver and writes its
unaltered actual journals. Search sees the original semantic identity count.
Only private, already obtained evidence can satisfy a new semantic identity;
its current-time certificate is recomputed by the unchanged pure certifier.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import time

from .nature34_clpu_environment import CLPUBudgetExhausted, CLPUReliabilityError
from .nature34_clpu_physical import certify_clpu_point

VERSION = 'nc34-semantic-electrical-cache-ledger-v1'
KEY_VERSION = 'nc34-electrical-input-equivalence-v1'
VARIANTS = ('semantic_identity', 'electrical_equivalent')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha_value(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def electrical_input_key(env, state, seconds, scope):
    """Exact frozen solver inputs; no rounding or omitted disabled load values."""
    if not isinstance(scope, Mapping) or not scope:
        raise ValueError('Nonempty frozen physical scope required')
    return {'version': KEY_VERSION, 'scope_sha256': sha_value(scope),
        'action': list(env.graph.action(state.command)),
        'case': {'name': env.scenario['scenario_id'], 'fault_lines': list(env.scenario['fault_lines'])},
        'alpha': .5, 'algorithm': 0, 'condition': deepcopy(env.condition),
        'multipliers': env.multipliers(state, seconds)}


def physical_evidence_payload(row):
    return {'raw_outputs': row['raw_outputs'], 'metadata_override': row['metadata_override']}


def _append(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
    stream.flush()


def _make_delegate(parent, cell, env, budget):
    # Importing this helper itself never instantiates OpenDSS. Runtime entry is
    # still owned by the existing, separately budgeted SnapshotOracle context.
    from run_nature34_clpu_pilot import SnapshotOracle
    return SnapshotOracle(parent, cell, env, budget)


class _CaptureRuntime:
    """Observe the returned real point without changing any runtime argument."""
    def __init__(self, runtime, owner):
        self._runtime, self._owner = runtime, owner

    def __getattr__(self, name):
        return getattr(self._runtime, name)

    def evaluate_point(self, *args, **kwargs):
        row = self._runtime.evaluate_point(*args, **kwargs)
        self._owner._captured = row
        return row


class EquivalenceOracle:
    def __init__(self, parent, cell, env, budget, variant, scope):
        if type(budget) is not int or budget <= 0 or variant not in VARIANTS:
            raise ValueError('Positive semantic budget and declared cache variant required')
        if not isinstance(scope, Mapping) or not scope:
            raise ValueError('Nonempty frozen physical scope required')
        self.parent, self.cell = Path(parent), Path(cell)
        self.env, self.budget, self.variant = env, budget, variant
        self.scope = json.loads(canonical(scope))
        self.scope_sha256 = sha_value(self.scope)
        self._condition_sha256 = file_sha(self.parent / 'condition.json')
        self.points = self.requests = self.hits = self.electrical_reuses = 0
        self.cache, self._logical_rows, self._electrical_cache = {}, {}, {}
        self.statuses = Counter()
        self.alias_certification_seconds = 0.
        self.deadline_monotonic = None
        self._delegate = self._captured = None
        self._entered = self.closed = False
        self._logical_log = self._request_log = None
        self._terminal_status = 'NOT_STARTED'

    @property
    def logical_points(self):
        return self.points

    @property
    def physical_points(self):
        return self._delegate.points if self._delegate is not None else 0

    @property
    def factory(self):
        return self._delegate.factory

    def __enter__(self):
        if self._entered or self.closed:
            raise CLPUReliabilityError('An equivalence oracle cannot be entered twice')
        self._delegate = _make_delegate(self.parent, self.cell, self.env, self.budget)
        self._delegate.__enter__()
        try:
            if self._delegate.points != 0 or self._delegate.cache:
                raise CLPUReliabilityError('Physical delegate must begin with a private empty cache')
            self._delegate.runtime = _CaptureRuntime(self._delegate.runtime, self)
            self._logical_log = gzip.open(self.cell / 'logical_points.jsonl.gz', 'wt', encoding='utf-8', compresslevel=3)
            self._request_log = (self.cell / 'logical_requests.jsonl').open('x', encoding='utf-8')
        except BaseException as exc:
            if self._logical_log is not None:
                self._logical_log.close()
            self._delegate.__exit__(type(exc), exc, exc.__traceback__)
            raise
        self._entered, self._terminal_status = True, 'RUNNING'
        return self

    def _certificate(self, physical_certificate, logical_id):
        if (physical_certificate.get('reliable') is not True
                or physical_certificate.get('status') not in ('PASS', 'FAIL')):
            raise CLPUReliabilityError('Only reliable actual certificates can support logical identities')
        return {'point_id': logical_id, 'status': physical_certificate['status'], 'reliable': True,
                'feedback': deepcopy(physical_certificate.get('feedback', []))}

    def _alias_certificate(self, evidence, identity, seconds, logical_id):
        # The pure certifier builds fresh checked outputs internally. A shallow
        # shell avoids a redundant deep copy of the very large saved raw point.
        row = {**evidence['payload'], 'time_minutes': seconds / 60., 'state_identity': identity}
        began = time.perf_counter()
        certify_clpu_point(row, self._delegate.runtime.initial_metadata,
                           self._delegate.runtime.physical_config)
        elapsed = time.perf_counter() - began
        self.alias_certification_seconds += elapsed
        certificate = self._certificate({'status': row['status'], 'reliable': row['reliable'],
            'feedback': row['structured_feedback']['violations']}, logical_id)
        if certificate['status'] != evidence['certificate']['status']:
            raise CLPUReliabilityError('Pure recertification changed the physical evidence status')
        return certificate, elapsed

    def certify(self, state, seconds):
        if not self._entered or self.closed:
            raise CLPUReliabilityError('Logical certification requires an active private oracle')
        if self.deadline_monotonic is not None and time.perf_counter() > self.deadline_monotonic:
            raise TimeoutError('Frozen cache-pilot worker deadline exceeded')
        if sha_value(self.scope) != self.scope_sha256:
            raise CLPUReliabilityError('Frozen equivalence scope changed within a private oracle')
        identity = {**self.env.identity(state, seconds), 'alpha': .5,
                    'base_condition_sha256': self._condition_sha256}
        semantic_key = sha_value(identity)
        self.requests += 1
        request = {'request': self.requests, 'key': semantic_key,
                   'semantic_cache_hit': False, 'electrical_reuse': False}
        try:
            if semantic_key in self.cache:
                self.hits += 1
                certificate = self.cache[semantic_key]
                logical = self._logical_rows[certificate['point_id']]
                request.update(status=certificate['status'], semantic_cache_hit=True,
                    logical_point_id=certificate['point_id'], physical_point_id=logical['physical_point_id'])
                _append(self._request_log, request)
                return deepcopy(certificate)
            if self.points >= self.budget:
                request['status'] = 'BUDGET_UNKNOWN'
                _append(self._request_log, request)
                raise CLPUBudgetExhausted('Frozen semantic point budget exhausted')
            key = electrical_input_key(self.env, state, seconds, self.scope)
            fingerprint = sha_value(key)
            logical_id, alias_seconds = self.points, 0.
            reused = self.variant == 'electrical_equivalent' and fingerprint in self._electrical_cache
            if reused:
                evidence = self._electrical_cache[fingerprint]
                certificate, alias_seconds = self._alias_certificate(evidence, identity, seconds, logical_id)
                physical_id = evidence['physical_point_id']
            else:
                self._captured = None
                before = self._delegate.points
                physical_certificate = self._delegate.certify(state, seconds)
                if self._delegate.points != before + 1 or physical_certificate.get('point_id') != before:
                    raise CLPUReliabilityError('A new actual identity must produce one counted physical point')
                if self._captured is None:
                    raise CLPUReliabilityError('Actual runtime returned no captured point evidence')
                certificate = self._certificate(physical_certificate, logical_id)
                physical_id = before
                payload = physical_evidence_payload(self._captured)
                evidence = {'payload': payload, 'physical_point_id': physical_id,
                    'certificate': deepcopy(physical_certificate), 'sha256': sha_value(payload)}
                self._electrical_cache.setdefault(fingerprint, evidence)
            logical = {'logical_point_id': logical_id, 'physical_point_id': physical_id,
                'semantic_identity': identity, 'semantic_identity_sha256': semantic_key,
                'electrical_input_key': key, 'electrical_input_sha256': fingerprint,
                'dynamic_state': state.to_dict(), 'time_seconds': float(seconds),
                'certificate': certificate,
                'reuse_kind': 'ELECTRICAL_EQUIVALENCE' if reused else 'ACTUAL_PHYSICAL',
                'physical_evidence_sha256': evidence['sha256'],
                'alias_certification_seconds': alias_seconds}
            _append(self._logical_log, logical)
            self.points += 1
            self.electrical_reuses += int(reused)
            self.statuses[certificate['status']] += 1
            self.cache[semantic_key] = deepcopy(certificate)
            self._logical_rows[logical_id] = logical
            request.update(status=certificate['status'], electrical_reuse=reused,
                logical_point_id=logical_id, physical_point_id=physical_id)
            _append(self._request_log, request)
            return deepcopy(certificate)
        except CLPUBudgetExhausted:
            raise
        except BaseException as exc:
            request.update(status='ERROR', error=repr(exc))
            _append(self._request_log, request)
            raise

    def summary(self):
        stats = self.factory.stats() if self._delegate is not None else None
        actual = stats['returned']['numerical_solution_entries'] if stats is not None else 0
        return {'version': VERSION, 'variant': self.variant, 'scope_sha256': self.scope_sha256,
            'points': self.points, 'logical_points': self.points, 'physical_points': self.physical_points,
            'point_budget': self.budget, 'requests': self.requests, 'cache_hits': self.hits,
            'semantic_cache_hits': self.hits, 'electrical_reuses': self.electrical_reuses,
            'statuses': dict(self.statuses), 'actual_numerical_entries': actual,
            'initialization_numerical_entries': 3 if self._entered else 0,
            'expected_numerical_entries': 3 + 4 * self.physical_points if self._entered else 0,
            'alias_certification_seconds': self.alias_certification_seconds,
            'private_cache_only': True, 'cross_arm_cache_reuse': False,
            'closed': self.closed, 'status': self._terminal_status}

    def __exit__(self, typ, value, tb):
        pending = None
        try:
            self._delegate.__exit__(typ, value, tb)
            if typ is None and (self.points != self.physical_points + self.electrical_reuses
                    or self.variant == 'semantic_identity' and self.electrical_reuses != 0):
                raise CLPUReliabilityError('Semantic and physical point ledgers do not reconcile')
        except BaseException as exc:
            pending = exc
        finally:
            for stream in (self._logical_log, self._request_log):
                if stream is not None:
                    stream.close()
            self.closed = True
            self._terminal_status = 'COMPLETE_DUAL_LEDGER_PENDING_AUDIT' if typ is None and pending is None else 'STOPPED'
            (self.cell / 'equivalent_cache_summary.json').write_text(
                json.dumps(self.summary(), indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
        if pending is not None:
            raise pending
        return False


EquivalentCacheOracle = EquivalenceOracle
