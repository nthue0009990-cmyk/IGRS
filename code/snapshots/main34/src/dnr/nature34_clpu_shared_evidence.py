"""One physical runtime, independent search identities, causal evidence pools.

Private and shared modes use the same implementation and one initialization.
Only reliable, exact-input physical evidence crosses views in shared mode.
The unchanged SnapshotOracle owns every real call and its original journal.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
import gzip
import json
from pathlib import Path
import time

from .nature34_clpu_environment import CLPUBudgetExhausted, CLPUReliabilityError
from .nature34_clpu_equivalent_cache import (
    _CaptureRuntime, _append, _make_delegate, canonical, electrical_input_key,
    file_sha, physical_evidence_payload, sha_value,
)
from .nature34_clpu_physical import certify_clpu_point

VERSION = 'clpu-shared-physical-evidence-v1'
MODES = ('private', 'shared')
VIEWS = ('EARLY', 'GUARD')


def _save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def _certificate(value, logical_id):
    if value.get('reliable') is not True or value.get('status') not in ('PASS', 'FAIL'):
        raise CLPUReliabilityError('Only reliable PASS/FAIL evidence may enter an evidence pool')
    return {'point_id': logical_id, 'status': value['status'], 'reliable': True,
            'feedback': deepcopy(value.get('feedback', []))}


class SharedEvidenceService:
    def __init__(self, parent, cell, env, scope, *, mode='shared', physical_budget=512, logical_budget=256):
        if mode not in MODES or any(type(n) is not int or n <= 0 for n in (physical_budget, logical_budget)):
            raise ValueError('Declared mode and positive integer budgets required')
        if not isinstance(scope, Mapping) or not scope:
            raise ValueError('Nonempty frozen physical scope required')
        self.parent, self.cell = Path(parent).resolve(), Path(cell).resolve()
        self.physical_cell = self.cell / 'physical'
        self.env, self.mode = env, mode
        self.scope = json.loads(canonical(scope))
        self.scope_sha256 = sha_value(self.scope)
        self.physical_budget, self.logical_budget = physical_budget, logical_budget
        self._condition_sha256 = file_sha(self.parent / 'condition.json')
        self._environment_sha256 = sha_value(self._environment_binding())
        self.views, self._pools, self._delegate_caches = {}, {}, {}
        self._delegate = self._captured = self._events = self._active = None
        self._entered = self.closed = self._poisoned = False
        self._initialization_complete = False
        self._sequence = 0
        self.deadline_monotonic = None
        self.status = 'NOT_STARTED'

    def _environment_binding(self):
        return {'scenario': self.env.scenario, 'condition': self.env.condition,
                'event': self.env.event, 'load_names': list(self.env.load_names),
                'horizon': self.env.horizon, 'slot_seconds': self.env.slot_seconds,
                'sample_seconds': self.env.sample_seconds}

    @property
    def physical_points(self):
        return self._delegate.points if self._delegate is not None else 0

    @property
    def factory(self):
        return self._delegate.factory

    def _event(self, kind, **fields):
        self._sequence += 1
        _append(self._events, {'sequence': self._sequence, 'kind': kind, **fields})
        return self._sequence

    def _ensure(self):
        if not self._entered or self.closed or self._poisoned:
            raise CLPUReliabilityError('Shared service is not active or has a retained error')
        if sha_value(self.scope) != self.scope_sha256 or sha_value(self._environment_binding()) != self._environment_sha256:
            raise CLPUReliabilityError('Frozen physical scope or environment changed')
        if self.deadline_monotonic is not None and time.perf_counter() > self.deadline_monotonic:
            raise TimeoutError('Shared evidence task deadline exceeded')

    def __enter__(self):
        if self._entered or self.closed:
            raise CLPUReliabilityError('A physical service cannot be entered twice')
        self.cell.mkdir(parents=True, exist_ok=True)
        self.physical_cell.mkdir(exist_ok=False)
        self._events = (self.cell / 'service_events.jsonl').open('x', encoding='utf-8')
        self._event('SERVICE_STARTED', mode=self.mode, scope_sha256=self.scope_sha256,
                    physical_budget=self.physical_budget, logical_budget=self.logical_budget)
        delegate_entered = False
        try:
            self._delegate = _make_delegate(self.parent, self.physical_cell, self.env, self.physical_budget)
            self._delegate.__enter__()
            delegate_entered = True
            if self._delegate.points != 0 or self._delegate.cache:
                raise CLPUReliabilityError('Shared physical delegate must start empty')
            self._delegate.runtime = _CaptureRuntime(self._delegate.runtime, self)
            self._initialization_complete = self._entered = True
            self.status = 'RUNNING'
            self._event('SERVICE_READY', physical_points=0)
            return self
        except BaseException as exc:
            self._poisoned, self.closed, self.status = True, True, 'STARTUP_ERROR'
            self._event('SERVICE_ERROR', error=repr(exc))
            try:
                # SnapshotOracle itself owns cleanup when its entry fails.
                if delegate_entered:
                    self._delegate.__exit__(type(exc), exc, exc.__traceback__)
            finally:
                self._events.close()
                _save(self.cell / 'service_summary.json', self.summary())
            raise

    def view(self, name):
        self._ensure()
        if name not in VIEWS or name in self.views:
            raise ValueError('Each declared logical view may be created once')
        if self._active is not None or (name == 'GUARD' and ('EARLY' not in self.views or not self.views['EARLY'].closed)):
            raise CLPUReliabilityError('EARLY must close before GUARD; views cannot overlap')
        if name == 'EARLY' and self.views:
            raise CLPUReliabilityError('EARLY is the first logical view')
        item = LogicalEvidenceView(self, name)
        self.views[name] = item
        return item

    def _actual(self, view, state, seconds, identity, key, fingerprint):
        if self.physical_points >= self.physical_budget:
            raise CLPUBudgetExhausted('Shared actual physical budget exhausted')
        # SnapshotOracle has a semantic cache of its own. Isolate that cache by
        # pool, without altering its identities, point IDs, counters or journals.
        self._delegate.cache = self._delegate_caches.setdefault(view.pool, {})
        if sha_value(identity) in self._delegate.cache:
            raise CLPUReliabilityError('Unexpected delegate hit bypassing the evidence pool')
        before = self.physical_points
        self._captured = None
        self._event('ACTUAL_STARTED', view=view.name, pool=view.pool, physical_point_id=before,
                    semantic_identity_sha256=sha_value(identity), electrical_input_key=key,
                    electrical_input_sha256=fingerprint)
        try:
            cert = self._delegate.certify(state, seconds)
        finally:
            view._physical_points += self.physical_points - before
        if self.physical_points != before + 1 or cert.get('point_id') != before or self._captured is None:
            raise CLPUReliabilityError('A physical dispatch must produce exactly one captured actual point')
        _certificate(cert, view.points)
        payload = physical_evidence_payload(self._captured)
        digest = sha_value(payload)
        sequence = self._event('ACTUAL_COMPLETE', view=view.name, pool=view.pool, physical_point_id=before,
                              status=cert['status'], physical_evidence_sha256=digest,
                              electrical_input_sha256=fingerprint)
        evidence = {'payload': payload, 'physical_point_id': before, 'certificate': deepcopy(cert),
                    'sha256': digest, 'source_view': view.name, 'source_event_sequence': sequence}
        self._pools.setdefault(view.pool, {})[fingerprint] = evidence
        return evidence

    def summary(self):
        factory = getattr(self._delegate, 'factory', None)
        stats = factory.stats() if factory is not None else None
        actual = stats['returned']['numerical_solution_entries'] if stats is not None else None
        return {'version': VERSION, 'mode': self.mode, 'status': self.status, 'closed': self.closed,
                'scope_sha256': self.scope_sha256, 'physical_points': self.physical_points,
                'physical_budget': self.physical_budget, 'logical_budget_per_view': self.logical_budget,
                'logical_points': sum(v.points for v in self.views.values()),
                'actual_numerical_entries': actual,
                'initialization_numerical_entries': 3 if self._initialization_complete else None,
                'expected_numerical_entries': 3 + 4 * self.physical_points if self._initialization_complete and not self._poisoned else None,
                'physical_cost_complete': self._initialization_complete and not self._poisoned,
                'runtime_initializations': int(self._initialization_complete),
                'runtime_initialization_attempts': int(self._delegate is not None),
                'cross_task_evidence_reuse': False, 'views': {k: v.summary() for k, v in self.views.items()}}

    def __exit__(self, typ, value, tb):
        pending = None
        if typ is not None:
            self._poisoned = True
        try:
            if self._active is not None or any(not v.closed for v in self.views.values()):
                raise CLPUReliabilityError('Logical view must close before its service')
            if not self._poisoned:
                if (sum(v._physical_points for v in self.views.values()) != self.physical_points
                        or sum(v.points - v.electrical_reuses for v in self.views.values()) != self.physical_points):
                    raise CLPUReliabilityError('Shared logical and actual ledgers do not reconcile')
        except BaseException as exc:
            pending, self._poisoned = exc, True
        if typ is None and self._poisoned and pending is None:
            pending = CLPUReliabilityError('Retained service error prevents successful close')
        try:
            error = value or pending or (CLPUReliabilityError('Retained service error') if self._poisoned else None)
            self._delegate.__exit__(type(error) if error is not None else None, error, tb)
        except BaseException as exc:
            pending, self._poisoned = exc, True
        finally:
            for view in self.views.values():
                if not view.closed:
                    view._close(stopped=True)
            self.closed = True
            self.status = 'STOPPED' if self._poisoned else 'COMPLETE_SHARED_EVIDENCE_PENDING_AUDIT'
            self._event('SERVICE_CLOSED', status=self.status, physical_points=self.physical_points)
            self._events.close()
            _save(self.cell / 'service_summary.json', self.summary())
        if pending is not None:
            raise pending
        return False


class LogicalEvidenceView:
    def __init__(self, service, name):
        self.service, self.name, self.env = service, name, service.env
        self.cell, self.budget = service.cell / name, service.logical_budget
        self.pool = 'SHARED' if service.mode == 'shared' else name
        self.points = self.requests = self.hits = self.electrical_reuses = self.cross_view_reuses = 0
        self._physical_points = 0
        self._numerical_before = self._numerical_at_close = None
        self.cache, self._rows = {}, {}
        self.statuses = Counter()
        self.alias_certification_seconds = 0.
        self._entered = self.closed = False
        self._logical_log = self._request_log = None
        self.status = 'NOT_STARTED'
        self.deadline_monotonic = None

    @property
    def logical_points(self):
        return self.points

    @property
    def physical_points(self):
        return self._physical_points

    @property
    def total_physical_points(self):
        return self.service.physical_points

    @property
    def factory(self):
        return self.service.factory

    def __enter__(self):
        self.service._ensure()
        if self._entered or self.closed or self.service._active is not None:
            raise CLPUReliabilityError('Logical view cannot be reentered or overlap')
        self.cell.mkdir(parents=True, exist_ok=True)
        try:
            self._request_log = (self.cell / 'logical_requests.jsonl').open('x', encoding='utf-8')
            self._logical_log = gzip.open(self.cell / 'logical_points.jsonl.gz', 'xb', compresslevel=3)
            # Text wrapper retains exclusive-create semantics for the gzip file.
            import io
            self._logical_log = io.TextIOWrapper(self._logical_log, encoding='utf-8')
        except BaseException:
            for stream in (self._request_log, self._logical_log):
                if stream is not None:
                    stream.close()
            self.service._poisoned = True
            raise
        self._entered, self.service._active, self.status = True, self, 'RUNNING'
        self._numerical_before = self.factory.stats()['returned']['numerical_solution_entries']
        self.service._event('VIEW_STARTED', view=self.name, pool=self.pool,
                            physical_points_before=self.total_physical_points, logical_budget=self.budget)
        return self

    def _alias(self, evidence, identity, seconds):
        row = {**evidence['payload'], 'time_minutes': seconds / 60., 'state_identity': identity}
        began = time.perf_counter()
        certify_clpu_point(row, self.service._delegate.runtime.initial_metadata,
                           self.service._delegate.runtime.physical_config)
        elapsed = time.perf_counter() - began
        self.alias_certification_seconds += elapsed
        cert = _certificate({'status': row['status'], 'reliable': row['reliable'],
                             'feedback': row['structured_feedback']['violations']}, self.points)
        if cert['status'] != evidence['certificate']['status']:
            raise CLPUReliabilityError('Current-time recertification changed evidence status')
        return cert, elapsed

    def certify(self, state, seconds):
        if not self._entered or self.closed or self.service._active is not self:
            raise CLPUReliabilityError('Certification requires the active logical view')
        request = None
        try:
            self.service._ensure()
            if self.deadline_monotonic is not None and time.perf_counter() > self.deadline_monotonic:
                raise TimeoutError('Logical view deadline exceeded')
            identity = {**self.env.identity(state, seconds), 'alpha': .5,
                        'base_condition_sha256': self.service._condition_sha256}
            semantic = sha_value(identity)
            self.requests += 1
            sequence = self.service._event('REQUEST_STARTED', view=self.name, pool=self.pool,
                request=self.requests, semantic_identity_sha256=semantic,
                logical_points_before=self.points, physical_points_before=self.total_physical_points)
            request = {'request': self.requests, 'key': semantic, 'service_request_sequence': sequence,
                       'semantic_cache_hit': False, 'electrical_reuse': False, 'cross_view_reuse': False}
            if semantic in self.cache:
                self.hits += 1
                certificate = self.cache[semantic]
                logical = self._rows[certificate['point_id']]
                request.update(semantic_cache_hit=True, status=certificate['status'],
                    logical_point_id=certificate['point_id'], physical_point_id=logical['physical_point_id'],
                    source_view=logical['source_view'], source_event_sequence=logical['source_event_sequence'])
            else:
                if self.points >= self.budget:
                    raise CLPUBudgetExhausted('Independent logical view budget exhausted')
                key = electrical_input_key(self.env, state, seconds, self.service.scope)
                fingerprint = sha_value(key)
                pool = self.service._pools.setdefault(self.pool, {})
                evidence = pool.get(fingerprint)
                reused = evidence is not None
                if reused:
                    certificate, alias_seconds = self._alias(evidence, identity, seconds)
                else:
                    evidence = self.service._actual(self, state, seconds, identity, key, fingerprint)
                    certificate, alias_seconds = _certificate(evidence['certificate'], self.points), 0.
                crossed = reused and evidence['source_view'] != self.name
                sequence = self.service._event('LOGICAL_COMPLETE', view=self.name, pool=self.pool,
                    request=self.requests, logical_point_id=self.points, physical_point_id=evidence['physical_point_id'],
                    source_event_sequence=evidence['source_event_sequence'], electrical_input_sha256=fingerprint,
                    reuse_kind='ELECTRICAL_EQUIVALENCE' if reused else 'ACTUAL_PHYSICAL')
                logical = {'view': self.name, 'pool': self.pool, 'logical_point_id': self.points,
                    'physical_point_id': evidence['physical_point_id'], 'service_sequence': sequence,
                    'source_event_sequence': evidence['source_event_sequence'], 'source_view': evidence['source_view'],
                    'semantic_identity': identity, 'semantic_identity_sha256': semantic,
                    'electrical_input_key': key, 'electrical_input_sha256': fingerprint,
                    'dynamic_state': state.to_dict(), 'time_seconds': float(seconds), 'certificate': certificate,
                    'reuse_kind': 'ELECTRICAL_EQUIVALENCE' if reused else 'ACTUAL_PHYSICAL',
                    'cross_view_reuse': crossed, 'physical_evidence_sha256': evidence['sha256'],
                    'alias_certification_seconds': alias_seconds}
                _append(self._logical_log, logical)
                self._rows[self.points] = logical
                self.points += 1
                self.electrical_reuses += int(reused)
                self.cross_view_reuses += int(crossed)
                self.statuses[certificate['status']] += 1
                self.cache[semantic] = deepcopy(certificate)
                request.update(status=certificate['status'], electrical_reuse=reused, cross_view_reuse=crossed,
                    logical_point_id=certificate['point_id'], physical_point_id=evidence['physical_point_id'],
                    source_view=evidence['source_view'], source_event_sequence=evidence['source_event_sequence'])
            request['service_completion_sequence'] = self.service._event('REQUEST_COMPLETE', view=self.name, **request)
            _append(self._request_log, request)
            return deepcopy(certificate)
        except CLPUBudgetExhausted:
            request.update(status='BUDGET_UNKNOWN')
            request['service_completion_sequence'] = self.service._event('REQUEST_COMPLETE', view=self.name, **request)
            _append(self._request_log, request)
            raise
        except BaseException as exc:
            self.service._poisoned = True
            self.service._event('VIEW_ERROR', view=self.name, error=repr(exc))
            if request is not None:
                request.update(status='ERROR', error=repr(exc))
                _append(self._request_log, request)
            raise

    def summary(self):
        actual = (self._numerical_at_close if self.closed else
            self.factory.stats()['returned']['numerical_solution_entries'] - self._numerical_before
            if self._numerical_before is not None else None)
        return {'version': VERSION, 'view': self.name, 'pool': self.pool, 'mode': self.service.mode,
            'scope_sha256': self.service.scope_sha256, 'status': self.status, 'closed': self.closed,
            'points': self.points, 'logical_points': self.points, 'point_budget': self.budget,
            'physical_points': self.physical_points, 'requests': self.requests,
            'cache_hits': self.hits, 'semantic_cache_hits': self.hits,
            'electrical_reuses': self.electrical_reuses, 'cross_view_reuses': self.cross_view_reuses,
            'statuses': dict(self.statuses), 'alias_certification_seconds': self.alias_certification_seconds,
            'initialization_numerical_entries_attributed': 0,
            'actual_numerical_entries_attributed': actual,
            'expected_numerical_entries_attributed': 4 * self.physical_points,
            'physical_evidence_directory': '../physical'}

    def _close(self, *, stopped):
        if self._numerical_before is not None:
            self._numerical_at_close = self.factory.stats()['returned']['numerical_solution_entries'] - self._numerical_before
        for stream in (self._logical_log, self._request_log):
            if stream is not None:
                stream.close()
        self.closed = True
        self.status = 'STOPPED' if stopped else 'CLOSED_LOGICAL_VIEW_PENDING_AUDIT'
        if self.service._active is self:
            self.service._active = None
        self.service._event('VIEW_CLOSED', view=self.name, status=self.status,
                            physical_points_after=self.total_physical_points, logical_points=self.points)
        self.cell.mkdir(parents=True, exist_ok=True)
        _save(self.cell / 'view_summary.json', self.summary())

    def __exit__(self, typ, value, tb):
        if typ is not None:
            self.service._poisoned = True
        self._close(stopped=typ is not None or self.service._poisoned)
        return False
