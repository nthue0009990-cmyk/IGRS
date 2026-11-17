"""Cold-task NC13 exact evidence sharing; original physics/search stay frozen."""
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

from .nature13_clpu_environment import CLPUBudgetExhausted, CLPUReliabilityError


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def append(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')
    stream.flush()


def physical_input(request):
    return {k: request[k] for k in ('action', 'case', 'alpha', 'algorithm', 'multipliers')}


def replay_certificate(row, request, metadata, config, logical_id):
    """Rebuild current identity feedback from identical saved numerical input."""
    from .nature13_clpu_physical import MODEL_SCOPE, check_nature13_constraints, verify_nature13_isolation
    from .restoration_ac_audit import audit_ac_snapshot
    from .nature34_clpu_physical import structured_feedback
    expected = {'action': row['action'], 'case': {'name': row['case'], 'fault_lines': row['fault_lines']},
                'alpha': row['alpha'], 'algorithm': row['algorithm'], 'multipliers': row['clpu_multipliers']}
    if physical_input(request) != expected or row['scope'] != MODEL_SCOPE:
        raise CLPUReliabilityError('Saved physical input/scope does not match current request')
    if row.get('reliable') is not True or row.get('status') not in ('PASS', 'FAIL'):
        raise CLPUReliabilityError('Unknown evidence cannot be reused')
    merged = {**metadata, **row['metadata_override']}
    raw = row['raw_outputs']
    known = check_nature13_constraints(raw, merged, **config['constraint_tolerances'])
    iso = verify_nature13_isolation(raw, merged)
    ac = audit_ac_snapshot(raw, merged, **config['ac_audit_tolerances'])
    sections = ('numerical', 'topology', 'voltage', 'line_thermal', 'transformer_thermal', 'gfm_active_limit')
    reliable = (raw['converged'] and ac['status'] == iso['status'] == 'PASS'
        and known['numerical']['status'] == known['topology']['status'] == 'PASS'
        and not any(known[k]['status'] == 'UNKNOWN' or known[k].get('unknown') for k in sections))
    status = ('PASS' if known['known_constraints_satisfied'] is True else 'FAIL') if reliable else 'UNKNOWN'
    if not reliable or status != row['status']:
        raise CLPUReliabilityError('Saved physical evidence fails current recertification')
    feedback = structured_feedback(known, raw, reliable=True, time_minutes=request['time_minutes'])
    return {'point_id': logical_id, 'status': status, 'reliable': True, 'feedback': feedback['violations']}


class EvidenceService:
    def __init__(self, root, cell, env, config, scope, phase_order, deadline,
                 *, physical_budget=512, logical_budget=256, entry_budget=7171, runtime_factory=None):
        self.root, self.cell, self.env = Path(root), Path(cell), env
        self.config, self.scope = deepcopy(config), deepcopy(scope)
        self.scope_hash = digest(self.scope)
        self.config_hash = digest(self.config)
        self.phase_order = list(phase_order)
        if self.phase_order not in (['GUARD', 'EARLY'], ['EARLY', 'GUARD']):
            raise ValueError('Two declared phases required')
        self.physical_budget, self.logical_budget, self.entry_budget = physical_budget, logical_budget, entry_budget
        self.deadline, self.runtime_factory = deadline, runtime_factory
        self.pool, self.views, self.events = {}, {}, []
        self.active = None
        self.closed = self.poisoned = False
        self.runtime = self.budget = None
        self._log = None
        self.binding = digest(self.environment_binding())

    def environment_binding(self):
        return {'scenario': self.env.scenario, 'condition': self.env.condition, 'event': self.env.event,
                'load_names': list(self.env.load_names), 'horizon': self.env.horizon,
                'slot_seconds': self.env.slot_seconds, 'sample_seconds': self.env.sample_seconds}

    def ensure(self):
        if self.closed or self.poisoned:
            raise CLPUReliabilityError('Evidence service closed or failed')
        if time.monotonic() >= self.deadline:
            raise TimeoutError('Cold task deadline reached')
        if (digest(self.scope) != self.scope_hash or digest(self.environment_binding()) != self.binding
                or digest(self.config) != self.config_hash):
            raise CLPUReliabilityError('Frozen scope or context changed')

    def __enter__(self):
        from .nature13_clpu_physical import Nature13Budget, Nature13CLPUPhysicalRuntime
        self.cell.mkdir(parents=True, exist_ok=True)
        (self.cell/'physical').mkdir(exist_ok=False)
        self._log = (self.cell/'numerical_events.jsonl').open('x', encoding='utf-8')
        def durable(event):
            append(self._log, event)
            os.fsync(self._log.fileno())
        self.budget = Nature13Budget(max_queries=self.physical_budget, max_entries=self.entry_budget,
                                    deadline_monotonic=self.deadline, emit=durable)
        try:
            factory = self.runtime_factory or Nature13CLPUPhysicalRuntime
            self.runtime = factory(self.root, self.cell, self.config, budget=self.budget, events=self.events)
            save(self.cell/'metadata.json', self.runtime.initial_metadata)
            save(self.cell/'runtime_adaptation.json', self.runtime.metadata)
        except BaseException:
            self.poisoned = True
            self.__exit__(*__import__('sys').exc_info())
            raise
        return self

    def view(self, name):
        self.ensure()
        if self.active is not None or name in self.views or len(self.views) >= 2:
            raise CLPUReliabilityError('Views must be serial and unique')
        if name != self.phase_order[len(self.views)] or any(not v.closed for v in self.views.values()):
            raise CLPUReliabilityError('Frozen phase order violated')
        view = EvidenceView(self, name)
        self.views[name] = view
        return view

    def actual(self, view, request, fingerprint):
        if self.budget.queries >= self.physical_budget:
            raise CLPUBudgetExhausted('Actual physical query limit')
        qid = self.budget.queries
        path = self.cell/'physical'/f'{qid:04d}.json.gz'
        save(self.cell/'physical'/f'{qid:04d}_request.json', request)
        before = self.budget.queries
        try:
            row = self.runtime.evaluate_point(**{k: request[k] for k in
                ('action', 'case', 'alpha', 'algorithm', 'multipliers', 'time_minutes', 'state_identity')}, query_id=qid)
            with gzip.open(path, 'xt', encoding='utf-8', compresslevel=3) as out:
                json.dump(row, out, ensure_ascii=False, allow_nan=True)
        except BaseException:
            partial = getattr(self.runtime, 'last_query_evidence', None)
            if partial is not None and partial.get('query_id') == qid:
                from .nature34_clpu_physical import engine_numeric_lists
                with gzip.open(self.cell/'physical'/f'{qid:04d}_partial.json.gz', 'xt', encoding='utf-8') as out:
                    json.dump(engine_numeric_lists(partial), out, ensure_ascii=False, allow_nan=True)
            self.poisoned = True
            raise
        finally:
            view._physical_points += self.budget.queries - before
        if row['reliable'] is not True or row['status'] not in ('PASS', 'FAIL'):
            self.poisoned = True
            raise CLPUReliabilityError('Actual physical point is UNKNOWN; raw evidence retained')
        evidence = {'row': row, 'source_view': view.name, 'point_id': qid, 'sha256': file_sha(path)}
        self.pool[fingerprint] = evidence
        return evidence

    def summary(self):
        return {'scope_sha256': self.scope_hash, 'phase_order': self.phase_order,
            'closed': self.closed, 'poisoned': self.poisoned,
            'physical_points': self.budget.queries if self.budget else 0,
            'budget': self.budget.stats() if self.budget else None,
            'cross_task_evidence_reuse': False, 'runtime_initializations': int(self.runtime is not None),
            'logical_points': sum(v.points for v in self.views.values()),
            'views': {k: v.summary() for k, v in self.views.items()}}

    def __exit__(self, typ, value, tb):
        self.poisoned |= typ is not None
        if self.active is not None:
            self.active.__exit__(typ or RuntimeError, value, tb)
            self.poisoned = True
        if typ is None and self.runtime is not None:
            if sum(v._physical_points for v in self.views.values()) != self.budget.queries:
                self.poisoned = True
        self.closed = True
        if self._log is not None:
            self._log.close()
        save(self.cell/'engine_calls.json', self.events)
        save(self.cell/'service_summary.json', self.summary())
        if typ is None and self.poisoned:
            raise CLPUReliabilityError('Evidence service did not close consistently')
        return False


class EvidenceView:
    def __init__(self, service, name):
        self.service, self.name, self.env = service, name, service.env
        self.budget = service.logical_budget
        self.points = self.requests = self.hits = self.reuses = self.cross_reuses = self._physical_points = 0
        self.cache, self.statuses = {}, Counter()
        self.closed = False
        self.log = None

    @property
    def physical_points(self):
        return self._physical_points

    def __enter__(self):
        self.service.ensure()
        if self.closed or self.service.active is not None:
            raise CLPUReliabilityError('Logical view reentry')
        folder = self.service.cell/self.name
        folder.mkdir(exist_ok=True)
        self.log = gzip.open(folder/'requests.jsonl.gz', 'xt', encoding='utf-8', compresslevel=3)
        self.service.active = self
        return self

    def certify(self, state, seconds):
        self.service.ensure()
        if self.closed or self.service.active is not self:
            raise CLPUReliabilityError('Inactive logical view')
        request = self.env.snapshot_request(state, seconds, algorithm=0)
        semantic = digest(request['state_identity'])
        key = digest({'scope': self.service.scope, 'input': physical_input(request)})
        self.requests += 1
        entry = {'request_id': self.requests-1, 'state': state.to_dict(), 'time_seconds': float(seconds),
                 'semantic_sha256': semantic, 'physical_input_sha256': key}
        try:
            if semantic in self.cache:
                certificate, physical_id = self.cache[semantic]
                self.hits += 1
                entry.update(kind='HIT', physical_point_id=physical_id)
            else:
                if self.points >= self.budget:
                    raise CLPUBudgetExhausted('Independent logical point limit')
                evidence = self.service.pool.get(key)
                reused = evidence is not None
                if not reused:
                    evidence = self.service.actual(self, request, key)
                certificate = replay_certificate(evidence['row'], request, self.service.runtime.initial_metadata,
                                                 self.service.config, self.points)
                crossed = reused and evidence['source_view'] != self.name
                physical_id = evidence['point_id']
                entry.update(kind='REUSE' if reused else 'ACTUAL', physical_point_id=physical_id,
                    source_view=evidence['source_view'], source_sha256=evidence['sha256'], cross_view_reuse=crossed)
                self.cache[semantic] = (deepcopy(certificate), physical_id)
                self.points += 1
                self.reuses += int(reused)
                self.cross_reuses += int(crossed)
                self.statuses[certificate['status']] += 1
            entry.update(certificate=deepcopy(certificate), logical_points_after=self.points,
                         physical_points_after=self.physical_points)
            append(self.log, entry)
            return deepcopy(certificate)
        except CLPUBudgetExhausted:
            append(self.log, {**entry, 'kind': 'BUDGET', 'logical_points_after': self.points,
                              'physical_points_after': self.physical_points})
            raise
        except BaseException as exc:
            self.service.poisoned = True
            append(self.log, {**entry, 'kind': 'ERROR', 'error': repr(exc),
                              'logical_points_after': self.points, 'physical_points_after': self.physical_points})
            raise

    def summary(self):
        return {'view': self.name, 'closed': self.closed, 'points': self.points, 'logical_points': self.points,
            'physical_points': self.physical_points, 'requests': self.requests, 'semantic_hits': self.hits,
            'electrical_reuses': self.reuses, 'cross_view_reuses': self.cross_reuses, 'statuses': dict(self.statuses),
            'logical_budget': self.budget}

    def __exit__(self, typ, value, tb):
        self.closed = True
        self.service.active = None
        if typ is not None:
            self.service.poisoned = True
        if self.log is not None:
            self.log.close()
        save(self.service.cell/self.name/'view_summary.json', self.summary())
        return False
