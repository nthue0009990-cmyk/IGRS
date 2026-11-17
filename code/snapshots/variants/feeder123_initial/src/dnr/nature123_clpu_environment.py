"""CLPU request preparation for the sparse NC123 graph, with no engine import.

Graph-connected loads are candidate supplies only. A physical oracle must verify
phase connectivity and the named constraints. Synthetic oracle outcomes can
never produce a physical completion certificate.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
import math
import re

from .clpu_curves import multiplier


RELIABILITY_CHECKS = ('numerical_convergence', 'topology', 'isolation',
                      'phase_connectivity', 'ac_balance')
CONSTRAINT_CHECKS = ('voltage', 'line_phase_thermal', 'transformer_winding')
CHECKS = RELIABILITY_CHECKS + CONSTRAINT_CHECKS


class CLPUBudgetExhausted(RuntimeError):
    """A caller-owned pre-call budget refused a request; no new result exists."""


class CLPUReliabilityError(RuntimeError):
    """Missing or unbound evidence cannot certify a complete path."""


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_canonical(value).encode('utf-8')).hexdigest()


def _sha(value, name):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError(name + ' must be a lowercase SHA-256')
    return value


def _number(value, name, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(name + ' must be finite and nonnegative')
    return float(value)


@dataclass(frozen=True)
class DynamicState:
    command: int
    step: int
    pickups: tuple
    actions: tuple
    context_sha256: str

    def to_dict(self):
        return {'command': self.command, 'step': self.step, 'pickups': list(self.pickups),
                'actions': list(self.actions), 'context_sha256': self.context_sha256}

    @classmethod
    def from_dict(cls, value):
        if (type(value) is not dict or type(value.get('command')) is not int
                or type(value.get('step')) is not int or type(value.get('pickups')) is not list
                or type(value.get('actions')) is not list):
            raise ValueError('Strict state types required; no numeric/string coercion')
        return cls(value['command'], value['step'], tuple(value['pickups']),
                   tuple(value['actions']), _sha(value.get('context_sha256'), 'state context'))


class CLPUEnvironment:
    interface_version = 'nc123-clpu-sparse-requests-v1'

    def __init__(self, graph, *, model_fingerprint, condition, event,
                 horizon=30, slot_seconds=60, sample_seconds=30, allow_physical=False):
        if any(type(v) is not int or v <= 0 for v in (horizon, slot_seconds, sample_seconds)):
            raise ValueError('Positive integer horizon and sampling grid required')
        if slot_seconds % sample_seconds or type(allow_physical) is not bool:
            raise ValueError('Divisible sampling grid and Boolean execution gate required')
        self.graph = graph
        self.model_fingerprint = _sha(model_fingerprint, 'model fingerprint')
        self.horizon, self.slot_seconds, self.sample_seconds = horizon, slot_seconds, sample_seconds
        self.allow_physical = allow_physical
        self._condition = json.loads(_canonical(condition))
        self._event = json.loads(_canonical(event))
        for value, name in ((self._condition, 'condition'), (self._event, 'event')):
            _sha(value.get('source_sha256'), name + ' source')
        if not isinstance(self._condition.get('condition_id'), str) or not self._condition['condition_id']:
            raise ValueError('Named condition required')
        if type(self._event.get('event_id')) not in (int, str) or self._event['event_id'] == '':
            raise ValueError('Named source event required')
        _number(self._event.get('a'), 'CLPU amplitude')
        _number(self._event.get('tau_seconds'), 'CLPU decay', positive=True)
        if type(self._condition.get('der_setpoints')) is not dict or type(self._condition.get('operating_config')) is not dict:
            raise ValueError('Explicit DER and operating configuration bindings required')
        for name, row in self._condition['der_setpoints'].items():
            if type(name) is not str or not name or type(row) is not dict:
                raise ValueError('Named DER setpoint rows required')
            for key in ('kw', 'kvar'):
                if type(row.get(key)) not in (int, float) or not math.isfinite(row[key]):
                    raise ValueError('Finite explicit DER P/Q required')
        self.load_names = tuple(sorted(graph.loads))
        rows = self._condition.get('loads')
        if type(rows) is not dict or set(rows) != set(self.load_names):
            raise ValueError('Exactly the complete graph load inventory is required')
        scale = _number(self._condition.get('load_multiplier'), 'load multiplier', positive=True)
        self._normal = {}
        for name in self.load_names:
            row = rows[name]
            if type(row) is not dict or row.get('bus') != graph.loads[name][0]:
                raise ValueError('Load bus differs from source graph')
            p = _number(row.get('nominal_kw'), name + ':base kW')
            q = _number(row.get('nominal_kvar'), name + ':base kvar')
            if Decimal(str(row['nominal_kw'])) != graph.loads[name][1]:
                raise ValueError('Base demand differs from source graph')
            if not math.isfinite(p * scale) or not math.isfinite(q * scale):
                raise ValueError('Scaled demand overflow')
            self._normal[name] = (p * scale, q * scale)
        self.total_kw = math.fsum(p for p, _ in self._normal.values())
        if not math.isfinite(self.total_kw) or self.total_kw <= 0:
            raise ValueError('Positive complete normal-demand denominator required')
        self.condition_fingerprint = _hash(self._condition)
        self.event_fingerprint = _hash(self._event)
        self.graph_fingerprint = self._graph_hash()
        self.context_sha256 = self._context_hash()
        self.initial_energized = graph.metadata(graph.initial_state).energized_loads
        self.initial = DynamicState(graph.initial_state, 0,
            tuple(-1 if name in self.initial_energized else None for name in self.load_names),
            (), self.context_sha256)

    def _context_hash(self):
        return _hash({'model': self.model_fingerprint,
            'condition': self.condition_fingerprint, 'event': self.event_fingerprint,
            'graph': self.graph_fingerprint, 'horizon': self.horizon, 'slot_seconds': self.slot_seconds,
            'sample_seconds': self.sample_seconds, 'interface': self.interface_version})

    @property
    def condition(self):
        return deepcopy(self._condition)

    @property
    def event(self):
        return deepcopy(self._event)

    def _graph_hash(self):
        g = self.graph
        return _hash({'nodes': sorted(g.nodes), 'edges': g.edges, 'source': g.source,
            'switch_names': g.switch_names, 'switch_edges': g.switch_edges,
            'dispatch': g.dispatch, 'loads': {n: [b, str(p)] for n, (b, p) in g.loads.items()},
            'fault_lines': g.fault_lines, 'original_lines': g.original_lines,
            'initial_state': g.initial_state})

    def _validate_history(self, state):
        if self._graph_hash() != self.graph_fingerprint:
            raise ValueError('Graph configuration changed after binding')
        if self._context_hash() != self.context_sha256:
            raise ValueError('Bound model or time context changed')
        if (not isinstance(state, DynamicState) or type(state.command) is not int
                or type(state.step) is not int or not 0 <= state.step <= self.horizon
                or type(state.pickups) is not tuple or type(state.actions) is not tuple
                or len(state.pickups) != len(self.load_names) or len(state.actions) != state.step
                or state.context_sha256 != self.context_sha256):
            raise ValueError('History shape, context or types differ')
        if any(p is not None and type(p) is not int for p in state.pickups):
            raise ValueError('Pickup times must be exact integer boundaries or None')
        command, pickups = self.initial.command, self.initial.pickups
        for step, action in enumerate(state.actions):
            command = self.graph.transition(command, action)
            after = self.graph.metadata(command).energized_loads
            pickups = tuple(step * self.slot_seconds if p is None and n in after else p
                            for n, p in zip(self.load_names, pickups))
        if command != state.command or pickups != state.pickups:
            raise ValueError('History does not replay from the bound initial state')

    def legal_actions(self, state):
        self._validate_history(state)
        if state.step == self.horizon:
            return ()
        return tuple(r['action'] for r in self.graph.successors(state.command) if r['graph_admissible'])

    def advance(self, state, action):
        self._validate_history(state)
        if state.step >= self.horizon:
            raise ValueError('Exhausted horizon')
        command = self.graph.transition(state.command, action)
        after = self.graph.metadata(command).energized_loads
        pickups = tuple(state.step * self.slot_seconds if p is None and n in after else p
                        for n, p in zip(self.load_names, state.pickups))
        return DynamicState(command, state.step + 1, pickups,
                            state.actions + (action,), self.context_sha256)

    def multipliers(self, state, time_seconds):
        self._validate_history(state)
        t = _number(time_seconds, 'sample time')
        lo = max(0, (state.step - 1) * self.slot_seconds)
        hi = state.step * self.slot_seconds
        if not lo <= t <= hi:
            raise ValueError('Sample time is outside the represented state interval')
        return {n: 1.0 if p is None or p == -1 else multiplier(self._event, t - p)
                for n, p in zip(self.load_names, state.pickups)}

    def snapshot_request(self, state, time_seconds):
        factors = self.multipliers(state, time_seconds)
        meta = self.graph.metadata(state.command)
        loads = {n: {'bus': self._condition['loads'][n]['bus'],
            'normal_kw': self._normal[n][0], 'normal_kvar': self._normal[n][1],
            'clpu_multiplier': factors[n], 'requested_kw': self._normal[n][0] * factors[n],
            'requested_kvar': self._normal[n][1] * factors[n], 'enabled': n in meta.energized_loads}
            for n in self.load_names}
        request = {'interface_version': self.interface_version,
            'model_fingerprint': self.model_fingerprint, 'graph_fingerprint': self.graph_fingerprint,
            'condition_fingerprint': self.condition_fingerprint, 'event_fingerprint': self.event_fingerprint,
            'condition': self.condition, 'event': self.event, 'source_bus': self.graph.source,
            'fault_lines': list(self.graph.fault_lines), 'state': state.to_dict(),
            'effective_command': meta.effective_command,
            'effective_action': list(self.graph.command_bits(meta.effective_command)),
            'time_seconds': float(time_seconds), 'loads': loads,
            'normal_demand_denominator_kw': self.total_kw,
            'energization_scope': 'BUS_GRAPH_CANDIDATE_REQUIRES_PHASE_AND_PHYSICAL_CHECKS'}
        request['request_sha256'] = _hash(request)
        return request

    def service_kw(self, state):
        self._validate_history(state)
        return math.fsum(self._normal[n][0] for n in self.graph.metadata(state.command).energized_loads)

    def plan_states(self, start, actions):
        self._validate_history(start)
        states = [start]
        for action in actions:
            states.append(self.advance(states[-1], action))
        return states

    def validate_plan(self, start, actions, oracle):
        """Always check the starting snapshot, then all samples; never reuse it free.

        Request hashes bind certificates. The caller owns oracle resource limits.
        Unknown/missing evidence and budget refusal stop with no completed energy.
        """
        self._validate_history(start)
        actions = tuple(actions)
        if len(actions) != self.horizon - start.step:
            raise ValueError('A complete remaining-window path is required')
        kind = getattr(oracle, 'evidence_kind', None)
        if kind not in ('SIMULATED', 'PHYSICAL') or (kind == 'PHYSICAL' and not self.allow_physical):
            raise ValueError('Oracle evidence kind missing or physical execution not enabled')
        record = {'status': 'IN_PROGRESS', 'evidence_kind': kind, 'physical_certified': False,
            'energy_kwh': None, 'samples': [], 'actions': list(actions), 'states': [],
            'normal_demand_denominator_kw': self.total_kw}
        try:
            states = self.plan_states(start, actions)
        except ValueError as exc:
            return {**record, 'status': 'GRAPH_FAIL', 'error': str(exc)}
        record['states'] = [s.to_dict() for s in states]
        schedule = [('INITIAL', start, start.step * self.slot_seconds)]
        schedule += [('SLOT', after, (after.step - 1) * self.slot_seconds + offset)
                     for after in states[1:]
                     for offset in range(0, self.slot_seconds + 1, self.sample_seconds)]
        for phase, state, t in schedule:
            request = self.snapshot_request(state, t)
            try:
                cert = oracle.certify(deepcopy(request))
            except CLPUBudgetExhausted as exc:
                return {**record, 'status': 'BUDGET_EXHAUSTED', 'error': str(exc),
                        'stopped_request_sha256': request['request_sha256']}
            sample = {'phase': phase, 'step': state.step, 'time_seconds': t,
                      'request_sha256': request['request_sha256'], 'certificate': deepcopy(cert)}
            record['samples'].append(sample)
            valid = (type(cert) is dict and cert.get('evidence_kind') == kind
                     and cert.get('request_sha256') == request['request_sha256']
                     and cert.get('reliable') is True and cert.get('status') in ('PASS', 'FAIL')
                     and type(cert.get('point_id')) in (str, int) and cert['point_id'] != '')
            if valid and kind == 'PHYSICAL':
                checks = cert.get('checks', {})
                valid = (type(checks) is dict
                         and all(checks.get(k) == 'PASS' for k in RELIABILITY_CHECKS)
                         and all(checks.get(k) in ('PASS', 'FAIL') for k in CONSTRAINT_CHECKS)
                         and (all(checks[k] == 'PASS' for k in CONSTRAINT_CHECKS) == (cert['status'] == 'PASS')))
            if not valid:
                return {**record, 'status': 'UNKNOWN', 'error': 'Missing, inconsistent or unbound evidence'}
            if cert['status'] == 'FAIL':
                return {**record, 'status': 'SIMULATED_FAIL' if kind == 'SIMULATED' else 'FAIL',
                        'failed_phase': phase, 'failed_step': state.step, 'failed_time_seconds': t}
        energy = math.fsum(self.service_kw(s) * self.slot_seconds / 3600 for s in states[1:])
        return {**record, 'status': 'SIMULATED_COMPLETE' if kind == 'SIMULATED' else 'COMPLETE',
            'physical_certified': kind == 'PHYSICAL', 'energy_kwh': energy,
            'energy_scope': 'NORMAL_DEMAND_NOT_CLPU_SURGE',
            'final_fraction': self.service_kw(states[-1]) / self.total_kw,
            'operations': sum(a != 0 for a in actions)}
