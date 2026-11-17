"""Immutable CLPU history and full-window quasi-static path validation.

No physical runtime or learning algorithm is initialized here. Slot-boundary
operations are idealized instantaneous commands; samples do not certify
continuous-time or transient safety. Unserved loads do not decay in advance.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import math

from .clpu_curves import multiplier
from .nature34_sequence_catalog import state_metadata


class CLPUBudgetExhausted(RuntimeError):
    pass


class CLPUReliabilityError(RuntimeError):
    pass


@dataclass(frozen=True)
class DynamicState:
    command: int
    step: int
    pickups: tuple

    def to_dict(self):
        return {'command': self.command, 'step': self.step, 'pickups': list(self.pickups)}

    @classmethod
    def from_dict(cls, value):
        return cls(int(value['command']), int(value['step']), tuple(value['pickups']))


class CLPUEnvironment:
    model_version = 'nc34-clpu-ed-snapshot-v1'

    def __init__(self, graph, static, loads, scenario, condition, event,
                 horizon=30, slot_seconds=60, sample_seconds=30):
        if (type(horizon) is not int or horizon <= 0 or type(slot_seconds) is not int
                or type(sample_seconds) is not int or sample_seconds <= 0
                or slot_seconds <= 0 or slot_seconds % sample_seconds):
            raise ValueError('Positive integer horizon and divisible sampling grid required')
        if graph.scenario_id != scenario['scenario_id']:
            raise ValueError('Graph/scenario identity differs')
        self.graph, self.static, self.loads = graph, deepcopy(static), deepcopy(loads)
        self.scenario, self.condition, self.event = deepcopy(scenario), deepcopy(condition), deepcopy(event)
        self.horizon, self.slot_seconds, self.sample_seconds = horizon, slot_seconds, sample_seconds
        self.load_names = tuple(sorted(loads))
        self._metadata_cache = {}
        self.load_multiplier = float(condition['load_multiplier'])
        self.total_kw = sum(float(v['nominal_kw']) for v in loads.values()) * self.load_multiplier
        if not math.isfinite(self.total_kw) or self.total_kw <= 0:
            raise ValueError('Positive normal load inventory required')
        if 'denominator_nominal_kw' in condition and not math.isclose(
                self.total_kw, condition['denominator_nominal_kw'], rel_tol=1e-10, abs_tol=1e-9):
            raise ValueError('Normal-demand denominator differs')
        multiplier(event, 0.)  # Validate curve, including near-zero amplitude/large tau.
        initial_command = int(graph.initial_state)
        powered = self.energized(initial_command)
        self.initial_energized = powered
        self.initial = DynamicState(initial_command, 0, tuple(-1 if n in powered else None for n in self.load_names))

    def metadata(self, command):
        if command not in self._metadata_cache:
            mask, ready = command >> 10, command & 1023
            bits = [(mask >> i) & 1 for i in range(9)] + [(ready >> i) & 1 for i in range(10)]
            row = state_metadata(self.static, self.loads, self.scenario, bits)
            if row['forest'] is not True:
                raise ValueError('Nonforest command')
            self._metadata_cache[command] = row
        return self._metadata_cache[command]

    def energized(self, command):
        return frozenset(self.metadata(command)['energized_load_ids'])

    def legal_actions(self, state):
        if state.step >= self.horizon:
            return []
        successors = set(self.graph.successors(state.command))
        return [0] + [a for a in range(1, 20) if state.command ^ (1 << (a - 1)) in successors]

    def _validate_history(self, state):
        if (len(state.pickups) != len(self.load_names) or type(state.step) is not int
                or not 0 <= state.step <= self.horizon):
            raise ValueError('Invalid dynamic history dimensions/time')
        powered = self.energized(state.command)
        for name, pickup in zip(self.load_names, state.pickups):
            if (pickup is not None) != (name in powered):
                raise ValueError('Actual electrical connection and pickup history differ')
            if (pickup == -1) != (name in self.initial_energized):
                raise ValueError('Initially steady load history cannot be reassigned')
            if pickup is not None and (isinstance(pickup, bool) or not isinstance(pickup, (int, float))
                    or not math.isfinite(pickup) or (pickup != -1 and pickup < 0)
                    or pickup > state.step * self.slot_seconds):
                raise ValueError('Invalid pickup timestamp')
            if pickup is not None and pickup != -1 and (
                    state.step == 0 or pickup > (state.step - 1) * self.slot_seconds
                    or pickup % self.slot_seconds != 0):
                raise ValueError('First pickup must be at a past operation boundary')

    def advance(self, state, action):
        self._validate_history(state)
        if type(action) is not int or action not in self.legal_actions(state):
            raise ValueError('Illegal action or exhausted horizon')
        command = state.command if action == 0 else state.command ^ (1 << (action - 1))
        before, after = self.energized(state.command), self.energized(command)
        if not before <= after:
            raise ValueError('Reinterruption of any supplied load is forbidden')
        pickups = tuple(state.step * self.slot_seconds if p is None and n in after else p
                        for n, p in zip(self.load_names, state.pickups))
        result = DynamicState(command, state.step + 1, pickups)
        self._validate_history(result)
        return result

    def multipliers(self, state, time_seconds):
        self._validate_history(state)
        if not math.isfinite(time_seconds) or time_seconds < 0:
            raise ValueError('Invalid physical sample time')
        result = {}
        for name, pickup in zip(self.load_names, state.pickups):
            result[name] = 1. if pickup is None or pickup == -1 else multiplier(self.event, time_seconds - pickup)
        return result

    def service_kw(self, state):
        return self.graph.served_base_kw(state.command) * self.load_multiplier

    def identity(self, state, time_seconds):
        # Step is a planning index, not a physical input. At a shared t-/t+
        # boundary reuse requires identical command AND full pickup history.
        # New pickups necessarily alter this identity even at the same time.
        return {'model_version': self.model_version, 'scenario_id': self.scenario['scenario_id'],
                'condition_id': self.condition['condition_id'],
                'event_id': self.event['event_id'], 'a': self.event['a'],
                'tau_seconds': self.event['tau_seconds'], 'command': state.command,
                'pickups': list(state.pickups), 'load_names': list(self.load_names),
                'time_seconds': float(time_seconds), 'multipliers': self.multipliers(state, time_seconds)}

    def optimistic_plan(self, state, depth=6):
        remaining = self.horizon - state.step
        if remaining <= 0:
            return []
        result = self.graph.solve(state.command, min(int(depth), remaining, 12))
        if result['status'] != 'COMPLETE':
            raise ValueError('Optimistic graph route absent')
        actions = []
        for before, after in zip(result['path'], result['path'][1:]):
            diff = before ^ after
            if diff and diff & (diff - 1):
                raise ValueError('Graph route changes multiple controls')
            actions.append(diff.bit_length())
        return actions + [0] * (remaining - len(actions))

    def plan_states(self, start, actions):
        states = [start]
        for action in actions:
            states.append(self.advance(states[-1], action))
        return states

    def validate_plan(self, start, actions, oracle):
        actions = list(actions)
        if len(actions) != self.horizon - start.step:
            raise ValueError('Only complete remaining-window paths can be certified')
        try:
            states = self.plan_states(start, actions)
        except ValueError as exc:
            return {'status': 'GRAPH_FAIL', 'actions': actions, 'states': [], 'samples': [],
                    'feedback': [], 'failed_step': None, 'error': str(exc)}
        record = {'status': 'IN_PROGRESS', 'actions': actions,
                  'states': [s.to_dict() for s in states], 'samples': [],
                  'failed_step': None, 'feedback': [], 'energy_kwh': None,
                  'final_fraction': None}
        for index, after in enumerate(states[1:]):
            t0 = (after.step - 1) * self.slot_seconds
            for offset in range(0, self.slot_seconds + 1, self.sample_seconds):
                t = t0 + offset
                cert = oracle.certify(after, t)
                if cert.get('reliable') is not True or cert.get('status') not in ('PASS', 'FAIL'):
                    raise CLPUReliabilityError('Unknown/unreliable snapshot cannot certify or repair a path')
                sample = {'step': after.step, 'time_seconds': t, 'point_id': cert['point_id'],
                          'status': cert['status'], 'state': after.to_dict()}
                record['samples'].append(sample)
                if cert['status'] == 'FAIL':
                    record.update(status='FAIL', failed_step=index,
                                  failed_time_seconds=t, failed_state=after.to_dict(),
                                  feedback=deepcopy(cert.get('feedback', [])))
                    return record
        energy = math.fsum(self.service_kw(s) * self.slot_seconds / 3600 for s in states[1:])
        record.update(status='COMPLETE', energy_kwh=energy,
                      final_fraction=self.service_kw(states[-1]) / self.total_kw,
                      mean_served_fraction=energy / (self.total_kw * len(actions) * self.slot_seconds / 3600)
                      if actions else self.service_kw(start) / self.total_kw,
                      operations=sum(a != 0 for a in actions))
        return record
