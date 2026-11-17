"""NC13 control-layout adapter; inherit the original complete CLPU history rules.

This module imports only pure graph/history code. It never imports a physical
runtime or changes the original NC34 modules or their stored results.
"""
from .nature34_clpu_environment import (
    CLPUEnvironment as _BaseEnvironment, DynamicState,
    CLPUBudgetExhausted, CLPUReliabilityError,
)
from .nature13_clpu_graph import state_metadata, N_SWITCH, N_DISPATCH, N_COMMAND, READY_MASK


class CLPUEnvironment(_BaseEnvironment):
    model_version = 'nc13-clpu-ed-snapshot-v1-dimension-adapter'

    def __init__(self, graph, static, loads, scenario, condition, event,
                 horizon=30, slot_seconds=60, sample_seconds=30):
        if len(loads) != 15 or sum(float(v['nominal_kw']) for v in loads.values()) != 3466:
            raise ValueError('Complete author NC13 fifteen-load 3466 kW inventory required')
        if len(static['switch_edges']) != N_SWITCH or len(static['dispatch_loads']) != N_DISPATCH:
            raise ValueError('Original four switches and seven dispatch loads required')
        if float(condition['load_multiplier']) != 1.0:
            raise ValueError('NC13 adaptation fixes the original nominal demand multiplier')
        super().__init__(graph, static, loads, scenario, condition, event,
                         horizon=horizon, slot_seconds=slot_seconds, sample_seconds=sample_seconds)

    def metadata(self, command):
        if command not in self._metadata_cache:
            self.graph.effective_key(command)  # Enforce the complete initial-reachable command domain.
            mask, ready = command >> N_DISPATCH, command & READY_MASK
            command_bits = [(mask >> i) & 1 for i in range(N_SWITCH)]
            command_bits += [(ready >> i) & 1 for i in range(N_DISPATCH)]
            row = state_metadata(self.static, self.loads, self.scenario, command_bits)
            if row['forest'] is not True:
                raise ValueError('Nonforest command')
            self._metadata_cache[command] = row
        return self._metadata_cache[command]

    def legal_actions(self, state):
        if state.step >= self.horizon:
            return []
        successors = set(self.graph.successors(state.command))
        return [0] + [a for a in range(1, N_COMMAND + 1)
                      if state.command ^ (1 << (a - 1)) in successors]

    def snapshot_request(self, state, seconds, *, algorithm=0):
        """Full actual-input request only; no PASS/FAIL label or runtime call."""
        if type(algorithm) is not int or algorithm not in (0, 1):
            raise ValueError('Only the two frozen original solver algorithms are allowed')
        identity = {**self.identity(state, seconds), 'alpha': 0.375}
        return {'action': list(self.metadata(state.command)['command']),
                'effective_action': self.graph.action(state.command),
                'case': {'name': self.scenario['scenario_id'], 'fault_lines': list(self.scenario['fault_lines'])},
                'alpha': 0.375, 'algorithm': algorithm,
                'multipliers': self.multipliers(state, seconds),
                'time_minutes': seconds / 60.0, 'state_identity': identity,
                'physical_feasibility': 'UNASSESSED'}
