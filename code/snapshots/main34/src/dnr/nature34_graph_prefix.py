"""Unfitted all-on graph ordering control; no simulator or electrical labels."""
from __future__ import annotations

import numpy as np

from .nature34_baselines import GraphRepairSearch


def propose_graph_all_on_prefix(static, loads, table, scenario, *, count=8):
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError('A positive integer count is required')
    metadata = GraphRepairSearch(static, loads, table, scenario, budget=count).metadata
    roots = metadata['ranked_roots']
    n_switch = len(static['switch_edges'])
    off = np.asarray(table['load_forced_off_table'])
    chosen = roots[:count]
    actions = [[(row['action_index'] >> bit) & 1 for bit in range(n_switch)] +
               (~off[row['action_index']]).astype(int).tolist() for row in chosen]
    return {'actions': actions, 'count': len(actions), 'requested_count': count,
            'exhausted': len(actions) < count, 'unique_action_count': len({tuple(a) for a in actions}),
            'scenario_id': scenario['scenario_id'], 'ranked_roots': roots,
            'selected_roots': chosen, 'root_order': metadata['root_order'],
            'graph_upper_kw': roots[0]['reachable_nominal_kw'],
            'generation_rule': 'First distinct graph-ranked topologies; all reachable dispatch loads on',
            'physical_labels_used': False, 'feedback_used': False, 'physical_evaluations': 0}
