"""Complete-graph-energy ranking within the unchanged staged VERIFY window.

This changes only the public rule selector. The fixed raw actors, candidate
generation, exposure, feature formulas, certifier, budgets and guard are not
implemented or replaced here. Graph energy is a ranking score, not a physical
certificate or a label from a later evaluation.
"""
from copy import deepcopy
import math

from .nature34_clpu_environment import CLPUReliabilityError
from .nature34_clpu_staged_control import StagedMacroController

VERSION = 'clpu-staged-visible-complete-energy-selector-v1'
RULE = {
    'version': VERSION,
    'eligible': 'current_common_visible_window_and_true_action_mask_only',
    'primary': 'maximum_complete_graph_energy_kwh',
    'tie': 'exact_equal_energy_then_smallest_branch_id',
    'prefix_depth_priority': False,
    'future_physical_outcomes': False,
    'fixed_actor_path': 'inherited_StagedMacroController_choose_verify_unchanged',
    'guard': 'unchanged_original_BranchSearch_beam',
}


def complete_energy_action(observation):
    """Return a slot, using only exact public scores in the current window."""
    branches, mask = observation.get('branches'), observation.get('mask')
    if (not isinstance(branches, list) or not isinstance(mask, list)
            or not branches or len(branches) != len(mask) or len(branches) > 32
            or any(type(value) is not bool for value in mask)):
        raise CLPUReliabilityError('Malformed common verification window or mask')
    ids = [row.get('branch_id') for row in branches]
    if (any(type(value) is not int or value < 0 for value in ids)
            or len(set(ids)) != len(ids) or observation.get('branch_ids') != ids):
        raise CLPUReliabilityError('Common verification branch identities differ')
    candidates = []
    for slot, (node, available) in enumerate(zip(branches, mask)):
        energy = node.get('graph_energy_kwh')
        if type(energy) not in (int, float) or not math.isfinite(energy) or energy < 0:
            raise CLPUReliabilityError('Nonfinite or negative complete graph objective')
        if available:
            candidates.append(slot)
    if not candidates:
        raise CLPUReliabilityError('No available verification slot')
    return min(candidates, key=lambda slot: (-branches[slot]['graph_energy_kwh'], branches[slot]['branch_id']))


class EnergyStagedMacroController(StagedMacroController):
    """Preserve the old IO/proposal schema and replace only the Beam scorer."""
    def _choose_verify(self, observation):
        if self.method != 'beam':
            return super()._choose_verify(observation)
        return {'action': complete_energy_action(observation), 'logprob': None,
                'value': None, 'method': 'beam', 'policy_used': False}

    def result(self):
        result = super().result()
        result['control'].update(selector_version=VERSION, beam_rule=deepcopy(RULE),
            active_selector='complete_graph_energy' if self.method == 'beam' else 'fixed_raw_actor')
        return result
