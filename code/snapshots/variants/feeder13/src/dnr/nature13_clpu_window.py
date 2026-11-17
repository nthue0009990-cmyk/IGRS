# NC13 dimension/import adapter of nature34_clpu_window.py; original method rules retained.
"""Versioned candidate-window change for an otherwise unchanged branch engine.

No frontier node is pruned.  On alternate expansion turns an eligible batch
contains only untested, graph-quality-improving prefix + WAIT plans of one
depth.  This is necessary because reserving a few slots beside shallower
completion nodes does not make those holds selectable by the original Beam
rule.  The other turns keep the exact original objective/hold/FIFO window,
including checked plans that remain useful expansion or repair parents.

This module does not certify a graph objective, infer electrical feasibility,
or import a solver.  Its only evidence is the current frontier, checked-plan
keys, already certified incumbent, and completed expansion count.
"""
from __future__ import annotations

from .nature13_clpu_branch import BranchSearch


WINDOW_RULE_VERSION = 'alternating_unique_hold_coverage_v1'
WINDOW_RULE = {
    'version': WINDOW_RULE_VERSION,
    'coverage_turn': 'completed_expansions_mod_2_equals_1',
    'eligible': 'prefix_wait_and_unchecked_and_graph_energy_strictly_above_incumbent',
    'coverage_depth': 'prefix_depth_of_oldest_eligible_branch',
    'coverage_order': 'oldest_unique_full_plans_at_coverage_depth',
    'ordinary_window': 'original_BranchSearch_window_unchanged',
    'no_eligible_fallback': 'original_BranchSearch_window_unchanged',
    'scope': 'alternate_turn_candidate_exposure_only_Beam_selection_unchanged',
    'unexposed_frontier': 'preserved_without_mutation',
    'selection_and_physics': 'inherited_unchanged_from_BranchSearch',
}


class WindowBranchSearch(BranchSearch):
    """Change exposure only; retain the original Beam and all paid mechanics."""

    def _window(self):
        rows = sorted(self._frontier.values(), key=lambda row: row['id'])
        size = self.config.action_window
        if self._beam_turn % 2 == 1 and self.incumbent is not None:
            incumbent_energy = float(self.incumbent['validation']['energy_kwh'])
            eligible = [row for row in rows
                        if row['origin'] == 'prefix_wait'
                        and row['info']['actions'] not in self._checked
                        and row['info']['energy_kwh'] > incumbent_energy]
            if eligible:
                depth = len(eligible[0]['prefix'])
                selected, plans = [], set()
                for row in eligible:
                    actions = row['info']['actions']
                    if len(row['prefix']) == depth and actions not in plans:
                        selected.append(row)
                        plans.add(actions)
                        if len(selected) == size:
                            break
                return selected

        return super()._window()

