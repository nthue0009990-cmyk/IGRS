"""Original complete Beam guard followed by a common coverage-window tail.

Only candidate exposure during CONTINUATION changes. The frozen tail owns the
baseline, reliable failed-prefix evidence, repair operations, floor, counters,
and deterministic VERIFY stage. No oracle or search constructor is rerun at
the stage boundary; no solver or learner is imported by this module.
"""
from __future__ import annotations

from copy import deepcopy

from .nature34_clpu_environment import CLPUReliabilityError
from .nature34_clpu_tail import TailBranchSearch, _ContinuationSearch
from .nature34_clpu_window import WINDOW_RULE, WindowBranchSearch


VERSION = 'protected-beam-coverage-tail-v1'
RESULT_SCHEMA_VERSION = 'clpu-integrated-search-result-v1'


class _CoverageContinuationSearch(_ContinuationSearch, WindowBranchSearch):
    """Exact FAIL filtering, then coverage exposure, then unchanged selection.

The inherited tail window calls ``super`` after its temporary FAIL filter.
The MRO therefore reaches WindowBranchSearch before the original BranchSearch.
VERIFY returns the original one-element tail window before either exposure
rule. The all-known-failing fallback keeps repair parents in the frontier.
"""


class CoverageTailBranchSearch(TailBranchSearch):
    def __init__(self, env, oracle, config=None, *, physics_identity,
                 continuation=True, extra_branch_expansions=256,
                 record_callback=None, event_callback=None):
        self._coverage_adopted = False
        self._coverage_adoption_turn = None
        super().__init__(env, oracle, config, physics_identity=physics_identity,
                         continuation=continuation,
                         extra_branch_expansions=extra_branch_expansions,
                         record_callback=record_callback, event_callback=event_callback)

    def _finish_baseline(self):
        # The original implementation seals the complete, unchanged Beam
        # result and establishes the floor before constructing its tail state.
        super()._finish_baseline()
        if type(self.search) is _ContinuationSearch:
            previous = self.search
            adopted = _CoverageContinuationSearch.__new__(_CoverageContinuationSearch)
            adopted.__dict__ = previous.__dict__.copy()
            self.search = adopted
            self._coverage_adopted = True
            self._coverage_adoption_turn = adopted._beam_turn

    def integration_metadata(self):
        config = self.original_config
        raw_width = 27 + self.search.horizon + 3 * len(self.env.load_names)
        return {
            'version': VERSION, 'schema_version': RESULT_SCHEMA_VERSION,
            'baseline_engine': 'original_BranchSearch', 'baseline_selector': 'beam',
            'continuation_window_rule': deepcopy(WINDOW_RULE),
            'continuation_order': ['exact_failed_prefix_filter', 'coverage_window', 'configured_selector'],
            'verification_rule': 'unchanged_tail_priority',
            'turn_counter': 'cumulative_beam_turn_no_reset',
            'oracle_scope': 'same_private_oracle_baseline_through_verify',
            'logical_point_budget': config.point_budget,
            'baseline_expansion_cap': config.max_branch_expansions,
            'additional_expansion_cap': self.extra_branch_expansions,
            'total_expansion_cap': config.max_branch_expansions + self.extra_branch_expansions,
            'cumulative_checked_cap': config.max_checked_candidates,
            'cumulative_generated_cap': config.max_generated_candidates,
            'continuation_adopted': self._coverage_adopted,
            'adoption_completed_expansions': self._coverage_adoption_turn,
            'feature_schema': {
                'source': 'BranchSearch.observation_unchanged',
                'raw_candidate_dim': raw_width, 'associated_candidate_dim': raw_width + 8,
                'global_dim': 11 + len(self.env.load_names),
                'action_window_capacity': config.action_window,
                'load_names_in_feature_order': list(self.env.load_names),
            },
            'nested_counter_semantics': 'Legacy tail, baseline, plan, event, reward and observation point fields are cumulative logical counts; point_id is logical.',
            'physical_counter_semantics': 'Top-level actual_physical_points and the oracle physical ledger count real solves; no counter resets at phase transitions.',
        }

    def result(self):
        result = super().result()
        logical = self.search._points()
        physical = getattr(self.oracle, 'physical_points', None)
        if type(physical) is not int or not 0 <= physical <= logical:
            raise CLPUReliabilityError('Integrated output requires an explicit valid real physical-point counter')
        result['logical_points'] = logical
        result['actual_physical_points'] = physical
        result['plan_point_id_semantics'] = 'logical_point_id'
        result['integration'] = self.integration_metadata()
        return result
