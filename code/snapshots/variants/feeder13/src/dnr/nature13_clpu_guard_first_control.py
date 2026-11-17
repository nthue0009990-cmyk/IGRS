# NC13 dimension/import adapter of nature34_clpu_guard_first_control.py; original method rules retained.
"""Public complete-Beam floor before staged candidate certification.

The donor certificate stays in the GUARD namespace. EARLY's best_plan and
attempts contain only certificates obtained through its own logical view.
No physical solver, model loader, or external IO is implemented here.
"""
from copy import deepcopy
import math

from .nature34_clpu_early import complete_plan
from .nature13_clpu_environment import CLPUReliabilityError
from .nature13_clpu_search import graph_plan
from .nature13_clpu_staged_control import StagedCertificationSearch, StagedVerificationAdapter
from .nature13_clpu_staged_energy_control import EnergyStagedMacroController

VERSION = 'clpu-public-guard-first-certified-floor-v1'
FLOOR_RULE = {
    'phase_order': ['GUARD', 'EARLY'],
    'guard': 'unchanged_original_BranchSearch_beam_completed_before_EARLY',
    'floor': 'external_GUARD_complete_certificate_not_an_EARLY_logical_certificate',
    'objective': 'original_graph_plan_complete_priority_tuple_exact_comparison',
    'prune': 'pending_complete_priority_greater_than_or_equal_to_effective_certified_priority',
    'pruning_cost': 'zero_VERIFY_choices_zero_oracle_requests',
    'pruning_feasibility': 'unknown_unless_already_certified_not_a_FAIL_label',
    'exposure': 'unchanged_staged_pending_window_after_public_safe_pruning',
    'feedback': 'EARLY_own_validation_history_only_GUARD_answers_available_via_shared_service',
    'final_merge': 'strict_energy_improvement_select_EARLY_exact_energy_tie_select_GUARD',
}


# Physical evidence service is deliberately supplied by the NC13 runner.

class GuardFirstStagedSearch(StagedCertificationSearch):
    def __init__(self, env, oracle, config=None, record_callback=None, *, guard_result, guard_reference):
        if guard_result.get('done') is not True:
            raise CLPUReliabilityError('Complete GUARD search must precede EARLY')
        plan = complete_plan(guard_result)
        if plan is None or guard_result.get('stop_reason') in ('INITIAL_FAIL', 'DEFERRED_RELIABILITY_OR_IMPLEMENTATION_ERROR'):
            raise CLPUReliabilityError('Certified complete GUARD incumbent required')
        info = graph_plan(env, env.initial, plan['actions'])
        if not math.isclose(info['energy_kwh'], plan['validation']['energy_kwh'], rel_tol=1e-12, abs_tol=1e-10):
            raise CLPUReliabilityError('GUARD certificate and current graph objective differ')
        if (not isinstance(guard_reference, dict) or guard_reference.get('phase') != 'GUARD'
                or guard_reference.get('file') != 'GUARD/phase_result.json'
                or not isinstance(guard_reference.get('sha256'), str)
                or len(guard_reference['sha256']) != 64):
            raise CLPUReliabilityError('Explicit completed GUARD evidence reference required')
        self.guard_floor = deepcopy(plan)
        self.guard_floor_key = info['priority']
        self.guard_reference = deepcopy(guard_reference)
        self.free_pruning = []
        super().__init__(env, oracle, config, record_callback)

    @property
    def effective_incumbent(self):
        if self.incumbent is not None and self._incumbent_key < self.guard_floor_key:
            return self.incumbent
        return self.guard_floor

    @property
    def effective_priority(self):
        if self.incumbent is not None and self._incumbent_key < self.guard_floor_key:
            return self._incumbent_key
        return self.guard_floor_key

    def _refresh_stop(self):
        if not self.done:
            floor = self.effective_priority
            source = 'EARLY' if self.effective_incumbent is self.incumbent else 'GUARD'
            for plan, branch_id in list(self.verification_queue.items()):
                node = self._frontier[branch_id]
                if node['info']['priority'] >= floor:
                    self.verification_queue.pop(plan)
                    self.skipped_dominated_plans.add(plan)
                    self.free_pruning.append({'branch_id': branch_id, 'actions': list(plan),
                        'status': 'PRUNED_CERTIFIED_COMPLETE_OBJECTIVE', 'physical_feasibility': 'UNASSESSED',
                        'candidate_priority': deepcopy(node['info']['priority']),
                        'certified_priority': deepcopy(floor), 'incumbent_source_phase': source,
                        'completed_verify_attempts': self.verify_attempts,
                        'logical_points': self._points(), 'actual_physical_points': self._actual(self.oracle),
                        'consumed_verification_choice': False})
        super()._refresh_stop()

    def result(self):
        result = super().result()
        result['guard_first'] = {'version': VERSION, 'rule': deepcopy(FLOOR_RULE),
            'guard_reference': deepcopy(self.guard_reference),
            'external_guard_plan': deepcopy(self.guard_floor),
            'external_guard_priority': deepcopy(self.guard_floor_key),
            'effective_incumbent_source_phase': 'EARLY' if self.effective_incumbent is self.incumbent else 'GUARD',
            'effective_incumbent_quality_pct': 100. * self.effective_incumbent['validation']['energy_kwh'] / self.denominator,
            'own_best_plan_namespace': 'EARLY', 'free_pruning': deepcopy(self.free_pruning),
            'free_pruned_complete_plans': len(self.free_pruning)}
        return result


class GuardFirstVerificationAdapter(StagedVerificationAdapter):
    @property
    def incumbent(self):
        return self.engine.effective_incumbent

    @property
    def _incumbent_key(self):
        return self.engine.effective_priority

    def metadata(self):
        result = super().metadata()
        result.update(guard_first_version=VERSION,
            incumbent_features='best_of_external_completed_GUARD_and_EARLY_own_certificate_by_original_priority',
            dominated_pending='free_public_objective_pruning_before_window_without_VERIFY_or_physical_query')
        return result


class GuardFirstMacroController(EnergyStagedMacroController):
    def __init__(self, search, **kwargs):
        if not isinstance(search, GuardFirstStagedSearch):
            raise TypeError('Guard-first staged search required')
        super().__init__(search, **kwargs)
        self.adapter = GuardFirstVerificationAdapter(search)

    def result(self):
        result = super().result()
        result['control'].update(guard_first_version=VERSION,
            guard_established=True, complete_guard_still_required=False,
            handoff_semantics='EARLY_termination_after_already_completed_GUARD_no_second_guard_run')
        result.update(guard_established=True, complete_guard_still_required=False)
        return result

