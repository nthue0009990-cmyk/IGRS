"""Development ablation: retain the GUARD floor but disable free queue removal.

No physical constraint, candidate generator, paid VERIFY operation, or final
certificate merge changes. Native EARLY own-incumbent skips remain enabled.
"""
from .nature34_clpu_guard_first_control import (
    GuardFirstStagedSearch, GuardFirstVerificationAdapter, GuardFirstMacroController,
)
from .nature34_clpu_staged_control import StagedCertificationSearch

VERSION = 'clpu-public-no-free-pruning-v1'


class NoFreePruneStagedSearch(GuardFirstStagedSearch):
    def _refresh_stop(self):
        # Skip only the intervening GuardFirst free-removal implementation.
        # Calling the staged implementation retains all native stopping rules.
        StagedCertificationSearch._refresh_stop(self)

    def result(self):
        result = super().result()
        floor = result['guard_first']
        floor.update(ablation_version=VERSION, free_objective_pruning_enabled=False)
        floor['rule'].update(prune='disabled_for_A_prune_only',
            ablation='free_certified_objective_queue_removal_disabled',
            pruning_cost='not_applicable_free_removal_disabled',
            exposure='unchanged_staged_pending_window_with_external_GUARD_features_without_free_removal')
        return result


class NoFreePruneVerificationAdapter(GuardFirstVerificationAdapter):
    def metadata(self):
        result = super().metadata()
        result.update(ablation_version=VERSION,
            dominated_pending='no_free_queue_removal_original_EARLY_own_incumbent_verify_skip_retained')
        return result


class NoFreePruneMacroController(GuardFirstMacroController):
    def __init__(self, search, **kwargs):
        if not isinstance(search, NoFreePruneStagedSearch):
            raise TypeError('No-free-pruning ablation search required')
        super().__init__(search, **kwargs)
        self.adapter = NoFreePruneVerificationAdapter(search)

    def result(self):
        result = super().result()
        result['control'].update(ablation_version=VERSION, free_objective_pruning_enabled=False)
        return result
