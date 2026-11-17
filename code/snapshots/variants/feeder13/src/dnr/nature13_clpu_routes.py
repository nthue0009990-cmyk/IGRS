"""Wire frozen NC13 search components to a supplied evidence service."""
from .nature13_clpu_branch import BranchSearch, BranchConfig
from .nature13_clpu_staged_control import StagedCertificationSearch
from .nature13_clpu_staged_energy_control import EnergyStagedMacroController
from .nature13_clpu_guard_first_control import GuardFirstStagedSearch, GuardFirstMacroController
from .nature13_clpu_lazysp import LazySPCertificationSearch
from .nature34_clpu_early import merge_results, complete_plan


def run_routes(env, service, branch_config, arm, expected_pool, write_phase, emit=lambda *a: None):
    if arm not in ('P', 'E', 'Forward'):
        raise ValueError('Unknown frozen comparison arm')
    order = ['EARLY', 'GUARD'] if arm == 'E' else ['GUARD', 'EARLY']
    if service.phase_order != order:
        raise ValueError('Evidence phase order mismatch')
    phases, hashes = {}, {}
    config = BranchConfig(**branch_config)
    for phase in order:
        callback = lambda row: emit(phase, 'plan', row)
        with service.view(phase) as oracle:
            if phase == 'GUARD':
                search = BranchSearch(env, oracle, config, record_callback=callback)
                while not search.done:
                    action = search.rule_action('beam')
                    response = search.step(action)
                    emit(phase, 'decision', {'action': action, 'response': response})
                result = search.result()
            else:
                reference = {'phase': 'GUARD', 'file': 'GUARD/phase_result.json',
                             'sha256': hashes['GUARD']} if 'GUARD' in phases else None
                if arm == 'P':
                    search = GuardFirstStagedSearch(env, oracle, config, record_callback=callback,
                        guard_result=phases['GUARD'], guard_reference=reference)
                    control = GuardFirstMacroController(search)
                else:
                    search = StagedCertificationSearch(env, oracle, config, record_callback=callback)
                    control = EnergyStagedMacroController(search) if arm == 'E' else None
                if search.stage_generation != expected_pool:
                    raise ValueError('Candidate generation differs from frozen pool')
                if arm == 'Forward':
                    lazy = LazySPCertificationSearch(search, guard_result=phases['GUARD'],
                        guard_reference=reference, selector='forward', max_selected_plans=16)
                    result = lazy.run()
                else:
                    proposal = control.prepare()
                    while proposal is not None:
                        response = control.apply(proposal)
                        emit(phase, 'decision', {'proposal': proposal, 'response': response})
                        proposal = control.prepare()
                    result = control.result()
                if result.get('verify_attempts', 0) > 16:
                    raise ValueError('Frozen VERIFY limit exceeded')
            if not result['done'] or complete_plan(result) is None:
                raise ValueError('Every completed phase must retain a full certified plan')
        result.update(oracle=oracle.summary())
        hashes[phase] = write_phase(phase, result)
        phases[phase] = result
    merged = merge_results(phases['EARLY'], phases['GUARD'], protected=True)
    if not merged['guard_audit']['guard_established']:
        raise ValueError('Full Beam protection absent')
    return phases, merged
