"""New public floor/order boundaries; all certificates here are synthetic."""
from copy import deepcopy

import pytest

from dnr.nature34_clpu_branch import BranchSearch
from dnr.nature34_clpu_early import merge_results
from dnr.nature34_clpu_environment import CLPUReliabilityError
from dnr.nature34_clpu_search import graph_plan
from dnr.nature34_clpu_guard_first_control import (
    GuardFirstStagedSearch, GuardFirstVerificationAdapter, GuardFirstMacroController,
    GuardFirstEvidenceService,
)
from dnr.nature34_clpu_staged_pickup import generate_staged_pickup_candidates
from test_nature34_clpu_branch import environment
from test_nature34_clpu_deferred import ActualCountedOracle
from test_nature34_clpu_shared_evidence import build
from test_run_clpu_deferred_pilot import FakePolicy

REFERENCE = {'phase': 'GUARD', 'file': 'GUARD/phase_result.json', 'sha256': 'a' * 64}


def donor(env, actions=None):
    oracle = ActualCountedOracle(env)
    guard = BranchSearch(env, oracle)
    if actions is not None:
        guard._evaluate(graph_plan(env, env.initial, actions), 'synthetic_certified_donor', {})
    guard._stop('PHYSICAL_BUDGET_EXHAUSTED')
    return guard.result()


def test_floor_prunes_without_queries_or_macro_choices_and_keeps_certificate_namespace():
    env = environment(horizon=30)
    plans = generate_staged_pickup_candidates(env)['candidates']
    record = donor(env, min(plans, key=lambda row: graph_plan(env, env.initial, row['actions'])['priority'])['actions'])
    original = deepcopy(record)
    oracle = ActualCountedOracle(env)
    search = GuardFirstStagedSearch(env, oracle, guard_result=record, guard_reference=REFERENCE)
    assert (oracle.points, oracle.requests) == (61, 91)
    assert search.total_decisions == search.verify_attempts == 0
    expected = {n['id'] for n in search._frontier.values() if n['info']['priority'] >= search.guard_floor_key}
    assert expected and {r['branch_id'] for r in search.free_pruning} == expected
    assert len(search._frontier) == len(search.stage_generation['candidates'])
    assert all(not r['consumed_verification_choice'] for r in search.free_pruning)
    assert all(tuple(r['actions']) not in search._checked for r in search.free_pruning)
    result = search.result()
    assert result['best_plan']['origin'] == 'wait_incumbent'
    assert result['guard_first']['external_guard_plan'] == record['best_plan']
    assert result['guard_first']['guard_reference'] == REFERENCE
    assert record == original and search.guard_floor is not record['best_plan']
    adapter = GuardFirstVerificationAdapter(search)
    assert adapter.incumbent == record['best_plan']
    assert adapter.quality == record['best_plan']['validation']['energy_kwh'] / search.denominator
    assert merge_results(result, record, protected=True)['selected_phase'] == 'GUARD'


def test_exact_full_objective_retains_equal_energy_better_secondary_candidate():
    env = environment(horizon=30)
    search = GuardFirstStagedSearch(env, ActualCountedOracle(env),
        guard_result=donor(env), guard_reference=REFERENCE)
    # Exercise the public predicate with the genuine complete-objective shape.
    node = next(iter(search._frontier.values()))
    key = node['info']['priority']
    assert len(key) > 1
    search.guard_floor_key = (key[0], key[1] + 1, *key[2:])
    search._refresh_stop()
    assert node['info']['actions'] in search.verification_queue
    search.guard_floor_key = key
    search._refresh_stop()
    assert node['info']['actions'] not in search.verification_queue
    assert any(r['branch_id'] == node['id'] for r in search.free_pruning)


def test_fixed_actor_uses_same_masked_call_and_cannot_receive_a_pruned_plan():
    env = environment(horizon=30)
    search = GuardFirstStagedSearch(env, ActualCountedOracle(env),
        guard_result=donor(env), guard_reference=REFERENCE)
    policy = FakePolicy(env)
    control = GuardFirstMacroController(search, method='bc_raw', policy=policy)
    proposal = control.prepare()
    assert proposal and proposal['policy_used'] and policy.calls == 1
    assert proposal['observation']['mask'][proposal['action']]
    assert proposal['branch_id'] not in {r['branch_id'] for r in search.free_pruning}
    control.apply(proposal)
    assert control.result()['control']['active_selector'] == 'fixed_raw_actor'
    assert control.result()['control']['guard_established'] is True
    assert control.result()['complete_guard_still_required'] is False


@pytest.mark.parametrize('mutation', ['unfinished', 'no_certificate', 'wrong_reference'])
def test_invalid_donor_is_rejected_before_any_early_request(mutation):
    env = environment(horizon=30); record = donor(env); reference = deepcopy(REFERENCE)
    if mutation == 'unfinished': record['done'] = False
    if mutation == 'no_certificate': record['best_plan']['validation']['status'] = 'FAIL'
    if mutation == 'wrong_reference': reference['phase'] = 'EARLY'
    oracle = ActualCountedOracle(env)
    with pytest.raises(CLPUReliabilityError):
        GuardFirstStagedSearch(env, oracle, guard_result=record, guard_reference=reference)
    assert oracle.points == oracle.requests == 0


@pytest.mark.parametrize('failed', [False, True])
def test_service_reverses_only_causal_view_order_and_keeps_true_cost(build, failed):
    template = build(failed=failed)
    service = GuardFirstEvidenceService(template.parent, template.cell, template.env, template.scope)
    with service:
        with pytest.raises(CLPUReliabilityError, match='GUARD must close'):
            service.view('EARLY')
        with service.view('GUARD') as guard:
            first = guard.certify(service.env.initial, 0)
            with pytest.raises(CLPUReliabilityError): service.view('EARLY')
        with service.view('EARLY') as early:
            second = early.certify(service.env.initial, 60)
            assert first['status'] == second['status'] == ('FAIL' if failed else 'PASS')
            assert early.cross_view_reuses == 1 and early.physical_points == 0
            assert guard.points == early.points == 1
    assert list(service.views) == ['GUARD', 'EARLY']
    assert service.physical_points == 1 and service.summary()['actual_numerical_entries'] == 7
    assert len(build.delegates) == 1 and service.closed
