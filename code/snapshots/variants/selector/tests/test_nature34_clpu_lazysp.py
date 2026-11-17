"""Synthetic boundary tests only: no solver, no saved experimental outcomes."""
from copy import deepcopy
from collections import defaultdict

import pytest

from dnr.nature34_clpu_branch import BranchConfig
from dnr.nature34_clpu_early import merge_results
from dnr.nature34_clpu_environment import CLPUReliabilityError
from dnr.nature34_clpu_staged_control import StagedCertificationSearch
from dnr.nature34_clpu_lazysp import LazySPCertificationSearch
from test_nature34_clpu_branch import environment
from test_nature34_clpu_deferred import ActualCountedOracle
from test_nature34_clpu_guard_first_control import donor, REFERENCE


def build(selector='forward', budget=256, max_selected_plans=16):
    env = environment(horizon=30)
    guard = donor(env)
    oracle = ActualCountedOracle(env, budget=budget)
    bootstrap = StagedCertificationSearch(env, oracle, BranchConfig(point_budget=budget))
    search = LazySPCertificationSearch(bootstrap, guard_result=guard,
        guard_reference=REFERENCE, selector=selector, max_selected_plans=max_selected_plans)
    return env, oracle, bootstrap, search, guard


@pytest.mark.parametrize('selector', ['forward', 'reverse'])
def test_complete_plan_keeps_all_original_samples_and_candidate_pool(selector):
    env, oracle, bootstrap, search, guard = build(selector)
    paid = (oracle.points, oracle.requests)
    assert paid == (61, 91)
    original = deepcopy(bootstrap.stage_generation)
    result = search.run()
    assert result['best_plan']['validation']['status'] == 'COMPLETE'
    assert len(result['best_plan']['validation']['samples']) == 90
    assert bootstrap.stage_generation == original
    assert {tuple(c['actions']) for c in search.candidates} == {
        tuple(c['actions']) for c in original['candidates']}
    assert all(len(e['samples']) == 3 for e in search.edges if e['status'] == 'PASS')
    canonical = [e for e in search.journal if e['event'] == 'CANONICAL_COMPLETE_CERTIFICATE']
    assert len(canonical) == 1
    assert canonical[0]['before']['actual_physical_points'] == canonical[0]['after']['actual_physical_points']
    assert canonical[0]['after']['requests'] - canonical[0]['before']['requests'] == 90
    assert merge_results(result, guard, protected=True)['selected_phase'] == 'EARLY'


def test_forward_reverse_are_distinct_edge_schedules_not_relabelled_macro_rules():
    first = []
    for selector in ('forward', 'reverse'):
        _, _, _, search, _ = build(selector)
        search.step()
        event = search.journal[-1]
        first.append(search.edges[event['edge_id']]['step'])
    assert first == [1, 30]


def test_reliable_failed_prefix_blocks_exact_descendants_only():
    _, oracle, _, search, _ = build()
    edge = next(e for e in search.edges if len(e['candidate_ids']) > 1 and e['state']['command'] != search.env.initial.command)
    target = search._states[edge['edge_id']]
    oracle.fail = lambda state, seconds: state.command == target.command and state.pickups == target.pickups
    candidate = next(c for c in search.candidates if edge['edge_id'] in c['edge_ids'])
    search._evaluate_edge(candidate, edge['edge_id'])
    assert edge['status'] == 'FAIL' and len(edge['samples']) == 1
    assert {c['branch_id'] for c in search.candidates if c['status'] == 'FAIL'} == set(edge['candidate_ids'])
    assert all(c['failed_edge_id'] == edge['edge_id'] for c in search.candidates if c['status'] == 'FAIL')


def test_same_topology_different_clpu_history_is_not_a_shared_tree_edge():
    _, _, _, search, _ = build()
    groups = defaultdict(list)
    for edge in search.edges:
        groups[(edge['state']['command'], edge['step'])].append(edge)
    differing = [(a, b) for group in groups.values() for a in group for b in group
        if a['state']['pickups'] != b['state']['pickups']]
    assert differing
    assert all(a['edge_id'] != b['edge_id'] for a, b in differing)


def test_budget_partial_edge_stays_unknown_and_complete_guard_is_retained():
    _, oracle, _, search, guard = build('reverse', budget=65)
    result = search.run()
    assert result['stop_reason'] == 'PHYSICAL_BUDGET_EXHAUSTED'
    assert oracle.points == 65
    assert any(e['status'] == 'UNKNOWN' and e['samples'] for e in search.edges)
    assert not search.attempts
    assert merge_results(result, guard, protected=True)['selected_phase'] == 'GUARD'


def test_unknown_is_error_not_fail_or_guard_masked_success():
    _, oracle, _, search, _ = build('reverse')
    oracle.unknown = True
    with pytest.raises(CLPUReliabilityError, match='UNKNOWN'):
        search.run()
    assert search.failed and search.result()['best_plan'] is None
    assert not any(e['status'] == 'FAIL' for e in search.edges)
    assert search.result()['status'] == 'RELIABILITY_OR_IMPLEMENTATION_ERROR'


def test_distinct_plan_cap_is_not_edge_cap():
    _, oracle, _, search, _ = build('reverse', max_selected_plans=1)
    oracle.fail = lambda state, seconds: seconds == 1770 and state.command != search.env.initial.command
    result = search.run()
    assert len(search.selected) == 1
    assert result['stop_reason'] in ('DISTINCT_PLAN_SELECTION_CAP', 'CERTIFIED_FIXED_POOL_OBJECTIVE_BOUND')
    assert result['lazysp']['max_selected_plans'] == 1


def test_completed_guard_floor_can_stop_without_any_new_request():
    env, oracle, bootstrap, _, _ = build()
    best = max(bootstrap._frontier.values(), key=lambda n: n['info']['energy_kwh'])
    guard = donor(env, best['info']['actions'])
    search = LazySPCertificationSearch(bootstrap, guard_result=guard, guard_reference=REFERENCE)
    before = oracle.requests
    result = search.run()
    assert oracle.requests == before and not search.selected and not search.journal
    assert merge_results(result, guard, protected=True)['selected_phase'] == 'GUARD'


def test_guard_and_bootstrap_inputs_are_not_mutated():
    _, _, bootstrap, search, guard = build()
    old_guard, old_pool = deepcopy(guard), deepcopy(bootstrap.stage_generation)
    search.run()
    assert guard == old_guard and bootstrap.stage_generation == old_pool
    with pytest.raises(RuntimeError): search.step()


def test_new_complete_certificate_is_emitted_to_original_plan_log_callback():
    _, _, bootstrap, search, _ = build()
    emitted = []
    bootstrap.record_callback = lambda row: emitted.append(deepcopy(row))
    result = search.run()
    assert emitted == search.attempts
    assert emitted[-1] == result['best_plan']
