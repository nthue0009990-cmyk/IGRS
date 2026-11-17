"""A_prune boundaries using synthetic certificates, never physical samples."""
from copy import deepcopy

import pytest

from dnr.nature34_clpu_guard_first_control import GuardFirstStagedSearch
from dnr.nature34_clpu_public_ablation_control import (
    NoFreePruneStagedSearch, NoFreePruneMacroController, VERSION,
)
from dnr.nature34_clpu_search import graph_plan
from dnr.nature34_clpu_staged_pickup import generate_staged_pickup_candidates
from test_nature34_clpu_branch import environment
from test_nature34_clpu_deferred import ActualCountedOracle
from test_nature34_clpu_guard_first_control import donor, REFERENCE


def make_search():
    env = environment(horizon=30)
    plans = generate_staged_pickup_candidates(env)['candidates']
    best = min(plans, key=lambda p: graph_plan(env, env.initial, p['actions'])['priority'])
    guard = donor(env, best['actions'])
    oracle = ActualCountedOracle(env)
    search = NoFreePruneStagedSearch(env, oracle, guard_result=guard, guard_reference=REFERENCE)
    return env, guard, oracle, search


def test_same_pool_and_guard_information_but_no_free_removal_or_extra_query():
    env, guard, oracle, search = make_search()
    p_oracle = ActualCountedOracle(env)
    public = GuardFirstStagedSearch(env, p_oracle, guard_result=guard, guard_reference=REFERENCE)
    assert search.stage_generation == public.stage_generation
    assert search.effective_incumbent == public.effective_incumbent == guard['best_plan']
    assert search.effective_priority == public.effective_priority
    assert public.free_pruning and not search.free_pruning
    assert len(search.verification_queue) > len(public.verification_queue)
    assert (oracle.points, oracle.requests) == (p_oracle.points, p_oracle.requests)
    assert search.verify_attempts == search.total_decisions == 0
    assert search.result()['best_plan']['origin'] == 'wait_incumbent'


def test_native_own_incumbent_skip_is_still_paid_as_one_choice_without_a_query():
    _, _, oracle, search = make_search()
    node = next(iter(search._frontier.values()))
    # A synthetic completed own certificate tests the native VERIFY boundary.
    search.incumbent = deepcopy(search.guard_floor)
    search._incumbent_key = node['info']['priority']
    before = (oracle.points, oracle.requests)
    result = search.verify(node['id'])
    assert result['verification_status'] == 'SKIPPED_DOMINATED'
    assert search.verify_attempts == search.total_decisions == 1
    assert (oracle.points, oracle.requests) == before
    assert not search.free_pruning


def test_native_logical_budget_stop_and_ablation_metadata_are_retained():
    _, _, oracle, search = make_search()
    control = NoFreePruneMacroController(search, method='beam')
    result = control.result()
    assert result['guard_first']['free_objective_pruning_enabled'] is False
    assert result['guard_first']['free_pruned_complete_plans'] == 0
    assert result['guard_first']['rule']['prune'] == 'disabled_for_A_prune_only'
    assert result['control']['ablation_version'] == VERSION
    assert result['control']['adapter']['ablation_version'] == VERSION
    assert result['guard_established'] is True
    assert result['complete_guard_still_required'] is False
    oracle.points = search.config.point_budget
    search._refresh_stop()
    assert search.done and search.stop_reason == 'PHYSICAL_BUDGET_EXHAUSTED'


def test_ablation_controller_rejects_non_ablation_search():
    env, guard, _, _ = make_search()
    public = GuardFirstStagedSearch(env, ActualCountedOracle(env),
        guard_result=guard, guard_reference=REFERENCE)
    with pytest.raises(TypeError, match='ablation'):
        NoFreePruneMacroController(public, method='beam')
