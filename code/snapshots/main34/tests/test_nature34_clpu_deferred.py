"""No-solver tests for graph proposals, delayed certificates and handoff."""
from copy import deepcopy

import pytest

from dnr import nature34_clpu_deferred as deferred
from dnr.nature34_clpu_branch import BranchConfig
from dnr.nature34_clpu_environment import CLPUReliabilityError
from test_nature34_clpu_branch import CountedOracle, environment


class ActualCountedOracle(CountedOracle):
    @property
    def physical_points(self):
        # This synthetic oracle actually creates one row per new semantic key.
        return self.points


def make(*, budget=256, **config):
    env = environment(horizon=6)
    oracle = ActualCountedOracle(env, budget=budget)
    search = deferred.DeferredCertificationSearch(env, oracle, BranchConfig(point_budget=budget, **config))
    return search, oracle


def completion(search):
    return next(r for r in search.pending() if r['origin'] == 'prefix_graph_completion'
                and r['prefix'] == [16] and r['actions'][2] != 0)


def test_bootstrap_is_paid_and_pure_expansion_pre_generates_hold_before_parent_failure():
    search, oracle = make()
    assert search.bootstrap_cost['actual_physical_points_after'] == 13
    parent = completion(search)
    hold = tuple(parent['actions'][:2] + [0] * 4)
    before = deepcopy(search.result()['best_plan'])
    calls, requests = len(oracle.calls), oracle.requests
    event = search.expand(parent['branch_id'])
    assert event['new_actual_physical_points'] == 0
    assert len(oracle.calls) == calls and oracle.requests == requests
    assert search.result()['best_plan'] == before
    assert hold in search.verification_queue and tuple(parent['actions']) in search.verification_queue
    assert tuple(parent['actions']) not in search._checked
    assert any(r['actions'] == list(hold) and not r['verified'] for r in search.pending())
    assert search.graph_expansions == search.total_decisions == 1 and search.verify_attempts == 0


def test_reliable_failure_repairs_only_after_verify_and_preserves_generated_holds(monkeypatch):
    search, oracle = make()
    parent = completion(search)
    calls = []
    original = deferred.generic_neighbors
    def tracked(*args, **kwargs):
        calls.append(kwargs['failed_step'])
        return original(*args, **kwargs)
    monkeypatch.setattr(deferred, 'generic_neighbors', tracked)
    search.expand(parent['branch_id'])
    assert calls == [None] and not search.failure_repairs
    pending_before = set(search.verification_queue)
    state = search.env.plan_states(search.start, parent['actions'])[3]
    oracle.fail = lambda observed, seconds: observed.step == 3 and observed.command == state.command
    event = search.verify(parent['branch_id'])
    assert event['verification_status'] == 'FAIL' and calls == [None, 2]
    assert tuple(parent['actions']) in search.failure_repairs
    assert pending_before - {tuple(parent['actions'])} <= set(search.verification_queue)
    assert search._checked[tuple(parent['actions'])]['validation']['failed_step'] == 2
    assert search.quality == event['quality_before']
    assert search.total_decisions == 2 and search.graph_expansions == search.verify_attempts == 1


def test_verification_deduplicates_full_plan_across_different_prefix_nodes():
    search, oracle = make()
    parent = completion(search)
    plan = tuple(parent['actions'])
    search._add(plan[:2], plan, 'probe_same_complete_plan', parent['branch_id'])
    second = next(r for r in search.pending() if r['origin'] == 'probe_same_complete_plan')
    event = search.verify(parent['branch_id'])
    assert event['verification_status'] == 'COMPLETE'
    paid = oracle.physical_points
    with pytest.raises(ValueError, match='already checked'):
        search.verify(second['branch_id'])
    assert oracle.physical_points == paid
    assert all(r['verified'] and not r['verification_pending'] for r in search.pending() if r['actions'] == list(plan))
    assert search.result()['best_plan']['certified_complete']


def test_checked_low_quality_plan_remains_available_as_graph_parent():
    search, oracle = make()
    wait = next(r for r in search.pending() if r['actions'] == [0] * search.horizon)
    assert wait['verified'] and wait['can_expand'] and not wait['can_verify']
    before = oracle.physical_points
    event = search.expand(wait['branch_id'])
    assert event['new_branches'] > 0 and oracle.physical_points == before


def test_dominated_complete_objective_skips_payment_but_preserves_graph_parent():
    search, oracle = make()
    search.verify(completion(search)['branch_id'])
    node = next(n for n in search._frontier.values()
                if n['info']['actions'] in search.verification_queue
                and n['info']['priority'] >= search._incumbent_key)
    plan = node['info']['actions']
    before = (oracle.requests, oracle.physical_points)
    event = search.verify(node['id'])
    assert event['verification_status'] == 'SKIPPED_DOMINATED'
    assert (oracle.requests, oracle.physical_points) == before
    assert plan not in search._checked and plan not in search.verification_queue
    row = next(r for r in search.pending() if r['branch_id'] == node['id'])
    assert row['verification_status'] == 'SKIPPED_DOMINATED' and not row['verified']
    assert row['can_expand'] and not row['can_verify']
    search.expand(node['id'])
    assert plan not in search.verification_queue and (oracle.requests, oracle.physical_points) == before
    assert search.result()['skipped_dominated_complete_plans'] == 1


def test_graph_expansion_cap_does_not_discard_unverified_queue():
    search, oracle = make(max_branch_expansions=1, max_generated_candidates=100)
    parent = completion(search)
    search.expand(parent['branch_id'])
    assert len(search._branch_keys) <= 100 and search.verification_queue
    target = next(r for r in search.pending() if r['origin'] == 'prefix_wait' and r['prefix'] == [16])
    with pytest.raises(ValueError, match='cap'):
        search.expand(target['branch_id'])
    event = search.verify(target['branch_id'])
    assert event['verification_status'] == 'COMPLETE'
    assert search.graph_expansions == search.verify_attempts == 1 and search.total_decisions == 2
    assert oracle.points <= oracle.budget


def test_partial_budget_attempt_is_not_a_failed_or_complete_certificate():
    search, oracle = make(budget=14)
    parent = completion(search)
    baseline = deepcopy(search.incumbent)
    event = search.verify(parent['branch_id'])
    assert event['verification_status'] == 'BUDGET_EXHAUSTED_PARTIAL'
    assert search.done and not search.failure_repairs and search.incumbent == baseline
    assert tuple(parent['actions']) in search.verification_queue
    row = next(r for r in search.pending() if r['branch_id'] == parent['branch_id'])
    assert not row['verified'] and row['verification_pending']
    assert oracle.points == 14 and event['new_actual_physical_points'] == 1


def test_unknown_certificate_stops_without_failure_repair_or_quality_credit():
    search, oracle = make()
    parent = completion(search)
    baseline = deepcopy(search.incumbent)
    oracle.unknown = True
    with pytest.raises(CLPUReliabilityError):
        search.verify(parent['branch_id'])
    assert search.done and not search.failure_repairs and search.incumbent == baseline
    assert tuple(parent['actions']) in search.verification_queue
    assert search.deferred_trace[-1]['error'] is not None
    with pytest.raises(RuntimeError):
        search.expand(parent['branch_id'])


def test_yield_terminates_early_without_claiming_completed_guard_or_old_actor_compatibility():
    search, oracle = make()
    before = oracle.physical_points
    with pytest.raises(TypeError):
        search.step(0)
    with pytest.raises(TypeError):
        search.observation('raw')
    event = search.yield_to_beam()
    assert event['operation'] == 'YIELD_TO_BEAM' and oracle.physical_points == before
    result = search.result()
    assert result['handed_off'] and result['complete_guard_still_required'] and not result['guard_established']
    assert result['total_decisions'] == 1 and result['graph_expansions'] == result['verify_attempts'] == 0
    with pytest.raises(RuntimeError):
        search.yield_to_beam()
    with pytest.raises(RuntimeError):
        search.verify(search.pending()[0]['branch_id'])
