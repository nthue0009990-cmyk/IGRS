"""Shared branch-engine checks using immutable histories and a fake oracle.

These tests never import the DSS runtime or solve a physical power flow.
"""
from copy import deepcopy
import math

import pytest

from dnr.nature34_clpu_branch import BranchConfig, BranchSearch
from dnr.nature34_clpu_environment import CLPUEnvironment, CLPUBudgetExhausted, CLPUReliabilityError
from dnr.nature34_clpu_search import graph_plan
from dnr.nature34_joint_sequence import JointGraph


def environment(horizon=6, fault_lines=()):
    edges = [['s', str(i)] for i in range(1, 10)]
    loads = {f'load.l{i}': {'bus': str(i), 'nominal_kw': float(i), 'nominal_kvar': i / 2.}
             for i in range(1, 10)}
    loads['load.l10'] = {'bus': 's', 'nominal_kw': 1., 'nominal_kvar': .5}
    loads['load.fixed'] = {'bus': '6', 'nominal_kw': 7., 'nominal_kvar': 3.5}
    static = {'nodes': ['s'] + [str(i) for i in range(1, 10)], 'source_bus': 's',
              'bus_edges': edges, 'switch_edges': edges,
              'original_lines': {f'line.l{i}': edge for i, edge in enumerate(edges, 1)},
              'dispatch_loads': [f'load.l{i}' for i in range(1, 11)]}
    scenario = {'scenario_id': 'branch_synthetic', 'partition': 'build', 'fault_lines': list(fault_lines)}
    condition = {'condition_id': 'toy_normal', 'load_multiplier': 1.,
                 'denominator_nominal_kw': sum(v['nominal_kw'] for v in loads.values())}
    event = {'event_id': 91, 'a': .5, 'tau_seconds': 120.}
    return CLPUEnvironment(JointGraph(static, loads, scenario), static, loads, scenario, condition,
                           event, horizon=horizon, slot_seconds=60, sample_seconds=30)


class CountedOracle:
    def __init__(self, env, budget=256, *, fail=None, unknown=False):
        self.env, self.budget = env, budget
        self.points = self.requests = self.hits = 0
        self.cache, self.calls = {}, []
        self.fail, self.unknown = fail, unknown

    def certify(self, state, seconds):
        self.requests += 1
        key = (state.command, state.pickups, seconds)
        if key in self.cache:
            self.hits += 1
            return deepcopy(self.cache[key])
        if self.points >= self.budget:
            raise CLPUBudgetExhausted('Synthetic point cap')
        pid = self.points
        self.points += 1
        self.calls.append((state, seconds))
        failed = bool(self.fail and self.fail(state, seconds))
        result = {'point_id': pid, 'status': 'UNKNOWN' if self.unknown else 'FAIL' if failed else 'PASS',
                  'reliable': not self.unknown, 'feedback': [{'constraint': 'voltage', 'bus': '6',
                  'phase': 1, 'magnitude': .02}] if failed else []}
        self.cache[key] = deepcopy(result)
        return result


def engine(budget=256, *, env=None, fail=None, **config):
    env = env or environment()
    oracle = CountedOracle(env, budget, fail=fail)
    return BranchSearch(env, oracle, BranchConfig(point_budget=budget, **config)), oracle


def choose(search, predicate):
    for i, row in enumerate(search.observation()['branches']):
        if predicate(row):
            return i
    raise AssertionError('Required synthetic branch missing from common window')


def test_bootstrap_charges_initial_and_complete_wait_in_same_oracle():
    search, oracle = engine()
    assert oracle.calls[0] == (search.env.initial, 0.)
    assert len(search.attempts) == 1
    assert search.attempts[0]['origin'] == 'wait_incumbent'
    assert search.attempts[0]['status'] == 'COMPLETE'
    assert len(search.attempts[0]['validation']['samples']) == search.horizon * 3
    assert oracle.points == 2 * search.horizon + 1
    assert oracle.requests == 1 + 3 * search.horizon
    assert oracle.hits == oracle.requests - oracle.points


def test_partial_budget_does_not_create_complete_wait_fallback():
    search, oracle = engine(2)
    result = search.result()
    assert search.done and oracle.points == 2
    assert result['status'] == 'NO_CERTIFIED_COMPLETE_PLAN'
    assert result['incumbent'] is None and result['best_plan'] is None
    assert result['attempts'][0]['status'] == 'BUDGET_EXHAUSTED_PARTIAL'
    assert not result['attempts'][0]['certified_complete']


def test_initial_fail_is_retained_without_unpaid_search():
    search, oracle = engine(fail=lambda state, seconds: True)
    assert search.done and search.result()['status'] == 'INITIAL_FAIL'
    assert oracle.points == 1 and not search.attempts
    assert search.result()['initial_certificate']['status'] == 'FAIL'


def test_unknown_initial_certificate_propagates_as_reliability_error():
    env = environment()
    with pytest.raises(CLPUReliabilityError, match='Unknown'):
        BranchSearch(env, CountedOracle(env, unknown=True))


def test_dynamic_selection_adds_branches_and_preserves_actual_initial_history():
    search, _ = engine()
    original = search.env.initial.to_dict()
    initial_ids = set(search._frontier)
    selected = choose(search, lambda row: row['prefix'] == [16] and row['origin'] == 'prefix_wait')
    event = search.step(selected)
    assert event['new_branches'] > 0
    assert set(search._frontier) - initial_ids
    assert any(len(row['prefix']) > 1 for row in search._frontier.values())
    assert search.env.initial.to_dict() == original
    assert event['physical_points_used'] > 0


def test_partial_restoration_is_a_complete_wait_suffix_plan_not_a_prefix_label():
    search, _ = engine()
    selected = choose(search, lambda row: row['prefix'] == [16] and row['origin'] == 'prefix_wait')
    search.step(selected)
    record = search.attempts[-1]
    assert record['actions'] == [16] + [0] * (search.horizon - 1)
    assert record['status'] == 'COMPLETE'
    assert len(record['validation']['samples']) == search.horizon * 3
    assert record['validation']['states'][-1]['pickups'] != list(search.start.pickups)
    assert len(record['validation']['states'][-1]['pickups']) == len(search.env.load_names)


def test_different_branch_choices_generate_different_dynamic_frontiers():
    left, _ = engine()
    right, _ = engine()
    left.step(choose(left, lambda r: r['prefix'] == [16] and r['origin'] == 'prefix_wait'))
    right.step(choose(right, lambda r: r['prefix'] == [17] and r['origin'] == 'prefix_wait'))
    assert left._branch_keys != right._branch_keys


def test_common_window_never_deletes_unexposed_branches_and_includes_best_score():
    search, _ = engine(action_window=8, beam_width=8)
    before = set(search._frontier)
    observation = search.observation()
    assert len(observation['branch_ids']) == 8
    assert set(search._frontier) == before
    best = min(search._frontier.values(), key=lambda b: (b['info']['priority'], b['id']))
    assert best['id'] in observation['branch_ids']
    assert set(search._frontier) - set(observation['branch_ids'])
    assert observation['mask'] == observation['action_mask'] == [True] * 8


def test_feedback_is_equal_and_association_appends_only_public_derived_features():
    env = environment()
    idx = env.load_names.index('load.l6')
    search, _ = engine(env=env, fail=lambda state, seconds: state.pickups[idx] is not None)
    search.step(choose(search, lambda r: r['prefix'] == [16] and r['origin'] == 'prefix_wait'))
    raw, associated = search.observation('raw'), search.observation('associated')
    assert raw['global_features'] == associated['global_features']
    assert raw['observed_feedback'] == associated['observed_feedback']
    assert raw['observed_failure_history'] == associated['observed_failure_history']
    assert raw['observed_feedback'][0]['phase'] == 1
    assert raw['branch_ids'] == associated['branch_ids']
    assert len(raw['global_features']) == 11 + len(env.load_names)
    for a, b in zip(raw['candidate_features'], associated['candidate_features']):
        assert a == b[:-8]
        assert len(a) == 27 + env.horizon + 3 * len(env.load_names)
        assert all(math.isfinite(x) for x in b)
    assert len(raw['candidate_feature_names']) == len(raw['candidate_features'][0])
    assert len(associated['candidate_feature_names']) == len(associated['candidate_features'][0])


def test_rules_and_learned_indices_use_identical_engine_window_and_operators():
    left, _ = engine()
    right, _ = engine()
    index = left.rule_action('greedy')
    assert left.observation()['branch_ids'] == right.observation()['branch_ids']
    assert left.step(index) == right.step(index)
    assert left.result()['attempts'] == right.result()['attempts']
    assert left._branch_keys == right._branch_keys


@pytest.mark.parametrize('rule', ['greedy', 'beam'])
def test_rules_complete_bounded_run_with_all_costs_and_records_preserved(rule):
    search, oracle = engine(budget=80, max_generated_candidates=60)
    steps = 0
    while not search.done:
        event = search.step(search.rule_action(rule))
        assert event['reward'] == pytest.approx(event['quality_after'] - event['quality_before']
                                               - .001 * event['physical_points_used'] / 256.)
        steps += 1
        assert steps <= 60
    result = search.result()
    assert oracle.points <= 80
    assert result['assessed_graph_candidates'] <= 60
    assert result['generated_unique_candidates'] <= 60
    assert result['best_plan'] == result['incumbent']
    assert result['incumbent']['status'] == 'COMPLETE'
    assert all(r['status'] != 'BUDGET_EXHAUSTED_PARTIAL' or not r['certified_complete']
               for r in result['attempts'])


def test_dominated_plan_is_expanded_without_pay_to_certify_known_inferior_objective():
    search, oracle = engine()
    index = choose(search, lambda r: r['actions'] == [0] * search.horizon)
    before = oracle.points
    result = search.step(index)
    assert result['query_reason'] == 'previously_checked'
    assert oracle.points == before
    assert result['new_branches'] > 0


def test_six_step_completion_is_never_claimed_to_be_a_global_upper_bound():
    search, _ = engine()
    assert search.incumbent['validation']['energy_kwh'] < search.denominator
    assert all(b['upper_bound_kwh'] == search.denominator for b in search._frontier.values())
    # Even a high-value generated candidate does not remove unexpanded choices.
    search.step(search.rule_action('greedy'))
    assert search.stop_reason != 'CERTIFIED_FRONTIER_UPPER_BOUND'


def test_valid_global_bound_can_stop_without_paying_for_irrelevant_expansion():
    env = environment(horizon=2, fault_lines=['line.l6', 'line.l7', 'line.l8', 'line.l9'])
    # All currently unpowered buses are severed by fixed synthetic faults.
    # WAIT attains the graph limit without inventing a feasible restoration.
    search, _ = engine(env=env)
    for row in search._frontier.values():
        assert row['upper_bound_kwh'] < search.denominator
        assert row['upper_bound_kwh'] == search.incumbent['validation']['energy_kwh']
    assert search.done and search.stop_reason == 'CERTIFIED_FRONTIER_UPPER_BOUND'
    assert len(search.attempts) == 1 and search.result()['optimality_established']


def test_reachable_graph_bound_contains_every_normal_service_command():
    env = environment(fault_lines=['line.l9'])
    search, _ = engine(env=env)
    assert search.max_graph_kw == env.total_kw - 9.
    assert search.global_upper_bound < search.denominator
    assert all(env.graph.served_base_kw(int(s)) <= search.max_graph_kw for s in env.graph.states)


def test_generation_records_allow_independent_graph_and_parent_reconstruction():
    search, _ = engine()
    search.step(search.rule_action('greedy'))
    result = search.result()
    trace = result['generation_trace']
    assert len(trace) == result['generated_unique_candidates']
    assert len(result['graph_assessment_trace']) == result['assessed_graph_candidates']
    known = set()
    for row in trace:
        assert row['branch_id'] not in known
        assert row['parent_id'] is None or row['parent_id'] in known
        known.add(row['branch_id'])
        assert row['actions'][:len(row['prefix'])] == row['prefix']
        independent = graph_plan(search.env, search.start, row['actions'])
        assert independent['energy_kwh'] == row['graph_energy_kwh']
        assert row['upper_bound_kwh'] >= independent['energy_kwh']


def test_callback_receives_every_plan_attempt_without_mutating_internal_records():
    env = environment()
    oracle = CountedOracle(env, budget=15)
    emitted = []

    def callback(record):
        emitted.append(deepcopy(record))
        record['status'] = 'CALLBACK_MUTATION'

    search = BranchSearch(env, oracle, BranchConfig(point_budget=15), callback)
    while not search.done:
        search.step(search.rule_action())
    assert search.attempts == emitted
    assert all(row['status'] != 'CALLBACK_MUTATION' for row in search.attempts)


def test_invalid_action_and_oracle_budget_mismatch_are_not_silently_fixed():
    search, _ = engine()
    with pytest.raises(ValueError, match='outside'):
        search.step(32)
    with pytest.raises(ValueError, match='budgets differ'):
        BranchSearch(environment(), CountedOracle(environment(), budget=100), BranchConfig())


def test_generated_full_paths_preserve_original_legal_history_and_reward_formula():
    search, _ = engine()
    search.step(choose(search, lambda r: r['prefix'] == [16] and r['origin'] == 'prefix_wait'))
    for node in search._frontier.values():
        plan = graph_plan(search.env, search.start, node['info']['actions'])
        assert len(plan['actions']) == search.horizon
        assert plan['actions'][:len(node['prefix'])] == node['prefix']
        assert plan['energy_kwh'] <= search.denominator + 1e-10
