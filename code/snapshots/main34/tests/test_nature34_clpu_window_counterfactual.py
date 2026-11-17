"""Coverage-window teacher checks; synthetic oracle only, no physical solver."""
from copy import deepcopy

import pytest

from dnr.nature34_clpu_branch import BranchConfig
from dnr.nature34_clpu_environment import CLPUReliabilityError
from dnr.nature34_clpu_window import WindowBranchSearch
from dnr.nature34_clpu_window_counterfactual import (
    SnapshotUnavailable, fork_snapshot, quality_curve_metric, restore_snapshot,
    sha_value, snapshot_from_dict, snapshot_prefix_point_count, state_eligibility,
)
from test_nature34_clpu_branch import environment
from test_nature34_clpu_counterfactual import (
    BINDING, RecordedOracle, factory_for, fail_new_pickup, nonwait_id,
)
from test_nature34_clpu_window import row, shell


def source(*, fail=fail_new_pickup, cap=32):
    env = environment(horizon=4)
    oracle = RecordedOracle(env, fail=fail)
    search = WindowBranchSearch(env, oracle, BranchConfig(max_branch_expansions=cap))
    observations, steps = {}, []
    while not search.done:
        observations[search._beam_turn] = deepcopy(search.observation('raw'))
        action = search.rule_action('beam')
        window = search._window()
        boundary = {'transition_id': search._beam_turn, 'status': 'STARTED', 'method': 'beam',
                    'action': action, 'branch_id': window[action]['id'],
                    'window_branch_ids': [x['id'] for x in window],
                    'oracle_before': {'points': oracle.points, 'requests': oracle.requests}}
        steps.append(deepcopy(boundary))
        response = search.step(action)
        boundary.update(status='COMPLETED', response=deepcopy(response),
                        oracle_after={'points': oracle.points, 'requests': oracle.requests})
        steps.append(boundary)
    result = {'config': search.result()['config'], 'best_plan': search.incumbent,
              'spec': {'method': 'beam', 'window_variant': 'coverage_v1'}}
    return env, oracle, result, steps, search.attempts, observations


def restored(*, position=0, fail=fail_new_pickup):
    env, oracle, result, steps, plans, obs = source(fail=fail)
    count = snapshot_prefix_point_count(result, steps, plans, position)
    value = restore_snapshot(env, result, steps, plans, oracle.rows[:count], position,
                             BINDING, expected_observation=obs[position])
    return value, oracle, result, steps, plans


@pytest.mark.parametrize('position', [0, 2, 3, 8])
def test_arbitrary_action_boundary_replay_is_exact_and_zero_solve(position):
    snapshot, oracle, result, steps, plans = restored(position=position)
    before = oracle.points
    again = snapshot_from_dict(snapshot.env, snapshot.to_dict())
    assert oracle.points == before
    assert again.sha256 == snapshot.sha256
    assert again.search.result() == snapshot.search.result()
    assert isinstance(again.search, WindowBranchSearch)
    payload = again.to_dict()['payload']
    assert len(payload['source_trace']) == position
    assert len(payload['prefix_points']) == snapshot_prefix_point_count(result, steps, plans, position)
    assert payload['source_attempts'] == again.search.attempts


def test_snapshot_ignores_future_final_result_future_plans_and_future_steps():
    env, oracle, result, steps, plans, _ = source()
    count = snapshot_prefix_point_count(result, steps, plans, 0)
    left = restore_snapshot(env, result, steps, plans, oracle.rows[:count], 0, BINDING)
    altered = deepcopy(result)
    altered['best_plan'] = {'fake_future_answer': 1000}
    right = restore_snapshot(env, altered, [{'not_yet_observed': True}],
        plans[:1] + [{'future_malformed': True}], oracle.rows[:count], 0, BINDING)
    assert left.sha256 == right.sha256
    assert len(left.to_dict()['payload']['source_attempts']) == 1


def test_future_physics_and_modified_plan_prefix_are_rejected():
    env, oracle, result, steps, plans, _ = source()
    count = snapshot_prefix_point_count(result, steps, plans, 0)
    assert oracle.points > count
    with pytest.raises(CLPUReliabilityError, match='exactly'):
        restore_snapshot(env, result, steps, plans, oracle.rows, 0, BINDING)
    changed = deepcopy(plans)
    changed[0]['physical_points_after'] -= 1
    with pytest.raises(CLPUReliabilityError):
        restore_snapshot(env, result, steps, changed, oracle.rows[:count], 0, BINDING)


def test_replay_checks_window_action_boundary_observation_and_scope():
    env, oracle, result, steps, plans, obs = source()
    position = 3
    count = snapshot_prefix_point_count(result, steps, plans, position)
    changed = deepcopy(steps)
    changed[1]['branch_id'] += 1
    with pytest.raises(CLPUReliabilityError, match='boundary differs'):
        restore_snapshot(env, result, changed, plans, oracle.rows[:count], position, BINDING)
    changed = deepcopy(obs[position])
    changed['global_features'][0] += .1
    with pytest.raises(CLPUReliabilityError, match='observation'):
        restore_snapshot(env, result, steps, plans, oracle.rows[:count], position, BINDING,
                         expected_observation=changed)
    snapshot, _, _, _, _ = restored(position=position)
    original = snapshot.to_dict()
    tampered = deepcopy(original)
    tampered['payload']['window_rule_version'] = 'different'
    tampered['snapshot_sha256'] = sha_value(tampered['payload'])
    with pytest.raises(CLPUReliabilityError, match='window'):
        snapshot_from_dict(snapshot.env, tampered)
    snapshot.env.event['a'] += .01
    with pytest.raises(CLPUReliabilityError, match='scope'):
        snapshot_from_dict(snapshot.env, original)


def test_siblings_pay_again_without_mutating_sealed_history():
    snapshot, _, _, _, _ = restored()
    before = snapshot.to_dict()
    factory, calls = factory_for(snapshot.env)
    forced = nonwait_id(snapshot)
    left = fork_snapshot(snapshot, forced, factory, max_expansions=3)
    snapshot.search._frontier.clear()
    right = fork_snapshot(snapshot, forced, factory, max_expansions=3)
    assert len(calls) == 2 and all(x.closed for x in calls)
    assert left['oracle']['new_physical_points'] == right['oracle']['new_physical_points'] > 0
    assert left['fork_trace'] == right['fork_trace']
    assert snapshot.to_dict() == before
    assert left['oracle']['point_id_mappings'][0]['local_point_id'] == 0
    assert left['oracle']['historical_restore_new_physical_points'] == 0


def test_partial_certificate_has_no_curve_credit_and_budget_is_paid():
    snapshot, _, _, _, _ = restored(fail=None)
    factory, calls = factory_for(snapshot.env, fail=None)
    result = fork_snapshot(snapshot, nonwait_id(snapshot), factory, max_new_points=1)
    assert result['new_attempts'][-1]['status'] == 'BUDGET_EXHAUSTED_PARTIAL'
    assert result['oracle']['new_physical_points'] == 1
    assert result['certificate_updates'] == []
    assert len(result['quality_point_trajectory']) == 1
    assert result['auc_gain_pp'] == result['gH_pp'] == 0
    assert result['best_plan']['certified_complete'] and calls[0].closed


def test_complete_callback_updates_curve_before_incumbent_assignment():
    snapshot, _, _, _, _ = restored(fail=None)
    factory, calls = factory_for(snapshot.env, fail=None)
    search = snapshot.search
    forced = search.observation()['branch_ids'][search.rule_action('beam')]
    plans, transitions = [], []
    result = fork_snapshot(snapshot, forced, factory, record_callback=plans.append,
                           transition_callback=transitions.append)
    assert result['gH_pp'] > 0
    assert result['certificate_updates']
    update = result['certificate_updates'][0]
    assert update['quality_before'] == result['initial_quality']
    assert update['quality_after'] > update['quality_before']
    assert result['quality_point_trajectory'][-1]['quality'] == result['final_quality']
    assert result['auc_gain_pp'] > 0
    assert len(plans) == len(result['new_attempts'])
    assert len(transitions) == 2 * result['completed_expansions']
    assert result['transitions'][0]['forced']
    assert all(not t['forced'] for t in result['transitions'][1:])


def test_continuation_calls_current_window_beam_not_original_search(monkeypatch):
    snapshot, _, _, _, _ = restored()
    factory, _ = factory_for(snapshot.env)
    original = WindowBranchSearch.rule_action
    calls = []

    def spy(self, method='greedy'):
        calls.append((id(self), self._beam_turn, [x['id'] for x in self._window()]))
        return original(self, method)

    monkeypatch.setattr(WindowBranchSearch, 'rule_action', spy)
    result = fork_snapshot(snapshot, nonwait_id(snapshot), factory, max_expansions=3)
    assert result['completed_expansions'] == 3
    assert [turn for _, turn, _ in calls] == [1, 2]


def test_zero_point_allowance_cache_action_does_not_fabricate_auc():
    snapshot, _, _, _, _ = restored()
    obs = snapshot.search.observation()
    forced = next(obs['branch_ids'][i] for i, r in enumerate(obs['branches']) if not any(r['actions']))
    factory, calls = factory_for(snapshot.env)
    result = fork_snapshot(snapshot, forced, factory, max_new_points=0, max_expansions=1)
    assert not calls
    assert result['oracle']['new_physical_points'] == 0
    assert result['auc_gain_pp'] is None


def curve(initial, rows):
    return [{'new_physical_points': 0, 'quality': initial, 'kind': 'INITIAL_CERTIFIED_INCUMBENT'}] + [
        {'new_physical_points': p, 'quality': q, 'kind': 'COMPLETE_CERTIFICATE'} for p, q in rows]


@pytest.mark.parametrize('rows,budget,expected', [
    ([], 4, 0.), ([(0, .6)], 4, 40.), ([(4, .6)], 4, 10.),
    ([(2, .4), (2, .6)], 4, 30.), ([(2, .4), (4, .6)], 4, 20.),
    ([(1, .6)], 4, 40.), ([], 0, None),
])
def test_exact_discrete_curve_boundary_semantics(rows, budget, expected):
    result = quality_curve_metric(.2, curve(.2, rows), budget)
    assert result is None if expected is None else result == pytest.approx(expected)


@pytest.mark.parametrize('rows', [[(2, .1)], [(3, .4), (2, .5)], [(5, .5)], [(1, float('nan'))]])
def test_invalid_nonmonotone_or_unpaid_curves_rejected(rows):
    with pytest.raises(ValueError):
        quality_curve_metric(.2, curve(.2, rows), 4)


def test_state_selection_is_unique_untested_graph_improvement_not_electrical_claim():
    first = row(1, (2,), energy=3.)
    duplicate = row(2, (2,), energy=3., actions=first['info']['actions'])
    search = shell([first, duplicate, row(3, (3,), energy=4.), row(4, (4,), energy=.1)], turn=1)
    state = state_eligibility(search)
    assert state['mode'] == 'coverage' and state['eligible']
    assert state['candidate_ids'] == [1, 3]
    search._checked[first['info']['actions']] = {'status': 'FAIL'}
    assert not state_eligibility(search)['eligible']
    search._checked[search._frontier[3]['info']['actions']] = {'status': 'FAIL'}
    assert state_eligibility(search)['mode'] == 'normal'


def test_terminal_missing_wrong_source_and_bad_limits_are_not_labels():
    env, oracle, result, steps, plans, _ = source(cap=2)
    with pytest.raises(SnapshotUnavailable, match='TERMINAL'):
        snapshot_prefix_point_count(result, steps, plans, 2)
    with pytest.raises(SnapshotUnavailable, match='ABSENT'):
        snapshot_prefix_point_count(result, steps, plans, 3)
    with pytest.raises(ValueError, match='Nonnegative'):
        snapshot_prefix_point_count(result, steps, plans, -1)
    wrong = deepcopy(result)
    wrong['spec']['window_variant'] = 'original'
    with pytest.raises(ValueError, match='coverage_v1'):
        snapshot_prefix_point_count(wrong, steps, plans, 0)
    snapshot, _, _, _, _ = restored()
    factory, calls = factory_for(snapshot.env)
    with pytest.raises(ValueError, match='128'):
        fork_snapshot(snapshot, nonwait_id(snapshot), factory, max_new_points=129)
    with pytest.raises(ValueError, match='16'):
        fork_snapshot(snapshot, nonwait_id(snapshot), factory, max_expansions=17)
    assert not calls


def test_contaminated_new_oracle_is_closed_and_rejected():
    snapshot, _, _, _, _ = restored()
    calls = []

    def contaminated(*, budget):
        live = RecordedOracle(snapshot.env, budget)
        live.cache['future'] = {'status': 'PASS'}
        calls.append(live)
        return live

    with pytest.raises(CLPUReliabilityError, match='nonempty'):
        fork_snapshot(snapshot, nonwait_id(snapshot), contaminated, max_expansions=1)
    assert calls[0].closed
