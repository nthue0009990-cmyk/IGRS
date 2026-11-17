"""Source-prefix and independent teacher-fork checks, with no physical solver."""
from copy import deepcopy

import pytest

from dnr.nature34_clpu_branch import BranchConfig, BranchSearch
from dnr.nature34_clpu_counterfactual import (
    SnapshotUnavailable, fork_snapshot, restore_snapshot, sha_value,
    snapshot_from_dict, snapshot_prefix_point_count,
)
from dnr.nature34_clpu_environment import CLPUReliabilityError
from test_nature34_clpu_branch import CountedOracle, environment


BINDING = {'alpha': .5, 'base_condition_sha256': 'synthetic-condition',
           'physical_config_sha256': 'synthetic-policy', 'source_sha256': 'fixture-v1'}


def fail_new_pickup(state, seconds):
    return any(value is not None and value >= 0 for value in state.pickups)


class RecordedOracle(CountedOracle):
    def __init__(self, env, budget=256, fail=None):
        super().__init__(env, budget, fail=fail)
        self.rows = []
        self.closed = False

    def certify(self, state, seconds):
        before = self.points
        result = super().certify(state, seconds)
        if self.points > before:
            identity = {**self.env.identity(state, seconds), 'alpha': BINDING['alpha'],
                        'base_condition_sha256': BINDING['base_condition_sha256']}
            self.rows.append({'state_identity': identity, 'identity_sha256': sha_value(identity),
                              'point_id': result['point_id'], 'status': result['status'],
                              'reliable': result['reliable'],
                              'structured_feedback': {'violations': result['feedback']}})
        return result

    def __enter__(self):
        return self

    def __exit__(self, kind, value, tb):
        self.closed = True


def source(*, fail=fail_new_pickup, cap=32):
    env = environment(horizon=4)
    oracle = RecordedOracle(env, fail=fail)
    search = BranchSearch(env, oracle, BranchConfig(max_branch_expansions=cap))
    observations = {}
    while not search.done:
        if len(search.trace) in (0, 8):
            observations[len(search.trace)] = deepcopy(search.observation('raw'))
        search.step(search.rule_action('beam'))
    result = search.result()
    result['spec'] = {'method': 'beam'}
    return env, oracle, result, observations


def restored(*, position=0, fail=fail_new_pickup):
    env, oracle, result, observations = source(fail=fail)
    count = snapshot_prefix_point_count(result, position)
    snapshot = restore_snapshot(env, result, oracle.rows[:count], position, BINDING,
                                expected_observation=observations[position],
                                source_binding={'source_result_sha256': sha_value(result)})
    return snapshot, oracle, result


def factory_for(env, *, fail=fail_new_pickup):
    calls = []

    def factory(*, budget):
        live = RecordedOracle(env, budget, fail=fail)
        calls.append(live)
        return live

    return factory, calls


def nonwait_id(snapshot):
    observation = snapshot.search.observation('raw')
    for index, branch in enumerate(observation['branches']):
        if any(branch['actions']) and tuple(branch['actions']) not in snapshot.search._checked:
            return observation['branch_ids'][index]
    raise AssertionError('No unqueried non-WAIT candidate')


@pytest.mark.parametrize('position', [0, 8])
def test_exact_restore_uses_only_historical_prefix_and_zero_new_physics(position):
    snapshot, old_oracle, result = restored(position=position)
    before = old_oracle.points
    payload = snapshot.to_dict()['payload']
    restored_again = snapshot_from_dict(snapshot.env, snapshot.to_dict())
    assert old_oracle.points == before
    assert restored_again.sha256 == snapshot.sha256
    assert restored_again.search.result() == snapshot.search.result()
    assert restored_again.search.observation('raw') == payload['raw_observation']
    assert len(payload['source_trace']) == position
    assert len(payload['prefix_points']) == snapshot_prefix_point_count(result, position)
    assert len(restored_again.search.oracle.cache) == len(payload['prefix_points'])


def test_future_points_are_rejected_instead_of_loaded_for_free():
    env, oracle, result, observations = source()
    count = snapshot_prefix_point_count(result, 0)
    assert len(oracle.rows) > count
    with pytest.raises(CLPUReliabilityError, match='exactly'):
        restore_snapshot(env, result, oracle.rows, 0, BINDING)


def test_restore_rejects_source_identity_hash_or_unknown_status():
    env, oracle, result, _ = source()
    count = snapshot_prefix_point_count(result, 0)
    rows = deepcopy(oracle.rows[:count])
    rows[0]['identity_sha256'] = 'bad'
    with pytest.raises(CLPUReliabilityError, match='SHA'):
        restore_snapshot(env, result, rows, 0, BINDING)
    rows = deepcopy(oracle.rows[:count])
    rows[0]['status'] = 'UNKNOWN'
    rows[0]['reliable'] = False
    with pytest.raises(CLPUReliabilityError, match='Unknown'):
        restore_snapshot(env, result, rows, 0, BINDING)


def test_snapshot0_ignores_future_final_incumbent_and_future_trace_outcomes():
    env, oracle, result, obs = source()
    count = snapshot_prefix_point_count(result, 0)
    left = restore_snapshot(env, result, oracle.rows[:count], 0, BINDING)
    altered = deepcopy(result)
    altered['best_plan'] = {'fake_future_answer': 1000}
    altered['trace'][-1]['quality_after'] = -7
    right = restore_snapshot(env, altered, oracle.rows[:count], 0, BINDING)
    assert left.sha256 == right.sha256


def test_replay_detects_changed_recorded_action_or_observation():
    env, oracle, result, obs = source()
    count = snapshot_prefix_point_count(result, 8)
    altered = deepcopy(result)
    altered['trace'][1]['selected_index'] = (altered['trace'][1]['selected_index'] + 1) % 32
    with pytest.raises(CLPUReliabilityError, match='selection'):
        restore_snapshot(env, altered, oracle.rows[:count], 8, BINDING)
    wrong = deepcopy(obs[8])
    wrong['global_features'][0] += .1
    with pytest.raises(CLPUReliabilityError, match='observation'):
        restore_snapshot(env, result, oracle.rows[:count], 8, BINDING, expected_observation=wrong)


def test_changed_event_or_physics_and_payload_tampering_are_rejected():
    snapshot, _, _ = restored()
    changed = snapshot.to_dict()
    changed['payload']['prefix_points'][0]['certificate']['status'] = 'FAIL'
    with pytest.raises(CLPUReliabilityError, match='SHA'):
        snapshot_from_dict(snapshot.env, changed)
    snapshot.env.event['tau_seconds'] += 1
    with pytest.raises(CLPUReliabilityError, match='scope'):
        snapshot_from_dict(snapshot.env, snapshot.to_dict())


def test_independent_siblings_pay_again_and_map_local_ids_without_sharing():
    snapshot, _, _ = restored()
    before = snapshot.to_dict()
    factory, calls = factory_for(snapshot.env)
    branch = nonwait_id(snapshot)
    first = fork_snapshot(snapshot, branch, factory, max_expansions=1)
    second = fork_snapshot(snapshot, branch, factory, max_expansions=1)
    assert len(calls) == 2 and calls[0] is not calls[1]
    assert all(live.closed for live in calls)
    assert first['oracle']['new_physical_points'] == second['oracle']['new_physical_points'] > 0
    prefix = len(before['payload']['prefix_points'])
    assert first['oracle']['point_id_mappings'][0]['local_point_id'] == 0
    assert first['oracle']['point_id_mappings'][0]['cumulative_point_id'] == prefix
    assert first['new_attempts'][0]['validation']['samples'][0]['point_id'] == prefix
    assert snapshot.to_dict() == before
    assert first['fork_trace'] == second['fork_trace']


def test_new_answers_in_mutated_live_search_cannot_pollute_sealed_snapshot():
    snapshot, _, _ = restored()
    original = snapshot.to_dict()
    snapshot.search._frontier.clear()
    factory, _ = factory_for(snapshot.env)
    forced = original['payload']['raw_observation']['branch_ids'][0]
    result = fork_snapshot(snapshot, forced, factory, max_expansions=1)
    assert result['completed_expansions'] == 1
    assert snapshot.to_dict() == original


def test_cache_only_forced_wait_never_opens_a_cold_runtime():
    snapshot, _, _ = restored()
    observation = snapshot.search.observation()
    branch = next(observation['branch_ids'][i] for i, row in enumerate(observation['branches']) if not any(row['actions']))
    factory, calls = factory_for(snapshot.env)
    result = fork_snapshot(snapshot, branch, factory, max_expansions=1, max_new_points=0)
    assert calls == []
    assert result['oracle']['new_physical_points'] == 0
    assert result['oracle']['live_initialization_occurred'] is False
    assert result['oracle']['new_numerical_entries_expected_for_snapshot_oracle'] == 0
    assert result['g1'] == result['gH'] == 0


def test_local_budget_partial_retains_certified_incumbent_and_failure_record():
    snapshot, _, _ = restored(fail=None)
    factory, calls = factory_for(snapshot.env, fail=None)
    result = fork_snapshot(snapshot, nonwait_id(snapshot), factory, max_new_points=1, max_expansions=16)
    assert result['oracle']['new_physical_points'] == 1
    assert result['stop_reason'] == 'FORK_NEW_PHYSICAL_POINT_LIMIT'
    assert result['new_attempts'][-1]['status'] == 'BUDGET_EXHAUSTED_PARTIAL'
    assert result['new_attempts'][-1]['certified_complete'] is False
    assert result['best_plan']['certified_complete'] is True
    assert result['g1'] == result['gH'] == 0
    assert calls[0].closed


def test_gain_labels_use_complete_incumbent_and_forced_step_counts_inside_limit():
    snapshot, _, _ = restored(fail=None)
    factory, calls = factory_for(snapshot.env, fail=None)
    observation = snapshot.search.observation()
    forced = observation['branch_ids'][snapshot.search.rule_action('beam')]
    completed = []
    plans = []
    result = fork_snapshot(snapshot, forced, factory, max_expansions=2,
                           transition_callback=completed.append, record_callback=plans.append)
    assert result['completed_expansions'] <= 2
    assert result['transitions'][0]['forced']
    assert not any(row['forced'] for row in result['transitions'][1:])
    assert result['g1'] > 0 and result['gH'] >= result['g1']
    assert result['g1'] == pytest.approx(result['first_step_quality'] - result['initial_quality'])
    assert result['gH'] == pytest.approx(result['final_quality'] - result['initial_quality'])
    assert result['best_plan']['certified_complete']
    assert len(completed) == 2 * result['completed_expansions']
    assert [row['status'] for row in completed[:2]] == ['STARTED', 'COMPLETED']
    assert 'obs_raw' not in completed[0]
    assert sha_value(completed[1]['obs_raw']) == completed[1]['observation_sha256']
    assert all(sha_value(row['obs_raw']) == row['observation_sha256'] for row in result['transitions'])
    assert len(plans) == len(result['new_attempts'])


def test_nonempty_live_cache_is_rejected_and_context_closes():
    snapshot, _, _ = restored()
    calls = []

    def contaminated(*, budget):
        live = RecordedOracle(snapshot.env, budget)
        live.cache['future_answer'] = {'status': 'PASS'}
        calls.append(live)
        return live

    with pytest.raises(CLPUReliabilityError, match='nonempty'):
        fork_snapshot(snapshot, nonwait_id(snapshot), contaminated, max_expansions=1)
    assert calls[0].closed


@pytest.mark.parametrize('change', ['budget', 'event'])
def test_live_oracle_must_match_the_frozen_context_and_admitted_budget(change):
    snapshot, _, _ = restored()
    calls = []

    def mismatched(*, budget):
        env = environment(horizon=4)
        if change == 'event':
            env.event['a'] += .5
        live = RecordedOracle(env, budget + 1 if change == 'budget' else budget)
        calls.append(live)
        return live

    with pytest.raises(CLPUReliabilityError, match='budget differs|environment differs'):
        fork_snapshot(snapshot, nonwait_id(snapshot), mismatched, max_expansions=1)
    assert calls[0].closed


def test_unavailable_positions_and_invalid_resource_limits_are_not_labels():
    env, oracle, result, _ = source(cap=2)
    with pytest.raises(SnapshotUnavailable, match='POSITION_ABSENT'):
        snapshot_prefix_point_count(result, 8)
    with pytest.raises(ValueError, match='0/8'):
        snapshot_prefix_point_count(result, 3)
    snapshot, _, _ = restored()
    factory, calls = factory_for(snapshot.env)
    with pytest.raises(ValueError, match='128'):
        fork_snapshot(snapshot, nonwait_id(snapshot), factory, max_new_points=129)
    with pytest.raises(ValueError, match='16'):
        fork_snapshot(snapshot, nonwait_id(snapshot), factory, max_expansions=17)
    assert calls == []
