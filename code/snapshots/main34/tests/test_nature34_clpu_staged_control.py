from copy import deepcopy
import hashlib
import json

import pytest

from dnr.nature34_clpu_branch import BranchConfig, BranchSearch
from dnr.nature34_clpu_staged_control import StagedCertificationSearch, StagedVerificationAdapter, StagedMacroController
from dnr.nature34_clpu_staged_pickup import generate_staged_pickup_candidates
from test_nature34_clpu_branch import environment
from test_nature34_clpu_deferred import ActualCountedOracle
from test_nature34_clpu_deferred_control import model_environment, MODEL_SOURCE


def make(*, budget=256, fail=None, env=None):
    env = env or environment(horizon=30)
    oracle = ActualCountedOracle(env, budget=budget, fail=fail)
    search = StagedCertificationSearch(env, oracle, BranchConfig(point_budget=budget))
    return search, oracle


def test_bootstrap_original_cost_and_fixed_pool_are_not_reset_or_replaced_later():
    search, oracle = make()
    reference_oracle = ActualCountedOracle(search.env, budget=256)
    reference = BranchSearch(search.env, reference_oracle)
    assert oracle.points == reference_oracle.points == 61
    assert oracle.requests == reference_oracle.requests == 91
    assert search.initial_certificate == reference.initial_certificate
    assert search.incumbent == reference.incumbent
    assert search.stage_generation == generate_staged_pickup_candidates(search.env)
    assert len(search._frontier) == 12 and {n['origin'] for n in search._frontier.values()} == {'staged_pickup'}
    assert search.total_decisions == search.graph_expansions == search.verify_attempts == 0
    assert search.stage_generation_seconds <= search.bootstrap_cost['elapsed_seconds']


def test_adapter_uses_exact_old_features_and_odd_pending_held_out_subset():
    search, oracle = make()
    adapter = StagedVerificationAdapter(search)
    before = deepcopy(search.stage_generation)
    ordinary = adapter.observation('raw')
    legacy = BranchSearch.__new__(BranchSearch)
    legacy.__dict__ = search.__dict__.copy()
    legacy._window = adapter._window
    assert ordinary == BranchSearch.observation(legacy, 'raw')
    assert adapter.rule_action('beam') == BranchSearch.rule_action(legacy, 'beam')
    assert len(ordinary['mask']) == 12 and all(ordinary['mask'])
    assert all(row[5] == 0 for row in ordinary['candidate_features'])
    search.verify_attempts = 1
    expected = [n['id'] for n in search._frontier.values()
                if search.stage_branch_metadata[n['id']]['held_out_loads']
                and n['info']['energy_kwh'] > search.incumbent['validation']['energy_kwh']]
    assert expected and adapter.observation('raw')['branch_ids'] == expected
    assert search.stage_generation == before and oracle.points == 61


def test_verify_failure_updates_feedback_but_fixed_pool_never_grows():
    search, oracle = make()
    control = StagedMacroController(search)
    count = len(search._frontier)
    oracle.fail = lambda state, seconds: state != search.env.initial
    proposal = control.prepare()
    response = control.apply(proposal)
    assert response['verification_status'] == 'FAIL'
    assert search.latest_failure and search.verify_attempts == 1
    assert len(search._frontier) == count and search.graph_expansions == 0
    assert proposal['branch_id'] not in control.adapter.observation('raw')['branch_ids']
    while (proposal := control.prepare()) is not None:
        control.apply(proposal)
    result = control.result()
    assert result['total_decisions'] <= 17 and result['verify_attempts'] <= 16
    assert result['guard_established'] is False and result['complete_guard_still_required'] is True


@pytest.mark.parametrize('budget,fail,reason', [(2, None, 'PHYSICAL_BUDGET_EXHAUSTED'),
    (256, lambda state, seconds: True, 'INITIAL_FAIL')])
def test_bootstrap_stop_never_generates_pool_or_calls_policy(budget, fail, reason):
    search, oracle = make(budget=budget, fail=fail)
    control = StagedMacroController(search)
    assert search.done and search.stage_generation is None
    assert control.prepare() is None
    assert control.result()['control']['finish_reason'] == reason
    assert control.result()['total_decisions'] == 0


def test_stale_selection_and_no_expansion_are_rejected():
    search, oracle = make()
    control = StagedMacroController(search)
    proposal = control.prepare()
    with pytest.raises(TypeError, match='fixed'):
        search.expand(proposal['branch_id'])
    altered = deepcopy(proposal)
    altered['branch_id'] += 1
    with pytest.raises(RuntimeError, match='stale'):
        control.apply(altered)


def test_all_bound_actors_accept_staged_raw_features_and_mask_without_physics():
    import torch
    from dnr.nature34_clpu_branch_learning import load_policy, encode_observations
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        search, oracle = make(env=model_environment())
        adapter = StagedVerificationAdapter(search)
        observation = adapter.observation('raw')
        assert len(observation['global_features']) == 79
        assert all(len(x) == 261 for x in observation['candidate_features'])
        assert tuple(encode_observations([observation], 'raw')['candidates'].shape) == (1, 32, 261)
        before = oracle.points, oracle.requests
        bindings = json.loads((MODEL_SOURCE / 'model_bindings.json').read_text(encoding='utf8'))
        assert len(bindings) == 6
        for binding in bindings:
            path = MODEL_SOURCE / binding['checkpoint']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == binding['source_sha256']
            policy = load_policy(path, 'raw')
            adapter.validate_policy(policy)
            choice = policy.act(observation, observation['mask'], deterministic=True)
            assert observation['mask'][choice['action']]
        assert before == (oracle.points, oracle.requests)
    finally:
        torch.set_num_threads(old_threads)
