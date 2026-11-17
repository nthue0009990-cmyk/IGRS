"""Fixed-schedule/controller checks; all certificates below use counted fakes."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from dnr.nature34_clpu_branch import BranchConfig, BranchSearch
from dnr.nature34_clpu_deferred import DeferredCertificationSearch
from dnr.nature34_clpu_deferred_control import (
    DeferredMacroController, PendingVerificationAdapter, fixed_expand_choice, EXPAND, VERIFY, YIELD,
)
from dnr.nature34_clpu_environment import CLPUEnvironment, CLPUReliabilityError
from dnr.nature34_joint_sequence import JointGraph
from test_nature34_clpu_branch import environment
from test_nature34_clpu_deferred import ActualCountedOracle, completion


ROOT = Path(__file__).resolve().parents[1]
MODEL_SOURCE = ROOT / 'runs/20261031T020246_028867Z_clpu_early_pilot'


def make(*, env=None, budget=256, **config):
    env = env or environment(horizon=6)
    oracle = ActualCountedOracle(env, budget=budget)
    return DeferredCertificationSearch(env, oracle, BranchConfig(point_budget=budget, **config)), oracle


class LastSlotPolicy:
    representation = 'raw'

    def __init__(self, search):
        metadata = PendingVerificationAdapter(search).metadata()
        self.architecture = {'candidate_dim': metadata['candidate_raw_dim'],
            'global_dim': metadata['global_dim'], 'representation': 'raw', 'hidden': 128}
        self.calls = 0

    def act(self, obs, mask, **kwargs):
        assert kwargs['deterministic'] and mask == obs['mask']
        self.calls += 1
        return {'action': len(mask) - 1, 'logprob': -.1, 'value': .2}


def finish(controller):
    proposals = []
    while (proposal := controller.prepare()) is not None:
        proposals.append(proposal)
        controller.apply(proposal)
    return proposals, controller.result()


def test_pending_alias_is_shortest_prefix_and_does_not_mutate_engine_or_features():
    search, oracle = make()
    parent = completion(search)
    hold = tuple(parent['actions'][:2] + [0] * 4)
    search._add(hold, hold, 'generic_replace', parent['branch_id'])
    old_alias = search._next_id - 1
    search._add(hold[:2], hold, 'prefix_wait', parent['branch_id'])
    hold_alias = search._next_id - 1
    adapter = PendingVerificationAdapter(search)
    before = deepcopy(search.__dict__)
    before_oracle = (oracle.points, oracle.requests)
    representative = next(n for n in adapter._frontier.values() if n['info']['actions'] == hold)
    assert representative['id'] == hold_alias and old_alias in search._frontier
    raw, associated = adapter.observation('raw'), adapter.observation('associated')
    assert raw['branch_ids'] == associated['branch_ids']
    assert all(a == b[:-8] for a, b in zip(raw['candidate_features'], associated['candidate_features']))
    # Compare the inherited formulas against the original class given exactly
    # the same nodes, without calling the prototype's disabled observation.
    legacy = BranchSearch.__new__(BranchSearch)
    legacy.__dict__ = search.__dict__.copy()
    legacy._window = adapter._window
    assert BranchSearch.observation(legacy, 'raw') == raw
    assert adapter.rule_action('beam') == BranchSearch.rule_action(legacy, 'beam')
    assert (oracle.points, oracle.requests) == before_oracle
    for key in ('_frontier', '_checked', 'verification_queue', '_beam_turn', 'graph_expansions',
                'verify_attempts', 'total_decisions', 'incumbent'):
        assert search.__dict__[key] == before[key]
    assert len({tuple(n['actions']) for n in raw['branches']}) == len(raw['branches']) <= 32
    assert not any(row[raw['candidate_feature_names'].index('already_checked')] for row in raw['candidate_features'])
    with pytest.raises(TypeError, match='Read-only'):
        adapter.step(0)


def test_coverage_parity_uses_verify_count_not_sixteen_completed_graph_expansions():
    search, oracle = make()
    controller = DeferredMacroController(search)
    for _ in range(16):
        proposal = controller.prepare()
        assert proposal['operation'] == EXPAND
        controller.apply(proposal)
    assert search._beam_turn == 16 and search.verify_attempts == 0
    first = controller.prepare()
    # Preserve a low incumbent so the next odd verification turn has a
    # genuine improving-hold family; the ordinary first choice must fail.
    oracle.fail = lambda state, seconds: True
    controller.apply(first)
    assert search._beam_turn == 16 and search.verify_attempts == 1
    if search.done:
        pytest.fail('Synthetic fixture must retain a verification opportunity')
    window = controller.adapter._window()
    eligible = [n for n in controller.adapter._frontier.values()
                if n['origin'] == 'prefix_wait'
                and n['info']['energy_kwh'] > search.incumbent['validation']['energy_kwh']]
    assert eligible
    oldest = min(eligible, key=lambda n: n['id'])
    expected = sorted((n for n in eligible if len(n['prefix']) == len(oldest['prefix'])), key=lambda n: n['id'])[:32]
    assert [n['id'] for n in window] == [n['id'] for n in expected]


@pytest.mark.parametrize('method', ['beam', 'bc_raw', 'ppo_raw'])
def test_all_methods_share_identical_pure_graph_prefix_before_any_policy_call(method):
    search, oracle = make()
    policy = None if method == 'beam' else LastSlotPolicy(search)
    controller = DeferredMacroController(search, method=method, policy=policy)
    original_points, original_requests = oracle.points, oracle.requests
    expected, expected_oracle = make()
    chosen = set()
    for _ in range(16):
        reference = fixed_expand_choice(expected, chosen)
        proposal = controller.prepare()
        assert proposal['operation'] == EXPAND and not proposal['policy_used']
        assert proposal['branch_id'] == reference['id']
        assert proposal['actions'] == list(reference['info']['actions'])
        chosen.add(reference['info']['actions'])
        expected.expand(reference['id'])
        controller.apply(proposal)
    assert search.pending() == expected.pending()
    assert (oracle.points, oracle.requests) == (original_points, original_requests)
    assert expected_oracle.points == oracle.points
    assert policy is None or policy.calls == 0
    assert controller.prepare()['operation'] == VERIFY


@pytest.mark.parametrize('schedule', ['deferred', 'interleaved'])
def test_complete_schedules_bound_each_operation_and_charge_bootstrap_separately(schedule):
    search, oracle = make()
    controller = DeferredMacroController(search, schedule=schedule)
    proposals, result = finish(controller)
    operations = [p['operation'] for p in proposals]
    assert operations.count(EXPAND) <= 16 and operations.count(VERIFY) <= 16
    assert operations.count(YIELD) <= 1 and len(operations) <= 33
    assert search.total_decisions == len(operations)
    assert search.graph_expansions == operations.count(EXPAND)
    assert search.verify_attempts == operations.count(VERIFY)
    assert result['bootstrap_cost']['actual_physical_points_after'] == 13
    assert result['actual_physical_points'] == oracle.points <= 256
    assert result['control']['bootstrap_outside_macro_limits_but_paid']
    assert not result['guard_established'] and result['complete_guard_still_required']
    if schedule == 'interleaved':
        assert [p['slot_index'] for p in proposals if p['operation'] != YIELD] == sorted(
            p['slot_index'] for p in proposals if p['operation'] != YIELD)
        assert all(p['slot_index'] % 2 == (p['operation'] == VERIFY)
                   for p in proposals if p['operation'] != YIELD)
    assert controller.prepare() is None


def test_reliable_verify_failure_repairs_without_counting_extra_macro_expansion():
    search, oracle = make()
    controller = DeferredMacroController(search, schedule='interleaved')
    controller.apply(controller.prepare())
    proposal = controller.prepare()
    assert proposal['operation'] == VERIFY
    plan = tuple(proposal['actions'])
    state = search._frontier[proposal['branch_id']]['info']['states'][1]
    oracle.fail = lambda observed, seconds: observed.command == state.command and seconds == 30.
    response = controller.apply(proposal)
    assert response['verification_status'] == 'FAIL'
    assert plan in search.failure_repairs and plan not in search.verification_queue
    assert search.graph_expansions == 1 and search.verify_attempts == 1 and search.total_decisions == 2
    assert all(tuple(row['actions']) != plan for row in controller.adapter.window_mapping())
    assert controller.prepare()['operation'] == EXPAND


def test_partial_and_unknown_are_not_repaired_or_hidden_by_handoff():
    search, oracle = make(budget=14)
    controller = DeferredMacroController(search, schedule='interleaved')
    proposals, result = finish(controller)
    assert oracle.points == 14 and not search.failure_repairs
    assert result['stop_reason'] == 'PHYSICAL_BUDGET_EXHAUSTED'
    assert result['attempts'][-1]['status'] == 'BUDGET_EXHAUSTED_PARTIAL'
    assert YIELD not in [p['operation'] for p in proposals]
    search, oracle = make()
    controller = DeferredMacroController(search, schedule='interleaved')
    controller.apply(controller.prepare())
    oracle.unknown = True
    with pytest.raises(CLPUReliabilityError):
        controller.apply(controller.prepare())
    assert controller.failed and not search.failure_repairs and not search.handed_off
    with pytest.raises(RuntimeError, match='failed'):
        controller.prepare()


def test_unavailable_slots_and_dominated_choices_do_not_fake_paid_certificates():
    search, oracle = make(max_branch_expansions=1)
    controller = DeferredMacroController(search)
    proposals, result = finish(controller)
    assert search.graph_expansions == 1
    assert len([r for r in result['control']['skipped_slots'] if r['operation'] == EXPAND]) == 15
    skips = [r for r in search.deferred_trace if r['verification_status'] == 'SKIPPED_DOMINATED']
    assert skips and all(r['new_actual_physical_points'] == 0 for r in skips)
    assert len([p for p in proposals if p['operation'] == VERIFY]) == search.verify_attempts
    assert not any(plan in search._checked for plan in search.skipped_dominated_plans)


def test_prepare_is_idempotent_and_rejects_illegal_mapping_or_out_of_band_changes():
    search, oracle = make()
    controller = DeferredMacroController(search)
    proposal = controller.prepare()
    assert controller.prepare() == proposal
    bad = deepcopy(proposal); bad['branch_id'] += 1
    with pytest.raises(CLPUReliabilityError, match='Changed or stale'):
        controller.apply(bad)
    assert search.total_decisions == 0
    search.expand(proposal['branch_id'])
    with pytest.raises(CLPUReliabilityError, match='changed after'):
        controller.prepare()
    search, _ = make()
    policy = LastSlotPolicy(search)
    policy.architecture['candidate_dim'] += 1
    with pytest.raises(ValueError, match='architecture'):
        DeferredMacroController(search, method='ppo_raw', policy=policy)


@pytest.mark.parametrize('initial_failure', [False, True])
def test_bootstrap_stop_direct_result_has_reason_without_macro_or_actor_calls(initial_failure):
    env = environment(horizon=6)
    budget = 256 if initial_failure else 2
    oracle = ActualCountedOracle(env, budget=budget,
                                fail=(lambda state, seconds: True) if initial_failure else None)
    search = DeferredCertificationSearch(env, oracle, BranchConfig(point_budget=budget))
    assert search.done
    expected = 'INITIAL_FAIL' if initial_failure else 'PHYSICAL_BUDGET_EXHAUSTED'
    assert search.stop_reason == expected
    policy = LastSlotPolicy(search)
    controller = DeferredMacroController(search, method='ppo_raw', policy=policy)
    before = (oracle.requests, oracle.points)
    # The real runner does not call prepare when bootstrap already stopped.
    result = controller.result()
    assert result['control']['finish_reason'] == result['stop_reason'] == expected
    assert result['control']['records'] == result['control']['skipped_slots'] == []
    assert result['control']['schedule_slots_consumed'] == 0
    assert search.total_decisions == search.graph_expansions == search.verify_attempts == 0
    assert policy.calls == result['policy_decisions'] == 0 and not search.handed_off
    assert controller.prepare() is None and (oracle.requests, oracle.points) == before


def model_environment():
    original = environment(horizon=30)
    loads = deepcopy(original.loads)
    for i in range(57):
        loads[f'load.extra{i:02d}'] = {'bus': 's', 'nominal_kw': 1., 'nominal_kvar': .5}
    condition = {**original.condition, 'denominator_nominal_kw': sum(v['nominal_kw'] for v in loads.values())}
    return CLPUEnvironment(JointGraph(original.static, loads, original.scenario), original.static,
        loads, original.scenario, condition, original.event, horizon=30, slot_seconds=60, sample_seconds=30)


def test_all_six_bound_real_raw_checkpoints_accept_pending_observation_and_legal_mask():
    # Read fixed training-population checkpoints only; no new model is trained.
    import torch
    from dnr.nature34_clpu_branch_learning import encode_observations, load_policy
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        search, oracle = make(env=model_environment())
        chosen = set()
        for _ in range(16):
            node = fixed_expand_choice(search, chosen)
            chosen.add(node['info']['actions'])
            search.expand(node['id'])
        adapter = PendingVerificationAdapter(search)
        obs = adapter.observation('raw')
        assert len(obs['global_features']) == 79 and len(obs['candidate_features'][0]) == 261
        tensors = encode_observations([obs], 'raw')
        assert tuple(tensors['candidates'].shape) == (1, 32, 261)
        assert int(tensors['mask'].sum()) == len(obs['mask'])
        before = (oracle.points, oracle.requests)
        bindings = json.loads((MODEL_SOURCE / 'model_bindings.json').read_text(encoding='utf-8'))
        assert {(r['method'], r['seed']) for r in bindings} == {
            (method, seed) for method in ('bc_raw', 'ppo_raw') for seed in (17, 41, 73)}
        for binding in bindings:
            path = MODEL_SOURCE / binding['checkpoint']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == binding['source_sha256']
            policy = load_policy(path, 'raw')
            adapter.validate_policy(policy)
            choice = policy.act(obs, obs['mask'], deterministic=True)
            assert type(choice['action']) is int and obs['mask'][choice['action']]
            assert obs['branch_ids'][choice['action']] in search._frontier
        assert (oracle.points, oracle.requests) == before
    finally:
        torch.set_num_threads(previous_threads)
