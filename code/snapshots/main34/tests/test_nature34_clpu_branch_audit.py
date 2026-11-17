"""Mutation checks for independent branch replay. Synthetic certificates only."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_nature34_clpu_branch as audit
from dnr.nature34_clpu_branch import BranchConfig, BranchSearch
from dnr.nature34_clpu_environment import CLPUEnvironment, CLPUBudgetExhausted
from dnr.nature34_joint_sequence import JointGraph
from nature34_clpu_branch_execution import collect_episode


class LoggedFakeOracle:
    def __init__(self, env, budget, fail=False):
        self.env, self.budget, self.fail = env, budget, fail
        self.points = self.requests = 0
        self.cache, self.raw, self.rows = {}, {}, []

    def certify(self, state, seconds):
        self.requests += 1
        identity = {**self.env.identity(state, seconds), 'alpha': .5, 'base_condition_sha256': 'synthetic'}
        key = hashlib.sha256(audit.canonical(identity).encode()).hexdigest()
        if key in self.cache:
            cert = self.cache[key]
            self.rows.append({'request': self.requests, 'key': key, 'cache_hit': True, 'point_id': cert['point_id']})
            return deepcopy(cert)
        if self.points == self.budget:
            self.rows.append({'request': self.requests, 'key': key, 'cache_hit': False, 'status': 'BUDGET_UNKNOWN'})
            raise CLPUBudgetExhausted('Synthetic budget')
        pid = self.points
        self.points += 1
        row = {'request': self.requests, 'key': key, 'cache_hit': False, 'point_id': pid, 'status': 'STARTED'}
        self.rows.append(deepcopy(row))
        cold = [p for p in state.pickups if p is not None and p >= 0]
        failed = self.fail and bool(cold) and seconds == min(cold)
        cert = {'point_id': pid, 'status': 'FAIL' if failed else 'PASS', 'reliable': True,
                'feedback': [{'constraint': 'voltage', 'bus': '6', 'phase': 1, 'magnitude': .02}] if failed else []}
        row['status'] = cert['status']
        self.rows.append(row)
        self.cache[key] = cert
        self.raw[pid] = {'status': cert['status'], 'reliable': True, 'clpu_multipliers': identity['multipliers'],
                         'state_identity': identity, 'action': self.env.graph.action(state.command),
                         'time_minutes': seconds / 60., 'structured_feedback': {'violations': cert['feedback']}}
        return deepcopy(cert)


@pytest.fixture(scope='module')
def inputs():
    edges = [['s', str(i)] for i in range(1, 10)]
    loads = {f'load.l{i}': {'bus': str(i), 'nominal_kw': float(i), 'nominal_kvar': i / 2.}
             for i in range(1, 10)}
    loads['load.l10'] = {'bus': 's', 'nominal_kw': 1., 'nominal_kvar': .5}
    for i in range(58):
        loads[f'load.fixed{i}'] = {'bus': '6' if i == 0 else 's', 'nominal_kw': 7. if i == 0 else 0., 'nominal_kvar': 0.}
    static = {'nodes': ['s'] + [str(i) for i in range(1, 10)], 'source_bus': 's', 'bus_edges': edges,
              'switch_edges': edges, 'original_lines': {f'line.l{i}': e for i, e in enumerate(edges, 1)},
              'dispatch_loads': [f'load.l{i}' for i in range(1, 11)],
              'sectional_switches': [{'line': f'l{i}'} for i in range(1, 6)]}
    scenario = {'scenario_id': 'synthetic_branch_audit', 'partition': 'build', 'fault_lines': ['line.l8', 'line.l9']}
    condition = {'condition_id': 'synthetic', 'load_multiplier': 1.}
    event = {'event_id': 1, 'a': .5, 'tau_seconds': 120.}
    branch_cfg = asdict(BranchConfig(lookahead_depth=2, max_generated_candidates=60, max_checked_candidates=20,
                                    max_branch_expansions=20, point_budget=45, action_window=8, beam_width=4))
    cfg = {'horizon': 3, 'slot_seconds': 60, 'sample_seconds': 30, 'branch_search': branch_cfg}
    mech = audit.Mechanics(static, loads, scenario, condition, event, cfg)
    env = CLPUEnvironment(JointGraph(static, loads, scenario), static, loads, scenario, condition, event, horizon=3)
    return mech, env, branch_cfg


def make_saved(inputs, method='greedy', fail=False, budget=None):
    mech, env, cfg = inputs
    cfg = deepcopy(cfg)
    if budget is not None:
        cfg['point_budget'] = budget
    oracle = LoggedFakeOracle(env, cfg['point_budget'], fail)
    search = BranchSearch(env, oracle, BranchConfig(**cfg))
    transitions, decisions = [], []
    spec = {'method': method, 'representation': 'raw', 'seed': 17, 'deterministic': True}
    count = collect_episode(search, spec, transitions.append, decisions.append)
    result = search.result()
    result.update(spec=spec, search_decisions=count)
    return {'mech': mech, 'branch_cfg': cfg, 'result': result, 'transitions': transitions,
            'decisions': decisions, 'request_rows': oracle.rows, 'points': oracle.raw, 'condition_sha': 'synthetic'}


@pytest.mark.parametrize('method,fail', [('greedy', False), ('beam', False), ('greedy', True), ('beam', True)])
def test_independent_queue_graph_history_and_paid_requests_match(inputs, method, fail):
    saved = make_saved(inputs, method, fail)
    result = audit.check_branch_records(**saved)
    assert result['decisions'] == saved['result']['search_decisions']


def test_unobserved_physical_feedback_in_before_observation_is_rejected(inputs):
    saved = make_saved(inputs, fail=True)
    saved['transitions'][0]['obs_raw']['observed_feedback'] = [{'constraint': 'voltage', 'bus': '6', 'magnitude': .1}]
    with pytest.raises(AssertionError, match='observation_raw'):
        audit.check_branch_records(**saved)


def test_wrong_short_horizon_upper_bound_stop_is_rejected(inputs):
    saved = make_saved(inputs)
    saved['result']['stop_reason'] = 'CERTIFIED_FRONTIER_UPPER_BOUND'
    saved['result']['optimality_established'] = True
    with pytest.raises(AssertionError, match='upper-bound stop'):
        audit.check_branch_records(**saved)


def test_changed_generation_history_cannot_pass_from_matching_hashes(inputs):
    saved = make_saved(inputs)
    saved['result']['generation_trace'][0]['prefix'] = [19]
    with pytest.raises(AssertionError, match='generation_trace'):
        audit.check_branch_records(**saved)


def test_complete_partial_restore_requires_wait_tail_samples(inputs):
    saved = make_saved(inputs)
    saved['result']['attempts'][0]['validation']['samples'].pop()
    with pytest.raises(AssertionError, match='tail'):
        audit.check_branch_records(**saved)


def test_budget_truncated_bootstrap_has_no_incumbent(inputs):
    saved = make_saved(inputs, budget=2)
    checked = audit.check_branch_records(**saved)
    assert checked['decisions'] == 0
    assert saved['result']['incumbent'] is None
    assert checked['stop_reason'] == 'PHYSICAL_BUDGET_EXHAUSTED'


def test_cold_cache_cannot_reference_an_unpaid_point(inputs):
    saved = make_saved(inputs)
    saved['request_rows'][0] = {'request': 1, 'key': saved['request_rows'][0]['key'], 'cache_hit': True, 'point_id': 0}
    with pytest.raises(AssertionError, match='Unpaid'):
        audit.check_branch_records(**saved)


def test_reward_mutation_cannot_hide_cost_or_invent_quality_gain(inputs):
    saved = make_saved(inputs)
    saved['transitions'][0]['reward'] += .01
    with pytest.raises(AssertionError, match='Reward'):
        audit.check_branch_records(**saved)


def test_future_request_must_not_precede_recorded_decision(inputs):
    saved = make_saved(inputs)
    saved['transitions'][0]['oracle_before']['points'] += 1
    with pytest.raises(AssertionError, match='query before decision'):
        audit.check_branch_records(**saved)

