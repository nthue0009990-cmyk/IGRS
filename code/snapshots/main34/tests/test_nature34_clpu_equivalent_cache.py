"""Exact-input reuse and causal dual ledgers with a mocked physical delegate."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dnr import nature34_clpu_equivalent_cache as cache
from dnr import nature34_clpu_physical as physical
from dnr.nature34_clpu_branch import BranchConfig
from dnr.nature34_clpu_environment import CLPUBudgetExhausted, CLPUReliabilityError
from dnr.nature34_clpu_window import WindowBranchSearch
from test_nature34_clpu_branch import environment


def lines(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def known_result(raw, metadata, **kwargs):
    result = {key: {'status': 'PASS', 'violations': [], 'unknown': []} for key in
        ('numerical', 'topology', 'voltage', 'line_thermal', 'transformer_thermal', 'gfm_active_limit')}
    result['known_constraints_satisfied'] = not raw['fail']
    if raw['fail']:
        result['voltage'] = {'status': 'FAIL', 'unknown': [], 'violations': [
            {'node': '812.1', 'value': .88, 'lower': .9, 'upper': 1.1}]}
    return result


class FakeRuntime:
    def __init__(self, *, failed=False, unknown=False):
        self.failed, self.unknown, self.rows = failed, unknown, []
        self.initial_metadata = {}
        self.physical_config = {'constraint_tolerances': {}, 'ac_audit_tolerances': {}}

    def evaluate_point(self, action, case, alpha, algorithm, condition, multipliers,
                       *, time_minutes, state_identity):
        row = {'raw_outputs': {'converged': not self.unknown, 'elements': [], 'fail': self.failed},
            'metadata_override': {'planned_enabled': {}, 'deenergized_buses': []},
            'time_minutes': time_minutes, 'state_identity': deepcopy(state_identity),
            'action': action, 'clpu_multipliers': multipliers}
        physical.certify_clpu_point(row, self.initial_metadata, self.physical_config)
        self.rows.append(row)
        return row


class FakeDelegate:
    def __init__(self, parent, cell, env, budget, *, failed=False, unknown=False):
        self.parent, self.cell, self.env, self.budget = parent, cell, env, budget
        self.points = self.requests = 0
        self.cache = {}
        self.runtime = FakeRuntime(failed=failed, unknown=unknown)
        self.factory = SimpleNamespace(stats=lambda: {'returned': {
            'numerical_solution_entries': 3 + 4 * self.points}})
        self.closed = False

    def __enter__(self):
        self.cell.mkdir(parents=True, exist_ok=True)
        self.raw = gzip.open(self.cell / 'points.jsonl.gz', 'wt', encoding='utf-8')
        self.request_log = (self.cell / 'requests.jsonl').open('w', encoding='utf-8')
        return self

    def certify(self, state, seconds):
        identity = {**self.env.identity(state, seconds), 'alpha': .5,
                    'base_condition_sha256': cache.file_sha(self.parent / 'condition.json')}
        key = cache.sha_value(identity)
        assert key not in self.cache
        self.requests += 1
        row = self.runtime.evaluate_point(self.env.graph.action(state.command),
            {'name': self.env.scenario['scenario_id'], 'fault_lines': self.env.scenario['fault_lines']},
            .5, 0, self.env.condition, self.env.multipliers(state, seconds),
            time_minutes=seconds / 60., state_identity=identity)
        row['point_id'] = self.points
        row['identity_sha256'] = key
        row['actual_call_counts'] = {'numerical_solution_entries': 4}
        cache._append(self.raw, row)
        cache._append(self.request_log, {'point_id': self.points, 'key': key})
        self.points += 1
        cert = {'point_id': row['point_id'], 'status': row['status'], 'reliable': row['reliable'],
                'feedback': row['structured_feedback']['violations']}
        if not cert['reliable']:
            raise CLPUReliabilityError('Synthetic retained unreliable actual point')
        self.cache[key] = cert
        return cert

    def __exit__(self, typ, value, tb):
        self.raw.close()
        self.request_log.close()
        self.closed = True
        save(self.cell / 'oracle_summary.json', {'points': self.points,
            'actual_numerical_entries': 3 + 4 * self.points, 'initialization_numerical_entries': 3})


@pytest.fixture
def oracle_factory(tmp_path, monkeypatch):
    monkeypatch.setattr(physical, 'check_known_constraints', known_result)
    monkeypatch.setattr(physical, 'audit_ac_snapshot', lambda *args, **kwargs: {'status': 'PASS'})
    monkeypatch.setattr(physical, 'verify_isolation', lambda *args, **kwargs: {'status': 'PASS'})
    delegates = []

    def build(variant='electrical_equivalent', *, budget=256, failed=False, unknown=False, env=None):
        env = env or environment(horizon=4)
        parent = tmp_path / f'parent{len(delegates)}'
        cell = parent / 'cell'
        save(parent / 'condition.json', env.condition)

        def delegate(parent, cell, env, budget):
            item = FakeDelegate(parent, cell, env, budget, failed=failed, unknown=unknown)
            delegates.append(item)
            return item

        monkeypatch.setattr(cache, '_make_delegate', delegate)
        return cache.EquivalenceOracle(parent, cell, env, budget, variant, {'frozen_input_sha256': 'fixture'})

    return build, delegates


def test_electrical_key_excludes_time_and_planning_index_but_binds_all_actual_inputs():
    env = environment(horizon=4)
    steady, later = env.initial, env.advance(env.initial, 0)
    scope = {'physics': 'fixed', 'dependencies': {'engine': 'fixed'}}
    left = cache.electrical_input_key(env, steady, 0, scope)
    right = cache.electrical_input_key(env, later, 60, scope)
    assert left == right
    assert env.identity(steady, 0) != env.identity(later, 60)
    assert set(left) == {'version', 'scope_sha256', 'action', 'case', 'alpha', 'algorithm', 'condition', 'multipliers'}
    assert len(left['action']) == 19 and left['condition'] == env.condition
    assert set(left['multipliers']) == set(env.load_names)


@pytest.mark.parametrize('changed', ['action', 'fault', 'condition', 'multiplier', 'scope'])
def test_different_exact_solver_inputs_never_share_a_key(changed):
    env = environment(horizon=4)
    state, scope, seconds = env.initial, {'physics': 'fixed'}, 0.
    original = cache.electrical_input_key(env, state, seconds, scope)
    if changed == 'action':
        state = env.advance(state, 16)
    elif changed == 'fault':
        env.scenario['fault_lines'] = ['line.l1']
    elif changed == 'condition':
        env.condition['load_multiplier'] = 1.000000000000001
    elif changed == 'multiplier':
        state = env.advance(state, 16)
        original = cache.electrical_input_key(env, state, 30., scope)
        seconds = 30.0000000001
    else:
        scope['physics'] = 'different'
    assert cache.electrical_input_key(env, state, seconds, scope) != original


def test_different_pickup_histories_only_merge_when_full_electrical_inputs_are_exactly_equal():
    env = environment(horizon=4)
    early = env.advance(env.initial, 16)
    late = env.advance(env.advance(env.initial, 0), 16)
    assert early.command == late.command and early.pickups != late.pickups
    scope = {'frozen': True}
    assert cache.electrical_input_key(env, early, 120, scope) != cache.electrical_input_key(env, late, 120, scope)
    env.event['a'] = 0.
    assert cache.electrical_input_key(env, early, 120, scope) == cache.electrical_input_key(env, late, 120, scope)
    assert env.identity(early, 120) != env.identity(late, 120)


@pytest.mark.parametrize('variant,actual,reuses', [('semantic_identity', 3, 0), ('electrical_equivalent', 1, 2)])
def test_dual_ledgers_preserve_semantic_ids_and_charge_only_actual_points(oracle_factory, variant, actual, reuses):
    build, delegates = oracle_factory
    oracle = build(variant)
    with oracle:
        state = oracle.env.initial
        certs = [oracle.certify(state, t) for t in (0, 30, 60)]
        assert [c['point_id'] for c in certs] == [0, 1, 2]
        assert oracle.certify(state, 30) == certs[1]
        assert oracle.points == 3 and oracle.physical_points == actual
        assert oracle.requests == 4 and oracle.hits == 1 and oracle.electrical_reuses == reuses
    logical, physical_rows = lines(oracle.cell / 'logical_points.jsonl.gz'), lines(oracle.cell / 'points.jsonl.gz')
    assert len(logical) == 3 and len(physical_rows) == actual
    assert all('actual_call_counts' not in row for row in logical)
    assert all(row['actual_call_counts']['numerical_solution_entries'] == 4 for row in physical_rows)
    actual_summary = json.loads((oracle.cell / 'oracle_summary.json').read_text())
    assert actual_summary['points'] == actual
    assert oracle.summary()['actual_numerical_entries'] == 3 + 4 * actual
    assert oracle.summary()['points'] == oracle.summary()['logical_points'] == 3
    assert delegates[0].closed


def test_alias_recertifies_fail_feedback_at_current_time_without_mutating_raw(oracle_factory):
    build, delegates = oracle_factory
    oracle = build(failed=True)
    with oracle:
        first = oracle.certify(oracle.env.initial, 0)
        original = deepcopy(delegates[0].runtime.rows[0])
        second = oracle.certify(oracle.env.initial, 60)
        assert first['status'] == second['status'] == 'FAIL'
        assert first['feedback'][0]['time_minutes'] == 0
        assert second['feedback'][0]['time_minutes'] == 1
        assert first['feedback'][0]['magnitude'] == second['feedback'][0]['magnitude']
        assert delegates[0].runtime.rows[0] == original
        assert oracle.physical_points == 1
    logical = lines(oracle.cell / 'logical_points.jsonl.gz')
    assert logical[0]['physical_evidence_sha256'] == logical[1]['physical_evidence_sha256']
    assert logical[1]['physical_point_id'] == 0 and logical[1]['reuse_kind'] == 'ELECTRICAL_EQUIVALENCE'


def test_semantic_budget_is_not_replaced_by_cheaper_actual_budget(oracle_factory):
    build, _ = oracle_factory
    oracle = build(budget=2)
    with oracle:
        oracle.certify(oracle.env.initial, 0)
        oracle.certify(oracle.env.initial, 30)
        with pytest.raises(CLPUBudgetExhausted):
            oracle.certify(oracle.env.initial, 60)
        assert oracle.certify(oracle.env.initial, 0)['point_id'] == 0
        assert oracle.points == 2 and oracle.physical_points == 1
    requests = lines(oracle.cell / 'logical_requests.jsonl')
    assert [r['request'] for r in requests] == [1, 2, 3, 4]
    assert requests[2]['status'] == 'BUDGET_UNKNOWN' and 'logical_point_id' not in requests[2]
    assert requests[3]['semantic_cache_hit'] and not requests[3]['electrical_reuse']


def test_private_caches_never_borrow_previous_arm_or_sibling_answers(oracle_factory):
    build, delegates = oracle_factory
    first = build()
    with first:
        first.certify(first.env.initial, 0)
        first.certify(first.env.initial, 30)
    second = build()
    with second:
        second.certify(second.env.initial, 30)
        assert second.points == second.physical_points == 1 and second.electrical_reuses == 0
    assert len(delegates) == 2 and all(d.closed for d in delegates)


def test_unreliable_actual_point_is_retained_but_never_aliased(oracle_factory):
    build, delegates = oracle_factory
    oracle = build(unknown=True)
    with pytest.raises(CLPUReliabilityError, match='unreliable'):
        with oracle:
            oracle.certify(oracle.env.initial, 0)
    assert oracle.points == 0 and oracle.physical_points == 1
    assert oracle.cache == {} and oracle._electrical_cache == {}
    assert len(lines(oracle.cell / 'points.jsonl.gz')) == 1
    assert lines(oracle.cell / 'logical_points.jsonl.gz') == []
    assert lines(oracle.cell / 'logical_requests.jsonl')[0]['status'] == 'ERROR'
    assert oracle.summary()['status'] == 'STOPPED' and delegates[0].closed


def test_alias_cannot_turn_unreliable_recheck_into_success(oracle_factory, monkeypatch):
    build, _ = oracle_factory
    oracle = build()
    with pytest.raises(CLPUReliabilityError, match='reliable'):
        with oracle:
            oracle.certify(oracle.env.initial, 0)
            monkeypatch.setattr(physical, 'audit_ac_snapshot', lambda *args, **kwargs: {'status': 'FAIL'})
            oracle.certify(oracle.env.initial, 30)
    assert oracle.points == 1 and oracle.physical_points == 1 and oracle.electrical_reuses == 0
    assert len(lines(oracle.cell / 'logical_points.jsonl.gz')) == 1


def test_returned_certificate_mutation_does_not_modify_private_evidence(oracle_factory):
    build, _ = oracle_factory
    oracle = build(failed=True)
    with oracle:
        result = oracle.certify(oracle.env.initial, 0)
        result['status'] = 'PASS'
        result['feedback'].clear()
        again = oracle.certify(oracle.env.initial, 0)
        assert again['status'] == 'FAIL' and again['feedback']


def test_frozen_scope_mutation_is_rejected_before_a_new_request(oracle_factory):
    build, _ = oracle_factory
    oracle = build()
    with oracle:
        oracle.scope['tampered'] = True
        with pytest.raises(CLPUReliabilityError, match='scope changed'):
            oracle.certify(oracle.env.initial, 0)
        assert oracle.requests == oracle.physical_points == 0


def test_deadline_stops_before_starting_another_physical_point(oracle_factory):
    build, _ = oracle_factory
    oracle = build()
    with oracle:
        oracle.deadline_monotonic = -1
        with pytest.raises(TimeoutError):
            oracle.certify(oracle.env.initial, 0)
        assert oracle.points == oracle.physical_points == 0


def test_search_has_identical_logical_path_and_complete_certification_in_both_arms(oracle_factory):
    build, _ = oracle_factory
    results, counts = [], []
    for variant in cache.VARIANTS:
        oracle = build(variant)
        with oracle:
            search = WindowBranchSearch(oracle.env, oracle, BranchConfig())
            while not search.done:
                search.step(search.rule_action('beam'))
            results.append(search.result())
            counts.append(oracle.physical_points)
    assert results[0] == results[1]
    assert results[0]['best_plan']['certified_complete']
    assert counts[1] < counts[0]


def test_each_logical_alias_links_only_to_an_already_paid_matching_actual_point(oracle_factory):
    build, _ = oracle_factory
    oracle = build()
    with oracle:
        for t in (0, 30, 60):
            oracle.certify(oracle.env.initial, t)
    logical, actual = lines(oracle.cell / 'logical_points.jsonl.gz'), lines(oracle.cell / 'points.jsonl.gz')
    seen = {}
    for row in logical:
        assert row['semantic_identity_sha256'] == cache.sha_value(row['semantic_identity'])
        assert row['electrical_input_sha256'] == cache.sha_value(row['electrical_input_key'])
        pid = row['physical_point_id']
        assert row['physical_evidence_sha256'] == cache.sha_value(cache.physical_evidence_payload(actual[pid]))
        if row['reuse_kind'] == 'ACTUAL_PHYSICAL':
            seen.setdefault(row['electrical_input_sha256'], pid)
        else:
            assert seen[row['electrical_input_sha256']] == pid


@pytest.mark.parametrize('variant,budget', [('invented', 256), ('semantic_identity', 0), ('electrical_equivalent', True)])
def test_undeclared_modes_and_invalid_budgets_are_rejected(tmp_path, variant, budget):
    with pytest.raises(ValueError):
        cache.EquivalenceOracle(tmp_path, tmp_path / 'cell', environment(), budget, variant, {'frozen': True})
