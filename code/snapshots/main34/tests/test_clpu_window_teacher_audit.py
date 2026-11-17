"""Independent teacher replay and tamper checks with synthetic physical answers."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_clpu_window_teacher as audit
from dnr.nature34_clpu_branch import BranchConfig
from dnr.nature34_clpu_window import WindowBranchSearch
from dnr.nature34_clpu_window_counterfactual import restore_snapshot, fork_snapshot
from test_nature34_clpu_branch_audit import inputs, LoggedFakeOracle


class LiveFake(LoggedFakeOracle):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def saved(inputs, position=0, allowance=128, fail=True):
    mech, env, cfg = inputs
    cfg = {**cfg, 'point_budget': 256}
    source = LoggedFakeOracle(env, 256, fail)
    search = WindowBranchSearch(env, source, BranchConfig(**cfg))
    steps = []
    while not search.done:
        action = search.rule_action('beam')
        start = {'transition_id': search._beam_turn, 'status': 'STARTED', 'method': 'beam',
            'action': action, 'branch_id': search._window()[action]['id'],
            'window_branch_ids': [r['id'] for r in search._window()],
            'oracle_before': {'requests': source.requests, 'points': source.points}}
        steps.append(start)
        response = search.step(action)
        steps.append({**start, 'status': 'COMPLETED', 'response': response,
                      'oracle_after': {'requests': source.requests, 'points': source.points}})
    count = steps[2 * position]['oracle_before']['points']
    requests = steps[2 * position]['oracle_before']['requests']
    prefix = {i: r for i, r in source.raw.items() if i < count}
    compact = [{'identity': r['state_identity'], 'identity_sha256': audit.digest(r['state_identity']),
        'certificate': {'point_id': i, 'status': r['status'], 'reliable': r['reliable'],
                        'feedback': r['structured_feedback']['violations']},
        'source_row_sha256': 'synthetic'} for i, r in prefix.items()]
    source_result = {'config': cfg, 'spec': {'method': 'beam', 'window_variant': 'coverage_v1'}}
    snapshot = restore_snapshot(env, source_result, steps, search.attempts, compact, position,
                                {'alpha': .5, 'base_condition_sha256': 'synthetic'})
    forced = snapshot.search._window()[snapshot.search.rule_action('beam')]['id']
    opened = []

    def factory(*, budget):
        live = LiveFake(env, budget, fail)
        opened.append(live)
        return live

    result = fork_snapshot(snapshot, forced, factory, max_new_points=allowance, max_expansions=16)
    live = opened[0] if opened else None
    return {'mech': mech, 'cfg': cfg, 'payload': snapshot.to_dict()['payload'], 'result': result,
        'source_steps': steps, 'source_plans': search.attempts, 'prefix_points': prefix,
        'source_requests': [r for r in source.rows if r['request'] <= requests],
        'new_points': live.raw if live else {}, 'new_requests': live.rows if live else [],
        'condition_sha': 'synthetic', 'forced_branch_id': forced, 'max_new_points': allowance,
        'max_expansions': 16, 'plan_journal': deepcopy(result['new_attempts']),
        'step_journal': deepcopy(result['transitions'])}


@pytest.mark.parametrize('position,allowance,fail', [(0, 128, False), (0, 1, False), (0, 0, True),
    (0, 128, True), (2, 128, True), (4, 2, True), (8, 128, True)])
def test_independent_arbitrary_snapshot_and_complete_curve_replay(inputs, position, allowance, fail):
    data = saved(inputs, position, allowance, fail)
    checked = audit.check_records(**data)
    assert checked['status'] == 'PASS'
    assert checked['auc_gain_pp'] == pytest.approx(data['result']['auc_gain_pp']) if allowance else checked['auc_gain_pp'] is None


@pytest.mark.parametrize('mutation,message', [
    (lambda d: d['payload']['source_attempts'].append(deepcopy(d['source_plans'][-1])), 'future cache-only'),
    (lambda d: d['payload']['source_trace'][0].__setitem__('quality_after', .999), 'history differs'),
    (lambda d: d['payload']['raw_observation']['global_features'].__setitem__(0, .999), 'snapshot_observation'),
    (lambda d: d['payload']['prefix_oracle_counters'].__setitem__('requests', 0), 'accounting'),
    (lambda d: d['result'].__setitem__('auc_gain_pp', 50.), 'independent_auc'),
    (lambda d: d['result']['oracle'].__setitem__('new_physical_points', 0), 'ledger'),
    (lambda d: d['result']['transitions'][0]['obs_raw']['branch_ids'].reverse(), 'journal'),
    (lambda d: d['result']['fork_trace'].pop(), 'fork_trace'),
    (lambda d: d['result'].__setitem__('forced_branch_id', -1), 'Forced candidate'),
])
def test_modified_prefix_choice_cost_or_favorable_label_rejected(inputs, mutation, message):
    data = saved(inputs, position=2)
    mutation(data)
    with pytest.raises(AssertionError, match=message):
        audit.check_records(**data)


def test_partial_plan_cannot_gain_quality_at_an_intermediate_pass_point(inputs):
    data = saved(inputs, allowance=1, fail=False)
    assert data['result']['new_attempts'][-1]['status'] == 'BUDGET_EXHAUSTED_PARTIAL'
    assert data['result']['certificate_updates'] == []
    data['result']['quality_point_trajectory'].append(
        {'new_physical_points': 1, 'quality': .99, 'kind': 'COMPLETE_CERTIFICATE'})
    with pytest.raises(AssertionError, match='quality_point_trajectory'):
        audit.check_records(**data)


@pytest.mark.parametrize('field', ['new_physical_points', 'quality_before', 'quality_after', 'expansion_index'])
def test_complete_certificate_boundary_cannot_be_moved_or_improved(inputs, field):
    data = saved(inputs, fail=False)
    data['result']['certificate_updates'][0][field] += 1
    with pytest.raises(AssertionError, match='complete_certificate_updates'):
        audit.check_records(**data)


@pytest.mark.parametrize('rows,budget,expected', [([], 4, 0), ([(0, .6)], 4, 40),
    ([(4, .6)], 4, 10), ([(2, .4), (2, .6)], 4, 30), ([(2, .4), (4, .6)], 4, 20),
    ([(1, .6)], 4, 40), ([], 0, None)])
def test_dense_budget_grid_credits_exact_b0_and_endpoint_semantics(rows, budget, expected):
    curve = [{'new_physical_points': 0, 'quality': .2, 'kind': 'INITIAL_CERTIFIED_INCUMBENT'}]
    curve += [{'new_physical_points': p, 'quality': q, 'kind': 'COMPLETE_CERTIFICATE'} for p, q in rows]
    result = audit.independent_auc(.2, curve, budget)
    assert result is None if expected is None else result == pytest.approx(expected)


def test_budget_grid_rejects_uncertified_or_nonmonotone_boundaries():
    curve = [{'new_physical_points': 0, 'quality': .2, 'kind': 'INITIAL_CERTIFIED_INCUMBENT'},
             {'new_physical_points': 1, 'quality': .5, 'kind': 'PARTIAL_PASS'}]
    with pytest.raises(AssertionError, match='uncertified'):
        audit.independent_auc(.2, curve, 4)


def test_snapshot_cannot_consume_future_source_requests(inputs):
    data = saved(inputs, position=0)
    data['source_requests'].append(deepcopy(data['source_requests'][-1]))
    with pytest.raises(AssertionError, match='unconsumed'):
        audit.check_records(**data)


def test_first_eligible_state_comes_from_preaction_queue_and_replays_every_source_step(inputs):
    from test_clpu_window_audit import saved as window_saved
    data = window_saved(inputs, budget=256)
    data['result']['spec'].update(scenario_id='synthetic_fault', event_id=3)
    source = {'requests': data['request_rows'], 'points': data['points'], 'plans': data['attempts'],
              'steps': data['steps'], 'result': data['result']}
    states = audit.first_eligible_states(data['mech'], data['branch_cfg'], source, 'synthetic')
    assert states['normal']['completed_steps'] == 0
    assert states['normal']['eligible']
    assert any('window_beam' in row['roles'] for row in states['normal']['candidates'])
    source['steps'][0]['window_branch_ids'] = []
    with pytest.raises(AssertionError, match='boundary'):
        audit.first_eligible_states(data['mech'], data['branch_cfg'], source, 'synthetic')


def test_candidate_subset_preserves_beam_id_when_greedy_is_duplicate_full_plan():
    class Replay:
        @staticmethod
        def window():
            return [{'id': 9, 'info': {'actions': (1, 0), 'priority': (-5, 0)}},
                    {'id': 3, 'info': {'actions': (1, 0), 'priority': (-5, 0)}},
                    {'id': 4, 'info': {'actions': (2, 0), 'priority': (-4, 0)}},
                    {'id': 5, 'info': {'actions': (3, 0), 'priority': (-3, 0)}}]

        @staticmethod
        def rule(method):
            assert method == 'beam'
            return 0

    state = {'candidate_ids': [3, 4, 5], 'mode': 'normal', 'completed_steps': 2}
    selected = audit.candidate_subset(Replay(), state, 'fault', 3)
    assert len(selected) == 3
    assert selected[0]['branch_id'] == 9
    assert selected[0]['roles'][:2] == ['window_beam', 'graph_greedy']
    assert 3 not in [row['branch_id'] for row in selected]


def controller_data():
    schedule = [{'cell': 'episodes/000'}, {'cell': 'episodes/001'}]
    checks = [{'new_physical_points': 2, 'new_numerical_entries': 11} for _ in schedule]
    controller = {'status': 'COMPLETE_CELLS_PENDING_AUDIT', 'errors': [], 'unknown_cost_cells': [],
        'scheduled': 2, 'completed': [r['cell'] for r in schedule],
        'exits': [{'cell': r['cell'], 'exit_code': 0} for r in schedule],
        'known_physical_points': 4, 'known_numerical_entries': 22, 'wall_seconds': 1.}
    return controller, schedule, checks


def test_controller_checks_actual_exits_and_full_initialization_cost():
    args = controller_data()
    assert audit.check_controller(*args) == (4, 22)
    args[0]['known_numerical_entries'] = 16
    with pytest.raises(AssertionError, match='cost differs'):
        audit.check_controller(*args)
    args = controller_data()
    args[0]['exits'][1]['exit_code'] = 1
    with pytest.raises(AssertionError, match='worker exit'):
        audit.check_controller(*args)


def historical_source_fixture(tmp_path, *, duplicate_audit=False, wrong_cell_file=False):
    source = tmp_path / 'historical'
    cell = source / 'episodes/000'
    cell.mkdir(parents=True)
    (source / 'audit').mkdir()
    spec = {'method': 'beam', 'window_variant': 'coverage_v1', 'context_id': 0}
    (cell / 'result.json').write_text(json.dumps({'spec': spec}), encoding='utf-8')
    for name in ('points.jsonl.gz', 'steps.jsonl.gz', 'plans.jsonl.gz', 'requests.jsonl'):
        (cell / name).write_bytes(('synthetic-' + name).encode())
    files = {path.name: audit.sha(path) for path in cell.iterdir()}
    cell_files = dict(files)
    if wrong_cell_file:
        cell_files['requests.jsonl'] = '0' * 64
    (cell / 'completion_manifest.json').write_text(json.dumps(cell_files), encoding='utf-8')
    item = {'cell': 'episodes/000', 'spec': spec,
            'cell_completion_sha256': audit.sha(cell / 'completion_manifest.json')}
    summary = {'status': 'PASS', 'results': [item, deepcopy(item)] if duplicate_audit else [item]}
    (source / 'audit/summary.json').write_text(json.dumps(summary), encoding='utf-8')
    root_files = {'episodes/000/' + name: digest for name, digest in files.items()}
    root_files['audit/summary.json'] = audit.sha(source / 'audit/summary.json')
    (source / 'completion_manifest.json').write_text(json.dumps(root_files), encoding='utf-8')
    return source, cell, audit.sha(source / 'completion_manifest.json')


def test_historical_nested_manifest_bound_through_root_sealed_independent_audit(tmp_path):
    source, cell, root_sha = historical_source_fixture(tmp_path)
    assert 'episodes/000/completion_manifest.json' not in audit.read(source / 'completion_manifest.json')
    binding = audit.source_cell_binding(source, 'episodes/000', root_sha)
    assert binding['source_completion_sha256'] == audit.sha(cell / 'completion_manifest.json')
    assert binding['source_result_sha256'] == audit.sha(cell / 'result.json')


def test_resealed_scientific_cell_cannot_replace_historically_audited_seal(tmp_path):
    source, cell, root_sha = historical_source_fixture(tmp_path)
    (cell / 'requests.jsonl').write_bytes(b'changed-paid-record')
    rows = audit.read(cell / 'completion_manifest.json')
    rows['requests.jsonl'] = audit.sha(cell / 'requests.jsonl')
    (cell / 'completion_manifest.json').write_text(json.dumps(rows), encoding='utf-8')
    with pytest.raises(AssertionError, match='audited cell seal changed'):
        audit.source_cell_binding(source, 'episodes/000', root_sha)


@pytest.mark.parametrize('duplicate,wrong_file,pattern', [
    (True, False, 'ambiguous'), (False, True, 'scientific file binding')])
def test_historical_binding_requires_unique_audit_and_both_scientific_seals(tmp_path, duplicate, wrong_file, pattern):
    source, _, root_sha = historical_source_fixture(tmp_path, duplicate_audit=duplicate, wrong_cell_file=wrong_file)
    with pytest.raises(AssertionError, match=pattern):
        audit.source_cell_binding(source, 'episodes/000', root_sha)


def test_historical_audit_cannot_be_changed_under_fixed_root(tmp_path):
    source, _, root_sha = historical_source_fixture(tmp_path)
    (source / 'audit/summary.json').write_text(json.dumps({'status': 'PASS', 'results': []}), encoding='utf-8')
    with pytest.raises(AssertionError, match='audit is not root-bound'):
        audit.source_cell_binding(source, 'episodes/000', root_sha)
