"""Population and accounting checks for the bounded window teacher runner."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_clpu_window_teacher as runner
from test_nature34_clpu_window import row, shell, selection_observation
from test_nature34_clpu_window_counterfactual import source, BINDING


def state_slots():
    return [{'state_id': f'c{i:02d}_{mode}', 'context_id': i, 'scenario_id': f'N2_fault{i}',
        'event_id': [30, 20, 6, 3][i % 4], 'source_cell': f'episodes/{i:03d}',
        'mode': mode, 'status': 'AVAILABLE', 'completed_steps': 0 if mode == 'normal' else 1,
        'snapshot': f'snapshots/c{i:02d}_{mode}/snapshot.json', 'prefix_physical_points': 61,
        'candidates': [{'branch_id': 1, 'roles': ['window_beam', 'graph_greedy']},
                       {'branch_id': 2, 'roles': ['fixed_hash_fill']}], 'reason': None, 'details': {}}
        for i in range(8) for mode in runner.MODES]


def test_schedule_retains_slots_pairing_fixed_order_and_paid_allowance():
    states = state_slots()
    left = runner.schedule_for(states)
    right = runner.schedule_for(list(reversed(states)))
    assert left == right
    assert len(left) == 32
    assert [r['episode_id'] for r in left] == list(range(32))
    assert {r['state_id'] for r in left} == {r['state_id'] for r in states}
    assert sum(r['effective_new_point_allowance'] for r in left) == 4096
    assert all(r['max_new_points'] == 128 and r['max_expansions'] == 16 for r in left)
    assert all(r['phase'] == 'training_teacher' and r['method'] == 'beam' and r['seed'] == 0 for r in left)


def test_unavailable_modes_remain_slots_but_do_not_become_forks():
    states = state_slots()
    states[1].update(status='UNAVAILABLE', candidates=[], prefix_physical_points=None,
                     snapshot=None, completed_steps=None, reason='NO_ELIGIBLE_COVERAGE_STATE')
    states[0]['prefix_physical_points'] = 200
    schedule = runner.schedule_for(states)
    assert len(states) == 16 and len(schedule) == 30
    assert not any(row['state_id'] == states[1]['state_id'] for row in schedule)
    assert {row['effective_new_point_allowance'] for row in schedule if row['state_id'] == states[0]['state_id']} == {56}


@pytest.mark.parametrize('kind', ['slot', 'baseline', 'candidate', 'false_missing'])
def test_population_omissions_and_duplicate_baselines_fail(kind):
    states = state_slots()
    if kind == 'slot':
        states.pop()
    elif kind == 'baseline':
        states[0]['candidates'][1]['roles'].append('window_beam')
    elif kind == 'candidate':
        states[0]['candidates'][1]['branch_id'] = 1
    else:
        states[0]['status'] = 'UNAVAILABLE'
    with pytest.raises(ValueError):
        runner.schedule_for(states)


def test_candidates_keep_checked_beam_and_deduplicate_full_plans():
    duplicate = row(2, (1,), energy=8.)
    rows = [row(1, (1,), energy=8.), duplicate, row(3, (3,), energy=7.), row(4, (4,), energy=6.)]
    search = shell(rows, turn=0)
    search.observation = lambda representation='raw': selection_observation(search)
    # Beam may choose a previously checked expansion parent. It is retained.
    search._checked[rows[0]['info']['actions']] = {'status': 'FAIL'}
    context = {'scenario_id': 'fixed', 'event_id': 6}
    chosen = runner.choose_candidates(search, context, 'normal')
    assert chosen[0]['branch_id'] == 1 and 'window_beam' in chosen[0]['roles']
    assert {row['branch_id'] for row in chosen} == {1, 3, 4}
    assert 'graph_greedy' in next(row for row in chosen if row['branch_id'] == 3)['roles']
    assert chosen == runner.choose_candidates(search, context, 'normal')


def test_candidate_roles_merge_when_beam_and_greedy_match():
    search = shell([row(1, (1,), energy=8.), row(2, (2,), energy=7.)], turn=1)
    search.observation = lambda representation='raw': selection_observation(search)
    chosen = runner.choose_candidates(search, {'scenario_id': 'fixed', 'event_id': 6}, 'coverage')
    assert len(chosen) == 2
    assert 'window_beam' in chosen[0]['roles'] and 'graph_greedy' in chosen[0]['roles']


def test_progressive_discovery_selects_first_current_state_and_consumes_exact_history():
    env, oracle, result, steps, plans, observations = source()
    completed = [row['response'] for row in steps if row['status'] == 'COMPLETED']
    result.update(completed_expansions=len(completed), stop_reason=completed[-1]['stop_reason'],
                  oracle={'points': oracle.points})
    context = {'scenario_id': env.scenario['scenario_id'], 'event_id': 3}
    before = oracle.points
    selected, trace = runner.discover_state_positions(env, result, steps, plans, oracle.rows, BINDING, context)
    assert oracle.points == before
    assert trace['new_physical_points'] == 0
    assert trace['source_decisions_replayed'] == len(completed)
    assert trace['source_points_replayed'] == oracle.points
    for mode, state in selected.items():
        first = next(row for row in trace['visited_states'] if row['mode'] == mode and row['eligible'])
        assert state['completed_steps'] == first['completed_steps']
        assert state['observation'] == observations[state['completed_steps']]


def test_progressive_replay_rejects_future_identity_or_changed_final_evidence():
    env, oracle, result, steps, plans, _ = source()
    completed = [row['response'] for row in steps if row['status'] == 'COMPLETED']
    result.update(completed_expansions=len(completed), stop_reason=completed[-1]['stop_reason'],
                  oracle={'points': oracle.points})
    context = {'scenario_id': env.scenario['scenario_id'], 'event_id': 3}
    altered = deepcopy(result)
    altered['best_plan'] = {'unrecorded': True}
    with pytest.raises(ValueError, match='final incumbent'):
        runner.discover_state_positions(env, altered, steps, plans, oracle.rows, BINDING, context)
    points = deepcopy(oracle.rows)
    points[0], points[1] = points[1], points[0]
    with pytest.raises(Exception, match='contiguous'):
        runner.discover_state_positions(env, result, steps, plans, points, BINDING, context)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def test_actual_costs_include_cold_initialization_and_reliable_zero_call_cells(tmp_path):
    specs = [{'cell': 'episodes/000'}, {'cell': 'episodes/001'}]
    write(tmp_path / 'episodes/000/oracle_summary.json', {'points': 61, 'actual_numerical_entries': 247})
    write(tmp_path / 'episodes/001/result.json', {'oracle': {'live_initialization_attempted': False},
        'live_oracle_summary': {'points': 0, 'actual_numerical_entries': 0, 'status': 'NO_NEW_PHYSICAL_CALLS'}})
    assert runner.costs(tmp_path, specs) == {'known_physical_points': 61, 'known_numerical_entries': 247,
                                             'unknown_cost_cells': []}


def test_missing_partial_oracle_accounting_is_unknown_never_zero(tmp_path):
    specs = [{'cell': 'episodes/000'}, {'cell': 'episodes/001'}]
    write(tmp_path / 'episodes/001/result.json', {'oracle': {'live_initialization_attempted': True},
        'live_oracle_summary': {'points': 0, 'actual_numerical_entries': 0, 'status': 'NO_NEW_PHYSICAL_CALLS'}})
    result = runner.costs(tmp_path, specs)
    assert result['unknown_cost_cells'] == ['episodes/000', 'episodes/001']


@pytest.mark.parametrize('body', ['', '<testcase><failure/></testcase>', '<testcase><skipped/></testcase>'])
def test_empty_failed_or_skipped_tests_are_not_execution_evidence(tmp_path, body):
    path = tmp_path / 'tests.xml'
    path.write_text('<testsuite>' + body + '</testsuite>', encoding='utf-8')
    with pytest.raises(ValueError, match='all-passed'):
        runner._passed_tests(path)


def test_nonempty_passed_tests_counted(tmp_path):
    path = tmp_path / 'tests.xml'
    path.write_text('<testsuite><testcase name="real"/></testsuite>', encoding='utf-8')
    assert runner._passed_tests(path) == 1


def test_gate_is_frozen_before_outcomes_and_requires_both_diversities():
    assert runner.GATE['minimum_states'] == runner.GATE['minimum_faults'] == runner.GATE['minimum_events'] == 2
    assert runner.GATE['minimum_auc_advantage_pp'] == .1
    assert runner.GATE['final_quality_tolerance_pp'] == 1e-9


def bound_source_fixture(tmp_path):
    spec = {'cell': 'episodes/000', 'context_id': 0, 'method': 'beam', 'window_variant': 'coverage_v1'}
    cell = tmp_path / spec['cell']
    for name in ('result.json', 'points.jsonl.gz', 'steps.jsonl.gz', 'plans.jsonl.gz', 'requests.jsonl'):
        write(cell / name, {'fixture': name})
    cell_manifest = {p.name: runner.sha(p) for p in cell.iterdir()}
    write(cell / 'completion_manifest.json', cell_manifest)
    write(tmp_path / 'audit/summary.json', {'status': 'PASS', 'results': [{
        'cell': spec['cell'], 'spec': spec, 'cell_completion_sha256': runner.sha(cell / 'completion_manifest.json')}]})
    parent_manifest = {spec['cell'] + '/' + name: digest for name, digest in cell_manifest.items()}
    parent_manifest['audit/summary.json'] = runner.sha(tmp_path / 'audit/summary.json')
    assert spec['cell'] + '/completion_manifest.json' not in parent_manifest
    return spec, parent_manifest


def test_nested_cell_seal_is_bound_by_parent_sealed_independent_audit(tmp_path):
    spec, parent = bound_source_fixture(tmp_path)
    assert runner.verify_source_cell(tmp_path, parent, spec) == tmp_path / spec['cell']


@pytest.mark.parametrize('changed', ['audit', 'nested_seal', 'journal', 'parent_entry', 'audit_spec', 'duplicate_audit'])
def test_nested_cell_binding_rejects_independent_tampering(tmp_path, changed):
    spec, parent = bound_source_fixture(tmp_path)
    cell = tmp_path / spec['cell']
    if changed == 'audit':
        write(tmp_path / 'audit/summary.json', {'status': 'PASS', 'results': []})
    elif changed == 'nested_seal':
        write(cell / 'completion_manifest.json', {'result.json': 'replacement'})
    elif changed == 'journal':
        write(cell / 'result.json', {'changed': True})
    elif changed == 'parent_entry':
        parent[spec['cell'] + '/result.json'] = 'different'
    else:
        value = runner.read(tmp_path / 'audit/summary.json')
        if changed == 'audit_spec':
            value['results'][0]['spec']['context_id'] = 9
        else:
            value['results'].append(deepcopy(value['results'][0]))
        write(tmp_path / 'audit/summary.json', value)
        parent['audit/summary.json'] = runner.sha(tmp_path / 'audit/summary.json')
    with pytest.raises(ValueError):
        runner.verify_source_cell(tmp_path, parent, spec)
