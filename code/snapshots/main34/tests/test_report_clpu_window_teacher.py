"""Synthetic fixtures test teacher accounting and reporting, not method results."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import report_clpu_window_teacher as report


def curve(point=4, final=.6, *, initial=.2):
    return [{'new_physical_points': 0, 'quality': initial, 'kind': 'INITIAL_CERTIFIED_INCUMBENT'},
            {'new_physical_points': point, 'quality': final, 'kind': 'COMPLETE_CERTIFICATE'}]


def row(branch=10, *, baseline=True, point=4, final=.6):
    trajectory = curve(point, final)
    return {'forced_branch_id': branch, 'roles': ['window_beam'] if baseline else ['fixed_hash_fill'],
        'cell': f'episodes/{branch}', 'snapshot_sha256': 'one-common-state',
        'source_snapshot_file_sha256': 'one-snapshot-file', 'new_point_allowance': 4,
        'initial_quality': .2, 'initial_quality_pct': 20, 'final_quality': final,
        'final_quality_pct': 100 * final, 'quality_point_trajectory': trajectory,
        'auc_gain_pp': report.curve_auc(.2, trajectory, 4),
        'g1_pp': 100 * (final - .2), 'gH_pp': 100 * (final - .2),
        'new_physical_points': 4, 'new_numerical_entries': 19, 'stop_reason': 'FORK_EXPANSION_LIMIT'}


def state(identity, event, *, gain=True):
    return {'state_id': identity, 'scenario_id': identity.split('_')[0], 'event_id': event,
            'effect': {'has_rule_superiority_signal': gain}}


def test_curve_includes_certificate_at_paid_hit_and_carries_after_stop():
    assert report.curve_auc(.2, curve(4), 4) == pytest.approx(10)
    assert report.curve_auc(.2, curve(1), 4) == pytest.approx(40)
    assert report.curve_auc(.2, curve(2), 4) == pytest.approx(30)


def test_same_point_zero_point_cache_certificates_and_zero_allowance():
    values = curve(0, .3)
    values.append({'new_physical_points': 0, 'quality': .6, 'kind': 'COMPLETE_CERTIFICATE'})
    assert report.validate_curve(.2, values, 4, 0) == values
    assert report.curve_auc(.2, values, 4) == pytest.approx(40)
    assert report.curve_auc(.2, values, 0) is None


@pytest.mark.parametrize('edit', [
    lambda c: c[0].update(quality=.1),
    lambda c: c[1].update(new_physical_points=5),
    lambda c: c[1].update(quality=.1),
    lambda c: c[1].update(quality=float('nan')),
    lambda c: c[1].update(kind='GRAPH_ONLY_ESTIMATE'),
])
def test_invalid_or_unpaid_quality_boundary_rejected(edit):
    values = curve()
    edit(values)
    with pytest.raises(ValueError):
        report.validate_curve(.2, values, 4, 4)


def test_first_reach_keeps_unreached_null_and_initial_hit_zero():
    assert report.first_reach(curve(), .2)['new_physical_points'] == 0
    assert report.first_reach(curve(), .6)['new_physical_points'] == 4
    value = report.first_reach(curve(), .61)
    assert value == {'status': 'NOT_REACHED_WITHIN_BOUNDED_FORK',
                     'new_physical_points': None, 'quality_pct': None}


def test_early_equal_quality_alternative_counts_but_not_as_model_success():
    effect = report.state_effect([row(), row(20, baseline=False, point=1)])
    assert effect['auc_span_pp'] == pytest.approx(30)
    assert effect['eligible_alternative_branch_ids'] == [20]
    assert effect['has_rule_superiority_signal'] is True
    assert effect['candidates'][0]['counts_for_investment_gate'] is False


def test_early_gain_with_final_regression_does_not_count_and_stays_in_table():
    effect = report.state_effect([row(), row(20, baseline=False, point=1, final=.59)])
    assert effect['best_measured_alternative_U_delta_pp'] > .1
    assert effect['has_rule_superiority_signal'] is False
    assert effect['unreached_baseline_target_candidates'] == 1
    assert len(effect['candidates']) == 2
    assert effect['candidates'][1]['final_quality_minus_window_beam_pp'] == pytest.approx(-1)


def test_gate_uses_any_qualified_alternative_not_only_best_auc():
    effect = report.state_effect([row(), row(20, baseline=False, point=0, final=.59),
                                  row(30, baseline=False, point=2)])
    assert effect['best_measured_alternative_branch_id'] == 20
    assert effect['eligible_alternative_branch_ids'] == [30]
    assert effect['has_rule_superiority_signal'] is True


def test_ties_or_no_positive_budget_do_not_pass():
    values = [row(), row(20, baseline=False)]
    assert report.state_effect(values)['has_rule_superiority_signal'] is False
    for value in values:
        value['new_point_allowance'] = 0
        value['auc_gain_pp'] = None
    effect = report.state_effect(values)
    assert effect['auc_span_pp'] is None and effect['has_rule_superiority_signal'] is False


@pytest.mark.parametrize('edit', [
    lambda r: r[1].update(snapshot_sha256='another-state'),
    lambda r: r[1].update(source_snapshot_file_sha256='another-file'),
    lambda r: r[1].update(new_point_allowance=3),
    lambda r: r[1].update(initial_quality_pct=21),
    lambda r: r[1].update(forced_branch_id=10),
    lambda r: r[1].update(roles=['window_beam']),
])
def test_candidates_must_share_exact_start_and_budget(edit):
    values = [row(), row(20, baseline=False)]
    edit(values)
    with pytest.raises(ValueError):
        report.state_effect(values)


def test_gate_requires_separate_states_faults_and_events():
    assert not report.mechanism_gate([state('a_normal', 1), state('a_coverage', 2)])['passed']
    assert not report.mechanism_gate([state('a_normal', 1), state('b_normal', 1)])['passed']
    passed = report.mechanism_gate([state('a_normal', 1), state('b_normal', 2)])
    assert passed['passed'] and passed['qualifying_states'] == 2
    assert 'BC superiority' in passed['interpretation']


def seal(directory):
    report.save(directory / 'completion_manifest.json', {p.name: report.sha(p)
        for p in directory.iterdir() if p.is_file() and p.name != 'completion_manifest.json'})


def rewrite(path, mutate):
    import json
    value = report.read(path)
    mutate(value)
    path.write_text(json.dumps(value), encoding='utf-8')


def build_run(tmp_path, monkeypatch):
    root = tmp_path / 'project'
    source = root / 'runs' / report.SOURCE_RUN
    source.mkdir(parents=True)
    report.save(source / 'completion_manifest.json', {'fixture': 'SYNTHETIC_UNIT_TEST_ONLY'})
    monkeypatch.setattr(report, 'ROOT', root)
    monkeypatch.setattr(report, 'SOURCE_SHA', report.sha(source / 'completion_manifest.json'))
    run = root / 'runs' / 'synthetic_teacher_test'
    run.mkdir()
    states, schedule, audits, snapshots = [], [], [], []
    for index in range(8):
        for mode in report.MODES:
            available = index < 2 and mode == 'normal'
            state_id = f'c{index:02d}_{mode}'
            candidates = [{'branch_id': 10, 'roles': ['window_beam']},
                          {'branch_id': 20, 'roles': ['fixed_hash_fill']}] if available else []
            state = {'state_id': state_id, 'context_id': index, 'scenario_id': f'fault{index}',
                'event_id': [30, 20, 6, 3][index % 4], 'source_cell': f'episodes/{index:03d}',
                'mode': mode, 'status': 'AVAILABLE' if available else 'UNAVAILABLE',
                'completed_steps': 5 if available else None,
                'snapshot': f'snapshots/{state_id}/snapshot.json' if available else None,
                'prefix_physical_points': 252 if available else None, 'candidates': candidates,
                'reason': None if available else 'NO_ELIGIBLE_STATE', 'details': {}}
            states.append(state)
            if not available:
                continue
            snap = run / state['snapshot']
            snap.parent.mkdir(parents=True)
            report.save(snap, {'synthetic_fixture': state_id})
            snapshots.append(state['snapshot'])
            for candidate in candidates:
                spec = {key: state[key] for key in ('state_id', 'context_id', 'scenario_id',
                    'event_id', 'source_cell', 'completed_steps', 'snapshot')}
                spec.update(episode_id=len(schedule), cell=f'episodes/{len(schedule):03d}',
                    forced_branch_id=candidate['branch_id'], roles=candidate['roles'],
                    max_new_points=128, effective_new_point_allowance=4, max_expansions=16,
                    phase='training_teacher', method='beam', seed=0)
                schedule.append(spec)
                directory = run / spec['cell']
                directory.mkdir(parents=True)
                trajectory = curve(4 if candidate['branch_id'] == 10 else 1)
                result = {'spec': spec, 'status': 'BOUNDED_COUNTERFACTUAL_COMPLETE',
                    'new_training': 0, 'optimizer_updates': 0, 'formal_holdout_numerical_access': False,
                    'tune_numerical_access': False, 'forced_branch_id': candidate['branch_id'],
                    'live_oracle_summary': {'points': 4, 'actual_numerical_entries': 19},
                    'oracle': {'new_physical_points': 4, 'historical_restore_new_physical_points': 0},
                    'new_point_allowance': 4, 'curve_budget_grid': report.GRID,
                    'initial_quality': .2, 'first_step_quality': .6, 'final_quality': .6,
                    'quality_point_trajectory': trajectory,
                    'best_plan': {'certified_complete': True, 'status': 'COMPLETE',
                        'validation': {'status': 'COMPLETE', 'mean_served_fraction': .6}},
                    'auc_gain_pp': report.curve_auc(.2, trajectory, 4),
                    'certificate_updates': [{'new_physical_points': trajectory[1]['new_physical_points'],
                                             'quality_after': .6}],
                    'completed_expansions': 1, 'g1_pp': 40, 'gH_pp': 40,
                    'snapshot_sha256': state_id + '-internal-digest',
                    'source_snapshot_file_sha256': report.sha(snap), 'elapsed_seconds': 2,
                    'stop_reason': 'FORK_EXPANSION_LIMIT'}
                report.save(directory / 'spec.json', spec)
                report.save(directory / 'result.json', result)
                seal(directory)
                audits.append({'cell': spec['cell'],
                    'cell_completion_sha256': report.sha(directory / 'completion_manifest.json')})
    report.save(run / 'states.json', states)
    report.save(run / 'schedule.json', schedule)
    report.save(run / 'source_binding.json', {'run': report.SOURCE_RUN, 'completion_sha256': report.SOURCE_SHA})
    report.save(run / 'prephysical_freeze_manifest.json', {name: report.sha(run / name)
        for name in ['states.json', 'schedule.json', 'source_binding.json'] + snapshots})
    report.save(run / 'controller_completion.json', {'status': 'COMPLETE_CELLS_PENDING_AUDIT',
        'errors': [], 'unknown_cost_cells': [], 'completed': [row['cell'] for row in schedule],
        'known_physical_points': 16, 'known_numerical_entries': 76, 'wall_seconds': 5})
    audit = run / 'audit'
    audit.mkdir()
    summary = {'status': 'PASS', 'audited_cells': 4, 'results': audits,
        'physical_points': 16, 'numerical_entries': 76}
    for key, name in [('schedule_sha256', 'schedule.json'), ('states_sha256', 'states.json'),
                      ('source_binding_sha256', 'source_binding.json'),
                      ('freeze_sha256', 'prephysical_freeze_manifest.json'),
                      ('controller_completion_sha256', 'controller_completion.json')]:
        summary[key] = report.sha(run / name)
    report.save(audit / 'summary.json', summary)
    seal(audit)
    return run


def test_report_retains_all_slots_and_gates_only_learning_potential(tmp_path, monkeypatch):
    run = build_run(tmp_path, monkeypatch)
    result = report.report(run)
    assert result['status'] == 'TEACHER_SIGNAL_FOR_BOUNDED_LEARNING'
    assert result['available_states'] == 2 and result['unavailable_states'] == 14
    assert result['measured_forks'] == 4 and result['new_physical_points'] == 16
    assert result['new_numerical_entries'] == 76 and result['new_optimizer_updates'] == 0
    states = report.read(run / 'derived_report/all_states.json')
    assert len(states) == 16
    assert len(report.read(run / 'derived_report/all_forks.json')) == 4
    assert len(report.verify_manifest(run / 'derived_report')) == 4
    assert not (run / 'terminal_status.json').exists()
    assert not (run / 'completion_manifest.json').exists()
    with pytest.raises(ValueError, match='overwrite'):
        report.report(run)


def test_changed_result_rejected_before_writing(tmp_path, monkeypatch):
    run = build_run(tmp_path, monkeypatch)
    rewrite(run / 'episodes/000/result.json', lambda value: value.update(auc_gain_pp=100))
    with pytest.raises(ValueError, match='Changed sealed evidence'):
        report.report(run)
    assert not (run / 'derived_report').exists()


def test_resealed_result_cannot_reuse_old_audit(tmp_path, monkeypatch):
    run = build_run(tmp_path, monkeypatch)
    cell = run / 'episodes/000'
    rewrite(cell / 'result.json', lambda value: value.update(auc_gain_pp=100))
    rewrite(cell / 'completion_manifest.json', lambda value: value.update({'result.json': report.sha(cell / 'result.json')}))
    with pytest.raises(ValueError, match='Audited cell seal differs'):
        report.report(run)


def test_frozen_state_or_source_binding_cannot_change(tmp_path, monkeypatch):
    run = build_run(tmp_path, monkeypatch)
    rewrite(run / 'states.json', lambda value: value.pop())
    with pytest.raises(ValueError, match='Changed sealed evidence'):
        report.report(run)


def test_unknown_cost_blocks_report(tmp_path, monkeypatch):
    run = build_run(tmp_path, monkeypatch)
    rewrite(run / 'controller_completion.json', lambda value: value.update(unknown_cost_cells=['episodes/000']))
    with pytest.raises(ValueError, match='not closed'):
        report.report(run)


def test_duplicate_audit_cells_do_not_count_as_complete(tmp_path, monkeypatch):
    run = build_run(tmp_path, monkeypatch)
    audit = run / 'audit'
    rewrite(audit / 'summary.json', lambda value: value['results'].__setitem__(1, value['results'][0]))
    rewrite(audit / 'completion_manifest.json', lambda value: value.update({'summary.json': report.sha(audit / 'summary.json')}))
    with pytest.raises(ValueError, match='omitted/repeated/added'):
        report.report(run)


def test_missing_slot_or_candidate_cannot_be_removed(tmp_path, monkeypatch):
    run = build_run(tmp_path, monkeypatch)
    states, schedule = report.read(run / 'states.json'), report.read(run / 'schedule.json')
    with pytest.raises(ValueError, match='sixteen'):
        report.validate_population(states[:-1], schedule)
    with pytest.raises(ValueError, match='Missing, repeated or extra'):
        report.validate_population(states, schedule[:-1])


def test_wrong_cost_or_auc_cannot_be_reported_even_inside_otherwise_valid_case(tmp_path, monkeypatch):
    run = build_run(tmp_path, monkeypatch)
    result = report.read(run / 'episodes/000/result.json')
    changed = deepcopy(result)
    changed['auc_gain_pp'] = 100
    with pytest.raises(ValueError, match='Reported U differs'):
        report.case_row(result['spec'], changed)
    changed = deepcopy(result)
    changed['oracle']['historical_restore_new_physical_points'] = 1
    with pytest.raises(ValueError, match='Hidden restore'):
        report.case_row(result['spec'], changed)
