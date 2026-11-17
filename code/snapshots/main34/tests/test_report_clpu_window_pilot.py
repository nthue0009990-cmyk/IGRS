"""Reporting estimands and evidence gates; these tests make no physical calls."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import report_clpu_window_pilot as report


def checkpoint(step, quality, points=None, seconds=None):
    points = 3 + step if points is None else points
    return {'search_choices': step, 'quality_pct': quality,
            'fully_certified': quality is not None, 'physical_points': points,
            'numerical_entries': 4 * points, 'elapsed_seconds': 1 + step if seconds is None else seconds}


def case(group, qualities, *, points=None, seconds=None):
    timeline = [checkpoint(i, q) for i, q in enumerate(qualities)]
    last = timeline[-1]
    return {'group': group, 'context_id': 2, 'cell': 'cells/' + group, 'scenario_id': 'N2__line.l10__line.l24',
        'event_id': 6, 'quality_pct': qualities[-1], 'wait_baseline_quality_pct': qualities[0],
        'energy_kwh': qualities[-1], 'best_plan': {'actual_plan': group} if qualities[-1] is not None else None,
        'cell_completion_sha256': group + '-seal', 'result_sha256': group + '-result',
        'physical_points': last['physical_points'] if points is None else points,
        'numerical_entries': last['numerical_entries'],
        'elapsed_seconds': last['elapsed_seconds'] if seconds is None else seconds,
        'search_choices': last['search_choices'], 'checkpoints': timeline}


def test_primary_target_is_paired_g0_final_not_g1_final_or_wait_gain():
    g0 = case(report.GROUPS[0], [4, 4, 60, 80])
    g1 = case(report.GROUPS[1], [4, 80, 95, 99])
    pair = report.pair_cases(g0, g1)
    assert pair['target_original_final_quality_pct'] == 80
    assert pair['first_target_reach'][report.GROUPS[0]]['search_choices'] == 3
    assert pair['first_target_reach'][report.GROUPS[1]]['search_choices'] == 1
    assert pair['first_target_reach_G1_minus_G0']['search_choices'] == -2
    assert pair['final_quality_G1_minus_G0_pp'] == 19


def test_unreached_quality_is_null_with_explicit_status_and_remains_denominator():
    g0 = case(report.GROUPS[0], [4, 80])
    g1 = case(report.GROUPS[1], [4, 79])
    pair = report.pair_cases(g0, g1)
    reach = pair['first_target_reach'][report.GROUPS[1]]
    assert reach['status'] == 'NOT_REACHED_WITHIN_BUDGET'
    assert reach['physical_points'] is None and reach['elapsed_seconds'] is None
    assert all(v is None for v in pair['first_target_reach_G1_minus_G0'].values())
    summary = report.describe_pairs([pair])
    assert summary['pairs'] == 1
    assert summary['earlier_target_reach']['search_choices']['missing_pairs'] == 1
    assert summary['final_comparison_counts'] == {'G1_WORSE': 1}


def test_initialization_hit_costs_real_points_and_cold_time():
    points = [checkpoint(0, 80, points=61, seconds=13.4), checkpoint(1, 80, points=72, seconds=18)]
    reach = report.first_reach(points, 80)
    assert reach['search_choices'] == 0
    assert reach['physical_points'] == 61
    assert reach['elapsed_seconds'] == 13.4


def test_numeric_tolerance_and_strict_improvement_are_different():
    points = [checkpoint(0, 80), checkpoint(1, 80 + report.EPS_PP / 2),
              checkpoint(2, 80 + report.EPS_PP * 2)]
    assert report.first_reach(points, 80 + report.EPS_PP / 2)['search_choices'] == 0
    assert report.first_reach(points, 80, strict=True)['search_choices'] == 2


def test_uncertified_entries_do_not_hit_target_or_fabricate_baseline():
    points = [checkpoint(0, None), checkpoint(1, 50)]
    assert report.first_reach(points, 0)['search_choices'] == 1
    assert report.first_reach(points, None)['status'] == 'TARGET_UNAVAILABLE'


def test_protected_quality_does_not_hide_intrinsic_regression_and_pays_both_runs():
    g0 = case(report.GROUPS[0], [4, 95], points=120, seconds=20)
    g1 = case(report.GROUPS[1], [4, 10], points=256, seconds=100)
    pair = report.pair_cases(g0, g1)
    protected = pair['protected_output']
    assert pair['final_quality_G1_minus_G0_pp'] == -85
    assert protected['selected_group'] == report.GROUPS[0]
    assert protected['quality_pct'] == 95
    assert protected['physical_points'] == 376
    assert protected['sum_worker_elapsed_seconds'] == 120
    assert protected['gain_over_original_pp'] == 0


def test_protection_selects_actual_energy_maximum_and_exact_tie_uses_original():
    a = case(report.GROUPS[0], [4, 80])
    b = case(report.GROUPS[1], [4, 80])
    assert report.protected_output(a, b)['selected_group'] == report.GROUPS[0]
    b['quality_pct'] = b['energy_kwh'] = 80 + report.EPS_PP / 2
    selected = report.protected_output(a, b)
    assert selected['selected_group'] == report.GROUPS[1]
    assert selected['best_plan'] == b['best_plan']
    assert selected['source_cell_completion_sha256'][a['cell']] == a['cell_completion_sha256']


def test_no_certificates_does_not_become_safe_zero_output():
    a = case(report.GROUPS[0], [None])
    b = case(report.GROUPS[1], [None])
    chosen = report.protected_output(a, b)
    assert chosen['status'] == 'NO_CERTIFIED_OUTPUT'
    assert chosen['quality_pct'] is None and chosen['selected_cell'] is None
    assert chosen['physical_points'] == 6


def test_fixed_step_anytime_does_not_interpolate_or_discard_early_stop():
    timeline = [checkpoint(i, 4 if i < 17 else 80) for i in range(20)]
    values = report.anytime(timeline)
    assert values['16']['quality_pct'] == 4
    assert values['16']['status'] == 'OBSERVED_SEARCH_BOUNDARY'
    assert values['64']['quality_pct'] == 80
    assert values['64']['observed_at_search_choices'] == 19
    assert values['64']['status'] == 'TERMINATED_INCUMBENT_CARRIED_FORWARD'
    assert set(values) == {'16', '64', '128', '256'}


@pytest.mark.parametrize('field,value', [('search_choices', 3), ('physical_points', 1),
    ('numerical_entries', 1), ('elapsed_seconds', .1), ('quality_pct', 2),
    ('fully_certified', False)])
def test_invalid_boundaries_cannot_support_early_plan_claim(field, value):
    timeline = [checkpoint(0, 4), checkpoint(1, 5)]
    timeline[1][field] = value
    with pytest.raises(ValueError):
        report.validate_checkpoints(timeline)


def test_negative_or_nonfinite_cost_is_rejected():
    for value in (-1, float('nan'), float('inf')):
        point = checkpoint(0, 4)
        point['elapsed_seconds'] = value
        with pytest.raises(ValueError):
            report.validate_checkpoints([point])


def test_wrong_context_pair_is_rejected():
    a = case(report.GROUPS[0], [4, 80])
    b = case(report.GROUPS[1], [4, 80])
    b['event_id'] = 30
    with pytest.raises(ValueError, match='Different contexts'):
        report.pair_cases(a, b)


def test_diagnosed_two_are_flags_not_a_filter():
    pairs = []
    for event in (6, 30):
        a = case(report.GROUPS[0], [4, 80])
        b = case(report.GROUPS[1], [4, 80])
        a['event_id'] = b['event_id'] = event
        pairs.append(report.pair_cases(a, b))
    assert [p['previously_diagnosed_context'] for p in pairs] == [True, False]
    assert report.describe_pairs(pairs)['pairs'] == 2


def build_run(run):
    """Synthetic certificates test report binding only; no scientific result."""
    contexts = [
        ('l21', 'l26', 30), ('l4', 'l30', 20), ('l10', 'l24', 6), ('l21', 'l30', 3),
        ('l11', 'l16', 30), ('l17', 'l20', 20), ('l17', 'l29', 6), ('l12', 'l24', 3)]
    population = {'contexts': [{'scenario_id': f'N2__line.{a}__line.{b}', 'event_id': e}
                                for a, b, e in contexts]}
    schedule, audit_results = [], []
    for index, context in enumerate(population['contexts']):
        for gi, group in enumerate(report.GROUPS):
            spec = {**context, 'context_id': index, 'cell': f'episodes/{2 * index + gi:03d}',
                    'group': group, 'method': 'beam', 'seed': 0,
                    'window_variant': 'original' if gi == 0 else 'coverage_v1'}
            schedule.append(spec)
            cell = run / spec['cell']
            cell.mkdir(parents=True)
            timeline = [checkpoint(0, 4), checkpoint(1, 80)]
            result = {'spec': spec, 'status': 'CERTIFIED_INCUMBENT',
                'best_plan': {'actions': [index, gi], 'certified_complete': True, 'status': 'COMPLETE',
                    'validation': {'status': 'COMPLETE', 'mean_served_fraction': .8, 'energy_kwh': 80}},
                'optimizer_updates': 0, 'formal_holdout_numerical_access': False,
                'tune_numerical_access': False, 'checkpoints': timeline,
                'completed_expansions': 1, 'wait_baseline_quality_pct': 4, 'elapsed_seconds': 3,
                'oracle': {'points': 4, 'actual_numerical_entries': 16}, 'stop_reason': 'BUDGET'}
            report.save(cell / 'spec.json', spec)
            report.save(cell / 'result.json', result)
            report.save(cell / 'completion_manifest.json', {
                name: report.sha(cell / name) for name in ('spec.json', 'result.json')})
            audit_results.append({'cell': spec['cell'],
                'cell_completion_sha256': report.sha(cell / 'completion_manifest.json')})
    report.save(run / 'population.json', population)
    report.save(run / 'schedule.json', schedule)
    report.save(run / 'prephysical_freeze_manifest.json', {'population.json': report.sha(run / 'population.json')})
    report.save(run / 'controller_completion.json', {'status': 'COMPLETE_CELLS_PENDING_AUDIT',
        'completed': [r['cell'] for r in schedule], 'errors': [], 'unknown_cost_cells': [],
        'known_physical_points': 64, 'known_numerical_entries': 256, 'wall_seconds': 25})
    audit = run / 'audit'
    audit.mkdir()
    report.save(audit / 'summary.json', {'status': 'PASS', 'audited_cells': 16,
        'results': audit_results, 'physical_points': 64, 'numerical_entries': 256,
        'schedule_sha256': report.sha(run / 'schedule.json'),
        'freeze_sha256': report.sha(run / 'prephysical_freeze_manifest.json'),
        'controller_completion_sha256': report.sha(run / 'controller_completion.json')})
    report.save(audit / 'completion_manifest.json', {'summary.json': report.sha(audit / 'summary.json')})
    return run


def change_json(path, mutate):
    value = report.read(path)
    mutate(value)
    path.write_text(json.dumps(value), encoding='utf-8')


def test_complete_report_preserves_all_pairs_and_delivers_actual_paid_plans(tmp_path):
    run = build_run(tmp_path)
    summary = report.report(run)
    assert summary['intrinsic_all_eight_pairs']['pairs'] == 8
    assert summary['diagnosed_two_descriptive_only']['pairs'] == 2
    assert summary['remaining_six_descriptive_only']['pairs'] == 6
    assert summary['protected_actual_output']['physical_points'] == 64
    assert summary['protected_actual_output']['sum_worker_elapsed_seconds'] == 48
    outputs = list((run / 'derived_report/protected_outputs').glob('*.json'))
    assert len(outputs) == 8
    for path in outputs:
        value = report.read(path)
        assert value['selected_group'] == report.GROUPS[0]
        assert value['best_plan']['certified_complete'] is True
        assert len(value['source_cell_completion_sha256']) == 2
        assert value['physical_points'] == 8 and value['sum_worker_elapsed_seconds'] == 6
    manifest = report.verify_manifest(run / 'derived_report')
    assert len(manifest) == 12
    with pytest.raises(ValueError, match='overwrite'):
        report.report(run)


def test_changed_result_is_rejected_even_if_old_audit_still_says_pass(tmp_path):
    run = build_run(tmp_path)
    change_json(run / 'episodes/000/result.json', lambda r: r.update(elapsed_seconds=0))
    with pytest.raises(ValueError, match='Changed sealed result'):
        report.report(run)
    assert not (run / 'derived_report').exists()


def test_resealing_changed_cell_cannot_reuse_old_audit(tmp_path):
    run = build_run(tmp_path)
    cell = run / 'episodes/000'
    change_json(cell / 'result.json', lambda r: r.update(elapsed_seconds=0))
    change_json(cell / 'completion_manifest.json', lambda m: m.update({'result.json': report.sha(cell / 'result.json')}))
    with pytest.raises(ValueError, match='Audited cell seal differs'):
        report.report(run)


def test_repeated_audit_cell_does_not_count_as_sixteen(tmp_path):
    run = build_run(tmp_path)
    audit = run / 'audit'
    change_json(audit / 'summary.json', lambda r: r['results'].__setitem__(1, r['results'][0]))
    change_json(audit / 'completion_manifest.json', lambda m: m.update({'summary.json': report.sha(audit / 'summary.json')}))
    with pytest.raises(ValueError, match='omitted/repeated/added'):
        report.report(run)


def test_unclosed_controller_cost_blocks_report_before_publication(tmp_path):
    run = build_run(tmp_path)
    change_json(run / 'controller_completion.json', lambda r: r.update(unknown_cost_cells=['episodes/000']))
    with pytest.raises(ValueError, match='not closed'):
        report.report(run)


def test_missing_checkpoint_cannot_fabricate_early_hit():
    row = case(report.GROUPS[0], [4, 4, 80])
    del row['checkpoints'][1]
    with pytest.raises(ValueError, match='boundary'):
        report.validate_checkpoints(row['checkpoints'])


def test_population_cannot_drop_an_unfavourable_context():
    population = {'contexts': [{'scenario_id': f'fault{i}', 'event_id': [30, 20, 6, 3][i % 4]}
                                for i in range(8)]}
    schedule = [{**p, 'group': g, 'cell': f'{i}_{g}'} for i, p in enumerate(population['contexts'])
                for g in report.GROUPS]
    schedule.pop()
    with pytest.raises(ValueError, match='Missing, repeated or extra'):
        report.validate_population(schedule, population)

