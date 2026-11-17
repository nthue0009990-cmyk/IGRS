"""Synthetic crossed-design reporting checks; no holdout data or physics."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import report_clpu_public_confirmation as report


def rows():
    data = []
    for fault in range(4):
        for event in range(6):
            context = fault * 6 + event
            for arm in report.ARMS:
                row = {key: 1. for key in report.METRICS}
                row.update(cell=f'episodes/{len(data):03d}', context_id=context, scenario_id=f'fault{fault}',
                    event_id=event, arm=arm, seed=0, method='beam', guard_energy_kwh=5.,
                    quality_pct=75., final_energy_kwh=10., physical_points=20,
                    process_wall_seconds=30.)
                data.append(row)
    return data


@pytest.mark.parametrize('change', ('missing', 'duplicate_arm', 'unpaired_event', 'duplicated_context', 'wrong_grid'))
def test_complete_crossed_population_cannot_drop_or_replace_any_pair(change):
    data = rows()
    if change == 'missing':
        data.pop()
    elif change == 'duplicate_arm':
        data[1]['arm'] = 'P'
    elif change == 'unpaired_event':
        data[1]['event_id'] += 1
    elif change == 'duplicated_context':
        for row in data[-2:]:
            row.update(scenario_id='fault0', event_id=0)
    else:
        for row in data[-2:]:
            row['scenario_id'] = 'fault_extra'
    with pytest.raises(ValueError):
        report.population(data)


def test_ratio_of_means_exact_energy_and_both_cluster_margins():
    data = rows()
    p, e = data[:2]
    p.update(physical_points=90, final_energy_kwh=10. + 1e-12)
    e['physical_points'] = 100
    groups, pairs, bootstrap, decision = report.summarize(data, bootstrap_replicates=50)
    primary = groups['overall']['P_minus_E']['physical_points']
    assert primary['ratio_of_means_P_over_E'] == pytest.approx(550 / 560)
    mean_pair_ratio = sum(p['physical_points_pair_ratio_P_over_E'] for p in pairs) / 24
    assert abs(primary['ratio_of_means_P_over_E'] - mean_pair_ratio) > .01
    assert pairs[0]['signed_delta']['quality_pct'] == 0
    assert pairs[0]['p_relations']['quality_pct'] == 'win'
    assert groups['overall']['P_minus_E']['quality_pct']['P_win_tie_loss_missing'] == {'win': 1, 'tie': 23, 'loss': 0, 'missing': 0}
    assert len(groups['by_fault']) == 4 and all(x['denominator_pairs'] == 6 for x in groups['by_fault'])
    assert len(groups['by_event']) == 6 and all(x['denominator_pairs'] == 4 for x in groups['by_event'])
    assert decision['all_24_exact_energy_pairs_nondegrading'] is True
    assert bootstrap['independent_pair_resampling'] is False


def test_missing_all_optional_certificates_and_one_primary_value_remain_null():
    data = rows()
    for row in data:
        row['early_quality_pct'] = None
    data[0]['physical_points'] = None
    groups, pairs, boot, decision = report.summarize(data, bootstrap_replicates=25)
    early = groups['overall']['arms']['P']['metrics']['early_quality_pct']
    assert early['mean'] is None and early['missing'] == early['denominator'] == 24
    assert early['min_observed'] is None
    pf = groups['overall']['P_minus_E']['physical_points']
    assert pf['P_mean'] is pf['ratio_of_means_P_over_E'] is None
    assert pf['P_win_tie_loss_missing']['missing'] == 1
    assert len(pairs) == 24 and pairs[0]['P']['physical_points'] is None
    interval = boot['metrics']['physical_points']['ratio_of_means_P_over_E']
    assert interval['percentile_95_interval'] is None and interval['undefined_replicates'] == 25
    assert interval['interval_status'] == 'SUPPRESSED_MISSING_INPUT'
    assert decision['same_or_higher_quality_with_lower_PF_descriptive_condition'] is False


def test_two_cluster_weights_are_cartesian_multiplicities_not_iid_pairs():
    weights = list(report.cluster_weights(60, 20261024))
    assert weights == list(report.cluster_weights(60, 20261024))
    assert weights != list(report.cluster_weights(60, 20261025))
    for sample in weights:
        assert len(sample) == 24 and sum(sample) == 24
        matrix = [sample[f * 6:(f + 1) * 6] for f in range(4)]
        # Cartesian fault/event cluster weights have rank one and correct margins.
        assert all(sum(row) % 6 == 0 for row in matrix)
        assert all(sum(matrix[f][e] for f in range(4)) % 4 == 0 for e in range(6))
        assert all(matrix[f][e] * matrix[g][j] == matrix[f][j] * matrix[g][e]
                   for f in range(4) for g in range(4) for e in range(6) for j in range(6))


def test_constant_paired_effect_has_exact_cluster_interval_and_frozen_defaults():
    data = rows()
    for row in data:
        row['physical_points'] = 10 if row['arm'] == 'P' else 20
    groups, pairs, bootstrap, decision = report.summarize(data)
    interval = bootstrap['metrics']['physical_points']['ratio_of_means_P_over_E']
    assert interval['percentile_95_interval'] == [.5, .5]
    assert interval['defined_replicates'] == bootstrap['replicates'] == 10000
    assert bootstrap['seed'] == 20261024
    assert decision['primary_PF_saving_pct'] == 50
    assert report.STATISTICAL_RULES['bootstrap']['replicates'] == 10000
    assert report.quantile([0, 10], .25) == 2.5


def test_zero_denominator_resamples_are_reported_not_removed_or_replaced():
    data = rows()
    for row in data:
        row['physical_points'] = 1 if row['arm'] == 'P' else int(row['context_id'] == 0)
    _, _, bootstrap, _ = report.summarize(data, bootstrap_replicates=100)
    interval = bootstrap['metrics']['physical_points']['ratio_of_means_P_over_E']
    assert 0 < interval['undefined_replicates'] < 100
    assert interval['defined_replicates'] + interval['undefined_replicates'] == 100
    assert interval['percentile_95_interval'] is None
    assert interval['interval_status'] == 'SUPPRESSED_UNDEFINED_REPLICATE'


@pytest.mark.parametrize('status,count', [('FAIL', 48), ('PASS', 47)])
def test_report_rejects_incomplete_or_failed_audit_before_opening_outcomes(tmp_path, status, count):
    (tmp_path / 'audit').mkdir()
    (tmp_path / 'audit/summary.json').write_text(json.dumps({'status': status, 'errors': [],
        'audited_cells': count, 'resolved_slots': count, 'actual_tasks': count}), encoding='utf-8')
    with pytest.raises(ValueError, match='complete PASS'):
        report.load_cases(tmp_path)


def test_exact_energy_loss_is_not_hidden_by_equal_rounded_quality():
    data = rows()
    data[-2]['final_energy_kwh'] -= 1e-12
    groups, pairs, bootstrap, decision = report.summarize(data, bootstrap_replicates=30)
    assert pairs[-1]['p_relations']['quality_pct'] == 'loss'
    assert groups['overall']['P_minus_E']['quality_pct']['signed_delta']['mean'] == 0
    assert decision['all_24_exact_energy_pairs_nondegrading'] is False
    assert decision['worst_exact_energy_delta_kwh'] < 0


def test_report_source_identity_clone_does_not_change_old_report():
    assert report.case_row.__globals__['STUDY'] == report.STUDY
    assert report.case_row.__globals__['SCHEMA'] == report.SCHEMA
    assert report.previous.case_row.__globals__['STUDY'] == 'CLPU_PUBLIC_MECHANISM_COMMON_PILOT_V1'
    assert report.previous.ARMS == ('P', 'E', 'A_prune')


def test_analysis_code_must_match_its_preaccess_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(report, 'ROOT', tmp_path)
    monkeypatch.setattr(report, 'IMPLEMENTATIONS', ('scripts/analysis.py',))
    current = tmp_path / 'scripts/analysis.py'
    current.parent.mkdir()
    current.write_text('frozen = True\n', encoding='utf-8')
    run = tmp_path / 'runs/synthetic'
    frozen = run / 'source/scripts/analysis.py'
    frozen.parent.mkdir(parents=True)
    frozen.write_bytes(current.read_bytes())
    report.save(run / 'method_freeze_manifest.json', {'source/scripts/analysis.py': report.sha(frozen)})
    report.check_analysis_freeze(run)
    current.write_text('frozen = False\n', encoding='utf-8')
    with pytest.raises(ValueError, match='pre-access method freeze'):
        report.check_analysis_freeze(run)


@pytest.mark.parametrize('executed', (1, 48))
def test_status_export_keeps_missing_or_all_failed_cells_and_never_reads_quality(tmp_path, monkeypatch, executed):
    monkeypatch.setattr(report, 'ROOT', tmp_path)
    monkeypatch.setattr(report, 'IMPLEMENTATIONS', ())
    monkeypatch.setattr(report, 'check_analysis_freeze', lambda run: None)
    run = tmp_path / 'runs/synthetic'
    run.mkdir(parents=True)
    schedule = [{k: row[k] for k in ('cell', 'context_id', 'scenario_id', 'event_id', 'arm', 'seed', 'method')} for row in rows()]
    report.save(run / 'schedule.json', schedule)
    report.save(run / 'config.json', {'study': report.STUDY, 'confirmation_statistics': report.STATISTICAL_RULES})
    report.save(run / 'method_freeze_manifest.json', {})
    controller = {'status': 'STOPPED_INCOMPLETE', 'dispatched': schedule[:executed],
                  'exits': [{'cell': spec['cell'], 'exit_code': 1, 'process_wall_seconds': 3.} for spec in schedule[:executed]]}
    report.save(run / 'controller_completion.json', controller)
    for spec in schedule[:executed]:
        cell = run / spec['cell']
        (cell / 'physical').mkdir(parents=True)
        report.save(cell / 'physical/oracle_summary.json', {'points': 2, 'actual_numerical_entries': 11})
        report.save(cell / 'service_summary.json', {'logical_points': 3, 'requests': 4})
        # Malformed on purpose: the status exporter must never open these outcomes.
        (cell / 'result.json').write_text('DO NOT READ QUALITY', encoding='utf-8')
    result = report.status_report(run)
    saved_rows = report.read(run / 'status_report/all_cells.json')
    assert result['status'] == 'INCOMPLETE_OR_UNAUDITED_PUBLIC_CONFIRMATION_NO_ESTIMATES'
    assert result['retained_cells'] == len(saved_rows) == 48 and result['retained_pairs'] == 24
    assert all(value is None for value in result['full_population_estimands'].values())
    assert all(row['quality_pct'] is None and row['final_energy_kwh'] is None for row in saved_rows)
    assert all(not row['quality_or_certificate_results_read'] for row in saved_rows)
    observed = result['unverified_cost_observations']['physical_points']
    assert observed['observed_cells'] == executed and observed['missing_cells'] == 48 - executed
    assert observed['unverified_observed_sum'] == 2 * executed
    assert observed['full_population_total'] is observed['full_population_mean'] is None
    assert result['confirmation_success'] is False
    if executed < 48:
        assert saved_rows[-1]['recorded_costs']['physical_points'] is None


def test_frozen_interval_rule_does_not_accept_ratio_one_as_saving():
    groups, pairs, bootstrap, decision = report.summarize(rows(), bootstrap_replicates=20)
    assert decision['all_24_exact_energy_pairs_nondegrading'] is True
    assert decision['primary_PF_ratio_P_over_E'] == 1
    assert decision['primary_PF_ratio_interval_upper_strictly_below_one'] is False
    assert decision['same_quality_PF_saving_supported_under_frozen_interval_rule'] is False
