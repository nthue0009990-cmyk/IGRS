"""Reporting checks for complete populations, independent guard cost and causal curves."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import report_clpu_early_pilot as report


def population():
    return [{'cell': f'episodes/{context}_{arm}_{seed}', 'context_id': context,
        'scenario_id': f's{context}', 'event_id': context, 'arm': arm, 'seed': seed,
        'protected': arm != 'window_reference', 'early_policy_expansions': 16,
        'cold_searches': 1 if arm == 'window_reference' else 2}
        for context in range(8) for arm in report.ARMS for seed in report.ARM_SEEDS[arm]]


def rows():
    result = []
    for spec in population():
        row = {**spec, **{key: 0. for key in report.METRICS}, 'fully_certified': True,
            'guard_established': True if spec['protected'] else None, 'policy_decisions': 16 if spec['seed'] else 0}
        row.update(quality_pct=50. + spec['context_id'] + (3 if spec['arm'] == 'protected_ppo' and spec['seed'] == 73 else 0),
            physical_points=80 + (70 if spec['protected'] else 0),
            logical_points=100 + (110 if spec['protected'] else 0),
            numerical_entries=323 + (283 if spec['protected'] else 0), requests=200,
            process_wall_seconds=100. if spec['protected'] else 40., worker_elapsed_seconds=95., computation_elapsed_seconds=90.)
        row['phase_summary'] = {'EARLY': {'physical_points': 80, 'quality_pct': row['quality_pct']}}
        if spec['protected']:
            row['phase_summary']['GUARD'] = {'physical_points': 70, 'logical_points': 110,
                'numerical_entries': 283, 'requests': 100, 'quality_pct': 50. + spec['context_id']}
        row['checkpoints'] = [
            {'phase': 'EARLY', 'physical_points': 1, 'worker_elapsed_seconds': 2., 'quality_pct': None},
            {'phase': 'EARLY', 'physical_points': 80, 'worker_elapsed_seconds': 40., 'quality_pct': row['quality_pct']},
        ]
        if spec['protected']:
            row['checkpoints'].append({'phase': 'GUARD', 'physical_points': 150,
                'worker_elapsed_seconds': 95., 'quality_pct': row['quality_pct']})
        result.append(row)
    return result


def test_population_keeps_every_seed_and_does_not_accept_duplicate_replacement():
    schedule = population()
    report.validate_population(schedule)
    schedule[-1] = deepcopy(schedule[-2])
    with pytest.raises(ValueError):
        report.validate_population(schedule)


def test_population_does_not_silently_remove_failed_context():
    with pytest.raises(ValueError):
        report.validate_population(population()[:-1])


def test_missing_certificate_keeps_context_and_null_full_mean():
    data = rows()
    next(r for r in data if r['arm'] == 'protected_ppo' and r['context_id'] == 0)['quality_pct'] = None
    grouped = report.aggregate_contexts(data)
    target = next(r for r in grouped if r['context_id'] == 0 and r['arm'] == 'protected_ppo')
    assert target['metrics']['quality_pct'] is None
    assert target['metric_coverage']['quality_pct'] == 2
    summary = report.make_summary(data, grouped, report.all_pairs(data), {}, {'wall_seconds': 12.}, 'test')
    assert summary['arms']['protected_ppo']['context_balanced_metrics']['quality_pct']['mean'] is None
    assert len(grouped) == 32


def test_fixed_seed_average_and_all_own_guard_comparisons_are_kept():
    data = rows()
    pairs = report.all_pairs(data)
    assert len(pairs) == 152
    summary = report.stats.comparison_summary(pairs)
    compare = summary['ppo_minus_bc']
    assert compare['pairs'] == 24
    assert compare['contexts'] == 8
    assert compare['context_balanced_delta']['quality_pp']['mean'] == 1.
    assert compare['per_seed']['73']['quality_pp']['mean'] == 3.
    assert compare['per_seed']['17']['quality_pp']['mean'] == 0.


def test_guard_cost_is_real_and_its_process_time_is_not_invented():
    pairs = report.all_pairs(rows())
    own = [p for p in pairs if p['own_paid_baseline']]
    assert len(own) == 56
    assert all(p['signed_delta']['physical_points'] == 80 for p in own)
    assert all(p['signed_delta']['numerical_entries'] == 323 for p in own)
    assert all(p['signed_delta']['process_wall_seconds'] is None for p in own)
    assert all(p['signed_delta']['worker_elapsed_seconds'] is None for p in own)


def test_complete_curves_include_guard_and_early_curves_explicitly_exclude_it():
    data = rows()
    full = report.make_curve_data(data)
    early = report.make_curve_data(data, phase_only=True)
    assert full['axes']['physical_points'] == list(range(513))
    assert early['axes']['physical_points'] == list(range(257))
    assert len(full['cells']) == len(early['cells']) == 64
    f = next(r for r in full['cells'] if r['arm'] == 'protected_ppo')
    e = next(r for r in early['cells'] if r['cell'] == f['cell'])
    assert f['stop_physical_points'] == 150
    assert e['stop_physical_points'] == 80
    assert f['physical']['quality_pct'][79] is None
    assert f['physical']['after_stop_carry'][150] is False
    assert f['physical']['after_stop_carry'][151] is True
    assert e['physical']['after_stop_carry'][81] is True
    assert e['worker_time']['quality_pct'][39] is None


def test_free_same_point_actions_do_not_imply_zero_elapsed_time():
    checkpoints = [{'physical_points': 5, 'worker_elapsed_seconds': 3., 'quality_pct': 20.},
        {'physical_points': 5, 'worker_elapsed_seconds': 4., 'quality_pct': 30.}]
    physical = report.stats.step_curve(checkpoints, [4, 5, 6], 'physical_points', 5)
    timed = report.stats.step_curve(checkpoints, [2, 3, 4], 'worker_elapsed_seconds', 4)
    assert physical['quality_pct'] == [None, 30., 30.]
    assert timed['quality_pct'] == [None, 20., 30.]


def test_comparison_worst_context_and_negative_seed_are_not_hidden():
    data = rows()
    ppo = [r for r in data if r['arm'] == 'protected_ppo' and r['context_id'] == 7]
    for row in ppo:
        row['quality_pct'] -= 10.
    comparisons = report.stats.comparison_summary(report.all_pairs(data))
    q = comparisons['ppo_minus_bc']['context_balanced_delta']['quality_pp']
    assert q['minimum'] == -9.
    assert q['mean'] == -.25
    assert comparisons['ppo_minus_bc']['all_pair_quality_relations']['loss'] == 3


def mock_worker_case(tmp_path, monkeypatch, arm):
    from test_run_clpu_early_pilot import worker_fixture, runner
    run, cell, spec = worker_fixture(tmp_path, monkeypatch, arm)
    assert runner.worker(run, cell) == 0
    result = runner.read(cell / 'result.json')
    audited = {**spec, 'status': 'PASS', 'quality_checkpoints_verified': True,
        'quality_pct': result['quality_pct'], 'guard_established': result['guard_audit']['guard_established'] if spec['protected'] else False,
        'worker_elapsed_seconds': result['elapsed_seconds'], 'process_wall_seconds': result['elapsed_seconds'] + 1.,
        'policy_decisions': result['policy_decisions'], 'checkpoints': deepcopy(result['checkpoints']),
        'physical_statuses': {'PASS': result['actual_physical_points'], 'FAIL': 0},
        **{k: result['oracle'][k] for k in ('logical_points', 'physical_points', 'requests')},
        'numerical_entries': result['numerical_entries']}
    return spec, result, audited


@pytest.mark.parametrize('arm', report.ARMS)
def test_actual_new_worker_schema_is_consumed_without_losing_phase_costs(tmp_path, monkeypatch, arm):
    spec, result, audited = mock_worker_case(tmp_path, monkeypatch, arm)
    row = report.case_row(spec, result, audited)
    assert row['physical_points'] == sum(p['physical_points'] for p in result['phases'].values())
    assert row['numerical_entries'] == 3 * spec['cold_searches'] + 4 * row['physical_points']
    early = report.early_only_rows([row])[0]
    assert early['physical_points'] == result['phases']['EARLY']['physical_points']
    assert early['runtime_initialization_seconds'] == result['phases']['EARLY']['runtime_initialization_seconds']
    assert early['selected_phase'] == 'EARLY' and early['gain_over_paid_baseline_kwh'] is None
    assert early['baseline_quality_pct'] is None and early['guard_established'] is None
    assert early['process_wall_seconds'] is None
    assert 'physical_statuses' not in early and 'semantic_cache_hits' not in early


def test_report_rejects_omitted_guard_cost_even_if_total_still_looks_plausible(tmp_path, monkeypatch):
    spec, result, audited = mock_worker_case(tmp_path, monkeypatch, 'protected_ppo')
    result['oracle']['physical_points'] -= 1
    with pytest.raises(ValueError, match='Phase cost omitted'):
        report.case_row(spec, result, audited)


def test_report_rejects_retroactive_guard_claim_in_early_curve(tmp_path, monkeypatch):
    spec, result, audited = mock_worker_case(tmp_path, monkeypatch, 'protected_rule')
    result['checkpoints'][0]['guard_established'] = True
    audited['checkpoints'] = deepcopy(result['checkpoints'])
    with pytest.raises(ValueError, match='Guard guaranteed before completion'):
        report.case_row(spec, result, audited)
