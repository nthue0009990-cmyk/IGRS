"""Small sealed fixtures, seed weighting, phase curves and real fake-service IO.

The service in the cross-component check uses the existing counted test factory;
no native physical runtime, model checkpoint or power flow is executed.
"""
from copy import deepcopy
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import report_clpu_deferred_pilot as report
import plot_clpu_deferred_pilot as plot
from test_report_clpu_shared_pilot import cell_data as shared_cell_data


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    report.save(path, value)


def seal(path):
    save(path / 'completion_manifest.json', {p.relative_to(path).as_posix(): report.sha(p)
        for p in path.rglob('*') if p.is_file() and p.name != 'completion_manifest.json'})


def schedule():
    rows = []
    for context in range(8):
        for arm in report.ARMS:
            for seed in report.ARM_SEEDS[arm]:
                learned = arm in report.ARMS[2:]
                rows.append({'cell': f'episodes/{len(rows):03d}', 'context_id': context,
                    'scenario_id': f'N2__l{context}__l24', 'event_id': context,
                    'arm': arm, 'seed': seed, 'mode': 'shared', 'protected': True,
                    'method': 'bc_raw' if arm == 'deferred_bc' else 'ppo_raw' if learned else 'beam',
                    'macro_schedule': 'interleaved' if arm == 'interleaved_window' else 'deferred',
                    'checkpoint_sha256': 'a' * 64 if learned else None})
    return rows


def data(spec):
    result, physical = shared_cell_data(spec)
    learned = spec['arm'] in report.ARMS[2:]
    result.update(study=report.STUDY, schema='clpu-deferred-protected-result-v1',
        checkpoint_sha256=spec['checkpoint_sha256'], model_load_seconds=.2 if learned else 0.,
        policy_decisions=int(learned))
    result['guard_audit'].update(final_energy_kwh=80., baseline_energy_kwh=80.)
    for phase, row in result['phase_summary'].items():
        row.update(policy_decisions=int(learned and phase == 'EARLY'),
            graph_expansions=0 if phase == 'EARLY' else 1,
            verify_attempts=1 if phase == 'EARLY' else None)
    return result, physical


def audited_fields(result, physical):
    return {'guard_quality_pct': result['baseline_quality_pct'],
        'requests': result['requests'], 'worker_elapsed_seconds': result['worker_elapsed_seconds'],
        'model_load_seconds': result['model_load_seconds'], 'physical_statuses': physical['statuses'],
        'phase_summary': deepcopy(result['phase_summary']), 'exact_checkpoints': deepcopy(result['checkpoints'])}


@pytest.fixture
def frozen(tmp_path):
    run = tmp_path / 'run'
    specs, audits, exits = schedule(), [], []
    for spec in specs:
        cell = run / spec['cell']
        result, physical = data(spec)
        for name, value in (('spec.json', spec), ('result.json', result),
                ('service_summary.json', result['service']), ('physical/oracle_summary.json', physical)):
            save(cell / name, value)
        (cell / 'cost_checkpoints.jsonl').write_text(''.join(json.dumps(p) + '\n' for p in result['checkpoints']), encoding='utf-8')
        seal(cell)
        elapsed = 10. + spec['context_id']
        if spec['arm'] == 'deferred_ppo':
            elapsed += {17: -1., 41: 1., 73: 3.}[spec['seed']]
            if spec['context_id'] == 7:
                elapsed += 6.  # A bad context and a bad seed must remain visible.
        check = {'cell': spec['cell'], 'status': 'PASS',
            'result_sha256': report.sha(cell / 'result.json'),
            'cell_completion_sha256': report.sha(cell / 'completion_manifest.json'),
            'process_wall_seconds': elapsed, 'worker_elapsed_seconds': result['worker_elapsed_seconds'],
            'quality_pct': result['quality_pct'], 'guard_established': True,
            'logical_points': result['logical_points'], 'physical_points': result['actual_physical_points'],
            'numerical_entries': result['numerical_entries'], **audited_fields(result, physical)}
        audits.append(check)
        exits.append({'cell': spec['cell'], 'exit_code': 0, 'process_wall_seconds': elapsed})
    for name, value in (('schedule.json', specs), ('config.json', {'deferred_study': report.STUDY}),
            ('physical_scope.json', {'fixture': 'fixed'}), ('source_binding.json', {'fixture': 'two sealed sources'})):
        save(run / name, value)
    names = ['scripts/report_clpu_deferred_pilot.py', 'scripts/report_clpu_integrated_pilot.py', 'scripts/report_clpu_shared_pilot.py']
    for name in names:
        target = run / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    inputs = ['config.json', 'schedule.json', 'physical_scope.json', 'source_binding.json'] + ['source/' + n for n in names]
    save(run / 'prephysical_freeze_manifest.json', {n: report.sha(run / n) for n in inputs})
    save(run / 'prepared.json', {'status': 'FROZEN_PREPARED_NO_DSS',
        'freeze_sha256': report.sha(run / 'prephysical_freeze_manifest.json'), 'source_files': names})
    totals = {k: sum(r[k] for r in audits) for k in ('physical_points', 'logical_points', 'numerical_entries')}
    save(run / 'controller_completion.json', {'status': 'COMPLETE_CELLS_PENDING_AUDIT', 'scheduled': 64,
        'completed': [r['cell'] for r in specs], 'exits': exits, 'errors': [], 'unknown_cost_cells': [],
        'wall_seconds': 100., **totals})
    save(run / 'audit/summary.json', {'status': 'PASS', 'audited_cells': 64, 'errors': [], 'results': audits, **totals,
        'freeze_sha256': report.sha(run / 'prephysical_freeze_manifest.json'),
        'config_sha256': report.sha(run / 'config.json'), 'schedule_sha256': report.sha(run / 'schedule.json'),
        'controller_completion_sha256': report.sha(run / 'controller_completion.json')})
    seal(run / 'audit')
    return run


def test_all64_sealed_report_keeps_seed_context_pair_denominators_and_bad_results(frozen):
    summary = report.report(frozen)
    output = frozen / 'derived_report'
    assert summary['cells'] == summary['audited_cells'] == 64 and summary['contexts'] == 8
    assert summary['physical_statuses']['FAIL'] == 64
    assert len(report.read(output / 'all_pairs.json')) == 80
    assert len(report.read(output / 'context_pairs.json')) == 32
    assert len(report.read(output / 'context_summary.json')) == 32
    assert summary['arms']['deferred_ppo']['cells'] == 24
    timing = summary['comparisons']['ppo_minus_bc']['metrics']['process_wall_seconds']
    assert timing['signed_delta']['mean'] == pytest.approx(1.75)
    assert timing['worst_signed_delta'] == 7 and timing['worst_contexts'] == [7]
    assert timing['win_tie_loss_missing'] == {'win': 0, 'tie': 0, 'loss': 8, 'missing': 0}
    assert summary['source_bindings']['audit/summary.json'] == report.sha(frozen / 'audit/summary.json')
    assert summary['source_bindings']['controller_completion.json'] == report.sha(frozen / 'controller_completion.json')
    report.shared.manifest(output, all_files=True)
    plot.load_inputs(output)


def test_missing_seed_certificate_nulls_context_and_population_mean_without_dropping(frozen):
    rows, binding, controller = report.load_cases(frozen)
    next(r for r in rows if r['context_id'] == 0 and r['arm'] == 'deferred_ppo' and r['seed'] == 73)['quality_pct'] = None
    contexts = report.aggregate_contexts(rows)
    context = next(r for r in contexts if r['context_id'] == 0 and r['arm'] == 'deferred_ppo')
    assert context['metrics']['quality_pct'] is None and context['metric_coverage']['quality_pct'] == 2
    _, _, comparisons = report.comparisons(rows, contexts)
    values = comparisons['ppo_minus_bc']['metrics']['quality_pct']
    assert values['signed_delta']['mean'] is None and values['signed_delta']['missing_count'] == 1
    assert values['win_tie_loss_missing']['missing'] == 1 and values['worst_signed_delta'] is None
    summary = report.make_summary(rows, contexts, comparisons, binding, controller, frozen.name)
    assert summary['arms']['deferred_ppo']['context_balanced_metrics']['quality_pct']['mean'] is None
    assert len(rows) == 64


def test_small_cost_difference_is_not_erased_by_quality_tolerance():
    assert report.relation(1e-12, 'quality_pct') == 'tie'
    assert report.relation(1e-12, 'process_wall_seconds') == 'loss'
    assert report.relation(-1e-12, 'physical_points') == 'win'


@pytest.mark.parametrize('target', ['freeze', 'audit', 'controller', 'result', 'cell_seal'])
def test_changed_bound_evidence_blocks_output_creation(frozen, target):
    names = {'freeze': 'physical_scope.json', 'audit': 'audit/summary.json',
        'controller': 'controller_completion.json', 'result': 'episodes/000/result.json',
        'cell_seal': 'episodes/000/completion_manifest.json'}
    path = frozen / names[target]
    path.write_text(path.read_text(encoding='utf-8') + ' ', encoding='utf-8')
    with pytest.raises(ValueError):
        report.report(frozen)
    assert not (frozen / 'derived_report').exists()


@pytest.mark.parametrize('mutation', ['omit', 'duplicate_seed', 'cross_fault'])
def test_population_rejects_missing_seed_or_changed_context(mutation):
    rows = schedule()
    if mutation == 'omit':
        rows.pop()
    elif mutation == 'duplicate_seed':
        rows[-1]['seed'] = 17
    else:
        rows[-1]['scenario_id'] = 'other'
    with pytest.raises(ValueError):
        report.population(rows)


def test_nonzero_actual_exit_rejected_even_when_audit_hash_is_updated(frozen):
    path = frozen / 'controller_completion.json'
    value = report.read(path)
    value['exits'][0]['exit_code'] = 1
    save(path, value)
    audit = report.read(frozen / 'audit/summary.json')
    audit['controller_completion_sha256'] = report.sha(path)
    save(frozen / 'audit/summary.json', audit)
    seal(frozen / 'audit')
    with pytest.raises(ValueError, match='successful exit'):
        report.report(frozen)


def test_full_early_curves_use_exact_worker_boundaries_keep_gap_and_stop_carry(frozen):
    rows, _, _ = report.load_cases(frozen)
    full, early = report.make_curves(rows, early=False), report.make_curves(rows, early=True)
    assert len(full['axes']['physical_points']) == 513 and len(early['axes']['physical_points']) == 257
    assert early['axes']['worker_elapsed_seconds'] == [0., 1., 2., 3.]
    assert full['axes']['worker_elapsed_seconds'] == [0., 1., 2., 3., 4., 5., 6.]
    assert early['cells'][0]['worker_time']['quality_pct'][:2] == [None, None]
    assert early['cells'][0]['physical']['after_stop_carry'][3] is True
    assert all(p['phase'] == 'EARLY' and not p['guard_established'] for p in early['cells'][0]['exact_checkpoints'])
    plot.validate_curves(full, rows, early=False)
    plot.validate_curves(early, rows, early=True)
    broken = deepcopy(early)
    broken['cells'][0]['exact_checkpoints'][-1]['phase'] = 'GUARD'
    with pytest.raises(ValueError, match='checkpoints'):
        plot.validate_curves(broken, rows, early=True)


def test_renderer_uses_post_steps_and_does_not_fill_missing_certificates():
    calls = []
    axis = SimpleNamespace(step=lambda *args, **kw: calls.append((args, kw)),
                           set_ylim=lambda *a: None, grid=lambda **kw: None)
    rows = [{'arm': arm, 'physical': {'quality_pct': [None, 50., 60.]}} for arm in report.ARMS]
    plot.curve_lines(axis, rows, 'physical', [0, 1, 2])
    assert len(calls) == 4 and all(kw['where'] == 'post' for _, kw in calls)
    assert all(str(args[1][0]) == 'nan' for args, _ in calls)


@pytest.mark.parametrize('arm', ['interleaved_window', 'deferred_window', 'deferred_bc', 'deferred_ppo'])
def test_actual_runner_routes_schema_with_counted_fake_service(build, arm):
    from test_run_clpu_deferred_pilot import route_fixture
    from dnr.nature34_clpu_early import merge_results
    service, spec, phases, points, policy = route_fixture(build, arm)
    early, guard = phases['EARLY'], phases['GUARD']
    merged = merge_results(early, guard, protected=True)
    chosen = phases[merged['selected_phase']]
    worker = points[-1]['worker_elapsed_seconds'] + .1
    result = {**merged, 'spec': spec, 'study': report.STUDY,
        'schema': 'clpu-deferred-protected-result-v1', 'quality_pct': chosen['quality_pct'],
        'fully_certified': chosen['fully_certified'], 'new_training': 0,
        'checkpoint_sha256': spec['checkpoint_sha256'], 'service': service.summary(),
        'actual_physical_points': service.physical_points,
        'logical_points': sum(p['logical_points'] for p in phases.values()),
        'numerical_entries': service.factory.stats()['returned']['numerical_solution_entries'],
        'requests': sum(p['oracle']['requests'] for p in phases.values()),
        'search_decisions': sum(p['search_decisions'] for p in phases.values()),
        'policy_decisions': early['policy_decisions'], 'early_quality_pct': early['quality_pct'],
        'baseline_quality_pct': guard['quality_pct'], 'phase_summary': phases, 'checkpoints': points,
        'worker_elapsed_seconds': worker, 'model_load_seconds': .001 if policy else 0.,
        'preflight_elapsed_seconds': .002, 'runtime_initialization_seconds': .0001,
        'computation_elapsed_seconds': worker - .002}
    audited = {'cell': spec['cell'], 'status': 'PASS', 'guard_established': True,
        'quality_pct': result['quality_pct'], 'physical_points': result['actual_physical_points'],
        'logical_points': result['logical_points'], 'numerical_entries': result['numerical_entries'],
        'process_wall_seconds': worker + .1}
    physical = report.read(service.cell / 'physical/oracle_summary.json')
    # This older fake delegate deliberately omits status counts from its small
    # summary. Derive them from its retained counted output, not assumed PASS.
    from test_nature34_clpu_equivalent_cache import lines
    physical['statuses'] = dict(Counter(r['status'] for r in lines(service.cell / 'physical/points.jsonl.gz')))
    audited.update(audited_fields(result, physical))
    row = report.case_row(spec, result, audited,
        {'cell': spec['cell'], 'exit_code': 0, 'process_wall_seconds': worker + .1}, physical)
    assert row['guard_established'] and row['early_total_decisions'] <= 33
    assert row['phase_summary']['GUARD']['verify_attempts'] is None
    assert row['policy_decisions'] == (early['verify_attempts'] if policy else 0)
    assert row['guard_physical_points'] + row['early_physical_points'] == row['physical_points']


# Existing shared-service fixture replaces the physical factory with a counted
# deterministic test implementation. No new service behavior is invented here.
from test_nature34_clpu_shared_evidence import build
