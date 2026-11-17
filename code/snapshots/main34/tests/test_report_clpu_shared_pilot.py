"""Small synthetic sealed runs; no solver, model, or experiment execution."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import report_clpu_shared_pilot as report
import plot_clpu_shared_pilot as plot


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    report.save(path, value)


def seal(path):
    save(path / 'completion_manifest.json', {p.relative_to(path).as_posix(): report.sha(p)
        for p in path.rglob('*') if p.is_file() and p.name != 'completion_manifest.json'})


def cell_data(spec):
    q, mode = 80. + spec['context_id'], spec['mode']
    shared = mode == 'shared'
    phase = {name: {'quality_pct': q, 'fully_certified': True, 'logical_points': 3,
        'actual_physical_points': 0 if name == 'GUARD' and shared else 2,
        'search_decisions': 1, 'phase_elapsed_seconds': 2., 'stage_component_seconds': {}}
        for name in ('EARLY', 'GUARD')}
    views = {name: {'logical_points': 3, 'physical_points': p['actual_physical_points'], 'requests': 4,
        'actual_numerical_entries_attributed': 4 * p['actual_physical_points'],
        'initialization_numerical_entries_attributed': 0,
        'cross_view_reuses': 3 if name == 'GUARD' and shared else 0} for name, p in phase.items()}
    actual = sum(p['actual_physical_points'] for p in phase.values())
    cps = []
    for name in ('EARLY', 'GUARD'):
        previous_l, previous_p, previous_requests, previous_choices = (0, 0, 0, 0) if name == 'EARLY' else (3, 2, 4, 1)
        for index, kind in enumerate(('INITIAL', 'DECISION', 'CLOSE')):
            choice = min(index, 1)
            local_l = 1 if index == 0 else 3
            local_p = 0 if name == 'GUARD' and shared else 1 if index == 0 else 2
            available = not (name == 'EARLY' and index == 0)
            cps.append({'checkpoint_index': len(cps), 'phase': name, 'kind': kind,
                'phase_choices': choice, 'search_choices': previous_choices + choice,
                'logical_points': previous_l + local_l, 'physical_points': previous_p + local_p,
                'numerical_entries': 3 + 4 * (previous_p + local_p),
                'requests': previous_requests + (1 if index == 0 else 4),
                'phase_logical_points': local_l, 'phase_physical_points': local_p,
                'phase_numerical_entries': 4 * local_p, 'phase_quality_pct': q if available else None,
                'quality_pct': q if available else None, 'fully_certified': available,
                'guard_established': name == 'GUARD' and kind == 'CLOSE',
                'worker_elapsed_seconds': float(len(cps) + 1)})
    service = {'status': 'COMPLETE_SHARED_EVIDENCE_PENDING_AUDIT', 'closed': True,
        'physical_cost_complete': True, 'runtime_initializations': 1, 'mode': mode,
        'physical_points': actual, 'logical_points': 6, 'actual_numerical_entries': 3 + 4 * actual, 'views': views}
    result = {'spec': spec, 'study': report.STUDY, 'new_training': 0, 'policy_decisions': 0,
        'service': service, 'actual_physical_points': actual, 'logical_points': 6,
        'numerical_entries': 3 + 4 * actual, 'requests': 8, 'search_decisions': 2,
        'guard_audit': {'guard_established': True}, 'quality_pct': q, 'fully_certified': True,
        'early_quality_pct': q, 'baseline_quality_pct': q, 'phase_summary': phase, 'checkpoints': cps,
        'worker_elapsed_seconds': 7., 'preflight_elapsed_seconds': 1.,
        'runtime_initialization_seconds': .5, 'computation_elapsed_seconds': 6.}
    physical = {'points': actual, 'actual_numerical_entries': 3 + 4 * actual,
        'statuses': {'PASS': actual - 1, 'FAIL': 1}}
    return result, physical


@pytest.fixture
def frozen(tmp_path):
    run = tmp_path / 'run'
    schedule = []
    results, exits = [], []
    for context in range(8):
        for mode in report.MODES:
            spec = {'cell': f'episodes/{len(schedule):03d}', 'context_id': context,
                'scenario_id': f'N2__line.l{context + 1}__line.l24', 'event_id': context,
                'mode': mode, 'method': 'beam', 'seed': 0, 'arm': 'protected_rule'}
            schedule.append(spec)
            result, physical = cell_data(spec)
            cell = run / spec['cell']
            save(cell / 'spec.json', spec)
            save(cell / 'result.json', result)
            save(cell / 'service_summary.json', result['service'])
            save(cell / 'physical/oracle_summary.json', physical)
            (cell / 'cost_checkpoints.jsonl').write_text(''.join(json.dumps(p) + '\n' for p in result['checkpoints']), encoding='utf-8')
            seal(cell)
            elapsed = 10. + context if mode == 'private' else 9. + context
            if context == 7 and mode == 'shared':
                elapsed = 20.  # An unfavorable paired timing must remain visible.
            audit = {'cell': spec['cell'], 'status': 'PASS', 'result_sha256': report.sha(cell / 'result.json'),
                'cell_completion_sha256': report.sha(cell / 'completion_manifest.json'),
                'process_wall_seconds': elapsed, 'quality_pct': result['quality_pct'], 'guard_established': True,
                'logical_points': 6, 'physical_points': result['actual_physical_points'],
                'numerical_entries': result['numerical_entries']}
            results.append(audit)
            exits.append({'cell': spec['cell'], 'exit_code': 0, 'process_wall_seconds': elapsed})
    save(run / 'schedule.json', schedule)
    save(run / 'config.json', {'shared_study': report.STUDY})
    save(run / 'physical_scope.json', {'fixture': 'fixed'})
    save(run / 'source_binding.json', {'source': 'fixture'})
    names = ['scripts/report_clpu_shared_pilot.py', 'scripts/report_clpu_integrated_pilot.py']
    for name in names:
        target = run / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((report.ROOT / name).read_bytes())
    frozen_files = ['config.json', 'schedule.json', 'physical_scope.json', 'source_binding.json'] + ['source/' + n for n in names]
    save(run / 'prephysical_freeze_manifest.json', {n: report.sha(run / n) for n in frozen_files})
    save(run / 'prepared.json', {'status': 'FROZEN_PREPARED_NO_DSS', 'freeze_sha256': report.sha(run / 'prephysical_freeze_manifest.json'),
        'source_files': names})
    totals = {k: sum(r[k] for r in results) for k in ('physical_points', 'logical_points', 'numerical_entries')}
    save(run / 'controller_completion.json', {'status': 'COMPLETE_CELLS_PENDING_AUDIT', 'scheduled': 16,
        'completed': [s['cell'] for s in schedule], 'exits': exits, 'errors': [], 'unknown_cost_cells': [], 'wall_seconds': 100., **totals})
    save(run / 'audit/summary.json', {'status': 'PASS', 'audited_cells': 16, 'errors': [], 'results': results, **totals,
        'freeze_sha256': report.sha(run / 'prephysical_freeze_manifest.json'), 'config_sha256': report.sha(run / 'config.json'),
        'schedule_sha256': report.sha(run / 'schedule.json'), 'controller_completion_sha256': report.sha(run / 'controller_completion.json')})
    seal(run / 'audit')
    return run


def test_full_sealed_report_keeps_all_pairs_failures_and_ratio_of_means(frozen):
    summary = report.report(frozen)
    assert summary['status'] == report.STATUS and summary['cells'] == 16
    assert summary['physical_statuses']['FAIL'] == 16
    assert summary['source_bindings']['audit/summary.json'] == report.sha(frozen / 'audit/summary.json')
    pairs = report.read(frozen / 'derived_report/all_pairs.json')
    assert len(pairs) == 8 and pairs[-1]['signed_delta']['process_wall_seconds'] == 3
    comp = summary['comparison']
    a, b = [summary['modes'][m]['metrics']['process_wall_seconds']['mean'] for m in report.MODES]
    assert comp['relative_change_pct']['process_wall_seconds'] == 100 * (b / a - 1)
    assert comp['signed_delta']['process_wall_seconds']['maximum'] == 3
    assert summary['modes']['private']['metrics']['physical_points']['mean'] == 4
    assert summary['modes']['shared']['metrics']['physical_points']['mean'] == 2
    assert summary['total_cost']['numerical_entries'] == 16 * 3 + 4 * 48
    report.manifest(frozen / 'derived_report', all_files=True)
    curves = report.read(frozen / 'derived_report/curve_data.json')
    assert not curves['sampled_grid'] and len(curves['cells']) == 16
    assert all(c['checkpoints'][0]['quality_pct'] is None for c in curves['cells'])


def test_missing_certificate_and_negative_quality_pair_are_retained(frozen):
    cases, _, _ = report.load_cases(frozen)
    cases[1]['quality_pct'] = None
    cases[3]['quality_pct'] = cases[2]['quality_pct'] - 2
    modes, pairs, comp = report.summarize(cases)
    assert modes['shared']['metrics']['quality_pct']['mean'] is None
    assert modes['shared']['metrics']['quality_pct']['coverage_count'] == 7
    assert pairs[0]['quality_relation'] == 'missing' and pairs[1]['quality_relation'] == 'worse'
    assert pairs[1]['signed_delta']['quality_pct'] == -2
    assert comp['signed_delta']['quality_pct']['mean'] is None and len(pairs) == 8


@pytest.mark.parametrize('target', ['freeze', 'result', 'cell_seal', 'audit', 'controller'])
def test_changed_bound_evidence_blocks_report(frozen, target):
    paths = {'freeze': 'physical_scope.json', 'result': 'episodes/000/result.json',
        'cell_seal': 'episodes/000/completion_manifest.json', 'audit': 'audit/summary.json', 'controller': 'controller_completion.json'}
    p = frozen / paths[target]
    p.write_text(p.read_text(encoding='utf-8') + ' ', encoding='utf-8')
    with pytest.raises(ValueError):
        report.report(frozen)
    assert not (frozen / 'derived_report').exists()


def test_nonzero_real_exit_cannot_be_reported_even_with_matching_hashes(frozen):
    path = frozen / 'controller_completion.json'
    c = report.read(path); c['exits'][0]['exit_code'] = 1; save(path, c)
    audit = report.read(frozen / 'audit/summary.json')
    audit['controller_completion_sha256'] = report.sha(path)
    save(frozen / 'audit/summary.json', audit); seal(frozen / 'audit')
    with pytest.raises(ValueError, match='really exit'):
        report.report(frozen)


@pytest.mark.parametrize('change', ['missing_mode', 'context', 'learned'])
def test_population_cannot_select_or_replace_cases(frozen, change):
    cases, _, _ = report.load_cases(frozen)
    if change == 'missing_mode': cases.pop()
    elif change == 'context': cases[0]['context_id'] = 1
    else: cases[0]['method'] = 'ppo_raw'
    with pytest.raises(ValueError):
        report.summarize(cases)


def test_checkpoint_cost_and_missing_quality_are_checked(frozen):
    result = report.read(frozen / 'episodes/000/result.json')
    points = deepcopy(result['checkpoints']); points[0]['quality_pct'] = 0
    with pytest.raises(ValueError, match='certificate'):
        report.checkpoints(points, result)
    points = deepcopy(result['checkpoints']); points[1]['physical_points'] = 99
    with pytest.raises(ValueError, match='costs'):
        report.checkpoints(points, result)


def test_manifest_cannot_escape_or_alias_paths(tmp_path):
    folder = tmp_path / 'seal'; folder.mkdir()
    save(folder / 'completion_manifest.json', {'../escape.json': '0' * 64})
    with pytest.raises(ValueError, match='Unsafe'):
        report.manifest(folder)


def test_plot_inputs_recompute_means_and_refuse_incomplete_or_changed_report(frozen):
    report.report(frozen)
    folder = frozen / 'derived_report'
    summary, cases = plot.load_inputs(folder)
    assert len(cases) == 16 and summary['contexts'] == 8
    c = report.read(folder / 'all_cells.json'); c[0]['physical_points'] = 0; save(folder / 'all_cells.json', c)
    seal(folder)
    with pytest.raises(ValueError, match='means or paired'):
        plot.load_inputs(folder)


def test_plot_passes_all_exact_cost_values_without_smoothing_or_filtering(frozen, monkeypatch):
    cases, _, _ = report.load_cases(frozen)
    seen = []
    class Axis:
        def scatter(self, x, y, **kwargs): seen.append((list(x), list(y), kwargs['label']))
        def set_xticks(self, *a, **k): pass
        def set_ylabel(self, *a): pass
        def set_ylim(self, *a, **k): pass
        def grid(self, *a, **k): pass
        def get_legend_handles_labels(self): return [], []
    class Figure:
        def legend(self, *a, **k): pass
        def suptitle(self, *a, **k): pass
        def text(self, *a, **k): pass
        def tight_layout(self, *a, **k): pass
    plt = SimpleNamespace(subplots=lambda *a, **k: (Figure(), [Axis(), Axis()]))
    exported = []
    monkeypatch.setattr(plot.plotting, 'export', lambda *args: exported.append(args[2]))
    plot.draw_costs(cases, frozen, plt)
    assert len(seen) == 4 and all(len(row[1]) == 8 for row in seen)
    for index, metric in enumerate(('physical_points', 'process_wall_seconds')):
        for offset, mode in enumerate(report.MODES):
            expected = [next(c[metric] for c in cases if c['context_id'] == context and c['mode'] == mode) for context in range(8)]
            assert seen[index * 2 + offset][1] == expected
    assert exported == ['01_all_context_shared_costs']
