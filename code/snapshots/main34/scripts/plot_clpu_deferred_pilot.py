"""Full-population figures from the sealed deferred-verification report."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import report_clpu_deferred_pilot as reporting
import plot_clpu_integrated_pilot as plotting

read, save, sha, require = reporting.read, reporting.save, reporting.sha, reporting.require
ARMS = reporting.ARMS
LABELS = {'interleaved_window': 'Interleaved window', 'deferred_window': 'Deferred window',
          'deferred_bc': 'Fixed BC ranking', 'deferred_ppo': 'Fixed PPO ranking'}
COLORS = dict(zip(ARMS, ('#666666', '#0072B2', '#009E73', '#D55E00')))
STYLES = dict(zip(ARMS, (':', '-', '--', '-.')))
INPUTS = ('summary.json', 'all_cells.json', 'context_summary.json', 'curve_data.json', 'early_curve_data.json')


def validate_curves(curves, cases, *, early):
    require(curves['schema'] == 'clpu-deferred-checkpoint-curves-v1'
            and curves['early_phase_only'] is early, 'Curve stage/schema changed')
    reporting.population(curves['cells'])
    lookup = {r['cell']: r for r in cases}
    cap = 256 if early else 512
    require(curves['axes']['physical_points'] == list(range(cap + 1)), 'Wrong full/early physical axis')
    times = curves['axes']['worker_elapsed_seconds']
    require(times and times[0] == 0 and times == sorted(set(times)), 'Time axis is not exact ordered boundaries')
    expected_times = {0.}
    for row in curves['cells']:
        case = lookup[row['cell']]
        require(all(row[k] == case[k] for k in ('context_id', 'scenario_id', 'event_id', 'arm', 'seed')),
                'Curve cell identity changed')
        expected = [p for p in case['checkpoints'] if not early or p['phase'] == 'EARLY']
        require(row['exact_checkpoints'] == expected and expected, 'Curve omitted or altered measured checkpoints')
        require({p['phase'] for p in expected} == ({'EARLY'} if early else {'EARLY', 'GUARD'}),
                'Full and early phases mixed or omitted')
        expected_times.update(p['worker_elapsed_seconds'] for p in expected)
        if early:
            require(all(not p['guard_established'] for p in expected), 'EARLY curve falsely establishes Beam floor')
        require(0 <= row['stop_physical_points'] <= cap, 'Declared curve cost exceeds stage budget')
        for name, axis, field, stop in (
            ('physical', curves['axes']['physical_points'], 'physical_points', row['stop_physical_points']),
            ('worker_time', times, 'worker_elapsed_seconds', row['stop_worker_elapsed_seconds'])):
            require(row[name] == reporting.stats.step_curve(expected, axis, field, stop),
                    'Stored curve is not the measured post-step function')
    require(times == sorted(expected_times), 'Curve time grid omits or invents a boundary')
    contexts = curves['context_curves']
    require(len(contexts) == 32 and {(r['context_id'], r['arm']) for r in contexts}
        == {(c, a) for c in range(8) for a in ARMS}, 'Missing context/group curve')
    for row in contexts:
        chosen = [r for r in curves['cells'] if r['context_id'] == row['context_id'] and r['arm'] == row['arm']]
        require(row['seeds'] == list(reporting.ARM_SEEDS[row['arm']]), 'Curve picked or omitted a seed')
        for axis in ('physical', 'worker_time'):
            require(row[axis] == reporting.stats.average_curves(chosen, axis), 'Context mean curve changed')
    require(len(curves['group_curves']) == 4 and {r['arm'] for r in curves['group_curves']} == set(ARMS),
            'Missing group mean curve')
    for row in curves['group_curves']:
        chosen = [r for r in contexts if r['arm'] == row['arm']]
        for axis in ('physical', 'worker_time'):
            require(row[axis] == reporting.stats.average_curves(chosen, axis, contexts=True),
                    'Eight-context mean curve changed')


def load_inputs(report):
    reporting.shared.manifest(report, required=INPUTS, all_files=True)
    summary, cases, contexts, full, early = (read(report / n) for n in INPUTS)
    require(summary['status'] == reporting.STATUS and summary['cells'] == summary['audited_cells'] == 64
            and summary['contexts'] == 8, 'Complete independently audited deferred report required')
    reporting.population(cases)
    require(contexts == reporting.aggregate_contexts(cases), 'Seed-balanced metrics changed')
    for row in cases:
        for metric in ('physical_points', 'process_wall_seconds'):
            reporting.stats.finite(row[metric], metric)
    validate_curves(full, cases, early=False)
    validate_curves(early, cases, early=True)
    return summary, cases, full, early


def curve_lines(ax, rows, field, x):
    for arm in ARMS:
        selected = [r for r in rows if r['arm'] == arm]
        require(len(selected) == 1, 'Display arm missing or duplicated')
        values = selected[0][field]['quality_pct']
        require(len(values) == len(x), 'Curve axes do not match')
        y = [math.nan if v is None else reporting.stats.finite(v, 'curve quality') for v in values]
        ax.step(x, y, where='post', color=COLORS[arm], linestyle=STYLES[arm],
                linewidth=1.7, label=LABELS[arm])
    ax.set_ylim(0, 102)
    ax.grid(alpha=.18)


def draw_costs(cases, output, plt):
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.8))
    offsets = (-.27, -.09, .09, .27)
    for axis, metric, label in zip(axes, ('physical_points', 'process_wall_seconds'),
            ('Complete actual physical evaluations', 'Complete task process time (s)')):
        for index, arm in enumerate(ARMS):
            means = []
            for context in range(8):
                chosen = sorted((r for r in cases if r['context_id'] == context and r['arm'] == arm),
                                key=lambda r: r['seed'])
                values = [r[metric] for r in chosen]
                means.append(math.fsum(values) / len(values))
                if len(values) == 3:
                    axis.scatter([context + offsets[index] + v for v in (-.022, 0, .022)],
                        values, s=15, alpha=.65, color=COLORS[arm], zorder=2)
            axis.scatter([c + offsets[index] for c in range(8)], means, marker=('o', 's', '^', 'D')[index],
                s=44, facecolor='white', edgecolor=COLORS[arm], linewidth=1.5, label=LABELS[arm], zorder=3)
        axis.set_xticks(range(8), [f'c{c:02d}' for c in range(8)])
        axis.set_xlabel('All eight observed training contexts')
        axis.set_ylabel(label)
        axis.set_ylim(bottom=0)
        axis.grid(axis='y', alpha=.18)
    fig.suptitle('Deferred verification: complete costs, including the original Beam guard', y=.965, fontsize=13)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='upper center', bbox_to_anchor=(.5, .90), frameon=False, fontsize=9)
    fig.text(.5, .035, 'Hollow markers: within-context means. Small dots: all three fixed BC/PPO seeds.\n'
             'All arms use the same single-runtime shared evidence service; full task costs are retained.',
             ha='center', fontsize=8.5)
    fig.subplots_adjust(left=.075, right=.98, top=.75, bottom=.20, wspace=.25)
    plotting.export(fig, output, '01_all_context_complete_costs', plt)


def draw_means(full, early, output, plt):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for row_index, curves in enumerate((full, early)):
        title = 'Complete task: EARLY + original Beam GUARD' if row_index == 0 else 'EARLY only: no Beam floor yet'
        for column, (field, key, label) in enumerate((
            ('physical', 'physical_points', 'Actual physical evaluations'),
            ('worker_time', 'worker_elapsed_seconds', 'Measured checkpoint time from worker entry (s)'))):
            axis = axes[row_index, column]
            x = curves['axes'][key]
            curve_lines(axis, curves['group_curves'], field, x)
            axis.set_xlim(0, max(1, x[-1]))
            axis.set_title(title, fontsize=10)
            axis.set_xlabel(label)
            axis.set_ylabel('Mean certified quality (%)')
    fig.suptitle('Certified quality at measured resource boundaries', y=.975, fontsize=14)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='upper center', bbox_to_anchor=(.5, .935), frameon=False, fontsize=9)
    fig.text(.5, .028, 'Post-step curves; seeds are averaged within context, then all eight contexts equally.\n'
             'Missing certificates remain gaps; obtained certificates carry after stop. Worker checkpoints are not process exits.\n'
             'Only completed GUARD establishes the original Beam floor.', ha='center', fontsize=8.5)
    fig.subplots_adjust(left=.075, right=.98, top=.84, bottom=.16, hspace=.40, wspace=.22)
    plotting.export(fig, output, '02_complete_and_early_mean_quality', plt)


def draw_contexts(early, output, plt):
    fig, axes = plt.subplots(2, 4, figsize=(14, 8), sharex=True, sharey=True)
    for context, axis in enumerate(axes.flat):
        chosen = [r for r in early['context_curves'] if r['context_id'] == context]
        curve_lines(axis, chosen, 'physical', early['axes']['physical_points'])
        axis.set_title(f'c{context:02d}', fontsize=11)
        axis.set_xlim(0, 256)
        if context >= 4:
            axis.set_xlabel('EARLY actual evaluations')
        if context % 4 == 0:
            axis.set_ylabel('Certified quality (%)')
    fig.suptitle('EARLY certified quality in every observed context', y=.97, fontsize=14)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='upper center', bbox_to_anchor=(.5, .92), frameon=False, fontsize=9)
    fig.text(.5, .035, 'Identical 0–100% quality scale. All seeds retained in each context mean. No interpolation or outcome-dependent cropping.\n'
             'Overlapping curves are not moved; EARLY quality alone does not establish the Beam floor.', ha='center', fontsize=8.5)
    fig.subplots_adjust(left=.065, right=.985, top=.79, bottom=.16, hspace=.30, wspace=.15)
    plotting.export(fig, output, '03_all_context_early_quality', plt)


def plot(report, output):
    report, output = Path(report).resolve(), Path(output).resolve()
    require(output.is_relative_to(ROOT / 'reports') and not output.exists(), 'New project figure folder required')
    summary, cases, full, early = load_inputs(report)
    plt, versions = plotting.load_plotting()
    output.mkdir(parents=True)
    draw_costs(cases, output, plt)
    draw_means(full, early, output, plt)
    draw_contexts(early, output, plt)
    source = output / 'source'
    source.mkdir()
    files = (Path(__file__), Path(reporting.__file__), Path(reporting.stats.__file__),
             Path(reporting.shared.__file__), Path(plotting.__file__))
    for path in files:
        shutil.copyfile(path, source / path.name)
    save(output / 'provenance.json', {'report': str(report),
        'report_manifest_sha256': sha(report / 'completion_manifest.json'),
        'report_summary_sha256': sha(report / 'summary.json'),
        'source_bindings': summary['source_bindings'],
        'source_sha256': {p.name: sha(p) for p in files}, 'plotting_runtime': str(plotting.RUNTIME),
        'package_versions': versions, 'cells': 64, 'contexts': 8, 'figure_files': 6,
        'measurement_values_changed': False, 'new_physical_points': 0, 'new_training': 0,
        'visual_qa': 'Not performed by generation; inspect every PNG separately.'})
    (output / 'README.md').write_text(
        '# Deferred verification: complete-population figures\n\n'
        'Figure 1 retains every context and all three BC/PPO seeds; large hollow markers are context means. '
        'Costs include one shared physical runtime, EARLY and the complete original Beam GUARD. Process time '
        'runs from creation to actual exit, including model loading and recording.\n\n'
        'Figure 2 separates full-task and EARLY-only curves. Figure 3 retains each EARLY context on a common '
        '0–100% quality scale. Curves use post steps at measured physical/worker-clock boundaries, with no '
        'smoothing, interpolation, best-seed selection, or hidden failures. Missing certificates remain gaps '
        'and only existing certificates carry after stopping. Checkpoint time is not complete process time.\n\n'
        'Only a completed GUARD establishes the original Beam floor. The interleaved/deferred rules test '
        'the scheduling-order package; BC/PPO are fixed candidate-ranker transfers under the common deferred '
        'schedule, not newly trained three-operation policies. Compare PPO against the equally optimized '
        'deferred window and same-source BC; PPO versus interleaved combines schedule and ranking. These '
        'eight observed training contexts do not establish generalization. Source seals and plotting '
        'dependencies are recorded; generation alone does not certify visual QA.\n', encoding='utf-8')
    save(output / 'completion_manifest.json', {p.relative_to(output).as_posix(): sha(p)
        for p in sorted(output.rglob('*')) if p.is_file()})
    return {'status': 'COMPLETE_DEFERRED_FIGURES_PENDING_VISUAL_QA', 'files': 6, 'output': str(output)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(plot(args.report, args.output), ensure_ascii=False))
