"""Three scientific figure groups from a complete audited early-selector report.

Only standard plotting is used. This entry point imports no solver, policy, or
experiment runner and never changes a measured value or certifies an outcome.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.dont_write_bytecode = True
import plot_clpu_integrated_pilot as base

ARMS, LABELS, COLORS, STYLES = base.ARMS, base.LABELS, base.COLORS, base.STYLES
read, save, sha, require = base.read, base.save, base.sha, base.require
STATUS = 'AUDITED_DESCRIPTIVE_EARLY_INTERVENTION_PILOT'
SCHEMA = 'clpu-early-checkpoint-curves-v1'
INPUTS = ('summary.json', 'all_cells.json', 'curve_data.json', 'early_curve_data.json')


def validate_curves(curves, cases, *, early):
    """Validate display structure and population, not repeat scientific audit."""
    require(curves['schema'] == SCHEMA and curves['early_phase_only'] is early, 'Wrong curve schema or phase')
    cap = 256 if early else 512
    axes = curves['axes']
    require(axes['physical_points'] == list(range(cap + 1)), 'Wrong physical budget axis')
    time_axis = axes['worker_elapsed_seconds']
    require(isinstance(time_axis, list) and time_axis and time_axis == list(range(len(time_axis))),
            'Worker checkpoint time axis must start at zero without omitted bins')
    base.validate_population(curves['cells'])
    lookup = {r['cell']: r for r in cases}
    require({r['cell'] for r in curves['cells']} == set(lookup), 'Curve/cost task populations differ')
    for row in curves['cells']:
        source = lookup[row['cell']]
        require(all(row[k] == source[k] for k in ('context_id', 'scenario_id', 'event_id', 'arm', 'seed')),
                'Curve physical context or model changed')
        checkpoints = row['exact_checkpoints']
        require(checkpoints, 'No measured curve checkpoints')
        expected_phases = {'EARLY'} if early or row['arm'] == 'window_reference' else {'EARLY', 'GUARD'}
        require({p['phase'] for p in checkpoints} == expected_phases, 'EARLY and GUARD curves mixed or omitted')
        require(0 <= row['stop_physical_points'] <= (256 if row['arm'] == 'window_reference' else cap),
                'Curve exceeds declared budget')
        if early:
            require(all(p.get('guard_established') is False for p in checkpoints), 'EARLY curve claims Beam guarantee')
    contexts, groups = curves['context_curves'], curves['group_curves']
    require(len(contexts) == 32 and {(r['context_id'], r['arm']) for r in contexts}
            == {(c, a) for c in range(8) for a in ARMS}, 'Missing/repeated context curve')
    require(len(groups) == 4 and {r['arm'] for r in groups} == set(ARMS), 'Missing/repeated mean curve')
    for row in contexts:
        expected = [17, 41, 73] if row['arm'] in ('protected_bc', 'protected_ppo') else [0]
        require(row['seeds'] == expected, 'Context curve omitted or selected a model seed')
    for rows in (curves['cells'], contexts, groups):
        for row in rows:
            for name, axis in (('physical', axes['physical_points']), ('worker_time', time_axis)):
                values = row[name]['quality_pct']
                require(len(values) == len(axis), 'Curve axis/value lengths differ')
                base.values_with_gaps(values)
                marker = row[name].get('after_stop_carry', row[name].get('after_stop_carry_count'))
                require(isinstance(marker, list) and len(marker) == len(axis), 'Stop-carry evidence missing')


def load_inputs(report):
    sealed = base.verify_flat(report)
    require(set(INPUTS) <= set(sealed), 'Missing full or EARLY report inputs')
    summary, cases, full, early = (read(report / name) for name in INPUTS)
    require(summary['status'] == STATUS and summary['cells'] == summary['audited_cells'] == 64
            and summary['contexts'] == 8, 'Only the complete audited early pilot can be plotted')
    base.validate_population(cases)
    for row in cases:
        for key in ('physical_points', 'process_wall_seconds'):
            value = row[key]
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                    'Missing or invalid complete measured cost')
    validate_curves(full, cases, early=False)
    validate_curves(early, cases, early=True)
    return summary, cases, full, early


def curve_lines(ax, rows, axis, x):
    """Post steps preserve recorded resource bins; no linear interpolation."""
    for arm in ARMS:
        found = [row for row in rows if row['arm'] == arm]
        require(len(found) == 1, 'Missing or repeated display arm')
        values = found[0][axis]['quality_pct']
        require(len(values) == len(x), 'Display axis length changed')
        ax.step(x, base.values_with_gaps(values), where='post', label=LABELS[arm],
                color=COLORS[arm], linestyle=STYLES[arm], linewidth=1.8)
    ax.set_ylim(0, 102)
    ax.grid(alpha=.18)


def plot_costs(cases, output, plt):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))
    offsets = (-.24, -.08, .08, .24)
    for ax, metric, label in zip(axes, ('physical_points', 'process_wall_seconds'),
                               ('Complete actual physical evaluations', 'Complete parent process time (s)')):
        for index, arm in enumerate(ARMS):
            means = []
            for context in range(8):
                rows = base.select_cases(cases, context, arm)
                values = [row[metric] for row in rows]
                means.append(math.fsum(values) / len(values))
                if len(values) == 3:
                    ax.scatter([context + offsets[index] + jitter for jitter in (-.022, 0, .022)],
                        values, s=15, alpha=.6, color=COLORS[arm], zorder=2)
            ax.scatter([context + offsets[index] for context in range(8)], means,
                marker=('o', 's', '^', 'D')[index], s=45, facecolor='white', edgecolor=COLORS[arm],
                linewidth=1.5, label=LABELS[arm], zorder=3)
        ax.set_xticks(range(8), [base.short_context(base.select_cases(cases, c, ARMS[0])[0]) for c in range(8)],
                      rotation=40, ha='right')
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=.18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='upper center', bbox_to_anchor=(.5, 1.015), frameon=False)
    fig.suptitle('Early-selector pilot: complete cost in all eight training contexts', y=1.07)
    fig.text(.5, -.04, 'Hollow markers: within-context means. Small dots: all three fixed BC/PPO seeds.\n'
             'Guarded tasks pay both cold routes; direct window pays one route.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, .95))
    base.export(fig, output, '01_all_context_complete_costs', plt)


def plot_means(full, early, output, plt):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), sharey=True)
    for row_index, curves in enumerate((full, early)):
        scope = 'Complete task (EARLY + paid GUARD)' if row_index == 0 else 'EARLY route only (no Beam floor yet)'
        for column, (field, axis_key, label) in enumerate((
            ('physical', 'physical_points', 'Actual physical evaluations'),
            ('worker_time', 'worker_elapsed_seconds', 'Elapsed checkpoint time from worker entry (s)'))):
            ax = axes[row_index, column]
            x = curves['axes'][axis_key]
            curve_lines(ax, curves['group_curves'], field, x)
            ax.set_xlim(0, max(1, x[-1]))
            ax.set_xlabel(label)
            ax.set_ylabel('Mean certified quality (%)')
            ax.set_title(scope, fontsize=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='upper center', bbox_to_anchor=(.5, 1.015), frameon=False)
    fig.suptitle('Certified quality at measured resource boundaries', y=1.06)
    fig.text(.5, -.025, 'Post-step curves; eight equal context weights after seed averaging. Missing certificates remain gaps.\n'
             'Obtained certificates carry forward after stop. Worker checkpoints are not complete process exits.\n'
             'Only a completed GUARD establishes the final Beam floor.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .015, 1, .96))
    base.export(fig, output, '02_complete_and_early_mean_quality', plt)


def plot_contexts(cases, early, output, plt):
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 7), sharex=True, sharey=True)
    for context, ax in enumerate(axes.flat):
        rows = [r for r in early['context_curves'] if r['context_id'] == context]
        curve_lines(ax, rows, 'physical', early['axes']['physical_points'])
        title = base.short_context(base.select_cases(cases, context, ARMS[0])[0]).replace('\n', ' | ')
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, 256)
        if context >= 4:
            ax.set_xlabel('EARLY actual physical evaluations')
        if context % 4 == 0:
            ax.set_ylabel('Certified quality (%)')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='upper center', bbox_to_anchor=(.5, 1.025), frameon=False)
    fig.suptitle('EARLY route: all eight contexts, before the independent guard', y=1.07)
    fig.text(.5, -.025, 'All quality axes retain 0–100%. Post-step curves keep missing certificates and stop carry.\n'
             'BC/PPO are three-seed means. EARLY certificates do not establish a Beam guarantee.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .01, 1, .97))
    base.export(fig, output, '03_all_context_early_quality', plt)


CAPTION = '''# 前期策略介入小试：图表口径

图1保留全部8个已观察训练工况，展示完整任务真实物理点和父进程从启动到真实退出的时间。
保护任务计入EARLY和独立GUARD两条冷路径；直接窗口只有一条路径，预算分别为512和256。
大空心标记是工况内均值，小点完整保留BC/PPO的17、41、73三个固定种子。横向错开仅用于区分类别，数值未修改。

图2上排是完整任务曲线，下排是明确排除GUARD的EARLY曲线；左列为真实物理点，右列为worker入口后检查点时间。
各曲线先在工况内平均种子，再对8工况等权。采用post阶梯，不做插值或平滑。
认证前缺失保持空缺，停止后只保持已取得证书；原报告的stop-carry标记完整保留并绑定。
检查点时间不是求解瞬间，也不是完整进程退出时间；不能把EARLY时间称为完整保护方案速度。

图3保留全部8工况的EARLY质量曲线，统一显示0至100%范围，保留低质量及失败造成的证书缺失。
线条重叠表示记录相同，不移动曲线、不修改数据。完整种子曲线和精确边界见源报告。

所有路径共用相同等价缓存规则，但各冷路径缓存独立。只有完整GUARD完成才建立最终Beam保底；EARLY曲线本身没有此保证。
本轮是固定模型前16次扩展的介入对照，不是新训练、泛化检验或缓存消融。BC/PPO共享原始起点，但历史总训练数据与成本不同。
本绘图过程不新增物理求解或训练，不预填结果，不删除样本或填补未取得的证书。
'''


def plot(report, output):
    report, output = Path(report).resolve(), Path(output).resolve()
    require(report.name == 'derived_report' and report.parent.parent == ROOT / 'runs', 'Expected a project run report')
    require(output.is_relative_to(ROOT / 'reports') and output != ROOT / 'reports' and not output.exists(),
            'New reports subdirectory required')
    summary, cases, full, early = load_inputs(report)
    plt, versions = base.load_plotting()
    output.mkdir(parents=True)
    plot_costs(cases, output, plt)
    plot_means(full, early, output, plt)
    plot_contexts(cases, early, output, plt)
    (output / 'README.md').write_text(CAPTION, encoding='utf-8')
    source_dir = output / 'source'
    source_dir.mkdir()
    sources = (Path(__file__), Path(base.__file__), ROOT / 'tests/test_plot_clpu_early_pilot.py')
    for path in sources:
        shutil.copy2(path, source_dir / path.name)
    save(output / 'provenance.json', {'status': 'FIGURES_FROM_COMPLETE_AUDITED_EARLY_REPORT',
        'source_report': report.relative_to(ROOT).as_posix(), 'source_report_status': summary['status'],
        'source_completion_sha256': sha(report / 'completion_manifest.json'),
        'source_files': {name: sha(report / name) for name in INPUTS},
        'plotting_sources': {p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
        'plotting_runtime': str(base.RUNTIME.resolve()), 'python_executable': sys.executable,
        'plotting_dependencies': versions, 'cases': 64, 'contexts': 8, 'figure_files': 6,
        'quality_display': 'post steps', 'full_physical_axis': [0, 512], 'early_physical_axis': [0, 256],
        'new_physical_points': 0, 'new_training': 0, 'measurement_values_changed': False,
        'missing_certificates_imputed': False, 'stop_carry_preserved': True})
    save(output / 'completion_manifest.json', {p.relative_to(output).as_posix(): sha(p)
        for p in sorted(output.rglob('*')) if p.is_file()})
    return {'status': 'COMPLETE_AUDITED_EARLY_REPORT_FIGURES', 'output': str(output), 'files': 6}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(plot(args.report, args.output), ensure_ascii=False))
