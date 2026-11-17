"""Two-panel, all-context cost figure from the sealed shared-rule report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import plot_clpu_integrated_pilot as plotting
import report_clpu_shared_pilot as reporting

read, save, sha, require = reporting.read, reporting.save, reporting.sha, reporting.require
MODES = reporting.MODES
COLORS = {'private': '#555B66', 'shared': '#0072B2'}


def load_inputs(report):
    reporting.manifest(report, required=('summary.json', 'all_cells.json', 'all_pairs.json'), all_files=True)
    summary, cases = read(report / 'summary.json'), read(report / 'all_cells.json')
    require(summary['status'] == reporting.STATUS and summary['cells'] == summary['audited_cells'] == 16
            and summary['contexts'] == 8, 'Complete audited shared report required')
    reporting.population(cases)
    modes, pairs, comparison = reporting.summarize(cases)
    require(summary['modes'] == modes and summary['comparison'] == comparison and read(report / 'all_pairs.json') == pairs,
            'Report means or paired values differ from retained cells')
    return summary, cases


def draw_costs(cases, output, plt):
    reporting.population(cases)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
    for axis, metric, label in zip(axes, ('physical_points', 'process_wall_seconds'),
            ('Complete actual physical evaluations', 'Complete task process time (s)')):
        for index, mode in enumerate(MODES):
            selected = [next(c for c in cases if c['mode'] == mode and c['context_id'] == context) for context in range(8)]
            values = [reporting.base.finite(c[metric], metric) for c in selected]
            axis.scatter([c + (-.10 if index == 0 else .10) for c in range(8)], values,
                marker=('o', 's')[index], s=42, color=COLORS[mode], label=mode.capitalize(), zorder=3)
        labels = [plotting.short_context(next(c for c in cases if c['context_id'] == context)) for context in range(8)]
        axis.set_xticks(range(8), labels, rotation=35, ha='right')
        axis.set_ylabel(label)
        axis.set_ylim(bottom=0)
        axis.grid(axis='y', alpha=.18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc='upper center', frameon=False)
    fig.suptitle('Shared physical evidence: all eight observed training contexts', y=1.04)
    fig.text(.5, -.04, 'Both modes: one runtime, complete EARLY window + independent logical Beam guard.\n'
             'Private: separate evidence pools. Shared: causal exact-input reuse. Rule only; no PPO intervention.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, .94))
    plotting.export(fig, output, '01_all_context_shared_costs', plt)


def plot(report, output):
    report, output = Path(report).resolve(), Path(output).resolve()
    require(output.is_relative_to(ROOT / 'reports') and not output.exists(), 'New project reports figure directory required')
    summary, cases = load_inputs(report)
    plt, versions = plotting.load_plotting()
    output.mkdir(parents=True)
    draw_costs(cases, output, plt)
    source = output / 'source'
    source.mkdir()
    files = (Path(__file__), Path(reporting.__file__), Path(reporting.base.__file__), Path(plotting.__file__))
    for item in files:
        shutil.copyfile(item, source / item.name)
    save(output / 'provenance.json', {'report': str(report), 'report_manifest_sha256': sha(report / 'completion_manifest.json'),
        'report_summary_sha256': sha(report / 'summary.json'), 'source_bindings': summary['source_bindings'],
        'source_sha256': {p.name: sha(p) for p in files}, 'plotting_runtime': str(plotting.RUNTIME),
        'package_versions': versions, 'cells': 16, 'contexts': 8, 'figure_files': 2,
        'measurement_values_changed': False, 'new_physical_points': 0, 'new_training': 0,
        'quality_or_speed_gate_added': False, 'visual_qa': 'Not performed by plot generation; inspect PNG separately.'})
    (output / 'README.md').write_text('# 共享证据规则对照：完整成本图\n\n'
        '保留全部8工况及private/shared两模式。左图为真实物理验证点，右图为任务进程创建至真实退出的完整时间；'
        '包括一次初始化、EARLY窗口与独立逻辑GUARD的全部费用。两模式使用同一单运行时实现，仅证据池共享方式不同。\n\n'
        '数据未经平滑、筛选或修改；每个标记对应一个真实任务。该公共验证层对照不包含PPO，不与旧双进程时间计算加速比。'
        'PNG和PDF来自同一图；生成成功本身不代表已做视觉核对。可复现源码快照、来源SHA和包版本随图封存。\n', encoding='utf-8')
    save(output / 'completion_manifest.json', {p.relative_to(output).as_posix(): sha(p)
        for p in sorted(output.rglob('*')) if p.is_file()})
    return {'status': 'COMPLETE_SHARED_RULE_FIGURES_PENDING_VISUAL_QA', 'files': 2, 'output': str(output)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(plot(args.report, args.output), ensure_ascii=False))
