"""Export standard scientific figures from the complete audited pilot report.

Uses the project's Python with a separate, recorded plotting-only package
directory. No solver, policy, or experiment runner is imported or executed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'artifacts/clpu_integrated_plot_runtime_20261030'
ARMS = ('window_reference', 'protected_rule', 'protected_bc', 'protected_ppo')
LABELS = dict(zip(ARMS, ('Direct window', 'Guarded rule', 'Guarded BC', 'Guarded PPO')))
COLORS = dict(zip(ARMS, ('#555B66', '#0072B2', '#D68910', '#9B4F96')))
STYLES = dict(zip(ARMS, ('-', '--', '-.', ':')))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def verify_flat(directory):
    directory = Path(directory).resolve()
    files = read(directory / 'completion_manifest.json')
    require(bool(files), 'Empty report seal')
    for name, digest in files.items():
        path = (directory / name).resolve()
        require(path != directory and path.is_relative_to(directory), 'Unsafe sealed path')
        require(path.is_file() and sha(path) == digest, 'Changed sealed figure source: ' + name)
    return files


def select_cases(cases, context, arm):
    rows = [r for r in cases if r['context_id'] == context and r['arm'] == arm]
    expected = {17, 41, 73} if arm in ('protected_bc', 'protected_ppo') else {0}
    require(len(rows) == len(expected) and {r['seed'] for r in rows} == expected,
            'Missing, repeated, or selected seed')
    return sorted(rows, key=lambda r: r['seed'])


def validate_population(cases):
    require(len(cases) == 64 and {r['context_id'] for r in cases} == set(range(8)),
            'All 64 cells and eight contexts are required')
    require(len({r['cell'] for r in cases}) == 64, 'Repeated cell')
    for context in range(8):
        groups = [select_cases(cases, context, arm) for arm in ARMS]
        require(len({(r['scenario_id'], r['event_id']) for group in groups for r in group}) == 1,
                'Context has mismatched physical inputs')


def values_with_gaps(values):
    require(all(v is None or (type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 100+1e-9)
                for v in values), 'Invalid certified quality')
    return [math.nan if v is None else v for v in values]


def short_context(row):
    fault = row['scenario_id'].replace('N2__line.', '').replace('__line.', '/').upper()
    return f"{fault}\nEvent {row['event_id']}"


def load_plotting():
    require(RUNTIME.is_dir(), 'Separate plotting runtime is unavailable')
    os.environ['MPLCONFIGDIR'] = str(ROOT / 'artifacts/clpu_integrated_matplotlib_cache')
    sys.path.insert(0, str(RUNTIME))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    require(Path(matplotlib.__file__).resolve().is_relative_to(RUNTIME), 'Unrecorded plotting runtime')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.labelsize': 11, 'axes.titlesize': 12, 'legend.fontsize': 10,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white'})
    versions = {name: importlib.metadata.version(name) for name in
                ('matplotlib', 'numpy', 'contourpy', 'cycler', 'fonttools', 'kiwisolver', 'pillow', 'pyparsing')}
    return plt, versions


def export(fig, output, name, plt):
    fig.savefig(output / (name + '.png'), dpi=240, bbox_inches='tight')
    fig.savefig(output / (name + '.pdf'), bbox_inches='tight')
    plt.close(fig)


def plot(report, output):
    report, output = Path(report).resolve(), Path(output).resolve()
    require(report.name == 'derived_report' and report.parent.parent == ROOT / 'runs',
            'Expected a complete project run report')
    require(output.is_relative_to(ROOT / 'reports') and not output.exists(), 'New reports output directory required')
    sealed = verify_flat(report)
    require({'summary.json', 'all_cells.json', 'curve_data.json'}.issubset(sealed), 'Incomplete report inputs')
    summary, cases, curves = (read(report / name) for name in ('summary.json', 'all_cells.json', 'curve_data.json'))
    require(summary['status'] == 'AUDITED_DESCRIPTIVE_INTEGRATED_PILOT', 'Report is not independently audited')
    validate_population(cases)
    require(curves['schema'] == 'clpu-integrated-curves-v1', 'Unexpected curves schema')
    plt, versions = load_plotting()
    output.mkdir(parents=True)
    axes_data = curves['axes']

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
    offsets = (-.24, -.08, .08, .24)
    for panel, metric, label in ((0, 'physical_points', 'Actual physical evaluations'),
                                 (1, 'process_wall_seconds', 'Complete process time (s)')):
        ax = axs[panel]
        for ai, arm in enumerate(ARMS):
            means = []
            for context in range(8):
                rows = select_cases(cases, context, arm)
                vals = [r[metric] for r in rows]
                require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in vals),
                        'Missing or invalid actual cost')
                means.append(sum(vals)/len(vals))
                if len(vals) == 3:
                    ax.scatter([context+offsets[ai]+j for j in (-.022, 0, .022)], vals,
                               s=14, alpha=.55, color=COLORS[arm], zorder=2)
            ax.scatter([i+offsets[ai] for i in range(8)], means, marker=('o','s','^','D')[ai],
                       s=43, facecolor='white', edgecolor=COLORS[arm], linewidth=1.5,
                       label=LABELS[arm], zorder=3)
        ax.set_xticks(range(8), [short_context(select_cases(cases, i, ARMS[0])[0]) for i in range(8)],
                      rotation=45, ha='right')
        ax.set_ylabel(label); ax.set_ylim(bottom=0); ax.grid(axis='y', alpha=.18)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='upper center', bbox_to_anchor=(.5, 1.01), frameon=False)
    fig.suptitle('Integrated pilot: all eight observed training contexts', y=1.065)
    fig.text(.5, -.04, 'Large markers: within-context seed means. Small dots: all three fixed BC/PPO seeds.', ha='center')
    fig.tight_layout(rect=(0, 0, 1, .95))
    export(fig, output, '01_all_context_costs', plt)

    def curve_lines(ax, rows, axis, x, group=False):
        for arm in ARMS:
            found = [r for r in rows if r['arm'] == arm]
            require(len(found) == 1, 'Missing/repeated curve arm')
            y = found[0][axis]['quality_pct']
            require(len(y) == len(x), 'Curve axis length mismatch')
            ax.plot(x, values_with_gaps(y), label=LABELS[arm], color=COLORS[arm],
                    linestyle=STYLES[arm], linewidth=2.0)
        ax.set_ylim(0, 102); ax.grid(alpha=.18)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.7))
    curve_lines(axs[0], curves['group_curves'], 'physical', axes_data['physical_points'], True)
    curve_lines(axs[1], curves['group_curves'], 'worker_time', axes_data['worker_elapsed_seconds'], True)
    axs[0].set_xlabel('Actual physical evaluations'); axs[1].set_xlabel('Elapsed time from worker entry (s)')
    axs[0].set_ylabel('Mean certified quality (%)'); axs[1].set_ylabel('Mean certified quality (%)')
    axs[0].set_xlim(0, 256); axs[1].set_xlim(0, max(axes_data['worker_elapsed_seconds']))
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='upper center', bbox_to_anchor=(.5, 1.02), frameon=False)
    fig.suptitle('Certified incumbent quality at measured resource boundaries', y=1.08)
    fig.text(.5, -.035, 'Equal context weights; BC/PPO averaged over three seeds. Missing certificates remain gaps. '
             'Finished results are held forward.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, .95))
    export(fig, output, '02_mean_quality_resources', plt)

    fig, axs = plt.subplots(2, 4, figsize=(13, 6.7), sharex=True, sharey=True)
    for context, ax in enumerate(axs.flat):
        rows = [r for r in curves['context_curves'] if r['context_id'] == context]
        curve_lines(ax, rows, 'physical', axes_data['physical_points'])
        ax.set_title(short_context(select_cases(cases, context, ARMS[0])[0]).replace('\n', ' | '), fontsize=10)
        ax.set_xlim(0, 256)
        if context >= 4: ax.set_xlabel('Actual physical evaluations')
        if context % 4 == 0: ax.set_ylabel('Certified quality (%)')
    handles, labels = axs.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='upper center', bbox_to_anchor=(.5, 1.025), frameon=False)
    fig.suptitle('All contexts retained: quality versus actual physical evaluations', y=1.07)
    fig.text(.5, -.025, 'Each panel keeps the same 0–100% quality scale. BC/PPO curves are three-seed means; no failed case is removed.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, .97))
    export(fig, output, '03_all_context_quality', plt)

    caption = '''# 集成小试图表说明

图1展示全部8个已观察训练工况的实际求解和完整进程成本。大标记为工况内种子均值，小点保留BC/PPO全部三个固定种子。

图2展示完整认证质量随真实求解和worker入口后时间的变化，先按工况内种子平均，再给8工况等权。首次取得完整证书前为空缺；完成后的曲线只保持已取得结果。worker时间包括预检和模型加载，但不等于控制器创建进程至真实退出的完整进程时间，后者在图1另报。

图3保留全部工况，纵轴统一0至100%，不裁去低质量工况。种子明细和原始精确认证边界见源报告。线条重叠表示记录曲线相同，不人为错开或修改数值。

所有方法使用相同精确电气输入缓存。Direct window从初态执行256扩展；Guarded方法先付完整原Beam，再最多256次额外续搜。BC是对应PPO的共同BC起点，不代表两者历史总训练数据/成本相等。图表仅为固定训练工况集成验证，不是独立泛化或缓存消融结论。
'''
    (output / 'README.md').write_text(caption, encoding='utf-8')
    save(output / 'provenance.json', {'source_report': report.relative_to(ROOT).as_posix(),
        'source_completion_sha256': sha(report / 'completion_manifest.json'),
        'source_files': {n: sha(report / n) for n in ('summary.json','all_cells.json','curve_data.json')},
        'plotting_script_sha256': sha(Path(__file__)), 'python_executable': sys.executable,
        'plotting_runtime': str(RUNTIME.resolve()), 'plotting_dependencies': versions,
        'new_physical_points': 0, 'new_training': 0, 'cases': 64, 'contexts': 8,
        'measurement_values_changed': False, 'missing_certificates_imputed': False,
        'figure_files': 6})
    save(output / 'completion_manifest.json', {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    return {'status':'COMPLETE_AUDITED_REPORT_FIGURES', 'output':str(output), 'files':6}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(plot(args.report, args.output), ensure_ascii=False))
