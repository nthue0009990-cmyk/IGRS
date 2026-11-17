"""Audit-bound descriptive results for the early-16, independent-guard pilot.

No solver, actor or training import. Old descriptive arithmetic is reused;
population, phase accounting and report bindings are explicit for this study.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import report_clpu_integrated_pilot as stats

STUDY = 'CLPU_EARLY_PROTECTED_PILOT_V1'
STATUS = 'AUDITED_DESCRIPTIVE_EARLY_INTERVENTION_PILOT'
ARMS, SEEDS, ARM_SEEDS = stats.ARMS, stats.SEEDS, stats.ARM_SEEDS
require, read, save, sha = stats.require, stats.read, stats.save, stats.sha
METRICS = stats.METRICS + ('early_quality_pct', 'baseline_quality_pct',
    'early_physical_points', 'guard_physical_points', 'early_block_quality_pct',
    'early_block_physical_points')


def validate_population(schedule):
    require(isinstance(schedule, list) and len(schedule) == 64, 'All 64 tasks required')
    require(len({r['cell'] for r in schedule}) == 64, 'Duplicate task')
    require({r['context_id'] for r in schedule} == set(range(8)), 'Eight contexts required')
    for context in range(8):
        rows = [r for r in schedule if r['context_id'] == context]
        require(len({(r['scenario_id'], r['event_id']) for r in rows}) == 1, 'Context changed')
        require(Counter((r['arm'], r['seed']) for r in rows) ==
            Counter((a, s) for a in ARMS for s in ARM_SEEDS[a]), 'Arm or seed omitted')
    for row in schedule:
        require(row['protected'] is (row['arm'] != 'window_reference')
            and row['early_policy_expansions'] == 16
            and row['cold_searches'] == (2 if row['protected'] else 1), 'Intervention design changed')


def case_row(spec, result, audited):
    require(result['spec'] == spec and audited['status'] == 'PASS'
        and audited['quality_checkpoints_verified'] is True, 'Unbound or unaudited task')
    for key in ('cell', 'context_id', 'scenario_id', 'event_id', 'arm', 'seed', 'method'):
        require(audited[key] == spec[key], 'Wrong audit task: ' + key)
    require(all(result[k] == 0 for k in ('new_training', 'optimizer_updates', 'training_interactions')),
        'Frozen actor was trained')
    require(all(result[k] is False for k in ('formal_holdout_numerical_access',
        'tune_numerical_access', 'clpu_holdout_access')), 'Population boundary crossed')
    phases = result['phases']
    require(set(phases) == ({'EARLY', 'GUARD'} if spec['protected'] else {'EARLY'}), 'Cold phase omitted')
    oracle = result['oracle']
    costs = {key: stats.integer(oracle[key], key) for key in
        ('physical_points', 'logical_points', 'requests', 'semantic_cache_hits', 'electrical_reuses')}
    costs['numerical_entries'] = stats.integer(oracle['actual_numerical_entries'], 'numerical entries')
    for key in ('physical_points', 'logical_points', 'requests', 'numerical_entries'):
        require(costs[key] == sum(p[key] for p in phases.values()) == audited[key], 'Phase cost omitted: ' + key)
    require(costs['physical_points'] <= costs['logical_points'] <= 256 * len(phases)
        and oracle['initialization_numerical_entries'] == 3 * len(phases)
        and costs['numerical_entries'] == 3 * len(phases) + 4 * costs['physical_points'], 'Cold accounting differs')
    timing = {target: stats.finite(result[source], target) for target, source in (
        ('worker_elapsed_seconds', 'elapsed_seconds'), ('computation_elapsed_seconds', 'computation_elapsed_seconds'),
        ('preflight_elapsed_seconds', 'preflight_elapsed_seconds'), ('model_load_seconds', 'model_load_seconds'),
        ('runtime_initialization_seconds', 'runtime_initialization_seconds'))}
    timing['process_wall_seconds'] = stats.finite(audited['process_wall_seconds'], 'process wall')
    require(timing['process_wall_seconds'] >= timing['worker_elapsed_seconds']
        >= timing['preflight_elapsed_seconds'] + timing['computation_elapsed_seconds']
        and timing['preflight_elapsed_seconds'] >= timing['model_load_seconds'], 'Complete timing omitted')
    require(audited['worker_elapsed_seconds'] == timing['worker_elapsed_seconds'], 'Audited worker time changed')
    quality = result['quality_pct']
    require(quality == audited['quality_pct'], 'Audited output quality changed')
    if quality is not None:
        require(result['fully_certified'] is True and 0 <= stats.finite(quality, 'quality') <= 100 + stats.TIE_PP,
            'Quality lacks complete certificate')
    else:
        require(result['fully_certified'] is False, 'Missing output marked complete')
    guard = result['guard_audit']
    established = False if not spec['protected'] else guard['guard_established']
    require(established == audited['guard_established'], 'Audited guard flag changed')
    if established:
        require(guard['final_energy_kwh'] >= guard['baseline_energy_kwh'], 'Strict Beam floor violated')
    points = result['checkpoints']
    require(points and points == audited['checkpoints'], 'Audited quality/cost boundaries changed')
    for index, point in enumerate(points):
        require(point['checkpoint_index'] == index, 'Missing chronological checkpoint')
        if index:
            require(all(point[k] >= points[index - 1][k] for k in ('physical_points', 'logical_points',
                'numerical_entries', 'requests', 'worker_elapsed_seconds')), 'Aggregate cost reset')
        require(point['guard_established'] is False or (point['phase'] == 'GUARD' and point['kind'] == 'CLOSE'),
            'Guard guaranteed before completion')
    require(all(points[-1][k] == costs[k] for k in ('physical_points', 'logical_points', 'numerical_entries', 'requests')),
        'Final checkpoint omits a phase')
    require(points[-1]['worker_elapsed_seconds'] <= timing['worker_elapsed_seconds'], 'Clock exceeds worker')
    count = sum(p['policy_decisions'] for p in phases.values())
    require(count == audited['policy_decisions'] and 0 <= count <= 16
        and phases.get('GUARD', {}).get('policy_decisions', 0) == 0, 'Policy outside first sixteen')
    block = result['early_block_checkpoint']
    require(block['phase'] == 'EARLY' and block['phase_choices'] == min(16, phases['EARLY']['search_decisions']),
        'Early decision block boundary changed')
    return {**{k: spec[k] for k in ('cell', 'context_id', 'scenario_id', 'event_id', 'arm', 'seed', 'method')},
        **costs, **timing, 'quality_pct': quality, 'fully_certified': result['fully_certified'],
        'guard_established': established, 'policy_decisions': count, 'selected_phase': result['selected_phase'],
        'early_quality_pct': result['early_quality_pct'], 'baseline_quality_pct': result['baseline_quality_pct'],
        'early_physical_points': phases['EARLY']['physical_points'],
        'guard_physical_points': phases.get('GUARD', {}).get('physical_points', 0),
        'early_block_quality_pct': block['quality_pct'], 'early_block_physical_points': block['physical_points'],
        'gain_over_paid_baseline_kwh': result['gain_over_paid_baseline_kwh'],
        'phase_summary': deepcopy(phases), 'early_block_checkpoint': deepcopy(block),
        'checkpoints': deepcopy(points), 'physical_statuses': deepcopy(audited['physical_statuses'])}


def aggregate_contexts(rows):
    output = []
    for context in range(8):
        for arm in ARMS:
            chosen = sorted((r for r in rows if r['context_id'] == context and r['arm'] == arm), key=lambda r: r['seed'])
            require([r['seed'] for r in chosen] == list(ARM_SEEDS[arm]), 'Incomplete context/seed population')
            metrics, coverage = {}, {}
            for key in METRICS:
                metrics[key], coverage[key] = stats.strict_mean([r[key] for r in chosen], len(chosen))
            output.append({'context_id': context, 'scenario_id': chosen[0]['scenario_id'], 'event_id': chosen[0]['event_id'],
                'arm': arm, 'seeds': list(ARM_SEEDS[arm]), 'cells': [r['cell'] for r in chosen],
                'metrics': metrics, 'metric_coverage': coverage,
                'fully_certified_cells': sum(r['fully_certified'] for r in chosen), 'expected_seed_count': len(chosen)})
    return output


def all_pairs(rows, include_guard=True):
    lookup = {(r['context_id'], r['arm'], r['seed']): r for r in rows}
    output = []
    for context in range(8):
        for seed in SEEDS:
            ppo, bc = (lookup[context, arm, seed] for arm in ('protected_ppo', 'protected_bc'))
            output.extend((stats.pair(ppo, bc, 'ppo_minus_bc'),
                stats.pair(ppo, lookup[context, 'protected_rule', 0], 'ppo_minus_rule'),
                stats.pair(ppo, lookup[context, 'window_reference', 0], 'ppo_minus_window'),
                stats.pair(bc, lookup[context, 'protected_rule', 0], 'bc_minus_rule')))
    if include_guard:
        for row in rows:
            guard = row['phase_summary'].get('GUARD')
            if guard is None:
                continue
            right = {**row, **{k: guard[k] for k in ('physical_points', 'logical_points', 'numerical_entries', 'requests')},
                'quality_pct': guard['quality_pct'], 'arm': 'own_paid_beam', 'process_wall_seconds': None,
                'worker_elapsed_seconds': None, 'computation_elapsed_seconds': None}
            output.append(stats.pair(row, right, row['arm'] + '_minus_own_paid_beam', own_baseline=True))
    return output


def early_only_rows(rows):
    """Explicit EARLY-only mechanism accounting; no fictitious process exit."""
    output = []
    for row in rows:
        phase = row['phase_summary']['EARLY']
        end = [p for p in row['checkpoints'] if p['phase'] == 'EARLY'][-1]
        keep = ('cell', 'context_id', 'scenario_id', 'event_id', 'arm', 'seed', 'method',
            'preflight_elapsed_seconds', 'model_load_seconds', 'policy_decisions',
            'early_quality_pct', 'early_physical_points', 'early_block_quality_pct',
            'early_block_physical_points', 'early_block_checkpoint')
        output.append({**{k: row[k] for k in keep},
            **{k: phase[k] for k in ('physical_points', 'logical_points', 'numerical_entries', 'requests')},
            'quality_pct': phase['quality_pct'], 'fully_certified': phase['fully_certified'],
            'worker_elapsed_seconds': end['worker_elapsed_seconds'],
            'computation_elapsed_seconds': end['elapsed_seconds'], 'process_wall_seconds': None,
            'runtime_initialization_seconds': phase['runtime_initialization_seconds'],
            'selected_phase': 'EARLY', 'gain_over_paid_baseline_kwh': None,
            'guard_established': None, 'guard_physical_points': 0, 'baseline_quality_pct': None,
            'phase_summary': {'EARLY': deepcopy(phase)}, 'early_phase_only': True,
            'checkpoints': deepcopy([p for p in row['checkpoints'] if p['phase'] == 'EARLY']),
            'timing_scope': 'Worker-entry to EARLY close, including preflight and model load. No independent process ended here.'})
    return output


def make_curve_data(rows, *, phase_only=False):
    physical_axis = list(range(257 if phase_only else 513))
    timeline = []
    for row in rows:
        points = [p for p in row['checkpoints'] if p['phase'] == 'EARLY'] if phase_only else row['checkpoints']
        timeline.append((row, points))
    time_axis = list(range(math.ceil(max(p[-1]['worker_elapsed_seconds'] for _, p in timeline)) + 1))
    cells = []
    for row, points in timeline:
        end = points[-1]
        cells.append({**{k: row[k] for k in ('cell', 'context_id', 'scenario_id', 'event_id', 'arm', 'seed')},
            'stop_physical_points': end['physical_points'], 'stop_worker_elapsed_seconds': end['worker_elapsed_seconds'],
            'physical': stats.step_curve(points, physical_axis, 'physical_points', end['physical_points']),
            'worker_time': stats.step_curve(points, time_axis, 'worker_elapsed_seconds', end['worker_elapsed_seconds']),
            'exact_checkpoints': deepcopy(points)})
    contexts = []
    for context in range(8):
        for arm in ARMS:
            chosen = [r for r in cells if r['context_id'] == context and r['arm'] == arm]
            require(sorted(r['seed'] for r in chosen) == list(ARM_SEEDS[arm]), 'Missing curve seed')
            contexts.append({'context_id': context, 'arm': arm, 'scenario_id': chosen[0]['scenario_id'],
                'event_id': chosen[0]['event_id'], 'seeds': list(ARM_SEEDS[arm]),
                **{axis: stats.average_curves(chosen, axis) for axis in ('physical', 'worker_time')}})
    groups = [{'arm': arm, **{axis: stats.average_curves([r for r in contexts if r['arm'] == arm], axis, contexts=True)
        for axis in ('physical', 'worker_time')}} for arm in ARMS]
    return {'schema': 'clpu-early-checkpoint-curves-v1', 'early_phase_only': phase_only,
        'axes': {'physical_points': physical_axis, 'worker_elapsed_seconds': time_axis},
        'cells': cells, 'context_curves': contexts, 'group_curves': groups,
        'semantics': {'quality': 'Latest fully certified plan available at the completed checkpoint; never a graph prediction.',
            'guard': 'Beam floor exists only after complete GUARD close, marked in exact_checkpoints. EARLY curves do not imply it.',
            'budget': 'Real paid solves including both independent cold phases in complete curves; EARLY-only excludes GUARD explicitly.',
            'time': 'Worker-entry clock with loading and phase initialization, not exact solver return or process-start certificate time.',
            'same_cost': 'All completed checkpoints at a physical count are included; these free actions still require measured time.',
            'carry': 'Only an already obtained certificate is carried after phase/task stop, explicitly flagged.',
            'mean': 'Seeds within context, then all eight contexts; any missing certificate keeps complete-population mean null.',
            'smoothing': 'None; use post-step display with no interpolation or retrospective early stopping.'}}


def make_summary(rows, contexts, pairs, binding, controller, run_name):
    arms = {}
    for arm in ARMS:
        chosen = [r for r in rows if r['arm'] == arm]
        grouped = [r for r in contexts if r['arm'] == arm]
        arms[arm] = {'cells': len(chosen), 'contexts': 8, 'seeds': list(ARM_SEEDS[arm]),
            'context_balanced_metrics': {k: stats.distribution([r['metrics'][k] for r in grouped], 8) for k in METRICS},
            'fully_certified_cells': sum(r['fully_certified'] for r in chosen),
            'guard_established_cells': sum(r['guard_established'] is True for r in chosen),
            'zero_policy_decision_cells': sum(r['policy_decisions'] == 0 for r in chosen),
            'all_cell_timing': {k: stats.distribution([r[k] for r in chosen]) for k in
                ('process_wall_seconds', 'worker_elapsed_seconds', 'computation_elapsed_seconds', 'model_load_seconds')}}
    return {'status': STATUS, 'study': STUDY, 'run': run_name, 'utc': datetime.now(timezone.utc).isoformat(),
        'contexts': 8, 'cells': 64, 'audited_cells': 64, 'arms': arms, 'comparisons': stats.comparison_summary(pairs),
        'total_cost': {k: sum(r[k] for r in rows) for k in ('physical_points', 'logical_points', 'numerical_entries')},
        'fully_certified_cells': sum(r['fully_certified'] for r in rows),
        'uncertified_cells_retained': [r['cell'] for r in rows if not r['fully_certified']],
        'evaluation_controller_wall_seconds': stats.finite(controller['wall_seconds'], 'controller wall'),
        'new_training': 0, 'optimizer_updates': 0, 'new_report_physical_points': 0,
        'source_bindings': binding, 'interpretation': [
            'Fixed early-16 closed-loop intervention on eight observed training contexts, not new training or generalization evidence.',
            'EARLY and original Beam GUARD are separate cold searches with isolated caches. Protected total cap is 512, reference cap 256.',
            'Primary selector comparison is among protected window/BC/PPO. Direct window is an explicitly different deployment reference.',
            'Only final completed GUARD establishes the Beam floor; early certified proposals alone have no Beam guarantee.',
            'BC shares teacher and starting model with PPO, but PPO used additional historical interactions and updates.',
            'Mean first averages seeds within each context, then eight contexts; all failures and zero-participation tasks remain.',
            'Complete process times use this common compact recording implementation. Do not splice earlier run timings or deduct logging.',
            'Full curves carry only past certificates, show phase boundaries, and do not implement retrospective early stopping.',
            'No K sweep, automatic training, sample selection, physical relaxation or attribution of shared caching to PPO.']}


def markdown(summary, contexts):
    fmt = stats.fmt
    out = ['# 前16扩展策略介入与独立Beam保护：完整小试结果', '',
        '全部64任务及独立证据审计通过；以下为固定8个已观察训练工况的描述性结果，保留全部种子及失败。', '',
        '| 方法 | 任务数 | 最终质量/% | EARLY质量/% | 真实点（全流程） | 完整进程/秒 |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        group = summary['arms'][arm]
        m = group['context_balanced_metrics']
        out.append(f"| {arm} | {group['cells']} | {fmt(m['quality_pct']['mean'])} | {fmt(m['early_quality_pct']['mean'])} | {fmt(m['physical_points']['mean'])} | {fmt(m['process_wall_seconds']['mean'])} |")
    out += ['', '保护方法先完成EARLY，再以独立空缓存完整执行原Beam；费用均已相加。每保护任务上限512真实点，直接窗口上限256。EARLY阶段不提前声明Beam下界；最终在完整认证结果之间严格择大，精确能量并列选GUARD。', '',
        '| 比较（左减右） | 平均质量差/pp | 最差工况均值差/pp | 真实点差 | 完整进程差/秒 |',
        '|---|---:|---:|---:|---:|---:|']
    for name, group in summary['comparisons'].items():
        m = group['context_balanced_delta']
        out.append(f"| {name} | {fmt(m['quality_pp']['mean'],signed=True)} | {fmt(m['quality_pp']['minimum'],signed=True)} | {fmt(m['physical_points']['mean'],signed=True)} | {fmt(m['process_wall_seconds']['mean'],signed=True)} |")
    out += ['', '质量差正值更好，费用差负值更省。自身GUARD不是独立进程，所以其完整进程/worker差记未定义，不能把阶段时长冒作冷进程时间。全部152配对及每种子、最差情况在机器可读文件中保存。', '',
        '| EARLY路比较（不含GUARD） | 平均质量差/pp | 真实点差 | 至EARLY完成的worker时间差/秒 |',
        '|---|---:|---:|---:|']
    for name, group in summary.get('early_only_comparisons', {}).items():
        m = group['context_balanced_delta']
        out.append(f"| {name} | {fmt(m['quality_pp']['mean'],signed=True)} | {fmt(m['physical_points']['mean'],signed=True)} | {fmt(m['worker_elapsed_seconds']['mean'],signed=True)} |")
    out += ['', 'EARLY路比较隔离前16连续决策块及其共同后续搜索；worker时间从任务入口算起，含模型加载。它不包含GUARD，不能代替保护方案完整费用或称为完整进程速度。', '',
        '| 工况 | 事件 | 方法 | 最终质量/% | EARLY质量/% | 前16块质量/% | 真实点 | 完整进程/秒 |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for row in contexts:
        m = row['metrics']
        out.append(f"| {row['scenario_id']} | {row['event_id']} | {row['arm']} | {fmt(m['quality_pct'])} | {fmt(m['early_quality_pct'])} | {fmt(m['early_block_quality_pct'])} | {fmt(m['physical_points'])} | {fmt(m['process_wall_seconds'])} |")
    out += ['', '先在工况内平均三个固定种子，再平均全部8工况；缺证书均值为空，不删除样本。完整曲线物理轴0—512点，EARLY-only曲线0—256点；时间轴为worker入口后的检查点，首次完整认证前为空，结束后仅保持已有证书并标记。后置阶梯，无插值、平滑或事后早停。', '',
        '本轮只改变前16步的连续控制，不把单个动作分歧当作独立因果收益。BC/PPO共用教师与BC起点，但历史训练总数据/费用不相等。完整时间属于本轮共同记录实现，不与旧轮日志实现直接比较。', '',
        f"合计真实物理点{summary['total_cost']['physical_points']}、数值入口{summary['total_cost']['numerical_entries']}；新增训练与报告求解均0。", '',
        '状态：`' + STATUS + '`。全部结果、配对、工况汇总、完整与EARLY曲线及来源绑定在同目录保留。', '']
    return '\n'.join(out)


def report(run, output=None):
    run = Path(run).resolve()
    output = Path(output).resolve() if output else run / 'derived_report'
    require(output != run and output.is_relative_to(run) and not output.exists(), 'New within-run output required')
    frozen = stats.verify_manifest(run, 'prephysical_freeze_manifest.json', required=(
        'config.json', 'schedule.json', 'source_binding.json', 'physical_scope.json', 'model_bindings.json'))
    require(read(run / 'prepared.json')['freeze_sha256'] == sha(run / 'prephysical_freeze_manifest.json'), 'Prepared freeze differs')
    require(read(run / 'config.json')['early_study'] == STUDY, 'Wrong early study')
    require(frozen['source/scripts/report_clpu_early_pilot.py'] == sha(Path(__file__)), 'Reporter changed after freeze')
    require(frozen['source/scripts/report_clpu_integrated_pilot.py'] == sha(Path(stats.__file__)), 'Statistics implementation changed')
    stats.verify_manifest(run / 'audit', required=('summary.json',))
    audit, controller = read(run / 'audit/summary.json'), read(run / 'controller_completion.json')
    require(audit['status'] == 'PASS' and audit['errors'] == [] and audit['audited_cells'] == 64
        and audit['new_audit_physical_points'] == audit['new_training_tasks'] == 0, 'All 64 audits required')
    bound = {'freeze_sha256': 'prephysical_freeze_manifest.json', 'config_sha256': 'config.json',
        'schedule_sha256': 'schedule.json', 'source_binding_sha256': 'source_binding.json',
        'controller_completion_sha256': 'controller_completion.json', 'controller_dispatch_sha256': 'controller_dispatch.jsonl',
        'physical_scope_file_sha256': 'physical_scope.json', 'model_bindings_sha256': 'model_bindings.json'}
    for key, name in bound.items():
        require(audit[key] == sha(run / name), 'Audited source binding changed: ' + name)
    canonical_scope = json.dumps(read(run / 'physical_scope.json'), sort_keys=True, separators=(',', ':'), allow_nan=False)
    require(audit['physical_scope_canonical_sha256'] == hashlib.sha256(canonical_scope.encode()).hexdigest(), 'Canonical scope differs')
    schedule = read(run / 'schedule.json')
    validate_population(schedule)
    audited = {r['cell']: r for r in audit['results']}
    require(len(audited) == len(audit['results']) == 64 and set(audited) == {r['cell'] for r in schedule}, 'Audit population changed')
    rows, bindings = [], {name: sha(run / name) for name in (*bound.values(), 'audit/summary.json', 'audit/completion_manifest.json')}
    for spec in schedule:
        check = audited[spec['cell']]
        cell = (run / spec['cell']).resolve()
        require(cell != run and cell.is_relative_to(run), 'Cell path escaped')
        for name, key in (('result.json', 'result_sha256'), ('completion_manifest.json', 'cell_completion_sha256')):
            require(sha(cell / name) == check[key], 'Audited task changed: ' + spec['cell'] + '/' + name)
            bindings[spec['cell'] + '/' + name] = check[key]
        rows.append(case_row(spec, read(cell / 'result.json'), check))
    contexts, pairs = aggregate_contexts(rows), all_pairs(rows)
    require(len(contexts) == 32 and len(pairs) == 152, 'Report omitted comparisons')
    summary = make_summary(rows, contexts, pairs, bindings, controller, run.name)
    early_rows = early_only_rows(rows)
    early_pairs = all_pairs(early_rows, include_guard=False)
    require(len(early_pairs) == 96, 'EARLY-only comparisons omitted')
    summary['early_only_comparisons'] = stats.comparison_summary(early_pairs)
    require(all(summary['total_cost'][k] == audit[k] for k in ('physical_points', 'logical_points', 'numerical_entries')),
        'Aggregated costs differ from audit')
    curves = make_curve_data(rows)
    early_curves = make_curve_data(rows, phase_only=True)
    output.mkdir()
    for name, value in (('all_cells.json', rows), ('all_pairs.json', pairs), ('context_summary.json', contexts),
        ('curve_data.json', curves), ('early_curve_data.json', early_curves), ('summary.json', summary),
        ('early_cells.json', early_rows), ('early_pairs.json', early_pairs), ('report_binding.json', bindings)):
        save(output / name, value)
    (output / 'report.md').write_text(markdown(summary, contexts), encoding='utf-8')
    save(output / 'completion_manifest.json', {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = report(args.run, args.output)
    print(json.dumps({'status': result['status'], 'cells': result['cells'], 'total_cost': result['total_cost']}))
