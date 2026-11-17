"""Audit-bound paired report for a common single-runtime evidence service."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import report_clpu_integrated_pilot as base

read, save, sha, require = base.read, base.save, base.sha, base.require
STUDY = 'CLPU_SHARED_RULE_PILOT_V1'
STATUS = 'AUDITED_DESCRIPTIVE_SHARED_RULE_PILOT'
MODES = ('private', 'shared')
METRICS = ('quality_pct', 'physical_points', 'logical_points', 'numerical_entries',
           'requests', 'process_wall_seconds', 'worker_elapsed_seconds')


def manifest(directory, *, name='completion_manifest.json', required=(), all_files=False):
    """Check paths in the full seal; hash requested members or all members.

Cell raw evidence has already been independently audited. The exact cell seal
is bound to that audit before reading its report inputs; no raw solver replay
or second pass through large raw journals is performed here.
"""
    directory = Path(directory).resolve()
    value = read(directory / name)
    require(isinstance(value, dict) and value, 'Nonempty flat seal required')
    normalized, targets = {}, set()
    for raw, digest in value.items():
        require(isinstance(raw, str) and isinstance(digest, str) and len(digest) == 64
                and all(c in '0123456789abcdef' for c in digest), 'Invalid seal entry')
        key = raw.replace('\\', '/')
        path = (directory / key).resolve()
        require(path != directory and path.is_relative_to(directory) and key not in normalized
                and path not in targets, 'Unsafe or aliased sealed path')
        normalized[key] = digest
        targets.add(path)
    require(set(required) <= set(normalized), 'Missing sealed report input')
    for key in normalized if all_files else required:
        require(sha(directory / key) == normalized[key], 'Changed sealed input: ' + key)
    return normalized


def population(rows):
    require(isinstance(rows, list) and len(rows) == 16 and len({r['cell'] for r in rows}) == 16,
            'All sixteen unique tasks required')
    require({r['context_id'] for r in rows} == set(range(8)), 'All eight contexts required')
    for context in range(8):
        pair = [r for r in rows if r['context_id'] == context]
        require(len(pair) == 2 and {r['mode'] for r in pair} == set(MODES), 'Missing or duplicate paired mode')
        require(len({(r['scenario_id'], r['event_id']) for r in pair}) == 1, 'Mismatched paired physical context')
    require(len({r['scenario_id'] for r in rows}) == 8, 'Fault population changed')
    for row in rows:
        require(row['method'] == 'beam' and row['seed'] == 0 and row['arm'] == 'protected_rule', 'Unexpected learned arm')


def checkpoints(points, result):
    require(isinstance(points, list) and points and len(points) == result['search_decisions'] + 4,
            'Missing phase initial, decision or close checkpoints')
    previous, global_choices, phase_sequence = None, 0, []
    for phase in ('EARLY', 'GUARD'):
        rows = [r for r in points if r['phase'] == phase]
        count = result['phase_summary'][phase]['search_decisions']
        require(len(rows) == count + 2 and [r['kind'] for r in rows] == ['INITIAL'] + ['DECISION'] * count + ['CLOSE'],
                'Phase checkpoint sequence changed')
        for offset, row in enumerate(rows):
            choice = min(offset, count)
            require(row['phase_choices'] == choice and row['search_choices'] == global_choices + choice,
                    'Choice counters reset or omitted')
            phase_sequence.append(phase)
        global_choices += count
    require([p['phase'] for p in points] == phase_sequence, 'Logical phases interleaved')
    for index, row in enumerate(points):
        require(row['checkpoint_index'] == index, 'Checkpoint index changed')
        logical = base.integer(row['logical_points'], 'checkpoint logical')
        actual = base.integer(row['physical_points'], 'checkpoint physical')
        require(actual <= logical <= 512 and row['numerical_entries'] == 3 + 4 * actual,
                'Checkpoint physical or initialization costs changed')
        require(0 <= row['phase_physical_points'] <= row['phase_logical_points'] <= 256,
                'Per-view budget changed')
        require(row['phase_numerical_entries'] == 4 * row['phase_physical_points'], 'Initialization charged twice')
        base.finite(row['worker_elapsed_seconds'], 'checkpoint time')
        require(type(row['fully_certified']) is bool and row['fully_certified'] is (row['quality_pct'] is not None),
                'Missing certificate replaced by a number')
        if row['quality_pct'] is not None:
            require(base.finite(row['quality_pct'], 'checkpoint quality') <= 100 + 1e-9, 'Quality outside range')
        if previous:
            require(all(row[k] >= previous[k] for k in ('logical_points', 'physical_points', 'numerical_entries',
                    'requests', 'worker_elapsed_seconds')), 'Checkpoint cost or time decreased')
            require(previous['quality_pct'] is None or row['quality_pct'] is not None
                    and row['quality_pct'] >= previous['quality_pct'], 'Available certified quality regressed')
        previous = row
    require(all(points[-1][k] == result[k] for k in ('logical_points', 'numerical_entries', 'requests'))
            and points[-1]['physical_points'] == result['actual_physical_points'], 'Final checkpoint ledger differs')
    require(points[-1]['worker_elapsed_seconds'] <= result['worker_elapsed_seconds'], 'Worker excludes final checkpoint')


def case_row(spec, result, audited, exit_record, physical):
    require(result['spec'] == spec and result['study'] == STUDY, 'Result identity differs')
    require(result['new_training'] == result['policy_decisions'] == 0, 'Unexpected learning or updates')
    require(audited['status'] == 'PASS' and audited['cell'] == spec['cell'], 'Task was not audited')
    require(exit_record['exit_code'] == 0 and exit_record['cell'] == spec['cell'], 'Task did not really exit successfully')
    service = result['service']
    require(service['status'] == 'COMPLETE_SHARED_EVIDENCE_PENDING_AUDIT' and service['closed'] is True
            and service['physical_cost_complete'] is True and service['runtime_initializations'] == 1
            and service['mode'] == spec['mode'], 'Incomplete common physical service')
    actual, logical, numerical = result['actual_physical_points'], result['logical_points'], result['numerical_entries']
    require(0 <= actual <= logical <= 512 and service['physical_points'] == actual
            and service['logical_points'] == logical and numerical == service['actual_numerical_entries'] == 3 + 4 * actual,
            'Actual and logical ledgers do not close')
    require(physical['points'] == actual and physical['actual_numerical_entries'] == numerical,
            'Original physical ledger differs')
    require(sum(physical['statuses'].values()) == actual, 'Physical failures omitted')
    guard = result['guard_audit']['guard_established']
    q = result['quality_pct']
    require(type(guard) is bool and result['fully_certified'] is (q is not None), 'Certification availability differs')
    if q is not None:
        require(base.finite(q, 'final quality') <= 100 + 1e-9 and guard, 'Protected quality lacks guard')
    require(all(audited[k] == v for k, v in {'physical_points': actual, 'logical_points': logical,
        'numerical_entries': numerical, 'quality_pct': q, 'guard_established': guard}.items()), 'Audit/result measurements differ')
    elapsed = base.finite(exit_record['process_wall_seconds'], 'complete process wall')
    require(elapsed == audited['process_wall_seconds'] and elapsed >= result['worker_elapsed_seconds'], 'Complete process time differs')
    for key in ('worker_elapsed_seconds', 'preflight_elapsed_seconds', 'runtime_initialization_seconds', 'computation_elapsed_seconds'):
        base.finite(result[key], key)
    require(result['worker_elapsed_seconds'] + 1e-6 >= result['preflight_elapsed_seconds'] + result['computation_elapsed_seconds'],
            'Worker timing components exceed total')
    require(result['computation_elapsed_seconds'] >= result['runtime_initialization_seconds'], 'Initialization excluded')
    for phase in ('EARLY', 'GUARD'):
        phase_row, oracle = result['phase_summary'][phase], service['views'][phase]
        require(phase_row['logical_points'] == oracle['logical_points'] <= 256
                and phase_row['actual_physical_points'] == oracle['physical_points']
                and oracle['actual_numerical_entries_attributed'] == 4 * oracle['physical_points']
                and oracle['initialization_numerical_entries_attributed'] == 0, 'View ledger differs')
    require(sum(x['physical_points'] for x in service['views'].values()) == actual
            and sum(x['logical_points'] for x in service['views'].values()) == logical, 'Shared view costs do not add up')
    checkpoints(result['checkpoints'], result)
    return {**spec, 'quality_pct': q, 'fully_certified': result['fully_certified'], 'guard_established': guard,
        'early_quality_pct': result['early_quality_pct'], 'baseline_quality_pct': result['baseline_quality_pct'],
        'physical_points': actual, 'logical_points': logical, 'numerical_entries': numerical,
        'requests': result['requests'], 'process_wall_seconds': elapsed,
        'worker_elapsed_seconds': result['worker_elapsed_seconds'],
        'preflight_elapsed_seconds': result['preflight_elapsed_seconds'],
        'runtime_initialization_seconds': result['runtime_initialization_seconds'],
        'computation_elapsed_seconds': result['computation_elapsed_seconds'],
        'search_decisions': result['search_decisions'], 'phase_summary': result['phase_summary'],
        'view_ledgers': service['views'], 'physical_statuses': physical['statuses'],
        'cross_view_reuses': sum(v['cross_view_reuses'] for v in service['views'].values()),
        'checkpoints': result['checkpoints']}


def summarize(cases):
    population(cases)
    modes = {mode: {'cells': 8, 'contexts': 8, 'metrics': {metric: base.distribution(
        [c[metric] for c in cases if c['mode'] == mode], 8) for metric in METRICS}} for mode in MODES}
    pairs = []
    for context in range(8):
        pair = {r['mode']: r for r in cases if r['context_id'] == context}
        private, shared = pair['private'], pair['shared']
        delta = {k: None if private[k] is None or shared[k] is None else shared[k] - private[k] for k in METRICS}
        quality_relation = ('missing' if delta['quality_pct'] is None else 'tie' if delta['quality_pct'] == 0
                            else 'better' if delta['quality_pct'] > 0 else 'worse')
        pairs.append({'context_id': context, 'scenario_id': private['scenario_id'], 'event_id': private['event_id'],
            'private_cell': private['cell'], 'shared_cell': shared['cell'], 'signed_delta': delta,
            'quality_relation': quality_relation,
            'private': {k: private[k] for k in METRICS}, 'shared': {k: shared[k] for k in METRICS}})
    relative = {}
    for k in METRICS:
        a, b = (modes[m]['metrics'][k]['mean'] for m in MODES)
        relative[k] = 100 * (b / a - 1) if a is not None and b is not None and a != 0 else None
    comparison = {'direction': 'shared_minus_private', 'pairs': 8,
        'signed_delta': {k: base.distribution([p['signed_delta'][k] for p in pairs], 8) for k in METRICS},
        'relative_change_pct': relative, 'relative_definition': 'ratio_of_eight_context_means',
        'quality_relations': dict(Counter(p['quality_relation'] for p in pairs))}
    return modes, pairs, comparison


def load_cases(run):
    run = Path(run).resolve()
    prepared = read(run / 'prepared.json')
    require(prepared['status'] == 'FROZEN_PREPARED_NO_DSS'
            and sha(run / 'prephysical_freeze_manifest.json') == prepared['freeze_sha256'], 'Freeze binding changed')
    manifest(run, name='prephysical_freeze_manifest.json', required=('config.json', 'schedule.json', 'physical_scope.json'), all_files=True)
    for name in prepared['source_files']:
        require(sha(ROOT / name) == sha(run / 'source' / name), 'Frozen executing source changed: ' + name)
    cfg, schedule = read(run / 'config.json'), read(run / 'schedule.json')
    require(cfg['shared_study'] == STUDY, 'Wrong shared study')
    population(schedule)
    manifest(run / 'audit', required=('summary.json',), all_files=True)
    audit = read(run / 'audit/summary.json')
    require(audit['status'] == 'PASS' and audit['audited_cells'] == 16 and audit['errors'] == [], 'All sixteen audits required')
    bindings = {}
    for key, filename in [('freeze_sha256', 'prephysical_freeze_manifest.json'), ('config_sha256', 'config.json'),
            ('schedule_sha256', 'schedule.json'), ('controller_completion_sha256', 'controller_completion.json')]:
        require(audit[key] == sha(run / filename), 'Audit source changed: ' + filename)
    for name in ('prephysical_freeze_manifest.json', 'prepared.json', 'config.json', 'schedule.json', 'physical_scope.json',
                 'source_binding.json', 'controller_completion.json', 'audit/summary.json', 'audit/completion_manifest.json'):
        bindings[name] = sha(run / name)
    controller = read(run / 'controller_completion.json')
    require(controller['status'] == 'COMPLETE_CELLS_PENDING_AUDIT' and controller['scheduled'] == 16
            and not controller['errors'] and not controller['unknown_cost_cells'], 'Incomplete or failed controller')
    names = {s['cell'] for s in schedule}
    require(len(controller['completed']) == 16 and set(controller['completed']) == names, 'Tasks omitted from completion')
    exits, audited = controller['exits'], audit['results']
    require(len(exits) == len(audited) == 16 and {x['cell'] for x in exits} == {x['cell'] for x in audited} == names,
            'Missing, repeated or extra exits/audits')
    exits, audited = {x['cell']: x for x in exits}, {x['cell']: x for x in audited}
    cases = []
    required = ('spec.json', 'result.json', 'cost_checkpoints.jsonl', 'service_summary.json', 'physical/oracle_summary.json')
    for spec in schedule:
        cell = (run / spec['cell']).resolve()
        require(cell != run and cell.is_relative_to(run), 'Cell escaped run')
        row = audited[spec['cell']]
        require(sha(cell / 'completion_manifest.json') == row['cell_completion_sha256'], 'Audited cell seal changed')
        manifest(cell, required=required)
        require(read(cell / 'spec.json') == spec and sha(cell / 'result.json') == row['result_sha256'], 'Audited result changed')
        result = read(cell / 'result.json')
        require(read(cell / 'service_summary.json') == result['service'], 'Embedded service summary differs')
        stored = [json.loads(line) for line in (cell / 'cost_checkpoints.jsonl').read_text(encoding='utf-8').splitlines()]
        require(stored == result['checkpoints'], 'Embedded checkpoint journal differs')
        cases.append(case_row(spec, result, row, exits[spec['cell']], read(cell / 'physical/oracle_summary.json')))
        for filename in ('result.json', 'completion_manifest.json'):
            bindings[spec['cell'] + '/' + filename] = sha(cell / filename)
    for key in ('physical_points', 'logical_points', 'numerical_entries'):
        require(sum(r[key] for r in cases) == audit[key] == controller[key], 'Audited total cost differs: ' + key)
    return cases, bindings, controller['wall_seconds']


def fmt(value):
    return '未认证／未定义' if value is None else f'{value:.6f}'


def report(run, output=None):
    run = Path(run).resolve()
    output = Path(output).resolve() if output is not None else run / 'derived_report'
    require(not output.exists() and output != run and not run.is_relative_to(output), 'New report directory required')
    cases, bindings, wall = load_cases(run)
    modes, pairs, comparison = summarize(cases)
    summary = {'status': STATUS, 'study': STUDY, 'run': run.name, 'utc': datetime.now(timezone.utc).isoformat(),
        'cells': 16, 'audited_cells': 16, 'contexts': 8, 'modes': modes, 'comparison': comparison,
        'total_cost': {k: sum(c[k] for c in cases) for k in ('physical_points', 'logical_points', 'numerical_entries')},
        'physical_statuses': dict(sum((Counter(c['physical_statuses']) for c in cases), Counter())),
        'guard_established_cells': sum(c['guard_established'] for c in cases),
        'fully_certified_cells': sum(c['fully_certified'] for c in cases),
        'uncertified_cells_retained': [c['cell'] for c in cases if not c['fully_certified']],
        'evaluation_controller_wall_seconds': wall, 'source_bindings': bindings,
        'new_training': 0, 'new_report_physical_points': 0,
        'interpretation': 'Eight observed training contexts, rule only, one runtime in both modes; no PPO attribution or cross-date timing comparison.'}
    curves = {'schema': 'clpu-shared-exact-checkpoints-v1', 'sampled_grid': False,
        'clock': 'worker_entry_completion_checkpoints_not_complete_process_exit',
        'cells': [{k: c[k] for k in ('cell', 'context_id', 'mode', 'scenario_id', 'event_id', 'checkpoints')} for c in cases]}
    lines = ['# 共享物理证据：完整规则保护配对小试', '',
        '全部16任务与独立审计通过；保留8个已观察训练工况、两种模式及全部物理失败。两模式都仅初始化一个物理运行时，并完整执行EARLY窗口与GUARD原Beam；区别仅为跨路证据池是否共享。', '',
        '| 模式 | 最终质量/% | 完整真实点 | 完整进程/秒 |', '|---|---:|---:|---:|']
    for mode in MODES:
        means = modes[mode]['metrics']
        lines.append(f"| {mode} | {fmt(means['quality_pct']['mean'])} | {fmt(means['physical_points']['mean'])} | {fmt(means['process_wall_seconds']['mean'])} |")
    lines += ['', '差值方向为shared−private；质量差单位为百分点。相对变化使用8工况等权均值之比，不挑工况或删除不利配对。', '',
        '| 指标 | 平均差 | 最小配对差 | 最大配对差 | 均值相对变化/% |', '|---|---:|---:|---:|---:|']
    for metric in ('quality_pct', 'physical_points', 'process_wall_seconds'):
        d = comparison['signed_delta'][metric]
        lines.append(f"| {metric} | {fmt(d['mean'])} | {fmt(d['minimum'])} | {fmt(d['maximum'])} | {fmt(comparison['relative_change_pct'][metric])} |")
    lines += ['', '| 工况 | 故障；事件 | private质量/% | shared质量/% | private真实点 | shared真实点 | private完整秒 | shared完整秒 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for pair in pairs:
        a, b = pair['private'], pair['shared']
        lines.append(f"| c{pair['context_id']:02d} | {pair['scenario_id']}；{pair['event_id']} | {fmt(a['quality_pct'])} | {fmt(b['quality_pct'])} | {fmt(a['physical_points'])} | {fmt(b['physical_points'])} | {fmt(a['process_wall_seconds'])} | {fmt(b['process_wall_seconds'])} |")
    lines += ['', '逻辑预算每路256、全局实际预算512保持；初始化3个数值入口只在服务总账计一次，每个实际点4个入口。共享证据仍按当前语义身份与时间重新认证，不伪造第二次物理调用。', '',
        '完整时间包含进程创建至真实退出、初始化、两路搜索及全部记录关闭；worker检查点不是完整交付时间。两模式使用本轮同一实现，不与旧双进程运行计算加速比。', '',
        '若出现缺认证，质量和相关均值保持为空且任务保留。本轮为公共验证层的规则对照，不含PPO、BC、新训练或新正式测试；不能将收益归为强化学习独立贡献。', '',
        f"合计实际点{summary['total_cost']['physical_points']}，数值入口{summary['total_cost']['numerical_entries']}；新增报告求解与训练均0。状态：{STATUS}。"]
    output.mkdir(parents=True)
    for filename, value in [('summary.json', summary), ('all_cells.json', cases), ('all_pairs.json', pairs),
            ('context_summary.json', pairs), ('curve_data.json', curves)]:
        save(output / filename, value)
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    save(output / 'completion_manifest.json', {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = report(args.run, args.output)
    print(json.dumps({'status': result['status'], 'cells': result['cells']}, ensure_ascii=False))
