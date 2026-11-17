"""Audit-bound, descriptive reporting for the two-window training pilot.

No simulator is imported and no measurement is interpolated.  G1's intrinsic
comparison and the paid G0/G1 protection envelope are different estimands.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.dont_write_bytecode = True
from d060_validation_common import inside, read, require, sha, utc_now

GROUPS = ('G0_original_beam', 'G1_coverage_window_beam')
EPS_PP = 1e-9
ANYTIME_STEPS = (16, 64, 128, 256)
DIAGNOSED = {('N2__line.l10__line.l24', 6), ('N2__line.l12__line.l24', 3)}


def finite(value, name, *, minimum=None):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), 'Nonfinite or missing ' + name)
    require(minimum is None or value >= minimum, 'Negative ' + name)
    return float(value)


def integer(value, name):
    require(type(value) is int and value >= 0, 'Invalid nonnegative integer ' + name)
    return value


def validate_checkpoints(checkpoints):
    """Reject missing boundaries, hidden costs, and uncertified numeric quality."""
    require(bool(checkpoints), 'Initialization checkpoint is missing')
    previous = None
    for index, point in enumerate(checkpoints):
        require(integer(point['search_choices'], 'search choices') == index,
                'Every completed search boundary must be recorded exactly once')
        integer(point['physical_points'], 'physical points')
        integer(point['numerical_entries'], 'numerical entries')
        finite(point['elapsed_seconds'], 'cold elapsed seconds', minimum=0)
        require(type(point['fully_certified']) is bool, 'Explicit certificate flag required')
        quality = point['quality_pct']
        if point['fully_certified']:
            finite(quality, 'certified quality', minimum=0)
            require(quality <= 100 + EPS_PP, 'Quality outside percentage range')
        else:
            require(quality is None, 'Uncertified quality must remain null')
        if previous is not None:
            for key in ('physical_points', 'numerical_entries', 'elapsed_seconds'):
                require(point[key] >= previous[key], 'Checkpoint cost decreased: ' + key)
            if previous['fully_certified']:
                require(point['fully_certified'] and quality + EPS_PP >= previous['quality_pct'],
                        'Certified incumbent disappeared or decreased')
        previous = point
    return checkpoints


def first_reach(checkpoints, target_quality_pct, *, strict=False):
    """First observed full certificate; missingness is never encoded as zero."""
    if target_quality_pct is None:
        return {'status': 'TARGET_UNAVAILABLE', 'search_choices': None,
                'physical_points': None, 'numerical_entries': None,
                'elapsed_seconds': None, 'quality_pct': None}
    finite(target_quality_pct, 'target quality', minimum=0)
    for point in checkpoints:
        if not point['fully_certified']:
            continue
        delta = point['quality_pct'] - target_quality_pct
        if (delta > EPS_PP) if strict else (delta >= -EPS_PP):
            return {'status': 'REACHED', **{key: point[key] for key in
                ('search_choices', 'physical_points', 'numerical_entries',
                 'elapsed_seconds', 'quality_pct')}}
    return {'status': 'NOT_REACHED_WITHIN_BUDGET', 'search_choices': None,
            'physical_points': None, 'numerical_entries': None,
            'elapsed_seconds': None, 'quality_pct': None}


def anytime(checkpoints, steps=ANYTIME_STEPS):
    """Carry a completed run's incumbent after termination, clearly labelled."""
    last = checkpoints[-1]
    return {str(step): {
        'logical_step': step,
        'status': ('OBSERVED_SEARCH_BOUNDARY' if step <= last['search_choices'] else
                   'TERMINATED_INCUMBENT_CARRIED_FORWARD'),
        'observed_at_search_choices': checkpoints[min(step, last['search_choices'])]['search_choices'],
        'quality_pct': checkpoints[min(step, last['search_choices'])]['quality_pct'],
        'physical_points': checkpoints[min(step, last['search_choices'])]['physical_points'],
        'elapsed_seconds': checkpoints[min(step, last['search_choices'])]['elapsed_seconds'],
    } for step in steps}


def protected_output(original, coverage):
    """Select only actual full certificates and charge both independent runs."""
    a, b = original['quality_pct'], coverage['quality_pct']
    chosen = original if a is not None else coverage if b is not None else None
    if a is not None and b is not None and coverage['energy_kwh'] > original['energy_kwh']:
        chosen = coverage
    return {'status': 'CERTIFIED_PROTECTED_OUTPUT' if chosen else 'NO_CERTIFIED_OUTPUT',
        'selected_group': chosen['group'] if chosen else None,
        'selected_cell': chosen['cell'] if chosen else None,
        'quality_pct': chosen['quality_pct'] if chosen else None,
        'energy_kwh': chosen['energy_kwh'] if chosen else None,
        'best_plan': chosen['best_plan'] if chosen else None,
        'source_cell_completion_sha256': {
            row['cell']: row['cell_completion_sha256'] for row in (original, coverage)},
        'source_result_sha256': {row['cell']: row['result_sha256'] for row in (original, coverage)},
        'gain_over_original_pp': (chosen['quality_pct'] - a) if chosen and a is not None else None,
        'physical_points': original['physical_points'] + coverage['physical_points'],
        'numerical_entries': original['numerical_entries'] + coverage['numerical_entries'],
        'sum_worker_elapsed_seconds': original['elapsed_seconds'] + coverage['elapsed_seconds'],
        'search_choices': original['search_choices'] + coverage['search_choices'],
        'time_semantics': 'Both cold workers charged; sum is not concurrent wall-clock latency',
        'quality_tie_rule': 'Compare actual certified energy_kwh exactly; exact tie retains original G0'}


def pair_cases(original, coverage):
    require(original['group'] == GROUPS[0] and coverage['group'] == GROUPS[1], 'Pair groups differ')
    require((original['scenario_id'], original['event_id']) ==
            (coverage['scenario_id'], coverage['event_id']), 'Different contexts paired')
    require(original['context_id'] == coverage['context_id'], 'Context IDs differ')
    target = original['quality_pct']
    first_a = first_reach(original['checkpoints'], target)
    first_b = first_reach(coverage['checkpoints'], target)
    both = first_a['status'] == first_b['status'] == 'REACHED'
    complete = target is not None and coverage['quality_pct'] is not None
    gap = coverage['quality_pct'] - target if complete else None
    key = original['scenario_id'], original['event_id']
    return {'context_id': original['context_id'], 'scenario_id': key[0], 'event_id': key[1],
        'previously_diagnosed_context': key in DIAGNOSED,
        'original_cell': original['cell'], 'coverage_cell': coverage['cell'],
        'target_original_final_quality_pct': target,
        'first_target_reach': {GROUPS[0]: first_a, GROUPS[1]: first_b},
        'first_target_reach_G1_minus_G0': {key: first_b[key] - first_a[key] if both else None
            for key in ('search_choices', 'physical_points', 'numerical_entries', 'elapsed_seconds')},
        'first_strict_gain_over_wait': {row['group']: first_reach(
            row['checkpoints'], row['wait_baseline_quality_pct'], strict=True)
            for row in (original, coverage)},
        'final_quality_G1_minus_G0_pp': gap,
        'final_quality_comparison': ('UNAVAILABLE' if gap is None else 'G1_BETTER' if gap > EPS_PP
            else 'G1_WORSE' if gap < -EPS_PP else 'NUMERICAL_TIE'),
        'final_cost_G1_minus_G0': {key: coverage[key] - original[key]
            for key in ('physical_points', 'numerical_entries', 'elapsed_seconds', 'search_choices')},
        'anytime_quality': {row['group']: anytime(row['checkpoints']) for row in (original, coverage)},
        'protected_output': protected_output(original, coverage)}


def mean(values):
    return statistics.mean(values) if values else None


def describe_pairs(pairs):
    gaps = [p['final_quality_G1_minus_G0_pp'] for p in pairs
            if p['final_quality_G1_minus_G0_pp'] is not None]
    reaches = {group: Counter(p['first_target_reach'][group]['status'] for p in pairs)
               for group in GROUPS}
    differences = {}
    for metric in ('search_choices', 'physical_points', 'elapsed_seconds'):
        values = [p['first_target_reach_G1_minus_G0'][metric] for p in pairs
                  if p['first_target_reach_G1_minus_G0'][metric] is not None]
        differences[metric] = {'complete_pairs': len(values), 'missing_pairs': len(pairs) - len(values),
            'mean_G1_minus_G0': mean(values),
            'G1_lower': sum(v < 0 for v in values),
            'tied': sum(v == 0 for v in values),
            'G1_higher': sum(v > 0 for v in values)}
    return {'pairs': len(pairs), 'final_complete_pairs': len(gaps),
        'mean_final_quality_G1_minus_G0_pp': mean(gaps),
        'worst_final_quality_G1_minus_G0_pp': min(gaps) if gaps else None,
        'final_comparison_counts': dict(Counter(p['final_quality_comparison'] for p in pairs)),
        'target_reach_statuses': {group: dict(counts) for group, counts in reaches.items()},
        'earlier_target_reach': differences,
        'interpretation': 'Descriptive training mechanism results; no generalization, ML or significance claim'}


def save(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def verify_manifest(directory, *, required=()):
    manifest = read(directory / 'completion_manifest.json')
    for name in required:
        require(name in manifest, 'Missing sealed evidence: ' + name)
    for name, digest in manifest.items():
        require(sha(inside(directory, name)) == digest, 'Changed sealed evidence: ' + name)
    return manifest


def certified_quality(result):
    """Only the audited full-plan certificate defines an output quality."""
    if result['status'] != 'CERTIFIED_INCUMBENT':
        require(result.get('best_plan') is None, 'Uncertified result carries a best plan')
        return None
    plan = result.get('best_plan')
    require(plan and plan['certified_complete'] and plan['status'] == 'COMPLETE'
            and plan['validation']['status'] == 'COMPLETE', 'Incomplete final certificate')
    value = 100 * finite(plan['validation']['mean_served_fraction'], 'final quality', minimum=0)
    require(value <= 100 + EPS_PP, 'Final quality exceeds 100 percent')
    finite(plan['validation']['energy_kwh'], 'certified energy', minimum=0)
    return value


def case_row(spec, result):
    require(result['spec'] == spec, 'Result does not match frozen cell schedule')
    require(result['optimizer_updates'] == 0 and not result['formal_holdout_numerical_access']
            and not result['tune_numerical_access'], 'Training/held-out scope changed')
    require(spec['group'] in GROUPS, 'Unexpected window group')
    checkpoints = validate_checkpoints(result['checkpoints'])
    quality = certified_quality(result)
    last = checkpoints[-1]
    require((quality is None and last['quality_pct'] is None) or
            (quality is not None and last['quality_pct'] is not None
             and abs(quality - last['quality_pct']) <= EPS_PP),
            'Final certificate differs from last observed boundary')
    points = integer(result['oracle']['points'], 'total physical points')
    entries = integer(result['oracle']['actual_numerical_entries'], 'total numerical entries')
    choices = integer(result['completed_expansions'], 'completed expansions')
    elapsed = finite(result['elapsed_seconds'], 'final elapsed seconds', minimum=0)
    require(last['physical_points'] == points and last['numerical_entries'] == entries
            and last['search_choices'] == choices and elapsed >= last['elapsed_seconds'],
            'Boundary costs do not match completed worker')
    require(points <= 256 and choices <= 256, 'Per-cell frozen budget exceeded')
    baseline = result['wait_baseline_quality_pct']
    initial = checkpoints[0]['quality_pct']
    require((baseline is None and initial is None) or
            (baseline is not None and initial is not None and abs(baseline - initial) <= EPS_PP),
            'Initial quality is not the complete WAIT baseline')
    return {**spec, 'status': result['status'], 'quality_pct': quality,
        'energy_kwh': result['best_plan']['validation']['energy_kwh'] if quality is not None else None,
        'best_plan': result.get('best_plan'),
        'wait_baseline_quality_pct': baseline, 'physical_points': points,
        'numerical_entries': entries, 'search_choices': choices, 'elapsed_seconds': elapsed,
        'stop_reason': result['stop_reason'], 'checkpoints': checkpoints,
        'previously_diagnosed_context': (spec['scenario_id'], spec['event_id']) in DIAGNOSED}


def validate_population(schedule, population):
    expected = {(c['scenario_id'], c['event_id']) for c in population['contexts']}
    require(len(population['contexts']) == len(expected) == 8, 'Expected all eight training contexts')
    require(len({s for s, _ in expected}) == 8 and
            Counter(e for _, e in expected) == Counter({30: 2, 20: 2, 6: 2, 3: 2}),
            'Eight-fault balanced training population changed')
    actual = [(s['scenario_id'], s['event_id'], s['group']) for s in schedule]
    require(len(schedule) == 16 and len({s['cell'] for s in schedule}) == 16
            and len(set(actual)) == 16 and set(actual) ==
            {(s, e, g) for s, e in expected for g in GROUPS}, 'Missing, repeated or extra scheduled cell')
    require(DIAGNOSED <= expected, 'Previously diagnosed pair flags lost')


def report(run, output=None):
    run = Path(run).resolve()
    output = Path(output).resolve() if output else run / 'derived_report'
    require(output != run and output.is_relative_to(run), 'Derived report must remain within its run')
    require(not output.exists(), 'Do not overwrite a prior scientific report')
    ctl = read(run / 'controller_completion.json')
    require(ctl['status'] == 'COMPLETE_CELLS_PENDING_AUDIT' and not ctl['errors']
            and not ctl['unknown_cost_cells'], 'Controller completion/cost is not closed')
    audit = read(run / 'audit/summary.json')
    require(audit['status'] == 'PASS' and audit['audited_cells'] == 16, 'Independent audit incomplete')
    verify_manifest(run / 'audit', required=('summary.json',))
    for key, name in [('schedule_sha256', 'schedule.json'),
                      ('freeze_sha256', 'prephysical_freeze_manifest.json'),
                      ('controller_completion_sha256', 'controller_completion.json')]:
        require(audit[key] == sha(run / name), 'Audit binding differs: ' + name)
    schedule = read(run / 'schedule.json')
    validate_population(schedule, read(run / 'population.json'))
    expected_cells = {spec['cell'] for spec in schedule}
    require(len(ctl['completed']) == 16 and set(ctl['completed']) == expected_cells,
            'Controller did not close exactly the scheduled population')
    bindings = {item['cell']: item for item in audit['results']}
    require(len(audit['results']) == len(bindings) == 16 and set(bindings) == expected_cells,
            'Independent audit omitted/repeated/added a scheduled cell')
    cases = []
    for spec in schedule:
        cell = inside(run, spec['cell'])
        require(bindings[spec['cell']]['cell_completion_sha256'] == sha(cell / 'completion_manifest.json'),
                'Audited cell seal differs')
        manifest = read(cell / 'completion_manifest.json')
        for name in ('result.json', 'spec.json'):
            require(name in manifest and manifest[name] == sha(cell / name), 'Changed sealed ' + name)
        require(read(cell / 'spec.json') == spec, 'Saved specification differs from schedule')
        row = case_row(spec, read(cell / 'result.json'))
        row['result_sha256'] = sha(cell / 'result.json')
        row['cell_completion_sha256'] = sha(cell / 'completion_manifest.json')
        cases.append(row)
    points = sum(row['physical_points'] for row in cases)
    entries = sum(row['numerical_entries'] for row in cases)
    require(points == ctl['known_physical_points'] == audit['physical_points'] and
            entries == ctl['known_numerical_entries'] == audit['numerical_entries'],
            'Controller, physical audit and reporting costs differ')
    lookup = {(r['scenario_id'], r['event_id'], r['group']): r for r in cases}
    pairs = sorted([pair_cases(row, lookup[(row['scenario_id'], row['event_id'], GROUPS[1])])
                    for row in cases if row['group'] == GROUPS[0]], key=lambda p: p['context_id'])
    summary = {'status': 'AUDITED_DESCRIPTIVE_WINDOW_PILOT', 'utc': utc_now(), 'run': run.name,
        'cold_evaluations': 16, 'training_contexts': 8, 'physical_points': points,
        'numerical_entries': entries, 'maximum_physical_points': 4096,
        'new_training_tasks': 0, 'new_optimizer_updates': 0,
        'quality_tolerance_pp': EPS_PP, 'anytime_logical_steps': list(ANYTIME_STEPS),
        'primary_target': 'First observed full certificate at paired G0 final quality minus 1e-9 pp',
        'intrinsic_all_eight_pairs': describe_pairs(pairs),
        'diagnosed_two_descriptive_only': describe_pairs([p for p in pairs if p['previously_diagnosed_context']]),
        'remaining_six_descriptive_only': describe_pairs([p for p in pairs if not p['previously_diagnosed_context']]),
        'groups': {g: {'cases': 8, 'certified_cases': sum(r['quality_pct'] is not None for r in cases if r['group'] == g),
            'missing_quality_cases': sum(r['quality_pct'] is None for r in cases if r['group'] == g),
            'mean_is_conditional_on_certification': any(r['quality_pct'] is None for r in cases if r['group'] == g),
            'mean_final_quality_pct': mean([r['quality_pct'] for r in cases if r['group'] == g and r['quality_pct'] is not None]),
            'physical_points': sum(r['physical_points'] for r in cases if r['group'] == g),
            'numerical_entries': sum(r['numerical_entries'] for r in cases if r['group'] == g),
            'sum_worker_elapsed_seconds': sum(r['elapsed_seconds'] for r in cases if r['group'] == g)}
            for g in GROUPS},
        'protected_actual_output': {'pairs': 8,
            'certified_pairs': sum(p['protected_output']['quality_pct'] is not None for p in pairs),
            'selected_groups': dict(Counter(p['protected_output']['selected_group'] for p in pairs)),
            'physical_points': sum(p['protected_output']['physical_points'] for p in pairs),
            'numerical_entries': sum(p['protected_output']['numerical_entries'] for p in pairs),
            'sum_worker_elapsed_seconds': sum(p['protected_output']['sum_worker_elapsed_seconds'] for p in pairs),
            'claim': 'Actual final quality guard with both complete cold-run costs charged; no total speedup claim'},
        'evaluation_wall_seconds': finite(ctl['wall_seconds'], 'controller wall seconds', minimum=0),
        'sum_worker_elapsed_seconds': sum(r['elapsed_seconds'] for r in cases),
        'source_bindings': {name: sha(run / name) for name in
            ('schedule.json', 'population.json', 'prephysical_freeze_manifest.json',
             'controller_completion.json', 'audit/summary.json', 'audit/completion_manifest.json')},
        'scope': 'Eight previously observed training contexts; no seed repeats, learning or formal generalization',
        'missing_data': 'Unreached target and unavailable certification retained as null with explicit status',
        'next_action': 'No automatic population expansion, additional budget or training'}
    output.mkdir(parents=True, exist_ok=False)
    protected_dir = output / 'protected_outputs'
    protected_dir.mkdir()
    for pair in pairs:
        name = f"protected_outputs/context_{pair['context_id']:02d}.json"
        save(output / name, {'context_id': pair['context_id'],
            'scenario_id': pair['scenario_id'], 'event_id': pair['event_id'],
            **pair['protected_output']})
        pair['protected_output_file'] = name
    save(output / 'all_cases.json', cases)
    save(output / 'paired_cases.json', pairs)
    save(output / 'summary.json', summary)
    (output / 'report.md').write_text(markdown(summary, cases, pairs), encoding='utf-8')
    save(output / 'completion_manifest.json', {p.relative_to(output).as_posix(): sha(p)
         for p in output.rglob('*') if p.is_file()})
    return summary


def _format(value, digits=3):
    return '未达到/不可用' if value is None else f'{value:.{digits}f}'


def markdown(summary, cases, pairs):
    full = summary['intrinsic_all_eight_pairs']
    earlier = full['earlier_target_reach']['search_choices']
    lines = ['# 候选窗口机制小试', '',
        f"完成16次独立冷启动及16次独立审计，覆盖全部8个既有训练工况。实际{summary['physical_points']}物理点、{summary['numerical_entries']}数值入口；新增训练0。", '',
        f"按首次达到本次G0最终质量所需搜索选择数，G1在{earlier['G1_lower']}对更早、{earlier['tied']}对相同、{earlier['G1_higher']}对更晚；另{earlier['missing_pairs']}对未达到或目标缺失，全部保留在8对分母。", '',
        f"同预算G1相对G0最终累计供电平均差{_format(full['mean_final_quality_G1_minus_G0_pp'], 6)}个百分点，最差差值{_format(full['worst_final_quality_G1_minus_G0_pp'], 6)}个百分点；完整胜/平/负计数见summary.json。", '',
        '主指标是首次达到各工况本次G0最终完整认证质量的实际观测边界，同时报告搜索选择数、物理点与从冷worker开始的记录时间。仅初始化后及完整搜索步后可命中；不插值、不从未认证前缀推算。质量比较容差为1e-9个百分点。', '',
        '| 训练工况 | 已诊断 | 最终G1−G0/pp | 达标选择数 G0/G1 | 达标点数 G0/G1 | 达标秒数 G0/G1 |',
        '|---|---|---:|---:|---:|---:|']
    for pair in pairs:
        a, b = (pair['first_target_reach'][g] for g in GROUPS)
        name = pair['scenario_id'].replace('N2__line.', '').replace('__line.', '/') + f" · {pair['event_id']}"
        lines.append(f"| {name} | {'是' if pair['previously_diagnosed_context'] else '否'} | "
            f"{_format(pair['final_quality_G1_minus_G0_pp'], 6)} | "
            f"{_format(a['search_choices'], 0)}/{_format(b['search_choices'], 0)} | "
            f"{_format(a['physical_points'], 0)}/{_format(b['physical_points'], 0)} | "
            f"{_format(a['elapsed_seconds'])}/{_format(b['elapsed_seconds'])} |")
    protection = summary['protected_actual_output']
    lines += ['',
        '两组均独立使用≤256物理点、≤256搜索扩展、≤5000图候选及32候选窗口。同预算效果只比较原始G1与原始G0；最终保护输出按完整证书的供电能量严格择大，精确并列保留G0。每工况实际方案及双方封存绑定另存protected_outputs/。', '',
        f"保护输出的两组物理点合计为{protection['physical_points']}，两组worker时间合计{protection['sum_worker_elapsed_seconds']:.3f}秒；已计入完整原Beam成本，不能称为免费保底或总体提速。两worker累计时间也不等于并行墙钟时延。", '',
        f"整批评价墙钟{summary['evaluation_wall_seconds']:.3f}秒；独立审计成本另列于audit/summary.json。冷worker时间包含输入处理、搜索及记录，进程启动器与控制器成本由整批墙钟补充。", '',
        '全部8对构成主描述；两个人口内已诊断工况单列为结果后定位子集，其余6对亦保留。这里没有独立测试、种子重复、bootstrap或学习器，因此不作泛化、统计显著或机器学习增益声明。', '',
        'all_cases.json保存全部逐项结果与初始化/逐步轨迹；paired_cases.json保存首次相对WAIT严格改善、16/64/128/256步累计认证质量、全部有符号差及保护输出。提前结束后的质量为明确标注的终态保持；未达标保留null和状态，不能填零或从分母剔除。', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    print(json.dumps(report(args.run, args.output), ensure_ascii=False, allow_nan=False))

