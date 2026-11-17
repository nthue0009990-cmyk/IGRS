"""Conditional-population, audit-bound staged-candidate comparison report."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import report_clpu_integrated_pilot as stats
from report_clpu_shared_pilot import manifest, checkpoints

read, save, sha, require = stats.read, stats.save, stats.sha, stats.require
STUDY = 'CLPU_STAGED_COMMON_PILOT_V1'
STATUS = 'AUDITED_DESCRIPTIVE_STAGED_COMMON_PILOT'
ARMS = ('staged_window', 'staged_bc')
SEEDS = {'staged_window': (0,), 'staged_bc': (17, 41, 73)}
METRICS = ('quality_pct', 'early_quality_pct', 'guard_quality_pct', 'gain_over_guard_kwh',
    'physical_points', 'logical_points', 'numerical_entries', 'requests', 'process_wall_seconds',
    'worker_elapsed_seconds', 'preflight_elapsed_seconds', 'model_load_seconds',
    'runtime_initialization_seconds', 'computation_elapsed_seconds', 'stage_generation_seconds',
    'early_physical_points', 'guard_physical_points', 'early_verify_attempts', 'policy_decisions')
QUALITY = {'quality_pct', 'early_quality_pct', 'guard_quality_pct', 'gain_over_guard_kwh'}
PRIMARY = ('quality_pct', 'physical_points', 'process_wall_seconds')
RANKING_LIMITATION = (
    'The frozen common window retained the original BranchSearch Beam rule, which first restricts selection '
    'to the shallowest exposed prefix. Complete-plan VERIFY can therefore favor a shorter low-energy HOLD '
    'before a higher-energy full plan. A cheap complete-graph-objective ranking rule was not measured in '
    'this experiment. A BC cost difference against this comparator is not isolated learning benefit or '
    'evidence of superiority over an appropriately strengthened rule baseline.')
RANKING_DIAGNOSTIC = ROOT / 'artifacts/c05_cost_diagnostic_20261031/audited'


def bind_ranking_diagnostic(run, directory=RANKING_DIAGNOSTIC):
    directory = Path(directory).resolve()
    manifest(directory, required=('summary.json', 'c05_cells.json'), all_files=True)
    value = read(directory / 'summary.json')
    require(value['status'] == 'AUDITED_SOURCE_POSTHOC_DESCRIPTIVE' and value['source_run'] == run.name
        and value['analyzed_population_cells'] == 32 and value['new_physical_points'] == value['new_training'] == 0,
        'Ranking limitation needs the complete same-run zero-solver diagnosis')
    require(value['audit_binding']['summary_sha256'] == sha(run / 'audit/summary.json')
        and value['audit_binding']['completion_manifest_sha256'] == sha(run / 'audit/completion_manifest.json'),
        'Ranking diagnosis refers to a different final audit')
    for name in ('controller_completion.json', 'prephysical_freeze_manifest.json', 'schedule.json'):
        require(value['source_hashes'][name] == sha(run / name), 'Ranking diagnosis source changed: ' + name)
    return {'path': directory.relative_to(ROOT).as_posix() if directory.is_relative_to(ROOT) else str(directory),
        'completion_manifest_sha256': sha(directory / 'completion_manifest.json'),
        'summary_sha256': sha(directory / 'summary.json'), 'summary': value}


def population(rows):
    require(len(rows) == 32 and len({r['cell'] for r in rows}) == 32, 'All 32 unique slots must be retained')
    require({r['context_id'] for r in rows} == set(range(8)), 'All eight scheduled contexts required')
    for context in range(8):
        group = [r for r in rows if r['context_id'] == context]
        require(Counter((r['arm'], r['seed']) for r in group) == Counter(
            (arm, seed) for arm in ARMS for seed in SEEDS[arm]), 'Missing or repeated arm/seed')
        require(len({(r['scenario_id'], r['event_id']) for r in group}) == 1, 'Unpaired physical context')


def energy(phase):
    plan = phase.get('best_plan')
    if not phase['fully_certified']:
        return None
    require(plan and plan['certified_complete'] and plan['validation']['status'] == 'COMPLETE',
            'Quality lacks a complete plan certificate')
    return stats.finite(plan['validation']['energy_kwh'], 'certified energy')


def plan_evidence(spec, early):
    """Retain every generated proposal; only verified plan records yield quality."""
    generation = early.get('stage_generation')
    if generation is None:
        return []
    attempts = {tuple(r['actions']): r for r in early['attempts'] if r['origin'] == 'staged_pickup'}
    transitions = {tuple(r['actions']): r for r in early['deferred_trace']
                   if r['operation'] == 'VERIFY_COMPLETE_PLAN'}
    output = []
    for candidate in generation['candidates']:
        key = tuple(candidate['actions'])
        attempt, transition = attempts.get(key), transitions.get(key)
        validation = attempt.get('validation') if attempt else None
        status = attempt['status'] if attempt else transition['verification_status'] if transition else 'NOT_VERIFIED'
        complete = bool(validation and validation['status'] == 'COMPLETE')
        failed = bool(validation and validation['status'] == 'FAIL')
        limit = validation['failed_time_seconds'] if failed else None
        reached = (deepcopy(candidate['pickup_batches']) if complete else
            [deepcopy(b) for b in candidate['pickup_batches'] if b['time_seconds'] <= limit] if failed else None)
        row = {**{k: spec[k] for k in ('cell', 'context_id', 'scenario_id', 'event_id', 'arm', 'seed')},
            'candidate': deepcopy(candidate), 'verification_status': status,
            'fully_certified': complete,
            'quality_pct': 100 * validation['mean_served_fraction'] if complete else None,
            'certified_energy_kwh': validation['energy_kwh'] if complete else None,
            'first_failed_step': validation['failed_step'] if failed else None,
            'first_failed_time_seconds': limit,
            'first_failed_state': deepcopy(validation.get('failed_state')) if failed else None,
            'failure_feedback': deepcopy(validation.get('feedback')) if failed else None,
            'planned_pickup_batches': deepcopy(candidate['pickup_batches']),
            'validation_reached_pickup_batches': reached,
            'new_actual_physical_points': transition['new_actual_physical_points'] if transition else 0,
            'logical_points_before': transition['logical_points_before'] if transition else None,
            'logical_points_after': transition['logical_points_after'] if transition else None,
            'attempt_record': deepcopy(attempt), 'transition': deepcopy(transition),
            'semantics': 'Graph-required pickup times and loads; reached batches refer to offline validation, not deployed restoration or continuous-time safety.'}
        output.append(row)
    return output


def final_plan_description(result, early, loads, load_multiplier):
    """Describe only the selected complete certificate's original whole loads."""
    if not result['fully_certified']:
        return None
    best = result['best_plan']
    validation = best['validation']
    require(best['certified_complete'] and validation['status'] == 'COMPLETE', 'Final plan must be completely certified')
    names = early['control']['adapter']['load_names_in_feature_order']
    require(names == sorted(loads), 'Final pickup identity order differs from original load inventory')
    final = validation['states'][-1]
    require(len(final['pickups']) == len(names), 'Final history dimension changed')
    def load(name, pickup):
        nominal = loads[name]['nominal_kw']
        return {'load_name': name, 'nominal_kw': nominal,
            'condition_scaled_kw': nominal * load_multiplier,
            'pickup_time_seconds': pickup,
            'service_state': 'UNSERVED_FULL_HORIZON' if pickup is None else 'INITIALLY_ENERGIZED' if pickup == -1 else 'RESTORED'}
    inventory = [load(name, pickup) for name, pickup in zip(names, final['pickups'])]
    batches = [{'time_seconds': stamp, 'loads': [deepcopy(row) for row in inventory if row['pickup_time_seconds'] == stamp]}
               for stamp in sorted({p for p in final['pickups'] if p is not None and p >= 0})]
    chosen = next((row for row in (early.get('stage_generation') or {}).get('candidates', [])
                   if row['actions'] == best['actions']), None) if result['selected_phase'] == 'EARLY' else None
    blocked = [row['load_name'] for row in chosen['blocked_loads']] if chosen else []
    return {'selected_phase': result['selected_phase'], 'actions': deepcopy(best['actions']),
        'certified_energy_kwh': validation['energy_kwh'], 'pickup_batches': batches,
        'all_loads': inventory, 'unserved_full_horizon_loads': [r for r in inventory if r['pickup_time_seconds'] is None],
        'staged_blocked_whole_loads': [r for r in inventory if r['load_name'] in blocked],
        'all_loads_restored': all(r['pickup_time_seconds'] is not None for r in inventory),
        'staged_candidate_id': chosen['candidate_id'] if chosen else None,
        'nominal_load_inventory_unchanged': True, 'unserved_loads_remain_in_objective_denominator': True,
        'semantics': 'Certified planning-horizon pickup times; not physical deployment or continuous-time safety.'}


def actual_row(spec, result, phases, checked, exit_row):
    require(result['study'] == STUDY and result['spec'] == spec and result['new_training'] == 0,
            'Result identity or training changed')
    require(result['schema'] == 'clpu-staged-protected-result-v1', 'Wrong staged result schema')
    require(checked['status'] == 'PASS' and exit_row['exit_code'] == 0, 'True exit and audit PASS required')
    for key, result_key in (('physical_points', 'actual_physical_points'), ('logical_points', 'logical_points'),
            ('numerical_entries', 'numerical_entries'), ('quality_pct', 'quality_pct')):
        require(checked[key] == result[result_key], 'Audited metric changed: ' + key)
    physical = stats.integer(result['actual_physical_points'], 'actual cost')
    logical = stats.integer(result['logical_points'], 'logical cost')
    require(physical <= logical <= 512 and result['numerical_entries'] == 3 + 4 * physical,
            'Actual/logical/initialization ledger changed')
    require(result['service']['physical_points'] == physical, 'Service cost mismatch')
    require(sum(p['actual_physical_points'] for p in phases.values()) == physical
            and sum(p['logical_points'] for p in phases.values()) == logical, 'Phase cost attribution changed')
    early, guard = energy(phases['EARLY']), energy(phases['GUARD'])
    require(guard is not None and result['guard_audit']['guard_established'], 'Complete original Beam floor missing')
    final = result['best_plan']['validation']['energy_kwh']
    require(final == max(guard, early if early is not None else guard), 'Protected complete selection changed')
    process = stats.finite(exit_row['process_wall_seconds'], 'complete cold process')
    worker = stats.finite(result['worker_elapsed_seconds'], 'worker time')
    require(process + 1e-6 >= worker, 'Worker time exceeds cold process')
    checkpoints(result['checkpoints'], result)
    row = {**deepcopy(spec), 'executed': True, 'execution_status': 'AUDITED_COMPLETE',
        'fully_certified': result['fully_certified'], 'guard_established': True,
        'quality_pct': result['quality_pct'], 'early_quality_pct': result['early_quality_pct'],
        'guard_quality_pct': result['baseline_quality_pct'], 'gain_over_guard_kwh': final - guard,
        'early_energy_kwh': early, 'guard_energy_kwh': guard, 'final_energy_kwh': final,
        'physical_points': physical, 'logical_points': logical, 'numerical_entries': result['numerical_entries'],
        'process_wall_seconds': process, 'early_physical_points': phases['EARLY']['actual_physical_points'],
        'guard_physical_points': phases['GUARD']['actual_physical_points'],
        'early_verify_attempts': phases['EARLY']['verify_attempts'],
        'new_physical_points': physical, 'new_numerical_entries': result['numerical_entries'],
        'phase_summary': deepcopy(result['phase_summary']), 'checkpoints': deepcopy(result['checkpoints']),
        'stage_generation_sha256': result['stage_generation_sha256'],
        'stage_candidate_count': result['stage_candidate_count'],
        'physical_statuses': deepcopy(checked['physical_statuses']),
        'selected_phase': result['selected_phase'],
        'phase_stop_reasons': {name: value['stop_reason'] for name, value in phases.items()},
        'result_sha256': checked['result_sha256'], 'cell_completion_sha256': checked['cell_completion_sha256']}
    for key in ('requests', 'worker_elapsed_seconds', 'preflight_elapsed_seconds', 'model_load_seconds',
            'runtime_initialization_seconds', 'computation_elapsed_seconds', 'stage_generation_seconds', 'policy_decisions'):
        row[key] = stats.finite(result[key], key)
    return row


def load_cases(run):
    prepared = read(run / 'prepared.json')
    require(prepared['status'] == 'FROZEN_PREPARED_NO_DSS'
            and prepared['freeze_sha256'] == sha(run / 'prephysical_freeze_manifest.json'), 'Prepared freeze changed')
    manifest(run, name='prephysical_freeze_manifest.json',
        required=('config.json', 'schedule.json', 'stage_plan.json', 'physical_scope.json'), all_files=True)
    for name in prepared['source_files']:
        require(sha(ROOT / name) == sha(run / 'source' / name), 'Frozen source changed: ' + name)
    cfg, schedule = read(run / 'config.json'), read(run / 'schedule.json')
    require(cfg['staged_study'] == STUDY, 'Wrong study')
    population(schedule)
    manifest(run / 'audit', required=('summary.json',), all_files=True)
    audit, controller, gate, stage_plan = (read(run / p) for p in
        ('audit/summary.json', 'controller_completion.json', 'phase_gate.json', 'stage_plan.json'))
    require(audit['status'] == 'PASS' and not audit['errors'] and audit['resolved_slots'] == 32,
            'Independent final audit must resolve all slots')
    require(controller['status'] in ('COMPLETE_CELLS_PENDING_AUDIT', 'COMPLETE_PARTIAL_EXPLORATORY_NO_GAIN')
            and controller['scheduled'] == controller['resolved_slots'] == 32
            and not controller['errors'] and not controller['unknown_cost_cells'], 'Controller incomplete or unknown costs')
    for key, path in (('freeze_sha256', 'prephysical_freeze_manifest.json'), ('config_sha256', 'config.json'),
        ('schedule_sha256', 'schedule.json'), ('stage_plan_sha256', 'stage_plan.json'),
        ('controller_completion_sha256', 'controller_completion.json')):
        require(audit[key] == sha(run / path), 'Audit binding changed: ' + path)
    require(controller['phase_gate_sha256'] == sha(run / 'phase_gate.json'), 'Gate binding changed')
    expected = set(stage_plan['phase1_cells']) | (set(stage_plan['phase2_cells']) if gate['phase2_authorized'] else set())
    require(len(controller['completed']) == len(expected) == audit['audited_cells']
            and set(controller['completed']) == expected, 'Conditional executed population changed')
    exits, checks = controller['exits'], audit['results']
    require(len(exits) == len(checks) == len(expected)
            and {r['cell'] for r in exits} == {r['cell'] for r in checks} == expected, 'Missing actual exit/audit')
    exits, checks = {r['cell']: r for r in exits}, {r['cell']: r for r in checks}
    bindings = {p: sha(run / p) for p in ('prephysical_freeze_manifest.json', 'prepared.json', 'config.json',
        'schedule.json', 'stage_plan.json', 'phase_gate.json', 'physical_scope.json', 'source_binding.json',
        'controller_stage1_completion.json', 'controller_completion.json', 'audit_stage1/summary.json',
        'audit_stage1/completion_manifest.json', 'audit/summary.json', 'audit/completion_manifest.json')}
    require(gate['audit_stage1_summary_sha256'] == bindings['audit_stage1/summary.json']
            and gate['audit_stage1_manifest_sha256'] == bindings['audit_stage1/completion_manifest.json'], 'Stage-one gate evidence changed')
    if gate['phase2_authorized']:
        bindings['controller_stage2_completion.json'] = sha(run / 'controller_stage2_completion.json')
    rows, plans, pools = [], [], deepcopy(stage_plan['contexts'])
    loads, condition = read(run / 'load_reference.json'), read(run / 'condition.json')
    bindings['load_reference.json'] = sha(run / 'load_reference.json')
    bindings['condition.json'] = sha(run / 'condition.json')
    for spec in schedule:
        cell = (run / spec['cell']).resolve()
        require(cell.is_relative_to(run) and cell != run, 'Cell path escaped run')
        if spec['cell'] not in expected:
            skip = read(cell / 'not_run.json')
            require(skip['status'] == 'NOT_RUN_GATE_NO_GAIN' and skip['executed'] is False
                    and not any((cell / name).exists() for name in ('result.json', 'worker_started.json', 'execution.json')),
                    'Unexecuted slot has physical evidence or wrong disposition')
            row = {**deepcopy(spec), **{k: None for k in METRICS}, 'executed': False,
                'execution_status': skip['status'], 'fully_certified': None, 'guard_established': None,
                'new_physical_points': 0, 'new_numerical_entries': 0, 'checkpoints': []}
            bindings[spec['cell'] + '/not_run.json'] = sha(cell / 'not_run.json')
        else:
            check = checks[spec['cell']]
            require(check['cell_completion_sha256'] == sha(cell / 'completion_manifest.json'), 'Audited cell seal changed')
            required = ('spec.json', 'result.json', 'cost_checkpoints.jsonl', 'service_summary.json',
                'EARLY/phase_result.json', 'GUARD/phase_result.json', 'EARLY/candidate_pool.json')
            manifest(cell, required=required)
            require(check['result_sha256'] == sha(cell / 'result.json') and read(cell / 'spec.json') == spec,
                    'Audited result or identity changed')
            result = read(cell / 'result.json')
            phases = {name: read(cell / name / 'phase_result.json') for name in ('EARLY', 'GUARD')}
            require(read(cell / 'service_summary.json') == result['service'], 'Service journal differs')
            require([json.loads(x) for x in (cell / 'cost_checkpoints.jsonl').read_text(encoding='utf8').splitlines()]
                    == result['checkpoints'], 'Checkpoint journal differs')
            row = actual_row(spec, result, phases, check, exits[spec['cell']])
            row['final_plan_summary'] = final_plan_description(result, phases['EARLY'], loads, condition['load_multiplier'])
            plans.extend(plan_evidence(spec, phases['EARLY']))
            expected_pool = next(p['generation'] for p in pools if p['context_id'] == spec['context_id'])
            require(expected_pool == phases['EARLY']['stage_generation'], 'Cold common pool changed')
            for path in required + ('completion_manifest.json',):
                bindings[spec['cell'] + '/' + path] = sha(cell / path)
        rows.append(row)
    for key in ('physical_points', 'logical_points', 'numerical_entries'):
        require(sum(r[key] for r in rows if r['executed']) == audit[key] == controller[key], 'Cost total changed: ' + key)
    return rows, plans, pools, bindings, controller, gate


def aggregate(rows):
    population(rows)
    contexts = []
    for context in range(8):
        group = [r for r in rows if r['context_id'] == context]
        require(sum(r['executed'] for r in group) in (0, 4), 'A partial context cannot be averaged')
        for arm in ARMS:
            cells = [r for r in group if r['arm'] == arm]
            contexts.append({'context_id': context, 'scenario_id': cells[0]['scenario_id'],
                'event_id': cells[0]['event_id'], 'arm': arm, 'executed': all(r['executed'] for r in cells),
                'seeds': list(SEEDS[arm]), 'cells': [r['cell'] for r in cells],
                'metrics': {key: stats.strict_mean([r[key] for r in cells], len(cells))[0] for key in METRICS}})
    selected = sorted({r['context_id'] for r in contexts if r['executed']})
    groups = {arm: {'evaluated_context_ids': selected, 'contexts': len(selected),
        'context_balanced_metrics': {key: stats.distribution([r['metrics'][key] for r in contexts
            if r['arm'] == arm and r['executed']], len(selected)) for key in METRICS},
        'all_eight_context_means': {key: stats.strict_mean([r['metrics'][key] for r in contexts
            if r['arm'] == arm], 8)[0] for key in METRICS}}
        for arm in ARMS}
    return contexts, groups, selected


def relation(delta, metric):
    if delta is None:
        return 'missing'
    if abs(delta) <= (stats.TIE_PP if metric in QUALITY else 0):
        return 'tie'
    return 'win' if (delta > 0 if metric in QUALITY else delta < 0) else 'loss'


def comparisons(rows, contexts, selected):
    lookup = {(r['context_id'], r['arm'], r['seed']): r for r in rows}
    all_pairs = []
    for context in range(8):
        rule = lookup[context, 'staged_window', 0]
        for seed in SEEDS['staged_bc']:
            bc = lookup[context, 'staged_bc', seed]
            delta = {key: bc[key] - rule[key] if bc[key] is not None and rule[key] is not None else None for key in METRICS}
            all_pairs.append({'comparison': 'bc_minus_common_window', 'context_id': context, 'seed': seed,
                'left_cell': bc['cell'], 'right_cell': rule['cell'], 'executed': bc['executed'] and rule['executed'],
                'signed_delta': delta, 'metric_relations': {k: relation(delta[k], k) for k in METRICS}})
    clookup = {(r['context_id'], r['arm']): r for r in contexts}
    summary = {}
    for key in METRICS:
        left = [clookup[c, 'staged_bc']['metrics'][key] for c in selected]
        right = [clookup[c, 'staged_window']['metrics'][key] for c in selected]
        delta = [a - b if a is not None and b is not None else None for a, b in zip(left, right)]
        lm, _ = stats.strict_mean(left, len(selected)); rm, _ = stats.strict_mean(right, len(selected))
        complete = len(delta) == len(selected) and all(d is not None for d in delta)
        worst = (min(delta) if key in QUALITY else max(delta)) if complete and delta else None
        summary[key] = {'context_balanced_signed_delta': stats.distribution(delta, len(selected)),
            'relative_change_pct_ratio_of_means': None if lm is None or rm in (None, 0) else 100 * (lm / rm - 1),
            'context_win_tie_loss_missing': dict(Counter(relation(d, key) for d in delta)),
            'scheduled_seed_win_tie_loss_missing': dict(Counter(p['metric_relations'][key] for p in all_pairs)),
            'executed_seed_win_tie_loss_missing': dict(Counter(p['metric_relations'][key] for p in all_pairs if p['executed'])),
            'worst_context_delta': worst,
            'worst_context_ids': [c for c, d in zip(selected, delta) if worst is not None and d == worst],
            'per_seed': {str(seed): stats.distribution([p['signed_delta'][key] for p in all_pairs
                if p['seed'] == seed and p['executed']], len(selected)) for seed in SEEDS['staged_bc']}}
    return all_pairs, summary


def markdown(summary, contexts, rows):
    fmt = lambda x: '未执行/缺失' if x is None else f'{x:.6f}'
    comparison = summary['comparisons']['bc_minus_common_window']
    lines = ['# 共同分批候选池对照', '',
        f"完整输出严格超过自身原Beam的任务：{len(summary['positive_gain_cells'])}/{summary['executed_cells']}；"
        f"完整Beam保护成立：{summary['guard_established_cells']}/{summary['executed_cells']}。",
        'BC相对同样优化窗口的平均差：完整质量 '
        + fmt(comparison['quality_pct']['context_balanced_signed_delta']['mean']) + ' 个百分点，实际验证 '
        + fmt(comparison['physical_points']['context_balanced_signed_delta']['mean']) + ' 点，完整冷进程 '
        + fmt(comparison['process_wall_seconds']['context_balanced_signed_delta']['mean']) + ' 秒。'
        '这些是本轮描述性结果，不作为B蒸馏收益。', '',
        '**比较限制：**公共窗口沿用原Beam的浅前缀优先规则。在完整计划VERIFY中，这会先选较短、较低能量的HOLD。'
        'c05中，按完整图目标直接排序的简单规则也能先选BC选择的较高能量方案；'
        '该简单规则的完整运行费用尚未实测。因此BC相对当前窗口的省点不能归为学习的独立贡献，也不能宣称超过加强后的规则基线。', '',
        f"全部32槽位保留；实际完成并独立核验 {summary['executed_cells']} 项，未执行 {summary['not_run_cells']} 项。",
        f"本轮实测工况：{summary['evaluated_context_ids']}。只有全部32项完成才允许8工况总体均值；本报告不以缺失补零。", '',
        '| 方法 | 完整质量/% | EARLY质量/% | GUARD质量/% | 实际点/任务 | 完整冷进程/s |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        metrics = summary['arms'][arm]['context_balanced_metrics']
        lines.append('| ' + arm + ' | ' + ' | '.join(fmt(metrics[k]['mean']) for k in
            ('quality_pct', 'early_quality_pct', 'guard_quality_pct', 'physical_points', 'process_wall_seconds')) + ' |')
    lines += ['', '均值先在同一工况内平均全部3个BC种子，再对本轮完整实测工况等权。'
        '固定BC只是候选排序迁移，本轮没有B学习器、新训练或已证泛化。', '',
        '| BC减共同窗口 | 平均差 | 相对变化/% | 最差工况差 | 全部已执行种子赢/同/输/缺失 |',
        '|---|---:|---:|---:|---|']
    for key in PRIMARY:
        value = summary['comparisons']['bc_minus_common_window'][key]
        counts = value['executed_seed_win_tie_loss_missing']
        lines.append(f'| {key} | {fmt(value["context_balanced_signed_delta"]["mean"])} | '
            f'{fmt(value["relative_change_pct_ratio_of_means"])} | {fmt(value["worst_context_delta"])} | '
            + '/'.join(str(counts.get(k, 0)) for k in ('win', 'tie', 'loss', 'missing')) + ' |')
    lines += ['', '| 工况 | 方法 | 状态 | 完整质量/% | EARLY质量/% | 实际点 | 完整进程/s |',
        '|---|---|---|---:|---:|---:|---:|']
    for row in contexts:
        lines.append(f'| c{row["context_id"]:02d} | {row["arm"]} | ' + ('已核验' if row['executed'] else '预定门槛未通过，未执行')
            + ' | ' + ' | '.join(fmt(row['metrics'][k]) for k in
                ('quality_pct', 'early_quality_pct', 'physical_points', 'process_wall_seconds')) + ' |')
    lines += ['', f"完整保护成立任务数：{summary['guard_established_cells']}；"
        f"完整输出严格优于自身GUARD的任务数：{len(summary['positive_gain_cells'])}。"
        + ('已按预定认证增益门槛进入第二阶段。' if summary['phase_gate']['phase2_authorized']
           else '未达到预定认证增益门槛，其余任务保持未执行。'), '',
        '全部初始认证、候选生成、验证失败、模型加载、共享服务、完整GUARD和进程退出费用均计入。'
        'GUARD新增实际点是在同任务共享下的归属费用，不是独立冷Beam成本；不与旧双进程日期的耗时拼接。'
        'phase_elapsed截止于阶段结果写盘前，完整进程包括后续关闭、封存和退出。', '',
        'plan_evidence.json保留所有合法候选的验证/未验证状态、完整认证目标、首次失败及计划pickup批次。'
        '失败点之后的计划批次不是已验证或实际部署动作；只将完整认证方案作为最终质量。'
        'candidate_pools.json完整保留非法、重复、负荷不足、超时域、留置不接回及图非改善提议。', '',
        '本轮是8个已观察训练工况上的条件执行探索。初始图潜力筛查与后续真实认证收益是两道不同门槛；'
        '负结果、未执行槽位及所有种子均保留，不宣称SCI等级、创新性或学习贡献已经成立。']
    seen = set()
    winners = []
    for row in rows:
        if not row['executed'] or row['gain_over_guard_kwh'] <= 0:
            continue
        plan = row['final_plan_summary']
        key = (row['context_id'], tuple(plan['actions']))
        if key in seen:
            continue
        seen.add(key)
        details = []
        for load in plan['staged_blocked_whole_loads']:
            timing = ('整个30分钟保持未供电' if load['pickup_time_seconds'] is None
                      else f"在{load['pickup_time_seconds']:g}秒接回")
            details.append(f"{load['load_name']}（原整负荷{load['nominal_kw']:g} kW）{timing}")
        scope = ('未供电负荷仍计入原分母和未供电损失；这不是全部负荷恢复。'
                 if not plan['all_loads_restored'] else '最终已恢复全部原负荷，延迟接入的时间损失仍按原目标计算。')
        winners.append(f"- c{row['context_id']:02d}认证胜出方案：" + '；'.join(details)
            + '。' + scope + '完整批次与原负荷名称见all_cells.json的final_plan_summary。')
    if winners:
        lines += ['', '实际胜出完整方案的负荷范围：', '', *winners]
    return '\n'.join(lines) + '\n'


def report(run, output=None):
    run = Path(run).resolve()
    output = Path(output).resolve() if output else run / 'derived_report'
    require(not output.exists() and output != run and not run.is_relative_to(output), 'New report directory required')
    rows, plans, pools, bindings, controller, gate = load_cases(run)
    ranking_diagnostic = bind_ranking_diagnostic(run)
    contexts, arms, selected = aggregate(rows)
    pairs, comparison = comparisons(rows, contexts, selected)
    actual = [r for r in rows if r['executed']]
    summary = {'status': STATUS, 'study': STUDY, 'run': run.name,
        'utc': datetime.now(timezone.utc).isoformat(), 'scheduled_cells': 32,
        'executed_cells': len(actual), 'not_run_cells': 32 - len(actual),
        'evaluated_context_ids': selected, 'scheduled_contexts': 8,
        'all_eight_context_mean_allowed': len(actual) == 32,
        'arms': arms, 'comparisons': {'bc_minus_common_window': comparison},
        'guard_established_cells': sum(r['guard_established'] for r in actual),
        'positive_gain_cells': [r['cell'] for r in actual if r['gain_over_guard_kwh'] > 0],
        'total_cost': {k: sum(r[k] for r in actual) for k in ('physical_points', 'logical_points', 'numerical_entries', 'requests')},
        'physical_statuses': dict(sum((Counter(r['physical_statuses']) for r in actual), Counter())),
        'candidate_verification_statuses': dict(Counter(r['verification_status'] for r in plans)),
        'phase_gate': deepcopy(gate), 'controller_wall_seconds': controller['wall_seconds'],
        'new_report_physical_points': 0, 'new_training': 0, 'B_learner_present': False,
        'limitations': [RANKING_LIMITATION,
            'Three fixed BC endpoints are repeated seeds within each observed context, not independent faults.',
            'The staged candidate generator is shared by every ranker; a common improvement over GUARD is not a learned contribution.'],
        'comparator_rule': 'original_BranchSearch_Beam_shallowest_prefix_priority',
        'ranking_diagnostic_binding': ranking_diagnostic,
        'source_bindings': bindings,
        'interpretation': 'Conditional exploratory comparison of one common candidate mechanism and fixed BC endpoints; no new learner or confirmatory held-out evidence.'}
    implementations = ('scripts/report_clpu_staged_pilot.py', 'scripts/report_clpu_shared_pilot.py',
        'scripts/report_clpu_integrated_pilot.py', 'tests/test_report_clpu_staged_pilot.py')
    summary['report_source_sha256'] = {name: sha(ROOT / name) for name in implementations}
    output.mkdir(parents=True)
    for name, data in (('summary.json', summary), ('all_cells.json', rows), ('all_pairs.json', pairs),
        ('context_summary.json', contexts), ('plan_evidence.json', plans), ('candidate_pools.json', pools)):
        save(output / name, data)
    (output / 'report.md').write_text(markdown(summary, contexts, rows), encoding='utf8')
    for name in implementations:
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    save(output / 'completion_manifest.json', {p.relative_to(output).as_posix(): sha(p)
        for p in sorted(output.rglob('*')) if p.is_file()})
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = report(args.run, args.output)
    print(json.dumps({k: result[k] for k in ('status', 'scheduled_cells', 'executed_cells', 'not_run_cells')}))
