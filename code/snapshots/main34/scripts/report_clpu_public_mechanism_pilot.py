"""All-24 audited P/E/A_prune report; no solver or experiment is invoked.

Read only sealed small results and ledger summaries after the independent audit.
Raw physical journals are not reinterpreted, and historical clocks are excluded.
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
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.dont_write_bytecode = True
import report_clpu_shared_pilot as shared
import report_clpu_staged_pilot as staged

read, save, sha, require, manifest = shared.read, shared.save, shared.sha, shared.require, shared.manifest
STUDY = 'CLPU_PUBLIC_MECHANISM_COMMON_PILOT_V1'
SCHEMA = 'clpu-public-mechanism-protected-result-v1'
STATUS = 'AUDITED_DESCRIPTIVE_PUBLIC_MECHANISM_COMMON_PILOT'
ARMS = ('P', 'E', 'A_prune')
PRIMARY = ('quality_pct', 'physical_points', 'process_wall_seconds')
COSTS = ('physical_points', 'logical_points', 'numerical_entries', 'requests')
MECHANISMS = ('free_pruned_complete_plans', 'own_incumbent_verify_skips',
    'early_candidate_complete', 'early_candidate_fail', 'early_zero_actual_candidate_fail',
    'early_positive_actual_candidate_fail', 'early_other_verify_outcomes',
    'physical_pass', 'physical_fail', 'physical_unknown',
    'semantic_cache_hits', 'electrical_reuses', 'cross_view_reuses')
METRICS = PRIMARY + ('final_energy_kwh', 'guard_quality_pct', 'early_quality_pct',
    'gain_over_guard_kwh', 'logical_points', 'numerical_entries', 'requests',
    'worker_elapsed_seconds', 'early_physical_points', 'guard_physical_points',
    'early_verify_attempts') + MECHANISMS
AUDIT_FILES = {
    'freeze_sha256': 'prephysical_freeze_manifest.json', 'config_sha256': 'config.json',
    'schedule_sha256': 'schedule.json', 'stage_plan_sha256': 'stage_plan.json',
    'source_binding_sha256': 'source_binding.json', 'physical_scope_file_sha256': 'physical_scope.json',
    'comparison_source_binding_sha256': 'comparison_source_binding.json',
    'controller_completion_sha256': 'controller_completion.json',
    'controller_execution_sha256': 'controller_stage1_completion.json',
}


def population(rows):
    require(isinstance(rows, list) and len(rows) == 24 and len({r['cell'] for r in rows}) == 24,
        'All 24 unique actual tasks required')
    require({r['context_id'] for r in rows} == set(range(8)), 'All eight observed contexts required')
    for context in range(8):
        group = [r for r in rows if r['context_id'] == context]
        require(Counter((r['arm'], r['seed']) for r in group) == Counter((a, 0) for a in ARMS),
            'Missing or repeated deterministic arm')
        require(len({(r['scenario_id'], r['event_id']) for r in group}) == 1, 'Unpaired physical context')
    require(all(r['method'] == 'beam' for r in rows), 'No learned arm belongs to this batch')


def checkpoint_check(result):
    order = ['EARLY', 'GUARD'] if result['spec']['arm'] == 'E' else ['GUARD', 'EARLY']
    require(result['phase_order'] == order, 'Wrong causal phase order')
    points = result['checkpoints']
    if order[0] == 'GUARD':
        swap = {'GUARD': 'EARLY', 'EARLY': 'GUARD'}
        shared.checkpoints([{**p, 'phase': swap[p['phase']]} for p in points],
            {**result, 'phase_summary': {swap[k]: v for k, v in result['phase_summary'].items()}})
    else:
        shared.checkpoints(points, result)
    established = False
    for point in points:
        if point['phase'] == 'GUARD' and point['kind'] == 'CLOSE':
            require(point['fully_certified'], 'GUARD close lacks a certificate')
            established = True
        require(point['guard_established'] is established, 'GUARD floor claimed before actual completion')
    require(established, 'Complete GUARD missing')


def mechanism_counts(early, service, physical):
    trace = [e for e in early['deferred_trace'] if e['operation'] == 'VERIFY_COMPLETE_PLAN']
    require(len(trace) == early['verify_attempts'], 'VERIFY decisions omitted')
    states = Counter(e['verification_status'] for e in trace)
    free = early.get('guard_first', {}).get('free_pruning', [])
    require(all(p['status'] == 'PRUNED_CERTIFIED_COMPLETE_OBJECTIVE'
        and p['physical_feasibility'] == 'UNASSESSED' and p['consumed_verification_choice'] is False for p in free),
        'Free objective pruning is not physical failure')
    values = {'free_pruned_complete_plans': len(free),
        'own_incumbent_verify_skips': states['SKIPPED_DOMINATED'],
        'early_candidate_complete': states['COMPLETE'], 'early_candidate_fail': states['FAIL'],
        'early_zero_actual_candidate_fail': sum(e['verification_status'] == 'FAIL'
            and e['new_actual_physical_points'] == 0 for e in trace),
        'early_positive_actual_candidate_fail': sum(e['verification_status'] == 'FAIL'
            and e['new_actual_physical_points'] > 0 for e in trace),
        'early_other_verify_outcomes': sum(v for k, v in states.items() if k not in ('COMPLETE', 'FAIL', 'SKIPPED_DOMINATED')),
        'physical_pass': physical['statuses'].get('PASS', 0),
        'physical_fail': physical['statuses'].get('FAIL', 0),
        'physical_unknown': physical['statuses'].get('UNKNOWN', 0)}
    require(set(physical['statuses']) <= {'PASS', 'FAIL', 'UNKNOWN'}, 'Unrecognized raw physical status')
    for key in ('semantic_cache_hits', 'electrical_reuses', 'cross_view_reuses'):
        values[key] = sum(view[key] for view in service['views'].values())
    require(values['early_candidate_fail'] == values['early_zero_actual_candidate_fail']
        + values['early_positive_actual_candidate_fail'], 'Failure cost categories do not close')
    require(sum(values[k] for k in ('physical_pass', 'physical_fail', 'physical_unknown')) == physical['points'],
        'Raw point statuses do not close')
    require(service['logical_points'] - values['electrical_reuses'] == service['physical_points'],
        'Electrical reuse/actual point ledger does not close')
    require(values['cross_view_reuses'] <= values['electrical_reuses'], 'Cross-view reuse is an electrical-reuse subset')
    return values


def first_attainment(points, target):
    return next((deepcopy(p) for p in points if p['fully_certified'] and p['quality_pct'] >= target), None)


def check_audit_bindings(run, audit):
    for key, name in AUDIT_FILES.items():
        require(audit[key] == sha(run / name), 'Audit source changed: ' + name)
    scope = read(run / 'physical_scope.json')
    # Match the independent auditor's JSON default (ensure_ascii=True).
    canonical = json.dumps(scope, sort_keys=True, separators=(',', ':'), allow_nan=False)
    require(audit['physical_scope_canonical_sha256'] == hashlib.sha256(canonical.encode()).hexdigest(),
        'Canonical physical scope changed')


def case_row(spec, result, phases, checked, exit_row, physical):
    require(result['study'] == STUDY and result['schema'] == SCHEMA and result['spec'] == spec,
        'Result identity/schema changed')
    require(checked['status'] == 'PASS' and exit_row['exit_code'] == 0 and checked['cell'] == exit_row['cell'] == spec['cell'],
        'Actual successful exit and independent audit required')
    require(result['new_training'] == result['models_used'] == result['policy_decisions'] == result['model_load_seconds'] == 0,
        'Unapproved learning or model cost')
    require(result['fully_certified'] and result['guard_audit']['guard_established'], 'Complete Beam protection missing')
    service = result['service']
    require(service['status'] == 'COMPLETE_SHARED_EVIDENCE_PENDING_AUDIT' and service['closed']
        and service['physical_cost_complete'] and service['runtime_initializations'] == 1 and service['mode'] == 'shared',
        'Shared physical service incomplete')
    row = {**deepcopy(spec), 'executed': True, 'fully_certified': True, 'guard_established': True,
        'execution_status': 'AUDITED_COMPLETE', 'physical_points': result['actual_physical_points']}
    for key in ('logical_points', 'numerical_entries', 'requests', 'quality_pct', 'worker_elapsed_seconds'):
        row[key] = result[key]
    require(all(checked[k] == row[k] for k in (*COSTS, 'quality_pct', 'worker_elapsed_seconds')),
        'Audit/result metric changed')
    require(0 <= row['physical_points'] <= row['logical_points'] <= 512
        and row['numerical_entries'] == 3 + 4 * row['physical_points'], 'Actual/logical/initialization ledger changed')
    require(service['physical_points'] == physical['points'] == row['physical_points']
        and service['logical_points'] == row['logical_points']
        and service['actual_numerical_entries'] == physical['actual_numerical_entries'] == row['numerical_entries']
        and physical['statuses'] == checked['physical_statuses'], 'Physical ledger changed')
    for phase in ('EARLY', 'GUARD'):
        value = phases[phase]
        require(value['oracle'] == service['views'][phase] and value['oracle']['closed'], 'View ledger changed')
        require(value['logical_points'] == value['oracle']['logical_points'] <= 256
            and value['actual_physical_points'] == value['oracle']['physical_points'], 'Phase ledger changed')
    require(sum(p['actual_physical_points'] for p in phases.values()) == row['physical_points']
        and sum(p['logical_points'] for p in phases.values()) == row['logical_points']
        and sum(p['oracle']['requests'] for p in phases.values()) == row['requests'], 'Phase totals changed')
    early, guard = staged.energy(phases['EARLY']), staged.energy(phases['GUARD'])
    require(guard is not None, 'GUARD has no complete certificate')
    final = staged.energy(result)
    require(final == max(guard, early if early is not None else guard), 'Protected complete selection changed')
    selected = 'EARLY' if early is not None and early > guard else 'GUARD'
    require(result['selected_phase'] == selected and result['quality_pct'] == phases[selected]['quality_pct'],
        'Strict-energy tie handling changed')
    row.update(final_energy_kwh=final, guard_energy_kwh=guard, early_energy_kwh=early,
        early_quality_pct=phases['EARLY']['quality_pct'], guard_quality_pct=phases['GUARD']['quality_pct'],
        gain_over_guard_kwh=final - guard, selected_phase=selected,
        early_physical_points=phases['EARLY']['actual_physical_points'],
        guard_physical_points=phases['GUARD']['actual_physical_points'], early_verify_attempts=phases['EARLY']['verify_attempts'])
    require(row['early_verify_attempts'] <= 16, 'EARLY VERIFY allowance changed')
    process = exit_row['process_wall_seconds']
    require(math.isfinite(process) and process == checked['process_wall_seconds'] and process >= row['worker_elapsed_seconds'],
        'Complete cold process time differs')
    row['process_wall_seconds'] = process
    checkpoint_check(result)
    require(result['checkpoints'] == checked['exact_checkpoints'], 'Audited checkpoints changed')
    counts = mechanism_counts(phases['EARLY'], service, physical)
    require(counts['free_pruned_complete_plans'] == checked['free_pruned_complete_plans']
        and counts['own_incumbent_verify_skips'] == checked['own_incumbent_verify_skips'], 'Audited mechanism count changed')
    require(spec['arm'] == 'P' or counts['free_pruned_complete_plans'] == 0, 'Unapproved free pruning')
    row.update(counts, phase_order=deepcopy(result['phase_order']),
        physical_statuses=deepcopy(physical['statuses']), view_ledgers=deepcopy(service['views']),
        phase_summary=deepcopy(result['phase_summary']), checkpoints=deepcopy(result['checkpoints']),
        phase_stop_reasons={k: p['stop_reason'] for k, p in phases.items()},
        stage_candidate_count=result['stage_candidate_count'], stage_generation_sha256=result['stage_generation_sha256'],
        first_guard_attainment=first_attainment(result['checkpoints'], row['guard_quality_pct']),
        first_own_final_attainment=first_attainment(result['checkpoints'], row['quality_pct']),
        result_sha256=checked['result_sha256'], cell_completion_sha256=checked['cell_completion_sha256'])
    for key in ('preflight_elapsed_seconds', 'runtime_initialization_seconds', 'computation_elapsed_seconds', 'stage_generation_seconds'):
        row[key] = result[key]
    return row


def load_cases(run):
    prepared = read(run / 'prepared.json')
    require(prepared['status'] == 'FROZEN_PREPARED_NO_DSS'
        and prepared['freeze_sha256'] == sha(run / 'prephysical_freeze_manifest.json'), 'Prepared freeze changed')
    manifest(run, name='prephysical_freeze_manifest.json', required=('config.json', 'schedule.json', 'stage_plan.json'), all_files=True)
    # Frozen copies, rather than future working-tree revisions, identify the experiment.
    manifest(run / 'audit', required=('summary.json',), all_files=True)
    audit, controller = read(run / 'audit/summary.json'), read(run / 'controller_completion.json')
    require(audit['status'] == 'PASS' and not audit['errors']
        and audit['audited_cells'] == audit['resolved_slots'] == audit['actual_tasks'] == 24,
        'Only the complete independently audited 24-task batch may be reported')
    require(controller['status'] == 'COMPLETE_CELLS_PENDING_AUDIT'
        and controller['actual_tasks'] == controller['resolved_slots'] == 24
        and not controller['errors'] and not controller['not_run'] and not controller['unknown_cost_cells'],
        'All 24 true complete exits required')
    check_audit_bindings(run, audit)
    cfg, schedule, plan = (read(run / name) for name in ('config.json', 'schedule.json', 'stage_plan.json'))
    require(cfg['staged_study'] == STUDY and cfg['staged_pilot']['models_used'] == 0, 'Wrong rule-only study')
    population(schedule)
    require(not (run / 'phase_gate.json').exists() and plan['execution'] == 'single_stage_all24', 'Conditional population forbidden')
    names = {s['cell'] for s in schedule}
    require(len(audit['results']) == len(controller['exits']) == len(controller['completed']) == 24
        and {r['cell'] for r in audit['results']} == {r['cell'] for r in controller['exits']} == set(controller['completed']) == names,
        'Schedule/audit/true-exit populations differ')
    equalities = audit['context_equalities']
    require(len(equalities) == 8 and {r['context_id'] for r in equalities} == set(range(8))
        and all(r['status'] == 'PASS' and r['guard_trajectories_equal'] for r in equalities), 'Common GUARD/pool checks missing')
    checks, exits = ({r['cell']: r for r in collection} for collection in (audit['results'], controller['exits']))
    bindings = {name: sha(run / name) for name in set(AUDIT_FILES.values()) | {
        'prepared.json', 'audit/summary.json', 'audit/completion_manifest.json'}}
    rows, evidence, pools = [], [], {}
    required = ('spec.json', 'result.json', 'worker_finished.json', 'service_summary.json',
        'physical/oracle_summary.json', 'cost_checkpoints.jsonl', 'EARLY/phase_result.json',
        'GUARD/phase_result.json', 'EARLY/candidate_pool.json', 'EARLY/view_summary.json', 'GUARD/view_summary.json')
    for spec in schedule:
        cell = (run / spec['cell']).resolve()
        require(cell.is_relative_to(run) and cell != run, 'Unsafe cell path')
        check = checks[spec['cell']]
        require(sha(cell / 'completion_manifest.json') == check['cell_completion_sha256'], 'Audited cell seal changed')
        manifest(cell, required=required)
        result = read(cell / 'result.json')
        require(read(cell / 'spec.json') == spec and sha(cell / 'result.json') == check['result_sha256'], 'Audited result changed')
        require(read(cell / 'worker_finished.json')['result_sha256'] == check['result_sha256'], 'Worker result binding changed')
        phases = {k: read(cell / k / 'phase_result.json') for k in ('EARLY', 'GUARD')}
        require(all(sha(cell / k / 'phase_result.json') == check['phase_summary'][k]['phase_result_sha256'] for k in phases),
            'Audited phase result changed')
        require(read(cell / 'service_summary.json') == result['service'], 'Service summary changed')
        require(all(read(cell / k / 'view_summary.json') == result['service']['views'][k] for k in phases), 'View summary changed')
        require([json.loads(line) for line in (cell / 'cost_checkpoints.jsonl').read_text(encoding='utf8').splitlines()]
            == result['checkpoints'], 'Checkpoint journal changed')
        generation = phases['EARLY']['stage_generation']
        expected = next(p['generation'] for p in plan['contexts'] if p['context_id'] == spec['context_id'])
        require(generation == expected == read(cell / 'EARLY/candidate_pool.json')['generation'], 'Common candidate pool changed')
        pools.setdefault(spec['context_id'], generation)
        row = case_row(spec, result, phases, check, exits[spec['cell']], read(cell / 'physical/oracle_summary.json'))
        rows.append(row)
        entries = staged.plan_evidence(spec, phases['EARLY'])
        pruned = {tuple(p['actions']): p for p in phases['EARLY'].get('guard_first', {}).get('free_pruning', [])}
        for item in entries:
            prune = pruned.get(tuple(item['candidate']['actions']))
            if prune:
                require(item['verification_status'] == 'NOT_VERIFIED', 'Pruned plan also claimed as executed')
                item.update(verification_status=prune['status'], public_pruning=deepcopy(prune), physical_feasibility='UNASSESSED')
        evidence.extend(entries)
        for name in (*required, 'completion_manifest.json'):
            bindings[spec['cell'] + '/' + name] = sha(cell / name)
    population(rows)
    for key in COSTS:
        require(sum(r[key] for r in rows) == audit[key] == controller[key], 'Whole-run audited cost changed: ' + key)
    require(dict(sum((Counter(r['physical_statuses']) for r in rows), Counter())) == audit['physical_statuses'],
        'Whole-run physical statuses changed')
    return rows, evidence, pools, bindings, audit, controller


def distribution(values):
    require(len(values) == 8, 'Eight equally weighted contexts required')
    present = [v for v in values if v is not None]
    require(all(isinstance(v, (int, float)) and math.isfinite(v) for v in present), 'Nonfinite report metric')
    return {'mean': statistics.fmean(present) if len(present) == 8 else None,
        'median': statistics.median(present) if len(present) == 8 else None,
        'min': min(present) if present else None, 'max': max(present) if present else None,
        'present': len(present), 'missing': 8 - len(present), 'denominator': 8}


def summarize(rows):
    population(rows)
    arms = {arm: {'tasks': 8, 'contexts': 8,
        'context_balanced_metrics': {k: distribution([r[k] for r in rows if r['arm'] == arm]) for k in METRICS},
        'total_cost': {k: sum(r[k] for r in rows if r['arm'] == arm) for k in COSTS},
        'mechanism_totals': {k: sum(r[k] for r in rows if r['arm'] == arm) for k in MECHANISMS}}
        for arm in ARMS}
    lookup = {(r['context_id'], r['arm']): r for r in rows}
    pairs, comparisons = [], {}
    for control in ('E', 'A_prune'):
        contrast = 'P_minus_' + control
        local = []
        for context in range(8):
            p, c = lookup[context, 'P'], lookup[context, control]
            require(p['guard_energy_kwh'] == c['guard_energy_kwh'], 'Common GUARD energy changed')
            delta = {k: None if p[k] is None or c[k] is None else p[k] - c[k] for k in METRICS}
            relations = {}
            for k in PRIMARY:
                score = p['final_energy_kwh'] - c['final_energy_kwh'] if k == 'quality_pct' else delta[k]
                relations[k] = 'tie' if score == 0 else 'win' if (score > 0 if k == 'quality_pct' else score < 0) else 'loss'
            local.append({'comparison': contrast, 'context_id': context, 'scenario_id': p['scenario_id'], 'event_id': p['event_id'],
                'p_cell': p['cell'], 'control_cell': c['cell'], 'signed_delta': delta, 'p_relations': relations,
                'quality_relation_uses_exact_complete_energy': True})
        pairs.extend(local)
        comparisons[contrast] = {}
        for metric in PRIMARY:
            pm = arms['P']['context_balanced_metrics'][metric]['mean']
            cm = arms[control]['context_balanced_metrics'][metric]['mean']
            differences = [r['signed_delta'][metric] for r in local]
            counts = Counter(r['p_relations'][metric] for r in local)
            comparisons[contrast][metric] = {'signed_delta': distribution(differences),
                'p_mean': pm, 'control_mean': cm,
                'relative_change_pct_ratio_of_means': 100 * (pm / cm - 1) if pm is not None and cm not in (None, 0) else None,
                'p_win_tie_loss': {k: counts[k] for k in ('win', 'tie', 'loss')},
                'worst_context_delta': (min if metric == 'quality_pct' else max)(differences),
                'higher_is_better': metric == 'quality_pct',
                'quality_ties': 'exact complete energy; no epsilon' if metric == 'quality_pct' else None}
    return arms, pairs, comparisons


def markdown(summary, rows):
    fmt = lambda x: '未知' if x is None else f'{x:.6f}'
    lines = ['# 公共搜索 P / E / A_prune 同批开发对照', '',
        '全部 8 个已观察工况 × 3 臂的 24 个冷任务保留，均已完成独立审计与完整 Beam 保护；新增训练 0。', '',
        '| 臂 | 最终质量/% | 实际物理点/任务 | 完整冷进程/s |', '|---|---:|---:|---:|']
    for arm in ARMS:
        m = summary['arms'][arm]['context_balanced_metrics']
        lines.append('| ' + arm + ' | ' + ' | '.join(fmt(m[k]['mean']) for k in PRIMARY) + ' |')
    lines += ['', '质量为原积分正常供电目标；物理点与完整时间包含 GUARD、EARLY、初始化及记录等成本。各工况等权，无历史时间分母。', '',
        '| P 减对照 | 指标 | 均差 | 比值均值变化/% | P 胜/平/负 | 最坏工况差 |', '|---|---|---:|---:|---|---:|']
    for contrast, items in summary['comparisons'].items():
        for metric, item in items.items():
            lines.append(f'| {contrast} | {metric} | {fmt(item["signed_delta"]["mean"])} | '
                f'{fmt(item["relative_change_pct_ratio_of_means"])} | '
                + '/'.join(str(item['p_win_tie_loss'][k]) for k in ('win', 'tie', 'loss'))
                + f' | {fmt(item["worst_context_delta"])} |')
    lines += ['', 'P−E 是先完成 GUARD 后使用其证书的联合顺序/信息/筛除机制；P−A_prune 隔离同 GUARD-first 条件下的免费整批移除。两项不相加。', '',
        '| 工况 | 臂 | 最终质量/% | 实际点 | 完整进程/s | 免费移除 | 自有目标跳过 | 候选FAIL(零新增点) | 真实物理FAIL |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in sorted(rows, key=lambda r: (r['context_id'], ARMS.index(r['arm']))):
        lines.append(f'| c{row["context_id"]:02d} | {row["arm"]} | {fmt(row["quality_pct"])} | '
            f'{row["physical_points"]} | {fmt(row["process_wall_seconds"])} | {row["free_pruned_complete_plans"]} | '
            f'{row["own_incumbent_verify_skips"]} | {row["early_candidate_fail"]} ({row["early_zero_actual_candidate_fail"]}) | {row["physical_fail"]} |')
    lines += ['', '免费移除的物理可行性仍未评估；自有目标跳过和缓存候选 FAIL 都可能消耗 VERIFY 选择。候选 FAIL 次数不是实际物理 FAIL 点数。',
        'semantic_cache_hits 是同逻辑身份请求复用；electrical_reuses 是新增逻辑身份对应的精确电气证据复用；cross_view_reuses 是后者子集，不能三项相加。',
        '阶段物理点归属随先后顺序改变，EARLY 单段点数不等于全任务节省。selection/search_step/journal 分项不是纯模型/纯求解/全部日志时间。',
        '首次到达字段仅表示已完成决策的认证检查点，不是每次物理调用内部返回时间，也不是本轮实施的提前停止。',
        '本轮是 8 个反复观察的开发工况，每臂每工况一次。无未见泛化、稳定延迟保证、外部方法优越性或学习收益结论。',
        f"控制器总墙钟 {fmt(summary['controller_wall_seconds'])} 秒；独立核验 {fmt(summary['audit_elapsed_seconds'])} 秒，另列、不加入逐任务方法时间。"]
    return '\n'.join(lines) + '\n'


def report(run, output=None):
    run = Path(run).resolve()
    output = Path(output).resolve() if output else run / 'derived_report'
    require(output.is_relative_to(ROOT) and output != run and not run.is_relative_to(output) and not output.exists(),
        'Fresh project report directory required')
    rows, evidence, pools, bindings, audit, controller = load_cases(run)
    arms, pairs, comparisons = summarize(rows)
    q = comparisons['P_minus_E']['quality_pct']
    supported = q['p_mean'] >= q['control_mean'] and any(
        comparisons['P_minus_E'][k]['p_mean'] < comparisons['P_minus_E'][k]['control_mean']
        for k in ('physical_points', 'process_wall_seconds'))
    summary = {'status': STATUS, 'study': STUDY, 'run': run.name, 'utc': datetime.now(timezone.utc).isoformat(),
        'scheduled_cells': 24, 'executed_cells': 24, 'not_run_cells': 0, 'observed_contexts': list(range(8)),
        'guard_established_cells': 24, 'all_eight_context_mean_allowed': True, 'arms': arms, 'comparisons': comparisons,
        'total_cost': {k: sum(r[k] for r in rows) for k in COSTS},
        'physical_statuses': dict(sum((Counter(r['physical_statuses']) for r in rows), Counter())),
        'candidate_verification_statuses': dict(Counter(r['verification_status'] for r in evidence)),
        'controller_wall_seconds': controller['wall_seconds'], 'audit_elapsed_seconds': audit['elapsed_seconds'],
        'necessary_development_condition_met': supported,
        'condition_interpretation': 'Necessary descriptive condition only; worst contexts and magnitude still require judgment. No automatic further run.',
        'historical_times_used': False, 'models_used': 0, 'new_report_physical_points': 0, 'new_training': 0,
        'mechanism_count_semantics': {
            'free_pruned_complete_plans': 'Zero-choice complete-objective removal, electrical feasibility unassessed.',
            'own_incumbent_verify_skips': 'Zero-query own-certificate skip consuming one VERIFY.',
            'early_zero_actual_candidate_fail': 'Reliable failed complete-plan attempt with zero newly paid physical points; still consumes VERIFY.',
            'physical_fail': 'Actual paid raw FAIL points, distinct from candidate FAIL attempts.',
            'cross_view_reuses': 'Subset of electrical_reuses; not additive.',
            'semantic_cache_hits': 'Repeated logical identity requests; not a measured whole-task saving by itself.'},
        'source_bindings': bindings,
        'limitations': ['Eight repeatedly observed development contexts; one deterministic cold task per arm/context.',
            'P vs E combines order, available GUARD certificate, exposure and pruning; P vs A_prune isolates free queue removal conditional on GUARD-first.',
            'No learned model, no new training, no unseen evaluation and no external-method superiority claim.',
            'Single-run time differences include instrumentation and machine variability.']}
    implementations = ('scripts/report_clpu_public_mechanism_pilot.py', 'scripts/report_clpu_shared_pilot.py',
        'scripts/report_clpu_staged_pilot.py', 'scripts/report_clpu_integrated_pilot.py')
    summary['report_source_sha256'] = {name: sha(ROOT / name) for name in implementations}
    output.mkdir(parents=True)
    for name, value in (('summary.json', summary), ('all_cells.json', rows), ('all_pairs.json', pairs),
        ('context_summary.json', [{'context_id': c, 'arms': {r['arm']: {k: r[k] for k in METRICS}
            for r in rows if r['context_id'] == c}} for c in range(8)]),
        ('plan_evidence.json', evidence), ('candidate_pools.json', pools)):
        save(output / name, value)
    (output / 'report.md').write_text(markdown(summary, rows), encoding='utf8')
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
    answer = report(args.run, args.output)
    print(json.dumps({k: answer[k] for k in ('status', 'scheduled_cells', 'executed_cells', 'necessary_development_condition_met')}))
