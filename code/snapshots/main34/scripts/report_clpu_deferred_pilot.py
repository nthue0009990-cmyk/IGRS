"""Audit-bound descriptive report for common-engine deferred verification.

Eight physical contexts are the comparison units. The three fixed BC/PPO
endpoints are averaged inside each context before eight-context aggregation.
No solver, actor, learning or fitted statistical model is imported.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import report_clpu_integrated_pilot as stats
import report_clpu_shared_pilot as shared

read, save, sha, require = stats.read, stats.save, stats.sha, stats.require
STUDY = 'CLPU_DEFERRED_VERIFICATION_PILOT_V1'
STATUS = 'AUDITED_DESCRIPTIVE_DEFERRED_VERIFICATION_PILOT'
ARMS = ('interleaved_window', 'deferred_window', 'deferred_bc', 'deferred_ppo')
SEEDS = (17, 41, 73)
ARM_SEEDS = {arm: ((0,) if arm in ARMS[:2] else SEEDS) for arm in ARMS}
METRICS = ('quality_pct', 'physical_points', 'logical_points', 'numerical_entries',
    'requests', 'process_wall_seconds', 'worker_elapsed_seconds',
    'preflight_elapsed_seconds', 'model_load_seconds', 'runtime_initialization_seconds',
    'computation_elapsed_seconds', 'early_quality_pct', 'baseline_quality_pct',
    'early_physical_points', 'guard_physical_points', 'early_worker_elapsed_seconds',
    'early_graph_expansions', 'early_verify_attempts', 'early_total_decisions', 'policy_decisions')
PRIMARY_METRICS = ('quality_pct', 'physical_points', 'process_wall_seconds')
QUALITY_METRICS = {'quality_pct', 'early_quality_pct', 'baseline_quality_pct'}
COMPARISONS = (('ppo_minus_deferred_window', 'deferred_ppo', 'deferred_window'),
               ('ppo_minus_bc', 'deferred_ppo', 'deferred_bc'),
               ('ppo_minus_interleaved_window', 'deferred_ppo', 'interleaved_window'),
               ('deferred_window_minus_interleaved_window', 'deferred_window', 'interleaved_window'))


def population(rows):
    require(isinstance(rows, list) and len(rows) == 64 and len({r['cell'] for r in rows}) == 64,
            'All 64 unique scheduled cells required')
    require({r['context_id'] for r in rows} == set(range(8)), 'All eight contexts required')
    expected = Counter((a, s) for a in ARMS for s in ARM_SEEDS[a])
    for context in range(8):
        chosen = [r for r in rows if r['context_id'] == context]
        require(Counter((r['arm'], r['seed']) for r in chosen) == expected,
                'Missing or duplicated arm/seed within context')
        require(len({(r['scenario_id'], r['event_id']) for r in chosen}) == 1,
                'Paired physical context changed')
    require(len({r['scenario_id'] for r in rows}) == 8, 'Fault population changed')


def aggregate_contexts(rows):
    population(rows)
    output = []
    for context in range(8):
        for arm in ARMS:
            chosen = sorted((r for r in rows if r['context_id'] == context and r['arm'] == arm),
                            key=lambda r: r['seed'])
            metrics, coverage = {}, {}
            for key in METRICS:
                metrics[key], coverage[key] = stats.strict_mean([r[key] for r in chosen], len(chosen))
            output.append({'context_id': context, 'scenario_id': chosen[0]['scenario_id'],
                'event_id': chosen[0]['event_id'], 'arm': arm, 'seeds': list(ARM_SEEDS[arm]),
                'cells': [r['cell'] for r in chosen], 'expected_seed_count': len(chosen),
                'metrics': metrics, 'metric_coverage': coverage,
                'fully_certified_cells': sum(r['fully_certified'] for r in chosen),
                'guard_established_cells': sum(r['guard_established'] for r in chosen)})
    return output


def relation(delta, metric):
    if delta is None:
        return 'missing'
    threshold = stats.TIE_PP if metric in QUALITY_METRICS else 0.
    if abs(delta) <= threshold:
        return 'tie'
    better = delta > 0 if metric in QUALITY_METRICS else delta < 0
    return 'win' if better else 'loss'


def pair(left, right, name, *, context_means=False):
    require(left['context_id'] == right['context_id'] and left['scenario_id'] == right['scenario_id']
            and left['event_id'] == right['event_id'], 'Unpaired physical cases')
    a, b = (left['metrics'], right['metrics']) if context_means else (left, right)
    delta = {k: None if a[k] is None or b[k] is None else a[k] - b[k] for k in METRICS}
    return {'comparison': name, 'context_id': left['context_id'], 'scenario_id': left['scenario_id'],
        'event_id': left['event_id'], 'left_arm': left['arm'], 'right_arm': right['arm'],
        'left_cells': left['cells'] if context_means else [left['cell']],
        'right_cells': right['cells'] if context_means else [right['cell']],
        'left_seed': None if context_means else left['seed'],
        'right_seed': None if context_means else right['seed'],
        'context_means': context_means, 'signed_delta': delta,
        'metric_relations': {k: relation(delta[k], k) for k in PRIMARY_METRICS}}


def comparisons(rows, contexts):
    lookup = {(r['context_id'], r['arm'], r['seed']): r for r in rows}
    context_lookup = {(r['context_id'], r['arm']): r for r in contexts}
    all_pairs, context_pairs, summary = [], [], {}
    for name, left_arm, right_arm in COMPARISONS:
        group = []
        for context in range(8):
            p = pair(context_lookup[context, left_arm], context_lookup[context, right_arm],
                     name, context_means=True)
            context_pairs.append(p)
            group.append(p)
            for seed in ARM_SEEDS[left_arm]:
                right_seed = seed if right_arm in ARMS[2:] else 0
                all_pairs.append(pair(lookup[context, left_arm, seed],
                    lookup[context, right_arm, right_seed], name))
        metrics = {}
        for key in METRICS:
            values = [p['signed_delta'][key] for p in group]
            distribution = stats.distribution(values, 8)
            valid = [p for p in group if p['signed_delta'][key] is not None]
            worst = None if len(valid) != 8 else (min if key in QUALITY_METRICS else max)(
                p['signed_delta'][key] for p in valid)
            left, _ = stats.strict_mean([context_lookup[c, left_arm]['metrics'][key] for c in range(8)], 8)
            right, _ = stats.strict_mean([context_lookup[c, right_arm]['metrics'][key] for c in range(8)], 8)
            metrics[key] = {'signed_delta': distribution,
                'relative_change_pct': None if left is None or right in (None, 0) else 100 * (left / right - 1),
                'win_tie_loss_missing': {label: sum(relation(v, key) == label for v in values)
                                       for label in ('win', 'tie', 'loss', 'missing')},
                'worst_signed_delta': worst,
                'worst_contexts': [] if worst is None else [p['context_id'] for p in valid
                    if p['signed_delta'][key] == worst]}
        interpretation = ('Scheduling-order package' if name == COMPARISONS[-1][0]
            else 'Combined schedule and fixed-ranking difference; not ranking-only'
            if right_arm == 'interleaved_window'
            else 'Fixed candidate-ranking transfer on the common deferred schedule')
        summary[name] = {'left_arm': left_arm, 'right_arm': right_arm, 'contexts': 8, 'metrics': metrics,
            'interpretation': interpretation,
            'independent_units': 'Eight observed contexts; three seeds are repeated model endpoints.'}
    return all_pairs, context_pairs, summary


def make_curves(rows, *, early):
    population(rows)
    selected = [[p for p in r['checkpoints'] if not early or p['phase'] == 'EARLY'] for r in rows]
    require(all(selected), 'Missing measured curve checkpoints')
    physical_axis = list(range(257 if early else 513))
    # Exact observed boundaries, plus zero: no integer-time rounding of events.
    time_axis = sorted({0.} | {p['worker_elapsed_seconds'] for points in selected for p in points})
    cells = []
    for row, points in zip(rows, selected):
        stop_time = points[-1]['worker_elapsed_seconds'] if early else row['worker_elapsed_seconds']
        stop_physical = points[-1]['physical_points'] if early else row['physical_points']
        cells.append({**{k: row[k] for k in ('cell', 'context_id', 'scenario_id', 'event_id', 'arm', 'seed')},
            'stop_physical_points': stop_physical, 'stop_worker_elapsed_seconds': stop_time,
            'physical': stats.step_curve(points, physical_axis, 'physical_points', stop_physical),
            'worker_time': stats.step_curve(points, time_axis, 'worker_elapsed_seconds', stop_time),
            'exact_checkpoints': deepcopy(points)})
    context_rows = []
    for context in range(8):
        for arm in ARMS:
            chosen = [r for r in cells if r['context_id'] == context and r['arm'] == arm]
            require(sorted(r['seed'] for r in chosen) == list(ARM_SEEDS[arm]), 'Curve seed omitted')
            context_rows.append({'context_id': context, 'arm': arm, 'seeds': list(ARM_SEEDS[arm]),
                'scenario_id': chosen[0]['scenario_id'], 'event_id': chosen[0]['event_id'],
                **{axis: stats.average_curves(chosen, axis) for axis in ('physical', 'worker_time')}})
    groups = [{'arm': arm, **{axis: stats.average_curves([r for r in context_rows if r['arm'] == arm],
        axis, contexts=True) for axis in ('physical', 'worker_time')}} for arm in ARMS]
    return {'schema': 'clpu-deferred-checkpoint-curves-v1', 'early_phase_only': early,
        'axes': {'physical_points': physical_axis, 'worker_elapsed_seconds': time_axis},
        'cells': cells, 'context_curves': context_rows, 'group_curves': groups,
        'semantics': {'quality': 'Only a certificate available at the completed checkpoint; null before certification.',
            'time': 'Exact worker-entry completion boundaries including preflight/model loading; not process exit time.',
            'physical': 'All completed free actions at the same actual point count are included; not a time guarantee.',
            'guard': 'Only completed GUARD establishes the original Beam floor; EARLY alone does not.',
            'carry': 'Only an obtained certificate is held after stopping; explicitly flagged, not a new solve.',
            'mean': 'All seeds within each context, then all eight contexts; any missing member keeps mean null.',
            'smoothing': 'None; right-continuous post-step curves; no outcome-dependent truncation.'}}


def case_row(spec, result, audited, exit_record, physical):
    require(result['spec'] == spec and result['study'] == STUDY
            and result['schema'] == 'clpu-deferred-protected-result-v1', 'Result identity/schema changed')
    require(result['new_training'] == 0 and spec['mode'] == 'shared' and spec['protected'] is True,
            'Unexpected training or unprotected/nonshared arm')
    learned = spec['arm'] in ARMS[2:]
    expected_method = 'bc_raw' if spec['arm'] == 'deferred_bc' else 'ppo_raw' if learned else 'beam'
    require(spec['method'] == expected_method and spec['macro_schedule'] ==
            ('interleaved' if spec['arm'] == 'interleaved_window' else 'deferred'), 'Ranker/schedule changed')
    require(result['checkpoint_sha256'] == spec['checkpoint_sha256']
            and (spec['checkpoint_sha256'] is not None) is learned, 'Fixed model binding changed')
    require(audited['status'] == 'PASS' and audited['cell'] == spec['cell'], 'Unaudited cell')
    require(exit_record['exit_code'] == 0 and exit_record['cell'] == spec['cell'], 'No real successful exit')
    service = result['service']
    require(service['status'] == 'COMPLETE_SHARED_EVIDENCE_PENDING_AUDIT'
            and service['closed'] is True and service['physical_cost_complete'] is True
            and service['runtime_initializations'] == 1 and service['mode'] == 'shared',
            'Common physical service incomplete or changed')
    actual, logical, numerical = (stats.integer(result[k], k) for k in
        ('actual_physical_points', 'logical_points', 'numerical_entries'))
    require(actual <= logical <= 512 and service['physical_points'] == actual
            and service['logical_points'] == logical
            and numerical == service['actual_numerical_entries'] == 3 + 4 * actual,
            'Dual ledgers or single initialization do not close')
    require(physical['points'] == actual and physical['actual_numerical_entries'] == numerical
            and sum(physical['statuses'].values()) == actual, 'Original physical costs/failures differ')
    q, guard = result['quality_pct'], result['guard_audit']['guard_established']
    require(type(guard) is bool and result['fully_certified'] is (q is not None), 'Certificate availability changed')
    if q is not None:
        require(stats.finite(q, 'quality') <= 100 + stats.TIE_PP and guard, 'Final quality lacks complete guard')
        require(result['guard_audit']['final_energy_kwh'] >= result['guard_audit']['baseline_energy_kwh'],
                'Exact Beam energy floor regressed')
    for k, value in {'quality_pct': q, 'guard_established': guard, 'physical_points': actual,
                     'logical_points': logical, 'numerical_entries': numerical,
                     'guard_quality_pct': result['baseline_quality_pct'],
                     'requests': result['requests'], 'worker_elapsed_seconds': result['worker_elapsed_seconds'],
                     'model_load_seconds': result['model_load_seconds'],
                     'physical_statuses': physical['statuses']}.items():
        require(audited[k] == value, 'Audited measurement changed: ' + k)
    process = stats.finite(exit_record['process_wall_seconds'], 'process wall')
    require(process == audited['process_wall_seconds'] and process >= result['worker_elapsed_seconds'],
            'Full measured process time changed')
    for key in ('worker_elapsed_seconds', 'preflight_elapsed_seconds', 'model_load_seconds',
                'runtime_initialization_seconds', 'computation_elapsed_seconds'):
        stats.finite(result[key], key)
    require(result['worker_elapsed_seconds'] + 1e-6 >=
            result['preflight_elapsed_seconds'] + result['computation_elapsed_seconds']
            and result['preflight_elapsed_seconds'] >= result['model_load_seconds']
            and result['computation_elapsed_seconds'] >= result['runtime_initialization_seconds'],
            'Timing components omit preparation or initialization')
    require(learned or result['model_load_seconds'] == 0, 'Rule arm paid an unexpected actor load')
    phases = result['phase_summary']
    require(set(phases) == {'EARLY', 'GUARD'} and set(service['views']) == set(phases), 'Missing complete route')
    require(audited['exact_checkpoints'] == result['checkpoints'], 'Audited checkpoints changed')
    for phase in ('EARLY', 'GUARD'):
        summary, view = phases[phase], service['views'][phase]
        require(all(audited['phase_summary'][phase][key] == value for key, value in summary.items()),
                'Audited phase measurement changed: ' + phase)
        require(summary['logical_points'] == view['logical_points'] <= 256
                and summary['actual_physical_points'] == view['physical_points']
                and view['physical_points'] <= view['logical_points']
                and view['actual_numerical_entries_attributed'] == 4 * view['physical_points']
                and view['initialization_numerical_entries_attributed'] == 0,
                'Per-route physical/logical/initialization costs changed')
    require(sum(p['physical_points'] for p in service['views'].values()) == actual
            and sum(p['logical_points'] for p in service['views'].values()) == logical,
            'Route costs do not add to service totals')
    early, baseline = phases['EARLY'], phases['GUARD']
    graph = stats.integer(early['graph_expansions'], 'EARLY graph expansions')
    verify = stats.integer(early['verify_attempts'], 'EARLY verification choices')
    ops = stats.integer(early['search_decisions'], 'EARLY atomic operations')
    policy = stats.integer(result['policy_decisions'], 'fixed ranker decisions')
    require(graph <= 16 and verify <= 16 and ops <= 33 and ops - graph - verify in (0, 1),
            'Fixed 16+16+1 atomic-operation budget changed')
    require(policy == early['policy_decisions'] == (verify if learned else 0)
            and baseline['policy_decisions'] == 0 and baseline['verify_attempts'] is None
            and baseline['search_decisions'] == baseline['graph_expansions'] <= 256,
            'Ranking scope or original combined-step guard changed')
    require(result['search_decisions'] == ops + baseline['search_decisions'], 'Total operation count changed')
    require(result['early_quality_pct'] == early['quality_pct']
            and result['baseline_quality_pct'] == baseline['quality_pct'], 'Phase quality changed')
    shared.checkpoints(result['checkpoints'], result)
    # These counters represent atomic EARLY operations (including YIELD), not
    # graph expansions or a claimed equivalent to old combined search choices.
    for point in result['checkpoints']:
        require(type(point['guard_established']) is bool
            and (not point['guard_established'] or point['phase'] == 'GUARD' and point['kind'] == 'CLOSE'),
            'Beam floor claimed before complete guard close')
    early_close = [p for p in result['checkpoints'] if p['phase'] == 'EARLY' and p['kind'] == 'CLOSE']
    require(len(early_close) == 1 and early_close[0]['quality_pct'] == early['quality_pct'], 'EARLY close missing')
    return {**spec, 'quality_pct': q, 'fully_certified': result['fully_certified'], 'guard_established': guard,
        'physical_points': actual, 'logical_points': logical, 'numerical_entries': numerical,
        'requests': result['requests'], 'process_wall_seconds': process,
        **{key: result[key] for key in ('worker_elapsed_seconds', 'preflight_elapsed_seconds',
            'model_load_seconds', 'runtime_initialization_seconds', 'computation_elapsed_seconds')},
        'early_quality_pct': early['quality_pct'], 'baseline_quality_pct': baseline['quality_pct'],
        'early_physical_points': early['actual_physical_points'],
        'guard_physical_points': baseline['actual_physical_points'],
        'early_worker_elapsed_seconds': early_close[0]['worker_elapsed_seconds'],
        'early_graph_expansions': graph, 'early_verify_attempts': verify, 'early_total_decisions': ops,
        'early_yield_operations': ops - graph - verify, 'policy_decisions': policy,
        'phase_summary': deepcopy(phases), 'view_ledgers': deepcopy(service['views']),
        'physical_statuses': deepcopy(physical['statuses']),
        'cross_view_reuses': sum(v['cross_view_reuses'] for v in service['views'].values()),
        'checkpoints': deepcopy(result['checkpoints']),
        'counter_semantics': 'EARLY search/phase choices are atomic operations including YIELD; GUARD choices are old combined steps.'}


def load_cases(run):
    run = Path(run).resolve()
    prepared = read(run / 'prepared.json')
    require(prepared['status'] == 'FROZEN_PREPARED_NO_DSS'
            and sha(run / 'prephysical_freeze_manifest.json') == prepared['freeze_sha256'], 'Prepared freeze changed')
    shared.manifest(run, name='prephysical_freeze_manifest.json',
        required=('config.json', 'schedule.json', 'physical_scope.json'), all_files=True)
    for relative in prepared['source_files']:
        require(sha(ROOT / relative) == sha(run / 'source' / relative), 'Executing frozen source changed: ' + relative)
    cfg, schedule = read(run / 'config.json'), read(run / 'schedule.json')
    require(cfg['deferred_study'] == STUDY, 'Wrong frozen study')
    population(schedule)
    shared.manifest(run / 'audit', required=('summary.json',), all_files=True)
    audit = read(run / 'audit/summary.json')
    require(audit['status'] == 'PASS' and audit['audited_cells'] == 64 and audit['errors'] == [],
            'Complete 64-cell independent audit required')
    bindings = {}
    for field, relative in (('freeze_sha256', 'prephysical_freeze_manifest.json'),
        ('config_sha256', 'config.json'), ('schedule_sha256', 'schedule.json'),
        ('controller_completion_sha256', 'controller_completion.json')):
        require(audit[field] == sha(run / relative), 'Audit binding changed: ' + relative)
    for relative in ('prephysical_freeze_manifest.json', 'prepared.json', 'config.json', 'schedule.json',
        'physical_scope.json', 'source_binding.json', 'controller_completion.json',
        'audit/summary.json', 'audit/completion_manifest.json'):
        bindings[relative] = sha(run / relative)
    controller = read(run / 'controller_completion.json')
    require(controller['status'] == 'COMPLETE_CELLS_PENDING_AUDIT' and controller['scheduled'] == 64
        and not controller['errors'] and not controller['unknown_cost_cells'], 'Incomplete or failed evaluation')
    names = {r['cell'] for r in schedule}
    require(len(controller['completed']) == 64 and set(controller['completed']) == names,
            'Incomplete controller population')
    exits, audited = controller['exits'], audit['results']
    require(len(exits) == len(audited) == 64 and {r['cell'] for r in exits} == {r['cell'] for r in audited} == names,
            'Missing/extra/repeated actual exit or cell audit')
    exits, audited = {r['cell']: r for r in exits}, {r['cell']: r for r in audited}
    rows = []
    required = ('spec.json', 'result.json', 'cost_checkpoints.jsonl', 'service_summary.json', 'physical/oracle_summary.json')
    for spec in schedule:
        cell = (run / spec['cell']).resolve()
        require(cell != run and cell.is_relative_to(run), 'Cell escaped run')
        check = audited[spec['cell']]
        require(sha(cell / 'completion_manifest.json') == check['cell_completion_sha256'], 'Audited cell seal changed')
        shared.manifest(cell, required=required)
        require(read(cell / 'spec.json') == spec and sha(cell / 'result.json') == check['result_sha256'],
                'Audited cell identity/result changed')
        result = read(cell / 'result.json')
        require(read(cell / 'service_summary.json') == result['service'], 'Service summary mismatch')
        import json
        points = [json.loads(line) for line in (cell / 'cost_checkpoints.jsonl').read_text(encoding='utf-8').splitlines()]
        require(points == result['checkpoints'], 'Checkpoint journal differs from result')
        rows.append(case_row(spec, result, check, exits[spec['cell']], read(cell / 'physical/oracle_summary.json')))
        for relative in ('completion_manifest.json', 'result.json'):
            bindings[spec['cell'] + '/' + relative] = sha(cell / relative)
    for key in ('physical_points', 'logical_points', 'numerical_entries'):
        require(sum(r[key] for r in rows) == audit[key] == controller[key], 'Audited total cost changed: ' + key)
    return rows, bindings, controller


def make_summary(rows, contexts, comparison, bindings, controller, run_name):
    arms = {}
    for arm in ARMS:
        cells = [r for r in rows if r['arm'] == arm]
        grouped = [r for r in contexts if r['arm'] == arm]
        arms[arm] = {'cells': len(cells), 'contexts': 8, 'seeds': list(ARM_SEEDS[arm]),
            'context_balanced_metrics': {k: stats.distribution([r['metrics'][k] for r in grouped], 8) for k in METRICS},
            'fully_certified_cells': sum(r['fully_certified'] for r in cells),
            'guard_established_cells': sum(r['guard_established'] for r in cells),
            'all_cell_timing': {k: stats.distribution([r[k] for r in cells]) for k in
                ('process_wall_seconds', 'worker_elapsed_seconds', 'model_load_seconds')},
            'resource_totals_not_method_comparisons': {k: sum(r[k] for r in cells) for k in
                ('physical_points', 'logical_points', 'numerical_entries', 'process_wall_seconds')}}
    return {'status': STATUS, 'study': STUDY, 'run': run_name, 'utc': datetime.now(timezone.utc).isoformat(),
        'contexts': 8, 'cells': 64, 'audited_cells': 64, 'arms': arms, 'comparisons': comparison,
        'total_cost': {k: sum(r[k] for r in rows) for k in ('physical_points', 'logical_points', 'numerical_entries')},
        'physical_statuses': dict(sum((Counter(r['physical_statuses']) for r in rows), Counter())),
        'fully_certified_cells': sum(r['fully_certified'] for r in rows),
        'guard_established_cells': sum(r['guard_established'] for r in rows),
        'uncertified_cells_retained': [r['cell'] for r in rows if not r['fully_certified']],
        'evaluation_controller_wall_seconds': stats.finite(controller['wall_seconds'], 'evaluation wall'),
        'new_report_physical_points': 0, 'new_training': 0, 'source_bindings': bindings,
        'quality_relation_tolerance_pp': stats.TIE_PP, 'cost_relation_tolerance': 0,
        'interpretation': [
            'Eight observed training contexts, not a confirmatory generalization evaluation.',
            'All four arms use the same optimized pending window, exact physical sharing and complete original Beam guard.',
            'Interleaved versus deferred rule tests scheduling order; PPO versus interleaved changes both order and ranking.',
            'PPO versus deferred window or same-source BC uses the same deferred macro and compares fixed ranker transfer.',
            'Old model checkpoints only rank candidate VERIFY choices; no learned EXPAND/VERIFY/YIELD policy or new training.',
            'BC and PPO have common historical source/initialization, not equal total historical training interactions.',
            'EARLY operation limits are 16 graph, 16 verification and at most one handoff; bootstrap is paid separately.',
            'Verification includes free dominated skips; reliable-failure repair generation is not an extra macro EXPAND.',
            'GUARD actual points are newly paid points attributed under sharing, not the cost of an isolated Beam run.',
            'Seed means precede eight-context means; no missing certificates or unfavorable seeds are dropped.',
            'Complete process durations include loading, both routes and journals; never splice prior-run time baselines.',
            'Phase elapsed clocks stop before phase-result writing; the complete process also includes final journals, sealing and exit.',
            'Selection includes control preparation and actor inference; search_step is not pure physics and journal is not all recording.',
            'Compact observation construction/hashing and remaining work are retained in full clocks, not subtracted as deployment speed.',
            'No success gate or preferred result is prefilled; quality, actual cost and time must be read together.']}


def markdown(summary, contexts):
    fmt = lambda value: '缺失' if value is None else f'{value:.6f}'
    lines = ['# 同一优化窗口下的延期验证对照', '',
        '64项评价与独立审计完成，全部8工况、BC/PPO各3个固定种子保留。所有方法使用同一共享证据服务及完整原Beam保底。', '',
        'interleaved_window与deferred_window只比较操作调度顺序；PPO相对deferred_window/BC比较同一延期调度下的固定候选排序。旧模型没有学习三类操作，本轮无新训练。', '',
        '| 方法 | 最终质量/% | 实际点/任务 | 完整进程/秒 | EARLY质量/% | EARLY操作数 |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        metrics = summary['arms'][arm]['context_balanced_metrics']
        keys = ('quality_pct', 'physical_points', 'process_wall_seconds', 'early_quality_pct', 'early_total_decisions')
        lines.append('| ' + arm + ' | ' + ' | '.join(fmt(metrics[k]['mean']) for k in keys) + ' |')
    lines += ['', '均值先在每工况内平均全部模型种子，再对8工况等权。完整进程从创建至真实退出；EARLY操作数不是耗时，也不等于旧合并扩展步。', '',
        '| 比较（左减右） | 指标 | 平均差 | 相对变化/% | 赢/同/输/缺失（8工况） | 最差差值 |',
        '|---|---|---:|---:|---|---:|']
    for name, comparison in summary['comparisons'].items():
        for key in PRIMARY_METRICS:
            row = comparison['metrics'][key]
            counts = row['win_tie_loss_missing']
            lines.append(f'| {name} | {key} | {fmt(row["signed_delta"]["mean"])} | '
                f'{fmt(row["relative_change_pct"])} | ' + '/'.join(str(counts[k]) for k in ('win', 'tie', 'loss', 'missing'))
                + f' | {fmt(row["worst_signed_delta"])} |')
    lines += ['', 'PPO相对interleaved_window同时改变调度与排序，不归为纯排序效果。质量差单位为百分点，成本/时间减少为赢；质量关系容差1e-9个百分点只用于描述分类，不降低Beam保底。', '',
        '| 工况 | 方法 | 最终质量/% | 实际点 | 完整进程/秒 | EARLY质量/% |',
        '|---|---|---:|---:|---:|---:|']
    for row in contexts:
        lines.append(f'| c{row["context_id"]:02d} | {row["arm"]} | ' + ' | '.join(fmt(row['metrics'][k]) for k in
            ('quality_pct', 'physical_points', 'process_wall_seconds', 'early_quality_pct')) + ' |')
    lines += ['', '完整费用包含初始化、固定模型加载、全部图处理/验证/记录及GUARD；共享命中仍保留可靠失败和语义引用。'
        '只在完整GUARD完成后建立原Beam终值保底。两种曲线分别展示完整任务和EARLY，检查点时间不是进程退出时间；'
        '停止后仅保持已获得证书，不构造额外求解或事后早停。phase_elapsed在阶段结果写盘前截止；完整进程另包含关闭、封存和退出。'
        'selection包含控制准备及模型推理，search_step不等于纯物理调用，journal也不是全部日志开销；不减去这些分项冒充部署速度。', '',
        f'全64总成本（不是方法均值比较）：{summary["total_cost"]}；物理状态：{summary["physical_statuses"]}。'
        f'缺认证任务完整保留：{summary["uncertified_cells_retained"]}。新增报告求解与训练均0。', '',
        '完整原GUARD查询集G保留时，精确共享成本|E∪G|≥|G|。相同终值下无法仅靠EARLY调度保证少于直接执行G的物理点；'
        '如实报告本轮质量、实际点和完整耗时，不引用旧实现时间来计算超越。']
    return '\n'.join(lines) + '\n'


def report(run, output=None):
    run = Path(run).resolve()
    output = Path(output).resolve() if output is not None else run / 'derived_report'
    require(not output.exists() and output != run and not run.is_relative_to(output), 'New report folder required')
    rows, bindings, controller = load_cases(run)
    contexts = aggregate_contexts(rows)
    all_pairs, context_pairs, comparison = comparisons(rows, contexts)
    summary = make_summary(rows, contexts, comparison, bindings, controller, run.name)
    full, early = make_curves(rows, early=False), make_curves(rows, early=True)
    output.mkdir(parents=True)
    for name, value in (('summary.json', summary), ('all_cells.json', rows), ('all_pairs.json', all_pairs),
        ('context_pairs.json', context_pairs), ('context_summary.json', contexts),
        ('curve_data.json', full), ('early_curve_data.json', early)):
        save(output / name, value)
    (output / 'report.md').write_text(markdown(summary, contexts), encoding='utf-8')
    save(output / 'completion_manifest.json', {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', '--out', type=Path)
    args = parser.parse_args()
    result = report(args.run, args.output)
    print({'status': result['status'], 'cells': result['cells'], 'contexts': result['contexts']})
