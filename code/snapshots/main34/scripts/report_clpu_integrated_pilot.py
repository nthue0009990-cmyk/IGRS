"""Audit-bound, seed-balanced descriptive report and causal step curves.

Standard library only: no physical solver, learning code, or training is used.
Missing certificates remain null and every frozen cell remains in the report.
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
import statistics


STUDY = 'CLPU_INTEGRATED_PROTECTED_PILOT_V1'
STATUS = 'AUDITED_DESCRIPTIVE_INTEGRATED_PILOT'
SEEDS = (17, 41, 73)
ARMS = ('window_reference', 'protected_rule', 'protected_bc', 'protected_ppo')
ARM_SEEDS = {arm: ((0,) if arm in ARMS[:2] else SEEDS) for arm in ARMS}
METRICS = ('quality_pct', 'physical_points', 'logical_points', 'numerical_entries', 'requests',
           'process_wall_seconds', 'worker_elapsed_seconds', 'computation_elapsed_seconds',
           'preflight_elapsed_seconds', 'model_load_seconds', 'runtime_initialization_seconds',
           'policy_decisions')
PAIR_METRICS = ('physical_points', 'logical_points', 'numerical_entries', 'requests',
                'process_wall_seconds', 'worker_elapsed_seconds', 'computation_elapsed_seconds')
TIE_PP = 1e-9


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def integer(value, name):
    require(type(value) is int and value >= 0, 'Invalid nonnegative integer: ' + name)
    return value


def finite(value, name, *, nonnegative=True):
    require(type(value) in (int, float) and math.isfinite(value)
            and (not nonnegative or value >= 0), 'Invalid finite value: ' + name)
    return float(value)


def verify_manifest(directory, name='completion_manifest.json', required=()):
    directory = Path(directory).resolve()
    manifest = read(directory / name)
    require(isinstance(manifest, dict) and manifest, 'Missing nonempty flat manifest')
    normalized, targets = {}, set()
    for raw, digest in manifest.items():
        require(isinstance(raw, str) and isinstance(digest, str) and len(digest) == 64
                and all(x in '0123456789abcdef' for x in digest), 'Invalid flat manifest entry')
        key = raw.replace('\\', '/')
        target = (directory / key).resolve()
        require(key not in normalized and target not in targets and target != directory
                and target.is_relative_to(directory), 'Manifest path alias or escape')
        require(sha(target) == digest, 'Manifest digest changed: ' + key)
        normalized[key] = digest
        targets.add(target)
    require(set(required) <= set(normalized), 'Required sealed evidence missing')
    return normalized


def strict_mean(values, expected):
    require(len(values) == expected, 'Population size changed before averaging')
    present = [finite(x, 'mean member', nonnegative=False) for x in values if x is not None]
    return (statistics.mean(present) if expected > 0 and len(present) == expected else None), len(present)


def distribution(values, expected=None):
    expected = len(values) if expected is None else expected
    mean, coverage = strict_mean(values, expected)
    ordered = sorted(finite(x, 'distribution member', nonnegative=False) for x in values if x is not None)
    complete = coverage == expected and expected > 0
    return {'expected_count': expected, 'coverage_count': coverage, 'missing_count': expected - coverage,
            'mean': mean if complete else None, 'median': statistics.median(ordered) if complete else None,
            'minimum': ordered[0] if complete else None, 'maximum': ordered[-1] if complete else None,
            'p95_nearest_rank': ordered[math.ceil(.95 * expected) - 1] if complete else None}


def validate_population(schedule):
    require(isinstance(schedule, list) and len(schedule) == 64, 'All 64 scheduled cells required')
    require(len({r['cell'] for r in schedule}) == 64, 'Duplicate scheduled cell')
    require({r['context_id'] for r in schedule} == set(range(8)), 'All eight contexts required')
    for context in range(8):
        selected = [r for r in schedule if r['context_id'] == context]
        require(len({(r['scenario_id'], r['event_id']) for r in selected}) == 1, 'Context identity differs')
        require(Counter((r['arm'], r['seed']) for r in selected)
                == Counter((arm, seed) for arm in ARMS for seed in ARM_SEEDS[arm]), 'Arm/seed population changed')
    for spec in schedule:
        require(spec['cache_variant'] == 'electrical_equivalent' and spec['point_budget'] == 256
                and spec['logical_point_budget'] == spec['physical_point_budget'] == 256
                and spec['protected'] is (spec['arm'] != 'window_reference'), 'Shared cache/budget/guard differs')
        require(spec['continuation_expansions'] == (0 if spec['arm'] == 'window_reference' else 256),
                'Frozen expansion allowance changed')


def validate_checkpoints(points, oracle, result):
    require(isinstance(points, list) and len(points) == result['search_decisions'] + 1,
            'Every completed decision boundary and initial boundary required')
    previous = None
    for index, point in enumerate(points):
        require(integer(point['search_choices'], 'search choices') == index, 'Checkpoint choice sequence changed')
        logical, physical = (integer(point[k], k) for k in ('logical_points', 'physical_points'))
        require(physical <= logical <= 256, 'Checkpoint logical/physical budgets differ')
        require(integer(point['numerical_entries'], 'checkpoint entries') == 3 + 4 * physical,
                'Checkpoint omits actual initialization or solve cost')
        integer(point['requests'], 'checkpoint requests')
        computation = finite(point['elapsed_seconds'], 'checkpoint computation time')
        worker = finite(point['worker_elapsed_seconds'], 'checkpoint worker time')
        require(worker >= result['preflight_elapsed_seconds'] + computation,
                'Checkpoint worker clock omits preflight or computation')
        require(type(point['fully_certified']) is bool, 'Certificate availability must be explicit')
        if point['fully_certified']:
            require(finite(point['quality_pct'], 'certified quality') <= 100 + TIE_PP, 'Quality outside range')
        else:
            require(point['quality_pct'] is None, 'Uncertified quality must remain null')
        if previous is not None:
            require(all(point[k] >= previous[k] for k in ('logical_points', 'physical_points',
                'numerical_entries', 'requests', 'elapsed_seconds', 'worker_elapsed_seconds')),
                'Cumulative checkpoint cost or clock decreased')
            if previous['fully_certified']:
                require(point['fully_certified'] and point['quality_pct'] >= previous['quality_pct'],
                        'Certified incumbent regressed')
        previous = point
    require(all(points[-1][a] == oracle[b] for a, b in (
        ('logical_points', 'logical_points'), ('physical_points', 'physical_points'),
        ('numerical_entries', 'actual_numerical_entries'), ('requests', 'requests'))), 'Final checkpoint cost differs')
    require(points[-1]['worker_elapsed_seconds'] <= result['elapsed_seconds']
            and points[-1]['elapsed_seconds'] <= result['computation_elapsed_seconds'], 'Final timing omits certificate boundary')


def case_row(spec, result, audited, process_wall):
    require(result['spec'] == spec, 'Result differs from scheduled specification')
    require(audited['status'] == 'PASS' and audited['quality_checkpoints_verified'] is True, 'Cell was not fully audited')
    require(all(audited[k] == spec[k] for k in ('cell', 'context_id', 'arm', 'seed', 'method')), 'Audit identifies another cell')
    require(all(result[k] == 0 for k in ('new_training', 'training_interactions', 'optimizer_updates'))
            and all(result[k] is False for k in ('formal_holdout_numerical_access', 'tune_numerical_access', 'clpu_holdout_access')),
            'Frozen scope violated')
    oracle = result['oracle']
    require(oracle['variant'] == 'electrical_equivalent' and oracle['closed'] is True
            and oracle['status'] == 'COMPLETE_DUAL_LEDGER_PENDING_AUDIT', 'Dual ledger incomplete')
    for name in ('physical_points', 'logical_points', 'requests', 'semantic_cache_hits', 'electrical_reuses',
                 'actual_numerical_entries', 'initialization_numerical_entries'):
        integer(oracle[name], name)
    require(oracle['physical_points'] <= oracle['logical_points'] <= 256
            and oracle['logical_points'] == oracle['physical_points'] + oracle['electrical_reuses']
            and oracle['initialization_numerical_entries'] == 3
            and oracle['actual_numerical_entries'] == 3 + 4 * oracle['physical_points'], 'Actual cost accounting differs')
    timing = {key: finite(result[source], key) for key, source in (
        ('worker_elapsed_seconds', 'elapsed_seconds'), ('computation_elapsed_seconds', 'computation_elapsed_seconds'),
        ('preflight_elapsed_seconds', 'preflight_elapsed_seconds'), ('model_load_seconds', 'model_load_seconds'),
        ('runtime_initialization_seconds', 'runtime_initialization_seconds'))}
    timing['process_wall_seconds'] = finite(process_wall, 'complete process wall')
    require(timing['process_wall_seconds'] >= timing['worker_elapsed_seconds']
            >= timing['preflight_elapsed_seconds'] + timing['computation_elapsed_seconds']
            and timing['preflight_elapsed_seconds'] >= timing['model_load_seconds']
            and timing['computation_elapsed_seconds'] >= timing['runtime_initialization_seconds'], 'Cold timing cost omitted')
    decisions = integer(result['search_decisions'], 'search decisions')
    require(decisions <= (256 if spec['arm'] == 'window_reference' else 768), 'Decision bound exceeded')
    validate_checkpoints(result['checkpoints'], oracle, result)
    quality = result['checkpoints'][-1]['quality_pct']
    best = result['best_plan']
    if quality is None:
        require(best is None or best.get('certified_complete') is not True, 'Missing quality contradicts certificate')
    else:
        require(isinstance(best, dict) and best.get('certified_complete') is True
                and best.get('status') == 'COMPLETE' and best.get('validation', {}).get('status') == 'COMPLETE',
                'Final quality lacks a complete certificate')
    physical = {k: oracle[k] for k in ('physical_points', 'logical_points', 'requests', 'semantic_cache_hits', 'electrical_reuses')}
    physical['numerical_entries'] = oracle['actual_numerical_entries']
    require(all(audited[k] == value for k, value in physical.items()) and audited['quality_pct'] == quality,
            'Independent audit and oracle/certificate costs differ')
    require(all(audited[k] == timing[k] for k in ('worker_elapsed_seconds', 'computation_elapsed_seconds',
                                                'process_wall_seconds', 'preflight_elapsed_seconds', 'model_load_seconds')),
            'Audited timing differs from current result/controller')
    policy_count = integer(audited['policy_decisions'], 'policy decisions')
    require(isinstance(result['stage_decisions'], dict) and sum(result['stage_decisions'].values()) == decisions,
            'Stage decision counts differ')
    require(policy_count == (result['stage_decisions'].get('CONTINUATION', 0)
                            if spec['arm'] in ('protected_bc', 'protected_ppo') else 0), 'Policy decisions misattributed')
    baseline_cost = result['baseline_real_cost']
    require(baseline_cost == audited['baseline_real_cost'], 'Audited baseline real cost differs')
    final_cost, last = result['final_real_cost'], result['checkpoints'][-1]
    require(set(final_cost) == set(last) and final_cost['phase'] == 'DONE', 'Final paid boundary schema differs')
    for key in last:
        if key not in ('worker_elapsed_seconds', 'elapsed_seconds'):
            require(final_cost[key] == last[key], 'Final paid boundary differs from last decision')
    for key, total in (('worker_elapsed_seconds', 'worker_elapsed_seconds'),
                       ('elapsed_seconds', 'computation_elapsed_seconds')):
        require(last[key] <= finite(final_cost[key], 'final paid ' + key) <= timing[total],
                'Final paid clock outside measured worker duration')
    phase_names = ('logical_points', 'physical_points', 'numerical_entries', 'requests',
                   'worker_elapsed_seconds', 'elapsed_seconds')
    expected_phases = {'WINDOW': {k: final_cost[k] for k in phase_names}} if baseline_cost is None else {
        'BASELINE': {k: baseline_cost[k] for k in phase_names},
        'AFTER_BASELINE': {k: final_cost[k] - baseline_cost[k] for k in phase_names}}
    require(all(v >= 0 for row in expected_phases.values() for v in row.values())
            and result['phase_costs'] == expected_phases, 'Phase costs omit or reset actual paid work')
    if spec['arm'] == 'window_reference':
        require(baseline_cost is None and audited['guard_established'] is None
                and audited['complete_baseline_replay'] is False, 'Standalone window claims a paid Beam guard')
    else:
        require(isinstance(baseline_cost, dict) and audited['complete_baseline_replay'] is True
                and type(audited['guard_established']) is bool, 'Protected baseline evidence missing')
        for name in ('logical_points', 'physical_points', 'numerical_entries', 'requests'):
            require(integer(baseline_cost[name], 'baseline ' + name) <= physical[name], 'Baseline cost exceeds full pipeline')
        require(baseline_cost['physical_points'] <= baseline_cost['logical_points'] <= 256
                and baseline_cost['numerical_entries'] == 3 + 4 * baseline_cost['physical_points'], 'Invalid paid baseline accounting')
        require(finite(baseline_cost['worker_elapsed_seconds'], 'baseline worker') <= timing['worker_elapsed_seconds']
                and finite(baseline_cost['elapsed_seconds'], 'baseline compute') <= timing['computation_elapsed_seconds'], 'Baseline clock exceeds pipeline')
        require(audited['baseline_quality_pct'] == baseline_cost['quality_pct'], 'Baseline quality differs')
        if audited['guard_established']:
            finite(audited['gain_over_paid_baseline_kwh'], 'strict nonnegative energy gain')
            require(baseline_cost['quality_pct'] is not None and quality is not None
                    and quality >= baseline_cost['quality_pct'], 'Paid Beam quality floor regressed')
        else:
            require(quality is None and baseline_cost['quality_pct'] is None, 'Uncertified guard reported as successful')
    return {**{k: spec[k] for k in ('cell', 'context_id', 'scenario_id', 'event_id', 'arm', 'seed', 'method', 'arm_slot')},
            **physical, **timing, 'quality_pct': quality, 'fully_certified': quality is not None,
            'status': result['status'], 'stop_reason': result['stop_reason'], 'search_decisions': decisions,
            'stage_decisions': deepcopy(result['stage_decisions']), 'policy_decisions': policy_count,
            'stage_component_seconds': deepcopy(result['stage_component_seconds']),
            'phase_costs': deepcopy(result['phase_costs']), 'final_real_cost': deepcopy(final_cost),
            'guard_established': audited['guard_established'], 'baseline_quality_pct': audited['baseline_quality_pct'],
            'baseline_real_cost': deepcopy(baseline_cost), 'gain_over_paid_baseline_kwh': audited['gain_over_paid_baseline_kwh'],
            'complete_baseline_replay': audited['complete_baseline_replay'],
            'baseline_signature_sha256': audited['baseline_signature_sha256'],
            'checkpoints': deepcopy(result['checkpoints']),
            'physical_statuses': deepcopy(audited['physical_statuses'])}


def aggregate_contexts(rows):
    output = []
    for context in range(8):
        for arm in ARMS:
            selected = sorted((r for r in rows if r['context_id'] == context and r['arm'] == arm), key=lambda r: r['seed'])
            require([r['seed'] for r in selected] == list(ARM_SEEDS[arm]), 'Missing context/arm seed')
            metrics, coverage = {}, {}
            for key in METRICS:
                metrics[key], coverage[key] = strict_mean([r[key] for r in selected], len(ARM_SEEDS[arm]))
            output.append({'context_id': context, 'scenario_id': selected[0]['scenario_id'], 'event_id': selected[0]['event_id'],
                           'arm': arm, 'seeds': list(ARM_SEEDS[arm]), 'cells': [r['cell'] for r in selected],
                           'expected_seed_count': len(selected), 'metrics': metrics, 'metric_coverage': coverage,
                           'fully_certified_cells': sum(r['fully_certified'] for r in selected)})
    return output


def pair(left, right, comparison, *, own_baseline=False):
    delta = {'quality_pp': left['quality_pct'] - right['quality_pct']
             if left['quality_pct'] is not None and right['quality_pct'] is not None else None}
    for key in PAIR_METRICS:
        delta[key] = left[key] - right[key] if left.get(key) is not None and right.get(key) is not None else None
    q = delta['quality_pp']
    return {'comparison': comparison, 'context_id': left['context_id'], 'scenario_id': left['scenario_id'],
            'event_id': left['event_id'], 'seed': left['seed'], 'left_cell': left['cell'], 'left_arm': left['arm'],
            'right_cell': right['cell'], 'right_arm': right['arm'], 'right_seed': right['seed'],
            'own_paid_baseline': own_baseline, 'left_quality_pct': left['quality_pct'], 'right_quality_pct': right['quality_pct'],
            'signed_delta': delta, 'quality_relation': 'missing' if q is None else 'win' if q > TIE_PP else 'loss' if q < -TIE_PP else 'tie',
            'cost_saving_fraction': {key: (right[key] - left[key]) / right[key]
                if right.get(key) is not None and right[key] > 0 and left.get(key) is not None else None
                for key in ('physical_points', 'process_wall_seconds')}}


def all_pairs(rows):
    lookup = {(r['context_id'], r['arm'], r['seed']): r for r in rows}
    output = []
    for context in range(8):
        for seed in SEEDS:
            ppo, bc = (lookup[context, arm, seed] for arm in ('protected_ppo', 'protected_bc'))
            output.extend((pair(ppo, bc, 'ppo_minus_bc'),
                pair(ppo, lookup[context, 'protected_rule', 0], 'ppo_minus_rule'),
                pair(ppo, lookup[context, 'window_reference', 0], 'ppo_minus_window'),
                pair(bc, lookup[context, 'protected_rule', 0], 'bc_minus_rule')))
    for row in rows:
        if row['baseline_real_cost'] is None:
            continue
        cost = row['baseline_real_cost']
        right = {**row, **{k: cost[k] for k in ('logical_points', 'physical_points', 'numerical_entries', 'requests')},
                 'quality_pct': cost['quality_pct'], 'process_wall_seconds': None,
                 'worker_elapsed_seconds': cost['worker_elapsed_seconds'], 'computation_elapsed_seconds': cost['elapsed_seconds'],
                 'arm': 'own_paid_beam'}
        output.append(pair(row, right, row['arm'] + '_minus_own_paid_beam', own_baseline=True))
    return output


def comparison_summary(pairs):
    output = {}
    for name in sorted({r['comparison'] for r in pairs}):
        selected = [r for r in pairs if r['comparison'] == name]
        seeds = sorted({r['seed'] for r in selected})
        context_rows = []
        for context in range(8):
            group = [r for r in selected if r['context_id'] == context]
            require(sorted(r['seed'] for r in group) == seeds, 'Comparison seed population differs')
            means, coverage = {}, {}
            for metric in ('quality_pp',) + PAIR_METRICS:
                means[metric], coverage[metric] = strict_mean([r['signed_delta'][metric] for r in group], len(seeds))
            context_rows.append({'context_id': context, 'signed_delta': means, 'metric_coverage': coverage,
                                 'quality_relations': dict(Counter(r['quality_relation'] for r in group))})
        output[name] = {'pairs': len(selected), 'contexts': 8, 'seeds': seeds, 'context_rows': context_rows,
            'context_balanced_delta': {metric: distribution([r['signed_delta'][metric] for r in context_rows], 8)
                for metric in ('quality_pp',) + PAIR_METRICS},
            'all_pair_quality_relations': dict(Counter(r['quality_relation'] for r in selected)),
            'per_seed': {str(seed): {metric: distribution([r['signed_delta'][metric] for r in selected if r['seed'] == seed], 8)
                for metric in ('quality_pp',) + PAIR_METRICS} for seed in seeds},
            'independent_units': 'Eight observed training contexts; paired seeds are repeated model endpoints, not additional faults.'}
    return output


def step_curve(checkpoints, axis, field, stopped_at):
    values, carry, index, latest = [], [], 0, None
    for boundary in axis:
        while index < len(checkpoints) and checkpoints[index][field] <= boundary:
            latest = checkpoints[index]['quality_pct']
            index += 1
        values.append(latest)
        carry.append(latest is not None and boundary > stopped_at)
    return {'quality_pct': values, 'after_stop_carry': carry}


def average_curves(items, axis_name, *, contexts=False):
    curves = [r[axis_name] for r in items]
    size = len(curves[0]['quality_pct'])
    values, coverage, carries = [], [], []
    for i in range(size):
        value, count = strict_mean([r['quality_pct'][i] for r in curves], len(curves))
        values.append(value)
        coverage.append(count)
        if contexts:
            carries.append(sum(r['after_stop_carry_count'][i] == r['expected_count'] and r['quality_pct'][i] is not None for r in curves))
        else:
            carries.append(sum(r['after_stop_carry'][i] for r in curves))
    return {'quality_pct': values,
            'context_coverage_count' if contexts else 'coverage_count': coverage,
            'expected_contexts' if contexts else 'expected_count': len(curves),
            'after_stop_carry_count': carries}


def make_curve_data(rows):
    physical_axis = list(range(257))
    time_axis = list(range(math.ceil(max(r['worker_elapsed_seconds'] for r in rows)) + 1))
    cells = []
    for row in rows:
        cells.append({**{k: row[k] for k in ('cell', 'context_id', 'scenario_id', 'event_id', 'arm', 'seed')},
            'stop_physical_points': row['physical_points'], 'stop_worker_elapsed_seconds': row['worker_elapsed_seconds'],
            'physical': step_curve(row['checkpoints'], physical_axis, 'physical_points', row['physical_points']),
            'worker_time': step_curve(row['checkpoints'], time_axis, 'worker_elapsed_seconds', row['worker_elapsed_seconds']),
            'exact_checkpoints': deepcopy(row['checkpoints'])})
    context_curves = []
    for context in range(8):
        for arm in ARMS:
            selected = [r for r in cells if r['context_id'] == context and r['arm'] == arm]
            require(sorted(r['seed'] for r in selected) == list(ARM_SEEDS[arm]), 'Curve seed population incomplete')
            context_curves.append({'context_id': context, 'arm': arm,
                'scenario_id': selected[0]['scenario_id'], 'event_id': selected[0]['event_id'],
                'cells': [r['cell'] for r in selected], 'seeds': list(ARM_SEEDS[arm]),
                **{axis: average_curves(selected, axis) for axis in ('physical', 'worker_time')}})
    groups = [{'arm': arm, **{axis: average_curves([r for r in context_curves if r['arm'] == arm], axis, contexts=True)
                            for axis in ('physical', 'worker_time')}} for arm in ARMS]
    return {'schema': 'clpu-integrated-curves-v1',
        'axes': {'physical_points': physical_axis, 'worker_elapsed_seconds': time_axis},
        'cells': cells, 'context_curves': context_curves, 'group_curves': groups,
        'semantics': {
            'quality': 'Only already fully certified incumbents; null before first certificate or whenever no certificate exists.',
            'physical_axis': 'Latest completed checkpoint with actual points <= budget. All free actions at the same point count are included; this is a physical-budget curve, not a time guarantee.',
            'worker_axis': 'Latest checkpoint whose recorded worker-entry elapsed time <= integer second; preflight and model loading included. Not full-process certificate time.',
            'stop_carry': 'After the recorded final physical cost or worker finish, only the already obtained final certificate is carried forward; marked in each cell and aggregated counts.',
            'mean': 'Mean over all scheduled seeds within each context, then all eight contexts. Any missing member makes that mean null; coverage counts retained.',
            'smoothing': 'None; right-continuous checkpoint step functions with no interpolation.',
            'full_process_time': 'Terminal controller-observed process duration only, in all_cells and cost summaries; not this time axis.'}}


def make_summary(rows, contexts, pairs, binding, controller, run_name):
    arms = {}
    for arm in ARMS:
        selected = [r for r in rows if r['arm'] == arm]
        grouped = [r for r in contexts if r['arm'] == arm]
        arms[arm] = {'cells': len(selected), 'contexts': 8, 'seeds': list(ARM_SEEDS[arm]),
            'context_balanced_metrics': {key: distribution([r['metrics'][key] for r in grouped], 8) for key in METRICS},
            'all_cell_timing': {key: distribution([r[key] for r in selected]) for key in
                ('process_wall_seconds', 'worker_elapsed_seconds', 'computation_elapsed_seconds', 'model_load_seconds')},
            'resource_totals_not_method_comparisons': {key: sum(r[key] for r in selected) for key in
                ('physical_points', 'logical_points', 'numerical_entries', 'process_wall_seconds')},
            'fully_certified_cells': sum(r['fully_certified'] for r in selected),
            'guard_established_cells': sum(r['guard_established'] is True for r in selected),
            'zero_policy_decision_cells': sum(r['policy_decisions'] == 0 for r in selected)}
    return {'status': STATUS, 'study': STUDY, 'run': run_name, 'utc': datetime.now(timezone.utc).isoformat(),
        'contexts': 8, 'cells': 64, 'audited_cells': 64, 'arms': arms, 'comparisons': comparison_summary(pairs),
        'total_cost': {key: sum(r[key] for r in rows) for key in ('physical_points', 'logical_points', 'numerical_entries')},
        'evaluation_controller_wall_seconds': finite(controller['wall_seconds'], 'controller wall'),
        'fully_certified_cells': sum(r['fully_certified'] for r in rows),
        'uncertified_cells_retained': [r['cell'] for r in rows if not r['fully_certified']],
        'new_training': 0, 'optimizer_updates': 0, 'new_report_physical_points': 0,
        'formal_holdout_access': 0, 'tune_access': 0, 'clpu_holdout_access': 0,
        'source_bindings': binding,
        'interpretation': [
            'Descriptive integrated evaluation on eight previously observed training contexts, not a confirmatory generalization test.',
            'All methods use the same electrical-equivalent cache. No no-new-cache protected PPO arm exists, so this run does not isolate cache speedup for the complete protected PPO pipeline.',
            'The standalone window has a 256-expansion cap. Protected arms pay original Beam then up to 256 additional expansions and common VERIFY, with a cumulative logical/actual point cap of 256.',
            'BC and same-seed PPO share teacher data and a BC starting model; PPO had additional interactions/updates. Total training data and training cost are not claimed equal.',
            'Method means first average fixed seeds within a context, then eight contexts. Twenty-four paired endpoints are not twenty-four independent faults.',
            'Whole-process terminal costs include launch/import/preflight/model loading/compute/logging/seal/observed exit; worker curves end before final serialization/seal.',
            'A paid Beam floor concerns the fixed planning objective and excludes computation waiting from restoration quality. Missing certificates are never counted as successful protection.',
            'No new performance success threshold, automatic training, sample removal, smoothing or attribution of common caching gains to the learner.']}


def fmt(value, *, signed=False):
    return '缺认证/未定义' if value is None else format(value, '+.6f' if signed else '.6f')


def markdown(summary, contexts):
    out = ['# 窗口与固定学习器集成对照', '',
        '全部64个冷任务及独立审计通过。保留全部8个已观察训练工况；本轮是描述性集成验证，不设新的效果成功门槛。', '',
        '| 方法 | 任务数 | 完整认证数 | 工况平衡质量/% | 工况平衡真实点 | 工况平衡完整进程/秒 |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        values = summary['arms'][arm]
        metric = values['context_balanced_metrics']
        out.append(f"| {arm} | {values['cells']} | {values['fully_certified_cells']} | {fmt(metric['quality_pct']['mean'])} | "
                   f"{fmt(metric['physical_points']['mean'])} | {fmt(metric['process_wall_seconds']['mean'])} |")
    out += ['', '先在每工况内平均三个固定模型种子，再平均8工况。缺少任何必要认证时相应均值为空，不删除该工况。单规则只有seed0，不把24个学习任务总成本直接除以8个规则任务总成本。', '',
        '| 比较：左减右 | 工况平衡质量差/pp | 最差工况平均质量差/pp | 真实点差 | 完整进程差/秒 |',
        '|---|---:|---:|---:|---:|']
    for name, values in summary['comparisons'].items():
        metric = values['context_balanced_delta']
        out.append(f"| {name} | {fmt(metric['quality_pp']['mean'], signed=True)} | {fmt(metric['quality_pp']['minimum'], signed=True)} | "
                   f"{fmt(metric['physical_points']['mean'], signed=True)} | {fmt(metric['process_wall_seconds']['mean'], signed=True)} |")
    out += ['', '质量差正值表示左侧更好，成本差负值表示左侧更省。对自身实付Beam的完整进程差未定义：阶段边界只有worker时钟，不能伪造基线进程退出时间。每seed的8工况统计、全部152个配对/自身基线增量均在机器可读文件中保留。', '',
        '| 工况 | 事件 | 方法 | 质量/% | 真实点 | 完整进程/秒 | 认证seed数/预定数 |',
        '|---|---:|---|---:|---:|---:|---|']
    for row in contexts:
        m = row['metrics']
        out.append(f"| {row['scenario_id']} | {row['event_id']} | {row['arm']} | {fmt(m['quality_pct'])} | "
                   f"{fmt(m['physical_points'])} | {fmt(m['process_wall_seconds'])} | {row['fully_certified_cells']}/{row['expected_seed_count']} |")
    out += ['', '曲线使用全部精确认证检查点：物理预算轴为0至256的整数，时间轴为worker入口后的整数秒。首次完整认证前留空；结束后只延续已经取得的证书并标注，不平滑或插值。完整进程时间仅用于终态成本表，不能替代证书取得时刻。', '',
        '各方法共同使用等价缓存，保护组共同支付完整原Beam。直接窗口参考与保护流程的扩展上限不同；无新缓存PPO对照未设置，因此本轮不能单独归因完整保护PPO的缓存提速。', '',
        'BC与同seed PPO共享教师和BC起点，PPO还有额外交互及更新；这里只复用固定端点，不声称训练数据总量或训练成本相等。', '',
        f"合计真实物理点 {summary['total_cost']['physical_points']}，数值入口 {summary['total_cost']['numerical_entries']}；新增训练、优化和报告求解均0。", '',
        '状态：`' + STATUS + '`。详见all_cells.json、all_pairs.json、context_summary.json、curve_data.json、summary.json和report_binding.json。', '']
    return '\n'.join(out)


def report(run, output=None):
    run = Path(run).resolve()
    output = Path(output).resolve() if output else run / 'derived_report'
    require(output != run and output.is_relative_to(run) and not output.exists(), 'New within-run report directory required')
    frozen = verify_manifest(run, 'prephysical_freeze_manifest.json', required=(
        'config.json', 'schedule.json', 'source_binding.json', 'physical_scope.json', 'model_bindings.json'))
    require(read(run / 'prepared.json')['freeze_sha256'] == sha(run / 'prephysical_freeze_manifest.json'), 'Prepared freeze changed')
    require(read(run / 'config.json')['integrated_study'] == STUDY, 'Wrong integrated study')
    reporter_key = 'source/scripts/report_clpu_integrated_pilot.py'
    require(reporter_key in frozen and frozen[reporter_key] == sha(Path(__file__)), 'Current report code differs from prephysical freeze')
    audit_manifest = verify_manifest(run / 'audit', required=('summary.json',))
    audit, controller = read(run / 'audit/summary.json'), read(run / 'controller_completion.json')
    require(audit['status'] == 'PASS' and audit['errors'] == [] and audit['audited_cells'] == 64
            and audit['new_audit_physical_points'] == audit['new_training_tasks'] == 0, 'All 64 independent audits must pass')
    bound = {
        'freeze_sha256': 'prephysical_freeze_manifest.json', 'schedule_sha256': 'schedule.json',
        'source_binding_sha256': 'source_binding.json', 'controller_completion_sha256': 'controller_completion.json',
        'controller_dispatch_sha256': 'controller_dispatch.jsonl', 'physical_scope_sha256': 'physical_scope.json',
        'model_bindings_sha256': 'model_bindings.json',
        'model_preflight_manifest_sha256': 'model_preflight/completion_manifest.json',
        'scope_review_manifest_sha256': 'scope_review/completion_manifest.json',
        'exact_key_review_manifest_sha256': 'exact_key_review/completion_manifest.json'}
    for key, name in bound.items():
        require(audit[key] == sha(run / name), 'Audited current source binding changed: ' + name)
    schedule = read(run / 'schedule.json')
    validate_population(schedule)
    cells = {r['cell'] for r in schedule}
    require(controller['status'] == 'COMPLETE_CELLS_PENDING_AUDIT' and controller['scheduled'] == 64
            and controller['errors'] == controller['unknown_cost_cells'] == []
            and len(controller['completed']) == 64 and set(controller['completed']) == cells,
            'Controller did not complete the entire population')
    require(len(controller['exits']) == 64 and {r['cell'] for r in controller['exits']} == cells
            and all(r['exit_code'] == 0 and r['drained_after_controller_error'] is False for r in controller['exits']),
            'Missing or failed real process exit')
    require(len(audit['results']) == 64 and {r['cell'] for r in audit['results']} == cells, 'Audit selectively omitted or duplicated a cell')
    require(len(audit['baseline_equality']) == 8
            and {r['context_id'] for r in audit['baseline_equality']} == set(range(8))
            and all(r['status'] == 'PASS' and r['cold_paid_baselines'] == 7 for r in audit['baseline_equality']),
            'All eight independently paid Beam baseline equalities required')
    audited, exits = ({r['cell']: r for r in collection} for collection in (audit['results'], controller['exits']))
    rows = []
    for spec in schedule:
        cell = (run / spec['cell']).resolve()
        require(cell != run and cell.is_relative_to(run), 'Cell path escaped run')
        require(audited[spec['cell']]['cell_completion_sha256'] == sha(cell / 'completion_manifest.json'), 'Audited current cell seal changed')
        required = ['spec.json', 'result.json', 'equivalent_cache_summary.json', 'cost_checkpoints.jsonl']
        if spec['protected']:
            required += ['baseline_result.json', 'baseline_checkpoint.json']
        verify_manifest(cell, required=required)
        require(not (cell / 'error.json').exists() and read(cell / 'spec.json') == spec, 'Cell error or schedule change')
        result = read(cell / 'result.json')
        require(result['oracle'] == read(cell / 'equivalent_cache_summary.json'), 'Result/oracle journal differs')
        checkpoint_lines = [json.loads(x) for x in (cell / 'cost_checkpoints.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
        require(result['checkpoints'] == checkpoint_lines, 'Exact checkpoint journal differs')
        if spec['protected']:
            checkpoint = read(cell / 'baseline_checkpoint.json')
            require(result['baseline_real_cost'] == checkpoint['real_cost'], 'Paid baseline cost checkpoint differs')
        row = case_row(spec, result, audited[spec['cell']], exits[spec['cell']]['process_wall_seconds'])
        row.update(cell_completion_sha256=sha(cell / 'completion_manifest.json'), result_sha256=sha(cell / 'result.json'))
        rows.append(row)
    totals = {key: sum(r[key] for r in rows) for key in ('physical_points', 'logical_points', 'numerical_entries')}
    require(all(totals[k] == audit[k] == controller['known_' + k] for k in totals), 'Report/audit/controller costs differ')
    require(totals['physical_points'] <= 16384 and totals['logical_points'] <= 16384
            and totals['numerical_entries'] == 64 * 3 + 4 * totals['physical_points'] <= 65728, 'Global actual cost cap exceeded')
    contexts, pairs = aggregate_contexts(rows), all_pairs(rows)
    binding = {'run': run.name, 'study': STUDY, **{key: sha(run / name) for key, name in bound.items()},
        **{name: sha(run / name) for name in bound.values()},
        'audit/summary.json': sha(run / 'audit/summary.json'),
        'audit/completion_manifest.json': sha(run / 'audit/completion_manifest.json'),
        'audit_summary_sha256': sha(run / 'audit/summary.json'), 'audit_completion_sha256': sha(run / 'audit/completion_manifest.json'),
        'reporter_sha256': sha(Path(__file__)),
        'cells': [{k: r[k] for k in ('cell', 'cell_completion_sha256', 'result_sha256')} for r in rows]}
    summary = make_summary(rows, contexts, pairs, binding, controller, run.name)
    curves = make_curve_data(rows)
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (('all_cells.json', rows), ('all_pairs.json', pairs), ('context_summary.json', contexts),
                        ('curve_data.json', curves), ('summary.json', summary), ('report_binding.json', binding)):
        save(output / name, value)
    (output / 'report.md').write_text(markdown(summary, contexts), encoding='utf-8')
    save(output / 'completion_manifest.json', {p.relative_to(output).as_posix(): sha(p) for p in output.rglob('*') if p.is_file()})
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', '--output', dest='output', type=Path)
    args = parser.parse_args()
    print(json.dumps(report(args.run, args.output), ensure_ascii=False))
