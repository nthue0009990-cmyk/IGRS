"""Describe every paid, audit-bound candidate fork in the window teacher pilot.

This report measures possible teacher signal.  The best measured fork is an
offline diagnostic, never a trained policy or a free online decision.
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

EPS_PP = 1e-9
GATE_GAIN_PP = .1
MODES = ('normal', 'coverage')
SOURCE_RUN = '20261027T121023_469270Z_clpu_window_pilot'
SOURCE_SHA = '9967ea771263b5b9bb45835116dec1c5fd70f8908c17ec2348d90d4a6377fc7e'
GRID = 'integer_new_physical_points_1_through_allowance_inclusive'


def save(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def integer(value, name):
    require(type(value) is int and value >= 0, 'Invalid nonnegative integer: ' + name)
    return value


def finite(value, name, *, minimum=None, maximum=None):
    require(type(value) in (int, float) and math.isfinite(value), 'Invalid finite number: ' + name)
    require(minimum is None or value >= minimum, 'Below allowed range: ' + name)
    require(maximum is None or value <= maximum, 'Above allowed range: ' + name)
    return float(value)


def verify_manifest(directory, *, name='completion_manifest.json', required=()):
    directory = Path(directory)
    manifest = read(directory / name)
    require(isinstance(manifest, dict), 'Invalid sealed manifest')
    for key in required:
        require(key in manifest, 'Missing sealed evidence: ' + key)
    for key, digest in manifest.items():
        require(sha(inside(directory, key)) == digest, 'Changed sealed evidence: ' + key)
    return manifest


def validate_curve(initial, trajectory, allowance, actual_points):
    """The incumbent changes only at a completed physical certificate boundary."""
    finite(initial, 'initial certified quality', minimum=0, maximum=1)
    integer(allowance, 'point allowance')
    integer(actual_points, 'actual physical points')
    require(actual_points <= allowance <= 128, 'Physical point allowance exceeded')
    require(trajectory and trajectory[0] == {'new_physical_points': 0, 'quality': initial,
        'kind': 'INITIAL_CERTIFIED_INCUMBENT'}, 'Missing initial complete certificate')
    point, quality = 0, initial
    for row in trajectory[1:]:
        current = integer(row['new_physical_points'], 'certificate point')
        value = finite(row['quality'], 'certified quality', minimum=0, maximum=1)
        require(point <= current <= actual_points and quality <= value
                and row['kind'] == 'COMPLETE_CERTIFICATE', 'Invalid certificate trajectory')
        point, quality = current, value
    return trajectory


def curve_auc(initial, trajectory, allowance):
    """Mean incumbent gain on the declared integer grid, including hit points."""
    if allowance == 0:
        return None
    return 100 * math.fsum(
        next(row['quality'] for row in reversed(trajectory) if row['new_physical_points'] <= budget)
        - initial for budget in range(1, allowance + 1)) / allowance


def first_reach(trajectory, target_quality):
    finite(target_quality, 'baseline target', minimum=0, maximum=1)
    for row in trajectory:
        if 100 * (row['quality'] - target_quality) >= -EPS_PP:
            return {'status': 'REACHED', 'new_physical_points': row['new_physical_points'],
                    'quality_pct': 100 * row['quality']}
    return {'status': 'NOT_REACHED_WITHIN_BOUNDED_FORK',
            'new_physical_points': None, 'quality_pct': None}


def case_row(spec, result):
    require(result['spec'] == spec and result['status'] == 'BOUNDED_COUNTERFACTUAL_COMPLETE',
            'Fork result differs from frozen schedule')
    require(result['new_training'] == result['optimizer_updates'] == 0
            and not result['formal_holdout_numerical_access'] and not result['tune_numerical_access'],
            'Teacher scope changed')
    require(result['forced_branch_id'] == spec['forced_branch_id'], 'Forced branch differs')
    points = integer(result['live_oracle_summary']['points'], 'new physical points')
    entries = integer(result['live_oracle_summary']['actual_numerical_entries'], 'new numerical entries')
    require(points == result['oracle']['new_physical_points']
            and result['oracle']['historical_restore_new_physical_points'] == 0,
            'Hidden restore or physical costs')
    allowance = integer(result['new_point_allowance'], 'new point allowance')
    require(allowance == spec['effective_new_point_allowance'] and spec['max_new_points'] == 128
            and spec['max_expansions'] == 16, 'Frozen fork budget changed')
    require(result['curve_budget_grid'] == GRID, 'Curve budget grid changed')
    trajectory = validate_curve(result['initial_quality'], result['quality_point_trajectory'], allowance, points)
    final = finite(result['final_quality'], 'final quality', minimum=0, maximum=1)
    require(abs(100 * (trajectory[-1]['quality'] - final)) <= EPS_PP,
            'Final quality differs from observed complete certificates')
    plan = result['best_plan']
    require(plan and plan['certified_complete'] and plan['status'] == 'COMPLETE'
            and plan['validation']['status'] == 'COMPLETE', 'Final plan lacks a complete certificate')
    require(abs(100 * (plan['validation']['mean_served_fraction'] - final)) <= EPS_PP,
            'Final plan certificate quality differs')
    measured = curve_auc(result['initial_quality'], trajectory, allowance)
    require((measured is None and result['auc_gain_pp'] is None) or
            (measured is not None and result['auc_gain_pp'] is not None
             and abs(finite(result['auc_gain_pp'], 'curve gain') - measured) <= EPS_PP),
            'Reported U differs from certified quality curve')
    require(len(result['certificate_updates']) == len(trajectory) - 1,
            'Complete certificate update inventory differs')
    for update, row in zip(result['certificate_updates'], trajectory[1:]):
        require(update['new_physical_points'] == row['new_physical_points']
                and update['quality_after'] == row['quality'], 'Certificate update boundary differs')
    expansions = integer(result['completed_expansions'], 'completed expansions')
    require(1 <= expansions <= 16, 'Missing forced step or excess continuation')
    for key, expected in [('g1_pp', 100 * (result['first_step_quality'] - result['initial_quality'])),
                          ('gH_pp', 100 * (final - result['initial_quality']))]:
        require(abs(finite(result[key], key) - expected) <= EPS_PP, 'Gain label differs: ' + key)
    return {**spec, 'snapshot_sha256': result['snapshot_sha256'],
        'source_snapshot_file_sha256': result['source_snapshot_file_sha256'],
        'initial_quality': result['initial_quality'], 'final_quality': final,
        'initial_quality_pct': 100 * result['initial_quality'], 'final_quality_pct': 100 * final,
        'first_step_quality_pct': 100 * result['first_step_quality'],
        'g1_pp': result['g1_pp'], 'gH_pp': result['gH_pp'], 'auc_gain_pp': measured,
        'new_point_allowance': allowance, 'new_physical_points': points,
        'new_numerical_entries': entries, 'completed_expansions': expansions,
        'stop_reason': result['stop_reason'],
        'elapsed_seconds': finite(result['elapsed_seconds'], 'worker elapsed seconds', minimum=0),
        'certificate_updates': result['certificate_updates'], 'quality_point_trajectory': trajectory,
        'trajectory_semantics': 'Certified incumbent held after bounded termination through declared point allowance'}


def state_effect(rows):
    require(2 <= len(rows) <= 3, 'Available state requires two or three measured candidates')
    require(len({row['forced_branch_id'] for row in rows}) == len(rows), 'Repeated measured candidate')
    require(len({row['snapshot_sha256'] for row in rows}) == 1, 'Different source states paired')
    require(len({row['source_snapshot_file_sha256'] for row in rows}) == 1, 'Different snapshot files paired')
    require(len({row['new_point_allowance'] for row in rows}) == 1, 'Different candidate allowances')
    require(max(row['initial_quality_pct'] for row in rows) -
            min(row['initial_quality_pct'] for row in rows) <= EPS_PP, 'Different initial certificate qualities')
    baselines = [row for row in rows if 'window_beam' in row['roles']]
    require(len(baselines) == 1, 'Exactly one current WindowBeam baseline required')
    baseline = baselines[0]
    candidates = []
    for row in rows:
        delta = None if row['auc_gain_pp'] is None or baseline['auc_gain_pp'] is None else (
            row['auc_gain_pp'] - baseline['auc_gain_pp'])
        gap = row['final_quality_pct'] - baseline['final_quality_pct']
        alternative = row['forced_branch_id'] != baseline['forced_branch_id']
        eligible = alternative and delta is not None and delta >= GATE_GAIN_PP and gap >= -EPS_PP
        candidates.append({'branch_id': row['forced_branch_id'], 'roles': row['roles'],
            'cell': row['cell'], 'auc_gain_pp': row['auc_gain_pp'], 'U_minus_window_beam_pp': delta,
            'g1_pp': row['g1_pp'], 'gH_pp': row['gH_pp'], 'final_quality_pct': row['final_quality_pct'],
            'final_quality_minus_window_beam_pp': gap, 'new_physical_points': row['new_physical_points'],
            'new_numerical_entries': row['new_numerical_entries'],
            'first_reach_bounded_beam_final': first_reach(row['quality_point_trajectory'], baseline['final_quality']),
            'counts_for_investment_gate': eligible, 'stop_reason': row['stop_reason']})
    values = [row['auc_gain_pp'] for row in rows if row['auc_gain_pp'] is not None]
    alternatives = [row for row in candidates if row['branch_id'] != baseline['forced_branch_id']]
    valid_alternatives = [row for row in alternatives if row['U_minus_window_beam_pp'] is not None]
    best = max(valid_alternatives, key=lambda row: (row['U_minus_window_beam_pp'], -row['branch_id'])) if valid_alternatives else None
    return {'baseline_branch_id': baseline['forced_branch_id'], 'baseline_cell': baseline['cell'],
        'baseline_auc_gain_pp': baseline['auc_gain_pp'], 'baseline_final_quality_pct': baseline['final_quality_pct'],
        'auc_span_pp': max(values) - min(values) if len(values) == len(rows) else None,
        'final_quality_span_pp': max(row['final_quality_pct'] for row in rows) - min(row['final_quality_pct'] for row in rows),
        'best_measured_alternative_branch_id': best['branch_id'] if best else None,
        'best_measured_alternative_U_delta_pp': best['U_minus_window_beam_pp'] if best else None,
        'best_measured_alternative_final_quality_delta_pp': best['final_quality_minus_window_beam_pp'] if best else None,
        'eligible_alternative_branch_ids': [row['branch_id'] for row in candidates if row['counts_for_investment_gate']],
        'has_rule_superiority_signal': any(row['counts_for_investment_gate'] for row in candidates),
        'unreached_baseline_target_candidates': sum(row['first_reach_bounded_beam_final']['status'] != 'REACHED' for row in candidates),
        'candidates': candidates}


def mechanism_gate(states):
    improving = [row for row in states if row.get('effect', {}).get('has_rule_superiority_signal')]
    faults = {row['scenario_id'] for row in improving}
    events = {row['event_id'] for row in improving}
    passed = len(improving) >= 2 and len(faults) >= 2 and len(events) >= 2
    return {'passed': passed, 'qualifying_states': len(improving), 'qualifying_faults': len(faults),
        'qualifying_events': len(events), 'qualifying_state_ids': [row['state_id'] for row in improving],
        'required': {'states': 2, 'faults': 2, 'events': 2, 'alternative_U_gain_pp': GATE_GAIN_PP,
                     'minimum_final_quality_delta_pp': -EPS_PP},
        'interpretation': 'Bounded teacher investment gate only; not learned-policy superiority, BC superiority or generalization'}


def validate_population(states, schedule):
    require(len(states) == 16 and len({row['state_id'] for row in states}) == 16,
            'All sixteen predeclared state slots must be retained')
    require({row['context_id'] for row in states} == set(range(8)), 'Eight training context IDs changed')
    contexts = {}
    for row in states:
        key = row['context_id']
        identity = row['scenario_id'], row['event_id']
        require(key not in contexts or contexts[key] == identity, 'Context identity differs between modes')
        contexts[key] = identity
        require(row['state_id'] == f"c{key:02d}_{row['mode']}" and row['mode'] in MODES,
                'State slot identity or mode changed')
        require(row['status'] in ('AVAILABLE', 'UNAVAILABLE'), 'Unknown state availability')
    require(len(set(contexts.values())) == 8 and len({s for s, _ in contexts.values()}) == 8
            and Counter(e for _, e in contexts.values()) == Counter({30: 2, 20: 2, 6: 2, 3: 2}),
            'Original balanced training population changed')
    expected = set()
    for state in states:
        candidates = state['candidates']
        if state['status'] == 'UNAVAILABLE':
            require(not candidates and state['reason'], 'Unavailable slot must retain reason and no substitute forks')
            continue
        require(2 <= len(candidates) <= 3 and len({c['branch_id'] for c in candidates}) == len(candidates),
                'Available state candidate inventory differs')
        require(sum('window_beam' in c['roles'] for c in candidates) == 1, 'WindowBeam candidate missing or repeated')
        expected.update((state['state_id'], candidate['branch_id']) for candidate in candidates)
    actual = [(spec['state_id'], spec['forced_branch_id']) for spec in schedule]
    require(len(actual) == len(set(actual)) == len(expected) and set(actual) == expected
            and len({spec['cell'] for spec in schedule}) == len(schedule),
            'Missing, repeated or extra scheduled candidate')
    lookup = {row['state_id']: row for row in states}
    for spec in schedule:
        state = lookup[spec['state_id']]
        for key in ('context_id', 'scenario_id', 'event_id', 'source_cell', 'completed_steps', 'snapshot'):
            require(spec[key] == state[key], 'Scheduled candidate state differs: ' + key)
        expected_candidate = next(c for c in state['candidates'] if c['branch_id'] == spec['forced_branch_id'])
        require(spec['roles'] == expected_candidate['roles'], 'Scheduled candidate roles differ')


def report(run, output=None):
    run = Path(run).resolve()
    output = Path(output).resolve() if output else run / 'derived_report'
    require(output != run and output.is_relative_to(run), 'Derived report must remain within its run')
    require(not output.exists(), 'Do not overwrite a prior scientific report')
    ctl = read(run / 'controller_completion.json')
    require(ctl['status'] == 'COMPLETE_CELLS_PENDING_AUDIT' and not ctl['errors']
            and not ctl['unknown_cost_cells'], 'Controller completion/cost is not closed')
    verify_manifest(run, name='prephysical_freeze_manifest.json',
                    required=('states.json', 'schedule.json', 'source_binding.json'))
    audit = read(run / 'audit/summary.json')
    require(audit['status'] == 'PASS', 'Independent teacher audit has not passed')
    verify_manifest(run / 'audit', required=('summary.json',))
    for key, name in [('schedule_sha256', 'schedule.json'), ('states_sha256', 'states.json'),
                      ('source_binding_sha256', 'source_binding.json'),
                      ('freeze_sha256', 'prephysical_freeze_manifest.json'),
                      ('controller_completion_sha256', 'controller_completion.json')]:
        require(audit[key] == sha(run / name), 'Audit binding differs: ' + name)
    source = read(run / 'source_binding.json')
    require(source['run'] == SOURCE_RUN and source['completion_sha256'] == SOURCE_SHA
            and sha(ROOT / 'runs' / source['run'] / 'completion_manifest.json') == SOURCE_SHA,
            'Sealed source-window run binding differs')
    states, schedule = read(run / 'states.json'), read(run / 'schedule.json')
    validate_population(states, schedule)
    expected = {row['cell'] for row in schedule}
    require(audit['audited_cells'] == len(schedule) == len(ctl['completed'])
            and set(ctl['completed']) == expected, 'Incomplete scheduled candidate population')
    bindings = {row['cell']: row for row in audit['results']}
    require(len(bindings) == len(audit['results']) == len(schedule) and set(bindings) == expected,
            'Independent audit omitted/repeated/added a scheduled candidate')
    cases, by_state = [], {}
    for spec in schedule:
        cell = inside(run, spec['cell'])
        require(bindings[spec['cell']]['cell_completion_sha256'] == sha(cell / 'completion_manifest.json'),
                'Audited cell seal differs')
        verify_manifest(cell, required=('spec.json', 'result.json'))
        require(read(cell / 'spec.json') == spec, 'Cell spec differs from frozen schedule')
        row = case_row(spec, read(cell / 'result.json'))
        require(row['source_snapshot_file_sha256'] == sha(inside(run, spec['snapshot'])),
                'Bound source snapshot changed')
        row.update(cell_completion_sha256=sha(cell / 'completion_manifest.json'),
                   result_sha256=sha(cell / 'result.json'))
        cases.append(row)
        by_state.setdefault(spec['state_id'], []).append(row)
    described = []
    for state in states:
        row = dict(state)
        if state['status'] == 'AVAILABLE':
            row['effect'] = state_effect(by_state.get(state['state_id'], []))
        else:
            require(state['state_id'] not in by_state, 'Unavailable state was silently replaced')
        described.append(row)
    gate = mechanism_gate(described)
    points, entries = (sum(row[key] for row in cases) for key in ('new_physical_points', 'new_numerical_entries'))
    require(points == ctl['known_physical_points'] == audit['physical_points']
            and entries == ctl['known_numerical_entries'] == audit['numerical_entries'],
            'Controller, audit and report costs differ')
    effects = [row['effect'] for row in described if 'effect' in row]
    deltas = [row['best_measured_alternative_U_delta_pp'] for row in effects
              if row['best_measured_alternative_U_delta_pp'] is not None]
    summary = {'status': 'TEACHER_SIGNAL_FOR_BOUNDED_LEARNING' if gate['passed'] else 'NO_RULE_SUPERIORITY_SIGNAL',
        'run': run.name, 'utc': utc_now(), 'training_contexts': 8, 'predeclared_states': 16,
        'available_states': len(effects), 'unavailable_states': 16 - len(effects), 'measured_forks': len(cases),
        'new_physical_points': points, 'new_numerical_entries': entries,
        'source_restore_physical_points': 0, 'new_optimizer_updates': 0, 'gate': gate,
        'unavailable_reasons': dict(Counter(row['reason'] for row in states if row['status'] == 'UNAVAILABLE')),
        'fork_stop_counts': dict(Counter(row['stop_reason'] for row in cases)),
        'unreached_baseline_target_candidates': sum(row['unreached_baseline_target_candidates'] for row in effects),
        'descriptive_best_measured_alternative_U_delta_pp': {
            'available_states': len(deltas), 'mean': statistics.mean(deltas) if deltas else None,
            'minimum': min(deltas) if deltas else None, 'maximum': max(deltas) if deltas else None,
            'scope': 'Offline per-state measured best alternative; not a learned or deployable selector'},
        'evaluation_wall_seconds': finite(ctl['wall_seconds'], 'controller wall seconds', minimum=0),
        'sum_worker_elapsed_seconds': sum(row['elapsed_seconds'] for row in cases),
        'source_bindings': {name: sha(run / name) for name in ('states.json', 'schedule.json',
            'source_binding.json', 'prephysical_freeze_manifest.json', 'controller_completion.json',
            'audit/summary.json', 'audit/completion_manifest.json')},
        'scope': 'Fixed observed training contexts; bounded teacher alternatives; no model, BC comparison or unseen-fault evaluation',
        'next_action': 'No automatic training; inspect this mechanism gate before starting any bounded learning stage'}
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'all_forks.json', cases)
    save(output / 'all_states.json', described)
    save(output / 'summary.json', summary)
    (output / 'report.md').write_text(markdown(summary, described), encoding='utf-8')
    save(output / 'completion_manifest.json', {path.relative_to(output).as_posix(): sha(path)
         for path in output.rglob('*') if path.is_file()})
    return summary


def fmt(value, digits=6):
    return '不可用' if value is None else f'{value:.{digits}f}'


def markdown(summary, states):
    gate = summary['gate']
    lines = ['# 候选认证调度教师小试', '',
        '**' + ('出现继续有限学习的教师信号。' if gate['passed'] else '未出现足够的规则超越信号，不自动扩充或训练。') + '**', '',
        f"固定8个既有训练工况、16个事前状态位置；{summary['available_states']}个可用，{summary['unavailable_states']}个缺失，实际完成{summary['measured_forks']}个独立付费分叉。",
        f"新增{summary['new_physical_points']}物理点、{summary['new_numerical_entries']}数值入口；历史状态恢复新增物理点0、模型更新0。全部分叉通过独立物理、搜索和完整证书时点复核。", '',
        'U是从新增物理点预算1到该状态剩余允许预算B，完整认证最好供电质量相对起始值的平均增益，单位为百分点。完整证书在其实际付费点完成后才生效；提前结束后明确保持终态，不假定未验证方案可行。每个候选只强制首次选择，其后使用同一覆盖窗口Beam续搜。', '',
        f"事前投入门槛：至少2个状态存在相对WindowBeam的U增益≥0.1pp且最终质量不低于它（容差1e-9pp）的已测替代候选，覆盖至少2个故障和2个事件。实际符合{gate['qualifying_states']}个状态、{gate['qualifying_faults']}个故障、{gate['qualifying_events']}个事件。", '',
        '| 工况 | 窗口模式 | 状态/搜索位置 | Beam U/pp | 候选U跨度/pp | 最佳已测替代 ΔU/pp | 该替代最终Δ质量/pp | 达到门槛 |',
        '|---|---|---|---:|---:|---:|---:|---|']
    for state in states:
        name = state['scenario_id'].replace('N2__line.', '').replace('__line.', '/') + f" · 事件{state['event_id']}"
        effect = state.get('effect')
        if effect:
            lines.append(f"| {name} | {state['mode']} | 可用/{state['completed_steps']} | {fmt(effect['baseline_auc_gain_pp'])} | "
                f"{fmt(effect['auc_span_pp'])} | {fmt(effect['best_measured_alternative_U_delta_pp'])} | "
                f"{fmt(effect['best_measured_alternative_final_quality_delta_pp'])} | {'是' if effect['has_rule_superiority_signal'] else '否'} |")
        else:
            lines.append(f"| {name} | {state['mode']} | 缺失：{state['reason']} | — | — | — | — | 不计为成功 |")
    lines += ['', '| 状态 | 候选/角色 | U/pp | 最终质量/% | 相对Beam ΔU/pp | 新物理点 | 达到本状态Beam终态所需新点 |',
              '|---|---|---:|---:|---:|---:|---:|']
    for state in states:
        for row in state.get('effect', {}).get('candidates', []):
            target = row['first_reach_bounded_beam_final']
            reach = str(target['new_physical_points']) if target['status'] == 'REACHED' else '预算内未达到'
            lines.append(f"| {state['state_id']} | {row['branch_id']}/{'、'.join(row['roles'])} | {fmt(row['auc_gain_pp'])} | "
                f"{fmt(row['final_quality_pct'])} | {fmt(row['U_minus_window_beam_pp'])} | {row['new_physical_points']} | {reach} |")
    lines += ['',
        '所有缺失位置、失败候选、并列、负差和未达标结果均保留；未达到目标的成本为null并附状态，不填零、不剔除。未测候选保持未知。表中“最佳已测替代”来自离线看完多个真实分叉后的描述，不是在线学习策略；它与门槛允许的最终质量条件分别报告。', '',
        '门槛通过只支持开展有限学习，不代表已超过同数据BC、已获得泛化收益或已解决最终创新问题。门槛不通过表示当前有限配置未提供足够信号，也不等于证明所有候选或其他预算下无收益。', '',
        f"评价墙钟{summary['evaluation_wall_seconds']:.3f}秒，全部worker累计{summary['sum_worker_elapsed_seconds']:.3f}秒；实际全部分叉计算均计入，历史源轨迹成本另属原封存试验。这些教师生成成本不能当作最终方法的在线运行成本。", '',
        'all_forks.json包含每条完整认证质量曲线、实际成本和封存绑定；all_states.json包含全部16个状态、逐候选有符号差及到达状态。原始物理失败和数值入口在各episode目录。', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    print(json.dumps(report(args.run, args.output), ensure_ascii=False, allow_nan=False))
