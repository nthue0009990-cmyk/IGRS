"""All-pair, audit-bound descriptive cache report; never invokes a solver."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.dont_write_bytecode = True
from d060_validation_common import canonical, read, require, sha, utc_now
from run_nature34_clpu_tail_pilot import save
from run_clpu_equivalent_cache_pilot import (ARMS, SOURCE_RUN, SOURCE_SHA, STUDY,
    verify_manifest, verify_exact_key_review, schedule_for)


def lines(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def finite(value, name):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0, 'Invalid nonnegative finite ' + name)
    return float(value)


def integer(value, name):
    require(type(value) is int and value >= 0, 'Invalid nonnegative integer ' + name)
    return value


def distribution(values):
    values = [finite(v, 'distribution sample') for v in values]
    if not values:
        return {'n':0, 'mean':None, 'median':None, 'p95_nearest_rank':None, 'maximum':None}
    ordered = sorted(values)
    return {'n':len(values), 'mean':statistics.mean(values), 'median':statistics.median(values),
        'p95_nearest_rank':ordered[math.ceil(.95*len(values))-1], 'maximum':max(values)}


def validate_checkpoints(points, oracle):
    require(bool(points), 'Initial certification boundary missing')
    previous = None
    for index, point in enumerate(points):
        require(integer(point['search_choices'], 'search choices') == index,
                'Every completed search boundary required')
        logical = integer(point['logical_points'], 'logical points')
        physical = integer(point['physical_points'], 'physical points')
        require(physical <= logical <= 256, 'Checkpoint point budgets inconsistent')
        require(integer(point['numerical_entries'], 'numerical entries') == 3+4*physical,
                'Checkpoint must charge actual initialization and physical entries')
        finite(point['elapsed_seconds'], 'checkpoint time')
        require(type(point['fully_certified']) is bool, 'Certificate availability must be explicit')
        if point['fully_certified']:
            require(finite(point['quality_pct'], 'quality') <= 100+1e-9, 'Quality outside percentage range')
        else:
            require(point['quality_pct'] is None, 'Uncertified quality must remain missing')
        if previous:
            require(all(point[k] >= previous[k] for k in
                ('logical_points','physical_points','numerical_entries','elapsed_seconds')), 'Checkpoint cost decreased')
            if previous['fully_certified']:
                require(point['fully_certified'] and point['quality_pct'] >= previous['quality_pct'],
                        'Certified incumbent regressed')
        previous = point
    require(points[-1]['logical_points'] == oracle['logical_points']
            and points[-1]['physical_points'] == oracle['physical_points']
            and points[-1]['numerical_entries'] == oracle['actual_numerical_entries'],
            'Final checkpoint/accounting differs')


def semantic_record(result, plans, steps, *, historical=False):
    """All search semantics, including exact first-failure records, excluding only timing/real cost."""
    result_keys = ('status','config','stop_reason','best_plan','initial_certificate',
                   'wait_baseline_quality_pct','completed_expansions','search_decisions','counts')
    decisions = []
    for row in steps:
        value = {k:v for k,v in row.items() if k != 'elapsed_seconds'}
        for name in ('oracle_before','oracle_after'):
            if name in value:
                value[name] = {k:row[name][k] for k in ('requests','points')}
        decisions.append(value)
    checkpoints = [{k:p[k] for k in ('search_choices','quality_pct','fully_certified')}
        | {'logical_points':p['physical_points' if historical else 'logical_points']}
        for p in result['checkpoints']]
    return {**{k:result[k] for k in result_keys}, 'plans':plans, 'steps':decisions,
            'checkpoints':checkpoints,
            'logical_points':result['oracle']['points' if historical else 'logical_points']}


def signature(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def first_failures(plans):
    return [{'plan_index':i, 'actions':p['actions'],
             'failed_step':p['validation']['failed_step'],
             'failed_time_seconds':p['validation']['failed_time_seconds'],
             'failed_state':p['validation']['failed_state'],
             'point_id':p['validation']['samples'][-1]['point_id'],
             'feedback':p['validation']['feedback']}
            for i,p in enumerate(plans) if p['status'] == 'FAIL']


def case_row(spec, result, process_wall, plans, steps, historical):
    require(result['spec'] == spec and spec['arm'] in ARMS, 'Wrong scheduled result identity')
    require(result['training_interactions'] == result['optimizer_updates'] == result['new_training'] == 0
            and result['formal_holdout_numerical_access'] is False and result['tune_numerical_access'] is False,
            'Study scope violated')
    oracle = result['oracle']
    require(oracle['variant'] == spec['arm'], 'Wrong oracle arm')
    for key in ('logical_points','physical_points','requests','semantic_cache_hits','electrical_reuses',
                'actual_numerical_entries','initialization_numerical_entries'):
        integer(oracle[key], key)
    require(oracle['physical_points'] <= oracle['logical_points'] <= 256
            and oracle['actual_numerical_entries'] == 3+4*oracle['physical_points']
            and oracle['initialization_numerical_entries'] == 3, 'Actual numerical cost not closed')
    require(oracle['logical_points']-oracle['physical_points'] == oracle['electrical_reuses'],
            'Electrical reuse does not reconcile actual cost')
    if spec['arm'] == ARMS[0]:
        require(oracle['electrical_reuses'] == 0, 'Semantic control used electrical aliases')
    validate_checkpoints(result['checkpoints'], oracle)
    require(result['completed_expansions'] == result['search_decisions'] == len(result['checkpoints'])-1 <= 256,
            'Expansion budget or checkpoints changed')
    semantics = semantic_record(result, plans, steps)
    source_semantics = semantic_record(*historical, historical=True)
    current_failures, source_failures = first_failures(plans), first_failures(historical[1])
    row = {**{k:spec[k] for k in ('cell','context_id','scenario_id','event_id','arm','pair_order','arm_launch_order')},
        'logical_points':oracle['logical_points'], 'physical_points':oracle['physical_points'],
        'numerical_entries':oracle['actual_numerical_entries'], 'requests':oracle['requests'],
        'semantic_cache_hits':oracle['semantic_cache_hits'], 'electrical_reuses':oracle['electrical_reuses'],
        'quality_pct':result['checkpoints'][-1]['quality_pct'], 'stop_reason':result['stop_reason'],
        'completed_expansions':result['completed_expansions'], 'checkpoints':result['checkpoints'],
        'status':result['status'], 'failed_plan_count':len(current_failures),
        'first_failures':current_failures, 'complete_plan_count':sum(p['status']=='COMPLETE' for p in plans),
        'semantic_signature':signature(semantics), 'historical_semantic_signature':signature(source_semantics),
        'matches_historical_semantics':semantics == source_semantics,
        'matches_historical_first_failures':current_failures == source_failures,
        'historical_plan_and_step_signatures':{'plans':signature(historical[1]),'steps':signature(source_semantics['steps'])},
        'process_wall_seconds':finite(process_wall,'whole process wall'),
        **{k:finite(result[k],k) for k in ('elapsed_seconds','computation_elapsed_seconds',
             'preflight_elapsed_seconds','runtime_initialization_seconds','initial_elapsed_seconds')}}
    require(row['elapsed_seconds'] >= row['computation_elapsed_seconds']
            and row['computation_elapsed_seconds'] >= row['initial_elapsed_seconds']
            and row['initial_elapsed_seconds'] >= row['runtime_initialization_seconds'], 'Timing decomposition inconsistent')
    return row


def pair_rows(semantic, equivalent):
    require(semantic['arm'] == ARMS[0] and equivalent['arm'] == ARMS[1], 'Pair arms differ')
    require(all(semantic[k] == equivalent[k] for k in ('context_id','scenario_id','event_id','pair_order')),
            'Different contexts paired')
    same = semantic['semantic_signature'] == equivalent['semantic_signature']
    quality_equal = semantic['quality_pct'] == equivalent['quality_pct']
    first_equal = semantic['first_failures'] == equivalent['first_failures']
    gate = same and quality_equal and first_equal and all(r['matches_historical_semantics']
        and r['matches_historical_first_failures'] for r in (semantic,equivalent))
    deltas = {k:equivalent[k]-semantic[k] for k in ('logical_points','physical_points','numerical_entries',
        'elapsed_seconds','computation_elapsed_seconds','process_wall_seconds','runtime_initialization_seconds')}
    base = semantic['physical_points']
    return {**{k:semantic[k] for k in ('context_id','scenario_id','event_id','pair_order')},
        'cells':{r['arm']:r['cell'] for r in (semantic,equivalent)},
        'equivalence_gate_passed':gate, 'both_match_source':all(r['matches_historical_semantics'] for r in (semantic,equivalent)),
        'same_full_search_semantics':same, 'same_first_failures':first_equal, 'same_final_quality':quality_equal,
        'quality_semantic_pct':semantic['quality_pct'], 'quality_equivalent_pct':equivalent['quality_pct'],
        'actual_physical_points':{r['arm']:r['physical_points'] for r in (semantic,equivalent)},
        'logical_points':{r['arm']:r['logical_points'] for r in (semantic,equivalent)},
        'electrical_reuses':equivalent['electrical_reuses'], 'equivalent_minus_semantic':deltas,
        'physical_point_saving_fraction':(base-equivalent['physical_points'])/base if base else None,
        'process_wall_saving_fraction':(semantic['process_wall_seconds']-equivalent['process_wall_seconds'])/semantic['process_wall_seconds']
            if semantic['process_wall_seconds'] else None,
        'timing':{r['arm']:{k:r[k] for k in ('process_wall_seconds','elapsed_seconds','computation_elapsed_seconds',
            'preflight_elapsed_seconds','runtime_initialization_seconds','initial_elapsed_seconds')} for r in (semantic,equivalent)}}


def report(run, output=None):
    run = Path(run).resolve()
    output = Path(output).resolve() if output else run/'derived_report'
    require(output != run and output.is_relative_to(run) and not output.exists(), 'New within-run report folder required')
    frozen = verify_manifest(run,'prephysical_freeze_manifest.json',
        ('schedule.json','source_cells.json','source_binding.json','physical_scope.json',
         'scope_review_binding.json','exact_key_review_binding.json'))
    require(sha(run/'prephysical_freeze_manifest.json') == read(run/'prepared.json')['freeze_sha256'], 'Freeze seal changed')
    require(read(run/'config.json')['equivalent_cache_study'] == STUDY, 'Wrong study')
    source = read(run/'source_binding.json')
    require(source == {'run':SOURCE_RUN,'completion_sha256':SOURCE_SHA}, 'Source identity changed')
    ctl = read(run/'controller_completion.json')
    require(ctl['status'] == 'COMPLETE_CELLS_PENDING_AUDIT' and not ctl['errors']
            and not ctl['unknown_cost_cells'], 'Controller incomplete; no partial-success report')
    verify_manifest(run/'audit',required=('summary.json',))
    audit = read(run/'audit/summary.json')
    require(audit['status'] == 'PASS' and not audit['errors']
            and audit['new_audit_physical_points'] == audit['new_training_tasks'] == 0, 'Independent audit incomplete')
    for key,name in [('freeze_sha256','prephysical_freeze_manifest.json'),('schedule_sha256','schedule.json'),
                     ('source_binding_sha256','source_binding.json'),('controller_completion_sha256','controller_completion.json')]:
        require(audit[key] == sha(run/name), 'Audit binding changed: '+name)
    require(audit['source_window_completion_sha256'] == SOURCE_SHA, 'Audit source differs')
    verify_exact_key_review(run)
    require(audit['exact_key_review_manifest_sha256'] == sha(run/'exact_key_review/completion_manifest.json'),
            'Audited final-key review seal differs')
    bindings = read(run/'source_cells.json')
    contexts = [{k:r[k] for k in ('context_id','scenario_id','event_id')} for r in bindings]
    schedule = read(run/'schedule.json')
    require(schedule == schedule_for(contexts), 'Frozen population/order/budgets differ')
    expected = {r['cell'] for r in schedule}
    require(len(schedule) == audit['audited_cells'] == ctl['scheduled'] == len(ctl['completed']) == 16
            and set(ctl['completed']) == expected, 'All 16 scheduled results required')
    audited = {r['cell']:r for r in audit['results']}
    exits = {r['cell']:r for r in ctl['exits']}
    require(len(audited) == len(audit['results']) == len(exits) == len(ctl['exits']) == 16
            and set(audited) == set(exits) == expected, 'Repeated/missing cells or exits')
    histories = {}
    for binding in bindings:
        directory = run/binding['historical_directory']
        require(directory.resolve().is_relative_to(run) and
                sha(directory/'completion_manifest.json') == binding['source_completion_sha256'], 'Historical seal changed')
        for name,digest in binding['source_files'].items():
            if name == 'points.jsonl.gz':
                continue
            require((directory/name).relative_to(run).as_posix() in frozen
                    and sha(directory/name) == digest, 'Historical comparison changed')
        histories[binding['context_id']] = (read(directory/'result.json'),lines(directory/'plans.jsonl.gz'),lines(directory/'steps.jsonl.gz'))
    cases = []
    for spec in schedule:
        cell = (run/spec['cell']).resolve()
        require(cell.is_relative_to(run) and exits[spec['cell']]['exit_code'] == 0, 'Cell incomplete')
        require(audited[spec['cell']]['cell_completion_sha256'] == sha(cell/'completion_manifest.json'), 'Audited cell seal changed')
        verify_manifest(cell,required=('spec.json','result.json','plans.jsonl.gz','steps.jsonl.gz','equivalent_cache_summary.json'))
        require(read(cell/'spec.json') == spec, 'Scheduled cell spec changed')
        result = read(cell/'result.json')
        require(result['oracle'] == read(cell/'equivalent_cache_summary.json'), 'Result/oracle dual ledger differs')
        row = case_row(spec,result,exits[spec['cell']]['process_wall_seconds'],
            lines(cell/'plans.jsonl.gz'),lines(cell/'steps.jsonl.gz'),histories[spec['context_id']])
        row.update(cell_completion_sha256=sha(cell/'completion_manifest.json'),result_sha256=sha(cell/'result.json'))
        cases.append(row)
    pairs = [pair_rows(next(r for r in cases if r['context_id']==i and r['arm']==ARMS[0]),
                       next(r for r in cases if r['context_id']==i and r['arm']==ARMS[1])) for i in range(8)]
    totals = {k:sum(r[k] for r in cases) for k in ('logical_points','physical_points','numerical_entries')}
    require(totals['physical_points'] == audit['physical_points'] == ctl['known_physical_points']
            and totals['numerical_entries'] == audit['numerical_entries'] == ctl['known_numerical_entries']
            and totals['logical_points'] == ctl['known_logical_points'], 'Controller/audit/report costs differ')
    arms = {}
    for arm in ARMS:
        selected = [r for r in cases if r['arm']==arm]
        arms[arm] = {'cells':len(selected), **{k:sum(r[k] for r in selected) for k in
            ('logical_points','physical_points','numerical_entries','requests','semantic_cache_hits','electrical_reuses')},
            'timing':{k:distribution([r[k] for r in selected]) for k in
                ('process_wall_seconds','elapsed_seconds','computation_elapsed_seconds','preflight_elapsed_seconds',
                 'runtime_initialization_seconds','initial_elapsed_seconds')},
            'sum_process_wall_seconds':sum(r['process_wall_seconds'] for r in selected)}
    gate = all(r['equivalence_gate_passed'] for r in pairs)
    reduced = arms[ARMS[1]]['physical_points'] < arms[ARMS[0]]['physical_points']
    summary = {'status':'EQUIVALENT_WITH_MEASURED_PHYSICAL_REDUCTION' if gate and reduced else
        'EQUIVALENT_WITHOUT_PHYSICAL_REDUCTION' if gate else 'EQUIVALENCE_NOT_ESTABLISHED',
        'run':run.name,'utc':utc_now(),'contexts':8,'cells':16,'equivalence_gate_passed':gate,
        'equivalent_pairs':sum(p['equivalence_gate_passed'] for p in pairs), **totals,'arms':arms,
        'physical_points_saved':arms[ARMS[0]]['physical_points']-arms[ARMS[1]]['physical_points'],
        'physical_saving_fraction':1-arms[ARMS[1]]['physical_points']/arms[ARMS[0]]['physical_points'],
        'process_wall_sum_saving_fraction':1-arms[ARMS[1]]['sum_process_wall_seconds']/arms[ARMS[0]]['sum_process_wall_seconds'],
        'process_wall_faster_pairs':sum(p['equivalent_minus_semantic']['process_wall_seconds']<0 for p in pairs),
        'evaluation_wall_seconds':finite(ctl['wall_seconds'],'controller wall'),
        'new_training':0,'optimizer_updates':0,'formal_holdout_access':0,'tune_access':0,
        'source_bindings':{n:sha(run/n) for n in ('schedule.json','source_binding.json','source_cells.json','physical_scope.json',
            'scope_review_binding.json','exact_key_review_binding.json','exact_key_review/completion_manifest.json',
            'prephysical_freeze_manifest.json','controller_completion.json','audit/summary.json',
            'audit/completion_manifest.json')},
        'limits':['Eight fixed observed training contexts, one paired run each, two concurrent workers; descriptive timing only, no confidence interval or generalization claim.',
            'Historical G1 is used only for exact search-semantic comparison; its old timing is never used to estimate speed.',
            'Semantic identity and electrical-equivalent arms use the same wrapper and logical budget. All valid failures, complete certificates and original source measurements are retained.',
            'Physical reuse is a common engineering baseline improvement. No trained policy or same-data BC is evaluated; no PPO/BC superiority is claimed.',
            'Process wall includes launcher/import/preflight/compute/output/seal and observed exit; controller polling resolution is about 0.1 second. Worker elapsed excludes final result serialization/seal.',
            'A saved physical evaluation is not a changed logical sample or removed certification constraint. Reused certificates retain current-time identity and feedback.']}
    output.mkdir(parents=True,exist_ok=False)
    save(output/'all_cells.json',cases);save(output/'all_pairs.json',pairs);save(output/'summary.json',summary)
    (output/'report.md').write_text(markdown(summary,pairs),encoding='utf-8')
    save(output/'completion_manifest.json',{p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()})
    return summary


def markdown(summary,pairs):
    rows=['# 物理输入等价缓存小试','',
        f"语义等价门槛：{'通过' if summary['equivalence_gate_passed'] else '未通过'}（{summary['equivalent_pairs']}/8 对）。完整保留 16 个冷启动结果。",'',
        '| 工况/事件 | 两组同历史且互相等价 | 语义组真实点 | 等价组真实点 | 真实点减少 | 完整进程时间差/秒 |',
        '|---|---|---:|---:|---:|---:|']
    for p in pairs:
        rows.append(f"| {p['scenario_id']} / {p['event_id']} | {'是' if p['equivalence_gate_passed'] else '否'} | "
            f"{p['actual_physical_points'][ARMS[0]]} | {p['actual_physical_points'][ARMS[1]]} | "
            f"{-p['equivalent_minus_semantic']['physical_points']} | {p['equivalent_minus_semantic']['process_wall_seconds']:+.6f} |")
    rows += ['',f"实际物理点减少 {summary['physical_points_saved']}，相对本轮语义缓存组减少 {100*summary['physical_saving_fraction']:.3f}%。",
        f"两组完整进程时间总和的相对减少为 {100*summary['process_wall_sum_saving_fraction']:.3f}%；等价组在 {summary['process_wall_faster_pairs']}/8 对中更快。完整进程时间与初始化、计算、预检时间均在机器可读结果中逐项保留。",
        f"本轮合计 {summary['physical_points']} 个真实物理点、{summary['numerical_entries']} 个数值入口；训练和优化均为 0。",'',
        '同一机器两进程并发，每工况仅一对。这是固定已观察训练工况的小试，时间统计只作描述；p95 使用最近秩，在 8 个值中等于最大值。',
        '旧 G1 时间不参与加速比较。等价门槛同时核对全部计划、首次失败位置及反馈、选择轨迹、完整认证曲线和终止原因；节省求解不意味着减少逻辑认证或放宽物理约束。',
        '这是所有方法均可使用的公共计算优化，尚未证明 PPO 超过简单窗口或同数据 BC。','']
    return '\n'.join(rows)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();print(json.dumps(report(args.run,args.output),ensure_ascii=False))
