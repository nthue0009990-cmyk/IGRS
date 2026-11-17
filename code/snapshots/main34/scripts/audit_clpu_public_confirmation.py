"""Independent audit for the frozen 24-context, 48-cell P/E confirmation.

Reviewed independent replay is reused. NEW population provenance is explicit:
formal faults and holdout events are never relabelled as training inputs.
No production search, driver, learner, or solver is imported.
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import math
import ast
import sys
import time
import traceback
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
sys.dont_write_bytecode=True
import audit_clpu_public_mechanism_pilot as public

gf,energy,staged,shared,early,prior,branch,window,old=(public.gf,public.energy,public.staged,
    public.shared,public.early,public.prior,public.branch,public.window,public.old)
require,close,same=public.require,public.close,public.same
read,save,sha,digest,read_lines=public.read,public.save,public.sha,public.digest,public.read_lines
private=public.private
STUDY='CLPU_PUBLIC_CONFIRMATION_COMPACT_V1'
SCHEMA='clpu-public-confirmation-protected-result-v1'
PROTOCOL='docs/public_confirmation_compact_protocol_v1.md'
ARMS=('P','E')
CELLS,CONTEXTS,TOTAL_CAP=48,24,24576
SOURCE_RUN='20261031T141705_478719Z_clpu_public_mechanism_pilot'
SOURCE_SHA='d9fa4705c494bb2a56f2076ca524d6c2234e91c1658796bdd71b2a2db88c62b8'
FAULT_IDS=('N2__line.l15__line.l23','N2__line.l4__line.l14','N2__line.l19__line.l21','N2__line.l17__line.l22')
EVENT_IDS=(22,7,2,26,10,5)
FAULT_SOURCE='runs/20261019T051045_857703Z_nature34_migration/fault_partitions.json'
FAULT_SHA='456e5d0fd63a71128c98db43e3d6c5fa7cf219cbe77e4b199bd31d03ba0e3660'
EVENT_SOURCE='runs/20261024T021749_253682Z_clpu_curve_import'
EVENT_MANIFEST_SHA='f3d752c036134a5bd64b470e4d6e6258fe5dc2e169ad7076f24ea00e8e8db5e4'
EVENT_FILE_SHA='810a392c0c7266384b45414d1f5ed5dfa8fbf7c874f43f6c49f706e93c2e7a40'
POPULATION_SOURCE='artifacts/public_search_freeze_design_20261031/population_manifest.json'
POPULATION_SHA='95fef70607cd3416f2b3daecfd820ad710dcdd607f5dd93c6d6c96be75eb95e2'
PHYSICAL_INPUTS=('author_static_metadata.json','load_reference.json','conditions.json','physical_config.json',
    'operating_reference.json','base_load_properties.json','command_inventory.json','source_pointer.json',
    'metadata.json','adaptation.json','author_input_precheck.json','condition.json','source_provenance.json','event_split.json')
ADDITIONS={PROTOCOL,'scripts/run_clpu_public_confirmation.py','scripts/audit_clpu_public_confirmation.py',
    'scripts/report_clpu_public_confirmation.py','scripts/report_clpu_public_mechanism_pilot.py',
    'scripts/report_clpu_shared_pilot.py','scripts/report_clpu_staged_pilot.py','scripts/report_clpu_integrated_pilot.py',
    'tests/test_run_clpu_public_confirmation.py','tests/test_audit_clpu_public_confirmation.py',
    'tests/test_report_clpu_public_confirmation.py'}
expected_phase_spec=staged.expected_phase_spec
check_staged_records=energy.check_staged_records


def phase_order(arm):
    require(arm in ARMS,'Unknown confirmation arm')
    return ['EARLY','GUARD'] if arm=='E' else ['GUARD','EARLY']


def expected_contexts():
    return [{'context_id':i*6+j,'context_key':f'{fault}|event={event}','scenario_id':fault,'event_id':event,
        'fault_partition':'formal_test','event_partition':'holdout'}
        for i,fault in enumerate(FAULT_IDS) for j,event in enumerate(EVENT_IDS)]


def expected_schedule(contexts):
    require(contexts==expected_contexts(),'Fixed outcome-free formal/holdout Cartesian identities changed')
    ordered=sorted(contexts,key=lambda c:prior.digest_text('public-confirmation-context-v1|'+prior.canonical(c)))
    rows=[]
    for rank,c in enumerate(ordered):
        for position,arm in enumerate(ARMS if rank%2==0 else tuple(reversed(ARMS))):
            rows.append({**c,'cell':f'episodes/{len(rows):03d}','episode_id':len(rows),'pair_order':rank,
                'context_order':rank,'arm_position':position,'arm_launch_order':position,'dispatch_round':0,
                'arm_slot':arm,'arm':arm,'route_arm':arm,'phase_order':phase_order(arm),
                'external_floor_enabled':arm=='P','free_external_floor_pruning_enabled':arm=='P',
                'mode':'shared','method':'beam','seed':0,'macro_schedule':'staged','phase':'formal_confirmation',
                'group':arm,'protected':True,'representation':'raw','deterministic':True,'checkpoint':None,
                'checkpoint_sha256':None,'point_budget':256,'logical_point_budget':256,'physical_point_budget':512,
                'total_logical_point_budget':512,'window_variant':'staged_common_pool_coverage_v1'})
    return rows


def protocol_settings():
    return {'study':STUDY,'contexts':24,'cells':48,'slots':list(ARMS),'models_used':0,'new_training':0,
        'execution':'single_stage_all48','max_workers':2,'worker_seconds':900,'controller_seconds':3600,
        'logical_point_budget_per_view':256,'physical_point_budget_per_cell':512,
        'total_physical_budget':24576,'maximum_numerical_entries':98448,
        'fault_partition':'formal_test','event_partition':'holdout','quality_gate':False,
        'dispatch':'public-confirmation-context-v1 hashed contexts; adjacent P/E pair; alternate first arm',
        'cross_cell_cache':False,'historical_answers_loaded':False,'automatic_retry':False}


def expected_source_binding():
    return {'source_run':SOURCE_RUN,'source_completion_sha256':SOURCE_SHA,
        'role':'physical inputs and immutable public algorithms only; no historical outcome or timing comparator',
        'fault_source':FAULT_SOURCE,'fault_sha256':FAULT_SHA,'event_source':EVENT_SOURCE,
        'event_manifest_sha256':EVENT_MANIFEST_SHA,'event_file_sha256':EVENT_FILE_SHA,
        'models_used':0,'historical_answers_loaded':False}


def expected_source_closure(inherited):
    names=set(inherited)|ADDITIONS;pending=list(names);visited=set()
    while pending:
        name=pending.pop()
        if name in visited or not name.endswith('.py') or not (ROOT/name).is_file():continue
        visited.add(name);package=name.removesuffix('.py').split('/')[1:-1];modules=[]
        for node in ast.walk(ast.parse((ROOT/name).read_text(encoding='utf-8-sig'))):
            if isinstance(node,ast.Import):modules.extend(alias.name for alias in node.names)
            elif isinstance(node,ast.ImportFrom):
                module='.'.join(package[:len(package)-node.level+1]+([node.module] if node.module else [])) if node.level else node.module or ''
                modules.extend([module,*[module+'.'+alias.name for alias in node.names]])
        for module in modules:
            for parent in ('scripts','src'):
                candidate=parent+'/'+module.replace('.','/')+'.py'
                if (ROOT/candidate).is_file() and candidate not in names:names.add(candidate);pending.append(candidate)
    return sorted(names)


# Explicit copy: only the obsolete historical-development GUARD comparison
# is removed. Current GUARD is replayed in full then paired between P and E.
def _check_base_cell(run, spec, cfg, scope):
    cell = (run / spec['cell']).resolve()
    require(cell.is_relative_to(run) and cell != run, 'Escaped cold task')
    sealed = shared.checked_manifest(cell)
    required = {'spec.json', 'result.json', 'worker_started.json', 'worker_finished.json', 'execution.json',
        'service_events.jsonl', 'service_summary.json', 'cost_checkpoints.jsonl', 'physical/points.jsonl.gz',
        'physical/requests.jsonl', 'physical/oracle_summary.json', 'physical/command_ledger.jsonl'}
    for name in shared.VIEWS:
        required |= {name + '/' + p for p in ('spec.json', 'phase_result.json', 'initial_state.json', 'plans.jsonl.gz',
            'decisions.jsonl.gz', 'logical_points.jsonl.gz', 'logical_requests.jsonl', 'view_summary.json')}
        require(not (cell / name / 'points.jsonl.gz').exists() and not (cell / name / 'command_ledger.jsonl').exists(),
                'A logical view fabricates paid physical evidence')
    require(required <= set(sealed) and not (cell / 'error.json').exists(), 'Incomplete/failed task evidence')
    result = read(cell / 'result.json')
    require(read(cell / 'spec.json') == result['spec'] == spec and result['new_training'] == 0,
            'Task identity or unauthorized training changed')
    require(result['version'] == result['study'] == STUDY and result['schema'] == 'clpu-staged-protected-result-v1',
            'Deferred scientific output version differs')
    require(result['checkpoint_sha256'] == spec['checkpoint_sha256'], 'Fixed checkpoint differs')
    policy = early.IndependentPolicy(run / spec['checkpoint'], spec) if spec['checkpoint'] else None
    worker, finished, execution = [read(cell / p) for p in ('worker_started.json', 'worker_finished.json', 'execution.json')]
    require(type(worker['pid']) is int and worker['pid'] > 0 and finished['pid'] == worker['pid']
            and finished['status'] == 'COMPLETE' and finished['result_sha256'] == sha(cell / 'result.json'),
            'Cold worker completion unbound')
    global_execution = read(run / 'execution_started.json')
    require(execution['spec_sha256'] == sha(cell / 'spec.json') and execution['controller_pid'] == read(run / ('controller_stage1_started.json' if execution['stage'] == 'phase1' else 'controller_stage2_started.json'))['pid']
            and execution['deadline_epoch_seconds'] == min(global_execution['deadline_epoch_seconds'],
                                                           execution['started_epoch_seconds'] + 900), 'Wrong worker deadline')
    scenarios = {r['scenario_id']: r for r in read(run / 'selected_scenarios.json')}
    events = {r['event_id']: r for r in read(run / 'selected_events.json')}
    mech = branch.Mechanics(read(run / 'author_static_metadata.json'), read(run / 'load_reference.json'),
        scenarios[spec['scenario_id']], read(run / 'condition.json'), events[spec['event_id']], cfg)
    points, statuses = shared.check_actual_points(run, cell / 'physical', spec, mech)
    commands = old.check_commands(cell / 'physical', points, read(run / 'command_inventory.json'))
    shared.check_actual_requests(list(read_lines(cell / 'physical/requests.jsonl')), points,
        read(cell / 'physical/oracle_summary.json'), commands, statuses)
    views = {name: {'logical_rows': list(read_lines(cell / name / 'logical_points.jsonl.gz')),
                   'requests': list(read_lines(cell / name / 'logical_requests.jsonl'))} for name in shared.VIEWS}
    evidence, _ = shared.check_service_ledger(list(read_lines(cell / 'service_events.jsonl')), views, points,
        mech, scope, sha(run / 'condition.json'), 'shared')
    service = read(cell / 'service_summary.json')
    shared.check_service_summaries(service, {name: read(cell / name / 'view_summary.json') for name in shared.VIEWS},
                                  evidence, scope, 'shared', commands['numerical_entries'])
    require(result['service'] == service, 'Complete service accounting differs')
    phases, checks = {}, {}
    for name in shared.VIEWS:
        directory = cell / name
        p = read(directory / 'phase_result.json')
        require(read(directory / 'spec.json') == p['spec'] == expected_phase_spec(spec, name)
                and read(directory / 'initial_state.json') == mech.initial, 'Search state or identity changed')
        require(p['oracle'] == service['views'][name], 'View summary differs from physical service')
        plans, decisions = list(read_lines(directory / 'plans.jsonl.gz')), list(read_lines(directory / 'decisions.jsonl.gz'))
        require(p['attempts'] == plans, 'Omitted/relabelled attempted plan')
        if name == 'EARLY':
            checked = check_staged_records(mech, cfg['branch_search'], p['spec'], p, plans, decisions,
                                            evidence[name], sha(run / 'condition.json'), policy)
        else:
            checked = shared.check_phase_records(mech, cfg['branch_search'], name, p['spec'], p, plans,
                decisions, evidence[name], sha(run / 'condition.json'))
            require(p['verify_attempts'] is None and p['graph_expansions'] == checked['decisions'],
                    'Original Beam combined steps confused with atomic VERIFY')
        phases[name], checks[name] = p, checked
    # Only version labels are projected to the frozen pure protection checker;
    # all actual results and the new version were validated immediately above.
    chosen = shared.check_selection({**result, 'version': shared.STUDY, 'study': shared.STUDY,
                                     'schema': 'clpu-shared-protected-result-v1'}, phases)
    cps = list(read_lines(cell / 'cost_checkpoints.jsonl'))
    require(result['checkpoints'] == cps, 'Top quality/cost curve omits boundaries')
    shared.check_checkpoints(cps, checks, phases)
    for name in shared.VIEWS:
        require(result['phase_summary'][name] == {k: phases[name][k] for k in ('quality_pct', 'fully_certified',
            'logical_points', 'actual_physical_points', 'search_decisions', 'policy_decisions', 'graph_expansions',
            'verify_attempts', 'phase_elapsed_seconds', 'stage_component_seconds')}, 'Phase summary incomplete')
    require(set(result['phase_summary']) == set(shared.VIEWS), 'Extra/missing search phases')
    for key, expected in {'logical_points': service['logical_points'], 'actual_physical_points': len(points),
        'numerical_entries': commands['numerical_entries'], 'requests': sum(e['requests'] for e in evidence.values()),
        'search_decisions': sum(p['search_decisions'] for p in phases.values()),
        'policy_decisions': checks['EARLY']['policy_decisions']}.items():
        require(result[key] == expected, 'Task cost/participation differs: ' + key)
    shared.check_times(result, phases, finished, math.fsum(p['elapsed_seconds'] for p in points.values()))
    model_time = window.finite_time(result['model_load_seconds'], 'model loading cost')
    require(model_time <= result['preflight_elapsed_seconds'] and (spec['checkpoint'] is not None or model_time == 0.),
            'Model work hidden from cold end-to-end time')
    if spec['checkpoint']:
        require(sha(run / spec['checkpoint']) == spec['checkpoint_sha256'], 'Frozen model changed during audit')
    return {**{k: spec[k] for k in ('cell', 'context_id', 'scenario_id', 'event_id', 'mode', 'arm', 'seed',
        'pair_order', 'arm_slot', 'macro_schedule')}, 'status': 'PASS', 'result_sha256': sha(cell / 'result.json'),
        'cell_completion_sha256': sha(cell / 'completion_manifest.json'), 'quality_pct': result['quality_pct'],
        'guard_established': chosen['guard_established'], 'guard_quality_pct': result['baseline_quality_pct'],
        'gain_over_guard_kwh': chosen['gain_over_guard_kwh'], 'logical_points': service['logical_points'],
        'physical_points': len(points), 'numerical_entries': commands['numerical_entries'], 'requests': result['requests'],
        'search_decisions': result['search_decisions'], 'policy_decisions': result['policy_decisions'],
        'worker_elapsed_seconds': result['worker_elapsed_seconds'], 'model_load_seconds': model_time,
        'worker_finished_elapsed_seconds': finished['worker_elapsed_seconds'], 'worker_pid': worker['pid'],
        'worker_parent_pid': worker['parent_pid'], 'execution': execution, 'physical_statuses': dict(statuses),
        'history_verified': False, 'guard_replay_verified': True, 'quality_checkpoints_verified': True,
        'cross_view_reuses': sum(e['cross_view_reuses'] for e in evidence.values()),
        'first_verify_state_sha256': checks['EARLY']['first_verify_state_sha256'],
        'actual_and_independent_policy_actions_equal': True,
        'phase_summary': {name: {**result['phase_summary'][name], 'requests': evidence[name]['requests'],
            'semantic_signature_sha256': checks[name]['semantic_signature_sha256'],
            'phase_result_sha256': sha(cell / name / 'phase_result.json'),
            **({k: checks[name][k] for k in ('total_decisions', 'handed_off', 'compact_hash_exact_matches', 'raw_hash_exact_matches')}
               if name == 'EARLY' else {})} for name in shared.VIEWS}, 'exact_checkpoints': cps}



def _check_staged_cell(run, spec, cfg, scope):
    checked = _check_base_cell(run, spec, cfg, scope)
    cell = run / spec['cell']
    result = read(cell / 'result.json')
    phases = {p: read(cell / p / 'phase_result.json') for p in shared.VIEWS}
    generation = phases['EARLY']['stage_generation']
    require(generation is not None, 'Successful staged cell has no complete paid bootstrap/generation')
    pool = read(cell / 'EARLY/candidate_pool.json')
    require(pool == {'generation': generation, 'generation_sha256': digest(generation)}, 'Cold candidate pool evidence differs')
    require('EARLY/candidate_pool.json' in shared.checked_manifest(cell), 'Pool was not sealed with the cell')
    # Independent StagedReplay rebuilds the pool from this task's actual frozen
    # inputs; an old development graph profile is not a new-context reference.
    require(result['stage_generation_sha256'] == digest(generation) and
        result['stage_candidate_count'] == len(generation['candidates']) and
        result['stage_generation_seconds'] == phases['EARLY']['stage_generation_seconds'], 'Generation count/cost hidden')
    energies = {}
    for name, phase in phases.items():
        best = phase['best_plan']
        require(phase['fully_certified'] is (best is not None), 'Complete phase quality status differs')
        if best is not None:
            require(best['certified_complete'] and best['validation']['status'] == 'COMPLETE', 'Uncertified phase incumbent')
        energies[name] = best['validation']['energy_kwh'] if best is not None else None
    checked.update(early_energy_kwh=energies['EARLY'], early_fully_certified=phases['EARLY']['fully_certified'],
        guard_energy_kwh=energies['GUARD'], candidate_pool_sha256=digest(generation),
        candidate_count=len(generation['candidates']), stage_generation_seconds=result['stage_generation_seconds'])
    return checked



# Private function namespaces avoid mutating original auditor module globals.
_staged_facade=SimpleNamespace(**{**vars(staged),'_check_base_cell':_check_base_cell,
    'check_cell':_check_staged_cell})
_p_cell=private(gf.check_cell,STUDY=STUDY,SCHEMA=SCHEMA,staged=_staged_facade)
_e_cell=private(energy.check_cell,STUDY=STUDY,SCHEMA=SCHEMA,staged=_staged_facade)
_public_cell=private(public.check_cell,ARMS=ARMS,phase_order=phase_order,_p_cell=_p_cell,_e_cell=_e_cell)


def check_cell(run,spec,cfg,scope):
    result=read(run/spec['cell']/'result.json')
    require(spec['phase']=='formal_confirmation' and spec['fault_partition']=='formal_test'
        and spec['event_partition']=='holdout' and result['formal_holdout_numerical_access'] is True
        and result['tune_numerical_access'] is False
        and result['candidate_generation_role']=='actual cold task, not preobserved inventory',
        'Actual worker is not the explicitly authorized new-population route')
    return _public_cell(run,spec,cfg,scope)


def context_equalities(checks):
    require(len(checks)==CELLS and {c['context_id'] for c in checks}==set(range(CONTEXTS)),
        'Incomplete fixed confirmation population')
    results=[]
    for context in range(CONTEXTS):
        group=[r for r in checks if r['context_id']==context]
        require(len(group)==2 and {r['arm'] for r in group}==set(ARMS),'Missing/duplicate confirmation arm')
        require(len({(r['scenario_id'],r['event_id']) for r in group})==1,'Context pair has different identities')
        require(all(r['guard_established'] and r['guard_replay_verified'] and r['history_verified'] is False
            and r['models_used']==0 for r in group),'Missing actual GUARD replay or invented historical verification')
        require(len({r['candidate_pool_sha256'] for r in group})==1,'Initial complete candidate pools differ')
        require(len({r['phase_summary']['GUARD']['semantic_signature_sha256'] for r in group})==1
            and len({r['guard_quality_pct'] for r in group})==1,'Actual original GUARD trajectories differ')
        results.append({'context_id':context,'scenario_id':group[0]['scenario_id'],'event_id':group[0]['event_id'],
            'status':'PASS','tasks':2,'guard_trajectories_equal':True,'guard_replay_verified':True,
            'history_verified':False,'candidate_pool_sha256':group[0]['candidate_pool_sha256'],
            'guard_quality_pct':group[0]['guard_quality_pct'],'equal_initial_history_or_exposure_claimed':False})
    return results


def check_controller(run,schedule,checks):
    require(len(schedule)==len(checks)==CELLS,'All48 actual tasks required')
    require(not any((run/name).exists() for name in ('phase_gate.json','audit_stage1','controller_stage2_started.json')),
        'Unapproved conditional population or second stage')
    execution=read(run/'controller_stage1_completion.json');controller=read(run/'controller_completion.json')
    require(execution['exits']==read(run/'controller_stage1_exits.json'),'True exit journal changed')
    dispatch=list(read_lines(run/'controller_dispatch.jsonl'))
    require(dispatch==list(read_lines(run/'controller_stage1_dispatch.jsonl')),'Dispatch journal differs')
    names=[s['cell'] for s in schedule]
    require(len(set(names))==CELLS and {c['cell'] for c in checks}==set(names),'Duplicate or omitted task')
    require(execution['status']=='COMPLETE_STAGE1_PENDING_AUDIT' and execution['stage']=='phase1'
        and execution['scheduled']==execution['scheduled_total']==CELLS and execution['selected_cells']==names
        and execution['dispatched']==schedule and len(execution['completed'])==CELLS and set(execution['completed'])==set(names)
        and execution['errors']==execution['unknown_cost_cells']==[] and execution['peak_active']<=2,'Actual controller did not close all tasks')
    require([d['cell'] for d in dispatch]==names and len(execution['exits'])==CELLS,'Retry/omission/order changed')
    exits={e['cell']:e for e in execution['exits']};by_cell={c['cell']:c for c in checks}
    require(set(exits)==set(names),'Exit identity mismatch')
    previous_end=0.
    for start in range(0,CELLS,2):
        ends=[]
        for spec,d in zip(schedule[start:start+2],dispatch[start:start+2]):
            e,c=exits[spec['cell']],by_cell[spec['cell']]
            require(d['stage']==c['execution']['stage']=='phase1' and e['exit_code']==0
                and e['drained_after_controller_error'] is False and e['launcher_pid']==d['launcher_pid']
                and e['pair_order']==d['pair_order']==spec['pair_order'],'Wrong route stage/failed exit or retry')
            begin=window.finite_time(e['started_epoch_seconds'],'process start');end=window.finite_time(e['finished_epoch_seconds'],'process exit')
            elapsed=window.finite_time(e['process_wall_seconds'],'full process time')
            require(begin==d['started_epoch_seconds']==c['execution']['started_epoch_seconds'] and previous_end<=begin<=end
                and elapsed>=c['worker_finished_elapsed_seconds'],'Earlier pair not drained or paid time omitted')
            require(c['worker_parent_pid'] in (d['launcher_pid'],c['execution']['controller_pid'])
                and abs(end-begin-elapsed)<=2.,'Process ancestry or wall clocks differ')
            c['process_wall_seconds']=elapsed;ends.append(end)
        previous_end=max(ends)
    totals={k:sum(c[k] for c in checks) for k in ('logical_points','physical_points','numerical_entries','requests')}
    require(all(execution[k]==v for k,v in totals.items()) and totals['physical_points']<=TOTAL_CAP
        and totals['logical_points']<=TOTAL_CAP and totals['numerical_entries']==CELLS*3+4*totals['physical_points'],'Full cold cost does not close')
    require(controller['status']=='COMPLETE_CELLS_PENDING_AUDIT' and controller['scheduled']==controller['resolved_slots']==controller['actual_tasks']==CELLS
        and controller['not_run']==[] and controller['all_fixed_context_mean_allowed'] is True
        and controller['errors']==controller['unknown_cost_cells']==[],'Top full-population completion changed')
    require(controller['controller_execution_sha256']==sha(run/'controller_stage1_completion.json'),'Completion not bound to true process records')
    for key in ('completed','dispatched','exits','logical_points','physical_points','numerical_entries','requests'):
        require(controller[key]==execution[key],'Whole-run execution cost altered: '+key)
    require(window.finite_time(controller['wall_seconds'],'controller wall')>=window.finite_time(execution['wall_seconds'],'stage wall')
        >=max(c['process_wall_seconds'] for c in checks),'Controller timer omits actual processes')
    return totals


def check_selected_inputs(faults,events,selected_scenarios,selected_events,event_split):
    """Exact original row projection; called only after design freeze/hydration."""
    require(len({r['scenario_id'] for r in faults})==len(faults),'Duplicate original fault ID')
    expected_faults=[next(r for r in faults if r['scenario_id']==sid) for sid in FAULT_IDS]
    require(selected_scenarios==expected_faults and all(r['partition']=='formal_test' and r['fault_order']==2
        for r in selected_scenarios),'Formal source rows changed, relabelled or taken from training')
    require(len(events)==31 and {r['event_id'] for r in events}==set(range(1,32)),'Original full event import identity differs')
    expected_events=[next(r for r in events if r['event_id']==eid) for eid in EVENT_IDS]
    require(selected_events==expected_events and all(r['validation']['passed'] is True for r in selected_events),
        'Selected holdout original parameters/validation changed')
    require(set(event_split)=={'train','dev','holdout'} and set(event_split['holdout'])==set(EVENT_IDS)
        and not (set(EVENT_IDS)&(set(event_split['train'])|set(event_split['dev']))),
        'Requested events are not the sealed holdout partition')


def check_frozen_physics(run,prepared,cfg,source_manifest):
    """New population contract; never invoke the old train-only checker."""
    source=ROOT/'runs'/SOURCE_RUN
    oldcfg=read(source/'config.json')
    for key in ('condition_id','alpha','algorithm','horizon','slot_seconds','sample_seconds','branch_search'):
        require(cfg[key]==oldcfg[key],'Frozen physical/search setting changed: '+key)
    for name in PHYSICAL_INPUTS:
        require(sha(run/name)==source_manifest[name]==sha(source/name),'Inherited physical input changed: '+name)
    for name in ('source_inputs/metadata.json','source_inputs/base_load_properties.json'):
        require(sha(run/name)==source_manifest[name]==sha(source/name),'Original native solver source changed: '+name)
    require(not any((run/name).exists() for name in ('training_scenarios.json','training_contexts.json',
        'historical_bindings.json','historical','models')),'Training/history population disguised as formal scope')
    require(sha(run/'fault_partitions.json')==sha(ROOT/FAULT_SOURCE)==FAULT_SHA,'Formal fault source changed')
    require(sha(run/'event_source_manifest.json')==sha(ROOT/EVENT_SOURCE/'manifest.json')==EVENT_MANIFEST_SHA,
        'Original event import attribution changed')
    event_manifest=read(run/'event_source_manifest.json')
    require(event_manifest['status']=='COMPLETE' and event_manifest['output_sha256']['events.json']==EVENT_FILE_SHA,
        'Original event source is incomplete/unbound')
    # Deliberately occurs only after check_parent has verified the first freeze.
    require(sha(ROOT/EVENT_SOURCE/'events.json')==EVENT_FILE_SHA,'Original event bytes differ from predeclared source')
    faults=read(run/'fault_partitions.json')['scenarios'];events=read(ROOT/EVENT_SOURCE/'events.json')
    selected=read(run/'selected_events.json')
    check_selected_inputs(faults,events,read(run/'selected_scenarios.json'),selected,read(run/'event_split.json'))
    hydrated=read(run/'hydration.json');started=read(run/'hydration_started.json')
    require(hydrated['method_freeze_sha256']==started['method_freeze_sha256']==sha(run/'method_freeze_manifest.json')
        and started['expected_sha256']==hydrated['source_file_sha256']==EVENT_FILE_SHA
        and Path(started['source']).resolve()==(ROOT/EVENT_SOURCE/'events.json').resolve()
        and hydrated['selected_event_ids']==list(EVENT_IDS) and hydrated['selected_events_sha256']==sha(run/'selected_events.json')
        and hydrated['row_sha256']=={str(r['event_id']):digest(r) for r in selected}
        and hydrated['original_values_unchanged'] is True and hydrated['new_physical_points']==hydrated['new_training']==0,
        'Original parameter hydration is unbound, changed or uses numerical responses')
    scope=read(run/'physical_scope.json')
    needed=set(PHYSICAL_INPUTS)|{'config.json','selected_scenarios.json','selected_events.json','confirmation_contexts.json',
        'fault_partitions.json','event_source_manifest.json','hydration.json','dependencies.json','python_environment.json',
        'source_inputs/metadata.json','source_inputs/base_load_properties.json'}
    require(scope['scope_version']=='CLPU_FORMAL_HOLDOUT_FIXED_PHYSICAL_SCOPE_V1'
        and scope['input_files']=={name:sha(run/name) for name in needed},'Actual formal/holdout physical input scope differs')
    require(scope['science_source_sha256']=={name:sha(ROOT/name) for name in prepared['source_files'] if name.endswith('.py')},
        'Complete implementation scope differs')
    expected={'source_provenance':read(run/'source_provenance.json'),'runtime_dependencies':read(run/'dependencies.json'),
        'python_environment':read(run/'python_environment.json'),'physical_config':read(run/'physical_config.json'),
        'alpha':.5,'algorithm':0,'load_setpoint_precision':'.15g',
        'reset_semantics':'INDEPENDENT_QUASI_STATIC_AUTHOR_RESET_EACH_SNAPSHOT',
        'cache_scope':'one cold cell only; no historical or cross-arm/cross-context answers',
        'fault_partition':'formal_test','event_partition':'holdout','new_training':0}
    require(set(scope)=={'scope_version','input_files','science_source_sha256'}|set(expected),'Unknown/incomplete physical scope')
    for key,value in expected.items():require(scope[key]==value,'Physical scope changed: '+key)
    from nature34_experiment_support import dependencies
    require(dependencies()==scope['runtime_dependencies'] and sha(Path(sys.executable))==scope['python_environment']['sha256'],
        'Physical runtime environment differs from freeze')
    binding=read(run/'scope_review_binding.json')
    require(binding['path']=='artifacts/clpu_equivalent_cache_scope_review_20261030'
        and sha(run/'scope_review/completion_manifest.json')==binding['completion_sha256']
        and sha(run/'scope_review/summary.json')==binding['summary_sha256'],'Historical scope mechanism binding changed')
    shared.checked_manifest(run/'scope_review');review=read(run/'scope_review/summary.json')
    require(review['status']=='PASS' and review['new_physical_points']==review['new_training']==0
        and review['physical_points_reviewed']==1121 and review['duplicate_pairs_reviewed']==538
        and review['certificate_or_feedback_disagreements']==0 and review['constraint_or_audit_tolerances_changed'] is False,
        'Historical scope mechanism review incomplete')
    prior.cache.check_exact_key_review(run,binding['completion_sha256'])
    return scope


def check_parent(run):
    run=Path(run).resolve();require(run.parent==ROOT/'runs','Confirmation outside project runs')
    prepared=read(run/'prepared.json');design=read(run/'design_prepared.json')
    require(prepared['status']=='FROZEN_PREPARED_NO_DSS'
        and prepared['freeze_sha256']==sha(run/'prephysical_freeze_manifest.json')
        and prepared['method_freeze_sha256']==sha(run/'method_freeze_manifest.json'),'Two-stage freeze binding changed')
    method=shared.checked_manifest(run,'method_freeze_manifest.json')
    frozen=shared.checked_manifest(run,'prephysical_freeze_manifest.json')
    require(set(method)<=set(frozen) and all(frozen[k]==v for k,v in method.items())
        and frozen['method_freeze_manifest.json']==prepared['method_freeze_sha256'],'Method changed after new values were opened')
    require(not any(n in method for n in ('selected_events.json','hydration.json','hydration_started.json','physical_scope.json',
        'prepared.json','execution_started.json')),'Method freeze includes post-hydration values or results')
    require({'config.json','contexts.json','schedule.json','source_binding.json','software_tests.xml','design_access.json',
        'selected_scenarios.json','population_source.json','design_prepared.json','authorization.json'}<=set(method),
        'Design freeze lacks required identity/software/authorization evidence')
    source=ROOT/'runs'/SOURCE_RUN
    require(sha(source/'completion_manifest.json')==SOURCE_SHA,'Original public source root changed')
    source_manifest=energy.normalized_manifest(read(source/'completion_manifest.json'))
    for name in ('prepared.json','config.json','audit/summary.json'):
        require(sha(source/name)==source_manifest[name],'Original public source input changed: '+name)
    oldaudit=read(source/'audit/summary.json')
    require(oldaudit['status']=='PASS' and oldaudit['audited_cells']==24 and not oldaudit['errors'],'Original public evidence not audited')
    require(read(run/'source_binding.json')==expected_source_binding(),'Unknown or substituted population source')
    inherited=set(read(source/'prepared.json')['source_files'])
    require(design['source_files']==prepared['source_files']==expected_source_closure(inherited),'Frozen implementation closure incomplete')
    for name in prepared['source_files']:
        target=(ROOT/name).resolve()
        require(target.is_relative_to(ROOT) and target!=ROOT and 'source/'+name in method
            and sha(target)==sha(run/'source'/name),'Frozen source changed: '+name)
        if name in inherited:require(sha(target)==source_manifest['source/'+name],'Previously sealed science changed: '+name)
    require(sha(run/'source'/PROTOCOL)==sha(ROOT/PROTOCOL),'New protocol differs from method freeze')
    require(design['status']=='METHOD_IDENTITIES_STATISTICS_FROZEN_BEFORE_EVENT_VALUES'
        and design['new_physical_points']==design['new_training']==0,'Design is not a pre-value software freeze')
    require(read(run/'design_access.json')=={'new_physical_points':0,'new_training':0,'holdout_event_table_opened':False,
        'formal_fault_identity_metadata_read':True,'event_expected_digest_from_existing_manifest_only':True},'First-stage access declaration changed')
    require(sha(run/'population_source.json')==sha(ROOT/POPULATION_SOURCE)==POPULATION_SHA,'Outcome-free population source changed')
    population=read(run/'population_source.json')
    require(population['selected_formal_n2'][:4]==list(FAULT_IDS) and population['all_holdout_events']==list(EVENT_IDS)
        and population['selection_uses_graph_or_physical_outcomes'] is False,'Not the preselected ID-only Cartesian population')
    contexts=expected_contexts();schedule=expected_schedule(contexts)
    require(read(run/'contexts.json')==read(run/'confirmation_contexts.json')==contexts
        and read(run/'schedule.json')==schedule,'Fixed48 full identity/order differs')
    require(read(run/'model_bindings.json')==[],'No-model confirmation contains a model')
    require(read(run/'stage_plan.json')=={'schema':'clpu-public-confirmation-population-v1','execution':'single_stage_all48',
        'contexts':contexts,'all_cells':[s['cell'] for s in schedule],'quality_gate':False,
        'candidate_generation_role':'generated cold by each actual task; independently replayed after execution'},
        'Population secretly gated by graph result or historical outcome')
    cfg=read(run/'config.json')
    require(set(cfg)=={'condition_id','alpha','algorithm','horizon','slot_seconds','sample_seconds','branch_search','study',
        'execution_allowed','training_enabled','formal_holdout_numerical_access','tune_numerical_access','methods',
        'confirmation','confirmation_statistics'},'Undeclared confirmation configuration')
    require(cfg['study']==STUDY and cfg['confirmation']==protocol_settings() and cfg['methods']==list(ARMS)
        and cfg['execution_allowed'] is True and cfg['training_enabled'] is False
        and cfg['formal_holdout_numerical_access'] is True and cfg['tune_numerical_access'] is False,
        'New formal/holdout authorization is missing or learning/tuning was enabled')
    for spec in schedule:
        require(read(run/spec['cell']/'spec.json')==spec,'Frozen cold spec differs')
        for phase in phase_order(spec['arm']):
            require(read(run/spec['cell']/phase/'spec.json')==expected_phase_spec(spec,phase),'Frozen logical phase spec differs')
    cases=list(ET.parse(run/'software_tests.xml').getroot().iter('testcase'))
    require(cases and not any(list(c.iter('failure'))+list(c.iter('error'))+list(c.iter('skipped')) for c in cases),
        'Passing complete component test evidence missing')
    for part in ('test_run_clpu_public_confirmation','test_audit_clpu_public_confirmation','test_report_clpu_public_confirmation'):
        require(any(part in c.get('classname','') for c in cases),'Missing confirmation component: '+part)
    rate=649788.8131868131;required=14*2**30+math.ceil(TOTAL_CAP*rate*1.25)
    for name in ('resource_budget.json','prephysical_capacity.json'):
        value=read(run/name)
        require(value['historical_max_bytes_per_actual_point']==rate and value['remaining_cells']==CELLS
            and value['point_cap']==TOTAL_CAP and value['numerical_entry_cap']==98448
            and value['required_free_bytes']==required and value['free_bytes']>=required,'Incomplete physical storage forecast')
    require(read(run/'authorization.json')=={'user_continuation_received':True,
        'scope':'4 formal N2 combinations x all six holdout events x P/E','new_training':0,'external_X_included':False,
        'historical_answers_loaded':False},'Missing continuation scope or changed comparison')
    scope=check_frozen_physics(run,prepared,cfg,source_manifest)
    return cfg,scope,schedule


AUDIT_FILES={'freeze_sha256':'prephysical_freeze_manifest.json','method_freeze_sha256':'method_freeze_manifest.json',
    'config_sha256':'config.json','schedule_sha256':'schedule.json','contexts_sha256':'contexts.json',
    'source_binding_sha256':'source_binding.json','physical_scope_file_sha256':'physical_scope.json',
    'scope_review_manifest_sha256':'scope_review/completion_manifest.json',
    'exact_key_review_manifest_sha256':'exact_key_review/completion_manifest.json',
    'model_bindings_sha256':'model_bindings.json','stage_plan_sha256':'stage_plan.json',
    'population_source_sha256':'population_source.json','hydration_sha256':'hydration.json',
    'selected_scenarios_sha256':'selected_scenarios.json','selected_events_sha256':'selected_events.json'}


def source_hashes(run,scope):
    return {**{key:sha(run/name) for key,name in AUDIT_FILES.items()},
        'source_hashes':{name:sha(run/name) for name in AUDIT_FILES.values()},'physical_scope_canonical_sha256':digest(scope),
        'source_public_completion_sha256':SOURCE_SHA,'audit_implementation_sha256':sha(Path(__file__)),
        'core_audit_sha256':sha(Path(staged.__file__)),'guard_first_audit_sha256':sha(Path(gf.__file__)),
        'ranking_audit_sha256':sha(Path(energy.__file__)),'public_audit_sha256':sha(Path(public.__file__))}


def audit(run,output=None):
    from audit_d059_supplement import NoSolverImports,no_solver_loaded
    run=Path(run).resolve();output=Path(output).resolve() if output else run/'audit'
    require(output.is_relative_to(ROOT) and output!=run and not output.exists(),'Preserve previous audit evidence')
    output.mkdir(parents=True);started=time.perf_counter();checks=[]
    result={'status':'FAIL','study':STUDY,'source_run':run.name,'scheduled_cells':CELLS,'errors':[],
        'audited_cells':0,'results':[],'new_audit_physical_points':0,'new_training_tasks':0,'models_used':0,
        'history_verified':False,'all_fixed_context_mean_allowed':False}
    guard=NoSolverImports();sys.meta_path.insert(0,guard)
    try:
        no_solver_loaded();cfg,scope,schedule=check_parent(run);result.update(source_hashes(run,scope))
        shared.checked_manifest(run/'precheck');precheck=read(run/'precheck/summary.json')
        require(precheck['status']=='PASS' and not precheck['errors'] and precheck['freeze_sha256']==sha(run/'prephysical_freeze_manifest.json'),
            'Complete independent precheck absent or bound to another freeze')
        result['precheck_summary_sha256']=sha(run/'precheck/summary.json')
        result['source_hashes']['precheck/summary.json']=result['precheck_summary_sha256']
        execution=read(run/'execution_started.json')
        require(execution['deadline_epoch_seconds']==execution['started_epoch_seconds']+3600
            and execution['execution']=='single_stage_all48','Global deadline or one-pass contract changed')
        for spec in schedule:
            checked=check_cell(run,spec,cfg,scope);checks.append(checked)
            save(output/(spec['cell'].replace('/','_')+'.json'),checked)
            save(output/'progress.json',{'status':'RUNNING','audited_cells':len(checks),'scheduled':CELLS,
                'last_cell':spec['cell'],'elapsed_seconds':time.perf_counter()-started,'new_physical_points':0})
        totals=check_controller(run,schedule,checks);equalities=context_equalities(checks);statuses=Counter()
        for row in checks:
            statuses.update(row['physical_statuses']);save(output/(row['cell'].replace('/','_')+'.json'),row)
        result.update(status='PASS',audited_cells=CELLS,resolved_slots=CELLS,actual_tasks=CELLS,not_run=[],results=checks,
            all_fixed_context_mean_allowed=True,complete_protected_cells=CELLS,physical_runtimes=CELLS,logical_searches=2*CELLS,
            context_equalities=equalities,context_checks=equalities,physical_statuses=dict(statuses),**totals,
            controller_completion_sha256=sha(run/'controller_completion.json'),
            controller_execution_sha256=sha(run/'controller_stage1_completion.json'),
            controller_wall_seconds=read(run/'controller_completion.json')['wall_seconds'])
        no_solver_loaded()
    except BaseException as exc:
        result.update(audited_cells=len(checks),results=checks)
        result['errors'].append({'error':repr(exc),'traceback':traceback.format_exc()})
    finally:sys.meta_path.remove(guard)
    result['elapsed_seconds']=time.perf_counter()-started;save(output/'summary.json',result)
    save(output/'completion_manifest.json',{p.relative_to(output).as_posix():sha(p)
        for p in output.rglob('*') if p.is_file() and p.name!='completion_manifest.json'})
    return result


def preflight(run,output):
    from audit_d059_supplement import NoSolverImports,no_solver_loaded
    run,output=Path(run).resolve(),Path(output).resolve()
    require(output.is_relative_to(ROOT) and not output.exists(),'Preserve earlier independent precheck')
    output.mkdir(parents=True);result={'status':'FAIL','study':STUDY,'errors':[],'new_physical_points':0,'new_training':0}
    guard=NoSolverImports();sys.meta_path.insert(0,guard)
    try:
        no_solver_loaded();_,scope,schedule=check_parent(run);no_solver_loaded()
        result.update(status='PASS',contexts=CONTEXTS,cells=len(schedule),physical_runtimes=CELLS,
            logical_searches=2*CELLS,models_used=0,formal_source_verified=True,holdout_source_verified=True,
            **source_hashes(run,scope))
    except BaseException as exc:result['errors'].append({'error':repr(exc),'traceback':traceback.format_exc()})
    finally:sys.meta_path.remove(guard)
    save(output/'summary.json',result);save(output/'completion_manifest.json',{'summary.json':sha(output/'summary.json')})
    return result


main=private(energy.main,audit=audit,preflight=preflight,__doc__=__doc__)
if __name__=='__main__':raise SystemExit(main())

