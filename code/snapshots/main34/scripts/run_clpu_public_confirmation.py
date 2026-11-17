"""Frozen 4 x 6 public P/E confirmation; prepare is identity-only, hydrate is explicit.

No old training guard is disabled.  New population checks surround the unchanged
public cold search loops and physical evidence services.  Importing is zero-PF.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import FunctionType
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
sys.dont_write_bytecode = True
import run_clpu_public_mechanism_pilot as public
from nature34_experiment_support import create_run, dependencies

read, save, require, sha, canonical = public.read, public.save, public.require, public.sha, public.canonical
seal, utc, cache, base = public.seal, public.utc, public.cache, public.base
phase_spec, run_routes = public.phase_spec, public.run_routes
STUDY = 'CLPU_PUBLIC_CONFIRMATION_COMPACT_V1'
SCHEMA = 'clpu-public-confirmation-protected-result-v1'
PROTOCOL = 'docs/public_confirmation_compact_protocol_v1.md'
ACTIVE = ROOT/'artifacts/clpu_public_confirmation_active.json'
SLOTS, CONTEXTS, CELLS = ('P', 'E'), 24, 48
MAX_SECONDS, WORKER_SECONDS, TOTAL_CAP = 3600, 900, 24576
MIN_FREE, RESERVE = 12*2**30, 2*2**30
BYTE_RATE = 649788.8131868131
FAULT_IDS = ('N2__line.l15__line.l23','N2__line.l4__line.l14',
             'N2__line.l19__line.l21','N2__line.l17__line.l22')
EVENT_IDS = (22,7,2,26,10,5)
PUBLIC_SOURCE = ROOT/'runs/20261031T141705_478719Z_clpu_public_mechanism_pilot'
PUBLIC_SHA = 'd9fa4705c494bb2a56f2076ca524d6c2234e91c1658796bdd71b2a2db88c62b8'
FAULT_SOURCE = ROOT/'runs/20261019T051045_857703Z_nature34_migration/fault_partitions.json'
FAULT_SHA = '456e5d0fd63a71128c98db43e3d6c5fa7cf219cbe77e4b199bd31d03ba0e3660'
EVENT_SOURCE = ROOT/'runs/20261024T021749_253682Z_clpu_curve_import'
EVENT_MANIFEST_SHA = 'f3d752c036134a5bd64b470e4d6e6258fe5dc2e169ad7076f24ea00e8e8db5e4'
EVENT_FILE_SHA = '810a392c0c7266384b45414d1f5ed5dfa8fbf7c874f43f6c49f706e93c2e7a40'
POPULATION_SOURCE = ROOT/'artifacts/public_search_freeze_design_20261031/population_manifest.json'
POPULATION_SHA = '95fef70607cd3416f2b3daecfd820ad710dcdd607f5dd93c6d6c96be75eb95e2'
PHYSICAL_INPUTS = tuple(n for n in base.ROOT_INPUTS if n != 'training_scenarios.json')
_WORKER_ENTERED = False


def contexts_for():
    return [{'context_id':i*6+j,'context_key':f'{f}|event={e}','scenario_id':f,'event_id':e,
             'fault_partition':'formal_test','event_partition':'holdout'}
            for i,f in enumerate(FAULT_IDS) for j,e in enumerate(EVENT_IDS)]


def schedule_for(contexts):
    require(contexts == contexts_for(), 'Fixed outcome-free Cartesian identities changed')
    ordered = sorted(contexts,key=lambda c:hashlib.sha256(
        ('public-confirmation-context-v1|'+canonical(c)).encode()).hexdigest())
    rows=[]
    for rank,c in enumerate(ordered):
        for position,arm in enumerate(SLOTS if rank%2==0 else tuple(reversed(SLOTS))):
            rows.append({**c,'cell':f'episodes/{len(rows):03d}','episode_id':len(rows),
                'pair_order':rank,'context_order':rank,'arm_position':position,'arm_launch_order':position,'dispatch_round':0,
                'arm_slot':arm,'arm':arm,'route_arm':arm,
                'phase_order':['GUARD','EARLY'] if arm=='P' else ['EARLY','GUARD'],
                'external_floor_enabled':arm=='P','free_external_floor_pruning_enabled':arm=='P',
                'mode':'shared','method':'beam','seed':0,'macro_schedule':'staged',
                'phase':'formal_confirmation','group':arm,'protected':True,'representation':'raw',
                'deterministic':True,'checkpoint':None,'checkpoint_sha256':None,
                'point_budget':256,'logical_point_budget':256,'physical_point_budget':512,
                'total_logical_point_budget':512,'window_variant':'staged_common_pool_coverage_v1'})
    return rows


def settings():
    return {'study':STUDY,'contexts':24,'cells':48,'slots':list(SLOTS),'models_used':0,'new_training':0,
        'execution':'single_stage_all48','max_workers':2,'worker_seconds':900,'controller_seconds':3600,
        'logical_point_budget_per_view':256,'physical_point_budget_per_cell':512,
        'total_physical_budget':24576,'maximum_numerical_entries':98448,
        'fault_partition':'formal_test','event_partition':'holdout','quality_gate':False,
        'dispatch':'public-confirmation-context-v1 hashed contexts; adjacent P/E pair; alternate first arm',
        'cross_cell_cache':False,'historical_answers_loaded':False,'automatic_retry':False}


def source_names():
    additions = [PROTOCOL,'scripts/run_clpu_public_confirmation.py',
        'scripts/audit_clpu_public_confirmation.py','scripts/report_clpu_public_confirmation.py',
        'scripts/report_clpu_public_mechanism_pilot.py','scripts/report_clpu_shared_pilot.py',
        'scripts/report_clpu_staged_pilot.py','scripts/report_clpu_integrated_pilot.py',
        'tests/test_run_clpu_public_confirmation.py','tests/test_audit_clpu_public_confirmation.py',
        'tests/test_report_clpu_public_confirmation.py']
    names=set(public.source_names()) | set(additions)
    # Freeze local Python imports transitively as well as the historical closure.
    # This parses source text only; it does not import a model or read event data.
    pending=list(names);visited=set()
    while pending:
        name=pending.pop()
        if name in visited or not name.endswith('.py') or not (ROOT/name).is_file():continue
        visited.add(name)
        package=name.removesuffix('.py').split('/')[1:-1]
        modules=[]
        for node in ast.walk(ast.parse((ROOT/name).read_text(encoding='utf-8-sig'))):
            if isinstance(node,ast.Import):modules.extend(alias.name for alias in node.names)
            elif isinstance(node,ast.ImportFrom):
                if node.level:
                    prefix=package[:len(package)-node.level+1]
                    module='.'.join(prefix+([node.module] if node.module else []))
                else:module=node.module or ''
                modules.append(module)
                modules.extend(module+'.'+alias.name for alias in node.names)
        for module in modules:
            for parent in ('scripts','src'):
                candidate=parent+'/'+module.replace('.','/')+'.py'
                if (ROOT/candidate).is_file() and candidate not in names:
                    names.add(candidate);pending.append(candidate)
    return sorted(names)


def source_binding():
    return {'source_run':PUBLIC_SOURCE.name,'source_completion_sha256':PUBLIC_SHA,
        'role':'physical inputs and immutable public algorithms only; no historical outcome or timing comparator',
        'fault_source':FAULT_SOURCE.relative_to(ROOT).as_posix(),'fault_sha256':FAULT_SHA,
        'event_source':EVENT_SOURCE.relative_to(ROOT).as_posix(),'event_manifest_sha256':EVENT_MANIFEST_SHA,
        'event_file_sha256':EVENT_FILE_SHA,'models_used':0,'historical_answers_loaded':False}


def check_tests(path):
    cases=list(ET.parse(path).getroot().iter('testcase'))
    require(cases and not any(list(c.iter('failure'))+list(c.iter('error'))+list(c.iter('skipped')) for c in cases),
            'All component tests must pass')
    for part in ('test_run_clpu_public_confirmation','test_audit_clpu_public_confirmation','test_report_clpu_public_confirmation'):
        require(any(part in c.get('classname','') for c in cases),'Missing confirmation component '+part)
    return len(cases)


def capacity(remaining=CELLS):
    import math
    required=MIN_FREE+RESERVE+math.ceil(remaining*512*BYTE_RATE*1.25)
    return {'free_bytes':shutil.disk_usage(ROOT).free,'required_free_bytes':required,
        'remaining_cells':remaining,'historical_max_bytes_per_actual_point':BYTE_RATE,
        'point_cap':remaining*512,'numerical_entry_cap':remaining*(3+4*512)}


def write_freeze(run,name):
    require(not (run/name).exists(),'Freeze cannot be overwritten')
    save(run/name,{p.relative_to(run).as_posix():sha(p) for p in sorted(run.rglob('*')) if p.is_file()})


def verify_design(run):
    require(run.is_absolute() and run.parent==ROOT/'runs','Wrong confirmation run location')
    cache.verify_manifest(run,'method_freeze_manifest.json',('config.json','contexts.json','schedule.json','source_binding.json'))
    cfg=read(run/'config.json')
    require(cfg['study']==STUDY and cfg['confirmation']==settings(),'Confirmation contract changed')
    require(cfg['formal_holdout_numerical_access'] is True and cfg['tune_numerical_access'] is False,
            'Explicit formal/holdout numerical scope required')
    require(read(run/'contexts.json')==contexts_for() and read(run/'schedule.json')==schedule_for(contexts_for()),'Population changed')
    require(read(run/'source_binding.json')==source_binding(),'Source attribution changed')
    for name in read(run/'design_prepared.json')['source_files']:
        require(sha(ROOT/name)==sha(run/'source'/name),'Design-frozen code changed: '+name)
    require(dependencies()==read(run/'dependencies.json'),'Dependencies changed')
    require(sha(Path(sys.executable))==read(run/'python_environment.json')['sha256'],'Interpreter changed')


def prepare(tests_path):
    """Only IDs, source manifests, seen physical configuration, and code are read."""
    from report_clpu_public_confirmation import STATISTICAL_RULES
    check_tests(tests_path)
    require(sha(PUBLIC_SOURCE/'completion_manifest.json')==PUBLIC_SHA,'Public development root changed')
    manifest=base.normalize_manifest(read(PUBLIC_SOURCE/'completion_manifest.json'))
    names=source_names()
    require(all((ROOT/n).is_file() for n in names),'Complete driver/auditor/statistics/test closure required')
    for name in public.source_names():
        require(sha(ROOT/name)==manifest['source/'+name],'Public inherited implementation changed: '+name)
    require(sha(FAULT_SOURCE)==FAULT_SHA and sha(POPULATION_SOURCE)==POPULATION_SHA,'ID source changed')
    require(sha(EVENT_SOURCE/'manifest.json')==EVENT_MANIFEST_SHA,'Event import manifest changed')
    event_manifest=read(EVENT_SOURCE/'manifest.json')
    require(event_manifest['status']=='COMPLETE' and event_manifest['output_sha256']['events.json']==EVENT_FILE_SHA,
            'Original event import is not complete/bound')
    # Deliberately do not open or hash events.json before the method freeze.
    old_cfg=read(PUBLIC_SOURCE/'config.json')
    require(sha(PUBLIC_SOURCE/'config.json')==manifest['config.json'],'Old scientific config changed')
    cfg={k:deepcopy(old_cfg[k]) for k in ('condition_id','alpha','algorithm','horizon','slot_seconds','sample_seconds','branch_search')}
    cfg.update(study=STUDY,execution_allowed=True,training_enabled=False,formal_holdout_numerical_access=True,
               tune_numerical_access=False,methods=['P','E'],confirmation=settings(),confirmation_statistics=STATISTICAL_RULES)
    run=create_run('clpu_public_confirmation',cfg)
    for name in PHYSICAL_INPUTS:
        base.bound_copy(PUBLIC_SOURCE,run/name,manifest,name)
    for name in ('metadata.json','base_load_properties.json'):
        base.bound_copy(PUBLIC_SOURCE,run/'source_inputs'/name,manifest,'source_inputs/'+name)
    shutil.copyfile(FAULT_SOURCE,run/'fault_partitions.json')
    shutil.copyfile(POPULATION_SOURCE,run/'population_source.json')
    shutil.copyfile(EVENT_SOURCE/'manifest.json',run/'event_source_manifest.json')
    faults=read(run/'fault_partitions.json')['scenarios']
    selected=[next(r for r in faults if r['scenario_id']==sid) for sid in FAULT_IDS]
    require(all(r['partition']=='formal_test' and r['fault_order']==2 for r in selected),'Selected fault scope differs')
    save(run/'selected_scenarios.json',selected)
    contexts=contexts_for(); schedule=schedule_for(contexts)
    save(run/'contexts.json',contexts);save(run/'confirmation_contexts.json',contexts)
    save(run/'schedule.json',schedule);save(run/'model_bindings.json',[])
    save(run/'stage_plan.json',{'schema':'clpu-public-confirmation-population-v1','execution':'single_stage_all48',
        'contexts':contexts,'all_cells':[s['cell'] for s in schedule],'quality_gate':False,
        'candidate_generation_role':'generated cold by each actual task; independently replayed after execution'})
    save(run/'source_binding.json',source_binding())
    cache.bind_scope_review(run);cache.bind_exact_key_review(run)
    save(run/'historical_mechanism_review_role.json',{'role':'Previously seen evidence-key/scope mechanism checks only; not formal-population certification',
        'history_verified_for_new_population':False,'historical_answers_loaded':False})
    save(run/'authorization.json',{'user_continuation_received':True,'scope':'4 formal N2 combinations x all six holdout events x P/E',
        'new_training':0,'external_X_included':False,'historical_answers_loaded':False})
    save(run/'resource_budget.json',{**settings(),**capacity()})
    save(run/'design_access.json',{'new_physical_points':0,'new_training':0,'holdout_event_table_opened':False,
        'formal_fault_identity_metadata_read':True,'event_expected_digest_from_existing_manifest_only':True})
    for spec in schedule:
        save(run/spec['cell']/'spec.json',spec)
        for phase in ('EARLY','GUARD'):save(run/spec['cell']/phase/'spec.json',phase_spec(spec,phase))
    shutil.copyfile(tests_path,run/'software_tests.xml')
    for name in names:
        target=run/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    save(run/'design_prepared.json',{'status':'METHOD_IDENTITIES_STATISTICS_FROZEN_BEFORE_EVENT_VALUES',
        'run':run.name,'source_files':names,'utc':utc(),'new_physical_points':0,'new_training':0})
    write_freeze(run,'method_freeze_manifest.json')
    save(ACTIVE,{'run':run.name,'status':'DESIGN_FROZEN_PENDING_EXPLICIT_HYDRATION',
        'method_freeze_sha256':sha(run/'method_freeze_manifest.json')})
    return run


def selected_event_projection(rows):
    """Exact original dictionaries, only the already frozen six IDs; no fitting."""
    require(len(rows)==31 and {r['event_id'] for r in rows}==set(range(1,32)),'Original full import identity differs')
    selected=[deepcopy(next(r for r in rows if r['event_id']==eid)) for eid in EVENT_IDS]
    from dnr.clpu_curves import split_events
    require({r['event_id'] for r in split_events([{'event_id':i} for i in range(1,32)])['holdout']}==set(EVENT_IDS),
            'Original holdout split differs')
    require(all(r['validation']['passed'] is True for r in selected),'Selected original curve validation incomplete')
    return selected


def physical_scope(run,names):
    inputs=[*PHYSICAL_INPUTS,'config.json','selected_scenarios.json','selected_events.json','confirmation_contexts.json',
        'fault_partitions.json','event_source_manifest.json','hydration.json','dependencies.json','python_environment.json',
        'source_inputs/metadata.json','source_inputs/base_load_properties.json']
    return {'scope_version':'CLPU_FORMAL_HOLDOUT_FIXED_PHYSICAL_SCOPE_V1',
        'input_files':{n:sha(run/n) for n in sorted(set(inputs))},
        'science_source_sha256':{n:sha(ROOT/n) for n in names if n.endswith('.py')},
        'source_provenance':read(run/'source_provenance.json'),'runtime_dependencies':read(run/'dependencies.json'),
        'python_environment':read(run/'python_environment.json'),'physical_config':read(run/'physical_config.json'),
        'alpha':.5,'algorithm':0,'load_setpoint_precision':'.15g',
        'reset_semantics':'INDEPENDENT_QUASI_STATIC_AUTHOR_RESET_EACH_SNAPSHOT',
        'cache_scope':'one cold cell only; no historical or cross-arm/cross-context answers',
        'fault_partition':'formal_test','event_partition':'holdout','new_training':0}


def hydrate(run):
    verify_design(run)
    require(not any((run/n).exists() for n in ('hydration_started.json','selected_events.json','prepared.json','execution_started.json')),
            'No repeated hydration or prior execution')
    budget=capacity();require(budget['free_bytes']>=budget['required_free_bytes'],'Storage forecast unmet before hydration')
    save(run/'hydration_started.json',{'utc':utc(),'method_freeze_sha256':sha(run/'method_freeze_manifest.json'),
        'source':str(EVENT_SOURCE/'events.json'),'expected_sha256':EVENT_FILE_SHA})
    require(sha(EVENT_SOURCE/'events.json')==EVENT_FILE_SHA,'Original event file changed')
    selected=selected_event_projection(read(EVENT_SOURCE/'events.json'))
    save(run/'selected_events.json',selected)
    save(run/'hydration.json',{'utc':utc(),'method_freeze_sha256':sha(run/'method_freeze_manifest.json'),
        'source_file_sha256':EVENT_FILE_SHA,'selected_event_ids':list(EVENT_IDS),
        'selected_events_sha256':sha(run/'selected_events.json'),
        'row_sha256':{str(r['event_id']):hashlib.sha256(canonical(r).encode()).hexdigest() for r in selected},
        'access':'full original imported JSON bytes decoded; only six frozen holdout rows materialized and used; no tuning',
        'original_values_unchanged':True,'new_physical_points':0,'new_training':0})
    names=read(run/'design_prepared.json')['source_files']
    save(run/'physical_scope.json',physical_scope(run,names))
    save(run/'prephysical_capacity.json',budget)
    write_freeze(run,'prephysical_freeze_manifest.json')
    save(run/'prepared.json',{'status':'FROZEN_PREPARED_NO_DSS','run':run.name,'source_files':names,
        'freeze_sha256':sha(run/'prephysical_freeze_manifest.json'),'method_freeze_sha256':sha(run/'method_freeze_manifest.json')})
    return {'run':run.name,'status':'HYDRATED_FROZEN_PENDING_INDEPENDENT_PRECHECK','new_physical_points':0}


def verify_execution(run,*,require_precheck=True):
    verify_design(run)
    p=read(run/'prepared.json')
    require(p['status']=='FROZEN_PREPARED_NO_DSS' and sha(run/'prephysical_freeze_manifest.json')==p['freeze_sha256'],'Physical freeze changed')
    cache.verify_manifest(run,'prephysical_freeze_manifest.json',('physical_scope.json','selected_events.json','hydration.json'))
    require(physical_scope(run,p['source_files'])==read(run/'physical_scope.json'),'Actual scientific scope changed')
    require([r['event_id'] for r in read(run/'selected_events.json')]==list(EVENT_IDS),'Actual events differ')
    require(read(run/'model_bindings.json')==[],'Models forbidden')
    for spec in read(run/'schedule.json'):
        require(read(run/spec['cell']/'spec.json')==spec,'Cell spec changed')
        for phase in ('EARLY','GUARD'):
            require(read(run/spec['cell']/phase/'spec.json')==phase_spec(spec,phase),'Phase spec changed')
    if require_precheck:
        cache.verify_manifest(run/'precheck',required=('summary.json',))
        check=read(run/'precheck/summary.json')
        require(check['status']=='PASS' and not check.get('errors') and
            check['freeze_sha256']==sha(run/'prephysical_freeze_manifest.json'),'Independent precheck required')


def make_env(run,spec):
    from dnr.nature34_clpu_environment import CLPUEnvironment
    from dnr.nature34_joint_sequence import JointGraph
    require(spec in schedule_for(contexts_for()),'Unscheduled formal identity')
    scenarios={r['scenario_id']:r for r in read(run/'selected_scenarios.json')}
    events={r['event_id']:r for r in read(run/'selected_events.json')}
    scenario,event=scenarios[spec['scenario_id']],events[spec['event_id']]
    require(scenario['partition']=='formal_test' and spec['event_partition']=='holdout','Formal input partition differs')
    static,loads,cfg=read(run/'author_static_metadata.json'),read(run/'load_reference.json'),read(run/'config.json')
    require(len(loads)==68,'Original 68-load inventory required')
    return CLPUEnvironment(JointGraph(static,loads,scenario),static,loads,scenario,read(run/'condition.json'),event,
        horizon=cfg['horizon'],slot_seconds=cfg['slot_seconds'],sample_seconds=cfg['sample_seconds'])


def execute(run):
    verify_execution(run)
    require(not (run/'execution_started.json').exists(),'No automatic rerun')
    budget=capacity();require(budget['free_bytes']>=budget['required_free_bytes'],'Storage forecast unmet')
    tick,epoch=time.perf_counter(),time.time()
    save(run/'execution_started.json',{'utc':utc(),'pid':os.getpid(),'deadline_epoch_seconds':epoch+MAX_SECONDS,
        'started_epoch_seconds':epoch,'started_monotonic_seconds':tick,'execution':'single_stage_all48'})
    # Reuse the already tested pair-drain launcher only, with a private globals
    # mapping. Old prepare/verify/population guards are never called or patched.
    original=base._execute_selected
    namespace={**vars(base),'__file__':__file__,'CELLS':CELLS,'TOTAL_CAP':TOTAL_CAP,'ACTIVE':ACTIVE,
        'MAX_SECONDS':MAX_SECONDS,'WORKER_SECONDS':WORKER_SECONDS,'MIN_FREE':MIN_FREE,'RESERVE':RESERVE}
    launch=FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
    outcome=launch(run,'phase1',read(run/'schedule.json'),[])
    dispatched={s['cell'] for s in outcome['dispatched']}
    complete=outcome['status']=='COMPLETE_STAGE1_PENDING_AUDIT' and len(outcome['completed'])==CELLS
    not_run=[{'cell':s['cell'],'context_id':s['context_id'],'arm':s['arm'],'status':'NOT_RUN_AFTER_STOP'}
             for s in read(run/'schedule.json') if s['cell'] not in dispatched]
    controller={**outcome,'status':'COMPLETE_CELLS_PENDING_AUDIT' if complete else 'STOPPED_INCOMPLETE',
        'execution':'single_stage_all48','conditional_gate':False,'actual_tasks':len(dispatched),
        'resolved_slots':len(outcome['completed']),'not_run':not_run,'all_fixed_context_mean_allowed':complete,
        'models_used':0,'new_training':0,'controller_execution_sha256':sha(run/'controller_stage1_completion.json'),
        'wall_seconds':time.perf_counter()-tick}
    save(run/'controller_completion.json',controller);save(ACTIVE,{**controller,'run':run.name})
    return 0 if complete else 1


def finalize(run):
    verify_execution(run)
    require(not (run/'completion_manifest.json').exists(),'Already sealed')
    cache.verify_manifest(run/'audit',required=('summary.json',))
    cache.verify_manifest(run/'derived_report',required=('summary.json','all_cells.json','all_pairs.json'))
    audit,report,controller=(read(run/n) for n in ('audit/summary.json','derived_report/summary.json','controller_completion.json'))
    require(controller['status']=='COMPLETE_CELLS_PENDING_AUDIT' and controller['actual_tasks']==controller['resolved_slots']==CELLS
        and not controller['errors'] and not controller['not_run'] and not controller['unknown_cost_cells'],'All48 true successful exits required')
    require(audit['status']=='PASS' and audit['audited_cells']==CELLS and not audit['errors']
        and audit['controller_completion_sha256']==sha(run/'controller_completion.json'),'Full48 independent audit required')
    require(report['status']=='AUDITED_DESCRIPTIVE_PUBLIC_CONFIRMATION'
        and report['scheduled_cells']==report['executed_cells']==CELLS,'Audited confirmation report required')
    for name in ('audit/summary.json','controller_completion.json'):
        require(report['source_bindings'][name]==sha(run/name),'Report source binding changed: '+name)
    terminal={'status':'COMPLETE_PUBLIC_CONFIRMATION_AUDITED_AND_REPORTED','run':run.name,'utc':utc(),
        'audit_sha256':sha(run/'audit/summary.json'),'report_sha256':sha(run/'derived_report/summary.json'),
        'scheduled_cells':48,'executed_cells':48,'new_training':0}
    save(run/'terminal_status.json',terminal);seal(run);save(ACTIVE,{**terminal,'completion_sha256':sha(run/'completion_manifest.json')})
    return terminal


def worker(run,cell):
    global _WORKER_ENTERED
    started=time.perf_counter()
    require(not _WORKER_ENTERED,'A physical worker may enter once');_WORKER_ENTERED=True
    spec=read(cell/'spec.json')
    require(not (cell/'worker_started.json').exists(),'Repeated worker forbidden')
    save(cell/'worker_started.json',{'pid':os.getpid(),'parent_pid':os.getppid(),'utc':utc(),'started_monotonic':started})
    try:
        verify_execution(run)
        require(spec in read(run/'schedule.json') and cell==run/spec['cell'],'Unscheduled worker')
        require(spec['method']=='beam' and spec['seed']==0 and spec['checkpoint'] is None and spec['checkpoint_sha256'] is None,'No model loading permitted')
        deadline=read(cell/'execution.json')['deadline_epoch_seconds']
        require(deadline<=read(run/'execution_started.json')['deadline_epoch_seconds'] and time.time()<deadline,'Invalid task deadline')
        require(shutil.disk_usage(ROOT).free>=MIN_FREE,'Storage floor reached')
        from dnr.nature34_clpu_branch import BranchConfig
        from dnr.nature34_clpu_early import merge_results
        from dnr.nature34_clpu_guard_first_control import GuardFirstEvidenceService
        from dnr.nature34_clpu_shared_evidence import SharedEvidenceService
        service_class=SharedEvidenceService if spec['arm']=='E' else GuardFirstEvidenceService
        env=make_env(run,spec);cfg=BranchConfig(**read(run/'config.json')['branch_search'])
        compute_started=time.perf_counter()
        with service_class(run,cell,env,read(run/'physical_scope.json'),mode=spec['mode'],physical_budget=512,logical_budget=256) as service:
            service.deadline_monotonic=time.perf_counter()+max(0,deadline-time.time())
            init_seconds=time.perf_counter()-compute_started
            phases,cps=run_routes(service,env,cfg,cell,spec,started,deadline,expected_generation=None)
        merged=merge_results(phases['EARLY'],phases['GUARD'],protected=True)
        selected=phases[merged['selected_phase']];finished_tick=time.perf_counter()
        result={**merged,'version':STUDY,'schema':SCHEMA,'spec':spec,'study':STUDY,
            'route_arm':spec['arm'],'phase_order':spec['phase_order'],
            'external_floor_enabled':spec['external_floor_enabled'],
            'free_external_floor_pruning_enabled':spec['free_external_floor_pruning_enabled'],
            'intermediate_beam_floor_claimed':spec['arm']!='E',
            'guard_first':phases['EARLY'].get('guard_first'),
            'quality_pct':selected['quality_pct'],'fully_certified':selected['fully_certified'],
            'early_quality_pct':phases['EARLY']['quality_pct'],'baseline_quality_pct':phases['GUARD']['quality_pct'],
            'logical_points':sum(p['logical_points'] for p in phases.values()),'actual_physical_points':service.physical_points,
            'numerical_entries':service.factory.stats()['returned']['numerical_solution_entries'],
            'requests':sum(p['oracle']['requests'] for p in phases.values()),
            'search_decisions':sum(p['search_decisions'] for p in phases.values()),
            'policy_decisions':0,'models_used':0,'model_load_seconds':0.,'checkpoint_sha256':None,
            'stage_generation_sha256':hashlib.sha256(canonical(phases['EARLY']['stage_generation']).encode()).hexdigest(),
            'stage_generation_seconds':phases['EARLY']['stage_generation_seconds'],
            'stage_candidate_count':len((phases['EARLY']['stage_generation'] or {}).get('candidates',[])),
            'service':service.summary(),'phase_summary':{k:{n:v[n] for n in ('quality_pct','fully_certified','logical_points','actual_physical_points','search_decisions','policy_decisions','graph_expansions','verify_attempts','phase_elapsed_seconds','stage_component_seconds')} for k,v in phases.items()},
            'checkpoints':cps,'worker_elapsed_seconds':finished_tick-started,
            'preflight_elapsed_seconds':compute_started-started,'runtime_initialization_seconds':init_seconds,
            'computation_elapsed_seconds':finished_tick-compute_started,'new_training':0,
            'formal_holdout_numerical_access':True,'tune_numerical_access':False,
            'candidate_generation_role':'actual cold task, not preobserved inventory'}
        require(all(p['policy_decisions']==0 for p in phases.values()),'Unexpected learned decision')
        save(cell/'result.json',result)
        save(cell/'worker_finished.json',{'status':'COMPLETE','pid':os.getpid(),'utc':utc(),
            'result_sha256':sha(cell/'result.json'),'worker_elapsed_seconds':time.perf_counter()-started})
        seal(cell);return 0
    except BaseException as exc:
        save(cell/'error.json',{'status':'STOPPED_RELIABILITY_OR_IMPLEMENTATION_ERROR','error':repr(exc),'traceback':traceback.format_exc(),
            'partial_evidence_retained':True,'no_automatic_retry':True,'pid':os.getpid()})
        return 1


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','hydrate','worker','execute','finalize'))
    parser.add_argument('--run',type=Path);parser.add_argument('--cell',type=Path);parser.add_argument('--tests',type=Path)
    args=parser.parse_args()
    if args.command=='prepare':print(prepare(args.tests.resolve()))
    elif args.command=='hydrate':print(hydrate(args.run.resolve()))
    elif args.command=='worker':sys.exit(worker(args.run.resolve(),args.cell.resolve()))
    elif args.command=='execute':sys.exit(execute(args.run.resolve()))
    else:print(finalize(args.run.resolve()))
