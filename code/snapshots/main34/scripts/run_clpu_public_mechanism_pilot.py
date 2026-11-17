"""Cold matched P/E/A_prune public-mechanism pilot; no learned models.

The original physical sources, complete GUARD and staged candidate generator
are reused unchanged. Each cell owns one runtime and its own two-view cache.
"""
from __future__ import annotations

import argparse
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
import run_clpu_guard_first_pilot as previous
energy = previous.previous
base = previous.base

read, require, sha, canonical = base.read, base.require, base.sha, base.canonical
save, seal, utc, cache = base.save, base.seal, base.utc, base.cache
SOURCE, SOURCE_SHA = base.SOURCE, base.SOURCE_SHA
GRAPH_PROFILE, GRAPH_PROFILE_SHA = base.GRAPH_PROFILE, base.GRAPH_PROFILE_SHA
STUDY = 'CLPU_PUBLIC_MECHANISM_COMMON_PILOT_V1'
SCHEMA = 'clpu-public-mechanism-protected-result-v1'
PROTOCOL = 'docs/clpu_public_mechanism_pilot_protocol_v1.md'
ACTIVE = ROOT/'artifacts/clpu_public_mechanism_pilot_active.json'
COMPARISON_SOURCE = ROOT/'runs/20261031T103802_907721Z_clpu_guard_first_pilot'
COMPARISON_SHA = 'af9dc11957274cb84317e6f227fd94bf0f75df35f207667e26ff466d0678ac02'
COMPARISON_FILES = previous.COMPARISON_FILES
SLOTS, CELLS = ('P', 'E', 'A_prune'), 24
MAX_SECONDS, WORKER_SECONDS, TOTAL_CAP = 3600, 900, CELLS*512
MIN_FREE, RESERVE = 12*2**30, 2*2**30
_WORKER_ENTERED = False
ADDITIONS = (PROTOCOL, 'src/dnr/nature34_clpu_public_ablation_control.py',
    'scripts/run_clpu_public_mechanism_pilot.py', 'scripts/audit_clpu_public_mechanism_pilot.py',
    'tests/test_nature34_clpu_public_ablation_control.py', 'tests/test_run_clpu_public_mechanism_pilot.py',
    'tests/test_clpu_public_mechanism_audit.py')
ExitStack, gzip, math = base.ExitStack, base.gzip, base.math
append, numeric_lists = base.append, base.numeric_lists
phase_spec, costs, quality, compact_proposal = base.phase_spec, base.costs, base.quality, base.compact_proposal


def settings():
    value = base.settings()
    for key in ('conditional_stages', 'no_gain_remaining_status'):
        value.pop(key)
    value.update(cells=CELLS, slots=list(SLOTS), seeds=[], models_used=0,
        execution='single_stage_all24', quality_gate=False,
        total_physical_budget=TOTAL_CAP, maximum_numerical_entries=CELLS*3+4*TOTAL_CAP,
        phase_order='explicit per arm', phases=['EARLY', 'GUARD'],
        arm_phase_orders={'P':['GUARD','EARLY'], 'E':['EARLY','GUARD'], 'A_prune':['GUARD','EARLY']},
        external_floor={'P':True, 'E':False, 'A_prune':True},
        free_external_floor_pruning={'P':True, 'E':False, 'A_prune':False},
        own_incumbent_validation_skip='unchanged original EARLY complete-objective skip in all arms',
        model_role='none; all three arms use the unchanged visible complete-energy selector',
        dispatch='public-mechanism-context-v1 hashed contexts; three rounds, arm=(context_rank+round)%3; adjacent pairs drained',
        aggregation='paired three deterministic arms within each of all eight seen contexts; all24 retained',
        comparison='current cold24 process times only; prior GUARD-first data is source/reference, not a timing denominator',
        internal_execution_journal='controller_stage1_* names retained for one all24 pass only',
        contrast_P_E='whole GUARD-first information/order mechanism, not identical intermediate windows',
        contrast_P_A_prune='free certified-floor batch removal conditional on GUARD-first; original own skip retained')
    return value


def schedule_for(contexts, models=()):
    require(not models, 'This rule-only study forbids model bindings')
    require(len(contexts)==8 and {c['context_id'] for c in contexts}==set(range(8)), 'All eight contexts required')
    require(len({c['scenario_id'] for c in contexts})==8, 'Fixed faults changed')
    ordered=sorted(contexts,key=lambda c:hashlib.sha256(('public-mechanism-context-v1|'+canonical(c)).encode()).hexdigest())
    rows=[]
    for turn in range(3):
        for rank, context in enumerate(ordered):
            arm=SLOTS[(rank+turn)%3]
            order=['EARLY','GUARD'] if arm=='E' else ['GUARD','EARLY']
            rows.append({**{k:context[k] for k in ('context_id','scenario_id','event_id')},
                'cell':f'episodes/{len(rows):03d}', 'episode_id':len(rows), 'pair_order':len(rows)//2,
                'context_order':rank, 'arm_position':turn, 'dispatch_round':turn,
                'arm_slot':arm, 'arm':arm, 'route_arm':arm, 'phase_order':order,
                'external_floor_enabled':arm!='E', 'free_external_floor_pruning_enabled':arm=='P',
                'mode':'shared', 'method':'beam', 'seed':0, 'macro_schedule':'staged',
                'phase':'evaluation', 'group':arm, 'protected':True, 'representation':'raw',
                'deterministic':True, 'checkpoint':None, 'checkpoint_sha256':None,
                'point_budget':256, 'logical_point_budget':256, 'physical_point_budget':512,
                'total_logical_point_budget':512, 'window_variant':'staged_common_pool_coverage_v1'})
    return rows


def partition_schedule(schedule, rows):
    require(len(schedule)==CELLS and len({s['cell'] for s in schedule})==CELLS, 'All24 actual tasks required')
    require(len(rows)==8 and [r['context_id'] for r in rows]==list(range(8)), 'Original8 graph inventories required')
    for row in rows:
        require(len(row['generation']['candidates'])<=24, 'Original staged candidate cap changed')
    eligible=[r['context_id'] for r in rows if any(c['graph_energy_kwh']>r['reference_energy_kwh'] for c in r['generation']['candidates'])]
    return {'schema':'clpu-public-mechanism-fixed-population-v1','original_contexts':8,'scheduled_cells':CELLS,
        'execution':'single_stage_all24','quality_gate':False,'all_cells':[s['cell'] for s in schedule],
        'eligible_contexts':eligible,'eligible_contexts_role':'historical graph diagnostic only; does not gate dispatch',
        'contexts':rows,'new_physical_points':0,'new_training':0}


def source_names():
    return sorted(set(previous.source_names()) | set(ADDITIONS))


def validate_tests(path):
    cases=list(ET.parse(path).getroot().iter('testcase'))
    require(cases and not any(list(c.iter('failure'))+list(c.iter('error'))+list(c.iter('skipped')) for c in cases), 'Successful component tests required')
    for part in ('test_nature34_clpu_public_ablation_control','test_run_clpu_public_mechanism_pilot','test_clpu_public_mechanism_audit'):
        require(any(part in c.get('classname','') for c in cases), 'Missing new component '+part)
    return len(cases)


def source_binding():
    return {'source_run':SOURCE.name,'source_completion_sha256':SOURCE_SHA,
        'models_used':0, 'historical_answers_loaded':False}


def comparison_binding():
    return {'source_run':COMPARISON_SOURCE.name,'source_completion_sha256':COMPARISON_SHA,
        'role':'matched prior candidate/scientific reference; no historical answers or cross-date time denominator'}


def bind_comparison(run):
    require(sha(COMPARISON_SOURCE/'completion_manifest.json')==COMPARISON_SHA, 'Comparison root seal changed')
    manifest=base.normalize_manifest(read(COMPARISON_SOURCE/'completion_manifest.json'))
    require(read(COMPARISON_SOURCE/'prepared.json')['source_files']==previous.source_names(), 'Prior scientific closure differs')
    for name in COMPARISON_FILES:
        base.bound_copy(COMPARISON_SOURCE,run/'comparison_source'/name,manifest,name)
    shutil.copyfile(COMPARISON_SOURCE/'completion_manifest.json',run/'comparison_source_completion_manifest.json')
    save(run/'comparison_source_binding.json',comparison_binding())
    verify_comparison(run)


def verify_comparison(run):
    require(read(run/'comparison_source_binding.json')==comparison_binding(), 'Changed comparison attribution')
    require(sha(run/'comparison_source_completion_manifest.json')==COMPARISON_SHA, 'Changed comparison manifest')
    manifest=base.normalize_manifest(read(run/'comparison_source_completion_manifest.json'))
    for name in COMPARISON_FILES:
        require(sha(run/'comparison_source'/name)==manifest[name], 'Changed comparison small input: '+name)
    audit=read(run/'comparison_source/audit/summary.json')
    require(audit['status']=='PASS' and audit['audited_cells']==32 and not audit['errors'], 'Prior full independent audit required')
    require(read(run/'comparison_source/contexts.json')==read(run/'contexts.json'), 'Comparison context changed')
    require(read(run/'comparison_source/stage_plan.json')['contexts']==read(run/'stage_plan.json')['contexts'], 'Candidate generation/reference changed')
    for name in previous.source_names():
        require(sha(ROOT/name)==manifest['source/'+name], 'Previously executed source changed: '+name)


def rebound(name):
    original=getattr(base,name)
    function=FunctionType(original.__code__,_PRIVATE,name,original.__defaults__,original.__closure__)
    function.__kwdefaults__=original.__kwdefaults__
    return function


def prepare(tests_path):
    from dnr.nature34_clpu_branch import BranchConfig
    count=validate_tests(tests_path)
    require(sha(SOURCE/'completion_manifest.json')==SOURCE_SHA, 'Inherited source seal changed')
    sm=base.normalize_manifest(read(SOURCE/'completion_manifest.json'))
    for name in ('config.json','schedule.json','training_contexts.json','selected_scenarios.json','selected_events.json','prepared.json','audit/summary.json','contexts.json'):
        require(sha(SOURCE/name)==sm[name], 'Inherited input changed: '+name)
    audit=read(SOURCE/'audit/summary.json')
    require(audit['status']=='PASS' and audit['audited_cells']==64 and not audit['errors'], 'Complete source audit required')
    chosen=[s for s in read(SOURCE/'schedule.json') if s['arm']=='deferred_window']
    contexts=read(SOURCE/'contexts.json'); schedule=schedule_for(contexts)
    stage_plan=rebound('build_stage_plan')(contexts,schedule,sm)
    graph_files=base.check_graph_profile(GRAPH_PROFILE,stage_plan)
    cfg=deepcopy(read(SOURCE/'config.json'))
    require(cfg['branch_search']==base.asdict(BranchConfig()), 'Original search changed')
    cfg.update(staged_study=STUDY,staged_pilot=settings())
    rates=[]
    for spec in chosen:
        folder=SOURCE/spec['cell']; result=read(folder/'result.json')
        rates.append(sum(f.stat().st_size for f in folder.rglob('*') if f.is_file())/max(1,result['actual_physical_points']))
    budget={'historical_max_bytes_per_actual_point':max(rates),'total_physical_cap':TOTAL_CAP,
        'forecast_bytes':math.ceil(max(rates)*TOTAL_CAP*1.25),'minimum_free_bytes':MIN_FREE,'report_reserve_bytes':RESERVE,
        'free_bytes_at_prepare':shutil.disk_usage(ROOT).free,'successful_component_tests':count}
    budget['required_free_bytes']=MIN_FREE+RESERVE+budget['forecast_bytes']
    require(budget['free_bytes_at_prepare']>=budget['required_free_bytes'], 'Storage forecast unmet')
    run=base.create_run('clpu_public_mechanism_pilot',cfg)
    try:
        for name in (*base.ROOT_INPUTS,'population.json','selected_scenarios.json','selected_events.json','training_contexts.json'):
            base.bound_copy(SOURCE,run/name,sm,name)
        for name in sm:
            if name.startswith('source_inputs/'):
                base.bound_copy(SOURCE,run/name,sm,name)
        save(run/'contexts.json',contexts);save(run/'stage_plan.json',stage_plan)
        for name in graph_files:
            target=run/'graph_profile'/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(GRAPH_PROFILE/name,target)
        shutil.copyfile(GRAPH_PROFILE/'completion_manifest.json',run/'graph_profile/completion_manifest.json')
        save(run/'graph_profile_binding.json',{'path':GRAPH_PROFILE.relative_to(ROOT).as_posix(),'completion_sha256':GRAPH_PROFILE_SHA})
        save(run/'source_binding.json',source_binding());save(run/'model_bindings.json',[])
        shutil.copyfile(SOURCE/'completion_manifest.json',run/'source_completion_manifest.json')
        historical=[]
        for spec in chosen:
            item={k:spec[k] for k in ('context_id','scenario_id','event_id')}
            item.update(source_cell=spec['cell'],phases={})
            original=spec['cell']+'/GUARD/phase_result.json'
            target=f'historical/c{spec["context_id"]:02d}/GUARD/phase_result.json'
            base.bound_copy(SOURCE,run/target,sm,original)
            item['phases']['GUARD']={'result_file':target,'source_file':original,'sha256':sm[original]}
            historical.append(item)
        save(run/'historical_bindings.json',historical)
        cache.bind_scope_review(run);cache.bind_exact_key_review(run)
        save(run/'resource_budget.json',budget);save(run/'schedule.json',schedule)
        for spec in schedule:
            save(run/spec['cell']/'spec.json',spec)
            for phase in spec['phase_order']:
                save(run/spec['cell']/phase/'spec.json',phase_spec(spec,phase))
        shutil.copyfile(tests_path,run/'software_tests.xml');shutil.copyfile(ROOT/PROTOCOL,run/'protocol.md')
        save(run/'authorization.json',{'utc':utc(),'execution_authorized':True,
            'scope':'User accepted public P as main claim and authorized all8 seen contexts x P/E/A_prune =24 fresh cold tasks; complete Beam and original physics retained; no training.',
            'models_used':0,'new_training':0,'no_automatic_retry':True})
        names=source_names()
        for name in names:
            if 'source/'+name in sm:
                require(sha(ROOT/name)==sm['source/'+name], 'Inherited code changed: '+name)
        bind_comparison(run)
        save(run/'physical_scope.json',cache.physical_scope(run,names));base.freeze(run,names)
        save(ACTIVE,{'status':'FROZEN_NOT_STARTED','run':run.name,'scheduled':CELLS,'completed':0,'utc':utc()})
        return run
    except BaseException as exc:
        save(run/'preparation_error.json',{'error':repr(exc),'traceback':traceback.format_exc(),'new_physical_points':0})
        raise


def verify_execution(run):
    rebound('verify_execution')(run)
    verify_comparison(run)
    require(read(run/'model_bindings.json')==[], 'No model may enter this rule-only comparison')
    require(not any((run/name).exists() for name in ('phase_gate.json','audit_stage1','controller_stage2_started.json')),
        'Single all24 execution forbids conditional-stage artifacts')


def run_routes(service,env,branch_config,cell,spec,started,deadline,policy=None,generator=None,search_classes=None,expected_generation=None):
    require(policy is None and generator is None, 'Learned policies forbidden')
    if spec['arm']=='E':
        return energy.run_routes(service,env,branch_config,cell,spec,started,deadline,
            search_classes=search_classes,expected_generation=expected_generation)
    return _guard_routes(service,env,branch_config,cell,spec,started,deadline,
        search_classes=search_classes,expected_generation=expected_generation)


def worker(run,cell):
    global _WORKER_ENTERED
    started=time.perf_counter()
    require(not _WORKER_ENTERED, 'A physical worker may enter once');_WORKER_ENTERED=True
    spec=read(cell/'spec.json')
    require(not (cell/'worker_started.json').exists(), 'Repeated worker forbidden')
    save(cell/'worker_started.json',{'pid':os.getpid(),'parent_pid':os.getppid(),'utc':utc(),'started_monotonic':started})
    try:
        verify_execution(run)
        require(spec in read(run/'schedule.json') and cell==run/spec['cell'], 'Unscheduled worker')
        require(spec['method']=='beam' and spec['seed']==0 and spec['checkpoint'] is None and spec['checkpoint_sha256'] is None, 'No model loading permitted')
        deadline=read(cell/'execution.json')['deadline_epoch_seconds']
        require(deadline<=read(run/'execution_started.json')['deadline_epoch_seconds'] and time.time()<deadline, 'Invalid task deadline')
        require(shutil.disk_usage(ROOT).free>=MIN_FREE, 'Storage floor reached')
        from run_nature34_clpu_pilot import make_env
        from dnr.nature34_clpu_branch import BranchConfig
        from dnr.nature34_clpu_early import merge_results
        from dnr.nature34_clpu_guard_first_control import GuardFirstEvidenceService
        from dnr.nature34_clpu_shared_evidence import SharedEvidenceService
        service_class=SharedEvidenceService if spec['arm']=='E' else GuardFirstEvidenceService
        env=make_env(run,spec); cfg=BranchConfig(**read(run/'config.json')['branch_search'])
        compute_started=time.perf_counter()
        with service_class(run,cell,env,read(run/'physical_scope.json'),mode=spec['mode'],physical_budget=512,logical_budget=256) as service:
            service.deadline_monotonic=time.perf_counter()+max(0,deadline-time.time())
            init_seconds=time.perf_counter()-compute_started
            expected=next(r['generation'] for r in read(run/'stage_plan.json')['contexts'] if r['context_id']==spec['context_id'])
            phases,cps=run_routes(service,env,cfg,cell,spec,started,deadline,expected_generation=expected)
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
            'computation_elapsed_seconds':finished_tick-compute_started,'new_training':0}
        require(all(p['policy_decisions']==0 for p in phases.values()), 'Unexpected learned decision')
        save(cell/'result.json',result)
        save(cell/'worker_finished.json',{'status':'COMPLETE','pid':os.getpid(),'utc':utc(),
            'result_sha256':sha(cell/'result.json'),'worker_elapsed_seconds':time.perf_counter()-started})
        seal(cell);return 0
    except BaseException as exc:
        save(cell/'error.json',{'status':'STOPPED_RELIABILITY_OR_IMPLEMENTATION_ERROR','error':repr(exc),'traceback':traceback.format_exc(),
            'partial_evidence_retained':True,'no_automatic_retry':True,'pid':os.getpid()})
        return 1


def execute(run):
    verify_execution(run)
    require(not (run/'execution_started.json').exists(), 'No automatic rerun')
    tick,epoch=time.perf_counter(),time.time()
    save(run/'execution_started.json',{'utc':utc(),'pid':os.getpid(),'deadline_epoch_seconds':epoch+MAX_SECONDS,
        'started_epoch_seconds':epoch,'started_monotonic_seconds':tick,'execution':'single_stage_all24'})
    schedule=read(run/'schedule.json')
    require(read(run/'stage_plan.json')['all_cells']==[s['cell'] for s in schedule], 'All24 dispatch population changed')
    outcome=rebound('_execute_selected')(run,'phase1',schedule,[])
    complete=outcome['status']=='COMPLETE_STAGE1_PENDING_AUDIT' and len(outcome['completed'])==CELLS
    dispatched={s['cell'] for s in outcome['dispatched']}
    not_run=[{'cell':s['cell'],'context_id':s['context_id'],'arm':s['arm'],'status':'NOT_RUN_AFTER_STOP'} for s in schedule if s['cell'] not in dispatched]
    controller={**outcome,'status':'COMPLETE_CELLS_PENDING_AUDIT' if complete else 'STOPPED_INCOMPLETE',
        'execution':'single_stage_all24','conditional_gate':False,'actual_tasks':len(dispatched),
        'resolved_slots':len(outcome['completed']),'not_run':not_run,'all_eight_context_mean_allowed':complete,
        'models_used':0,'controller_execution_sha256':sha(run/'controller_stage1_completion.json'),
        'wall_seconds':time.perf_counter()-tick}
    save(run/'controller_completion.json',controller);save(ACTIVE,{**controller,'run':run.name})
    return 0 if complete else 1


def finalize(run):
    require(not (run/'completion_manifest.json').exists(), 'Already sealed')
    verify_execution(run)
    cache.verify_manifest(run/'audit',required=('summary.json',))
    cache.verify_manifest(run/'derived_report',required=('summary.json','all_cells.json','all_pairs.json'))
    audit=read(run/'audit/summary.json');report=read(run/'derived_report/summary.json');controller=read(run/'controller_completion.json')
    require(controller['status']=='COMPLETE_CELLS_PENDING_AUDIT' and controller['actual_tasks']==controller['resolved_slots']==CELLS
        and not controller['errors'] and not controller['not_run'] and not controller['unknown_cost_cells'], 'All24 true successful exits required')
    require(audit['status']=='PASS' and audit['audited_cells']==audit['resolved_slots']==CELLS and not audit['errors']
        and audit['controller_completion_sha256']==sha(run/'controller_completion.json'), 'Full24 independent audit required')
    require(report['status']=='AUDITED_DESCRIPTIVE_PUBLIC_MECHANISM_COMMON_PILOT' and report['scheduled_cells']==report['executed_cells']==CELLS,
        'Full24 descriptive report required')
    for name in ('audit/summary.json','controller_completion.json'):
        require(report['source_bindings'][name]==sha(run/name), 'Report binding differs: '+name)
    terminal={'status':'COMPLETE_PUBLIC_MECHANISM_PILOT_AUDITED_AND_REPORTED','run':run.name,'utc':utc(),
        'audit_sha256':sha(run/'audit/summary.json'),'report_sha256':sha(run/'derived_report/summary.json'),
        'scheduled_cells':CELLS,'executed_cells':CELLS,'not_run_cells':0,'all_eight_context_mean_allowed':True,
        'physical_points':audit['physical_points'],'numerical_entries':audit['numerical_entries'],'models_used':0,'new_training':0}
    save(run/'terminal_status.json',terminal);seal(run)
    save(ACTIVE,{**terminal,'status':'COMPLETE_PUBLIC_MECHANISM_PILOT_SEALED','completion_sha256':sha(run/'completion_manifest.json')})
    return terminal


def _guard_routes(service,env,branch_config,cell,spec,started,deadline,policy=None,generator=None,search_classes=None,expected_generation=None):
    """Original GF recording loop, with an explicit A-only class selection."""
    from dnr.nature34_clpu_branch import BranchSearch
    require(spec['arm'] in ('P','A_prune'), 'Unknown guard-first arm')
    if spec['arm']=='A_prune':
        from dnr.nature34_clpu_public_ablation_control import NoFreePruneStagedSearch as StagedCertificationSearch
        from dnr.nature34_clpu_public_ablation_control import NoFreePruneMacroController as StagedMacroController
    else:
        from dnr.nature34_clpu_guard_first_control import GuardFirstStagedSearch as StagedCertificationSearch
        from dnr.nature34_clpu_guard_first_control import GuardFirstMacroController as StagedMacroController
    classes=search_classes or {'EARLY':StagedCertificationSearch,'GUARD':BranchSearch}
    results,checkpoints={},[]
    previous_logical=previous_requests=previous_choices=0
    previous_quality=None
    with (cell/'cost_checkpoints.jsonl').open('x',encoding='utf-8') as cpstream:
        for phase in ('GUARD','EARLY'):
            require(time.time()<deadline,'Task deadline before phase')
            folder=cell/phase; ps=phase_spec(spec,phase)
            phase_started=time.perf_counter()
            save(folder/'initial_state.json',env.initial.to_dict())
            timers={k:0. for k in ('selection_seconds','search_step_seconds','journal_seconds')}
            with ExitStack() as stack:
                streams={n:stack.enter_context(gzip.open(folder/(n+'.jsonl.gz'),'xt',encoding='utf-8',compresslevel=3)) for n in ('plans','decisions')}
                emit=lambda name,row:append(streams[name],numeric_lists(row))
                oracle=stack.enter_context(service.view(phase))
                require(oracle.points==oracle.physical_points==oracle.requests==0,'Logical view not cold')
                extra={}
                if phase=='EARLY':
                    donor=results['GUARD']
                    require(donor['done'] and donor['fully_certified'] and service.views['GUARD'].closed,
                        'GUARD must complete and close before EARLY')
                    extra={'guard_result':donor,'guard_reference':{'phase':'GUARD',
                        'file':'GUARD/phase_result.json','sha256':sha(cell/'GUARD/phase_result.json')}}
                search=classes[phase](env,oracle,branch_config,record_callback=lambda r:emit('plans',r),**extra)
                if phase=='EARLY':
                    if search.stage_generation is None:
                        save(folder/'candidate_pool.json',{'generation':None,'generation_sha256':None,
                            'status':'BOOTSTRAP_STOP_NOT_GENERATED','candidate_count':0})
                    require(search.stage_generation is not None,'Paid bootstrap did not complete staged generation')
                    pool=numeric_lists(search.stage_generation)
                    if expected_generation is not None:
                        require(pool==expected_generation,'Actual cold candidate generation differs from frozen pure-graph inventory')
                    save(folder/'candidate_pool.json',{'generation':pool,
                        'generation_sha256':hashlib.sha256(canonical(pool).encode()).hexdigest()})
                control=StagedMacroController(search,method=spec['method'],
                    policy=policy,generator=generator) if phase=='EARLY' else None
                count=0
                def cp(kind):
                    q=quality(search)
                    qs=[x for x in (previous_quality,q) if x is not None]
                    row={'checkpoint_index':len(checkpoints),'phase':phase,'kind':kind,'phase_choices':count,
                        'search_choices':previous_choices+count,'logical_points':previous_logical+oracle.points,
                        'physical_points':service.physical_points,
                        'numerical_entries':service.factory.stats()['returned']['numerical_solution_entries'],
                        'requests':previous_requests+oracle.requests,'phase_logical_points':oracle.points,
                        'phase_physical_points':oracle.physical_points,'phase_numerical_entries':4*oracle.physical_points,
                        'phase_quality_pct':q,'quality_pct':max(qs) if qs else None,'fully_certified':bool(qs),
                        'guard_established':(phase=='GUARD' and kind=='CLOSE' and q is not None)
                            or (phase=='EARLY' and results['GUARD']['fully_certified']),
                        'worker_elapsed_seconds':time.perf_counter()-started}
                    checkpoints.append(row); append(cpstream,row)
                cp('INITIAL')
                while not search.done:
                    require(time.time()<deadline,'Task deadline during search')
                    before=time.perf_counter()
                    proposal=control.prepare() if control else None
                    if control and proposal is None:
                        timers['selection_seconds']+=time.perf_counter()-before
                        break
                    action=proposal['action'] if control else search.rule_action('beam')
                    timers['selection_seconds']+=time.perf_counter()-before
                    if control:
                        compact,obs,raw_sha,compact_sha=compact_proposal(proposal)
                        identity={'transition_id':count,'phase_choice':count,'phase':phase,'proposal':compact,
                            'obs_raw_sha256':raw_sha,'obs_compact_sha256':compact_sha,
                            'checkpoint_sha256':spec['checkpoint_sha256'] if proposal['policy_used'] else None,
                            'oracle_before':costs(oracle)}
                    else:
                        identity={'transition_id':count,'phase_choice':count,'phase':phase,'method':'beam','action':action,'oracle_before':costs(oracle)}
                    before=time.perf_counter();emit('decisions',{**identity,'status':'STARTED',**({'obs_compact':obs} if control else {})})
                    timers['journal_seconds']+=time.perf_counter()-before
                    require(time.time()<deadline,'Task deadline before apply')
                    before=time.perf_counter();response=control.apply(proposal) if control else search.step(action)
                    timers['search_step_seconds']+=time.perf_counter()-before
                    require(type(response.get('done')) is bool,'Invalid step response')
                    if not control:require(math.isfinite(response['reward']),'Invalid guard reward')
                    before=time.perf_counter();emit('decisions',{**identity,'status':'COMPLETED','response':response,'oracle_after':costs(oracle)})
                    count+=1;cp('DECISION')
                    timers['journal_seconds']+=time.perf_counter()-before
                    require(count<=(17 if control else 256),'Phase operation limit exceeded')
                result=numeric_lists(control.result() if control else search.result());q=quality(search)
                require(search.done and search.stop_reason not in ('DEFERRED_RELIABILITY_OR_IMPLEMENTATION_ERROR',),
                    'Unfinished or erroneous EARLY route cannot establish success')
            cp('CLOSE')
            result.update(spec=ps,oracle=oracle.summary(),logical_points=oracle.points,
                actual_physical_points=oracle.physical_points,quality_pct=q,fully_certified=q is not None,
                search_decisions=count,policy_decisions=result.get('policy_decisions',0),
                graph_expansions=result.get('graph_expansions',count),verify_attempts=result.get('verify_attempts'),
                stage_component_seconds=timers,phase_elapsed_seconds=time.perf_counter()-phase_started,
                worker_start_seconds=phase_started-started,worker_end_seconds=time.perf_counter()-started)
            save(folder/'phase_result.json',result);results[phase]=result
            previous_logical+=oracle.points;previous_requests+=oracle.requests;previous_choices+=count
            previous_quality=checkpoints[-1]['quality_pct']
    return results,checkpoints


def _private_namespace():
    namespace=dict(vars(base))
    namespace.update(__file__=__file__,STUDY=STUDY,PROTOCOL=PROTOCOL,ACTIVE=ACTIVE,SLOTS=SLOTS,
        CELLS=CELLS,TOTAL_CAP=TOTAL_CAP,MAX_SECONDS=MAX_SECONDS,WORKER_SECONDS=WORKER_SECONDS,
        settings=settings,schedule_for=schedule_for,partition_schedule=partition_schedule,
        source_binding=source_binding,source_names=source_names,validate_tests=validate_tests,
        save=save,verify_execution=verify_execution,run_routes=run_routes)
    return namespace


_PRIVATE=_private_namespace()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','worker','execute','finalize'))
    parser.add_argument('--run',type=Path);parser.add_argument('--cell',type=Path);parser.add_argument('--tests',type=Path)
    args=parser.parse_args()
    if args.command=='prepare':print(prepare(args.tests.resolve()))
    elif args.command=='worker':sys.exit(worker(args.run.resolve(),args.cell.resolve()))
    elif args.command=='execute':sys.exit(execute(args.run.resolve()))
    else:print(finalize(args.run.resolve()))
