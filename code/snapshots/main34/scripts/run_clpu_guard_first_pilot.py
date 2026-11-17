"""Public GUARD-first all32 pilot; old sources and physical mechanics remain sealed.

GUARD completes and closes before EARLY. An external certificate floor is
explicitly referenced, while every EARLY certificate remains in its own ledger.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import shutil
import sys
import time
from types import FunctionType
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
sys.dont_write_bytecode = True
import run_clpu_staged_energy_pilot as previous
base = previous.base

read, require, sha, canonical = base.read, base.require, base.sha, base.canonical
save, seal, utc, cache = base.save, base.seal, base.utc, base.cache
SOURCE, SOURCE_SHA = base.SOURCE, base.SOURCE_SHA
MODELS_SOURCE, MODELS_SHA = base.MODELS_SOURCE, base.MODELS_SHA
GRAPH_PROFILE, GRAPH_PROFILE_SHA = base.GRAPH_PROFILE, base.GRAPH_PROFILE_SHA
STUDY = 'CLPU_GUARD_FIRST_COMMON_PILOT_V1'
SCHEMA = 'clpu-guard-first-protected-result-v1'
PROTOCOL = 'docs/clpu_guard_first_teacher_protocol_v1.md'
ACTIVE = ROOT/'artifacts/clpu_guard_first_pilot_active.json'
COMPARISON_SOURCE = ROOT/'runs/20261031T092543_308214Z_clpu_staged_energy_pilot'
COMPARISON_SHA = '9a2a6f376d11986e5483442457bdb51c5fb58d169a656cbe52c9d908bea4b2cc'
SEEDS, CELLS = (17,41,73), 32
SLOTS = ('guard_first_window', *(f'staged_bc{s}' for s in SEEDS))
MAX_SECONDS, WORKER_SECONDS, TOTAL_CAP = 3600, 900, 16384
MIN_FREE, RESERVE = 12*2**30, 2*2**30
ADDITIONS = (PROTOCOL, 'src/dnr/nature34_clpu_guard_first_control.py',
    'scripts/run_clpu_guard_first_pilot.py', 'scripts/audit_clpu_guard_first_pilot.py',
    'tests/test_nature34_clpu_guard_first_control.py', 'tests/test_run_clpu_guard_first_pilot.py',
    'tests/test_clpu_guard_first_audit.py')
COMPARISON_FILES = ('config.json','schedule.json','contexts.json','stage_plan.json',
    'prepared.json','audit/summary.json','terminal_status.json')

# Shared unchanged mechanics and serialization helpers used by run_routes.
ExitStack, gzip, math = base.ExitStack, base.gzip, base.math
append, numeric_lists = base.append, base.numeric_lists
phase_spec, costs, quality, compact_proposal = base.phase_spec, base.costs, base.quality, base.compact_proposal


def settings():
    value = base.settings()
    del value['conditional_stages']; del value['no_gain_remaining_status']
    value.update(slots=list(SLOTS), execution='single_stage_all32',
        internal_execution_journal='controller_stage1_* names retained for one all32 pass only',
        quality_gate=False, ranking_change='unchanged complete-energy selector and fixed BC; common external certified-floor pruning',
        comparison='only current cold32 process times; previous EARLY-first energy run is a quality/point reference; no cross-date time ratio',
        same_optimized_window_for_all_rankers=True, phases=['GUARD','EARLY'],phase_order=['GUARD','EARLY'],
        certified_floor='external complete GUARD certificate; EARLY own best_plan remains local',
        free_pruning='original complete priority >= effective certified priority; zero VERIFY and oracle requests',
        inherited_feedback='EARLY own feedback only; GUARD paid reliable evidence queried through shared service')
    return value


def schedule_for(contexts, models):
    # Reuse the frozen hash domains and Latin rotation, replacing only the arm
    # label in SLOTS. No post-result reordering or seed selection is permitted.
    return rebound('schedule_for')(contexts, models)


def partition_schedule(schedule, rows):
    require(len(schedule)==32 and len({s['cell'] for s in schedule})==32, 'All32 actual tasks required')
    require(len(rows)==8 and [r['context_id'] for r in rows]==list(range(8)), 'Original8 graph inventories required')
    eligible=[r['context_id'] for r in rows if any(c['graph_energy_kwh']>r['reference_energy_kwh'] for c in r['generation']['candidates'])]
    return {'schema':'clpu-guard-first-fixed-population-v1','original_contexts':8,'scheduled_cells':32,
        'execution':'single_stage_all32','quality_gate':False,'all_cells':[s['cell'] for s in schedule],
        'eligible_contexts':eligible,'eligible_contexts_role':'historical graph diagnostic only; does not gate dispatch',
        'contexts':rows,'new_physical_points':0,'new_training':0}


def source_names():
    return sorted(set(previous.source_names()) | set(ADDITIONS))


def validate_tests(path):
    cases=list(ET.parse(path).getroot().iter('testcase'))
    require(cases and not any(list(c.iter('failure'))+list(c.iter('error'))+list(c.iter('skipped')) for c in cases), 'Successful focused component tests required')
    for part in ('test_nature34_clpu_guard_first_control','test_run_clpu_guard_first_pilot','test_clpu_guard_first_audit'):
        require(any(part in c.get('classname','') for c in cases),'Missing new component '+part)
    return len(cases)


def comparison_binding():
    return {'source_run':COMPARISON_SOURCE.name,'source_completion_sha256':COMPARISON_SHA,
        'role':'matched prior candidate/scientific reference; no historical answers or cross-date time denominator'}


def bind_comparison(run):
    require(sha(COMPARISON_SOURCE/'completion_manifest.json')==COMPARISON_SHA,'Comparison root seal changed')
    manifest=base.normalize_manifest(read(COMPARISON_SOURCE/'completion_manifest.json'))
    require(read(COMPARISON_SOURCE/'prepared.json')['source_files']==previous.source_names(), 'Previous energy scientific closure differs')
    for name in COMPARISON_FILES:
        base.bound_copy(COMPARISON_SOURCE,run/'comparison_source'/name,manifest,name)
    shutil.copyfile(COMPARISON_SOURCE/'completion_manifest.json',run/'comparison_source_completion_manifest.json')
    save(run/'comparison_source_binding.json',comparison_binding())
    verify_comparison(run)


def verify_comparison(run):
    require(read(run/'comparison_source_binding.json')==comparison_binding(), 'Changed matched comparison attribution')
    require(sha(run/'comparison_source_completion_manifest.json')==COMPARISON_SHA,'Changed comparison manifest copy')
    manifest=base.normalize_manifest(read(run/'comparison_source_completion_manifest.json'))
    for name in COMPARISON_FILES:
        require(sha(run/'comparison_source'/name)==manifest[name], 'Changed comparison small input: '+name)
    audit=read(run/'comparison_source/audit/summary.json')
    require(audit['status']=='PASS' and audit['audited_cells']==32 and not audit['errors'], 'Previous full32 independent audit required')
    require(read(run/'comparison_source/contexts.json')==read(run/'contexts.json'), 'Comparison context changed')
    require(read(run/'comparison_source/stage_plan.json')['contexts']==read(run/'stage_plan.json')['contexts'], 'Common candidate generation/reference changed')
    for name in previous.source_names():
        require(sha(ROOT/name)==manifest['source/'+name], 'Previously executed source changed: '+name)


def freeze_with_comparison(run,names):
    bind_comparison(run)
    return base.freeze(run,names)


def new_run(_old_name,cfg):
    return base.create_run('clpu_guard_first_pilot',cfg)


def save_output(path,value):
    """Explicit label-only adapter before the inherited worker computes its seal."""
    path=Path(path)
    if path.name=='authorization.json':
        value={**value,'scope':'User authorized complete GUARD first then common staged window/BC3 for all8 contexts as32 cold tasks; public full-objective pruning; no training.'}
    elif path.name=='result.json':
        require(value['schema']=='clpu-staged-protected-result-v1' and value['study']==STUDY,'Unexpected inherited worker payload')
        value={**value,'schema':SCHEMA,'phase_order':['GUARD','EARLY'],
            'intermediate_beam_floor_claimed':True,
            'guard_first':read(path.parent/'EARLY/phase_result.json')['guard_first']}
    save(path,value)


def rebound(name):
    original=getattr(base,name)
    function=FunctionType(original.__code__,_PRIVATE,name,original.__defaults__,original.__closure__)
    function.__kwdefaults__=original.__kwdefaults__
    return function


def prepare(tests_path):
    return rebound('prepare')(tests_path)


def verify_execution(run):
    rebound('verify_execution')(run)
    verify_comparison(run)
    require(not any((run/name).exists() for name in ('phase_gate.json','audit_stage1','controller_stage2_started.json')),
        'Single all32 execution forbids conditional-stage artifacts')


def _worker_impl(run,cell):
    global _WORKER_ENTERED
    started=time.perf_counter()
    require(not _WORKER_ENTERED,'A physical worker may enter once'); _WORKER_ENTERED=True
    spec=read(cell/'spec.json')
    require(not (cell/'worker_started.json').exists(),'Repeated worker forbidden')
    save(cell/'worker_started.json',{'pid':os.getpid(),'parent_pid':os.getppid(),'utc':utc(),'started_monotonic':started})
    try:
        verify_execution(run)
        require(spec in read(run/'schedule.json') and cell==run/spec['cell'],'Unscheduled worker')
        ex=read(cell/'execution.json'); deadline=ex['deadline_epoch_seconds']
        require(deadline<=read(run/'execution_started.json')['deadline_epoch_seconds'] and time.time()<deadline,'Invalid task deadline')
        require(shutil.disk_usage(ROOT).free>=MIN_FREE,'Storage floor reached')
        from run_nature34_clpu_pilot import make_env
        from dnr.nature34_clpu_branch import BranchConfig
        from dnr.nature34_clpu_early import merge_results
        from dnr.nature34_clpu_guard_first_control import GuardFirstEvidenceService as SharedEvidenceService
        from nature34_clpu_tail_execution import load_fixed_policy
        model_path=run/spec['checkpoint'] if spec['checkpoint'] else None
        if model_path: require(sha(model_path)==spec['checkpoint_sha256'],'Fixed model changed')
        model_started=time.perf_counter()
        policy,generator=load_fixed_policy(model_path,spec)
        model_load_seconds=time.perf_counter()-model_started if model_path else 0.
        env=make_env(run,spec); cfg=BranchConfig(**read(run/'config.json')['branch_search'])
        compute_started=time.perf_counter()
        with SharedEvidenceService(run,cell,env,read(run/'physical_scope.json'),mode=spec['mode'],physical_budget=512,logical_budget=256) as service:
            service.deadline_monotonic=time.perf_counter()+max(0,deadline-time.time())
            init_seconds=time.perf_counter()-compute_started
            expected=next(r['generation'] for r in read(run/'stage_plan.json')['contexts'] if r['context_id']==spec['context_id'])
            phases,cps=run_routes(service,env,cfg,cell,spec,started,deadline,policy,generator,expected_generation=expected)
        if model_path: require(sha(model_path)==spec['checkpoint_sha256'],'Fixed model changed during task')
        merged=merge_results(phases['EARLY'],phases['GUARD'],protected=True)
        guard=phases['GUARD']; selected=phases[merged['selected_phase']]
        finished_tick=time.perf_counter()
        result={**merged,'version':STUDY,'schema':'clpu-staged-protected-result-v1','spec':spec,'study':STUDY,'quality_pct':selected['quality_pct'],
            'fully_certified':selected['fully_certified'],'early_quality_pct':phases['EARLY']['quality_pct'],
            'baseline_quality_pct':guard['quality_pct'],'logical_points':sum(p['logical_points'] for p in phases.values()),
            'actual_physical_points':service.physical_points,
            'numerical_entries':service.factory.stats()['returned']['numerical_solution_entries'],
            'requests':sum(p['oracle']['requests'] for p in phases.values()),
            'search_decisions':sum(p['search_decisions'] for p in phases.values()),
            'policy_decisions':phases['EARLY']['policy_decisions'],'model_load_seconds':model_load_seconds,
            'checkpoint_sha256':spec['checkpoint_sha256'],
            'stage_generation_sha256':hashlib.sha256(canonical(phases['EARLY']['stage_generation']).encode()).hexdigest(),
            'stage_generation_seconds':phases['EARLY']['stage_generation_seconds'],
            'stage_candidate_count':len((phases['EARLY']['stage_generation'] or {}).get('candidates',[])),
            'service':service.summary(),'phase_summary':{k:{name:v[name] for name in ('quality_pct','fully_certified','logical_points','actual_physical_points','search_decisions','policy_decisions','graph_expansions','verify_attempts','phase_elapsed_seconds','stage_component_seconds')} for k,v in phases.items()},
            'checkpoints':cps,'worker_elapsed_seconds':finished_tick-started,
            'preflight_elapsed_seconds':compute_started-started,'runtime_initialization_seconds':init_seconds,
            'computation_elapsed_seconds':finished_tick-compute_started,'new_training':0}
        save(cell/'result.json',result)
        save(cell/'worker_finished.json',{'status':'COMPLETE','pid':os.getpid(),'utc':utc(),
            'result_sha256':sha(cell/'result.json'),'worker_elapsed_seconds':time.perf_counter()-started})
        seal(cell); return 0
    except BaseException as exc:
        save(cell/'error.json',{'status':'STOPPED_RELIABILITY_OR_IMPLEMENTATION_ERROR','error':repr(exc),'traceback':traceback.format_exc(),
            'partial_evidence_retained':True,'no_automatic_retry':True,'pid':os.getpid()})
        return 1


def worker(run,cell):
    function=FunctionType(_worker_impl.__code__,_PRIVATE,'worker',_worker_impl.__defaults__,_worker_impl.__closure__)
    return function(run,cell)

def execute(run):
    verify_execution(run)
    require(not (run/'execution_started.json').exists(), 'No automatic rerun')
    tick,epoch=time.perf_counter(),time.time()
    save(run/'execution_started.json',{'utc':utc(),'pid':base.os.getpid(),'deadline_epoch_seconds':epoch+MAX_SECONDS,
        'started_epoch_seconds':epoch,'started_monotonic_seconds':tick,'execution':'single_stage_all32'})
    schedule=read(run/'schedule.json')
    require(read(run/'stage_plan.json')['all_cells']==[s['cell'] for s in schedule], 'All32 dispatch population changed')
    outcome=rebound('_execute_selected')(run,'phase1',schedule,[])
    complete=outcome['status']=='COMPLETE_STAGE1_PENDING_AUDIT' and len(outcome['completed'])==32
    controller={**outcome,'status':'COMPLETE_CELLS_PENDING_AUDIT' if complete else 'STOPPED_INCOMPLETE',
        'execution':'single_stage_all32','conditional_gate':False,'actual_tasks':len(outcome['dispatched']),
        'resolved_slots':len(outcome['completed']),'not_run':[], 'all_eight_context_mean_allowed':complete,
        'controller_execution_sha256':sha(run/'controller_stage1_completion.json'),
        'wall_seconds':time.perf_counter()-tick}
    save(run/'controller_completion.json',controller);save(ACTIVE,{**controller,'run':run.name})
    return 0 if complete else 1


def finalize(run):
    require(not (run/'completion_manifest.json').exists(), 'Already sealed')
    verify_execution(run)
    cache.verify_manifest(run/'audit',required=('summary.json',))
    cache.verify_manifest(run/'derived_report',required=('summary.json','all_cells.json','all_pairs.json'))
    audit=read(run/'audit/summary.json');report=read(run/'derived_report/summary.json');controller=read(run/'controller_completion.json')
    require(controller['status']=='COMPLETE_CELLS_PENDING_AUDIT' and controller['actual_tasks']==controller['resolved_slots']==32
        and not controller['errors'] and not controller['not_run'] and not controller['unknown_cost_cells'], 'All32 true successful exits required')
    require(audit['status']=='PASS' and audit['audited_cells']==audit['resolved_slots']==32 and not audit['errors']
        and audit['controller_completion_sha256']==sha(run/'controller_completion.json'), 'Full32 independent audit required')
    require(report['status']=='AUDITED_DESCRIPTIVE_GUARD_FIRST_COMMON_PILOT' and report['scheduled_cells']==report['executed_cells']==32, 'Full32 descriptive report required')
    for name in ('audit/summary.json','controller_completion.json'):
        require(report['source_bindings'][name]==sha(run/name), 'Report binding differs: '+name)
    terminal={'status':'COMPLETE_GUARD_FIRST_PILOT_AUDITED_AND_REPORTED','run':run.name,'utc':utc(),
        'audit_sha256':sha(run/'audit/summary.json'),'report_sha256':sha(run/'derived_report/summary.json'),
        'scheduled_cells':32,'executed_cells':32,'not_run_cells':0,'all_eight_context_mean_allowed':True,
        'physical_points':audit['physical_points'],'numerical_entries':audit['numerical_entries'],'new_training':0}
    save(run/'terminal_status.json',terminal);seal(run)
    save(ACTIVE,{**terminal,'status':'COMPLETE_GUARD_FIRST_PILOT_SEALED','completion_sha256':sha(run/'completion_manifest.json')})
    return terminal


def run_routes(service,env,branch_config,cell,spec,started,deadline,policy=None,generator=None,search_classes=None,expected_generation=None):
    from dnr.nature34_clpu_branch import BranchSearch
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
                    local=costs(oracle); q=quality(search)
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
                    before=time.perf_counter(); emit('decisions',{**identity,'status':'STARTED',**({'obs_compact':obs} if control else {})})
                    timers['journal_seconds']+=time.perf_counter()-before
                    require(time.time()<deadline,'Task deadline before apply')
                    before=time.perf_counter(); response=control.apply(proposal) if control else search.step(action)
                    timers['search_step_seconds']+=time.perf_counter()-before
                    require(type(response.get('done')) is bool,'Invalid step response')
                    if not control: require(math.isfinite(response['reward']),'Invalid guard reward')
                    before=time.perf_counter(); emit('decisions',{**identity,'status':'COMPLETED','response':response,'oracle_after':costs(oracle)})
                    count+=1; cp('DECISION')
                    timers['journal_seconds']+=time.perf_counter()-before
                    require(count<=(17 if control else 256),'Phase operation limit exceeded')
                result=numeric_lists(control.result() if control else search.result()); q=quality(search)
                require(search.done and search.stop_reason not in ('DEFERRED_RELIABILITY_OR_IMPLEMENTATION_ERROR',),
                    'Unfinished or erroneous EARLY route cannot establish success')
            cp('CLOSE')
            result.update(spec=ps,oracle=oracle.summary(),logical_points=oracle.points,
                actual_physical_points=oracle.physical_points,quality_pct=q,fully_certified=q is not None,
                search_decisions=count,policy_decisions=result.get('policy_decisions',0),
                graph_expansions=result.get('graph_expansions',count),verify_attempts=result.get('verify_attempts'),
                stage_component_seconds=timers,
                phase_elapsed_seconds=time.perf_counter()-phase_started,
                worker_start_seconds=phase_started-started,worker_end_seconds=time.perf_counter()-started)
            save(folder/'phase_result.json',result); results[phase]=result
            previous_logical+=oracle.points; previous_requests+=oracle.requests; previous_choices+=count
            previous_quality=checkpoints[-1]['quality_pct']
    return results,checkpoints


def _private_namespace():
    namespace=dict(vars(base))
    namespace.update(__file__=__file__,STUDY=STUDY,PROTOCOL=PROTOCOL,ACTIVE=ACTIVE,SLOTS=SLOTS,
        settings=settings,schedule_for=schedule_for,partition_schedule=partition_schedule,
        source_names=source_names,validate_tests=validate_tests,save=save_output,create_run=new_run,
        freeze=freeze_with_comparison,verify_execution=verify_execution,run_routes=run_routes)
    return namespace


_PRIVATE=_private_namespace()
_PRIVATE['build_stage_plan']=rebound('build_stage_plan')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','worker','execute','finalize'))
    parser.add_argument('--run',type=Path);parser.add_argument('--cell',type=Path);parser.add_argument('--tests',type=Path)
    args=parser.parse_args()
    if args.command=='prepare':print(prepare(args.tests.resolve()))
    elif args.command=='worker':sys.exit(worker(args.run.resolve(),args.cell.resolve()))
    elif args.command=='execute':sys.exit(execute(args.run.resolve()))
    else:print(finalize(args.run.resolve()))

