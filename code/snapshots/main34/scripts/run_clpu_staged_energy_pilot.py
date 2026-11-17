"""One complete 32-task rerun with a visible complete-energy window rule.

The frozen staged runner's preparation, worker, and pair-draining implementation
are reused with a private globals dictionary. The imported module is never
patched. Only the explicit boundaries below change attribution/configuration;
run_routes differs solely in the EARLY controller import.
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
import run_clpu_staged_pilot as base

read, require, sha, canonical = base.read, base.require, base.sha, base.canonical
save, seal, utc, cache = base.save, base.seal, base.utc, base.cache
SOURCE, SOURCE_SHA = base.SOURCE, base.SOURCE_SHA
MODELS_SOURCE, MODELS_SHA = base.MODELS_SOURCE, base.MODELS_SHA
GRAPH_PROFILE, GRAPH_PROFILE_SHA = base.GRAPH_PROFILE, base.GRAPH_PROFILE_SHA
STUDY = 'CLPU_STAGED_ENERGY_COMMON_PILOT_V1'
SCHEMA = 'clpu-staged-energy-protected-result-v1'
PROTOCOL = 'docs/clpu_staged_energy_pilot_protocol_v1.md'
ACTIVE = ROOT/'artifacts/clpu_staged_energy_pilot_active.json'
COMPARISON_SOURCE = ROOT/'runs/20261031T085600_616395Z_clpu_staged_pilot'
COMPARISON_SHA = '0d288959bece92f5e274ce43e0c2d29520fe15094dbc8663ebcc3a8e3a3ceefb'
SEEDS, CELLS = (17,41,73), 32
SLOTS = ('energy_window', *(f'staged_bc{s}' for s in SEEDS))
MAX_SECONDS, WORKER_SECONDS, TOTAL_CAP = 3600, 900, 16384
MIN_FREE, RESERVE = 12*2**30, 2*2**30
ADDITIONS = (PROTOCOL, 'src/dnr/nature34_clpu_staged_energy_control.py',
    'scripts/run_clpu_staged_energy_pilot.py', 'scripts/audit_clpu_staged_energy_pilot.py',
    'tests/test_nature34_clpu_staged_energy_control.py', 'tests/test_run_clpu_staged_energy_pilot.py',
    'tests/test_clpu_staged_energy_audit.py')
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
        quality_gate=False, ranking_change='current visible complete graph_energy_kwh descending then branch_id ascending; no prefix-depth restriction',
        comparison='only current cold32 process times; previous depth-priority run is a mechanism/semantic reference',
        same_optimized_window_for_all_rankers=True)
    return value


def schedule_for(contexts, models):
    # Reuse the frozen hash domains and Latin rotation, replacing only the arm
    # label in SLOTS. No post-result reordering or seed selection is permitted.
    return rebound('schedule_for')(contexts, models)


def partition_schedule(schedule, rows):
    require(len(schedule)==32 and len({s['cell'] for s in schedule})==32, 'All32 actual tasks required')
    require(len(rows)==8 and [r['context_id'] for r in rows]==list(range(8)), 'Original8 graph inventories required')
    eligible=[r['context_id'] for r in rows if any(c['graph_energy_kwh']>r['reference_energy_kwh'] for c in r['generation']['candidates'])]
    return {'schema':'clpu-staged-energy-fixed-population-v1','original_contexts':8,'scheduled_cells':32,
        'execution':'single_stage_all32','quality_gate':False,'all_cells':[s['cell'] for s in schedule],
        'eligible_contexts':eligible,'eligible_contexts_role':'historical graph diagnostic only; does not gate dispatch',
        'contexts':rows,'new_physical_points':0,'new_training':0}


def source_names():
    return sorted(set(base.source_names()) | set(ADDITIONS))


def validate_tests(path):
    cases=list(ET.parse(path).getroot().iter('testcase'))
    require(cases and not any(list(c.iter('failure'))+list(c.iter('error'))+list(c.iter('skipped')) for c in cases), 'Successful focused component tests required')
    for part in ('test_nature34_clpu_staged_energy_control','test_run_clpu_staged_energy_pilot','test_clpu_staged_energy_audit'):
        require(any(part in c.get('classname','') for c in cases),'Missing new component '+part)
    return len(cases)


def comparison_binding():
    return {'source_run':COMPARISON_SOURCE.name,'source_completion_sha256':COMPARISON_SHA,
        'role':'matched prior candidate/scientific reference; no historical answers or cross-date time denominator'}


def bind_comparison(run):
    require(sha(COMPARISON_SOURCE/'completion_manifest.json')==COMPARISON_SHA,'Comparison root seal changed')
    manifest=base.normalize_manifest(read(COMPARISON_SOURCE/'completion_manifest.json'))
    require(read(COMPARISON_SOURCE/'prepared.json')['source_files']==base.source_names(), 'Previous staged scientific closure differs')
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
    for name in base.source_names():
        require(sha(ROOT/name)==manifest['source/'+name], 'Previously executed source changed: '+name)


def freeze_with_comparison(run,names):
    bind_comparison(run)
    return base.freeze(run,names)


def new_run(_old_name,cfg):
    return base.create_run('clpu_staged_energy_pilot',cfg)


def save_output(path,value):
    """Explicit label-only adapter before the inherited worker computes its seal."""
    path=Path(path)
    if path.name=='authorization.json':
        value={**value,'scope':'User authorized all8 contexts x energy-window and BC3 fixed seeds as32 actual cold tasks, unchanged common candidates and complete Beam guard; no quality gate, no training.'}
    elif path.name=='result.json':
        require(value['schema']=='clpu-staged-protected-result-v1' and value['study']==STUDY,'Unexpected inherited worker payload')
        value={**value,'schema':SCHEMA}
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


def worker(run,cell):
    return rebound('worker')(run,cell)


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
    require(report['status']=='AUDITED_DESCRIPTIVE_STAGED_ENERGY_COMMON_PILOT' and report['scheduled_cells']==report['executed_cells']==32, 'Full32 descriptive report required')
    for name in ('audit/summary.json','controller_completion.json'):
        require(report['source_bindings'][name]==sha(run/name), 'Report binding differs: '+name)
    terminal={'status':'COMPLETE_STAGED_ENERGY_PILOT_AUDITED_AND_REPORTED','run':run.name,'utc':utc(),
        'audit_sha256':sha(run/'audit/summary.json'),'report_sha256':sha(run/'derived_report/summary.json'),
        'scheduled_cells':32,'executed_cells':32,'not_run_cells':0,'all_eight_context_mean_allowed':True,
        'physical_points':audit['physical_points'],'numerical_entries':audit['numerical_entries'],'new_training':0}
    save(run/'terminal_status.json',terminal);seal(run)
    save(ACTIVE,{**terminal,'status':'COMPLETE_STAGED_ENERGY_PILOT_SEALED','completion_sha256':sha(run/'completion_manifest.json')})
    return terminal


def run_routes(service,env,branch_config,cell,spec,started,deadline,policy=None,generator=None,search_classes=None,expected_generation=None):
    from dnr.nature34_clpu_branch import BranchSearch
    from dnr.nature34_clpu_staged_control import StagedCertificationSearch
    from dnr.nature34_clpu_staged_energy_control import EnergyStagedMacroController as StagedMacroController
    classes=search_classes or {'EARLY':StagedCertificationSearch,'GUARD':BranchSearch}
    results,checkpoints={},[]
    previous_logical=previous_requests=previous_choices=0
    previous_quality=None
    with (cell/'cost_checkpoints.jsonl').open('x',encoding='utf-8') as cpstream:
        for phase in ('EARLY','GUARD'):
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
                search=classes[phase](env,oracle,branch_config,record_callback=lambda r:emit('plans',r))
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
                        'guard_established':phase=='GUARD' and kind=='CLOSE' and q is not None,
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
