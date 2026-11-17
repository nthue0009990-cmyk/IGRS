"""One frozen eight-context P/direct-Beam comparison; no training or retries."""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
sys.dont_write_bytecode=True
import run_clpu_public_mechanism_pilot as prior
from nature34_experiment_support import create_run,freeze,dependencies
from dnr.nature34_clpu_branch import BranchConfig,BranchSearch

read,save,sha,require=prior.read,prior.save,prior.sha,prior.require
canonical,numeric_lists,append=prior.canonical,prior.numeric_lists,prior.append
SOURCE=ROOT/'runs/20261031T141705_478719Z_clpu_public_mechanism_pilot'
SOURCE_SHA='d9fa4705c494bb2a56f2076ca524d6c2234e91c1658796bdd71b2a2db88c62b8'
PROTOCOL='docs/clpu_direct_beam_budget_protocol_v1.md'
ACTIVE=ROOT/'artifacts/clpu_direct_beam_active.json'
WORKER_SECONDS,TOTAL_SECONDS=240,3900
MIN_FREE=12*2**30
NEW_SOURCES=[PROTOCOL,'scripts/run_clpu_direct_beam.py','scripts/audit_clpu_direct_beam.py',
    'scripts/clpu_direct_beam_audit_helpers.py','tests/test_clpu_direct_beam.py']


def beam_config():
    return {**asdict(BranchConfig()),'point_budget':512,'max_generated_candidates':10000,
        'max_checked_candidates':512,'max_branch_expansions':512}


def schedule_for(contexts):
    require(len(contexts)==8 and {c['context_id'] for c in contexts}==set(range(8)),'Original eight required')
    rows=[]
    for c in sorted(contexts,key=lambda x:x['context_id']):
        for arm in (('P','BEAM') if c['context_id']%2==0 else ('BEAM','P')):
            rows.append({**c,'cell':f'episodes/{len(rows):03d}','episode_id':len(rows),'arm':arm,
                'method':'beam','seed':0,'checkpoint':None,'checkpoint_sha256':None,
                'mode':'shared','phase_order':['GUARD','EARLY'] if arm=='P' else ['GUARD'],
                'point_budget':256 if arm=='P' else 512,'total_logical_point_budget':512,
                'physical_point_budget':512,'macro_schedule':'staged','representation':'raw'})
    return rows


def phase_spec(spec,name):
    if spec['arm']=='P':return prior.phase_spec(spec,name)
    return {**spec,'parent_cell':spec['cell'],'cell':spec['cell']+'/'+name,'search_phase':name,
        'logical_point_budget':512,'physical_point_budget':512}


def verify(run):
    prepared=read(run/'prepared.json')
    require(sha(run/'prephysical_freeze_manifest.json')==prepared['freeze_sha256'],'Freeze changed')
    for name,digest in read(run/'prephysical_freeze_manifest.json').items():
        require(sha(run/name)==digest,'Frozen input changed: '+name)
    for name in prepared['source_files']:
        require(sha(ROOT/name)==sha(run/'source'/name),'Executing source changed: '+name)
    require(read(run/'dependencies.json')==dependencies(),'Dependency change')
    cfg=read(run/'config.json')
    require(cfg['branch_search']==asdict(BranchConfig()) and cfg['direct_beam_search']==beam_config(),'Resource contract changed')
    require(read(run/'schedule.json')==schedule_for(read(run/'contexts.json')),'Schedule changed')
    require(prior.cache.physical_scope(run,prepared['source_files'])==read(run/'physical_scope.json'),'Physical scope changed')


def prepare(tests_path):
    cases=list(ET.parse(tests_path).getroot().iter('testcase'))
    require(cases and not any(list(c.iter('failure'))+list(c.iter('error'))+list(c.iter('skipped')) for c in cases),'Software checks not passed')
    require(any('test_clpu_direct_beam' in c.get('classname','') for c in cases),'Missing new checks')
    require(sha(SOURCE/'completion_manifest.json')==SOURCE_SHA,'Original source seal changed')
    sm=prior.base.normalize_manifest(read(SOURCE/'completion_manifest.json'))
    cfg=deepcopy(read(SOURCE/'config.json'))
    cfg.update(staged_study='CLPU_DIRECT_BEAM_TOTAL_BUDGET_V1',direct_beam_search=beam_config())
    cfg['staged_pilot']={'cells':16,'models_used':0,'new_training':0,'max_workers':1,
        'physical_budget_per_task':512,'total_logical_budget_per_task':512,'worker_seconds':WORKER_SECONDS,
        'controller_seconds':TOTAL_SECONDS,'physical_budget_total':8192,'numerical_entries_total':32816}
    run=create_run('clpu_direct_beam',cfg)
    try:
        for name in (*prior.base.ROOT_INPUTS,'population.json','selected_scenarios.json','selected_events.json','training_contexts.json','contexts.json'):
            prior.base.bound_copy(SOURCE,run/name,sm,name)
        for name in sm:
            if name.startswith('source_inputs/'):prior.base.bound_copy(SOURCE,run/name,sm,name)
        require(read(run/'dependencies.json')==read(SOURCE/'dependencies.json'),'Physical environment changed')
        names=sorted(set(prior.source_names()+NEW_SOURCES))
        for name in prior.source_names():require(sha(ROOT/name)==sm['source/'+name],'Inherited code changed: '+name)
        originals=read(SOURCE/'stage_plan.json')['contexts']
        inventories=[{'context_id':r['context_id'],'generation':r['generation']} for r in originals]
        save(run/'candidate_inventory.json',inventories)
        schedule=schedule_for(read(run/'contexts.json'));save(run/'schedule.json',schedule)
        for spec in schedule:
            save(run/spec['cell']/'spec.json',spec)
            for name in spec['phase_order']:save(run/spec['cell']/name/'spec.json',phase_spec(spec,name))
        rates=[]
        for spec in read(SOURCE/'schedule.json'):
            folder=SOURCE/spec['cell'];r=read(folder/'result.json')
            rates.append(sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())/max(1,r['actual_physical_points']))
        forecast=int(max(rates)*8192*1.25)+1
        storage={'physical_cap':8192,'forecast_bytes':forecast,'minimum_free_bytes':MIN_FREE,
            'report_reserve_bytes':2*2**30,'required_free_bytes':forecast+MIN_FREE+2*2**30,
            'free_bytes_at_prepare':shutil.disk_usage(ROOT).free}
        require(storage['free_bytes_at_prepare']>=storage['required_free_bytes'],'Insufficient storage')
        save(run/'resource_budget.json',storage)
        save(run/'source_binding.json',{'source_run':SOURCE.name,'completion_sha256':SOURCE_SHA,
            'historical_numerical_answers_loaded':False,'role':'frozen development inputs and graph-only candidate inventory'})
        save(run/'authorization.json',{'text':'User authorized original8 x P/direct same-total-budget Beam, at most16 cold tasks; freeze first; no training or physics changes; retain failures.',
            'models_used':0,'no_automatic_retry':True,'new_training':0})
        shutil.copyfile(ROOT/PROTOCOL,run/'protocol.md');shutil.copyfile(tests_path,run/'software_tests.xml')
        save(run/'physical_scope.json',prior.cache.physical_scope(run,names))
        freeze(run,names)
        save(ACTIVE,{'run':run.name,'status':'FROZEN_NOT_STARTED','tasks':16})
        return run
    except BaseException as exc:
        save(run/'preparation_error.json',{'error':repr(exc),'traceback':traceback.format_exc(),'new_physical_queries':0});raise


def beam_route(service,env,cfg,cell,spec,started,deadline):
    folder=cell/'GUARD';began=time.perf_counter();cps=[]
    save(folder/'initial_state.json',env.initial.to_dict())
    with gzip.open(folder/'plans.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as plans, gzip.open(folder/'decisions.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as decisions, (cell/'cost_checkpoints.jsonl').open('x',encoding='utf-8') as stream:
        with service.view('GUARD') as oracle:
            search=BranchSearch(env,oracle,cfg,record_callback=lambda r:append(plans,numeric_lists(r)))
            count=0
            def checkpoint(kind):
                q=prior.quality(search)
                row={'checkpoint_index':len(cps),'phase':'GUARD','kind':kind,'phase_choices':count,
                    'logical_points':oracle.points,'physical_points':service.physical_points,
                    'numerical_entries':service.factory.stats()['returned']['numerical_solution_entries'],
                    'quality_pct':q,'worker_elapsed_seconds':time.perf_counter()-started}
                cps.append(row);append(stream,row)
            checkpoint('INITIAL')
            while not search.done:
                require(time.time()<deadline,'Task deadline')
                action=search.rule_action('beam')
                identity={'transition_id':count,'phase_choice':count,'phase':'GUARD','method':'beam',
                    'action':action,'oracle_before':prior.costs(oracle)}
                append(decisions,{**identity,'status':'STARTED'})
                response=search.step(action)
                append(decisions,numeric_lists({**identity,'status':'COMPLETED','response':response,'oracle_after':prior.costs(oracle)}))
                count+=1;checkpoint('DECISION')
                require(count<=cfg.max_branch_expansions,'Expansion budget exceeded')
            result=numeric_lists(search.result());q=prior.quality(search)
        checkpoint('CLOSE')
    result.update(spec=phase_spec(spec,'GUARD'),oracle=oracle.summary(),logical_points=oracle.points,
        actual_physical_points=oracle.physical_points,quality_pct=q,fully_certified=q is not None,
        search_decisions=count,policy_decisions=0,graph_expansions=count,verify_attempts=None,
        phase_elapsed_seconds=time.perf_counter()-began)
    save(folder/'phase_result.json',result)
    return {'GUARD':result},cps


def worker(run,cell):
    started=time.perf_counter();spec=read(cell/'spec.json')
    require(not (cell/'worker_started.json').exists(),'No repeated task')
    save(cell/'worker_started.json',{'pid':os.getpid(),'parent_pid':os.getppid(),'utc':prior.utc()})
    try:
        verify(run);require(spec in read(run/'schedule.json') and cell==run/spec['cell'],'Unscheduled worker')
        deadline=read(cell/'execution.json')['deadline_epoch_seconds']
        require(time.time()<deadline and shutil.disk_usage(ROOT).free>=MIN_FREE,'Deadline/storage')
        from run_nature34_clpu_pilot import make_env
        from dnr.nature34_clpu_guard_first_control import GuardFirstEvidenceService
        from dnr.nature34_clpu_early import merge_results
        env=make_env(run,spec);cfg=BranchConfig(**(read(run/'config.json')['branch_search'] if spec['arm']=='P' else beam_config()))
        with GuardFirstEvidenceService(run,cell,env,read(run/'physical_scope.json'),mode='shared',physical_budget=512,logical_budget=spec['point_budget']) as service:
            service.deadline_monotonic=time.perf_counter()+max(0,deadline-time.time())
            if spec['arm']=='P':
                expected=next(r['generation'] for r in read(run/'candidate_inventory.json') if r['context_id']==spec['context_id'])
                phases,cps=prior.run_routes(service,env,cfg,cell,spec,started,deadline,expected_generation=expected)
                merged=merge_results(phases['EARLY'],phases['GUARD'],protected=True)
                selected=merged['selected_phase']
            else:
                phases,cps=beam_route(service,env,cfg,cell,spec,started,deadline);selected='GUARD'
        best=phases[selected]
        require(best['fully_certified'] and best['best_plan'] is not None,'No complete final certificate')
        result={'study':'CLPU_DIRECT_BEAM_TOTAL_BUDGET_V1','spec':spec,'selected_phase':selected,
            'best_plan':best['best_plan'],'quality_pct':best['quality_pct'],'fully_certified':True,
            'energy_kwh':best['best_plan']['validation']['energy_kwh'],
            'baseline_energy_kwh':phases['GUARD']['best_plan']['validation']['energy_kwh'],
            'logical_points':service.summary()['logical_points'],'actual_physical_points':service.physical_points,
            'numerical_entries':service.factory.stats()['returned']['numerical_solution_entries'],
            'requests':sum(p['oracle']['requests'] for p in phases.values()),'service':service.summary(),
            'phase_stop_reasons':{name:p['stop_reason'] for name,p in phases.items()},'checkpoints':cps,
            'worker_elapsed_seconds':time.perf_counter()-started,'models_used':0,'new_training':0}
        require(result['logical_points']<=512 and result['numerical_entries']==3+4*result['actual_physical_points'],'Full cost mismatch')
        save(cell/'result.json',result)
        save(cell/'worker_finished.json',{'status':'COMPLETE','pid':os.getpid(),'result_sha256':sha(cell/'result.json'),'worker_elapsed_seconds':time.perf_counter()-started})
        prior.seal(cell);return 0
    except BaseException as exc:
        save(cell/'error.json',{'status':'STOPPED','error':repr(exc),'traceback':traceback.format_exc(),'partial_evidence_retained':True,'no_retry':True});return 1


def execute(run):
    verify(run)
    pre=read(run/'preflight/summary.json')
    require(pre['status']=='PASS' and pre['freeze_sha256']==sha(run/'prephysical_freeze_manifest.json'),'No matching preflight')
    require(not (run/'execution_started.json').exists(),'No retry')
    schedule=read(run/'schedule.json');tick=time.perf_counter();start=time.time();exits=[]
    save(run/'execution_started.json',{'pid':os.getpid(),'started_epoch_seconds':start,'deadline_epoch_seconds':start+TOTAL_SECONDS,'max_tasks':16})
    for spec in schedule:
        if time.time()>=start+TOTAL_SECONDS or shutil.disk_usage(ROOT).free<MIN_FREE:break
        cell=run/spec['cell'];epoch=time.time();deadline=min(start+TOTAL_SECONDS,epoch+WORKER_SECONDS)
        execution={'controller_pid':os.getpid(),'spec_sha256':sha(cell/'spec.json'),'started_epoch_seconds':epoch,'deadline_epoch_seconds':deadline}
        save(cell/'execution.json',execution)
        with (cell/'stdout.log').open('xb') as out,(cell/'stderr.log').open('xb') as err:
            began=time.perf_counter()
            process=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-X','utf8','-B',str(Path(__file__).resolve()),'--worker',str(run),str(cell)],
                cwd=ROOT,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),
                env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
            timed_out=False
            try:code=process.wait(timeout=max(.01,deadline-time.time()))
            except subprocess.TimeoutExpired:
                timed_out=True;process.kill();code=process.wait()
            row={'cell':spec['cell'],'context_id':spec['context_id'],'arm':spec['arm'],'pid':process.pid,
                'exit_code':code,'timed_out':timed_out,'started_epoch_seconds':epoch,'finished_epoch_seconds':time.time(),
                'process_wall_seconds':time.perf_counter()-began}
            exits.append(row);save(run/'controller_exits.json',exits)
        save(ACTIVE,{'run':run.name,'status':'RUNNING' if code==0 else 'STOPPED','completed':sum(r['exit_code']==0 for r in exits),'dispatched':len(exits)})
        print(canonical(row),flush=True)
        if code!=0 or timed_out or not (cell/'result.json').exists():break
    complete=len(exits)==16 and all(r['exit_code']==0 and not r['timed_out'] for r in exits)
    summary={'status':'COMPLETE_PENDING_AUDIT' if complete else 'STOPPED_INCOMPLETE','exits':exits,
        'not_run':[s for s in schedule[len(exits):]],'wall_seconds':time.perf_counter()-tick,'actual_tasks':len(exits),'new_training':0}
    save(run/'controller_completion.json',summary);save(ACTIVE,{**summary,'run':run.name})
    return 0 if complete else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',type=Path);parser.add_argument('--execute',type=Path);parser.add_argument('--worker',nargs=2,type=Path)
    args=parser.parse_args()
    if args.worker:raise SystemExit(worker(*(p.resolve() for p in args.worker)))
    if args.prepare:print(prepare(args.prepare.resolve()))
    elif args.execute:raise SystemExit(execute(args.execute.resolve()))
