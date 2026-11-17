"""Frozen small cold comparison of unchanged P and two LazySP selectors."""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import math
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
import run_clpu_public_mechanism_pilot as old

SOURCE=ROOT/'runs/20261031T141705_478719Z_clpu_public_mechanism_pilot'
SOURCE_SHA='d9fa4705c494bb2a56f2076ca524d6c2234e91c1658796bdd71b2a2db88c62b8'
STUDY='CLPU_LAZYSP_COMPONENT_DEVELOPMENT_V1'
CONTEXTS=(0,2,4,7)
ARMS=('P','L_forward','L_reverse')
PROTOCOL='docs/clpu_lazysp_component_pilot_protocol_v1.md'
ACTIVE=ROOT/'artifacts/clpu_lazysp_pilot_active.json'
ADDITIONS=(PROTOCOL,'src/dnr/nature34_clpu_lazysp.py','tests/test_nature34_clpu_lazysp.py',
    'scripts/run_clpu_lazysp_pilot.py','scripts/audit_clpu_lazysp_pilot.py',
    'tests/test_run_clpu_lazysp_pilot.py','tests/test_audit_clpu_lazysp_pilot.py')
read,save,sha,require,canonical=old.read,old.save,old.sha,old.require,old.canonical
seal=old.seal

def utc():
    return datetime.now(timezone.utc).isoformat()

def settings():
    return dict(study=STUDY,contexts=list(CONTEXTS),arms=list(ARMS),cells=12,
        cold_workers=1,worker_seconds=180,controller_seconds=2400,seed=0,
        logical_points_per_phase=256,physical_points_per_task=512,total_physical_cap=6144,
        max_selected_plans=16,training=0,formal_test=False,statistics='all4 descriptive equal-weight mean ratios; no CI',
        retry=False,missing='retain all identities; no full-population claims when incomplete')

def environment():
    return ({d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
        {'executable':sys.executable,'version':sys.version,'sha256':sha(Path(sys.executable))})

def prepare(tests_path):
    require(tests_path is not None and tests_path.is_file(),'Current joint tests required')
    cases=list(ET.parse(tests_path).getroot().iter('testcase'))
    require(cases and not any(list(c.iter('failure'))+list(c.iter('error'))+list(c.iter('skipped')) for c in cases),'Tests failed/empty/skipped')
    for module in ('test_nature34_clpu_lazysp','test_run_clpu_lazysp_pilot','test_audit_clpu_lazysp_pilot'):
        require(any(module in c.get('classname','') for c in cases),'Missing current tests: '+module)
    require(sha(SOURCE/'completion_manifest.json')==SOURCE_SHA,'Old sealed source changed')
    original=old.base.normalize_manifest(read(SOURCE/'completion_manifest.json'))
    # Physical inputs come only from the old, already-seen sealed development run.
    names=list(old.base.ROOT_INPUTS)+['config.json','contexts.json','selected_scenarios.json',
        'selected_events.json','physical_scope.json','stage_plan.json']
    names += sorted(name for name in original if name.startswith('source_inputs/'))
    names=sorted(set(names))
    for name in names: require(sha(SOURCE/name)==original[name],'Old input changed: '+name)
    inherited=old.source_names()
    for name in inherited:
        if name.endswith('.py'):
            require(sha(ROOT/name)==original['source/'+name],'Original executable source changed: '+name)
    deps,pyenv=environment()
    require(deps==read(SOURCE/'dependencies.json') and pyenv==read(SOURCE/'python_environment.json'),'Original physical environment differs')
    require(sha(SOURCE/'resource_budget.json')==original['resource_budget.json'],'Old resource estimate changed')
    rate=read(SOURCE/'resource_budget.json')['historical_max_bytes_per_actual_point']
    forecast=math.ceil(rate*6144*1.25)
    required=14*2**30+forecast
    require(shutil.disk_usage(ROOT).free>=required,'Storage forecast plus reserve unmet')
    run=ROOT/'runs'/('20261101T'+datetime.now(timezone.utc).strftime('%H%M%S_%fZ')+'_clpu_lazysp_component')
    run.mkdir(exist_ok=False)
    for name in names:
        (run/name).parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(SOURCE/name,run/name)
    save(run/'dependencies.json',deps);save(run/'python_environment.json',pyenv)
    save(run/'environment_observation.json',{'observed_utc':utc(),'matches_source_environment':True,
        'source_dependencies_sha256':sha(SOURCE/'dependencies.json'),'source_python_sha256':sha(SOURCE/'python_environment.json')})
    shutil.copyfile(tests_path,run/'software_tests.xml')
    save(run/'settings.json',settings())
    save(run/'resource_budget.json',{'historical_max_bytes_per_actual_point':rate,
        'source_budget_sha256':sha(SOURCE/'resource_budget.json'),'total_physical_cap':6144,
        'forecast_bytes':forecast,'reserve_bytes':14*2**30,'required_free_bytes':required,
        'free_bytes_at_prepare':shutil.disk_usage(ROOT).free})
    save(run/'source_binding.json',{'source_run':SOURCE.name,'completion_sha256':SOURCE_SHA,
        'role':'fixed previously seen development inputs; no prior physical answers used'})
    old_schedule=read(SOURCE/'schedule.json')
    schedule=[]
    for turn in range(3):
        for rank,cid in enumerate(CONTEXTS):
            spec=deepcopy(next(s for s in old_schedule if s['context_id']==cid and s['arm']=='P'))
            arm=ARMS[(rank+turn)%3]
            spec.update(cell=f'episodes/{len(schedule):03d}',episode_id=len(schedule),arm=arm,route_arm=arm,
                arm_slot=arm,group=arm,phase_order=['GUARD','EARLY'],dispatch_round=turn,
                selector=None if arm=='P' else arm.removeprefix('L_'))
            if arm!='P':
                spec.update(macro_schedule='fixed_prefix_lazysp',window_variant='lazysp_global_fixed_pool_v1',
                    free_external_floor_pruning_enabled=False,external_floor_enabled=True)
            schedule.append(spec)
    save(run/'schedule.json',schedule)
    for spec in schedule:
        cell=run/spec['cell'];cell.mkdir(parents=True)
        for phase in ('GUARD','EARLY'): (cell/phase).mkdir()
        save(cell/'spec.json',spec)
    files=sorted(set(old.source_names())|set(ADDITIONS))
    hashes={}
    for name in files:
        src=ROOT/name;require(src.is_file(),'Missing reviewed source: '+name)
        dst=run/'source'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
        hashes[name]=sha(src)
    save(run/'prepared.json',{'utc':utc(),'source_files':files,'source_hashes':hashes,
        'input_hashes':{name:sha(run/name) for name in names+['dependencies.json','python_environment.json','software_tests.xml','resource_budget.json']},'settings_sha256':sha(run/'settings.json'),
        'schedule_sha256':sha(run/'schedule.json'),'physical_source_unchanged':True,
        'free_bytes':shutil.disk_usage(ROOT).free,'execution_authorized':False})
    save(ACTIVE,{'status':'PREPARED_REVIEW_REQUIRED','run':run.name,'new_physical_points':0,'new_training':0})
    return run

def verify(run):
    p=read(run/'prepared.json')
    for name,h in p['source_hashes'].items():
        require(sha(ROOT/name)==sha(run/'source'/name)==h,'Source changed after freeze: '+name)
    for name,h in p['input_hashes'].items(): require(sha(run/name)==h,'Input changed: '+name)
    require(settings()==read(run/'settings.json') and sha(run/'settings.json')==p['settings_sha256'],'Settings changed')
    require(sha(run/'schedule.json')==p['schedule_sha256'],'Schedule changed')
    deps,pyenv=environment()
    require(deps==read(run/'dependencies.json') and pyenv==read(run/'python_environment.json'),'Actual worker environment changed')

def lazy_routes(service,env,cfg,cell,spec,expected):
    from dnr.nature34_clpu_branch import BranchSearch
    from dnr.nature34_clpu_staged_control import StagedCertificationSearch
    from dnr.nature34_clpu_lazysp import LazySPCertificationSearch
    phases={}
    for phase in ('GUARD','EARLY'):
        folder=cell/phase;start=time.perf_counter()
        save(folder/'initial_state.json',env.initial.to_dict())
        with gzip.open(folder/'plans.jsonl.gz','xt',encoding='utf-8') as plans, gzip.open(folder/'decisions.jsonl.gz','xt',encoding='utf-8') as decisions:
            emit=lambda row: old.append(plans,old.numeric_lists(row))
            with service.view(phase) as oracle:
                if phase=='GUARD':
                    search=BranchSearch(env,oracle,cfg,record_callback=emit)
                    count=0
                    while not search.done:
                        action=search.rule_action('beam')
                        before=old.costs(oracle)
                        old.append(decisions,{'transition_id':count,'phase_choice':count,'phase':phase,
                            'method':'beam','action':action,'oracle_before':before,'status':'STARTED'})
                        response=search.step(action)
                        old.append(decisions,{'transition_id':count,'phase_choice':count,'phase':phase,
                            'method':'beam','action':action,'oracle_before':before,'status':'COMPLETED',
                            'response':response,'oracle_after':old.costs(oracle)})
                        count+=1;require(count<=256,'GUARD choices exceeded')
                    result=old.numeric_lists(search.result())
                else:
                    bootstrap=StagedCertificationSearch(env,oracle,cfg,record_callback=emit)
                    require(bootstrap.stage_generation is not None,'Bootstrap incomplete')
                    pool=old.numeric_lists(bootstrap.stage_generation)
                    require(pool==expected,'Candidates differ from frozen graph-only inventory')
                    save(folder/'candidate_pool.json',{'generation':pool,'generation_sha256':hashlib.sha256(canonical(pool).encode()).hexdigest()})
                    lazy=LazySPCertificationSearch(bootstrap,guard_result=phases['GUARD'],
                        guard_reference={'phase':'GUARD','file':'GUARD/phase_result.json','sha256':sha(cell/'GUARD/phase_result.json')},
                        selector=spec['selector'],max_selected_plans=16)
                    try:
                        result=old.numeric_lists(lazy.run())
                    except BaseException:
                        save(folder/'lazy_error_state.json',old.numeric_lists(lazy.result()))
                        raise
                    count=len(result['lazysp']['journal'])
                plan=result.get('best_plan')
                q=100*(plan['validation']['energy_kwh']/(env.total_kw*env.horizon*env.slot_seconds/3600)) if plan else None
            result.update(spec=old.phase_spec(spec,phase),oracle=oracle.summary(),logical_points=oracle.points,
                actual_physical_points=oracle.physical_points,quality_pct=q,fully_certified=q is not None,
                search_decisions=count,policy_decisions=0,phase_elapsed_seconds=time.perf_counter()-start)
            save(folder/'phase_result.json',result);phases[phase]=result
    return phases

def worker(run,cell):
    start=time.perf_counter()
    require(not (cell/'worker_started.json').exists(),'No repeated worker')
    save(cell/'worker_started.json',{'utc':utc(),'pid':os.getpid()})
    try:
        verify(run);spec=read(cell/'spec.json')
        require(spec in read(run/'schedule.json') and run/spec['cell']==cell,'Unscheduled cell')
        from run_nature34_clpu_pilot import make_env
        from dnr.nature34_clpu_branch import BranchConfig
        from dnr.nature34_clpu_guard_first_control import GuardFirstEvidenceService
        from dnr.nature34_clpu_early import merge_results
        env=make_env(run,spec);cfg=BranchConfig(**read(run/'config.json')['branch_search'])
        expected=next(c['generation'] for c in read(run/'stage_plan.json')['contexts'] if c['context_id']==spec['context_id'])
        deadline=read(cell/'execution.json')['deadline_epoch_seconds']
        with GuardFirstEvidenceService(run,cell,env,read(run/'physical_scope.json'),mode='shared',physical_budget=512,logical_budget=256) as service:
            service.deadline_monotonic=time.perf_counter()+max(0,deadline-time.time())
            if spec['arm']=='P':
                phases,_=old.run_routes(service,env,cfg,cell,spec,start,deadline,expected_generation=expected)
            else: phases=lazy_routes(service,env,cfg,cell,spec,expected)
        merged=merge_results(phases['EARLY'],phases['GUARD'],protected=True)
        selected=phases[merged['selected_phase']]
        require(merged['guard_audit']['guard_established'],'Missing full baseline')
        result={**merged,'study':STUDY,'spec':spec,'quality_pct':selected['quality_pct'],
            'baseline_quality_pct':phases['GUARD']['quality_pct'],'actual_physical_points':service.physical_points,
            'logical_points':sum(v['logical_points'] for v in phases.values()),
            'requests':sum(v['oracle']['requests'] for v in phases.values()),
            'numerical_entries':service.factory.stats()['returned']['numerical_solution_entries'],
            'service':service.summary(),'worker_elapsed_seconds':time.perf_counter()-start,
            'models_used':0,'new_training':0}
        save(cell/'result.json',result)
        save(cell/'worker_finished.json',{'status':'COMPLETE','utc':utc(),'pid':os.getpid(),'result_sha256':sha(cell/'result.json')})
        seal(cell);return 0
    except BaseException as exc:
        save(cell/'error.json',{'status':'STOPPED_ERROR','error':repr(exc),'traceback':traceback.format_exc(),'no_retry':True})
        return 1

def execute(run):
    verify(run)
    gate=read(run/'preflight.json')
    require(gate['status']=='PASS' and gate['prepared_sha256']==sha(run/'prepared.json'),'Independent preflight required')
    require(not (run/'execution_started.json').exists(),'No automatic retry')
    start=time.perf_counter();deadline=time.time()+2400
    save(run/'execution_started.json',{'utc':utc(),'deadline_epoch_seconds':deadline})
    rows=[]
    for spec in read(run/'schedule.json'):
        cell=run/spec['cell'];wall=min(180,deadline-time.time())
        if wall<=0: break
        require(shutil.disk_usage(ROOT).free>=14*2**30,'Storage reserve violated')
        begin=time.time();tick=time.perf_counter()
        save(cell/'execution.json',{'started_epoch_seconds':begin,'deadline_epoch_seconds':begin+wall})
        with (run/(cell.name+'_stdout.log')).open('x',encoding='utf-8') as out,(run/(cell.name+'_stderr.log')).open('x',encoding='utf-8') as err:
            proc=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-B',str(Path(__file__)), 'worker','--run',str(run),'--cell',str(cell)],cwd=ROOT,stdout=out,stderr=err)
            try: code=proc.wait(timeout=wall); timed_out=False
            except subprocess.TimeoutExpired:
                proc.kill();code=proc.wait();timed_out=True
        row={'cell':spec['cell'],'arm':spec['arm'],'context_id':spec['context_id'],'pid':proc.pid,
            'exit_code':code,'timed_out':timed_out,'started_epoch_seconds':begin,'finished_epoch_seconds':time.time(),
            'process_wall_seconds':time.perf_counter()-tick}
        rows.append(row);save(run/'controller_progress.json',rows)
        print(json.dumps(row),flush=True)
        if code or timed_out: break
    status='COMPLETE_PENDING_AUDIT' if len(rows)==12 and all(x['exit_code']==0 for x in rows) else 'STOPPED_INCOMPLETE'
    summary={'status':status,'exits':rows,'scheduled_cells':12,'wall_seconds':time.perf_counter()-start,
        'not_run':[s['cell'] for s in read(run/'schedule.json') if s['cell'] not in {r['cell'] for r in rows}],
        'new_training':0}
    save(run/'controller_completion.json',summary);save(ACTIVE,{**summary,'run':run.name})
    return 0 if status=='COMPLETE_PENDING_AUDIT' else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','execute','worker'))
    p.add_argument('--run',type=Path);p.add_argument('--cell',type=Path);p.add_argument('--tests',type=Path);a=p.parse_args()
    if a.command=='prepare':print(prepare(a.tests))
    elif a.command=='worker':sys.exit(worker(a.run.resolve(),a.cell.resolve()))
    else:sys.exit(execute(a.run.resolve()))
