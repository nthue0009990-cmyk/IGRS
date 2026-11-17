"""One authorized NC13 acceptance worker; prepare and audit are solver-free."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
sys.dont_write_bytecode = True
REF = ROOT/'runs/20261019T021340_566784Z_nature13_curtailment_validation'
REF_SHA = '104e888c005f4e94051eca922c8fcc5b4826f24a47e7d8f270cc36ae0019ace8'
STATIC = ROOT/'artifacts/nature13_clpu_adaptation_20261101/static'
PROTOCOL = 'docs/nature13_clpu_acceptance_protocol_v1.md'
FAULTS = ('line.632633', 'line.632670', 'line.670671')
ACTIVE = ROOT/'artifacts/nature13_clpu_acceptance_active.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def append(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')
    stream.flush()
    os.fsync(stream.fileno())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def settings():
    return {'study':'NATURE13_CLPU_BOUNDED_ACCEPTANCE_V1', 'max_queries':16,
            'max_numerical_entries':256, 'worker_seconds':240, 'workers':1,
            'retry':False, 'training_steps':0, 'formal_test':False, 'seed':0,
            'faults':list(FAULTS), 'acceptance_event_id':3, 'alpha':0.375,
            'algorithms':[0,1], 'horizon':30, 'slot_seconds':60, 'sample_seconds':30,
            'authorization':'User explicitly selected bounded plan; 18-task execution not authorized by this entry',
            'nonfinite_raw_serialization':'Python JSON NaN/Infinity retained in gzip raw only; never filled with zero'}


def environment():
    return {'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
            'python':sys.version, 'executable':sys.executable, 'executable_sha256':sha(Path(sys.executable))}


def make_schedule():
    from prepare_nature13_clpu_static import build_environment, load_inputs
    _, loads, _, _, _, _ = load_inputs()
    old_config = read(REF/'config.json')
    rows = []
    for case, ref_ids in zip(old_config['cases'], ((12,14),(48,50))):
        for algorithm, ref_id in enumerate(ref_ids):
            rows.append({'group':'reference', 'group_id':case['name'], 'source_point_id':ref_id,
                'request':{'action':case['action'], 'case':{'name':case['name'],'fault_lines':case['fault_lines']},
                    'alpha':.375, 'algorithm':algorithm, 'multipliers':{name:1. for name in loads},
                    'time_minutes':0., 'state_identity':None}})
    for group in ('initial','peak'):
        for fault in FAULTS:
            env = build_environment(ROOT, fault, 3)
            state = env.initial if group == 'initial' else env.advance(env.initial,11)
            for algorithm in (0,1):
                rows.append({'group':group, 'group_id':fault, 'fault_line':fault,
                             'request':env.snapshot_request(state,0,algorithm=algorithm)})
    for qid, row in enumerate(rows):
        row['query_id'] = qid
    require(len(rows)==16,'Expected fixed 16-point schedule')
    require(all(r['request']['action'][4:]==[1]*7 for r in rows[4:]),'Initial/peak logical readiness changed')
    return rows


def input_archive_name(index,path):
    """Short copy names avoid Windows path limits; full identity stays in map."""
    return f'inputs/{index:03d}_{Path(path).name}'


def prepare(tests):
    require(tests and tests.is_file(),'Current pure software test XML required')
    cases = list(ET.parse(tests).getroot().iter('testcase'))
    require(cases and not any(list(c.iter('failure'))+list(c.iter('error'))+list(c.iter('skipped')) for c in cases),
            'Tests failed, skipped, or empty')
    for module in ('test_nature13_clpu_static','test_nature13_clpu_physical',
                   'test_nature13_clpu_acceptance_comparison','test_run_nature13_clpu_acceptance'):
        require(any(module in c.get('classname','') for c in cases),'Missing current tests: '+module)
    require(sha(REF/'completion_manifest.json')==REF_SHA,'Old reference seal differs')
    for name, digest in read(REF/'completion_manifest.json').items():
        require(sha(REF/name)==digest,'Old reference altered: '+name)
    static = read(STATIC/'summary.json')
    require(static['status']=='STATIC_NONEMPTY_ORIGINAL_POOLS' and len(static['contexts'])==6,'Static pool gate failed')
    require(all(r['counts']['unique_legal_candidates']>0 for r in static['contexts']),'A fixed context has no original candidates')
    run = ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_nature13_clpu_acceptance')
    run.mkdir(exist_ok=False)
    save(run/'settings.json',settings())
    save(run/'physical_config.json',read(REF/'config.json'))
    save(run/'schedule.json',make_schedule())
    save(run/'environment.json',environment())
    shutil.copyfile(tests,run/'software_tests.xml')
    with gzip.open(REF/'points.jsonl.gz','rt',encoding='utf-8') as f:
        all_old = [json.loads(line) for line in f]
    references = [r for r in all_old if r['id'] in (12,14,48,50)]
    require(len(references)==4,'Four original alpha=.375 reference rows required')
    for row in references:
        repeat = next(r for r in all_old if r['id']==row['id']+1)
        require(row['raw_outputs']==repeat['raw_outputs'],'Original reference repeats differ')
    with gzip.open(run/'reference_points.json.gz','wt',encoding='utf-8') as f:
        json.dump(references,f,ensure_ascii=False,allow_nan=True)
    save(run/'reference_binding.json',{'run':REF.name,'manifest_sha256':REF_SHA,
        'ids':[12,14,48,50], 'purpose':'Acceptance regression only; never free answers for future search tasks'})
    names = [p.relative_to(ROOT).as_posix() for p in (ROOT/'src/dnr').glob('*.py')]
    names += ['scripts/run_nature13_clpu_acceptance.py','scripts/audit_nature13_clpu_acceptance.py',
        'scripts/audit_nature13_curtailment_validation.py','scripts/prepare_nature13_clpu_static.py',
        'scripts/audit_restoration_topology_choices.py',PROTOCOL,
        'docs/nature13_clpu_bounded_adaptation_proposal_v1.md']
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'tests').glob('*nature13*clpu*.py')]
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'tests').glob('*nature13*acceptance*.py')]
    hashes = {}
    for name in sorted(set(names)):
        source = ROOT/name; require(source.is_file(),'Missing source: '+name)
        target = run/'source'/name; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target); hashes[name]=sha(source)
    from prepare_nature13_clpu_static import load_inputs
    *_, dependencies = load_inputs()
    inputs = dict(dependencies)
    pointer_path = ROOT/'artifacts/restoration_shortlist_sources_latest.json'
    pointer = read(pointer_path)
    author = ROOT/pointer['derived_directory']/'nature_author_source'
    folder = author/'RL_Training/Environments/DSSdirect_13bus_loadandswitching'
    paths = list(folder.rglob('*'))+[author/'RL_Testing/13bus/IEEELineCodes.dss',pointer_path]
    for path in paths:
        if path.is_file():
            inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    input_copies = {}
    for index,(name, digest) in enumerate(sorted(inputs.items())):
        path = Path(name) if Path(name).is_absolute() else ROOT/name
        require(sha(path)==digest,'Input changed: '+name)
        input_copies[name] = input_archive_name(index,path)
        target = run/input_copies[name]
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    for name in ('summary.json','interface_schema.json'):
        shutil.copyfile(STATIC/name,run/('static_'+name))
    free = shutil.disk_usage(ROOT).free
    require(free>512*2**20,'Less than 512 MiB free for bounded evidence')
    prepared = {'utc':utc(),'execution_authorized':True,'source_hashes':hashes,
        'project_input_hashes':inputs,'project_input_copies':input_copies,'settings_sha256':sha(run/'settings.json'),
        'schedule_sha256':sha(run/'schedule.json'),'free_bytes':free,
        'input_hashes':{p.name:sha(p) for p in run.iterdir() if p.is_file()},
        'original_reference_sha256':REF_SHA,'new_physical_queries':0}
    save(run/'prepared.json',prepared)
    save(ACTIVE,{'run':run.name,'status':'PREPARED_PENDING_INDEPENDENT_REVIEW','new_physical_queries':0})
    return run


def verify(run):
    frozen = read(run/'prepared.json')
    require(settings()==read(run/'settings.json'),'Fixed settings changed')
    require(environment()==read(run/'environment.json'),'Environment changed after prepare')
    for name,digest in frozen['source_hashes'].items():
        require(sha(ROOT/name)==sha(run/'source'/name)==digest,'Frozen source changed: '+name)
    for name,digest in frozen['project_input_hashes'].items():
        path=Path(name) if Path(name).is_absolute() else ROOT/name
        target=(run/frozen['project_input_copies'][name]).resolve()
        require(target.is_relative_to(run.resolve()),'Archived input escapes run')
        require(sha(path)==sha(target)==digest,'Frozen data changed: '+name)
    for name,digest in frozen['input_hashes'].items():
        require(sha(run/name)==digest,'Frozen run input changed: '+name)


def physical_input(request):
    return {k:request[k] for k in ('action','case','alpha','algorithm','multipliers')}


def point_gate(spec,row):
    if row.get('reliable') is not True or row.get('status') not in ('PASS','FAIL'):
        return 'STOP_UNRELIABLE_POINT'
    if spec['group'] in ('reference','initial') and row['status'] != 'PASS':
        return 'STOP_REFERENCE_CONSTRAINT_FAIL' if spec['group']=='reference' else 'STOP_INITIAL_NO_WAIT_BASELINE'
    return None


def make_wait_certificate(env,source_spec,row,initial_metadata,budget):
    from dnr.nature13_clpu_physical import check_nature13_constraints,verify_nature13_isolation
    from dnr.nature34_clpu_environment import CLPUReliabilityError
    config = read(REF/'config.json')
    baseline = canonical(physical_input(source_spec['request']))
    metadata = {**initial_metadata,**row['metadata_override']}
    evidence = []
    class SavedOracle:
        def certify(self,state,t):
            budget.check()
            request=env.snapshot_request(state,t,algorithm=0)
            require(canonical(physical_input(request))==baseline,'WAIT exact physical input differs')
            known=check_nature13_constraints(row['raw_outputs'],metadata,**config['constraint_tolerances'])
            isolation=verify_nature13_isolation(row['raw_outputs'],metadata)
            if known['known_constraints_satisfied'] is not True or isolation['status']!='PASS' or row['ac_audit']['status']!='PASS':
                raise CLPUReliabilityError('Saved evidence cannot certify current WAIT identity')
            identity=request['state_identity']
            evidence.append({'identity':identity,'source_query_id':row['query_id'],
                             'same_physical_input':True,'current_constraints':'PASS','current_isolation':'PASS',
                             'ac_evidence_source_query_id':row['query_id']})
            return {'point_id':row['query_id'],'status':'PASS','reliable':True,'feedback':[]}
    before=budget.stats()
    cert=env.validate_plan(env.initial,[0]*30,SavedOracle())
    require(cert['status']=='COMPLETE' and len(cert['samples'])==90,'WAIT coverage incomplete')
    require(budget.stats()==before,'Saved WAIT reconstruction consumed physical budget')
    return {'status':'COMPLETE_WAIT_CERTIFICATE','source_query_id':row['query_id'],
            'actual_additional_queries':0,'actual_additional_entries':0,'plan':cert,'logical_evidence':evidence}


def worker(run,deadline):
    started=time.monotonic(); budget=None; events=[]; rows=[]; comparisons=[]; waits=[]; attempted=[]
    summary={'status':'STARTED','started_utc':utc(),'new_training_steps':0,'completed_queries':0}
    (run/'points').mkdir(exist_ok=False)
    try:
        require((run/'execution_started.json').is_file(),'Controller authorization marker required')
        with (run/'worker_started.json').open('x',encoding='utf-8') as marker:
            json.dump({'utc':utc(),'deadline_monotonic':deadline,'single_worker':True},marker)
        verify(run)
        from dnr.nature13_clpu_physical import Nature13Budget,Nature13CLPUPhysicalRuntime,engine_numeric_lists
        from prepare_nature13_clpu_static import build_environment
        from audit_nature13_curtailment_validation import algorithm_difference
        from audit_nature13_clpu_acceptance import compare_isolated_points
        config=read(run/'physical_config.json'); schedule=read(run/'schedule.json')
        with gzip.open(run/'reference_points.json.gz','rt',encoding='utf-8') as f:
            refs={r['id']:r for r in json.load(f)}
        with (run/'numerical_events.jsonl').open('x',encoding='utf-8') as log, (run/'attempts.jsonl').open('x',encoding='utf-8') as attempts:
            budget=Nature13Budget(max_queries=16,max_entries=256,deadline_monotonic=deadline,
                                 emit=lambda value:append(log,value))
            runtime=Nature13CLPUPhysicalRuntime(ROOT,run,config,budget=budget,events=events)
            save(run/'metadata.json',runtime.initial_metadata);save(run/'adaptation.json',runtime.metadata)
            save(run/'actual_work_input_hashes.json',{p.relative_to(run/'work').as_posix():sha(p) for p in (run/'work').rglob('*') if p.is_file()})
            for spec in schedule:
                qid=spec['query_id']; request=spec['request'];budget.check()
                append(attempts,{'query_id':qid,'event':'STARTED','group':spec['group'],'budget':budget.stats()})
                attempted.append(qid)
                call={k:request[k] for k in ('action','case','alpha','algorithm','multipliers','time_minutes','state_identity')}
                try:
                    row=runtime.evaluate_point(**call,query_id=qid)
                    path=run/'points'/f'{qid:03d}.json.gz'
                    require(not path.exists(),'A point output already exists; retry forbidden')
                    with gzip.open(path,'wt',encoding='utf-8') as f:
                        json.dump(row,f,ensure_ascii=False,allow_nan=True)
                    rows.append(row);summary['completed_queries']=len(rows)
                    append(attempts,{'query_id':qid,'event':'COMPLETED','status':row['status'],'budget':budget.stats()})
                except BaseException as exc:
                    partial=getattr(runtime,'last_query_evidence',None)
                    if isinstance(partial,dict) and partial.get('query_id')==qid:
                        with gzip.open(run/'points'/f'{qid:03d}_partial.json.gz','wt',encoding='utf-8') as f:
                            json.dump(engine_numeric_lists(partial),f,ensure_ascii=False,allow_nan=True)
                    append(attempts,{'query_id':qid,'event':'FAILED','error':repr(exc),'budget':budget.stats()})
                    raise
                stop=point_gate(spec,row)
                if stop:
                    summary.update(status=stop,stopped_query_id=qid);break
                if spec['group']=='reference':
                    original=refs[spec['source_point_id']]
                    diff=algorithm_difference(original['raw_outputs'],row['raw_outputs'],config['cross_algorithm_tolerances'])
                    comparisons.append({'kind':'old_reference','query_id':qid,'source_point_id':original['id'],**diff})
                    if diff['status']!='PASS':
                        summary.update(status='STOP_REFERENCE_NUMERICAL_MISMATCH',stopped_query_id=qid);break
                if request['algorithm']==1:
                    diff=compare_isolated_points(rows[-2],row,runtime.initial_metadata,config['cross_algorithm_tolerances'])
                    comparisons.append({'kind':'cross_algorithm','query_ids':[qid-1,qid],**diff})
                    if diff['status']!='PASS':
                        summary.update(status='STOP_CROSS_ALGORITHM_MISMATCH',stopped_query_id=qid);break
                    if spec['group']=='initial':
                        env=build_environment(ROOT,spec['fault_line'],3)
                        cert=make_wait_certificate(env,schedule[qid-1],rows[-2],runtime.initial_metadata,budget)
                        save(run/('wait_'+env.scenario['scenario_id']+'.json'),cert);waits.append(env.scenario['scenario_id'])
                save(run/'comparisons.json',comparisons)
                print(json.dumps({'query_id':qid,'group':spec['group'],'status':row['status'],
                                  'numerical_entries':budget.returned_entries}),flush=True)
            else:
                budget.check()
                require(len(rows)==16 and len(waits)==3,'Complete schedule and three WAIT certificates required')
                summary['status']='COMPLETE_ACCEPTANCE_PENDING_INDEPENDENT_AUDIT'
            save(run/'engine_calls.json',events)
    except BaseException as exc:
        summary.update(status='STOP_ACCEPTANCE_EXCEPTION',error=repr(exc),traceback=traceback.format_exc())
    finally:
        summary.update(elapsed_worker_seconds=time.monotonic()-started,
            budget=budget.stats() if budget else None,completed_wait_certificates=waits,
            completed_query_ids=[r['query_id'] for r in rows],
            attempted_query_ids=attempted,
            failed_attempt_query_ids=[i for i in attempted if i not in {r['query_id'] for r in rows}],
            not_run_query_ids=[i for i in range(16) if i not in attempted])
        save(run/'engine_calls.json',events);save(run/'comparisons.json',comparisons)
        save(run/'worker_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return 0 if summary['status'].startswith('COMPLETE_') else 2


def execute(run):
    verify(run)
    review=read(run/'pre_execution_review.json')
    require(review['status']=='PASS' and review['prepared_sha256']==sha(run/'prepared.json'),'Independent frozen precheck required')
    with (run/'execution_started.json').open('x',encoding='utf-8') as f:
        json.dump({'utc':utc(),'single_execution':True,'prepared_sha256':sha(run/'prepared.json')},f)
    child_env=os.environ.copy()
    child_env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    start=time.monotonic();deadline=start+240
    cmd=[sys.executable,'-X','utf8','-B',str(Path(__file__).resolve()),'worker','--run',str(run),'--deadline',str(deadline)]
    with (run/'worker.log').open('x',encoding='utf-8') as out:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=child_env,stdout=out,stderr=subprocess.STDOUT)
        timed_out=False
        try:
            code=proc.wait(timeout=max(0.001,deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out=True;proc.kill();code=proc.wait()
    controller={'started_utc':read(run/'execution_started.json')['utc'],'returncode':code,
        'timed_out':timed_out,'elapsed_cold_seconds':time.monotonic()-start,
        'status':'HARD_TIMEOUT_STOPPED' if timed_out else 'WORKER_EXITED','retries':0}
    save(run/'controller.json',controller)
    status=read(run/'worker_summary.json')['status'] if (run/'worker_summary.json').exists() else controller['status']
    if timed_out:status='HARD_TIMEOUT_STOPPED'
    save(ACTIVE,{'run':run.name,'status':status,'new_training_steps':0})
    save(run/'execution_manifest.json',{p.relative_to(run).as_posix():sha(p) for p in run.rglob('*') if p.is_file() and p.name!='execution_manifest.json'})
    print(json.dumps({'run':run.name,**controller,'worker_status':status},ensure_ascii=False))
    return code


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','execute','worker'])
    parser.add_argument('--tests',type=Path);parser.add_argument('--run',type=Path)
    parser.add_argument('--deadline',type=float)
    args=parser.parse_args()
    if args.mode=='prepare':
        print(prepare(args.tests));return 0
    require(args.run is not None,'Run required');run=args.run.resolve()
    require(run.parent==(ROOT/'runs').resolve(),'Run must be direct project runs child')
    if args.mode=='execute':return execute(run)
    require(args.deadline is not None and math.isfinite(args.deadline),'Parent wall deadline required')
    return worker(run,args.deadline)


if __name__=='__main__':
    raise SystemExit(main())
