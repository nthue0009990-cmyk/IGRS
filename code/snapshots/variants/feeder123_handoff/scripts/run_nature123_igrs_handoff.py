"""One cold, bounded IGRS migration task; external acceptance is a gate, not a seed."""
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from dnr.nature123_acceptance_runner import (DurableJournal, read_json, write_json, sha,
    inside, certificate_valid, supervise, seal_run)
from dnr.nature123_acceptance_budget import AcceptanceBudget, AcceptanceBudgetStop
from dnr.nature123_clpu_environment import CLPUEnvironment
from dnr.nature123_static_state import SparseRestorationGraph
from dnr.nature34_clpu_environment import CLPUBudgetExhausted, CLPUReliabilityError

SCOPE = 'NC123_IGRS_CANDIDATE_HANDOFF_V2'
LIMITS = {'logical_requests':512, 'physical_requests':512, 'numerical_entries':1536,
          'seconds':240, 'memory_bytes':4*1024**3, 'graph_state_cap':200000, 'graph_seconds':30}
SEARCH = {'lookahead_depth':6,'max_generated_candidates':5000,'max_checked_candidates':64,
          'max_branch_expansions':32,'beam_width':8,'point_budget':512,'action_window':32,
          'generic_neighbor_limit':32}


def new_run(suffix):
    path = ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_'+suffix)
    path.mkdir(); return path


def verify_seal(path):
    saved=read_json(path/'completion_manifest.json')
    assert all(sha(path/p)==v for p,v in saved.items())
    assert set(saved)=={p.relative_to(path).as_posix() for p in path.rglob('*')
                        if p.is_file() and p.name!='completion_manifest.json'}


def make_env(project, freeze, physical=False):
    from dnr.nature123_igrs_handoff import SearchEnvironment
    base=CLPUEnvironment(SparseRestorationGraph(read_json(inside(project,freeze['static_manifest_path'])),['l19']),
        model_fingerprint=freeze['model_fingerprint'],condition=freeze['condition'],event=freeze['event'],
        allow_physical=physical)
    return SearchEnvironment(base,graph_state_cap=LIMITS['graph_state_cap'],graph_seconds=LIMITS['graph_seconds'])


def validate(project, freeze, execute=False):
    assert freeze['schema_version']==SCOPE and freeze['budgets']==LIMITS and freeze['search_config']==SEARCH
    assert freeze['scope']=='ONE_DEVELOPMENT_CASE_FIRST_COMPLETE_GAIN_NO_COMPARISON'
    assert freeze['fault_lines']==['l19'] and freeze['automatic_retries']==0
    assert freeze['condition']['operating_config']['algorithm']==0
    assert freeze['physical_config']['runtime_variant']=='PHASE_AVAILABLE_V2'
    for rel,digest in freeze['source_sha256'].items(): assert sha(inside(project,rel))==digest,rel
    assert freeze['static_manifest_path'] in freeze['source_sha256']
    model=read_json(inside(project,'artifacts/nature123_model_v2/model_config.json'))
    assert freeze['model_fingerprint']==model['model_fingerprint']
    gate=freeze['acceptance_gate']
    assert sha(inside(project,gate['freeze']))==gate['freeze_sha256']
    prior=read_json(inside(project,gate['freeze']))
    assert prior['schema_version']=='NC123_PARTIAL_ACCEPTANCE_V1'
    assert freeze['physical_config']==prior['physical_config'] and freeze['event']==prior['event']
    condition=deepcopy(prior['condition']);condition['condition_id']='NC123_V2_L19_EVENT3_IGRS_HANDOFF'
    assert freeze['condition']==condition and freeze['static_manifest_path']==prior['static_manifest_path']
    assert sha(inside(project,gate['result']))==gate['result_sha256']
    accepted=read_json(inside(project,gate['result']))
    assert accepted['plans'][0]['status']=='COMPLETE' and accepted['plans'][0]['physical_certified'] is True
    assert accepted['plans'][0]['counts']=={'PASS':91,'FAIL':0,'UNKNOWN':0,'NOT_RUN':0}
    assert sha(inside(project,gate['audit']))==gate['audit_sha256']
    audit=read_json(inside(project,gate['audit']))
    assert audit['status']=='PASS_SAVED_RECERTIFICATION' and audit['kind']=='PARTIAL_ACCEPTANCE'
    assert Path(audit['source_run']).resolve()==inside(project,gate['result']).parent.resolve()
    assert audit['source_manifest_sha256']==sha(inside(project,gate['result']).parent/'completion_manifest.json')
    assert freeze['warm_start']=={'actions':None,'certificates':None,'external_threshold':None}
    if execute:
        assert freeze['physical_execution_allowed'] is True and freeze['software_checks']=='PASS'
        assert freeze['authorization']['scope']==SCOPE
    return make_env(project,freeze,physical=execute)


class PrivateOracle:
    evidence_kind='PHYSICAL'
    def __init__(self, project, run, env, freeze):
        from dnr.nature123_phase_available_physical import Nature123PhaseAvailablePhysicalRuntime
        self.env,self.run=env,run
        self.points=self.requests=self.returned_physical_results=self.cache_hits=self.logical_completed=0
        self.budget=LIMITS['logical_requests']; self.cache={}
        self.journal=DurableJournal(run/'events.jsonl')
        self.counter=AcceptanceBudget(max_requests=LIMITS['physical_requests'],max_entries=LIMITS['numerical_entries'],
                                      seconds=LIMITS['seconds'],emit=self.journal)
        self.runtime=Nature123PhaseAvailablePhysicalRuntime(project,budget=self.counter,
            physical_config=freeze['physical_config'],enable_physical=True)

    @property
    def physical_points(self):
        return self.counter.snapshot()['requests']

    def certify(self,state,seconds):
        from dnr.nature34_clpu_physical import structured_feedback
        if self.points>=self.budget: raise CLPUBudgetExhausted('Global logical request cap')
        try: self.counter.check()
        except AcceptanceBudgetStop as exc: raise CLPUBudgetExhausted(str(exc)) from exc
        request=self.env.snapshot_request(state,seconds)
        self.points+=1; self.requests=self.points
        key=request['request_sha256']
        self.journal({'event':'logical_request','logical_id':self.points,'request_sha256':key,
                      'cache_hit':key in self.cache})
        if key in self.cache:
            cert=self.cache[key]
            self.cache_hits+=1; self.logical_completed+=1
            self.journal({'event':'logical_completed','logical_id':self.points,'point_id':cert['point_id'],
                          'request_sha256':key,'cache_hit':True})
            return deepcopy(cert)
        point='point_%06d'%self.physical_points
        self.journal({'event':'request_started','point_id':point,'request_sha256':key})
        try: cert=self.runtime.evaluate(request,point_id=point)
        except AcceptanceBudgetStop as exc: raise CLPUBudgetExhausted(str(exc)) from exc
        self.returned_physical_results+=1
        path=self.run/'points'/(point+'.json'); write_json(path,cert)
        valid=certificate_valid(cert,request,point)
        self.journal({'event':'request_completed','point_id':point,'request_sha256':key,
                      'status':cert.get('status'),'reliable_certificate':valid,'result_sha256':sha(path)})
        if not valid: raise CLPUReliabilityError('Unbound or UNKNOWN physical evidence')
        answer={k:deepcopy(cert[k]) for k in ('status','reliable','point_id','request_sha256','evidence_kind','checks')}
        answer['feedback']=structured_feedback(cert['known_constraints'],cert['raw_outputs'],
            reliable=True,time_minutes=float(seconds)/60)
        self.cache[key]=deepcopy(answer)
        self.logical_completed+=1
        self.journal({'event':'logical_completed','logical_id':self.points,'point_id':point,
                      'request_sha256':key,'cache_hit':False})
        return answer


def prepare(partial_run, audit_path):
    verify_seal(partial_run)
    source=read_json(partial_run/'freeze.json')
    condition=deepcopy(source['condition']); condition['condition_id']='NC123_V2_L19_EVENT3_IGRS_HANDOFF'
    run=new_run('nc123_igrs_handoff_freeze')
    paths=list((ROOT/'src/dnr').rglob('*.py'))+list((ROOT/'tests').glob('test_nature123*.py'))
    paths += [p for p in (ROOT/'artifacts/nature123_model_v2').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    paths += [ROOT/'scripts/run_nature123_igrs_handoff.py',ROOT/'scripts/audit_nature123_handoff_result.py',
              ROOT/'docs/nature123_handoff_protocol_v2.md',ROOT/source['static_manifest_path'],
              ROOT/'artifacts/nature123_igrs_migration_20261107/derivation.json']
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(paths))}
    for rel in hashes:
        dest=run/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/rel).read_bytes())
    result_path=partial_run/'result.json'
    freeze={'schema_version':SCOPE,'scope':'ONE_DEVELOPMENT_CASE_FIRST_COMPLETE_GAIN_NO_COMPARISON',
        'authorization':{'scope':SCOPE,'record':'用户批准候选交接修订：保留GUARD，遗留完整候选优先共享16次验证，同预算一次冷任务，完整收益即停，无方法对照'},
        'physical_execution_allowed':False,'software_checks':'PENDING','budgets':LIMITS,'search_config':SEARCH,
        'model_fingerprint':source['model_fingerprint'],'physical_config':source['physical_config'],
        'condition':condition,'event':source['event'],'fault_lines':['l19'],
        'static_manifest_path':source['static_manifest_path'],'source_sha256':hashes,'seed':None,
        'automatic_retries':0,'cache_scope':'PRIVATE_EXACT_FULL_REQUEST_SHA_ONLY_NO_EXTERNAL_IMPORT',
        'warm_start':{'actions':None,'certificates':None,'external_threshold':None},
        'acceptance_gate':{'result':result_path.relative_to(ROOT).as_posix(),'result_sha256':sha(result_path),
            'freeze':(partial_run/'freeze.json').relative_to(ROOT).as_posix(),'freeze_sha256':sha(partial_run/'freeze.json'),
            'audit':audit_path.relative_to(ROOT).as_posix(),'audit_sha256':sha(audit_path)}}
    write_json(run/'freeze.json',freeze)
    environment=os.environ.copy();environment['PYTHONPATH']=str(ROOT/'src')
    tested=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_nature123*.py','-v'],
        cwd=ROOT,env=environment,capture_output=True,timeout=120)
    (run/'tests.stdout.log').write_bytes(tested.stdout);(run/'tests.stderr.log').write_bytes(tested.stderr)
    freeze.update(software_checks='PASS' if tested.returncode==0 else 'FAIL',physical_execution_allowed=tested.returncode==0)
    write_json(run/'freeze.json',freeze)
    write_json(run/'environment.json',{'python':sys.version,'executable':sys.executable,'seed':None,
        'dependencies':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()}})
    if tested.returncode==0: validate(ROOT,freeze)
    write_json(run/'result.json',{'status':'FROZEN' if tested.returncode==0 else 'FAILED_PRECHECK','physical_requests':0})
    seal_run(run);print(json.dumps({'run':str(run),'test_returncode':tested.returncode}),flush=True)
    return tested.returncode


def worker(project,run,freeze):
    from dnr.nature123_igrs_handoff import run_search
    env=validate(project,freeze,execute=True)
    oracle=PrivateOracle(project,run,env,freeze)
    oracle.journal({'event':'worker_started','pid':os.getpid(),'ppid':os.getppid(),'executable':sys.executable})
    records=DurableJournal(run/'attempts.jsonl');events=DurableJournal(run/'search_events.jsonl')
    try:
        result=run_search(env,oracle,config=freeze['search_config'],record_callback=records,event_callback=events)
        status='RETURNED'
    except BaseException as exc:
        result={'status':'UNKNOWN','error':repr(exc),'traceback':traceback.format_exc()};status='UNKNOWN'
    result['resource_ledger']={'logical_requests':oracle.points,'physical_requests':oracle.physical_points,
        'returned_physical_results':oracle.returned_physical_results,'cache_hits':oracle.cache_hits,
        'logical_completed':oracle.logical_completed,'logical_incomplete':oracle.points-oracle.logical_completed,
        'counter':oracle.counter.snapshot()}
    write_json(run/'worker_result.json',result)
    return 0 if status=='RETURNED' else 2


def main():
    import argparse
    if not __debug__:
        raise RuntimeError('Optimization mode may not disable scientific assertions')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('freeze',nargs='?',type=Path);p.add_argument('--prepare',action='store_true')
    p.add_argument('--partial-run',type=Path);p.add_argument('--partial-audit',type=Path)
    p.add_argument('--execute',action='store_true');p.add_argument('--worker',action='store_true')
    p.add_argument('--project',type=Path,default=ROOT);p.add_argument('--run-dir',type=Path);p.add_argument('--release-file',type=Path)
    args=p.parse_args()
    if Path(sys.executable).resolve()!=(args.project/'.venv/Scripts/python.exe').resolve():
        raise RuntimeError('The project .venv interpreter is required')
    if args.prepare: return prepare(args.partial_run.resolve(),args.partial_audit.resolve())
    freeze=read_json(args.freeze)
    if args.worker:
        assert args.execute and args.run_dir and args.release_file
        write_json(args.run_dir/'worker_identity.json',{'pid':os.getpid(),'ppid':os.getppid(),'executable':sys.executable})
        end=time.monotonic()+10
        while not args.release_file.exists():
            if time.monotonic()>end: raise RuntimeError('No supervisor release')
            time.sleep(.01)
        return worker(args.project,args.run_dir,freeze)
    validate(args.project,freeze,execute=args.execute)
    run=new_run('nc123_igrs_handoff_acceptance' if args.execute else 'nc123_igrs_handoff_dryrun')
    write_json(run/'freeze.json',freeze)
    for rel,digest in freeze['source_sha256'].items():
        dest=run/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(inside(args.project,rel).read_bytes())
        assert sha(dest)==digest
    write_json(run/'environment.json',{'python':sys.version,'executable':sys.executable,'seed':None,
        'dependencies':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},'freeze_sha256':sha(args.freeze)})
    if args.execute:
        release=run/'supervisor_release.json'
        command=[str(args.project/'.venv/Scripts/python.exe'),str(run/'source/scripts/run_nature123_igrs_handoff.py'),
                 str(run/'freeze.json'),'--execute','--worker','--project',str(args.project),
                 '--run-dir',str(run),'--release-file',str(release)]
        try:
            observed=supervise(command,cwd=run,run_dir=run,seconds=LIMITS['seconds'],memory_bytes=LIMITS['memory_bytes'],
                poll_seconds=.05,release_file=release,identity_file=run/'worker_identity.json')
            result={'status':'WORKER_RETURNED_REQUIRES_SAVED_AUDIT' if observed['returncode']==0 and observed['termination_reason'] is None else 'RESOURCE_OR_RELIABILITY_STOP',
                    'supervisor':observed,'method_comparison':False,'automatic_retries':0}
        except BaseException as exc:
            result={'status':'SUPERVISION_EXCEPTION','error':repr(exc),'traceback':traceback.format_exc()}
    else: result={'status':'PASS_DRY_RUN_ONLY','physical_requests':0}
    write_json(run/'result.json',result);seal_run(run)
    print(json.dumps({'run':str(run),**result}),flush=True)
    return 0 if result['status'] in ('WORKER_RETURNED_REQUIRES_SAVED_AUDIT','PASS_DRY_RUN_ONLY') else 2


if __name__=='__main__': raise SystemExit(main())


