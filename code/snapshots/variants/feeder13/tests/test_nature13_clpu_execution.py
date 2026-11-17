"""Software only: saved physical evidence or explicitly synthetic search labels."""
from copy import deepcopy
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import sys
import time
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from prepare_nature13_clpu_static import build_environment
from dnr.nature13_clpu_evidence import EvidenceService, physical_input, digest, replay_certificate
from dnr.nature13_clpu_environment import CLPUBudgetExhausted, CLPUReliabilityError
ACCEPT=ROOT/'runs/20261101T052022_022941Z_nature13_clpu_acceptance'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


@pytest.fixture
def evidence():
    rows=[]
    for i in (4,10):
        with gzip.open(ACCEPT/'points'/f'{i:03d}.json.gz','rt',encoding='utf-8') as stream:
            rows.append(json.load(stream))
    return read(ACCEPT/'metadata.json'),read(ACCEPT/'physical_config.json'),rows


def replay_runtime(evidence, *, unknown=False):
    metadata, config, rows=evidence
    class SavedRuntime:
        def __init__(self,root,cell,physical_config,*,budget,events):
            self.initial_metadata=deepcopy(metadata)
            self.metadata={'software_test_saved_record_replay':True,'new_physical_queries':0}
            self.budget=budget
            budget.invoke(lambda:None,'SYNTHETIC_INITIALIZATION',(),{}, {'numerical_solution_entries':3})
        def evaluate_point(self,**request):
            qid=request.pop('query_id')
            self.budget.begin_query(qid)
            matching=[]
            for row in rows:
                expected={'action':row['action'],'case':{'name':row['case'],'fault_lines':row['fault_lines']},
                    'alpha':row['alpha'],'algorithm':row['algorithm'],'multipliers':row['clpu_multipliers']}
                if expected==physical_input(request):matching.append(row)
            assert len(matching)==1,'Only two already measured inputs may be replayed in this test'
            self.budget.invoke(lambda:None,'SYNTHETIC_QUERY_COST',(),{}, {'numerical_solution_entries':14})
            result=deepcopy(matching[0]);result['query_id']=qid
            result['state_identity']=request['state_identity'];result['time_minutes']=request['time_minutes']
            if unknown:result.update(status='UNKNOWN',reliable=False)
            return result
    return SavedRuntime


def service(tmp_path,evidence,order=('GUARD','EARLY'),**kwargs):
    env=build_environment(ROOT,'line.632633',3)
    return EvidenceService(ROOT,tmp_path,env,evidence[1],{'test_scope':'software-only-saved-evidence'},
        order,time.monotonic()+60,runtime_factory=replay_runtime(evidence),**kwargs)


def test_exact_physical_reuse_preserves_independent_logical_costs(tmp_path,evidence):
    with service(tmp_path,evidence) as s:
        with s.view('GUARD') as g:
            g.certify(s.env.initial,0)
            g.certify(s.env.initial,0)
            g.certify(s.env.initial,30)
            assert (g.points,g.requests,g.physical_points,g.hits,g.reuses)==(2,3,1,1,1)
        with s.view('EARLY') as e:
            e.certify(s.env.initial,0)
            assert (e.points,e.physical_points,e.cross_reuses)==(1,0,1)
        assert s.budget.returned_entries==17 and s.budget.queries==1
    assert s.summary()['logical_points']==3


def test_reliable_failure_reuse_rebuilds_feedback_time(tmp_path,evidence):
    with service(tmp_path,evidence) as s:
        peak=s.env.advance(s.env.initial,11)
        with s.view('GUARD') as g:
            first=g.certify(peak,0)
            assert first['status']=='FAIL' and first['feedback']
        with s.view('EARLY') as e:
            delayed=s.env.advance(s.env.advance(s.env.initial,0),11)
            second=e.certify(delayed,60)
            assert second['status']=='FAIL'
            assert all(v['time_minutes']==1. for v in second['feedback'])
            assert all(v['time_minutes']==0. for v in first['feedback'])
            assert e.physical_points==0 and e.cross_reuses==1
        assert s.budget.queries==1


def test_logical_limit_does_not_make_free_new_identities(tmp_path,evidence):
    with service(tmp_path,evidence,logical_budget=1) as s:
        with s.view('GUARD') as g:
            g.certify(s.env.initial,0)
            with pytest.raises(CLPUBudgetExhausted):g.certify(s.env.initial,30)
            assert g.points==1 and s.budget.queries==1


def test_actual_limit_still_allows_already_paid_inputs(tmp_path,evidence):
    with service(tmp_path,evidence,physical_budget=1) as s:
        with s.view('GUARD') as g:
            g.certify(s.env.initial,0)
            with pytest.raises(CLPUBudgetExhausted):g.certify(s.env.advance(s.env.initial,11),0)
            assert g.certify(s.env.initial,30)['status']=='PASS'
            assert s.budget.queries==1


def test_unknown_stops_and_preserves_raw_never_cached(tmp_path,evidence):
    env=build_environment(ROOT,'line.632633',3)
    s=EvidenceService(ROOT,tmp_path,env,evidence[1],{'test':True},['GUARD','EARLY'],
        time.monotonic()+60,runtime_factory=replay_runtime(evidence,unknown=True))
    with pytest.raises(CLPUReliabilityError):
        with s:
            with s.view('GUARD') as g:g.certify(env.initial,0)
    assert not s.pool and s.poisoned and s.budget.queries==1
    assert (tmp_path/'physical/0000.json.gz').exists()


def test_phase_order_is_enforced(tmp_path,evidence):
    with service(tmp_path,evidence,order=('EARLY','GUARD')) as s:
        with pytest.raises(CLPUReliabilityError):s.view('GUARD')
        assert s.budget.queries==0


def test_mutated_physical_scope_or_config_is_rejected(tmp_path,evidence):
    with pytest.raises(CLPUReliabilityError):
        with service(tmp_path,evidence) as s:
            with s.view('GUARD') as g:
                s.config['voltage_limits_pu']=[0.,2.]
                g.certify(s.env.initial,0)
    assert s.budget.queries==0


def test_two_tasks_never_share_paid_evidence(tmp_path,evidence):
    for name in ('task_a','task_b'):
        with service(tmp_path/name,evidence) as s:
            with s.view('GUARD') as g:g.certify(s.env.initial,0)
            assert s.budget.queries==1 and g.physical_points==1


def test_input_key_contains_alpha_algorithm_full_loads_and_fault(evidence):
    env=build_environment(ROOT,'line.632633',3);r=env.snapshot_request(env.initial,0)
    original=digest(physical_input(r))
    for key,value in [('alpha',.5),('algorithm',1),('action',[0]*11),('case',{'name':'x','fault_lines':[]})]:
        assert digest(physical_input({**r,key:value}))!=original
    assert digest(physical_input({**r,'time_minutes':3.}))==original
    different=deepcopy(r);different['multipliers'][env.load_names[-1]]=1.0000000000001
    assert digest(physical_input(different))!=original


@pytest.mark.parametrize('arm',['P','E','Forward'])
def test_routes_use_frozen_methods_and_beam_with_synthetic_labels(arm):
    from dnr.nature13_clpu_branch import BranchConfig
    from dnr.nature13_clpu_staged_pickup import generate_staged_pickup_candidates
    from dnr.nature13_clpu_routes import run_routes
    env=build_environment(ROOT,'line.632633',3)
    class Oracle:
        def __init__(self,s):self.s=s;self.points=self.physical_points=self.requests=0;self.cache={};self.budget=256
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def summary(self):return {'points':self.points,'physical_points':self.physical_points,'requests':self.requests}
        def certify(self,state,t):
            self.requests+=1;req=env.snapshot_request(state,t);sem=digest(req['state_identity'])
            if sem in self.cache:return deepcopy(self.cache[sem])
            if self.points>=256:raise CLPUBudgetExhausted()
            key=digest(physical_input(req))
            if key not in self.s.pool:self.s.pool.add(key);self.physical_points+=1
            # Explicit synthetic boundary labels, never a scientific physical record.
            status='FAIL' if max(req['multipliers'].values())>1.65 else 'PASS'
            value={'point_id':self.points,'status':status,'reliable':True,'feedback':[]}
            self.points+=1;self.cache[sem]=value;return deepcopy(value)
    class Service:
        def __init__(self):self.pool=set();self.phase_order=['EARLY','GUARD'] if arm=='E' else ['GUARD','EARLY'];self.used=[]
        def view(self,name):
            assert name==self.phase_order[len(self.used)];self.used.append(name);return Oracle(self)
    s=Service();phases,merged=run_routes(env,s,asdict(BranchConfig()),arm,generate_staged_pickup_candidates(env),lambda *args:'a'*64)
    assert s.used==s.phase_order and merged['guard_audit']['nondegradation_verified']
    assert merged['guard_audit']['final_energy_kwh']>=merged['guard_audit']['baseline_energy_kwh']
    assert len(merged['best_plan']['validation']['samples'])==90
    assert phases['EARLY']['verify_attempts']<=16


def test_import_and_checks_have_no_numerical_engine():
    assert not any(n=='dss' or n.startswith(('dss.','opendssdirect','Environments')) for n in sys.modules)


def test_normalization_does_not_drop_physical_times():
    from audit_nature13_clpu_comparison import normalized
    assert normalized({'elapsed_seconds':2.,'stage_generation_seconds':1.,'time_seconds':60.,'tau_seconds':543.})=={'time_seconds':60.,'tau_seconds':543.}


@pytest.mark.parametrize('timeout',[False,True])
def test_comparison_controller_stops_without_retries(tmp_path,monkeypatch,timeout):
    import run_nature13_clpu_comparison as runner
    runner.save(tmp_path/'prepared.json',{'software_test':True})
    runner.save(tmp_path/'preflight.json',{'status':'PASS','prepared_sha256':runner.sha(tmp_path/'prepared.json')})
    runner.save(tmp_path/'design/settings.json',{'controller_seconds':3600,'task_seconds':240,'minimum_free_bytes':0})
    schedule=[{'task_id':i,'context_id':'software_test','arm':['P','E','Forward'][i%3]} for i in range(18)]
    runner.save(tmp_path/'design/schedule.json',schedule)
    monkeypatch.setattr(runner,'verify',lambda run:None)
    monkeypatch.setattr(runner,'ACTIVE',tmp_path/'test_active.json')
    calls=[]
    class Process:
        def wait(self,timeout=None):
            calls.append('wait')
            if timeout is not None and test_timeout:
                raise runner.subprocess.TimeoutExpired('software_test',timeout)
            return -9 if test_timeout else 2
        def kill(self):calls.append('kill')
    test_timeout=timeout
    def spawn(*args,**kwargs):calls.append('spawn');return Process()
    monkeypatch.setattr(runner.subprocess,'Popen',spawn)
    assert runner.execute(tmp_path)==2
    result=runner.read(tmp_path/'controller_summary.json')
    assert result['attempted_tasks']==1 and result['retries']==0
    assert result['tasks'][0]['execution_status']=='STOPPED_ERROR'
    assert all(x['execution_status']=='NOT_RUN' for x in result['tasks'][1:])
    assert calls.count('spawn')==1 and calls.count('kill')==int(timeout)
    with pytest.raises(FileExistsError):runner.execute(tmp_path)
    assert calls.count('spawn')==1


def test_saved_search_replay_rejects_changed_physical_time():
    from audit_nature13_clpu_comparison import ReplayService
    env=build_environment(ROOT,'line.632633',3)
    state=env.initial
    rows=[{'request_id':0,'state':state.to_dict(),'time_seconds':0.,'kind':'ACTUAL',
           'logical_points_after':1,'physical_points_after':1,
           'certificate':{'point_id':0,'status':'PASS','reliable':True,'feedback':[]}}]
    service=ReplayService(env,['GUARD','EARLY'],{'GUARD':rows})
    view=service.view('GUARD')
    with pytest.raises(ValueError,match='request sequence'):
        view.certify(state,30.)
    assert view.requests==0
    assert view.certify(state,0.)['status']=='PASS'
    view.__exit__(None,None,None)
