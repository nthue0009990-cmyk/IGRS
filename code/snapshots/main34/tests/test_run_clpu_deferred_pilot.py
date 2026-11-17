from dataclasses import asdict
import gzip
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import time

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_clpu_deferred_pilot as runner
from dnr.nature34_clpu_branch import BranchConfig
from test_nature34_clpu_shared_evidence import build
from test_run_clpu_shared_pilot import contexts


def models():
    return [{'method':m,'seed':s,'checkpoint':f'models/{m}_{s}.pt','source_sha256':'a'*64}
        for m in ('bc_raw','ppo_raw') for s in runner.SEEDS]


def rows(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',encoding='utf-8') as stream:return [json.loads(x) for x in stream if x.strip()]


class FakePolicy:
    representation='raw'
    def __init__(self,env):
        self.calls=0
        self.architecture={'candidate_dim':27+env.horizon+3*len(env.load_names),
            'global_dim':11+len(env.load_names),'representation':'raw','hidden':128}
    def act(self,observation,mask,**kwargs):
        self.calls+=1
        return {'action':next(i for i,v in enumerate(mask) if v),'logprob':0.,'value':0.}


def route_fixture(build,arm='deferred_window'):
    service=build('shared')
    spec=next(x for x in runner.schedule_for(contexts(),models()) if x['arm']==arm)
    spec={**spec,'cell':'task'}
    for phase in ('EARLY','GUARD'):runner.save(service.cell/phase/'spec.json',runner.phase_spec(spec,phase))
    policy=FakePolicy(service.env) if spec['checkpoint'] else None
    with service:
        phases,cps=runner.run_routes(service,service.env,BranchConfig(),service.cell,spec,time.perf_counter(),time.time()+60,policy)
    return service,spec,phases,cps,policy


def test_fixed64_population_allseeds_and_latin_positions():
    schedule=runner.schedule_for(contexts(),models())
    assert len(schedule)==64 and schedule==runner.schedule_for(list(reversed(contexts())),models())
    for context in range(8):
        group=[s for s in schedule if s['context_id']==context]
        assert {x['arm_slot'] for x in group}==set(runner.SLOTS)
    for slot in runner.SLOTS:
        assert {r['arm_position'] for r in schedule if r['arm_slot']==slot}==set(range(8))
    assert all(s['mode']=='shared' and s['protected'] for s in schedule)
    assert runner.settings()['maximum_numerical_entries']==131264


def test_reject_missing_context_or_seed():
    with pytest.raises(ValueError):runner.schedule_for(contexts()[:-1],models())
    with pytest.raises(ValueError):runner.schedule_for(contexts(),models()[:-1])


def test_real_model_binder_preserves_bound_nested_preflight_seal(tmp_path):
    run=tmp_path/'model_binding';run.mkdir()
    actual=runner.bind_models(run)
    binding=runner.read(run/'model_preflight_binding.json')
    assert len(actual)==6
    assert runner.sha(run/'model_preflight/completion_manifest.json')==binding['completion_sha256']
    assert runner.sha(run/'model_preflight/summary.json')==binding['summary_sha256']
    for model in actual:assert runner.sha(run/model['checkpoint'])==model['source_sha256']


@pytest.mark.parametrize('arm',['interleaved_window','deferred_window','deferred_bc','deferred_ppo'])
def test_real_macro_routes_full_guard_and_paid_accounting(build,arm):
    service,spec,phases,cps,policy=route_fixture(build,arm)
    early,guard=phases['EARLY'],phases['GUARD']
    assert early['search_decisions']<=33 and early['graph_expansions']<=16 and early['verify_attempts']<=16
    assert all(c['numerical_entries']==3+4*c['physical_points'] for c in cps)
    assert cps[-1]['guard_established'] and all(not c['guard_established'] for c in cps[:-1])
    assert sum(r['logical_points'] for r in phases.values())==cps[-1]['logical_points']
    assert sum(r['actual_physical_points'] for r in phases.values())==service.physical_points
    assert guard['policy_decisions']==0 and guard['verify_attempts'] is None
    assert early['control']['controller_failed'] is False and early['done']
    decisions=rows(service.cell/'EARLY/decisions.jsonl.gz')
    assert len(decisions)==early['search_decisions']*2
    for start,end in zip(decisions[::2],decisions[1::2]):
        assert start['status']=='STARTED' and end['status']=='COMPLETED'
        assert start['proposal']==end['proposal'] and start['oracle_before']==end['oracle_before']
        assert 'observation' not in start['proposal'] and 'obs_compact' not in end
        if start['proposal']['operation']=='VERIFY_COMPLETE_PLAN':
            assert set(start['obs_compact'])=={'representation','candidate_features','global_features','mask','branch_ids'}
            assert end['response']['logical_points_after']==end['oracle_after']['logical_points']
        else:
            assert start['obs_compact'] is None
            assert start['oracle_before']==end['oracle_after']
    if policy:
        assert policy.calls==early['policy_decisions']==early['verify_attempts']>0
    else:assert early['policy_decisions']==0


def test_macros_preserve_complete_guard_semantics(build):
    a=route_fixture(build,'interleaved_window'); b=route_fixture(build,'deferred_window')
    for key in ('best_plan','trace','stop_reason','logical_points','search_decisions'):
        assert a[2]['GUARD'][key]==b[2]['GUARD'][key]


def test_route_expired_deadline_makes_no_requests(build):
    service=build('shared'); spec=runner.schedule_for(contexts(),models())[0]
    with pytest.raises(ValueError,match='deadline'):
        with service:runner.run_routes(service,service.env,BranchConfig(),service.cell,spec,time.perf_counter(),0)
    assert service.physical_points==0 and not service.views


def test_complete_worker_closes_single_runtime_and_model_free_rule(build,monkeypatch):
    import run_nature34_clpu_pilot as original
    from dnr import nature34_clpu_shared_evidence as shared
    service=build('shared'); run,cell=service.parent,service.cell
    spec=next(s for s in runner.schedule_for(contexts(),models()) if s['arm']=='deferred_window')
    spec={**spec,'cell':'task'}
    runner.save(cell/'spec.json',spec); runner.save(run/'schedule.json',[spec])
    runner.save(run/'config.json',{'branch_search':asdict(BranchConfig())})
    runner.save(run/'physical_scope.json',service.scope)
    deadline=time.time()+60
    runner.save(cell/'execution.json',{'deadline_epoch_seconds':deadline})
    runner.save(run/'execution_started.json',{'deadline_epoch_seconds':deadline})
    for phase in ('EARLY','GUARD'):runner.save(cell/phase/'spec.json',runner.phase_spec(spec,phase))
    monkeypatch.setattr(runner,'_WORKER_ENTERED',False)
    monkeypatch.setattr(runner,'verify_execution',lambda _:None)
    monkeypatch.setattr(runner.shutil,'disk_usage',lambda _:SimpleNamespace(free=10**15))
    monkeypatch.setattr(original,'make_env',lambda *_:service.env)
    monkeypatch.setattr(shared,'SharedEvidenceService',lambda *a,**k:service)
    assert runner.worker(run,cell)==0
    result=runner.read(cell/'result.json')
    assert result['schema']=='clpu-deferred-protected-result-v1' and result['guard_audit']['guard_established']
    assert result['model_load_seconds']==0 and result['policy_decisions']==0
    assert result['worker_elapsed_seconds']==pytest.approx(result['preflight_elapsed_seconds']+result['computation_elapsed_seconds'])
    assert result['numerical_entries']==3+4*result['actual_physical_points']


def controller_fixture(tmp_path,monkeypatch,*,failure=False,progress_error=False):
    run=tmp_path/'run';run.mkdir()
    schedule=runner.schedule_for(contexts(),models());runner.save(run/'schedule.json',schedule)
    runner.save(run/'resource_budget.json',{'historical_max_bytes_per_actual_point':1})
    for spec in schedule:runner.save(run/spec['cell']/'spec.json',spec)
    monkeypatch.setattr(runner,'verify_execution',lambda _:None)
    monkeypatch.setattr(runner,'ACTIVE',tmp_path/'active.json')
    monkeypatch.setattr(runner.shutil,'disk_usage',lambda _:SimpleNamespace(free=10**15))
    monkeypatch.setattr(runner.time,'sleep',lambda _:None)
    launched=[]
    class Process:
        def __init__(self,cmd,**kw):
            if len(launched)%2==0:assert all(p.finished for p in launched)
            self.cell=Path(cmd[-1]);self.pid=100+len(launched);self.calls=0;self.finished=False
            self.code=19 if failure and not launched else 0;launched.append(self)
        def poll(self):
            self.calls+=1
            if self.calls<2:return None
            if not self.finished:
                runner.save(self.cell/'result.json',{'logical_points':2,'actual_physical_points':1,'numerical_entries':7})
                runner.save(self.cell/'completion_manifest.json',{})
                self.finished=True
            return self.code
    monkeypatch.setattr(runner.subprocess,'Popen',Process)
    if progress_error:
        original=runner.save
        def bad(path,value):
            if path==runner.ACTIVE and value.get('status')=='RUNNING':raise PermissionError('simulated progress failure')
            original(path,value)
        monkeypatch.setattr(runner,'save',bad)
    return run,launched


def test_controller64_real_exits_and_pair_drain(tmp_path,monkeypatch):
    run,launched=controller_fixture(tmp_path,monkeypatch)
    assert runner.execute(run)==0
    result=runner.read(run/'controller_completion.json')
    assert len(launched)==len(result['completed'])==len(result['exits'])==64
    assert result['physical_points']==64 and result['numerical_entries']==448


@pytest.mark.parametrize('kind',['exit','progress'])
def test_controller_errors_stop_dispatch_and_drain_active(tmp_path,monkeypatch,kind):
    run,launched=controller_fixture(tmp_path,monkeypatch,failure=kind=='exit',progress_error=kind=='progress')
    assert runner.execute(run)==1
    result=runner.read(run/'controller_completion.json')
    assert len(launched)==2 and all(p.finished for p in launched)
    assert len(result['exits'])==2 and result['physical_points']==2 and result['errors']
