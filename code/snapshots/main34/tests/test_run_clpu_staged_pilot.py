"""Staged common pilot and conditional dispatch, with no physical solver."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_clpu_staged_pilot as runner
from dnr.nature34_clpu_branch import BranchConfig
from dnr.nature34_clpu_staged_pickup import generate_staged_pickup_candidates
from dnr.nature34_clpu_shared_evidence import SharedEvidenceService
from test_nature34_clpu_shared_evidence import build
from test_nature34_clpu_branch import environment
from test_run_clpu_deferred_pilot import FakePolicy,models,rows
from test_run_clpu_shared_pilot import contexts


def test_complete_population_and_graph_only_partition_keep_every_seed_and_unexecuted_context():
    original=runner.read(runner.SOURCE/'contexts.json')
    model_bindings=runner.read(runner.MODELS_SOURCE/'model_bindings.json')
    schedule=runner.schedule_for(original,model_bindings)
    assert len(schedule)==32 and schedule==runner.schedule_for(list(reversed(original)),model_bindings)
    assert Counter(s['method'] for s in schedule)=={'beam':8,'bc_raw':24}
    assert {s['arm_slot'] for s in schedule}==set(runner.SLOTS)
    for slot in runner.SLOTS:
        assert Counter(s['arm_position'] for s in schedule if s['arm_slot']==slot)=={i:2 for i in range(4)}
    profile=runner.read(runner.GRAPH_PROFILE/'all_contexts.json')
    partition=runner.partition_schedule(schedule,profile)
    runner.check_graph_profile(runner.GRAPH_PROFILE,partition)
    changed=deepcopy(partition);changed['contexts'][0]['generation']['version']='unreviewed-version'
    with pytest.raises(ValueError,match='inventory differ'):runner.check_graph_profile(runner.GRAPH_PROFILE,changed)
    assert partition['eligible_contexts']==[2,7]
    assert len(partition['phase1_cells'])==8 and len(partition['phase2_cells'])==24
    assert set(partition['phase1_cells'])|set(partition['phase2_cells'])=={s['cell'] for s in schedule}
    assert runner.settings()['total_physical_budget']==16384
    assert runner.settings()['maximum_numerical_entries']==65632
    with pytest.raises(ValueError):runner.schedule_for(original[:-1],model_bindings)
    with pytest.raises(ValueError):runner.schedule_for(original,model_bindings[:-1])


def route_fixture(build,arm='staged_window'):
    template=build('shared')
    env=environment(horizon=30)
    service=SharedEvidenceService(template.parent,template.cell,env,template.scope,mode='shared',physical_budget=512,logical_budget=256)
    spec=next(s for s in runner.schedule_for(contexts(),models()) if s['arm']==arm)
    spec={**spec,'cell':'task'}
    for phase in ('EARLY','GUARD'):runner.save(service.cell/phase/'spec.json',runner.phase_spec(spec,phase))
    policy=FakePolicy(env) if spec['checkpoint'] else None
    expected=generate_staged_pickup_candidates(env)
    with service:
        phases,cps=runner.run_routes(service,env,BranchConfig(),service.cell,spec,time.perf_counter(),time.time()+60,
            policy,expected_generation=expected)
    return service,spec,phases,cps,policy


@pytest.mark.parametrize('arm',['staged_window','staged_bc'])
def test_common_pool_is_regenerated_and_complete_guard_is_paid_and_separate(build,arm):
    service,spec,phases,cps,policy=route_fixture(build,arm)
    early,guard=phases['EARLY'],phases['GUARD']
    assert early['graph_expansions']==0 and early['verify_attempts']<=16 and early['search_decisions']<=17
    assert len(early['stage_generation']['candidates'])<=24
    pool=runner.read(service.cell/'EARLY/candidate_pool.json')
    assert pool['generation']==early['stage_generation']==generate_staged_pickup_candidates(service.env)
    assert pool['generation_sha256']==runner.hashlib.sha256(runner.canonical(pool['generation']).encode()).hexdigest()
    assert list(service.views)==['EARLY','GUARD'] and service.closed
    assert all(c['numerical_entries']==3+4*c['physical_points'] for c in cps)
    assert cps[-1]['guard_established'] and not any(c['guard_established'] for c in cps[:-1])
    assert sum(r['actual_physical_points'] for r in phases.values())==service.physical_points
    assert guard['policy_decisions']==0 and guard['verify_attempts'] is None
    records=rows(service.cell/'EARLY/decisions.jsonl.gz')
    assert len(records)==2*early['search_decisions']
    assert {r['proposal']['operation'] for r in records}<={'VERIFY_COMPLETE_PLAN','YIELD_TO_BEAM'}
    for start,end in zip(records[::2],records[1::2]):
        assert start['proposal']==end['proposal'] and start['oracle_before']==end['oracle_before']
        assert 'observation' not in start['proposal'] and 'obs_compact' not in end
    if policy:assert policy.calls==early['policy_decisions']>0


def test_fixed_rankers_do_not_change_complete_guard_semantics(build):
    a=route_fixture(build,'staged_window');b=route_fixture(build,'staged_bc')
    for key in ('best_plan','trace','stop_reason','logical_points','search_decisions'):
        assert a[2]['GUARD'][key]==b[2]['GUARD'][key]
    assert a[2]['EARLY']['stage_generation']==b[2]['EARLY']['stage_generation']


def test_initial_fail_records_absent_pool_and_does_not_hide_it_with_guard(build):
    template=build('shared',failed=True);env=environment(horizon=30)
    service=SharedEvidenceService(template.parent,template.cell,env,template.scope,mode='shared',physical_budget=512,logical_budget=256)
    spec=next(s for s in runner.schedule_for(contexts(),models()) if s['arm']=='staged_window')
    with pytest.raises(ValueError,match='Paid bootstrap'):
        with service:
            runner.run_routes(service,env,BranchConfig(),service.cell,spec,time.perf_counter(),time.time()+60)
    assert list(service.views)==['EARLY'] and service.closed
    pool=runner.read(service.cell/'EARLY/candidate_pool.json')
    assert pool=={'generation':None,'generation_sha256':None,'status':'BOOTSTRAP_STOP_NOT_GENERATED','candidate_count':0}
    assert not (service.cell/'GUARD/phase_result.json').exists()


def controller_fixture(tmp_path,monkeypatch,failure=False):
    run=tmp_path/'run';run.mkdir()
    schedule=runner.schedule_for(contexts(),models())
    eligible=[s['cell'] for s in schedule if s['context_id'] in (2,7)]
    runner.save(run/'schedule.json',schedule)
    runner.save(run/'stage_plan.json',{'phase1_cells':eligible,'phase2_cells':[s['cell'] for s in schedule if s['cell'] not in eligible]})
    runner.save(run/'resource_budget.json',{'historical_max_bytes_per_actual_point':1})
    for s in schedule:runner.save(run/s['cell']/'spec.json',s)
    monkeypatch.setattr(runner,'verify_execution',lambda _:None)
    monkeypatch.setattr(runner,'ACTIVE',tmp_path/'active.json')
    monkeypatch.setattr(runner.shutil,'disk_usage',lambda _:SimpleNamespace(free=10**15))
    monkeypatch.setattr(runner.time,'sleep',lambda _:None)
    launched=[]
    class Process:
        def __init__(self,cmd,**kwargs):
            assert sum(not p.finished for p in launched)<2
            if len(launched)%2==0:assert all(p.finished for p in launched)
            self.cell=Path(cmd[-1]);self.pid=100+len(launched);self.calls=0;self.finished=False
            assert all(p.cell!=self.cell for p in launched)
            self.code=17 if failure and not launched else 0
            launched.append(self)
        def poll(self):
            self.calls+=1
            if self.calls<2:return None
            if not self.finished:
                runner.save(self.cell/'result.json',{'logical_points':2,'actual_physical_points':1,'numerical_entries':7,'requests':3})
                runner.save(self.cell/'worker_finished.json',{'status':'COMPLETE'})
                runner.seal(self.cell);self.finished=True
            return self.code
    monkeypatch.setattr(runner.subprocess,'Popen',Process)
    return run,launched


@pytest.mark.parametrize('gain',[False,True])
def test_two_stages_never_repeat_tasks_and_preserve_all_no_gain_slots(tmp_path,monkeypatch,gain):
    run,launched=controller_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(runner.os,'getpid',lambda:7101)
    assert runner.execute_stage1(run)==0 and len(launched)==8
    assert runner.read(run/'execution_started.json')['pid']==runner.read(run/'controller_stage1_started.json')['pid']==7101
    first=runner.read(run/'controller_stage1_completion.json')
    first_hashes={name:runner.sha(run/name/'completion_manifest.json') for name in first['completed']}
    runner.save(run/'audit_stage1/summary.json',{'elapsed_seconds':.1})
    monkeypatch.setattr(runner,'phase1_gate',lambda _: {'phase2_authorized':gain,'positive_cells':first['completed'][:1] if gain else []})
    monkeypatch.setattr(runner.os,'getpid',lambda:7102)
    assert runner.execute_stage2(run)==0
    result=runner.read(run/'controller_completion.json')
    assert len(launched)==(32 if gain else 8) and result['resolved_slots']==32
    assert len(result['exits'])==len(launched) and all(p.finished for p in launched)
    assert result['actual_tasks']==len(launched) and result['physical_points']==len(launched)
    assert result['all_eight_context_mean_allowed'] is gain
    assert len(result['not_run'])==(0 if gain else 24)
    assert all(r['status']=='NOT_RUN_GATE_NO_GAIN' and r['quality_pct'] is None for r in result['not_run'])
    assert all(not (run/r['cell']/'result.json').exists() for r in result['not_run'])
    assert all(runner.sha(run/name/'completion_manifest.json')==value for name,value in first_hashes.items())
    if gain:
        assert runner.read(run/'controller_stage2_started.json')['pid']==7102
        for spec in result['dispatched']:
            expected=7101 if spec['cell'] in first['completed'] else 7102
            assert runner.read(run/spec['cell']/'execution.json')['controller_pid']==expected
    with pytest.raises(ValueError,match='No repeated second-stage'):runner.execute_stage2(run)


def test_first_stage_failure_stops_next_pair_and_cannot_claim_gate_completion(tmp_path,monkeypatch):
    run,launched=controller_fixture(tmp_path,monkeypatch,failure=True)
    assert runner.execute_stage1(run)==1 and len(launched)==2
    result=runner.read(run/'controller_stage1_completion.json')
    assert result['status']=='STOPPED_INCOMPLETE' and result['errors'] and len(result['exits'])==2
    assert all(p.finished for p in launched) and not (run/'controller_completion.json').exists()


def gate_fixture(tmp_path,*,gain=True):
    run=tmp_path/'gate';run.mkdir()
    runner.save(run/'prephysical_freeze_manifest.json',{'test':'frozen'})
    runner.save(run/'stage_plan.json',{'phase1_cells':['episodes/000'],'phase2_cells':['episodes/001']})
    cell=run/'episodes/000'
    for name,energy in [('EARLY',11. if gain else 10.),('GUARD',10.)]:
        runner.save(cell/name/'phase_result.json',{'fully_certified':True,'best_plan':{'certified_complete':True,
            'validation':{'status':'COMPLETE','energy_kwh':energy}}})
    runner.save(cell/'result.json',{'test':'result'});runner.seal(cell)
    runner.save(run/'controller_stage1_completion.json',{'status':'COMPLETE_STAGE1_PENDING_AUDIT','completed':['episodes/000'],
        'exits':[{'exit_code':0}],'errors':[],'unknown_cost_cells':[]})
    audit={'status':'PASS','stage':'phase1','audited_cells':1,'errors':[],
        'freeze_sha256':runner.sha(run/'prephysical_freeze_manifest.json'),'stage_plan_sha256':runner.sha(run/'stage_plan.json'),
        'controller_stage1_completion_sha256':runner.sha(run/'controller_stage1_completion.json'),
        'results':[{'cell':'episodes/000','status':'PASS','cell_completion_sha256':runner.sha(cell/'completion_manifest.json'),
            'result_sha256':runner.sha(cell/'result.json'),'early_energy_kwh':11. if gain else 10.,'guard_energy_kwh':10.,
            'early_fully_certified':True,'guard_established':True}],
        'gate':{'phase2_authorized':gain,'positive_cells':['episodes/000'] if gain else []}}
    runner.save(run/'audit_stage1/summary.json',audit);runner.seal(run/'audit_stage1')
    return run


@pytest.mark.parametrize('gain',[False,True])
def test_gate_rechecks_bound_complete_energies_and_rejects_changed_controller(tmp_path,gain):
    run=gate_fixture(tmp_path,gain=gain)
    assert runner.phase1_gate(run)['phase2_authorized'] is gain
    value=runner.read(run/'controller_stage1_completion.json');value['exits'][0]['exit_code']=1
    runner.save(run/'controller_stage1_completion.json',value)
    with pytest.raises(ValueError,match='binding changed'):runner.phase1_gate(run)
