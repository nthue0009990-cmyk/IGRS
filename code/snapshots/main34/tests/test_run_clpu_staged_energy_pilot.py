"""Focused new-runner boundaries; only synthetic physical services are used."""
import ast
from collections import Counter
from copy import deepcopy
import inspect
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_clpu_staged_energy_pilot as runner
from dnr.nature34_clpu_branch import BranchConfig
from dnr.nature34_clpu_shared_evidence import SharedEvidenceService
from dnr.nature34_clpu_staged_pickup import generate_staged_pickup_candidates
from test_nature34_clpu_shared_evidence import build
from test_nature34_clpu_branch import environment
from test_run_clpu_deferred_pilot import FakePolicy, models, rows
from test_run_clpu_shared_pilot import contexts


def test_all32_fixed_population_keeps_every_context_seed_and_no_graph_gate():
    schedule=runner.schedule_for(contexts(),models())
    assert len(schedule)==32 and Counter(s['arm'] for s in schedule)=={'energy_window':8,'staged_bc':24}
    assert schedule==runner.schedule_for(list(reversed(contexts())),models())
    for slot in runner.SLOTS:
        assert Counter(s['arm_position'] for s in schedule if s['arm_slot']==slot)=={i:2 for i in range(4)}
    profile=runner.read(runner.GRAPH_PROFILE/'all_contexts.json')
    plan=runner.partition_schedule(schedule,profile)
    assert plan['eligible_contexts']==[2,7] and len(plan['all_cells'])==32 and plan['quality_gate'] is False
    assert not {'phase1_cells','phase2_cells'} & set(plan)
    assert runner.settings()['total_physical_budget']==16384 and runner.settings()['maximum_numerical_entries']==65632
    assert not {'conditional_stages','no_gain_remaining_status'} & set(runner.settings())
    assert set(runner.source_names())==set(runner.base.source_names())|set(runner.ADDITIONS)
    assert not any('report_clpu_staged_energy' in p for p in runner.source_names())


def test_driver_ast_diff_is_exactly_controller_import_and_old_module_is_unmodified(tmp_path):
    def driver_ast(function):
        node=ast.parse(inspect.getsource(function)).body[0]
        node.body=[v for v in node.body if not isinstance(v,ast.ImportFrom) or v.module not in
            ('dnr.nature34_clpu_staged_control','dnr.nature34_clpu_staged_energy_control')]
        return ast.dump(node,include_attributes=False)
    assert driver_ast(runner.run_routes)==driver_ast(runner.base.run_routes)
    assert runner.base.STUDY=='CLPU_STAGED_COMMON_PILOT_V1'
    assert runner.base.SLOTS[0]=='staged_window'
    assert runner.rebound('worker').__globals__ is runner._PRIVATE
    assert runner.rebound('worker').__globals__ is not runner.base.worker.__globals__
    assert runner._PRIVATE['run_routes'] is runner.run_routes
    original={'schema':'clpu-staged-protected-result-v1','study':runner.STUDY,'test_value':3}
    runner.save_output(tmp_path/'result.json',original)
    assert runner.read(tmp_path/'result.json')=={**original,'schema':runner.SCHEMA}
    assert original['schema']=='clpu-staged-protected-result-v1'


@pytest.mark.parametrize('arm',['energy_window','staged_bc'])
def test_new_selector_driver_keeps_shared_bootstrap_cold_views_and_complete_guard(build,arm):
    template=build('shared');env=environment(horizon=30)
    service=SharedEvidenceService(template.parent,template.cell,env,template.scope,mode='shared',physical_budget=512,logical_budget=256)
    spec={**next(s for s in runner.schedule_for(contexts(),models()) if s['arm']==arm),'cell':'task'}
    for phase in ('EARLY','GUARD'):runner.save(service.cell/phase/'spec.json',runner.phase_spec(spec,phase))
    policy=FakePolicy(env) if spec['checkpoint'] else None
    with service:
        phases,cps=runner.run_routes(service,env,BranchConfig(),service.cell,spec,time.perf_counter(),time.time()+60,
            policy,expected_generation=generate_staged_pickup_candidates(env))
    early,guard=phases['EARLY'],phases['GUARD']
    assert early['stage_generation']==generate_staged_pickup_candidates(env)
    assert early['control']['active_selector']==('complete_graph_energy' if arm=='energy_window' else 'fixed_raw_actor')
    assert early['verify_attempts']<=16 and early['graph_expansions']==0 and early['search_decisions']<=17
    assert list(service.views)==['EARLY','GUARD'] and service.closed
    assert cps[-1]['guard_established'] and all(c['numerical_entries']==3+4*c['physical_points'] for c in cps)
    assert sum(p['actual_physical_points'] for p in phases.values())==service.physical_points
    assert guard['verify_attempts'] is None and guard['policy_decisions']==0
    journal=rows(service.cell/'EARLY/decisions.jsonl.gz')
    assert len(journal)==2*early['search_decisions']
    if policy:assert policy.calls==early['policy_decisions']>0


def test_matched_source_copies_only_bound_small_inputs_and_rejects_changed_audit(tmp_path):
    run=tmp_path/'run';run.mkdir()
    previous=runner.COMPARISON_SOURCE
    contexts_value=runner.read(previous/'contexts.json')
    schedule=runner.schedule_for(contexts_value,runner.read(runner.MODELS_SOURCE/'model_bindings.json'))
    runner.save(run/'contexts.json',contexts_value)
    runner.save(run/'stage_plan.json',runner.partition_schedule(schedule,runner.read(previous/'stage_plan.json')['contexts']))
    runner.bind_comparison(run)
    assert {p.relative_to(run/'comparison_source').as_posix() for p in (run/'comparison_source').rglob('*') if p.is_file()}==set(runner.COMPARISON_FILES)
    assert not (run/'comparison_source/episodes').exists()
    changed=runner.read(run/'comparison_source/audit/summary.json');changed['audited_cells']=31
    runner.save(run/'comparison_source/audit/summary.json',changed)
    with pytest.raises(ValueError,match='Changed comparison small input'):
        runner.verify_comparison(run)


def controller_fixture(tmp_path,monkeypatch,failure):
    run=tmp_path/'run';run.mkdir();schedule=runner.schedule_for(contexts(),models())
    runner.save(run/'schedule.json',schedule);runner.save(run/'stage_plan.json',{'all_cells':[s['cell'] for s in schedule]})
    runner.save(run/'resource_budget.json',{'historical_max_bytes_per_actual_point':1})
    for spec in schedule:runner.save(run/spec['cell']/'spec.json',spec)
    monkeypatch.setattr(runner,'verify_execution',lambda _:None)
    monkeypatch.setattr(runner,'ACTIVE',tmp_path/'active.json')
    monkeypatch.setitem(runner._PRIVATE,'ACTIVE',tmp_path/'active.json')
    monkeypatch.setattr(runner.base.shutil,'disk_usage',lambda _:SimpleNamespace(free=10**15))
    monkeypatch.setattr(runner.base.time,'sleep',lambda _:None)
    launched=[]
    class Process:
        def __init__(self,cmd,**kwargs):
            assert Path(cmd[2]).name=='run_clpu_staged_energy_pilot.py'
            assert sum(not p.finished for p in launched)<2
            if len(launched)%2==0:assert all(p.finished for p in launched)
            self.cell=Path(cmd[-1]);self.pid=500+len(launched);self.calls=0;self.finished=False
            assert all(p.cell!=self.cell for p in launched)
            self.code=17 if failure and not launched else 0;launched.append(self)
        def poll(self):
            self.calls+=1
            if self.calls<2:return None
            if not self.finished:
                runner.save(self.cell/'result.json',{'logical_points':2,'actual_physical_points':1,'numerical_entries':7,'requests':3})
                runner.save(self.cell/'worker_finished.json',{'status':'COMPLETE'});runner.seal(self.cell);self.finished=True
            return self.code
    monkeypatch.setattr(runner.base.subprocess,'Popen',Process)
    return run,launched


@pytest.mark.parametrize('failure',[False,True])
def test_single_round_all32_or_error_pair_drain_without_gate_or_fake_completion(tmp_path,monkeypatch,failure):
    run,launched=controller_fixture(tmp_path,monkeypatch,failure)
    assert runner.execute(run)==int(failure)
    result=runner.read(run/'controller_completion.json')
    assert len(launched)==(2 if failure else 32) and all(p.finished for p in launched)
    assert result['actual_tasks']==len(launched) and result['physical_points']==len(launched)
    assert result['not_run']==[] and result['conditional_gate'] is False
    assert result['controller_execution_sha256']==runner.sha(run/'controller_stage1_completion.json')
    assert result['all_eight_context_mean_allowed'] is (not failure)
    assert not any((run/p).exists() for p in ('audit_stage1','phase_gate.json','controller_stage2_started.json'))
    assert result['status']==('STOPPED_INCOMPLETE' if failure else 'COMPLETE_CELLS_PENDING_AUDIT')
    assert result['resolved_slots']==(1 if failure else 32)
    with pytest.raises(ValueError,match='No automatic rerun'):runner.execute(run)
