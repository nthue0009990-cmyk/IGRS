"""Only the new selector/adapter/population contracts; no solver or training."""
import ast
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time
import os
from collections import Counter
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_clpu_staged_energy_pilot as audit
from dnr.nature34_clpu_branch import BranchConfig
from dnr.nature34_clpu_staged_control import StagedCertificationSearch
from dnr.nature34_clpu_staged_energy_control import EnergyStagedMacroController
from test_nature34_clpu_branch_audit import inputs
from test_clpu_deferred_audit import LogicalActualOracle,IndependentFirstRanker
from test_clpu_staged_audit import full_env,mechanics
from test_run_clpu_deferred_pilot import FakePolicy
from test_nature34_clpu_shared_evidence import build


def test_independent_energy_order_exact_tie_and_no_prefix_shortcut():
    nodes=[{'id':9,'prefix':tuple(range(10)),'info':{'energy_kwh':12.}},
        {'id':1,'prefix':(1,),'info':{'energy_kwh':10.}}]
    assert audit.visible_energy_choice(nodes)==0
    nodes[1]['info']['energy_kwh']=12.
    assert audit.visible_energy_choice(nodes)==1
    nodes[0]['info']['energy_kwh']=math.nextafter(12.,math.inf)
    assert audit.visible_energy_choice(nodes)==0


def saved(inputs,method):
    env=full_env(inputs)
    cfg=asdict(BranchConfig(max_branch_expansions=20,max_checked_candidates=20,max_generated_candidates=60))
    oracle=LogicalActualOracle(env,256)
    engine=StagedCertificationSearch(env,oracle,BranchConfig(**cfg))
    controller=EnergyStagedMacroController(engine,method=method,policy=FakePolicy(env) if method!='beam' else None)
    records=[]
    while (proposal:=controller.prepare()) is not None:
        records.append({'proposal':deepcopy(proposal),'response':controller.apply(proposal)})
    return {'mech':mechanics(env),'cfg':cfg,'method':method,'records':records,
        'result':json.loads(json.dumps(controller.result())), 'raw':oracle.raw,'requests':oracle.rows,
        'logical_rows':[{'reuse_kind':'ACTUAL_PHYSICAL'} for _ in oracle.raw]}


def replay(data):
    oracle=audit.branch.PaidRequests(data['requests'],data['raw'],data['mech'],256,'synthetic')
    search=audit.EnergyReplay(data['mech'],data['cfg'],oracle,data['result']['attempts'],data['logical_rows'])
    controller=audit.EnergyMacroReplay(search,'staged',data['method'])
    for record in data['records']:
        proposal=controller.check_proposal(record['proposal'],IndependentFirstRanker() if data['method']!='beam' else None)
        expected=deepcopy(record['response']);expected.pop('elapsed_seconds')
        audit.same(controller.apply(proposal),expected,'energy_selector_event')
    return controller.check_control_result(data['result'])


@pytest.mark.parametrize('method',['beam','bc_raw'])
def test_new_controller_independent_full_trajectory_preserves_visibility_features_and_paid_requests(inputs,method):
    result=replay(saved(inputs,method))
    assert result['graph_expansions']==0 and result['verify_attempts']<=16 and result['total_decisions']<=17


@pytest.mark.parametrize('mutation',['action','future_observation','new_rule_claim','different_pool'])
def test_changed_choice_observation_selector_or_pool_rejected(inputs,mutation):
    data=saved(inputs,'beam')
    first=data['records'][0]['proposal']
    if mutation=='action':first['action']=(first['action']+1)%len(first['observation']['mask'])
    elif mutation=='future_observation':first['observation']['global_features'][0]+=.5
    elif mutation=='new_rule_claim':data['result']['control']['beam_rule']['prefix_depth_priority']=True
    else:data['result']['stage_generation']['candidates'][0]['actions'][0]=0
    with pytest.raises(AssertionError):replay(data)


def test_no_production_rule_search_runner_import_and_no_old_global_mutation():
    before=audit.staged._check_records.__globals__['MacroReplay']
    assert before is audit.staged.StagedMacroReplay
    assert audit._records.__globals__['MacroReplay'] is audit.EnergyMacroReplay
    tree=ast.parse(Path(audit.__file__).read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        names=[n.name for n in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
        assert not any(n.startswith('dnr.') or n.startswith('run_clpu_staged') for n in names)


def test_new_fixed_contract_matches_runner_and_comparison_is_bound(tmp_path):
    import shutil
    import run_clpu_staged_energy_pilot as runner
    from test_run_clpu_deferred_pilot import contexts,models
    assert audit.protocol_settings()==runner.settings()
    assert audit.expected_schedule(contexts(),models())==runner.schedule_for(contexts(),models())
    previous=runner.COMPARISON_SOURCE
    rows=runner.read(previous/'stage_plan.json')['contexts']
    actual_contexts=runner.read(previous/'contexts.json')
    specs=runner.schedule_for(actual_contexts,runner.read(runner.MODELS_SOURCE/'model_bindings.json'))
    runner.save(tmp_path/'contexts.json',actual_contexts);runner.save(tmp_path/'schedule.json',specs)
    runner.save(tmp_path/'stage_plan.json',runner.partition_schedule(specs,rows))
    runner.bind_comparison(tmp_path)
    shutil.copytree(runner.GRAPH_PROFILE,tmp_path/'graph_profile')
    shutil.copyfile(previous/'graph_profile_binding.json',tmp_path/'graph_profile_binding.json')
    audit.check_comparison(tmp_path)
    changed=runner.read(tmp_path/'stage_plan.json');changed['quality_gate']=True
    runner.save(tmp_path/'stage_plan.json',changed)
    with pytest.raises(AssertionError):audit.check_comparison(tmp_path)


def test_new_worker_through_private_audit_adapters_and_full_cold_ledger(build,inputs,monkeypatch):
    import run_clpu_staged_energy_pilot as runner
    import run_nature34_clpu_pilot as physical_runner
    from dnr import nature34_clpu_shared_evidence as service_module
    from dnr.nature34_clpu_shared_evidence import SharedEvidenceService
    from dnr.nature34_clpu_staged_pickup import generate_staged_pickup_candidates
    from test_run_clpu_deferred_pilot import contexts,models
    from test_clpu_shared_audit import material
    template=build('shared');env=full_env(inputs);run,cell=template.parent,template.cell
    runner.save(run/'condition.json',env.condition)
    service=SharedEvidenceService(run,cell,env,template.scope,mode='shared',physical_budget=512,logical_budget=256)
    spec={**next(s for s in runner.schedule_for(contexts(),models()) if s['arm']=='energy_window'),
        'cell':'task','scenario_id':env.scenario['scenario_id'],'event_id':env.event['event_id']}
    cfg={'horizon':30,'slot_seconds':60,'sample_seconds':30,'branch_search':asdict(BranchConfig(
        max_branch_expansions=20,max_checked_candidates=20,max_generated_candidates=60))}
    runner.save(cell/'spec.json',spec)
    generation=generate_staged_pickup_candidates(env)
    for name,value in {'schedule.json':[spec],'config.json':cfg,'physical_scope.json':service.scope,
        'author_static_metadata.json':env.static,'load_reference.json':env.loads,'selected_scenarios.json':[env.scenario],
        'selected_events.json':[env.event],'command_inventory.json':{},
        'stage_plan.json':{'contexts':[{'context_id':spec['context_id'],'generation':generation,'generation_sha256':audit.digest(generation)}]},
        'controller_stage1_started.json':{'pid':os.getpid(),'stage':'phase1'}}.items():runner.save(run/name,value)
    for phase in audit.shared.VIEWS:runner.save(cell/phase/'spec.json',runner.phase_spec(spec,phase))
    epoch=time.time()
    runner.save(run/'execution_started.json',{'pid':os.getpid(),'started_epoch_seconds':epoch,'deadline_epoch_seconds':epoch+3600})
    runner.save(cell/'execution.json',{'controller_pid':os.getpid(),'started_epoch_seconds':epoch,
        'deadline_epoch_seconds':epoch+900,'spec_sha256':audit.sha(cell/'spec.json'),'stage':'phase1'})
    monkeypatch.setitem(runner._PRIVATE,'_WORKER_ENTERED',False)
    monkeypatch.setitem(runner._PRIVATE,'verify_execution',lambda _:None)
    monkeypatch.setattr(runner.base.shutil,'disk_usage',lambda _:SimpleNamespace(free=10**15))
    monkeypatch.setattr(physical_runner,'make_env',lambda *_:env)
    monkeypatch.setattr(service_module,'SharedEvidenceService',lambda *a,**k:service)
    assert runner.worker(run,cell)==0
    (cell/'physical/command_ledger.jsonl').write_text('',encoding='utf-8');runner.seal(cell)
    runner.save(run/'historical/guard.json',audit.read(cell/'GUARD/phase_result.json'))
    runner.save(run/'historical_bindings.json',[{'context_id':spec['context_id'],'phases':{'GUARD':{'result_file':'historical/guard.json'}}}])
    mech=mechanics(env);data=material(service,mech)
    for point in data['points'].values():point['elapsed_seconds']=0.
    statuses=Counter(p['status'] for p in data['points'].values())
    monkeypatch.setattr(audit.shared,'check_actual_points',lambda *a:(data['points'],statuses))
    monkeypatch.setattr(audit.old,'check_commands',lambda *a:{'numerical_entries':3+4*len(data['points'])})
    monkeypatch.setattr(audit.shared,'check_actual_requests',lambda *a:None)
    checked=audit.check_cell(run,spec,cfg,service.scope)
    assert checked['guard_established'] and checked['numerical_entries']==3+4*checked['physical_points']
    assert audit.read(cell/'result.json')['schema']==audit.SCHEMA
    assert audit.staged._check_base_cell.__globals__['STUDY']==audit.staged.STUDY


@pytest.mark.parametrize('mutation',[None,'missing','fake_no_gain','failed_exit','wrong_cost'])
def test_all32_controller_rejects_conditional_omission_or_unpaid_work(tmp_path,monkeypatch,mutation):
    import run_clpu_staged_energy_pilot as runner
    from test_run_clpu_staged_energy_pilot import controller_fixture
    run,_=controller_fixture(tmp_path,monkeypatch,False)
    assert runner.execute(run)==0
    schedule=runner.read(run/'schedule.json');controller=runner.read(run/'controller_completion.json')
    checks=[]
    for spec in schedule:
        ex=runner.read(run/spec['cell']/'execution.json')
        checks.append({**spec,'logical_points':2,'physical_points':1,'numerical_entries':7,'requests':3,
            'execution':ex,'worker_finished_elapsed_seconds':0.,'worker_parent_pid':ex['controller_pid']})
    if mutation=='missing':checks.pop()
    elif mutation=='fake_no_gain':controller['not_run']=[{'cell':schedule[-1]['cell'],'status':'NOT_RUN_GATE_NO_GAIN'}]
    elif mutation=='failed_exit':controller['exits'][0]['exit_code']=9
    elif mutation=='wrong_cost':controller['numerical_entries']-=3
    runner.save(run/'controller_completion.json',controller)
    if mutation:
        with pytest.raises(AssertionError):audit.check_controller(run,schedule,checks)
    else:assert audit.check_controller(run,schedule,checks)['numerical_entries']==224
