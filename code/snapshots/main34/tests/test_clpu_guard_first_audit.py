"""Independent external-floor semantics and reversed causal evidence; no PF."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import os
import time
import ast
from collections import Counter
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_clpu_guard_first_pilot as audit
from dnr.nature34_clpu_branch import BranchConfig,BranchSearch
from dnr.nature34_clpu_guard_first_control import GuardFirstStagedSearch,GuardFirstMacroController
from test_nature34_clpu_branch_audit import inputs
from test_clpu_staged_audit import full_env,mechanics
from test_clpu_deferred_audit import LogicalActualOracle,IndependentFirstRanker
from test_run_clpu_deferred_pilot import FakePolicy
from test_nature34_clpu_shared_evidence import build


def saved(inputs,method='beam',guard_budget=61):
    env=full_env(inputs)
    config=BranchConfig(max_branch_expansions=20,max_checked_candidates=20,max_generated_candidates=60)
    guard_oracle=LogicalActualOracle(env,guard_budget)
    guard=BranchSearch(env,guard_oracle,BranchConfig(**{**asdict(config),'point_budget':guard_budget}))
    while not guard.done:guard.step(guard.rule_action('beam'))
    donor=json.loads(json.dumps(guard.result()));donor['fully_certified']=donor['best_plan'] is not None
    reference={'phase':'GUARD','file':'GUARD/phase_result.json','sha256':audit.digest(donor)}
    oracle=LogicalActualOracle(env,256)
    engine=GuardFirstStagedSearch(env,oracle,config,guard_result=donor,guard_reference=reference)
    controller=GuardFirstMacroController(engine,method=method,policy=FakePolicy(env) if method!='beam' else None)
    records=[]
    while (proposal:=controller.prepare()) is not None:
        records.append({'proposal':deepcopy(proposal),'response':controller.apply(proposal)})
    return {'mech':mechanics(env),'cfg':asdict(config),'method':method,'guard':donor,'reference':reference,
        'records':records,'result':json.loads(json.dumps(controller.result())),'raw':oracle.raw,
        'requests':oracle.rows,'logical_rows':[{'reuse_kind':'ACTUAL_PHYSICAL'} for _ in oracle.raw]}


def replay(data):
    oracle=audit.branch.PaidRequests(data['requests'],data['raw'],data['mech'],256,'synthetic')
    search=audit.GuardFirstReplay(data['mech'],data['cfg'],oracle,data['result']['attempts'],data['logical_rows'],
        guard_result=data['guard'],guard_reference=data['reference'])
    controller=audit.GuardFirstMacroReplay(search,'staged',data['method'])
    for record in data['records']:
        proposal=controller.check_proposal(record['proposal'],IndependentFirstRanker() if data['method']!='beam' else None)
        expected=deepcopy(record['response']);expected.pop('elapsed_seconds')
        audit.same(controller.apply(proposal),expected,'guard_first_causal_event')
    return controller.check_control_result(data['result'])


@pytest.mark.parametrize('method,guard_budget',[('beam',61),('bc_raw',61),('beam',256)])
def test_external_certified_floor_pruning_and_observations_replay_independently(inputs,method,guard_budget):
    data=saved(inputs,method,guard_budget)
    result=replay(data)
    assert result['guard_established'] and result['complete_guard_still_required'] is False
    assert result['guard_first']['own_best_plan_namespace']=='EARLY'
    assert all(p['physical_feasibility']=='UNASSESSED' and not p['consumed_verification_choice']
        for p in result['guard_first']['free_pruning'])


@pytest.mark.parametrize('mutation',['incomplete_guard','external_alias','prune_as_fail','fake_query'])
def test_external_floor_and_free_pruning_mutations_are_rejected(inputs,mutation):
    data=saved(inputs,guard_budget=256)
    if mutation=='incomplete_guard':data['guard']['done']=False
    elif mutation=='external_alias':data['result']['guard_first']['own_best_plan_namespace']='GUARD'
    elif mutation=='prune_as_fail':data['result']['guard_first']['free_pruning'][0]['physical_feasibility']='FAIL'
    else:data['result']['guard_first']['free_pruning'][0]['logical_points']+=1
    with pytest.raises(AssertionError):replay(data)


def test_fixed32_contract_and_previous_energy_source_are_bound(tmp_path):
    import shutil
    import run_clpu_guard_first_pilot as runner
    from test_run_clpu_deferred_pilot import contexts,models
    assert audit.protocol_settings()==runner.settings()
    assert audit.expected_schedule(contexts(),models())==runner.schedule_for(contexts(),models())
    previous=runner.COMPARISON_SOURCE
    cs=runner.read(previous/'contexts.json')
    specs=runner.schedule_for(cs,runner.read(runner.MODELS_SOURCE/'model_bindings.json'))
    runner.save(tmp_path/'contexts.json',cs);runner.save(tmp_path/'schedule.json',specs)
    runner.save(tmp_path/'stage_plan.json',runner.partition_schedule(specs,runner.read(previous/'stage_plan.json')['contexts']))
    runner.bind_comparison(tmp_path)
    shutil.copytree(runner.GRAPH_PROFILE,tmp_path/'graph_profile')
    shutil.copyfile(previous/'graph_profile_binding.json',tmp_path/'graph_profile_binding.json')
    audit.check_comparison(tmp_path)
    plan=runner.read(tmp_path/'stage_plan.json');plan['all_cells'].pop()
    runner.save(tmp_path/'stage_plan.json',plan)
    with pytest.raises(AssertionError):audit.check_comparison(tmp_path)


@pytest.mark.parametrize('arm',['guard_first_window','staged_bc'])
def test_new_worker_complete_independent_ledger_floor_and_schema(build,inputs,monkeypatch,arm):
    import run_clpu_guard_first_pilot as runner
    import run_nature34_clpu_pilot as physical_runner
    import nature34_clpu_tail_execution as tail
    from dnr import nature34_clpu_guard_first_control as production
    from dnr.nature34_clpu_staged_pickup import generate_staged_pickup_candidates
    from test_run_clpu_deferred_pilot import contexts,models
    from test_clpu_shared_audit import material
    template=build('shared');env=full_env(inputs);run,cell=template.parent,template.cell
    runner.save(run/'condition.json',env.condition)
    service=production.GuardFirstEvidenceService(run,cell,env,template.scope,mode='shared',physical_budget=512,logical_budget=256)
    spec={**next(s for s in runner.schedule_for(contexts(),models()) if s['arm']==arm),
        'cell':'task','scenario_id':env.scenario['scenario_id'],'event_id':env.event['event_id']}
    if spec['checkpoint']:
        model=run/spec['checkpoint'];model.parent.mkdir(parents=True,exist_ok=True);model.write_bytes(b'fixed mock model')
        spec['checkpoint_sha256']=runner.sha(model)
    cfg={'horizon':30,'slot_seconds':60,'sample_seconds':30,'branch_search':asdict(BranchConfig(
        max_branch_expansions=20,max_checked_candidates=20,max_generated_candidates=60))}
    runner.save(cell/'spec.json',spec)
    generation=generate_staged_pickup_candidates(env)
    for name,value in {'schedule.json':[spec],'config.json':cfg,'physical_scope.json':service.scope,
        'author_static_metadata.json':env.static,'load_reference.json':env.loads,'selected_scenarios.json':[env.scenario],
        'selected_events.json':[env.event],'command_inventory.json':{},
        'stage_plan.json':{'contexts':[{'context_id':spec['context_id'],'generation':generation,'generation_sha256':audit.digest(generation)}]},
        'controller_stage1_started.json':{'pid':os.getpid(),'stage':'phase1'}}.items():runner.save(run/name,value)
    for phase in audit.VIEWS:runner.save(cell/phase/'spec.json',runner.phase_spec(spec,phase))
    epoch=time.time()
    runner.save(run/'execution_started.json',{'pid':os.getpid(),'started_epoch_seconds':epoch,'deadline_epoch_seconds':epoch+3600})
    runner.save(cell/'execution.json',{'controller_pid':os.getpid(),'started_epoch_seconds':epoch,
        'deadline_epoch_seconds':epoch+900,'spec_sha256':audit.sha(cell/'spec.json'),'stage':'phase1'})
    monkeypatch.setitem(runner._PRIVATE,'_WORKER_ENTERED',False)
    monkeypatch.setitem(runner._PRIVATE,'verify_execution',lambda _:None)
    monkeypatch.setattr(runner.base.shutil,'disk_usage',lambda _:SimpleNamespace(free=10**15))
    monkeypatch.setattr(physical_runner,'make_env',lambda *_:env)
    monkeypatch.setattr(production,'GuardFirstEvidenceService',lambda *a,**k:service)
    monkeypatch.setattr(tail,'load_fixed_policy',lambda path,s:(FakePolicy(env),None) if path else (None,None))
    monkeypatch.setattr(audit.early,'IndependentPolicy',lambda *a:IndependentFirstRanker())
    assert runner.worker(run,cell)==0
    (cell/'physical/command_ledger.jsonl').write_text('',encoding='utf-8');runner.seal(cell)
    runner.save(run/'historical/guard.json',audit.read(cell/'GUARD/phase_result.json'))
    runner.save(run/'historical_bindings.json',[{'context_id':spec['context_id'],'phases':{'GUARD':{'result_file':'historical/guard.json'}}}])
    mech=mechanics(env);data=material(service,mech)
    data['views']={name:data['views'][name] for name in audit.VIEWS}
    for point in data['points'].values():point['elapsed_seconds']=0.
    statuses=Counter(p['status'] for p in data['points'].values())
    monkeypatch.setattr(audit.shared,'check_actual_points',lambda *a:(data['points'],statuses))
    monkeypatch.setattr(audit.old,'check_commands',lambda *a:{'numerical_entries':3+4*len(data['points'])})
    monkeypatch.setattr(audit.shared,'check_actual_requests',lambda *a:None)
    checked=audit.check_cell(run,spec,cfg,service.scope)
    assert checked['causal_guard_first_verified'] and checked['external_floor_namespace_verified']
    assert checked['guard_established'] and checked['numerical_entries']==3+4*checked['physical_points']
    assert checked['free_pruned_complete_plans']>0
    assert audit.read(cell/'result.json')['schema']==audit.SCHEMA
    # The same actual synthetic journals must fail if a later answer is cited,
    # or the old EARLY-first service chronology is substituted.
    changed=deepcopy(data);changed['views']['EARLY']['logical_rows'][0]['source_event_sequence']=999999
    with pytest.raises(AssertionError):audit.check_service_ledger(**changed)
    changed=deepcopy(data);changed['views']={name:changed['views'][name] for name in reversed(audit.VIEWS)}
    with pytest.raises(AssertionError):audit.check_service_ledger(**changed)
    result=audit.read(cell/'EARLY/phase_result.json')
    result['guard_first']['guard_reference']['sha256']='0'*64
    runner.save(cell/'EARLY/phase_result.json',result);runner.seal(cell)
    with pytest.raises(AssertionError,match='External floor cites'):audit.check_cell(run,spec,cfg,service.scope)


@pytest.mark.parametrize('mutation',['premature','withdrawn','reordered','missing_close'])
def test_persistent_floor_boundary_rejects_future_and_false_promises(mutation):
    cps=[{'phase':'GUARD','kind':'INITIAL','guard_established':False},
        {'phase':'GUARD','kind':'CLOSE','guard_established':True},
        {'phase':'EARLY','kind':'INITIAL','guard_established':True}]
    if mutation=='premature':cps[0]['guard_established']=True
    elif mutation=='withdrawn':cps[2]['guard_established']=False
    elif mutation=='reordered':cps=[cps[2],*cps[:2]]
    else:cps.pop(1)
    with pytest.raises(AssertionError):audit.check_checkpoints(cps,{}, {'GUARD':{'done':True,'fully_certified':True}})


def test_independence_and_original_global_state_are_preserved():
    assert audit.shared.VIEWS==('EARLY','GUARD')
    assert audit.staged._check_base_cell.__globals__['STUDY']==audit.staged.STUDY
    assert audit.energy.check_parent.__globals__['STUDY']==audit.energy.STUDY
    tree=ast.parse(Path(audit.__file__).read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        names=[n.name for n in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
        assert not any(n.startswith('dnr.') or n.startswith('run_clpu_') for n in names)
