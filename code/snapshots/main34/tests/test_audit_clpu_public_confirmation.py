"""Independent confirmation replay, provenance and synthetic boundary checks."""
import ast
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_clpu_public_confirmation as audit
import run_clpu_public_confirmation as runner
from dnr.nature34_clpu_branch import BranchConfig
from dnr.nature34_clpu_staged_pickup import generate_staged_pickup_candidates
from test_nature34_clpu_branch_audit import inputs
from test_clpu_staged_audit import full_env,mechanics
from test_nature34_clpu_shared_evidence import build
from test_clpu_shared_audit import material
from test_run_clpu_shared_pilot import contexts


@pytest.mark.parametrize('arm',audit.ARMS)
def test_real_new_worker_is_independently_replayed_with_actual_order(build,inputs,monkeypatch,arm):
    from dnr import nature34_clpu_guard_first_control as guard_module
    from dnr import nature34_clpu_shared_evidence as shared_module
    template=build('shared');env=full_env(inputs);run,cell=template.parent,template.cell
    runner.save(run/'condition.json',env.condition)
    Service=shared_module.SharedEvidenceService if arm=='E' else guard_module.GuardFirstEvidenceService
    service=Service(run,cell,env,template.scope,mode='shared',physical_budget=512,logical_budget=256)
    spec={**next(s for s in runner.schedule_for(runner.contexts_for()) if s['arm']==arm),'cell':'task',
        'scenario_id':env.scenario['scenario_id'],'event_id':env.event['event_id']}
    cfg={'horizon':30,'slot_seconds':60,'sample_seconds':30,'branch_search':asdict(BranchConfig(
        max_branch_expansions=20,max_checked_candidates=20,max_generated_candidates=60))}
    generation=generate_staged_pickup_candidates(env)
    for name,value in {'schedule.json':[spec],'config.json':cfg,'physical_scope.json':service.scope,
        'author_static_metadata.json':env.static,'load_reference.json':env.loads,'selected_scenarios.json':[env.scenario],
        'selected_events.json':[env.event],'command_inventory.json':{},
        'stage_plan.json':{'contexts':[{'context_id':spec['context_id'],'generation':generation,'generation_sha256':audit.digest(generation)}]},
        'controller_stage1_started.json':{'pid':os.getpid(),'stage':'phase1'}}.items():runner.save(run/name,value)
    runner.save(cell/'spec.json',spec)
    for phase in spec['phase_order']:runner.save(cell/phase/'spec.json',runner.phase_spec(spec,phase))
    epoch=time.time();runner.save(run/'execution_started.json',{'pid':os.getpid(),'started_epoch_seconds':epoch,'deadline_epoch_seconds':epoch+3600})
    runner.save(cell/'execution.json',{'controller_pid':os.getpid(),'started_epoch_seconds':epoch,
        'deadline_epoch_seconds':epoch+900,'spec_sha256':audit.sha(cell/'spec.json'),'stage':'phase1'})
    monkeypatch.setattr(runner,'_WORKER_ENTERED',False);monkeypatch.setattr(runner,'verify_execution',lambda _:None)
    monkeypatch.setattr(runner.shutil,'disk_usage',lambda _:SimpleNamespace(free=10**15))
    monkeypatch.setattr(runner,'make_env',lambda *_:env)
    monkeypatch.setattr(guard_module,'GuardFirstEvidenceService',lambda *a,**k:service)
    monkeypatch.setattr(shared_module,'SharedEvidenceService',lambda *a,**k:service)
    assert runner.worker(run,cell)==0
    (cell/'physical/command_ledger.jsonl').write_text('',encoding='utf-8');runner.seal(cell)
    mech=mechanics(env);data=material(service,mech)
    data['views']={name:data['views'][name] for name in spec['phase_order']}
    for point in data['points'].values():point['elapsed_seconds']=0.
    statuses=Counter(p['status'] for p in data['points'].values())
    # Only the fake numerical payload lacks a real physical certificate. The
    # complete causal ledger, re-certification, graph, selection, clocks,
    # full GUARD and schema are independently replayed without stubbing.
    monkeypatch.setattr(audit.shared,'check_actual_points',lambda *a:(data['points'],statuses))
    monkeypatch.setattr(audit.old,'check_commands',lambda *a:{'numerical_entries':3+4*len(data['points'])})
    monkeypatch.setattr(audit.shared,'check_actual_requests',lambda *a:None)
    checked=audit.check_cell(run,spec,cfg,service.scope)
    assert checked['history_verified'] is False and checked['guard_replay_verified'] is True
    assert not (run/'historical_bindings.json').exists()
    assert checked['phase_order']==spec['phase_order'] and checked['causal_phase_order_verified']
    assert checked['models_used']==checked['policy_decisions']==checked['model_load_seconds']==0
    assert checked['numerical_entries']==3+4*checked['physical_points'] and checked['guard_established']
    assert checked['candidate_pool_sha256']==audit.digest(generation)
    early=audit.read(cell/'EARLY/phase_result.json')
    if arm=='A_prune':
        assert checked['free_pruned_complete_plans']==0
        assert checked['own_incumbent_verify_skips']>0
        early['guard_first']['free_objective_pruning_enabled']=True
        runner.save(cell/'EARLY/phase_result.json',early);runner.seal(cell)
        with pytest.raises(AssertionError):audit.check_cell(run,spec,cfg,service.scope)
    elif arm=='P':
        assert checked['free_pruned_complete_plans']>0
        early['guard_first']['guard_reference']['sha256']='0'*64
        runner.save(cell/'EARLY/phase_result.json',early);runner.seal(cell)
        with pytest.raises(AssertionError,match='External floor cites'):audit.check_cell(run,spec,cfg,service.scope)
    else:
        result=audit.read(cell/'result.json');result['intermediate_beam_floor_claimed']=True
        runner.save(cell/'result.json',result)
        finished=audit.read(cell/'worker_finished.json');finished['result_sha256']=audit.sha(cell/'result.json')
        runner.save(cell/'worker_finished.json',finished);runner.seal(cell)
        with pytest.raises(AssertionError):audit.check_cell(run,spec,cfg,service.scope)


def complete_context_checks():
    return [dict(context_id=i,scenario_id='fault'+str(i//6),event_id=i%6,arm=arm,models_used=0,
        guard_established=True,history_verified=False,guard_replay_verified=True,
        candidate_pool_sha256=str(i),guard_quality_pct=50.+i,
        phase_summary={'GUARD':{'semantic_signature_sha256':str(i)}})
        for i in range(24) for arm in audit.ARMS]


@pytest.mark.parametrize('change',[None,'history','missing','model','pair_identity','guard','pool'])
def test_full_new_population_requires_actual_guard_without_old_history(change):
    checks=complete_context_checks()
    if change=='history':checks[0]['history_verified']=True
    elif change=='missing':checks.pop()
    elif change=='model':checks[0]['models_used']=1
    elif change=='pair_identity':checks[0]['scenario_id']='other'
    elif change=='guard':checks[0]['phase_summary']['GUARD']['semantic_signature_sha256']='different'
    elif change=='pool':checks[0]['candidate_pool_sha256']='other'
    if change:
        with pytest.raises(AssertionError):audit.context_equalities(checks)
    else:
        result=audit.context_equalities(checks)
        assert len(result)==24 and all(r['guard_trajectories_equal'] and not r['history_verified'] for r in result)


def test_no_production_import_old_module_mutation_or_history_substitution():
    assert audit.public.STUDY=='CLPU_PUBLIC_MECHANISM_COMMON_PILOT_V1'
    assert audit.gf.STUDY=='CLPU_GUARD_FIRST_COMMON_PILOT_V1'
    assert audit.shared.VIEWS==('EARLY','GUARD')
    source=Path(audit.__file__).read_text(encoding='utf-8')
    tree=ast.parse(source)
    for node in ast.walk(tree):
        names=[n.name for n in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
        assert not any(n.startswith('dnr.') or n.startswith('run_clpu_') for n in names)
    import inspect
    assert 'historical_bindings' not in inspect.getsource(audit._check_base_cell)
    assert 'check_actual_points' in inspect.getsource(audit._check_base_cell)
    assert 'check_phase_records' in inspect.getsource(audit._check_base_cell)


def test_new_identity_scope_matches_driver_without_reading_new_event_values():
    import run_clpu_public_confirmation as confirmation
    cs=audit.expected_contexts();schedule=audit.expected_schedule(cs)
    assert cs==confirmation.contexts_for()
    assert schedule==confirmation.schedule_for(cs)
    assert audit.protocol_settings()==confirmation.settings()
    assert audit.expected_source_binding()==confirmation.source_binding()
    inherited=audit.read(audit.ROOT/'runs'/audit.SOURCE_RUN/'prepared.json')['source_files']
    assert audit.expected_source_closure(inherited)==confirmation.source_names()
    assert Counter(s['arm'] for s in schedule)=={'P':24,'E':24}
    assert len({(s['scenario_id'],s['event_id']) for s in schedule})==24
    assert all(s['fault_partition']=='formal_test' and s['event_partition']=='holdout' for s in schedule)
    for i in range(0,48,2):
        assert schedule[i]['context_id']==schedule[i+1]['context_id']
        assert {schedule[i]['arm'],schedule[i+1]['arm']}=={'P','E'}
    for key,value in [('scenario_id','different'),('fault_partition','build'),('event_partition','train')]:
        changed=deepcopy(cs);changed[0][key]=value
        with pytest.raises(AssertionError):audit.expected_schedule(changed)


@pytest.mark.parametrize('change',[None,'fault_partition','fault_data','event_value','event_validation','split','event_duplicate'])
def test_original_source_projection_rejects_relabelled_or_edited_inputs(change):
    faults=[{'scenario_id':sid,'partition':'formal_test','fault_order':2,'fault_lines':[f'fixture{i}']}
        for i,sid in enumerate(audit.FAULT_IDS)]
    # Entirely synthetic dictionaries; these are not imported event parameters.
    events=[{'event_id':i,'synthetic_fixture':i*7,'validation':{'passed':True}} for i in range(1,32)]
    chosen_f=deepcopy(faults);chosen_e=[deepcopy(r) for eid in audit.EVENT_IDS for r in events if r['event_id']==eid]
    split={'holdout':list(audit.EVENT_IDS),'train':list(set(range(1,32))-set(audit.EVENT_IDS)),'dev':[]}
    if change=='fault_partition':chosen_f[0]['partition']='build'
    elif change=='fault_data':chosen_f[0]['fault_lines']=['replacement']
    elif change=='event_value':chosen_e[0]['synthetic_fixture']+=1
    elif change=='event_validation':chosen_e[0]['validation']['passed']=False
    elif change=='split':split['train'].append(audit.EVENT_IDS[0])
    elif change=='event_duplicate':events[-1]=deepcopy(events[0])
    if change:
        with pytest.raises(AssertionError):audit.check_selected_inputs(faults,events,chosen_f,chosen_e,split)
    else:audit.check_selected_inputs(faults,events,chosen_f,chosen_e,split)


def controller_material():
    specs=audit.expected_schedule(audit.expected_contexts());checks=[];dispatch=[];exits=[]
    for i,spec in enumerate(specs):
        epoch=100.+2*(i//2);pid=1000+i
        execution=dict(stage='phase1',started_epoch_seconds=epoch,controller_pid=50)
        checks.append(dict(cell=spec['cell'],execution=execution,worker_finished_elapsed_seconds=.5,worker_parent_pid=pid,
            logical_points=2,physical_points=1,numerical_entries=7,requests=3))
        dispatch.append(dict(cell=spec['cell'],stage='phase1',launcher_pid=pid,pair_order=spec['pair_order'],started_epoch_seconds=epoch))
        exits.append(dict(cell=spec['cell'],exit_code=0,drained_after_controller_error=False,launcher_pid=pid,
            pair_order=spec['pair_order'],started_epoch_seconds=epoch,finished_epoch_seconds=epoch+1,process_wall_seconds=1.))
    stage=dict(status='COMPLETE_STAGE1_PENDING_AUDIT',stage='phase1',scheduled=48,scheduled_total=48,
        selected_cells=[s['cell'] for s in specs],dispatched=specs,completed=[s['cell'] for s in specs],
        errors=[],unknown_cost_cells=[],peak_active=2,exits=exits,logical_points=96,physical_points=48,
        numerical_entries=336,requests=144,wall_seconds=48.)
    controller={**stage,'status':'COMPLETE_CELLS_PENDING_AUDIT','resolved_slots':48,'actual_tasks':48,'not_run':[],
        'all_fixed_context_mean_allowed':True,'controller_execution_sha256':'seal'}
    return specs,checks,dispatch,stage,controller


@pytest.mark.parametrize('change',[None,'failed_exit','cost','overlap','partial'])
def test_controller_requires_all48_real_cost_and_pair_drain(monkeypatch,tmp_path,change):
    specs,checks,dispatch,stage,controller=controller_material()
    if change=='failed_exit':stage['exits'][0]['exit_code']=1
    elif change=='cost':stage['physical_points']=47
    elif change=='overlap':
        for item in (stage['exits'][2],dispatch[2],checks[2]['execution']):item['started_epoch_seconds']=100.5
    elif change=='partial':checks.pop()
    values={'controller_stage1_completion.json':stage,'controller_completion.json':controller,'controller_stage1_exits.json':stage['exits']}
    monkeypatch.setattr(audit,'read',lambda p:values[Path(p).name]);monkeypatch.setattr(audit,'read_lines',lambda p:iter(dispatch))
    monkeypatch.setattr(audit,'sha',lambda p:'seal')
    if change:
        with pytest.raises(AssertionError):audit.check_controller(tmp_path,specs,checks)
    else:assert audit.check_controller(tmp_path,specs,checks)==dict(logical_points=96,physical_points=48,numerical_entries=336,requests=144)
