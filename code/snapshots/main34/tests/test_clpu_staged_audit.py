"""Independent staged population, graph, policy, cost and conditional gates; zero PF."""
import ast
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sys
import time
import os
from collections import Counter
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_clpu_staged_pilot as audit
import run_clpu_staged_pilot as runner
from dnr.nature34_clpu_branch import BranchConfig
from dnr.nature34_clpu_staged_pickup import generate_staged_pickup_candidates
from test_nature34_clpu_branch import environment
from test_nature34_clpu_branch_audit import inputs
from dnr.nature34_clpu_environment import CLPUEnvironment
from dnr.nature34_clpu_shared_evidence import SharedEvidenceService
from test_run_clpu_deferred_pilot import FakePolicy, contexts, models
from test_nature34_clpu_shared_evidence import build
from test_run_clpu_staged_pilot import route_fixture
from test_clpu_shared_audit import material
from test_clpu_deferred_audit import IndependentFirstRanker
from test_nature34_clpu_equivalent_cache import lines


def mechanics(env):
    # The legacy cache keys graph identity, not requested DP depth. The old
    # imported fixture intentionally built depth two; this test needs six.
    audit.branch._graphs.clear()
    return audit.branch.Mechanics(env.static, env.loads, env.scenario, env.condition, env.event,
        {'horizon':30,'slot_seconds':60,'sample_seconds':30,'branch_search':asdict(BranchConfig())})


def full_env(inputs):
    old=deepcopy(inputs[1])
    return CLPUEnvironment(old.graph,old.static,old.loads,old.scenario,old.condition,old.event,horizon=30)


@pytest.mark.parametrize('tau', [20.,120.,121.,1800.])
def test_independent_full_generation_preserves_all_preparations_waits_and_whole_loads(inputs,tau):
    env=full_env(inputs);env.event['tau_seconds']=tau
    actual=generate_staged_pickup_candidates(env)
    rebuilt=audit.independently_generate(mechanics(env))
    audit.check_generation(actual,rebuilt)
    assert len(rebuilt['candidates'])<=24 and rebuilt['physical_certificates_created']==0
    assert all(len(p['actions'])==30 for p in rebuilt['candidates'])


def check_routes(build, inputs, arm):
    template=build('shared');env=full_env(inputs)
    runner.save(template.parent/'condition.json',env.condition)
    service=SharedEvidenceService(template.parent,template.cell,env,template.scope,mode='shared',physical_budget=512,logical_budget=256)
    spec={**next(s for s in runner.schedule_for(contexts(),models()) if s['arm']==arm),'cell':'task'}
    for phase in audit.shared.VIEWS:runner.save(service.cell/phase/'spec.json',runner.phase_spec(spec,phase))
    policy=FakePolicy(env) if spec['checkpoint'] else None
    config=BranchConfig(max_branch_expansions=20,max_checked_candidates=20,max_generated_candidates=60)
    with service:
        phases,cps=runner.run_routes(service,env,config,service.cell,spec,time.perf_counter(),time.time()+120,policy,
            expected_generation=generate_staged_pickup_candidates(env))
    phases={name:audit.read(service.cell/name/'phase_result.json') for name in audit.shared.VIEWS}
    mech=mechanics(service.env)
    data=material(service,mech)
    for point in data['points'].values():point['elapsed_seconds']=0.
    evidence,_=audit.shared.check_service_ledger(**data)
    checks={}
    for name in audit.shared.VIEWS:
        phase=phases[name]
        args=(mech,asdict(config),phase['spec'],phase,lines(service.cell/name/'plans.jsonl.gz'),
            lines(service.cell/name/'decisions.jsonl.gz'),evidence[name],audit.sha(service.parent/'condition.json'))
        checks[name]=(audit.check_staged_records(*args,IndependentFirstRanker() if policy else None)
            if name=='EARLY' else audit.shared.check_phase_records(mech,args[1],name,*args[2:]))
    audit.shared.check_checkpoints(cps,checks,phases)
    return service,spec,phases,cps,evidence,checks


@pytest.mark.parametrize('arm',['staged_window','staged_bc'])
def test_actual_runner_new_control_shared_ledgers_and_guard_replay(build,inputs,arm):
    *_,checks=check_routes(build,inputs,arm)
    assert checks['EARLY']['graph_expansions']==0 and checks['EARLY']['stage_generation_verified']
    assert checks['GUARD']['fully_certified']
    assert checks['EARLY']['policy_decisions']==(checks['EARLY']['verify_attempts'] if arm=='staged_bc' else 0)


def test_strict_common_window_not_prefix_wait_and_all_pending_fallback():
    nodes=[{'id':i,'info':{'energy_kwh':energy},'held_out_loads':held}
        for i,energy,held in [(0,12,['load']), (1,11,[]),(2,10,['load'])]]
    assert [n['id'] for n in audit.exposed_nodes(nodes,0,10)]==[0,1,2]
    assert [n['id'] for n in audit.exposed_nodes(nodes,1,10)]==[0]
    assert audit.exposed_nodes(nodes,1,12)==nodes
    assert audit.exposed_nodes(nodes,1,None)==nodes


def test_fixed_contract_matches_runner_all32_and_complete_model_population():
    from test_run_clpu_deferred_pilot import contexts,models
    assert audit.protocol_settings()==runner.settings()
    specs=audit.expected_schedule(contexts(),models())
    assert specs==runner.schedule_for(contexts(),models()) and len(specs)==32
    for spec in specs:
        for phase in audit.shared.VIEWS:assert audit.expected_phase_spec(spec,phase)==runner.phase_spec(spec,phase)


@pytest.mark.parametrize('difference,authorized',[(0,False),(-1,False),(1e-12,True)])
def test_gate_requires_complete_actual_early_and_own_paid_guard_without_epsilon(difference,authorized):
    row={'cell':'task','status':'PASS','guard_established':True,'early_fully_certified':True,
        'early_energy_kwh':10.+difference,'guard_energy_kwh':10.}
    assert audit.phase_gate([row])['phase2_authorized'] is authorized
    row['early_fully_certified']=False
    with pytest.raises(AssertionError):audit.phase_gate([row])
    row['early_energy_kwh']=None
    assert not audit.phase_gate([row])['phase2_authorized']
    row['guard_established']=False
    with pytest.raises(AssertionError):audit.phase_gate([row])


def test_independent_auditor_does_not_import_production_generator_controller_or_runner():
    tree=ast.parse(Path(audit.__file__).read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        names=[n.name for n in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
        assert not any(n.startswith('dnr.') or n.startswith('run_clpu_staged') for n in names)


def test_actual_worker_header_pool_stage_identity_and_full_cell_audit(build,inputs,monkeypatch):
    import run_nature34_clpu_pilot as physical_runner
    from dnr import nature34_clpu_shared_evidence as service_module
    template=build('shared');env=full_env(inputs);run,cell=template.parent,template.cell
    runner.save(run/'condition.json',env.condition)
    service=SharedEvidenceService(run,cell,env,template.scope,mode='shared',physical_budget=512,logical_budget=256)
    spec={**next(s for s in runner.schedule_for(contexts(),models()) if s['arm']=='staged_window'),
        'cell':'task','scenario_id':env.scenario['scenario_id'],'event_id':env.event['event_id']}
    cfg={'horizon':30,'slot_seconds':60,'sample_seconds':30,'branch_search':asdict(BranchConfig(
        max_branch_expansions=20,max_checked_candidates=20,max_generated_candidates=60))}
    for name,value in {'spec.json':spec}.items():runner.save(cell/name,value)
    values={'schedule.json':[spec],'config.json':cfg,'physical_scope.json':service.scope,
        'condition.json':env.condition,'author_static_metadata.json':env.static,'load_reference.json':env.loads,
        'selected_scenarios.json':[env.scenario],'selected_events.json':[env.event],'command_inventory.json':{},
        'stage_plan.json':{'contexts':[{'context_id':spec['context_id'],'generation':generate_staged_pickup_candidates(env),
            'generation_sha256':audit.digest(generate_staged_pickup_candidates(env))}]},
        'controller_stage1_started.json':{'pid':os.getpid(),'stage':'phase1'}}
    for name,value in values.items():runner.save(run/name,value)
    for phase in audit.shared.VIEWS:runner.save(cell/phase/'spec.json',runner.phase_spec(spec,phase))
    epoch=time.time();deadline=epoch+900
    runner.save(run/'execution_started.json',{'pid':os.getpid(),'started_epoch_seconds':epoch,'deadline_epoch_seconds':epoch+3600})
    runner.save(cell/'execution.json',{'controller_pid':os.getpid(),'started_epoch_seconds':epoch,
        'deadline_epoch_seconds':deadline,'spec_sha256':audit.sha(cell/'spec.json'),'stage':'phase1'})
    monkeypatch.setattr(runner,'_WORKER_ENTERED',False);monkeypatch.setattr(runner,'verify_execution',lambda _:None)
    monkeypatch.setattr(runner.shutil,'disk_usage',lambda _:SimpleNamespace(free=10**15))
    monkeypatch.setattr(physical_runner,'make_env',lambda *_:env)
    monkeypatch.setattr(service_module,'SharedEvidenceService',lambda *a,**k:service)
    assert runner.worker(run,cell)==0
    (cell/'physical/command_ledger.jsonl').write_text('',encoding='utf-8')
    runner.seal(cell)
    history='historical/guard.json';runner.save(run/history,audit.read(cell/'GUARD/phase_result.json'))
    runner.save(run/'historical_bindings.json',[{'context_id':spec['context_id'],'phases':{'GUARD':{'result_file':history}}}])
    mech=mechanics(env);data=material(service,mech)
    for point in data['points'].values():point['elapsed_seconds']=0.
    statuses=Counter(p['status'] for p in data['points'].values())
    # Only the synthetic native solver/output wrappers are mocked here.
    # The full independent logical/alias/search/model/pool/cost audit executes.
    monkeypatch.setattr(audit.shared,'check_actual_points',lambda *a:(data['points'],statuses))
    monkeypatch.setattr(audit.old,'check_commands',lambda *a:{'numerical_entries':3+4*len(data['points'])})
    monkeypatch.setattr(audit.shared,'check_actual_requests',lambda *a:None)
    checked=audit.check_cell(run,spec,cfg,service.scope)
    assert checked['guard_established'] and checked['candidate_count']<=24
    assert checked['numerical_entries']==3+4*checked['physical_points']
    assert checked['early_energy_kwh'] is not None and checked['guard_energy_kwh'] is not None


def cohort_fixture():
    specs=[s for s in audit.expected_schedule(contexts(),models()) if s['context_id'] in (2,7)]
    checks=[];dispatch=[];exits=[]
    for spec in specs:
        begin=1000+spec['pair_order']*20
        checks.append({**spec,'status':'PASS','guard_established':True,'history_verified':True,'guard_quality_pct':80.,
            'candidate_pool_sha256':str(spec['context_id']),'first_verify_state_sha256':str(spec['context_id']),
            'phase_summary':{'GUARD':{'semantic_signature_sha256':str(spec['context_id'])}},
            'logical_points':10,'physical_points':2,'numerical_entries':11,'requests':12,
            'worker_finished_elapsed_seconds':8.,'worker_parent_pid':99,
            'execution':{'stage':'phase1','started_epoch_seconds':begin,'controller_pid':99}})
        row={'cell':spec['cell'],'stage':'phase1','pair_order':spec['pair_order'],
            'launcher_pid':500+spec['episode_id'],'started_epoch_seconds':begin}
        dispatch.append(row);exits.append({**row,'exit_code':0,'drained_after_controller_error':False,
            'finished_epoch_seconds':begin+10,'process_wall_seconds':10.})
    controller={'stage':'phase1','status':'COMPLETE_STAGE1_PENDING_AUDIT','scheduled':8,'scheduled_total':32,
        'selected_cells':[s['cell'] for s in specs],'dispatched':specs,'completed':[s['cell'] for s in specs],
        'exits':exits,'errors':[],'unknown_cost_cells':[],'peak_active':2,'wall_seconds':500.,
        **{k:sum(c[k] for c in checks) for k in ('logical_points','physical_points','numerical_entries','requests')}}
    return controller,specs,checks,dispatch


@pytest.mark.parametrize('mutation',[None,'missing_seed','unpaid_initialization','failed_exit','different_pool','different_guard','wrong_stage'])
def test_phase1_population_costs_real_exit_and_common_pool_cannot_be_relabeled(mutation):
    controller,specs,checks,dispatch=cohort_fixture()
    if mutation=='missing_seed':checks.pop()
    elif mutation=='unpaid_initialization':controller['numerical_entries']-=3
    elif mutation=='failed_exit':controller['exits'][0]['exit_code']=19
    elif mutation=='different_pool':checks[0]['candidate_pool_sha256']='different'
    elif mutation=='different_guard':checks[0]['phase_summary']['GUARD']['semantic_signature_sha256']='different'
    elif mutation=='wrong_stage':checks[0]['execution']['stage']='phase2'
    def verify():
        audit.check_stage_controller(controller,specs,checks,dispatch,'phase1')
        return audit.context_equalities(checks)
    if mutation is None:assert len(verify())==2
    else:
        with pytest.raises(AssertionError):verify()
