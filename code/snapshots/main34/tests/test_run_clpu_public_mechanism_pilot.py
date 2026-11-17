"""Focused rule-only P/E/A integration; no real physical calls."""
from collections import Counter
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_clpu_public_mechanism_pilot as runner
from dnr.nature34_clpu_branch import BranchConfig
from dnr.nature34_clpu_staged_pickup import generate_staged_pickup_candidates
from test_nature34_clpu_branch import environment
from test_nature34_clpu_shared_evidence import build
from test_run_clpu_shared_pilot import contexts


def test_population_rotation_caps_and_no_old_global_changes():
    schedule=runner.schedule_for(contexts())
    assert schedule==runner.schedule_for(list(reversed(contexts())))
    assert len(schedule)==len({s['cell'] for s in schedule})==24
    assert Counter(s['arm'] for s in schedule)=={'P':8,'E':8,'A_prune':8}
    assert all({s['arm'] for s in schedule if s['context_id']==i}==set(runner.SLOTS) for i in range(8))
    for turn in range(3):
        group=schedule[turn*8:(turn+1)*8]
        assert {s['context_id'] for s in group}==set(range(8))
        assert all(s['arm']==runner.SLOTS[(s['context_order']+turn)%3] for s in group)
    assert all(s['pair_order']==i//2 and s['method']=='beam' and s['seed']==0 and s['checkpoint'] is None for i,s in enumerate(schedule))
    assert runner.settings()['total_physical_budget']==12288
    assert runner.settings()['maximum_numerical_entries']==49224
    assert runner.settings()['models_used']==0
    assert runner.previous.CELLS==runner.base.CELLS==32
    assert runner.previous.settings()['phase_order']==['GUARD','EARLY']
    assert runner.energy.settings()['phase_order']==['EARLY','GUARD']
    assert set(runner.source_names())==set(runner.previous.source_names())|set(runner.ADDITIONS)
    with pytest.raises(ValueError,match='model bindings'):runner.schedule_for(contexts(),[{'method':'bc_raw'}])
    with pytest.raises(ValueError,match='eight contexts'):runner.schedule_for(contexts()[:-1])


def worker_fixture(build,monkeypatch,arm='P',*,failed=False):
    template=build(failed=failed);run,cell=template.parent,template.cell
    env=environment(horizon=30)
    spec={**next(s for s in runner.schedule_for(contexts()) if s['arm']==arm),'cell':'task'}
    runner.save(run/'schedule.json',[spec]);runner.save(cell/'spec.json',spec)
    for phase in spec['phase_order']:
        runner.save(cell/phase/'spec.json',runner.phase_spec(spec,phase))
    runner.save(run/'config.json',{'branch_search':runner.base.asdict(BranchConfig())})
    runner.save(run/'physical_scope.json',template.scope)
    runner.save(run/'stage_plan.json',{'contexts':[{'context_id':spec['context_id'],'generation':generate_staged_pickup_candidates(env)}]})
    runner.save(run/'execution_started.json',{'deadline_epoch_seconds':time.time()+60})
    runner.save(cell/'execution.json',{'deadline_epoch_seconds':time.time()+50})
    monkeypatch.setattr(runner,'verify_execution',lambda _:None)
    monkeypatch.setattr(runner,'_WORKER_ENTERED',False)
    monkeypatch.setattr(runner.shutil,'disk_usage',lambda _:SimpleNamespace(free=10**15))
    import run_nature34_clpu_pilot as pilot
    monkeypatch.setattr(pilot,'make_env',lambda *a:env)
    return run,cell,spec


@pytest.mark.parametrize('arm',runner.SLOTS)
def test_real_worker_mock_oracle_has_actual_phase_order_floor_and_complete_cost(build,monkeypatch,arm):
    run,cell,spec=worker_fixture(build,monkeypatch,arm)
    assert runner.worker(run,cell)==0
    result=runner.read(cell/'result.json')
    assert result['schema']==runner.SCHEMA and result['study']==runner.STUDY
    assert result['route_arm']==arm and result['phase_order']==spec['phase_order']
    assert result['models_used']==result['policy_decisions']==result['model_load_seconds']==0
    assert len(build.delegates)==1 and build.delegates[0].closed
    assert result['numerical_entries']==3+4*result['actual_physical_points']
    assert result['guard_audit']['guard_established'] and result['guard_audit']['nondegradation_verified']
    assert abs(result['worker_elapsed_seconds']-result['preflight_elapsed_seconds']-result['computation_elapsed_seconds'])<1e-8
    assert result['actual_physical_points']==sum(p['actual_physical_points'] for p in result['phase_summary'].values())
    assert result['logical_points']==sum(p['logical_points'] for p in result['phase_summary'].values())
    cp=result['checkpoints'];boundary=next(i for i,c in enumerate(cp) if c['phase']=='GUARD' and c['kind']=='CLOSE')
    assert not any(c['guard_established'] for c in cp[:boundary])
    assert all(c['guard_established'] for c in cp[boundary:])
    assert list(dict.fromkeys(c['phase'] for c in cp))==spec['phase_order']
    early=runner.read(cell/'EARLY/phase_result.json');guard=runner.read(cell/'GUARD/phase_result.json')
    assert early['verify_attempts']<=16 and early['search_decisions']<=17
    assert guard['search_decisions']<=256 and guard['policy_decisions']==0
    if arm=='E':
        assert result['guard_first'] is None and not result['intermediate_beam_floor_claimed']
    else:
        assert early['guard_first']['external_guard_plan']==guard['best_plan']
        assert early['guard_first']['guard_reference']['sha256']==runner.sha(cell/'GUARD/phase_result.json')
        assert early['oracle']['cross_view_reuses']>=61
    if arm=='A_prune':
        assert early['guard_first']['free_pruned_complete_plans']==0
        assert early['guard_first']['free_objective_pruning_enabled'] is False
        assert early['control']['free_objective_pruning_enabled'] is False
        assert early['control']['ablation_version']=='clpu-public-no-free-pruning-v1'
    runner.cache.verify_manifest(cell,required=('result.json','worker_finished.json'))


def test_three_arms_rebuild_identical_candidates_and_unchanged_guard_trace(build,monkeypatch):
    outputs=[]
    for arm in runner.SLOTS:
        run,cell,_=worker_fixture(build,monkeypatch,arm)
        assert runner.worker(run,cell)==0
        outputs.append((runner.read(cell/'EARLY/candidate_pool.json'),runner.read(cell/'GUARD/phase_result.json')))
    assert outputs[0][0]==outputs[1][0]==outputs[2][0]
    for key in ('best_plan','attempts','trace','stop_reason','logical_points','search_decisions'):
        assert outputs[0][1][key]==outputs[1][1][key]==outputs[2][1][key]


def test_guard_failure_retains_unknown_cost_and_never_starts_early(build,monkeypatch):
    run,cell,_=worker_fixture(build,monkeypatch,'A_prune',failed=True)
    assert runner.worker(run,cell)==1
    assert build.delegates[0].closed
    assert (cell/'error.json').is_file() and not (cell/'result.json').exists()
    assert (cell/'GUARD/phase_result.json').is_file() and not (cell/'EARLY/phase_result.json').exists()
    assert runner.base.account(run,[{'cell':'task'}])['unknown_cost_cells']==['task']


@pytest.mark.parametrize('failure',[False,True])
def test_controller_pair_drain_no_retry_and_all_unstarted_slots_retained(tmp_path,monkeypatch,failure):
    run=tmp_path/'run';run.mkdir();schedule=runner.schedule_for(contexts())
    runner.save(run/'schedule.json',schedule)
    runner.save(run/'stage_plan.json',{'all_cells':[s['cell'] for s in schedule]})
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
            assert Path(cmd[2]).name=='run_clpu_public_mechanism_pilot.py'
            assert sum(not p.finished for p in launched)<2
            assert all(kwargs['env'][key]=='1' for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'))
            if len(launched)%2==0:assert all(p.finished for p in launched)
            self.cell=Path(cmd[-1]);self.pid=500+len(launched);self.calls=0;self.finished=False
            self.code=17 if failure and not launched else 0;launched.append(self)
        def poll(self):
            self.calls+=1
            if self.calls<2:return None
            if not self.finished:
                runner.save(self.cell/'result.json',{'logical_points':2,'actual_physical_points':1,'numerical_entries':7,'requests':3})
                runner.save(self.cell/'worker_finished.json',{'status':'COMPLETE'});runner.seal(self.cell);self.finished=True
            return self.code
    monkeypatch.setattr(runner.base.subprocess,'Popen',Process)
    assert runner.execute(run)==int(failure)
    result=runner.read(run/'controller_completion.json')
    assert len(launched)==(2 if failure else 24) and all(p.finished for p in launched)
    assert result['actual_tasks']==len(launched)
    assert result['resolved_slots']==(1 if failure else 24)
    assert len(result['not_run'])==(22 if failure else 0)
    assert result['all_eight_context_mean_allowed'] is (not failure)
    assert result['physical_points']==len(launched) and result['peak_active']==2
    with pytest.raises(ValueError,match='No automatic rerun'):runner.execute(run)


def test_prior_reference_binding_has_no_models_and_rejects_tampered_small_input(tmp_path):
    run=tmp_path/'run';run.mkdir()
    prior=runner.COMPARISON_SOURCE;cs=runner.read(prior/'contexts.json')
    runner.save(run/'contexts.json',cs)
    runner.save(run/'stage_plan.json',runner.partition_schedule(runner.schedule_for(cs),runner.read(prior/'stage_plan.json')['contexts']))
    runner.bind_comparison(run)
    assert not (run/'models').exists()
    value=runner.read(run/'comparison_source/audit/summary.json');value['audited_cells']=31
    runner.save(run/'comparison_source/audit/summary.json',value)
    with pytest.raises(ValueError,match='Changed comparison small input'):runner.verify_comparison(run)
