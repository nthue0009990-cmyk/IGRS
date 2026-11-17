"""Focused single-runtime GUARD-first driver and cold controller boundaries."""
from collections import Counter
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_clpu_guard_first_pilot as runner
from dnr.nature34_clpu_branch import BranchConfig
from dnr.nature34_clpu_guard_first_control import GuardFirstEvidenceService
from dnr.nature34_clpu_staged_pickup import generate_staged_pickup_candidates
from test_nature34_clpu_branch import environment
from test_nature34_clpu_shared_evidence import build
from test_run_clpu_deferred_pilot import FakePolicy, models, rows
from test_run_clpu_shared_pilot import contexts


def test_frozen32_population_order_budget_and_private_source_namespace():
    schedule = runner.schedule_for(contexts(), models())
    assert len(schedule) == 32
    assert Counter(s['arm'] for s in schedule) == {'guard_first_window': 8, 'staged_bc': 24}
    assert runner.settings()['phases'] == runner.settings()['phase_order'] == ['GUARD', 'EARLY']
    assert runner.settings()['maximum_numerical_entries'] == 65632
    assert runner.settings()['total_physical_budget'] == 16384
    assert runner.settings()['early_verification_slots'] == 16
    assert runner.base.settings()['phase_order'] == ['EARLY', 'GUARD']
    assert runner.previous.STUDY == 'CLPU_STAGED_ENERGY_COMMON_PILOT_V1'
    assert set(runner.source_names()) == set(runner.previous.source_names()) | set(runner.ADDITIONS)
    assert not any('report_clpu_guard_first' in n for n in runner.source_names())


def route_case(build, arm, *, failed=False):
    template = build(failed=failed); env = environment(horizon=30)
    service = GuardFirstEvidenceService(template.parent, template.cell, env, template.scope)
    spec = next(s for s in runner.schedule_for(contexts(), models()) if s['arm'] == arm)
    for phase in ('GUARD', 'EARLY'):
        runner.save(service.cell/phase/'spec.json', runner.phase_spec(spec, phase))
    policy = FakePolicy(env) if spec['checkpoint'] else None
    with service:
        phases, cps = runner.run_routes(service, env, BranchConfig(), service.cell, spec,
            time.perf_counter(), time.time()+60, policy, expected_generation=generate_staged_pickup_candidates(env))
    return service, policy, phases, cps


@pytest.mark.parametrize('arm', ['guard_first_window', 'staged_bc'])
def test_real_driver_with_mock_physics_has_sticky_floor_separate_owned_certificates(build, arm):
    service, policy, phases, cps = route_case(build, arm)
    assert list(service.views) == list(phases) == ['GUARD', 'EARLY']
    assert len(build.delegates) == 1 and service.closed
    early, guard = phases['EARLY'], phases['GUARD']
    assert early['guard_first']['external_guard_plan'] == guard['best_plan']
    assert early['guard_first']['guard_reference']['sha256'] == runner.sha(service.cell/'GUARD/phase_result.json')
    assert early['best_plan'] in early['attempts']
    assert early['logical_points'] >= 61 and early['oracle']['cross_view_reuses'] >= 61
    assert sum(p['actual_physical_points'] for p in phases.values()) == service.physical_points
    boundary = next(i for i,c in enumerate(cps) if c['phase']=='GUARD' and c['kind']=='CLOSE')
    assert not any(c['guard_established'] for c in cps[:boundary])
    assert all(c['guard_established'] for c in cps[boundary:])
    assert all(c['quality_pct'] >= guard['quality_pct'] for c in cps[boundary:])
    assert all(c['numerical_entries'] == 3+4*c['physical_points'] for c in cps)
    assert guard['policy_decisions'] == 0 and guard['verify_attempts'] is None
    assert early['verify_attempts'] <= 16 and early['search_decisions'] <= 17
    assert len(rows(service.cell/'EARLY/decisions.jsonl.gz')) == 2*early['search_decisions']
    assert not policy or policy.calls == early['policy_decisions']


def test_fixed_early_ranker_cannot_change_preceding_complete_guard(build):
    left = route_case(build, 'guard_first_window')[2]['GUARD']
    right = route_case(build, 'staged_bc')[2]['GUARD']
    for key in ('best_plan', 'attempts', 'trace', 'stop_reason', 'logical_points', 'search_decisions', 'actual_physical_points'):
        assert left[key] == right[key]


def test_missing_guard_certificate_prevents_early_and_retains_failure(build):
    with pytest.raises(ValueError, match='GUARD must complete'):
        route_case(build, 'guard_first_window', failed=True)
    service = build.delegates[0]
    assert service.closed
    # GUARD's outcome is written before the rejected EARLY construction.
    assert (service.cell.parent/'GUARD/phase_result.json').is_file()
    assert not (service.cell.parent/'EARLY/phase_result.json').exists()


def worker_fixture(build, monkeypatch, arm='guard_first_window'):
    template = build(); run, cell = template.parent, template.cell
    env = environment(horizon=30)
    spec = {**next(s for s in runner.schedule_for(contexts(), models()) if s['arm']==arm), 'cell':'task'}
    runner.save(run/'schedule.json', [spec]); runner.save(cell/'spec.json', spec)
    for phase in ('GUARD', 'EARLY'): runner.save(cell/phase/'spec.json', runner.phase_spec(spec, phase))
    runner.save(run/'config.json', {'branch_search': runner.base.asdict(BranchConfig())})
    runner.save(run/'physical_scope.json', template.scope)
    runner.save(run/'stage_plan.json', {'contexts':[{'context_id':spec['context_id'], 'generation':generate_staged_pickup_candidates(env)}]})
    runner.save(run/'execution_started.json', {'deadline_epoch_seconds':time.time()+60})
    runner.save(cell/'execution.json', {'deadline_epoch_seconds':time.time()+50})
    monkeypatch.setitem(runner._PRIVATE, 'verify_execution', lambda _: None)
    monkeypatch.setitem(runner._PRIVATE, '_WORKER_ENTERED', False)
    monkeypatch.setattr(runner.base.shutil, 'disk_usage', lambda _:SimpleNamespace(free=10**15))
    import run_nature34_clpu_pilot as pilot
    import nature34_clpu_tail_execution as tail
    monkeypatch.setattr(pilot, 'make_env', lambda *a:env)
    monkeypatch.setattr(tail, 'load_fixed_policy', lambda path,s:(FakePolicy(env),None) if path else (None,None))
    if spec['checkpoint']:
        model=run/spec['checkpoint'];model.parent.mkdir(parents=True,exist_ok=True);model.write_bytes(b'fixed mock model')
        spec['checkpoint_sha256']=runner.sha(model)
        runner.save(run/'schedule.json',[spec]);runner.save(cell/'spec.json',spec)
    return run, cell, spec


@pytest.mark.parametrize('arm', ['guard_first_window', 'staged_bc'])
def test_full_worker_uses_new_service_and_original_merge_complete_cost(build,monkeypatch,arm):
    run,cell,spec=worker_fixture(build,monkeypatch,arm)
    assert runner.worker(run,cell)==0
    result=runner.read(cell/'result.json')
    assert result['schema']==runner.SCHEMA and result['study']==runner.STUDY
    assert result['phase_order']==['GUARD','EARLY'] and result['intermediate_beam_floor_claimed']
    assert result['service']['phase_order']==['GUARD','EARLY']
    assert result['guard_audit']['guard_established'] and result['guard_audit']['nondegradation_verified']
    assert result['numerical_entries']==3+4*result['actual_physical_points']
    assert result['preflight_elapsed_seconds']>=result['model_load_seconds']
    assert abs(result['worker_elapsed_seconds']-result['preflight_elapsed_seconds']-result['computation_elapsed_seconds'])<1e-8
    assert (cell/'completion_manifest.json').is_file()


def test_matched_comparison_binding_rejects_changed_previous_energy_audit(tmp_path):
    run=tmp_path/'run';run.mkdir()
    previous=runner.COMPARISON_SOURCE
    cs=runner.read(previous/'contexts.json')
    runner.save(run/'contexts.json',cs)
    schedule=runner.schedule_for(cs,runner.read(runner.MODELS_SOURCE/'model_bindings.json'))
    runner.save(run/'stage_plan.json',runner.partition_schedule(schedule,runner.read(previous/'stage_plan.json')['contexts']))
    runner.bind_comparison(run)
    changed=runner.read(run/'comparison_source/audit/summary.json');changed['audited_cells']=31
    runner.save(run/'comparison_source/audit/summary.json',changed)
    with pytest.raises(ValueError,match='Changed comparison small input'): runner.verify_comparison(run)


@pytest.mark.parametrize('failure',[False,True])
def test_controller_at_most_two_workers_drains_pair_and_stops_after_error(tmp_path,monkeypatch,failure):
    run=tmp_path/'run';run.mkdir();schedule=runner.schedule_for(contexts(),models())
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
            assert Path(cmd[2]).name=='run_clpu_guard_first_pilot.py'
            assert sum(not p.finished for p in launched)<2
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
    assert len(launched)==(2 if failure else 32) and all(p.finished for p in launched)
    assert result['actual_tasks']==len(launched)
    assert result['resolved_slots']==(1 if failure else 32)
    assert result['all_eight_context_mean_allowed'] is (not failure)
    with pytest.raises(ValueError,match='No automatic rerun'):runner.execute(run)
