from copy import deepcopy
import gzip
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_clpu_shared_pilot as runner
from dnr.nature34_clpu_branch import BranchConfig
from test_nature34_clpu_shared_evidence import build


def contexts():
    return [{'context_id':i,'scenario_id':f'N2__line.l{i}__line.l30','event_id':[30,20,6,3][i%4]} for i in range(8)]


def rows(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',encoding='utf-8') as stream: return [json.loads(s) for s in stream if s.strip()]


def route_fixture(build,mode='shared'):
    service=build(mode)
    spec=runner.schedule_for(contexts())[0]
    spec={**spec,'mode':mode,'cell':'task'}
    cfg=BranchConfig(point_budget=256,max_branch_expansions=8,max_generated_candidates=100,max_checked_candidates=20,action_window=8)
    started=time.perf_counter()
    with service:
        result,cps=runner.run_routes(service,service.env,cfg,service.cell,spec,started,time.time()+60)
    return service,spec,result,cps


def test_fixed_population_pairs_and_counterbalanced_order():
    schedule=runner.schedule_for(contexts())
    assert len(schedule)==16 and len({r['cell'] for r in schedule})==16
    assert schedule==runner.schedule_for(list(reversed(contexts())))
    assert sum(schedule[i]['mode']=='private' for i in range(0,16,2))==4
    for i in range(0,16,2):
        pair=schedule[i:i+2]
        assert {r['mode'] for r in pair}=={'private','shared'}
        assert len({r['context_id'] for r in pair})==1
        assert all(r['method']=='beam' and r['checkpoint'] is None for r in pair)


def test_missing_or_repeated_context_rejected():
    with pytest.raises(ValueError): runner.schedule_for(contexts()[:-1])
    c=contexts(); c[-1]=deepcopy(c[0])
    with pytest.raises(ValueError): runner.schedule_for(c)


@pytest.mark.parametrize('mode',['private','shared'])
def test_real_search_routes_record_one_global_initialization_and_local_costs(build,mode):
    service,spec,result,cps=route_fixture(build,mode)
    assert [r['phase'] for r in cps if r['kind']=='INITIAL']==['EARLY','GUARD']
    assert all(r['numerical_entries']==3+4*r['physical_points'] for r in cps)
    assert all(r['phase_numerical_entries']==4*r['phase_physical_points'] for r in cps)
    assert all(not r['guard_established'] for r in cps[:-1])
    assert cps[-1]['guard_established']
    assert cps[-1]['physical_points']==service.physical_points
    assert sum(r['actual_physical_points'] for r in result.values())==service.physical_points
    assert sum(r['logical_points'] for r in result.values())==cps[-1]['logical_points']
    for phase,record in result.items():
        decisions=rows(service.cell/phase/'decisions.jsonl.gz')
        assert len(decisions)==2*record['search_decisions']
        for i in range(0,len(decisions),2):
            a,b=decisions[i:i+2]
            assert a['status']=='STARTED' and b['status']=='COMPLETED'
            assert a['action']==b['action'] and a['transition_id']==b['transition_id']==i//2
            assert b['response']['physical_points_after']==b['oracle_after']['logical_points']
            assert b['oracle_after']['numerical_entries']==4*b['oracle_after']['physical_points']
        assert record['policy_decisions']==0
    if mode=='shared': assert result['GUARD']['actual_physical_points']==0


def test_same_two_logical_searches_and_guards_across_modes(build):
    private=route_fixture(build,'private'); shared=route_fixture(build,'shared')
    for phase in ('EARLY','GUARD'):
        for key in ('best_plan','stop_reason','trace','logical_points','quality_pct','search_decisions'):
            assert private[2][phase][key]==shared[2][phase][key]
    assert private[0].physical_points>shared[0].physical_points


def test_deadline_before_phase_does_not_start_solver_requests(build):
    service=build()
    with pytest.raises(ValueError,match='deadline'):
        with service:
            runner.run_routes(service,service.env,BranchConfig(),service.cell,runner.schedule_for(contexts())[0],time.perf_counter(),0)
    assert service.physical_points==0 and not service.views


def test_incomplete_native_costs_remain_unknown_with_known_entries(tmp_path):
    spec={'cell':'episodes/000'}; cell=tmp_path/spec['cell']
    runner.save(cell/'physical/command_stats.json',{'returned':{'numerical_solution_entries':5}})
    runner.save(cell/'EARLY/view_summary.json',{'logical_points':2})
    value=runner.account(tmp_path,[spec])
    assert value=={'physical_points':0,'logical_points':2,'numerical_entries':5,'unknown_cost_cells':[spec['cell']]}


def controller_fixture(tmp_path,monkeypatch,*,failure=False,progress_error=False):
    run=tmp_path/'run'; run.mkdir()
    schedule=runner.schedule_for(contexts())
    runner.save(run/'schedule.json',schedule)
    runner.save(run/'resource_budget.json',{'historical_max_bytes_per_actual_point':1})
    for spec in schedule: runner.save(run/spec['cell']/'spec.json',spec)
    monkeypatch.setattr(runner,'verify_execution',lambda _:None)
    monkeypatch.setattr(runner,'ACTIVE',tmp_path/'active.json')
    monkeypatch.setattr(runner.shutil,'disk_usage',lambda _:SimpleNamespace(free=10**15))
    monkeypatch.setattr(runner.time,'sleep',lambda _:None)
    launched=[]
    class Process:
        def __init__(self,cmd,**kw):
            assert all(p.finished for p in launched[:-1]) if len(launched)%2==0 else True
            self.cell=Path(cmd[-1]); self.pid=100+len(launched); self.calls=0; self.finished=False
            self.code=19 if failure and not launched else 0
            launched.append(self)
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
        def bad_progress(path,value):
            if path==runner.ACTIVE and value.get('status')=='RUNNING':raise PermissionError('simulated active pointer')
            original(path,value)
        monkeypatch.setattr(runner,'save',bad_progress)
    return run,launched


def test_controller_counts_real_exits_and_drains_pairs(tmp_path,monkeypatch):
    run,launched=controller_fixture(tmp_path,monkeypatch)
    assert runner.execute(run)==0
    result=runner.read(run/'controller_completion.json')
    assert len(launched)==len(result['completed'])==len(result['exits'])==16
    assert result['physical_points']==16 and result['numerical_entries']==112
    assert all(p.finished for p in launched)


def test_nonzero_exit_after_result_seal_stops_further_pairs(tmp_path,monkeypatch):
    run,launched=controller_fixture(tmp_path,monkeypatch,failure=True)
    assert runner.execute(run)==1
    result=runner.read(run/'controller_completion.json')
    assert len(launched)==2 and all(p.finished for p in launched)
    assert len(result['completed'])==1 and len(result['exits'])==2
    assert result['physical_points']==2 and result['errors']


def test_progress_write_error_still_drains_active_writers(tmp_path,monkeypatch):
    run,launched=controller_fixture(tmp_path,monkeypatch,progress_error=True)
    assert runner.execute(run)==1
    result=runner.read(run/'controller_completion.json')
    assert len(launched)==2 and all(p.finished for p in launched)
    assert all(r['drained_after_controller_error'] for r in result['exits'])
    assert result['physical_points']==2 and result['errors']
