"""Pure schedule and stop-rule checks; no physical runtime is constructed."""
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_nature13_clpu_acceptance as runner


def test_fixed_schedule_preserves_requests_and_reference_ids():
    rows=runner.make_schedule()
    assert [r['query_id'] for r in rows]==list(range(16))
    assert [r['source_point_id'] for r in rows[:4]]==[12,14,48,50]
    assert [r['group'] for r in rows]==['reference']*4+['initial']*6+['peak']*6
    assert all(r['request']['action'][4:]==[1]*7 for r in rows[4:])
    assert all(r['request']['action'][2]==0 for r in rows)
    assert all(len(r['request']['multipliers'])==15 for r in rows)
    assert all(set(r['request']['multipliers'].values())=={1.0} for r in rows[:10])
    for row in rows[10:]:
        assert row['request']['action'][3]==1
        assert max(row['request']['multipliers'].values())>1
        assert row['request']['time_minutes']==0


@pytest.mark.parametrize('group,expected',[
    ('reference','STOP_REFERENCE_CONSTRAINT_FAIL'),
    ('initial','STOP_INITIAL_NO_WAIT_BASELINE'),('peak',None)])
def test_reliable_constraint_failure_has_group_specific_stop(group,expected):
    assert runner.point_gate({'group':group},{'status':'FAIL','reliable':True})==expected


@pytest.mark.parametrize('group',['reference','initial','peak'])
def test_unreliable_never_treated_as_constraint_fail(group):
    assert runner.point_gate({'group':group},{'status':'FAIL','reliable':False})=='STOP_UNRELIABLE_POINT'
    assert runner.point_gate({'group':group},{'status':'UNKNOWN','reliable':True})=='STOP_UNRELIABLE_POINT'


def test_physical_identity_retains_case_alpha_and_complete_multipliers():
    spec=runner.make_schedule()[4]['request']
    base=runner.canonical(runner.physical_input(spec))
    temporal={**spec,'time_minutes':30,'state_identity':{'different_time':True}}
    assert runner.canonical(runner.physical_input(temporal))==base
    changed={**spec,'multipliers':{**spec['multipliers'],next(iter(spec['multipliers'])):1.01}}
    assert runner.canonical(runner.physical_input(changed))!=base
    assert runner.canonical(runner.physical_input({**spec,'alpha':.5}))!=base


def test_no_solver_imported_by_schedule():
    runner.make_schedule()
    assert 'opendssdirect' not in sys.modules
    assert 'dss' not in sys.modules


def test_long_source_path_has_short_unique_traceable_archive_name():
    source=ROOT/('long_source_segment_'*14)/'IEEE13Nodeckt.dss'
    first=runner.input_archive_name(0,source)
    second=runner.input_archive_name(1,source)
    assert first=='inputs/000_IEEE13Nodeckt.dss'
    assert first!=second
    assert len(str(ROOT/'runs/20261101T051928_842917Z_nature13_clpu_acceptance'/first))<240


def test_worker_without_controller_marker_stops_before_runtime(tmp_path):
    assert runner.worker(tmp_path,runner.time.monotonic()+240)==2
    saved=runner.read(tmp_path/'worker_summary.json')
    assert saved['budget'] is None
    assert saved['attempted_query_ids']==[]
    assert saved['not_run_query_ids']==list(range(16))
    assert not (tmp_path/'work').exists()


def test_controller_timeout_kills_once_and_prevents_second_execution(tmp_path,monkeypatch):
    runner.save(tmp_path/'prepared.json',{'for':'pure controller test'})
    runner.save(tmp_path/'pre_execution_review.json',{'status':'PASS','prepared_sha256':runner.sha(tmp_path/'prepared.json')})
    monkeypatch.setattr(runner,'verify',lambda run:None)
    monkeypatch.setattr(runner,'ACTIVE',tmp_path/'active.json')
    calls=[]
    class FakeProcess:
        def wait(self,timeout=None):
            calls.append(('wait',timeout))
            if timeout is not None:
                raise runner.subprocess.TimeoutExpired('fake',timeout)
            return -9
        def kill(self):
            calls.append(('kill',None))
    def spawn(*args,**kwargs):
        calls.append(('spawn',None));return FakeProcess()
    monkeypatch.setattr(runner.subprocess,'Popen',spawn)
    assert runner.execute(tmp_path)==-9
    assert runner.read(tmp_path/'controller.json')['timed_out'] is True
    assert [x[0] for x in calls].count('spawn')==1
    assert [x[0] for x in calls].count('kill')==1
    with pytest.raises(FileExistsError):
        runner.execute(tmp_path)
    assert [x[0] for x in calls].count('spawn')==1
