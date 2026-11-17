"""Controller boundaries: no solver or previous physical outputs are used."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts'),str(Path(__file__).resolve().parents[1]/'src')]
import run_clpu_lazysp_pilot as m

def dump(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value),encoding='utf-8')

def setup_run(tmp_path,monkeypatch):
    run=tmp_path/'run';run.mkdir()
    schedule=[]
    for i in range(12):
        spec={'cell':f'episodes/{i:03d}','arm':m.ARMS[i%3],'context_id':m.CONTEXTS[i//3]}
        schedule.append(spec);(run/spec['cell']).mkdir(parents=True)
    dump(run/'schedule.json',schedule);dump(run/'prepared.json',{})
    dump(run/'preflight.json',{'status':'PASS','prepared_sha256':m.sha(run/'prepared.json')})
    monkeypatch.setattr(m,'verify',lambda run:None)
    monkeypatch.setattr(m,'ACTIVE',tmp_path/'active.json')
    monkeypatch.setattr(m.shutil,'disk_usage',lambda _:SimpleNamespace(free=50*2**30))
    return run

def test_controller_stops_after_first_bad_exit_and_preserves_unrun(tmp_path,monkeypatch):
    run=setup_run(tmp_path,monkeypatch);calls=[]
    class Process:
        pid=123
        def __init__(self,*a,**k):calls.append(a)
        def wait(self,timeout=None):return 7
    monkeypatch.setattr(m.subprocess,'Popen',Process)
    assert m.execute(run)==1
    result=m.read(run/'controller_completion.json')
    assert len(calls)==1 and len(result['not_run'])==11
    assert result['exits'][0]['exit_code']==7
    with pytest.raises(Exception,match='No automatic retry'):m.execute(run)

def test_timeout_kills_and_drains_before_stopping(tmp_path,monkeypatch):
    run=setup_run(tmp_path,monkeypatch);actions=[]
    class Process:
        pid=123
        def __init__(self,*a,**k):pass
        def wait(self,timeout=None):
            actions.append(('wait',timeout))
            if timeout is not None:raise m.subprocess.TimeoutExpired('synthetic',timeout)
            return -9
        def kill(self):actions.append(('kill',None))
    monkeypatch.setattr(m.subprocess,'Popen',Process)
    assert m.execute(run)==1
    result=m.read(run/'controller_completion.json')
    assert [a[0] for a in actions]==['wait','kill','wait']
    assert result['exits'][0]['timed_out'] and len(result['not_run'])==11

def test_no_execution_without_passing_bound_preflight(tmp_path,monkeypatch):
    run=setup_run(tmp_path,monkeypatch)
    dump(run/'preflight.json',{'status':'FAIL','prepared_sha256':m.sha(run/'prepared.json')})
    with pytest.raises(Exception,match='preflight'):m.execute(run)
    assert not (run/'execution_started.json').exists()
