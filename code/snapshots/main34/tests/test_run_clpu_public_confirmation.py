"""Identity, source freeze, scope and no-response entry gates; zero real PF."""
from copy import deepcopy
import importlib
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
driver=importlib.import_module('run_clpu_public_confirmation')


def test_fixed_cartesian_schedule_and_balanced_cold_pairs():
    contexts=driver.contexts_for();schedule=driver.schedule_for(contexts)
    assert len(contexts)==24 and len(schedule)==48
    assert {(c['scenario_id'],c['event_id']) for c in contexts} == {
        (f,e) for f in driver.FAULT_IDS for e in driver.EVENT_IDS}
    for i in range(24):
        a,b=schedule[2*i:2*i+2]
        assert a['context_id']==b['context_id'] and {a['arm'],b['arm']}=={'P','E'}
        assert a['arm']==('P' if i%2==0 else 'E')
        assert a['pair_order']==b['pair_order']==i
        assert all(x['fault_partition']=='formal_test' and x['event_partition']=='holdout' for x in (a,b))
    assert len({s['cell'] for s in schedule})==48
    assert all(s['checkpoint'] is None and s['seed']==0 for s in schedule)


@pytest.mark.parametrize('mutation',[lambda c:c.pop(),lambda c:c.reverse(),
    lambda c:c[0].update(event_id=20),lambda c:c[0].update(fault_partition='build')])
def test_schedule_rejects_subset_reordering_replacement_and_training_scope(mutation):
    contexts=driver.contexts_for();mutation(contexts)
    with pytest.raises((ValueError,RuntimeError)):driver.schedule_for(contexts)


def test_selected_event_projection_uses_exact_original_rows_without_refitting():
    rows=[{'event_id':i,'a':i/100,'tau_seconds':i*10,'validation':{'passed':True},
           'original_nested':{'sentinel':[i]}} for i in range(1,32)]
    before=deepcopy(rows);selected=driver.selected_event_projection(rows)
    assert [r['event_id'] for r in selected]==list(driver.EVENT_IDS)
    assert selected==[before[i-1] for i in driver.EVENT_IDS]
    selected[0]['original_nested']['sentinel'].append('changed')
    assert rows==before


def test_bad_curve_import_identity_or_validation_stops():
    rows=[{'event_id':i,'validation':{'passed':True}} for i in range(1,32)]
    with pytest.raises((ValueError,RuntimeError)):driver.selected_event_projection(rows[:-1])
    rows[driver.EVENT_IDS[0]-1]['validation']['passed']=False
    with pytest.raises((ValueError,RuntimeError)):driver.selected_event_projection(rows)


def test_hydration_verifies_design_before_any_event_bytes(monkeypatch):
    calls=[]
    def reject(run):calls.append('verify');raise RuntimeError('design source changed')
    monkeypatch.setattr(driver,'verify_design',reject)
    monkeypatch.setattr(driver,'read',lambda p:pytest.fail('read before design gate'))
    monkeypatch.setattr(driver,'sha',lambda p:pytest.fail('hash before design gate'))
    with pytest.raises(RuntimeError,match='design source changed'):driver.hydrate(ROOT/'runs/synthetic_unopened')
    assert calls==['verify']


def test_execute_requires_independent_precheck_before_clock_or_dispatch(monkeypatch):
    def reject(run,**kwargs):raise RuntimeError('independent precheck missing')
    monkeypatch.setattr(driver,'verify_execution',reject)
    monkeypatch.setattr(driver,'save',lambda *a:pytest.fail('dispatch artifact before independent gate'))
    with pytest.raises(RuntimeError,match='independent precheck missing'):driver.execute(ROOT/'runs/synthetic_unopened')


def test_old_training_environment_entrypoint_is_not_reused():
    import inspect
    source=inspect.getsource(driver.make_env)
    assert "scenario['partition']=='formal_test'" in source
    assert 'run_nature34_clpu_pilot' not in source
    assert driver.run_routes is driver.public.run_routes
    assert 'expected_generation=None' in inspect.getsource(driver.worker)


def test_budget_and_study_are_exact():
    settings=driver.settings();forecast=driver.capacity()
    assert settings['maximum_numerical_entries']==48*(3+4*512)==98448
    assert settings['total_physical_budget']==48*512==24576
    assert settings['max_workers']==2 and settings['controller_seconds']==3600 and settings['worker_seconds']==900
    assert forecast['required_free_bytes']>driver.MIN_FREE+driver.RESERVE
    assert settings['historical_answers_loaded'] is False and settings['new_training']==0


def test_scope_is_explicit_formal_and_binds_confirmation_not_training(monkeypatch):
    monkeypatch.setattr(driver,'sha',lambda p:'0'*64)
    monkeypatch.setattr(driver,'read',lambda p:{'sentinel':p.name})
    scope=driver.physical_scope(ROOT/'runs/synthetic_unopened',['scripts/example.py'])
    assert scope['fault_partition']=='formal_test' and scope['event_partition']=='holdout'
    assert 'confirmation_contexts.json' in scope['input_files']
    assert 'training_contexts.json' not in scope['input_files']
    assert 'training_scenarios.json' not in scope['input_files']
    assert scope['load_setpoint_precision']=='.15g' and scope['alpha']==.5
