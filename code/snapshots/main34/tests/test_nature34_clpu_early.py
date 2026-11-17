from copy import deepcopy
from types import SimpleNamespace

import pytest

from dnr.nature34_clpu_early import choose_action,complete_plan,merge_results,selected_method
from dnr.nature34_clpu_environment import CLPUReliabilityError


def result(energy=10.):
    return {'status':'CERTIFIED_INCUMBENT','best_plan':{'status':'COMPLETE','certified_complete':True,
        'actions':[0,1], 'validation':{'status':'COMPLETE','energy_kwh':energy}}}


@pytest.mark.parametrize('turn,expected',[(0,'ppo_raw'),(15,'ppo_raw'),(16,'beam'),(255,'beam')])
def test_fixed_early_boundary_does_not_extend_for_singletons(turn,expected):
    assert selected_method('EARLY','ppo_raw',turn)==expected
    assert selected_method('GUARD','ppo_raw',turn)=='beam'


def test_actual_model_mask_and_deterministic_action_are_preserved():
    calls=[]
    class Policy:
        def act(self,obs,mask,**kwargs):
            calls.append((deepcopy(obs),list(mask),kwargs))
            return {'action':1,'logprob':-.5,'value':.3}
    obs={'mask':[True,True]}
    search=SimpleNamespace(_beam_turn=15,rule_action=lambda method:0)
    assert choose_action(search,obs,'EARLY','bc_raw',Policy())['action']==1
    assert calls[0][2]['deterministic'] is True
    search._beam_turn=16
    assert choose_action(search,obs,'EARLY','bc_raw',None)=={'action':0,'logprob':None,'value':None,'method':'beam','policy_used':False}


@pytest.mark.parametrize('choice',[{'action':2,'logprob':0.,'value':0.},{'action':0,'logprob':float('nan'),'value':0.}])
def test_invalid_model_output_cannot_reach_search(choice):
    policy=SimpleNamespace(act=lambda *args,**kwargs:choice)
    with pytest.raises(CLPUReliabilityError):
        choose_action(SimpleNamespace(_beam_turn=0),{'mask':[True]},'EARLY','ppo_raw',policy)


@pytest.mark.parametrize('early_energy,selected',[(9.,'GUARD'),(10.,'GUARD'),(10.+1e-12,'EARLY')])
def test_strict_certified_selection_uses_guard_on_exact_energy_ties(early_energy,selected):
    early,guard=result(early_energy),result()
    before=deepcopy((early,guard))
    actual=merge_results(early,guard,protected=True)
    assert actual['selected_phase']==selected
    assert actual['guard_audit']['nondegradation_verified'] is True
    assert actual['best_plan']['validation']['energy_kwh']>=10.
    assert (early,guard)==before


def test_incomplete_early_plan_cannot_replace_full_guard():
    early={'status':'NO_CERTIFIED_COMPLETE_PLAN','best_plan':None}
    assert merge_results(early,result(),protected=True)['selected_phase']=='GUARD'


def test_absent_guard_never_becomes_a_protected_success():
    guard={'status':'INITIAL_FAIL','best_plan':None}
    actual=merge_results(result(),guard,protected=True)
    assert actual['status']=='INITIAL_FAIL' and actual['best_plan'] is None
    assert actual['guard_audit']['guard_established'] is False
    with pytest.raises(ValueError):merge_results(result(),None,protected=True)


def test_reference_has_no_hidden_guard_or_earlier_floor():
    actual=merge_results(result(),protected=False)
    assert actual['guard_audit'] is None and actual['selected_phase']=='EARLY'
    assert actual['intermediate_beam_floor_claimed'] is False
    with pytest.raises(ValueError):merge_results(result(),result(),protected=False)


@pytest.mark.parametrize('mutation',['partial','nan','status'])
def test_complete_evidence_is_required_before_quality_comparison(mutation):
    value=result()
    if mutation=='partial':value['best_plan']['certified_complete']=False
    elif mutation=='nan':value['best_plan']['validation']['energy_kwh']=float('nan')
    else:value['status']='NO_CERTIFIED_COMPLETE_PLAN'
    with pytest.raises(CLPUReliabilityError):complete_plan(value)
