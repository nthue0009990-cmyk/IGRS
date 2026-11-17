"""Synthetic handoff semantics only; no physical engine invocation."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_nature123_igrs import env,Oracle
from dnr.nature123_igrs import BranchConfig,CLPUReliabilityError
from dnr.nature123_igrs_search import graph_plan
from dnr.nature123_igrs_handoff import carryover_candidates,staged_choice,run_search


def node(e,index,delay,origin='prefix_wait'):
    actions=[0]*delay+[4]+[0]*(29-delay)
    return {'id':index,'parent_id':index+100,'prefix':tuple(actions[:delay+1]),
        'origin':origin,'info':graph_plan(e,e.initial,actions)}


def guard_class(nodes):
    class FixtureGuard:
        def __init__(self,e,o,config,record_callback):
            self.done=True;self.stop_reason='BRANCH_EXPANSION_CAP';self._frontier=deepcopy(nodes)
            validation=e.validate_plan(e.initial,[0]*30,o)
            self.incumbent={'actions':[0]*30,'origin':'wait_incumbent','status':'COMPLETE',
                'certified_complete':True,'validation':validation}
            self._checked={tuple([0]*30):self.incumbent};record_callback(self.incumbent)
        def result(self):return {'done':True,'stop_reason':self.stop_reason,'best_plan':self.incumbent}
    return FixtureGuard


class HandoffTests(unittest.TestCase):
    def test_actual_frontier_filters_deduplicates_and_original_priority(self):
        e=env();nodes={i:node(e,i,i) for i in range(4)}
        nodes[8]=node(e,8,0);nodes[9]=node(e,9,0,'graph_completion')
        nodes[10]={'id':10,'parent_id':None,'prefix':(), 'origin':'prefix_wait',
            'info':graph_plan(e,e.initial,[0]*30)}
        g=SimpleNamespace(done=True,incumbent={'validation':{'energy_kwh':0}},_frontier=nodes,
            _checked={tuple(nodes[2]['info']['actions']):{'status':'FAIL'}})
        r=carryover_candidates(g)
        self.assertEqual([x['source_branch_ids'] for x in r['candidates']],[[0,8],[1],[3]])
        self.assertEqual([x['rank'] for x in r['candidates']],[0,1,2])
        self.assertEqual({x['eligibility'] for x in r['inventory']},{'ELIGIBLE','OTHER_ORIGIN','ALREADY_CHECKED','NO_STRICT_ENERGY_GAIN'})

    def test_carryover_first_gain_stops_no_staged_or_extra_wait(self):
        e=env();o=Oracle(e);nodes={0:node(e,0,0)};records=[]
        with patch('dnr.nature123_igrs_handoff.BranchSearch',guard_class(nodes)),patch('dnr.nature123_igrs_handoff.generate_staged_pickup_candidates') as generator:
            r=run_search(e,o,record_callback=records.append)
        generator.assert_not_called();self.assertEqual(r['handoff']['verify_used'],1)
        self.assertEqual(o.points,182);self.assertFalse(r['handoff']['extra_wait_validation'])
        self.assertEqual(r['status'],'SIMULATED_FIRST_COMPLETE_GAIN');self.assertFalse(r['physical_certified'])
        self.assertEqual(len(r['best_plan']['validation']['samples']),91)
        self.assertEqual(r['best_plan']['details']['source_branch_ids'],[0])

    def test_carryover_and_staged_share_exactly_sixteen_choices(self):
        e=env();nodes={i:node(e,i,i) for i in range(2)}
        # Reliable synthetic failure at INITIAL uses one logical request per choice.
        o=Oracle(e,fail=lambda s,t,n:n>91)
        candidates=[{'actions':node(e,20+i,i+2)['info']['actions'],'held_out_loads':[]} for i in range(18)]
        with patch('dnr.nature123_igrs_handoff.BranchSearch',guard_class(nodes)),patch('dnr.nature123_igrs_handoff.generate_staged_pickup_candidates',return_value={'candidates':candidates}):
            r=run_search(e,o)
        h=r['handoff'];self.assertEqual((h['verify_used'],h['carryover_verify_used'],h['staged_verify_used']),(16,2,14))
        self.assertEqual(o.points,107);self.assertEqual(r['stop_reason'],'SHARED_VERIFY_LIMIT')
        self.assertEqual(sum(x['status']=='NOT_ATTEMPTED' for x in h['staged_candidates']),4)
        self.assertEqual(r['selected_phase'],'GUARD')

    def test_sixteen_carryover_prevent_staged_and_retain_unattempted(self):
        e=env();o=Oracle(e,fail=lambda s,t,n:n>91);nodes={i:node(e,i,i) for i in range(20)}
        with patch('dnr.nature123_igrs_handoff.BranchSearch',guard_class(nodes)),patch('dnr.nature123_igrs_handoff.generate_staged_pickup_candidates') as generator:
            r=run_search(e,o)
        generator.assert_not_called();self.assertEqual(r['handoff']['verify_used'],16)
        self.assertEqual(sum(x['status']=='NOT_ATTEMPTED' for x in r['handoff']['selection']['candidates']),4)

    def test_stage_free_screen_and_original_alternating_held_window(self):
        rows=[{'branch_id':i,'metadata':{'held_out_loads':['x'] if i==1 else []},
            'info':{'priority':(-energy,0,0,()),'energy_kwh':energy}} for i,energy in enumerate([5,3,0])]
        incumbent={'priority':(0,0,0,()),'energy_kwh':0}
        chosen,pruned=staged_choice(rows,incumbent,0);self.assertEqual(chosen['branch_id'],0);self.assertEqual(len(pruned),1)
        chosen,_=staged_choice(rows,incumbent,1);self.assertEqual(chosen['branch_id'],1)

    def test_unknown_propagates_and_task_budget_never_resets(self):
        e=env();nodes={0:node(e,0,0)};o=Oracle(e,unknown=lambda s,t,n:n==92)
        with patch('dnr.nature123_igrs_handoff.BranchSearch',guard_class(nodes)):
            with self.assertRaises(CLPUReliabilityError):run_search(e,o)
        self.assertEqual(o.points,92)
        o=Oracle(e,budget=95)
        with patch('dnr.nature123_igrs_handoff.BranchSearch',guard_class(nodes)):
            r=run_search(e,o,BranchConfig(point_budget=95))
        self.assertEqual(o.points,95);self.assertEqual(r['handoff']['verify_used'],1)
        self.assertEqual(r['handoff']['selection']['candidates'][0]['status'],'BUDGET_EXHAUSTED_PARTIAL')

    def test_real_guard_still_stops_at_its_own_first_gain(self):
        e=env();o=Oracle(e);r=run_search(e,o)
        self.assertEqual(r['status'],'SIMULATED_FIRST_COMPLETE_GAIN')
        self.assertEqual(r['selected_phase'],'GUARD');self.assertFalse(r['early_stage_reached'])
        self.assertEqual(r['handoff']['verify_used'],0)


if __name__=='__main__':unittest.main()
