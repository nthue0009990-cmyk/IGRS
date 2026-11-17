"""Pure synthetic software verification, with no physical engine import/run."""
from pathlib import Path
import sys
import unittest
from itertools import product
import json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_nature123_clpu_environment import fixture
from dnr.nature123_igrs import (SearchEnvironment,SparseOptimisticPlanner,GraphBudgetExhausted,
    run_search,CLPUBudgetExhausted,CLPUReliabilityError,BranchConfig)
from dnr.nature123_igrs_search import graph_plan
from dnr.nature34_clpu_search import graph_plan as original_graph_plan


class Oracle:
    def __init__(self,env,fail=None,unknown=None,budget=512):
        self.env=env;self.fail=fail;self.unknown=unknown;self.budget=budget
        self.points=self.physical_points=self.requests=0
    def certify(self,state,t):
        if self.points>=self.budget:raise CLPUBudgetExhausted('synthetic fixed cap')
        self.points+=1;self.requests+=1
        status='FAIL' if self.fail and self.fail(state,t,self.points) else 'PASS'
        if self.unknown and self.unknown(state,t,self.points):status='UNKNOWN'
        return {'status':status,'reliable':status!='UNKNOWN','point_id':'SIMULATED-'+str(self.points),
            'request_sha256':self.env.snapshot_request(state,t)['request_sha256'],'evidence_kind':'SIMULATED','feedback':[]}


def env(faults=('sa',),**kwargs):
    base,_,_=fixture(faults,**kwargs)
    return SearchEnvironment(base,allow_simulated=True)


class MigrationTests(unittest.TestCase):
    def test_sparse_exact_bellman_matches_full_exhaustive_ties(self):
        e=env();p=e.planner
        def exhaustive(command,depth):
            if not depth:return (0,p.weight(command),0,())
            choices=[]
            for child in p.successors(command):
                tail=exhaustive(child,depth-1)
                choices.append((p.weight(child)+tail[0],tail[1],int(child!=command)+tail[2],(child,)+tail[3]))
            return min(choices,key=lambda x:(-x[0],-x[1],x[2],x[3]))
        for depth in range(5):
            result=p.solve(e.initial.command,depth);truth=exhaustive(e.initial.command,depth)
            self.assertEqual((result['cumulative_weight'],result['final_weight'],result['operations'],tuple(result['path'][1:])),truth)
        self.assertFalse(p.summary()['complete_reachable_space_enumerated'])

    def test_graph_cap_never_returns_approximate_path(self):
        e=env();p=SparseOptimisticPlanner(e.graph,state_cap=1)
        with self.assertRaises(GraphBudgetExhausted):p.solve(e.initial.command,3)

    def test_original_graph_plan_scoring_unchanged(self):
        e=env(horizon=4)
        for actions in ([0]*4,[1,4,0,1],[4,0,0,0]):
            before=original_graph_plan(e,e.initial,actions);after=graph_plan(e,e.initial,actions)
            self.assertEqual(before,after)

    def test_complete_requires_initial_plus_90_bound_samples(self):
        e=env();o=Oracle(e);r=e.validate_plan(e.initial,[0]*30,o)
        self.assertEqual(len(r['samples']),91)
        self.assertEqual([x['time_seconds'] for x in r['samples'][:5]],[0,0,30,60,60])
        self.assertFalse(r['physical_certified']);self.assertEqual(r['evidence_kind'],'SIMULATED')
        self.assertTrue(all(x['request_sha256'] and x['point_id'] for x in r['samples']))

    def test_failure_step_zero_based_and_unknown_propagates(self):
        e=env();r=e.validate_plan(e.initial,[0]*30,Oracle(e,fail=lambda s,t,n:n==5))
        self.assertEqual(r['failed_step'],1)
        self.assertIsNone(r['energy_kwh']);self.assertEqual(len(r['samples']),5)
        with self.assertRaises(CLPUReliabilityError):e.validate_plan(e.initial,[0]*30,Oracle(e,unknown=lambda s,t,n:n==3))

    def test_simulated_requires_explicit_opt_in(self):
        e=env();e.allow_simulated=False
        with self.assertRaises(CLPUReliabilityError):e.validate_plan(e.initial,[0]*30,Oracle(e))

    def test_first_gain_cold_and_simulated_never_physical(self):
        e=env();o=Oracle(e);records=[]
        r=run_search(e,o,config=BranchConfig(point_budget=512,max_branch_expansions=32,max_checked_candidates=64),record_callback=records.append)
        self.assertEqual(r['status'],'SIMULATED_FIRST_COMPLETE_GAIN')
        self.assertFalse(r['physical_certified']);self.assertFalse(r['early_stage_reached'])
        self.assertFalse(r['free_pruning_exercised']);self.assertEqual(r['own_wait_energy_kwh'],0)
        self.assertEqual(len(r['best_plan']['validation']['samples']),91)
        self.assertEqual(o.physical_points,0);self.assertEqual(o.points,o.requests)
        self.assertEqual(sum(x['origin']=='wait_incumbent' for x in records),1)
        with self.assertRaises(ValueError):run_search(e,o)

    def test_early_phase_reachable_and_keeps_guard_certificate(self):
        e=env();o=Oracle(e,fail=lambda s,t,n:bool(e.service_kw(s)))
        cfg=BranchConfig(point_budget=512,max_branch_expansions=1,max_checked_candidates=64)
        r=run_search(e,o,cfg)
        self.assertTrue(r['early_stage_reached'])
        self.assertFalse(r['first_complete_gain_found']);self.assertFalse(r['physical_certified'])
        self.assertEqual(r['selected_phase'],'GUARD')
        self.assertTrue(r['stages']['EARLY']['guard_first']['guard_reference']['sha256'])
        self.assertLessEqual(o.points,512)

    def test_actual_123_staged_graph_only_all_dispatch_dimensions(self):
        from dnr.nature123_static_state import SparseRestorationGraph
        from dnr.nature123_clpu_environment import CLPUEnvironment
        from dnr.nature123_igrs_staged_pickup import generate_staged_pickup_candidates
        root=Path(__file__).resolve().parents[1]
        # Input definitions only. No saved physical result or fixed path is read.
        freeze=json.loads((root/'runs/20261107T150100_158527Z_nc123_partial_acceptance/freeze.json').read_text(encoding='utf8'))
        graph=SparseRestorationGraph(json.loads((root/freeze['static_manifest_path']).read_text(encoding='utf8')),['l19'])
        e=SearchEnvironment(CLPUEnvironment(graph,model_fingerprint=freeze['model_fingerprint'],
            condition=freeze['condition'],event=freeze['event']))
        r=generate_staged_pickup_candidates(e)
        self.assertEqual(graph.nl,19);self.assertEqual(graph.action_count,42)
        self.assertTrue(r['candidates']);self.assertEqual(r['new_physical_points'],0)
        for candidate in r['candidates']:
            self.assertEqual(len(candidate['actions']),30)
            graph_plan(e,e.initial,candidate['actions'])


if __name__=='__main__':unittest.main()
