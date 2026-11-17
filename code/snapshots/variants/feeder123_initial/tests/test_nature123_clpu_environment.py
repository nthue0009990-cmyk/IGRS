"""Synthetic software fixtures only; no engine and no physical certificate."""
from copy import deepcopy
from dataclasses import replace
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dnr.nature123_static_state import SparseRestorationGraph
from dnr.nature123_clpu_environment import CLPUEnvironment, DynamicState, CLPUBudgetExhausted, CHECKS, RELIABILITY_CHECKS


def tiny():
    return {'nodes':['s','a','b','c'], 'source_bus':'s',
        'bus_edges':[['s','a'],['a','b'],['s','b'],['b','c']],
        'switch_names':['ab','sb'], 'switch_edges':[['a','b'],['s','b']],
        'original_lines':{'sa':['s','a'],'ab':['a','b'],'bc':['b','c']},
        'normally_closed':[1,0], 'dispatch_loads':['bload','cload'],
        'loads':{'aload':{'bus':'a','nominal_kw':10},
                 'bload':{'bus':'b','nominal_kw':20},
                 'cload':{'bus':'c','nominal_kw':30}}}


def fixture(faults=(), **kwargs):
    manifest=tiny()
    graph=SparseRestorationGraph(manifest,faults)
    condition={'condition_id':'SYNTHETIC_TEST_ONLY', 'source_sha256':'1'*64,
        'load_multiplier':1.0, 'loads':{n:{**row,'nominal_kvar':row['nominal_kw']/2}
            for n,row in manifest['loads'].items()},
        'der_setpoints':{}, 'operating_config':{'scope':'SYNTHETIC_TEST_ONLY'}}
    event={'event_id':'SYNTHETIC_TEST_ONLY', 'source_sha256':'2'*64, 'a':1.0, 'tau_seconds':60.0}
    env=CLPUEnvironment(graph,model_fingerprint='0'*64,condition=condition,event=event,**kwargs)
    return env,condition,event


class SimulatedOracle:
    evidence_kind='SIMULATED'
    def __init__(self, *, failure=None, unknown=None, budget=None, corrupt=None):
        self.failure,self.unknown,self.budget,self.corrupt=failure,unknown,budget,corrupt
        self.requests=[]
    def certify(self,request):
        if self.budget is not None and len(self.requests)>=self.budget:
            raise CLPUBudgetExhausted('Synthetic request budget reached')
        self.requests.append(deepcopy(request));i=len(self.requests)
        return {'status':'FAIL' if i==self.failure else 'UNKNOWN' if i==self.unknown else 'PASS',
            'reliable':i!=self.unknown,'point_id':f'SIMULATED-{i}', 'evidence_kind':'SIMULATED',
            'request_sha256':('f'*64 if i==self.corrupt else request['request_sha256'])}


class IncompletePhysicalClaimFixture:
    evidence_kind='PHYSICAL'
    def __init__(self):self.calls=0
    def certify(self,request):
        self.calls+=1
        return {'status':'PASS','reliable':True,'point_id':'INVALID-CLAIM-NOT-A-SOLVE',
            'evidence_kind':'PHYSICAL','request_sha256':request['request_sha256']}


class CLPUTests(unittest.TestCase):
    def test_wait_delay_and_no_predecay(self):
        env,_,_=fixture(('sa',),horizon=4)
        s0=env.initial
        self.assertTrue(all(p is None for p in s0.pickups))
        s1=env.advance(s0,1) # withdraw b readiness while off
        s2=env.advance(s1,4) # transfer via s-b at60s, b stays held
        s3=env.advance(s2,0) # wait one further slot
        s4=env.advance(s3,1) # b first pickup180s, full peak
        f=env.multipliers(s4,180)
        self.assertEqual(f['bload'],2.0)
        self.assertAlmostEqual(f['aload'],1+math.exp(-2))
        self.assertEqual(env.multipliers(s3,180)['bload'],1.0)
        self.assertEqual(env.service_kw(s4),60.0)
        request=env.snapshot_request(s4,180)
        self.assertEqual(request['loads']['bload']['requested_kw'],40)
        self.assertEqual(request['loads']['bload']['requested_kvar'],20)
        self.assertEqual(env.snapshot_request(s3,180)['loads']['bload']['enabled'],False)
        self.assertNotEqual(env.snapshot_request(s3,180)['request_sha256'],request['request_sha256'])

    def test_initial_steady_and_full_sampling(self):
        env,_,_=fixture();oracle=SimulatedOracle()
        self.assertEqual(env.initial.pickups,(-1,-1,-1))
        r=env.validate_plan(env.initial,[0]*30,oracle)
        self.assertEqual(r['status'],'SIMULATED_COMPLETE')
        self.assertFalse(r['physical_certified'])
        self.assertEqual(len(r['samples']),91)
        self.assertEqual(r['samples'][0]['phase'],'INITIAL')
        self.assertEqual([s['time_seconds'] for s in r['samples'][:5]],[0,0,30,60,60])
        self.assertEqual(r['energy_kwh'],30)
        self.assertEqual(len(oracle.requests[0]['loads']),3)
        self.assertTrue(all(v['clpu_multiplier']==1 for req in oracle.requests for v in req['loads'].values()))

    def test_normal_energy_excludes_clpu(self):
        env,_,_=fixture(('sa',),horizon=4)
        r=env.validate_plan(env.initial,[1,4,0,1],SimulatedOracle())
        self.assertEqual(r['status'],'SIMULATED_COMPLETE')
        self.assertAlmostEqual(r['energy_kwh'],(0+40+40+60)/60)
        self.assertEqual(r['normal_demand_denominator_kw'],60)

    def test_failure_unknown_budget_and_binding_stop(self):
        env,_,_=fixture()
        for oracle,status,count in [(SimulatedOracle(failure=1),'SIMULATED_FAIL',1),
                (SimulatedOracle(failure=5),'SIMULATED_FAIL',5),
                (SimulatedOracle(unknown=5),'UNKNOWN',5),
                (SimulatedOracle(corrupt=3),'UNKNOWN',3),
                (SimulatedOracle(budget=4),'BUDGET_EXHAUSTED',4)]:
            r=env.validate_plan(env.initial,[0]*30,oracle)
            self.assertEqual(r['status'],status)
            self.assertEqual(len(r['samples']),count)
            self.assertEqual(len(oracle.requests),count)
            self.assertIsNone(r['energy_kwh'])
            self.assertFalse(r['physical_certified'])

    def test_incomplete_paths_and_untrusted_physical_claim(self):
        env,_,_=fixture();oracle=SimulatedOracle()
        with self.assertRaises(ValueError):env.validate_plan(env.initial,[0]*29,oracle)
        self.assertEqual(oracle.requests,[])
        bogus=IncompletePhysicalClaimFixture()
        with self.assertRaises(ValueError):env.validate_plan(env.initial,[0]*30,bogus)
        self.assertEqual(bogus.calls,0)
        enabled,_,_=fixture(allow_physical=True)
        r=enabled.validate_plan(enabled.initial,[0]*30,bogus)
        self.assertEqual(r['status'],'UNKNOWN')
        self.assertEqual(bogus.calls,1)
        self.assertFalse(r['physical_certified'])

    def test_history_replay_strict_types_and_reinterruption(self):
        env,_,_=fixture(('sa',),horizon=4)
        state=env.advance(env.initial,4)
        with self.assertRaises(ValueError):env.advance(state,1)
        with self.assertRaises(ValueError):env.advance(state,3) # fixed a disconnect
        for forged in [replace(state,step=True),replace(state,command=True),
                       replace(state,pickups=(0,60,0)),replace(state,actions=(True,)),
                       replace(state,context_sha256='f'*64),replace(state,pickups=(False,0,0))]:
            with self.assertRaises(ValueError):env.snapshot_request(forged,30)
        self.assertEqual(DynamicState.from_dict(state.to_dict()),state)
        d=state.to_dict();d['step']='1'
        with self.assertRaises(ValueError):DynamicState.from_dict(d)
        with self.assertRaises(ValueError):env.snapshot_request(state,61)
        with self.assertRaises(ValueError):env.snapshot_request(state,True)

    def test_unreliable_physical_fail_not_treated_as_constraint_fail(self):
        env,_,_=fixture(allow_physical=True)
        class BadFailureClaim:
            evidence_kind='PHYSICAL'
            def __init__(self, bad):self.bad=bad;self.calls=0
            def certify(self, request):
                self.calls+=1
                return {'status':'FAIL','reliable':True,'point_id':'INVALID-FAIL-NOT-A-SOLVE',
                    'evidence_kind':'PHYSICAL','request_sha256':request['request_sha256'],
                    'checks':{k:'FAIL' if k==self.bad else 'PASS' for k in CHECKS}}
        for key in RELIABILITY_CHECKS:
            oracle=BadFailureClaim(key)
            result=env.validate_plan(env.initial,[0]*30,oracle)
            self.assertEqual(result['status'],'UNKNOWN')
            self.assertIsNone(result['energy_kwh'])
            self.assertFalse(result['physical_certified'])
            self.assertEqual(oracle.calls,1)

    def test_full_condition_immutable_and_identity(self):
        env,condition,event=fixture();old=env.snapshot_request(env.initial,0)
        condition['loads']['aload']['nominal_kw']=999;event['a']=20
        copy=env.condition;copy['der_setpoints']['new']={'kw':1,'kvar':2}
        self.assertEqual(env.snapshot_request(env.initial,0),old)
        other,_,_=fixture();other.model_fingerprint='a'*64
        with self.assertRaises(ValueError):other.snapshot_request(other.initial,0)
        env.graph.source='a'
        with self.assertRaises(ValueError):env.snapshot_request(env.initial,0)

    def test_missing_load_and_nonsourced_condition_rejected(self):
        env,c,e=fixture();del c['loads']['bload']
        with self.assertRaises(ValueError):CLPUEnvironment(env.graph,model_fingerprint='0'*64,condition=c,event=e)
        env,c,e=fixture();c['source_sha256']='unknown'
        with self.assertRaises(ValueError):CLPUEnvironment(env.graph,model_fingerprint='0'*64,condition=c,event=e)
        env,c,e=fixture();e['tau_seconds']=True
        with self.assertRaises(ValueError):CLPUEnvironment(env.graph,model_fingerprint='0'*64,condition=c,event=e)


if __name__=='__main__':unittest.main()
