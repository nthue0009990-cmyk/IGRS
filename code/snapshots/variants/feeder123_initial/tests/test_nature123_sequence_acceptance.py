"""Fixed-sequence lifecycle and replay tests. All physical envelopes are synthetic."""
from copy import deepcopy
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dnr import nature123_sequence_acceptance as s
from dnr.nature123_acceptance_runner import RELIABILITY,CONSTRAINTS


def fixture():
    old_run=Path(s.read_json(ROOT/'artifacts/nature123_phase_available_20261107/freeze_latest.json')['run'])
    old=s.read_json(old_run/'freeze.json')
    req=s.read_json(ROOT/old['schedule']['path'])[0]['request']
    graph_path=Path(s.read_json(ROOT/'artifacts/epsr123_sparse_20261107/latest.json')['run'])/'static_manifest.json'
    graph=s.SparseRestorationGraph(s.read_json(graph_path),('l19',))
    env=s.CLPUEnvironment(graph,model_fingerprint=req['model_fingerprint'],condition=req['condition'],event=req['event'])
    plans=[]
    for i,(name,actions) in enumerate(s.PRESPECIFIED_PLANS):
        states=env.plan_states(env.initial,actions)
        requests=[env.snapshot_request(env.initial,0)]
        requests += [env.snapshot_request(state,(state.step-1)*60+t) for state in states[1:] for t in (0,30,60)]
        plans.append({'plan_id':name,'label':'SYNTHETIC_TEST_PLAN_'+str(i),'actions':actions,
            'requests':requests,'theoretical_nominal_energy_kwh':math.fsum(env.service_kw(st)/60 for st in states[1:])})
    source_paths=set(s.REQUIRED)|{graph_path.relative_to(ROOT).as_posix()}
    return {'schema_version':s.SCHEMA,'physical_execution_allowed':False,
        'source_sha256':{p:s.sha(ROOT/p) for p in source_paths},'static_manifest_path':graph_path.relative_to(ROOT).as_posix(),
        'fault_lines':['l19'],'physical_config':old['physical_config'],'model_fingerprint':req['model_fingerprint'],
        'condition':req['condition'],'event':req['event'],'plans':plans,
        'budgets':{'per_plan':{'max_requests':91,'max_entries':273,'seconds':240,'memory_bytes':4*1024**3,'poll_seconds':.05},
                   'global':{'max_plans':4,'max_requests':364,'max_entries':1092,'seconds':960}},
        'readiness_checks':{'software':'PASS'},'authorization':{'scope':s.SCOPE,'record':'SYNTHETIC_TEST_ONLY'}}


class FakeRuntime:
    fail_at=None
    unknown_at=None
    raise_at=None
    def __init__(self,project,*,budget,physical_config,enable_physical):
        self.budget=budget
        self.index=-1
    def evaluate(self,request,point_id):
        self.index+=1
        self.budget.begin_request(point_id,request['request_sha256'])
        for operation in ('SYNTHETIC_COMPILE','SYNTHETIC_INITIALIZE','SYNTHETIC_REQUEST'):
            token=self.budget.reserve_entry(operation=operation,count=1,request_id=point_id)
            if self.index==self.raise_at:
                raise RuntimeError('SYNTHETIC_RESOURCE_STOP')
            self.budget.finish_entry(token,returned=True)
        checks={k:'PASS' for k in RELIABILITY+CONSTRAINTS}
        status='PASS'
        if self.index==self.fail_at:
            status=checks['line_phase_thermal']='FAIL'
        if self.index==self.unknown_at:
            status=checks['ac_balance']='UNKNOWN'
        return {'status':status,'reliable':status!='UNKNOWN','evidence_kind':'PHYSICAL',
            'point_id':point_id,'request_sha256':request['request_sha256'],'checks':checks,
            'SYNTHETIC_TEST_ONLY_NO_ENGINE':True}


class SequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template=fixture()

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=os.environ.get('NC123_SEQUENCE_TEST_ROOT'))
        self.run=Path(self.temp.name)
        self.freeze=deepcopy(self.template)

    def tearDown(self):
        if os.environ.get('NC123_SEQUENCE_TEST_ROOT'):
            import shutil
            shutil.copytree(self.run,Path(os.environ['NC123_SEQUENCE_TEST_ROOT'])/self.id().split('.')[-1])
        self.temp.cleanup()

    def run_fake(self,klass,plan_index=0):
        plan=self.freeze['plans'][plan_index]
        rc=s.worker(ROOT,self.run,self.freeze,plan,runtime_factory=klass)
        result=s.reconcile_plan(self.run,plan,{'returncode':rc,'termination_reason':None},self.freeze['budgets']['per_plan'])
        return result

    def test_full_request_replay_and_boundary_identity(self):
        self.assertEqual(len(s.validate_freeze(ROOT,self.freeze)),4)
        plan=self.freeze['plans'][0]
        self.assertEqual(len(plan['requests']),91)
        self.assertEqual(plan['requests'][3]['time_seconds'],plan['requests'][4]['time_seconds'])
        self.assertNotEqual(plan['requests'][3]['request_sha256'],plan['requests'][4]['request_sha256'])
        with self.assertRaisesRegex(ValueError,'gate is closed'):
            s.validate_freeze(ROOT,self.freeze,execute=True)
        self.freeze['physical_execution_allowed']=True
        self.assertEqual(len(s.validate_freeze(ROOT,self.freeze,execute=True)),4)

    def test_sampling_energy_gate_and_budget_tampering_rejected(self):
        for mutate in ('request','energy','actions','algorithm','budget','source'):
            f=deepcopy(self.freeze)
            if mutate=='request':
                f['plans'][1]['requests'][4]['time_seconds']+=1
            elif mutate=='energy':
                f['plans'][0]['theoretical_nominal_energy_kwh']+=.1
            elif mutate=='actions':
                f['plans'][0]['actions'][0]=18
            elif mutate=='algorithm':
                f['condition']['operating_config']['algorithm']=1
            elif mutate=='budget':
                f['budgets']['per_plan']['max_requests']=92
            else:
                f['source_sha256']['src/dnr/nature123_sequence_acceptance.py']='0'*64
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):
                s.validate_freeze(ROOT,f)

    def test_complete_mock_never_certifies_physical_energy(self):
        result=self.run_fake(FakeRuntime)
        self.assertEqual(result['status'],'SIMULATED_COMPLETE')
        self.assertEqual(result['reserved_requests'],91)
        self.assertEqual(result['reserved_entries'],273)
        self.assertEqual(result['counts']['PASS'],91)
        self.assertIsNone(result['certified_energy_kwh'])
        self.assertFalse(result['physical_certified'])

    def test_first_reliable_fail_stops_plan_and_preserves_not_run(self):
        class Failure(FakeRuntime): fail_at=5
        result=self.run_fake(Failure,1)
        self.assertEqual(result['status'],'SIMULATED_RELIABLE_FAIL')
        self.assertEqual(result['counts'],{'PASS':5,'FAIL':1,'UNKNOWN':0,'NOT_RUN':85})
        self.assertEqual(result['reserved_entries'],18)
        self.assertIsNone(result['certified_energy_kwh'])
        self.assertGreater(result['theoretical_nominal_energy_kwh'],0)

    def test_unknown_and_unreturned_numerical_entry_stop(self):
        class Interrupted(FakeRuntime): raise_at=2
        result=self.run_fake(Interrupted)
        self.assertEqual(result['status'],'SIMULATED_UNKNOWN')
        self.assertEqual(result['reserved_entries'],7)
        self.assertEqual(result['unreported_entries'],1)
        self.assertEqual(result['counts'],{'PASS':2,'FAIL':0,'UNKNOWN':1,'NOT_RUN':88})

    def test_early_unknown_retains_prior_bound_passes(self):
        class EarlyUnknown(FakeRuntime):
            def evaluate(self,request,point_id):
                if self.index+1!=2:
                    return super().evaluate(request,point_id)
                self.index+=1
                self.budget.begin_request(point_id,request['request_sha256'])
                token=self.budget.reserve_entry(operation='SYNTHETIC_EARLY_RETURN',count=2,request_id=point_id)
                self.budget.finish_entry(token,returned=True)
                return {'status':'UNKNOWN','reliable':False,'evidence_kind':'PHYSICAL','point_id':point_id,
                    'request_sha256':request['request_sha256'],'checks':{k:'UNKNOWN' for k in RELIABILITY+CONSTRAINTS}}
        result=self.run_fake(EarlyUnknown)
        self.assertEqual(result['status'],'SIMULATED_UNKNOWN')
        self.assertEqual(result['counts'],{'PASS':2,'FAIL':0,'UNKNOWN':1,'NOT_RUN':88})
        self.assertEqual(result['reserved_entries'],8)
        self.assertFalse(result['journal_errors'])

    def test_duplicate_journal_and_partial_point_fail_closed(self):
        class Failure(FakeRuntime): fail_at=1
        self.run_fake(Failure)
        events,_=s.read_journal(self.run/'events.jsonl')
        event=next(e for e in events if e['event']=='request_reserved')
        s.DurableJournal(self.run/'events.jsonl')(event)
        result=s.reconcile_plan(self.run,self.freeze['plans'][0],{'returncode':0,'termination_reason':None},self.freeze['budgets']['per_plan'])
        self.assertEqual(result['status'],'SIMULATED_UNKNOWN')
        self.assertTrue(result['journal_errors'])

    def batch_launcher(self,statuses):
        self.called=[]
        def launch(plan,run,seconds):
            index=len(self.called)
            self.called.append(plan['plan_id'])
            status=statuses[index]
            result=s.initial_plan_status(plan)
            result.update(status='SIMULATED_'+status,reserved_requests=91 if status=='COMPLETE' else 1,
                reserved_entries=273 if status=='COMPLETE' else 3,synthetic_test_only=True)
            s.write_json(run/'result.json',result)
            return result
        return launch

    def test_all_wait_must_complete_before_any_other_plan(self):
        result=s.run_batch(ROOT,self.run,self.freeze,launch_plan=self.batch_launcher(['RELIABLE_FAIL']))
        self.assertEqual(len(self.called),1)
        self.assertEqual(result['stop_reason'],'ALL_WAIT_NOT_COMPLETELY_CERTIFIED')
        self.assertEqual([p['status'] for p in result['plans']][1:],['NOT_RUN']*3)
        self.assertTrue(all(len(p['requests'])==91 for p in result['plans']))

    def test_negative_control_failure_does_not_skip_staged_plans(self):
        result=s.run_batch(ROOT,self.run,self.freeze,launch_plan=self.batch_launcher(['COMPLETE','RELIABLE_FAIL','COMPLETE','COMPLETE']))
        self.assertEqual(len(self.called),4)
        self.assertEqual(result['status'],'SIMULATED_BATCH_FINISHED')
        self.assertTrue(all(p['certified_energy_kwh'] is None for p in result['plans']))
        self.assertEqual(result['reserved_requests'],274)
        self.assertEqual(result['reserved_entries'],822)

    def test_unknown_stops_remainder_and_global_bound_cannot_be_reused(self):
        result=s.run_batch(ROOT,self.run,self.freeze,launch_plan=self.batch_launcher(['COMPLETE','UNKNOWN']))
        self.assertEqual(len(self.called),2)
        self.assertEqual(result['stop_reason'],'RELIABILITY_OR_RESOURCE_STOP')
        self.assertEqual(result['plans'][2]['status'],'NOT_RUN')
        second=self.run/'second'
        second.mkdir()
        self.freeze['budgets']['global']['max_requests']=91
        result=s.run_batch(ROOT,second,self.freeze,launch_plan=self.batch_launcher(['COMPLETE']))
        self.assertEqual(len(self.called),1)
        self.assertEqual(result['stop_reason'],'GLOBAL_RESOURCE_BOUND')

    def test_supervisor_start_exception_preserves_unrun_batch(self):
        def broken(plan,run,seconds):
            raise OSError('SYNTHETIC_JOB_CREATION_FAILURE')
        result=s.run_batch(ROOT,self.run,self.freeze,launch_plan=broken)
        self.assertEqual(result['status'],'SIMULATED_BATCH_STOPPED')
        self.assertEqual(result['plans'][0]['status'],'SIMULATED_UNKNOWN')
        self.assertEqual([p['status'] for p in result['plans'][1:]],['NOT_RUN']*3)
        self.assertTrue(all(p['certified_energy_kwh'] is None for p in result['plans']))


if __name__=='__main__': unittest.main()
