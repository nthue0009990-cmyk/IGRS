"""Zero-physics accounting checks for cold search, cache and interrupted entries."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('nc123_igrs_runner_tested',ROOT/'scripts/run_nature123_igrs_acceptance.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class FakeEnvironment:
    def snapshot_request(self,state,seconds):
        return {'request_sha256':str(state)*64,'time_seconds':seconds}


class FakeRuntime:
    def __init__(self,*args,budget,**kwargs): self.counter=budget;self.interrupt=False
    def evaluate(self,request,point_id):
        self.counter.begin_request(point_id,request['request_sha256'])
        if self.interrupt: raise runner.AcceptanceBudgetStop('interrupted after reservation')
        for i in range(3):
            token=self.counter.reserve_entry(operation=str(i),count=1,request_id=point_id)
            self.counter.finish_entry(token,returned=True)
        return {'status':'PASS','reliable':True,'point_id':point_id,'request_sha256':request['request_sha256'],
                'evidence_kind':'PHYSICAL','checks':{},'known_constraints':{},'raw_outputs':{}}


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='nc123_runner_mock_',dir=ROOT/'runs')
        self.addCleanup(self.temp.cleanup)
        self.runtime_patch=patch('dnr.nature123_phase_available_physical.Nature123PhaseAvailablePhysicalRuntime',FakeRuntime)
        self.runtime_patch.start();self.addCleanup(self.runtime_patch.stop)
        self.valid=patch.object(runner,'certificate_valid',return_value=True)
        self.valid.start();self.addCleanup(self.valid.stop)
        self.feedback=patch('dnr.nature34_clpu_physical.structured_feedback',return_value={'usable_for_repair':True,'violations':[]})
        self.feedback.start();self.addCleanup(self.feedback.stop)
        self.oracle=runner.PrivateOracle(ROOT,Path(self.temp.name),FakeEnvironment(),{'physical_config':{}})

    def test_same_full_sha_private_hit_charged_logically(self):
        a=self.oracle.certify('a',0);b=self.oracle.certify('a',0)
        self.assertEqual(a,b)
        self.assertEqual((self.oracle.points,self.oracle.physical_points,self.oracle.cache_hits),(2,1,1))
        self.assertEqual(self.oracle.counter.snapshot()['reserved_entries'],3)

    def test_new_full_sha_is_new_paid_request(self):
        self.oracle.certify('a',0);self.oracle.certify('b',0)
        self.assertEqual((self.oracle.points,self.oracle.physical_points,self.oracle.cache_hits),(2,2,0))

    def test_partial_paid_request_not_refunded_or_cache_hit(self):
        self.oracle.runtime.interrupt=True
        with self.assertRaises(runner.CLPUBudgetExhausted): self.oracle.certify('a',0)
        self.assertEqual(self.oracle.physical_points,1)
        self.assertEqual(self.oracle.cache_hits,0)
        self.assertEqual(self.oracle.returned_physical_results,0)
        self.assertEqual(self.oracle.logical_completed,0)

    def test_unknown_never_enters_cache(self):
        with patch.object(runner,'certificate_valid',return_value=False):
            with self.assertRaises(runner.CLPUReliabilityError): self.oracle.certify('a',0)
        self.assertEqual(self.oracle.cache,{})
        self.assertEqual(self.oracle.physical_points,1)

    def test_logical_limit_even_cache_hit_stops_without_solve(self):
        self.oracle.certify('a',0);self.oracle.points=self.oracle.budget
        with self.assertRaises(runner.CLPUBudgetExhausted): self.oracle.certify('a',0)
        self.assertEqual(self.oracle.physical_points,1)


if __name__=='__main__': unittest.main()
