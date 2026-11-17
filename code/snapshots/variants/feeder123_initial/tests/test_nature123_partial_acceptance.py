"""Offline checks for exactly one partial restoration path; no DSS calls."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('nc123_partial_script',ROOT/'scripts/run_nature123_partial_acceptance.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
from dnr.nature123_acceptance_runner import RELIABILITY,CONSTRAINTS


class FakeRuntime:
    fail_at=None
    unknown_at=None
    def __init__(self,project,*,budget,physical_config,enable_physical):
        self.budget=budget;self.index=-1
    def evaluate(self,request,point_id):
        self.index+=1
        self.budget.begin_request(point_id,request['request_sha256'])
        token=self.budget.reserve_entry(operation='SYNTHETIC_ONLY',count=3,request_id=point_id)
        self.budget.finish_entry(token,returned=True)
        checks={k:'PASS' for k in RELIABILITY+CONSTRAINTS};status='PASS'
        if self.index==self.fail_at:status=checks['line_phase_thermal']='FAIL'
        if self.index==self.unknown_at:status=checks['ac_balance']='UNKNOWN'
        return {'status':status,'reliable':status!='UNKNOWN','point_id':point_id,
            'request_sha256':request['request_sha256'],'checks':checks,'evidence_kind':'PHYSICAL',
            'SYNTHETIC_TEST_ONLY':True}


class PartialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        latest=p.read_json(ROOT/'artifacts/nature123_full_sequence_20261107/freeze_latest.json')
        cls.template=p.derive_freeze(ROOT,Path(latest['run'])/'freeze.json')

    def setUp(self):
        self.freeze=deepcopy(self.template)

    def test_exact_single_path_and_independent_identity(self):
        plan=p.validate_freeze(ROOT,self.freeze)
        self.assertEqual(plan['actions'],[22,37]+[0]*28)
        self.assertEqual(len(plan['requests']),91)
        self.assertEqual(self.freeze['condition']['condition_id'],p.CONDITION_ID)
        self.assertAlmostEqual(plan['theoretical_nominal_energy_kwh'],1605+160*29/60)
        self.assertEqual(plan['requests'][3]['time_seconds'],plan['requests'][4]['time_seconds'])
        self.assertNotEqual(plan['requests'][3]['request_sha256'],plan['requests'][4]['request_sha256'])

    def test_closed_gate_and_exact_policy(self):
        with self.assertRaisesRegex(ValueError,'gate remains closed'):
            p.validate_freeze(ROOT,self.freeze,execute=True)
        self.freeze['physical_execution_allowed']=True
        self.freeze['readiness_checks']['software_regression']='PASS'
        p.validate_freeze(ROOT,self.freeze,execute=True)
        self.freeze['physical_config']['voltage_limits_pu']=[.85,1.15]
        with self.assertRaisesRegex(ValueError,'cannot change'):
            p.validate_freeze(ROOT,self.freeze)

    def test_no_alternative_plan_budget_history_or_parameter(self):
        for field in ('action','budget','history','event','energy','scope'):
            f=deepcopy(self.freeze)
            if field=='action':f['plans'][0]['actions'][10]=22
            elif field=='budget':f['budgets']['per_plan']['max_requests']=92
            elif field=='history':f['plans'][0]['requests'][4]['time_seconds']+=1
            elif field=='event':f['event']['a']+=.001
            elif field=='energy':f['plans'][0]['theoretical_nominal_energy_kwh']+=1
            else:
                f['physical_execution_allowed']=True;f['readiness_checks']['software_regression']='PASS'
                f['authorization']['scope']='FULL_SEQUENCE_ACCEPTANCE_30MIN'
            with self.subTest(field=field),self.assertRaises(ValueError):
                p.validate_freeze(ROOT,f,execute=field=='scope')

    def run_fake(self,runtime):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);plan=self.freeze['plans'][0]
            rc=p.worker(ROOT,folder,self.freeze,plan,runtime_factory=runtime)
            return p.reconcile_plan(folder,plan,{'returncode':rc,'termination_reason':None},p.LIMITS)

    def test_full_mock_keeps_energy_uncertified(self):
        result=self.run_fake(FakeRuntime)
        self.assertEqual(result['status'],'SIMULATED_COMPLETE')
        self.assertEqual(result['reserved_requests'],91)
        self.assertEqual(result['reserved_entries'],273)
        self.assertIsNone(result['certified_energy_kwh'])
        self.assertFalse(result['physical_certified'])

    def test_reliable_failure_preserves_unexecuted_suffix(self):
        class Failure(FakeRuntime):fail_at=4
        result=self.run_fake(Failure)
        self.assertEqual(result['status'],'SIMULATED_RELIABLE_FAIL')
        self.assertEqual(result['counts'],{'PASS':4,'FAIL':1,'UNKNOWN':0,'NOT_RUN':86})
        self.assertIsNone(result['energy_kwh'])

    def test_unknown_stops_without_certifying_partial_energy(self):
        class Unknown(FakeRuntime):unknown_at=4
        result=self.run_fake(Unknown)
        self.assertEqual(result['status'],'SIMULATED_UNKNOWN')
        self.assertEqual(result['counts'],{'PASS':4,'FAIL':0,'UNKNOWN':1,'NOT_RUN':86})
        self.assertIsNone(result['energy_kwh'])


if __name__=='__main__':unittest.main()
