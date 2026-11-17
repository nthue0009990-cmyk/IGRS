"""Saved-v1 rejection and explicit synthetic scope tests: zero real DSS calls."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from dnr.nature123_clpu_physical import proposed_physical_config
from dnr.nature123_phase_available_physical import (DER_POLICY, IDLE_POLICY,
    Nature123PhaseAvailablePhysicalRuntime, declared_node_order,
    derive_der_availability, certify_idle_phase_scope, certify_phase_available_snapshot, _checker_view)
from dnr.nature123_acceptance_budget import AcceptanceBudget
from test_nature123_clpu_physical import FakeEngine

ROOT=Path(__file__).resolve().parents[1]
POINT=ROOT/'runs/20261107T133711_696746Z_nc123_entry_acceptance/points/point_00.json'


def source_metadata(point):
    meta=deepcopy(point['metadata']); elements={e['name']:e for e in point['raw_outputs']['elements']}
    for name,layout in meta['element_layout'].items():
        layout['phases']=elements[name]['phases']
        layout['node_order']=declared_node_order(layout)
    return meta


def synthetic_scope():
    """Artificial idle-node proof fixture; never a physical certificate."""
    e={'name':'line.off','enabled':False,'buses':['a.1','b.1'],'conductors':1,'terminals':2,
       'phases':1,'node_order':[1,1],'currents_real_imag':[0.,0.,0.,0.]}
    layout={k:deepcopy(e[k]) for k in ('buses','conductors','terminals','phases','node_order')}
    layout['kind']='line'
    raw={'evidence_kind':'SIMULATED','converged':True,'elements':[e],
         'node_names':['a.1','b.1'],'node_vmag_pu':[0.,0.], 'node_voltages_real_imag':[0.,0.,0.,0.]}
    meta={'expected_element_names':['line.off'],'physical_phase_nodes_by_bus':{'a':[1],'b':[1]},
          'node_phases':{'a':[1],'b':[1]},'element_layout':{'line.off':layout},
          'node_base_kv_ln':{'a':2.4,'b':2.4},'isolation_current_tolerance_a':1e-8,
          'isolation_voltage_tolerance_pu':1e-8}
    return raw,meta


class AvailabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.point=json.loads(POINT.read_text(encoding='utf8'))
        cls.meta=source_metadata(cls.point)
        cls.config=proposed_physical_config(ROOT)

    def test_all250_source_declared_layouts_parse_including_two_phase(self):
        self.assertEqual(len(self.meta['element_layout']),250)
        for name,expected in [('line.l25',[1,3,1,3]),('line.l27',[1,3,1,3]),('line.l35',[1,2,1,2])]:
            self.assertEqual(self.meta['element_layout'][name]['node_order'],expected)
        # Source-open disabled-cache duplicates are not adopted as declarations.
        self.assertEqual(self.meta['element_layout']['line.sw8']['node_order'],[1,2,3,1,2,3])
        self.assertEqual(self.meta['element_layout']['line.l120']['node_order'],[1,2,3,1,2,3])

    def test_actual_v1_readback_der_plan_five_devices740kw(self):
        plan=derive_der_availability(self.point['raw_outputs'],self.meta)
        self.assertEqual(plan['available_generators'],['generator.g10','generator.g11','generator.g12','generator.g5','generator.g7'])
        self.assertEqual(plan['available_kw'],740.)
        self.assertEqual(plan['source_nominal_kw'],2230.)
        self.assertEqual(len(plan['unavailable_generators']),8)
        self.assertEqual(sum(len(r['missing_source_phase_nodes']) for r in plan['generators'].values()),16)

    def test_per_topology_recomputed_after_source_path_disconnection(self):
        raw=deepcopy(self.point['raw_outputs'])
        for e in raw['elements']:
            if e['name']=='transformer.reg1a': e['enabled']=False
        plan=derive_der_availability(raw,self.meta)
        self.assertEqual(plan['available_kw'],0.)
        self.assertEqual(len(plan['unavailable_generators']),13)

    def test_v1_unchanged_actual_data_stays_unknown_under_v2_audit(self):
        raw=deepcopy(self.point['raw_outputs']); meta=deepcopy(self.meta)
        # Predicting the new decision is only an audit annotation; observed
        # enabled states/currents/voltages are deliberately NOT changed.
        meta['der_availability_plan']=derive_der_availability(raw,meta)
        before=deepcopy(raw); before_meta=deepcopy(meta)
        result=certify_phase_available_snapshot(raw,meta,self.config)
        self.assertEqual(result['status'],'UNKNOWN')
        self.assertFalse(result['reliable'])
        self.assertEqual(len(result['der_availability_mismatches']),8)
        self.assertEqual(raw,before)
        self.assertEqual(meta,before_meta)
        self.assertNotIn('39.1',result['idle_phase_scope']['certified_excluded_phase_nodes'])

    def test_disabled_generator_with_unproven_currents_has_no_zero_supply_claim(self):
        raw=deepcopy(self.point['raw_outputs']); meta=deepcopy(self.meta)
        # Deliberately contradictory saved evidence: changing enabled status
        # without replacing currents must remain UNKNOWN, never a new result.
        for e in raw['elements']:
            if e['name']=='generator.g1':
                e['enabled']=False
                e['currents_real_imag']=[]
        meta['planned_enabled']['generator.g1']=False
        meta['der_availability_plan']=derive_der_availability(raw,meta)
        result=certify_phase_available_snapshot(raw,meta,self.config)
        self.assertEqual(result['status'],'UNKNOWN')
        self.assertIsNone(result['der_power_ledger']['supply_accounting_kw_by_generator']['generator.g1'])
        self.assertIsNone(result['der_power_ledger']['supply_accounting_kw'])

    def test_source_v2_gate_and_full_policy_binding_without_engine(self):
        config={**self.config,'phase_availability_policy':DER_POLICY,'idle_phase_policy':IDLE_POLICY}
        rt=Nature123PhaseAvailablePhysicalRuntime(ROOT,budget=AcceptanceBudget(),physical_config=config,
            enable_physical=True,engine=FakeEngine())
        self.assertEqual(rt.bundle.config['phase_availability_policy'],DER_POLICY)
        self.assertEqual(rt.evidence_kind,'SIMULATED')
        self.assertEqual(rt.dss.calls,[])
        config['idle_phase_policy']='IGNORE_GHOSTS'
        with self.assertRaisesRegex(ValueError,'unbound'):
            Nature123PhaseAvailablePhysicalRuntime(ROOT,budget=AcceptanceBudget(),physical_config=config,
                enable_physical=True,engine=FakeEngine())


class ScopeTests(unittest.TestCase):
    def test_explicit_synthetic_disabled_zero_node_scope(self):
        raw,meta=synthetic_scope(); original=deepcopy(raw)
        cert=certify_idle_phase_scope(raw,meta)
        self.assertEqual(cert['status'],'PASS')
        self.assertEqual(cert['certified_excluded_phase_nodes'],['a.1','b.1'])
        view,scoped,mapping=_checker_view(raw,meta,cert)
        self.assertEqual(view['node_names'],[])
        self.assertEqual(len(view['elements']),1)
        self.assertEqual(raw,original)
        self.assertEqual(mapping['retained_original_node_indices'],[])
        self.assertEqual(scoped['physical_phase_nodes_by_bus'],{'a':[],'b':[]})

    def test_missing_nan_or_nonzero_disabled_current_not_certified(self):
        for currents,status in [([], 'UNKNOWN'),([float('nan'),0.,0.,0.],'UNKNOWN'),([.001,0.,0.,0.],'FAIL')]:
            with self.subTest(currents=currents):
                raw,meta=synthetic_scope(); raw['elements'][0]['currents_real_imag']=currents
                cert=certify_idle_phase_scope(raw,meta)
                self.assertEqual(cert['status'],status)
                self.assertEqual(cert['certified_excluded_phase_nodes'],[])

    def test_nonzero_pu_or_complex_voltage_cannot_be_excluded(self):
        for key,value in [('node_vmag_pu',[1.,0.]),('node_voltages_real_imag',[1.,0.,0.,0.])]:
            raw,meta=synthetic_scope(); raw[key]=value
            self.assertEqual(certify_idle_phase_scope(raw,meta)['status'],'FAIL')

    def test_active_device_node_cannot_be_removed_even_voltage_zero(self):
        raw,meta=synthetic_scope(); raw['elements'][0]['enabled']=True
        cert=certify_idle_phase_scope(raw,meta)
        self.assertEqual(cert['certified_excluded_phase_nodes'],[])
        self.assertEqual(cert['required_enabled_phase_nodes'],['a.1','b.1'])

    def test_absent_node_allowed_only_with_disabled_zero_current_evidence(self):
        raw,meta=synthetic_scope()
        raw.update(node_names=[],node_vmag_pu=[],node_voltages_real_imag=[])
        cert=certify_idle_phase_scope(raw,meta)
        self.assertEqual(cert['status'],'PASS')
        self.assertTrue(all(r['observation']=='STRUCTURALLY_EXCLUDED_NOT_SOLVED' for r in cert['nodes']))

    def test_unexpected_nodes_and_enabled_nodeorder_mismatch_fail_closed(self):
        raw,meta=synthetic_scope(); raw['elements'][0]['enabled']=True
        raw['elements'][0]['node_order']=[1,2]
        self.assertEqual(certify_idle_phase_scope(raw,meta)['status'],'UNKNOWN')
        raw,meta=synthetic_scope(); meta['physical_phase_nodes_by_bus']['c']=[1]
        self.assertEqual(certify_idle_phase_scope(raw,meta)['status'],'UNKNOWN')

    def test_disabled_stale_nodeorder_never_redefines_declared_nodes(self):
        raw,meta=synthetic_scope(); raw['elements'][0]['node_order']=[3,3]
        cert=certify_idle_phase_scope(raw,meta)
        self.assertEqual(cert['status'],'PASS')
        self.assertEqual(cert['certified_excluded_phase_nodes'],['a.1','b.1'])


if __name__=='__main__': unittest.main()
