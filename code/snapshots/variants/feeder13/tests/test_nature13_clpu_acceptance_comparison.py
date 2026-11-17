"""Synthetic comparator contract tests; no runtime or DSS imports."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest


path = Path(__file__).resolve().parents[1] / 'scripts/audit_nature13_clpu_acceptance.py'
spec = importlib.util.spec_from_file_location('nc13_acceptance_comparison', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    elements = [
        {'name': 'Load.live', 'enabled': True, 'buses': ['live.1'], 'conductors': 1,
         'phases': 1, 'terminals': 1, 'node_order': [1], 'conductor_open': [[False]],
         'currents_real_imag': [1., 0.], 'powers_kw_kvar': [1., 0.]},
        {'name': 'Load.dead', 'enabled': False, 'buses': ['dead.1'], 'conductors': 1,
         'phases': 1, 'terminals': 1, 'node_order': [99], 'conductor_open': [[False]],
         'currents_real_imag': [0., 0.], 'powers_kw_kvar': [float('nan'), float('nan')],
         'voltages_real_imag': [float('nan'), float('nan')]},
    ]
    layouts = {e['name']: {k: e[k] for k in ('buses', 'conductors', 'terminals')} for e in elements}
    metadata = {'physical_phase_nodes_by_bus': {'live': [1], 'dead': [1]},
                'neutral_nodes_by_bus': {'live': [0], 'dead': [0]},
                'element_layout': layouts, 'expected_element_names': list(layouts)}
    override = {'planned_enabled': {'Load.live': True, 'Load.dead': False},
                'deenergized_buses': ['dead']}
    row = {'algorithm': 0, 'scope': module.FROZEN_SCOPE, 'case': 'synthetic', 'action': [0], 'fault_lines': ['line.test'],
           'alpha': .375, 'clpu_multipliers': {'Load.live': 1., 'Load.dead': 1.},
           'time_minutes': 0., 'state_identity': {'command': 0}, 'metadata_override': override,
           'reliable': True, 'status': 'PASS', 'ac_audit': {'status': 'PASS'},
           'isolation': {'status': 'PASS', 'certified_deenergized_buses': ['dead'],
                         'excluded_node_states': {'dead.1': 'STRUCTURALLY_EXCLUDED_NOT_SOLVED'}},
           'known_constraints': {k: {'status': 'PASS', 'unknown': []} for k in module.REQUIRED_SECTIONS},
           'raw_outputs': {'converged': True, 'algorithm': 0, 'graph_nodes': ['live', 'dead'],
                           'graph_edges': [], 'node_names': ['live.1'], 'node_vmag_pu': [1.],
                           'elements': elements}}
    other = deepcopy(row)
    other['algorithm'] = other['raw_outputs']['algorithm'] = 1
    return row, other, metadata


class ComparisonTests(unittest.TestCase):
    def compare(self, a, b, meta):
        return module.compare_isolated_points(a, b, meta, module.FROZEN_LIMITS)

    def test_raw_nonfinite_label_replay_never_fills_values(self):
        left={'unknown':[float('nan'),float('inf'),-float('inf')]}
        right=deepcopy(left)
        self.assertTrue(module._saved_equal(left,right))
        right['unknown'][0]=0.
        self.assertFalse(module._saved_equal(left,right))

    def test_absent_dead_nodes_and_disabled_stale_caches_are_not_fabricated(self):
        a,b,m = fixture()
        b['raw_outputs']['elements'][1]['powers_kw_kvar'] = [123., 456.]
        result = self.compare(a,b,m)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['checked_energized_phase_nodes'], 1)
        self.assertEqual(result['checked_disabled_zero_current_elements'], 1)

    def test_disabled_nonzero_current_rejects(self):
        a,b,m = fixture()
        b['raw_outputs']['elements'][1]['currents_real_imag'][0] = 1e-5
        with self.assertRaises(module.ComparisonInputError): self.compare(a,b,m)

    def test_observed_dead_nan_is_unknown_not_exempt(self):
        a,b,m = fixture()
        b['raw_outputs']['node_names'].append('dead.1')
        b['raw_outputs']['node_vmag_pu'].append(float('nan'))
        with self.assertRaises(module.ComparisonInputError): self.compare(a,b,m)

    def test_live_nonfinite_and_missing_phase_reject(self):
        for value in (float('nan'), float('inf')):
            a,b,m = fixture()
            b['raw_outputs']['node_vmag_pu'][0] = value
            with self.assertRaises(module.ComparisonInputError): self.compare(a,b,m)
        a,b,m = fixture(); b['raw_outputs']['node_names'] = []; b['raw_outputs']['node_vmag_pu'] = []
        with self.assertRaises(module.ComparisonInputError): self.compare(a,b,m)

    def test_original_voltage_limit_is_not_relaxed(self):
        a,b,m = fixture(); b['raw_outputs']['node_vmag_pu'][0] += 2e-5
        self.assertEqual(self.compare(a,b,m)['status'], 'FAIL')

    def test_known_physical_fail_can_be_compared_but_disagreement_fails(self):
        a,b,m = fixture()
        for row in (a,b):
            row['status'] = 'FAIL'; row['known_constraints']['voltage']['status'] = 'FAIL'
        self.assertEqual(self.compare(a,b,m)['status'], 'PASS')
        b['status'] = 'PASS'; b['known_constraints']['voltage']['status'] = 'PASS'
        self.assertEqual(self.compare(a,b,m)['status'], 'FAIL')

    def test_scope_action_or_certificate_mismatch_rejects(self):
        for key,value in [('action',[1]),('alpha',.5),('reliable',False)]:
            a,b,m = fixture(); b[key] = value
            with self.assertRaises(module.ComparisonInputError): self.compare(a,b,m)


if __name__ == '__main__':
    unittest.main()
