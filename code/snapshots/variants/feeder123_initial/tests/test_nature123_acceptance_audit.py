import copy
import unittest

from dnr.nature123_acceptance_audit import (compare_saved_pair, certificate_errors,
                                           audit_saved_acceptance, request_hash)
from dnr.nature123_clpu_environment import RELIABILITY_CHECKS, CONSTRAINT_CHECKS

TOL = dict(voltage_absolute_pu=1e-5, current_absolute_A=.01, power_absolute_kw=.01)


def raw():
    return {'node_names': ['1.1'], 'node_vmag_pu': [1.], 'elements': [
        {'name': 'load.a', 'enabled': True, 'buses': ['1.1'], 'phases': 1,
         'conductors': 1, 'terminals': 1, 'node_order': [1],
         'currents_mag_angle': [1., 0.], 'powers_kw_kvar': [1., .5]}]}


class AuditTests(unittest.TestCase):
    def test_all_terminal_values_compared(self):
        a, b = raw(), raw()
        b['elements'][0]['powers_kw_kvar'][1] += .02
        row = compare_saved_pair(a, b, TOL)
        self.assertEqual(row['status'], 'DISAGREEMENT')
        self.assertIn('power_absolute_kw', row['exceeded_tolerances'])

    def test_equal_missing_and_nonfinite(self):
        self.assertEqual(compare_saved_pair(raw(), raw(), TOL)['status'], 'AGREE')
        b = raw(); b['node_vmag_pu'][0] = float('nan')
        self.assertEqual(compare_saved_pair(raw(), b, TOL)['status'], 'UNKNOWN')
        b = raw(); del b['elements'][0]['node_order']
        self.assertEqual(compare_saved_pair(raw(), b, TOL)['status'], 'UNKNOWN')
        self.assertEqual(compare_saved_pair(None, None, TOL)['status'], 'UNKNOWN')

    def test_reliable_fail_is_retained_and_mock_rejected(self):
        request = {'identity': 'a'}; request['request_sha256'] = request_hash(request)
        cert = {'request_sha256': request['request_sha256'], 'reliable': True,
                'evidence_kind': 'PHYSICAL', 'status': 'FAIL',
                'checks': {k: 'PASS' for k in RELIABILITY_CHECKS + CONSTRAINT_CHECKS}}
        cert['checks']['voltage'] = 'FAIL'
        self.assertEqual(certificate_errors(request, cert), [])
        cert['evidence_kind'] = 'SIMULATED'
        self.assertTrue(certificate_errors(request, cert))

    def test_missing_pairs_not_promoted_and_binding_rejected(self):
        schedule = []
        for i in range(16):
            req = {'identity': i, 'loads': {}, 'fault_lines': [], 'effective_action': [],
                   'time_seconds': i//2, 'model_fingerprint': 'a', 'event': {},
                   'condition': {'der_setpoints': {}, 'operating_config': {'algorithm': i%2}}}
            req['request_sha256'] = request_hash(req)
            schedule.append({'order': i, 'algorithm': i%2, 'label': str(i//2), 'request': req})
        result = audit_saved_acceptance(schedule, {}, TOL)
        self.assertEqual(result['status'], 'NOT_ACCEPTED')
        self.assertEqual(len(result['pairs']), 8)
        self.assertTrue(all(p['status'] == 'UNKNOWN' for p in result['pairs']))
        certs = {}
        for row in schedule:
            h = row['request']['request_sha256']
            certs[h] = {'request_sha256': h, 'evidence_kind': 'PHYSICAL', 'reliable': True,
                'status': 'PASS', 'checks': {k: 'PASS' for k in RELIABILITY_CHECKS + CONSTRAINT_CHECKS},
                'raw_outputs': {**raw(), 'algorithm': 0}, 'operating_readback': {'algorithm': 0}}
        mismatch = audit_saved_acceptance(schedule, certs, TOL)
        self.assertTrue(all(p['status'] == 'UNKNOWN' for p in mismatch['pairs']))
        for row in schedule:
            c = certs[row['request']['request_sha256']]
            c['raw_outputs']['algorithm'] = c['operating_readback']['algorithm'] = row['algorithm']
        self.assertEqual(audit_saved_acceptance(schedule, certs, TOL)['status'], 'PAIRED_AGREEMENT')
        modified = copy.deepcopy(schedule); modified[0]['request']['time_seconds'] = 100
        with self.assertRaises(ValueError):
            audit_saved_acceptance(modified, {}, TOL)


if __name__ == '__main__':
    unittest.main()
