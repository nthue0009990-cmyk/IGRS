"""Pure cross-algorithm audit of saved NC123 entry points; never invokes DSS.

Agreement is a numerical consistency check, not an independent solver validation
or a certificate for an entire restoration sequence. Missing pairs stay unknown.
"""
from __future__ import annotations

import hashlib
import json
import math

from .nature123_clpu_environment import RELIABILITY_CHECKS, CONSTRAINT_CHECKS


def request_hash(request):
    payload = {k: v for k, v in request.items() if k != 'request_sha256'}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode('utf8')).hexdigest()


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def certificate_errors(request, cert):
    errors = []
    if not isinstance(cert, dict):
        return ['Missing saved certificate']
    if cert.get('request_sha256') != request_hash(request):
        errors.append('Certificate request binding differs')
    if cert.get('evidence_kind') != 'PHYSICAL':
        errors.append('Evidence is not PHYSICAL')
    checks = cert.get('checks', {})
    if not isinstance(checks, dict):
        return errors + ['Malformed checks']
    if cert.get('reliable') is not True or cert.get('status') not in ('PASS', 'FAIL'):
        errors.append('Point is not a reliable PASS/FAIL')
    if any(checks.get(k) != 'PASS' for k in RELIABILITY_CHECKS):
        errors.append('Missing reliability evidence')
    if any(checks.get(k) not in ('PASS', 'FAIL') for k in CONSTRAINT_CHECKS):
        errors.append('Missing known-constraint evidence')
    elif (all(checks[k] == 'PASS' for k in CONSTRAINT_CHECKS)) != (cert.get('status') == 'PASS'):
        errors.append('Overall status contradicts component checks')
    return errors


def reaudit_saved_certificate(request, cert, physical_config):
    """Recompute all pure physical checks from saved evidence, with no engine."""
    errors = certificate_errors(request, cert)
    if errors:
        return {'status': 'UNKNOWN', 'errors': errors, 'new_physical_calls': 0}
    if physical_config.get('runtime_variant', 'AUTHOR_V1') == 'PHASE_AVAILABLE_V2':
        from .nature123_phase_available_physical import certify_phase_available_snapshot as certify_snapshot
    else:
        from .nature123_clpu_physical import certify_snapshot
    try:
        derived = certify_snapshot(cert['raw_outputs'], cert['metadata'], physical_config)
        if any(derived.get(k) != cert.get(k) for k in ('status', 'reliable', 'checks')):
            errors.append('Saved certificate labels differ from recomputed raw audit')
        if not derived['reliable']:
            errors.append('Saved evidence does not establish reliable classification')
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        errors.append('Cannot recompute saved physical evidence: ' + repr(exc))
    return {'status': 'RECOMPUTED' if not errors else 'UNKNOWN', 'errors': errors,
            'new_physical_calls': 0}


def _node_map(raw):
    names, values = raw['node_names'], raw['node_vmag_pu']
    if (not isinstance(names, list) or not names or len(names) != len(values)
            or not all(isinstance(n, str) and n for n in names)
            or not all(_finite(v) for v in values)):
        raise ValueError('Missing or nonfinite node voltage array')
    result = dict(zip((n.lower() for n in names), values))
    if len(result) != len(names):
        raise ValueError('Duplicate nodes')
    return result


def _element_map(raw):
    items = raw['elements']
    result = {e['name'].lower(): e for e in items}
    if not items or len(result) != len(items):
        raise ValueError('Empty or duplicate physical inventory')
    return result


def compare_saved_pair(left, right, tolerances):
    """Compare all saved enabled terminal magnitudes/P/Q with exact layouts."""
    maxima = {'voltage_absolute_pu': 0., 'current_absolute_A': 0., 'power_absolute_kw': 0.}
    differences, unknown = [], []
    try:
        for key in maxima:
            if not _finite(tolerances[key]) or tolerances[key] < 0:
                raise ValueError('Invalid comparison tolerance: ' + key)
        a, b = _node_map(left), _node_map(right)
        if a.keys() != b.keys():
            raise ValueError('Node inventories differ')
        maxima['voltage_absolute_pu'] = max(abs(a[n] - b[n]) for n in a)
        a, b = _element_map(left), _element_map(right)
        if a.keys() != b.keys():
            raise ValueError('Physical element inventories differ')
        for name, ea in a.items():
            eb = b[name]
            if type(ea.get('enabled')) is not bool or type(eb.get('enabled')) is not bool:
                raise ValueError('Missing enabled state: ' + name)
            if ea['enabled'] != eb['enabled']:
                raise ValueError('Enabled states differ: ' + name)
            if not ea['enabled']:
                continue  # Disabled zero-current evidence belongs to isolation audit.
            for field in ('buses', 'conductors', 'terminals', 'phases', 'node_order'):
                if field not in ea or ea[field] != eb.get(field):
                    raise ValueError('Terminal layout differs: ' + name + ':' + field)
            size = ea['conductors'] * ea['terminals'] * 2
            for field, key, stride in [('currents_mag_angle', 'current_absolute_A', 2),
                                        ('powers_kw_kvar', 'power_absolute_kw', 1)]:
                va, vb = ea.get(field), eb.get(field)
                if (not isinstance(va, list) or not isinstance(vb, list)
                        or len(va) != size or len(vb) != size
                        or not all(_finite(x) for x in va + vb)):
                    raise ValueError('Malformed terminal evidence: ' + name + ':' + field)
                maxima[key] = max(maxima[key], max(abs(x-y) for x, y in zip(va[::stride], vb[::stride])))
        differences = [k for k, value in maxima.items() if value > tolerances[k]]
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        unknown.append(str(exc))
    return {'status': 'UNKNOWN' if unknown else 'DISAGREEMENT' if differences else 'AGREE',
            'max_absolute_differences': maxima, 'exceeded_tolerances': differences, 'unknown': unknown}


def audit_saved_acceptance(schedule, certificates, tolerances):
    """Certificates keyed by exact request hash; retains all eight planned pairs."""
    if len(schedule) != 16 or [r['order'] for r in schedule] != list(range(16)):
        raise ValueError('Exactly the ordered 16-entry schedule is required')
    hashes = [r['request']['request_sha256'] for r in schedule]
    if len(set(hashes)) != 16 or any(request_hash(r['request']) != h for r, h in zip(schedule, hashes)):
        raise ValueError('Schedule request hashes invalid')
    if set(certificates) - set(hashes):
        raise ValueError('Unexpected saved certificate')
    pairs = []
    for i in range(0, 16, 2):
        rows = schedule[i:i+2]
        if [r['algorithm'] for r in rows] != [0, 1] or rows[0]['label'] != rows[1]['label']:
            raise ValueError('Normal/Newton pair order differs')
        reqs = [r['request'] for r in rows]
        configs = [r['condition'].get('operating_config', {}) for r in reqs]
        for row, config in zip(rows, configs):
            if type(config.get('algorithm')) is not int or config['algorithm'] != row['algorithm']:
                raise ValueError('Request algorithm differs from schedule')
        settings = [{k: v for k, v in c.items() if k not in ('algorithm', 'algorithm_name')} for c in configs]
        if settings[0] != settings[1]:
            raise ValueError('Non-algorithm operating settings differ between pair')
        # Verify physical inputs agree even though condition/context hashes include algorithm.
        for key in ('loads', 'fault_lines', 'effective_action', 'time_seconds', 'model_fingerprint', 'event'):
            if reqs[0][key] != reqs[1][key]:
                raise ValueError('Different electrical inputs in pair: ' + key)
        for req in reqs:
            if req['condition']['der_setpoints'] != reqs[0]['condition']['der_setpoints']:
                raise ValueError('Different DER setpoints in pair')
        certs = [certificates.get(r['request_sha256']) for r in reqs]
        errors = [certificate_errors(r, c) for r, c in zip(reqs, certs)]
        for row, cert, err in zip(rows, certs, errors):
            if isinstance(cert, dict):
                observed = cert.get('operating_readback', {}).get('algorithm')
                raw_algorithm = (cert.get('raw_outputs') or {}).get('algorithm')
                if (type(observed) is not int or observed != row['algorithm']
                        or type(raw_algorithm) is not int or raw_algorithm != row['algorithm']):
                    err.append('Missing or different actual solver algorithm')
        row = {'orders': [i, i+1], 'label': rows[0]['label'], 'certificate_errors': errors}
        if any(errors):
            row.update(status='UNKNOWN', reason='Missing or unreliable paired evidence')
        else:
            row.update(compare_saved_pair(certs[0].get('raw_outputs'), certs[1].get('raw_outputs'), tolerances))
            if certs[0]['checks'] != certs[1]['checks']:
                row.update(status='DISAGREEMENT', reason='Constraint labels differ')
        pairs.append(row)
    return {'status': 'PAIRED_AGREEMENT' if all(r['status'] == 'AGREE' for r in pairs) else 'NOT_ACCEPTED',
            'pairs': pairs, 'new_physical_calls': 0, 'independent_solver_validation': False,
            'complete_restoration_sequence_certified': False}
