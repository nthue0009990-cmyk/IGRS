"""Pure accounting for the separately named NC34 author-daily load extension.

This module does not change or invoke the validated physical runtime. Actual
setpoint application and physical numerical validation remain separate gates.
"""
from __future__ import annotations

import math


def _finite_nonnegative(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('Invalid ' + label)
    return float(value)


def _same(a, b, label):
    if not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-8):
        raise ValueError(label + ' differs')


def validate_condition(condition, reference):
    """Bind all 68 P/Q requests and their denominator to immutable references."""
    multiplier = _finite_nonnegative(condition['load_multiplier'], 'multiplier')
    if multiplier <= 0:
        raise ValueError('A positive demand denominator is required')
    if len(reference) != 68 or len(condition['loads']) != 68:
        raise ValueError('All 68 loads are required')
    references = {r['name']: r for r in reference}
    rows = {r['name']: r for r in condition['loads']}
    if len(references) != 68 or len(rows) != 68 or references.keys() != rows.keys():
        raise ValueError('Load inventory differs or has duplicates')
    p_total, q_total, result = 0., 0., {}
    for name, ref in references.items():
        row = rows[name]
        if row['bus'] != ref['bus']:
            raise ValueError('Load bus differs: ' + name)
        p0 = _finite_nonnegative(ref['nominal_kw'], 'reference P')
        q0 = _finite_nonnegative(ref['nominal_kvar'], 'reference Q')
        for key, expected in [('reference_p_kw', p0), ('reference_q_kvar', q0),
                              ('demand_nominal_kw', p0 * multiplier), ('demand_nominal_kvar', q0 * multiplier)]:
            _same(_finite_nonnegative(row[key], key), expected, name + ':' + key)
        p_total += row['demand_nominal_kw']
        q_total += row['demand_nominal_kvar']
        result[name] = {'bus': ref['bus'], 'nominal_kw': row['demand_nominal_kw'],
                        'nominal_kvar': row['demand_nominal_kvar']}
    _same(sum(r['nominal_kw'] for r in reference), 1769., 'Reference P')
    _same(sum(r['nominal_kvar'] for r in reference), 1044., 'Reference Q')
    _same(p_total, _finite_nonnegative(condition['denominator_nominal_kw'], 'denominator'), 'Demand denominator')
    _same(q_total, _finite_nonnegative(condition['total_demand_kvar'], 'total Q'), 'Total demand Q')
    return result


def nominal_ledger(condition, reference, planned_loads, deenergized_buses):
    """Classify full-inventory demand without dropping disconnected loads."""
    loads = validate_condition(condition, reference)
    if loads.keys() != planned_loads.keys() or any(type(v) is not bool for v in planned_loads.values()):
        raise ValueError('Exactly one Boolean enable state per load is required')
    dead = set(deenergized_buses)
    ledger = {'condition_id': condition['condition_id'],
              'denominator_nominal_kw': condition['denominator_nominal_kw'],
              'total_demand_kvar': condition['total_demand_kvar'],
              'energized_nominal_kw': 0., 'deenergized_nominal_kw': 0.,
              'deliberately_shed_nominal_kw': 0., 'loads': [],
              'accounting_only': True, 'physical_witness_established': False}
    for name, ref in loads.items():
        enabled, disconnected = planned_loads[name], ref['bus'] in dead
        if enabled and disconnected:
            raise ValueError('Enabled load in deenergized component: ' + name)
        state = 'energized' if enabled else 'deenergized' if disconnected else 'deliberately_shed'
        ledger[state + '_nominal_kw'] += ref['nominal_kw']
        ledger['loads'].append({'name': name, **ref, 'state': state})
    _same(sum(ledger[k + '_nominal_kw'] for k in ['energized', 'deenergized', 'deliberately_shed']),
          ledger['denominator_nominal_kw'], 'Demand partition closure')
    ledger['nominal_restored_fraction'] = ledger['energized_nominal_kw'] / ledger['denominator_nominal_kw']
    return ledger


def verify_setpoint_readback(condition, reference, observed):
    """Check actual P/Q property readbacks; this does not certify an AC solution."""
    expected = validate_condition(condition, reference)
    if expected.keys() != observed.keys():
        raise ValueError('Full readback inventory required, including disabled loads')
    for name, row in expected.items():
        for key in ['nominal_kw', 'nominal_kvar']:
            _same(_finite_nonnegative(observed[name][key], 'readback ' + key), row[key], name + ':' + key)
    return {'status': 'SETPOINT_PROPERTIES_MATCH', 'checked_loads': 68,
            'ac_solution_certified': False, 'recovery_witness_established': False}
