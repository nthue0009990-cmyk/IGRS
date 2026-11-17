"""Nominal-phasor device linearization for an explicitly approximate MILP.

No simulator/optimizer is imported. Node and coil squared voltages are each
normalized by their own frozen nominal phasor magnitudes. Positive complex
power means consumption. Original AC validation remains the authority.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import shutil

from .nature34_two_stage_device_mapping import (
    digest, generator_voltage_factor, load_voltage_factors, read, require, sha,
)

VERSION = 'nc34-nominal-coil-affine-device-proxy-v2'
PHASE_DEGREES = {1: 0., 2: -120., 3: 120.}


def nominal_node(bus, phase, metadata):
    """Ground is exactly zero, not an independently energized voltage node."""
    bus = bus.split('.')[0].lower()
    if phase == 0:
        return {'node': None, 'bus': bus, 'phase': 0, 'volts': [0., 0.], 'base_volts': 0.}
    require(phase in PHASE_DEGREES, 'unsupported phase label')
    # Despite its legacy name, this frozen field is indexed by BUS, not node.
    bases = metadata['node_base_kv_ln']
    require(bus in bases, 'missing frozen nominal bus voltage: ' + bus)
    require(phase in metadata['physical_phase_nodes_by_bus'][bus], 'phase missing from frozen bus: ' + bus)
    magnitude = float(bases[bus]) * 1000.
    require(math.isfinite(magnitude) and magnitude > 0, 'invalid nominal node base')
    angle = PHASE_DEGREES[phase] + (30. if bus == 'sourcebus' else 0.)
    phasor = magnitude * complex(math.cos(math.radians(angle)), math.sin(math.radians(angle)))
    return {'node': f'{bus}.{phase}', 'bus': bus, 'phase': phase,
            'volts': [phasor.real, phasor.imag], 'base_volts': magnitude, 'angle_degrees': angle}


def coil_geometry(bus1, node1, bus2, node2, metadata):
    """First-order |V1-V2|² and frozen complex-power split at nominal phasors."""
    first, second = nominal_node(bus1, node1, metadata), nominal_node(bus2, node2, metadata)
    v1, v2 = complex(*first['volts']), complex(*second['volts'])
    difference = v1 - v2
    base2 = abs(difference) ** 2
    require(math.isfinite(base2) and base2 > 0, 'zero/invalid nominal coil voltage')
    weights, allocation = {}, []
    for endpoint, signed_voltage in ((first, v1), (second, -v2)):
        if endpoint['node'] is None:
            continue
        node = endpoint['node']
        # A single energized endpoint against ground is exactly its node u,
        # not merely a floating approximation to the value one.
        coefficient = 1.+0j if first['node'] is None or second['node'] is None else signed_voltage / difference
        weight = coefficient.real
        weights[node] = weights.get(node, 0.) + weight
        allocation.append({'node': node, 're': coefficient.real, 'im': coefficient.imag,
            'p_from_p': coefficient.real, 'p_from_q': -coefficient.imag,
            'q_from_p': coefficient.imag, 'q_from_q': coefficient.real})
    # Keep the small float remainder; this enforces nominal u_coil=1 exactly
    # in the same summation convention instead of silently rounding coefficients.
    constant = 1. - math.fsum(weights.values())
    return {'terminals': [first, second], 'nominal_difference_volts': [difference.real, difference.imag],
            'coil_nominal_volts': abs(difference), 'coil_nominal_squared_volts': base2,
            'u_coil': {'constant': constant, 'node_weights': weights,
                       'basis': '|V1-V2|^2 / |V1_nom-V2_nom|^2'},
            'phase_power': allocation,
            'power_conservation': {'sum_re': math.fsum(r['re'] for r in allocation),
                                   'sum_im': math.fsum(r['im'] for r in allocation)}}


def load_factor_derivatives(model, rho, *, vlow=.5, vmin=.85, vmax=1.05,
                            cvr_watts=1., cvr_vars=2.):
    """Analytic derivative of the exact pinned load law at a non-kink rho."""
    require(math.isfinite(rho) and rho > 0, 'invalid nominal load ratio')
    require(model in (1, 2, 4, 5), 'unsupported load law')
    if model == 2:
        return 2. * rho, 2. * rho, 'constant_impedance'
    require(rho not in (vlow, vmin, vmax), 'nominal point exactly at a load-law boundary: specify one-sided proxy explicitly')
    if rho < vlow:
        return 2. * rho, 2. * rho, 'low_constant_impedance'
    if rho < vmin:
        upper_current = 1. if model == 5 else 1. / vmin
        slope = (upper_current - vlow) / (vmin - vlow)
        derivative = vlow + slope * (2. * rho - vlow)
        return derivative, derivative, 'low_current_interpolation'
    if rho > vmax:
        derivative = 2. * rho / (vmax if model == 5 else vmax * vmax)
        return derivative, derivative, 'high_constant_impedance_fallback'
    if model == 1:
        return 0., 0., 'normal_constant_pq'
    if model == 4:
        return (cvr_watts * rho ** (cvr_watts - 1.),
                cvr_vars * rho ** (cvr_vars - 1.), 'normal_cvr')
    return 1., 1., 'normal_constant_current'


def factor_affine(model, rho0, law):
    kwargs = {key: law[key] for key in ('vlow', 'vmin', 'vmax', 'cvr_watts', 'cvr_vars')}
    p0, q0 = load_voltage_factors(model, rho0, **kwargs)
    dp, dq, region = load_factor_derivatives(model, rho0, **kwargs)
    bp, bq = dp * rho0 / 2., dq * rho0 / 2.
    return {'p': {'a': p0 - bp, 'b': bp}, 'q': {'a': q0 - bq, 'b': bq},
            'rho0': rho0, 'linearization_region': region, 'nominal_factors': [p0, q0]}


def scaled(coefficients, nominal):
    return {key: nominal * coefficients[key] for key in ('a', 'b')}


def evaluate_u(geometry, node_u):
    return geometry['u_coil']['constant'] + math.fsum(
        coefficient * node_u[node] for node, coefficient in geometry['u_coil']['node_weights'].items())


def phase_power(geometry, p, q):
    return {r['node']: {'p_kw': r['re'] * p - r['im'] * q,
                        'q_kvar': r['im'] * p + r['re'] * q}
            for r in geometry['phase_power']}


def _error_points(geometry, rho0, law):
    """Fixed physical 0.9/1.1 node voltage corners plus crossed law boundaries."""
    terms = geometry['terminals']
    names = [x['node'] for x in terms if x['node'] is not None]
    result = [('nominal', {node: 1. for node in names})]
    for values in itertools.product((.9, 1.1), repeat=len(names)):
        result.append(('voltage_corner', dict(zip(names, values))))
    for threshold in (law.get('vlow'), law.get('vmin'), law.get('vmax')):
        if threshold is None:
            continue
        amplitude = threshold / rho0
        if .9 <= amplitude <= 1.1:
            for label, a in [('boundary_below', math.nextafter(amplitude, 0.)),
                             ('boundary_exact', amplitude),
                             ('boundary_above', math.nextafter(amplitude, math.inf))]:
                result.append((label, {node: a for node in names}))
    return result


def error_diagnostics(record, factor_function, law):
    """Descriptive evaluations at supplied voltages, never a circuit solution."""
    geometry = record['geometry']
    ratio0 = record['rho0']
    rows = []
    for label, amplitudes in _error_points(geometry, ratio0, law):
        voltages = [complex(*t['volts']) * (amplitudes[t['node']] if t['node'] else 1.)
                    for t in geometry['terminals']]
        exact_u = abs(voltages[0] - voltages[1]) ** 2 / geometry['coil_nominal_squared_volts']
        linear_u = evaluate_u(geometry, {node: value ** 2 for node, value in amplitudes.items()})
        rho = ratio0 * math.sqrt(exact_u)
        pf, qf = factor_function(rho)
        actual = [record['nominal_coil_p_kw'] * pf, record['nominal_coil_q_kvar'] * qf]
        predicted = [record['p_affine_kw']['a'] + record['p_affine_kw']['b'] * linear_u,
                     record['q_affine_kvar']['a'] + record['q_affine_kvar']['b'] * linear_u]
        rows.append({'kind': label, 'node_voltage_multipliers': amplitudes, 'rho': rho,
                     'exact_coil_u': exact_u, 'affine_coil_u': linear_u,
                     'coil_u_error': linear_u - exact_u, 'exact_p_kw': actual[0], 'exact_q_kvar': actual[1],
                     'proxy_p_kw': predicted[0], 'proxy_q_kvar': predicted[1],
                     'p_error_kw': predicted[0] - actual[0], 'q_error_kvar': predicted[1] - actual[1]})
    return {'selection': 'nominal, fixed node-amplitude corners 0.9/1.1, uniform-scaling crossed law thresholds and adjacent floats',
            'rows': rows, 'max_abs_p_error_kw': max(abs(r['p_error_kw']) for r in rows),
            'max_abs_q_error_kvar': max(abs(r['q_error_kvar']) for r in rows),
            'max_abs_coil_u_error': max(abs(r['coil_u_error']) for r in rows),
            'claim': 'finite descriptive checks, not a uniform approximation bound or AC validation'}


def _record(name, kind, index, geometry, base_volts, nominal_p, nominal_q, factors, scaling):
    return {'device': name, 'kind': kind, 'coil_index': index, 'geometry': geometry,
            'load_or_device_rated_coil_volts': base_volts,
            'rho0': geometry['coil_nominal_volts'] / base_volts,
            'nominal_coil_p_kw': nominal_p, 'nominal_coil_q_kvar': nominal_q,
            'p_affine_kw': scaled(factors['p'], nominal_p), 'q_affine_kvar': scaled(factors['q'], nominal_q),
            'factor_affine': factors, 'multiplication_policy': scaling,
            'sign_convention': 'positive consumption; generator and capacitor injection negative'}


def build_load_proxy(device_mapping, metadata):
    require(device_mapping['version'] == 'nc34-static-device-mapping-v2', 'unsupported mapping version (v1 dispatch metadata must not be reused)')
    records, devices = [], []
    for load in device_mapping['loads']:
        name, bus = load['name'], load['buses'][0].split('.')[0]
        device = {'name': name, 'kind': 'load', 'dispatch_bit': load['dispatch_bit'],
                  'dispatch_bit_basis': load['dispatch_bit_basis'],
                  'physical_action_index': load['physical_action_index'], 'coils': []}
        for i, pair in enumerate(load['coil_nodes']):
            geometry = coil_geometry(bus, pair[0], bus, pair[1], metadata)
            rho0 = geometry['coil_nominal_volts'] / load['coil_base_volts']
            factors = factor_affine(load['model'], rho0, load['voltage_law'])
            row = _record(name, 'load', i, geometry, load['coil_base_volts'],
                          load['nominal_kw'] / load['phases'], load['nominal_kvar'] / load['phases'],
                          factors, 'multiply both P and Q coefficients by original condition multiplier and original per-load CLPU factor; energization separately gates the whole load')
            row['model'] = load['model']
            kwargs = {key: load['voltage_law'][key] for key in ('vlow', 'vmin', 'vmax', 'cvr_watts', 'cvr_vars')}
            row['error_diagnostics'] = error_diagnostics(row, lambda rho, model=load['model'], kw=kwargs: load_voltage_factors(model, rho, **kw), load['voltage_law'])
            records.append(row)
            device['coils'].append(len(records) - 1)
        devices.append(device)
    for gen in device_mapping['generators']:
        declaration = gen['declaration']
        phases, bus = declaration['numphase'], declaration['bus']
        base_volts = declaration['kV'] * 1000 / (math.sqrt(3) if phases in (2, 3) else 1.)
        device = {'name': gen['name'], 'kind': 'generator', 'coils': []}
        for i in range(phases):
            geometry = coil_geometry(bus, gen['node_order'][i], bus, gen['node_order'][-1], metadata)
            rho0 = geometry['coil_nominal_volts'] / base_volts
            require(rho0 not in (.9, 1.1), 'nominal generator at boundary')
            value = generator_voltage_factor(rho0)
            slope = value if rho0 < .9 or rho0 > 1.1 else 0.
            factors = {'p': {'a': value - slope, 'b': slope}, 'q': {'a': value - slope, 'b': slope},
                       'linearization_region': 'fixed_pq' if .9 < rho0 < 1.1 else 'impedance_fallback'}
            row = _record(gen['name'], 'generator', i, geometry, base_volts,
                          -gen['p_setpoint_kw'] / phases, -gen['q_setpoint_kvar'] / phases,
                          factors, 'fixed original alpha already included; no condition or CLPU scaling; enabled only in sourcebus component')
            row['error_diagnostics'] = error_diagnostics(row, lambda rho: (generator_voltage_factor(rho),) * 2,
                                                         {'vmin': .9, 'vmax': 1.1})
            records.append(row)
            device['coils'].append(len(records) - 1)
        devices.append(device)
    for cap in device_mapping['capacitors']:
        phases = cap['phases']
        base_volts = cap['kv_line_line'] * 1000 / math.sqrt(3)
        device = {'name': cap['name'], 'kind': 'capacitor', 'coils': []}
        for i in range(phases):
            geometry = coil_geometry(cap['buses'][0], cap['node_order'][i],
                                     cap['buses'][1], cap['node_order'][i + phases], metadata)
            rho0 = geometry['coil_nominal_volts'] / base_volts
            factors = {'p': {'a': 0., 'b': 0.}, 'q': {'a': 0., 'b': rho0 ** 2},
                       'linearization_region': 'exact_admittance_in_coil_squared_voltage'}
            row = _record(cap['name'], 'capacitor', i, geometry, base_volts, 0.,
                          -cap['nominal_kvar'] / phases, factors,
                          'fixed enabled shunt, no condition or CLPU scaling; no independently optimized switching')
            row['error_diagnostics'] = error_diagnostics(row, lambda rho: (0., rho ** 2), {})
            records.append(row)
            device['coils'].append(len(records) - 1)
        devices.append(device)
    return {'version': VERSION, 'status': 'AFFINE_DEVICE_COEFFICIENTS_BUILT_NOT_AC_EQUIVALENT',
            'source_device_mapping_digest': digest(device_mapping), 'source_metadata_digest': digest(metadata),
            'source_run': device_mapping['source_run'],
            'source_completion_manifest_sha256': device_mapping['source_completion_manifest_sha256'],
            'node_u_basis': '|Vnode|^2 / frozen metadata.node_base_kv_ln[bus]^2 / 10^6',
            'coil_u_basis': '|V1-V2|^2 / |V1_nom-V2_nom|^2; distinct from load rated Vbase',
            'phase_reference_degrees': {'downstream': PHASE_DEGREES, 'sourcebus_offset': 30},
            'power_units': 'kW,kvar; positive consumption', 'devices': devices, 'coils': records,
            'regcontrols': device_mapping['regcontrols'],
            'physical_voltage_limits': {'source': 'original physical_config supplied separately to electrical model',
                                        'values_duplicated_here': False},
            'counts': {'devices': len(devices), 'coils': len(records),
                       'low_interpolation_load_coils': sum(r['kind'] == 'load' and r['factor_affine']['linearization_region'] == 'low_current_interpolation' for r in records),
                       'load_models': device_mapping['counts']['load_models']},
            'max_descriptive_error_kw': max(r['error_diagnostics']['max_abs_p_error_kw'] for r in records),
            'max_descriptive_error_kvar': max(r['error_diagnostics']['max_abs_q_error_kvar'] for r in records),
            'new_physical_points': 0, 'new_training': 0,
            'limitations': ['Fixed nominal angle and power-allocation approximation; phase angles are not solved here.',
                           'Exact at nominal point to floating arithmetic; first-order away from it. Threshold crossings retain reported error, not silently refitted.',
                           'This device layer does not implement line/transformer loss, regulator policy, topology or AC certification.',
                           'Finite endpoint diagnostics are not uniform error guarantees. Original physical constraints and AC validation remain unchanged.']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device-mapping', required=True)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    mapping, metadata = read(args.device_mapping), read(args.metadata)
    require(sha(args.metadata) == mapping['source_files']['metadata.json'], 'metadata differs from mapped source')
    out = Path(args.output)
    require(not out.exists(), 'output must be new')
    result = build_load_proxy(mapping, metadata)
    result['source_files'] = {str(Path(args.device_mapping)): sha(args.device_mapping),
                              str(Path(args.metadata)): sha(args.metadata), str(Path(__file__)): sha(__file__)}
    out.mkdir(parents=True)
    (out / 'proxy.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    shutil.copyfile(__file__, out / Path(__file__).name)
    (out / 'completion_manifest.json').write_text(json.dumps({p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'counts': result['counts'], 'seal': sha(out / 'completion_manifest.json')}))


if __name__ == '__main__':
    main()
