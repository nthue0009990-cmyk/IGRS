"""Lossless multiphase electrical rows for the declared two-stage MILP proxy.

This builds affine/integer constraints; it does not run an optimizer or a PF.
The original complete AC certifier is the sole physical acceptance authority.
"""
from __future__ import annotations

import math

from .nature34_two_stage_linear import Expr, expression
from .nature34_two_stage_load_proxy import coil_geometry, nominal_node


VERSION = 'nc34-two-stage-lossless-electrical-proxy-v1'
HEX_VERTEX_RATIO = math.sqrt((2. * math.pi / 6.) / math.sin(2. * math.pi / 6.))
HEX_APOTHEM_RATIO = HEX_VERTEX_RATIO * math.cos(math.pi / 6.)


def _records(value):
    """No solver handles in the callback evidence; indices refer to LinModel."""
    if isinstance(value, Expr):
        return {'terms': sorted(value.terms.items()), 'constant': value.constant}
    if isinstance(value, dict):
        return {k: _records(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_records(v) for v in value]
    return value


def _name(bus, phase):
    return f'{bus.lower()}.{phase}'


def _affine_u(geometry, node_u):
    part = geometry['u_coil']
    return expression(part['constant']) + sum((v * node_u[n] for n, v in part['node_weights'].items()), expression(0))


def _allocation(geometry, p, q):
    out = {}
    for row in geometry['phase_power']:
        node = row['node']
        a, b = out.get(node, (expression(0), expression(0)))
        out[node] = (a + row['re'] * p - row['im'] * q,
                     b + row['im'] * p + row['re'] * q)
    return out


def add_capacity_hexagon(model, p, q, enabled, rated_kva, name):
    """Paper's six-sided equal-area approximation, NOT a circle inner bound."""
    if not math.isfinite(rated_kva) or rated_kva <= 0:
        raise ValueError('A positive sourced rating is required')
    radius = rated_kva * HEX_VERTEX_RATIO
    apothem = rated_kva * HEX_APOTHEM_RATIO
    for side in range(6):
        angle = math.pi / 6. + side * math.pi / 3.
        model.add(math.cos(angle) * p + math.sin(angle) * q - apothem * enabled,
                  ub=0, name=f'{name}:hex:{side}')
    return {'name': name, 'rated_kva': rated_kva, 'vertex_kva': radius, 'apothem_kva': apothem,
            'maximum_radial_overreach_fraction': HEX_VERTEX_RATIO - 1.,
            'maximum_radial_underreach_fraction': 1. - HEX_APOTHEM_RATIO,
            'source': 'Poudel-Dubey author preprint 2004.07921v1, equation 12, n=6 equal-area hexagon',
            'physical_certificate': False}


def add_regulator_control(model, *, enabled, tap, u_out, p_out, q_out,
                          nominal_coil_volts, control, name):
    """First-order LDC voltage and original deadband/saturation equilibrium set.

    P/Q are positive OUT of controlled winding 2; original current INTO that
    winding has the opposite sign.  This is a set of possible steady states,
    not an emulation of the actual original control queue or its chosen tap.
    """
    if control['winding'] != 2 or control['tap_winding'] != 2:
        raise ValueError('Only original winding-2 regulator mapping is supported')
    defaults = control['source_defaults']
    if defaults['reversible'] or defaults['cogen'] or defaults['pt_phase'] != 1:
        raise ValueError('Regulator mode differs from the named source')
    if (defaults['mintap'], defaults['maxtap'], defaults['numtaps']) != (.9, 1.1, 32):
        raise ValueError('Regulator tap range differs from frozen source')
    pt, ct = control['pt_ratio'], control['ct_primary_amperes']
    if nominal_coil_volts <= 0 or pt <= 0 or ct <= 0 or control['band_volts'] <= 0:
        raise ValueError('Invalid regulator scale or deadband')
    vc = (nominal_coil_volts / pt) * (1. + .5 * (u_out - 1.)) - (
        control['ldc_r_volts'] * p_out + control['ldc_x_volts'] * q_out) * (1000. / (ct * nominal_coil_volts))
    lo = control['vreg_volts'] - .5 * control['band_volts']
    hi = control['vreg_volts'] + .5 * control['band_volts']
    low = model.var(name + ':at_min', 0, 1, integer=True)
    interior = model.var(name + ':interior', 0, 1, integer=True)
    high = model.var(name + ':at_max', 0, 1, integer=True)
    model.add(low + interior + high - enabled, lb=0, ub=0, name=name + ':regions')
    model.add(tap + 16 * enabled, lb=0, name=name + ':tap_lower_enabled')
    model.add(tap - 16 * enabled, ub=0, name=name + ':tap_upper_enabled')
    model.implies(low, tap, lb=-16, ub=-16, name=name + ':minimum_tap')
    model.implies(interior, tap, lb=-15, ub=15, name=name + ':interior_tap')
    model.implies(high, tap, lb=16, ub=16, name=name + ':maximum_tap')
    model.implies(low, vc, lb=lo, name=name + ':high_voltage_saturation')
    model.implies(interior, vc, lb=lo, ub=hi, name=name + ':deadband')
    model.implies(high, vc, ub=hi, name=name + ':low_voltage_saturation')
    return {'tap_index': tap, 'vc_affine_volts': vc, 'lower_volts': lo, 'upper_volts': hi,
            'minimum_region': low, 'interior_region': interior, 'maximum_region': high,
            'source_control': control['name'],
            'semantics': 'per-snapshot LDC steady-state set with saturation; no temporal tap memory; not actual queue selection'}


def _source_pu(network):
    circuits = network['circuit_declarations']
    if len(circuits) != 1:
        raise ValueError('One explicit source circuit required')
    props = {r['key']: r['value'] for r in circuits[0]['properties']}
    if 'pu' not in props:
        raise ValueError('Explicit source voltage required')
    value = float(props['pu'])
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Invalid source voltage')
    return value


def _device_demands(model, devices, load_demands, bus_on, node_u, condition, tag):
    per_node = {n: (expression(0), expression(0)) for n in node_u}
    coils = []
    demand_keys = {str(k).lower(): v for k, v in load_demands.items()}
    loads = {r['device'].lower() for r in devices['coils'] if r['kind'] == 'load'}
    if len(demand_keys) != len(load_demands) or set(demand_keys) != loads:
        raise ValueError('Explicit unambiguous demand history for every whole load required')
    for index, coil in enumerate(devices['coils']):
        geometry = coil['geometry']
        u = _affine_u(geometry, node_u)
        p = coil['p_affine_kw']['a'] + coil['p_affine_kw']['b'] * u
        q = coil['q_affine_kvar']['a'] + coil['q_affine_kvar']['b'] * u
        bus = geometry['terminals'][0]['bus']
        name = f'{tag}:device:{index}'
        if coil['kind'] == 'load':
            terms = demand_keys[coil['device'].lower()]
            if not isinstance(terms, list):
                raise ValueError('A finite explicit pickup list is required')
            pp, qq, energized = expression(0), expression(0), expression(0)
            for k, item in enumerate(terms):
                coefficient = float(item['coefficient'])
                if not math.isfinite(coefficient) or coefficient <= 0:
                    raise ValueError('Positive original CLPU multiplier required')
                binary = model._binary(item['pickup'])
                pp += condition * coefficient * model.product(binary, p, name + f':P:{k}')
                qq += condition * coefficient * model.product(binary, q, name + f':Q:{k}')
                energized += binary
            model.add(energized - bus_on[bus], ub=0, name=name + ':load_requires_energized_bus')
            p, q = pp, qq
        elif coil['kind'] in ('generator', 'capacitor'):
            p = model.product(bus_on[bus], p, name + ':P:enabled')
            q = model.product(bus_on[bus], q, name + ':Q:enabled')
        else:
            raise ValueError('Unsupported proxy device kind')
        allocation = _allocation(geometry, p, q)
        for node, (pn, qn) in allocation.items():
            priorp, priorq = per_node[node]
            per_node[node] = (priorp + pn, priorq + qn)
        coils.append({'device': coil['device'], 'coil_index': coil['coil_index'],
                      'p_kw': p, 'q_kvar': q, 'u_coil': u, 'phase_consumption': allocation})
    return per_node, coils


def _line_shunt(model, line, bus, node_u, enabled, metadata, tag):
    """First-order terminal power of original pi-shunt at fixed phase angles."""
    result = {}
    for i, phase in enumerate(line['phase_order']):
        ni = nominal_node(bus, phase, metadata)
        vi = complex(*ni['volts'])
        p, q = expression(0), expression(0)
        for j, other in enumerate(line['phase_order']):
            nj = nominal_node(bus, other, metadata)
            vj = complex(*nj['volts'])
            entry = line['pi_terminal_shunt_y_siemens'][i][j]
            admittance = complex(entry['re'], entry['im'])
            # S_i = V_i conjugate(Y_ij V_j), expanded in squared-node voltages.
            coefficient = vi * (admittance * vj).conjugate() / 1000.
            scale = .5 * (node_u[ni['node']] + node_u[nj['node']])
            p += coefficient.real * scale
            q += coefficient.imag * scale
        result[phase] = (model.product(enabled, p, f'{tag}:shunt:{bus}:{phase}:P'),
                         model.product(enabled, q, f'{tag}:shunt:{bus}:{phase}:Q'))
    return result


def add_electrical_snapshot(model, network, devices, bus_on, edge_on, load_demands, tag,
                            *, condition_multiplier, voltage_limits):
    """Add one actual multiphase linear snapshot; all gate variables are causal.

    ``edge_on`` includes every line AND transformer, by full lowercase name.
    ``load_demands[name]`` is a list of {pickup: binary or 0/1, coefficient: CLPU}.
    Source control/physics values come from mapped inputs; no plan/result is read.
    """
    if network['status'] != 'STATIC_NETWORK_MAPPING_COMPLETE' or network['unsupported']:
        raise ValueError('Complete static network mapping required')
    if devices['version'] != 'nc34-nominal-coil-affine-device-proxy-v2':
        raise ValueError('Unsupported device coefficient version')
    condition = float(condition_multiplier)
    vmin, vmax = map(float, voltage_limits)
    if not (math.isfinite(condition) and math.isfinite(vmin) and math.isfinite(vmax)
            and condition > 0 and 0 < vmin < vmax):
        raise ValueError('Original finite condition and voltage limits required')
    phases_by_bus = network['buses']['physical_phase_nodes']
    bases = network['buses']['base_kv_ln_by_node']  # legacy field is indexed by BUS
    if set(bus_on) != set(phases_by_bus):
        raise ValueError('Complete original bus activation mapping required')
    names = {x['name'] for x in network['lines'] + network['transformers']}
    if set(edge_on) != names:
        raise ValueError('Every original line/transformer must have an explicit activation')
    bus_on = {k: model._binary(v) for k, v in bus_on.items()}
    edge_on = {k: model._binary(v) for k, v in edge_on.items()}
    metadata = {'node_base_kv_ln': bases, 'physical_phase_nodes_by_bus': phases_by_bus}
    start_vars, start_rows = len(model.names), len(model.rows)
    node_u, p_balance, q_balance = {}, {}, {}
    for bus, phases in phases_by_bus.items():
        for phase in phases:
            key = _name(bus, phase)
            u = model.var(f'{tag}:u:{key}', 0, vmax ** 2)
            model.add(u - vmin ** 2 * bus_on[bus], lb=0, name=f'{tag}:voltage_min:{key}')
            model.add(u - vmax ** 2 * bus_on[bus], ub=0, name=f'{tag}:voltage_max:{key}')
            node_u[key], p_balance[key], q_balance[key] = u, expression(0), expression(0)
    sourcepu = _source_pu(network)
    model.add(bus_on['sourcebus'], lb=1, ub=1, name=tag + ':source_on')
    for phase in phases_by_bus['sourcebus']:
        model.add(node_u[_name('sourcebus', phase)], lb=sourcepu ** 2, ub=sourcepu ** 2,
                  name=f'{tag}:source_voltage:{phase}')
    demands, device_coils = _device_demands(model, devices, load_demands, bus_on, node_u, condition, tag)
    for node, (p, q) in demands.items():
        p_balance[node] -= p
        q_balance[node] -= q
    flows, taps, thermal = {}, {}, []
    for line in network['lines']:
        name, on = line['name'], edge_on[line['name']]
        b1, b2 = line['bus1'], line['bus2']
        for b in (b1, b2):
            model.add(on - bus_on[b], ub=0, name=f'{tag}:{name}:requires:{b}')
        sh1 = _line_shunt(model, line, b1, node_u, on, metadata, f'{tag}:{name}:from')
        sh2 = _line_shunt(model, line, b2, node_u, on, metadata, f'{tag}:{name}:to')
        ps, qs = [], []
        terminals = [[], []]
        for phase in line['phase_order']:
            s1, s2 = line['phase_ampacity_a'] * bases[b1], line['phase_ampacity_a'] * bases[b2]
            # Series flow differs from terminal flow by each end's original shunt.
            extra_p = max(abs(x) for sh in (sh1[phase], sh2[phase]) for x in model.bounds(sh[0]))
            extra_q = max(abs(x) for sh in (sh1[phase], sh2[phase]) for x in model.bounds(sh[1]))
            pmax = max(s1, s2) * HEX_VERTEX_RATIO + extra_p
            qmax = max(s1, s2) * HEX_VERTEX_RATIO + extra_q
            p = model.var(f'{tag}:{name}:P:{phase}', -pmax, pmax)
            q = model.var(f'{tag}:{name}:Q:{phase}', -qmax, qmax)
            for variable, bound, suffix in ((p, pmax, 'P'), (q, qmax, 'Q')):
                model.add(variable - bound * on, ub=0, name=f'{tag}:{name}:{suffix}:on_hi:{phase}')
                model.add(variable + bound * on, lb=0, name=f'{tag}:{name}:{suffix}:on_lo:{phase}')
            ps.append(p); qs.append(q)
            fromp, fromq = p + sh1[phase][0], q + sh1[phase][1]
            top, toq = -p + sh2[phase][0], -q + sh2[phase][1]
            for bus, pp, qq, rated, terminal in ((b1, fromp, fromq, s1, 0), (b2, top, toq, s2, 1)):
                key = _name(bus, phase)
                p_balance[key] -= pp
                q_balance[key] -= qq
                terminals[terminal].append({'node': key, 'p_into_kw': pp, 'q_into_kvar': qq})
                thermal.append(add_capacity_hexagon(model, pp, qq, on, rated, f'{tag}:{name}:terminal{terminal}:{phase}'))
        for i, phase in enumerate(line['phase_order']):
            drop = sum((a * p + b * q for a, b, p, q in zip(
                line['linear_voltage_drop']['p_kv2_per_kw'][i],
                line['linear_voltage_drop']['q_kv2_per_kvar'][i], ps, qs)), expression(0))
            residual = bases[b2] ** 2 * node_u[_name(b2, phase)] - bases[b1] ** 2 * node_u[_name(b1, phase)] - drop
            model.implies(on, residual, lb=0, ub=0, name=f'{tag}:{name}:voltage_drop:{phase}')
        flows[name] = {'kind': 'line', 'phase_order': line['phase_order'], 'p_kw': ps, 'q_kvar': qs,
                       'terminals': terminals, 'shunts': [sh1, sh2], 'enabled': on}
    controls = {r['transformer'].lower(): r for r in devices['regcontrols']}
    for tr in network['transformers']:
        name, on = tr['name'], edge_on[tr['name']]
        # The named source uses the pinned backend's zero defaults.  An added
        # nonzero magnetizing/noload declaration cannot silently disappear.
        for prop in tr['properties']:
            if prop['key'] in ('%noloadloss', '%imag') and float(prop['value']) != 0:
                raise ValueError('Nonzero transformer shunt needs explicit proxy derivation')
        first, second = tr['windings']
        b1, b2 = first['terminal']['bus'], second['terminal']['bus']
        for b in (b1, b2):
            model.add(on - bus_on[b], ub=0, name=f'{tag}:{name}:requires:{b}')
        if second['conn'] != 'wye':
            raise ValueError('Named source has grounded-wye secondary; another connection needs new mapping')
        controlled = name in controls
        if controlled and (tr['phases'] != 1 or tr['regcontrols'] != [controls[name]['name']]):
            raise ValueError('Original one-phase regulator association differs')
        if not controlled and (first['tap'] != 1. or second['tap'] != 1.):
            raise ValueError('Uncontrolled nonunit declaration tap needs explicit proxy derivation')
        tap = model.var(f'{tag}:{name}:tap_index', -16, 16, integer=True) if controlled else expression(0)
        terminals = [{}, {}]
        coils = []
        for i in range(tr['phases']):
            pair1, pair2 = first['coil_node_pairs'][i], second['coil_node_pairs'][i]
            geometry1 = coil_geometry(b1, pair1[0], b1, pair1[1], metadata)
            geometry2 = coil_geometry(b2, pair2[0], b2, pair2[1], metadata)
            u1, u2 = _affine_u(geometry1, node_u), _affine_u(geometry2, node_u)
            bound = second['kva'] / tr['phases'] * HEX_VERTEX_RATIO
            p = model.var(f'{tag}:{name}:coilP:{i}', -bound, bound)
            q = model.var(f'{tag}:{name}:coilQ:{i}', -bound, bound)
            for variable, suffix in ((p, 'P'), (q, 'Q')):
                model.add(variable - bound * on, ub=0, name=f'{tag}:{name}:coil{suffix}:on_hi:{i}')
                model.add(variable + bound * on, lb=0, name=f'{tag}:{name}:coil{suffix}:on_lo:{i}')
            allocations = [_allocation(geometry1, p, q), _allocation(geometry2, -p, -q)]
            for terminal, allocation in enumerate(allocations):
                for node, (pp, qq) in allocation.items():
                    priorp, priorq = terminals[terminal].get(node, (expression(0), expression(0)))
                    terminals[terminal][node] = (priorp + pp, priorq + qq)
                    p_balance[node] -= pp
                    q_balance[node] -= qq
            n = tr['ideal_coil_voltage_ratio_1_over_2']
            base2_kv2 = geometry2['coil_nominal_squared_volts'] / 1e6
            ratio = geometry1['coil_nominal_squared_volts'] / (n * n * geometry2['coil_nominal_squared_volts'])
            z = tr['pair_leakage_on_winding2_coil_base_ohm']
            drop = -.002 / base2_kv2 * (z['re'] * p + z['im'] * q)
            residual = u2 - ratio * u1 - 2. * .00625 * tap - drop
            model.implies(on, residual, lb=0, ub=0, name=f'{tag}:{name}:coil_voltage_drop:{i}')
            if controlled:
                taps[name] = add_regulator_control(model, enabled=on, tap=tap, u_out=u2, p_out=p, q_out=q,
                    nominal_coil_volts=geometry2['coil_nominal_volts'], control=controls[name], name=f'{tag}:{name}:ldc')
            coils.append({'p_kw': p, 'q_kvar': q, 'u_primary': u1, 'u_secondary': u2,
                          'geometry_primary': geometry1, 'geometry_secondary': geometry2,
                          'voltage_ratio_coefficient': ratio, 'leakage_drop_pu2': drop})
        for terminal, (winding, allocation) in enumerate(zip(tr['windings'], terminals)):
            totalp = sum((p for p, q in allocation.values()), expression(0))
            totalq = sum((q for p, q in allocation.values()), expression(0))
            thermal.append(add_capacity_hexagon(model, totalp, totalq, on, winding['kva'], f'{tag}:{name}:total:{terminal}'))
            current_limit = winding['kva'] / (math.sqrt(3) * winding['kv']) if tr['phases'] == 3 else winding['kva'] / winding['kv']
            for node, (p, q) in allocation.items():
                rated = bases[node.rsplit('.', 1)[0]] * current_limit
                thermal.append(add_capacity_hexagon(model, p, q, on, rated, f'{tag}:{name}:terminalI:{terminal}:{node}'))
                if winding['conn'] == 'wye':
                    thermal.append(add_capacity_hexagon(model, p, q, on, winding['kva'] / tr['phases'],
                                                        f'{tag}:{name}:wye_phaseS:{terminal}:{node}'))
        flows[name] = {'kind': 'transformer', 'coils': coils, 'terminals': terminals, 'enabled': on}
    if set(controls) != {x['name'] for x in network['transformers'] if x['regcontrols']}:
        raise ValueError('A regulator declaration was omitted or added')
    slack = {}
    for node in node_u:
        if node.startswith('sourcebus.'):
            lo_p, hi_p = model.bounds(-p_balance[node])
            lo_q, hi_q = model.bounds(-q_balance[node])
            ps = model.var(f'{tag}:sourceP:{node}', lo_p, hi_p)
            qs = model.var(f'{tag}:sourceQ:{node}', lo_q, hi_q)
            p_balance[node] += ps; q_balance[node] += qs
            slack[node] = {'p_kw': ps, 'q_kvar': qs}
        model.add(p_balance[node], lb=0, ub=0, name=f'{tag}:P_balance:{node}')
        model.add(q_balance[node], lb=0, ub=0, name=f'{tag}:Q_balance:{node}')
    return _records({'version': VERSION, 'tag': tag, 'node_u': node_u, 'flows': flows, 'taps': taps,
            'device_coils': device_coils, 'source_injection': slack,
            'thermal_constraints': thermal,
            'constraint_coverage': {'phase_nodes': len(node_u), 'phase_power_balances': 2 * len(node_u),
                'lines': len(network['lines']), 'transformers': len(network['transformers']),
                'automatic_regulators': len(taps), 'capacity_polygons': len(thermal),
                'device_coils': len(device_coils), 'added_variables': len(model.names) - start_vars,
                'added_rows': len(model.rows) - start_rows},
            'capacity_approximation': {'kind': 'source_six_side_equal_area', 'vertex_ratio': HEX_VERTEX_RATIO,
                                       'apothem_ratio': HEX_APOTHEM_RATIO},
            'approximation_contract': [
                'No series power loss; fixed nominal phase angles and first-order coil/device allocation.',
                'Original line pi-shunt is retained with first-order node-power dependence and edge activation.',
                'Transformer coil connections, phase allocation, nominal ratios and series leakage are retained; no-load loss/magnetizing defaults are zero, numerical ppm-antifloat is omitted.',
                'Regulator relation keeps first-order 2*delta_tap and omits delta_tap² and delta_tap*(u-1); LDC linearizes voltage and current at nominal.',
                'Automatic tap is restricted by original deadband/saturation equilibrium, not a free voltage control; actual queue and reset trajectory are not reproduced.',
                'Line current and winding conductor limits use nominal-voltage apparent-power conversion plus the paper equal-area hexagon; this has both inner and outer circle error.',
                'Only original complete AC validation can certify voltage/current/thermal feasibility; linear feasibility is not a certificate.'],
            'new_physical_points': 0, 'physical_certificate': None})
