"""Static, source-bound NC34 device semantics; no DSS context or solver imports.

The voltage law functions evaluate algebra at supplied voltages; they do not
solve a circuit.  This is an input mapping and approximation contract, not an
implemented MILP, AC certificate, or permission to run an experiment.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

VERSION = 'nc34-static-device-mapping-v2'
SOURCE_ROOT_SHA = 'af9dc11957274cb84317e6f227fd94bf0f75df35f207667e26ff466d0678ac02'
DEFAULT_SOURCE = 'runs/20261031T103802_907721Z_clpu_guard_first_pilot'
OFFICIAL_DIR = 'data/raw/literature/two_stage_static_mapping_20261031/devices'
OFFICIAL_SHAS = {
    'Load.pas': 'a4e67bb32a4d394ffbc0d91d256582c86894d0f97d26d057fdc852733597b109',
    'generator.pas': '25d231b7560a51bbcf34c3cb4ccf3734168341dadb62a3517b3df5bb045cebe7',
    'RegControl.pas': '799f539669cfe70727f64d7e8fa9b96c4ba03a61860f176b62174e9cabdfe6ae',
    'Capacitor.pas': '51e11d383466090e9eb9bb11b7f6dc4c703272cc74ad4b530ba5362e91494fed',
    'Transformer.pas': '53148a652db204f033eae2422cee6be0fcd3357a035e4c5641342dd6d51c319a',
}
KINDS = ('load.', 'regcontrol.', 'capacitor.')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(test, message):
    if not test:
        raise ValueError(message)


def manifest(path):
    result = {}
    for key, value in read(path).items():
        key = key.replace('\\', '/')
        require(key not in result and '..' not in Path(key).parts, 'unsafe manifest alias')
        result[key] = value
    return result


def static_commands(text):
    """Extract source lines only.  Never send their text to an interpreter."""
    result, block = [], False
    for number, original in enumerate(text.splitlines(), 1):
        line = original.strip()
        if block:
            if '*/' in line:
                block = False
            continue
        if line.startswith('/*'):
            block = '*/' not in line[2:]
            continue
        line = re.split(r'!|//', line, maxsplit=1)[0].strip()
        if not line:
            continue
        if line.startswith('~'):
            require(bool(result), 'orphan continuation')
            result[-1]['text'] += ' ' + line[1:].strip()
            result[-1]['lines'].append(number)
        else:
            result.append({'text': line, 'lines': [number]})
    require(not block, 'unclosed block comment')
    return result


def properties(text):
    pattern = r'([%\w]+)\s*=\s*("[^"]*"|\x27[^\x27]*\x27|\([^)]*\)|\[[^]]*\]|[^\s]+)'
    return [(k.lower(), v.strip('"\x27')) for k, v in re.findall(pattern, text)]


def device_declarations(commands):
    result = {}
    for record in commands:
        line = record['text']
        new = re.match(r'new\s+(\S+)\s*(.*)$', line, re.I)
        edit = re.match(r'edit\s+(\S+)\s*(.*)$', line, re.I)
        direct = re.match(r'((?:load|regcontrol|capacitor)\.[^.\s]+)\.([%\w]+)\s*=\s*(.*)$', line, re.I)
        if new or edit:
            name, tail = (new or edit).groups()
            name = name.lower()
            if not name.startswith(KINDS):
                continue
            if new:
                require(name not in result, 'duplicate relevant declaration ' + name)
                result[name] = {'properties': {}, 'assignments': []}
            require(name in result, 'edit before declaration ' + name)
            for key, value in properties(tail):
                result[name]['properties'][key] = value
                result[name]['assignments'].append({'property': key, 'value': value, 'lines': record['lines']})
        elif direct:
            name, key, value = direct.groups()
            name, key = name.lower(), key.lower()
            require(name in result, 'direct edit before declaration')
            result[name]['properties'][key] = value.strip()
            result[name]['assignments'].append({'property': key, 'value': value.strip(), 'lines': record['lines']})
        elif re.match(r'batchedit\s+(load|regcontrol|capacitor)', line, re.I):
            raise ValueError('relevant BatchEdit requires explicit parser extension')
    return result


def coil_indices(phases, conductors, delta):
    require(phases in (1, 2, 3), 'unsupported phase count')
    if delta:
        require(conductors == (phases if phases == 3 else phases + 1), 'delta conductor mismatch')
        return [(i, (i + 1) % conductors) for i in range(phases)]
    require(conductors == phases + 1, 'wye conductor mismatch')
    return [(i, conductors - 1) for i in range(phases)]


def load_voltage_factors(model, rho, *, vlow=.5, vmin=.85, vmax=1.05,
                         cvr_watts=1., cvr_vars=2.):
    """Exact scalar P/Q law from pinned Load.pas (fundamental load flow).

    Low-voltage interpolation is of current magnitude, not impedance. Model 4
    outside its normal range uses the same PQ fallback as model 1 in this
    backend, even when this creates an endpoint discontinuity.
    """
    require(model in (1, 2, 4, 5), 'unsupported load model')
    require(math.isfinite(rho) and rho >= 0, 'invalid supplied voltage ratio')
    require(0 < vlow < vmin < vmax, 'invalid voltage-law thresholds')
    if model == 2 or rho <= vlow:
        return rho * rho, rho * rho
    if rho <= vmin:
        upper_current = 1. if model == 5 else 1. / vmin
        factor = rho * (vlow + (upper_current - vlow) * (rho - vlow) / (vmin - vlow))
        return factor, factor
    if rho > vmax:
        factor = rho * rho / (vmax if model == 5 else vmax * vmax)
        return factor, factor
    if model == 1:
        return 1., 1.
    if model == 4:
        return rho ** cvr_watts, rho ** cvr_vars
    return rho, rho


def generator_voltage_factor(rho, vmin=.9, vmax=1.1):
    require(math.isfinite(rho) and rho >= 0, 'invalid supplied generator voltage')
    return (rho / vmin) ** 2 if rho <= vmin else ((rho / vmax) ** 2 if rho > vmax else 1.)


def regulator_control_voltage(voltage, current_into_winding, *, pt_ratio, ct_primary, r, x):
    """Pinned RegControl.Sample sign: into-winding current uses PLUS LDC."""
    return abs(voltage / pt_ratio + complex(r, x) * current_into_winding / ct_primary)


def regulator_equilibrium_region(tap_index, controlled_volts, *, vreg, band, minimum=-16, maximum=16):
    """Proposed steady deadband/saturation proxy, NOT the original control queue.

    This predicate permits only endogenous automatic-control equilibrium. It
    is still a declared approximation if several tap states satisfy it.
    """
    require(isinstance(tap_index, int) and minimum <= tap_index <= maximum, 'invalid discrete tap')
    lower, upper = vreg - band / 2, vreg + band / 2
    if tap_index == minimum:
        return controlled_volts >= lower
    if tap_index == maximum:
        return controlled_volts <= upper
    return lower <= controlled_volts <= upper


def bound_read(run, seal, relative, bindings):
    path = run / relative
    value = sha(path)
    require(seal.get(relative) == value, 'sealed input mismatch ' + relative)
    bindings[relative] = value
    return read(path) if path.suffix == '.json' else path.read_text(encoding='utf-8')


def build_device_mapping(source_run, *, project=None):
    run = Path(source_run).resolve()
    project = Path(project).resolve() if project else run.parent.parent
    require(sha(run / 'completion_manifest.json') == SOURCE_ROOT_SHA, 'wrong observed source run')
    seal, bindings = manifest(run / 'completion_manifest.json'), {}
    data = {f: bound_read(run, seal, f + '.json', bindings) for f in
            ('base_load_properties', 'metadata', 'author_static_metadata', 'command_inventory',
             'condition', 'config', 'dependencies', 'physical_config')}
    require(data['dependencies']['dss-python-backend'] == '0.14.5', 'unsupported backend version')
    official = {}
    for name, expected in OFFICIAL_SHAS.items():
        path = project / OFFICIAL_DIR / name
        require(sha(path) == expected, 'official version/source mismatch ' + name)
        official[name] = {'path': str(path.relative_to(project)), 'sha256': expected,
                          'url': 'https://raw.githubusercontent.com/dss-extensions/dss_capi/0.14.5/src/' +
                          ('Controls/' if name == 'RegControl.pas' else 'PDElements/' if name in ('Capacitor.pas', 'Transformer.pas') else 'PCElements/') + name}
    inventory = data['command_inventory']
    dss_file = inventory['files']['ieee34Mod1.dss']
    dss_path = Path(dss_file['path'])
    require(sha(dss_path) == dss_file['sha256'], 'original DSS bytes changed')
    commands = static_commands(dss_path.read_text(encoding='utf-8'))
    declarations = device_declarations(commands)
    author_dir = 'episodes/000/physical/work/Environments/DSSdirect_34bus_loadandswitching/'
    init_text = bound_read(run, seal, author_dir + 'DSS_Initialize.py', bindings)
    setup_text = bound_read(run, seal, author_dir + 'DSS_CircuitSetup.py', bindings)
    require('Pf=0.85 Model=1' in setup_text, 'generator creation contract changed')
    generators = None
    for node in ast.parse(init_text).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'generators' for t in node.targets):
            generators = ast.literal_eval(node.value)
    require(generators is not None and len(generators) == 4, 'generator declaration inventory')
    runtime = bound_read(run, seal, 'source/src/dnr/nature34_clpu_physical.py', bindings)
    bound_read(run, seal, 'source/src/dnr/nature34_operating_physical.py', bindings)
    require('self.core.initialize()' in runtime and "startswith('regcontrol.')" in runtime.lower(), 'reset/control source contract')
    layouts = data['metadata']['element_layout']
    condition_loads = {r['name']: r for r in data['condition']['loads']}
    dispatch = {name.lower(): i for i, name in enumerate(data['author_static_metadata']['dispatch_loads'])}
    loads = []
    for name, base in data['base_load_properties'].items():
        dec, layout = declarations[name.lower()], layouts[name]
        props = dec['properties']
        for key, expected in [('kw', base['nominal_kw']), ('kvar', base['nominal_kvar']),
                              ('kv', base['kv']), ('model', base['model']), ('phases', base['phases'])]:
            require(float(props[key]) == expected, 'load readback/declaration mismatch ' + name + ':' + key)
        require((props['conn'].lower() == 'delta') == base['is_delta'], 'load connection mismatch')
        require(props['bus1'].lower() == layout['buses'][0].lower(), 'load bus mismatch')
        require(float(props['vminpu']) == .85, 'explicit vmin override changed')
        require(not set(props).difference({'bus1', 'phases', 'conn', 'model', 'kv', 'kw', 'kvar', 'vminpu'}), 'unexpected load properties')
        coils = coil_indices(base['phases'], base['conductors'], base['is_delta'])
        nodes = layout['node_order']
        vbase = base['kv'] * 1000 / (math.sqrt(3) if not base['is_delta'] and base['phases'] in (2, 3) else 1)
        normal = condition_loads[name]
        loads.append({'name': name, **base, 'connection': 'delta' if base['is_delta'] else 'wye',
                      'node_order': nodes, 'coil_indices': coils,
                      'coil_nodes': [[nodes[i], nodes[j]] for i, j in coils], 'coil_base_volts': vbase,
                      'normal_kw': normal['demand_nominal_kw'], 'normal_kvar': normal['demand_nominal_kvar'],
                      'dispatch_bit': dispatch.get(name.lower()),
                      'dispatch_bit_basis': 'joint command low ten readiness bits; action=bit+1',
                      'physical_action_index': None if name.lower() not in dispatch else 9 + dispatch[name.lower()],
                      'setpoint_rule': 'normal P and Q times the same original per-load CLPU multiplier; original model and connection retained',
                      'voltage_law': {'model': base['model'], 'vlow': .5, 'vmin': .85, 'vmax': 1.05,
                                      'cvr_watts': 1., 'cvr_vars': 2.,
                                      'normal_exponents': {1: [0, 0], 2: [2, 2], 4: [1, 2], 5: [1, 1]}[base['model']],
                                      'function': 'load_voltage_factors',
                                      'source': 'Load.pas: SetNominalLoad/DoConstantPQLoad/DoConstantZLoad/DoCVRModel/DoConstantILoad/InterpolateY95_YLow'},
                      'declaration': dec})
    alpha = data['config']['alpha']
    ders = []
    for declaration in generators:
        name = 'Generator.g' + str(declaration['no'])
        ref, layout = data['metadata']['generator_reference'][name], layouts[name]
        require(declaration['size'] == ref['available_p_kw'], 'DER reference mismatch')
        require(layout['node_order'] == ([1, 0] if declaration['numphase'] == 1 else [1, 2, 3, 0]), 'DER connection mismatch')
        ders.append({'name': name, 'declaration': declaration, 'connection': 'wye', 'model': 1,
                     'node_order': layout['node_order'], 'alpha': alpha,
                     'p_setpoint_kw': alpha * ref['available_p_kw'], 'q_setpoint_kvar': alpha * ref['reference_q_kvar'],
                     'reference': ref, 'optimization_dispatch_variable': False, 'island_source_allowed': False,
                     'voltage_law': {'function': 'generator_voltage_factor', 'vmin': .9, 'vmax': 1.1,
                                     'normal_region': 'fixed PQ injection', 'outside': 'constant impedance fallback'},
                     'availability': 'enabled only in sourcebus-connected supplied component',
                     'default_source': 'generator.pas constructor Connection=0, GenModel=1, Vminpu=.90,Vmaxpu=1.10'})
    regulators, capacitors = [], []
    for name, dec in declarations.items():
        p = dec['properties']
        if name.startswith('regcontrol.'):
            regulators.append({'name': name, 'declaration': dec,
                'transformer': 'Transformer.' + p['transformer'].lower(), 'winding': int(p['winding']),
                'tap_winding': int(p['winding']), 'vreg_volts': float(p['vreg']), 'band_volts': float(p['band']),
                'pt_ratio': float(p['ptratio']), 'ct_primary_amperes': float(p['ctprim']),
                'ldc_r_volts': float(p['r']), 'ldc_x_volts': float(p['x']),
                'source_defaults': {'delay_seconds': 15., 'tap_delay_seconds': 2., 'max_tap_change': 16,
                    'reversible': False, 'cogen': False, 'pt_phase': 1, 'mintap': .9, 'maxtap': 1.1, 'numtaps': 32},
                'control': 'automatic when its transformer is enabled; independently reset and re-solved at each physical sample',
                'tap_is_optimization_control': False, 'initial_operating_tap': None,
                'initial_tap_reason': 'constructor tap=1 is not the operating reset tap: original initialize already solves controls',
                'continuous_tap_memory_between_samples': False})
        elif name.startswith('capacitor.'):
            capname = next(k for k in layouts if k.lower() == name)
            capacitors.append({'name': capname, 'declaration': dec, 'connection': 'wye', 'phases': int(p['phases']),
                'node_order': layouts[capname]['node_order'], 'buses': layouts[capname]['buses'],
                'nominal_kvar': float(p['kvar']), 'kv_line_line': float(p['kv']),
                'voltage_law': 'each phase reactive injection = nominal_kvar/3 * (|V_phase-ground|/(kv_LL*1000/sqrt(3)))^2',
                'num_steps': 1, 'states': [1], 'switching_optimization_variable': False,
                'availability': 'whole element disabled if its buses are not sourcebus-connected',
                'source_defaults': 'Capacitor.pas: grounded wye, one step on, R=XL=0; no CapControl declared'})
    require(len(loads) == 68 and len(regulators) == 6 and len(capacitors) == 2, 'device population mismatch')
    require(not any(re.match(r'new\s+capcontrol\.', c['text'], re.I) for c in commands), 'unexpected capacitor controller')
    return {'version': VERSION, 'status': 'INPUT_MAPPING_COMPLETE_PROXY_NOT_IMPLEMENTED',
        'source_run': str(run), 'source_completion_manifest_sha256': SOURCE_ROOT_SHA,
        'source_files': bindings, 'original_dss': dss_file, 'official_sources': official,
        'dependencies': {k: data['dependencies'][k] for k in ('dss-python-backend', 'dss-python', 'opendssdirect.py')},
        'new_physical_points': 0, 'new_training': 0, 'dss_context_created': False,
        'loads': loads, 'generators': ders, 'regcontrols': regulators, 'capacitors': capacitors,
        'counts': {'loads': len(loads), 'load_models': dict(Counter(str(x['model']) for x in loads)),
                   'delta_ground_coils': sum(x['is_delta'] and any(0 in c for c in x['coil_nodes']) for x in loads),
                   'generators': len(ders), 'regcontrols': len(regulators), 'capacitors': len(capacitors)},
        'scope': 'fundamental quasi-static device semantics of observed training source only; no unseen data, no electrical solve',
        'approximation_contract': {
            'optimization_matrix_implemented': False, 'ac_equivalence_claim': False,
            'load_layer': 'Use each recorded coil incidence and branch rated voltage. P/Q factors are piecewise functions of rho=|coil voltage|/rated voltage; never replace all models by constant PQ.',
            'milp_option_not_frozen': 'Piecewise-affine approximation of these functions in squared coil voltage; segments must include every mapped low/min/max threshold and preserve delta phase-to-ground cases. Breakpoint grid, approximation error bounds, and bilinear energization/CLPU products still need implementation and freezing.',
            'regulator_option_not_frozen': 'Steady deadband/saturation proxy: n in {-16,...,16}, tap=1+n*.00625; Vc=abs(V_winding2/PT+(R+jX)*I_into_winding2/CT). Interior taps require vreg-band/2 <= Vc <= vreg+band/2; n=-16 requires Vc >= lower; n=16 requires Vc <= upper. Endogenous taps are constrained states, not free controls. Linearize the modulus explicitly in the MILP and state its error.',
            'regulator_equilibrium_limitation': 'This proxy need not reconstruct the control queue or the tap chosen after independent initialization, especially at discrete rounding/multiple equilibria. Multiple admissible taps are an approximation, not proven runtime equivalence. New candidate plans still require original automatic-control AC certification. No constructor tap is silently fixed.',
            'capacitor_layer': 'Fixed enabled shunt susceptance, voltage-squared kvar, no controllable capacitor binary apart from graph energization.',
            'der_layer': 'Fixed alpha=.5 setpoints, no new dispatch/VAR/islanding freedoms; original model1 voltage fallback retained.'},
        'unresolved': [
            {'code': 'CONTROL_PROXY_NOT_FROZEN', 'kind': 'modeling_choice', 'blocking_cold_comparison': True,
             'remedy': 'Implement and freeze declared steady LDC/deadband/saturation constraints (source reversible=false), bind reset semantics, and check rejection cases/old operating states. Full queue reproduction is not required for an explicitly named proxy; complexity alone is not missing source data.'},
            {'code': 'MILP_DEVICE_COEFFICIENTS_NOT_IMPLEMENTED', 'kind': 'implementation', 'blocking_cold_comparison': True,
             'remedy': 'Connect per-coil load/DER/capacitor formulas to three-phase network matrices and test residuals/energization; choose and document any linearization error.'}],
        'forbidden_inferences': ['input mapping success is not a solved MILP or AC feasibility',
                                'constructor tap=1 is not per-point operating tap',
                                'a few historical residual checks do not prove proxy accuracy on new plans']}


def old_point_checks(mapping, source_run):
    """Four preselected historical points; no solver and no result-based tuning."""
    run = Path(source_run)
    seal = manifest(run / 'completion_manifest.json')
    by_name = {x['name']: x for x in mapping['loads']}
    rows = []
    sources = {}
    for cell in ('011', '020'):
        relative = f'episodes/{cell}/physical/points.jsonl.gz'
        path = run / relative
        require(sha(path) == seal[relative], 'historical point journal changed')
        sources[relative] = seal[relative]
        chosen = []
        with gzip.open(path, 'rt', encoding='utf-8') as handle:
            for i, line in enumerate(handle):
                point = json.loads(line)
                if i == 0:
                    chosen.append(('INITIAL', point))
                elif point['status'] == 'FAIL' and point['reliable']:
                    chosen.append(('FIRST_RELIABLE_FAIL', point))
                    break
        require(len(chosen) == 2, 'expected fixed initial and first reliable fail')
        for role, point in chosen:
            residuals, taps = [], {}
            regulator_by_transformer = {r['transformer']: r for r in mapping['regcontrols']}
            for element in point['raw_outputs']['elements']:
                if element['name'].lower().startswith('transformer.reg'):
                    taps[element['name']] = {'enabled': element['enabled'], 'taps': element['winding_taps']}
                    if element['enabled']:
                        reg = regulator_by_transformer[element['name']]
                        offset = 2 * element['conductors'] * (reg['winding'] - 1)
                        v = complex(*element['voltages_real_imag'][offset:offset+2])
                        current = complex(*element['currents_real_imag'][offset:offset+2])
                        vc = regulator_control_voltage(v, current, pt_ratio=reg['pt_ratio'],
                            ct_primary=reg['ct_primary_amperes'], r=reg['ldc_r_volts'], x=reg['ldc_x_volts'])
                        n = round((element['winding_taps'][reg['tap_winding']-1] - 1.) / .00625)
                        taps[element['name']].update({'controlled_volts': vc, 'tap_index': n,
                            'steady_deadband_proxy_admissible': regulator_equilibrium_region(n, vc, vreg=reg['vreg_volts'], band=reg['band_volts']),
                            'requested_continuous_direction': 'increase_tap' if vc < reg['vreg_volts']-reg['band_volts']/2 else 'decrease_tap' if vc > reg['vreg_volts']+reg['band_volts']/2 else 'inside_deadband'})
                if element['name'] not in by_name or not element['enabled']:
                    continue
                item = by_name[element['name']]
                require(element['node_order'] == item['node_order'], 'historical load node order mismatch')
                values = element['voltages_real_imag']
                volts = [complex(*values[i:i+2]) for i in range(0, len(values), 2)]
                factors = [load_voltage_factors(item['model'], abs(volts[a] - volts[b]) / item['coil_base_volts'])
                           for a, b in item['coil_indices']]
                setpoint = point['load_readback']['after'][item['name']]
                predicted_p = setpoint['nominal_kw'] * math.fsum(x[0] for x in factors) / item['phases']
                predicted_q = setpoint['nominal_kvar'] * math.fsum(x[1] for x in factors) / item['phases']
                powers = element['powers_kw_kvar']
                actual_p, actual_q = math.fsum(powers[::2]), math.fsum(powers[1::2])
                residuals.append({'load': item['name'], 'model': item['model'], 'connection': item['connection'],
                    'node_order': item['node_order'], 'predicted_kw': predicted_p, 'actual_kw': actual_p,
                    'predicted_kvar': predicted_q, 'actual_kvar': actual_q,
                    'abs_kw_residual': abs(predicted_p - actual_p), 'abs_kvar_residual': abs(predicted_q - actual_q)})
            rows.append({'cell': cell, 'role': role, 'point_id': point['point_id'], 'status': point['status'],
                         'point_canonical_sha256': digest(point), 'control_iterations': point['raw_outputs']['control_iterations'],
                         'regulator_taps': taps, 'enabled_load_residuals': residuals,
                         'max_abs_kw_residual': max(x['abs_kw_residual'] for x in residuals),
                         'max_abs_kvar_residual': max(x['abs_kvar_residual'] for x in residuals)})
    return {'status': 'DESCRIPTIVE_ALGEBRA_REPLAY_NOT_PROXY_VALIDATION', 'selection_rule': 'c02 window011 and c07 window020; first paid point and first subsequent reliable FAIL each',
            'new_physical_points': 0, 'new_training': 0, 'sources': sources, 'points': rows,
            'limitations': 'Uses saved converged voltages to check device equations, not a network solution. Finite residuals are reported, not used to retune parameters. Does not establish regulator-control equivalence.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', default=DEFAULT_SOURCE)
    parser.add_argument('--output', required=True)
    parser.add_argument('--check-four-old-points', action='store_true')
    args = parser.parse_args(argv)
    out = Path(args.output)
    require(not out.exists(), 'output must be new')
    result = build_device_mapping(args.source_run)
    old = old_point_checks(result, args.source_run) if args.check_four_old_points else None
    out.mkdir(parents=True)
    for name, value in [('mapping.json', result), ('historical_four_point_checks.json', old)]:
        if value is not None:
            (out / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    shutil.copyfile(__file__, out / Path(__file__).name)
    (out / 'README.md').write_text('# Static device input mapping\n\nNo circuit was solved and no model was trained. `mapping.json` preserves the source device semantics and lists the unimplemented approximation decisions. The optional four-point file evaluates algebra at existing voltages; it is not a new simulation or validation of the future MILP. Input mapping completion does not authorize the four cold comparisons.\n', encoding='utf-8')
    files = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    (out / 'completion_manifest.json').write_text(json.dumps(files, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'counts': result['counts'], 'output': str(out),
                      'completion_sha256': sha(out / 'completion_manifest.json')}))


if __name__ == '__main__':
    main()
