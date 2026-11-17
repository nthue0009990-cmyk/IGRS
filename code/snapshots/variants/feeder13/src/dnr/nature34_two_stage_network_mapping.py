"""Static, source-bound network data for the NC34 two-stage adaptation.

This is a deliberately restricted DSS declaration reader, not a DSS interpreter.
It never imports an engine, evaluates RPN expressions, creates Y matrices or
solves an electrical system.  Unknown electrical syntax is reported, not defaulted.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

from .dss_command_inventory import _literal_include, _without_line_comment


VERSION = 'nc34-two-stage-static-network-v1'
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / 'data/raw/literature/two_stage_static_mapping_20261031/network/Transformer.pas'
DEFAULT_SOURCE_SHA256 = '53148a652db204f033eae2422cee6be0fcd3357a035e4c5641342dd6d51c319a'
DEFAULT_SOURCE_URL = 'https://raw.githubusercontent.com/dss-extensions/dss_capi/0.14.5/src/PDElements/Transformer.pas'
AUTHOR_SETUP = ROOT / ('data/raw/literature/restoration_shortlist/'
    '20261018T120520_932034Z_nature_author_subset/author/RL_Training/'
    'Environments/DSSdirect_34bus_loadandswitching/DSS_CircuitSetup.py')
METRES_PER_UNIT = {'m': 1., 'km': 1000., 'ft': .3048, 'kft': 304.8,
                   'mi': 1609.344, 'in': .0254, 'cm': .01, 'mm': .001}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _number(value):
    if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', str(value).strip()):
        raise ValueError('Nonliteral numeric property: ' + str(value))
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Nonfinite property')
    return result


def _integer(value):
    number = _number(value)
    if number != int(number):
        raise ValueError('Integer property required')
    return int(number)


def _words(value):
    text = value.strip()
    if text and text[0] in '[\"\'(':
        text = text[1:-1]
    return [x for x in re.split(r'[\s,]+', text.strip()) if x]


def parse_properties(text):
    """Ordered key=value pairs; quoted/parenthesized/bracketed values stay intact."""
    result, pos = [], 0
    while pos < len(text):
        while pos < len(text) and (text[pos].isspace() or text[pos] == ','):
            pos += 1
        if pos == len(text):
            break
        match = re.match(r'([%\w]+)\s*=\s*', text[pos:])
        if not match:
            raise ValueError('Unnamed/unsupported DSS property: ' + text[pos:])
        name = match.group(1).lower()
        pos += match.end()
        if pos == len(text):
            raise ValueError('Missing value for ' + name)
        start = pos
        pairs = {'[': ']', '(': ')', '"': '"', "'": "'"}
        if text[pos] in pairs:
            close = pairs[text[pos]]
            pos += 1
            while pos < len(text) and text[pos] != close:
                pos += 1
            if pos == len(text):
                raise ValueError('Unclosed DSS property')
            pos += 1
        else:
            while pos < len(text) and not text[pos].isspace() and text[pos] != ',':
                pos += 1
        result.append((name, text[start:pos]))
    return result


def read_static_objects(entry_file, *, allowed_root=None, expected_files=None):
    """Read effective NEW/EDIT declarations and literal includes, without execution.

    Reuses the project's pinned first-column block/line comment semantics. Every
    property retains its source line and order. Unknown commands are retained.
    """
    entry = Path(entry_file).resolve()
    root = Path(allowed_root).resolve() if allowed_root else entry.parent
    objects, files, commands = {}, {}, []
    active_object = None
    sequence = 0

    def visit(path, chain):
        nonlocal active_object, sequence
        path = path.resolve()
        if not path.is_relative_to(root) or path in chain:
            raise ValueError('Include escapes root or forms a cycle')
        data = path.read_bytes()
        ident = path.relative_to(root).as_posix()
        digest = hashlib.sha256(data).hexdigest()
        if expected_files is not None:
            expected = expected_files.get(ident)
            if expected is None or expected['sha256'] != digest:
                raise ValueError('DSS input differs from frozen command inventory: ' + ident)
        files[ident] = {'path': str(path), 'sha256': digest, 'bytes': len(data)}
        block = False
        directory = path.parent
        for line_number, line in enumerate(data.decode('utf-8-sig').splitlines(), 1):
            if not block and line.startswith('/*'):
                block = True
            if block:
                if '*/' in line:
                    block = False
                continue
            code = _without_line_comment(line)
            if not code:
                continue
            source = {'file': ident, 'line': line_number, 'sha256': digest}
            first, *rest = code.split(None, 1)
            keyword, payload = first.lower(), rest[0] if rest else ''
            if keyword in ('redirect', 'compile'):
                target = (directory / _literal_include(payload)).resolve()
                visit(target, (*chain, path))
                if keyword == 'compile':
                    directory = target.parent
                continue
            if keyword == 'cd' or (keyword == 'set' and re.match(r'datapath\s*=', payload, re.I)):
                raise ValueError('Unsupported directory mutation')
            if keyword == 'clear':
                if objects:
                    raise ValueError('CLEAR after declarations is unsupported')
                continue
            if keyword in ('new', 'edit'):
                payload = re.sub(r'^object\s*=\s*', '', payload, flags=re.I)
                match = re.match(r'([\w]+\.[\w-]+)(?:\s+|$)', payload)
                if not match:
                    raise ValueError('Unsupported object identifier')
                name = match.group(1).lower()
                if keyword == 'new':
                    if name in objects:
                        raise ValueError('Duplicate NEW object: ' + name)
                    objects[name] = {'name': name, 'class': name.split('.')[0],
                                     'created': source, 'properties': []}
                elif name not in objects:
                    raise ValueError('EDIT before NEW: ' + name)
                active_object = name
                if objects[name]['class'] == 'energymeter':
                    # This declaration only meters a branch; named source uses
                    # positional arguments, which are not electrical parameters.
                    objects[name]['raw_declaration'] = code
                    continue
                properties = parse_properties(payload[match.end():])
            elif keyword == '~':
                if active_object is None:
                    raise ValueError('Continuation without object')
                properties = parse_properties(payload)
            elif re.match(r'^[\w]+\.[\w-]+\.[%\w]+=', code):
                match = re.match(r'^([\w]+\.[\w-]+)\.(.*)$', code)
                active_object = match.group(1).lower()
                if active_object not in objects:
                    raise ValueError('Object property before NEW')
                properties = parse_properties(match.group(2))
            else:
                commands.append({'keyword': keyword, 'text': code, 'source': source,
                                 'executed': False})
                continue
            for key, value in properties:
                sequence += 1
                objects[active_object]['properties'].append(
                    {'key': key, 'value': value, 'sequence': sequence, 'source': deepcopy(source)})
    visit(entry, ())
    return {'objects': objects, 'source_files': files, 'non_object_commands': commands,
            'new_physical_points': 0, 'new_numerical_solutions': 0}


def _effective(obj):
    return {p['key']: p for p in obj['properties']}


def _required(props, key):
    if key not in props:
        raise ValueError('Missing explicit property: ' + key)
    return props[key]['value']


def symmetric_matrix(value, phases):
    text = value.strip()
    if not text or text[0] not in '[(' or text[-1] not in '])':
        raise ValueError('Literal matrix required')
    rows = [[_number(x) for x in _words(row)] for row in text[1:-1].split('|')]
    if len(rows) != phases or any(len(row) != i + 1 for i, row in enumerate(rows)):
        raise ValueError('Expected lower triangle matching phase count')
    return [[rows[max(i, j)][min(i, j)] for j in range(phases)] for i in range(phases)]


def terminal_nodes(spec, phases, *, transformer=False):
    words = spec.lower().split('.')
    bus = words[0]
    nodes = [_integer(x) for x in words[1:]]
    if not nodes:
        nodes = list(range(1, phases + 1))
    size = phases + 1 if transformer else phases
    if transformer and len(nodes) == phases:
        nodes += [0]  # ordinary DSS bus connection: omitted extra conductor is ground
    if len(nodes) != size or any(x not in (0, 1, 2, 3) for x in nodes):
        raise ValueError('Unsupported terminal conductor specification: ' + spec)
    if any(x == 0 for x in nodes[:phases]) or len(set(nodes[:phases])) != phases:
        raise ValueError('Invalid/duplicate phase nodes')
    return {'bus': bus, 'nodes': nodes, 'phase_order': nodes[:phases],
            'neutral_node': nodes[-1] if transformer else None, 'declared': spec}


def linear_voltage_drop(r_ohm, x_ohm, phase_order):
    """Balanced-angle, lossless coefficient derivation; not an AC solution.

    v_to[kV^2] = v_from[kV^2] + MP P[kW] + MQ Q[kvar].
    Re((V_phi/V_psi) conj(Z_phi,psi) (P+jQ)) supplies the cross terms.
    """
    angle = {1: 0., 2: -2. * math.pi / 3., 3: 2. * math.pi / 3.}
    mp, mq = [], []
    for i, p in enumerate(phase_order):
        prow, qrow = [], []
        for j, q in enumerate(phase_order):
            delta = angle[p] - angle[q]
            c, s = math.cos(delta), math.sin(delta)
            prow.append(-.002 * (r_ohm[i][j] * c + x_ohm[i][j] * s))
            qrow.append(-.002 * (x_ohm[i][j] * c - r_ohm[i][j] * s))
        mp.append(prow)
        mq.append(qrow)
    return {'p_kv2_per_kw': mp, 'q_kv2_per_kvar': mq,
            'equation': 'v_to_squared_kv2=v_from_squared_kv2+MP*P_kw+MQ*Q_kvar',
            'approximation': 'lossless balanced-angle equal-phase-magnitude linearization; not AC feasibility',
            'angle_reference_radians': {str(k): v for k, v in angle.items()}}


def _line(obj, objects, ratings):
    props = _effective(obj)
    allowed = {'phases', 'bus1', 'bus2', 'linecode', 'length', 'units'}
    if set(props) - allowed:
        raise ValueError('Unsupported line override(s): ' + ','.join(sorted(set(props) - allowed)))
    code_name = 'linecode.' + _required(props, 'linecode').lower()
    if code_name not in objects:
        raise ValueError('Missing linecode: ' + code_name)
    code = _effective(objects[code_name])
    allowed_code = {'nphases', 'basefreq', 'units', 'rmatrix', 'xmatrix', 'cmatrix'}
    if set(code) - allowed_code:
        raise ValueError('Unsupported active LineCode property')
    last_line_sequence = max((p.get('sequence', math.inf) for p in obj['properties']), default=0)
    if any(p.get('sequence', 0) > last_line_sequence for p in objects[code_name]['properties']):
        raise ValueError('LineCode edit after line declaration requires declaration-time copy semantics')
    if 'nphases' in code and any(code['nphases'].get('sequence', 0) > code.get(k, {}).get('sequence', math.inf)
                               for k in ('rmatrix', 'xmatrix', 'cmatrix')):
        raise ValueError('LineCode NPhases reset after matrix declaration is unsupported')
    phases = _integer(_required(code, 'nphases'))
    if phases not in (1, 3):
        raise ValueError('Only named one/three-phase lines are supported')
    if 'phases' in props and _integer(props['phases']['value']) != phases:
        raise ValueError('Line/LineCode phase count mismatch')
    unit, cunit = _required(props, 'units').lower(), _required(code, 'units').lower()
    if unit not in METRES_PER_UNIT or cunit not in METRES_PER_UNIT:
        raise ValueError('Explicit supported length units required')
    length = _number(_required(props, 'length'))
    if length <= 0:
        raise ValueError('Positive length required')
    scale = length * METRES_PER_UNIT[unit] / METRES_PER_UNIT[cunit]
    r, x, c = [symmetric_matrix(_required(code, k), phases) for k in ('rmatrix', 'xmatrix', 'cmatrix')]
    r, x, c = [[[a * scale for a in row] for row in matrix] for matrix in (r, x, c)]
    t1, t2 = [terminal_nodes(_required(props, k), phases) for k in ('bus1', 'bus2')]
    if t1['phase_order'] != t2['phase_order']:
        raise ValueError('Phase-changing line requires explicit additional mapping')
    frequency = _number(_required(code, 'basefreq'))
    rating = ratings.get(code_name.split('.')[1])
    if rating is None or _number(rating) <= 0:
        raise ValueError('No sourced phase ampacity for active LineCode')
    return {'name': obj['name'], 'phases': phases, 'bus1': t1['bus'], 'bus2': t2['bus'],
            'bus1_nodes': t1['nodes'], 'bus2_nodes': t2['nodes'], 'phase_order': t1['phase_order'],
            'linecode': code_name, 'length': length, 'length_unit': unit,
            'length_in_linecode_units': scale, 'linecode_unit': cunit, 'base_frequency_hz': frequency,
            'r_ohm': r, 'x_ohm': x, 'z_ohm': [[{'re': rr, 'im': xx} for rr, xx in zip(ra, xa)] for ra, xa in zip(r, x)],
            'shunt_capacitance_nf': c,
            'pi_terminal_shunt_y_siemens': [[{'re': 0., 'im': math.pi * frequency * a * 1e-9} for a in row] for row in c],
            'phase_ampacity_a': float(rating),
            'rating_source': 'physical_config.line_phase_ampacity_A; not OpenDSS default NormAmps',
            'linear_voltage_drop': linear_voltage_drop(r, x, t1['phase_order']),
            'properties': deepcopy(obj['properties']), 'linecode_properties': deepcopy(objects[code_name]['properties'])}


def _default_winding():
    return {'tap': 1., '%r': .2, 'mintap': .9, 'maxtap': 1.1, 'numtaps': 32,
            'rneut': -1., 'xneut': 0., 'conn': 'wye'}


def _transformer(obj, controls):
    """Ordered named-source semantics, including kVA and %loadloss side effects."""
    general, windings, sources, active = {}, [], {}, 1
    winding_keys = {'bus', 'conn', 'kv', 'kva', 'tap', '%r', 'rneut', 'xneut', 'mintap', 'maxtap', 'numtaps'}
    arrays = {'buses': 'bus', 'conns': 'conn', 'kvs': 'kv', 'kvas': 'kva', 'taps': 'tap', '%rs': '%r'}
    general_allowed = {'phases', 'windings', 'xhl', 'leadlag', '%loadloss', '%noloadloss', '%imag', 'ppm_antifloat'}
    for prop in obj['properties']:
        key, value = prop['key'], prop['value']
        if key == 'windings':
            if _integer(value) != 2:
                raise ValueError('Only explicit two-winding transformers supported')
            windings = [_default_winding(), _default_winding()]
            sources = {str(i): {k: {'default_source': DEFAULT_SOURCE_URL, 'symbol': 'TWinding.Init'}
                                for k in w} for i, w in enumerate(windings, 1)}
            general[key] = 2
        elif key == 'wdg':
            active = _integer(value)
            if active not in (1, 2) or not windings:
                raise ValueError('Invalid winding selector')
        elif key in arrays:
            values = _words(value)
            if len(values) != 2 or len(windings) != 2:
                raise ValueError('Winding array size mismatch')
            dest = arrays[key]
            for index, item in enumerate(values):
                windings[index][dest] = item.lower() if dest in ('bus', 'conn') else _number(item)
                sources[str(index + 1)][dest] = deepcopy(prop['source'])
            if dest == '%r':
                general['%loadloss'] = sum(w['%r'] for w in windings)
        elif key in winding_keys:
            if not windings:
                raise ValueError('Winding declaration order missing')
            parsed = value.lower() if key in ('bus', 'conn') else _number(value)
            windings[active - 1][key] = parsed
            sources[str(active)][key] = deepcopy(prop['source'])
            if key == 'kva':  # explicit single-winding setter forces both in a two-winding unit
                for i in range(2):
                    windings[i]['kva'] = parsed
                    sources[str(i + 1)]['kva'] = {**deepcopy(prop['source']), 'side_effect': 'two-winding kVA setter'}
            if key == '%r':
                general['%loadloss'] = sum(w['%r'] for w in windings)
        elif key in general_allowed:
            general[key] = value.lower() if key == 'leadlag' else _number(value)
            if key == '%loadloss':
                if not windings:
                    raise ValueError('%loadloss before windings unsupported')
                for i, w in enumerate(windings):
                    w['%r'] = general[key] / 2.
                    sources[str(i + 1)]['%r'] = {**deepcopy(prop['source']), 'side_effect': '%loadloss / 2'}
        else:
            raise ValueError('Unsupported transformer property: ' + key)
    if len(windings) != 2 or 'phases' not in general or 'xhl' not in general:
        raise ValueError('Explicit phases/windings/XHL required')
    phases = _integer(general['phases'])
    if phases not in (1, 3):
        raise ValueError('Only one/three-phase transformers supported')
    for w in windings:
        if not {'bus', 'conn', 'kv', 'kva'} <= set(w) or w['kv'] <= 0 or w['kva'] <= 0:
            raise ValueError('Incomplete winding ratings or connection')
        if w['conn'] not in ('wye', 'delta'):
            raise ValueError('Unsupported winding connection')
        if phases == 1 and w['conn'] == 'delta':
            raise ValueError('Single-phase delta transformer outside named source scope')
    leadlag = general.get('leadlag', 'lag')  # HVLeadsLV=False, ANSI default
    if leadlag not in ('lag', 'lead', 'ansi', 'euro'):
        raise ValueError('Unsupported LeadLag')
    hv_leads = leadlag in ('lead', 'euro')
    if windings[0]['conn'] == windings[1]['conn']:
        direction = 1
    else:
        hv = max(range(2), key=lambda i: windings[i]['kv'])
        direction = (-1 if hv_leads else 1) if windings[hv]['conn'] == 'wye' else (1 if hv_leads else -1)
    for index, w in enumerate(windings):
        terminal = terminal_nodes(w['bus'], phases, transformer=True)
        incidence, coil_pairs = [], []
        for phase in range(phases):
            other = phases if w['conn'] == 'wye' or phases == 1 else (phase + direction) % phases
            row = [0] * (phases + 1)
            row[phase], row[other] = 1, -1
            incidence.append(row)
            coil_pairs.append([terminal['nodes'][phase], terminal['nodes'][other]])
        coil_kv = w['kv'] / math.sqrt(3.) if phases == 3 and w['conn'] == 'wye' else w['kv']
        w.update(index=index + 1, terminal=terminal, coil_incidence=incidence,
                 coil_node_pairs=coil_pairs, coil_voltage_base_kv=coil_kv,
                 coil_power_base_kva=w['kva'] / phases,
                 coil_impedance_base_ohm=coil_kv ** 2 / (windings[0]['kva'] / phases / 1000.),
                 tap_step=(w['maxtap'] - w['mintap']) / w['numtaps'] if w['numtaps'] else 0.,
                 property_sources=sources[str(index + 1)])
    control_names = [name for name, ctrl in controls.items()
                     if _effective(ctrl).get('transformer', {}).get('value', '').lower().removeprefix('transformer.') == obj['name'].split('.')[1]]
    lowbase = windings[1]['coil_impedance_base_ohm']
    return {'name': obj['name'], 'phases': phases, 'windings': windings,
            'xhl_percent': general['xhl'], 'delta_direction': direction, 'leadlag': leadlag,
            'leadlag_default_source': DEFAULT_SOURCE_URL if 'leadlag' not in general else None,
            'pair_leakage_on_winding2_coil_base_ohm': {
                're': lowbase * sum(w['%r'] for w in windings) / 100.,
                'im': lowbase * general['xhl'] / 100.},
            'ideal_coil_voltage_ratio_1_over_2': windings[0]['coil_voltage_base_kv'] / windings[1]['coil_voltage_base_kv'],
            'regcontrols': control_names,
            'tap_semantics': 'initial declaration only; enabled original RegControl recomputes operating taps per snapshot' if control_names else 'original declaration',
            'neutral_semantics': 'RNeut=-1 omits an internal grounding branch; terminal node 0 is still ground',
            'properties': deepcopy(obj['properties']),
            'model_boundary': 'coil connection and leakage data only; no Y-matrix inversion or automatic-control solution'}


def build_network_mapping(source_run):
    """Export the frozen input network; no scenario/result or winner is read."""
    run = Path(source_run).resolve()
    names = ('command_inventory.json', 'author_static_metadata.json', 'physical_config.json',
             'metadata.json', 'dependencies.json')
    data = {name: json.loads((run / name).read_text(encoding='utf-8')) for name in names}
    inventory, static, config, metadata = [data[n] for n in names[:4]]
    if _sha(DEFAULT_SOURCE) != DEFAULT_SOURCE_SHA256:
        raise ValueError('Pinned backend default source changed')
    # The source scope was built with this engine; do not transplant another version's defaults.
    if not isinstance(data['dependencies.json'], dict) or data['dependencies.json'].get('dss-python-backend') != '0.14.5':
        raise ValueError('Cannot bind transformer defaults to frozen backend 0.14.5')
    parsed = read_static_objects(inventory['entry_file'], allowed_root=inventory['allowed_root'],
                                 expected_files=inventory['files'])
    objects = parsed['objects']
    setup_text = AUTHOR_SETUP.read_text(encoding='utf-8')
    expected_statement = 'New Line.{tline[\'from node\']}{tline[\'to node\']} Bus1={tline[\'from node\']}{tline[\'from conn\']} Bus2={tline[\'to node\']}{tline[\'to conn\']} LineCode={tline[\'code\']} Length={str(tline[\'length\'])} units=kft'
    if expected_statement not in setup_text:
        raise ValueError('Original tie construction semantics changed')
    tie_line = next(i for i, line in enumerate(setup_text.splitlines(), 1) if expected_statement in line)
    tie_source = {'file': str(AUTHOR_SETUP), 'line': tie_line, 'sha256': _sha(AUTHOR_SETUP),
                  'parameter_source': 'author_static_metadata.json:ties'}
    for tie in static['ties']:
        name = 'line.' + tie['from node'] + tie['to node']
        if name in objects:
            raise ValueError('Dynamic tie collides with DSS declaration')
        values = [('bus1', tie['from node'] + tie['from conn']),
                  ('bus2', tie['to node'] + tie['to conn']), ('linecode', tie['code']),
                  ('length', str(tie['length'])), ('units', 'kft')]
        objects[name] = {'name': name, 'class': 'line', 'created': tie_source,
                         'properties': [{'key': k, 'value': v, 'source': deepcopy(tie_source)} for k, v in values]}
    lines, transformers, unsupported = [], [], []
    controls = {k: v for k, v in objects.items() if v['class'] == 'regcontrol'}
    for name, obj in objects.items():
        if obj['class'] not in ('line', 'transformer'):
            continue
        try:
            if obj['class'] == 'line':
                lines.append(_line(obj, objects, config['line_phase_ampacity_A']))
            else:
                transformers.append(_transformer(obj, controls))
        except (KeyError, ValueError) as exc:
            unsupported.append({'object': name, 'reason': str(exc), 'declaration': deepcopy(obj)})
    switch_names = ['line.' + item['line'].lower() for item in static['sectional_switches']] + [
        name.lower() for name in static['tie_actual_names']]
    for line in lines:
        if line['name'] in switch_names:
            index = switch_names.index(line['name'])
            line['switch_control'] = {'index_zero_based': index, 'command_bit': 10 + index,
                                      'action_id': 11 + index, 'normal_closed': index < len(static['sectional_switches'])}
        else:
            line['switch_control'] = None
    layouts = {k.lower(): v for k, v in metadata['element_layout'].items()}
    checks = []
    for line in lines:
        layout = layouts.get(line['name'])
        ok = layout is not None and layout['node_order'] == line['bus1_nodes'] + line['bus2_nodes']
        checks.append({'name': line['name'], 'check': 'frozen_parameter_node_order', 'passed': ok})
    for tr in transformers:
        layout = layouts.get(tr['name'])
        expected = [node for w in tr['windings'] for node in w['terminal']['nodes']]
        ok = layout is not None and layout['node_order'] == expected
        checks.append({'name': tr['name'], 'check': 'frozen_parameter_node_order', 'passed': ok})
        reference = {k.lower(): v for k, v in metadata['transformer_ratings'].items()}.get(tr['name'])
        rated = reference is not None and reference['phases'] == tr['phases'] and all(
            w['kv'] == saved['kv'] and w['kva'] == saved['kva'] and w['conn'] == saved['connection']
            for w, saved in zip(tr['windings'], reference['windings']))
        checks.append({'name': tr['name'], 'check': 'frozen_winding_ratings_and_connection', 'passed': rated})
    declared_lines = {k.lower() for k in static['original_lines']} | {k.lower() for k in static['tie_actual_names']}
    checks.append({'check': 'complete_declared_line_set', 'passed': {x['name'] for x in lines} == declared_lines})
    by_name = {x['name']: x for x in lines}
    checks.append({'check': 'original_switch_order_and_edges', 'passed': len(switch_names) == 9 and all(
        name in by_name and sorted([by_name[name]['bus1'], by_name[name]['bus2']]) == sorted(static['switch_edges'][i])
        for i, name in enumerate(switch_names))})
    checks.append({'check': 'complete_declared_transformer_set', 'passed': {x['name'] for x in transformers} ==
                   {'transformer.' + x.lower() for x in config['expected_transformer_kva']}})
    for command in parsed['non_object_commands']:
        if command['keyword'] not in ('set', 'solve', 'calcvoltagebases', 'buscoords'):
            unsupported.append({'command': command, 'reason': 'Unsupported nondeclaration network command'})
        elif command['keyword'] == 'set' and not re.match(r'set\s+voltagebases\s*=', command['text'], re.I):
            unsupported.append({'command': command, 'reason': 'Unsupported SET mutation'})
    bindings = {name: {'path': str(run / name), 'sha256': _sha(run / name)} for name in names}
    bindings.update({'author_DSS_CircuitSetup.py': {'path': str(AUTHOR_SETUP), 'sha256': _sha(AUTHOR_SETUP)},
                     'backend_Transformer.pas': {'path': str(DEFAULT_SOURCE), 'sha256': DEFAULT_SOURCE_SHA256,
                                                'url': DEFAULT_SOURCE_URL, 'version': '0.14.5'}})
    return {'schema': VERSION, 'status': 'STATIC_NETWORK_MAPPING_COMPLETE' if not unsupported and all(c['passed'] for c in checks) else 'BLOCKED_STATIC_NETWORK_MAPPING',
            'source_bindings': bindings, 'dss_files': parsed['source_files'],
            'lines': lines, 'transformers': transformers,
            'switch_line_order': switch_names,
            'buses': {'physical_phase_nodes': deepcopy(metadata['physical_phase_nodes_by_bus']),
                      'neutral_nodes': deepcopy(metadata['neutral_nodes_by_bus']),
                      'base_kv_ln_by_node': deepcopy(metadata['node_base_kv_ln']),
                      'source': 'frozen input metadata only; no tap result/plan/certificate read'},
            'circuit_declarations': [v for v in objects.values() if v['class'] == 'circuit'],
            'regcontrol_declarations': list(controls.values()),
            'unsupported': unsupported, 'static_checks': checks,
            'nonexecuted_commands': parsed['non_object_commands'],
            'optimization_model_ready': False,
            'remaining_model_contract': ['automatic regulator and other device proxy must be explicitly reviewed',
                'lossless balanced-angle coefficients are approximations, not original AC matrices',
                'line shunt/transformer connection data retained; any omission from optimizer must be declared'],
            'new_physical_points': 0, 'new_numerical_solutions': 0, 'new_training': 0}
