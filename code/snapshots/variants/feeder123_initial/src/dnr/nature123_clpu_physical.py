"""Explicitly gated, budgeted NC123 snapshots. Importing never loads DSS.

Each request resets the frozen author-derived circuit. Source Python is parsed,
never imported. Whole-element isolation and independent static regulator reset
are named extensions; the certificate is not a hardware or transient guarantee.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import warnings

import networkx as nx

from .dss_command_inventory import inventory_dss_commands
from .nature123_static_state import SparseRestorationGraph
from .nature123_clpu_environment import (CLPUEnvironment, DynamicState, _hash,
    CHECKS, RELIABILITY_CHECKS, CONSTRAINT_CHECKS)
from .nature123_acceptance_budget import AcceptanceBudgetStop
from .nature34_operating_physical import read_load_properties, verify_unchanged_attributes
from .restoration_ac_audit import audit_ac_snapshot
from .restoration_constraints import check_known_constraints
from .restoration_isolation import verify_isolation


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf8')).hexdigest()


def json_safe_evidence(value):
    """Preserve nonfinite evidence as named strings, never numerical zeroes."""
    paths = []
    def visit(item, path):
        if isinstance(item, float) and not math.isfinite(item):
            paths.append(path)
            return 'NaN' if math.isnan(item) else '+Infinity' if item > 0 else '-Infinity'
        if isinstance(item, dict): return {k: visit(v, path + '.' + str(k)) for k, v in item.items()}
        if isinstance(item, (tuple, list)): return [visit(v, path + '[' + str(i) + ']') for i, v in enumerate(item)]
        return item
    return visit(value, '$'), paths


def proposed_physical_config(project):
    """Pure proposal, using unchanged D034 comparison and audit tolerances."""
    old = json.loads((Path(project) / 'configs/nature34_migration_v1.json').read_text(encoding='utf8'))
    return {key: deepcopy(old[key]) for key in ('solver_convergence_tolerance',
        'voltage_limits_pu', 'constraint_tolerances', 'ac_audit_tolerances')} | {
        'frequency_hz': 60, 'mode': 'snapshot', 'max_iterations': 5000,
        'max_control_iterations': 5000,
        'tolerance_source': 'configs/nature34_migration_v1.json; unchanged D034',
        'isolation_current_tolerance_a': 1e-8, 'isolation_voltage_tolerance_pu': 1e-8}


class SourceBundle:
    """Validate all nine source bytes and model identity without engine access."""
    def __init__(self, project, model_directory='artifacts/nature123_model_v1'):
        self.project = Path(project).resolve()
        base = self.project / model_directory
        self.config = json.loads((base / 'model_config.json').read_text(encoding='utf8'))
        self.manifest = json.loads((base / 'manifest.json').read_text(encoding='utf8'))
        self.root = (self.project / self.config['derived_source_directory']).resolve()
        self.verify()
        tree = ast.parse((self.root / 'DSS_Initialize.py').read_text(encoding='utf-8-sig'))
        wanted = {'sectional_swt', 'tie_swt', 'generators', 'dispatch_loads', 'substatn_id'}
        self.author = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                key = node.targets[0].id
                if key in wanted:
                    self.author[key] = ast.literal_eval(node.value)
        if set(self.author) != wanted or [len(self.author[k]) for k in
                ('sectional_swt', 'tie_swt', 'generators', 'dispatch_loads')] != [13, 9, 13, 19]:
            raise ValueError('Author literal inventory differs')
        if self.author['substatn_id'] != '150':
            raise ValueError('Unexpected source bus')
        self.master = self.root / self.config['source_entry_file']
        self.inventory = inventory_dss_commands(self.master)
        # This exact source has one CalcVoltageBases and no hidden Solve/alias.
        if self.inventory['command_triggered_numerical_solution_entries'] != 1:
            raise ValueError('Frozen master numerical command inventory differs')
        self.commands = self._commands()
        self.line_reference = {}
        source_lines = self.master.read_text(encoding='utf-8-sig').splitlines() + list(self.commands)
        for text in source_lines:
            match = re.match(r'^\s*new\s+line\.(\w+)\s+(.*)', text, re.I)
            if not match: continue
            name, body = match.groups()
            props = {k.lower(): v.lower() for k, v in re.findall(r'([\w]+)\s*=\s*([^\s!]+)', body)}
            if not {'bus1', 'bus2', 'length'} <= set(props):
                raise ValueError('Unsupported source line properties: ' + name)
            self.line_reference[name.lower()] = props
        if len(self.line_reference) != 133 or sum('linecode' in p for p in self.line_reference.values()) != 125:
            raise ValueError('Incomplete named line parameter reference')

    def verify(self):
        if canonical_sha(self.config) != self.manifest['config_sha256']:
            raise ValueError('Model config changed')
        cfg = {k: v for k, v in self.config.items() if k != 'model_fingerprint'}
        expected = canonical_sha({'config': cfg,
            'derived_source_sha256': self.manifest['derived_source_sha256']})
        if expected != self.config['model_fingerprint'] or expected != self.manifest['model_fingerprint']:
            raise ValueError('Model fingerprint mismatch')
        for name, digest in self.manifest['derived_source_sha256'].items():
            path = (self.root / name).resolve()
            if path.parent != self.root or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('Frozen source changed: ' + name)

    def _commands(self):
        commands = ['Set Maxiterations=5000', 'Set maxcontroliter=5000']
        for s in self.author['sectional_swt']:
            commands.append(f"New swtcontrol.swSec{s['no']} SwitchedObj=Line.{s['line']} Normal=c SwitchedTerm=1 Action=c")
        for t in self.author['tie_swt']:
            prefix = f"New Line.{t['name']} Bus1={t['from node']}{t['from conn']} Bus2={t['to node']}{t['to conn']}"
            suffix = (' r1=1e-3 r0=1e-3 x1=0.000 x0=0.000 c1=0.000 c0=0.000 Length=0.001'
                      if t['name'] in ('Sw7', 'Sw8') else f" LineCode={t['code']} Length={t['length']}")
            commands += [prefix + suffix,
                f"New swtcontrol.swTie{t['no']} SwitchedObj=Line.{t['name']} Normal=o SwitchedTerm=1 Action=o"]
        for g in self.author['generators']:
            commands.append(f"New Generator.G{g['no']} bus1={g['bus']}{g['phaseconn']} Phases={g['numphase']} Kv={g['kV']} Kw={g['size']} Pf=0.8 Model=1")
        commands += [f'Redirect [{self.root / "Loadshapes.dss"}]',
                     'BatchEdit Load..* Daily=loadshape_multload']
        return tuple(commands)


def phase_connectivity(raw, metadata):
    """Conservative phase reachability from actual terminal NodeOrder.

    Matching wye coils propagate individually. A delta-delta transformer only
    propagates a complete three-phase set. Unsupported layouts are UNKNOWN,
    never silently skipped. This is connectivity, not an AC feasibility test.
    """
    reached, links, groups, required, unknown = set(), [], [], set(), []
    for e in raw.get('elements', []):
        if not e.get('enabled'):
            continue
        name = e['name'].lower(); kind = name.split('.')[0]
        n, nt = e['conductors'], e['terminals']
        order = e.get('node_order', [])
        if len(order) != n * nt or len(e['buses']) != nt or any(type(v) is not int or v not in (0, 1, 2, 3) for v in order):
            unknown.append(name + ': unsupported terminal layout'); continue
        terms = [[(bus.split('.')[0].lower(), node) for node in order[t*n:(t+1)*n] if node]
                 for t, bus in enumerate(e['buses'])]
        required.update(v for terminal in terms for v in terminal)
        if kind == 'vsource':
            if name != 'vsource.source' or terms[0] != [('150', 1), ('150', 2), ('150', 3)]:
                unknown.append(name + ': unexpected source')
            else:
                reached.update(terms[0])
        elif kind in ('line', 'transformer'):
            if nt != 2:
                unknown.append(name + ': requires two terminals'); continue
            if kind == 'transformer':
                ref = metadata['transformer_ratings'].get(name)
                if not ref or len(ref['windings']) != 2:
                    unknown.append(name + ': missing winding metadata'); continue
                conns = [w['connection'] for w in ref['windings']]
                if conns == ['delta', 'delta']:
                    if [len(set(t)) for t in terms] != [3, 3] or e['phases'] != 3:
                        unknown.append(name + ': unsupported delta layout')
                    else:
                        groups.append((set(terms[0]), set(terms[1])))
                    continue
                if conns != ['wye', 'wye']:
                    unknown.append(name + ': unsupported transformer connection'); continue
            if len(terms[0]) != len(terms[1]) or len(terms[0]) != e['phases']:
                unknown.append(name + ': unmatched phase conductors'); continue
            links.extend(zip(terms[0], terms[1]))
        elif kind not in ('load', 'generator', 'capacitor'):
            unknown.append(name + ': unsupported device')
    changed = True
    while changed:
        previous = len(reached)
        for a, b in links:
            if a in reached or b in reached:
                reached.update((a, b))
        for a, b in groups:
            if a <= reached: reached.update(b)
            if b <= reached: reached.update(a)
        changed = len(reached) != previous
    missing = sorted(required - reached)
    physical = metadata.get('physical_phase_nodes_by_bus', {})
    dead = set(metadata.get('deenergized_buses', []))
    inactive_on_powered = sorted((b, p) for b, nodes in physical.items() if b not in dead
                                 for p in nodes if (b, p) not in required)
    if inactive_on_powered:
        unknown.append('Powered-bus phase nodes without enabled terminal support require explicit voltage-scope review')
    return {'status': 'UNKNOWN' if unknown else 'FAIL' if missing else 'PASS',
        'unknown': unknown, 'unpowered_enabled_phase_nodes': [f'{b}.{p}' for b, p in missing],
        'inactive_phase_nodes_on_powered_buses': [f'{b}.{p}' for b, p in inactive_on_powered],
        'source_reachable_phase_nodes': [f'{b}.{p}' for b, p in sorted(reached)],
        'scope': 'TERMINAL_NODEORDER_CONNECTIVITY_NOT_AC_FEASIBILITY'}


def certify_snapshot(raw, metadata, physical_config):
    """Pure certificate adapter, also used for independently saved snapshots."""
    limits = physical_config['voltage_limits_pu']
    known = check_known_constraints(raw, metadata, voltage_min=limits[0], voltage_max=limits[1],
                                    **physical_config['constraint_tolerances'])
    ac = audit_ac_snapshot(raw, metadata, **physical_config['ac_audit_tolerances'])
    isolation = verify_isolation(raw, metadata)
    phase = phase_connectivity(raw, metadata)
    checks = {'numerical_convergence': known['numerical']['status'],
        'topology': known['topology']['status'], 'isolation': isolation['status'],
        'phase_connectivity': phase['status'], 'ac_balance': ac['status'],
        'voltage': known['voltage']['status'], 'line_phase_thermal': known['line_thermal']['status'],
        'transformer_winding': known['transformer_thermal']['status']}
    if (raw.get('control_actions_done') is not True or type(raw.get('control_iterations')) is not int
            or raw['control_iterations'] >= physical_config.get('max_control_iterations', 5000)):
        checks['numerical_convergence'] = 'UNKNOWN'
    reliable = (raw.get('converged') is True and all(checks[k] == 'PASS' for k in RELIABILITY_CHECKS)
                and all(checks[k] in ('PASS', 'FAIL') for k in CONSTRAINT_CHECKS)
                and known['gfm_active_limit']['status'] == 'PASS')
    return {'status': ('PASS' if all(checks[k] == 'PASS' for k in CONSTRAINT_CHECKS) else 'FAIL') if reliable else 'UNKNOWN',
        'reliable': reliable, 'checks': checks, 'known_constraints': known,
        'ac_audit': ac, 'isolation': isolation, 'phase_audit': phase,
        'ideal_switch_hardware_thermal_status': 'NOT_MODELED_NOT_A_PASS',
        'full_hardware_feasibility': 'NOT_ESTABLISHED'}


class Nature123CLPUPhysicalRuntime:
    evidence_kind = 'PHYSICAL'

    def __init__(self, project, *, budget, physical_config, enable_physical=False, engine=None,
                 static_manifest=None, model_directory='artifacts/nature123_model_v1'):
        if enable_physical is not True:
            raise PermissionError('NC123 physical runtime requires explicit execution authorization')
        self.bundle = SourceBundle(project, model_directory=model_directory)
        self.project, self.budget = self.bundle.project, budget
        self.physical_config = deepcopy(physical_config)
        proposed = proposed_physical_config(project)
        for key in ('solver_convergence_tolerance', 'constraint_tolerances', 'ac_audit_tolerances',
                    'voltage_limits_pu', 'frequency_hz'):
            if self.physical_config.get(key) != proposed[key]:
                raise ValueError('Unregistered physical policy: ' + key)
        for key in ('mode', 'max_iterations', 'max_control_iterations',
                    'isolation_current_tolerance_a', 'isolation_voltage_tolerance_pu'):
            if key in self.physical_config and self.physical_config[key] != proposed[key]:
                raise ValueError('Unregistered operating setting: ' + key)
            self.physical_config[key] = proposed[key]
        if static_manifest is None:
            pointer = json.loads((self.project / 'artifacts/epsr123_sparse_20261107/latest.json').read_text(encoding='utf8'))
            static_manifest = json.loads((Path(pointer['run']) / 'static_manifest.json').read_text(encoding='utf8'))
        self.static_manifest = deepcopy(static_manifest)
        self.graph = SparseRestorationGraph(self.static_manifest)
        self._validate_static_author_mapping()
        self.reference_metadata = None
        self.request_id = None
        self.mock_engine = engine is not None
        # Supplying a fake engine never produces evidence_kind PHYSICAL.
        self.evidence_kind = 'SIMULATED' if self.mock_engine else 'PHYSICAL'
        self.dss = engine if engine is not None else importlib.import_module('opendssdirect')

    def _validate_static_author_mapping(self):
        a = self.bundle.author
        names = tuple(('line.' + s['line']).lower() for s in a['sectional_swt']) + tuple(
            ('line.' + s['name']).lower() for s in a['tie_swt'])
        # Static manifest uses short line names, in source control order.
        if tuple(n.removeprefix('line.') for n in names) != self.graph.switch_names:
            raise ValueError('Sparse switch inventory differs from author')
        if tuple(n.lower() for n in a['dispatch_loads']) != self.graph.dispatch:
            raise ValueError('Sparse dispatch order differs from author')
        if self.graph.ns != 22 or self.graph.nl != 19 or len(self.graph.loads) != 91:
            raise ValueError('Sparse inventory size differs')

    def _numeric(self, operation, target, *args, count=1):
        token = self.budget.reserve_entry(operation=operation, count=count, request_id=self.request_id)
        try:
            value = target(*args)
        except BaseException:
            self.budget.finish_entry(token, returned=False)
            raise
        self.budget.finish_entry(token, returned=True)
        return value

    def _command(self, command):
        self.budget.check()
        if not isinstance(command, str) or any(x in command for x in ('\n', '\r', ';')):
            raise ValueError('Single literal DSS command required')
        token = command.split()[0].lower()
        if token not in ('new', 'set', 'edit', 'batchedit', 'redirect', 'makebuslist'):
            raise ValueError('Unbudgeted numerical or unsupported command')
        if token == 'redirect':
            exact = f'Redirect [{self.bundle.root / "Loadshapes.dss"}]'
            if command != exact or inventory_dss_commands(self.bundle.root / 'Loadshapes.dss')['command_triggered_numerical_solution_entries']:
                raise ValueError('Only frozen zero-entry loadshape Redirect is supported')
            self.bundle.verify()
        return self.dss.Text.Command(command)

    def _initialize(self):
        self.bundle.verify(); self.budget.check()
        d = self.dss
        d.Basic.ClearAll(); d.Basic.AllowForms(0)
        self._numeric('Compile:embedded_CalcVoltageBases', d.Text.Command,
            f'Compile [{self.bundle.master}]', count=self.bundle.inventory['command_triggered_numerical_solution_entries'])
        for command in self.bundle.commands:
            self._command(command)
        # Explicit snapshot / 60 Hz policy; daily shape remains attached but is
        # not applied by time advancement. CLPU is applied once as full P/Q.
        self._command('Set mode=snapshot frequency=60')
        d.Solution.Convergence(self.physical_config['solver_convergence_tolerance'])
        self._numeric('AuthorInitialization.Solution.Solve', d.Solution.Solve)
        self.initialization_readback = {'converged': bool(d.Solution.Converged()),
            'control_actions_done': bool(d.Solution.ControlActionsDone()),
            'control_iterations': d.Solution.ControlIterations()}
        if (not self.initialization_readback['converged'] or
                not self.initialization_readback['control_actions_done'] or
                self.initialization_readback['control_iterations'] >= self.physical_config['max_control_iterations']):
            raise ValueError('Author initialization numerical/control convergence not established')
        meta = self._metadata()
        if self.reference_metadata is None:
            self.reference_metadata = deepcopy(meta)
        elif meta != self.reference_metadata:
            raise ValueError('Independent reset changed physical metadata or initial setpoints')
        return meta

    def _metadata(self):
        d = self.dss; phases = {}; bases = {}; layouts = {}; ratings = {}
        allnames = [n.lower() for n in d.Circuit.AllElementNames()]
        if len(allnames) != len(set(allnames)):
            raise ValueError('Duplicate engine element identities')
        controls = [n for n in allnames if n.split('.')[0] in ('regcontrol', 'swtcontrol')]
        expected = [n for n in allnames if n not in controls]
        kinds = ('line', 'transformer', 'load', 'generator', 'capacitor', 'vsource')
        if any(n.split('.')[0] not in kinds for n in expected):
            raise ValueError('Uncaptured engine device class')
        counts = {k: sum(n.startswith(k + '.') for n in expected) for k in kinds}
        if counts != dict(line=133, transformer=8, load=91, generator=13, capacitor=4, vsource=1):
            raise ValueError('Engine physical inventory differs from source: ' + str(counts))
        if sum(n.startswith('swtcontrol.') for n in controls) != 22 or sum(n.startswith('regcontrol.') for n in controls) != 7:
            raise ValueError('Control inventory differs')
        for bus in d.Circuit.AllBusNames():
            d.Circuit.SetActiveBus(bus)
            nodes = [int(v) for v in d.Bus.Nodes()]
            if any(n not in (0, 1, 2, 3) for n in nodes):
                raise ValueError('Unsupported physical phase node: ' + bus)
            phases[bus.lower()] = sorted(n for n in nodes if n)
            bases[bus.lower()] = float(d.Bus.kVBase())
            if not math.isfinite(bases[bus.lower()]) or bases[bus.lower()] <= 0:
                raise ValueError('Missing/nonpositive initialized bus voltage base: ' + bus)
        if set(phases) != self.graph.nodes:
            raise ValueError('Engine bus inventory differs from sparse graph')
        for name in expected:
            d.Circuit.SetActiveElement(name)
            layouts[name] = {'kind': name.split('.')[0], 'buses': list(d.CktElement.BusNames()),
                'node_order': list(d.CktElement.NodeOrder()), 'conductors': d.CktElement.NumConductors(),
                'terminals': d.CktElement.NumTerminals()}
        line_readback = {}
        for short in d.Lines.AllNames():
            key = short.lower(); d.Lines.Name(short)
            ref = self.bundle.line_reference[key]
            actual = {'length': float(d.Lines.Length()), 'linecode': d.Lines.LineCode().lower(),
                      'buses': layouts['line.' + key]['buses']}
            endpoints = tuple(sorted(b.split('.')[0].lower() for b in actual['buses']))
            expected_endpoints = tuple(sorted(ref[k].split('.')[0] for k in ('bus1', 'bus2')))
            if endpoints != expected_endpoints or not math.isclose(actual['length'], float(ref['length']), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('Named source line geometry differs: ' + key)
            for expected_bus, actual_bus in zip((ref['bus1'], ref['bus2']), actual['buses']):
                # Explicit phase suffixes must survive parsing exactly; omitted
                # suffixes are audited through the actual NodeOrder separately.
                if '.' in expected_bus and actual_bus.lower() != expected_bus:
                    raise ValueError('Explicit source terminal phases differ: ' + key)
            if 'linecode' in ref and actual['linecode'] != ref['linecode']:
                raise ValueError('LineCode mapping differs: ' + key)
            if key in self.graph.original_lines and endpoints != self.graph.original_lines[key]:
                raise ValueError('Fault line mapping differs: ' + key)
            if key in self.graph.switch_names:
                index = self.graph.switch_names.index(key)
                if endpoints != self.graph.switch_edges[index]:
                    raise ValueError('Switch endpoint mapping differs: ' + key)
            line_readback[key] = actual
        d.LineCodes.Name('4')
        c4 = list(d.LineCodes.Cmatrix())
        if len(c4) != 9 or any(not math.isclose(c4[i], -0.585011253, rel_tol=1e-12, abs_tol=1e-10) for i in (1, 3)):
            raise ValueError('Corrected C4 mutual capacitance did not enter engine')
        regulator_targets = {}
        for name in controls:
            if name.startswith('regcontrol.'):
                d.RegControls.Name(name.split('.')[1])
                target = d.RegControls.Transformer().lower()
                target = target if target.startswith('transformer.') else 'transformer.' + target
                if target not in layouts:
                    raise ValueError('Unknown regulated transformer: ' + target)
                regulator_targets[name] = target
        expected_kva = {'reg1a': 5000, 'xfm1': 150, **{n: 2000 for n in
            ('reg2a', 'reg3a', 'reg3c', 'reg4a', 'reg4b', 'reg4c')}}
        for short in d.Transformers.AllNames():
            d.Transformers.Name(short); d.Circuit.SetActiveElement('transformer.' + short)
            windings = []
            for i in range(1, d.Transformers.NumWindings() + 1):
                d.Transformers.Wdg(i)
                windings.append({'kva': float(d.Transformers.kVA()), 'kv': float(d.Transformers.kV()),
                    'connection': 'delta' if d.Transformers.IsDelta() else 'wye'})
            if short.lower() not in expected_kva or len(windings) != 2 or any(w['kva'] != expected_kva[short.lower()] for w in windings):
                raise ValueError('Transformer winding rating differs: ' + short)
            ratings['transformer.' + short.lower()] = {'phases': d.CktElement.NumPhases(), 'windings': windings}
        loads = {k.lower(): v for k, v in read_load_properties(d).items()}
        if set(loads) != set(self.graph.loads):
            raise ValueError('Engine named load inventory differs')
        for name, row in loads.items():
            if row['buses'][0].split('.')[0].lower() != self.graph.loads[name][0] or not math.isclose(row['nominal_kw'], float(self.graph.loads[name][1]), abs_tol=1e-8, rel_tol=1e-12):
                raise ValueError('Author base demand differs: ' + name)
        gen = {f"generator.g{g['no']}": {'available_p_kw': float(g['size']),
            'reference_q_kvar': .75 * g['size'], 'bus': g['bus'] + g['phaseconn']}
            for g in self.bundle.author['generators']}
        line_names = set(n for n in expected if n.startswith('line.'))
        cfg = self.bundle.config
        ideal = ['line.' + n for n in cfg['ideal_switch_lines']]
        rated = {'line.' + n: {'phase_amps': v} for n, v in cfg['line_phase_ampacity_A'].items()}
        if line_names != set(ideal) | set(rated) or set(ideal) & set(rated):
            raise ValueError('Named phase rating inventory incomplete')
        return {'node_phases': phases, 'physical_phase_nodes_by_bus': phases,
            'neutral_nodes_by_bus': {b: [0] for b in phases}, 'node_base_kv_ln': bases,
            'source_buses': ['150'], 'allowed_source_elements': ['vsource.source'],
            'expected_element_names': expected, 'excluded_controls': controls, 'element_layout': layouts,
            'regulator_targets': regulator_targets,
            'line_parameter_readback': line_readback, 'linecode4_cmatrix_nf_per_length': c4,
            'transformer_ratings': ratings, 'load_reference': loads, 'generator_reference': gen,
            'line_config_by_name': {n: n for n in rated}, 'line_ratings': rated,
            'ideal_line_names': ideal, 'neutral_thermal_mode': 'NOT_MODELED',
            'gfm_limits': {n: {'p_min_kw': 0., 'p_max_kw': r['available_p_kw']} for n, r in gen.items()},
            'known_limitations': cfg['not_modeled'], 'isolation_current_tolerance_a': 1e-8,
            'isolation_voltage_tolerance_pu': 1e-8}

    def validate_request(self, request):
        if request.get('model_fingerprint') != self.bundle.config['model_fingerprint']:
            raise ValueError('Request model fingerprint differs')
        graph = SparseRestorationGraph(self.static_manifest, fault_lines=request['fault_lines'])
        env = CLPUEnvironment(graph, model_fingerprint=request['model_fingerprint'],
            condition=request['condition'], event=request['event'])
        expected = env.snapshot_request(DynamicState.from_dict(request['state']), request['time_seconds'])
        if expected != request:
            raise ValueError('Request SHA, history, CLPU multipliers or effective commands differ')
        cfg = request['condition']['operating_config']
        if not self.mock_engine and (cfg.get('physical_execution_allowed') is not True or
                cfg.get('status') != 'FROZEN_FOR_ACCEPTANCE'):
            raise PermissionError('Prepared proposal request is not an execution-frozen request')
        if cfg.get('algorithm') not in (0, 1) or type(cfg.get('algorithm')) is not int or cfg.get('frequency_hz') != 60 or cfg.get('voltage_policy_pu') != [.9, 1.1] or cfg.get('source_fed_only') is not True:
            raise ValueError('Request operating policy differs')
        return graph, graph.metadata(request['state']['command'])

    def _apply(self, request, graph, state, meta):
        d = self.dss
        for name in meta['excluded_controls']:
            self._command(f'Edit {name} enabled=no')
        planned = {}; powered = state.source_reachable_buses
        switches, _ = graph.decode(state.command)
        edges = [e for e in graph.edges if e not in graph.fault_edges and
            (e not in graph._switch_index or switches & (1 << graph._switch_index[e]))]
        network = nx.Graph(); network.add_nodes_from(sorted(graph.nodes)); network.add_edges_from(edges)
        for name, layout in meta['element_layout'].items():
            buses = [b.split('.')[0].lower() for b in layout['buses']]
            kind = layout['kind']; enabled = all(b in powered for b in buses)
            if kind in ('line', 'transformer'):
                enabled = enabled and len(set(buses)) == 2 and network.has_edge(*buses)
            elif kind == 'load':
                enabled = enabled and request['loads'][name]['enabled']
            planned[name] = bool(enabled)
            self._command(f"Edit {name} enabled={'yes' if enabled else 'no'}")
            if enabled and kind in ('line', 'transformer'):
                d.Circuit.SetActiveElement(name)
                for t in range(1, layout['terminals'] + 1): d.CktElement.Close(t, 0)
        before = {k.lower(): v for k, v in read_load_properties(d).items()}
        if before != meta['load_reference']:
            raise ValueError('Reset load properties changed')
        for name, row in request['loads'].items():
            ref = meta['load_reference'][name]
            cond = request['condition']['loads'][name]
            for key in ('nominal_kw', 'nominal_kvar'):
                if not math.isclose(cond[key], ref[key], rel_tol=1e-12, abs_tol=1e-8):
                    raise ValueError('Condition base P/Q differs from source: ' + name)
            self._command(f"Edit {name} kW={row['requested_kw']:.17g} kvar={row['requested_kvar']:.17g}")
        observed = {k.lower(): v for k, v in read_load_properties(d).items()}
        verify_unchanged_attributes(before, observed)
        for name, row in request['loads'].items():
            for key, sourcekey in (('nominal_kw', 'requested_kw'), ('nominal_kvar', 'requested_kvar')):
                if not math.isclose(observed[name][key], row[sourcekey], rel_tol=1e-12, abs_tol=1e-8):
                    raise ValueError('Full load setpoint readback differs: ' + name)
        der = request['condition']['der_setpoints']; ref = meta['generator_reference']
        if set(der) != set(ref): raise ValueError('Exactly all thirteen DER required')
        der_observed = {}
        for name, row in der.items():
            if row.get('bus') != ref[name]['bus'] or not 0 <= row['kw'] <= ref[name]['available_p_kw'] or not math.isclose(row['kvar'], .75 * row['kw'], rel_tol=1e-12, abs_tol=1e-8):
                raise ValueError('DER setpoint outside named source PF/capacity domain')
            self._command(f"Edit {name} kW={row['kw']:.17g} kvar={row['kvar']:.17g}")
            d.Generators.Name(name.split('.')[1])
            der_observed[name] = {'kw': float(d.Generators.kW()), 'kvar': float(d.Generators.kvar())}
            if any(not math.isclose(der_observed[name][k], row[k], rel_tol=1e-12, abs_tol=1e-8) for k in ('kw', 'kvar')):
                raise ValueError('DER readback differs')
        control_plan = {}
        for name in meta['excluded_controls']:
            enabled = planned[meta['regulator_targets'][name]] if name.startswith('regcontrol.') else False
            control_plan[name] = enabled
            self._command(f"Edit {name} enabled={'yes' if enabled else 'no'}")
        self._command('MakeBusList')
        meta.update(planned_enabled=planned, deenergized_buses=sorted(graph.nodes - powered))
        meta['planned_controls_enabled'] = control_plan
        return network, {'loads': observed, 'generators': der_observed}

    def _verify_postsolve(self, request, meta, before):
        d = self.dss
        loads = {k.lower(): v for k, v in read_load_properties(d).items()}
        verify_unchanged_attributes(meta['load_reference'], loads)
        if loads != before['loads']:
            raise ValueError('Load setpoint properties changed during solve')
        generators = {}
        for name, expected in before['generators'].items():
            d.Generators.Name(name.split('.')[1])
            generators[name] = {'kw': float(d.Generators.kW()), 'kvar': float(d.Generators.kvar())}
            if generators[name] != expected:
                raise ValueError('DER setpoint changed during solve')
        controls = {}
        for name, expected in meta['planned_controls_enabled'].items():
            d.Circuit.SetActiveElement(name)
            controls[name] = bool(d.CktElement.Enabled())
            if controls[name] != expected:
                raise ValueError('Control enabled state differs: ' + name)
        return {'loads': loads, 'generators': generators, 'controls_enabled': controls}

    def _snapshot(self, network):
        d = self.dss; elements = []
        bases = {}
        for bus in d.Circuit.AllBusNames():
            d.Circuit.SetActiveBus(bus)
            bases[bus.lower()] = float(d.Bus.kVBase())
        for name in d.Circuit.AllElementNames():
            kind = name.split('.')[0].lower()
            if kind in ('regcontrol', 'swtcontrol'): continue
            d.Circuit.SetActiveElement(name)
            n, nt = d.CktElement.NumConductors(), d.CktElement.NumTerminals()
            e = {'name': name.lower(), 'enabled': bool(d.CktElement.Enabled()),
                'buses': list(d.CktElement.BusNames()), 'conductors': n, 'terminals': nt,
                'phases': d.CktElement.NumPhases(), 'node_order': list(d.CktElement.NodeOrder()),
                'currents_real_imag': list(d.CktElement.Currents()),
                'voltages_real_imag': list(d.CktElement.Voltages()),
                'conductor_open': [[bool(d.CktElement.IsOpen(t, c)) for c in range(1, n+1)] for t in range(1, nt+1)]}
            e['powers_kw_kvar'] = list(d.CktElement.Powers())
            e['currents_mag_angle'] = list(d.CktElement.CurrentsMagAng())
            e['linear_yprim_check'] = kind in ('line', 'transformer', 'capacitor')
            if e['linear_yprim_check']: e['yprim_real_imag'] = list(d.CktElement.YPrim())
            if kind == 'transformer':
                d.Transformers.Name(name.split('.')[1]); e['winding_taps'] = []
                for w in range(1, d.Transformers.NumWindings()+1):
                    d.Transformers.Wdg(w); e['winding_taps'].append(d.Transformers.Tap())
            elements.append(e)
        return {'converged': bool(d.Solution.Converged()), 'iterations': d.Solution.Iterations(),
            'control_iterations': d.Solution.ControlIterations(), 'algorithm': d.Solution.Algorithm(),
            'control_actions_done': bool(d.Solution.ControlActionsDone()),
            'node_base_kv_ln': bases,
            'node_names': list(d.Circuit.AllNodeNames()), 'node_vmag_pu': list(d.Circuit.AllBusMagPu()),
            'node_voltages_real_imag': list(d.Circuit.AllBusVolts()),
            'circuit_total_power_kw_kvar': list(d.Circuit.TotalPower()), 'losses_w_var': list(d.Circuit.Losses()),
            'graph_nodes': sorted(network.nodes), 'graph_edges': [list(e) for e in network.edges],
            'is_forest': nx.is_forest(network), 'components': [sorted(c) for c in nx.connected_components(network)],
            'elements': elements, 'circuit_all_element_names': list(d.Circuit.AllElementNames())}

    def evaluate(self, request, point_id):
        request = deepcopy(request)
        graph, state = self.validate_request(request)
        self.budget.begin_request(str(point_id), request['request_sha256']); self.request_id = str(point_id)
        row = {'status': 'UNKNOWN', 'reliable': False, 'point_id': str(point_id),
            'request_sha256': request['request_sha256'], 'evidence_kind': self.evidence_kind,
            'checks': {k: 'UNKNOWN' for k in CHECKS}, 'request': request,
            'model_fingerprint': self.bundle.config['model_fingerprint'],
            'physical_certified': False}
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                meta = self._initialize()
                row['metadata'] = meta
                network, readback = self._apply(request, graph, state, meta)
                row['setpoint_readback'] = readback
                self.dss.Solution.Algorithm(request['condition']['operating_config']['algorithm'])
                self._numeric('Request.Solution.Solve', self.dss.Solution.Solve)
                raw = self._snapshot(network); row['raw_outputs'] = raw
                if (raw['control_actions_done'] is not True or
                        raw['control_iterations'] >= self.physical_config['max_control_iterations']):
                    raise ValueError('Regulator control iteration completion not established')
                for bus in state.source_reachable_buses:
                    value = raw['node_base_kv_ln'].get(bus)
                    if value is None or not math.isfinite(value) or value <= 0 or not math.isclose(
                            value, meta['node_base_kv_ln'][bus], rel_tol=1e-12, abs_tol=1e-12):
                        raise ValueError('Post-isolation active bus voltage base missing/changed: ' + bus)
                row['postsolve_readback'] = self._verify_postsolve(request, meta, readback)
                settings = {'algorithm': self.dss.Solution.Algorithm(), 'frequency_hz': self.dss.Solution.Frequency(),
                    'mode': self.dss.Solution.Mode(), 'max_iterations': self.dss.Solution.MaxIterations(),
                    'max_control_iterations': self.dss.Solution.MaxControlIterations(),
                    'convergence': self.dss.Solution.Convergence()}
                row['operating_readback'] = settings
                if settings != {'algorithm': request['condition']['operating_config']['algorithm'],
                    'frequency_hz': 60., 'mode': 0, 'max_iterations': 5000, 'max_control_iterations': 5000,
                    'convergence': self.physical_config['solver_convergence_tolerance']}:
                    raise ValueError('Runtime operating settings differ')
                row.update(self._certify(raw, meta))
                row['physical_certified'] = row['reliable'] and self.evidence_kind == 'PHYSICAL'
                row['warnings'] = [{'category': w.category.__name__, 'message': str(w.message)} for w in caught]
        except AcceptanceBudgetStop:
            raise
        except Exception as exc:
            row.update(status='UNKNOWN', reliable=False, physical_certified=False,
                       error={'type': type(exc).__name__, 'message': str(exc)})
        row['budget'] = self.budget.snapshot()
        row['budget']['engine_instrumentation_implemented'] = True
        row['initialization_readback'] = getattr(self, 'initialization_readback', None)
        safe, nonfinite = json_safe_evidence(row)
        if nonfinite:
            safe.update(status='UNKNOWN', reliable=False, physical_certified=False,
                nonfinite_evidence_paths=nonfinite,
                nonfinite_encoding='Explicit strings preserve NaN/+Infinity/-Infinity; never imputed')
        return safe

    def certify(self, request):
        return self.evaluate(request, point_id='request_' + str(len(self.budget.requests)))

    def _certify(self, raw, metadata):
        return certify_snapshot(raw, metadata, self.physical_config)
