"""Bounded NC13 CLPU snapshots preserving the author's single-ended opens.

Importing this module does not import OpenDSS or an author environment. Only the
explicit runtime constructor may initialize DSS, behind the supplied budget.
The old NC13/NC34 implementations and their archived results are not modified.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import importlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
from types import FunctionType
import warnings

import networkx as nx
import numpy as np

from .dss_command_inventory import inventory_dss_commands, _literal_include, _without_line_comment
from .nature13_physical import Nature13PhysicalRuntime
from .nature34_operating_physical import read_load_properties, verify_unchanged_attributes
from .nature34_clpu_physical import engine_numeric_lists, structured_feedback
from .restoration_ac_audit import audit_ac_snapshot
from .restoration_constraints import check_known_constraints


MODEL_SCOPE = 'NC13_CLPU_SINGLE_TERMINAL_OPEN_DEAD_DEVICES_ISOLATED_V1'
DISPATCH_LOADS = ('Load.671', 'Load.634a', 'Load.634b', 'Load.634c', 'Load.611', 'Load.652', 'Load.670b')
SWITCH_LINES = ('Line.632670', 'Line.671692', 'Line.646684', 'Line.633692')


class PhysicalBudgetStop(RuntimeError):
    """A pre-call limit rejected execution; not a numerical result."""


def inventory_nature13_commands(entry_file):
    """Pure file inventory, including the source's CalcV alias (not a solve)."""
    result = inventory_dss_commands(entry_file)
    aliases = []
    for visit in result['file_visits']:
        filename = Path(result['allowed_root']) / visit['file']
        block = False
        for number, line in enumerate(filename.read_text(encoding='utf-8-sig').splitlines(), 1):
            if not block and line.startswith('/*'):
                block = True
            if block:
                if '*/' in line:
                    block = False
                continue
            code = _without_line_comment(line)
            if code and code.split(None, 1)[0].lower() == 'calcv':
                aliases.append({'file': visit['file'], 'line': number, 'text': code})
    result['calcv_alias_entries'] = len(aliases)
    result['calcv_alias_locations'] = aliases
    result['calcvoltagebases_entries'] += len(aliases)
    result['command_triggered_numerical_solution_entries'] = (
        result['explicit_solve_commands'] + result['calcvoltagebases_entries'])
    return result


class Nature13Budget:
    """One worker-wide reservation counter; append callback must flush durably.

    ``deadline_monotonic`` is set by the worker, so constructor work counts in
    its wall limit. Reservations are charged before invoking DSS; an exception
    never refunds them. A killed Compile has unknown internal completion.
    """
    def __init__(self, *, max_queries=16, max_entries=256, deadline_monotonic,
                 emit=None, clock=time.monotonic):
        if max_queries < 1 or max_entries < 1 or not math.isfinite(deadline_monotonic):
            raise ValueError('Invalid bounded execution limits')
        self.max_queries, self.max_entries = int(max_queries), int(max_entries)
        self.deadline_monotonic, self.clock = float(deadline_monotonic), clock
        self.emit = emit or (lambda row: None)
        self.queries = self.reserved_entries = self.returned_entries = 0
        self.records, self.context, self.used_query_ids = [], {'phase': 'initialization'}, set()
        self.next_call_id = 0

    def check(self):
        if self.clock() >= self.deadline_monotonic:
            raise PhysicalBudgetStop('Worker wall-clock limit reached')

    def record(self, row):
        row = {**row, 'context': deepcopy(self.context)}
        self.records.append(row)
        self.emit(deepcopy(row))

    def begin_query(self, query_id):
        self.check()
        if query_id in self.used_query_ids:
            raise PhysicalBudgetStop('Repeated query id; automatic retry is forbidden')
        if self.queries >= self.max_queries:
            raise PhysicalBudgetStop('Physical query limit reached')
        self.used_query_ids.add(query_id)
        self.queries += 1
        self.context = {'phase': 'query', 'query_id': query_id, 'query_number': self.queries}
        self.record({'event': 'query_started'})

    def invoke(self, target, method, args, kwargs, counts):
        self.check()
        cost = int(counts.get('numerical_solution_entries', 0))
        if self.reserved_entries + cost > self.max_entries:
            self.record({'event': 'call_rejected', 'method': method, 'args': list(args),
                         'counts': counts, 'reason': 'NUMERICAL_ENTRY_BUDGET', 'underlying_api_invoked': False})
            raise PhysicalBudgetStop('Numerical solution entry limit would be exceeded')
        cid = self.next_call_id
        self.next_call_id += 1
        self.reserved_entries += cost
        self.record({'event': 'call_started', 'call_id': cid, 'method': method,
                     'args': list(args), 'counts': counts, 'underlying_api_invoked': False})
        try:
            value = target(*args, **kwargs)
        except BaseException as exc:
            self.record({'event': 'call_failed', 'call_id': cid, 'method': method,
                         'counts': counts, 'error': repr(exc), 'completion': 'UNKNOWN',
                         'underlying_api_invoked': True})
            raise
        self.returned_entries += cost
        self.record({'event': 'call_returned', 'call_id': cid, 'method': method,
                     'counts': counts, 'completion': 'API_RETURNED_NOT_CONVERGENCE',
                     'underlying_api_invoked': True})
        return value

    def stats(self):
        return {'queries': self.queries, 'reserved_numerical_entries': self.reserved_entries,
                'returned_numerical_entries': self.returned_entries,
                'maximum_queries': self.max_queries, 'maximum_numerical_entries': self.max_entries,
                'deadline_monotonic': self.deadline_monotonic}


class _CountedDSS:
    def __init__(self, target, budget, events, source_dir, inventories, prefix=''):
        self.target, self.budget, self.events = target, budget, events
        self.source_dir, self.inventories, self.prefix = source_dir, inventories, prefix

    def __getattr__(self, name):
        target = getattr(self.target, name)
        if not self.prefix and name in ('Solution', 'Text'):
            return _CountedDSS(target, self.budget, self.events, self.source_dir, self.inventories, name)
        if self.prefix == 'Solution' and callable(target) and name.lower().startswith('solve') and name != 'Solve':
            raise ValueError('Uninventoried solution API: ' + name)
        if (self.prefix, name) not in (('Solution', 'Solve'), ('Text', 'Command')):
            return target

        def call(*args, **kwargs):
            self.budget.check()
            counts = {'python_solve_commands': 0, 'script_solve_commands': 0,
                      'text_solve_commands': 0, 'compile_commands': 0,
                      'calcvoltagebases_entries': 0, 'numerical_solution_entries': 0}
            if self.prefix == 'Solution':
                counts['python_solve_commands'] = counts['numerical_solution_entries'] = 1
            else:
                command = args[0] if len(args) == 1 and not kwargs else None
                if not isinstance(command, str) or any(c in command for c in (';', '\n', '\r')):
                    raise ValueError('Only one literal Text.Command is supported')
                code = _without_line_comment(command)
                parts = code.split(None, 1)
                token = parts[0].lower() if parts else ''
                if token in ('compile', 'redirect'):
                    if len(parts) != 2:
                        raise ValueError('Include needs a literal source')
                    path = Path(_literal_include(parts[1]))
                    path = (self.source_dir / path).resolve() if not path.is_absolute() else path.resolve()
                    key = path.name.lower()
                    if key not in self.inventories or path.parent != self.source_dir.resolve():
                        raise ValueError('Unfrozen include command')
                    actual = inventory_nature13_commands(path)
                    expected = self.inventories[key]
                    if ({k: v['sha256'] for k, v in actual['files'].items()} !=
                            {k: v['sha256'] for k, v in expected['files'].items()}):
                        raise ValueError('DSS include source changed before execution')
                    if token == 'compile' and key != 'ieee13nodeckt.dss':
                        raise ValueError('Only the frozen master may be compiled')
                    if token == 'redirect' and key != 'loadshapes.dss':
                        raise ValueError('Only frozen zero-solution Loadshapes redirect is allowed')
                    counts['compile_commands'] = int(token == 'compile')
                    counts['script_solve_commands'] = actual['explicit_solve_commands']
                    counts['calcvoltagebases_entries'] = actual['calcvoltagebases_entries']
                    counts['numerical_solution_entries'] = actual['command_triggered_numerical_solution_entries']
                elif token == 'solve':
                    counts['text_solve_commands'] = counts['numerical_solution_entries'] = 1
                elif token in ('calcv', 'calcvoltagebases'):
                    counts['calcvoltagebases_entries'] = counts['numerical_solution_entries'] = 1
                elif token not in ('set', 'new', 'edit', 'batchedit', 'open', 'close', 'makebuslist') and not (
                        token.startswith(('swtcontrol.', 'load.')) and '=' in code):
                    raise ValueError('Command outside audited NC13 scope: ' + token)
            method = self.prefix + '.' + name
            row = {'method': method, 'args': list(args), 'counts': dict(counts), 'status': 'REQUESTED'}
            self.events.append(row)
            try:
                result = self.budget.invoke(target, method, args, kwargs, counts)
            except BaseException as exc:
                row.update(status='FAILED_OR_NOT_INVOKED', error=repr(exc))
                raise
            row['status'] = 'RETURNED'
            return result
        return call


def validate_multipliers(reference, multipliers):
    if not isinstance(multipliers, Mapping) or set(reference) != set(multipliers) or len(reference) != 15:
        raise ValueError('Exactly the full fifteen source-named load multipliers are required')
    result = {}
    for name, ref in reference.items():
        value = multipliers[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 1:
            raise ValueError('Invalid CLPU multiplier: ' + name)
        result[name] = {**ref, 'clpu_multiplier': float(value),
                        'requested_kw': ref['nominal_kw'] * value,
                        'requested_kvar': ref['nominal_kvar'] * value}
    return result


def plan_isolation(layouts, graph, source_bus, dispatch, open_lines):
    """Single-ended open stubs stay enabled iff a closed end is powered.

    A disabled dead-side stub has no closed live terminal; energized stubs keep
    their source parameters and contacts. A live/dead closed connection rejects
    the plan rather than hiding it by disabling an energized element.
    """
    powered = set(nx.node_connected_component(graph, source_bus))
    dead, enabled, contacts = set(graph) - powered, {}, {}
    for name, layout in layouts.items():
        buses = [b.split('.')[0].lower() for b in layout['buses']]
        kind = layout['kind']
        opened = [[False] * layout['conductors'] for _ in range(layout['terminals'])]
        if name.lower() in open_lines:
            opened[0] = [True] * layout['conductors']
        contacts[name] = opened
        if kind in ('line', 'transformer'):
            closed_buses = [b for b, mask in zip(buses, opened) if not all(mask)]
            has_live, has_dead = bool(set(closed_buses) & powered), bool(set(closed_buses) & dead)
            if has_live and has_dead:
                raise ValueError('Closed live/dead terminal conflict: ' + name)
            enabled[name] = has_live
        else:
            enabled[name] = all(b in powered for b in buses)
            if kind == 'load':
                enabled[name] &= bool(dispatch.get(name.lower(), 1))
    return {'planned_enabled': enabled, 'planned_open_contacts': contacts,
            'deenergized_buses': sorted(dead), 'powered_buses': sorted(powered),
            'single_ended_open_lines': sorted(open_lines)}


def verify_nature13_isolation(raw, metadata):
    """Pure NC13 certificate; explicit single-terminal opens are not deleted."""
    unknown, bad, disabled_checks, observed = [], [], [], {}
    norm = lambda values: {str(k).lower(): v for k, v in values.items()}
    phases = norm(metadata.get('node_phases', {}))
    layouts, planned = norm(metadata.get('element_layout', {})), norm(metadata.get('planned_enabled', {}))
    contact_plan = norm(metadata.get('planned_open_contacts', {}))
    expected = {n.lower() for n in metadata.get('expected_element_names', [])}
    actual = {e['name'].lower(): e for e in raw.get('elements', [])}
    graph = nx.Graph()
    graph.add_nodes_from(raw.get('graph_nodes', []))
    graph.add_edges_from(raw.get('graph_edges', []))
    source = metadata.get('source_buses', [])
    if len(source) != 1 or source[0] not in graph or not nx.is_forest(graph):
        unknown.append('Missing source or invalid forest graph')
        powered = set()
    else:
        powered = set(nx.node_connected_component(graph, source[0]))
    dead = set(graph) - powered
    if set(phases) != set(graph) or set(metadata.get('deenergized_buses', [])) != dead:
        bad.append({'type': 'physical_graph_dead_inventory_mismatch'})
    if not expected or set(actual) != expected or set(layouts) != expected or set(planned) != expected or set(contact_plan) != expected:
        unknown.append('Physical/layout/plan inventory mismatch')
    if len(actual) != len(raw.get('elements', [])):
        unknown.append('Duplicate raw electrical elements')
    current_tol = metadata.get('isolation_current_tolerance_a', 1e-8)
    voltage_tol = metadata.get('isolation_voltage_tolerance_pu', 1e-8)
    finite = lambda v: isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
    if not finite(current_tol) or not finite(voltage_tol) or min(current_tol,voltage_tol) < 0:
        raise ValueError('Invalid isolation tolerance')
    enabled_edges, sources = set(), set()
    for name, elem in actual.items():
        if name not in layouts or name not in planned or name not in contact_plan:
            continue
        layout = layouts[name]
        buses = [b.split('.')[0].lower() for b in layout['buses']]
        ncond, nterm = layout['conductors'], layout['terminals']
        if [b.lower() for b in elem.get('buses', [])] != [b.lower() for b in layout['buses']]:
            bad.append({'type': 'terminal_bus_layout_mismatch', 'element': name})
        if type(elem.get('enabled')) is not bool or elem['enabled'] != planned[name]:
            bad.append({'type': 'actual_enabled_disagrees_with_plan', 'element': name})
        currents = elem.get('currents_real_imag', [])
        if len(currents) != 2*ncond*nterm or not all(finite(v) for v in currents):
            unknown.append('Missing/nonfinite physical currents: ' + name)
            continue
        if not elem.get('enabled'):
            largest = max((math.hypot(currents[i], currents[i+1]) for i in range(0, len(currents), 2)), default=0.)
            disabled_checks.append({'element': name, 'max_current_a': largest, 'tolerance_a': current_tol})
            if largest > current_tol:
                bad.append({'type': 'disabled_nonzero_current', 'element': name, 'current_a': largest})
            continue
        if (elem.get('conductors'), elem.get('terminals'), elem.get('node_order')) != (ncond, nterm, layout['node_order']):
            bad.append({'type': 'enabled_terminal_node_layout_mismatch', 'element': name})
        opened = elem.get('conductor_open')
        if (not isinstance(opened,list) or len(opened)!=nterm or
                any(not isinstance(mask,list) or len(mask)!=ncond or any(type(v) is not bool for v in mask) for mask in opened)):
            unknown.append('Missing/malformed enabled contact mask: '+name)
            continue
        if opened != contact_plan[name]:
            bad.append({'type': 'saved_contact_pattern_disagrees_with_original_single_open_plan', 'element': name})
            continue
        for terminal, bus in enumerate(buses):
            if bus in dead and not all(opened[terminal]):
                bad.append({'type': 'closed_enabled_terminal_touches_dead_bus', 'element': name, 'terminal': terminal+1})
            for conductor, is_open in enumerate(opened[terminal]):
                if is_open:
                    index = 2*(terminal*ncond+conductor)
                    value = math.hypot(currents[index], currents[index+1])
                    if value > current_tol:
                        bad.append({'type': 'open_contact_nonzero_current', 'element': name, 'current_a': value})
        kind = layout['kind']
        if kind in ('line', 'transformer') and not any(v for mask in opened for v in mask):
            enabled_edges.add(tuple(sorted(set(buses))))
        if kind == 'vsource':
            sources.add(name)
    if sources != {n.lower() for n in metadata.get('allowed_source_elements', ['Vsource.source'])}:
        bad.append({'type': 'unexpected_or_missing_source', 'actual': sorted(sources)})
    main_edges = {tuple(sorted(e)) for e in graph.edges if set(e) <= powered}
    if enabled_edges != main_edges:
        bad.append({'type': 'closed_enabled_PD_main_graph_mismatch'})
    names, volts = raw.get('node_names', []), raw.get('node_vmag_pu', [])
    if len(names) != len(volts) or len({n.lower() for n in names}) != len(names):
        unknown.append('Invalid solved node inventory')
    voltage_map = dict(zip((n.lower() for n in names), volts))
    for node,value in voltage_map.items():
        try:
            bus,phase_text=node.rsplit('.',1)
            phase=int(phase_text)
        except (ValueError,AttributeError):
            unknown.append('Malformed solved node: '+str(node))
            continue
        if bus not in phases or phase<0 or (phase>0 and phase not in phases[bus]):
            bad.append({'type':'undeclared_solved_node','node':node})
        if not finite(value):
            unknown.append('Nonfinite solved node voltage: '+node)
        elif value<0 or (phase==0 and value>voltage_tol):
            bad.append({'type':'invalid_voltage_magnitude_or_ground_reference','node':node,'value_pu':value})
    for bus in sorted(dead):
        for phase in phases.get(bus, []):
            if not phase:
                continue
            node = f'{bus}.{phase}'
            if node not in voltage_map:
                observed[node] = 'STRUCTURALLY_EXCLUDED_NOT_SOLVED'
            elif not finite(voltage_map[node]):
                observed[node] = 'UNKNOWN_NONFINITE'
                unknown.append('Nonfinite dead node voltage: ' + node)
            elif abs(voltage_map[node]) > voltage_tol:
                observed[node] = 'UNSOURCED_NONZERO_VOLTAGE'
                bad.append({'type': 'dead_node_nonzero_voltage', 'node': node, 'value_pu': voltage_map[node]})
            else:
                observed[node] = 'OBSERVED_FINITE_ZERO'
    if raw.get('converged') is not True:
        unknown.append('Solver convergence not established')
    status = 'UNKNOWN' if unknown else 'FAIL' if bad else 'PASS'
    return {'status': status, 'unknowns': unknown, 'violations': bad,
            'excluded_node_states': observed if status == 'PASS' else {}, 'observed_node_states': observed,
            'computed_deenergized_buses': sorted(dead), 'powered_buses': sorted(powered),
            'certified_deenergized_buses': sorted(dead) if status == 'PASS' else [],
            'disabled_current_checks': disabled_checks,
            'scope': MODEL_SCOPE, 'limitations': ['Single-ended open charging retained on powered stubs',
                'Absent dead nodes are structurally excluded, not fabricated zero measurements',
                'No field deenergization, transient, neutral hardware or complete inverter capability claim']}


# New binding, unchanged code object/defaults/thresholds. The original module's
# global namespace is never patched; only this model's isolation hook differs.
_constraint_globals = dict(check_known_constraints.__globals__)
_constraint_globals['verify_isolation'] = verify_nature13_isolation
check_nature13_constraints = FunctionType(check_known_constraints.__code__, _constraint_globals,
    'check_nature13_constraints', check_known_constraints.__defaults__, check_known_constraints.__closure__)
check_nature13_constraints.__kwdefaults__ = deepcopy(check_known_constraints.__kwdefaults__)
check_nature13_constraints.__annotations__ = deepcopy(check_known_constraints.__annotations__)


class Nature13CLPUPhysicalRuntime(Nature13PhysicalRuntime):
    def __init__(self, project, run, physical_config, *, budget, events=None):
        if not isinstance(budget, Nature13Budget):
            raise TypeError('An explicit worker-wide Nature13Budget is required before initialization')
        self.project, self.run = Path(project), Path(run)
        self.physical_config, self.budget = deepcopy(physical_config), budget
        self.events = [] if events is None else events
        self.budget.check()
        pointer = json.loads((self.project/'artifacts/restoration_shortlist_sources_latest.json').read_text())
        source = self.project/pointer['derived_directory']/'nature_author_source'
        relative = Path('Environments/DSSdirect_13bus_loadandswitching')
        self.work = self.run/'work'
        destination = self.work/relative
        shutil.copytree(source/'RL_Training'/relative, destination)
        shutil.copy2(source/'RL_Testing/13bus/IEEELineCodes.dss', destination/'IEEELineCodes.dss')
        inventories = {name.lower(): inventory_nature13_commands(destination/name)
                       for name in ('IEEE13Nodeckt.dss', 'Loadshapes.dss')}
        main = inventories['ieee13nodeckt.dss']
        if (main['explicit_solve_commands'], main['calcvoltagebases_entries']) != (1, 1):
            raise ValueError('Pinned source must have one Solve and one CalcV entry')
        if inventories['loadshapes.dss']['command_triggered_numerical_solution_entries']:
            raise ValueError('Loadshapes redirect must contain no numerical entries')
        package = 'Environments.DSSdirect_13bus_loadandswitching.'
        if any(n.startswith(package) for n in sys.modules):
            raise RuntimeError('Only one NC13 runtime per process; no pre-imported author environment')
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(self.work))
        import opendssdirect as dss  # explicit runtime boundary, never module import
        self.dss = _CountedDSS(dss, budget, self.events, destination, inventories)
        self.engine_version = dss.Basic.Version()
        setup = importlib.import_module(package+'DSS_CircuitSetup')
        setup.dss = self.dss  # Installed before DSS_Initialize import executes.
        self.core = importlib.import_module(package+'DSS_Initialize')
        self.initial_metadata = self._metadata()
        if (tuple(self.core.dispatch_loads) != DISPATCH_LOADS or len(self.core.AllSwitches) != 4
                or tuple(s['edge name'].lower() for s in self.core.AllSwitches) != tuple(n.lower() for n in SWITCH_LINES)):
            raise ValueError('Source action layout differs')
        self.base_load_properties = read_load_properties(self.dss)
        if abs(sum(r['available_p_kw'] for r in self.initial_metadata['generator_reference'].values())-2640.) > 1e-8:
            raise ValueError('Original seven-DER availability differs from 2640 kW')
        for name, ref in self.initial_metadata['generator_reference'].items():
            self.dss.Generators.Name(name.split('.',1)[1])
            if (not math.isclose(self.dss.Generators.kW(),ref['available_p_kw'],rel_tol=1e-12,abs_tol=1e-8)
                    or not math.isclose(self.dss.Generators.kvar(),ref['reference_q_kvar'],rel_tol=1e-12,abs_tol=1e-8)
                    or not math.isclose(self.dss.Generators.PF(),.8,rel_tol=1e-12,abs_tol=1e-8)):
                raise ValueError('Source DER P/Q/pf differs: '+name)
        loads = {name: {'bus': ref['buses'][0].split('.')[0].lower(),
                        'nominal_kw': ref['nominal_kw'], 'nominal_kvar': ref['nominal_kvar']}
                 for name, ref in self.base_load_properties.items()}
        if len(loads) != 15 or abs(sum(r['nominal_kw'] for r in loads.values())-3466.) > 1e-8:
            raise ValueError('Original fifteen-load inventory or 3466 kW total differs')
        layouts = {}
        for name in self.initial_metadata['expected_element_names']:
            self.dss.Circuit.SetActiveElement(name)
            layouts[name] = {'buses': list(self.dss.CktElement.BusNames()),
                'node_order': list(self.dss.CktElement.NodeOrder()),
                'conductors': self.dss.CktElement.NumConductors(), 'terminals': self.dss.CktElement.NumTerminals(),
                'kind': name.split('.')[0].lower()}
        self.initial_metadata.update(load_reference=loads, element_layout=layouts,
            physical_phase_nodes_by_bus=deepcopy(self.initial_metadata['node_phases']),
            neutral_nodes_by_bus={b:[0] for b in self.initial_metadata['node_phases']},
            allowed_source_elements=['Vsource.source'])
        self.metadata = {'model': MODEL_SCOPE, 'source': pointer, 'engine_version': self.engine_version,
            'switches': self.core.AllSwitches, 'dispatch_loads': list(DISPATCH_LOADS),
            'compile_inventory': main, 'all_include_inventories': inventories,
            'initialization_counts': budget.stats(), 'alpha': .375,
            'constraint_checker_code_identity': check_nature13_constraints.__code__ is check_known_constraints.__code__,
            'constraint_source_sha256': hashlib.sha256(Path(check_known_constraints.__code__.co_filename).read_bytes()).hexdigest(),
            'full_hardware_feasibility': 'NOT_ESTABLISHED'}

    def evaluate_point(self, action, case, alpha, algorithm, multipliers, *, query_id,
                       time_minutes=0., state_identity=None):
        # Reset before even reserving a query: a rejected request must never
        # expose the preceding query's raw evidence as its own failed point.
        self.last_query_evidence = {'query_id':query_id, 'phase':'request_received',
            'request':deepcopy({'action':action,'case':case,'alpha':alpha,'algorithm':algorithm,
                'multipliers':multipliers,'time_minutes':time_minutes,'state_identity':state_identity}),
            'evidence_status':'PARTIAL_NOT_CERTIFIED'}
        self.budget.begin_query(query_id)
        bits = np.asarray(action)
        if bits.shape != (11,) or not np.isin(bits, [0,1]).all() or bits[2]:
            raise ValueError('Expected eleven binary bits with tie1 open')
        if alpha != .375 or algorithm not in (0,1):
            raise ValueError('Only frozen alpha=.375 and Normal/Newton are permitted')
        requests = validate_multipliers(self.initial_metadata['load_reference'], multipliers)
        faults = self.fault_edges(case['fault_lines'])
        graph = self.core.G_init.copy()
        graph.remove_edges_from(faults)
        fault_set = {frozenset(e) for e in faults}
        open_lines = {n.lower() for n in case['fault_lines']}
        for bit, edge, line in zip(bits[:4], self.core.SwitchLines, SWITCH_LINES):
            if bit and frozenset(edge) in fault_set:
                raise ValueError('Faulted switch requested closed')
            if not bit:
                open_lines.add(line.lower())
                if graph.has_edge(*edge):
                    graph.remove_edge(*edge)
        if not nx.is_forest(graph):
            raise ValueError('Nonforest graph')
        dispatch = {n.lower():int(b) for n,b in zip(DISPATCH_LOADS,bits[4:])}
        plan = plan_isolation(self.initial_metadata['element_layout'], graph, 'sourcebus', dispatch, open_lines)
        self.last_query_evidence.update(phase='plan_built',metadata_override=deepcopy(plan))
        before, counts_before = len(self.events), self.budget.stats()
        self.last_query_evidence['budget_before'] = deepcopy(counts_before)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            obj, _, _ = self.core.initialize()
            readback_before = read_load_properties(obj.dss)
            self.last_query_evidence.update(phase='base_readback_acquired',
                                           load_readback={'before':deepcopy(readback_before)})
            if readback_before != self.base_load_properties:
                raise ValueError('Independent reset did not restore original load properties')
            for name, row in requests.items():
                obj.dss.Text.Command(f"Edit {name} kW={row['requested_kw']:.15g} kvar={row['requested_kvar']:.15g}")
            # The same action path serves old references and new partial states.
            i = obj.dss.SwtControls.First()
            while i > 0:
                obj.dss.Text.Command('Swtcontrol.'+obj.dss.SwtControls.Name()+'.Action='+('c' if bits[i-1] else 'o'))
                i = obj.dss.SwtControls.Next()
            obj.dss.Solution.Solve()
            for name, bit in zip(DISPATCH_LOADS, bits[4:]):
                if not bit:
                    obj.dss.Text.Command(name+'.enabled="False"')
                obj.dss.Solution.Solve()
            for edge in faults:
                label = self.core.G_init.edges[edge]['label'][0]
                obj.dss.Text.Command(f'Open {label} term=1')
                obj.dss.Solution.Solve()
            for name, enabled in plan['planned_enabled'].items():
                if not enabled:
                    obj.dss.Text.Command(f'Edit {name} enabled=no')
            # A control on a disabled device cannot reactivate a dead island.
            for control in self.initial_metadata['excluded_controls']:
                obj.dss.Circuit.SetActiveElement(control)
                if control.lower().startswith('regcontrol.'):
                    controlled = obj.dss.Properties.Value('Transformer')
                    controlled = controlled if controlled.lower().startswith('transformer.') else 'Transformer.'+controlled
                else:
                    controlled = obj.dss.Properties.Value('SwitchedObj')
                target = next((n for n in plan['planned_enabled'] if n.lower()==controlled.lower()), None)
                if target is None:
                    raise ValueError('Unmapped control target: '+control)
                if not plan['planned_enabled'][target]:
                    obj.dss.Text.Command(f'Edit {control} enabled=no')
            obj.dss.Solution.Solve()
            for name, ref in self.initial_metadata['generator_reference'].items():
                obj.dss.Text.Command(f"Edit {name} kW={alpha*ref['available_p_kw']:.12g} kvar={alpha*ref['reference_q_kvar']:.12g}")
            obj.dss.Solution.Algorithm(int(algorithm))
            obj.dss.Solution.Convergence(self.physical_config['solver_convergence_tolerance'])
            obj.dss.Solution.Solve()
            captured_raw = self.extended_snapshot(graph)
            # Keep the actual captured object before any readback or checker
            # can fail. Nothing here replaces nonfinite values or recasts FAIL.
            self.last_query_evidence.update(phase='raw_acquired',raw_outputs=captured_raw,
                                           budget_after_capture=deepcopy(self.budget.stats()))
            raw = engine_numeric_lists(captured_raw)
            observed = read_load_properties(obj.dss)
            self.last_query_evidence['load_readback']['postsolve'] = deepcopy(observed)
            self.last_query_evidence['phase'] = 'postsolve_readback_acquired'
            verify_unchanged_attributes(readback_before, observed)
            if set(observed) != set(requests):
                raise ValueError('Full load readback inventory differs')
            for name, request in requests.items():
                for actual_key, expected_key in [('nominal_kw','requested_kw'),('nominal_kvar','requested_kvar')]:
                    if not math.isclose(observed[name][actual_key], request[expected_key], rel_tol=1e-12,abs_tol=1e-8):
                        raise ValueError('Load setpoint readback differs: '+name)
            raw_generators = {e['name'].lower():e for e in raw['elements'] if e['name'].lower().startswith('generator.')}
            if set(raw_generators) != {n.lower() for n in self.initial_metadata['generator_reference']}:
                raise ValueError('Complete seven-DER readback inventory differs')
            for name, ref in self.initial_metadata['generator_reference'].items():
                observed_der = raw_generators[name.lower()]['operating_setpoint']
                for field, expected in [('kw',alpha*ref['available_p_kw']),('kvar',alpha*ref['reference_q_kvar'])]:
                    if not math.isclose(observed_der[field],expected,rel_tol=1e-12,abs_tol=1e-8):
                        raise ValueError('DER operating setpoint readback differs: '+name)
        metadata = engine_numeric_lists({**self.initial_metadata, **plan})
        self.last_query_evidence['phase'] = 'checking_isolation'
        isolation = verify_nature13_isolation(raw, metadata)
        self.last_query_evidence.update(isolation=deepcopy(isolation),phase='checking_known_constraints')
        known = check_nature13_constraints(raw, metadata, **self.physical_config['constraint_tolerances'])
        self.last_query_evidence.update(known_constraints=deepcopy(known),phase='checking_ac')
        ac = audit_ac_snapshot(raw, metadata, **self.physical_config['ac_audit_tolerances'])
        self.last_query_evidence.update(ac_audit=deepcopy(ac),phase='building_result')
        required = ('numerical','topology','voltage','line_thermal','transformer_thermal','gfm_active_limit')
        reliable = (raw['converged'] and ac['status']==isolation['status']=='PASS' and
            known['numerical']['status']==known['topology']['status']=='PASS' and
            not any(known[k]['status']=='UNKNOWN' or known[k].get('unknown') for k in required))
        ledger = {'denominator_nominal_kw':3466., 'energized_nominal_kw':0., 'deenergized_nominal_kw':0.,
                  'deliberately_shed_nominal_kw':0., 'loads':[]}
        for name, ref in self.initial_metadata['load_reference'].items():
            state = 'energized' if plan['planned_enabled'][name] else ('deenergized' if ref['bus'] in plan['deenergized_buses'] else 'deliberately_shed')
            ledger[state+'_nominal_kw'] += ref['nominal_kw']
            ledger['loads'].append({'name':name,**ref,'state':state,'clpu_multiplier':multipliers[name]})
        ledger['nominal_restored_fraction'] = ledger['energized_nominal_kw']/3466.
        enabled_der = {n:r for n,r in self.initial_metadata['generator_reference'].items() if plan['planned_enabled'][n]}
        generation_ledger = {'total_installed_available_p_kw':2640.,
            'enabled_available_p_kw':sum(r['available_p_kw'] for r in enabled_der.values()),
            'enabled_requested_p_kw':alpha*sum(r['available_p_kw'] for r in enabled_der.values()),
            'enabled_requested_q_kvar':alpha*sum(r['reference_q_kvar'] for r in enabled_der.values()),
            'readback_checked_generators':7, 'complete_inverter_capability_certified':False}
        after = self.budget.stats()
        row = {'query_id':query_id,'case':case['name'],'fault_lines':list(case['fault_lines']),
            'action':bits.astype(int).tolist(),'alpha':alpha,'algorithm':algorithm,'time_minutes':float(time_minutes),
            'state_identity':deepcopy(state_identity),'scope':MODEL_SCOPE,'raw_outputs':raw,
            'metadata_override':plan,'load_readback':{'before':readback_before,'postsolve':observed},
            'clpu_multipliers':dict(multipliers),'nominal_load_ledger':ledger,
            'generation_ledger':generation_ledger,
            'known_constraints':known,'ac_audit':ac,'isolation':isolation,'reliable':bool(reliable),
            'status':('PASS' if known['known_constraints_satisfied'] is True else 'FAIL') if reliable else 'UNKNOWN',
            'structured_feedback':structured_feedback(known,raw,reliable=reliable,time_minutes=time_minutes),
            'budget_before':counts_before,'budget_after':after,
            'actual_call_counts':{'numerical_solution_entries':after['returned_numerical_entries']-counts_before['returned_numerical_entries'],
                'python_solve_commands':sum(e['counts']['python_solve_commands'] for e in self.events[before:] if e['status']=='RETURNED'),
                'compile_commands':sum(e['counts']['compile_commands'] for e in self.events[before:] if e['status']=='RETURNED'),
                'script_solve_commands':sum(e['counts']['script_solve_commands'] for e in self.events[before:] if e['status']=='RETURNED'),
                'calcvoltagebases_entries':sum(e['counts']['calcvoltagebases_entries'] for e in self.events[before:] if e['status']=='RETURNED')},
            'warnings':[{'category':w.category.__name__,'message':str(w.message),'line':w.lineno} for w in caught],
            'full_hardware_feasibility':'NOT_ESTABLISHED'}
        self.budget.record({'event':'query_completed','status':row['status'],'actual_call_counts':row['actual_call_counts']})
        self.last_query_evidence.update(phase='query_completed',evidence_status='COMPLETE_QUERY_RESULT',
                                       result_status=row['status'])
        return row
