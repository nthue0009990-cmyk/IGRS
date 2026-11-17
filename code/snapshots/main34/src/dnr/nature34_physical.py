"""Explicit main-grid restoration extension of the pinned NC34 author model.

Whole-element isolation removes open-stub charging. Unsourced components are
retained in the proposed graph and demand ledger, with every device disabled.
No island voltage source, recursive reset, invented rating, or zero imputation.
"""
from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
import shutil
import sys
import warnings

import networkx as nx
import numpy as np

from .nature13_runtime import EngineCounter
from .nature13_physical import Nature13PhysicalRuntime


class Nature34PhysicalRuntime(Nature13PhysicalRuntime):
    def __init__(self, project: Path, run: Path, physical_config: dict, events=None):
        self.project, self.run, self.physical_config = project, run, physical_config
        self.events = [] if events is None else events
        pointer = json.loads((project / "artifacts/restoration_shortlist_sources_latest.json").read_text())
        source = project / pointer["derived_directory"] / "nature_author_source"
        relative = Path("Environments/DSSdirect_34bus_loadandswitching")
        self.work = run / "work"
        shutil.copytree(source / "RL_Training" / relative, self.work / relative)
        package = "Environments.DSSdirect_34bus_loadandswitching."
        if any(name.startswith(package) for name in sys.modules):
            raise RuntimeError("Only one Nature34 runtime per process is supported")
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(self.work))
        import opendssdirect as dss
        self.dss = dss
        self.engine_version = dss.Basic.Version()
        setup = importlib.import_module(package + "DSS_CircuitSetup")
        setup.dss = EngineCounter(dss, self.events)
        self.core = importlib.import_module(package + "DSS_Initialize")
        self.initial_metadata = self._metadata()
        self.metadata = {"source": pointer, "engine_version": self.engine_version,
                         "switches": self.core.AllSwitches, "dispatch_loads": self.core.dispatch_loads,
                         "model": "NC34_main_grid_whole_element_isolation_common_curtailment_v1",
                         "initialization": "Author initialize only; never import state_action_reward or recursive reset",
                         "physical_values_modified": "DER P/Q operating setpoints and explicit whole-element isolation",
                         "full_hardware_feasibility": "NOT_ESTABLISHED"}

    def _metadata(self):
        dss = self.dss
        electrical = {"line", "transformer", "load", "generator", "capacitor", "vsource"}
        all_names = list(dss.Circuit.AllElementNames())
        controls = [n for n in all_names if n.split('.')[0].lower() in {"regcontrol", "swtcontrol", "energymeter"}]
        expected = [n for n in all_names if n not in controls]
        if any(n.split('.')[0].lower() not in electrical for n in expected):
            raise ValueError("Uncaptured physical element class")
        phases, bases = {}, {}
        for bus in dss.Circuit.AllBusNames():
            dss.Circuit.SetActiveBus(bus)
            phases[bus.lower()] = [int(n) for n in dss.Bus.Nodes() if n > 0]
            bases[bus.lower()] = dss.Bus.kVBase()
        layouts = {}
        for name in expected:
            dss.Circuit.SetActiveElement(name)
            layouts[name] = {"buses": dss.CktElement.BusNames(), "node_order": dss.CktElement.NodeOrder(),
                             "conductors": dss.CktElement.NumConductors(), "terminals": dss.CktElement.NumTerminals(),
                             "kind": name.split('.')[0].lower()}
        codes = {}
        for name in dss.Lines.AllNames():
            dss.Lines.Name(name)
            codes['Line.' + name] = dss.Lines.LineCode().lower().removeprefix('mtx')
        transformers = {}
        for name in dss.Transformers.AllNames():
            dss.Transformers.Name(name)
            dss.Circuit.SetActiveElement('Transformer.' + name)
            count = dss.CktElement.NumPhases()
            windings = []
            for winding in range(1, dss.Transformers.NumWindings() + 1):
                dss.Transformers.Wdg(winding)
                windings.append({"kva": dss.Transformers.kVA(), "kv": dss.Transformers.kV(),
                                 "connection": "delta" if dss.Transformers.IsDelta() else "wye"})
            transformers['Transformer.' + name] = {"phases": count, "windings": windings}
        expected_ratings = self.physical_config['expected_transformer_kva']
        if {n.lower() for n in expected_ratings} != {n.split('.')[1].lower() for n in transformers}:
            raise ValueError("Transformer rating inventory mismatch")
        for name, rating in expected_ratings.items():
            if any(abs(w['kva'] - rating) > 1e-8 for w in transformers['Transformer.' + name.lower()]['windings']):
                raise ValueError("Author transformer rating mismatch: " + name)
        generators = {}
        for name in dss.Generators.AllNames():
            dss.Generators.Name(name)
            dss.Circuit.SetActiveElement('Generator.' + name)
            if abs(dss.Generators.PF() - .85) > 1e-12:
                raise ValueError("Source DER pf is not 0.85")
            generators['Generator.' + name] = {"bus": dss.CktElement.BusNames()[0].split('.')[0].lower(),
                                               "available_p_kw": dss.Generators.kW(),
                                               "reference_q_kvar": dss.Generators.kvar(), "reference_pf": dss.Generators.PF()}
        loads = {}
        for name in dss.Loads.AllNames():
            dss.Loads.Name(name)
            dss.Circuit.SetActiveElement('Load.' + name)
            loads['Load.' + name] = {"bus": dss.CktElement.BusNames()[0].split('.')[0].lower(),
                                    "nominal_kw": dss.Loads.kW(), "nominal_kvar": dss.Loads.kvar()}
        assert len(loads) == 68 and abs(sum(v['nominal_kw'] for v in loads.values()) - 1769) < 1e-8
        assert abs(sum(v['nominal_kvar'] for v in loads.values()) - 1044) < 1e-8
        assert abs(sum(v['available_p_kw'] for v in generators.values()) - 586) < 1e-8
        assert set(phases) == set(self.core.G_init)
        return {"node_phases": phases, "physical_phase_nodes_by_bus": phases,
                "neutral_nodes_by_bus": {bus: [0] for bus in phases}, "node_base_kv_ln": bases,
                "source_buses": ['sourcebus'], "allowed_source_elements": ['Vsource.source'],
                "line_config_by_name": codes, "line_ratings": {code: {"phase_amps": limit} for code, limit in self.physical_config['line_phase_ampacity_A'].items()},
                "transformer_ratings": transformers, "generator_reference": generators, "load_reference": loads,
                "gfm_limits": {name: {"p_min_kw": 0., "p_max_kw": ref['available_p_kw']} for name, ref in generators.items()},
                "expected_element_names": expected, "excluded_controls": controls, "element_layout": layouts,
                "ideal_line_names": [], "neutral_thermal_mode": "NOT_MODELED",
                "known_limitations": self.physical_config['explicit_model_boundaries']}

    def evaluate_point(self, action, case, alpha, algorithm):
        bits = np.asarray(action)
        if bits.shape != (19,) or not np.isin(bits, [0, 1]).all():
            raise ValueError("Expected nine switch and ten load bits")
        if not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha outside [0,1]")
        faults = self.fault_edges(case['fault_lines'])
        graph = self.core.G_init.copy()
        graph.remove_edges_from(faults)
        fault_set = {frozenset(e) for e in faults}
        for bit, edge in zip(bits[:9], self.core.SwitchLines):
            if bit and frozenset(edge) in fault_set:
                raise ValueError("Faulted switch requested closed")
            if not bit and graph.has_edge(*edge):
                graph.remove_edge(*edge)
        if not nx.is_forest(graph):
            raise ValueError("Proposed graph is not a forest")
        powered = set(nx.node_connected_component(graph, 'sourcebus'))
        dead = set(graph) - powered
        planned = {}
        dispatch = {n.lower(): int(b) for n, b in zip(self.core.dispatch_loads, bits[9:])}
        before = len(self.events)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            obj, _, _ = self.core.initialize()
            for name in self.initial_metadata['excluded_controls']:
                obj.dss.Text.Command(f'Edit {name} enabled=no')
            for name, layout in self.initial_metadata['element_layout'].items():
                buses = [b.split('.')[0].lower() for b in layout['buses']]
                kind = layout['kind']
                enabled = all(bus in powered for bus in buses)
                if kind in ('line', 'transformer'):
                    distinct = list(dict.fromkeys(buses))
                    enabled &= len(distinct) == 2 and graph.has_edge(*distinct)
                if kind == 'load':
                    enabled &= bool(dispatch.get(name.lower(), 1))
                planned[name] = bool(enabled)
                obj.dss.Text.Command(f"Edit {name} enabled={'yes' if enabled else 'no'}")
                if enabled and kind in ('line', 'transformer'):
                    obj.dss.Circuit.SetActiveElement(name)
                    for terminal in range(1, layout['terminals'] + 1):
                        obj.dss.CktElement.Close(terminal, 0)
            for name, ref in self.initial_metadata['generator_reference'].items():
                obj.dss.Text.Command(f"Edit {name} kW={alpha * ref['available_p_kw']:.15g} kvar={alpha * ref['reference_q_kvar']:.15g}")
            # Keep voltage regulation only on transformers in the powered tree.
            for name in self.initial_metadata['excluded_controls']:
                if name.lower().startswith('regcontrol.'):
                    obj.dss.Circuit.SetActiveElement(name)
                    transformer = obj.dss.Properties.Value('Transformer')
                    if not transformer.lower().startswith('transformer.'):
                        transformer = 'Transformer.' + transformer
                    actual = next((n for n in planned if n.lower() == transformer.lower()), None)
                    if actual is None:
                        raise ValueError('Unknown regulated transformer: ' + transformer)
                    obj.dss.Text.Command(f"Edit {name} enabled={'yes' if planned[actual] else 'no'}")
            obj.dss.Text.Command('MakeBusList')
            obj.dss.Solution.Algorithm(int(algorithm))
            obj.dss.Solution.Convergence(self.physical_config['solver_convergence_tolerance'])
            obj.dss.Solution.Solve()
            raw = self.extended_snapshot(graph)
        override = {'planned_enabled': planned, 'deenergized_buses': sorted(dead)}
        load_ledger = {'denominator_nominal_kw': 1769., 'energized_nominal_kw': 0.,
                       'deenergized_nominal_kw': 0., 'deliberately_shed_nominal_kw': 0., 'loads': []}
        for name, ref in self.initial_metadata['load_reference'].items():
            state = 'energized' if planned[name] else 'deenergized' if ref['bus'] in dead else 'deliberately_shed'
            load_ledger[state + '_nominal_kw'] += ref['nominal_kw']
            load_ledger['loads'].append({'name': name, **ref, 'state': state})
        load_ledger['nominal_restored_fraction'] = load_ledger['energized_nominal_kw'] / 1769.
        enabled_der = {name for name in self.initial_metadata['generator_reference'] if planned[name]}
        available = sum(ref['available_p_kw'] for name, ref in self.initial_metadata['generator_reference'].items() if name in enabled_der)
        reactive_ref = sum(ref['reference_q_kvar'] for name, ref in self.initial_metadata['generator_reference'].items() if name in enabled_der)
        def measured_power(kind, sign=1.):
            if not raw['converged']:
                return None
            values = [np.asarray(e['powers_kw_kvar'], dtype=float) for e in raw['elements']
                      if e['enabled'] and e['name'].split('.')[0].lower() == kind]
            if any(not np.isfinite(v).all() for v in values):
                return None
            return {'p_kw': sign * sum(float(v[::2].sum()) for v in values),
                    'q_kvar': sign * sum(float(v[1::2].sum()) for v in values)}
        return {'domain_status': 'MAIN_GRID_FOREST_WHOLE_ELEMENT_ISOLATION', 'case': case['name'],
                'action': bits.tolist(), 'fault_lines': case['fault_lines'], 'alpha': alpha,
                'algorithm': algorithm, 'raw_outputs': raw, 'metadata_override': override,
                'nominal_load_ledger': load_ledger,
                'generation_ledger': {'total_installed_available_p_kw': 586., 'enabled_available_p_kw': available,
                                      'enabled_requested_p_kw': alpha * available, 'enabled_requested_q_kvar': alpha * reactive_ref,
                                      'measured_generator_injection': measured_power('generator', -1.)},
                'measured_enabled_load': measured_power('load'),
                'python_solve_calls': sum(e['method'] == 'Solution.Solve' for e in self.events[before:]),
                'compile_commands': sum(e['method'] == 'Text.Command' and str(e['args'][0]).lower().startswith('compile') for e in self.events[before:]),
                'warnings': [{'category': w.category.__name__, 'message': str(w.message), 'line': w.lineno} for w in caught]}
