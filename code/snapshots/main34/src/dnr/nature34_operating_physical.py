"""Separately named, per-point author-daily P/Q extension of the NC34 runtime.

The original physical runtime is unchanged. Its evaluate_point body is retained
except for explicit load application/readback and the condition demand ledger.
"""
from __future__ import annotations

import math
import warnings

import networkx as nx
import numpy as np

from .nature34_physical import Nature34PhysicalRuntime
from .nature34_operating_ledger import (
    nominal_ledger, validate_condition, verify_setpoint_readback,
)


def read_load_properties(dss):
    """Read every load, even disabled loads; no solve or electrical-state claim."""
    result = {}
    for short in dss.Loads.AllNames():
        dss.Loads.Name(short)
        dss.Circuit.SetActiveElement('Load.' + short)
        result['Load.' + short] = {
            'nominal_kw': dss.Loads.kW(), 'nominal_kvar': dss.Loads.kvar(),
            'model': dss.Loads.Model(), 'is_delta': bool(dss.Loads.IsDelta()),
            'kv': dss.Loads.kV(), 'phases': dss.CktElement.NumPhases(),
            'buses': list(dss.CktElement.BusNames()),
            'conductors': dss.CktElement.NumConductors(),
            'terminals': dss.CktElement.NumTerminals(),
        }
    return result


def verify_unchanged_attributes(before, after):
    if before.keys() != after.keys():
        raise ValueError('Load attribute inventory changed')
    ignored = {'nominal_kw', 'nominal_kvar'}
    for name in before:
        if ({k: v for k, v in before[name].items() if k not in ignored} !=
                {k: v for k, v in after[name].items() if k not in ignored}):
            raise ValueError('Physical load attributes changed: ' + name)


def apply_condition(dss, condition, reference, base_properties):
    """Apply both P and Q after each reset; verify all inventory and attributes."""
    requests = validate_condition(condition, reference)
    before = read_load_properties(dss)
    if before != base_properties:
        raise ValueError('Per-point initialization did not restore base load properties')
    for name, row in requests.items():
        dss.Text.Command(
            f"Edit {name} kW={row['nominal_kw']:.15g} kvar={row['nominal_kvar']:.15g}")
    after = read_load_properties(dss)
    verify_setpoint_readback(condition, reference, after)
    verify_unchanged_attributes(before, after)
    return {'before': before, 'after': after}


class Nature34OperatingPhysicalRuntime(Nature34PhysicalRuntime):
    def __init__(self, project, run, physical_config, events=None):
        super().__init__(project, run, physical_config, events)
        self.base_load_properties = read_load_properties(self.dss)
        self.operating_reference = [
            {'name': name, **ref}
            for name, ref in self.initial_metadata['load_reference'].items()
        ]
        self.metadata = {
            **self.metadata,
            'model': 'NC34_author_daily_PQ_main_grid_whole_element_isolation_v1',
            'physical_values_modified': (
                'Explicit common author-daily multiplier on all 68 load P/Q; '
                'DER common P/Q curtailment and whole-element isolation'),
        }

    def evaluate_point(self, action, case, alpha, algorithm, condition):
        validate_condition(condition, self.operating_reference)
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
            load_readback = apply_condition(obj.dss, condition, self.operating_reference, self.base_load_properties)
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
            load_readback['postsolve'] = read_load_properties(obj.dss)
            verify_setpoint_readback(condition, self.operating_reference, load_readback['postsolve'])
            verify_unchanged_attributes(load_readback['before'], load_readback['postsolve'])
        override = {'planned_enabled': planned, 'deenergized_buses': sorted(dead)}
        load_ledger = nominal_ledger(condition, self.operating_reference,
            {name: planned[name] for name in self.initial_metadata['load_reference']}, sorted(dead))
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
                'condition_id': condition['condition_id'], 'load_readback': load_readback,
                'generation_ledger': {'total_installed_available_p_kw': 586., 'enabled_available_p_kw': available,
                                      'enabled_requested_p_kw': alpha * available, 'enabled_requested_q_kvar': alpha * reactive_ref,
                                      'measured_generator_injection': measured_power('generator', -1.)},
                'measured_enabled_load': measured_power('load'),
                'python_solve_calls': sum(e['method'] == 'Solution.Solve' for e in self.events[before:]),
                'compile_commands': sum(e['method'] == 'Text.Command' and str(e['args'][0]).lower().startswith('compile') for e in self.events[before:]),
                'warnings': [{'category': w.category.__name__, 'message': str(w.message), 'line': w.lineno} for w in caught]}

