"""NC34 per-load CLPU nominal-demand extension, with no training dependency.

Every evaluation is an independently reset quasi-static snapshot. Original ZIP,
phase, connection, ratings, DER and regulator solving semantics are retained;
this is not a continuous controller-state or transient simulation. The normal
demand ledger stays separate from CLPU burden, preventing cold-load energy from
being rewarded as useful restored demand. Importing this module does not load
or call OpenDSS. Actual calls require a separately budgeted runtime instance.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from numbers import Real
import warnings

import networkx as nx
import numpy as np

from .nature34_operating_physical import (Nature34OperatingPhysicalRuntime,
    read_load_properties, verify_unchanged_attributes)
from .nature34_operating_ledger import validate_condition, nominal_ledger
from .restoration_ac_audit import audit_ac_snapshot
from .restoration_constraints import check_known_constraints
from .restoration_isolation import verify_isolation


def _nonnegative(value, label):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError('Invalid ' + label)
    return float(value)


def _same(value, expected, label):
    if not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-8):
        raise ValueError(label + ' differs')


def engine_numeric_lists(value):
    """Match the old runner's numeric_lists before invoking pure checkers.

    OpenDSS raw_snapshot explicitly makes node_vmag_pu an ndarray, while the
    pure evidence checkers require Python Sequence. This representation-only
    adaptation preserves actual NaN/Inf and never fills or changes values.
    """
    if isinstance(value, np.ndarray):
        return engine_numeric_lists(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: engine_numeric_lists(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [engine_numeric_lists(item) for item in value]
    return value


def validate_clpu_requests(condition, reference, multipliers):
    """Strict base demand binding followed by explicit full-inventory scaling.

    A factor below one is outside this positive exponential-decay model. All
    68 setpoints are retained even for disabled loads; energization is applied
    independently and recorded by the graph/isolation ledger.
    """
    normal = validate_condition(condition, reference)
    if not isinstance(multipliers, Mapping) or normal.keys() != multipliers.keys():
        raise ValueError('Exactly all 68 named CLPU multipliers are required')
    result = {}
    for name, row in normal.items():
        factor = _nonnegative(multipliers[name], name + ':CLPU multiplier')
        if factor < 1:
            raise ValueError('CLPU multiplier below one: ' + name)
        p = _nonnegative(row['nominal_kw'] * factor, name + ':dynamic P')
        q = _nonnegative(row['nominal_kvar'] * factor, name + ':dynamic Q')
        result[name] = {'bus': row['bus'], 'nominal_kw': p, 'nominal_kvar': q,
                        'normal_nominal_kw': row['nominal_kw'],
                        'normal_nominal_kvar': row['nominal_kvar'],
                        'clpu_multiplier': factor}
    return result


def verify_clpu_readback(condition, reference, multipliers, observed):
    expected = validate_clpu_requests(condition, reference, multipliers)
    if expected.keys() != observed.keys():
        raise ValueError('Full CLPU readback inventory required, including disabled loads')
    for name, row in expected.items():
        for key in ('nominal_kw', 'nominal_kvar'):
            _same(_nonnegative(observed[name][key], 'readback ' + key), row[key], name + ':' + key)
    return {'status': 'CLPU_SETPOINT_PROPERTIES_MATCH', 'checked_loads': 68,
            'ac_solution_certified': False, 'recovery_witness_established': False}


def apply_clpu_condition(dss, condition, reference, multipliers, base_properties):
    requests = validate_clpu_requests(condition, reference, multipliers)
    before = read_load_properties(dss)
    if before != base_properties:
        raise ValueError('Per-point initialization did not restore base load properties')
    for name, row in requests.items():
        dss.Text.Command(f"Edit {name} kW={row['nominal_kw']:.15g} kvar={row['nominal_kvar']:.15g}")
    after = read_load_properties(dss)
    verify_clpu_readback(condition, reference, multipliers, after)
    verify_unchanged_attributes(before, after)
    return {'before': before, 'after': after}


def clpu_ledger(condition, reference, multipliers, planned_loads, deenergized_buses):
    normal = nominal_ledger(condition, reference, planned_loads, deenergized_buses)
    dynamic = validate_clpu_requests(condition, reference, multipliers)
    result = {'condition_id': condition['condition_id'],
              'normal_demand_denominator_kw': normal['denominator_nominal_kw'],
              'normal_restored_fraction': normal['nominal_restored_fraction'],
              'energized_normal_kw': normal['energized_nominal_kw'],
              'energized_dynamic_kw': 0., 'energized_dynamic_kvar': 0.,
              'energized_clpu_extra_kw': 0., 'energized_clpu_extra_kvar': 0.,
              'all_load_dynamic_setpoint_kw': sum(r['nominal_kw'] for r in dynamic.values()),
              'all_load_dynamic_setpoint_kvar': sum(r['nominal_kvar'] for r in dynamic.values()),
              'accounting_only': True, 'physical_witness_established': False,
              'loads': []}
    for load in normal['loads']:
        name, state = load['name'], load['state']
        row = {'name': name, **dynamic[name], 'state': state}
        result['loads'].append(row)
        if state == 'energized':
            result['energized_dynamic_kw'] += row['nominal_kw']
            result['energized_dynamic_kvar'] += row['nominal_kvar']
            result['energized_clpu_extra_kw'] += row['nominal_kw'] - row['normal_nominal_kw']
            result['energized_clpu_extra_kvar'] += row['nominal_kvar'] - row['normal_nominal_kvar']
    _same(result['energized_dynamic_kw'], result['energized_normal_kw'] + result['energized_clpu_extra_kw'],
          'Normal-plus-CLPU ledger')
    return result


def structured_feedback(known, raw, *, reliable, time_minutes):
    """Extract named physical violations, never convert iterates into evidence.

    A thermal phase uses the saved NodeOrder, not conductor-position guesses.
    Total winding apparent-power violations have phase=None because they do
    not identify one individual phase. Full original violations are retained.
    """
    if not reliable:
        return {'usable_for_repair': False, 'time_minutes': time_minutes,
                'reason': 'NUMERICAL_OR_EVIDENCE_NOT_RELIABLE', 'violations': []}
    elements = {e['name'].lower(): e for e in raw['elements']}
    violations = []
    for section in ('voltage', 'line_thermal', 'transformer_thermal'):
        for original in known[section]['violations']:
            row = {'constraint': section, 'time_minutes': time_minutes,
                   'original': deepcopy(original), 'phase': None, 'bus': None,
                   'element': original.get('element')}
            if section == 'voltage':
                node = original.get('node')
                row['node'] = node
                if isinstance(node, str) and '.' in node:
                    bus, phase = node.rsplit('.', 1)
                    row.update(bus=bus, phase=int(phase) if phase.isdigit() else None)
                row.update(value=original['value'], unit='pu')
                if 'lower' in original:
                    lower, upper = original['lower'], original['upper']
                    row.update(lower=lower, upper=upper,
                               magnitude=max(lower - row['value'], row['value'] - upper, 0.),
                               direction='low' if row['value'] < lower else 'high')
                else:
                    row.update(magnitude=abs(row['value']), direction='unsourced_nonzero')
            else:
                elem = elements.get(str(original.get('element')).lower())
                terminal = original.get('terminal', original.get('winding'))
                if elem is not None and terminal is not None:
                    row['bus'] = elem['buses'][terminal - 1].split('.')[0].lower()
                row.update(terminal=terminal, conductor=original.get('conductor'))
                if original.get('role') == 'phase':
                    row['phase'] = original.get('node')
                if original.get('type') == 'winding_total_s':
                    row.update(value=original['total_s_kva'], limit=original['limit_kva'], unit='kVA')
                elif original.get('type') == 'wye_phase_s':
                    row.update(value=original['phase_s_kva'], limit=original['limit_kva'], unit='kVA')
                else:
                    row.update(value=original['current_a'], limit=original['limit_a'], unit='A')
                row.update(magnitude=max(0., row['value'] - row['limit']),
                           relative_excess=(row['value'] - row['limit']) / row['limit'], direction='high')
            violations.append(row)
    return {'usable_for_repair': True, 'time_minutes': time_minutes,
            'violations': violations, 'causal_diagnosis_established': False}


def certify_clpu_point(row, initial_metadata, physical_config):
    """Attach unchanged pure checkers and actionable feedback to a raw point.

    UNKNOWN is retained separately from ordinary voltage/thermal FAIL. The
    caller must stop affected scientific work on reliability errors; it must
    save this row first. This function itself makes no numerical solver call.
    """
    row['raw_outputs'] = engine_numeric_lists(row['raw_outputs'])
    metadata = engine_numeric_lists({**initial_metadata, **row['metadata_override']})
    row['known_constraints'] = check_known_constraints(row['raw_outputs'], metadata,
                                                       **physical_config['constraint_tolerances'])
    row['ac_audit'] = audit_ac_snapshot(row['raw_outputs'], metadata,
                                       **physical_config['ac_audit_tolerances'])
    row['isolation'] = verify_isolation(row['raw_outputs'], metadata)
    known = row['known_constraints']
    required = ('numerical', 'topology', 'voltage', 'line_thermal', 'transformer_thermal', 'gfm_active_limit')
    unknown = any(known[k]['status'] == 'UNKNOWN' or known[k].get('unknown') for k in required)
    reliable = (bool(row['raw_outputs']['converged']) and
                row['ac_audit']['status'] == row['isolation']['status'] == 'PASS' and
                known['numerical']['status'] == known['topology']['status'] == 'PASS' and not unknown)
    row['reliable'] = bool(reliable)
    row['status'] = ('PASS' if known['known_constraints_satisfied'] is True else 'FAIL') if reliable else 'UNKNOWN'
    row['structured_feedback'] = structured_feedback(known, row['raw_outputs'], reliable=reliable,
                                                    time_minutes=row['time_minutes'])
    row['full_hardware_feasibility'] = 'NOT_ESTABLISHED'
    return row


class Nature34CLPUPhysicalRuntime(Nature34OperatingPhysicalRuntime):
    def __init__(self, project, run, physical_config, events=None, *, counter_factory=None):
        self.counter_factory = counter_factory
        super().__init__(project, run, physical_config, events)
        self.metadata.update(
            model='NC34_CLPU_PER_LOAD_PQ_QUASI_STATIC_v1',
            physical_values_modified='Author daily P/Q times explicit per-load CLPU multiplier; original DER/isolation retained',
            control_state_semantics='Independent author initialization and regulator solve at every snapshot; no continuous tap-state claim',
            full_hardware_feasibility='NOT_ESTABLISHED')

    def evaluate_point(self, action, case, alpha, algorithm, condition, multipliers,
                       *, time_minutes, state_identity=None):
        validate_clpu_requests(condition, self.operating_reference, multipliers)
        _nonnegative(time_minutes, "time_minutes")
        counter_before = self.counter_factory.stats()["returned"] if self.counter_factory else None
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
            load_readback = apply_clpu_condition(obj.dss, condition, self.operating_reference, multipliers, self.base_load_properties)
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
            verify_clpu_readback(condition, self.operating_reference, multipliers, load_readback['postsolve'])
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

        row = {'domain_status': 'MAIN_GRID_FOREST_WHOLE_ELEMENT_ISOLATION', 'case': case['name'],
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



        row.update(clpu_load_ledger=clpu_ledger(condition, self.operating_reference,
            multipliers, {name: planned[name] for name in self.initial_metadata['load_reference']}, sorted(dead)),
            clpu_multipliers=dict(multipliers), time_minutes=float(time_minutes),
            state_identity=deepcopy(state_identity),
            control_state_semantics='INDEPENDENT_QUASI_STATIC_AUTHOR_RESET_EACH_SNAPSHOT')
        if self.counter_factory is not None:
            counter_after = self.counter_factory.stats()['returned']
            row['actual_call_counts'] = {k: counter_after[k] - counter_before.get(k, 0)
                                         for k in counter_after}
        else:
            row['actual_call_counts'] = None
        certify_clpu_point(row, self.initial_metadata, self.physical_config)
        return row
