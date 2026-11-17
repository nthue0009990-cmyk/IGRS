"""NC123 v2: complete-phase DER availability and certified idle-phase scope.

The thirteen source DER definitions remain unchanged. No engine import or
numerical call occurs at module import or in any pure certificate function.
Full observations are preserved; derived checker views omit only independently
certified absent/zero idle nodes, never substitute numerical observations.
"""
from __future__ import annotations

from copy import deepcopy
import math

from .nature123_clpu_physical import (Nature123CLPUPhysicalRuntime, certify_snapshot,
    canonical_sha, phase_connectivity)
from .restoration_ac_audit import audit_ac_snapshot
from .restoration_isolation import verify_isolation


DER_POLICY = 'ALL_DECLARED_PHASES_SOURCE_REACHABLE_V1'
IDLE_POLICY = 'CERTIFIED_DISABLED_ONLY_PHASE_SCOPE_V1'


def declared_node_order(layout):
    """Parse DSS terminal declaration, explicitly distinct from observed nodes.

    Missing suffix means phase nodes 1..NumPhases then ground conductors;
    explicit suffixes pad with ground. Extra explicit zeroes (capacitor return
    terminal shorthand) are allowed; extra nonzero nodes are unsupported.
    """
    n, nt, phases = (layout[k] for k in ('conductors', 'terminals', 'phases'))
    if any(type(v) is not int or v < 1 for v in (n, nt, phases)) or phases not in (1, 2, 3) or n < phases or len(layout['buses']) != nt:
        raise ValueError('Unsupported declared device dimensions')
    result = []
    for bus in layout['buses']:
        parts = bus.lower().split('.')
        if not parts[0]: raise ValueError('Empty terminal bus')
        if len(parts) == 1:
            nodes = list(range(1, phases + 1))
        else:
            if any(not v.isdigit() for v in parts[1:]): raise ValueError('Unsupported bus node declaration')
            nodes = [int(v) for v in parts[1:]]
        if any(v not in (0, 1, 2, 3) for v in nodes) or any(nodes[n:]):
            raise ValueError('Unsupported declared phase/extra conductor')
        nodes = nodes[:n] + [0] * max(0, n-len(nodes))
        result.extend(nodes)
    return result


def _terminal_nodes(element, order=None):
    order = element['node_order'] if order is None else order
    n, nt = element['conductors'], element['terminals']
    if (type(n) is not int or type(nt) is not int or n < 1 or nt < 1 or
            len(order) != n*nt or len(element['buses']) != nt or
            any(type(v) is not int or v not in (0,1,2,3) for v in order)):
        raise ValueError('Unsupported terminal-node inventory')
    return {f"{element['buses'][t].split('.')[0].lower()}.{node}"
            for t in range(nt) for node in order[t*n:(t+1)*n] if node}


def derive_der_availability(raw, metadata):
    """Source reachability uses only enabled PDs and the actual main source.

    For planning, callers supply explicitly marked declared-layout predictions;
    the post-solve call receives actual enabled PD NodeOrder and must agree.
    Disabled DER caches never establish their declared connection.
    """
    elements = {e['name'].lower(): e for e in raw['elements']}
    refs = metadata['generator_reference']
    if set(refs) != {n for n in elements if n.startswith('generator.')}:
        raise ValueError('Complete source DER inventory required')
    pd = [e for n, e in elements.items() if n.split('.')[0] in ('line', 'transformer', 'vsource')]
    reach = phase_connectivity({'elements': pd}, {'transformer_ratings': metadata['transformer_ratings']})
    if reach['unknown']:
        raise ValueError('Unsupported source phase reachability: ' + str(reach['unknown']))
    reached = set(reach['source_reachable_phase_nodes'])
    rows = {}
    for name, ref in refs.items():
        declared = _terminal_nodes(metadata['element_layout'][name])
        if not declared: raise ValueError('DER has no declared positive phase')
        missing = sorted(declared - reached)
        rows[name] = {'declared_phase_nodes': sorted(declared), 'missing_source_phase_nodes': missing,
            'complete_phase_available': not missing, 'source_nominal_kw': ref['available_p_kw'],
            'available_kw': 0. if missing else ref['available_p_kw'],
            'reference_q_kvar': ref['reference_q_kvar']}
    return {'policy': DER_POLICY, 'source_reachable_phase_nodes': sorted(reached), 'generators': rows,
        'source_nominal_kw': math.fsum(r['source_nominal_kw'] for r in rows.values()),
        'available_kw': math.fsum(r['available_kw'] for r in rows.values()),
        'available_generators': sorted(n for n,r in rows.items() if r['complete_phase_available']),
        'unavailable_generators': sorted(n for n,r in rows.items() if not r['complete_phase_available'])}


def certify_idle_phase_scope(raw, metadata):
    """Full-inventory proof before any node is omitted from a checker view."""
    failures, unknown, rows = [], [], []
    full = {f'{bus}.{node}' for bus, nodes in metadata['physical_phase_nodes_by_bus'].items() for node in nodes}
    expected = {n.lower() for n in metadata['expected_element_names']}
    elements = {e['name'].lower(): e for e in raw['elements']}
    if set(elements) != expected or len(elements) != len(raw['elements']):
        unknown.append('Full physical element inventory differs')
    if set(metadata['element_layout']) != expected:
        unknown.append('Frozen declared layout inventory differs')
    contributors = {node: [] for node in full}; needed = set(); disabled_zero = {}
    itol = metadata['isolation_current_tolerance_a']; vtol = metadata['isolation_voltage_tolerance_pu']
    for name in sorted(expected):
        e = elements.get(name); layout = metadata['element_layout'].get(name)
        if e is None or layout is None: continue
        try:
            declared = _terminal_nodes(layout)
        except (KeyError, TypeError, ValueError) as exc:
            unknown.append(name + ': ' + str(exc)); continue
        if not declared <= full:
            unknown.append(name + ': declared nodes absent from full inventory')
        for node in declared & full: contributors[node].append(name)
        if type(e.get('enabled')) is not bool:
            unknown.append(name + ': missing enabled state'); continue
        if (e.get('buses') != layout['buses'] or e.get('conductors') != layout['conductors'] or
                e.get('terminals') != layout['terminals']):
            unknown.append(name + ': frozen declaration layout differs'); continue
        if e['enabled']:
            try:
                actual = _terminal_nodes(e)
            except (KeyError, TypeError, ValueError) as exc:
                unknown.append(name + ': ' + str(exc)); continue
            needed |= actual
            if list(e['node_order']) != list(layout['node_order']):
                failures.append({'type': 'enabled_actual_nodeorder_differs_from_declaration', 'element': name})
            if not actual <= full:
                unknown.append(name + ': active nodes outside full inventory')
        else:
            currents = e.get('currents_real_imag')
            size = 2 * layout['conductors'] * layout['terminals']
            finite = (isinstance(currents, (list,tuple)) and len(currents) == size and
                      all(type(v) in (int,float) and math.isfinite(v) for v in currents))
            if not finite:
                unknown.append(name + ': disabled current evidence missing/nonfinite')
                disabled_zero[name] = False
            else:
                maximum = max(math.hypot(currents[i],currents[i+1]) for i in range(0,size,2))
                disabled_zero[name] = maximum <= itol
                if maximum > itol:
                    failures.append({'type':'disabled_current_nonzero', 'element':name, 'max_current_a':maximum})
    names, volts, complex_volts = raw.get('node_names'), raw.get('node_vmag_pu'), raw.get('node_voltages_real_imag')
    valid_nodes = (isinstance(names,list) and isinstance(volts,list) and isinstance(complex_volts,list)
                   and len(names)==len(volts) and len(complex_volts)==2*len(names)
                   and all(isinstance(n,str) for n in names))
    observed = {}
    if not valid_nodes:
        unknown.append('Complete original node voltage arrays missing')
    else:
        observed = {n.lower(): i for i,n in enumerate(names)}
        if len(observed) != len(names): unknown.append('Duplicate observed node names')
        for n in observed:
            if n not in full and not n.endswith('.0'): unknown.append('Observed positive node outside full inventory: ' + n)
    idle = sorted(full - needed)
    for node in idle:
        owners = contributors[node]
        entry = {'node': node, 'declaring_elements': owners,
                 'enabled_terminal_support': False, 'observation': 'UNKNOWN'}
        if not owners:
            unknown.append('Idle node without declaring element: ' + node)
        elif any(elements.get(n,{}).get('enabled') is not False or not disabled_zero.get(n,False) for n in owners):
            failures.append({'type':'idle_node_not_supported_by_disabled_zero_devices', 'node':node})
        if valid_nodes and node not in observed:
            entry['observation'] = 'STRUCTURALLY_EXCLUDED_NOT_SOLVED'
        elif valid_nodes:
            i=observed[node]; value=volts[i]; vector=complex_volts[2*i:2*i+2]
            entry.update(observed_voltage_pu=value, observed_voltage_real_imag_v=vector)
            base = metadata['node_base_kv_ln'].get(node.rsplit('.',1)[0])
            if (type(value) not in (int,float) or not math.isfinite(value) or value < 0 or
                    any(type(v) not in (int,float) or not math.isfinite(v) for v in vector) or
                    type(base) not in (int,float) or not math.isfinite(base) or base<=0):
                unknown.append('Idle node voltage evidence nonfinite/invalid: ' + node)
            elif value > vtol or math.hypot(*vector) > vtol*base*1000:
                failures.append({'type':'idle_node_voltage_nonzero', 'node':node,
                                 'observed_voltage_pu':value, 'observed_voltage_v':math.hypot(*vector)})
                entry['observation']='UNSOURCED_NONZERO_VOLTAGE'
            else:
                entry['observation']='OBSERVED_FINITE_ZERO'
        rows.append(entry)
    if raw.get('converged') is not True: unknown.append('Numerical convergence absent')
    status = 'UNKNOWN' if unknown else 'FAIL' if failures else 'PASS'
    return {'status': status, 'policy': IDLE_POLICY, 'full_phase_node_count': len(full),
        'required_enabled_phase_nodes': sorted(needed), 'candidate_idle_phase_nodes': idle,
        'certified_excluded_phase_nodes': idle if status=='PASS' else [], 'nodes': rows,
        'disabled_zero_current_checks': disabled_zero, 'unknown': unknown, 'violations': failures,
        'raw_node_inventory_sha256': canonical_sha(names), 'full_phase_inventory_sha256': canonical_sha(metadata['physical_phase_nodes_by_bus']),
        'scope': 'ALL_PHYSICAL_ELEMENTS_AND_ALL_DECLARED_PHASES;NO_NODE_NAME_EXCEPTIONS'}


def _checker_view(raw, metadata, certificate):
    if certificate['status'] != 'PASS': raise ValueError('An uncertified idle node cannot be excluded')
    excluded = set(certificate['certified_excluded_phase_nodes'])
    selected = [i for i,n in enumerate(raw['node_names']) if n.lower() not in excluded]
    view = deepcopy(raw); meta = deepcopy(metadata)
    view['node_names'] = [raw['node_names'][i] for i in selected]
    view['node_vmag_pu'] = [raw['node_vmag_pu'][i] for i in selected]
    view['node_voltages_real_imag'] = [v for i in selected for v in raw['node_voltages_real_imag'][2*i:2*i+2]]
    phases = {bus:[n for n in nodes if f'{bus}.{n}' not in excluded]
              for bus,nodes in metadata['physical_phase_nodes_by_bus'].items()}
    meta['node_phases'] = deepcopy(phases); meta['physical_phase_nodes_by_bus'] = deepcopy(phases)
    return view, meta, {'retained_original_node_indices': selected,
        'excluded_phase_nodes': sorted(excluded), 'rule': 'SUBSEQUENCE_OF_SAVED_ORIGINAL_OBSERVATIONS_NO_IMPUTATION',
        'derived_phase_inventory': phases, 'scope_certificate_sha256':canonical_sha(certificate)}


def certify_phase_available_snapshot(raw, metadata, physical_config):
    """Saved-snapshot recheck; does not load DSS or alter the original data."""
    scope = certify_idle_phase_scope(raw, metadata)
    full_isolation = verify_isolation(raw, metadata)
    full_ac = audit_ac_snapshot(raw, metadata, **physical_config['ac_audit_tolerances'])
    decision = derive_der_availability(raw, metadata)
    mismatches = []
    elements = {e['name'].lower():e for e in raw['elements']}
    predicted = metadata['der_availability_plan']
    for name,r in decision['generators'].items():
        enabled = elements[name].get('enabled')
        if (enabled is not r['complete_phase_available'] or
                r['complete_phase_available'] is not predicted['generators'][name]['complete_phase_available'] or
                r['missing_source_phase_nodes'] != predicted['generators'][name]['missing_source_phase_nodes']):
            mismatches.append(name)
    if scope['status']=='PASS':
        view, meta_view, mapping = _checker_view(raw, metadata, scope)
        result = certify_snapshot(view, meta_view, physical_config)
        result['constraint_check_view'] = mapping
    else:
        result = certify_snapshot(raw, metadata, physical_config)
        result['checks']['phase_connectivity'] = 'UNKNOWN'
    result.update(idle_phase_scope=scope, isolation=full_isolation, ac_audit=full_ac,
                  der_availability_readback=decision, der_availability_mismatches=mismatches)
    result['checks']['isolation'] = full_isolation['status']
    result['checks']['ac_balance'] = full_ac['status']
    if mismatches or scope['status']!='PASS': result['checks']['phase_connectivity']='UNKNOWN'
    reliable = (all(result['checks'][k]=='PASS' for k in
        ('numerical_convergence','topology','isolation','phase_connectivity','ac_balance')) and
        all(result['checks'][k] in ('PASS','FAIL') for k in ('voltage','line_phase_thermal','transformer_winding')) and
        result['known_constraints']['gfm_active_limit']['status']=='PASS')
    result['reliable']=reliable
    result['status']=('PASS' if all(result['checks'][k]=='PASS' for k in
        ('voltage','line_phase_thermal','transformer_winding')) else 'FAIL') if reliable else 'UNKNOWN'
    actual_supply = {}; supply_basis = {}
    for n,e in elements.items():
        if n not in decision['generators']: continue
        values=e.get('powers_kw_kvar',[])
        if e['enabled'] and values and all(type(v) in (int,float) and math.isfinite(v) for v in values):
            actual_supply[n] = -math.fsum(values[::2])
            supply_basis[n] = 'OBSERVED_ENABLED_TERMINAL_SUM'
        elif e['enabled'] is False and scope['disabled_zero_current_checks'].get(n) is True:
            actual_supply[n] = 0.
            supply_basis[n] = 'STRUCTURALLY_ZERO_DISABLED_FINITE_ZERO_CURRENT'
        else:
            actual_supply[n] = None
            supply_basis[n] = 'UNVERIFIED_NO_SUPPLY_VALUE'
    result['der_power_ledger'] = {'source_nominal_kw':decision['source_nominal_kw'], 'phase_available_kw':decision['available_kw'],
        'enabled_generators':decision['available_generators'], 'disabled_generators':decision['unavailable_generators'],
        'supply_accounting_kw_by_generator':actual_supply, 'supply_basis_by_generator':supply_basis,
        'supply_accounting_kw':math.fsum(actual_supply.values()) if all(v is not None for v in actual_supply.values()) else None,
        'disabled_zero_current_evidence':scope['disabled_zero_current_checks'],
        'zero_disabled_supply_is_not_an_observed_cached_power':True}
    return result


class Nature123PhaseAvailablePhysicalRuntime(Nature123CLPUPhysicalRuntime):
    def __init__(self, project, *, model_directory='artifacts/nature123_model_v2', **kwargs):
        super().__init__(project, model_directory=model_directory, **kwargs)
        for key, value in (('phase_availability_policy',DER_POLICY),('idle_phase_policy',IDLE_POLICY)):
            if self.bundle.config.get(key)!=value or self.physical_config.get(key)!=value:
                raise ValueError('Missing or unbound approved v2 policy: '+key)

    def validate_request(self, request):
        graph,state=super().validate_request(request)
        for key,value in (('phase_availability_policy',DER_POLICY),('idle_phase_policy',IDLE_POLICY)):
            if request['condition']['operating_config'].get(key)!=value:
                raise ValueError('Request v2 policy missing or mismatched: '+key)
        refs={f"generator.g{r['no']}":r for r in self.bundle.author['generators']}
        der=request['condition']['der_setpoints']
        if set(der)!=set(refs): raise ValueError('Full thirteen DER request required')
        for name,row in der.items():
            ref=refs[name]
            if row['kw']!=ref['size'] or not math.isclose(row['kvar'],.75*ref['size'],rel_tol=1e-12,abs_tol=1e-8):
                raise ValueError('V2 availability policy does not change source DER P/Q')
        return graph,state

    def _generator_attributes(self):
        d=self.dss; rows={}
        for short in d.Generators.AllNames():
            d.Generators.Name(short); d.Circuit.SetActiveElement('generator.'+short)
            rows['generator.'+short.lower()]={'phases':d.CktElement.NumPhases(),
                'kv':float(d.Generators.kV()), 'model':d.Generators.Model(),
                'is_delta':bool(d.Generators.IsDelta()),'buses':list(d.CktElement.BusNames())}
        return rows

    def _metadata(self):
        meta=super()._metadata(); d=self.dss
        original={}
        for name,layout in meta['element_layout'].items():
            d.Circuit.SetActiveElement(name)
            layout['phases']=d.CktElement.NumPhases()
            original[name]=list(layout['node_order'])
            layout['node_order']=declared_node_order(layout)
        meta['initial_engine_node_order_by_element']=original
        meta['element_layout_node_order_scope']='DECLARED_TERMINALS_PREDICTION_REQUIRES_ENABLED_POSTSOLVE_EQUALITY'
        meta['phase_availability_policy']=DER_POLICY; meta['idle_phase_policy']=IDLE_POLICY
        attrs=self._generator_attributes()
        for row in self.bundle.author['generators']:
            expected={'phases':row['numphase'],'kv':float(row['kV']),'model':1,'is_delta':False,
                      'buses':[row['bus']+row['phaseconn']]}
            if attrs[f"generator.g{row['no']}"]!=expected:
                raise ValueError('Author DER connection/voltage/model changed')
        meta['generator_immutable_attributes']=attrs
        return meta

    def _apply(self, request, graph, state, meta):
        network,readback=super()._apply(request,graph,state,meta)
        predicted={'elements':[{'name':name,'enabled':meta['planned_enabled'][name],**deepcopy(layout)}
                               for name,layout in meta['element_layout'].items()]}
        decision=derive_der_availability(predicted,meta)
        decision['evidence_scope']='DECLARED_LAYOUT_PREDICTION_NOT_A_SOLVED_NODEORDER_OBSERVATION'
        meta['der_availability_plan']=decision
        for name,row in decision['generators'].items():
            enabled=row['complete_phase_available']
            self._command(f"Edit {name} enabled={'yes' if enabled else 'no'}")
            meta['planned_enabled'][name]=enabled
        self._command('MakeBusList')
        readback['der_availability_plan']=deepcopy(decision)
        return network,readback

    def _verify_postsolve(self, request, meta, before):
        result=super()._verify_postsolve(request,meta,before)
        attrs=self._generator_attributes()
        if attrs!=meta['generator_immutable_attributes']:
            raise ValueError('Immutable DER attributes changed during phase availability application')
        result['generator_immutable_attributes']=attrs
        result['generators_enabled']={}
        for name,row in meta['der_availability_plan']['generators'].items():
            self.dss.Circuit.SetActiveElement(name)
            enabled=bool(self.dss.CktElement.Enabled())
            result['generators_enabled'][name]=enabled
            if enabled is not row['complete_phase_available']:
                raise ValueError('Complete-phase DER availability not applied: '+name)
        return result

    def _certify(self, raw, metadata):
        return certify_phase_available_snapshot(raw,metadata,self.physical_config)
