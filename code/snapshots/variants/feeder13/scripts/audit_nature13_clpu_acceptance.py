"""Pure saved-point comparisons for the named NC13 CLPU acceptance scope.

No engine or physical runtime is imported. This comparator does not certify a
point: both inputs must already carry reliable AC, isolation and known-limit
evidence. Disabled element V/P caches are never substituted or compared.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real


REQUIRED_SECTIONS = ('numerical', 'topology', 'voltage', 'line_thermal',
                     'transformer_thermal', 'gfm_active_limit')
FROZEN_LIMITS = {'voltage_absolute_pu': 1e-5, 'current_absolute_A': .01,
                 'power_absolute_kw': .01}
FROZEN_SCOPE = 'NC13_CLPU_SINGLE_TERMINAL_OPEN_DEAD_DEVICES_ISOLATED_V1'
EXCLUDED_STATES = {'STRUCTURALLY_EXCLUDED_NOT_SOLVED', 'OBSERVED_FINITE_ZERO'}


class ComparisonInputError(ValueError):
    """Incomplete/mismatched evidence; never an ordinary physical violation."""


def _need(condition, message):
    if not condition:
        raise ComparisonInputError(message)


def _sequence(value):
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _finite_array(value, length, label):
    _need(_sequence(value) and len(value) == length and all(_finite(x) for x in value),
          'Missing/nonfinite/malformed ' + label)
    return value


def _mapping(value, label):
    _need(isinstance(value, Mapping), 'Missing ' + label)
    result = {str(k).lower(): v for k, v in value.items()}
    _need(len(result) == len(value), 'Duplicate normalized keys: ' + label)
    return result


def _saved_equal(left, right):
    """Compare retained nonfinite evidence without replacing its raw values."""
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(_saved_equal(left[k], right[k]) for k in left)
    if _sequence(left) and _sequence(right):
        return len(left) == len(right) and all(_saved_equal(a,b) for a,b in zip(left,right))
    if isinstance(left, float) and isinstance(right, float) and math.isnan(left) and math.isnan(right):
        return True
    return left == right


def _nodes(raw, phases, neutrals, powered):
    names, values = raw.get('node_names'), raw.get('node_vmag_pu')
    _need(_sequence(names) and _sequence(values) and len(names) == len(values),
          'Missing/malformed node voltage inventory')
    result = {}
    for name, value in zip(names, values):
        _need(isinstance(name, str) and '.' in name, 'Malformed node name')
        bus, node = name.lower().rsplit('.', 1)
        _need(node.isdigit(), 'Malformed node number')
        node = int(node)
        canonical = f'{bus}.{node}'
        _need(canonical not in result, 'Duplicate node: ' + canonical)
        _need(bus in phases and (node == 0 or node in phases[bus] or node in neutrals.get(bus, [])),
              'Unregistered solved node: ' + canonical)
        # Observed nonfinite bus values remain unknown. Only absent isolated
        # nodes and stale DISABLED ELEMENT V/P caches have an explicit exemption.
        _need(_finite(value) and value >= 0, 'Nonfinite/negative observed node voltage: ' + canonical)
        result[canonical] = value
    expected = {f'{bus}.{node}' for bus in powered for node in phases[bus]}
    _need(expected <= result.keys(), 'Missing energized phase-node voltage')
    return {name: result[name] for name in sorted(expected)}


def compare_isolated_points(left, right, initial_metadata, limits):
    """Compare Normal/Newton under the original NC13 absolute tolerances.

    Input points use the CLPU runtime record schema: raw_outputs, reliable,
    status, known_constraints, ac_audit, isolation, metadata_override, plus
    case/action/fault_lines/alpha/clpu_multipliers/time_minutes/state_identity.
    Returns PASS or FAIL for finite differences. Missing evidence raises
    ComparisonInputError; callers must save it as a reliability stop.
    """
    _need(dict(limits) == FROZEN_LIMITS, 'Original cross-algorithm tolerances changed')
    _need(all(type(p.get('algorithm')) is int for p in (left, right))
          and {left.get('algorithm'), right.get('algorithm')} == {0, 1},
          'One Normal and one Newton point required')
    _need(left.get('scope') == right.get('scope') == FROZEN_SCOPE,
          'Named NC13 physical scope differs')
    for key in ('case', 'action', 'fault_lines', 'alpha', 'clpu_multipliers',
                'time_minutes', 'state_identity', 'metadata_override'):
        _need(key in left and key in right and left[key] == right[key],
              'Query/physical-scope binding differs: ' + key)
    metadata = {**initial_metadata, **left['metadata_override']}
    phases = _mapping(metadata.get('physical_phase_nodes_by_bus'), 'physical phase inventory')
    neutrals = _mapping(metadata.get('neutral_nodes_by_bus', {}), 'neutral inventory')
    _need(phases and all(_sequence(v) and v and len(set(v)) == len(v)
                         and all(type(n) is int and n > 0 for n in v) for v in phases.values()),
          'Malformed physical phase inventory')
    dead = {str(x).lower() for x in metadata.get('deenergized_buses', [])}
    _need(dead <= phases.keys(), 'Unknown declared dead bus')
    powered = set(phases) - dead
    _need(powered, 'Empty energized bus set')
    layouts = _mapping(metadata.get('element_layout'), 'element layouts')
    planned = _mapping(metadata.get('planned_enabled'), 'planned enabled states')
    expected = {str(x).lower() for x in metadata.get('expected_element_names', [])}
    _need(expected and set(layouts) == expected == set(planned), 'Physical inventory differs')
    zero_tol = metadata.get('isolation_current_tolerance_a', 1e-8)
    _need(zero_tol == 1e-8, 'Original isolation current tolerance changed')
    raw = [left.get('raw_outputs'), right.get('raw_outputs')]
    inventories, volts = [], []
    for point, snapshot in zip((left, right), raw):
        _need(point.get('reliable') is True and point.get('status') in ('PASS', 'FAIL'),
              'Unreliable point cannot enter algorithm comparison')
        _need(isinstance(snapshot, Mapping) and snapshot.get('converged') is True,
              'Numerical convergence absent')
        _need(snapshot.get('algorithm') == point['algorithm'], 'Algorithm output label differs')
        _need(point.get('ac_audit', {}).get('status') == 'PASS', 'AC evidence not PASS')
        isolation = point.get('isolation', {})
        _need(isolation.get('status') == 'PASS', 'Isolation evidence not PASS')
        _need(set(isolation.get('certified_deenergized_buses', [])) == dead,
              'Isolation dead-bus certificate differs')
        excluded = isolation.get('excluded_node_states', {})
        for bus in dead:
            for node in set(phases[bus]) | set(neutrals.get(bus, [])):
                if node:
                    _need(excluded.get(f'{bus}.{node}') in EXCLUDED_STATES,
                          'Dead node lacks an allowed exclusion state')
        known = point.get('known_constraints', {})
        for name in REQUIRED_SECTIONS:
            item = known.get(name, {})
            _need(item.get('status') in ('PASS', 'FAIL') and not item.get('unknown'),
                  'Unknown constraint evidence: ' + name)
        _need(all(known[n]['status'] == 'PASS' for n in ('numerical', 'topology')),
              'Numerical/topology evidence not PASS')
        volts.append(_nodes(snapshot, phases, neutrals, powered))
        elements = snapshot.get('elements')
        _need(_sequence(elements), 'Missing element snapshots')
        inventory = {}
        for item in elements:
            _need(isinstance(item, Mapping) and isinstance(item.get('name'), str), 'Malformed element')
            name = item['name'].lower()
            _need(name not in inventory, 'Duplicate element')
            inventory[name] = item
        _need(set(inventory) == expected, 'Actual physical inventory differs')
        inventories.append(inventory)
    _need(raw[0].get('graph_nodes') == raw[1].get('graph_nodes')
          and raw[0].get('graph_edges') == raw[1].get('graph_edges'), 'Graph snapshots differ')
    max_v = max(abs(volts[0][n] - volts[1][n]) for n in volts[0])
    max_i = max_p = 0.
    disabled_count = enabled_count = 0
    for name in sorted(expected):
        a, b = inventories[0][name], inventories[1][name]
        layout = layouts[name]
        _need(type(planned[name]) is bool, 'Nonboolean planned enabled state')
        _need(a.get('enabled') is planned[name] and b.get('enabled') is planned[name],
              'Actual enabled state differs from frozen plan: ' + name)
        for key in ('buses', 'conductors', 'phases', 'terminals', 'conductor_open'):
            _need(key in a and a[key] == b.get(key), 'Terminal/open layout differs: ' + name + ':' + key)
        _need([str(v).lower() for v in a['buses']] == [str(v).lower() for v in layout['buses']],
              'Frozen terminal bus binding differs: ' + name)
        ncond, nterm = a['conductors'], a['terminals']
        _need(type(ncond) is int and type(nterm) is int and ncond > 0 and nterm > 0,
              'Malformed terminal dimensions')
        _need(ncond == layout['conductors'] and nterm == layout['terminals'],
              'Frozen terminal dimensions differ')
        n = ncond * nterm
        ai = _finite_array(a.get('currents_real_imag'), 2*n, name + ':left currents')
        bi = _finite_array(b.get('currents_real_imag'), 2*n, name + ':right currents')
        max_i = max(max_i, *(math.hypot(ai[i]-bi[i], ai[i+1]-bi[i+1]) for i in range(0, 2*n, 2)))
        if not planned[name]:
            disabled_count += 1
            for values in (ai, bi):
                _need(all(math.hypot(values[i], values[i+1]) <= zero_tol for i in range(0, 2*n, 2)),
                      'Disabled element has nonzero current: ' + name)
            continue
        enabled_count += 1
        _need(a.get('node_order') == b.get('node_order') and len(a.get('node_order', [])) == n,
              'Enabled node order differs')
        ap = _finite_array(a.get('powers_kw_kvar'), 2*n, name + ':left powers')
        bp = _finite_array(b.get('powers_kw_kvar'), 2*n, name + ':right powers')
        max_p = max(max_p, *(abs(x-y) for x,y in zip(ap,bp)))
    section_match = all(left['known_constraints'][k]['status'] == right['known_constraints'][k]['status']
                        for k in REQUIRED_SECTIONS)
    passed = (max_v <= limits['voltage_absolute_pu'] and max_i <= limits['current_absolute_A']
              and max_p <= limits['power_absolute_kw'] and section_match
              and left['status'] == right['status'])
    return {'status': 'PASS' if passed else 'FAIL',
            'scope': 'NC13_ENERGIZED_PHASE_V_ALL_ELEMENT_COMPLEX_I_ENABLED_ELEMENT_PQ',
            'max_node_voltage_diff_pu': max_v, 'max_complex_current_diff_A': max_i,
            'max_terminal_power_component_diff_kw_kvar': max_p,
            'constraint_classifications_match': section_match,
            'checked_energized_phase_nodes': len(volts[0]), 'checked_enabled_elements': enabled_count,
            'checked_disabled_zero_current_elements': disabled_count,
            'skipped_disabled_element_cache_fields': ['voltages_real_imag', 'powers_kw_kvar', 'yprim_real_imag'],
            'limits': dict(limits), 'new_powerflow_calls': 0}


def audit_saved_run(run, root):
    """Independently recheck a completed OR stopped run without any DSS calls.

    Does not modify the run. A correctly retained early stop can pass audit;
    acceptance_passed remains false. Missing/truncated evidence is reported as
    an audit error, never manufactured from expected outputs.
    """
    import ast
    import gzip
    import hashlib
    import json
    from pathlib import Path
    import sys

    run, root = Path(run).resolve(), Path(root).resolve()
    _need(run.parent == root / 'runs', 'Expected a direct project runs child')
    _need(not any(n == 'dss' or n.startswith(('dss.', 'opendssdirect', 'Environments')) for n in sys.modules),
          'Run saved audit in a fresh process without a numerical engine')
    sys.path.insert(0, str(root / 'src'))
    sys.path.insert(0, str(root / 'scripts'))
    from dnr.nature13_clpu_physical import check_nature13_constraints, verify_nature13_isolation
    from dnr.restoration_ac_audit import audit_ac_snapshot
    from dnr.nature34_operating_physical import verify_unchanged_attributes
    from prepare_nature13_clpu_static import build_environment

    def read(p):
        return json.loads(p.read_text(encoding='utf-8-sig'))
    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    def lines(p):
        return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()] if p.exists() else []
    def require(ok, text):
        if not ok:
            raise ComparisonInputError(text)

    report = {'run': run.name, 'new_powerflow_calls': 0, 'new_training_steps': 0,
              'status': 'AUDIT_STARTED', 'acceptance_passed': False, 'errors': []}
    try:
        frozen = read(run/'prepared.json')
        for name,digest in frozen['source_hashes'].items():
            require(sha(root/name) == sha(run/'source'/name) == digest, 'Source binding: '+name)
        input_copies = frozen['project_input_copies']
        require(set(input_copies) == set(frozen['project_input_hashes']), 'Archived input mapping inventory differs')
        for name,digest in frozen['project_input_hashes'].items():
            original = Path(name) if Path(name).is_absolute() else root/name
            archived = (run/input_copies[name]).resolve()
            require(archived.is_relative_to(run) and archived != run, 'Archived input path escapes run')
            require(sha(original) == sha(archived) == digest,
                    'Input binding: '+name)
        for name,digest in frozen['input_hashes'].items():
            require(sha(run/name) == digest, 'Prepared input binding: '+name)
        sealed = read(run/'execution_manifest.json')
        for name,digest in sealed.items():
            require(sha(run/name) == digest, 'Execution seal: '+name)
        settings = read(run/'settings.json'); config = read(run/'physical_config.json')
        require((settings['max_queries'],settings['max_numerical_entries'],settings['worker_seconds'],settings['workers']) == (16,256,240,1), 'Budget changed')
        require(settings['retry'] is False and settings['training_steps'] == 0, 'Retry/training changed')
        reference = root/'runs/20261019T021340_566784Z_nature13_curtailment_validation'
        require(config == read(reference/'config.json'), 'Physical configuration differs from original')
        require(sha(reference/'completion_manifest.json') == frozen['original_reference_sha256'], 'Old reference seal changed')
        for name,digest in read(reference/'completion_manifest.json').items():
            require(sha(reference/name) == digest, 'Old reference content changed: '+name)
        schedule = read(run/'schedule.json')
        require([s['query_id'] for s in schedule] == list(range(16)), 'Fixed 16-query coverage changed')
        require([s['group'] for s in schedule] == ['reference']*4+['initial']*6+['peak']*6, 'Query groups changed')
        with gzip.open(run/'reference_points.json.gz','rt',encoding='utf-8') as stream:
            refs = {p['id']:p for p in json.load(stream)}
        require(set(refs) == {12,14,48,50}, 'Old reference subset changed')
        for i,point_id in enumerate((12,14,48,50)):
            spec,old = schedule[i],refs[point_id]
            request = spec['request']
            require(spec['source_point_id'] == point_id and request['action'] == old['action']
                    and request['case'] == {'name':old['case'],'fault_lines':old['fault_lines']}
                    and request['alpha'] == .375 and request['algorithm'] == old['algorithm']
                    and len(request['multipliers']) == 15 and set(request['multipliers'].values()) == {1.},
                    'Old reference request changed')
        for offset,group in ((4,'initial'),(10,'peak')):
            for j,fault in enumerate(('line.632633','line.632670','line.670671')):
                env = build_environment(root,fault,3)
                state = env.initial if group == 'initial' else env.advance(env.initial,11)
                for alg in (0,1):
                    spec = schedule[offset+2*j+alg]
                    require(spec['fault_line'] == fault and spec['request'] == env.snapshot_request(state,0,algorithm=alg),
                            'Initial/peak request changed')
        attempts = lines(run/'attempts.jsonl'); events = lines(run/'numerical_events.jsonl')
        started = [r['query_id'] for r in attempts if r['event'] == 'STARTED']
        require(started == list(range(len(started))) and len(started) <= 16, 'Retried/nonprefix attempts')
        rows,partials = {},{}
        for p in sorted((run/'points').glob('*.json.gz')):
            with gzip.open(p,'rt',encoding='utf-8') as stream: row = json.load(stream)
            if p.name.endswith('_partial.json.gz'):
                qid = row['query_id']
                require(qid not in partials and p.name == f'{qid:03d}_partial.json.gz', 'Duplicate/misnamed partial evidence')
                partials[qid] = row
                continue
            qid = row['query_id'];require(qid not in rows and p.stem.split('.')[0] == f'{qid:03d}', 'Duplicate/misnamed point')
            rows[qid] = row
        require(set(rows) <= set(started), 'Point without started request')
        require(set(partials) <= set(started) and not set(partials) & set(rows), 'Partial evidence cannot count as a completed point')
        complete_ids = [r['query_id'] for r in attempts if r['event'] == 'COMPLETED']
        require(len(set(complete_ids)) == len(complete_ids) and set(complete_ids) == set(rows), 'Completed request/output mismatch')
        metadata = read(run/'metadata.json') if (run/'metadata.json').exists() else None
        require(not rows or metadata is not None, 'Saved points lack metadata')
        replay = []
        for qid,row in sorted(rows.items()):
            spec = schedule[qid]; request = spec['request']
            for key in ('action','alpha','algorithm','time_minutes','state_identity'):
                require(row[key] == request[key], f'Point {qid} request mismatch: '+key)
            require(row['case'] == request['case']['name'] and row['fault_lines'] == request['case']['fault_lines']
                    and row['clpu_multipliers'] == request['multipliers'] and row['scope'] == FROZEN_SCOPE,
                    f'Point {qid} case/multiplier/scope mismatch')
            merged = {**metadata,**row['metadata_override']};raw = row['raw_outputs']
            iso = verify_nature13_isolation(raw,merged)
            known = check_nature13_constraints(raw,merged,**config['constraint_tolerances'])
            ac = audit_ac_snapshot(raw,merged,**config['ac_audit_tolerances'])
            require(_saved_equal(iso,row['isolation']) and _saved_equal(known,row['known_constraints'])
                    and _saved_equal(ac,row['ac_audit']),
                    f'Point {qid} saved labels differ from independent replay')
            reliable = (raw['converged'] and ac['status'] == iso['status'] == 'PASS'
                        and known['numerical']['status'] == known['topology']['status'] == 'PASS'
                        and not any(known[k]['status'] == 'UNKNOWN' or known[k].get('unknown') for k in REQUIRED_SECTIONS))
            status = ('PASS' if known['known_constraints_satisfied'] is True else 'FAIL') if reliable else 'UNKNOWN'
            require(row['reliable'] is bool(reliable) and row['status'] == status, f'Point {qid} reliability classification changed')
            before,after = row['load_readback']['before'],row['load_readback']['postsolve']
            require(set(before) == set(after) == set(metadata['load_reference']) == set(request['multipliers']) and len(after) == 15,
                    'Full fifteen-load readback missing')
            verify_unchanged_attributes(before,after)
            for name,ref in metadata['load_reference'].items():
                for key in ('nominal_kw','nominal_kvar'):
                    require(math.isclose(before[name][key],ref[key],rel_tol=1e-12,abs_tol=1e-8)
                            and math.isclose(after[name][key],ref[key]*request['multipliers'][name],rel_tol=1e-12,abs_tol=1e-8),
                            f'Point {qid} readback differs: '+name+':'+key)
            generators = {e['name'].lower():e for e in raw['elements'] if e['name'].lower().startswith('generator.')}
            require(set(generators) == {n.lower() for n in metadata['generator_reference']}, 'Seven-DER inventory differs')
            for name,ref in metadata['generator_reference'].items():
                observed = generators[name.lower()]['operating_setpoint']
                require(math.isclose(observed['kw'],.375*ref['available_p_kw'],rel_tol=1e-12,abs_tol=1e-8)
                        and math.isclose(observed['kvar'],.375*ref['reference_q_kvar'],rel_tol=1e-12,abs_tol=1e-8),
                        'DER setpoint readback differs: '+name)
            require(row['nominal_load_ledger']['denominator_nominal_kw'] == 3466., 'Normal denominator changed')
            replay.append({'query_id':qid,'status':status,'reliable':bool(reliable),'labels_exact':True})
            if status == 'UNKNOWN' or (spec['group'] in ('reference','initial') and status != 'PASS'):
                require(not any(i > qid for i in started), 'Execution continued after required point stop')

        partial_replay = []
        for qid,partial in sorted(partials.items()):
            request = schedule[qid]['request']
            require(partial.get('request') == {k:request[k] for k in ('action','case','alpha','algorithm','multipliers','time_minutes','state_identity')},
                    'Partial request binding differs')
            require(not any(i > qid for i in started), 'Execution continued after failed partial point')
            note = {'query_id':qid,'phase':partial.get('phase'),'raw_retained':'raw_outputs' in partial,
                    'counted_as_completed':False,'checker_replay':{}}
            if 'raw_outputs' in partial and metadata is not None and 'metadata_override' in partial:
                merged = {**metadata,**partial['metadata_override']}
                for key,checker,tolerances in [('isolation',verify_nature13_isolation,{}),
                        ('known_constraints',check_nature13_constraints,config['constraint_tolerances']),
                        ('ac_audit',audit_ac_snapshot,config['ac_audit_tolerances'])]:
                    try:
                        fresh = checker(partial['raw_outputs'],merged,**tolerances)
                    except Exception as exc:
                        note['checker_replay'][key] = {'exception':repr(exc)}
                    else:
                        if key in partial:
                            require(_saved_equal(partial[key],fresh), 'Partial saved checker label differs: '+key)
                        note['checker_replay'][key] = {'status':fresh.get('status'),
                            'known_constraints_satisfied':fresh.get('known_constraints_satisfied'),
                            'saved_result_present':key in partial}
            partial_replay.append(note)

        calls,closed = {},{}
        for event in events:
            if event['event'] == 'call_started':
                cid = event['call_id'];require(cid not in calls, 'Repeated numerical call id');calls[cid] = event
            elif event['event'] in ('call_returned','call_failed'):
                cid = event['call_id'];require(cid in calls and cid not in closed, 'Unbound/repeated call completion')
                require(event['counts'] == calls[cid]['counts'] and event['context'] == calls[cid]['context'], 'Call completion binding differs')
                closed[cid] = event
        reserved = sum(c['counts']['numerical_solution_entries'] for c in calls.values())
        returned = sum(c['counts']['numerical_solution_entries'] for c in closed.values() if c['event'] == 'call_returned')
        require(reserved <= 256 and returned <= reserved, 'Numerical-entry budget exceeded')
        for cid,event in calls.items():
            counts = event['counts']
            expected = sum(counts.get(k,0) for k in ('python_solve_commands','script_solve_commands','text_solve_commands','calcvoltagebases_entries'))
            require(counts['numerical_solution_entries'] == expected, 'Entry arithmetic changed')
            if counts.get('compile_commands'):
                require((counts['script_solve_commands'],counts['calcvoltagebases_entries']) == (1,1), 'Compile hidden-entry inventory changed')
        for qid,row in rows.items():
            own = [c for c in closed.values() if c['event'] == 'call_returned' and c['context'].get('query_id') == qid]
            for key,value in row['actual_call_counts'].items():
                require(sum(c['counts'].get(key,0) for c in own) == value, f'Point {qid} call counter differs: '+key)
        query_events = [e['context']['query_id'] for e in events if e['event'] == 'query_started']
        require(query_events == started[:len(query_events)] and len(started)-len(query_events) <= 1
                and all(qid not in rows for qid in started[len(query_events):]),
                'Numerical/query attempt ledgers differ')

        tree = ast.parse((root/'scripts/audit_nature13_curtailment_validation.py').read_text(encoding='utf-8'))
        functions = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('require','algorithm_difference')]
        namespace = {'math':math};exec(compile(ast.Module(body=functions,type_ignores=[]),'<frozen-pure-comparison>','exec'),namespace)
        comparisons = read(run/'comparisons.json') if (run/'comparisons.json').exists() else []
        for saved in comparisons:
            if saved['kind'] == 'old_reference':
                fresh = namespace['algorithm_difference'](refs[saved['source_point_id']]['raw_outputs'],
                    rows[saved['query_id']]['raw_outputs'],config['cross_algorithm_tolerances'])
            else:
                require(saved['kind'] == 'cross_algorithm', 'Unknown comparison kind')
                a,b = saved['query_ids']
                fresh = compare_isolated_points(rows[a],rows[b],metadata,config['cross_algorithm_tolerances'])
            require(all(saved[k] == v for k,v in fresh.items()), 'Saved numerical comparison differs')
        waits = []
        for path in sorted(run.glob('wait_nc13_*.json')):
            cert = read(path);qid = cert['source_query_id'];spec = schedule[qid];source = rows[qid]
            require(spec['group'] == 'initial' and source['algorithm'] == 0 and source['status'] == 'PASS', 'WAIT lacks a valid initial source')
            require(cert['actual_additional_queries'] == cert['actual_additional_entries'] == 0, 'WAIT reuse charged unexpected queries')
            env = build_environment(root,spec['fault_line'],3)
            expected_states = env.plan_states(env.initial,[0]*30)
            require(cert['plan']['status'] == 'COMPLETE' and cert['plan']['actions'] == [0]*30
                    and cert['plan']['states'] == [s.to_dict() for s in expected_states], 'WAIT history/action coverage differs')
            samples,evidence = cert['plan']['samples'],cert['logical_evidence']
            require(len(samples) == len(evidence) == 90, 'WAIT is not full 30x3 coverage')
            index = 0
            for state in expected_states[1:]:
                for offset in (0,30,60):
                    t = (state.step-1)*60+offset; request = env.snapshot_request(state,t,algorithm=0)
                    for key in ('action','case','alpha','algorithm','multipliers'):
                        require(request[key] == spec['request'][key], 'WAIT reused unequal physical input')
                    require(samples[index] == {'step':state.step,'time_seconds':t,'point_id':qid,'status':'PASS','state':state.to_dict()}, 'WAIT sample identity differs')
                    require(evidence[index]['identity'] == request['state_identity'] and evidence[index]['source_query_id'] == qid
                            and evidence[index]['ac_evidence_source_query_id'] == qid
                            and evidence[index]['same_physical_input'] is True
                            and evidence[index]['current_constraints'] == evidence[index]['current_isolation'] == 'PASS', 'WAIT logical certificate differs')
                    index += 1
            energy = math.fsum(env.service_kw(s) / 60 for s in expected_states[1:])
            require(math.isclose(cert['plan']['energy_kwh'],energy,rel_tol=1e-12,abs_tol=1e-9), 'WAIT normal energy differs')
            waits.append(path.name)
        summary = read(run/'worker_summary.json') if (run/'worker_summary.json').exists() else None
        controller = read(run/'controller.json')
        if summary:
            require(summary['attempted_query_ids'] == started and set(summary['completed_query_ids']) == set(rows)
                    and summary['not_run_query_ids'] == [i for i in range(16) if i not in started]
                    and summary['failed_attempt_query_ids'] == [i for i in started if i not in rows], 'Attempted/failed/NOT_RUN classification differs')
            if summary.get('budget'):
                require(summary['budget']['reserved_numerical_entries'] == reserved
                        and summary['budget']['returned_numerical_entries'] == returned, 'Summary entry totals differ')
        accepted = bool(summary and summary['status'] == 'COMPLETE_ACCEPTANCE_PENDING_INDEPENDENT_AUDIT' and not controller['timed_out'])
        if accepted:
            require(len(rows) == 16 and len(waits) == 3 and len(comparisons) == 12
                    and all(c['status'] == 'PASS' for c in comparisons) and set(closed) == set(calls)
                    and all(c['event'] == 'call_returned' for c in closed.values()), 'Complete acceptance evidence missing')
        report.update(status='VERIFIED_SAVED_ACCEPTANCE' if accepted else 'VERIFIED_SAVED_STOP', acceptance_passed=accepted,
            worker_status=summary['status'] if summary else None, controller=controller,
            attempted_query_ids=started, completed_query_ids=sorted(rows),
            failed_attempt_query_ids=[i for i in started if i not in rows],
            not_run_query_ids=[i for i in range(16) if i not in started], point_replay=replay,
            partial_evidence=partial_replay,
            reserved_numerical_entries=reserved, returned_numerical_entries=returned,
            unresolved_call_ids=sorted(set(calls)-set(closed)), wait_certificates=waits,
            compared_records=len(comparisons), execution_manifest_sha256=sha(run/'execution_manifest.json'))
    except Exception as exc:
        report.update(status='AUDIT_EVIDENCE_ERROR',acceptance_passed=False)
        report['errors'].append(repr(exc))
    report['forbidden_modules_loaded'] = [n for n in sys.modules if n == 'dss' or n.startswith(('dss.','opendssdirect','Environments'))]
    if report['forbidden_modules_loaded']:
        report.update(status='AUDIT_EVIDENCE_ERROR',acceptance_passed=False)
        report['errors'].append('Numerical module imported during saved audit')
    return report
