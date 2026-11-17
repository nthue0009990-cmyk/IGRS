"""Bounded terminal-target then sequencing MILP for the documented proxy.

No Beam result, historical winning plan, physical cache, or physics runtime is
an input. Returned plans remain uncertified until the common AC validator runs.
"""
import time

from .clpu_curves import multiplier
from .nature34_free_sequence import enumerate_switch_space
from .nature34_joint_sequence import _integer_weights
from .nature34_two_stage_linear import LinModel, expression
from .nature34_two_stage_contract import check_target_plan

VERSION = 'nc34-two-stage-electrical-proxy-planner-v1'
STAGE1_SECONDS = 180.
STAGE2_SECONDS = 540.
NODE_LIMIT = 100000
SOFTWARE_RESIDUAL_TOLERANCE = 1e-6


def _variable_or_constant(model, name, values, selectors):
    unique = set(values)
    if len(unique) == 1:
        return expression(next(iter(unique)))
    value = model.var(name, 0, 1, integer=True)
    model.add(value - sum(v * s for v, s in zip(values, selectors)), 0, 0, name + ':select')
    return value


def topology_rows(env):
    rows = enumerate_switch_space(env.static, env.loads, env.scenario)['states'].values()
    reachable = set(int(state) >> 10 for state in env.graph.states)
    return [r for r in sorted(rows, key=lambda x: x['mask'])
            if r['forest'] is True and r['mask'] in reachable]


def topology_layer(model, env, network, rows, tag):
    selectors = [model.var(f'{tag}:mask:{r["mask"]}', 0, 1, integer=True) for r in rows]
    model.add(sum(selectors), 1, 1, tag + ':one_topology')
    xs = [_variable_or_constant(model, f'{tag}:x:{i}', [(r['mask'] >> i) & 1 for r in rows], selectors)
          for i in range(9)]
    powered = [set(b.lower() for b in r['source_reachable_buses']) for r in rows]
    bus_on = {b.lower(): _variable_or_constant(model, f'{tag}:bus:{b}',
               [int(b.lower() in buses) for buses in powered], selectors) for b in env.static['nodes']}
    fault_pairs = {frozenset(env.static['original_lines'][name]) for name in env.scenario['fault_lines']}
    switches = {frozenset(edge): i for i, edge in enumerate(env.static['switch_edges'])}
    edge_on = {}
    for item in network['lines'] + network['transformers']:
        if 'bus1' in item:
            b1, b2 = item['bus1'], item['bus2']
        else:
            b1, b2 = [w['terminal']['bus'] for w in item['windings']]
        edge = frozenset((b1, b2)); control = switches.get(edge)
        enabled = [int(b1 in buses and b2 in buses and edge not in fault_pairs and
                       (control is None or (row['mask'] >> control) & 1))
                   for row, buses in zip(rows, powered)]
        edge_on[item['name'].lower()] = _variable_or_constant(model, tag + ':' + item['name'], enabled, selectors)
    readiness = [model.var(f'{tag}:ready:{i}', 0, 1, integer=True) for i in range(10)]
    dispatch = {name.lower(): i for i, name in enumerate(env.static['dispatch_loads'])}
    load_on = {}
    for name, load in env.loads.items():
        bus = bus_on[load['bus'].lower()]
        if name.lower() in dispatch:
            r = readiness[dispatch[name.lower()]]
            y = model.var(tag + ':load:' + name, 0, 1, integer=True)
            model.add(y - bus, ub=0, name=tag + ':load_bus:' + name)
            model.add(y - r, ub=0, name=tag + ':load_ready:' + name)
            model.add(y - r - bus, lb=-1, name=tag + ':load_and:' + name)
        else:
            y = bus
        load_on[name.lower()] = y
    return {'selectors':selectors, 'rows':rows, 'switches':xs, 'readiness':readiness,
            'bus_on':bus_on, 'edge_on':edge_on, 'load_on':load_on}


def _electrical(model, env, network, proxy, layer, demands, tag):
    from .nature34_two_stage_electrical_proxy import add_electrical_snapshot
    return add_electrical_snapshot(model, network, proxy, layer['bus_on'], layer['edge_on'], demands, tag,
        condition_multiplier=env.load_multiplier,
        voltage_limits=(proxy['physical_voltage_limits']['min_pu'], proxy['physical_voltage_limits']['max_pu']))


def _energy_weights(env):
    return _integer_weights(env.loads)[0]


def build_terminal_model(env, network, proxy):
    model = LinModel(); layer = topology_layer(model, env, network, topology_rows(env), 'terminal')
    for name in env.initial_energized:
        model.add(layer['load_on'][name.lower()], 1, 1, 'terminal:retain_initial:' + name)
    demands = {name:[{'pickup':on, 'coefficient':1.}] for name, on in layer['load_on'].items()}
    electric = _electrical(model, env, network, proxy, layer, demands, 'terminal')
    weights = _energy_weights(env)
    service = sum(weights[n] * on for n, on in layer['load_on'].items())
    initial_mask = env.initial.command >> 10
    switches = sum((1 - x) if (initial_mask >> i) & 1 else x for i, x in enumerate(layer['switches']))
    command = sum((1 << (10 + i)) * x for i, x in enumerate(layer['switches'])) + sum(
        (1 << i) * r for i, r in enumerate(layer['readiness']))
    return model, layer, [-service, switches, command], electric


def build_sequence_model(env, network, proxy, target_mask, required_names, *, deadline_monotonic):
    model = LinModel(); initial_mask = env.initial.command >> 10
    changes = initial_mask ^ target_mask
    rows = [r for r in topology_rows(env) if not ((r['mask'] ^ initial_mask) & ~changes)]
    previous_x = [expression((initial_mask >> i) & 1) for i in range(9)]
    previous_r = [expression((env.initial.command >> i) & 1) for i in range(10)]
    previous_y = {n.lower(): expression(int(n in env.initial_energized)) for n in env.load_names}
    pickups = {n.lower(): [] for n in env.load_names}
    layers, electrical = [], []
    weights = _energy_weights(env); energy, operations = expression(0), expression(0)
    for k in range(30):
        if time.perf_counter() >= deadline_monotonic:
            raise TimeoutError('Sequence model construction exhausted its deadline')
        tag = f't{k:02d}'; layer = topology_layer(model, env, network, rows, tag)
        switch_ops = expression(0)
        for i, x in enumerate(layer['switches']):
            direction = 1 if (target_mask >> i) & 1 else -1
            change = direction * (x - previous_x[i])
            model.add(change, 0, 1, tag + f':line_monotone:{i}')
            switch_ops += change
        ready_ops = []
        for i, (r, previous) in enumerate(zip(layer['readiness'], previous_r)):
            delta = model.var(tag + f':ready_change:{i}', 0, 1, integer=True)
            model.add(delta - r + previous, lb=0, name=tag + f':xor_a:{i}')
            model.add(delta + r - previous, lb=0, name=tag + f':xor_b:{i}')
            model.add(delta - r - previous, ub=0, name=tag + f':xor_c:{i}')
            model.add(delta + r + previous, ub=2, name=tag + f':xor_d:{i}')
            ready_ops.append(delta)
        slot_ops = switch_ops + sum(ready_ops)
        model.add(slot_ops, 0, 1, tag + ':one_original_action_or_wait')
        operations += slot_ops
        for name, y in layer['load_on'].items():
            z = model.var(tag + ':pickup:' + name, 0, 1, integer=True)
            model.add(z - y + previous_y[name], 0, 0, tag + ':service_monotone:' + name)
            pickups[name].append(z)
        energy += sum(weights[n] * y for n, y in layer['load_on'].items())
        for offset in (0, 30, 60):
            demands = {}
            for original in env.load_names:
                name = original.lower()
                if original in env.initial_energized:
                    demands[name] = [{'pickup':expression(1), 'coefficient':1.}]
                else:
                    demands[name] = [{'pickup':z, 'coefficient':multiplier(env.event, (k-h)*60+offset)}
                                     for h, z in enumerate(pickups[name])]
            electrical.append(_electrical(model, env, network, proxy, layer, demands, tag + f':s{offset}'))
        layers.append(layer)
        previous_x, previous_r, previous_y = layer['switches'], layer['readiness'], layer['load_on']
    for i, x in enumerate(previous_x):
        model.add(x, (target_mask >> i) & 1, (target_mask >> i) & 1, f'final:line:{i}')
    for name in required_names:
        model.add(previous_y[name.lower()], 1, 1, 'final:required:' + name)
    final_service = sum(weights[n] * y for n, y in previous_y.items())
    return model, layers, [-energy, -final_service, operations], electrical


def _lexicographic(model, objectives, deadline, callback, tag):
    results, incumbent = [], None
    for index, objective in enumerate(objectives):
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        result = model.solve(objective, time_limit=remaining, node_limit=NODE_LIMIT, mip_rel_gap=0.)
        result['objective_index'] = index
        residuals = result['residuals']
        usable = result['status'] in (0, 1) and residuals is not None and max(
            residuals['max_variable_bound_violation'], residuals['max_integer_violation'],
            residuals['max_row_violation']) <= SOFTWARE_RESIDUAL_TOLERANCE
        result['usable_proxy_incumbent'] = usable
        results.append(result)
        callback(tag + f'_solve_{index}', result)
        if usable:
            incumbent = result['x']
        if not usable or result['status'] != 0:
            break
        # The three primary objectives have integer coefficients/values.
        value = round(objective.value(incumbent))
        if abs(value - objective.value(incumbent)) > SOFTWARE_RESIDUAL_TOLERANCE:
            raise ValueError('Lexicographic integer objective failed independent readback')
        model.add(objective, value, value, tag + f':fix_objective:{index}')
    return {'solves':results, 'incumbent':incumbent,
            'all_lexicographic_stages_optimal':len(results)==len(objectives) and all(
                r['status']==0 and r['usable_proxy_incumbent'] for r in results)}


def solve_two_stage(env, network, proxy, *, deadline_monotonic, record_callback=None):
    callback = record_callback or (lambda name, record: None)
    started = time.perf_counter()
    result = {'version':VERSION,'status':'IN_PROGRESS','actions':None,'stage1':None,'stage2':None,
              'static_plan_contract':None,'new_physical_points':0,'new_training':0,
              'remaining_exact_ties':'deterministic single-thread backend incumbent; no claim of action-lexicographic optimality'}
    first_deadline = min(deadline_monotonic, started + STAGE1_SECONDS)
    model, layer, objectives, electrical = build_terminal_model(env, network, proxy)
    callback('stage1_model', {**model.to_record(), 'electrical_coverage':electrical,
        'objectives':[{'terms':sorted(e.terms.items()),'constant':e.constant} for e in objectives]})
    first = _lexicographic(model, objectives, first_deadline, callback, 'stage1')
    result['stage1'] = {k:v for k,v in first.items() if k!='incumbent'}
    if first['incumbent'] is None:
        result['status']='NO_TERMINAL_PROXY_INCUMBENT'; return result
    x = first['incumbent']
    target_mask = sum(round(v.value(x)) << i for i,v in enumerate(layer['switches']))
    required = [n for n in env.load_names if round(layer['load_on'][n.lower()].value(x))]
    result['stage1'].update(target_switch_mask=target_mask, required_load_names=required)
    second_deadline = min(deadline_monotonic, time.perf_counter() + STAGE2_SECONDS)
    try:
        model, layers, objectives, electrical = build_sequence_model(env, network, proxy, target_mask, required,
                                                                     deadline_monotonic=second_deadline)
    except TimeoutError as exc:
        result['status']='SEQUENCE_BUILD_TIME_LIMIT'; result['stage2']={'error':str(exc)}; return result
    callback('stage2_model', {**model.to_record(), 'electrical_coverage':electrical,
        'objectives':[{'terms':sorted(e.terms.items()),'constant':e.constant} for e in objectives]})
    second = _lexicographic(model, objectives, second_deadline, callback, 'stage2')
    result['stage2'] = {k:v for k,v in second.items() if k!='incumbent'}
    if second['incumbent'] is None:
        result['status']='NO_SEQUENCE_PROXY_INCUMBENT'; return result
    values = second['incumbent']; command = env.initial.command; actions=[]
    for layer in layers:
        next_command = sum(round(v.value(values)) << (10+i) for i,v in enumerate(layer['switches'])) + sum(
            round(v.value(values)) << i for i,v in enumerate(layer['readiness']))
        difference = command ^ next_command
        if difference & (difference-1):
            raise ValueError('MILP readback changed multiple controls in a slot')
        actions.append(difference.bit_length()); command=next_command
    checked = check_target_plan(env, actions, target_switch_mask=target_mask, required_load_names=required)
    result.update(status='PROXY_PLAN_UNCERTIFIED',actions=actions,static_plan_contract=checked,
                  elapsed_seconds=time.perf_counter()-started)
    return result
