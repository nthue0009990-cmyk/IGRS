"""Literature current extension and bounded development feasibility checks."""
import heapq
import networkx as nx
import numpy as np
import pandapower as pp
import pandapower.networks as pn
from pandapower.powerflow import LoadflowNotConverged

from .audit import topology_status, verify_source_equivalence


def extended_network(source, config, load_scale):
    net = pn.case33bw()
    verify_source_equivalence(net, source)
    ratings = np.full(len(net.line), np.nan)
    for group in config['capacity_groups']:
        lo, hi = group['branch_ids_inclusive']
        if not 1 <= lo <= hi <= len(ratings):
            raise ValueError('Capacity branch IDs out of range')
        if np.isfinite(ratings[lo - 1:hi]).any():
            raise ValueError('Overlapping capacity groups')
        ratings[lo - 1:hi] = group['ampere']
    if not np.isfinite(ratings).all() or (ratings <= 0).any():
        raise ValueError('Missing/invalid current limits')
    net.line['max_i_ka'] = ratings / 1000
    net.load[['p_mw', 'q_mvar']] *= load_scale
    if not np.all(net.ext_grid.vm_pu == config['slack_vm_pu']):
        raise ValueError('Slack voltage mismatch')
    return net


def branch_exchange_neighbors(endpoints, opened):
    opened = tuple(sorted(opened))
    graph = nx.Graph()
    graph.add_nodes_from(range(33))
    for index, (a, b) in enumerate(endpoints, 1):
        if index not in opened:
            graph.add_edge(a, b, branch_id=index)
    if not nx.is_tree(graph):
        raise ValueError('Parent topology is not a spanning tree')
    neighbors = set()
    for close_id in opened:
        a, b = endpoints[close_id - 1]
        path = nx.shortest_path(graph, a, b)
        for u, v in zip(path, path[1:]):
            cut_id = graph[u][v]['branch_id']
            neighbors.add(tuple(sorted((set(opened) - {close_id}) | {cut_id})))
    return sorted(neighbors)


def validate_solution(net, config):
    v = net.res_bus.vm_pu.to_numpy().copy()
    active = net.line.in_service.to_numpy()
    flows = net.res_line
    if not np.isfinite(net.res_bus[['vm_pu', 'va_degree']].to_numpy()).all():
        raise AssertionError('Non-finite bus results')
    if not np.isfinite(flows.loc[active, ['p_from_mw', 'q_from_mvar', 'p_to_mw', 'q_to_mvar', 'i_from_ka', 'i_to_ka']]).all().all():
        raise AssertionError('Non-finite energized branch results')
    currents = np.zeros(len(flows))
    currents[active] = flows.loc[active, ['i_from_ka', 'i_to_ka']].max(axis=1).to_numpy() * 1000
    residual = net.res_bus[['p_mw', 'q_mvar']].to_numpy().copy()
    for index, line in net.line.iterrows():
        if not line.in_service:
            continue
        row = flows.loc[index]
        residual[int(line.from_bus)] += [row.p_from_mw, row.q_from_mvar]
        residual[int(line.to_bus)] += [row.p_to_mw, row.q_to_mvar]
    loss = float(flows.pl_mw.sum())
    pl = net.res_ext_grid.p_mw.sum() - net.res_load.p_mw.sum() - loss
    ql = net.res_ext_grid.q_mvar.sum() - net.res_load.q_mvar.sum() - flows.ql_mvar.sum()
    balance = float(max(abs(pl), abs(ql), np.max(np.abs(residual))))
    if balance > config['power_balance_tolerance_mw_mvar']:
        raise AssertionError(f'Power balance residual {balance}')
    return dict(voltage=v, current_a=currents, loss_mw=loss, balance_residual=balance)


def classify(v, current_a, ratings_a, config, multiplier):
    if (not np.isfinite(v).all() or not np.isfinite(current_a).all() or
        not np.isfinite(ratings_a).all() or (ratings_a <= 0).any() or
        (current_a < 0).any() or not np.isfinite(multiplier) or multiplier <= 0):
        raise ValueError('Non-finite or invalid constraint inputs')
    lo, hi = config['audit_voltage_bounds_pu']
    vt = config['classification_tolerance_voltage_pu']
    it = config['classification_tolerance_current_a']
    vbad = (v < lo - vt) | (v > hi + vt)
    limit = ratings_a * multiplier
    ibad = current_a > limit + it
    return dict(capacity_multiplier=multiplier, feasible=bool(not vbad.any() and not ibad.any()),
                voltage_violating_bus_count=int(vbad.sum()), thermal_violating_branch_count=int(ibad.sum()),
                undervoltage_pu=float(max(0, lo - v.min())), overvoltage_pu=float(max(0, v.max() - hi)),
                max_current_excess_a=float(max(0, np.max(current_a - limit))),
                max_loading_ratio=float(np.max(current_a / limit)))


def evaluate(net, opened, config, on_call):
    net.line['in_service'] = [i not in opened for i in range(1, 38)]
    topo = topology_status(net)
    if not topo['radial'] or not topo['connected']:
        raise ValueError('Invalid topology rejected before power flow')
    results = {}
    for algorithm in config['powerflow']['algorithms']:
        options = {k: v for k, v in config['powerflow'].items() if k != 'algorithms'}
        try:
            pp.runpp(net, algorithm=algorithm, check_connectivity=True, **options)
            if not net.converged:
                raise LoadflowNotConverged('converged=False')
            result = validate_solution(net, config)
            results[algorithm] = result
            on_call(algorithm, 'converged', result, None)
        except LoadflowNotConverged as exc:
            on_call(algorithm, 'nonconverged', None, str(exc))
    if len(results) != 2:
        return None
    nr, bfsw = results['nr'], results['bfsw']
    errors = dict(voltage=float(np.max(np.abs(nr['voltage'] - bfsw['voltage']))),
                  current=float(np.max(np.abs(nr['current_a'] - bfsw['current_a']))),
                  loss=abs(nr['loss_mw'] - bfsw['loss_mw']))
    if (errors['voltage'] > config['cross_solver_voltage_tolerance_pu'] or
        errors['current'] > config['cross_solver_current_tolerance_a'] or
        errors['loss'] > config['cross_solver_loss_tolerance_mw']):
        raise AssertionError(f'Solver disagreement: {errors}')
    nr['cross_solver_errors'] = errors
    return nr


def probe(net, config, on_call, on_result):
    """All evaluated states retained; a bounded search is not a proof of absence."""
    endpoints = list(zip(net.line.from_bus.astype(int), net.line.to_bus.astype(int)))
    ratings = net.line.max_i_ka.to_numpy() * 1000
    seen = set()
    frontier = []
    pending = [tuple(config['search']['initial_open_branch_ids'])]
    budget = config['search']['max_unique_topologies_per_scenario']
    while pending or frontier:
        if not pending:
            *_, parent = heapq.heappop(frontier)
            pending = branch_exchange_neighbors(endpoints, parent)
        opened = pending.pop(0)
        if opened in seen:
            continue
        seen.add(opened)
        result = evaluate(net, opened, config, lambda *args: on_call(opened, *args))
        if result is not None:
            metrics = [classify(result['voltage'], result['current_a'], ratings, config, m)
                       for m in config['capacity_sensitivity_multipliers']]
            result['sensitivity'] = metrics
            rank = (-float(result['voltage'].min()), float(max(result['current_a'] / ratings)), result['loss_mw'])
        else:
            # Failed states remain eligible for topology exploration, after converged states.
            rank = (float('inf'), float('inf'), float('inf'))
        heapq.heappush(frontier, (*rank, opened))
        on_result(opened, result)
        if len(seen) >= budget:
            break
    return len(seen)
