# Isolated NC123 dimension adapter. Source and exact replacements: artifacts/nature123_igrs_migration_20261107/derivation.json
"""Bounded graph-only composition of whole-load staged pickup plans.

No oracle, physical outcome, model, incumbent or reference quality is an input.
The ten original readiness bits are changed only while their loads are off.
All proposals, including illegal, duplicate and over-horizon ones, are retained.
"""
from __future__ import annotations

from copy import deepcopy
import math

from .nature123_igrs_search import SearchConfig, graph_plan, initial_candidates

VERSION = 'clpu-staged-whole-load-candidates-v2'
ORIGIN = 'staged_pickup'
CONFIG = {
    'route_limit': 2, 'route_operation_limit': 6, 'initial_lookahead_depth': 6,
    'blocked_load_counts': [1, 2], 'timings': ['NATURAL', 'ONE_TAU'],
    'restore_orders': ['SMALL_FIRST', 'LARGE_FIRST'],
    'max_raw_proposals_per_route': 12, 'max_unique_candidates': 24,
    'horizon': 30, 'slot_seconds': 60, 'sample_seconds': 30,
    'load_order': 'nominal_kw_descending_then_original_dispatch_action',
    'route_order': 'original_graph_priority_then_switch_route_lexicographic',
    'route_diversity': 'distinct_terminal_switch_mask',
    'route_legality': 'prepared_full_plan_only;bare_switch_route_is_diagnostic',
    'timing_rule': 'NATURAL=1;ONE_TAU=ceil(event.tau_seconds/slot_seconds)',
    'wait_insertion': 'gap_slots_minus_one_before_each_restore',
    'hold_generation': 'once_per_route_and_blocked_count_before_all_restore_prefixes',
    'physics': 'unchanged_whole_load_binary_actions_no_reinterruption',
}


def _prefix(actions):
    last = max((i for i, action in enumerate(actions) if action), default=-1)
    return list(actions[:last + 1])


def _batches(env, info):
    rows = []
    for before, after in zip(info['states'], info['states'][1:]):
        names = [name for name, old, new in zip(env.load_names, before.pickups, after.pickups)
                 if old is None and new is not None]
        if names:
            rows.append({'time_seconds': (after.step - 1) * env.slot_seconds,
                'step': after.step, 'load_names': names,
                'nominal_kw': math.fsum(float(env.loads[name]['nominal_kw']) for name in names)})
    return rows


def _seed_routes(env):
    seeds = []
    for plan in initial_candidates(env, env.initial, SearchConfig()):
        info = graph_plan(env, env.initial, plan)
        route = tuple(a for a in plan if env.graph.nl < a < env.graph.action_count)
        seeds.append((info['priority'], route, plan, info['states'][-1].command >> env.graph.nl))
    seeds.sort(key=lambda row: (row[0], row[1]))
    rows, masks = [], set()
    selected = 0
    for priority, route, plan, mask in seeds:
        row = {'seed_index': len(rows), 'seed_actions': list(plan),
            'seed_graph_energy_kwh': -priority[0], 'switch_route': list(route),
            'terminal_switch_mask': mask, 'selected': False, 'route_rank': None}
        if mask in masks:
            row['status'] = 'DUPLICATE_TERMINAL_TOPOLOGY'
        elif selected >= CONFIG['route_limit']:
            row['status'] = 'OUTSIDE_FIXED_ROUTE_LIMIT'
        else:
            # Route selection is committed without inspecting later physical
            # results. A bare route can be illegal although its prepared full
            # plan is legal, so it is not a candidate-generation gate.
            masks.add(mask)
            row.update(selected=True, route_rank=selected,
                       status='UNAVAILABLE_ROUTE_LENGTH' if len(route) > 6 else 'SELECTED')
            selected += 1
            if len(route) <= 6:
                terminal = (mask << env.graph.nl) | (env.initial.command & env.graph.ready_mask)
                row['projected_terminal_command'] = terminal
                row['projected_terminal_energized'] = sorted(env.energized(terminal))
                try:
                    state = env.initial
                    for action in route:
                        state = env.advance(state, action)
                    if state.command != terminal:
                        raise AssertionError('Switch-only route terminal identity changed')
                    row['bare_route_status'] = 'GRAPH_LEGAL'
                except ValueError as exc:
                    row.update(bare_route_status='GRAPH_ILLEGAL', bare_route_error=str(exc))
        rows.append(row)
    return rows


def generate_staged_pickup_candidates(env):
    """Return all proposals plus unique legal full-horizon candidates.

The selected route skeleton is graph-only. Every actual blocked/restore plan
is replayed independently; no seed trajectory is reused after readiness edits.
No objective pruning occurs here, so a low-energy hold cannot suppress its
fully generated restoration successors.
"""
    if (env.horizon, env.slot_seconds, env.sample_seconds, env.initial.step) != (30, 60, 30, 0):
        raise ValueError('Staged generator requires the unchanged initial 30/60/30 scope')
    dispatch = list(env.static['dispatch_loads'])
    if len(dispatch) != env.graph.nl or len(set(dispatch)) != env.graph.nl or any(n not in env.loads for n in dispatch):
        raise ValueError('All original ordered dispatch load identities are required')
    tau = float(env.event['tau_seconds'])
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError('Positive original event time constant required')
    gaps = {'NATURAL': 1, 'ONE_TAU': int(math.ceil(tau / env.slot_seconds))}
    routes, proposals, candidates, seen = _seed_routes(env), [], [], {}
    for route in (row for row in routes if row['selected']):
        if route['status'] != 'SELECTED':
            continue
        target = set(route['projected_terminal_energized'])
        eligible = [{'action': index + 1, 'load_name': name,
                     'nominal_kw': float(env.loads[name]['nominal_kw'])}
                    for index, name in enumerate(dispatch)
                    if name in target and name not in env.initial_energized
                    and env.initial.command & (1 << index)]
        eligible.sort(key=lambda row: (-row['nominal_kw'], row['action']))
        route['eligible_whole_loads'] = deepcopy(eligible)
        for count in (1, 2):
            if len(eligible) < count:
                proposals.append({'proposal_id': len(proposals), 'route_rank': route['route_rank'],
                    'blocked_count': count, 'status': 'UNAVAILABLE_LOAD_COUNT',
                    'available_load_count': len(eligible), 'actions': None})
                continue
            blocked = eligible[:count]
            variants = [('HOLD', [], 0, None)]
            orders = [('SMALL_FIRST', sorted(blocked, key=lambda x: (x['nominal_kw'], x['action'])))]
            if count == 2:
                orders.append(('LARGE_FIRST', list(blocked)))
            for order_name, order in orders:
                for released_count in range(1, count + 1):
                    for timing in ('NATURAL', 'ONE_TAU'):
                        variants.append((order_name, order[:released_count], released_count, timing))
            for order_name, released, released_count, timing in variants:
                operations = [row['action'] for row in blocked] + route['switch_route']
                gap = None if timing is None else gaps[timing]
                for load in released:
                    operations += [0] * (gap - 1) + [load['action']]
                held = [row['load_name'] for row in blocked if row not in released]
                proposal = {'proposal_id': len(proposals), 'route_rank': route['route_rank'],
                    'seed_index': route['seed_index'], 'origin': ORIGIN,
                    'blocked_count': count, 'blocked_loads': deepcopy(blocked),
                    'restore_order': order_name, 'release_count': released_count,
                    'released_loads': deepcopy(released), 'held_out_loads': held,
                    'terminal_service_class': 'FULL_HORIZON_HELD_OUT' if held else 'ALL_BLOCKED_RELEASED',
                    'timing': timing, 'gap_slots': gap, 'tau_seconds': tau,
                    'switch_route': list(route['switch_route']), 'operation_prefix': operations,
                    'planned_prefix_length': len(operations), 'actions': None,
                    'physical_status': 'UNVERIFIED', 'quality_pct': None}
                proposals.append(proposal)
                if len(operations) > env.horizon:
                    proposal['status'] = 'UNAVAILABLE_HORIZON'
                    continue
                actions = tuple(operations + [0] * (env.horizon - len(operations)))
                proposal.update(actions=list(actions), prefix=_prefix(actions))
                try:
                    info = graph_plan(env, env.initial, actions)
                except ValueError as exc:
                    proposal.update(status='GRAPH_ILLEGAL', error=str(exc))
                    continue
                proposal.update(graph_energy_kwh=info['energy_kwh'], graph_final_kw=info['final_kw'],
                    operations=info['operations'], pickup_batches=_batches(env, info))
                if actions in seen:
                    proposal.update(status='DUPLICATE', duplicate_of=seen[actions])
                    candidates[seen[actions]]['alias_proposal_ids'].append(proposal['proposal_id'])
                    continue
                candidate_id = len(candidates)
                proposal.update(status='LEGAL_UNIQUE', candidate_id=candidate_id)
                seen[actions] = candidate_id
                candidates.append({**deepcopy(proposal), 'alias_proposal_ids': [proposal['proposal_id']]})
    if len(proposals) > 24 or len(candidates) > 24:
        raise AssertionError('Frozen whole-load candidate bound exceeded')
    return {'version': VERSION, 'config': deepcopy(CONFIG), 'event_tau_seconds': tau,
        'gap_slots': gaps, 'routes': routes, 'proposals': proposals, 'candidates': candidates,
        'counts': {'seed_plans': len(routes), 'selected_routes': sum(r['selected'] for r in routes),
            'raw_proposals': len(proposals), 'unique_legal_candidates': len(candidates),
            'proposal_status': {status: sum(r['status'] == status for r in proposals)
                                for status in sorted({r['status'] for r in proposals})}},
        'new_physical_points': 0, 'new_training': 0, 'physical_certificates_created': 0}
