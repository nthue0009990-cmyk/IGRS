# Isolated NC123 dimension adapter. Source and exact replacements: artifacts/nature123_igrs_migration_20261107/derivation.json
"""Deterministic no-training search for a complete known-curve CLPU horizon.

All candidate quality scores are graph-only normal-service integrals. Only a
COMPLETE result from the dynamic environment can become the incumbent. This is
a whole-horizon mechanism pilot, not an implemented online rolling controller.
No physical time passes while candidates are searched. Solver budgets and
within-method dynamic-state caches belong to the supplied oracle.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import heapq
import math

import networkx as nx


@dataclass(frozen=True)
class SearchConfig:
    lookahead_depth: int = 6
    local_prefix_steps: int = 6
    generic_neighbor_limit: int = 32
    feedback_candidate_limit: int = 6
    max_generated_candidates: int = 5000
    max_checked_candidates: int = 256
    feedback_delays: tuple[int, ...] = (1, 2)
    queue_schedule: tuple[str, ...] = ('ordinary', 'feedback')
    switch_action_min: int = 11
    switch_action_max: int = 19

    def __post_init__(self):
        for field in ('lookahead_depth', 'local_prefix_steps', 'generic_neighbor_limit',
                      'feedback_candidate_limit', 'max_generated_candidates', 'max_checked_candidates'):
            if type(getattr(self, field)) is not int or getattr(self, field) <= 0:
                raise ValueError('Positive integer search bound required: ' + field)
        if self.lookahead_depth > 12 or self.feedback_candidate_limit > 6:
            raise ValueError('Pilot lookahead/feedback scope exceeded')
        if tuple(self.queue_schedule) != ('ordinary', 'feedback'):
            raise ValueError('Frozen ordinary/feedback alternation must be 1:1')
        if tuple(self.feedback_delays) != (1, 2):
            raise ValueError('Frozen feedback delays are one and two slots')


def _remaining(env, start):
    return int(env.horizon) - int(start.step)


def graph_plan(env, start, actions):
    """Return graph trajectory and normal-service objective, with no physics."""
    actions = tuple(actions)
    if len(actions) != _remaining(env, start) or any(type(a) is not int or not 0 <= a < env.graph.action_count for a in actions):
        raise ValueError('A complete remaining-horizon action plan is required')
    states = [start]
    for action in actions:
        if action not in env.legal_actions(states[-1]):
            raise ValueError('Graph-illegal candidate action')
        states.append(env.advance(states[-1], action))
    energy = math.fsum(float(env.service_kw(s)) * env.slot_seconds / 3600 for s in states[1:])
    final_kw = float(env.service_kw(states[-1]))
    operations = sum(a != 0 for a in actions)
    return {'actions': actions, 'states': states, 'energy_kwh': energy,
            'final_kw': final_kw, 'operations': operations,
            'priority': (-energy, -final_kw, operations, actions)}


def initial_candidates(env, start, config, *, assess=None):
    """Every legal first action receives the same optimistic graph suffix."""
    remaining = _remaining(env, start)
    if not remaining:
        return [()]
    plans = set()
    for action in sorted(env.legal_actions(start)):
        child = env.advance(start, action)
        suffix = tuple(env.optimistic_plan(child, depth=config.lookahead_depth))
        plan = (int(action),) + suffix
        if len(plan) != remaining:
            raise ValueError('Environment optimistic_plan must pad its full remaining horizon')
        plans.add(plan)
    assessed = [(assess(p) if assess else graph_plan(env, start, p)) for p in sorted(plans)]
    return [p['actions'] for p in sorted((p for p in assessed if p is not None), key=lambda p: p['priority'])]


def _rank_legal(env, start, candidates, *, assess=None):
    ranked = []
    seen = set()
    for plan, kind, details in candidates:
        plan = tuple(plan)
        if plan in seen:
            continue
        seen.add(plan)
        try:
            info = assess(plan) if assess else graph_plan(env, start, plan)
        except ValueError:
            continue
        if info is not None:
            ranked.append((info, kind, details))
    return sorted(ranked, key=lambda item: item[0]['priority'])


def generic_neighbors(env, start, actions, config, *, failed_step=None, assess=None):
    """Common local neighborhoods include WAIT and preparation for all methods.

    A fixed quota prevents replacements from crowding all waiting/swap
    candidates out of the generated set: 16/8/8 at the default bound of 32.
    The first failure location is usable by the ordinary rule too; detailed
    voltage/thermal feedback is not required here.
    """
    info = graph_plan(env, start, actions)
    actions, states = info['actions'], info['states']
    horizon = len(actions)
    positions = set(range(min(config.local_prefix_steps, horizon)))
    if failed_step is not None:
        positions.update(i for i in (failed_step - 1, failed_step, failed_step + 1) if 0 <= i < horizon)
    pools = {'replace': [], 'wait': [], 'swap': []}
    for i in sorted(positions):
        for action in sorted(env.legal_actions(states[i])):
            if action != actions[i]:
                plan = actions[:i] + (int(action),) + actions[i + 1:]
                pools['replace'].append((plan, 'generic_replace', {'position': i, 'action': int(action)}))
        plan = (actions[:i] + (0,) + actions[i:])[:horizon]
        pools['wait'].append((plan, 'generic_wait', {'position': i, 'delay_slots': 1}))
        if i + 1 < horizon and actions[i] != actions[i + 1]:
            plan = actions[:i] + (actions[i + 1], actions[i]) + actions[i + 2:]
            pools['swap'].append((plan, 'generic_adjacent_swap', {'positions': [i, i + 1]}))
    quota = {'replace': config.generic_neighbor_limit // 2,
             'wait': config.generic_neighbor_limit // 4}
    quota['swap'] = config.generic_neighbor_limit - quota['replace'] - quota['wait']
    selected, unused = [], []
    for key in ('replace', 'wait', 'swap'):
        rows = _rank_legal(env, start, pools[key], assess=assess)
        selected.extend(rows[:quota[key]])
        unused.extend(rows[quota[key]:])
    selected.extend(sorted(unused, key=lambda x: x[0]['priority'])[:config.generic_neighbor_limit - len(selected)])
    unique = {}
    for row in selected:
        if row[0]['actions'] != actions:
            unique.setdefault(row[0]['actions'], row)
    return sorted(unique.values(), key=lambda item: item[0]['priority'])[:config.generic_neighbor_limit]


def _new_pickups(before, after):
    return {i for i, (old, new) in enumerate(zip(before.pickups, after.pickups))
            if old is None and new is not None and new >= 0}


def _feedback_exposure(env, state, feedback):
    """Cheap source-path exposure, not a proof of electrical causation."""
    names = getattr(env, 'load_names', None)
    static, loads, scenario = (getattr(env, key, None) for key in ('static', 'loads', 'scenario'))
    if names is None or not all(isinstance(x, Mapping) for x in (static, loads, scenario)):
        return set(), False
    buses = {str(v['bus']).lower() for v in feedback.get('violations', []) if v.get('bus') is not None}
    if not buses:
        return set(), False
    try:
        graph = nx.Graph()
        graph.add_nodes_from(static['nodes'])
        graph.add_edges_from(static['bus_edges'])
        graph.remove_edges_from(static['original_lines'][name] for name in scenario['fault_lines'])
        mask = int(state.command) >> env.graph.nl
        for i, edge in enumerate(static['switch_edges']):
            if not (mask >> i) & 1 and graph.has_edge(*edge):
                graph.remove_edge(*edge)
        source = static['source_bus']
        exposed = set()
        for i, name in enumerate(names):
            bus = loads[name]['bus']
            if nx.has_path(graph, source, bus) and buses.intersection(nx.shortest_path(graph, source, bus)):
                exposed.add(i)
        return exposed, True
    except (KeyError, nx.NetworkXError):
        return set(), False


def feedback_neighbors(env, start, actions, failure, config, *, assess=None):
    """At most six temporal/spatial suggestions; every suggestion needs AC."""
    feedback = failure.get('feedback', failure.get('structured_feedback'))
    if isinstance(feedback, list) and failure.get('status') == 'FAIL':
        # CLPUEnvironment creates FAIL only after checking reliable=True; its
        # compact contract exposes the already qualified list directly.
        feedback = {'usable_for_repair': True, 'violations': feedback}
    if not isinstance(feedback, Mapping) or not feedback.get('usable_for_repair') or not feedback.get('violations'):
        return []
    failed = failure.get('failed_step')
    info = graph_plan(env, start, actions)
    actions, states = info['actions'], info['states']
    if type(failed) is not int or not 0 <= failed < len(actions):
        return []
    batches = {i: _new_pickups(states[i], states[i + 1]) for i in range(len(actions))}
    exposed, available = _feedback_exposure(env, states[failed + 1], feedback)
    prior = [i for i in range(failed + 1) if batches[i]]
    relevant = [i for i in prior if batches[i] & exposed]
    trigger = max(relevant or prior) if prior else failed
    details = {'failed_step': failed, 'trigger_step': trigger,
               'spatial_mapping_available': available, 'spatial_match': bool(relevant),
               'exposed_pickup_indices': sorted(batches[trigger] & exposed),
               'causal_diagnosis_established': False}
    pools = {'delay': [], 'swap': [], 'transfer': []}
    for delay in config.feedback_delays:
        plan = (actions[:trigger] + (0,) * delay + actions[trigger:])[:len(actions)]
        pools['delay'].append((plan, 'feedback_delay', {**details, 'delay_slots': delay}))
    for other in range(trigger + 1, len(actions)):
        if batches[other] and actions[trigger] != actions[other]:
            plan = list(actions)
            plan[trigger], plan[other] = plan[other], plan[trigger]
            pools['swap'].append((plan, 'feedback_pickup_swap', {**details, 'other_step': other}))
        if config.switch_action_min <= actions[other] <= config.switch_action_max:
            plan = actions[:trigger] + (actions[other],) + actions[trigger:other] + actions[other + 1:]
            pools['transfer'].append((plan, 'feedback_transfer_advance', {**details, 'other_step': other}))
    # Two per class before filling unused quota: no outcome-dependent expansion.
    selected, unused = [], []
    for kind in ('delay', 'swap', 'transfer'):
        ranked = _rank_legal(env, start, pools[kind], assess=assess)
        selected.extend(ranked[:2])
        unused.extend(ranked[2:])
    selected.extend(sorted(unused, key=lambda x: x[0]['priority'])[:max(0, config.feedback_candidate_limit - len(selected))])
    unique = {}
    for item in selected:
        if item[0]['actions'] != actions:
            unique.setdefault(item[0]['actions'], item)
    return sorted(unique.values(), key=lambda item: item[0]['priority'])[:config.feedback_candidate_limit]


def run_search(env, oracle, method, config=None, *, start=None, budget_exception=None, record_callback=None):
    """Run one independent method; never inherits another method's cache.

    The caller supplies the environment's budget exception class. Other
    exceptions, especially numerical UNKNOWN/reliability errors, propagate.
    Oracle/environment logging must retain partially evaluated budget attempts.
    """
    config = SearchConfig() if config is None else config
    if method not in ('greedy', 'strong', 'feedback', 'reference'):
        raise ValueError('Unknown fixed search method')
    if budget_exception is None:
        from .nature34_clpu_environment import CLPUBudgetExhausted
        budget_exception = CLPUBudgetExhausted
    start = env.initial if start is None else start
    horizon = _remaining(env, start)
    queues = {'ordinary': [], 'feedback': []}
    queued = {'ordinary': set(), 'feedback': set()}
    generated, checked = set(), set()
    graph_assessments = {}
    attempts, generation_events = [], []
    incumbent, incumbent_key = None, None
    next_queue, stop_reason = 0, 'FRONTIER_EXHAUSTED'
    rejected_by_generation_cap = 0

    def assess(plan):
        nonlocal rejected_by_generation_cap
        plan = tuple(plan)
        if plan not in graph_assessments:
            if len(graph_assessments) >= config.max_generated_candidates:
                rejected_by_generation_cap += 1
                return None
            try:
                info = graph_plan(env, start, plan)
                # Full history is rebuilt only when expanding a parent, so
                # the bounded graph registry does not retain 5000x31 histories.
                graph_assessments[plan] = {k: v for k, v in info.items() if k != 'states'}
            except ValueError:
                graph_assessments[plan] = None
        return graph_assessments[plan]

    def enqueue(info, queue, kind, details):
        nonlocal rejected_by_generation_cap
        plan = info['actions']
        if plan in checked or plan in queued[queue]:
            return
        generated.add(plan)
        queued[queue].add(plan)
        heapq.heappush(queues[queue], (info['priority'], plan, kind, details))

    def evaluate(plan, origin, details):
        nonlocal incumbent, incumbent_key
        checked.add(plan)
        try:
            result = env.validate_plan(start, list(plan), oracle)
        except budget_exception:
            attempt = {'actions': list(plan), 'origin': origin, 'details': details,
                       'status': 'BUDGET_EXHAUSTED_PARTIAL', 'certified_complete': False}
            attempts.append(attempt)
            if record_callback is not None:
                record_callback(attempt)
            raise
        record = {'actions': list(plan), 'origin': origin, 'details': details,
                  'status': result['status'], 'certified_complete': result['status'] == 'COMPLETE',
                  'validation': result}
        attempts.append(record)
        if record_callback is not None:
            record_callback(record)
        if result['status'] not in ('COMPLETE', 'FAIL', 'GRAPH_FAIL'):
            raise RuntimeError('Unexpected nonterminal/reliability validation status: ' + str(result['status']))
        if result['status'] == 'COMPLETE':
            energy, fraction = float(result['energy_kwh']), float(result['final_fraction'])
            if not math.isfinite(energy) or not math.isfinite(fraction):
                raise ValueError('Nonfinite complete-plan objective')
            key = (-energy, -fraction, sum(a != 0 for a in plan), plan)
            if incumbent_key is None or key < incumbent_key:
                incumbent, incumbent_key = record, key
        return result

    wait = tuple(0 for _ in range(horizon))
    assess(wait)
    generated.add(wait)
    try:
        evaluate(wait, 'wait_incumbent', {})
        for plan in initial_candidates(env, start, config, assess=assess):
            enqueue(assess(plan), 'ordinary', 'all_first_actions_graph_suffix', {})
        while len(checked) < config.max_checked_candidates:
            # Already-checked plans can exist in both queues: discard without
            # consuming oracle budget or changing the alternating schedule.
            for heap in queues.values():
                while heap and heap[0][1] in checked:
                    heapq.heappop(heap)
            if not any(queues.values()):
                break
            desired = config.queue_schedule[next_queue] if method in ('feedback', 'reference') else 'ordinary'
            queue = desired if queues[desired] else ('feedback' if desired == 'ordinary' else 'ordinary')
            frontier_counts = {k: len(v) for k, v in queues.items()}
            _, plan, origin, details = heapq.heappop(queues[queue])
            if method in ('feedback', 'reference'):
                next_queue = 1 - config.queue_schedule.index(queue)
            result = evaluate(plan, origin, {**details, 'queue': queue, 'desired_queue': desired,
                                            'frontier_counts_before_pop': frontier_counts})
            if method == 'greedy':
                continue
            normal = generic_neighbors(env, start, plan, config, failed_step=result.get('failed_step'), assess=assess)
            repaired = feedback_neighbors(env, start, plan, result, config, assess=assess) if method in ('feedback', 'reference') and result['status'] == 'FAIL' else []
            generation_events.append({'parent_actions': list(plan), 'parent_status': result['status'],
                                      'ordinary_generated': len(normal), 'feedback_generated': len(repaired)})
            for info, kind, detail in normal:
                enqueue(info, 'ordinary', kind, detail)
            for info, kind, detail in repaired:
                enqueue(info, 'feedback', kind, detail)
        else:
            stop_reason = 'LOGICAL_CHECK_CAP'
    except budget_exception:
        stop_reason = 'PHYSICAL_BUDGET_EXHAUSTED'
    return {'method': method, 'config': asdict(config), 'stop_reason': stop_reason,
            'status': 'CERTIFIED_INCUMBENT' if incumbent is not None else 'NO_CERTIFIED_COMPLETE_PLAN',
            'incumbent': incumbent, 'attempts': attempts, 'generation_events': generation_events,
            'generated_unique_candidates': len(generated), 'checked_candidates': len(checked),
            'assessed_graph_candidates': len(graph_assessments),
            'generation_cap_rejections': rejected_by_generation_cap,
            'scope': 'Known-curve complete-horizon no-training path search',
            'optimality_established': False, 'new_training_steps': 0}
