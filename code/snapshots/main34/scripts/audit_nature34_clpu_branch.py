"""Independent D059 replay. No production branch/environment/learner is used.

The old independent physical equations, complete-load graph and integer DP
auditors are reused. New branch generation, visibility, observations, requests,
incumbents and stopping are rebuilt from paid chronological evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
sys.dont_write_bytecode = True

import networkx as nx
import audit_nature34_clpu_pilot as old
from audit_nature34_joint_planning import IndependentGraph, IndependentDP

require, close, read, save, sha, canonical, read_lines = (
    old.require, old.close, old.read, old.save, old.sha, old.canonical, old.read_lines)
_graphs = OrderedDict()


def same(actual, expected, label):
    """Exact structure and integer identity, with original scalar tolerance."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), label + ' keys')
        for key in expected:
            same(actual[key], expected[key], label + '.' + str(key))
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), label + ' size')
        for i, (a, b) in enumerate(zip(actual, expected)):
            same(a, b, label + '[' + str(i) + ']')
    elif type(expected) is float:
        close(actual, expected, label)
    else:
        require(type(actual) is type(expected) and actual == expected, label + ' value')


class Mechanics:
    def __init__(self, static, loads, scenario, condition, event, cfg, graph=None, dp=None):
        self.static, self.loads, self.scenario = static, loads, scenario
        self.condition, self.event = condition, event
        self.horizon, self.slot, self.sample = (cfg[k] for k in ('horizon', 'slot_seconds', 'sample_seconds'))
        self.names = sorted(loads)
        self.multiplier = float(condition['load_multiplier'])
        self.total_kw = sum(float(v['nominal_kw']) for v in loads.values()) * self.multiplier
        self.initial = old.independent_initial(static, loads, scenario)
        if graph is None:
            key = hashlib.sha256(canonical([static, loads, scenario]).encode()).hexdigest()
            if key not in _graphs:
                independent = IndependentGraph(static, loads, scenario)
                _graphs[key] = (independent, IndependentDP(independent, [], max_horizon=cfg['branch_search']['lookahead_depth']))
                if len(_graphs) > 2:
                    _graphs.popitem(last=False)
            graph, dp = _graphs[key]
            _graphs.move_to_end(key)
        self.graph, self.dp = graph, dp
        self.max_kw = max(int(w) for w in graph.weights) / graph.scale * self.multiplier
        self.denominator = self.total_kw * self.horizon * self.slot / 3600.
        self.bound = math.fsum(self.max_kw * self.slot / 3600. for _ in range(self.horizon))
        self._legal = {}

    def legal(self, state):
        if state['step'] >= self.horizon:
            return []
        command = state['command']
        if command not in self._legal:
            position = self.graph.locate(command)
            targets = set(int(self.graph.states[i]) for i in self.graph.next[position] if i >= 0)
            self._legal[command] = [0] + [i for i in range(1, 20) if command ^ (1 << (i - 1)) in targets]
        return self._legal[command]

    def advance(self, state, action):
        require(type(action) is int and action in self.legal(state), 'Graph-illegal expansion action')
        return old.independent_advance(state, action, self.static, self.loads, self.scenario, self.slot)

    def service(self, state):
        return int(self.graph.weights[self.graph.locate(state['command'])]) / self.graph.scale * self.multiplier

    def plan(self, actions):
        require(len(actions) == self.horizon, 'Branch candidate is not full-window')
        states = [self.initial]
        for action in actions:
            states.append(self.advance(states[-1], action))
        energy = math.fsum(self.service(s) * self.slot / 3600. for s in states[1:])
        final = self.service(states[-1])
        operations = sum(a != 0 for a in actions)
        return {'actions': tuple(actions), 'states': states, 'energy_kwh': energy, 'final_kw': final,
                'operations': operations, 'priority': (-energy, -final, operations, tuple(actions))}

    def suffix(self, state, depth):
        remaining = self.horizon - state['step']
        path = self.dp.route(state['command'], min(depth, remaining, 12))['path']
        actions = [(a ^ b).bit_length() for a, b in zip(path, path[1:])]
        return tuple(actions + [0] * (remaining - len(actions)))

    def identity(self, state, seconds, condition_sha):
        return {'model_version': 'nc34-clpu-ed-snapshot-v1', 'scenario_id': self.scenario['scenario_id'],
                'condition_id': self.condition['condition_id'], 'event_id': self.event['event_id'],
                'a': self.event['a'], 'tau_seconds': self.event['tau_seconds'], 'command': state['command'],
                'pickups': state['pickups'], 'load_names': self.names, 'time_seconds': float(seconds),
                'multipliers': old.independent_multipliers(state, seconds, self.loads, self.event),
                'alpha': .5, 'base_condition_sha256': condition_sha}

    def exposure(self, state, feedback):
        buses = {str(v['bus']).lower() for v in feedback if v.get('bus') is not None}
        if not buses:
            return set(), False
        graph = old.independent_projection(self.static, self.loads, self.scenario, state['command'])['graph']
        source = self.static['source_bus']
        return {i for i, name in enumerate(self.names)
                if nx.has_path(graph, source, self.loads[name]['bus'])
                and buses.intersection(nx.shortest_path(graph, source, self.loads[name]['bus']))}, True


class PaidRequests:
    """Consume each saved logical request in its actual chronological place."""
    def __init__(self, rows, points, mech, budget, condition_sha):
        self.rows, self.points, self.mech = list(rows), points, mech
        self.budget, self.condition_sha = budget, condition_sha
        self.index, self.requests, self.used = 0, 0, 0
        self.cache = {}

    def certify(self, state, seconds):
        identity = self.mech.identity(state, seconds, self.condition_sha)
        key = hashlib.sha256(canonical(identity).encode()).hexdigest()
        require(self.index < len(self.rows), 'Missing physical request')
        self.requests += 1
        row = self.rows[self.index]
        self.index += 1
        require(row['request'] == self.requests and row['key'] == key, 'Physical query order/history changed')
        if key in self.cache:
            require(row == {'request': self.requests, 'key': key, 'cache_hit': True,
                            'point_id': self.cache[key]}, 'Cross-method, future or false cache hit')
            point_id = self.cache[key]
        elif self.used == self.budget:
            require(row == {'request': self.requests, 'key': key, 'cache_hit': False, 'status': 'BUDGET_UNKNOWN'},
                    'Missing exact budget denial')
            return None
        else:
            point_id = self.used
            expected = {'request': self.requests, 'key': key, 'cache_hit': False,
                        'point_id': point_id, 'status': 'STARTED'}
            require(row == expected and self.index < len(self.rows), 'Unpaid or duplicate query')
            point = self.points[point_id]
            expected['status'] = point['status']
            require(self.rows[self.index] == expected, 'Unclosed query or wrong physical outcome')
            self.index += 1
            self.used += 1
            self.cache[key] = point_id
            require(point['state_identity'] == identity, 'Query does not match raw point identity')
        point = self.points[point_id]
        return {'point_id': point_id, 'status': point['status'], 'reliable': point['reliable'],
                'feedback': point['structured_feedback']['violations']}

    def counts(self):
        return {'requests': self.requests, 'points': self.used}


class BranchReplay:
    """Independent queue reconstruction, using no production branch functions."""
    def __init__(self, mech, cfg, oracle, attempts):
        self.m, self.cfg, self.oracle = mech, cfg, oracle
        self.attempts, self.attempt_index = attempts, 0
        self.assessed, self.checked, self.frontier = {}, {}, {}
        self.keys, self.expanded = set(), set()
        self.generated, self.assessment_trace, self.trace, self.generation_events = [], [], [], []
        self.rejections, self.turn = 0, 0
        self.best = None
        self.latest = None
        self.failures = []
        self.done, self.reason = False, None
        self.initial_certificate = oracle.certify(mech.initial, 0.)
        require(self.initial_certificate is not None, 'Initial certificate budget missing')
        if self.initial_certificate['status'] == 'FAIL':
            self.stop('INITIAL_FAIL')
        else:
            self.evaluate(self.assess((0,) * mech.horizon), 'wait_incumbent', {'bootstrap': True})
            if not self.done:
                self.expand((), None)
                self.refresh()

    def stop(self, reason):
        self.done, self.reason = True, reason

    @property
    def quality(self):
        return self.best['validation']['energy_kwh'] / self.m.denominator if self.best else 0.

    def assess(self, actions):
        actions = tuple(actions)
        if actions in self.assessed:
            return self.assessed[actions]
        if len(self.assessed) >= self.cfg['max_generated_candidates']:
            self.rejections += 1
            return None
        try:
            result = self.m.plan(actions)
        except (AssertionError, ValueError, nx.NetworkXException):
            result = None
        self.assessed[actions] = result
        self.assessment_trace.append({'actions': list(actions), 'graph_valid': result is not None})
        return result

    def add(self, prefix, actions, origin, parent):
        prefix, actions = tuple(prefix), tuple(actions)
        if (prefix, actions) in self.keys:
            return
        if len(self.keys) >= self.cfg['max_generated_candidates']:
            self.rejections += 1
            return
        info = self.assess(actions)
        if info is None:
            return
        require(actions[:len(prefix)] == prefix, 'Branch prefix changed')
        index = len(self.keys)
        self.keys.add((prefix, actions))
        self.frontier[index] = {'id': index, 'prefix': prefix, 'info': info, 'origin': origin, 'parent_id': parent}
        self.generated.append({'branch_id': index, 'prefix': list(prefix), 'actions': list(actions),
                               'origin': origin, 'parent_id': parent, 'upper_bound_kwh': self.m.bound,
                               'bound_scope': 'maximum_initial_reachable_graph_service_in_every_slot',
                               'graph_energy_kwh': info['energy_kwh'], 'graph_final_kw': info['final_kw'],
                               'operations': info['operations']})

    def expand(self, prefix, parent):
        if prefix in self.expanded:
            return
        self.expanded.add(prefix)
        if len(prefix) >= min(self.cfg['lookahead_depth'], self.m.horizon):
            return
        state = self.m.initial
        for action in prefix:
            state = self.m.advance(state, action)
        for action in self.m.legal(state):
            child_prefix = prefix + (action,)
            child = self.m.advance(state, action)
            self.add(child_prefix, child_prefix + (0,) * (self.m.horizon - len(child_prefix)), 'prefix_wait', parent)
            self.add(child_prefix, child_prefix + self.m.suffix(child, self.cfg['lookahead_depth']),
                     'prefix_graph_completion', parent)

    def neighbors(self, plan, failed):
        info = self.m.plan(plan)
        positions = set(range(min(self.cfg['lookahead_depth'], self.m.horizon)))
        if failed is not None:
            positions.update(i for i in (failed - 1, failed, failed + 1) if 0 <= i < self.m.horizon)
        pools = [[], [], []]
        for i in sorted(positions):
            pools[0] += [(plan[:i] + (a,) + plan[i + 1:], 'generic_replace')
                         for a in self.m.legal(info['states'][i]) if a != plan[i]]
            pools[1].append(((plan[:i] + (0,) + plan[i:])[:self.m.horizon], 'generic_wait'))
            if i + 1 < self.m.horizon and plan[i] != plan[i + 1]:
                pools[2].append((plan[:i] + (plan[i + 1], plan[i]) + plan[i + 2:], 'generic_adjacent_swap'))
        limit = self.cfg['generic_neighbor_limit']
        quotas = [limit // 2, limit // 4, limit - limit // 2 - limit // 4]
        selected, extra = [], []
        for pool, quota in zip(pools, quotas):
            seen, ranked = set(), []
            for actions, kind in pool:
                if actions in seen:
                    continue
                seen.add(actions)
                candidate = self.assess(actions)
                if candidate is not None:
                    ranked.append((candidate, kind))
            ranked.sort(key=lambda x: x[0]['priority'])
            selected += ranked[:quota]
            extra += ranked[quota:]
        selected += sorted(extra, key=lambda x: x[0]['priority'])[:limit - len(selected)]
        unique = {}
        for item in selected:
            if item[0]['actions'] != plan:
                unique.setdefault(item[0]['actions'], item)
        return sorted(unique.values(), key=lambda x: x[0]['priority'])[:limit]

    def evaluate(self, info, origin, details):
        actions = info['actions']
        if actions in self.checked:
            return self.checked[actions]
        if len(self.checked) >= self.cfg['max_checked_candidates']:
            self.stop('LOGICAL_CHECK_CAP')
            return None
        require(self.attempt_index < len(self.attempts), 'Missing paid candidate validation')
        record = self.attempts[self.attempt_index]
        self.attempt_index += 1
        before = self.oracle.used
        require(record['actions'] == list(actions) and record['origin'] == origin and record['details'] == details,
                'Candidate attempt identity/order changed')
        expected_samples, ended = [], None
        for state in info['states'][1:]:
            for offset in range(0, self.m.slot + 1, self.m.sample):
                seconds = (state['step'] - 1) * self.m.slot + offset
                cert = self.oracle.certify(state, seconds)
                if cert is None:
                    ended = 'BUDGET_EXHAUSTED_PARTIAL'
                    break
                expected_samples.append({'step': state['step'], 'time_seconds': seconds,
                                         'point_id': cert['point_id'], 'status': cert['status'], 'state': state})
                if cert['status'] == 'FAIL':
                    ended = 'FAIL'
                    break
            if ended:
                break
        ended = ended or 'COMPLETE'
        require(record['status'] == ended and record['certified_complete'] == (ended == 'COMPLETE'),
                'Unfinished/failed path represented as complete')
        require(record['physical_points_before'] == before and record['physical_points_after'] == self.oracle.used,
                'Attempt charged points differ')
        if ended == 'BUDGET_EXHAUSTED_PARTIAL':
            require('validation' not in record, 'Budget partial fabricated complete evidence')
            self.stop('PHYSICAL_BUDGET_EXHAUSTED')
        else:
            validation = record['validation']
            require(validation['samples'] == expected_samples, 'Missing peak, tail or query-order evidence')
            old.check_saved_plan(validation, self.m.initial, self.m.static, self.m.loads, self.m.scenario,
                                 self.m.condition, self.m.event, self.m.horizon, self.m.slot, self.m.sample,
                                 self.oracle.points)
            if ended == 'COMPLETE':
                if self.best is None or info['priority'] < self.m.plan(self.best['actions'])['priority']:
                    self.best = record
            else:
                self.latest = validation
                self.failures.append(validation)
        self.checked[actions] = record
        return record

    def window(self):
        all_rows = list(self.frontier.values())
        ranked = sorted(all_rows, key=lambda b: (b['info']['priority'], b['id']))
        size = self.cfg['action_window']
        selected, seen = [], set()
        for pool, quota in ((ranked, max(1, size // 2)),
                            ([b for b in ranked if b['origin'] == 'prefix_wait'], max(1, size // 4)),
                            (sorted(all_rows, key=lambda b: b['id']), size)):
            added = 0
            for row in pool:
                if row['id'] not in seen and added < quota and len(selected) < size:
                    selected.append(row)
                    seen.add(row['id'])
                    added += 1
        return selected

    def refresh(self):
        if self.done:
            return
        if self.best and self.frontier and self.m.bound <= self.best['validation']['energy_kwh']:
            self.stop('CERTIFIED_FRONTIER_UPPER_BOUND')
        elif self.oracle.used >= self.cfg['point_budget']:
            self.stop('PHYSICAL_BUDGET_EXHAUSTED')
        elif len(self.checked) >= self.cfg['max_checked_candidates']:
            self.stop('LOGICAL_CHECK_CAP')
        elif self.turn >= self.cfg['max_branch_expansions']:
            self.stop('BRANCH_EXPANSION_CAP')
        elif not self.frontier:
            self.stop('BOUNDED_FRONTIER_EXHAUSTED')

    def rule(self, method):
        window = self.window()
        if method == 'greedy':
            return min(range(len(window)), key=lambda i: (window[i]['info']['priority'], window[i]['id']))
        require(method == 'beam', 'Unknown rule')
        depth = min(len(b['prefix']) for b in window)
        ids = sorted((i for i, b in enumerate(window) if len(b['prefix']) == depth),
                     key=lambda i: (window[i]['info']['priority'], window[i]['id']))
        by_prefix = {}
        for i in ids:
            by_prefix.setdefault(window[i]['prefix'], i)
        beam = list(by_prefix.values())[:self.cfg['beam_width']]
        observation = self.observation('associated')
        column = observation['candidate_feature_names'].index('cold_excess_exposed_weight')
        return min(beam, key=lambda i: (observation['candidate_features'][i][column],
                                       window[i]['info']['priority'], window[i]['id']))

    def step(self, index):
        require(not self.done and type(index) is int, 'Decision after termination')
        window = self.window()
        require(0 <= index < len(window), 'Selected unexposed branch')
        node = window[index]
        del self.frontier[node['id']]
        before, quality, generated = self.oracle.used, self.quality, len(self.keys)
        info, plan = node['info'], node['info']['actions']
        details = {'branch_id': node['id'], 'parent_id': node['parent_id'], 'prefix': list(node['prefix']),
                   'window_branch_ids': [b['id'] for b in window], 'selected_index': index,
                   'frontier_size_before': len(self.frontier) + 1}
        record = self.checked.get(plan)
        why = 'previously_checked' if record else 'query'
        if record is None and self.best and info['priority'] >= self.m.plan(self.best['actions'])['priority']:
            why = 'dominated_known_complete_objective'
        elif record is None:
            record = self.evaluate(info, node['origin'], details)
        if not self.done:
            validation = record.get('validation', {}) if record else {}
            failed = validation.get('failed_step') if validation.get('status') == 'FAIL' else None
            if not (isinstance(failed, int) and failed < len(node['prefix'])):
                self.expand(node['prefix'], node['id'])
            for length in range(1, min(self.cfg['lookahead_depth'], self.m.horizon) + 1):
                self.add(plan[:length], plan[:length] + (0,) * (self.m.horizon - length), 'prefix_wait', node['id'])
            for neighbor, kind in self.neighbors(plan, failed):
                self.add(neighbor['actions'][:min(self.cfg['lookahead_depth'], self.m.horizon)],
                         neighbor['actions'], kind, node['id'])
        self.turn += 1
        self.refresh()
        event = {**details, 'origin': node['origin'], 'actions': list(plan), 'query_reason': why,
                 'validation_status': record['status'] if record else None,
                 'quality_before': quality, 'quality_after': self.quality, 'physical_points_before': before,
                 'physical_points_after': self.oracle.used, 'physical_points_used': self.oracle.used - before,
                 'new_branches': len(self.keys) - generated, 'frontier_size_after': len(self.frontier),
                 'done': self.done, 'stop_reason': self.reason}
        event['reward'] = self.quality - quality - self.cfg['reward_cost_weight'] * (self.oracle.used - before) / self.cfg['reward_cost_normalizer']
        event['info'] = {key: event[key] for key in ('validation_status', 'physical_points_used', 'query_reason', 'stop_reason')}
        self.trace.append(event)
        self.generation_events.append({'branch_id': node['id'], 'new_branches': event['new_branches'],
                                       'assessed_graph_candidates': len(self.assessed)})
        return event

    def observation(self, representation):
        require(representation in ('raw', 'associated'), 'Representation differs')
        window = [] if self.done else self.window()
        m, names = self.m, self.m.names
        weights = [float(m.loads[n]['nominal_kw']) * m.multiplier / m.total_kw for n in names]
        failure = self.latest or {}
        feedback = failure.get('feedback', [])
        sample_index = min(max(int(failure.get('failed_step') or 0) + 1, 1), m.horizon)
        seconds = float(failure.get('failed_time_seconds', (sample_index - 1) * m.slot))
        duration = m.horizon * m.slot
        magnitude = max([0.] + [abs(float(v.get('magnitude', 0.))) for v in feedback])
        a, tau = float(m.event['a']), float(m.event['tau_seconds'])
        global_values = [self.quality, self.oracle.used / self.cfg['point_budget'], len(self.checked) / self.cfg['max_checked_candidates'],
                         len(self.assessed) / self.cfg['max_generated_candidates'], float(bool(feedback)), seconds / duration,
                         magnitude / (1. + magnitude), a / (1. + abs(a)), tau / (tau + duration),
                         sum('voltage' in str(v.get('constraint', '')) for v in feedback) / max(1, len(feedback)),
                         sum('thermal' in str(v.get('constraint', '')) for v in feedback) / max(1, len(feedback))] + weights
        global_names = ['incumbent_fraction', 'point_budget_used', 'queries_used', 'graph_cap_used',
                        'has_observed_failure', 'failure_time_fraction', 'failure_magnitude_bounded',
                        'event_a_bounded', 'event_tau_bounded', 'voltage_feedback_fraction', 'thermal_feedback_fraction']
        global_names += ['normal_weight:' + n for n in names]
        feature_names = ['known_fraction', 'known_final_fraction', 'operations_fraction', 'prefix_depth_fraction',
                         'upper_bound_fraction', 'is_prefix_wait', 'already_checked', 'source_path_mapping_available']
        feature_names += ['action:' + str(i) for i in range(m.horizon)]
        for label in ('pickup_time:', 'feedback_time_pickup:', 'feedback_path_exposure:'):
            feature_names += [label + n for n in names]
        feature_names += ['command_bit:' + str(i) for i in range(19)]
        if representation == 'associated':
            feature_names += ['exposed_normal_weight', 'cold_exposed_weight', 'new_batch_exposed_weight',
                              'cold_excess_exposed_weight', 'failure_age_weighted_exposure', 'exposed_unpicked_weight',
                              'pickup_batch_count_fraction', 'largest_pickup_batch_weight']
        matrix, rows = [], []
        norm = lambda p: -2. if p is None else -1. if p == -1 else float(p) / duration
        for node in window:
            info = node['info']
            state, final = info['states'][sample_index], info['states'][-1]
            exposed, available = m.exposure(state, feedback) if feedback else (set(), False)
            vector = [info['energy_kwh'] / m.denominator, info['final_kw'] / m.total_kw,
                      info['operations'] / m.horizon, len(node['prefix']) / m.horizon,
                      m.bound / m.denominator, float(node['origin'] == 'prefix_wait'),
                      float(info['actions'] in self.checked), float(available)]
            vector += [x / 19. for x in info['actions']]
            vector += [norm(p) for p in final['pickups']] + [norm(p) for p in state['pickups']]
            vector += [float(i in exposed) for i in range(len(names))]
            vector += [float((state['command'] >> i) & 1) for i in range(19)]
            if representation == 'associated':
                cold = {i for i, p in enumerate(state['pickups']) if p is not None and p >= 0}
                new = {i for i in cold if state['pickups'][i] == (sample_index - 1) * m.slot}
                groups = {}
                for i, pickup in enumerate(final['pickups']):
                    if pickup is not None and pickup >= 0:
                        groups[pickup] = groups.get(pickup, 0.) + weights[i]
                ages = {i: max(0., seconds - state['pickups'][i]) for i in cold}
                vector += [sum(weights[i] for i in exposed), sum(weights[i] for i in exposed & cold),
                           sum(weights[i] for i in exposed & new),
                           sum(weights[i] * a * math.exp(-ages[i] / tau) for i in exposed & cold),
                           sum(weights[i] * ages[i] / duration for i in exposed & cold),
                           sum(weights[i] for i in exposed if state['pickups'][i] is None),
                           len(groups) / m.horizon, max(groups.values(), default=0.)]
            matrix.append(vector)
            rows.append({'branch_id': node['id'], 'origin': node['origin'], 'prefix': list(node['prefix']),
                         'actions': list(info['actions']), 'graph_energy_kwh': info['energy_kwh'],
                         'upper_bound_kwh': m.bound})
        return {'representation': representation, 'candidate_features': matrix, 'global_features': global_values,
                'branch_ids': [b['id'] for b in window], 'action_mask': [True] * len(window), 'mask': [True] * len(window),
                'candidate_feature_names': feature_names, 'global_feature_names': global_names, 'branches': rows,
                'observed_feedback': feedback, 'done': self.done, 'observed_failure_history': self.failures,
                'raw_information_identical': True}


def check_branch_records(mech, branch_cfg, result, transitions, decisions, request_rows, points, condition_sha):
    oracle = PaidRequests(request_rows, points, mech, branch_cfg['point_budget'], condition_sha)
    replay = BranchReplay(mech, branch_cfg, oracle, result['attempts'])
    require(result['initial_certificate'] == replay.initial_certificate, 'Initial certificate changed')
    # Consume the journal in lockstep; retain every full observation/history
    # check while keeping only the current transition and journal row alive.
    decision_iterator, end = iter(decisions), object()
    transition_count = 0
    for index, transition in enumerate(transitions):
        transition_count += 1
        require(not replay.done and transition['transition_id'] == index, 'Transition chronology differs')
        for representation in ('raw', 'associated'):
            same(transition['obs_' + representation], replay.observation(representation),
                 'observation_' + representation)
        require(transition['representation'] == result['spec']['representation'], 'Policy representation differs')
        require(transition['obs'] == transition['obs_' + transition['representation']], 'Policy received hidden input')
        require(transition['mask'] == [True] * len(replay.window()), 'Action mask differs')
        require(transition['oracle_before'] == oracle.counts(), 'Unrecorded query before decision')
        selected = transition['action']
        if result['spec']['method'] in ('greedy', 'beam'):
            require(selected == replay.rule(result['spec']['method']), 'Strong rule order changed')
            require(transition['logprob'] is None and transition['value'] is None, 'Rule fabricated learned statistics')
        require(transition['branch_id'] == replay.window()[selected]['id'], 'Policy branch/index mismatch')
        started = {k: v for k, v in transition.items() if k not in ('reward', 'done', 'info', 'oracle_after')}
        started_record = next(decision_iterator, end)
        require(started_record is not end, 'Unclosed/omitted policy decision')
        require(started_record == {**started, 'status': 'STARTED'}, 'Started decision differs from transition')
        del started_record
        event = replay.step(selected)
        same(transition['info'], {k: v for k, v in event.items() if k not in ('reward', 'done')}, 'step_info')
        close(transition['reward'], event['reward'], 'Reward contains unobserved outcome or wrong cost')
        require(transition['done'] == event['done'] and transition['oracle_after'] == oracle.counts(),
                'Terminal flag or after-query count differs')
        completed = {'transition_id': index, 'status': 'COMPLETED', 'action': selected,
                     'branch_id': transition['branch_id'], 'reward': transition['reward'], 'done': transition['done'],
                     'info': transition['info'], 'oracle_before': transition['oracle_before'], 'oracle_after': transition['oracle_after']}
        completed_record = next(decision_iterator, end)
        require(completed_record is not end, 'Unclosed/omitted policy decision')
        require(completed_record == completed, 'Completed journal differs from transition')
        del completed_record
    require(next(decision_iterator, end) is end, 'Unclosed/omitted policy decision')
    require(replay.done and result['done'] is True and result['stop_reason'] == replay.reason,
            'Unfinished search or invalid upper-bound stop')
    require(replay.attempt_index == len(result['attempts']), 'Unqueried/omitted attempts')
    require(oracle.index == len(oracle.rows) and oracle.used == len(points), 'Unbound paid physical queries')
    require(result['best_plan'] == result['incumbent'] == replay.best, 'Returned uncertified or inferior plan')
    expected_status = ('INITIAL_FAIL' if replay.reason == 'INITIAL_FAIL' else
                       'CERTIFIED_INCUMBENT' if replay.best else 'NO_CERTIFIED_COMPLETE_PLAN')
    require(result['status'] == expected_status, 'Search status misclassified')
    for name, value in [('generation_trace', replay.generated), ('graph_assessment_trace', replay.assessment_trace),
                        ('trace', replay.trace), ('generation_events', replay.generation_events)]:
        same(result[name], value, name)
    for name, value in [('generated_unique_candidates', len(replay.keys)), ('assessed_graph_candidates', len(replay.assessed)),
                        ('checked_candidates', len(replay.checked)), ('frontier_size', len(replay.frontier)),
                        ('generation_cap_rejections', replay.rejections), ('actual_physical_points', oracle.used),
                        ('search_decisions', transition_count)]:
        require(result[name] == value, name + ' differs')
    close(result['maximum_reachable_graph_kw'], mech.max_kw, 'Maximum graph load')
    close(result['global_upper_bound_kwh'], mech.bound, 'Wrong or shallow-completion upper bound')
    require(result['optimality_established'] == (replay.reason == 'CERTIFIED_FRONTIER_UPPER_BOUND'),
            'Unsupported global-bound proof')
    return {'decisions': transition_count, 'generated': len(replay.keys), 'assessed': len(replay.assessed),
            'stop_reason': replay.reason, 'initial_fail': expected_status == 'INITIAL_FAIL'}


def check_raw_points(run, cell, spec, cfg, mech):
    metadata, physical_config = read(cell / 'metadata.json'), read(run / 'physical_config.json')
    require(metadata == read(run / 'source_inputs/metadata.json'), 'Inherited device metadata changed')
    reference = read(run / 'operating_reference.json')
    points, counts = {}, Counter()
    for row in read_lines(cell / 'points.jsonl.gz'):
        point_id = row['point_id']
        require(point_id == len(points) and point_id < spec['point_budget'], 'Raw point order/budget differs')
        state = row['dynamic_state']
        require(type(state['step']) is int and 0 <= state['step'] <= mech.horizon
                and len(state['pickups']) == len(mech.names) == 68, 'Raw dynamic dimensions differ')
        projection = old.independent_projection(mech.static, mech.loads, mech.scenario, state['command'])
        for name, pickup, original in zip(mech.names, state['pickups'], mech.initial['pickups']):
            require((pickup == -1) == (original == -1) and (pickup is not None) == (name in projection['energized']),
                    'Forged steady/unserved history')
            if pickup is not None and pickup != -1:
                require(type(pickup) in (int, float) and math.isfinite(pickup) and pickup >= 0 and state['step'] > 0
                        and pickup <= (state['step'] - 1) * mech.slot and pickup % mech.slot == 0,
                        'Off-grid/future pickup')
        seconds = row['state_identity']['time_seconds']
        require(seconds % mech.sample == 0 and 0 <= seconds <= mech.horizon * mech.slot, 'Off-grid physical time')
        identity = mech.identity(state, seconds, sha(run / 'condition.json'))
        require(row['state_identity'] == identity and row['identity_sha256'] == hashlib.sha256(canonical(identity).encode()).hexdigest(),
                'Raw cache identity differs')
        require(row['case'] == spec['scenario_id'] and row['fault_lines'] == mech.scenario['fault_lines']
                and row['alpha'] == .5 and row['algorithm'] == 0, 'Raw physical context changed')
        expected_enabled = {}
        for name, layout in metadata['element_layout'].items():
            buses = [b.split('.')[0].lower() for b in layout['buses']]
            enabled = all(b in projection['powered'] for b in buses)
            if layout['kind'] in ('line', 'transformer'):
                endpoints = list(dict.fromkeys(buses))
                enabled = enabled and len(endpoints) == 2 and projection['graph'].has_edge(*endpoints)
            if layout['kind'] == 'load':
                enabled = name in projection['energized']
            expected_enabled[name] = bool(enabled)
        require(row['metadata_override']['planned_enabled'] == expected_enabled, 'Device isolation differs')
        checked = old.check_saved_point(row, mech.condition, reference, metadata, physical_config,
                                       identity['multipliers'], projection['action'], seconds / 60., projection['energized'])
        require(checked['status'] != 'UNKNOWN', 'Unreliable point in completed experiment')
        counts[checked['status']] += 1
        points[point_id] = {key: row[key] for key in ('status', 'reliable', 'clpu_multipliers', 'action', 'time_minutes',
                            'state_identity', 'identity_sha256', 'actual_call_counts', 'dynamic_state', 'elapsed_seconds', 'structured_feedback')}
    require(points and len({p['identity_sha256'] for p in points.values()}) == len(points), 'Empty/duplicate point population')
    return points, counts


def check_cell(run, cell, cfg):
    cell = Path(cell).resolve()
    require(cell.is_relative_to(run) and cell != run, 'Cell outside declared run')
    require(not (cell / 'error.json').exists(), 'Cell reliability error retained')
    manifest = read(cell / 'completion_manifest.json')
    for name, digest in manifest.items():
        require(sha(cell / name) == digest, 'Cell artifact changed: ' + name)
    required = {'spec.json', 'result.json', 'points.jsonl.gz', 'plans.jsonl.gz', 'transitions.jsonl.gz',
                'decisions.jsonl.gz', 'requests.jsonl', 'command_ledger.jsonl', 'initial_state.json',
                'metadata.json', 'oracle_summary.json', 'command_stats.json'}
    require(required <= manifest.keys(), 'Cell has unsealed required evidence')
    spec, result = read(cell / 'spec.json'), read(cell / 'result.json')
    schedule = read(run / 'schedule.json')
    templates = {row['cell']: row for row in schedule}
    relative = cell.relative_to(run).as_posix()
    require(relative in templates, 'Unscheduled cell')
    template = templates[relative]
    extras = {'template_sha256', 'deadline_utc_epoch', 'checkpoint', 'checkpoint_sha256'}
    require({k: v for k, v in spec.items() if k not in extras} == template and
            spec['template_sha256'] == hashlib.sha256(canonical(template).encode()).hexdigest(), 'Cell differs from frozen schedule')
    require(type(spec['deadline_utc_epoch']) in (int, float) and math.isfinite(spec['deadline_utc_epoch']),
            'Missing runtime deadline')
    if spec['method'] in ('greedy', 'beam'):
        require(not spec.get('checkpoint') and not spec.get('checkpoint_sha256'), 'Rule secretly loaded model')
    require(result['spec'] == spec and result['formal_holdout_numerical_access'] is False
            and result['tune_numerical_access'] is False and result['optimizer_updates'] == 0,
            'Cell scope or in-worker learning differs')
    require(spec['point_budget'] == (1024 if spec['phase'] == 'reference' else 256), 'Frozen point budget changed')
    cohort = read(run / ('evaluation_contexts.json' if spec['phase'] in ('eval', 'evaluation') else 'training_contexts.json'))
    require({'scenario_id': spec['scenario_id'], 'event_id': spec['event_id']} in cohort, 'Cross-partition physical access')
    scenarios = {s['scenario_id']: s for s in read(run / 'selected_scenarios.json')}
    events = {e['event_id']: e for e in read(run / 'selected_events.json')}
    mech = Mechanics(read(run / 'author_static_metadata.json'), read(run / 'load_reference.json'),
                     scenarios[spec['scenario_id']], read(run / 'condition.json'), events[spec['event_id']], cfg)
    require(read(cell / 'initial_state.json') == mech.initial, 'Initial state/history differs')
    points, counts = check_raw_points(run, cell, spec, cfg, mech)
    commands = old.check_commands(cell, points, read(run / 'command_inventory.json'))
    requests = old.check_requests(cell, points, spec['point_budget'])
    oracle = read(cell / 'oracle_summary.json')
    require(result['oracle'] == oracle and oracle['status'] == 'COMPLETE_COUNTERS_PENDING_AUDIT'
            and oracle['points'] == len(points) and oracle['point_budget'] == spec['point_budget']
            and oracle['requests'] == requests['requests'] and oracle['cache_hits'] == requests['cache_hits']
            and oracle['statuses'] == dict(counts) and oracle['actual_numerical_entries'] == commands['numerical_entries']
            and oracle['maximum_numerical_entries'] == 3 + 4 * spec['point_budget']
            and oracle['point_seconds'] == [p['elapsed_seconds'] for p in points.values()], 'Oracle budget/cost/timing differs')
    require(list(read_lines(cell / 'plans.jsonl.gz')) == result['attempts'], 'Omitted paid candidate')
    transitions = read_lines(cell / 'transitions.jsonl.gz')
    decisions = read_lines(cell / 'decisions.jsonl.gz')
    branch_cfg = {**cfg['branch_search'], 'point_budget': spec['point_budget']}
    require(result['config'] == branch_cfg, 'Branch mechanics configuration differs')
    branch = check_branch_records(mech, branch_cfg, result, transitions, decisions,
                                  list(read_lines(cell / 'requests.jsonl')), points, sha(run / 'condition.json'))
    require(result['training_interactions'] == (branch['decisions'] if spec['phase'] == 'train' else 0), 'Learning interaction count differs')
    policy = None
    if spec['method'] not in ('greedy', 'beam'):
        checkpoint = (run / spec['checkpoint']).resolve()
        require(checkpoint.is_relative_to(run) and sha(checkpoint) == spec['checkpoint_sha256'], 'Changed/outside policy checkpoint')
        from audit_nature34_clpu_branch_learning import check_policy_transitions
        # A fresh streaming pass preserves the complete independent policy
        # replay without retaining the full historical records in memory.
        policy = check_policy_transitions(checkpoint, read_lines(cell / 'transitions.jsonl.gz'), spec)
    return {'cell': str(cell.relative_to(run)), 'spec': spec, 'status': result['status'], 'points': len(points),
            'physical_statuses': dict(counts), 'commands': commands, 'requests': requests, 'branch': branch,
            'policy': policy, 'plan_statuses': dict(Counter(a['status'] for a in result['attempts'])),
            'cell_completion_sha256': sha(cell / 'completion_manifest.json')}


def check_parent(run):
    cfg, prepared = read(run / 'config.json'), read(run / 'prepared.json')
    require(cfg['decision'] == 'D059' and prepared['status'] == 'FROZEN_PREPARED_NO_DSS', 'Unfrozen D059 run')
    require(sha(run / 'prephysical_freeze_manifest.json') == prepared['freeze_sha256'], 'Prephysical freeze changed')
    for name, digest in read(run / 'prephysical_freeze_manifest.json').items():
        require(sha(run / name) == digest, 'Frozen file changed: ' + name)
    for name in prepared['source_files']:
        require(sha(ROOT / name) == sha(run / 'source' / name), 'Frozen implementation changed: ' + name)
    require('scripts/audit_nature34_joint_planning.py' in {n.replace('\\', '/') for n in prepared['source_files']},
            'Independent graph auditor was not frozen')
    from nature34_experiment_support import dependencies
    require(dependencies() == read(run / 'dependencies.json') and
            sha(Path(sys.executable)) == read(run / 'python_environment.json')['sha256'], 'Dependency/interpreter changed')
    for name, expected in [('horizon', 30), ('slot_seconds', 60), ('sample_seconds', 30), ('alpha', .5), ('algorithm', 0)]:
        require(cfg[name] == expected, 'Physical/time configuration changed: ' + name)
    for name in ('formal_holdout_numerical_access', 'tune_numerical_access', 'test_access', 'tune_access'):
        require(cfg.get(name, False) is False, 'Forbidden numerical access')
    provenance = read(run / 'source_provenance.json')
    require(provenance['source_results_imported'] is False, 'D058 result-answer import')
    source = ROOT / 'runs' / '20261024T022849_740635Z_nature34_clpu_main'
    require(sha(source / 'completion_manifest.json') == '077aada1355fbf353ae3828100fc13cb4004aa4126dc037fcecf16b3926e5bcf',
            'D058 source identity changed')
    source_manifest = read(source / 'completion_manifest.json')
    for path in (run / 'source_inputs').rglob('*'):
        if path.is_file():
            name = str(path.relative_to(run / 'source_inputs'))
            require(name in source_manifest and sha(path) == source_manifest[name], 'Inherited scientific input changed: ' + name)
    for name in ('author_static_metadata.json', 'load_reference.json', 'physical_config.json', 'conditions.json',
                 'operating_reference.json', 'command_inventory.json', 'source_pointer.json', 'training_scenarios.json'):
        require(sha(run / name) == source_manifest[name], 'D058 scientific input changed: ' + name)
    scenarios = read(run / 'training_scenarios.json')
    ordered = sorted(scenarios, key=lambda s: hashlib.sha256(('clpu-pilot-faults-v1|' + s['scenario_id']).encode()).hexdigest())
    require(len(ordered) == len({s['scenario_id'] for s in ordered}) == 16 and
            all(s['partition'] == 'build' and s['fault_order'] == 2 for s in ordered), 'Fault population differs')
    require(read(run / 'selected_scenarios.json') == ordered and ordered[:8] == read(source / 'selected_scenarios.json'),
            'Outcome-dependent or overlapping fault split')
    events = sorted(range(1, 32), key=lambda i: hashlib.sha256(f'clpu-event-split-v1|{i}'.encode()).hexdigest())
    split = {'train': events[:19], 'dev': events[19:25], 'holdout': events[25:]}
    require(read(run / 'event_split.json') == split, 'Whole-event split changed')
    for cohort, faults, ids, expected in [('training', ordered[:8], split['train'][:4], 32),
                                         ('evaluation', ordered[8:], split['dev'][:2], 16)]:
        contexts = [{'scenario_id': s['scenario_id'], 'event_id': e} for s in faults for e in ids]
        require(read(run / (cohort + '_contexts.json')) == contexts and len(contexts) == expected,
                'Cohort changed: ' + cohort)
    # Source event parameters were already raw-CSV checked in D058; newly used
    # dev events are independently parsed by the same old source-only checker.
    archive = ROOT / 'data/raw/literature/clpu_design/20261024T015725_552682Z/hachmann2022_clpu_data.zip'
    require(sha(archive) == 'e46886903929d49380a7f1342588adebd26c2524667dbd74fa89021ccea27a22', 'CLPU source changed')
    import zipfile
    with zipfile.ZipFile(archive) as z:
        parameter_rows = [line.strip() for line in z.read('parameters.csv').decode('utf-8-sig').splitlines() if line.strip()]
    require(len(parameter_rows) == 32, 'Incomplete original CLPU events')
    selected_events = read(run / 'selected_events.json')
    require([e['event_id'] for e in selected_events] == split['train'][:4] + split['dev'][:2], 'Event selection differs')
    for event in selected_events:
        tokens = [token.strip() for token in parameter_rows[event['event_id']].split(',')]
        require(len(tokens) in (9, 10), 'Unknown original CLPU schema')
        close(event['a'], float(tokens[0] + '.' + tokens[1]), 'Original paired ED amplitude')
        close(event['tau_seconds'], float(tokens[2] + '.' + tokens[3]), 'Original paired ED decay')
        require(event['parameter_row'] == parameter_rows[event['event_id']], 'Original paired event row replaced')
    require(read(run / 'condition.json') == read(source / 'condition.json'), 'Inherited normal operating point changed')
    check_schedule(run)
    return cfg


def check_schedule(run):
    """Independently rebuild the entire planned population before any results."""
    training, evaluation = read(run / 'training_contexts.json'), read(run / 'evaluation_contexts.json')
    digest = lambda domain, value: hashlib.sha256((domain + '|' + canonical(value)).encode()).hexdigest()
    reference = sorted(training, key=lambda r: digest('d059-reference-v1', r))[:8]
    require(read(run / 'reference_contexts.json') == reference, 'Reference subset depends on outcome')
    templates = []

    def add(context, phase, method, seed, group, **extra):
        templates.append({**context, 'phase': phase, 'method': method, 'seed': seed, 'group': group,
                          'representation': 'associated' if method in ('supervised', 'bc_associated', 'ppo_associated') else 'raw',
                          'deterministic': phase != 'train', 'point_budget': 1024 if phase == 'reference' else 256, **extra})

    for context in training:
        for method in ('greedy', 'beam'):
            add(context, 'teacher', method, 0, 'mechanism' if context in reference else 'teacher_remaining')
    for context in reference:
        add(context, 'reference', 'beam', 0, 'mechanism')
    for representation in ('raw', 'associated'):
        for seed in (17, 41, 73):
            order = list(training)
            random.Random(seed).shuffle(order)
            for round_index in range(8):
                for context in (order * 2)[round_index * 8:(round_index + 1) * 8]:
                    sampling_seed = int(digest('d059-sample-v1', {'seed': seed, 'round': round_index, 'context': context})[:15], 16)
                    add(context, 'train', 'ppo_' + representation, seed, f'train_{representation}_{seed}_{round_index:02d}',
                        round_index=round_index, sampling_seed=sampling_seed)
    for context in evaluation:
        for method in ('greedy', 'beam'):
            add(context, 'evaluation', method, 0, 'evaluation')
        for seed in (17, 41, 73):
            for method in ('supervised', 'bc_raw', 'ppo_raw', 'bc_associated', 'ppo_associated'):
                add(context, 'evaluation', method, seed, 'evaluation')
    templates.sort(key=lambda r: digest('d059-cell-order-v1', r))
    for index, row in enumerate(templates):
        row.update(episode_id=index, cell=f'episodes/{index:04d}')
    require(read(run / 'schedule.json') == templates and len(templates) == 728, 'Frozen 728-cell population changed')
    require(sum(r['point_budget'] for r in templates) == 192512, 'Allocation differs')
    resource = read(run / 'resource_budget.json')
    require(resource['episodes'] == 728 and resource['allocated_physical_points'] == 192512
            and resource['maximum_physical_points'] == 200000 and resource['unallocated_reserve_points'] == 7488
            and resource['reserve_automatic_use'] is False and resource['maximum_numerical_entries'] == 728 * 3 + 4 * 192512
            and resource['deadline_seconds_including_audits'] == 28800 and resource['automatic_retry'] is False,
            'Frozen global resource bounds differ')
    return templates


def audit_cells(run, cell_paths, phase):
    run = Path(run).resolve()
    require(run.parent == ROOT / 'runs', 'Source run outside project')
    output = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_nature34_clpu_branch_audit')
    output.mkdir()
    began = time.perf_counter()
    result = {'status': 'FAILED_CLPU_BRANCH_CELLS', 'source_run': run.name, 'phase': phase, 'errors': [],
              'new_powerflow_calls': 0, 'new_training_interactions': 0, 'formal_holdout_numerical_access': False,
              'tune_numerical_access': False, 'audit_run': output.name}
    checks = []
    try:
        cfg = check_parent(run)
        cells = [(run / p).resolve() if not Path(p).is_absolute() else Path(p).resolve() for p in cell_paths]
        require(cells and len(cells) == len(set(cells)), 'Empty/duplicate audit cell population')
        scheduled = {(run / row['cell']).resolve() for row in read(run / 'schedule.json') if row['group'] == phase}
        require(set(cells) == scheduled, 'Audit batch omits/adds scheduled cells')
        save(output / 'requested_cells.json', [str(p.relative_to(run)) for p in cells])
        for cell in cells:
            checked = check_cell(run, cell, cfg)
            checks.append(checked)
            save(output / 'progress.json', {'completed': len(checks), 'scheduled': len(cells), 'errors': [],
                                           'elapsed_seconds': time.perf_counter() - began})
        require('opendssdirect' not in sys.modules and 'dss' not in sys.modules, 'Audit initialized physical runtime')
        result.update(status='VERIFIED_CLPU_BRANCH_CELLS', episodes=len(checks),
                      physical_points=sum(c['points'] for c in checks),
                      numerical_entries=sum(c['commands']['numerical_entries'] for c in checks),
                      physical_statuses=dict(sum((Counter(c['physical_statuses']) for c in checks), Counter())),
                      freeze_sha256=sha(run / 'prephysical_freeze_manifest.json'),
                      dependencies_sha256=sha(run / 'dependencies.json'),
                      schedule_sha256=sha(run / 'schedule.json'),
                      python_sha256=sha(Path(sys.executable)))
    except BaseException as exc:
        result['errors'].append({'error': repr(exc), 'traceback': traceback.format_exc()})
    result['elapsed_seconds'] = time.perf_counter() - began
    save(output / 'cell_checks.json', checks)
    save(output / 'verification.json', result)
    save(output / 'source_hashes.json', {str(Path(p).relative_to(ROOT)): sha(p) for p in
         (__file__, old.__file__, ROOT / 'scripts/audit_nature34_joint_planning.py')})
    save(output / 'completion_manifest.json', {str(p.relative_to(output)): sha(p) for p in output.rglob('*')
                                              if p.is_file() and p.name != 'completion_manifest.json'})
    result['completion_sha256'] = sha(output / 'completion_manifest.json')
    save(ROOT / 'artifacts/nature34_clpu_branch_audit_latest.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--cells-file', type=Path, required=True)
    parser.add_argument('--phase', required=True)
    args = parser.parse_args()
    result = audit_cells(args.run, read(args.cells_file), args.phase)
    print(canonical(result), flush=True)
    return 0 if not result['errors'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

