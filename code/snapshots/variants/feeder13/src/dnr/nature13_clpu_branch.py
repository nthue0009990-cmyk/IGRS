# NC13 dimension/import adapter of nature34_clpu_branch.py; original method rules retained.
"""Shared, budgeted CLPU branch expansion; no solver or learner is imported.

Policies choose a currently exposed branch.  Its paid full-window check and
the subsequent graph expansion are common to every policy.  Prefix + WAIT
plans are first-class candidates, rather than an unverified partial fallback.
The bounded action window never deletes its unexposed frontier entries.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

from .nature13_clpu_environment import CLPUBudgetExhausted, CLPUReliabilityError
from .nature13_clpu_search import (
    SearchConfig, _feedback_exposure, generic_neighbors, graph_plan,
)


@dataclass(frozen=True)
class BranchConfig:
    lookahead_depth: int = 6
    max_generated_candidates: int = 5000
    max_checked_candidates: int = 256
    max_branch_expansions: int = 256
    beam_width: int = 8
    point_budget: int = 256
    action_window: int = 32
    generic_neighbor_limit: int = 32
    reward_cost_weight: float = .001
    reward_cost_normalizer: int = 256

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key == 'reward_cost_weight':
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError('Finite nonnegative point-cost weight required')
                continue
            if type(value) is not int or value <= 0:
                raise ValueError('Positive integer branch bound required: ' + key)
        if self.lookahead_depth > 12 or self.action_window > 32:
            raise ValueError('Lookahead or common action-window scope exceeded')
        if self.beam_width > self.action_window:
            raise ValueError('Beam width exceeds the common action window')


def _state_dict(state):
    return {'command': int(state.command), 'step': int(state.step),
            'pickups': list(state.pickups)}


def _violations(result):
    value = result.get('feedback', []) if result else []
    if isinstance(value, Mapping):
        if not value.get('usable_for_repair', False):
            return []
        value = value.get('violations', [])
    return [dict(row) for row in value if isinstance(row, Mapping)]


class BranchSearch:
    """A single cold search episode, sharing all mechanics across policies.

    The oracle owns physical caching, logging and its hard point budget, and
    exposes ``points``/``budget`` (``used``/``limit`` also support synthetic
    tests).  Construction certifies the initial snapshot and complete WAIT
    baseline.  Reliability exceptions propagate; only budget exhaustion is a
    normal stop.  No learned or voluntary STOP action is provided: all policies
    share exactly the same certified bound, resource and frontier stops.
    """

    def __init__(self, env, oracle, config=None, record_callback=None):
        if isinstance(config, Mapping):
            config = BranchConfig(**config)
        self.config = config or BranchConfig()
        self.env, self.oracle, self.record_callback = env, oracle, record_callback
        self.start = env.initial
        self.horizon = int(env.horizon) - int(self.start.step)
        self.total_kw = float(env.total_kw)
        if self.horizon <= 0 or not math.isfinite(self.total_kw) or self.total_kw <= 0:
            raise ValueError('Positive remaining horizon and normal load required')
        self.denominator = self.total_kw * self.horizon * env.slot_seconds / 3600.
        # JointGraph has already enumerated every initial-reachable command
        # state without physical solves.  Its maximum exact normal-load weight
        # bounds each future slot, even after generic edits to an earlier
        # prefix.  This is not a depth-six completion and not an electrical
        # feasibility claim.  Match service_kw's arithmetic before fsum so that
        # an exactly attained bound can stop without adding a loose tolerance.
        if not (hasattr(env.graph, 'states') and hasattr(env.graph, 'served_weight')
                and hasattr(env.graph, 'weight_scale')):
            raise ValueError('Complete initial-reachable graph required for the common upper bound')
        maximum_weight = max(env.graph.served_weight(int(state)) for state in env.graph.states)
        self.max_graph_kw = (maximum_weight / env.graph.weight_scale) * env.load_multiplier
        self.global_upper_bound = math.fsum(
            self.max_graph_kw * env.slot_seconds / 3600. for _ in range(self.horizon))
        oracle_budget = getattr(oracle, 'budget', getattr(oracle, 'limit', None))
        if oracle_budget != self.config.point_budget:
            raise ValueError('Oracle and frozen branch point budgets differ')
        self._points()  # Require a real counter; never silently report zero.
        self.local_config = SearchConfig(
            lookahead_depth=self.config.lookahead_depth,
            local_prefix_steps=self.config.lookahead_depth,
            generic_neighbor_limit=self.config.generic_neighbor_limit,
            max_generated_candidates=self.config.max_generated_candidates,
            max_checked_candidates=self.config.max_checked_candidates,
        )
        self.done = False
        self.stop_reason = None
        self.initial_certificate = None
        self.incumbent = None
        self._incumbent_key = None
        self.attempts, self.trace, self.generation_events = [], [], []
        self.generation_trace, self.graph_assessment_trace = [], []
        self._assessed, self._checked, self._frontier = {}, {}, {}
        self._branch_keys, self._expanded_prefixes = set(), set()
        self._next_id = 0
        self.generation_cap_rejections = 0
        self.latest_failure = None
        self.observed_failure_history = []
        self._beam_turn = 0
        try:
            self.initial_certificate = deepcopy(oracle.certify(self.start, 0.))
            self._require_certificate(self.initial_certificate)
            if self.initial_certificate['status'] == 'FAIL':
                self._stop('INITIAL_FAIL')
                return
            wait = (0,) * self.horizon
            info = self._assess(wait)
            if info is None:
                raise RuntimeError('WAIT graph plan unexpectedly absent')
            self._evaluate(info, 'wait_incumbent', {'bootstrap': True})
            if not self.done:
                self._expand_prefix((), parent_id=None)
                self._refresh_stop()
        except CLPUBudgetExhausted:
            self._stop('PHYSICAL_BUDGET_EXHAUSTED')

    def _points(self):
        value = getattr(self.oracle, 'points', getattr(self.oracle, 'used', None))
        if type(value) is not int or value < 0:
            raise ValueError('Oracle must expose a nonnegative integer point counter')
        if value > self.config.point_budget:
            raise CLPUReliabilityError('Oracle exceeded the frozen physical point budget')
        return value

    @staticmethod
    def _require_certificate(cert):
        if cert.get('reliable') is not True or cert.get('status') not in ('PASS', 'FAIL'):
            raise CLPUReliabilityError('Unknown initial certificate cannot initialize search')

    def _stop(self, reason):
        self.done, self.stop_reason = True, reason

    def _assess(self, actions):
        actions = tuple(actions)
        if actions in self._assessed:
            return self._assessed[actions]
        if len(self._assessed) >= self.config.max_generated_candidates:
            self.generation_cap_rejections += 1
            return None
        try:
            info = graph_plan(self.env, self.start, actions)
        except ValueError:
            info = None
        self._assessed[actions] = info
        self.graph_assessment_trace.append({'actions': list(actions), 'graph_valid': info is not None})
        return info

    def _add(self, prefix, actions, origin, parent_id):
        prefix, actions = tuple(prefix), tuple(actions)
        key = (prefix, actions)
        if key in self._branch_keys:
            return
        if len(self._branch_keys) >= self.config.max_generated_candidates:
            self.generation_cap_rejections += 1
            return
        info = self._assess(actions)
        if info is None:
            return
        if actions[:len(prefix)] != prefix:
            raise ValueError('Branch plan does not retain its declared prefix')
        branch_id = self._next_id
        self._next_id += 1
        self._branch_keys.add(key)
        # Generic replacement can edit earlier actions.  The complete-graph
        # per-slot maximum is valid for all descendants of every such edit.
        self._frontier[branch_id] = {
            'id': branch_id, 'prefix': prefix, 'info': info, 'origin': origin,
            'parent_id': parent_id, 'upper_bound_kwh': self.global_upper_bound,
            'bound_scope': 'maximum_initial_reachable_graph_service_in_every_slot',
        }
        self.generation_trace.append({
            'branch_id': branch_id, 'prefix': list(prefix), 'actions': list(actions),
            'origin': origin, 'parent_id': parent_id,
            'upper_bound_kwh': self.global_upper_bound,
            'bound_scope': 'maximum_initial_reachable_graph_service_in_every_slot',
            'graph_energy_kwh': info['energy_kwh'], 'graph_final_kw': info['final_kw'],
            'operations': info['operations'],
        })

    def _expand_prefix(self, prefix, parent_id):
        prefix = tuple(prefix)
        if prefix in self._expanded_prefixes:
            return
        self._expanded_prefixes.add(prefix)
        if len(prefix) >= min(self.config.lookahead_depth, self.horizon):
            return
        state = self.start
        for action in prefix:
            state = self.env.advance(state, action)
        for action in sorted(self.env.legal_actions(state)):
            child_prefix = prefix + (int(action),)
            child = self.env.advance(state, int(action))
            self._add(child_prefix, child_prefix + (0,) * (self.horizon - len(child_prefix)),
                      'prefix_wait', parent_id)
            suffix = tuple(self.env.optimistic_plan(child, depth=self.config.lookahead_depth))
            self._add(child_prefix, child_prefix + suffix, 'prefix_graph_completion', parent_id)

    def _emit(self, record):
        self.attempts.append(record)
        if self.record_callback is not None:
            self.record_callback(deepcopy(record))

    def _evaluate(self, info, origin, details):
        actions = info['actions']
        if actions in self._checked:
            return self._checked[actions]
        if len(self._checked) >= self.config.max_checked_candidates:
            self._stop('LOGICAL_CHECK_CAP')
            return None
        before = self._points()
        try:
            validation = self.env.validate_plan(self.start, list(actions), self.oracle)
        except CLPUBudgetExhausted:
            record = {'actions': list(actions), 'origin': origin, 'details': details,
                      'status': 'BUDGET_EXHAUSTED_PARTIAL', 'certified_complete': False,
                      'physical_points_before': before, 'physical_points_after': self._points()}
            self._checked[actions] = record
            self._emit(record)
            self._stop('PHYSICAL_BUDGET_EXHAUSTED')
            return record
        status = validation.get('status')
        if status not in ('COMPLETE', 'FAIL'):
            raise CLPUReliabilityError('Unexpected full-path validation status: ' + str(status))
        record = {'actions': list(actions), 'origin': origin, 'details': details,
                  'status': status, 'certified_complete': status == 'COMPLETE',
                  'validation': validation, 'physical_points_before': before,
                  'physical_points_after': self._points()}
        self._checked[actions] = record
        self._emit(record)
        if status == 'COMPLETE':
            energy = float(validation['energy_kwh'])
            if not math.isfinite(energy) or not math.isclose(
                    energy, info['energy_kwh'], rel_tol=1e-12, abs_tol=1e-10):
                raise CLPUReliabilityError('Certified and graph normal-service objectives disagree')
            key = info['priority']
            if self._incumbent_key is None or key < self._incumbent_key:
                self.incumbent, self._incumbent_key = record, key
        else:
            self.latest_failure = deepcopy(validation)
            self.observed_failure_history.append(deepcopy(validation))
        return record

    def _window(self):
        """Common deterministic window: objective leaders, holds, then oldest.

        Half the slots retain the globally best graph scores, a quarter expose
        partial-restoration holds, and the rest use FIFO.  Unexposed nodes stay
        in the frontier, and every policy receives this identical window.
        """
        rows = list(self._frontier.values())
        ranked = sorted(rows, key=lambda b: (b['info']['priority'], b['id']))
        size = self.config.action_window
        selected, seen = [], set()

        def take(source, count):
            if count <= 0:
                return
            added = 0
            for row in source:
                if row['id'] not in seen:
                    selected.append(row)
                    seen.add(row['id'])
                    added += 1
                    if added >= count:
                        break

        take(ranked, max(1, size // 2))
        take((b for b in ranked if b['origin'] == 'prefix_wait'), max(1, size // 4))
        take(sorted(rows, key=lambda b: b['id']), size - len(selected))
        return selected[:size]

    def _refresh_stop(self):
        if self.done:
            return
        if self.incumbent is not None:
            energy = float(self.incumbent['validation']['energy_kwh'])
            # No numerical tolerance may convert a lower incumbent into a
            # proof of the bound.  Include ALL frontier entries, not the window.
            if self._frontier and all(b['upper_bound_kwh'] <= energy for b in self._frontier.values()):
                self._stop('CERTIFIED_FRONTIER_UPPER_BOUND')
                return
        if self._points() >= self.config.point_budget:
            self._stop('PHYSICAL_BUDGET_EXHAUSTED')
        elif len(self._checked) >= self.config.max_checked_candidates:
            self._stop('LOGICAL_CHECK_CAP')
        elif self._beam_turn >= self.config.max_branch_expansions:
            self._stop('BRANCH_EXPANSION_CAP')
        elif not self._frontier:
            self._stop('BOUNDED_FRONTIER_EXHAUSTED')

    def rule_action(self, method='greedy'):
        if self.done:
            raise RuntimeError('Cannot choose a branch after search is done')
        window = self._window()
        if method == 'greedy':
            return min(range(len(window)), key=lambda i: (window[i]['info']['priority'], window[i]['id']))
        if method == 'beam':
            # Width-eight layered beam-style expansion rule, not a reproduction
            # of permanently pruned conventional beam search.  It ranks up to
            # eight distinct prefixes at the shallowest exposed depth and uses
            # deterministic rotation.  The common frontier is never deleted.
            depth = min(len(b['prefix']) for b in window)
            candidates = sorted((i for i, b in enumerate(window) if len(b['prefix']) == depth),
                                key=lambda i: (window[i]['info']['priority'], window[i]['id']))
            best_by_prefix = {}
            for i in candidates:
                best_by_prefix.setdefault(window[i]['prefix'], i)
            beam = list(best_by_prefix.values())[:self.config.beam_width]
            # A concrete non-learning use of the SAME observed CLPU/violation
            # information. Among this fixed-width objective-ranked beam, first
            # prefer smaller cold excess on the observed violating path; this
            # is a ranking heuristic, never an infeasibility certificate.
            obs = self.observation('associated')
            exposure_column = obs['candidate_feature_names'].index('cold_excess_exposed_weight')
            return min(beam, key=lambda i: (obs['candidate_features'][i][exposure_column],
                                            window[i]['info']['priority'], window[i]['id']))
        raise ValueError('Unknown shared-engine rule: ' + str(method))

    def step(self, branch_index):
        if self.done:
            raise RuntimeError('Cannot expand a branch after search is done')
        window = self._window()
        if type(branch_index) is not int or not 0 <= branch_index < len(window):
            raise ValueError('Branch index outside the common action window')
        node = window[branch_index]
        del self._frontier[node['id']]
        before_points = self._points()
        before_quality = self.quality
        before_generated = len(self._branch_keys)
        plan, info = node['info']['actions'], node['info']
        details = {'branch_id': node['id'], 'parent_id': node['parent_id'],
                   'prefix': list(node['prefix']), 'window_branch_ids': [b['id'] for b in window],
                   'selected_index': branch_index, 'frontier_size_before': len(self._frontier) + 1}
        record = self._checked.get(plan)
        query_reason = 'previously_checked' if record else 'query'
        if record is None and self._incumbent_key is not None and info['priority'] >= self._incumbent_key:
            # It may still be a useful expansion parent, but paying to verify a
            # known inferior complete plan cannot improve the incumbent.
            query_reason = 'dominated_known_complete_objective'
        elif record is None:
            record = self._evaluate(info, node['origin'], details)
        if not self.done:
            validation = record.get('validation', {}) if record else {}
            failed_step = validation.get('failed_step') if validation.get('status') == 'FAIL' else None
            invalid_prefix = isinstance(failed_step, int) and failed_step < len(node['prefix'])
            if not invalid_prefix:
                self._expand_prefix(node['prefix'], node['id'])
            # Existing common local replacements can repair an invalid prefix,
            # so a prefix failure never discards these other legal branches.
            for partial_length in range(1, min(self.config.lookahead_depth, self.horizon) + 1):
                prefix = plan[:partial_length]
                self._add(prefix, prefix + (0,) * (self.horizon - partial_length),
                          'prefix_wait', node['id'])
            for neighbor, kind, _ in generic_neighbors(
                    self.env, self.start, plan, self.local_config,
                    failed_step=failed_step, assess=self._assess):
                prefix = neighbor['actions'][:min(self.config.lookahead_depth, self.horizon)]
                self._add(prefix, neighbor['actions'], kind, node['id'])
        self._beam_turn += 1
        self._refresh_stop()
        event = {**details, 'origin': node['origin'], 'actions': list(plan),
                 'query_reason': query_reason, 'validation_status': record['status'] if record else None,
                 'quality_before': before_quality, 'quality_after': self.quality,
                 'physical_points_before': before_points, 'physical_points_after': self._points(),
                 'physical_points_used': self._points() - before_points,
                 'new_branches': len(self._branch_keys) - before_generated,
                 'frontier_size_after': len(self._frontier), 'done': self.done,
                 'stop_reason': self.stop_reason}
        event['reward'] = (event['quality_after'] - event['quality_before']
                           - self.config.reward_cost_weight * event['physical_points_used']
                           / self.config.reward_cost_normalizer)
        event['info'] = {'validation_status': event['validation_status'],
                         'physical_points_used': event['physical_points_used'],
                         'query_reason': query_reason, 'stop_reason': self.stop_reason}
        self.trace.append(event)
        self.generation_events.append({'branch_id': node['id'], 'new_branches': event['new_branches'],
                                       'assessed_graph_candidates': len(self._assessed)})
        return deepcopy(event)

    @property
    def quality(self):
        return (float(self.incumbent['validation']['energy_kwh']) / self.denominator
                if self.incumbent is not None else 0.)

    def observation(self, representation='raw'):
        if representation not in ('raw', 'associated'):
            raise ValueError('Representation must be raw or associated')
        window = [] if self.done else self._window()
        names = tuple(self.env.load_names)
        weights = [float(self.env.loads[n]['nominal_kw']) * self.env.load_multiplier / self.total_kw
                   for n in names]
        failure = self.latest_failure or {}
        feedback = _violations(failure)
        failed_step = failure.get('failed_step')
        sample_index = min(max(int(failed_step or 0) + 1, 1), self.horizon)
        failure_seconds = float(failure.get('failed_time_seconds', (sample_index - 1) * self.env.slot_seconds))
        duration = self.horizon * self.env.slot_seconds
        magnitudes = [float(v.get('magnitude', 0.)) for v in feedback]
        if any(not math.isfinite(v) for v in magnitudes):
            raise CLPUReliabilityError('Nonfinite observed feedback magnitude')
        magnitude = max([0.] + [abs(v) for v in magnitudes])
        a, tau = float(self.env.event['a']), float(self.env.event['tau_seconds'])
        global_features = [self.quality, self._points() / self.config.point_budget,
                           len(self._checked) / self.config.max_checked_candidates,
                           len(self._assessed) / self.config.max_generated_candidates,
                           float(bool(feedback)), failure_seconds / duration,
                           magnitude / (1. + magnitude), a / (1. + abs(a)), tau / (tau + duration),
                           sum('voltage' in str(v.get('constraint', '')) for v in feedback) / max(1, len(feedback)),
                           sum('thermal' in str(v.get('constraint', '')) for v in feedback) / max(1, len(feedback))]
        global_names = ['incumbent_fraction', 'point_budget_used', 'queries_used', 'graph_cap_used',
                        'has_observed_failure', 'failure_time_fraction', 'failure_magnitude_bounded',
                        'event_a_bounded', 'event_tau_bounded', 'voltage_feedback_fraction',
                        'thermal_feedback_fraction']
        global_features += weights
        global_names += ['normal_weight:' + n for n in names]
        feature_names = ['known_fraction', 'known_final_fraction', 'operations_fraction',
                         'prefix_depth_fraction', 'upper_bound_fraction', 'is_prefix_wait',
                         'already_checked', 'source_path_mapping_available']
        feature_names += ['action:' + str(i) for i in range(self.horizon)]
        feature_names += ['pickup_time:' + n for n in names]
        feature_names += ['feedback_time_pickup:' + n for n in names]
        feature_names += ['feedback_path_exposure:' + n for n in names]
        feature_names += ['command_bit:' + str(i) for i in range(11)]
        associated_names = ['exposed_normal_weight', 'cold_exposed_weight', 'new_batch_exposed_weight',
                            'cold_excess_exposed_weight', 'failure_age_weighted_exposure',
                            'exposed_unpicked_weight', 'pickup_batch_count_fraction',
                            'largest_pickup_batch_weight']
        matrix = []
        rows = []
        for node in window:
            info = node['info']
            state = info['states'][sample_index]
            exposed, available = _feedback_exposure(
                self.env, state, {'usable_for_repair': True, 'violations': feedback}) if feedback else (set(), False)
            final = info['states'][-1]

            def normalized(p):
                return -2. if p is None else -1. if p == -1 else float(p) / duration

            vector = [info['energy_kwh'] / self.denominator, info['final_kw'] / self.total_kw,
                      info['operations'] / self.horizon, len(node['prefix']) / self.horizon,
                      node['upper_bound_kwh'] / self.denominator, float(node['origin'] == 'prefix_wait'),
                      float(info['actions'] in self._checked), float(available)]
            vector += [x / 11. for x in info['actions']]
            vector += [normalized(p) for p in final.pickups]
            vector += [normalized(p) for p in state.pickups]
            vector += [float(i in exposed) for i in range(len(names))]
            vector += [float((state.command >> i) & 1) for i in range(11)]
            # Raw and associated observers have identical underlying facts.
            # Associated features are explicit products/sums of the raw facts,
            # event parameters and public load weights, never future physics.
            if representation == 'associated':
                cold = {i for i, p in enumerate(state.pickups) if p is not None and p >= 0}
                new = {i for i in cold if state.pickups[i] == (sample_index - 1) * self.env.slot_seconds}
                groups = {}
                for i, pickup in enumerate(final.pickups):
                    if pickup is not None and pickup >= 0:
                        groups[pickup] = groups.get(pickup, 0.) + weights[i]
                ages = {i: max(0., failure_seconds - state.pickups[i]) for i in cold}
                vector += [sum(weights[i] for i in exposed),
                           sum(weights[i] for i in exposed & cold),
                           sum(weights[i] for i in exposed & new),
                           sum(weights[i] * a * math.exp(-ages[i] / tau) for i in exposed & cold),
                           sum(weights[i] * ages[i] / duration for i in exposed & cold),
                           sum(weights[i] for i in exposed if state.pickups[i] is None),
                           len(groups) / self.horizon, max(groups.values(), default=0.)]
            if not all(math.isfinite(x) for x in vector):
                raise CLPUReliabilityError('Nonfinite search observation')
            matrix.append(vector)
            rows.append({'branch_id': node['id'], 'origin': node['origin'],
                         'prefix': list(node['prefix']), 'actions': list(info['actions']),
                         'graph_energy_kwh': info['energy_kwh'],
                         'upper_bound_kwh': node['upper_bound_kwh']})
        return {'representation': representation, 'candidate_features': matrix,
                'global_features': global_features, 'branch_ids': [b['id'] for b in window],
                'action_mask': [True] * len(window), 'mask': [True] * len(window), 'candidate_feature_names':
                feature_names + (associated_names if representation == 'associated' else []),
                'global_feature_names': global_names, 'branches': rows,
                'observed_feedback': deepcopy(feedback), 'done': self.done,
                'observed_failure_history': deepcopy(self.observed_failure_history),
                'raw_information_identical': True}

    def result(self):
        status = ('INITIAL_FAIL' if self.stop_reason == 'INITIAL_FAIL' else
                  'CERTIFIED_INCUMBENT' if self.incumbent is not None else 'NO_CERTIFIED_COMPLETE_PLAN')
        return {'status': status, 'method': 'shared_clpu_branch_search', 'config': asdict(self.config),
                'done': self.done, 'stop_reason': self.stop_reason,
                'initial_certificate': deepcopy(self.initial_certificate),
                'best_plan': deepcopy(self.incumbent),
                'incumbent': deepcopy(self.incumbent), 'attempts': deepcopy(self.attempts),
                'trace': deepcopy(self.trace), 'generation_events': deepcopy(self.generation_events),
                'generation_trace': deepcopy(self.generation_trace),
                'graph_assessment_trace': deepcopy(self.graph_assessment_trace),
                'generated_unique_candidates': len(self._branch_keys),
                'assessed_graph_candidates': len(self._assessed),
                'checked_candidates': len(self._checked), 'frontier_size': len(self._frontier),
                'generation_cap_rejections': self.generation_cap_rejections,
                'actual_physical_points': self._points(),
                'maximum_reachable_graph_kw': self.max_graph_kw,
                'global_upper_bound_kwh': self.global_upper_bound,
                'optimality_established': self.stop_reason == 'CERTIFIED_FRONTIER_UPPER_BOUND',
                'optimality_scope': 'normal-service objective only; declared snapshot model',
                'scope': 'Known-curve full-window dynamic branch search; not rolling control'}

