"""History-exact depth-family allocation with unchanged physical mechanics.

The first four expansions use one full-frontier hold depth, then the current
coverage-window Beam resumes. Missing families fall back explicitly. There is
no frontier deletion, changed physical rule, or shared future answer.
"""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
import math
import time

from .nature34_clpu_branch import BranchConfig
from .nature34_clpu_counterfactual import (
    SnapshotUnavailable, RestoredSnapshot, _PrefixOracle, _compact_point,
    _json_copy, _scope, _state_digest, canonical, sha_value,
)
from .nature34_clpu_environment import CLPUReliabilityError
from .nature34_clpu_window import WindowBranchSearch, WINDOW_RULE_VERSION
from .nature34_clpu_window_counterfactual import (
    _completed_prefix, snapshot_prefix_point_count, quality_curve_metric,
)

VERSION = 'clpu-family-counterfactual-prefix-v1'
FAMILY_RULE_VERSION = 'depth_hold_allocation_four_expansions_v1'
MACRO_EXPANSIONS = 4
DEPTH_ACTIONS = (None, 1, 2, 3, 4, 5, 6)


def _check_depth(depth):
    if depth is not None and (type(depth) is not int or depth not in range(1, 7)):
        raise ValueError('Requested family depth must be None or an integer 1..6')


def family_rows(search, depth):
    """FIFO unique improving, unchecked complete WAIT plans at this depth."""
    _check_depth(depth)
    if depth is None or search.incumbent is None:
        return []
    energy = float(search.incumbent['validation']['energy_kwh'])
    selected, plans = [], set()
    for row in sorted(search._frontier.values(), key=lambda row: row['id']):
        actions = row['info']['actions']
        if (row['origin'] == 'prefix_wait' and len(row['prefix']) == depth
                and actions not in search._checked and actions not in plans
                and row['info']['energy_kwh'] > energy):
            selected.append(row)
            plans.add(actions)
            if len(selected) == search.config.action_window:
                break
    return selected


class FamilyWindowSearch(WindowBranchSearch):
    """Only exposure changes; Beam, physics, generation and stop rules inherit."""

    def _window(self):
        rows = family_rows(self, getattr(self, 'active_depth', None))
        return rows if rows else super()._window()


def describe_action(search, depth):
    """Use current history only and restore the transient exposure mode."""
    _check_depth(depth)
    if not isinstance(search, FamilyWindowSearch):
        raise TypeError('Family actions require FamilyWindowSearch exposure semantics')
    if search.done or search.incumbent is None:
        raise SnapshotUnavailable('TERMINAL_OR_NO_CERTIFIED_INCUMBENT')
    baseline = WindowBranchSearch._window(search)
    had_depth = 'active_depth' in search.__dict__
    previous = getattr(search, 'active_depth', None)
    try:
        search.active_depth = depth
        rows = search._window()
        index = search.rule_action('beam')
        chosen = rows[index]
    finally:
        if had_depth:
            search.active_depth = previous
        else:
            del search.active_depth
    family_active = depth is not None and bool(family_rows(search, depth))
    return {'action_id': 0 if depth is None else depth, 'requested_depth': depth,
        'window_branch_ids': [row['id'] for row in rows],
        'selected_branch_id': chosen['id'], 'selected_index': index,
        'selected_actions': list(chosen['info']['actions']),
        'effective_mode': 'depth_family' if family_active else 'current_window',
        'fallback': depth is not None and not family_active,
        'selected_plan_in_a0_window': any(row['info']['actions'] == chosen['info']['actions']
                                         for row in baseline)}


def describe_actions(search):
    rows = [describe_action(search, depth) for depth in DEPTH_ACTIONS]
    for row in rows:
        row['same_selected_plan_as_a0'] = row['selected_actions'] == rows[0]['selected_actions']
    return rows


def state_eligibility(search):
    """A depth-family selection differs from A0, even at a singleton window.

    Window membership is reported separately from effective selection. All
    seven actions remain admitted, including explicit fallback actions.
    """
    if search.done or search.incumbent is None:
        return {'eligible': False, 'completed_steps': search._beam_turn, 'actions': [],
                'distinct_alternative_depths': [], 'reason': 'TERMINAL_OR_NO_CERTIFIED_INCUMBENT'}
    rows = describe_actions(search)
    different = [row['requested_depth'] for row in rows[1:] if not row['same_selected_plan_as_a0']]
    return {'eligible': bool(different), 'completed_steps': search._beam_turn, 'actions': rows,
            'distinct_alternative_depths': different,
            'reason': None if different else 'NO_DIFFERENT_FAMILY_SELECTION'}


def _replay(env, payload, *, check_state=True):
    if sha_value(_scope(env, payload['physics_identity'])) != payload['scope_sha256']:
        raise CLPUReliabilityError('Snapshot environment/physics/history scope mismatch')
    if (payload.get('window_rule_version') != WINDOW_RULE_VERSION
            or payload.get('family_rule_version') != FAMILY_RULE_VERSION
            or payload.get('source_window_variant') != 'coverage_v1'
            or payload.get('source_method') != 'beam'):
        raise CLPUReliabilityError('Snapshot window/family or source rule changed')
    config = BranchConfig(**payload['config'])
    oracle = _PrefixOracle(env, payload['prefix_points'], config.point_budget, payload['physics_identity'])
    search = FamilyWindowSearch(env, oracle, config)
    if canonical(search.attempts[0] if search.attempts else None) != canonical(payload['source_bootstrap_attempt']):
        raise CLPUReliabilityError('Source WAIT bootstrap differs during zero-solve replay')
    if len(payload['source_trace']) != payload['completed_steps']:
        raise CLPUReliabilityError('Snapshot action prefix length changed')
    for expected in payload['source_trace']:
        if search.done:
            raise CLPUReliabilityError('Replay ended before the recorded source position')
        observation = search.observation('raw')
        if observation['branch_ids'] != expected['window_branch_ids']:
            raise CLPUReliabilityError('Replay source candidate window differs')
        index = expected['selected_index']
        if search.rule_action('beam') != index:
            raise CLPUReliabilityError('Recorded source selection differs from coverage-window Beam')
        if canonical(search.step(index)) != canonical(expected):
            raise CLPUReliabilityError('Replay source response/quality/budget/generation differs')
    if oracle.points != len(payload['prefix_points']):
        raise CLPUReliabilityError('Supplied source prefix contains unobserved future points')
    if search.done:
        raise SnapshotUnavailable('RESTORED_POSITION_TERMINAL', stop_reason=search.stop_reason)
    if not search.observation('raw')['branch_ids']:
        raise SnapshotUnavailable('NO_CURRENT_CANDIDATES')
    if check_state:
        if canonical(search.attempts) != canonical(payload['source_attempts']):
            raise CLPUReliabilityError('Restored source attempted-plan prefix differs')
        if _state_digest(search) != payload['state_sha256']:
            raise CLPUReliabilityError('Reconstructed search state differs from the sealed snapshot')
        if canonical(search.observation('raw')) != canonical(payload['raw_observation']):
            raise CLPUReliabilityError('Reconstructed raw observation differs from the sealed snapshot')
        counters = {'points': oracle.points, 'requests': oracle.requests, 'cache_hits': oracle.hits}
        if counters != payload['prefix_oracle_counters']:
            raise CLPUReliabilityError('Reconstructed historical oracle counters differ')
    return search


def restore_snapshot(env, source_result, source_steps, source_plans, prefix_points,
                     completed_steps, physics_identity, *, expected_observation=None,
                     source_binding=None):
    required = snapshot_prefix_point_count(source_result, source_steps, source_plans, completed_steps)
    config = BranchConfig(**source_result['config'])
    if config.point_budget != 256:
        raise ValueError('Family source must use the original 256-point budget')
    points = [_compact_point(row) for row in prefix_points]
    if len(points) != required:
        raise CLPUReliabilityError('Source points must contain exactly the requested historical prefix')
    payload = {'version': VERSION, 'window_rule_version': WINDOW_RULE_VERSION,
        'family_rule_version': FAMILY_RULE_VERSION, 'config': asdict(config),
        'completed_steps': completed_steps, 'source_method': 'beam', 'source_window_variant': 'coverage_v1',
        'source_binding': deepcopy(source_binding or {}),
        'source_trace': _completed_prefix(source_steps, completed_steps),
        'source_bootstrap_attempt': _json_copy(source_plans[0]),
        'prefix_points': points, 'physics_identity': _json_copy(physics_identity),
        'scope_sha256': sha_value(_scope(env, physics_identity))}
    search = _replay(env, payload, check_state=False)
    if canonical(search.attempts) != canonical(source_plans[:len(search.attempts)]):
        raise CLPUReliabilityError('Source attempted-plan history differs during replay')
    observation = search.observation('raw')
    if expected_observation is not None and canonical(observation) != canonical(expected_observation):
        raise CLPUReliabilityError('Recovered observation differs from the original recorded observation')
    payload['source_attempts'] = _json_copy(search.attempts)
    payload['raw_observation'], payload['state_sha256'] = _json_copy(observation), _state_digest(search)
    payload['prefix_oracle_counters'] = {'points': search.oracle.points, 'requests': search.oracle.requests,
                                       'cache_hits': search.oracle.hits}
    return RestoredSnapshot(env, search, payload)


def snapshot_from_dict(env, value):
    payload = _json_copy(value['payload'])
    if payload.get('version') != VERSION or sha_value(payload) != value['snapshot_sha256']:
        raise CLPUReliabilityError('Snapshot version or immutable payload SHA mismatch')
    return RestoredSnapshot(env, _replay(env, payload), payload)


def fork_snapshot(snapshot, forced_depth, live_oracle_factory, *, max_new_points=128,
                  max_expansions=16, transition_callback=None, record_callback=None):
    _check_depth(forced_depth)
    if type(max_new_points) is not int or not 0 <= max_new_points <= 128:
        raise ValueError('Fork new-point allowance must be in [0,128]')
    if type(max_expansions) is not int or not 1 <= max_expansions <= 16:
        raise ValueError('Fork expansion allowance must be in [1,16] including family allocation')
    started = time.perf_counter()
    restored = snapshot_from_dict(snapshot.env, snapshot.to_dict())
    search, oracle = restored.search, restored.search.oracle
    initial_action = describe_action(search, forced_depth)
    if search.incumbent is None or not search.incumbent.get('certified_complete'):
        raise CLPUReliabilityError('A fork requires a complete certified starting incumbent')
    initial_quality, base_points = search.quality, oracle.points
    start_trace, start_attempts = len(search.trace), len(search.attempts)
    allowance = min(max_new_points, search.config.point_budget - base_points)
    expansions = min(max_expansions, search.config.max_branch_expansions - search._beam_turn)
    if expansions <= 0:
        raise SnapshotUnavailable('NO_ORIGINAL_EXPANSION_REMAINDER')
    transitions, certificates = [], []
    trajectory = [{'new_physical_points': 0, 'quality': initial_quality,
                   'kind': 'INITIAL_CERTIFIED_INCUMBENT'}]
    running_quality, active_expansion = initial_quality, 0

    def on_plan(record):
        nonlocal running_quality
        if record.get('status') == 'COMPLETE' and record.get('certified_complete') is True:
            quality = float(record['validation']['energy_kwh']) / search.denominator
            if not math.isfinite(quality) or not 0 <= quality <= 1:
                raise CLPUReliabilityError('Invalid complete certified objective')
            point = record['physical_points_after'] - base_points
            before = running_quality
            running_quality = max(running_quality, quality)
            certificates.append({'expansion_index': active_expansion,
                'new_physical_points': point, 'cumulative_physical_points': record['physical_points_after'],
                'quality_before': before, 'quality_after': running_quality,
                'plan_quality': quality, 'actions_sha256': sha_value(record['actions'])})
            trajectory.append({'new_physical_points': point, 'quality': running_quality,
                               'kind': 'COMPLETE_CERTIFICATE'})
        if record_callback:
            record_callback(deepcopy(record))

    search.record_callback = on_plan
    first_quality = None
    with ExitStack() as stack:
        oracle.activate(live_oracle_factory, stack, allowance)
        for index in range(expansions):
            if search.done:
                break
            active_expansion = index
            requested_depth = forced_depth if index < MACRO_EXPANSIONS else None
            search.active_depth = requested_depth
            decision = describe_action(search, requested_depth)
            observation = search.observation('raw')
            action = search.rule_action('beam')
            if decision['window_branch_ids'] != observation['branch_ids'] or decision['selected_index'] != action:
                raise CLPUReliabilityError('Family decision changed before its paid expansion')
            row = {'transition_id': index, 'status': 'STARTED',
                'family_allocation_phase': index < MACRO_EXPANSIONS,
                'requested_depth': requested_depth, 'effective_mode': decision['effective_mode'],
                'fallback': decision['fallback'], 'family_decision': decision,
                'action': action, 'branch_id': observation['branch_ids'][action],
                'observation_sha256': sha_value(observation),
                'oracle_before': {'points': oracle.points, 'requests': oracle.requests, 'hits': oracle.hits}}
            if transition_callback:
                transition_callback(deepcopy(row))
            response = search.step(action)
            if not math.isclose(search.quality, running_quality, rel_tol=0, abs_tol=1e-12):
                raise CLPUReliabilityError('Complete-plan curve differs from certified incumbent')
            if index == 0:
                first_quality = search.quality
            row.update(status='COMPLETED', response=deepcopy(response), obs_raw=deepcopy(observation),
                       oracle_after={'points': oracle.points, 'requests': oracle.requests, 'hits': oracle.hits})
            transitions.append(row)
            if transition_callback:
                transition_callback(deepcopy(row))
        before_close = oracle.summary()
    summary = oracle.summary()
    if summary['cumulative_points'] != before_close['cumulative_points']:
        raise CLPUReliabilityError('Closing physical runtime changed the point ledger')
    if first_quality is None or summary['new_physical_points'] > allowance:
        raise CLPUReliabilityError('Missing family step or exceeded admitted point allowance')
    stop = search.stop_reason if search.done else 'FORK_EXPANSION_LIMIT'
    if stop == 'PHYSICAL_BUDGET_EXHAUSTED' and oracle.points < search.config.point_budget:
        stop = 'FORK_NEW_PHYSICAL_POINT_LIMIT'
    return {'version': VERSION, 'window_rule_version': WINDOW_RULE_VERSION,
        'family_rule_version': FAMILY_RULE_VERSION, 'macro_expansions': MACRO_EXPANSIONS,
        'status': 'BOUNDED_COUNTERFACTUAL_COMPLETE', 'snapshot_sha256': snapshot.sha256,
        'forced_depth': forced_depth, 'initial_action_description': initial_action,
        'source_completed_steps': restored.to_dict()['payload']['completed_steps'],
        'initial_quality': initial_quality, 'first_step_quality': first_quality, 'final_quality': search.quality,
        'g1': first_quality - initial_quality, 'gH': search.quality - initial_quality,
        'g1_pp': 100 * (first_quality - initial_quality), 'gH_pp': 100 * (search.quality - initial_quality),
        'auc_gain_pp': quality_curve_metric(initial_quality, trajectory, allowance),
        'curve_budget_grid': 'integer_new_physical_points_1_through_allowance_inclusive',
        'certificate_updates': certificates, 'quality_point_trajectory': trajectory,
        'new_point_allowance': allowance, 'expansion_allowance_including_family': expansions,
        'completed_expansions': len(transitions), 'cumulative_expansions': search._beam_turn,
        'stop_reason': stop, 'core_stop_reason': search.stop_reason, 'oracle': summary,
        'fork_trace': deepcopy(search.trace[start_trace:]), 'transitions': transitions,
        'new_attempts': deepcopy(search.attempts[start_attempts:]), 'best_plan': deepcopy(search.incumbent),
        'elapsed_seconds_including_restore_and_cold_initialization': time.perf_counter() - started,
        'labels_scope': 'Measured complete-certified quality versus actual additional physical points; four-step depth-family allocation followed by WindowBeam, not optimal Q.',
        'unmeasured_candidates': 'UNKNOWN_NOT_FAILURE', 'new_training': 0}
