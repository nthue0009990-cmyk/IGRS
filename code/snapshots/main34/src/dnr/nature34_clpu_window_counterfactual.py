"""Exact historical replay and paid counterfactuals for the coverage window.

This independent version consumes the compact window-pilot journals.  It does
not change the original engine, its observation, or the original 0/8 teacher.
Only the historical physical prefix enters a snapshot.  Forks independently
pay for new identities and use the same coverage-window Beam continuation.
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


VERSION = 'clpu-window-counterfactual-prefix-v1'


def _completed_prefix(source_steps, completed_steps):
    if type(completed_steps) is not int or completed_steps < 0:
        raise ValueError('Nonnegative integer completed_steps required')
    selected, pending = [], None
    if completed_steps == 0:
        return selected
    for row in source_steps:
        if row.get('status') == 'STARTED':
            if pending is not None:
                raise CLPUReliabilityError('Source started another step before completing its predecessor')
            pending = row
        elif row.get('status') == 'COMPLETED':
            if pending is None:
                raise CLPUReliabilityError('Source completion has no corresponding STARTED boundary')
            for key in ('transition_id', 'method', 'action', 'branch_id', 'window_branch_ids', 'oracle_before'):
                if row.get(key) != pending.get(key):
                    raise CLPUReliabilityError('Source step boundary differs: ' + key)
            response = row['response']
            if (row['transition_id'] != len(selected) or row['method'] != 'beam'
                    or row['action'] != response['selected_index']
                    or row['branch_id'] != response['branch_id']
                    or row['window_branch_ids'] != response['window_branch_ids']
                    or row['oracle_before']['points'] != response['physical_points_before']
                    or row['oracle_after']['points'] != response['physical_points_after']):
                raise CLPUReliabilityError('Source step metadata disagrees with its response')
            selected.append(_json_copy(response))
            pending = None
            if len(selected) == completed_steps:
                return selected
        else:
            raise CLPUReliabilityError('Unknown source step boundary status')
    raise SnapshotUnavailable('SOURCE_POSITION_ABSENT', completed_steps=completed_steps,
                              available_steps=len(selected))


def snapshot_prefix_point_count(source_result, source_steps, source_plans, completed_steps):
    """Number of paid source identities visible at an arbitrary action boundary."""
    trace = _completed_prefix(source_steps, completed_steps)
    if source_result.get('spec', {}).get('method') != 'beam':
        raise ValueError('Only the frozen Beam source is admitted')
    if source_result.get('spec', {}).get('window_variant') != 'coverage_v1':
        raise ValueError('Only the coverage_v1 source window is admitted')
    if not source_plans or source_plans[0].get('origin') != 'wait_incumbent':
        raise SnapshotUnavailable('NO_COMPLETE_BOOTSTRAP')
    bootstrap = source_plans[0]
    if bootstrap.get('status') != 'COMPLETE' or bootstrap.get('certified_complete') is not True:
        raise SnapshotUnavailable('NO_COMPLETE_BOOTSTRAP')
    if trace and trace[-1]['done']:
        raise SnapshotUnavailable('SOURCE_POSITION_TERMINAL', completed_steps=completed_steps,
                                  stop_reason=trace[-1].get('stop_reason'))
    count = (trace[-1] if trace else bootstrap)['physical_points_after']
    if type(count) is not int or count < 0:
        raise CLPUReliabilityError('Invalid source physical prefix count')
    return count


def state_eligibility(search):
    """Outcome-blind first-state predicate for the parent population builder.

    A coverage turn means the actual special window was active, not merely an
    odd turn whose lack of eligible holds caused the original-window fallback.
    Counts are full-plan unique, unchecked, graph-improving visible candidates.
    They are not claimed electrically feasible.
    """
    if search.done or search.incumbent is None:
        return {'eligible': False, 'mode': None, 'completed_steps': search._beam_turn,
                'candidate_ids': [], 'reason': 'TERMINAL_OR_NO_CERTIFIED_INCUMBENT'}
    energy = float(search.incumbent['validation']['energy_kwh'])
    has_hold = any(row['origin'] == 'prefix_wait' and row['info']['actions'] not in search._checked
                   and row['info']['energy_kwh'] > energy for row in search._frontier.values())
    mode = 'coverage' if search._beam_turn % 2 and has_hold else 'normal'
    selected, plans = [], set()
    for row in search._window():
        actions = row['info']['actions']
        if actions not in plans and actions not in search._checked and row['info']['energy_kwh'] > energy:
            selected.append(row['id'])
            plans.add(actions)
    return {'eligible': len(selected) >= 2, 'mode': mode, 'completed_steps': search._beam_turn,
            'candidate_ids': selected, 'reason': None if len(selected) >= 2 else 'FEWER_THAN_TWO_UNTESTED_IMPROVING_PLANS'}


def _replay(env, payload, *, check_state=True):
    if sha_value(_scope(env, payload['physics_identity'])) != payload['scope_sha256']:
        raise CLPUReliabilityError('Snapshot environment/physics/history scope mismatch')
    if (payload.get('window_rule_version') != WINDOW_RULE_VERSION
            or payload.get('source_window_variant') != 'coverage_v1'
            or payload.get('source_method') != 'beam'):
        raise CLPUReliabilityError('Snapshot window or source rule changed')
    config = BranchConfig(**payload['config'])
    oracle = _PrefixOracle(env, payload['prefix_points'], config.point_budget, payload['physics_identity'])
    search = WindowBranchSearch(env, oracle, config)
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
    if len(search.observation('raw')['branch_ids']) < 2:
        raise SnapshotUnavailable('FEWER_THAN_TWO_CANDIDATES')
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
        raise ValueError('Window teacher source must use the original 256-point budget')
    points = [_compact_point(row) for row in prefix_points]
    if len(points) != required:
        raise CLPUReliabilityError('Source points must contain exactly the requested historical prefix')
    payload = {'version': VERSION, 'window_rule_version': WINDOW_RULE_VERSION,
               'config': asdict(config), 'completed_steps': completed_steps,
               'source_method': 'beam', 'source_window_variant': 'coverage_v1',
               'source_binding': deepcopy(source_binding or {}),
               'source_trace': _completed_prefix(source_steps, completed_steps),
               'source_bootstrap_attempt': _json_copy(source_plans[0]),
               'prefix_points': points, 'physics_identity': _json_copy(physics_identity),
               'scope_sha256': sha_value(_scope(env, physics_identity))}
    search = _replay(env, payload, check_state=False)
    # Future plans at the same point count must not enter the snapshot.  Replay
    # determines the exact consumed prefix length, including cache-only plans.
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


def quality_curve_metric(initial_quality, trajectory, allowance):
    """Discrete mean gain on b=1..B, with certificates credited at completion.

    A b=0 improvement is available at every positive budget; a b=B improvement
    contributes exactly one term.  Finishing early holds the certified value
    through B.  B=0 has no positive-budget observations and returns None.
    """
    if type(allowance) is not int or allowance < 0:
        raise ValueError('Nonnegative integer curve allowance required')
    if not math.isfinite(initial_quality) or not 0 <= initial_quality <= 1:
        raise ValueError('Finite certified initial quality in [0,1] required')
    previous_point, quality = 0, float(initial_quality)
    gain_sum = 0.
    if not trajectory or trajectory[0] != {'new_physical_points': 0, 'quality': initial_quality,
                                          'kind': 'INITIAL_CERTIFIED_INCUMBENT'}:
        raise ValueError('Curve must begin with its certified initial incumbent')
    for row in trajectory[1:]:
        point, value = row['new_physical_points'], row['quality']
        if (type(point) is not int or not previous_point <= point <= allowance
                or not math.isfinite(value) or not quality <= value <= 1
                or row.get('kind') != 'COMPLETE_CERTIFICATE'):
            raise ValueError('Invalid or uncertified quality-curve boundary')
        # The previous quality is in force at positive budgets before point.
        gain_sum += max(0, point - max(1, previous_point)) * (quality - initial_quality)
        previous_point, quality = point, float(value)
    gain_sum += max(0, allowance - max(1, previous_point) + 1) * (quality - initial_quality)
    return 100. * gain_sum / allowance if allowance else None


def fork_snapshot(snapshot, forced_branch_id, live_oracle_factory, *, max_new_points=128,
                  max_expansions=16, transition_callback=None, record_callback=None):
    if type(max_new_points) is not int or not 0 <= max_new_points <= 128:
        raise ValueError('Fork new-point allowance must be in [0,128]')
    if type(max_expansions) is not int or not 1 <= max_expansions <= 16:
        raise ValueError('Fork expansion allowance must be in [1,16] including its forced step')
    started = time.perf_counter()
    restored = snapshot_from_dict(snapshot.env, snapshot.to_dict())
    search, oracle = restored.search, restored.search.oracle
    observation = search.observation('raw')
    if type(forced_branch_id) is not int or forced_branch_id not in observation['branch_ids']:
        raise ValueError('Forced candidate is not in the saved common window')
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
            observation = search.observation('raw')
            action = (observation['branch_ids'].index(forced_branch_id)
                      if index == 0 else search.rule_action('beam'))
            row = {'transition_id': index, 'status': 'STARTED', 'forced': index == 0,
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
        raise CLPUReliabilityError('Missing forced step or exceeded admitted point allowance')
    stop = search.stop_reason if search.done else 'FORK_EXPANSION_LIMIT'
    if stop == 'PHYSICAL_BUDGET_EXHAUSTED' and oracle.points < search.config.point_budget:
        stop = 'FORK_NEW_PHYSICAL_POINT_LIMIT'
    return {'version': VERSION, 'window_rule_version': WINDOW_RULE_VERSION,
        'status': 'BOUNDED_COUNTERFACTUAL_COMPLETE', 'snapshot_sha256': snapshot.sha256,
        'forced_branch_id': forced_branch_id, 'source_completed_steps': restored.to_dict()['payload']['completed_steps'],
        'initial_quality': initial_quality, 'first_step_quality': first_quality, 'final_quality': search.quality,
        'g1': first_quality - initial_quality, 'gH': search.quality - initial_quality,
        'g1_pp': 100 * (first_quality - initial_quality), 'gH_pp': 100 * (search.quality - initial_quality),
        'auc_gain_pp': quality_curve_metric(initial_quality, trajectory, allowance),
        'curve_budget_grid': 'integer_new_physical_points_1_through_allowance_inclusive',
        'certificate_updates': certificates, 'quality_point_trajectory': trajectory,
        'new_point_allowance': allowance, 'expansion_allowance_including_forced': expansions,
        'completed_expansions': len(transitions), 'cumulative_expansions': search._beam_turn,
        'stop_reason': stop, 'core_stop_reason': search.stop_reason, 'oracle': summary,
        'fork_trace': deepcopy(search.trace[start_trace:]), 'transitions': transitions,
        'new_attempts': deepcopy(search.attempts[start_attempts:]), 'best_plan': deepcopy(search.incumbent),
        'elapsed_seconds_including_restore_and_cold_initialization': time.perf_counter() - started,
        'labels_scope': 'Measured complete-certified quality versus actual additional physical points; bounded WindowBeam continuation, not optimal Q.',
        'unmeasured_candidates': 'UNKNOWN_NOT_FAILURE', 'new_training': 0}
