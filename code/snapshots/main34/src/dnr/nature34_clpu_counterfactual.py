"""History-exact teacher snapshots and isolated bounded counterfactual forks.

No physical runtime or learner is imported here. Source recovery uses only a
sealed point prefix. A caller-supplied context manager is opened lazily for a
fork's first genuinely new physical identity. New answers never enter a shared
snapshot or a sibling fork.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import time

from .nature34_clpu_branch import BranchConfig, BranchSearch
from .nature34_clpu_environment import CLPUBudgetExhausted, CLPUReliabilityError


VERSION = 'clpu-counterfactual-prefix-v1'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha_value(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def _json_copy(value):
    return json.loads(canonical(value))


class SnapshotUnavailable(ValueError):
    """A predeclared snapshot position is absent, terminal, or uninformative."""
    def __init__(self, code, **details):
        self.code, self.details = code, details
        super().__init__(code + ': ' + canonical(details))


def snapshot_prefix_point_count(source_result, completed_steps):
    if type(completed_steps) is not int or completed_steps not in (0, 8):
        raise ValueError('Only the predeclared 0/8 completed-action snapshots are admitted')
    trace = source_result.get('trace', [])
    if completed_steps > len(trace):
        raise SnapshotUnavailable('SOURCE_POSITION_ABSENT', completed_steps=completed_steps,
                                  available_steps=len(trace))
    if completed_steps:
        if trace[completed_steps - 1]['done']:
            raise SnapshotUnavailable('SOURCE_POSITION_TERMINAL', completed_steps=completed_steps,
                                      stop_reason=trace[completed_steps - 1].get('stop_reason'))
        return int(trace[completed_steps - 1]['physical_points_after'])
    attempts = source_result.get('attempts', [])
    if not attempts or attempts[0].get('origin') != 'wait_incumbent':
        raise SnapshotUnavailable('NO_COMPLETE_BOOTSTRAP', source_status=source_result.get('status'))
    if attempts[0].get('status') != 'COMPLETE' or not attempts[0].get('certified_complete'):
        raise SnapshotUnavailable('NO_COMPLETE_BOOTSTRAP', bootstrap_status=attempts[0].get('status'))
    return int(attempts[0]['physical_points_after'])


def _scope(env, physics_identity):
    if not isinstance(physics_identity, Mapping) or not physics_identity:
        raise ValueError('Frozen physics identity is required')
    for key in ('alpha', 'base_condition_sha256'):
        if key not in physics_identity:
            raise ValueError('Missing oracle identity binding: ' + key)
    return {'physics_identity': _json_copy(physics_identity),
            'model_version': env.model_version, 'static': _json_copy(env.static),
            'loads': _json_copy(env.loads), 'scenario': _json_copy(env.scenario),
            'condition': _json_copy(env.condition), 'event': _json_copy(env.event),
            'initial_state': env.initial.to_dict(), 'load_names': list(env.load_names),
            'horizon': env.horizon, 'slot_seconds': env.slot_seconds,
            'sample_seconds': env.sample_seconds}


def _identity(env, state, seconds, binding):
    return {**env.identity(state, seconds), 'alpha': binding['alpha'],
            'base_condition_sha256': binding['base_condition_sha256']}


def _compact_point(row):
    if 'certificate' in row and 'identity' in row:
        compact = _json_copy(row)
    else:
        identity = row['state_identity']
        compact = {'identity': _json_copy(identity),
                   'identity_sha256': row['identity_sha256'],
                   'certificate': {'point_id': row['point_id'], 'status': row['status'],
                                   'reliable': row['reliable'],
                                   'feedback': _json_copy(row['structured_feedback']['violations'])},
                   'source_row_sha256': sha_value(row)}
    if sha_value(compact['identity']) != compact['identity_sha256']:
        raise CLPUReliabilityError('Source point identity SHA mismatch')
    certificate = compact['certificate']
    if certificate.get('status') not in ('PASS', 'FAIL') or certificate.get('reliable') is not True:
        raise CLPUReliabilityError('Unknown or unreliable source point cannot restore a snapshot')
    return compact


class _PrefixOracle:
    def __init__(self, env, points, budget, binding):
        self.env, self.budget, self.binding = env, budget, deepcopy(binding)
        self._source = points
        self._source_by_key = {row['identity_sha256']: row for row in points}
        if len(self._source_by_key) != len(points):
            raise CLPUReliabilityError('Duplicate physical identities in a source point prefix')
        if [row['certificate']['point_id'] for row in points] != list(range(len(points))):
            raise CLPUReliabilityError('Source point prefix must have contiguous original IDs starting at zero')
        self.cache = {}
        self.points = self.requests = self.hits = 0
        self.mode = 'REPLAY'
        self._base_points = None
        self._post_requests = self._post_hits = 0
        self._factory = self._stack = self._live = None
        self._max_new = None
        self.initialization_attempted = False
        self.point_id_mappings = []
        self.new_cache = {}

    def activate(self, factory, stack, max_new):
        if self.mode != 'REPLAY' or self.points != len(self._source):
            raise CLPUReliabilityError('Cannot fork before exact prefix recovery has finished')
        self.mode, self._base_points = 'FORK', self.points
        self._factory, self._stack, self._max_new = factory, stack, max_new

    def certify(self, state, seconds):
        identity = _identity(self.env, state, seconds, self.binding)
        key = sha_value(identity)
        self.requests += 1
        if self.mode == 'FORK':
            self._post_requests += 1
        if key in self.cache:
            self.hits += 1
            if self.mode == 'FORK':
                self._post_hits += 1
            return deepcopy(self.cache[key])
        if self.mode == 'REPLAY':
            row = self._source_by_key.get(key)
            if row is None:
                raise CLPUReliabilityError('Replay requested an identity outside its historical point prefix')
            if row['identity'] != identity or row['certificate']['point_id'] != self.points:
                raise CLPUReliabilityError('Replay changed physical first-use order or identity')
            self.points += 1
            result = deepcopy(row['certificate'])
        else:
            used = self.points - self._base_points
            if self.points >= self.budget or used >= self._max_new:
                raise CLPUBudgetExhausted('Counterfactual new-point allowance exhausted')
            if self._live is None:
                self.initialization_attempted = True
                self._live = self._stack.enter_context(self._factory(budget=self._max_new))
                if getattr(self._live, 'points', None) != 0:
                    raise CLPUReliabilityError('A fork live oracle must start empty; sibling/future answers forbidden')
                if getattr(self._live, 'cache', {}):
                    raise CLPUReliabilityError('A fork live oracle arrived with a nonempty physical cache')
                if getattr(self._live, 'budget', None) != self._max_new:
                    raise CLPUReliabilityError('Live oracle budget differs from the admitted fork allowance')
                if not hasattr(self._live, 'env') or _scope(self._live.env, self.binding) != _scope(self.env, self.binding):
                    raise CLPUReliabilityError('Live oracle environment differs from the frozen snapshot')
            before = self._live.points
            try:
                local = self._live.certify(state, seconds)
            finally:
                self.points = self._base_points + self._live.points
            if self._live.points != before + 1 or local.get('point_id') != before:
                raise CLPUReliabilityError('New fork identity did not produce exactly one new local physical point')
            if local.get('status') not in ('PASS', 'FAIL') or local.get('reliable') is not True:
                raise CLPUReliabilityError('Unknown or unreliable counterfactual physical result')
            result = deepcopy(local)
            result['point_id'] = self._base_points + local['point_id']
            self.point_id_mappings.append({'identity_sha256': key,
                                           'local_point_id': local['point_id'],
                                           'cumulative_point_id': result['point_id']})
            self.new_cache[key] = {'identity': _json_copy(identity), 'certificate': deepcopy(result),
                                   'local_point_id': local['point_id']}
        self.cache[key] = deepcopy(result)
        return result

    def summary(self):
        new_points = self.points - self._base_points if self.mode == 'FORK' else 0
        actual_entries = None
        if self._live is not None and hasattr(self._live, 'factory'):
            stats = self._live.factory.stats()
            actual_entries = stats.get('returned', {}).get('numerical_solution_entries')
        return {'cumulative_points': self.points, 'cumulative_point_budget': self.budget,
                'cumulative_requests': self.requests, 'cumulative_cache_hits': self.hits,
                'historical_prefix_points': self._base_points if self.mode == 'FORK' else self.points,
                'new_physical_points': new_points, 'new_requests_including_cache': self._post_requests,
                'new_cache_hits': self._post_hits, 'live_initialization_occurred': self._live is not None,
                'live_initialization_attempted': self.initialization_attempted,
                'live_requests': getattr(self._live, 'requests', 0),
                'new_numerical_entries_from_runtime': actual_entries,
                'new_numerical_entries_expected_for_snapshot_oracle': 3 + 4 * new_points if self._live is not None else 0,
                'historical_restore_new_physical_points': 0,
                'point_id_mappings': deepcopy(self.point_id_mappings)}


def _state_digest(search):
    # Save graph-state fingerprints, not a huge duplicate of every candidate's
    # full trajectory. Deterministic replay reconstructs those exact trajectories.
    structure = {'result': search.result(),
                 'frontier_ids': sorted(search._frontier),
                 'expanded_prefixes': sorted(search._expanded_prefixes),
                 'branch_keys': sorted(search._branch_keys),
                 'next_id': search._next_id, 'beam_turn': search._beam_turn,
                 'checked_plans': sorted(search._checked),
                 'assessed_plans': sorted(search._assessed)}
    return sha_value(structure)


class RestoredSnapshot:
    def __init__(self, env, search, payload):
        self.env, self.search = env, search
        self._payload_bytes = canonical(payload).encode('utf-8')
        self.sha256 = hashlib.sha256(self._payload_bytes).hexdigest()

    def to_dict(self):
        return {'snapshot_sha256': self.sha256, 'payload': json.loads(self._payload_bytes)}


def _replay(env, payload, *, check_state=True):
    if sha_value(_scope(env, payload['physics_identity'])) != payload['scope_sha256']:
        raise CLPUReliabilityError('Snapshot environment/physics/history scope mismatch')
    config = BranchConfig(**payload['config'])
    oracle = _PrefixOracle(env, payload['prefix_points'], config.point_budget, payload['physics_identity'])
    search = BranchSearch(env, oracle, config)
    if canonical(search.attempts[0] if search.attempts else None) != canonical(payload['source_bootstrap_attempt']):
        raise CLPUReliabilityError('Source WAIT bootstrap differs during zero-solve replay')
    for expected in payload['source_trace']:
        if search.done:
            raise CLPUReliabilityError('Replay ended before the recorded source position')
        observation = search.observation('raw')
        index = expected['selected_index']
        if observation['branch_ids'] != expected['window_branch_ids']:
            raise CLPUReliabilityError('Replay source candidate window differs')
        if search.rule_action(payload['source_method']) != index:
            raise CLPUReliabilityError('Recorded source selection differs from its declared fixed rule')
        actual = search.step(index)
        if canonical(actual) != canonical(expected):
            raise CLPUReliabilityError('Replay source response/quality/budget/generation differs')
    if oracle.points != len(payload['prefix_points']):
        raise CLPUReliabilityError('Supplied source prefix contains unobserved future points')
    if search.done:
        raise SnapshotUnavailable('RESTORED_POSITION_TERMINAL', stop_reason=search.stop_reason)
    if len(search.observation('raw')['branch_ids']) < 2:
        raise SnapshotUnavailable('FEWER_THAN_TWO_CANDIDATES')
    if check_state:
        if _state_digest(search) != payload['state_sha256']:
            raise CLPUReliabilityError('Reconstructed search state differs from the sealed snapshot')
        if canonical(search.observation('raw')) != canonical(payload['raw_observation']):
            raise CLPUReliabilityError('Reconstructed raw observation differs from the sealed snapshot')
        counters = {'points': oracle.points, 'requests': oracle.requests, 'cache_hits': oracle.hits}
        if counters != payload['prefix_oracle_counters']:
            raise CLPUReliabilityError('Reconstructed historical oracle counters differ')
    return search


def restore_snapshot(env, source_result, prefix_points, completed_steps, physics_identity,
                     expected_observation=None, source_binding=None):
    required = snapshot_prefix_point_count(source_result, completed_steps)
    config = BranchConfig(**source_result['config'])
    if config.point_budget != 256:
        raise ValueError('Route B source must use the original 256-point budget')
    method = source_result.get('spec', {}).get('method', 'beam')
    if method not in ('beam', 'greedy'):
        raise ValueError('Only frozen Beam/Greedy teacher sources are admitted')
    points = [_compact_point(row) for row in prefix_points]
    if len(points) != required:
        raise CLPUReliabilityError('Source points must contain exactly the requested historical prefix')
    payload = {'version': VERSION, 'config': asdict(config), 'completed_steps': completed_steps,
               'source_method': method, 'source_binding': deepcopy(source_binding or {}),
               'source_trace': _json_copy(source_result['trace'][:completed_steps]),
               'source_bootstrap_attempt': _json_copy(source_result['attempts'][0]),
               'prefix_points': points, 'physics_identity': _json_copy(physics_identity),
               'scope_sha256': sha_value(_scope(env, physics_identity))}
    search = _replay(env, payload, check_state=False)
    observation = search.observation('raw')
    if expected_observation is not None and canonical(observation) != canonical(expected_observation):
        raise CLPUReliabilityError('Recovered observation differs from the original recorded observation')
    if completed_steps < len(source_result['trace']):
        if observation['branch_ids'] != source_result['trace'][completed_steps]['window_branch_ids']:
            raise CLPUReliabilityError('Recovered next action window differs from the source')
    payload['raw_observation'], payload['state_sha256'] = _json_copy(observation), _state_digest(search)
    payload['prefix_oracle_counters'] = {'points': search.oracle.points, 'requests': search.oracle.requests,
                                         'cache_hits': search.oracle.hits}
    return RestoredSnapshot(env, search, payload)


def snapshot_from_dict(env, value):
    payload = _json_copy(value['payload'])
    if payload.get('version') != VERSION or sha_value(payload) != value['snapshot_sha256']:
        raise CLPUReliabilityError('Snapshot version or immutable payload SHA mismatch')
    search = _replay(env, payload)
    return RestoredSnapshot(env, search, payload)


def fork_snapshot(snapshot, forced_branch_id, live_oracle_factory, *, max_new_points=128,
                  max_expansions=16, transition_callback=None, record_callback=None):
    if type(max_new_points) is not int or not 0 <= max_new_points <= 128:
        raise ValueError('Fork new-point allowance must be in [0,128]')
    if type(max_expansions) is not int or not 1 <= max_expansions <= 16:
        raise ValueError('Fork expansion allowance must include its forced step and be in [1,16]')
    started = time.perf_counter()
    restored = snapshot_from_dict(snapshot.env, snapshot.to_dict())
    search, oracle = restored.search, restored.search.oracle
    initial_observation = search.observation('raw')
    if type(forced_branch_id) is not int or forced_branch_id not in initial_observation['branch_ids']:
        raise ValueError('Forced candidate is not in the saved common window')
    base_quality = search.quality
    if search.incumbent is None or not search.incumbent.get('certified_complete'):
        raise CLPUReliabilityError('A fork must start from a complete certified incumbent')
    start_trace, start_attempts = len(search.trace), len(search.attempts)
    start_generated, start_assessed = len(search._branch_keys), len(search._assessed)
    allowance = min(max_new_points, search.config.point_budget - oracle.points)
    expansions = min(max_expansions, search.config.max_branch_expansions - search._beam_turn)
    if expansions <= 0:
        raise SnapshotUnavailable('NO_ORIGINAL_EXPANSION_REMAINDER')
    transitions = []
    g1 = None
    search.record_callback = (lambda row: record_callback(deepcopy(row))) if record_callback else None
    with ExitStack() as stack:
        oracle.activate(live_oracle_factory, stack, allowance)
        for index in range(expansions):
            if search.done:
                break
            observation = search.observation('raw')
            action = observation['branch_ids'].index(forced_branch_id) if index == 0 else search.rule_action('beam')
            row = {'transition_id': index, 'status': 'STARTED', 'forced': index == 0,
                   'action': action, 'branch_id': observation['branch_ids'][action],
                   'observation_sha256': sha_value(observation),
                   'oracle_before': {'points': oracle.points, 'requests': oracle.requests, 'hits': oracle.hits}}
            if transition_callback:
                transition_callback(deepcopy(row))
            response = search.step(action)
            if search.quality < base_quality:
                raise CLPUReliabilityError('A fork lost its certified starting incumbent')
            if index == 0:
                g1 = search.quality - base_quality
            row.update(status='COMPLETED', response=deepcopy(response),
                       obs_raw=deepcopy(observation),
                       oracle_after={'points': oracle.points, 'requests': oracle.requests, 'hits': oracle.hits})
            transitions.append(row)
            if transition_callback:
                transition_callback(deepcopy(row))
        oracle_before_close = oracle.summary()
    # The context manager closes all actual physics and writes its own ledger.
    oracle_summary = oracle.summary()
    if oracle_summary['cumulative_points'] != oracle_before_close['cumulative_points']:
        raise CLPUReliabilityError('Closing the physical context changed its point ledger')
    if g1 is None:
        raise CLPUReliabilityError('No forced first action was completed; no label may be invented')
    if oracle_summary['new_physical_points'] > allowance or len(transitions) > expansions:
        raise CLPUReliabilityError('Fork exceeded its admitted local allowance')
    final_quality = search.quality
    stop = search.stop_reason if search.done else 'FORK_EXPANSION_LIMIT'
    if (search.stop_reason == 'PHYSICAL_BUDGET_EXHAUSTED'
            and oracle_summary['cumulative_points'] < search.config.point_budget):
        stop = 'FORK_NEW_PHYSICAL_POINT_LIMIT'
    return {'version': VERSION, 'status': 'BOUNDED_COUNTERFACTUAL_COMPLETE',
            'snapshot_sha256': snapshot.sha256, 'forced_branch_id': forced_branch_id,
            'source_completed_steps': restored.to_dict()['payload']['completed_steps'],
            'initial_quality': base_quality, 'first_step_quality': base_quality + g1,
            'final_quality': final_quality, 'g1': g1, 'gH': final_quality - base_quality,
            'g1_pp': 100 * g1, 'gH_pp': 100 * (final_quality - base_quality),
            'stop_reason': stop, 'core_stop_reason': search.stop_reason,
            'new_point_allowance': allowance, 'expansion_allowance_including_forced': expansions,
            'completed_expansions': len(transitions), 'cumulative_expansions': search._beam_turn,
            'new_generated_candidates': len(search._branch_keys) - start_generated,
            'new_graph_assessed_candidates': len(search._assessed) - start_assessed,
            'cumulative_generated_candidates': len(search._branch_keys),
            'cumulative_graph_assessed_candidates': len(search._assessed),
            'cumulative_checked_candidates': len(search._checked),
            'fork_trace': deepcopy(search.trace[start_trace:]), 'transitions': transitions,
            'new_attempts': deepcopy(search.attempts[start_attempts:]),
            'best_plan': deepcopy(search.incumbent), 'oracle': oracle_summary,
            'elapsed_seconds_including_restore_and_cold_initialization': time.perf_counter() - started,
            'labels_scope': 'Certified incumbent gains after one forced search selection and bounded fixed-Beam continuation, not optimal Q or exclusive physical-path causal gain.',
            'unmeasured_candidates': 'UNKNOWN_NOT_FAILURE', 'new_training': 0}
