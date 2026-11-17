"""Protected Beam continuation with exact, history-bound failure evidence.

The frozen BranchSearch is never edited. Its complete Beam stage runs first.
This module imports no physical solver or learner; the caller owns the oracle.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import time

from .nature34_clpu_branch import BranchConfig, BranchSearch, _state_dict
from .nature34_clpu_environment import CLPUReliabilityError, DynamicState


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def _sha(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def _scope(env, start, physics_identity):
    return {'physics_identity': deepcopy(physics_identity),
            'model_version': env.model_version, 'static': deepcopy(env.static),
            'loads': deepcopy(env.loads), 'scenario': deepcopy(env.scenario),
            'condition': deepcopy(env.condition), 'event': deepcopy(env.event),
            'initial_state': _state_dict(start), 'environment_initial': _state_dict(env.initial),
            'horizon': env.horizon, 'slot_seconds': env.slot_seconds,
            'sample_seconds': env.sample_seconds, 'load_names': list(env.load_names)}


class _WitnessOracle:
    """Observe unchanged oracle replies so FAIL reliability is not inferred."""
    def __init__(self, oracle, env):
        self.delegate, self.env = oracle, env
        self.failures = {}

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def certify(self, state, seconds):
        certificate = self.delegate.certify(state, seconds)
        if certificate.get('status') == 'FAIL' and certificate.get('reliable') is True:
            identity = self.env.identity(state, seconds)
            key = (_sha(identity), certificate.get('point_id'))
            entry = {'identity': deepcopy(identity), 'certificate': deepcopy(certificate)}
            previous = self.failures.get(key)
            if previous is not None and previous != entry:
                raise CLPUReliabilityError('One physical identity has inconsistent FAIL certificates')
            self.failures[key] = entry
        return certificate


class _ContinuationSearch(BranchSearch):
    """Adopt the completed search state without reinitializing any resource."""
    def _window(self):
        owner = self._tail_owner
        if owner.phase == 'VERIFY':
            return owner._tail_candidates()[:1]
        eligible = {key: row for key, row in self._frontier.items()
                    if owner._match_failure(row['info']['actions']) is None}
        # Known-failing complete plans remain available as repair parents. They
        # are never deleted merely because their unchanged prefix is invalid.
        if not eligible:
            return super()._window()
        original = self._frontier
        try:
            self._frontier = eligible
            return super()._window()
        finally:
            self._frontier = original

    def _evaluate(self, info, origin, details):
        owner = self._tail_owner
        evidence = owner._match_failure(info['actions'])
        if evidence is None:
            return super()._evaluate(info, origin, details)
        # Reuse is not an independent complete-plan check and is not inserted
        # in _checked/attempts. Its separate paid logical record is mandatory.
        owner.counters['certificate_reuses'] += 1
        owner._step_reuse = evidence['certificate_id']
        return {'actions': list(info['actions']), 'origin': origin, 'details': deepcopy(details),
                'status': 'FAIL', 'certified_complete': False,
                'validation': {'status': 'FAIL', 'failed_step': evidence['failed_step'],
                               'feedback': deepcopy(evidence['feedback'])},
                'physical_points_before': self._points(), 'physical_points_after': self._points(),
                'prefix_certificate_id': evidence['certificate_id']}


class TailBranchSearch:
    """Exact original Beam, then bounded same-oracle continuation and VERIFY.

    BASELINE always uses the original Beam selector. CONTINUATION may use a
    rule or a fixed learner. VERIFY ignores learners and uses tail_priority.
    ``physics_identity`` must bind the caller's actual physics/numerical policy
    and frozen source/input versions; no certificate crosses this instance.
    """
    def __init__(self, env, oracle, config=None, *, physics_identity,
                 continuation=True, extra_branch_expansions=256,
                 record_callback=None, event_callback=None):
        if not isinstance(physics_identity, Mapping) or not physics_identity:
            raise ValueError('Nonempty frozen physics/numerical identity binding required')
        if type(extra_branch_expansions) is not int or not 0 <= extra_branch_expansions <= 256:
            raise ValueError('Additional expansion allowance must be in [0, 256]')
        if isinstance(config, Mapping):
            config = BranchConfig(**config)
        self.original_config = config or BranchConfig()
        if self.original_config.max_branch_expansions + extra_branch_expansions > 512:
            raise ValueError('Total expansion allowance exceeds 512')
        self.env = env
        self.physics_identity = deepcopy(dict(physics_identity))
        self.identity_binding = _scope(env, env.initial, self.physics_identity)
        self.identity_sha256 = _sha(self.identity_binding)
        self.event_callback = event_callback
        self.continuation = bool(continuation)
        self.extra_branch_expansions = extra_branch_expansions
        self.events, self.prefix_certificates = [], []
        self.counters = {'screen_calls': 0, 'screen_cache_hits': 0,
                         'screen_computations': 0, 'screen_matches': 0,
                         'screen_seconds': 0., 'certificate_reuses': 0,
                         'repair_only_expansions': 0, 'continuation_expansions': 0,
                         'tail_verifications': 0, 'identity_checks': 0,
                         'identity_check_seconds': 0., 'step_seconds': 0.}
        self._screen_cache = {}
        self._indexed_attempts = 0
        self._baseline = None
        self._baseline_sha256 = None
        self._baseline_energy = None
        self._step_reuse = None
        self.phase = 'BASELINE'
        self.oracle = _WitnessOracle(oracle, env)
        self.search = BranchSearch(env, self.oracle, self.original_config, record_callback)
        if self.search.done:
            self._finish_baseline()

    @property
    def done(self):
        return self.phase == 'DONE'

    @property
    def quality(self):
        return self.search.quality

    @property
    def baseline_result(self):
        return deepcopy(self._baseline['result']) if self._baseline is not None else None

    def _emit(self, event):
        record = {'tail_event_id': len(self.events), **deepcopy(event)}
        self.events.append(record)
        if self.event_callback is not None:
            self.event_callback(deepcopy(record))
        return record

    def _check_identity(self):
        started = time.perf_counter()
        current = _scope(self.env, self.search.start, self.physics_identity)
        self.counters['identity_checks'] += 1
        self.counters['identity_check_seconds'] += time.perf_counter() - started
        if _sha(current) != self.identity_sha256:
            raise CLPUReliabilityError('Frozen physical/history identity changed during protected search')

    def _assert_floor(self, *, full_checkpoint=False):
        if self._baseline is None or not self._baseline['guarantee_established']:
            return
        if full_checkpoint and _sha(self._baseline) != self._baseline_sha256:
            raise CLPUReliabilityError('Immutable Beam checkpoint changed')
        new = self.search.incumbent
        if new is None or new.get('certified_complete') is not True or new.get('status') != 'COMPLETE':
            raise CLPUReliabilityError('Complete Beam incumbent was lost')
        if new['validation']['energy_kwh'] < self._baseline_energy:
            raise CLPUReliabilityError('Protected result is worse than complete Beam baseline')

    def _finish_baseline(self):
        original = self.search.result()
        valid = (original['status'] == 'CERTIFIED_INCUMBENT'
                 and original['best_plan'].get('certified_complete') is True)
        self._baseline = {
            'result': deepcopy(original), 'identity_sha256': self.identity_sha256,
            'guarantee_established': valid,
            'oracle_counters': self._oracle_counters(),
            'replay_state': {
                'start': _state_dict(self.search.start),
                'frontier_branch_ids': sorted(self.search._frontier),
                'expanded_prefixes': [list(v) for v in sorted(self.search._expanded_prefixes)],
                'branch_keys': [[list(a), list(b)] for a, b in sorted(self.search._branch_keys)],
                'beam_turn': self.search._beam_turn, 'next_id': self.search._next_id,
                'note': 'Full candidates and checked plans are in result; graph states reconstruct from frozen environment without physical calls.',
            },
        }
        self._baseline_sha256 = _sha(self._baseline)
        self._baseline_energy = original['best_plan']['validation']['energy_kwh'] if valid else None
        can_continue = (self.continuation and valid
                        and self.search.stop_reason == 'BRANCH_EXPANSION_CAP'
                        and self.search._points() < self.original_config.point_budget
                        and len(self.search._checked) < self.original_config.max_checked_candidates)
        self._emit({'kind': 'BASELINE_SEALED', 'stage': 'BASELINE',
                    'baseline_sha256': self._baseline_sha256,
                    'guarantee_established': valid, 'continued': can_continue,
                    'oracle_counters': self._baseline['oracle_counters']})
        if not can_continue:
            self.phase = 'DONE'
            return
        continuation = _ContinuationSearch.__new__(_ContinuationSearch)
        continuation.__dict__ = self.search.__dict__.copy()
        continuation._tail_owner = self
        continuation.config = replace(self.original_config,
                                      max_branch_expansions=self.search._beam_turn + self.extra_branch_expansions)
        continuation.done, continuation.stop_reason = False, None
        self.search = continuation
        self.phase = 'CONTINUATION' if self.extra_branch_expansions else 'VERIFY'
        self._index_failures()
        self._refresh_phase()

    def _oracle_counters(self):
        return {'points': self.search._points(),
                'requests': getattr(self.oracle, 'requests', None),
                'cache_hits': getattr(self.oracle, 'cache_hits', getattr(self.oracle, 'hits', None))}

    def baseline_checkpoint(self):
        if self._baseline is None:
            return None
        self._assert_floor(full_checkpoint=True)
        return {**deepcopy(self._baseline), 'sha256': self._baseline_sha256}

    def _index_failures(self):
        while self._indexed_attempts < len(self.search.attempts):
            index = self._indexed_attempts
            attempt = self.search.attempts[index]
            self._indexed_attempts += 1
            if attempt.get('status') != 'FAIL':
                continue
            validation = attempt.get('validation', {})
            failed_step = validation.get('failed_step')
            samples = validation.get('samples', [])
            if not isinstance(failed_step, int) or failed_step < 0 or not samples:
                raise CLPUReliabilityError('FAIL plan lacks exact failed-prefix evidence')
            sample = samples[-1]
            if sample.get('status') != 'FAIL':
                raise CLPUReliabilityError('FAIL plan lacks failed terminal sample')
            state = DynamicState.from_dict(sample['state'])
            identity = self.env.identity(state, sample['time_seconds'])
            witness = self.oracle.failures.get((_sha(identity), sample['point_id']))
            if witness is None or witness['certificate'].get('reliable') is not True:
                raise CLPUReliabilityError('FAIL prefix lacks an actually observed reliable certificate')
            prefix = attempt['actions'][:failed_step + 1]
            reached = self.search.start
            for action in prefix:
                reached = self.env.advance(reached, action)
            if self.env.identity(reached, sample['time_seconds']) != identity:
                raise CLPUReliabilityError('Failed prefix and actual physical identity disagree')
            key = (tuple(prefix), _sha(identity))
            if any((tuple(row['actions_prefix']), _sha(row['identity'])) == key
                   for row in self.prefix_certificates):
                continue
            evidence = {'certificate_id': len(self.prefix_certificates),
                        'attempt_index': index, 'failed_step': failed_step,
                        'actions_prefix': list(prefix), 'point_id': sample['point_id'],
                        'identity': deepcopy(identity), 'scope_sha256': self.identity_sha256,
                        'certificate': deepcopy(witness['certificate']),
                        'feedback': deepcopy(validation.get('feedback', []))}
            self.prefix_certificates.append(evidence)
            self._emit({'kind': 'PREFIX_CERTIFICATE_INDEXED', 'stage': self.phase, **evidence})

    def _match_failure(self, actions):
        started = time.perf_counter()
        self.counters['screen_calls'] += 1
        key = (tuple(actions), len(self.prefix_certificates))
        if key in self._screen_cache:
            self.counters['screen_cache_hits'] += 1
            result = self._screen_cache[key]
        else:
            self.counters['screen_computations'] += 1
            result = None
            for evidence in self.prefix_certificates:
                prefix = evidence['actions_prefix']
                if list(actions[:len(prefix)]) != prefix:
                    continue
                state = self.search.start
                for action in prefix:
                    state = self.env.advance(state, action)
                if self.env.identity(state, evidence['identity']['time_seconds']) == evidence['identity']:
                    result = evidence
                    break
            self._screen_cache[key] = result
        if result is not None:
            self.counters['screen_matches'] += 1
        self.counters['screen_seconds'] += time.perf_counter() - started
        return result

    def _tail_candidates(self):
        seen, rows = set(self.search._checked), []
        ordered = sorted(self.search._frontier.values(), key=lambda row: (row['info']['priority'], row['id']))
        for row in ordered:
            actions = tuple(row['info']['actions'])
            if actions in seen:
                continue
            seen.add(actions)
            if self.search._incumbent_key is not None and row['info']['priority'] >= self.search._incumbent_key:
                continue
            if self._match_failure(actions) is None:
                rows.append(row)
        return rows

    def _refresh_phase(self):
        if self.phase == 'DONE':
            return
        if self.phase == 'CONTINUATION' and self.search.done:
            if self.search.stop_reason == 'BRANCH_EXPANSION_CAP':
                self.phase = 'VERIFY'
                self.search.done, self.search.stop_reason = False, None
            else:
                self.phase = 'DONE'
        if self.phase == 'VERIFY':
            if self.search._points() >= self.original_config.point_budget:
                self.search._stop('PHYSICAL_BUDGET_EXHAUSTED')
            elif len(self.search._checked) >= self.original_config.max_checked_candidates:
                self.search._stop('LOGICAL_CHECK_CAP')
            elif not self._tail_candidates():
                self.search._stop('TAIL_VERIFICATION_EXHAUSTED')
            if self.search.done:
                self.phase = 'DONE'
        self._assert_floor()

    def observation(self, representation='raw'):
        self._check_identity()
        return self.search.observation(representation)

    def rule_action(self, method='beam'):
        self._check_identity()
        if self.phase == 'VERIFY':
            if method != 'tail_priority':
                raise ValueError('VERIFY uses tail_priority only')
            if not self._tail_candidates():
                raise RuntimeError('No tail candidate remains')
            return 0
        if self.phase == 'BASELINE' and method != 'beam':
            raise ValueError('Baseline must follow the unchanged original Beam')
        return self.search.rule_action(method)

    def step(self, branch_index):
        if self.done:
            raise RuntimeError('Protected search is already done')
        self._check_identity()
        started = time.perf_counter()
        stage = self.phase
        before = {'physical_points': self.search._points(), 'checked_candidates': len(self.search._checked),
                  'expansions': self.search._beam_turn, 'generated': len(self.search._branch_keys),
                  'graph_assessed': len(self.search._assessed)}
        self._step_reuse = None
        if stage == 'BASELINE':
            if branch_index != self.search.rule_action('beam'):
                raise ValueError('Caller attempted to change the protected original Beam selection')
            event = self.search.step(branch_index)
            core_done = event['done']
            if self.search.done:
                self._finish_baseline()
        elif stage == 'CONTINUATION':
            node = self.search._window()[branch_index]
            evidence = self._match_failure(node['info']['actions'])
            event = self.search.step(branch_index)
            core_done = event['done']
            self.counters['continuation_expansions'] += 1
            if evidence is not None:
                self.counters['repair_only_expansions'] += 1
                self._step_reuse = evidence['certificate_id']
            self._index_failures()
            self._refresh_phase()
        else:
            if branch_index != 0:
                raise ValueError('Tail priority has only deterministic index zero')
            rows = self._tail_candidates()
            if not rows:
                raise RuntimeError('No tail candidate remains')
            node = rows[0]
            quality_before = self.quality
            record = self.search._evaluate(node['info'], 'tail_verify',
                                           {'branch_id': node['id'], 'stage': 'VERIFY',
                                            'expansion_performed': False})
            self.counters['tail_verifications'] += 1
            event = {'branch_id': node['id'], 'actions': list(node['info']['actions']),
                     'query_reason': 'tail_verify', 'validation_status': record['status'],
                     'quality_before': quality_before, 'quality_after': self.quality,
                     'physical_points_before': before['physical_points'],
                     'physical_points_after': self.search._points(),
                     'physical_points_used': self.search._points() - before['physical_points'],
                     'new_branches': 0, 'done': self.search.done,
                     'stop_reason': self.search.stop_reason}
            event['reward'] = (event['quality_after'] - event['quality_before']
                               - self.original_config.reward_cost_weight * event['physical_points_used']
                               / self.original_config.reward_cost_normalizer)
            event['info'] = {'validation_status': event['validation_status'],
                             'physical_points_used': event['physical_points_used'],
                             'query_reason': 'tail_verify', 'stop_reason': self.search.stop_reason}
            core_done = self.search.done
            self._index_failures()
            self._refresh_phase()
        self._assert_floor()
        elapsed = time.perf_counter() - started
        self.counters['step_seconds'] += elapsed
        after = {'physical_points': self.search._points(), 'checked_candidates': len(self.search._checked),
                 'expansions': self.search._beam_turn, 'generated': len(self.search._branch_keys),
                 'graph_assessed': len(self.search._assessed)}
        operation = 'VERIFY' if stage == 'VERIFY' else 'REPAIR_ONLY' if self._step_reuse is not None else 'EXPAND'
        self._emit({'kind': 'SEARCH_STEP', 'stage': stage, 'operation': operation,
                    'branch_id': event['branch_id'], 'prefix_certificate_id': self._step_reuse,
                    'before': before, 'after': after, 'duration_seconds': elapsed,
                    'core_event': deepcopy(event), 'phase_after': self.phase,
                    'counters': deepcopy(self.counters)})
        return {**deepcopy(event), 'stage': stage, 'operation': operation,
                'core_done': core_done, 'done': self.done, 'phase_after': self.phase}

    def result(self):
        self._check_identity()
        self._assert_floor(full_checkpoint=True)
        result = self.search.result()
        result['tail'] = {'version': 'protected-beam-tail-v1', 'phase': self.phase,
                          'baseline_result': self.baseline_result,
                          'baseline_sha256': self._baseline_sha256,
                          'guarantee_established': self._baseline is not None and self._baseline['guarantee_established'],
                          'identity_binding': deepcopy(self.identity_binding),
                          'identity_sha256': self.identity_sha256,
                          'prefix_certificates': deepcopy(self.prefix_certificates),
                          'events': deepcopy(self.events), 'counters': deepcopy(self.counters),
                          'original_config': asdict(self.original_config),
                          'additional_expansion_allowance': self.extra_branch_expansions,
                          'accounting': 'Original checks include WAIT/FAIL/partial; certified-prefix reuse is separate; all expansions, screening and time charged. No cross-episode cache.'}
        return result
