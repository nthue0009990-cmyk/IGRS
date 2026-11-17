# Isolated NC123 dimension adapter. Source and exact replacements: artifacts/nature123_igrs_migration_20261107/derivation.json
"""Common staged-only verification window for fixed candidate scorers.

Construction pays the original INITIAL/WAIT. It substitutes only the first
graph pool generation, never the physical validator or the separate Beam guard.
"""
from copy import deepcopy
import time

from .nature123_igrs_deferred import DeferredCertificationSearch
from .nature123_igrs_deferred_control import PendingVerificationAdapter, DeferredMacroController, VERIFY, YIELD
from .nature34_clpu_environment import CLPUReliabilityError
from .nature123_igrs_staged_pickup import generate_staged_pickup_candidates, VERSION as GENERATOR_VERSION, ORIGIN

VERSION = 'clpu-staged-fixed-verification-controller-v1'
ADAPTER_VERSION = 'clpu-staged-whole-load-pending-raw-v1'
VERIFY_LIMIT, MAX_OPERATIONS = 16, 17


class StagedCertificationSearch(DeferredCertificationSearch):
    def __init__(self, env, oracle, config=None, record_callback=None):
        self.stage_generation = None
        self.stage_generation_seconds = 0.
        self.stage_branch_metadata = {}
        super().__init__(env, oracle, config, record_callback)

    def _expand_prefix(self, prefix, parent_id):
        if prefix or parent_id is not None or self.stage_generation is not None:
            raise RuntimeError('Staged pool is generated once after paid bootstrap')
        before = (self._points(), self._actual(self.oracle), getattr(self.oracle, 'requests', None))
        began = time.perf_counter()
        self.stage_generation = generate_staged_pickup_candidates(self.env)
        for row in self.stage_generation['candidates']:
            branch_id = self._next_id
            self._add(row['prefix'], row['actions'], ORIGIN, None)
            if self._next_id != branch_id + 1:
                raise CLPUReliabilityError('Unique staged candidate could not enter the fixed pool')
            self.stage_branch_metadata[branch_id] = deepcopy(row)
        self.stage_generation_seconds = time.perf_counter() - began
        if before != (self._points(), self._actual(self.oracle), getattr(self.oracle, 'requests', None)):
            raise CLPUReliabilityError('Graph-only staged generation invoked physical certification')

    def _holds_and_neighbors(self, node, *, failed_step):
        # Full-plan validation still records reliable feedback through _evaluate.
        # This bounded experiment compares one fixed common pool, not a further
        # outcome-dependent generation mechanism.
        return None

    def _refresh_stop(self):
        super()._refresh_stop()
        if not self.done and not self.verification_queue:
            self._stop('STAGED_VERIFICATION_POOL_EXHAUSTED')

    def expand(self, *args, **kwargs):
        raise TypeError('The staged pool is fixed; no macro graph expansion is available')

    def result(self):
        value = super().result()
        value.update(version=VERSION, method='staged_complete_verification',
            stage_generation=deepcopy(self.stage_generation),
            stage_generation_seconds=self.stage_generation_seconds,
            stage_branch_metadata=deepcopy(self.stage_branch_metadata),
            stage_generation_status='GENERATED' if self.stage_generation is not None else 'BOOTSTRAP_STOP_NOT_GENERATED',
            bootstrap_graph_generation='fixed_staged_pool_replaces_original_initial_prefix_expansion',
            reliable_failure_generation='none_fixed_pool_feedback_still_recorded')
        return value


class StagedVerificationAdapter(PendingVerificationAdapter):
    def __init__(self, search):
        if not isinstance(search, StagedCertificationSearch):
            raise TypeError('A staged certification engine is required')
        super().__init__(search)

    def _window(self):
        rows = sorted(self._frontier.values(), key=lambda row: row['id'])
        if len(rows) > 24 or self.config.action_window != 32:
            raise CLPUReliabilityError('Frozen staged pool/window capacity exceeded')
        if self.engine.verify_attempts % 2 == 1 and self.incumbent is not None:
            energy = self.incumbent['validation']['energy_kwh']
            eligible = [row for row in rows if row['info']['energy_kwh'] > energy
                        and self.engine.stage_branch_metadata[row['id']]['held_out_loads']]
            if eligible:
                return eligible
        return rows

    def window_mapping(self):
        mapping = super().window_mapping()
        for row in mapping:
            metadata = self.engine.stage_branch_metadata[row['branch_id']]
            row.update(candidate_id=metadata['candidate_id'],
                held_out_loads=deepcopy(metadata['held_out_loads']),
                timing=metadata['timing'], route_rank=metadata['route_rank'])
        return mapping

    def metadata(self):
        value = super().metadata()
        value.update(version=ADAPTER_VERSION,
            window_rule={'ordinary': 'all_pending_staged_unique_plans_in_branch_id_order',
                'coverage_turn': 'completed_verify_attempts_mod_2_equals_1',
                'eligible': 'still_held_out_and_graph_energy_strictly_above_certified_incumbent',
                'no_eligible_fallback': 'all_pending_staged_unique_plans_in_branch_id_order',
                'capacity': 32, 'max_pool_size': 24},
            representative_order=['one_unique_full_plan_per_generated_candidate'],
            new_origin=ORIGIN, prefix_wait_feature_for_staged=0,
            distribution_shift='staged_only_full_plan_pool_and_ready_bit_restore_timing')
        return value


class StagedMacroController(DeferredMacroController):
    def __init__(self, search, *, method='beam', policy=None, generator=None):
        if method not in ('beam', 'bc_raw', 'ppo_raw'):
            raise ValueError('Unknown fixed ranker')
        if search.total_decisions or search.graph_expansions or search.verify_attempts:
            raise ValueError('Staged control must start immediately after bootstrap')
        self.search, self.adapter = search, StagedVerificationAdapter(search)
        if method == 'beam':
            if policy is not None:
                raise ValueError('Rule arm must not carry a learned policy')
        else:
            self.adapter.validate_policy(policy)
        self.schedule, self.method, self.policy, self.generator = 'staged', method, policy, generator
        self.cursor = self.policy_decisions = 0
        self.skipped_slots, self.records = [], []
        self._pending = None
        self.failed = False
        self.finish_reason = search.stop_reason if search.done else None

    def prepare(self):
        if self.failed:
            raise RuntimeError('Controller failed; Beam success cannot hide an EARLY error')
        if self._pending is not None:
            if self._pending['state_before'] != self._state_token():
                raise CLPUReliabilityError('Engine changed after staged decision preparation')
            return deepcopy(self._pending)
        if self.search.done:
            self.finish_reason = self.search.stop_reason
            return None
        observation = self.adapter.observation('raw') if self.cursor < VERIFY_LIMIT else None
        operation = VERIFY if observation is not None and observation['mask'] else YIELD
        mapping = self.adapter.window_mapping() if operation == VERIFY else []
        choice = self._choose_verify(observation) if operation == VERIFY else {
            'action': None, 'logprob': None, 'value': None, 'method': 'fixed_handoff', 'policy_used': False}
        node = None
        if operation == VERIFY:
            action = choice['action']
            if type(action) is not int or not 0 <= action < len(observation['mask']) or not observation['mask'][action]:
                raise CLPUReliabilityError('Fixed scorer chose an unavailable staged slot')
            if observation['branch_ids'] != [row['branch_id'] for row in mapping]:
                raise CLPUReliabilityError('Staged observation and plan mapping disagree')
            node = self.search._frontier[mapping[action]['branch_id']]
        proposal = {'version': VERSION, 'schedule': 'staged', 'slot_index': self.cursor,
            'operation': operation, 'decision_id': self.search.total_decisions,
            'branch_id': node['id'] if node is not None else None,
            'actions': list(node['info']['actions']) if node is not None else None,
            'prefix': list(node['prefix']) if node is not None else None,
            'observation': observation, 'window_mapping': mapping,
            'state_before': self._state_token(), **choice}
        self._pending = deepcopy(proposal)
        return proposal

    def apply(self, proposal):
        if self.failed or self._pending is None:
            raise RuntimeError('A fresh prepared staged decision is required')
        if proposal != self._pending or proposal['state_before'] != self._state_token():
            raise CLPUReliabilityError('Changed or stale staged decision')
        try:
            if proposal['operation'] == VERIFY:
                response = self.search.verify(proposal['branch_id'])
                self.policy_decisions += int(proposal['policy_used'])
                self.cursor += 1
            elif proposal['operation'] == YIELD:
                response = self.search.yield_to_beam()
            else:
                raise CLPUReliabilityError('No graph expansion in staged macro control')
            if self.search.graph_expansions or self.search.verify_attempts > VERIFY_LIMIT or self.search.total_decisions > MAX_OPERATIONS:
                raise CLPUReliabilityError('Frozen staged operation budget exceeded')
        except BaseException:
            self.failed = True
            raise
        self.records.append({'decision_id': proposal['decision_id'], 'slot_index': proposal['slot_index'],
            'operation': proposal['operation'], 'branch_id': proposal['branch_id'],
            'method': proposal['method'], 'policy_used': proposal['policy_used'],
            'verification_status': response['verification_status']})
        self._pending = None
        if self.search.done:
            self.finish_reason = self.search.stop_reason
        return response

    def result(self):
        value = self.search.result()
        value.update(method='fixed_staged_candidate_certification', new_training=0,
            policy_decisions=self.policy_decisions, training_interactions=0, optimizer_updates=0)
        value['control'] = {'version': VERSION, 'engine_version': VERSION,
            'generator_version': GENERATOR_VERSION, 'schedule': 'staged',
            'requested_method': self.method, 'adapter': self.adapter.metadata(),
            'graph_operation_limit': 0, 'verification_choice_limit': VERIFY_LIMIT,
            'total_operation_limit_including_yield': MAX_OPERATIONS,
            'bootstrap_outside_macro_limits_but_paid': True,
            'verify_failure_repairs': 'none_fixed_candidate_pool_feedback_updates_unchanged',
            'dominated_skip_consumes_verification_choice': True,
            'schedule_slots_consumed': self.cursor, 'skipped_slots': [],
            'records': deepcopy(self.records), 'policy_decisions': self.policy_decisions,
            'controller_failed': self.failed, 'finish_reason': self.finish_reason,
            'guard_established': False, 'complete_guard_still_required': True}
        return value
