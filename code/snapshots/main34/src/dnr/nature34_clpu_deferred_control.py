"""Fixed graph/verification schedules and a read-only legacy scorer adapter.

Old actors rank complete plans in the common pending window. They do not pick
EXPAND, VERIFY or YIELD. No solver, model loader, training, logging or complete
Beam guard is implemented here. The caller owns those lifecycle boundaries.
"""
from __future__ import annotations

from copy import deepcopy
import math

from .nature34_clpu_branch import BranchSearch
from .nature34_clpu_deferred import DeferredCertificationSearch, VERSION as ENGINE_VERSION
from .nature34_clpu_environment import CLPUReliabilityError
from .nature34_clpu_window import WINDOW_RULE, WindowBranchSearch


VERSION = 'clpu-deferred-fixed-macro-controller-v1'
ADAPTER_VERSION = 'clpu-pending-canonical-coverage-raw-adapter-v1'
GRAPH_LIMIT = VERIFY_LIMIT = 16
MAX_OPERATIONS = GRAPH_LIMIT + VERIFY_LIMIT + 1
EXPAND, VERIFY, YIELD = 'EXPAND_GRAPH', 'VERIFY_COMPLETE_PLAN', 'YIELD_TO_BEAM'
SCHEDULES = {
    'deferred': (EXPAND,) * GRAPH_LIMIT + (VERIFY,) * VERIFY_LIMIT,
    'interleaved': (EXPAND, VERIFY) * GRAPH_LIMIT,
}


def fixed_expand_choice(search, chosen_plans):
    """The sealed exposure diagnostic's graph-only rule, without target input."""
    if search.done or search.graph_expansions >= search.config.max_branch_expansions:
        return None
    rows = [node for node in search._frontier.values()
            if node['id'] not in search.expanded_nodes
            and node['info']['actions'] not in chosen_plans]
    return min(rows, key=lambda n: (len(n['prefix']), n['info']['priority'], n['id'])) if rows else None


def _canonical_key(node):
    return (len(node['prefix']), 0 if node['origin'] == 'prefix_wait' else 1, node['id'])


class PendingVerificationAdapter(WindowBranchSearch):
    """Read-only view; inherit the exact feature formulas and Beam scorer.

Only the pending complete-plan representatives and coverage turn differ from
the legacy view. All aliases remain in the engine for graph provenance and
future legal expansion. No frontier/counter/cache is overwritten temporarily.
"""
    def __init__(self, search):
        if not isinstance(search, DeferredCertificationSearch):
            raise TypeError('A deferred certification engine is required')
        if search.config.action_window != 32:
            raise ValueError('The frozen scorer adapter requires a 32-slot capacity')
        self.engine = search

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def _groups(self):
        groups = {}
        for node in self.engine._frontier.values():
            plan = node['info']['actions']
            if plan in self.engine.verification_queue and plan not in self.engine._checked:
                groups.setdefault(plan, []).append(node)
        return groups

    @property
    def _frontier(self):
        rows = [min(group, key=_canonical_key) for group in self._groups().values()]
        return {row['id']: row for row in sorted(rows, key=lambda row: row['id'])}

    @property
    def _beam_turn(self):
        return self.engine.verify_attempts

    def window_mapping(self):
        groups = self._groups()
        return [{'slot': slot, 'branch_id': node['id'], 'prefix': list(node['prefix']),
                 'origin': node['origin'], 'actions': list(node['info']['actions']),
                 'alias_branch_ids': sorted(alias['id'] for alias in groups[node['info']['actions']]),
                 'has_prefix_wait_alias': any(alias['origin'] == 'prefix_wait'
                                              for alias in groups[node['info']['actions']])}
                for slot, node in enumerate([] if self.done else self._window())]

    def metadata(self):
        return {
            'version': ADAPTER_VERSION, 'feature_source': 'BranchSearch.observation_unchanged',
            'rule_source': 'BranchSearch.rule_action_beam_unchanged',
            'window_rule': deepcopy(WINDOW_RULE),
            'coverage_turn_counter': 'completed_verify_attempts_including_dominated_skips',
            'representative_order': ['prefix_length_ascending', 'prefix_wait_before_other_origins', 'branch_id'],
            'eligibility': 'complete_plan_in_verification_queue_and_not_in_checked',
            'dominated_pending': 'visible_until_original_free_skip_consumes_one_VERIFY_choice',
            'unexposed_and_alias_nodes': 'retained_unchanged_in_engine',
            'candidate_raw_dim': 27 + self.horizon + 3 * len(self.env.load_names),
            'candidate_associated_dim': 35 + self.horizon + 3 * len(self.env.load_names),
            'global_dim': 11 + len(self.env.load_names), 'action_window_capacity': 32,
            'load_names_in_feature_order': list(self.env.load_names),
            'point_budget_feature': 'cumulative_per_view_logical_points_including_bootstrap',
            'model_scope': 'fixed_candidate_ranking_only_not_a_three_operation_policy',
            'distribution_shift': 'new_pending_only_window_and_graph_before_verification_schedule',
        }

    def validate_policy(self, policy):
        schema = self.metadata()
        expected = {'candidate_dim': schema['candidate_raw_dim'], 'global_dim': schema['global_dim'],
                    'representation': 'raw', 'hidden': 128}
        if getattr(policy, 'representation', None) != 'raw' or getattr(policy, 'architecture', None) != expected:
            raise ValueError('Fixed policy architecture does not match unchanged raw feature schema')

    def step(self, *args, **kwargs):
        raise TypeError('Read-only adapter: use the deferred engine operations')


class DeferredMacroController:
    """Prepare/journal/apply contract with at most 16 + 16 + 1 operations.

``prepare`` returns the next proposal or None. The caller writes STARTED and
checks its deadline before calling ``apply``. Apply does no IO and propagates
all exceptions. Only successful operations advance the controller. An error
poisons this controller so a caller cannot silently continue to the guard.

Unavailable schedule slots are retained in ``skipped_slots``. Interleaved
empty VERIFY slots do not prevent a later EXPAND from producing candidates.
Bootstrap is outside this operation budget and remains charged to the oracle.
"""
    def __init__(self, search, *, schedule='deferred', method='beam', policy=None, generator=None):
        if schedule not in SCHEDULES or method not in ('beam', 'bc_raw', 'ppo_raw'):
            raise ValueError('Unknown frozen schedule or fixed ranker')
        if schedule == 'interleaved' and method != 'beam':
            raise ValueError('Interleaved control arm uses the fixed Beam ranker')
        if search.total_decisions or search.graph_expansions or search.verify_attempts:
            raise ValueError('Macro control must start immediately after bootstrap')
        self.search = search
        self.adapter = PendingVerificationAdapter(search)
        if method == 'beam':
            if policy is not None:
                raise ValueError('Rule arm must not carry a learned policy')
        else:
            self.adapter.validate_policy(policy)
        self.schedule, self.method, self.policy, self.generator = schedule, method, policy, generator
        self.cursor = self.policy_decisions = 0
        self.chosen_expansion_plans = set()
        self.skipped_slots = []
        self.records = []
        self._pending = None
        self.failed = False
        self.finish_reason = search.stop_reason if search.done else None

    def _state_token(self):
        s = self.search
        return {'total_decisions': s.total_decisions, 'graph_expansions': s.graph_expansions,
                'verify_attempts': s.verify_attempts, 'logical_points': s._points(),
                'actual_physical_points': s._actual(s.oracle), 'requests': getattr(s.oracle, 'requests', None),
                'generated_branch_count': len(s._branch_keys), 'checked_plan_count': len(s._checked),
                'pending_complete_plans': len(s.verification_queue), 'done': s.done,
                'stop_reason': s.stop_reason}

    def _choose_verify(self, observation):
        if self.method == 'beam':
            return {'action': self.adapter.rule_action('beam'), 'logprob': None, 'value': None,
                    'method': 'beam', 'policy_used': False}
        choice = self.policy.act(observation, observation['mask'], deterministic=True, generator=self.generator)
        action = choice['action'].item() if hasattr(choice['action'], 'item') else choice['action']
        if not all(math.isfinite(float(choice[key])) for key in ('logprob', 'value')):
            raise CLPUReliabilityError('Nonfinite fixed candidate scorer output')
        return {'action': action, 'logprob': float(choice['logprob']), 'value': float(choice['value']),
                'method': self.method, 'policy_used': True}

    def prepare(self):
        if self.failed:
            raise RuntimeError('Controller failed; complete-guard success must not hide this error')
        if self._pending is not None:
            if self._pending['state_before'] != self._state_token():
                raise CLPUReliabilityError('Engine changed after decision preparation')
            return deepcopy(self._pending)
        if self.search.done:
            self.finish_reason = self.search.stop_reason
            return None
        while self.cursor < len(SCHEDULES[self.schedule]):
            operation = SCHEDULES[self.schedule][self.cursor]
            node = fixed_expand_choice(self.search, self.chosen_expansion_plans) if operation == EXPAND else None
            if operation == EXPAND and node is None:
                self.skipped_slots.append({'slot_index': self.cursor, 'operation': EXPAND,
                                           'reason': 'NO_ELIGIBLE_GRAPH_EXPANSION'})
                self.cursor += 1
                continue
            observation = self.adapter.observation('raw') if operation == VERIFY else None
            if operation == VERIFY and not observation['mask']:
                self.skipped_slots.append({'slot_index': self.cursor, 'operation': VERIFY,
                                           'reason': 'NO_PENDING_VERIFICATION_WINDOW'})
                self.cursor += 1
                continue
            break
        else:
            operation, node, observation = YIELD, None, None
        choice = self._choose_verify(observation) if operation == VERIFY else {
            'action': None, 'logprob': None, 'value': None,
            'method': 'fixed_graph_rule' if operation == EXPAND else 'fixed_handoff', 'policy_used': False}
        mapping = self.adapter.window_mapping() if operation == VERIFY else []
        if operation == VERIFY:
            action = choice['action']
            if type(action) is not int or not 0 <= action < len(observation['mask']) or not observation['mask'][action]:
                raise CLPUReliabilityError('Fixed scorer chose an unavailable verification slot')
            if observation['branch_ids'] != [row['branch_id'] for row in mapping]:
                raise CLPUReliabilityError('Verification observation and branch mapping disagree')
            node = self.search._frontier[mapping[action]['branch_id']]
        proposal = {'version': VERSION, 'schedule': self.schedule, 'slot_index': self.cursor,
            'operation': operation, 'decision_id': self.search.total_decisions,
            'branch_id': node['id'] if node is not None else None,
            'actions': list(node['info']['actions']) if node is not None else None,
            'prefix': list(node['prefix']) if node is not None else None,
            'observation': observation, 'window_mapping': mapping, 'state_before': self._state_token(), **choice}
        self._pending = deepcopy(proposal)
        return proposal

    def apply(self, proposal):
        if self.failed or self._pending is None:
            raise RuntimeError('A fresh prepared decision is required')
        if proposal != self._pending or self._pending['state_before'] != self._state_token():
            raise CLPUReliabilityError('Changed or stale deferred decision')
        operation = proposal['operation']
        try:
            if operation == EXPAND:
                response = self.search.expand(proposal['branch_id'])
                self.chosen_expansion_plans.add(tuple(proposal['actions']))
            elif operation == VERIFY:
                response = self.search.verify(proposal['branch_id'])
                self.policy_decisions += int(proposal['policy_used'])
            else:
                response = self.search.yield_to_beam()
            if (self.search.graph_expansions > GRAPH_LIMIT or self.search.verify_attempts > VERIFY_LIMIT
                    or self.search.total_decisions > MAX_OPERATIONS):
                raise CLPUReliabilityError('Fixed macro operation budget exceeded')
        except BaseException:
            self.failed = True
            raise
        self.records.append({'decision_id': proposal['decision_id'], 'slot_index': proposal['slot_index'],
            'operation': operation, 'branch_id': proposal['branch_id'], 'method': proposal['method'],
            'policy_used': proposal['policy_used'], 'verification_status': response['verification_status']})
        self._pending = None
        if operation != YIELD:
            self.cursor += 1
        if self.search.done:
            self.finish_reason = self.search.stop_reason
        return response

    def result(self):
        result = self.search.result()
        result.update(method='fixed_macro_deferred_certification', new_training=0,
                      policy_decisions=self.policy_decisions, training_interactions=0, optimizer_updates=0,
                      scope='Fixed candidate scorer migration under common graph/verification schedules; '
                            'EARLY lane only, complete original Beam guard still required')
        result['control'] = {
            'version': VERSION, 'engine_version': ENGINE_VERSION, 'schedule': self.schedule,
            'requested_method': self.method, 'adapter': self.adapter.metadata(),
            'graph_operation_limit': GRAPH_LIMIT, 'verification_choice_limit': VERIFY_LIMIT,
            'total_operation_limit_including_yield': MAX_OPERATIONS,
            'bootstrap_outside_macro_limits_but_paid': True,
            'verify_failure_repairs': 'original_reliable_failed_step_repairs_not_extra_macro_EXPAND',
            'dominated_skip_consumes_verification_choice': True,
            'schedule_slots_consumed': self.cursor, 'skipped_slots': deepcopy(self.skipped_slots),
            'records': deepcopy(self.records), 'policy_decisions': self.policy_decisions,
            'controller_failed': self.failed, 'finish_reason': self.finish_reason,
            'guard_established': False, 'complete_guard_still_required': True,
        }
        return result
