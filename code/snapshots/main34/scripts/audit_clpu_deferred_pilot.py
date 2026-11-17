"""Independent delayed-certification replay; no production search or solver.

The frozen shared-evidence audit remains authoritative for actual electrical
evidence and causal logical aliases. This module independently reconstructs
graph-only expansion, complete-plan verification, pending exposure and handoff.
"""
from __future__ import annotations

from copy import deepcopy
import argparse
from collections import Counter
import math
from pathlib import Path
import sys
import time
import traceback
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
sys.dont_write_bytecode = True

import audit_clpu_shared_pilot as shared

early, prior, cache, branch, window, tail, old = (shared.early, shared.prior, shared.cache,
    shared.branch, shared.window, shared.tail, shared.old)
require, close, same = shared.require, shared.close, shared.same
read, save, sha, digest, read_lines = shared.read, shared.save, shared.sha, shared.digest, shared.read_lines
PROTOTYPE_VERSION = 'clpu-deferred-complete-verification-prototype-v1'
CONTROL_VERSION = 'clpu-deferred-fixed-macro-controller-v1'
ADAPTER_VERSION = 'clpu-pending-canonical-coverage-raw-adapter-v1'
MAX_EXPAND = MAX_VERIFY = 16
STUDY = 'CLPU_DEFERRED_VERIFICATION_PILOT_V1'
SOURCE_RUN = '20261031T045725_491175Z_clpu_shared_rule_pilot'
SOURCE_SHA = '779f0f56987749761884539703a759a4ee3723c2a953d60018a2880b33bfeede'
MODELS_RUN, MODELS_SHA = shared.SOURCE_RUN, shared.SOURCE_SHA
SEEDS = (17, 41, 73)
SLOTS = ('interleaved_window', 'deferred_window', *(f'deferred_bc{s}' for s in SEEDS), *(f'deferred_ppo{s}' for s in SEEDS))


class DeferredReplay(branch.BranchReplay):
    """Reconstruct all plans and graph effects without production engine calls."""
    def __init__(self, mech, cfg, oracle, attempts, logical_rows):
        self.expanded_nodes, self.skipped, self.repairs = set(), set(), set()
        self.queue = {}
        self.graph_expansions = self.verify_attempts = self.total_decisions = 0
        self.handed_off = False
        self.deferred_trace = []
        self.logical_rows = logical_rows
        before = self.costs()
        super().__init__(mech, cfg, oracle, attempts)
        self.bootstrap = {'logical_points_before': before['logical_points'],
            'logical_points_after': oracle.used, 'actual_physical_points_before': before['physical_points'],
            'actual_physical_points_after': self.actual(), 'includes_initial_snapshot_and_complete_wait_attempt': True}

    def actual(self):
        return sum(r['reuse_kind'] == 'ACTUAL_PHYSICAL' for r in self.logical_rows[:self.oracle.used])

    def costs(self):
        if not hasattr(self, 'oracle'):
            return {'logical_points': 0, 'physical_points': 0, 'numerical_entries': 0, 'requests': 0}
        return {'logical_points': self.oracle.used, 'physical_points': self.actual(),
                'numerical_entries': 4 * self.actual(), 'requests': self.oracle.counts()['requests']}

    def add(self, prefix, actions, origin, parent):
        index = len(self.keys)
        super().add(prefix, actions, origin, parent)
        if len(self.keys) > index:
            plan = self.frontier[index]['info']['actions']
            if plan not in self.checked and plan not in self.skipped:
                self.queue.setdefault(plan, index)

    def refresh(self):
        if self.done:
            return
        if self.best and self.frontier and self.m.bound <= self.best['validation']['energy_kwh']:
            self.stop('CERTIFIED_FRONTIER_UPPER_BOUND')
        elif self.oracle.used >= self.cfg['point_budget']:
            self.stop('PHYSICAL_BUDGET_EXHAUSTED')
        elif len(self.checked) >= self.cfg['max_checked_candidates']:
            self.stop('LOGICAL_CHECK_CAP')
        elif not self.queue and (self.graph_expansions >= self.cfg['max_branch_expansions']
                or all(i in self.expanded_nodes for i in self.frontier)):
            self.stop('DEFERRED_FRONTIER_EXHAUSTED')

    def pending(self):
        rows = []
        for bid, node in sorted(self.frontier.items()):
            plan = node['info']['actions']
            status = 'SKIPPED_DOMINATED' if plan in self.skipped else self.checked.get(plan, {}).get('status')
            rows.append({'branch_id': bid, 'parent_id': node['parent_id'], 'prefix': list(node['prefix']),
                'actions': list(plan), 'origin': node['origin'], 'graph_energy_kwh': node['info']['energy_kwh'],
                'graph_expanded': bid in self.expanded_nodes, 'verification_status': status,
                'verified': status in ('COMPLETE', 'FAIL'), 'verification_pending': plan in self.queue,
                'can_expand': not self.done and bid not in self.expanded_nodes
                    and self.graph_expansions < self.cfg['max_branch_expansions'],
                'can_verify': not self.done and plan in self.queue})
        return rows

    def holds_and_repairs(self, node, failed):
        plan = node['info']['actions']
        for length in range(1, min(self.cfg['lookahead_depth'], self.m.horizon) + 1):
            self.add(plan[:length], plan[:length] + (0,) * (self.m.horizon - length), 'prefix_wait', node['id'])
        for neighbor, kind in self.neighbors(plan, failed):
            self.add(neighbor['actions'][:min(self.cfg['lookahead_depth'], self.m.horizon)],
                     neighbor['actions'], kind, node['id'])

    def operate(self, operation, branch_id):
        require(not self.done and operation in ('EXPAND_GRAPH', 'VERIFY_COMPLETE_PLAN', 'YIELD_TO_BEAM'),
                'Unknown deferred operation or operation after stop')
        node = None
        if operation != 'YIELD_TO_BEAM':
            require(type(branch_id) is int and branch_id in self.frontier, 'Unknown stable branch ID')
            node = self.frontier[branch_id]
        else:
            require(branch_id is None, 'Handoff cites a hidden branch')
        before, quality, generated = self.costs(), self.quality, len(self.keys)
        self.total_decisions += 1
        status = None
        if operation == 'EXPAND_GRAPH':
            require(branch_id not in self.expanded_nodes and self.graph_expansions < self.cfg['max_branch_expansions'],
                    'Repeated expansion or graph expansion cap exceeded')
            self.graph_expansions += 1
            self.turn = self.graph_expansions
            validation = self.checked.get(node['info']['actions'], {}).get('validation', {})
            failed = validation.get('failed_step') if validation.get('status') == 'FAIL' else None
            if not (type(failed) is int and failed < len(node['prefix'])):
                self.expand(node['prefix'], node['id'])
            self.holds_and_repairs(node, None)
            self.expanded_nodes.add(branch_id)
            require(self.costs() == before and self.quality == quality, 'Graph expansion consumed physical evidence')
            self.refresh()
        elif operation == 'VERIFY_COMPLETE_PLAN':
            plan = node['info']['actions']
            require(plan in self.queue, 'Complete plan verified twice or not pending')
            self.verify_attempts += 1
            if self.best and node['info']['priority'] >= self.m.plan(self.best['actions'])['priority']:
                del self.queue[plan]
                self.skipped.add(plan)
                status = 'SKIPPED_DOMINATED'
            else:
                record = self.evaluate(node['info'], node['origin'], {'branch_id': branch_id,
                    'parent_id': node['parent_id'], 'prefix': list(node['prefix']), 'operation': 'VERIFY_COMPLETE_PLAN'})
                status = record['status'] if record else None
                if status in ('COMPLETE', 'FAIL'):
                    del self.queue[plan]
                if status == 'FAIL' and plan not in self.repairs:
                    failed = record['validation']['failed_step']
                    require(type(failed) is int, 'Reliable failure missing its true first failed step')
                    self.holds_and_repairs(node, failed)
                    self.repairs.add(plan)
            self.refresh()
        else:
            self.handed_off = True
            self.stop('YIELDED_TO_BEAM')
        after = self.costs()
        event = {'decision_id': self.total_decisions - 1, 'operation': operation,
            'branch_id': branch_id, 'actions': list(node['info']['actions']) if node else None,
            'verification_status': status, 'quality_before': quality, 'quality_after': self.quality,
            'logical_points_before': before['logical_points'], 'logical_points_after': after['logical_points'],
            'actual_physical_points_before': before['physical_points'], 'actual_physical_points_after': after['physical_points'],
            'new_actual_physical_points': after['physical_points'] - before['physical_points'],
            'new_branches': len(self.keys) - generated, 'graph_expansions': self.graph_expansions,
            'verify_attempts': self.verify_attempts, 'total_decisions': self.total_decisions,
            'pending_complete_plans': len(self.queue), 'done': self.done, 'stop_reason': self.reason, 'error': None}
        self.deferred_trace.append(event)
        return event

    def verification_view(self):
        representatives = {}
        aliases = {}
        for node in self.frontier.values():
            plan = node['info']['actions']
            if plan not in self.queue or plan in self.checked:
                continue
            aliases.setdefault(plan, []).append(node['id'])
            key = lambda n: (len(n['prefix']), 0 if n['origin'] == 'prefix_wait' else 1, n['id'])
            if plan not in representatives or key(node) < key(representatives[plan]):
                representatives[plan] = node
        view = object.__new__(window.WindowReplay)
        view.__dict__ = dict(self.__dict__)
        view.frontier = {n['id']: n for n in representatives.values()}
        view.turn = self.verify_attempts
        return view, aliases

    def fixed_expansion(self, chosen_plans):
        if self.done or self.graph_expansions >= self.cfg['max_branch_expansions']:
            return None
        eligible = [n for n in self.frontier.values() if n['id'] not in self.expanded_nodes
                    and n['info']['actions'] not in chosen_plans]
        return min(eligible, key=lambda n: (len(n['prefix']), n['info']['priority'], n['id'])) if eligible else None

    def scientific_snapshot(self):
        result = tail.baseline_snapshot(self)
        result.update(version=PROTOTYPE_VERSION, method='deferred_complete_verification_prototype',
            logical_points=self.oracle.used, actual_physical_points=self.actual(), graph_expansions=self.graph_expansions,
            verify_attempts=self.verify_attempts, skipped_dominated_complete_plans=len(self.skipped),
            total_decisions=self.total_decisions, deferred_trace=self.deferred_trace, bootstrap_cost=self.bootstrap,
            handed_off=self.handed_off, guard_established=False, complete_guard_still_required=True,
            pending_complete_plans=len(self.queue), new_training=0,
            scope='Prototype EARLY lane only; no new exposure policy, reward, learner or complete guard execution')
        return result


def check_deferred_result(replay, result, controlled=False):
    expected = replay.scientific_snapshot()
    if controlled:
        expected['method'] = 'fixed_macro_deferred_certification'
        expected['scope'] = ('Fixed candidate scorer migration under common graph/verification schedules; '
                             'EARLY lane only, complete original Beam guard still required')
    actual = deepcopy(result)
    for row in actual['deferred_trace']:
        window.finite_time(row.pop('elapsed_seconds'), 'deferred operation elapsed time')
    window.finite_time(actual['bootstrap_cost'].pop('elapsed_seconds'), 'paid bootstrap elapsed time')
    for key, value in expected.items():
        same(actual[key], value, 'independent_deferred_result.' + key)
    require(replay.done and replay.oracle.index == len(replay.oracle.rows)
            and replay.oracle.used == len(replay.logical_rows) and replay.attempt_index == len(replay.attempts),
            'Unfinished deferred search or unconsumed evidence')
    return expected


class MacroReplay:
    """Independent fixed-slot scheduler; skips never invent spent operations."""
    def __init__(self, replay, schedule, method):
        require(schedule in ('deferred', 'interleaved') and method in ('beam', 'bc_raw', 'ppo_raw')
                and (schedule != 'interleaved' or method == 'beam'), 'Unapproved macro schedule/ranker')
        self.search, self.schedule, self.method = replay, schedule, method
        self.slots = (['EXPAND_GRAPH'] * 16 + ['VERIFY_COMPLETE_PLAN'] * 16 if schedule == 'deferred'
                      else ['EXPAND_GRAPH', 'VERIFY_COMPLETE_PLAN'] * 16)
        self.cursor = self.policy_decisions = 0
        self.chosen_plans, self.skipped_slots, self.records = set(), [], []

    def state(self):
        s = self.search
        return {'total_decisions': s.total_decisions, 'graph_expansions': s.graph_expansions,
            'verify_attempts': s.verify_attempts, 'logical_points': s.oracle.used,
            'actual_physical_points': s.actual(), 'requests': s.oracle.counts()['requests'],
            'generated_branch_count': len(s.keys), 'checked_plan_count': len(s.checked),
            'pending_complete_plans': len(s.queue), 'done': s.done, 'stop_reason': s.reason}

    def next(self):
        if self.search.done:
            return None
        while self.cursor < len(self.slots):
            operation = self.slots[self.cursor]
            node = self.search.fixed_expansion(self.chosen_plans) if operation == 'EXPAND_GRAPH' else None
            view, aliases = self.search.verification_view()
            if operation == 'EXPAND_GRAPH' and node is None:
                self.skipped_slots.append({'slot_index': self.cursor, 'operation': operation,
                                          'reason': 'NO_ELIGIBLE_GRAPH_EXPANSION'})
            elif operation == 'VERIFY_COMPLETE_PLAN' and not view.window():
                self.skipped_slots.append({'slot_index': self.cursor, 'operation': operation,
                                          'reason': 'NO_PENDING_VERIFICATION_WINDOW'})
            else:
                return operation, node, view, aliases
            self.cursor += 1
        return 'YIELD_TO_BEAM', None, None, None

    def mapping(self, view, aliases):
        return [{'slot': i, 'branch_id': n['id'], 'prefix': list(n['prefix']), 'origin': n['origin'],
            'actions': list(n['info']['actions']), 'alias_branch_ids': sorted(aliases[n['info']['actions']]),
            'has_prefix_wait_alias': any(self.search.frontier[bid]['origin'] == 'prefix_wait'
                                        for bid in aliases[n['info']['actions']])}
            for i, n in enumerate(view.window())]

    def check_proposal(self, proposal, policy=None):
        selected = self.next()
        require(selected is not None, 'Proposal after natural termination')
        operation, node, view, aliases = selected
        observation = view.observation('raw') if operation == 'VERIFY_COMPLETE_PLAN' else None
        mapping = self.mapping(view, aliases) if operation == 'VERIFY_COMPLETE_PLAN' else []
        learned = operation == 'VERIFY_COMPLETE_PLAN' and self.method != 'beam'
        if observation is not None:
            same(proposal['observation'], observation, 'independent_pending_current_observation')
            actual = early.compact_observation(proposal['observation'])
            rebuilt = early.compact_observation(observation)
            if learned:
                require(policy is not None, 'Fixed learned ranking lacks independently loaded model')
                actual_choice, rebuilt_choice = policy.evaluate(actual), policy.evaluate(rebuilt)
                require(proposal['action'] == actual_choice[0] == rebuilt_choice[0], 'Masked dual-observation argmax differs')
                for output in (actual_choice, rebuilt_choice):
                    early.learning.close(proposal['logprob'], output[1], 'fixed ranking logprob')
                    early.learning.close(proposal['value'], output[2], 'fixed ranking value')
            else:
                require(proposal['action'] == view.rule('beam') and proposal['logprob'] is proposal['value'] is None,
                        'Common window rule or nonlearned statistics changed')
            action = proposal['action']
            require(type(action) is int and 0 <= action < len(mapping) and observation['mask'][action], 'Unexposed/masked verification')
            node = view.window()[action]
        else:
            require(proposal['action'] is proposal['logprob'] is proposal['value'] is None,
                    'Fixed graph/handoff operation secretly uses a learned choice')
        expected = {'version': CONTROL_VERSION, 'schedule': self.schedule, 'slot_index': self.cursor,
            'operation': operation, 'decision_id': self.search.total_decisions,
            'branch_id': node['id'] if node else None, 'actions': list(node['info']['actions']) if node else None,
            'prefix': list(node['prefix']) if node else None, 'observation': observation,
            'window_mapping': mapping, 'state_before': self.state(), 'action': proposal['action'],
            'logprob': proposal['logprob'], 'value': proposal['value'],
            'method': self.method if observation is not None else ('fixed_graph_rule' if operation == 'EXPAND_GRAPH' else 'fixed_handoff'),
            'policy_used': learned}
        same(proposal, expected, 'independent_current_macro_proposal')
        return expected

    def apply(self, proposal):
        response = self.search.operate(proposal['operation'], proposal['branch_id'])
        if proposal['operation'] == 'EXPAND_GRAPH':
            self.chosen_plans.add(tuple(proposal['actions']))
        elif proposal['operation'] == 'VERIFY_COMPLETE_PLAN':
            self.policy_decisions += int(proposal['policy_used'])
        self.records.append({'decision_id': proposal['decision_id'], 'slot_index': proposal['slot_index'],
            'operation': proposal['operation'], 'branch_id': proposal['branch_id'], 'method': proposal['method'],
            'policy_used': proposal['policy_used'], 'verification_status': response['verification_status']})
        if proposal['operation'] != 'YIELD_TO_BEAM':
            self.cursor += 1
        require(self.search.graph_expansions <= 16 and self.search.verify_attempts <= 16 and self.search.total_decisions <= 33,
                'Macro budget exceeded')
        return response

    def check_control_result(self, result):
        require(self.next() is None and self.search.done, 'Macro phase never closed')
        expected = {'version': CONTROL_VERSION, 'engine_version': PROTOTYPE_VERSION, 'schedule': self.schedule,
            'requested_method': self.method, 'adapter': adapter_metadata(self.search), 'graph_operation_limit': 16,
            'verification_choice_limit': 16, 'total_operation_limit_including_yield': 33,
            'bootstrap_outside_macro_limits_but_paid': True,
            'verify_failure_repairs': 'original_reliable_failed_step_repairs_not_extra_macro_EXPAND',
            'dominated_skip_consumes_verification_choice': True,
            'schedule_slots_consumed': self.cursor, 'skipped_slots': self.skipped_slots, 'records': self.records,
            'policy_decisions': self.policy_decisions, 'controller_failed': False, 'finish_reason': self.search.reason,
            'guard_established': False, 'complete_guard_still_required': True}
        same(result['control'], expected, 'independent_complete_macro_control')
        require(result['policy_decisions'] == self.policy_decisions and result['training_interactions'] == result['optimizer_updates'] == 0,
                'Fixed ranking participation/training count differs')
        return check_deferred_result(self.search, result, controlled=True)


def adapter_metadata(replay):
    return {'version': ADAPTER_VERSION, 'feature_source': 'BranchSearch.observation_unchanged',
        'rule_source': 'BranchSearch.rule_action_beam_unchanged',
        'window_rule': {'version': 'alternating_unique_hold_coverage_v1',
            'coverage_turn': 'completed_expansions_mod_2_equals_1',
            'eligible': 'prefix_wait_and_unchecked_and_graph_energy_strictly_above_incumbent',
            'coverage_depth': 'prefix_depth_of_oldest_eligible_branch',
            'coverage_order': 'oldest_unique_full_plans_at_coverage_depth',
            'ordinary_window': 'original_BranchSearch_window_unchanged',
            'no_eligible_fallback': 'original_BranchSearch_window_unchanged',
            'scope': 'alternate_turn_candidate_exposure_only_Beam_selection_unchanged',
            'unexposed_frontier': 'preserved_without_mutation',
            'selection_and_physics': 'inherited_unchanged_from_BranchSearch'},
        'coverage_turn_counter': 'completed_verify_attempts_including_dominated_skips',
        'representative_order': ['prefix_length_ascending', 'prefix_wait_before_other_origins', 'branch_id'],
        'eligibility': 'complete_plan_in_verification_queue_and_not_in_checked',
        'dominated_pending': 'visible_until_original_free_skip_consumes_one_VERIFY_choice',
        'unexposed_and_alias_nodes': 'retained_unchanged_in_engine',
        'candidate_raw_dim': 27 + replay.m.horizon + 3 * len(replay.m.names),
        'candidate_associated_dim': 35 + replay.m.horizon + 3 * len(replay.m.names),
        'global_dim': 11 + len(replay.m.names), 'action_window_capacity': 32,
        'load_names_in_feature_order': list(replay.m.names),
        'point_budget_feature': 'cumulative_per_view_logical_points_including_bootstrap',
        'model_scope': 'fixed_candidate_ranking_only_not_a_three_operation_policy',
        'distribution_shift': 'new_pending_only_window_and_graph_before_verification_schedule'}


def check_deferred_records(mech, cfg, spec, result, attempts, decisions, evidence, condition_sha, policy=None):
    oracle = branch.PaidRequests(evidence['request_rows'], evidence['logical'], mech, 256, condition_sha)
    replay = DeferredReplay(mech, cfg, oracle, attempts, evidence['logical_rows'])
    macro = MacroReplay(replay, spec['macro_schedule'], spec['method'])
    require(len(decisions) % 2 == 0, 'Unclosed deferred decision')
    boundaries, first_verify_state, raw_hash_matches, compact_hash_matches = [], None, 0, 0
    elapsed_operations = 0.

    def boundary(choice):
        boundaries.append({'phase_choices': choice, **replay.costs(),
            'quality_pct': 100 * replay.quality if replay.best is not None else None,
            'fully_certified': replay.best is not None})
    boundary(0)
    identities = {'transition_id', 'phase_choice', 'phase', 'proposal', 'obs_raw_sha256',
                  'obs_compact_sha256', 'checkpoint_sha256', 'oracle_before'}
    for choice in range(len(decisions) // 2):
        start, finish = decisions[2 * choice:2 * choice + 2]
        require(set(start) == identities | {'status', 'obs_compact'}
                and set(finish) == identities | {'status', 'response', 'oracle_after'}, 'Deferred journal schema differs')
        require(start['status'] == 'STARTED' and finish['status'] == 'COMPLETED'
                and start['phase'] == 'EARLY' and start['transition_id'] == start['phase_choice'] == choice,
                'Deferred decision sequence differs')
        require(all(start[k] == finish[k] for k in identities), 'Deferred decision changed after selection')
        require(start['oracle_before'] == replay.costs(), 'Unobserved request before deferred decision')
        require(not replay.done and replay.total_decisions == choice, 'Deferred decision after stop or counter reset')
        next_operation = macro.next()
        require(next_operation is not None, 'Unexpected extra macro proposal')
        _, _, view, _ = next_operation
        expected_observation = view.observation('raw') if next_operation[0] == 'VERIFY_COMPLETE_PLAN' else None
        expected_compact = early.compact_observation(expected_observation) if expected_observation is not None else None
        actual_compact = start['obs_compact']
        require(start['obs_compact_sha256'] == digest(actual_compact), 'Recorded compact payload hash differs')
        require(isinstance(start['obs_raw_sha256'], str) and len(start['obs_raw_sha256']) == 64
                and all(c in '0123456789abcdef' for c in start['obs_raw_sha256']), 'Malformed raw observation digest')
        same(actual_compact, expected_compact, 'independent_pending_compact_observation')
        compact_hash_matches += int(digest(actual_compact) == digest(expected_compact))
        raw_hash_matches += int(start['obs_raw_sha256'] == digest(expected_observation))
        if expected_observation is None:
            require(start['obs_raw_sha256'] == digest(None), 'Graph/handoff carries hidden observation')
            actual_observation = None
        else:
            require(set(actual_compact) == set(early.COMPACT_FIELDS)
                    and all(type(v) is bool for v in actual_compact['mask'])
                    and actual_compact['mask'] == [True] * len(view.window()), 'Pending mask differs')
            actual_observation = {**expected_observation, **actual_compact}
            if first_verify_state is None:
                first_verify_state = digest({'generated': replay.generated, 'assessed': replay.assessment_trace,
                    'pending': replay.pending(), 'logical_points': oracle.used, 'requests': oracle.counts()['requests'],
                    'incumbent': replay.best})
        require('observation' not in start['proposal'], 'Redundant hidden proposal observation')
        proposal = {**start['proposal'], 'observation': actual_observation}
        expected = macro.check_proposal(proposal, policy)
        require(start['checkpoint_sha256'] == (spec['checkpoint_sha256'] if expected['policy_used'] else None),
                'Unapproved fixed model or learned macro operation')
        before_logical = oracle.used
        response = macro.apply(expected)
        actual_response = deepcopy(finish['response'])
        elapsed = window.finite_time(actual_response.pop('elapsed_seconds'), 'macro operation elapsed')
        physical = math.fsum(evidence['logical'][i]['elapsed_seconds'] for i in range(before_logical, oracle.used))
        require(elapsed + 1e-8 >= physical, 'Macro elapsed excludes paid physical calls')
        elapsed_operations += elapsed
        same(actual_response, response, 'independent_macro_response')
        require(finish['oracle_after'] == replay.costs(), 'Deferred after-query counts differ')
        boundary(choice + 1)
    scientific = macro.check_control_result(result)
    require(result['search_decisions'] == len(decisions) // 2 == replay.total_decisions
            and result['fully_certified'] is (replay.best is not None)
            and result['quality_pct'] == boundaries[-1]['quality_pct'], 'Deferred phase counts/quality differ')
    require(elapsed_operations <= result['stage_component_seconds']['search_step_seconds'] + 1e-8,
            'Phase time omits macro operations')
    return {'phase': 'EARLY', 'decisions': replay.total_decisions, 'policy_decisions': macro.policy_decisions,
        'quality_pct': boundaries[-1]['quality_pct'], 'fully_certified': replay.best is not None,
        'stop_reason': replay.reason, 'boundaries': boundaries, 'scientific_result': scientific,
        'semantic_signature_sha256': digest(scientific), 'graph_expansions': replay.graph_expansions,
        'verify_attempts': replay.verify_attempts, 'total_decisions': replay.total_decisions,
        'handed_off': replay.handed_off, 'first_verify_state_sha256': first_verify_state,
        'compact_hash_exact_matches': compact_hash_matches, 'raw_hash_exact_matches': raw_hash_matches,
        'actual_and_independent_policy_actions_equal': True}


def protocol_settings():
    return {'cells': 64, 'contexts': 8, 'mode': 'shared', 'phases': ['EARLY', 'GUARD'],
        'seeds': list(SEEDS), 'slots': list(SLOTS), 'logical_budget_per_view': 256,
        'early_expansion_slots': 16, 'early_verification_slots': 16, 'maximum_early_operations': 33,
        'bootstrap': 'original paid initial snapshot and complete WAIT', 'guard_expansion_cap': 256,
        'total_physical_budget_per_cell': 512, 'total_physical_budget': 32768,
        'maximum_numerical_entries': 131264, 'physical_runtimes_per_cell': 1,
        'max_workers': 2, 'worker_seconds': 900, 'controller_seconds': 7200,
        'minimum_free_disk_bytes': 12 * 2**30, 'storage_margin': 1.25, 'report_reserve_bytes': 2 * 2**30,
        'phase_order': ['EARLY', 'GUARD'], 'tie_rule': 'exact_equal_energy_select_GUARD',
        'same_runtime_both_modes': True, 'new_training': 0, 'no_cross_cell_cache': True,
        'no_historical_numerical_answers': True, 'no_automatic_retry': True,
        'no_upper_bound_change': True, 'model_role': 'fixed candidate ranking only, not learned EXPAND/VERIFY/YIELD',
        'macro_control': '16 EXPAND then up to16 VERIFY; interleaved rule alternates16 EXPAND/VERIFY; at most1 YIELD',
        'same_optimized_window_for_all_rankers': True, 'no_chosen_plan_targets': True,
        'timing': 'cold complete cell process; all initialization, both searches, logging and exit paid',
        'recording': 'STARTED proposal with raw observation and stable mapping; COMPLETED exact response; separate logical and real costs',
        'dispatch': 'hashed contexts and slots, Latin slot rotation; adjacent pair drained before next pair',
        'aggregation': 'mean three seeds within context then equal mean eight contexts; retain all64 tasks'}


def expected_schedule(contexts, models):
    require(len(contexts) == 8 and {c['context_id'] for c in contexts} == set(range(8))
            and len({c['scenario_id'] for c in contexts}) == 8, 'Complete fixed eight contexts required')
    lookup = {(m['method'], m['seed']): m for m in models}
    require(len(models) == len(lookup) == 6 and set(lookup) == {(m, s) for m in ('bc_raw', 'ppo_raw') for s in SEEDS},
            'Fixed six checkpoint population differs')
    ordered = sorted(contexts, key=lambda c: prior.digest_text('deferred-context-v1|' + prior.canonical(c)))
    slots = sorted(SLOTS, key=lambda s: prior.digest_text('deferred-slots-v1|' + s))
    rows = []
    for rank, c in enumerate(ordered):
        for position in range(8):
            slot = slots[(rank + position) % 8]
            seed = next((s for s in SEEDS if slot.endswith(str(s))), 0)
            arm = slot[:-len(str(seed))] if seed else slot
            method = 'bc_raw' if arm == 'deferred_bc' else 'ppo_raw' if arm == 'deferred_ppo' else 'beam'
            model = lookup.get((method, seed))
            index = len(rows)
            rows.append({**c, 'cell': f'episodes/{index:03d}', 'episode_id': index, 'pair_order': rank * 4 + position // 2,
                'context_order': rank, 'arm_position': position, 'arm_slot': slot,
                'mode': 'shared', 'arm': arm, 'method': method, 'seed': seed,
                'macro_schedule': 'interleaved' if arm == 'interleaved_window' else 'deferred',
                'phase': 'evaluation', 'group': arm, 'protected': True, 'representation': 'raw', 'deterministic': True,
                'checkpoint': model['checkpoint'] if model else None,
                'checkpoint_sha256': model['source_sha256'] if model else None,
                'point_budget': 256, 'logical_point_budget': 256, 'physical_point_budget': 512,
                'total_logical_point_budget': 512, 'window_variant': 'pending_unique_coverage_v1'})
    return rows


def expected_phase_spec(spec, name):
    require(name in shared.VIEWS, 'Unapproved phase')
    return {**spec, 'parent_cell': spec['cell'], 'cell': spec['cell'] + '/' + name, 'search_phase': name,
        'method': spec['method'] if name == 'EARLY' else 'beam', 'seed': spec['seed'] if name == 'EARLY' else 0,
        'checkpoint': spec['checkpoint'] if name == 'EARLY' else None,
        'checkpoint_sha256': spec['checkpoint_sha256'] if name == 'EARLY' else None,
        'point_budget': 256, 'logical_point_budget': 256, 'physical_point_budget': 256}


def check_models(run):
    source = ROOT / 'runs' / MODELS_RUN
    require(sha(source / 'completion_manifest.json') == sha(run / 'model_source_completion_manifest.json') == MODELS_SHA,
            'Model source root seal changed')
    sealed = {k.replace('\\', '/'): v for k, v in read(run / 'model_source_completion_manifest.json').items()}
    models = read(run / 'model_bindings.json')
    original_models, _, _ = prior.model_sources()
    require(sha(run / 'model_bindings.json') == sha(source / 'model_bindings.json') == sealed['model_bindings.json']
            and models == original_models, 'Same-source fixed model endpoints changed')
    for model in models:
        name = model['checkpoint']
        require(sha(run / name) == sha(source / name) == sealed[name] == model['source_sha256'], 'Fixed model bytes changed')
    for name, expected in sealed.items():
        if name.startswith('model_preflight/') or name == 'model_preflight_binding.json':
            require(sha(run / name) == expected, 'Original model preflight changed')
    prior.check_model_preflight(run, models)
    return models


def check_parent(run):
    run = Path(run).resolve()
    require(run.parent == ROOT / 'runs', 'Deferred pilot outside project run directory')
    prepared = read(run / 'prepared.json')
    require(prepared['status'] == 'FROZEN_PREPARED_NO_DSS'
            and prepared['freeze_sha256'] == sha(run / 'prephysical_freeze_manifest.json'), 'Prephysical freeze changed')
    frozen = shared.checked_manifest(run, 'prephysical_freeze_manifest.json')
    source = ROOT / 'runs' / SOURCE_RUN
    require(sha(source / 'completion_manifest.json') == sha(run / 'source_completion_manifest.json') == SOURCE_SHA,
            'Original shared source seal changed')
    sealed = {k.replace('\\', '/'): v for k, v in read(run / 'source_completion_manifest.json').items()}
    require(read(run / 'source_binding.json') == {'source_run': SOURCE_RUN, 'source_completion_sha256': SOURCE_SHA,
        'models_run': MODELS_RUN, 'models_completion_sha256': MODELS_SHA}, 'Source/model attribution changed')
    for name in ('config.json', 'schedule.json', 'prepared.json', 'audit/summary.json', 'contexts.json'):
        require(sha(source / name) == sealed[name], 'Changed historical input: ' + name)
    source_audit = read(source / 'audit/summary.json')
    require(source_audit['status'] == 'PASS' and source_audit['audited_cells'] == 16 and not source_audit['errors'],
            'Shared source not completely audited')
    additions = {'docs/clpu_deferred_pilot_protocol_v1.md', 'src/dnr/nature34_clpu_deferred.py',
        'src/dnr/nature34_clpu_deferred_control.py', 'scripts/run_clpu_deferred_pilot.py',
        'scripts/audit_clpu_deferred_pilot.py', 'scripts/report_clpu_deferred_pilot.py',
        'scripts/plot_clpu_deferred_pilot.py', 'tests/test_nature34_clpu_deferred.py',
        'tests/test_nature34_clpu_deferred_control.py', 'tests/test_run_clpu_deferred_pilot.py',
        'tests/test_clpu_deferred_audit.py', 'tests/test_report_clpu_deferred_pilot.py'}
    require(prepared['source_files'] == sorted(set(read(source / 'prepared.json')['source_files']) | additions),
            'Required independent/control/source closure changed')
    for name in prepared['source_files']:
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT) and path != ROOT and 'source/' + name in frozen
                and sha(path) == sha(run / 'source' / name), 'Changed frozen execution source: ' + name)
        if 'source/' + name in sealed:
            require(sha(path) == sealed['source/' + name], 'Inherited scientific source changed: ' + name)
    require(sha(run / 'protocol.md') == sha(ROOT / 'docs/clpu_deferred_pilot_protocol_v1.md'), 'Protocol changed')
    cfg = read(run / 'config.json')
    same(cfg, {**read(source / 'config.json'), 'deferred_study': STUDY, 'deferred_pilot': protocol_settings()},
         'complete_deferred_scientific_contract')
    scope = prior.check_frozen_physics(run, prepared, cfg)
    models = check_models(run)
    contexts = read(source / 'contexts.json')
    require(read(run / 'contexts.json') == contexts, 'Original training population changed')
    schedule = read(run / 'schedule.json')
    same(schedule, expected_schedule(contexts, models), 'all_64_cold_deferred_tasks')
    for spec in schedule:
        require(read(run / spec['cell'] / 'spec.json') == spec, 'Prepared task differs')
        for name in shared.VIEWS:
            require(read(run / spec['cell'] / name / 'spec.json') == expected_phase_spec(spec, name), 'Prepared logical view differs')
    chosen = [s for s in read(source / 'schedule.json') if s['mode'] == 'shared']
    expected_history = []
    for spec in chosen:
        item = {k: spec[k] for k in ('context_id', 'scenario_id', 'event_id')}
        original = spec['cell'] + '/GUARD/phase_result.json'
        target = f'historical/c{spec["context_id"]:02d}/GUARD/phase_result.json'
        require(sha(run / target) == sha(source / original) == sealed[original], 'Original complete Beam result changed')
        item.update(source_cell=spec['cell'], phases={'GUARD': {'result_file': target,
                    'source_file': original, 'sha256': sealed[original]}})
        expected_history.append(item)
    require(read(run / 'historical_bindings.json') == expected_history, 'Guard source attribution changed')
    cases = list(ET.parse(run / 'software_tests.xml').getroot().iter('testcase'))
    require(cases and not any(list(c.iter('failure')) + list(c.iter('error')) + list(c.iter('skipped')) for c in cases),
            'Passed complete frozen tests required')
    for component in ('test_nature34_clpu_deferred_control', 'test_run_clpu_deferred_pilot',
                      'test_clpu_deferred_audit', 'test_report_clpu_deferred_pilot'):
        require(any(component in c.get('classname', '') for c in cases), 'Missing frozen component: ' + component)
    resource = read(run / 'resource_budget.json')
    for key, value in {'total_physical_cap': 32768, 'minimum_free_bytes': 12 * 2**30,
        'report_reserve_bytes': 2 * 2**30, 'successful_component_tests': len(cases)}.items():
        require(resource[key] == value, 'Resource contract changed: ' + key)
    rate = max(sum(p.stat().st_size for p in (source / s['cell']).rglob('*') if p.is_file()) /
               max(1, read(source / s['cell'] / 'result.json')['actual_physical_points']) for s in chosen)
    require(resource['historical_max_bytes_per_actual_point'] == rate
            and resource['forecast_bytes'] == math.ceil(rate * 32768 * 1.25)
            and resource['required_free_bytes'] == 14 * 2**30 + resource['forecast_bytes']
            and resource['free_bytes_at_prepare'] >= resource['required_free_bytes'], 'Incomplete storage forecast')
    require(read(run / 'authorization.json')['execution_authorized'] is True, 'Missing execution authorization')
    return cfg, scope, schedule


def check_cell(run, spec, cfg, scope):
    cell = (run / spec['cell']).resolve()
    require(cell.is_relative_to(run) and cell != run, 'Escaped cold task')
    sealed = shared.checked_manifest(cell)
    required = {'spec.json', 'result.json', 'worker_started.json', 'worker_finished.json', 'execution.json',
        'service_events.jsonl', 'service_summary.json', 'cost_checkpoints.jsonl', 'physical/points.jsonl.gz',
        'physical/requests.jsonl', 'physical/oracle_summary.json', 'physical/command_ledger.jsonl'}
    for name in shared.VIEWS:
        required |= {name + '/' + p for p in ('spec.json', 'phase_result.json', 'initial_state.json', 'plans.jsonl.gz',
            'decisions.jsonl.gz', 'logical_points.jsonl.gz', 'logical_requests.jsonl', 'view_summary.json')}
        require(not (cell / name / 'points.jsonl.gz').exists() and not (cell / name / 'command_ledger.jsonl').exists(),
                'A logical view fabricates paid physical evidence')
    require(required <= set(sealed) and not (cell / 'error.json').exists(), 'Incomplete/failed task evidence')
    result = read(cell / 'result.json')
    require(read(cell / 'spec.json') == result['spec'] == spec and result['new_training'] == 0,
            'Task identity or unauthorized training changed')
    require(result['version'] == result['study'] == STUDY and result['schema'] == 'clpu-deferred-protected-result-v1',
            'Deferred scientific output version differs')
    require(result['checkpoint_sha256'] == spec['checkpoint_sha256'], 'Fixed checkpoint differs')
    policy = early.IndependentPolicy(run / spec['checkpoint'], spec) if spec['checkpoint'] else None
    worker, finished, execution = [read(cell / p) for p in ('worker_started.json', 'worker_finished.json', 'execution.json')]
    require(type(worker['pid']) is int and worker['pid'] > 0 and finished['pid'] == worker['pid']
            and finished['status'] == 'COMPLETE' and finished['result_sha256'] == sha(cell / 'result.json'),
            'Cold worker completion unbound')
    global_execution = read(run / 'execution_started.json')
    require(execution['spec_sha256'] == sha(cell / 'spec.json') and execution['controller_pid'] == global_execution['pid']
            and execution['deadline_epoch_seconds'] == min(global_execution['deadline_epoch_seconds'],
                                                           execution['started_epoch_seconds'] + 900), 'Wrong worker deadline')
    scenarios = {r['scenario_id']: r for r in read(run / 'selected_scenarios.json')}
    events = {r['event_id']: r for r in read(run / 'selected_events.json')}
    mech = branch.Mechanics(read(run / 'author_static_metadata.json'), read(run / 'load_reference.json'),
        scenarios[spec['scenario_id']], read(run / 'condition.json'), events[spec['event_id']], cfg)
    points, statuses = shared.check_actual_points(run, cell / 'physical', spec, mech)
    commands = old.check_commands(cell / 'physical', points, read(run / 'command_inventory.json'))
    shared.check_actual_requests(list(read_lines(cell / 'physical/requests.jsonl')), points,
        read(cell / 'physical/oracle_summary.json'), commands, statuses)
    views = {name: {'logical_rows': list(read_lines(cell / name / 'logical_points.jsonl.gz')),
                   'requests': list(read_lines(cell / name / 'logical_requests.jsonl'))} for name in shared.VIEWS}
    evidence, _ = shared.check_service_ledger(list(read_lines(cell / 'service_events.jsonl')), views, points,
        mech, scope, sha(run / 'condition.json'), 'shared')
    service = read(cell / 'service_summary.json')
    shared.check_service_summaries(service, {name: read(cell / name / 'view_summary.json') for name in shared.VIEWS},
                                  evidence, scope, 'shared', commands['numerical_entries'])
    require(result['service'] == service, 'Complete service accounting differs')
    phases, checks = {}, {}
    history = next(r for r in read(run / 'historical_bindings.json') if r['context_id'] == spec['context_id'])
    for name in shared.VIEWS:
        directory = cell / name
        p = read(directory / 'phase_result.json')
        require(read(directory / 'spec.json') == p['spec'] == expected_phase_spec(spec, name)
                and read(directory / 'initial_state.json') == mech.initial, 'Search state or identity changed')
        require(p['oracle'] == service['views'][name], 'View summary differs from physical service')
        plans, decisions = list(read_lines(directory / 'plans.jsonl.gz')), list(read_lines(directory / 'decisions.jsonl.gz'))
        require(p['attempts'] == plans, 'Omitted/relabelled attempted plan')
        if name == 'EARLY':
            checked = check_deferred_records(mech, cfg['branch_search'], p['spec'], p, plans, decisions,
                                            evidence[name], sha(run / 'condition.json'), policy)
        else:
            checked = shared.check_phase_records(mech, cfg['branch_search'], name, p['spec'], p, plans,
                decisions, evidence[name], sha(run / 'condition.json'))
            require(p['verify_attempts'] is None and p['graph_expansions'] == checked['decisions'],
                    'Original Beam combined steps confused with atomic VERIFY')
            original = read(run / history['phases']['GUARD']['result_file'])
            original = {**original, 'actual_physical_points': original['logical_points']}
            for key, value in checked['scientific_result'].items():
                same(original[key], value, 'historical_complete_original_Beam.' + key)
            require(original['oracle']['requests'] == evidence[name]['requests'], 'Original Beam query order/count changed')
        phases[name], checks[name] = p, checked
    # Only version labels are projected to the frozen pure protection checker;
    # all actual results and the new version were validated immediately above.
    chosen = shared.check_selection({**result, 'version': shared.STUDY, 'study': shared.STUDY,
                                     'schema': 'clpu-shared-protected-result-v1'}, phases)
    cps = list(read_lines(cell / 'cost_checkpoints.jsonl'))
    require(result['checkpoints'] == cps, 'Top quality/cost curve omits boundaries')
    shared.check_checkpoints(cps, checks, phases)
    for name in shared.VIEWS:
        require(result['phase_summary'][name] == {k: phases[name][k] for k in ('quality_pct', 'fully_certified',
            'logical_points', 'actual_physical_points', 'search_decisions', 'policy_decisions', 'graph_expansions',
            'verify_attempts', 'phase_elapsed_seconds', 'stage_component_seconds')}, 'Phase summary incomplete')
    require(set(result['phase_summary']) == set(shared.VIEWS), 'Extra/missing search phases')
    for key, expected in {'logical_points': service['logical_points'], 'actual_physical_points': len(points),
        'numerical_entries': commands['numerical_entries'], 'requests': sum(e['requests'] for e in evidence.values()),
        'search_decisions': sum(p['search_decisions'] for p in phases.values()),
        'policy_decisions': checks['EARLY']['policy_decisions']}.items():
        require(result[key] == expected, 'Task cost/participation differs: ' + key)
    shared.check_times(result, phases, finished, math.fsum(p['elapsed_seconds'] for p in points.values()))
    model_time = window.finite_time(result['model_load_seconds'], 'model loading cost')
    require(model_time <= result['preflight_elapsed_seconds'] and (spec['checkpoint'] is not None or model_time == 0.),
            'Model work hidden from cold end-to-end time')
    if spec['checkpoint']:
        require(sha(run / spec['checkpoint']) == spec['checkpoint_sha256'], 'Frozen model changed during audit')
    return {**{k: spec[k] for k in ('cell', 'context_id', 'scenario_id', 'event_id', 'mode', 'arm', 'seed',
        'pair_order', 'arm_slot', 'macro_schedule')}, 'status': 'PASS', 'result_sha256': sha(cell / 'result.json'),
        'cell_completion_sha256': sha(cell / 'completion_manifest.json'), 'quality_pct': result['quality_pct'],
        'guard_established': chosen['guard_established'], 'guard_quality_pct': result['baseline_quality_pct'],
        'gain_over_guard_kwh': chosen['gain_over_guard_kwh'], 'logical_points': service['logical_points'],
        'physical_points': len(points), 'numerical_entries': commands['numerical_entries'], 'requests': result['requests'],
        'search_decisions': result['search_decisions'], 'policy_decisions': result['policy_decisions'],
        'worker_elapsed_seconds': result['worker_elapsed_seconds'], 'model_load_seconds': model_time,
        'worker_finished_elapsed_seconds': finished['worker_elapsed_seconds'], 'worker_pid': worker['pid'],
        'worker_parent_pid': worker['parent_pid'], 'execution': execution, 'physical_statuses': dict(statuses),
        'history_verified': True, 'quality_checkpoints_verified': True,
        'cross_view_reuses': sum(e['cross_view_reuses'] for e in evidence.values()),
        'first_verify_state_sha256': checks['EARLY']['first_verify_state_sha256'],
        'actual_and_independent_policy_actions_equal': True,
        'phase_summary': {name: {**result['phase_summary'][name], 'requests': evidence[name]['requests'],
            'semantic_signature_sha256': checks[name]['semantic_signature_sha256'],
            'phase_result_sha256': sha(cell / name / 'phase_result.json'),
            **({k: checks[name][k] for k in ('total_decisions', 'handed_off', 'compact_hash_exact_matches', 'raw_hash_exact_matches')}
               if name == 'EARLY' else {})} for name in shared.VIEWS}, 'exact_checkpoints': cps}


def check_controller(controller, schedule, checks, dispatch):
    names = [s['cell'] for s in schedule]
    require(len(names) == len(set(names)) == len(checks) == 64 and {c['cell'] for c in checks} == set(names),
            'Missing or duplicated audited tasks')
    require(controller['status'] == 'COMPLETE_CELLS_PENDING_AUDIT' and controller['scheduled'] == 64
            and controller['dispatched'] == schedule and len(controller['completed']) == 64
            and set(controller['completed']) == set(names) and not controller['errors'] and not controller['unknown_cost_cells'],
            'Controller not completely successful or unknown paid costs')
    require([d['cell'] for d in dispatch] == names and len(controller['exits']) == 64, 'Wrong dispatch sequence/retry')
    exits = {e['cell']: e for e in controller['exits']}
    require(set(exits) == set(names), 'Missing actual process exits')
    by_cell, previous_end = {c['cell']: c for c in checks}, 0.
    for start in range(0, 64, 2):
        ends = []
        for spec, d in zip(schedule[start:start+2], dispatch[start:start+2]):
            e, c = exits[spec['cell']], by_cell[spec['cell']]
            require(e['exit_code'] == 0 and e['drained_after_controller_error'] is False
                    and e['launcher_pid'] == d['launcher_pid'] and e['pair_order'] == d['pair_order'] == spec['pair_order'],
                    'Failed/drained/wrongly attributed worker exit')
            begin = window.finite_time(e['started_epoch_seconds'], 'cold process start')
            end = window.finite_time(e['finished_epoch_seconds'], 'cold process exit')
            elapsed = window.finite_time(e['process_wall_seconds'], 'whole cold process time')
            require(begin == d['started_epoch_seconds'] == c['execution']['started_epoch_seconds']
                    and previous_end <= begin <= end and elapsed >= c['worker_finished_elapsed_seconds'],
                    'Overlapping pairs or omitted complete worker time')
            require(c['worker_parent_pid'] in (d['launcher_pid'], c['execution']['controller_pid']), 'Worker ancestry differs')
            require(abs((end - begin) - elapsed) <= 2., 'Wall/monotonic process timing disagree materially')
            c['process_wall_seconds'] = elapsed
            ends.append(end)
        previous_end = max(ends)
    totals = {key: sum(c[key] for c in checks) for key in ('logical_points', 'physical_points', 'numerical_entries')}
    require(all(controller[k] == v for k, v in totals.items()) and totals['physical_points'] <= 32768
            and totals['logical_points'] <= 32768 and totals['numerical_entries'] == 192 + 4 * totals['physical_points'],
            'All-task logical/actual budgets or initialization cost do not close')
    require(window.finite_time(controller['wall_seconds'], 'controller time') >= max(c['process_wall_seconds'] for c in checks),
            'Controller clock omits worker execution')
    return totals


def context_equalities(checks):
    require(len(checks) == 64 and {c['context_id'] for c in checks} == set(range(8)), 'All eight contexts required')
    rows = []
    for context in range(8):
        group = [c for c in checks if c['context_id'] == context]
        require(len(group) == 8 and {c['arm_slot'] for c in group} == set(SLOTS), 'Missing/mislabeled fixed arm or seed')
        require(all(c['guard_established'] and c['history_verified'] for c in group), 'Original completed Beam protection absent')
        require(len({c['phase_summary']['GUARD']['semantic_signature_sha256'] for c in group}) == 1
                and len({c['guard_quality_pct'] for c in group}) == 1, 'Original GUARD trajectory differs across arms')
        deferred = [c for c in group if c['macro_schedule'] == 'deferred']
        require(len(deferred) == 7 and len({c['first_verify_state_sha256'] for c in deferred}) == 1,
                'Rankers did not receive the same initial deferred graph exposure')
        rows.append({'context_id': context, 'status': 'PASS', 'guard_trajectories_equal': True,
            'guard_quality_pct': group[0]['guard_quality_pct'], 'common_deferred_initial_exposure_verified': True,
            'first_verify_state_sha256': deferred[0]['first_verify_state_sha256'], 'tasks': 8})
    return rows


def source_hashes(run, scope):
    return {'freeze_sha256': sha(run / 'prephysical_freeze_manifest.json'), 'config_sha256': sha(run / 'config.json'),
        'schedule_sha256': sha(run / 'schedule.json'), 'source_binding_sha256': sha(run / 'source_binding.json'),
        'physical_scope_file_sha256': sha(run / 'physical_scope.json'), 'physical_scope_canonical_sha256': digest(scope),
        'scope_review_manifest_sha256': sha(run / 'scope_review/completion_manifest.json'),
        'exact_key_review_manifest_sha256': sha(run / 'exact_key_review/completion_manifest.json'),
        'historical_bindings_sha256': sha(run / 'historical_bindings.json'), 'source_shared_completion_sha256': SOURCE_SHA,
        'model_bindings_sha256': sha(run / 'model_bindings.json'), 'model_source_completion_sha256': MODELS_SHA,
        'model_preflight_manifest_sha256': sha(run / 'model_preflight/completion_manifest.json')}


def audit(run, output=None):
    from audit_d059_supplement import NoSolverImports, no_solver_loaded
    run = Path(run).resolve()
    output = Path(output).resolve() if output else run / 'audit'
    require(output.is_relative_to(ROOT) and output != run and not output.exists(), 'Preserve existing audit evidence')
    output.mkdir(parents=True)
    started, checks = time.perf_counter(), []
    result = {'status': 'FAIL', 'source_run': run.name, 'errors': [], 'audited_cells': 0, 'results': [],
              'new_audit_physical_points': 0, 'new_training_tasks': 0}
    guard = NoSolverImports()
    sys.meta_path.insert(0, guard)
    try:
        no_solver_loaded()
        cfg, scope, schedule = check_parent(run)
        result.update(source_hashes(run, scope))
        for spec in schedule:
            checked = check_cell(run, spec, cfg, scope)
            checks.append(checked)
            save(output / (spec['cell'].replace('/', '_') + '.json'), checked)
            save(output / 'progress.json', {'status': 'RUNNING', 'audited_cells': len(checks), 'scheduled': 64,
                'last_cell': spec['cell'], 'elapsed_seconds': time.perf_counter() - started,
                'new_physical_points': 0, 'new_training': 0})
        controller = read(run / 'controller_completion.json')
        require(controller['exits'] == read(run / 'controller_exits.json'), 'Process exit journal differs')
        totals = check_controller(controller, schedule, checks, list(read_lines(run / 'controller_dispatch.jsonl')))
        equalities = context_equalities(checks)
        no_solver_loaded()
        statuses = Counter()
        for row in checks:
            statuses.update(row['physical_statuses'])
            save(output / (row['cell'].replace('/', '_') + '.json'), row)
        result.update(status='PASS', audited_cells=len(checks), results=checks, context_equalities=equalities,
            **totals, physical_runtimes=64, logical_searches=128, complete_protected_cells=64, physical_statuses=dict(statuses),
            controller_completion_sha256=sha(run / 'controller_completion.json'),
            controller_dispatch_sha256=sha(run / 'controller_dispatch.jsonl'),
            interpretation='Eight observed training contexts; common fixed macro and pending-window optimization; frozen same-source candidate rankers, not equal total training cost or learned three-operation control. All original Beam guards completed and fully charged.')
    except BaseException as exc:
        result.update(audited_cells=len(checks), results=checks)
        result['errors'].append({'error': repr(exc), 'traceback': traceback.format_exc()})
    finally:
        sys.meta_path.remove(guard)
    result['elapsed_seconds'] = time.perf_counter() - started
    save(output / 'summary.json', result)
    save(output / 'completion_manifest.json', {p.relative_to(output).as_posix(): sha(p)
        for p in output.rglob('*') if p.is_file() and p.name != 'completion_manifest.json'})
    return result


def preflight(run, output):
    from audit_d059_supplement import NoSolverImports, no_solver_loaded
    run, output = Path(run).resolve(), Path(output).resolve()
    require(output.is_relative_to(ROOT) and not output.exists(), 'Preserve earlier preflight evidence')
    output.mkdir(parents=True)
    result = {'status': 'FAIL', 'source_run': run.name, 'errors': [], 'new_physical_points': 0, 'new_training': 0}
    guard = NoSolverImports()
    sys.meta_path.insert(0, guard)
    try:
        no_solver_loaded()
        _, scope, schedule = check_parent(run)
        no_solver_loaded()
        result.update(status='PASS', contexts=8, cells=len(schedule), logical_searches=128, physical_runtimes=64,
                      fixed_models=6, **source_hashes(run, scope))
    except BaseException as exc:
        result['errors'].append({'error': repr(exc), 'traceback': traceback.format_exc()})
    finally:
        sys.meta_path.remove(guard)
    save(output / 'summary.json', result)
    save(output / 'completion_manifest.json', {'summary.json': sha(output / 'summary.json')})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        require(args.output is not None, 'Preflight requires a separate evidence output directory')
        result = preflight(args.run, args.output)
    else:
        result = audit(args.run, args.output)
    print(prior.canonical({k: v for k, v in result.items() if k not in ('results', 'context_equalities')}), flush=True)
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
