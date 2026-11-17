"""Causal parent-prefix recovery and isolated support-repair teacher forks.

Only a caller-supplied lazy live view can invoke a solver. Parent certificates
retain their logical IDs; fork evidence receives explicit source namespaces.
"""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import math

from .nature34_clpu_branch import BranchConfig
from .nature34_clpu_environment import CLPUBudgetExhausted, CLPUReliabilityError
from .nature34_clpu_equivalent_cache import canonical, sha_value, electrical_input_key, physical_evidence_payload
from .nature34_clpu_physical import certify_clpu_point
from .nature34_clpu_guard_first_control import GuardFirstStagedSearch, GuardFirstVerificationAdapter, GuardFirstMacroController
from .nature34_clpu_staged_energy_control import complete_energy_action
from .nature34_clpu_support_repair import generate_support_repair_candidates, ORIGIN

VERSION = 'clpu-causal-support-repair-teacher-v1'
CANDIDATE_HASH_DOMAIN = 'clpu-support-teacher-candidates-v1'
REOPENABLE = ('STAGED_VERIFICATION_POOL_EXHAUSTED', 'DEFERRED_FRONTIER_EXHAUSTED')
INHERITANCE_POLICY = {
    'version': 'clpu-parent-cutoff-evidence-inheritance-v1',
    'authorization': 'explicit_offline_teacher_parent_history_recovery',
    'scope_bytes': 'source_physical_scope_unchanged_as_scientific_input_identity_namespace',
    'source_policy': 'source_cold_cell_policy_remains_true_for_source_run_not_rewritten',
    'allowed_evidence': 'same_parent_completed_GUARD_and_EARLY_prefix_at_declared_cutoff_only',
    'alias': 'exact_full_input_key_and_current_identity_time_pure_recertification',
    'forbidden': ['future_parent_evidence', 'sibling_fork_evidence', 'other_parent_state_cache', 'other_context'],
    'charging': 'inherited_paid_cost_separate_from_new_actual_points_and_live_initialization',
}


def require(condition, message):
    if not condition:
        raise CLPUReliabilityError(message)


def certificate(row, point_id):
    return {'point_id': point_id, 'status': row['status'], 'reliable': row['reliable'],
            'feedback': deepcopy(row['structured_feedback']['violations'])}


class CausalPrefixOracle:
    def __init__(self, env, payload):
        self.env, self.payload, self.budget = env, payload, 256
        self.scope = deepcopy(payload['physical_scope'])
        self.binding = deepcopy(payload['physics_identity'])
        self.points = self.requests = self.hits = self._own_actual = 0
        self.cache, self.records, self._pool = {}, [], {}
        self._logical, self._requests = payload['logical_prefix'], payload['request_prefix']
        self.mode = 'REPLAY'
        self.journal = lambda event: None
        self._live = self._stack = self._factory = self._allowance = self._base = None
        require([r['logical_point_id'] for r in self._logical] == list(range(len(self._logical))),
                'Source logical prefix must be contiguous')
        require([r['request'] for r in self._requests] == list(range(1, len(self._requests)+1)),
                'Source requests must be contiguous')
        require(len(payload['physical_prefix']) == payload['parent_global_actual_points'], 'Wrong physical prefix length')
        for index, evidence in enumerate(payload['physical_prefix']):
            row, key = evidence['raw'], evidence['electrical_input_key']
            require(row['point_id'] == index and evidence['physical_point_id'] == index,
                    'Source paid IDs must be contiguous')
            require(row['status'] in ('PASS', 'FAIL') and row['reliable'] is True,
                    'Only reliable parent paid evidence is inheritable')
            require(sha_value(row) == evidence['raw_sha256'] and key['scope_sha256'] == sha_value(self.scope),
                    'Parent raw evidence or scientific scope changed')
            require(evidence['source_event_sequence'] <= payload['cutoff_service_sequence'],
                    'Future physical evidence entered parent prefix')
            fingerprint = sha_value(key)
            require(fingerprint not in self._pool, 'Duplicate paid electrical input in source')
            self._pool[fingerprint] = evidence

    @property
    def physical_points(self):
        return self._own_actual + self.new_physical_points

    @property
    def total_physical_points(self):
        return self.payload['parent_guard_actual_points'] + self.physical_points

    @property
    def new_physical_points(self):
        return self._live.physical_points if self._live is not None else 0

    def activate(self, factory, stack, allowance):
        require(self.mode == 'REPLAY' and self.requests == len(self._requests)
                and self.points == len(self._logical), 'Incomplete or overlong causal replay')
        require(self._own_actual == self.payload['parent_early_actual_points'], 'Parent actual ownership changed')
        require(type(allowance) is int and 0 <= allowance <= min(128, 512-self.total_physical_points),
                'Fork actual allowance exceeds parent remaining capacity')
        self.mode, self._factory, self._stack, self._allowance = 'FORK', factory, stack, allowance
        self._base = {'logical_points': self.points, 'requests': self.requests, 'hits': self.hits,
                      'early_actual_points': self._own_actual, 'global_actual_points': self.total_physical_points}

    def _historical_alias(self, evidence, identity, seconds):
        raw = evidence['raw']
        row = {**physical_evidence_payload(raw), 'time_minutes': seconds/60., 'state_identity': identity}
        certify_clpu_point(row, self.payload['physical_metadata'], self.payload['physical_config'])
        require(row['status'] == raw['status'] and row['reliable'] is True,
                'Current-identity recertification changed parent evidence status')
        return certificate(row, self.points)

    def _record(self, row):
        self.records.append(row)
        self.journal({'kind': 'ORACLE_REQUEST_COMPLETE', 'record': deepcopy(row)})

    def certify(self, state, seconds):
        identity = {**self.env.identity(state, seconds), **self.binding}
        semantic = sha_value(identity)
        self.requests += 1
        if self.mode == 'REPLAY':
            require(self.requests <= len(self._requests), 'Replay requested future parent answer')
            req = self._requests[self.requests-1]
            require(req['key'] == semantic and req['status'] in ('PASS', 'FAIL'), 'Source request identity/status changed')
            row = self._logical[req['logical_point_id']]
            require(row['semantic_identity'] == identity and row['semantic_identity_sha256'] == semantic
                    and row['service_sequence'] <= self.payload['cutoff_service_sequence'], 'Source semantic history changed')
            if semantic in self.cache:
                self.hits += 1
                require(req['semantic_cache_hit'] is True and self.cache[semantic] == row['certificate'], 'Source hit changed')
            else:
                require(req['semantic_cache_hit'] is False and row['logical_point_id'] == self.points, 'Source logical order changed')
                self.points += 1
                self._own_actual += int(row['reuse_kind'] == 'ACTUAL_PHYSICAL')
                self.cache[semantic] = deepcopy(row['certificate'])
            return deepcopy(self.cache[semantic])
        require(self.mode == 'FORK', 'Unknown prefix oracle mode')
        if semantic in self.cache:
            self.hits += 1
            result = deepcopy(self.cache[semantic])
            self._record({'request': self.requests, 'semantic_identity_sha256': semantic,
                'kind': 'SEMANTIC_HIT', 'logical_point_id': result['point_id'], 'new_physical_points': 0})
            return result
        if self.points >= self.budget:
            self._record({'request': self.requests, 'semantic_identity_sha256': semantic, 'kind': 'LOGICAL_BUDGET_UNKNOWN'})
            raise CLPUBudgetExhausted('Parent cumulative logical budget exhausted')
        return self._new_semantic(state, seconds, identity, semantic)

    def _new_semantic(self, state, seconds, identity, semantic):
        before = self.new_physical_points
        key = electrical_input_key(self.env, state, seconds, self.scope)
        fingerprint = sha_value(key)
        evidence = self._pool.get(fingerprint)
        if evidence is not None:
            require(key == evidence['electrical_input_key'], 'Full historical electrical key differs')
            result = self._historical_alias(evidence, identity, seconds)
            reference = {'namespace': 'PARENT', 'physical_point_id': evidence['physical_point_id'],
                'source_event_sequence': evidence['source_event_sequence'], 'raw_sha256': evidence['raw_sha256']}
            kind = 'PARENT_ELECTRICAL_ALIAS'
        else:
            if self._allowance == 0:
                self._record({'request': self.requests, 'semantic_identity_sha256': semantic, 'kind': 'ACTUAL_BUDGET_UNKNOWN'})
                raise CLPUBudgetExhausted('No new actual point allowance')
            if self._live is None:
                self._live = self._stack.enter_context(self._factory(self._allowance))
                require(self._live.points == self._live.physical_points == self._live.requests == 0,
                        'Live fork view must be cold; sibling answers forbidden')
                require(self._live.budget == 256 and self._live.service.physical_budget == self._allowance,
                        'Live fork budgets differ')
                require(self._live.service.scope == self.scope, 'Live scientific scope differs')
            try:
                local = self._live.certify(state, seconds)
            except CLPUBudgetExhausted:
                self._record({'request': self.requests, 'semantic_identity_sha256': semantic,
                    'kind': 'ACTUAL_BUDGET_UNKNOWN', 'new_physical_points': self.new_physical_points-before})
                raise
            result = {**deepcopy(local), 'point_id': self.points}
            logical = self._live._rows[local['point_id']]
            reference = {'namespace': 'FORK', 'logical_point_id': local['point_id'],
                         'physical_point_id': logical['physical_point_id']}
            kind = 'FORK_LIVE_VIEW'
        require(result['status'] in ('PASS', 'FAIL') and result['reliable'] is True, 'Unreliable fork result')
        self._record({'request': self.requests, 'kind': kind, 'semantic_identity': identity,
            'semantic_identity_sha256': semantic, 'electrical_input_key': key,
            'electrical_input_sha256': fingerprint, 'logical_point_id': self.points,
            'certificate': deepcopy(result), 'evidence_reference': reference,
            'new_physical_points': self.new_physical_points-before})
        self.cache[semantic] = deepcopy(result)
        self.points += 1
        return result

    def summary(self):
        live = self._live.service.summary() if self._live is not None else None
        return {'version': VERSION, 'mode': self.mode, 'inheritance_policy': deepcopy(INHERITANCE_POLICY),
            'parent_paid': deepcopy(self._base), 'cumulative_logical_points': self.points,
            'cumulative_requests': self.requests, 'cumulative_hits': self.hits,
            'new_actual_points': self.new_physical_points, 'new_point_allowance': self._allowance,
            'new_numerical_entries': live['actual_numerical_entries'] if live else 0,
            'live_initialization_occurred': self._live is not None, 'live_service': live,
            'historical_restore_new_physical_points': 0}


class SupportRepairSearch(GuardFirstStagedSearch):
    def __init__(self, *args, **kwargs):
        self.repair_generations = []
        self.repair_enabled = False
        super().__init__(*args, **kwargs)

    def add_repairs(self, parent_record, parent_metadata, generation_round):
        require(generation_round == len(self.repair_generations)+1 <= 2, 'Repair call order or cap changed')
        require(not self.done or self.stop_reason in REOPENABLE, 'Budget/reliability/bound stop cannot reopen')
        before = (self.oracle.points, self.oracle.physical_points, self.oracle.requests)
        generation = generate_support_repair_candidates(self.env, parent_record, parent_metadata,
                                                       self.stage_generation, generation_round)
        existing = {tuple(n['info']['actions']) for n in self._frontier.values()} | set(self._checked)
        admitted = []
        for candidate in generation['candidates']:
            actions = tuple(candidate['actions'])
            if actions in existing:
                admitted.append({'candidate_id': candidate['candidate_id'], 'status': 'ALREADY_KNOWN_COMPLETE_PLAN',
                                 'actions': list(actions), 'branch_id': None})
                continue
            branch = self._next_id
            self._add(candidate['prefix'], actions, ORIGIN, parent_record.get('details', {}).get('branch_id'))
            require(self._next_id == branch+1, 'Legal unique repair was not admitted')
            metadata = deepcopy(candidate)
            metadata.update(generated_candidate_id=candidate['candidate_id'], candidate_id=branch)
            self.stage_branch_metadata[branch] = metadata
            admitted.append({'candidate_id': candidate['candidate_id'], 'status': 'ADMITTED',
                             'actions': list(actions), 'branch_id': branch})
            existing.add(actions)
        prior_stop = self.stop_reason
        self.repair_generations.append({'round': generation_round, 'after_verify_attempts': self.verify_attempts,
            'prior_stop_reason': prior_stop, 'generation': generation, 'admissions': admitted})
        if self.done and self.stop_reason in REOPENABLE and self.verification_queue:
            self.done, self.stop_reason = False, None
        self._refresh_stop()
        require(before == (self.oracle.points, self.oracle.physical_points, self.oracle.requests),
                'Pure repair generation spent physical/logical requests')
        return self.repair_generations[-1]

    def _holds_and_neighbors(self, node, *, failed_step):
        if not self.repair_enabled or len(self.repair_generations) != 1:
            return
        metadata = self.stage_branch_metadata[node['id']]
        if metadata.get('origin') == ORIGIN and metadata.get('repair_round') == 1:
            record = self._checked[node['info']['actions']]
            require(record['status'] == 'FAIL', 'Only reliable FAIL triggers round2')
            self.add_repairs(record, metadata, 2)

    def result(self):
        result = super().result()
        result.update(support_teacher_version=VERSION, repair_generations=deepcopy(self.repair_generations),
                      repair_enabled=self.repair_enabled)
        return result


class SupportVerificationAdapter(GuardFirstVerificationAdapter):
    def _window(self):
        rows = sorted(self._frontier.values(), key=lambda n: (n['info']['priority'], n['id']))
        if self.engine.verify_attempts % 2 == 1 and self.incumbent is not None:
            energy = self.incumbent['validation']['energy_kwh']
            selected = [n for n in rows if n['info']['energy_kwh'] > energy
                        and self.engine.stage_branch_metadata[n['id']]['held_out_loads']]
            if selected:
                rows = selected
        return rows[:32]

    def metadata(self):
        result = super().metadata()
        result.update(support_teacher_version=VERSION,
            exposure='all_pending_retained_original_complete_priority_then_id_first32',
            hidden_plan_policy='not_deleted_can_be_exposed_after_later_verification',
            repair_calls_limit=2,
            window_rule={'ordinary': 'all_pending_sorted_original_complete_priority_then_stable_id_first32',
                'coverage_turn': 'cumulative_parent_plus_fork_verify_attempts_mod_2_equals_1',
                'eligible': 'still_held_out_and_graph_energy_strictly_above_effective_certified_incumbent',
                'no_eligible_fallback': 'all_pending_sorted_original_complete_priority_then_stable_id_first32',
                'capacity': 32, 'max_generated_candidates': 5000})
        return result


def compact_observation(observation):
    return {k: observation[k] for k in ('representation', 'candidate_features', 'global_features', 'mask', 'branch_ids')}


def scientific(value):
    """Only nondeterministic measurement clocks are excluded from replay equality."""
    if isinstance(value, dict):
        return {k: scientific(v) for k, v in value.items() if k not in ('elapsed_seconds',)}
    if isinstance(value, (tuple, list)):
        return [scientific(v) for v in value]
    return value


def replay_parent(env, payload):
    oracle = CausalPrefixOracle(env, payload)
    search = SupportRepairSearch(env, oracle, BranchConfig(**payload['branch_config']),
        guard_result=payload['guard_result'], guard_reference=payload['guard_reference'])
    control = GuardFirstMacroController(search, method='beam')
    require(len(payload['decisions']) == payload['completed_early_decisions'], 'Source choice prefix length changed')
    for pair in payload['decisions']:
        proposal = control.prepare()
        require(proposal is not None, 'Source ended before declared parent')
        saved = pair['started']
        compact = {k: v for k, v in proposal.items() if k != 'observation'}
        require(canonical(compact) == canonical(saved['proposal']), 'Restored source decision changed')
        require(canonical(compact_observation(proposal['observation'])) == canonical(saved['obs_compact']),
                'Restored source observation changed')
        response = control.apply(proposal)
        require(scientific(response) == scientific(pair['completed']['response']), 'Restored source transition changed')
    require(oracle.requests == len(payload['request_prefix']) and oracle.points == len(payload['logical_prefix']),
            'Source history contains unseen future or missing answers')
    require(oracle.physical_points == payload['parent_early_actual_points']
            and oracle.total_physical_points == payload['parent_global_actual_points'], 'Source paid counters changed')
    require(search.verify_attempts == payload['parent_verify_attempts'], 'Parent VERIFY counter changed')
    parent = search._checked.get(tuple(payload['parent_actions']))
    require(parent == payload['parent_record'] and parent['status'] == 'FAIL', 'Parent reliable failure changed')
    require(search.effective_incumbent['validation']['energy_kwh'] == payload['parent_effective_energy_kwh'],
            'Parent effective certificate changed')
    search.repair_enabled = True
    search.add_repairs(parent, search.stage_branch_metadata[payload['parent_branch_id']], 1)
    return search, oracle


def select_candidates(search, context_id, state_id):
    adapter = SupportVerificationAdapter(search)
    observation = adapter.observation('raw')
    if search.done or len(observation['mask']) < 3:
        return []
    rows = adapter._window()
    selected, roles = [], {}
    def add(slot, role):
        branch = rows[slot]['id']
        if branch not in roles:
            selected.append(branch); roles[branch] = []
        roles[branch].append(role)
    add(complete_energy_action(observation), 'strong_energy')
    add(min(range(len(rows)), key=lambda i: (rows[i]['info']['priority'], rows[i]['id'])), 'greedy_full_priority')
    def hashed(node):
        value = {'context_id': context_id, 'state_id': state_id, 'actions': list(node['info']['actions'])}
        return hashlib.sha256((CANDIDATE_HASH_DOMAIN+canonical(value)).encode()).hexdigest(), node['id']
    for node in sorted(rows, key=hashed):
        if len(selected) == 3:
            break
        if node['id'] not in roles:
            add(next(i for i, row in enumerate(rows) if row['id'] == node['id']), 'hash')
    by_id = {n['id']: n for n in rows}
    return [{'slot': index, 'branch_id': branch, 'actions': list(by_id[branch]['info']['actions']),
        'plan_sha256': sha_value(list(by_id[branch]['info']['actions'])), 'selection_roles': roles[branch]}
        for index, branch in enumerate(selected)]


def state_digest(search):
    return sha_value({'own_incumbent': search.incumbent, 'own_priority': search._incumbent_key,
        'guard_priority': search.guard_floor_key, 'frontier': search.generation_trace,
        'checked': sorted((list(k), v) for k, v in search._checked.items()),
        'pending': sorted((list(k), v) for k, v in search.verification_queue.items()),
        'free_pruning': search.free_pruning, 'repairs': search.repair_generations,
        'logical_points': search.oracle.points, 'requests': search.oracle.requests,
        'actual_early_points': search.oracle.physical_points, 'verify_attempts': search.verify_attempts,
        'done': search.done, 'stop_reason': search.stop_reason})


def fork_snapshot(env, payload, selected, live_factory, *, check_deadline=lambda: None, journal=lambda event: None):
    """Restore exactly, force one visible candidate, then use the common rule."""
    import time
    began = time.perf_counter()
    search, oracle = replay_parent(env, payload)
    digest = state_digest(search)
    require(digest == payload['restored_state_sha256'], 'Prepared causal state changed')
    require(selected in select_candidates(search, payload['context_id'], payload['state_id']), 'Unfrozen forced candidate')
    oracle.journal = journal
    journal({'kind': 'PARENT_RESTORED', 'restored_state_sha256': digest, 'new_physical_points': 0})
    restore_seconds = time.perf_counter()-began
    baseline = payload['parent_effective_energy_kwh']
    q0 = 100.*baseline/search.denominator
    allowance = min(128, 512-payload['parent_global_actual_points'])
    remaining = min(16, 16-payload['parent_verify_attempts'])
    require(remaining > 0 and oracle.points < 256 and not search.done, 'Parent has no admitted resource opportunity')
    records, curve, g1, g1_kwh = [], [], None, None
    fork_started = time.perf_counter()
    with ExitStack() as stack:
        oracle.activate(live_factory, stack, allowance)
        for index in range(remaining):
            if search.done:
                break
            check_deadline()
            adapter = SupportVerificationAdapter(search)
            observation = adapter.observation('raw')
            require(observation['mask'], 'Active support search has no exposed candidate')
            slot = observation['branch_ids'].index(selected['branch_id']) if index == 0 else complete_energy_action(observation)
            branch = observation['branch_ids'][slot]
            mapping = adapter.window_mapping()
            hidden = sorted(set(adapter._frontier)-set(observation['branch_ids']))
            before = {'logical_points': oracle.points, 'physical_points': oracle.new_physical_points, 'requests': oracle.requests}
            response = search.verify(branch)
            current = search.effective_incumbent['validation']['energy_kwh']
            gain = 100.*(current-baseline)/search.denominator
            if index == 0:
                g1, g1_kwh = gain, current-baseline
            records.append({'fork_choice': index, 'branch_id': branch, 'action': slot,
                'forced': index == 0, 'observation': compact_observation(observation),
                'window_mapping': mapping, 'hidden_pending_branch_ids': hidden,
                'hidden_reason': 'common_coverage_filter_or_complete_priority_window32',
                'oracle_before': before, 'response': response})
            journal({'kind': 'FORK_VERIFY_COMPLETE', 'record': records[-1]})
            curve.append({'completed_fork_verifications': index+1, 'new_physical_points': oracle.new_physical_points,
                'cumulative_logical_points': oracle.points, 'quality_pct': 100.*current/search.denominator,
                'gH_pp': gain, 'fork_elapsed_seconds': time.perf_counter()-fork_started})
        if not search.done:
            search.yield_to_beam()
        require(search.stop_reason != 'DEFERRED_RELIABILITY_OR_IMPLEMENTATION_ERROR', 'Erroneous fork cannot be completed')
    final = search.effective_incumbent['validation']['energy_kwh']
    require(g1 is not None, 'Forced first verification was not completed')
    return {'version': VERSION, 'status': 'COMPLETE_BOUNDED_SUPPORT_TEACHER_FORK',
        'state_id': payload['state_id'], 'context_id': payload['context_id'],
        'selected_branch_id': selected['branch_id'], 'selected_plan_sha256': selected['plan_sha256'],
        'selection_roles': selected['selection_roles'], 'parent_quality_pct': q0,
        'g1_pp': g1, 'gH_pp': 100.*(final-baseline)/search.denominator,
        'g1_kwh': g1_kwh, 'gH_kwh': final-baseline,
        'final_quality_pct': 100.*final/search.denominator, 'new_physical_points': oracle.new_physical_points,
        'new_logical_points': oracle.points-len(payload['logical_prefix']),
        'new_requests': oracle.requests-len(payload['request_prefix']),
        'restoration_seconds': restore_seconds, 'fork_elapsed_seconds': time.perf_counter()-fork_started,
        'verification_allowance': remaining, 'new_actual_allowance': allowance,
        'restored_state_sha256': digest, 'oracle': oracle.summary(), 'oracle_records': oracle.records,
        'decisions': records, 'quality_curve': curve, 'search_result': search.result(),
        'new_training': 0, 'new_restore_physical_points': 0, 'scope': 'offline_bounded_teacher_not_new_deployment'}
