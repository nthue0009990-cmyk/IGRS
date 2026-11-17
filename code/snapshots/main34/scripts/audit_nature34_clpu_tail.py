"""Independent saved-evidence audit of the Beam-protected CLPU pilot.

No solver is initialized.  The physical equations and original Beam replay are
the unchanged D059 independent auditors; the continuation is reconstructed here
without importing its production implementation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
sys.dont_write_bytecode = True

import audit_nature34_clpu_branch as branch
from audit_d060_validation import inside, manifest, verify_manifest

old = branch.old
require, close, read, save, sha, canonical, read_lines, same = (
    branch.require, branch.close, branch.read, branch.save, branch.sha,
    branch.canonical, branch.read_lines, branch.same)
SOURCE_RUN = '20261024T073617_269327Z_nature34_clpu_branch'
SOURCE_SHA = '4c3777f938d21f074b40db56c82cd62ca1b956f6aa7d79215832bf192745c56f'
SEEDS = (17, 41, 73)
WORST_CONTEXT = {'scenario_id': 'N2__line.l13__line.l24', 'event_id': 12}


def baseline_snapshot(replay):
    """Original search-result schema independently reconstructed from evidence."""
    status = ('INITIAL_FAIL' if replay.reason == 'INITIAL_FAIL' else
              'CERTIFIED_INCUMBENT' if replay.best else 'NO_CERTIFIED_COMPLETE_PLAN')
    return {'status': status, 'method': 'shared_clpu_branch_search', 'config': replay.cfg,
            'done': replay.done, 'stop_reason': replay.reason,
            'initial_certificate': replay.initial_certificate, 'best_plan': replay.best,
            'incumbent': replay.best, 'attempts': replay.attempts[:replay.attempt_index],
            'trace': replay.trace, 'generation_events': replay.generation_events,
            'generation_trace': replay.generated, 'graph_assessment_trace': replay.assessment_trace,
            'generated_unique_candidates': len(replay.keys), 'assessed_graph_candidates': len(replay.assessed),
            'checked_candidates': len(replay.checked), 'frontier_size': len(replay.frontier),
            'generation_cap_rejections': replay.rejections, 'actual_physical_points': replay.oracle.used,
            'maximum_reachable_graph_kw': replay.m.max_kw, 'global_upper_bound_kwh': replay.m.bound,
            'optimality_established': replay.reason == 'CERTIFIED_FRONTIER_UPPER_BOUND',
            'optimality_scope': 'normal-service objective only; declared snapshot model',
            'scope': 'Known-curve full-window dynamic branch search; not rolling control'}


def check_protection(baseline, result):
    """Protect a complete certificate, not a scalar copied from another run."""
    require(baseline['done'] is True and baseline['status'] == 'CERTIFIED_INCUMBENT',
            'Beam baseline did not finish with a complete incumbent')
    before, after = baseline['incumbent'], result['incumbent']
    require(before is not None and after is not None and result['best_plan'] == after,
            'Missing protected baseline/final plan')
    for plan in (before, after):
        require(plan['status'] == 'COMPLETE' and plan['certified_complete'] is True
                and plan['validation']['status'] == 'COMPLETE', 'Incomplete protected certificate')
    require(float(after['validation']['energy_kwh']) >= float(before['validation']['energy_kwh']),
            'Final certified quality is below the paid Beam baseline')
    return float(after['validation']['energy_kwh']) - float(before['validation']['energy_kwh'])


def check_cumulative_caps(result, baseline, cfg):
    limits = {'actual_physical_points': cfg['branch_search']['point_budget'],
              'checked_candidates': cfg['branch_search']['max_checked_candidates'],
              'generated_unique_candidates': cfg['branch_search']['max_generated_candidates'],
              'assessed_graph_candidates': cfg['branch_search']['max_generated_candidates']}
    for key, limit in limits.items():
        require(type(result[key]) is int and baseline[key] <= result[key] <= limit,
                'Cumulative resource reset or limit exceeded: ' + key)
    require(len(baseline['trace']) <= cfg['branch_search']['max_branch_expansions'],
            'Original Beam expansion limit exceeded')


def check_physics(run, cell, spec, cfg, mech, result):
    """Reuse all original independent AC/graph/device/certificate accounting."""
    points, counts = branch.check_raw_points(run, cell, spec, cfg, mech)
    commands = old.check_commands(cell, points, read(run / 'command_inventory.json'))
    requests = old.check_requests(cell, points, spec['point_budget'])
    oracle = read(cell / 'oracle_summary.json')
    require(result['oracle'] == oracle and oracle['status'] == 'COMPLETE_COUNTERS_PENDING_AUDIT'
            and oracle['points'] == len(points) and oracle['point_budget'] == spec['point_budget']
            and oracle['requests'] == requests['requests'] and oracle['cache_hits'] == requests['cache_hits']
            and oracle['statuses'] == dict(counts) and oracle['actual_numerical_entries'] == commands['numerical_entries']
            and oracle['maximum_numerical_entries'] == 3 + 4 * spec['point_budget']
            and oracle['point_seconds'] == [p['elapsed_seconds'] for p in points.values()],
            'Oracle budget/cost/timing differs')
    require(list(read_lines(cell / 'plans.jsonl.gz')) == result['attempts'], 'Omitted paid candidate')
    return points, counts, commands, requests


def check_transition(replay, row, decisions, spec, index, stage_index):
    phase = replay.phase
    require(row['transition_id'] == index and row['stage_transition_id'] == stage_index
            and row['stage'] == phase, 'Stage/global transition chronology differs')
    method = 'beam' if phase == 'BASELINE' else 'tail_priority' if phase == 'VERIFY' else spec['method']
    require(row['method'] == method and row['representation'] == 'raw' and row['seed'] == spec['seed']
            and row['sampling_seed'] == spec.get('sampling_seed', spec['seed']), 'Actual stage policy attribution differs')
    policy = method in ('bc_raw', 'ppo_raw')
    require(row['checkpoint_sha256'] == (spec.get('checkpoint_sha256') if policy else None),
            'Baseline/verification or fixed continuation model misattributed')
    for representation in ('raw', 'associated'):
        same(row['obs_' + representation], replay.observation(representation), 'observation_' + representation)
    require(row['obs'] == row['obs_raw'] and row['mask'] == [True] * len(replay.window()),
            'Policy received hidden input or changed mask')
    require(row['oracle_before'] == replay.oracle.counts(), 'Unrecorded physical query before decision')
    selected = row['action']
    if not policy:
        require(selected == replay.rule(method), 'Strong rule/verification order changed')
        require(row['logprob'] is None and row['value'] is None, 'Rule fabricated learned statistics')
    require(type(selected) is int and 0 <= selected < len(replay.window())
            and row['branch_id'] == replay.window()[selected]['id'], 'Selected unexposed branch/index')
    started = {k: v for k, v in row.items() if k not in ('reward', 'done', 'info', 'oracle_after')}
    require(next(decisions, None) == {**started, 'status': 'STARTED'}, 'Started decision differs from transition')
    response = replay.step(selected)
    same(row['info'], {k: v for k, v in response.items() if k not in ('reward', 'done')}, 'step_info')
    close(row['reward'], response['reward'], 'Reward contains unobserved outcome or wrong cost')
    require(row['done'] == response['done'] and row['oracle_after'] == replay.oracle.counts(),
            'Terminal flag or after-query count differs')
    completed = {'transition_id': index, 'stage_transition_id': stage_index, 'stage': phase,
                 'status': 'COMPLETED', 'action': selected, 'branch_id': row['branch_id'],
                 'reward': row['reward'], 'done': row['done'], 'info': row['info'],
                 'oracle_before': row['oracle_before'], 'oracle_after': row['oracle_after']}
    require(next(decisions, None) == completed, 'Completed journal differs from transition')


def object_sha(value):
    import json
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class TailCore(branch.BranchReplay):
    """Independent continuation modifications to the original independent Beam."""
    owner = None

    def window(self):
        if self.owner is None:
            return super().window()
        if self.owner.phase == 'VERIFY':
            return self.owner.tail_candidates()[:1]
        available = {key: value for key, value in self.frontier.items()
                     if self.owner.match(value['info']['actions']) is None}
        if not available:
            return super().window()
        original = self.frontier
        try:
            self.frontier = available
            return super().window()
        finally:
            self.frontier = original

    def evaluate(self, info, origin, details):
        cert = self.owner.match(info['actions']) if self.owner is not None else None
        if cert is None:
            return super().evaluate(info, origin, details)
        self.owner.reuses += 1
        self.owner.step_reuse = cert['certificate_id']
        return {'actions': list(info['actions']), 'origin': origin, 'details': deepcopy(details),
                'status': 'FAIL', 'certified_complete': False,
                'validation': {'status': 'FAIL', 'failed_step': cert['failed_step'], 'feedback': cert['feedback']},
                'physical_points_before': self.oracle.used, 'physical_points_after': self.oracle.used,
                'prefix_certificate_id': cert['certificate_id']}


class TailReplay:
    def __init__(self, mech, cfg, oracle, attempts, identity, continuation, events,
                 cache_hits_reported=True):
        self.m, self.cfg, self.oracle = mech, cfg, oracle
        self.continuation, self.phase = continuation, 'BASELINE'
        self.events, self.event_index = events, 0
        self.certificates, self.indexed = [], 0
        self.match_cache = {}
        self.reuses = self.repairs = self.continued = self.verified = 0
        self.step_reuse, self.baseline = None, None
        self.cache_hits_reported = cache_hits_reported
        self.identity = {'physics_identity': identity, 'model_version': 'nc34-clpu-ed-snapshot-v1',
            'static': mech.static, 'loads': mech.loads, 'scenario': mech.scenario, 'condition': mech.condition,
            'event': mech.event, 'initial_state': mech.initial, 'environment_initial': mech.initial,
            'horizon': mech.horizon, 'slot_seconds': mech.slot, 'sample_seconds': mech.sample, 'load_names': mech.names}
        self.identity_sha = object_sha(self.identity)
        self.core = TailCore(mech, deepcopy(cfg['branch_search']), oracle, attempts)
        if self.core.done:
            self.finish_baseline()

    @property
    def done(self):
        return self.phase == 'DONE'

    def emit(self, expected):
        require(self.event_index < len(self.events), 'Missing chronological tail event')
        actual = self.events[self.event_index]
        expected = {'tail_event_id': self.event_index, **expected}
        if expected['kind'] == 'SEARCH_STEP':
            duration = actual.get('duration_seconds')
            require(type(duration) in (int, float) and math.isfinite(duration) and duration >= 0,
                    'Missing/nonfinite measured step duration')
            expected['duration_seconds'] = duration
            self.check_counters(actual['counters'])
            expected['counters'] = actual['counters']
        same(actual, expected, 'tail_event')
        self.event_index += 1

    def oracle_counters(self):
        hits = sum(bool(r.get('cache_hit')) for r in self.oracle.rows[:self.oracle.index])
        return {'points': self.oracle.used, 'requests': self.oracle.requests,
                'cache_hits': hits if self.cache_hits_reported else None}

    def finish_baseline(self):
        core = self.core
        original = deepcopy(baseline_snapshot(core))
        valid = original['status'] == 'CERTIFIED_INCUMBENT' and original['best_plan']['certified_complete'] is True
        self.baseline = {'result': original, 'identity_sha256': self.identity_sha, 'guarantee_established': valid,
            'oracle_counters': self.oracle_counters(), 'replay_state': {
                'start': self.m.initial, 'frontier_branch_ids': sorted(core.frontier),
                'expanded_prefixes': [list(v) for v in sorted(core.expanded)],
                'branch_keys': [[list(a), list(b)] for a, b in sorted(core.keys)],
                'beam_turn': core.turn, 'next_id': len(core.keys),
                'note': 'Full candidates and checked plans are in result; graph states reconstruct from frozen environment without physical calls.'}}
        self.baseline_sha = object_sha(self.baseline)
        can_continue = self.continuation and valid and core.reason == 'BRANCH_EXPANSION_CAP' and (
            self.oracle.used < self.cfg['branch_search']['point_budget'] and
            len(core.checked) < self.cfg['branch_search']['max_checked_candidates'])
        self.emit({'kind': 'BASELINE_SEALED', 'stage': 'BASELINE', 'baseline_sha256': self.baseline_sha,
                   'guarantee_established': valid, 'continued': can_continue,
                   'oracle_counters': self.baseline['oracle_counters']})
        if not can_continue:
            self.phase = 'DONE'
            return
        allowance = self.cfg['tail_search']['continuation_expansions']
        core.cfg = {**core.cfg, 'max_branch_expansions': core.turn + allowance}
        core.owner = self
        core.done, core.reason = False, None
        self.phase = 'CONTINUATION' if allowance else 'VERIFY'
        self.index_failures()
        self.refresh()

    def environment_identity(self, state, seconds):
        identity = self.m.identity(state, seconds, self.oracle.condition_sha)
        return {k: v for k, v in identity.items() if k not in ('alpha', 'base_condition_sha256')}

    def index_failures(self):
        while self.indexed < self.core.attempt_index:
            index = self.indexed
            attempt = self.core.attempts[index]
            self.indexed += 1
            if attempt['status'] != 'FAIL':
                continue
            validation = attempt['validation']
            failed = validation['failed_step']
            require(type(failed) is int and failed >= 0 and validation['samples'], 'FAIL lacks exact failed prefix')
            sample = validation['samples'][-1]
            point = self.oracle.points[sample['point_id']]
            require(sample['status'] == point['status'] == 'FAIL' and point['reliable'] is True
                    and sample['point_id'] < self.oracle.used, 'Unpaid/unreliable/future prefix FAIL')
            prefix = attempt['actions'][:failed + 1]
            state = self.m.initial
            for action in prefix:
                state = self.m.advance(state, action)
            identity = self.environment_identity(state, sample['time_seconds'])
            require(sample['state'] == state and self.m.identity(state, sample['time_seconds'], self.oracle.condition_sha)
                    == point['state_identity'], 'Exact prefix and physical history disagree')
            if any(r['actions_prefix'] == prefix and r['identity'] == identity for r in self.certificates):
                continue
            witness = {'point_id': sample['point_id'], 'status': 'FAIL', 'reliable': True,
                       'feedback': point['structured_feedback']['violations']}
            evidence = {'certificate_id': len(self.certificates), 'attempt_index': index, 'failed_step': failed,
                'actions_prefix': prefix, 'point_id': sample['point_id'], 'identity': identity,
                'scope_sha256': self.identity_sha, 'certificate': witness, 'feedback': validation.get('feedback', [])}
            self.certificates.append(evidence)
            self.emit({'kind': 'PREFIX_CERTIFICATE_INDEXED', 'stage': self.phase, **evidence})

    def match(self, actions):
        key = (tuple(actions), len(self.certificates))
        if key in self.match_cache:
            return self.match_cache[key]
        for evidence in self.certificates:
            prefix = evidence['actions_prefix']
            if list(actions[:len(prefix)]) != prefix:
                continue
            state = self.m.initial
            for action in prefix:
                state = self.m.advance(state, action)
            if self.environment_identity(state, evidence['identity']['time_seconds']) == evidence['identity']:
                self.match_cache[key] = evidence
                return evidence
        self.match_cache[key] = None
        return None

    def tail_candidates(self):
        seen, rows = set(self.core.checked), []
        for row in sorted(self.core.frontier.values(), key=lambda value: (value['info']['priority'], value['id'])):
            actions = row['info']['actions']
            if actions in seen:
                continue
            seen.add(actions)
            if self.core.best and row['info']['priority'] >= self.m.plan(self.core.best['actions'])['priority']:
                continue
            if self.match(actions) is None:
                rows.append(row)
        return rows

    def refresh(self):
        core = self.core
        if self.phase == 'CONTINUATION' and core.done:
            if core.reason == 'BRANCH_EXPANSION_CAP':
                self.phase = 'VERIFY'
                core.done, core.reason = False, None
            else:
                self.phase = 'DONE'
        if self.phase == 'VERIFY':
            if self.oracle.used >= self.cfg['branch_search']['point_budget']:
                core.stop('PHYSICAL_BUDGET_EXHAUSTED')
            elif len(core.checked) >= self.cfg['branch_search']['max_checked_candidates']:
                core.stop('LOGICAL_CHECK_CAP')
            elif not self.tail_candidates():
                core.stop('TAIL_VERIFICATION_EXHAUSTED')
            if core.done:
                self.phase = 'DONE'

    def counts(self):
        return {'physical_points': self.oracle.used, 'checked_candidates': len(self.core.checked),
                'expansions': self.core.turn, 'generated': len(self.core.keys), 'graph_assessed': len(self.core.assessed)}

    def window(self):
        return self.core.window()

    def observation(self, representation):
        return self.core.observation(representation)

    def rule(self, method):
        if self.phase == 'VERIFY':
            require(method == 'tail_priority' and self.tail_candidates(), 'Invalid deterministic tail selection')
            return 0
        return self.core.rule(method)

    def step(self, action):
        core, phase, before = self.core, self.phase, self.counts()
        self.step_reuse = None
        if phase == 'BASELINE':
            require(action == core.rule('beam'), 'Original baseline Beam order changed')
            event = core.step(action)
            core_done = event['done']
            if core.done:
                self.finish_baseline()
        elif phase == 'CONTINUATION':
            cert = self.match(core.window()[action]['info']['actions'])
            event = core.step(action)
            core_done = event['done']
            self.continued += 1
            if cert is not None:
                self.repairs += 1
                self.step_reuse = cert['certificate_id']
            self.index_failures()
            self.refresh()
        else:
            require(phase == 'VERIFY' and action == 0, 'Invalid tail verification stage/action')
            node = self.tail_candidates()[0]
            quality = core.quality
            record = core.evaluate(node['info'], 'tail_verify',
                                   {'branch_id': node['id'], 'stage': 'VERIFY', 'expansion_performed': False})
            self.verified += 1
            event = {'branch_id': node['id'], 'actions': list(node['info']['actions']), 'query_reason': 'tail_verify',
                'validation_status': record['status'], 'quality_before': quality, 'quality_after': core.quality,
                'physical_points_before': before['physical_points'], 'physical_points_after': self.oracle.used,
                'physical_points_used': self.oracle.used - before['physical_points'], 'new_branches': 0,
                'done': core.done, 'stop_reason': core.reason}
            event['reward'] = core.quality - quality - core.cfg['reward_cost_weight'] * (
                self.oracle.used - before['physical_points']) / core.cfg['reward_cost_normalizer']
            event['info'] = {'validation_status': event['validation_status'],
                             'physical_points_used': event['physical_points_used'],
                             'query_reason': 'tail_verify', 'stop_reason': core.reason}
            core_done = core.done
            self.index_failures()
            self.refresh()
        operation = 'VERIFY' if phase == 'VERIFY' else 'REPAIR_ONLY' if self.step_reuse is not None else 'EXPAND'
        self.emit({'kind': 'SEARCH_STEP', 'stage': phase, 'operation': operation, 'branch_id': event['branch_id'],
            'prefix_certificate_id': self.step_reuse, 'before': before, 'after': self.counts(),
            'core_event': event, 'phase_after': self.phase})
        return {**deepcopy(event), 'stage': phase, 'operation': operation, 'core_done': core_done,
                'done': self.done, 'phase_after': self.phase}

    def check_counters(self, counters):
        expected = {'certificate_reuses': self.reuses, 'repair_only_expansions': self.repairs,
                    'continuation_expansions': self.continued, 'tail_verifications': self.verified}
        for key, value in expected.items():
            require(counters[key] == value, 'Counter hides a continuation/repair/verification operation: ' + key)
        for key in ('screen_calls', 'screen_cache_hits', 'screen_computations', 'screen_matches', 'identity_checks'):
            require(type(counters[key]) is int and counters[key] >= 0, 'Invalid measured logical count')
        require(counters['screen_calls'] == counters['screen_cache_hits'] + counters['screen_computations']
                and counters['screen_matches'] <= counters['screen_calls'], 'Screening accounting does not close')
        for key in ('screen_seconds', 'identity_check_seconds', 'step_seconds'):
            require(type(counters[key]) in (int, float) and math.isfinite(counters[key]) and counters[key] >= 0,
                    'Invalid measured component duration')
        if hasattr(self, 'last_counters'):
            require(all(counters[k] >= self.last_counters[k] for k in counters), 'Counter/time reset between phases')
        self.last_counters = deepcopy(counters)


def check_tail_records(mech, cfg, result, baseline, checkpoint, transitions, decisions,
                       request_rows, points, condition_sha, identity, stage_events,
                       cache_hits_reported=True):
    oracle = branch.PaidRequests(request_rows, points, mech, cfg['branch_search']['point_budget'], condition_sha)
    tail = result['tail']
    replay = TailReplay(mech, cfg, oracle, result['attempts'], identity,
                        result['spec']['tail_group'] != 'G0', tail['events'], cache_hits_reported)
    spec, decisions = result['spec'], iter(decisions)
    stage_index, stages, previous, count = 0, Counter(), None, 0
    checkpoint_checked = False
    def stage_entry(index):
        nonlocal previous, stage_index, checkpoint_checked
        if replay.phase == previous:
            return
        if previous == 'BASELINE' or (previous is None and replay.phase != 'BASELINE'):
            require(not checkpoint_checked and replay.baseline is not None, 'Missing/repeated baseline seal')
            same(baseline, replay.baseline['result'], 'complete_original_Beam_baseline')
            expected = {**replay.baseline, 'sha256': replay.baseline_sha}
            same(checkpoint['engine_checkpoint'], expected, 'baseline_replay_checkpoint')
            require(checkpoint['transition_boundary'] == index
                    and checkpoint['attempt_boundary'] == len(baseline['attempts'])
                    and checkpoint['oracle_boundary'] == oracle.counts(), 'Baseline phase boundary changed')
            require(math.isfinite(checkpoint['elapsed_seconds']) and checkpoint['elapsed_seconds'] >= 0,
                    'Invalid measured baseline duration')
            checkpoint_checked = True
        require(stage_index < len(stage_events), 'Missing stage entry journal')
        event = stage_events[stage_index]
        require(set(event) == {'stage', 'transition_boundary', 'oracle', 'utc'}
                and event['stage'] == replay.phase and event['transition_boundary'] == index
                and event['oracle'] == oracle.counts(), 'Stage entry, phase state or physical boundary differs')
        datetime.fromisoformat(event['utc'])
        stage_index += 1
        previous = replay.phase
    for index, row in enumerate(transitions):
        stage_entry(index)
        require(not replay.done, 'Decision after protected search termination')
        phase = replay.phase
        check_transition(replay, row, decisions, spec, index, stages[phase])
        stages[phase] += 1
        count += 1
    stage_entry(count)
    require(replay.done and checkpoint_checked and stage_index == len(stage_events), 'Unfinished or omitted phase')
    require(next(decisions, None) is None, 'Unclosed or omitted policy decision')
    require(replay.event_index == len(replay.events), 'Unbound tail events')
    require(oracle.index == len(oracle.rows) and oracle.used == len(points), 'Unbound paid physical queries')
    require(replay.core.attempt_index == len(result['attempts']), 'Unqueried or omitted complete-plan attempts')
    expected_core = baseline_snapshot(replay.core)
    for key, value in expected_core.items():
        same(result[key], value, 'final_core.' + key)
    require(result['search_decisions'] == count and result['stage_decisions'] == dict(stages),
            'Selected-stage operation counts changed')
    require(replay.continued <= cfg['tail_search']['continuation_expansions']
            and replay.core.turn == stages['BASELINE'] + stages['CONTINUATION'] <= 512
            and replay.verified == stages['VERIFY'], 'Sequential expansion or verification budget reset')
    replay.check_counters(tail['counters'])
    close(tail['counters']['step_seconds'], sum(e['duration_seconds'] for e in tail['events']
          if e['kind'] == 'SEARCH_STEP'), 'Measured search step time omitted')
    expected_tail = {'version': 'protected-beam-tail-v1', 'phase': replay.phase,
        'baseline_result': replay.baseline['result'], 'baseline_sha256': replay.baseline_sha,
        'guarantee_established': replay.baseline['guarantee_established'],
        'identity_binding': replay.identity, 'identity_sha256': replay.identity_sha,
        'prefix_certificates': replay.certificates, 'events': tail['events'], 'counters': tail['counters'],
        'original_config': cfg['branch_search'],
        'additional_expansion_allowance': cfg['tail_search']['continuation_expansions'],
        'accounting': 'Original checks include WAIT/FAIL/partial; certified-prefix reuse is separate; all expansions, screening and time charged. No cross-episode cache.'}
    same(tail, expected_tail, 'protected_tail_result')
    for phase, times in result.get('stage_component_seconds', {}).items():
        require(phase in stages and set(times) == {'observation_seconds', 'selection_seconds', 'search_step_seconds', 'journal_seconds'}
                and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in times.values()),
                'Measured stage component time invalid')
    return {'decisions': count, 'baseline_decisions': stages['BASELINE'],
            'continuation_decisions': stages['CONTINUATION'], 'verification_decisions': stages['VERIFY'],
            'prefix_certificates': len(replay.certificates), 'repair_only_expansions': replay.repairs,
            'certificate_reuses': replay.reuses, 'stop_reason': replay.core.reason,
            'complete_baseline_replay': True, 'all_original_physical_checks_retained': True,
            'logical_telemetry_check': 'integer/nonnegative/monotone/component closure; measured runtime values are not re-timed'}


def independent_contexts(source_schedule):
    """Deterministic selection from actual D059 train tasks, before outcomes."""
    train = [row for row in source_schedule if row['phase'] == 'train']
    faults = sorted({r['scenario_id'] for r in train}, key=lambda value:
                    hashlib.sha256(('ppo-tail-pilot-v1|fault|' + value).encode()).hexdigest())[:2]
    events = sorted({r['event_id'] for r in train}, key=lambda value:
                    hashlib.sha256(('ppo-tail-pilot-v1|event|' + str(value)).encode()).hexdigest())[:2]
    require(len(faults) == len(events) == 2, 'Insufficient inherited training population')
    return [{'scenario_id': f, 'event_id': e} for f in faults for e in events]


def check_population(schedule, contexts):
    require(len(schedule) == 40 and len({r['cell'] for r in schedule}) == 40,
            'Pilot must retain all 40 independent tasks')
    expected_contexts = {(r['scenario_id'], r['event_id']) for r in contexts}
    require(len(expected_contexts) == 5 and
            {(r['scenario_id'], r['event_id']) for r in schedule} == expected_contexts,
            'Pilot context population changed')
    expected_groups = {('G0', 'beam', 0), ('G1', 'beam', 0)} | {
        (group, method, seed) for group, method in [('G2', 'bc_raw'), ('G3', 'ppo_raw')] for seed in SEEDS}
    for context in expected_contexts:
        rows = [r for r in schedule if (r['scenario_id'], r['event_id']) == context]
        require(len(rows) == 8 and {(r['tail_group'], r['method'], r['seed']) for r in rows} == expected_groups,
                'Missing/duplicated method or seed')
    for index, row in enumerate(schedule):
        require(row['episode_id'] == index and row['cell'] == f'episodes/{index:04d}'
                and row['phase'] == 'evaluation' and row['representation'] == 'raw'
                and row['deterministic'] is True and row['point_budget'] == 256,
                'Frozen task identity or resources changed')
    return schedule


def check_parent(run, completed=False):
    run = Path(run).resolve()
    require(run.parent == (ROOT / 'runs').resolve(), 'Run outside project runs')
    prepared = read(run / 'prepared.json')
    require(prepared['status'] == 'FROZEN_PREPARED_NO_DSS' and prepared['run'] == run.name,
            'Missing exact prepared identity')
    require(sha(run / 'prephysical_freeze_manifest.json') == prepared['freeze_sha256'], 'Changed freeze identity')
    frozen = verify_manifest(run, 'prephysical_freeze_manifest.json')
    sources = {name.replace('\\', '/') for name in prepared['source_files']}
    required = {'scripts/audit_nature34_clpu_tail.py', 'scripts/audit_nature34_clpu_branch.py',
                'scripts/audit_nature34_clpu_pilot.py', 'scripts/audit_nature34_joint_planning.py',
                'scripts/audit_nature34_clpu_branch_learning.py', 'tests/test_nature34_clpu_tail_audit.py',
                'scripts/audit_d060_validation.py', 'scripts/audit_d059_supplement.py'}
    require(required <= sources, 'Independent audit sources are not frozen')
    for name in sources:
        require('source/' + name in frozen and sha(inside(ROOT, name)) == frozen['source/' + name],
                'Executing implementation differs from frozen copy: ' + name)
    from nature34_experiment_support import dependencies
    require(dependencies() == read(run / 'dependencies.json') and
            sha(Path(sys.executable)) == read(run / 'python_environment.json')['sha256'],
            'Frozen dependency/interpreter changed')
    cfg = read(run / 'config.json')
    require(cfg['decision'] == 'D059' and cfg['validation_study'] == 'PPO_TAIL_PILOT_V1', 'Wrong pilot scope')
    for key, expected in [('horizon', 30), ('slot_seconds', 60), ('sample_seconds', 30), ('alpha', .5), ('algorithm', 0)]:
        require(cfg[key] == expected, 'Physical/time configuration changed: ' + key)
    for name in ('formal_holdout_numerical_access', 'tune_numerical_access', 'test_access', 'tune_access'):
        require(cfg.get(name, False) is False, 'Forbidden numerical access')
    source = ROOT / 'runs' / SOURCE_RUN
    require(sha(source / 'completion_manifest.json') == SOURCE_SHA, 'Original D059 seal changed')
    source_manifest = manifest(read(source / 'completion_manifest.json'))
    def original(name):
        require(name in source_manifest and sha(source / name) == source_manifest[name], 'Changed original input: ' + name)
        return read(source / name)
    for name in original('prepared.json')['source_files']:
        relative = name.replace('\\', '/')
        if not relative.endswith('.py'):
            continue
        require('source/' + relative in source_manifest
                and sha(inside(ROOT, relative)) == source_manifest['source/' + relative],
                'Original scientific implementation changed: ' + relative)
    require(cfg['branch_search'] == original('config.json')['branch_search'], 'Original Beam configuration changed')
    require(cfg['tail_search'] == {'continuation_expansions': 256, 'enable_exact_failed_prefix': True,
                                   'enable_tail_verification': True}, 'Continuation configuration differs')
    for path in (run / 'source_inputs').rglob('*'):
        if path.is_file():
            relative = path.relative_to(run).as_posix()
            require(relative in source_manifest and sha(path) == source_manifest[relative],
                    'Copied original input changed: ' + relative)
    for name in ('author_static_metadata.json', 'load_reference.json', 'physical_config.json', 'condition.json',
                 'operating_reference.json', 'command_inventory.json', 'event_split.json'):
        original(name)
        require(sha(run / name) == source_manifest[name], 'Inherited physical input changed: ' + name)
    training = independent_contexts(original('schedule.json'))
    contexts = [{**r, 'cohort': 'training_mechanism'} for r in training] + [
        {**WORST_CONTEXT, 'cohort': 'selected_development_regression'}]
    require(read(run / 'evaluation_contexts.json') == contexts and read(run / 'training_contexts.json') == [],
            'Result-dependent pilot selection or unexpected training')
    schedule = check_population(read(run / 'schedule.json'), contexts)
    for row in schedule:
        expected = 'selected_development_regression' if (row['scenario_id'], row['event_id']) == tuple(WORST_CONTEXT.values()) else 'training_mechanism'
        require(row['cohort'] == expected, 'Training/regression cohort relabeled')
    unordered = [{k: v for k, v in r.items() if k not in ('episode_id', 'cell')} for r in schedule]
    require(unordered == sorted(unordered, key=lambda r: hashlib.sha256(
        ('ppo-tail-order-v1|' + canonical(r)).encode()).hexdigest()), 'Pilot execution order changed')
    all_scenarios = {r['scenario_id']: r for r in original('selected_scenarios.json')}
    selected = read(run / 'selected_scenarios.json')
    require(len(selected) == 3 and {r['scenario_id'] for r in selected} == {r['scenario_id'] for r in contexts}
            and all(r == all_scenarios[r['scenario_id']] for r in selected), 'Selected fault metadata changed')
    from audit_d060_validation import CURVE_ARCHIVE, CURVE_SHA
    import zipfile
    archive = ROOT / CURVE_ARCHIVE
    require(sha(archive) == CURVE_SHA, 'CLPU original archive changed')
    with zipfile.ZipFile(archive) as stream:
        parameters = [s.strip() for s in stream.read('parameters.csv').decode('utf-8-sig').splitlines() if s.strip()]
    events = read(run / 'selected_events.json')
    require(len(events) == 3 and {r['event_id'] for r in events} == {r['event_id'] for r in contexts}, 'CLPU event population changed')
    for event in events:
        tokens = [s.strip() for s in parameters[event['event_id']].split(',')]
        require(event['parameter_row'] == parameters[event['event_id']], 'Paired CLPU source row changed')
        close(event['a'], float(tokens[0] + '.' + tokens[1]), 'Original paired ED amplitude')
        close(event['tau_seconds'], float(tokens[2] + '.' + tokens[3]), 'Original paired ED decay')
    endpoints, expected_models = original('final_checkpoints.json'), []
    for seed in SEEDS:
        for method, suffix in [('bc_raw', 'raw/start.pt'), ('ppo_raw', 'raw/ppo_round07/after.pt')]:
            name = f'learning/seed{seed}/' + suffix
            require(endpoints[str(seed)][method].replace('\\', '/') == name and name in source_manifest
                    and sha(source / name) == source_manifest[name], 'Inherited BC/PPO endpoint differs')
            expected_models.append({'seed': seed, 'method': method, 'source_relative': name,
                'source_sha256': source_manifest[name], 'checkpoint': f'models/seed{seed}/{method}.pt',
                'learning_audit_binding': 'all 49 sealed source learning audits and final attribution'})
    require(read(run / 'model_bindings.json') == expected_models, 'Six fixed endpoint bindings differ')
    for model in expected_models:
        require(sha(inside(run, model['checkpoint'])) == model['source_sha256'], 'Copied endpoint changed')
    binding = read(run / 'source_binding.json')
    require(binding['source_run'] == SOURCE_RUN and binding['source_completion_sha256'] == SOURCE_SHA,
            'Original run binding changed')
    from audit_d060_validation import SUPPLEMENT_RUN, SUPPLEMENT_SHA
    supplement = ROOT / 'runs' / SUPPLEMENT_RUN
    require(sha(supplement / 'completion_manifest.json') == SUPPLEMENT_SHA, 'Original independent supplement changed')
    supplement_manifest = manifest(read(supplement / 'completion_manifest.json'))
    for name in ('learning_evidence_validation.json', 'final_attribution.json'):
        require(sha(supplement / name) == supplement_manifest[name], 'Original endpoint attribution changed')
    attribution = read(supplement / 'final_attribution.json')
    require(attribution['status'] == 'VERIFIED_CLPU_BRANCH_LEARNING_ATTRIBUTION' and attribution['errors'] == []
            and attribution['same_seed_bc_ppo_start'], 'Inherited endpoints lack independent attribution')
    require(binding['supplement_run'] == SUPPLEMENT_RUN and binding['supplement_completion_sha256'] == SUPPLEMENT_SHA,
            'Supplement binding changed')
    d060 = inside(ROOT / 'runs', binding['d060_run'])
    require(sha(d060 / 'completion_manifest.json') == binding['d060_completion_sha256'], 'Original D060 seal changed')
    diagnosis_path = inside(ROOT, binding['diagnosis_summary'])
    require(sha(diagnosis_path) == binding['diagnosis_summary_sha256'], 'Original tail diagnosis changed')
    diagnosis = read(diagnosis_path)
    require(binding['diagnosis_inputs'] == diagnosis['input_sha256'], 'Original tail input binding differs')
    for name, digest in binding['diagnosis_inputs'].items():
        require(sha(inside(ROOT, name)) == digest, 'Original result/source seal changed: ' + name)
    for row in diagnosis['result_sources'].values():
        path, seal = inside(ROOT, row['path']), inside(ROOT, row['completion_manifest_path'])
        require(sha(path) == row['sha256'] == read(seal)['result.json'], 'Original worst-case result was modified')
    population = read(run / 'source_population.json')
    require(population['source_run'] == SOURCE_RUN and population['training_contexts'] == original('training_contexts.json')
            and population['training_events'] == original('training_events.json')
            and population['selected_contexts'] == contexts, 'Original population binding differs')
    for name, digest in population['source_sha256'].items():
        require(source_manifest[name] == digest and sha(inside(source, name)) == digest, 'Original selection input changed')
    resource = read(run / 'resource_budget.json')
    require(resource['episodes'] == 40 and resource['allocated_physical_points'] == 10240
            and resource['maximum_numerical_entries'] == 40 * 3 + 4 * 10240
            and resource['max_workers'] == 2 and resource['automatic_retry'] is False
            and resource['new_training_tasks'] == resource['new_optimizer_updates'] == 0,
            'Frozen total resources or training scope changed')
    require(sum(row['point_budget'] for row in schedule) == 10240, 'Global physical allocation changed')
    if completed:
        require(all((run / row['cell'] / 'completion_manifest.json').is_file() for row in schedule),
                'Incomplete pilot cannot be reported verified')
    return cfg


def check_cell(run, cell, cfg):
    run, cell = Path(run).resolve(), Path(cell).resolve()
    require(cell != run and cell.is_relative_to(run), 'Cell outside declared run')
    require(not (cell / 'error.json').exists(), 'Cell reliability error retained')
    seals = verify_manifest(cell, 'completion_manifest.json')
    required = {'spec.json', 'result.json', 'points.jsonl.gz', 'plans.jsonl.gz', 'transitions.jsonl.gz',
                'decisions.jsonl.gz', 'requests.jsonl', 'command_ledger.jsonl', 'initial_state.json',
                'metadata.json', 'oracle_summary.json', 'command_stats.json', 'baseline_result.json',
                'baseline_checkpoint.json', 'physical_identity.json', 'stage_events.jsonl', 'tail_events.jsonl'}
    require(required <= seals.keys(), 'Cell has unsealed required evidence')
    spec, result = read(cell / 'spec.json'), read(cell / 'result.json')
    templates = {row['cell']: row for row in read(run / 'schedule.json')}
    relative = cell.relative_to(run).as_posix()
    require(relative in templates, 'Unscheduled cell')
    template = templates[relative]
    extras = {'template_sha256', 'deadline_utc_epoch', 'checkpoint', 'checkpoint_sha256'}
    require({k: v for k, v in spec.items() if k not in extras} == template and
            spec['template_sha256'] == hashlib.sha256(canonical(template).encode()).hexdigest(),
            'Cell differs from frozen schedule')
    require(type(spec['deadline_utc_epoch']) in (int, float) and math.isfinite(spec['deadline_utc_epoch']),
            'Missing runtime deadline')
    require(result['spec'] == spec and result['formal_holdout_numerical_access'] is False
            and result['tune_numerical_access'] is False and result['clpu_holdout_access'] is False
            and result['optimizer_updates'] == result['training_interactions'] == 0,
            'Cell scope or in-worker learning differs')
    bindings = [row for row in read(run / 'model_bindings.json')
                if row['method'] == spec['method'] and row['seed'] == spec['seed']]
    if spec['tail_group'] in ('G0', 'G1'):
        require(not bindings and not spec.get('checkpoint') and not spec.get('checkpoint_sha256'),
                'Rule secretly loaded model')
    else:
        require(len(bindings) == 1 and spec['checkpoint'] == bindings[0]['checkpoint']
                and spec['checkpoint_sha256'] == bindings[0]['source_sha256']
                and sha(inside(run, spec['checkpoint'])) == spec['checkpoint_sha256'], 'Wrong fixed model endpoint')
    scenarios = {s['scenario_id']: s for s in read(run / 'selected_scenarios.json')}
    events = {e['event_id']: e for e in read(run / 'selected_events.json')}
    mech = branch.Mechanics(read(run / 'author_static_metadata.json'), read(run / 'load_reference.json'),
        scenarios[spec['scenario_id']], read(run / 'condition.json'), events[spec['event_id']], cfg)
    require(read(cell / 'initial_state.json') == mech.initial, 'Initial state/history differs')
    identity = {'study': 'PPO_TAIL_PILOT_V1', 'freeze_sha256': sha(run / 'prephysical_freeze_manifest.json'),
                'scenario_id': spec['scenario_id'], 'event_id': spec['event_id'], 'initial_state': mech.initial,
                'condition_sha256': sha(run / 'condition.json'),
                'physical_config_sha256': sha(run / 'physical_config.json'),
                'author_static_metadata_sha256': sha(run / 'author_static_metadata.json'),
                'source_provenance_sha256': sha(run / 'source_provenance.json')}
    require(read(cell / 'physical_identity.json') == identity, 'Physical prefix scope differs')
    points, counts, commands, requests = check_physics(run, cell, spec, cfg, mech, result)
    baseline = read(cell / 'baseline_result.json')
    require(list(read_lines(cell / 'tail_events.jsonl')) == result['tail']['events'], 'Missing immediate tail event record')
    checkpoint = read(cell / 'baseline_checkpoint.json')
    require(checkpoint['baseline_result_sha256'] == result['baseline_result_sha256'] == sha(cell / 'baseline_result.json')
            and result['baseline_checkpoint_sha256'] == sha(cell / 'baseline_checkpoint.json')
            and checkpoint['physical_identity_sha256'] == sha(cell / 'physical_identity.json'),
            'Immutable Beam checkpoint binding changed')
    replay = check_tail_records(mech, cfg, result, baseline, checkpoint,
        read_lines(cell / 'transitions.jsonl.gz'), read_lines(cell / 'decisions.jsonl.gz'),
        list(read_lines(cell / 'requests.jsonl')), points, sha(run / 'condition.json'), identity,
        list(read_lines(cell / 'stage_events.jsonl')))
    policy = None
    if spec['tail_group'] in ('G2', 'G3'):
        from audit_nature34_clpu_branch_learning import check_policy_transitions
        def policy_rows():
            for row in read_lines(cell / 'transitions.jsonl.gz'):
                if row['stage'] == 'CONTINUATION':
                    yield {**row, 'transition_id': row['stage_transition_id']}
        policy = check_policy_transitions(inside(run, spec['checkpoint']), policy_rows(), spec)
    check_cumulative_caps(result, baseline, cfg)
    established = baseline['status'] == 'CERTIFIED_INCUMBENT'
    gain = check_protection(baseline, result) if established else None
    if not established:
        require(spec['tail_group'] == 'G0' or replay['continuation_decisions'] == replay['verification_decisions'] == 0,
                'Continued despite missing certified Beam baseline')
        require(result['best_plan'] == baseline['best_plan'], 'Missing baseline fabricated final certificate')
        guard = {'guard_established': False, 'baseline_status': baseline['status'],
                 'reason': 'NO_COMPLETE_BASELINE_CERTIFICATE'}
    else:
        guard = {'guard_established': True, 'baseline_status': baseline['status'],
                 'baseline_energy_kwh': float(baseline['best_plan']['validation']['energy_kwh']),
                 'final_energy_kwh': float(result['best_plan']['validation']['energy_kwh']),
                 'nondegradation_verified': True,
                 'scope': 'Fixed 30-minute normal-service planning objective; computation waiting is not in the objective'}
    same(result['guard_audit'], guard, 'guard_claim')
    return {'cell': relative, 'spec': spec, 'status': result['status'], 'points': len(points),
            'physical_statuses': dict(counts), 'commands': commands, 'requests': requests, 'branch': replay,
            'policy': policy, 'guard_established': established, 'continuation_gain_kwh': gain,
            'plan_statuses': dict(Counter(a['status'] for a in result['attempts'])),
            'cell_completion_sha256': sha(cell / 'completion_manifest.json')}


def audit_cells(run, output=None):
    run = Path(run).resolve()
    output = (Path(output).resolve() if output else run / 'audit')
    require(output != run and output.is_relative_to(ROOT.resolve()), 'Audit output outside project')
    output.mkdir(parents=True, exist_ok=False)
    began, checks = time.perf_counter(), []
    result = {'status': 'FAILED_CLPU_TAIL_PILOT', 'source_run': run.name, 'errors': [],
              'new_powerflow_calls': 0, 'new_training_interactions': 0,
              'formal_holdout_numerical_access': False, 'tune_numerical_access': False}
    from audit_d059_supplement import NoSolverImports, no_solver_loaded
    guard = NoSolverImports()
    sys.meta_path.insert(0, guard)
    try:
        cfg = check_parent(run, completed=True)
        schedule = read(run / 'schedule.json')
        for row in schedule:
            checked = check_cell(run, run / row['cell'], cfg)
            checks.append(checked)
            save(output / f'cell_{row["episode_id"]:04d}.json', checked)
            save(output / 'progress.json', {'completed': len(checks), 'scheduled': 40, 'errors': [],
                                           'elapsed_seconds': time.perf_counter() - began})
        no_solver_loaded()
        require(sum(c['points'] for c in checks) <= 10240, 'Global physical budget exceeded')
        # All eight independently paid baselines in each context must coincide.
        for context in read(run / 'evaluation_contexts.json'):
            rows = [r for r in schedule if (r['scenario_id'], r['event_id']) ==
                    (context['scenario_id'], context['event_id'])]
            canonical_baseline = read(run / rows[0]['cell'] / 'baseline_result.json')
            for row in rows[1:]:
                same(read(run / row['cell'] / 'baseline_result.json'), canonical_baseline,
                     'independently_paid_original_Beam')
        result.update(status='VERIFIED_CLPU_TAIL_PILOT', episodes=len(checks),
            physical_points=sum(c['points'] for c in checks), numerical_entries=sum(c['commands']['numerical_entries'] for c in checks),
            physical_statuses=dict(sum((Counter(c['physical_statuses']) for c in checks), Counter())),
            guard_established=sum(c['guard_established'] for c in checks),
            freeze_sha256=sha(run / 'prephysical_freeze_manifest.json'),
            schedule_sha256=sha(run / 'schedule.json'), dependencies_sha256=sha(run / 'dependencies.json'))
    except BaseException as exc:
        result['errors'].append({'error': repr(exc), 'traceback': traceback.format_exc()})
    finally:
        sys.meta_path.remove(guard)
    result['elapsed_seconds'] = time.perf_counter() - began
    save(output / 'cell_checks.json', checks)
    save(output / 'verification.json', result)
    save(output / 'completion_manifest.json', {str(p.relative_to(output)): sha(p) for p in output.rglob('*')
                                              if p.is_file() and p.name != 'completion_manifest.json'})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit_cells(args.run, args.output)
    print(canonical(result), flush=True)
    return 0 if not result['errors'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
