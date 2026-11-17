"""Independent saved-evidence audit of coverage-window teacher experiments.

No production window/counterfactual module or solver is imported. Physical
answers, graph/search evolution, complete-certificate timing and the discrete
budget curve are reconstructed independently from paid, sealed records.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
sys.dont_write_bytecode = True
import audit_clpu_window_pilot as window
import audit_clpu_counterfactual_pilot as previous

branch, old = window.branch, window.old
require, close, same = window.require, window.close, window.same
read, save, sha, canonical, read_lines = window.read, window.save, window.sha, window.canonical, window.read_lines
manifest, digest = window.manifest, previous.digest
VERSION = 'clpu-window-counterfactual-prefix-v1'
WINDOW_VERSION = 'alternating_unique_hold_coverage_v1'
SOURCE_RUN = '20261027T121023_469270Z_clpu_window_pilot'
SOURCE_SHA = '9967ea771263b5b9bb45835116dec1c5fd70f8908c17ec2348d90d4a6377fc7e'


def eligibility(replay):
    """Only the current queue and already certified incumbent inform selection."""
    if replay.done or replay.best is None:
        return {'eligible': False, 'mode': None, 'completed_steps': replay.turn,
                'candidate_ids': [], 'reason': 'TERMINAL_OR_NO_CERTIFIED_INCUMBENT'}
    energy = replay.best['validation']['energy_kwh']
    hold = any(row['origin'] == 'prefix_wait' and row['info']['actions'] not in replay.checked
               and row['info']['energy_kwh'] > energy for row in replay.frontier.values())
    mode = 'coverage' if replay.turn % 2 and hold else 'normal'
    ids, seen = [], set()
    for row in replay.window():
        actions = row['info']['actions']
        if actions not in seen and actions not in replay.checked and row['info']['energy_kwh'] > energy:
            seen.add(actions)
            ids.append(row['id'])
    return {'eligible': len(ids) >= 2, 'mode': mode, 'completed_steps': replay.turn,
            'candidate_ids': ids, 'reason': None if len(ids) >= 2 else 'FEWER_THAN_TWO_UNTESTED_IMPROVING_PLANS'}


def independent_auc(initial, trajectory, allowance):
    """Dense integer-budget reference; b=0 completions count at every b>=1."""
    require(type(allowance) is int and allowance >= 0, 'Invalid curve allowance')
    require(type(initial) in (int, float) and math.isfinite(initial) and 0 <= initial <= 1,
            'Invalid initial certified quality')
    require(trajectory and trajectory[0] == {'new_physical_points': 0, 'quality': initial,
            'kind': 'INITIAL_CERTIFIED_INCUMBENT'}, 'Missing initial quality boundary')
    last_point, last_quality = 0, initial
    for row in trajectory[1:]:
        point, quality = row['new_physical_points'], row['quality']
        require(type(point) is int and last_point <= point <= allowance
                and type(quality) in (int, float) and math.isfinite(quality)
                and last_quality <= quality <= 1 and row['kind'] == 'COMPLETE_CERTIFICATE',
                'Invalid or uncertified curve boundary')
        last_point, last_quality = point, quality
    if allowance == 0:
        return None
    values = [next(row['quality'] for row in reversed(trajectory) if row['new_physical_points'] <= b)
              for b in range(1, allowance + 1)]
    return 100 * math.fsum(value - initial for value in values) / allowance


class CertificateReplay(window.WindowReplay):
    """Record quality only after independent full-plan validation completes."""
    collecting = False

    def begin_collection(self):
        self.collecting = True
        self.base_points = self.oracle.used
        self.expansion_index = 0
        self.updates = []
        self.trajectory = [{'new_physical_points': 0, 'quality': self.quality,
                            'kind': 'INITIAL_CERTIFIED_INCUMBENT'}]

    def evaluate(self, info, origin, details):
        count, before = self.attempt_index, self.quality
        value = super().evaluate(info, origin, details)
        if self.collecting and self.attempt_index != count and value['status'] == 'COMPLETE':
            point = self.oracle.used - self.base_points
            quality = value['validation']['energy_kwh'] / self.m.denominator
            self.updates.append({'expansion_index': self.expansion_index, 'new_physical_points': point,
                'cumulative_physical_points': self.oracle.used, 'quality_before': before,
                'quality_after': self.quality, 'plan_quality': quality, 'actions_sha256': digest(value['actions'])})
            self.trajectory.append({'new_physical_points': point, 'quality': self.quality,
                                    'kind': 'COMPLETE_CERTIFICATE'})
        return value


def completed_steps(journal):
    """Strictly paired source boundary records, not fork compact records."""
    require(len(journal) % 2 == 0, 'Source journal contains an unclosed step')
    result = []
    for index in range(len(journal) // 2):
        started, finished = journal[2 * index:2 * index + 2]
        require(started['status'] == 'STARTED' and finished['status'] == 'COMPLETED'
                and started['transition_id'] == finished['transition_id'] == index,
                'Source journal order changed')
        for key in ('method', 'action', 'branch_id', 'window_branch_ids', 'oracle_before'):
            same(finished[key], started[key], 'source_boundary.' + key)
        response = finished['response']
        require(started['method'] == 'beam' and started['action'] == response['selected_index']
                and started['branch_id'] == response['branch_id']
                and started['window_branch_ids'] == response['window_branch_ids']
                and started['oracle_before']['points'] == response['physical_points_before']
                and finished['oracle_after']['points'] == response['physical_points_after'],
                'Source journal and search response disagree')
        result.append(finished)
    return result


def check_records(mech, cfg, payload, result, source_steps, source_plans, prefix_points,
                  source_requests, new_points, new_requests, condition_sha, *, forced_branch_id,
                  max_new_points=128, max_expansions=16, plan_journal=None, step_journal=None):
    """Audit exact source-prefix recovery followed by one independently paid fork."""
    require(payload['version'] == result['version'] == VERSION
            and payload['window_rule_version'] == result['window_rule_version'] == WINDOW_VERSION
            and payload['source_method'] == 'beam' and payload['source_window_variant'] == 'coverage_v1',
            'Teacher version or frozen window changed')
    require(payload['config'] == cfg, 'Teacher scientific configuration changed')
    steps = completed_steps(source_steps)
    position = payload['completed_steps']
    require(type(position) is int and 0 <= position <= len(steps)
            and result['source_completed_steps'] == position, 'Unavailable snapshot position')
    require(payload['source_bootstrap_attempt'] == source_plans[0]
            and payload['source_trace'] == [row['response'] for row in steps[:position]],
            'Snapshot source history differs')
    n = (steps[position - 1]['oracle_after']['points'] if position else source_plans[0]['physical_points_after'])
    require(len(prefix_points) == len(payload['prefix_points']) == n
            and set(prefix_points) == set(range(n)), 'Future or incomplete physical prefix')
    for index, row in prefix_points.items():
        compact = payload['prefix_points'][index]
        require(compact['identity'] == row['state_identity']
                and compact['identity_sha256'] == digest(row['state_identity'])
                and compact['certificate'] == {'point_id': index, 'status': row['status'],
                    'reliable': row['reliable'], 'feedback': row['structured_feedback']['violations']},
                'Snapshot physical answer differs from historical paid evidence')
    allowance = min(max_new_points, cfg['point_budget'] - n)
    require(type(max_new_points) is int and 0 <= max_new_points <= 128
            and type(max_expansions) is int and 1 <= max_expansions <= 16,
            'Changed local counterfactual cap')
    oracle = previous.ForkRequests(source_requests, prefix_points, new_requests, new_points,
                                   mech, condition_sha, allowance)
    replay = CertificateReplay(mech, cfg, oracle, source_plans)
    for saved in steps[:position]:
        require(not replay.done and saved['window_branch_ids'] == [row['id'] for row in replay.window()]
                and saved['action'] == replay.rule('beam'), 'Source queue or rule differs')
        same(replay.step(saved['action']), saved['response'], 'source_replay_response')
    require(payload['source_attempts'] == source_plans[:replay.attempt_index],
            'Snapshot includes future cache-only plans or omits source plans')
    same(payload['raw_observation'], replay.observation('raw'), 'snapshot_observation')
    require(payload['prefix_oracle_counters'] == {'points': oracle.used, 'requests': oracle.requests,
            'cache_hits': oracle.hits}, 'Snapshot accounting differs')
    require(not replay.done and replay.best and replay.best['certified_complete'],
            'Snapshot is terminal or lacks a certified incumbent')
    oracle.activate()
    replay.begin_collection()
    initial, first = replay.quality, None
    replay.attempts = source_plans[:replay.attempt_index] + result['new_attempts']
    if plan_journal is not None:
        same(plan_journal, result['new_attempts'], 'complete_plan_journal')
    transitions = result['transitions']
    if step_journal is not None:
        same(step_journal, transitions, 'fork_step_journal')
    expansions = min(max_expansions, cfg['max_branch_expansions'] - position)
    require(1 <= len(transitions) <= expansions, 'Forced step absent or excess continuation')
    require(result['forced_branch_id'] == forced_branch_id, 'Forced candidate changed')
    for index, row in enumerate(transitions):
        require(row['transition_id'] == index and row['status'] == 'COMPLETED'
                and row['forced'] == (index == 0) and not replay.done, 'Fork chronology changed')
        observation = replay.observation('raw')
        same(row['obs_raw'], observation, 'fork_observation')
        require(digest(observation) == row['observation_sha256'] and row['oracle_before'] == oracle.counts(),
                'Fork observation or initial charges changed')
        selected = row['action']
        require(type(selected) is int and 0 <= selected < len(replay.window())
                and replay.window()[selected]['id'] == row['branch_id'], 'Fork selected hidden candidate')
        require((index == 0 and row['branch_id'] == forced_branch_id)
                or (index > 0 and selected == replay.rule('beam')), 'Forced choice or continuation rule changed')
        replay.expansion_index = index
        same(replay.step(selected), row['response'], 'fork_response')
        require(row['oracle_after'] == oracle.counts(), 'Fork completed charges changed')
        if index == 0:
            first = replay.quality
    same(result['fork_trace'], [row['response'] for row in transitions], 'fork_trace')
    require(replay.attempt_index == len(replay.attempts), 'Unconsumed complete or failed plan evidence')
    require(replay.done or len(transitions) == expansions, 'Unexplained early fork termination')
    stop = replay.reason if replay.done else 'FORK_EXPANSION_LIMIT'
    if stop == 'PHYSICAL_BUDGET_EXHAUSTED' and oracle.used < cfg['point_budget']:
        stop = 'FORK_NEW_PHYSICAL_POINT_LIMIT'
    require(result['stop_reason'] == stop and result['core_stop_reason'] == replay.reason,
            'Fork stop reason changed')
    for key, expected in {'new_point_allowance': allowance, 'completed_expansions': len(transitions),
                          'cumulative_expansions': replay.turn,
                          'expansion_allowance_including_forced': expansions}.items():
        require(result[key] == expected, 'Fork count differs: ' + key)
    require(oracle.live.index == len(new_requests) and oracle.live.used == len(new_points),
            'Unbound or hidden new physical records')
    expected = {'cumulative_points': oracle.used, 'cumulative_point_budget': cfg['point_budget'],
        'cumulative_requests': oracle.requests, 'cumulative_cache_hits': oracle.hits,
        'historical_prefix_points': n, 'new_physical_points': len(new_points),
        'new_requests_including_cache': oracle.requests - oracle.base_requests,
        'new_cache_hits': oracle.hits - oracle.base_hits,
        'live_initialization_occurred': bool(new_points), 'live_initialization_attempted': bool(new_points),
        'live_requests': len(new_points), 'historical_restore_new_physical_points': 0,
        'point_id_mappings': oracle.mappings}
    for key, value in expected.items():
        require(result['oracle'][key] == value, 'Fork ledger differs: ' + key)
    previous.check_labels(result, initial, first, replay.quality, replay.best)
    same(result['certificate_updates'], replay.updates, 'complete_certificate_updates')
    same(result['quality_point_trajectory'], replay.trajectory, 'quality_point_trajectory')
    require(result['curve_budget_grid'] == 'integer_new_physical_points_1_through_allowance_inclusive',
            'Curve budget grid changed')
    measured = independent_auc(initial, replay.trajectory, allowance)
    same(result['auc_gain_pp'], measured, 'independent_auc_gain_pp')
    return {'new_physical_points': len(new_points), 'historical_prefix_points': n,
        'completed_expansions': len(transitions), 'g1_pp': result['g1_pp'], 'gH_pp': result['gH_pp'],
        'auc_gain_pp': measured, 'status': 'PASS', 'stop_reason': stop,
        'complete_certificate_updates': len(replay.updates)}


def candidate_subset(replay, state, scenario_id, event_id):
    """Independent Beam / graph-priority / hash subset with full-plan dedup."""
    window_rows = replay.window()
    by_id = {row['id']: row for row in window_rows}
    chosen, by_plan = [], {}

    def add(branch_id, role):
        actions = by_id[branch_id]['info']['actions']
        if actions in by_plan:
            item = by_plan[actions]
            if role not in item['roles']:
                item['roles'].append(role)
        elif len(chosen) < 3:
            item = {'branch_id': branch_id, 'roles': [role]}
            by_plan[actions] = item
            chosen.append(item)

    add(window_rows[replay.rule('beam')]['id'], 'window_beam')
    best = min(state['candidate_ids'], key=lambda bid: (by_id[bid]['info']['priority'], bid))
    add(best, 'graph_greedy')
    salt = f'clpu-window-teacher-v1|{scenario_id}|{event_id}|{state["mode"]}|{state["completed_steps"]}|'
    for bid in sorted(state['candidate_ids'], key=lambda value: (hashlib.sha256((salt + str(value)).encode()).hexdigest(), value)):
        add(bid, 'fixed_hash_fill')
        if len(chosen) == 3:
            break
    return chosen


def source_cell_binding(source, cell_name, expected_root_sha=SOURCE_SHA):
    """Bind a nested seal through its root-sealed independent audit result.

    The historical root seal deliberately omits every completion_manifest.json.
    Its audited cell digest supplies that edge; scientific files must still
    agree with both the root manifest and the independently bound cell seal.
    """
    source = Path(source).resolve()
    cell = (source / cell_name).resolve()
    require(cell.is_relative_to(source) and cell != source, 'Source cell escaped frozen window run')
    require(sha(source / 'completion_manifest.json') == expected_root_sha, 'Historical root seal changed')
    root_manifest = read(source / 'completion_manifest.json')
    audit_path = source / 'audit/summary.json'
    require(root_manifest.get('audit/summary.json') == sha(audit_path), 'Historical audit is not root-bound')
    audited = read(audit_path)
    result = read(cell / 'result.json')
    require(audited['status'] == 'PASS', 'Historical source audit did not pass')
    matches = [row for row in audited['results'] if row['cell'] == cell_name and row['spec'] == result['spec']]
    require(len(matches) == 1, 'Historical audit cell/spec binding absent or ambiguous')
    cell_sha = sha(cell / 'completion_manifest.json')
    require(matches[0]['cell_completion_sha256'] == cell_sha, 'Historical audited cell seal changed')
    cell_manifest = read(cell / 'completion_manifest.json')
    fields = {'result.json': 'source_result_sha256', 'completion_manifest.json': 'source_completion_sha256',
        'points.jsonl.gz': 'source_points_sha256', 'steps.jsonl.gz': 'source_steps_sha256',
        'plans.jsonl.gz': 'source_plans_sha256', 'requests.jsonl': 'source_requests_sha256'}
    binding = {'source_run': SOURCE_RUN, 'source_cell': cell_name}
    for name, field in fields.items():
        actual = sha(cell / name)
        if name != 'completion_manifest.json':
            require(root_manifest.get(cell_name + '/' + name) == cell_manifest.get(name) == actual,
                    'Root/cell scientific file binding differs: ' + name)
        binding[field] = actual
    return binding


def source_data(cell_name, cache):
    if cell_name in cache:
        return cache[cell_name]
    source = ROOT / 'runs' / SOURCE_RUN
    binding = source_cell_binding(source, cell_name)
    cell = source / cell_name
    points, compact = {}, []
    keep = ('status', 'reliable', 'clpu_multipliers', 'action', 'time_minutes', 'state_identity',
            'identity_sha256', 'actual_call_counts', 'dynamic_state', 'elapsed_seconds', 'structured_feedback')
    for row in read_lines(cell / 'points.jsonl.gz'):
        index = len(points)
        require(row['point_id'] == index, 'Historical points not chronologically contiguous')
        points[index] = {key: row[key] for key in keep}
        compact.append({'identity': row['state_identity'], 'identity_sha256': row['identity_sha256'],
            'certificate': {'point_id': index, 'status': row['status'], 'reliable': row['reliable'],
                            'feedback': row['structured_feedback']['violations']}, 'source_row_sha256': digest(row)})
    value = {'result': read(cell / 'result.json'), 'steps': list(read_lines(cell / 'steps.jsonl.gz')),
             'plans': list(read_lines(cell / 'plans.jsonl.gz')), 'requests': list(read_lines(cell / 'requests.jsonl')),
             'points': points, 'compact': compact, 'binding': binding}
    cache[cell_name] = value
    return value


def mechanics(run, spec, cfg):
    scenarios = {row['scenario_id']: row for row in read(run / 'selected_scenarios.json')}
    events = {row['event_id']: row for row in read(run / 'selected_events.json')}
    return branch.Mechanics(read(run / 'author_static_metadata.json'), read(run / 'load_reference.json'),
        scenarios[spec['scenario_id']], read(run / 'condition.json'), events[spec['event_id']], cfg)


def first_eligible_states(mech, cfg, source, condition_sha):
    """Replay in chronology; first-state decisions use no later payoff."""
    oracle = branch.PaidRequests(source['requests'], source['points'], mech, cfg['point_budget'], condition_sha)
    replay = window.WindowReplay(mech, cfg, oracle, source['plans'])
    selected = {}
    source_spec = source['result']['spec']
    for row in completed_steps(source['steps']):
        state = eligibility(replay)
        if state['eligible'] and state['mode'] not in selected:
            selected[state['mode']] = {**state,
                'prefix_physical_points': oracle.used,
                'candidates': candidate_subset(replay, state, source_spec['scenario_id'], source_spec['event_id'])}
        require(not replay.done and row['window_branch_ids'] == [item['id'] for item in replay.window()]
                and row['action'] == replay.rule('beam'), 'Historical source selection changed')
        same(replay.step(row['action']), row['response'], 'historical_state_selection_replay')
    require(replay.done and oracle.index == len(source['requests']) and oracle.used == len(source['points'])
            and replay.attempt_index == len(source['plans']), 'Historical state population replay incomplete')
    return selected


def check_states(run, cfg, cache):
    source = ROOT / 'runs' / SOURCE_RUN
    population = read(run / 'population.json')
    require(population == read(source / 'population.json'), 'Fixed training population changed')
    contexts = population['contexts']
    states, schedule = read(run / 'states.json'), read(run / 'schedule.json')
    require(len(contexts) == 8 and len(states) == 16
            and {row['state_id'] for row in states} == {f'c{i:02d}_{mode}' for i in range(8) for mode in ('normal', 'coverage')},
            'Missing fixed context/mode slot')
    source_specs = [s for s in read(source / 'schedule.json') if s['window_variant'] == 'coverage_v1']
    require(len(source_specs) == 8, 'Frozen coverage source population differs')
    for index, context in enumerate(contexts):
        spec = next(s for s in source_specs if s['context_id'] == index)
        stored = source_data(spec['cell'], cache)
        require(stored['result']['spec'] == spec, 'Historical spec not bound to original schedule')
        selected = first_eligible_states(mechanics(run, spec, cfg), cfg['branch_search'], stored, sha(run / 'condition.json'))
        for mode in ('normal', 'coverage'):
            row = next(s for s in states if s['state_id'] == f'c{index:02d}_{mode}')
            require(row['context_id'] == index and row['scenario_id'] == context['scenario_id']
                    and row['event_id'] == context['event_id'] and row['mode'] == mode
                    and row['source_cell'] == spec['cell'], 'State context/window mode changed')
            forks = [s for s in schedule if s['state_id'] == row['state_id']]
            if mode not in selected:
                require(row['status'] == 'UNAVAILABLE' and not row['candidates'] and not forks
                        and row.get('completed_steps') is None
                        and row['reason'] == 'NO_ELIGIBLE_' + mode.upper() + '_STATE',
                        'Unavailable state was replaced or fabricated')
                continue
            expected = selected[mode]
            require(row['status'] == 'AVAILABLE' and row['completed_steps'] == expected['completed_steps']
                    and row['prefix_physical_points'] == expected['prefix_physical_points'],
                    'Snapshot is not the first eligible historical state')
            same(row['candidates'], expected['candidates'], 'outcome_blind_candidate_subset')
            require(len(forks) == len(row['candidates'])
                    and {f['forced_branch_id'] for f in forks} == {c['branch_id'] for c in row['candidates']},
                    'Missing, duplicated, or additional measured candidate')
            snapshot_path = (run / row['snapshot']).resolve()
            require(snapshot_path.is_relative_to(run), 'Snapshot path escaped teacher run')
            snapshot = read(snapshot_path)
            require(snapshot['snapshot_sha256'] == digest(snapshot['payload']), 'Snapshot payload digest differs')
            payload = snapshot['payload']
            require(payload['source_binding'] == stored['binding']
                    and payload['completed_steps'] == row['completed_steps']
                    and payload['prefix_points'] == stored['compact'][:row['prefix_physical_points']],
                    'Snapshot includes future answers or changed provenance')
            for fork in forks:
                candidate = next(c for c in row['candidates'] if c['branch_id'] == fork['forced_branch_id'])
                require(fork['source_cell'] == row['source_cell']
                        and fork['completed_steps'] == row['completed_steps'] and fork['context_id'] == index
                        and fork['scenario_id'] == row['scenario_id'] and fork['event_id'] == row['event_id']
                        and fork['phase'] == 'training_teacher' and fork['method'] == 'beam' and fork['seed'] == 0
                        and fork['max_new_points'] == 128 and fork['max_expansions'] == 16
                        and fork['effective_new_point_allowance'] == min(128, 256 - row['prefix_physical_points'])
                        and fork['snapshot'] == row['snapshot'] and fork['mode'] == mode
                        and fork['roles'] == candidate['roles'],
                        'Teacher task changed context, scope, or local cap')
    require(len(schedule) <= 48 and len({s['cell'] for s in schedule}) == len(schedule)
            and len({s['episode_id'] for s in schedule}) == len(schedule), 'Duplicate task or excessive teacher allocation')
    return states, schedule


def check_parent(run, cache):
    require(run.parent == ROOT / 'runs', 'Teacher run outside project')
    prepared = read(run / 'prepared.json')
    require(sha(run / 'prephysical_freeze_manifest.json') == prepared['freeze_sha256'], 'Teacher freeze changed')
    manifest(run, 'prephysical_freeze_manifest.json')
    for relative in prepared['source_files']:
        require(sha(ROOT / relative) == sha(run / 'source' / relative), 'Executing frozen source changed: ' + relative)
    source = ROOT / 'runs' / SOURCE_RUN
    require(sha(source / 'completion_manifest.json') == SOURCE_SHA
            and read(run / 'source_binding.json') == {'run': SOURCE_RUN, 'completion_sha256': SOURCE_SHA},
            'Source window seal changed')
    source_manifest = read(source / 'completion_manifest.json')
    require(sha(source / 'audit/summary.json') == source_manifest['audit/summary.json']
            and read(source / 'audit/summary.json')['status'] == 'PASS', 'Prior physical audit not bound to source seal')
    cfg, original = read(run / 'config.json'), read(source / 'config.json')
    require(cfg['window_teacher_study'] == 'CLPU_WINDOW_TEACHER_V1'
            and cfg['branch_search'] == original['branch_search'], 'Teacher method or branch limits changed')
    for key in ('horizon', 'slot_seconds', 'sample_seconds', 'alpha', 'algorithm', 'condition_id'):
        require(cfg[key] == original[key], 'Scientific configuration changed: ' + key)
    for name in ('physical_config.json', 'condition.json', 'operating_reference.json', 'author_static_metadata.json',
                 'load_reference.json', 'command_inventory.json', 'selected_scenarios.json', 'selected_events.json'):
        require(sha(run / name) == source_manifest[name] == sha(source / name), 'Physical input changed: ' + name)
    for name, expected in source_manifest.items():
        if name.startswith('source_inputs/'):
            require(sha(run / name) == expected, 'Physical source input changed: ' + name)
    states, schedule = check_states(run, cfg, cache)
    return cfg, states, schedule


def check_cell(run, spec, cfg, states, cache):
    cell = (run / spec['cell']).resolve()
    require(cell.is_relative_to(run) and cell != run, 'Teacher cell escaped run')
    sealed = manifest(cell)
    require({'spec.json', 'result.json', 'initial_state.json', 'plans.jsonl.gz', 'steps.jsonl.gz',
             'worker_started.json'} <= set(sealed) and not (cell / 'error.json').exists(),
            'Missing or failed teacher cell evidence')
    result = read(cell / 'result.json')
    require(read(cell / 'spec.json') == result['spec'] == spec
            and result['status'] == 'BOUNDED_COUNTERFACTUAL_COMPLETE', 'Teacher cell spec or result status changed')
    for key in ('new_training', 'optimizer_updates', 'new_source_rebuild_points'):
        require(result[key] == 0, 'Unapproved extra work: ' + key)
    require(result['formal_holdout_numerical_access'] is False and result['tune_numerical_access'] is False,
            'Teacher accessed another partition')
    state = next(row for row in states if row['state_id'] == spec['state_id'])
    snapshot = read(run / state['snapshot'])
    payload = snapshot['payload']
    require(snapshot['snapshot_sha256'] == result['snapshot_sha256'] == digest(payload)
            and result['source_snapshot_file_sha256'] == sha(run / state['snapshot']), 'Fork snapshot binding changed')
    stored = source_data(spec['source_cell'], cache)
    n = len(payload['prefix_points'])
    prefix = {index: row for index, row in stored['points'].items() if index < n}
    request_limit = payload['prefix_oracle_counters']['requests']
    requests = [row for row in stored['requests'] if row['request'] <= request_limit]
    mech = mechanics(run, spec, cfg)
    require(read(cell / 'initial_state.json') == mech.initial, 'Initial restoration history changed')
    physics = {'alpha': cfg['alpha'], 'base_condition_sha256': sha(run / 'condition.json')}
    scope = {'physics_identity': physics, 'model_version': 'nc34-clpu-ed-snapshot-v1',
        'static': mech.static, 'loads': mech.loads, 'scenario': mech.scenario, 'condition': mech.condition,
        'event': mech.event, 'initial_state': mech.initial, 'load_names': mech.names,
        'horizon': mech.horizon, 'slot_seconds': mech.slot, 'sample_seconds': mech.sample}
    require(payload['physics_identity'] == physics and payload['scope_sha256'] == digest(scope),
            'Snapshot physical scope/history changed')
    allowance = min(128, cfg['branch_search']['point_budget'] - n)
    count = result['oracle']['new_physical_points']
    require(type(count) is int and 0 <= count <= allowance, 'Invalid new physical cost')
    if count:
        require({'metadata.json', 'points.jsonl.gz', 'requests.jsonl', 'command_ledger.jsonl',
                 'command_stats.json', 'oracle_summary.json'} <= set(sealed), 'Missing paid new physical evidence')
        new_points, statuses = branch.check_raw_points(run, cell, {**spec, 'point_budget': allowance}, cfg, mech)
        commands = old.check_commands(cell, new_points, read(run / 'command_inventory.json'))
        counts = old.check_requests(cell, new_points, allowance)
        live = read(cell / 'oracle_summary.json')
        require(live == result['live_oracle_summary'] and live['points'] == len(new_points) == count
                and live['actual_numerical_entries'] == commands['numerical_entries']
                and live['requests'] == counts['requests'] and live['cache_hits'] == counts['cache_hits'] == 0
                and counts['denied'] == 0 and live['statuses'] == dict(statuses), 'Cold physical evidence differs')
        new_requests = list(read_lines(cell / 'requests.jsonl'))
    else:
        require(not any((cell / name).exists() for name in ('points.jsonl.gz', 'command_ledger.jsonl', 'oracle_summary.json')),
                'Zero-call fork concealed physical runtime')
        require(result['live_oracle_summary'] == {'points': 0, 'actual_numerical_entries': 0,
            'initialization_numerical_entries': 0, 'status': 'NO_NEW_PHYSICAL_CALLS'}, 'False zero-call summary')
        new_points, new_requests, statuses, commands = {}, [], {}, {'numerical_entries': 0}
    require(result['oracle']['new_numerical_entries_expected_for_snapshot_oracle'] == commands['numerical_entries']
            and result['oracle']['new_numerical_entries_from_runtime'] == (commands['numerical_entries'] if count else None),
            'Runtime initialization or numerical entries omitted')
    elapsed = window.finite_time(result['elapsed_seconds_including_restore_and_cold_initialization'], 'fork elapsed')
    require(elapsed >= math.fsum(window.finite_time(p['elapsed_seconds'], 'physical duration') for p in new_points.values()),
            'Fork elapsed time omits paid physical work')
    checked = check_records(mech, cfg['branch_search'], payload, result, stored['steps'], stored['plans'], prefix,
        requests, new_points, new_requests, sha(run / 'condition.json'), forced_branch_id=spec['forced_branch_id'],
        max_new_points=spec['max_new_points'], max_expansions=spec['max_expansions'],
        plan_journal=list(read_lines(cell / 'plans.jsonl.gz')), step_journal=list(read_lines(cell / 'steps.jsonl.gz')))
    return {**checked, 'cell': spec['cell'], 'cell_completion_sha256': sha(cell / 'completion_manifest.json'),
            'state_id': spec['state_id'], 'forced_branch_id': spec['forced_branch_id'],
            'new_numerical_entries': commands['numerical_entries'], 'physical_statuses': dict(statuses)}


def check_controller(controller, schedule, checks):
    cells = {row['cell'] for row in schedule}
    require(controller['status'] == 'COMPLETE_CELLS_PENDING_AUDIT' and controller['errors'] == []
            and controller['unknown_cost_cells'] == [] and controller['scheduled'] == len(schedule)
            and len(controller['completed']) == len(cells) and set(controller['completed']) == cells,
            'Teacher controller did not complete all cells')
    require(len(controller['exits']) == len(cells) and {row['cell'] for row in controller['exits']} == cells
            and all(row['exit_code'] == 0 for row in controller['exits']), 'Failed or missing worker exit')
    points, entries = sum(r['new_physical_points'] for r in checks), sum(r['new_numerical_entries'] for r in checks)
    require(points == controller['known_physical_points'] <= 48 * 128
            and entries == controller['known_numerical_entries'] <= 48 * (3 + 4 * 128),
            'Controller cost differs from independent physical audit')
    window.finite_time(controller['wall_seconds'], 'controller elapsed')
    return points, entries


def audit(run, output=None):
    run = Path(run).resolve()
    output = Path(output).resolve() if output else run / 'audit'
    require(not output.exists(), 'Audit output exists; preserve prior attempt')
    output.mkdir(parents=True)
    started, checks = time.perf_counter(), []
    result = {'status': 'FAIL', 'source_run': run.name, 'errors': [], 'audited_cells': 0, 'results': checks,
              'new_audit_physical_points': 0, 'new_training_tasks': 0}
    from audit_d059_supplement import NoSolverImports, no_solver_loaded
    guard = NoSolverImports()
    sys.meta_path.insert(0, guard)
    try:
        cache = {}
        cfg, states, schedule = check_parent(run, cache)
        for spec in schedule:
            checked = check_cell(run, spec, cfg, states, cache)
            checks.append(checked)
            save(output / f'cell_{spec["episode_id"]:03d}.json', checked)
            save(output / 'progress.json', {'audited_cells': len(checks), 'scheduled': len(schedule)})
        points, entries = check_controller(read(run / 'controller_completion.json'), schedule, checks)
        no_solver_loaded()
        result.update(status='PASS', audited_cells=len(checks), physical_points=points, numerical_entries=entries,
            freeze_sha256=sha(run / 'prephysical_freeze_manifest.json'), schedule_sha256=sha(run / 'schedule.json'),
            states_sha256=sha(run / 'states.json'), source_binding_sha256=sha(run / 'source_binding.json'),
            controller_completion_sha256=sha(run / 'controller_completion.json'),
            source_window_completion_sha256=SOURCE_SHA)
    except BaseException as exc:
        result['errors'].append({'error': repr(exc), 'traceback': traceback.format_exc()})
    finally:
        sys.meta_path.remove(guard)
    result['elapsed_seconds'] = time.perf_counter() - started
    save(output / 'summary.json', result)
    save(output / 'completion_manifest.json', {p.relative_to(output).as_posix(): sha(p)
        for p in output.rglob('*') if p.is_file() and p.name != 'completion_manifest.json'})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    outcome = audit(args.run, args.output)
    print(canonical({key: value for key, value in outcome.items() if key != 'results'}))
    raise SystemExit(0 if outcome['status'] == 'PASS' else 1)
