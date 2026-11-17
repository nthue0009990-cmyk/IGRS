"""Independent saved-log audit of the bounded, training-only counterfactual pilot.

Reuses the existing independent graph/path/physical auditors. No production
counterfactual, search, environment or solver implementation is imported.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
sys.dont_write_bytecode = True
import audit_nature34_clpu_branch as branch
from d060_validation_common import SOURCE_RUN, SOURCE_SHA

old = branch.old
require, close, same = branch.require, branch.close, branch.same
read, save, sha, canonical, read_lines = branch.read, branch.save, branch.sha, branch.canonical, branch.read_lines


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def manifest(directory, name='completion_manifest.json'):
    rows = read(directory / name)
    for relative, expected in rows.items():
        target = (directory / relative.replace('\\', '/')).resolve()
        require(target.is_relative_to(directory.resolve()) and sha(target) == expected,
                'Changed or outside sealed artifact: ' + str(target))
    return rows


class ForkRequests:
    """Independently combine paid historical cache with one cold fork ledger."""
    def __init__(self, source_requests, prefix_points, new_requests, new_points, mech, condition_sha, allowance):
        self.prefix = branch.PaidRequests(source_requests, prefix_points, mech, 256, condition_sha)
        self.live = branch.PaidRequests(new_requests, new_points, mech, allowance, condition_sha)
        self.points = dict(prefix_points)
        self.base = len(prefix_points)
        self.points.update({self.base + i: row for i, row in new_points.items()})
        self.mech, self.condition_sha, self.allowance = mech, condition_sha, allowance
        self.used = self.requests = self.hits = 0
        self.cache, self.mappings = {}, []
        self.forking = False
        self.base_requests = self.base_hits = 0

    def activate(self):
        require(self.prefix.index == len(self.prefix.rows) and self.used == self.base,
                'Snapshot restored future, missing, or unconsumed source requests')
        self.forking = True
        self.base_requests, self.base_hits = self.requests, self.hits

    def certify(self, state, seconds):
        if not self.forking:
            value = self.prefix.certify(state, seconds)
            self.used, self.requests = self.prefix.used, self.prefix.requests
            self.cache = dict(self.prefix.cache)
            self.hits = self.requests - self.used
            return value
        self.requests += 1
        identity = self.mech.identity(state, seconds, self.condition_sha)
        key = digest(identity)
        if key in self.cache:
            self.hits += 1
            point_id = self.cache[key]
        elif self.used >= 256 or self.used - self.base >= self.allowance:
            return None
        else:
            before = self.live.used
            local = self.live.certify(state, seconds)
            require(local is not None and self.live.used == before + 1,
                    'New fork point reused a sibling/future answer')
            point_id = self.base + local['point_id']
            self.used = self.base + self.live.used
            self.cache[key] = point_id
            self.mappings.append({'identity_sha256': key, 'local_point_id': local['point_id'],
                                  'cumulative_point_id': point_id})
        point = self.points[point_id]
        return {'point_id': point_id, 'status': point['status'], 'reliable': point['reliable'],
                'feedback': point['structured_feedback']['violations']}

    def counts(self):
        return {'points': self.used, 'requests': self.requests, 'hits': self.hits}


def check_labels(result, initial, first, final, best):
    require(best is not None and best['certified_complete'] is True
            and best['status'] == best['validation']['status'] == 'COMPLETE',
            'Uncertified incumbent cannot form a label')
    require(final >= initial and first >= initial, 'Certified starting quality lost')
    for key, value in {'initial_quality': initial, 'first_step_quality': first, 'final_quality': final,
                       'g1': first - initial, 'gH': final - initial,
                       'g1_pp': 100 * (first - initial), 'gH_pp': 100 * (final - initial)}.items():
        close(result[key], value, 'Measured label ' + key)
    same(result['best_plan'], best, 'independently_certified_best')


def source_prefix(source, cell_name, payload, cache):
    binding = payload['source_binding']
    require(binding['source_run'] == SOURCE_RUN and binding['source_cell'] == cell_name,
            'Source snapshot provenance changed')
    cell = source / cell_name
    if '_source_manifest' not in cache:
        cache['_source_manifest'] = {k.replace('\\', '/'): v for k, v in read(source / 'completion_manifest.json').items()}
    for name, field in [('result.json', 'source_result_sha256'), ('completion_manifest.json', 'source_completion_sha256'),
                        ('points.jsonl.gz', 'source_points_sha256'), ('transitions.jsonl.gz', 'source_transitions_sha256')]:
        require(cache['_source_manifest'].get(cell_name + '/' + name) == binding[field],
                'Snapshot provenance not bound to original source seal')
        if cell_name not in cache:
            require(sha(cell / name) == binding[field], 'Bound source changed: ' + name)
    if cell_name not in cache:
        full_result = read(cell / 'result.json')
        compact_result = {k: full_result[k] for k in ('spec', 'config')}
        compact_result.update(attempts=full_result['attempts'][:9], trace=full_result['trace'][:8])
        del full_result
        transitions = []
        for transition in read_lines(cell / 'transitions.jsonl.gz'):
            transitions.append(transition)
            if len(transitions) == 9:
                break
        requests = []
        maximum_request = transitions[-1]['oracle_before']['requests']
        for row in read_lines(cell / 'requests.jsonl'):
            if row['request'] > maximum_request:
                break
            requests.append(row)
        cache[cell_name] = {'result': compact_result, 'transitions': transitions, 'requests': requests}
    stored = cache[cell_name]
    step = payload['completed_steps']
    require(step in (0, 8) and step < len(stored['transitions']), 'Unavailable source position fabricated')
    transition = stored['transitions'][step]
    n = transition['oracle_before']['points']
    require(len(payload['prefix_points']) == n, 'Future physical cache loaded before snapshot')
    points = {}
    fields = ('status', 'reliable', 'clpu_multipliers', 'action', 'time_minutes', 'state_identity',
              'identity_sha256', 'actual_call_counts', 'dynamic_state', 'elapsed_seconds', 'structured_feedback')
    for row in read_lines(cell / 'points.jsonl.gz'):
        if len(points) == n:
            break
        index = len(points)
        require(row['point_id'] == index, 'Source points not in chronological order')
        compact = {'identity': row['state_identity'], 'identity_sha256': row['identity_sha256'],
                   'certificate': {'point_id': index, 'status': row['status'], 'reliable': row['reliable'],
                                   'feedback': row['structured_feedback']['violations']},
                   'source_row_sha256': digest(row)}
        require(compact == payload['prefix_points'][index], 'Snapshot point differs from sealed source row')
        points[index] = {k: row[k] for k in fields}
    require(len(points) == n, 'Incomplete historical point prefix')
    request_limit = transition['oracle_before']['requests']
    requests = [r for r in stored['requests'] if r['request'] <= request_limit]
    return stored['result'], stored['transitions'], points, requests


def check_fork(run, spec, cfg, cache):
    cell = run / spec['cell']
    sealed = manifest(cell)
    require({'spec.json', 'result.json', 'initial_state.json', 'plans_delta.jsonl.gz',
             'transitions_delta.jsonl.gz', 'worker_started.json'} <= set(sealed), 'Missing sealed fork evidence')
    require(not (cell / 'error.json').exists() and read(cell / 'spec.json') == spec, 'Fork error or spec mismatch')
    result = read(cell / 'result.json')
    require(result['spec'] == spec and result['status'] == 'BOUNDED_COUNTERFACTUAL_COMPLETE'
            and result['new_training'] == result['optimizer_updates'] == result['new_source_rebuild_points'] == 0
            and result['formal_holdout_numerical_access'] is False and result['tune_numerical_access'] is False,
            'Fork scope/result mismatch')
    snapshot = read(run / spec['snapshot'])
    payload = snapshot['payload']
    require(digest(payload) == snapshot['snapshot_sha256'] == result['snapshot_sha256']
            and result['source_snapshot_file_sha256'] == sha(run / spec['snapshot']), 'Snapshot binding changed')
    source = ROOT / 'runs' / SOURCE_RUN
    source_result, source_transitions, prefix, source_requests = source_prefix(source, payload['source_binding']['source_cell'], payload, cache)
    require(source_result['spec']['phase'] == 'teacher' and source_result['spec']['method'] == payload['source_method']
            and payload['source_method'] in ('beam', 'greedy') and source_result['config'] == cfg['branch_search'] == payload['config'],
            'Wrong source teacher or scientific configuration')
    require((source_result['spec']['scenario_id'], source_result['spec']['event_id']) ==
            (spec['scenario_id'], spec['event_id']), 'Source scenario/event mismatch')
    scenarios = {r['scenario_id']: r for r in read(run / 'selected_scenarios.json')}
    events = {r['event_id']: r for r in read(run / 'selected_events.json')}
    mech = branch.Mechanics(read(run / 'author_static_metadata.json'), read(run / 'load_reference.json'),
                            scenarios[spec['scenario_id']], read(run / 'condition.json'), events[spec['event_id']], cfg)
    require(read(cell / 'initial_state.json') == mech.initial, 'Initial load history changed')
    physics = {'alpha': .5, 'base_condition_sha256': sha(run / 'condition.json')}
    scope = {'physics_identity': physics, 'model_version': 'nc34-clpu-ed-snapshot-v1',
             'static': mech.static, 'loads': mech.loads, 'scenario': mech.scenario, 'condition': mech.condition,
             'event': mech.event, 'initial_state': mech.initial, 'load_names': mech.names,
             'horizon': mech.horizon, 'slot_seconds': mech.slot, 'sample_seconds': mech.sample}
    require(payload['physics_identity'] == physics and payload['scope_sha256'] == digest(scope), 'Snapshot physical scope changed')
    require(spec['max_new_points'] == min(128, 256 - len(prefix)) and spec['max_expansions'] == 16,
            'Local budget changed')
    new_count = result['oracle']['new_physical_points']
    require(type(new_count) is int and 0 <= new_count <= spec['max_new_points'], 'Invalid paid fork count')
    if new_count:
        require({'metadata.json', 'points.jsonl.gz', 'requests.jsonl', 'command_ledger.jsonl',
                 'command_stats.json', 'oracle_summary.json'} <= set(sealed), 'Missing new physical evidence')
        new_points, statuses = branch.check_raw_points(run, cell, {**spec, 'point_budget': spec['max_new_points']}, cfg, mech)
        commands = old.check_commands(cell, new_points, read(run / 'command_inventory.json'))
        requests = old.check_requests(cell, new_points, spec['max_new_points'])
        live = read(cell / 'oracle_summary.json')
        require(live == result['live_oracle_summary'] and live['points'] == len(new_points) == new_count
                and live['actual_numerical_entries'] == commands['numerical_entries']
                and live['requests'] == requests['requests'] and live['cache_hits'] == requests['cache_hits'] == 0
                and requests['denied'] == 0 and live['statuses'] == dict(statuses), 'Cold physical ledger differs')
        new_requests = list(read_lines(cell / 'requests.jsonl'))
    else:
        require(not any((cell / name).exists() for name in ('points.jsonl.gz', 'command_ledger.jsonl', 'oracle_summary.json')),
                'Zero-cost fork concealed physical initialization')
        require(result['live_oracle_summary'] == {'points': 0, 'actual_numerical_entries': 0,
                'initialization_numerical_entries': 0, 'status': 'NO_NEW_PHYSICAL_CALLS'}, 'False zero-call claim')
        new_points, new_requests, statuses, commands = {}, [], {}, {'numerical_entries': 0}
    oracle = ForkRequests(source_requests, prefix, new_requests, new_points, mech, sha(run / 'condition.json'), spec['max_new_points'])
    replay = branch.BranchReplay(mech, cfg['branch_search'], oracle, source_result['attempts'])
    step = spec['after_steps']
    require(payload['completed_steps'] == result['source_completed_steps'] == step, 'Snapshot position changed')
    require(payload['source_bootstrap_attempt'] == source_result['attempts'][0]
            and payload['source_trace'] == source_result['trace'][:step], 'Source prefix altered')
    for i in range(step):
        same(source_transitions[i]['obs_raw'], replay.observation('raw'), 'source_observation')
        selected = source_transitions[i]['action']
        require(selected == replay.rule(payload['source_method']), 'Source rule changed')
        same(replay.step(selected), payload['source_trace'][i], 'source_response')
    same(payload['raw_observation'], replay.observation('raw'), 'snapshot_observation')
    same(payload['raw_observation'], source_transitions[step]['obs_raw'], 'recorded_source_observation')
    require(payload['prefix_oracle_counters'] == {'points': oracle.used, 'requests': oracle.requests, 'cache_hits': oracle.hits},
            'Historical charges changed')
    require(not replay.done and replay.best and replay.best['certified_complete'], 'No certified active snapshot')
    ids = [row['id'] for row in replay.window()]
    expected_candidates = list(dict.fromkeys([source_transitions[step]['branch_id'], ids[replay.rule('greedy')]]))
    salt = f"route-b-s1-candidate-v1|{spec['scenario_id']}|{spec['event_id']}|{step}|"
    for candidate in sorted(ids, key=lambda bid: hashlib.sha256((salt + str(bid)).encode()).hexdigest()):
        if candidate not in expected_candidates:
            expected_candidates.append(candidate)
        if len(expected_candidates) == min(3, len(ids)):
            break
    require({row['branch_id'] for row in read(run / 'schedule.json') if row['state_id'] == spec['state_id']}
            == set(expected_candidates), 'Predetermined teacher/greedy/hash candidate selection changed')
    oracle.activate()
    start_turn, start_generated, start_assessed = replay.turn, len(replay.keys), len(replay.assessed)
    initial, first = replay.quality, None
    replay.attempts = source_result['attempts'][:replay.attempt_index] + result['new_attempts']
    require(list(read_lines(cell / 'plans_delta.jsonl.gz')) == result['new_attempts'], 'Plan journal differs')
    journal = list(read_lines(cell / 'transitions_delta.jsonl.gz'))
    transitions = result['transitions']
    require(1 <= len(transitions) <= min(16, 256 - step) and len(journal) == 2 * len(transitions),
            'Missing forced action or excess/unclosed transitions')
    for i, row in enumerate(transitions):
        require(row['transition_id'] == i and row['status'] == 'COMPLETED' and row['forced'] == (i == 0),
                'Fork transition order changed')
        observation = replay.observation('raw')
        same(row['obs_raw'], observation, 'fork_observation')
        require(digest(row['obs_raw']) == row['observation_sha256'], 'Observation digest changed')
        require(row['oracle_before'] == oracle.counts(), 'Fork pre-action charges differ')
        selected = row['action']
        require(row['branch_id'] == replay.window()[selected]['id'], 'Selected branch not in window')
        require((i == 0 and row['branch_id'] == spec['branch_id'] == result['forced_branch_id'])
                or (i > 0 and selected == replay.rule('beam')), 'Wrong forced action or continuation policy')
        started = {k: v for k, v in row.items() if k not in ('response', 'oracle_after', 'obs_raw')}
        started['status'] = 'STARTED'
        require(journal[2*i] == started and journal[2*i+1] == row, 'Decision journal not closed')
        response = replay.step(selected)
        same(row['response'], response, 'fork_response')
        same(result['fork_trace'][i], response, 'fork_trace')
        require(row['oracle_after'] == oracle.counts(), 'Fork post-action charges differ')
        if i == 0:
            first = replay.quality
    require(len(result['fork_trace']) == len(transitions) and replay.attempt_index == len(replay.attempts),
            'Unconsumed/hidden candidate attempts')
    require(replay.done or len(transitions) == min(16, 256 - step), 'Unexplained early fork stop')
    expected_stop = replay.reason if replay.done else 'FORK_EXPANSION_LIMIT'
    if replay.reason == 'PHYSICAL_BUDGET_EXHAUSTED' and oracle.used < 256:
        expected_stop = 'FORK_NEW_PHYSICAL_POINT_LIMIT'
    require(result['stop_reason'] == expected_stop and result['core_stop_reason'] == replay.reason, 'False termination reason')
    for key, value in {'completed_expansions': len(transitions), 'cumulative_expansions': replay.turn,
                       'new_generated_candidates': len(replay.keys)-start_generated,
                       'new_graph_assessed_candidates': len(replay.assessed)-start_assessed,
                       'cumulative_generated_candidates': len(replay.keys),
                       'cumulative_graph_assessed_candidates': len(replay.assessed),
                       'cumulative_checked_candidates': len(replay.checked),
                       'new_point_allowance': spec['max_new_points'],
                       'expansion_allowance_including_forced': min(16, 256-start_turn)}.items():
        require(result[key] == value, 'Fork counter differs: ' + key)
    require(oracle.live.index == len(new_requests) and oracle.live.used == len(new_points), 'Unbound new physical records')
    summary = result['oracle']
    expected = {'cumulative_points': oracle.used, 'cumulative_point_budget': 256, 'cumulative_requests': oracle.requests,
                'cumulative_cache_hits': oracle.hits, 'historical_prefix_points': len(prefix),
                'new_physical_points': len(new_points), 'new_requests_including_cache': oracle.requests-oracle.base_requests,
                'new_cache_hits': oracle.hits-oracle.base_hits, 'live_initialization_occurred': bool(new_points),
                'live_initialization_attempted': bool(new_points), 'live_requests': len(new_points),
                'new_numerical_entries_expected_for_snapshot_oracle': commands['numerical_entries'],
                'historical_restore_new_physical_points': 0, 'point_id_mappings': oracle.mappings}
    for key, value in expected.items():
        require(summary[key] == value, 'Cumulative oracle ledger differs: ' + key)
    require(summary['new_numerical_entries_from_runtime'] == (commands['numerical_entries'] if new_points else None),
            'Runtime cost mismatch')
    check_labels(result, initial, first, replay.quality, replay.best)
    return {'cell': spec['cell'], 'cell_completion_sha256': sha(cell / 'completion_manifest.json'),
            'state_id': spec['state_id'], 'branch_id': spec['branch_id'], 'g1_pp': result['g1_pp'], 'gH_pp': result['gH_pp'],
            'new_physical_points': len(new_points), 'new_numerical_entries': commands['numerical_entries'],
            'historical_prefix_points': len(prefix), 'completed_expansions': len(transitions),
            'physical_statuses': dict(statuses), 'stop_reason': expected_stop, 'status': 'PASS'}


def check_parent(run):
    require(run.parent == ROOT / 'runs', 'Run outside project')
    prepared = read(run / 'prepared.json')
    require(sha(run / 'prephysical_freeze_manifest.json') == prepared['freeze_sha256'], 'Input freeze changed')
    manifest(run, 'prephysical_freeze_manifest.json')
    source = ROOT / 'runs' / SOURCE_RUN
    require(sha(source / 'completion_manifest.json') == SOURCE_SHA, 'D059 source seal changed')
    cfg = read(run / 'config.json')
    require(cfg['branch_search'] == read(source / 'config.json')['branch_search']
            and cfg['counterfactual_study'] == 'CLPU_COUNTERFACTUAL_S1_V1', 'Scientific search configuration changed')
    population = read(run / 'population.json')
    train = read(source / 'training_contexts.json')
    require(len(population['contexts']) == 8 and all({'scenario_id': c['scenario_id'], 'event_id': c['event_id']} in train
            for c in population['contexts']), 'Teacher population left training partition')
    inventory, schedule = read(run / 'state_inventory.json'), read(run / 'schedule.json')
    require(len(inventory) == 16 and {(r['scenario_id'],r['event_id'],r['after_steps']) for r in inventory} ==
            {(c['scenario_id'],c['event_id'],step) for c in population['contexts'] for step in (0,8)}, 'Missing predetermined state')
    for state in inventory:
        specs = [s for s in schedule if s['state_id'] == state['state_id']]
        require(len(specs) == len(state['candidates']) and
                len(specs) == len(set(s['branch_id'] for s in specs)), 'Repeated/missing fork')
        require({s['branch_id'] for s in specs} == {r['branch_id'] for r in state['candidates']}, 'Candidate population changed')
        if state['status'] != 'AVAILABLE':
            require(not specs and state['status'] == 'UNAVAILABLE', 'Unavailable state silently replaced')
    require(len(schedule) <= 48 and sum(s['max_new_points'] for s in schedule) <= 6144, 'Pilot budget exceeded')
    controller = read(run / 'controller_completion.json')
    require(controller['status'] == 'COMPLETE_FORKS_PENDING_AUDIT' and controller['errors'] == []
            and controller['unknown_cost_cells'] == [] and controller['scheduled'] == len(schedule)
            and {r['cell'] for r in controller['completed']} == {s['cell'] for s in schedule}
            and all(r['exit_code'] == 0 for r in controller['completed']), 'Controller completion not closed')
    return cfg, schedule, controller


def audit(run):
    run = Path(run).resolve()
    output = run / 'audit'
    output.mkdir(exist_ok=False)
    started, results = time.perf_counter(), []
    summary = {'status': 'FAIL', 'errors': [], 'results': results, 'new_audit_physical_points': 0,
               'new_training_tasks': 0}
    from audit_d059_supplement import NoSolverImports, no_solver_loaded
    guard = NoSolverImports()
    sys.meta_path.insert(0, guard)
    try:
        cfg, schedule, controller = check_parent(run)
        cache = {}
        for spec in schedule:
            checked = check_fork(run, spec, cfg, cache)
            results.append(checked)
            save(output / f'fork_{spec["episode_id"]:03d}.json', checked)
            save(output / 'progress.json', {'completed': len(results), 'scheduled': len(schedule), 'errors': []})
        points = sum(r['new_physical_points'] for r in results)
        entries = sum(r['new_numerical_entries'] for r in results)
        require(points == controller['known_new_physical_points'] <= 6144
                and entries == controller['known_new_numerical_entries'], 'Controller and independent costs differ')
        no_solver_loaded()
        summary.update(status='PASS', audited_forks=len(results), new_physical_points=points,
                       new_numerical_entries=entries, freeze_sha256=sha(run / 'prephysical_freeze_manifest.json'))
    except BaseException as exc:
        summary['errors'].append({'error': repr(exc), 'traceback': traceback.format_exc()})
    finally:
        sys.meta_path.remove(guard)
    summary['elapsed_seconds'] = time.perf_counter() - started
    save(output / 'summary.json', summary)
    save(output / 'completion_manifest.json', {p.relative_to(output).as_posix(): sha(p)
                                               for p in output.rglob('*') if p.is_file() and p.name != 'completion_manifest.json'})
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.run)
    print(canonical({k: v for k, v in result.items() if k != 'results'}))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)
