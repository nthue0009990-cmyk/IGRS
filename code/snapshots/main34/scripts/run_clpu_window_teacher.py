"""One frozen coverage-window teacher diagnostic; no training or formal test."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
sys.dont_write_bytecode = True

from d060_validation_common import canonical, normalize_manifest, read, require, sha
from nature34_experiment_support import create_run, freeze, dependencies, append, numeric_lists
from run_nature34_clpu_tail_pilot import save
from run_clpu_counterfactual_pilot import verify_files, seal, utc
from run_d060_validation import ROOT_INPUTS, bound_copy

STUDY = 'CLPU_WINDOW_TEACHER_V1'
SOURCE_RUN = '20261027T121023_469270Z_clpu_window_pilot'
SOURCE_SHA = '9967ea771263b5b9bb45835116dec1c5fd70f8908c17ec2348d90d4a6377fc7e'
ACTIVE = ROOT / 'artifacts/clpu_window_teacher_active.json'
MODES = ('normal', 'coverage')
POINT_CAP, STEP_CAP, TOTAL_CAP = 128, 16, 6144
MIN_FREE = 12 * 2**30
CANDIDATE_SALT = 'clpu-window-teacher-v1'
GATE = {'minimum_states': 2, 'minimum_faults': 2, 'minimum_events': 2,
        'minimum_auc_advantage_pp': .1, 'final_quality_tolerance_pp': 1e-9,
        'baseline': 'same_state_window_beam_fork',
        'interpretation': 'Permission to consider bounded learning, not a trained-method win'}


def lines(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def choose_candidates(search, context, mode):
    """Always retain actual Beam; alternatives use only current graph facts."""
    from dnr.nature34_clpu_window_counterfactual import state_eligibility
    state = state_eligibility(search)
    require(state['eligible'] and state['mode'] == mode, 'State is not eligible in the requested mode')
    window = search._window()
    by_id = {row['id']: row for row in window}
    beam = window[search.rule_action('beam')]['id']
    eligible = state['candidate_ids']
    greedy = min(eligible, key=lambda identifier: (by_id[identifier]['info']['priority'], identifier))
    chosen, by_plan = [], {}

    def add(identifier, role):
        plan = tuple(by_id[identifier]['info']['actions'])
        if plan in by_plan:
            row = chosen[by_plan[plan]]
            if role not in row['roles']:
                row['roles'].append(role)
            return
        if len(chosen) < 3:
            by_plan[plan] = len(chosen)
            chosen.append({'branch_id': identifier, 'roles': [role]})

    add(beam, 'window_beam')
    add(greedy, 'graph_greedy')
    prefix = f"{CANDIDATE_SALT}|{context['scenario_id']}|{context['event_id']}|{mode}|{search._beam_turn}|"
    for identifier in sorted(eligible, key=lambda identifier:
            (hashlib.sha256((prefix + str(identifier)).encode()).hexdigest(), identifier)):
        if len(chosen) >= 3:
            break
        add(identifier, 'fixed_hash_fill')
    require(2 <= len(chosen) <= 3, 'At least two distinct measured plans required')
    return chosen


def discover_state_positions(env, source_result, source_steps, source_plans, source_points,
                             physics_identity, context):
    """One progressive saved-answer replay; selection sees only current history.

    The adapter enforces chronological first-use identity IDs even though its
    read-only backing file is complete.  It cannot compute any new answer.
    Selected snapshots are rebuilt separately with strictly truncated prefixes.
    """
    from dnr.nature34_clpu_branch import BranchConfig
    from dnr.nature34_clpu_counterfactual import _PrefixOracle, _compact_point
    from dnr.nature34_clpu_window import WindowBranchSearch
    from dnr.nature34_clpu_window_counterfactual import _completed_prefix, state_eligibility
    require(source_result['spec']['window_variant'] == 'coverage_v1'
            and source_result['spec']['method'] == 'beam', 'Wrong historical discovery source')
    config = BranchConfig(**source_result['config'])
    trace = _completed_prefix(source_steps, source_result['completed_expansions'])
    points = [_compact_point(row) for row in source_points]
    oracle = _PrefixOracle(env, points, config.point_budget, physics_identity)
    search = WindowBranchSearch(env, oracle, config)
    require(search.attempts and canonical(search.attempts[0]) == canonical(source_plans[0]),
            'Source bootstrap differs during progressive replay')
    selected = {}
    visited = []
    for index, expected in enumerate(trace):
        require(not search.done and search._beam_turn == index, 'Historical trace length differs')
        eligibility = state_eligibility(search)
        visited.append({'completed_steps': index, 'mode': eligibility['mode'],
                        'eligible': eligibility['eligible'], 'candidate_ids': eligibility['candidate_ids'],
                        'physical_points': oracle.points})
        mode = eligibility['mode']
        if eligibility['eligible'] and mode not in selected:
            selected[mode] = {'completed_steps': index, 'prefix_physical_points': oracle.points,
                'candidates': choose_candidates(search, context, mode),
                'observation': numeric_lists(search.observation('raw')), 'eligibility': eligibility}
        observation = search.observation('raw')
        require(observation['branch_ids'] == expected['window_branch_ids'], 'Historical window differs')
        action = search.rule_action('beam')
        require(action == expected['selected_index'], 'Historical Beam action differs')
        require(canonical(search.step(action)) == canonical(expected), 'Historical step outcome differs')
    require(search.done and search.stop_reason == source_result['stop_reason'], 'Historical final stop differs')
    require(canonical(search.incumbent) == canonical(source_result['best_plan']), 'Historical final incumbent differs')
    require(canonical(search.attempts) == canonical(source_plans), 'Historical attempted-plan journal differs')
    require(oracle.points == len(points) == source_result['oracle']['points'], 'Historical paid point order/count differs')
    return selected, {'visited_states': visited, 'source_points_replayed': oracle.points,
        'source_decisions_replayed': search._beam_turn, 'new_physical_points': 0,
        'selection': 'first eligible pre-action state in each actual window mode'}


def schedule_for(states):
    require(len(states) == 16 and {(s['context_id'], s['mode']) for s in states} ==
            {(i, mode) for i in range(8) for mode in MODES}, 'All sixteen state slots required')
    rows = []
    for state in states:
        if state['status'] == 'UNAVAILABLE':
            require(not state['candidates'], 'Unavailable state has measured candidates')
            continue
        require(state['status'] == 'AVAILABLE' and 2 <= len(state['candidates']) <= 3,
                'Available state needs two or three candidates')
        require(sum('window_beam' in c['roles'] for c in state['candidates']) == 1,
                'Exactly one WindowBeam baseline required')
        require(len({c['branch_id'] for c in state['candidates']}) == len(state['candidates']),
                'Duplicate scheduled candidate')
        require(type(state['prefix_physical_points']) is int and 0 <= state['prefix_physical_points'] < 256,
                'Available source has invalid or exhausted physical allowance')
        for candidate in state['candidates']:
            rows.append({key: state[key] for key in ('state_id', 'context_id', 'scenario_id', 'event_id',
                'source_cell', 'completed_steps', 'snapshot', 'mode')} | {
                'forced_branch_id': candidate['branch_id'], 'roles': candidate['roles'],
                'max_new_points': POINT_CAP, 'effective_new_point_allowance': min(POINT_CAP, 256 - state['prefix_physical_points']),
                'max_expansions': STEP_CAP, 'phase': 'training_teacher', 'method': 'beam', 'seed': 0})
    rows.sort(key=lambda row: hashlib.sha256(('clpu-window-teacher-order-v1|' + canonical(row)).encode()).hexdigest())
    for index, row in enumerate(rows):
        row.update(episode_id=index, cell=f'episodes/{index:03d}')
    require(len(rows) <= 48 and sum(row['effective_new_point_allowance'] for row in rows) <= TOTAL_CAP,
            'Teacher allocation exceeds declared pilot budget')
    return rows


def _passed_tests(tests_path):
    cases = list(ET.parse(tests_path).getroot().iter('testcase'))
    require(cases and not any(list(case.iter('failure')) + list(case.iter('error')) + list(case.iter('skipped'))
                             for case in cases), 'Nonempty all-passed software evidence required')
    return len(cases)


def verify_source_cell(source, root_manifest, source_spec):
    """Bind an excluded nested seal through the root-bound independent audit.

    The inherited seal writer excludes every completion_manifest.json basename,
    including nested cell seals.  Its root-bound audit records the exact seal
    hash for each reviewed cell.  Both that chain and the root/cell file hashes
    must agree; a locally self-consistent replacement is insufficient.
    """
    source = source.resolve()
    cell = (source / source_spec['cell']).resolve()
    require(cell != source and cell.is_relative_to(source), 'Historical cell escaped source run')
    audit_path = source / 'audit/summary.json'
    require(root_manifest.get('audit/summary.json') == sha(audit_path), 'Source audit changed')
    audit = read(audit_path)
    require(audit['status'] == 'PASS', 'Historical independent audit did not pass')
    matches = [row for row in audit['results'] if row['cell'] == source_spec['cell']]
    require(len(matches) == 1 and matches[0]['spec'] == source_spec, 'Audited historical cell/spec absent or ambiguous')
    seal_path = cell / 'completion_manifest.json'
    require(sha(seal_path) == matches[0]['cell_completion_sha256'], 'Historical cell seal changed from audited version')
    cell_manifest = normalize_manifest(read(seal_path))
    prefix = cell.relative_to(source).as_posix() + '/'
    for name, expected in cell_manifest.items():
        target = (cell / name).resolve()
        require(target.is_relative_to(cell) and target != cell, 'Historical cell seal path escaped')
        require(root_manifest.get(prefix + name) == expected, 'Historical parent and cell seals disagree: ' + name)
    for name in ('result.json', 'points.jsonl.gz', 'steps.jsonl.gz', 'plans.jsonl.gz', 'requests.jsonl'):
        require(name in cell_manifest and sha(cell / name) == cell_manifest[name],
                'Historical source journal changed: ' + prefix + name)
    return cell


def prepare(tests_path):
    from run_nature34_clpu_pilot import make_env
    from dnr.nature34_clpu_window import WINDOW_RULE_VERSION
    from dnr.nature34_clpu_window_counterfactual import restore_snapshot, snapshot_prefix_point_count
    count_tests = _passed_tests(tests_path)
    source = ROOT / 'runs' / SOURCE_RUN
    require(sha(source / 'completion_manifest.json') == SOURCE_SHA, 'Sealed window source changed')
    sm = normalize_manifest(read(source / 'completion_manifest.json'))
    for name in ('config.json', 'population.json', 'schedule.json', 'training_contexts.json',
                 'selected_scenarios.json', 'selected_events.json', 'audit/summary.json'):
        require(sha(source / name) == sm[name], 'Bound source input changed: ' + name)
    require(read(source / 'audit/summary.json')['status'] == 'PASS', 'Source physical audit missing')
    population = read(source / 'population.json')
    contexts = population['contexts']
    require(len(contexts) == 8 and len({(c['scenario_id'], c['event_id']) for c in contexts}) == 8,
            'Original eight training contexts required')
    training = {(row['scenario_id'], row['event_id']) for row in read(source / 'training_contexts.json')}
    require({(c['scenario_id'], c['event_id']) for c in contexts} == training, 'Training population changed')
    cfg = deepcopy(read(source / 'config.json'))
    cfg['window_teacher_study'] = STUDY
    cfg['window_teacher'] = {'source_window_rule_version': WINDOW_RULE_VERSION,
        'state_selection': 'first pre-action state of each actual normal/coverage mode with >=2 unique unchecked graph-improving plans',
        'max_candidates': 3, 'candidate_hash_domain': CANDIDATE_SALT,
        'max_new_points': POINT_CAP, 'max_expansions': STEP_CAP, 'max_workers': 2,
        'worker_watchdog_seconds': 1800, 'minimum_free_disk_bytes': MIN_FREE,
        'total_physical_budget': TOTAL_CAP, 'state_slots': 16, 'gate': deepcopy(GATE),
        'target': 'mean complete-certified gain on integer additional physical budgets 1..B',
        'source_restore_points': 0, 'new_training': 0, 'no_formal_test': True,
        'no_automatic_retry_or_parameter_scan': True}
    require(shutil.disk_usage(ROOT).free >= MIN_FREE, 'Insufficient frozen teacher storage reserve')
    run = create_run('clpu_window_teacher', cfg)
    try:
        for name in ROOT_INPUTS:
            bound_copy(source, run / name, sm, name)
        for path in (source / 'source_inputs').rglob('*'):
            if path.is_file():
                relative = path.relative_to(source).as_posix()
                bound_copy(source, run / relative, sm, relative)
        for name in ('population.json', 'selected_scenarios.json', 'selected_events.json', 'training_contexts.json'):
            bound_copy(source, run / name, sm, name)
        shutil.copyfile(tests_path, run / 'software_tests.xml')
        shutil.copyfile(ROOT / 'docs/clpu_window_teacher_protocol_v1.md', run / 'protocol.md')
        save(run / 'authorization.json', {'utc': utc(), 'execution_authorized': True,
            'user_message': '行那你开始做吧，项目文件夹里的存储空间你先清理一下，没用的删掉，可压缩的压缩',
            'scope': 'Storage maintenance followed by one real training-only candidate-teacher validation; subsequent learning depends on this gate',
            'data_handling': 'Preserve measured outcomes and original records; no synthetic result claims'})
        save(run / 'source_binding.json', {'run': SOURCE_RUN, 'completion_sha256': SOURCE_SHA})
        source_schedule = read(source / 'schedule.json')
        physics = {'alpha': cfg['alpha'], 'base_condition_sha256': sha(run / 'condition.json')}
        states = []
        for ci, context in enumerate(contexts):
            matches = [row for row in source_schedule if row['context_id'] == ci
                and row['window_variant'] == 'coverage_v1']
            require(len(matches) == 1, 'Coverage source absent or duplicated')
            source_spec = matches[0]
            require((source_spec['scenario_id'], source_spec['event_id']) ==
                    (context['scenario_id'], context['event_id']), 'Source context differs')
            source_cell = verify_source_cell(source, sm, source_spec)
            result, steps = read(source_cell / 'result.json'), lines(source_cell / 'steps.jsonl.gz')
            plans, points = lines(source_cell / 'plans.jsonl.gz'), lines(source_cell / 'points.jsonl.gz')
            require(result['spec'] == source_spec, 'Historical result/spec mismatch')
            env = make_env(run, source_spec)
            selected, discovery = discover_state_positions(env, result, steps, plans, points, physics, context)
            binding = {'source_run': SOURCE_RUN, 'source_cell': source_spec['cell'],
                'source_result_sha256': sha(source_cell / 'result.json'),
                'source_completion_sha256': sha(source_cell / 'completion_manifest.json'),
                'source_points_sha256': sha(source_cell / 'points.jsonl.gz'),
                'source_steps_sha256': sha(source_cell / 'steps.jsonl.gz'),
                'source_plans_sha256': sha(source_cell / 'plans.jsonl.gz'),
                'source_requests_sha256': sha(source_cell / 'requests.jsonl')}
            save(run / 'discovery' / f'context_{ci:02d}.json', {**discovery, 'source_binding': binding})
            for mode in MODES:
                state_id = f'c{ci:02d}_{mode}'
                row = {'state_id': state_id, 'context_id': ci, 'scenario_id': context['scenario_id'],
                    'event_id': context['event_id'], 'source_cell': source_spec['cell'], 'mode': mode,
                    'status': 'UNAVAILABLE', 'completed_steps': None, 'snapshot': None,
                    'prefix_physical_points': None, 'candidates': [],
                    'reason': f'NO_ELIGIBLE_{mode.upper()}_STATE', 'details': {}}
                if mode in selected:
                    chosen = selected[mode]
                    pos = chosen['completed_steps']
                    n = snapshot_prefix_point_count(result, steps, plans, pos)
                    require(n == chosen['prefix_physical_points'], 'Discovered prefix point count differs')
                    snapshot = restore_snapshot(env, result, steps, plans, points[:n], pos, physics,
                        expected_observation=chosen['observation'], source_binding=binding)
                    require(choose_candidates(snapshot.search, context, mode) == chosen['candidates'],
                            'Candidate choice changed after exact prefix restoration')
                    path = f'snapshots/{state_id}/snapshot.json'
                    save(run / path, snapshot.to_dict())
                    row.update(status='AVAILABLE', completed_steps=pos, snapshot=path,
                        prefix_physical_points=n, candidates=chosen['candidates'], reason=None,
                        details={'eligibility': chosen['eligibility'], 'snapshot_sha256': snapshot.sha256})
                states.append(row)
            save(run / 'preparation_progress.json', {'states': states, 'utc': utc(), 'new_physical_points': 0})
        schedule = schedule_for(states)
        save(run / 'states.json', states)
        save(run / 'schedule.json', schedule)
        effective = sum(row['effective_new_point_allowance'] for row in schedule)
        save(run / 'resource_budget.json', {'contexts': 8, 'state_slots': 16, 'scheduled_forks': len(schedule),
            'allocated_new_physical_points': effective, 'max_new_physical_points': TOTAL_CAP,
            'max_new_numerical_entries': 3 * len(schedule) + 4 * effective, 'source_rebuild_points': 0,
            'workers': 2, 'minimum_free_bytes': MIN_FREE, 'free_bytes_at_prepare': shutil.disk_usage(ROOT).free,
            'new_training': 0, 'software_tests': count_tests})
        for spec in schedule:
            save(run / spec['cell'] / 'spec.json', spec)
        names = [path.relative_to(ROOT).as_posix() for path in (ROOT / 'src/dnr').glob('*.py')]
        names += ['scripts/' + name for name in ('run_clpu_window_teacher.py', 'audit_clpu_window_teacher.py',
            'report_clpu_window_teacher.py', 'run_nature34_clpu_pilot.py', 'nature34_experiment_support.py',
            'run_nature34_development_training.py', 'run_clpu_counterfactual_pilot.py',
            'prepare_nature34_clpu_pilot.py', 'report_nature34_imitation.py', 'audit_d059_supplement.py',
            'run_nature34_clpu_tail_pilot.py', 'run_d060_validation.py', 'd060_validation_common.py',
            'audit_clpu_window_pilot.py', 'audit_clpu_counterfactual_pilot.py',
            'audit_nature34_clpu_branch.py', 'audit_nature34_clpu_pilot.py', 'audit_nature34_joint_planning.py')]
        names += [path.relative_to(ROOT).as_posix() for path in (ROOT / 'tests').glob('*clpu_window*.py')]
        for name in names:
            if 'source/' + name in sm:
                require(sha(ROOT / name) == sm['source/' + name], 'Inherited source changed: ' + name)
        freeze(run, names)
        save(ACTIVE, {'status': 'FROZEN_TEACHER_READY', 'run': run.name, 'scheduled': len(schedule),
            'completed': 0, 'available_states': sum(row['status'] == 'AVAILABLE' for row in states), 'utc': utc()})
        return run
    except BaseException as exc:
        save(run / 'preparation_error.json', {'error': repr(exc), 'traceback': traceback.format_exc(),
            'new_physical_points': 0, 'new_training': 0, 'utc': utc()})
        raise


def verify_execution(run):
    prepared = read(run / 'prepared.json')
    require(sha(run / 'prephysical_freeze_manifest.json') == prepared['freeze_sha256'], 'Freeze changed')
    verify_files(run, read(run / 'prephysical_freeze_manifest.json'))
    for name in prepared['source_files']:
        require(sha(ROOT / name) == sha(run / 'source' / name), 'Executing source changed: ' + name)
    require(dependencies() == read(run / 'dependencies.json'), 'Runtime dependencies changed')
    require(sha(Path(sys.executable)) == read(run / 'python_environment.json')['sha256'], 'Interpreter changed')
    require(read(run / 'config.json')['window_teacher_study'] == STUDY, 'Wrong teacher study')


def worker(run, cell):
    from run_nature34_clpu_pilot import SnapshotOracle, make_env
    from dnr.nature34_clpu_window_counterfactual import fork_snapshot, snapshot_from_dict
    started = time.perf_counter()
    spec = read(cell / 'spec.json')
    try:
        require(cell != run and cell.is_relative_to(run), 'Cell outside run')
        require(not (cell / 'worker_started.json').exists(), 'Repeated teacher fork forbidden')
        verify_execution(run)
        require(spec in read(run / 'schedule.json'), 'Unscheduled teacher fork')
        with (cell / 'worker_started.json').open('x', encoding='utf-8') as stream:
            append(stream, {'utc': utc(), 'pid': os.getpid(), 'spec_sha256': sha(cell / 'spec.json')})
        env = make_env(run, spec)
        save(cell / 'initial_state.json', env.initial.to_dict())
        snapshot = snapshot_from_dict(env, read(run / spec['snapshot']))

        def factory(*, budget):
            require(time.perf_counter() - started <= 1800, 'Teacher worker deadline exceeded')
            return SnapshotOracle(run, cell, env, budget)

        with gzip.open(cell / 'plans.jsonl.gz', 'wt', encoding='utf-8', compresslevel=3) as plans, \
                gzip.open(cell / 'steps.jsonl.gz', 'wt', encoding='utf-8', compresslevel=3) as steps:
            def record_transition(row):
                require(time.perf_counter() - started <= 1800, 'Teacher worker deadline exceeded')
                if row['status'] == 'COMPLETED':
                    append(steps, numeric_lists(row))
            result = fork_snapshot(snapshot, spec['forced_branch_id'], factory,
                max_new_points=spec['max_new_points'], max_expansions=spec['max_expansions'],
                transition_callback=record_transition,
                record_callback=lambda row: append(plans, numeric_lists(row)))
        require(result['new_point_allowance'] == spec['effective_new_point_allowance'], 'Effective point allowance changed')
        if (cell / 'oracle_summary.json').exists():
            live = read(cell / 'oracle_summary.json')
        else:
            require(not result['oracle']['live_initialization_attempted'], 'Physical initialization lacks a final cost ledger')
            live = {'points': 0, 'actual_numerical_entries': 0, 'initialization_numerical_entries': 0,
                    'status': 'NO_NEW_PHYSICAL_CALLS'}
        result.update(spec=spec, live_oracle_summary=live, elapsed_seconds=time.perf_counter() - started,
            new_training=0, optimizer_updates=0, new_source_rebuild_points=0,
            source_snapshot_file_sha256=sha(run / spec['snapshot']),
            formal_holdout_numerical_access=False, tune_numerical_access=False)
        save(cell / 'result.json', numeric_lists(result))
        seal(cell)
        return 0
    except BaseException as exc:
        save(cell / 'error.json', {'status': 'STOPPED_RELIABILITY_OR_IMPLEMENTATION_ERROR',
            'error': repr(exc), 'traceback': traceback.format_exc(), 'spec': spec,
            'elapsed_seconds': time.perf_counter() - started})
        return 1


def costs(run, dispatched):
    points = entries = 0
    unknown = []
    for spec in dispatched:
        cell = run / spec['cell']
        try:
            if (cell / 'oracle_summary.json').exists():
                summary = read(cell / 'oracle_summary.json')
            else:
                result = read(cell / 'result.json')
                require(result['oracle']['live_initialization_attempted'] is False,
                        'Unfinished initialization cost unknown')
                summary = result['live_oracle_summary']
                require(summary['status'] == 'NO_NEW_PHYSICAL_CALLS', 'Missing live accounting')
            require(type(summary['points']) is int and summary['points'] >= 0
                    and type(summary['actual_numerical_entries']) is int and summary['actual_numerical_entries'] >= 0,
                    'Invalid physical accounting')
            points += summary['points']
            entries += summary['actual_numerical_entries']
        except (OSError, ValueError, KeyError, TypeError):
            unknown.append(spec['cell'])
    return {'known_physical_points': points, 'known_numerical_entries': entries, 'unknown_cost_cells': unknown}


def execute(run):
    require(not (run / 'execution_started.json').exists(), 'Teacher already started; no automatic repeat')
    verify_execution(run)
    with (run / 'execution_started.json').open('x', encoding='utf-8') as stream:
        append(stream, {'utc': utc(), 'pid': os.getpid()})
    schedule = read(run / 'schedule.json')
    completed, dispatched, errors, exits, active = [], [], [], [], {}
    started = time.perf_counter()

    def finished(proc, spec, code, drained=False):
        exits.append({'cell': spec['cell'], 'exit_code': code, 'launcher_pid': proc.pid,
                      'drained_after_controller_error': drained, 'utc': utc()})
        if code == 0 and (run / spec['cell'] / 'completion_manifest.json').exists():
            completed.append(spec['cell'])
        else:
            errors.append(exits[-1])

    try:
        while active or (len(dispatched) < len(schedule) and not errors):
            if not errors and shutil.disk_usage(ROOT).free < MIN_FREE:
                errors.append({'error': 'LOW_DISK_STOP_NEW_DISPATCH'})
            while not errors and len(active) < 2 and len(dispatched) < len(schedule):
                spec = schedule[len(dispatched)]
                cell = run / spec['cell']
                out = (cell / 'stdout.log').open('w', encoding='utf-8')
                err = (cell / 'stderr.log').open('w', encoding='utf-8')
                try:
                    proc = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
                        'worker', '--run', str(run), '--cell', str(cell)], cwd=ROOT, stdout=out, stderr=err,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                except BaseException:
                    out.close(); err.close(); raise
                active[spec['cell']] = (proc, spec, out, err, time.perf_counter())
                dispatched.append(spec)
            for key, (proc, spec, out, err, began) in list(active.items()):
                code = proc.poll()
                if code is None:
                    if time.perf_counter() - began > 1800 and not errors:
                        errors.append({'cell': key, 'error': 'WATCHDOG_STOP_NEW_DISPATCH_DRAIN_STARTED'})
                    continue
                out.close(); err.close(); del active[key]
                finished(proc, spec, code)
            progress = {'status': 'RUNNING' if not errors else 'DRAINING_AFTER_ERROR', 'run': run.name,
                'scheduled': len(schedule), 'completed': len(completed), 'dispatched': len(dispatched),
                'active': list(active), 'errors': errors, 'utc': utc(), 'wall_seconds': time.perf_counter() - started}
            save(run / 'controller_progress.json', progress)
            save(ACTIVE, progress)
            if active:
                time.sleep(1)
    except BaseException as exc:
        errors.append({'error': repr(exc), 'traceback': traceback.format_exc()})
    finally:
        for proc, spec, out, err, began in active.values():
            try:
                finished(proc, spec, proc.wait(), True)
            except BaseException as exc:
                errors.append({'cell': spec['cell'], 'drain_error': repr(exc)})
            for stream in (out, err):
                try:
                    stream.close()
                except BaseException as exc:
                    errors.append({'cell': spec['cell'], 'stream_close_error': repr(exc)})
    accounting = costs(run, dispatched)
    status = ('COMPLETE_CELLS_PENDING_AUDIT' if not errors and not accounting['unknown_cost_cells']
              and len(completed) == len(schedule) else 'INCONCLUSIVE')
    summary = {'status': status, 'scheduled': len(schedule), 'completed': completed, 'dispatched': dispatched,
        'exits': exits, 'errors': errors, **accounting, 'wall_seconds': time.perf_counter() - started, 'utc': utc()}
    save(run / 'controller_completion.json', summary)
    save(ACTIVE, {'status': status, 'run': run.name, 'completed': len(completed),
                 'scheduled': len(schedule), **accounting, 'errors': errors, 'utc': utc()})
    return 0 if status == 'COMPLETE_CELLS_PENDING_AUDIT' else 1


def finalize(run):
    audit = read(run / 'audit/summary.json')
    report = read(run / 'derived_report/summary.json')
    require(audit['status'] == 'PASS' and audit['audited_cells'] == len(read(run / 'schedule.json')),
            'Independent teacher audit incomplete')
    for name in ('audit', 'derived_report'):
        verify_files(run / name, read(run / name / 'completion_manifest.json'))
    terminal = {'status': 'COMPLETE_WINDOW_TEACHER_AUDITED_AND_REPORTED', 'run': run.name, 'utc': utc(),
        'audit_sha256': sha(run / 'audit/summary.json'), 'report_sha256': sha(run / 'derived_report/summary.json'),
        'report_manifest_sha256': sha(run / 'derived_report/completion_manifest.json'),
        'physical_points': audit['physical_points'], 'numerical_entries': audit['numerical_entries'],
        'new_training': 0, 'all_valid_failures_preserved': True,
        'scientific_status': report.get('status', report.get('scientific_status', 'SEE_DERIVED_REPORT'))}
    save(run / 'terminal_status.json', terminal)
    seal(run)
    save(ACTIVE, {**terminal, 'status': 'COMPLETE_WINDOW_TEACHER_SEALED',
                  'completion_sha256': sha(run / 'completion_manifest.json')})
    return terminal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'worker', 'execute', 'finalize'))
    parser.add_argument('--tests', type=Path)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--cell', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        require(args.tests is not None, '--tests required')
        print(prepare(args.tests.resolve()), flush=True)
        return 0
    require(args.run is not None, '--run required')
    run = args.run.resolve()
    require(run.parent == ROOT / 'runs', 'Run outside project runs')
    if args.command == 'worker':
        require(args.cell is not None, '--cell required')
        return worker(run, args.cell.resolve())
    if args.command == 'finalize':
        print(canonical(finalize(run)), flush=True)
        return 0
    return execute(run)


if __name__ == '__main__':
    raise SystemExit(main())
