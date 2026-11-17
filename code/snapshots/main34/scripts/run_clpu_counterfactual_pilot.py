"""Bounded route-B teacher pilot; reuse sealed prefixes, pay for each new fork.

Preparation restores the predeclared training states without physical calls.
Every measured alternative runs in a fresh process; no policy is trained here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
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

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
sys.dont_write_bytecode = True

from d060_validation_common import SOURCE_RUN, SOURCE_SHA, read, sha, canonical, require, normalize_manifest
from nature34_experiment_support import create_run, dependencies, freeze, numeric_lists, append
from run_nature34_clpu_tail_pilot import save
from run_d060_validation import ROOT_INPUTS, bound_copy

ACTIVE = ROOT / 'artifacts/clpu_counterfactual_pilot_active.json'
STUDY = 'CLPU_COUNTERFACTUAL_S1_V1'
POINT_CAP, STEP_CAP, MIN_DISK = 128, 16, 12 * 2**30


def utc():
    return datetime.now(timezone.utc).isoformat()


def lines(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def seal(directory):
    save(directory / 'completion_manifest.json', {
        p.relative_to(directory).as_posix(): sha(p) for p in directory.rglob('*')
        if p.is_file() and p.name not in ('completion_manifest.json', 'stdout.log', 'stderr.log')})


def choose_candidates(search, source_result, after_steps, context):
    observation = numeric_lists(search.observation('raw'))
    ids = observation['branch_ids']
    require(len(ids) >= 2 and all(observation['mask']), 'At least two legal candidates required')
    teacher = source_result['trace'][after_steps]['branch_id']
    greedy = ids[search.rule_action('greedy')]
    require(teacher in ids and greedy in ids, 'Original teacher/Greedy choice not in restored window')
    chosen = list(dict.fromkeys([teacher, greedy]))
    salt = f"route-b-s1-candidate-v1|{context['scenario_id']}|{context['event_id']}|{after_steps}|"
    for candidate in sorted(ids, key=lambda bid: hashlib.sha256((salt + str(bid)).encode()).hexdigest()):
        if candidate not in chosen:
            chosen.append(candidate)
        if len(chosen) == min(3, len(ids)):
            break
    return [{'branch_id': bid, 'roles': [role for role, value in
             [('source_teacher', teacher), ('common_greedy', greedy)] if value == bid] or ['fixed_hash_fill']}
            for bid in chosen]


def verify_files(directory, manifest):
    for name, digest in manifest.items():
        target = (directory / name).resolve()
        require(target.is_relative_to(directory.resolve()) and sha(target) == digest,
                'Changed or outside frozen file: ' + name)


def validate_population(population, training_contexts):
    faults = {r['scenario_id'] for r in training_contexts}
    events = {r['event_id'] for r in training_contexts}
    require(len(faults) == 8 and events == {20, 3, 6, 30} and len(training_contexts) == 32,
            'Original 8 by 4 training population required')
    faults = sorted(faults, key=lambda s: hashlib.sha256(('route-b-s1-fault-v1|' + s).encode()).hexdigest())
    events = sorted(events, key=lambda e: hashlib.sha256(('route-b-s1-event-v1|' + str(e)).encode()).hexdigest())
    expected = [(fault, events[i % 4]) for i, fault in enumerate(faults)]
    require([(c['scenario_id'], c['event_id']) for c in population['contexts']] == expected,
            'S1 population differs from the predeclared balanced hash assignment')


def prepare(population_path, tests_path):
    from run_nature34_clpu_pilot import make_env
    from dnr.nature34_clpu_counterfactual import (
        restore_snapshot, snapshot_prefix_point_count, SnapshotUnavailable)
    import xml.etree.ElementTree as ET
    cases = list(ET.parse(tests_path).getroot().iter('testcase'))
    require(cases and not any(list(c.iter('failure')) + list(c.iter('error')) + list(c.iter('skipped'))
                             for c in cases), 'Tests not all passed')
    population = read(population_path)
    require(population['source_run'] == SOURCE_RUN and len(population['contexts']) == 8,
            'Expected eight original training contexts')
    source = ROOT / 'runs' / SOURCE_RUN
    require(sha(source / 'completion_manifest.json') == SOURCE_SHA, 'Sealed D059 identity changed')
    sm = normalize_manifest(read(source / 'completion_manifest.json'))
    for name in ('config.json', 'training_scenarios.json', 'selected_events.json', 'training_contexts.json'):
        require(name in sm and sha(source / name) == sm[name], 'Sealed source input changed: ' + name)
    tc = read(source / 'training_contexts.json')
    require(sha(source / 'training_contexts.json') == sm['training_contexts.json'], 'Training partition changed')
    validate_population(population, tc)
    require(all({'scenario_id': c['scenario_id'], 'event_id': c['event_id']} in tc
                for c in population['contexts']), 'Non-training context forbidden')
    cfg = deepcopy(read(source / 'config.json'))
    cfg.update(counterfactual_study=STUDY)
    cfg['counterfactual'] = {'states_after_actions': [0, 8], 'max_candidates': 3,
        'max_new_physical_points_per_fork': POINT_CAP, 'max_expansions_per_fork': STEP_CAP,
        'max_new_physical_points_total': 6144, 'max_workers': 2,
        'worker_watchdog_seconds': 1800, 'minimum_free_disk_bytes': MIN_DISK,
        'source_rebuild_physical_points': 0, 'optimizer_updates': 0,
        'gate': {'minimum_distinguishable_states': 4, 'span_pp': .1,
                 'minimum_faults': 2, 'minimum_events': 2, 'minimum_delayed_or_reversed_states': 2},
        'candidate_hash_domain': 'route-b-s1-candidate-v1',
        'no_automatic_expansion_or_hyperparameter_scan': True}
    require(shutil.disk_usage(ROOT).free >= MIN_DISK + 6 * 2**30, 'Insufficient pilot disk reserve')
    run = create_run('clpu_counterfactual_s1', cfg)
    save(run / 'authorization.json', {'execution_authorized': True,
        'user_message': 'ok，按你的下一步做吧，还是不需要如此严苛（必要时候可进行数据的删减或修改），要保证最终结果好看，可以逻辑闭环地讲故事即可',
        'scope': 'Existing-log diagnosis and eight-training-context finite-rollout label pilot; no model training or formal test',
        'effective_interpretation': 'Reuse evidence and compact duplicate records; retain measured values and valid failures',
        'utc': utc()})
    for name in ROOT_INPUTS:
        bound_copy(source, run / name, sm, name)
    for p in (source / 'source_inputs').rglob('*'):
        if p.is_file():
            bound_copy(source, run / p.relative_to(source), sm, p.relative_to(source).as_posix())
    scenarios = read(source / 'training_scenarios.json')
    events = read(source / 'selected_events.json')
    save(run / 'selected_scenarios.json', [s for s in scenarios if s['scenario_id'] in
         {c['scenario_id'] for c in population['contexts']}])
    save(run / 'selected_events.json', [e for e in events if e['event_id'] in {3, 6, 20, 30}])
    save(run / 'training_contexts.json', [{'scenario_id': c['scenario_id'], 'event_id': c['event_id']}
                                         for c in population['contexts']])
    shutil.copyfile(population_path, run / 'population.json')
    shutil.copyfile(tests_path, run / 'software_tests.xml')
    shutil.copyfile(ROOT / 'docs/clpu_counterfactual_s1_protocol_v1.md', run / 'protocol.md')
    states, schedule = [], []
    for ci, context in enumerate(population['contexts']):
        cell = (source / context['source_cell']).resolve()
        require(cell.is_relative_to(source.resolve()) and cell != source.resolve(), 'Source cell outside D059')
        for name in ('result.json', 'completion_manifest.json', 'points.jsonl.gz', 'transitions.jsonl.gz'):
            relative = (cell / name).relative_to(source).as_posix()
            require(relative in sm and sha(cell / name) == sm[relative], 'Sealed teacher input changed: ' + relative)
        result = read(cell / 'result.json')
        require(sha(cell / 'result.json') == context['source_result_sha256'], 'Teacher result changed')
        require(result['spec']['phase'] == 'teacher' and result['spec']['point_budget'] == 256,
                'Only original 256-point rule teachers permitted')
        require((result['spec']['scenario_id'], result['spec']['event_id']) ==
                (context['scenario_id'], context['event_id']), 'Teacher context differs')
        prefix_observations = {}
        for ti, transition in enumerate(lines(cell / 'transitions.jsonl.gz')):
            if ti in (0, 8):
                prefix_observations[ti] = transition['obs_raw']
            if ti >= 8:
                break
        env = make_env(run, context)
        for steps in (0, 8):
            state_id = f'context{ci:02d}_after{steps:02d}'
            row = {**{k: context[k] for k in ('scenario_id', 'event_id', 'source_cell')},
                   'state_id': state_id, 'after_steps': steps}
            try:
                n = snapshot_prefix_point_count(result, steps)
                point_rows = []
                for p in lines(cell / 'points.jsonl.gz'):
                    if len(point_rows) >= n:
                        break
                    point_rows.append(p)
                require(len(point_rows) == n, 'Missing source prefix physical records')
                binding = {'source_run': SOURCE_RUN, 'source_cell': context['source_cell'],
                    'source_result_sha256': sha(cell / 'result.json'),
                    'source_completion_sha256': sha(cell / 'completion_manifest.json'),
                    'source_points_sha256': sha(cell / 'points.jsonl.gz'),
                    'source_transitions_sha256': sha(cell / 'transitions.jsonl.gz')}
                snap = restore_snapshot(env, result, point_rows, steps,
                    {'alpha': .5, 'base_condition_sha256': sha(run / 'condition.json')},
                    expected_observation=prefix_observations.get(steps), source_binding=binding)
                selected = choose_candidates(snap.search, result, steps, context)
                location = run / 'snapshots' / state_id
                location.mkdir(parents=True)
                save(location / 'snapshot.json', snap.to_dict())
                row.update(status='AVAILABLE', snapshot=f'snapshots/{state_id}/snapshot.json',
                           prefix_physical_points=n, candidates=selected)
                for candidate in selected:
                    schedule.append({'state_id': state_id, 'snapshot': row['snapshot'],
                        'scenario_id': context['scenario_id'], 'event_id': context['event_id'],
                        'after_steps': steps, **candidate, 'max_new_points': min(POINT_CAP, 256 - n),
                        'max_expansions': STEP_CAP})
            except SnapshotUnavailable as exc:
                row.update(status='UNAVAILABLE', reason=exc.code, details=exc.details, candidates=[])
            states.append(row)
            save(run / 'preparation_progress.json', {'utc': utc(), 'states': states,
                                                     'new_physical_points': 0})
    schedule.sort(key=lambda r: hashlib.sha256(('route-b-s1-order-v1|' + canonical(r)).encode()).hexdigest())
    for i, spec in enumerate(schedule):
        spec.update(episode_id=i, cell=f'forks/{i:03d}')
    require(len(schedule) <= 48 and sum(s['max_new_points'] for s in schedule) <= 6144,
            'S1 budget exceeded')
    save(run / 'state_inventory.json', states)
    save(run / 'schedule.json', schedule)
    save(run / 'resource_budget.json', {'scheduled_forks': len(schedule),
        'allocated_new_physical_points': sum(s['max_new_points'] for s in schedule),
        'max_new_physical_points': 6144, 'source_rebuild_points': 0,
        'max_new_numerical_entries': sum(3 + 4 * s['max_new_points'] for s in schedule),
        'workers': 2, 'initial_free_bytes': shutil.disk_usage(ROOT).free,
        'minimum_free_bytes': MIN_DISK})
    for spec in schedule:
        save(run / spec['cell'] / 'spec.json', spec)
    # Freeze actual mechanics and callers; existing physics implementation is reused verbatim.
    sources = [p.relative_to(ROOT).as_posix() for p in (ROOT / 'src/dnr').glob('*.py')]
    sources += ['scripts/' + name for name in ('run_clpu_counterfactual_pilot.py',
        'prepare_clpu_counterfactual_population.py', 'audit_clpu_counterfactual_pilot.py',
        'report_clpu_counterfactual_pilot.py',
        'run_nature34_clpu_pilot.py', 'nature34_experiment_support.py',
        'run_nature34_development_training.py', 'audit_nature34_clpu_branch.py',
        'audit_nature34_clpu_pilot.py', 'audit_nature34_joint_planning.py',
        'run_nature34_clpu_tail_pilot.py', 'run_d060_validation.py', 'd060_validation_common.py')]
    sources += [p.relative_to(ROOT).as_posix() for p in (ROOT / 'tests').glob('*counterfactual*.py')]
    for name in sources:
        old = 'source/' + name
        if old in sm:
            require(sha(ROOT / name) == sm[old], 'Inherited science changed: ' + name)
    freeze(run, sources)
    save(ACTIVE, {'status': 'FROZEN_S1_READY', 'run': run.name, 'scheduled': len(schedule), 'completed': 0,
                  'available_states': sum(s['status'] == 'AVAILABLE' for s in states), 'utc': utc()})
    return run


def worker(run, cell):
    from run_nature34_clpu_pilot import SnapshotOracle, make_env
    from dnr.nature34_clpu_counterfactual import snapshot_from_dict, fork_snapshot
    started = time.perf_counter()
    spec = read(cell / 'spec.json')
    try:
        require(not (cell / 'worker_started.json').exists(), 'No automatic repeat of a measured fork')
        prepared = read(run / 'prepared.json')
        require(sha(run / 'prephysical_freeze_manifest.json') == prepared['freeze_sha256'], 'Freeze changed')
        verify_files(run, read(run / 'prephysical_freeze_manifest.json'))
        for name in prepared['source_files']:
            require(sha(ROOT / name) == sha(run / 'source' / name), 'Executing source changed: ' + name)
        require(spec in read(run / 'schedule.json'), 'Fork not scheduled')
        require(dependencies() == read(run / 'dependencies.json'), 'Runtime dependencies changed')
        with (cell / 'worker_started.json').open('x', encoding='utf-8') as stream:
            append(stream, {'utc': utc(), 'pid': os.getpid(), 'spec_sha256': sha(cell / 'spec.json')})
        env = make_env(run, spec)
        save(cell / 'initial_state.json', env.initial.to_dict())
        snapshot = snapshot_from_dict(env, read(run / spec['snapshot']))
        def live_factory(*, budget):
            return SnapshotOracle(run, cell, env, budget)
        with gzip.open(cell / 'plans_delta.jsonl.gz', 'wt', encoding='utf-8', compresslevel=3) as plans, \
                gzip.open(cell / 'transitions_delta.jsonl.gz', 'wt', encoding='utf-8', compresslevel=3) as transitions:
            result = fork_snapshot(snapshot, spec['branch_id'], live_factory,
                max_new_points=spec['max_new_points'], max_expansions=spec['max_expansions'],
                transition_callback=lambda row: append(transitions, numeric_lists(row)),
                record_callback=lambda row: append(plans, numeric_lists(row)))
        result.update(spec=spec, elapsed_seconds=time.perf_counter() - started,
            optimizer_updates=0, new_source_rebuild_points=0, formal_holdout_numerical_access=False,
            tune_numerical_access=False, source_snapshot_file_sha256=sha(run / spec['snapshot']))
        if (cell / 'oracle_summary.json').exists():
            result['live_oracle_summary'] = read(cell / 'oracle_summary.json')
        else:
            result['live_oracle_summary'] = {'points': 0, 'actual_numerical_entries': 0,
                'initialization_numerical_entries': 0, 'status': 'NO_NEW_PHYSICAL_CALLS'}
        save(cell / 'result.json', numeric_lists(result))
        seal(cell)
        return 0
    except BaseException as exc:
        save(cell / 'error.json', {'status': 'STOPPED_RELIABILITY_OR_IMPLEMENTATION_ERROR',
            'error': repr(exc), 'traceback': traceback.format_exc(), 'spec': spec,
            'elapsed_seconds': time.perf_counter() - started})
        return 1


def current_cost(run, rows):
    total = numerical = 0
    unknown = []
    for spec in rows:
        cell = run / spec['cell']
        try:
            if (cell / 'result.json').exists():
                oracle = read(cell / 'result.json')['live_oracle_summary']
                total += oracle['points']; numerical += oracle['actual_numerical_entries']
            elif (cell / 'oracle_summary.json').exists():
                oracle = read(cell / 'oracle_summary.json')
                total += oracle['points']; numerical += oracle['actual_numerical_entries']
            else:
                unknown.append(spec['cell'])
        except (OSError, ValueError, KeyError, TypeError):
            unknown.append(spec['cell'])
    return {'known_new_physical_points': total, 'known_new_numerical_entries': numerical,
            'unknown_cost_cells': unknown}


def execute(run):
    require(not (run / 'execution_started.json').exists(), 'This frozen pilot already started')
    verify_files(run, read(run / 'prephysical_freeze_manifest.json'))
    save(run / 'execution_started.json', {'utc': utc(), 'pid': os.getpid()})
    schedule = read(run / 'schedule.json')
    completed, dispatched, errors, active = [], [], [], {}
    wall = time.perf_counter()
    try:
        while active or (len(dispatched) < len(schedule) and not errors):
            if not errors and shutil.disk_usage(ROOT).free < MIN_DISK:
                errors.append({'error': 'LOW_DISK_STOP_NEW_DISPATCH', 'free_bytes': shutil.disk_usage(ROOT).free})
            while not errors and len(active) < 2 and len(dispatched) < len(schedule):
                spec = schedule[len(dispatched)]; cell = run / spec['cell']
                out = (cell / 'stdout.log').open('w', encoding='utf-8')
                err = (cell / 'stderr.log').open('w', encoding='utf-8')
                proc = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
                    'worker', '--run', str(run), '--cell', str(cell)], cwd=ROOT,
                    stdout=out, stderr=err, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                active[spec['cell']] = (proc, spec, out, err, time.perf_counter())
                dispatched.append(spec)
            for key, (proc, spec, out, err, began) in list(active.items()):
                code = proc.poll()
                if code is None:
                    if time.perf_counter() - began > 1800 and not errors:
                        errors.append({'cell': key, 'error': 'WATCHDOG_STOP_DISPATCH_DRAIN_CURRENT_WORKERS'})
                    continue
                out.close(); err.close(); del active[key]
                status = {'cell': key, 'exit_code': code, 'launcher_pid': proc.pid, 'utc': utc()}
                if code != 0 or not (run / key / 'completion_manifest.json').exists():
                    errors.append(status)
                else:
                    completed.append(status)
            progress = {'status': 'RUNNING_S1' if not errors else 'DRAINING_AFTER_ERROR', 'run': run.name,
                'completed': len(completed), 'scheduled': len(schedule), 'dispatched': len(dispatched),
                'active': list(active), 'errors': errors, 'utc': utc(),
                'elapsed_seconds': time.perf_counter() - wall}
            save(ACTIVE, progress)
            save(run / 'controller_progress.json', progress)
            if active:
                time.sleep(1)
    except BaseException as exc:
        errors.append({'error': repr(exc), 'traceback': traceback.format_exc()})
    finally:
        # An interrupted controller must not leave hidden writers unaccounted for.
        for proc, spec, out, err, began in list(active.values()):
            try:
                code = proc.wait()
                status = {'cell': spec['cell'], 'exit_code': code, 'launcher_pid': proc.pid, 'utc': utc(),
                          'drained_after_controller_error': True}
                if code == 0 and (run / spec['cell'] / 'completion_manifest.json').exists():
                    completed.append(status)
                else:
                    errors.append(status)
            except BaseException as exc:
                errors.append({'cell': spec['cell'], 'drain_error': repr(exc)})
            for stream in (out, err):
                try:
                    stream.close()
                except BaseException as exc:
                    errors.append({'cell': spec['cell'], 'stream_close_error': repr(exc)})
    costs = current_cost(run, dispatched)
    summary = {'status': 'COMPLETE_FORKS_PENDING_AUDIT' if not errors and not costs['unknown_cost_cells']
               and len(completed) == len(schedule) else 'INCONCLUSIVE',
        'scheduled': len(schedule), 'completed': completed, 'dispatched': dispatched, 'errors': errors,
        'wall_seconds': time.perf_counter() - wall, **costs, 'utc': utc()}
    save(run / 'controller_completion.json', summary)
    save(ACTIVE, {**summary, 'run': run.name, 'completed': len(completed)})
    return 0 if summary['status'] == 'COMPLETE_FORKS_PENDING_AUDIT' else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'worker', 'execute'])
    parser.add_argument('--population', type=Path)
    parser.add_argument('--tests', type=Path)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--cell', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(prepare(args.population.resolve(), args.tests.resolve()))
        return 0
    run = args.run.resolve()
    require(run.parent == ROOT / 'runs', 'Run outside workspace')
    return worker(run, args.cell.resolve()) if args.command == 'worker' else execute(run)


if __name__ == '__main__':
    raise SystemExit(main())
