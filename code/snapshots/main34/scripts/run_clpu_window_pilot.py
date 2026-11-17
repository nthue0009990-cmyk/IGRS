"""One frozen, training-only candidate-window comparison with a paid Beam guard."""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import asdict
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
from d060_validation_common import SOURCE_RUN, SOURCE_SHA, normalize_manifest, canonical, read, sha, require
from nature34_experiment_support import create_run, freeze, dependencies, append, numeric_lists
from run_nature34_clpu_tail_pilot import save
from run_clpu_counterfactual_pilot import validate_population, verify_files, seal, utc
from run_d060_validation import ROOT_INPUTS, bound_copy

STUDY = 'CLPU_WINDOW_COVERAGE_PILOT_V1'
ACTIVE = ROOT / 'artifacts/clpu_window_pilot_active.json'
GROUPS = {'G0_original_beam': 'original', 'G1_coverage_window_beam': 'coverage_v1'}
MIN_FREE = 12 * 2**30


def schedule_for(contexts):
    require(len(contexts) == 8 and len({(c['scenario_id'], c['event_id']) for c in contexts}) == 8,
            'Eight unique paired training contexts required')
    rows = []
    for i, c in enumerate(contexts):
        for group, variant in GROUPS.items():
            rows.append({'context_id': i, 'scenario_id': c['scenario_id'], 'event_id': c['event_id'],
                'group': group, 'method': 'beam', 'window_variant': variant, 'seed': 0,
                'point_budget': 256, 'deterministic': True, 'phase': 'training_mechanism_evaluation'})
    rows.sort(key=lambda r: hashlib.sha256(('clpu-window-order-v1|' + canonical(r)).encode()).hexdigest())
    for i, row in enumerate(rows):
        row.update(episode_id=i, cell=f'episodes/{i:03d}')
    return rows


def prepare(population_path, tests_path):
    from dnr.nature34_clpu_window import WINDOW_RULE, WINDOW_RULE_VERSION
    cases = list(ET.parse(tests_path).getroot().iter('testcase'))
    require(cases and not any(list(c.iter('failure')) + list(c.iter('error')) + list(c.iter('skipped'))
                             for c in cases), 'Nonempty passed software tests required')
    source = ROOT / 'runs' / SOURCE_RUN
    require(sha(source / 'completion_manifest.json') == SOURCE_SHA, 'D059 sealed source changed')
    sm = normalize_manifest(read(source / 'completion_manifest.json'))
    for name in ('config.json', 'training_contexts.json', 'training_scenarios.json', 'selected_events.json'):
        require(sha(source / name) == sm[name], 'Source input changed: ' + name)
    population = read(population_path)
    require(population['source_run'] == SOURCE_RUN, 'Wrong population provenance')
    validate_population(population, read(source / 'training_contexts.json'))
    schedule = schedule_for(population['contexts'])
    cfg = deepcopy(read(source / 'config.json'))
    cfg.update(window_study=STUDY, window_rule_version=WINDOW_RULE_VERSION, window_rule=WINDOW_RULE)
    cfg['window_pilot'] = {'groups': GROUPS, 'max_workers': 2, 'worker_watchdog_seconds': 1800,
        'total_physical_budget': 4096, 'per_arm_physical_budget': 256,
        'protected_output_cost': 'sum of both independent cold arms',
        'protected_output_selection': 'strictly higher certified energy; exact ties original Beam',
        'target_reach_tolerance_pp': 1e-9, 'anytime_choices': [16, 64, 128, 256],
        'no_training': True, 'no_formal_test': True, 'no_automatic_retry_or_parameter_scan': True}
    require(shutil.disk_usage(ROOT).free >= MIN_FREE + 5 * 2**30, 'Insufficient small-pilot storage reserve')
    run = create_run('clpu_window_pilot', cfg)
    for name in ROOT_INPUTS:
        bound_copy(source, run / name, sm, name)
    for p in (source / 'source_inputs').rglob('*'):
        if p.is_file():
            bound_copy(source, run / p.relative_to(source), sm, p.relative_to(source).as_posix())
    wanted = {c['scenario_id'] for c in population['contexts']}
    save(run / 'selected_scenarios.json', [s for s in read(source / 'training_scenarios.json') if s['scenario_id'] in wanted])
    save(run / 'selected_events.json', [e for e in read(source / 'selected_events.json') if e['event_id'] in {20, 3, 6, 30}])
    save(run / 'training_contexts.json', [{'scenario_id': c['scenario_id'], 'event_id': c['event_id']}
                                         for c in population['contexts']])
    shutil.copyfile(population_path, run / 'population.json')
    shutil.copyfile(tests_path, run / 'software_tests.xml')
    shutil.copyfile(ROOT / 'docs/clpu_window_pilot_protocol_v1.md', run / 'protocol.md')
    save(run / 'authorization.json', {'utc': utc(), 'execution_authorized': True,
        'user_message': 'ok，那先按你说的改候选窗口，保留Beam保底，做一轮小规模对照，看看能不能更早找到好方案，赶紧把效果搞出来，不需要如此严苛，要保证最终结果好看，可以逻辑闭环地讲故事即可',
        'scope': 'One eight-context fixed candidate-window comparison; full paid original Beam guard; no training',
        'data_handling': 'Compact duplicated logs and reuse old inputs; retain all measured outcomes and valid failures'})
    save(run / 'source_binding.json', {'run': SOURCE_RUN, 'completion_sha256': SOURCE_SHA,
        'population_sha256': sha(population_path), 'new_scenario_selection': False,
        'diagnosis_artifact': 'artifacts/counterfactual_s1_visibility_diagnosis_20261027.json',
        'diagnosis_sha256': sha(ROOT / 'artifacts/counterfactual_s1_visibility_diagnosis_20261027.json')})
    save(run / 'schedule.json', schedule)
    save(run / 'resource_budget.json', {'cells': 16, 'contexts': 8, 'physical_points': 4096,
        'numerical_entries': 16 * 3 + 4 * 4096, 'workers': 2, 'minimum_free_bytes': MIN_FREE,
        'free_bytes_at_prepare': shutil.disk_usage(ROOT).free, 'new_training': 0})
    for spec in schedule:
        save(run / spec['cell'] / 'spec.json', spec)
    source_names = [p.relative_to(ROOT).as_posix() for p in (ROOT / 'src/dnr').glob('*.py')]
    source_names += ['scripts/' + name for name in (
        'run_clpu_window_pilot.py', 'audit_clpu_window_pilot.py', 'report_clpu_window_pilot.py',
        'run_nature34_clpu_pilot.py', 'nature34_experiment_support.py', 'run_nature34_development_training.py',
        'run_clpu_counterfactual_pilot.py', 'run_nature34_clpu_tail_pilot.py', 'run_d060_validation.py',
        'd060_validation_common.py', 'audit_nature34_clpu_branch.py', 'audit_nature34_clpu_pilot.py',
        'audit_nature34_joint_planning.py')]
    source_names += [p.relative_to(ROOT).as_posix() for p in (ROOT / 'tests').glob('*clpu_window*.py')]
    for name in source_names:
        if 'source/' + name in sm:
            require(sha(ROOT / name) == sm['source/' + name], 'Inherited science changed: ' + name)
    freeze(run, source_names)
    save(ACTIVE, {'status': 'FROZEN_NOT_STARTED', 'run': run.name, 'scheduled': 16, 'completed': 0, 'utc': utc()})
    return run


def checkpoint(search, started):
    best = search.incumbent
    certified = bool(best and best.get('certified_complete') is True and best.get('validation', {}).get('status') == 'COMPLETE')
    return {'search_choices': search._beam_turn, 'physical_points': search.oracle.points,
        'numerical_entries': search.oracle.factory.stats()['returned']['numerical_solution_entries'],
        'elapsed_seconds': time.perf_counter() - started,
        'quality_pct': 100 * search.quality if certified else None, 'fully_certified': certified}


def worker(run, cell):
    from run_nature34_clpu_pilot import SnapshotOracle, make_env
    from dnr.nature34_clpu_branch import BranchConfig, BranchSearch
    from dnr.nature34_clpu_window import WindowBranchSearch
    started = time.perf_counter()
    spec = read(cell / 'spec.json')
    try:
        require(cell != run and cell.is_relative_to(run), 'Cell outside run')
        require(not (cell / 'worker_started.json').exists(), 'Repeated task forbidden')
        prepared = read(run / 'prepared.json')
        require(sha(run / 'prephysical_freeze_manifest.json') == prepared['freeze_sha256'], 'Freeze changed')
        verify_files(run, read(run / 'prephysical_freeze_manifest.json'))
        for name in prepared['source_files']:
            require(sha(ROOT / name) == sha(run / 'source' / name), 'Executing source changed: ' + name)
        require(spec in read(run / 'schedule.json') and GROUPS[spec['group']] == spec['window_variant'], 'Unscheduled variant')
        require(dependencies() == read(run / 'dependencies.json'), 'Runtime dependencies changed')
        require(sha(Path(sys.executable)) == read(run / 'python_environment.json')['sha256'], 'Interpreter changed')
        cfg = read(run / 'config.json')
        require(cfg['window_study'] == STUDY and cfg['branch_search'] == asdict(BranchConfig()), 'Search limits changed')
        with (cell / 'worker_started.json').open('x', encoding='utf-8') as stream:
            append(stream, {'pid': os.getpid(), 'utc': utc(), 'spec_sha256': sha(cell / 'spec.json')})
        env = make_env(run, spec)
        save(cell / 'initial_state.json', env.initial.to_dict())
        with gzip.open(cell / 'plans.jsonl.gz', 'wt', encoding='utf-8', compresslevel=3) as plans, \
                gzip.open(cell / 'steps.jsonl.gz', 'wt', encoding='utf-8', compresslevel=3) as steps, \
                SnapshotOracle(run, cell, env, 256) as oracle:
            engine = BranchSearch if spec['window_variant'] == 'original' else WindowBranchSearch
            search = engine(env, oracle, BranchConfig(**cfg['branch_search']),
                            record_callback=lambda row: append(plans, numeric_lists(row)))
            checkpoints = [checkpoint(search, started)]
            while not search.done:
                require(time.perf_counter() - started <= 1800, 'Worker deadline exceeded; partial evidence retained')
                index = search.rule_action('beam')
                window_ids = [node['id'] for node in search._window()]
                decision = {'transition_id': search._beam_turn, 'status': 'STARTED', 'method': 'beam',
                    'action': index, 'branch_id': window_ids[index], 'window_branch_ids': window_ids,
                    'oracle_before': {'requests': oracle.requests, 'points': oracle.points},
                    'elapsed_seconds': time.perf_counter() - started}
                append(steps, decision)
                response = search.step(index)
                decision.update(status='COMPLETED', response=numeric_lists(response),
                    oracle_after={'requests': oracle.requests, 'points': oracle.points},
                    elapsed_seconds=time.perf_counter() - started)
                append(steps, decision)
                checkpoints.append(checkpoint(search, started))
            result = {'status': 'INITIAL_FAIL' if search.stop_reason == 'INITIAL_FAIL' else
                       'CERTIFIED_INCUMBENT' if search.incumbent is not None else 'NO_CERTIFIED_COMPLETE_PLAN',
                'config': asdict(search.config), 'stop_reason': search.stop_reason,
                'best_plan': deepcopy(search.incumbent), 'initial_certificate': deepcopy(search.initial_certificate),
                'initial_elapsed_seconds': checkpoints[0]['elapsed_seconds'],
                'wait_baseline_quality_pct': checkpoints[0]['quality_pct'], 'checkpoints': checkpoints,
                'completed_expansions': search._beam_turn, 'search_decisions': search._beam_turn,
                'counts': {'generated_candidates': len(search._branch_keys), 'graph_assessed_candidates': len(search._assessed),
                    'checked_candidates': len(search._checked), 'branch_expansions': search._beam_turn,
                    'generation_cap_rejections': search.generation_cap_rejections}}
        result.update(spec=spec, oracle=read(cell / 'oracle_summary.json'), elapsed_seconds=time.perf_counter() - started,
            training_interactions=0, optimizer_updates=0, new_training=0,
            formal_holdout_numerical_access=False, tune_numerical_access=False)
        save(cell / 'result.json', result)
        seal(cell)
        return 0
    except BaseException as exc:
        save(cell / 'error.json', {'status': 'STOPPED_RELIABILITY_OR_IMPLEMENTATION_ERROR', 'spec': spec,
            'error': repr(exc), 'traceback': traceback.format_exc(), 'elapsed_seconds': time.perf_counter() - started})
        return 1


def costs(run, dispatched):
    points = entries = 0
    unknown = []
    for spec in dispatched:
        try:
            summary = read(run / spec['cell'] / 'oracle_summary.json')
            require(type(summary['points']) is int and type(summary['actual_numerical_entries']) is int,
                    'Incomplete physical accounting')
            points += summary['points']; entries += summary['actual_numerical_entries']
        except (OSError, ValueError, KeyError, TypeError):
            unknown.append(spec['cell'])
    return {'known_physical_points': points, 'known_numerical_entries': entries, 'unknown_cost_cells': unknown}


def execute(run):
    require(not (run / 'execution_started.json').exists(), 'Frozen pilot already started; no repeat')
    verify_files(run, read(run / 'prephysical_freeze_manifest.json'))
    save(run / 'execution_started.json', {'utc': utc(), 'pid': os.getpid()})
    schedule = read(run / 'schedule.json')
    completed, dispatched, errors, exits, active = [], [], [], [], {}
    wall = time.perf_counter()

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
                spec = schedule[len(dispatched)]; cell = run / spec['cell']
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
                        errors.append({'cell': key, 'error': 'WATCHDOG_STOP_DISPATCH_DRAIN_STARTED_WORKERS'})
                    continue
                out.close(); err.close(); del active[key]
                finished(proc, spec, code)
            progress = {'status': 'RUNNING' if not errors else 'DRAINING_AFTER_ERROR', 'run': run.name,
                'completed': len(completed), 'scheduled': len(schedule), 'dispatched': len(dispatched),
                'active': list(active), 'errors': errors, 'utc': utc(), 'elapsed_seconds': time.perf_counter() - wall}
            save(run / 'controller_progress.json', progress); save(ACTIVE, progress)
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
    summary = {'status': 'COMPLETE_CELLS_PENDING_AUDIT' if not errors and not accounting['unknown_cost_cells']
               and len(completed) == len(schedule) else 'INCONCLUSIVE',
        'scheduled': len(schedule), 'completed': completed, 'dispatched': dispatched, 'exits': exits,
        'errors': errors, **accounting, 'wall_seconds': time.perf_counter() - wall, 'utc': utc()}
    save(run / 'controller_completion.json', summary)
    save(ACTIVE, {'status': summary['status'], 'run': run.name, 'completed': len(completed),
                  'scheduled': len(schedule), **accounting, 'errors': errors, 'utc': utc()})
    return 0 if summary['status'] == 'COMPLETE_CELLS_PENDING_AUDIT' else 1


def finalize(run):
    report = read(run / 'derived_report/summary.json')
    audit = read(run / 'audit/summary.json')
    require(audit['status'] == 'PASS' and audit['audited_cells'] == 16, 'Independent audit incomplete')
    require((run / 'derived_report/completion_manifest.json').exists(), 'Report not sealed')
    verify_files(run / 'derived_report', read(run / 'derived_report/completion_manifest.json'))
    terminal = {'status': 'COMPLETE_WINDOW_PILOT_AUDITED_AND_REPORTED', 'run': run.name, 'utc': utc(),
        'audit_sha256': sha(run / 'audit/summary.json'), 'report_sha256': sha(run / 'derived_report/summary.json'),
        'report_manifest_sha256': sha(run / 'derived_report/completion_manifest.json'),
        'physical_points': audit['physical_points'], 'numerical_entries': audit['numerical_entries'],
        'new_training': 0, 'all_valid_failures_preserved': True}
    save(run / 'terminal_status.json', terminal); seal(run)
    save(ACTIVE, {**terminal, 'status': 'COMPLETE_WINDOW_PILOT_SEALED',
                  'completion_sha256': sha(run / 'completion_manifest.json')})
    return terminal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'worker', 'execute', 'finalize'])
    parser.add_argument('--run', type=Path)
    parser.add_argument('--cell', type=Path)
    parser.add_argument('--population', type=Path)
    parser.add_argument('--tests', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(prepare(args.population.resolve(), args.tests.resolve())); return 0
    run = args.run.resolve()
    require(run.parent == ROOT / 'runs', 'Run outside project')
    if args.command == 'worker':
        return worker(run, args.cell.resolve())
    if args.command == 'finalize':
        print(finalize(run)); return 0
    return execute(run)


if __name__ == '__main__':
    raise SystemExit(main())
