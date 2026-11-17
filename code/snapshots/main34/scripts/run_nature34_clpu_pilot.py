"""Frozen, cold-process CLPU mechanism pilot. No training entry point."""
from __future__ import annotations

import argparse
from collections import Counter
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

from nature34_experiment_support import read, save, sha, append, numeric_lists, freeze, verify_prepared
from prepare_nature34_clpu_pilot import prepare
from dnr.nature34_clpu_environment import CLPUEnvironment, CLPUBudgetExhausted, CLPUReliabilityError
from dnr.nature34_joint_sequence import JointGraph


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def guard_author(run):
    provenance = read(run / 'source_provenance.json')
    if read(ROOT / 'artifacts/restoration_shortlist_sources_latest.json') != read(run / 'source_pointer.json'):
        raise RuntimeError('Author source pointer changed')
    for name, digest in provenance['author_model_input_sha256'].items():
        if sha(ROOT / provenance['author_model_directory'] / name) != digest:
            raise RuntimeError('Author model input changed: ' + name)


class SnapshotOracle:
    def __init__(self, parent, cell, env, budget):
        self.parent, self.cell, self.env, self.budget = parent, cell, env, budget
        self.cache, self.rows = {}, {}
        self.events, self.context = [], {'phase': 'initialization'}
        self.requests = self.hits = self.points = 0
        self.statuses = Counter()
        self.point_seconds = []

    def __enter__(self):
        from dnr.durable_dss_counter import make_durable_engine_counter
        from dnr import nature34_physical
        from dnr.nature34_clpu_physical import Nature34CLPUPhysicalRuntime
        guard_author(self.parent)
        self.factory = make_durable_engine_counter(self.cell / 'command_ledger.jsonl',
            expected_compile_inventory=read(self.parent / 'command_inventory.json'),
            context_provider=lambda: dict(self.context))
        nature34_physical.EngineCounter = self.factory
        try:
            self.runtime = Nature34CLPUPhysicalRuntime(ROOT, self.cell,
                read(self.parent / 'physical_config.json'), self.events, counter_factory=self.factory)
            stats = self.factory.stats()
            if stats['returned']['numerical_solution_entries'] != 3 or stats['requested'] != stats['returned']:
                raise RuntimeError('Initialization counter differs from original runtime')
            if (self.runtime.initial_metadata != read(self.parent / 'source_inputs/metadata.json') or
                    self.runtime.base_load_properties != read(self.parent / 'source_inputs/base_load_properties.json')):
                raise RuntimeError('Physical initialization differs from the sealed source')
            save(self.cell / 'metadata.json', self.runtime.initial_metadata)
            save(self.cell / 'adaptation.json', self.runtime.metadata)
            self.raw = gzip.open(self.cell / 'points.jsonl.gz', 'wt', encoding='utf-8', compresslevel=3)
            self.request_log = (self.cell / 'requests.jsonl').open('w', encoding='utf-8')
            return self
        except BaseException:
            save(self.cell / 'command_stats.json', self.factory.stats())
            self.factory.close()
            raise

    def certify(self, state, seconds):
        identity = {**self.env.identity(state, seconds), 'alpha': .5,
                    'base_condition_sha256': sha(self.parent / 'condition.json')}
        fingerprint = hashlib.sha256(canonical(identity).encode()).hexdigest()
        self.requests += 1
        if fingerprint in self.cache:
            self.hits += 1
            result = self.cache[fingerprint]
            append(self.request_log, {'request': self.requests, 'key': fingerprint,
                                     'cache_hit': True, 'point_id': result['point_id']})
            return result
        if self.points >= self.budget:
            append(self.request_log, {'request': self.requests, 'key': fingerprint,
                                     'status': 'BUDGET_UNKNOWN', 'cache_hit': False})
            raise CLPUBudgetExhausted('Frozen physical point budget exhausted')
        pid = self.points
        self.context.clear()
        self.context.update(phase='snapshot', point_id=pid, time_seconds=seconds, identity_sha256=fingerprint)
        append(self.request_log, {'request': self.requests, 'key': fingerprint, 'cache_hit': False,
                                 'point_id': pid, 'status': 'STARTED'})
        self.events.clear()
        started = time.perf_counter()
        row = numeric_lists(self.runtime.evaluate_point(self.env.graph.action(state.command),
            {'name': self.env.scenario['scenario_id'], 'fault_lines': self.env.scenario['fault_lines']},
            .5, 0, self.env.condition, self.env.multipliers(state, seconds),
            time_minutes=seconds / 60., state_identity=identity))
        row.update(point_id=pid, identity_sha256=fingerprint,
                   elapsed_seconds=time.perf_counter() - started, dynamic_state=state.to_dict())
        append(self.raw, row)
        self.points += 1
        self.point_seconds.append(row['elapsed_seconds'])
        self.statuses[row['status']] += 1
        if row['actual_call_counts']['numerical_solution_entries'] != 4:
            raise CLPUReliabilityError('Per-snapshot numerical entry counter differs')
        result = {'point_id': pid, 'status': row['status'], 'reliable': row['reliable'],
                  'feedback': row['structured_feedback']['violations']}
        append(self.request_log, {'request': self.requests, 'key': fingerprint, 'cache_hit': False,
                                 'point_id': pid, 'status': row['status']})
        if row['reliable'] is not True:
            raise CLPUReliabilityError('Unreliable numerical/AC/isolation/constraint evidence; row retained')
        self.cache[fingerprint] = result
        return result

    def __exit__(self, typ, value, tb):
        self.raw.close()
        self.request_log.close()
        stats = self.factory.stats()
        save(self.cell / 'command_stats.json', stats)
        self.factory.close()
        save(self.cell / 'oracle_summary.json', {'points': self.points, 'point_budget': self.budget,
             'requests': self.requests, 'cache_hits': self.hits, 'statuses': dict(self.statuses),
             'point_seconds': self.point_seconds, 'initialization_numerical_entries': 3,
             'actual_numerical_entries': stats['returned']['numerical_solution_entries'],
             'maximum_numerical_entries': 3 + 4 * self.budget,
             'status': 'COMPLETE_COUNTERS_PENDING_AUDIT' if typ is None else 'STOPPED',
             'training_interactions': 0})
        if typ is None and (stats['requested'] != stats['returned'] or stats['open_call_ids']
                or stats['calls_failed'] or stats['calls_rejected']
                or stats['returned']['numerical_solution_entries'] != 3 + 4 * self.points):
            raise CLPUReliabilityError('Unclosed physical call ledger or count mismatch')


def make_env(run, spec):
    scenarios = {r['scenario_id']: r for r in read(run / 'selected_scenarios.json')}
    events = {r['event_id']: r for r in read(run / 'selected_events.json')}
    static, loads = read(run / 'author_static_metadata.json'), read(run / 'load_reference.json')
    if len(loads) != 68:
        raise RuntimeError('Physical pilot requires exactly 68 loads')
    scenario, event, config = scenarios[spec['scenario_id']], events[spec['event_id']], read(run / 'config.json')
    if scenario['partition'] != 'build':
        raise RuntimeError('Non-training fault is forbidden')
    graph = JointGraph(static, loads, scenario)
    return CLPUEnvironment(graph, static, loads, scenario, read(run / 'condition.json'), event,
        horizon=config['horizon'], slot_seconds=config['slot_seconds'], sample_seconds=config['sample_seconds'])


def run_worker(run, cell):
    run, cell = run.resolve(), cell.resolve()
    spec, config = read(cell / 'spec.json'), read(run / 'config.json')
    tic = time.perf_counter()
    try:
        # Each worker is a fresh process: one native DSS runtime, no shared answers.
        for name in read(run / 'prepared.json')['source_files']:
            if sha(ROOT / name) != sha(run / 'source' / name):
                raise RuntimeError('Frozen implementation changed: ' + name)
        env = make_env(run, spec)
        save(cell / 'initial_state.json', env.initial.to_dict())
        with SnapshotOracle(run, cell, env, spec['point_budget']) as oracle:
            if config['phase'] == 'calibration':
                first = env.optimistic_plan(env.initial, depth=6)[0]
                waited = env.advance(env.initial, 0)
                advanced = env.advance(env.initial, first)
                samples = [(env.initial, 0), (waited, 30), (advanced, 0), (advanced, 30)]
                records = [{'state': state.to_dict(), 'time_seconds': t, **oracle.certify(state, t)}
                           for state, t in samples]
                result = {'status': 'CALIBRATION_COMPLETE', 'samples': records,
                          'first_nominal_action': first, 'training_interactions': 0}
            else:
                from dnr.nature34_clpu_search import run_search, SearchConfig
                initial = oracle.certify(env.initial, 0)
                if initial['status'] == 'FAIL':
                    result = {'status': 'INITIAL_FAIL', 'initial_certificate': initial,
                              'best_plan': None, 'training_interactions': 0}
                else:
                    with gzip.open(cell / 'plans.jsonl.gz', 'wt', encoding='utf-8', compresslevel=3) as plans:
                        result = run_search(env, oracle, spec['method'],
                            SearchConfig(**config['search']),
                            record_callback=lambda record: append(plans, record))
                    result['initial_certificate'] = initial
        result.update(spec=spec, elapsed_seconds=time.perf_counter() - tic,
                      oracle=read(cell / 'oracle_summary.json'), training_interactions=0,
                      formal_holdout_numerical_access=False, tune_numerical_access=False)
        save(cell / 'result.json', result)
        save(cell / 'completion_manifest.json', {str(p.relative_to(cell)): sha(p) for p in cell.rglob('*')
             if p.is_file() and p.name not in ('completion_manifest.json', 'stdout.log', 'stderr.log')})
        return 0
    except BaseException as exc:
        save(cell / 'error.json', {'status': 'STOPPED_RELIABILITY_OR_IMPLEMENTATION_ERROR',
             'error': repr(exc), 'traceback': traceback.format_exc(),
             'elapsed_seconds': time.perf_counter() - tic, 'spec': spec})
        return 1


def source_names():
    paths = list((ROOT / 'src/dnr').glob('*.py'))
    paths += [ROOT / 'scripts' / name for name in (
        'run_nature34_clpu_pilot.py', 'prepare_nature34_clpu_pilot.py',
        'nature34_experiment_support.py', 'run_nature34_development_training.py',
        'report_nature34_imitation.py', 'audit_nature34_clpu_pilot.py',
        'report_nature34_clpu_pilot.py')]
    paths += list((ROOT / 'tests').glob('test*clpu*.py'))
    return [str(p.relative_to(ROOT)) for p in paths if p.is_file()]


def prepare_run(config_path):
    config = read(config_path)
    run = prepare(config, source_names=source_names())
    scenarios, events = read(run / 'selected_scenarios.json'), read(run / 'selected_events.json')
    specs = []
    if config['phase'] == 'calibration':
        specs = [{'scenario_id': scenarios[0]['scenario_id'], 'event_id': events[0]['event_id'],
                  'method': 'calibration', 'point_budget': 4}]
    else:
        for scenario in scenarios:
            for event in events:
                for budget in config['point_budgets']:
                    for method in config['methods']:
                        specs.append({'scenario_id': scenario['scenario_id'], 'event_id': event['event_id'],
                                      'method': method, 'point_budget': budget})
                specs.append({'scenario_id': scenario['scenario_id'], 'event_id': event['event_id'],
                              'method': 'reference', 'point_budget': config['reference_point_budget']})
        # Fixed hash ordering avoids confounding methods with wall-clock order.
        specs.sort(key=lambda row: hashlib.sha256(('clpu-run-order-v1|' + canonical(row)).encode()).hexdigest())
    for index, spec in enumerate(specs):
        spec['episode_id'] = index
        cell = run / 'episodes' / f'{index:04d}'
        cell.mkdir(parents=True)
        save(cell / 'spec.json', spec)
    save(run / 'schedule.json', specs)
    save(run / 'resource_budget.json', {'episodes': len(specs),
         'max_physical_points': sum(s['point_budget'] for s in specs),
         'max_numerical_entries': sum(3 + 4 * s['point_budget'] for s in specs),
         'max_workers': config.get('max_workers', 1), 'training_interactions': 0})
    save(run / 'machine.json', {'cpu_count': os.cpu_count(), 'platform': sys.platform,
         'free_disk_bytes_before_run': shutil.disk_usage(ROOT).free})
    evidence = ROOT / 'runs/test_only_clpu_integration_fixed_v1/pytest.log'
    if evidence.exists():
        shutil.copyfile(evidence, run / 'preparation_tests.log')
    names = source_names()
    if (ROOT / 'docs/nature34_clpu_pilot_protocol_v1.md').exists():
        names.append('docs/nature34_clpu_pilot_protocol_v1.md')
    freeze(run, names)
    return run


def execute(run):
    run = run.resolve()
    verify_prepared(run)
    guard_author(run)
    config, schedule = read(run / 'config.json'), read(run / 'schedule.json')
    save(run / 'execution_started.json', {'utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid()})
    active, finished, next_index, stopped = {}, [], 0, False
    started = time.perf_counter()
    try:
        while active or (next_index < len(schedule) and not stopped):
            while not stopped and next_index < len(schedule) and len(active) < config.get('max_workers', 1):
                if shutil.disk_usage(ROOT).free < config.get('minimum_free_disk_bytes', 15_000_000_000):
                    stopped = True
                    save(run / 'controller_error.json', {'reason': 'DISK_FLOOR'})
                    break
                spec = schedule[next_index]
                cell = run / 'episodes' / f"{spec['episode_id']:04d}"
                out, err = (cell / 'stdout.log').open('w'), (cell / 'stderr.log').open('w')
                process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
                    '--worker', str(run), str(cell)], cwd=str(ROOT), stdout=out, stderr=err,
                    env={**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
                active[process.pid] = (process, cell, out, err)
                next_index += 1
            for pid, (process, cell, out, err) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                out.close(); err.close()
                finished.append({'episode_id': read(cell / 'spec.json')['episode_id'], 'exit_code': code})
                del active[pid]
                if code != 0:
                    stopped = True  # Let in-flight processes close journals; dispatch no more.
            progress = {'status': 'STOPPING_ON_ERROR' if stopped else 'RUNNING',
                        'completed': len(finished), 'scheduled': len(schedule),
                        'running_episode_ids': [read(c / 'spec.json')['episode_id'] for _, c, _, _ in active.values()],
                        'elapsed_seconds': time.perf_counter() - started}
            save(run / 'progress.json', progress)
            save(ROOT / 'artifacts/nature34_clpu_active.json', {**progress, 'run': run.name,
                  'training_interactions': 0})
            if active:
                time.sleep(.5)
        summary = {'status': 'STOPPED' if stopped else 'COMPLETE_PENDING_INDEPENDENT_AUDIT',
                   'completed': len(finished), 'scheduled': len(schedule), 'workers': finished,
                   'elapsed_seconds': time.perf_counter() - started, 'training_interactions': 0,
                   'formal_holdout_numerical_access': False, 'tune_numerical_access': False}
        save(run / 'summary.json', summary)
        save(run / 'completion_manifest.json', {str(p.relative_to(run)): sha(p) for p in run.rglob('*')
             if p.is_file() and p != run / 'completion_manifest.json'})
        save(ROOT / 'artifacts/nature34_clpu_active.json', {**summary, 'run': run.name})
        print(canonical({'run': str(run), **summary}), flush=True)
        return 1 if stopped else 0
    except BaseException as exc:
        save(run / 'controller_error.json', {'error': repr(exc), 'traceback': traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', type=Path)
    parser.add_argument('--execute', type=Path)
    parser.add_argument('--worker', nargs=2, type=Path)
    args = parser.parse_args()
    if args.worker:
        return run_worker(*args.worker)
    if args.prepare:
        run = prepare_run(args.prepare.resolve())
        print(canonical({'run': str(run), 'status': 'FROZEN_NO_PHYSICAL_EXECUTION'}))
        return 0
    if args.execute:
        return execute(args.execute)
    parser.error('Choose prepare, execute or worker')


if __name__ == '__main__':
    raise SystemExit(main())
