"""D037 durable experiment utilities; importing does not initialize DSS."""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import gzip
import importlib.metadata
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
sys.dont_write_bytecode = True
from run_nature34_development_training import (read, save, sha, append, numeric_lists,
    verify_manifest, require_reliable_point, select_reward, INPUT_KEYS, CRITICAL_PHYSICAL_FILES)
from dnr.nature34_resume_helpers import require, assert_point_delta

ORIGINAL = '20261019T052118_527814Z_nature34_development_training'
BASELINE = '20261019T081517_951435Z_nature34_baseline_probe'
BASELINE_SHA = '3bd306ba4789afc8a2519b2368a2ce33a546a4f7542012c78df7b1a2b5a0beaa'
BASELINE_AUDIT = '20261019T085152_371760Z_nature34_baseline_probe_saved_audit'
BASELINE_AUDIT_SHA = '59b5f76819c0fa72c0fcd208aa9f116bee58d0c0c6f53a55467d66a0d10adb93'


def dependencies():
    return {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}


def create_run(suffix, config):
    run = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_' + suffix)
    run.mkdir()
    save(run / 'config.json', config)
    save(run / 'dependencies.json', dependencies())
    save(run / 'python_environment.json', {'executable': sys.executable, 'version': sys.version, 'sha256': sha(Path(sys.executable))})
    return run


def freeze(run, source_names):
    for name in sorted(set(source_names)):
        target = run / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    save(run / 'prephysical_freeze_manifest.json', {str(p.relative_to(run)): sha(p) for p in run.rglob('*') if p.is_file()})
    prepared = {'status': 'FROZEN_PREPARED_NO_DSS', 'run': run.name,
        'freeze_sha256': sha(run / 'prephysical_freeze_manifest.json'), 'source_files': sorted(set(source_names))}
    save(run / 'prepared.json', prepared)
    return prepared


def verify_prepared(run):
    require(run.is_absolute(), 'Execution paths must be absolute before DSS can change cwd')
    require(run.resolve().parent == (ROOT / 'runs').resolve(), 'Run outside project runs')
    require(not (run / 'execution_started.json').exists(), 'Already started; no automatic retry')
    prepared = read(run / 'prepared.json')
    require(sha(run / 'prephysical_freeze_manifest.json') == prepared['freeze_sha256'], 'Freeze hash differs')
    for name, digest in read(run / 'prephysical_freeze_manifest.json').items():
        require(sha(run / name) == digest, 'Frozen input changed: ' + name)
    for name in prepared['source_files']:
        require(sha(ROOT / name) == sha(run / 'source' / name), 'Executing source changed: ' + name)
    require(dependencies() == read(run / 'dependencies.json'), 'Dependencies changed')


def common_physical_inputs(run):
    original = ROOT / 'runs' / ORIGINAL
    baseline = ROOT / 'runs' / BASELINE
    require(dependencies() == read(original / 'dependencies.json'), 'Original dependencies differ')
    for name in CRITICAL_PHYSICAL_FILES:
        require(sha(ROOT / name) == sha(original / 'source' / name), 'Physical implementation changed: ' + name)
    require(read(original / 'physical_config.json') == read(ROOT / 'configs/nature34_migration_v1.json'), 'Physical parameters changed')
    for name in ('physical_config.json', 'fault_partitions.json', 'author_static_metadata.json', 'source_pointer.json'):
        shutil.copyfile(original / name, run / name)
    shutil.copyfile(baseline / 'command_inventory.json', run / 'command_inventory.json')
    shutil.copyfile(baseline / 'load_reference.json', run / 'load_reference.json')


def seal(run, summary, pointer_name):
    require(run.is_absolute(), 'Sealing requires the original absolute run path')
    save(run / 'summary.json', summary)
    save(run / 'completion_manifest.json', {str(p.relative_to(run)): sha(p) for p in run.rglob('*')
        if p.is_file() and p.name != 'completion_manifest.json'})
    pointer = {**summary, 'run': run.name, 'completion_sha256': sha(run / 'completion_manifest.json')}
    save(ROOT / 'artifacts' / pointer_name, pointer)
    return pointer


class PhysicalOracle:
    """One runtime, durable per-entry journal, every alpha paid and retained."""
    def __init__(self, run, *, max_actions, max_points, partition, phase):
        self.run, self.partition, self.phase = run.resolve(), partition, phase
        self.max_actions, self.max_points = max_actions, max_points
        self.max_entries = 3 + 4 * max_points
        self.context = {'phase': 'initialization'}
        self.events, self.factory, self.runtime = [], None, None
        self.files = []
        self.summary = {'run': run.name, 'status': 'RUNNING', 'phase': phase, 'process_id': os.getpid(),
            'action_attempts': 0, 'completed_actions': 0, 'point_attempts': 0, 'point_evaluations': 0,
            'physical_cache_hits': 0, 'tune_numerical_access': partition == 'tune',
            'formal_holdout_numerical_access': False, 'full_hardware_feasibility': 'NOT_ESTABLISHED'}
        self.counts = {name: Counter() for name in ('converged', 'known_constraints', 'ac', 'isolation')}
        self.started = time.perf_counter()

    def __enter__(self):
        from dnr.durable_dss_counter import make_durable_engine_counter
        from dnr import nature34_physical
        from dnr.restoration_ac_audit import audit_ac_snapshot
        from dnr.restoration_constraints import check_known_constraints
        from dnr.restoration_isolation import verify_isolation
        self.certifiers = check_known_constraints, audit_ac_snapshot, verify_isolation
        self.physical = read(self.run / 'physical_config.json')
        try:
            self.factory = make_durable_engine_counter(self.run / 'command_ledger.jsonl',
                expected_compile_inventory=read(self.run / 'command_inventory.json'), context_provider=lambda: dict(self.context))
            nature34_physical.EngineCounter = self.factory
            self.runtime = nature34_physical.Nature34PhysicalRuntime(ROOT, self.run, self.physical, self.events)
            assert_point_delta({}, self.factory.stats()['returned'], initialization=True)
            save(self.run / 'metadata.json', self.runtime.initial_metadata)
            save(self.run / 'adaptation.json', self.runtime.metadata)
            self.raw = gzip.open(self.run / 'points.jsonl.gz', 'wt', encoding='utf-8', compresslevel=6)
            self.files.append(self.raw)
            self.actions = (self.run / 'actions.jsonl').open('w', encoding='utf-8'); self.files.append(self.actions)
            self.attempts = (self.run / 'attempts.jsonl').open('w', encoding='utf-8'); self.files.append(self.attempts)
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def evaluate(self, action, scenario, identity):
        require(scenario['partition'] == self.partition and self.partition in ('build', 'tune'), 'Partition boundary')
        require(self.summary['action_attempts'] < self.max_actions, 'Action budget exhausted')
        identity = {**identity, 'action_id': self.summary['action_attempts'], 'scenario_id': scenario['scenario_id'],
            'partition': self.partition, 'fault_lines': scenario['fault_lines'], 'action': action}
        append(self.attempts, {**identity, 'event': 'action_started'})
        self.summary['action_attempts'] += 1
        compact = []
        check_known, audit_ac, verify_isolation = self.certifiers
        for alpha in (0., .25, .5, .75, 1.):
            pid = self.summary['point_attempts']
            require(pid < self.max_points and self.factory.stats()['requested']['numerical_solution_entries'] + 4 <= self.max_entries, 'Point/entry budget')
            point_identity = {**identity, 'point_id': pid, 'alpha': alpha, 'algorithm': 0}
            self.context.clear(); self.context.update({'phase': 'point', 'point_id': pid, 'action_id': identity['action_id']})
            before = self.factory.stats()['returned']
            self.events.clear()
            append(self.attempts, {**point_identity, 'event': 'point_started'})
            self.summary['point_attempts'] += 1
            tic, row = time.perf_counter(), None
            try:
                row = numeric_lists(self.runtime.evaluate_point(action, {'name': scenario['scenario_id'], 'fault_lines': scenario['fault_lines']}, alpha, 0))
                metadata = {**self.runtime.initial_metadata, **row['metadata_override']}
                row['known_constraints'] = check_known(row['raw_outputs'], metadata, **self.physical['constraint_tolerances'])
                row['ac_audit'] = audit_ac(row['raw_outputs'], metadata, **self.physical['ac_audit_tolerances'])
                row['isolation'] = verify_isolation(row['raw_outputs'], metadata)
            except BaseException as exc:
                if row is not None: append(self.raw, {**row, **point_identity, 'certification_exception': repr(exc)})
                append(self.attempts, {**point_identity, 'event': 'point_failed', 'error': repr(exc), 'traceback': traceback.format_exc()})
                raise
            row.update(point_identity, elapsed_seconds=time.perf_counter() - tic)
            append(self.raw, row)
            self.summary['point_evaluations'] += 1
            entry = {'point_id': pid, 'alpha': alpha, 'converged': bool(row['raw_outputs']['converged']),
                'known_constraints_satisfied': row['known_constraints']['known_constraints_satisfied'],
                'ac_status': row['ac_audit']['status'], 'isolation_status': row['isolation']['status'],
                'nominal_restored_fraction': row['nominal_load_ledger']['nominal_restored_fraction']}
            compact.append(entry)
            append(self.attempts, {**point_identity, **entry, 'event': 'point_completed'})
            assert_point_delta(before, self.factory.stats()['returned'])
            require_reliable_point(entry['converged'], row['isolation'], row['ac_audit'])
            for name, field in (('converged', 'converged'), ('known_constraints', 'known_constraints_satisfied'), ('ac', 'ac_status'), ('isolation', 'isolation_status')):
                self.counts[name][str(entry[field])] += 1
        result = select_reward(compact)
        record = {**identity, **result, 'points': compact}
        append(self.actions, record)
        append(self.attempts, {**identity, **result, 'event': 'action_completed'})
        self.summary['completed_actions'] += 1
        return record

    def progress(self, extra=None):
        result = {**self.summary, **(extra or {}), 'command_counts': self.factory.stats()['returned']}
        # Unique append-only snapshots avoid overwriting a file being read.
        save(self.run / 'progress' / f"{self.summary['completed_actions']:06d}.json", result)
        return result

    def __exit__(self, typ, exc, tb):
        try:
            if exc is None:
                require(self.summary['completed_actions'] == self.max_actions and self.summary['point_evaluations'] == self.max_points, 'Incomplete fixed budget')
                stats = self.factory.stats()
                require(stats['requested'] == stats['returned'] and not stats['open_call_ids'] and stats['calls_failed'] == stats['calls_rejected'] == 0, 'Unclosed physical calls')
                require(stats['returned']['numerical_solution_entries'] == self.max_entries, 'Final physical budget mismatch')
                self.summary['status'] = 'COMPLETE_PENDING_SAVED_AUDIT'
            else:
                self.summary.update(status='STOPPED_RELIABILITY_OR_EXECUTION_FAILURE', error=repr(exc), traceback=''.join(traceback.format_exception(typ, exc, tb)))
        except BaseException as final_error:
            self.summary.update(status='STOPPED_RELIABILITY_OR_EXECUTION_FAILURE', error=repr(final_error), traceback=traceback.format_exc())
            raise
        finally:
            for stream in self.files: stream.close()
            if self.factory:
                save(self.run / 'command_stats.json', self.factory.stats())
                self.factory.close()
            self.summary.update(elapsed_seconds=time.perf_counter() - self.started, point_counts={k: dict(v) for k, v in self.counts.items()})
            save(self.run / 'physical_summary.json', self.summary)
