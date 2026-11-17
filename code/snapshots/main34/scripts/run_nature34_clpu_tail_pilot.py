"""Prepare and run the approved 40-cell, no-training tail-reliability pilot.

All old science files remain immutable. Each treatment pays for its own Beam
prefix. Evaluation, independent audit and descriptive reporting are separate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
sys.dont_write_bytecode = True

from d060_validation_common import (SOURCE_RUN, SOURCE_SHA, SUPPLEMENT_RUN,
    SUPPLEMENT_SHA, require, read, sha, canonical, utc_now, inside, normalize_manifest)
from run_d060_validation import (source_evidence, ROOT_INPUTS, bound_copy,
    verify_frozen, journal, seal_directory, directory_bytes)
from nature34_experiment_support import dependencies, freeze

STUDY = 'PPO_TAIL_PILOT_V1'
SEEDS = (17, 41, 73)
REGRESSION = {'scenario_id': 'N2__line.l13__line.l24', 'event_id': 12,
              'cohort': 'selected_development_regression'}
ACTIVE = ROOT / 'artifacts/ppo_tail_pilot_active.json'
EXTRA_SOURCES = (
    'src/dnr/nature34_clpu_tail.py', 'scripts/nature34_clpu_tail_execution.py',
    'scripts/audit_nature34_clpu_tail.py', 'scripts/run_nature34_clpu_tail_pilot.py',
    'tests/test_nature34_clpu_tail.py', 'tests/test_nature34_clpu_tail_execution.py',
    'tests/test_nature34_clpu_tail_audit.py', 'tests/test_run_nature34_clpu_tail_pilot.py',
    'scripts/run_d060_validation.py', 'scripts/d060_validation_common.py',
    'scripts/audit_d060_validation.py', 'scripts/audit_d059_supplement.py',
    'scripts/report_nature34_clpu_tail_pilot.py', 'tests/test_report_nature34_clpu_tail_pilot.py',
)


def save(path, value):
    """Atomic control updates with bounded Windows-reader collision retries."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.{time.time_ns()}.tmp')
    with tmp.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    for attempt in range(6):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.05 * (attempt + 1))


def select_population(training_contexts, training_events, training_schedule):
    actual = {r['scenario_id'] for r in training_schedule if r['phase'] == 'train'}
    cohort = {r['scenario_id'] for r in training_contexts}
    require(actual == cohort and len(actual) == 8, 'Actual training fault cohort differs')
    event_ids = {r['event_id'] for r in training_events}
    require(event_ids == {20, 3, 6, 30}, 'Original training event cohort differs')
    require({(r['scenario_id'], r['event_id']) for r in training_contexts} ==
            {(s, e) for s in actual for e in event_ids}, 'Training cross is incomplete')
    faults = sorted(actual, key=lambda s: hashlib.sha256(
        ('ppo-tail-pilot-v1|fault|' + s).encode()).hexdigest())[:2]
    events = sorted(event_ids, key=lambda e: hashlib.sha256(
        ('ppo-tail-pilot-v1|event|' + str(e)).encode()).hexdigest())[:2]
    selected = [{'scenario_id': s, 'event_id': e, 'cohort': 'training_mechanism'}
                for s in faults for e in events] + [dict(REGRESSION)]
    require(len({(r['scenario_id'], r['event_id']) for r in selected}) == 5,
            'Regression overlaps selected training population')
    return selected


def build_schedule(contexts):
    require(len(contexts) == 5 and sum(r['cohort'] == 'training_mechanism' for r in contexts) == 4
            and REGRESSION in contexts, 'Pilot population differs')
    rows = []
    for context in contexts:
        require(set(context) == {'scenario_id', 'event_id', 'cohort'}, 'Undeclared cohort fields')
        for group, method, seeds in [('G0', 'beam', (0,)), ('G1', 'beam', (0,)),
                                     ('G2', 'bc_raw', SEEDS), ('G3', 'ppo_raw', SEEDS)]:
            for seed in seeds:
                rows.append({**context, 'tail_group': group, 'method': method, 'seed': seed,
                    'phase': 'evaluation', 'group': 'evaluation', 'point_budget': 256,
                    'representation': 'raw', 'deterministic': True})
    rows.sort(key=lambda r: hashlib.sha256(('ppo-tail-order-v1|' + canonical(r)).encode()).hexdigest())
    for i, row in enumerate(rows):
        row.update(episode_id=i, cell=f'episodes/{i:04d}')
    require(len(rows) == 40 and sum(r['point_budget'] for r in rows) == 10240, 'Task budget differs')
    return rows


def software_evidence(path):
    cases = list(ET.parse(path).getroot().iter('testcase'))
    require(cases and not any(list(c.iter('failure')) + list(c.iter('error')) + list(c.iter('skipped'))
                             for c in cases), 'Software validation failed, skipped or empty')
    for name in ('test_nature34_clpu_tail', 'test_nature34_clpu_tail_execution',
                 'test_nature34_clpu_tail_audit', 'test_run_nature34_clpu_tail_pilot',
                 'test_report_nature34_clpu_tail_pilot'):
        require(any(name in c.get('classname', '').split('.') for c in cases), 'Missing tests: ' + name)
    return len(cases)


def resource_projection(source, sm):
    require(sha(source / 'schedule.json') == sm['schedule.json'], 'Unbound historical size population')
    # Metadata only: old files may be NTFS-compressed; logical size is conservative
    # for new uncompressed logs. Take the maximum observed Beam/BC/PPO cell.
    historical = []
    for row in read(source / 'schedule.json'):
        if row['phase'] == 'evaluation' and row['method'] in ('beam', 'bc_raw', 'ppo_raw'):
            historical.append(directory_bytes(source / row['cell']))
    require(historical and max(historical) > 0, 'Historical storage evidence missing')
    per_cell = int(max(historical) * 2)  # up to twice expansion/trace plus baseline snapshot
    reserve = 12 * 2**30
    required = int(40 * per_cell * 1.25 + 2 * 2**30 + reserve)
    return {'observed_utc': utc_now(), 'historical_cells': len(historical),
        'historical_max_cell_logical_bytes': max(historical),
        'estimated_cell_logical_bytes': per_cell, 'estimate_is_not_an_upper_bound': True,
        'expansion_and_snapshot_multiplier': 2, 'forecast_margin': 1.25,
        'report_reserve_bytes': 2 * 2**30, 'minimum_free_bytes': reserve,
        'required_free_bytes': required, 'free_bytes': shutil.disk_usage(ROOT).free,
        'resource_policy': 'Stop new dispatch on insufficient space; drain writers; no deletion',
        'runtime_observation_rule': 'max(original per-cell forecast, largest completed new cell); subtract existing in-flight bytes',
        'max_workers': 2, 'audit_runs_after_evaluation': True}


def prepare(junit_path, authorization_path, protocol_path):
    test_count = software_evidence(junit_path)
    auth = read(authorization_path)
    require(auth.get('execution_authorized') is True and auth.get('new_training_authorized') is False,
            'Pilot authorization is missing or widened')
    source, supplement, sm, am, learning, bindings = source_evidence()
    consumed = {}
    def checked(name):
        require(name in sm and sha(source / name) == sm[name], 'Source population changed: ' + name)
        consumed[name] = sm[name]
        return read(source / name)
    tc, te, ts = checked('training_contexts.json'), checked('training_events.json'), checked('schedule.json')
    contexts = select_population(tc, te, ts)
    scenarios = checked('training_scenarios.json')
    events = checked('selected_events.json')
    d060 = ROOT / 'runs/20261025T123350_150996Z_d060_main'
    dm = normalize_manifest(read(d060 / 'completion_manifest.json'))
    require(sha(d060 / 'selected_events.json') == dm['selected_events.json'], 'D060 event source changed')
    event12 = [e for e in read(d060 / 'selected_events.json') if e['event_id'] == 12]
    require(len(event12) == 1, 'Regression event missing')
    events = [e for e in events if e['event_id'] in {r['event_id'] for r in contexts}] + event12
    faults = [s for s in scenarios if s['scenario_id'] in {r['scenario_id'] for r in contexts}]
    require(len(events) == 3 and len(faults) == 3, 'Selected input inventory differs')
    schedule = build_schedule(contexts)
    model_bindings = [b for b in bindings if b['method'] in ('bc_raw', 'ppo_raw')]
    require(len(model_bindings) == 6, 'Expected six immutable endpoints')
    # Explicitly bind these endpoints to the already used D060 copies as well.
    d060_bindings = read(d060 / 'model_bindings.json')
    require(sha(d060 / 'model_bindings.json') == dm['model_bindings.json'], 'D060 model bindings changed')
    for b in model_bindings:
        other = next(x for x in d060_bindings if x['method'] == b['method'] and x['seed'] == b['seed'])
        require(other['source_sha256'] == b['source_sha256'] == sha(d060 / other['checkpoint']),
                'D060 endpoint identity differs')
    require(dependencies() == read(source / 'dependencies.json') and
            sha(Path(sys.executable)) == read(source / 'python_environment.json')['sha256'], 'Runtime changed')
    inherited = {x.replace('\\', '/') for x in read(source / 'prepared.json')['source_files'] if x.endswith('.py')}
    for name in inherited:
        require(sha(ROOT / name) == sm['source/' + name], 'Inherited scientific source changed: ' + name)
    resources = resource_projection(source, sm)
    require(resources['free_bytes'] >= resources['required_free_bytes'], 'Insufficient projected storage')
    run = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_clpu_tail_pilot')
    run.mkdir(exist_ok=False)
    for name in ROOT_INPUTS:
        bound_copy(source, run / name, sm, name)
    for path in (source / 'source_inputs').rglob('*'):
        if path.is_file():
            name = path.relative_to(source).as_posix()
            bound_copy(source, run / name, sm, name)
    cfg = deepcopy(read(source / 'config.json'))
    cfg.update(validation_study=STUDY, pilot_version=1)
    cfg['tail_search'] = {'continuation_expansions': 256, 'enable_exact_failed_prefix': True,
                          'enable_tail_verification': True}
    cfg['resources'] = {'max_physical_points': 10240, 'max_workers': 2,
                        'worker_deadline_seconds': 3600, 'minimum_free_disk_bytes': 12 * 2**30,
                        'no_automatic_retry_or_budget_extension': True}
    save(run / 'config.json', cfg)
    save(run / 'selected_scenarios.json', faults); save(run / 'selected_events.json', events)
    save(run / 'evaluation_contexts.json', contexts); save(run / 'training_contexts.json', [])
    save(run / 'schedule.json', schedule); save(run / 'model_bindings.json', model_bindings)
    save(run / 'source_population.json', {'source_run': SOURCE_RUN, 'source_sha256': consumed,
        'training_contexts': tc, 'training_events': te,
        'actual_training_fault_ids': sorted({r['scenario_id'] for r in ts if r['phase'] == 'train'}),
        'selection_fault_domain': 'ppo-tail-pilot-v1|fault|',
        'selection_event_domain': 'ppo-tail-pilot-v1|event|', 'selected_contexts': contexts,
        'regression_event_source_run': d060.name, 'regression_event_source_sha256': dm['selected_events.json'],
        'd060_completion_sha256': sha(d060 / 'completion_manifest.json')})
    save(run / 'source_binding.json', {'source_run': SOURCE_RUN, 'source_completion_sha256': SOURCE_SHA,
        'supplement_run': SUPPLEMENT_RUN, 'supplement_completion_sha256': SUPPLEMENT_SHA,
        'd060_run': d060.name, 'd060_completion_sha256': sha(d060 / 'completion_manifest.json'),
        'old_failures_preserved': True, 'new_training_tasks': 0,
        'config_backend_decision_is_compatibility_only': True,
        'diagnosis_summary': 'artifacts/d060_worst_tail_diagnosis_20261027/summary.json',
        'diagnosis_summary_sha256': sha(ROOT / 'artifacts/d060_worst_tail_diagnosis_20261027/summary.json'),
        'diagnosis_inputs': read(ROOT / 'artifacts/d060_worst_tail_diagnosis_20261027/summary.json')['input_sha256']})
    for b in model_bindings:
        bound_copy(source, run / b['checkpoint'], sm, b['source_relative'])
    shutil.copyfile(authorization_path, run / 'authorization.json')
    shutil.copyfile(protocol_path, run / 'protocol.md')
    shutil.copyfile(junit_path, run / 'test_evidence.xml')
    save(run / 'resource_review.json', resources)
    save(run / 'resource_budget.json', {'episodes': 40, 'allocated_physical_points': 10240,
        'maximum_numerical_entries': 40 * 3 + 4 * 10240, 'max_workers': 2,
        'minimum_free_bytes': 12 * 2**30, 'worker_watchdog_seconds': 3600,
        'hard_total_deadline_seconds': None, 'new_training_tasks': 0,
        'new_optimizer_updates': 0, 'automatic_retry': False})
    save(run / 'dependencies.json', dependencies())
    save(run / 'python_environment.json', {'executable': sys.executable, 'version': sys.version,
                                          'sha256': sha(Path(sys.executable))})
    sources = sorted(inherited | set(EXTRA_SOURCES))
    freeze(run, sources)
    save(run / 'prepare_summary.json', {'status': 'FROZEN_TAIL_PILOT_NO_PHYSICS',
        'tests': test_count, 'source_files': len(sources), 'contexts': contexts, 'tasks': 40,
        'physical_budget': 10240, 'new_training_tasks': 0, 'utc': utc_now()})
    save(ACTIVE, {'status': 'FROZEN_NOT_STARTED', 'run': run.name, 'completed': 0,
                  'scheduled': 40, 'utc': utc_now()})
    return run


def make_spec(row, run, now=None):
    spec = dict(row, template_sha256=hashlib.sha256(canonical(row).encode()).hexdigest(),
                deadline_utc_epoch=(time.time() if now is None else now) + 3600)
    if row['method'] != 'beam':
        binding = next(b for b in read(run / 'model_bindings.json')
                       if b['method'] == row['method'] and b['seed'] == row['seed'])
        require(sha(run / binding['checkpoint']) == binding['source_sha256'], 'Endpoint changed')
        spec.update(checkpoint=binding['checkpoint'], checkpoint_sha256=binding['source_sha256'])
    return spec


def cell_cost(run, row):
    cell = run / row['cell']
    value = {'episode_id': row['episode_id'], 'cell': row['cell'], 'tail_group': row['tail_group'],
             'cohort': row['cohort'], 'allocated_points': 256,
             'physical_points': None, 'numerical_entries': None, 'counters_closed': False,
             'oracle_complete': False, 'cost_errors': []}
    if (cell / 'oracle_summary.json').exists():
        try:
            oracle = read(cell / 'oracle_summary.json')
            require(type(oracle['points']) is int and 0 <= oracle['points'] <= 256 and
                    oracle['point_budget'] == 256, 'Invalid physical point accounting')
            value.update(physical_points=oracle['points'], physical_statuses=oracle.get('statuses', {}),
                         oracle_complete=oracle.get('status') == 'COMPLETE_COUNTERS_PENDING_AUDIT')
        except Exception as exc:
            value['cost_errors'].append({'file': 'oracle_summary.json', 'error': repr(exc)})
    else:
        value['cost_errors'].append({'file': 'oracle_summary.json', 'error': 'MISSING_COST_EVIDENCE'})
    if (cell / 'command_stats.json').exists():
        try:
            stats = read(cell / 'command_stats.json')
            entries = stats['returned']['numerical_solution_entries']
            require(type(entries) is int and 0 <= entries <= 3 + 4 * 256,
                    'Invalid numerical entry accounting')
            value.update(numerical_entries=entries,
                         requested_numerical_entries=stats['requested']['numerical_solution_entries'],
                         counters_closed=(stats['returned'] == stats['requested']
                             and stats['open_call_ids'] == [] and stats['calls_failed'] == 0
                             and stats['calls_rejected'] == 0))
            if value['physical_points'] is not None:
                value['counters_closed'] = (value['counters_closed'] and
                    entries == 3 + 4 * value['physical_points'])
        except Exception as exc:
            value['cost_errors'].append({'file': 'command_stats.json', 'error': repr(exc)})
    else:
        value['cost_errors'].append({'file': 'command_stats.json', 'error': 'MISSING_COST_EVIDENCE'})
    value['worker_error_present'] = (cell / 'error.json').exists()
    value['completed'] = ((cell / 'completion_manifest.json').exists() and (cell / 'result.json').exists()
                          and not value['worker_error_present'])
    try:
        value['logical_bytes'] = directory_bytes(cell)
    except Exception as exc:
        value['logical_bytes'] = None
        value['cost_errors'].append({'field': 'logical_bytes', 'error': repr(exc)})
    value['cost_complete'] = (not value['cost_errors'] and value['physical_points'] is not None
                             and value['numerical_entries'] is not None and value['counters_closed']
                             and value['oracle_complete'])
    return value


def safe_cell_cost(run, row):
    """An unreadable record remains UNKNOWN, never a fabricated zero-cost task."""
    try:
        return cell_cost(run, row)
    except Exception as exc:
        return {'episode_id': row['episode_id'], 'cell': row['cell'], 'tail_group': row['tail_group'],
                'cohort': row['cohort'], 'allocated_points': 256, 'physical_points': None,
                'numerical_entries': None, 'counters_closed': False, 'cost_complete': False,
                'completed': False, 'logical_bytes': None,
                'cost_errors': [{'error': repr(exc), 'traceback': traceback.format_exc()}]}


def cost_totals(costs):
    costs = list(costs)
    unknown_points = [c['episode_id'] for c in costs if c.get('physical_points') is None]
    unknown_entries = [c['episode_id'] for c in costs if c.get('numerical_entries') is None]
    points = sum(c['physical_points'] for c in costs if c.get('physical_points') is not None)
    entries = sum(c['numerical_entries'] for c in costs if c.get('numerical_entries') is not None)
    return {'physical_points': None if unknown_points else points,
            'numerical_entries': None if unknown_entries else entries,
            'known_physical_points': points, 'known_numerical_entries': entries,
            'unknown_physical_point_cells': unknown_points, 'unknown_numerical_entry_cells': unknown_entries,
            'unknown_cost_cells': [c['episode_id'] for c in costs if not c.get('cost_complete')]}


def remaining_storage_projection(run, rows, completed, forecast, budget):
    observed = [c['logical_bytes'] for c in completed.values()
                if type(c.get('logical_bytes')) is int and c['logical_bytes'] >= 0]
    observed_max = max(observed, default=0)
    per_cell = max(forecast['estimated_cell_logical_bytes'], observed_max)
    remaining, pending, in_flight = 0, 0, 0
    for row in rows:
        if row['episode_id'] in completed:
            continue
        cell = run / row['cell']
        written = directory_bytes(cell) if cell.exists() else 0
        remaining += max(0, per_cell - written)
        if cell.exists():
            in_flight += 1
        else:
            pending += 1
    required = int(remaining * forecast.get('forecast_margin', 1.25)
                   + forecast['report_reserve_bytes'] + budget['minimum_free_bytes'])
    return {'estimated_remaining_bytes': remaining, 'estimated_cell_logical_bytes': per_cell,
            'largest_completed_new_cell_bytes': observed_max, 'pending': pending, 'in_flight': in_flight,
            'required_free_bytes': max(required, budget['minimum_free_bytes']),
            'free_bytes': shutil.disk_usage(ROOT).free, 'estimate_is_not_an_upper_bound': True}


def drain_workers(active, errors):
    """Wait for every writer before any cost reads or journal writes can fail."""
    exited = []
    for identifier, (proc, row, handles) in list(active.items()):
        try:
            rc = proc.wait()
        except BaseException as exc:
            errors.append({'episode_id': identifier, 'error': 'PROCESS_EXIT_NOT_CONFIRMED',
                           'detail': repr(exc), 'launcher_pid': proc.pid})
            continue
        # Removal is independent of disk I/O; a later failure cannot reap twice.
        del active[identifier]
        exited.append((identifier, proc, row, handles, rc))
    return exited


def run_evaluations(run):
    run = Path(run).resolve(); verify_frozen(run)
    require(not (run / 'execution_started.json').exists(), 'Pilot already started; no automatic rerun')
    rows = read(run / 'schedule.json')
    require(len(rows) == 40, 'Schedule differs')
    (run / 'launcher').mkdir()
    with (run / 'execution_started.json').open('x', encoding='utf-8') as stream:
        json.dump({'utc': utc_now(), 'pid': os.getpid(), 'schedule_sha256': sha(run / 'schedule.json')}, stream)
    active, completed, errors, exit_journal_attempted, created_ids = {}, {}, [], set(), set()
    cursor = 0
    budget = read(run / 'resource_budget.json'); forecast = read(run / 'resource_review.json')
    last_status = None

    def record_exit(identifier, proc, row, handles, rc):
        for handle in handles:
            try:
                handle.close()
            except Exception as exc:
                errors.append({'episode_id': identifier, 'error': 'LAUNCH_LOG_CLOSE_FAILED', 'detail': repr(exc)})
        cost = safe_cell_cost(run, row)
        record = {**cost, 'returncode': rc, 'launcher_pid': proc.pid, 'utc': utc_now()}
        worker = run / row['cell'] / 'worker_started.json'
        try:
            if worker.exists():
                record['worker_pid'] = json.loads(worker.read_text(encoding='utf-8').splitlines()[0])['pid']
        except Exception as exc:
            errors.append({'episode_id': identifier, 'error': 'WORKER_IDENTITY_READ_FAILED', 'detail': repr(exc)})
        completed[identifier] = record
        require(identifier not in exit_journal_attempted, 'Duplicate EXITED journal attempt')
        exit_journal_attempted.add(identifier)
        try:
            journal(run / 'dispatch_order.jsonl', {'event': 'EXITED', **record})
        except Exception as exc:
            # Append/fsync may have written bytes before raising. Do not append
            # another EXITED. The independent exit record below retains truth.
            errors.append({'episode_id': identifier, 'error': 'EXIT_JOURNAL_WRITE_FAILED', 'detail': repr(exc)})
        if rc != 0 or not cost['completed'] or not cost['cost_complete']:
            errors.append({'episode_id': identifier, 'error': 'WORKER_FAILURE_OR_INCOMPLETE_EVIDENCE',
                           'returncode': rc, 'cost_errors': cost.get('cost_errors', [])})

    try:
        while cursor < len(rows) or active:
            for identifier, item in list(active.items()):
                proc, row, handles = item
                rc = proc.poll()
                if rc is None:
                    continue
                del active[identifier]
                record_exit(identifier, proc, row, handles, rc)
                if errors:
                    break
            if errors:
                break
            while not errors and cursor < len(rows) and len(active) < budget['max_workers']:
                storage = remaining_storage_projection(run, rows, completed, forecast, budget)
                if storage['free_bytes'] < storage['required_free_bytes']:
                    errors.append({'error': 'INSUFFICIENT_STORAGE', **storage})
                    break
                row = rows[cursor]; cell = run / row['cell']; cell.mkdir(parents=True, exist_ok=False)
                created_ids.add(row['episode_id'])
                save(cell / 'spec.json', make_spec(row, run))
                out = (run / 'launcher' / f'{row["episode_id"]:04d}.stdout.log').open('xb')
                err = (run / 'launcher' / f'{row["episode_id"]:04d}.stderr.log').open('xb')
                environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                    OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
                proc = subprocess.Popen([sys.executable, '-B', str(ROOT / 'scripts/nature34_clpu_tail_execution.py'),
                    '--run', str(run), '--cell', str(cell)], cwd=ROOT, env=environment, stdout=out, stderr=err,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                active[row['episode_id']] = (proc, row, (out, err))
                cursor += 1
                journal(run / 'dispatch_order.jsonl', {'event': 'STARTED', 'episode_id': row['episode_id'],
                    'cell': row['cell'], 'launcher_pid': proc.pid, 'spec_sha256': sha(cell / 'spec.json'), 'utc': utc_now()})
            status = (len(completed), len(active), len(errors))
            if status != last_status:
                totals = cost_totals(completed.values())
                value = {'status': 'DRAINING_ON_ERROR' if errors else 'EVALUATING', 'run': run.name,
                    'pid': os.getpid(), 'completed': len(completed), 'scheduled': 40, 'in_flight': len(active),
                    'physical_points_completed': totals['physical_points'], **totals,
                    'errors': errors, 'utc': utc_now()}
                save(run / 'progress.json', value); save(ACTIVE, value)
                print(json.dumps(value), flush=True); last_status = status
            if errors and not active:
                break
            time.sleep(.5)
    except BaseException as exc:
        errors.append({'error': repr(exc), 'traceback': traceback.format_exc()})
    finally:
        # First wait for ALL writers; only then perform fallible file work.
        drained = drain_workers(active, errors)
        for item in drained:
            record_exit(*item)
    costs = [safe_cell_cost(run, r) for r in rows if r['episode_id'] in created_ids]
    totals = cost_totals(costs)
    if totals['unknown_cost_cells'] and not errors:
        errors.append({'error': 'INCOMPLETE_COST_EVIDENCE', 'cells': totals['unknown_cost_cells']})
    accounting = {'status': 'COMPLETE_EVALUATION' if len(completed) == 40 and not errors else 'STOPPED_INCOMPLETE',
        'run': run.name, 'tasks': costs, 'completed': len(completed), 'scheduled': 40, 'errors': errors,
        **totals, 'actual_exits': list(completed.values()),
        'unconfirmed_processes': [{'episode_id': i, 'launcher_pid': item[0].pid}
                                 for i, item in active.items()],
        'maximum_physical_points': 10240, 'new_training_tasks': 0, 'new_optimizer_updates': 0, 'utc': utc_now()}
    if active:
        accounting['status'] = 'STOPPED_PROCESS_EXIT_UNCONFIRMED'
    require(accounting['known_physical_points'] <= 10240, 'Known physical allocation exceeded')
    for target in (run / 'cost_accounting.json', ACTIVE):
        try:
            save(target, accounting)
        except Exception as exc:
            errors.append({'error': 'FINAL_CONTROL_WRITE_FAILED', 'path': str(target), 'detail': repr(exc)})
            accounting['status'] = 'STOPPED_INCOMPLETE'
    if errors:
        # A separate best-effort artifact records post-write failures without
        # retrying an ambiguous append or modifying any scientific evidence.
        try:
            save(run / 'controller_stop_evidence.json', accounting)
        except Exception as exc:
            errors.append({'error': 'STOP_EVIDENCE_WRITE_FAILED', 'detail': repr(exc)})
            print(json.dumps(accounting), flush=True)
    require(not errors and len(completed) == 40, 'Evaluation stopped; inspect saved evidence')
    save(run / 'evaluation_completion.json', {'status': 'ALL_40_WORKERS_COMPLETED',
        'cost_accounting_sha256': sha(run / 'cost_accounting.json'), 'utc': utc_now()})
    return accounting


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--tests', required=True); p.add_argument('--authorization', required=True)
    p.add_argument('--protocol', default=str(ROOT / 'docs/ppo_tail_pilot_protocol_v1.md'))
    p = sub.add_parser('evaluate'); p.add_argument('--run', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        run = prepare(Path(args.tests), Path(args.authorization), Path(args.protocol))
        print(json.dumps({'run': str(run), 'status': 'FROZEN_NOT_STARTED'}))
    else:
        run_evaluations(Path(args.run))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
