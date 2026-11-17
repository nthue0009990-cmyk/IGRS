"""Prepare and execute the fixed 384-cell D060 validation; never train a model.

The unchanged D059 cold worker is the scientific backend. Two independent
backend runs allow a single predeclared Beam expansion-cap sensitivity check.
The controller only dispatches the frozen schedule, records cumulative costs,
and drains writers on failure. Scientific audit/report are separate stages.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
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
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
sys.dont_write_bytecode = True

from d060_validation_common import (SOURCE_RUN, SOURCE_SHA, SUPPLEMENT_RUN, SUPPLEMENT_SHA,
    EVENT_IDS, SEEDS, METHODS, VARIANTS, TOTAL_CELLS, TOTAL_POINTS, require, read, sha,
    canonical, utc_now, atomic_save, inside, normalize_manifest, build_schedule, validate_fixed_config)
from nature34_experiment_support import dependencies, freeze

ACTIVE = ROOT / 'artifacts/d060_validation_active.json'
LOCK = ROOT / 'artifacts/d060_validation_execution.lock'
NEW_SOURCES = ['scripts/d060_validation_common.py', 'scripts/run_d060_validation.py',
               'tests/test_run_d060_validation.py', 'scripts/audit_d060_validation.py',
               'tests/test_audit_d060_validation.py', 'scripts/report_d060_validation.py',
               'tests/test_report_d060_validation.py', 'scripts/audit_d059_supplement.py']
ROOT_INPUTS = ['training_scenarios.json', 'author_static_metadata.json', 'load_reference.json',
    'conditions.json', 'physical_config.json', 'operating_reference.json', 'base_load_properties.json',
    'command_inventory.json', 'source_pointer.json', 'metadata.json', 'adaptation.json',
    'author_input_precheck.json', 'condition.json', 'source_provenance.json', 'event_split.json']


def validate_junit(path):
    tree = ET.parse(path)
    cases = list(tree.getroot().iter('testcase'))
    require(cases and not any(list(c.iter('failure')) or list(c.iter('error')) or list(c.iter('skipped')) for c in cases),
            'Software validation has failed/skipped/empty test evidence')
    for module in ('test_run_d060_validation', 'test_audit_d060_validation', 'test_report_d060_validation'):
        require(any(module in c.get('classname', '') for c in cases),
                'D060 required test evidence is missing: ' + module)
    return len(cases)


def directory_bytes(directory):
    """Logical size metadata only; never parse or load large scientific logs."""
    return sum(p.stat().st_size for p in Path(directory).rglob('*') if p.is_file())


def historical_storage_projection(source, source_manifest):
    require(sha(source / 'schedule.json') == source_manifest['schedule.json'], 'Unbound historical size population')
    grouped = {method: [] for method in METHODS}
    for row in read(source / 'schedule.json'):
        if row['phase'] == 'evaluation' and row['method'] in grouped:
            grouped[row['method']].append(directory_bytes(source / row['cell']))
    require(all(grouped.values()), 'Historical storage population is incomplete')
    return {'source_run': SOURCE_RUN, 'source_completion_sha256': SOURCE_SHA,
        'source_schedule_sha256': source_manifest['schedule.json'], 'metadata_only': True,
        'by_method': {method: {'n': len(values), 'mean_logical_bytes': sum(values) / len(values),
                              'maximum_logical_bytes': max(values)} for method, values in grouped.items()},
        'pressure_initial_source': 'Historical beam; empirical estimate, not a guaranteed upper bound',
        'forecast_margin': 1.25, 'report_reserve_bytes': 2 * 2**30,
        'observation_rule': 'max(historical method mean, completed current variant/method mean)',
        'observed_utc': utc_now()}


def storage_forecast(study, schedule, completed, minimum_free_bytes):
    projection = read(study / 'storage_projection.json')
    observed = {}
    for record in completed.values():
        row = record['template']; key = (row['variant'], row['method'])
        size = record.get('logical_bytes')
        if size is None:
            size = directory_bytes(ROOT / 'runs' / row['backend_run'] / row['cell'])
        observed.setdefault(key, []).append(size)
    remaining, pending, in_flight = 0.0, 0, 0
    for row in schedule:
        if row['episode_id'] in completed:
            continue
        history = projection['by_method'][row['method']]['mean_logical_bytes']
        samples = observed.get((row['variant'], row['method']), [])
        estimate = max(history, sum(samples) / len(samples) if samples else history)
        cell = ROOT / 'runs' / row['backend_run'] / row['cell']
        if cell.exists():
            estimate = max(0.0, estimate - directory_bytes(cell)); in_flight += 1
        else:
            pending += 1
        remaining += estimate
    required = int(remaining * projection['forecast_margin'] + projection['report_reserve_bytes'] + minimum_free_bytes + .999999)
    return {'remaining_predicted_logical_bytes': remaining, 'required_free_bytes': max(minimum_free_bytes, required),
        'free_bytes': shutil.disk_usage(ROOT).free, 'minimum_free_bytes': minimum_free_bytes,
        'margin': projection['forecast_margin'], 'report_reserve_bytes': projection['report_reserve_bytes'],
        'pending_tasks': pending, 'in_flight_tasks': in_flight, 'completed_tasks': len(completed),
        'estimate_is_not_upper_bound': True, 'metadata_only': True, 'utc': utc_now()}


def bound_copy(source, target, manifest, relative):
    relative = str(relative).replace('\\', '/')
    require(relative in manifest, 'Consumed source was not sealed: ' + relative)
    original = inside(source, relative)
    require(sha(original) == manifest[relative], 'Source changed: ' + relative)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(original, target)
    require(sha(target) == manifest[relative], 'Copy changed: ' + relative)
    return manifest[relative]


def source_evidence():
    source, supplement = ROOT / 'runs' / SOURCE_RUN, ROOT / 'runs' / SUPPLEMENT_RUN
    require(sha(source / 'completion_manifest.json') == SOURCE_SHA, 'Original D059 source seal changed')
    require(sha(supplement / 'completion_manifest.json') == SUPPLEMENT_SHA, 'Verified D059 supplement seal changed')
    sm = normalize_manifest(read(source / 'completion_manifest.json'))
    am = normalize_manifest(read(supplement / 'completion_manifest.json'))
    def checked(base, manifest, name):
        require(name in manifest and sha(base / name) == manifest[name], 'Unbound source evidence: ' + name)
        return read(base / name)
    for name in ('config.json', 'prepared.json', 'dependencies.json', 'python_environment.json'):
        checked(source, sm, name)
    validation = checked(supplement, am, 'learning_evidence_validation.json')
    require(validation['status'] == 'VERIFIED_EXISTING_LEARNING_AUDIT_BINDINGS' and len(validation['records']) == 49,
            'Complete old learning-audit attribution missing')
    require(len({r['verification'].replace('\\', '/') for r in validation['records']}) == 49,
            'Duplicated old learning audit')
    for row in validation['records']:
        name = row['verification'].replace('\\', '/')
        record = checked(source, sm, name)
        require(row['sha256'] == sm[name], 'Old learning audit hash binding differs')
        output = record['output']
        require(output.get('errors') == [] and output.get('status', '').startswith('VERIFIED_CLPU_BRANCH_'),
                'Old learning audit was not successful')
    attrib = checked(supplement, am, 'final_attribution.json')
    require(attrib['status'] == 'VERIFIED_CLPU_BRANCH_LEARNING_ATTRIBUTION' and attrib['errors'] == [] and
            attrib['evaluation_cells'] == 240 and attrib['same_seed_bc_ppo_start'], 'Original endpoint attribution missing')
    checkpoints = checked(source, sm, 'final_checkpoints.json')
    bindings = []
    for seed in SEEDS:
        for method in METHODS[2:]:
            name = checkpoints[str(seed)][method].replace('\\', '/')
            require(name in sm and sha(source / name) == sm[name], 'Old endpoint changed')
            bindings.append({'seed': seed, 'method': method, 'source_relative': name,
                'source_sha256': sm[name], 'checkpoint': f'models/seed{seed}/{method}.pt',
                'learning_audit_binding': 'all 49 sealed source learning audits and final attribution'})
    require(len(bindings) == 9, 'Expected exactly nine inherited model endpoints')
    return source, supplement, sm, am, validation, bindings


def select_contexts(source, source_manifest):
    for name in ('evaluation_scenarios.json', 'event_split.json', 'evaluation_contexts.json'):
        require(name in source_manifest and sha(source / name) == source_manifest[name], 'Unbound population: ' + name)
    faults = read(source / 'evaluation_scenarios.json')
    split = read(source / 'event_split.json')
    require(len(faults) == len({r['scenario_id'] for r in faults}) == 8 and
            all(s['partition'] == 'build' and s['fault_order'] == 2 for s in faults), 'Source dev faults differ')
    require(split['dev'] == [21, 25] + EVENT_IDS, 'Frozen development whole-event order changed')
    require({r['event_id'] for r in read(source / 'evaluation_contexts.json')} == {21, 25}, 'Old evaluation population differs')
    from dnr.clpu_curves import load_events, split_events, SOURCE_ZIP_SHA256
    from prepare_nature34_clpu_pilot import CURVE_ZIP
    require(sha(CURVE_ZIP) == SOURCE_ZIP_SHA256, 'Archived CLPU source changed')
    partitions = split_events(load_events(CURVE_ZIP))
    require({k: [r['event_id'] for r in v] for k, v in partitions.items()} == split, 'Recomputed event partition differs')
    events = partitions['dev'][2:]
    require([r['event_id'] for r in events] == EVENT_IDS, 'New event selection differs')
    contexts = [{'scenario_id': s['scenario_id'], 'event_id': e} for s in faults for e in EVENT_IDS]
    return faults, events, contexts


def prepare(config_path, protocol_path, resources_path, junit_path, extra_sources=()):
    config = validate_fixed_config(read(config_path))
    tests = validate_junit(junit_path)
    resources = read(resources_path)
    require(resources.get('launch_ready') is True and type(resources.get('required_free_bytes')) is int,
            'Resource readiness has not been established')
    require(resources['required_free_bytes'] >= config['minimum_free_bytes'], 'Resource reserve is below safety floor')
    require(shutil.disk_usage(ROOT).free >= resources['required_free_bytes'], 'Insufficient estimated launch capacity')
    source, supplement, sm, am, learning_validation, model_bindings = source_evidence()
    faults, events, contexts = select_contexts(source, sm)
    schedule = build_schedule(contexts)
    require(dependencies() == read(source / 'dependencies.json') and
            sha(Path(sys.executable)) == read(source / 'python_environment.json')['sha256'], 'Runtime dependencies changed')
    inherited = {name.replace('\\', '/') for name in read(source / 'prepared.json')['source_files']
                 if str(name).endswith('.py')}
    for name in inherited:
        archived = 'source/' + name
        require(archived in sm and sha(ROOT / name) == sm[archived], 'Inherited scientific code changed: ' + name)
    sources = sorted(inherited | set(NEW_SOURCES) | {str(x).replace('\\', '/') for x in extra_sources})
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    study = ROOT / 'runs' / (stamp + '_d060_validation')
    study.mkdir(exist_ok=False)
    atomic_save(study / 'study_config.json', config)
    shutil.copyfile(protocol_path, study / 'protocol.md')
    shutil.copyfile(resources_path, study / 'resource_review.json')
    shutil.copyfile(junit_path, study / 'test_evidence.xml')
    atomic_save(study / 'source_binding.json', {'source_run': SOURCE_RUN, 'source_completion_sha256': SOURCE_SHA,
        'source_status_retained': 'INCONCLUSIVE', 'supplement_run': SUPPLEMENT_RUN,
        'supplement_completion_sha256': SUPPLEMENT_SHA, 'old_failures_preserved': True,
        'new_training_tasks': 0, 'new_optimizer_updates': 0})
    atomic_save(study / 'model_bindings.json', model_bindings)
    atomic_save(study / 'storage_projection.json', historical_storage_projection(source, sm))
    atomic_save(study / 'learning_evidence_validation.json', learning_validation)
    for row in learning_validation['records']:
        bound_copy(source, study / 'inherited_learning_audits' / row['verification'].replace('\\', '/'), sm, row['verification'])
    for name in ('final_attribution.json', 'verification.json', 'resource_review.json'):
        bound_copy(supplement, study / 'source_supplement' / name, am, name)
    atomic_save(study / 'evaluation_contexts.json', contexts)
    children = {}
    for variant, cap in VARIANTS.items():
        child = ROOT / 'runs' / (stamp + '_d060_' + variant)
        child.mkdir(exist_ok=False)
        children[variant] = child.name
        for name in ROOT_INPUTS:
            bound_copy(source, child / name, sm, name)
        for path in (source / 'source_inputs').rglob('*'):
            if path.is_file():
                name = path.relative_to(source).as_posix()
                bound_copy(source, child / name, sm, name)
        child_config = deepcopy(read(source / 'config.json'))
        child_config.update(validation_study='D060', variant=variant)
        child_config['branch_search']['max_branch_expansions'] = cap
        atomic_save(child / 'config.json', child_config)
        atomic_save(child / 'selected_scenarios.json', faults)
        atomic_save(child / 'selected_events.json', events)
        atomic_save(child / 'evaluation_contexts.json', contexts)
        atomic_save(child / 'training_contexts.json', [])
        atomic_save(child / 'schedule.json', [r for r in schedule if r['variant'] == variant])
        atomic_save(child / 'model_bindings.json', model_bindings if variant == 'main' else [])
        atomic_save(child / 'study_binding.json', {'study_run': study.name, 'variant': variant,
            'source_run': SOURCE_RUN, 'source_completion_sha256': SOURCE_SHA,
            'supplement_run': SUPPLEMENT_RUN, 'supplement_completion_sha256': SUPPLEMENT_SHA,
            'config_backend_decision_is_compatibility_only': True, 'new_training_tasks': 0})
        if variant == 'main':
            for row in model_bindings:
                bound_copy(source, child / row['checkpoint'], sm, row['source_relative'])
        atomic_save(child / 'dependencies.json', dependencies())
        atomic_save(child / 'python_environment.json', {'executable': sys.executable, 'version': sys.version,
                                                      'sha256': sha(Path(sys.executable))})
        freeze(child, sources)
    atomic_save(study / 'children.json', children)
    atomic_save(study / 'schedule.json', [{**row, 'backend_run': children[row['variant']]} for row in schedule])
    atomic_save(study / 'dependencies.json', dependencies())
    atomic_save(study / 'python_environment.json', {'executable': sys.executable, 'version': sys.version,
                                                  'sha256': sha(Path(sys.executable))})
    atomic_save(study / 'resource_budget.json', {'episodes': TOTAL_CELLS, 'main_episodes': 352,
        'beam_exp1024_episodes': 32, 'allocated_physical_points': TOTAL_POINTS,
        'maximum_numerical_entries': TOTAL_CELLS * 3 + 4 * TOTAL_POINTS,
        'hard_total_deadline_seconds': None, 'worker_watchdog_seconds': 3600,
        'new_training_tasks': 0, 'new_optimizer_updates': 0, 'automatic_retry': False})
    prepared = freeze(study, sources)
    atomic_save(study / 'prepare_summary.json', {'status': 'FROZEN_D060_NO_NEW_PHYSICS',
        'study_run': study.name, 'children': children, 'tests': tests,
        'prepared_sha256': sha(study / 'prepared.json'), 'freeze_sha256': prepared['freeze_sha256'],
        'new_powerflow_calls': 0, 'new_training_tasks': 0})
    atomic_save(ACTIVE, {'status': 'FROZEN_NOT_STARTED', 'run': study.name, 'children': children,
                         'pid': None, 'completed': 0, 'scheduled': TOTAL_CELLS, 'utc': utc_now()})
    return study


def verify_frozen(run):
    run = Path(run).resolve()
    require(run.parent == ROOT / 'runs', 'Run is outside workspace runs')
    prepared = read(run / 'prepared.json')
    require(sha(run / 'prephysical_freeze_manifest.json') == prepared['freeze_sha256'], 'Prephysical freeze changed')
    for name, digest in read(run / 'prephysical_freeze_manifest.json').items():
        require(sha(inside(run, name)) == digest, 'Frozen input changed: ' + name)
    for name in prepared['source_files']:
        require(sha(ROOT / name) == sha(run / 'source' / name), 'Frozen implementation changed: ' + name)
    require(dependencies() == read(run / 'dependencies.json') and
            sha(Path(sys.executable)) == read(run / 'python_environment.json')['sha256'], 'Frozen runtime changed')


def task_template(row):
    return {k: v for k, v in row.items() if k != 'backend_run'}


def make_spec(row, child, watchdog_seconds, now=None):
    template = task_template(row)
    spec = dict(template, template_sha256=hashlib.sha256(canonical(template).encode()).hexdigest(),
                deadline_utc_epoch=(time.time() if now is None else now) + watchdog_seconds)
    if row['method'] not in ('greedy', 'beam'):
        bindings = [b for b in read(child / 'model_bindings.json') if b['method'] == row['method'] and b['seed'] == row['seed']]
        require(len(bindings) == 1, 'Missing/duplicate model endpoint')
        binding = bindings[0]
        require(sha(child / binding['checkpoint']) == binding['source_sha256'], 'Inherited model endpoint changed')
        spec.update(checkpoint=binding['checkpoint'], checkpoint_sha256=binding['source_sha256'])
    return spec


def journal(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8', newline='\n') as stream:
        stream.write(canonical(record) + '\n'); stream.flush(); os.fsync(stream.fileno())


def seal_directory(directory):
    directory = Path(directory)
    manifest = {p.relative_to(directory).as_posix(): sha(p) for p in sorted(directory.rglob('*'))
                if p.is_file() and p != directory / 'completion_manifest.json'}
    atomic_save(directory / 'completion_manifest.json', manifest)
    return sha(directory / 'completion_manifest.json')


def actual_cost_accounting(study, schedule):
    """Retain costs from every started task, including failed/partial tasks."""
    rows, errors = [], []
    for template in schedule:
        cell = ROOT / 'runs' / template['backend_run'] / template['cell']
        if not cell.exists():
            continue
        row = {'episode_id': template['episode_id'], 'backend_run': template['backend_run'],
               'cell': template['cell'], 'allocated_points': template['point_budget'],
               'worker_completion_present': (cell / 'completion_manifest.json').exists(),
               'worker_error_present': (cell / 'error.json').exists()}
        if (cell / 'oracle_summary.json').exists():
            oracle = read(cell / 'oracle_summary.json')
            row['physical_points'] = oracle['points']
            row['physical_statuses'] = oracle.get('statuses', {})
        else:
            row['physical_points'] = None
        if (cell / 'command_stats.json').exists():
            stats = read(cell / 'command_stats.json')
            row['requested_numerical_entries'] = stats['requested']['numerical_solution_entries']
            row['returned_numerical_entries'] = stats['returned']['numerical_solution_entries']
            row['numerical_counters_closed'] = stats['requested'] == stats['returned']
        else:
            row.update(requested_numerical_entries=None, returned_numerical_entries=None, numerical_counters_closed=False)
        if row['physical_points'] is None or not row['numerical_counters_closed']:
            errors.append({'episode_id': template['episode_id'], 'kind': 'INCOMPLETE_COST_EVIDENCE'})
        rows.append(row)
    accounting = {'status': 'CLOSED_D060_COSTS' if not errors else 'INCOMPLETE_D060_COSTS',
        'started_tasks': len(rows), 'scheduled_tasks': TOTAL_CELLS,
        'allocated_started_points': sum(r['allocated_points'] for r in rows), 'maximum_physical_points': TOTAL_POINTS,
        'returned_physical_points': sum(r['physical_points'] or 0 for r in rows),
        'returned_numerical_entries': sum(r['returned_numerical_entries'] or 0 for r in rows),
        'new_training_tasks': 0, 'new_optimizer_updates': 0, 'budget_reset': False,
        'all_failures_retained': True, 'errors': errors, 'tasks': rows, 'utc': utc_now()}
    atomic_save(study / 'cost_accounting.json', accounting)
    return accounting


def verify_saved_manifest(directory, name='completion_manifest.json'):
    values = normalize_manifest(read(directory / name))
    for relative, digest in values.items():
        require(sha(inside(directory, relative)) == digest, 'Sealed stage evidence changed: ' + relative)
    return values


def verify_backend_seal_index(child):
    """Bind closed indexes without rereading every large physical-output file.

    The independent scientific auditor performs the full content verification.
    The controller's resumed-stage check binds its immutable manifest indexes.
    """
    values = normalize_manifest(read(child / 'completion_manifest.json'))
    for name in ('prepared.json', 'prephysical_freeze_manifest.json', 'config.json', 'schedule.json',
                 'evaluation_contexts.json', 'model_bindings.json'):
        require(name in values and sha(child / name) == values[name], 'Backend input index changed: ' + name)
    for row in read(child / 'schedule.json'):
        name = row['cell'] + '/completion_manifest.json'
        require(name in values and sha(child / name) == values[name], 'Backend/cell completion index changed')
        for relative, digest in normalize_manifest(read(child / name)).items():
            require(values.get(row['cell'] + '/' + relative) == digest, 'Cell/file index disagreement')
    for model in read(child / 'model_bindings.json'):
        require(values.get(model['checkpoint']) == model['source_sha256'] == sha(child / model['checkpoint']),
                'Backend inherited model binding changed')
    return values


def ensure_evaluation_sealed(study, children):
    """Reusing this stage never rewrites the audit's population/cost identity."""
    saved = study / 'evaluation_completion_manifest.json'
    if saved.exists():
        values = verify_saved_manifest(study, saved.name)
        require({'evaluation_status.json', 'dispatch_order.jsonl', 'cell_seals.json', 'cost_accounting.json'} <= values.keys(),
                'Saved evaluation seal omits mandatory evidence')
        require(read(study / 'evaluation_status.json')['status'] == 'EVALUATION_COMPLETE_PENDING_AUDIT',
                'Uncompleted scientific evaluation cannot enter the resumed audit')
        for name in children.values():
            verify_backend_seal_index(ROOT / 'runs' / name)
        for key, digest in read(study / 'cell_seals.json').items():
            require(sha(inside(ROOT / 'runs', key) / 'completion_manifest.json') == digest, 'Saved cell seal changed')
        return
    for variant, name in children.items():
        child = ROOT / 'runs' / name
        if (child / 'completion_manifest.json').exists():
            verify_backend_seal_index(child)
        else:
            atomic_save(child / 'evaluation_completion.json', {'status': 'COLD_EVALUATION_COMPLETE_PENDING_AUDIT',
                'variant': variant, 'cells': 352 if variant == 'main' else 32,
                'new_training_tasks': 0, 'new_optimizer_updates': 0, 'utc': utc_now()})
            seal_directory(child)
    cell_seals = {}
    for row in read(study / 'schedule.json'):
        cell = ROOT / 'runs' / row['backend_run'] / row['cell']
        cell_seals[row['backend_run'] + '/' + row['cell']] = sha(cell / 'completion_manifest.json')
    atomic_save(study / 'cell_seals.json', cell_seals)
    evidence_paths = [study / 'evaluation_status.json', study / 'dispatch_order.jsonl', study / 'cell_seals.json',
                      study / 'cost_accounting.json']
    evidence_paths += sorted((study / 'dispatched').glob('*.json')) + sorted((study / 'completed').glob('*.json'))
    atomic_save(study / 'evaluation_completion_manifest.json', {
        p.relative_to(study).as_posix(): sha(p) for p in evidence_paths})


def phase_already_complete(study, children, phase):
    if phase == 'audit':
        directory, seal = study / 'audit', 'audit_completion_manifest.json'
        if not (directory / seal).exists():
            return False
        verify_saved_manifest(directory, seal)
        verification, parent = read(directory / 'verification.json'), read(directory / 'parent_evidence.json')
        require(verification.get('status') == 'VERIFIED_D060_VALIDATION' and verification.get('errors') == []
                and verification.get('completed') == TOTAL_CELLS, 'Prior audit did not completely pass')
        require(parent['study_freeze_sha256'] == sha(study / 'prephysical_freeze_manifest.json') and
                parent['dispatch']['evaluation_completion_sha256'] == sha(study / 'evaluation_completion_manifest.json') and
                parent['child_completion_sha256'] == {v: sha(ROOT / 'runs' / n / 'completion_manifest.json') for v, n in children.items()},
                'Saved audit is bound to different evaluation evidence')
        return True
    directory = study / 'derived_report'
    if not (directory / 'completion_manifest.json').exists():
        return False
    verify_saved_manifest(directory)
    summary = read(directory / 'summary.json')
    require(summary.get('status') in ('PRACTICAL_EFFECT', 'PROMISING_SMALL_EFFECT', 'NO_GO') and summary.get('errors') == [],
            'Prior report records incomplete or unreliable evidence; automatic retry forbidden')
    binding = summary['evidence_bindings']
    require(binding['prepared_sha256'] == sha(study / 'prepared.json') and
            binding['audit_completion_sha256'] == sha(study / 'audit/audit_completion_manifest.json') and
            binding['evaluation_completion_sha256'] == sha(study / 'evaluation_completion_manifest.json') and
            binding['model_bindings_sha256'] == sha(study / 'model_bindings.json') and
            binding['backend_completion_sha256'] == {v: sha(ROOT / 'runs' / n / 'completion_manifest.json') for v, n in children.items()},
            'Saved report is bound to different scientific evidence')
    return True


def prepare_incomplete_phase(study, phase):
    """Only an interruption without a scientific failure can be resumed."""
    if phase == 'audit':
        audit = study / 'audit'
        records = list((audit / 'invocations').glob('*/result.json'))
        if (audit / 'verification.json').exists():
            records.append(audit / 'verification.json')
        for path in records:
            saved = read(path)
            require(saved.get('status') != 'FAILED_D060_VALIDATION' and
                    (not saved.get('errors') or saved.get('status') == 'STOPPED_D060_VALIDATION'),
                    'Real independent audit error retained; automatic retry forbidden')
        return
    destination = study / 'derived_report'
    if destination.exists():
        if (destination / 'summary.json').exists():
            summary = read(destination / 'summary.json')
            require(summary.get('status') != 'INCONCLUSIVE' and not summary.get('errors'),
                    'Real report evidence error retained; automatic retry forbidden')
        archive = study / 'interrupted_reports' / str(time.time_ns())
        require(destination.resolve().is_relative_to(study.resolve()) and archive.resolve().is_relative_to(study.resolve()),
                'Interrupted report preservation path is outside study')
        archive.parent.mkdir(exist_ok=True)
        os.replace(destination, archive)  # Preserve unsealed outputs; never delete them.


def finalize_scientific_evidence(study, children, env):
    """All cold writers have exited before any independent audit is started."""
    ensure_evaluation_sealed(study, children)
    phase_dir = study / 'phase_logs'; phase_dir.mkdir(exist_ok=True)
    for phase, script, evidence_path, required_status in [
        ('audit', 'audit_d060_validation.py', study / 'audit/verification.json', 'VERIFIED_D060_VALIDATION'),
        ('report', 'report_d060_validation.py', study / 'derived_report/summary.json', None)]:
        if phase_already_complete(study, children, phase):
            continue
        prepare_incomplete_phase(study, phase)
        atomic_save(study / 'pipeline_progress.json', {'phase': phase, 'status': 'RUNNING', 'utc': utc_now(), 'pid': os.getpid()})
        atomic_save(ACTIVE, {**read(study / 'progress.json'), 'status': 'AUDITING' if phase == 'audit' else 'REPORTING',
                            'phase': phase, 'pid': os.getpid(), 'utc': utc_now()})
        invocation = phase + '_' + str(time.time_ns())
        with (phase_dir / (invocation + '_stdout.log')).open('xb') as out, (phase_dir / (invocation + '_stderr.log')).open('xb') as err:
            proc = subprocess.Popen([sys.executable, '-B', str(ROOT / 'scripts' / script), '--study', str(study)],
                cwd=ROOT, env=env, stdout=out, stderr=err, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            atomic_save(study / 'pipeline_progress.json', {'phase': phase, 'status': 'RUNNING', 'utc': utc_now(),
                                                         'pid': os.getpid(), 'child_pid': proc.pid})
            code = proc.wait()  # Fixed audit/report only; no mechanical wall-time kill.
        require(code == 0 and evidence_path.exists(), 'Independent ' + phase + ' failed; original evidence retained')
        evidence = read(evidence_path)
        if required_status is not None:
            require(evidence.get('status') == required_status and evidence.get('errors') == [],
                    'Independent audit did not pass')
        else:
            require((study / 'derived_report/completion_manifest.json').exists(), 'Derived report is not sealed')
        require(phase_already_complete(study, children, phase), 'Completed phase lacks valid bound seal')
        atomic_save(study / 'pipeline_progress.json', {'phase': phase, 'status': 'COMPLETE', 'utc': utc_now(),
            'pid': os.getpid(), 'evidence': evidence_path.relative_to(study).as_posix(), 'evidence_sha256': sha(evidence_path)})
    return read(study / 'derived_report/summary.json')


def verify_cell_seal(cell):
    manifest = read(cell / 'completion_manifest.json')
    require({'spec.json', 'result.json', 'points.jsonl.gz', 'transitions.jsonl.gz',
             'oracle_summary.json', 'command_stats.json'} <= set(manifest), 'Incomplete worker seal')
    require(not (cell / 'error.json').exists(), 'Worker has a reliability error')
    for name, digest in manifest.items():
        require(sha(inside(cell, name)) == digest, 'Completed worker evidence changed: ' + name)
    oracle = read(cell / 'oracle_summary.json')
    stats = read(cell / 'command_stats.json')
    require(stats['requested'] == stats['returned'] and oracle['points'] <= 256 and
            stats['returned']['numerical_solution_entries'] == 3 + 4 * oracle['points'], 'Worker cost not closed')
    return {'cell_completion_sha256': sha(cell / 'completion_manifest.json'), 'physical_points': oracle['points'],
            'numerical_entries': stats['returned']['numerical_solution_entries']}


def reconcile_completed(study, schedule):
    completed = {}
    journal_path = study / 'dispatch_order.jsonl'
    events = [json.loads(line) for line in journal_path.read_text(encoding='utf-8').splitlines() if line.strip()] if journal_path.exists() else []
    def require_exit(row):
        matching = [event for event in events if event.get('episode_id') == row['episode_id']]
        starts = [event for event in matching if event.get('event') == 'STARTED']
        exits = [event for event in matching if event.get('event') == 'EXITED']
        require(len(starts) == len(exits) == 1 and exits[0].get('returncode') == 0 and
                starts[0].get('pid') == exits[0].get('pid') and all(
                event.get(key) == row[key] for event in matching for key in ('backend_run', 'cell')),
                'Completed cell lacks unique observed successful EXITED evidence; review required, retry forbidden')
    for row in schedule:
        child = ROOT / 'runs' / row['backend_run']
        cell = child / row['cell']
        record_path = study / 'completed' / f"{row['episode_id']:04d}.json"
        if record_path.exists():
            require_exit(row)
            record = read(record_path)
            require(record['template'] == row, 'Completion record population changed')
            evidence = verify_cell_seal(cell)
            require(all(record[k] == v for k, v in evidence.items()), 'Completed evidence changed')
            completed[row['episode_id']] = record
        elif cell.exists():
            # A valid worker seal left after a controller interruption can be
            # adopted exactly once; it is never a reason to rerun the worker.
            require((cell / 'completion_manifest.json').exists() and not (cell / 'error.json').exists(),
                    'Started cell lacks valid completion; scientific retry is forbidden: ' + row['cell'])
            require_exit(row)
            evidence = verify_cell_seal(cell)
            expected = task_template(row)
            spec = read(cell / 'spec.json')
            require({k: v for k, v in spec.items() if k not in ('template_sha256', 'deadline_utc_epoch',
                    'checkpoint', 'checkpoint_sha256')} == expected, 'Orphan completed task differs')
            record = {'template': row, **evidence, 'logical_bytes': directory_bytes(cell), 'worker_returncode': 0,
                      'adopted_after_controller_interruption': True, 'utc': utc_now()}
            atomic_save(record_path, record)
            completed[row['episode_id']] = record
    return completed


def archive_controller_state(study):
    """Keep overwritten control evidence byte-for-byte for every invocation."""
    history = study / 'controller_invocations' / str(time.time_ns())
    history.mkdir(parents=True, exist_ok=False)
    evidence = {}
    for name in ('progress.json', 'evaluation_status.json', 'summary.json', 'pipeline_progress.json',
                 'storage_forecast.json', 'cost_accounting.json'):
        source = study / name
        if source.exists():
            destination = history / name
            shutil.copyfile(source, destination)
            require(sha(source) == sha(destination), 'Controller history copy differs: ' + name)
            evidence[name] = sha(destination)
    atomic_save(history / 'prior_state_manifest.json', evidence)
    atomic_save(history / 'invocation.json', {'pid': os.getpid(), 'utc': utc_now(), 'budget_reset': False})
    return history


def require_resumable_evaluation(study):
    """Only a drained storage stop may automatically continue evaluation."""
    allowed = {'DISK_SAFETY_FLOOR', 'DYNAMIC_STORAGE_FORECAST'}
    for name in ('progress.json', 'evaluation_status.json', 'summary.json'):
        path = study / name
        if path.exists():
            errors = read(path).get('errors', [])
            require(all(error.get('kind') in allowed for error in errors),
                    'Prior evaluation reliability error requires review; automatic continuation is forbidden: ' + name)


def record_worker_exit(study, row, pid, code, completed):
    """Record an actually observed exit, including exits during exception drain."""
    journal(study / 'dispatch_order.jsonl', {'event': 'EXITED', 'episode_id': row['episode_id'],
        'backend_run': row['backend_run'], 'cell': row['cell'], 'pid': pid, 'returncode': code, 'utc': utc_now()})
    cell = ROOT / 'runs' / row['backend_run'] / row['cell']
    base = {'episode_id': row['episode_id'], 'returncode': code, 'backend_run': row['backend_run'],
            'cell': row['cell'], 'utc': utc_now()}
    if code != 0 or not (cell / 'completion_manifest.json').exists() or (cell / 'error.json').exists():
        return {'kind': 'WORKER_FAILURE', **base}
    try:
        evidence = verify_cell_seal(cell)
        record = {'template': row, **evidence, 'logical_bytes': directory_bytes(cell),
                  'utc': utc_now(), 'worker_returncode': code}
        atomic_save(study / 'completed' / f"{row['episode_id']:04d}.json", record)
        completed[row['episode_id']] = record
        return None
    except BaseException as exc:
        return {'kind': 'WORKER_EVIDENCE_FAILURE', **base, 'error': repr(exc), 'traceback': traceback.format_exc()}


def pid_alive(pid):
    if os.name == 'nt':
        import ctypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return ctypes.get_last_error() != 87
        code = ctypes.c_ulong()
        ok = kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))
        kernel.CloseHandle(ctypes.c_void_p(handle))
        return not ok or code.value == 259
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False


def acquire_lock(study):
    if LOCK.exists():
        prior = read(LOCK)
        require(not pid_alive(prior['pid']), 'A D060 controller is already active')
        prior_study = ROOT / 'runs' / prior['study_run']
        prior_progress = prior_study / 'progress.json'
        if prior_progress.exists():
            require(not any(pid_alive(row['pid']) for row in read(prior_progress).get('active_workers', [])),
                    'A prior D060 cold worker is still active; wait for its writer to exit')
        prior_pipeline = prior_study / 'pipeline_progress.json'
        if prior_pipeline.exists():
            pipe = read(prior_pipeline)
            require(pipe.get('status') != 'RUNNING' or not pipe.get('child_pid') or not pid_alive(pipe['child_pid']),
                    'Prior audit/report child is still active; do not overlap writers')
        # Preserve a stale lock as evidence; this never adopts a live worker.
        stale = study / ('stale_lock_' + str(time.time_ns()) + '.json')
        shutil.copyfile(LOCK, stale)
        LOCK.unlink()
    elif (study / 'progress.json').exists():
        require(not any(pid_alive(row['pid']) for row in read(study / 'progress.json').get('active_workers', [])),
                'Prior writer is still active despite a missing controller lock')
    if (study / 'pipeline_progress.json').exists():
        pipe = read(study / 'pipeline_progress.json')
        require(pipe.get('status') != 'RUNNING' or not pipe.get('child_pid') or not pid_alive(pipe['child_pid']),
                'Audit/report writer is still active despite a missing or stale controller lock')
    LOCK.parent.mkdir(exist_ok=True)
    with LOCK.open('x', encoding='utf-8') as stream:
        json.dump({'pid': os.getpid(), 'study_run': study.name, 'utc': utc_now()}, stream)


def finish_pipeline(study, children, env, epoch):
    """No evaluation-state rewrite occurs during a resumed audit/report."""
    try:
        report = finalize_scientific_evidence(study, children, env)
        final = {**read(study / 'evaluation_status.json'), 'status': 'COMPLETE_VERIFIED',
            'scientific_audit_complete': True, 'report_status': report.get('status'),
            'report': str(study / 'derived_report/report.md'), 'completed_utc': utc_now(),
            'elapsed_since_first_start_seconds': time.time() - epoch, 'pid': None,
            'errors': [], 'result_claim': 'See complete paired development validation report'}
        atomic_save(study / 'summary.json', final)
        digest = seal_directory(study)
        atomic_save(ACTIVE, {**final, 'completion_sha256': digest})
        return 0
    except BaseException as exc:
        final = {**read(study / 'evaluation_status.json'), 'status': 'STOP_NEEDS_REVIEW', 'pid': None,
            'errors': [{'kind': 'AUDIT_OR_REPORT_FAILURE', 'error': repr(exc), 'traceback': traceback.format_exc()}],
            'elapsed_since_first_start_seconds': time.time() - epoch, 'utc': utc_now()}
        atomic_save(study / 'summary.json', final); atomic_save(ACTIVE, final)
        return 1
    finally:
        if LOCK.exists() and read(LOCK).get('pid') == os.getpid():
            LOCK.unlink()


def execute(study):
    study = Path(study).resolve()
    if (study / 'summary.json').exists() and read(study / 'summary.json').get('status') == 'COMPLETE_VERIFIED':
        # Idempotent completion inspection cannot launch another scientific task
        # or overwrite a previously successful report.
        for name, digest in read(study / 'completion_manifest.json').items():
            require(sha(inside(study, name)) == digest, 'Completed study seal changed')
        return 0
    verify_frozen(study)
    config = validate_fixed_config(read(study / 'study_config.json'))
    schedule = read(study / 'schedule.json')
    children = read(study / 'children.json')
    require(len(schedule) == TOTAL_CELLS, 'Fixed schedule size changed')
    for name in children.values():
        verify_frozen(ROOT / 'runs' / name)
    acquire_lock(study)
    sealed_evaluation = (study / 'evaluation_completion_manifest.json').exists() or any(
        (ROOT / 'runs' / name / 'completion_manifest.json').exists() for name in children.values())
    try:
        archive_controller_state(study)
        if not sealed_evaluation:
            require_resumable_evaluation(study)
    except BaseException:
        if LOCK.exists() and read(LOCK).get('pid') == os.getpid():
            LOCK.unlink()
        raise
    active, completed, errors = {}, {}, []
    started = time.time()
    first = study / 'execution_started.json'
    if not first.exists():
        atomic_save(first, {'utc': utc_now(), 'epoch': started, 'pid': os.getpid(), 'budget_reset': False})
    epoch = read(first)['epoch']
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    if sealed_evaluation:
        require((study / 'evaluation_status.json').exists(), 'Closed backend lacks evaluation status')
        return finish_pipeline(study, children, env, epoch)
    status = 'RUNNING'
    def progress():
        value = {'status': status, 'run': study.name, 'children': children, 'pid': os.getpid(),
            'completed': len(completed), 'scheduled': TOTAL_CELLS, 'active_workers': [
                {'pid': pid, 'episode_id': v[1]['episode_id'], 'backend_run': v[1]['backend_run'],
                 'cell': v[1]['cell'], 'deadline_utc_epoch': v[4]} for pid, v in active.items()],
            'completed_physical_points': sum(r['physical_points'] for r in completed.values()),
            'completed_numerical_entries': sum(r['numerical_entries'] for r in completed.values()),
            'allocated_started_points': 256 * sum((ROOT / 'runs' / r['backend_run'] / r['cell']).exists() for r in schedule),
            'physical_point_cap': TOTAL_POINTS, 'elapsed_since_first_start_seconds': time.time() - epoch,
            'estimate_exceeded_16h': time.time() - epoch > 16 * 3600, 'budget_reset': False,
            'new_training_tasks': 0, 'new_optimizer_updates': 0, 'free_GiB': shutil.disk_usage(ROOT).free / 2**30,
            'errors': errors, 'utc': utc_now()}
        atomic_save(study / 'progress.json', value); atomic_save(ACTIVE, value)
    try:
        completed = reconcile_completed(study, schedule)
        pending = [r for r in schedule if r['episode_id'] not in completed]
        while pending or active:
            while pending and len(active) < config['max_workers'] and status == 'RUNNING':
                forecast = storage_forecast(study, schedule, completed, config['minimum_free_bytes'])
                atomic_save(study / 'storage_forecast.json', forecast)
                if forecast['free_bytes'] < forecast['required_free_bytes']:
                    status = 'DRAINING_RESOURCE_STOP'
                    errors.append({'kind': 'DISK_SAFETY_FLOOR' if forecast['free_bytes'] < config['minimum_free_bytes']
                                   else 'DYNAMIC_STORAGE_FORECAST', 'forecast': forecast, 'utc': utc_now()})
                    break
                row = pending.pop(0)
                child = ROOT / 'runs' / row['backend_run']; cell = child / row['cell']
                require(not cell.exists(), 'Scientific task already started; refusing repeat')
                cell.mkdir(parents=True, exist_ok=False)
                created_epoch = time.time()
                spec = make_spec(row, child, config['worker_deadline_seconds'], now=created_epoch)
                atomic_save(cell / 'spec.json', spec)
                dispatch_record = {'template': row, 'spec_created_epoch': created_epoch,
                    'deadline_utc_epoch': spec['deadline_utc_epoch'], 'watchdog_seconds': config['worker_deadline_seconds'],
                    'spec_sha256': sha(cell / 'spec.json'), 'status': 'DISPATCH_PREPARED', 'utc': utc_now()}
                dispatch_path = study / 'dispatched' / f"{row['episode_id']:04d}.json"
                require(not dispatch_path.exists(), 'Dispatch evidence already exists; repeat forbidden')
                atomic_save(dispatch_path, dispatch_record)
                out, err = (cell / 'stdout.log').open('xb'), (cell / 'stderr.log').open('xb')
                try:
                    proc = subprocess.Popen([sys.executable, '-B', str(ROOT / 'scripts/nature34_clpu_branch_execution.py'),
                        '--run', str(child), '--cell', str(cell)], cwd=ROOT, env=env, stdout=out, stderr=err,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                except BaseException:
                    out.close(); err.close(); raise
                active[proc.pid] = (proc, row, out, err, spec['deadline_utc_epoch'])
                atomic_save(dispatch_path, {**dispatch_record, 'status': 'DISPATCHED', 'pid': proc.pid, 'started_utc': utc_now()})
                journal(study / 'dispatch_order.jsonl', {'event': 'STARTED', 'episode_id': row['episode_id'],
                    'backend_run': row['backend_run'], 'cell': row['cell'], 'pid': proc.pid, 'utc': utc_now()})
            for pid, (proc, row, out, err, deadline) in list(active.items()):
                code = proc.poll()
                if code is None:
                    if time.time() > deadline and status == 'RUNNING':
                        status = 'DRAINING_WATCHDOG_STOP'
                        errors.append({'kind': 'WORKER_WATCHDOG_EXPIRED', 'episode_id': row['episode_id'], 'pid': pid,
                                       'action': 'stop dispatch; wait for writer to exit, never kill it', 'utc': utc_now()})
                    continue
                out.close(); err.close(); del active[pid]
                failure = record_worker_exit(study, row, pid, code, completed)
                if failure:
                    status = 'DRAINING_RELIABILITY_STOP'
                    errors.append(failure)
            progress()
            if status != 'RUNNING' and not active:
                break
            if active:
                time.sleep(2)
        status = 'EVALUATION_COMPLETE_PENDING_AUDIT' if len(completed) == TOTAL_CELLS and not errors else 'STOP_NEEDS_REVIEW'
    except BaseException as exc:
        errors.append({'kind': 'CONTROLLER_ERROR', 'error': repr(exc), 'traceback': traceback.format_exc(), 'utc': utc_now()})
        status = 'DRAINING_RELIABILITY_STOP'
        # Waiting is bounded by the already finite workers, not by forced
        # termination. All started evidence remains present for diagnosis.
        while active:
            for pid, (proc, row, out, err, deadline) in list(active.items()):
                code = proc.poll()
                if code is not None:
                    out.close(); err.close(); del active[pid]
                    failure = record_worker_exit(study, row, pid, code, completed)
                    if failure:
                        errors.append(failure)
            progress()
            if active:
                time.sleep(2)
        status = 'STOP_NEEDS_REVIEW'
    finally:
        progress()
        actual_cost_accounting(study, schedule)
        atomic_save(study / 'evaluation_status.json', {**read(study / 'progress.json'), 'pid': None,
            'scientific_audit_complete': False, 'result_claim': 'Pending independent audit; no performance conclusion'})
        atomic_save(ACTIVE, read(study / 'evaluation_status.json'))
        if status != 'EVALUATION_COMPLETE_PENDING_AUDIT' and LOCK.exists() and read(LOCK).get('pid') == os.getpid():
            LOCK.unlink()
    if status != 'EVALUATION_COMPLETE_PENDING_AUDIT':
        return 1
    # Retain the same single-owner lock throughout the dependent audit/report
    # pipeline. No physical task is dispatched after this point.
    return finish_pipeline(study, children, env, epoch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--execute', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--resources', type=Path)
    parser.add_argument('--junit', type=Path)
    parser.add_argument('--extra-source', action='append', default=[])
    args = parser.parse_args()
    if args.prepare:
        require(not args.execute and all((args.config, args.protocol, args.resources, args.junit)), 'Missing prepare inputs')
        run = prepare(args.config, args.protocol, args.resources, args.junit, args.extra_source)
        print(json.dumps({'run': str(run), 'status': 'FROZEN_D060_NO_NEW_PHYSICS'})); return 0
    require(args.execute is not None, 'Choose prepare or execute')
    return execute(args.execute)


if __name__ == '__main__':
    raise SystemExit(main())
