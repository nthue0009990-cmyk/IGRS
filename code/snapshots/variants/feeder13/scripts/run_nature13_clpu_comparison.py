"""Single, bounded execution of the frozen eighteen NC13 cold tasks."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gzip
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
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
sys.dont_write_bytecode = True
from dnr.nature13_clpu_evidence import save, append, file_sha as sha, digest

DESIGN = ROOT/'runs/20261105T005125_405484Z_nature13_clpu_comparison_design'
DESIGN_SHA = '0803a1b92a40120255d47140695f0d3bf244e403218f240c3527de42cc4e7624'
ACTIVE = ROOT/'artifacts/nature13_clpu_comparison_active.json'
ADDITIONS = ['src/dnr/nature13_clpu_evidence.py', 'src/dnr/nature13_clpu_routes.py',
             'scripts/run_nature13_clpu_comparison.py', 'scripts/audit_nature13_clpu_comparison.py',
             'tests/test_nature13_clpu_execution.py']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def verify_design(folder=DESIGN):
    require(sha(folder/'freeze_manifest.json') == DESIGN_SHA, 'Scientific freeze changed')
    for name, h in read(folder/'freeze_manifest.json').items():
        require(sha(folder/name) == h, 'Scientific file changed: '+name)
    binding = read(folder/'source_binding.json')
    for name, h in binding['source_hashes'].items():
        require(sha(ROOT/name) == sha(folder/'source'/name) == h, 'Method source changed: '+name)
    for name, h in binding['original_input_hashes'].items():
        require(sha(ROOT/name) == sha(folder/binding['input_copies'][name]) == h, 'Input changed: '+name)
    from run_nature13_clpu_acceptance import environment
    require(environment() == read(folder/'environment.json'), 'Frozen interpreter/dependencies changed')


def prepare(tests):
    verify_design()
    require(tests and tests.is_file(), 'Passing integration test XML required')
    cases = list(ET.parse(tests).getroot().iter('testcase'))
    require(cases and not any(list(c.iter('failure'))+list(c.iter('error'))+list(c.iter('skipped')) for c in cases), 'Tests not all passed')
    require(any('test_nature13_clpu_execution' in c.get('classname','') for c in cases), 'New bridge checks missing')
    require(not ACTIVE.exists(), 'Inspect previous execution; no automatic repeat preparation')
    acceptance = read(ROOT/'artifacts/nature13_clpu_acceptance_active.json')
    require(acceptance['acceptance_passed'] is True, 'Acceptance gate absent')
    parent = ROOT/'runs'/acceptance['run']
    require(sha(parent/'completion_manifest.json') == acceptance['completion_manifest_sha256'], 'Acceptance seal changed')
    for name, h in read(parent/'completion_manifest.json').items():
        require(sha(parent/name) == h, 'Acceptance file changed: '+name)
    run = ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_nature13_clpu_comparison')
    run.mkdir(exist_ok=False)
    shutil.copytree(DESIGN, run/'design')
    shutil.copyfile(tests, run/'software_tests.xml')
    hashes = dict(read(DESIGN/'source_binding.json')['source_hashes'])
    for name in ADDITIONS:
        hashes[name] = sha(ROOT/name)
    for name, h in hashes.items():
        target = run/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
        require(sha(target) == h, 'Source copy changed')
    schedule = read(DESIGN/'schedule.json')
    for spec in schedule:
        cell = run/'tasks'/f"{spec['task_id']:03d}"
        cell.mkdir(parents=True)
        save(cell/'spec.json', spec)
    scope = {'version': 'NC13_SHARED_EVIDENCE_COMPARISON_V1', 'scientific_design_sha256': DESIGN_SHA,
        'physical_config_sha256': sha(DESIGN/'physical_config.json'),
        'condition_sha256': sha(DESIGN/'condition.json'), 'source_hashes': hashes,
        'original_input_hashes': read(DESIGN/'source_binding.json')['original_input_hashes'],
        'physical_model': 'NC13_CLPU_SINGLE_TERMINAL_OPEN_DEAD_DEVICES_ISOLATED_V1',
        'alpha': .375, 'algorithm': 0, 'load_count': 15, 'command_bits': 11}
    save(run/'physical_scope.json', scope)
    save(run/'authorization.json', {'date': '2026-11-05', 'execution_authorized': True,
        'instruction': 'Connect NC13 evidence sharing, cost accounting and audit; after checks run the frozen 18 P/E/Forward tasks; preserve Beam, all parameters and failures; no training.',
        'scientific_design_sha256': DESIGN_SHA, 'single_controller': True, 'retry': False})
    prepared = {'prepared_utc': utc(), 'source_hashes': hashes, 'design_sha256': DESIGN_SHA,
        'input_hashes': {p.relative_to(run).as_posix(): sha(p) for p in run.rglob('*') if p.is_file() and 'source' not in p.relative_to(run).parts},
        'new_physical_queries': 0, 'training_steps': 0, 'test_count': len(cases)}
    save(run/'prepared.json', prepared)
    save(ACTIVE, {'run': run.name, 'status': 'PREPARED_PENDING_PREFLIGHT', 'physical_queries': 0})
    return run


def verify(run):
    frozen = read(run/'prepared.json')
    verify_design(run/'design')
    for name, h in frozen['source_hashes'].items():
        require(sha(ROOT/name) == sha(run/'source'/name) == h, 'Execution source changed: '+name)
    for name, h in frozen['input_hashes'].items():
        require(sha(run/name) == h, 'Execution input changed: '+name)
    require(read(run/'authorization.json')['execution_authorized'] is True, 'User execution authority absent')


def worker(run, task_id, deadline):
    cell = run/'tasks'/f'{task_id:03d}'
    start = time.monotonic()
    service = None
    summary = {'task_id': task_id, 'status': 'STARTED', 'new_training_steps': 0}
    try:
        require((run/'execution_started.json').exists() and (cell/'launch.json').exists(), 'Controller marker absent')
        launch = read(cell/'launch.json')
        require(launch['deadline_monotonic'] == deadline, 'Deadline binding changed')
        with (cell/'worker_started.json').open('x', encoding='utf-8') as out:
            json.dump({'started_utc': utc(), 'pid': os.getpid(), 'deadline_monotonic': deadline}, out)
        verify(run)
        from prepare_nature13_clpu_static import build_environment
        from dnr.nature13_clpu_evidence import EvidenceService
        from dnr.nature13_clpu_routes import run_routes
        from dnr.nature34_clpu_physical import engine_numeric_lists
        spec = read(cell/'spec.json')
        require(spec == read(run/'design/schedule.json')[task_id], 'Scheduled identity changed')
        ctx = read(run/'design'/spec['candidate_context_file'])
        env = build_environment(ROOT, 'line.'+spec['scenario_id'].removeprefix('nc13_'), spec['event_id'])
        config = read(run/'design/physical_config.json')
        settings = read(run/'design/settings.json')
        with gzip.open(cell/'search_journal.jsonl.gz', 'xt', encoding='utf-8', compresslevel=3) as journal:
            def emit(phase, kind, value):
                append(journal, {'phase': phase, 'kind': kind, 'value': engine_numeric_lists(value)})
            def write_phase(phase, result):
                path = cell/phase/'phase_result.json'
                save(path, engine_numeric_lists(result))
                return sha(path)
            with EvidenceService(ROOT, cell, env, config, read(run/'physical_scope.json'),
                    spec['phase_order'], deadline, physical_budget=settings['physical_queries_per_task'],
                    logical_budget=settings['logical_points_per_phase'], entry_budget=settings['numerical_entries_per_task']) as service:
                phases, merged = run_routes(env, service, read(run/'design/branch_config.json'), spec['arm'],
                    ctx['candidate_generation'], write_phase, emit)
        plan = merged['best_plan']['validation']
        final = {**engine_numeric_lists(merged), 'task_id': task_id, 'spec': spec,
            'quality_pct': 100*plan['energy_kwh']/(env.total_kw*.5),
            'energy_kwh': plan['energy_kwh'], 'guard_energy_kwh': phases['GUARD']['best_plan']['validation']['energy_kwh'],
            'actual_physical_points': service.budget.queries,
            'logical_points': sum(v.points for v in service.views.values()),
            'numerical_entries': service.budget.returned_entries, 'service': service.summary(),
            'worker_elapsed_seconds': time.monotonic()-start, 'training_steps': 0}
        save(cell/'result.json', final)
        summary.update(status='COMPLETE_PENDING_AUDIT', result_sha256=sha(cell/'result.json'))
    except BaseException as exc:
        summary.update(status='STOPPED_ERROR', error=repr(exc), traceback=traceback.format_exc())
    finally:
        summary.update(worker_elapsed_seconds=time.monotonic()-start,
            budget=service.budget.stats() if service and service.budget else None)
        save(cell/'worker_summary.json', summary)
        save(cell/'worker_manifest.json', {p.relative_to(cell).as_posix(): sha(p) for p in cell.rglob('*')
                                          if p.is_file() and p.name not in ('worker_manifest.json', 'worker.log', 'controller_result.json')})
    return 0 if summary['status'] == 'COMPLETE_PENDING_AUDIT' else 2


def execute(run):
    verify(run)
    gate = read(run/'preflight.json')
    require(gate['status'] == 'PASS' and gate['prepared_sha256'] == sha(run/'prepared.json'), 'Preflight gate missing')
    settings = read(run/'design/settings.json')
    with (run/'execution_started.json').open('x', encoding='utf-8') as out:
        json.dump({'started_utc': utc(), 'prepared_sha256': sha(run/'prepared.json'), 'retry': False}, out)
    start = time.monotonic()
    end = start+settings['controller_seconds']
    completed = []
    stop = None
    for spec in read(run/'design/schedule.json'):
        if time.monotonic() >= end:
            stop = 'CONTROLLER_TIME_LIMIT'; break
        if shutil.disk_usage(ROOT).free < settings['minimum_free_bytes']:
            stop = 'SPACE_RESERVE_LIMIT'; break
        cell = run/'tasks'/f"{spec['task_id']:03d}"
        began = time.monotonic()
        deadline = min(end, began+settings['task_seconds'])
        save(cell/'launch.json', {'started_utc': utc(), 'deadline_monotonic': deadline, 'no_retry': True})
        child_env = os.environ.copy()
        child_env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        with (cell/'worker.log').open('x', encoding='utf-8') as out:
            proc = subprocess.Popen([sys.executable, '-X', 'utf8', '-B', str(Path(__file__).resolve()),
                'worker', '--run', str(run), '--task-id', str(spec['task_id']), '--deadline', str(deadline)],
                cwd=ROOT, env=child_env, stdout=out, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            timeout = False
            try:
                code = proc.wait(timeout=max(.001, deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                timeout = True; proc.kill(); code = proc.wait()
        record = {'task_id': spec['task_id'], 'context_id': spec['context_id'], 'arm': spec['arm'],
            'returncode': code, 'timed_out': timeout, 'full_cold_seconds': time.monotonic()-began,
            'status': 'COMPLETE' if code == 0 and not timeout else 'STOPPED_ERROR'}
        save(cell/'controller_result.json', record)
        completed.append(record)
        save(run/'progress.json', {'finished_tasks': completed, 'elapsed_seconds': time.monotonic()-start})
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if code != 0 or timeout:
            stop = 'TASK_TIMEOUT' if timeout else 'TASK_RELIABILITY_OR_IMPLEMENTATION_ERROR'
            break
    attempted = {x['task_id'] for x in completed}
    all_tasks = [{**spec, 'execution_status': next((r['status'] for r in completed if r['task_id']==spec['task_id']), 'NOT_RUN')}
                 for spec in read(run/'design/schedule.json')]
    result = {'status': 'COMPLETE_18_PENDING_AUDIT' if len(completed)==18 and stop is None else 'STOPPED_RETAINED',
        'stop_reason': stop, 'elapsed_controller_seconds': time.monotonic()-start, 'tasks': all_tasks,
        'attempted_tasks': len(attempted), 'retries': 0, 'training_steps': 0}
    save(run/'controller_summary.json', result)
    save(run/'execution_manifest.json', {p.relative_to(run).as_posix(): sha(p) for p in run.rglob('*')
        if p.is_file() and p.name != 'execution_manifest.json'})
    save(ACTIVE, {'run': run.name, 'status': result['status'], 'attempted_tasks': len(attempted)})
    print(json.dumps({k: v for k,v in result.items() if k!='tasks'}, ensure_ascii=False), flush=True)
    return 0 if result['status']=='COMPLETE_18_PENDING_AUDIT' else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare','execute','worker'])
    parser.add_argument('--tests', type=Path)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--task-id', type=int)
    parser.add_argument('--deadline', type=float)
    args = parser.parse_args()
    if args.mode=='prepare':
        print(prepare(args.tests))
    else:
        require(args.run and args.run.resolve().parent==(ROOT/'runs').resolve(), 'Bound project run required')
        raise SystemExit(execute(args.run.resolve()) if args.mode=='execute'
                         else worker(args.run.resolve(), args.task_id, args.deadline))
