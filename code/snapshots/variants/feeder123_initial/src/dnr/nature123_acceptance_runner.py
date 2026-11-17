"""Bounded NC123 entry acceptance. Importing this module never imports DSS.

The public default is a hash-checked dry run. A separate explicit execution
freeze is required. Windows Job Objects bound the worker process tree, including
the venv launcher, before a release handshake permits backend construction.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .nature123_acceptance_budget import AcceptanceBudget


RELIABILITY = ('numerical_convergence', 'topology', 'isolation',
               'phase_connectivity', 'ac_balance')
CONSTRAINTS = ('voltage', 'line_phase_thermal', 'transformer_winding')
REQUIRED_SOURCES = (
    'scripts/run_nature123_acceptance.py',
    'src/dnr/nature123_acceptance_runner.py',
    'src/dnr/nature123_acceptance_budget.py',
    'src/dnr/nature123_clpu_physical.py',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode('utf8')).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write_json(path, value):
    """Durable complete JSON, with atomic replacement of a private temp file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


class DurableJournal:
    def __init__(self, path):
        self.path = Path(path)

    def __call__(self, row):
        row = {'wall_utc': datetime.now(timezone.utc).isoformat(),
               'monotonic': time.monotonic(), **row}
        with self.path.open('a', encoding='utf8', newline='\n') as f:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            f.flush()
            os.fsync(f.fileno())


def inside(project, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError('Bound paths must be project-relative')
    root = Path(project).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError('Bound path escapes project')
    return path


def validate_freeze(project, freeze, *, execute=False):
    """Read-only preflight. All scientific requests retain their original hashes."""
    if freeze.get('schema_version') != 'NC123_ACCEPTANCE_V1':
        raise ValueError('Unknown acceptance freeze schema')
    if type(freeze.get('physical_execution_allowed')) is not bool:
        raise ValueError('Explicit Boolean physical gate required')
    limits = freeze.get('budget', {})
    for name, bound in [('max_requests', 16), ('max_entries', 256),
                        ('memory_bytes', 4 * 1024**3)]:
        value = limits.get(name)
        if type(value) is not int or not 1 <= value <= bound:
            raise ValueError('Invalid or enlarged ' + name)
    for name, bound in [('seconds', 240), ('poll_seconds', .1)]:
        value = limits.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= bound:
            raise ValueError('Invalid or enlarged ' + name)
    if not isinstance(freeze.get('physical_config'), dict):
        raise ValueError('Explicit physical policy required')
    variant = freeze['physical_config'].get('runtime_variant', 'AUTHOR_V1')
    if variant not in ('AUTHOR_V1', 'PHASE_AVAILABLE_V2'):
        raise ValueError('Unknown frozen NC123 runtime variant')
    sources = freeze.get('source_sha256')
    if not isinstance(sources, dict) or not set(REQUIRED_SOURCES) <= set(sources):
        raise ValueError('Incomplete frozen runtime source inventory')
    if variant == 'PHASE_AVAILABLE_V2' and not {
            'src/dnr/nature123_phase_available_physical.py',
            'artifacts/nature123_model_v2/model_config.json',
            'artifacts/nature123_model_v2/manifest.json'} <= set(sources):
        raise ValueError('Incomplete phase-available model binding')
    for relative, digest in sources.items():
        if sha(inside(project, relative)) != digest:
            raise ValueError('Frozen source mismatch: ' + relative)
    binding = freeze.get('schedule', {})
    schedule_path = inside(project, binding.get('path'))
    if sha(schedule_path) != binding.get('sha256'):
        raise ValueError('Schedule SHA-256 mismatch')
    rows = read_json(schedule_path)
    if not isinstance(rows, list) or len(rows) != 16:
        raise ValueError('Exactly the 16 prepared requests must be retained')
    hashes = set()
    for i, row in enumerate(rows):
        if type(row.get('order')) is not int or row['order'] != i or row.get('status') != 'NOT_RUN':
            raise ValueError('Prepared request order/status changed')
        request = row['request']
        digest = canonical_sha({k: v for k, v in request.items() if k != 'request_sha256'})
        if request.get('request_sha256') != digest or digest in hashes:
            raise ValueError('Duplicate or invalid request SHA-256')
        hashes.add(digest)
    if execute:
        if freeze['physical_execution_allowed'] is not True:
            raise ValueError('Physical execution is not enabled in this freeze')
        readiness = freeze.get('readiness_checks', {})
        if not readiness or any(value != 'PASS' for value in readiness.values()):
            raise ValueError('Software/parameter execution readiness is incomplete')
        auth = freeze.get('authorization', {})
        if auth.get('scope') != 'ENTRY_ACCEPTANCE_16_REQUESTS' or not auth.get('record'):
            raise ValueError('Bounded execution scope must be recorded')
        policy = freeze['physical_config']
        for row in rows:
            op = row['request'].get('condition', {}).get('operating_config', {})
            if op.get('physical_execution_allowed') is not True or op.get('status') != 'FROZEN_FOR_ACCEPTANCE':
                raise ValueError('Every request must independently enable the frozen physical policy')
            if (type(op.get('algorithm')) is not int or op['algorithm'] not in (0, 1)
                    or op['algorithm'] != row.get('algorithm')
                    or op.get('algorithm_name') != ('Normal' if op['algorithm'] == 0 else 'Newton')
                    or op.get('voltage_policy_pu') != policy.get('voltage_limits_pu')
                    or op.get('frequency_hz') != policy.get('frequency_hz')):
                raise ValueError('Request and frozen runtime policies disagree')
    return rows


def certificate_valid(cert, request, point_id):
    if not isinstance(cert, dict):
        return False
    checks = cert.get('checks', {})
    return (cert.get('evidence_kind') == 'PHYSICAL'
            and cert.get('point_id') == point_id
            and cert.get('request_sha256') == request['request_sha256']
            and cert.get('reliable') is True
            and cert.get('status') in ('PASS', 'FAIL')
            and isinstance(checks, dict)
            and all(checks.get(k) == 'PASS' for k in RELIABILITY)
            and all(checks.get(k) in ('PASS', 'FAIL') for k in CONSTRAINTS)
            and (all(checks.get(k) == 'PASS' for k in CONSTRAINTS)
                 == (cert['status'] == 'PASS')))


def execute_worker(project, run_dir, freeze, rows, *, runtime_factory=None):
    """Worker body. Test injection is in-process only, absent from the CLI."""
    if runtime_factory is None and freeze.get('physical_execution_allowed') is not True:
        raise ValueError('Worker body cannot enable a closed physical freeze')
    run = Path(run_dir)
    journal = DurableJournal(run / 'events.jsonl')
    limits = freeze['budget']
    budget = AcceptanceBudget(max_requests=limits['max_requests'],
                              max_entries=limits['max_entries'], seconds=limits['seconds'],
                              emit=journal)
    runtime = None
    completed = 0
    reason = None
    current = None
    injected = runtime_factory is not None
    try:
        if runtime_factory is None:
            if freeze['physical_config'].get('runtime_variant', 'AUTHOR_V1') == 'PHASE_AVAILABLE_V2':
                from .nature123_phase_available_physical import Nature123PhaseAvailablePhysicalRuntime
                runtime_factory = Nature123PhaseAvailablePhysicalRuntime
            else:
                from .nature123_clpu_physical import Nature123CLPUPhysicalRuntime
                runtime_factory = Nature123CLPUPhysicalRuntime
        journal({'event': 'worker_started', 'pid': os.getpid(), 'ppid': os.getppid(),
                 'executable': sys.executable, 'test_runtime_injected':
                 injected})
        runtime = runtime_factory(project, budget=budget,
                                  physical_config=freeze['physical_config'], enable_physical=True)
        for row in rows:
            current = 'point_%02d' % row['order']
            request = row['request']
            journal({'event': 'request_started', 'point_id': current,
                     'request_sha256': request['request_sha256']})
            # The backend owns begin_request and numerical-entry reservations.
            cert = runtime.evaluate(request, point_id=current)
            write_json(run / 'points' / (current + '.json'), cert)
            valid = certificate_valid(cert, request, current)
            journal({'event': 'request_completed', 'point_id': current,
                     'request_sha256': request['request_sha256'],
                     'status': cert.get('status'), 'reliable_certificate': valid,
                     'result_sha256': sha(run / 'points' / (current + '.json'))})
            completed += 1
            if not valid:
                reason = 'UNKNOWN_RELIABILITY_STOP'
                break
            current = None
    except BaseException as exc:
        reason = type(exc).__name__ + ': ' + str(exc)
        error = {'event': 'worker_exception', 'point_id': current, 'reason': reason,
                 'traceback': traceback.format_exc(), 'budget': budget.snapshot()}
        write_json(run / 'worker_exception.json', error)
        journal(error)
    finally:
        write_json(run / 'worker_final.json', {
            'status': 'RETURNED_ALL_REQUESTS' if completed == len(rows) and reason is None else 'STOPPED',
            'reason': reason, 'completed_results': completed, 'budget': budget.snapshot(),
            'actual_worker_pid': os.getpid(), 'automatic_retries': 0,
            'evidence_kind': 'SIMULATED' if injected else 'PHYSICAL',
            'synthetic_test_only': injected,
        })
    return 0 if completed == len(rows) and reason is None else 2


class WindowsJob:
    """Job-wide committed-memory hard limit and descendant cleanup on close."""
    def __init__(self, memory_bytes):
        if os.name != 'nt':
            raise RuntimeError('The frozen supervisor requires Windows Job Objects')
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        ULONG_PTR = ctypes.c_size_t
        class BASIC(ctypes.Structure):
            _fields_ = [('PerProcessUserTimeLimit', ctypes.c_int64),
                        ('PerJobUserTimeLimit', ctypes.c_int64), ('LimitFlags', wintypes.DWORD),
                        ('MinimumWorkingSetSize', ULONG_PTR), ('MaximumWorkingSetSize', ULONG_PTR),
                        ('ActiveProcessLimit', wintypes.DWORD), ('Affinity', ULONG_PTR),
                        ('PriorityClass', wintypes.DWORD), ('SchedulingClass', wintypes.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in ('ReadOperationCount', 'WriteOperationCount',
                        'OtherOperationCount', 'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]
        class EXTENDED(ctypes.Structure):
            _fields_ = [('BasicLimitInformation', BASIC), ('IoInfo', IO),
                        ('ProcessMemoryLimit', ULONG_PTR), ('JobMemoryLimit', ULONG_PTR),
                        ('PeakProcessMemoryUsed', ULONG_PTR), ('PeakJobMemoryUsed', ULONG_PTR)]
        self.extended_type = EXTENDED
        k = self.kernel
        k.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
        k.CreateJobObjectW.restype = wintypes.HANDLE
        k.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
        k.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        k.QueryInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                               wintypes.DWORD, ctypes.c_void_p)
        k.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
        k.IsProcessInJob.argtypes = (wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL))
        k.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        k.OpenProcess.restype = wintypes.HANDLE
        k.CloseHandle.argtypes = (wintypes.HANDLE,)
        self.handle = k.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        info = EXTENDED()
        # PROCESS_MEMORY | JOB_MEMORY | DIE_ON_UNHANDLED_EXCEPTION | KILL_ON_JOB_CLOSE
        info.BasicLimitInformation.LimitFlags = 0x100 | 0x200 | 0x400 | 0x2000
        info.ProcessMemoryLimit = info.JobMemoryLimit = memory_bytes
        if not k.SetInformationJobObject(self.handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())

    def assign(self, process):
        if not self.kernel.AssignProcessToJobObject(self.handle, wintypes.HANDLE(int(process._handle))):
            raise ctypes.WinError(ctypes.get_last_error())

    def contains_pid(self, pid):
        handle = self.kernel.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            answer = wintypes.BOOL()
            if not self.kernel.IsProcessInJob(handle, self.handle, ctypes.byref(answer)):
                raise ctypes.WinError(ctypes.get_last_error())
            return bool(answer.value)
        finally:
            self.kernel.CloseHandle(handle)

    def peak_bytes(self):
        info = self.extended_type()
        if not self.kernel.QueryInformationJobObject(self.handle, 9, ctypes.byref(info), ctypes.sizeof(info), None):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(info.PeakJobMemoryUsed)

    def terminate(self):
        if not self.kernel.TerminateJobObject(self.handle, 3):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def resume_initial_thread(process):
    """Resume only the sole suspended initial thread of our own newly created PID.

    Popen closes the primary-thread handle. Documented Toolhelp/OpenThread APIs
    reacquire it without starting the venv launcher before Job assignment.
    """
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    class THREADENTRY32(ctypes.Structure):
        _fields_ = [('dwSize', wintypes.DWORD), ('cntUsage', wintypes.DWORD),
                    ('th32ThreadID', wintypes.DWORD), ('th32OwnerProcessID', wintypes.DWORD),
                    ('tpBasePri', wintypes.LONG), ('tpDeltaPri', wintypes.LONG), ('dwFlags', wintypes.DWORD)]
    kernel.CreateToolhelp32Snapshot.argtypes = (wintypes.DWORD, wintypes.DWORD)
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.Thread32First.argtypes = (wintypes.HANDLE, ctypes.POINTER(THREADENTRY32))
    kernel.Thread32Next.argtypes = (wintypes.HANDLE, ctypes.POINTER(THREADENTRY32))
    kernel.OpenThread.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenThread.restype = wintypes.HANDLE
    kernel.ResumeThread.argtypes = (wintypes.HANDLE,)
    kernel.ResumeThread.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    snapshot = kernel.CreateToolhelp32Snapshot(4, 0)
    if snapshot == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    own_threads = []
    try:
        entry = THREADENTRY32()
        entry.dwSize = ctypes.sizeof(entry)
        okay = kernel.Thread32First(snapshot, ctypes.byref(entry))
        while okay:
            if entry.th32OwnerProcessID == process.pid:
                own_threads.append(entry.th32ThreadID)
            okay = kernel.Thread32Next(snapshot, ctypes.byref(entry))
    finally:
        kernel.CloseHandle(snapshot)
    if len(own_threads) != 1:
        raise RuntimeError('Suspended child must have exactly one initial thread')
    thread = kernel.OpenThread(2, False, own_threads[0])
    if not thread:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if kernel.ResumeThread(thread) != 1:
            raise RuntimeError('Unexpected initial-thread suspend count')
    finally:
        kernel.CloseHandle(thread)


def supervise(command, *, cwd, run_dir, seconds, memory_bytes, poll_seconds=.05,
              release_file=None, identity_file=None):
    """One launch, no retry; the job encloses the venv launcher and descendants."""
    run = Path(run_dir)
    started = time.monotonic()
    job = WindowsJob(memory_bytes)
    process = None
    peak = 0
    reason = None
    try:
        with (run / 'worker.stdout.log').open('wb') as out, (run / 'worker.stderr.log').open('wb') as err:
            process = subprocess.Popen(command, cwd=cwd, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                       creationflags=subprocess.CREATE_NO_WINDOW | 0x00000004)  # CREATE_SUSPENDED
            job.assign(process)
            if not job.contains_pid(process.pid):
                raise RuntimeError('Launcher not in the assigned Job')
            resume_initial_thread(process)
            write_json(run / 'supervisor_start.json', {
                'launcher_pid': process.pid, 'parent_pid': os.getpid(), 'command': command,
                'wall_seconds': seconds, 'memory_bytes': memory_bytes, 'job_assigned': True,
                'memory_measure': 'Job-wide committed bytes, all worker descendants',
                'automatic_retries': 0, 'process_launches': 1, 'suspended_assignment': True})
            if identity_file is not None:
                while not Path(identity_file).exists():
                    if time.monotonic() - started >= seconds:
                        raise TimeoutError('Worker identity handshake exceeded wall budget')
                    if process.poll() is not None:
                        raise RuntimeError('Worker exited before identity handshake')
                    time.sleep(poll_seconds)
                identity = read_json(identity_file)
                if type(identity.get('pid')) is not int or not job.contains_pid(identity['pid']):
                    raise RuntimeError('Actual Python worker not in this supervisor Job')
                write_json(run / 'worker_job_membership.json', {
                    **identity, 'actual_worker_in_supervisor_job': True,
                    'checked_before_release': True, 'launcher_pid': process.pid})
            if release_file is not None:
                write_json(release_file, {'released': True, 'parent_pid': os.getpid(),
                                          'launcher_pid': process.pid})
            while process.poll() is None:
                elapsed = time.monotonic() - started
                peak = max(peak, job.peak_bytes())
                if elapsed >= seconds:
                    reason = 'WALL_LIMIT'
                    job.terminate()
                    break
                if peak > memory_bytes:
                    reason = 'MEMORY_LIMIT'
                    job.terminate()
                    break
                time.sleep(min(poll_seconds, max(.001, seconds - elapsed)))
            process.wait(timeout=5)
            peak = max(peak, job.peak_bytes())
    except BaseException as exc:
        reason = 'SUPERVISOR_ERROR: ' + repr(exc)
        if process is not None and process.poll() is None:
            try:
                job.terminate()
            finally:
                process.kill()
                process.wait(timeout=5)
    finally:
        job.close()
    result = {'termination_reason': reason, 'returncode': None if process is None else process.returncode,
              'wall_seconds_observed': time.monotonic() - started, 'job_peak_committed_bytes': peak,
              'process_launches': int(process is not None), 'automatic_retries': 0,
              'hard_memory_scope': 'JOB_PROCESS_TREE', 'wall_scope': 'LAUNCH_TO_WORKER_EXIT'}
    if reason is None and process is not None and process.returncode != 0:
        result['termination_reason'] = 'WORKER_NONZERO_RESOURCE_OR_BACKEND_REASON_REQUIRES_LOG_AUDIT'
    write_json(run / 'supervisor_final.json', result)
    return result


def read_journal(path):
    events, errors = [], []
    if not Path(path).exists():
        return events, ['Journal missing']
    for i, line in enumerate(Path(path).read_bytes().decode('utf8', errors='replace').splitlines()):
        try:
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError('Object event required')
            events.append(event)
        except (ValueError, TypeError):
            errors.append('Invalid journal line ' + str(i + 1))
    return events, errors


def audit_charges(events, rows):
    """Reject duplicate reservations/completions rather than silently deduplicating."""
    expected = {'point_%02d' % r['order']: r['request']['request_sha256'] for r in rows}
    requests, charges, starts, completions, errors = {}, {}, {}, {}, []
    for event in events:
        kind = event.get('event')
        if kind in ('request_started', 'request_completed'):
            point = event.get('point_id')
            dest = starts if kind == 'request_started' else completions
            if point not in expected or event.get('request_sha256') != expected.get(point) or point in dest:
                errors.append('Duplicate/unbound ' + kind)
            dest[point] = event
        elif kind == 'request_reserved':
            point = event.get('request_id')
            if (point not in expected or event.get('request_sha256') != expected.get(point)
                    or point in requests or event.get('request_number') != len(requests) + 1):
                errors.append('Duplicate/unbound/out-of-order request reservation')
            requests[point] = event
        elif kind == 'numerical_entries_reserved':
            token, count, completion = event.get('token'), event.get('count'), event.get('completion')
            point = event.get('request_id')
            if (type(token) is not int or token < 0 or type(count) is not int or count < 1
                    or (point is not None and point not in requests)):
                errors.append('Malformed or unbound numerical reservation')
                continue
            if completion == 'NOT_REPORTED':
                if token in charges or token != len(charges):
                    errors.append('Duplicate/out-of-order numerical reservation')
                else:
                    charges[token] = event
            elif completion in ('API_RETURNED_NOT_CONVERGENCE', 'UNKNOWN_NO_REFUND'):
                original = charges.get(token)
                if (original is None or original['completion'] != 'NOT_REPORTED'
                        or any(original.get(k) != event.get(k) for k in
                               ('count', 'request_id', 'operation', 'phase'))):
                    errors.append('Unbound/duplicate numerical completion')
                else:
                    charges[token] = event
            else:
                errors.append('Unknown numerical completion status')
    total = sum(e['count'] for e in charges.values())
    if len(requests) > 16 or total > 256:
        errors.append('Frozen numerical/request maximum exceeded')
    for point in completions:
        entries = [e for e in charges.values() if e['request_id'] == point]
        if point not in requests or point not in starts or not entries:
            errors.append('Completed point lacks a bound start/request/numerical entry')
        elif any(e['completion'] != 'API_RETURNED_NOT_CONVERGENCE' for e in entries):
            errors.append('Completed point has an unfinished/failed numerical entry')
    return requests, charges, starts, completions, errors


def reconcile(run_dir, rows, supervisor_result):
    """Independent saved-file binding/status review; never turns interruption into FAIL."""
    run = Path(run_dir)
    events, journal_errors = read_journal(run / 'events.jsonl')
    requests, charges, started, completed, charge_errors = audit_charges(events, rows)
    journal_errors += charge_errors
    results = []
    for row in rows:
        point_id = 'point_%02d' % row['order']
        path = run / 'points' / (point_id + '.json')
        status = 'UNKNOWN' if point_id in started else 'NOT_RUN'
        explanation = 'Started without a bound reliable result' if point_id in started else 'No request start recorded'
        try:
            cert = read_json(path) if path.exists() else None
            record = completed.get(point_id, {})
            if (not journal_errors and path.exists() and record.get('result_sha256') == sha(path)
                    and certificate_valid(cert, row['request'], point_id)):
                status = cert['status']
                explanation = 'Saved result identity and all eight status checks validated'
            elif path.exists():
                status, explanation = 'UNKNOWN', 'Unreliable, incomplete, or unbound saved certificate'
        except (ValueError, OSError, TypeError) as exc:
            status, explanation = 'UNKNOWN', 'Unreadable result: ' + repr(exc)
        results.append({'order': row['order'], 'point_id': point_id, 'label': row['label'],
                        'algorithm': row['algorithm'], 'request_sha256': row['request']['request_sha256'],
                        'status': status, 'explanation': explanation})
    statuses = [r['status'] for r in results]
    synthetic = any(e.get('test_runtime_injected') is True for e in events if e.get('event') == 'worker_started')
    reliable_all = (all(s in ('PASS', 'FAIL') for s in statuses)
                    and not journal_errors and supervisor_result.get('returncode') == 0
                    and supervisor_result.get('termination_reason') is None)
    result = {'status': ('SYNTHETIC_TEST_CLASSIFIED' if synthetic else 'ALL_REQUESTS_RELIABLY_CLASSIFIED')
                         if reliable_all else 'INCOMPLETE_OR_UNKNOWN',
              'synthetic_test_only': synthetic, 'evidence_kind': 'SIMULATED' if synthetic else 'PHYSICAL',
              'physical_model_acceptance': 'NOT_APPLICABLE_SYNTHETIC' if synthetic else 'REQUIRES_CROSS_ALGORITHM_AND_RAW_AUDIT',
              'requests': results, 'counts': {s: statuses.count(s) for s in ('PASS', 'FAIL', 'UNKNOWN', 'NOT_RUN')},
              'journal_errors': journal_errors,
              'reserved_entries': sum(e['count'] for e in charges.values()),
              'unreported_entries': sum(e['count'] for e in charges.values() if e['completion'] == 'NOT_REPORTED'),
              'reserved_requests': len(requests),
              'automatic_retries': 0, 'training': 0}
    write_json(run / 'result.json', result)
    return result


def prepare_run(project, freeze_path, *, execute=False):
    project = Path(project).resolve()
    freeze_path = Path(freeze_path).resolve()
    freeze = read_json(freeze_path)
    rows = validate_freeze(project, freeze, execute=execute)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run = project / 'runs' / (stamp + ('_nc123_entry_acceptance' if execute else '_nc123_runtime_dryrun'))
    run.mkdir(parents=True)
    (run / 'freeze.json').write_bytes(freeze_path.read_bytes())
    (run / 'schedule.json').write_bytes(inside(project, freeze['schedule']['path']).read_bytes())
    for relative in freeze['source_sha256']:
        dest = run / 'source' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(inside(project, relative).read_bytes())
        if sha(dest) != freeze['source_sha256'][relative]:
            raise ValueError('Snapshot changed while copying ' + relative)
    packages = sorted((d.metadata.get('Name', 'UNKNOWN'), d.version) for d in importlib.metadata.distributions())
    write_json(run / 'environment.json', {'executable': sys.executable, 'python': sys.version,
               'platform': sys.platform, 'dependencies': packages, 'seed': freeze.get('seed'),
               'freeze_sha256': sha(freeze_path), 'mode': 'EXECUTE' if execute else 'DRY_RUN'})
    if not execute:
        write_json(run / 'result.json', {'status': 'PASS_DRY_RUN_ONLY', 'physical_requests': 0,
                   'numerical_entries': 0, 'training': 0, 'requests_retained': len(rows),
                   'physical_execution_allowed': freeze['physical_execution_allowed']})
    return run, freeze, rows


def seal_run(run):
    run = Path(run)
    write_json(run / 'completion_manifest.json', {
        str(p.relative_to(run)).replace('\\', '/'): sha(p)
        for p in sorted(run.rglob('*')) if p.is_file() and p.name != 'completion_manifest.json'})


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('freeze', type=Path)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--run-dir', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--release-file', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if not args.execute or args.run_dir is None or args.release_file is None:
            parser.error('Worker requires explicit execution and parent handshake')
        write_json(args.run_dir / 'worker_identity.json', {
            'pid': os.getpid(), 'ppid': os.getppid(), 'executable': sys.executable})
        # Waiting does not import/construct the physical backend.
        deadline = time.monotonic() + 10
        while not args.release_file.exists():
            if time.monotonic() >= deadline:
                raise RuntimeError('No supervisor release; physical engine remains uninitialized')
            time.sleep(.01)
        freeze = read_json(args.freeze)
        rows = validate_freeze(args.project, freeze, execute=True)
        return execute_worker(args.project, args.run_dir, freeze, rows)
    expected = (args.project / '.venv/Scripts/python.exe').resolve()
    if Path(sys.executable).resolve() != expected:
        raise ValueError('Use the project .venv interpreter')
    run, freeze, rows = prepare_run(args.project, args.freeze, execute=args.execute)
    if args.execute:
        release = run / 'supervisor_release.json'
        command = [str(expected), str(run / 'source/scripts/run_nature123_acceptance.py'),
                   str(run / 'freeze.json'), '--project', str(args.project.resolve()),
                   '--execute', '--worker', '--run-dir', str(run), '--release-file', str(release)]
        limits = freeze['budget']
        observed = supervise(command, cwd=run, run_dir=run, seconds=limits['seconds'],
                             memory_bytes=limits['memory_bytes'], poll_seconds=limits['poll_seconds'],
                             release_file=release, identity_file=run / 'worker_identity.json')
        result = reconcile(run, rows, observed)
    else:
        result = read_json(run / 'result.json')
    seal_run(run)
    print(json.dumps({'run': str(run), **result}, ensure_ascii=False))
    return 0 if not args.execute or result['status'] == 'ALL_REQUESTS_RELIABLY_CLASSIFIED' else 2
