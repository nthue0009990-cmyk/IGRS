"""Frozen Chen-only c02/c07 cold interface check. No retries or other arms."""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
BASE = ROOT / 'runs/20261105T024027_869785Z_clpu_direct_beam'
BASE_SHA = '2989962bf0e870540c9fb37a07c8e0d31c7cbf951c0eb20e8177127a8a603932'
PROTOCOL = ROOT / 'docs/proposals/ijepes_joint_action_comparator_v1.md'
CASES = [(2, 'N2__line.l10__line.l24', 6), (7, 'N2__line.l12__line.l24', 3)]
INPUTS = ('config.json', 'condition.json', 'contexts.json', 'selected_events.json',
          'selected_scenarios.json', 'schedule.json', 'author_static_metadata.json',
          'load_reference.json', 'command_inventory.json', 'physical_config.json',
          'physical_scope.json', 'source_inputs/base_load_properties.json',
          'source_inputs/metadata.json', 'source_pointer.json', 'source_provenance.json',
          'dependencies.json', 'prepared.json', 'metadata.json')
GiB = 2 ** 30
PHYSICAL_LIMIT, LOGICAL_LIMIT, WALL_S = 512, 512, 240
RSS_LIMIT = 4 * GiB
DISK_FLOOR = 14 * GiB
EXTRA_SPACE = 4 * GiB
EXPECTED_PYTHON = ROOT / '.venv/Scripts/python.exe'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as out:
        json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False)


def sha(path):
    with Path(path).open('rb') as src:
        return hashlib.file_digest(src, 'sha256').hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def env_check():
    require(os.name == 'nt' and Path(sys.executable).resolve() == EXPECTED_PYTHON,
            'Windows project .venv interpreter required')
    from nature34_experiment_support import dependencies
    require(dependencies() == read(BASE / 'dependencies.json'), 'Project dependencies changed')


def memory_state():
    class Status(ctypes.Structure):
        _fields_ = [('length',wintypes.DWORD),('load',wintypes.DWORD)] + [
            (n,ctypes.c_ulonglong) for n in ('total_phys','avail_phys','total_page',
             'avail_page','total_virtual','avail_virtual','avail_extended')]
    data = Status()
    data.length = ctypes.sizeof(data)
    if not ctypes.WinDLL('kernel32',use_last_error=True).GlobalMemoryStatusEx(ctypes.byref(data)):
        raise ctypes.WinError(ctypes.get_last_error())
    return dict(total_physical_bytes=data.total_phys, available_physical_bytes=data.avail_phys)


def preflight_evidence(summary):
    gate = read(summary)
    require(gate.get('status') == 'PASS' and gate.get('contexts') == [2, 7],
            'A passing zero-flow gate on both requested contexts is required')
    require(gate.get('graph_run') and gate.get('window_audit') and gate.get('hashes'),
            'Missing graph/window gate bindings')
    paths = {rel(summary): sha(summary)}
    for name, digest in gate['hashes'].items():
        local = ROOT / name
        require(sha(local) == digest, f'Preflight evidence changed: {name}')
        paths[rel(local)] = digest
    return paths


def proxy_source_files():
    from run_two_stage_clpu_development_check import SOURCE, SOURCE_SHA
    from dnr.nature34_two_stage_network_mapping import AUTHOR_SETUP
    seal = SOURCE/'completion_manifest.json'
    require(sha(seal) == SOURCE_SHA, 'Original static mapping seal changed')
    manifest = {n.replace(chr(92), '/'): v for n, v in read(seal).items()}
    names = {f'{n}.json' for n in ('base_load_properties','metadata','author_static_metadata',
             'command_inventory','condition','config','dependencies','physical_config')}
    for n in ('DSS_Initialize.py','DSS_CircuitSetup.py'):
        names.add('episodes/000/physical/work/Environments/DSSdirect_34bus_loadandswitching/' + n)
    names.update(['source/src/dnr/nature34_clpu_physical.py','source/src/dnr/nature34_operating_physical.py'])
    files = {rel(seal), rel(AUTHOR_SETUP), 'scripts/run_two_stage_clpu_development_check.py'}
    for n in names:
        require(sha(SOURCE/n) == manifest[n], 'Bound proxy input changed: ' + n)
        files.add(rel(SOURCE/n))
    for row in read(SOURCE/'command_inventory.json')['files'].values():
        require(sha(row['path']) == row['sha256'], 'Bound DSS input changed')
        files.add(rel(row['path']))
    for subdir in ('network', 'devices'):
        directory = ROOT/'data/raw/literature/two_stage_static_mapping_20261031'/subdir
        files.update(rel(p) for p in directory.iterdir() if p.is_file())
    return files


def prepare(summary):
    env_check()
    require(sha(BASE / 'completion_manifest.json') == BASE_SHA, 'Base seal changed')
    manifest = {n.replace(chr(92), '/'): v for n, v in read(BASE / 'completion_manifest.json').items()}
    gate_files = preflight_evidence(summary)
    base_hashes = {}
    for name in INPUTS:
        digest = sha(BASE / name)
        require(digest == manifest[name], f'Changed archived input: {name}')
        base_hashes[name] = digest
    inherited = read(BASE / 'prepared.json')['source_files']
    for name in inherited:
        require(sha(ROOT / name) == manifest['source/' + name.replace(chr(92), '/')],
                f'Changed archived implementation: {name}')
    # Capture the full scientific module tree including proxy model dependencies.
    sources = {rel(p) for p in (ROOT / 'src/dnr').glob('*.py')}
    sources.update(inherited)
    sources.update(proxy_source_files())
    sources.update(rel(p) for p in (ROOT / 'scripts').glob('*chen*.py'))
    sources.update(rel(p) for p in (ROOT / 'tests').glob('*chen*.py'))
    sources.update([rel(Path(__file__)), rel(PROTOCOL), 'scripts/nature34_experiment_support.py'])
    require((ROOT / 'src/dnr/chen_joint_rolling.py').is_file() and
            (ROOT / 'src/dnr/chen_joint_domain.py').is_file(), 'Missing scientific solver/adapter')
    forecast = math.ceil(read(BASE / 'resource_budget.json')['forecast_bytes'] * 1024 / 8192)
    free = shutil.disk_usage(ROOT).free
    require(free >= DISK_FLOOR + EXTRA_SPACE + forecast, 'Disk budget unavailable')
    memory = memory_state()
    require(memory['available_physical_bytes'] >= RSS_LIMIT, 'Insufficient free physical memory for worker')
    cfg = dict(seed=0, worker_seconds=WALL_S, max_workers=1, cold_tasks=2, contexts=[2, 7],
               physical_per_task=PHYSICAL_LIMIT, logical_per_task=LOGICAL_LIMIT,
               numerical_entries_per_task=2051, milp_per_task=8, milp_seconds_per_call=20,
               window_slots=5, commit_slots=4, process_memory_limit_bytes=RSS_LIMIT,
               disk_floor_bytes=DISK_FLOOR, preflight_bytes=free, forecast_log_bytes=forecast,
               extra_space_bytes=EXTRA_SPACE, allow_retry=False, fallback=False)
    run = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_chen_two_cold')
    run.mkdir()
    for name in sources:
        target = run / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    for name in INPUTS:
        target = run / 'inputs' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(BASE / name, target)
    schedule = [dict(context_id=c, scenario_id=f, event_id=e, arm='CHEN', seed=0, cell=f'episodes/c{c:02d}')
                for c, f, e in CASES]
    save(run / 'schedule.json', schedule)
    save(run / 'config.json', cfg)
    save(run / 'machine.json', dict(platform=platform.platform(), architecture=platform.machine(),
         cpu=platform.processor(), logical_cpus=os.cpu_count(), memory=memory, venv_python=str(sys.executable),
         base_python=sys._base_executable, version=sys.version,
         venv_exe_sha256=sha(EXPECTED_PYTHON), base_exe_sha256=sha(sys._base_executable)))
    save(run / 'freeze.json', dict(source_run=BASE.name, source_seal=BASE_SHA,
         sources={n:sha(run/'source'/n) for n in sorted(sources)}, inputs=base_hashes,
         gate_files=gate_files, preflight=rel(summary),
         files={n:sha(run/n) for n in ('schedule.json','config.json','machine.json')}))
    save(run / 'prepared.json', dict(freeze_sha256=sha(run/'freeze.json'), physical_queries=0))
    print(str(run), flush=True)


def verify(run):
    require(run.parent == ROOT/'runs' and run.name.endswith('_chen_two_cold'), 'Incorrect target run')
    env_check()
    require(sha(run/'freeze.json') == read(run/'prepared.json')['freeze_sha256'], 'Freeze changed')
    f = read(run/'freeze.json')
    require(sha(BASE/'completion_manifest.json') == f['source_seal'] == BASE_SHA, 'Base seal changed')
    for n,d in f['sources'].items():
        require(sha(ROOT/n) == sha(run/'source'/n) == d, 'Executing source differs: ' + n)
    for n,d in f['inputs'].items():
        require(sha(BASE/n) == sha(run/'inputs'/n) == d, 'Original/frozen input differs: ' + n)
    for n,d in f['files'].items():
        require(sha(run/n) == d, 'Frozen metadata differs: ' + n)
    require(preflight_evidence(ROOT/f['preflight']) == f['gate_files'], 'Gate changed')
    machine = read(run/'machine.json')
    require(sha(EXPECTED_PYTHON) == machine['venv_exe_sha256'] and
            sha(sys._base_executable) == machine['base_exe_sha256'], 'Interpreter changed')


def certify_own_plan(env, actions, deadline, cell):
    from dnr.nature34_clpu_guard_first_control import GuardFirstEvidenceService
    require(len(actions) == env.horizon == 30 and all(type(a) is int for a in actions),
            'Solver output must contain exactly 30 integer commands')
    states = env.plan_states(env.initial, actions)
    require(len(states) == 31, 'Illegal state count')
    require(time.perf_counter() < deadline, 'Task deadline before certification')
    validation = None
    with GuardFirstEvidenceService(BASE, cell, env, read(BASE/'physical_scope.json'),
            mode='shared', physical_budget=PHYSICAL_LIMIT, logical_budget=LOGICAL_LIMIT) as service:
        service.deadline_monotonic = deadline
        # GUARD is the ledger namespace, not a Beam or baseline search.
        with service.view('GUARD') as view:
            initial = view.certify(env.initial, 0)
            require(initial['reliable'] and initial['status'] in ('PASS','FAIL'), 'Initial evidence unreliable')
            if initial['status'] == 'PASS':
                validation = env.validate_plan(env.initial, actions, view)
                require(validation['status'] in ('COMPLETE','FAIL'), 'Graph or reliability mismatch')
    account = service.summary()
    require(account['physical_points'] <= PHYSICAL_LIMIT and account['logical_points'] <= LOGICAL_LIMIT and
            account['actual_numerical_entries'] == 3 + 4 * account['physical_points'], 'Physical cost mismatch')
    complete = validation is not None and validation['status'] == 'COMPLETE'
    if complete:
        require(len(validation['samples']) == 90 and all(x['status']=='PASS' for x in validation['samples']),
                'Missing required samples')
    return dict(status='CERTIFIED_OWN_PLAN' if complete else 'RELIABLE_CONSTRAINT_FAIL',
                fully_certified=complete, energy_kwh=validation['energy_kwh'] if complete else None,
                initial_certificate=initial, validation=validation, account=account)


def worker(run, index):
    spec = read(run/'schedule.json')[index]
    cell = run/spec['cell']
    launch = read(cell/'launch.json')
    # The controller assigns this single PID to a hard-capped job before Go.
    while not (cell/'go.json').is_file() and time.perf_counter() < launch['deadline']:
        time.sleep(.02)
    require(read(cell/'go.json')['worker_pid'] == os.getpid(), 'No controller job binding')
    save(cell/'worker_started.json', dict(pid=os.getpid(), ppid=os.getppid(), python=sys.executable))
    try:
        verify(run)
        from run_nature34_clpu_pilot import make_env
        from dnr.chen_joint_domain import ChenEnv
        from dnr.chen_joint_rolling import own_plan
        env = ChenEnv.from_env(make_env(BASE, spec))
        actions = own_plan(env, BASE, cell, launch['deadline'])
        planner = read(cell/'chen_planner.json')
        require(planner['actions'] == actions and 1<=len(planner['windows'])<=8,
                'Solver actions/window count inconsistent')
        window_receipts = {}
        calls = 0
        for j,window in enumerate(planner['windows'],1):
            name = f'chen_window_{j:02d}.json'
            require(read(cell/name)==window and window['index']==j and window['backend_calls'] in (0,1),
                    'Missing or inconsistent per-window accounting')
            calls += window['backend_calls']
            window_receipts[name]=sha(cell/name)
        require(calls == planner['backend_calls'] and calls <= 8, 'Solver budget mismatch')
        if actions is None:
            result = dict(status='NO_OWN_PLAN', fully_certified=False, energy_kwh=None,
                          account={'physical_points':0, 'logical_points':0, 'actual_numerical_entries':0})
        else:
            save(cell/'own_commands.json', actions)
            result = certify_own_plan(env, actions, launch['deadline'], cell)
        result.update(spec=spec, own_commands=actions, fallback=False, seed=0,
              planner_status=planner['status'], planner_sha256=sha(cell/'chen_planner.json'),
              window_receipts=window_receipts, backend_calls=calls, cross_task_evidence_reuse=False,
              physical_budget_used=result['account']['physical_points'])
        save(cell/'result.json', result)
        # A complete plan is first returnable once its complete evidence is on disk.
        elapsed = time.perf_counter() - launch['start']
        save(cell/'availability.json', dict(first_certified_elapsed_s=elapsed if result['fully_certified'] else None,
             checkpoints={str(t):(result['energy_kwh'] if result['fully_certified'] and elapsed<=t else None)
                          for t in (60,120,240)}))
        if actions is None:
            end = planner['windows'][-1]
            if planner['status'] != 'NO_WINDOW_INCUMBENT' or end.get('solve',{}).get('status') not in (1,2):
                raise RuntimeError('Unusable backend/model/deadline: ' + planner['status'])
    except BaseException:
        save(cell/'worker_error.json', dict(traceback=traceback.format_exc()))
        raise


class WinJob:
    """Single native worker: limit commit, police peak RSS, kill on handle close."""
    def __init__(self):
        class Basic(ctypes.Structure):
            _fields_ = [('process_time',ctypes.c_longlong), ('job_time',ctypes.c_longlong),
                        ('flags',wintypes.DWORD), ('min_ws',ctypes.c_size_t), ('max_ws',ctypes.c_size_t),
                        ('active',wintypes.DWORD), ('affinity',ctypes.c_size_t),
                        ('priority',wintypes.DWORD), ('scheduling',wintypes.DWORD)]
        class Extended(ctypes.Structure):
            _fields_ = [('basic',Basic), ('io',ctypes.c_ulonglong*6)] + [
                (n,ctypes.c_size_t) for n in ('process_limit','job_limit','peak_process','peak_job')]
        self.k = ctypes.WinDLL('kernel32',use_last_error=True)
        self.k.CreateJobObjectW.argtypes = [ctypes.c_void_p,wintypes.LPCWSTR]
        self.k.CreateJobObjectW.restype = wintypes.HANDLE
        self.k.SetInformationJobObject.argtypes = [wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD]
        self.k.AssignProcessToJobObject.argtypes = [wintypes.HANDLE,wintypes.HANDLE]
        self.k.CloseHandle.argtypes = [wintypes.HANDLE]
        self.handle = self.k.CreateJobObjectW(None,None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = Extended()
        limits.basic.flags = 0x2000 | 0x100 | 0x8  # kill on close, process memory, active process count
        limits.basic.active = 1
        limits.process_limit = RSS_LIMIT
        if not self.k.SetInformationJobObject(self.handle,9,ctypes.byref(limits),ctypes.sizeof(limits)):
            err = ctypes.get_last_error()
            self.close()
            raise ctypes.WinError(err)

    def attach(self, proc):
        if not self.k.AssignProcessToJobObject(self.handle,int(proc._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            self.k.CloseHandle(self.handle)
            self.handle = None


def peak_rss(proc):
    class Counters(ctypes.Structure):
        _fields_ = [('cb',wintypes.DWORD), ('faults',wintypes.DWORD)] + [
            (n,ctypes.c_size_t) for n in ('peak_ws','ws','p1','p2','p3','p4','page','peak_page')]
    ps = ctypes.WinDLL('psapi',use_last_error=True)
    ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    value = Counters()
    value.cb = ctypes.sizeof(value)
    if not ps.GetProcessMemoryInfo(int(proc._handle), ctypes.byref(value), value.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return value.peak_ws


def run_one(run, index, spec):
    cell = run / spec['cell']
    cell.mkdir(parents=True, exist_ok=False)
    save(cell/'spec.json',spec)
    runtime = {**os.environ,'__PYVENV_LAUNCHER__':str(EXPECTED_PYTHON),
               'PYTHONDONTWRITEBYTECODE':'1','PYTHONHASHSEED':'0',
               'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    job = WinJob()
    proc = None
    peak = 0
    began = time.perf_counter()
    deadline = began + WALL_S
    failure = None
    try:
        require(memory_state()['available_physical_bytes'] >= RSS_LIMIT,
                'Insufficient free physical memory for cold worker')
        save(cell/'launch.json',dict(start=began, deadline=deadline, controller_pid=os.getpid(),
             utc=datetime.now(timezone.utc).isoformat()))
        with (cell/'stdout.log').open('xb') as out,(cell/'stderr.log').open('xb') as err:
            # Launch the real interpreter once. CPython's Windows venv hook restores project sys.prefix.
            proc = subprocess.Popen([sys._base_executable,'-B','-X','utf8',__file__,'worker',
                str(run),str(index)],cwd=ROOT,env=runtime,stdout=out,stderr=err,
                creationflags=subprocess.CREATE_NO_WINDOW)
            job.attach(proc)
            save(cell/'go.json',dict(worker_pid=proc.pid))
            while True:
                finished = proc.poll() is not None
                peak = max(peak, peak_rss(proc))
                require(peak <= RSS_LIMIT, 'Worker exceeded RSS limit')
                require(shutil.disk_usage(ROOT).free >= DISK_FLOOR, 'Disk reserve exhausted')
                require(time.perf_counter() <= deadline, 'Hard cold wall limit')
                if finished:
                    require(proc.returncode == 0 and (cell/'availability.json').is_file(), 'Cold worker failed')
                    break
                time.sleep(.04)
    except BaseException:
        failure = traceback.format_exc()
    finally:
        job.close()
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
    row = dict(spec=spec, exit_code=None if proc is None else proc.returncode,
               cold_seconds=time.perf_counter()-began, peak_rss_bytes=peak, failure=failure,
               result_sha256=sha(cell/'result.json') if (cell/'result.json').exists() else None)
    save(cell/'cold_exit.json',row)
    return row


def execute(run):
    verify(run)
    cfg = read(run/'config.json')
    require(shutil.disk_usage(ROOT).free >= DISK_FLOOR + EXTRA_SPACE + cfg['forecast_log_bytes'],
            'Storage no longer sufficient')
    schedule = read(run/'schedule.json')
    require([(s['context_id'],s['scenario_id'],s['event_id']) for s in schedule]==CASES, 'Wrong schedule')
    save(run/'started.json',dict(controller_pid=os.getpid(), freeze_sha256=sha(run/'freeze.json'),
                                utc=datetime.now(timezone.utc).isoformat()))
    rows=[]
    try:
        for i,spec in enumerate(schedule):
            row = run_one(run,i,spec)
            rows.append(row)
            print(spec['context_id'], row['exit_code'], row['failure'] or 'FINISHED',flush=True)
            if row['failure']:
                break
    finally:
        save(run/'controller_result.json',dict(exits=rows,not_dispatched=schedule[len(rows):],
             status='COMPLETE_PENDING_INDEPENDENT_AUDIT' if len(rows)==2 and all(
                 not r['failure'] for r in rows) else 'STOPPED_INCOMPLETE'))
    return int(len(rows)!=2 or any(r['failure'] for r in rows))


def main():
    parser=argparse.ArgumentParser()
    commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('prepare').add_argument('--preflight',required=True,type=Path)
    commands.add_parser('execute').add_argument('run',type=Path)
    worker_parser=commands.add_parser('worker')
    worker_parser.add_argument('run',type=Path)
    worker_parser.add_argument('index',type=int,choices=(0,1))
    args=parser.parse_args()
    if args.command=='prepare':
        prepare(args.preflight.resolve())
    elif args.command=='execute':
        return execute(args.run.resolve())
    else:
        worker(args.run.resolve(),args.index)
    return 0


if __name__ == '__main__':
    sys.exit(main())
