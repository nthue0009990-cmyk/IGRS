"""Offline, fake-runtime and fake-worker coverage. No DSS import or solver calls."""
from copy import deepcopy
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dnr import nature123_acceptance_runner as r


def rows():
    result = []
    for i in range(16):
        request = {'condition': {'operating_config': {
            'physical_execution_allowed': True, 'status': 'FROZEN_FOR_ACCEPTANCE', 'algorithm': i % 2,
            'algorithm_name': 'Normal' if i % 2 == 0 else 'Newton',
            'voltage_policy_pu': [.9, 1.1], 'frequency_hz': 60}}, 'synthetic_index': i}
        request['request_sha256'] = r.canonical_sha(request)
        result.append({'order': i, 'algorithm': i % 2, 'label': 'SYNTHETIC_TEST',
                       'status': 'NOT_RUN', 'request': request})
    return result


def freeze():
    return {'schema_version': 'NC123_ACCEPTANCE_V1', 'physical_execution_allowed': False,
            'budget': {'max_requests': 16, 'max_entries': 256, 'seconds': 240,
                       'memory_bytes': 4 * 1024**3, 'poll_seconds': .05},
            'physical_config': {'voltage_limits_pu': [.9, 1.1], 'frequency_hz': 60},
            'readiness_checks': {'OFFLINE': 'PASS'}, 'authorization': {
                'scope': 'ENTRY_ACCEPTANCE_16_REQUESTS', 'record': 'SYNTHETIC_TEST_ONLY'}}


class FakeRuntime:
    """PHYSICAL is deliberately a test envelope; receipts label all data synthetic."""
    unknown_at = None
    fail_at = None

    def __init__(self, project, *, budget, physical_config, enable_physical):
        self.budget = budget

    def evaluate(self, request, point_id):
        self.budget.begin_request(point_id, request['request_sha256'])
        token = self.budget.reserve_entry(operation='SYNTHETIC_NO_DSS', count=1, request_id=point_id)
        self.budget.finish_entry(token, returned=True)
        checks = {k: 'PASS' for k in r.RELIABILITY + r.CONSTRAINTS}
        index = request['synthetic_index']
        status = 'PASS'
        if index == self.fail_at:
            checks['voltage'] = status = 'FAIL'
        if index == self.unknown_at:
            checks['ac_balance'] = status = 'UNKNOWN'
        return {'status': status, 'reliable': status != 'UNKNOWN', 'point_id': point_id,
                'request_sha256': request['request_sha256'], 'evidence_kind': 'PHYSICAL',
                'checks': checks, 'SYNTHETIC_TEST_ONLY_NO_ENGINE': True}


class RunnerTests(unittest.TestCase):
    def setUp(self):
        base = os.environ.get('NC123_RUNNER_TEST_ROOT')
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.root = Path(self.temp.name)

    def tearDown(self):
        evidence_root = os.environ.get('NC123_RUNNER_TEST_ROOT')
        if evidence_root:
            import shutil
            dest = Path(evidence_root) / self.id().split('.')[-1]
            shutil.copytree(self.root, dest)
        self.temp.cleanup()

    def bind_fixture(self):
        frozen = freeze()
        for name in r.REQUIRED_SOURCES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# synthetic fixture; never executable\n', encoding='utf8')
        frozen['source_sha256'] = {name: r.sha(self.root / name) for name in r.REQUIRED_SOURCES}
        r.write_json(self.root / 'schedule.json', rows())
        frozen['schedule'] = {'path': 'schedule.json', 'sha256': r.sha(self.root / 'schedule.json')}
        return frozen

    def test_dryrun_gate_and_exact_inputs(self):
        frozen = self.bind_fixture()
        self.assertEqual(len(r.validate_freeze(self.root, frozen)), 16)
        with self.assertRaisesRegex(ValueError, 'not enabled'):
            r.validate_freeze(self.root, frozen, execute=True)
        frozen['physical_execution_allowed'] = True
        self.assertEqual(len(r.validate_freeze(self.root, frozen, execute=True)), 16)
        (self.root / r.REQUIRED_SOURCES[0]).write_text('changed', encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'source mismatch'):
            r.validate_freeze(self.root, frozen)
        with self.assertRaisesRegex(ValueError, 'closed physical freeze'):
            r.execute_worker(self.root, self.root, freeze(), rows())

    def test_request_proposal_and_budget_increase_rejected(self):
        frozen = self.bind_fixture()
        frozen['physical_execution_allowed'] = True
        proposal = rows()
        proposal[0]['request']['condition']['operating_config']['physical_execution_allowed'] = False
        request = proposal[0]['request']
        request['request_sha256'] = r.canonical_sha({k: v for k, v in request.items() if k != 'request_sha256'})
        r.write_json(self.root / 'schedule.json', proposal)
        frozen['schedule']['sha256'] = r.sha(self.root / 'schedule.json')
        with self.assertRaisesRegex(ValueError, 'independently enable'):
            r.validate_freeze(self.root, frozen, execute=True)
        for key, value in [('max_requests', 17), ('max_entries', 257), ('seconds', 241),
                           ('memory_bytes', 4 * 1024**3 + 1), ('poll_seconds', .11)]:
            bad = deepcopy(frozen)
            bad['budget'][key] = value
            with self.assertRaisesRegex(ValueError, 'enlarged'):
                r.validate_freeze(self.root, bad)

    def test_policy_and_order_tamper_rejected(self):
        frozen = self.bind_fixture()
        frozen['physical_execution_allowed'] = True
        frozen['physical_config']['frequency_hz'] = 50
        with self.assertRaisesRegex(ValueError, 'policies disagree'):
            r.validate_freeze(self.root, frozen, execute=True)
        with self.assertRaises(ValueError):
            r.inside(self.root, '../outside')

    def test_reliable_failure_retained_and_worker_continues(self):
        class FailureRuntime(FakeRuntime):
            fail_at = 3
        rc = r.execute_worker(self.root, self.root, freeze(), rows(), runtime_factory=FailureRuntime)
        self.assertEqual(rc, 0)
        result = r.reconcile(self.root, rows(), {'returncode': 0, 'termination_reason': None})
        self.assertEqual(result['status'], 'SYNTHETIC_TEST_CLASSIFIED')
        self.assertTrue(result['synthetic_test_only'])
        self.assertEqual(result['counts'], {'PASS': 15, 'FAIL': 1, 'UNKNOWN': 0, 'NOT_RUN': 0})
        self.assertEqual(result['reserved_entries'], 16)
        self.assertTrue(r.read_json(self.root / 'points/point_00.json')['SYNTHETIC_TEST_ONLY_NO_ENGINE'])

    def test_unknown_stops_but_retains_all_rows(self):
        class UnknownRuntime(FakeRuntime):
            unknown_at = 2
        rc = r.execute_worker(self.root, self.root, freeze(), rows(), runtime_factory=UnknownRuntime)
        self.assertEqual(rc, 2)
        result = r.reconcile(self.root, rows(), {'returncode': 2, 'termination_reason': None})
        self.assertEqual(result['counts'], {'PASS': 2, 'FAIL': 0, 'UNKNOWN': 1, 'NOT_RUN': 13})
        self.assertEqual(result['reserved_requests'], 3)

    def test_partial_write_and_duplicate_journal_fail_closed(self):
        r.execute_worker(self.root, self.root, freeze(), rows(), runtime_factory=FakeRuntime)
        (self.root / 'points/point_00.json').write_text('{partial', encoding='utf8')
        result = r.reconcile(self.root, rows(), {'returncode': 0, 'termination_reason': None})
        self.assertEqual(result['requests'][0]['status'], 'UNKNOWN')
        events, _ = r.read_journal(self.root / 'events.jsonl')
        reservation = next(e for e in events if e['event'] == 'request_reserved')
        r.DurableJournal(self.root / 'events.jsonl')(reservation)
        result = r.reconcile(self.root, rows(), {'returncode': 0, 'termination_reason': None})
        self.assertTrue(result['journal_errors'])
        self.assertEqual(result['counts']['UNKNOWN'], 16)

    def test_killed_call_has_no_refund_and_unstarted_rows_remain(self):
        journal = r.DurableJournal(self.root / 'events.jsonl')
        req = rows()[0]['request']
        journal({'event': 'request_started', 'point_id': 'point_00', 'request_sha256': req['request_sha256']})
        budget = r.AcceptanceBudget(emit=journal)
        budget.begin_request('point_00', req['request_sha256'])
        budget.reserve_entry(operation='SYNTHETIC_HANG', count=2, request_id='point_00')
        with (self.root / 'events.jsonl').open('a', encoding='utf8') as f:
            f.write('{truncated')
        result = r.reconcile(self.root, rows(), {'returncode': 3, 'termination_reason': 'WALL_LIMIT'})
        self.assertEqual(result['counts'], {'PASS': 0, 'FAIL': 0, 'UNKNOWN': 1, 'NOT_RUN': 15})
        self.assertEqual(result['unreported_entries'], 2)
        self.assertEqual(result['reserved_entries'], 2)

    @unittest.skipUnless(os.name == 'nt', 'Windows Job Object implementation')
    def test_real_fake_worker_wall_termination(self):
        # An actual descendant Python (beyond any venv shim) must die with the Job.
        release = self.root / 'release.json'
        identity = self.root / 'worker_identity.json'
        child_identity = self.root / 'descendant_pid.txt'
        child_script = self.root / 'child.py'
        child_script.write_text('import os,sys,time\nfrom pathlib import Path\n'
                                'Path(sys.argv[1]).write_text(str(os.getpid()))\ntime.sleep(30)\n', encoding='utf8')
        script = self.root / 'fake_wall_worker.py'
        script.write_text('import os,sys,time,json,subprocess\nfrom pathlib import Path\n'
                          'subprocess.Popen([sys.executable,sys.argv[3],sys.argv[4]], creationflags=subprocess.CREATE_NO_WINDOW)\n'
                          'while not Path(sys.argv[4]).exists():time.sleep(.01)\n'
                          "Path(sys.argv[2]).write_text(json.dumps({'pid':os.getpid()}))\n"
                          'while not Path(sys.argv[1]).exists():time.sleep(.01)\ntime.sleep(30)\n', encoding='utf8')
        command = [sys.executable, str(script), str(release), str(identity), str(child_script), str(child_identity)]
        result = r.supervise(command, cwd=self.root, run_dir=self.root,
                             seconds=.8, memory_bytes=256 * 1024**2, poll_seconds=.01,
                             release_file=release, identity_file=identity)
        self.assertEqual(result['termination_reason'], 'WALL_LIMIT')
        self.assertEqual(result['process_launches'], 1)
        self.assertLess(result['wall_seconds_observed'], 3)
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        pid = int(child_identity.read_text())
        handle = kernel.OpenProcess(0x100000, False, pid)
        if handle:
            try:
                self.assertEqual(kernel.WaitForSingleObject(handle, 2000), 0)
            finally:
                kernel.CloseHandle(handle)
        else:
            self.assertEqual(ctypes.get_last_error(), 87)  # terminated and no longer present
        r.write_json(self.root / 'descendant_termination_proof.json', {
            'synthetic_only': True, 'pid': pid, 'terminated_or_absent_after_job_kill': True})

    @unittest.skipUnless(os.name == 'nt', 'Windows Job Object implementation')
    def test_real_fake_worker_memory_hard_cap_and_release(self):
        release = self.root / 'release.json'
        identity = self.root / 'worker_identity.json'
        script = self.root / 'fake_memory_worker.py'
        script.write_text("from pathlib import Path\nimport sys,time,os,json\n"
                          "Path(sys.argv[2]).write_text(json.dumps({'pid':os.getpid()}))\n"
                          "while not Path(sys.argv[1]).exists():time.sleep(.01)\n"
                          "try:\n x=bytearray(512*1024**2)\n"
                          "except MemoryError:\n print('EXPECTED_MEMORY_ERROR');sys.exit(7)\n"
                          "sys.exit(0)\n", encoding='utf8')
        result = r.supervise([sys.executable, str(script), str(release), str(identity)], cwd=self.root, run_dir=self.root,
                             seconds=5, memory_bytes=128 * 1024**2, poll_seconds=.01,
                             release_file=release, identity_file=identity)
        self.assertEqual(result['returncode'], 7)
        self.assertLessEqual(result['job_peak_committed_bytes'], 128 * 1024**2)
        self.assertIn('EXPECTED_MEMORY_ERROR', (self.root / 'worker.stdout.log').read_text())
        self.assertEqual(result['automatic_retries'], 0)
        membership = r.read_json(self.root / 'worker_job_membership.json')
        self.assertTrue(membership['actual_worker_in_supervisor_job'])


if __name__ == '__main__':
    unittest.main()
