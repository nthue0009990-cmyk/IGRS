"""Actual runner-to-report schema check with the physical delegate mocked."""
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_clpu_shared_pilot as runner
import report_clpu_shared_pilot as report
import run_nature34_clpu_pilot as physical_runner
from dnr import nature34_clpu_shared_evidence as service_module
from dnr.nature34_clpu_branch import BranchConfig
from test_nature34_clpu_shared_evidence import build
from test_nature34_clpu_equivalent_cache import lines
from test_run_clpu_shared_pilot import contexts


@pytest.mark.parametrize('mode', ['private', 'shared'])
def test_actual_worker_result_passes_report_contract(build, monkeypatch, mode):
    service = build(mode)
    run, cell = service.parent, service.cell
    spec = {**runner.schedule_for(contexts())[0], 'cell': 'task', 'mode': mode,
            'scenario_id': service.env.scenario['scenario_id'], 'event_id': service.env.event['event_id']}
    runner.save(cell / 'spec.json', spec)
    runner.save(run / 'schedule.json', [spec])
    runner.save(run / 'config.json', {'branch_search': asdict(BranchConfig())})
    runner.save(run / 'physical_scope.json', service.scope)
    deadline = time.time() + 60
    runner.save(cell / 'execution.json', {'deadline_epoch_seconds': deadline})
    runner.save(run / 'execution_started.json', {'deadline_epoch_seconds': deadline})
    monkeypatch.setattr(runner, '_WORKER_ENTERED', False)
    monkeypatch.setattr(runner, 'verify_execution', lambda _: None)
    monkeypatch.setattr(runner.shutil, 'disk_usage', lambda _: SimpleNamespace(free=10**15))
    monkeypatch.setattr(physical_runner, 'make_env', lambda *_: service.env)
    monkeypatch.setattr(service_module, 'SharedEvidenceService', lambda *a, **k: service)
    assert runner.worker(run, cell) == 0
    result = report.read(cell / 'result.json')
    physical = report.read(cell / 'physical/oracle_summary.json')
    # The legacy FakeDelegate saves a compact summary; retain its actual raw
    # point statuses to supply the omitted summary column, without a solver.
    physical['statuses'] = dict(Counter(r['status'] for r in lines(cell / 'physical/points.jsonl.gz')))
    audit_row = {'cell': spec['cell'], 'status': 'PASS', 'process_wall_seconds': result['worker_elapsed_seconds'] + .1,
        'physical_points': result['actual_physical_points'], 'logical_points': result['logical_points'],
        'numerical_entries': result['numerical_entries'], 'quality_pct': result['quality_pct'],
        'guard_established': result['guard_audit']['guard_established']}
    exit_row = {'cell': spec['cell'], 'exit_code': 0, 'process_wall_seconds': audit_row['process_wall_seconds']}
    row = report.case_row(spec, result, audit_row, exit_row, physical)
    assert row['guard_established'] and row['fully_certified']
    assert row['numerical_entries'] == 3 + 4 * row['physical_points']
    assert row['worker_elapsed_seconds'] == pytest.approx(
        row['preflight_elapsed_seconds'] + row['computation_elapsed_seconds'])
    assert row['checkpoints'][-1]['guard_established']
    assert row['view_ledgers']['GUARD']['initialization_numerical_entries_attributed'] == 0
    if mode == 'shared':
        assert row['phase_summary']['GUARD']['actual_physical_points'] == 0
        assert row['cross_view_reuses'] > 0
