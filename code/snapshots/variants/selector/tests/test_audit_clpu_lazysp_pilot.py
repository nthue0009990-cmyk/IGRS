"""Zero-solver independent replay and evidence-tampering checks."""
import ast
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from collections import Counter

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_clpu_lazysp_pilot as audit
from dnr.nature34_clpu_branch import BranchConfig, BranchSearch
from dnr.nature34_clpu_staged_control import StagedCertificationSearch
from dnr.nature34_clpu_lazysp import LazySPCertificationSearch
from test_nature34_clpu_branch_audit import inputs
from test_clpu_staged_audit import full_env, mechanics
from test_clpu_deferred_audit import LogicalActualOracle
from test_nature34_clpu_shared_evidence import build


def saved(inputs, selector='forward', failed=False):
    env = full_env(inputs)
    config = BranchConfig(max_branch_expansions=20, max_checked_candidates=20, max_generated_candidates=60)
    guard_oracle = LogicalActualOracle(env, 61)
    guard_search = BranchSearch(env, guard_oracle, BranchConfig(**{**asdict(config), 'point_budget': 61}))
    while not guard_search.done:
        guard_search.step(guard_search.rule_action('beam'))
    guard = json.loads(json.dumps(guard_search.result()))
    guard['fully_certified'] = guard['best_plan'] is not None
    ref = {'phase': 'GUARD', 'file': 'GUARD/phase_result.json', 'sha256': audit.digest(guard)}
    oracle = LogicalActualOracle(env, 256, failed)
    emitted = []
    bootstrap = StagedCertificationSearch(env, oracle, config, record_callback=lambda x: emitted.append(deepcopy(x)))
    search = LazySPCertificationSearch(bootstrap, guard_result=guard, guard_reference=ref, selector=selector)
    result = json.loads(json.dumps(search.run()))
    result.update(search_decisions=len(result['lazysp']['journal']), policy_decisions=0,
                  fully_certified=result['best_plan'] is not None,
                  quality_pct=100 * result['best_plan']['validation']['energy_kwh'] /
                      (env.total_kw * env.horizon * env.slot_seconds / 3600) if result['best_plan'] else None)
    assert json.loads(json.dumps(emitted)) == result['attempts']
    evidence = {'request_rows': oracle.rows, 'logical': oracle.raw,
                'logical_rows': [{'reuse_kind': 'ACTUAL_PHYSICAL'} for _ in oracle.raw],
                'physical_points': oracle.physical_points}
    return dict(mech=mechanics(env), cfg=asdict(config), result=result, attempts=result['attempts'],
                evidence=evidence, condition_sha='synthetic', guard=guard, reference=ref, selector=selector)


@pytest.mark.parametrize('selector', ['forward', 'reverse'])
@pytest.mark.parametrize('failed', [False, True])
def test_complete_new_selector_and_failed_prefix_replay_independently(inputs, selector, failed):
    data = saved(inputs, selector, failed)
    checked = audit.check_lazy_records(**data)
    assert checked['full_causal_replay']
    assert checked['selected_plan_count'] <= 16


@pytest.mark.parametrize('mutation', ['sample_deleted', 'prefix_history', 'wrong_edge_order',
                                    'fake_failure', 'missing_canonical_request', 'false_cost', 'false_guard'])
def test_mutated_edges_history_causal_requests_and_cost_are_rejected(inputs, mutation):
    data = saved(inputs, 'reverse')
    result = data['result']; lazy = result['lazysp']
    if mutation == 'sample_deleted':
        next(e for e in lazy['edges'] if e['status'] == 'PASS')['samples'].pop()
    elif mutation == 'prefix_history':
        next(e for e in lazy['edges'] if e['status'] == 'PASS')['state']['pickups'][0] = 999
    elif mutation == 'wrong_edge_order':
        next(e for e in lazy['journal'] if e['event'] == 'EDGE_SELECTED')['edge_id'] += 1
    elif mutation == 'fake_failure':
        next(e for e in lazy['edges'] if e['status'] == 'PASS')['status'] = 'FAIL'
    elif mutation == 'missing_canonical_request':
        data['evidence']['request_rows'].pop()
    elif mutation == 'false_cost':
        result['actual_physical_points'] -= 1
    else:
        lazy['external_guard_plan']['validation']['energy_kwh'] += 1
    with pytest.raises(AssertionError):
        audit.check_lazy_records(**data)


def test_auditor_does_not_import_production_selector_or_runner():
    tree = ast.parse(Path(audit.__file__).read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        names = [n.name for n in node.names] if isinstance(node, ast.Import) else [node.module or ''] if isinstance(node, ast.ImportFrom) else []
        assert not any(n.startswith('dnr.') or n.startswith('run_clpu_lazysp') for n in names)


def test_frozen_population_and_cap_match_runner_contract():
    import run_clpu_lazysp_pilot as runner
    assert audit.fixed_settings() == runner.settings()
    schedule = audit.expected_schedule()
    assert len(schedule) == 12
    assert {(s['context_id'], s['arm']) for s in schedule} == {(c, a) for c in audit.CONTEXTS for a in audit.ARMS}


@pytest.mark.parametrize('arm', ['P', 'L_forward', 'L_reverse'])
def test_actual_worker_to_independent_cell_audit_with_synthetic_physics(build, inputs, monkeypatch, arm):
    import run_clpu_lazysp_pilot as runner
    import run_nature34_clpu_pilot as physical_runner
    from dnr import nature34_clpu_guard_first_control as production
    from dnr.nature34_clpu_staged_pickup import generate_staged_pickup_candidates
    from test_clpu_shared_audit import material
    template = build('shared'); env = full_env(inputs)
    run, cell = template.parent, template.cell
    runner.save(run / 'condition.json', env.condition)
    service = production.GuardFirstEvidenceService(run, cell, env, template.scope,
                mode='shared', physical_budget=512, logical_budget=256)
    spec = deepcopy(next(x for x in audit.expected_schedule() if x['arm'] == arm))
    spec.update(cell='task', scenario_id=env.scenario['scenario_id'], event_id=env.event['event_id'])
    cfg = {'horizon': 30, 'slot_seconds': 60, 'sample_seconds': 30,
           'branch_search': asdict(BranchConfig(max_branch_expansions=20, max_checked_candidates=20, max_generated_candidates=60))}
    values = {'schedule.json': [spec], 'config.json': cfg, 'physical_scope.json': service.scope,
              'author_static_metadata.json': env.static, 'load_reference.json': env.loads,
              'selected_scenarios.json': [env.scenario], 'selected_events.json': [env.event],
              'command_inventory.json': {},
              'stage_plan.json': {'contexts': [{'context_id': spec['context_id'], 'generation': generate_staged_pickup_candidates(env)}]}}
    for name, value in values.items(): runner.save(run / name, value)
    runner.save(cell / 'spec.json', spec)
    runner.save(cell / 'execution.json', {'started_epoch_seconds': time.time(), 'deadline_epoch_seconds': time.time() + 180})
    monkeypatch.setattr(runner, 'verify', lambda _: None)
    monkeypatch.setattr(physical_runner, 'make_env', lambda *_: env)
    monkeypatch.setattr(production, 'GuardFirstEvidenceService', lambda *a, **k: service)
    assert runner.worker(run, cell) == 0
    (cell / 'physical/command_ledger.jsonl').write_text('', encoding='utf-8')
    runner.seal(cell)
    mech = mechanics(env); data = material(service, mech)
    for point in data['points'].values(): point['elapsed_seconds'] = 0.
    statuses = Counter(p['status'] for p in data['points'].values())
    monkeypatch.setattr(audit.shared, 'check_actual_points', lambda *a: (data['points'], statuses))
    monkeypatch.setattr(audit.old, 'check_commands', lambda *a: {'numerical_entries': 3 + 4 * len(data['points'])})
    monkeypatch.setattr(audit.shared, 'check_actual_requests', lambda *a: None)
    checked = audit.check_cell(run, spec)
    assert checked['guard_established'] and checked['status'] == 'PASS'
    assert checked['numerical_entries'] == 3 + 4 * checked['physical_points']
