"""Independent shared-service mutation and search-replay checks; no solver."""
import ast
from copy import deepcopy
from pathlib import Path
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_clpu_shared_pilot as audit
from dnr import nature34_clpu_shared_evidence as production
from dnr import nature34_clpu_physical as physical
from dnr.nature34_clpu_branch import BranchConfig
from dnr.nature34_clpu_environment import CLPUBudgetExhausted
from test_nature34_clpu_branch_audit import inputs
from test_nature34_clpu_equivalent_cache import FakeDelegate, known_result, lines, save


@pytest.fixture
def make_service(tmp_path, monkeypatch, inputs):
    monkeypatch.setattr(physical, 'check_known_constraints', known_result)
    monkeypatch.setattr(physical, 'audit_ac_snapshot', lambda *a, **k: {'status': 'PASS'})
    monkeypatch.setattr(physical, 'verify_isolation', lambda *a, **k: {'status': 'PASS'})
    created = []

    def make(mode='shared', failed=True, long=False):
        mech, env, cfg = deepcopy(inputs)
        if long:
            env.horizon = mech.horizon = 257
        run = tmp_path / str(len(created))
        save(run / 'condition.json', env.condition)
        monkeypatch.setattr(production, '_make_delegate',
            lambda parent, cell, env, budget: FakeDelegate(parent, cell, env, budget, failed=failed))
        service = production.SharedEvidenceService(run, run / 'task', env, {'fixed_scope': 'fixture'}, mode=mode)
        created.append(service)
        return service, mech, {**cfg, 'point_budget': 256}
    return make


def material(service, mech):
    views = {name: {'logical_rows': lines(service.cell / name / 'logical_points.jsonl.gz'),
                   'requests': lines(service.cell / name / 'logical_requests.jsonl')} for name in audit.VIEWS}
    points = {r['point_id']: r for r in lines(service.physical_cell / 'points.jsonl.gz')}
    for view in views.values():
        for row in view['logical_rows']:
            if row['reuse_kind'] == 'ACTUAL_PHYSICAL':
                # Synthetic delegate omits only real SnapshotOracle bookkeeping.
                points[row['physical_point_id']].update(dynamic_state=row['dynamic_state'], elapsed_seconds=.001)
    for point in points.values():
        point['physical_evidence_sha256'] = audit.digest({k: point[k] for k in ('raw_outputs', 'metadata_override')})
    return dict(events=lines(service.cell / 'service_events.jsonl'), views=views, points=points, mech=mech,
                scope=service.scope, condition_sha=audit.sha(service.parent / 'condition.json'), mode=service.mode)


def short_case(make_service, mode='shared', failed=True):
    service, mech, _ = make_service(mode, failed=failed)
    with service:
        for name in audit.VIEWS:
            with service.view(name) as view:
                for seconds in (0, 30, 60, 0):
                    view.certify(service.env.initial, seconds)
    return service, material(service, mech)


@pytest.mark.parametrize('mode,actual,cross', [('private', 2, 0), ('shared', 1, 3)])
@pytest.mark.parametrize('failed', [True, False])
def test_service_actual_once_local_ids_causal_aliases_and_true_cost(make_service, mode, actual, cross, failed):
    service, data = short_case(make_service, mode, failed)
    checked, sources = audit.check_service_ledger(**data)
    assert len(sources) == actual and checked['GUARD']['cross_view_reuses'] == cross
    assert all(v['logical_points'] == 3 and v['semantic_cache_hits'] == 1 for v in checked.values())
    assert checked['GUARD']['physical_points'] == (1 if mode == 'private' else 0)
    if failed:
        assert checked['GUARD']['logical'][2]['structured_feedback']['violations'][0]['time_minutes'] == 1.
    audit.check_service_summaries(service.summary(), {n: service.views[n].summary() for n in audit.VIEWS},
                                  checked, service.scope, mode, 3 + 4 * actual)
    if mode == 'private':
        assert data['points'][0]['identity_sha256'] == data['points'][1]['identity_sha256']


@pytest.mark.parametrize('mutation', [
    lambda d: d['views']['GUARD']['logical_rows'][0].__setitem__('source_view', 'GUARD'),
    lambda d: d['views']['GUARD']['logical_rows'][0].__setitem__('source_event_sequence', 999999),
    lambda d: d['views']['GUARD']['logical_rows'][0].__setitem__('physical_point_id', 1),
    lambda d: d['views']['GUARD']['logical_rows'][0].__setitem__('physical_evidence_sha256', '0' * 64),
    lambda d: d['views']['GUARD']['logical_rows'][0].__setitem__('pool', 'GUARD'),
    lambda d: d['views']['GUARD']['logical_rows'][0].__setitem__('cross_view_reuse', False),
    lambda d: d['views']['GUARD']['logical_rows'][1]['certificate'].__setitem__('status', 'PASS'),
    lambda d: d['views']['GUARD']['logical_rows'][1]['certificate']['feedback'][0].__setitem__('time_minutes', 0.),
    lambda d: d['views']['GUARD']['logical_rows'][1].__setitem__('reuse_kind', 'ACTUAL_PHYSICAL'),
    lambda d: d['views']['GUARD']['logical_rows'][1]['electrical_input_key']['multipliers'].__setitem__('load.l1', 1.000000000001),
    lambda d: d['views']['EARLY']['logical_rows'][0].__setitem__('alias_certification_seconds', .1),
    lambda d: d['views']['GUARD']['logical_rows'][1].__setitem__('alias_certification_seconds', float('nan')),
    lambda d: d['views']['GUARD']['requests'][-1].__setitem__('semantic_cache_hit', False),
    lambda d: d['views']['GUARD']['requests'][0].__setitem__('service_request_sequence', 1),
    lambda d: d['views']['GUARD']['requests'][0].__setitem__('service_completion_sequence', 999999),
    lambda d: d['events'][0].__setitem__('physical_budget', 256),
    lambda d: d['events'][1].__setitem__('physical_points', 1),
    lambda d: d['events'].pop(3),
    lambda d: d['events'].append(deepcopy(d['events'][-1])),
    lambda d: d.__setitem__('mode', 'private'),
    lambda d: d['points'][0].__setitem__('reliable', False),
    lambda d: d['scope'].__setitem__('changed', True),
])
def test_causality_scope_status_current_feedback_pool_and_cost_mutations_rejected(make_service, mutation):
    _, data = short_case(make_service)
    mutation(data)
    with pytest.raises((AssertionError, KeyError)):
        audit.check_service_ledger(**data)


def test_each_view_preserves_full_logical_budget_even_when_service_pays_one_point(make_service):
    service, mech, _ = make_service(failed=False, long=True)
    with service:
        for name in audit.VIEWS:
            with service.view(name) as view:
                for i in range(256):
                    view.certify(service.env.initial, 60 * i)
                with pytest.raises(CLPUBudgetExhausted):
                    view.certify(service.env.initial, 60 * 256)
                view.certify(service.env.initial, 0)
    checked, _ = audit.check_service_ledger(**material(service, mech))
    assert service.physical_points == 1
    assert all(c['logical_points'] == 256 and c['denied'] == 1 for c in checked.values())


@pytest.mark.parametrize('mode', ['private', 'shared'])
def test_actual_runner_rule_routes_replay_independently(make_service, mode):
    import run_clpu_shared_pilot as runner
    service, mech, cfg = make_service(mode, failed=False)
    spec = {'cell': 'task', 'method': 'beam', 'mode': mode, 'point_budget': 256}
    for name in audit.VIEWS:
        save(service.cell / name / 'spec.json', runner.phase_spec(spec, name))
    with service:
        results, checkpoints = runner.run_routes(service, service.env, BranchConfig(**cfg), service.cell,
                                                spec, time.perf_counter(), time.time() + 60)
    checked, _ = audit.check_service_ledger(**material(service, mech))
    for name in audit.VIEWS:
        result = audit.check_phase_records(mech, cfg, name, results[name]['spec'], results[name],
            lines(service.cell / name / 'plans.jsonl.gz'), lines(service.cell / name / 'decisions.jsonl.gz'),
            checked[name], audit.sha(service.parent / 'condition.json'))
        assert result['fully_certified'] and result['decisions'] > 0
    assert checkpoints[-1]['numerical_entries'] == 3 + 4 * service.physical_points


def test_auditor_does_not_import_production_service_search_or_runner():
    tree = ast.parse(Path(audit.__file__).read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [x.name for x in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or '']
        else:
            continue
        assert not any(n.startswith('dnr.') or n.startswith('run_clpu_shared') for n in names)


def route_material(make_service, mode='shared'):
    import run_clpu_shared_pilot as runner
    service, mech, cfg = make_service(mode, failed=False)
    spec = {'cell': 'task', 'method': 'beam', 'mode': mode, 'point_budget': 256}
    for name in audit.VIEWS:
        save(service.cell / name / 'spec.json', runner.phase_spec(spec, name))
    started = time.perf_counter()
    with service:
        results, checkpoints = runner.run_routes(service, service.env, BranchConfig(**cfg), service.cell,
                                                spec, started, time.time() + 60)
    evidence, _ = audit.check_service_ledger(**material(service, mech))
    checks = {name: audit.check_phase_records(mech, cfg, name, results[name]['spec'], results[name],
        lines(service.cell / name / 'plans.jsonl.gz'), lines(service.cell / name / 'decisions.jsonl.gz'),
        evidence[name], audit.sha(service.parent / 'condition.json')) for name in audit.VIEWS}
    return service, results, checks, checkpoints


@pytest.mark.parametrize('mode', ['private', 'shared'])
def test_full_cost_curves_from_actual_runner_include_single_init_and_both_logical_searches(make_service, mode):
    service, phases, checks, checkpoints = route_material(make_service, mode)
    audit.check_checkpoints(checkpoints, checks, phases)
    assert checkpoints[-1]['numerical_entries'] == 3 + 4 * service.physical_points
    assert len(checkpoints) == sum(c['decisions'] for c in checks.values()) + 4
    assert all(not c['guard_established'] for c in checkpoints[:-1])


@pytest.mark.parametrize('mutation', [
    lambda c: c.pop(0),
    lambda c: c[0].__setitem__('guard_established', True),
    lambda c: c[-1].__setitem__('numerical_entries', c[-1]['numerical_entries'] + 3),
    lambda c: c[-1].__setitem__('physical_points', c[-1]['logical_points']),
    lambda c: c[-1].__setitem__('phase_physical_points', c[-1]['phase_logical_points']),
    lambda c: c[-1].__setitem__('logical_points', c[-1]['phase_logical_points']),
    lambda c: c[-1].__setitem__('quality_pct', 101.),
    lambda c: c[-1].__setitem__('worker_elapsed_seconds', 0.),
    lambda c: c[-1].__setitem__('phase_numerical_entries', 3),
])
def test_boundary_omission_premature_guard_cost_resets_and_time_reset_rejected(make_service, mutation):
    _, phases, checks, checkpoints = route_material(make_service)
    mutation(checkpoints)
    with pytest.raises(AssertionError):
        audit.check_checkpoints(checkpoints, checks, phases)


def test_actual_runner_worker_result_selection_timing_and_complete_lifecycle(make_service, monkeypatch):
    import os
    from types import SimpleNamespace
    import run_clpu_shared_pilot as runner
    import run_nature34_clpu_pilot as original
    service, mech, cfg = make_service('shared', failed=False)
    run, cell = service.parent, service.cell
    spec = {'cell': 'task', 'method': 'beam', 'mode': 'shared', 'point_budget': 256}
    save(cell / 'spec.json', spec)
    for name in audit.VIEWS:
        save(cell / name / 'spec.json', runner.phase_spec(spec, name))
    save(run / 'config.json', {'branch_search': cfg})
    save(run / 'physical_scope.json', service.scope)
    save(run / 'schedule.json', [spec])
    save(run / 'execution_started.json', {'deadline_epoch_seconds': time.time() + 60})
    save(cell / 'execution.json', {'deadline_epoch_seconds': time.time() + 30})
    monkeypatch.setattr(runner, '_WORKER_ENTERED', False)
    monkeypatch.setattr(runner, 'verify_execution', lambda run: None)
    monkeypatch.setattr(runner.shutil, 'disk_usage', lambda p: SimpleNamespace(free=100 * 2**30))
    monkeypatch.setattr(original, 'make_env', lambda *a: service.env)
    monkeypatch.setattr(production, 'SharedEvidenceService', lambda *a, **k: service)
    assert runner.worker(run, cell) == 0
    result = audit.read(cell / 'result.json')
    phases = {name: audit.read(cell / name / 'phase_result.json') for name in audit.VIEWS}
    assert audit.check_selection(result, phases)['guard_established']
    assert result['selected_phase'] == 'GUARD'  # Exact energy tie.
    audit.check_times(result, phases, audit.read(cell / 'worker_finished.json'), 0.)
    assert audit.read(cell / 'worker_started.json')['pid'] == os.getpid()
    for mutation in ('quality', 'tie', 'unfinished', 'time', 'init'):
        changed, altered = deepcopy(result), deepcopy(phases)
        if mutation == 'quality': changed['quality_pct'] += 1.
        elif mutation == 'tie': changed['selected_phase'] = 'EARLY'
        elif mutation == 'unfinished': altered['GUARD']['done'] = False
        elif mutation == 'time': changed['worker_elapsed_seconds'] = 0.
        else: changed['runtime_initialization_seconds'] = changed['computation_elapsed_seconds'] + 1.
        with pytest.raises(AssertionError):
            if mutation in ('time', 'init'):
                audit.check_times(changed, altered, audit.read(cell / 'worker_finished.json'), 0.)
            else:
                audit.check_selection(changed, altered)


def controller_material():
    contexts = [{'context_id': i, 'scenario_id': f'fault{i}', 'event_id': i % 4} for i in range(8)]
    schedule = audit.expected_schedule(contexts)
    checks, dispatch, exits = [], [], []
    for s in schedule:
        start = 100. + s['pair_order'] * 20
        checks.append({**s, 'logical_points': 10, 'physical_points': 4 if s['mode'] == 'private' else 2,
            'numerical_entries': 19 if s['mode'] == 'private' else 11, 'requests': 20, 'search_decisions': 5,
            'quality_pct': 90., 'guard_quality_pct': 90., 'guard_established': True, 'history_verified': True,
            'cross_view_reuses': 0 if s['mode'] == 'private' else 2,
            'worker_parent_pid': 99, 'worker_finished_elapsed_seconds': 8.,
            'execution': {'started_epoch_seconds': start, 'controller_pid': 99},
            'phase_summary': {n: {'semantic_signature_sha256': f'{s["context_id"]}-{n}'} for n in audit.VIEWS}})
        identity = {'cell': s['cell'], 'pair_order': s['pair_order'], 'launcher_pid': 300 + s['episode_id'],
                    'started_epoch_seconds': start, 'utc': 'fixture'}
        dispatch.append(identity)
        exits.append({**identity, 'finished_epoch_seconds': start + 10., 'process_wall_seconds': 10.,
                      'exit_code': 0, 'drained_after_controller_error': False})
    totals = {k: sum(c[k] for c in checks) for k in ('logical_points', 'physical_points', 'numerical_entries')}
    controller = {'status': 'COMPLETE_CELLS_PENDING_AUDIT', 'scheduled': 16, 'dispatched': schedule,
        'completed': [s['cell'] for s in schedule], 'errors': [], 'unknown_cost_cells': [], 'exits': exits,
        'wall_seconds': 151., **totals}
    return controller, schedule, checks, dispatch


def test_complete_counterbalanced_controller_and_mode_equivalence():
    import run_clpu_shared_pilot as runner
    data = controller_material()
    assert data[1] == runner.schedule_for([{'context_id': i, 'scenario_id': f'fault{i}', 'event_id': i % 4} for i in range(8)])
    assert audit.protocol_settings() == runner.settings()
    totals = audit.check_controller(*data)
    assert totals['physical_points'] == 48 and totals['numerical_entries'] == 240
    assert len(audit.paired_equalities(data[2])) == 8


@pytest.mark.parametrize('mutation', [
    lambda c,s,r,d: c['exits'].pop(),
    lambda c,s,r,d: c['exits'][0].__setitem__('exit_code', 1),
    lambda c,s,r,d: c['exits'][0].__setitem__('drained_after_controller_error', True),
    lambda c,s,r,d: c['exits'][0].__setitem__('process_wall_seconds', 1.),
    lambda c,s,r,d: c.__setitem__('numerical_entries', c['numerical_entries'] - 3),
    lambda c,s,r,d: d.reverse(),
    lambda c,s,r,d: c['unknown_cost_cells'].append(s[0]['cell']),
    lambda c,s,r,d: r[0].__setitem__('worker_parent_pid', -1),
])
def test_real_exit_full_cost_and_population_mutations_rejected(mutation):
    data = controller_material()
    mutation(*data)
    with pytest.raises(AssertionError): audit.check_controller(*data)


@pytest.mark.parametrize('mutation', [
    lambda r: r.pop(),
    lambda r: r[0].__setitem__('guard_established', False),
    lambda r: r[0].__setitem__('history_verified', False),
    lambda r: r[0].__setitem__('requests', 21),
    lambda r: r[0]['phase_summary']['GUARD'].__setitem__('semantic_signature_sha256', 'changed'),
])
def test_paired_scientific_equality_is_not_quality_only(mutation):
    checks = controller_material()[2]
    mutation(checks)
    with pytest.raises(AssertionError): audit.paired_equalities(checks)


def test_real_windows_freeze_normalization_and_content_binding(tmp_path):
    source = tmp_path / 'source/docs/protocol.md'
    source.parent.mkdir(parents=True)
    source.write_text('unchanged scientific protocol', encoding='utf-8')
    value = audit.sha(source)
    save(tmp_path / 'prephysical_freeze_manifest.json', {'source\\docs\\protocol.md': value})
    assert audit.checked_manifest(tmp_path, 'prephysical_freeze_manifest.json') == {'source/docs/protocol.md': value}
    source.write_text('changed protocol', encoding='utf-8')
    with pytest.raises(AssertionError): audit.checked_manifest(tmp_path, 'prephysical_freeze_manifest.json')


@pytest.mark.parametrize('alias', ['source/docs/protocol.md', 'SOURCE/docs/protocol.md',
    '../protocol.md', 'source/../docs/protocol.md', './source/docs/protocol.md',
    'source//docs/protocol.md', 'C:/outside/protocol.md', '/outside/protocol.md'])
def test_manifest_separator_case_aliases_and_traversals_rejected(tmp_path, alias):
    source = tmp_path / 'source/docs/protocol.md'
    source.parent.mkdir(parents=True)
    source.write_text('fixed', encoding='utf-8')
    value = audit.sha(source)
    save(tmp_path / 'completion_manifest.json', {'source\\docs\\protocol.md': value, alias: value})
    with pytest.raises(AssertionError): audit.checked_manifest(tmp_path)


def test_full_real_prepared_parent_readonly_after_path_fix(monkeypatch):
    """Development regression, not a substitute for a new frozen preflight.

Only the two authorized edited source files are read from the old snapshot for
its original SHA assertions. Every other real prepared byte and full parent
constraint is checked normally under the no-solver import guard.
"""
    from audit_d059_supplement import NoSolverImports, no_solver_loaded
    run = audit.ROOT / 'runs/20261031T045421_777631Z_clpu_shared_rule_pilot'
    if not run.exists(): pytest.skip('Prephysical Windows regression run unavailable')
    original_sha = audit.sha
    changed = {audit.ROOT / p: run / 'source' / p for p in
        ('scripts/audit_clpu_shared_pilot.py', 'tests/test_clpu_shared_audit.py')}
    def historical_source_sha(path):
        path = Path(path).resolve()
        return original_sha(changed.get(path, path))
    monkeypatch.setattr(audit, 'sha', historical_source_sha)
    monkeypatch.setattr(audit.prior, 'sha', historical_source_sha)
    guard = NoSolverImports()
    sys.meta_path.insert(0, guard)
    try:
        no_solver_loaded()
        cfg, scope, schedule = audit.check_parent(run)
        no_solver_loaded()
    finally:
        sys.meta_path.remove(guard)
    assert cfg['shared_study'] == audit.STUDY and len(schedule) == 16
    assert {s['context_id'] for s in schedule} == set(range(8))
    assert scope['alpha'] == .5
