"""Solver-free checks of causal sharing and independent search accounting."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from dnr import nature34_clpu_shared_evidence as shared
from dnr import nature34_clpu_physical as physical
from dnr.nature34_clpu_branch import BranchConfig, BranchSearch
from dnr.nature34_clpu_window import WindowBranchSearch
from dnr.nature34_clpu_environment import CLPUBudgetExhausted, CLPUReliabilityError
from test_nature34_clpu_branch import environment
from test_nature34_clpu_equivalent_cache import FakeDelegate, known_result, lines, save


@pytest.fixture
def build(tmp_path, monkeypatch):
    monkeypatch.setattr(physical, 'check_known_constraints', known_result)
    monkeypatch.setattr(physical, 'audit_ac_snapshot', lambda *a, **k: {'status': 'PASS'})
    monkeypatch.setattr(physical, 'verify_isolation', lambda *a, **k: {'status': 'PASS'})
    delegates = []
    created = []

    def make(mode='shared', *, failed=False, unknown=False, physical_budget=512, logical_budget=256,
             delegate_class=FakeDelegate):
        env = environment(horizon=4)
        parent = tmp_path / str(len(created))
        save(parent / 'condition.json', env.condition)

        def factory(parent, cell, env, budget):
            delegate = delegate_class(parent, cell, env, budget, failed=failed, unknown=unknown)
            delegates.append(delegate)
            return delegate

        monkeypatch.setattr(shared, '_make_delegate', factory)
        service = shared.SharedEvidenceService(parent, parent / 'task', env, {'frozen': 'fixture'},
            mode=mode, physical_budget=physical_budget, logical_budget=logical_budget)
        created.append(service)
        return service

    make.delegates = delegates
    return make


@pytest.mark.parametrize('mode,total,cross', [('private', 2, 0), ('shared', 1, 2)])
@pytest.mark.parametrize('failed', [False, True])
def test_two_views_share_only_reliable_exact_evidence_with_independent_ids(build, mode, total, cross, failed):
    service = build(mode, failed=failed)
    certificates = []
    with service:
        for name in ('EARLY', 'GUARD'):
            with service.view(name) as view:
                certificates.append([view.certify(service.env.initial, t) for t in (0, 60)])
                assert view.certify(service.env.initial, 0) == certificates[-1][0]
                assert view.points == 2 and view.requests == 3 and view.hits == 1
                assert [c['point_id'] for c in certificates[-1]] == [0, 1]
                if failed:
                    assert certificates[-1][0]['feedback'][0]['time_minutes'] == 0
                    assert certificates[-1][1]['feedback'][0]['time_minutes'] == 1
        assert service.physical_points == total
    assert len(build.delegates) == 1 and build.delegates[0].closed
    assert service.views['GUARD'].cross_view_reuses == cross
    assert certificates[0] == certificates[1]
    summary = service.summary()
    assert summary['actual_numerical_entries'] == 3 + 4 * total
    assert summary['runtime_initializations'] == 1
    assert sum(v.summary()['actual_numerical_entries_attributed'] for v in service.views.values()) == 4 * total
    assert all(v.summary()['initialization_numerical_entries_attributed'] == 0 for v in service.views.values())
    rows = lines(service.physical_cell / 'points.jsonl.gz')
    assert len(rows) == total and [r['point_id'] for r in rows] == list(range(total))
    assert all(r['actual_call_counts']['numerical_solution_entries'] == 4 for r in rows)
    if mode == 'private':
        # Same actual identity is paid twice; the old delegate cache cannot leak.
        assert rows[0]['identity_sha256'] == rows[1]['identity_sha256']


def test_every_alias_has_a_prior_same_pool_actual_source_and_immutable_payload(build):
    service = build(failed=True)
    with service:
        with service.view('EARLY') as early:
            early.certify(service.env.initial, 0)
            original = deepcopy(build.delegates[0].runtime.rows[0])
        with service.view('GUARD') as guard:
            cert = guard.certify(service.env.initial, 60)
            cert['status'] = 'PASS'
            cert['feedback'].clear()
            assert guard.certify(service.env.initial, 60)['status'] == 'FAIL'
        assert build.delegates[0].runtime.rows[0] == original
    events = lines(service.cell / 'service_events.jsonl')
    assert [r['sequence'] for r in events] == list(range(1, len(events) + 1))
    by_sequence = {r['sequence']: r for r in events}
    actual = lines(service.physical_cell / 'points.jsonl.gz')
    for view in service.views.values():
        for row in lines(view.cell / 'logical_points.jsonl.gz'):
            source = by_sequence[row['source_event_sequence']]
            assert source['kind'] == 'ACTUAL_COMPLETE'
            assert source['sequence'] < row['service_sequence']
            assert source['pool'] == row['pool']
            assert source['physical_point_id'] == row['physical_point_id']
            assert source['electrical_input_sha256'] == row['electrical_input_sha256']
            assert row['physical_evidence_sha256'] == shared.sha_value(shared.physical_evidence_payload(actual[row['physical_point_id']]))
            assert 'actual_call_counts' not in row
        for request in lines(view.cell / 'logical_requests.jsonl'):
            assert by_sequence[request['service_request_sequence']]['kind'] == 'REQUEST_STARTED'
            assert by_sequence[request['service_completion_sequence']]['kind'] == 'REQUEST_COMPLETE'


def test_changed_electrical_input_is_paid_and_future_view_never_backfills_history(build):
    service = build()
    with service:
        with service.view('EARLY') as early:
            early.certify(service.env.initial, 0)
        before = (early.cell / 'logical_points.jsonl.gz').read_bytes()
        with service.view('GUARD') as guard:
            changed = service.env.advance(service.env.initial, 16)
            guard.certify(changed, 0)
            guard.certify(changed, 30)  # Different CLPU multiplier is not rounded.
            assert guard.points == guard.physical_points == 2
            assert guard.cross_view_reuses == 0
        assert (early.cell / 'logical_points.jsonl.gz').read_bytes() == before
    assert service.physical_points == 3


def test_each_view_budget_is_independent_even_when_every_request_reuses(build):
    service = build(logical_budget=2)
    with service:
        for name in ('EARLY', 'GUARD'):
            with service.view(name) as view:
                view.certify(service.env.initial, 0)
                view.certify(service.env.initial, 30)
                with pytest.raises(CLPUBudgetExhausted):
                    view.certify(service.env.initial, 60)
                assert view.certify(service.env.initial, 0)['point_id'] == 0
                assert view.points == 2
    assert service.physical_points == 1 and service.summary()['logical_points'] == 4
    assert all(lines(v.cell / 'logical_requests.jsonl')[2]['status'] == 'BUDGET_UNKNOWN' for v in service.views.values())


def test_global_budget_prevents_new_solve_but_allows_causal_alias(build):
    service = build(physical_budget=1)
    with service:
        with service.view('EARLY') as early:
            early.certify(service.env.initial, 0)
        with service.view('GUARD') as guard:
            guard.certify(service.env.initial, 60)
            with pytest.raises(CLPUBudgetExhausted):
                guard.certify(service.env.advance(service.env.initial, 16), 0)
            assert guard.points == 1 and guard.physical_points == 0
    assert service.physical_points == 1


@pytest.mark.parametrize('mutation', ['scope', 'environment'])
def test_frozen_scope_and_environment_changes_stop_before_any_new_point(build, mutation):
    service = build()
    with pytest.raises(CLPUReliabilityError, match='scope or environment'):
        with service:
            with service.view('EARLY') as view:
                if mutation == 'scope':
                    service.scope['changed'] = True
                else:
                    service.env.condition['load_multiplier'] += .1
                view.certify(service.env.initial, 0)
    assert service.physical_points == 0 and service.status == 'STOPPED'


def test_unreliable_actual_is_retained_never_pooled_and_poison_cannot_be_swallowed(build):
    service = build(unknown=True)
    with pytest.raises(CLPUReliabilityError, match='Retained service error'):
      with service:
        with service.view('EARLY') as early:
            with pytest.raises(CLPUReliabilityError):
                early.certify(service.env.initial, 0)
            with pytest.raises(CLPUReliabilityError, match='retained error'):
                early.certify(service.env.initial, 30)
        with pytest.raises(CLPUReliabilityError):
            service.view('GUARD')
    assert service.physical_points == 1 and service.views['EARLY'].points == 0
    assert service.status == 'STOPPED' and not service.summary()['physical_cost_complete']
    assert not any(service._pools.values())
    assert len(lines(service.physical_cell / 'points.jsonl.gz')) == 1
    assert lines(service.views['EARLY'].cell / 'logical_points.jsonl.gz') == []


def test_alias_recertification_error_cannot_change_pass_to_success(build, monkeypatch):
    service = build()
    with pytest.raises(CLPUReliabilityError, match='reliable'):
        with service:
            with service.view('EARLY') as early:
                early.certify(service.env.initial, 0)
            monkeypatch.setattr(physical, 'audit_ac_snapshot', lambda *a, **k: {'status': 'FAIL'})
            with service.view('GUARD') as guard:
                guard.certify(service.env.initial, 60)
    assert service.physical_points == 1 and service.views['GUARD'].points == 0


def test_new_tasks_never_share_previous_service_evidence(build):
    for _ in range(2):
        service = build()
        with service:
            with service.view('EARLY') as view:
                view.certify(service.env.initial, 60)
                assert view.physical_points == 1 and view.electrical_reuses == 0
    assert len(build.delegates) == 2


def test_original_search_logical_results_are_identical_between_modes(build):
    results, paid = [], []
    for mode in ('private', 'shared'):
        service = build(mode)
        mode_results = []
        with service:
            for name, cls in [('EARLY', WindowBranchSearch), ('GUARD', BranchSearch)]:
                with service.view(name) as view:
                    search = cls(service.env, view, BranchConfig())
                    while not search.done:
                        search.step(search.rule_action('beam'))
                    mode_results.append(search.result())
        results.append(mode_results)
        paid.append(service.physical_points)
    assert results[0] == results[1]
    assert all(row['best_plan']['certified_complete'] for row in results[0])
    assert paid[1] < paid[0]


def test_view_order_overlap_reentry_and_deadline(build):
    service = build()
    with pytest.raises(CLPUReliabilityError, match='Retained service error'):
      with service:
        with pytest.raises(CLPUReliabilityError):
            service.view('GUARD')
        with service.view('EARLY') as early:
            with pytest.raises(CLPUReliabilityError):
                service.view('GUARD')
            with pytest.raises(CLPUReliabilityError):
                early.__enter__()
        with service.view('GUARD') as guard:
            guard.deadline_monotonic = -1
            with pytest.raises(TimeoutError):
                guard.certify(service.env.initial, 0)
    assert service.status == 'STOPPED' and service.physical_points == 0


def test_partial_native_failure_keeps_measured_entries_without_inventing_a_point(build):
    class Broken(FakeDelegate):
        def __enter__(self):
            super().__enter__()
            self.partial = 0
            self.factory = SimpleNamespace(stats=lambda: {'returned': {
                'numerical_solution_entries': 3 + 4 * self.points + self.partial}})
            return self

        def certify(self, state, seconds):
            self.partial += 2
            raise RuntimeError('partial native call')

    service = build(delegate_class=Broken)
    with pytest.raises(RuntimeError, match='partial native'):
        with service:
            with service.view('EARLY') as view:
                view.certify(service.env.initial, 0)
    assert service.physical_points == view.points == 0
    assert view.summary()['actual_numerical_entries_attributed'] == 2
    assert service.summary()['actual_numerical_entries'] == 5
    assert service.summary()['expected_numerical_entries'] is None
    assert not any(service._pools.values())


def test_startup_failure_is_recorded_with_unknown_initialization_cost(build):
    class Broken(FakeDelegate):
        def __enter__(self):
            self.closed = True  # Original SnapshotOracle cleans up its failed entry.
            raise RuntimeError('initialization failure')
    service = build(delegate_class=Broken)
    with pytest.raises(RuntimeError, match='initialization'):
        with service:
            pass
    summary = json.loads((service.cell / 'service_summary.json').read_text())
    assert summary['status'] == 'STARTUP_ERROR'
    assert summary['initialization_numerical_entries'] is None
    assert summary['expected_numerical_entries'] is None and not summary['physical_cost_complete']
    assert lines(service.cell / 'service_events.jsonl')[0]['kind'] == 'SERVICE_STARTED'


def test_delegate_close_failure_is_retained_and_propagated(build):
    class Broken(FakeDelegate):
        def __exit__(self, *args):
            super().__exit__(*args)
            raise RuntimeError('close failure')
    service = build(delegate_class=Broken)
    with pytest.raises(RuntimeError, match='close failure'):
        with service:
            with service.view('EARLY') as view:
                view.certify(service.env.initial, 0)
    assert service.closed and service.status == 'STOPPED'
    assert service.physical_points == 1 and build.delegates[0].closed


@pytest.mark.parametrize('kwargs', [{'mode': 'other'}, {'physical_budget': 0}, {'logical_budget': True}])
def test_invalid_modes_and_budgets_are_rejected(build, kwargs):
    with pytest.raises(ValueError):
        build(**kwargs)
