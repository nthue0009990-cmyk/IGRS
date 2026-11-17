from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('early_opportunities', ROOT / 'scripts/diagnose_clpu_early_opportunities.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


def test_causal_plans_do_not_consume_future_outcomes():
    consumed = []
    def rows():
        for index in range(3):
            consumed.append(index)
            yield {'plan': index}
    plans = d.CausalPlans(rows())
    assert consumed == []
    assert len(plans) == 1
    assert len(plans) == 1
    assert plans[0] == {'plan': 0}
    assert consumed == [0]
    with pytest.raises(AssertionError, match='Noncausal'):
        plans[0]


class Mechanics:
    def identity(self, state, seconds, condition_sha):
        return {'state': state, 'time': seconds, 'condition': condition_sha, 'multipliers': {'load': 1.}}


def fixture_rows():
    m = Mechanics()
    state = {'command': 1}
    electrical = {'action': [1, 0]}
    identities = [m.identity(state, time, 'condition') for time in (0., 30.)]
    points = []
    for index, identity in enumerate(identities):
        points.append({'logical_point_id': index, 'physical_point_id': 0, 'semantic_identity': identity,
            'semantic_identity_sha256': d.digest(identity), 'dynamic_state': state, 'time_seconds': identity['time'],
            'certificate': {'point_id': index, 'status': 'PASS', 'reliable': True, 'feedback': []},
            'electrical_input_key': electrical, 'electrical_input_sha256': d.digest(electrical),
            'reuse_kind': 'ACTUAL_PHYSICAL' if index == 0 else 'ELECTRICAL_EQUIVALENCE',
            'physical_evidence_sha256': 'raw'})
    requests = [{'request': i + 1, 'key': d.digest(identity), 'semantic_cache_hit': False,
                 'electrical_reuse': bool(i), 'status': 'PASS', 'logical_point_id': i, 'physical_point_id': 0}
                for i, identity in enumerate(identities)]
    return m, state, requests, points


def test_prefix_oracle_reuses_only_prior_evidence_and_consumes_only_requested_rows():
    m, state, requests, points = fixture_rows()
    visited = []
    def point_rows():
        for p in points:
            visited.append(p['logical_point_id'])
            yield p
        raise AssertionError('Future outcome was consumed')
    requests.append({'request': 3, 'key': requests[0]['key'], 'semantic_cache_hit': True,
                     'electrical_reuse': False, 'status': 'PASS', 'logical_point_id': 0, 'physical_point_id': 0})
    oracle = d.PrefixOracle(requests, point_rows(), m, 256, 'condition')
    assert oracle.certify(state, 0.)['point_id'] == 0
    assert visited == [0]
    assert oracle.certify(state, 30.)['point_id'] == 1
    assert oracle.certify(state, 0.)['point_id'] == 0
    assert visited == [0, 1]
    assert oracle.costs() == {'requests': 3, 'points': 2, 'logical_points': 2, 'physical_points': 1}


def test_prefix_oracle_rejects_future_alias():
    m, state, requests, points = fixture_rows()
    points[0]['reuse_kind'] = 'ELECTRICAL_EQUIVALENCE'
    with pytest.raises(AssertionError, match='future certificate'):
        d.PrefixOracle(requests, points, m, 256, 'condition').certify(state, 0.)


def test_prefix_oracle_rejects_changed_semantic_identity():
    m, state, requests, points = fixture_rows()
    points[0]['time_seconds'] = 99.
    with pytest.raises(AssertionError, match='altered logical'):
        d.PrefixOracle(requests, points, m, 256, 'condition').certify(state, 0.)


def test_prefix_oracle_rejects_alias_status_change():
    m, state, requests, points = fixture_rows()
    oracle = d.PrefixOracle(requests, points, m, 256, 'condition')
    oracle.certify(state, 0.)
    points[1]['certificate']['status'] = 'FAIL'
    with pytest.raises(AssertionError, match='changed physical status'):
        oracle.certify(state, 30.)


def test_candidate_identity_counts_full_plans_separately_from_branches():
    info = {'actions': (0, 1), 'energy_kwh': 3., 'final_kw': 4., 'operations': 1, 'priority': (-3., -4., 1, (0, 1))}
    branches = [{'id': i, 'prefix': (0,), 'origin': 'prefix_wait', 'parent_id': None, 'info': info} for i in range(2)]
    core = SimpleNamespace(best={'actions': [0, 1]}, m=SimpleNamespace(plan=lambda _: info), checked={(0, 1): {'status': 'COMPLETE'}},
                           window=lambda: branches)
    rows = d.candidate_rows(core)
    assert rows[0]['branch_id'] != rows[1]['branch_id']
    assert rows[0]['plan_sha256'] == rows[1]['plan_sha256']
    assert all(r['checked'] and r['priority_dominated'] and r['graph_energy_nonimproving'] for r in rows)


def population():
    return [{'context_id': c, 'position': p, 'status': 'UNAVAILABLE'} for c in range(8) for p in range(1, 17)]


def test_summary_retains_all_terminal_slots_and_contexts():
    rows = population()
    contexts, summary = d.summarize(rows, [{'context_id': c} for c in range(8)])
    assert len(contexts) == 8
    assert summary['slots'] == summary['unavailable_slots'] == 128
    assert summary['available_slots'] == 0


def test_summary_rejects_removed_or_duplicate_slot():
    rows = population()
    with pytest.raises(AssertionError, match='Missing or duplicate'):
        d.summarize(rows[:-1], [])
    rows[-1] = deepcopy(rows[0])
    with pytest.raises(AssertionError, match='Missing or duplicate'):
        d.summarize(rows, [])


def test_compact_hash_preserves_float_representation_and_ignores_post_action_fields():
    obs = {'candidate_features': [[1.]], 'global_features': [2.], 'branch_ids': [3], 'mask': [True], 'reward': 99.}
    other = deepcopy(obs)
    other['reward'] = -100.
    assert d.digest(d.compact(obs)) == d.digest(d.compact(other))
    other['candidate_features'][0][0] += 1e-14
    assert d.digest(d.compact(obs)) != d.digest(d.compact(other))


def test_manifest_rejects_windows_alias_and_escape(tmp_path):
    target = tmp_path / 'manifest.json'
    target.write_text(json.dumps({'a\\b': '0' * 64, 'a/b': '0' * 64}))
    with pytest.raises(AssertionError, match='duplicate'):
        d.normalized_manifest(target)
    target.write_text(json.dumps({'../outside': '0' * 64}))
    with pytest.raises(AssertionError, match='Unsafe'):
        d.normalized_manifest(target)


def test_no_solver_import_guard():
    with pytest.raises(RuntimeError, match='forbids solver'):
        d.NoSolverImports().find_spec('opendssdirect.Circuit')
    assert d.NoSolverImports().find_spec('numpy') is None


def test_changed_bound_source_rejected(tmp_path):
    (tmp_path / 'source.json').write_text('{}')
    expected = d.sha(tmp_path / 'source.json')
    (tmp_path / 'source.json').write_text('{"changed": true}')
    with pytest.raises(AssertionError, match='Changed/unbound'):
        d.bound_file(tmp_path, {'source.json': expected}, 'source.json', {})


def test_real_source_root_omits_cell_seals_but_audit_binds_them():
    run = ROOT / 'runs' / d.SOURCE_RUN
    assert d.sha(run / 'completion_manifest.json') == d.SOURCE_SHA
    sealed = d.normalized_manifest(run / 'completion_manifest.json')
    audit = d.read(d.bound_file(run, sealed, 'audit/summary.json', {}))
    for row in audit['results']:
        if row['arm'] == 'window_reference':
            name = row['cell'] + '/completion_manifest.json'
            assert name not in sealed
            assert d.sha(run / name) == row['cell_completion_sha256']
