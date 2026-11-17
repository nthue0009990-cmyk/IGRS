"""Independent alias/trajectory/cost audits; physical execution is mocked or historical."""
from copy import deepcopy
import ast
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_clpu_equivalent_cache as audit
from dnr.nature34_clpu_branch import BranchConfig
from dnr.nature34_clpu_environment import CLPUBudgetExhausted
from dnr.nature34_clpu_window import WindowBranchSearch
from test_nature34_clpu_branch_audit import inputs
from test_nature34_clpu_equivalent_cache import oracle_factory, lines
from test_clpu_window_teacher_audit import historical_source_fixture


def ledger(oracle, mech):
    logical = lines(oracle.cell / 'logical_points.jsonl.gz')
    actual = {row['point_id']: row for row in lines(oracle.cell / 'points.jsonl.gz')}
    # The mocked delegate omits these real SnapshotOracle bookkeeping fields.
    # Fill only synthetic test metadata; raw evidence and certificate are untouched.
    for row in logical:
        if row['reuse_kind'] == 'ACTUAL_PHYSICAL':
            actual[row['physical_point_id']].update(dynamic_state=row['dynamic_state'], elapsed_seconds=.001)
    return {'logical_rows': logical, 'requests': lines(oracle.cell / 'logical_requests.jsonl'),
        'actual_rows': actual, 'mech': mech, 'scope': oracle.scope,
        'condition_sha': audit.sha(oracle.parent / 'condition.json'), 'arm': oracle.variant}


def small_ledger(inputs, oracle_factory, arm='electrical_equivalent', failed=True):
    mech, env, _ = inputs
    build, _ = oracle_factory
    oracle = build(arm, failed=failed, env=deepcopy(env))
    with oracle:
        for seconds in (0, 30, 60):
            oracle.certify(oracle.env.initial, seconds)
        oracle.certify(oracle.env.initial, 30)
    return oracle, ledger(oracle, mech)


def saved_search(inputs, oracle_factory, arm='electrical_equivalent', failed=False):
    mech, env, cfg = inputs
    cfg = {**cfg, 'point_budget': 256}
    build, _ = oracle_factory
    oracle = build(arm, env=deepcopy(env), failed=failed)
    steps, plans = [], []
    with oracle:
        search = WindowBranchSearch(oracle.env, oracle, BranchConfig(**cfg), record_callback=plans.append)

        def counters():
            return {'requests': oracle.requests, 'points': oracle.points,
                    'logical_points': oracle.points, 'physical_points': oracle.physical_points}

        def boundary(i, elapsed):
            return {'search_choices': i, 'logical_points': oracle.points, 'physical_points': oracle.physical_points,
                    'numerical_entries': 3 + 4 * oracle.physical_points, 'elapsed_seconds': elapsed,
                    'quality_pct': 100 * search.quality if search.incumbent is not None else None,
                    'fully_certified': search.incumbent is not None}

        checkpoints, n = [boundary(0, 1.)], 0
        while not search.done:
            ids = search.observation('raw')['branch_ids']
            action = search.rule_action('beam')
            start = {'transition_id': n, 'status': 'STARTED', 'method': 'beam', 'action': action,
                'branch_id': ids[action], 'window_branch_ids': ids, 'oracle_before': counters(),
                'elapsed_seconds': n * 2 + 2.}
            steps.append(start)
            response = search.step(action)
            steps.append({**start, 'status': 'COMPLETED', 'response': response,
                'oracle_after': counters(), 'elapsed_seconds': n * 2 + 3.})
            n += 1
            checkpoints.append(boundary(n, n * 2 + 1.))
        result = search.result()
        result.update(spec={'method': 'beam', 'window_variant': 'coverage_v1'},
            completed_expansions=n, search_decisions=n, initial_elapsed_seconds=1.,
            elapsed_seconds=n * 2 + 2., checkpoints=checkpoints,
            wait_baseline_quality_pct=checkpoints[0]['quality_pct'],
            counts={'generated_candidates': len(search._branch_keys), 'graph_assessed_candidates': len(search._assessed),
                'checked_candidates': len(search._checked), 'branch_expansions': search._beam_turn,
                'generation_cap_rejections': search.generation_cap_rejections})
    data = ledger(oracle, mech)
    logical, requests, costs = audit.check_logical_records(**data)
    return {'run': oracle.parent, 'spec': result['spec'], 'cfg': {'branch_search': cfg}, 'result': result,
        'logical_rows': data['logical_rows'], 'points': logical, 'requests': requests,
        'plans': plans, 'steps': steps, 'mech': mech}, costs


@pytest.mark.parametrize('arm,physical,reuses', [('semantic_identity', 3, 0), ('electrical_equivalent', 1, 2)])
def test_independent_dual_ledger_retimes_fail_and_charges_actual_only(inputs, oracle_factory, arm, physical, reuses):
    oracle, data = small_ledger(inputs, oracle_factory, arm)
    logical, requests, costs = audit.check_logical_records(**data)
    assert len(logical) == 3 and costs['physical_points'] == physical and costs['electrical_reuses'] == reuses
    assert costs['semantic_cache_hits'] == 1 and len(requests) == 7
    assert logical[2]['status'] == 'FAIL'
    assert logical[2]['structured_feedback']['violations'][0]['time_minutes'] == 1.
    assert sum(p['elapsed_seconds'] for p in logical.values()) == .001 * physical
    audit.check_wrapper_summary(oracle.summary(), oracle.summary(), costs, logical, oracle.scope, arm, 3 + 4 * physical)


@pytest.mark.parametrize('mutation,pattern', [
    (lambda d: d['logical_rows'][1]['certificate']['feedback'][0].__setitem__('time_minutes', 0), 'current-time feedback'),
    (lambda d: d['logical_rows'][1]['certificate'].__setitem__('status', 'PASS'), 'certificate'),
    (lambda d: d['logical_rows'][1]['certificate'].__setitem__('point_id', 0), 'certificate'),
    (lambda d: d['logical_rows'][1].__setitem__('physical_evidence_sha256', '0' * 64), 'raw evidence hash'),
    (lambda d: d['logical_rows'][1].__setitem__('physical_point_id', 1), 'absent physical'),
    (lambda d: d['logical_rows'][0].__setitem__('reuse_kind', 'ELECTRICAL_EQUIVALENCE'), 'Future'),
    (lambda d: d['logical_rows'][1].__setitem__('reuse_kind', 'ACTUAL_PHYSICAL'), 'identity or order'),
    (lambda d: d.__setitem__('arm', 'semantic_identity'), 'unapproved'),
    (lambda d: d['scope'].__setitem__('solver_reset', 'changed'), 'equivalence key'),
    (lambda d: d['logical_rows'][1]['electrical_input_key']['multipliers'].__setitem__('load.l1', 1.000000000000001), 'equivalence key'),
    (lambda d: d['logical_rows'][1]['semantic_identity'].__setitem__('time_seconds', 0), 'semantic identity'),
    (lambda d: d['logical_rows'][1].__setitem__('time_seconds', 31), 'outside frozen scope'),
    (lambda d: d['logical_rows'][0].__setitem__('alias_certification_seconds', 1), 'falsely claims'),
    (lambda d: d['logical_rows'][1].__setitem__('alias_certification_seconds', float('nan')), 'Invalid elapsed'),
    (lambda d: d['actual_rows'][0]['raw_outputs'].__setitem__('fail', False), 'raw evidence hash'),
    (lambda d: d['actual_rows'][0].__setitem__('reliable', False), 'certificate'),
    (lambda d: d['requests'][0].__setitem__('semantic_cache_hit', True), 'Future'),
    (lambda d: d['requests'][-1].__setitem__('logical_point_id', 2), 'false semantic'),
    (lambda d: d['requests'][1].__setitem__('electrical_reuse', False), 'request disagree'),
    (lambda d: d['requests'][1].__setitem__('semantic_cache_hit', 0), 'cache flags'),
    (lambda d: d['requests'][1].__setitem__('request', 9), 'request order'),
    (lambda d: d['requests'].pop(1), 'request order'),
])
def test_alias_causality_identity_feedback_and_cost_tampering_rejected(inputs, oracle_factory, mutation, pattern):
    _, data = small_ledger(inputs, oracle_factory)
    data = deepcopy(data)
    mutation(data)
    with pytest.raises(AssertionError, match=pattern):
        audit.check_logical_records(**data)


def test_unused_paid_evidence_and_unrequested_logical_evidence_rejected(inputs, oracle_factory):
    _, data = small_ledger(inputs, oracle_factory)
    altered = deepcopy(data)
    altered['actual_rows'][1] = deepcopy(altered['actual_rows'][0])
    with pytest.raises(AssertionError, match='Unconsumed'):
        audit.check_logical_records(**altered)
    data['requests'] = data['requests'][:2]
    with pytest.raises(AssertionError, match='Unrequested'):
        audit.check_logical_records(**data)


def test_actual_budget_savings_cannot_create_early_logical_budget_denial(inputs, oracle_factory):
    _, data = small_ledger(inputs, oracle_factory)
    data['requests'].append({'request': 5, 'key': 'unseen', 'semantic_cache_hit': False,
        'electrical_reuse': False, 'status': 'BUDGET_UNKNOWN'})
    with pytest.raises(AssertionError, match='Premature'):
        audit.check_logical_records(**data)


def test_full_256_logical_budget_stops_even_when_only_one_point_was_actually_paid(inputs, oracle_factory):
    mech, env, _ = deepcopy(inputs)
    mech.horizon = env.horizon = 256
    build, _ = oracle_factory
    oracle = build(env=env)
    with oracle:
        for index in range(256):
            oracle.certify(env.initial, index * 30)
        with pytest.raises(CLPUBudgetExhausted):
            oracle.certify(env.initial, 256 * 30)
        assert oracle.certify(env.initial, 0)['point_id'] == 0
    _, requests, costs = audit.check_logical_records(**ledger(oracle, mech))
    assert costs['logical_points'] == 256 and costs['physical_points'] == 1
    assert costs['denied'] == costs['semantic_cache_hits'] == 1
    assert requests[-2]['status'] == 'BUDGET_UNKNOWN' and requests[-1]['cache_hit'] is True


@pytest.mark.parametrize('field,value', [('physical_points', 3), ('actual_numerical_entries', 15),
    ('expected_numerical_entries', 15), ('initialization_numerical_entries', 0), ('points', 1),
    ('logical_points', 1), ('statuses', {'PASS': 3}), ('electrical_reuses', 0),
    ('private_cache_only', False), ('cross_arm_cache_reuse', True), ('status', 'STOPPED'), ('closed', False)])
def test_summary_cannot_hide_physics_or_claim_incomplete_evidence_complete(inputs, oracle_factory, field, value):
    oracle, data = small_ledger(inputs, oracle_factory)
    logical, _, costs = audit.check_logical_records(**data)
    summary = oracle.summary()
    summary[field] = value
    with pytest.raises(AssertionError):
        audit.check_wrapper_summary(summary, summary, costs, logical, oracle.scope, oracle.variant, 7)


@pytest.mark.parametrize('failed', [False, True])
def test_both_wrappers_pass_unchanged_independent_complete_window_replay(inputs, oracle_factory, failed):
    semantics, costs = [], []
    for arm in audit.ARMS:
        saved, cost = saved_search(inputs, oracle_factory, arm, failed)
        checked, semantic = audit.check_search(**saved)
        assert checked['best_plan'] == saved['result']['best_plan']
        semantics.append(semantic)
        costs.append(cost)
    assert semantics[0] == semantics[1]
    assert costs[0]['logical_points'] == costs[1]['logical_points']
    assert costs[1]['physical_points'] <= costs[0]['physical_points']
    if not failed:
        assert costs[1]['physical_points'] < costs[0]['physical_points']


@pytest.mark.parametrize('mutation,pattern', [
    (lambda d: d['steps'][0]['oracle_before'].__setitem__('physical_points', 999), 'actual physical cost'),
    (lambda d: d['steps'][0]['oracle_before'].__setitem__('logical_points', 0), 'actual physical cost'),
    (lambda d: d['result']['checkpoints'][0].__setitem__('physical_points', 999), 'paid physical work'),
    (lambda d: d['result']['checkpoints'][0].__setitem__('numerical_entries', 999), 'paid physical work'),
    (lambda d: d['steps'][0]['window_branch_ids'].reverse(), 'window'),
    (lambda d: d['steps'][1]['response'].__setitem__('quality_after', .999), 'response'),
    (lambda d: d['result'].__setitem__('best_plan', None), 'best'),
    (lambda d: d['plans'][0]['validation']['samples'].pop(), 'tail'),
    (lambda d: d['steps'].pop(), 'Unclosed'),
    (lambda d: d['result'].__setitem__('stop_reason', 'PHYSICAL_BUDGET_EXHAUSTED'), 'stop'),
])
def test_logical_projection_preserves_independent_search_mutation_detection(inputs, oracle_factory, mutation, pattern):
    saved, _ = saved_search(inputs, oracle_factory)
    assert saved['steps']
    mutation(saved)
    with pytest.raises(AssertionError, match=pattern):
        audit.check_search(**saved)


def test_disabled_cached_fields_are_not_a_blanket_permission_to_change_raw():
    left = {'elements': [{'name': 'line.l1', 'enabled': False, 'voltages_real_imag': [1., 0.],
        'currents_real_imag': [0., 0.], 'buses': ['s.1', 'b.1'], 'node_order': [1]}]}
    right = deepcopy(left)
    path = ('elements', 0, 'voltages_real_imag', 0)
    assert audit.disabled_unused_difference(path, left, right)
    for field in ('currents_real_imag', 'buses', 'enabled', 'name'):
        assert not audit.disabled_unused_difference(('elements', 0, field), left, right)
    right['elements'][0]['enabled'] = True
    assert not audit.disabled_unused_difference(path, left, right)
    right['elements'][0]['enabled'] = False
    right['elements'][0]['name'] = 'line.other'
    assert not audit.disabled_unused_difference(path, left, right)


def test_difference_enumeration_preserves_numeric_and_structural_changes():
    values = list(audit.differences({'a': [1, 2], 'only_left': 1}, {'a': [1, 3, 4], 'only_right': 1}))
    assert {p for p, _, _ in values} == {('DICT_KEYS',), ('a', 'LIST_LENGTH'), ('a', 1)}


def test_full_historical_scope_review_is_sealed_and_covers_all_538_pairs():
    directory = audit.ROOT / 'artifacts/clpu_equivalent_cache_scope_review_20261030'
    audit.manifest(directory)
    summary = audit.read(directory / 'summary.json')
    assert summary['status'] == 'PASS' and summary['source_completion_sha256'] == audit.SOURCE_SHA
    assert summary['new_physical_points'] == summary['new_training'] == 0
    records = [audit.read(directory / f'context_{i:02d}.json') for i in range(8)]
    pairs = [pair for record in records for pair in record['pairs']]
    assert sum(record['physical_points_reviewed'] for record in records) == 1121
    assert len(pairs) == 538 and sum(not row['raw_equal'] for row in pairs) == 300
    assert all(pair['unchanged_certificate_exactly_equal'] and pair['time_corrected_feedback_exactly_equal'] for pair in pairs)
    assert all(change['disabled_unused_field'] for pair in pairs for change in pair['differences'])
    assert summary['constraint_or_audit_tolerances_changed'] is False


def exact_review_fixture(tmp_path):
    source = audit.ROOT / 'artifacts/clpu_equivalent_cache_exact_key_review_20261030'
    target = tmp_path / 'exact_key_review'
    shutil.copytree(source, target)
    binding = {'path': 'artifacts/clpu_equivalent_cache_exact_key_review_20261030',
        'completion_sha256': audit.sha(target / 'completion_manifest.json'),
        'summary_sha256': audit.sha(target / 'summary.json')}
    audit.save(tmp_path / 'exact_key_review_binding.json', binding)
    prior = audit.sha(audit.ROOT / 'artifacts/clpu_equivalent_cache_scope_review_20261030/completion_manifest.json')
    return target, prior


def test_final_exact_key_review_covers_all_556_pairs_including_18_command_projections(tmp_path):
    target, prior = exact_review_fixture(tmp_path)
    review = audit.check_exact_key_review(tmp_path, prior)
    assert review['physical_points_reviewed'] == 1121 and review['duplicate_pairs_reviewed'] == 556
    pairs = [pair for i in range(8) for pair in audit.read(target / f'context_{i:02d}.json')['pairs']]
    assert len(pairs) == 556 and sum(not p['raw_equal'] for p in pairs) == 354
    merges = [pair for pair in pairs if pair['new_group_merge_beyond_prior_538_review']]
    assert len(merges) == 18
    assert all([d['path'] for d in p['old_group_identity_differences']] == [['command']] for p in merges)
    assert all(p['unchanged_certificate_exactly_equal'] and p['time_corrected_feedback_exactly_equal'] for p in pairs)
    assert all(d['disabled_unused_field'] for p in pairs for d in p['differences'])


def test_final_key_review_must_bind_prior_review_and_every_original_file(tmp_path):
    target, prior = exact_review_fixture(tmp_path)
    with pytest.raises(AssertionError, match='complete raw'):
        audit.check_exact_key_review(tmp_path, '0' * 64)
    (target / 'context_02.json').write_text('{}', encoding='utf-8')
    with pytest.raises(AssertionError):
        audit.check_exact_key_review(tmp_path, prior)


def test_review_source_snapshot_matches_final_key_review_auditor_hash():
    evidence = audit.ROOT / 'artifacts/clpu_equivalent_cache_review_source_20261030'
    audit.manifest(evidence)
    reviewed = audit.read(audit.ROOT / 'artifacts/clpu_equivalent_cache_exact_key_review_20261030/summary.json')
    assert audit.sha(evidence / 'audit_clpu_equivalent_cache.py') == reviewed['auditor_sha256']
    assert audit.sha(evidence / 'review_clpu_equivalent_cache_exact_keys_20261030.py') == reviewed['script_sha256']


def test_historical_source_uses_root_audit_and_nested_cell_seal(tmp_path):
    source, cell, root_sha = historical_source_fixture(tmp_path)
    bound = audit.source_cell_binding(source, 'episodes/000', root_sha)
    assert bound['source_completion_sha256'] == audit.sha(cell / 'completion_manifest.json')
    (cell / 'requests.jsonl').write_bytes(b'changed')
    with pytest.raises(AssertionError, match='scientific file binding'):
        audit.source_cell_binding(source, 'episodes/000', root_sha)


@pytest.mark.parametrize('duplicate,wrong_file,pattern', [(True, False, 'ambiguous'), (False, True, 'scientific file')])
def test_source_binding_requires_unique_audit_and_both_seals(tmp_path, duplicate, wrong_file, pattern):
    source, _, root_sha = historical_source_fixture(tmp_path, duplicate_audit=duplicate, wrong_cell_file=wrong_file)
    with pytest.raises(AssertionError, match=pattern):
        audit.source_cell_binding(source, 'episodes/000', root_sha)


def controller_fixture():
    schedule = [{'cell': f'episodes/{i:03d}'} for i in range(16)]
    checks = [{**row, 'logical_points': 20, 'physical_points': 10, 'numerical_entries': 43,
               'worker_elapsed_seconds': 2.} for row in schedule]
    controller = {'status': 'COMPLETE_CELLS_PENDING_AUDIT', 'errors': [], 'unknown_cost_cells': [],
        'scheduled': 16, 'completed': [row['cell'] for row in schedule],
        'exits': [{**row, 'exit_code': 0, 'process_wall_seconds': 3.} for row in schedule],
        'known_physical_points': 160, 'known_logical_points': 320, 'known_numerical_entries': 688, 'wall_seconds': 25.}
    return controller, schedule, checks


def test_controller_requires_real_counts_and_full_cold_process_wall():
    args = controller_fixture()
    assert audit.check_controller(*args) == (320, 160, 688)
    assert all(row['process_wall_seconds'] == 3. for row in args[2])


@pytest.mark.parametrize('mutation', [
    lambda c: c.__setitem__('known_numerical_entries', 640),
    lambda c: c.__setitem__('known_physical_points', 320),
    lambda c: c['exits'][0].__setitem__('process_wall_seconds', 1.),
    lambda c: c['exits'][0].__setitem__('exit_code', 1),
    lambda c: c['completed'].pop(),
    lambda c: c['unknown_cost_cells'].append('episodes/000'),
])
def test_controller_cannot_omit_initialization_failed_cells_or_cold_start(mutation):
    controller, schedule, checks = controller_fixture()
    mutation(controller)
    with pytest.raises(AssertionError):
        audit.check_controller(controller, schedule, checks)


def test_new_auditor_does_not_import_production_cache_search_or_environment():
    tree = ast.parse(Path(audit.__file__).read_text(encoding='utf-8'))
    imported = [node.module or '' for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imported += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert not any(any(term in name for term in ('nature34_clpu_equivalent_cache', 'nature34_clpu_environment',
                    'nature34_clpu_window', 'nature34_clpu_branch', 'opendss', 'run_clpu')) for name in imported)
