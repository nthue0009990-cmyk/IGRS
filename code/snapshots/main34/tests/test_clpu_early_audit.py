"""Independent early-policy and cold-guard mutation tests; no solver calls."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_clpu_early_pilot as audit
from test_nature34_clpu_branch_audit import inputs, LoggedFakeOracle


@pytest.mark.parametrize('phase,choice,method,expected', [
    ('EARLY', 0, 'ppo_raw', 'ppo_raw'), ('EARLY', 15, 'bc_raw', 'bc_raw'),
    ('EARLY', 16, 'ppo_raw', 'beam'), ('EARLY', 255, 'bc_raw', 'beam'),
    ('GUARD', 0, 'ppo_raw', 'beam'), ('GUARD', 15, 'bc_raw', 'beam'),
    ('EARLY', 0, 'beam', 'beam')])
def test_only_first_sixteen_early_choices_use_fixed_selector(phase, choice, method, expected):
    assert audit.expected_method(phase, choice, method) == expected


def certified(energy, actions):
    plan = {'status': 'COMPLETE', 'certified_complete': True, 'actions': actions,
            'validation': {'status': 'COMPLETE', 'energy_kwh': energy}}
    return {'done': True, 'status': 'CERTIFIED_INCUMBENT', 'best_plan': plan}


def test_exact_energy_tie_uses_guard_even_when_early_has_other_priority_advantages():
    early, guard = certified(1., [0]), certified(1., [1])
    result = audit.protected_choice(early, guard, True)
    assert result['selected_phase'] == 'GUARD' and result['best_plan'] == guard['best_plan']
    assert result['guard_established'] and result['gain_over_guard_kwh'] == 0


def test_strict_energy_not_descriptive_tolerance_determines_winner():
    early, guard = certified(1. + 1e-12, [0]), certified(1., [1])
    assert audit.protected_choice(early, guard, True)['selected_phase'] == 'EARLY'
    early['best_plan']['validation']['energy_kwh'] = 1. - 1e-12
    assert audit.protected_choice(early, guard, True)['selected_phase'] == 'GUARD'


def test_unfinished_guard_cannot_establish_floor_or_complete_task():
    guard = certified(1., [1]); guard['done'] = False
    with pytest.raises(AssertionError, match='not completed'):
        audit.protected_choice(certified(2., [0]), guard, True)


def test_unavailable_early_certificate_still_uses_complete_guard():
    early = {'done': True, 'status': 'NO_CERTIFIED_COMPLETE_PLAN', 'best_plan': None}
    assert audit.protected_choice(early, certified(1., [1]), True)['selected_phase'] == 'GUARD'


def test_uncertified_complete_guard_does_not_create_quality_guarantee():
    guard = {'done': True, 'status': 'NO_CERTIFIED_COMPLETE_PLAN', 'best_plan': None}
    assert not audit.protected_choice(certified(1., [0]), guard, True)['guard_established']


class Replay:
    def __init__(self):
        self.obs = {'representation': 'raw', 'candidate_features': [[1.], [2.]],
                    'global_features': [0.], 'mask': [True, True], 'branch_ids': [4, 5]}

    def observation(self, representation):
        assert representation == 'raw'
        return deepcopy(self.obs)

    def window(self):
        return [{}, {}]

    def rule(self, method):
        assert method == 'beam'
        return 0


class Policy:
    def evaluate(self, compact):
        return 1, -.1, .2


def observation_fixture():
    replay = Replay()
    compact = audit.compact_observation(replay.obs)
    spec = {'method': 'ppo_raw', 'checkpoint_sha256': 'fixed'}
    row = {'obs_compact': compact, 'obs_compact_sha256': audit.digest(compact),
           'mask_sha256': audit.digest(compact['mask']), 'method': 'ppo_raw', 'action': 1,
           'branch_id': 5, 'checkpoint_sha256': 'fixed', 'logprob': -.1, 'value': .2}
    return replay, spec, row


def test_current_observation_and_fixed_action_are_independently_checked():
    replay, spec, row = observation_fixture()
    result = audit.check_observation_action(row, replay, 'EARLY', 0, spec, Policy())
    assert result['policy_decision'] and result['compact_exact_hash_equal']


@pytest.mark.parametrize('kind', ['hash', 'future', 'mask', 'action', 'boundary', 'weight'])
def test_compact_binding_current_features_mask_action_and_boundary_mutations_rejected(kind):
    replay, spec, row = observation_fixture()
    choice = 0
    if kind == 'hash': row['obs_compact_sha256'] = 'changed'
    if kind == 'future':
        row['obs_compact']['global_features'][0] = 1.
        row['obs_compact_sha256'] = audit.digest(row['obs_compact'])
    if kind == 'mask':
        row['obs_compact']['mask'][1] = False
        row['obs_compact_sha256'] = audit.digest(row['obs_compact'])
        row['mask_sha256'] = audit.digest(row['obs_compact']['mask'])
    if kind == 'action': row['action'], row['branch_id'] = 0, 4
    if kind == 'boundary': choice = 16
    if kind == 'weight': row['checkpoint_sha256'] = 'other'
    with pytest.raises(AssertionError):
        audit.check_observation_action(row, replay, 'EARLY', choice, spec, Policy())


def test_legitimate_recomputation_roundoff_is_not_claimed_to_have_same_hash():
    replay, spec, row = observation_fixture()
    row['obs_compact']['global_features'][0] = 1e-12
    row['obs_compact_sha256'] = audit.digest(row['obs_compact'])
    checked = audit.check_observation_action(row, replay, 'EARLY', 0, spec, Policy())
    assert not checked['compact_exact_hash_equal']
    assert checked['recorded_compact_sha256'] != checked['independent_compact_sha256']


def test_roundoff_that_changes_fixed_argmax_blocks_the_comparison():
    replay, spec, row = observation_fixture()
    row['obs_compact']['global_features'][0] = 1e-12
    row['obs_compact_sha256'] = audit.digest(row['obs_compact'])

    class SensitivePolicy:
        def evaluate(self, compact):
            return (1 if compact['global_features'][0] else 0), -.1, .2

    with pytest.raises(AssertionError, match='argmax'):
        audit.check_observation_action(row, replay, 'EARLY', 0, spec, SensitivePolicy())


def phase_fixture(inputs, phase='EARLY', learned=False, fail=True):
    from dnr.nature34_clpu_branch import BranchConfig, BranchSearch
    from dnr.nature34_clpu_window import WindowBranchSearch
    from run_clpu_early_pilot import collect_phase
    import time

    class Fake(LoggedFakeOracle):
        @property
        def physical_points(self):
            return self.points

        @property
        def factory(self):
            return self

        def stats(self):
            return {'returned': {'numerical_solution_entries': 3 + 4 * self.points}}

    class Actor:
        def act(self, observation, mask, **kwargs):
            return {'action': 0, 'logprob': -.1, 'value': .2}

        def evaluate(self, compact):
            return 0, -.1, .2

    mech, env, cfg = inputs
    cfg = {**cfg, 'max_generated_candidates': 100, 'max_checked_candidates': 60,
           'max_branch_expansions': 20, 'point_budget': 128}
    oracle = Fake(env, cfg['point_budget'], fail)
    cls = WindowBranchSearch if phase == 'EARLY' else BranchSearch
    search = cls(env, oracle, BranchConfig(**cfg))
    method = 'ppo_raw' if learned else 'beam'
    spec = {'method': method, 'search_phase': phase, 'checkpoint_sha256': 'synthetic' if learned else None,
            'deadline_utc_epoch': time.time() + 60}
    actor = Actor() if learned else None
    decisions = []
    collected = collect_phase(search, spec, actor, None, decisions.append, lambda count: None)
    result = search.result()
    result.update(**collected, logical_points=oracle.points, quality_pct=100 * search.quality,
                  fully_certified=search.incumbent is not None)
    logical_rows = [{'physical_point_id': i} for i in range(oracle.points)]
    return {'mech': mech, 'branch_cfg': cfg, 'phase': phase, 'spec': spec, 'result': result,
            'attempts': result['attempts'], 'decisions': decisions, 'request_rows': oracle.rows,
            'points': oracle.raw, 'logical_rows': logical_rows, 'condition_sha': 'synthetic', 'policy': actor}


@pytest.mark.parametrize('phase,learned,fail', [('EARLY', False, False), ('EARLY', False, True),
    ('EARLY', True, True), ('GUARD', False, True)])
def test_independent_window_or_original_beam_replays_compact_phase(inputs, phase, learned, fail):
    saved = phase_fixture(inputs, phase, learned, fail)
    checked = audit.check_phase_records(**saved)
    assert checked['decisions'] == saved['result']['search_decisions']
    assert checked['actual_and_independent_policy_actions_equal']
    assert len(checked['boundaries']) == checked['decisions'] + 1


@pytest.mark.parametrize('mutation', ['future_cost', 'fake_response', 'different_guard', 'omitted_plan', 'incomplete'])
def test_compact_phase_cannot_hide_future_costs_response_or_guard_changes(inputs, mutation):
    saved = phase_fixture(inputs, 'GUARD')
    if mutation == 'future_cost':
        saved['decisions'][0]['oracle_before']['logical_points'] += 1
        saved['decisions'][1]['oracle_before']['logical_points'] += 1
    if mutation == 'fake_response': saved['decisions'][1]['response']['reward'] += .1
    if mutation == 'different_guard': saved['result']['generation_trace'][0]['origin'] = 'changed'
    if mutation == 'omitted_plan': saved['attempts'].pop()
    if mutation == 'incomplete': saved['decisions'].pop()
    with pytest.raises((AssertionError, IndexError)):
        audit.check_phase_records(**saved)


def cost_fixture(protected=True):
    local = [
        {'phase_choices': 0, 'logical_points': 3, 'physical_points': 2, 'numerical_entries': 11,
         'requests': 4, 'quality_pct': 40., 'fully_certified': True},
        {'phase_choices': 1, 'logical_points': 4, 'physical_points': 3, 'numerical_entries': 15,
         'requests': 5, 'quality_pct': 50., 'fully_certified': True}]
    phases = {'EARLY': {'decisions': 1, 'boundaries': deepcopy(local)}}
    if protected:
        later = deepcopy(local); later[-1]['quality_pct'] = 52.
        phases['GUARD'] = {'decisions': 1, 'boundaries': later}
    checkpoints = []
    for index, (phase, kind, phase_choice, logical, physical, entries, requests, quality, guard) in enumerate([
        ('EARLY', 'INITIAL', 0, 3, 2, 11, 4, 40., False),
        ('EARLY', 'DECISION', 1, 4, 3, 15, 5, 50., False),
        ('EARLY', 'CLOSE', 1, 4, 3, 15, 5, 50., False),
        *([('GUARD', 'INITIAL', 0, 7, 5, 26, 9, 50., False),
           ('GUARD', 'DECISION', 1, 8, 6, 30, 10, 52., False),
           ('GUARD', 'CLOSE', 1, 8, 6, 30, 10, 52., True)] if protected else [])]):
        checkpoints.append({'checkpoint_index': index, 'phase': phase, 'kind': kind,
            'phase_choices': phase_choice, 'search_choices': phase_choice + (phase == 'GUARD'),
            'logical_points': logical, 'physical_points': physical, 'numerical_entries': entries, 'requests': requests,
            'phase_logical_points': logical - (4 if phase == 'GUARD' else 0),
            'phase_physical_points': physical - (3 if phase == 'GUARD' else 0),
            'quality_pct': quality, 'fully_certified': True, 'guard_established': guard,
            'worker_elapsed_seconds': 5. + index, 'elapsed_seconds': float(index)})
    result = {'elapsed_seconds': 12., 'computation_elapsed_seconds': 7., 'preflight_elapsed_seconds': 5.,
              'early_block_checkpoint': deepcopy(checkpoints[1]), 'final_real_cost': deepcopy(checkpoints[-1])}
    return checkpoints, phases, result


@pytest.mark.parametrize('protected', [False, True])
def test_all_boundaries_sum_independent_routes_and_do_not_establish_early_floor(protected):
    rows, phases, result = cost_fixture(protected)
    audit.check_quality_costs(rows, phases, result)
    assert not any(r['guard_established'] for r in rows[:-1])
    assert rows[-1]['numerical_entries'] == (6 if protected else 3) + 4 * rows[-1]['physical_points']


@pytest.mark.parametrize('kind', ['reset', 'one_initialization', 'future_quality', 'early_floor', 'omitted_close', 'omit_preflight', 'block'])
def test_paid_guard_and_causal_curve_boundary_mutations_fail(kind):
    rows, phases, result = cost_fixture()
    if kind == 'reset': rows[3]['physical_points'] = 2
    if kind == 'one_initialization': rows[-1]['numerical_entries'] -= 3
    if kind == 'future_quality': rows[3]['quality_pct'] = 52.
    if kind == 'early_floor': rows[4]['guard_established'] = True
    if kind == 'omitted_close': rows.pop(2)
    if kind == 'omit_preflight': rows[0]['worker_elapsed_seconds'] = 0.
    if kind == 'block': result['early_block_checkpoint'] = deepcopy(rows[0])
    with pytest.raises(AssertionError): audit.check_quality_costs(rows, phases, result)


def test_schedule_and_protocol_match_independent_frozen_design():
    from test_run_clpu_early_pilot import contexts, models, runner
    schedule = audit.expected_schedule(contexts(), models())
    assert schedule == runner.schedule_for(contexts(), models())
    assert audit.protocol_settings() == runner.settings()
    assert len(schedule) == 64 and sum(r['cold_searches'] for r in schedule) == 120
    assert sum(r['physical_point_budget'] for r in schedule) == 30720
    for spec in schedule:
        for phase in ('EARLY', 'GUARD') if spec['protected'] else ('EARLY',):
            assert audit.expected_phase_spec(spec, phase) == runner.phase_spec(spec, phase)


def controller_fixture():
    from test_run_clpu_early_pilot import contexts, models
    schedule = audit.expected_schedule(contexts(), models())
    checks, dispatch, exits = [], [], []
    for i, spec in enumerate(schedule):
        cold = spec['cold_searches']; began = 100. + 2 * spec['pair_order']
        checks.append({'cell': spec['cell'], 'logical_points': 2 * cold, 'physical_points': cold,
            'numerical_entries': 7 * cold, 'requests': 2 * cold, 'cold_searches': cold, 'worker_elapsed_seconds': .5})
        start = {k: spec[k] for k in ('cell', 'pair_order', 'context_order', 'arm_position')}
        start.update(launcher_pid=1000 + i, started_epoch_seconds=began, utc='2026-10-31T00:00:00Z')
        dispatch.append(start)
        exits.append({'cell': spec['cell'], 'pair_order': spec['pair_order'], 'launcher_pid': 1000 + i,
            'started_epoch_seconds': began, 'finished_epoch_seconds': began + 1., 'exit_code': 0,
            'process_wall_seconds': 1., 'drained_after_controller_error': False, 'utc': '2026-10-31T00:00:01Z'})
    controller = {'status': 'COMPLETE_CELLS_PENDING_AUDIT', 'scheduled': 64, 'errors': [], 'unknown_cost_cells': [],
        'completed': [s['cell'] for s in schedule], 'dispatched': schedule, 'exits': exits,
        'known_logical_points': 240, 'known_physical_points': 120, 'known_numerical_entries': 840, 'wall_seconds': 64.}
    return controller, schedule, checks, dispatch


def test_controller_covers_every_cold_phase_and_full_real_process():
    args = controller_fixture()
    totals = audit.check_controller(*args)
    assert totals['physical_points'] == 120 and totals['numerical_entries'] == 840
    assert all(r['process_wall_seconds'] == 1. for r in args[2])


@pytest.mark.parametrize('kind', ['phase_omission', 'cost_omission', 'next_pair_early', 'failed_exit', 'worker_time_omitted'])
def test_controller_does_not_hide_guard_cost_or_failed_exit(kind):
    controller, schedule, checks, dispatch = controller_fixture()
    if kind == 'phase_omission': checks[0]['cold_searches'] -= 1
    if kind == 'cost_omission': controller['known_numerical_entries'] -= 3
    if kind == 'next_pair_early':
        dispatch[2]['started_epoch_seconds'] = 100.1
        controller['exits'][2]['started_epoch_seconds'] = 100.1
    if kind == 'failed_exit': controller['exits'][0]['exit_code'] = 1
    if kind == 'worker_time_omitted': checks[0]['worker_elapsed_seconds'] = 2.
    with pytest.raises(AssertionError): audit.check_controller(controller, schedule, checks, dispatch)


def test_production_merge_matches_audit_exact_ties_and_missing_guard_evidence():
    from dnr.nature34_clpu_early import merge_results
    for early, guard in ((certified(2., [0]), certified(1., [1])),
                         (certified(1., [0]), certified(1., [1])),
                         (certified(1., [0]), {'done': True, 'status': 'INITIAL_FAIL', 'best_plan': None})):
        actual = merge_results(early, guard, protected=True)
        expected = audit.protected_choice(early, guard, True)
        assert actual['selected_phase'] == expected['selected_phase']
        assert actual['best_plan'] == expected['best_plan']
        assert actual['guard_audit']['guard_established'] == expected['guard_established']


def test_real_diagnostic_has_all_slots_and_actual_independent_argmax_agreement():
    folder = audit.ROOT / 'artifacts/clpu_early_opportunities_20261031'
    summary = audit.read(folder / 'summary.json')
    assert summary['status'] == 'COMPLETE_CAUSAL_PREFIX_DIAGNOSTIC'
    assert summary['slots'] == 128 and summary['available_slots'] + summary['unavailable_slots'] == 128
    assert summary['independent_model_argmax_mismatch_positions'] == 0
    assert summary['new_physical_points'] == summary['new_training'] == 0


def test_auditor_does_not_import_new_execution_or_production_search():
    import ast
    tree = ast.parse(Path(audit.__file__).read_text(encoding='utf-8'))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import): imported += [n.name for n in node.names]
        if isinstance(node, ast.ImportFrom) and node.module: imported.append(node.module)
    assert not any(name.startswith('dnr.') or name == 'run_clpu_early_pilot' for name in imported)


def child_process_fixture():
    spec, worker = {'cell': 'episodes/000', 'protected': True}, {'pid': 10, 'started_monotonic': 100., 'started_epoch_seconds': 300.}
    dispatch, exits, lifecycle = [], [], {}
    result = {'elapsed_seconds': 20., 'phases': {}}
    for i, phase in enumerate(('EARLY', 'GUARD')):
        began, ended = 102. + 6 * i, 107. + 6 * i
        launcher, child = 20 + i, 30 + i
        dispatch.append({'phase': phase, 'phase_cell': spec['cell'] + '/' + phase.lower(),
            'task_parent_pid': 10, 'launcher_pid': launcher, 'started_monotonic': began,
            'started_epoch_seconds': 200. + began})
        exits.append({'phase': phase, 'task_parent_pid': 10, 'launcher_pid': launcher,
            'actual_child_pid': child, 'started_monotonic': began, 'finished_monotonic': ended,
            'started_epoch_seconds': 200. + began, 'process_wall_seconds': 5., 'exit_code': 0,
            'phase_completion_sha256': 'phase' + str(i)})
        execution = {'phase': phase, 'task_parent_pid': 10, 'parent_started_monotonic': 100., 'parent_started_epoch_seconds': 300.}
        start = {'phase': phase, 'task_parent_pid': 10, 'pid': child, 'observed_parent_pid': launcher,
                 'worker_elapsed_seconds': began - 100. + .5}
        finish = {'phase': phase, 'status': 'COMPLETE', 'pid': child, 'worker_elapsed_seconds': ended - 100. - .5}
        lifecycle[phase] = (execution, start, finish)
        result['phases'][phase] = {'completion_manifest_sha256': 'phase' + str(i)}
    result['phase_processes'] = deepcopy(exits)
    return spec, worker, result, dispatch, exits, lifecycle


def test_two_real_cold_children_allow_windows_launcher_indirection():
    checked = audit.check_child_processes(*child_process_fixture())
    assert checked['cold_processes'] == 2 and checked['actual_child_pids'] == [30, 31]
    assert checked['serial_real_exits_verified']


@pytest.mark.parametrize('kind', ['same_process', 'parent_runtime', 'early_guard_start', 'exit_failure', 'clock_reset', 'uncounted_child_time'])
def test_child_lifecycles_reject_same_runtime_future_guard_and_incomplete_cost(kind):
    spec, worker, result, dispatch, exits, lifecycle = child_process_fixture()
    if kind == 'same_process':
        lifecycle['GUARD'][1]['pid'] = lifecycle['GUARD'][2]['pid'] = exits[1]['actual_child_pid'] = 30
    if kind == 'parent_runtime':
        lifecycle['EARLY'][1]['pid'] = lifecycle['EARLY'][2]['pid'] = exits[0]['actual_child_pid'] = 10
    if kind == 'early_guard_start':
        dispatch[1]['started_monotonic'] = exits[1]['started_monotonic'] = 106.
    if kind == 'exit_failure': exits[0]['exit_code'] = 1
    if kind == 'clock_reset': lifecycle['GUARD'][0]['parent_started_monotonic'] = 106.
    if kind == 'uncounted_child_time': result['elapsed_seconds'] = 11.
    result['phase_processes'] = deepcopy(exits)
    with pytest.raises(AssertionError): audit.check_child_processes(spec, worker, result, dispatch, exits, lifecycle)


@pytest.mark.parametrize('arm', ['window_reference', 'protected_rule', 'protected_bc', 'protected_ppo'])
def test_actual_new_worker_schema_has_independent_children_and_auditable_parent_curves(tmp_path, monkeypatch, arm):
    from test_run_clpu_early_pilot import worker_fixture, rows, runner
    run, cell, spec = worker_fixture(tmp_path, monkeypatch, arm)
    parent_pid = runner.os.getpid()
    # The production fixture emulates child PIDs in this pytest process. Supply
    # the corresponding live ancestry instead of this pytest's real parent.
    monkeypatch.setattr(runner.os, 'getppid', lambda: parent_pid)
    assert runner.worker(run, cell) == 0
    result, worker = audit.read(cell / 'result.json'), audit.read(cell / 'worker_started.json')
    exits = audit.read(cell / 'phase_process_exits.json')
    phase_results, phase_checks, lifecycles = {}, {}, {}
    for phase in result['phases']:
        folder = cell / phase.lower()
        phase_result = audit.read(folder / 'phase_result.json')
        journal = rows(folder / 'decisions.jsonl.gz')
        boundaries = []
        for index, event in enumerate(journal[::2]):
            if index == 0:
                boundaries.append({'phase_choices': 0, **event['oracle_before'],
                    'quality_pct': 100 * journal[1]['response']['quality_before'], 'fully_certified': True})
            completion = journal[2 * index + 1]
            boundaries.append({'phase_choices': index + 1, **completion['oracle_after'],
                'quality_pct': 100 * completion['response']['quality_after'], 'fully_certified': True})
        assert boundaries  # these four normal fixtures all make a decision
        checked = {'decisions': phase_result['search_decisions'], 'boundaries': boundaries}
        clock = audit.read(folder / 'preflight_boundary.json')
        audit.check_local_checkpoints(rows(folder / 'cost_checkpoints.jsonl'), phase, checked, phase_result, clock)
        phase_results[phase], phase_checks[phase] = phase_result, checked
        lifecycles[phase] = tuple(audit.read(folder / name) for name in
            ('phase_execution.json', 'phase_started.json', 'phase_finished.json'))
    processes = audit.check_child_processes(spec, worker, result, rows(cell / 'phase_dispatch.jsonl'), exits, lifecycles)
    assert processes['cold_processes'] == spec['cold_searches']
    audit.check_result_selection(result, phase_results)
    audit.check_quality_costs(result['checkpoints'], phase_checks, result)
    for phase in phase_checks:
        local = rows(cell / phase.lower() / 'cost_checkpoints.jsonl')
        combined = [r for r in result['checkpoints'] if r['phase'] == phase]
        assert [r['worker_elapsed_seconds'] for r in local] == [r['worker_elapsed_seconds'] for r in combined]
        assert all(r['elapsed_seconds'] == r['worker_elapsed_seconds'] - result['preflight_elapsed_seconds'] for r in combined)
    assert result['preflight_elapsed_seconds'] >= result['model_load_seconds']
    assert result['elapsed_seconds'] >= result['preflight_elapsed_seconds'] + result['computation_elapsed_seconds']
