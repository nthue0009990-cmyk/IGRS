"""Independent replay and tamper checks, exclusively synthetic certificates."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_clpu_window_pilot as audit
from dnr.nature34_clpu_branch import BranchConfig, BranchSearch
from dnr.nature34_clpu_window import WindowBranchSearch
from test_nature34_clpu_branch_audit import inputs, LoggedFakeOracle


def saved(inputs, variant='coverage_v1', fail=True, budget=None):
    mech, env, cfg = inputs
    cfg = deepcopy(cfg)
    if budget is not None:
        cfg['point_budget'] = budget
    oracle = LoggedFakeOracle(env, cfg['point_budget'], fail)
    cls = BranchSearch if variant == 'original' else WindowBranchSearch
    search = cls(env, oracle, BranchConfig(**cfg))
    initial = 1.
    def boundary(i, elapsed):
        return {'search_choices': i, 'physical_points': oracle.points,
                'numerical_entries': 3 + 4 * oracle.points, 'elapsed_seconds': elapsed,
                'quality_pct': 100 * search.quality if search.incumbent is not None else None,
                'fully_certified': search.incumbent is not None}
    checkpoints, steps, n = [boundary(0, initial)], [], 0
    while not search.done:
        ids = search.observation('raw')['branch_ids']
        action = search.rule_action('beam')
        start = {'transition_id': n, 'status': 'STARTED', 'method': 'beam', 'action': action,
                 'branch_id': ids[action], 'window_branch_ids': ids,
                 'oracle_before': {'requests': oracle.requests, 'points': oracle.points},
                 'elapsed_seconds': n * 2 + 2.}
        steps.append(start)
        response = search.step(action)
        steps.append({**start, 'status': 'COMPLETED', 'response': response,
                      'oracle_after': {'requests': oracle.requests, 'points': oracle.points},
                      'elapsed_seconds': n * 2 + 3.})
        n += 1
        checkpoints.append(boundary(n, n * 2 + 1.))
    result = search.result()
    for point in oracle.raw.values():
        point['elapsed_seconds'] = .001
    result.update(spec={'window_variant': variant, 'method': 'beam'}, search_decisions=n,
        completed_expansions=n, initial_elapsed_seconds=initial, elapsed_seconds=n * 2 + 2.,
        checkpoints=checkpoints, wait_baseline_quality_pct=checkpoints[0]['quality_pct'],
        counts={'generated_candidates': len(search._branch_keys),
                'graph_assessed_candidates': len(search._assessed), 'checked_candidates': len(search._checked),
                'branch_expansions': search._beam_turn, 'generation_cap_rejections': search.generation_cap_rejections})
    return {'mech': mech, 'branch_cfg': cfg, 'result': result, 'attempts': result['attempts'],
            'steps': steps, 'request_rows': oracle.rows, 'points': oracle.raw, 'condition_sha': 'synthetic'}


@pytest.mark.parametrize('variant,fail', [('original', False), ('original', True),
                                         ('coverage_v1', False), ('coverage_v1', True)])
def test_independent_cold_rule_replay_matches_full_attempts(inputs, variant, fail):
    data = saved(inputs, variant, fail)
    checked = audit.check_records(**data)
    assert checked['decisions'] == data['result']['completed_expansions']
    assert checked['best_plan'] == data['result']['best_plan']


def test_incomplete_bootstrap_has_no_certified_quality(inputs):
    data = saved(inputs, budget=2)
    checked = audit.check_records(**data)
    assert checked['best_plan'] is None
    assert checked['quality_checkpoints'][0]['quality_pct'] is None


@pytest.mark.parametrize('mutation,label', [
    (lambda d: d['steps'][0]['window_branch_ids'].reverse(), 'window'),
    (lambda d: d['steps'][1]['response'].__setitem__('quality_after', .999), 'response'),
    (lambda d: d['steps'][1]['oracle_after'].__setitem__('points', 0), 'counts'),
    (lambda d: d['steps'][1].__setitem__('elapsed_seconds', 0.), 'Elapsed'),
    (lambda d: d['result']['checkpoints'][0].__setitem__('physical_points', 0), 'checkpoints'),
    (lambda d: d['result']['checkpoints'][0].__setitem__('numerical_entries', 0), 'checkpoints'),
    (lambda d: d['result']['counts'].__setitem__('generated_candidates', 0), 'counts'),
    (lambda d: d['result'].__setitem__('elapsed_seconds', 0.), 'omits'),
    (lambda d: d['steps'].pop(), 'Unclosed'),
    (lambda d: d['attempts'][0]['validation']['samples'].pop(), 'tail'),
    (lambda d: d['result'].__setitem__('best_plan', None), 'best'),
])
def test_corrupt_evidence_rejected(inputs, mutation, label):
    data = saved(inputs)
    assert data['steps']
    mutation(data)
    with pytest.raises(AssertionError, match=label):
        audit.check_records(**data)


def test_extra_unconsumed_request_rejected(inputs):
    data = saved(inputs)
    data['request_rows'].append(deepcopy(data['request_rows'][-1]))
    with pytest.raises(AssertionError, match='Unbound'):
        audit.check_records(**data)


def test_checkpoint_may_include_completion_journal_cost(inputs):
    data = saved(inputs)
    for item in data['result']['checkpoints'][1:]:
        item['elapsed_seconds'] += .1
    checked = audit.check_records(**data)
    assert checked['quality_checkpoints'] == data['result']['checkpoints']


def test_quality_reach_time_cannot_precede_certificate(inputs):
    data = saved(inputs)
    data['result']['checkpoints'][1]['elapsed_seconds'] -= .1
    with pytest.raises(AssertionError, match='precedes'):
        audit.check_records(**data)


def test_claimed_time_cannot_omit_known_paid_physical_durations(inputs):
    data = saved(inputs)
    data['points'][0]['elapsed_seconds'] = 10000.
    with pytest.raises(AssertionError, match='omits paid physical'):
        audit.check_records(**data)


def test_population_pairs_are_fixed_training_contexts_and_both_variants():
    training = [{'scenario_id': f'fault_{i}', 'event_id': event} for i in range(8) for event in (3, 6, 20, 30)]
    faults = sorted({r['scenario_id'] for r in training},
                    key=lambda s: audit.hashlib.sha256(('route-b-s1-fault-v1|' + s).encode()).hexdigest())
    events = sorted({r['event_id'] for r in training},
                    key=lambda e: audit.hashlib.sha256(('route-b-s1-event-v1|' + str(e)).encode()).hexdigest())
    population = {'contexts': [{'scenario_id': s, 'event_id': events[i % 4]} for i, s in enumerate(faults)]}
    schedule = [{'context_id': i, **context, 'window_variant': variant, 'group': group,
                 'method': 'beam', 'point_budget': 256, 'seed': 0, 'cell': f'episodes/{2*i+j:03d}'}
                for i, context in enumerate(population['contexts']) for j, (variant, group) in enumerate(audit.GROUPS.items())]
    audit.check_population(population, training, schedule)
    schedule[1]['window_variant'] = 'original'
    with pytest.raises(AssertionError, match='paired method'):
        audit.check_population(population, training, schedule)


def test_unpaid_cache_hit_rejected(inputs):
    data = saved(inputs)
    data['request_rows'][0] = {**data['request_rows'][0], 'cache_hit': True}
    with pytest.raises(AssertionError, match='Unpaid'):
        audit.check_records(**data)


def test_independent_window_filters_checked_ties_and_duplicate_action_tuples():
    replay = object.__new__(audit.WindowReplay)
    replay.turn, replay.best = 1, {'validation': {'energy_kwh': 5.}}
    replay.cfg, replay.checked = {'action_window': 3}, {(8, 0): {}}
    def row(i, actions, depth=1, energy=6., origin='prefix_wait'):
        return {'id': i, 'prefix': tuple(range(depth)), 'origin': origin,
                'info': {'actions': actions, 'energy_kwh': energy, 'priority': (-energy, i)}}
    replay.frontier = {r['id']: r for r in [row(0, (8, 0)), row(1, (1, 0), energy=5.),
        row(2, (2, 0), depth=2), row(3, (3, 0)), row(4, (2, 0), depth=2),
        row(5, (5, 0), depth=2), row(6, (6, 0), depth=2, origin='generic_wait')]}
    before = deepcopy(replay.frontier)
    assert [r['id'] for r in replay.window()] == [2, 5]
    assert replay.frontier == before
    replay.turn = 2
    assert replay.window() == audit.branch.BranchReplay.window(replay)
    replay.turn, replay.best = 1, None
    assert replay.window() == audit.branch.BranchReplay.window(replay)


def controller_fixture():
    schedule = [{'cell': f'episodes/{i:03d}'} for i in range(16)]
    checks = [{'cell': row['cell'], 'points': 2, 'commands': {'numerical_entries': 11}} for row in schedule]
    controller = {'status': 'COMPLETE_CELLS_PENDING_AUDIT', 'errors': [], 'unknown_cost_cells': [],
        'completed': [row['cell'] for row in schedule], 'known_physical_points': 32,
        'known_numerical_entries': 176, 'wall_seconds': 3.,
        'exits': [{'cell': row['cell'], 'exit_code': 0, 'launcher_pid': 100+i} for i, row in enumerate(schedule)]}
    return controller, schedule, checks


def test_controller_cost_and_all_real_worker_exits_required():
    args = controller_fixture()
    assert audit.check_controller(*args) == (32, 176)
    args[0]['exits'].pop()
    with pytest.raises(AssertionError, match='exits'):
        audit.check_controller(*args)


def test_controller_cannot_hide_failed_worker_or_extra_cost():
    args = controller_fixture()
    args[0]['exits'][3]['exit_code'] = 1
    with pytest.raises(AssertionError, match='exits'):
        audit.check_controller(*args)
    args = controller_fixture()
    args[0]['known_numerical_entries'] -= 3
    with pytest.raises(AssertionError, match='cost'):
        audit.check_controller(*args)
