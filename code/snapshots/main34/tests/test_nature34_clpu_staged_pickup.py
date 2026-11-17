from copy import deepcopy

import pytest

from dnr.nature34_clpu_search import graph_plan
from dnr.nature34_clpu_staged_pickup import generate_staged_pickup_candidates
from test_nature34_clpu_branch import environment


def test_bounded_generator_keeps_whole_loads_and_every_restore_successor():
    env = environment(horizon=30)
    before = deepcopy((env.static, env.loads, env.event, env.initial))
    result = generate_staged_pickup_candidates(env)
    assert result['counts']['raw_proposals'] == result['counts']['unique_legal_candidates'] == 12
    assert result == generate_staged_pickup_candidates(env)
    assert before == (env.static, env.loads, env.event, env.initial)
    assert {c['release_count'] for c in result['candidates'] if c['blocked_count'] == 2} == {0, 1, 2}
    assert {c['timing'] for c in result['candidates']} == {None, 'NATURAL', 'ONE_TAU'}
    for candidate in result['candidates']:
        info = graph_plan(env, env.initial, candidate['actions'])
        assert len(candidate['actions']) == 30
        assert tuple(candidate['prefix']) == info['actions'][:len(candidate['prefix'])]
        assert candidate['origin'] == 'staged_pickup'
        for before_state, action, after in zip(info['states'], info['actions'], info['states'][1:]):
            assert env.energized(before_state.command) <= env.energized(after.command)
            if 1 <= action <= 10 and before_state.command & (1 << (action - 1)):
                assert env.static['dispatch_loads'][action - 1] not in env.energized(before_state.command)
        held = set(candidate['held_out_loads'])
        assert not held.intersection(env.energized(info['states'][-1].command))
        assert all(x['nominal_kw'] == env.loads[x['load_name']]['nominal_kw'] for x in candidate['blocked_loads'])


def test_tau_spacing_is_between_each_release_and_no_horizon_truncation():
    env = environment(horizon=30)
    env.event['tau_seconds'] = 121.
    result = generate_staged_pickup_candidates(env)
    assert result['gap_slots'] == {'NATURAL': 1, 'ONE_TAU': 3}
    for row in result['candidates']:
        if row['timing'] == 'ONE_TAU':
            start = row['blocked_count'] + len(row['switch_route'])
            assert row['actions'][start:start + 2] == [0, 0]
            if row['release_count'] == 2:
                assert row['actions'][start + 3:start + 5] == [0, 0]
    env.event['tau_seconds'] = 1800.
    over = generate_staged_pickup_candidates(env)
    rejected = [r for r in over['proposals'] if r['status'] == 'UNAVAILABLE_HORIZON']
    assert rejected and all(r['actions'] is None and r['planned_prefix_length'] > 30 for r in rejected)
    assert all(len(r['actions']) == 30 for r in over['candidates'])
    assert len(over['proposals']) == 12  # Failed timing proposals are not replaced.


def test_equal_timing_is_explicit_duplicate_and_hold_generated_once():
    env = environment(horizon=30)
    env.event['tau_seconds'] = 20.
    result = generate_staged_pickup_candidates(env)
    assert result['counts']['raw_proposals'] == 12
    assert result['counts']['proposal_status']['DUPLICATE'] == 5
    assert len({tuple(r['actions']) for r in result['candidates']}) == len(result['candidates']) == 7
    assert sum(r['restore_order'] == 'HOLD' for r in result['proposals']) == 2
    assert sum(len(r['alias_proposal_ids']) for r in result['candidates']) == 12


def test_generator_rejects_scope_changes_and_does_not_read_incumbent_or_outcome():
    with pytest.raises(ValueError, match='30/60/30'):
        generate_staged_pickup_candidates(environment(horizon=29))
    env = environment(horizon=30)
    env.incumbent = {'invented_outcome': object()}
    assert generate_staged_pickup_candidates(env)['physical_certificates_created'] == 0


def test_preparation_can_make_bare_reinterruption_route_legal(monkeypatch):
    from dnr import nature34_clpu_staged_pickup as module
    env = environment(horizon=30)
    seed = (9, 19, 19, 19, 9) + (0,) * 25
    graph_plan(env, env.initial, seed)  # Real unchanged JointGraph legality.
    with pytest.raises(ValueError, match='Illegal action'):
        state = env.initial
        for action in (19, 19, 19):
            state = env.advance(state, action)
    monkeypatch.setattr(module, 'initial_candidates', lambda *args, **kwargs: [seed])
    result = generate_staged_pickup_candidates(env)
    route = result['routes'][0]
    assert route['selected'] and route['status'] == 'SELECTED'
    assert route['bare_route_status'] == 'GRAPH_ILLEGAL'
    assert result['candidates']
    assert all(row['blocked_loads'][0]['action'] == 9 for row in result['candidates'])
    assert any(tuple(row['actions']) == seed for row in result['candidates'])
