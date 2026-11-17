"""Window-policy tests; synthetic oracle only, with zero physical solves."""
from copy import deepcopy

import pytest

from dnr.nature34_clpu_branch import BranchConfig, BranchSearch
from dnr.nature34_clpu_window import WINDOW_RULE, WINDOW_RULE_VERSION, WindowBranchSearch
from test_nature34_clpu_branch import CountedOracle, environment


def row(identifier, prefix, *, energy=2., origin='prefix_wait', actions=None):
    prefix = tuple(prefix)
    actions = tuple(actions) if actions is not None else prefix + (0,) * (6 - len(prefix))
    return {'id': identifier, 'prefix': prefix, 'origin': origin,
            'info': {'actions': actions, 'energy_kwh': energy,
                     'priority': (-energy, -energy, len(prefix), actions)}}


def shell(rows, *, turn=0, energy=1., size=32):
    search = WindowBranchSearch.__new__(WindowBranchSearch)
    search.config = BranchConfig(action_window=size, beam_width=min(8, size))
    search._frontier = {item['id']: item for item in rows}
    search._checked = {}
    search._beam_turn = turn
    search.incumbent = {'validation': {'energy_kwh': energy}}
    search.done = False
    return search


def ids(search):
    return [item['id'] for item in search._window()]


def selection_observation(search, exposures=None):
    exposures = exposures or {}
    return {'candidate_feature_names': ['cold_excess_exposed_weight'],
            'candidate_features': [[exposures.get(item['id'], 0.)]
                                   for item in search._window()]}


@pytest.mark.parametrize('turn', [0, 1, 2, 3])
@pytest.mark.parametrize('size', [1, 8, 32])
def test_window_deterministic_bounded_and_preserves_entire_frontier(turn, size):
    rows = [row(i, (i + 1,), energy=float(100 - i)) for i in range(75)]
    rows += [row(100 + i, item['prefix'], actions=item['info']['actions'],
                 energy=item['info']['energy_kwh']) for i, item in enumerate(rows[:12])]
    search = shell(rows, turn=turn, size=size)
    original = deepcopy(search.__dict__)
    selected = search._window()
    assert 0 < len(selected) <= size <= 32
    if turn % 2:
        assert len({item['info']['actions'] for item in selected}) == len(selected)
    else:
        assert selected == BranchSearch._window(search)
    assert ids(search) == ids(search)
    assert search.__dict__ == original
    search._frontier = dict(reversed(list(search._frontier.items())))
    assert ids(search) == [item['id'] for item in selected]


def test_ordinary_window_is_exact_original_including_fifo_repair_parent():
    rows = [row(i, (i + 1, 0), energy=100. - i, origin='generic_replace',
                actions=(i + 1, 1, 0, 0, 0, 0)) for i in range(20, 60)]
    rows += [row(i, (i + 1,), energy=3.) for i in range(10, 18)]
    rows += [row(2, (2,), energy=.2, origin='generic_swap')]
    search = shell(rows)
    selected = search._window()
    assert [item['id'] for item in selected[:16]] == list(range(20, 36))
    assert {item['id'] for item in selected[16:24]} == set(range(10, 18))
    assert selected == BranchSearch._window(search)
    assert selected[24]['id'] == 2
    assert len(selected) == 32


def test_reserved_deep_hold_is_selected_on_coverage_turn_despite_shallow_leaders():
    shallow = [row(i, (i + 1,), energy=100., origin='prefix_graph_completion',
                   actions=(i + 1, 9, 8, 7, 6, 5)) for i in range(20)]
    hold = row(100, (21, 22), energy=2.)
    search = shell(shallow + [hold])
    search.observation = lambda representation: selection_observation(search)
    ordinary = search._window()
    assert hold in ordinary
    assert ordinary[search.rule_action('beam')]['id'] != hold['id']
    search._beam_turn = 1
    assert ids(search) == [hold['id']]
    assert search._window()[search.rule_action('beam')]['id'] == hold['id']
    assert set(search._frontier) == {item['id'] for item in shallow + [hold]}


def test_coverage_uses_oldest_eligible_depth_not_shallowest_or_highest_quality():
    search = shell([row(5, (1, 2, 3)), row(9, (9,), energy=100.),
                    row(6, (4, 5, 6)), row(7, (7, 8)),
                    row(1, (1,), energy=.5)], turn=1)
    assert ids(search) == [5, 6]
    search._checked[search._frontier[5]['info']['actions']] = {'status': 'FAIL'}
    assert ids(search) == [6]
    search._checked[search._frontier[6]['info']['actions']] = {'status': 'FAIL'}
    assert ids(search) == [7]


def test_coverage_uniqueness_does_not_delete_duplicate_expansion_parents():
    first = row(1, (3, 4))
    duplicate = row(2, (3, 4, 0), actions=first['info']['actions'])
    search = shell([first, duplicate], turn=1)
    assert ids(search) == [1]
    del search._frontier[1]
    search._beam_turn = 2
    search._checked[first['info']['actions']] = {'status': 'FAIL'}
    assert ids(search) == [2]


def test_absent_eligible_coverage_falls_back_to_the_same_ordinary_window():
    search = shell([row(1, (1,), energy=1.),
                    row(2, (2,), energy=100., origin='generic_replace'),
                    row(3, (3,), energy=50.)])
    search._checked[search._frontier[3]['info']['actions']] = {'status': 'FAIL'}
    ordinary = ids(search)
    search._beam_turn = 1
    assert ids(search) == ordinary
    assert search._window() == BranchSearch._window(search)
    search.incumbent = None
    assert ids(search) == ordinary


def test_no_future_evidence_or_feedback_is_read_by_exposure():
    class OpaqueEvidence:
        def __getattribute__(self, name):
            raise AssertionError('Window read unobserved or irrelevant evidence: ' + name)

    class MembershipOnly(dict):
        def __getitem__(self, key):
            raise AssertionError('Checked result contents must not enter window selection')

        def get(self, key, default=None):
            raise AssertionError('Checked result contents must not enter window selection')

    search = shell([row(1, (1, 2)), row(2, (3, 4)), row(3, (5, 6))], turn=1)
    search._checked = MembershipOnly({search._frontier[1]['info']['actions']: OpaqueEvidence()})
    search.env = search.oracle = search.latest_failure = OpaqueEvidence()
    search.future_outcomes = OpaqueEvidence()
    assert ids(search) == [2, 3]
    search._beam_turn = 0
    assert ids(search) == [1, 2, 3]


def test_original_beam_top_eight_and_observed_exposure_ties_are_retained():
    search = shell([row(i, (i + 1, 1), energy=20. - i) for i in range(12)], turn=1)
    search.observation = lambda representation: selection_observation(
        search, {i: (0. if i in (6, 10) else 1.) for i in range(12)})
    selected = search._window()[search.rule_action('beam')]
    assert selected['id'] == 6


def test_all_nonwindow_methods_are_the_original_engine_methods():
    for name in ('__init__', '_add', '_assess', '_expand_prefix', '_evaluate',
                 '_refresh_stop', 'rule_action', 'step', 'observation', 'result'):
        assert getattr(WindowBranchSearch, name) is getattr(BranchSearch, name)
    assert WINDOW_RULE['version'] == WINDOW_RULE_VERSION


def test_independent_cold_engines_replay_paid_logic_without_shared_cache():
    left_env, right_env = environment(), environment()
    left_oracle, right_oracle = CountedOracle(left_env, 80), CountedOracle(right_env, 80)
    config = BranchConfig(point_budget=80, max_generated_candidates=100,
                          max_branch_expansions=24)
    left = WindowBranchSearch(left_env, left_oracle, config)
    right = WindowBranchSearch(right_env, right_oracle, config)
    assert left_oracle.cache is not right_oracle.cache
    while not left.done:
        assert left.observation('associated') == right.observation('associated')
        chosen = left.rule_action('beam')
        assert chosen == right.rule_action('beam')
        assert left.step(chosen) == right.step(chosen)
    assert right.done
    assert left.result() == right.result()
    assert 0 < left_oracle.points == right_oracle.points <= 80
    result = left.result()
    assert result['incumbent']['certified_complete']
    assert result['incumbent']['status'] == 'COMPLETE'
    assert len(result['trace']) <= 24
    assert all(not attempt['certified_complete']
               for attempt in result['attempts'] if attempt['status'] == 'BUDGET_EXHAUSTED_PARTIAL')
