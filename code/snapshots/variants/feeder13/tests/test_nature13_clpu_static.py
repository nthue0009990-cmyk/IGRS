"""Actual-source pure software checks; no physical runtime or engine import."""
from copy import deepcopy
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys

import networkx as nx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_nature13_clpu_static import load_inputs, make_env, build_environment
from dnr.nature13_clpu_graph import state_metadata
from dnr.nature13_clpu_staged_pickup import generate_staged_pickup_candidates
from dnr.nature13_clpu_search import graph_plan


@pytest.fixture(scope='module')
def inputs():
    return load_inputs()


def environment(inputs, fault_index=0, event_index=0):
    static, loads, condition, scenarios, events, _ = inputs
    return make_env(static, loads, condition, scenarios[fault_index], events[event_index])


def independent_metadata(env, state):
    """Direct source graph construction, independent of graph lookup tables."""
    switch_mask, ready = state >> 7, state & 127
    if switch_mask & 4:
        return None
    graph = nx.Graph()
    graph.add_nodes_from(env.static['nodes'])
    graph.add_edges_from(env.static['bus_edges'])
    fault_edges = [env.static['original_lines'][name] for name in env.scenario['fault_lines']]
    if any((switch_mask >> i) & 1 and frozenset(edge) in {frozenset(e) for e in fault_edges}
           for i, edge in enumerate(env.static['switch_edges'])):
        return None
    graph.remove_edges_from(fault_edges)
    graph.remove_edges_from(edge for i, edge in enumerate(env.static['switch_edges']) if not (switch_mask >> i) & 1)
    if not nx.is_forest(graph):
        return None
    powered = nx.node_connected_component(graph, env.static['source_bus'])
    dispatch = {name: i for i, name in enumerate(env.static['dispatch_loads'])}
    return frozenset(name for name, row in env.loads.items()
                     if row['bus'] in powered and (name not in dispatch or (ready >> dispatch[name]) & 1))


@pytest.mark.parametrize('fault_index', range(3))
def test_complete_joint_graph_matches_independent_enumeration(inputs, fault_index):
    env = environment(inputs, fault_index)
    inventory = {state: independent_metadata(env, state) for state in range(2048)}
    reached, pending = {env.initial.command}, [env.initial.command]
    while pending:
        state = pending.pop()
        expected = {state}
        for bit in range(11):
            other = state ^ (1 << bit)
            if inventory[other] is not None and inventory[state] <= inventory[other]:
                expected.add(other)
                if other not in reached:
                    reached.add(other)
                    pending.append(other)
        assert set(env.graph.successors(state)) == expected
        assert env.energized(state) == inventory[state]
        assert env.graph.served_base_kw(state) == sum(env.loads[name]['nominal_kw'] for name in inventory[state])
    assert reached == set(env.graph.states)
    assert all(not (int(state) >> 9) & 1 for state in env.graph.states)


@pytest.mark.parametrize('fault_index', range(3))
def test_integer_bellman_route_matches_exhaustive_short_paths(inputs, fault_index):
    env = environment(inputs, fault_index)
    paths = [(env.initial.command,)]
    for _ in range(3):
        paths = [path + (other,) for path in paths for other in env.graph.successors(path[-1])]
    def objective(path):
        return (-sum(env.graph.served_weight(s) for s in path[1:]),
                -env.graph.served_weight(path[-1]), sum(a != b for a, b in zip(path, path[1:])), path)
    assert env.graph.solve(env.initial.command, 3)['path'] == list(min(paths, key=objective))


def test_history_is_actual_pickup_not_readiness_change(inputs):
    env = environment(inputs)
    name = 'Load.634a'
    bit = env.static['dispatch_loads'].index(name) + 1
    before = env.initial
    assert name not in env.energized(before.command)
    prepared = env.advance(before, bit)
    waited = env.advance(prepared, 0)
    connected = env.advance(waited, 11)
    assert name not in env.energized(connected.command)
    assert env.multipliers(connected, 120)[name] == 1
    restored = env.advance(connected, bit)
    assert restored.pickups[env.load_names.index(name)] == 180
    assert env.multipliers(restored, 180)[name] == pytest.approx(1 + env.event['a'])
    assert env.multipliers(restored, 180 + env.event['tau_seconds'])[name] == pytest.approx(1 + env.event['a'] / math.e)
    assert all(env.multipliers(restored, 180)[n] == 1 for n in env.initial_energized)
    with pytest.raises(ValueError, match='Illegal action'):
        env.advance(restored, bit)
    with pytest.raises(ValueError, match='Illegal action'):
        env.advance(before, 10)


@pytest.mark.parametrize('fault_index,event_index', itertools.product(range(3), range(2)))
def test_original_staged_pools_preserve_all_proposals_and_full_history(inputs, fault_index, event_index):
    env = environment(inputs, fault_index, event_index)
    snapshot = deepcopy((env.static, env.loads, env.event, env.initial))
    pool = generate_staged_pickup_candidates(env)
    assert pool == generate_staged_pickup_candidates(env)
    assert snapshot == (env.static, env.loads, env.event, env.initial)
    assert 0 < len(pool['candidates']) <= 24
    assert len({tuple(c['actions']) for c in pool['candidates']}) == len(pool['candidates'])
    assert sum(len(c['alias_proposal_ids']) for c in pool['candidates']) == sum(p['status'] in ('LEGAL_UNIQUE', 'DUPLICATE') for p in pool['proposals'])
    for candidate in pool['candidates']:
        info = graph_plan(env, env.initial, candidate['actions'])
        assert len(candidate['actions']) == 30
        assert not set(candidate['held_out_loads']) & env.energized(info['states'][-1].command)
        for before, after in zip(info['states'], info['states'][1:]):
            assert env.energized(before.command) <= env.energized(after.command)
            for name, old, new in zip(env.load_names, before.pickups, after.pickups):
                assert new == ((after.step - 1) * 60 if old is None and name in env.energized(after.command) else old)
        if candidate['timing'] == 'ONE_TAU':
            first_release = candidate['blocked_count'] + len(candidate['switch_route'])
            gap = pool['gap_slots']['ONE_TAU']
            assert candidate['actions'][first_release:first_release + gap - 1] == [0] * (gap - 1)


def test_duplicate_and_horizon_edges_keep_original_rules(inputs):
    env = environment(inputs)
    # Synthetic event values below are unit-test fixtures, never saved scientific inputs.
    env.event['tau_seconds'] = 20.0
    pool = generate_staged_pickup_candidates(env)
    assert pool['counts']['proposal_status']['DUPLICATE'] > 0
    env.event['tau_seconds'] = 1800.0
    pool = generate_staged_pickup_candidates(env)
    rejected = [r for r in pool['proposals'] if r['status'] == 'UNAVAILABLE_HORIZON']
    assert rejected and all(r['actions'] is None and r['planned_prefix_length'] > 30 for r in rejected)


def test_full_wait_logical_coverage_uses_only_constant_actual_inputs(inputs):
    env = environment(inputs)
    requests = []
    class UnitStubOnly:
        def certify(self, state, seconds):
            requests.append(env.snapshot_request(state, seconds))
            return {'point_id': len(requests) - 1, 'status': 'PASS', 'reliable': True}
    result = env.validate_plan(env.initial, [0] * 30, UnitStubOnly())
    assert len(result['samples']) == len(requests) == 90
    physical = [{key: request[key] for key in ('action', 'case', 'alpha', 'algorithm', 'multipliers')} for request in requests]
    assert all(row == physical[0] for row in physical)
    assert len(requests[0]['multipliers']) == 15
    assert env.total_kw == 3466
    assert requests[0]['action'][4:] == [1] * 7
    assert requests[0]['action'] == env.metadata(env.initial.command)['command']
    assert requests[0]['effective_action'] == env.graph.action(env.initial.command)
    assert requests[0]['effective_action'] != requests[0]['action']


def test_import_and_helper_never_import_physics():
    code = '''
import importlib.abc, sys
from pathlib import Path
class NoPhysics(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('opendssdirect', 'dss', 'Environments')) or fullname.endswith(('_runtime', '_physical')):
            raise AssertionError('Forbidden physical import: ' + fullname)
sys.meta_path.insert(0, NoPhysics())
sys.path[:0]=['src','scripts']
from prepare_nature13_clpu_static import build_environment
from dnr.nature13_clpu_guard_first_control import GuardFirstStagedSearch
from dnr.nature13_clpu_lazysp import LazySPCertificationSearch
env=build_environment(Path('.'), 'line.632633', 3)
assert len(env.load_names)==15
'''
    result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


class CountedUnitOracle:
    """Synthetic labels for software boundary tests; not saved physics evidence."""
    def __init__(self, env, budget=256):
        self.env, self.budget = env, budget
        self.points = self.requests = 0
        self.cache = {}

    @property
    def physical_points(self):
        return self.points

    def certify(self, state, seconds):
        from dnr.nature13_clpu_environment import CLPUBudgetExhausted
        self.requests += 1
        key = json.dumps(self.env.identity(state, seconds), sort_keys=True)
        if key in self.cache:
            return self.cache[key]
        if self.points >= self.budget:
            raise CLPUBudgetExhausted('Unit-test budget')
        value = {'point_id': self.points, 'status': 'PASS', 'reliable': True, 'feedback': []}
        self.points += 1
        self.cache[key] = value
        return value


def unit_wait_donor(env):
    from dnr.nature13_clpu_branch import BranchSearch
    guard = BranchSearch(env, CountedUnitOracle(env))
    guard._stop('PHYSICAL_BUDGET_EXHAUSTED')
    return guard.result()


@pytest.mark.parametrize('arm', ['P', 'E', 'Forward'])
def test_public_search_adapters_keep_pool_budget_and_complete_guard(inputs, arm):
    from dnr.nature13_clpu_branch import BranchConfig
    from dnr.nature13_clpu_guard_first_control import GuardFirstStagedSearch, GuardFirstMacroController
    from dnr.nature13_clpu_staged_control import StagedCertificationSearch
    from dnr.nature13_clpu_staged_energy_control import EnergyStagedMacroController
    from dnr.nature13_clpu_lazysp import LazySPCertificationSearch
    from dnr.nature34_clpu_early import merge_results
    env = environment(inputs)
    guard = unit_wait_donor(env)
    reference = {'phase': 'GUARD', 'file': 'GUARD/phase_result.json', 'sha256': 'a' * 64}
    oracle = CountedUnitOracle(env)
    if arm == 'P':
        search = GuardFirstStagedSearch(env, oracle, guard_result=guard, guard_reference=reference)
        control = GuardFirstMacroController(search)
    else:
        search = StagedCertificationSearch(env, oracle)
        control = EnergyStagedMacroController(search) if arm == 'E' else None
    assert search.stage_generation == generate_staged_pickup_candidates(env)
    if arm == 'Forward':
        lazy = LazySPCertificationSearch(search, guard_result=guard, guard_reference=reference, selector='forward')
        result = lazy.run()
        assert len(lazy.selected) <= 16
    else:
        proposal = control.prepare()
        if proposal is not None:
            assert len(proposal['observation']['candidate_feature_names']) == 19 + 30 + 3 * 15
        while proposal is not None:
            control.apply(proposal)
            proposal = control.prepare()
        result = control.result()
        assert search.verify_attempts <= 16
    assert oracle.points <= 256
    merged = merge_results(result, guard, protected=True)
    assert merged['selected_phase'] in ('EARLY', 'GUARD')
    complete = result['best_plan'] or guard['best_plan']
    assert len(complete['validation']['samples']) == 90
    assert search.stage_generation['counts']['unique_legal_candidates'] == 12


def test_forward_budget_stop_preserves_unknown_and_guard(inputs):
    from dnr.nature13_clpu_branch import BranchConfig
    from dnr.nature13_clpu_staged_control import StagedCertificationSearch
    from dnr.nature13_clpu_lazysp import LazySPCertificationSearch
    from dnr.nature34_clpu_early import merge_results
    env = environment(inputs)
    guard, oracle = unit_wait_donor(env), CountedUnitOracle(env, 65)
    bootstrap = StagedCertificationSearch(env, oracle, BranchConfig(point_budget=65))
    search = LazySPCertificationSearch(bootstrap, guard_result=guard,
        guard_reference={'phase': 'GUARD', 'file': 'GUARD/phase_result.json', 'sha256': 'a' * 64}, selector='forward')
    result = search.run()
    assert oracle.points == 65
    assert result['stop_reason'] == 'PHYSICAL_BUDGET_EXHAUSTED'
    assert any(edge['status'] == 'UNKNOWN' for edge in search.edges)
    assert merge_results(result, guard, protected=True)['selected_phase'] == 'GUARD'


def test_feedback_source_path_uses_seven_ready_bit_offset(inputs):
    from dnr.nature13_clpu_search import _feedback_exposure
    env = environment(inputs)
    restored = env.advance(env.initial, 11)
    indices, available = _feedback_exposure(env, restored, {'violations': [{'bus': '633'}]})
    assert available
    assert {env.load_names[i] for i in indices} == {'Load.634a', 'Load.634b', 'Load.634c'}
