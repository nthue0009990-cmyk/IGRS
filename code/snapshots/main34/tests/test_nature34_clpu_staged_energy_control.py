"""Targeted public-selector checks; no solver or training is run."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from dnr.nature34_clpu_branch import BranchConfig, BranchSearch
from dnr.nature34_clpu_staged_control import StagedMacroController, StagedCertificationSearch
from dnr.nature34_clpu_staged_energy_control import EnergyStagedMacroController, complete_energy_action, VERSION, RULE
from test_nature34_clpu_deferred import ActualCountedOracle
from test_nature34_clpu_deferred_control import LastSlotPolicy
from test_nature34_clpu_staged_control import make

SOURCE = ROOT / 'runs/20261031T085600_616395Z_clpu_staged_pilot'
SOURCE_SHA = '0d288959bece92f5e274ce43e0c2d29520fe15094dbc8663ebcc3a8e3a3ceefb'


def observation(rows, mask=None):
    return {'branches': rows, 'branch_ids': [r['branch_id'] for r in rows],
            'mask': [True] * len(rows) if mask is None else mask}


def test_complete_energy_beats_shorter_prefix_and_uses_exact_smallest_id_tie():
    obs = observation([{'branch_id': 9, 'graph_energy_kwh': 12., 'prefix': [1] * 10},
                       {'branch_id': 1, 'graph_energy_kwh': 10., 'prefix': [1]}])
    assert complete_energy_action(obs) == 0
    obs['branches'][1]['graph_energy_kwh'] = 12.
    assert complete_energy_action(obs) == 1
    obs['branches'][0]['graph_energy_kwh'] = math.nextafter(12., math.inf)
    assert complete_energy_action(obs) == 0  # No tolerance silently changes an exact ordering.


def test_only_current_masked_window_is_ranked_and_missing_invalid_windows_fail():
    obs = observation([{'branch_id': 4, 'graph_energy_kwh': 999.},
                       {'branch_id': 8, 'graph_energy_kwh': 8.},
                       {'branch_id': 3, 'graph_energy_kwh': 8.}], [False, True, True])
    assert complete_energy_action(obs) == 2
    assert obs['mask'] == [False, True, True]
    for changed in (observation([], []), observation(obs['branches'], [False] * 3),
                    {**obs, 'branch_ids': [4, 8, 8]}):
        with pytest.raises(RuntimeError):
            complete_energy_action(changed)
    invalid = deepcopy(obs); invalid['branches'][1]['graph_energy_kwh'] = float('nan')
    with pytest.raises(RuntimeError, match='Nonfinite'):
        complete_energy_action(invalid)


def test_real_c05_same_first_window_selects_branch_one_without_physical_answers():
    from audit_d059_supplement import NoSolverImports, no_solver_loaded
    from run_nature34_clpu_pilot import make_env
    guard = NoSolverImports(); sys.meta_path.insert(0, guard)
    try:
        assert hashlib.sha256((SOURCE / 'completion_manifest.json').read_bytes()).hexdigest() == SOURCE_SHA
        schedule = json.loads((SOURCE / 'schedule.json').read_text(encoding='utf8'))
        spec = next(r for r in schedule if r['context_id'] == 5 and r['method'] == 'beam')
        env = make_env(SOURCE, spec)
        oracle = ActualCountedOracle(env, budget=256)
        search = StagedCertificationSearch(env, oracle, BranchConfig())
        old, new = StagedMacroController(search), EnergyStagedMacroController(search)
        before = oracle.points, oracle.requests, deepcopy(search.stage_generation)
        first_old, first_new = old.prepare(), new.prepare()
        assert first_old['observation'] == first_new['observation']
        assert first_old['window_mapping'] == first_new['window_mapping']
        assert first_old['branch_id'] == 0 and first_new['branch_id'] == 1
        assert first_new['actions'] == [1, 18, 1] + [0] * 27
        assert before == (oracle.points, oracle.requests, search.stage_generation)
        no_solver_loaded()
    finally:
        sys.meta_path.remove(guard)


def test_fixed_bc_proposals_and_trajectory_are_unchanged_with_new_wrapper():
    left, left_oracle = make(); right, right_oracle = make()
    old = StagedMacroController(left, method='bc_raw', policy=LastSlotPolicy(left))
    new = EnergyStagedMacroController(right, method='bc_raw', policy=LastSlotPolicy(right))
    while (before := old.prepare()) is not None:
        after = new.prepare()
        assert before == after
        old.apply(before); new.apply(after)
    assert new.prepare() is None
    assert left.incumbent == right.incumbent
    assert left._checked == right._checked
    assert (left_oracle.points, left_oracle.requests) == (right_oracle.points, right_oracle.requests)
    result = new.result()
    assert result['control']['selector_version'] == VERSION
    assert result['control']['beam_rule'] == RULE and result['control']['active_selector'] == 'fixed_raw_actor'


def test_common_coverage_exposure_and_original_guard_mechanics_are_not_replaced():
    search, oracle = make()
    old, new = StagedMacroController(search), EnergyStagedMacroController(search)
    for turn in (0, 1):
        search.verify_attempts = turn
        assert old.adapter.observation('raw') == new.adapter.observation('raw')
        assert old.adapter.window_mapping() == new.adapter.window_mapping()
    # The new selector wraps only EARLY. Its module never substitutes the guard
    # class, whose frozen source bytes must remain exactly the sealed version.
    manifest = json.loads((SOURCE / 'completion_manifest.json').read_text(encoding='utf8'))
    manifest = {k.replace('\\', '/'): v for k, v in manifest.items()}
    assert hashlib.sha256((ROOT / 'src/dnr/nature34_clpu_branch.py').read_bytes()).hexdigest() == manifest['source/src/dnr/nature34_clpu_branch.py']
    assert BranchSearch.__module__ == 'dnr.nature34_clpu_branch'
