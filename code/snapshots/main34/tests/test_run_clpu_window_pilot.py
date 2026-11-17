from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import pytest
import run_clpu_window_pilot as runner


def test_paired_schedule_is_fixed_independent_and_keeps_all_contexts():
    contexts = [{'scenario_id': f'fault{i}', 'event_id': (20, 3, 6, 30)[i % 4]} for i in range(8)]
    schedule = runner.schedule_for(contexts)
    assert schedule == runner.schedule_for(contexts)
    assert len(schedule) == 16 and sum(s['point_budget'] for s in schedule) == 4096
    assert {s['context_id'] for s in schedule} == set(range(8))
    for c in contexts:
        pair = [s for s in schedule if s['scenario_id'] == c['scenario_id']]
        assert {s['group'] for s in pair} == set(runner.GROUPS)
        assert len({s['cell'] for s in pair}) == 2


def test_duplicate_contexts_rejected():
    with pytest.raises(ValueError):
        runner.schedule_for([{'scenario_id': 'a', 'event_id': 3}] * 8)


def test_missing_physical_ledger_is_unknown_not_free(tmp_path):
    result = runner.costs(tmp_path, [{'cell': 'episodes/000'}])
    assert result['unknown_cost_cells'] == ['episodes/000']


def test_counter_aggregate_includes_initialization(tmp_path):
    runner.save(tmp_path / 'episodes/000/oracle_summary.json', {'points': 61, 'actual_numerical_entries': 247})
    result = runner.costs(tmp_path, [{'cell': 'episodes/000'}])
    assert result == {'known_physical_points': 61, 'known_numerical_entries': 247, 'unknown_cost_cells': []}
