"""Display-only checks: population, stage separation and honest post-step drawing."""
from copy import deepcopy
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import plot_clpu_early_pilot as plot


def population():
    return [dict(cell=f'{c}_{a}_{s}', context_id=c, arm=a, seed=s, scenario_id=f'fault{c}', event_id=c)
        for c in range(8) for a in plot.ARMS for s in ((17, 41, 73) if a in ('protected_bc', 'protected_ppo') else (0,))]


def curves(early):
    cap = 256 if early else 512
    def one(count=False):
        def axis(size):
            return {'quality_pct': [None] + [30.] * (size - 1),
                'after_stop_carry_count' if count else 'after_stop_carry': [0 if count else False] * size}
        return {'physical': axis(cap + 1), 'worker_time': axis(4)}
    cells = []
    for row in population():
        points = [{'phase': 'EARLY', 'guard_established': False}]
        if not early and row['arm'] != 'window_reference':
            points.append({'phase': 'GUARD', 'guard_established': True})
        cells.append({**row, **one(), 'stop_physical_points': 100, 'exact_checkpoints': points})
    return {'schema': plot.SCHEMA, 'early_phase_only': early,
        'axes': {'physical_points': list(range(cap + 1)), 'worker_elapsed_seconds': [0, 1, 2, 3]},
        'cells': cells,
        'context_curves': [{'context_id': c, 'arm': a, 'seeds': [17, 41, 73] if a in ('protected_bc', 'protected_ppo') else [0],
            **one(True)} for c in range(8) for a in plot.ARMS],
        'group_curves': [{'arm': a, **one(True)} for a in plot.ARMS]}


def test_both_complete_populations_and_distinct_phase_axes_are_supported():
    cases = population()
    plot.base.validate_population(cases)
    plot.validate_curves(curves(False), cases, early=False)
    plot.validate_curves(curves(True), cases, early=True)


@pytest.mark.parametrize('mutation', ['missing_case', 'duplicate_seed', 'missing_context', 'missing_group'])
def test_no_selected_context_or_seed_population(mutation):
    data = curves(True)
    if mutation == 'missing_case':
        data['cells'].pop()
    elif mutation == 'duplicate_seed':
        data['cells'][-1]['seed'] = 17
    elif mutation == 'missing_context':
        data['context_curves'].pop()
    else:
        data['group_curves'].pop()
    with pytest.raises(ValueError):
        plot.validate_curves(data, population(), early=True)


@pytest.mark.parametrize('mutation', ['full_in_early', 'guard_claim', 'wrong_cap', 'wrong_schema', 'omitted_stop_carry'])
def test_full_and_early_are_not_interchangeable(mutation):
    data = curves(True)
    if mutation == 'full_in_early':
        data['cells'][-1]['exact_checkpoints'].append({'phase': 'GUARD', 'guard_established': True})
    elif mutation == 'guard_claim':
        data['cells'][-1]['exact_checkpoints'][0]['guard_established'] = True
    elif mutation == 'wrong_cap':
        data['axes']['physical_points'] = list(range(513))
    elif mutation == 'wrong_schema':
        data['schema'] = 'clpu-integrated-curves-v1'
    else:
        del data['group_curves'][-1]['physical']['after_stop_carry_count']
    with pytest.raises(ValueError):
        plot.validate_curves(data, population(), early=True)


def test_missing_certificates_stay_gaps_and_draw_only_post_steps():
    recorded = []
    ax = SimpleNamespace(step=lambda x, y, **kwargs: recorded.append((x, y, kwargs)),
                         set_ylim=lambda *_: None, grid=lambda **_: None)
    rows = [{'arm': arm, 'physical': {'quality_pct': [None, 0., 100.]}} for arm in plot.ARMS]
    plot.curve_lines(ax, rows, 'physical', [0, 1, 2])
    assert len(recorded) == 4
    assert all(r[2]['where'] == 'post' and math.isnan(r[1][0]) and r[1][1:] == [0., 100.] for r in recorded)


def test_input_gate_rejects_non_early_audited_status(tmp_path):
    report = tmp_path / 'derived_report'
    report.mkdir()
    cases = population()
    for row in cases:
        row.update(physical_points=100, process_wall_seconds=2.)
    for name, value in zip(plot.INPUTS, ({'status': 'AUDITED_DESCRIPTIVE_INTEGRATED_PILOT',
        'cells': 64, 'audited_cells': 64, 'contexts': 8}, cases, curves(False), curves(True))):
        plot.save(report / name, value)
    plot.save(report / 'completion_manifest.json', {p.name: plot.sha(p) for p in report.iterdir()})
    with pytest.raises(ValueError, match='complete audited early'):
        plot.load_inputs(report)
