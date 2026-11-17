"""Calendar expansion and build-only tie handling for D009."""
import random
import numpy as np


def select_calendar(timeline, previous, cfg):
    rng = random.Random(cfg['selection_seed'])
    selected = {'build': list(previous['build']), 'tune': []}
    used_weeks = {r['iso_week'] for r in selected['build']}
    for month, clock in zip(cfg['additional_build_months'], cfg['additional_build_times']):
        dates = sorted({r['local_date'] for r in timeline if r['partition'] == 'build'
                        and int(r['local_date'][5:7]) == month and r['iso_week'] not in used_weeks})
        if not dates:
            raise ValueError('No build date in predeclared month')
        date = rng.choice(dates)
        rows = [r for r in timeline if r['local_date'] == date and r['local_label'].endswith(clock)]
        if len(rows) != 1:
            raise ValueError('Selected local time ambiguous or missing')
        selected['build'].append(rows[0])
        used_weeks.add(rows[0]['iso_week'])
    old_tune = {r['iso_week']: r for r in previous['tune']}
    weeks = sorted({r['iso_week'] for r in timeline if r['partition'] == 'tune'})
    for i, week in enumerate(weeks):
        if week in old_tune:
            selected['tune'].append(old_tune[week])
            continue
        date = rng.choice(sorted({r['local_date'] for r in timeline if r['iso_week'] == week and r['partition'] == 'tune'}))
        clock = cfg['tune_time_cycle'][i % len(cfg['tune_time_cycle'])]
        rows = [r for r in timeline if r['local_date'] == date and r['local_label'].endswith(clock)]
        if len(rows) != 1:
            raise ValueError('Selected local time ambiguous or missing')
        selected['tune'].append(rows[0])
    return selected


def greedy_ranked_order(losses):
    """Same greedy coverage objective; ties prefer low mean individual loss.

    This avoids filling a saturated fixed set by arbitrary branch-ID ordering.
    The original lexical-tie rule remains a separately reported comparator.
    """
    losses = np.asarray(losses, dtype=float)
    best = losses.min(axis=1)
    if not np.isfinite(best).all() or (best <= 0).any():
        raise ValueError('Missing positive feasible construction reference')
    relative = losses / best[:, None]
    individual = relative.mean(axis=0)
    if not np.isfinite(individual).any():
        raise ValueError('No common feasible singleton anchor')
    selected = [int(individual.argmin())]
    current = relative[:, selected[0]].copy()
    remaining = set(range(losses.shape[1])) - set(selected)
    while remaining:
        winner = min(remaining, key=lambda j: (float(np.minimum(current, relative[:, j]).mean()), float(individual[j]), j))
        selected.append(winner)
        current = np.minimum(current, relative[:, winner])
        remaining.remove(winner)
    return selected
