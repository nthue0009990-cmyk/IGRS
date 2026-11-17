"""Deterministic development inputs, independent of power-flow/search libraries."""
from collections import defaultdict
import hashlib
import math
import random
import re

import numpy as np
import pandas as pd


def base_loads(text):
    block = re.search(r'mpc\.bus\s*=\s*\[(.*?)\];', text, re.S)[1]
    rows = []
    for line in block.splitlines():
        clean = line.split('%')[0].strip().rstrip(';')
        if clean:
            rows.append([float(x) for x in clean.split()])
    table = np.asarray(rows)
    if table.shape != (33, 13) or not np.array_equal(table[:, 0], np.arange(1, 34)):
        raise ValueError('Unexpected source bus table')
    p, q = table[:, 2] / 1000., table[:, 3] / 1000.
    if not np.isclose(p.sum(), 3.715, atol=1e-14, rtol=0) or not np.isclose(q.sum(), 2.3, atol=1e-14, rtol=0):
        raise ValueError('Base load units or totals')
    return p, q


def time_axis(raw_labels):
    local = pd.DatetimeIndex(pd.to_datetime(raw_labels, format='%d.%m.%Y %H:%M'))
    aware = local.tz_localize('Europe/Berlin', ambiguous='infer', nonexistent='raise')
    utc = aware.tz_convert('UTC')
    if not utc.is_unique or not (utc[1:] - utc[:-1] == pd.Timedelta(minutes=15)).all():
        raise ValueError('Non-contiguous physical clock')
    return local, aware, utc


def assign_profiles(pool, seed):
    rng = random.Random(seed)
    names = sorted(pool)
    return {bus: rng.choice(names) for bus in range(2, 34)}


def partition_weeks(local, seed, ratios):
    weeks = [f'{stamp.isocalendar().year}-W{stamp.isocalendar().week:02d}' for stamp in local]
    groups = sorted(set(weeks))
    rng = random.Random(seed)
    rng.shuffle(groups)
    first = math.floor(ratios[0] * len(groups))
    second = first + math.floor(ratios[1] * len(groups))
    labels = ['build'] * first + ['tune'] * (second-first) + ['development_holdout'] * (len(groups)-second)
    mapping = dict(zip(groups, labels))
    return weeks, mapping, [mapping[week] for week in weeks]


def day_digest(p, q):
    # Canonical little-endian bytes, including row count to distinguish 23/24/25h days.
    header = str(p.shape).encode('ascii') + b'|'
    return hashlib.sha256(header + np.asarray(p, dtype='<f8').tobytes() + np.asarray(q, dtype='<f8').tobytes()).hexdigest()


def cross_partition_duplicates(records):
    groups = defaultdict(list)
    for row in records:
        groups[row['digest']].append(row)
    return [rows for rows in groups.values() if len({row['partition'] for row in rows}) > 1]
