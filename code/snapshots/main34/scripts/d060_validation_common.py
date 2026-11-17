"""Fixed D060 population and durable bookkeeping, with no simulator imports."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUN = '20261024T073617_269327Z_nature34_clpu_branch'
SOURCE_SHA = '4c3777f938d21f074b40db56c82cd62ca1b956f6aa7d79215832bf192745c56f'
SUPPLEMENT_RUN = '20261025T070549_299770Z_d059_supplement'
SUPPLEMENT_SHA = '879850ffa7b22aa86ea379de535e9bb8caee1db961fe1635c8fe50e56cf19299'
EVENT_IDS = [19, 8, 16, 12]
SEEDS = [17, 41, 73]
METHODS = ['greedy', 'beam', 'supervised', 'bc_raw', 'ppo_raw']
VARIANTS = {'main': 256, 'beam_exp1024': 1024}
TOTAL_CELLS = 384
TOTAL_POINTS = 98304
ORDER_DOMAIN = 'd060-cell-order-v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def inside(base, relative):
    base = Path(base).resolve()
    result = (base / str(relative).replace('\\', '/')).resolve()
    require(result != base and result.is_relative_to(base), 'Path escapes declared directory')
    return result


def normalize_manifest(value):
    result = {}
    for name, digest in value.items():
        name = name.replace('\\', '/')
        require(name not in result, 'Duplicated manifest path')
        result[name] = digest
    return result


def build_schedule(contexts):
    require(len(contexts) == len({canonical(c) for c in contexts}) == 32, 'Expected 32 unique contexts')
    require(len({c['scenario_id'] for c in contexts}) == 8 and
            {c['event_id'] for c in contexts} == set(EVENT_IDS), 'Expected eight by four new-event cross')
    rows = []
    for context in contexts:
        require(set(context) == {'scenario_id', 'event_id'}, 'Context has undeclared fields')
        require(context['scenario_id'].startswith('N2__'), 'Expected N2 scenario')
        for method in METHODS:
            for seed in ([0] if method in ('greedy', 'beam') else SEEDS):
                rows.append({**context, 'phase': 'evaluation', 'group': 'evaluation',
                    'method': method, 'seed': seed, 'variant': 'main', 'point_budget': 256,
                    'representation': 'associated' if method == 'supervised' else 'raw',
                    'deterministic': True})
        rows.append({**context, 'phase': 'evaluation', 'group': 'evaluation', 'method': 'beam',
            'seed': 0, 'variant': 'beam_exp1024', 'point_budget': 256,
            'representation': 'raw', 'deterministic': True})
    rows.sort(key=lambda row: hashlib.sha256((ORDER_DOMAIN + '|' + canonical(row)).encode()).hexdigest())
    for index, row in enumerate(rows):
        row.update(episode_id=index, cell=f'episodes/{index:04d}')
    require(len(rows) == TOTAL_CELLS and sum(r['point_budget'] for r in rows) == TOTAL_POINTS,
            'Fixed task or physical allocation differs')
    return rows


def validate_fixed_config(config):
    expected = {'study': 'D060', 'point_budget': 256, 'variants': VARIANTS,
                'worker_deadline_seconds': 3600, 'max_workers': 4,
                'minimum_free_bytes': 12 * 2**30, 'max_physical_points': TOTAL_POINTS,
                'new_training_tasks': 0}
    for key, value in expected.items():
        require(config.get(key) == value, 'Fixed D060 setting differs: ' + key)
    for key in ('formal_holdout_numerical_access', 'tune_numerical_access', 'clpu_holdout_access'):
        require(config.get(key, False) is False, 'Forbidden held-out access')
    require(config.get('automatic_retry', False) is False, 'Automatic scientific retry is forbidden')
    require(config.get('hard_total_deadline_seconds') is None, 'No mechanical study deadline was authorized')
    return config
