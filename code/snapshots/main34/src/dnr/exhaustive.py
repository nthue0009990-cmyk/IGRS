"""Complete graph enumeration and durable, auditable numerical evidence."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sqlite3
import time
import warnings
import zlib

import numpy as np
import pandapower as pp
from pandapower.powerflow import LoadflowNotConverged
from .feasibility import classify, validate_solution


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


@contextmanager
def run_lock(path):
    stream = Path(path).open('a+b')
    if stream.tell() == 0:
        stream.write(b'0')
        stream.flush()
    stream.seek(0)
    if os.name == 'nt':
        import msvcrt
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        yield
    finally:
        stream.close()  # OS lock also releases after a crash; no stale-lock deletion.


def is_spanning_tree(edges, opened, nodes=33):
    opened = set(opened)
    if len(edges) - len(opened) != nodes - 1:
        return False
    parent = list(range(nodes))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for index, (a, b) in enumerate(edges, 1):
        if index in opened:
            continue
        a, b = find(a), find(b)
        if a == b:
            return False
        parent[a] = b
    return len({find(a) for a in range(nodes)}) == 1


def enumerate_trees(edges, nodes=33):
    opened_count = len(edges) - nodes + 1
    return [list(ids) for ids in itertools.combinations(range(1, len(edges) + 1), opened_count)
            if is_spanning_tree(edges, ids, nodes)]


class Journal:
    """Append-only SQLite events; every solver call is durable separately.

    A topology is committed only after its two calls and classification.
    SHA-256 chains plus externally saved checkpoints detect inconsistent recovery.
    """
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute('PRAGMA journal_mode=DELETE')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, kind TEXT NOT NULL, payload BLOB NOT NULL, digest TEXT NOT NULL)')
        self.db.execute("CREATE TRIGGER IF NOT EXISTS no_update BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT, 'append only'); END")
        self.db.execute("CREATE TRIGGER IF NOT EXISTS no_delete BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT, 'append only'); END")
        self.db.commit()
        row = self.db.execute('SELECT seq,digest FROM events ORDER BY seq DESC LIMIT 1').fetchone()
        self.seq, self.digest = row if row else (0, '0' * 64)
        self.counts = dict(solver_start_records=0, completed_solver_calls=0, nonconverged_solver_calls=0,
                           error_solver_calls=0, warning_count=0)

    def append(self, kind, value):
        seq = self.seq + 1
        raw = canonical(value)
        digest = hashlib.sha256(self.digest.encode() + str(seq).encode() + kind.encode() + raw).hexdigest()
        with self.db:
            self.db.execute('INSERT INTO events VALUES (?,?,?,?)', (seq, kind, zlib.compress(raw), digest))
        self.seq, self.digest = seq, digest
        if kind == 'solver_started':
            self.counts['solver_start_records'] += 1
        elif kind == 'solver_finished':
            self.counts['completed_solver_calls'] += 1
            self.counts['nonconverged_solver_calls'] += value['status'] == 'nonconverged'
            self.counts['error_solver_calls'] += value['status'] == 'validation_or_execution_error'
            self.counts['warning_count'] += len(value.get('warnings', []))
        return seq

    def verified_events(self, checkpoint=None):
        require(self.db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok', 'SQLite integrity failure')
        previous, expected, checkpoint_seen = '0' * 64, 1, checkpoint is None
        for seq, kind, blob, digest in self.db.execute('SELECT seq,kind,payload,digest FROM events ORDER BY seq'):
            require(seq == expected, 'Journal sequence gap')
            raw = zlib.decompress(blob)
            computed = hashlib.sha256(previous.encode() + str(seq).encode() + kind.encode() + raw).hexdigest()
            require(computed == digest, 'Journal checksum mismatch')
            if checkpoint and seq == checkpoint['seq']:
                require(digest == checkpoint['digest'], 'Checkpoint checksum mismatch')
                checkpoint_seen = True
            yield seq, kind, json.loads(raw)
            previous, expected = digest, seq + 1
        require(checkpoint_seen, 'Journal truncated before saved checkpoint')

    def checkpoint(self):
        return dict(seq=self.seq, digest=self.digest)

    def close(self):
        self.db.close()


BUS_COLS = ['vm_pu', 'va_degree', 'p_mw', 'q_mvar']
LINE_COLS = ['p_from_mw', 'q_from_mvar', 'p_to_mw', 'q_to_mvar', 'pl_mw', 'ql_mvar', 'i_from_ka', 'i_to_ka']


def full_tables(net):
    tables = {}
    for name, frame, cols in (
        ('bus', net.res_bus, BUS_COLS), ('line', net.res_line, LINE_COLS),
        ('load', net.res_load, ['p_mw', 'q_mvar']),
        ('ext_grid', net.res_ext_grid, ['p_mw', 'q_mvar'])):
        values = frame.reindex(columns=cols).to_numpy(dtype=float)
        # None denotes a non-finite diagnostic value, never an accepted solution.
        tables[name] = dict(columns=cols, index=[int(x) for x in frame.index],
            data=[[float(x) if math.isfinite(x) else None for x in row] for row in values])
    return tables


def verify_full_tables(tables, opened, source, cfg):
    """Reconstruct balance and summaries using only archived tables + raw source."""
    for name, cols, count in [('bus', BUS_COLS, 33), ('line', LINE_COLS, 37),
                             ('load', ['p_mw', 'q_mvar'], 32), ('ext_grid', ['p_mw', 'q_mvar'], 1)]:
        table = tables[name]
        require(table['columns'] == cols and table['index'] == list(range(count)), f'{name}: schema/index mismatch')
        require(len(table['data']) == count, f'{name}: missing rows')
    bus = np.array(tables['bus']['data'], dtype=float)
    line = np.array(tables['line']['data'], dtype=float)
    load = np.array(tables['load']['data'], dtype=float)
    ext = np.array(tables['ext_grid']['data'], dtype=float)
    active = np.array([i not in opened for i in range(1, 38)])
    require(all(np.isfinite(a).all() for a in [bus, line[active], load, ext]), 'Non-finite converged data')
    require(np.all(line[active, 6:8] >= 0), 'Invalid current magnitude')
    expected_load = source['bus'][1:, 2:4] / 1000
    require(np.max(np.abs(load - expected_load)) <= 1e-12, 'PQ demand was changed')
    require(np.max(np.abs(bus[1:, 2:4] - expected_load)) <= 1e-12, 'Bus injection/source load mismatch')
    require(np.max(np.abs(bus[0, 2:4] + ext.sum(axis=0))) <= 1e-12, 'Slack injection mismatch')
    require(abs(bus[0, 0] - cfg['slack_vm_pu']) <= 1e-12, 'Slack voltage changed')
    residual = bus[:, 2:4].copy()
    for idx in np.flatnonzero(active):
        a, b = source['branch'][idx, :2].astype(int) - 1
        residual[a] += line[idx, :2]
        residual[b] += line[idx, 2:4]
    loss = line[active, 4:6].sum(axis=0)
    terminal_loss = line[active, :2] + line[active, 2:4]
    balance = max(float(np.abs(residual).max()),
                  float(np.abs(ext.sum(axis=0) - load.sum(axis=0) - loss).max()),
                  float(np.abs(terminal_loss - line[active, 4:6]).max()))
    require(balance <= cfg['power_balance_tolerance_mw_mvar'], f'Archived power balance residual {balance}')
    currents = np.zeros(37)
    currents[active] = line[active, 6:8].max(axis=1) * 1000
    return dict(voltage=bus[:, 0], current_a=currents, loss_mw=float(loss[0]), balance_residual=balance)


def compare_pair(results, cfg, ratings):
    if len(results) != 2:
        return dict(jointly_converged=False, status='NUMERICAL_UNRESOLVED')
    nr, bfsw = results['nr'], results['bfsw']
    errors = dict(voltage=float(np.max(np.abs(nr['voltage'] - bfsw['voltage']))),
                  current=float(np.max(np.abs(nr['current_a'] - bfsw['current_a']))),
                  loss=abs(nr['loss_mw'] - bfsw['loss_mw']))
    for key, unit in [('voltage', 'pu'), ('current', 'a'), ('loss', 'mw')]:
        require(errors[key] <= cfg[f'cross_solver_{key}_tolerance_{unit}'], f'Solver disagreement: {errors}')
    metrics = [classify(nr['voltage'], nr['current_a'], ratings, cfg, m) for m in cfg['capacity_sensitivity_multipliers']]
    for metric, multiplier in zip(metrics, cfg['capacity_sensitivity_multipliers']):
        other = classify(bfsw['voltage'], bfsw['current_a'], ratings, cfg, multiplier)
        require(metric['feasible'] == other['feasible'], 'Solver feasibility classification disagrees at boundary')
    return dict(jointly_converged=True, status='NUMERICALLY_VERIFIED',
        v_min_pu=float(nr['voltage'].min()), v_max_pu=float(nr['voltage'].max()),
        v_min_bus=int(nr['voltage'].argmin()) + 1, loss_mw=nr['loss_mw'],
        max_current_a=float(nr['current_a'].max()), cross_solver_errors=errors, sensitivity=metrics)


def solve_topology(net, source, cfg, opened, ordinal, journal):
    edges = (source['branch'][:, :2].astype(int) - 1).tolist()
    require(is_spanning_tree(edges, opened), 'Invalid tree before power flow')
    net.line['in_service'] = [i not in opened for i in range(1, 38)]
    results, call_ids = {}, []
    for algorithm in cfg['powerflow']['algorithms']:
        context = dict(ordinal=ordinal, open_branch_ids=opened, algorithm=algorithm)
        call_id = journal.append('solver_started', dict(**context, utc=datetime.now(timezone.utc).isoformat()))
        call_ids.append(call_id)
        started = time.perf_counter()
        status, error, result = 'converged', None, None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                options = {k: v for k, v in cfg['powerflow'].items() if k != 'algorithms'}
                pp.runpp(net, algorithm=algorithm, check_connectivity=True, **options)
                require(net.converged, 'Solver returned without convergence')
                result = validate_solution(net, cfg)
                tables = full_tables(net)
                archived = verify_full_tables(tables, opened, source, cfg)
                require(abs(archived['loss_mw'] - result['loss_mw']) <= cfg['cross_solver_loss_tolerance_mw'], 'Stored loss mismatch')
                results[algorithm] = archived
            except LoadflowNotConverged as exc:
                status, error = 'nonconverged', str(exc)
            except Exception as exc:
                status, error = 'validation_or_execution_error', repr(exc)
        record = dict(**context, call_id=call_id, status=status, error=error,
            elapsed_seconds=time.perf_counter() - started,
            warnings=[str(w.message) for w in caught],
            tables=full_tables(net), tables_are_valid_solution=status == 'converged')
        journal.append('solver_finished', record)
        if status == 'validation_or_execution_error':
            raise ValueError(error)
    state = compare_pair(results, cfg, net.line.max_i_ka.to_numpy() * 1000)
    state.update(ordinal=ordinal, open_branch_ids=opened, call_ids=call_ids)
    journal.append('topology_finished', state)
    return state


def initial_summary(target):
    return dict(status='RUNNING', target_topologies=target, evaluated_topologies=0,
        jointly_converged_topologies=0, numerically_unresolved_topologies=0,
        feasible_counts={str(m): 0 for m in (0.8, 1.0, 1.2)},
        classification_counts={str(m): dict(feasible=0, voltage_only=0, thermal_only=0, both=0, numerical_unresolved=0) for m in (0.8, 1.0, 1.2)},
        best_minimum_voltage=None, witnesses={},
        max_cross_solver_errors=dict(voltage=0., current=0., loss=0.))


def add_state(summary, state):
    summary['evaluated_topologies'] += 1
    if not state['jointly_converged']:
        summary['numerically_unresolved_topologies'] += 1
        for row in summary['classification_counts'].values():
            row['numerical_unresolved'] += 1
        return
    summary['jointly_converged_topologies'] += 1
    best = summary['best_minimum_voltage']
    if best is None or state['v_min_pu'] > best['v_min_pu']:
        summary['best_minimum_voltage'] = state
    for key, value in state['cross_solver_errors'].items():
        summary['max_cross_solver_errors'][key] = max(summary['max_cross_solver_errors'][key], value)
    for metric in state['sensitivity']:
        key = str(metric['capacity_multiplier'])
        vb, ib = metric['voltage_violating_bus_count'] > 0, metric['thermal_violating_branch_count'] > 0
        category = 'both' if vb and ib else 'voltage_only' if vb else 'thermal_only' if ib else 'feasible'
        summary['classification_counts'][key][category] += 1
        if metric['feasible']:
            summary['feasible_counts'][key] += 1
            if key not in summary['witnesses'] or state['loss_mw'] < summary['witnesses'][key]['loss_mw']:
                summary['witnesses'][key] = state


def replay(journal, source, cfg, trees, ratings, checkpoint=None):
    """Verify all archived physics and reconstruct committed progress, including orphans."""
    summary = initial_summary(len(trees))
    started, finished = {}, {}
    counts = dict(solver_start_records=0, completed_solver_calls=0, nonconverged_solver_calls=0,
                  error_solver_calls=0, warning_count=0)
    max_balance = 0.
    for seq, kind, row in journal.verified_events(checkpoint):
        if kind == 'solver_started':
            require(row['open_branch_ids'] == trees[row['ordinal'] - 1], 'Started topology/ordinal mismatch')
            started[seq] = row
            counts['solver_start_records'] += 1
        elif kind == 'solver_finished':
            call_id = row['call_id']
            require(call_id in started, 'Finished call without unconsumed start')
            context = started.pop(call_id)
            require(all(context[k] == row[k] for k in ('ordinal', 'open_branch_ids', 'algorithm')), 'Call identity mismatch')
            require(row['status'] in ('converged', 'nonconverged', 'validation_or_execution_error'), 'Unknown call status')
            result = None
            if row['status'] == 'converged':
                result = verify_full_tables(row['tables'], row['open_branch_ids'], source, cfg)
                max_balance = max(max_balance, result['balance_residual'])
            else:
                require(bool(row['error']), 'Failure missing reason')
            finished[call_id] = (context, result)
            counts['completed_solver_calls'] += 1
            counts['nonconverged_solver_calls'] += row['status'] == 'nonconverged'
            counts['error_solver_calls'] += row['status'] == 'validation_or_execution_error'
            counts['warning_count'] += len(row.get('warnings', []))
        elif kind == 'topology_finished':
            require(row['ordinal'] == summary['evaluated_topologies'] + 1, 'Topology commit gap/duplicate')
            require(row['open_branch_ids'] == trees[row['ordinal'] - 1], 'Committed topology mismatch')
            require(len(set(row['call_ids'])) == 2, 'Not two unique calls')
            require(all(i in finished for i in row['call_ids']), 'Topology references missing solver call')
            pair = [finished.pop(i) for i in row['call_ids']]
            require([p[0]['algorithm'] for p in pair] == ['nr', 'bfsw'], 'Solver pair ordering')
            require(all(p[0]['ordinal'] == row['ordinal'] for p in pair), 'Cross-topology solver pair')
            expected = compare_pair({c['algorithm']: r for c, r in pair if r is not None}, cfg, ratings)
            expected.update(ordinal=row['ordinal'], open_branch_ids=row['open_branch_ids'], call_ids=row['call_ids'])
            require(canonical(expected) == canonical(row), 'Stored topology metrics disagree with archived powers')
            add_state(summary, row)
        else:
            require(kind in ('session_started', 'session_completed', 'session_aborted', 'recovery'), 'Unknown journal event')
    journal.counts = counts
    summary.update(counts)
    summary['starts_without_completion_record'] = len(started)
    summary['completed_calls_not_in_committed_topology'] = len(finished)
    summary['max_recomputed_power_balance_residual'] = max_balance
    return summary
