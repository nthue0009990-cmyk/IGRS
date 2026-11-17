"""Five-slot joint topology/load optimization on the frozen NC34 proxy.

Each window starts from the actual committed history. Its incumbent is only
a proposal; certification of the initial state and full path is external.
"""
import hashlib
import json
import math
from pathlib import Path
import time

from .clpu_curves import multiplier
from .nature34_joint_sequence import _integer_weights
from .nature34_two_stage_linear import LinModel, expression
from .nature34_two_stage_planner import (
    _electrical, _variable_or_constant, topology_layer, topology_rows,
    NODE_LIMIT, SOFTWARE_RESIDUAL_TOLERANCE,
)


WINDOW, COMMIT, BACKEND_SECONDS = 5, 4, 20
VERSION = 'chen-joint-rolling-5-4-nc34-v1'


def _bit_changes(model, now, previous, tag):
    """Delta equals XOR for binary or fixed bit pair; no free switches."""
    changes = []
    for i, (x, p) in enumerate(zip(now, previous, strict=True)):
        d = model.var(f'{tag}:{i}', 0, 1, integer=True)
        model.add(d-x+p, lb=0, name=f'{tag}:{i}:xor_a')
        model.add(d+x-p, lb=0, name=f'{tag}:{i}:xor_b')
        model.add(d-x-p, ub=0, name=f'{tag}:{i}:xor_c')
        model.add(d+x+p, ub=2, name=f'{tag}:{i}:xor_d')
        changes.append(d)
    return sum(changes, expression(0))


def sample_demands(env, start, pickups, slot, offset):
    """Initial steady load, actual earlier pickup, or within-window pickup."""
    time_s = slot * env.slot_seconds + offset
    demands = {}
    for name, past in zip(env.load_names, start.pickups, strict=True):
        key = name.lower()
        if past is not None:
            coefficient = 1. if past == -1 else multiplier(env.event, time_s-past)
            demands[key] = [{'pickup': expression(1), 'coefficient': coefficient}]
        else:
            demands[key] = [
                {'pickup': z, 'coefficient': multiplier(env.event, time_s - h*env.slot_seconds)}
                for h, z in enumerate(pickups[key], start.step)
            ]
    return demands


def build_window(env, start, network, proxy, rows, deadline):
    """Build the current horizon; use only provably reachable mask filters."""
    env._validate_history(start)
    if (env.horizon, env.slot_seconds, env.sample_seconds) != (30, 60, 30):
        raise ValueError('Unsupported experiment time grid')
    count = min(WINDOW, env.horizon-start.step)
    if count <= 0:
        raise ValueError('No remaining decisions')
    weights, _ = _integer_weights(env.loads)
    model = LinModel()
    mask0 = start.command >> 10
    live0 = env.live_mask(start.command)
    old_live = [expression((live0 >> i) & 1) for i in range(9)]
    old_x = [expression((mask0 >> i) & 1) for i in range(9)]
    old_r = [expression((start.command >> i) & 1) for i in range(10)]
    old_y = {n.lower(): expression(int(p is not None))
             for n, p in zip(env.load_names, start.pickups, strict=True)}
    pickups = {name: [] for name in old_y}
    # One eligible command can change at most one switch bit. Equation 47
    # forbids losing an initially live controlled line. Readiness is independent
    # of whether a controlled *line* is connected to the main source.
    candidates = [(row, env.live_mask(row['mask'] << 10)) for row in rows
                  if (mask0 ^ row['mask']).bit_count() <= count]
    candidates = [(row, live) for row, live in candidates if not (live0 & ~live)]
    energy, operations = expression(0), expression(0)
    layers = []
    for k in range(count):
        if time.perf_counter() >= deadline:
            raise TimeoutError('Common deadline during window build')
        slot = start.step + k
        tag = f't{slot:02d}'
        group = [(row, live) for row, live in candidates if (mask0 ^ row['mask']).bit_count() <= k+1]
        layer = topology_layer(model, env, network, [row for row, _ in group], tag)
        live = [_variable_or_constant(model, tag + f':live:{i}',
                [(bits >> i) & 1 for _, bits in group], layer['selectors']) for i in range(9)]
        for i, (now, prior) in enumerate(zip(live, old_live)):
            model.add(now-prior, lb=0, name=tag + f':live_monotone:{i}')
        ops = (_bit_changes(model, layer['switches'], old_x, tag + ':switch_change')
               + _bit_changes(model, layer['readiness'], old_r, tag + ':ready_change'))
        model.add(ops, 0, 1, tag + ':one_original_action_or_wait')
        operations += ops
        for name, y in layer['load_on'].items():
            z = model.var(tag + ':pickup:' + name, 0, 1, integer=True)
            model.add(z-y+old_y[name], 0, 0, tag + ':service_monotone:' + name)
            pickups[name].append(z)
        energy += sum(weights[name]*y for name, y in layer['load_on'].items())
        coverage = []
        for offset in (0, 30, 60):
            if time.perf_counter() >= deadline:
                raise TimeoutError('Common deadline during electrical proxy build')
            result = _electrical(model, env, network, proxy, layer,
                                 sample_demands(env, start, pickups, slot, offset),
                                 tag + f':s{offset}')
            coverage.append(result['constraint_coverage'])
        layer.update(live=live, slot=slot, sample_coverage=coverage)
        layers.append(layer)
        old_live, old_x, old_r, old_y = live, layer['switches'], layer['readiness'], layer['load_on']
    # Integer nominal service units are proportional to normal kWh. The maximum
    # possible change in operations is count, so this is exact for both levels.
    if (count+1)*count*sum(weights.values())+count >= 2**53:
        raise ValueError('Objective cannot be represented exactly in binary64')
    return model, layers, operations-(count+1)*energy


def decode_window(env, start, layers, values):
    actions, states = [], [start]
    for layer in layers:
        bit = lambda x: round(x.value(values))
        command = sum(bit(x) << (10+i) for i, x in enumerate(layer['switches']))
        command |= sum(bit(r) << i for i, r in enumerate(layer['readiness']))
        diff = states[-1].command ^ command
        if diff.bit_count() > 1:
            raise ValueError('Incumbent changed multiple command bits')
        action = diff.bit_length()
        state = env.advance(states[-1], action)
        served = {n.lower() for n in env.energized(command)}
        if (served != {n for n, v in layer['load_on'].items() if bit(v)}
                or env.live_mask(command) != sum(bit(x) << i for i, x in enumerate(layer['live']))):
            raise ValueError('Proxy live loads/lines disagree with independent graph replay')
        actions.append(action)
        states.append(state)
    return actions, states


def _save(path, data):
    with Path(path).open('x', encoding='utf-8') as out:
        json.dump(data, out, allow_nan=False, sort_keys=True)
        out.write('\n')


def own_plan(env, base, cell, deadline):
    """Return 30 solver-proposed actions or None. Persist each window separately."""
    import numpy as np
    from scipy.optimize._highspy._core import HighsOptions
    from run_two_stage_clpu_development_check import build_cold_proxy

    cell = Path(cell)
    if HighsOptions().random_seed != 0:
        raise RuntimeError('Installed HiGHS default seed differs from frozen seed 0')
    windows, actions = [], []
    started = time.perf_counter()
    network, _, proxy = build_cold_proxy(Path(base))
    rows = topology_rows(env)
    state = env.initial

    def finish(status, answer):
        _save(cell/'chen_planner.json', {
            'version': VERSION, 'status': status, 'actions': answer,
            'committed_prefix': actions, 'windows': windows,
            'backend_calls': sum(w['backend_calls'] for w in windows),
            'elapsed_seconds': time.perf_counter()-started, 'physical_certificate': None})
        return answer

    for index in range(1, 9):
        tick = time.perf_counter()
        stem = f'chen_window_{index:02d}'
        info = {'index': index, 'start': state.to_dict(), 'backend_calls': 0,
                'backend_seed': 0, 'backend_threads': 1, 'deadline_clock': 'perf_counter',
                'future_slots': min(WINDOW, env.horizon-state.step)}
        windows.append(info)
        try:
            model, layers, obj = build_window(env, state, network, proxy, rows, deadline)
        except TimeoutError:
            info.update(status='WINDOW_BUILD_DEADLINE', build_seconds=time.perf_counter()-tick)
            _save(cell/(stem+'.json'), info)
            return finish(info['status'], None)
        info.update(build_seconds=time.perf_counter()-tick, variable_count=len(model.names),
                    row_count=len(model.rows), coverage=[l['sample_coverage'] for l in layers])
        seconds = min(BACKEND_SECONDS, deadline-time.perf_counter())
        if seconds <= 0:
            info.update(status='WINDOW_PRE_SOLVE_DEADLINE')
            _save(cell/(stem+'.json'), info)
            return finish(info['status'], None)
        result = model.solve(obj, time_limit=seconds, node_limit=NODE_LIMIT, mip_rel_gap=0.)
        values = result.pop('x')
        called = result['options'] is not None
        info.update(backend_calls=int(called), solve=result, has_incumbent=values is not None)
        if values is not None:
            path = cell/(stem+'_x.npz')
            with path.open('xb') as out:
                np.savez_compressed(out, x=np.asarray(values, dtype=np.float64))
            info.update(solution_file=path.name, solution_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if values is None:
            info.update(status='NO_WINDOW_INCUMBENT' if called else 'SPARSE_BUILD_DEADLINE')
        elif result['status'] not in (0, 1):
            info.update(status='UNUSABLE_BACKEND_STATUS')
        elif max(result['residuals'][k] for k in ('max_row_violation','max_integer_violation',
                                                  'max_variable_bound_violation')) > SOFTWARE_RESIDUAL_TOLERANCE:
            info.update(status='INCUMBENT_RESIDUAL_FAILURE')
        else:
            seq, states = decode_window(env, state, layers, values)
            weights, _ = _integer_weights(env.loads)
            actual = sum(a != 0 for a in seq)-(len(seq)+1)*sum(
                sum(weights[n.lower()] for n in env.energized(s.command)) for s in states[1:])
            if not math.isclose(actual, result['objective'], abs_tol=SOFTWARE_RESIDUAL_TOLERANCE, rel_tol=0):
                raise ValueError('Objective does not match independent graph replay')
            n = min(COMMIT, len(seq))
            actions.extend(seq[:n])
            state = states[n]
            info.update(status='COMMITTED_PROXY_PREFIX', window_actions=seq, committed=seq[:n],
                        end=state.to_dict(), replay_objective=actual)
        _save(cell/(stem+'.json'), info)
        if info['status'] != 'COMMITTED_PROXY_PREFIX':
            return finish(info['status'], None)
        if state.step == env.horizon:
            return finish('PROXY_PLAN_UNCERTIFIED', actions)
    raise AssertionError('Eight windows did not cover the frozen horizon')
