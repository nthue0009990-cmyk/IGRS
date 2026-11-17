"""D013 finite-pool endpoint model; no PF, file access, or learned policy."""
import numpy as np


def switch_counts(pool):
    sets = [set(x) for x in pool]
    return np.array([[len(a ^ b) for b in sets] for a in sets], dtype=int)


def validate(loss_mw, feasible, hours):
    loss = np.asarray(loss_mw, dtype=float)
    ok = np.asarray(feasible, dtype=bool)
    duration = np.asarray(hours, dtype=float)
    if loss.ndim != 2 or ok.shape != loss.shape or duration.shape != (len(loss),):
        raise ValueError('Shape mismatch')
    if not np.isfinite(loss).all() or not np.isfinite(duration).all() or np.any(duration <= 0):
        raise ValueError('Unknown numerical state or invalid duration')
    if np.any(loss < -1e-10):
        raise ValueError('Negative active losses')
    return loss, ok, duration


def choose_current(loss_row, feasible_row, previous, counts, interval_hours,
                   price, action_cost, method, horizon_hours=None, remaining_hours=None):
    """Only one current row is accepted: future electrical data is unavailable."""
    row = np.asarray(loss_row, dtype=float)
    ok = np.asarray(feasible_row, dtype=bool)
    if not np.isfinite(row).all():
        raise ValueError('Unresolved current data')
    candidates = np.flatnonzero(ok)
    if not len(candidates):
        return None
    if method == 'no_reconfiguration':
        return previous if ok[previous] else None
    if method == 'current_minimum_loss':
        values = row
    elif method == 'current_single_step_total_cost':
        values = row*1000*interval_hours*price + counts[previous]*action_cost
    elif method == 'persistence_hysteresis':
        if horizon_hours is None or remaining_hours is None:
            raise ValueError('Missing persistence horizon')
        horizon = min(horizon_hours, remaining_hours)
        if horizon <= 0:
            raise ValueError('Invalid persistence horizon')
        values = row*1000*horizon*price + counts[previous]*action_cost
    else:
        raise ValueError('Unknown causal method')
    # Exact objective tie: fewer real operations, then frozen lexical pool order.
    return int(min(candidates, key=lambda j: (values[j], counts[previous,j], j)))


def evaluate_path(path, loss_mw, feasible, hours, counts, initial, price, action_cost):
    loss, ok, duration = validate(loss_mw, feasible, hours)
    if len(path) != len(loss):
        raise ValueError('Incomplete trajectory')
    path = np.asarray(path, dtype=int)
    if np.any(path < 0) or np.any(path >= loss.shape[1]):
        raise ValueError('Invalid candidate index')
    previous = np.r_[initial, path[:-1]]
    operations = counts[previous, path]
    energy = loss[np.arange(len(loss)), path]*1000*duration
    return dict(path=path.tolist(), feasible=bool(ok[np.arange(len(loss)),path].all()),
        loss_energy_kwh=float(energy.sum()), switch_actions=int(operations.sum()),
        energy_cost_usd=float(energy.sum()*price), switching_cost_usd=float(operations.sum()*action_cost),
        total_cost_usd=float(energy.sum()*price+operations.sum()*action_cost),
        per_step_energy_kwh=energy.tolist(), per_step_actions=operations.tolist())


def causal_path(loss_mw, feasible, hours, counts, initial, price, action_cost, method, horizon=None):
    loss, ok, duration = validate(loss_mw, feasible, hours)
    previous, path = initial, []
    for t in range(len(loss)):
        chosen = choose_current(loss[t], ok[t], previous, counts, duration[t], price,
            action_cost, method, horizon, float(duration[t:].sum()))
        if chosen is None:
            return dict(feasible=False, path=path, failed_step=t, status='NO_FEASIBLE_CAUSAL_ACTION')
        path.append(chosen)
        previous = chosen
    return evaluate_path(path, loss, ok, duration, counts, initial, price, action_cost)


def best_constant(loss_mw, feasible, hours, counts, initial, price, action_cost):
    loss, ok, duration = validate(loss_mw, feasible, hours)
    candidates = np.flatnonzero(ok.all(axis=0))
    if not len(candidates):
        return dict(feasible=False, path=[], status='NO_ALL_DAY_FEASIBLE_CONSTANT')
    values = (loss*duration[:,None]*1000*price).sum(axis=0)+counts[initial]*action_cost
    j = int(min(candidates, key=lambda j:(values[j], counts[initial,j], j)))
    return evaluate_path([j]*len(loss), loss, ok, duration, counts, initial, price, action_cost)


def dynamic_programming(loss_mw, feasible, hours, counts, initial, price, action_cost):
    """Finite verified state space, unrestricted endpoint transitions, free terminal."""
    loss, ok, duration = validate(loss_mw, feasible, hours)
    energy_cost = loss*duration[:,None]*1000*price
    value = np.where(ok[0], energy_cost[0]+counts[initial]*action_cost, np.inf)
    actions = counts[initial].copy()
    parents = np.full(loss.shape, -1, dtype=int)
    for t in range(1,len(loss)):
        trial = value[:,None]+counts*action_cost
        trial_actions = actions[:,None]+counts
        for j in range(loss.shape[1]):
            best_value = trial[:,j].min()
            ties = np.flatnonzero(trial[:,j] == best_value)
            i = int(min(ties, key=lambda i:(trial_actions[i,j], i)))
            parents[t,j] = i
        pred = parents[t]
        actions = trial_actions[pred, np.arange(loss.shape[1])]
        value = np.where(ok[t], trial[pred,np.arange(loss.shape[1])]+energy_cost[t], np.inf)
    if not np.isfinite(value).any():
        return dict(feasible=False, path=[], status='NO_ALL_DAY_FEASIBLE_PATH')
    j = min(np.flatnonzero(np.isfinite(value)), key=lambda j:(value[j], actions[j], j))
    optimum = float(value[j])
    path = [int(j)]
    for t in range(len(loss)-1,0,-1):
        j = parents[t,j]
        path.append(int(j))
    result = evaluate_path(path[::-1], loss, ok, duration, counts, initial, price, action_cost)
    if not result['feasible'] or abs(result['total_cost_usd']-optimum) > 1e-8:
        raise ValueError('DP objective reconstruction mismatch')
    result['bellman_value_usd'] = optimum
    return result
