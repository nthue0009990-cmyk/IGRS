"""D017 seed-aware policies; online entry point has no historical AC input."""
import time
from .equal_budget import best_state

ONLINE_METHODS = ('fixed_build', 'fixed_proxy', 'adaptive_original',
                  'adaptive_build_seed', 'adaptive_proxy_seed')


def search(method, budget, initial, fixed_order, proxy, neighbors, evaluate):
    if method not in ONLINE_METHODS:
        raise ValueError('Online policy cannot consume privileged seeds')
    return _search(method, budget, initial, fixed_order, proxy, neighbors, evaluate, None)


def privileged_diagnostic(budget, initial, fixed_order, proxy, neighbors, evaluate, seed):
    """Offline diagnostic only; neither deployable method nor RL upper bound."""
    if tuple(seed) not in set(map(tuple, fixed_order)):
        raise ValueError('Privileged seed outside frozen pool')
    return _search('adaptive_privileged_seed', budget, initial, fixed_order,
                   proxy, neighbors, evaluate, tuple(seed))


def _search(method, budget, initial, fixed_order, proxy, neighbors, evaluate, privileged):
    if budget < 2 or budget % 2:
        raise ValueError('Budget must contain whole solver pairs')
    initial = tuple(initial)
    fixed = list(map(tuple, fixed_order))
    if not fixed or len(fixed) != len(set(fixed)):
        raise ValueError('Empty or duplicate fixed pool')
    states, trace, visited, scores = [], [], set(), {}
    rank_seconds = 0.
    def score(t):
        if t not in scores:
            scores[t] = float(proxy(t))
        return scores[t]
    tick = time.perf_counter()
    ordered = sorted(fixed, key=lambda t: (score(t), t)) if method in (
        'fixed_proxy', 'adaptive_proxy_seed') else fixed
    seed = (fixed[0] if method == 'adaptive_build_seed' else
            ordered[0] if method == 'adaptive_proxy_seed' else privileged)
    rank_seconds += time.perf_counter() - tick
    stop_reason = 'budget_exhausted'
    for step in range(budget // 2):
        tick = time.perf_counter()
        parent = None
        if step == 0:
            candidates, stage = [initial], 'initial'
        elif seed is not None and seed not in visited:
            candidates, stage = [seed], 'seed'
        elif method.startswith('adaptive_'):
            incumbent = best_state(states)
            parent = tuple(incumbent['open_branch_ids']) if incumbent else initial
            candidates = sorted((tuple(t) for t in neighbors(parent) if tuple(t) not in visited),
                                key=lambda t: (score(t), t))
            stage = 'neighbor'
        else:
            candidates, stage = [t for t in ordered if t not in visited], 'fixed'
        rank_seconds += time.perf_counter() - tick
        if not candidates:
            stop_reason = 'candidate_exhausted'
            break
        chosen = candidates[0]
        if chosen in visited:
            raise ValueError('Duplicate selection')
        trace.append(dict(step=step, stage=stage, parent=parent, candidates=candidates, selected=chosen))
        state = evaluate(chosen, step + 1)
        if tuple(state['open_branch_ids']) != chosen:
            raise ValueError('Evaluator returned wrong topology')
        states.append(state)
        visited.add(chosen)
    return dict(states=states, trace=trace, proxy_scores=[dict(opened=t, loss_mw=v) for t,v in sorted(scores.items())],
                ranking_seconds=rank_seconds, solver_calls=2*len(states), stop_reason=stop_reason,
                seed=seed, winner=best_state(states))
