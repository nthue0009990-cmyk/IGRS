"""D010 deterministic search policies. No file access or historical AC lookup."""
import time


def eligible(state):
    return state['jointly_converged'] and next(
        x for x in state['sensitivity'] if x['capacity_multiplier'] == 1.)['feasible']


def best_state(states):
    return min((s for s in states if eligible(s)),
               key=lambda s: (s['loss_mw'], tuple(s['open_branch_ids'])), default=None)


def search(method, budget, initial, fixed_order, proxy, neighbors, evaluate):
    """Each evaluate is exactly two fresh solver calls, including initial.

    Proxy values are cached only within this decision. Failed numerical or
    physical candidates remain visited. The incumbent only accepts jointly
    verified feasible results. Without one, initial is a search seed only.
    """
    if method not in ('fixed_build', 'fixed_proxy', 'adaptive_proxy'):
        raise ValueError('Unknown method')
    if budget < 2 or budget % 2:
        raise ValueError('Budget must be a positive number of solver pairs')
    states, trace, visited, scores = [], [], set(), {}
    rank_seconds = 0.
    initial = tuple(initial)
    fixed_order = [tuple(t) for t in fixed_order]

    def score(t):
        if t not in scores:
            scores[t] = float(proxy(t))
        return scores[t]

    rank_start = time.perf_counter()
    if method == 'fixed_proxy':
        ordered = sorted(fixed_order, key=lambda t: (score(t), t))
    else:
        ordered = fixed_order
    rank_seconds += time.perf_counter() - rank_start
    stop_reason = 'budget_exhausted'
    for step in range(budget // 2):
        rank_start = time.perf_counter()
        parent = None
        if step == 0:
            candidates = [initial]
        elif method == 'adaptive_proxy':
            incumbent = best_state(states)
            parent = tuple(incumbent['open_branch_ids']) if incumbent else initial
            candidates = sorted((tuple(t) for t in neighbors(parent) if tuple(t) not in visited),
                                key=lambda t: (score(t), t))
        else:
            candidates = [t for t in ordered if t not in visited]
        if not candidates:
            stop_reason = 'candidate_exhausted'
            rank_seconds += time.perf_counter() - rank_start
            break
        chosen = candidates[0]
        if chosen in visited:
            raise ValueError('Repeated candidate')
        trace.append(dict(step=step, parent=parent, candidates=candidates, selected=chosen))
        rank_seconds += time.perf_counter() - rank_start
        state = evaluate(chosen, step + 1)
        if tuple(state['open_branch_ids']) != chosen:
            raise ValueError('Evaluator returned wrong topology')
        states.append(state)
        visited.add(chosen)
    return dict(states=states, trace=trace, proxy_scores=[dict(opened=k, loss_mw=v) for k, v in sorted(scores.items())],
                ranking_seconds=rank_seconds, solver_calls=2*len(states),
                stop_reason=stop_reason, winner=best_state(states))
