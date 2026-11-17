"""Development-only fixed-pool coverage diagnostic; no learned policy."""
import numpy as np


def loss_proxy(edges, resistance, opened, p, q, voltage_kv):
    """Lossless downstream P/Q, nominal-voltage I^2 R (MW), not AC feasibility."""
    n = p.shape[-1]
    adjacency = [[] for _ in range(n)]
    for i, (a, b) in enumerate(edges, 1):
        if i not in opened:
            adjacency[a].append((b, i - 1))
            adjacency[b].append((a, i - 1))
    parents = [-1] * n
    parents[0] = 0
    order = [0]
    incoming = [-1] * n
    for a in order:
        for b, e in adjacency[a]:
            if b == parents[a]:
                continue
            if parents[b] != -1:
                raise ValueError('Cycle in proxy input')
            parents[b], incoming[b] = a, e
            order.append(b)
    if len(order) != n:
        raise ValueError('Disconnected proxy input')
    downstream_p, downstream_q = p.copy(), q.copy()
    loss = np.zeros(p.shape[:-1], dtype=float)
    for b in reversed(order[1:]):
        loss += resistance[incoming[b]] * (downstream_p[..., b] ** 2 + downstream_q[..., b] ** 2) / voltage_kv ** 2
        downstream_p[..., parents[b]] += downstream_p[..., b]
        downstream_q[..., parents[b]] += downstream_q[..., b]
    return loss


def greedy_order(losses):
    """Nested fixed sets minimizing build mean relative loss, lexical tie order.

    Columns must already be in lexical topology order. Infinity denotes an
    ineligible observation, never a measured high loss. First select an anchor
    with verified feasibility on all construction cases, then add complements.
    """
    losses = np.asarray(losses, float)
    best = losses.min(axis=1)
    if not np.isfinite(best).all() or (best <= 0).any():
        raise ValueError('Missing positive feasible construction reference')
    relative = losses / best[:, None]
    single = relative.mean(axis=0)
    if not np.isfinite(single).any():
        raise ValueError('No common feasible singleton anchor; review required')
    first = int(single.argmin())
    chosen, current = [first], relative[:, first].copy()
    remaining = set(range(losses.shape[1])) - {first}
    while remaining:
        winner = min(remaining, key=lambda j: (float(np.minimum(current, relative[:, j]).mean()), j))
        chosen.append(winner)
        current = np.minimum(current, relative[:, winner])
        remaining.remove(winner)
    return chosen
