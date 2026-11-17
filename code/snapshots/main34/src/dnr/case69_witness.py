"""Deterministic one-exchange candidates; no power-flow imports."""
from collections import deque


def is_tree(nodes, edges, opens):
    parent = {i: i for i in nodes}
    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i
    count = 0
    for k, (a, b) in edges.items():
        if k in opens:
            continue
        a, b = find(a), find(b)
        if a == b:
            return False
        parent[a] = b
        count += 1
    return count == len(nodes)-1 and len({find(i) for i in nodes}) == 1


def cycle_neighbors(nodes, edges, seed):
    seed = set(seed)
    assert is_tree(nodes, edges, seed)
    adj = {i: [] for i in nodes}
    for k, (u, v) in edges.items():
        if k not in seed:
            adj[u].append((v, k)); adj[v].append((u, k))
    found = set()
    for close in sorted(seed):
        start, target = edges[close]
        queue = deque([start]); previous = {start: None}
        while queue and target not in previous:
            u = queue.popleft()
            for v, edge in adj[u]:
                if v not in previous:
                    previous[v] = (u, edge); queue.append(v)
        assert target in previous
        v = target
        while v != start:
            u, cut = previous[v]
            found.add(tuple(sorted((seed-{close}) | {cut})))
            v = u
    return sorted(found)


def exhaustive_neighbors(nodes, edges, seed):
    # Independent enumeration: try every exchange and reject non-trees by DSU.
    seed = set(seed)
    return sorted({tuple(sorted((seed-{close}) | {cut}))
        for close in seed for cut in set(edges)-seed
        if is_tree(nodes, edges, (seed-{close}) | {cut})})


def preflight(branches, seed):
    triangle = {1: (1, 2), 2: (2, 3), 3: (1, 3)}
    assert cycle_neighbors([1, 2, 3], triangle, [3]) == [(1,), (2,)]
    assert not is_tree([1, 2, 3, 4], triangle, [])  # cycle plus isolated node
    assert not is_tree([1, 2, 3], triangle, [2, 3])  # disconnected
    edges = {int(r['branch_id']): (int(r['from_bus']), int(r['to_bus'])) for r in branches}
    nodes = list(range(1, 70))
    candidates = cycle_neighbors(nodes, edges, seed)
    independent = exhaustive_neighbors(nodes, edges, seed)
    assert candidates == independent and 0 < len(candidates) <= 340
    for c in candidates:
        assert len(c) == 5 and len(set(c)^set(seed)) == 2 and is_tree(nodes, edges, c)
    return candidates, dict(status='PASS', tests=['triangle_known_neighbors', 'reject_cycle_and_isolation',
        'reject_disconnected', 'full_340_exchange_independent_enumeration', 'all_candidate_tree_and_distance'],
        candidate_count=len(candidates), test_solver_calls=0)
