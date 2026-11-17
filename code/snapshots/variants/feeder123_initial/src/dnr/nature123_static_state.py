"""Sparse, bus-level restoration commands; no physical engine or certificate.

The integer uses low load-readiness bits followed by switch bits, as in NC34.
Action 0 waits, 1..n_load toggle readiness, then n_switch switch toggles.
Physical input keys must additionally bind the model, fault and CLPU history.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal


def _integer(value, upper, name):
    if type(value) is not int or not 0 <= value < upper:
        raise ValueError(f'{name} must be an integer in [0,{upper})')
    return value


@dataclass(frozen=True)
class GraphState:
    command: int
    forest: bool
    fault_closed: tuple[str, ...]
    source_reachable_buses: frozenset[str]
    energized_loads: frozenset[str]
    disconnected_loads: frozenset[str]
    held_loads: frozenset[str]
    effective_command: int
    nominal_served_kw: Decimal
    nominal_total_kw: Decimal
    physical_status: str = 'UNASSESSED'

    @property
    def graph_admissible(self):
        return self.forest and not self.fault_closed


class SparseRestorationGraph:
    """On-demand graph necessary conditions; not an enumerated reachable set.

    Manifest bus_edges must already collapse phase-parallel regulators, while
    retaining element identities in original_lines for fault bookkeeping.
    DER are not independent grid-forming sources in this static source-fed mode.
    """

    def __init__(self, manifest, fault_lines=()):
        nodes = tuple(manifest['nodes'])
        if not nodes or len(set(nodes)) != len(nodes) or any(type(n) is not str or not n for n in nodes):
            raise ValueError('Unique nonempty bus names required')
        self.nodes = frozenset(nodes)
        self.source = manifest['source_bus']
        if self.source not in self.nodes:
            raise ValueError('Unknown source bus')
        def edge(e):
            if len(e) != 2 or any(n not in self.nodes for n in e) or e[0] == e[1]:
                raise ValueError('Invalid bus edge')
            return tuple(sorted(e))
        self.edges = tuple(edge(e) for e in manifest['bus_edges'])
        if len(set(self.edges)) != len(self.edges):
            raise ValueError('Bus edges must be explicitly deduplicated')
        self.switch_names = tuple(manifest['switch_names'])
        self.switch_edges = tuple(edge(e) for e in manifest['switch_edges'])
        if (len(self.switch_names) != len(self.switch_edges) or not self.switch_names
                or len(set(self.switch_names)) != len(self.switch_names)
                or len(set(self.switch_edges)) != len(self.switch_edges)
                or not set(self.switch_edges) <= set(self.edges)):
            raise ValueError('Switch identities and edges must be distinct and known')
        self.original_lines = {n: edge(e) for n, e in manifest['original_lines'].items()}
        if not set(self.original_lines.values()) <= set(self.edges):
            raise ValueError('Unknown original line edge')
        self.fault_lines = tuple(fault_lines)
        if len(set(self.fault_lines)) != len(self.fault_lines) or not set(self.fault_lines) <= set(self.original_lines):
            raise ValueError('Fault lines must be distinct known original lines')
        self.fault_edges = frozenset(self.original_lines[n] for n in self.fault_lines)
        self.dispatch = tuple(manifest['dispatch_loads'])
        self.loads = {n: (row['bus'], Decimal(str(row['nominal_kw']))) for n, row in manifest['loads'].items()}
        if (not self.loads or len(set(self.dispatch)) != len(self.dispatch)
                or not set(self.dispatch) <= set(self.loads)
                or any(b not in self.nodes or not kw.is_finite() or kw < 0 for b, kw in self.loads.values())):
            raise ValueError('Complete valid nominal load inventory required')
        self.total_kw = sum((kw for _, kw in self.loads.values()), Decimal(0))
        if self.total_kw <= 0:
            raise ValueError('Demand denominator must be positive')
        self.ns, self.nl = len(self.switch_names), len(self.dispatch)
        self.ready_mask = (1 << self.nl) - 1
        self.state_limit = 1 << (self.ns + self.nl)
        self.action_count = 1 + self.ns + self.nl
        self._dispatch_index = {n: i for i, n in enumerate(self.dispatch)}
        self._switch_index = {e: i for i, e in enumerate(self.switch_edges)}
        # Only visited switch assignments are cached; no 2**ns table exists.
        self._topologies = {}
        normal = tuple(manifest['normally_closed'])
        if len(normal) != self.ns or any(type(b) is not int or b not in (0, 1) for b in normal):
            raise ValueError('Explicit binary normal switch settings required')
        initial_mask = sum((b if e not in self.fault_edges else 0) << i for i, (b, e) in enumerate(zip(normal, self.switch_edges)))
        self.initial_state = self.encode(initial_mask, self.ready_mask)
        if not self.metadata(self.initial_state).graph_admissible:
            raise ValueError('Initial topology is not a forest with faults isolated')

    def encode(self, switch_mask, readiness_mask):
        _integer(switch_mask, 1 << self.ns, 'switch mask')
        _integer(readiness_mask, 1 << self.nl, 'readiness mask')
        return (switch_mask << self.nl) | readiness_mask

    def decode(self, command):
        _integer(command, self.state_limit, 'command')
        return command >> self.nl, command & self.ready_mask

    def command_bits(self, command):
        switches, ready = self.decode(command)
        return tuple((switches >> i) & 1 for i in range(self.ns)) + tuple((ready >> i) & 1 for i in range(self.nl))

    def raw_transition(self, command, action):
        """Coordinate operation only. It does not imply graph or physical validity."""
        self.decode(command)
        _integer(action, self.action_count, 'action')
        return command if action == 0 else command ^ (1 << (action - 1))

    def _topology(self, switch_mask):
        if switch_mask in self._topologies:
            return self._topologies[switch_mask]
        active = [e for e in self.edges if e not in self.fault_edges and
                  (e not in self._switch_index or switch_mask & (1 << self._switch_index[e]))]
        parent = {n: n for n in self.nodes}
        def find(n):
            while parent[n] != n:
                parent[n] = parent[parent[n]]
                n = parent[n]
            return n
        forest = True
        for a, b in active:
            ra, rb = find(a), find(b)
            if ra == rb:
                forest = False
            else:
                parent[ra] = rb
        reachable = frozenset(n for n in self.nodes if find(n) == find(self.source))
        fault_closed = tuple(n for i, (n, e) in enumerate(zip(self.switch_names, self.switch_edges))
                             if e in self.fault_edges and switch_mask & (1 << i))
        result = forest, reachable, fault_closed
        self._topologies[switch_mask] = result
        return result

    def metadata(self, command):
        switches, ready = self.decode(command)
        forest, reachable, fault_closed = self._topology(switches)
        energized, disconnected, held = set(), set(), set()
        effective_ready = ready
        for name, (bus, _) in self.loads.items():
            index = self._dispatch_index.get(name)
            if bus not in reachable:
                disconnected.add(name)
                if index is not None:
                    effective_ready &= ~(1 << index)
            elif index is not None and not ready & (1 << index):
                held.add(name)
            else:
                energized.add(name)
        return GraphState(command, forest, fault_closed, reachable, frozenset(energized),
                          frozenset(disconnected), frozenset(held), self.encode(switches, effective_ready),
                          sum((self.loads[n][1] for n in energized), Decimal(0)), self.total_kw)

    def transition(self, command, action):
        """No reinterruption of any source-connected admitted load, including fixed loads."""
        destination = self.raw_transition(command, action)
        before, after = self.metadata(command), self.metadata(destination)
        if not before.graph_admissible or not after.graph_admissible:
            raise ValueError('Fault-closed or nonforest command')
        if not before.energized_loads <= after.energized_loads:
            raise ValueError('Reinterruption of supplied load')
        return destination

    def successors(self, command):
        """Enumerate at most 42 actions, retaining an explicit rejection reason."""
        self.decode(command)
        rows = []
        for action in range(self.action_count):
            destination = self.raw_transition(command, action)
            try:
                self.transition(command, action)
                reason = None
            except ValueError as exc:
                reason = str(exc)
            rows.append({'action': action, 'destination': destination,
                         'graph_admissible': reason is None, 'rejection': reason,
                         'physical_status': 'UNASSESSED'})
        return tuple(rows)

    def explore(self, start, max_depth, max_states):
        """Bounded diagnostic BFS. A cap is explicit truncation, never completeness."""
        _integer(max_depth, 1 << 31, 'max_depth')
        _integer(max_states, 1 << 63, 'max_states')
        if max_states == 0 or not self.metadata(start).graph_admissible:
            raise ValueError('Positive state cap and admissible start required')
        parents, depth = {start: None}, {start: 0}
        queue, rejected = deque([start]), 0
        capped = False
        while queue and not capped:
            current = queue.popleft()
            if depth[current] == max_depth:
                continue
            for row in self.successors(current):
                if not row['graph_admissible']:
                    rejected += 1
                    continue
                other = row['destination']
                if other in parents:
                    continue
                if len(parents) == max_states:
                    capped = True
                    break
                parents[other] = (current, row['action'])
                depth[other] = depth[current] + 1
                queue.append(other)
        return {'parents': parents, 'depth': depth, 'rejected_actions': rejected,
                'state_cap_reached': capped, 'complete_through_requested_depth': not capped,
                'max_depth': max_depth, 'max_states': max_states, 'physical_status': 'UNASSESSED'}

    @staticmethod
    def reconstruct(parents, target):
        if target not in parents:
            raise ValueError('Target not in bounded exploration')
        path, actions, seen = [target], [], set()
        while parents[path[-1]] is not None:
            if path[-1] in seen:
                raise ValueError('Cyclic parent record')
            seen.add(path[-1])
            previous, action = parents[path[-1]]
            actions.append(action)
            path.append(previous)
        return tuple(reversed(path)), tuple(reversed(actions))
