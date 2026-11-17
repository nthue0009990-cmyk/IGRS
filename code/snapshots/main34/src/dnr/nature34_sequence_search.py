"""Finite, explicitly supplied state-graph diagnostics for restoration sequences.

The caller must verify the saved physical evidence, build legal edges, and keep
operating conditions (including DER settings) fixed within each graph. This
module does not certify power-flow records or switching transients. Missing
states/edges are unknown, not proved infeasible. All policies can query the same
saved graph; their query counts describe offline diagnostic work, not an online
equal-power-flow-budget competition.

Each transition occupies one abstract operation step. Its reward is the nominal
served kW of the destination. The initial state is not counted; stay transitions
pad every path to H steps. kW-step is not kWh without sourced step durations.
The fixed lexicographic order is: more accumulated service, more final service,
fewer load interruption events, fewer non-stay operations, then smaller path of
state IDs. No weighted combination of unlike units is used. Decimal input
values are summed as rational numbers so ties do not depend on addition order.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from math import isfinite
from types import MappingProxyType
from typing import Mapping, Sequence


@dataclass(frozen=True)
class CertifiedState:
    state_id: str
    controls: tuple[int, ...]
    served_kw: float
    energized_load_ids: tuple[str, ...]
    evidence_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.state_id, str) or not self.state_id:
            raise ValueError("state_id must be a nonempty string")
        controls = tuple(self.controls)
        if not controls or any(type(x) is not int or x not in (0, 1) for x in controls):
            raise ValueError("controls must be nonempty binary integer coordinates")
        if isinstance(self.served_kw, bool) or not isfinite(self.served_kw) or self.served_kw < 0:
            raise ValueError("served_kw must be finite and nonnegative")
        loads = tuple(self.energized_load_ids)
        if any(not isinstance(x, str) or not x for x in loads) or len(set(loads)) != len(loads):
            raise ValueError("energized load IDs must be distinct nonempty strings")
        if not isinstance(self.evidence_id, str) or not self.evidence_id:
            raise ValueError("each state needs an external evidence identifier")
        object.__setattr__(self, "controls", controls)
        object.__setattr__(self, "energized_load_ids", tuple(sorted(loads)))


@dataclass(frozen=True, init=False)
class StateGraph:
    """Immutable finite graph with caller-supplied directed single-bit edges.

    A stay is available at every state. A Hamming-one relationship alone never
    creates an edge: the caller must explicitly supply every non-stay edge.
    The evidence IDs identify external checks; this class does not validate them.
    """

    states: Mapping[str, CertifiedState]
    successors: Mapping[str, tuple[str, ...]]
    denominator_kw: float

    def __init__(self, states: Mapping[str, CertifiedState],
                 successors: Mapping[str, Sequence[str]], denominator_kw: float):
        if not states:
            raise ValueError("at least one state is required")
        if isinstance(denominator_kw, bool) or not isfinite(denominator_kw) or denominator_kw <= 0:
            raise ValueError("denominator_kw must be positive and finite")
        copied = dict(states)
        if any(not isinstance(s, CertifiedState) or k != s.state_id for k, s in copied.items()):
            raise ValueError("state mapping keys must match CertifiedState IDs")
        widths = {len(s.controls) for s in copied.values()}
        if len(widths) != 1:
            raise ValueError("all control vectors must have the same width")
        if len({s.controls for s in copied.values()}) != len(copied):
            raise ValueError("duplicate controls: deduplicate evidence before graph creation")
        if set(successors) - set(copied):
            raise ValueError("successor mapping contains unknown source states")
        edges = {}
        for source, state in copied.items():
            targets = tuple(successors.get(source, ()))
            if len(set(targets)) != len(targets):
                raise ValueError("duplicate explicit edges are not permitted")
            for target in targets:
                if target not in copied:
                    raise ValueError("an edge targets an unknown state")
                distance = sum(a != b for a, b in zip(state.controls, copied[target].controls))
                if source != target and distance != 1:
                    raise ValueError("each non-stay edge must change exactly one binary control")
            edges[source] = tuple(sorted(set(targets) | {source}))
        object.__setattr__(self, "states", MappingProxyType(copied))
        object.__setattr__(self, "successors", MappingProxyType(edges))
        object.__setattr__(self, "denominator_kw", denominator_kw)


@dataclass(frozen=True)
class SequenceStep:
    from_state: str
    to_state: str
    changed_control: int | None
    served_kw: float
    interrupted_load_ids: tuple[str, ...]


@dataclass(frozen=True)
class QueryAccounting:
    state_value_requests: int
    state_value_cache_hits: int
    unique_state_values: int
    successor_requests: int
    successor_cache_hits: int
    unique_successor_lists: int
    dynamic_program_requests: int
    dynamic_program_cache_hits: int
    dynamic_program_entries: int
    candidate_edges_considered: int
    result_assembly_state_reads: int
    new_physical_solves: int = 0
    information_regime: str = "shared_saved_graph_offline_diagnostic"
    viability_preprocessing_edge_scans: int = 0
    viability_allowed_edges: int = 0
    viability_reverse_expansions: int = 0
    viability_distance_lookups: int = 0
    viability_reachable_states: int = 0


@dataclass(frozen=True)
class SequenceResult:
    method: str
    horizon: int
    state_path: tuple[str, ...]
    steps: tuple[SequenceStep, ...]
    cumulative_kw_steps: float
    cumulative_fraction_steps: float
    mean_served_fraction: float | None
    final_served_kw: float
    final_served_fraction: float
    interruption_events: int
    unique_interrupted_load_ids: tuple[str, ...]
    initial_load_interruption_events: int
    initial_load_ids_unserved_at_end: tuple[str, ...]
    non_stay_operations: int
    queries: QueryAccounting
    optimality_scope: str
    terminal_ids: tuple[str, ...] | None
    terminal_reached: bool | None
    terminal_required: bool
    status: str = "COMPLETE_TRAJECTORY"
    edge_constraint: str = "supplied_graph_edges"


@dataclass(frozen=True)
class NoPathResult:
    method: str
    horizon: int
    start_state: str
    terminal_ids: tuple[str, ...]
    queries: QueryAccounting
    status: str = "NO_PATH_TO_REQUIRED_TERMINAL"
    optimality_scope: str = "no_path_only_in_supplied_graph_at_fixed_horizon_with_stay"
    edge_constraint: str = "supplied_graph_edges"


@dataclass(frozen=True)
class _Plan:
    destinations: tuple[str, ...]
    cumulative: Fraction
    final: Fraction
    interruptions: int
    operations: int

    @property
    def order(self) -> tuple:
        return (-self.cumulative, -self.final, self.interruptions,
                self.operations, self.destinations)


class _Solver:
    def __init__(self, graph: StateGraph, required_terminals: frozenset[str] | None = None):
        self.graph = graph
        self.required_terminals = required_terminals
        self.value_cache: dict[str, Fraction] = {}
        self.edge_cache: dict[str, tuple[str, ...]] = {}
        self.dp_cache: dict[tuple[str, int], _Plan | None] = {}
        self.value_requests = self.value_hits = 0
        self.edge_requests = self.edge_hits = 0
        self.dp_requests = self.dp_hits = self.candidate_edges = 0
        self.information_regime = "shared_saved_graph_offline_diagnostic"
        self.viability_edge_scans = self.viability_allowed_edges = 0
        self.viability_reverse_expansions = self.viability_distance_lookups = 0
        self.viability_reachable_states = 0
        self.edge_constraint = "supplied_graph_edges"

    def value(self, state_id: str) -> Fraction:
        self.value_requests += 1
        if state_id in self.value_cache:
            self.value_hits += 1
        else:
            self.value_cache[state_id] = Fraction(str(self.graph.states[state_id].served_kw))
        return self.value_cache[state_id]

    def edges(self, state_id: str) -> tuple[str, ...]:
        self.edge_requests += 1
        if state_id in self.edge_cache:
            self.edge_hits += 1
        else:
            self.edge_cache[state_id] = self.graph.successors[state_id]
        return self.edge_cache[state_id]

    def plan(self, start: str, remaining: int) -> _Plan | None:
        self.dp_requests += 1
        key = (start, remaining)
        if key in self.dp_cache:
            self.dp_hits += 1
            return self.dp_cache[key]
        if remaining == 0:
            best = (_Plan((), Fraction(0), self.value(start), 0, 0)
                    if self.required_terminals is None or start in self.required_terminals else None)
        else:
            candidates = []
            before = frozenset(self.graph.states[start].energized_load_ids)
            for target in self.edges(start):
                self.candidate_edges += 1
                tail = self.plan(target, remaining - 1)
                if tail is None:
                    continue
                lost = before - frozenset(self.graph.states[target].energized_load_ids)
                candidates.append(_Plan(
                    (target,) + tail.destinations, self.value(target) + tail.cumulative,
                    tail.final, len(lost) + tail.interruptions,
                    int(start != target) + tail.operations))
            best = min(candidates, key=lambda p: p.order) if candidates else None
        self.dp_cache[key] = best
        return best

    def accounting(self, result_reads: int) -> QueryAccounting:
        return QueryAccounting(
            self.value_requests, self.value_hits, len(self.value_cache),
            self.edge_requests, self.edge_hits, len(self.edge_cache),
            self.dp_requests, self.dp_hits, len(self.dp_cache), self.candidate_edges,
            result_reads, information_regime=self.information_regime,
            viability_preprocessing_edge_scans=self.viability_edge_scans,
            viability_allowed_edges=self.viability_allowed_edges,
            viability_reverse_expansions=self.viability_reverse_expansions,
            viability_distance_lookups=self.viability_distance_lookups,
            viability_reachable_states=self.viability_reachable_states)

    def result(self, method: str, path: Sequence[str], exact: bool,
               terminal_ids: tuple[str, ...] | None) -> SequenceResult:
        states = [self.graph.states[x] for x in path]
        steps = []
        initially_energized = frozenset(states[0].energized_load_ids)
        initial_loss_events = 0
        all_lost: set[str] = set()
        for before, after in zip(states, states[1:]):
            changed = [i for i, (a, b) in enumerate(zip(before.controls, after.controls)) if a != b]
            lost = tuple(sorted(set(before.energized_load_ids) - set(after.energized_load_ids)))
            all_lost.update(lost)
            initial_loss_events += len(initially_energized.intersection(lost))
            steps.append(SequenceStep(before.state_id, after.state_id,
                                      changed[0] if changed else None, after.served_kw, lost))
        cumulative = sum((Fraction(str(s.served_kw)) for s in states[1:]), Fraction(0))
        denominator = Fraction(str(self.graph.denominator_kw))
        normalized = cumulative / denominator
        horizon = len(steps)
        queries = self.accounting(len(states))
        return SequenceResult(
            method, horizon, tuple(path), tuple(steps), float(cumulative), float(normalized),
            float(normalized / horizon) if horizon else None, states[-1].served_kw,
            float(Fraction(str(states[-1].served_kw)) / denominator),
            sum(len(s.interrupted_load_ids) for s in steps), tuple(sorted(all_lost)),
            initial_loss_events, tuple(sorted(initially_energized - set(states[-1].energized_load_ids))),
            sum(s.changed_control is not None for s in steps), queries,
            "exact_only_within_supplied_finite_graph_fixed_horizon_and_lexicographic_objective"
            if exact else "no_global_optimality_claim", terminal_ids,
            path[-1] in terminal_ids if terminal_ids is not None else None,
            self.required_terminals is not None, edge_constraint=self.edge_constraint)


def _validate_request(graph: StateGraph, start: str, horizon: int) -> None:
    if start not in graph.states:
        raise ValueError("unknown starting state")
    if type(horizon) is not int or horizon < 0:
        raise ValueError("horizon must be a nonnegative integer")


def _terminals(graph: StateGraph, terminal_ids: Sequence[str] | None) -> tuple[str, ...] | None:
    if terminal_ids is None:
        return None
    if isinstance(terminal_ids, str):
        raise ValueError("terminal_ids must be a sequence of IDs, not a string")
    terminals = tuple(terminal_ids)
    if not terminals or len(set(terminals)) != len(terminals) or set(terminals) - set(graph.states):
        raise ValueError("terminal_ids must be nonempty, unique, known state IDs")
    return tuple(sorted(terminals))


def solve_receding_horizon(graph: StateGraph, start: str, horizon: int,
                          depth: int = 2, *, terminal_ids: Sequence[str] | None = None) -> SequenceResult:
    """Re-plan to depth, execute its first edge; terminal IDs only label outcome.

    No terminal reachability oracle or future terminal repair is supplied to this
    policy. It can finish outside the target; that outcome is retained explicitly.
    """
    _validate_request(graph, start, horizon)
    terminals = _terminals(graph, terminal_ids)
    if type(depth) is not int or depth < 1:
        raise ValueError("depth must be a positive integer")
    solver = _Solver(graph)
    path = [start]
    for remaining in range(horizon, 0, -1):
        plan = solver.plan(path[-1], min(depth, remaining))
        assert plan is not None  # Every certified state has a stay, no terminal constraint.
        path.append(plan.destinations[0])
    method = "greedy" if depth == 1 else f"receding_horizon_depth_{depth}"
    return solver.result(method, path, exact=False, terminal_ids=terminals)


def solve_greedy(graph: StateGraph, start: str, horizon: int, *,
                 terminal_ids: Sequence[str] | None = None) -> SequenceResult:
    """Select the best one-step service, with the same deterministic tie rules."""
    return solve_receding_horizon(graph, start, horizon, depth=1, terminal_ids=terminal_ids)


def solve_finite_horizon_exact(graph: StateGraph, start: str, horizon: int, *,
                              terminal_ids: Sequence[str] | None = None,
                              require_terminal: bool = False) -> SequenceResult | NoPathResult:
    """DP optimum restricted to this graph/horizon, optionally required terminals.

    Required-terminal and unrestricted references are different optimization
    problems and must be reported separately. Greedy/RHC only receive terminal
    labels for reporting, not a terminal-reachability oracle.
    """
    _validate_request(graph, start, horizon)
    terminals = _terminals(graph, terminal_ids)
    if type(require_terminal) is not bool or (require_terminal and terminals is None):
        raise ValueError("require_terminal must be Boolean and needs terminal_ids when true")
    solver = _Solver(graph, frozenset(terminals) if require_terminal else None)
    plan = solver.plan(start, horizon)
    method = "finite_graph_dynamic_program_required_terminal" if require_terminal else "finite_graph_dynamic_program"
    if plan is None:
        assert terminals is not None
        return NoPathResult(method, horizon, start, terminals, solver.accounting(0))
    return solver.result(method, (start,) + plan.destinations, exact=True, terminal_ids=terminals)


def solve_viability_filtered_greedy(graph: StateGraph, start: str, horizon: int,
                                    terminal_ids: Sequence[str], *,
                                    no_reinterruptions: bool = False) -> SequenceResult | NoPathResult:
    """Immediate greedy with an offline full-graph terminal-reachability oracle.

    Reverse BFS finds minimum distances to any service terminal, independently
    of rewards and cumulative-service DP. Because every state has a stay, a
    successor is viable exactly when its distance is no greater than the steps
    remaining after this operation. The one-step objective and tie rules are
    identical to ``solve_greedy``; only the viable action set changes.

    This method scans the entire supplied certified graph in advance. It is a
    mechanism reference, not a demonstrated lightweight online algorithm. With
    ``no_reinterruptions=True``, an additional explicit graph constraint removes
    every edge that drops any currently energized load (including earlier
    pickups). Report that constrained result separately from the default graph.
    """
    _validate_request(graph, start, horizon)
    terminals = _terminals(graph, terminal_ids)
    if terminals is None or type(no_reinterruptions) is not bool:
        raise ValueError("explicit terminal_ids and Boolean no_reinterruptions are required")
    solver = _Solver(graph, frozenset(terminals))
    solver.information_regime = "shared_saved_graph_offline_full_graph_viability_oracle"
    if no_reinterruptions:
        solver.edge_constraint = "supplied_edges_with_no_drop_of_currently_energized_loads"
    method = "viability_filtered_greedy" + ("_no_reinterruptions" if no_reinterruptions else "")
    reverse = {state_id: [] for state_id in graph.states}
    allowed = {}
    for state_id, state in graph.states.items():
        before = frozenset(state.energized_load_ids)
        destinations = []
        for target in solver.edges(state_id):
            solver.viability_edge_scans += 1
            if no_reinterruptions and not before.issubset(graph.states[target].energized_load_ids):
                continue
            destinations.append(target)
            reverse[target].append(state_id)
            solver.viability_allowed_edges += 1
        allowed[state_id] = tuple(destinations)
    distances = {terminal: 0 for terminal in terminals}
    pending = deque(terminals)
    while pending:
        target = pending.popleft()
        solver.viability_reverse_expansions += 1
        for predecessor in reverse[target]:
            if predecessor not in distances:
                distances[predecessor] = distances[target] + 1
                pending.append(predecessor)
    solver.viability_reachable_states = len(distances)
    solver.viability_distance_lookups += 1
    if start not in distances or distances[start] > horizon:
        return NoPathResult(method, horizon, start, terminals, solver.accounting(0),
                            edge_constraint=solver.edge_constraint)
    path = [start]
    for remaining in range(horizon, 0, -1):
        current = path[-1]
        before = frozenset(graph.states[current].energized_load_ids)
        candidates = []
        for target in solver.edges(current):
            solver.candidate_edges += 1
            if target not in allowed[current]:
                continue
            solver.viability_distance_lookups += 1
            if target not in distances or distances[target] > remaining - 1:
                continue
            value = solver.value(target)
            lost = before - frozenset(graph.states[target].energized_load_ids)
            candidates.append(_Plan((target,), value, value, len(lost), int(current != target)))
        # Initial reachability and each viability-filtered choice ensure a successor.
        assert candidates
        path.append(min(candidates, key=lambda p: p.order).destinations[0])
    assert path[-1] in terminals
    return solver.result(method, path, exact=False, terminal_ids=terminals)
