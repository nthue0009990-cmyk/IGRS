"""Complete NC13 graph adapter of the existing NC34 exact integer algorithm.

Only network dimensions, original normally-closed bits and the declared tie1
ban differ. Every original bus/load remains present; no electrical claim.
No OpenDSS or author environment import occurs in this module.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import networkx as nx

from array import array
from decimal import Decimal, InvalidOperation
from numbers import Integral
import time
import numpy as np



N_SWITCH = 4
N_DISPATCH = 7
N_COMMAND = N_SWITCH + N_DISPATCH


def _bits(values: Sequence[int]) -> list[int]:
    values = list(values)
    if len(values) != N_COMMAND or any(type(v) is not int or v not in (0, 1) for v in values):
        raise ValueError("NC13 command must contain four binary switch and seven binary load integers")
    return values


def _edge(value) -> tuple[str, str]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("each edge needs two named buses")
    if any(not isinstance(v, str) or not v for v in value):
        raise ValueError("bus names must be nonempty strings")
    result = tuple(v.lower() for v in value)
    if result[0] == result[1]:
        raise ValueError("self-loop bus edge is invalid")
    return result


def _context(static: Mapping, scenario: Mapping) -> dict:
    nodes = [str(n).lower() for n in static["nodes"]]
    if not nodes or len(nodes) != len(set(nodes)):
        raise ValueError("static nodes must be nonempty and unique case-insensitively")
    source = str(static["source_bus"]).lower()
    edges = [_edge(e) for e in static["bus_edges"]]
    switches = [_edge(e) for e in static["switch_edges"]]
    if len(switches) != N_SWITCH or len({frozenset(e) for e in switches}) != N_SWITCH:
        raise ValueError("NC13 requires four distinct controlled bus edges")
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    if set(graph) != set(nodes) or source not in graph:
        raise ValueError("source and edge buses must belong to the saved node inventory")
    if len(edges) != graph.number_of_edges():
        raise ValueError("saved bus edges must be unique undirected edges")
    original = {str(k).lower(): _edge(e) for k, e in static["original_lines"].items()}
    if len(original) != len(static["original_lines"]) or not original:
        raise ValueError("original line identities must be unique case-insensitively")
    if any(not graph.has_edge(*e) for e in [*switches, *original.values()]):
        raise ValueError("all switched/original lines must exist in the saved bus graph")
    faults = [str(x).lower() for x in scenario["fault_lines"]]
    if len(faults) != len(set(faults)) or not set(faults) <= set(original):
        raise ValueError("scenario faults must be distinct known original lines")
    fault_edges = [original[x] for x in faults]
    if len({frozenset(e) for e in fault_edges}) != len(fault_edges):
        raise ValueError("scenario fault edges must be unique")
    forced = [i == 2 or frozenset(e) in {frozenset(f) for f in fault_edges} for i, e in enumerate(switches)]
    dispatch = list(static["dispatch_loads"])
    if (len(dispatch) != N_DISPATCH or any(not isinstance(n, str) or not n for n in dispatch)
            or len({n.lower() for n in dispatch}) != N_DISPATCH):
        raise ValueError("NC13 requires seven distinct ordered dispatch loads")
    graph.remove_edges_from(fault_edges)
    return {"graph": graph, "source": source, "switches": switches,
            "forced_open": forced, "dispatch": dispatch}


def _loads(static: Mapping, loads: Mapping, context: Mapping) -> dict:
    if not isinstance(loads, Mapping) or not loads:
        raise ValueError("the original load inventory is required")
    result = {}
    for name, row in loads.items():
        if not isinstance(name, str) or not name or name.lower() in result:
            raise ValueError("load names must be unique case-insensitively")
        bus, kw = str(row["bus"]).lower(), row["nominal_kw"]
        if bus not in context["graph"] or isinstance(kw, bool) or not math.isfinite(kw) or kw < 0:
            raise ValueError("load bus and finite nonnegative nominal kW must be known")
        result[name.lower()] = {"name": name, "bus": bus, "nominal_kw": float(kw)}
    if not {n.lower() for n in context["dispatch"]} <= set(result):
        raise ValueError("dispatch load is absent from the nominal inventory")
    if sum(r["nominal_kw"] for r in result.values()) <= 0:
        raise ValueError("nominal load denominator must be positive")
    for key in ("all_load_buses", "dispatch_load_buses"):
        if key in static:
            references = {n.lower(): str(b).lower() for n, b in static[key].items()}
            expected = set(result) if key == "all_load_buses" else {n.lower() for n in context["dispatch"]}
            if set(references) != expected or any(result[n]["bus"] != b for n, b in references.items()):
                raise ValueError("saved static and nominal load bus inventories disagree")
    return result


def _metadata(context: Mapping, loads: Mapping, command: Sequence[int]) -> dict:
    command = _bits(command)
    if any(command[i] and forced for i, forced in enumerate(context["forced_open"])):
        raise ValueError("faulted switch command must already be forced open")
    graph = context["graph"].copy()
    graph.remove_edges_from(e for bit, e in zip(command[:N_SWITCH], context["switches"]) if not bit)
    powered = set(nx.node_connected_component(graph, context["source"]))
    effective = command.copy()
    dispatch = {name.lower(): i for i, name in enumerate(context["dispatch"])}
    for name, i in dispatch.items():
        if loads[name]["bus"] not in powered:
            effective[N_SWITCH + i] = 0
    energized, deenergized, shed = [], [], []
    for lower_name, row in loads.items():
        if row["bus"] not in powered:
            deenergized.append(row["name"])
        elif lower_name in dispatch and not effective[N_SWITCH + dispatch[lower_name]]:
            shed.append(row["name"])
        else:
            energized.append(row["name"])
    energized_lower = {n.lower() for n in energized}
    return {"command": command, "effective_action": effective,
            "forest": bool(nx.is_forest(graph)), "source_reachable_buses": sorted(powered),
            "deenergized_buses": sorted(set(graph) - powered),
            "energized_load_ids": sorted(energized), "deenergized_load_ids": sorted(deenergized),
            "deliberately_shed_load_ids": sorted(shed),
            "base_served_kw": sum(r["nominal_kw"] for n, r in loads.items() if n in energized_lower),
            "base_denominator_kw": sum(r["nominal_kw"] for r in loads.values()),
            "physical_feasibility_inferred": False}


def initial_command(static: Mapping, scenario: Mapping) -> list[int]:
    """Two normally closed sections, two open ties, seven ready loads; tie1/fault=0."""
    context = _context(static, scenario)
    switches = [1] * 2 + [0] * 2
    switches = [int(bit and not forced) for bit, forced in zip(switches, context["forced_open"])]
    return switches + [1] * N_DISPATCH


def state_metadata(static: Mapping, loads: Mapping, scenario: Mapping, command: Sequence[int]) -> dict:
    """Whole-graph forest and nominal source reachability; no physical certificate."""
    context = _context(static, scenario)
    return _metadata(context, _loads(static, loads, context), command)



def enumerate_switch_space(static: Mapping, loads: Mapping, scenario: Mapping) -> dict:
    """Enumerate all 16 raw assignments; graph edges flip one healthy switch.

    Mask bit ``i`` is switch coordinate ``i``. Seven readiness commands always stay
    one, including at unpowered buses. Effective load bits are derived from
    reachability. Invalid fault-closed assignments are not silently remapped to
    another command or counted as evaluated topologies. The caller separately
    certifies forest states at each condition and fixed operating point.
    """
    initial = initial_command(static, scenario)  # Validate static graph and faults.
    initial_meta = state_metadata(static, loads, scenario, initial)
    original = {str(k).lower(): frozenset(str(x).lower() for x in edge)
                for k, edge in static["original_lines"].items()}
    fault_edges = {original[str(name).lower()] for name in scenario["fault_lines"]}
    forced = [i for i, edge in enumerate(static["switch_edges"])
              if i == 2 or frozenset(str(x).lower() for x in edge) in fault_edges]
    states = {}
    for mask in range(1 << N_SWITCH):
        state_id = f"s{mask:03x}"
        command = [(mask >> i) & 1 for i in range(N_SWITCH)] + [1] * N_DISPATCH
        if any(command[i] for i in forced):
            metadata = {"command": command, "effective_action": None, "forest": None,
                        "source_reachable_buses": None, "deenergized_buses": None,
                        "energized_load_ids": None, "deenergized_load_ids": None,
                        "deliberately_shed_load_ids": None, "base_served_kw": None,
                        "base_denominator_kw": initial_meta["base_denominator_kw"],
                        "physical_feasibility_inferred": False}
            status = "FORCED_SWITCH_CLOSED"
        else:
            metadata = state_metadata(static, loads, scenario, command)
            status = "FOREST_TOPOLOGY_ONLY" if metadata["forest"] else "NONFOREST"
        states[state_id] = {"state_id": state_id, "mask": mask, **metadata,
                            "command_bits": command.copy(), "topology_status": status}
    successors = {}
    for sid, row in states.items():
        successors[sid] = sorted(f"s{row['mask'] ^ (1 << i):03x}" for i in range(N_SWITCH)
                                 if row["forest"] is True and
                                 states[f"s{row['mask'] ^ (1 << i):03x}"]["forest"] is True)
    arcs = sum(map(len, successors.values()))
    initial_mask = sum(bit << i for i, bit in enumerate(initial[:N_SWITCH]))
    return {"schema_version": 1, "scenario_id": scenario.get("scenario_id"),
            "initial_id": f"s{initial_mask:03x}", "initial_command": initial,
            "states": states, "successors": successors,
            "counts": {"switch_assignments": 1 << N_SWITCH,
                       "forest_states": sum(r["forest"] is True for r in states.values()),
                       "nonforest_states": sum(r["forest"] is False for r in states.values()),
                       "fault_closed_assignments": sum(r["forest"] is None for r in states.values()),
                       "directed_forest_edges": arcs, "undirected_forest_edges": arcs // 2},
            "forced_open_switch_indices": forced,
            "scope": "complete_four_switch_assignments_fixed_all_ready_free_terminal",
            "load_readiness_changes_allowed": False, "physical_feasibility_inferred": False,
            "new_physical_solves": 0,
            "stay_rule": "sequence solver adds stay at every physically certified state",
            "operating_point_rule": "caller must build a separate graph for each fixed alpha and condition"}


STATE_LIMIT = 1 << N_COMMAND
READY_MASK = (1 << N_DISPATCH) - 1
MAX_HORIZON = 12


def _integer_weights(loads: Mapping) -> tuple[dict[str, int], int]:
    """Interpret each saved nominal-kW decimal exactly, without float summation."""
    values = {}
    places = 0
    for name, row in loads.items():
        value = row["nominal_kw"]
        if isinstance(value, bool):
            raise ValueError("Nominal kW cannot be Boolean")
        try:
            decimal = Decimal(str(value))
        except InvalidOperation as exc:
            raise ValueError("Nominal kW must be a finite decimal") from exc
        if not decimal.is_finite() or decimal < 0:
            raise ValueError("Nominal kW must be finite and nonnegative")
        decimal = decimal.normalize()
        places = max(places, -decimal.as_tuple().exponent)
        key = str(name).lower()
        if key in values:
            raise ValueError("Load names must be unique case-insensitively")
        values[key] = decimal
    scale = 10 ** places
    weights = {name: int(value * scale) for name, value in values.items()}
    if not weights or sum(weights.values()) <= 0:
        raise ValueError("A positive complete nominal demand inventory is required")
    if sum(weights.values()) > np.iinfo(np.int64).max:
        raise OverflowError("Exact nominal weights exceed int64")
    return weights, scale


class JointGraph:
    """All pure-graph states reachable without interrupting any energized load.

    State encoding is ``(four_switch_bits << N_DISPATCH) | seven_ready_bits``. State IDs
    and every tie are ordered by this integer. A physical effective-action key
    has the same layout, but unreachable ready bits are zeroed. External caches
    must additionally identify the model, fault, condition, and fixed alpha.
    """

    def __init__(self, static: Mapping, loads: Mapping, scenario: Mapping):
        started = time.perf_counter()
        weights, self.weight_scale = _integer_weights(loads)
        self.total_weight = sum(weights.values())
        self.load_count = len(weights)
        self.scenario_id = scenario.get("scenario_id")
        catalog = enumerate_switch_space(static, loads, scenario)
        dispatch = [name.lower() for name in static["dispatch_loads"]]
        fixed_names = sorted(set(weights) - set(dispatch))
        dispatch_weights = [weights[name] for name in dispatch]
        self._reachable = np.zeros((1 << N_SWITCH), dtype=np.int32)
        fixed_bits = [0] * (1 << N_SWITCH)
        fixed_weight = np.zeros((1 << N_SWITCH), dtype=np.int64)
        neighbors = [[] for _ in range((1 << N_SWITCH))]
        forest_masks = []
        for row in catalog["states"].values():
            if row["forest"] is not True:
                continue
            mask = row["mask"]
            forest_masks.append(mask)
            energized = {name.lower() for name in row["energized_load_ids"]}
            fixed_bits[mask] = sum(1 << i for i, name in enumerate(fixed_names) if name in energized)
            fixed_weight[mask] = sum(weights[name] for name in fixed_names if name in energized)
            self._reachable[mask] = sum(1 << i for i, name in enumerate(dispatch) if name in energized)
            neighbors[mask] = [catalog["states"][sid]["mask"] for sid in catalog["successors"][row["state_id"]]]
        initial_mask = catalog["states"][catalog["initial_id"]]["mask"]
        self.initial_state = (initial_mask << N_DISPATCH) | READY_MASK
        # Precompute complete switch transitions. The fixed-load test includes
        # every uncontrollable load, not just dispatch loads or demand totals.
        switch_ok = np.zeros(((1 << N_SWITCH), N_SWITCH), dtype=np.bool_)
        dispatch_drop = np.zeros(((1 << N_SWITCH), N_SWITCH), dtype=np.int32)
        for mask in forest_masks:
            for other in neighbors[mask]:
                slot = (mask ^ other).bit_length() - 1
                switch_ok[mask, slot] = not (fixed_bits[mask] & ~fixed_bits[other])
                dispatch_drop[mask, slot] = int(self._reachable[mask]) & ~int(self._reachable[other])
        reached = bytearray(STATE_LIMIT)
        reached[self.initial_state] = 1
        queue = array("I", [self.initial_state])
        position = 0
        while position < len(queue):
            state = queue[position]
            position += 1
            mask, ready = state >> N_DISPATCH, state & READY_MASK
            powered = int(self._reachable[mask])
            for other in neighbors[mask]:
                slot = (mask ^ other).bit_length() - 1
                if not switch_ok[mask, slot] or ready & int(dispatch_drop[mask, slot]):
                    continue
                destination = (other << N_DISPATCH) | ready
                if not reached[destination]:
                    reached[destination] = 1
                    queue.append(destination)
            for bit in (1 << i for i in range(N_DISPATCH)):
                if ready & powered & bit:
                    continue
                destination = state ^ bit
                if not reached[destination]:
                    reached[destination] = 1
                    queue.append(destination)
        self.states = np.flatnonzero(np.frombuffer(reached, dtype=np.uint8)).astype(np.int32)
        self.states.flags.writeable = False
        count = len(self.states)
        self._index = np.full(STATE_LIMIT, -1, dtype=np.int32)
        self._index[self.states] = np.arange(count, dtype=np.int32)
        masks, ready = self.states >> N_DISPATCH, self.states & READY_MASK
        physical_ready = ready & self._reachable[masks]
        self._keys = (masks << N_DISPATCH) | physical_ready
        ready_weight = np.asarray([sum(dispatch_weights[i] for i in range(N_DISPATCH) if bits & (1 << i))
                                   for bits in range(1 << N_DISPATCH)], dtype=np.int64)
        self._weights = fixed_weight[masks] + ready_weight[physical_ready]
        # Row zero is stay; eleven rows are single-coordinate flips. Invalid
        # successors use -1. This is about 18 MB for the largest actual graph.
        self._next = np.full((N_COMMAND + 1, count), -1, dtype=np.int32)
        self._next[0] = np.arange(count, dtype=np.int32)
        for bit_index in range(N_DISPATCH):
            bit = 1 << bit_index
            valid = (ready & self._reachable[masks] & bit) == 0
            self._next[1 + bit_index, valid] = self._index[self.states[valid] ^ bit]
        for bit_index in range(N_SWITCH):
            valid = switch_ok[masks, bit_index] & ((ready & dispatch_drop[masks, bit_index]) == 0)
            self._next[N_DISPATCH + 1 + bit_index, valid] = self._index[self.states[valid] ^ (1 << (N_DISPATCH + bit_index))]
        if np.any(self._next[0] < 0) or np.any(self._weights < 0) or np.any(self._weights > self.total_weight):
            raise AssertionError("Incomplete graph or nominal weight inventory")
        self.graph_summary = {
            "scenario_id": self.scenario_id, "load_count": self.load_count,
            "forest_topologies": len(forest_masks), "reachable_topologies": len(np.unique(masks)),
            "reachable_command_states": count, "effective_action_keys": len(np.unique(self._keys)),
            "nonwaiting_directed_edges": int(np.count_nonzero(self._next[1:] >= 0)),
            "weight_scale": self.weight_scale, "total_weight": self.total_weight,
            "scope": "complete pure-graph no-reinterruptions reachability, no electrical certificate",
            "construction_seconds": time.perf_counter() - started,
        }
        # One FAIL-key population only. Values and choices for at most thirteen
        # layers use about 35 MB in the largest actual graph; other starts and
        # shorter horizons reuse the same exact all-state layers.
        self._dp_cache = None

    def _position(self, state: int) -> int:
        if isinstance(state, bool) or not isinstance(state, Integral) or not 0 <= state < STATE_LIMIT:
            raise ValueError("State must be an integer in the eleven-bit domain")
        index = int(self._index[int(state)])
        if index < 0:
            raise ValueError("State is outside the complete initial-reachable command graph")
        return index

    def effective_key(self, state: int) -> int:
        return int(self._keys[self._position(state)])

    def action(self, state: int) -> list[int]:
        key = self.effective_key(state)
        switches, ready = key >> N_DISPATCH, key & READY_MASK
        return [(switches >> i) & 1 for i in range(N_SWITCH)] + [(ready >> i) & 1 for i in range(N_DISPATCH)]

    def successors(self, state: int) -> tuple[int, ...]:
        column = self._next[:, self._position(state)]
        return tuple(sorted(int(value) for value in self.states[column[column >= 0]]))

    def served_base_kw(self, state: int) -> float:
        return int(self._weights[self._position(state)]) / self.weight_scale

    def served_weight(self, state: int) -> int:
        """Exact integer nominal service; divide by weight_scale only for display."""
        return int(self._weights[self._position(state)])

    def solve(self, start: int, horizon: int, blocked_keys=None) -> dict:
        """Exact optimistic route; ``path`` includes start and H destinations.

        Lexicographic objective: cumulative exact nominal service (initial not
        counted), final service, fewer non-wait operations, smallest state path.
        Positive common demand multipliers preserve this ordering. For every
        H<=12, T total integer demand, B=13 and A=(T+1)B, encode C*A+F*B+12-ops.
        Since 0<=F<=T and 0<=ops<=H<=12, each higher priority dominates all lower
        priorities. Fixed bases allow cached all-state layers to serve different
        starts/horizons. A Python-integer upper bound is checked before int64
        arithmetic; no overflow, floating summation, or tolerance resolves ties.
        """
        started = time.perf_counter()
        start_index = self._position(start)
        if isinstance(horizon, bool) or not isinstance(horizon, Integral) or not 0 <= horizon <= MAX_HORIZON:
            raise ValueError(f"Horizon must be an integer from zero through {MAX_HORIZON}")
        horizon = int(horizon)
        blocked = frozenset() if blocked_keys is None else frozenset(blocked_keys)
        if any(isinstance(key, bool) or not isinstance(key, Integral) or not 0 <= key < STATE_LIMIT for key in blocked):
            raise ValueError("Blocked effective keys must be eleven-bit integers")
        base = {"status": "NO_PATH", "path": [], "effective_keys": [], "cumulative_weight": None,
                "final_weight": None, "operations": None, "weight_scale": self.weight_scale,
                "total_weight": self.total_weight, "horizon": horizon, "blocked_key_count": len(blocked),
                "optimistic_unknown_nodes": True}
        if int(self._keys[start_index]) in blocked:
            return {**base, "dp_cache_hit": False, "dp_layers_computed_this_call": 0,
                    "elapsed_seconds": time.perf_counter() - started}
        lower_base = MAX_HORIZON + 1
        cumulative_base = (self.total_weight + 1) * lower_base
        upper = horizon * self.total_weight * cumulative_base + self.total_weight * lower_base + MAX_HORIZON
        if upper > np.iinfo(np.int64).max:
            raise OverflowError("Exact Bellman objective exceeds int64; do not round or rescale silently")
        if self._dp_cache is None or self._dp_cache["blocked"] != blocked:
            self._dp_cache = None
            blocked_lookup = np.zeros(STATE_LIMIT, dtype=np.bool_)
            if blocked:
                blocked_lookup[list(blocked)] = True
            allowed = ~blocked_lookup[self._keys]
            self._dp_cache = {"blocked": blocked, "allowed": allowed,
                              "values": [np.where(allowed, self._weights * lower_base + MAX_HORIZON, -1)],
                              "choices": [None], "service_addition": None}
        cache = self._dp_cache
        previous_layers = len(cache["values"]) - 1
        cache_hit = previous_layers >= horizon
        if horizon and cache["service_addition"] is None:
            cache["service_addition"] = self._weights * cumulative_base
        allowed = cache["allowed"]
        for remaining in range(previous_layers + 1, horizon + 1):
            value = cache["values"][-1]
            best = np.full(len(self.states), -1, dtype=np.int64)
            next_choice = np.full(len(self.states), -1, dtype=np.int32)
            for slot, destinations in enumerate(self._next):
                safe = np.maximum(destinations, 0)
                candidate = value[safe] + cache["service_addition"][safe] - int(slot != 0)
                valid = (destinations >= 0) & allowed & (value[safe] >= 0)
                # self.states is sorted, so index comparison is state comparison.
                replace = valid & ((candidate > best) | ((candidate == best) & (destinations < next_choice)))
                best[replace] = candidate[replace]
                next_choice[replace] = destinations[replace]
            cache["choices"].append(next_choice)
            cache["values"].append(best)
        value = cache["values"][horizon]
        if value[start_index] < 0:
            return {**base, "elapsed_seconds": time.perf_counter() - started}
        indices = [start_index]
        for remaining in range(horizon, 0, -1):
            destination = int(cache["choices"][remaining][indices[-1]])
            if destination < 0:
                raise AssertionError("Bellman route missing a stored decision")
            indices.append(destination)
        path = [int(self.states[index]) for index in indices]
        cumulative = sum(int(self._weights[index]) for index in indices[1:])
        final = int(self._weights[indices[-1]])
        operations = sum(a != b for a, b in zip(path, path[1:]))
        expected = cumulative * cumulative_base + final * lower_base + MAX_HORIZON - operations
        if expected != int(value[start_index]):
            raise AssertionError("Encoded Bellman value and traced exact objective differ")
        return {**base, "status": "COMPLETE", "path": path,
                "effective_keys": [int(self._keys[index]) for index in indices],
                "cumulative_weight": cumulative, "final_weight": final, "operations": operations,
                "objective_encoding": {"cumulative_base": cumulative_base, "terminal_base": lower_base,
                                       "operation_bonus": MAX_HORIZON, "encoding_horizon_cap": MAX_HORIZON,
                                       "safe_int64_upper_bound": upper, "encoded_objective": expected},
                "dp_cache_hit": cache_hit, "dp_layers_computed_this_call": max(0, horizon - previous_layers),
                "dp_cached_max_horizon": len(cache["values"]) - 1,
                "elapsed_seconds": time.perf_counter() - started}

