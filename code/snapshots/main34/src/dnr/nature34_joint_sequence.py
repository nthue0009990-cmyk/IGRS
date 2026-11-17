"""Complete command-state graph and exact integer optimistic Bellman search.

No physical engine is imported. Electrical FAIL keys may be excluded; UNKNOWN
keys remain optimistic. A returned route is not an electrical certificate.
Load readiness remains part of the decision state even when it has no current
physical effect. All loads, including uncontrollable ones, impose continuity.
"""
from __future__ import annotations

from array import array
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from numbers import Integral
import time

import numpy as np

from .nature34_free_sequence import enumerate_switch_space


STATE_LIMIT = 1 << 19
READY_MASK = (1 << 10) - 1
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

    State encoding is ``(nine_switch_bits << 10) | ten_ready_bits``. State IDs
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
        self._reachable = np.zeros(512, dtype=np.int32)
        fixed_bits = [0] * 512
        fixed_weight = np.zeros(512, dtype=np.int64)
        neighbors = [[] for _ in range(512)]
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
        self.initial_state = (initial_mask << 10) | READY_MASK
        # Precompute complete switch transitions. The fixed-load test includes
        # every uncontrollable load, not just dispatch loads or demand totals.
        switch_ok = np.zeros((512, 9), dtype=np.bool_)
        dispatch_drop = np.zeros((512, 9), dtype=np.int32)
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
            mask, ready = state >> 10, state & READY_MASK
            powered = int(self._reachable[mask])
            for other in neighbors[mask]:
                slot = (mask ^ other).bit_length() - 1
                if not switch_ok[mask, slot] or ready & int(dispatch_drop[mask, slot]):
                    continue
                destination = (other << 10) | ready
                if not reached[destination]:
                    reached[destination] = 1
                    queue.append(destination)
            for bit in (1 << i for i in range(10)):
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
        masks, ready = self.states >> 10, self.states & READY_MASK
        physical_ready = ready & self._reachable[masks]
        self._keys = (masks << 10) | physical_ready
        ready_weight = np.asarray([sum(dispatch_weights[i] for i in range(10) if bits & (1 << i))
                                   for bits in range(1024)], dtype=np.int64)
        self._weights = fixed_weight[masks] + ready_weight[physical_ready]
        # Row zero is stay; nineteen rows are single-coordinate flips. Invalid
        # successors use -1. This is about 18 MB for the largest actual graph.
        self._next = np.full((20, count), -1, dtype=np.int32)
        self._next[0] = np.arange(count, dtype=np.int32)
        for bit_index in range(10):
            bit = 1 << bit_index
            valid = (ready & self._reachable[masks] & bit) == 0
            self._next[1 + bit_index, valid] = self._index[self.states[valid] ^ bit]
        for bit_index in range(9):
            valid = switch_ok[masks, bit_index] & ((ready & dispatch_drop[masks, bit_index]) == 0)
            self._next[11 + bit_index, valid] = self._index[self.states[valid] ^ (1 << (10 + bit_index))]
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
            raise ValueError("State must be an integer in the nineteen-bit domain")
        index = int(self._index[int(state)])
        if index < 0:
            raise ValueError("State is outside the complete initial-reachable command graph")
        return index

    def effective_key(self, state: int) -> int:
        return int(self._keys[self._position(state)])

    def action(self, state: int) -> list[int]:
        key = self.effective_key(state)
        switches, ready = key >> 10, key & READY_MASK
        return [(switches >> i) & 1 for i in range(9)] + [(ready >> i) & 1 for i in range(10)]

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
            raise ValueError("Blocked effective keys must be nineteen-bit integers")
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
