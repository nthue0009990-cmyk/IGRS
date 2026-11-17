"""Independent D043B command-graph and lazy-planning audit; zero DSS.

The graph uses full 68-load membership bitsets and direct one-bit candidate
transitions. The Bellman implementation keeps cumulative, terminal and operation
objectives separately; it does not import the production graph or planner.
Electrical certificates are inherited from separately sealed physical audits.
"""
from __future__ import annotations

import argparse
from array import array
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/audit_nature34_joint_planning.py"
TEST = "tests/test_nature34_joint_planning_audit.py"
ALPHAS = (0., .25, .5, .75, 1.)
HORIZONS = (6, 8, 12)
METHODS = ("greedy", "receding_horizon_depth_2", "receding_horizon_depth_3",
           "receding_horizon_depth_6", "finite_horizon_exact")
SCENARIOS = ("N2__line.l21__line.l24", "N2__line.l15__line.l22")
LIMIT = 1 << 19


def require(value, message):
    if not value:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized(name):
    path = PurePosixPath(name.replace("\\", "/"))
    require(not path.is_absolute() and path.parts and
            all(p not in (".", "..") and ":" not in p for p in path.parts), "Unsafe manifest path")
    return path.as_posix()


def consume(directory, digest, names):
    require(isinstance(digest, str) and re.fullmatch("[0-9a-f]{64}", digest), "Bad completion SHA")
    require(sha(directory / "completion_manifest.json") == digest, "Completion SHA changed")
    manifest = {}
    for name, value in read(directory / "completion_manifest.json").items():
        name = normalized(name)
        require(name not in manifest, "Ambiguous manifest path")
        manifest[name] = value
    used = {}
    for name in names:
        name = normalized(name)
        target = (directory / name).resolve()
        require(target.is_relative_to(directory.resolve()) and name in manifest, "Unsealed input: " + name)
        require(sha(target) == manifest[name], "Consumed input changed: " + name)
        used[name] = manifest[name]
    return {"run": directory.name, "completion_sha256": digest, "consumed_files": used}


def run_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / "runs" / path
    path = path.resolve()
    require(path.parent == (ROOT / "runs").resolve() and path.is_dir(), "Not an existing project run")
    return path


def close(actual, expected, label):
    require(type(actual) in (int, float) and math.isfinite(actual) and
            math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9), label)


def blocked_sha(keys):
    return hashlib.sha256(json.dumps(sorted(keys), separators=(",", ":")).encode()).hexdigest()


class IndependentGraph:
    """Direct membership test of every full command's one-bit neighbors.

    All topologies are independently reconstructed with disjoint sets and source
    BFS. A 68-bit integer represents actual energized load identity. Readiness
    is never used as a proxy for delivered supply, and equal kW never substitutes
    for set inclusion. This differs from production's transition decomposition.
    """

    def __init__(self, static, loads, scenario):
        began = time.perf_counter()
        nodes = tuple(str(x).lower() for x in static["nodes"])
        source = static["source_bus"].lower()
        edge = lambda value: tuple(sorted(str(x).lower() for x in value))
        base = {edge(e) for e in static["bus_edges"]}
        switches = tuple(edge(e) for e in static["switch_edges"])
        original = {str(k).lower(): edge(e) for k, e in static["original_lines"].items()}
        require(scenario["partition"] == "build", "Non-build graph")
        faults = {original[name.lower()] for name in scenario["fault_lines"]}
        require(len(nodes) == len(set(nodes)) and source in nodes and len(switches) == len(set(switches)) == 9,
                "Bad node/switch inventory")
        require(len(base) == len(static["bus_edges"]) and set(switches) <= base and
                all(len(e) == 2 and e[0] != e[1] and set(e) <= set(nodes) for e in base), "Bad bus edges")
        named = {name.lower(): row for name, row in loads.items()}
        order = sorted(named)
        dispatch = [name.lower() for name in static["dispatch_loads"]]
        require(len(named) == len(loads) == 68 and len(dispatch) == len(set(dispatch)) == 10 and
                set(dispatch) <= set(named), "Bad complete load inventory")
        fractions = [Fraction(str(named[name]["nominal_kw"])) for name in order]
        require(all(x >= 0 for x in fractions) and sum(fractions) > 0, "Invalid nominal service weights")
        self.scale = 1
        while any((x * self.scale).denominator != 1 for x in fractions):
            self.scale *= 10
        weights = [int(x * self.scale) for x in fractions]
        self.total_weight = sum(weights)
        require(self.total_weight * 12 < np.iinfo(np.int64).max, "Nominal objective overflows int64")
        self.load_names = order
        member = {name: 1 << i for i, name in enumerate(order)}
        dispatch_set = set(dispatch)
        ready_members = [0] * 1024
        ready_weights = [0] * 1024
        for ready in range(1, 1024):
            least = ready & -ready
            previous = ready ^ least
            name = dispatch[least.bit_length() - 1]
            ready_members[ready] = ready_members[previous] | member[name]
            ready_weights[ready] = ready_weights[previous] + weights[order.index(name)]
        normally_closed = {original["line." + row["line"].lower()] for row in static["sectional_switches"]}
        require(len(normally_closed) == 5 and normally_closed == set(switches[:5]), "Natural initial switches differ")
        initial_mask = sum(1 << i for i, e in enumerate(switches) if e in normally_closed and e not in faults)
        self.initial = (initial_mask << 10) | 1023
        # Nonforest/fault-closed commands have sentinel -1 and no transitions.
        energized = [-1] * LIMIT
        full_keys = np.full(LIMIT, -1, dtype=np.int32)
        full_weight = np.zeros(LIMIT, dtype=np.int64)
        forest_count = 0
        for mask in range(512):
            if any(mask & (1 << i) and e in faults for i, e in enumerate(switches)):
                continue
            kept = base - faults - {e for i, e in enumerate(switches) if not mask & (1 << i)}
            parents = dict(zip(nodes, nodes))
            def find(node):
                while parents[node] != node:
                    parents[node] = parents[parents[node]]
                    node = parents[node]
                return node
            adjacent = {node: [] for node in nodes}
            forest = True
            for a, b in kept:
                ra, rb = find(a), find(b)
                if ra == rb:
                    forest = False
                else:
                    parents[ra] = rb
                adjacent[a].append(b)
                adjacent[b].append(a)
            if not forest:
                continue
            forest_count += 1
            powered, pending = {source}, deque([source])
            while pending:
                for neighbor in adjacent[pending.popleft()]:
                    if neighbor not in powered:
                        powered.add(neighbor)
                        pending.append(neighbor)
            all_powered = {name for name in order if named[name]["bus"].lower() in powered}
            fixed = all_powered - dispatch_set
            fixed_members = sum(member[name] for name in fixed)
            fixed_weight = sum(weights[order.index(name)] for name in fixed)
            live = sum(1 << i for i, name in enumerate(dispatch) if name in all_powered)
            offset = mask << 10
            for ready in range(1024):
                key = ready & live
                energized[offset + ready] = fixed_members | ready_members[key]
                full_keys[offset + ready] = offset + key
                full_weight[offset + ready] = fixed_weight + ready_weights[key]
        require(energized[self.initial] >= 0, "Natural initial topology is invalid")
        seen = bytearray(LIMIT)
        seen[self.initial] = 1
        pending = array("I", [self.initial])
        edge_count = 0
        position = 0
        while position < len(pending):
            state = pending[position]
            position += 1
            before = energized[state]
            for bit in range(19):
                other = state ^ (1 << bit)
                after = energized[other]
                if after >= 0 and before & ~after == 0:
                    edge_count += 1
                    if not seen[other]:
                        seen[other] = 1
                        pending.append(other)
        self.states = np.asarray(sorted(pending), dtype=np.int32)
        self.index = np.full(LIMIT, -1, dtype=np.int32)
        self.index[self.states] = np.arange(len(self.states), dtype=np.int32)
        self.keys = full_keys[self.states]
        self.weights = full_weight[self.states]
        self.energized = energized
        self.next = np.full((len(self.states), 20), -1, dtype=np.int32)
        for i, raw in enumerate(self.states):
            state = int(raw)
            before = energized[state]
            successors = [state] + [state ^ (1 << bit) for bit in range(19)
                                     if energized[state ^ (1 << bit)] >= 0 and
                                     before & ~energized[state ^ (1 << bit)] == 0]
            successors.sort()
            targets = self.index[successors]
            require(np.all(targets >= 0), "Independent reachable graph is not closed")
            self.next[i, :len(targets)] = targets
        self.counts = {"scenario_id": scenario["scenario_id"], "load_count": len(order),
            "forest_topologies": forest_count, "reachable_topologies": len(np.unique(self.states >> 10)),
            "reachable_command_states": len(self.states), "effective_action_keys": len(np.unique(self.keys)),
            "nonwaiting_directed_edges": edge_count, "weight_scale": self.scale,
            "total_weight": self.total_weight, "construction_seconds": time.perf_counter() - began}

    def locate(self, state):
        require(type(state) is int and 0 <= state < LIMIT and self.index[state] >= 0, "Unknown command state")
        return int(self.index[state])

    def key(self, state):
        return int(self.keys[self.locate(state)])

    def path_values(self, path):
        require(bool(path), "Empty path")
        indices = [self.locate(state) for state in path]
        for a, b in zip(indices, indices[1:]):
            require(b in self.next[a], "Path violates one-bit operation or complete-load continuity")
        service = [int(self.weights[i]) for i in indices]
        return {"path": path, "effective_keys": [int(self.keys[i]) for i in indices],
                "cumulative_weight": sum(service[1:]), "final_weight": service[-1],
                "operations": sum(a != b for a, b in zip(path, path[1:])), "weights": service}


class IndependentDP:
    """Separate integer objective arrays, with sorted-successor tie resolution."""

    def __init__(self, graph, blocked, max_horizon=12):
        self.graph = graph
        self.blocked = frozenset(blocked)
        allowed = ~np.isin(graph.keys, list(blocked))
        size = len(graph.states)
        cumulative = np.where(allowed, 0, -1).astype(np.int64)
        terminal = np.where(allowed, graph.weights, -1).astype(np.int64)
        operations = np.zeros(size, dtype=np.int16)
        self.choices = [None]
        self.layers = [(cumulative, terminal, operations)]
        origins = np.arange(size, dtype=np.int32)
        for _ in range(max_horizon):
            best_c = np.full(size, -1, dtype=np.int64)
            best_t = np.full(size, -1, dtype=np.int64)
            best_o = np.full(size, 32767, dtype=np.int16)
            selected = np.full(size, -1, dtype=np.int32)
            for slot in range(graph.next.shape[1]):
                dest = graph.next[:, slot]
                safe = np.maximum(dest, 0)
                candidate_c = cumulative[safe] + graph.weights[safe]
                candidate_t = terminal[safe]
                candidate_o = operations[safe] + (safe != origins)
                valid = allowed & (dest >= 0) & (cumulative[safe] >= 0)
                better = ((candidate_c > best_c) | ((candidate_c == best_c) &
                          ((candidate_t > best_t) | ((candidate_t == best_t) & (candidate_o < best_o)))))
                take = valid & better
                best_c[take], best_t[take], best_o[take] = candidate_c[take], candidate_t[take], candidate_o[take]
                selected[take] = dest[take]
            cumulative, terminal, operations = best_c, best_t, best_o
            self.layers.append((cumulative, terminal, operations))
            self.choices.append(selected)

    def route(self, start, horizon):
        require(type(horizon) is int and 0 <= horizon < len(self.layers), "Invalid DP horizon")
        index = self.graph.locate(start)
        require(self.layers[horizon][0][index] >= 0, "No optimistic route")
        path = [start]
        for remaining in range(horizon, 0, -1):
            index = int(self.choices[remaining][index])
            require(index >= 0, "Missing DP successor")
            path.append(int(self.graph.states[index]))
        values = self.graph.path_values(path)
        source = self.graph.locate(start)
        for name, column in zip(("cumulative_weight", "final_weight", "operations"), self.layers[horizon]):
            require(values[name] == int(column[source]), "Independent path objective differs from DP")
        return values


def check_path_record(record, expected, label):
    for name in ("path", "effective_keys", "cumulative_weight", "final_weight", "operations"):
        require(record.get(name) == expected[name], label + ": " + name)


def validate_query_population(config, scenarios, conditions, queries):
    expected = []
    for si, scenario in enumerate(scenarios):
        for ci, condition in enumerate(conditions):
            for ai, alpha in enumerate(ALPHAS):
                for horizon in HORIZONS:
                    for method in METHODS:
                        expected.append({"query_id": len(expected), "cell_index": si * len(conditions) + ci,
                            "scenario_id": scenario["scenario_id"], "condition_id": condition["condition_id"],
                            "alpha_index": ai, "alpha": alpha, "horizon": horizon, "method": method})
    require(len(expected) == len(queries) == config["expected_queries"] == 450, "Missing method rows")
    for actual, identity in zip(queries, expected):
        require(all(actual.get(k) == v for k, v in identity.items()), "Changed/duplicate query identity")
        require(actual["status"] in ("COMPLETE", "UNRESOLVED_BUDGET"), "Unexpected final query status")
    return expected


def independent_certificates(actions, source_name, source_digest=None):
    """Raw sealed action records supply every alpha, never selected-alpha fields."""
    cells = defaultdict(dict)
    for action_id, action in enumerate(actions):
        require(action["action_id"] == action_id and action["partition"] == "build", "Action identity/partition differs")
        bits = action["action"]
        require(len(bits) == 19 and all(type(bit) is int and bit in (0, 1) for bit in bits), "Invalid effective action")
        key = (sum(bit << i for i, bit in enumerate(bits[:9])) << 10) | sum(bit << i for i, bit in enumerate(bits[9:]))
        cell = action["scenario_id"], action["condition_id"]
        require(key not in cells[cell] and len(action["points"]) == 5, "Repeated or incomplete new action")
        certificates = []
        for ai, point in enumerate(action["points"]):
            require(point["point_id"] == 5 * action_id + ai and point["alpha"] == ALPHAS[ai] and
                    point["converged"] is True and point["ac_status"] == point["isolation_status"] == "PASS" and
                    type(point["known_constraints_satisfied"]) is bool, "Unreliable/incomplete alpha certificate")
            certificates.append({"pass": point["known_constraints_satisfied"], "source_run": source_name,
                "source_completion_sha256": source_digest,
                "source_action_id": action_id, "source_point_id": point["point_id"], "alpha": point["alpha"],
                "nominal_restored_fraction": point["nominal_restored_fraction"]})
        cells[cell][key] = certificates
    return dict(cells)


def merge_certificates(old, new):
    require(set(new) <= set(old), "New certificate has an unknown fault/condition")
    merged = {cell: dict(values) for cell, values in old.items()}
    for cell, additions in new.items():
        require(not (set(additions) & set(merged[cell])), "Paid duplicate of old certified effective action")
        merged[cell].update(additions)
    return merged


def check_certificate(reference, expected, label):
    """Require immutable point identity, alpha and PASS/FAIL for every citation."""
    require(isinstance(reference, dict), label + ": missing certificate")
    for name in ("pass", "source_run", "source_completion_sha256", "source_action_id", "source_point_id", "alpha"):
        require(reference.get(name) == expected[name] and
                (name != "pass" or type(reference[name]) is bool), label + ": " + name)


def verify_planning(config, scenarios, conditions, graphs, queries, events, baseline, additions,
                    actions, actionplans, physical_cells, output=None):
    identities = validate_query_population(config, scenarios, conditions, queries)
    merged = merge_certificates(baseline, additions)
    condition_by_id = {row["condition_id"]: row for row in conditions}
    cell_list = [(s["scenario_id"], c["condition_id"]) for s in scenarios for c in conditions]
    require(set(baseline) == set(merged) == set(cell_list), "Missing baseline/cell population")
    paid = Counter((row["scenario_id"], row["condition_id"]) for row in actions)
    expected_cells = [{"cell_index": i, "scenario_id": sid, "condition_id": cid,
                       "paid_actions": paid[sid, cid], "maximum_actions": config["max_new_actions_per_cell"]}
                      for i, (sid, cid) in enumerate(cell_list)]
    require(physical_cells == expected_cells, "Physical/planning cell budget disagrees")
    require(all(n <= config["max_new_actions_per_cell"] for n in paid.values()) and
            len(actions) <= config["max_new_actions"], "Exceeded fixed resource budget")
    require(len(actions) == len(actionplans), "Actionplan population differs")
    visit_ids = [plan["visit_id"] for plan in actionplans]
    require(all(type(visit) is int and visit >= 0 for visit in visit_ids) and len(set(visit_ids)) == len(visit_ids),
            "A query visit paid for more than one new effective action")
    # Maps the action's global completion position to its physically evaluated key.
    new_identity = []
    for action, plan in zip(actions, actionplans):
        cell = action["scenario_id"], action["condition_id"]
        graph = graphs[cell[0]]
        state = plan.get("command_state_id", plan["command_state"])
        if isinstance(state, list):
            state = (sum(bit << i for i, bit in enumerate(state[:9])) << 10) | sum(bit << i for i, bit in enumerate(state[9:]))
        key = graph.key(state)
        require(key == plan["effective_key"] and key in additions[cell], "Paid command/key differs")
        point = additions[cell][key][0]
        require(point["source_action_id"] == action["action_id"], "New certificate source action differs")
        query_id = plan["query_id"]
        require(type(query_id) is int and 0 <= query_id < len(queries) and
                (identities[query_id]["scenario_id"], identities[query_id]["condition_id"]) == cell,
                "Paid request belongs to another query/cell")
        new_identity.append((cell, key))
    # Replay only evidence availability and executed prefixes; no production
    # scheduling helper is imported. Certified plan optimality is proven below
    # against the final excluded set (a superset that preserves that whole path).
    prefixes = {i: [graphs[q["scenario_id"]].initial] for i, q in enumerate(identities)}
    plan_counts = Counter()
    events_by_group = defaultdict(list)
    available = {cell: dict(records) for cell, records in baseline.items()}
    completed = 0
    consumed_request_events = set()
    for event_id, event in enumerate(events):
        require(event["event_id"] == event_id, "Planning event sequence differs")
        target_count = event["new_actions_completed"]
        require(type(target_count) is int and completed <= target_count <= len(actions), "Nonmonotone event certificate clock")
        while completed < target_count:
            cell, key = new_identity[completed]
            plan = actionplans[completed]
            request_id = plan["event_id"]
            require(type(request_id) is int and 0 <= request_id < event_id and request_id not in consumed_request_events,
                    "Paid action lacks a unique earlier unknown request event")
            request = events[request_id]
            require(request["outcome"] == "NEEDS_CERTIFICATE" and request["query_id"] == plan["query_id"] and
                    request["new_actions_completed"] == completed, "Paid request timing/identity differs")
            unknowns = [r for r in request["path_certificates"] if r["certificate"] is None]
            require(unknowns and unknowns[0]["effective_key"] == key, "Paid action is not first unknown path key")
            require(key not in available[cell], "Paid key was already available")
            consumed_request_events.add(request_id)
            available[cell][key] = additions[cell][key]
            completed += 1
        qid = event["query_id"]
        require(type(qid) is int and 0 <= qid < len(identities), "Unknown event query")
        query = identities[qid]
        for name in ("cell_index", "scenario_id", "condition_id", "alpha", "horizon", "method"):
            require(event[name] == query[name], "Event query identity differs: " + name)
        cell = query["scenario_id"], query["condition_id"]
        ai = query["alpha_index"]
        graph = graphs[cell[0]]
        prefix = prefixes[qid]
        remaining = query["horizon"] + 1 - len(prefix)
        require(remaining > 0 and event["current_state"] == prefix[-1] and event["remaining"] == remaining,
                "Event does not begin at its actually certified prefix")
        method = query["method"]
        depth = remaining if method == "finite_horizon_exact" else min(remaining,
                1 if method == "greedy" else int(method.rsplit("_", 1)[1]))
        require(event["planning_depth"] == depth and len(event["optimistic_path"]) == depth + 1,
                "Wrong rolling lookahead depth")
        blocked = {key for key, points in available[cell].items() if points[ai]["pass"] is False}
        require(event["blocked_keys_sha256"] == blocked_sha(blocked) and event["blocked_key_count"] == len(blocked),
                "Historical FAIL population digest differs")
        path = event["optimistic_path"]
        expected = graph.path_values(path)
        require(path[0] == prefix[-1], "Candidate starts at another command")
        for name in ("cumulative_weight", "final_weight", "operations"):
            require(event[name] == expected[name], "Candidate path objective differs")
        refs = event["path_certificates"]
        require(len(refs) == len(path), "Candidate certificate count differs")
        unknowns = 0
        for state, key, ref in zip(path, expected["effective_keys"], refs):
            require(ref["state"] == state and ref["effective_key"] == key, "Candidate certificate coordinate differs")
            require(key not in blocked, "Candidate contains known FAIL")
            if key not in available[cell]:
                require(ref["certificate"] is None, "Future certificate used before available")
                unknowns += 1
            else:
                check_certificate(ref["certificate"], available[cell][key][ai], "Candidate certificate")
        require(event["outcome"] == ("NEEDS_CERTIFICATE" if unknowns else "PLAN_CERTIFIED"),
                "Unknown candidate mislabeled certified")
        if not unknowns:
            plan_counts[qid] += 1
            if method == "finite_horizon_exact":
                prefix.extend(path[1:])
            else:
                prefix.append(path[1])
        events_by_group[cell + (ai,)].append(event)
    require(completed == len(actions) and len(consumed_request_events) == len(actions),
            "Final event stream does not observe every completed paid action")
    require(available == merged, "Final available certificate population differs")
    checks, event_checks = [], []
    for cell_index, cell in enumerate(cell_list):
        graph = graphs[cell[0]]
        condition = condition_by_id[cell[1]]
        scale = Fraction(str(condition["load_multiplier"])) / graph.scale
        for ai, alpha in enumerate(ALPHAS):
            certs = {key: points[ai] for key, points in merged[cell].items()}
            require(certs[graph.key(graph.initial)]["pass"] is True, "Initial alpha lacks a PASS certificate")
            blocked = {key for key, cert in certs.items() if cert["pass"] is False}
            dp = IndependentDP(graph, blocked, 12)
            local_count = 0
            for event in events_by_group[cell + (ai,)]:
                if event["outcome"] != "PLAN_CERTIFIED":
                    continue
                expected = dp.route(event["current_state"], event["planning_depth"])
                require(event["optimistic_path"] == expected["path"], "Certified local path is not independently optimal")
                for name in ("cumulative_weight", "final_weight", "operations"):
                    require(event[name] == expected[name], "Certified local DP objective differs")
                local_count += 1
            group_queries = [q for q in queries if q["cell_index"] == cell_index and q["alpha_index"] == ai]
            require(len(group_queries) == 15, "Missing fixed-alpha method/H group")
            for query in group_queries:
                qid, horizon = query["query_id"], query["horizon"]
                prefix = prefixes[qid]
                require(query["path"] == prefix and query["certified_local_plans"] == plan_counts[qid],
                        "Final query prefix/decision accounting differs")
                complete = len(prefix) == horizon + 1
                require(query["status"] == ("COMPLETE" if complete else "UNRESOLVED_BUDGET") and
                        query["fallback_is_original_method"] is complete, "Fallback/completion mislabeled")
                if not complete:
                    require(paid[cell] == config["max_new_actions_per_cell"], "Unresolved query with unused cell budget")
                    remaining = horizon + 1 - len(prefix)
                    depth = remaining if query["method"] == "finite_horizon_exact" else min(remaining,
                        1 if query["method"] == "greedy" else int(query["method"].rsplit("_", 1)[1]))
                    final_proposal = dp.route(prefix[-1], depth)
                    require(any(key not in certs for key in final_proposal["effective_keys"]),
                            "Unresolved query could be closed without new physics")
                fallback = prefix + [prefix[-1]] * (horizon + 1 - len(prefix))
                expected = graph.path_values(fallback)
                check_path_record(query["certified_feasible_path"], expected, "Feasible fallback")
                for metric, value in (("cumulative_kw_steps", float(expected["cumulative_weight"] * scale)),
                        ("final_served_kw", float(expected["final_weight"] * scale)),
                        ("mean_served_fraction", expected["cumulative_weight"] / (horizon * graph.total_weight))):
                    close(query["certified_feasible_path"][metric], value, "Wrong reported " + metric)
                require(len(query["certified_feasible_path"]["path_served_kw"]) == len(fallback), "Missing step service")
                for actual, weight in zip(query["certified_feasible_path"]["path_served_kw"], expected["weights"]):
                    close(actual, float(weight * scale), "Step served kW differs")
                require(len(query["path_certificates"]) == len(fallback), "Fallback lacks full certificate coverage")
                for key, ref in zip(expected["effective_keys"], query["path_certificates"]):
                    require(key in certs and certs[key]["pass"] is True, "Fallback includes UNKNOWN/FAIL")
                    check_certificate(ref, certs[key], "Fallback certificate")
                upper = dp.route(graph.initial, horizon)
                check_path_record(query["reference_optimistic_upper_bound"], upper, "Complete optimistic reference")
                require(query["reference_optimistic_upper_bound"]["status"] == "COMPLETE", "Reference DP incomplete")
                certified = all(certs.get(key, {}).get("pass") is True for key in upper["effective_keys"])
                require(query["reference_upper_is_certified"] is certified, "Upper certification mislabeled")
                score = lambda r: (-r["cumulative_weight"], -r["final_weight"], r["operations"], tuple(r["path"][1:]))
                require(score(upper) <= score(expected), "Feasible lower bound exceeds optimistic upper")
                if query["method"] == "finite_horizon_exact" and complete:
                    require(certified and upper["path"] == fallback, "Completed exact reference not proven optimal")
                checks.append({"query_id": qid, "status": query["status"], "certified_local_plans": plan_counts[qid],
                    "lower_cumulative_weight": expected["cumulative_weight"],
                    "upper_cumulative_weight": upper["cumulative_weight"], "reference_upper_is_certified": certified})
            event_checks.append({"cell_index": cell_index, "scenario_id": cell[0], "condition_id": cell[1],
                "alpha": alpha, "final_fail_keys": len(blocked), "final_fail_sha256": blocked_sha(blocked),
                "certified_local_plans_independently_optimal": local_count,
                "planning_events": len(events_by_group[cell + (ai,)])})
    checks.sort(key=lambda row: row["query_id"])
    if output is not None:
        save(output / "query_checks.json", checks)
        save(output / "planning_event_checks.json", event_checks)
        save(output / "cell_accounting.json", expected_cells)
    return {"queries": len(checks), "exact_reference_queries": 90, "baseline_queries": 360,
        "query_statuses": dict(Counter(q["status"] for q in queries)), "planning_events": len(events),
        "certified_local_plans": sum(plan_counts.values()), "paid_requests": len(consumed_request_events),
        "all_alpha_certificates": sum(5 * len(values) for values in merged.values()),
        "certificate_statuses": dict(Counter(str(c["pass"]) for values in merged.values()
                                               for points in values.values() for c in points)),
        "historical_unknown_proposals_optimality_recomputed": False,
        "certified_plan_optimality": "independent tuple DP on final FAIL exclusion set; certified paths preserved"}


def seal(output, result):
    save(output / "verification.json", result)
    save(output / "summary.json", result)
    save(output / "completion_manifest.json", {p.relative_to(output).as_posix(): sha(p)
        for p in output.rglob("*") if p.is_file() and p.name != "completion_manifest.json"})
    pointer = {**result, "run": output.name, "completion_sha256": sha(output / "completion_manifest.json")}
    suffix = "failed" if result["errors"] else "verified"
    save(ROOT / "artifacts" / ("nature34_joint_planning_" + suffix + ".json"), pointer)
    print(json.dumps(pointer), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--physical-audit", required=True)
    parser.add_argument("--physical-audit-sha256", required=True)
    args = parser.parse_args(argv)
    output = ROOT / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_nature34_joint_planning_audit")
    output.mkdir()
    save(output / "config.json", vars(args))
    save(output / "dependencies.json", {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()})
    save(output / "python_environment.json", {"executable": sys.executable, "version": sys.version,
                                               "sha256": sha(Path(sys.executable))})
    for name in (SELF, TEST):
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    save(output / "preaudit_manifest.json", {p.relative_to(output).as_posix(): sha(p)
                                              for p in output.rglob("*") if p.is_file()})
    result = {"status": "FAILED_JOINT_PLANNING", "errors": [], "new_powerflow_calls": 0,
        "new_training_interactions": 0, "new_optimizer_steps": 0, "formal_holdout_numerical_access": False,
        "tune_numerical_access": False, "production_graph_or_planner_imported": False,
        "physical_certification_repeated": False}
    began = time.perf_counter()
    try:
        command = [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                   "--basetemp", str(output / "test_tmp"), TEST]
        save(output / "test_command.json", command)
        tested = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        (output / "tests_stdout.txt").write_text(tested.stdout, encoding="utf-8")
        (output / "tests_stderr.txt").write_text(tested.stderr, encoding="utf-8")
        require(tested.returncode == 0, "Independent planning auditor software checks failed")
        source, physical_audit = run_path(args.run), run_path(args.physical_audit)
        source_names = ["summary.json", "config.json", "prepared.json", "author_static_metadata.json",
            "load_reference.json", "scenarios.json", "conditions.json", "actions.jsonl", "actionplans.jsonl",
            "planning_events.jsonl", "query_results.json", "cell_accounting.json", "graph_summaries.json",
            "source/" + SELF, "source/" + TEST]
        source_gate = consume(source, args.sha256, source_names)
        audit_gate = consume(physical_audit, args.physical_audit_sha256,
                             ["verification.json", "physical_counts.json", "cell_accounting.json"])
        verdict = read(physical_audit / "verification.json")
        require(verdict["status"] == "VERIFIED_JOINT_SEQUENCE_SAVED_RESULTS" and verdict["errors"] == [] and
                verdict["source_run"] == source.name and verdict["source_completion_sha256"] == args.sha256 and
                verdict["new_powerflow_calls"] == verdict["new_training_interactions"] == 0 and
                verdict["formal_holdout_numerical_access"] is verdict["tune_numerical_access"] is False,
                "Missing independent physical certification of this run")
        require(all(sha(source / "source" / name) == sha(ROOT / name) for name in (SELF, TEST)),
                "Independent planning code differs from prephysical freeze")
        prepared = read(source / "prepared.json")
        require({SELF, TEST} <= set(prepared["source_files"]), "Planning audit omitted from frozen sources")
        config = read(source / "config.json")
        fixed = {"decision": "D043B", "execution_allowed": True, "scenario_ids": list(SCENARIOS),
            "fixed_alphas": list(ALPHAS), "horizons": list(HORIZONS), "methods": list(METHODS),
            "regime": "no_reinterruptions", "max_new_actions_per_cell": 128, "max_new_actions": 768,
            "max_new_points": 3840, "max_numerical_entries": 15363, "expected_queries": 450,
            "new_training_interactions": 0, "tune_access": False, "test_access": False}
        require(all(type(config.get(k)) is type(v) and config[k] == v for k, v in fixed.items()),
                "Frozen scope/budget/method population differs")
        summary = read(source / "summary.json")
        require(summary["status"] == "COMPLETE_JOINT_SEQUENCE_PENDING_SAVED_AUDIT" and
                summary["new_training_interactions"] == 0 and summary["formal_holdout_numerical_access"] is False and
                summary["tune_numerical_access"] is False, "Incomplete joint source run")
        baseline, old_audit = run_path(config["source_run"]), run_path(config["source_audit_run"])
        old_names = ["actions.jsonl", "config.json", "scenarios.json", "conditions.json",
                     "author_static_metadata.json", "load_reference.json"]
        baseline_gate = consume(baseline, config["source_sha256"], old_names)
        old_gate = consume(old_audit, config["source_audit_sha256"], ["verification.json", "physical_counts.json"])
        old_verdict = read(old_audit / "verification.json")
        require(old_verdict["status"] == "VERIFIED_FREE_SEQUENCE_SAVED_RESULTS" and old_verdict["errors"] == [] and
                old_verdict["source_run"] == baseline.name and
                old_verdict["source_completion_sha256"] == config["source_sha256"], "Baseline lacks sealed physical audit")
        for name in ("scenarios.json", "conditions.json", "author_static_metadata.json", "load_reference.json"):
            require(sha(source / name) == sha(baseline / name), "Population/graph source differs: " + name)
        scenarios, conditions = read(source / "scenarios.json"), read(source / "conditions.json")
        require([s["scenario_id"] for s in scenarios] == list(SCENARIOS) and
                len(conditions) == len({c["condition_id"] for c in conditions}) == 3,
                "Original two faults/three conditions changed")
        static, loads = read(source / "author_static_metadata.json"), read(source / "load_reference.json")
        graphs = {s["scenario_id"]: IndependentGraph(static, loads, s) for s in scenarios}
        production_counts = read(source / "graph_summaries.json")
        if isinstance(production_counts, dict):
            production_counts = [production_counts[s["scenario_id"]] for s in scenarios]
        require(len(production_counts) == len(graphs), "Missing graph summaries")
        independent_counts = [graphs[s["scenario_id"]].counts for s in scenarios]
        for expected, actual in zip(independent_counts, production_counts):
            for key, value in expected.items():
                if key != "construction_seconds":
                    require(actual.get(key) == value, "Complete independent graph mismatch: " + key)
        save(output / "independent_graph_counts.json", independent_counts)
        old_actions, actions = rows(baseline / "actions.jsonl"), rows(source / "actions.jsonl")
        base_certs = independent_certificates(old_actions, baseline.name, config["source_sha256"])
        new_certs = independent_certificates(actions, source.name)
        require(len(old_actions) == 2448 and len(actions) == verdict["actions"] and
                5 * len(actions) == verdict["points"], "Physical/certificate population differs")
        for action_list, counts in ((old_actions, read(old_audit / "physical_counts.json")),
                                   (actions, read(physical_audit / "physical_counts.json"))):
            seen = Counter(str(point["known_constraints_satisfied"]) for a in action_list for point in a["points"])
            require(dict(seen) == counts["known_constraints"], "Physical PASS/FAIL certificates omitted")
        physical_cells = read(physical_audit / "cell_accounting.json")
        require(read(source / "cell_accounting.json") == physical_cells, "Saved/physical cell accounting differs")
        checked = verify_planning(config, scenarios, conditions, graphs, read(source / "query_results.json"),
            rows(source / "planning_events.jsonl"), base_certs, new_certs, actions,
            rows(source / "actionplans.jsonl"), physical_cells, output)
        gates = [(source, source_gate), (physical_audit, audit_gate), (baseline, baseline_gate), (old_audit, old_gate)]
        for directory, gate in gates:
            consume(directory, gate["completion_sha256"], gate["consumed_files"])
        save(output / "consumed_sources.json", {name: gate for name, (_, gate) in zip(
            ("source", "physical_audit", "baseline", "baseline_audit"), gates)})
        require(not any(name in sys.modules for name in ("opendssdirect", "dss", "nature34_joint_scheduler",
            "dnr.nature34_joint_sequence", "dnr.nature34_free_sequence")), "Independent audit imported production planning/engine")
        result.update(status="VERIFIED_JOINT_PLANNING", source_run=source.name,
            source_completion_sha256=args.sha256, physical_audit_run=physical_audit.name,
            physical_audit_completion_sha256=args.physical_audit_sha256,
            graphs=len(graphs), **checked,
            scope="Complete independent 68-load graph, final optimistic references, all certified local plans, "
                  "historical certificate availability/FAIL hashes, actual prefixes, bounded unresolved fallbacks and full population")
    except BaseException as exc:
        result["errors"].append({"error": repr(exc), "traceback": traceback.format_exc()})
    result["elapsed_seconds"] = time.perf_counter() - began
    seal(output, result)
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

