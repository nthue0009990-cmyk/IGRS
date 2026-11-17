"""Pure command-state catalogs for a known-target NC34 ordering diagnostic.

Command load bits preserve readiness while the bus is deenergized. The existing
physical evaluator receives the effective action, whose unreachable load bits
are zero. Changing reachability can therefore restore a ready load without an
invented extra load operation. These functions prove only graph reachability
and forest structure, never electrical or switching-transient feasibility.

A corridor fixes one preselected target and changes each differing command bit
at most once. It is a restricted operation-order diagnostic, not the complete
restoration state space. Every combination is retained, including nonforests.
The caller must retain unknown/failed physical certificates and derive separate
fixed-operating-point certified subgraphs before using the sequence solver.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import networkx as nx


N_SWITCH = 9
N_DISPATCH = 10
N_COMMAND = N_SWITCH + N_DISPATCH


def _bits(values: Sequence[int]) -> list[int]:
    values = list(values)
    if len(values) != N_COMMAND or any(type(v) is not int or v not in (0, 1) for v in values):
        raise ValueError("NC34 command must contain nine binary switch and ten binary load integers")
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
        raise ValueError("NC34 requires nine distinct controlled bus edges")
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
    forced = [frozenset(e) in {frozenset(f) for f in fault_edges} for e in switches]
    dispatch = list(static["dispatch_loads"])
    if (len(dispatch) != N_DISPATCH or any(not isinstance(n, str) or not n for n in dispatch)
            or len({n.lower() for n in dispatch}) != N_DISPATCH):
        raise ValueError("NC34 requires ten distinct ordered dispatch loads")
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
    """Five normally closed sections, four open ties, ten ready loads, fault=0."""
    context = _context(static, scenario)
    switches = [1] * 5 + [0] * 4
    switches = [int(bit and not forced) for bit, forced in zip(switches, context["forced_open"])]
    return switches + [1] * N_DISPATCH


def state_metadata(static: Mapping, loads: Mapping, scenario: Mapping, command: Sequence[int]) -> dict:
    """Whole-graph forest and nominal source reachability; no physical certificate."""
    context = _context(static, scenario)
    return _metadata(context, _loads(static, loads, context), command)


def _lift(context: Mapping, loads: Mapping, archived_target: Sequence[int]) -> list[int]:
    target = _bits(archived_target)
    for i, forced in enumerate(context["forced_open"]):
        if forced:
            target[i] = 0
    metadata = _metadata(context, loads, target)
    reachable = set(metadata["source_reachable_buses"])
    for i, name in enumerate(context["dispatch"]):
        if loads[name.lower()]["bus"] not in reachable:
            target[N_SWITCH + i] = 1
    return target


def lift_target(static: Mapping, loads: Mapping, scenario: Mapping,
                archived_target: Sequence[int]) -> list[int]:
    """Preserve ready=1 for target-unreachable loads; reachable bits retain target."""
    context = _context(static, scenario)
    return _lift(context, _loads(static, loads, context), archived_target)


def enumerate_corridor(static: Mapping, loads: Mapping, scenario: Mapping,
                       archived_target: Sequence[int], *, initial: Sequence[int] | None = None,
                       max_states: int = 4096) -> dict:
    """Retain every subset of differing bit changes and all monotone forest edges.

    ``mask`` bit j means command coordinate ``diff_indices[j]`` has reached its
    target value. IDs are fixed five-hex-digit mask encodings. The cap rejects a
    whole oversized request; it never truncates or filters unfavorable states.
    ``terminal_service_ids`` are graph-only candidate terminals with service at
    least the lifted target. Electrical certification remains the caller's job.
    """
    if type(max_states) is not int or max_states < 1:
        raise ValueError("max_states must be a positive integer")
    context = _context(static, scenario)
    inventory = _loads(static, loads, context)
    start = initial_command(static, scenario) if initial is None else _bits(initial)
    _metadata(context, inventory, start)  # Reject inconsistent fault commands before enumeration.
    target = _lift(context, inventory, archived_target)
    different = [i for i, (a, b) in enumerate(zip(start, target)) if a != b]
    count = 1 << len(different)
    if count > max_states:
        raise ValueError(f"complete corridor needs {count} states, above explicit cap {max_states}")
    states = {}
    for mask in range(count):
        command = start.copy()
        for j, coordinate in enumerate(different):
            if mask & (1 << j):
                command[coordinate] = target[coordinate]
        state_id = f"m{mask:05x}"
        states[state_id] = {"state_id": state_id, "mask": mask,
                            **_metadata(context, inventory, command)}
    successors = {}
    for state_id, row in states.items():
        successors[state_id] = [f"m{row['mask'] | (1 << j):05x}" for j in range(len(different))
                                if not row["mask"] & (1 << j) and row["forest"]
                                and states[f"m{row['mask'] | (1 << j):05x}"]["forest"]]
    target_id = f"m{count - 1:05x}"
    target_kw = states[target_id]["base_served_kw"]
    terminal_service = [sid for sid, row in states.items() if row["forest"] and row["base_served_kw"] >= target_kw]
    return {"schema_version": 1, "scenario_id": scenario.get("scenario_id"),
            "initial_id": "m00000", "target_id": target_id, "initial_command": start,
            "lifted_target_command": target, "archived_target": _bits(archived_target),
            "diff_indices": different, "states": states, "successors": successors,
            "terminal_service_ids": terminal_service, "target_base_served_kw": target_kw,
            "counts": {"states": count, "forest_states": sum(r["forest"] for r in states.values()),
                       "nonforest_states": sum(not r["forest"] for r in states.values()),
                       "monotone_forest_edges": sum(len(v) for v in successors.values()),
                       "distinct_effective_actions": len({tuple(r["effective_action"]) for r in states.values()})},
            "scope": "restricted_known_target_monotone_command_corridor_not_full_state_space",
            "physical_feasibility_inferred": False, "new_physical_solves": 0,
            "stay_rule": "sequence solver adds stay at every physically certified state",
            "operating_point_rule": "caller must build a separate graph for each fixed alpha and condition"}
