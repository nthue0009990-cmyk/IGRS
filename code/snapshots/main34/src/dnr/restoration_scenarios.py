"""Zero-solve topology support and pre-label fault partitions.

Only the component containing the main-grid source is supplied. Other
components may remain disconnected with their electrical devices deactivated
by the runtime. Forest validity is checked before this deactivation, so an
unsupplied cycle is not silently accepted. These functions establish no voltage,
thermal, phase-connectivity, or AC feasibility result.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import itertools
import json
import re

import networkx as nx


PARTITION_DOMAIN = "restoration-fault-partition-v1"
SPLITS = ("build", "tune", "formal_test")


def _edge(edge):
    if not isinstance(edge, (tuple, list)) or len(edge) != 2:
        raise ValueError("Each bus edge must contain exactly two node names")
    return tuple(edge)


def enumerate_topology_support(
    graph: nx.Graph,
    switch_edges: Sequence,
    fault_edges: Sequence,
    *,
    source_bus: str,
    load_buses: Mapping[str, str],
    max_switches: int = 16,
) -> dict:
    """Enumerate every switch assignment, retaining every whole-graph forest.

    Bit ``i`` controls ``switch_edges[i]`` and has integer weight ``2**i``.
    Faulted switches have a deterministic zero bit. All graph nodes are retained
    even when disconnected; no support row is removed because it supplies fewer
    buses. ``load_reachable`` means main-grid bus reachability only.
    """
    if not isinstance(graph, nx.Graph) or graph.is_directed() or graph.is_multigraph():
        raise ValueError("An undirected simple bus graph is required")
    if not graph or source_bus not in graph:
        raise ValueError("The main-grid source must be a node of the nonempty graph")
    if any(not isinstance(node, str) or not node for node in graph):
        raise ValueError("Bus node names must be nonempty strings")
    if not isinstance(max_switches, int) or isinstance(max_switches, bool) or max_switches < 0:
        raise ValueError("max_switches must be a nonnegative integer")
    switches = [_edge(edge) for edge in switch_edges]
    faults = [_edge(edge) for edge in fault_edges]
    if len(switches) > max_switches:
        raise ValueError("Switch enumeration exceeds the explicit maximum")
    switch_keys, fault_keys = [frozenset(edge) for edge in switches], [frozenset(edge) for edge in faults]
    if len(switch_keys) != len(set(switch_keys)) or len(fault_keys) != len(set(fault_keys)):
        raise ValueError("Duplicate undirected switch or fault edges are not permitted")
    if any(not graph.has_edge(*edge) for edge in switches + faults):
        raise ValueError("Every controlled/faulted edge must exist in the original graph")
    if not isinstance(load_buses, Mapping) or any(bus not in graph for bus in load_buses.values()):
        raise ValueError("Every load must map to a known original bus")
    if any(not isinstance(name, str) or not name for name in load_buses):
        raise ValueError("Load names must be nonempty strings")
    loads = {name: load_buses[name] for name in sorted(load_buses)}
    forced = [key in set(fault_keys) for key in switch_keys]
    total = 2 ** len(switches)
    support_mask = [False] * total
    rows, exclusions = [], Counter()
    faulted = graph.copy()
    faulted.remove_edges_from(faults)
    all_buses = set(graph)
    for action_index in range(total):
        bits = [(action_index >> index) & 1 for index in range(len(switches))]
        if any(bit and forbidden for bit, forbidden in zip(bits, forced)):
            exclusions["faulted_switch_requested_closed"] += 1
            continue
        candidate = faulted.copy()
        candidate.remove_edges_from(edge for bit, edge in zip(bits, switches) if not bit)
        if not nx.is_forest(candidate):
            exclusions["cycle_in_supplied_or_unsupplied_component"] += 1
            continue
        served = set(nx.node_connected_component(candidate, source_bus))
        deenergized = all_buses - served
        components = sorted((sorted(component) for component in nx.connected_components(candidate)), key=tuple)
        support_mask[action_index] = True
        rows.append({
            "action_index": action_index,
            "switch_bits": bits,
            "served_buses": sorted(served),
            "deenergized_buses": sorted(deenergized),
            "components": components,
            "deenergized_components": [component for component in components if source_bus not in component],
            "load_reachable": {name: bus in served for name, bus in loads.items()},
            "retained_edges": sorted([sorted(edge) for edge in candidate.edges]),
        })
    assert len(rows) + sum(exclusions.values()) == total
    return {
        "status": "TOPOLOGICAL_SUPPORT_ONLY",
        "source_bus": source_bus,
        "n_switch": len(switches),
        "switch_index_encoding": "sum(switch_bits[i] * 2**i)",
        "switch_edges": [list(edge) for edge in switches],
        "fault_edges": sorted([sorted(edge) for edge in faults]),
        "switch_forced_open": forced,
        "support_mask": support_mask,
        "rows": rows,
        "valid_switch_count": len(rows),
        "enumerated_switch_assignments": total,
        "exclusion_counts": dict(sorted(exclusions.items())),
        "all_original_buses": sorted(all_buses),
        "load_buses": loads,
        "island_voltage_sources_created": False,
        "physical_feasibility_inferred": False,
        "unreachable_device_rule": "Runtime must deactivate DER/Load/Capacitor/PD in every deenergized component",
    }


def _line_id(value):
    if not isinstance(value, str):
        raise ValueError("Line source IDs must be strings")
    value = value.strip().lower()
    if not re.fullmatch(r"[a-z0-9_.-]+", value):
        raise ValueError("Line IDs must be nonempty and use only letters, digits, _, -, .")
    return value


def _line_key(value):
    # Source L2 precedes L10; tagged pieces remain comparable for other IDs.
    return tuple((0, int(piece)) if piece.isdigit() else (1, piece)
                 for piece in re.split(r"(\d+)", value) if piece), value


def partition_content_sha256(frozen: Mapping) -> str:
    """Hash a complete metadata partition, excluding its own checksum field."""
    payload = {key: value for key, value in frozen.items() if key != "content_sha256"}
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(content.encode("ascii")).hexdigest()


def freeze_fault_partitions(
    line_ids: Sequence[str],
    *,
    partition_seed: int = 20261019,
    n1_counts: tuple[int, int, int] = (20, 6, 6),
    n2_counts: tuple[int, int, int] = (300, 98, 98),
) -> dict:
    """Split every N1 line and N2 pair by a role-separated SHA-256 ranking.

    This function accepts IDs only, never physical labels, solver outcomes, or
    training seeds. N1 is grouped by line identity; N2 by the unordered complete
    fault combination. Lines may occur across partitions inside different N2
    combinations. Consequently this is not an unseen-line protocol.
    """
    if not isinstance(partition_seed, int) or isinstance(partition_seed, bool):
        raise ValueError("partition_seed must be an integer dedicated to partitioning")
    lines = [_line_id(value) for value in line_ids]
    if not lines or len(lines) != len(set(lines)):
        raise ValueError("Nonempty, unique case-insensitive source line IDs are required")
    lines.sort(key=_line_key)
    for order, counts, expected in ((1, n1_counts, len(lines)), (2, n2_counts, len(lines) * (len(lines) - 1) // 2)):
        if len(counts) != 3 or any(not isinstance(count, int) or isinstance(count, bool) or count < 0 for count in counts):
            raise ValueError("Each order requires three nonnegative integer split counts")
        if sum(counts) != expected:
            raise ValueError(f"N{order} split counts must retain every source combination ({expected})")
    scenarios = []
    for order, counts in ((1, n1_counts), (2, n2_counts)):
        family = []
        for faults in itertools.combinations(lines, order):
            scenario_id = f"N{order}__" + "__".join(faults)
            ranking_input = f"{PARTITION_DOMAIN}\nseed_role=partition\npartition_seed={partition_seed}\norder={order}\n" + "|".join(faults)
            rank = hashlib.sha256(ranking_input.encode("ascii")).hexdigest()
            family.append({"scenario_id": scenario_id, "fault_order": order,
                           "fault_lines": list(faults), "partition_hash": rank})
        ranked = sorted(family, key=lambda row: (row["partition_hash"], row["scenario_id"]))
        first, second = counts[0], counts[0] + counts[1]
        for index, row in enumerate(ranked):
            row["partition"] = "build" if index < first else "tune" if index < second else "formal_test"
        # Preserve the source-ID order in exported rows, not the pseudorandom
        # ranking. This also fixes migration-test selection before any labels.
        scenarios.extend(family)
    splits = {split: [row["scenario_id"] for row in scenarios if row["partition"] == split] for split in SPLITS}
    build_lines = set(line for row in scenarios if row["partition"] == "build" for line in row["fault_lines"])
    every_line_seen = build_lines == set(lines)
    result = {
        "schema_version": 1,
        "partition_domain": PARTITION_DOMAIN,
        "partition_seed": partition_seed,
        "seed_role": "partition_only; independent of scenario ordering and model training seeds",
        "algorithm": "SHA256 domain/seed_role/partition_seed/fault_order/canonical_unordered_fault_IDs; rank separately per order",
        "line_ids": lines,
        "counts_by_order": {"N1": dict(zip(SPLITS, n1_counts)), "N2": dict(zip(SPLITS, n2_counts))},
        "scenarios": scenarios,
        "splits": splits,
        "all_source_ids": [row["scenario_id"] for row in scenarios],
        "all_scenarios_retained": True,
        "physical_labels_accessed": False,
        "training_seed_used": False,
        "build_line_coverage": sorted(build_lines, key=_line_key),
        "all_lines_seen_in_build": every_line_seen,
        "generalization_scope": "seen-line new-fault-combination" if every_line_seen else "fault-combination split with incomplete build line coverage",
        "unseen_line_generalization_claimed": False,
        "formal_test_rule": "Metadata-only freeze; no physical labels used to select, normalize, tune, or exclude scenarios",
    }
    result["content_sha256"] = partition_content_sha256(result)
    return result


def select_build_migration_scenarios(
    frozen: Mapping,
    *,
    n1_count: int = 2,
    n2_count: int = 2,
) -> list[dict]:
    """Select only build IDs by original line number, never by physical quality."""
    if partition_content_sha256(frozen) != frozen.get("content_sha256"):
        raise ValueError("Frozen fault-partition checksum does not match")
    selected = []
    for order, count in ((1, n1_count), (2, n2_count)):
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("Migration counts must be nonnegative integers")
        available = [row for row in frozen["scenarios"] if row["partition"] == "build" and row["fault_order"] == order]
        available.sort(key=lambda row: tuple(_line_key(line) for line in row["fault_lines"]))
        if count > len(available):
            raise ValueError(f"Requested more N{order} migration cases than the build partition contains")
        selected.extend({**row, "fault_lines": list(row["fault_lines"])} for row in available[:count])
    return selected
