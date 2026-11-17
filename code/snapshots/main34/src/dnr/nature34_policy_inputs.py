"""Pure NC34 fault observations and shared topology-support feature tables."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json

import networkx as nx
import numpy as np

from .restoration_scenarios import enumerate_topology_support


def build_policy_inputs(
    graph: nx.Graph,
    switch_edges: Sequence,
    fault_edges: Sequence,
    *,
    source_bus: str,
    dispatch_loads: Sequence[str],
    load_buses: Mapping[str, str],
    line_ids: Sequence[str],
    fault_lines: Sequence[str],
    line_edges: Mapping | None = None,
) -> dict:
    """Return32 fault bits and512-by32 topology features without physical labels.

    For each of the10 author-ordered loads, features are main-source reachable,
    its component size/37, and source-path hop count/36 (0 if unreachable).
    Forest=1 and number of components/37 complete the32 channels. Invalid switch
    rows contain finite zeros and are excluded by the shared support mask.
    """
    ids = [str(value).lower() for value in line_ids]
    faults = [str(value).lower() for value in fault_lines]
    dispatch = list(dispatch_loads)
    if len(graph) != 37 or len(switch_edges) != 9 or len(dispatch) != 10 or len(ids) != 32:
        raise ValueError("NC34 input dimensions must be37 buses/9 switches/10 loads/32 original lines")
    if len(ids) != len(set(ids)) or len(faults) != len(set(faults)) or not set(faults) <= set(ids):
        raise ValueError("Fault and original-line identities must be unique and known")
    if len({name.lower() for name in dispatch}) != 10:
        raise ValueError("Dispatch load identity/order must be unique")
    buses = {str(name).lower(): bus for name, bus in load_buses.items()}
    if any(name.lower() not in buses for name in dispatch):
        raise ValueError("Dispatch load bus mapping is incomplete")
    ordered_loads = {name: buses[name.lower()] for name in dispatch}
    if line_edges is not None:
        source_edges = {str(name).lower(): frozenset(edge) for name, edge in line_edges.items()}
        if not set(ids) <= set(source_edges) or {source_edges[name] for name in faults} != {frozenset(edge) for edge in fault_edges}:
            raise ValueError("Fault geometry disagrees with original-line identities")
    elif len(faults) != len(fault_edges):
        raise ValueError("Fault ID and edge counts differ")
    support = enumerate_topology_support(graph, switch_edges, fault_edges, source_bus=source_bus, load_buses=ordered_loads)
    observation = np.asarray([int(name in faults) for name in ids], dtype=np.float32)
    context = np.zeros((512, 32), dtype=np.float32)
    load_off = np.zeros((512, 10), dtype=bool)
    for row in support["rows"]:
        candidate = nx.Graph()
        candidate.add_nodes_from(graph.nodes)
        candidate.add_edges_from(row["retained_edges"])
        distances = nx.single_source_shortest_path_length(candidate, source_bus)
        component_size = {bus: len(component) for component in row["components"] for bus in component}
        values = []
        for index, name in enumerate(dispatch):
            bus = ordered_loads[name]
            reachable = row["load_reachable"][name]
            values.extend((float(reachable), component_size[bus] / 37., distances.get(bus, 0) / 36.))
            load_off[row["action_index"], index] = not reachable
        values.extend((1., len(row["components"]) / 37.))
        context[row["action_index"]] = values
    arrays = {"observation": observation, "switch_forced_open": np.asarray(support["switch_forced_open"], dtype=bool),
              "context_table": context, "switch_support_mask": np.asarray(support["support_mask"], dtype=bool),
              "load_forced_off_table": load_off}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Pure topology input contains nonfinite features")
    details = {"source_bus": source_bus, "line_ids": ids, "fault_lines": [name for name in ids if name in faults],
               "dispatch_loads": dispatch, "dispatch_load_buses": ordered_loads,
               "switch_edges": [list(edge) for edge in switch_edges],
               "valid_switch_count": support["valid_switch_count"], "physical_labels_used": False,
               "observation_definition": "Original32-line binary fault indicator in frozen line_ids order",
               "context_definition": "For each author-ordered load: reachable,component_size/37,source_hops/36; then forest1,components/37",
               "invalid_rows": "Finite zero features and zero load masks; switch support false; never sampled",
               "island_sources_created": False}
    array_hashes = {name: hashlib.sha256(value.tobytes(order="C")).hexdigest() for name, value in arrays.items()}
    content = {"metadata": details, "arrays": {name: {"sha256": array_hashes[name], "shape": list(value.shape), "dtype": str(value.dtype)}
                                               for name, value in arrays.items()}}
    digest = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {**arrays, "metadata": details, "array_hashes": array_hashes, "input_sha256": digest}
