"""Pure isolation-scope certificate for the named whole-element model.

This is structural/numerical evidence for excluded components, not proof of
field safety. It does not call DSS or replace AC and equipment-limit checks.
Disabled NodeOrder/V/P caches are not operating-point evidence. Inventory,
frozen BusNames and explicitly saved zero finite currents remain mandatory.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Integral, Real

import networkx as nx


def _seq(value):
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _finite(value):
    if not isinstance(value, Real) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _bus(value):
    return value.split('.')[0].strip().lower() if isinstance(value, str) else ""


def verify_isolation(raw: Mapping, metadata: Mapping) -> dict:
    """Independently verify a planned whole-element isolation snapshot.

    Required metadata: physical_phase_nodes_by_bus, neutral_nodes_by_bus,
    element_layout, planned_enabled, deenergized_buses, source_buses and
    expected_element_names. Enabled PDs must have saved conductor_open masks
    showing every contact closed. Physical/dead node inventories are never
    filled with synthetic voltage observations.
    """
    unknowns, violations = [], []
    observed_states, disabled_checks = {}, []
    if not isinstance(raw, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError("raw and metadata must be mappings")
    current_tol = metadata.get("isolation_current_tolerance_a", 1e-8)
    voltage_tol = metadata.get("isolation_voltage_tolerance_pu", 1e-8)
    if not all(_finite(value) and value >= 0 for value in (current_tol, voltage_tol)):
        raise ValueError("Isolation numerical tolerances must be finite and nonnegative")

    def bad(kind, **details):
        violations.append({"type": kind, **details})

    def names(value, label, *, buses=False, nonempty=False):
        if not _seq(value) or (nonempty and not value):
            unknowns.append("Missing/malformed " + label)
            return set()
        output = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                unknowns.append("Invalid entry in " + label)
                continue
            output.append(_bus(item) if buses else item.strip().lower())
        if len(set(output)) != len(output):
            unknowns.append("Duplicate entries in " + label)
        return set(output)

    def mapping(value, label):
        if not isinstance(value, Mapping):
            unknowns.append("Missing/malformed " + label)
            return {}
        result = {str(key).strip().lower(): item for key, item in value.items()}
        if len(result) != len(value):
            unknowns.append("Duplicate normalized keys in " + label)
        return result

    def node_inventory(value, label, phase=False):
        output = {}
        for bus, nodes in mapping(value, label).items():
            if not bus or _bus(bus) != bus or not _seq(nodes) or not all(
                isinstance(node, Integral) and not isinstance(node, bool) and node >= (1 if phase else 0)
                for node in nodes
            ):
                unknowns.append("Malformed node inventory: " + label + ":" + bus)
                continue
            if len(set(nodes)) != len(nodes):
                unknowns.append("Duplicate nodes: " + label + ":" + bus)
            output[bus] = {int(node) for node in nodes}
        return output

    phases = node_inventory(metadata.get("physical_phase_nodes_by_bus"), "physical_phase_nodes_by_bus", True)
    neutrals = node_inventory(metadata.get("neutral_nodes_by_bus"), "neutral_nodes_by_bus")
    if not phases:
        unknowns.append("Empty physical phase inventory")
    for bus, nodes in neutrals.items():
        if bus not in phases or nodes & phases.get(bus, set()):
            bad("ambiguous_neutral_inventory", bus=bus)
    if "node_phases" in metadata:
        inherited = node_inventory(metadata["node_phases"], "node_phases", True)
        if inherited != phases:
            bad("physical_phase_inventory_disagrees_with_node_phases")
    graph_nodes = names(raw.get("graph_nodes"), "graph_nodes", buses=True, nonempty=True)
    if graph_nodes != set(phases):
        bad("graph_physical_bus_inventory_mismatch", missing=sorted(set(phases) - graph_nodes),
            unexpected=sorted(graph_nodes - set(phases)))
    graph = nx.Graph()
    graph.add_nodes_from(graph_nodes)
    edges = raw.get("graph_edges")
    if not _seq(edges):
        unknowns.append("Missing/malformed graph_edges")
        edges = []
    seen_edges = set()
    for edge in edges:
        if not _seq(edge) or len(edge) != 2 or any(not _bus(bus) or _bus(bus) not in graph_nodes for bus in edge):
            unknowns.append("Malformed graph edge")
            continue
        pair = tuple(sorted(_bus(bus) for bus in edge))
        if pair[0] == pair[1] or pair in seen_edges:
            bad("duplicate_or_self_graph_edge", edge=list(pair))
        seen_edges.add(pair)
        graph.add_edge(*pair)
    if graph and not nx.is_forest(graph):
        bad("proposed_graph_is_not_a_forest")
    sources = names(metadata.get("source_buses"), "source_buses", buses=True, nonempty=True)
    if not sources <= graph_nodes:
        bad("source_bus_missing_from_graph", buses=sorted(sources - graph_nodes))
    powered = set().union(*(set(nx.node_connected_component(graph, bus)) for bus in sources if bus in graph))
    computed_dead = graph_nodes - powered
    declared_dead = names(metadata.get("deenergized_buses"), "deenergized_buses", buses=True)
    if declared_dead != computed_dead:
        bad("deenergized_component_mismatch", computed=sorted(computed_dead), declared=sorted(declared_dead))
    if raw.get("components") is not None:
        value = raw["components"]
        if not _seq(value) or any(not _seq(component) for component in value):
            unknowns.append("Malformed saved graph components")
        elif {frozenset(_bus(bus) for bus in component) for component in value} != {
            frozenset(component) for component in nx.connected_components(graph)
        }:
            bad("saved_components_disagree_with_graph")

    expected = names(metadata.get("expected_element_names"), "expected_element_names", nonempty=True)
    layouts = mapping(metadata.get("element_layout"), "element_layout")
    planned = mapping(metadata.get("planned_enabled"), "planned_enabled")
    if set(layouts) != expected or set(planned) != expected:
        unknowns.append("Frozen layouts/planned-enabled inventory differs from expected physical inventory")
    excluded_controls = names(metadata.get("excluded_controls", []), "excluded_controls")
    if expected & excluded_controls:
        bad("physical_inventory_overlaps_excluded_controls")
    elements = raw.get("elements")
    if not _seq(elements):
        unknowns.append("Missing/malformed physical elements")
        elements = []
    actual = {}
    for item in elements:
        if not isinstance(item, Mapping) or not isinstance(item.get("name"), str) or not item["name"].strip():
            unknowns.append("Malformed/unnamed physical element")
            continue
        name = item["name"].strip().lower()
        if name in actual:
            unknowns.append("Duplicate physical element: " + name)
        actual[name] = item
    if set(actual) != expected:
        unknowns.append("Actual physical inventory differs from expected: missing=" + str(sorted(expected - set(actual)))
                        + "; unexpected=" + str(sorted(set(actual) - expected)))
    allowed_sources = names(metadata.get("allowed_source_elements", ["Vsource.source"]),
                            "allowed_source_elements", nonempty=True)
    enabled_sources, actual_source_buses = set(), set()
    enabled_edges = set()
    series_pd = {"line", "transformer", "reactor"}
    for name, item in actual.items():
        layout = layouts.get(name)
        if not isinstance(layout, Mapping):
            unknowns.append("Missing frozen element layout: " + name)
            continue
        ncond, nterm = layout.get("conductors"), layout.get("terminals")
        buses, order = layout.get("buses"), layout.get("node_order")
        kind = str(layout.get("kind", "")).lower()
        dimensions_valid = all(isinstance(value, Integral) and not isinstance(value, bool) and value > 0
                               for value in (ncond, nterm))
        layout_valid = (dimensions_valid and _seq(buses) and len(buses) == nterm and
                        all(_bus(bus) in phases for bus in buses) and _seq(order) and len(order) == ncond * nterm and
                        all(isinstance(node, Integral) and not isinstance(node, bool) and node >= 0 for node in order))
        if not layout_valid or kind != name.split('.')[0]:
            unknowns.append("Malformed frozen element layout/kind: " + name)
            continue
        expected_buses = [str(bus).strip().lower() for bus in buses]
        saved_buses = item.get("buses")
        if not _seq(saved_buses) or [str(bus).strip().lower() for bus in saved_buses] != expected_buses:
            bad("terminal_bus_layout_mismatch", element=name)
        enabled = item.get("enabled")
        if not isinstance(enabled, bool) or not isinstance(planned.get(name), bool):
            unknowns.append("Missing/nonboolean actual/planned enabled: " + name)
            continue
        if enabled != planned[name]:
            bad("actual_enabled_disagrees_with_plan", element=name, actual=enabled, planned=planned[name])
        if not enabled:
            currents = item.get("currents_real_imag")
            if not _seq(currents) or len(currents) != 2 * ncond * nterm or not all(_finite(value) for value in currents):
                unknowns.append("Missing/nonfinite disabled currents: " + name)
            else:
                largest = max(math.hypot(currents[i], currents[i + 1]) for i in range(0, len(currents), 2))
                if not math.isfinite(largest):
                    unknowns.append("Nonfinite derived disabled current: " + name)
                else:
                    disabled_checks.append({"element": name, "max_current_a": largest, "tolerance_a": current_tol})
                    if largest > current_tol:
                        bad("disabled_nonzero_current", element=name, current_a=largest)
            continue
        if (item.get("conductors") != ncond or item.get("terminals") != nterm or
                not _seq(item.get("node_order")) or list(item["node_order"]) != list(order) or
                any(not isinstance(node, Integral) or isinstance(node, bool) for node in item["node_order"])):
            bad("enabled_terminal_node_layout_mismatch", element=name)
        terminal_buses = []
        for terminal in range(nterm):
            bus = _bus(buses[terminal])
            nodes = order[terminal * ncond:(terminal + 1) * ncond]
            if any(node > 0 for node in nodes):
                terminal_buses.append(bus)
                if bus in computed_dead:
                    bad("enabled_terminal_touches_deenergized_bus", element=name, terminal=terminal + 1, bus=bus)
            for node in nodes:
                if node > 0 and node not in phases[bus] and node not in neutrals.get(bus, set()):
                    bad("undeclared_positive_terminal_node", element=name, node=f"{bus}.{node}")
        if kind in series_pd:
            opened = item.get("conductor_open")
            if (not _seq(opened) or len(opened) != nterm or
                    any(not _seq(row) or len(row) != ncond or any(not isinstance(value, bool) for value in row) for row in opened)):
                unknowns.append("Missing/malformed enabled PD conductor-open mask: " + name)
            elif any(value for row in opened for value in row):
                bad("enabled_PD_has_open_contact_in_whole_element_model", element=name)
            distinct = list(dict.fromkeys(terminal_buses))
            for other in distinct[1:]:
                enabled_edges.add(tuple(sorted((distinct[0], other))))
        if kind in {"vsource", "isource"}:
            enabled_sources.add(name)
            actual_source_buses.update(terminal_buses)
            if kind != "vsource" or name not in allowed_sources:
                bad("unexpected_enabled_source", element=name)
    if enabled_sources != allowed_sources or actual_source_buses != sources:
        bad("enabled_source_inventory_mismatch", actual=sorted(enabled_sources), expected=sorted(allowed_sources),
            actual_buses=sorted(actual_source_buses), expected_buses=sorted(sources))
    main_edges = {tuple(sorted(edge)) for edge in graph.edges if edge[0] in powered and edge[1] in powered}
    if main_edges != enabled_edges:
        bad("main_graph_enabled_PD_edge_mismatch", missing=sorted(main_edges - enabled_edges),
            unexpected=sorted(enabled_edges - main_edges))

    node_names, node_voltages = raw.get("node_names"), raw.get("node_vmag_pu")
    voltage_map = {}
    if not _seq(node_names) or not _seq(node_voltages) or len(node_names) != len(node_voltages):
        unknowns.append("Missing/malformed solved node voltage inventory")
    else:
        for name, value in zip(node_names, node_voltages):
            if not isinstance(name, str) or '.' not in name:
                unknowns.append("Malformed solved node name")
                continue
            normalized = name.strip().lower()
            bus, node_string = normalized.rsplit('.', 1)
            try:
                node = int(node_string)
            except ValueError:
                unknowns.append("Malformed solved node number: " + normalized)
                continue
            canonical = f"{bus}.{node}"
            if canonical in voltage_map:
                unknowns.append("Duplicate solved node: " + canonical)
            voltage_map[canonical] = value
            if bus not in phases or node < 0 or (node > 0 and node not in phases.get(bus, set()) and node not in neutrals.get(bus, set())):
                bad("undeclared_solved_node", node=normalized)
            if not _finite(value):
                unknowns.append("Nonfinite solved node voltage: " + normalized)
            elif value < 0:
                bad("negative_voltage_magnitude", node=normalized)
            elif node == 0 and value > voltage_tol:
                bad("ground_reference_nonzero_voltage", node=canonical, value_pu=value)
    for bus in sorted(computed_dead):
        for node in sorted((phases.get(bus, set()) | neutrals.get(bus, set())) - {0}):
            name = f"{bus}.{node}"
            if name not in voltage_map:
                observed_states[name] = "STRUCTURALLY_EXCLUDED_NOT_SOLVED"
            elif not _finite(voltage_map[name]):
                observed_states[name] = "UNKNOWN_NONFINITE"
            elif abs(voltage_map[name]) <= voltage_tol:
                observed_states[name] = "OBSERVED_FINITE_ZERO"
            else:
                observed_states[name] = "UNSOURCED_NONZERO_VOLTAGE"
                bad("deenergized_node_nonzero_voltage", node=name, value_pu=voltage_map[name])
    if raw.get("converged") is not True:
        unknowns.append("Solver convergence not established")
    status = "UNKNOWN" if unknowns else "FAIL" if violations else "PASS"
    return {"status": status,
            "certified_deenergized_buses": sorted(computed_dead) if status == "PASS" else [],
            "excluded_node_states": observed_states if status == "PASS" else {},
            "observed_node_states": observed_states,
            "computed_deenergized_buses": sorted(computed_dead), "powered_buses": sorted(powered),
            "unknowns": unknowns, "violations": violations, "disabled_current_checks": disabled_checks,
            "numerical_tolerances": {"current_tolerance_a": current_tol, "voltage_tolerance_pu": voltage_tol},
            "limitations": [
                "Certificate applies to the named whole-element isolation model; open-stub charging is removed",
                "Excluded absent nodes are not synthetic measured zero voltages",
                "Disabled NodeOrder, voltage, power and Yprim caches are not operating-point evidence",
                "Enabled-subnetwork AC equations and equipment constraints require separate checks",
                "This certificate does not establish field deenergization safety or switching transients",
            ]}
