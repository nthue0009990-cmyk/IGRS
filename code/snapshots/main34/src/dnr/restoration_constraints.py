"""Pure saved-snapshot checks for the explicitly named NC physical extension.

Metadata keys are ``node_phases`` (bus -> phase node numbers), ``source_buses``
(actually enabled voltage-source buses), ``line_config_by_name``, ``line_ratings``
(configuration -> phase_amps and optional neutral_amps), ``transformer_ratings``
(element -> phases and ordered windings containing kva, kv, connection), and
``gfm_limits`` (source element -> p_min_kw/p_max_kw). Element names are case
insensitive. Each raw element must include terminal-major ``node_order``.

Optional explicit model boundaries are ``ideal_line_names`` and
``neutral_thermal_mode='NOT_MODELED'``. Missing ratings are otherwise UNKNOWN.
No OpenDSS engine is imported and no input object is modified.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real

import networkx as nx

from .restoration_isolation import verify_isolation


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _positive(value):
    return _finite(value) and value > 0


def _mapping(value):
    return {str(key).lower(): item for key, item in value.items()} if isinstance(value, Mapping) else {}


def _bus(value):
    return str(value).lower().split('.')[0]


def _section(violations=None, unknown=None, **details):
    violations, unknown = violations or [], unknown or []
    return {"status": "FAIL" if violations else "UNKNOWN" if unknown else "PASS",
            "violations": violations, "unknown": unknown, **details}


def check_known_constraints(
    raw: Mapping,
    metadata: Mapping,
    *,
    voltage_min: float = 0.9,
    voltage_max: float = 1.1,
    balance_abs_tol_kw: float = 1e-4,
    balance_rel_tol: float = 1e-6,
    comparison_abs_tol: float = 1e-8,
    comparison_rel_tol: float = 1e-9,
    deenergized_voltage_tol: float = 1e-8,
    inactive_current_tol_a: float = 1e-8,
) -> dict:
    """Separate numerical evidence and each known physical constraint.

    Power tolerance is ``abs_tol + rel_tol*sum(abs(each P or Q entry))`` for P
    and Q separately. All tolerances are numerical comparison tolerances, not
    changes to physical limits. A nonconverged solution cannot pass physical
    operation checks. Bus-level forest checks do not establish phase topology.

    ``known_constraints_satisfied`` refers only to these named checks. Q/S
    hardware capability, delta internal winding powers, neutral limits explicitly
    excluded by metadata, and ideal switching hardware remain unmodeled.
    """
    settings = (voltage_min, voltage_max, balance_abs_tol_kw, balance_rel_tol,
                comparison_abs_tol, comparison_rel_tol, deenergized_voltage_tol,
                inactive_current_tol_a)
    if not all(_finite(value) for value in settings) or not 0 < voltage_min < voltage_max:
        raise ValueError("Finite numerical settings and 0 < voltage_min < voltage_max are required")
    if min(settings[2:]) < 0:
        raise ValueError("Numerical tolerances must be nonnegative")
    limitations = list(metadata.get("known_limitations", []))
    limitations.extend([
        "GFM reactive and apparent-power hardware capability: NOT_MODELED",
        "Bus-level forest does not certify phase connectivity or switching transients",
        "Global terminal power ledger does not establish per-node KCL or complete AC equations",
    ])
    phase_nodes = {str(bus).lower(): set(phases) for bus, phases in metadata.get("node_phases", {}).items()}
    source_buses = {_bus(bus) for bus in metadata.get("source_buses", [])}
    line_codes = _mapping(metadata.get("line_config_by_name", {}))
    line_ratings = _mapping(metadata.get("line_ratings", {}))
    transformer_ratings = _mapping(metadata.get("transformer_ratings", {}))
    gfm_limits = _mapping(metadata.get("gfm_limits", {}))
    ideal_lines = {str(name).lower() for name in metadata.get("ideal_line_names", [])}
    numerical_unknown, numerical_bad, topo_unknown, topo_bad = [], [], [], []
    voltage_unknown, voltage_bad, line_unknown, line_bad = [], [], [], []
    xfmr_unknown, xfmr_bad, gfm_unknown, gfm_bad = [], [], [], []
    line_checks, xfmr_checks, gfm_checks = [], [], []
    converged = raw.get("converged") is True
    if not converged:
        numerical_unknown.append("Solver did not establish convergence")

    graph = nx.Graph()
    nodes = raw.get("graph_nodes", metadata.get("graph_nodes"))
    edges = raw.get("graph_edges", metadata.get("graph_edges"))
    if nodes is None or edges is None:
        topo_unknown.append("Graph nodes/edges unavailable")
    else:
        graph.add_nodes_from(_bus(node) for node in nodes)
        for edge in edges:
            if not isinstance(edge, (list, tuple)) or len(edge) != 2 or any(_bus(node) not in graph for node in edge):
                topo_unknown.append(f"Malformed or unknown graph edge: {edge}")
                continue
            graph.add_edge(*(_bus(node) for node in edge))
        if not graph.nodes:
            topo_unknown.append("Empty bus graph")
        elif not nx.is_forest(graph):
            topo_bad.append({"type": "nonforest_bus_graph"})
    if not source_buses:
        topo_unknown.append("Actually enabled source buses unavailable")
    if source_buses - set(graph):
        topo_unknown.append("Source bus missing from graph: " + str(sorted(source_buses - set(graph))))
    components = list(nx.connected_components(graph))
    component_of = {bus: component for component in components for bus in component}
    powered = {bus: bool(component & source_buses) for bus, component in component_of.items()}
    if raw.get("components") is not None:
        saved = {frozenset(_bus(bus) for bus in component) for component in raw["components"]}
        actual = {frozenset(component) for component in components}
        if saved != actual:
            topo_unknown.append("Saved graph components disagree with graph edges")

    isolation = None
    excluded_node_states = {}
    if "physical_phase_nodes_by_bus" in metadata:
        isolation = verify_isolation(raw, metadata)
        if isolation["status"] == "FAIL":
            topo_bad.append({"type": "isolation_scope_not_certified", "violations": isolation["violations"]})
        elif isolation["status"] != "PASS":
            topo_unknown.append("Isolation scope is numerically/structurally unknown")
        else:
            excluded_node_states = isolation["excluded_node_states"]

    node_names, voltages = raw.get("node_names"), raw.get("node_vmag_pu")
    voltage_map = {}
    if node_names is None or voltages is None or len(node_names) != len(voltages):
        numerical_unknown.append("Node voltage inventory/length unavailable")
        voltage_unknown.append("Node voltage inventory/length unavailable")
    else:
        for name, value in zip(node_names, voltages):
            key = str(name).lower()
            if key in voltage_map:
                numerical_unknown.append("Duplicate node voltage: " + key)
            voltage_map[key] = value
            if not _finite(value):
                numerical_unknown.append("Nonfinite node voltage: " + key)
        if not phase_nodes:
            voltage_unknown.append("Expected bus phase-node inventory unavailable")
        for bus, phases in phase_nodes.items():
            for phase in phases:
                if phase == 0:
                    continue
                name = f"{bus}.{phase}"
                if (name not in voltage_map and
                        excluded_node_states.get(name) == "STRUCTURALLY_EXCLUDED_NOT_SOLVED"):
                    continue
                value = voltage_map.get(name)
                if not _finite(value):
                    voltage_unknown.append("Missing/nonfinite phase voltage: " + name)
                    continue
                if value < 0:
                    numerical_bad.append({"type": "negative_voltage_magnitude", "node": name, "value": value})
                if bus not in powered:
                    voltage_unknown.append("Phase bus missing from graph: " + bus)
                elif not powered[bus]:
                    if abs(value) > deenergized_voltage_tol:
                        voltage_bad.append({"type": "unsourced_nonzero_voltage", "node": name, "value": value})
                elif value < voltage_min - comparison_abs_tol or value > voltage_max + comparison_abs_tol:
                    voltage_bad.append({"type": "energized_voltage", "node": name, "value": value,
                                        "lower": voltage_min, "upper": voltage_max})
    complex_volts = raw.get("node_voltages_real_imag")
    if complex_volts is not None and (node_names is None or len(complex_volts) != 2 * len(node_names)
                                    or not all(_finite(value) for value in complex_volts)):
        numerical_unknown.append("Missing/nonfinite complex node voltage entries")

    elements = list(raw.get("elements", []))
    names = [str(element.get("name", "")).lower() for element in elements]
    if not elements or len(names) != len(set(names)) or "" in names:
        numerical_unknown.append("Empty, duplicate, or unnamed electrical element inventory")
    expected = metadata.get("expected_element_names")
    if expected is not None:
        absent = {str(name).lower() for name in expected} - set(names)
        if absent:
            numerical_unknown.append("Missing electrical elements: " + str(sorted(absent)))
    else:
        limitations.append("Power ledger inventory completeness relies on caller supplying every enabled electrical element")
    p_entries, q_entries, ledger_complete = [], [], True
    unsourced_load_p = 0.0

    def above(value, limit):
        return value > limit + comparison_abs_tol + comparison_rel_tol * abs(limit)

    def conductor_role(bus, node):
        if node == 0:
            return "neutral"
        if bus not in phase_nodes:
            return "unknown"
        if node in phase_nodes[bus]:
            return "phase"
        if isolation is None:
            return "neutral"
        neutral_nodes = metadata.get("neutral_nodes_by_bus", {})
        neutral_nodes = _mapping(neutral_nodes)
        registered = neutral_nodes.get(bus, [])
        return "neutral" if isinstance(registered, (list, tuple, set)) and node in registered else "unknown"

    for element in elements:
        name = str(element.get("name", "")).lower()
        kind = name.split('.')[0]
        if element.get("enabled") is False:
            continue
        if element.get("enabled") is not True:
            numerical_unknown.append("Element enabled state unavailable: " + name)
        ncond = element.get("conductors", element.get("num_conductors"))
        nterm = element.get("terminals", element.get("num_terminals"))
        nphase = element.get("phases", element.get("num_phases"))
        dimensions_valid = all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in (ncond, nterm, nphase))
        dimensions_valid = dimensions_valid and nphase <= ncond
        powers, currents, order = element.get("powers_kw_kvar", []), element.get("currents_mag_angle", []), element.get("node_order", [])
        buses = element.get("buses", [])
        power_valid = dimensions_valid and len(powers) == 2 * ncond * nterm and all(_finite(value) for value in powers)
        current_valid = dimensions_valid and len(currents) == 2 * ncond * nterm and all(_finite(value) for value in currents)
        layout_valid = (dimensions_valid and len(order) == ncond * nterm and len(buses) == nterm
                        and all(isinstance(node, int) and not isinstance(node, bool) and node >= 0 for node in order))
        if not power_valid:
            ledger_complete = False
            numerical_unknown.append("Missing/nonfinite/malformed terminal powers: " + name)
        else:
            p_entries.extend(powers[::2])
            q_entries.extend(powers[1::2])
        if not current_valid:
            numerical_unknown.append("Missing/nonfinite/malformed terminal currents: " + name)
        elif any(value < 0 for value in currents[::2]):
            numerical_bad.append({"type": "negative_current_magnitude", "element": name})
        if not dimensions_valid:
            numerical_unknown.append("Element terminal dimensions unavailable: " + name)
        if kind == "load":
            bus = _bus(buses[0]) if buses else ""
            if bus not in powered:
                topo_unknown.append("Load bus missing from graph: " + name)
            elif not powered[bus]:
                if not converged:
                    topo_unknown.append("Unsourced load iterate does not establish actual supply: " + name)
                elif not power_valid:
                    topo_unknown.append("Unsourced load power is numerically unknown: " + name)
                else:
                    actual_p = math.fsum(powers[::2])
                    actual_q = math.fsum(powers[1::2])
                    unsourced_load_p += actual_p
                    if abs(actual_p) > balance_abs_tol_kw or abs(actual_q) > balance_abs_tol_kw:
                        topo_bad.append({"type": "unsourced_load_not_zero", "element": name,
                                         "p_kw": actual_p, "q_kvar": actual_q})
        if kind == "line":
            if name in ideal_lines:
                limitations.append(f"Ideal switching element {name} hardware ampacity: NOT_MODELED")
                continue
            config = str(line_codes.get(name, "")).lower()
            rating = line_ratings.get(config, line_ratings.get(name))
            if isinstance(rating, Real):
                rating = {"phase_amps": rating}
            if not isinstance(rating, Mapping) or not _positive(rating.get("phase_amps")):
                line_unknown.append("Missing positive phase-conductor rating: " + name)
                continue
            if not current_valid or not layout_valid:
                line_unknown.append("Terminal current/node-order evidence unavailable: " + name)
                continue
            for terminal in range(nterm):
                bus = _bus(buses[terminal])
                for conductor in range(ncond):
                    index = terminal * ncond + conductor
                    role = conductor_role(bus, order[index])
                    value = currents[2 * index]
                    limit = rating.get("phase_amps") if role == "phase" else rating.get("neutral_amps")
                    if role == "unknown":
                        line_unknown.append(f"Unknown phase/neutral role: {name} terminal {terminal + 1}")
                        continue
                    if not _positive(limit):
                        if role == "neutral" and value <= inactive_current_tol_a:
                            continue
                        if role == "neutral" and metadata.get("neutral_thermal_mode") == "NOT_MODELED":
                            limitations.append(f"{name} active neutral-conductor ampacity: NOT_MODELED")
                        else:
                            line_unknown.append(f"Missing {role} ampacity: {name} terminal {terminal + 1}")
                        continue
                    check = {"element": name, "terminal": terminal + 1, "conductor": conductor + 1,
                             "node": order[index], "role": role, "current_a": value, "limit_a": limit}
                    line_checks.append(check)
                    if above(value, limit):
                        line_bad.append(check)
        if kind == "transformer":
            rating = transformer_ratings.get(name)
            windings = rating.get("windings", []) if isinstance(rating, Mapping) else []
            if not dimensions_valid or len(windings) != nterm or not power_valid or not current_valid or not layout_valid:
                xfmr_unknown.append("Winding ratings/terminal evidence unavailable: " + name)
                continue
            for terminal, winding in enumerate(windings):
                kva, kv = winding.get("kva"), winding.get("kv")
                phases = winding.get("phases", rating.get("phases", nphase))
                connection = str(winding.get("connection", "")).lower()
                if not _positive(kva) or not _positive(kv) or phases not in (1, 3) or connection not in ("wye", "delta"):
                    xfmr_unknown.append(f"Incomplete nominal winding specification: {name} winding {terminal + 1}")
                    continue
                offset = 2 * terminal * ncond
                vector = powers[offset:offset + 2 * ncond]
                total_s = math.hypot(math.fsum(vector[::2]), math.fsum(vector[1::2]))
                current_limit = kva / (math.sqrt(3) * kv) if phases == 3 else kva / kv
                check = {"element": name, "winding": terminal + 1, "total_s_kva": total_s,
                         "limit_kva": kva, "line_current_limit_a": current_limit,
                         "connection": connection, "phase_s_check": "WYE_TERMINAL_PHASE" if connection == "wye" else "NOT_MODELED_DELTA_INTERNAL"}
                xfmr_checks.append(check)
                if above(total_s, kva):
                    xfmr_bad.append({**check, "type": "winding_total_s"})
                if connection == "delta":
                    limitations.append(f"{name} winding {terminal + 1} delta internal phase S: NOT_MODELED; total S and line-conductor I checked")
                bus = _bus(buses[terminal])
                for conductor in range(ncond):
                    index = terminal * ncond + conductor
                    value = currents[2 * index]
                    role = conductor_role(bus, order[index])
                    detail = {"element": name, "winding": terminal + 1, "conductor": conductor + 1,
                              "node": order[index], "role": role, "current_a": value}
                    if role == "phase":
                        if above(value, current_limit):
                            xfmr_bad.append({**detail, "type": "winding_line_current", "limit_a": current_limit})
                        if connection == "wye":
                            phase_s = math.hypot(powers[2 * index], powers[2 * index + 1])
                            if above(phase_s, kva / phases):
                                xfmr_bad.append({**detail, "type": "wye_phase_s", "phase_s_kva": phase_s,
                                                 "limit_kva": kva / phases})
                    elif role == "neutral":
                        neutral_limit = winding.get("neutral_amps")
                        if _positive(neutral_limit):
                            if above(value, neutral_limit):
                                xfmr_bad.append({**detail, "type": "neutral_current", "limit_a": neutral_limit})
                        elif value > inactive_current_tol_a:
                            if metadata.get("neutral_thermal_mode") == "NOT_MODELED":
                                limitations.append(f"{name} winding {terminal + 1} neutral ampacity: NOT_MODELED")
                            else:
                                xfmr_unknown.append(f"Missing neutral ampacity: {name} winding {terminal + 1}")
                    else:
                        xfmr_unknown.append(f"Unknown conductor role: {name} winding {terminal + 1}")
        if name in gfm_limits or name.startswith("vsource.bus_"):
            limit = gfm_limits.get(name)
            if not isinstance(limit, Mapping) or not _finite(limit.get("p_min_kw")) or not _positive(limit.get("p_max_kw")):
                gfm_unknown.append("Missing GFM active-power boundary: " + name)
            elif limit["p_min_kw"] < 0 or limit["p_min_kw"] > limit["p_max_kw"]:
                gfm_unknown.append("Invalid supply-only GFM boundary: " + name)
            elif not power_valid:
                gfm_unknown.append("GFM terminal active power unavailable: " + name)
            else:
                supply = -math.fsum(powers[::2])
                check = {"element": name, "supply_p_kw": supply, "min_kw": limit["p_min_kw"], "max_kw": limit["p_max_kw"]}
                gfm_checks.append(check)
                if above(supply, limit["p_max_kw"]) or supply < limit["p_min_kw"] - comparison_abs_tol - comparison_rel_tol * abs(limit["p_min_kw"]):
                    gfm_bad.append(check)

    ledger = {"status": "UNKNOWN", "residual_p_kw": None, "residual_q_kvar": None,
              "tolerance_p_kw": None, "tolerance_q_kvar": None}
    if ledger_complete and elements:
        residual_p, residual_q = math.fsum(p_entries), math.fsum(q_entries)
        tol_p = balance_abs_tol_kw + balance_rel_tol * math.fsum(abs(value) for value in p_entries)
        tol_q = balance_abs_tol_kw + balance_rel_tol * math.fsum(abs(value) for value in q_entries)
        passed = abs(residual_p) <= tol_p and abs(residual_q) <= tol_q
        ledger.update(status="PASS" if passed else "FAIL", residual_p_kw=residual_p, residual_q_kvar=residual_q,
                      tolerance_p_kw=tol_p, tolerance_q_kvar=tol_q)
        if not passed:
            numerical_bad.append({"type": "terminal_power_ledger", **ledger})
    else:
        numerical_unknown.append("Complete finite enabled-element power ledger unavailable")
    if not converged:
        # Iterates can be useful diagnostics, but cannot establish an operating
        # constraint violation or satisfaction for a converged electrical state.
        for unknown in (voltage_unknown, line_unknown, xfmr_unknown, gfm_unknown):
            unknown.append("Operating solution not numerically converged")

    def physical(violations, unknown, **details):
        result = _section(violations, unknown, **details)
        if not converged:
            result["observed_iterate_violations"] = result["violations"]
            result["violations"] = []
            result["status"] = "UNKNOWN"
        return result

    result = {
        "numerical": _section(numerical_bad, numerical_unknown, converged=converged, power_ledger=ledger),
        "topology": _section(topo_bad, topo_unknown, components=[sorted(component) for component in components],
                             unsourced_components=[sorted(component) for component in components if not component & source_buses],
                             unsourced_load_p_kw=unsourced_load_p),
        "voltage": physical(voltage_bad, voltage_unknown, limits_pu=[voltage_min, voltage_max]),
        "line_thermal": physical(line_bad, line_unknown, checks=line_checks),
        "transformer_thermal": physical(xfmr_bad, xfmr_unknown, checks=xfmr_checks),
        "gfm_active_limit": physical(gfm_bad, gfm_unknown, checks=gfm_checks),
        "gfm_reactive_limit": {"status": "NOT_MODELED"},
        "gfm_apparent_limit": {"status": "NOT_MODELED"},
        "unknown_limitations": sorted(set(str(item) for item in limitations)),
        "full_hardware_feasibility": "NOT_ESTABLISHED",
        "numerical_tolerances": {"balance_abs_tol_kw": balance_abs_tol_kw, "balance_rel_tol": balance_rel_tol,
                                 "comparison_abs_tol": comparison_abs_tol, "comparison_rel_tol": comparison_rel_tol,
                                 "deenergized_voltage_tol": deenergized_voltage_tol, "inactive_current_tol_a": inactive_current_tol_a},
    }
    if not converged:
        result["numerical"]["status"] = "UNKNOWN"
    if isolation is not None:
        result["isolation"] = isolation
        result["voltage"]["excluded_node_states"] = excluded_node_states
    required = ("numerical", "topology", "voltage", "line_thermal", "transformer_thermal", "gfm_active_limit")
    statuses = [result[key]["status"] for key in required]
    result["known_constraints_satisfied"] = False if "FAIL" in statuses else None if "UNKNOWN" in statuses else True
    return result
