"""Independent terminal-array consistency checks; never imports or calls DSS.

``raw`` contains ``converged`` and ``elements``. Complex arrays are interleaved
real/imaginary values, in terminal-major conductor order. Yprim is column-major.
``metadata.expected_element_names`` must inventory every physical element,
including disabled ones; ``excluded_controls`` documents nonphysical controls
separately. Passing these checks establishes numerical snapshot consistency,
not equipment feasibility or nonlinear PC-device constitutive equations.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Integral, Real


def _finite(value):
    if not isinstance(value, Real) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _sequence(value):
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _complex_finite(value):
    return math.isfinite(value.real) and math.isfinite(value.imag)


def _parts(value):
    return [float(value.real), float(value.imag)]


def _magnitude(value):
    return math.hypot(value.real, value.imag)


def _section():
    return {"violations": [], "unknown": [], "checks": [], "checked_count": 0,
            "max_residual": None}


def _residual(section, residual):
    previous = section["max_residual"]
    section["max_residual"] = residual if previous is None else max(previous, residual)


def audit_ac_snapshot(
    raw: Mapping,
    metadata: Mapping,
    abs_current_tolerance_a: float = 1e-4,
    relative_tolerance: float = 1e-6,
    abs_power_tolerance_kw: float = 1e-4,
    voltage_tolerance_v: float = 1e-5,
) -> dict:
    """Audit saved currents, voltages, powers, and explicitly linear Yprim.

    KCL uses ``absI + rel*sum(abs(each terminal injection))`` at each bus/node.
    Other identities use ``absolute + rel*max(abs(left), abs(right))``. These
    tolerances accommodate numerical error; they do not relax device limits.
    NodeOrder, not phase count or bus-string suffixes, identifies conductors.
    Node zero is excluded from KCL and cross-element voltage comparisons.

    Missing/nonfinite operating-point data and stale nonzero disabled-element
    currents produce UNKNOWN. Disabled-element NodeOrder, voltage, power, and
    Yprim caches are excluded from operating-point identities: only their saved
    currents must be finite and zero. Frozen disabled BusNames/layout agreement
    belongs to the isolation certificate. Finite residuals are retained even when
    other evidence is missing. Nonconverged iterates never pass this audit.
    """
    if not isinstance(raw, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError("raw and metadata must be mappings")
    tolerances = {"abs_current_tolerance_a": abs_current_tolerance_a,
                  "relative_tolerance": relative_tolerance,
                  "abs_power_tolerance_kw": abs_power_tolerance_kw,
                  "voltage_tolerance_v": voltage_tolerance_v}
    if not all(_finite(value) and value >= 0 for value in tolerances.values()):
        raise ValueError("Numerical tolerances must be finite and nonnegative")

    sections = {key: _section() for key in (
        "inventory", "kcl", "voltage_consistency", "power_identity",
        "linear_yprim", "disabled_elements")}
    inventory = sections["inventory"]
    elements = raw.get("elements")
    if not _sequence(elements):
        inventory["unknown"].append("Missing/malformed physical element list")
        elements = []
    expected = metadata.get("expected_element_names")
    excluded = metadata.get("excluded_controls", [])

    def name_list(value, label, require_nonempty=False):
        if not _sequence(value) or (require_nonempty and not value):
            inventory["unknown"].append("Missing/malformed " + label)
            return set()
        names = []
        for name in value:
            if not isinstance(name, str) or not name.strip():
                inventory["unknown"].append("Invalid name in " + label)
                continue
            names.append(name.strip().lower())
        if len(names) != len(set(names)):
            inventory["unknown"].append("Duplicate names in " + label)
        return set(names)

    expected_names = name_list(expected, "expected_element_names", True)
    excluded_names = name_list(excluded, "excluded_controls")
    if expected_names & excluded_names:
        inventory["unknown"].append("Physical inventory overlaps excluded controls")
    present, duplicates = set(), set()
    for item in elements:
        if not isinstance(item, Mapping) or not isinstance(item.get("name"), str) or not item["name"].strip():
            inventory["unknown"].append("Malformed or unnamed physical element")
            continue
        name = item["name"].strip().lower()
        if name in present:
            duplicates.add(name)
        present.add(name)
    absent, unexpected = expected_names - present, present - expected_names
    if absent:
        inventory["unknown"].append("Missing physical elements: " + str(sorted(absent)))
    if unexpected:
        inventory["unknown"].append("Unexpected physical elements: " + str(sorted(unexpected)))
    if duplicates:
        inventory["unknown"].append("Duplicate physical elements: " + str(sorted(duplicates)))
    inventory.update(expected_names=sorted(expected_names), present_names=sorted(present),
                     missing_names=sorted(absent), unexpected_names=sorted(unexpected),
                     excluded_controls=sorted(excluded_names), checked_count=len(elements))

    currents_by_node, voltages_by_node = {}, {}
    layout_complete = True
    supported_linear = {"line", "transformer", "capacitor"}
    skipped_linear = []
    skipped_disabled = []
    frozen_layouts = metadata.get("element_layout", {})
    frozen_layouts = ({str(key).strip().lower(): value for key, value in frozen_layouts.items()}
                      if isinstance(frozen_layouts, Mapping) else {})

    def problem(section, message):
        sections[section]["unknown"].append(message)

    def complex_array(item, name, field, size, section, optional=False):
        value = item.get(field)
        if optional and value is None:
            return None
        if not _sequence(value) or len(value) != 2 * size or not all(_finite(x) for x in value):
            problem(section, f"{name}: missing/nonfinite/malformed {field}")
            return None
        return [complex(float(value[2 * i]), float(value[2 * i + 1])) for i in range(size)]

    def identity(section_name, left, right, absolute, context):
        section = sections[section_name]
        difference = left - right
        if not all(_complex_finite(value) for value in (left, right, difference)):
            problem(section_name, f"Nonfinite derived identity: {context}")
            return
        residual = _magnitude(difference)
        bound = absolute + relative_tolerance * max(_magnitude(left), _magnitude(right))
        if not math.isfinite(residual) or not math.isfinite(bound):
            problem(section_name, f"Nonfinite derived residual/tolerance: {context}")
            return
        check = {**context, "residual": residual, "tolerance": bound,
                 "left_real_imag": _parts(left), "right_real_imag": _parts(right)}
        section["checks"].append(check)
        section["checked_count"] += 1
        _residual(section, residual)
        if residual > bound:
            section["violations"].append(check)

    for item in elements:
        if not isinstance(item, Mapping):
            layout_complete = False
            continue
        name = str(item.get("name", "")).strip().lower()
        enabled = item.get("enabled")
        if not isinstance(enabled, bool):
            problem("inventory", f"{name}: missing/nonboolean enabled state")
            layout_complete = False
        dimension_source = frozen_layouts.get(name, item) if enabled is False else item
        if not isinstance(dimension_source, Mapping):
            problem("inventory", f"{name}: malformed frozen disabled-element layout")
            layout_complete = False
            continue
        dimension_keys = ("conductors", "terminals") if enabled is False else ("conductors", "terminals", "phases")
        dimensions = [dimension_source.get(key) for key in dimension_keys]
        valid_dimensions = all(isinstance(x, Integral) and not isinstance(x, bool) and x > 0 for x in dimensions)
        if not valid_dimensions or (enabled is not False and dimensions[2] > dimensions[0]):
            problem("inventory", f"{name}: malformed conductor/terminal/phase dimensions")
            layout_complete = False
            continue
        ncond, nterm = map(int, dimensions[:2])
        size = ncond * nterm
        currents = complex_array(item, name, "currents_real_imag", size, "kcl")
        if enabled is False:
            disabled = sections["disabled_elements"]
            disabled["checked_count"] += 1
            skipped_disabled.append(name)
            skipped_linear.append({"element": name, "reason": "disabled_cache_not_operating_point"})
            if currents is None:
                problem("disabled_elements", f"{name}: disabled currents unavailable/nonfinite")
            else:
                largest = max(_magnitude(value) for value in currents)
                if not math.isfinite(largest):
                    problem("disabled_elements", f"{name}: nonfinite derived disabled-current magnitude")
                else:
                    _residual(disabled, largest)
                    disabled["checks"].append({"element": name, "max_current_a": largest,
                                               "tolerance_a": abs_current_tolerance_a,
                                               "cached_voltage_power_yprim_not_audited": True,
                                               "cached_node_order_not_audited": True,
                                               "current_dimensions_source": "frozen_element_layout" if name in frozen_layouts else "saved_dimensions"})
                if largest > abs_current_tolerance_a:
                    problem("disabled_elements", f"{name}: nonzero disabled currents; stale snapshot cannot be accepted")
            continue

        buses, nodes = item.get("buses"), item.get("node_order")
        valid_layout = (_sequence(buses) and len(buses) == nterm and
                        all(isinstance(bus, str) and bus.split('.')[0].strip() for bus in buses) and
                        _sequence(nodes) and len(nodes) == size and
                        all(isinstance(node, Integral) and not isinstance(node, bool) and node >= 0 for node in nodes))
        if not valid_layout:
            problem("inventory", f"{name}: malformed terminal buses/NodeOrder")
            layout_complete = False
        volts = complex_array(item, name, "voltages_real_imag", size, "voltage_consistency")
        powers = complex_array(item, name, "powers_kw_kvar", size, "power_identity")
        requested = item.get("linear_yprim_check")
        if not isinstance(requested, bool):
            problem("linear_yprim", f"{name}: missing/nonboolean linear_yprim_check")
        yprim = complex_array(item, name, "yprim_real_imag", size * size,
                              "linear_yprim", optional=requested is not True)

        if enabled is True and (not valid_layout or currents is None):
            layout_complete = False
        if valid_layout:
            for index, node in enumerate(nodes):
                if node == 0:
                    continue
                bus = buses[index // ncond].split('.')[0].strip().lower()
                key = (bus, int(node))
                context = {"element": name, "terminal": index // ncond + 1,
                           "conductor": index % ncond + 1, "node": f"{bus}.{node}"}
                if enabled is True and currents is not None:
                    currents_by_node.setdefault(key, []).append((name, index, currents[index]))
                if volts is not None:
                    if key in voltages_by_node:
                        reference = voltages_by_node[key]
                        identity("voltage_consistency", volts[index], reference[1], voltage_tolerance_v,
                                 {**context, "reference_element": reference[0]})
                    else:
                        voltages_by_node[key] = (name, volts[index])
        if volts is not None and currents is not None and powers is not None:
            for index in range(size):
                computed = volts[index] * currents[index].conjugate() / 1000.0
                identity("power_identity", powers[index], computed, abs_power_tolerance_kw,
                         {"element": name, "terminal": index // ncond + 1,
                          "conductor": index % ncond + 1})

        kind = name.split('.')[0]
        if requested is True and kind not in supported_linear:
            problem("linear_yprim", f"{name}: Yprim identity forbidden for non-PD/unsupported element class")
        elif requested is True and enabled is True and yprim is not None and volts is not None and currents is not None:
            for row in range(size):
                computed = sum((yprim[row + column * size] * volts[column] for column in range(size)), 0j)
                identity("linear_yprim", currents[row], computed, abs_current_tolerance_a,
                         {"element": name, "terminal": row // ncond + 1,
                          "conductor": row % ncond + 1, "matrix_order": "column_major"})
        else:
            skipped_linear.append({"element": name, "reason": "disabled" if enabled is False else "not_requested_or_incomplete"})

    kcl = sections["kcl"]
    if not layout_complete or inventory["unknown"]:
        kcl["unknown"].append("KCL physical-element coverage is incomplete or ambiguous")
    if not currents_by_node:
        kcl["unknown"].append("No enabled nonground node-current contributions")
    for (bus, node), entries in sorted(currents_by_node.items()):
        values = [entry[2] for entry in entries]
        try:
            total = complex(math.fsum(value.real for value in values), math.fsum(value.imag for value in values))
            scale = math.fsum(_magnitude(value) for value in values)
        except (OverflowError, ValueError):
            problem("kcl", f"Nonfinite derived KCL sum: {bus}.{node}")
            continue
        residual = _magnitude(total)
        bound = abs_current_tolerance_a + relative_tolerance * scale
        if not _complex_finite(total) or not math.isfinite(residual) or not math.isfinite(bound):
            problem("kcl", f"Nonfinite derived KCL residual/tolerance: {bus}.{node}")
            continue
        check = {"node": f"{bus}.{node}", "residual_a": residual,
                 "residual_real_imag_a": _parts(total), "tolerance_a": bound,
                 "sum_absolute_injections_a": scale,
                 "contributions": [{"element": name, "index": index, "current_real_imag_a": _parts(current)}
                                   for name, index, current in entries]}
        kcl["checks"].append(check)
        kcl["checked_count"] += 1
        _residual(kcl, residual)
        if residual > bound:
            kcl["violations"].append(check)

    converged = raw.get("converged") is True
    sections["linear_yprim"]["skipped_elements"] = skipped_linear
    for key in ("voltage_consistency", "power_identity"):
        sections[key]["skipped_disabled_elements"] = list(skipped_disabled)
    sections["disabled_elements"]["cache_scope"] = (
        "Disabled NodeOrder/voltage/power/Yprim caches may be stale and are not operating-point evidence; only saved zero finite currents are verified; frozen BusNames/layout agreement requires the separate isolation certificate")
    for key, section in sections.items():
        diagnostic = "FAIL" if section["violations"] else "UNKNOWN" if section["unknown"] else "PASS"
        section["diagnostic_status"] = diagnostic
        section["status"] = ("UNKNOWN" if section["unknown"] or (not converged and key != "inventory")
                             else diagnostic)
        if not converged and key != "inventory":
            section["unknown"].append("Solver convergence not established; residuals describe an iterate only")
    unknown = [item for section in sections.values() for item in section["unknown"]]
    violations = [key for key, section in sections.items() if section["violations"]]
    status = "UNKNOWN" if unknown or not converged else "FAIL" if violations else "PASS"
    return {"status": status,
            "numerical_snapshot_verified": True if status == "PASS" else False if status == "FAIL" else None,
            "converged": converged, **sections,
            "unknown": unknown, "failed_sections": violations,
            "numerical_tolerances": tolerances,
            "full_ac_model_verified": "NOT_ESTABLISHED",
            "limitations": [
                "These numerical tolerances do not relax equipment operating limits",
                "Node zero is excluded from KCL and cross-element voltage comparison",
                "Disabled-element NodeOrder/voltage/power/Yprim caches are not operating-point evidence and are excluded from identities",
                "Disabled frozen BusNames/layout agreement requires the separate isolation certificate",
                "Yprim checks apply only to explicitly flagged enabled linear Line/Transformer/Capacitor elements",
                "Nonlinear Load/Generator/Vsource constitutive equations are not independently verified",
                "Equipment ratings, DER capabilities, topology operability, and switching transients require separate checks",
            ]}
