"""Common finite-grid DER curtailment for the named grid-connected extension.

Operating setpoints change; installed/available capacities and feeder data do not.
The first validation domain excludes energized tie1 and island voltage sources.
"""
from __future__ import annotations

import math
import warnings

import networkx as nx
import numpy as np

from .nature13_runtime import Nature13Runtime


def action_graph(core, action, faults):
    graph = core.G_init.copy()
    graph.remove_edges_from(faults)
    for bit, edge in zip(action[:len(core.SwitchLines)], core.SwitchLines):
        if bit == 0 and graph.has_edge(*edge):
            graph.remove_edge(*edge)
    return graph


def validated_domain(core, action, faults):
    bits = np.asarray(action)
    if bits.shape != (11,) or not np.isin(bits, [0, 1]).all():
        raise ValueError("Expected eleven binary switch/load entries")
    if bits[2]:
        return False, "TIE1_PHASE_MAPPING_OUTSIDE_VALIDATED_DOMAIN"
    fault_set = {frozenset(edge) for edge in faults}
    for bit, edge in zip(bits[:4], core.SwitchLines):
        if bit and frozenset(edge) in fault_set:
            return False, "FAULTED_SWITCH_REQUEST"
    graph = action_graph(core, bits, faults)
    if not nx.is_forest(graph):
        return False, "NONFOREST_TOPOLOGY"
    if not nx.is_connected(graph):
        return False, "ISLANDING_OUTSIDE_VALIDATED_DOMAIN"
    return True, "GRID_CONNECTED_DOMAIN"


def select_grid_point(records):
    """Largest verified alpha; never assume monotone feasibility or optimality."""
    eligible = [row for row in records if row["known_constraints"]["known_constraints_satisfied"] is True
                and row["ac_audit"]["status"] == "PASS"]
    if not eligible:
        return None
    return max(eligible, key=lambda row: (row["alpha"], -row["id"]))


class Nature13PhysicalRuntime(Nature13Runtime):
    def __init__(self, project, run, physical_config, events=None):
        super().__init__(project, run, events)
        self.physical_config = physical_config
        self.initial_metadata = self._metadata()

    def _metadata(self):
        dss = self.dss
        all_elements = list(dss.Circuit.AllElementNames())
        controls = [name for name in all_elements if name.split(".")[0].lower() in ("regcontrol", "swtcontrol")]
        expected = [name for name in all_elements if name not in controls]
        allowed = {"line", "transformer", "load", "generator", "capacitor", "vsource"}
        unknown = [name for name in expected if name.split(".")[0].lower() not in allowed]
        if unknown:
            raise ValueError("Uncaptured electrical element classes: " + str(unknown))
        phases, bases = {}, {}
        for bus in dss.Circuit.AllBusNames():
            dss.Circuit.SetActiveBus(bus)
            phases[bus.lower()] = [int(node) for node in dss.Bus.Nodes() if node > 0]
            bases[bus.lower()] = dss.Bus.kVBase()
        line_codes = {}
        for name in dss.Lines.AllNames():
            dss.Lines.Name(name)
            line_codes["Line." + name] = dss.Lines.LineCode().lower().removeprefix("mtx")
        transformers = {}
        for name in dss.Transformers.AllNames():
            dss.Transformers.Name(name)
            dss.Circuit.SetActiveElement("Transformer." + name)
            nphase = dss.CktElement.NumPhases()
            windings = []
            for winding in range(1, dss.Transformers.NumWindings() + 1):
                dss.Transformers.Wdg(winding)
                windings.append({"kva": dss.Transformers.kVA(), "kv": dss.Transformers.kV(),
                                 "connection": "delta" if dss.Transformers.IsDelta() else "wye"})
            transformers["Transformer." + name] = {"phases": nphase, "windings": windings}
        for name, rating in self.physical_config["expected_transformer_kva"].items():
            actual = transformers["Transformer." + name.lower()]
            if any(abs(w["kva"] - rating) > 1e-8 for w in actual["windings"]):
                raise ValueError("Transformer model rating differs from registered source: " + name)
        generation = {"Generator.g" + str(g["no"]): {
            "bus": g["bus"], "available_p_kw": float(g["size"]),
            "reference_q_kvar": 0.75 * float(g["size"]), "grid_forming_capability": g["Gridforming"] == "Yes"}
            for g in self.core.generators}
        return {"node_phases": phases, "node_base_kv_ln": bases,
                "source_buses": [self.core.substatn_id], "line_config_by_name": line_codes,
                "line_ratings": {code: {"phase_amps": limit} for code, limit in self.physical_config["line_phase_ampacity_A"].items()},
                "transformer_ratings": transformers,
                "gfm_limits": {name: {"p_min_kw": 0.0, "p_max_kw": data["available_p_kw"]} for name, data in generation.items()},
                "generator_reference": generation, "expected_element_names": expected, "excluded_controls": controls,
                "ideal_line_names": ["Line.671692"], "neutral_thermal_mode": "NOT_MODELED",
                "known_limitations": self.physical_config["explicit_model_boundaries"]}

    def extended_snapshot(self, graph):
        raw = self.raw_snapshot(graph)
        dss = self.dss
        for element in raw["elements"]:
            dss.Circuit.SetActiveElement(element["name"])
            element.update(node_order=dss.CktElement.NodeOrder(),
                           currents_real_imag=dss.CktElement.Currents(),
                           voltages_real_imag=dss.CktElement.Voltages(),
                           conductor_open=[[bool(dss.CktElement.IsOpen(t, c)) for c in range(1, element["conductors"] + 1)]
                                           for t in range(1, element["terminals"] + 1)])
            # Refresh the same element's power/current caches together. In
            # particular, Newton PC currents can otherwise be cached at the
            # previous iterate when powers were collected in the first pass.
            element["powers_kw_kvar"] = dss.CktElement.Powers()
            element["currents_mag_angle"] = dss.CktElement.CurrentsMagAng()
            kind, name = element["name"].split(".", 1)
            element["linear_yprim_check"] = kind.lower() in ("line", "transformer", "capacitor")
            if element["linear_yprim_check"]:
                element["yprim_real_imag"] = dss.CktElement.YPrim()
            if kind.lower() == "generator":
                dss.Generators.Name(name)
                element["operating_setpoint"] = {"kw": dss.Generators.kW(), "kvar": dss.Generators.kvar(), "pf": dss.Generators.PF()}
            if kind.lower() == "transformer":
                dss.Transformers.Name(name)
                element["winding_taps"] = []
                for winding in range(1, dss.Transformers.NumWindings() + 1):
                    dss.Transformers.Wdg(winding)
                    element["winding_taps"].append(dss.Transformers.Tap())
        raw["circuit_all_element_names"] = dss.Circuit.AllElementNames()
        raw["control_iterations"] = dss.Solution.ControlIterations()
        raw["algorithm"] = dss.Solution.Algorithm()
        return raw

    def evaluate_point(self, action, case, alpha, algorithm):
        if not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("Curtailment multiplier must be in [0,1]")
        accepted, reason = validated_domain(self.core, action, case["fault_edges"])
        if not accepted:
            return {"domain_status": reason, "alpha": alpha, "raw_outputs": None}
        before = len(self.events)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            obj, graph = self.core.take_action(np.asarray(action, dtype=int), case["fault_edges"])
            for name, ref in self.initial_metadata["generator_reference"].items():
                obj.dss.Circuit.SetActiveElement(name)
                if not obj.dss.CktElement.Enabled():
                    raise ValueError("Grid-connected curtailment unexpectedly encountered a disabled DER")
                obj.dss.Text.Command(f"Edit {name} kW={alpha * ref['available_p_kw']:.12g} kvar={alpha * ref['reference_q_kvar']:.12g}")
            obj.dss.Solution.Algorithm(algorithm)
            obj.dss.Solution.Convergence(self.physical_config.get("solver_convergence_tolerance", 0.0001))
            obj.dss.Solution.Solve()
            raw = self.extended_snapshot(graph)
        if any(e["name"].lower().startswith("vsource.bus_") for e in raw["elements"]):
            raise ValueError("An island voltage source appeared inside the validated domain")
        return {"domain_status": reason, "alpha": alpha, "action": list(action), "case": case["name"],
                "fault_lines": case["fault_lines"], "algorithm": algorithm, "raw_outputs": raw,
                "requested_der_p_kw": alpha * sum(r["available_p_kw"] for r in self.initial_metadata["generator_reference"].values()),
                "python_solve_calls": sum(e["method"] == "Solution.Solve" for e in self.events[before:]),
                "compile_commands": sum(e["method"] == "Text.Command" and str(e["args"][0]).lower().startswith("compile") for e in self.events[before:]),
                "warnings": [{"category": w.category.__name__, "message": str(w.message), "line": w.lineno} for w in caught]}
