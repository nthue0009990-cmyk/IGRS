"""Explicit, serial adapter of the pinned NC2024 numerical benchmark.

This adapter does not certify ampacity, inverter capability, or island balance.
It avoids the author's recursive reset and preserves raw numerical outcomes.
"""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import warnings

import networkx as nx
import numpy as np


class EngineCounter:
    def __init__(self, target, events, prefix=""):
        self.target, self.events, self.prefix = target, events, prefix

    def __getattr__(self, name):
        target = getattr(self.target, name)
        if name in ("Solution", "Text"):
            return EngineCounter(target, self.events, name)
        if callable(target) and (self.prefix, name) in (("Solution", "Solve"), ("Text", "Command")):
            def call(*args, **kwargs):
                row = {"method": self.prefix + "." + name, "args": list(args)}
                self.events.append(row)
                try:
                    return target(*args, **kwargs)
                except Exception as exc:
                    row["error"] = repr(exc)
                    raise
            return call
        return target


def topology_context(nodes, edges, switch_edges, load_buses, source, forming_buses, fault_edges):
    """Pure graph features for every little-endian switch assignment; no PF."""
    table = []
    for index in range(2 ** len(switch_edges)):
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        graph.remove_edges_from(fault_edges)
        for bit, edge in enumerate(switch_edges):
            if not (index >> bit) & 1 and graph.has_edge(*edge):
                graph.remove_edge(*edge)
        components = list(nx.connected_components(graph))
        component_of = {node: comp for comp in components for node in comp}
        values = []
        for bus in load_buses:
            comp = component_of[bus]
            values.extend((float(source in comp), float(bool(set(forming_buses) & comp)), len(comp) / len(nodes)))
        values.extend((float(nx.is_forest(graph)), len(components) / len(nodes)))
        table.append(values)
    return np.asarray(table, dtype=np.float32)


class Nature13Runtime:
    def __init__(self, project: Path, run: Path, events=None):
        self.project, self.run = project, run
        self.events = [] if events is None else events
        pointer = json.loads((project / "artifacts/restoration_shortlist_sources_latest.json").read_text())
        source = project / pointer["derived_directory"] / "nature_author_source"
        relative = Path("Environments/DSSdirect_13bus_loadandswitching")
        self.work = run / "work"
        shutil.copytree(source / "RL_Training" / relative, self.work / relative)
        extra = source / "RL_Testing/13bus/IEEELineCodes.dss"
        shutil.copy2(extra, self.work / relative / extra.name)
        # Only one global DSS engine in this process. Never instantiate two runtimes.
        package = "Environments.DSSdirect_13bus_loadandswitching."
        if any(name.startswith(package) for name in sys.modules):
            raise RuntimeError("A Nature13Runtime has already been imported in this process")
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(self.work))
        import opendssdirect as dss
        self.dss = dss
        self.engine_version = dss.Basic.Version()
        setup = importlib.import_module(package + "DSS_CircuitSetup")
        setup.dss = EngineCounter(dss, self.events)
        self.core = importlib.import_module(package + "state_action_reward")
        self.metadata = {
            "source": pointer, "extra_linecodes_source": str(extra.relative_to(project)),
            "nodes": list(self.core.node_list), "switches": self.core.AllSwitches,
            "dispatch_loads": self.core.dispatch_loads, "engine_version": self.engine_version,
            "initial_observation_adaptation": "Graph excludes normally open ties as well as faulted edges; no recursive resampling",
            "physical_values_modified": False,
            "full_operational_feasibility": "NOT_ESTABLISHED",
        }

    def fault_edges(self, lines):
        result = []
        for line in lines:
            matches = [(u, v) for u, v, data in self.core.G_init.edges(data=True)
                       if line.lower() in [label.lower() for label in data["label"]]]
            if len(matches) != 1:
                raise ValueError((line, matches))
            result.extend(matches)
        return result

    def _state(self, obj, graph, faults):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            state = self.core.get_state(obj, graph, faults)
        notes = [{"message": str(w.message), "category": w.category.__name__, "line": w.lineno} for w in caught]
        return state, notes

    @staticmethod
    def vectorize(state):
        # An explicit missingness channel, shared by both policies, replaces the
        # source reset's hidden rejection of nonfinite observations.
        keys = ["EnergySupp", "NodeFeat(BusVoltage)", "EdgeFeat(Branchflow)", "Adjacency",
                "VoltageViolation", "ConvergenceViolation", "ActionMasking"]
        values = np.concatenate([np.asarray(state[key], dtype=np.float64).ravel() for key in keys])
        finite = np.isfinite(values)
        encoded = np.concatenate([np.where(finite, values, 0.0), (~finite).astype(float)]).astype(np.float32)
        if not np.isfinite(encoded).all():
            raise ValueError("Observation exceeds float32 representation")
        return encoded, int((~finite).sum())

    def prepare_case(self, name, lines):
        faults = self.fault_edges(lines)
        obj, graph, _ = self.core.initialize()
        graph = graph.copy()
        for edge in faults:
            label = self.core.G_init.edges[edge]["label"][0]
            obj.dss.Text.Command(f"Open {label} term=1")
            obj.dss.Solution.Solve()
            graph.remove_edge(*edge)
        for switch in self.core.AllSwitches:
            if not switch["status"] and graph.has_edge(switch["from bus"], switch["to bus"]):
                graph.remove_edge(switch["from bus"], switch["to bus"])
        state, notes = self._state(obj, graph, faults)
        observation, missing = self.vectorize(state)
        table = topology_context(list(self.core.G_init.nodes()), list(self.core.G_init.edges()),
                                 self.core.SwitchLines,
                                 [self.core.Load_Buses[name] for name in self.core.dispatch_loads],
                                 self.core.substatn_id,
                                 [bus for bus, data in self.core.Gen_Info.items() if data["Blackstart"]], faults)
        forced = np.asarray(state["ActionMasking"][:len(self.core.SwitchLines)], dtype=bool)
        return {"name": name, "fault_lines": lines, "fault_edges": faults, "observation": observation,
                "state": state, "nonfinite_feature_count": missing, "warnings": notes,
                "switch_forced_open": forced, "context_table": table}

    def evaluate(self, action, case):
        action = np.asarray(action, dtype=np.int64)
        if action.shape != (11,) or not np.isin(action, [0, 1]).all():
            raise ValueError("Expected 11 binary action entries")
        if np.any(action[:4][case["switch_forced_open"]]):
            raise ValueError("A sampled action closes a faulted switch")
        before = len(self.events)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            obj, graph = self.core.take_action(action, case["fault_edges"])
            # Capture raw values before the author's get_state zero imputation.
            raw = self.raw_snapshot(graph)
            state = self.core.get_state(obj, graph, case["fault_edges"])
        energy = float(np.asarray(state["EnergySupp"]).item())
        voltage_penalty = float(np.asarray(state["VoltageViolation"]).item())
        reward = energy - voltage_penalty if raw["converged"] and np.isfinite(voltage_penalty) else 0.0
        if not np.isfinite(reward):
            raise FloatingPointError("Nonfinite author-model reward; no sample substitution")
        return {"case": case["name"], "fault_lines": case["fault_lines"], "action": action,
                "reward": reward, "author_power_fraction": energy, "author_voltage_penalty": voltage_penalty,
                "raw_outputs": raw,
                "warnings": [{"message": str(w.message), "category": w.category.__name__, "line": w.lineno} for w in caught],
                "python_solve_calls": sum(e["method"] == "Solution.Solve" for e in self.events[before:]),
                "compile_commands": sum(e["method"] == "Text.Command" and str(e["args"][0]).lower().startswith("compile") for e in self.events[before:])}

    def raw_snapshot(self, graph):
        dss = self.dss
        elements = []
        for name in dss.Circuit.AllElementNames():
            if name.lower().split(".")[0] not in ("line", "load", "generator", "transformer", "vsource", "capacitor"):
                continue
            dss.Circuit.SetActiveElement(name)
            elements.append({"name": name, "enabled": dss.CktElement.Enabled(),
                             "buses": dss.CktElement.BusNames(), "conductors": dss.CktElement.NumConductors(),
                             "phases": dss.CktElement.NumPhases(), "terminals": dss.CktElement.NumTerminals(),
                             "powers_kw_kvar": dss.CktElement.Powers(), "currents_mag_angle": dss.CktElement.CurrentsMagAng()})
        volts = np.asarray(dss.Circuit.AllBusMagPu())
        return {"converged": dss.Solution.Converged(), "iterations": dss.Solution.Iterations(),
                "node_names": dss.Circuit.AllNodeNames(), "node_vmag_pu": volts,
                "node_voltages_real_imag": dss.Circuit.AllBusVolts(),
                "voltage_nonfinite_count": int((~np.isfinite(volts)).sum()),
                "circuit_total_power_kw_kvar": dss.Circuit.TotalPower(), "losses_w_var": dss.Circuit.Losses(),
                "graph_nodes": list(graph.nodes()), "graph_edges": list(graph.edges()),
                "is_forest": nx.is_forest(graph), "components": [sorted(c) for c in nx.connected_components(graph)],
                "elements": elements, "full_operational_feasibility": "NOT_ESTABLISHED"}
