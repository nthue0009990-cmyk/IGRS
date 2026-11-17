"""Independent, solver-free replay of the bounded CLPU pilot.

This audit recomputes dynamic demand and service accounting from saved inputs;
it never initializes OpenDSS and does not regard a saved PASS label as proof.
"""
from __future__ import annotations

from copy import deepcopy
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback
import zipfile

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
sys.dont_write_bytecode = True

from dnr.restoration_ac_audit import audit_ac_snapshot
from dnr.restoration_constraints import check_known_constraints
from dnr.restoration_isolation import verify_isolation

_projection_context = None
_projection_cache = {}


def require(value, message):
    if not value:
        raise AssertionError(message)


def close(actual, expected, label, *, abs_tol=1e-8):
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and math.isfinite(actual) and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=abs_tol),
            label + f": {actual!r} != {expected!r}")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_lines(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def check_inputs(run):
    """Independently bind pilot inputs to sealed build data and whole events."""
    provenance = read(run / "source_provenance.json")
    for label in ("scheduler", "scheduler_audit"):
        source = provenance[label]
        directory = ROOT / "runs" / source["run"]
        require(sha(directory / "completion_manifest.json") == source["completion_sha256"],
                label + " completion changed")
        manifest = read(directory / "completion_manifest.json")
        for name, expected in source["consumed_files"].items():
            require(manifest[name] == expected == sha(directory / name), "Unsealed consumed source: " + name)
            if label == "scheduler":
                require(sha(run / "source_inputs" / name) == expected, "Copied source input changed: " + name)
                # Runtime metadata is separately allowed to describe CLPU.
                if name != "metadata.json":
                    require(sha(run / name) == expected, "Scientific input modified: " + name)
    source_scenarios = read(run / "training_scenarios.json")
    require(len(source_scenarios) == len({s["scenario_id"] for s in source_scenarios}) == 16
            and all(s["partition"] == "build" and s["fault_order"] == 2 for s in source_scenarios),
            "Pilot source includes nontraining or non-N2 faults")
    fault_domain, event_domain = "clpu-pilot-faults-v1", "clpu-event-split-v1"
    ordered_scenarios = sorted(source_scenarios, key=lambda s: hashlib.sha256(
        f"{fault_domain}|{s['scenario_id']}".encode()).hexdigest())
    require(read(run / "selected_scenarios.json") == ordered_scenarios[:8], "Outcome-dependent fault selection")
    event_ids = sorted(range(1, 32), key=lambda n: hashlib.sha256(f"{event_domain}|{n}".encode()).hexdigest())
    expected_partition = {"train": event_ids[:19], "dev": event_ids[19:25], "holdout": event_ids[25:]}
    require(read(run / "event_split.json") == expected_partition, "Whole-event split differs")
    selected_events = read(run / "selected_events.json")
    require([e["event_id"] for e in selected_events] == event_ids[:4], "Event selection changed")
    curve = provenance["clpu_source"]
    source_zip = ROOT / curve["path"]
    require(sha(source_zip) == curve["sha256"] == "e46886903929d49380a7f1342588adebd26c2524667dbd74fa89021ccea27a22",
            "Raw public CLPU source changed")
    with zipfile.ZipFile(source_zip) as archive:
        rows = [line.strip() for line in archive.read("parameters.csv").decode("utf-8-sig").splitlines()
                if line.strip()]
        require(len(rows) == 32, "Expected 31 complete source events")
        for event in selected_events:
            tokens = [token.strip() for token in rows[event["event_id"]].split(",")]
            require(len(tokens) in (9, 10), "Unknown raw parameter CSV schema")
            a = float(tokens[0] + "." + tokens[1])
            tau = float(tokens[2] + "." + tokens[3])
            close(event["a"], a, "Paired event ED amplitude")
            close(event["tau_seconds"], tau, "Paired event ED time constant")
            require(event["parameter_row"] == rows[event["event_id"]], "Source ED parameter pair replaced")
    condition = read(run / "condition.json")
    require(condition["condition_id"] == "author_daily_lower_median" and condition in read(run / "conditions.json"),
            "Operating condition differs from sealed source")
    cfg = read(run / "config.json")
    for key in ("test_access", "tune_access", "formal_holdout_numerical_access", "tune_numerical_access"):
        require(cfg.get(key, False) is False, "Forbidden numerical data access")
    require(cfg.get("alpha", .5) == .5, "Fixed DER setting changed")
    return {"training_source_faults": 16, "selected_faults": 8, "selected_events": 4,
            "selected_event_ids": event_ids[:4], "source_hash_checks": "PASS",
            "outcome_dependent_selection": False}


def independent_projection(static, loads, scenario, command):
    """Rebuild the source component directly; do not call JointGraph or env."""
    global _projection_context, _projection_cache
    if (_projection_context is None or _projection_context[0] is not static
            or _projection_context[1] is not loads or _projection_context[2] is not scenario):
        _projection_context, _projection_cache = (static, loads, scenario), {}
    require(type(command) is int and 0 <= command < (1 << 19), "Invalid packed command")
    if command in _projection_cache:
        return _projection_cache[command]
    network = nx.Graph()
    network.add_nodes_from(str(n).lower() for n in static["nodes"])
    network.add_edges_from((str(a).lower(), str(b).lower()) for a, b in static["bus_edges"])
    originals = {name.lower(): tuple(str(n).lower() for n in edge)
                 for name, edge in static["original_lines"].items()}
    faults = {frozenset(originals[name.lower()]) for name in scenario["fault_lines"]}
    for edge in faults:
        a, b = tuple(edge)
        if network.has_edge(a, b):
            network.remove_edge(a, b)
    for slot, edge in enumerate(static["switch_edges"]):
        edge = tuple(str(n).lower() for n in edge)
        closed = bool(command & (1 << (slot + 10)))
        require(not (closed and frozenset(edge) in faults), "Faulted switch closed")
        if not closed and network.has_edge(*edge):
            network.remove_edge(*edge)
    require(nx.is_forest(network), "Nonradial command in accepted plan")
    powered = set(nx.node_connected_component(network, static["source_bus"].lower()))
    dispatch = {name.lower(): slot for slot, name in enumerate(static["dispatch_loads"])}
    energized = set()
    for name, row in loads.items():
        ready = name.lower() not in dispatch or bool(command & (1 << dispatch[name.lower()]))
        if str(row["bus"]).lower() in powered and ready:
            energized.add(name)
    switches = [(command >> (10 + slot)) & 1 for slot in range(9)]
    lower_energized = {name.lower() for name in energized}
    load_bits = [int(name.lower() in lower_energized) for name in static["dispatch_loads"]]
    result = {"energized": energized, "powered": powered, "graph": network, "action": switches + load_bits,
              "service_base_kw": sum(loads[name]["nominal_kw"] for name in energized)}
    _projection_cache[command] = result
    return result


def independent_initial(static, loads, scenario):
    originals = {name.lower(): frozenset(str(n).lower() for n in edge)
                 for name, edge in static["original_lines"].items()}
    faults = {originals[name.lower()] for name in scenario["fault_lines"]}
    switch_mask = sum(1 << i for i in range(5)
                      if frozenset(str(n).lower() for n in static["switch_edges"][i]) not in faults)
    command = (switch_mask << 10) | 1023
    energized = independent_projection(static, loads, scenario, command)["energized"]
    return {"command": command, "step": 0,
            "pickups": [-1 if name in energized else None for name in sorted(loads)]}


def independent_advance(before, action, static, loads, scenario, slot_seconds):
    require(type(action) is int and 0 <= action < 20, "Illegal action ID")
    before_projection = independent_projection(static, loads, scenario, before["command"])
    after_command = before["command"] if action == 0 else before["command"] ^ (1 << (action - 1))
    after_projection = independent_projection(static, loads, scenario, after_command)
    require(before_projection["energized"] <= after_projection["energized"], "Reinterrupted supplied user")
    require(type(before["step"]) is int and before["step"] >= 0, "Invalid elapsed step")
    pickups = list(before["pickups"])
    require(len(pickups) == len(loads), "Incomplete timer inventory")
    for slot, name in enumerate(sorted(loads)):
        if name in after_projection["energized"] - before_projection["energized"]:
            require(pickups[slot] is None, "Reconnection replaced previous history")
            pickups[slot] = before["step"] * slot_seconds
    return {"command": after_command, "step": before["step"] + 1, "pickups": pickups}


def independent_multipliers(state, time_seconds, loads, event):
    require(len(state["pickups"]) == len(loads), "Missing full-load pickup history")
    result = {}
    for name, startup in zip(sorted(loads), state["pickups"]):
        if startup is None or startup == -1:
            result[name] = 1.
        else:
            require(isinstance(startup, (int, float)) and not isinstance(startup, bool)
                    and 0 <= startup <= time_seconds, "Negative/future pickup age")
            result[name] = 1. + event["a"] * math.exp(-(time_seconds - startup) / event["tau_seconds"])
    return result


def check_saved_plan(record, start, static, loads, scenario, condition, event,
                     horizon, slot_seconds, sample_seconds, points):
    """Independently replay a whole candidate, including every failure prefix."""
    actions = record["actions"]
    require(len(actions) == horizon - start["step"], "Candidate is not the full remaining window")
    expected_states = [deepcopy(start)]
    graph_error = None
    try:
        for action in actions:
            expected_states.append(independent_advance(expected_states[-1], action, static, loads,
                                                       scenario, slot_seconds))
    except (AssertionError, nx.NetworkXException) as exc:
        graph_error = str(exc)
    if record["status"] == "GRAPH_FAIL":
        require(graph_error is not None and record["samples"] == [] and record["states"] == [],
                "Graph failure has no independent invalid-operation witness")
        return {"status": "GRAPH_FAIL", "samples": 0}
    require(graph_error is None and record["states"] == expected_states,
            "Saved dynamic states differ from independent full-inventory replay")
    expected_samples = []
    for after in expected_states[1:]:
        t0 = (after["step"] - 1) * slot_seconds
        expected_samples.extend((after, t0 + offset) for offset in range(0, slot_seconds + 1, sample_seconds))
    saved = record["samples"]
    require(record["status"] in ("COMPLETE", "FAIL"), "Unresolved path presented as certificate")
    if record["status"] == "COMPLETE":
        require(len(saved) == len(expected_samples), "Complete path lacks some required sample points")
    else:
        require(0 < len(saved) <= len(expected_samples), "Invalid failed-path sample prefix")
    for index, (sample, (state, t)) in enumerate(zip(saved, expected_samples)):
        require(sample["state"] == state and sample["step"] == state["step"]
                and sample["time_seconds"] == t, "Sample history/time differs from operation boundary")
        require(sample["point_id"] in points, "Path references an unpaid physical point")
        point = points[sample["point_id"]]
        require(sample["status"] == point["status"] and point["reliable"] is True,
                "Path status differs from raw physical evidence")
        if index < len(saved) - 1 or record["status"] == "COMPLETE":
            require(point["status"] == "PASS", "Path continues after known physical failure")
        expected_factors = independent_multipliers(state, t, loads, event)
        require(point["clpu_multipliers"].keys() == expected_factors.keys(), "Path point multiplier inventory")
        for name, factor in expected_factors.items():
            close(point["clpu_multipliers"][name], factor, "Path history/physical multiplier " + name)
        projection = independent_projection(static, loads, scenario, state["command"])
        require(point["action"] == projection["action"], "Path effective physical action differs")
        close(point["time_minutes"], t / 60., "Path point absolute time")
        identity = point.get("state_identity")
        if identity is not None:
            require(identity["command"] == state["command"] and identity["pickups"] == state["pickups"],
                    "Cached point reused with different energization history")
    if record["status"] == "FAIL":
        require(saved[-1]["status"] == "FAIL", "Failed path lacks a reliable final violation")
        samples_per_slot = slot_seconds // sample_seconds + 1
        require(record["failed_step"] == (len(saved) - 1) // samples_per_slot,
                "Feedback assigned to wrong recovery operation")
        require(record["failed_time_seconds"] == saved[-1]["time_seconds"]
                and record["failed_state"] == saved[-1]["state"], "Failure history/time differs")
        if "structured_feedback" in points[saved[-1]["point_id"]]:
            require(record["feedback"] == points[saved[-1]["point_id"]]["structured_feedback"]["violations"],
                    "Path repair feedback differs from its actual reliable failing snapshot")
        return {"status": "FAIL", "samples": len(saved)}
    total_kw = sum(row["nominal_kw"] for row in loads.values()) * condition["load_multiplier"]
    supplied_kw = [independent_projection(static, loads, scenario, state["command"])["service_base_kw"]
                   * condition["load_multiplier"] for state in expected_states[1:]]
    energy = math.fsum(p * slot_seconds / 3600. for p in supplied_kw)
    fraction = supplied_kw[-1] / total_kw if supplied_kw else independent_projection(
        static, loads, scenario, start["command"])["service_base_kw"] * condition["load_multiplier"] / total_kw
    close(record["energy_kwh"], energy, "Full-path useful normal energy")
    close(record["final_fraction"], fraction, "Final normal service")
    close(record["mean_served_fraction"], energy / (total_kw * len(actions) * slot_seconds / 3600.)
          if actions else fraction, "Mean useful normal service")
    require(record["operations"] == sum(a != 0 for a in actions), "Nonwait operation accounting")
    return {"status": "COMPLETE", "samples": len(saved), "energy_kwh": energy,
            "mean_served_fraction": record["mean_served_fraction"]}


def check_saved_point(row, condition, reference, metadata, physical_config,
                      expected_multipliers, expected_action, expected_time,
                      expected_energized):
    """Recheck one complete raw result without using the CLPU implementation.

    The caller independently replays energization times to supply the expected
    multipliers. Normal service and the extra burden are separately recomputed.
    All disabled loads remain in the full requested-setpoint inventory.
    """
    refs = {r["name"]: r for r in reference}
    normal = {r["name"]: r for r in condition["loads"]}
    require(len(refs) == len(reference) == len(normal) == 68, "Full 68-load inventory missing")
    require(refs.keys() == normal.keys() == expected_multipliers.keys()
            == row["clpu_multipliers"].keys(), "Dynamic multiplier names differ")
    require(row["action"] == list(expected_action), "Wrong physical action")
    close(row["time_minutes"], expected_time, "Wrong sample time")
    require(row["condition_id"] == condition["condition_id"], "Wrong normal-load condition")
    close(sum(r["nominal_kw"] for r in refs.values()), 1769., "Author normal P")
    close(sum(r["nominal_kvar"] for r in refs.values()), 1044., "Author normal Q")
    require(row["control_state_semantics"] == "INDEPENDENT_QUASI_STATIC_AUTHOR_RESET_EACH_SNAPSHOT",
            "Controller state convention differs")
    denominator = sum(r["nominal_kw"] for r in refs.values()) * condition["load_multiplier"]
    close(condition["denominator_nominal_kw"], denominator, "Normal demand denominator")
    expected_energized = set(expected_energized)
    require(expected_energized <= refs.keys(), "Unknown energized load")
    planned = row["metadata_override"]["planned_enabled"]
    require({name for name in refs if planned[name]} == expected_energized,
            "Physical load enable state differs from independent graph")
    observed = row["load_readback"]
    require(all(set(observed[stage]) == refs.keys() for stage in ("before", "after", "postsolve")),
            "Incomplete P/Q property readback")
    dynamic_rows = {r["name"]: r for r in row["clpu_load_ledger"]["loads"]}
    normal_rows = {r["name"]: r for r in row["nominal_load_ledger"]["loads"]}
    require(len(dynamic_rows) == len(normal_rows) == 68 and dynamic_rows.keys() == normal_rows.keys() == refs.keys(),
            "Dynamic/normal ledger missing loads")
    normal_kw = dynamic_kw = dynamic_kvar = extra_kw = extra_kvar = 0.
    all_dynamic_kw = all_dynamic_kvar = 0.
    for name, ref in refs.items():
        factor = expected_multipliers[name]
        require(math.isfinite(factor) and factor >= 1., "Invalid independently replayed ED factor")
        close(row["clpu_multipliers"][name], factor, name + ": multiplier")
        p0 = ref["nominal_kw"] * condition["load_multiplier"]
        q0 = ref["nominal_kvar"] * condition["load_multiplier"]
        p, q = p0 * factor, q0 * factor
        close(normal[name]["demand_nominal_kw"], p0, name + ": base P")
        close(normal[name]["demand_nominal_kvar"], q0, name + ": base Q")
        require(ref["bus"] == normal[name]["bus"] == dynamic_rows[name]["bus"], "Changed load bus")
        for stage in ("after", "postsolve"):
            close(observed[stage][name]["nominal_kw"], p, name + ": P readback")
            close(observed[stage][name]["nominal_kvar"], q, name + ": Q readback")
            before_static = {k: v for k, v in observed["before"][name].items()
                             if k not in ("nominal_kw", "nominal_kvar")}
            after_static = {k: v for k, v in observed[stage][name].items()
                            if k not in ("nominal_kw", "nominal_kvar")}
            require(before_static == after_static, name + ": ZIP/phase/connection altered")
        for key, value in (("normal_nominal_kw", p0), ("normal_nominal_kvar", q0),
                           ("nominal_kw", p), ("nominal_kvar", q), ("clpu_multiplier", factor)):
            close(dynamic_rows[name][key], value, name + ": dynamic ledger " + key)
        close(normal_rows[name]["nominal_kw"], p0, name + ": normal ledger P")
        close(normal_rows[name]["nominal_kvar"], q0, name + ": normal ledger Q")
        expected_state = ("energized" if name in expected_energized else
                          "deenergized" if ref["bus"] in row["metadata_override"]["deenergized_buses"]
                          else "deliberately_shed")
        require(dynamic_rows[name]["state"] == normal_rows[name]["state"] == expected_state,
                name + ": load service classification")
        all_dynamic_kw += p
        all_dynamic_kvar += q
        if name in expected_energized:
            normal_kw += p0
            dynamic_kw += p
            dynamic_kvar += q
            extra_kw += p - p0
            extra_kvar += q - q0
    expected = {"normal_demand_denominator_kw": denominator,
                "normal_restored_fraction": normal_kw / denominator,
                "energized_normal_kw": normal_kw, "energized_dynamic_kw": dynamic_kw,
                "energized_dynamic_kvar": dynamic_kvar, "energized_clpu_extra_kw": extra_kw,
                "energized_clpu_extra_kvar": extra_kvar,
                "all_load_dynamic_setpoint_kw": all_dynamic_kw,
                "all_load_dynamic_setpoint_kvar": all_dynamic_kvar}
    for key, value in expected.items():
        close(row["clpu_load_ledger"][key], value, "CLPU total " + key)
    close(row["nominal_load_ledger"]["energized_nominal_kw"], normal_kw, "Normal served P")
    close(row["nominal_load_ledger"]["nominal_restored_fraction"], normal_kw / denominator,
          "Reward ledger must exclude CLPU extra burden")

    merged = {**metadata, **row["metadata_override"]}
    constraints = check_known_constraints(row["raw_outputs"], merged,
                                          **physical_config["constraint_tolerances"])
    ac = audit_ac_snapshot(row["raw_outputs"], merged, **physical_config["ac_audit_tolerances"])
    isolation = verify_isolation(row["raw_outputs"], merged)
    require(row["known_constraints"] == constraints and row["ac_audit"] == ac
            and row["isolation"] == isolation, "Raw result fails saved-checker replay")
    sections = ("numerical", "topology", "voltage", "line_thermal", "transformer_thermal", "gfm_active_limit")
    reliable = (row["raw_outputs"]["converged"] is True and ac["status"] == isolation["status"] == "PASS"
                and constraints["numerical"]["status"] == constraints["topology"]["status"] == "PASS"
                and not any(constraints[k]["status"] == "UNKNOWN" or constraints[k].get("unknown") for k in sections))
    status = ("PASS" if constraints["known_constraints_satisfied"] else "FAIL") if reliable else "UNKNOWN"
    require(type(row["reliable"]) is bool and row["reliable"] == reliable and row["status"] == status,
            "Convergence, reliability and physical feasibility were conflated")
    feedback = row["structured_feedback"]
    close(feedback["time_minutes"], expected_time, "Feedback time")
    require(feedback["usable_for_repair"] is reliable, "Unknown iterates used as physical feedback")
    expected_violations = [(section, v) for section in ("voltage", "line_thermal", "transformer_thermal")
                           for v in constraints[section]["violations"]] if reliable else []
    require([(v["constraint"], v["original"]) for v in feedback["violations"]] == expected_violations,
            "Feedback omits or invents physical violations")
    for violation in feedback["violations"]:
        close(violation["time_minutes"], expected_time, "Violation time")
        if violation["constraint"] == "voltage" and "lower" in violation:
            close(violation["magnitude"], max(violation["lower"] - violation["value"],
                  violation["value"] - violation["upper"], 0.), "Voltage excess")
        elif violation["constraint"] != "voltage":
            close(violation["magnitude"], max(0., violation["value"] - violation["limit"]), "Thermal excess")
    require(row["full_hardware_feasibility"] == "NOT_ESTABLISHED", "Unsupported hardware safety claim")
    return {"status": status, "normal_kw": normal_kw, "dynamic_kw": dynamic_kw,
            "extra_kw": extra_kw, "violations": len(expected_violations)}


def check_commands(cell, points, command_inventory):
    keys = ("python_solve_commands", "script_solve_commands", "text_solve_commands", "compile_commands",
            "explicit_solve_commands", "calcvoltagebases_entries", "numerical_solution_entries")
    totals = Counter({key: 0 for key in keys})
    per_point = defaultdict(Counter)
    open_call, started, returned, records = None, 0, 0, 0
    for row in read_lines(cell / "command_ledger.jsonl"):
        records += 1
        if row["event"] == "call_started":
            require(open_call is None and row["call_id"] == started, "Overlapping/unordered physical call IDs")
            open_call = row
            started += 1
            counts = Counter({key: 0 for key in keys})
            command = row["args"][0] if row["method"] == "Text.Command" and row["args"] else ""
            token = command.lstrip().split(None, 1)[0].lower() if isinstance(command, str) and command.strip() else ""
            if row["method"] == "Solution.Solve":
                counts["python_solve_commands"] = 1
            elif token == "compile":
                counts["compile_commands"] = counts["script_solve_commands"] = counts["calcvoltagebases_entries"] = 1
            elif token == "solve":
                counts["text_solve_commands"] = 1
            elif token == "calcvoltagebases":
                counts["calcvoltagebases_entries"] = 1
            require(token != "redirect", "Uncounted external command include")
            counts["explicit_solve_commands"] = sum(counts[k] for k in ("python_solve_commands", "script_solve_commands", "text_solve_commands"))
            counts["numerical_solution_entries"] = counts["explicit_solve_commands"] + counts["calcvoltagebases_entries"]
            require(dict(counts) == row["requested_counts"], "Recorded command-entry plan differs")
        else:
            require(row["event"] == "call_returned" and open_call is not None
                    and row["call_id"] == open_call["call_id"], "Physical call failed or lacks matching return")
            require(all(row[key] == open_call[key] for key in ("method", "args", "kwargs", "context", "requested_counts")),
                    "Physical call identity changed during execution")
            require(row["underlying_api_invoked"] is True and row["status"] == "RETURNED", "Call not actually invoked")
            if row["requested_counts"]["compile_commands"]:
                inventory = row["compile_inventory"]
                require(inventory["files"] == {name: item["sha256"] for name, item in command_inventory["files"].items()}
                        and inventory["explicit_solve_commands"] == inventory["calcvoltagebases_entries"] == 1,
                        "Actual compile source inventory differs")
            totals.update(row["requested_counts"])
            phase = row["context"]["phase"]
            require(phase in ("initialization", "snapshot"), "Unassigned physical call cost")
            point_id = row["context"].get("point_id", -1)
            if phase == "snapshot":
                require(point_id in points and row["context"]["identity_sha256"] == points[point_id]["identity_sha256"],
                        "Physical command attributed to wrong point")
            per_point[point_id].update(row["requested_counts"])
            open_call = None
            returned += 1
    require(open_call is None and started == returned, "Unclosed physical calls")
    stats = read(cell / "command_stats.json")
    require(stats["requested"] == stats["returned"] == dict(totals)
            and stats["calls_started"] == stats["calls_returned"] == started
            and stats["calls_failed"] == stats["calls_rejected"] == 0 and stats["open_call_ids"] == []
            and stats["ledger_records"] == records, "Durable command totals differ")
    require(all(v == 0 for name in ("unknown_completion", "not_invoked") for v in stats[name].values()),
            "Unknown or uninvoked numerical entries")
    require(per_point[-1]["numerical_solution_entries"] == 3, "Initialization cost omitted")
    for point_id, row in points.items():
        require(dict(per_point[point_id]) == row["actual_call_counts"]
                and per_point[point_id]["numerical_solution_entries"] == 4, "Per-point cost differs")
    require(totals["numerical_solution_entries"] == 3 + 4 * len(points), "Total numerical cost differs")
    return {"numerical_entries": totals["numerical_solution_entries"], "closed_calls": returned,
            "ledger_records": records}


def check_requests(cell, points, budget):
    rows = list(read_lines(cell / "requests.jsonl"))
    known = {}
    pending = None
    requests = hits = denied = 0
    for row in rows:
        if pending is not None:
            require(row["request"] == pending["request"] and row["key"] == pending["key"]
                    and row["point_id"] == pending["point_id"] and row["cache_hit"] is False
                    and row["status"] in ("PASS", "FAIL"), "Unclosed snapshot request")
            point = points[row["point_id"]]
            require(point["status"] == row["status"] and point["identity_sha256"] == row["key"], "Snapshot reply differs")
            known[row["key"]] = row["point_id"]
            pending = None
            continue
        requests += 1
        require(row["request"] == requests, "Logical requests out of order")
        if row["cache_hit"]:
            require(row["key"] in known and known[row["key"]] == row["point_id"], "Cross-method or future cache answer")
            hits += 1
        elif row.get("status") == "BUDGET_UNKNOWN":
            require(len(known) == budget and row["key"] not in known, "Premature budget rejection")
            denied += 1
        else:
            require(row["status"] == "STARTED" and row["key"] not in known
                    and row["point_id"] == len(known) and len(known) < budget, "Duplicate/overbudget cold request")
            pending = row
    require(pending is None and len(known) == len(points), "Unclosed/missing logical physical request")
    return {"requests": requests, "cache_hits": hits, "denied": denied, "new_points": len(known)}


def check_cell(run, cell, spec, cfg):
    require(read(cell / "spec.json") == spec and not (cell / "error.json").exists(), "Cell failure or identity mismatch")
    manifest = read(cell / "completion_manifest.json")
    for name, digest in manifest.items():
        require(sha(cell / name) == digest, "Saved cell artifact changed: " + name)
    scenarios = {s["scenario_id"]: s for s in read(run / "selected_scenarios.json")}
    events = {e["event_id"]: e for e in read(run / "selected_events.json")}
    scenario, event = scenarios[spec["scenario_id"]], events[spec["event_id"]]
    static, loads = read(run / "author_static_metadata.json"), read(run / "load_reference.json")
    condition, reference = read(run / "condition.json"), read(run / "operating_reference.json")
    initial = independent_initial(static, loads, scenario)
    require(read(cell / "initial_state.json") == initial, "Initial source component/history differs")
    metadata, physical_config = read(cell / "metadata.json"), read(run / "physical_config.json")
    require(metadata == read(run / "source_inputs/metadata.json"), "Initial physical metadata differs from source")
    points, point_counts = {}, Counter()
    for row in read_lines(cell / "points.jsonl.gz"):
        point_id = row["point_id"]
        require(point_id == len(points) and len(points) < spec["point_budget"], "Physical point sequence/budget differs")
        state = row["dynamic_state"]
        projection = independent_projection(static, loads, scenario, state["command"])
        require(type(state["step"]) is int and 0 <= state["step"] <= cfg["horizon"]
                and len(state["pickups"]) == 68, "Raw point history dimensions")
        for name, startup, original in zip(sorted(loads), state["pickups"], initial["pickups"]):
            require((startup == -1) == (original == -1) and (startup is not None) == (name in projection["energized"]),
                    "Raw point forged steady or disconnected history")
            if startup is not None and startup != -1:
                require(type(startup) in (int, float) and math.isfinite(startup) and startup >= 0
                        and state["step"] > 0 and startup <= (state["step"] - 1) * cfg["slot_seconds"]
                        and startup % cfg["slot_seconds"] == 0, "Off-grid or future first pickup")
        identity = row["state_identity"]
        t = identity["time_seconds"]
        require(t % cfg["sample_seconds"] == 0 and 0 <= t <= cfg["horizon"] * cfg["slot_seconds"], "Off-grid physical sample")
        factors = independent_multipliers(state, t, loads, event)
        expected_identity = {"model_version": "nc34-clpu-ed-snapshot-v1", "scenario_id": scenario["scenario_id"],
                             "condition_id": condition["condition_id"], "event_id": event["event_id"],
                             "a": event["a"], "tau_seconds": event["tau_seconds"], "command": state["command"],
                             "pickups": state["pickups"], "load_names": sorted(loads), "time_seconds": float(t),
                             "multipliers": factors, "alpha": .5, "base_condition_sha256": sha(run / "condition.json")}
        require(identity == expected_identity and row["identity_sha256"] == hashlib.sha256(canonical(identity).encode()).hexdigest(),
                "Cache identity omits or changes dynamic physical inputs")
        require(row["case"] == scenario["scenario_id"] and row["fault_lines"] == scenario["fault_lines"]
                and row["alpha"] == .5 and row["algorithm"] == 0, "Physical context or numerical algorithm changed")
        expected_enabled = {}
        for name, layout in metadata["element_layout"].items():
            buses = [b.split(".")[0].lower() for b in layout["buses"]]
            enabled = all(b in projection["powered"] for b in buses)
            if layout["kind"] in ("line", "transformer"):
                endpoints = list(dict.fromkeys(buses))
                enabled = enabled and len(endpoints) == 2 and projection["graph"].has_edge(*endpoints)
            if layout["kind"] == "load":
                enabled = name in projection["energized"]
            expected_enabled[name] = bool(enabled)
        require(row["metadata_override"]["planned_enabled"] == expected_enabled,
                "Whole-device isolation differs from graph projection")
        checked = check_saved_point(row, condition, reference, metadata, physical_config,
                                    factors, projection["action"], t / 60., projection["energized"])
        require(checked["status"] != "UNKNOWN", "Unreliable physical point in completed experiment")
        point_counts[checked["status"]] += 1
        # Keep only fields needed by the later path/cache/entry replay.
        points[point_id] = {key: row[key] for key in ("status", "reliable", "clpu_multipliers", "action", "time_minutes",
                          "state_identity", "identity_sha256", "actual_call_counts", "dynamic_state", "elapsed_seconds",
                          "structured_feedback")}
    require(points and len({p["identity_sha256"] for p in points.values()}) == len(points), "Empty or duplicate physical population")
    commands = check_commands(cell, points, read(run / "command_inventory.json"))
    requests = check_requests(cell, points, spec["point_budget"])
    oracle, result = read(cell / "oracle_summary.json"), read(cell / "result.json")
    require(oracle["status"] == "COMPLETE_COUNTERS_PENDING_AUDIT" and oracle["points"] == len(points)
            and oracle["point_budget"] == spec["point_budget"] and oracle["requests"] == requests["requests"]
            and oracle["cache_hits"] == requests["cache_hits"] and oracle["statuses"] == dict(point_counts)
            and oracle["actual_numerical_entries"] == commands["numerical_entries"]
            and oracle["maximum_numerical_entries"] == 3 + 4 * spec["point_budget"], "Oracle accounting differs")
    require(oracle["point_seconds"] == [p["elapsed_seconds"] for p in points.values()], "Physical timing ledger differs")
    require(result["spec"] == spec and result["oracle"] == oracle and result["training_interactions"] == 0
            and result["formal_holdout_numerical_access"] is False and result["tune_numerical_access"] is False,
            "Wrong result scope or physical accounting")
    plan_counts = Counter()
    if cfg["phase"] == "calibration":
        require(result["status"] == "CALIBRATION_COMPLETE" and len(result["samples"]) == 4, "Incomplete calibration")
        first = result["first_nominal_action"]
        advanced = independent_advance(initial, first, static, loads, scenario, cfg["slot_seconds"])
        waited = independent_advance(initial, 0, static, loads, scenario, cfg["slot_seconds"])
        expected_samples = [(initial, 0), (waited, 30), (advanced, 0), (advanced, 30)]
        for saved, (state, t) in zip(result["samples"], expected_samples):
            require(saved["state"] == state and saved["time_seconds"] == t, "Calibration state/time changed")
            point = points[saved["point_id"]]
            require(point["status"] == saved["status"] and saved["reliable"] is True, "Calibration evidence mismatch")
            require(point["state_identity"]["command"] == state["command"]
                    and point["state_identity"]["pickups"] == state["pickups"], "Calibration cache history mismatch")
            close(point["time_minutes"], t / 60., "Calibration point time")
    elif result["status"] == "INITIAL_FAIL":
        cert = result["initial_certificate"]
        require(len(points) == 1 and cert["point_id"] == 0 and points[0]["status"] == cert["status"] == "FAIL",
                "Initial failure omitted or mismatched")
        require(points[0]["dynamic_state"] == initial and points[0]["time_minutes"] == 0., "Wrong initial failure")
    else:
        require(result["status"] in ("CERTIFIED_INCUMBENT", "NO_CERTIFIED_COMPLETE_PLAN"), "Unfinished search cell")
        require(points[0]["dynamic_state"] == initial and points[0]["status"] == "PASS"
                and points[0]["time_minutes"] == 0., "Missing passing initial certificate")
        attempts = list(read_lines(cell / "plans.jsonl.gz"))
        require(attempts == result["attempts"] and result["checked_candidates"] == len(attempts), "Search attempt omission")
        require(attempts[0]["origin"] == "wait_incumbent" and attempts[0]["actions"] == [0] * cfg["horizon"],
                "No complete-window WAIT fallback attempted first")
        complete = []
        for index, attempt in enumerate(attempts):
            if attempt["status"] == "BUDGET_EXHAUSTED_PARTIAL":
                require(index == len(attempts) - 1 and attempt["certified_complete"] is False
                        and requests["denied"] == 1 and result["stop_reason"] == "PHYSICAL_BUDGET_EXHAUSTED",
                        "Budget partial misrepresented as accepted path")
                plan_counts["BUDGET_EXHAUSTED_PARTIAL"] += 1
                continue
            validation = attempt["validation"]
            require(attempt["actions"] == validation["actions"] and attempt["status"] == validation["status"]
                    and attempt["certified_complete"] == (validation["status"] == "COMPLETE"), "Attempt/validation mismatch")
            checked = check_saved_plan(validation, initial, static, loads, scenario, condition, event,
                                      cfg["horizon"], cfg["slot_seconds"], cfg["sample_seconds"], points)
            plan_counts[checked["status"]] += 1
            if checked["status"] == "COMPLETE":
                complete.append(attempt)
        selected = min(complete, key=lambda p: (-p["validation"]["energy_kwh"], -p["validation"]["final_fraction"],
                                                sum(a != 0 for a in p["actions"]), tuple(p["actions"]))) if complete else None
        require(result["incumbent"] == selected and (selected is not None) == (result["status"] == "CERTIFIED_INCUMBENT"),
                "Returned plan is not the best fully certified observed candidate")
        require(result["new_training_steps"] == 0 and result["optimality_established"] is False,
                "Unsupported training or optimality claim")
    return {"episode_id": spec["episode_id"], "status": result["status"], "points": len(points),
            "physical_statuses": dict(point_counts), "commands": commands, "requests": requests,
            "plan_statuses": dict(plan_counts), "cell_completion_sha256": sha(cell / "completion_manifest.json")}


def audit(run):
    run = run.resolve()
    require(run.parent == (ROOT / "runs").resolve(), "Source must be an existing project run")
    output = ROOT / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_nature34_clpu_pilot_audit")
    output.mkdir()
    started = time.perf_counter()
    result = {"status": "FAILED_CLPU_PILOT_AUDIT", "source_run": run.name, "errors": [],
              "new_powerflow_calls": 0, "new_training_interactions": 0,
              "formal_holdout_numerical_access": False, "tune_numerical_access": False,
              "scope": "Saved raw three-phase equations/constraints, dynamic load history, full candidate paths, provenance and actual numerical costs; no new solver call or optimality claim"}
    cells = []
    try:
        source_completion_sha256 = sha(run / "completion_manifest.json")
        source_manifest = read(run / "completion_manifest.json")
        for name, expected in source_manifest.items():
            require(sha(run / name) == expected, "Completed source artifact changed: " + name)
        result["source_completion_sha256"] = source_completion_sha256
        summary, schedule, cfg = read(run / "summary.json"), read(run / "schedule.json"), read(run / "config.json")
        require(summary["status"] == "COMPLETE_PENDING_INDEPENDENT_AUDIT"
                and summary["completed"] == summary["scheduled"] == len(schedule)
                and all(w["exit_code"] == 0 for w in summary["workers"]), "Incomplete or stopped source cannot pass audit")
        require(sorted(w["episode_id"] for w in summary["workers"]) == list(range(len(schedule))), "Missing/duplicate worker completion")
        require(summary["training_interactions"] == 0 and summary["formal_holdout_numerical_access"] is False
                and summary["tune_numerical_access"] is False, "Source reports unsupported scope")
        prepared = read(run / "prepared.json")
        require(sha(run / "prephysical_freeze_manifest.json") == prepared["freeze_sha256"], "Prephysical freeze hash changed")
        for name, expected in read(run / "prephysical_freeze_manifest.json").items():
            require(sha(run / name) == expected, "Frozen input changed after execution: " + name)
        result["inputs"] = check_inputs(run)
        scenarios, events = read(run / "selected_scenarios.json"), read(run / "selected_events.json")
        if cfg["phase"] == "calibration":
            expected_schedule = [{"scenario_id": scenarios[0]["scenario_id"], "event_id": events[0]["event_id"],
                                  "method": "calibration", "point_budget": 4}]
        else:
            require(cfg["phase"] == "main", "Unknown experiment phase")
            expected_schedule = []
            for scenario in scenarios:
                for event in events:
                    for budget in cfg["point_budgets"]:
                        for method in cfg["methods"]:
                            expected_schedule.append({"scenario_id": scenario["scenario_id"], "event_id": event["event_id"],
                                                      "method": method, "point_budget": budget})
                    expected_schedule.append({"scenario_id": scenario["scenario_id"], "event_id": event["event_id"],
                                              "method": "reference", "point_budget": cfg["reference_point_budget"]})
            expected_schedule.sort(key=lambda row: hashlib.sha256(("clpu-run-order-v1|" + canonical(row)).encode()).hexdigest())
        expected_schedule = [{**spec, "episode_id": index} for index, spec in enumerate(expected_schedule)]
        require(schedule == expected_schedule, "Predetermined case/method/budget population changed")
        resources = read(run / "resource_budget.json")
        require(resources["episodes"] == len(schedule)
                and resources["max_physical_points"] == sum(s["point_budget"] for s in schedule)
                and resources["max_numerical_entries"] == sum(3 + 4 * s["point_budget"] for s in schedule),
                "Total frozen physical resource ledger differs")
        for spec in schedule:
            cell = run / "episodes" / f"{spec['episode_id']:04d}"
            cells.append(check_cell(run, cell, spec, cfg))
            save(output / "progress.json", {"completed": len(cells), "scheduled": len(schedule),
                                           "elapsed_seconds": time.perf_counter() - started})
        require("opendssdirect" not in sys.modules and "dss" not in sys.modules, "Audit imported physical runtime")
        require(sha(run / "completion_manifest.json") == source_completion_sha256,
                "Source completion identity changed during audit")
        result.update(status="VERIFIED_CLPU_PILOT_SAVED_RESULTS", phase=cfg["phase"], episodes=len(cells),
                      physical_points=sum(c["points"] for c in cells),
                      numerical_entries=sum(c["commands"]["numerical_entries"] for c in cells),
                      source_summary_sha256=sha(run / "summary.json"),
                      source_schedule_sha256=sha(run / "schedule.json"),
                      source_freeze_sha256=sha(run / "prephysical_freeze_manifest.json"))
    except BaseException as exc:
        result["errors"].append({"error": repr(exc), "traceback": traceback.format_exc()})
    result["elapsed_seconds"] = time.perf_counter() - started
    save(output / "cell_checks.json", cells)
    save(output / "verification.json", result)
    source_paths = [Path(__file__), ROOT / "src/dnr/restoration_ac_audit.py",
                    ROOT / "src/dnr/restoration_constraints.py", ROOT / "src/dnr/restoration_isolation.py"]
    save(output / "audit_source_hashes.json", {str(p.relative_to(ROOT)): sha(p) for p in source_paths})
    save(output / "completion_manifest.json", {str(p.relative_to(output)): sha(p) for p in output.rglob("*")
                                              if p.is_file() and p.name != "completion_manifest.json"})
    pointer = {**result, "run": output.name, "completion_sha256": sha(output / "completion_manifest.json")}
    save(ROOT / "artifacts/nature34_clpu_audit_latest.json", pointer)
    print(canonical({"run": str(output), "status": result["status"], "episodes": len(cells), "errors": result["errors"]}), flush=True)
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(audit(args.run))
