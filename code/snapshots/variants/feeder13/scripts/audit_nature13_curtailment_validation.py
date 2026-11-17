"""Audit saved D033 data only: never import an engine or execute power flow.

The v2 operating-point checks are replayed with the frozen pure checkers.
An optional legacy run is audited for provenance and accounting only; its old
numerical labels are preserved rather than reinterpreted with a newer checker.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))
SECTIONS = ("numerical", "topology", "voltage", "line_thermal",
            "transformer_thermal", "gfm_active_limit")
AC_SECTIONS = ("inventory", "kcl", "voltage_consistency", "power_identity",
               "linear_yprim", "disabled_elements")
PURE_SOURCES = ("src/dnr/restoration_constraints.py", "src/dnr/restoration_ac_audit.py")
AVAILABLE_KW = {"generator.g1": 20.0, "generator.g2": 20.0,
                "generator.g3": 1000.0, "generator.g4": 50.0,
                "generator.g5": 50.0, "generator.g6": 1000.0,
                "generator.g7": 500.0}
NAMEPLATES_KVA = {"sub": 5000, "xfm1": 500,
                  "reg1": 1666, "reg2": 1666, "reg3": 1666}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False), encoding="utf-8")


def run_path(value):
    path = (ROOT / "runs" / value).resolve()
    require(path.parent == (ROOT / "runs").resolve(), "Source must be a direct runs child")
    return path


def check_manifest(directory, expected=None):
    manifest_path = directory / "completion_manifest.json"
    if expected is not None:
        require(sha(manifest_path) == expected, "Completion manifest pointer hash differs")
    manifest = read(manifest_path)
    for name, digest in manifest.items():
        path = (directory / name).resolve()
        require(path.is_relative_to(directory.resolve()), "Manifest path escapes source run")
        require(path.is_file() and sha(path) == digest, "Manifest hash differs: " + name)
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*")
              if p.is_file() and p != manifest_path}
    require(actual == set(manifest), "Manifest does not cover exactly the saved files")
    for name, digest in read(directory / "source_hashes.json").items():
        require(sha(directory / "source" / name) == digest, "Frozen source hash differs: " + name)
    for name, digest in read(directory / "input_hashes.json").items():
        require(sha(directory / "work" / name) == digest, "Work input hash differs: " + name)
    return len(manifest)


def event_counts(events):
    python = sum(e["method"] == "Solution.Solve" for e in events)
    compile_count = sum(e["method"] == "Text.Command" and e.get("args") and
                        str(e["args"][0]).lower().startswith("compile") for e in events)
    return {"python_solve_calls": python, "compile_commands": compile_count,
            "explicit_solve_commands": python + compile_count}


def point_key(row):
    return row["case"], row["alpha"], row["algorithm"], row["rep"]


def archive_and_budget(directory, expected_manifest=None):
    """Version-independent checks; intentionally do not replay old AC labels."""
    manifest_count = check_manifest(directory, expected_manifest)
    config, metadata, summary = (read(directory / name) for name in
                                 ("config.json", "metadata.json", "summary.json"))
    with gzip.open(directory / "points.jsonl.gz", "rt", encoding="utf-8") as stream:
        points = [json.loads(line) for line in stream]
    attempts = [json.loads(line) for line in (directory / "attempts.jsonl").read_text(encoding="utf-8").splitlines()]
    events = read(directory / "engine_calls.json")
    require(len(points) == config["max_point_evaluations"] == 72, "Expected all 72 saved points")
    require(config["alpha_grid"] == [i / 8 for i in range(9)], "Alpha grid changed")
    require(config["algorithms"] == [0, 1] and config["repetitions"] == 2,
            "Expected Normal/Newton and two repetitions")
    cases = config["cases"]
    expected_keys = list(itertools.product([c["name"] for c in cases], config["alpha_grid"],
                                          config["algorithms"], range(2)))
    require([point_key(p) for p in points] == expected_keys, "Saved point order/coverage differs")
    require(len(attempts) == 2 * len(points), "Request log must contain one start and end per point")
    case_of = {c["name"]: c for c in cases}
    cursor = attempts[0]["engine_event_start"]
    initial = event_counts(events[:cursor])
    require(initial["python_solve_calls"] == 1 and initial["compile_commands"] == 1,
            "Initial engine accounting differs")
    for index, point in enumerate(points):
        require(point["id"] == index, "Point IDs must be contiguous")
        start, end = attempts[2 * index:2 * index + 2]
        require(start["event"] == "started" and end["event"] == "completed",
                "Nonclosed request log: " + str(index))
        for key in ("id", "case", "alpha", "algorithm", "rep", "action", "fault_lines"):
            require(start[key] == end[key] == point[key], "Request/output mismatch: " + str((index, key)))
        require(start["engine_event_start"] == end["engine_event_start"] == cursor,
                "Engine event intervals are not contiguous")
        next_cursor = end["engine_event_end"]
        require(cursor < next_cursor <= len(events), "Invalid engine event interval")
        counted = event_counts(events[cursor:next_cursor])
        require(counted["python_solve_calls"] == point["python_solve_calls"] and
                counted["compile_commands"] == point["compile_commands"] == 1,
                "Point engine accounting mismatch: " + str(index))
        case = case_of[point["case"]]
        require(point["action"] == case["action"] and point["fault_lines"] == case["fault_lines"],
                "Fixed scenario/action changed")
        require(point["domain_status"] == "GRID_CONNECTED_DOMAIN" and point["action"][2] == 0,
                "Point lies outside the frozen connected/tie1-open domain")
        require(point["raw_outputs"]["algorithm"] == point["algorithm"], "Algorithm label differs")
        edits = {}
        for event in events[cursor:next_cursor]:
            require("error" not in event, "Engine event contains a saved error")
            if event["method"] == "Text.Command" and event.get("args"):
                match = re.fullmatch(r"Edit\s+(Generator\.g\d+)\s+kW=([^ ]+)\s+kvar=([^ ]+)",
                                     str(event["args"][0]), flags=re.IGNORECASE)
                if match:
                    require(match[1].lower() not in edits, "Duplicate DER operating edit")
                    edits[match[1].lower()] = (float(match[2]), float(match[3]))
        require(set(edits) == set(AVAILABLE_KW), "All seven DER operating edits must be recorded")
        for name, available in AVAILABLE_KW.items():
            require(abs(edits[name][0] - point["alpha"] * available) < 1e-8 and
                    abs(edits[name][1] - point["alpha"] * available * 0.75) < 1e-8,
                    "Recorded edit changes the alpha operating rule")
        cursor = next_cursor
    require(cursor == len(events), "Unattributed engine calls after the last point")
    totals = event_counts(events)
    require(totals == {"python_solve_calls": 901, "compile_commands": 73,
                       "explicit_solve_commands": 974}, "Actual command budget differs")
    for key, value in totals.items():
        require(summary[key] == value, "Summary command count differs: " + key)
    require(config["expected_total_explicit_solve_commands"] == 974 and
            config["max_total_explicit_solve_commands"] == 974, "Frozen budget differs")
    require(summary["point_attempts"] == summary["point_evaluations"] == 72,
            "Summary request count differs")
    require(summary["rl_training_steps"] == config["rl_training_steps"] == 0 and
            summary["formal_holdout_access"] is config["formal_holdout_access"] is False,
            "Unexpected training/holdout access declaration")
    refs = {name.lower(): ref for name, ref in metadata["generator_reference"].items()}
    require(set(refs) == set(AVAILABLE_KW), "Generator reference inventory differs")
    for name, available in AVAILABLE_KW.items():
        require(refs[name]["available_p_kw"] == available and
                refs[name]["reference_q_kvar"] == 0.75 * available,
                "Original available DER capacity/reference PF changed: " + name)
    ratings = {name.lower().split(".", 1)[1]: value for name, value in metadata["transformer_ratings"].items()}
    require(set(ratings) == set(NAMEPLATES_KVA), "Transformer inventory differs")
    for name, value in NAMEPLATES_KVA.items():
        require(all(w["kva"] == value for w in ratings[name]["windings"]),
                "Original transformer capacity changed: " + name)
    for point in points:
        require(abs(point["requested_der_p_kw"] - point["alpha"] * 2640.0) < 1e-8,
                "Requested DER total differs")
        generators = {e["name"].lower(): e for e in point["raw_outputs"]["elements"]
                      if e["name"].lower().startswith("generator.")}
        require(set(generators) == set(AVAILABLE_KW), "Snapshot generator inventory differs")
        require(not any(e["name"].lower().startswith("vsource.bus_")
                        for e in point["raw_outputs"]["elements"]), "Unexpected virtual island source")
        for name, element in generators.items():
            target = point["alpha"] * AVAILABLE_KW[name]
            setting = element["operating_setpoint"]
            require(element["enabled"] is True and abs(setting["kw"] - target) < 1e-8 and
                    abs(setting["kvar"] - 0.75 * target) < 1e-8,
                    "Saved DER operating setpoint differs: " + name)
    # Check the copied physical definition against the immutable author copy.
    author_pointer = read(ROOT / "artifacts/restoration_shortlist_sources_latest.json")
    require(author_pointer["nature_commit"] == config["source_commit"], "Pinned author commit changed")
    author = ROOT / author_pointer["derived_directory"] / "nature_author_source"
    relative = Path("Environments/DSSdirect_13bus_loadandswitching")
    for name in ("IEEE13Nodeckt.dss", "DSS_Initialize.py"):
        require(sha(directory / "work" / relative / name) == sha(author / "RL_Training" / relative / name),
                "Working physical input differs from pinned author source: " + name)
    archival = {"run": directory.name, "manifest_entries_verified": manifest_count, **totals,
                "action_points": len(points), "request_log_entries": len(attempts),
                "converged_count": sum(p["raw_outputs"]["converged"] is True for p in points),
                "saved_ac_status_counts": dict(Counter(p["ac_audit"]["status"] for p in points)),
                "warnings_retained": sum(len(p["warnings"]) for p in points),
                "scope": "ARCHIVAL_INPUT_AND_BUDGET_ONLY; original numerical labels are not reclassified"}
    return config, metadata, summary, points, archival


def algorithm_difference(left, right, limits):
    """Independent array comparison; do not import runner comparison logic."""
    if left["converged"] is not True or right["converged"] is not True:
        return {"status": "UNKNOWN", "reason": "At least one algorithm did not converge"}
    require(left["node_names"] == right["node_names"], "Algorithm node inventories differ")
    require(len(left["node_vmag_pu"]) == len(right["node_vmag_pu"]), "Voltage dimensions differ")
    voltage = max(abs(a - b) for a, b in zip(left["node_vmag_pu"], right["node_vmag_pu"]))
    other = {e["name"].lower(): e for e in right["elements"]}
    require(set(other) == {e["name"].lower() for e in left["elements"]}, "Algorithm element inventories differ")
    current, power = 0.0, 0.0
    for element in left["elements"]:
        counterpart = other[element["name"].lower()]
        for key in ("currents_real_imag", "powers_kw_kvar"):
            require(len(element[key]) == len(counterpart[key]), "Algorithm terminal dimensions differ")
        delta = [a - b for a, b in zip(element["currents_real_imag"], counterpart["currents_real_imag"])]
        current = max(current, *(math.hypot(delta[i], delta[i + 1]) for i in range(0, len(delta), 2)))
        power = max(power, *(abs(a - b) for a, b in zip(element["powers_kw_kvar"], counterpart["powers_kw_kvar"])))
    finite = all(math.isfinite(value) for value in (voltage, current, power))
    passed = finite and voltage <= limits["voltage_absolute_pu"] and current <= limits["current_absolute_A"] and power <= limits["power_absolute_kw"]
    return {"status": "PASS" if passed else "FAIL" if finite else "UNKNOWN",
            "max_node_voltage_diff_pu": voltage, "max_complex_current_diff_A": current,
            "max_terminal_power_component_diff_kw_kvar": power}


def selected_metrics(point, metadata, checked):
    raw = point["raw_outputs"]
    nodes = list(zip(raw["node_names"], raw["node_vmag_pu"]))
    low, high = min(nodes, key=lambda pair: pair[1]), max(nodes, key=lambda pair: pair[1])
    line = max(checked["line_thermal"]["checks"], key=lambda c: c["current_a"] / c["limit_a"])
    elements = {e["name"].lower(): e for e in raw["elements"]}
    element = elements["transformer.xfm1"]
    rating = next(value for name, value in metadata["transformer_ratings"].items()
                  if name.lower() == "transformer.xfm1")
    winding_metrics = []
    ncond = element["conductors"]
    for terminal, winding in enumerate(rating["windings"]):
        phase = []
        offset = terminal * ncond
        powers = element["powers_kw_kvar"][2 * offset:2 * (offset + ncond)]
        limit_a = winding["kva"] / (math.sqrt(3) * winding["kv"])
        for c in range(ncond):
            index = offset + c
            node = element["node_order"][index]
            if node not in (1, 2, 3):
                continue
            current = math.hypot(*element["currents_real_imag"][2 * index:2 * index + 2])
            apparent = math.hypot(*element["powers_kw_kvar"][2 * index:2 * index + 2])
            phase.append({"node": node, "current_a": current, "phase_s_kva": apparent,
                          "current_ratio": current / limit_a,
                          "phase_s_ratio": apparent / (winding["kva"] / 3)})
        winding_metrics.append({"winding": terminal + 1, "bus": element["buses"][terminal],
                                "total_s_kva": math.hypot(math.fsum(powers[::2]), math.fsum(powers[1::2])),
                                "nameplate_kva": winding["kva"], "phases": phase})
    def actual(kind, sign):
        return {name: sign * math.fsum(e["powers_kw_kvar"][:2 * e["conductors"]:2])
                for name, e in elements.items() if name.startswith(kind + ".") and e["enabled"]}
    loads, generators = actual("load", 1), actual("generator", -1)
    return {"case": point["case"], "point_id": point["id"], "alpha": point["alpha"],
            "voltage_min": {"node": low[0], "pu": low[1]},
            "voltage_max": {"node": high[0], "pu": high[1]},
            "maximum_line_current_utilization": {**line, "ratio": line["current_a"] / line["limit_a"]},
            "xfm1_windings": winding_metrics, "actual_load_kw_by_element": loads,
            "actual_load_kw": math.fsum(loads.values()), "actual_der_kw_by_element": generators,
            "actual_der_kw": math.fsum(generators.values()), "requested_der_kw": point["requested_der_p_kw"],
            "available_der_kw_unchanged": 2640.0,
            "full_hardware_feasibility": "NOT_ESTABLISHED"}


def replay_v2(directory, config, metadata, summary, points, output):
    for name in PURE_SOURCES:
        require(sha(ROOT / name) == sha(directory / "source" / name),
                "Pure checker differs from this run's frozen version: " + name)
    require(config.get("solver_convergence_tolerance") == 1e-8,
            "Full replay requires v2; use --budget-only for legacy data")
    from dnr.restoration_constraints import check_known_constraints
    from dnr.restoration_ac_audit import audit_ac_snapshot
    require("opendssdirect" not in sys.modules and "dss" not in sys.modules,
            "Saved-data audit unexpectedly imported an engine")
    recomputed, errors = {}, []
    status_counts = {key: Counter() for key in SECTIONS + AC_SECTIONS}
    maximum_residuals = {key: 0.0 for key in AC_SECTIONS}
    maximum_ledger = {"p_kw": 0.0, "q_kvar": 0.0}
    with gzip.open(output / "point_checks.jsonl.gz", "wt", encoding="utf-8") as stream:
        for point in points:
            known = check_known_constraints(point["raw_outputs"], metadata, **config["constraint_tolerances"])
            audit = audit_ac_snapshot(point["raw_outputs"], metadata, **config["ac_audit_tolerances"])
            mismatches = [key for key, value in (("known_constraints", known), ("ac_audit", audit))
                          if point[key] != value]
            if mismatches:
                errors.append({"point_id": point["id"], "replayed_label_mismatches": mismatches})
            recomputed[point["id"]] = (known, audit)
            for key in SECTIONS:
                status_counts[key][known[key]["status"]] += 1
            for key in AC_SECTIONS:
                status_counts[key][audit[key]["status"]] += 1
                value = audit[key].get("max_residual")
                if value is not None:
                    maximum_residuals[key] = max(maximum_residuals[key], value)
            ledger = known["numerical"]["power_ledger"]
            for short, key in (("p_kw", "residual_p_kw"), ("q_kvar", "residual_q_kvar")):
                if ledger[key] is not None:
                    maximum_ledger[short] = max(maximum_ledger[short], abs(ledger[key]))
            stream.write(json.dumps({"id": point["id"], "case": point["case"], "alpha": point["alpha"],
                                     "algorithm": point["algorithm"], "rep": point["rep"],
                                     "label_mismatches": mismatches, "known_constraints": known,
                                     "ac_audit": audit}, allow_nan=False) + "\n")
    groups = defaultdict(dict)
    for point in points:
        groups[point["case"], point["alpha"]][point["algorithm"], point["rep"]] = point
    comparisons, selections, metrics = [], [], []
    for case in config["cases"]:
        accepted = []
        for alpha in config["alpha_grid"]:
            group = groups[case["name"], alpha]
            consistent = True
            for algorithm in config["algorithms"]:
                equal = group[algorithm, 0]["raw_outputs"] == group[algorithm, 1]["raw_outputs"]
                comparisons.append({"case": case["name"], "alpha": alpha,
                                    "algorithm": algorithm, "repeat_exact": equal})
                consistent = consistent and equal
            difference = algorithm_difference(group[0, 0]["raw_outputs"], group[1, 0]["raw_outputs"],
                                              config["cross_algorithm_tolerances"])
            comparisons.append({"case": case["name"], "alpha": alpha, "cross_algorithm": difference})
            consistent = consistent and difference["status"] == "PASS"
            if consistent and all(recomputed[p["id"]][0]["known_constraints_satisfied"] is True and
                                  recomputed[p["id"]][1]["status"] == "PASS" for p in group.values()):
                accepted.append(alpha)
        choice = max(accepted) if accepted else None
        selections.append({"case": case["name"], "valid_alphas": accepted, "selected_alpha": choice,
                           "status": "VERIFIED_GRID_WITNESS" if accepted else "NO_VERIFIED_GRID_WITNESS"})
        if choice is not None:
            point = groups[case["name"], choice][0, 0]
            metrics.append(selected_metrics(point, metadata, recomputed[point["id"]][0]))
    if comparisons != read(directory / "comparisons.json"):
        errors.append({"independent_comparisons_differ": True})
    if selections != read(directory / "selections.json") or selections != summary["selections"]:
        errors.append({"independent_selections_differ": True})
    ac_counts = dict(Counter(value[1]["status"] for value in recomputed.values()))
    known_counts = dict(Counter(str(value[0]["known_constraints_satisfied"]) for value in recomputed.values()))
    require(ac_counts == summary["ac_status_counts"] and known_counts == summary["known_constraints_counts"],
            "Summary classification counts differ")
    require(sum(p["raw_outputs"]["converged"] is True for p in points) == summary["converged_count"],
            "Summary convergence count differs")
    save(output / "comparisons.json", comparisons)
    save(output / "selections.json", selections)
    save(output / "selected_case_metrics.json", metrics)
    return {"errors": errors, "ac_status_counts": ac_counts,
            "known_constraints_counts": known_counts,
            "section_status_counts": {key: dict(value) for key, value in status_counts.items()},
            "maximum_ac_residuals": maximum_residuals,
            "maximum_terminal_ledger_residuals": maximum_ledger,
            "selections": selections, "selected_case_metrics": metrics,
            "cross_algorithm_comparison_count": sum("cross_algorithm" in row for row in comparisons),
            "exact_repeat_comparison_count": sum("repeat_exact" in row for row in comparisons),
            "all_repeats_exact": all(row["repeat_exact"] for row in comparisons if "repeat_exact" in row),
            "cross_algorithm_status_counts": dict(Counter(row["cross_algorithm"]["status"] for row in comparisons
                                                          if "cross_algorithm" in row))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", help="Run ID; omitted means the current completed latest pointer")
    parser.add_argument("--legacy-run", help="Optional v1 run ID, archival/accounting checks only")
    parser.add_argument("--budget-only", action="store_true", help="Do not replay or relabel AC/constraint checks")
    args = parser.parse_args()
    pointer = read(ROOT / "artifacts/nature13_curtailment_latest.json")
    source = run_path(args.source_run or pointer["run"])
    expected = pointer["completion_sha256"] if source.name == pointer["run"] else None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_nature13_curtailment_saved_audit"
    output = ROOT / "runs" / stamp
    output.mkdir(parents=True)
    result = {"run": stamp, "source_run": source.name, "status": "FAILED_SAVED_AUDIT",
              "new_powerflow_calls": 0, "rl_training_steps": 0, "formal_holdout_access": False,
              "full_hardware_feasibility": "NOT_ESTABLISHED"}
    exit_code = 1
    try:
        config, metadata, summary, points, archival = archive_and_budget(source, expected)
        save(output / "source_archival_checks.json", archival)
        result.update(source_completion_sha256=sha(source / "completion_manifest.json"),
                      source_archival_checks=archival)
        for name in ("scripts/audit_nature13_curtailment_validation.py",) + PURE_SOURCES:
            destination = output / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((ROOT / name).read_bytes())
        save(output / "audit_source_hashes.json", {name: sha(ROOT / name) for name in
                                                   ("scripts/audit_nature13_curtailment_validation.py",) + PURE_SOURCES})
        if args.legacy_run:
            legacy = run_path(args.legacy_run)
            require(legacy != source, "Legacy and main source runs must differ")
            legacy_archival = archive_and_budget(legacy)[4]
            save(output / "legacy_archival_checks.json", legacy_archival)
            result["legacy_archival_checks"] = legacy_archival
            result["combined_explicit_solve_commands"] = archival["explicit_solve_commands"] + legacy_archival["explicit_solve_commands"]
        if args.budget_only:
            result["status"] = "VERIFIED_ARCHIVAL_AND_BUDGET_ONLY"
            exit_code = 0
        else:
            replay = replay_v2(source, config, metadata, summary, points, output)
            result.update(replay)
            if not replay["errors"]:
                result["status"] = "VERIFIED_BOUNDED_SAVED_RESULTS"
                exit_code = 0
    except Exception as exc:
        result.update(error=repr(exc), traceback=traceback.format_exc())
    finally:
        save(output / "verification.json", result)
        save(output / "completion_manifest.json", {str(p.relative_to(output)): sha(p)
                                                   for p in output.rglob("*") if p.is_file()})
        save(ROOT / "artifacts/nature13_curtailment_saved_audit_latest.json",
             {"run": stamp, "source_run": source.name, "status": result["status"],
              "completion_sha256": sha(output / "completion_manifest.json")})
        print(json.dumps({"run": stamp, "status": result["status"], "source_run": source.name,
                          "error": result.get("error"), "new_powerflow_calls": 0}, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
