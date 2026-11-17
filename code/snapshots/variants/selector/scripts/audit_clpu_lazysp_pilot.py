"""Independent zero-solver replay of the bounded P/LazySP development pilot.

Production LazySP, its runner, and its selector are never imported here.
The existing independent physical, command, shared-ledger, graph, and staged
generation checkers are retained. New slot-edge choices are reconstructed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys
import traceback
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
sys.dont_write_bytecode = True
import audit_clpu_guard_first_pilot as gf

shared, staged, branch, old = gf.shared, gf.staged, gf.branch, gf.old
read, save, sha, digest, read_lines = gf.read, gf.save, gf.sha, gf.digest, gf.read_lines
require, same = gf.require, gf.same
SOURCE = ROOT / "runs/20261031T141705_478719Z_clpu_public_mechanism_pilot"
SOURCE_SHA = "d9fa4705c494bb2a56f2076ca524d6c2234e91c1658796bdd71b2a2db88c62b8"
STUDY = "CLPU_LAZYSP_COMPONENT_DEVELOPMENT_V1"
VERSION = "clpu-fixed-prefix-lazysp-forward-reverse-v1"
CONTEXTS, ARMS = (0, 2, 4, 7), ("P", "L_forward", "L_reverse")


def fixed_settings():
    return dict(study=STUDY, contexts=list(CONTEXTS), arms=list(ARMS), cells=12,
        cold_workers=1, worker_seconds=180, controller_seconds=2400, seed=0,
        logical_points_per_phase=256, physical_points_per_task=512, total_physical_cap=6144,
        max_selected_plans=16, training=0, formal_test=False,
        statistics="all4 descriptive equal-weight mean ratios; no CI", retry=False,
        missing="retain all identities; no full-population claims when incomplete")


def mechanics(run, spec):
    scenarios = {x["scenario_id"]: x for x in read(run / "selected_scenarios.json")}
    events = {x["event_id"]: x for x in read(run / "selected_events.json")}
    return branch.Mechanics(read(run / "author_static_metadata.json"), read(run / "load_reference.json"),
        scenarios[spec["scenario_id"]], read(run / "condition.json"), events[spec["event_id"]], read(run / "config.json"))


def expected_schedule():
    original = read(SOURCE / "schedule.json")
    result = []
    for turn in range(3):
        for rank, cid in enumerate(CONTEXTS):
            spec = deepcopy(next(x for x in original if x["context_id"] == cid and x["arm"] == "P"))
            arm = ARMS[(rank + turn) % 3]
            spec.update(cell=f"episodes/{len(result):03d}", episode_id=len(result), arm=arm,
                route_arm=arm, arm_slot=arm, group=arm, phase_order=["GUARD", "EARLY"], dispatch_round=turn,
                selector=None if arm == "P" else arm.removeprefix("L_"))
            if arm != "P":
                spec.update(macro_schedule="fixed_prefix_lazysp", window_variant="lazysp_global_fixed_pool_v1",
                            free_external_floor_pruning_enabled=False, external_floor_enabled=True)
            result.append(spec)
    return result


def check_frozen(run, *, require_tests=True):
    run = Path(run).resolve()
    require(run.parent == ROOT / "runs", "Project run required")
    require(sha(SOURCE / "completion_manifest.json") == SOURCE_SHA, "Source seal changed")
    original = {k.replace("\\", "/"): v for k, v in read(SOURCE / "completion_manifest.json").items()}
    prepared = read(run / "prepared.json")
    fresh = {"software_tests.xml", "resource_budget.json", "dependencies.json", "python_environment.json"}
    for name, h in prepared["input_hashes"].items():
        require(sha(run / name) == h, "Frozen input changed: " + name)
        if name not in fresh:
            require(name in original and sha(SOURCE / name) == original[name] == h,
                    "Scientific input is not original frozen development input: " + name)
    for name in ("dependencies.json", "python_environment.json"):
        require(read(run / name) == read(SOURCE / name), "Environment differs from prior physical environment")
    resource = read(run / "resource_budget.json")
    rate = read(SOURCE / "resource_budget.json")["historical_max_bytes_per_actual_point"]
    require(resource["source_budget_sha256"] == original["resource_budget.json"] == sha(SOURCE / "resource_budget.json")
        and resource["historical_max_bytes_per_actual_point"] == rate and resource["total_physical_cap"] == 6144
        and resource["forecast_bytes"] == math.ceil(rate * 6144 * 1.25) and resource["reserve_bytes"] == 14 * 2**30
        and resource["required_free_bytes"] == 14 * 2**30 + resource["forecast_bytes"]
        and resource["free_bytes_at_prepare"] >= resource["required_free_bytes"], "Capacity forecast differs")
    require({"config.json", "contexts.json", "selected_scenarios.json", "selected_events.json", "physical_scope.json",
             "stage_plan.json", "physical_config.json", "condition.json", "author_static_metadata.json", "load_reference.json"}
            <= set(prepared["input_hashes"]), "Incomplete input binding")
    source_inputs = {name for name in original if name.startswith("source_inputs/")}
    require(source_inputs and source_inputs <= set(prepared["input_hashes"]), "Original physical source_inputs missing")
    require(all(x["partition"] == "build" for x in read(run / "selected_scenarios.json")), "Non-development scenario")
    require(prepared["source_files"] == sorted(prepared["source_hashes"]), "Source list/hash mismatch")
    required_sources = {"scripts/audit_clpu_lazysp_pilot.py", "scripts/run_clpu_lazysp_pilot.py",
        "src/dnr/nature34_clpu_lazysp.py", "tests/test_nature34_clpu_lazysp.py", "tests/test_audit_clpu_lazysp_pilot.py",
        "tests/test_run_clpu_lazysp_pilot.py", "docs/clpu_lazysp_component_pilot_protocol_v1.md"}
    require(required_sources <= set(prepared["source_files"]), "Missing new source closure")
    for name, h in prepared["source_hashes"].items():
        require(sha(ROOT / name) == sha(run / "source" / name) == h, "Frozen source changed: " + name)
        if name.endswith('.py') and "source/" + name in original:
            require(h == original["source/" + name], "Inherited implementation changed: " + name)
    require(read(run / "settings.json") == fixed_settings() and sha(run / "settings.json") == prepared["settings_sha256"],
            "Frozen settings changed")
    require(read(run / "schedule.json") == expected_schedule() and sha(run / "schedule.json") == prepared["schedule_sha256"],
            "Frozen population/order changed")
    for s in expected_schedule():
        require(read(run / s["cell"] / "spec.json") == s, "Cell spec differs")
    require(prepared["physical_source_unchanged"] is True, "Changed physical source")
    if require_tests:
        cases = list(ET.parse(run / "software_tests.xml").getroot().iter("testcase"))
        require(cases and not any(list(c.iter("failure")) + list(c.iter("error")) + list(c.iter("skipped")) for c in cases),
                "All software checks must pass")
        require(any("test_nature34_clpu_lazysp" in c.get("classname", "") for c in cases), "Missing component checks")
        require(any("test_audit_clpu_lazysp" in c.get("classname", "") for c in cases), "Missing independent audit checks")
    return prepared


def preflight(run):
    prepared = check_frozen(run)
    require(shutil.disk_usage(ROOT).free >= 14 * 2**30, "Storage reserve insufficient")
    expected = {x["context_id"]: x["generation"] for x in read(run / "stage_plan.json")["contexts"]}
    for cid in CONTEXTS:
        spec = next(s for s in expected_schedule() if s["context_id"] == cid)
        staged.check_generation(expected[cid], staged.independently_generate(mechanics(run, spec)))
    result = {"status": "PASS", "prepared_sha256": sha(run / "prepared.json"),
        "created_utc": datetime.now(timezone.utc).isoformat(), "auditor_sha256": sha(Path(__file__)),
        "new_physical_points": 0, "new_training": 0, "candidate_pools_independently_checked": list(CONTEXTS)}
    require(not (run / "preflight.json").exists(), "Preflight output exists")
    save(run / "preflight.json", result)
    return result


def counters(oracle, evidence):
    return {"logical_points": oracle.used,
        "actual_physical_points": sum(x["reuse_kind"] == "ACTUAL_PHYSICAL" for x in evidence["logical_rows"][:oracle.used]),
        "requests": oracle.requests}


def independent_tree(replay):
    edges, candidates, index = [], [], {}
    for bid, node in sorted(replay.frontier.items()):
        info = replay.m.plan(node["info"]["actions"])
        ids = []
        for step, state in enumerate(info["states"][1:], 1):
            prefix = info["actions"][:step]
            if prefix not in index:
                eid = index[prefix] = len(edges)
                cost = (replay.m.total_kw - replay.m.service(state)) * replay.m.slot / 3600.
                require(math.isfinite(cost) and cost >= 0, "Invalid unserved-energy bound")
                edges.append({"edge_id": eid, "prefix": list(prefix), "state": state, "step": step,
                    "estimated_unserved_kwh": cost, "status": "UNKNOWN", "samples": [], "candidate_ids": []})
            eid = index[prefix]
            require(edges[eid]["state"] == state, "Same prefix has inconsistent full history")
            edges[eid]["candidate_ids"].append(bid)
            ids.append(eid)
        candidates.append({"branch_id": bid, "actions": list(info["actions"]), "energy_kwh": info["energy_kwh"],
            "edge_ids": ids, "status": "PENDING", "failed_edge_id": None})
    return edges, candidates


def check_lazy_records(mech, cfg, result, attempts, evidence, condition_sha, guard, reference, selector):
    require(selector in ("forward", "reverse"), "Unfrozen selector")
    oracle = branch.PaidRequests(evidence["request_rows"], evidence["logical"], mech, 256, condition_sha)
    bootstrap = staged.StagedReplay(mech, cfg, oracle, attempts, evidence["logical_rows"])
    require(bootstrap.stage_generation is not None, "Bootstrap lacks complete pool")
    staged.check_generation(result["stage_generation"], bootstrap.stage_generation)
    best = deepcopy(bootstrap.best)
    edges, candidates = independent_tree(bootstrap)
    selected, journal = [], []
    extra_attempts = []
    reason = "BOOTSTRAP_" + str(bootstrap.reason) if bootstrap.done else None
    while reason is None:
        floor = max(guard["best_plan"]["validation"]["energy_kwh"],
                    best["validation"]["energy_kwh"] if best else -math.inf)
        active = [c for c in candidates if c["status"] == "PENDING" and c["energy_kwh"] > floor]
        if not active:
            reason = "CERTIFIED_FIXED_POOL_OBJECTIVE_BOUND"
            break
        c = min(active, key=lambda x: (-x["energy_kwh"], x["branch_id"]))
        bid = c["branch_id"]
        if bid not in selected:
            if len(selected) == 16:
                reason = "DISTINCT_PLAN_SELECTION_CAP"
                break
            selected.append(bid)
            journal.append({"event": "PATH_SELECTED", "branch_id": bid,
                "optimistic_energy_kwh": c["energy_kwh"], "incumbent_energy_kwh": floor})
        unknown = [eid for eid in c["edge_ids"] if edges[eid]["status"] == "UNKNOWN"]
        if unknown:
            edge = edges[unknown[0] if selector == "forward" else unknown[-1]]
            event = {"event": "EDGE_SELECTED", "branch_id": bid, "edge_id": edge["edge_id"],
                "selector": selector, "before": counters(oracle, evidence)}
            for offset in range(0, mech.slot + 1, mech.sample):
                seconds = (edge["step"] - 1) * mech.slot + offset
                cert = oracle.certify(edge["state"], seconds)
                if cert is None:
                    event["status"] = "BUDGET_UNKNOWN"
                    reason = "PHYSICAL_BUDGET_EXHAUSTED"
                    break
                require(cert["reliable"] is True and cert["status"] in ("PASS", "FAIL"), "Unreliable slot sample")
                edge["samples"].append({"step": edge["step"], "time_seconds": seconds,
                    "point_id": cert["point_id"], "status": cert["status"], "state": edge["state"]})
                if cert["status"] == "FAIL":
                    edge["status"] = "FAIL"
                    break
            else:
                edge["status"] = "PASS"
            if reason is None:
                event["status"] = edge["status"]
            event["after"] = counters(oracle, evidence)
            journal.append(event)
            if edge["status"] == "FAIL":
                for child in candidates:
                    if edge["edge_id"] in child["edge_ids"]:
                        child.update(status="FAIL", failed_edge_id=edge["edge_id"])
            continue
        require(all(edges[eid]["status"] == "PASS" for eid in c["edge_ids"]), "Invalid active path")
        before = counters(oracle, evidence)
        info = mech.plan(c["actions"])
        samples = []
        for state in info["states"][1:]:
            for offset in range(0, mech.slot + 1, mech.sample):
                seconds = (state["step"] - 1) * mech.slot + offset
                cert = oracle.certify(state, seconds)
                require(cert is not None and cert["status"] == "PASS" and cert["reliable"] is True,
                        "Complete path reconstruction is not fully paid PASS")
                samples.append({"step": state["step"], "time_seconds": seconds,
                    "point_id": cert["point_id"], "status": cert["status"], "state": state})
        require(bootstrap.attempt_index + len(extra_attempts) < len(attempts), "Missing successful complete record")
        record = attempts[bootstrap.attempt_index + len(extra_attempts)]
        require(record["actions"] == c["actions"] and record["origin"] == "lazysp_" + selector
            and record["details"] == {"branch_id": bid, "edge_ids": c["edge_ids"]}
            and record["status"] == "COMPLETE" and record["certified_complete"] is True,
            "Canonical record identity changed")
        require(record["physical_points_before"] == before["logical_points"]
            and record["physical_points_after"] == oracle.used
            and record["validation"]["samples"] == samples, "Canonical samples/cost changed")
        old.check_saved_plan(record["validation"], mech.initial, mech.static, mech.loads, mech.scenario,
            mech.condition, mech.event, mech.horizon, mech.slot, mech.sample, oracle.points)
        same(record["validation"]["energy_kwh"], c["energy_kwh"], "canonical_energy")
        extra_attempts.append(record); best = record; c["status"] = "COMPLETE"
        journal.append({"event": "CANONICAL_COMPLETE_CERTIFICATE", "branch_id": bid,
            "before": before, "after": counters(oracle, evidence)})
        reason = "SHORTEST_FIXED_POOL_PATH_CERTIFIED"
    require(oracle.index == len(oracle.rows) and oracle.used == len(evidence["logical"])
            and bootstrap.attempt_index + len(extra_attempts) == len(attempts), "Unconsumed paid evidence/attempts")
    lazy = result["lazysp"]
    expected = {"version": VERSION, "selector": selector,
        "graph": "exact_fixed_complete_plan_prefix_tree_no_new_recombined_paths",
        "cost": "slot_unserved_normal_energy_or_infinity_for_reliable_FAIL",
        "tie": "maximum_exact_graph_energy_then_smallest_branch_id", "guard_reference": reference,
        "external_guard_plan": guard["best_plan"], "max_selected_plans": 16,
        "selected_plan_count": len(selected), "selected_plan_ids": selected,
        "candidate_records": candidates, "edges": edges, "journal": journal,
        "certificate_reconstruction": "original_validate_plan_chronological_all_samples_paid_requests",
        "coverage_window": "global_shortest_path_not_alternating_held_out_subwindow", "budget_optimality_claim": False}
    same(lazy, expected, "independent_lazysp_full_causal_replay")
    require(result["version"] == VERSION and result["done"] is True and result["stop_reason"] == reason
        and result["best_plan"] == best and result["incumbent"] == best
        and result["method"] == "fixed_prefix_tree_lazysp_" + selector
        and result["status"] == ("CERTIFIED_INCUMBENT" if best else "NO_CERTIFIED_COMPLETE_PLAN")
        and result["verify_attempts"] == len(selected)
        and result["total_decisions"] == sum(e["event"] == "EDGE_SELECTED" for e in journal)
        and result["checked_candidates"] == len(bootstrap.checked) + len(extra_attempts)
        and result["pending_complete_plans"] == sum(c["status"] == "PENDING" for c in candidates)
        and result["optimality_established"] is False and result["guard_established"] is True
        and result["complete_guard_still_required"] is False, "LazySP result/termination differs")
    require(result["logical_points"] == oracle.used and result["actual_physical_points"] == evidence["physical_points"]
        and result["search_decisions"] == len(journal) and result["policy_decisions"] == 0
        and result["new_training"] == 0 and result["physical_points"] == oracle.used, "LazySP cost/model attribution differs")
    require(result["initial_certificate"] == bootstrap.initial_certificate, "Bootstrap initial evidence changed")
    require(result["fully_certified"] is (best is not None), "Incomplete plan certified")
    if best:
        same(result["quality_pct"], 100 * best["validation"]["energy_kwh"] / mech.denominator, "early_quality")
    return {"phase": "EARLY", "candidate_pool_sha256": digest(result["stage_generation"]),
        "selected_plan_count": len(selected), "edge_evaluations": sum(e["event"] == "EDGE_SELECTED" for e in journal),
        "stop_reason": reason, "full_causal_replay": True}


def check_cell(run, spec):
    cell = run / spec["cell"]
    shared.checked_manifest(cell)
    require(not (cell / "error.json").exists(), "Worker error retained")
    cfg, scope = read(run / "config.json"), read(run / "physical_scope.json")
    mech = mechanics(run, spec)
    result = read(cell / "result.json")
    require(result["study"] == STUDY and result["spec"] == spec and result["models_used"] == result["new_training"] == 0,
            "Wrong task identity or hidden training")
    points, statuses = shared.check_actual_points(run, cell / "physical", spec, mech)
    commands = old.check_commands(cell / "physical", points, read(run / "command_inventory.json"))
    shared.check_actual_requests(list(read_lines(cell / "physical/requests.jsonl")), points,
        read(cell / "physical/oracle_summary.json"), commands, statuses)
    views = {name: {"logical_rows": list(read_lines(cell / name / "logical_points.jsonl.gz")),
                    "requests": list(read_lines(cell / name / "logical_requests.jsonl"))} for name in ("GUARD", "EARLY")}
    evidence, _ = gf.check_service_ledger(list(read_lines(cell / "service_events.jsonl")), views, points,
        mech, scope, sha(run / "condition.json"), "shared")
    service = read(cell / "service_summary.json")
    gf.check_service_summaries(service, {n: read(cell / n / "view_summary.json") for n in views},
        evidence, scope, "shared", commands["numerical_entries"])
    require(result["service"] == service, "Top service differs")
    phases, checks = {}, {}
    for name in views:
        folder = cell / name
        phase = read(folder / "phase_result.json")
        require(read(folder / "initial_state.json") == mech.initial and phase["oracle"] == service["views"][name],
                "Initial state or oracle attribution differs")
        require(phase["spec"] == shared.expected_phase_spec(spec, name),
                "Phase namespace changed")
        attempts = list(read_lines(folder / "plans.jsonl.gz"))
        require(phase["attempts"] == attempts, "Plan journal differs")
        decisions = list(read_lines(folder / "decisions.jsonl.gz"))
        if name == "GUARD":
            checked = shared.check_phase_records(mech, cfg["branch_search"], name, phase["spec"], phase,
                attempts, decisions, evidence[name], sha(run / "condition.json"))
            require(phase["done"] and phase["fully_certified"], "No complete GUARD")
        else:
            reference = {"phase": "GUARD", "file": "GUARD/phase_result.json", "sha256": sha(cell / "GUARD/phase_result.json")}
            if spec["arm"] == "P":
                checked = gf.check_early_records(mech, cfg["branch_search"], phase["spec"], phase,
                    attempts, decisions, evidence[name], sha(run / "condition.json"), phases["GUARD"], reference)
            else:
                require(not decisions, "Unexplained extra LazySP decisions")
                checked = check_lazy_records(mech, cfg["branch_search"], phase, attempts, evidence[name],
                    sha(run / "condition.json"), phases["GUARD"], reference, spec["selector"])
        phases[name], checks[name] = phase, checked
    chosen = gf.early.protected_choice(phases["EARLY"], phases["GUARD"], True)
    require(result["best_plan"] == chosen["best_plan"] and result["selected_phase"] == chosen["selected_phase"]
        and chosen["guard_established"] is True, "Final complete baseline protection differs")
    selected = phases[chosen["selected_phase"]]
    require(result["quality_pct"] == selected["quality_pct"]
        and result["baseline_quality_pct"] == phases["GUARD"]["quality_pct"]
        and result["status"] == selected["status"] and result["protected"] is True
        and result["intermediate_beam_floor_claimed"] is False, "Final quality/status differs")
    expected_guard = {"guard_established": True, "baseline_status": phases["GUARD"]["status"],
        "baseline_energy_kwh": phases["GUARD"]["best_plan"]["validation"]["energy_kwh"],
        "final_energy_kwh": chosen["best_plan"]["validation"]["energy_kwh"],
        "nondegradation_verified": True, "tie_rule": "exact_equal_energy_select_guard",
        "scope": "Fixed 30-minute normal-service objective; computation waiting excluded; final completed output only"}
    require(result["guard_audit"] == expected_guard, "Final guard evidence differs")
    for key, expected in {"logical_points": sum(v["logical_points"] for v in evidence.values()),
        "actual_physical_points": len(points), "requests": sum(v["requests"] for v in evidence.values()),
        "numerical_entries": commands["numerical_entries"]}.items():
        require(result[key] == expected, "Total cost differs: " + key)
    started, finished = read(cell / "worker_started.json"), read(cell / "worker_finished.json")
    require(started["pid"] == finished["pid"] and finished["status"] == "COMPLETE"
        and finished["result_sha256"] == sha(cell / "result.json"), "Worker completion differs")
    require(math.isfinite(result["worker_elapsed_seconds"]) and result["worker_elapsed_seconds"] > 0,
            "Invalid worker time")
    require(sum(v["phase_elapsed_seconds"] for v in phases.values()) <= result["worker_elapsed_seconds"],
            "Phase times exceed worker time")
    return {"status": "PASS", **{k: spec[k] for k in ("cell", "arm", "context_id", "scenario_id", "event_id")},
        "worker_pid": started["pid"], "worker_elapsed_seconds": result["worker_elapsed_seconds"],
        "quality_pct": result["quality_pct"], "energy_kwh": chosen["best_plan"]["validation"]["energy_kwh"],
        "guard_quality_pct": result["baseline_quality_pct"], "guard_established": True,
        "physical_points": len(points), "logical_points": result["logical_points"],
        "requests": result["requests"], "numerical_entries": commands["numerical_entries"],
        "physical_statuses": dict(statuses), "phase_checks": checks,
        "candidate_pool_sha256": digest(phases["EARLY"]["stage_generation"]),
        "guard_scientific_sha256": checks["GUARD"]["semantic_signature_sha256"],
        "result_sha256": sha(cell / "result.json")}


def audit(run):
    output = run / "audit"
    require(not output.exists(), "Refuse to replace previous audit")
    output.mkdir()
    checks, errors = [], []
    try:
        check_frozen(run)
        gate = read(run / "preflight.json")
        require(gate["status"] == "PASS" and gate["prepared_sha256"] == sha(run / "prepared.json"), "Missing preflight")
        schedule, control = read(run / "schedule.json"), read(run / "controller_completion.json")
        require(control["status"] == "COMPLETE_PENDING_AUDIT" and len(control["exits"]) == 12
            and not control["not_run"] and control["new_training"] == 0, "Incomplete population retained")
        previous_end = 0
        for spec, exit_row in zip(schedule, control["exits"]):
            checked = check_cell(run, spec)
            require(exit_row["cell"] == spec["cell"] and exit_row["arm"] == spec["arm"]
                and exit_row["context_id"] == spec["context_id"] and exit_row["exit_code"] == 0
                and exit_row["timed_out"] is False, "Wrong/failed/retried exit")
            start, end, wall = (exit_row[k] for k in ("started_epoch_seconds", "finished_epoch_seconds", "process_wall_seconds"))
            require(previous_end <= start <= end and wall >= checked["worker_elapsed_seconds"]
                and abs(end - start - wall) < 2., "Cold process overlap/incomplete clock")
            execution = read(run / spec["cell"] / "execution.json")
            require(execution["started_epoch_seconds"] == start
                and 0 < execution["deadline_epoch_seconds"] - start <= 180.01, "Wrong task wall limit")
            previous_end = end
            checked["process_wall_seconds"] = wall
            checks.append(checked)
            save(output / "progress.json", {"audited_cells": len(checks), "status": "RUNNING"})
        for cid in CONTEXTS:
            rows = [c for c in checks if c["context_id"] == cid]
            require(len(rows) == 3 and {x["arm"] for x in rows} == set(ARMS)
                and len({x["candidate_pool_sha256"] for x in rows}) == 1
                and len({x["guard_scientific_sha256"] for x in rows}) == 1,
                "Candidate pools or original GUARD differ by arm")
        require(sum(x["physical_points"] for x in checks) <= 6144, "Total physical cap exceeded")
    except BaseException as exc:
        errors.append({"error": repr(exc), "traceback": traceback.format_exc()})
    result = {"status": "PASS" if len(checks) == 12 and not errors else "FAIL", "audited_cells": len(checks),
        "scheduled_cells": 12, "checks": checks, "errors": errors,
        "auditor_sha256": sha(Path(__file__)), "new_physical_points": 0, "new_training": 0}
    save(output / "summary.json", result)
    save(output / "completion_manifest.json", {p.relative_to(output).as_posix(): sha(p)
         for p in sorted(output.rglob("*")) if p.is_file()})
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("preflight", "audit"))
    p.add_argument("--run", required=True, type=Path)
    args = p.parse_args()
    out = preflight(args.run.resolve()) if args.command == "preflight" else audit(args.run.resolve())
    print(json.dumps({k: v for k, v in out.items() if k != "checks"}, ensure_ascii=False))
    sys.exit(0 if out["status"] == "PASS" else 1)
