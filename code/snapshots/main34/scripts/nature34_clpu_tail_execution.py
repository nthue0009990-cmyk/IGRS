"""One frozen, cold Beam-guarded continuation episode; never trains or retries.

The original physical adapter and policy loader are reused unchanged.  The
baseline, continuation and deterministic verification selections have separate
provenance in one continuous journal and one counted physical oracle.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
sys.dont_write_bytecode = True

from nature34_experiment_support import read, save, sha, append, numeric_lists, dependencies
from nature34_clpu_branch_execution import _observation, validate_spec as validate_old_spec
from run_nature34_clpu_pilot import SnapshotOracle, make_env

STUDY = "PPO_TAIL_PILOT_V1"
GROUP_METHODS = {"G0": "beam", "G1": "beam", "G2": "bc_raw", "G3": "ppo_raw"}
_WORKER_ENTERED = False


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_spec(run, spec):
    require(isinstance(spec, dict), "Cell specification must be an object")
    require(spec.get("tail_group") in GROUP_METHODS, "Unknown tail group")
    require(spec.get("method") == GROUP_METHODS[spec["tail_group"]], "Tail group/method mismatch")
    require(spec.get("phase") == "evaluation" and spec.get("deterministic") is True,
            "Tail pilot admits deterministic evaluation only; no training")
    require(spec.get("representation") == "raw" and spec.get("point_budget") == 256,
            "Tail pilot requires frozen raw representation and 256 physical points")
    require(type(spec.get("seed")) is int and spec["seed"] in
            ((0,) if spec["tail_group"] in ("G0", "G1") else (17, 41, 73)), "Unfrozen group seed")
    require(type(spec.get("deadline_utc_epoch")) in (int, float) and
            math.isfinite(spec["deadline_utc_epoch"]), "Missing finite runtime deadline")
    return validate_old_spec(run, spec)


def validate_config(config):
    from dataclasses import asdict
    from dnr.nature34_clpu_branch import BranchConfig
    require(config.get("decision") == "D059" and config.get("validation_study") == STUDY,
            "Expected independently frozen tail pilot")
    actual = {**config.get("branch_search", {}), "point_budget": 256}
    require(actual == asdict(BranchConfig()), "Original Beam mechanics must remain unchanged")
    tail = config.get("tail_search", {})
    require(tail.get("continuation_expansions") == 256 and
            tail.get("enable_exact_failed_prefix") is True and
            tail.get("enable_tail_verification") is True, "Unfrozen continuation settings")
    return actual


def verify_worker_inputs(run, cell, spec):
    require(run.parent == (ROOT / "runs").resolve(), "Run is outside project runs")
    require(cell != run and cell.is_relative_to(run), "Cell is outside its run")
    validate_config(read(run / "config.json"))
    prepared = read(run / "prepared.json")
    require(prepared.get("status") == "FROZEN_PREPARED_NO_DSS", "Run has not been frozen")
    require(sha(run / "prephysical_freeze_manifest.json") == prepared["freeze_sha256"],
            "Freeze identity changed")
    for name, digest in read(run / "prephysical_freeze_manifest.json").items():
        target = (run / name).resolve()
        require(target.is_relative_to(run) and sha(target) == digest, "Frozen input changed: " + name)
    for name in prepared["source_files"]:
        source = (ROOT / name).resolve()
        require(source.is_relative_to(ROOT.resolve()) and sha(source) == sha(run / "source" / name),
                "Frozen implementation changed: " + name)
    require(dependencies() == read(run / "dependencies.json"), "Frozen dependencies changed")
    require(sha(Path(sys.executable)) == read(run / "python_environment.json")["sha256"],
            "Python interpreter changed")
    checkpoint = validate_spec(run, spec)
    validate_schedule_binding(run, cell, spec)
    return checkpoint


def validate_schedule_binding(run, cell, spec):
    templates = {row["cell"]: row for row in read(run / "schedule.json")}
    relative = cell.relative_to(run).as_posix()
    require(relative in templates, "Cell is outside the frozen schedule")
    template = templates[relative]
    extras = {"template_sha256", "deadline_utc_epoch", "checkpoint", "checkpoint_sha256"}
    require({k: v for k, v in spec.items() if k not in extras} == template,
            "Cell differs from frozen schedule")
    canonical = json.dumps(template, sort_keys=True, separators=(",", ":"), allow_nan=False)
    require(spec.get("template_sha256") == hashlib.sha256(canonical.encode()).hexdigest(),
            "Schedule template hash differs")
    if spec["method"] != "beam":
        matches = [b for b in read(run / "model_bindings.json")
                   if b["method"] == spec["method"] and b["seed"] == spec["seed"]]
        require(len(matches) == 1 and matches[0]["checkpoint"] == spec["checkpoint"]
                and matches[0]["source_sha256"] == spec["checkpoint_sha256"],
                "Fixed endpoint does not match the frozen method/seed binding")


def physical_identity(run, spec, env):
    """Bind exact history to the immutable model/physics inputs, not a topology alone."""
    return {"study": STUDY, "freeze_sha256": sha(run / "prephysical_freeze_manifest.json"),
            "scenario_id": spec["scenario_id"], "event_id": spec["event_id"],
            "initial_state": numeric_lists(env.initial.to_dict()),
            "condition_sha256": sha(run / "condition.json"),
            "physical_config_sha256": sha(run / "physical_config.json"),
            "author_static_metadata_sha256": sha(run / "author_static_metadata.json"),
            "source_provenance_sha256": sha(run / "source_provenance.json")}


def create_search(env, oracle, config, spec, identity, record_callback, event_callback):
    from dnr.nature34_clpu_branch import BranchConfig
    from dnr.nature34_clpu_tail import TailBranchSearch
    return TailBranchSearch(env, oracle, BranchConfig(**{**config["branch_search"], "point_budget": 256}),
                            physics_identity=identity, continuation=spec["tail_group"] != "G0",
                            extra_branch_expansions=config["tail_search"]["continuation_expansions"],
                            record_callback=record_callback, event_callback=event_callback)


def load_fixed_policy(checkpoint, spec):
    if checkpoint is None:
        return None, None
    import torch
    from dnr.nature34_clpu_branch_learning import load_policy
    torch.set_num_threads(1)
    policy = load_policy(checkpoint, representation="raw")
    require(policy.representation == "raw" and not policy.associated, "Loaded policy representation differs")
    generator = torch.Generator(device="cpu").manual_seed(spec.get("sampling_seed", spec["seed"]))
    return policy, generator


def check_deadline(spec):
    if time.time() >= spec["deadline_utc_epoch"]:
        raise TimeoutError("Frozen worker deadline reached; partial evidence retained")


def guard_evidence(baseline, result):
    """A reliable physical FAIL is retained; it does not establish a guard."""
    before, after = baseline.get("best_plan"), result.get("best_plan")
    established = (baseline.get("status") == "CERTIFIED_INCUMBENT" and isinstance(before, dict)
                   and before.get("certified_complete") is True
                   and before.get("validation", {}).get("status") == "COMPLETE")
    if not established:
        require(baseline.get("status") in ("INITIAL_FAIL", "NO_CERTIFIED_COMPLETE_PLAN"),
                "Malformed baseline certificate")
        require(result.get("status") == baseline["status"] and after == before,
                "Missing baseline must not be silently replaced by unguarded continuation")
        return {"guard_established": False, "baseline_status": baseline["status"],
                "reason": "NO_COMPLETE_BASELINE_CERTIFICATE"}
    require(result.get("status") == "CERTIFIED_INCUMBENT" and isinstance(after, dict)
            and after.get("certified_complete") is True
            and after.get("validation", {}).get("status") == "COMPLETE",
            "Complete certified baseline was lost")
    lower, final = float(before["validation"]["energy_kwh"]), float(after["validation"]["energy_kwh"])
    require(math.isfinite(lower) and math.isfinite(final) and final >= lower,
            "Final certified objective is below the baseline; no tolerance may excuse degradation")
    return {"guard_established": True, "baseline_status": baseline["status"],
            "baseline_energy_kwh": lower, "final_energy_kwh": final,
            "nondegradation_verified": True,
            "scope": "Fixed 30-minute normal-service planning objective; computation waiting is not in the objective"}


def process_memory_evidence():
    """Read optional OS counters; unavailable counters are never imputed."""
    if os.name != "nt":
        return {"available": False, "reason": "WINDOWS_PROCESS_COUNTER_NOT_AVAILABLE"}
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("page_fault_count", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("peak_working_set_bytes", "working_set_bytes",
                "quota_peak_paged_pool", "quota_paged_pool", "quota_peak_nonpaged_pool",
                "quota_nonpaged_pool", "pagefile_usage", "peak_pagefile_usage")]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {"available": True, "source": "Windows GetProcessMemoryInfo",
                "working_set_bytes": int(counters.working_set_bytes),
                "peak_working_set_bytes": int(counters.peak_working_set_bytes),
                "scope": "Current worker process before final result serialization/sealing"}
    except (OSError, AttributeError) as exc:
        return {"available": False, "reason": repr(exc)}


def collect_episode(search, spec, transition_callback, decision_callback, stage_callback,
                    checkpoint_callback, policy=None, generator=None):
    """Select by the *actual* stage, never attributing baseline/verify actions to PPO."""
    count, stage_counts, timing = 0, {}, {}
    previous_phase = None
    while True:
        phase = search.phase
        if phase != previous_phase:
            if previous_phase == "BASELINE" or (previous_phase is None and phase != "BASELINE"):
                checkpoint_callback(search, count)
            stage_callback({"stage": phase, "transition_boundary": count,
                            "oracle": {"requests": search.oracle.requests, "points": search.oracle.points},
                            "utc": datetime.now(timezone.utc).isoformat()})
            previous_phase = phase
        if search.done:
            break
        check_deadline(spec)
        require(phase in ("BASELINE", "CONTINUATION", "VERIFY"), "Unknown live search phase")
        timer = timing.setdefault(phase, {"observation_seconds": 0., "selection_seconds": 0.,
                                          "search_step_seconds": 0., "journal_seconds": 0.})
        began = time.perf_counter()
        raw, associated = _observation(search, "raw"), _observation(search, "associated")
        require(raw["branch_ids"] == associated["branch_ids"] and raw["mask"] == associated["mask"],
                "Representation changed available branches")
        timer["observation_seconds"] += time.perf_counter() - began
        method = "beam" if phase == "BASELINE" else "tail_priority" if phase == "VERIFY" else spec["method"]
        is_policy = method in ("bc_raw", "ppo_raw")
        began = time.perf_counter()
        if is_policy:
            require(policy is not None, "Missing fixed continuation policy")
            selected = policy.act(raw, raw["mask"], deterministic=True, generator=generator)
            action = selected["action"]
            action = action.item() if hasattr(action, "item") else action
            logprob, value = float(selected["logprob"]), float(selected["value"])
            require(math.isfinite(logprob) and math.isfinite(value), "Nonfinite fixed policy evidence")
        else:
            action = search.rule_action(method)
            logprob = value = None
        timer["selection_seconds"] += time.perf_counter() - began
        require(type(action) is int and 0 <= action < len(raw["mask"]) and raw["mask"][action],
                "Unavailable branch selected")
        stage_index = stage_counts.get(phase, 0)
        row = {"transition_id": count, "stage_transition_id": stage_index, "stage": phase,
               "obs": raw, "obs_raw": raw, "obs_associated": associated,
               "mask": raw["mask"], "action": action, "branch_id": raw["branch_ids"][action],
               "logprob": logprob, "value": value, "method": method, "representation": "raw",
               "seed": spec["seed"], "sampling_seed": spec.get("sampling_seed", spec["seed"]),
               "checkpoint_sha256": spec.get("checkpoint_sha256") if is_policy else None,
               "oracle_before": {"requests": search.oracle.requests, "points": search.oracle.points}}
        began = time.perf_counter()
        decision_callback({**row, "status": "STARTED"})
        timer["journal_seconds"] += time.perf_counter() - began
        began = time.perf_counter()
        response = search.step(action)
        timer["search_step_seconds"] += time.perf_counter() - began
        require(isinstance(response, dict) and "reward" in response and type(response.get("done")) is bool,
                "Malformed search response")
        reward = float(response["reward"])
        require(math.isfinite(reward), "Nonfinite search reward")
        info = numeric_lists({key: val for key, val in response.items() if key not in ("reward", "done")})
        row.update(reward=reward, done=response["done"], info=info,
                   oracle_after={"requests": search.oracle.requests, "points": search.oracle.points})
        began = time.perf_counter()
        transition_callback(row)
        decision_callback({"transition_id": count, "stage_transition_id": stage_index, "stage": phase,
                           "status": "COMPLETED", "action": action, "branch_id": row["branch_id"],
                           "reward": reward, "done": row["done"], "info": info,
                           "oracle_before": row["oracle_before"], "oracle_after": row["oracle_after"]})
        timer["journal_seconds"] += time.perf_counter() - began
        count += 1
        stage_counts[phase] = stage_index + 1
        require(count <= 768, "Total selected branches exceed expansion plus verification limits")
    return {"search_decisions": count, "stage_decisions": stage_counts, "stage_component_seconds": timing}


def run_worker(run, cell):
    global _WORKER_ENTERED
    run, cell = Path(run).resolve(), Path(cell).resolve()
    require(not _WORKER_ENTERED, "Each task requires a fresh worker process")
    _WORKER_ENTERED = True
    require(not any((cell / name).exists() for name in ("worker_started.json", "result.json", "error.json",
                "command_ledger.jsonl", "points.jsonl.gz", "baseline_result.json")),
            "Cell already started; automatic retry is forbidden")
    started = time.perf_counter()
    spec, config = read(cell / "spec.json"), read(run / "config.json")
    baseline_saved = False
    try:
        checkpoint = verify_worker_inputs(run, cell, spec)
        with (cell / "worker_started.json").open("x", encoding="utf-8") as stream:
            append(stream, {"utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
                            "spec_sha256": sha(cell / "spec.json"), "checkpoint_sha256": spec.get("checkpoint_sha256")})
        model_started = time.perf_counter()
        policy, generator = load_fixed_policy(checkpoint, spec)
        model_load_seconds = time.perf_counter() - model_started
        env = make_env(run, spec)
        save(cell / "initial_state.json", env.initial.to_dict())
        identity = physical_identity(run, spec, env)
        save(cell / "physical_identity.json", identity)
        check_deadline(spec)

        def save_baseline(search, count):
            nonlocal baseline_saved
            require(not baseline_saved, "Baseline checkpoint would be overwritten")
            engine_checkpoint = numeric_lists(search.baseline_checkpoint())
            baseline_result = engine_checkpoint["result"]
            save(cell / "baseline_result.json", baseline_result)
            save(cell / "baseline_checkpoint.json", {
                "engine_checkpoint": engine_checkpoint,
                "baseline_result_sha256": sha(cell / "baseline_result.json"),
                "transition_boundary": count, "attempt_boundary": len(baseline_result["attempts"]),
                "oracle_boundary": {"requests": search.oracle.requests, "points": search.oracle.points},
                "elapsed_seconds": time.perf_counter() - started,
                "physical_identity_sha256": sha(cell / "physical_identity.json")})
            baseline_saved = True

        with gzip.open(cell / "plans.jsonl.gz", "xt", encoding="utf-8", compresslevel=3) as plans, \
                gzip.open(cell / "transitions.jsonl.gz", "xt", encoding="utf-8", compresslevel=3) as transitions, \
                gzip.open(cell / "decisions.jsonl.gz", "xt", encoding="utf-8", compresslevel=3) as decisions, \
                (cell / "stage_events.jsonl").open("x", encoding="utf-8") as stages, \
                (cell / "tail_events.jsonl").open("x", encoding="utf-8") as tail_events, \
                SnapshotOracle(run, cell, env, spec["point_budget"]) as oracle:
            search = create_search(env, oracle, config, spec, identity,
                                   lambda record: append(plans, numeric_lists(record)),
                                   lambda event: append(tail_events, numeric_lists(event)))
            collected = collect_episode(search, spec, lambda row: append(transitions, row),
                                        lambda row: append(decisions, row), lambda row: append(stages, row),
                                        save_baseline, policy, generator)
            result = numeric_lists(search.result())
            require(baseline_saved, "No immutable baseline checkpoint saved")
            protection = guard_evidence(read(cell / "baseline_result.json"), result)
        if checkpoint is not None:
            require(sha(checkpoint) == spec["checkpoint_sha256"], "Fixed checkpoint changed during evaluation")
        oracle_summary = read(cell / "oracle_summary.json")
        require(oracle_summary["points"] <= 256, "Physical budget exceeded")
        result.update(spec=spec, elapsed_seconds=time.perf_counter() - started,
                      model_load_seconds=model_load_seconds, oracle=oracle_summary, **collected,
                      training_interactions=0, optimizer_updates=0, trajectory_file="transitions.jsonl.gz",
                      checkpoint_sha256=spec.get("checkpoint_sha256"),
                      guard_audit=protection,
                      baseline_result_sha256=sha(cell / "baseline_result.json"),
                      baseline_checkpoint_sha256=sha(cell / "baseline_checkpoint.json"),
                      process_memory=process_memory_evidence(),
                      formal_holdout_numerical_access=False, tune_numerical_access=False,
                      clpu_holdout_access=False,
                      time_scope="Cold worker verification/loading/search/logging; excludes process launch and final sealing")
        save(cell / "result.json", result)
        save(cell / "completion_manifest.json", {str(path.relative_to(cell)): sha(path)
             for path in sorted(cell.rglob("*")) if path.is_file()
             and path.name not in ("completion_manifest.json", "stdout.log", "stderr.log")})
        return 0
    except BaseException as exc:
        save(cell / "error.json", {"status": "STOPPED_RELIABILITY_OR_IMPLEMENTATION_ERROR",
             "error": repr(exc), "traceback": traceback.format_exc(), "spec": spec,
             "elapsed_seconds": time.perf_counter() - started, "baseline_checkpoint_saved": baseline_saved,
             "no_automatic_retry": True, "partial_evidence_retained": True})
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cell", type=Path, required=True)
    args = parser.parse_args()
    return run_worker(args.run, args.cell)


if __name__ == "__main__":
    raise SystemExit(main())
