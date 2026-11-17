"""Report D037 only after all three independent saved-result audits pass.

This saved-data reporter uses only the Python standard library. It never
imports a simulator, policy, optimizer, or the execution/audit entry points.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/nature34_imitation_2026-10-19.md"
BUDGETS, SEEDS, DENOMINATOR, TOL = [1, 2, 4, 8], [17, 41, 73], 1769., 1e-9
METHODS = ["independent_bc", "independent_ppo", "conditional_bc", "conditional_ppo", "legal_random", "graph_repair"]
LABELS = {"independent_bc": "状态·BC", "independent_ppo": "状态·BC+PPO", "conditional_bc": "条件·BC",
          "conditional_ppo": "条件·BC+PPO", "legal_random": "合法随机", "graph_repair": "两起点图规则"}
COMPARISONS = [("independent_ppo-minus-independent_bc", "状态PPO−BC", "independent_ppo", "independent_bc", "primary"),
               ("conditional_ppo-minus-conditional_bc", "条件PPO−BC", "conditional_ppo", "conditional_bc", "primary"),
               ("conditional_ppo-minus-independent_ppo", "条件PPO−状态PPO", "conditional_ppo", "independent_ppo", "secondary"),
               ("conditional_bc-minus-independent_bc", "条件BC−状态BC", "conditional_bc", "independent_bc", "secondary"),
               ("conditional_ppo-minus-graph_repair", "条件PPO−图规则", "conditional_ppo", "graph_repair", "secondary"),
               ("conditional_bc-minus-graph_repair", "条件BC−图规则", "conditional_bc", "graph_repair", "secondary"),
               ("independent_ppo-minus-graph_repair", "状态PPO−图规则", "independent_ppo", "graph_repair", "secondary")]
FAILURE_STATUS = "VERIFIED_ZERO_CANDIDATE_INITIALIZATION_FAILURE"
FAILED_INITIALIZATION_RUN = "20261019T100044_861915Z_nature34_expert_data"
SUPERSEDED_PREPARATION = "20261019T095714_292294Z_nature34_expert_data"
STAGES = {"expert": {"prefix": "nature34_expert_data", "audit_status": "VERIFIED_BUILD_GRAPH8_EXPERT_DATA_SAVED_RESULTS",
                       "source_status": "COMPLETE_GRAPH8_EXPERT_DATA_PENDING_SAVED_AUDIT", "actions": 2432, "points": 12160, "entries": 48643},
          "training": {"prefix": "nature34_imitation_training", "audit_status": "VERIFIED_BC_PPO_TRAINING_SAVED_RESULTS",
                         "source_status": "COMPLETE_BC_PPO_TRAINING_PENDING_SAVED_AUDIT", "actions": 6144, "points": 30720, "entries": 122883},
          "evaluation": {"prefix": "nature34_imitation_evaluation", "audit_status": "VERIFIED_TUNE_BC_PPO_EVALUATION_SAVED_RESULTS",
                           "source_status": "COMPLETE_TUNE_BC_PPO_EVALUATION_PENDING_SAVED_AUDIT", "actions": 4096, "points": 20480, "entries": 81923}}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lines(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(row) for row in stream if row.strip()]


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def mean(values):
    require(bool(values) and all(finite(value) for value in values), "Empty/nonfinite mean")
    return sum(values) / len(values)


def run_path(name):
    require(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_]+", name), "Invalid run identity")
    return ROOT / "runs" / name


def consumed_gate(directory, expected, names):
    require(sha(directory / "completion_manifest.json") == expected, "Sealed completion identity differs")
    manifest = read(directory / "completion_manifest.json")
    checked = {}
    for name in names:
        target = (directory / name).resolve()
        require(target.is_relative_to(directory.resolve()) and name in manifest and sha(target) == manifest[name],
                "Consumed sealed file changed: " + name)
        checked[name] = manifest[name]
    return checked


def stage_gate(stage, explicit_run=None):
    spec = STAGES[stage]
    require(isinstance(spec["audit_status"], str) and isinstance(spec["source_status"], str),
            "Final independent " + stage + " audit/schema is pending; do not infer completion")
    pointer = read(ROOT / "artifacts" / (spec["prefix"] + "_latest.json"))
    audit_pointer = read(ROOT / "artifacts" / (spec["prefix"] + "_saved_audit_latest.json"))
    source, audit_run = run_path(pointer["run"]), run_path(audit_pointer["run"])
    require(explicit_run is None or explicit_run == source.name, "Explicit stage run differs from sealed pointer")
    require(audit_pointer["source_run"] == source.name and audit_pointer["status"] == spec["audit_status"],
            "Required independent " + stage + " audit has not passed")
    audit_files = consumed_gate(audit_run, audit_pointer["completion_sha256"], ["verification.json"])
    verification = read(audit_run / "verification.json")
    require(verification["status"] == spec["audit_status"] and verification["errors"] == [] and
            verification["source_run"] == source.name and
            verification["source_completion_sha256"] == pointer["completion_sha256"] and
            audit_pointer["source_completion_sha256"] == pointer["completion_sha256"] and
            verification["new_powerflow_calls"] == verification["new_optimizer_steps"] == 0 and
            verification.get("new_training_interactions", 0) == 0, "Audit identity or zero-computation scope differs")
    source_files = consumed_gate(source, pointer["completion_sha256"], ["config.json", "summary.json", "command_stats.json"])
    config, summary, commands = [read(source / name) for name in ("config.json", "summary.json", "command_stats.json")]
    require(summary["status"] == spec["source_status"],
            "Incomplete " + stage + " source")
    physical = summary["physical_summary"] if stage == "training" else summary
    if stage == "training":
        source_files.update(consumed_gate(source, pointer["completion_sha256"], ["physical_summary.json"]))
        require(physical == read(source / "physical_summary.json") and physical["status"] == "COMPLETE_PENDING_SAVED_AUDIT",
                "Training nested physical summary differs from sealed oracle summary")
    require(physical["completed_actions"] == spec["actions"] and physical["point_evaluations"] == spec["points"] and
            physical["formal_holdout_numerical_access"] is False and physical["physical_cache_hits"] == 0,
            "Named " + stage + " action/point boundary differs")
    returned = commands["returned"]
    require(commands["requested"] == returned and not commands["open_call_ids"] and
            commands["calls_failed"] == commands["calls_rejected"] == 0 and
            returned["explicit_solve_commands"] == 3 * spec["points"] + 2 and
            returned["calcvoltagebases_entries"] == spec["points"] + 1 and
            returned["numerical_solution_entries"] == spec["entries"], "Unclosed or incorrect stage numerical-entry budget")
    require(summary["tune_numerical_access"] is (stage == "evaluation") and
            physical["tune_numerical_access"] is (stage == "evaluation") and summary["formal_holdout_numerical_access"] is False,
            "Tune/formal electrical boundary differs")
    require(config["decision"] == "D037" and config["max_numerical_solution_entries"] == 253449 and
            config["formal_holdout_numerical_access"] is False and config["bc_epochs"] == 128 and
            config["ppo_actions_per_job"] == 1024, "Frozen D037 configuration differs")
    return {"source": source, "audit": audit_run, "pointer": pointer, "audit_pointer": audit_pointer,
            "verification": verification, "config": config, "summary": summary, "commands": commands,
            "physical_summary": physical,
            "consumed_source": source_files, "consumed_audit": audit_files}


def validate_training_jobs(jobs, summary, labeled_scenarios, zero_label_scenarios):
    expected = {(seed, variant, stage) for seed in SEEDS for variant in ("independent", "conditional") for stage in ("BC", "BC_PLUS_PPO")}
    require(len(jobs) == 12 and {(r["seed"], r["variant"], r["stage"]) for r in jobs} == expected,
            "Twelve distinct BC/PPO job summaries are required")
    records = []
    for row in jobs:
        require(finite(row["elapsed_seconds"]) and row["elapsed_seconds"] >= 0, "Unknown job compute time")
        if row["stage"] == "BC":
            require(row["status"] == "COMPLETE_FIXED_BC" and row["epochs_completed"] == 128 and
                    row["scheduled_scenario_visits"] == 320 * 128 and
                    row["labeled_scenario_visits"] == labeled_scenarios * 128 and
                    row["zero_reward_label_visits"] == zero_label_scenarios * 128 and
                    row["optimizer_steps"] + row["all_masked_batches"] == 128 * 20 and
                    row["new_DSS_calls"] == row["new_physical_interactions"] == 0,
                    "BC actual updates, full-scene visits or no-label handling differ")
        else:
            require(row["status"] == "COMPLETE_FIXED_PPO_FINETUNING" and row["interactions"] == row["planned_interactions"] == 1024 and
                    row["rollouts_updated"] == 32 and row["optimizer_steps"] == 128,
                    "Fixed PPO interaction/update budget differs")
        records.append({key: row[key] for key in ("seed", "variant", "stage", "status", "elapsed_seconds", "optimizer_steps")})
    bc_steps = sum(r["optimizer_steps"] for r in jobs if r["stage"] == "BC")
    ppo_steps = sum(r["optimizer_steps"] for r in jobs if r["stage"] == "BC_PLUS_PPO")
    require(summary["bc_jobs_completed"] == summary["ppo_jobs_completed"] == 6 and summary["completed_checkpoints"] == 12 and
            summary["bc_optimizer_steps"] == bc_steps and summary["ppo_optimizer_steps"] == ppo_steps == 768 and
            summary["ppo_training_interactions"] == 6144 and summary["final_epoch"] == 128 and
            summary["final_ppo_interactions_per_job"] == 1024, "Training top-level actual counters differ from job summaries")
    return {"jobs": records, "bc_optimizer_steps": bc_steps, "ppo_optimizer_steps": ppo_steps,
            "total_optimizer_steps": bc_steps + ppo_steps,
            "bc_job_seconds_sum": sum(r["elapsed_seconds"] for r in jobs if r["stage"] == "BC"),
            "ppo_job_seconds_sum": sum(r["elapsed_seconds"] for r in jobs if r["stage"] == "BC_PLUS_PPO"),
            "timing_scope": "Sum of recorded individual jobs; PPO includes its physical callbacks; not an independent wall-clock benchmark"}


def training_links(stages, labels):
    expert, training, evaluation = (stages[k] for k in ("expert", "training", "evaluation"))
    directory = training["source"]
    names = ["expert_gate.json", "checkpoint_index.json", "initial_checkpoint_index.json", "bc_frozen_checkpoint_index.json",
             "job_summaries.json", "training_schedules.json", "scenarios.json", "expert_labels.json", "tune_scenarios_frozen.json", "tune_selection.json"]
    training["consumed_source"].update(consumed_gate(directory, training["pointer"]["completion_sha256"], names))
    link = read(directory / "expert_gate.json")
    require(link["expert"]["run"] == expert["source"].name and
            link["expert"]["completion_sha256"] == expert["pointer"]["completion_sha256"] and
            link["independent_audit"]["run"] == expert["audit"].name and
            link["independent_audit"]["completion_sha256"] == expert["audit_pointer"]["completion_sha256"],
            "Training is not linked to this independently audited expert collection")
    require(read(directory / "expert_labels.json") == labels, "Training teacher labels differ from audited expert data")
    source_index, initial = read(directory / "checkpoint_index.json"), read(directory / "initial_checkpoint_index.json")
    expected = {(seed, variant, stage) for seed in SEEDS for variant in ("independent", "conditional") for stage in ("bc", "ppo")}
    require(len(source_index) == 12 and {(r["seed"], r["variant"], r["stage"]) for r in source_index} == expected,
            "Fixed twelve-checkpoint identity coverage differs")
    require(len(initial) == 6 and {(r["seed"], r["variant"]) for r in initial} == {(seed, v) for seed in SEEDS for v in ("independent", "conditional")},
            "D036 initial checkpoint references are incomplete")
    initial_map = {(r["seed"], r["variant"]): r for r in initial}
    model_map = {(r["seed"], r["variant"], r["stage"]): r for r in source_index}
    checkpoint_files = [r["checkpoint"] for r in initial + source_index]
    require(len(checkpoint_files) == len(set(checkpoint_files)), "Checkpoint paths overlap")
    training["consumed_source"].update(consumed_gate(directory, training["pointer"]["completion_sha256"], checkpoint_files))
    for key, row in model_map.items():
        require(row["reload"] == "PASS_ZERO_DSS" and sha(directory / row["checkpoint"]) == row["checkpoint_sha256"], "Fixed model bytes/reload identity differ")
        seed, variant, stage = key
        if stage == "bc":
            origin = initial_map[seed, variant]
            require(origin["source_run"] == training["config"]["required_d036_run"] and
                    sha(directory / origin["checkpoint"]) == origin["checkpoint_sha256"] == row["initial_checkpoint_sha256"],
                    "BC did not reference the exact D036 INITIAL checkpoint")
        else:
            require(row["bc_start_checkpoint_sha256"] == model_map[seed, variant, "bc"]["checkpoint_sha256"],
                    "PPO starting model does not reference its own fixed BC comparator")
    require(read(directory / "bc_frozen_checkpoint_index.json") == [r for r in source_index if r["stage"] == "bc"],
            "Pure BC comparator index changed after fine-tuning")
    streams = read(evaluation["source"] / "streams.json")
    for stream in streams:
        if stream["method"] in METHODS[:4]:
            key = stream["seed"], stream["variant"], stream["stage"]
            require(key in model_map and stream["checkpoint_sha256"] == model_map[key]["checkpoint_sha256"],
                    "Evaluation used a different final BC/PPO checkpoint")
    evaluation["consumed_source"].update(consumed_gate(evaluation["source"], evaluation["pointer"]["completion_sha256"],
        [r["checkpoint"] for r in streams if r["method"] in METHODS[:4]]))
    require(all(evaluation["consumed_source"][r["checkpoint"]] == r["checkpoint_sha256"]
                for r in streams if r["method"] in METHODS[:4]), "Evaluation model bytes differ from the fixed training checkpoint identity")
    require(read(directory / "tune_scenarios_frozen.json") == read(evaluation["source"] / "scenarios.json") and
            read(directory / "tune_selection.json") == read(evaluation["source"] / "scenario_selection.json"),
            "Evaluation changed the preselected tune faults")
    pools = list(labels["labels_by_scenario"].values())
    compute = validate_training_jobs(read(directory / "job_summaries.json"), training["summary"],
                                    sum(bool(pool) for pool in pools), sum(bool(pool) and pool[0]["teacher_reward"] == 0 for pool in pools))
    return {"checkpoint_index": source_index, "initial_checkpoint_index": initial, "compute": compute}


def validate_failed_initialization(verification, expected_run):
    require(verification["status"] == FAILURE_STATUS and verification["errors"] == [] and
            verification["source_run"] == expected_run and
            verification["new_powerflow_calls"] == verification["new_training_interactions"] == verification["new_optimizer_steps"] == 0,
            "Initialization-failure independent audit has not passed")
    require(verification["all_failed_source_files_sha256_preserved"] is True and verification["source_file_count"] == 443 and
            verification["closed_api_calls"] == 21 and verification["unknown_or_failed_api_calls"] == 0 and
            verification["physical_candidate_actions"] == verification["physical_alpha_points"] == verification["expert_labels_created"] == 0,
            "Initialization failure scope or preservation differs")
    commands = verification["command_counts"]
    require((commands["explicit_solve_commands"], commands["calcvoltagebases_entries"], commands["numerical_solution_entries"]) == (2, 1, 3) and
            verification["extra_initialization_solution_entries"] == 3,
            "Initialization failure must retain its three real numerical entries")


def failure_gate():
    pointer = read(ROOT / "artifacts/nature34_expert_initialization_failure_audit_latest.json")
    require(pointer["status"] == FAILURE_STATUS and pointer["source_run"] == FAILED_INITIALIZATION_RUN,
            "Different failed initialization evidence")
    directory = run_path(pointer["run"])
    consumed = consumed_gate(directory, pointer["completion_sha256"],
        ["verification.json", "failed_source_inventory_before.json", "failed_source_inventory_after.json"])
    verification = read(directory / "verification.json")
    validate_failed_initialization(verification, pointer["source_run"])
    before, after = (read(directory / name) for name in ("failed_source_inventory_before.json", "failed_source_inventory_after.json"))
    require(before == after and len(before) == 443, "Failed source inventory was not preserved exactly")
    preparation = run_path(SUPERSEDED_PREPARATION)
    prepared = read(preparation / "prepared.json")
    require(prepared["run"] == SUPERSEDED_PREPARATION and prepared["status"] == "FROZEN_PREPARED_NO_DSS" and
            sha(preparation / "prephysical_freeze_manifest.json") == prepared["freeze_sha256"] and
            not any((preparation / name).exists() for name in ("execution_started.json", "command_ledger.jsonl", "actions.jsonl", "points.jsonl.gz")),
            "Superseded preparation is no longer an unexecuted frozen preparation")
    return {"pointer": pointer, "verification": verification, "audit": directory, "consumed": consumed,
            "superseded_preparation": {"run": SUPERSEDED_PREPARATION, "status": "SUPERSEDED_PREPARATION_NOT_EXECUTED",
                "prepared_sha256": sha(preparation / "prepared.json"), "freeze_sha256": prepared["freeze_sha256"],
                "new_actions": 0, "new_points": 0, "numerical_solution_entries": 0}}


def cost_inventory(stages, failure):
    commands = {key: stage["commands"]["returned"] for key, stage in stages.items()}
    planned = {name: sum(row[name] for row in commands.values()) for name in
               ("explicit_solve_commands", "calcvoltagebases_entries", "numerical_solution_entries")}
    require(planned == {"explicit_solve_commands": 190086, "calcvoltagebases_entries": 63363,
                        "numerical_solution_entries": 253449}, "Completed named-stage cost differs from frozen plan")
    validate_failed_initialization(failure["verification"], FAILED_INITIALIZATION_RUN)
    additional = {name: failure["verification"]["command_counts"][name] for name in planned}
    return {"frozen_config_plan": planned, "completed_named_stages": commands,
            "failed_initialization": {"source_run": FAILED_INITIALIZATION_RUN, "actions": 0, "points": 0, **additional},
            "superseded_preparation": failure["superseded_preparation"],
            "actual_d037_total": {name: planned[name] + additional[name] for name in planned},
            "interpretation": "253449 planned named-stage entries plus3 preserved failed-initialization entries; no modification of the original configuration plan"}


def outcome(witness, reward):
    return "NO_WITNESS" if not witness else "ZERO_RESTORATION" if reward == 0 else "POSITIVE_RESTORATION"


def validate_action(action):
    points = action["points"]
    require(len(points) == 5 and [p["alpha"] for p in points] == [0, .25, .5, .75, 1], "Incomplete alpha grid")
    fractions = [p["nominal_restored_fraction"] for p in points]
    require(all(finite(value) and 0 <= value <= 1 for value in fractions) and len(set(fractions)) == 1,
            "Unknown/alpha-dependent nominal restoration")
    eligible = [p for p in points if p["converged"] is True and p["known_constraints_satisfied"] is True and
                p["ac_status"] == p["isolation_status"] == "PASS"]
    selected = max(eligible, key=lambda p: p["alpha"]) if eligible else None
    expected = {"reward": fractions[0] if selected else 0., "witness_found": selected is not None,
                "selected_alpha": selected["alpha"] if selected else None,
                "witness_point_id": selected["point_id"] if selected else None,
                "accepted_alphas": [p["alpha"] for p in eligible]}
    require(all(action[key] == value for key, value in expected.items()), "Recorded candidate selection differs")


def aggregate_evaluation(scenarios, streams, actions, decisions, rule_metadata):
    ids = [s["scenario_id"] for s in scenarios]
    require(len(ids) == len(set(ids)) == 32 and all(s["partition"] == "tune" for s in scenarios), "Not frozen tune32")
    stream_map = {s["stream_id"]: s for s in streams}
    require(len(stream_map) == len(streams) == 16 and {s["method"] for s in streams} == set(METHODS), "Missing method streams")
    for method in METHODS:
        actual = [s["seed"] for s in streams if s["method"] == method]
        require(actual == [None] if method == "graph_repair" else sorted(actual) == SEEDS, "Wrong method seed coverage")
    require(len(actions) == 4096 and len(decisions) == 2048, "Incomplete evaluation candidates or prefixes")
    faults = {s["scenario_id"]: s["fault_lines"] for s in scenarios}
    groups, counts = defaultdict(list), {k: Counter() for k in ("converged", "known_constraints", "ac", "isolation")}
    for index, action in enumerate(actions):
        require(action["action_id"] == index and action["scenario_id"] in faults and action["stream_id"] in stream_map,
                "Action identities differ")
        stream = stream_map[action["stream_id"]]
        require(action["partition"] == "tune" and action["fault_lines"] == faults[action["scenario_id"]] and
                action["method"] == stream["method"] and action["seed"] == stream["seed"], "Action scope differs")
        validate_action(action)
        for local, point in enumerate(action["points"]):
            require(point["point_id"] == 5 * index + local, "Incomplete point identities")
            for label, field in (("converged", "converged"), ("known_constraints", "known_constraints_satisfied"),
                                 ("ac", "ac_status"), ("isolation", "isolation_status")):
                counts[label][str(point[field])] += 1
        groups[(action["scenario_id"], action["stream_id"])].append(action)
    require(len(groups) == 32 * 16, "Missing scenario/stream trajectory")
    decision_map = {(d["scenario_id"], d["stream_id"], d["candidate_budget"]): d for d in decisions}
    require(len(decision_map) == len(decisions), "Duplicate prefix identity")
    checked, duplicates = [], []
    for (sid, stream_id), history in groups.items():
        history.sort(key=lambda r: r["candidate_index"])
        require([r["candidate_index"] for r in history] == list(range(8)), "Incomplete charged prefix")
        stream, previous_time = stream_map[stream_id], -1.
        duplicates.append({"scenario_id": sid, "stream_id": stream_id,
                           "duplicate_candidates_still_billed": 8 - len({tuple(r["action"]) for r in history})})
        for budget in BUDGETS:
            eligible = [r for r in history[:budget] if r["witness_found"]]
            best = max(eligible, key=lambda r: (r["reward"], -r["candidate_index"])) if eligible else None
            decision = decision_map[(sid, stream_id, budget)]
            expected = {"method": stream["method"], "seed": stream["seed"], "reward": best["reward"] if best else 0.,
                        "witness_found": best is not None, "selected_action_id": best["action_id"] if best else None,
                        "selected_alpha": best["selected_alpha"] if best else None,
                        "witness_point_id": best["witness_point_id"] if best else None}
            require(all(decision[k] == value for k, value in expected.items()), "Saved prefix choice differs")
            elapsed = decision["charged_proposal_and_evaluation_seconds"]
            require(finite(elapsed) and elapsed >= previous_time, "Invalid prefix timing")
            previous_time = elapsed
            checked.append({**decision, "outcome": outcome(decision["witness_found"], decision["reward"]),
                            "scored_nominal_kw": decision["reward"] * DENOMINATOR})
    cases, seeds, methods = [], [], []
    for method in METHODS:
        method_seeds = [None] if method == "graph_repair" else SEEDS
        for budget in BUDGETS:
            selected = [r for r in checked if r["method"] == method and r["candidate_budget"] == budget]
            for sid in ids:
                rows = [r for r in selected if r["scenario_id"] == sid]
                require(len(rows) == len(method_seeds), "Incomplete case seed coverage")
                kinds = Counter(r["outcome"] for r in rows)
                cases.append({"method": method, "candidate_budget": budget, "scenario_id": sid,
                    "mean_reward": mean([r["reward"] for r in rows]), "mean_seconds": mean([r["charged_proposal_and_evaluation_seconds"] for r in rows]),
                    "positive": kinds["POSITIVE_RESTORATION"] / len(rows), "zero": kinds["ZERO_RESTORATION"] / len(rows),
                    "no_witness": kinds["NO_WITNESS"] / len(rows)})
            for seed in method_seeds:
                rows = [r for r in selected if r["seed"] == seed]
                require(len(rows) == 32, "Incomplete per-seed case coverage")
                seeds.append({"method": method, "candidate_budget": budget, "seed": seed,
                    "mean_reward": mean([r["reward"] for r in rows]), "outcomes": dict(Counter(r["outcome"] for r in rows))})
            rows = [r for r in cases if r["method"] == method and r["candidate_budget"] == budget]
            methods.append({"method": method, "candidate_budget": budget, "scenarios": 32,
                "mean_reward": mean([r["mean_reward"] for r in rows]), "mean_nominal_kw": DENOMINATOR * mean([r["mean_reward"] for r in rows]),
                "mean_seconds": mean([r["mean_seconds"] for r in rows]),
                **{k: sum(r[k] for r in rows) for k in ("positive", "zero", "no_witness")}})
    lookup = {(r["method"], r["candidate_budget"], r["scenario_id"]): r for r in cases}
    seed_lookup = {(r["method"], r["candidate_budget"], r["scenario_id"], r["seed"]): r for r in checked}
    paired, case_pairs, seed_pairs = [], [], []
    for name, label, lhs, rhs, level in COMPARISONS:
        for budget in BUDGETS:
            differences = []
            for sid in ids:
                left, right = lookup[lhs, budget, sid], lookup[rhs, budget, sid]
                delta = left["mean_reward"] - right["mean_reward"]
                differences.append(delta)
                case_pairs.append({"comparison": name, "candidate_budget": budget, "scenario_id": sid,
                                   "left_reward": left["mean_reward"], "right_reward": right["mean_reward"], "difference": delta})
                for seed in SEEDS:
                    a = seed_lookup[lhs, budget, sid, seed]
                    b = seed_lookup[rhs, budget, sid, None if rhs == "graph_repair" else seed]
                    seed_pairs.append({"comparison": name, "candidate_budget": budget, "scenario_id": sid, "seed": seed,
                        "left_reward": a["reward"], "right_reward": b["reward"], "difference": a["reward"] - b["reward"],
                        "left_outcome": a["outcome"], "right_outcome": b["outcome"], "graph_reference_not_independent": rhs == "graph_repair"})
            paired.append({"comparison": name, "label": label, "level": level, "candidate_budget": budget,
                "mean_difference": mean(differences), "mean_nominal_difference_kw": DENOMINATOR * mean(differences),
                "wins": sum(v > TOL for v in differences), "ties": sum(abs(v) <= TOL for v in differences), "losses": sum(v < -TOL for v in differences)})
    best_known = []
    for sid in ids:
        eligible = [r for r in actions if r["scenario_id"] == sid and r["witness_found"]]
        best = max(eligible, key=lambda r: (r["reward"], -r["action_id"])) if eligible else None
        roots = rule_metadata.get(sid, {}).get("ranked_roots")
        values = [r.get("reachable_nominal_kw") for r in roots] if isinstance(roots, list) else []
        valid_upper = bool(values) and rule_metadata[sid].get("nominal_total_kw") == DENOMINATOR and all(finite(v) and 0 <= v <= DENOMINATOR for v in values)
        upper, nominal = max(values) if valid_upper else None, best["reward"] * DENOMINATOR if best else None
        require(upper is None or nominal is None or nominal <= upper + TOL * DENOMINATOR, "Witness exceeds proven graph upper")
        best_known.append({"scenario_id": sid, "action_id": best["action_id"] if best else None,
            "stream_id": best["stream_id"] if best else None, "nominal_served_kw": nominal,
            "outcome": outcome(best is not None, best["reward"] if best else 0.), "graph_upper_kw": upper,
            "gap_kw": max(0., upper - nominal) if upper is not None and nominal is not None else None,
            "reaches_upper": best is not None and upper is not None and abs(upper - nominal) <= TOL * DENOMINATOR})
    return {"method_budget": methods, "per_scenario_method_budget": cases, "per_seed_method_budget": seeds,
            "paired_comparisons": paired, "paired_scenario_details": case_pairs, "paired_seed_details": seed_pairs,
            "best_known_by_scenario": best_known, "decision_checks": checked, "duplicate_requests": duplicates,
            "point_status_counts": {k: dict(v) for k, v in counts.items()}}


def render(stages, data, labels, costs, failure, report_run):
    methods8 = {r["method"]: r for r in data["method_budget"] if r["candidate_budget"] == 8}
    pairs8 = {r["comparison"]: r for r in data["paired_comparisons"] if r["candidate_budget"] == 8}
    main = [pairs8[r[0]] for r in COMPARISONS if r[4] == "primary"]
    findings = "；".join(f"{r['label']}{100*r['mean_difference']:+.4f}个百分点（{r['wins']}/{r['ties']}/{r['losses']}胜/平/负）" for r in main)
    positive = [r["label"] for r in main if r["mean_difference"] > TOL]
    judgment = ("本预算下出现正平均增量的比较为" + "、".join(positive) + "，仍需结合逐故障和逐种子的方向判断可重复性。" if positive else
                "预算8未观察到两个变体的PPO相对其BC起点有正平均增量，本轮不支持追加PPO已带来独立收益的主张。")
    graph = pairs8["conditional_ppo-minus-graph_repair"]
    best = data["best_known_by_scenario"]
    pools = list(labels["labels_by_scenario"].values())
    positive_labels = sum(bool(pool) and pool[0]["teacher_reward"] > 0 for pool in pools)
    zero_labels = sum(bool(pool) and pool[0]["teacher_reward"] == 0 for pool in pools)
    no_labels = sum(not pool for pool in pools)
    diverse_labels = sum(len(pool) > 1 for pool in pools)
    text = ["# NC34 规则专家、BC与PPO微调开发结果", "", "日期：2026-10-19。三个阶段均通过独立保存结果审核；本报告新增DSS、训练交互及优化器步均为0。", "",
        "## 主要发现", "", "预算8恢复均值：" + "；".join(f"{LABELS[m]}{100*methods8[m]['mean_reward']:.4f}%" for m in METHODS) + "。", "",
        "两个预定主比较为：" + findings + "。" + judgment, "",
        f"条件PPO相对图规则为{100*graph['mean_difference']:+.4f}个百分点（{graph['wins']}/{graph['ties']}/{graph['losses']}胜/平/负）。这检验固定1024交互后的增量，不能推断训练充分，也不将追加交互后的结果称同总离线成本优势。", "",
        "## 专家与训练", "",
        f"320个build故障统一图规则8候选，2560动作/12800点；其中128动作/640点来自已核验D036，新增2432动作/12160点，原目录未改。全部每拓扑最佳共{labels['per_topology_label_count']}个留档，其中最高奖励并列拓扑形成{labels['bc_label_count']}个主BC标签，覆盖{labels['labeled_scenario_count']}/320故障。正恢复标签故障{positive_labels}、仅零恢复标签故障{zero_labels}、无标签故障{no_labels}，{diverse_labels}个故障保留多个最高奖励拓扑。没有见证不补造标签，也不从故障人口删除。", "",
        "六个BC任务读取D036精确初始张量并fresh Adam，128 epoch；每epoch320故障各访问一次，均匀抽一个并列最佳拓扑，mask0行两损失置0并保留完整batch分母。六个PPO任务各从相同BC末态出发，另开fresh Adam，固定1024交互，共6144交互/30720点、192 rollout/768有效优化步；价值和奖励为即时名义恢复，无bootstrap。训练均值只作诊断，不用于模型排名。", "",
        "## 同预算一次性tune评价", "",
        "32故障在新增电气标签前从104 tune按固定SHA选择，所有训练完成后一次评价；正式104故障未解封。各方法八候选，1/2/4/8为共同轨迹前缀。先每故障平均三种子，再均全部32故障；规则只有一次确定性评价。正/零/无见证为种子平均故障数，合计32；无见证计分0不表示已执行零供电。分母为全68负荷名义1769 kW，actual功率另存。", "",
        "| 方法 | 预算 | 恢复均值 % | 计分名义 kW | 正/零/无见证（个/32） | 计入时间 s |", "|---|---:|---:|---:|---|---:|"]
    for r in data["method_budget"]:
        text.append(f"| {LABELS[r['method']]} | {r['candidate_budget']} | {100*r['mean_reward']:.4f} | {r['mean_nominal_kw']:.3f} | {r['positive']:.2f}/{r['zero']:.2f}/{r['no_witness']:.2f} | {r['mean_seconds']:.4f} |")
    if "training_evidence" in data:
        compute = data["training_evidence"]["compute"]
        training_rows = [f"BC实际{compute['bc_optimizer_steps']}优化步，PPO实际{compute['ppo_optimizer_steps']}步，合计{compute['total_optimizer_steps']}；12个固定模型重载/来源链均通过。各任务计算时间只作成本披露，不据训练损失排名。", "",
            "| 变体 | 种子 | 阶段 | 实际优化步 | 任务时间 s |", "|---|---:|---|---:|---:|"]
        for row in compute["jobs"]:
            training_rows.append(f"| {'状态' if row['variant']=='independent' else '条件'} | {row['seed']} | {row['stage']} | {row['optimizer_steps']} | {row['elapsed_seconds']:.3f} |")
        training_rows += ["", f"六个BC任务时间合计{compute['bc_job_seconds_sum']:.3f}秒，六个PPO任务合计{compute['ppo_job_seconds_sum']:.3f}秒；PPO任务时间含物理回调，与整个阶段时间口径不同，不能作为独立速度基准。", ""]
        index = text.index("## 同预算一次性tune评价")
        text[index:index] = training_rows
    text += ["", "## 预定配对比较", "", "胜/平/负以32个故障均值为单位，阈值1e-9仅用于统计；96个seed故障组不视为96独立故障。不作正式显著性或总体泛化结论。", "",
             "| 层级 | 比较 | 预算 | 平均差 pp | 名义差 kW | 胜/平/负 |", "|---|---|---:|---:|---:|---|"]
    for r in data["paired_comparisons"]:
        text.append(f"| {'主要' if r['level']=='primary' else '次要'} | {r['label']} | {r['candidate_budget']} | {100*r['mean_difference']:+.4f} | {r['mean_nominal_difference_kw']:+.3f} | {r['wins']}/{r['ties']}/{r['losses']} |")
    text += ["", "## 最佳已知与成本", "",
        f"联合每故障本轮128个已评价候选，{sum(r['outcome']=='POSITIVE_RESTORATION' for r in best)}个有正恢复、{sum(r['outcome']=='ZERO_RESTORATION' for r in best)}个仅零恢复、{sum(r['outcome']=='NO_WITNESS' for r in best)}个无见证。保存纯图上界的{sum(r['graph_upper_kw'] is not None for r in best)}个故障中，{sum(r['reaches_upper'] for r in best)}个最佳已知达到上界，其中{sum(r['reaches_upper'] and r['outcome']=='ZERO_RESTORATION' for r in best)}个为零上界。未达到图上界不证明仍有电气可行改善；本轮最佳已知不称全局最优。", "",
        "| 阶段 | 新点 | 复用点 | 收敛/不收敛 | 已知约束通过/失败/未知 | AC通过/失败/未知 | 隔离通过/失败/未知 | 显式Solve+无载=入口 | 阶段时间 s |", "|---|---:|---:|---|---|---|---|---|---:|"]
    for key in ("expert", "training", "evaluation"):
        stage = stages[key]
        s, c = stage["summary"], stage["commands"]["returned"]
        physical = stage.get("physical_summary", s)
        counts = physical["point_counts"]
        ac = "/".join(str(counts["ac"].get(label, 0)) for label in ("PASS", "FAIL", "UNKNOWN"))
        isolation = "/".join(str(counts["isolation"].get(label, 0)) for label in ("PASS", "FAIL", "UNKNOWN"))
        text.append(f"| {dict(expert='专家新增',training='BC/PPO阶段物理交互',evaluation='tune评价')[key]} | {physical['point_evaluations']} | {640 if key=='expert' else 0} | {counts['converged'].get('True',0)}/{counts['converged'].get('False',0)} | {counts['known_constraints'].get('True',0)}/{counts['known_constraints'].get('False',0)}/{counts['known_constraints'].get('None',0)} | {ac} | {isolation} | {c['explicit_solve_commands']}+{c['calcvoltagebases_entries']}={c['numerical_solution_entries']} | {s['elapsed_seconds']:.3f} |")
    actual = costs["actual_d037_total"]
    text += ["| 首次专家初始化路径错误（独立后审） | 0 | 0 | 未形成候选 | 未形成物理判定 | 未形成物理判定 | 未形成物理判定 | 2+1=3 | 不并入成功阶段时间 |",
        "| 未执行的superseded准备 | 0 | 0 | 未执行 | 未执行 | 未执行 | 未执行 | 0+0=0 | 不作为执行时间 |", "",
        f"三个完成阶段按原配置共63360新增点、另640复用点，计划190086显式Solve+63363无载=253449入口保持不变。另有首次专家启动的路径实现错误：0候选/0alpha点、21 API成对返回，实际2显式Solve+1无载=3入口；该失败已独立核验，原443文件保留，没有将初始化返回当物理/恢复见证。本轮实际累计因此为{actual['explicit_solve_commands']}显式Solve+{actual['calcvoltagebases_entries']}无载={actual['numerical_solution_entries']}入口。更早因文字修正未执行的superseded准备为0DSS，单列保留。", "",
        "BC软件更新本身零DSS，专家构建、BC和PPO离线成本不可隐藏；旧D036成本另见原报告。入口不等于引擎内部迭代或等价CPU工作。所有初始化、失败和重复计费，无物理缓存。", "",
        "全部启用运行证据及停用库存/零电流按既定规则核验；停用旧电压缓存不是运行证据，不宣称全部raw数组有限。AC、隔离与已知约束分列，零供电合格不计恢复成功；完整硬件/孤岛等具名物理边界保持。", "",
        "评价时间含候选生成、物理核验和证据写盘，模型已载入、静态图已准备，固定八候选的完整生成时间在早期预算也计入。训练和载入等离线成本分列，不称完整部署延迟或显著速度优势。", "",
        "每故障只有一次最终联合配置，搜索试算不代表现场顺序恢复。本轮为固定开发预算检验；若依据tune结果修改方法，该数据继续视作调试数据。当前未检验等总预算从头PPO，也未排除有效容量差异，不能把预训练或条件形式本身当独占创新。", "", "## 证据", ""]
    for key in ("expert", "training", "evaluation"):
        stage = stages[key]
        text.append(f"- {key}：[原始摘要](../runs/{stage['source'].name}/summary.json)、[独立审核](../runs/{stage['audit'].name}/verification.json)。")
    text.append(f"- [首次初始化失败后审](../runs/{failure['audit'].name}/verification.json)、[累计成本清单](../runs/{report_run.name}/cost_inventory.json)、[未执行准备](../runs/{SUPERSEDED_PREPARATION}/prepared.json)。")
    text += [f"- [完整汇总](../runs/{report_run.name}/aggregates.json)、[逐故障配对](../runs/{report_run.name}/paired_scenario_comparisons.json)、[逐种子配对](../runs/{report_run.name}/paired_seed_comparisons.json)、[最佳已知](../runs/{report_run.name}/best_known_by_scenario.json)。",
             f"- [冻结协议](../docs/nature34_imitation_protocol_v1.md)、[来源及统计口径](../runs/{report_run.name}/inputs.json)。", ""]
    return "\n".join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Complete fixed tune evaluation run name")
    parser.add_argument("--expert-run")
    parser.add_argument("--training-run")
    args = parser.parse_args()
    stages = {"expert": stage_gate("expert", args.expert_run), "training": stage_gate("training", args.training_run),
              "evaluation": stage_gate("evaluation", args.run)}
    failure = failure_gate()
    costs = cost_inventory(stages, failure)
    expert, training, evaluation = (stages[k] for k in ("expert", "training", "evaluation"))
    require(expert["config"] == training["config"] == evaluation["config"], "Three stages used different frozen configuration")
    selected = ["scenarios.json", "streams.json", "rule_metadata.json", "actions.jsonl", "decisions.jsonl", "training_gate.json", "scenario_selection.json"]
    evaluation["consumed_source"].update(consumed_gate(evaluation["source"], evaluation["pointer"]["completion_sha256"], selected))
    link = read(evaluation["source"] / "training_gate.json")
    require(link["training_run"] == training["source"].name and
            link["training_completion_sha256"] == training["pointer"]["completion_sha256"], "Evaluation is not linked to audited training")
    expert["consumed_source"].update(consumed_gate(expert["source"], expert["pointer"]["completion_sha256"], ["expert_labels.json"]))
    labels = read(expert["source"] / "expert_labels.json")
    require(labels["scenario_count"] == 320 and labels["candidate_count"] == 2560 and labels["alpha_point_count"] == 12800, "Expert scope differs")
    training_evidence = training_links(stages, labels)
    source = evaluation["source"]
    scenarios, streams, metadata = [read(source / f) for f in ("scenarios.json", "streams.json", "rule_metadata.json")]
    data = aggregate_evaluation(scenarios, streams, lines(source / "actions.jsonl"), lines(source / "decisions.jsonl"), metadata)
    data["training_evidence"] = training_evidence
    require(data["point_status_counts"] == evaluation["summary"]["point_counts"] == evaluation["verification"]["point_status_counts"],
            "Independent compact evaluation counts differ from audited raw")
    require(not REPORT.exists(), "Report exists: preserve the sealed report rather than overwrite it")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_nature34_imitation_report"
    output = ROOT / "runs" / stamp
    output.mkdir()
    inputs = {key: {"source_pointer": value["pointer"], "audit_pointer": value["audit_pointer"],
              "consumed_source_files": value["consumed_source"], "consumed_audit_files": value["consumed_audit"]} for key, value in stages.items()}
    save(output / "inputs.json", {"stages": inputs, "initialization_failure_audit": failure["pointer"],
         "consumed_failure_audit_files": failure["consumed"], "superseded_preparation": failure["superseded_preparation"],
         "seed_aggregation": "Mean3 seeds per fault, then mean32 faults",
         "paired_unit": "32 scenario means; graph occurs once per scenario", "denominator_kw": DENOMINATOR,
         "statistics_tie_tolerance": TOL, "raw_recheck": "Linked independent auditors; no report-time raw traversal",
         "python": sys.version, "new_powerflow_calls": 0, "new_training_interactions": 0, "new_optimizer_steps": 0})
    save(output / "aggregates.json", {k: data[k] for k in ("method_budget", "paired_comparisons", "point_status_counts")})
    for field, filename in (("per_scenario_method_budget", "per_scenario_method_budget.json"), ("per_seed_method_budget", "per_seed_method_budget.json"),
        ("paired_scenario_details", "paired_scenario_comparisons.json"), ("paired_seed_details", "paired_seed_comparisons.json"),
        ("best_known_by_scenario", "best_known_by_scenario.json"), ("decision_checks", "decision_checks.json"), ("duplicate_requests", "duplicate_requests.json")):
        save(output / filename, data[field])
    save(output / "expert_coverage.json", labels["scenario_coverage"])
    save(output / "training_evidence.json", training_evidence)
    save(output / "cost_inventory.json", costs)
    (output / "report_generator.py").write_bytes(Path(__file__).read_bytes())
    report = render(stages, data, labels, costs, failure, output)
    require(not any(name.split('.')[0] in {"dss", "opendssdirect", "torch"} for name in sys.modules), "Reporting imported an engine or learner")
    REPORT.write_text(report, encoding="utf-8")
    (output / "report_snapshot.md").write_text(report, encoding="utf-8")
    save(output / "summary.json", {"status": "COMPLETE_AUDITED_D037_DEVELOPMENT_REPORT", "source_run": source.name,
         "report": str(REPORT.relative_to(ROOT)), "report_sha256": sha(REPORT), "actions_rechecked": 4096,
         "compact_points_rechecked": 20480, "decisions_rechecked": 2048,
         "planned_numerical_solution_entries": 253449, "actual_d037_numerical_solution_entries": 253452,
         "new_powerflow_calls": 0, "new_training_interactions": 0, "new_optimizer_steps": 0})
    save(output / "completion_manifest.json", {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()})
    print(json.dumps({"report": str(REPORT), "report_run": stamp, "report_sha256": sha(REPORT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
