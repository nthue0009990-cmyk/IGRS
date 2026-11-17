"""Frozen build-only NC34 short training; no formal ranking or extra DSS probes.

Raw alpha points are written immediately to gzip. Only static metadata tables,
the current rollout, compact action records and the engine-event ledger stay in
memory. Importing this module never imports or initializes the DSS runtime.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.dont_write_bytecode = True
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

INPUT_KEYS = ("observation", "switch_forced_open", "context_table", "switch_support_mask", "load_forced_off_table")
CRITICAL_PHYSICAL_FILES = (
    "src/dnr/nature34_physical.py", "src/dnr/nature13_physical.py", "src/dnr/nature13_runtime.py",
    "src/dnr/restoration_constraints.py", "src/dnr/restoration_ac_audit.py", "src/dnr/restoration_isolation.py",
)
TEST_FILES = ("tests/test_nature34_policy_inputs.py", "tests/test_nature34_development_training.py",
              "tests/test_restoration_scenarios.py", "tests/test_restoration_supported_policy.py",
              "tests/test_restoration_constraints.py", "tests/test_restoration_ac_audit.py",
              "tests/test_restoration_isolation.py")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value):
    """Preserve nonfinite raw numbers explicitly instead of inventing zeros."""
    import numpy as np
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else {"nonfinite": str(float(value))}
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def numeric_lists(value):
    """Adapt engine arrays for pure checkers, retaining actual NaN/Inf values."""
    import numpy as np
    if isinstance(value, np.ndarray):
        return numeric_lists(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: numeric_lists(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [numeric_lists(item) for item in value]
    return value


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(value), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def append(stream, value) -> None:
    stream.write(json.dumps(json_ready(value), ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_manifest(folder: Path, expected: str) -> int:
    manifest = folder / "completion_manifest.json"
    if sha(manifest) != expected:
        raise ValueError("Completion manifest hash differs: " + str(folder))
    entries = read(manifest)
    for relative, digest in entries.items():
        path = folder / relative
        if not path.is_file():
            path = ROOT / relative
        if not path.is_file() or sha(path) != digest:
            raise ValueError("Manifest dependency differs: " + relative)
    return len(entries)


def validate_config(config: dict) -> None:
    fixed = {"training_seeds": [17, 41, 73], "variants": ["independent", "conditional"],
             "actions_per_seed_variant": 256, "observation_dim": 32, "switch_count": 9,
             "dispatch_load_count": 10, "context_dim": 32, "hidden": 64, "rollout_size": 32,
             "epochs": 2, "minibatch_size": 16, "alpha_grid": [0, .25, .5, .75, 1], "algorithm": 0,
             "bootstrap": False, "physical_cache_enabled": False, "pre_post_checkpoint_physical_evaluations": 0,
             "formal_holdout_numerical_access": False, "tune_numerical_access": False,
             "performance_ranking_allowed": False}
    if any(config.get(key) != value for key, value in fixed.items()):
        raise ValueError("Configuration changes the named short-training scope")
    interactions = len(config["training_seeds"]) * len(config["variants"]) * config["actions_per_seed_variant"]
    points = interactions * len(config["alpha_grid"])
    commands = config["expected_initialization_explicit_solve_commands"] + points * config["expected_explicit_solve_commands_per_point"]
    updates = interactions // config["rollout_size"] * config["epochs"] * (config["rollout_size"] // config["minibatch_size"])
    if (interactions, points, commands, updates) != (1536, 7680, 23042, 192):
        raise ValueError("Derived short-training budget differs")
    expected = {"max_training_interactions": interactions, "max_point_evaluations": points,
                "expected_total_explicit_solve_commands": commands, "max_total_explicit_solve_commands": commands,
                "expected_optimizer_steps": updates}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Declared budget differs from derived counts")


def schedule_seed(base: int, training_seed: int) -> int:
    raw = f"NC34-development-scenario-schedule-v1|{base}|{training_seed}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")


def make_schedules(build: list[dict], config: dict) -> dict:
    import numpy as np
    if len(build) != 320 or len({row["scenario_id"] for row in build}) != 320 or any(row["partition"] != "build" for row in build):
        raise ValueError("Schedules require all320 unique frozen build scenarios")
    result = {}
    for seed in config["training_seeds"]:
        derived = schedule_seed(config["scenario_schedule_base_seed"], seed)
        indices = np.random.default_rng(derived).integers(0, len(build), size=config["actions_per_seed_variant"])
        ids = [build[int(index)]["scenario_id"] for index in indices]
        result[str(seed)] = {"training_seed": seed, "schedule_seed": derived, "scenario_ids": ids,
                             "sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()}
    return result


def select_reward(points: list[dict]) -> dict:
    """All alpha failures remain in points; alpha0 is a valid possible witness."""
    if len(points) != 5 or sorted(row["alpha"] for row in points) != [0, .25, .5, .75, 1]:
        raise ValueError("Reward requires every frozen alpha exactly once")
    fractions = [row["nominal_restored_fraction"] for row in points]
    if any(not isinstance(value, (int, float)) or not 0 <= value <= 1 for value in fractions) or len(set(fractions)) != 1:
        raise ValueError("Nominal served demand must be finite and invariant across alpha")
    good = [row for row in points if row["converged"] and row["known_constraints_satisfied"] is True
            and row["ac_status"] == "PASS" and row["isolation_status"] == "PASS"]
    chosen = max(good, key=lambda row: row["alpha"]) if good else None
    fraction = chosen["nominal_restored_fraction"] if chosen else 0.
    if not isinstance(fraction, (int, float)) or not 0 <= fraction <= 1:
        raise ValueError("Nominal restoration reward is missing/nonfinite/out of range")
    return {"reward": float(fraction), "selected_alpha": chosen["alpha"] if chosen else None,
            "witness_point_id": chosen["point_id"] if chosen else None,
            "accepted_alphas": [row["alpha"] for row in good], "witness_found": bool(good)}


def count_commands(events: list[dict]) -> dict:
    python = sum(row["method"] == "Solution.Solve" for row in events)
    compiles = sum(row["method"] == "Text.Command" and str(row["args"][0]).lower().startswith("compile") for row in events)
    return {"python_solve_calls": python, "compile_commands": compiles, "explicit_solve_commands": python + compiles}


def require_reliable_point(converged: bool, isolation: dict, ac: dict) -> None:
    """Retain ordinary nonconvergence, stop structural or converged AC failures."""
    if isolation.get("violations") or isolation["status"] == "FAIL":
        raise AssertionError("Isolation structure certification failed; affected work stopped")
    if converged and (isolation["status"] != "PASS" or ac["status"] != "PASS"):
        raise AssertionError("Converged point failed AC/isolation certification; affected work stopped")


def prepare_inputs(config: dict, run: Path) -> tuple[dict, dict, dict, dict]:
    """Verify frozen evidence and construct static tables before importing DSS."""
    from dnr.nature34_policy_inputs import build_policy_inputs
    from dnr.restoration_scenarios import partition_content_sha256
    import networkx as nx
    import numpy as np

    pointer = read(ROOT / "artifacts/nature34_migration_saved_audit_latest.json")
    if (pointer["run"] != config["required_migration_audit_run"] or pointer["completion_sha256"] != config["required_migration_audit_completion_sha256"]
            or pointer["status"] not in config["accepted_migration_audit_status"]):
        raise ValueError("Required independent migration audit is not verified")
    audit_run = ROOT / "runs" / pointer["run"]
    audit_count = verify_manifest(audit_run, pointer["completion_sha256"])
    audit = read(audit_run / "verification.json")
    if (audit["status"] != pointer["status"] or audit["errors"] or audit["source_run"] != config["required_migration_run"]
            or audit["source_completion_sha256"] != config["required_migration_completion_sha256"]
            or audit["point_evaluations"] != 280 or audit["ac_status_counts"] != {"PASS": 280}
            or audit["isolation_status_counts"] != {"PASS": 280} or audit["new_powerflow_calls"] != 0):
        raise ValueError("Independent migration certificates do not meet the training gate")
    physical_run = ROOT / "runs" / config["required_migration_run"]
    physical_count = verify_manifest(physical_run, config["required_migration_completion_sha256"])
    physical_config = read(physical_run / "config.json")
    if physical_config != read(ROOT / "configs/nature34_migration_v1.json"):
        raise ValueError("Physical configuration differs from validated migration")
    for name in CRITICAL_PHYSICAL_FILES:
        if sha(ROOT / name) != sha(physical_run / "source" / name):
            raise ValueError("Validated physical implementation changed: " + name)
    plan_run = ROOT / "runs" / config["required_plan_run"]
    plan_count = verify_manifest(plan_run, config["required_plan_completion_sha256"])
    frozen = read(plan_run / "fault_partitions.json")
    if (partition_content_sha256(frozen) != frozen["content_sha256"] or frozen["content_sha256"] != config["expected_partition_content_sha256"]
            or frozen["partition_seed"] != config["partition_seed"]):
        raise ValueError("Frozen fault partition differs")
    source_pointer = read(ROOT / "artifacts/restoration_shortlist_sources_latest.json")
    if source_pointer["nature_commit"] != config["source_commit"] or physical_config["source_commit"] != config["source_commit"]:
        raise ValueError("Pinned author commit differs")
    source_count = verify_manifest(ROOT / "runs" / source_pointer["run"], source_pointer["completion_sha256"])
    static = read(plan_run / "author_static_metadata.json")
    build = [row for row in frozen["scenarios"] if row["partition"] == "build"]
    schedules = make_schedules(build, config)
    graph = nx.Graph()
    graph.add_nodes_from(static["nodes"])
    graph.add_edges_from(static["bus_edges"])
    tables, hashes = {}, {}
    for scenario in build:
        faults = scenario["fault_lines"]
        table = build_policy_inputs(graph, static["switch_edges"], [static["original_lines"][name] for name in faults],
                                    source_bus=static["source_bus"], dispatch_loads=static["dispatch_loads"],
                                    load_buses=static["dispatch_load_buses"], line_ids=frozen["line_ids"],
                                    fault_lines=faults, line_edges=static["original_lines"])
        key = scenario["scenario_id"]
        tables[key] = {name: table[name] for name in INPUT_KEYS}
        hashes[key] = {"input_sha256": table["input_sha256"], "array_hashes": table["array_hashes"],
                       "valid_switch_count": table["metadata"]["valid_switch_count"], "fault_lines": faults}
        target = run / "policy_inputs" / (key + ".npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **tables[key])
    save(run / "migration_audit_pointer.json", pointer)
    save(run / "migration_verification_gate.json", {"status": "PASS", "audit_manifest_entries": audit_count,
         "migration_manifest_entries": physical_count, "plan_manifest_entries": plan_count, "author_source_manifest_entries": source_count,
         "no_new_physical_calls": True, "audit_status": audit["status"]})
    save(run / "physical_config.json", physical_config)
    save(run / "source_pointer.json", source_pointer)
    save(run / "fault_partitions.json", frozen)
    save(run / "author_static_metadata.json", static)
    save(run / "scenario_schedules.json", schedules)
    save(run / "policy_input_hashes.json", hashes)
    return tables, {row["scenario_id"]: row for row in build}, schedules, physical_config


def batch_inputs(tables: dict, ids: list[str]):
    import numpy as np
    import torch
    return tuple(torch.from_numpy(np.stack([tables[key][name] for key in ids])) for name in INPUT_KEYS)


def software_probe(policy, tables: dict, ids: list[str]) -> dict:
    import torch
    with torch.no_grad():
        inputs = batch_inputs(tables, ids)
        result = policy.sample(*inputs, deterministic=True)
        reevaluated = policy.evaluate_actions(*inputs, result.actions)
        if not torch.equal(result.log_prob, reevaluated.log_prob) or not torch.isfinite(result.log_prob).all():
            raise AssertionError("Pure deterministic action density mismatch")
    return {"scenario_ids": ids, "actions": result.actions.tolist(), "log_prob": result.log_prob.tolist(),
            "entropy": result.entropy.tolist(), "value": result.value.tolist(),
            "physical_evaluations": 0, "performance_interpretation": "NOT_A_RESTORATION_PERFORMANCE_EVALUATION"}


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_nature34_development_training"
    run = ROOT / "runs" / stamp
    run.mkdir(parents=True)
    config = read(ROOT / "configs/nature34_development_training_v1.json")
    save(run / "config.json", config)
    events, actions, updates, jobs = [], [], [], []
    summary = {"run": stamp, "status": "PREPARING", "action_attempts": 0, "training_interactions": 0,
               "point_attempts": 0, "point_evaluations": 0, "optimizer_steps": 0, "tune_numerical_access": False,
               "formal_holdout_numerical_access": False, "performance_ranking_allowed": False,
               "full_hardware_feasibility": "NOT_ESTABLISHED", "physical_cache_hits": 0}
    active_pointer = ROOT / "artifacts/nature34_development_training_active.json"
    save(active_pointer, {**summary, "process_id": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat()})
    started = time.perf_counter()
    point_counts = {name: Counter() for name in ("converged", "known_constraints", "ac", "isolation")}
    try:
        validate_config(config)
        if not config["execution_allowed"]:
            raise ValueError("Named development execution is disabled")
        files = [*CRITICAL_PHYSICAL_FILES, "src/dnr/restoration_policy.py", "src/dnr/restoration_supported_policy.py",
                 "src/dnr/restoration_scenarios.py", "src/dnr/nature34_policy_inputs.py", __file__,
                 "configs/nature34_development_training_v1.json", "configs/nature34_migration_v1.json",
                 "docs/nature34_development_training_v1.md", *TEST_FILES]
        snapshots = {}
        for filename in files:
            path = Path(filename) if Path(filename).is_absolute() else ROOT / filename
            relative = str(path.relative_to(ROOT))
            target = run / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
            snapshots[relative] = sha(path)
        save(run / "source_hashes.json", snapshots)
        save(run / "dependencies.json", {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()})
        with (run / "tests.log").open("w", encoding="utf-8") as stream:
            result = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                                     "--basetemp", str(run / "pure_test_tmp"), *TEST_FILES],
                                    cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise AssertionError("Pure checks failed before DSS initialization")
        tables, scenarios, schedules, physical_config = prepare_inputs(config, run)
        save(run / "prephysical_freeze_manifest.json", {str(p.relative_to(run)): sha(p) for p in run.rglob("*") if p.is_file()})
        import numpy as np
        import torch
        from dnr.restoration_supported_policy import SupportedRestorationPolicy, ppo_supported_update
        from dnr.nature34_physical import Nature34PhysicalRuntime
        from dnr.restoration_constraints import check_known_constraints
        from dnr.restoration_ac_audit import audit_ac_snapshot
        from dnr.restoration_isolation import verify_isolation
        torch.set_num_threads(config["torch_threads"])
        torch.use_deterministic_algorithms(True)
        runtime = Nature34PhysicalRuntime(ROOT, run, physical_config, events)
        command_totals = count_commands(events)
        if command_totals["explicit_solve_commands"] != config["expected_initialization_explicit_solve_commands"]:
            raise AssertionError("Initialization command count differs; stop before training")
        save(run / "metadata.json", runtime.initial_metadata)
        save(run / "adaptation.json", runtime.metadata)
        save(run / "runtime_input_hashes.json", {str(p.relative_to(run / "work")): sha(p) for p in (run / "work").rglob("*") if p.is_file()})
        summary["status"] = "RUNNING_DEVELOPMENT_SHORT_TRAINING"
        initial_state_by_seed = {}
        with (run / "attempts.jsonl").open("w", encoding="utf-8") as attempts, gzip.open(run / "points.jsonl.gz", "wt", encoding="utf-8", compresslevel=6) as raw_stream, (run / "actions.jsonl").open("w", encoding="utf-8") as action_stream:
            for seed in config["training_seeds"]:
                schedule = schedules[str(seed)]["scenario_ids"]
                probe_ids = list(dict.fromkeys(schedule))[:config["pure_probe_scenario_count"]]
                for variant in config["variants"]:
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    policy = SupportedRestorationPolicy(32, 9, 10, 32, variant, hidden=config["hidden"])
                    state = {name: value.clone() for name, value in policy.state_dict().items()}
                    if seed in initial_state_by_seed:
                        if any(not torch.equal(value, initial_state_by_seed[seed][name]) for name, value in state.items()):
                            raise AssertionError("Variant initial parameters differ")
                    else:
                        initial_state_by_seed[seed] = state
                    optimizer = policy.make_optimizer(learning_rate=config["learning_rate"], betas=tuple(config["adam_betas"]),
                                                      eps=config["adam_eps"], weight_decay=config["adam_weight_decay"])
                    job = {"seed": seed, "variant": variant, "parameter_count": sum(p.numel() for p in policy.parameters()),
                           "scenario_schedule_sha256": schedules[str(seed)]["sha256"], "pre_probe": software_probe(policy, tables, probe_ids)}
                    reward_values = []
                    for batch_start in range(0, len(schedule), config["rollout_size"]):
                        ids = schedule[batch_start:batch_start + config["rollout_size"]]
                        inputs = batch_inputs(tables, ids)
                        rollout_checkpoint = run / "rollout_checkpoints" / f"seed{seed}_{variant}_{batch_start:03d}.pt"
                        rollout_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                        torch.save({"policy": policy.state_dict(), "seed": seed, "variant": variant, "batch_start": batch_start,
                                    "scenario_ids": ids, "scenario_schedule_sha256": schedules[str(seed)]["sha256"]}, rollout_checkpoint)
                        rollout_reference = {"rollout_checkpoint": str(rollout_checkpoint.relative_to(run)),
                                             "rollout_checkpoint_sha256": sha(rollout_checkpoint)}
                        with torch.no_grad():
                            sampled = policy.sample(*inputs)
                            checked = policy.evaluate_actions(*inputs, sampled.actions)
                            if not torch.equal(sampled.log_prob, checked.log_prob):
                                raise AssertionError("Sampled joint action density mismatch")
                        rewards = []
                        for local_index, scenario_id in enumerate(ids):
                            index = batch_start + local_index
                            scenario = scenarios[scenario_id]
                            action = sampled.actions[local_index].to(torch.int64).tolist()
                            identity = {"action_id": summary["action_attempts"], "seed": seed, "variant": variant,
                                        "schedule_index": index, "scenario_id": scenario_id, "partition": "build",
                                        "fault_lines": scenario["fault_lines"], "action": action, **rollout_reference,
                                        "policy_input_file": f"policy_inputs/{scenario_id}.npz",
                                        "policy_input_sha256": sha(run / "policy_inputs" / (scenario_id + ".npz"))}
                            append(attempts, {**identity, "event": "action_started", "engine_event_start": len(events)})
                            summary["action_attempts"] += 1
                            if summary["action_attempts"] > config["max_training_interactions"]:
                                raise AssertionError("Interaction budget exceeded")
                            compact = []
                            for alpha in config["alpha_grid"]:
                                point_id, before = summary["point_attempts"], len(events)
                                point_identity = {**identity, "point_id": point_id, "alpha": alpha, "algorithm": config["algorithm"], "engine_event_start": before}
                                append(attempts, {**point_identity, "event": "point_started"})
                                summary["point_attempts"] += 1
                                if (summary["point_attempts"] > config["max_point_evaluations"] or command_totals["explicit_solve_commands"] + config["expected_explicit_solve_commands_per_point"] > config["max_total_explicit_solve_commands"]):
                                    raise AssertionError("Physical command budget would be exceeded")
                                point_start = time.perf_counter()
                                row = None
                                try:
                                    row = numeric_lists(runtime.evaluate_point(action, {"name": scenario_id, "fault_lines": scenario["fault_lines"]}, alpha, config["algorithm"]))
                                    point_metadata = {**runtime.initial_metadata, **row["metadata_override"]}
                                    row["isolation"] = verify_isolation(row["raw_outputs"], point_metadata)
                                    row["known_constraints"] = check_known_constraints(row["raw_outputs"], point_metadata, **physical_config["constraint_tolerances"])
                                    row["ac_audit"] = audit_ac_snapshot(row["raw_outputs"], point_metadata, **physical_config["ac_audit_tolerances"])
                                except Exception as exc:
                                    if row is not None:
                                        append(raw_stream, {**row, **point_identity, "engine_event_end": len(events),
                                                            "certification_exception": repr(exc), "point_status": "CERTIFICATION_EXCEPTION_RETAINED"})
                                    append(attempts, {**point_identity, "event": "point_failed", "engine_event_end": len(events), "exception": repr(exc), "traceback": traceback.format_exc()})
                                    raise
                                row.update(point_identity, engine_event_end=len(events), elapsed_seconds=time.perf_counter() - point_start)
                                append(raw_stream, row)
                                summary["point_evaluations"] += 1
                                entry = {"point_id": point_id, "alpha": alpha, "converged": bool(row["raw_outputs"]["converged"]),
                                         "known_constraints_satisfied": row["known_constraints"]["known_constraints_satisfied"],
                                         "ac_status": row["ac_audit"]["status"], "isolation_status": row["isolation"]["status"],
                                         "nominal_restored_fraction": row["nominal_load_ledger"]["nominal_restored_fraction"]}
                                compact.append(entry)
                                for key, field in (("converged", "converged"), ("known_constraints", "known_constraints_satisfied"), ("ac", "ac_status"), ("isolation", "isolation_status")):
                                    point_counts[key][str(entry[field])] += 1
                                append(attempts, {**point_identity, "event": "point_completed", "engine_event_end": len(events), **entry})
                                point_commands = count_commands(events[before:])
                                for key, value in point_commands.items():
                                    command_totals[key] += value
                                if point_commands["explicit_solve_commands"] != config["expected_explicit_solve_commands_per_point"]:
                                    raise AssertionError("Actual point command count differs; evidence retained")
                                require_reliable_point(entry["converged"], row["isolation"], row["ac_audit"])
                                del row
                            decision = select_reward(compact)
                            record = {**identity, **decision, "points": compact, "old_log_prob": sampled.log_prob[local_index].item(),
                                      "old_value": sampled.value[local_index].item(), "old_exact_entropy": sampled.entropy[local_index].item(),
                                      "scenario_schedule_sha256": schedules[str(seed)]["sha256"]}
                            append(action_stream, record)
                            append(attempts, {**identity, "event": "action_completed", "engine_event_end": len(events), **decision})
                            actions.append(record)
                            rewards.append(decision["reward"])
                            reward_values.append(decision["reward"])
                            summary["training_interactions"] += 1
                        metric = ppo_supported_update(policy, optimizer, *inputs, sampled.actions, torch.tensor(rewards, dtype=torch.float32),
                                                      sampled.log_prob, sampled.value, clip_range=config["clip_range"],
                                                      value_coefficient=config["value_coefficient"], entropy_coefficient=config["entropy_coefficient"],
                                                      max_grad_norm=config["max_grad_norm"], normalize_advantage=config["normalize_advantage"],
                                                      epochs=config["epochs"], minibatch_size=config["minibatch_size"])
                        update = {"seed": seed, "variant": variant, "batch_start": batch_start, **rollout_reference, **metric}
                        updates.append(update)
                        summary["optimizer_steps"] += int(metric["optimizer_steps"])
                        save(run / "updates.json", updates)
                        save(run / f"progress_{seed}_{variant}_{batch_start:03d}.json", {**summary, **command_totals, "latest_update": update})
                        try:
                            save(active_pointer, {"run": stamp, "status": summary["status"], "process_id": os.getpid(),
                                 "progress_file": f"progress_{seed}_{variant}_{batch_start:03d}.json",
                                 "training_interactions": summary["training_interactions"], "point_evaluations": summary["point_evaluations"],
                                 **command_totals})
                        except OSError as exc:
                            append(attempts, {"event": "active_pointer_write_failed", "error": repr(exc),
                                              "progress_file": f"progress_{seed}_{variant}_{batch_start:03d}.json"})
                        print(json.dumps({"seed": seed, "variant": variant, "completed_actions_in_job": batch_start + len(ids),
                                          "point_evaluations": summary["point_evaluations"], "training_reward_mean": sum(rewards) / len(rewards)}), flush=True)
                    job["post_probe"] = software_probe(policy, tables, probe_ids)
                    checkpoint = run / "checkpoints" / f"seed{seed}_{variant}.pt"
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"policy": policy.state_dict(), "optimizer": optimizer.state_dict(), "seed": seed, "variant": variant,
                                "scenario_schedule_sha256": schedules[str(seed)]["sha256"], "config": config}, checkpoint)
                    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
                    restored = SupportedRestorationPolicy(32, 9, 10, 32, variant, hidden=config["hidden"])
                    restored.load_state_dict(loaded["policy"])
                    if any(not torch.equal(value, restored.state_dict()[name]) for name, value in policy.state_dict().items()):
                        raise AssertionError("Checkpoint tensor reload differs")
                    if software_probe(restored, tables, probe_ids) != job["post_probe"]:
                        raise AssertionError("Checkpoint pure output reload differs")
                    job.update(checkpoint_reload="PASS_ZERO_DSS", checkpoint_sha256=sha(checkpoint),
                               training_reward_mean=sum(reward_values) / len(reward_values), training_interactions=len(reward_values),
                               interpretation="On-policy development interaction rewards; no pre/post physical evaluation or method ranking")
                    jobs.append(job)
                    save(run / "jobs.json", jobs)
        counts = count_commands(events)
        if counts != command_totals:
            raise AssertionError("Incremental and complete command ledgers differ")
        if (summary["training_interactions"], summary["point_evaluations"], counts["explicit_solve_commands"], summary["optimizer_steps"]) != (1536, 7680, 23042, 192):
            raise AssertionError("Final logical/physical/update ledger differs from frozen budget")
        summary.update(status="COMPLETE_DEVELOPMENT_SHORT_TRAINING_PENDING_SAVED_AUDIT", checkpoint_reload_count=len(jobs))
    except Exception as exc:
        summary.update(status="DEVELOPMENT_TRAINING_STOPPED_RELIABILITY_REVIEW", error=repr(exc), traceback=traceback.format_exc())
        print(summary["traceback"], flush=True)
    finally:
        with gzip.open(run / "engine_calls.jsonl.gz", "wt", encoding="utf-8", compresslevel=6) as stream:
            for index, event in enumerate(events):
                stream.write(json.dumps(json_ready({"event_id": index, **event}), ensure_ascii=False, allow_nan=False) + "\n")
        summary.update(**count_commands(events), elapsed_seconds=time.perf_counter() - started,
                       point_status_counts={name: dict(values) for name, values in point_counts.items()},
                       engine_event_count=len(events), completed_jobs=len(jobs))
        save(run / "summary.json", summary)
        save(run / "actions.json", actions)
        save(run / "jobs.json", jobs)
        save(run / "updates.json", updates)
        save(run / "completion_manifest.json", {str(p.relative_to(run)): sha(p) for p in run.rglob("*") if p.is_file()})
        save(ROOT / "artifacts/nature34_development_training_latest.json", {"run": stamp, "status": summary["status"],
             "completion_sha256": sha(run / "completion_manifest.json"), "training_interactions": summary["training_interactions"],
             "point_evaluations": summary["point_evaluations"], "explicit_solve_commands": summary["explicit_solve_commands"]})
        save(active_pointer, {"run": stamp, "status": summary["status"], "process_id": os.getpid(),
             "training_interactions": summary["training_interactions"], "point_evaluations": summary["point_evaluations"],
             "explicit_solve_commands": summary["explicit_solve_commands"], "completion_sha256": sha(run / "completion_manifest.json")})
        print(json.dumps(json_ready(summary)), flush=True)
    return 0 if summary["status"].startswith("COMPLETE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
