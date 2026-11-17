"""One cold D059 branch-search episode with durable physics and policy records.

This entry point does not fit a policy, choose checkpoints, or retry a task. The
controller supplies a frozen phase/context specification and (when applicable)
the exact checkpoint hash. Original SnapshotOracle remains the only simulator
adapter. Every selected branch and both input representations are retained.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
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
from run_nature34_clpu_pilot import SnapshotOracle, make_env

METHODS = {"greedy", "beam", "supervised", "bc_raw", "ppo_raw", "bc_associated", "ppo_associated"}
RULE_METHODS = {"greedy", "beam"}
_WORKER_ENTERED = False


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _context_key(value):
    return value["scenario_id"], value["event_id"]


def validate_spec(run, spec):
    """Phase admission is by exact fault/event pairs, not the sixteen-fault inventory."""
    require(isinstance(spec, dict), "Cell specification must be an object")
    method, phase = spec.get("method"), spec.get("phase")
    require(method in METHODS, "Unknown branch method")
    require(phase in ("teacher", "reference", "train", "evaluation", "eval"), "Unknown phase")
    require(isinstance(spec.get("seed"), int) and not isinstance(spec["seed"], bool), "Missing integer seed")
    if "sampling_seed" in spec:
        require(type(spec["sampling_seed"]) is int and 0 <= spec["sampling_seed"] < 2**63,
                "Invalid episode sampling seed")
    require(isinstance(spec.get("deterministic"), bool), "Explicit deterministic/sample setting required")
    require(spec.get("representation") in ("raw", "associated"), "Unknown representation")
    if method.endswith("_raw"):
        require(spec["representation"] == "raw", "Raw method representation mismatch")
    if method.endswith("_associated"):
        require(spec["representation"] == "associated", "Associated method representation mismatch")
    expected_budget = 1024 if phase == "reference" else 256
    require(type(spec.get("point_budget")) is int and spec["point_budget"] == expected_budget,
            "Cell point budget differs from its frozen phase")
    if phase in ("teacher", "reference"):
        require(method in RULE_METHODS, "Teacher/reference requires the frozen rule")
    if phase == "train":
        require(method in ("ppo_raw", "ppo_associated") and not spec["deterministic"],
                "Physical training rollouts require a stochastic PPO method")
    if phase in ("evaluation", "eval", "teacher", "reference"):
        require(spec["deterministic"], "Teacher/reference/evaluation must be deterministic")
    cohort_file = "evaluation_contexts.json" if phase in ("evaluation", "eval") else "training_contexts.json"
    require(_context_key(spec) in {_context_key(row) for row in read(run / cohort_file)},
            "Fault/event pair is outside this phase's frozen cohort")
    if phase == "reference" and (run / "reference_contexts.json").exists():
        require(_context_key(spec) in {_context_key(row) for row in read(run / "reference_contexts.json")},
                "Context is outside frozen reference subset")
    if method in RULE_METHODS:
        require(not spec.get("checkpoint") and not spec.get("checkpoint_sha256"), "Rule must not load a policy")
        return None
    require(isinstance(spec.get("checkpoint"), str) and bool(spec["checkpoint"]), "Policy checkpoint is required")
    checkpoint = (run / spec["checkpoint"]).resolve()
    require(checkpoint.is_relative_to(run.resolve()) and checkpoint.is_file(), "Checkpoint must exist within this run")
    require(isinstance(spec.get("checkpoint_sha256"), str) and len(spec["checkpoint_sha256"]) == 64,
            "Checkpoint identity is required")
    require(sha(checkpoint) == spec["checkpoint_sha256"], "Checkpoint hash changed")
    return checkpoint


def verify_worker_inputs(run, cell, spec):
    require(run.parent == (ROOT / "runs").resolve(), "Run is outside project runs")
    require(cell != run and cell.is_relative_to(run), "Cell is outside its run")
    require(read(run / "config.json").get("decision") == "D059", "Expected D059 run")
    prepared = read(run / "prepared.json")
    require(prepared.get("status") == "FROZEN_PREPARED_NO_DSS", "Run has not been frozen")
    require(sha(run / "prephysical_freeze_manifest.json") == prepared["freeze_sha256"], "Freeze identity changed")
    for name, digest in read(run / "prephysical_freeze_manifest.json").items():
        target = (run / name).resolve()
        require(target.is_relative_to(run) and sha(target) == digest, "Frozen input changed: " + name)
    for name in prepared["source_files"]:
        require(sha(ROOT / name) == sha(run / "source" / name), "Frozen implementation changed: " + name)
    require(dependencies() == read(run / "dependencies.json"), "Frozen dependencies changed")
    require(sha(Path(sys.executable)) == read(run / "python_environment.json")["sha256"], "Python interpreter changed")
    return validate_spec(run, spec)


def _observation(search, representation):
    observation = numeric_lists(search.observation(representation))
    require(isinstance(observation, dict), "Observation must retain named features")
    candidates, identifiers, mask = (observation.get("candidate_features"),
                                    observation.get("branch_ids"), observation.get("mask"))
    require(isinstance(candidates, list) and isinstance(identifiers, list) and isinstance(mask, list),
            "Missing candidate features, identities, or legal-action mask")
    require(len(candidates) == len(identifiers) == len(mask) and len(mask) > 0, "Observation dimensions differ")
    require(all(type(value) is bool for value in mask) and any(mask), "Invalid or empty legal-action mask")
    require(len(set(identifiers)) == len(identifiers), "Duplicate candidate branch identities")
    return observation


def collect_episode(search, spec, transition_callback, decision_callback, policy=None, generator=None):
    """Policy/rule interaction only; injectable fake search makes tests solver-free.

    The returned count is actual selected search decisions, not physical points
    and not PPO optimizer updates. A started-but-interrupted decision has no
    fabricated reward or transition, and remains in the decision journal.
    """
    count = 0
    while not search.done:
        if spec.get("deadline_utc_epoch") is not None and time.time() >= spec["deadline_utc_epoch"]:
            raise TimeoutError("Shared study deadline reached before the next branch selection")
        raw, associated = _observation(search, "raw"), _observation(search, "associated")
        require(raw["branch_ids"] == associated["branch_ids"] and raw["mask"] == associated["mask"],
                "Representation changed the available branches")
        observation = raw if spec["representation"] == "raw" else associated
        mask = observation["mask"]
        if spec["method"] in RULE_METHODS:
            action = search.rule_action(spec["method"])
            logprob = value = None
        else:
            require(policy is not None, "Missing policy")
            selected = policy.act(observation, mask, deterministic=spec["deterministic"], generator=generator)
            action, logprob, value = selected["action"], selected["logprob"], selected["value"]
            # Tensor/scalar conversion is representation-only; no rounding or imputation.
            action = action.item() if hasattr(action, "item") else action
            logprob = float(logprob)
            value = float(value)
            require(math.isfinite(logprob) and math.isfinite(value), "Nonfinite policy evidence")
        require(type(action) is int and 0 <= action < len(mask) and mask[action], "Policy selected an unavailable branch")
        row = {"transition_id": count, "obs": observation, "obs_raw": raw, "obs_associated": associated,
               "mask": mask, "action": action, "branch_id": observation["branch_ids"][action],
               "logprob": logprob, "value": value, "checkpoint_sha256": spec.get("checkpoint_sha256"),
               "method": spec["method"], "representation": spec["representation"],
               "seed": spec["seed"], "sampling_seed": spec.get("sampling_seed", spec["seed"]),
               "oracle_before": {"requests": search.oracle.requests, "points": search.oracle.points}}
        decision_callback({**row, "status": "STARTED"})
        response = search.step(action)
        require(isinstance(response, dict) and "reward" in response and "done" in response,
                "Branch step must return reward/done/info")
        reward = float(response["reward"])
        require(math.isfinite(reward) and type(response["done"]) is bool, "Nonfinite reward or malformed terminal flag")
        require(response["done"] == bool(search.done), "Step terminal flag differs from search state")
        # The common engine exposes costs/gains as named top-level fields; keep
        # the entire response, including any future nested info, without loss.
        info = numeric_lists({key: value for key, value in response.items() if key not in ("reward", "done")})
        row.update(reward=reward, done=response["done"], info=info,
                   oracle_after={"requests": search.oracle.requests, "points": search.oracle.points})
        transition_callback(row)
        decision_callback({"transition_id": count, "status": "COMPLETED", "action": action,
                           "branch_id": row["branch_id"], "reward": reward, "done": row["done"], "info": row["info"],
                           "oracle_before": row["oracle_before"], "oracle_after": row["oracle_after"]})
        count += 1
        require(count <= 5000, "Branch search exceeded frozen generation cap")
    return count


def run_worker(run, cell):
    global _WORKER_ENTERED
    run, cell = Path(run).resolve(), Path(cell).resolve()
    require(not _WORKER_ENTERED, "Each task requires a fresh worker process")
    _WORKER_ENTERED = True
    require(not any((cell / name).exists() for name in (
        "worker_started.json", "result.json", "error.json", "command_ledger.jsonl", "points.jsonl.gz")),
        "Cell already started; automatic retry is forbidden")
    spec, config, started = read(cell / "spec.json"), read(run / "config.json"), time.perf_counter()
    try:
        checkpoint = verify_worker_inputs(run, cell, spec)
        # Exclusive creation also prevents two cold processes from owning one cell.
        with (cell / "worker_started.json").open("x", encoding="utf-8") as stream:
            append(stream, {"utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
                            "spec_sha256": sha(cell / "spec.json"), "checkpoint_sha256": spec.get("checkpoint_sha256")})
        policy = generator = None
        if checkpoint is not None:
            import torch
            from dnr.nature34_clpu_branch_learning import load_policy
            torch.set_num_threads(1)
            policy = load_policy(checkpoint, representation=spec["representation"])
            require(bool(policy.associated) == (spec["representation"] == "associated"), "Loaded policy representation differs")
            generator = torch.Generator(device="cpu").manual_seed(spec.get("sampling_seed", spec["seed"]))
        from dnr.nature34_clpu_branch import BranchConfig, BranchSearch
        env = make_env(run, spec)
        save(cell / "initial_state.json", env.initial.to_dict())
        branch_cfg = dict(config.get("branch_search", config.get("search", {})))
        branch_cfg["point_budget"] = spec["point_budget"]
        if spec.get("deadline_utc_epoch") is not None and time.time() >= spec["deadline_utc_epoch"]:
            raise TimeoutError("Shared study deadline reached before physical initialization")
        with gzip.open(cell / "plans.jsonl.gz", "wt", encoding="utf-8", compresslevel=3) as plans, \
                gzip.open(cell / "transitions.jsonl.gz", "wt", encoding="utf-8", compresslevel=3) as transitions, \
                gzip.open(cell / "decisions.jsonl.gz", "xt", encoding="utf-8", compresslevel=3) as decisions, \
                SnapshotOracle(run, cell, env, spec["point_budget"]) as oracle:
            search = BranchSearch(env, oracle, BranchConfig(**branch_cfg),
                                  record_callback=lambda record: append(plans, numeric_lists(record)))
            decision_count = collect_episode(search, spec, lambda row: append(transitions, row),
                                             lambda row: append(decisions, row), policy, generator)
            result = numeric_lists(search.result())
        if checkpoint is not None:
            require(sha(checkpoint) == spec["checkpoint_sha256"], "Checkpoint changed during the episode")
        oracle_summary = read(cell / "oracle_summary.json")
        require(oracle_summary["points"] <= spec["point_budget"], "Physical budget exceeded")
        result.update(spec=spec, elapsed_seconds=time.perf_counter() - started, oracle=oracle_summary,
                      search_decisions=decision_count,
                      training_interactions=decision_count if spec["phase"] == "train" else 0,
                      optimizer_updates=0, trajectory_file="transitions.jsonl.gz",
                      checkpoint_sha256=spec.get("checkpoint_sha256"),
                      formal_holdout_numerical_access=False, tune_numerical_access=False)
        save(cell / "result.json", result)
        save(cell / "completion_manifest.json", {str(path.relative_to(cell)): sha(path)
             for path in sorted(cell.rglob("*")) if path.is_file()
             and path.name not in ("completion_manifest.json", "stdout.log", "stderr.log")})
        return 0
    except BaseException as exc:
        save(cell / "error.json", {"status": "STOPPED_RELIABILITY_OR_IMPLEMENTATION_ERROR",
             "error": repr(exc), "traceback": traceback.format_exc(), "spec": spec,
             "elapsed_seconds": time.perf_counter() - started,
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
