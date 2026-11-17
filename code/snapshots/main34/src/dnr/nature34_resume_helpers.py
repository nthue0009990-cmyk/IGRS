"""Pure cursor, reuse and budget rules for the named NC34 continuation."""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import zlib

INPUT_KEYS = ("observation", "switch_forced_open", "context_table", "switch_support_mask", "load_forced_off_table")
ALPHAS = (0, .25, .5, .75, 1)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def validate_resume_budget(config):
    fixed = {"original_target_training_interactions": 1536, "original_target_alpha_points": 7680,
             "original_target_optimizer_steps": 192, "retained_completed_training_interactions": 475,
             "retained_completed_alpha_points": 2376, "retained_completed_rollout_updates": 14,
             "retained_completed_optimizer_steps": 56, "remaining_new_alpha_points": 5304,
             "new_initialization_explicit_solve": 2, "new_initialization_no_load_entry": 1,
             "per_alpha_explicit_solve": 3, "per_alpha_no_load_entry": 1,
             "max_new_explicit_solve_commands": 15914, "max_new_no_load_entries": 5305,
             "max_new_total_entries": 21219, "authorized_total_entry_upper": 30730,
             "combined_total_entry_lower_upper": [30726, 30730]}
    require(all(config.get(key) == value for key, value in fixed.items()), "Resume scope/budget differs from named plan")
    require(2376 + 5304 == 7680 and 2 + 5304 * 3 == 15914 and 1 + 5304 == 5305, "Internal resume arithmetic differs")
    require(config["old_prefix_total_entry_lower_upper"][1] + config["max_new_total_entries"] <= config["authorized_total_entry_upper"],
            "Combined conservative entry cap exceeded")


def pending_alpha_points(action_id):
    require(isinstance(action_id, int) and 475 <= action_id < 1536, "Action lies outside pending continuation")
    return [(action_id * 5 + index, alpha) for index, alpha in enumerate(ALPHAS) if action_id * 5 + index >= 2376]


def continuation_jobs():
    return [(17, "conditional", 192), (41, "independent", 0), (41, "conditional", 0),
            (73, "independent", 0), (73, "conditional", 0)]


def global_action_id(seed, variant, schedule_index):
    require(seed in (17, 41, 73) and variant in ("independent", "conditional") and 0 <= schedule_index < 256,
            "Unknown original training identity")
    return ((17, 41, 73).index(seed) * 2 + ("independent", "conditional").index(variant)) * 256 + schedule_index


def combine_points(reused, new, action_id):
    """Keep unsuccessful old labels unchanged; never filter reuse by outcome."""
    rows = [*reused, *new]
    require(len(rows) == 5, "Action lacks all five alpha outcomes")
    require(sorted((row["point_id"], row["alpha"]) for row in rows) == [(action_id * 5 + i, a) for i, a in enumerate(ALPHAS)],
            "Duplicated/missing alpha or incorrect logical point identity")
    return sorted(rows, key=lambda row: row["point_id"])


def verify_point_reuse(row, *, seed, variant, scenario_id, action, alpha, algorithm=0):
    expected = {"seed": seed, "variant": variant, "scenario_id": scenario_id, "action": action, "alpha": alpha, "algorithm": algorithm}
    require(all(row[key] == value for key, value in expected.items()), "Saved point identity is not reusable")


def assert_point_delta(before, after, *, initialization=False):
    expected = (2, 1) if initialization else (3, 1)
    actual = (after["explicit_solve_commands"] - before.get("explicit_solve_commands", 0),
              after["calcvoltagebases_entries"] - before.get("calcvoltagebases_entries", 0))
    require(actual == expected, "Actual explicit/no-load entries differ from frozen budget")


def verify_nested_exact(left, right, path="state"):
    import torch
    if isinstance(left, torch.Tensor):
        require(isinstance(right, torch.Tensor) and left.dtype == right.dtype and left.shape == right.shape and
                left.device == right.device and torch.equal(left, right), path + ": tensor mismatch")
    elif isinstance(left, Mapping):
        require(isinstance(right, Mapping) and left.keys() == right.keys(), path + ": keys mismatch")
        for key in left:
            verify_nested_exact(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        require(type(left) is type(right) and len(left) == len(right), path + ": sequence mismatch")
        for index, (a, b) in enumerate(zip(left, right)):
            verify_nested_exact(a, b, f"{path}.{index}")
    else:
        require(type(left) is type(right) and left == right, path + ": value mismatch")


def load_frozen_learner(original):
    namespace = "_nc34_resumed_frozen"
    package = types.ModuleType(namespace)
    package.__path__ = [str(original / "source/src/dnr")]
    sys.modules[namespace] = package
    learner = importlib.import_module(namespace + ".restoration_supported_policy")
    spec = importlib.util.spec_from_file_location("_nc34_resumed_original_runner", original / "source/scripts/run_nature34_development_training.py")
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    require(tuple(runner.INPUT_KEYS) == INPUT_KEYS, "Frozen input order changed")
    return learner, runner


def complete_gzip_lines(path):
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    pending = b""
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            pending += decoder.decompress(block)
            lines = pending.split(b"\n")
            pending = lines.pop()
            for line in lines:
                if line:
                    yield json.loads(line)
    pending += decoder.flush()
    lines = pending.split(b"\n")
    require(not lines[-1], "Unexpected incomplete JSON tail")
    for line in lines[:-1]:
        if line:
            yield json.loads(line)


def failure_record(exc):
    """Also describe KeyboardInterrupt/SystemExit; do not claim completion."""
    return {"status": "RESUME_STOPPED_PARTIAL_EVIDENCE_SEALED", "exception_type": type(exc).__name__,
            "error": repr(exc), "completed": False}
