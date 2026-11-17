"""Prepare a build-only CLPU pilot without importing or initializing DSS.

Source identity and every consumed file are checked against sealed manifests.
This is an input/code snapshot, deliberately not the final prephysical freeze:
the caller must freeze the completed implementation before any simulation.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
sys.dont_write_bytecode = True

from dnr.clpu_curves import DEFAULT_SPLIT_DOMAIN, SOURCE_ZIP_SHA256, load_events, split_events
from report_nature34_imitation import consumed_gate


SOURCE_RUN = "20261023T015932_921283Z_nature34_scheduler"
SOURCE_COMPLETION_SHA256 = "b1f4749d3da5b32094f9ba5518e67d544b8d764ebf93528d7371a0514b563d5e"
AUDIT_RUN = "20261023T021651_750531Z_nature34_scheduler_audit"
AUDIT_COMPLETION_SHA256 = "3900d46c021faf972dec37d63a012263db07520eae82ae72054efbf590617f48"
CURVE_ZIP = ROOT / "data/raw/literature/clpu_design/20261024T015725_552682Z/hachmann2022_clpu_data.zip"
FAULT_SPLIT_DOMAIN = "clpu-pilot-faults-v1"
CONDITION_ID = "author_daily_lower_median"
ALPHA = 0.5
SOURCE_INPUTS = (
    "training_scenarios.json", "author_static_metadata.json", "load_reference.json",
    "conditions.json", "physical_config.json", "operating_reference.json",
    "base_load_properties.json", "command_inventory.json", "source_pointer.json",
    "metadata.json", "adaptation.json", "author_input_precheck.json",
)
DEFAULT_SOURCE_NAMES = (
    "scripts/prepare_nature34_clpu_pilot.py", "src/dnr/clpu_curves.py",
    "scripts/report_nature34_imitation.py",
)


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _sha(path: Path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare(config: dict | str | Path, source_names=None) -> Path:
    """Create a new input-only run from a configuration mapping or JSON path.

    Required ``phase`` is ``calibration`` or ``main``.  Both receive the same
    fixed eight build faults and four training events; a calibration runner may
    use its separately declared subset.  Selection never reads old outcomes.
    ``source_names`` adds current project-relative code snapshots; no final
    ``prepared.json`` or prephysical freeze manifest is written here.
    """
    config_path = None if isinstance(config, dict) else Path(config).resolve()
    cfg = deepcopy(config) if isinstance(config, dict) else _read(config_path)
    _require(isinstance(cfg, dict), "Configuration must be a JSON object")
    _require(cfg.get("phase") in ("calibration", "main"), "Only calibration/main preparation is allowed")
    _require(cfg.get("condition_id", CONDITION_ID) == CONDITION_ID, "Fixed condition differs")
    _require(cfg.get("alpha", ALPHA) == ALPHA, "Fixed alpha differs")
    for key in ("test_access", "tune_access", "formal_holdout_numerical_access", "tune_numerical_access"):
        _require(not cfg.get(key, False), "Forbidden numerical data access: " + key)
    _require(cfg.get("fault_selection_domain", FAULT_SPLIT_DOMAIN) == FAULT_SPLIT_DOMAIN, "Fault hash domain differs")
    _require(cfg.get("event_split_domain", DEFAULT_SPLIT_DOMAIN) == DEFAULT_SPLIT_DOMAIN, "Event hash domain differs")
    _require(cfg.get("scenario_count", 8) == 8 and cfg.get("event_count", 4) == 4, "Fixed cohort size differs")

    source, audit = ROOT / "runs" / SOURCE_RUN, ROOT / "runs" / AUDIT_RUN
    source_hashes = consumed_gate(source, SOURCE_COMPLETION_SHA256, list(SOURCE_INPUTS))
    audit_hashes = consumed_gate(audit, AUDIT_COMPLETION_SHA256, ["verification.json"])
    verification = _read(audit / "verification.json")
    _require(verification.get("status") == "VERIFIED_SCHEDULER" and verification.get("errors") == [], "Source audit did not pass")
    _require(verification.get("source_run") == SOURCE_RUN and verification.get("source_completion_sha256") == SOURCE_COMPLETION_SHA256, "Source audit identity differs")
    _require(not verification.get("formal_holdout_numerical_access") and not verification.get("tune_numerical_access"), "Source audit reports forbidden numerical access")

    scenarios = _read(source / "training_scenarios.json")
    ids = [row["scenario_id"] for row in scenarios]
    _require(len(scenarios) == len(set(ids)) == 16, "Source must contain exactly sixteen unique training faults")
    _require(all(row.get("partition") == "build" and row.get("fault_order") == 2 and len(set(row["fault_lines"])) == 2 and row["scenario_id"].startswith("N2__") for row in scenarios), "Source includes a non-build or non-N2 fault")
    ordered = sorted(scenarios, key=lambda row: hashlib.sha256(f"{FAULT_SPLIT_DOMAIN}|{row['scenario_id']}".encode("utf-8")).hexdigest())
    selected_scenarios = ordered[:8]
    if "scenario_ids" in cfg:
        _require(cfg["scenario_ids"] == [row["scenario_id"] for row in selected_scenarios], "Configured faults differ from fixed hash selection")

    conditions = _read(source / "conditions.json")
    matching_conditions = [row for row in conditions if row["condition_id"] == CONDITION_ID]
    _require(len(matching_conditions) == 1, "Missing or duplicated median operating condition")
    condition = matching_conditions[0]
    _require(len(condition["loads"]) == 68, "Expected all 68 load demands")
    _require(_sha(CURVE_ZIP) == SOURCE_ZIP_SHA256, "Public CLPU ZIP changed")
    events = load_events(CURVE_ZIP)
    partitions = split_events(events)
    selected_events = partitions["train"][:4]
    _require([event["event_id"] for event in selected_events] == [20, 3, 6, 30], "Unexpected fixed curve cohort")
    if "event_ids" in cfg:
        _require(cfg["event_ids"] == [event["event_id"] for event in selected_events], "Configured event IDs differ")

    pointer = _read(source / "source_pointer.json")
    _require(_read(ROOT / "artifacts/restoration_shortlist_sources_latest.json") == pointer, "Live author source pointer differs from sealed source")
    author_directory = (ROOT / pointer["derived_directory"] / "nature_author_source/RL_Training/Environments/DSSdirect_34bus_loadandswitching").resolve()
    _require(author_directory.is_relative_to(ROOT), "Author input path outside project")
    author_expected = _read(source / "author_input_precheck.json")
    author_actual = {}
    for name, expected in author_expected.items():
        target = (author_directory / name).resolve()
        _require(target.is_relative_to(author_directory), "Author input path escapes its directory")
        _require(_sha(target) == expected, "Author model input changed: " + name)
        author_actual[name] = expected

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run = ROOT / "runs" / f"{stamp}_nature34_clpu_{cfg['phase']}"
    run.mkdir(exist_ok=False)
    _write(run / "config.json", cfg)
    _write(run / "dependencies.json", {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()})
    _write(run / "python_environment.json", {"executable": sys.executable, "version": sys.version, "sha256": _sha(Path(sys.executable))})
    for name in SOURCE_INPUTS:
        (run / "source_inputs").mkdir(exist_ok=True)
        shutil.copyfile(source / name, run / "source_inputs" / name)
        shutil.copyfile(source / name, run / name)
        _require(_sha(run / "source_inputs" / name) == source_hashes[name] and _sha(run / name) == source_hashes[name], "Copied source input differs: " + name)
    for name in author_actual:
        destination = run / "author_model_inputs" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(author_directory / name, destination)
        _require(_sha(destination) == author_actual[name], "Copied author input differs: " + name)
    _write(run / "selected_scenarios.json", selected_scenarios)
    _write(run / "selected_events.json", selected_events)
    _write(run / "condition.json", condition)
    _write(run / "event_split.json", {key: [event["event_id"] for event in rows] for key, rows in partitions.items()})
    _write(run / "selection.json", {
        "fault_selection_domain": FAULT_SPLIT_DOMAIN,
        "fault_sort_rule": "ascending sha256(domain + '|' + scenario_id), first eight of the sealed sixteen build N2 faults",
        "ordered_source_scenario_ids": [row["scenario_id"] for row in ordered],
        "selected_scenario_ids": [row["scenario_id"] for row in selected_scenarios],
        "event_split_domain": DEFAULT_SPLIT_DOMAIN,
        "event_selection": "first four of the nineteen hash-ordered complete training events",
        "selected_event_ids": [event["event_id"] for event in selected_events],
        "operating_condition_id": CONDITION_ID, "alpha": ALPHA,
        "outcome_dependent_filtering": False,
        "failures_and_zero_headroom_scenarios": "retain all; no replacement",
    })
    _write(run / "source_provenance.json", {
        "scheduler": {"run": SOURCE_RUN, "completion_sha256": SOURCE_COMPLETION_SHA256, "consumed_files": source_hashes},
        "scheduler_audit": {"run": AUDIT_RUN, "completion_sha256": AUDIT_COMPLETION_SHA256, "consumed_files": audit_hashes, "status": verification["status"]},
        "clpu_source": {"path": str(CURVE_ZIP.relative_to(ROOT)), "sha256": SOURCE_ZIP_SHA256, "curve_integrity_checked": "all 31 complete events; only selected four are supplied as pilot conditions", "csv_parser": "src/dnr/clpu_curves.py"},
        "author_model_directory": str(author_directory.relative_to(ROOT)), "author_model_input_sha256": author_actual,
        "configuration_source": None if config_path is None else {"path": str(config_path), "sha256": _sha(config_path)},
        "source_results_imported": False,
        "sealed_manifest_scope": "completion manifest identity plus each explicitly consumed input only; no scan of unrelated result files",
    })
    _write(run / "source_verification.json", verification)
    for name in sorted(set(source_names or ()) | set(DEFAULT_SOURCE_NAMES)):
        source_path = (ROOT / name).resolve()
        _require(source_path.is_relative_to(ROOT) and source_path.is_file(), "Invalid source snapshot: " + str(name))
        destination = run / "source" / source_path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination)
    _write(run / "input_snapshot_manifest.json", {str(path.relative_to(run)): _sha(path) for path in sorted(run.rglob("*")) if path.is_file()})
    _write(run / "input_preparation.json", {
        "status": "INPUTS_PREPARED_FINAL_IMPLEMENTATION_FREEZE_PENDING_NO_DSS",
        "run": run.name, "phase": cfg["phase"],
        "input_snapshot_manifest_sha256": _sha(run / "input_snapshot_manifest.json"),
        "selected_scenarios": len(selected_scenarios), "selected_events": len(selected_events),
        "contexts": len(selected_scenarios) * len(selected_events),
        "new_powerflow_calls": 0, "new_training_interactions": 0,
        "formal_holdout_numerical_access": False, "tune_numerical_access": False,
        "final_prephysical_freeze": False,
    })
    return run.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/nature34_clpu_pilot_v1.json")
    arguments = parser.parse_args()
    run = prepare(arguments.config)
    print(json.dumps({"run": str(run), "status": "INPUTS_PREPARED_FINAL_IMPLEMENTATION_FREEZE_PENDING_NO_DSS"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
