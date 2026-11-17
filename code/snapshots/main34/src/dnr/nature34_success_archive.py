"""Build-only, deterministic success labels; no simulator or optimizer."""
from collections import defaultdict
from copy import deepcopy
from dnr.restoration_feedback import binary_action, unit_fraction


def build_archive(rows, actions, cells, source_run):
    """Retain a catalog of every attempt; select one best witnessed action/cell."""
    keys = [(r["scenario_id"], r["condition_id"]) for r in cells]
    if len(set(keys)) != len(keys) or len(rows) != len(actions):
        raise ValueError("Duplicate cells or incomplete source")
    groups = defaultdict(list)
    catalog = []
    for index, (row, action) in enumerate(zip(rows, actions)):
        key = (row["scenario_id"], row["condition_id"])
        if key not in keys or action["partition"] != "build":
            raise ValueError("Archive contains an unexpected partition/cell")
        if row["action_id"] != index or action["action_id"] != index:
            raise ValueError("Noncontiguous source identities")
        if any(row[k] != action[k] for k in ("scenario_id", "condition_id", "action", "seed")):
            raise ValueError("Transition/action identity differs")
        binary_action(row["action"])
        receipt = row["receipt"]
        witnessed = any(r["known_witness"] for r in receipt["feedback"])
        if witnessed != row["witness_found"] or witnessed != action["witness_found"]:
            raise ValueError("Witness mismatch")
        if len(receipt["feedback"]) != 5 or len(action["points"]) != 5:
            raise ValueError("Incomplete paid alpha grid")
        if witnessed != (action["witness_point_id"] is not None):
            raise ValueError("Missing/false witness point")
        fraction = unit_fraction(receipt["requested_kw"] / receipt["denominator_kw"])
        item = {"scenario_id": key[0], "condition_id": key[1], "source_run": source_run,
                "source_action_id": index, "source_seed": row["seed"], "action": row["action"],
                "source_step": row["step"], "source_episode_index": row["episode_index"],
                "witness_found": witnessed, "requested_kw": receipt["requested_kw"],
                "denominator_kw": receipt["denominator_kw"], "restored_fraction": fraction if witnessed else None,
                "witness_point_id": action["witness_point_id"], "selected_alpha": action["selected_alpha"],
                "point_ids": [p["point_id"] for p in action["points"]]}
        catalog.append(item)
        if witnessed: groups[key].append(item)
    if set(groups) != set(keys):
        raise ValueError("A cell lacks a verified witness; do not silently remove it")
    archive = []
    for key in keys:
        best = min(groups[key], key=lambda r: (-r["requested_kw"], r["source_action_id"]))
        archive.append(deepcopy(best))
    return catalog, archive


def success_labels(teacher_rows, archive, teacher_run):
    """Relabel recorded *predecision* states with a same-cell certified candidate.

    Independent virtual point reset makes candidate outcomes history-independent.
    Value target is attainable remaining improvement, not a fabricated trajectory.
    """
    by = {(r["scenario_id"], r["condition_id"]): r for r in archive}
    labels = []
    for offset in range(0, len(teacher_rows), 8):
        group = teacher_rows[offset:offset + 8]
        if len(group) != 8 or [r["step"] for r in group] != list(range(8)):
            raise ValueError("Incomplete expert episode")
        key = (group[0]["scenario_id"], group[0]["condition_id"])
        if key not in by or any((r["scenario_id"], r["condition_id"]) != key for r in group):
            raise ValueError("Mixed/missing cell")
        best = by[key]
        for step in range(1, 8):
            row = group[step]
            before = group[step - 1]["incumbent_fraction"]
            labels.append({"scenario_id": key[0], "condition_id": key[1],
                "observation": deepcopy(row["observation"]), "action": deepcopy(best["action"]),
                "teacher_return": max(0., unit_fraction(best["restored_fraction"]) - unit_fraction(before)),
                "teacher_state_run": teacher_run, "teacher_state_action_id": row["action_id"],
                "success_source_run": best["source_run"], "success_action_id": best["source_action_id"],
                "success_point_id": best["witness_point_id"], "success_alpha": best["selected_alpha"]})
    if len(labels) != 7 * len(archive):
        raise ValueError("All cells require seven equal-weight labels")
    return labels
