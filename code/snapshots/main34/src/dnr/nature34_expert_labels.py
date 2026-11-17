"""Pure D037 finite-graph8 teacher selection; no simulator or learner imports."""
from __future__ import annotations

from collections import defaultdict
import math

ALPHAS = (0, .25, .5, .75, 1)


def _require(test, message):
    if not test:
        raise ValueError(message)


def _witness(record):
    points = record["points"]
    _require(len(points) == 5 and [p["alpha"] for p in points] == list(ALPHAS), "Five ordered alpha points required")
    fractions = [p["nominal_restored_fraction"] for p in points]
    _require(all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0 <= v <= 1
                 for v in fractions) and len(set(fractions)) == 1, "Invalid or alpha-dependent nominal restoration")
    accepted = [p for p in points if p["converged"] is True and p["known_constraints_satisfied"] is True
                and p["ac_status"] == p["isolation_status"] == "PASS"]
    chosen = accepted[-1] if accepted else None
    expected = {"witness_found": chosen is not None, "reward": fractions[0] if chosen else 0.,
                "selected_alpha": chosen["alpha"] if chosen else None,
                "witness_point_id": chosen["point_id"] if chosen else None,
                "accepted_alphas": [p["alpha"] for p in accepted]}
    _require(all(record[key] == value for key, value in expected.items()), "Recorded candidate selection contradicts certificates")
    return chosen


def build_expert_labels(actions, scenarios):
    """Archive per-topology witnesses and select per-scenario best BC targets.

    ``actions`` contains all eight paid graph candidates for every build fault,
    including failed and zero-restoration candidates. Each record explicitly
    identifies ``source_run`` and ``source_action_id``. ``scenarios`` contains
    the complete frozen build dictionaries with scenario_id/fault_lines.

    The output keeps all scenarios. Only tied maximum-reward witnessed
    topologies train BC. Ties are exact nominal fractions; within one topology
    the earliest maximum-reward candidate wins. Global weights average labeled
    scenarios, then their tied topologies. A real zero witness has mask1; absence
    of a witness has mask0 and no invented label.
    """
    _require(len(scenarios) == len({s["scenario_id"] for s in scenarios}), "Duplicate scenario identity")
    by_scenario = {s["scenario_id"]: s for s in scenarios}
    groups, identities = defaultdict(list), set()
    for record in actions:
        sid = record["scenario_id"]
        _require(sid in by_scenario and record["partition"] == "build" and record["method"] == "graph_repair",
                 "Teacher evidence must be from the frozen build graph8 population")
        _require(record["fault_lines"] == by_scenario[sid]["fault_lines"], "Fault identity differs from frozen scenario")
        _require(isinstance(record.get("source_run"), str) and bool(record["source_run"]) and
                 type(record.get("source_action_id")) is int and record["source_action_id"] >= 0,
                 "Explicit source-run/action provenance required")
        identity = (record["source_run"], record["source_action_id"])
        _require(identity not in identities, "Duplicate physical action provenance")
        identities.add(identity)
        bits = record["action"]
        _require(len(bits) == 19 and all(type(bit) is int and bit in (0, 1) for bit in bits), "Nineteen binary action bits required")
        _require(type(record["candidate_index"]) is int, "Candidate index must be an integer")
        _witness(record)
        groups[sid].append(record)
    primary, archive, masks, coverage = {}, {}, {}, {}
    for sid in by_scenario:
        records = sorted(groups[sid], key=lambda r: r["candidate_index"])
        _require([r["candidate_index"] for r in records] == list(range(8)), "Every fault requires all eight paid graph candidates")
        topology = defaultdict(list)
        for record in records:
            switch_index = sum(bit << i for i, bit in enumerate(record["action"][:9]))
            topology[switch_index].append(record)
        all_labels = []
        for switch_index in sorted(topology):
            witnessed = [r for r in topology[switch_index] if r["witness_found"]]
            if not witnessed:
                continue
            chosen = min(witnessed, key=lambda r: (-r["reward"], r["candidate_index"]))
            point = _witness(chosen)
            point_source = point.get("source_run", chosen["source_run"])
            _require(point_source == chosen["source_run"], "Graph candidate alpha points cannot cross physical source runs")
            all_labels.append({"scenario_id": sid, "partition": "build", "fault_lines": chosen["fault_lines"],
                "action": list(chosen["action"]), "teacher_reward": chosen["reward"],
                "source_run": chosen["source_run"], "source_action_id": chosen["source_action_id"],
                "candidate_index": chosen["candidate_index"], "switch_index": switch_index,
                "selected_alpha": chosen["selected_alpha"], "witness_point_id": chosen["witness_point_id"],
                "source_point_id": point.get("source_point_id", point["point_id"]),
                "witness_kind": "positive" if chosen["reward"] > 0 else "zero"})
        maximum = max((label["teacher_reward"] for label in all_labels), default=None)
        bc_labels = [dict(label) for label in all_labels if label["teacher_reward"] == maximum]
        for label in all_labels:
            label["within_scenario_archive_weight"] = 1. / len(all_labels)
        for label in bc_labels:
            label["within_scenario_weight"] = 1. / len(bc_labels)
        primary[sid], archive[sid], masks[sid] = bc_labels, all_labels, int(bool(bc_labels))
        coverage[sid] = {"candidate_count": 8, "alpha_point_count": 40, "distinct_evaluated_topologies": len(topology),
            "per_topology_label_count": len(all_labels), "bc_label_count": len(bc_labels), "bc_mask": masks[sid],
            "scenario_teacher_reward": maximum, "zero_witness_candidates": sum(r["witness_found"] and r["reward"] == 0 for r in records),
            "positive_witness_candidates": sum(r["witness_found"] and r["reward"] > 0 for r in records),
            "no_witness_candidates": sum(not r["witness_found"] for r in records),
            "no_label_reason": None if bc_labels else "NO_VERIFIED_GRAPH8_WITNESS"}
    labeled = sum(masks.values())
    for labels in primary.values():
        for label in labels:
            label["global_normalized_weight"] = label["within_scenario_weight"] / labeled
    return {"schema_version": "nature34_graph8_expert_labels_v1", "labels_by_scenario": primary,
        "per_topology_labels_by_scenario": archive, "bc_mask_by_scenario": masks, "scenario_coverage": coverage,
        "scenario_count": len(scenarios), "labeled_scenario_count": labeled,
        "bc_label_count": sum(len(labels) for labels in primary.values()),
        "per_topology_label_count": sum(len(labels) for labels in archive.values()),
        "candidate_count": len(actions), "alpha_point_count": 5 * len(actions),
        "selection_rule": "Each topology: maximum witnessed reward, earliest candidate tie; primary BC: scenario maximum witnessed reward tied topologies",
        "weighting_rule": "Uniform labeled scenario, then uniform tied primary topology; no-label scenarios retained mask0",
        "optimality_claim": "FINITE_GRAPH8_BEST_WITNESSED_ONLY", "new_powerflow_calls": 0}
