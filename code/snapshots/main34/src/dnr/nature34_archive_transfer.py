"""Deterministic archive transfer and repair for fixed-model NC34 development.

The archive contributes action templates, never target-fault electrical proof.
Only the caller evaluates proposals. Every returned action is a paid request,
including repeats; the initial offline template canonicalization is recorded.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math

import numpy as np

from .nature34_baselines import GraphRepairSearch


ALGORITHM = "same-condition-min-hamming-archive-transfer-alternating-graph-v1"


def _binary(values, size):
    values = list(values)
    if len(values) != size or any(type(value) not in (int, bool) or value not in (0, 1) for value in values):
        raise ValueError("Archive actions require the exact binary switch/load dimensions")
    return [int(value) for value in values]


def _code(action):
    return sum(bit << index for index, bit in enumerate(action))


def _decision(decision):
    reward, witness, alpha = (decision[key] for key in ("reward", "witness_found", "selected_alpha"))
    if not isinstance(witness, bool) or isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise ValueError("Explicit witness and finite numerical reward required")
    if not math.isfinite(reward) or not 0 <= reward <= 1:
        raise ValueError("Nominal restored reward must be in [0,1]")
    if (witness and alpha not in (0, .25, .5, .75, 1)) or (not witness and (alpha is not None or reward != 0)):
        raise ValueError("Selected alpha/reward contradict the target witness status")
    return {"reward": float(reward), "witness_found": witness, "selected_alpha": alpha}


class ArchiveTransferSearch:
    """Use four transfer/repair slots interleaved with the original graph rule.

    Slot0 is exactly the common GraphRepairSearch starter. Odd slots first use
    distinct offline archive projections in frozen order. Even slots continue
    that same graph-rule instance. Once templates are exhausted, odd slots try
    one additional reachable dispatch load from this stream's best witnessed
    action; failed pickups retain that witness. If no pickup remains, use the
    original graph rule. It receives feedback only for its own billed actions.

    Projection: minimum switch Hamming distance to a source action, then maximum
    target requested nominal kW, minimum all-reachable load-weighted source hops,
    and minimum switch index. Unreachable target dispatch loads are cleared.
    Template ordering: descending requested kW, source/target fault symmetric
    difference, switch distance, weighted hops, little-endian joint action code,
    source scenario ID, source action ID. No source electrical reward ranks it.
    """

    def __init__(self, static, load_reference, table, scenario, archive, source_scenarios, condition_id, budget=8):
        self._graph = GraphRepairSearch(static, load_reference, table, scenario, budget=budget)
        self.budget = budget
        self._scenario_id = scenario["scenario_id"]
        self._condition_id = condition_id
        self._n_switch = len(static["switch_edges"])
        self._dispatch = list(static["dispatch_loads"])
        self._n_load = len(self._dispatch)
        self._off = np.asarray(table["load_forced_off_table"], dtype=bool).copy()
        self._roots = self._graph.metadata["ranked_roots"]
        self._roots_by_index = {row["action_index"]: row for row in self._roots}
        lowered_loads = {str(name).lower(): row for name, row in load_reference.items()}
        self._dispatch_kw = [float(lowered_loads[name.lower()]["nominal_kw"]) for name in self._dispatch]
        self._nominal_total = self._graph.metadata["nominal_total_kw"]
        if not isinstance(condition_id, str) or not condition_id:
            raise ValueError("A frozen public condition ID is required")
        if scenario.get("partition") != "build":
            raise ValueError("Archive transfer is limited to the frozen build development partition")
        source_list = list(source_scenarios.values()) if isinstance(source_scenarios, Mapping) else list(source_scenarios)
        source_by = {row["scenario_id"]: row for row in source_list}
        if not source_by or len(source_by) != len(source_list):
            raise ValueError("Source scenarios must be unique and nonempty")
        edges = {str(name).lower() for name in static["original_lines"]}
        target_faults = frozenset(str(name).lower() for name in scenario["fault_lines"])
        source_faults = {}
        for sid, row in source_by.items():
            faults = [str(name).lower() for name in row["fault_lines"]]
            if row.get("partition") != "build" or not faults or len(faults) != len(set(faults)) or not set(faults) <= edges:
                raise ValueError("Source faults must be unique known build original lines")
            if sid == self._scenario_id or frozenset(faults) == target_faults:
                raise ValueError("Target must be an unseen fault combination for this archive")
            source_faults[sid] = frozenset(faults)
        archive = list(archive)
        identities = [(row["scenario_id"], row["condition_id"]) for row in archive]
        if len(set(identities)) != len(identities) or not archive:
            raise ValueError("Archive cells must be unique and nonempty")
        for row in archive:
            if row["scenario_id"] not in source_by or row.get("witness_found") is not True or row.get("witness_point_id") is None:
                raise ValueError("Only traced, source-witnessed archive templates are allowed")
            _binary(row["action"], self._n_switch + self._n_load)
            if not isinstance(row.get("source_action_id"), int) or isinstance(row["source_action_id"], bool) or row["source_action_id"] < 0:
                raise ValueError("Archive source action identity must be a nonnegative integer")
        selected = [row for row in archive if row["condition_id"] == condition_id]
        if {row["scenario_id"] for row in selected} != set(source_by):
            raise ValueError("Same-condition archive must include every source scenario exactly once")

        starter_index = self._roots[0]["action_index"]
        self._starter = self._switch_bits(starter_index) + (~self._off[starter_index]).astype(int).tolist()
        self._catalog = []
        for row in sorted(selected, key=lambda item: (item["scenario_id"], item["source_action_id"])):
            source_action = _binary(row["action"], self._n_switch + self._n_load)
            candidates = []
            for root in self._roots:
                index = root["action_index"]
                action = self._switch_bits(index) + [int(bit and not off) for bit, off in
                           zip(source_action[self._n_switch:], self._off[index])]
                distance = sum(left != right for left, right in zip(action[:self._n_switch], source_action[:self._n_switch]))
                nominal = self._requested_kw(action)
                candidates.append(((distance, -nominal, root["weighted_hops_kw"], index), action))
            key, projected = min(candidates)
            item = {"source_scenario_id": row["scenario_id"], "source_action_id": row["source_action_id"],
                    "source_run": row.get("source_run"), "source_witness_point_id": row["witness_point_id"],
                    "source_action": source_action, "action": projected, "switch_index": key[3],
                    "switch_hamming": key[0], "requested_nominal_kw": -key[1], "weighted_hops_kw": key[2],
                    "fault_symmetric_difference": len(source_faults[row["scenario_id"]] ^ target_faults),
                    "joint_action_code": _code(projected), "target_electrical_witness_inherited": False}
            self._catalog.append(item)
        ranked = sorted(self._catalog, key=self._rank)
        grouped = {}
        self._queue = []
        self._excluded_starter = []
        for item in ranked:
            if item["action"] == self._starter:
                self._excluded_starter.append(deepcopy(item))
                continue
            key = tuple(item["action"])
            if key not in grouped:
                proposal = deepcopy(item)
                proposal["queue_index"] = len(self._queue)
                proposal["source_templates"] = []
                self._queue.append(proposal)
                grouped[key] = proposal
            grouped[key]["source_templates"].append(deepcopy(item))
        self._position = 0
        self._history = []
        self._observations = []
        self._seen = set()
        self._pending = None
        self._incumbent = None
        self._repair_attempts = set()

    def _switch_bits(self, index):
        return [(index >> bit) & 1 for bit in range(self._n_switch)]

    def _requested_kw(self, action):
        index = _code(action[:self._n_switch])
        return self._roots_by_index[index]["reachable_nominal_kw"] - sum(
            self._dispatch_kw[i] for i, bit in enumerate(action[self._n_switch:]) if not bit and not self._off[index, i])

    @staticmethod
    def _rank(item):
        return (-item["requested_nominal_kw"], item["fault_symmetric_difference"], item["switch_hamming"],
                item["weighted_hops_kw"], item["joint_action_code"], item["source_scenario_id"], item["source_action_id"])

    def _repair(self):
        if self._incumbent is None:
            return None
        base = self._incumbent["action"]
        index = _code(base[:self._n_switch])
        eligible = [i for i in range(self._n_load) if not base[self._n_switch + i] and not self._off[index, i]
                    and (tuple(base), i) not in self._repair_attempts]
        if not eligible:
            return None
        pickup = min(eligible, key=lambda i: (-self._dispatch_kw[i], i))
        action = base.copy()
        action[self._n_switch + pickup] = 1
        return action, {"repair_base_action": base.copy(), "repair_load_index": pickup,
                        "repair_load_name": self._dispatch[pickup], "repair_load_nominal_kw": self._dispatch_kw[pickup]}

    def next_action(self):
        if self._pending is not None:
            raise RuntimeError("Observe the pending candidate before requesting another")
        step = len(self._history)
        if step >= self.budget:
            raise StopIteration("Frozen candidate budget exhausted")
        graph_action = False
        details = {}
        if step == 0:
            kind, action, graph_action = "common_graph_starter", self._graph.next_action(), True
            if action != self._starter:
                raise AssertionError("Archive common starter differs from the original graph rule")
        elif step % 2 == 0:
            kind, action, graph_action = "graph_continue", self._graph.next_action(), True
        elif self._position < len(self._queue):
            item = self._queue[self._position]
            self._position += 1
            kind, action = "archive_projection", item["action"].copy()
            details = {"queue_index": item["queue_index"], "source_scenario_id": item["source_scenario_id"],
                       "source_action_id": item["source_action_id"]}
        else:
            repair = self._repair()
            if repair is not None:
                kind, (action, details) = "witness_load_pickup", repair
                self._repair_attempts.add((tuple(details["repair_base_action"]), details["repair_load_index"]))
            else:
                kind, action, graph_action = "graph_fallback", self._graph.next_action(), True
        row = {"candidate_index": step, "kind": kind, "action": action.copy(),
               "requested_nominal_kw": self._requested_kw(action),
               "duplicate_request_still_billed": tuple(action) in self._seen, **details}
        self._history.append(row)
        self._pending = {"action": action.copy(), "graph_action": graph_action}
        return action.copy()

    def observe(self, action, decision):
        if self._pending is None or list(action) != self._pending["action"]:
            raise ValueError("Feedback must exactly match the pending paid proposal")
        feedback = _decision(decision)
        if self._pending["graph_action"]:
            self._graph.observe(action, feedback)
        current = {"action": list(action), **feedback}
        if feedback["witness_found"] and (self._incumbent is None or
                (-feedback["reward"], tuple(action)) < (-self._incumbent["reward"], tuple(self._incumbent["action"]))):
            self._incumbent = deepcopy(current)
        self._seen.add(tuple(action))
        self._observations.append({"candidate_index": len(self._history) - 1, **feedback,
                                   "incumbent": deepcopy(self._incumbent)})
        self._pending = None

    @property
    def metadata(self):
        return deepcopy({"algorithm": ALGORITHM, "scenario_id": self._scenario_id, "condition_id": self._condition_id,
                "budget": self.budget, "common_starter": self._starter,
                "projection_rule": "min switch Hamming, max target requested nominal kW, min all-reachable weighted hops, min switch index",
                "ranking_rule": "-requested nominal kW, fault symmetric difference, switch Hamming, weighted hops, little-endian joint action code, source scenario ID, source action ID",
                "slot_rule": "slot0 common graph; odd archive then witnessed dispatch pickup then graph fallback; positive even graph continuation",
                "repair_rule": "best current paid witness by -reward/action tuple; max nominal-kW untried reachable off dispatch load then author index; failed pickup retains base",
                "projected_catalog": self._catalog, "transfer_queue": self._queue,
                "offline_common_starter_templates_excluded": self._excluded_starter,
                "offline_duplicate_templates_merged": len(self._catalog) - len(self._excluded_starter) - len(self._queue),
                "proposal_history": self._history, "observed_feedback": self._observations,
                "graph_metadata": self._graph.metadata, "nominal_total_kw": self._nominal_total,
                "duplicate_requests_still_physically_billed": sum(row["duplicate_request_still_billed"] for row in self._history),
                "physical_requests_skipped_by_cache": 0, "no_physical_cache": True, "no_cross_stream_feedback": True,
                "target_electrical_witness_inherited": False, "physical_evaluations_executed_by_generator": 0})
