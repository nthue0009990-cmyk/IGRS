"""Reproducible candidate generators for the bounded NC34 build comparison.

These are independent implementations for this named development model. The
graph rule is not a reproduction of a named restoration/search algorithm.
No function imports an electrical simulator or updates a learned policy.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import math

import networkx as nx
import numpy as np


POLICY_DOMAIN = "nature34-policy-proposals-v1"
RANDOM_DOMAIN = "nature34-random-proposals-v1"


def _seed(domain, seed, scenario_id):
    if not isinstance(seed, int) or isinstance(seed, bool) or not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError("A fixed integer evaluation seed and scenario identity are required")
    payload = f"{domain}|{seed}|{scenario_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def _count(count):
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("Candidate count must be a positive integer")


def _arrays(table):
    keys = ("observation", "switch_forced_open", "context_table", "switch_support_mask", "load_forced_off_table")
    result = {key: np.asarray(table[key]) for key in keys}
    switches = result["switch_forced_open"]
    support = result["switch_support_mask"]
    off = result["load_forced_off_table"]
    if switches.ndim != 1 or switches.dtype != np.bool_ or support.dtype != np.bool_ or support.shape != (2 ** len(switches),):
        raise ValueError("Fault and support masks must have consistent Boolean shapes")
    if off.ndim != 2 or off.shape[0] != len(support) or off.dtype != np.bool_:
        raise ValueError("Load forced-off table must be Boolean[topologies,loads]")
    if not support.any() or not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("Finite inputs with at least one legal switch assignment are required")
    bits = (np.arange(len(support))[:, None] >> np.arange(len(switches))) & 1
    if np.any(support & np.any(bits.astype(bool) & switches, axis=1)):
        raise ValueError("Legal support conflicts with fault-forced open switches")
    return result


def propose_policy_actions(policy, table, *, seed, scenario_id, count=8):
    """Draw a fixed iid prefix, paired across initial/final and both variants.

    Each action is sampled as a batch of one, so changing ``count`` preserves
    the prefix. The caller's global Torch RNG and policy training flag survive.
    Variant/checkpoint are deliberately absent from the domain-separated seed.
    """
    import torch
    _count(count)
    arrays = _arrays(table)
    parameter = next(policy.parameters())
    if parameter.device.type != "cpu":
        raise ValueError("The frozen development comparison uses CPU policies")
    inputs = tuple(torch.as_tensor(arrays[key], dtype=parameter.dtype if arrays[key].dtype != np.bool_ else torch.bool,
                                   device=parameter.device).unsqueeze(0) for key in
                   ("observation", "switch_forced_open", "context_table", "switch_support_mask", "load_forced_off_table"))
    derived = _seed(POLICY_DOMAIN, seed, scenario_id)
    actions, log_prob, entropy, value = [], [], [], []
    training = policy.training
    try:
        policy.eval()
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(derived)
            for _ in range(count):
                output = policy.sample(*inputs, deterministic=False)
                evaluated = policy.evaluate_actions(*inputs, output.actions)
                if not torch.equal(output.log_prob, evaluated.log_prob) or not torch.isfinite(output.log_prob).all():
                    raise AssertionError("Frozen policy candidate density does not match its recorded action")
                actions.append(output.actions[0].to(torch.int64).tolist())
                log_prob.append(output.log_prob[0].item())
                entropy.append(output.entropy[0].item())
                value.append(output.value[0].item())
    finally:
        policy.train(training)
    return {"actions": actions, "log_prob": log_prob, "entropy": entropy, "value": value,
            "seed": seed, "scenario_id": scenario_id, "rng_seed": derived, "rng_domain": POLICY_DOMAIN,
            "count": count, "generation_rule": "One-at-a-time iid supported-policy sampling; prefix invariant",
            "paired_across_initial_final_and_variant": True, "physical_evaluations": 0}


def propose_random_actions(table, *, seed, scenario_id, count=8):
    """Uniform legal switch assignment, then Bernoulli(.5) reachable loads.

    This is not uniform over all joint actions when topologies have different
    numbers of reachable loads. Every unreachable load remains exactly zero.
    """
    _count(count)
    arrays = _arrays(table)
    derived = _seed(RANDOM_DOMAIN, seed, scenario_id)
    rng = np.random.default_rng(derived)
    valid = np.flatnonzero(arrays["switch_support_mask"])
    n_switch = len(arrays["switch_forced_open"])
    n_load = arrays["load_forced_off_table"].shape[1]
    actions, log_prob = [], []
    for _ in range(count):
        index = int(rng.choice(valid))
        switches = ((index >> np.arange(n_switch)) & 1).tolist()
        loads = rng.integers(0, 2, size=n_load)
        forced_off = arrays["load_forced_off_table"][index]
        loads[forced_off] = 0
        actions.append(switches + loads.tolist())
        log_prob.append(-math.log(len(valid)) - int((~forced_off).sum()) * math.log(2))
    return {"actions": actions, "log_prob": log_prob, "seed": seed, "scenario_id": scenario_id,
            "rng_seed": derived, "rng_domain": RANDOM_DOMAIN, "count": count,
            "generation_rule": "Uniform legal switch assignment; iid Bernoulli0.5 loads conditional on reachability",
            "uniform_joint_distribution": False, "physical_evaluations": 0}


class GraphRepairSearch:
    """Two-start graph ordering with feedback-driven greedy load pickup.

    Forest roots rank by all68 reachable nominal kW descending, then
    sum(nominal_kW * source_path_hops) ascending, then switch index. Two active
    roots alternate. A root first tests all reachable dispatch loads on. If
    that fails, it tests all dispatch loads off, retaining non-dispatch loads.
    It then picks dispatch loads up in descending nominal kW/author-index order.

    Once a witness exists, a failed pickup is reverted. Before the first
    witness, pickups accumulate, so an infeasible all-off endpoint does not
    incorrectly prove that intermediate load combinations are infeasible.
    An all-on witness reaches that root's nominal upper bound and ends it.
    Completed roots are replaced by the next graph-ranked root. All physical
    feedback comes solely from this search stream's already billed requests.
    """

    def __init__(self, static, load_reference, table, scenario, budget=8):
        _count(budget)
        self.budget = budget
        self._table = _arrays(table)
        self._dispatch = list(static["dispatch_loads"])
        self._switch_edges = [tuple(edge) for edge in static["switch_edges"]]
        self._n_switch = len(self._switch_edges)
        self._n_load = len(self._dispatch)
        if self._table["load_forced_off_table"].shape != (2 ** self._n_switch, self._n_load):
            raise ValueError("Static switch/load dimensions disagree with the shared tables")
        if not isinstance(load_reference, Mapping) or not load_reference:
            raise ValueError("Every original nominal load needs its saved reference")
        self._loads = {str(name).lower(): {"bus": str(row["bus"]).lower(), "nominal_kw": float(row["nominal_kw"])}
                       for name, row in load_reference.items()}
        if len(self._loads) != len(load_reference) or any(not math.isfinite(row["nominal_kw"]) or row["nominal_kw"] < 0 for row in self._loads.values()):
            raise ValueError("Nominal load identities and nonnegative kW must be known and finite")
        if any(name.lower() not in self._loads for name in self._dispatch):
            raise ValueError("Dispatch loads are missing from the original nominal demand inventory")
        self._nominal_total = sum(row["nominal_kw"] for row in self._loads.values())
        if self._nominal_total <= 0:
            raise ValueError("The nominal demand denominator must be positive")
        graph = nx.Graph()
        graph.add_nodes_from(static["nodes"])
        graph.add_edges_from(static["bus_edges"])
        source = static["source_bus"]
        edges = {str(name).lower(): edge for name, edge in static["original_lines"].items()}
        faults = [str(name).lower() for name in scenario["fault_lines"]]
        if len(faults) != len(set(faults)) or any(name not in edges for name in faults):
            raise ValueError("Scenario faults must be unique original lines")
        graph.remove_edges_from(edges[name] for name in faults)
        self._roots = []
        for index in np.flatnonzero(self._table["switch_support_mask"]):
            index = int(index)
            candidate = graph.copy()
            bits = [(index >> bit) & 1 for bit in range(self._n_switch)]
            candidate.remove_edges_from(edge for bit, edge in zip(bits, self._switch_edges) if not bit)
            if not nx.is_forest(candidate) or source not in candidate:
                raise ValueError("Shared support contains a non-forest or lacks the main source")
            distances = nx.single_source_shortest_path_length(candidate, source)
            for row in self._loads.values():
                if row["bus"] not in candidate:
                    raise ValueError("Original nominal load refers to an unknown bus")
            expected_off = np.asarray([self._loads[name.lower()]["bus"] not in distances for name in self._dispatch], dtype=bool)
            if not np.array_equal(expected_off, self._table["load_forced_off_table"][index]):
                raise ValueError("Shared load mask disagrees with main-source graph reachability")
            nominal = sum(row["nominal_kw"] for row in self._loads.values() if row["bus"] in distances)
            hops = sum(row["nominal_kw"] * distances[row["bus"]] for row in self._loads.values() if row["bus"] in distances)
            reachable = [i for i in range(self._n_load) if not expected_off[i]]
            pickup = sorted(reachable, key=lambda i: (-self._loads[self._dispatch[i].lower()]["nominal_kw"], i))
            self._roots.append({"action_index": index, "switch_bits": bits, "full_loads": (~expected_off).astype(int).tolist(),
                                "reachable_nominal_kw": nominal, "weighted_hops_kw": hops, "pickup_order": pickup,
                                "phase": "all_on", "pickup_position": 0, "base": [0] * self._n_load,
                                "has_witness": False, "finished": False})
        self._roots.sort(key=lambda row: (-row["reachable_nominal_kw"], row["weighted_hops_kw"], row["action_index"]))
        self._next_root = 0
        self._slots = [None, None]
        self._turn = 0
        self._pending = None
        self._seen = {}
        self._history = []
        self._observations = []
        self._duplicate_requests = 0

    def _proposal(self, root):
        if root["phase"] == "all_on":
            loads = root["full_loads"].copy()
        elif root["phase"] == "all_off":
            loads = [0] * self._n_load
        else:
            if root["pickup_position"] >= len(root["pickup_order"]):
                root["finished"] = True
                return None
            loads = root["base"].copy()
            loads[root["pickup_order"][root["pickup_position"]]] = 1
        return root["switch_bits"] + loads

    def _advance(self, root, action, decision):
        witness = decision["witness_found"]
        if root["phase"] == "all_on":
            if witness:
                root["finished"] = True
            else:
                root["phase"] = "all_off"
        elif root["phase"] == "all_off":
            root["phase"] = "pickup"
            root["base"] = action[self._n_switch:].copy()
            root["has_witness"] = witness
        else:
            if witness or not root["has_witness"]:
                root["base"] = action[self._n_switch:].copy()
            root["has_witness"] |= witness
            root["pickup_position"] += 1
            if root["pickup_position"] >= len(root["pickup_order"]):
                root["finished"] = True

    def next_action(self):
        if self._pending is not None:
            raise RuntimeError("Observe the pending candidate before requesting the next action")
        if len(self._history) >= self.budget:
            raise StopIteration("Frozen candidate budget exhausted")
        for _ in range(2 * len(self._roots) * (self._n_load + 3) + 2):
            slot = self._turn % 2
            self._turn += 1
            root = self._slots[slot]
            if root is None or root["finished"]:
                if self._next_root >= len(self._roots):
                    self._slots[slot] = None
                    if all(value is None or value["finished"] for value in self._slots):
                        break
                    continue
                root = self._roots[self._next_root]
                self._next_root += 1
                self._slots[slot] = root
            action = self._proposal(root)
            if action is None:
                continue
            key = tuple(action)
            if key in self._seen:
                self._duplicate_requests += 1
            self._pending = (root, action.copy())
            self._history.append({"candidate_index": len(self._history), "action": action.copy(), "action_index": root["action_index"],
                                  "stage": root["phase"], "reachable_nominal_kw": root["reachable_nominal_kw"],
                                  "weighted_hops_kw": root["weighted_hops_kw"], "duplicate_request_still_billed": key in self._seen})
            return action.copy()
        # Tiny/degenerate supports can exhaust this rule's distinct proposals.
        # Repeated requests remain visible and are still physically billed.
        if not self._seen:
            raise RuntimeError("No candidate could be produced from legal support")
        key = min(self._seen, key=lambda action: (-self._seen[action]["reward"], not self._seen[action]["witness_found"], action))
        action = list(key)
        self._duplicate_requests += 1
        self._pending = (None, action.copy())
        self._history.append({"candidate_index": len(self._history), "action": action.copy(), "stage": "exhausted_rule_repeat",
                              "duplicate_request_still_billed": True})
        return action

    def observe(self, action, decision):
        if self._pending is None or list(action) != self._pending[1]:
            raise ValueError("Feedback must correspond exactly to the pending action")
        reward = decision["reward"]
        witness = decision["witness_found"]
        selected = decision["selected_alpha"]
        if not isinstance(witness, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward) or not 0 <= reward <= 1:
            raise ValueError("Witness and nominal reward feedback must be explicit and finite")
        if (witness and selected not in (0, .25, .5, .75, 1)) or (not witness and (selected is not None or reward != 0)):
            raise ValueError("Selected alpha/reward contradict witness status")
        root, pending_action = self._pending
        feedback = {"reward": float(reward), "witness_found": witness, "selected_alpha": selected}
        self._seen[tuple(pending_action)] = feedback
        self._observations.append({"candidate_index": len(self._history) - 1, **feedback})
        if root is not None:
            self._advance(root, pending_action, feedback)
        self._pending = None

    @property
    def metadata(self):
        return deepcopy({"algorithm": "Two-start graph-ranked greedy load pickup; independent named development rule",
                "root_order": "Descending all-original reachable nominal kW, ascending sum(kW*source hops), ascending switch index",
                "active_roots": 2, "root_rotation": "Alternate two active roots; replace finished root with next ranked root",
                "load_rule": "all-on; if no witness all-off; then descending nominal-kW pickups; revert failed pickup only after a witness exists",
                "nominal_total_kw": self._nominal_total, "original_load_count": len(self._loads),
                "budget": self.budget, "ranked_roots": [{key: root[key] for key in ("action_index", "reachable_nominal_kw", "weighted_hops_kw", "pickup_order")}
                                                       for root in self._roots],
                "proposal_history": [dict(row) for row in self._history], "observed_feedback": [dict(row) for row in self._observations],
                "duplicate_requests_still_physically_billed": self._duplicate_requests,
                "duplicates_skipped_without_physical_request": 0,
                "no_physical_cache": True, "no_cross_stream_feedback": True, "not_a_named_classical_algorithm_reproduction": True,
                "physical_evaluations_executed_by_generator": 0})
