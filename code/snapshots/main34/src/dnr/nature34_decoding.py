"""Exact, distinct joint-policy decoding without an electrical simulator.

The existing deterministic policy sampler first selects a switch topology and
then its load modes.  That sequential choice need not maximize the joint
probability.  Here every legal topology contributes its exact best load
assignments before a global merge.  This changes decoding, not policy weights
or the policy's topology and reachable-load support.
"""
from __future__ import annotations

import math

import numpy as np


INPUT_KEYS = ("observation", "switch_forced_open", "context_table",
              "switch_support_mask", "load_forced_off_table")


def _positive_count(count):
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("count must be a positive integer")


def joint_topk_from_logits(
    switch_logits, load_logits, switch_forced_open, switch_support_mask,
    load_forced_off_table, *, count=8,
):
    """Rank exact legal joint actions from fixed logits, using float64 scores.

    Switch row indices and the first action bits use little-endian order.
    The complete action code is ``sum(action[i] * 2**i)``: switch bits occupy
    the low positions, followed by load bits.  Exactly equal computed scores
    are ordered by this integer code.  No probability tolerance creates ties.

    At most ten load bits are accepted so each topology needs at most 1024
    explicit assignments.  Keeping the first ``count`` assignments from each
    topology is sufficient for the global first ``count``: every omitted
    assignment already has ``count`` predecessors in its own topology under
    the same total order.  Disjoint topology indices and unique local load
    codes prove that the returned full action codes are distinct.
    """
    _positive_count(count)
    raw_switch = np.asarray(switch_logits)
    raw_load = np.asarray(load_logits)
    switch = np.asarray(switch_logits, dtype=np.float64)
    loads = np.asarray(load_logits, dtype=np.float64)
    forced = np.asarray(switch_forced_open)
    support = np.asarray(switch_support_mask)
    off = np.asarray(load_forced_off_table)
    if switch.ndim != 1 or not 1 <= len(switch) <= 10:
        raise ValueError("switch_logits must have between one and ten switch bits")
    n_switch = len(switch)
    combinations = 2 ** n_switch
    if loads.ndim != 2 or loads.shape[0] != combinations or not 1 <= loads.shape[1] <= 10:
        raise ValueError("load_logits must have shape [2**n_switch, one to ten loads]")
    n_load = loads.shape[1]
    if forced.dtype != np.bool_ or forced.shape != (n_switch,):
        raise ValueError("switch_forced_open must be Boolean [n_switch]")
    if support.dtype != np.bool_ or support.shape != (combinations,) or not support.any():
        raise ValueError("switch_support_mask must be nonempty Boolean [2**n_switch]")
    if off.dtype != np.bool_ or off.shape != loads.shape:
        raise ValueError("load_forced_off_table must be Boolean [2**n_switch,n_load]")
    if not np.isfinite(switch).all() or not np.isfinite(loads).all():
        raise ValueError("All frozen switch and load logits must be finite")
    switch_bits = ((np.arange(combinations)[:, None] >> np.arange(n_switch)) & 1)
    forbidden = (switch_bits.astype(bool) & forced).any(axis=1)
    if np.any(support & forbidden):
        raise ValueError("Switch support conflicts with forced-open faults")

    # Stable Bernoulli log probabilities, evaluated after logits conversion.
    switch_zero = -np.logaddexp(0., switch)
    switch_one = -np.logaddexp(0., -switch)
    terms = np.where(switch_bits, switch_one, switch_zero)
    terms[:, forced] = 0.
    raw_scores = terms.sum(axis=1)
    legal = np.flatnonzero(support)
    maximum = float(raw_scores[legal].max())
    normalizer = maximum + math.log(math.fsum(float(x) for x in np.exp(raw_scores[legal] - maximum)))
    switch_logs = raw_scores - normalizer

    candidates = []
    per_switch = []
    legal_joint_count = 0
    for switch_index in legal.tolist():
        reachable = np.flatnonzero(~off[switch_index])
        local_count = 2 ** len(reachable)
        legal_joint_count += local_count
        assignments = ((np.arange(local_count)[:, None] >> np.arange(len(reachable))) & 1)
        load_codes = (assignments * (2 ** reachable)).sum(axis=1, dtype=np.int64)
        zero_logs = -np.logaddexp(0., loads[switch_index, reachable])
        one_logs = -np.logaddexp(0., -loads[switch_index, reachable])
        scores = switch_logs[switch_index] + np.where(assignments, one_logs, zero_logs).sum(axis=1)
        action_codes = switch_index + (load_codes << n_switch)
        order = np.lexsort((action_codes, -scores))
        retained = order[:count]
        for index in retained.tolist():
            candidates.append((float(scores[index]), int(action_codes[index]), switch_index, int(load_codes[index])))
        first_omitted = int(order[count]) if local_count > count else None
        per_switch.append({
            "switch_index": switch_index,
            "switch_log_prob": float(switch_logs[switch_index]),
            "reachable_load_count": int(len(reachable)),
            "legal_load_action_count": local_count,
            "retained_local_count": len(retained),
            "first_omitted_log_prob": None if first_omitted is None else float(scores[first_omitted]),
            "first_omitted_action_code": None if first_omitted is None else int(action_codes[first_omitted]),
        })
    candidates.sort(key=lambda row: (-row[0], row[1]))
    selected = candidates[:count]
    action_codes = [row[1] for row in selected]
    if len(set(action_codes)) != len(action_codes):
        raise AssertionError("Distinct joint enumeration produced a duplicate action")
    return {
        "actions": [[(code >> bit) & 1 for bit in range(n_switch + n_load)] for code in action_codes],
        "log_prob": [row[0] for row in selected],
        "probability": [math.exp(row[0]) for row in selected],
        "action_codes": action_codes,
        "switch_indices": [row[2] for row in selected],
        "load_codes": [row[3] for row in selected],
        "count": len(selected), "requested_count": count,
        "exhausted": legal_joint_count <= count,
        "legal_joint_action_count": legal_joint_count,
        "unique_action_count": len(action_codes),
        "retained_merge_count": len(candidates),
        "per_switch": per_switch,
        "generation_rule": "Exact global joint top-k via each legal topology's first k reachable-load assignments",
        "tie_break": "Equal float64 log probability: ascending sum(action[i]*2**i); no tolerance",
        "action_code_definition": "Little endian switch bits, then little endian load bits",
        "scoring_dtype": "float64",
        "switch_logits_dtype": str(raw_switch.dtype), "load_logits_dtype": str(raw_load.dtype),
        "deduplication_proof": "Unique load codes per disjoint switch index imply unique full integer action codes",
        "topk_completeness_proof": "Each omitted action has at least requested_count predecessors in its topology under the global total order",
        "physical_evaluations": 0,
    }


def propose_joint_topk_actions(policy, table, *, count=8, scenario_id=None):
    """Decode one observation with one shared network pass, preserving state.

    All neural layers run at the checkpoint's original floating dtype.  Their
    detached logits are then promoted to float64 before Bernoulli scores and
    the legal-switch normalizer are recomputed.  Consequently reported scores
    may differ by float32 rounding from ``policy.evaluate_actions``.  This
    routine does not invoke that expensive all-topology evaluator per action.
    A single additional batch evaluation records its original-dtype log
    probabilities as a numerical diagnostic, never as the ranking criterion.
    """
    import torch
    from .restoration_supported_policy import SupportedRestorationPolicy

    _positive_count(count)
    if not isinstance(policy, SupportedRestorationPolicy):
        raise TypeError("An existing SupportedRestorationPolicy is required")
    if scenario_id is not None and (not isinstance(scenario_id, str) or not scenario_id):
        raise ValueError("scenario_id must be a nonempty string or None")
    parameter = next(policy.parameters())
    if parameter.device.type != "cpu":
        raise ValueError("The frozen development decoder requires a CPU policy")
    if policy.n_load > 10:
        raise ValueError("Exact load enumeration supports at most ten load bits")
    arrays = {key: np.asarray(table[key]) for key in INPUT_KEYS}
    inputs = tuple(torch.as_tensor(arrays[key], dtype=torch.bool if arrays[key].dtype == np.bool_ else parameter.dtype,
                                   device=parameter.device).unsqueeze(0) for key in INPUT_KEYS)
    obs, forced, context, support, off = inputs
    training = policy.training
    try:
        policy.eval()
        policy._validate_supported_inputs(*inputs)
        with torch.no_grad():
            hidden = policy.encoder(obs)
            switch_logits = policy.switch_head(hidden)
            bits = policy.switch_combinations.to(obs.dtype)[None, :, :]
            expanded_hidden = hidden[:, None, :].expand(1, bits.shape[1], policy.hidden)
            load_logits = policy._load_logits(expanded_hidden, bits, context)
            value = float(policy.value_head(hidden).squeeze().item())
            if not math.isfinite(value):
                raise ValueError("Policy value must be finite")
        result = joint_topk_from_logits(
            switch_logits[0].numpy(), load_logits[0].numpy(), arrays["switch_forced_open"],
            arrays["switch_support_mask"], arrays["load_forced_off_table"], count=count,
        )
        size = result["count"]
        repeated_inputs = tuple(tensor.expand(size, *tensor.shape[1:]) for tensor in inputs)
        with torch.no_grad():
            evaluated = policy.evaluate_actions(*repeated_inputs, torch.tensor(result["actions"], dtype=parameter.dtype))
        evaluated_logs = evaluated.log_prob.tolist()
        difference = max(abs(left - right) for left, right in zip(result["log_prob"], evaluated_logs))
        if not torch.isfinite(evaluated.log_prob).all() or difference > 2e-5:
            raise AssertionError("Float64 mathematical scores differ from evaluate_actions by more than 2e-5")
        result.update({"scenario_id": scenario_id, "variant": policy.variant,
                       "value": [value] * result["count"],
                       "generation_forward_passes": 1, "density_check_forward_passes": 1,
                       "evaluate_actions_log_prob": evaluated_logs,
                       "evaluate_actions_max_abs_log_prob_difference": difference,
                       "evaluate_actions_check_atol": 2e-5,
                       "evaluate_actions_dtype": str(parameter.dtype), "policy_dtype": str(parameter.dtype),
                       "rng_used": False, "weights_updated": False})
        return result
    finally:
        policy.train(training)
