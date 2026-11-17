"""Operating-state adapter and ordinary episodic PPO for the D039 build pilot."""
from __future__ import annotations

import math
import numpy as np
import torch

from .nature34_operating_ledger import nominal_ledger, validate_condition, verify_setpoint_readback
from .nature34_operating_physical import verify_unchanged_attributes
from .restoration_feedback import CandidateEvidence, complete_episode_returns
from .restoration_supported_policy import ppo_supported_update


def episode_key(scenario_id, condition_id):
    return scenario_id + "|" + condition_id


def feature_static_metadata(static, generators):
    """Join the authored inventory with the already accepted runtime P/Q map."""
    authored = static["generator_reference"]
    expected = {"Generator.g" + str(row["no"]): row for row in authored}
    if set(expected) != set(generators):
        raise ValueError("Author/runtime generator identities differ")
    for name, source in expected.items():
        if generators[name]["bus"] != source["bus"] or generators[name]["available_p_kw"] != source["size"]:
            raise ValueError("Author/runtime generator capacity differs")
    return {**static, "generator_reference": generators}


def operating_receipt(points, scenario, condition, reference):
    """Bind full P/Q readbacks and demand accounting before extracting feedback."""
    validate_condition(condition, reference)
    if len(points) != 5:
        raise ValueError("Five paid alpha points required")
    projected = []
    for row in points:
        if (row["scenario_id"] != scenario["scenario_id"] or row["condition_id"] != condition["condition_id"]
                or row["fault_lines"] != scenario["fault_lines"]):
            raise ValueError("Operating receipt identity differs")
        readbacks = row["load_readback"]
        if set(readbacks) != {"before", "after", "postsolve"}:
            raise ValueError("Incomplete per-load readbacks")
        for name, ref in ((r["name"], r) for r in reference):
            for key in ("nominal_kw", "nominal_kvar"):
                if not math.isclose(readbacks["before"][name][key], ref[key], rel_tol=1e-12, abs_tol=1e-8):
                    raise ValueError("Load reset differs")
        for phase in ("after", "postsolve"):
            verify_setpoint_readback(condition, reference, readbacks[phase])
            verify_unchanged_attributes(readbacks["before"], readbacks[phase])
        planned = row["metadata_override"]["planned_enabled"]
        expected = nominal_ledger(condition, reference, {r["name"]: planned[r["name"]] for r in reference},
                                  row["metadata_override"]["deenergized_buses"])
        if expected != row["nominal_load_ledger"]:
            raise ValueError("Operating demand ledger differs from the full inventory")
        # Only after explicit operating validation may the generic receipt reader
        # consume this numerical ledger. Original point records remain untouched.
        copy = {k: v for k, v in row.items() if k not in ("load_readback", "operating_condition")}
        copy["scenario_id"] = episode_key(scenario["scenario_id"], condition["condition_id"])
        projected.append(copy)
    return CandidateEvidence.from_peak_points(projected)


def feature_schema(static, reference):
    return {
        "line_order": list(static["original_lines"]),
        "load_order": [r["name"] for r in reference],
        "generator_order": list(static["generator_reference"]),
        "p_scale_kw": sum(r["nominal_kw"] for r in reference),
        "q_scale_kvar": sum(r["nominal_kvar"] for r in reference),
        "der_p_scale_kw": sum(r["available_p_kw"] for r in static["generator_reference"].values()),
        "der_q_scale_kvar": sum(r["reference_q_kvar"] for r in static["generator_reference"].values()),
        "static_dimension": 176, "dynamic_dimension": 88, "observation_dimension": 264,
        "order": "32 fault bits;68 P;68 Q;4 DER P;4 DER Q;88 dynamic feedback features",
        "normalization": "Original authored inventory totals; no fitted/test statistics",
    }


def static_features(schema, static, reference, scenario, condition):
    requests = validate_condition(condition, reference)
    faults = set(scenario["fault_lines"])
    if len(faults) != len(scenario["fault_lines"]) or not faults <= set(schema["line_order"]):
        raise ValueError("Unknown/duplicate fault")
    if schema != feature_schema(static, reference):
        raise ValueError("Feature schema drift")
    generators = static["generator_reference"]
    values = ([float(line in faults) for line in schema["line_order"]]
              + [requests[name]["nominal_kw"] / schema["p_scale_kw"] for name in schema["load_order"]]
              + [requests[name]["nominal_kvar"] / schema["q_scale_kvar"] for name in schema["load_order"]]
              + [generators[name]["available_p_kw"] / schema["der_p_scale_kw"] for name in schema["generator_order"]]
              + [generators[name]["reference_q_kvar"] / schema["der_q_scale_kvar"] for name in schema["generator_order"]])
    if len(values) != 176 or not all(math.isfinite(value) for value in values):
        raise ValueError("Invalid static feature inventory")
    return np.asarray(values, dtype=np.float32)


def observation(base_features, episode):
    result = np.concatenate((np.asarray(base_features, dtype=np.float32),
                             np.asarray(episode.dynamic_observation(), dtype=np.float32)))
    if result.shape != (264,) or not np.isfinite(result).all():
        raise ValueError("Feedback observation schema differs")
    return result


def batch_inputs(tables, rows):
    """Rows carry a recorded state, never a recomputed state from later feedback."""
    keys = ("switch_forced_open", "context_table", "switch_support_mask", "load_forced_off_table")
    return (torch.tensor(np.stack([r["observation"] for r in rows]), dtype=torch.float32),
            *(torch.from_numpy(np.stack([tables[r["scenario_id"]][k] for r in rows])) for k in keys))


def episodic_ppo_update(policy, optimizer, inputs, actions, increments, terminal, old_logs, old_values,
                        *, generator, epochs=2, minibatch_size=28):
    """Standard PPO with complete-episode Monte Carlo targets and old baselines."""
    returns = complete_episode_returns(increments, terminal)
    targets = torch.tensor(returns, dtype=old_values.dtype, device=old_values.device)
    with torch.no_grad():
        reevaluated = policy.evaluate_actions(*inputs, actions)
        if not torch.allclose(reevaluated.log_prob, old_logs, rtol=0., atol=1e-5):
            raise ValueError("Recorded old joint densities do not match the frozen rollout policy")
        if not torch.allclose(reevaluated.value, old_values, rtol=0., atol=1e-6):
            raise ValueError("Recorded old values do not match the frozen rollout policy")
    metrics = ppo_supported_update(
        policy, optimizer, *inputs, actions, targets, old_logs, old_values,
        clip_range=.2, value_coefficient=.5, entropy_coefficient=.01, max_grad_norm=.5,
        normalize_advantage=True, epochs=epochs, minibatch_size=minibatch_size, generator=generator)
    return {**metrics, "returns": list(returns), "discount": 1.,
            "target_semantics": "Undiscounted full-episode return, not immediate incumbent increment"}
