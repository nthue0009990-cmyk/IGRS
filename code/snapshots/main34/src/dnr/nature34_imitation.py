"""Scenario-balanced imitation and fixed-budget single-step PPO integration.

This module has no electrical-engine dependency. Teacher data, scenario tables
and the physical reward callback are supplied by an audited external runner.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torch.distributions import Bernoulli

from .restoration_supported_policy import ppo_supported_update

INPUT_KEYS = ("observation", "switch_forced_open", "context_table", "switch_support_mask", "load_forced_off_table")


def _role_seed(domain: str, seed: int) -> int:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("Seed must be a fixed integer")
    return int.from_bytes(hashlib.sha256(f"{domain}|{seed}".encode()).digest()[:8], "little") & ((1 << 63) - 1)


def _optimizer_coverage(policy, optimizer):
    expected = {id(parameter) for parameter in policy.parameters() if parameter.requires_grad}
    actual = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Optimizer must cover every trainable parameter exactly once")


def supported_action_log_prob_and_value(policy, obs, switch_forced_open, context_table, switch_support_mask,
                                        load_forced_off_table, actions):
    """Exact supported joint density, computing only selected load logits.

    Switch normalization still enumerates every legal assignment. The omitted
    load logits are irrelevant to a target action's likelihood; joint entropy
    is not approximated or used in the imitation loss.
    """
    policy._validate_supported_inputs(obs, switch_forced_open, context_table, switch_support_mask, load_forced_off_table)
    if actions.shape != (obs.shape[0], policy.n_switch + policy.n_load) or actions.device != obs.device:
        raise ValueError("Target actions must share [batch,switches+loads] and device")
    if not ((actions == 0) | (actions == 1)).all():
        raise ValueError("Target actions must be finite binary values")
    actions = actions.to(obs.dtype)
    hidden = policy.encoder(obs)
    switch_logits = policy.switch_head(hidden)
    batch, combinations = switch_support_mask.shape
    bits = policy.switch_combinations.to(obs.dtype)[None, :, :].expand(batch, combinations, policy.n_switch)
    scores = policy._switch_log_prob(switch_logits[:, None, :].expand_as(bits), bits,
                                    switch_forced_open[:, None, :].expand_as(bits)).masked_fill(~switch_support_mask, -torch.inf)
    logs = scores - torch.logsumexp(scores, dim=-1, keepdim=True)
    switches, loads = actions.split((policy.n_switch, policy.n_load), dim=-1)
    indices = (switches.to(torch.long) * policy.switch_index_weights).sum(dim=-1)
    rows = torch.arange(batch, device=obs.device)
    selected_context = context_table[rows, indices]
    selected_logits = policy._load_logits(hidden, switches, selected_context)
    selected_mask = load_forced_off_table[rows, indices]
    load_log_prob = Bernoulli(logits=selected_logits, validate_args=False).log_prob(loads).masked_fill(selected_mask, 0.).sum(dim=-1)
    log_prob = logs[rows, indices] + load_log_prob
    log_prob = log_prob.masked_fill((selected_mask & (loads != 0)).any(dim=-1), -torch.inf)
    return log_prob, policy.value_head(hidden).squeeze(-1)


def bc_update(policy, optimizer, obs, switch_forced_open, context_table, switch_support_mask, load_forced_off_table,
              actions, teacher_rewards, label_mask, *, value_coefficient=.5, max_grad_norm=.5):
    """One BC optimizer step: joint NLL + weighted teacher-value regression.

    Every batch row is one scheduled scenario. Masked rows have no invented
    expert target and contribute zero to both losses; the denominator remains
    the full batch size. A real zero-restoration witness has mask=True.
    """
    _optimizer_coverage(policy, optimizer)
    policy._validate_supported_inputs(obs, switch_forced_open, context_table, switch_support_mask, load_forced_off_table)
    count = obs.shape[0]
    if teacher_rewards.shape != (count,) or teacher_rewards.device != obs.device or not torch.isfinite(teacher_rewards).all():
        raise ValueError("Teacher rewards must be finite [batch] on the observation device")
    if ((teacher_rewards < 0) | (teacher_rewards > 1)).any():
        raise ValueError("Teacher nominal rewards must lie in[0,1]")
    if label_mask.shape != (count,) or label_mask.dtype != torch.bool or label_mask.device != obs.device:
        raise ValueError("Label mask must be Boolean[batch] on the observation device")
    if not math.isfinite(value_coefficient) or value_coefficient < 0 or not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("Invalid imitation coefficients")
    optimizer.zero_grad(set_to_none=True)
    if not label_mask.any():
        return {"loss": 0., "joint_nll": 0., "value_loss": 0., "grad_norm": 0., "scenarios": float(count),
                "labeled_scenarios": 0., "zero_reward_labels": 0., "optimizer_steps": 0., "all_masked_batch": True}
    logs, values = supported_action_log_prob_and_value(policy, obs, switch_forced_open, context_table, switch_support_mask,
                                                      load_forced_off_table, actions)
    if not torch.isfinite(logs[label_mask]).all():
        raise ValueError("Expert target lies outside the shared legal action support")
    nll = -torch.where(label_mask, logs, torch.zeros_like(logs)).mean()
    value_loss = torch.where(label_mask, (values - teacher_rewards).square(), torch.zeros_like(values)).mean()
    loss = nll + value_coefficient * value_loss
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite imitation loss; no optimizer update applied")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    return {"loss": loss.detach().item(), "joint_nll": nll.detach().item(), "value_loss": value_loss.detach().item(),
            "grad_norm": norm.detach().item(), "scenarios": float(count), "labeled_scenarios": float(label_mask.sum()),
            "zero_reward_labels": float((label_mask & (teacher_rewards == 0)).sum()), "optimizer_steps": 1., "all_masked_batch": False}


def make_bc_schedule(scenario_ids, labels_by_scenario, *, seed, epochs=128, batch_size=16):
    """One visit per scenario/epoch; uniformly choose one tied best topology.

    Order/label RNGs are separate from initialization and PPO randomness. The
    schedule omits variant, so paired variants use exactly the same examples.
    No-label scenarios remain as a scheduled index with label index-1.
    """
    ids = list(scenario_ids)
    if not ids or len(ids) != len(set(ids)) or set(ids) != set(labels_by_scenario):
        raise ValueError("Every unique frozen training scenario must have an explicit label list")
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in (epochs, batch_size)):
        raise ValueError("Epochs and scenario batch size must be positive integers")
    order_seed = _role_seed("NC34-D037-BC-scenario-order-v1", seed)
    label_seed = _role_seed("NC34-D037-BC-tied-topology-v1", seed)
    order_rng, label_rng = np.random.default_rng(order_seed), np.random.default_rng(label_seed)
    orders, labels = [], []
    for _ in range(epochs):
        order = order_rng.permutation(len(ids)).tolist()
        chosen = [int(label_rng.integers(len(labels_by_scenario[ids[index]]))) if labels_by_scenario[ids[index]] else -1 for index in order]
        orders.append(order)
        labels.append(chosen)
    schedule = {"seed": seed, "scenario_ids": ids, "epochs": epochs, "batch_size": batch_size,
                "scenario_order_seed": order_seed, "topology_label_seed": label_seed,
                "scenario_indices": orders, "label_indices": labels,
                "rule": "Each scenario once per epoch; one uniformly selected tied-best topology; no-label retains mask0"}
    schedule["content_sha256"] = hashlib.sha256(json.dumps(schedule, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return schedule


def _batch_inputs(tables, ids, *, dtype=torch.float32, device="cpu"):
    return tuple(torch.as_tensor(np.stack([np.asarray(tables[sid][key]) for sid in ids]),
                                 dtype=torch.bool if key in ("switch_forced_open", "switch_support_mask", "load_forced_off_table") else dtype,
                                 device=device) for key in INPUT_KEYS)


def prepare_bc_batch(policy, tables, labels_by_scenario, ids, label_indices):
    if len(ids) != len(label_indices) or not ids:
        raise ValueError("Batch scenario IDs and chosen-label indices must agree")
    parameter = next(policy.parameters())
    inputs = _batch_inputs(tables, ids, dtype=parameter.dtype, device=parameter.device)
    actions, rewards, masks = [], [], []
    for sid, chosen in zip(ids, label_indices):
        labels = labels_by_scenario[sid]
        if chosen == -1:
            if labels:
                raise ValueError("A scenario with teacher labels cannot be masked")
            table = tables[sid]
            index = int(np.flatnonzero(np.asarray(table["switch_support_mask"]))[0])
            dummy = [(index >> bit) & 1 for bit in range(policy.n_switch)] + [0] * policy.n_load
            actions.append(dummy)
            rewards.append(0.)
            masks.append(False)
        else:
            if not 0 <= chosen < len(labels):
                raise ValueError("Chosen teacher topology index is outside the frozen label list")
            label = labels[chosen]
            if not label.get("witness_found", True):
                raise ValueError("A no-witness outcome must not become an imitation target")
            actions.append(label["action"])
            rewards.append(label["teacher_reward"])
            masks.append(True)
    return (*inputs, torch.as_tensor(actions, dtype=parameter.dtype, device=parameter.device),
            torch.as_tensor(rewards, dtype=parameter.dtype, device=parameter.device), torch.as_tensor(masks, dtype=torch.bool, device=parameter.device))


def make_ppo_scenario_schedule(scenario_ids, *, seed, count=1024, domain="D037-ppo-build-v1"):
    ids = list(scenario_ids)
    if not ids or len(ids) != len(set(ids)) or not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("PPO needs every unique build ID and a positive fixed interaction budget")
    derived = _role_seed(domain, seed)
    indices = np.random.default_rng(derived).integers(0, len(ids), size=count).tolist()
    return {"seed": seed, "scenario_schedule_seed": derived, "rng_domain": domain, "scenario_ids": [ids[index] for index in indices],
            "count": count, "rule": "Uniform with replacement over all frozen build scenarios; same seed shared across variants"}


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    return value


def _save(path, value):
    Path(path).write_text(json.dumps(_json_ready(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _append(stream, value):
    stream.write(json.dumps(_json_ready(value), ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()


def _fresh_directory(run_dir, sentinel):
    path = Path(run_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    if (path / sentinel).exists() or (path / "summary.json").exists():
        raise ValueError("Use a fresh training job directory; saved evidence must not be overwritten")
    return path


def train_bc_job(policy, tables, labels_by_scenario, schedule, run_dir, *, seed, variant,
                 learning_rate=3e-4, adam_betas=(.9, .999), adam_eps=1e-5, weight_decay=0.,
                 value_coefficient=.5, max_grad_norm=.5, progress_callback=None):
    """Run the complete frozen BC schedule, without early stopping or DSS.

    The caller supplies the common initial weights. Loss normalization is one
    scenario per batch row. Final weights are always the last scheduled epoch.
    """
    path = _fresh_directory(run_dir, "bc_initial.pt")
    if schedule["seed"] != seed or policy.variant != variant:
        raise ValueError("BC job identity disagrees with the frozen schedule/policy")
    ids = schedule["scenario_ids"]
    if set(ids) != set(labels_by_scenario) or not set(ids) <= set(tables):
        raise ValueError("BC tables/labels must retain every scheduled build scenario")
    checked = {key: value for key, value in schedule.items() if key != "content_sha256"}
    digest = hashlib.sha256(json.dumps(checked, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != schedule["content_sha256"]:
        raise ValueError("BC schedule content hash differs")
    optimizer = policy.make_optimizer(learning_rate=learning_rate, betas=tuple(adam_betas), eps=adam_eps, weight_decay=weight_decay)
    config = {"seed": seed, "variant": variant, "epochs": schedule["epochs"], "scenario_batch_size": schedule["batch_size"],
              "learning_rate": learning_rate, "adam_betas": list(adam_betas), "adam_eps": adam_eps, "weight_decay": weight_decay,
              "value_coefficient": value_coefficient, "max_grad_norm": max_grad_norm, "checkpoint_rule": "Last fixed epoch; no tune access",
              "scenario_weight": "Equal scheduled scenario rows, including explicit mask0 for no expert witness"}
    _save(path / "config.json", config)
    _save(path / "schedule.json", schedule)
    torch.save({"policy": policy.state_dict(), "seed": seed, "variant": variant, "stage": "BC_INITIAL", "config": config}, path / "bc_initial.pt")
    summary = {"stage": "BC", "status": "STARTED", "seed": seed, "variant": variant, "epochs_completed": 0,
               "optimizer_steps": 0, "scheduled_scenario_visits": 0, "labeled_scenario_visits": 0, "zero_reward_label_visits": 0,
               "all_masked_batches": 0, "new_DSS_calls": 0, "new_physical_interactions": 0}
    start = time.perf_counter()
    try:
        with (path / "updates.jsonl").open("w", encoding="utf-8") as stream:
            for epoch in range(schedule["epochs"]):
                order, chosen = schedule["scenario_indices"][epoch], schedule["label_indices"][epoch]
                if sorted(order) != list(range(len(ids))) or len(chosen) != len(ids):
                    raise ValueError("Each BC epoch must visit every frozen scenario exactly once")
                epoch_checkpoint = path / "epoch_checkpoints" / f"epoch_{epoch:03d}_before_first_batch.pt"
                epoch_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"policy": policy.state_dict(), "seed": seed, "variant": variant, "epoch": epoch,
                            "stage": "BC_EPOCH_FIRST_BATCH_BEFORE_UPDATE"}, epoch_checkpoint)
                epoch_reference = {"epoch_start_checkpoint": str(epoch_checkpoint.relative_to(path)),
                                   "epoch_start_checkpoint_sha256": hashlib.sha256(epoch_checkpoint.read_bytes()).hexdigest()}
                sums = {key: 0. for key in ("loss", "joint_nll", "value_loss", "grad_norm")}
                epoch_rows = 0
                for first in range(0, len(ids), schedule["batch_size"]):
                    positions = order[first:first + schedule["batch_size"]]
                    batch_ids = [ids[index] for index in positions]
                    label_ids = chosen[first:first + schedule["batch_size"]]
                    data = prepare_bc_batch(policy, tables, labels_by_scenario, batch_ids, label_ids)
                    metric = bc_update(policy, optimizer, *data, value_coefficient=value_coefficient, max_grad_norm=max_grad_norm)
                    _append(stream, {"epoch": epoch, "batch_start": first, "scenario_ids": batch_ids,
                                     "label_indices": label_ids, "label_mask": data[-1].tolist(),
                                     "teacher_rewards": data[-2].tolist(), "target_actions": data[-3].to(torch.int64).tolist(),
                                     **(epoch_reference if first == 0 else {}), **metric})
                    for key in sums:
                        sums[key] += metric[key] * len(batch_ids)
                    epoch_rows += len(batch_ids)
                    summary["optimizer_steps"] += int(metric["optimizer_steps"])
                    summary["scheduled_scenario_visits"] += len(batch_ids)
                    summary["labeled_scenario_visits"] += int(metric["labeled_scenarios"])
                    summary["zero_reward_label_visits"] += int(metric["zero_reward_labels"])
                    summary["all_masked_batches"] += int(metric["all_masked_batch"])
                summary["epochs_completed"] = epoch + 1
                summary["latest_epoch_metrics"] = {key: value / epoch_rows for key, value in sums.items()}
                _save(path / f"epoch_{epoch:03d}.json", {**summary, "elapsed_seconds": time.perf_counter() - start})
                if progress_callback:
                    progress_callback({**summary})
        torch.save({"policy": policy.state_dict(), "optimizer": optimizer.state_dict(), "seed": seed, "variant": variant,
                    "stage": "BC_FINAL", "config": config, "schedule_sha256": digest}, path / "bc_final.pt")
        summary.update(status="COMPLETE_FIXED_BC", final_checkpoint=str(path / "bc_final.pt"))
    except BaseException as exc:
        summary.update(status="BC_STOPPED_PARTIAL", error=repr(exc), traceback=traceback.format_exc())
        torch.save({"policy": policy.state_dict(), "optimizer": optimizer.state_dict(), "summary": summary,
                    "torch_rng": torch.get_rng_state()}, path / "partial_state.pt")
        raise
    finally:
        summary["elapsed_seconds"] = time.perf_counter() - start
        _save(path / "summary.json", summary)
    return summary


def finetune_ppo_job(policy, tables, scenario_schedule, run_dir, *, seed, variant, oracle,
                     rollout_size=32, epochs=2, minibatch_size=16, learning_rate=3e-4, adam_betas=(.9, .999),
                     adam_eps=1e-5, weight_decay=0., clip_range=.2, value_coefficient=.5, entropy_coefficient=.01,
                     max_grad_norm=.5, progress_callback=None):
    """Fixed-budget PPO from caller-loaded BC weights, using a fresh Adam.

    ``oracle(context)`` receives seed/variant/schedule_index/scenario_id/action
    and returns reward,witness_found,selected_alpha plus its evidence fields.
    All physical work and its failure sealing belong to the external oracle.
    No bootstrap, demonstrations or tune feedback enter these PPO updates.
    """
    path = _fresh_directory(run_dir, "ppo_initial.pt")
    if any(not isinstance(number, int) or isinstance(number, bool) or number <= 0 for number in (rollout_size, epochs, minibatch_size)):
        raise ValueError("PPO rollout, epoch and minibatch sizes must be positive integers")
    schedule = scenario_schedule["scenario_ids"] if isinstance(scenario_schedule, Mapping) else list(scenario_schedule)
    if not schedule or not set(schedule) <= set(tables) or len(schedule) % rollout_size:
        raise ValueError("PPO schedule must use frozen build tables and complete fixed-size rollouts")
    if policy.variant != variant or not callable(oracle):
        raise ValueError("PPO policy identity or reward callback is invalid")
    if isinstance(scenario_schedule, Mapping) and scenario_schedule["seed"] != seed:
        raise ValueError("PPO scenario schedule seed differs")
    parameter = next(policy.parameters())
    if parameter.device.type != "cpu":
        raise ValueError("The fixed laptop experiment uses CPU policies")
    optimizer = policy.make_optimizer(learning_rate=learning_rate, betas=tuple(adam_betas), eps=adam_eps, weight_decay=weight_decay)
    if optimizer.state:
        raise AssertionError("PPO must start with a new Adam state, without BC momentum")
    sampling_seed = _role_seed("NC34-D037-PPO-action-sampling-v1", seed)
    torch.manual_seed(sampling_seed)
    config = {"seed": seed, "variant": variant, "rollout_size": rollout_size, "epochs": epochs, "minibatch_size": minibatch_size,
              "learning_rate": learning_rate, "adam_betas": list(adam_betas), "adam_eps": adam_eps, "weight_decay": weight_decay,
              "clip_range": clip_range, "value_coefficient": value_coefficient, "entropy_coefficient": entropy_coefficient,
              "max_grad_norm": max_grad_norm, "action_sampling_seed": sampling_seed, "optimizer_reset_after_BC": True,
              "advantage": "Fixed reward-old_value, normalized once across full rollout", "bootstrap": False,
              "checkpoint_rule": "Last fixed interaction; no online/tune model selection"}
    _save(path / "config.json", config)
    _save(path / "scenario_schedule.json", scenario_schedule)
    torch.save({"policy": policy.state_dict(), "optimizer": optimizer.state_dict(), "seed": seed, "variant": variant,
                "stage": "PPO_INITIAL_FROM_BC", "config": config, "torch_rng": torch.get_rng_state()}, path / "ppo_initial.pt")
    summary = {"stage": "BC_PLUS_PPO", "status": "STARTED", "seed": seed, "variant": variant, "interactions": 0,
               "rollouts_updated": 0, "optimizer_steps": 0, "planned_interactions": len(schedule)}
    start = time.perf_counter()
    cursor = {"phase": "before_first_rollout"}
    try:
        with (path / "actions.jsonl").open("w", encoding="utf-8") as actions_stream, (path / "attempts.jsonl").open("w", encoding="utf-8") as attempts, (path / "updates.jsonl").open("w", encoding="utf-8") as updates:
            for first in range(0, len(schedule), rollout_size):
                ids = schedule[first:first + rollout_size]
                inputs = _batch_inputs(tables, ids, dtype=parameter.dtype, device=parameter.device)
                checkpoint = path / f"rollout_{first:04d}.pt"
                torch.save({"policy": policy.state_dict(), "optimizer": optimizer.state_dict(), "seed": seed, "variant": variant,
                            "batch_start": first, "scenario_ids": ids, "torch_rng_before_sample": torch.get_rng_state()}, checkpoint)
                with torch.no_grad():
                    sampled = policy.sample(*inputs)
                    evaluated = policy.evaluate_actions(*inputs, sampled.actions)
                    if not torch.equal(sampled.log_prob, evaluated.log_prob):
                        raise AssertionError("PPO sampled action density differs before any physical evaluation")
                checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                rewards = []
                for offset, sid in enumerate(ids):
                    context = {"seed": seed, "variant": variant, "schedule_index": first + offset, "scenario_id": sid,
                               "action": sampled.actions[offset].to(torch.int64).tolist(), "rollout_checkpoint": str(checkpoint),
                               "rollout_checkpoint_sha256": checkpoint_hash,
                               "old_log_prob": sampled.log_prob[offset].item(), "old_value": sampled.value[offset].item(),
                               "old_exact_entropy": sampled.entropy[offset].item()}
                    cursor = {**context, "phase": "oracle"}
                    _append(attempts, {**context, "event": "oracle_started"})
                    before_rng = torch.get_rng_state().clone()
                    result = oracle(dict(context))
                    if not torch.equal(before_rng, torch.get_rng_state()):
                        raise AssertionError("Physical oracle must not consume the learner's Torch RNG")
                    reward, witness, alpha = result["reward"], result["witness_found"], result["selected_alpha"]
                    if not isinstance(reward, (float, int, np.floating)) or not math.isfinite(reward) or not 0 <= reward <= 1 or not isinstance(witness, (bool, np.bool_)):
                        raise ValueError("Oracle must return a finite nominal reward and explicit witness flag")
                    if (not witness and (reward != 0 or alpha is not None)) or (witness and alpha not in (0, .25, .5, .75, 1)):
                        raise ValueError("Oracle reward/alpha contradict witness status")
                    record = {**result, **context, "old_log_prob": sampled.log_prob[offset].item(),
                              "old_value": sampled.value[offset].item(), "old_exact_entropy": sampled.entropy[offset].item()}
                    _append(actions_stream, record)
                    _append(attempts, {**context, "event": "oracle_completed", "reward": float(reward), "action_id": result.get("action_id")})
                    rewards.append(float(reward))
                    summary["interactions"] += 1
                cursor = {"phase": "optimizer", "batch_start": first}
                _append(attempts, {**cursor, "event": "optimizer_started"})
                metric = ppo_supported_update(policy, optimizer, *inputs, sampled.actions,
                    torch.tensor(rewards, dtype=parameter.dtype, device=parameter.device), sampled.log_prob, sampled.value,
                    clip_range=clip_range, value_coefficient=value_coefficient, entropy_coefficient=entropy_coefficient,
                    max_grad_norm=max_grad_norm, normalize_advantage=True, epochs=epochs, minibatch_size=minibatch_size)
                _append(updates, {"batch_start": first, "rollout_checkpoint": str(checkpoint), "rollout_checkpoint_sha256": checkpoint_hash, **metric})
                _append(attempts, {**cursor, "event": "optimizer_completed", "optimizer_steps": metric["optimizer_steps"]})
                summary["rollouts_updated"] += 1
                summary["optimizer_steps"] += int(metric["optimizer_steps"])
                summary["latest_metrics"] = metric
                _save(path / f"progress_{first:04d}.json", {**summary, "elapsed_seconds": time.perf_counter() - start})
                if progress_callback:
                    progress_callback({**summary})
        torch.save({"policy": policy.state_dict(), "optimizer": optimizer.state_dict(), "seed": seed, "variant": variant,
                    "stage": "BC_PLUS_PPO_FINAL", "config": config, "torch_rng": torch.get_rng_state()}, path / "ppo_final.pt")
        summary.update(status="COMPLETE_FIXED_PPO_FINETUNING", final_checkpoint=str(path / "ppo_final.pt"))
    except BaseException as exc:
        summary.update(status="PPO_FINETUNING_STOPPED_PARTIAL", error=repr(exc), cursor=cursor, traceback=traceback.format_exc())
        torch.save({"policy": policy.state_dict(), "optimizer": optimizer.state_dict(), "summary": summary,
                    "torch_rng": torch.get_rng_state(), "partial_optimizer_completion": "UNKNOWN_IF_INTERRUPTED_DURING_OPTIMIZER"}, path / "partial_state.pt")
        raise
    finally:
        summary["elapsed_seconds"] = time.perf_counter() - start
        _save(path / "summary.json", summary)
    return summary
