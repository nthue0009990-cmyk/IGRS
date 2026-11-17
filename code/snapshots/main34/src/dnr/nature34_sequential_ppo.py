"""D044 masked categorical PPO utilities, with no electrical oracle access.

Command states retain ten ready bits followed by nine switch bits. Action zero
waits; action ``1 + bit`` flips that command bit. The only mask is the complete
pure-graph no-reinterruption support. Electrical rejection and reward accounting
belong to the separately logged environment, not to this policy module.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical


OBSERVATION_DIM = 26
ACTION_DIM = 20
HIDDEN = 128
TOTAL_WEIGHT = 17690
MAX_HORIZON = 12
PRECHECK_ATOL = 2e-6
PRECHECK_RTOL = 2e-5


class SequentialActorCritic(nn.Module):
    """Two shared 128-unit tanh layers, categorical actor, scalar critic."""

    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(OBSERVATION_DIM, HIDDEN), nn.Tanh(),
                                   nn.Linear(HIDDEN, HIDDEN), nn.Tanh())
        self.actor = nn.Linear(HIDDEN, ACTION_DIM)
        self.critic = nn.Linear(HIDDEN, 1)
        for layer in (self.trunk[0], self.trunk[2]):
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.actor.bias)
        nn.init.zeros_(self.critic.bias)

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        if obs.ndim not in (1, 2) or obs.shape[-1] != OBSERVATION_DIM:
            raise ValueError("obs must be [26] or [batch, 26]")
        if obs.dtype != torch.float32 or obs.device.type != "cpu":
            raise ValueError("D044 observations must be CPU float32")
        if not torch.isfinite(obs).all():
            raise ValueError("nonfinite observation")
        hidden = self.trunk(obs)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: Tensor, mask: Tensor, *, deterministic: bool = False
            ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        logits, values = self(obs)
        dist = masked_dist(logits, mask)
        # torch.argmax deterministically chooses the smallest tied action ID.
        actions = dist.logits.argmax(dim=-1) if deterministic else dist.sample()
        return actions, dist.log_prob(actions), values, dist.entropy()


def make_model(seed: int | None = None) -> SequentialActorCritic:
    """Explicit seed initializes training; seed=None preserves the global RNG.

    Evaluation should call make_model() and then load_state_dict, so constructing
    an evaluation model cannot change a running training sampler's RNG sequence.
    """
    torch.set_num_threads(2)
    if seed is not None:
        torch.manual_seed(int(seed))
        return SequentialActorCritic()
    with torch.random.fork_rng(devices=[]):
        return SequentialActorCritic()


def make_optimizer(model: SequentialActorCritic, *, learning_rate: float = 3e-4,
                   adam_eps: float = 1e-5) -> torch.optim.Adam:
    """Include both shared encoder and both heads in the frozen Adam setup."""
    return torch.optim.Adam(model.parameters(), lr=learning_rate, eps=adam_eps,
                            betas=(0.9, 0.999), weight_decay=0.0)


def masked_dist(logits: Tensor, mask: Tensor) -> Categorical:
    if logits.ndim not in (1, 2) or logits.shape[-1] != ACTION_DIM:
        raise ValueError("logits must be [20] or [batch, 20]")
    if mask.shape != logits.shape or mask.dtype != torch.bool:
        raise ValueError("mask must be Boolean with the logits shape")
    if mask.device != logits.device or not mask.any(dim=-1).all():
        raise ValueError("mask has wrong device or an empty action support")
    if not torch.isfinite(logits).all():
        raise ValueError("nonfinite unmasked logits")
    return Categorical(logits=logits.masked_fill(~mask, -torch.inf))


def graph_action_mask(graph: Any, state: int) -> np.ndarray:
    """Read only pure-graph successors; never look at physical certificates."""
    successors = set(graph.successors(int(state)))
    mask = np.asarray([int(state) in successors] +
                      [(int(state) ^ (1 << bit)) in successors for bit in range(19)],
                      dtype=np.bool_)
    if not mask[0]:
        raise ValueError("valid graph state must admit waiting")
    return mask


def make_observation(graph: Any, state: int, fault_index: int,
                     load_multiplier: float, alpha: float, remaining: int,
                     initial_horizon: int) -> np.ndarray:
    if not 0 <= int(state) < (1 << 19) or int(state) != state:
        raise ValueError("state must be a complete 19-bit command integer")
    if fault_index not in (0, 1):
        raise ValueError("fault_index must be 0 or 1")
    if not (math.isfinite(load_multiplier) and load_multiplier > 0):
        raise ValueError("invalid load multiplier")
    if not (math.isfinite(alpha) and 0 <= alpha <= 1):
        raise ValueError("invalid alpha")
    if (int(remaining) != remaining or int(initial_horizon) != initial_horizon or
            not 0 <= remaining <= initial_horizon <= MAX_HORIZON or initial_horizon == 0):
        raise ValueError("horizon must satisfy 0 <= remaining <= initial <= 12")
    if graph.total_weight != TOTAL_WEIGHT:
        raise ValueError("D044 observation normalization requires exact total weight 17690")
    weight = graph.served_weight(int(state))
    if not 0 <= weight <= TOTAL_WEIGHT:
        raise ValueError("invalid served weight")
    return np.asarray([(int(state) >> bit) & 1 for bit in range(19)] +
                      [int(fault_index == 0), int(fault_index == 1),
                       load_multiplier, alpha, remaining / MAX_HORIZON,
                       initial_horizon / MAX_HORIZON, weight / TOTAL_WEIGHT],
                      dtype=np.float32)


def compute_gae(rewards: Tensor, values: Tensor, dones: Tensor, *,
                gamma: float = 1.0, gae_lambda: float = 0.95
                ) -> tuple[Tensor, Tensor]:
    """GAE for concatenated complete episodes; every terminal bootstraps zero.

    Returns detached (advantages, returns). In particular the final transition
    must be terminal: this utility never silently bootstraps a truncated rollout.
    """
    if rewards.ndim != 1 or len(rewards) == 0 or values.shape != rewards.shape or dones.shape != rewards.shape:
        raise ValueError("rewards, values, dones must be equally sized nonempty vectors")
    if rewards.dtype != torch.float32 or values.dtype != torch.float32 or dones.dtype != torch.bool:
        raise ValueError("rewards/values must be float32 and dones Boolean")
    if any(x.device.type != "cpu" for x in (rewards, values, dones)):
        raise ValueError("D044 GAE requires CPU tensors")
    if not torch.isfinite(rewards).all() or not torch.isfinite(values).all():
        raise ValueError("nonfinite reward or old value")
    if not bool(dones[-1]):
        raise ValueError("rollout must end with a complete episode")
    if not (0 <= gamma <= 1 and 0 <= gae_lambda <= 1):
        raise ValueError("invalid GAE discount")
    with torch.no_grad():
        advantages = torch.zeros_like(rewards)
        following_advantage = torch.zeros((), dtype=torch.float32)
        following_value = torch.zeros((), dtype=torch.float32)
        for i in range(len(rewards) - 1, -1, -1):
            nonterminal = 1.0 - dones[i].float()
            delta = rewards[i] + gamma * following_value * nonterminal - values[i]
            following_advantage = delta + gamma * gae_lambda * nonterminal * following_advantage
            advantages[i] = following_advantage
            following_value = values[i]
        returns = advantages + values.detach()
    return advantages, returns


# Keep the concise interface offered to the runner, without a second algorithm.
gae = compute_gae


def _validated_rollout(model: SequentialActorCritic, rollout: Mapping[str, Tensor]
                       ) -> tuple[dict[str, Tensor], dict[str, float]]:
    names = ("obs", "mask", "actions", "logprob", "values", "rewards", "dones")
    aliases = {"mask": "masks", "logprob": "old_log_probs", "values": "old_values"}
    batch = {}
    for name in names:
        alias = aliases.get(name)
        if name in rollout and alias in rollout and not torch.equal(rollout[name], rollout[alias]):
            raise ValueError(f"conflicting rollout aliases: {name} and {alias}")
        key = name if name in rollout else alias
        if key not in rollout:
            raise ValueError(f"missing rollout field: {name}")
        batch[name] = rollout[key].detach()
    if batch["actions"].ndim != 1 or len(batch["actions"]) == 0:
        raise ValueError("nonempty one-dimensional actions required")
    count = len(batch["actions"])
    expected_shapes = {"obs": (count, OBSERVATION_DIM), "mask": (count, ACTION_DIM),
                       **{name: (count,) for name in names[2:]}}
    for name, value in batch.items():
        if value.shape != expected_shapes[name] or value.device.type != "cpu":
            raise ValueError(f"invalid {name} shape/device")
        expected_dtype = torch.bool if name in ("mask", "dones") else (
            torch.int64 if name == "actions" else torch.float32)
        if value.dtype != expected_dtype:
            raise ValueError(f"invalid {name} dtype")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"nonfinite {name}")
    actions = batch["actions"]
    if ((actions < 0) | (actions >= ACTION_DIM)).any():
        raise ValueError("action outside [0, 19]")
    if not batch["mask"][torch.arange(count), actions].all():
        raise ValueError("recorded action is forbidden by the pure-graph mask")
    with torch.no_grad():
        logits, recomputed_values = model(batch["obs"])
        recomputed_logprob = masked_dist(logits, batch["mask"]).log_prob(actions)
    stats = {}
    for name, actual in (("logprob", recomputed_logprob), ("values", recomputed_values)):
        stats[f"old_{name}_max_abs_error"] = float((batch[name] - actual).abs().max())
        if not torch.allclose(batch[name], actual, atol=PRECHECK_ATOL, rtol=PRECHECK_RTOL):
            raise ValueError(f"old {name} precheck failed: stale/corrupt rollout or changed model")
    return batch, stats


def ppo_update(model: SequentialActorCritic, optimizer: torch.optim.Optimizer,
               rollout: Mapping[str, Tensor], *, generator: torch.Generator,
               gamma: float = 1.0, gae_lambda: float = 0.95,
               clip_range: float = 0.2, ppo_epochs: int = 4,
               minibatch_size: int = 256, entropy_coefficient: float = 0.01,
               value_coefficient: float = 0.5, max_grad_norm: float = 0.5
               ) -> dict[str, Any]:
    """Run fixed epochs, with no KL early stop or outcome-based selection.

    Value loss is MSE (no hidden 1/2), multiplied once by value_coefficient=.5.
    Advantages are standardized once over the full rollout using population std.
    Loss metrics average each pre-update minibatch statistic weighted by its
    number of samples, over all epochs; they are not final-checkpoint losses.
    Save model, optimizer, global RNG and generator state *before* this call for
    independent replay. ``compute_gae`` separately exposes exact saved targets.
    """
    if generator.device.type != "cpu":
        raise ValueError("shuffle generator must be CPU")
    if ppo_epochs < 1 or int(ppo_epochs) != ppo_epochs or minibatch_size < 1 or int(minibatch_size) != minibatch_size:
        raise ValueError("positive integer epochs/minibatch size required")
    if not (0 < clip_range < 1 and max_grad_norm > 0 and entropy_coefficient >= 0 and value_coefficient >= 0):
        raise ValueError("invalid PPO coefficients")
    model_parameters = {id(p) for p in model.parameters() if p.requires_grad}
    optimized = [p for group in optimizer.param_groups for p in group["params"]]
    if {id(p) for p in optimized} != model_parameters or len(optimized) != len(model_parameters):
        raise ValueError("optimizer must include every trainable parameter exactly once")
    batch, stats = _validated_rollout(model, rollout)
    advantages, returns = compute_gae(batch["rewards"], batch["values"], batch["dones"],
                                      gamma=gamma, gae_lambda=gae_lambda)
    for name, expected in (("advantages", advantages), ("returns", returns)):
        if name in rollout and (rollout[name].dtype != torch.float32 or
                                rollout[name].device.type != "cpu" or
                                not torch.equal(rollout[name], expected)):
            raise ValueError(f"saved {name} differs from recomputed complete-episode GAE")
    advantage_mean = advantages.mean()
    advantage_std = advantages.std(unbiased=False)
    normalized_advantages = (advantages - advantage_mean) / (advantage_std + 1e-8)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    count = len(advantages)
    totals = {name: 0.0 for name in ("loss", "policy_loss", "value_loss", "entropy",
                                   "approx_kl", "clip_fraction", "grad_norm")}
    permutation_hashes = []
    steps = 0
    samples = 0
    minibatch_sizes = []
    for _ in range(ppo_epochs):
        permutation = torch.randperm(count, generator=generator)
        permutation_hashes.append(hashlib.sha256(permutation.numpy().astype("<i8", copy=False).tobytes()).hexdigest())
        for offset in range(0, count, minibatch_size):
            indices = permutation[offset:offset + minibatch_size]
            logits, values = model(batch["obs"][indices])
            distribution = masked_dist(logits, batch["mask"][indices])
            new_logprob = distribution.log_prob(batch["actions"][indices])
            logratio = new_logprob - batch["logprob"][indices]
            ratio = logratio.exp()
            advantage = normalized_advantages[indices]
            policy_loss = -torch.minimum(ratio * advantage,
                                         ratio.clamp(1.0 - clip_range, 1.0 + clip_range) * advantage).mean()
            value_loss = (values - returns[indices]).square().mean()
            entropy = distribution.entropy().mean()
            loss = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm,
                                                 error_if_nonfinite=True)
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError("nonfinite model after optimizer step")
            with torch.no_grad():
                metrics = {"loss": loss, "policy_loss": policy_loss,
                           "value_loss": value_loss, "entropy": entropy,
                           "approx_kl": ((ratio - 1.0) - logratio).mean(),
                           "clip_fraction": ((ratio - 1.0).abs() > clip_range).float().mean(),
                           "grad_norm": grad_norm}
                size = len(indices)
                for name, value in metrics.items():
                    totals[name] += float(value) * size
            steps += 1
            samples += size
            minibatch_sizes.append(size)
    deltas = {name: (parameter.detach() - before[name]).double()
              for name, parameter in model.named_parameters()}
    def delta_l2(prefix: str = "") -> float:
        return math.sqrt(sum(float(delta.square().sum()) for name, delta in deltas.items()
                             if name.startswith(prefix)))
    stats.update({name: value / samples for name, value in totals.items()})
    stats.update({"rollout_steps": count, "complete_episodes": int(batch["dones"].sum()),
                  "optimizer_steps": steps, "sample_visits": samples,
                  "minibatch_sizes": minibatch_sizes, "permutation_sha256": permutation_hashes,
                  "advantage_mean": float(advantage_mean), "advantage_population_std": float(advantage_std),
                  "return_mean": float(returns.mean()), "return_min": float(returns.min()),
                  "return_max": float(returns.max()), "parameter_delta_l2": delta_l2(),
                  "parameter_delta_max_abs": max(float(delta.abs().max()) for delta in deltas.values()),
                  "actor_delta_l2": delta_l2("actor."), "critic_delta_l2": delta_l2("critic."),
                  "trunk_delta_l2": delta_l2("trunk."),
                  "changed_parameter_tensors": sum(bool(delta.abs().max() > 0) for delta in deltas.values()),
                  "precheck_atol": PRECHECK_ATOL, "precheck_rtol": PRECHECK_RTOL,
                  "value_loss_definition": "MSE; total applies value_coefficient once",
                  "metrics_aggregation": "sample-weighted pre-step minibatch means over all epochs"})
    return stats
