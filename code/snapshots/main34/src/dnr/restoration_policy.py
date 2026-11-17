"""Small joint restoration policies, independent of the electrical simulator.

The topology context is supplied by the caller as a complete lookup table.  This
module neither infers electrical feasibility nor introduces device capacities.
Switch index ``i`` has weight ``2**i`` in that table's row index.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import Literal

import torch
from torch import Tensor, nn
from torch.distributions import Bernoulli


@dataclass(frozen=True)
class PolicyOutput:
    actions: Tensor
    log_prob: Tensor
    entropy: Tensor
    value: Tensor


class RestorationPolicy(nn.Module):
    """Parameter-matched independent and topology-conditional binary policies.

    ``switch_forced_open`` is a Boolean array: True means the switch must be 0.
    Entropy is the exact joint entropy, enumerating all switch combinations and
    analytically marginalizing the conditionally independent load decisions.
    This is intended for small switch counts (the initial author case has four).
    """

    def __init__(
        self,
        obs_dim: int,
        n_switch: int,
        n_load: int,
        context_dim: int,
        variant: Literal["independent", "conditional"],
        hidden: int = 64,
        *,
        max_enumerated_switches: int = 10,
    ) -> None:
        super().__init__()
        if any(not isinstance(n, int) or isinstance(n, bool) or n <= 0
               for n in (obs_dim, n_switch, n_load, hidden)):
            raise ValueError("observation, action and hidden dimensions must be positive integers")
        if not isinstance(context_dim, int) or isinstance(context_dim, bool) or context_dim < 0:
            raise ValueError("context_dim must be a nonnegative integer")
        if variant not in ("independent", "conditional"):
            raise ValueError("variant must be independent or conditional")
        if n_switch > max_enumerated_switches:
            raise ValueError("switch count exceeds the explicit exact-entropy enumeration limit")
        self.obs_dim = obs_dim
        self.n_switch = n_switch
        self.n_load = n_load
        self.context_dim = context_dim
        self.variant = variant
        self.hidden = hidden
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.switch_head = nn.Linear(hidden, n_switch)
        # Both variants have identical dimensions and parameter counts.
        self.load_head = nn.Linear(hidden + n_switch + context_dim, n_load)
        self.value_head = nn.Linear(hidden, 1)
        weights = 2 ** torch.arange(n_switch, dtype=torch.long)
        states = torch.arange(2 ** n_switch, dtype=torch.long)
        self.register_buffer("switch_index_weights", weights)
        self.register_buffer("switch_combinations", ((states[:, None] // weights) % 2))

    def make_optimizer(
        self,
        *,
        learning_rate: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-5,
        weight_decay: float = 0.0,
    ) -> torch.optim.Adam:
        """Create Adam after every policy layer has been registered."""
        return torch.optim.Adam(
            self.parameters(), lr=learning_rate, betas=betas, eps=eps,
            weight_decay=weight_decay,
        )

    def _validate_inputs(self, obs: Tensor, mask: Tensor, contexts: Tensor) -> None:
        if obs.ndim != 2 or obs.shape[1] != self.obs_dim or obs.shape[0] == 0:
            raise ValueError("obs must have nonempty shape [batch, obs_dim]")
        batch = obs.shape[0]
        if mask.dtype != torch.bool or mask.shape != (batch, self.n_switch):
            raise ValueError("switch_forced_open must be Boolean [batch, n_switch]")
        if contexts.shape != (batch, 2 ** self.n_switch, self.context_dim):
            raise ValueError("context_table must have shape [batch, 2**n_switch, context_dim]")
        parameter = next(self.parameters())
        if obs.device != parameter.device or obs.dtype != parameter.dtype:
            raise ValueError("obs must share the policy's floating dtype and device")
        if contexts.device != obs.device or contexts.dtype != obs.dtype or mask.device != obs.device:
            raise ValueError("context_table and mask must share the observation device/dtype")
        if not torch.isfinite(obs).all() or not torch.isfinite(contexts).all():
            raise ValueError("observations and topology contexts must be finite")

    def _load_logits(self, hidden: Tensor, switches: Tensor, context: Tensor) -> Tensor:
        if self.variant == "independent":
            switches = torch.zeros_like(switches)
            context = torch.zeros_like(context)
        return self.load_head(torch.cat((hidden, switches, context), dim=-1))

    def _context(self, contexts: Tensor, switches: Tensor) -> Tensor:
        # Invalid actions are assigned -inf later; safe indices avoid indexing
        # errors while evaluating those rows.
        safe_bits = switches.nan_to_num().clamp(0, 1).to(torch.long)
        indices = (safe_bits * self.switch_index_weights).sum(dim=-1)
        rows = torch.arange(switches.shape[0], device=switches.device)
        return contexts[rows, indices]

    @staticmethod
    def _switch_log_prob(logits: Tensor, switches: Tensor, mask: Tensor) -> Tensor:
        terms = Bernoulli(logits=logits, validate_args=False).log_prob(switches)
        terms = torch.where(mask, torch.zeros_like(terms), terms)
        result = terms.sum(dim=-1)
        forbidden = (mask & (switches != 0)).any(dim=-1)
        return result.masked_fill(forbidden, -torch.inf)

    def _joint_entropy(self, hidden: Tensor, logits: Tensor, mask: Tensor, contexts: Tensor) -> Tensor:
        switch_entropy = Bernoulli(logits=logits).entropy().masked_fill(mask, 0).sum(dim=-1)
        batch = hidden.shape[0]
        if self.variant == "independent":
            bits = hidden.new_zeros((batch, self.n_switch))
            context = hidden.new_zeros((batch, self.context_dim))
            load_entropy = Bernoulli(logits=self._load_logits(hidden, bits, context)).entropy().sum(-1)
            return switch_entropy + load_entropy
        combinations = self.switch_combinations.to(dtype=hidden.dtype)
        count = combinations.shape[0]
        bits = combinations.unsqueeze(0).expand(batch, count, self.n_switch)
        log_weights = self._switch_log_prob(
            logits[:, None, :].expand_as(bits), bits,
            mask[:, None, :].expand_as(bits),
        )
        expanded_hidden = hidden[:, None, :].expand(batch, count, self.hidden)
        load_logits = self._load_logits(expanded_hidden, bits, contexts)
        conditional_entropy = Bernoulli(logits=load_logits).entropy().sum(-1)
        return switch_entropy + (log_weights.exp() * conditional_entropy).sum(-1)

    def evaluate_actions(
        self,
        obs: Tensor,
        switch_forced_open: Tensor,
        context_table: Tensor,
        actions: Tensor,
    ) -> PolicyOutput:
        """Evaluate the joint density; nonbinary/forbidden actions get -inf."""
        self._validate_inputs(obs, switch_forced_open, context_table)
        if actions.shape != (obs.shape[0], self.n_switch + self.n_load):
            raise ValueError("actions must have shape [batch, n_switch + n_load]")
        if actions.device != obs.device:
            raise ValueError("actions must share the observation device")
        actions = actions.to(dtype=obs.dtype)
        binary = ((actions == 0) | (actions == 1)).all(dim=-1)
        # Compute using finite binary placeholders, then explicitly invalidate
        # illegal rows.  NaN/inf actions must not contaminate valid batch rows.
        safe_actions = actions.nan_to_num().clamp(0, 1)
        switches, loads = safe_actions.split((self.n_switch, self.n_load), dim=-1)
        hidden = self.encoder(obs)
        switch_logits = self.switch_head(hidden)
        context = self._context(context_table, switches)
        load_logits = self._load_logits(hidden, switches, context)
        log_prob = self._switch_log_prob(switch_logits, switches, switch_forced_open)
        log_prob = log_prob + Bernoulli(logits=load_logits, validate_args=False).log_prob(loads).sum(-1)
        log_prob = log_prob.masked_fill(~binary, -torch.inf)
        return PolicyOutput(
            actions=actions,
            log_prob=log_prob,
            entropy=self._joint_entropy(hidden, switch_logits, switch_forced_open, context_table),
            value=self.value_head(hidden).squeeze(-1),
        )

    def sample(
        self,
        obs: Tensor,
        switch_forced_open: Tensor,
        context_table: Tensor,
        deterministic: bool = False,
    ) -> PolicyOutput:
        """Sample switch bits first, then load bits conditional on that sample.

        Deterministic mode uses sequential Bernoulli modes (ties map to zero),
        which need not be the globally most probable joint action.
        """
        self._validate_inputs(obs, switch_forced_open, context_table)
        hidden = self.encoder(obs)
        switch_logits = self.switch_head(hidden)
        probabilities = switch_logits.sigmoid().masked_fill(switch_forced_open, 0)
        switches = (probabilities > 0.5).to(obs.dtype) if deterministic else torch.bernoulli(probabilities)
        context = self._context(context_table, switches)
        load_logits = self._load_logits(hidden, switches, context)
        load_probabilities = load_logits.sigmoid()
        loads = (load_probabilities > 0.5).to(obs.dtype) if deterministic else torch.bernoulli(load_probabilities)
        actions = torch.cat((switches, loads), dim=-1)
        # Reuse the exact same joint probability/entropy path for rollouts and
        # PPO reevaluation, so selected topology context cannot diverge.
        return self.evaluate_actions(obs, switch_forced_open, context_table, actions)

    def forward(
        self,
        obs: Tensor,
        switch_forced_open: Tensor,
        context_table: Tensor,
        deterministic: bool = False,
    ) -> PolicyOutput:
        return self.sample(obs, switch_forced_open, context_table, deterministic)


def ppo_update(
    policy: RestorationPolicy,
    optimizer: torch.optim.Optimizer,
    obs: Tensor,
    switch_forced_open: Tensor,
    context_table: Tensor,
    actions: Tensor,
    rewards: Tensor,
    old_log_prob: Tensor,
    old_values: Tensor | None = None,
    *,
    clip_range: float = 0.2,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01,
    max_grad_norm: float = 0.5,
    normalize_advantage: bool = True,
    epochs: int = 1,
    minibatch_size: int | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """PPO for independent single-step episodes; return equals immediate reward.

    All reward/old-policy quantities are detached.  Advantages are frozen once
    per rollout, with no next-state bootstrap.  The clipping ratio uses the
    complete switch-plus-load joint probability, never per-bit ratios.
    """
    policy._validate_inputs(obs, switch_forced_open, context_table)
    count = obs.shape[0]
    for name, values in (("rewards", rewards), ("old_log_prob", old_log_prob)):
        if values.shape != (count,) or values.device != obs.device or not torch.isfinite(values).all():
            raise ValueError(f"{name} must be a finite [batch] tensor on the observation device")
    if old_values is not None and (old_values.shape != (count,) or old_values.device != obs.device
                                   or not torch.isfinite(old_values).all()):
        raise ValueError("old_values must be finite [batch] on the observation device")
    if not 0 < clip_range < 1 or value_coefficient < 0 or entropy_coefficient < 0 or max_grad_norm <= 0:
        raise ValueError("invalid PPO coefficients")
    if not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    batch_size = count if minibatch_size is None else minibatch_size
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("minibatch_size must be a positive integer")
    expected_parameters = {id(p) for p in policy.parameters() if p.requires_grad}
    parameter_ids = [id(p) for p in chain.from_iterable(g["params"] for g in optimizer.param_groups)]
    actual_parameters = set(parameter_ids)
    if expected_parameters != actual_parameters or len(parameter_ids) != len(actual_parameters):
        raise ValueError("optimizer must cover every trainable policy parameter exactly")
    actions = actions.detach()
    with torch.no_grad():
        initial = policy.evaluate_actions(obs, switch_forced_open, context_table, actions)
        if not torch.isfinite(initial.log_prob).all():
            raise ValueError("PPO rollout includes a forbidden or nonbinary action")
        baseline = initial.value if old_values is None else old_values
        advantages = rewards.detach() - baseline.detach()
        if normalize_advantage and count > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    returns = rewards.detach()
    old_log_prob = old_log_prob.detach()
    totals: dict[str, float] = {
        "loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0,
        "entropy": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0,
        "grad_norm": 0.0,
    }
    seen = 0
    for _ in range(epochs):
        order = torch.randperm(count, generator=generator, device=obs.device)
        for start in range(0, count, batch_size):
            indices = order[start:start + batch_size]
            result = policy.evaluate_actions(
                obs[indices], switch_forced_open[indices], context_table[indices], actions[indices]
            )
            log_ratio = result.log_prob - old_log_prob[indices]
            ratio = log_ratio.exp()
            unclipped = ratio * advantages[indices]
            clipped = ratio.clamp(1 - clip_range, 1 + clip_range) * advantages[indices]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = (result.value - returns[indices]).square().mean()
            entropy = result.entropy.mean()
            loss = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite PPO objective; no optimizer step applied")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                metrics = {
                    "loss": loss, "policy_loss": policy_loss, "value_loss": value_loss,
                    "entropy": entropy, "approx_kl": ((ratio - 1) - log_ratio).mean(),
                    "clip_fraction": ((ratio - 1).abs() > clip_range).float().mean(),
                    "grad_norm": norm,
                }
                for key, value in metrics.items():
                    totals[key] += value.detach().item() * len(indices)
                seen += len(indices)
    return {**{key: value / seen for key, value in totals.items()},
            "optimizer_steps": float(epochs * ((count + batch_size - 1) // batch_size)),
            "samples": float(count), "epochs": float(epochs)}
