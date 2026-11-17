"""Joint restoration policies with the same topology-dependent support.

The ``independent`` variant has state-only load logits. Its final joint
distribution is still conditional through the shared topology/load support;
it must not be described as an independent joint policy. Electrical feasibility
is not inferred from these graph-derived masks.
"""
from __future__ import annotations

from itertools import chain
import math

import torch
from torch import Tensor, nn
from torch.distributions import Bernoulli

from .restoration_policy import PolicyOutput, RestorationPolicy


class SupportedRestorationPolicy(RestorationPolicy):
    """Reuse identical layers while conditioning both variants on legal support.

    Switch probabilities are the original factorized Bernoulli probabilities
    restricted to ``switch_support_mask`` and exactly renormalized. Each chosen
    topology then has the same deterministic forced-off load bits in both
    variants. True in either forced mask means the corresponding bit must be 0.
    """

    def _validate_supported_inputs(
        self, obs: Tensor, switch_forced_open: Tensor, context_table: Tensor,
        switch_support_mask: Tensor, load_forced_off_table: Tensor,
    ) -> None:
        self._validate_inputs(obs, switch_forced_open, context_table)
        count, combinations = obs.shape[0], 2 ** self.n_switch
        if switch_support_mask.dtype != torch.bool or switch_support_mask.shape != (count, combinations):
            raise ValueError("switch_support_mask must be Boolean [batch, 2**n_switch]")
        if load_forced_off_table.dtype != torch.bool or load_forced_off_table.shape != (count, combinations, self.n_load):
            raise ValueError("load_forced_off_table must be Boolean [batch, 2**n_switch, n_load]")
        if switch_support_mask.device != obs.device or load_forced_off_table.device != obs.device:
            raise ValueError("support tables must share the observation device")
        if not switch_support_mask.any(dim=-1).all():
            raise ValueError("No legal switch support for at least one observation")
        forbidden = (self.switch_combinations.bool()[None, :, :] & switch_forced_open[:, None, :]).any(dim=-1)
        if (switch_support_mask & forbidden).any():
            raise ValueError("Switch support conflicts with a forced-open fault mask")

    def _supported_terms(
        self, obs: Tensor, switch_forced_open: Tensor, context_table: Tensor,
        switch_support_mask: Tensor, load_forced_off_table: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        hidden = self.encoder(obs)
        logits = self.switch_head(hidden)
        count, combinations = switch_support_mask.shape
        bits = self.switch_combinations.to(obs.dtype)[None, :, :].expand(count, combinations, self.n_switch)
        scores = self._switch_log_prob(
            logits[:, None, :].expand_as(bits), bits,
            switch_forced_open[:, None, :].expand_as(bits),
        ).masked_fill(~switch_support_mask, -torch.inf)
        switch_log_probs = scores - torch.logsumexp(scores, dim=-1, keepdim=True)
        probabilities = switch_log_probs.exp()
        safe_logs = switch_log_probs.masked_fill(~switch_support_mask, 0.)
        switch_entropy = -(probabilities * safe_logs).sum(dim=-1)
        expanded_hidden = hidden[:, None, :].expand(count, combinations, self.hidden)
        load_logits = self._load_logits(expanded_hidden, bits, context_table)
        conditional_load_entropy = Bernoulli(logits=load_logits).entropy().masked_fill(load_forced_off_table, 0.).sum(dim=-1)
        entropy = switch_entropy + (probabilities * conditional_load_entropy).sum(dim=-1)
        return hidden, switch_log_probs, load_logits, entropy

    def evaluate_actions(
        self, obs: Tensor, switch_forced_open: Tensor, context_table: Tensor,
        switch_support_mask: Tensor, load_forced_off_table: Tensor, actions: Tensor,
    ) -> PolicyOutput:
        """Reevaluate recorded actions without resampling their switch topology."""
        self._validate_supported_inputs(obs, switch_forced_open, context_table, switch_support_mask, load_forced_off_table)
        if actions.shape != (obs.shape[0], self.n_switch + self.n_load) or actions.device != obs.device:
            raise ValueError("actions must be [batch, n_switch+n_load] on the observation device")
        actions = actions.to(dtype=obs.dtype)
        binary = ((actions == 0) | (actions == 1)).all(dim=-1)
        safe_actions = actions.nan_to_num().clamp(0., 1.)
        switches, loads = safe_actions.split((self.n_switch, self.n_load), dim=-1)
        indices = (switches.to(torch.long) * self.switch_index_weights).sum(dim=-1)
        rows = torch.arange(obs.shape[0], device=obs.device)
        hidden, switch_logs, all_load_logits, entropy = self._supported_terms(
            obs, switch_forced_open, context_table, switch_support_mask, load_forced_off_table,
        )
        selected_load_logits = all_load_logits[rows, indices]
        selected_load_mask = load_forced_off_table[rows, indices]
        load_logs = Bernoulli(logits=selected_load_logits, validate_args=False).log_prob(loads)
        load_logs = load_logs.masked_fill(selected_load_mask, 0.).sum(dim=-1)
        log_prob = switch_logs[rows, indices] + load_logs
        forbidden_load = (selected_load_mask & (loads != 0)).any(dim=-1)
        log_prob = log_prob.masked_fill(~binary | forbidden_load, -torch.inf)
        return PolicyOutput(actions, log_prob, entropy, self.value_head(hidden).squeeze(-1))

    def sample(
        self, obs: Tensor, switch_forced_open: Tensor, context_table: Tensor,
        switch_support_mask: Tensor, load_forced_off_table: Tensor,
        deterministic: bool = False,
    ) -> PolicyOutput:
        """Sample the categorical switch index, then its masked load distribution.

        Deterministic mode chooses the most probable switch assignment followed
        by load bit modes. This sequential mode is not a joint-MAP guarantee.
        """
        self._validate_supported_inputs(obs, switch_forced_open, context_table, switch_support_mask, load_forced_off_table)
        hidden, switch_logs, all_load_logits, entropy = self._supported_terms(
            obs, switch_forced_open, context_table, switch_support_mask, load_forced_off_table,
        )
        indices = switch_logs.argmax(dim=-1) if deterministic else torch.multinomial(switch_logs.exp(), 1).squeeze(-1)
        rows = torch.arange(obs.shape[0], device=obs.device)
        switches = self.switch_combinations[indices].to(obs.dtype)
        load_logits = all_load_logits[rows, indices]
        load_mask = load_forced_off_table[rows, indices]
        probabilities = load_logits.sigmoid().masked_fill(load_mask, 0.)
        loads = (probabilities > .5).to(obs.dtype) if deterministic else torch.bernoulli(probabilities)
        actions = torch.cat((switches, loads), dim=-1)
        load_logs = Bernoulli(logits=load_logits, validate_args=False).log_prob(loads).masked_fill(load_mask, 0.).sum(dim=-1)
        return PolicyOutput(actions, switch_logs[rows, indices] + load_logs, entropy, self.value_head(hidden).squeeze(-1))

    def forward(
        self, obs: Tensor, switch_forced_open: Tensor, context_table: Tensor,
        switch_support_mask: Tensor, load_forced_off_table: Tensor,
        deterministic: bool = False,
    ) -> PolicyOutput:
        return self.sample(obs, switch_forced_open, context_table, switch_support_mask, load_forced_off_table, deterministic)


def ppo_supported_update(
    policy: SupportedRestorationPolicy,
    optimizer: torch.optim.Optimizer,
    obs: Tensor,
    switch_forced_open: Tensor,
    context_table: Tensor,
    switch_support_mask: Tensor,
    load_forced_off_table: Tensor,
    actions: Tensor,
    rewards: Tensor,
    old_log_prob: Tensor,
    old_values: Tensor | None = None,
    *,
    clip_range: float = .2,
    value_coefficient: float = .5,
    entropy_coefficient: float = .01,
    max_grad_norm: float = .5,
    normalize_advantage: bool = True,
    epochs: int = 1,
    minibatch_size: int | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Single-step PPO with frozen rollout advantages and a joint density ratio.

    Immediate reward is the value target; there is no bootstrap. Advantage
    normalization is performed once over the full rollout before minibatches.
    Support tables and recorded actions remain fixed throughout all epochs.
    """
    if not isinstance(policy, SupportedRestorationPolicy):
        raise TypeError("ppo_supported_update requires SupportedRestorationPolicy")
    policy._validate_supported_inputs(obs, switch_forced_open, context_table, switch_support_mask, load_forced_off_table)
    count = obs.shape[0]
    for name, values in (("rewards", rewards), ("old_log_prob", old_log_prob), ("old_values", old_values)):
        if values is None and name == "old_values":
            continue
        if values is None or values.shape != (count,) or values.device != obs.device or not torch.isfinite(values).all():
            raise ValueError(f"{name} must be finite [batch] on the observation device")
    coefficients = (clip_range, value_coefficient, entropy_coefficient, max_grad_norm)
    if not all(math.isfinite(value) for value in coefficients) or not 0 < clip_range < 1 or min(value_coefficient, entropy_coefficient) < 0 or max_grad_norm <= 0:
        raise ValueError("Invalid PPO coefficients")
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    batch_size = count if minibatch_size is None else minibatch_size
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("minibatch_size must be a positive integer")
    expected = {id(parameter) for parameter in policy.parameters() if parameter.requires_grad}
    registered = [id(parameter) for parameter in chain.from_iterable(group["params"] for group in optimizer.param_groups)]
    if set(registered) != expected or len(registered) != len(set(registered)):
        raise ValueError("Optimizer must cover every trainable policy parameter exactly once")
    actions = actions.detach()
    with torch.no_grad():
        initial = policy.evaluate_actions(obs, switch_forced_open, context_table, switch_support_mask, load_forced_off_table, actions)
        if not torch.isfinite(initial.log_prob).all():
            raise ValueError("Rollout contains an action outside the shared legal support")
        baseline = initial.value if old_values is None else old_values
        advantages = rewards.detach() - baseline.detach()
        if normalize_advantage and count > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    targets, old_logs = rewards.detach(), old_log_prob.detach()
    totals = {key: 0. for key in ("loss", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm")}
    seen = 0
    for _ in range(epochs):
        order = torch.randperm(count, generator=generator, device=obs.device)
        for start in range(0, count, batch_size):
            ids = order[start:start + batch_size]
            output = policy.evaluate_actions(
                obs[ids], switch_forced_open[ids], context_table[ids], switch_support_mask[ids], load_forced_off_table[ids], actions[ids],
            )
            log_ratio = output.log_prob - old_logs[ids]
            ratio = log_ratio.exp()
            policy_loss = -torch.minimum(ratio * advantages[ids], ratio.clamp(1 - clip_range, 1 + clip_range) * advantages[ids]).mean()
            value_loss = (output.value - targets[ids]).square().mean()
            entropy = output.entropy.mean()
            loss = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite supported PPO loss; current optimizer step not applied")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                metrics = {"loss": loss, "policy_loss": policy_loss, "value_loss": value_loss, "entropy": entropy,
                           "approx_kl": ((ratio - 1) - log_ratio).mean(),
                           "clip_fraction": ((ratio - 1).abs() > clip_range).float().mean(), "grad_norm": norm}
                for key, value in metrics.items():
                    totals[key] += value.detach().item() * len(ids)
                seen += len(ids)
    return {**{key: value / seen for key, value in totals.items()}, "samples": float(count), "epochs": float(epochs),
            "optimizer_steps": float(epochs * ((count + batch_size - 1) // batch_size))}
