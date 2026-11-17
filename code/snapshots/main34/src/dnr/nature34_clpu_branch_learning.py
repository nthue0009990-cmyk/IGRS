"""D059 shared candidate scorer and unchanged clipped PPO, CPU only.

The explicit CLPU association is the only raw/associated difference.  Variable
candidate counts are padded under a mask; no unqueried physical answer enters
the network.  Checkpoints retain architecture, optimizer and RNG provenance.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import torch
from torch import nn
from torch.distributions import Categorical

WINDOW = 32
HIDDEN = 128
PRECHECK_ATOL, PRECHECK_RTOL = 2e-6, 2e-5


def encode_observations(observations, representation=None):
    if not observations:
        raise ValueError('Nonempty observations required')
    first = observations[0]
    width = len(first['candidate_features'][0])
    global_width = len(first['global_features'])
    candidates = torch.zeros((len(observations), WINDOW, width), dtype=torch.float32)
    context = torch.zeros((len(observations), global_width), dtype=torch.float32)
    masks = torch.zeros((len(observations), WINDOW), dtype=torch.bool)
    for i, obs in enumerate(observations):
        if representation is not None and obs['representation'] != representation:
            raise ValueError('Representation mismatch')
        size = len(obs['candidate_features'])
        if not 0 < size <= WINDOW or len(obs['mask']) != size or len(obs['branch_ids']) != size:
            raise ValueError('Invalid candidate count/mask')
        x = torch.tensor(obs['candidate_features'], dtype=torch.float32)
        g = torch.tensor(obs['global_features'], dtype=torch.float32)
        m = torch.tensor(obs['mask'], dtype=torch.bool)
        if x.shape != (size, width) or g.shape != (global_width,) or not m.any():
            raise ValueError('Feature shape or empty mask')
        if not torch.isfinite(x).all() or not torch.isfinite(g).all():
            raise ValueError('Nonfinite features')
        candidates[i, :size], context[i], masks[i, :size] = x, g, m
    return {'candidates': candidates, 'global': context, 'mask': masks}


class CandidateActorCritic(nn.Module):
    kind = 'actor_critic'

    def __init__(self, candidate_dim, global_dim, representation='raw', hidden=HIDDEN):
        super().__init__()
        if representation not in ('raw', 'associated'):
            raise ValueError('Unknown representation')
        self.representation, self.associated = representation, representation == 'associated'
        self.architecture = dict(candidate_dim=int(candidate_dim), global_dim=int(global_dim),
                                 representation=representation, hidden=int(hidden))
        self.trunk = nn.Sequential(nn.Linear(candidate_dim + global_dim, hidden), nn.Tanh(),
                                   nn.Linear(hidden, hidden), nn.Tanh())
        self.actor, self.critic = nn.Linear(hidden, 1), nn.Linear(hidden, 1)
        for layer in (self.trunk[0], self.trunk[2]):
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor.weight, gain=.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.)
        nn.init.zeros_(self.actor.bias)
        nn.init.zeros_(self.critic.bias)

    def hidden(self, candidates, context, mask):
        if candidates.ndim != 3 or context.ndim != 2 or mask.shape != candidates.shape[:2]:
            raise ValueError('Invalid tensor dimensions')
        if not mask.any(dim=1).all() or mask.dtype != torch.bool:
            raise ValueError('Invalid legal mask')
        if candidates.dtype != torch.float32 or context.dtype != torch.float32:
            raise ValueError('CPU float32 features required')
        if any(t.device.type != 'cpu' for t in (candidates, context, mask)):
            raise ValueError('CPU only')
        return self.trunk(torch.cat((candidates, context[:, None, :].expand(-1, candidates.shape[1], -1)), -1))

    def forward(self, candidates, context, mask):
        h = self.hidden(candidates, context, mask)
        logits = self.actor(h).squeeze(-1).masked_fill(~mask, -torch.inf)
        pooled = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        return logits, self.critic(pooled).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, mask=None, *, deterministic=False, generator=None):
        if mask is not None and list(mask) != obs['mask']:
            raise ValueError('Mask differs from observation')
        x = encode_observations([obs], self.representation)
        logits, values = self(x['candidates'], x['global'], x['mask'])
        dist = Categorical(logits=logits)
        action = logits.argmax(-1) if deterministic else torch.multinomial(dist.probs, 1, generator=generator).squeeze(-1)
        return {'action': int(action[0]), 'logprob': float(dist.log_prob(action)[0]), 'value': float(values[0])}


class SupervisedScorer(CandidateActorCritic):
    kind = 'supervised'

    def __init__(self, candidate_dim, global_dim, representation='associated', hidden=HIDDEN):
        super().__init__(candidate_dim, global_dim, representation, hidden)
        del self.actor, self.critic
        self.feasible, self.cost = nn.Linear(hidden, 1), nn.Linear(hidden, 1)

    def predict(self, candidates, context, mask):
        h = self.hidden(candidates, context, mask)
        return self.feasible(h).squeeze(-1), self.cost(h).squeeze(-1).sigmoid()

    @torch.no_grad()
    def act(self, obs, mask=None, *, deterministic=True, generator=None):
        if not deterministic:
            raise ValueError('Supervised comparator is deterministic')
        x = encode_observations([obs], self.representation)
        feasibility, cost = self.predict(x['candidates'], x['global'], x['mask'])
        gain = (x['candidates'][..., 0] - x['global'][:, 0:1]).clamp_min(0.)
        score = feasibility.sigmoid() * gain / (1. + 256. * cost)
        score = score.masked_fill(~x['mask'], -torch.inf)
        # Public graph gain, then the shared window order resolve exact ties.
        best = max(range(len(obs['mask'])), key=lambda i: (float(score[0, i]), float(gain[0, i]), -i))
        return {'action': best, 'logprob': 0., 'value': 0.}


def checkpoint(model, *, seed, optimizer=None, training_metadata=None):
    return {'kind': model.kind, 'representation': model.representation,
            'architecture': deepcopy(model.architecture), 'model_state': deepcopy(model.state_dict()),
            'optimizer_state': deepcopy(optimizer.state_dict()) if optimizer else None,
            'seed': int(seed), 'training_metadata': deepcopy(training_metadata or {}),
            'rng_state': torch.get_rng_state().clone()}


def load_policy(path, representation=None):
    data = torch.load(Path(path), map_location='cpu', weights_only=False)
    if representation is not None and data['representation'] != representation:
        raise ValueError('Checkpoint representation mismatch')
    cls = {'actor_critic': CandidateActorCritic, 'supervised': SupervisedScorer}[data['kind']]
    # Loading must not advance the caller's random stream.
    with torch.random.fork_rng(devices=[]):
        model = cls(**data['architecture'])
    model.load_state_dict(data['model_state'], strict=True)
    model.eval()
    return model


def gae(rewards, values, dones, gamma=1., lam=.95):
    if rewards.ndim != 1 or rewards.shape != values.shape or rewards.shape != dones.shape or not len(rewards):
        raise ValueError('Invalid rollout vectors')
    if not bool(dones[-1]):
        raise ValueError('Only complete search episodes may be used')
    advantages = torch.zeros_like(rewards)
    last = torch.zeros((), dtype=torch.float32)
    next_value = torch.zeros((), dtype=torch.float32)
    for i in range(len(rewards) - 1, -1, -1):
        nonterminal = 1. - dones[i].float()
        delta = rewards[i] + gamma * next_value * nonterminal - values[i]
        last = delta + gamma * lam * nonterminal * last
        advantages[i] = last
        next_value = values[i]
    return advantages, advantages + values


def ppo_update(model, optimizer, rollout, generator, config):
    x, g, mask = rollout['candidates'], rollout['global'], rollout['mask']
    actions, old_lp, old_v = rollout['actions'], rollout['old_logprob'], rollout['values']
    count = len(actions)
    if count == 0 or not mask[torch.arange(count), actions].all():
        raise ValueError('Empty or illegal rollout')
    with torch.no_grad():
        lp_parts, v_parts = [], []
        for i in range(count):
            logits, v = model(x[i:i+1], g[i:i+1], mask[i:i+1])
            lp_parts.append(Categorical(logits=logits).log_prob(actions[i:i+1]))
            v_parts.append(v)
        actual_lp, actual_v = torch.cat(lp_parts), torch.cat(v_parts)
        if not torch.allclose(old_lp, actual_lp, atol=PRECHECK_ATOL, rtol=PRECHECK_RTOL):
            raise ValueError('Old policy logprob precheck failed')
        if not torch.allclose(old_v, actual_v, atol=PRECHECK_ATOL, rtol=PRECHECK_RTOL):
            raise ValueError('Old value precheck failed')
        advantages, returns = gae(rollout['rewards'], old_v, rollout['dones'], config['gamma'], config['gae_lambda'])
        standardized = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    orders, losses = [], []
    for epoch in range(config['ppo_epochs']):
        order = torch.randperm(count, generator=generator)
        orders.append(order.tolist())
        for start in range(0, count, config['minibatch_size']):
            idx = order[start:start+config['minibatch_size']]
            logits, value = model(x[idx], g[idx], mask[idx])
            dist = Categorical(logits=logits)
            ratio = (dist.log_prob(actions[idx]) - old_lp[idx]).exp()
            pg = -torch.minimum(ratio * standardized[idx], ratio.clamp(1-config['clip_range'], 1+config['clip_range']) * standardized[idx]).mean()
            vl, entropy = (value - returns[idx]).square().mean(), dist.entropy().mean()
            loss = pg + config['value_coefficient'] * vl - config['entropy_coefficient'] * entropy
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite PPO loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            if not torch.isfinite(grad):
                raise ValueError('Nonfinite PPO gradient')
            optimizer.step()
            losses.append({'epoch': epoch, 'start': start, 'count': len(idx), 'policy': float(pg.detach()),
                           'value': float(vl.detach()), 'entropy': float(entropy.detach()),
                           'total': float(loss.detach()), 'gradient_norm': float(grad)})
    return {'orders': orders, 'losses': losses, 'advantages': advantages, 'returns': returns,
            'old_logprob_max_error': float((old_lp-actual_lp).abs().max()),
            'old_value_max_error': float((old_v-actual_v).abs().max()), 'transition_count': count}
