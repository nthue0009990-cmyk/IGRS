"""Independent D059 policy/learning replay; no production learner is imported.

The physical/branch auditor establishes trajectory observations and rewards.
This module independently binds those saved facts to model decisions, training
populations, fixed update samples and numerical optimizer results. It never
initializes a simulator or launches a new learning experiment.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn


RAW_DIM, ASSOCIATED_DIM, GLOBAL_DIM, WINDOW, HIDDEN = 261, 269, 79, 32, 128
ATOL, RTOL = 2e-6, 2e-5


def require(value, message):
    if not value:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load(path):
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def lines(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def close(actual, expected, label, *, atol=ATOL, rtol=RTOL):
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and math.isfinite(actual) and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol),
            f"{label}: {actual!r} differs from {expected!r}")


def same_tree(actual, expected, label):
    """Exact checkpoint/optimizer equality, preserving dtype and shape."""
    if isinstance(expected, torch.Tensor):
        require(isinstance(actual, torch.Tensor) and actual.dtype == expected.dtype
                and actual.shape == expected.shape and torch.equal(actual, expected), label)
    elif isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), label + " keys")
        for key in expected:
            same_tree(actual[key], expected[key], f"{label}/{key}")
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, type(expected)) and len(actual) == len(expected), label + " length/type")
        for index, item in enumerate(expected):
            same_tree(actual[index], item, f"{label}/{index}")
    else:
        require(type(actual) is type(expected) and actual == expected, label)


def tensorize(observations, representation):
    """Independent padding; forbidden/missing branches never gain mass."""
    require(representation in ("raw", "associated"), "Unknown representation")
    width = ASSOCIATED_DIM if representation == "associated" else RAW_DIM
    candidates = torch.zeros((len(observations), WINDOW, width), dtype=torch.float32)
    global_features = torch.zeros((len(observations), GLOBAL_DIM), dtype=torch.float32)
    mask = torch.zeros((len(observations), WINDOW), dtype=torch.bool)
    for index, observation in enumerate(observations):
        require(observation["representation"] == representation, "Observation representation differs")
        rows, ids, support = observation["candidate_features"], observation["branch_ids"], observation["mask"]
        require(0 < len(rows) <= WINDOW and len(rows) == len(ids) == len(support)
                and len(set(ids)) == len(ids) and all(type(v) is bool for v in support) and any(support),
                "Malformed observed candidate support")
        require(len(observation["global_features"]) == GLOBAL_DIM
                and all(len(row) == width for row in rows), "Observation feature dimension differs")
        candidates[index, :len(rows)] = torch.tensor(rows, dtype=torch.float32)
        global_features[index] = torch.tensor(observation["global_features"], dtype=torch.float32)
        mask[index, :len(rows)] = torch.tensor(support, dtype=torch.bool)
    require(torch.isfinite(candidates).all().item() and torch.isfinite(global_features).all().item(),
            "Nonfinite input features")
    return candidates, global_features, mask


class IndependentActorCritic(nn.Module):
    """Independent implementation of the frozen candidate-wise architecture."""

    def __init__(self, associated):
        super().__init__()
        self.associated = bool(associated)
        width = ASSOCIATED_DIM if self.associated else RAW_DIM
        self.trunk = nn.Sequential(nn.Linear(width + GLOBAL_DIM, HIDDEN), nn.Tanh(),
                                   nn.Linear(HIDDEN, HIDDEN), nn.Tanh())
        self.actor = nn.Linear(HIDDEN, 1)
        self.critic = nn.Linear(HIDDEN, 1)
        for layer in (self.trunk[0], self.trunk[2]):
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor.weight, gain=.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.actor.bias)
        nn.init.zeros_(self.critic.bias)

    def forward(self, candidates, globals_, mask):
        joined = torch.cat((candidates, globals_[:, None, :].expand(-1, candidates.shape[1], -1)), dim=-1)
        hidden = self.trunk(joined)
        logits = self.actor(hidden).squeeze(-1).masked_fill(~mask, -torch.inf)
        pooled = (hidden * mask[:, :, None]).sum(dim=1) / mask.sum(dim=1)[:, None]
        values = self.critic(pooled).squeeze(-1)
        return logits, values


class IndependentSupervised(IndependentActorCritic):
    def __init__(self, associated=True):
        super().__init__(associated)
        del self.actor, self.critic
        self.feasible = nn.Linear(HIDDEN, 1)
        self.cost = nn.Linear(HIDDEN, 1)

    def forward(self, candidates, globals_, mask):
        x = torch.cat((candidates, globals_[:, None, :].expand(-1, candidates.shape[1], -1)), dim=-1)
        hidden = self.trunk(x)
        return self.feasible(hidden).squeeze(-1), self.cost(hidden).squeeze(-1).sigmoid()


def load_independent(checkpoint):
    saved = load(checkpoint) if isinstance(checkpoint, (str, Path)) else checkpoint
    representation = saved["representation"]
    require(representation in ("raw", "associated"), "Checkpoint representation differs")
    expected = {"candidate_dim": ASSOCIATED_DIM if representation == "associated" else RAW_DIM,
                "global_dim": GLOBAL_DIM, "representation": representation, "hidden": HIDDEN}
    require(saved["architecture"] == expected, "Checkpoint architecture differs from frozen dimensions")
    require(saved["kind"] in ("actor_critic", "supervised"), "Unknown checkpoint kind")
    with torch.random.fork_rng(devices=[]):
        cls = IndependentSupervised if saved["kind"] == "supervised" else IndependentActorCritic
        model = cls(representation == "associated")
    model.load_state_dict(saved["model_state"], strict=True)
    model.eval()
    return model, saved


def check_policy_transitions(checkpoint, transitions, spec):
    """Replay each selected action and old policy evidence from a bound model."""
    torch.set_num_threads(1)
    checkpoint = Path(checkpoint)
    require(sha(checkpoint) == spec["checkpoint_sha256"], "Policy identity changed")
    model, saved = load_independent(checkpoint)
    require(saved["representation"] == spec["representation"], "Policy/episode representation mismatch")
    generator = torch.Generator(device="cpu").manual_seed(spec.get("sampling_seed", spec["seed"]))
    count = 0
    with torch.no_grad():
        for index, row in enumerate(transitions):
            require(row["transition_id"] == index and row["checkpoint_sha256"] == spec["checkpoint_sha256"], "Transition/model identity differs")
            require(row["representation"] == spec["representation"] and row["method"] == spec["method"], "Transition method/representation differs")
            require(row["seed"] == spec["seed"] and row["sampling_seed"] == spec.get("sampling_seed", spec["seed"]),
                    "Transition model/sampling seed differs")
            observation = row["obs_" + spec["representation"]]
            require(row["obs"] == observation and row["mask"] == observation["mask"], "Actual policy observation differs from retained representation")
            require(row["obs_raw"]["branch_ids"] == row["obs_associated"]["branch_ids"]
                    and row["obs_raw"]["mask"] == row["obs_associated"]["mask"], "Representation changed available actions")
            x, g, mask = tensorize([observation], spec["representation"])
            output, value = model(x, g, mask)
            if saved["kind"] == "supervised":
                require(spec["deterministic"], "Supervised comparator cannot sample actions")
                gain = torch.clamp_min(x[..., 0] - g[:, 0:1], 0.)
                score = (output.sigmoid() * gain / (1. + 256. * value)).masked_fill(~mask, -torch.inf)
                selected = max(range(len(observation["mask"])), key=lambda i: (float(score[0, i]), float(gain[0, i]), -i))
                expected_lp = expected_value = 0.0
            else:
                dist = torch.distributions.Categorical(logits=output)
                action = output.argmax(dim=-1) if spec["deterministic"] else torch.multinomial(dist.probs, 1, generator=generator).squeeze(-1)
                selected = int(action[0])
                expected_lp, expected_value = float(dist.log_prob(action)[0]), float(value[0])
            require(type(row["action"]) is int and row["action"] == selected and mask[0, selected], "Policy action or exact sampler replay differs")
            require(row["branch_id"] == observation["branch_ids"][selected], "Selected branch identity differs")
            close(row["logprob"], expected_lp, "Policy log probability")
            close(row["value"], expected_value, "Policy critic value")
            count += 1
    return {"policy_decisions": count, "checkpoint_sha256": spec["checkpoint_sha256"],
            "representation": spec["representation"], "kind": saved["kind"]}


def independent_gae(rewards, values, dones, gamma=1.0, gae_lambda=.95):
    require(rewards.ndim == values.ndim == dones.ndim == 1 and len(rewards) == len(values) == len(dones)
            and len(rewards) > 0 and bool(dones[-1]), "GAE needs complete concatenated episodes")
    advantages = torch.zeros_like(rewards)
    accumulator = torch.zeros((), dtype=torch.float32)
    future_value = torch.zeros((), dtype=torch.float32)
    for index in reversed(range(len(rewards))):
        continuing = 1.0 - dones[index].float()
        delta = rewards[index] + gamma * future_value * continuing - values[index]
        accumulator = delta + gamma * gae_lambda * continuing * accumulator
        advantages[index] = accumulator
        future_value = values[index]
    return advantages, advantages + values


def check_pairing_population(training_contexts, evaluation_contexts, teacher_cells):
    """No event/fault leakage and no outcome-dependent omission of contexts."""
    key = lambda row: (row["scenario_id"], row["event_id"])
    train, evaluation = {key(r) for r in training_contexts}, {key(r) for r in evaluation_contexts}
    require(len(train) == len(training_contexts) and len(evaluation) == len(evaluation_contexts), "Duplicate context")
    require(train.isdisjoint(evaluation), "Development context leaked into training")
    require({f for f, _ in train}.isdisjoint({f for f, _ in evaluation}), "Development fault leaked into training")
    require({e for _, e in train}.isdisjoint({e for _, e in evaluation}), "Development event leaked into training")
    population = Counter()
    for cell in teacher_cells:
        require(cell.get("phase") in ("teacher", "reference") and key(cell) in train,
                "Nontraining teacher/reference source")
        require(cell.get("method") in ("greedy", "beam"), "Teacher source is not a frozen rule")
        population[key(cell)] += 1
    require(set(population) == train, "Teacher population omits a frozen training context")
    return {"training_contexts": len(train), "evaluation_contexts": len(evaluation),
            "teacher_cells": sum(population.values())}


def validate_dataset(data, representation, *, rollout=False):
    """Reject tensor/schema corruption before attempting numerical replay."""
    width = ASSOCIATED_DIM if representation == "associated" else RAW_DIM
    n = len(data["actions"])
    require(n > 0 and data["candidates"].shape == (n, WINDOW, width)
            and data["global"].shape == (n, GLOBAL_DIM)
            and data["mask"].shape == (n, WINDOW), "Dataset dimensions differ")
    for name in ("candidates", "global"):
        require(data[name].dtype == torch.float32 and data[name].device.type == "cpu"
                and torch.isfinite(data[name]).all().item(), "Invalid dataset floats: " + name)
    require(data["mask"].dtype == torch.bool and data["mask"].any(dim=1).all().item(), "Invalid dataset mask")
    actions = data["actions"]
    require(actions.shape == (n,) and actions.dtype == torch.int64
            and (actions >= 0).all().item() and (actions < WINDOW).all().item()
            and data["mask"][torch.arange(n), actions].all().item(), "Invalid dataset action")
    if rollout:
        for name in ("old_logprob", "values", "rewards"):
            require(data[name].shape == (n,) and data[name].dtype == torch.float32
                    and torch.isfinite(data[name]).all().item(), "Invalid rollout vector: " + name)
        require(data["dones"].shape == (n,) and data["dones"].dtype == torch.bool
                and bool(data["dones"][-1]), "Partial rollout cannot be learned")
    return n


def bind_rollout(rollout, transitions, representation):
    """Exact binding to independently audited episode trajectories."""
    n = validate_dataset(rollout, representation, rollout=True)
    require(len(transitions) == n, "Rollout changed retained sample population")
    x, g, mask = tensorize([r["obs_" + representation] for r in transitions], representation)
    expected = {"candidates": x, "global": g, "mask": mask,
                "actions": torch.tensor([r["action"] for r in transitions], dtype=torch.long),
                "old_logprob": torch.tensor([r["logprob"] for r in transitions], dtype=torch.float32),
                "values": torch.tensor([r["value"] for r in transitions], dtype=torch.float32),
                "rewards": torch.tensor([r["reward"] for r in transitions], dtype=torch.float32),
                "dones": torch.tensor([r["done"] for r in transitions], dtype=torch.bool)}
    for key in expected:
        same_tree(rollout[key], expected[key], "Saved rollout/trajectory " + key)
    return n


def adam(parameters, config):
    return torch.optim.Adam(parameters, lr=config["learning_rate"], eps=config["adam_eps"])


def compare_model_optimizer(model, optimizer, saved, label):
    same_tree(saved["model_state"], model.state_dict(), label + " model")
    same_tree(saved["optimizer_state"], optimizer.state_dict(), label + " optimizer")


def replay_ppo_update(before, after, rollout, update, config, *, generator_state=None):
    """Recompute standard PPO from frozen samples without production helpers.

    This is an audit replay only: no environment is queried, no new rollout is
    collected, and the resulting parameters are discarded after comparison.
    """
    torch.set_num_threads(1)
    model, before = load_independent(before)
    after = load(after) if isinstance(after, (str, Path)) else after
    require(before["kind"] == after["kind"] == "actor_critic"
            and before["representation"] == after["representation"]
            and before["seed"] == after["seed"], "PPO checkpoint lineage differs")
    model.train()
    optimizer = adam(model.parameters(), config)
    require(before["optimizer_state"] is not None, "Missing PPO optimizer starting state")
    optimizer.load_state_dict(deepcopy(before["optimizer_state"]))
    n = validate_dataset(rollout, before["representation"], rollout=True)
    require(update["transition_count"] == n, "PPO transition count differs")
    x, g, mask = rollout["candidates"], rollout["global"], rollout["mask"]
    actions, old_lp, old_values = rollout["actions"], rollout["old_logprob"], rollout["values"]
    with torch.no_grad():
        logps, values = [], []
        for row in range(n):
            logits, value = model(x[row:row+1], g[row:row+1], mask[row:row+1])
            logps.append(torch.distributions.Categorical(logits=logits).log_prob(actions[row:row+1]))
            values.append(value)
        actual_lp, actual_value = torch.cat(logps), torch.cat(values)
        require(torch.allclose(old_lp, actual_lp, atol=ATOL, rtol=RTOL), "Old log probability is not from the frozen policy")
        require(torch.allclose(old_values, actual_value, atol=ATOL, rtol=RTOL), "Old value is not from the frozen policy")
        advantage, returns = independent_gae(rollout["rewards"], old_values, rollout["dones"],
                                             config["gamma"], config["gae_lambda"])
        normalized = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
    for key, expected in (("advantages", advantage), ("returns", returns)):
        actual = update[key] if isinstance(update[key], torch.Tensor) else torch.tensor(update[key], dtype=torch.float32)
        same_tree(actual, expected, "PPO " + key)
    close(update["old_logprob_max_error"], float((old_lp - actual_lp).abs().max()), "Old logprob precheck error", atol=0., rtol=0.)
    close(update["old_value_max_error"], float((old_values - actual_value).abs().max()), "Old value precheck error", atol=0., rtol=0.)
    require(len(update["orders"]) == config["ppo_epochs"], "PPO epoch count differs")
    sampler = None
    if generator_state is not None:
        sampler = torch.Generator(device="cpu")
        sampler.set_state(generator_state)
    losses = []
    for epoch, saved_order in enumerate(update["orders"]):
        require(len(saved_order) == n and all(type(v) is int for v in saved_order)
                and sorted(saved_order) == list(range(n)), "PPO minibatch order drops/duplicates a transition")
        order = torch.tensor(saved_order, dtype=torch.long)
        if sampler is not None:
            same_tree(order, torch.randperm(n, generator=sampler), "PPO seeded epoch permutation")
        for start in range(0, n, config["minibatch_size"]):
            index = order[start:start + config["minibatch_size"]]
            logits, value = model(x[index], g[index], mask[index])
            distribution = torch.distributions.Categorical(logits=logits)
            ratio = torch.exp(distribution.log_prob(actions[index]) - old_lp[index])
            surrogate = torch.minimum(ratio * normalized[index],
                                      ratio.clamp(1-config["clip_range"], 1+config["clip_range"]) * normalized[index])
            policy_loss = -surrogate.mean()
            value_loss = (value - returns[index]).square().mean()
            entropy = distribution.entropy().mean()
            loss = policy_loss + config["value_coefficient"] * value_loss - config["entropy_coefficient"] * entropy
            require(bool(torch.isfinite(loss)), "Nonfinite replay PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            require(bool(torch.isfinite(norm)), "Nonfinite replay PPO gradient")
            optimizer.step()
            losses.append({"epoch": epoch, "start": start, "count": len(index),
                           "policy": float(policy_loss.detach()), "value": float(value_loss.detach()),
                           "entropy": float(entropy.detach()), "total": float(loss.detach()),
                           "gradient_norm": float(norm)})
    same_tree(update["losses"], losses, "PPO per-minibatch update evidence")
    compare_model_optimizer(model, optimizer, after, "PPO final")
    same_tree(after["rng_state"], before["rng_state"], "PPO global RNG must not advance")
    return {"transitions": n, "optimizer_updates": len(losses), "exact_optimizer_replay": True,
            "generator_state_after": sampler.get_state() if sampler is not None else None}


def inside(run, value):
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (run / path).resolve()
    require(path.is_relative_to(run), "Learning source escapes this experiment")
    return path


def transition_refs(run, cells):
    transitions, refs = [], []
    for cell in cells:
        path = inside(run, cell)
        require((path / "completion_manifest.json").is_file(), "Learning used an incomplete episode")
        rows = lines(path / "transitions.jsonl.gz")
        require(not rows or rows[-1]["done"] is True, "Learning used an unfinished trajectory")
        require(all(row["transition_id"] == i for i, row in enumerate(rows)), "Source transition IDs differ")
        transitions.extend(rows)
        refs.extend({"cell": str(path.relative_to(run)), "transition_id": row["transition_id"]} for row in rows)
    return transitions, refs



def iter_transition_refs(run, cells):
    """Read one original JSON row at a time; retain all source files unchanged."""
    for cell in cells:
        path = inside(run, cell)
        require((path / "completion_manifest.json").is_file(), "Learning used an incomplete episode")
        count, last_done = 0, None
        with gzip.open(path / "transitions.jsonl.gz", "rt", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                require(row["transition_id"] == count, "Source transition IDs differ")
                last_done = row["done"]
                yield row, {"cell": str(path.relative_to(run)), "transition_id": row["transition_id"]}
                count += 1
        require(count == 0 or last_done is True, "Learning used an unfinished trajectory")


def bind_one_observation(data, index, row, representation, label):
    """Same float32 conversion and exact check, with one-row temporary storage."""
    require(index < len(data["actions"]), label + " contains extra source rows")
    x, g, mask = tensorize([row["obs_" + representation]], representation)
    for name, expected in (("candidates", x), ("global", g), ("mask", mask)):
        same_tree(data[name][index:index+1], expected, label + " " + name)
    same_tree(data["actions"][index:index+1], torch.tensor([row["action"]], dtype=torch.long), label + " actions")


def bind_rollout_sources(data, run, cells, representation):
    n = validate_dataset(data, representation, rollout=True) if data is not None else 0
    refs, count = [], 0
    for row, ref in iter_transition_refs(run, cells):
        require(data is not None, "Nonempty source episode has no saved rollout")
        bind_one_observation(data, count, row, representation, "Saved rollout/trajectory")
        for field, source, dtype in (("old_logprob", "logprob", torch.float32),
                                     ("values", "value", torch.float32),
                                     ("rewards", "reward", torch.float32),
                                     ("dones", "done", torch.bool)):
            same_tree(data[field][count:count+1], torch.tensor([row[source]], dtype=dtype), "Saved rollout/trajectory " + field)
        refs.append(ref)
        count += 1
    require(count == n, "Rollout changed retained sample population")
    if data is not None:
        same_tree(data["source_refs"], refs, "Saved PPO tensors changed sample provenance")
    return count, refs


def bind_supervised_data(data, population):
    n = validate_dataset(data, "associated")
    require(set(data) == {"candidates", "global", "mask", "actions", "feasible", "costs", "source_refs"},
            "Supervised reliable queried population keys")
    refs, count = [], 0
    for row, ref in iter_transition_refs(population["run"], population["population"]["teacher_cells"]):
        info = row["info"]
        if info["query_reason"] != "query" or info["validation_status"] not in ("COMPLETE", "FAIL"):
            continue
        bind_one_observation(data, count, row, "associated", "Supervised reliable queried population")
        same_tree(data["feasible"][count:count+1], torch.tensor([info["validation_status"] == "COMPLETE"], dtype=torch.float32),
                  "Supervised reliable queried population feasible")
        same_tree(data["costs"][count:count+1], torch.tensor([info["physical_points_used"] / 256.], dtype=torch.float32),
                  "Supervised reliable queried population costs")
        refs.append(ref)
        count += 1
    require(count == n == population["supervised_count"], "Supervised reliable population count differs")
    same_tree(data["source_refs"], refs, "Supervised reliable queried population source_refs")
    same_tree(refs, population["supervised_refs"], "Supervised population order differs")


def teacher_population(run):
    population = read(run / "learning/teacher_population.json")
    schedule = read(run / "schedule.json")
    schedule = schedule["episodes"] if isinstance(schedule, dict) and "episodes" in schedule else schedule
    require(isinstance(schedule, list), "Unknown frozen schedule schema")
    expected_cells = {str(Path(t["cell"])) for t in schedule if t["phase"] in ("teacher", "reference")}
    cells = population["teacher_cells"]
    require(len(cells) == len(set(cells)) and {str(Path(p)) for p in cells} == expected_cells,
            "Teacher dataset omitted or added a frozen source episode")
    train, evaluation = read(run / "training_contexts.json"), read(run / "evaluation_contexts.json")
    specs = [read(inside(run, p) / "spec.json") for p in cells]
    counts = check_pairing_population(train, evaluation, specs)
    regular = Counter((s["scenario_id"], s["event_id"], s["method"]) for s in specs if s["phase"] == "teacher")
    require(regular == Counter((c["scenario_id"], c["event_id"], method)
                              for c in train for method in ("greedy", "beam")), "Frozen two-rule teacher population differs")
    require(all(s["point_budget"] == (1024 if s["phase"] == "reference" else 256) for s in specs), "Teacher budget differs")
    candidates = {}
    for cell in cells:
        path = inside(run, cell)
        result, spec = read(path / "result.json"), read(path / "spec.json")
        require(result["spec"] == spec, "Teacher result/spec identity differs")
        best = result.get("best_plan")
        quality = float(best["validation"]["mean_served_fraction"]) if best else 0.
        key = (spec["scenario_id"], spec["event_id"])
        priority = (-quality, result["oracle"]["points"], path.name)
        if key not in candidates or priority < candidates[key][0]:
            candidates[key] = (priority, path)
    selected_cells = [str(candidates[key][1].relative_to(run)) for key in sorted(candidates)]
    require(population["selected_cells"] == selected_cells, "Teacher selection differs from fixed quality/cost/cell tie rule")
    require(population["all_failures_retained_in_source_cells"] is True, "Failures not retained")
    refs, returns = [], []
    for cell in selected_cells:
        # Only scalar reward/terminal pairs are needed for the original reverse recurrence.
        scalar_rows = []
        for row, ref in iter_transition_refs(run, [cell]):
            refs.append(ref)
            scalar_rows.append((row["reward"], row["done"]))
        tail, targets = 0., []
        for reward, done in reversed(scalar_rows):
            tail = reward + (0. if done else tail)
            targets.append(tail)
        returns.extend(reversed(targets))
    require(population["selected_transitions"] == refs, "Teacher selected sample population differs")
    supervised_refs = []
    for row, ref in iter_transition_refs(run, cells):
        info = row["info"]
        if info["query_reason"] == "query" and info["validation_status"] in ("COMPLETE", "FAIL"):
            supervised_refs.append(ref)
    require(refs and supervised_refs, "No reliable teacher decisions; audit cannot fabricate samples")
    return {"run": run, "population": population, "refs": refs, "selected_count": len(refs),
            "returns": torch.tensor(returns, dtype=torch.float32),
            "supervised_count": len(supervised_refs), "supervised_refs": supervised_refs, "counts": counts}


def check_initialization(saved, seed, representation, kind, config, generator_offset):
    require(saved["seed"] == seed and saved["kind"] == kind
            and saved["representation"] == representation, "Initial checkpoint identity differs")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = (IndependentSupervised if kind == "supervised" else IndependentActorCritic)(representation == "associated")
        optimizer = adam(model.parameters(), config)
        compare_model_optimizer(model, optimizer, saved, "Seeded initial")
        same_tree(saved["rng_state"], torch.get_rng_state(), "Initial global RNG")
    expected_sampler = torch.Generator(device="cpu").manual_seed(seed + generator_offset)
    same_tree(saved["training_metadata"]["generator_state"], expected_sampler.get_state(), "Initial sample RNG")
    return expected_sampler


def bind_teacher_data(data, population, representation):
    n = validate_dataset(data, representation)
    refs, count = [], 0
    for row, ref in iter_transition_refs(population["run"], population["population"]["selected_cells"]):
        bind_one_observation(data, count, row, representation, "Teacher data")
        refs.append(ref)
        count += 1
    require(count == n == population["selected_count"], "Teacher data population count differs")
    same_tree(data["returns"], population["returns"], "Teacher data returns")
    same_tree(data["source_refs"], refs, "Teacher data source_refs")
    same_tree(refs, population["refs"], "Teacher population order differs")


def replay_initial_fit(directory, config, data=None):
    """Exact BC and value-head fit replay, with immutable actor attribution."""
    directory = Path(directory)
    initial, bc, critic_initial, critic_final, start = [load(directory / name) for name in
        ("initial.pt", "bc.pt", "critic_initial.pt", "critic_final.pt", "start.pt")]
    data = load(directory / "batches.pt") if data is None else data
    log = read(directory / "fit_log.json")
    require(log["config"] == config and log["checkpoint_sha256"] == sha(directory / "start.pt"), "Fit config/checkpoint changed")
    require(log["bc_updates"] == config["bc_updates"] and log["critic_updates"] == config["critic_updates"], "Fit update counts differ")
    require(log["actor_unchanged_by_critic_fit"] is True and log["source_refs"] == data["source_refs"], "Fit sample metadata differs")
    representation, seed = initial["representation"], initial["seed"]
    n = validate_dataset(data, representation)
    sampler = check_initialization(initial, seed, representation, "actor_critic", config, 10000)
    model, _ = load_independent(initial)
    optimizer = adam(model.parameters(), config)
    optimizer.load_state_dict(deepcopy(initial["optimizer_state"]))
    require(data["bc_indices"].shape == (config["bc_updates"], config["bc_batch_size"])
            and data["critic_indices"].shape == (config["critic_updates"], config["bc_batch_size"]), "Fit minibatch dimensions differ")
    losses = []
    for stage, steps in (("bc", config["bc_updates"]), ("critic", config["critic_updates"])):
        if stage == "critic":
            compare_model_optimizer(model, optimizer, bc, "BC endpoint")
            same_tree(bc["training_metadata"]["generator_state"], sampler.get_state(), "BC sampler endpoint")
            same_tree(critic_initial["model_state"], bc["model_state"], "Critic fit starts at BC endpoint")
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(name.startswith("critic."))
            optimizer = adam(model.critic.parameters(), config)
            compare_model_optimizer(model, optimizer, critic_initial, "Critic initial")
            same_tree(critic_initial["training_metadata"]["generator_state"], sampler.get_state(), "Critic initial sampler")
        for step in range(steps):
            index = torch.randint(n, (config["bc_batch_size"],), generator=sampler)
            same_tree(data[stage + "_indices"][step], index, "Exact seeded " + stage + " sample indices")
            logits, value = model(data["candidates"][index], data["global"][index], data["mask"][index])
            loss = (nn.functional.cross_entropy(logits, data["actions"][index]) if stage == "bc"
                    else (value - data["returns"][index]).square().mean())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters() if stage == "bc" else model.critic.parameters(), config["max_grad_norm"])
            require(bool(torch.isfinite(loss)) and bool(torch.isfinite(grad)), "Nonfinite initial-fit replay")
            optimizer.step()
            losses.append({"stage": stage, "update": step, "loss": float(loss.detach()), "gradient_norm": float(grad)})
    compare_model_optimizer(model, optimizer, critic_final, "Critic endpoint")
    for key in bc["model_state"]:
        if not key.startswith("critic."):
            same_tree(critic_final["model_state"][key], bc["model_state"][key], "Critic fit altered actor/trunk " + key)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    compare_model_optimizer(model, adam(model.parameters(), config), start, "Common BC endpoint/PPO initial state")
    for saved in (bc, critic_initial, critic_final, start):
        require(saved["seed"] == seed and saved["representation"] == representation and saved["kind"] == "actor_critic",
                "Initial-fit checkpoint lineage differs")
        same_tree(saved["architecture"], initial["architecture"], "Initial-fit checkpoint architecture")
        same_tree(saved["rng_state"], initial["rng_state"], "Initial-fit global RNG must not advance")
    for saved in (critic_final, start):
        same_tree(saved["training_metadata"]["generator_state"], sampler.get_state(), "Initial-fit sampler endpoint")
    same_tree(log["losses"], losses, "Initial-fit per-update evidence")
    return {"seed": seed, "representation": representation, "transitions": n,
            "bc_updates": config["bc_updates"], "critic_updates": config["critic_updates"],
            "actor_unchanged_by_critic_fit": True, "exact_optimizer_replay": True}


def replay_supervised(directory, data, config):
    directory = Path(directory)
    initial, final = load(directory / "initial.pt"), load(directory / "final.pt")
    batches, log = load(directory / "batches.pt"), read(directory / "fit_log.json")
    require(log["config"] == config and log["source_refs"] == data["source_refs"], "Supervised fit config/population differs")
    n = validate_dataset(data, "associated")
    require(data["feasible"].shape == data["costs"].shape == (n,), "Supervised target dimensions differ")
    sampler = check_initialization(initial, initial["seed"], "associated", "supervised", config, 20000)
    model, _ = load_independent(initial)
    optimizer = adam(model.parameters(), config)
    optimizer.load_state_dict(deepcopy(initial["optimizer_state"]))
    require(batches["indices"].shape == (config["supervised_updates"], config["bc_batch_size"]), "Supervised minibatch dimensions differ")
    losses = []
    for step in range(config["supervised_updates"]):
        index = torch.randint(n, (config["bc_batch_size"],), generator=sampler)
        same_tree(batches["indices"][step], index, "Exact supervised seeded sample indices")
        logits, cost = model(data["candidates"][index], data["global"][index], data["mask"][index])
        rows, actions = torch.arange(len(index)), data["actions"][index]
        feasible_loss = nn.functional.binary_cross_entropy_with_logits(logits[rows, actions], data["feasible"][index])
        cost_loss = (cost[rows, actions] - data["costs"][index]).square().mean()
        loss = feasible_loss + cost_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
        require(bool(torch.isfinite(loss)) and bool(torch.isfinite(grad)), "Nonfinite supervised replay")
        optimizer.step()
        losses.append({"update": step, "feasible": float(feasible_loss.detach()), "cost": float(cost_loss.detach()),
                       "loss": float(loss.detach()), "gradient_norm": float(grad)})
    same_tree(log["losses"], losses, "Supervised per-update evidence")
    compare_model_optimizer(model, optimizer, final, "Supervised endpoint")
    for key in ("kind", "seed", "representation", "architecture"):
        same_tree(final[key], initial[key], "Supervised checkpoint lineage " + key)
    same_tree(final["rng_state"], initial["rng_state"], "Supervised global RNG must not advance")
    same_tree(final["training_metadata"]["generator_state"], sampler.get_state(), "Supervised sample RNG endpoint")
    return {"seed": initial["seed"], "transitions": n, "optimizer_updates": len(losses), "exact_optimizer_replay": True}


def audit_initial(run):
    run = Path(run).resolve()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = read(run / "config.json")["learning"]
    population = teacher_population(run)
    supervised = load(run / "learning/supervised_data.pt")
    bind_supervised_data(supervised, population)
    fits, scorer_fits, expected_map = [], [], {}
    for seed in config["seeds"]:
        expected_map[str(seed)] = {}
        for representation in ("raw", "associated"):
            directory = run / "learning" / f"seed{seed}" / representation
            identity = load(directory / "initial.pt")
            require(identity["seed"] == seed and identity["representation"] == representation,
                    "Fit directory does not match checkpoint seed/representation")
            data = load(directory / "batches.pt")
            bind_teacher_data(data, population, representation)
            fits.append(replay_initial_fit(directory, config, data=data))
            del data
            expected_map[str(seed)]["bc_" + representation] = str((directory / "start.pt").relative_to(run))
        directory = run / "learning" / f"seed{seed}" / "supervised"
        require(load(directory / "initial.pt")["seed"] == seed, "Supervised seed directory differs")
        scorer_fits.append(replay_supervised(directory, supervised, config))
        expected_map[str(seed)]["supervised"] = str((directory / "final.pt").relative_to(run))
    require(read(run / "learning/initial_checkpoints.json") == expected_map, "Initial endpoint mapping differs")
    return {"status": "VERIFIED_CLPU_BRANCH_INITIAL_LEARNING", "errors": [],
            **population["counts"], "selected_teacher_cells": len(population["population"]["selected_cells"]),
            "selected_teacher_transitions": population["selected_count"], "supervised_transitions": population["supervised_count"],
            "fits": fits, "supervised_fits": scorer_fits,
            "teacher_population_sha256": sha(run / "learning/teacher_population.json"),
            "new_powerflow_calls": 0, "new_training_interactions": 0,
            "optimizer_steps_replayed_for_audit": sum(r["bc_updates"] + r["critic_updates"] for r in fits)
                + sum(r["optimizer_updates"] for r in scorer_fits)}



def ordered_update_cells(run, representation, seed, round_index, previous, schedule):
    """Bind the actual immutable task order without sorting completed workers."""
    group = f"train_{representation}_{seed}_{round_index:02d}"
    task_dir = run / "learning_tasks" / (group + "_update")
    task_path, started_path = task_dir / "task.json", task_dir / "started.json"
    task, started = read(task_path), read(started_path)
    require(started["task_sha256"] == sha(task_path), "Started PPO task hash differs")
    require(task["kind"] == "ppo_update" and Path(task["run"]).resolve() == run
            and task["representation"] == representation
            and type(task["seed"]) is int and task["seed"] == seed
            and type(task["round_index"]) is int and task["round_index"] == round_index
            and inside(run, task["checkpoint"]) == previous, "PPO task scope/checkpoint differs")
    expected = [inside(run, t["cell"]) for t in schedule
                if t["phase"] == "train" and t["representation"] == representation
                and t["seed"] == seed and t["round_index"] == round_index]
    require(len(expected) == len(set(expected)) == 8, "PPO round does not contain all eight frozen episodes")
    require(isinstance(task["cells"], list), "PPO task cells must be an ordered list")
    actual = [inside(run, p) for p in task["cells"]]
    require(len(actual) == len(set(actual)) == 8 and set(actual) == set(expected),
            "PPO task omitted/duplicated/added a frozen episode")
    audit_dir = run / "audit_batches" / group
    prior_order = [inside(run, p) for p in read(audit_dir / "cells.json")]
    require(actual == prior_order, "PPO task order differs from prior episode audit order")
    verification = read(audit_dir / "verification.json")
    require(verification.get("status") == "VERIFIED_CLPU_BRANCH_CELLS" and verification.get("errors") == [],
            "PPO source episodes have not passed their prior independent audit")
    return [str(p.relative_to(run)) for p in actual]


def audit_update(run, representation, seed, round_index):
    run = Path(run).resolve()
    config = read(run / "config.json")["learning"]
    require(representation in ("raw", "associated") and seed in config["seeds"] and 0 <= round_index < 8,
            "Update outside frozen representation/seed/round scope")
    base = run / "learning" / f"seed{seed}" / representation
    directory = base / f"ppo_round{round_index:02d}"
    previous = base / ("start.pt" if round_index == 0 else f"ppo_round{round_index-1:02d}/after.pt")
    before, after = load(directory / "before.pt"), load(directory / "after.pt")
    source = load(previous)
    require(inside(run, before["training_metadata"]["source_checkpoint"]) == previous
            and before["training_metadata"]["source_sha256"] == sha(previous), "PPO common starting endpoint/round lineage differs")
    for name in ("model_state", "optimizer_state", "rng_state", "seed", "representation", "architecture", "kind"):
        same_tree(before[name], source[name], "PPO exact predecessor " + name)
    sampler = torch.Generator(device="cpu").manual_seed(seed + 30000 + round_index)
    same_tree(before["training_metadata"]["generator_state"], sampler.get_state(), "PPO seeded initial minibatch RNG")
    update = read(directory / "update.json")
    require(update["config"] == config and update["round_index"] == round_index, "PPO frozen config/round differs")
    schedule = read(run / "schedule.json")
    schedule = schedule["episodes"] if isinstance(schedule, dict) and "episodes" in schedule else schedule
    cells = ordered_update_cells(run, representation, seed, round_index, previous, schedule)
    allowed = {(t["scenario_id"], t["event_id"]) for t in read(run / "training_contexts.json")}
    for cell in cells:
        spec = read(inside(run, cell) / "spec.json")
        require((spec["scenario_id"], spec["event_id"]) in allowed and spec["phase"] == "train"
                and spec["checkpoint_sha256"] == sha(previous) and spec["deterministic"] is False,
                "PPO used evaluation or mixed-policy episode")
    rollout = load(directory / "rollout.pt") if (directory / "rollout.pt").is_file() else None
    transition_count, refs = bind_rollout_sources(rollout, run, cells, representation)
    require(update["source_refs"] == refs, "PPO sample population differs from all scheduled episodes")
    if not transition_count:
        require(update["status"] == "NO_DECISIONS_NO_OPTIMIZER_UPDATE" and update["transition_count"] == 0,
                "Empty rollout reported as a learned update")
        for name in ("model_state", "optimizer_state", "rng_state", "seed", "representation", "architecture", "kind"):
            same_tree(after[name], before[name], "Empty update changed " + name)
        same_tree(after["training_metadata"]["generator_state"], sampler.get_state(), "Empty update RNG changed")
        require(not (directory / "rollout.pt").exists() and not (directory / "targets.pt").exists(), "Empty update fabricated training tensors")
        result = {"transitions": 0, "optimizer_updates": 0, "exact_optimizer_replay": True}
    else:
        require(update["status"] == "PPO_UPDATE_COMPLETE", "PPO update is not completed")
        targets = load(directory / "targets.pt")
        require(set(targets) == {"advantages", "returns"}, "PPO target schema differs")
        result = replay_ppo_update(before, after, rollout, {**update, **targets}, config,
                                   generator_state=sampler.get_state())
        same_tree(after["training_metadata"]["generator_state"], result.pop("generator_state_after"), "PPO minibatch sampler endpoint")
    return {"status": "VERIFIED_CLPU_BRANCH_PPO_UPDATE", "errors": [], "seed": seed,
            "representation": representation, "round_index": round_index, **result,
            "before_sha256": sha(directory / "before.pt"), "after_sha256": sha(directory / "after.pt"),
            "new_powerflow_calls": 0, "new_training_interactions": 0}


def audit_learning(run, cfg=None, require_complete=True):
    run = Path(run).resolve()
    actual = read(run / "config.json")
    require(cfg is None or cfg == actual, "Audit config differs from run")
    initial = audit_initial(run)
    updates = []
    for representation in ("raw", "associated"):
        for seed in actual["learning"]["seeds"]:
            for index in range(8):
                path = run / "learning" / f"seed{seed}" / representation / f"ppo_round{index:02d}/after.pt"
                if path.is_file():
                    updates.append(audit_update(run, representation, seed, index))
                else:
                    require(not require_complete, "Missing frozen PPO update")
    attribution = audit_final_attribution(run) if require_complete else None
    return {"status": "VERIFIED_CLPU_BRANCH_LEARNING", "errors": [], "initial": initial, "updates": updates,
            "attribution": attribution, "new_powerflow_calls": 0, "new_training_interactions": 0}


def audit_final_attribution(run):
    """Cheap final endpoint-to-method binding after incremental update audits."""
    run = Path(run).resolve()
    config = read(run / "config.json")["learning"]
    endpoints = read(run / "final_checkpoints.json")
    require(set(endpoints) == {str(s) for s in config["seeds"]}, "Final checkpoint seed population differs")
    expected = {}
    for seed in config["seeds"]:
        models = endpoints[str(seed)]
        require(set(models) == {"supervised", "bc_raw", "bc_associated", "ppo_raw", "ppo_associated"},
                "Incomplete 2×2 endpoint comparison")
        expected[(seed, "supervised")] = run / "learning" / f"seed{seed}" / "supervised/final.pt"
        for representation in ("raw", "associated"):
            base = run / "learning" / f"seed{seed}" / representation
            expected[(seed, "bc_" + representation)] = base / "start.pt"
            expected[(seed, "ppo_" + representation)] = base / "ppo_round07/after.pt"
        for method in models:
            require(inside(run, models[method]) == expected[(seed, method)],
                    "2×2 endpoint mapping breaks common-BC-start attribution")
    schedule = read(run / "schedule.json")
    schedule = schedule["episodes"] if isinstance(schedule, dict) and "episodes" in schedule else schedule
    count = 0
    for template in schedule:
        if template["phase"] not in ("eval", "evaluation") or template["method"] in ("greedy", "beam"):
            continue
        path = inside(run, template["cell"])
        require((path / "completion_manifest.json").is_file(), "Missing evaluation endpoint evidence")
        spec = read(path / "spec.json")
        method, seed = template["method"], template["seed"]
        require(spec["method"] == method and spec["seed"] == seed and spec["deterministic"] is True,
                "Evaluation identity differs from frozen method/seed")
        endpoint = expected[(seed, method)]
        require(inside(run, spec["checkpoint"]) == endpoint and spec["checkpoint_sha256"] == sha(endpoint),
                "Evaluation method used the wrong BC/PPO/supervised endpoint")
        count += 1
    contexts = read(run / "evaluation_contexts.json")
    require(count == len(contexts) * len(config["seeds"]) * 5, "Evaluation learned-method population differs")
    return {"status": "VERIFIED_CLPU_BRANCH_LEARNING_ATTRIBUTION", "errors": [], "evaluation_cells": count,
            "same_seed_bc_ppo_start": True, "same_teacher_population_for_raw_and_associated": True,
            "cross_representation_weights_claimed_identical": False,
            "new_powerflow_calls": 0, "new_training_interactions": 0}


audit_initial_learning = audit_initial
audit_ppo_learning = audit_update
