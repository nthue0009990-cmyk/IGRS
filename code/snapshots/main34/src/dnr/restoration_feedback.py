"""Pure state/reward layer for bounded, virtual candidate-improvement episodes.

No simulator or optimizer is imported here. Physical evaluation and its durable
cost journal remain external. A receipt is accepted only after all five paid
alpha points pass the existing numerical/AC/isolation reliability checks.
Known voltage/thermal failures are ordinary observations, not dropped samples.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


ALPHAS = (0., .25, .5, .75, 1.)
SECTIONS = ("numerical", "topology", "voltage", "line_thermal",
            "transformer_thermal", "gfm_active_limit")


def finite(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Expected a finite numeric observation")
    return float(value)


def binary_action(action: Sequence[int]) -> tuple[int, ...]:
    if len(action) != 19 or any(type(bit) is not int or bit not in (0, 1) for bit in action):
        raise ValueError("Expected the recorded 9-switch/10-load binary action")
    return tuple(action)


def unit_fraction(value: float) -> float:
    """Canonicalize at most 1e-12 roundoff at the dimensionless [0,1] endpoints.

    Original kW ledgers and physical limits remain unchanged and are retained.
    Larger discrepancies are errors, not clipped research outcomes.
    """
    value = finite(value)
    if value < -1e-12 or value > 1. + 1e-12:
        raise ValueError("Restoration fraction outside its numerical accounting range")
    return min(1., max(0., value))


def masked_extreme(values: Sequence[float], *, largest: bool) -> tuple[float, int]:
    """A missing measurement is encoded as zero WITH a separate validity bit."""
    clean = [finite(value) for value in values]
    return ((max(clean) if largest else min(clean)), 1) if clean else (0., 0)


@dataclass(frozen=True)
class AlphaFeedback:
    alpha: float
    known_witness: bool
    # min/max energized node voltage; max phase-line I/limit; max winding S/limit.
    metrics: tuple[float, float, float, float]
    valid: tuple[int, int, int, int]

    def vector(self) -> tuple[float, ...]:
        return (*self.metrics, *self.valid, float(self.known_witness))


@dataclass(frozen=True)
class CandidateEvidence:
    scenario_id: str
    action: tuple[int, ...]
    denominator_kw: float
    requested_kw: float
    feedback: tuple[AlphaFeedback, ...]

    @property
    def witness_found(self) -> bool:
        return any(item.known_witness for item in self.feedback)

    @property
    def restored_kw(self) -> float | None:
        # None and a physically verified zero-restoration witness are distinct.
        return self.requested_kw if self.witness_found else None

    @classmethod
    def from_peak_points(cls, points: Sequence[dict]) -> "CandidateEvidence":
        """Read five original-runtime peak points; does not recertify physics.

        Accept only independently audited source runs in the calling workflow.
        The operating-adapter schema is intentionally a separate future adapter,
        so the old nominal ledger cannot silently stand for scaled demand.
        """
        if len(points) != len(ALPHAS):
            raise ValueError("A complete paid five-alpha candidate is required")
        rows = sorted(points, key=lambda row: finite(row["alpha"]))
        if tuple(finite(row["alpha"]) for row in rows) != ALPHAS:
            raise ValueError("Alpha grid incomplete or duplicated")
        first = rows[0]
        sid = first["scenario_id"]
        if not isinstance(sid, str) or not sid:
            raise ValueError("Missing scenario identity")
        action = binary_action(first["action"])
        denominator = finite(first["nominal_load_ledger"]["denominator_nominal_kw"])
        requested = finite(first["nominal_load_ledger"]["energized_nominal_kw"])
        if denominator <= 0 or requested < 0:
            raise ValueError("Invalid demand ledger")
        unit_fraction(requested / denominator)
        feedback = []
        for row in rows:
            if row["scenario_id"] != sid or binary_action(row["action"]) != action:
                raise ValueError("Mixed scenario/action receipt")
            if row["partition"] != "build" or row["algorithm"] != 0:
                raise ValueError("This software probe is build/Normal only")
            if "load_readback" in row or "operating_condition" in row:
                raise ValueError("Use a separately validated operating-demand adapter")
            ledger = row["nominal_load_ledger"]
            if (finite(ledger["denominator_nominal_kw"]) != denominator
                    or finite(ledger["energized_nominal_kw"]) != requested
                    or not math.isclose(finite(ledger["nominal_restored_fraction"]),
                                        requested / denominator, rel_tol=0., abs_tol=1e-12)):
                raise ValueError("Inconsistent demand across alpha points")
            raw, known = row["raw_outputs"], row["known_constraints"]
            if (raw["converged"] is not True or row["ac_audit"]["status"] != "PASS"
                    or row["isolation"]["status"] != "PASS"):
                raise RuntimeError("Numerical/AC/isolation failure: retain and stop the affected episode")
            statuses = [known[name]["status"] for name in SECTIONS]
            if any(status not in ("PASS", "FAIL") for status in statuses):
                raise ValueError("Unknown required constraint status")
            witness = known["known_constraints_satisfied"]
            if type(witness) is not bool or witness != all(status == "PASS" for status in statuses):
                raise ValueError("Known-constraint result conflicts with its sections")
            names, magnitudes = raw["node_names"], raw["node_vmag_pu"]
            if len(names) != len(magnitudes) or len(set(names)) != len(names):
                raise ValueError("Node observation inventory mismatch")
            excluded = known["voltage"]["excluded_node_states"]
            voltages = [finite(v) for name, v in zip(names, magnitudes) if name not in excluded]
            current_ratios, winding_ratios = [], []
            for check in known["line_thermal"]["checks"]:
                limit = finite(check["limit_a"])
                current = finite(check["current_a"])
                if limit <= 0 or current < 0:
                    raise ValueError("Invalid existing line rating/readback")
                current_ratios.append(current / limit)
            for check in known["transformer_thermal"]["checks"]:
                limit = finite(check["limit_kva"])
                power = finite(check["total_s_kva"])
                if limit <= 0 or power < 0:
                    raise ValueError("Invalid existing winding rating/readback")
                winding_ratios.append(power / limit)
            pairs = (masked_extreme(voltages, largest=False), masked_extreme(voltages, largest=True),
                     masked_extreme(current_ratios, largest=True), masked_extreme(winding_ratios, largest=True))
            feedback.append(AlphaFeedback(finite(row["alpha"]), witness,
                                          tuple(p[0] for p in pairs), tuple(p[1] for p in pairs)))
        return cls(sid, action, denominator, requested, tuple(feedback))


class FeedbackEpisode:
    """Incumbent improvement with every submitted complete candidate charged.

    A common graph starter is submitted using accept() and consumes slot one.
    This class does not itself grant physical-call authorization or count API
    entries. Replayed receipts count logical slots only; new solver cost is zero.
    """
    def __init__(self, scenario_id: str, denominator_kw: float, budget: int = 8):
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ValueError("Missing scenario")
        if type(budget) is not int or budget <= 0:
            raise ValueError("Budget must be a positive integer")
        self.denominator_kw = finite(denominator_kw)
        if self.denominator_kw <= 0:
            raise ValueError("Demand denominator must be positive")
        self.scenario_id, self.budget = scenario_id, budget
        self.used = 0
        self.last: CandidateEvidence | None = None
        self.incumbent: CandidateEvidence | None = None

    @property
    def done(self) -> bool:
        return self.used == self.budget

    @property
    def best_fraction(self) -> float:
        return 0. if self.incumbent is None else unit_fraction(self.incumbent.requested_kw / self.denominator_kw)

    def accept(self, receipt: CandidateEvidence) -> float:
        if self.done:
            raise RuntimeError("Candidate budget exhausted")
        if receipt.scenario_id != self.scenario_id or receipt.denominator_kw != self.denominator_kw:
            raise ValueError("Receipt does not belong to this episode")
        if len(receipt.feedback) != 5 or tuple(item.alpha for item in receipt.feedback) != ALPHAS:
            raise ValueError("Incomplete receipt")
        before = self.best_fraction
        self.used += 1
        self.last = receipt
        if receipt.witness_found and (self.incumbent is None or receipt.requested_kw > self.incumbent.requested_kw):
            self.incumbent = receipt  # strict improvement; ties keep the first witness
        return self.best_fraction - before

    def dynamic_observation(self) -> tuple[float, ...]:
        """88 features; caller prepends frozen faults/current P,Q/DER data.

        Last action, incumbent action, five alpha feedback blocks, last-present,
        incumbent-witness, requested fraction, incumbent fraction, remaining slots.
        Missing initial actions have a presence flag; they are not trial results.
        """
        last_action = (0,) * 19 if self.last is None else self.last.action
        best_action = (0,) * 19 if self.incumbent is None else self.incumbent.action
        feedback = (0.,) * 45 if self.last is None else tuple(
            value for item in self.last.feedback for value in item.vector())
        return (*last_action, *best_action, *feedback, float(self.last is not None),
                float(self.incumbent is not None),
                0. if self.last is None else unit_fraction(self.last.requested_kw / self.denominator_kw),
                self.best_fraction, (self.budget - self.used) / self.budget)


def complete_episode_returns(rewards: Sequence[float], terminal: Sequence[bool]) -> tuple[float, ...]:
    """Undiscounted Monte Carlo returns, resetting at each true episode end.

    gamma=1 makes the sum of incumbent increments equal final recovery gain.
    Do not feed one-step increments into the old contextual-bandit PPO target.
    Rollouts here must contain full episodes; no implicit truncation bootstrap.
    """
    if not rewards or len(rewards) != len(terminal) or terminal[-1] is not True:
        raise ValueError("Only complete, nonempty episodic rollouts are supported")
    if any(type(flag) is not bool for flag in terminal):
        raise ValueError("Terminal flags must be Boolean")
    clean = [finite(value) for value in rewards]
    returns = [0.] * len(clean)
    running = 0.
    for i in range(len(clean) - 1, -1, -1):
        if terminal[i]:
            running = 0.
        running += clean[i]
        returns[i] = running
    return tuple(returns)
