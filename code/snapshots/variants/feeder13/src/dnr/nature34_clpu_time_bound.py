"""Full-horizon optimistic graph bound; no physical runtime or solver imports.

Bellman rewards belong to the AFTER state, as in CLPUEnvironment.validate_plan.
All legal edges of the complete JointGraph are retained, including WAIT. No
prefix, depth-six limit, learned selection, or electrical rejection is applied.
Thus every full-horizon feasible repair from the original initial state is in
this relaxation, even a repair that rewrites the earliest action.

Two integer programs deliberately remain separate: exact nominal decimal load
weights and exact sums of the binary64 per-slot objective terms. The latter is
the relevant relaxation of the existing floating objective. A conservative
floating envelope is rounded upward and then padded by one additional ULP;
nominal or exact-binary closure alone is NOT the strict floating stop gate.
"""
from __future__ import annotations

from fractions import Fraction
import math
import time

import numpy as np


VERSION = 'clpu-full-horizon-graph-bound-v1'
INT64_MAX = int(np.iinfo(np.int64).max)


def upward_float_envelope(value: Fraction) -> tuple[float, float]:
    """Ceiling binary64 and a further ULP for fsum's documented last-bit risk.

    This assumes the supported IEEE binary64 Python arithmetic environment;
    unsupported/nonfinite arithmetic is rejected. No closeness tolerance is
    admitted. The extra ULP intentionally can prevent a nominally closed stop.
    """
    rounded = float(value)
    if not math.isfinite(rounded) or value < 0:
        raise ValueError('A finite nonnegative binary64 envelope is required')
    if Fraction.from_float(rounded) < value:
        rounded = math.nextafter(rounded, math.inf)
    upper = math.nextafter(rounded, math.inf)
    if not math.isfinite(upper):
        raise OverflowError('No finite conservative floating envelope')
    return rounded, upper


class FullHorizonGraphBound:
    """Global bound from JointGraph.initial_state, with no prefix conditioning.

    The graph argument must be the complete original JointGraph. This module
    validates array/edge consistency; graph completeness and continuity are
    inherited from that graph construction, separately source-bound/audited.
    The production caller uses 30 slots; smaller horizons support unit checks.
    """

    def __init__(self, graph, *, horizon=30, load_multiplier, slot_seconds):
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError('Positive integer horizon required')
        if (not math.isfinite(float(load_multiplier)) or load_multiplier <= 0
                or isinstance(slot_seconds, bool) or not isinstance(slot_seconds, int)
                or slot_seconds <= 0):
            raise ValueError('Positive finite multiplier and integer slot required')
        self.graph = graph
        self.horizon = horizon
        self.multiplier = float(load_multiplier)
        self.slot_seconds = slot_seconds
        self.states = graph.states
        self.edges = graph._next
        self.weights = graph._weights
        self.scale = int(graph.weight_scale)
        n = len(self.states)
        if (n == 0 or self.edges.shape != (20, n) or self.weights.shape != (n,)
                or self.scale <= 0 or np.any(self.weights < 0)
                or not np.issubdtype(self.weights.dtype, np.integer)
                or not np.issubdtype(self.edges.dtype, np.integer)
                or np.any(self.states[1:] <= self.states[:-1])
                or np.any(self.edges < -1) or np.any(self.edges >= n)
                or not np.array_equal(self.edges[0], np.arange(n))):
            raise ValueError('Invalid complete command graph arrays or WAIT edges')
        self.start = int(np.searchsorted(self.states, int(graph.initial_state)))
        if self.start >= n or self.states[self.start] != graph.initial_state:
            raise ValueError('Original graph initial state missing')
        for action in range(1, 20):
            valid = self.edges[action] >= 0
            target = self.edges[action, valid]
            if np.any((self.states[valid] ^ self.states[target]) != 1 << (action - 1)):
                raise ValueError('An edge is not its declared single command operation')
            if np.any(self.weights[target] < self.weights[valid]):
                raise ValueError('Graph edge decreases served nominal weight')
        unique, inverse = np.unique(self.weights, return_inverse=True)
        # Preserve the EXACT operation order of service_kw(s)*slot_seconds/3600.
        atoms = [((int(w) / self.scale) * self.multiplier) * self.slot_seconds / 3600
                 for w in unique]
        if any(not math.isfinite(x) or x < 0 for x in atoms):
            raise ValueError('Nonfinite per-slot objective')
        ratios = [x.as_integer_ratio() for x in atoms]
        denominator = max(d for _, d in ratios)
        numerators = [p * (denominator // d) for p, d in ratios]
        divisor = math.gcd(denominator, *numerators)
        self.binary_denominator = denominator // divisor
        numerators = [v // divisor for v in numerators]
        # Bound BEFORE constructing int64 or performing any cumulative add.
        self._check_overflow(int(max(self.weights)))
        self._check_overflow(max(numerators))
        self.binary_rewards = np.asarray(numerators, dtype=np.int64)[inverse]
        self.float_atoms = np.asarray(atoms, dtype=np.float64)[inverse]

    def _check_overflow(self, maximum):
        if self.horizon * int(maximum) > INT64_MAX:
            raise OverflowError('Analytic horizon * maximum reward exceeds int64')

    def _solve_rewards(self, rewards):
        self._check_overflow(int(max(rewards)))
        n = len(self.states)
        values = np.zeros(n, dtype=np.int64)
        # A WAIT-first deterministic tie has no effect on the primary optimum.
        choices = np.empty((self.horizon + 1, n), dtype=np.uint8)
        for remaining in range(1, self.horizon + 1):
            best = rewards + values
            choices[remaining].fill(0)
            for action in range(1, 20):
                target = self.edges[action]
                valid = target >= 0
                safe_target = np.maximum(target, 0)
                candidate = rewards[safe_target] + values[safe_target]
                better = valid & (candidate > best)
                best[better] = candidate[better]
                choices[remaining, better] = action
            values = best
        position = self.start
        path = [int(self.states[position])]
        indices = []
        actions = []
        for remaining in range(self.horizon, 0, -1):
            action = int(choices[remaining, position])
            position = int(self.edges[action, position])
            actions.append(action)
            indices.append(position)
            path.append(int(self.states[position]))
        optimum = int(values[self.start])
        if sum(int(rewards[i]) for i in indices) != optimum:
            raise AssertionError('DP witness does not attain exact optimum')
        return {'optimum_integer': optimum, 'commands': path, 'actions': actions,
                'served_weights': [int(self.weights[i]) for i in indices],
                'witness_fsum_energy_kwh': math.fsum(float(self.float_atoms[i]) for i in indices)}

    def solve(self):
        started = time.perf_counter()
        nominal = self._solve_rewards(self.weights)
        binary = self._solve_rewards(self.binary_rewards)
        exact = Fraction(binary['optimum_integer'], self.binary_denominator)
        ceiling, upper = upward_float_envelope(exact)
        return {'version': VERSION, 'horizon': self.horizon,
                'initial_command': int(self.graph.initial_state),
                'reachable_command_states': len(self.states), 'weight_scale': self.scale,
                'load_multiplier': self.multiplier, 'slot_seconds': self.slot_seconds,
                'nominal': nominal, 'binary': binary,
                'binary_reward_denominator': self.binary_denominator,
                'exact_binary_optimum': {'numerator': exact.numerator, 'denominator': exact.denominator},
                'ceiling_exact_binary_energy_kwh': ceiling,
                'conservative_float_upper_bound_kwh': upper,
                'float_padding_ulps_after_ceiling': 1,
                'new_physical_points': 0, 'new_training': 0,
                'electrical_certificate_claimed': False, 'integrated_into_search': False,
                'computation_seconds': time.perf_counter() - started}

    def assess_actions(self, actions, bound, *, certified_energy=None):
        """Exact full-route comparison; caller supplies external certification.

        A claimed certified energy must equal the existing service/math.fsum
        expression bit for bit. This helper does not produce any AC certificate.
        """
        if len(actions) != self.horizon:
            raise ValueError('Only complete original-horizon plans are comparable')
        position = self.start
        indices = []
        for action in actions:
            if isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < 20:
                raise ValueError('Invalid action')
            position = int(self.edges[action, position])
            if position < 0:
                raise ValueError('Plan contains illegal graph transition')
            indices.append(position)
        nominal = sum(int(self.weights[i]) for i in indices)
        binary = sum(int(self.binary_rewards[i]) for i in indices)
        energy = math.fsum(float(self.float_atoms[i]) for i in indices)
        if certified_energy is not None and float(certified_energy) != energy:
            raise ValueError('Saved certified objective differs from exact original expression')
        if nominal > bound['nominal']['optimum_integer'] or binary > bound['binary']['optimum_integer']:
            raise AssertionError('A legal witness exceeds its DP upper bound')
        return {'nominal_weight_sum': nominal, 'binary_integer_sum': binary,
                'fsum_energy_kwh': energy,
                'nominal_exactly_closed': nominal == bound['nominal']['optimum_integer'],
                'binary_exactly_closed': binary == bound['binary']['optimum_integer'],
                'strict_float_stop_proven': certified_energy is not None
                    and energy >= bound['conservative_float_upper_bound_kwh'],
                'conservative_gap_kwh': bound['conservative_float_upper_bound_kwh'] - energy}
