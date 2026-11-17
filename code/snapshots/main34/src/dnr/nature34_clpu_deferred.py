"""Small graph-expansion / complete-verification / handoff research prototype.

Construction retains the original paid initial and full-WAIT certification.
Graph proposals are not certificates. No learner, reward, new physical rule,
or complete-guard execution is implemented by this module.
"""
from __future__ import annotations

from copy import deepcopy
import time

from .nature34_clpu_branch import BranchSearch
from .nature34_clpu_environment import CLPUReliabilityError
from .nature34_clpu_search import generic_neighbors

VERSION = 'clpu-deferred-complete-verification-prototype-v1'


class DeferredCertificationSearch(BranchSearch):
    """Stable branch IDs for probes; no frozen 32-slot actor compatibility.

``expand`` counts graph expansions; ``verify`` counts complete-plan attempts;
``yield_to_beam`` ends this EARLY search without claiming a Beam guarantee.
These are separate operations, not equivalents of the old combined step.
"""
    def __init__(self, env, oracle, config=None, record_callback=None):
        self.expanded_nodes = set()
        self.verification_queue = {}
        self.skipped_dominated_plans = set()
        self.failure_repairs = set()
        self.total_decisions = self.graph_expansions = self.verify_attempts = 0
        self.deferred_trace = []
        self.handed_off = False
        before_actual = self._actual(oracle)
        before_logical = getattr(oracle, 'points', getattr(oracle, 'used', None))
        began = time.perf_counter()
        super().__init__(env, oracle, config, record_callback)
        self.bootstrap_cost = {'logical_points_before': before_logical,
            'logical_points_after': self._points(), 'actual_physical_points_before': before_actual,
            'actual_physical_points_after': self._actual(oracle),
            'elapsed_seconds': time.perf_counter() - began,
            'includes_initial_snapshot_and_complete_wait_attempt': True}

    @staticmethod
    def _actual(oracle):
        value = getattr(oracle, 'physical_points', None)
        if type(value) is not int or value < 0:
            raise ValueError('Deferred prototype requires an explicit actual physical point counter')
        return value

    def _add(self, prefix, actions, origin, parent_id):
        previous = self._next_id
        super()._add(prefix, actions, origin, parent_id)
        if self._next_id != previous:
            node = self._frontier[previous]
            plan = node['info']['actions']
            if plan not in self._checked and plan not in self.skipped_dominated_plans:
                self.verification_queue.setdefault(plan, previous)

    def _refresh_stop(self):
        if self.done:
            return
        if self.incumbent is not None and self._frontier:
            energy = self.incumbent['validation']['energy_kwh']
            if all(n['upper_bound_kwh'] <= energy for n in self._frontier.values()):
                self._stop('CERTIFIED_FRONTIER_UPPER_BOUND')
                return
        if self._points() >= self.config.point_budget:
            self._stop('PHYSICAL_BUDGET_EXHAUSTED')  # Original logical-counter stop name.
        elif len(self._checked) >= self.config.max_checked_candidates:
            self._stop('LOGICAL_CHECK_CAP')
        elif not self.verification_queue and (self.graph_expansions >= self.config.max_branch_expansions
                or all(i in self.expanded_nodes for i in self._frontier)):
            self._stop('DEFERRED_FRONTIER_EXHAUSTED')

    def pending(self):
        rows = []
        for branch_id, node in sorted(self._frontier.items()):
            plan = node['info']['actions']
            status = ('SKIPPED_DOMINATED' if plan in self.skipped_dominated_plans
                      else self._checked.get(plan, {}).get('status'))
            rows.append({'branch_id': branch_id, 'parent_id': node['parent_id'],
                'prefix': list(node['prefix']), 'actions': list(plan), 'origin': node['origin'],
                'graph_energy_kwh': node['info']['energy_kwh'],
                'graph_expanded': branch_id in self.expanded_nodes,
                'verification_status': status, 'verified': status in ('COMPLETE', 'FAIL'),
                'verification_pending': plan in self.verification_queue,
                'can_expand': not self.done and branch_id not in self.expanded_nodes
                    and self.graph_expansions < self.config.max_branch_expansions,
                'can_verify': not self.done and plan in self.verification_queue})
        return rows

    def _node(self, branch_id):
        if self.done:
            raise RuntimeError('No deferred operation after stop or handoff')
        if type(branch_id) is not int or branch_id not in self._frontier:
            raise ValueError('Unknown stable branch ID')
        return self._frontier[branch_id]

    def _holds_and_neighbors(self, node, failed_step):
        plan = node['info']['actions']
        for size in range(1, min(self.config.lookahead_depth, self.horizon) + 1):
            prefix = plan[:size]
            self._add(prefix, prefix + (0,) * (self.horizon - size), 'prefix_wait', node['id'])
        for neighbor, kind, _ in generic_neighbors(self.env, self.start, plan, self.local_config,
                failed_step=failed_step, assess=self._assess):
            prefix = neighbor['actions'][:min(self.config.lookahead_depth, self.horizon)]
            self._add(prefix, neighbor['actions'], kind, node['id'])

    def _before(self):
        return {'logical': self._points(), 'actual': self._actual(self.oracle), 'quality': self.quality,
                'generated': len(self._branch_keys), 'clock': time.perf_counter()}

    def _record(self, operation, node, before, *, status=None, error=None):
        event = {'decision_id': self.total_decisions - 1, 'operation': operation,
            'branch_id': node['id'] if node is not None else None,
            'actions': list(node['info']['actions']) if node is not None else None,
            'verification_status': status, 'quality_before': before['quality'], 'quality_after': self.quality,
            'logical_points_before': before['logical'], 'logical_points_after': self._points(),
            'actual_physical_points_before': before['actual'],
            'actual_physical_points_after': self._actual(self.oracle),
            'new_actual_physical_points': self._actual(self.oracle) - before['actual'],
            'new_branches': len(self._branch_keys) - before['generated'],
            'graph_expansions': self.graph_expansions, 'verify_attempts': self.verify_attempts,
            'total_decisions': self.total_decisions, 'pending_complete_plans': len(self.verification_queue),
            'elapsed_seconds': time.perf_counter() - before['clock'],
            'done': self.done, 'stop_reason': self.stop_reason, 'error': error}
        self.deferred_trace.append(event)
        return deepcopy(event)

    def expand(self, branch_id):
        node = self._node(branch_id)
        if branch_id in self.expanded_nodes or self.graph_expansions >= self.config.max_branch_expansions:
            raise ValueError('Graph expansion already consumed or expansion cap reached')
        before = self._before()
        self.total_decisions += 1
        self.graph_expansions += 1
        self._beam_turn = self.graph_expansions
        try:
            validation = self._checked.get(node['info']['actions'], {}).get('validation', {})
            failed = validation.get('failed_step') if validation.get('status') == 'FAIL' else None
            # A known failed prefix is not extended unchanged. Legal generic
            # edits of earlier actions remain available, as in the old engine.
            if not (type(failed) is int and failed < len(node['prefix'])):
                self._expand_prefix(node['prefix'], node['id'])
            self._holds_and_neighbors(node, failed_step=None)
            self.expanded_nodes.add(branch_id)
            if self._actual(self.oracle) != before['actual'] or self._points() != before['logical']:
                raise CLPUReliabilityError('Graph expansion unexpectedly invoked physical certification')
            self._refresh_stop()
            return self._record('EXPAND_GRAPH', node, before)
        except BaseException as exc:
            self._stop('DEFERRED_RELIABILITY_OR_IMPLEMENTATION_ERROR')
            self._record('EXPAND_GRAPH', node, before, error=repr(exc))
            raise

    def verify(self, branch_id):
        node = self._node(branch_id)
        plan = node['info']['actions']
        if plan not in self.verification_queue:
            raise ValueError('Complete plan was already checked; duplicate verification forbidden')
        before = self._before()
        self.total_decisions += 1
        self.verify_attempts += 1
        try:
            # Preserve the common complete-objective skip. This says nothing
            # about electrical feasibility or usefulness as an expansion parent.
            if self._incumbent_key is not None and node['info']['priority'] >= self._incumbent_key:
                self.verification_queue.pop(plan)
                self.skipped_dominated_plans.add(plan)
                self._refresh_stop()
                return self._record('VERIFY_COMPLETE_PLAN', node, before, status='SKIPPED_DOMINATED')
            record = self._evaluate(node['info'], node['origin'],
                {'branch_id': branch_id, 'parent_id': node['parent_id'],
                 'prefix': list(node['prefix']), 'operation': 'VERIFY_COMPLETE_PLAN'})
            status = record['status'] if record is not None else None
            if status in ('COMPLETE', 'FAIL'):
                self.verification_queue.pop(plan)
            if status == 'FAIL' and plan not in self.failure_repairs:
                failed = record['validation']['failed_step']
                if type(failed) is not int:
                    raise CLPUReliabilityError('Reliable complete-plan failure needs its recorded failed step')
                self._holds_and_neighbors(node, failed_step=failed)
                self.failure_repairs.add(plan)
            # PARTIAL/UNKNOWN is neither reliable FAIL nor a repair certificate.
            self._refresh_stop()
            return self._record('VERIFY_COMPLETE_PLAN', node, before, status=status)
        except BaseException as exc:
            self._stop('DEFERRED_RELIABILITY_OR_IMPLEMENTATION_ERROR')
            self._record('VERIFY_COMPLETE_PLAN', node, before, error=repr(exc))
            raise

    def yield_to_beam(self):
        if self.done:
            raise RuntimeError('No deferred operation after stop or handoff')
        before = self._before()
        self.total_decisions += 1
        self.handed_off = True
        self._stop('YIELDED_TO_BEAM')
        return self._record('YIELD_TO_BEAM', None, before)

    def step(self, *args, **kwargs):
        raise TypeError('Use expand, verify or yield_to_beam; the old combined action is not this interface')

    def observation(self, *args, **kwargs):
        raise TypeError('No frozen actor observation for three operations; pending() is a probe inventory')

    def result(self):
        value = super().result()
        value.update(version=VERSION, method='deferred_complete_verification_prototype',
            logical_points=self._points(), actual_physical_points=self._actual(self.oracle),
            graph_expansions=self.graph_expansions, verify_attempts=self.verify_attempts,
            skipped_dominated_complete_plans=len(self.skipped_dominated_plans),
            total_decisions=self.total_decisions, deferred_trace=deepcopy(self.deferred_trace),
            bootstrap_cost=deepcopy(self.bootstrap_cost), handed_off=self.handed_off,
            guard_established=False, complete_guard_still_required=True,
            pending_complete_plans=len(self.verification_queue), new_training=0,
            scope='Prototype EARLY lane only; no new exposure policy, reward, learner or complete guard execution')
        return value
