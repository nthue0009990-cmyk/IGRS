"""LazySP Forward/Reverse on the *fixed* staged-plan prefix tree.

This is a CLPU-domain adaptation of Dellin--Srinivasa Algorithm 1, not their
robot benchmark reproduction. No new physical model, prior, or learner is used.
The caller pays the unchanged StagedCertificationSearch bootstrap and GUARD.
"""
from copy import deepcopy
import math

from .nature34_clpu_early import complete_plan
from .nature34_clpu_environment import CLPUBudgetExhausted, CLPUReliabilityError
from .nature34_clpu_search import graph_plan
from .nature34_clpu_staged_control import StagedCertificationSearch

VERSION = 'clpu-fixed-prefix-lazysp-forward-reverse-v1'


class LazySPCertificationSearch:
    """Consume a fresh paid staged bootstrap; run() returns an EARLY result.

    GUARD is an external certified path, not an EARLY certificate. The 16-plan
    cap counts distinct optimistic paths selected, not individual slot edges.
    The original oracle enforces logical/actual budgets. A budget stop returns
    only already complete certificates; UNKNOWN is an error, never an edge FAIL.
    """
    def __init__(self, bootstrap, *, guard_result, guard_reference,
                 selector='forward', max_selected_plans=16):
        if not isinstance(bootstrap, StagedCertificationSearch):
            raise TypeError('Original paid StagedCertificationSearch bootstrap required')
        if bootstrap.total_decisions or bootstrap.verify_attempts or bootstrap.graph_expansions:
            raise ValueError('LazySP must start immediately after the unchanged bootstrap')
        if (bootstrap.initial_certificate is not None and
                (bootstrap.initial_certificate.get('reliable') is not True or
                 bootstrap.initial_certificate.get('status') != 'PASS')):
            raise CLPUReliabilityError('Bootstrap initial evidence contradicts a valid baseline')
        if selector not in ('forward', 'reverse'):
            raise ValueError('Only the two frozen literature selectors are supported')
        if type(max_selected_plans) is not int or not 1 <= max_selected_plans <= 16:
            raise ValueError('Distinct selected-plan bound must lie in 1..16')
        if guard_result.get('done') is not True:
            raise CLPUReliabilityError('A completed GUARD is required')
        guard = complete_plan(guard_result)
        if guard is None or guard_result.get('stop_reason') in (
                'INITIAL_FAIL', 'DEFERRED_RELIABILITY_OR_IMPLEMENTATION_ERROR'):
            raise CLPUReliabilityError('A valid complete GUARD certificate is required')
        if (not isinstance(guard_reference, dict) or guard_reference.get('phase') != 'GUARD'
                or guard_reference.get('file') != 'GUARD/phase_result.json'
                or not isinstance(guard_reference.get('sha256'), str)
                or len(guard_reference['sha256']) != 64):
            raise CLPUReliabilityError('Explicit original GUARD evidence reference required')
        self.bootstrap, self.env, self.oracle = bootstrap, bootstrap.env, bootstrap.oracle
        self.selector, self.max_selected_plans = selector, max_selected_plans
        self.guard = deepcopy(guard)
        self.guard_reference = deepcopy(guard_reference)
        guard_info = graph_plan(self.env, self.env.initial, guard['actions'])
        if not math.isclose(guard_info['energy_kwh'], guard['validation']['energy_kwh'],
                            rel_tol=1e-12, abs_tol=1e-10):
            raise CLPUReliabilityError('GUARD graph energy and certificate differ')
        self.best_plan = deepcopy(bootstrap.incumbent)
        self.done, self.failed, self.stop_reason = False, False, None
        self.selected = []
        self.journal, self.edges, self.candidates, self.attempts = [], [], [], []
        self._states, prefix_ids = {}, {}
        self._original_bootstrap_result = deepcopy(bootstrap.result())
        for branch_id, node in sorted(bootstrap._frontier.items()):
            info = graph_plan(self.env, self.env.initial, node['info']['actions'])
            if info['priority'] != node['info']['priority']:
                raise CLPUReliabilityError('Candidate graph data changed after bootstrap')
            edge_ids = []
            for index, state in enumerate(info['states'][1:]):
                prefix = info['actions'][:index + 1]
                cost = (self.env.total_kw - self.env.service_kw(state)) * self.env.slot_seconds / 3600.
                if not math.isfinite(cost) or cost < 0:
                    raise CLPUReliabilityError('Nonnegative unserved-energy edge cost required')
                if prefix not in prefix_ids:
                    edge_id = prefix_ids[prefix] = len(self.edges)
                    self._states[edge_id] = state
                    self.edges.append({'edge_id': edge_id, 'prefix': list(prefix),
                        'state': state.to_dict(), 'step': state.step,
                        'estimated_unserved_kwh': cost, 'status': 'UNKNOWN',
                        'samples': [], 'candidate_ids': []})
                edge_id = prefix_ids[prefix]
                if self.edges[edge_id]['state'] != state.to_dict():
                    raise CLPUReliabilityError('A shared prefix has inconsistent complete history')
                self.edges[edge_id]['candidate_ids'].append(branch_id)
                edge_ids.append(edge_id)
            self.candidates.append({'branch_id': branch_id, 'actions': list(info['actions']),
                'energy_kwh': info['energy_kwh'], 'edge_ids': edge_ids,
                'status': 'PENDING', 'failed_edge_id': None})
        if len(self.candidates) != len({tuple(c['actions']) for c in self.candidates}):
            raise CLPUReliabilityError('Duplicate complete plans in the staged pool')
        # Bootstrap may already have hit a hard budget or a common bound.
        if bootstrap.done:
            self.done, self.stop_reason = True, 'BOOTSTRAP_' + str(bootstrap.stop_reason)

    @property
    def incumbent_energy(self):
        own = self.best_plan['validation']['energy_kwh'] if self.best_plan is not None else -math.inf
        return max(own, self.guard['validation']['energy_kwh'])

    def _counters(self):
        return {'logical_points': self.bootstrap._points(),
                'actual_physical_points': self.bootstrap._actual(self.oracle),
                'requests': getattr(self.oracle, 'requests', None)}

    def _evaluate_edge(self, candidate, edge_id):
        edge, state = self.edges[edge_id], self._states[edge_id]
        if edge['status'] != 'UNKNOWN':
            raise CLPUReliabilityError('Only an unknown edge can be selected')
        event = {'event': 'EDGE_SELECTED', 'branch_id': candidate['branch_id'],
                 'edge_id': edge_id, 'selector': self.selector, 'before': self._counters()}
        self.journal.append(event)
        t0 = (state.step - 1) * self.env.slot_seconds
        try:
            for offset in range(0, self.env.slot_seconds + 1, self.env.sample_seconds):
                seconds = t0 + offset
                cert = self.oracle.certify(state, seconds)
                if cert.get('reliable') is not True or cert.get('status') not in ('PASS', 'FAIL'):
                    raise CLPUReliabilityError('UNKNOWN/unreliable edge sample cannot be classified')
                edge['samples'].append({'step': state.step, 'time_seconds': seconds,
                    'point_id': cert['point_id'], 'status': cert['status'], 'state': state.to_dict()})
                if cert['status'] == 'FAIL':
                    edge['status'] = 'FAIL'
                    break
            else:
                edge['status'] = 'PASS'
        except CLPUBudgetExhausted:
            event['status'] = 'BUDGET_UNKNOWN'
            raise
        except BaseException:
            event['status'] = 'RELIABILITY_OR_IMPLEMENTATION_ERROR'
            raise
        finally:
            event['after'] = self._counters()
        event['status'] = edge['status']
        if edge['status'] == 'FAIL':
            # Tree-prefix sharing is exact; no endpoint/topology-only merging.
            for item in self.candidates:
                if edge_id in item['edge_ids']:
                    item.update(status='FAIL', failed_edge_id=edge_id)

    def step(self):
        if self.done:
            raise RuntimeError('No LazySP action after termination')
        try:
            active = [c for c in self.candidates if c['status'] == 'PENDING'
                      and c['energy_kwh'] > self.incumbent_energy]
            if not active:
                self.done, self.stop_reason = True, 'CERTIFIED_FIXED_POOL_OBJECTIVE_BOUND'
                return
            # Min unserved energy is max service energy; direct service comparison
            # avoids a subtract-then-sum floating-point ordering change.
            candidate = min(active, key=lambda c: (-c['energy_kwh'], c['branch_id']))
            if candidate['branch_id'] not in self.selected:
                if len(self.selected) >= self.max_selected_plans:
                    self.done, self.stop_reason = True, 'DISTINCT_PLAN_SELECTION_CAP'
                    return
                self.selected.append(candidate['branch_id'])
                self.journal.append({'event': 'PATH_SELECTED', 'branch_id': candidate['branch_id'],
                    'optimistic_energy_kwh': candidate['energy_kwh'],
                    'incumbent_energy_kwh': self.incumbent_energy})
            unknown = [i for i in candidate['edge_ids'] if self.edges[i]['status'] == 'UNKNOWN']
            if unknown:
                self._evaluate_edge(candidate, unknown[0] if self.selector == 'forward' else unknown[-1])
                return
            if any(self.edges[i]['status'] != 'PASS' for i in candidate['edge_ids']):
                raise CLPUReliabilityError('An active candidate contains an invalid edge')
            before = self._counters()
            validation = self.env.validate_plan(self.env.initial, candidate['actions'], self.oracle)
            if validation.get('status') != 'COMPLETE' or not math.isclose(
                    validation['energy_kwh'], candidate['energy_kwh'], rel_tol=1e-12, abs_tol=1e-10):
                raise CLPUReliabilityError('Canonical full-plan reconstruction disagrees with edge evidence')
            record = {'actions': list(candidate['actions']), 'origin': 'lazysp_' + self.selector,
                'details': {'branch_id': candidate['branch_id'], 'edge_ids': list(candidate['edge_ids'])},
                'status': 'COMPLETE', 'certified_complete': True, 'validation': validation,
                'physical_points_before': before['logical_points'],
                'physical_points_after': self._counters()['logical_points']}
            self.attempts.append(record)
            if self.bootstrap.record_callback is not None:
                self.bootstrap.record_callback(deepcopy(record))
            self.best_plan = deepcopy(record)
            candidate['status'] = 'COMPLETE'
            self.journal.append({'event': 'CANONICAL_COMPLETE_CERTIFICATE',
                'branch_id': candidate['branch_id'], 'before': before, 'after': self._counters()})
            # The selected optimistic shortest path is now wholly certified.
            self.done, self.stop_reason = True, 'SHORTEST_FIXED_POOL_PATH_CERTIFIED'
        except CLPUBudgetExhausted:
            self.done, self.stop_reason = True, 'PHYSICAL_BUDGET_EXHAUSTED'
        except BaseException:
            self.done, self.failed = True, True
            self.stop_reason = 'LAZYSP_RELIABILITY_OR_IMPLEMENTATION_ERROR'
            raise

    def run(self):
        while not self.done:
            self.step()
        return self.result()

    def result(self):
        result = deepcopy(self._original_bootstrap_result)
        result.update(version=VERSION, method='fixed_prefix_tree_lazysp_' + self.selector,
            done=self.done, stop_reason=self.stop_reason,
            status='RELIABILITY_OR_IMPLEMENTATION_ERROR' if self.failed else
                'CERTIFIED_INCUMBENT' if self.best_plan is not None else 'NO_CERTIFIED_COMPLETE_PLAN',
            best_plan=None if self.failed else deepcopy(self.best_plan),
            incumbent=None if self.failed else deepcopy(self.best_plan),
            attempts=result.get('attempts', []) + deepcopy(self.attempts),
            logical_points=self.bootstrap._points(), physical_points=self.bootstrap._points(),
            actual_physical_points=self.bootstrap._actual(self.oracle), new_training=0,
            guard_established=True, complete_guard_still_required=False,
            optimality_established=False,
            optimality_scope='No global or budget-limited optimality assertion',
            verify_attempts=len(self.selected),
            total_decisions=sum(e['event'] == 'EDGE_SELECTED' for e in self.journal),
            checked_candidates=len(self.bootstrap._checked) + len(self.attempts),
            pending_complete_plans=sum(c['status'] == 'PENDING' for c in self.candidates),
            scope='Literature LazySP adapted to the exact fixed staged-plan prefix tree')
        result['lazysp'] = {'version': VERSION, 'selector': self.selector,
            'graph': 'exact_fixed_complete_plan_prefix_tree_no_new_recombined_paths',
            'cost': 'slot_unserved_normal_energy_or_infinity_for_reliable_FAIL',
            'tie': 'maximum_exact_graph_energy_then_smallest_branch_id',
            'guard_reference': deepcopy(self.guard_reference), 'external_guard_plan': deepcopy(self.guard),
            'max_selected_plans': self.max_selected_plans, 'selected_plan_count': len(self.selected),
            'selected_plan_ids': list(self.selected), 'candidate_records': deepcopy(self.candidates),
            'edges': deepcopy(self.edges), 'journal': deepcopy(self.journal),
            'certificate_reconstruction': 'original_validate_plan_chronological_all_samples_paid_requests',
            'coverage_window': 'global_shortest_path_not_alternating_held_out_subwindow',
            'budget_optimality_claim': False}
        return result
