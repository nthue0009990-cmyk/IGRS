"""Independent staged-pickup replay and conditional common-pilot audit.

Only independent graph mechanics and previously audited physical/ledger/model
checkers are reused. The new production generator and search are not imported.
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import FunctionType
import argparse
import math
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
sys.dont_write_bytecode = True
import audit_clpu_deferred_pilot as previous
shared, early, prior, branch, window, tail, old = (previous.shared, previous.early, previous.prior,
    previous.branch, previous.window, previous.tail, previous.old)
require, close, same = previous.require, previous.close, previous.same
read, save, sha, digest, read_lines = previous.read, previous.save, previous.sha, previous.digest, previous.read_lines
STUDY = 'CLPU_STAGED_COMMON_PILOT_V1'
GENERATOR_VERSION = 'clpu-staged-whole-load-candidates-v2'
ORIGIN = 'staged_pickup'
CONTROL_VERSION = 'clpu-staged-fixed-verification-controller-v1'
ADAPTER_VERSION = 'clpu-staged-whole-load-pending-raw-v1'
PROTOCOL = 'docs/clpu_staged_pickup_pilot_protocol_v1.md'
GENERATOR_CONFIG = {
    'route_limit': 2, 'route_operation_limit': 6, 'initial_lookahead_depth': 6,
    'blocked_load_counts': [1, 2], 'timings': ['NATURAL', 'ONE_TAU'],
    'restore_orders': ['SMALL_FIRST', 'LARGE_FIRST'],
    'max_raw_proposals_per_route': 12, 'max_unique_candidates': 24,
    'horizon': 30, 'slot_seconds': 60, 'sample_seconds': 30,
    'load_order': 'nominal_kw_descending_then_original_dispatch_action',
    'route_order': 'original_graph_priority_then_switch_route_lexicographic',
    'route_diversity': 'distinct_terminal_switch_mask',
    'route_legality': 'prepared_full_plan_only;bare_switch_route_is_diagnostic',
    'timing_rule': 'NATURAL=1;ONE_TAU=ceil(event.tau_seconds/slot_seconds)',
    'wait_insertion': 'gap_slots_minus_one_before_each_restore',
    'hold_generation': 'once_per_route_and_blocked_count_before_all_restore_prefixes',
    'physics': 'unchanged_whole_load_binary_actions_no_reinterruption'}


def independently_generate(mech):
    """Enumerate the exact bounded proposal universe without production helpers."""
    require((mech.horizon, mech.slot, mech.sample, mech.initial['step']) == (30, 60, 30, 0), 'Changed staging time domain')
    dispatch = mech.static['dispatch_loads']
    require(len(dispatch) == len(set(dispatch)) == 10 and all(n in mech.loads for n in dispatch), 'Dispatch inventory changed')
    tau = float(mech.event['tau_seconds'])
    require(math.isfinite(tau) and tau > 0, 'Invalid original time constant')
    gaps = {'NATURAL': 1, 'ONE_TAU': math.ceil(tau / 60)}
    initial_plans = set()
    for action in sorted(mech.legal(mech.initial)):
        child = mech.advance(mech.initial, action)
        initial_plans.add((action,) + mech.suffix(child, 6))
    seeds = []
    for actions in initial_plans:
        plan = mech.plan(actions)
        route = tuple(a for a in actions if 11 <= a <= 19)
        seeds.append((plan['priority'], route, actions, plan['states'][-1]['command'] >> 10))
    seeds.sort(key=lambda row: (row[0], row[1]))
    routes, selected_masks, selected_count = [], set(), 0
    for priority, route, actions, mask in seeds:
        record = {'seed_index': len(routes), 'seed_actions': list(actions), 'seed_graph_energy_kwh': -priority[0],
            'switch_route': list(route), 'terminal_switch_mask': mask, 'selected': False, 'route_rank': None}
        if mask in selected_masks:
            record['status'] = 'DUPLICATE_TERMINAL_TOPOLOGY'
        elif selected_count >= 2:
            record['status'] = 'OUTSIDE_FIXED_ROUTE_LIMIT'
        else:
            selected_masks.add(mask)
            record.update(selected=True, route_rank=selected_count,
                status='UNAVAILABLE_ROUTE_LENGTH' if len(route) > 6 else 'SELECTED')
            selected_count += 1
            if len(route) <= 6:
                terminal = (mask << 10) | (mech.initial['command'] & 1023)
                projection = old.independent_projection(mech.static, mech.loads, mech.scenario, terminal)
                record.update(projected_terminal_command=terminal, projected_terminal_energized=sorted(projection['energized']))
                try:
                    state = mech.initial
                    for action in route:
                        state = mech.advance(state, action)
                    require(state['command'] == terminal, 'Bare switch route terminal changed')
                    record['bare_route_status'] = 'GRAPH_LEGAL'
                except (AssertionError, ValueError) as exc:
                    record.update(bare_route_status='GRAPH_ILLEGAL', bare_route_error=str(exc))
        routes.append(record)
    initially_powered = {n for n, p in zip(mech.names, mech.initial['pickups']) if p is not None}
    proposals, candidates, unique = [], [], {}
    for route in routes:
        if not route['selected'] or route['status'] != 'SELECTED':
            continue
        eligible = [{'action': i + 1, 'load_name': name, 'nominal_kw': float(mech.loads[name]['nominal_kw'])}
            for i, name in enumerate(dispatch) if name in route['projected_terminal_energized'] and
            name not in initially_powered and mech.initial['command'] & (1 << i)]
        eligible.sort(key=lambda row: (-row['nominal_kw'], row['action']))
        route['eligible_whole_loads'] = deepcopy(eligible)
        for count in (1, 2):
            if len(eligible) < count:
                proposals.append({'proposal_id': len(proposals), 'route_rank': route['route_rank'],
                    'blocked_count': count, 'status': 'UNAVAILABLE_LOAD_COUNT', 'available_load_count': len(eligible), 'actions': None})
                continue
            blocked = eligible[:count]
            variants = [('HOLD', [], 0, None)]
            orders = [('SMALL_FIRST', sorted(blocked, key=lambda r: (r['nominal_kw'], r['action'])))]
            if count == 2:
                orders.append(('LARGE_FIRST', list(blocked)))
            for order_name, order in orders:
                for released_count in range(1, count + 1):
                    for timing in ('NATURAL', 'ONE_TAU'):
                        variants.append((order_name, order[:released_count], released_count, timing))
            for order_name, released, released_count, timing in variants:
                operations = [r['action'] for r in blocked] + route['switch_route']
                gap = gaps[timing] if timing is not None else None
                for load in released:
                    operations += [0] * (gap - 1) + [load['action']]
                held = [r['load_name'] for r in blocked if r not in released]
                p = {'proposal_id': len(proposals), 'route_rank': route['route_rank'], 'seed_index': route['seed_index'],
                    'origin': ORIGIN, 'blocked_count': count, 'blocked_loads': deepcopy(blocked),
                    'restore_order': order_name, 'release_count': released_count, 'released_loads': deepcopy(released),
                    'held_out_loads': held, 'terminal_service_class': 'FULL_HORIZON_HELD_OUT' if held else 'ALL_BLOCKED_RELEASED',
                    'timing': timing, 'gap_slots': gap, 'tau_seconds': tau, 'switch_route': list(route['switch_route']),
                    'operation_prefix': operations, 'planned_prefix_length': len(operations), 'actions': None,
                    'physical_status': 'UNVERIFIED', 'quality_pct': None}
                proposals.append(p)
                if len(operations) > 30:
                    p['status'] = 'UNAVAILABLE_HORIZON'
                    continue
                actions = tuple(operations + [0] * (30 - len(operations)))
                last = max((i for i, a in enumerate(actions) if a), default=-1)
                p.update(actions=list(actions), prefix=list(actions[:last + 1]))
                try:
                    info = mech.plan(actions)
                except (AssertionError, ValueError) as exc:
                    p.update(status='GRAPH_ILLEGAL', error=str(exc))
                    continue
                batches = []
                for before, after in zip(info['states'], info['states'][1:]):
                    picked = [n for n, a, b in zip(mech.names, before['pickups'], after['pickups']) if a is None and b is not None]
                    if picked:
                        batches.append({'time_seconds': (after['step'] - 1) * 60, 'step': after['step'],
                            'load_names': picked, 'nominal_kw': math.fsum(float(mech.loads[n]['nominal_kw']) for n in picked)})
                require(all(info['states'][-1]['pickups'][mech.names.index(n)] is None for n in held), 'Held load already supplied')
                p.update(graph_energy_kwh=info['energy_kwh'], graph_final_kw=info['final_kw'],
                    operations=info['operations'], pickup_batches=batches)
                if actions in unique:
                    p.update(status='DUPLICATE', duplicate_of=unique[actions])
                    candidates[unique[actions]]['alias_proposal_ids'].append(p['proposal_id'])
                else:
                    i = len(candidates)
                    p.update(status='LEGAL_UNIQUE', candidate_id=i)
                    unique[actions] = i
                    candidates.append({**deepcopy(p), 'alias_proposal_ids': [p['proposal_id']]})
    require(len(proposals) <= 24 and len(candidates) <= 24, 'Staged proposal budget exceeded')
    return {'version': GENERATOR_VERSION, 'config': deepcopy(GENERATOR_CONFIG), 'event_tau_seconds': tau,
        'gap_slots': gaps, 'routes': routes, 'proposals': proposals, 'candidates': candidates,
        'counts': {'seed_plans': len(routes), 'selected_routes': sum(r['selected'] for r in routes),
            'raw_proposals': len(proposals), 'unique_legal_candidates': len(candidates),
            'proposal_status': dict(sorted(Counter(p['status'] for p in proposals).items()))},
        'new_physical_points': 0, 'new_training': 0, 'physical_certificates_created': 0}


def check_generation(actual, expected):
    """Graph rejection categories matter; implementation-specific errors do not."""
    def cleaned(value):
        if isinstance(value, dict):
            for key in ('error', 'bare_route_error'):
                if key in value:
                    require(isinstance(value[key], str) and value[key], 'Empty graph rejection reason')
            return {k: cleaned(v) for k, v in value.items() if k not in ('error', 'bare_route_error')}
        if isinstance(value, list):
            return [cleaned(v) for v in value]
        return value
    same(cleaned(actual), cleaned(expected), 'independent_staged_generator')


def exposed_nodes(nodes, verify_attempts, incumbent_energy):
    require(len(nodes) <= 24 and len({n['id'] for n in nodes}) == len(nodes), 'Changed common staged pool')
    ordered = sorted(nodes, key=lambda n: n['id'])
    if verify_attempts % 2 == 1 and incumbent_energy is not None:
        preferred = [n for n in ordered if n['info']['energy_kwh'] > incumbent_energy and n['held_out_loads']]
        if preferred:
            return preferred
    return ordered


def phase_gate(checks):
    """Only completed cold paired searches can authorize the frozen second cohort."""
    require(checks and len({r['cell'] for r in checks}) == len(checks), 'Phase one audit incomplete')
    positive = []
    for row in checks:
        require(row['status'] == 'PASS' and row['guard_established'] is True, 'Unaudited or incomplete GUARD cannot gate')
        e, g = row['early_energy_kwh'], row['guard_energy_kwh']
        require(type(g) in (int, float) and math.isfinite(g), 'Missing complete GUARD energy')
        if row['early_fully_certified']:
            require(type(e) in (int, float) and math.isfinite(e), 'Missing complete EARLY energy')
            if e > g:
                positive.append(row['cell'])
        else:
            require(e is None, 'Incomplete EARLY has invented complete quality')
    return {'phase2_authorized': bool(positive),
        'criterion': 'any fully certified EARLY energy strictly above own complete GUARD', 'positive_cells': positive}


class StagedView(branch.BranchReplay):
    def window(self):
        rows = [{**node, 'held_out_loads': self.stage_branch_metadata[node['id']]['held_out_loads']}
            for node in self.frontier.values()]
        return exposed_nodes(rows, self.verify_attempts,
            self.best['validation']['energy_kwh'] if self.best is not None else None)


class StagedReplay(previous.DeferredReplay):
    def __init__(self, *args):
        self.stage_generation, self.stage_branch_metadata = None, {}
        super().__init__(*args)

    def expand(self, prefix, parent):
        require(not prefix and parent is None and self.stage_generation is None, 'Unexpected extra staged generation')
        before = self.costs()
        self.stage_generation = independently_generate(self.m)
        for candidate in self.stage_generation['candidates']:
            bid = len(self.keys)
            self.add(candidate['prefix'], candidate['actions'], ORIGIN, None)
            require(len(self.keys) == bid + 1, 'Unique staged candidate lost before exposure')
            self.stage_branch_metadata[bid] = deepcopy(candidate)
        require(self.costs() == before, 'Staged generation consumed numerical evidence')

    def holds_and_repairs(self, node, failed):
        return None

    def refresh(self):
        super().refresh()
        if not self.done and not self.queue:
            self.stop('STAGED_VERIFICATION_POOL_EXHAUSTED')

    def operate(self, operation, branch_id):
        require(operation in ('VERIFY_COMPLETE_PLAN', 'YIELD_TO_BEAM'), 'Staged phase performs hidden graph expansions')
        response = super().operate(operation, branch_id)
        require(self.graph_expansions == 0 and self.verify_attempts <= 16 and self.total_decisions <= 17, 'Staged operation cap exceeded')
        return response

    def verification_view(self):
        view = object.__new__(StagedView)
        view.__dict__ = dict(self.__dict__)
        view.frontier = {bid: node for bid, node in self.frontier.items()
            if node['info']['actions'] in self.queue and node['info']['actions'] not in self.checked}
        view.turn = self.verify_attempts
        aliases = {node['info']['actions']: [bid] for bid, node in view.frontier.items()}
        return view, aliases

    def scientific_snapshot(self):
        result = super().scientific_snapshot()
        result.update(version=CONTROL_VERSION, method='fixed_staged_candidate_certification',
            stage_generation=deepcopy(self.stage_generation),
            stage_branch_metadata={str(k): deepcopy(v) for k, v in self.stage_branch_metadata.items()},
            stage_generation_status='GENERATED' if self.stage_generation is not None else 'BOOTSTRAP_STOP_NOT_GENERATED',
            bootstrap_graph_generation='fixed_staged_pool_replaces_original_initial_prefix_expansion',
            reliable_failure_generation='none_fixed_pool_feedback_still_recorded')
        return result


def adapter_metadata(replay):
    expected = previous.adapter_metadata(replay)
    expected.update(version=ADAPTER_VERSION,
        window_rule={'ordinary': 'all_pending_staged_unique_plans_in_branch_id_order',
            'coverage_turn': 'completed_verify_attempts_mod_2_equals_1',
            'eligible': 'still_held_out_and_graph_energy_strictly_above_certified_incumbent',
            'no_eligible_fallback': 'all_pending_staged_unique_plans_in_branch_id_order',
            'capacity': 32, 'max_pool_size': 24},
        representative_order=['one_unique_full_plan_per_generated_candidate'],
        new_origin=ORIGIN, prefix_wait_feature_for_staged=0,
        distribution_shift='staged_only_full_plan_pool_and_ready_bit_restore_timing')
    return expected


class StagedMacroReplay(previous.MacroReplay):
    def __init__(self, replay, schedule, method):
        require(schedule == 'staged' and method in ('beam', 'bc_raw'), 'Unapproved staged ranker')
        self.search, self.schedule, self.method = replay, schedule, method
        self.cursor = self.policy_decisions = 0
        self.skipped_slots, self.records, self.chosen_plans = [], [], set()

    def next(self):
        if self.search.done:
            return None
        view, aliases = self.search.verification_view()
        if self.cursor < 16 and view.window():
            return 'VERIFY_COMPLETE_PLAN', None, view, aliases
        return 'YIELD_TO_BEAM', None, None, None

    def mapping(self, view, aliases):
        rows = super().mapping(view, aliases)
        for row in rows:
            metadata = self.search.stage_branch_metadata[row['branch_id']]
            row.update(candidate_id=metadata['candidate_id'], held_out_loads=deepcopy(metadata['held_out_loads']),
                timing=metadata['timing'], route_rank=metadata['route_rank'])
        return rows

    def check_control_result(self, result):
        require(self.next() is None and self.search.done, 'Staged phase did not finish')
        expected = {'version': CONTROL_VERSION, 'engine_version': CONTROL_VERSION, 'generator_version': GENERATOR_VERSION,
            'schedule': 'staged', 'requested_method': self.method, 'adapter': adapter_metadata(self.search),
            'graph_operation_limit': 0, 'verification_choice_limit': 16, 'total_operation_limit_including_yield': 17,
            'bootstrap_outside_macro_limits_but_paid': True,
            'verify_failure_repairs': 'none_fixed_candidate_pool_feedback_updates_unchanged',
            'dominated_skip_consumes_verification_choice': True, 'schedule_slots_consumed': self.cursor,
            'skipped_slots': [], 'records': self.records, 'policy_decisions': self.policy_decisions,
            'controller_failed': False, 'finish_reason': self.search.reason,
            'guard_established': False, 'complete_guard_still_required': True}
        same(result['control'], expected, 'independent_staged_controller')
        require(result['policy_decisions'] == self.policy_decisions and
            result['training_interactions'] == result['optimizer_updates'] == 0, 'Hidden training or policy participation')
        scientific = self.search.scientific_snapshot()
        actual = deepcopy(result)
        for record in actual['deferred_trace']:
            window.finite_time(record.pop('elapsed_seconds'), 'staged operation elapsed')
        window.finite_time(actual['bootstrap_cost'].pop('elapsed_seconds'), 'staged paid bootstrap')
        window.finite_time(actual['stage_generation_seconds'], 'staged graph generation elapsed')
        check_generation(actual['stage_generation'], scientific['stage_generation'])
        for key, value in scientific.items():
            if key != 'stage_generation':
                same(actual[key], value, 'independent_staged_result.' + key)
        require(self.search.oracle.index == len(self.search.oracle.rows) and
            self.search.oracle.used == len(self.search.logical_rows) and
            self.search.attempt_index == len(self.search.attempts), 'Unconsumed staged physical/plan evidence')
        return scientific


# Reuse the complete old independent dual-observation masked-policy checks and
# compact journal checks, adapting only the independent replay classes/version.
StagedMacroReplay.check_proposal = FunctionType(previous.MacroReplay.check_proposal.__code__,
    {**previous.MacroReplay.check_proposal.__globals__, 'CONTROL_VERSION': CONTROL_VERSION},
    'check_proposal', previous.MacroReplay.check_proposal.__defaults__, previous.MacroReplay.check_proposal.__closure__)
_check_records = FunctionType(previous.check_deferred_records.__code__,
    {**previous.check_deferred_records.__globals__, 'DeferredReplay': StagedReplay, 'MacroReplay': StagedMacroReplay},
    '_check_records', previous.check_deferred_records.__defaults__, previous.check_deferred_records.__closure__)


def check_staged_records(mech, cfg, spec, result, attempts, decisions, evidence, condition_sha, policy=None):
    checked = _check_records(mech, cfg, spec, result, attempts, decisions, evidence, condition_sha, policy)
    require(checked['graph_expansions'] == 0 and checked['verify_attempts'] <= 16 and
        checked['total_decisions'] <= 17, 'Staged decision budgets changed')
    checked.update(candidate_pool_sha256=digest(result['stage_generation']),
        stage_generation_verified=True, complete_energy_kwh=result['best_plan']['validation']['energy_kwh']
            if result['best_plan'] is not None else None)
    return checked


import xml.etree.ElementTree as ET
import hashlib
SEEDS = (17, 41, 73)
SLOTS = ("staged_window", "staged_bc17", "staged_bc41", "staged_bc73")
CELLS = 32
MIN_FREE, RESERVE, MAX_SECONDS, WORKER_SECONDS, TOTAL_CAP = 12*2**30, 2*2**30, 3600, 900, 16384
SOURCE_RUN = "20261031T053323_555256Z_clpu_deferred_pilot"
SOURCE_SHA = "4dfecffaee422d8d712e4b8bb2c01d95b22d97eb3721f947866a4861b4c1f56c"
MODELS_RUN, MODELS_SHA = previous.MODELS_RUN, previous.MODELS_SHA
canonical = prior.canonical
cache = previous.cache


def protocol_settings():
    return {'cells':CELLS, 'contexts':8, 'mode':'shared', 'phases':['EARLY','GUARD'],
        'seeds':list(SEEDS), 'slots':list(SLOTS), 'logical_budget_per_view':256,
        'early_expansion_slots':0, 'early_verification_slots':16, 'maximum_early_operations':17,
        'candidate_generation':'one fresh pure-graph staged pool after paid bootstrap; at most24 unique full plans',
        'failure_repairs_add_candidates':False,
        'bootstrap':'original paid initial snapshot and complete WAIT', 'guard_expansion_cap':256,
        'total_physical_budget_per_cell':512, 'total_physical_budget':TOTAL_CAP,
        'maximum_numerical_entries':CELLS*3+4*TOTAL_CAP, 'physical_runtimes_per_cell':1,
        'max_workers':2, 'worker_seconds':WORKER_SECONDS, 'controller_seconds':MAX_SECONDS,
        'minimum_free_disk_bytes':MIN_FREE, 'storage_margin':1.25, 'report_reserve_bytes':RESERVE,
        'phase_order':['EARLY','GUARD'], 'tie_rule':'exact_equal_energy_select_GUARD',
        'same_runtime_both_modes':True, 'new_training':0, 'no_cross_cell_cache':True,
        'no_historical_numerical_answers':True, 'no_automatic_retry':True,
        'no_upper_bound_change':True, 'model_role':'fixed BC candidate ranking only; no B learner or trained staging policy',
        'macro_control':'one shared generator call then up to16 VERIFY and at most1 YIELD; no EXPAND',
        'same_optimized_window_for_all_rankers':True, 'no_chosen_plan_targets':True,
        'timing':'cold complete cell process; all initialization, both searches, logging and exit paid',
        'recording':'STARTED proposal with raw observation and stable mapping; COMPLETED exact response; separate logical and real costs',
        'dispatch':'hashed contexts and slots, Latin slot rotation; adjacent pair drained before next pair',
        'aggregation':'mean three BC seeds within context then equal mean eight contexts; retain all32 tasks',
        'conditional_stages':'all32 frozen; graph-improving contexts first; remaining only after audited EARLY improvement over own GUARD',
        'no_gain_remaining_status':'NOT_RUN_GATE_NO_GAIN'}


def expected_schedule(contexts, models):
    require(len(contexts) == 8 and {c['context_id'] for c in contexts} == set(range(8))
            and len({c['scenario_id'] for c in contexts}) == 8, 'Complete fixed eight contexts required')
    lookup = {(m['method'], m['seed']): m for m in models}
    require(len(models) == len(lookup) == 6 and set(lookup) == {(m, s) for m in ('bc_raw', 'ppo_raw') for s in SEEDS},
            'Fixed six checkpoint population differs')
    ordered = sorted(contexts, key=lambda c: prior.digest_text('staged-context-v1|' + prior.canonical(c)))
    slots = sorted(SLOTS, key=lambda s: prior.digest_text('staged-slots-v1|' + s))
    rows = []
    for rank, c in enumerate(ordered):
        for position in range(4):
            slot = slots[(rank + position) % 4]
            seed = next((s for s in SEEDS if slot.endswith(str(s))), 0)
            arm = slot[:-len(str(seed))] if seed else slot
            method = 'bc_raw' if arm == 'staged_bc' else 'beam'
            model = lookup.get((method, seed))
            index = len(rows)
            rows.append({**c, 'cell': f'episodes/{index:03d}', 'episode_id': index, 'pair_order': rank * 2 + position // 2,
                'context_order': rank, 'arm_position': position, 'arm_slot': slot,
                'mode': 'shared', 'arm': arm, 'method': method, 'seed': seed,
                'macro_schedule': 'staged',
                'phase': 'evaluation', 'group': arm, 'protected': True, 'representation': 'raw', 'deterministic': True,
                'checkpoint': model['checkpoint'] if model else None,
                'checkpoint_sha256': model['source_sha256'] if model else None,
                'point_budget': 256, 'logical_point_budget': 256, 'physical_point_budget': 512,
                'total_logical_point_budget': 512, 'window_variant': 'staged_common_pool_coverage_v1'})
    return rows


expected_phase_spec = previous.expected_phase_spec
check_models = previous.check_models


def check_parent(run):
    run = Path(run).resolve()
    require(run.parent == ROOT / 'runs', 'Deferred pilot outside project run directory')
    prepared = read(run / 'prepared.json')
    require(prepared['status'] == 'FROZEN_PREPARED_NO_DSS'
            and prepared['freeze_sha256'] == sha(run / 'prephysical_freeze_manifest.json'), 'Prephysical freeze changed')
    frozen = shared.checked_manifest(run, 'prephysical_freeze_manifest.json')
    source = ROOT / 'runs' / SOURCE_RUN
    require(sha(source / 'completion_manifest.json') == sha(run / 'source_completion_manifest.json') == SOURCE_SHA,
            'Original shared source seal changed')
    sealed = {k.replace('\\', '/'): v for k, v in read(run / 'source_completion_manifest.json').items()}
    require(read(run / 'source_binding.json') == {'source_run': SOURCE_RUN, 'source_completion_sha256': SOURCE_SHA,
        'models_run': MODELS_RUN, 'models_completion_sha256': MODELS_SHA}, 'Source/model attribution changed')
    for name in ('config.json', 'schedule.json', 'prepared.json', 'audit/summary.json', 'contexts.json'):
        require(sha(source / name) == sealed[name], 'Changed historical input: ' + name)
    source_audit = read(source / 'audit/summary.json')
    require(source_audit['status'] == 'PASS' and source_audit['audited_cells'] == 64 and not source_audit['errors'],
            'Shared source not completely audited')
    additions = {PROTOCOL, 'src/dnr/nature34_clpu_staged_pickup.py',
        'src/dnr/nature34_clpu_staged_control.py', 'scripts/run_clpu_staged_pilot.py',
        'scripts/audit_clpu_staged_pilot.py', 'tests/test_nature34_clpu_staged_pickup.py',
        'tests/test_nature34_clpu_staged_control.py', 'tests/test_run_clpu_staged_pilot.py',
        'tests/test_clpu_staged_audit.py'}
    require(prepared['source_files'] == sorted(set(read(source / 'prepared.json')['source_files']) | additions),
            'Required independent/control/source closure changed')
    for name in prepared['source_files']:
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT) and path != ROOT and 'source/' + name in frozen
                and sha(path) == sha(run / 'source' / name), 'Changed frozen execution source: ' + name)
        if 'source/' + name in sealed:
            require(sha(path) == sealed['source/' + name], 'Inherited scientific source changed: ' + name)
    require(sha(run / 'protocol.md') == sha(ROOT / PROTOCOL), 'Protocol changed')
    cfg = read(run / 'config.json')
    same(cfg, {**read(source / 'config.json'), 'staged_study': STUDY, 'staged_pilot': protocol_settings()},
         'complete_deferred_scientific_contract')
    scope = prior.check_frozen_physics(run, prepared, cfg)
    models = check_models(run)
    contexts = read(source / 'contexts.json')
    require(read(run / 'contexts.json') == contexts, 'Original training population changed')
    schedule = read(run / 'schedule.json')
    same(schedule, expected_schedule(contexts, models), 'all_64_cold_deferred_tasks')
    for spec in schedule:
        require(read(run / spec['cell'] / 'spec.json') == spec, 'Prepared task differs')
        for name in shared.VIEWS:
            require(read(run / spec['cell'] / name / 'spec.json') == expected_phase_spec(spec, name), 'Prepared logical view differs')
    chosen = [s for s in read(source / 'schedule.json') if s['arm'] == 'deferred_window']
    expected_history = []
    for spec in chosen:
        item = {k: spec[k] for k in ('context_id', 'scenario_id', 'event_id')}
        original = spec['cell'] + '/GUARD/phase_result.json'
        target = f'historical/c{spec["context_id"]:02d}/GUARD/phase_result.json'
        require(sha(run / target) == sha(source / original) == sealed[original], 'Original complete Beam result changed')
        item.update(source_cell=spec['cell'], phases={'GUARD': {'result_file': target,
                    'source_file': original, 'sha256': sealed[original]}})
        expected_history.append(item)
    require(read(run / 'historical_bindings.json') == expected_history, 'Guard source attribution changed')
    cases = list(ET.parse(run / 'software_tests.xml').getroot().iter('testcase'))
    require(cases and not any(list(c.iter('failure')) + list(c.iter('error')) + list(c.iter('skipped')) for c in cases),
            'Passed complete frozen tests required')
    for component in ('test_nature34_clpu_staged_pickup', 'test_nature34_clpu_staged_control',
                      'test_run_clpu_staged_pilot', 'test_clpu_staged_audit'):
        require(any(component in c.get('classname', '') for c in cases), 'Missing frozen component: ' + component)
    resource = read(run / 'resource_budget.json')
    for key, value in {'total_physical_cap': 16384, 'minimum_free_bytes': 12 * 2**30,
        'report_reserve_bytes': 2 * 2**30, 'successful_component_tests': len(cases)}.items():
        require(resource[key] == value, 'Resource contract changed: ' + key)
    rate = max(sum(p.stat().st_size for p in (source / s['cell']).rglob('*') if p.is_file()) /
               max(1, read(source / s['cell'] / 'result.json')['actual_physical_points']) for s in chosen)
    require(resource['historical_max_bytes_per_actual_point'] == rate
            and resource['forecast_bytes'] == math.ceil(rate * 16384 * 1.25)
            and resource['required_free_bytes'] == 14 * 2**30 + resource['forecast_bytes']
            and resource['free_bytes_at_prepare'] >= resource['required_free_bytes'], 'Incomplete storage forecast')
    require(read(run / 'authorization.json')['execution_authorized'] is True, 'Missing execution authorization')
    check_stage_plan(run, cfg, schedule, sealed)
    return cfg, scope, schedule


def _check_base_cell(run, spec, cfg, scope):
    cell = (run / spec['cell']).resolve()
    require(cell.is_relative_to(run) and cell != run, 'Escaped cold task')
    sealed = shared.checked_manifest(cell)
    required = {'spec.json', 'result.json', 'worker_started.json', 'worker_finished.json', 'execution.json',
        'service_events.jsonl', 'service_summary.json', 'cost_checkpoints.jsonl', 'physical/points.jsonl.gz',
        'physical/requests.jsonl', 'physical/oracle_summary.json', 'physical/command_ledger.jsonl'}
    for name in shared.VIEWS:
        required |= {name + '/' + p for p in ('spec.json', 'phase_result.json', 'initial_state.json', 'plans.jsonl.gz',
            'decisions.jsonl.gz', 'logical_points.jsonl.gz', 'logical_requests.jsonl', 'view_summary.json')}
        require(not (cell / name / 'points.jsonl.gz').exists() and not (cell / name / 'command_ledger.jsonl').exists(),
                'A logical view fabricates paid physical evidence')
    require(required <= set(sealed) and not (cell / 'error.json').exists(), 'Incomplete/failed task evidence')
    result = read(cell / 'result.json')
    require(read(cell / 'spec.json') == result['spec'] == spec and result['new_training'] == 0,
            'Task identity or unauthorized training changed')
    require(result['version'] == result['study'] == STUDY and result['schema'] == 'clpu-staged-protected-result-v1',
            'Deferred scientific output version differs')
    require(result['checkpoint_sha256'] == spec['checkpoint_sha256'], 'Fixed checkpoint differs')
    policy = early.IndependentPolicy(run / spec['checkpoint'], spec) if spec['checkpoint'] else None
    worker, finished, execution = [read(cell / p) for p in ('worker_started.json', 'worker_finished.json', 'execution.json')]
    require(type(worker['pid']) is int and worker['pid'] > 0 and finished['pid'] == worker['pid']
            and finished['status'] == 'COMPLETE' and finished['result_sha256'] == sha(cell / 'result.json'),
            'Cold worker completion unbound')
    global_execution = read(run / 'execution_started.json')
    require(execution['spec_sha256'] == sha(cell / 'spec.json') and execution['controller_pid'] == read(run / ('controller_stage1_started.json' if execution['stage'] == 'phase1' else 'controller_stage2_started.json'))['pid']
            and execution['deadline_epoch_seconds'] == min(global_execution['deadline_epoch_seconds'],
                                                           execution['started_epoch_seconds'] + 900), 'Wrong worker deadline')
    scenarios = {r['scenario_id']: r for r in read(run / 'selected_scenarios.json')}
    events = {r['event_id']: r for r in read(run / 'selected_events.json')}
    mech = branch.Mechanics(read(run / 'author_static_metadata.json'), read(run / 'load_reference.json'),
        scenarios[spec['scenario_id']], read(run / 'condition.json'), events[spec['event_id']], cfg)
    points, statuses = shared.check_actual_points(run, cell / 'physical', spec, mech)
    commands = old.check_commands(cell / 'physical', points, read(run / 'command_inventory.json'))
    shared.check_actual_requests(list(read_lines(cell / 'physical/requests.jsonl')), points,
        read(cell / 'physical/oracle_summary.json'), commands, statuses)
    views = {name: {'logical_rows': list(read_lines(cell / name / 'logical_points.jsonl.gz')),
                   'requests': list(read_lines(cell / name / 'logical_requests.jsonl'))} for name in shared.VIEWS}
    evidence, _ = shared.check_service_ledger(list(read_lines(cell / 'service_events.jsonl')), views, points,
        mech, scope, sha(run / 'condition.json'), 'shared')
    service = read(cell / 'service_summary.json')
    shared.check_service_summaries(service, {name: read(cell / name / 'view_summary.json') for name in shared.VIEWS},
                                  evidence, scope, 'shared', commands['numerical_entries'])
    require(result['service'] == service, 'Complete service accounting differs')
    phases, checks = {}, {}
    history = next(r for r in read(run / 'historical_bindings.json') if r['context_id'] == spec['context_id'])
    for name in shared.VIEWS:
        directory = cell / name
        p = read(directory / 'phase_result.json')
        require(read(directory / 'spec.json') == p['spec'] == expected_phase_spec(spec, name)
                and read(directory / 'initial_state.json') == mech.initial, 'Search state or identity changed')
        require(p['oracle'] == service['views'][name], 'View summary differs from physical service')
        plans, decisions = list(read_lines(directory / 'plans.jsonl.gz')), list(read_lines(directory / 'decisions.jsonl.gz'))
        require(p['attempts'] == plans, 'Omitted/relabelled attempted plan')
        if name == 'EARLY':
            checked = check_staged_records(mech, cfg['branch_search'], p['spec'], p, plans, decisions,
                                            evidence[name], sha(run / 'condition.json'), policy)
        else:
            checked = shared.check_phase_records(mech, cfg['branch_search'], name, p['spec'], p, plans,
                decisions, evidence[name], sha(run / 'condition.json'))
            require(p['verify_attempts'] is None and p['graph_expansions'] == checked['decisions'],
                    'Original Beam combined steps confused with atomic VERIFY')
            original = read(run / history['phases']['GUARD']['result_file'])
            original = {**original, 'actual_physical_points': original['logical_points']}
            for key, value in checked['scientific_result'].items():
                same(original[key], value, 'historical_complete_original_Beam.' + key)
            require(original['oracle']['requests'] == evidence[name]['requests'], 'Original Beam query order/count changed')
        phases[name], checks[name] = p, checked
    # Only version labels are projected to the frozen pure protection checker;
    # all actual results and the new version were validated immediately above.
    chosen = shared.check_selection({**result, 'version': shared.STUDY, 'study': shared.STUDY,
                                     'schema': 'clpu-shared-protected-result-v1'}, phases)
    cps = list(read_lines(cell / 'cost_checkpoints.jsonl'))
    require(result['checkpoints'] == cps, 'Top quality/cost curve omits boundaries')
    shared.check_checkpoints(cps, checks, phases)
    for name in shared.VIEWS:
        require(result['phase_summary'][name] == {k: phases[name][k] for k in ('quality_pct', 'fully_certified',
            'logical_points', 'actual_physical_points', 'search_decisions', 'policy_decisions', 'graph_expansions',
            'verify_attempts', 'phase_elapsed_seconds', 'stage_component_seconds')}, 'Phase summary incomplete')
    require(set(result['phase_summary']) == set(shared.VIEWS), 'Extra/missing search phases')
    for key, expected in {'logical_points': service['logical_points'], 'actual_physical_points': len(points),
        'numerical_entries': commands['numerical_entries'], 'requests': sum(e['requests'] for e in evidence.values()),
        'search_decisions': sum(p['search_decisions'] for p in phases.values()),
        'policy_decisions': checks['EARLY']['policy_decisions']}.items():
        require(result[key] == expected, 'Task cost/participation differs: ' + key)
    shared.check_times(result, phases, finished, math.fsum(p['elapsed_seconds'] for p in points.values()))
    model_time = window.finite_time(result['model_load_seconds'], 'model loading cost')
    require(model_time <= result['preflight_elapsed_seconds'] and (spec['checkpoint'] is not None or model_time == 0.),
            'Model work hidden from cold end-to-end time')
    if spec['checkpoint']:
        require(sha(run / spec['checkpoint']) == spec['checkpoint_sha256'], 'Frozen model changed during audit')
    return {**{k: spec[k] for k in ('cell', 'context_id', 'scenario_id', 'event_id', 'mode', 'arm', 'seed',
        'pair_order', 'arm_slot', 'macro_schedule')}, 'status': 'PASS', 'result_sha256': sha(cell / 'result.json'),
        'cell_completion_sha256': sha(cell / 'completion_manifest.json'), 'quality_pct': result['quality_pct'],
        'guard_established': chosen['guard_established'], 'guard_quality_pct': result['baseline_quality_pct'],
        'gain_over_guard_kwh': chosen['gain_over_guard_kwh'], 'logical_points': service['logical_points'],
        'physical_points': len(points), 'numerical_entries': commands['numerical_entries'], 'requests': result['requests'],
        'search_decisions': result['search_decisions'], 'policy_decisions': result['policy_decisions'],
        'worker_elapsed_seconds': result['worker_elapsed_seconds'], 'model_load_seconds': model_time,
        'worker_finished_elapsed_seconds': finished['worker_elapsed_seconds'], 'worker_pid': worker['pid'],
        'worker_parent_pid': worker['parent_pid'], 'execution': execution, 'physical_statuses': dict(statuses),
        'history_verified': True, 'quality_checkpoints_verified': True,
        'cross_view_reuses': sum(e['cross_view_reuses'] for e in evidence.values()),
        'first_verify_state_sha256': checks['EARLY']['first_verify_state_sha256'],
        'actual_and_independent_policy_actions_equal': True,
        'phase_summary': {name: {**result['phase_summary'][name], 'requests': evidence[name]['requests'],
            'semantic_signature_sha256': checks[name]['semantic_signature_sha256'],
            'phase_result_sha256': sha(cell / name / 'phase_result.json'),
            **({k: checks[name][k] for k in ('total_decisions', 'handed_off', 'compact_hash_exact_matches', 'raw_hash_exact_matches')}
               if name == 'EARLY' else {})} for name in shared.VIEWS}, 'exact_checkpoints': cps}


def check_stage_plan(run, cfg, schedule, source_manifest):
    plan = read(run / 'stage_plan.json')
    contexts = read(run / 'contexts.json')
    rows = plan['contexts']
    require(len(rows) == 8 and [r['context_id'] for r in rows] == list(range(8)), 'Missing or reordered graph cohorts')
    scenarios = {r['scenario_id']: r for r in read(run / 'selected_scenarios.json')}
    events = {r['event_id']: r for r in read(run / 'selected_events.json')}
    histories = {r['context_id']: r for r in read(run / 'historical_bindings.json')}
    eligible = []
    for context, row in zip(sorted(contexts, key=lambda r: r['context_id']), rows):
        require(all(row[k] == v for k, v in context.items()), 'Graph cohort identity changed')
        binding = histories[context['context_id']]['phases']['GUARD']
        reference = read(run / binding['result_file'])
        best = reference['best_plan']
        require(reference['fully_certified'] and best['certified_complete'] and best['validation']['status'] == 'COMPLETE', 'Incomplete historical reference')
        require(row['reference_source_file'] == binding['source_file'] and
            row['reference_source_sha256'] == binding['sha256'] == source_manifest[binding['source_file']] and
            row['reference_energy_kwh'] == best['validation']['energy_kwh'] and
            row['reference_quality_pct'] == reference['quality_pct'], 'Graph selection reference changed')
        mech = branch.Mechanics(read(run / 'author_static_metadata.json'), read(run / 'load_reference.json'),
            scenarios[context['scenario_id']], read(run / 'condition.json'), events[context['event_id']], cfg)
        rebuilt = independently_generate(mech)
        check_generation(row['generation'], rebuilt)
        require(row['generation_sha256'] == digest(row['generation']), 'Frozen graph inventory hash differs')
        # The two computations must agree on the strict cohort boundary; no
        # close()/isclose() can turn a merely equal objective into headroom.
        a = any(c['graph_energy_kwh'] > row['reference_energy_kwh'] for c in row['generation']['candidates'])
        b = any(c['graph_energy_kwh'] > row['reference_energy_kwh'] for c in rebuilt['candidates'])
        require(a == b, 'Strict graph eligibility is numerically ambiguous')
        if b:
            eligible.append(context['context_id'])
    expected = {'schema': 'clpu-staged-fixed-partition-v1', 'original_contexts': 8, 'scheduled_cells': 32,
        'eligible_contexts': eligible,
        'phase1_cells': [s['cell'] for s in schedule if s['context_id'] in eligible],
        'phase2_cells': [s['cell'] for s in schedule if s['context_id'] not in eligible],
        'phase1_rule': 'any complete staged graph objective strictly above same-context historical certified reference',
        'phase2_rule': 'any audited complete EARLY objective strictly above own complete GUARD',
        'contexts': rows, 'new_physical_points': 0, 'new_training': 0}
    require(plan == expected, 'Conditional partition or original 32 slots changed')
    check_graph_profile(run, plan)
    return plan


def check_graph_profile(run, plan):
    expected_sha = '96227b394dd6a447cdbf06470c9fad6581653c0606caf51afdb22bb566143c32'
    require(read(run/'graph_profile_binding.json') == {
        'path':'artifacts/clpu_staged_pickup_graph_20261031_v2','completion_sha256':expected_sha}, 'Graph profile attribution changed')
    path = run/'graph_profile'
    require(sha(path/'completion_manifest.json') == expected_sha, 'Zero-solver profile root seal changed')
    shared.checked_manifest(path)
    frozen, summary = read(path/'frozen.json'), read(path/'summary.json')
    require(frozen['source_completion_sha256'] == SOURCE_SHA and summary['contexts'] == 8
        and summary['new_physical_points'] == summary['new_training'] == 0, 'Wrong original graph profile scope')
    for name in ('src/dnr/nature34_clpu_staged_pickup.py','src/dnr/nature34_clpu_staged_control.py'):
        require(frozen['source_sha256'][name] == sha(ROOT/name), 'Graph inventory generator changed')
    profile = read(path/'all_contexts.json')
    require(len(profile) == 8 and {p['context_id'] for p in profile} == set(range(8)), 'Graph profile population missing')
    for row in plan['contexts']:
        original = next(p for p in profile if p['context_id'] == row['context_id'])
        require(all(row[k] == original[k] for k in ('scenario_id','event_id','reference_energy_kwh',
            'reference_quality_pct','generation','generation_sha256')), 'Prephysical and cold profile disagree')
    require(plan['eligible_contexts'] == summary['contexts_with_graph_potential'], 'Graph cohort changed')


def check_cell(run, spec, cfg, scope):
    checked = _check_base_cell(run, spec, cfg, scope)
    cell = run / spec['cell']
    result = read(cell / 'result.json')
    phases = {p: read(cell / p / 'phase_result.json') for p in shared.VIEWS}
    generation = phases['EARLY']['stage_generation']
    require(generation is not None, 'Successful staged cell has no complete paid bootstrap/generation')
    pool = read(cell / 'EARLY/candidate_pool.json')
    require(pool == {'generation': generation, 'generation_sha256': digest(generation)}, 'Cold candidate pool evidence differs')
    require('EARLY/candidate_pool.json' in shared.checked_manifest(cell), 'Pool was not sealed with the cell')
    frozen = next(r for r in read(run / 'stage_plan.json')['contexts'] if r['context_id'] == spec['context_id'])
    require(generation == frozen['generation'] and digest(generation) == frozen['generation_sha256'], 'Cold shared pool differs from prephysical graph cohort')
    require(result['stage_generation_sha256'] == digest(generation) and
        result['stage_candidate_count'] == len(generation['candidates']) and
        result['stage_generation_seconds'] == phases['EARLY']['stage_generation_seconds'], 'Generation count/cost hidden')
    energies = {}
    for name, phase in phases.items():
        best = phase['best_plan']
        require(phase['fully_certified'] is (best is not None), 'Complete phase quality status differs')
        if best is not None:
            require(best['certified_complete'] and best['validation']['status'] == 'COMPLETE', 'Uncertified phase incumbent')
        energies[name] = best['validation']['energy_kwh'] if best is not None else None
    checked.update(early_energy_kwh=energies['EARLY'], early_fully_certified=phases['EARLY']['fully_certified'],
        guard_energy_kwh=energies['GUARD'], candidate_pool_sha256=digest(generation),
        candidate_count=len(generation['candidates']), stage_generation_seconds=result['stage_generation_seconds'])
    return checked


def check_stage_controller(controller, schedule, checks, dispatch, stage):
    names = [s['cell'] for s in schedule]
    require(stage in ('phase1', 'phase2') and len(names) == len(set(names)) == len(checks)
        and {c['cell'] for c in checks} == set(names) and len(names) % 4 == 0, 'Partial/missing stage comparison population')
    status = 'COMPLETE_STAGE1_PENDING_AUDIT' if stage == 'phase1' else 'COMPLETE_STAGE2_PENDING_AUDIT'
    require(controller['status'] == status and controller['stage'] == stage and controller['scheduled'] == len(names)
        and controller['scheduled_total'] == 32 and controller['selected_cells'] == names
        and controller['dispatched'] == schedule and len(controller['completed']) == len(names)
        and set(controller['completed']) == set(names) and not controller['errors'] and not controller['unknown_cost_cells']
        and controller['peak_active'] <= 2, 'Stage controller not completely closed')
    require([d['cell'] for d in dispatch] == names and len(controller['exits']) == len(names), 'Retry/omitted dispatch/exit')
    exits = {e['cell']: e for e in controller['exits']}
    require(set(exits) == set(names), 'Duplicated or missing true exits')
    by_cell, previous_end = {c['cell']: c for c in checks}, 0.
    for start in range(0, len(names), 2):
        ends = []
        for spec, d in zip(schedule[start:start+2], dispatch[start:start+2]):
            e, c = exits[spec['cell']], by_cell[spec['cell']]
            require(d['stage'] == c['execution']['stage'] == stage and e['exit_code'] == 0
                and e['drained_after_controller_error'] is False and e['launcher_pid'] == d['launcher_pid']
                and e['pair_order'] == d['pair_order'] == spec['pair_order'], 'Wrong stage/failed process/changed order')
            begin = window.finite_time(e['started_epoch_seconds'], 'process start')
            end = window.finite_time(e['finished_epoch_seconds'], 'process exit')
            elapsed = window.finite_time(e['process_wall_seconds'], 'complete process elapsed')
            require(begin == d['started_epoch_seconds'] == c['execution']['started_epoch_seconds']
                and previous_end <= begin <= end and elapsed >= c['worker_finished_elapsed_seconds'], 'Pair not drained or time omitted')
            require(c['worker_parent_pid'] in (d['launcher_pid'], c['execution']['controller_pid'])
                and abs((end-begin)-elapsed) <= 2., 'Process ancestry or clocks differ')
            c['process_wall_seconds'] = elapsed
            ends.append(end)
        previous_end = max(ends)
    totals = {k: sum(c[k] for c in checks) for k in ('logical_points', 'physical_points', 'numerical_entries', 'requests')}
    require(all(controller[k] == v for k, v in totals.items()) and totals['physical_points'] <= 512*len(names)
        and totals['logical_points'] <= 512*len(names) and totals['numerical_entries'] == 3*len(names)+4*totals['physical_points'], 'Full stage cost does not close')
    require(window.finite_time(controller['wall_seconds'], 'controller stage time') >= max((c['process_wall_seconds'] for c in checks), default=0), 'Controller time omits workers')
    return totals


def context_equalities(checks):
    rows = []
    for context in sorted({c['context_id'] for c in checks}):
        group = [c for c in checks if c['context_id'] == context]
        require(len(group) == 4 and {c['arm_slot'] for c in group} == set(SLOTS), 'Missing rule or one of three fixed BC seeds')
        require(all(c['guard_established'] and c['history_verified'] for c in group), 'Complete original Beam guard missing')
        require(len({c['phase_summary']['GUARD']['semantic_signature_sha256'] for c in group}) == 1
            and len({c['guard_quality_pct'] for c in group}) == 1, 'GUARD scientific trajectory differs across arms')
        require(len({c['candidate_pool_sha256'] for c in group}) == 1
            and len({c['first_verify_state_sha256'] for c in group}) == 1, 'Fixed rankers received different candidate pools/history')
        rows.append({'context_id': context, 'status': 'PASS', 'tasks': 4,
            'guard_trajectories_equal': True, 'candidate_pool_sha256': group[0]['candidate_pool_sha256'],
            'common_staged_initial_exposure_verified': True, 'guard_quality_pct': group[0]['guard_quality_pct']})
    return rows


def source_hashes(run, scope):
    result = previous.source_hashes(run, scope)
    result.pop('source_shared_completion_sha256')
    result.update(source_deferred_completion_sha256=SOURCE_SHA, stage_plan_sha256=sha(run/'stage_plan.json'),
        graph_profile_manifest_sha256=sha(run/'graph_profile/completion_manifest.json'),
        audit_implementation_sha256=sha(Path(__file__)), core_audit_sha256=sha(Path(previous.__file__)))
    return result


def reuse_stage1(run, scope, plan):
    shared.checked_manifest(run/'audit_stage1')
    summary = read(run/'audit_stage1/summary.json')
    require(summary['status'] == 'PASS' and summary['stage'] == 'phase1' and not summary['errors']
        and summary['audited_cells'] == len(plan['phase1_cells']), 'Missing complete first-stage independent audit')
    for k, v in source_hashes(run, scope).items():
        require(summary[k] == v, 'Changed first-stage audit/source binding: '+k)
    require(summary['controller_stage1_completion_sha256'] == sha(run/'controller_stage1_completion.json'), 'Changed first-stage exits')
    checks = summary['results']
    require([r['cell'] for r in checks] == plan['phase1_cells'], 'Reused stage-one population changed')
    for row in checks:
        cell = run/row['cell']
        require(row['status'] == 'PASS' and sha(cell/'completion_manifest.json') == row['cell_completion_sha256']
            and sha(cell/'result.json') == row['result_sha256'], 'Audited cell bytes changed')
        # Revalidate sealed files, without repeating any numerical certificate
        # computations. A summary cannot excuse a changed paid-evidence file.
        shared.checked_manifest(cell)
    require(summary['gate'] == phase_gate(checks), 'Reused strict-gain decision changed')
    return summary, deepcopy(checks)


def check_final_disposition(run, schedule, checks, plan, first_summary):
    gate = phase_gate(first_summary['results'])
    expected_gate = {**gate, 'audit_stage1_manifest_sha256': sha(run/'audit_stage1/completion_manifest.json'),
        'audit_stage1_summary_sha256': sha(run/'audit_stage1/summary.json'),
        'freeze_sha256': sha(run/'prephysical_freeze_manifest.json'), 'stage_plan_sha256': sha(run/'stage_plan.json'),
        'controller_stage1_completion_sha256': sha(run/'controller_stage1_completion.json'),
        'phase1_cells': plan['phase1_cells'], 'phase2_cells': plan['phase2_cells']}
    require(read(run/'phase_gate.json') == expected_gate, 'Second-stage authorization changed')
    first = read(run/'controller_stage1_completion.json')
    stages = [first]
    not_run = []
    if gate['phase2_authorized']:
        stages.append(read(run/'controller_stage2_completion.json'))
        require({c['cell'] for c in checks} == {s['cell'] for s in schedule}, 'Authorized full comparison omitted tasks')
    else:
        require(not (run/'controller_stage2_started.json').exists(), 'Second stage started without certified gain')
        require([c['cell'] for c in checks] == plan['phase1_cells'], 'Unapproved actual task')
        for spec in schedule:
            if spec['cell'] not in plan['phase2_cells']: continue
            cell = run/spec['cell']
            expected = {'cell':spec['cell'],'context_id':spec['context_id'],'arm':spec['arm'],'seed':spec['seed'],
                'status':'NOT_RUN_GATE_NO_GAIN','executed':False,'quality_pct':None,'process_wall_seconds':None,
                'new_physical_points':0,'new_numerical_entries':0,'new_runtime_initializations':0}
            require(read(cell/'not_run.json') == expected and not any((cell/p).exists() for p in
                ('worker_started.json','execution.json','result.json','service_events.jsonl','physical/points.jsonl.gz')), 'Unrun slot has invented evidence/cost')
            not_run.append(expected)
    controller = read(run/'controller_completion.json')
    require(controller['status'] == ('COMPLETE_CELLS_PENDING_AUDIT' if gate['phase2_authorized'] else 'COMPLETE_PARTIAL_EXPLORATORY_NO_GAIN')
        and controller['scheduled'] == controller['resolved_slots'] == 32 and controller['actual_tasks'] == len(checks)
        and controller['not_run'] == not_run and not controller['errors'] and not controller['unknown_cost_cells']
        and controller['all_eight_context_mean_allowed'] is (len(checks) == 32), 'Final conditional disposition/counts invalid')
    for key in ('completed','dispatched','exits'):
        require(controller[key] == [v for stage in stages for v in stage[key]], 'Final stage aggregation changed: '+key)
    require(list(read_lines(run/'controller_dispatch.jsonl')) ==
        [v for n in range(1,len(stages)+1) for v in read_lines(run/f'controller_stage{n}_dispatch.jsonl')], 'Cumulative dispatch journal differs')
    for k in ('logical_points','physical_points','numerical_entries','requests'):
        require(controller[k] == sum(c[k] for c in checks), 'Final true cost differs: '+k)
    require(controller['stage1_controller_sha256'] == sha(run/'controller_stage1_completion.json')
        and controller['stage2_controller_sha256'] == (sha(run/'controller_stage2_completion.json') if len(stages)==2 else None)
        and controller['phase_gate_sha256'] == sha(run/'phase_gate.json')
        and controller['stage1_audit_seconds'] == first_summary['elapsed_seconds'], 'Interim audit time/source hidden')
    wall = window.finite_time(controller['wall_seconds'], 'whole conditional execution time')
    require(wall >= sum(stage['wall_seconds'] for stage in stages)+first_summary['elapsed_seconds'], 'Whole time omits first-stage audit or execution')
    if len(stages) == 2 and stages[1]['exits']:
        require(min(e['started_epoch_seconds'] for e in stages[1]['exits']) >= first_summary['audit_finished_epoch_seconds'], 'Second stage used unaudited future evidence')
    return not_run, controller


def audit(run, output=None, stage='final'):
    from audit_d059_supplement import NoSolverImports, no_solver_loaded
    run = Path(run).resolve()
    output = Path(output).resolve() if output else run/('audit_stage1' if stage=='phase1' else 'audit')
    require(stage in ('phase1','final') and output.is_relative_to(ROOT) and output != run and not output.exists(), 'Preserve prior independent audit evidence')
    output.mkdir(parents=True)
    started, epoch, checks = time.perf_counter(), time.time(), []
    result = {'status':'FAIL','stage':stage,'source_run':run.name,'scheduled_cells':32,'errors':[],
        'audited_cells':0,'results':[],'new_audit_physical_points':0,'new_training_tasks':0,
        'audit_started_epoch_seconds':epoch}
    guard = NoSolverImports();sys.meta_path.insert(0,guard)
    try:
        no_solver_loaded()
        cfg,scope,schedule = check_parent(run)
        result.update(source_hashes(run,scope))
        plan = read(run/'stage_plan.json')
        execution = read(run/'execution_started.json')
        require(execution['deadline_epoch_seconds'] == execution['started_epoch_seconds']+3600, 'Global deadline changed')
        if stage=='final':
            first_summary,checks = reuse_stage1(run,scope,plan)
            selected = plan['phase2_cells'] if first_summary['gate']['phase2_authorized'] else []
            result['reused_stage1_audit_sha256'] = sha(run/'audit_stage1/summary.json')
        else:
            selected = plan['phase1_cells']
        for spec in schedule:
            if spec['cell'] not in selected: continue
            checked = check_cell(run,spec,cfg,scope);checks.append(checked)
            save(output/(spec['cell'].replace('/','_')+'.json'),checked)
            save(output/'progress.json',{'status':'RUNNING','audited_cells':len(checks),'scheduled_cells':32,
                'last_cell':spec['cell'],'elapsed_seconds':time.perf_counter()-started,'new_physical_points':0})
        stages = [1] if stage=='phase1' else [1,2] if selected else [1]
        for index in stages:
            names=plan[f'phase{index}_cells']; part=[s for s in schedule if s['cell'] in names]
            c=read(run/f'controller_stage{index}_completion.json')
            require(c['exits']==read(run/f'controller_stage{index}_exits.json'), 'Exit journal changed')
            check_stage_controller(c,part,[r for r in checks if r['cell'] in names],
                list(read_lines(run/f'controller_stage{index}_dispatch.jsonl')),f'phase{index}')
        equalities=context_equalities(checks)
        if stage=='phase1':
            result.update(gate=phase_gate(checks),controller_stage1_completion_sha256=sha(run/'controller_stage1_completion.json'),
                phase1_cells=plan['phase1_cells'],phase2_cells=plan['phase2_cells'])
        else:
            not_run,controller=check_final_disposition(run,schedule,checks,plan,first_summary)
            result.update(resolved_slots=32,actual_tasks=len(checks),not_run=not_run,
                all_eight_context_mean_allowed=len(checks)==32,gate=first_summary['gate'],
                controller_completion_sha256=sha(run/'controller_completion.json'),
                controller_wall_seconds=controller['wall_seconds'],source_audit_seconds=first_summary['elapsed_seconds'])
        for row in checks:save(output/(row['cell'].replace('/','_')+'.json'),row)
        statuses=Counter()
        for row in checks:statuses.update(row['physical_statuses'])
        result.update(status='PASS',audited_cells=len(checks),results=checks,context_equalities=equalities,
            complete_protected_cells=len(checks),physical_runtimes=len(checks),logical_searches=2*len(checks),
            physical_statuses=dict(statuses),**{k:sum(c[k] for c in checks) for k in ('logical_points','physical_points','numerical_entries','requests')})
        no_solver_loaded()
    except BaseException as exc:
        result.update(audited_cells=len(checks),results=checks)
        result['errors'].append({'error':repr(exc),'traceback':traceback.format_exc()})
    finally:sys.meta_path.remove(guard)
    result.update(elapsed_seconds=time.perf_counter()-started,audit_finished_epoch_seconds=time.time())
    save(output/'summary.json',result)
    save(output/'completion_manifest.json',{p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file() and p.name!='completion_manifest.json'})
    return result


def preflight(run, output):
    from audit_d059_supplement import NoSolverImports, no_solver_loaded
    run,output=Path(run).resolve(),Path(output).resolve()
    require(output.is_relative_to(ROOT) and not output.exists(), 'Preserve previous preflight evidence')
    output.mkdir(parents=True)
    result={'status':'FAIL','errors':[],'new_physical_points':0,'new_training':0}
    guard=NoSolverImports();sys.meta_path.insert(0,guard)
    try:
        no_solver_loaded();_,scope,schedule=check_parent(run);no_solver_loaded()
        plan=read(run/'stage_plan.json')
        result.update(status='PASS',contexts=8,scheduled_cells=len(schedule),phase1_cells=plan['phase1_cells'],
            phase2_cells=plan['phase2_cells'],physical_runtimes_upper_bound=32,logical_searches_upper_bound=64,
            fixed_BC_models=3,**source_hashes(run,scope))
    except BaseException as exc:result['errors'].append({'error':repr(exc),'traceback':traceback.format_exc()})
    finally:sys.meta_path.remove(guard)
    save(output/'summary.json',result);save(output/'completion_manifest.json',{'summary.json':sha(output/'summary.json')})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path)
    parser.add_argument('--stage',choices=('phase1','final'),default='final');parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    if args.preflight:
        require(args.output is not None,'Separate preflight output required');result=preflight(args.run,args.output)
    else:result=audit(args.run,args.output,args.stage)
    print(canonical({k:v for k,v in result.items() if k not in ('results','context_equalities')}),flush=True)
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
