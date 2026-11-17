"""Independent, solver-free audit of early fixed policies and a separate Beam.

Production search, policy loaders and the new execution runner are not imported.
Compact observations are bound exactly to their recorded payload, reconstructed
with the original independent feature tolerances, and evaluated both ways.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import math
from pathlib import Path
import sys
import time
import traceback
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
sys.dont_write_bytecode = True

import audit_clpu_integrated_pilot as prior

cache, branch, window, tail, old = prior.cache, prior.branch, prior.window, prior.tail, prior.old
learning = prior.learning
require, close, same = prior.require, prior.close, prior.same
read, save, sha, digest, manifest, read_lines = prior.read, prior.save, prior.sha, prior.digest, prior.manifest, prior.read_lines
SOURCE_RUN = '20261030T141744_232986Z_clpu_integrated_pilot'
SOURCE_SHA = '461826a8e3f9c415821d42b47c81d66acd8c2fa2d01b09f388dd102e9938ba62'
STUDY = 'CLPU_EARLY_PROTECTED_PILOT_V1'
SEEDS = (17, 41, 73)
EARLY_CHOICES = 16
COMPACT_FIELDS = ('representation', 'candidate_features', 'global_features', 'mask', 'branch_ids')


def compact_observation(observation):
    return {key: deepcopy(observation[key]) for key in COMPACT_FIELDS}


def expected_method(phase, choice, method):
    require(phase in ('EARLY', 'GUARD') and type(choice) is int and choice >= 0,
            'Invalid phase/choice')
    require(method in ('beam', 'bc_raw', 'ppo_raw'), 'Unapproved fixed selector')
    return method if phase == 'EARLY' and choice < EARLY_CHOICES else 'beam'


class IndependentPolicy:
    """Load a frozen actor once; never import its production implementation."""

    def __init__(self, checkpoint, spec):
        self.spec = spec
        checkpoint = Path(checkpoint)
        require(sha(checkpoint) == spec['checkpoint_sha256'], 'Fixed policy SHA changed')
        self.model, saved = learning.load_independent(checkpoint)
        require(saved['seed'] == spec['seed'] and saved['kind'] == 'actor_critic'
                and saved['representation'] == spec['representation'] == 'raw'
                and spec['deterministic'] is True, 'Unapproved fixed policy identity')
        learning.torch.set_num_threads(1)

    def evaluate(self, compact):
        observation = {'representation': 'raw', **compact}
        x, g, mask = learning.tensorize([observation], 'raw')
        with learning.torch.no_grad():
            logits, value = self.model(x, g, mask)
            action = logits.argmax(dim=-1)
            distribution = learning.torch.distributions.Categorical(logits=logits)
        selected = int(action[0])
        require(bool(mask[0, selected]), 'Policy selected a masked action')
        return selected, float(distribution.log_prob(action)[0]), float(value[0])


def check_observation_action(row, replay, phase, choice, spec, policy=None):
    """Current-only observations and both actual/reconstructed masked actions."""
    expected = compact_observation(replay.observation('raw'))
    actual = row['obs_compact']
    require(set(actual) == set(COMPACT_FIELDS), 'Unexpected policy input fields')
    require(row['obs_compact_sha256'] == digest(actual), 'Recorded compact payload hash differs')
    require(row['mask_sha256'] == digest(actual['mask']), 'Recorded mask hash differs')
    same(actual, expected, 'independent_current_compact_observation')
    require(all(type(v) is bool for v in actual['mask'])
            and actual['mask'] == [True] * len(replay.window()), 'Current action mask differs')
    method = expected_method(phase, choice, spec['method'])
    require(row['method'] == method, 'Early sixteen-choice selector boundary changed')
    selected = row['action']
    require(type(selected) is int and 0 <= selected < len(actual['branch_ids'])
            and actual['mask'][selected]
            and row['branch_id'] == actual['branch_ids'][selected], 'Unexposed/masked branch selected')
    if method == 'beam':
        require(selected == replay.rule('beam') and row['checkpoint_sha256'] is None
                and row['logprob'] is None and row['value'] is None, 'Original Beam selection changed')
    else:
        require(policy is not None and row['checkpoint_sha256'] == spec['checkpoint_sha256'],
                'Missing or changed fixed policy')
        observed = policy.evaluate(actual)
        rebuilt = policy.evaluate(expected)
        require(selected == observed[0] == rebuilt[0], 'Actual/reconstructed policy argmax differs')
        for label, value in (('recorded', observed), ('independent', rebuilt)):
            learning.close(row['logprob'], value[1], label + ' policy log probability')
            learning.close(row['value'], value[2], label + ' policy value')
    return {'method': method, 'policy_decision': method != 'beam',
            'recorded_compact_sha256': digest(actual),
            'independent_compact_sha256': digest(expected),
            'compact_exact_hash_equal': digest(actual) == digest(expected)}


def phase_counters(oracle, logical_rows):
    physical = cache.actual_prefix_count(logical_rows, oracle.used)
    return {'logical_points': oracle.used, 'physical_points': physical,
            'numerical_entries': 3 + 4 * physical, 'requests': oracle.counts()['requests']}


def check_phase_records(mech, branch_cfg, phase, spec, result, attempts, decisions,
                        request_rows, points, logical_rows, condition_sha, policy=None):
    """Replay either cold search with its own empty logical/request namespace."""
    require(phase in ('EARLY', 'GUARD') and spec['search_phase'] == phase,
            'Phase identity changed')
    oracle = branch.PaidRequests(request_rows, points, mech, branch_cfg['point_budget'], condition_sha)
    replay_class = window.WindowReplay if phase == 'EARLY' else branch.BranchReplay
    replay = replay_class(mech, branch_cfg, oracle, attempts)
    require(len(decisions) % 2 == 0, 'Unclosed compact decision')
    boundaries, policy_checks = [], []

    def boundary(choice):
        boundaries.append({'phase_choices': choice, **phase_counters(oracle, logical_rows),
            'quality_pct': 100 * replay.quality if replay.best is not None else None,
            'fully_certified': replay.best is not None})

    boundary(0)
    identity_fields = ('transition_id', 'phase_choice', 'phase', 'method', 'action', 'branch_id',
        'obs_raw_sha256', 'obs_compact_sha256', 'mask_sha256', 'checkpoint_sha256',
        'logprob', 'value', 'oracle_before')
    for choice in range(len(decisions) // 2):
        started, completed = decisions[2 * choice:2 * choice + 2]
        require(not replay.done and replay.turn == choice, 'Decision after termination or turn reset')
        require(set(started) == set(identity_fields) | {'status', 'obs_compact'}
                and set(completed) == set(identity_fields) | {'status', 'response', 'oracle_after'},
                'Compact decision schema changed or hidden inputs added')
        require(started['status'] == 'STARTED' and completed['status'] == 'COMPLETED'
                and started['phase_choice'] == started['transition_id'] == choice
                and started['phase'] == phase, 'Compact decision chronology changed')
        for key in identity_fields:
            require(started[key] == completed[key], 'Decision STARTED/COMPLETED identity differs: ' + key)
        require(isinstance(started['obs_raw_sha256'], str) and len(started['obs_raw_sha256']) == 64
                and all(c in '0123456789abcdef' for c in started['obs_raw_sha256']),
                'Invalid full observation evidence digest')
        same(started['oracle_before'], phase_counters(oracle, logical_rows), 'phase_before_cost')
        policy_checks.append(check_observation_action(started, replay, phase, choice, spec, policy))
        response = replay.step(started['action'])
        same(completed['response'], response, 'independent_phase_response')
        same(completed['oracle_after'], phase_counters(oracle, logical_rows), 'phase_after_cost')
        boundary(choice + 1)
    require(replay.done and oracle.index == len(oracle.rows) and oracle.used == len(points)
            and replay.attempt_index == len(attempts), 'Unfinished phase or unbound causal evidence')
    require(result['logical_points'] == len(points) and result['actual_physical_points'] ==
            cache.actual_prefix_count(logical_rows, len(points)), 'Phase dual totals differ')
    normalized = {**result, 'actual_physical_points': result['logical_points']}
    scientific = tail.baseline_snapshot(replay)
    for key, value in scientific.items():
        same(normalized[key], value, 'complete_independent_phase.' + key)
    require(result['search_decisions'] == replay.turn == len(decisions) // 2
            and replay.turn <= branch_cfg['max_branch_expansions'], 'Phase decision allowance changed')
    policy_count = sum(row['policy_decision'] for row in policy_checks)
    require(result['policy_decisions'] == policy_count
            and policy_count == (min(EARLY_CHOICES, replay.turn)
                if phase == 'EARLY' and spec['method'] != 'beam' else 0), 'Policy participation miscounted')
    require(result['fully_certified'] is (replay.best is not None)
            and result['quality_pct'] == boundaries[-1]['quality_pct'], 'Phase certified quality misreported')
    return {'phase': phase, 'decisions': replay.turn, 'policy_decisions': policy_count,
            'quality_pct': boundaries[-1]['quality_pct'], 'fully_certified': replay.best is not None,
            'stop_reason': replay.reason, 'boundaries': boundaries,
            'compact_hash_exact_matches': sum(p['compact_exact_hash_equal'] for p in policy_checks),
            'compact_hash_numerical_reconstruction_differences': sum(not p['compact_exact_hash_equal'] for p in policy_checks),
            'actual_and_independent_policy_actions_equal': True,
            'semantic_signature_sha256': digest(scientific), 'scientific_result': scientific}


def complete_plan(result):
    if not result or result.get('done') is not True:
        return None
    plan = result.get('best_plan')
    if result.get('status') != 'CERTIFIED_INCUMBENT':
        require(plan is None, 'Uncertified result carried a complete plan')
        return None
    require(isinstance(plan, dict), 'Certified result omitted its complete plan')
    require(plan.get('certified_complete') is True and plan.get('status') == 'COMPLETE'
            and plan.get('validation', {}).get('status') == 'COMPLETE', 'Incomplete claimed final certificate')
    energy = plan['validation']['energy_kwh']
    require(type(energy) in (int, float) and math.isfinite(energy) and energy >= 0,
            'Invalid certified energy')
    return plan


def protected_choice(early, guard, protected):
    """Exact energy comparison; a tie always selects the completed GUARD."""
    require(early.get('done') is True, 'Early search not normally completed')
    first = complete_plan(early)
    if not protected:
        require(guard is None, 'Direct reference secretly received a Beam result')
        return {'selected_phase': 'EARLY', 'best_plan': first,
                'guard_established': False, 'gain_over_guard_kwh': None}
    require(guard is not None and guard.get('done') is True, 'Independent Beam guard not completed')
    baseline = complete_plan(guard)
    if baseline is None:
        return {'selected_phase': 'GUARD', 'best_plan': None,
                'guard_established': False, 'gain_over_guard_kwh': None}
    pick_first = first is not None and first['validation']['energy_kwh'] > baseline['validation']['energy_kwh']
    plan = first if pick_first else baseline
    return {'selected_phase': 'EARLY' if pick_first else 'GUARD', 'best_plan': plan,
            'guard_established': True,
            'gain_over_guard_kwh': plan['validation']['energy_kwh'] - baseline['validation']['energy_kwh']}


def protocol_settings():
    return {'phase_logical_point_budget': 256, 'phase_physical_point_budget': 256,
        'phase_expansion_cap': 256, 'early_policy_expansions': 16,
        'protected_total_point_budget': 512, 'reference_total_point_budget': 256,
        'max_selected_decisions': 512, 'cold_searches': 120, 'max_workers': 2,
        'cooperative_seconds': 14400, 'worker_seconds': 3600,
        'total_physical_budget': 30720, 'total_numerical_entry_upper_bound': 123240,
        'minimum_free_disk_bytes': 12 * 2**30, 'forecast_margin': 1.25,
        'report_source_reserve_bytes': 2 * 2**30, 'cache_variant': 'electrical_equivalent',
        'no_training': True, 'no_historical_answers': True, 'no_cross_cell_cache': True,
        'no_cross_phase_cache': True, 'no_automatic_retry': True,
        'phase_order': ['EARLY', 'GUARD'], 'exact_energy_tie': 'GUARD',
        'phase_process_model': 'one_fresh_exec_child_per_phase; parent never initializes a solver; children run serially',
        'maximum_concurrent_physics_processes': 2,
        'recording': 'compact_raw_features_once_per_started_decision; full_search_result_once_per_phase',
        'dispatch': 'Hashed context and initial slot order; Latin rotation; adjacent two-worker pairs drain before next pair and context',
        'timing': 'Same-run cold full-process and worker/phase timing; no old-run speed comparator',
        'aggregation': 'Average seeds within context before averaging eight contexts'}


def expected_schedule(contexts, models):
    require(len(contexts) == 8 and {r['context_id'] for r in contexts} == set(range(8)), 'Eight fixed contexts required')
    require(len({r['scenario_id'] for r in contexts}) == 8
            and Counter(r['event_id'] for r in contexts) == Counter({30: 2, 20: 2, 6: 2, 3: 2}),
            'Fixed training context assignment changed')
    contexts = [{k: r[k] for k in ('context_id', 'scenario_id', 'event_id')} for r in contexts]
    ordered = sorted(contexts, key=lambda r: prior.digest_text('clpu-early-context-order-v1|' + prior.canonical(r)))
    slots = {(arm if seed == 0 else arm + str(seed)): (arm, seed) for arm, seed in prior.ARM_SEEDS}
    order = sorted(slots, key=lambda slot: prior.digest_text('clpu-early-arm-order-v1|' + slot))
    model = {(r['method'], r['seed']): r for r in models}
    require(len(model) == len(models) == 6
            and set(model) == {(m, s) for m in ('bc_raw', 'ppo_raw') for s in SEEDS}, 'Six fixed models required')
    rows = []
    for rank, context in enumerate(ordered):
        for position in range(8):
            slot = order[(position + rank) % 8]
            arm, seed = slots[slot]
            method = 'bc_raw' if arm == 'protected_bc' else 'ppo_raw' if arm == 'protected_ppo' else 'beam'
            binding = model.get((method, seed))
            protected = arm != 'window_reference'
            rows.append({**context, 'arm': arm, 'arm_slot': slot, 'seed': seed, 'method': method,
                'phase': 'evaluation', 'group': 'early_intervention', 'representation': 'raw', 'deterministic': True,
                'point_budget': 256, 'logical_point_budget': 256, 'physical_point_budget': 512 if protected else 256,
                'total_logical_point_budget': 512 if protected else 256,
                'early_policy_expansions': 16, 'cold_searches': 2 if protected else 1,
                'window_variant': 'coverage_v1', 'protected': protected, 'cache_variant': 'electrical_equivalent',
                'context_order': rank, 'arm_position': position, 'pair_order': rank * 4 + position // 2,
                'episode_id': len(rows), 'cell': f'episodes/{len(rows):03d}',
                'checkpoint': binding['checkpoint'] if binding else None,
                'checkpoint_sha256': binding['source_sha256'] if binding else None})
    return rows


def expected_phase_spec(spec, phase):
    require(phase in ('EARLY', 'GUARD') and (phase != 'GUARD' or spec['protected']), 'Forbidden cold phase')
    return {**spec, 'parent_cell': spec['cell'], 'parent_arm': spec['arm'], 'parent_seed': spec['seed'],
        'cell': spec['cell'] + '/' + phase.lower(), 'search_phase': phase,
        'method': spec['method'] if phase == 'EARLY' else 'beam',
        'seed': spec['seed'] if phase == 'EARLY' else 0,
        'checkpoint': spec['checkpoint'] if phase == 'EARLY' else None,
        'checkpoint_sha256': spec['checkpoint_sha256'] if phase == 'EARLY' else None,
        'point_budget': 256, 'logical_point_budget': 256, 'physical_point_budget': 256,
        'total_logical_point_budget': 256, 'cold_searches': 1}


def check_parent(run):
    run = Path(run).resolve()
    require(run.parent == ROOT / 'runs', 'Early pilot must be a project run')
    prepared = read(run / 'prepared.json')
    require(prepared['status'] == 'FROZEN_PREPARED_NO_DSS'
            and prepared['freeze_sha256'] == sha(run / 'prephysical_freeze_manifest.json'), 'Prephysical freeze changed')
    manifest(run, 'prephysical_freeze_manifest.json')
    required = {'scripts/audit_clpu_early_pilot.py', 'scripts/run_clpu_early_pilot.py',
        'scripts/report_clpu_early_pilot.py', 'src/dnr/nature34_clpu_early.py',
        'scripts/audit_clpu_integrated_pilot.py', 'scripts/audit_clpu_equivalent_cache.py',
        'scripts/audit_clpu_window_pilot.py', 'scripts/audit_nature34_clpu_branch_learning.py',
        'docs/clpu_early_pilot_protocol_v1.md', 'tests/test_clpu_early_audit.py'}
    require(required <= set(prepared['source_files'])
            and len(set(prepared['source_files'])) == len(prepared['source_files']), 'Required sources not frozen')
    for name in prepared['source_files']:
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT) and path != ROOT
                and sha(path) == sha(run / 'source' / name), 'Current frozen source changed: ' + name)
    sources = (('integrated', SOURCE_RUN, SOURCE_SHA), ('cache', prior.CACHE_RUN, prior.CACHE_SHA),
        ('window', cache.SOURCE_RUN, cache.SOURCE_SHA), ('tail', prior.TAIL_RUN, prior.TAIL_SHA),
        ('d059', prior.LEARNING_RUN, prior.LEARNING_SHA), ('d060', prior.D060_RUN, prior.D060_SHA))
    expected_binding = {key: value for label, origin, value in sources
        for key, value in ((label + '_run', origin), (label + '_completion_sha256', value))}
    require(read(run / 'source_binding.json') == expected_binding, 'Source chain changed')
    for label, origin, expected in sources:
        source = ROOT / 'runs' / origin
        require(sha(source / 'completion_manifest.json') == expected
                and sha(run / 'provenance' / label / 'source_completion_manifest.json') == expected,
                'Copied/original source seal changed: ' + label)
        sealed = {k.replace('\\', '/'): v for k, v in read(source / 'completion_manifest.json').items()}
        for name in prepared['source_files']:
            if 'source/' + name in sealed:
                require(sha(ROOT / name) == sealed['source/' + name], 'Inherited scientific source changed: ' + name)
        for name in ('model_bindings.json', 'source_binding.json', 'terminal_status.json'):
            if name in sealed:
                require(sha(run / 'provenance' / label / name) == sealed[name], 'Copied source attribution changed')
    source = ROOT / 'runs' / SOURCE_RUN
    source_seal = {k.replace('\\', '/'): v for k, v in read(source / 'completion_manifest.json').items()}
    for name in ('config.json', 'model_bindings.json', 'source_cells.json', 'audit/summary.json'):
        require(sha(source / name) == source_seal[name], 'Integrated source input changed: ' + name)
    source_audit = read(source / 'audit/summary.json')
    require(source_audit['status'] == 'PASS' and source_audit['audited_cells'] == 64
            and source_audit['errors'] == [], 'Integrated source not fully audited')
    models, _, _ = prior.model_sources()
    require(read(run / 'model_bindings.json') == read(source / 'model_bindings.json') == models,
            'Original six model endpoints changed')
    for model in models:
        require(sha(run / model['checkpoint']) == source_seal[model['checkpoint']]
                == sha(source / model['checkpoint']) == model['source_sha256'], 'Fixed model bytes differ')
    prior.check_model_preflight(run, models)
    check_early_diagnostic(run)
    cfg = read(run / 'config.json')
    same(cfg, {**read(source / 'config.json'), 'early_study': STUDY, 'early_pilot': protocol_settings()},
         'entire_early_experimental_contract')
    scope = prior.check_frozen_physics(run, prepared, cfg)
    contexts = [{k: c[k] for k in ('context_id', 'scenario_id', 'event_id')}
                for c in read(source / 'source_cells.json')]
    schedule = read(run / 'schedule.json')
    same(schedule, expected_schedule(contexts, models), 'all_64_early_Latin_scheduled_cells')
    for spec in schedule:
        require(read(run / spec['cell'] / 'spec.json') == spec, 'Prepared cell spec differs')
        for phase in (('EARLY', 'GUARD') if spec['protected'] else ('EARLY',)):
            expected = expected_phase_spec(spec, phase)
            require(read(run / expected['cell'] / 'spec.json') == expected, 'Prepared independent phase spec differs')
    require(sha(run / 'historical/source_completion_manifest.json') == cache.SOURCE_SHA
            and sha(run / 'historical/source_audit_summary.json') ==
            read(ROOT / 'runs' / cache.SOURCE_RUN / 'completion_manifest.json')['audit/summary.json'],
            'Original Beam/window historical audit changed')
    prior.check_historical_bindings(run, contexts)
    cases = list(ET.parse(run / 'software_tests.xml').getroot().iter('testcase'))
    require(cases and not any(list(c.iter('failure')) + list(c.iter('error')) + list(c.iter('skipped')) for c in cases),
            'Nonempty all-passed frozen software tests required')
    for label in ('test_run_clpu_early_pilot', 'test_report_clpu_early_pilot',
                  'test_nature34_clpu_early', 'test_clpu_early_audit'):
        require(any(label in c.get('classname', '') for c in cases), 'Missing frozen component tests: ' + label)
    budget = read(run / 'resource_budget.json')
    for key, value in {'contexts': 8, 'cells': 64, 'physical_points': 30720, 'numerical_entries': 123240,
        'cold_searches': 120, 'workers': 2, 'minimum_free_bytes': 12 * 2**30,
        'cooperative_seconds': 14400, 'worker_seconds': 3600, 'new_training': 0,
        'software_tests': len(cases), 'forecast_margin': 1.25, 'report_source_reserve_bytes': 2 * 2**30,
        'historical_cache_cells': 16, 'historical_training_tail_cells': 32,
        'forecast_is_hard_upper_bound': False}.items():
        require(budget[key] == value, 'Early resource contract changed: ' + key)
    sizes = []
    for origin, cohort, n in ((prior.CACHE_RUN, None, 16), (prior.TAIL_RUN, 'training_mechanism', 32)):
        source = ROOT / 'runs' / origin
        specs = [s for s in read(source / 'schedule.json') if cohort is None or s['cohort'] == cohort]
        require(len(specs) == n, 'Resource estimate cohort changed')
        sizes.append(max(sum(p.stat().st_size for p in (source / s['cell']).rglob('*') if p.is_file()) for s in specs))
    require(budget['cache_max_cell_bytes'] == sizes[0] and budget['tail_training_max_cell_bytes'] == sizes[1]
            and budget['historical_max_cell_bytes'] == 2 * max(sizes)
            and budget['initial_required_free_bytes'] == math.ceil(64 * 2 * max(sizes) * 1.25) + 14 * 2**30
            and budget['free_bytes_at_prepare'] >= budget['initial_required_free_bytes'], 'Two-route storage reserve changed')
    require(sha(run / 'protocol.md') == sha(ROOT / 'docs/clpu_early_pilot_protocol_v1.md'), 'Frozen prose protocol changed')
    return cfg, scope, schedule


def check_early_diagnostic(run):
    binding = read(run / 'early_diagnostic_binding.json')
    directory = run / 'early_diagnostic'
    expected_sha = '73a44b3306c5be53a44cb2b0c983dfa48a788a4ad2093fdd20ee61c404fda605'
    require(binding == {'path': 'artifacts/clpu_early_opportunities_20261031',
        'completion_sha256': expected_sha, 'summary_sha256': sha(directory / 'summary.json'),
        'usage': 'Opportunity description only; no filtering of eight contexts, six models, K, actions or budgets'},
        'Early diagnostic binding changed')
    require(sha(directory / 'completion_manifest.json') == expected_sha, 'Early diagnostic seal changed')
    sealed = manifest(directory)
    require({'summary.json', 'all_slots.json', 'contexts.json', 'source_bindings.json'} <= set(sealed),
            'Incomplete causal opportunity diagnostic')
    evidence = read(directory / 'summary.json')
    require(evidence['status'] == 'COMPLETE_CAUSAL_PREFIX_DIAGNOSTIC'
            and evidence['source_run'] == SOURCE_RUN and evidence['source_completion_sha256'] == SOURCE_SHA
            and evidence['contexts'] == 8 and evidence['slots'] == 128
            and evidence['available_slots'] + evidence['unavailable_slots'] == 128
            and evidence['new_physical_points'] == evidence['new_training'] == evidence['optimizer_updates']
            == evidence['alternative_actions_executed'] == evidence['independent_model_argmax_mismatch_positions'] == 0
            and evidence['formal_or_tune_or_holdout_access'] is False, 'Diagnostic scope or causal model evidence changed')


def check_quality_costs(checkpoints, phase_checks, result):
    """Every constructor, completed decision and close has a paid causal boundary."""
    expected = []
    total = {k: 0 for k in ('logical_points', 'physical_points', 'numerical_entries', 'requests')}
    prior_choices, best = 0, None
    for phase, checked in phase_checks.items():
        boundaries = checked['boundaries']
        expanded = [('INITIAL' if i == 0 else 'DECISION', b) for i, b in enumerate(boundaries)]
        expanded.append(('CLOSE', boundaries[-1]))
        for kind, local in expanded:
            qualities = [v for v in (best, local['quality_pct']) if v is not None]
            expected.append({'checkpoint_index': len(expected), 'phase': phase, 'kind': kind,
                'phase_choices': local['phase_choices'], 'search_choices': prior_choices + local['phase_choices'],
                **{key: total[key] + local[key] for key in total},
                'phase_logical_points': local['logical_points'], 'phase_physical_points': local['physical_points'],
                'quality_pct': max(qualities) if qualities else None, 'fully_certified': bool(qualities),
                'guard_established': phase == 'GUARD' and kind == 'CLOSE' and local['fully_certified']})
        best = expected[-1]['quality_pct']
        total = {key: expected[-1][key] for key in total}
        prior_choices = expected[-1]['search_choices']
    require(len(checkpoints) == len(expected), 'Missing initialization/decision/close checkpoint')
    prior_worker = prior_elapsed = 0.
    for row, wanted in zip(checkpoints, expected):
        require(set(row) == set(wanted) | {'worker_elapsed_seconds', 'elapsed_seconds'}, 'Cost checkpoint schema differs')
        for key, value in wanted.items():
            same(row[key], value, 'causal_full_cost_checkpoint.' + key)
        worker = window.finite_time(row['worker_elapsed_seconds'], 'checkpoint worker time')
        elapsed = window.finite_time(row['elapsed_seconds'], 'checkpoint computation time')
        require(prior_worker <= worker <= result['elapsed_seconds']
                and prior_elapsed <= elapsed <= result['computation_elapsed_seconds'], 'Checkpoint clock reversed or exceeds worker')
        clock_tolerance = 4 * max(math.ulp(worker), math.ulp(elapsed), math.ulp(result['preflight_elapsed_seconds']))
        require(worker + clock_tolerance >= elapsed + result['preflight_elapsed_seconds'],
                'Checkpoint omits preflight/model cost')
        prior_worker, prior_elapsed = worker, elapsed
    require(result['final_real_cost'] == checkpoints[-1], 'Final real cost differs from last closed phase')
    early = phase_checks['EARLY']
    expected_choice = min(16, early['decisions'])
    block = [r for r in checkpoints if r['phase'] == 'EARLY'
             and r['phase_choices'] == expected_choice and r['kind'] != 'CLOSE']
    require(len(block) == 1 and result['early_block_checkpoint'] == block[0], 'Early block boundary moved or omitted')
    return expected


def phase_manifest(cell, parent_seal, phase, summary):
    directory = cell / phase.lower()
    require(summary['directory'] == phase.lower()
            and sha(directory / 'completion_manifest.json') == summary['completion_manifest_sha256'], 'Phase seal binding changed')
    raw = read(directory / 'completion_manifest.json')
    require(isinstance(raw, dict) and raw, 'Empty phase seal')
    normalized, targets = {}, set()
    for name, value in raw.items():
        key = name.replace('\\', '/')
        target = (directory / key).resolve()
        require(target.is_relative_to(directory) and target != directory and target not in targets
                and key not in normalized, 'Phase seal alias or escaped path')
        require(parent_seal.get(phase.lower() + '/' + key) == value, 'Phase file not bound to parent cell seal')
        normalized[key] = value
        targets.add(target)
    needed = {'spec.json', 'initial_state.json', 'physical_identity.json', 'phase_started.json', 'phase_finished.json',
        'phase_execution.json', 'preflight_boundary.json', 'cost_checkpoints.jsonl',
        'phase_result.json', 'plans.jsonl.gz', 'decisions.jsonl.gz', 'points.jsonl.gz', 'requests.jsonl',
        'logical_points.jsonl.gz', 'logical_requests.jsonl', 'oracle_summary.json', 'equivalent_cache_summary.json',
        'metadata.json', 'command_ledger.jsonl', 'command_stats.json'}
    require(needed <= set(normalized) and not (directory / 'phase_error.json').exists(), 'Missing/failed cold phase evidence')
    require(sha(directory / 'phase_result.json') == summary['phase_result_sha256'], 'Phase result not bound')
    return directory


def check_historical_phase(run, spec, phase, checked, request_rows):
    """Guard equals original Beam; rule EARLY equals original coverage window."""
    if phase == 'GUARD':
        core = checked['scientific_result']
        oracle_boundary = {key: checked['boundaries'][-1][field]
                           for key, field in (('points', 'logical_points'), ('requests', 'requests'))}
        prior.check_historical_baseline(prior.historical_directory(run, spec['context_id'], 'g0'),
                                       core, {'oracle_boundary': oracle_boundary}, request_rows)
        return True
    if spec['method'] != 'beam':
        return False
    historical = prior.historical_semantics(prior.historical_directory(run, spec['context_id'], 'g1'))
    core = checked['scientific_result']
    for key in ('status', 'config', 'stop_reason', 'best_plan', 'initial_certificate'):
        same(core[key], historical['result'][key], 'historical_complete_window.' + key)
    same(core['attempts'], historical['plans'], 'historical_complete_window.attempts')
    same(core['trace'], [r['response'] for r in historical['steps'] if r['status'] == 'COMPLETED'],
         'historical_complete_window.trace')
    same(request_rows, historical['requests'], 'historical_complete_window.requests')
    require(checked['decisions'] == historical['result']['completed_expansions']
            == historical['result']['search_decisions'], 'Original window expansion count changed')
    expected_counts = {'generated_candidates': core['generated_unique_candidates'],
        'graph_assessed_candidates': core['assessed_graph_candidates'], 'checked_candidates': core['checked_candidates'],
        'branch_expansions': checked['decisions'], 'generation_cap_rejections': core['generation_cap_rejections']}
    same(expected_counts, historical['result']['counts'], 'historical_complete_window.counts')
    return True


def check_result_selection(result, phases):
    chosen = protected_choice(phases['EARLY'], phases.get('GUARD'), result['protected'])
    require(result['version'] == 'early16-window-independent-beam-guard-v1'
            and result['schema'] == 'clpu-early-protected-pilot-result-v1'
            and result['intermediate_beam_floor_claimed'] is False, 'Output protection version or intermediate claim changed')
    require(result['selected_phase'] == chosen['selected_phase']
            and result['best_plan'] == chosen['best_plan'], 'Final selection violates exact energy/guard tie rule')
    selected = phases[chosen['selected_phase']]
    require(result['status'] == selected['status'] and result['fully_certified'] == selected['fully_certified']
            and result['quality_pct'] == selected['quality_pct'], 'Final certified output misreported')
    baseline = phases.get('GUARD')
    require(result['early_quality_pct'] == phases['EARLY']['quality_pct']
            and result['baseline_quality_pct'] == (baseline['quality_pct'] if baseline else None)
            and result['gain_over_paid_baseline_kwh'] == chosen['gain_over_guard_kwh'], 'Reported gain differs from certificates')
    if not result['protected']:
        require(result['guard_audit'] is None, 'Reference invents a protected output')
    else:
        before = complete_plan(baseline)
        expected = {'guard_established': chosen['guard_established'], 'baseline_status': baseline['status'],
            'baseline_energy_kwh': before['validation']['energy_kwh'] if before else None,
            'final_energy_kwh': chosen['best_plan']['validation']['energy_kwh'] if before else None,
            'nondegradation_verified': chosen['guard_established'], 'tie_rule': 'exact_equal_energy_select_guard',
            'scope': 'Fixed 30-minute normal-service objective; computation waiting excluded; final completed output only'}
        require(result['guard_audit'] == expected, 'Guard claim differs from completed independent certificate')
    return chosen


def check_child_processes(spec, worker, result, dispatch, exits, lifecycles):
    """Each physics runtime belongs to a distinct, actually exited child.

On Windows the venv launcher may differ from the actual Python PID. Both are
bound explicitly; neither is silently substituted for the other.
"""
    phases = ['EARLY', 'GUARD'] if spec['protected'] else ['EARLY']
    require([r['phase'] for r in dispatch] == [r['phase'] for r in exits] == phases
            and set(lifecycles) == set(phases), 'Missing/reordered/retried phase process')
    require(result['phase_processes'] == exits, 'Result omits real phase process exits')
    parent_pid, prior_exit, child_pids = worker['pid'], None, set()
    parent_origin = parent_epoch = None
    for phase, launched, exit_record in zip(phases, dispatch, exits):
        execution, start, finish = lifecycles[phase]
        require(execution['phase'] == start['phase'] == finish['phase'] == phase
                and execution['task_parent_pid'] == launched['task_parent_pid']
                == exit_record['task_parent_pid'] == start['task_parent_pid'] == parent_pid,
                'Cold child parent/phase binding changed')
        origin = window.finite_time(execution['parent_started_monotonic'], 'parent monotonic origin')
        epoch = window.finite_time(execution['parent_started_epoch_seconds'], 'parent epoch origin')
        if parent_origin is None:
            parent_origin, parent_epoch = origin, epoch
        require(origin == parent_origin == worker['started_monotonic']
                and epoch == parent_epoch == worker['started_epoch_seconds'], 'Phase restarted parent task clock')
        require(launched['phase_cell'] == spec['cell'] + '/' + phase.lower(), 'Wrong phase dispatch target')
        require(type(launched['launcher_pid']) is int and launched['launcher_pid'] > 0
                and exit_record['launcher_pid'] == launched['launcher_pid'], 'Child launcher exit mismatch')
        child = start['pid']
        require(type(child) is int and child > 0 and child != parent_pid and child not in child_pids
                and finish['pid'] == exit_record['actual_child_pid'] == child,
                'Physical runtime was reused in the parent or same phase process')
        require(start['observed_parent_pid'] in (parent_pid, launched['launcher_pid']), 'Unbound child/venv process ancestry')
        require(exit_record['exit_code'] == 0 and finish['status'] == 'COMPLETE'
                and exit_record['phase_completion_sha256'] == result['phases'][phase]['completion_manifest_sha256'],
                'Child did not actually finish and seal successfully')
        began = window.finite_time(launched['started_monotonic'], 'child launch monotonic')
        ended = window.finite_time(exit_record['finished_monotonic'], 'child exit monotonic')
        require(exit_record['started_monotonic'] == began
                and exit_record['started_epoch_seconds'] == launched['started_epoch_seconds'], 'Child start identity differs')
        require(origin <= began <= ended and (prior_exit is None or prior_exit <= began),
                'GUARD started before EARLY actually exited')
        close(exit_record['process_wall_seconds'], ended - began, 'complete child process time')
        require(began - origin <= start['worker_elapsed_seconds'] <= finish['worker_elapsed_seconds'] <= ended - origin
                and ended - origin <= result['elapsed_seconds'], 'Child/parent recorded clocks omit work')
        child_pids.add(child)
        prior_exit = ended
    return {'cold_processes': len(phases), 'actual_child_pids': sorted(child_pids),
            'serial_real_exits_verified': True, 'parent_started_monotonic': parent_origin}


def check_local_checkpoints(rows, phase, checked, result, clock):
    expected = [('INITIAL' if i == 0 else 'DECISION', b) for i, b in enumerate(checked['boundaries'])]
    expected.append(('CLOSE', checked['boundaries'][-1]))
    require(len(rows) == len(expected), 'Child omitted initial/decision/close certificate boundary')
    last_worker = last_elapsed = 0.
    for index, (row, (kind, local)) in enumerate(zip(rows, expected)):
        wanted = {'checkpoint_index': index, 'phase': phase, 'kind': kind,
            'phase_choices': local['phase_choices'], 'search_choices': local['phase_choices'],
            **{k: local[k] for k in ('logical_points', 'physical_points', 'numerical_entries', 'requests',
                                   'quality_pct', 'fully_certified')},
            'phase_logical_points': local['logical_points'], 'phase_physical_points': local['physical_points'],
            'guard_established': phase == 'GUARD' and kind == 'CLOSE' and local['fully_certified']}
        require(set(row) == set(wanted) | {'worker_elapsed_seconds', 'elapsed_seconds'}, 'Child checkpoint schema differs')
        for key, value in wanted.items():
            same(row[key], value, 'independent_local_checkpoint.' + key)
        worker = window.finite_time(row['worker_elapsed_seconds'], 'child parent-clock checkpoint')
        elapsed = window.finite_time(row['elapsed_seconds'], 'child compute-clock checkpoint')
        require(last_worker <= worker and last_elapsed <= elapsed <= result['phase_elapsed_seconds'], 'Child checkpoint clock reversed')
        close(worker - elapsed, clock['parent_worker_compute_started_seconds'], 'child certificate clock origin')
        last_worker, last_elapsed = worker, elapsed
    return rows


def check_cell(run, spec, cfg, scope):
    cell = (run / spec['cell']).resolve()
    require(cell != run and cell.is_relative_to(run), 'Parent cell escaped run')
    sealed = manifest(cell)
    needed = {'spec.json', 'result.json', 'worker_started.json', 'worker_deadline.json',
              'cost_checkpoints.jsonl', 'equivalent_cache_summary.json', 'phase_dispatch.jsonl', 'phase_process_exits.json'}
    require(needed <= set(sealed) and not (cell / 'error.json').exists(), 'Missing/failed parent evidence')
    result, worker = read(cell / 'result.json'), read(cell / 'worker_started.json')
    require(read(cell / 'spec.json') == result['spec'] == spec and result['protected'] is spec['protected'],
            'Parent scientific identity changed')
    require(worker['spec_sha256'] == sha(cell / 'spec.json') and type(worker['pid']) is int and worker['pid'] > 0
            and worker['checkpoint_sha256'] == spec['checkpoint_sha256'] == result['checkpoint_sha256'], 'Worker/model identity changed')
    require(all(result[k] == 0 for k in ('new_training', 'optimizer_updates', 'training_interactions'))
            and all(result[k] is False for k in ('formal_holdout_numerical_access', 'tune_numerical_access', 'clpu_holdout_access')),
            'Unapproved training or heldout access')
    deadline = read(cell / 'worker_deadline.json')
    require(deadline['worker_seconds'] == 3600 and deadline['deadline_epoch_seconds'] ==
            min(read(run / 'execution_started.json')['deadline_epoch_seconds'], worker['started_epoch_seconds'] + 3600),
            'Worker/global time allowance changed')
    phase_order = ('EARLY', 'GUARD') if spec['protected'] else ('EARLY',)
    require(list(result['phases']) == list(phase_order), 'Cold phase population/order changed')
    scenarios = {row['scenario_id']: row for row in read(run / 'selected_scenarios.json')}
    events = {row['event_id']: row for row in read(run / 'selected_events.json')}
    mech = branch.Mechanics(read(run / 'author_static_metadata.json'), read(run / 'load_reference.json'),
        scenarios[spec['scenario_id']], read(run / 'condition.json'), events[spec['event_id']], cfg)
    policy = IndependentPolicy(run / spec['checkpoint'], spec) if spec['checkpoint'] else None
    phase_results, phase_checks, phase_summaries, oracles = {}, {}, {}, []
    child_exits = read(cell / 'phase_process_exits.json')
    require([r['phase'] for r in child_exits] == list(phase_order), 'Missing/reordered child process completion')
    child_exit = {row['phase']: row for row in child_exits}
    lifecycles, local_checkpoints = {}, {}
    statuses, previous_finish, physical_seconds = Counter(), 0., 0.
    for phase in phase_order:
        summary = result['phases'][phase]
        directory = phase_manifest(cell, sealed, phase, summary)
        ps = expected_phase_spec(spec, phase)
        phase_result = read(directory / 'phase_result.json')
        require(read(directory / 'spec.json') == phase_result['spec'] == ps, 'Cold phase spec changed')
        require(read(directory / 'initial_state.json') == mech.initial, 'Phase inherited another search state')
        identity = {'study': STUDY, 'freeze_sha256': sha(run / 'prephysical_freeze_manifest.json'),
            'scenario_id': spec['scenario_id'], 'event_id': spec['event_id'], 'initial_state': mech.initial,
            'condition_sha256': sha(run / 'condition.json'), 'physical_config_sha256': sha(run / 'physical_config.json'),
            'author_static_metadata_sha256': sha(run / 'author_static_metadata.json'),
            'source_provenance_sha256': sha(run / 'source_provenance.json')}
        require(read(directory / 'physical_identity.json') == identity, 'Cold phase fixed physical scope changed')
        points, point_statuses, commands, logical_rows, logical, requests, costs = prior.check_physical_and_logical(
            run, directory, ps, cfg, scope, mech, phase_result)
        statuses.update(point_statuses)
        physical_seconds += math.fsum(p['elapsed_seconds'] for p in points.values())
        plans = list(read_lines(directory / 'plans.jsonl.gz'))
        require(phase_result['attempts'] == plans
                and phase_result['plan_point_id_semantics'] == 'phase_local_logical_point_id', 'Phase plans or point identity changed')
        checked = check_phase_records(mech, cfg['branch_search'], phase, ps, phase_result, plans,
            list(read_lines(directory / 'decisions.jsonl.gz')), requests, logical, logical_rows,
            sha(run / 'condition.json'), policy if phase == 'EARLY' else None)
        historical = check_historical_phase(run, spec, phase, checked, requests)
        start, finish = read(directory / 'phase_started.json'), read(directory / 'phase_finished.json')
        execution, clock = read(directory / 'phase_execution.json'), read(directory / 'preflight_boundary.json')
        require(execution['phase_spec_sha256'] == sha(directory / 'spec.json')
                and start['phase_execution_sha256'] == sha(directory / 'phase_execution.json')
                and execution['deadline_epoch_seconds'] == deadline['deadline_epoch_seconds']
                and execution['clock_implementation'] == time.get_clock_info('perf_counter').implementation,
                'Child launch spec/deadline/clock binding differs')
        require(start['phase'] == finish['phase'] == phase and start['pid'] == finish['pid']
                == phase_result['actual_child_pid'] == child_exit[phase]['actual_child_pid']
                and start['task_parent_pid'] == finish['task_parent_pid'] == phase_result['task_parent_pid'] == worker['pid']
                and start['spec_sha256'] == sha(directory / 'spec.json')
                and finish['status'] == 'COMPLETE' and finish['phase_result_sha256'] == sha(directory / 'phase_result.json')
                and finish['oracle_summary_sha256'] == sha(directory / 'equivalent_cache_summary.json'), 'Phase lifecycle binding changed')
        for entry in (start, finish):
            datetime.fromisoformat(entry['utc'])
        began = window.finite_time(start['worker_elapsed_seconds'], 'phase start')
        ended = window.finite_time(finish['worker_elapsed_seconds'], 'phase end')
        duration = window.finite_time(phase_result['phase_elapsed_seconds'], 'cold phase duration')
        initialization = window.finite_time(phase_result['runtime_initialization_seconds'], 'cold initialization')
        require(previous_finish <= began <= ended <= result['elapsed_seconds']
                and ended - began >= duration >= initialization + math.fsum(p['elapsed_seconds'] for p in points.values()),
                'Cold phases overlap, start before prior close, or omit physical/initialization time')
        close(finish['phase_elapsed_seconds'], duration, 'phase duration lifecycle')
        require(clock['worker_time_origin'] == 'parent_worker_entry_perf_counter'
                and clock['child_started_monotonic'] == start['started_monotonic']
                and phase_result['child_entry_worker_seconds'] == began
                and clock['model_load_seconds'] == phase_result['model_load_seconds']
                and clock['child_preflight_elapsed_seconds'] == phase_result['child_preflight_elapsed_seconds']
                and clock['parent_worker_compute_started_seconds'] == phase_result['parent_worker_compute_started_seconds'],
                'Child preflight/model clock attribution differs')
        close(clock['child_started_monotonic'] - execution['parent_started_monotonic'], began, 'child entry parent clock')
        close(clock['phase_compute_started_monotonic'] - clock['child_started_monotonic'],
              clock['child_preflight_elapsed_seconds'], 'child actual preflight duration')
        close(clock['phase_compute_started_monotonic'] - execution['parent_started_monotonic'],
              clock['parent_worker_compute_started_seconds'], 'child actual computation origin')
        require(window.finite_time(clock['child_preflight_elapsed_seconds'], 'child preflight')
                >= window.finite_time(clock['model_load_seconds'], 'child model loading'), 'Child model load not charged')
        if phase == 'EARLY':
            require(result['preflight_elapsed_seconds'] == clock['parent_worker_compute_started_seconds']
                    and result['model_load_seconds'] == clock['model_load_seconds']
                    and 0 <= result['parent_validation_seconds'] <= result['preflight_elapsed_seconds'],
                    'Top preflight does not include parent start and EARLY child/model cost')
        else:
            require(clock['model_load_seconds'] == 0., 'Independent Beam child loaded a learned policy')
        local = list(read_lines(directory / 'cost_checkpoints.jsonl'))
        check_local_checkpoints(local, phase, checked, phase_result, clock)
        timers = phase_result['stage_component_seconds']
        require(set(timers) == {'observation_seconds', 'selection_seconds', 'search_step_seconds', 'journal_seconds'},
                'Phase timing intervals changed')
        require(duration >= initialization + math.fsum(window.finite_time(v, k) for k, v in timers.items()),
                'Phase component timing exceeds whole phase')
        expected_summary = {'directory': phase.lower(), 'status': phase_result['status'],
            'quality_pct': checked['quality_pct'], 'fully_certified': checked['fully_certified'],
            'logical_points': len(logical), 'physical_points': len(points),
            'numerical_entries': commands['numerical_entries'], 'requests': costs['requests'],
            'stop_reason': checked['stop_reason'], 'search_decisions': checked['decisions'],
            'policy_decisions': checked['policy_decisions'], 'stage_component_seconds': timers,
            'phase_elapsed_seconds': duration, 'runtime_initialization_seconds': initialization,
            'process_wall_seconds': child_exit[phase]['process_wall_seconds'],
            'actual_child_pid': child_exit[phase]['actual_child_pid'], 'launcher_pid': child_exit[phase]['launcher_pid'],
            'phase_result_sha256': sha(directory / 'phase_result.json'),
            'completion_manifest_sha256': sha(directory / 'completion_manifest.json')}
        same(summary, expected_summary, 'exact_phase_summary')
        previous_finish = ended
        phase_results[phase], phase_checks[phase] = phase_result, checked
        lifecycles[phase], local_checkpoints[phase] = (execution, start, finish), local
        phase_summaries[phase] = {**summary, 'phase': phase, 'historical_equivalence': historical,
            'semantic_signature_sha256': checked['semantic_signature_sha256'],
            'compact_hash_exact_matches': checked['compact_hash_exact_matches'],
            'compact_hash_numerical_reconstruction_differences': checked['compact_hash_numerical_reconstruction_differences'],
            'actual_and_independent_policy_actions_equal': True, 'physical_statuses': dict(point_statuses),
            'phase_started_worker_seconds': began, 'phase_finished_worker_seconds': ended,
            'model_load_seconds': clock['model_load_seconds'],
            'child_preflight_elapsed_seconds': clock['child_preflight_elapsed_seconds'],
            'child_compute_started_worker_seconds': clock['parent_worker_compute_started_seconds'],
            'local_checkpoints': local}
        oracles.append(phase_result['oracle'])
    fields = ('logical_points', 'physical_points', 'actual_numerical_entries', 'initialization_numerical_entries',
              'requests', 'semantic_cache_hits', 'electrical_reuses')
    combined = {'variant': 'electrical_equivalent', 'scope': 'sum_of_independent_cold_phase_ledgers',
                'cold_searches': len(phase_order), **{k: sum(o[k] for o in oracles) for k in fields}}
    require(result['oracle'] == read(cell / 'equivalent_cache_summary.json') == combined, 'Combined cold ledgers do not sum')
    require(combined['logical_points'] == result['logical_points'] <= 256 * len(phase_order)
            and combined['physical_points'] == result['actual_physical_points'] <= combined['logical_points']
            and combined['actual_numerical_entries'] == result['numerical_entries']
            == 3 * len(phase_order) + 4 * combined['physical_points'], 'Two-route actual cost/initializations omitted')
    chosen = check_result_selection(result, phase_results)
    child_processes = check_child_processes(spec, worker, result,
        list(read_lines(cell / 'phase_dispatch.jsonl')), child_exits, lifecycles)
    require(result['search_decisions'] == sum(v['decisions'] for v in phase_checks.values())
            and result['policy_decisions'] == sum(v['policy_decisions'] for v in phase_checks.values())
            and result['stage_decisions'] == {k: v['decisions'] for k, v in phase_checks.items()}
            and result['stage_component_seconds'] == {k: v['stage_component_seconds'] for k, v in phase_summaries.items()},
            'Total decisions or timing attribution changed')
    require(result['checkpoints'] == list(read_lines(cell / 'cost_checkpoints.jsonl')), 'Exact cost checkpoint journal differs')
    check_quality_costs(result['checkpoints'], phase_checks, result)
    for phase, summary in phase_summaries.items():
        checkpoints = [r for r in result['checkpoints'] if r['phase'] == phase]
        require(len(checkpoints) == len(local_checkpoints[phase]), 'Parent dropped child certificate boundaries')
        for row, local in zip(checkpoints, local_checkpoints[phase]):
            require(row['worker_elapsed_seconds'] == local['worker_elapsed_seconds'], 'Parent changed actual certificate arrival time')
            require(row['elapsed_seconds'] == local['worker_elapsed_seconds'] - result['preflight_elapsed_seconds'],
                    'Unified parent computation certificate time changed')
        require(summary['phase_started_worker_seconds'] <= checkpoints[0]['worker_elapsed_seconds']
                <= checkpoints[-1]['worker_elapsed_seconds'] <= summary['phase_finished_worker_seconds'],
                'Phase certification boundary outside cold phase lifecycle')
        summary['checkpoints'] = checkpoints
    timing = prior.check_cold_costs(result, {0: {'elapsed_seconds': physical_seconds}})
    close(result['runtime_initialization_seconds'], sum(p['runtime_initialization_seconds'] for p in phase_summaries.values()),
          'all_cold_initialization_time')
    require(result['computation_elapsed_seconds'] >= sum(p['phase_elapsed_seconds'] for p in phase_summaries.values()),
            'Total computation omits a cold route')
    return {**{k: spec[k] for k in ('cell', 'context_id', 'scenario_id', 'event_id', 'arm', 'arm_slot', 'seed', 'method')},
        'status': 'PASS', 'result_status': result['status'], 'cell_completion_sha256': sha(cell / 'completion_manifest.json'),
        'result_sha256': sha(cell / 'result.json'), 'quality_pct': result['quality_pct'],
        'fully_certified': result['fully_certified'], 'logical_points': combined['logical_points'],
        'physical_points': combined['physical_points'], 'numerical_entries': combined['actual_numerical_entries'],
        'requests': combined['requests'], 'cold_searches': len(phase_order),
        'semantic_cache_hits': combined['semantic_cache_hits'], 'electrical_reuses': combined['electrical_reuses'],
        'physical_statuses': dict(statuses), 'policy_decisions': result['policy_decisions'],
        'guard_established': chosen['guard_established'], 'guard_quality_pct': result['baseline_quality_pct'],
        'baseline_quality_pct': result['baseline_quality_pct'], 'selected_phase': result['selected_phase'],
        'gain_over_guard_kwh': chosen['gain_over_guard_kwh'], 'guard_complete_replay': spec['protected'],
        'early_quality_pct': result['early_quality_pct'], 'phase_summary': phase_summaries,
        'quality_checkpoints_verified': True, 'checkpoints': result['checkpoints'],
        'early_block_checkpoint': result['early_block_checkpoint'], 'final_real_cost': result['final_real_cost'],
        'phase_process_verification': child_processes, 'parent_validation_seconds': result['parent_validation_seconds'], **timing}


def check_controller(controller, schedule, checks, dispatch):
    cells = {r['cell'] for r in schedule}
    require(len(schedule) == len(cells) == 64 and controller['status'] == 'COMPLETE_CELLS_PENDING_AUDIT'
            and controller['scheduled'] == 64 and controller['errors'] == controller['unknown_cost_cells'] == []
            and controller['dispatched'] == schedule and len(controller['completed']) == 64
            and set(controller['completed']) == cells, 'Incomplete, changed or retried cold population')
    exits = controller['exits']
    require(len(exits) == 64 and {r['cell'] for r in exits} == cells
            and all(r['exit_code'] == 0 and r['drained_after_controller_error'] is False for r in exits),
            'Missing/failed cold process exit')
    require(len(checks) == 64 and {r['cell'] for r in checks} == cells
            and [r['cell'] for r in dispatch] == [r['cell'] for r in schedule], 'Audit/dispatch population differs')
    ended = {r['cell']: r for r in exits}
    previous_finished, pair_finished = 0., []
    for index, (spec, start) in enumerate(zip(schedule, dispatch)):
        finish = ended[spec['cell']]
        if index % 2 == 0:
            if pair_finished:
                previous_finished = max(pair_finished)
            pair_finished = []
        for key in ('cell', 'pair_order', 'context_order', 'arm_position'):
            require(start[key] == spec[key], 'Controller Latin order differs')
        require(type(start['launcher_pid']) is int and start['launcher_pid'] > 0
                and finish['launcher_pid'] == start['launcher_pid'] and finish['pair_order'] == spec['pair_order']
                and finish['started_epoch_seconds'] == start['started_epoch_seconds'], 'Process identity changed')
        began = window.finite_time(start['started_epoch_seconds'], 'launch epoch')
        finished = window.finite_time(finish['finished_epoch_seconds'], 'exit epoch')
        require(previous_finished <= began <= finished, 'Next pair began before previous pair drained')
        datetime.fromisoformat(start['utc'])
        datetime.fromisoformat(finish['utc'])
        pair_finished.append(finished)
    totals = {k: sum(r[k] for r in checks) for k in ('logical_points', 'physical_points', 'numerical_entries', 'requests')}
    require(sum(r['cold_searches'] for r in checks) == 120
            and controller['known_logical_points'] == totals['logical_points'] <= 30720
            and controller['known_physical_points'] == totals['physical_points'] <= totals['logical_points']
            and controller['known_numerical_entries'] == totals['numerical_entries']
            == 120 * 3 + 4 * totals['physical_points'] <= 123240, 'Controller omits a cold route or paid initialization')
    for checked in checks:
        seconds = window.finite_time(ended[checked['cell']]['process_wall_seconds'], 'complete process duration')
        require(seconds >= checked['worker_elapsed_seconds'], 'Controller process duration omits worker work')
        checked['process_wall_seconds'] = seconds
    window.finite_time(controller['wall_seconds'], 'controller wall duration')
    return totals


def baseline_equalities(checks):
    output = []
    for context in range(8):
        selected = [r for r in checks if r['context_id'] == context and r['guard_complete_replay']]
        require(len(selected) == 7 and all(r['phase_summary']['GUARD']['historical_equivalence'] for r in selected),
                'Seven independent original Beam guards required')
        signatures = {r['phase_summary']['GUARD']['semantic_signature_sha256'] for r in selected}
        require(len(signatures) == 1, 'Independent original Beam scientific traces differ')
        output.append({'context_id': context, 'status': 'PASS', 'cold_paid_baselines': 7,
            'semantic_signature_sha256': next(iter(signatures)), 'cells': [r['cell'] for r in selected]})
    return output


def source_hashes(run, scope):
    return {'freeze_sha256': sha(run / 'prephysical_freeze_manifest.json'),
        'config_sha256': sha(run / 'config.json'), 'schedule_sha256': sha(run / 'schedule.json'),
        'source_binding_sha256': sha(run / 'source_binding.json'),
        'physical_scope_file_sha256': sha(run / 'physical_scope.json'),
        'physical_scope_canonical_sha256': digest(scope), 'model_bindings_sha256': sha(run / 'model_bindings.json'),
        'model_preflight_manifest_sha256': sha(run / 'model_preflight/completion_manifest.json'),
        'scope_review_manifest_sha256': sha(run / 'scope_review/completion_manifest.json'),
        'exact_key_review_manifest_sha256': sha(run / 'exact_key_review/completion_manifest.json'),
        'early_diagnostic_manifest_sha256': sha(run / 'early_diagnostic/completion_manifest.json'),
        'source_integrated_completion_sha256': SOURCE_SHA}


def audit(run, output=None):
    from audit_d059_supplement import NoSolverImports, no_solver_loaded
    run = Path(run).resolve()
    output = Path(output).resolve() if output else run / 'audit'
    require(output.is_relative_to(ROOT) and output != run and not output.exists(), 'Preserve earlier audit evidence')
    output.mkdir(parents=True)
    started, checks = time.perf_counter(), []
    result = {'status': 'FAIL', 'source_run': run.name, 'errors': [], 'audited_cells': 0, 'results': [],
              'new_audit_physical_points': 0, 'new_training_tasks': 0}
    guard = NoSolverImports()
    sys.meta_path.insert(0, guard)
    try:
        no_solver_loaded()
        cfg, scope, schedule = check_parent(run)
        result.update(source_hashes(run, scope))
        for spec in schedule:
            checked = check_cell(run, spec, cfg, scope)
            checks.append(checked)
            save(output / (spec['cell'].replace('/', '_') + '.json'), checked)
            save(output / 'progress.json', {'status': 'RUNNING', 'audited_cells': len(checks), 'scheduled': 64,
                'last_cell': spec['cell'], 'elapsed_seconds': time.perf_counter() - started,
                'new_physical_points': 0, 'new_training': 0})
        controller = read(run / 'controller_completion.json')
        require(controller['exits'] == read(run / 'controller_exits.json'), 'Independent process exit journal differs')
        totals = check_controller(controller, schedule, checks, list(read_lines(run / 'controller_dispatch.jsonl')))
        equalities = baseline_equalities(checks)
        no_solver_loaded()
        statuses = Counter()
        for row in checks:
            statuses.update(row['physical_statuses'])
            save(output / (row['cell'].replace('/', '_') + '.json'), row)
        result.update(status='PASS', audited_cells=len(checks), results=checks, baseline_equality=equalities,
            **totals, cold_searches=120, physical_statuses=dict(statuses),
            controller_completion_sha256=sha(run / 'controller_completion.json'),
            controller_dispatch_sha256=sha(run / 'controller_dispatch.jsonl'),
            interpretation='Eight observed training contexts; fixed same-source BC/PPO, not equal total training data/cost. Two independent paid searches; no cross-phase answers or uncompleted guard claim.')
    except BaseException as exc:
        result.update(audited_cells=len(checks), results=checks)
        result['errors'].append({'error': repr(exc), 'traceback': traceback.format_exc()})
    finally:
        sys.meta_path.remove(guard)
    result['elapsed_seconds'] = time.perf_counter() - started
    save(output / 'summary.json', result)
    save(output / 'completion_manifest.json', {p.relative_to(output).as_posix(): sha(p)
        for p in output.rglob('*') if p.is_file() and p.name != 'completion_manifest.json'})
    return result


def preflight(run, output):
    from audit_d059_supplement import NoSolverImports, no_solver_loaded
    run, output = Path(run).resolve(), Path(output).resolve()
    require(output.is_relative_to(ROOT) and not output.exists(), 'Preserve prior preflight evidence')
    output.mkdir(parents=True)
    guard = NoSolverImports()
    sys.meta_path.insert(0, guard)
    result = {'status': 'FAIL', 'errors': [], 'source_run': run.name, 'new_physical_points': 0, 'new_training': 0}
    try:
        no_solver_loaded()
        cfg, scope, schedule = check_parent(run)
        no_solver_loaded()
        result.update(status='PASS', contexts=8, cells=len(schedule), cold_searches=120, fixed_models=6,
                      **source_hashes(run, scope))
    except BaseException as exc:
        result['errors'].append({'error': repr(exc), 'traceback': traceback.format_exc()})
    finally:
        sys.meta_path.remove(guard)
    save(output / 'summary.json', result)
    save(output / 'completion_manifest.json', {'summary.json': sha(output / 'summary.json')})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        require(args.output is not None, 'Preflight requires an independent evidence output directory')
        result = preflight(args.run, args.output)
    else:
        result = audit(args.run, args.output)
    print(prior.canonical({k: v for k, v in result.items() if k not in ('results', 'baseline_equality')}), flush=True)
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
