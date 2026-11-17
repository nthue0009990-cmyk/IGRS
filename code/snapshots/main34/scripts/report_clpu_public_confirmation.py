"""Audit-bound 4-fault x 6-event P/E confirmation report, with zero physics.

Only a complete independent PASS for all 48 frozen cold tasks is consumable.
The crossed bootstrap resamples fault and event labels, never 24 iid pairs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import sys
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.dont_write_bytecode = True
import report_clpu_public_mechanism_pilot as previous

read, save, sha, require, manifest = previous.read, previous.save, previous.sha, previous.require, previous.manifest
STUDY = 'CLPU_PUBLIC_CONFIRMATION_COMPACT_V1'
SCHEMA = 'clpu-public-confirmation-protected-result-v1'
STATUS = 'AUDITED_DESCRIPTIVE_PUBLIC_CONFIRMATION'
ARMS = ('P', 'E')
CONTEXTS, CELLS, FAULTS, EVENTS = 24, 48, 4, 6
PRIMARY, METRICS, COSTS, MECHANISMS = previous.PRIMARY, previous.METRICS, previous.COSTS, previous.MECHANISMS
IMPLEMENTATIONS = ('scripts/report_clpu_public_confirmation.py',
    'scripts/report_clpu_public_mechanism_pilot.py', 'scripts/report_clpu_shared_pilot.py',
    'scripts/report_clpu_staged_pilot.py', 'scripts/report_clpu_integrated_pilot.py')
AUDIT_FILES = {'freeze_sha256': 'prephysical_freeze_manifest.json', 'method_freeze_sha256': 'method_freeze_manifest.json',
    'config_sha256': 'config.json', 'schedule_sha256': 'schedule.json', 'contexts_sha256': 'contexts.json',
    'source_binding_sha256': 'source_binding.json', 'physical_scope_file_sha256': 'physical_scope.json',
    'scope_review_manifest_sha256': 'scope_review/completion_manifest.json',
    'exact_key_review_manifest_sha256': 'exact_key_review/completion_manifest.json',
    'model_bindings_sha256': 'model_bindings.json', 'stage_plan_sha256': 'stage_plan.json',
    'population_source_sha256': 'population_source.json', 'hydration_sha256': 'hydration.json',
    'selected_scenarios_sha256': 'selected_scenarios.json', 'selected_events_sha256': 'selected_events.json',
    'controller_completion_sha256': 'controller_completion.json',
    'controller_execution_sha256': 'controller_stage1_completion.json',
    'precheck_summary_sha256': 'precheck/summary.json'}
STATISTICAL_RULES = {
    'version': 'public-confirmation-crossed-paired-statistics-v1',
    'population': {'faults': FAULTS, 'events': EVENTS, 'contexts': CONTEXTS, 'cold_tasks': CELLS,
                   'arms': list(ARMS), 'seeds_per_arm_context': [0], 'full_fault_event_cartesian_product': True},
    'primary_metric': 'physical_points',
    'primary_estimand': 'ratio_of_equal_context_means_P_over_E',
    'saving_pct': '100*(1-ratio_of_means_P_over_E); never mean_of_pair_ratios',
    'quality_comparison': 'exact final complete energy P versus E; no epsilon or rounded quality ties',
    'PF_saving_claim_rule': 'all 24 exact complete-energy pairs P>=E and primary PF ratio percentile interval upper bound strictly <1',
    'pairing': ['context_id', 'scenario_id', 'event_id'],
    'group_reports': ['all_24_pairs', 'each_of_4_faults_over_6_events', 'each_of_6_events_over_4_faults'],
    'missing': 'retain all frozen rows and nulls; any missing suppresses that group mean, ratio and interval; no imputation or complete-case estimates',
    'incomplete_status_export': 'all 48 frozen identities; controller exits and raw cost snapshots only, explicitly unverified; no quality read or full-population estimand',
    'bootstrap': {'method': 'crossed_pigeonhole_two_cluster_percentile', 'replicates': 10000,
                  'seed': 20261024, 'fault_draws': 4, 'event_draws': 6,
                  'resampling': 'independent with-replacement draws of sorted fault and event labels, Cartesian multiplicity weights, same weights for P and E',
                  'percentiles': [0.025, 0.975], 'quantile': 'linear interpolation at (n-1)*p',
                  'undefined_replicates': 'retain count and suppress entire interval; never discard or redraw'},
    'time': 'complete cold process seconds; one run per arm/context, descriptive machine-dependent secondary endpoint',
    'inference_limit': 'four fault clusters and six event clusters; not 24 independent replications; intervals exploratory, no p-value or guaranteed generalization',
    'external_method_claim': False,
    'learning_claim': False,
}


def private(function, **changes):
    globals_copy = dict(function.__globals__)
    globals_copy.update(changes)
    result = FunctionType(function.__code__, globals_copy, function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


# Reuse validated cost/certificate parsing with new identity, not old population.
case_row = private(previous.case_row, STUDY=STUDY, SCHEMA=SCHEMA)
check_audit_bindings = private(previous.check_audit_bindings, AUDIT_FILES=AUDIT_FILES)


def check_analysis_freeze(run):
    frozen_analysis = tuple('source/' + name for name in IMPLEMENTATIONS)
    manifest(run, name='method_freeze_manifest.json', required=frozen_analysis, all_files=True)
    require(all(sha(ROOT / name) == sha(run / 'source' / name) for name in IMPLEMENTATIONS),
            'Working analysis code differs from the pre-access method freeze')


def population(rows):
    require(isinstance(rows, list) and len(rows) == CELLS and len({r['cell'] for r in rows}) == CELLS,
            'All 48 unique frozen tasks required; no failed/missing row may be dropped')
    require({r['context_id'] for r in rows} == set(range(CONTEXTS)), 'All 24 contexts required')
    contexts = []
    for context in range(CONTEXTS):
        group = [r for r in rows if r['context_id'] == context]
        require(Counter((r['arm'], r['seed']) for r in group) == Counter((arm, 0) for arm in ARMS),
                'Exactly one deterministic P/E pair per context required')
        require(len({(r['scenario_id'], r['event_id']) for r in group}) == 1, 'Unpaired fault/event identity')
        contexts.append((group[0]['scenario_id'], group[0]['event_id']))
    faults = sorted({f for f, _ in contexts})
    events = sorted({e for _, e in contexts})
    require(len(faults) == FAULTS and len(events) == EVENTS and len(set(contexts)) == CONTEXTS
            and set(contexts) == {(f, e) for f in faults for e in events},
            'Frozen 4x6 full crossed population required; no repeated context substitution')
    require(all(r['method'] == 'beam' and r['seed'] == 0 for r in rows), 'No learned or extra-seed tasks permitted')
    return faults, events


def distribution(values, denominator):
    require(len(values) == denominator, 'Frozen denominator changed')
    present = [v for v in values if v is not None]
    require(all(type(v) in (int, float) and math.isfinite(v) for v in present), 'Nonfinite/non-numeric report metric')
    complete = len(present) == denominator
    return {'mean': statistics.fmean(present) if complete else None,
            'median': statistics.median(present) if complete else None,
            'min_observed': min(present) if present else None, 'max_observed': max(present) if present else None,
            'present': len(present), 'missing': denominator - len(present), 'denominator': denominator}


def ratio(p_mean, e_mean):
    return p_mean / e_mean if p_mean is not None and e_mean is not None and e_mean != 0 else None


def quantile(values, probability):
    require(bool(values) and 0 <= probability <= 1, 'Quantile needs finite data and probability')
    values = sorted(values)
    location = (len(values) - 1) * probability
    lo, hi = math.floor(location), math.ceil(location)
    return values[lo] + (values[hi] - values[lo]) * (location - lo)


def cluster_weights(replicates=10000, seed=20261024):
    """Yield 24 multiplicities: product of 4 fault and 6 event draw counts."""
    require(type(replicates) is int and replicates > 0 and type(seed) is int, 'Invalid bootstrap controls')
    rng = random.Random(seed)
    for _ in range(replicates):
        fault_counts = Counter(rng.randrange(FAULTS) for _ in range(FAULTS))
        event_counts = Counter(rng.randrange(EVENTS) for _ in range(EVENTS))
        yield tuple(fault_counts[f] * event_counts[e] for f in range(FAULTS) for e in range(EVENTS))


def crossed_bootstrap(pairs, *, replicates=10000, seed=20261024):
    require(len(pairs) == CONTEXTS, 'All paired contexts required for cluster bootstrap')
    lookup = {(p['scenario_id'], p['event_id']): p for p in pairs}
    faults = sorted({f for f, _ in lookup})
    events = sorted({e for _, e in lookup})
    require(len(faults) == FAULTS and len(events) == EVENTS and len(lookup) == CONTEXTS,
            'Bootstrap requires the complete 4x6 grid')
    ordered = [lookup[f, e] for f in faults for e in events]
    metrics = PRIMARY + ('final_energy_kwh',)
    vectors = {key: ([p['P'][key] for p in ordered], [p['E'][key] for p in ordered]) for key in metrics}
    complete = {key: all(v is not None for arm in vectors[key] for v in arm) for key in metrics}
    for key, arms in vectors.items():
        for vector in arms:
            distribution(vector, CONTEXTS)
    samples = {key: {'signed_delta_mean': [], 'ratio_of_means_P_over_E': []} for key in metrics}
    weights_hash = hashlib.sha256()
    for weights in cluster_weights(replicates, seed):
        require(sum(weights) == CONTEXTS, 'Cluster resampling changed weighted population size')
        weights_hash.update((','.join(map(str, weights)) + '\n').encode('ascii'))
        for key, (p_values, e_values) in vectors.items():
            if not complete[key]:
                samples[key]['signed_delta_mean'].append(None)
                samples[key]['ratio_of_means_P_over_E'].append(None)
                continue
            pm = math.fsum(w * v for w, v in zip(weights, p_values)) / CONTEXTS
            em = math.fsum(w * v for w, v in zip(weights, e_values)) / CONTEXTS
            samples[key]['signed_delta_mean'].append(pm - em)
            samples[key]['ratio_of_means_P_over_E'].append(ratio(pm, em))
    result = {}
    for metric, estimands in samples.items():
        result[metric] = {}
        for estimand, values in estimands.items():
            valid = [v for v in values if v is not None]
            all_defined = len(valid) == replicates
            result[metric][estimand] = {
                'percentile_95_interval': [quantile(valid, .025), quantile(valid, .975)] if all_defined else None,
                'defined_replicates': len(valid), 'undefined_replicates': replicates - len(valid),
                'interval_status': 'EXPLORATORY_SMALL_CLUSTER_INTERVAL' if all_defined else
                                   'SUPPRESSED_MISSING_INPUT' if not complete[metric] else 'SUPPRESSED_UNDEFINED_REPLICATE'}
    return {'method': STATISTICAL_RULES['bootstrap']['method'], 'replicates': replicates, 'seed': seed,
            'fault_clusters': faults, 'event_clusters': events, 'pair_count': CONTEXTS,
            'independent_pair_resampling': False, 'weights_sha256': weights_hash.hexdigest(),
            'metrics': result, 'limitations': STATISTICAL_RULES['inference_limit']}


def group_summary(rows, context_ids):
    size = len(context_ids)
    group = [r for r in rows if r['context_id'] in context_ids]
    arms = {arm: {'tasks': size, 'metrics': {key: distribution([r[key] for r in group if r['arm'] == arm], size)
                   for key in METRICS}} for arm in ARMS}
    lookup = {(r['context_id'], r['arm']): r for r in group}
    contrasts = {}
    for metric in PRIMARY:
        deltas, relations = [], []
        for context in context_ids:
            p, e = lookup[context, 'P'], lookup[context, 'E']
            deltas.append(None if p[metric] is None or e[metric] is None else p[metric] - e[metric])
            quality = metric == 'quality_pct'
            source = 'final_energy_kwh' if quality else metric
            score = None if p[source] is None or e[source] is None else p[source] - e[source]
            relations.append('missing' if score is None else 'tie' if score == 0
                             else 'win' if (score > 0 if quality else score < 0) else 'loss')
        pm, em = (arms[arm]['metrics'][metric]['mean'] for arm in ARMS)
        value = ratio(pm, em)
        counts = Counter(relations)
        complete = all(v is not None for v in deltas)
        contrasts[metric] = {'P_mean': pm, 'E_mean': em, 'signed_delta': distribution(deltas, size),
            'ratio_of_means_P_over_E': value,
            'relative_change_pct': None if value is None else 100 * (value - 1),
            'saving_pct': None if value is None else 100 * (1 - value),
            'P_win_tie_loss_missing': {key: counts[key] for key in ('win', 'tie', 'loss', 'missing')},
            'worst_context_delta': (min if metric == 'quality_pct' else max)(deltas) if complete else None,
            'quality_relation_uses_exact_complete_energy': metric == 'quality_pct'}
    return {'context_ids': list(context_ids), 'denominator_pairs': size, 'arms': arms, 'P_minus_E': contrasts}


def summarize(rows, *, bootstrap_replicates=10000, bootstrap_seed=20261024):
    faults, events = population(rows)
    lookup = {(r['context_id'], r['arm']): r for r in rows}
    pairs = []
    for context in range(CONTEXTS):
        p, e = lookup[context, 'P'], lookup[context, 'E']
        require(p['guard_energy_kwh'] == e['guard_energy_kwh'], 'Paired complete GUARD energy differs')
        quality_delta = None if p['final_energy_kwh'] is None or e['final_energy_kwh'] is None else p['final_energy_kwh'] - e['final_energy_kwh']
        relations = {}
        for metric in PRIMARY:
            score = quality_delta if metric == 'quality_pct' else None if p[metric] is None or e[metric] is None else p[metric] - e[metric]
            relations[metric] = 'missing' if score is None else 'tie' if score == 0 else 'win' if (score > 0 if metric == 'quality_pct' else score < 0) else 'loss'
        pairs.append({'context_id': context, 'scenario_id': p['scenario_id'], 'event_id': p['event_id'],
            'p_cell': p['cell'], 'control_cell': e['cell'],
            'P': {k: p[k] for k in METRICS}, 'E': {k: e[k] for k in METRICS},
            'signed_delta': {k: None if p[k] is None or e[k] is None else p[k] - e[k] for k in METRICS},
            'p_relations': relations, 'exact_energy_delta_kwh': quality_delta,
            'quality_relation_uses_exact_complete_energy': True,
            'physical_points_pair_ratio_P_over_E': ratio(p['physical_points'], e['physical_points'])})
    groups = {'overall': group_summary(rows, list(range(CONTEXTS))),
        'by_fault': [{'scenario_id': fault, **group_summary(rows, [p['context_id'] for p in pairs if p['scenario_id'] == fault])} for fault in faults],
        'by_event': [{'event_id': event, **group_summary(rows, [p['context_id'] for p in pairs if p['event_id'] == event])} for event in events]}
    boot = crossed_bootstrap(pairs, replicates=bootstrap_replicates, seed=bootstrap_seed)
    quality_known = all(p['exact_energy_delta_kwh'] is not None for p in pairs)
    quality_ok = all(p['exact_energy_delta_kwh'] >= 0 for p in pairs) if quality_known else None
    primary = groups['overall']['P_minus_E']['physical_points']
    primary_ratio = primary['ratio_of_means_P_over_E']
    primary_interval = boot['metrics']['physical_points']['ratio_of_means_P_over_E']['percentile_95_interval']
    interval_below_one = primary_interval is not None and primary_interval[1] < 1
    decision = {'all_24_exact_energy_pairs_nondegrading': quality_ok,
                'worst_exact_energy_delta_kwh': min(p['exact_energy_delta_kwh'] for p in pairs) if quality_known else None,
                'primary_PF_ratio_P_over_E': primary_ratio,
                'primary_PF_saving_pct': primary['saving_pct'],
                'primary_PF_ratio_interval_upper_strictly_below_one': interval_below_one,
                'same_quality_PF_saving_supported_under_frozen_interval_rule': quality_ok is True and interval_below_one,
                'same_or_higher_quality_with_lower_PF_descriptive_condition': quality_ok is True and primary_ratio is not None and primary_ratio < 1,
                'not_an_automatic_expansion_or_publication_decision': True}
    return groups, pairs, boot, decision


def load_cases(run):
    """Read small outcome files only after the complete independent audit gate."""
    audit = read(run / 'audit/summary.json')
    require(audit['status'] == 'PASS' and not audit['errors']
            and audit['audited_cells'] == audit['resolved_slots'] == audit['actual_tasks'] == CELLS,
            'Only an independent complete PASS for all 48 frozen tasks may be reported')
    manifest(run / 'audit', required=('summary.json',), all_files=True)
    # The analysis implementation itself must be fixed before numerical access,
    # not merely copied into a report after outcomes have become visible.
    check_analysis_freeze(run)
    prepared = read(run / 'prepared.json')
    require(prepared['status'] == 'FROZEN_PREPARED_NO_DSS' and prepared['freeze_sha256'] == sha(run / 'prephysical_freeze_manifest.json'),
            'Prepared prephysical freeze changed')
    manifest(run, name='prephysical_freeze_manifest.json', required=('config.json', 'schedule.json', 'stage_plan.json'), all_files=True)
    cfg, schedule, plan = (read(run / name) for name in ('config.json', 'schedule.json', 'stage_plan.json'))
    require(cfg['study'] == STUDY and cfg['confirmation']['models_used'] == 0
            and cfg['confirmation_statistics'] == STATISTICAL_RULES, 'Frozen study or statistical rules differ')
    population(schedule)
    require(not (run / 'phase_gate.json').exists() and plan['execution'] == 'single_stage_all48', 'Conditional population forbidden')
    controller = read(run / 'controller_completion.json')
    require(controller['status'] == 'COMPLETE_CELLS_PENDING_AUDIT' and controller['actual_tasks'] == controller['resolved_slots'] == CELLS
            and not controller['errors'] and not controller['not_run'] and not controller['unknown_cost_cells'],
            'All 48 real complete exits required')
    # Explicit new-audit bindings, with no historical-performance comparator.
    check_audit_bindings(run, audit)
    bindings = {name: audit[key] for key, name in AUDIT_FILES.items()}
    if 'source_hashes' in audit:
        for name, digest in audit['source_hashes'].items():
            path = (run / name).resolve()
            require(path.is_relative_to(run) and path != run and sha(path) == digest,
                    'Audit source map changed: ' + name)
            require(name not in bindings or bindings[name] == digest, 'Conflicting audit source bindings')
            bindings[name] = digest
    names = {s['cell'] for s in schedule}
    require(len(audit['results']) == len(controller['exits']) == len(controller['completed']) == CELLS
            and {r['cell'] for r in audit['results']} == {r['cell'] for r in controller['exits']} == set(controller['completed']) == names,
            'Schedule/audit/real-exit populations differ')
    equalities = audit['context_equalities']
    require(len(equalities) == CONTEXTS and {r['context_id'] for r in equalities} == set(range(CONTEXTS))
            and all(r['status'] == 'PASS' and r['guard_trajectories_equal'] for r in equalities),
            'Paired complete GUARD/candidate-pool equality is missing')
    checks, exits = ({r['cell']: r for r in items} for items in (audit['results'], controller['exits']))
    output_bindings = {**bindings, 'audit/summary.json': sha(run / 'audit/summary.json'),
                       'audit/completion_manifest.json': sha(run / 'audit/completion_manifest.json')}
    required = ('spec.json', 'result.json', 'worker_finished.json', 'service_summary.json',
        'physical/oracle_summary.json', 'cost_checkpoints.jsonl', 'EARLY/phase_result.json',
        'GUARD/phase_result.json', 'EARLY/candidate_pool.json', 'EARLY/view_summary.json', 'GUARD/view_summary.json')
    rows, evidence, pools = [], [], {}
    for spec in schedule:
        cell = (run / spec['cell']).resolve()
        require(cell.is_relative_to(run) and cell != run, 'Unsafe task path')
        checked = checks[spec['cell']]
        require(checked['guard_replay_verified'] is True and checked['history_verified'] is False,
                'Current complete GUARD replay required; historical outcome substitution forbidden')
        require(sha(cell / 'completion_manifest.json') == checked['cell_completion_sha256'], 'Audited task seal changed')
        manifest(cell, required=required)
        result = read(cell / 'result.json')
        require(read(cell / 'spec.json') == spec and sha(cell / 'result.json') == checked['result_sha256'], 'Audited task identity/result changed')
        require(read(cell / 'worker_finished.json')['result_sha256'] == checked['result_sha256'], 'Worker binding changed')
        phases = {phase: read(cell / phase / 'phase_result.json') for phase in ('EARLY', 'GUARD')}
        require(all(sha(cell / phase / 'phase_result.json') == checked['phase_summary'][phase]['phase_result_sha256'] for phase in phases),
                'Audited phase changed')
        require(read(cell / 'service_summary.json') == result['service']
                and all(read(cell / phase / 'view_summary.json') == result['service']['views'][phase] for phase in phases),
                'Service/view ledger changed')
        require([json.loads(line) for line in (cell / 'cost_checkpoints.jsonl').read_text(encoding='utf-8').splitlines()] == result['checkpoints'],
                'Audited checkpoint journal changed')
        generation = phases['EARLY']['stage_generation']
        require(generation == read(cell / 'EARLY/candidate_pool.json')['generation'], 'Candidate generation file changed')
        if spec['context_id'] in pools:
            require(pools[spec['context_id']] == generation, 'P/E candidate generation differs')
        pools[spec['context_id']] = generation
        rows.append(case_row(spec, result, phases, checked, exits[spec['cell']], read(cell / 'physical/oracle_summary.json')))
        entries = previous.staged.plan_evidence(spec, phases['EARLY'])
        pruned = {tuple(p['actions']): p for p in phases['EARLY'].get('guard_first', {}).get('free_pruning', [])}
        for entry in entries:
            prune = pruned.get(tuple(entry['candidate']['actions']))
            if prune:
                require(entry['verification_status'] == 'NOT_VERIFIED', 'Pruned plan also claimed executed')
                entry.update(verification_status=prune['status'], public_pruning=deepcopy(prune), physical_feasibility='UNASSESSED')
        evidence.extend(entries)
        output_bindings.update({spec['cell'] + '/' + name: sha(cell / name) for name in (*required, 'completion_manifest.json')})
    population(rows)
    for key in COSTS:
        require(sum(row[key] for row in rows) == audit[key] == controller[key], 'Audited total changed: ' + key)
    require(dict(sum((Counter(row['physical_statuses']) for row in rows), Counter())) == audit['physical_statuses'],
            'Audited raw status totals changed')
    return rows, evidence, pools, output_bindings, audit, controller


def status_observations(run, schedule):
    """Retain all identities; observe exits/costs without opening any result file."""
    population(schedule)
    bindings, errors = {}, []

    def snapshot(name):
        path = (run / name).resolve()
        require(path.is_relative_to(run) and path != run, 'Unsafe status input path')
        if not path.exists():
            return None
        try:
            raw = path.read_bytes()
            bindings[name] = hashlib.sha256(raw).hexdigest()
            return json.loads(raw.decode('utf-8-sig'))
        except (OSError, UnicodeError, ValueError) as exc:
            errors.append({'path': name, 'error': repr(exc), 'observation': 'UNREADABLE_UNVERIFIED'})
            return None

    controller = snapshot('controller_completion.json')
    stage = snapshot('controller_stage1_completion.json')
    controller = controller if isinstance(controller, dict) else stage if isinstance(stage, dict) else {}
    exits_file = snapshot('controller_stage1_exits.json')
    exits = exits_file if isinstance(exits_file, list) else controller.get('exits', [])
    if not isinstance(exits, list):
        errors.append({'path': 'controller', 'error': 'Exit collection is not a list'})
        exits = []
    dispatched = controller.get('dispatched', [])
    dispatched = dispatched if isinstance(dispatched, list) else []
    dispatched_names = {item.get('cell') if isinstance(item, dict) else item for item in dispatched
                        if isinstance(item, (dict, str))}
    scheduled_names = {item['cell'] for item in schedule}
    exit_rows = {}
    for entry in exits:
        if not isinstance(entry, dict) or entry.get('cell') not in scheduled_names:
            errors.append({'path': 'controller exits', 'error': 'Unexpected/malformed exit identity', 'entry': entry})
            continue
        exit_rows.setdefault(entry['cell'], []).append(entry)
    unexpected = sorted(str(name) for name in dispatched_names - scheduled_names)
    if unexpected:
        errors.append({'path': 'controller dispatched', 'error': 'Unexpected dispatched identities', 'cells': unexpected})

    def nonnegative(value, *, integer=False):
        good_type = type(value) is int if integer else type(value) in (int, float)
        return value if good_type and math.isfinite(value) and value >= 0 else None

    rows = []
    for spec in schedule:
        cell = spec['cell']
        observed_exits = exit_rows.get(cell, [])
        exit_row = observed_exits[0] if len(observed_exits) == 1 else None
        dispatched_here = cell in dispatched_names or bool(observed_exits) or (run / cell / 'execution.json').exists()
        state = ('AMBIGUOUS_EXIT_RECORD_UNVERIFIED' if len(observed_exits) > 1 else
                 'EXITED_ZERO_UNAUDITED' if exit_row is not None and exit_row.get('exit_code') == 0 else
                 'EXITED_NONZERO_UNVERIFIED' if exit_row is not None else
                 'DISPATCHED_NO_RECORDED_EXIT' if dispatched_here else 'NOT_DISPATCHED_OR_NO_RECORD')
        service = snapshot(cell + '/service_summary.json')
        oracle = snapshot(cell + '/physical/oracle_summary.json')
        service = service if isinstance(service, dict) else {}
        oracle = oracle if isinstance(oracle, dict) else {}
        costs = {'physical_points': nonnegative(oracle.get('points'), integer=True),
                 'numerical_entries': nonnegative(oracle.get('actual_numerical_entries'), integer=True),
                 'logical_points': nonnegative(service.get('logical_points'), integer=True),
                 'requests': nonnegative(service.get('requests'), integer=True),
                 'process_wall_seconds': nonnegative(exit_row.get('process_wall_seconds')) if exit_row is not None else None}
        row = {**deepcopy(spec), 'execution_status': state, 'dispatched_observed': dispatched_here,
               'exit_code_observed': exit_row.get('exit_code') if exit_row is not None else None,
               'cost_observation_status': 'UNVERIFIED_RECORDED_SNAPSHOT', 'recorded_costs': costs,
               'cost_missing_fields': [key for key, value in costs.items() if value is None],
               'quality_pct': None, 'final_energy_kwh': None, 'fully_certified': None,
               'quality_or_certificate_results_read': False, 'independent_scientific_audit_pass': False}
        rows.append(row)
    observations = {}
    for metric in (*COSTS, 'process_wall_seconds'):
        values = [row['recorded_costs'][metric] for row in rows]
        known = [value for value in values if value is not None]
        observations[metric] = {'unverified_observed_sum': math.fsum(known) if known else None,
            'observed_cells': len(known), 'missing_cells': CELLS - len(known), 'scheduled_denominator': CELLS,
            'full_population_total': None, 'full_population_mean': None}
    return rows, {'controller_status_observed': controller.get('status', 'NO_COMPLETION_RECORD'),
        'execution_status_counts': dict(Counter(row['execution_status'] for row in rows)),
        'unverified_cost_observations': observations, 'errors': errors, 'source_bindings': bindings}


def status_report(run, output=None):
    """A separate, non-success export for missing, failed or unaudited tasks."""
    run = Path(run).resolve()
    output = Path(output).resolve() if output else run / 'status_report'
    require(run.is_relative_to(ROOT / 'runs') and output.is_relative_to(ROOT) and output != run
            and not run.is_relative_to(output) and not output.exists(), 'Fresh project status directory required')
    check_analysis_freeze(run)
    cfg, schedule = read(run / 'config.json'), read(run / 'schedule.json')
    require(cfg['study'] == STUDY and cfg['confirmation_statistics'] == STATISTICAL_RULES,
            'Status export must use the same frozen study/statistical contract')
    rows, observed = status_observations(run, schedule)
    pairs = []
    for context in range(CONTEXTS):
        group = {row['arm']: row for row in rows if row['context_id'] == context}
        pairs.append({'context_id': context, 'scenario_id': group['P']['scenario_id'], 'event_id': group['P']['event_id'],
            'P_status': group['P']['execution_status'], 'E_status': group['E']['execution_status'],
            'quality_delta': None, 'PF_ratio': None, 'full_cost_delta': None, 'complete_audited_pair': False})
    summary = {'status': 'INCOMPLETE_OR_UNAUDITED_PUBLIC_CONFIRMATION_NO_ESTIMATES', 'study': STUDY,
        'run': run.name, 'utc': datetime.now(timezone.utc).isoformat(), 'scheduled_cells': CELLS,
        'retained_cells': len(rows), 'retained_pairs': len(pairs), 'statistical_rules': deepcopy(STATISTICAL_RULES),
        'full_population_estimands': {'quality_mean_P': None, 'quality_mean_E': None, 'quality_delta_mean': None,
            'PF_mean_P': None, 'PF_mean_E': None, 'PF_ratio_P_over_E': None, 'PF_saving_pct': None,
            'time_mean_P': None, 'time_mean_E': None, 'time_ratio_P_over_E': None, 'bootstrap_intervals': None},
        'all_quality_and_certificate_values_intentionally_unread': True, 'confirmation_success': False,
        'new_report_physical_points': 0, 'new_training': 0, 'models_used': 0, **observed,
        'limitations': ['Recorded exit and cost snapshots are unverified observations, never complete scientific certificates.',
            'No result.json or phase_result.json is read; all quality and full-population statistics remain null.',
            'Missing/not-started cells are retained, never assigned invented zero cost or quality.']}
    summary['source_bindings'].update({name: sha(run / name) for name in ('config.json', 'schedule.json', 'method_freeze_manifest.json')})
    summary['report_source_sha256'] = {name: sha(ROOT / name) for name in IMPLEMENTATIONS}
    output.mkdir(parents=True)
    for name, value in (('summary.json', summary), ('all_cells.json', rows), ('all_pairs.json', pairs), ('statistics_contract.json', STATISTICAL_RULES)):
        save(output / name, value)
    lines = ['# 公共P/E状态保留：未形成完整核验统计', '',
        '全部48个冻结任务和24对身份保留；这里只列退出及成本记录的未核验观测。未读取质量或证书结果；全人口均值、比值和区间全部为null。', '',
        '| 工况 | 臂 | 状态 | 记录PF点 | 记录完整进程/s |', '|---|---|---|---:|---:|']
    for row in rows:
        value = row['recorded_costs']
        lines.append(f"| c{row['context_id']:02d} | {row['arm']} | {row['execution_status']} | {value['physical_points']} | {value['process_wall_seconds']} |")
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    for name in IMPLEMENTATIONS:
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    save(output / 'completion_manifest.json', {p.relative_to(output).as_posix(): sha(p)
        for p in sorted(output.rglob('*')) if p.is_file()})
    return summary


def markdown(summary, rows):
    def fmt(value):
        return '未知/未估计' if value is None else f'{value:.6f}'
    group = summary['groups']['overall']
    lines = ['# 公共 P / E：冻结4故障×6事件确认对照', '',
        '48个冷任务全部保留并通过独立核验。24个配对来自4个故障与6个事件的完整交叉，不能当作24次独立重复。', '',
        '| 臂 | 最终质量/% | 实际PF点/任务 | 完整冷进程/s |', '|---|---:|---:|---:|']
    for arm in ARMS:
        lines.append('| ' + arm + ' | ' + ' | '.join(fmt(group['arms'][arm]['metrics'][k]['mean']) for k in PRIMARY) + ' |')
    primary = summary['decision']
    interval = summary['crossed_bootstrap']['metrics']['physical_points']['ratio_of_means_P_over_E']['percentile_95_interval']
    lines += ['', '主指标为两臂等权工况PF均值之比，不是逐对节省率的均值。',
        f"P/E PF均值比：{fmt(primary['primary_PF_ratio_P_over_E'])}；节省：{fmt(primary['primary_PF_saving_pct'])}%。",
        '两维簇重抽样95%百分位区间：' + ('未估计' if interval is None else '[' + ', '.join(fmt(x) for x in interval) + ']') + '。',
        f"全部24对完整energy无退化：{primary['all_24_exact_energy_pairs_nondegrading']}；最坏energy差：{fmt(primary['worst_exact_energy_delta_kwh'])} kWh。", '',
        '| 分组 | 配对数 | P−E质量/百分点 | PF均值比P/E | P−E完整时间/s |', '|---|---:|---:|---:|---:|']
    for dimension, key in (('by_fault', 'scenario_id'), ('by_event', 'event_id')):
        for item in summary['groups'][dimension]:
            contrast = item['P_minus_E']
            lines.append(f"| {item[key]} | {item['denominator_pairs']} | {fmt(contrast['quality_pct']['signed_delta']['mean'])} | "
                f"{fmt(contrast['physical_points']['ratio_of_means_P_over_E'])} | {fmt(contrast['process_wall_seconds']['signed_delta']['mean'])} |")
    lines += ['', '| 工况 | 故障 | 事件 | P质量/% | E质量/% | P实际点 | E实际点 | P时间/s | E时间/s | 精确质量关系 |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---|']
    for pair in summary['pairs_for_markdown']:
        p, e = pair['P'], pair['E']
        lines.append(f"| c{pair['context_id']:02d} | {pair['scenario_id']} | {pair['event_id']} | {fmt(p['quality_pct'])} | "
            f"{fmt(e['quality_pct'])} | {fmt(p['physical_points'])} | {fmt(e['physical_points'])} | {fmt(p['process_wall_seconds'])} | "
            f"{fmt(e['process_wall_seconds'])} | {pair['p_relations']['quality_pct']} |")
    lines += ['', '质量胜平负按完整认证energy精确比较，无四舍五入阈值。任何缺失保留为null；对应均值、比值及区间不使用完整案例子集补算。',
        'bootstrap为故障与事件分别有放回重抽后取交叉乘积权重，10000次，seed=20261024；4个故障簇/6个事件簇仍很少，区间仅作探索性不确定性描述。',
        '完整时间包含保底、候选、初始化及记录；每对各一次，不能证明稳定加速。阶段点数归属、缓存命中数与实际物理节省不能混为一谈。',
        '这是冻结范围内的内部P/E确认；未复现外部近邻，不能据此声称超过外部CLPU方法、获得学习收益或保证投稿分区。']
    return '\n'.join(lines) + '\n'


def report(run, output=None):
    run = Path(run).resolve()
    output = Path(output).resolve() if output else run / 'derived_report'
    require(run.is_relative_to(ROOT / 'runs') and output.is_relative_to(ROOT) and output != run
            and not run.is_relative_to(output) and not output.exists(), 'Fresh project report directory required')
    rows, evidence, pools, bindings, audit, controller = load_cases(run)
    groups, pairs, boot, decision = summarize(rows)
    summary = {'status': STATUS, 'study': STUDY, 'run': run.name, 'utc': datetime.now(timezone.utc).isoformat(),
        'scheduled_cells': CELLS, 'executed_cells': CELLS, 'not_run_cells': 0, 'paired_contexts': CONTEXTS,
        'fault_clusters': FAULTS, 'event_clusters': EVENTS, 'statistical_rules': deepcopy(STATISTICAL_RULES),
        'groups': groups, 'crossed_bootstrap': boot, 'decision': decision,
        'total_cost': {key: sum(r[key] for r in rows) for key in COSTS},
        'physical_statuses': dict(sum((Counter(r['physical_statuses']) for r in rows), Counter())),
        'candidate_verification_statuses': dict(Counter(r['verification_status'] for r in evidence)),
        'controller_wall_seconds': controller['wall_seconds'], 'audit_elapsed_seconds': audit['elapsed_seconds'],
        'source_bindings': bindings, 'report_source_sha256': {name: sha(ROOT / name) for name in IMPLEMENTATIONS},
        'historical_times_used': False, 'new_report_physical_points': 0, 'new_training': 0, 'models_used': 0,
        'external_method_superiority_claim': False,
        'limitations': [STATISTICAL_RULES['inference_limit'], STATISTICAL_RULES['time'],
            'P/E compares the shared public mechanism; no external-method or learned-ranker superiority is established.']}
    output.mkdir(parents=True)
    for name, value in (('summary.json', summary), ('all_cells.json', rows), ('all_pairs.json', pairs),
                        ('plan_evidence.json', evidence), ('candidate_pools.json', pools), ('statistics_contract.json', STATISTICAL_RULES)):
        save(output / name, value)
    (output / 'report.md').write_text(markdown({**summary, 'pairs_for_markdown': pairs}, rows), encoding='utf-8')
    for name in IMPLEMENTATIONS:
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    save(output / 'completion_manifest.json', {p.relative_to(output).as_posix(): sha(p)
        for p in sorted(output.rglob('*')) if p.is_file()})
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--print-contract', action='store_true')
    parser.add_argument('--status-only', action='store_true', help='Preserve all 48 statuses/cost observations without reading quality or computing estimates')
    args = parser.parse_args()
    if args.print_contract:
        print(json.dumps(STATISTICAL_RULES, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        require(args.run is not None, '--run required when not printing the no-data statistical contract')
        result = status_report(args.run, args.output) if args.status_only else report(args.run, args.output)
        print(json.dumps({key: result[key] for key in ('status', 'scheduled_cells', 'executed_cells', 'decision', 'retained_cells') if key in result}))
