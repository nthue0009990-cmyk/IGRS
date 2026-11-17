"""Frozen eight-pair cold-process cache pilot; no training or historical answers."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
sys.dont_write_bytecode = True
from d060_validation_common import canonical, normalize_manifest, read, require, sha
from nature34_experiment_support import create_run, freeze, dependencies, append, numeric_lists
from run_nature34_clpu_tail_pilot import save
from run_clpu_counterfactual_pilot import seal, utc
from run_d060_validation import ROOT_INPUTS, bound_copy

STUDY = 'CLPU_EQUIVALENT_CACHE_PILOT_V1'
SOURCE_RUN = '20261027T121023_469270Z_clpu_window_pilot'
SOURCE_SHA = '9967ea771263b5b9bb45835116dec1c5fd70f8908c17ec2348d90d4a6377fc7e'
ARMS = ('semantic_identity', 'electrical_equivalent')
ACTIVE = ROOT / 'artifacts/clpu_equivalent_cache_pilot_active.json'
REVIEW = ROOT / 'artifacts/clpu_equivalent_cache_scope_review_20261030'
EXACT_REVIEW = ROOT / 'artifacts/clpu_equivalent_cache_exact_key_review_20261030'
MIN_FREE, MAX_SECONDS, TOTAL_CAP = 12 * 2**30, 1800, 4096
PROTOCOL = 'docs/clpu_equivalent_cache_protocol_v1.md'
HISTORY_FILES = ('spec.json', 'result.json', 'steps.jsonl.gz', 'plans.jsonl.gz', 'requests.jsonl')


def verify_manifest(directory, filename='completion_manifest.json', required=()):
    directory = Path(directory).resolve()
    raw = read(directory / filename)
    require(isinstance(raw, dict) and bool(raw), 'Nonempty flat file manifest required')
    manifest = normalize_manifest(raw)
    targets = set()
    for name, digest in manifest.items():
        target = (directory / name).resolve()
        require(not Path(name).is_absolute() and not Path(name).drive and target != directory
                and target.is_relative_to(directory), 'Manifest path escaped')
        identity = os.path.normcase(str(target))
        require(identity not in targets, 'Manifest contains duplicate target aliases')
        targets.add(identity)
        require(isinstance(digest, str) and len(digest) == 64
                and all(c in '0123456789abcdef' for c in digest), 'Manifest digest malformed')
        require(target.is_file() and sha(target) == digest, 'Changed sealed input: ' + name)
    require(set(required).issubset(manifest), 'Required frozen input absent')
    return manifest


def schedule_for(contexts):
    require(len(contexts) == 8 and {c['context_id'] for c in contexts} == set(range(8)),
            'All eight original context IDs required')
    require(len({c['scenario_id'] for c in contexts}) == 8
            and Counter(c['event_id'] for c in contexts) == Counter({30: 2, 20: 2, 6: 2, 3: 2}),
            'Original eight-fault balanced event population required')
    ordered = sorted(contexts, key=lambda c: hashlib.sha256(
        ('clpu-equivalent-cache-context-order-v1|' + canonical(c)).encode()).hexdigest())
    rows = []
    for rank, c in enumerate(ordered):
        arms = ARMS if rank % 2 == 0 else tuple(reversed(ARMS))
        for position, arm in enumerate(arms):
            rows.append({**c, 'arm': arm, 'pair_order': rank, 'arm_launch_order': position,
                'method': 'beam', 'window_variant': 'coverage_v1', 'seed': 0,
                'logical_point_budget': 256, 'physical_point_budget': 256,
                'max_expansions': 256, 'phase': 'training_cache_mechanism_evaluation',
                'deterministic': True, 'episode_id': len(rows), 'cell': f'episodes/{len(rows):03d}'})
    return rows


def passed_tests(path):
    cases = list(ET.parse(path).getroot().iter('testcase'))
    require(cases and not any(list(c.iter('failure')) + list(c.iter('error')) + list(c.iter('skipped'))
                             for c in cases), 'Nonempty all-passed software tests required')
    for name in ('test_run_clpu_equivalent_cache', 'test_report_clpu_equivalent_cache',
                 'test_nature34_clpu_equivalent_cache', 'test_clpu_equivalent_cache_audit'):
        require(any(name in c.get('classname', '') for c in cases), 'Missing new component tests: ' + name)
    return len(cases)


def verify_source_cell(source, manifest, spec):
    cell = (source / spec['cell']).resolve()
    require(cell != source and cell.is_relative_to(source), 'Source cell escaped')
    require(sha(source / 'audit/summary.json') == manifest['audit/summary.json'], 'Source audit changed')
    audit = read(source / 'audit/summary.json')
    matches = [r for r in audit['results'] if r['cell'] == spec['cell']]
    require(audit['status'] == 'PASS' and len(matches) == 1 and matches[0]['spec'] == spec,
            'Source cell/spec has no unique passed audit')
    require(sha(cell / 'completion_manifest.json') == matches[0]['cell_completion_sha256'],
            'Source cell seal differs from independent audit')
    cm = normalize_manifest(read(cell / 'completion_manifest.json'))
    for name, digest in cm.items():
        target = (cell / name).resolve()
        require(target != cell and target.is_relative_to(cell), 'Source cell path escaped')
        require(manifest.get(spec['cell'] + '/' + name) == digest, 'Source root/cell manifests disagree')
    for name in (*HISTORY_FILES, 'points.jsonl.gz'):
        require(name in cm and sha(cell / name) == cm[name], 'Source journal changed: ' + name)
    return cell, cm


def source_names():
    names = [p.relative_to(ROOT).as_posix() for p in (ROOT / 'src/dnr').glob('*.py')]
    names += ['scripts/' + n for n in (
        'run_clpu_equivalent_cache_pilot.py', 'audit_clpu_equivalent_cache.py', 'report_clpu_equivalent_cache.py',
        'run_nature34_clpu_pilot.py', 'nature34_experiment_support.py', 'run_nature34_development_training.py',
        'run_clpu_counterfactual_pilot.py', 'run_nature34_clpu_tail_pilot.py', 'run_d060_validation.py',
        'd060_validation_common.py', 'prepare_nature34_clpu_pilot.py', 'report_nature34_imitation.py',
        'audit_d059_supplement.py', 'audit_nature34_clpu_branch.py', 'audit_nature34_clpu_pilot.py',
        'audit_nature34_joint_planning.py', 'audit_clpu_window_pilot.py')]
    names += [p.relative_to(ROOT).as_posix() for pattern in ('*clpu_equivalent_cache*.py', '*clpu_window*.py')
              for p in (ROOT / 'tests').glob(pattern)]
    return sorted(set(names + [PROTOCOL]))


def physical_scope(run, names):
    inputs = [*ROOT_INPUTS, 'config.json', 'selected_scenarios.json', 'selected_events.json',
              'training_contexts.json', 'dependencies.json', 'python_environment.json']
    inputs += [p.relative_to(run).as_posix() for p in (run / 'source_inputs').rglob('*') if p.is_file()]
    return {'scope_version': 'CLPU_FULL_FIXED_PHYSICAL_SCOPE_V1',
        'input_files': {n: sha(run / n) for n in sorted(set(inputs))},
        'science_source_sha256': {n: sha(ROOT / n) for n in names if n.endswith('.py')},
        'source_provenance': read(run / 'source_provenance.json'),
        'runtime_dependencies': read(run / 'dependencies.json'),
        'python_environment': read(run / 'python_environment.json'),
        'physical_config': read(run / 'physical_config.json'),
        'alpha': .5, 'algorithm': 0, 'load_setpoint_precision': '.15g',
        'reset_semantics': 'INDEPENDENT_QUASI_STATIC_AUTHOR_RESET_EACH_SNAPSHOT',
        'cache_scope': 'one cold cell only; no historical or cross-arm/cross-context answers'}


def bind_scope_review(run):
    manifest = verify_manifest(REVIEW, required=('summary.json',))
    review = read(REVIEW / 'summary.json')
    require(review['status'] == 'PASS' and review['new_physical_points'] == review['new_training'] == 0,
            'Full zero-solve observed duplicate/constraint scope review has not passed')
    for name in manifest:
        destination = run / 'scope_review' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REVIEW / name, destination)
    shutil.copyfile(REVIEW / 'completion_manifest.json', run / 'scope_review/completion_manifest.json')
    binding = {'path': REVIEW.relative_to(ROOT).as_posix(),
               'completion_sha256': sha(REVIEW / 'completion_manifest.json'),
               'summary_sha256': sha(REVIEW / 'summary.json')}
    save(run / 'scope_review_binding.json', binding)
    return binding


def validate_exact_review(directory, prior_review_sha):
    required = ('summary.json', *(f'context_{i:02d}.json' for i in range(8)))
    manifest = verify_manifest(directory, required=required)
    review = read(directory / 'summary.json')
    require(review['status'] == 'PASS' and review['source_run'] == SOURCE_RUN
            and review['source_completion_sha256'] == SOURCE_SHA
            and review['prior_review_manifest_sha256'] == prior_review_sha
            and review['new_physical_points'] == review['new_training'] == 0
            and review['physical_points_reviewed'] == 1121 and review['duplicate_pairs_reviewed'] == 556
            and review['new_group_merges'] == 18 and review['certificate_or_feedback_disagreements'] == 0
            and review['all_differences_in_existing_excluded_disabled_cache_fields'] is True
            and review['constraint_or_audit_tolerances_changed'] is False and not review['errors'],
            'Final electrical key lacks a complete, unchanged-source zero-solve review')
    return manifest


def bind_exact_key_review(run):
    prior = read(run / 'scope_review_binding.json')['completion_sha256']
    manifest = validate_exact_review(EXACT_REVIEW, prior)
    for name in manifest:
        destination = run / 'exact_key_review' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(EXACT_REVIEW / name, destination)
    shutil.copyfile(EXACT_REVIEW / 'completion_manifest.json', run / 'exact_key_review/completion_manifest.json')
    binding = {'path': EXACT_REVIEW.relative_to(ROOT).as_posix(),
        'completion_sha256': sha(EXACT_REVIEW / 'completion_manifest.json'),
        'summary_sha256': sha(EXACT_REVIEW / 'summary.json')}
    save(run / 'exact_key_review_binding.json', binding)
    return binding


def verify_exact_key_review(run):
    binding = read(run / 'exact_key_review_binding.json')
    directory = run / 'exact_key_review'
    require(binding['path'] == 'artifacts/clpu_equivalent_cache_exact_key_review_20261030'
            and sha(directory / 'completion_manifest.json') == binding['completion_sha256']
            and sha(directory / 'summary.json') == binding['summary_sha256'], 'Final-key review binding changed')
    return validate_exact_review(directory, read(run / 'scope_review_binding.json')['completion_sha256'])


def prepare(tests_path):
    from dnr.nature34_clpu_branch import BranchConfig
    count = passed_tests(tests_path)
    source = ROOT / 'runs' / SOURCE_RUN
    require(sha(source / 'completion_manifest.json') == SOURCE_SHA, 'Frozen original G1 source changed')
    sm = normalize_manifest(read(source / 'completion_manifest.json'))
    for name in ('config.json', 'population.json', 'schedule.json', 'training_contexts.json',
                 'selected_scenarios.json', 'selected_events.json', 'audit/summary.json'):
        require(sha(source / name) == sm[name], 'Source input changed: ' + name)
    pop = read(source / 'population.json')['contexts']
    contexts = [{'context_id': i, 'scenario_id': c['scenario_id'], 'event_id': c['event_id']}
                for i, c in enumerate(pop)]
    require({(c['scenario_id'], c['event_id']) for c in contexts}
            == {(c['scenario_id'], c['event_id']) for c in read(source / 'training_contexts.json')},
            'Eight training contexts changed')
    schedule = schedule_for(contexts)
    cfg = deepcopy(read(source / 'config.json'))
    require(cfg['branch_search'] == asdict(BranchConfig()), 'Inherited search configuration differs')
    cfg.update(equivalent_cache_study=STUDY)
    cfg['equivalent_cache'] = {'arms': list(ARMS), 'logical_point_budget': 256,
        'physical_point_budget': 256, 'max_expansions': 256, 'max_workers': 2,
        'cooperative_seconds': MAX_SECONDS, 'total_physical_budget': TOTAL_CAP,
        'minimum_free_disk_bytes': MIN_FREE, 'no_training': True,
        'no_historical_answers': True, 'no_cross_cell_cache': True, 'no_automatic_retry': True,
        'dispatch': 'Hashed context order; alternate arm launch order 4/4; drain each two-worker pair before next',
        'semantic_gate': 'Every logical certificate, plan, first failure, step, incumbent and stop matches both arms and source',
        'timing': 'Fresh same-run cold pair comparison; original source timing is not a speed comparator'}
    require(shutil.disk_usage(ROOT).free >= MIN_FREE, 'Insufficient free space')
    run = create_run('clpu_equivalent_cache_pilot', cfg)
    try:
        for name in ROOT_INPUTS:
            bound_copy(source, run / name, sm, name)
        for path in (source / 'source_inputs').rglob('*'):
            if path.is_file():
                relative = path.relative_to(source).as_posix()
                bound_copy(source, run / relative, sm, relative)
        for name in ('population.json', 'selected_scenarios.json', 'selected_events.json', 'training_contexts.json'):
            bound_copy(source, run / name, sm, name)
        shutil.copyfile(tests_path, run / 'software_tests.xml')
        shutil.copyfile(ROOT / PROTOCOL, run / 'protocol.md')
        history = run / 'historical'
        history.mkdir()
        shutil.copyfile(source / 'completion_manifest.json', history / 'source_completion_manifest.json')
        shutil.copyfile(source / 'audit/summary.json', history / 'source_audit_summary.json')
        source_schedule = read(source / 'schedule.json')
        bindings = []
        for c in contexts:
            candidates = [s for s in source_schedule if s['context_id'] == c['context_id']
                          and s['window_variant'] == 'coverage_v1']
            require(len(candidates) == 1, 'Missing/repeated historical G1 context')
            old = candidates[0]
            require(all(old[k] == c[k] for k in ('scenario_id', 'event_id')), 'Historical context mismatch')
            source_cell, cm = verify_source_cell(source, sm, old)
            prefix = f"historical/c{c['context_id']:02d}"
            for name in (*HISTORY_FILES, 'completion_manifest.json'):
                destination = run / prefix / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_cell / name, destination)
            bindings.append({**c, 'source_cell': old['cell'],
                'source_completion_sha256': sha(source_cell / 'completion_manifest.json'),
                'source_files': {n: cm[n] for n in (*HISTORY_FILES, 'points.jsonl.gz')},
                'historical_directory': prefix})
        save(run / 'source_cells.json', bindings)
        save(run / 'source_binding.json', {'run': SOURCE_RUN, 'completion_sha256': SOURCE_SHA})
        bind_scope_review(run)
        bind_exact_key_review(run)
        diagnostic = ROOT / 'artifacts/physical_verification_cost_20261030'
        diagnostic_names = ('analyze.py', 'g1_cost_and_candidate_input_equivalence.json', 'raw_output_difference_detail.json')
        save(run / 'prior_diagnostic_binding.json', {'path': diagnostic.relative_to(ROOT).as_posix(),
            'files': {n: sha(diagnostic / n) for n in diagnostic_names}, 'read_only_prior_observed_training': True})
        save(run / 'authorization.json', {'utc': utc(), 'execution_authorized': True,
            'user_local_date': '2026-10-30', 'user_message': '做吧',
            'scope': 'One frozen eight-pair physical-input-equivalence cache implementation pilot; no training or parameter scan',
            'data_handling': 'Preserve all real outcomes and failures; no old answer reuse; no fabricated or adjusted results'})
        save(run / 'schedule.json', schedule)
        save(run / 'resource_budget.json', {'contexts': 8, 'cells': 16, 'physical_points': TOTAL_CAP,
            'numerical_entries': 16*3+4*TOTAL_CAP, 'workers': 2, 'minimum_free_bytes': MIN_FREE,
            'cooperative_seconds': MAX_SECONDS, 'new_training': 0, 'software_tests': count,
            'free_bytes_at_prepare': shutil.disk_usage(ROOT).free})
        for spec in schedule:
            save(run / spec['cell'] / 'spec.json', spec)
        names = source_names()
        for name in names:
            if 'source/' + name in sm:
                require(sha(ROOT / name) == sm['source/' + name], 'Inherited science changed: ' + name)
        save(run / 'physical_scope.json', physical_scope(run, names))
        freeze(run, names)
        save(ACTIVE, {'status': 'FROZEN_NOT_STARTED', 'run': run.name, 'scheduled': 16, 'completed': 0, 'utc': utc()})
        return run
    except BaseException as exc:
        save(run / 'preparation_error.json', {'error': repr(exc), 'traceback': traceback.format_exc(),
            'new_physical_points': 0, 'new_training': 0, 'utc': utc()})
        raise


def verify_execution(run):
    prepared = read(run / 'prepared.json')
    require(sha(run / 'prephysical_freeze_manifest.json') == prepared['freeze_sha256'], 'Freeze changed')
    verify_manifest(run, 'prephysical_freeze_manifest.json',
                    ('schedule.json', 'physical_scope.json', 'scope_review_binding.json',
                     'exact_key_review_binding.json', 'source_cells.json'))
    for name in prepared['source_files']:
        require(sha(ROOT / name) == sha(run / 'source' / name), 'Executing source changed: ' + name)
    require(dependencies() == read(run / 'dependencies.json'), 'Dependencies changed')
    require(sha(Path(sys.executable)) == read(run / 'python_environment.json')['sha256'], 'Interpreter changed')
    require(read(run / 'config.json')['equivalent_cache_study'] == STUDY, 'Wrong study identity')
    require(physical_scope(run, prepared['source_files']) == read(run / 'physical_scope.json'), 'Physical scope changed')
    rb = read(run / 'scope_review_binding.json')
    require(sha(run / 'scope_review/completion_manifest.json') == rb['completion_sha256']
            and sha(run / 'scope_review/summary.json') == rb['summary_sha256'], 'Scope review binding changed')
    verify_manifest(run / 'scope_review', required=('summary.json',))
    require(read(run / 'scope_review/summary.json')['status'] == 'PASS', 'Scope review has not passed')
    verify_exact_key_review(run)


def checkpoint(search, started):
    best = search.incumbent
    certified = bool(best and best.get('certified_complete') is True
                     and best.get('validation', {}).get('status') == 'COMPLETE')
    return {'search_choices': search._beam_turn, 'logical_points': search.oracle.points,
        'physical_points': search.oracle.physical_points,
        'numerical_entries': search.oracle.factory.stats()['returned']['numerical_solution_entries'],
        'elapsed_seconds': time.perf_counter() - started,
        'quality_pct': 100*search.quality if certified else None, 'fully_certified': certified}


def worker(run, cell):
    from run_nature34_clpu_pilot import make_env
    from dnr.nature34_clpu_branch import BranchConfig
    from dnr.nature34_clpu_window import WindowBranchSearch
    from dnr.nature34_clpu_equivalent_cache import EquivalenceOracle
    started = time.perf_counter()
    spec = read(cell / 'spec.json')
    try:
        require(cell != run and cell.is_relative_to(run), 'Cell outside run')
        require(not (cell / 'worker_started.json').exists(), 'Repeated task forbidden')
        verify_execution(run)
        require(spec in read(run / 'schedule.json') and spec['arm'] in ARMS, 'Unscheduled arm')
        cfg = read(run / 'config.json')
        require(cfg['branch_search'] == asdict(BranchConfig()), 'Search configuration changed')
        execution = read(run / 'execution_started.json')
        deadline = execution['deadline_epoch_seconds']
        require(time.time() < deadline and shutil.disk_usage(ROOT).free >= MIN_FREE, 'Worker dispatch deadline/storage exhausted')
        with (cell / 'worker_started.json').open('x', encoding='utf-8') as stream:
            append(stream, {'pid': os.getpid(), 'utc': utc(), 'spec_sha256': sha(cell / 'spec.json')})
        preflight_seconds = time.perf_counter()-started
        env = make_env(run, spec)
        save(cell / 'initial_state.json', env.initial.to_dict())
        computation_started = time.perf_counter()
        with gzip.open(cell / 'plans.jsonl.gz', 'wt', encoding='utf-8', compresslevel=3) as plans, \
                gzip.open(cell / 'steps.jsonl.gz', 'wt', encoding='utf-8', compresslevel=3) as steps, \
                EquivalenceOracle(run, cell, env, 256, spec['arm'], read(run / 'physical_scope.json')) as oracle:
            oracle.deadline_monotonic = time.perf_counter() + max(0, deadline-time.time())
            runtime_initialization_seconds = time.perf_counter()-computation_started
            search = WindowBranchSearch(env, oracle, BranchConfig(**cfg['branch_search']),
                         record_callback=lambda row: append(plans, numeric_lists(row)))
            checkpoints = [checkpoint(search, computation_started)]
            while not search.done:
                require(time.time() <= deadline and time.perf_counter()-started <= MAX_SECONDS,
                        'Cooperative deadline exceeded; partial evidence retained')
                index = search.rule_action('beam')
                window_ids = [n['id'] for n in search._window()]
                decision = {'transition_id': search._beam_turn, 'status': 'STARTED', 'method': 'beam',
                    'action': index, 'branch_id': window_ids[index], 'window_branch_ids': window_ids,
                    'oracle_before': {'requests': oracle.requests, 'points': oracle.points,
                                      'logical_points': oracle.points, 'physical_points': oracle.physical_points},
                    'elapsed_seconds': time.perf_counter()-computation_started}
                append(steps, decision)
                response = search.step(index)
                decision.update(status='COMPLETED', response=numeric_lists(response),
                    oracle_after={'requests': oracle.requests, 'points': oracle.points,
                                  'logical_points': oracle.points, 'physical_points': oracle.physical_points},
                    elapsed_seconds=time.perf_counter()-computation_started)
                append(steps, decision)
                checkpoints.append(checkpoint(search, computation_started))
            result = {'status': 'INITIAL_FAIL' if search.stop_reason == 'INITIAL_FAIL' else
                'CERTIFIED_INCUMBENT' if search.incumbent is not None else 'NO_CERTIFIED_COMPLETE_PLAN',
                'config': asdict(search.config), 'stop_reason': search.stop_reason,
                'best_plan': deepcopy(search.incumbent), 'initial_certificate': deepcopy(search.initial_certificate),
                'initial_elapsed_seconds': checkpoints[0]['elapsed_seconds'],
                'wait_baseline_quality_pct': checkpoints[0]['quality_pct'], 'checkpoints': checkpoints,
                'completed_expansions': search._beam_turn, 'search_decisions': search._beam_turn,
                'counts': {'generated_candidates': len(search._branch_keys), 'graph_assessed_candidates': len(search._assessed),
                    'checked_candidates': len(search._checked), 'branch_expansions': search._beam_turn,
                    'generation_cap_rejections': search.generation_cap_rejections}}
        result.update(spec=spec, oracle=read(cell / 'equivalent_cache_summary.json'),
            computation_elapsed_seconds=time.perf_counter()-computation_started,
            preflight_elapsed_seconds=preflight_seconds,
            runtime_initialization_seconds=runtime_initialization_seconds,
            elapsed_seconds=time.perf_counter()-started,
            timing_semantics='elapsed includes worker preflight and cold computation through journal close, excludes final result/seal; controller per-cell process wall includes launch, full exit and polling',
            plan_point_id_semantics='logical semantic identity IDs; plan physical_points_before/after retain inherited logical-search counter names',
            training_interactions=0, optimizer_updates=0, new_training=0,
            formal_holdout_numerical_access=False, tune_numerical_access=False)
        save(cell / 'result.json', result)
        seal(cell)
        return 0
    except BaseException as exc:
        save(cell / 'error.json', {'status': 'STOPPED_RELIABILITY_OR_IMPLEMENTATION_ERROR', 'spec': spec,
            'error': repr(exc), 'traceback': traceback.format_exc(), 'elapsed_seconds': time.perf_counter()-started})
        return 1


def costs(run, dispatched):
    logical = points = entries = 0
    unknown = []
    for spec in dispatched:
        try:
            s = read(run / spec['cell'] / 'equivalent_cache_summary.json')
            require(all(type(s[k]) is int and s[k] >= 0 for k in
                        ('logical_points', 'physical_points', 'actual_numerical_entries')), 'Invalid accounting')
            require(s['physical_points'] <= s['logical_points'] <= 256, 'Cell accounting exceeds fixed caps')
            logical += s['logical_points']; points += s['physical_points']; entries += s['actual_numerical_entries']
        except (OSError, ValueError, KeyError, TypeError):
            unknown.append(spec['cell'])
    return {'known_logical_points': logical, 'known_physical_points': points,
            'known_numerical_entries': entries, 'unknown_cost_cells': unknown}


def execute(run):
    require(not (run / 'execution_started.json').exists(), 'Already started; no automatic repeat')
    verify_execution(run)
    schedule = read(run / 'schedule.json')
    contexts = [{k:r[k] for k in ('context_id','scenario_id','event_id')}
                for r in read(run / 'source_cells.json')]
    require(len(schedule) == 16 and schedule == schedule_for(contexts),
        'Frozen balanced schedule changed')
    wall = time.perf_counter()
    save(run / 'execution_started.json', {'utc': utc(), 'pid': os.getpid(),
        'deadline_epoch_seconds': time.time()+MAX_SECONDS})
    completed, dispatched, errors, exits, active = [], [], [], [], {}

    def finished(proc, spec, code, began, drained=False):
        exits.append({'cell': spec['cell'], 'exit_code': code, 'launcher_pid': proc.pid,
            'process_wall_seconds': time.perf_counter()-began,
            'process_wall_semantics': 'Popen call through observed process exit; includes import/preflight/compute/write/seal, polling at most about 0.1s',
            'drained_after_controller_error': drained, 'utc': utc()})
        if code == 0 and (run / spec['cell'] / 'completion_manifest.json').exists():
            completed.append(spec['cell'])
        else:
            errors.append(exits[-1])
    try:
        for rank in range(8):
            if errors:
                break
            if shutil.disk_usage(ROOT).free < MIN_FREE or time.perf_counter()-wall >= MAX_SECONDS:
                errors.append({'error':'LOW_DISK_OR_DEADLINE_STOP_NEW_DISPATCH'}); break
            pair = [s for s in schedule if s['pair_order'] == rank]
            require(len(pair) == 2 and {s['arm'] for s in pair} == set(ARMS), 'Invalid paired dispatch')
            for spec in pair:
                cell = run / spec['cell']
                out = (cell / 'stdout.log').open('w', encoding='utf-8')
                err = (cell / 'stderr.log').open('w', encoding='utf-8')
                began = time.perf_counter()
                try:
                    proc = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
                        'worker', '--run', str(run), '--cell', str(cell)], cwd=ROOT, stdout=out, stderr=err,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                except BaseException:
                    out.close(); err.close(); raise
                active[spec['cell']] = (proc, spec, out, err, began)
                dispatched.append(spec)
            while active:
                for key, (proc, spec, out, err, began) in list(active.items()):
                    code = proc.poll()
                    if code is None:
                        if time.perf_counter()-wall > MAX_SECONDS and not errors:
                            errors.append({'error':'COOPERATIVE_DEADLINE_STOP_DISPATCH_DRAIN_STARTED_WORKERS'})
                        continue
                    out.close(); err.close(); del active[key]
                    finished(proc, spec, code, began)
                progress = {'status':'RUNNING' if not errors else 'DRAINING_AFTER_ERROR', 'run':run.name,
                    'scheduled':16, 'completed':len(completed), 'dispatched':len(dispatched),
                    'active':list(active), 'errors':errors, 'utc':utc(), 'elapsed_seconds':time.perf_counter()-wall}
                save(run / 'controller_progress.json', progress); save(ACTIVE, progress)
                if active:
                    time.sleep(.1)
    except BaseException as exc:
        errors.append({'error':repr(exc), 'traceback':traceback.format_exc()})
    finally:
        for proc, spec, out, err, began in active.values():
            try:
                finished(proc, spec, proc.wait(), began, True)
            except BaseException as exc:
                errors.append({'cell':spec['cell'], 'drain_error':repr(exc)})
            out.close(); err.close()
    accounting = costs(run, dispatched)
    require(accounting['known_physical_points'] <= TOTAL_CAP, 'Total physical cap exceeded')
    summary = {'status':'COMPLETE_CELLS_PENDING_AUDIT' if not errors and not accounting['unknown_cost_cells']
               and len(completed) == 16 else 'INCONCLUSIVE', 'scheduled':16,
        'completed':completed, 'dispatched':dispatched, 'exits':exits, 'errors':errors,
        **accounting, 'wall_seconds':time.perf_counter()-wall, 'utc':utc()}
    save(run / 'controller_completion.json', summary)
    save(ACTIVE, {'status':summary['status'], 'run':run.name, 'scheduled':16,
        'completed':len(completed), **accounting, 'errors':errors, 'utc':utc()})
    return 0 if summary['status'] == 'COMPLETE_CELLS_PENDING_AUDIT' else 1


def finalize(run):
    require(not (run / 'completion_manifest.json').exists(), 'Run already sealed')
    report = read(run / 'derived_report/summary.json')
    audit = read(run / 'audit/summary.json')
    verify_manifest(run / 'audit', required=('summary.json',))
    verify_manifest(run / 'derived_report', required=('summary.json', 'all_cells.json', 'all_pairs.json'))
    require(audit['status'] == 'PASS' and audit['audited_cells'] == 16, 'Independent audit incomplete')
    require(report['source_bindings']['audit/summary.json'] == sha(run / 'audit/summary.json'), 'Report audit binding differs')
    terminal = {'status':'COMPLETE_EQUIVALENT_CACHE_PILOT_AUDITED_AND_REPORTED',
        'scientific_status':report['status'], 'run':run.name, 'utc':utc(),
        'audit_sha256':sha(run / 'audit/summary.json'), 'report_sha256':sha(run / 'derived_report/summary.json'),
        'report_manifest_sha256':sha(run / 'derived_report/completion_manifest.json'),
        'physical_points':audit['physical_points'], 'numerical_entries':audit['numerical_entries'],
        'new_training':0, 'all_valid_failures_preserved':True}
    save(run / 'terminal_status.json', terminal); seal(run)
    save(ACTIVE, {**terminal, 'status':'COMPLETE_EQUIVALENT_CACHE_PILOT_SEALED',
        'completion_sha256':sha(run / 'completion_manifest.json')})
    return terminal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare','worker','execute','finalize'])
    parser.add_argument('--run', type=Path); parser.add_argument('--cell', type=Path)
    parser.add_argument('--tests', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(prepare(args.tests.resolve())); return 0
    run = args.run.resolve()
    require(run.parent == ROOT / 'runs', 'Run outside project')
    if args.command == 'worker':
        return worker(run, args.cell.resolve())
    if args.command == 'finalize':
        print(finalize(run)); return 0
    return execute(run)


if __name__ == '__main__':
    raise SystemExit(main())
