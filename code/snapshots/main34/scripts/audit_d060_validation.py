"""Independent, durable audit of the fixed D060 evaluation-only study.

The D060 population, event pairing, source/model attribution, frozen inputs and
dispatch ledger are rebuilt here, independently of its production controller.
Only the unchanged D059 cell-level scientific replay is reused. No solver or
learner is initialized and no D059 hard-coded parent/population check is called.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
sys.dont_write_bytecode = True

import audit_nature34_clpu_branch as old
from audit_d059_supplement import NoSolverImports, no_solver_loaded, atomic_save, sha, read, utc_now
from nature34_experiment_support import dependencies

SOURCE_RUN = '20261024T073617_269327Z_nature34_clpu_branch'
SOURCE_SHA = '4c3777f938d21f074b40db56c82cd62ca1b956f6aa7d79215832bf192745c56f'
SUPPLEMENT_RUN = '20261025T070549_299770Z_d059_supplement'
SUPPLEMENT_SHA = '879850ffa7b22aa86ea379de535e9bb8caee1db961fe1635c8fe50e56cf19299'
CURVE_ARCHIVE = 'data/raw/literature/clpu_design/20261024T015725_552682Z/hachmann2022_clpu_data.zip'
CURVE_SHA = 'e46886903929d49380a7f1342588adebd26c2524667dbd74fa89021ccea27a22'
EVENT_IDS = [19, 8, 16, 12]
SEEDS = (17, 41, 73)
EXPECTED_CELLS = 384
INHERITED_INPUTS = ('training_scenarios.json', 'author_static_metadata.json', 'load_reference.json',
    'conditions.json', 'physical_config.json', 'operating_reference.json', 'base_load_properties.json',
    'command_inventory.json', 'source_pointer.json', 'metadata.json', 'adaptation.json',
    'author_input_precheck.json', 'condition.json', 'source_provenance.json', 'event_split.json')
REQUIRED_AUDITOR_SOURCES = {'scripts/audit_d060_validation.py', 'scripts/audit_nature34_clpu_branch.py',
    'scripts/audit_nature34_clpu_pilot.py', 'scripts/audit_nature34_joint_planning.py',
    'scripts/audit_nature34_clpu_branch_learning.py', 'scripts/audit_d059_supplement.py',
    'tests/test_audit_d060_validation.py'}
require = old.require


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def inside(base, name):
    base = Path(base).resolve()
    require(isinstance(name, str) and bool(name), 'Invalid evidence path')
    value = (base / name.replace('\\', '/')).resolve()
    require(value != base and value.is_relative_to(base), 'Evidence path escapes its run')
    return value


def manifest(value):
    require(isinstance(value, dict), 'Manifest must be an object')
    result = {}
    for name, digest in value.items():
        require(isinstance(name, str), 'Non-string manifest path')
        key = name.replace('\\', '/')
        require(key not in result and isinstance(digest, str) and len(digest) == 64
                and all(c in '0123456789abcdef' for c in digest), 'Invalid/ambiguous manifest entry')
        result[key] = digest
    return result


def bound(root, seals, name):
    name = name.replace('\\', '/')
    require(name in seals and sha(inside(root, name)) == seals[name], 'Changed/unsealed input: ' + name)
    return read(inside(root, name))


def verify_manifest(root, name):
    values = manifest(read(root / name))
    for relative, digest in values.items():
        require(sha(inside(root, relative)) == digest, 'Sealed input changed: ' + relative)
    return values


def verify_freeze(run):
    require(run.parent == (ROOT / 'runs').resolve(), 'Run outside project runs')
    prepared = read(run / 'prepared.json')
    require(prepared['status'] == 'FROZEN_PREPARED_NO_DSS' and prepared['run'] == run.name,
            'Missing exact prepared identity')
    require(sha(run / 'prephysical_freeze_manifest.json') == prepared['freeze_sha256'], 'Changed freeze identity')
    frozen = verify_manifest(run, 'prephysical_freeze_manifest.json')
    sources = {name.replace('\\', '/') for name in prepared['source_files']}
    require(REQUIRED_AUDITOR_SOURCES <= sources, 'Required independent auditor source was not frozen')
    for name in sources:
        require('source/' + name in frozen and sha(inside(ROOT, name)) == frozen['source/' + name],
                'Executing implementation differs from its frozen copy: ' + name)
    require(dependencies() == bound(run, frozen, 'dependencies.json')
            and sha(Path(sys.executable)) == bound(run, frozen, 'python_environment.json')['sha256'],
            'Frozen dependency/interpreter changed')
    return frozen


def rebuild_schedule(contexts):
    """Independent literal reconstruction, not the runner's build_schedule."""
    require(len(contexts) == len({canonical(c) for c in contexts}) == 32, 'Incomplete context cross')
    require(len({c['scenario_id'] for c in contexts}) == 8 and {c['event_id'] for c in contexts} == set(EVENT_IDS)
            and all(set(c) == {'scenario_id', 'event_id'} and c['scenario_id'].startswith('N2__') for c in contexts),
            'Expected exactly eight development N2 faults by four prescribed new events')
    rows = []
    for context in contexts:
        for method in ('greedy', 'beam', 'supervised', 'bc_raw', 'ppo_raw'):
            for seed in ((0,) if method in ('greedy', 'beam') else SEEDS):
                rows.append({**context, 'phase': 'evaluation', 'group': 'evaluation', 'method': method,
                    'seed': seed, 'variant': 'main', 'point_budget': 256,
                    'representation': 'associated' if method == 'supervised' else 'raw', 'deterministic': True})
        rows.append({**context, 'phase': 'evaluation', 'group': 'evaluation', 'method': 'beam', 'seed': 0,
            'variant': 'beam_exp1024', 'point_budget': 256, 'representation': 'raw', 'deterministic': True})
    rows.sort(key=lambda r: hashlib.sha256(('d060-cell-order-v1|' + canonical(r)).encode()).hexdigest())
    for index, row in enumerate(rows):
        row.update(episode_id=index, cell=f'episodes/{index:04d}')
    require(len(rows) == 384 and sum(r['point_budget'] for r in rows) == 98304, 'Allocation/population differs')
    return rows


def validate_events(events, split, archive):
    ordered = sorted(range(1, 32), key=lambda n: hashlib.sha256(f'clpu-event-split-v1|{n}'.encode()).hexdigest())
    expected = {'train': ordered[:19], 'dev': ordered[19:25], 'holdout': ordered[25:]}
    require(split == expected and split['dev'][2:] == EVENT_IDS, 'Whole-event partition changed')
    require([e['event_id'] for e in events] == EVENT_IDS, 'New development event population changed')
    require(sha(archive) == CURVE_SHA, 'Original CLPU ZIP changed')
    with zipfile.ZipFile(archive) as stream:
        rows = [r.strip() for r in stream.read('parameters.csv').decode('utf-8-sig').splitlines() if r.strip()]
    require(len(rows) == 32 and rows[0].startswith('ED model, parameter a in p.u.'), 'Raw CLPU parameter schema changed')
    for event in events:
        row = rows[event['event_id']]
        tokens = [p.strip() for p in row.split(',')]
        require(len(tokens) in (9, 10) and event['parameter_row'] == row, 'Paired raw CLPU event row changed')
        for key, index in (('a', 0), ('tau_seconds', 2), ('ded_a', 4), ('ded_tau_seconds', 6)):
            value = float(tokens[index] + '.' + tokens[index + 1])
            require(math.isfinite(value), 'Nonfinite CLPU source parameter')
            old.close(event[key], value, 'Original paired ' + key)
        delay = 0.0 if len(tokens) == 9 and tokens[8] == '0' else float('.'.join(tokens[8:]))
        old.close(event['ded_delay_seconds'], delay, 'Original paired DED delay')
        require(event['a'] >= 0 and event['tau_seconds'] > 0 and event['ded_a'] >= 0
                and event['ded_tau_seconds'] > 0 and delay >= 0, 'Invalid source parameter domain')


def validate_inherited_learning(study, source, sm, supplement, am):
    summary = bound(source, sm, 'summary.json')
    validation = bound(supplement, am, 'learning_evidence_validation.json')
    require(read(study / 'learning_evidence_validation.json') == validation
            and validation['status'] == 'VERIFIED_EXISTING_LEARNING_AUDIT_BINDINGS', 'Changed inherited learning validation')
    records = validation['records']
    require(len(records) == 49 and len({r['verification'].replace('\\', '/') for r in records}) == 49,
            'Expected 49 unique old learning audits')
    expected_entries = [r for r in summary['audits'] if r['status'] in
        ('VERIFIED_CLPU_BRANCH_INITIAL_LEARNING', 'VERIFIED_CLPU_BRANCH_PPO_UPDATE')]
    require({(r['verification'].replace('\\', '/'), r['sha256']) for r in records} ==
            {(r['verification'].replace('\\', '/'), r['sha256']) for r in expected_entries}, 'Learning audit membership differs')
    initial, seen, updates = 0, set(), 0
    for row in records:
        name = row['verification'].replace('\\', '/')
        item = bound(source, sm, name)
        require(row['sha256'] == sm[name] and sha(inside(study / 'inherited_learning_audits', name)) == sm[name],
                'Copied learning audit does not bind the original')
        require(item['status'] == 'SAVED_LEARNING_AUDIT_COMPLETE', 'Old audit not saved complete')
        out = item['output']
        require(out['errors'] == [] and out['new_powerflow_calls'] == out['new_training_interactions'] == 0,
                'Old audit failed or added research tasks')
        if out['status'] == 'VERIFIED_CLPU_BRANCH_INITIAL_LEARNING':
            initial += 1
            require({(v['seed'], v['representation']) for v in out['fits']} ==
                    {(s, r) for s in SEEDS for r in ('raw', 'associated')}
                    and len(out['fits']) == 6 and all(v['exact_optimizer_replay']
                        and v['actor_unchanged_by_critic_fit'] and v['bc_updates'] == 1000
                        and v['critic_updates'] == 250 for v in out['fits']), 'Initial BC/critic attribution differs')
            require(len(out['supervised_fits']) == 3 and {v['seed'] for v in out['supervised_fits']} == set(SEEDS)
                    and all(v['exact_optimizer_replay'] and v['optimizer_updates'] == 500 for v in out['supervised_fits']),
                    'Initial supervised attribution differs')
        else:
            require(out['status'] == 'VERIFIED_CLPU_BRANCH_PPO_UPDATE' and out['exact_optimizer_replay'] is True,
                    'Unknown/non-exact old learning audit')
            key = (out['seed'], out['representation'], out['round_index'])
            require(key not in seen, 'Repeated inherited PPO audit')
            seen.add(key); updates += out['optimizer_updates']
    require(initial == 1 and seen == {(s, r, n) for s in SEEDS for r in ('raw', 'associated') for n in range(8)},
            'Incomplete inherited learning attribution')
    require(updates == summary['learning_counts']['ppo_optimizer_updates'] == 1448
            and summary['learning_counts']['completed_ppo_rounds'] == 48, 'Historical learning cost omitted')
    return records


def validate_models(study, children, source, sm):
    endpoints = bound(source, sm, 'final_checkpoints.json')
    expected = []
    for seed in SEEDS:
        for method in ('supervised', 'bc_raw', 'ppo_raw'):
            suffix = {'supervised': 'supervised/final.pt', 'bc_raw': 'raw/start.pt', 'ppo_raw': 'raw/ppo_round07/after.pt'}[method]
            name = f'learning/seed{seed}/' + suffix
            require(endpoints[str(seed)][method].replace('\\', '/') == name and name in sm
                    and sha(inside(source, name)) == sm[name], 'Wrong final source endpoint')
            expected.append({'seed': seed, 'method': method, 'source_relative': name, 'source_sha256': sm[name],
                'checkpoint': f'models/seed{seed}/{method}.pt',
                'learning_audit_binding': 'all 49 sealed source learning audits and final attribution'})
    require(read(study / 'model_bindings.json') == expected, 'Nine endpoint bindings differ')
    main = children['main']
    require(read(main / 'model_bindings.json') == expected and read(children['beam_exp1024'] / 'model_bindings.json') == [],
            'Backend model population differs')
    for model in expected:
        require(sha(inside(main, model['checkpoint'])) == model['source_sha256'], 'Copied endpoint changed')
    return expected


def validate_dispatch(study, schedule, children):
    seals = verify_manifest(study, 'evaluation_completion_manifest.json')
    require({'evaluation_status.json', 'dispatch_order.jsonl', 'cell_seals.json', 'cost_accounting.json'} <= seals.keys(), 'Incomplete evaluation dispatch seal')
    state = read(study / 'evaluation_status.json')
    require(state['status'] == 'EVALUATION_COMPLETE_PENDING_AUDIT' and state['completed'] == 384
            and state['scheduled'] == 384 and not state['active_workers'] and state['errors'] == []
            and state['new_training_tasks'] == state['new_optimizer_updates'] == 0, 'Evaluation is incomplete or still writing')
    events = [json.loads(line) for line in (study / 'dispatch_order.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
    starts = [r for r in events if r['event'] == 'STARTED']
    exits = [r for r in events if r['event'] == 'EXITED']
    require(len(events) == len(starts) + len(exits) == 768 and
            [r['episode_id'] for r in starts] == [r['episode_id'] for r in schedule]
            and len({r['episode_id'] for r in exits}) == 384, 'Actual dispatch order/count changed or tasks repeated')
    active, known = {}, {r['episode_id']: r for r in schedule}
    for event in events:
        index = event['episode_id']; row = known[index]
        require(event['backend_run'] == row['backend_run'] and event['cell'] == row['cell'], 'Dispatch identity differs')
        if event['event'] == 'STARTED':
            require(index not in active and len(active) < 4, 'Too many/repeated live workers')
            active[index] = event['pid']
        else:
            require(active.pop(index, None) == event['pid'] and event['returncode'] == 0, 'Worker exit/error ledger differs')
    require(not active, 'Audit overlaps active worker')
    starts_by_id = {r['episode_id']: r for r in starts}
    cell_seals = read(study / 'cell_seals.json')
    expected_cell_keys = {r['backend_run'] + '/' + r['cell'] for r in schedule}
    require(set(cell_seals) == expected_cell_keys, 'Cell seal population differs')
    point_total = numerical_total = 0
    expected_cost_rows = []
    for row in schedule:
        index = row['episode_id']; child = children[row['variant']]; cell = inside(child, row['cell'])
        dispatch_name, complete_name = f'dispatched/{index:04d}.json', f'completed/{index:04d}.json'
        dispatch, complete = bound(study, seals, dispatch_name), bound(study, seals, complete_name)
        require(dispatch['template'] == complete['template'] == row and dispatch['status'] == 'DISPATCHED'
                and dispatch['watchdog_seconds'] == 3600 and dispatch['pid'] == starts_by_id[index]['pid'],
                'Dispatch template/watchdog/PID changed')
        began, deadline = dispatch['spec_created_epoch'], dispatch['deadline_utc_epoch']
        require(type(began) in (int, float) and type(deadline) in (int, float) and
                math.isfinite(began) and math.isfinite(deadline) and deadline - began == 3600,
                'Worker watchdog is not 3600 seconds after task specification')
        spec, worker = read(cell / 'spec.json'), read(cell / 'worker_started.json')
        worker_epoch = datetime.fromisoformat(worker['utc']).timestamp()
        require(spec['deadline_utc_epoch'] == deadline and began <= worker_epoch < deadline
                and worker['pid'] == dispatch['pid'] and worker['spec_sha256'] == dispatch['spec_sha256'] == sha(cell / 'spec.json'),
                'Worker startup/spec does not match the real dispatch')
        digest = sha(cell / 'completion_manifest.json')
        require(cell_seals[row['backend_run'] + '/' + row['cell']] == complete['cell_completion_sha256'] == digest,
                'Worker result seal differs from completion bookkeeping')
        require(complete.get('worker_returncode', 0) == 0, 'Failed worker result was adopted')
        point_total += complete['physical_points']; numerical_total += complete['numerical_entries']
        oracle, stats = read(cell / 'oracle_summary.json'), read(cell / 'command_stats.json')
        require(complete['physical_points'] == oracle['points'] and
                complete['numerical_entries'] == stats['returned']['numerical_solution_entries']
                and stats['requested'] == stats['returned'], 'Completion accounting differs from worker evidence')
        expected_cost_rows.append({'episode_id': index, 'backend_run': row['backend_run'], 'cell': row['cell'],
            'allocated_points': 256, 'worker_completion_present': True, 'worker_error_present': False,
            'physical_points': oracle['points'], 'physical_statuses': oracle.get('statuses', {}),
            'requested_numerical_entries': stats['requested']['numerical_solution_entries'],
            'returned_numerical_entries': stats['returned']['numerical_solution_entries'], 'numerical_counters_closed': True})
    require(point_total == state['completed_physical_points'] <= 98304 and
            numerical_total == state['completed_numerical_entries'] <= 394368, 'Recorded global cost is not closed')
    cost = bound(study, seals, 'cost_accounting.json')
    require(cost['status'] == 'CLOSED_D060_COSTS' and cost['errors'] == [] and cost['tasks'] == expected_cost_rows
            and cost['started_tasks'] == cost['scheduled_tasks'] == 384 and cost['allocated_started_points'] == 98304
            and cost['maximum_physical_points'] == 98304 and cost['returned_physical_points'] == point_total
            and cost['returned_numerical_entries'] == numerical_total and cost['new_training_tasks'] == 0
            and cost['new_optimizer_updates'] == 0 and cost['budget_reset'] is False and cost['all_failures_retained'] is True,
            'Evaluation cost accounting is incomplete or omits failures')
    return {'evaluation_completion_sha256': sha(study / 'evaluation_completion_manifest.json'),
        'dispatch_order_sha256': seals['dispatch_order.jsonl'], 'cell_seals_sha256': seals['cell_seals.json'],
        'real_started_tasks': len(starts), 'real_exited_tasks': len(exits), 'physical_points': point_total,
        'numerical_entries': numerical_total, 'watchdog_seconds': 3600}


def check_parent(study, *, completed=True):
    """D060-only parent validation. Never call the old D059 check_parent."""
    study = Path(study).resolve(); sf = verify_freeze(study)
    config = bound(study, sf, 'study_config.json')
    for key, value in {'study': 'D060', 'point_budget': 256, 'variants': {'main': 256, 'beam_exp1024': 1024},
        'worker_deadline_seconds': 3600, 'max_workers': 4, 'max_physical_points': 98304,
        'new_training_tasks': 0, 'new_optimizer_updates': 0, 'event_ids': EVENT_IDS, 'seeds': list(SEEDS),
        'methods': ['greedy', 'beam', 'supervised', 'bc_raw', 'ppo_raw'],
        'main_cells': 352, 'pressure_cells': 32, 'total_cells': 384}.items():
        require(config.get(key) == value, 'D060 setting changed: ' + key)
    require(config.get('hard_total_deadline_seconds') is None and config.get('automatic_retry', False) is False,
            'Unapproved deadline/retry policy')
    for key in ('formal_holdout_numerical_access', 'tune_numerical_access', 'clpu_holdout_access'):
        require(config.get(key, False) is False, 'Forbidden holdout access')
    source, supplement = ROOT / 'runs' / SOURCE_RUN, ROOT / 'runs' / SUPPLEMENT_RUN
    require(sha(source / 'completion_manifest.json') == SOURCE_SHA and
            sha(supplement / 'completion_manifest.json') == SUPPLEMENT_SHA, 'D059 source identity changed')
    sm, am = manifest(read(source / 'completion_manifest.json')), manifest(read(supplement / 'completion_manifest.json'))
    source_prepared = bound(source, sm, 'prepared.json')
    require(bound(source, sm, 'dependencies.json') == dependencies() and
            bound(source, sm, 'python_environment.json')['sha256'] == sha(Path(sys.executable)),
            'Original scientific dependency/interpreter differs')
    inherited_code = {str(n).replace('\\', '/') for n in source_prepared['source_files'] if str(n).endswith('.py')}
    for name in inherited_code:
        require('source/' + name in sm and sha(inside(ROOT, name)) == sm['source/' + name],
                'Inherited scientific implementation changed: ' + name)
    binding = bound(study, sf, 'source_binding.json')
    require(binding == {'source_run': SOURCE_RUN, 'source_completion_sha256': SOURCE_SHA,
        'source_status_retained': 'INCONCLUSIVE', 'supplement_run': SUPPLEMENT_RUN,
        'supplement_completion_sha256': SUPPLEMENT_SHA, 'old_failures_preserved': True,
        'new_training_tasks': 0, 'new_optimizer_updates': 0}, 'D059 inherited source binding differs')
    for name in ('final_attribution.json', 'verification.json', 'resource_review.json'):
        original = bound(supplement, am, name)
        require(sha(study / 'source_supplement' / name) == am[name], 'Inherited supplementary evidence changed')
        if name == 'final_attribution.json':
            require(original['status'] == 'VERIFIED_CLPU_BRANCH_LEARNING_ATTRIBUTION' and original['errors'] == []
                    and original['evaluation_cells'] == 240 and original['same_seed_bc_ppo_start'] is True,
                    'D059 final attribution incomplete')
        elif name == 'verification.json':
            require(original['status'] == 'VERIFIED_CLPU_BRANCH_CELLS' and original['errors'] == []
                    and original['completed'] == original['cells'] == 272, 'D059 supplementary audit incomplete')
        else:
            require(original['status'] == 'VERIFIED_SUPPLEMENTARY_RESOURCE_ACCOUNTING', 'D059 resource exception unresolved')
    names = bound(study, sf, 'children.json')
    require(set(names) == {'main', 'beam_exp1024'} and len(set(names.values())) == 2, 'Backend population differs')
    children = {variant: inside(ROOT / 'runs', name) for variant, name in names.items()}
    require(all(child != study and child.parent == ROOT / 'runs' for child in children.values()), 'Invalid backend run')
    faults16 = bound(source, sm, 'training_scenarios.json')
    ordered = sorted(faults16, key=lambda f: hashlib.sha256(('clpu-pilot-faults-v1|' + f['scenario_id']).encode()).hexdigest())
    require(len(ordered) == len({f['scenario_id'] for f in ordered}) == 16
            and all(f['partition'] == 'build' and f['fault_order'] == 2 and len(set(f['fault_lines'])) == 2 for f in ordered),
            'Original source fault population differs')
    faults = ordered[8:]
    require(faults == bound(source, sm, 'evaluation_scenarios.json'), 'D059 eight evaluation faults changed')
    old_contexts = bound(source, sm, 'evaluation_contexts.json')
    require(len(old_contexts) == 16 and {canonical(c) for c in old_contexts} ==
            {canonical({'scenario_id': f['scenario_id'], 'event_id': e}) for f in faults for e in (21, 25)},
            'New-event validation overlaps or changes the original development context definition')
    contexts = [{'scenario_id': f['scenario_id'], 'event_id': e} for f in faults for e in EVENT_IDS]
    require(bound(study, sf, 'evaluation_contexts.json') == contexts, 'D060 context population changed')
    expected = rebuild_schedule(contexts)
    schedule = [{**r, 'backend_run': names[r['variant']]} for r in expected]
    require(bound(study, sf, 'schedule.json') == schedule, 'D060 scheduled population/order changed')
    expected_resource = {'episodes': 384, 'main_episodes': 352, 'beam_exp1024_episodes': 32,
        'allocated_physical_points': 98304, 'maximum_numerical_entries': 394368,
        'hard_total_deadline_seconds': None, 'worker_watchdog_seconds': 3600, 'new_training_tasks': 0,
        'new_optimizer_updates': 0, 'automatic_retry': False}
    require(bound(study, sf, 'resource_budget.json') == expected_resource, 'D060 resource allocation differs')
    source_config = bound(source, sm, 'config.json'); configs, child_seals = {}, {}
    require(config['criteria'] == source_config['criteria'] and
            config['bootstrap_repetitions'] == source_config['criteria']['bootstrap_replicates'] == 10000 and
            config['bootstrap_seed'] == source_config['criteria']['bootstrap_seed'] == 20261024,
            'Prespecified thresholds/statistical settings differ from D059')
    for variant, child in children.items():
        cf = verify_freeze(child)
        cfg = bound(child, cf, 'config.json'); target = deepcopy(source_config)
        target.update(validation_study='D060', variant=variant)
        target['branch_search']['max_branch_expansions'] = 256 if variant == 'main' else 1024
        require(cfg == target, 'Backend changed more than the declared Beam cap')
        configs[variant] = cfg
        require(bound(child, cf, 'study_binding.json') == {'study_run': study.name, 'variant': variant,
            'source_run': SOURCE_RUN, 'source_completion_sha256': SOURCE_SHA,
            'supplement_run': SUPPLEMENT_RUN, 'supplement_completion_sha256': SUPPLEMENT_SHA,
            'config_backend_decision_is_compatibility_only': True, 'new_training_tasks': 0},
            'Backend study/source identity differs')
        for name in INHERITED_INPUTS:
            bound(source, sm, name)
            require(sha(child / name) == sm[name], 'Changed inherited scientific input: ' + name)
        wanted_inputs = {k: v for k, v in sm.items() if k.startswith('source_inputs/')}
        found_inputs = {p.relative_to(child).as_posix(): sha(p) for p in (child / 'source_inputs').rglob('*') if p.is_file()}
        require(found_inputs == wanted_inputs, 'Inherited complete source input tree differs')
        require(bound(child, cf, 'selected_scenarios.json') == faults and
                bound(child, cf, 'evaluation_contexts.json') == contexts and bound(child, cf, 'training_contexts.json') == []
                and bound(child, cf, 'schedule.json') == [r for r in expected if r['variant'] == variant],
                'Backend fault/event/schedule population differs')
        events = bound(child, cf, 'selected_events.json')
        validate_events(events, bound(child, cf, 'event_split.json'), ROOT / CURVE_ARCHIVE)
        if completed:
            child_seals[variant] = manifest(read(child / 'completion_manifest.json'))
            expected_cells = {r['cell'] for r in expected if r['variant'] == variant}
            actual_cells = {p.relative_to(child).as_posix() for p in (child / 'episodes').iterdir() if p.is_dir()}
            require(actual_cells == expected_cells and all(p.is_dir() for p in (child / 'episodes').iterdir()),
                    'Unscheduled/missing backend task directory')
            require(all(r['cell'] + '/completion_manifest.json' in child_seals[variant] for r in expected if r['variant'] == variant),
                    'Backend did not seal every evaluation task')
            require(sha(child / 'prephysical_freeze_manifest.json') == child_seals[variant]['prephysical_freeze_manifest.json'],
                    'Completed backend freeze binding changed')
        else:
            require(not (child / 'episodes').exists(), 'Input preflight is valid only before any scientific task')
    learning = validate_inherited_learning(study, source, sm, supplement, am)
    models = validate_models(study, children, source, sm)
    dispatch = validate_dispatch(study, schedule, children) if completed else {'status': 'NOT_STARTED'}
    evidence = {'status': 'VERIFIED_D060_PARENT_AND_MODEL_ATTRIBUTION' if completed else 'PREPARED_D060_INPUTS_VALIDATED', 'errors': [],
        'source_run': SOURCE_RUN, 'source_completion_sha256': SOURCE_SHA,
        'supplement_run': SUPPLEMENT_RUN, 'supplement_completion_sha256': SUPPLEMENT_SHA,
        'study_freeze_sha256': sha(study / 'prephysical_freeze_manifest.json'),
        'schedule_sha256': sha(study / 'schedule.json'), 'contexts_sha256': sha(study / 'evaluation_contexts.json'),
        'dependencies_sha256': sha(study / 'dependencies.json'), 'python_sha256': sha(Path(sys.executable)),
        'children': names, 'child_freeze_sha256': {v: sha(c / 'prephysical_freeze_manifest.json') for v, c in children.items()},
        'child_completion_sha256': {v: sha(c / 'completion_manifest.json') for v, c in children.items()} if completed else {},
        'model_bindings': models, 'learning_audit_bindings': learning, 'dispatch': dispatch,
        'historical_ppo_rounds': 48, 'historical_ppo_optimizer_updates': 1448, 'new_training_tasks': 0,
        'new_optimizer_updates': 0, 'new_powerflow_calls': 0, 'contexts': contexts,
        'cells': 384, 'main_cells': 352, 'beam_exp1024_cells': 32,
        'formal_holdout_numerical_access': False, 'tune_numerical_access': False, 'clpu_holdout_access': False}
    return schedule, children, configs, child_seals, evidence


def cell_bindings(child, row, source_seals):
    cell = inside(child, row['cell'])
    seal_name = row['cell'] + '/completion_manifest.json'
    local = bound(child, source_seals, seal_name)
    values = {seal_name: source_seals[seal_name]}
    for name, digest in manifest(local).items():
        inside(cell, name)
        key = row['cell'] + '/' + name
        require(source_seals.get(key) == digest, 'Backend/cell seal disagreement')
        values[key] = digest
    spec = read(cell / 'spec.json')
    if row['method'] in ('greedy', 'beam'):
        require(not spec.get('checkpoint') and not spec.get('checkpoint_sha256'), 'Rule has a model endpoint')
    else:
        endpoints = [r for r in read(child / 'model_bindings.json')
                     if r['seed'] == row['seed'] and r['method'] == row['method']]
        require(len(endpoints) == 1 and spec.get('checkpoint') == endpoints[0]['checkpoint']
                and spec.get('checkpoint_sha256') == endpoints[0]['source_sha256'],
                'Evaluation used a different method/seed endpoint')
        key = inside(child, spec['checkpoint']).relative_to(child).as_posix()
        require(source_seals.get(key) == spec['checkpoint_sha256'], 'Endpoint not bound by backend seal')
        values[key] = spec['checkpoint_sha256']
    return values


def verify_bindings(child, values):
    for name, digest in values.items():
        require(sha(inside(child, name)) == digest, 'Audited input changed: ' + name)


def _audit_study_locked(study_path):
    study = Path(study_path).resolve()
    require(study.parent == (ROOT / 'runs').resolve(), 'Study outside project runs')
    output = study / 'audit'; output.mkdir(exist_ok=True)
    if (output / 'audit_completion_manifest.json').exists():
        verify_manifest(output, 'audit_completion_manifest.json')
        result = read(output / 'verification.json')
        require(result['status'] == 'VERIFIED_D060_VALIDATION' and result['completed'] == EXPECTED_CELLS,
                'False prior audit completion')
        # Verify source/population/dispatch again, without replaying science.
        schedule, children, _, child_seals, evidence = check_parent(study)
        require(evidence == read(output / 'parent_evidence.json'), 'Completed audit input identity changed')
        for row in schedule:
            child = children[row['variant']]
            verify_bindings(child, cell_bindings(child, row, child_seals[row['variant']]))
        result['completion_sha256'] = sha(output / 'audit_completion_manifest.json')
        return result
    invocation = output / 'invocations' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_' + uuid.uuid4().hex[:8])
    invocation.mkdir(parents=True)
    began = time.perf_counter(); checks, reused = [], 0
    result = {'status': 'FAILED_D060_VALIDATION', 'errors': [], 'study_run': study.name, 'completed': 0,
        'cells': EXPECTED_CELLS, 'new_training_tasks': 0, 'new_training_interactions': 0,
        'optimizer_updates': 0, 'new_powerflow_calls': 0, 'formal_holdout_numerical_access': False,
        'tune_numerical_access': False, 'clpu_holdout_access': False}
    atomic_save(invocation / 'started.json', {'utc': utc_now(), 'pid': os.getpid(), 'study_run': study.name})
    guard = NoSolverImports(); sys.meta_path.insert(0, guard)
    try:
        no_solver_loaded()
        for prior in (output / 'invocations').glob('*/result.json'):
            require(read(prior)['status'] != 'FAILED_D060_VALIDATION', 'Earlier reliability failure retained; automatic retry prohibited')
        schedule, children, configs, child_seals, evidence = check_parent(study)
        for name, value in [('parent_evidence.json', evidence), ('audit_identity.json', {
            'parent_evidence': evidence, 'implementation': 'independent D060 parent plus unchanged D059 check_cell',
            'auditor_sha256': sha(Path(__file__)), 'check_cell_module_sha256': sha(Path(old.__file__))})]:
            path = output / name
            if path.exists():
                require(read(path) == value, 'Audit resume identity changed')
            else:
                atomic_save(path, value)
        identity = sha(output / 'audit_identity.json')
        expected_directories = {f'{i:04d}_{r["variant"]}' for i, r in enumerate(schedule)}
        if (output / 'per_cell').exists():
            require(all(p.is_dir() and p.name in expected_directories for p in (output / 'per_cell').iterdir()),
                    'Unexpected durable audit population')
        prefix = True
        for index, row in enumerate(schedule):
            child = children[row['variant']]; cell = inside(child, row['cell'])
            directory = output / 'per_cell' / f'{index:04d}_{row["variant"]}'
            inputs = cell_bindings(child, row, child_seals[row['variant']])
            if (directory / 'seal.json').exists():
                require(prefix, 'Saved audit cells are not a contiguous prefix')
                seal = read(directory / 'seal.json')
                require(set(seal) == {'record.json'} and sha(directory / 'record.json') == seal['record.json'], 'Saved audit record changed')
                record = read(directory / 'record.json')
                require(record['status'] == 'VERIFIED_D060_CELL' and record['errors'] == [] and record['index'] == index
                        and record['template'] == row and record['audit_identity_sha256'] == identity
                        and record['input_sha256'] == inputs, 'Saved audit binding changed')
                verify_bindings(child, inputs); checked = record['check']; reused += 1
            else:
                prefix = False
                if directory.exists():
                    preserved = directory / ('unsealed_' + uuid.uuid4().hex); preserved.mkdir()
                    for path in list(directory.iterdir()):
                        if path.is_file():
                            os.replace(path, preserved / path.name)
                check_began = time.perf_counter()
                checked = old.check_cell(child, cell, configs[row['variant']])
                no_solver_loaded(); verify_bindings(child, inputs)
                require(checked['cell_completion_sha256'] == inputs[row['cell'] + '/completion_manifest.json'], 'Scientific audit returned different input identity')
                checked.update(index=index, variant=row['variant'], backend_run=row['backend_run'])
                record = {'status': 'VERIFIED_D060_CELL', 'errors': [], 'index': index, 'template': row,
                    'audit_identity_sha256': identity, 'input_sha256': inputs, 'check': checked,
                    'completed_utc': utc_now(), 'elapsed_seconds': time.perf_counter() - check_began,
                    'new_powerflow_calls': 0, 'new_training_tasks': 0, 'optimizer_updates': 0}
                atomic_save(directory / 'record.json', record)
                atomic_save(directory / 'seal.json', {'record.json': sha(directory / 'record.json')})
            checks.append(checked); result['completed'] = len(checks)
            atomic_save(output / 'progress.json', {'status': 'AUDITING_D060_EXISTING_LOGS', 'completed': len(checks),
                'scheduled': len(schedule), 'last_cell': row, 'reused_sealed_cells': reused, 'errors': [],
                'elapsed_seconds_this_invocation': time.perf_counter() - began, 'updated_utc': utc_now()})
        require(len(checks) == EXPECTED_CELLS == 384, 'Incomplete audit cannot be VERIFIED')
        require(all(sha(children[v] / 'completion_manifest.json') == evidence['child_completion_sha256'][v] for v in children),
                'Backend seal changed during audit')
        point_total = sum(r['points'] for r in checks)
        numerical_total = sum(r['commands']['numerical_entries'] for r in checks)
        require(point_total == evidence['dispatch']['physical_points'] and numerical_total == evidence['dispatch']['numerical_entries'],
                'Scientific costs differ from dispatch ledger')
        atomic_save(output / 'cell_checks.json', checks)
        result.update(status='VERIFIED_D060_VALIDATION', episodes=len(checks), main_cells=352, beam_exp1024_cells=32,
            physical_points=point_total, numerical_entries=numerical_total,
            physical_statuses=dict(sum((Counter(r['physical_statuses']) for r in checks), Counter())),
            plan_statuses=dict(sum((Counter(r['plan_statuses']) for r in checks), Counter())),
            group_counts=dict(Counter(r['variant'] for r in checks)),
            model_attribution_status='VERIFIED_D060_PARENT_AND_MODEL_ATTRIBUTION',
            cell_checks_sha256=sha(output / 'cell_checks.json'), audit_identity_sha256=identity,
            parent_evidence_sha256=sha(output / 'parent_evidence.json'))
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            result['status'] = 'STOPPED_D060_VALIDATION'
        result['errors'].append({'error': repr(exc), 'traceback': traceback.format_exc()})
    finally:
        sys.meta_path.remove(guard)
    result.update(elapsed_seconds=time.perf_counter() - began, reused_sealed_cells=reused, completed_utc=utc_now())
    atomic_save(invocation / 'result.json', result); atomic_save(output / 'verification.json', result)
    if result['status'] == 'VERIFIED_D060_VALIDATION':
        values = {p.relative_to(output).as_posix(): sha(p) for p in output.rglob('*')
                  if p.is_file() and p.name != 'audit_completion_manifest.json' and not p.name.endswith('.tmp')}
        atomic_save(output / 'audit_completion_manifest.json', values)
        result['completion_sha256'] = sha(output / 'audit_completion_manifest.json')
    return result


def pid_running(pid):
    if os.name == 'nt':
        import ctypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return ctypes.get_last_error() != 87
        code = ctypes.c_ulong()
        ok = kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))
        kernel.CloseHandle(ctypes.c_void_p(handle))
        return not ok or code.value == 259
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False


def audit_study(study_path):
    study = Path(study_path).resolve()
    require(study.parent == (ROOT / 'runs').resolve(), 'Study outside project runs')
    lock = study / 'audit_execution.lock'
    if lock.exists():
        prior = read(lock)
        require(not pid_running(prior['pid']), 'An independent audit is already running')
        saved = study / ('stale_audit_lock_' + uuid.uuid4().hex + '.json')
        os.replace(lock, saved)
    with lock.open('x', encoding='utf8') as stream:
        json.dump({'pid': os.getpid(), 'utc': utc_now()}, stream)
        stream.flush(); os.fsync(stream.fileno())
    try:
        return _audit_study_locked(study)
    finally:
        if lock.exists() and read(lock)['pid'] == os.getpid():
            lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    args = parser.parse_args()
    result = audit_study(args.study)
    print(json.dumps(result, ensure_ascii=True, allow_nan=False), flush=True)
    return 0 if result['status'] == 'VERIFIED_D060_VALIDATION' else 1


if __name__ == '__main__':
    raise SystemExit(main())
