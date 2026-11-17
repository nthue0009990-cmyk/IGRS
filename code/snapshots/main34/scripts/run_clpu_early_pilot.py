"""Cold early-16 intervention with a separately paid original Beam guard.

Only this new entry point is changed. Preparation freezes every source and model;
workers never read historical numerical answers, train, or automatically retry.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
import gzip
import hashlib
import json
import math
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
import run_clpu_equivalent_cache_pilot as cache

STUDY = 'CLPU_EARLY_PROTECTED_PILOT_V1'
SEEDS = (17, 41, 73)
SOURCES = {
    'integrated': ('20261030T141744_232986Z_clpu_integrated_pilot', '461826a8e3f9c415821d42b47c81d66acd8c2fa2d01b09f388dd102e9938ba62'),
    'cache': ('20261030T123732_313636Z_clpu_equivalent_cache_pilot', 'f3f1bbfa9a4eeecbdec22b55aad4c80550db00f360b3326e6a3acc764694e040'),
    'window': (cache.SOURCE_RUN, cache.SOURCE_SHA),
    'tail': ('20261027T045048_590910Z_clpu_tail_pilot', '3661145543e8d84c8771a6a3d13a24bd7336d1a1ba3eb2699a3efa545d47bb8e'),
    'd059': ('20261024T073617_269327Z_nature34_clpu_branch', '4c3777f938d21f074b40db56c82cd62ca1b956f6aa7d79215832bf192745c56f'),
    'd060': ('20261025T123350_150996Z_d060_main', 'b2f85e5f21d4e1f316c018a7d53428149f750ae32443fda7c3e572c2333ce7a6'),
}
SLOTS = ('window_reference', 'protected_rule', *(f'protected_bc{s}' for s in SEEDS),
         *(f'protected_ppo{s}' for s in SEEDS))
PROTOCOL = 'docs/clpu_early_pilot_protocol_v1.md'
ACTIVE = ROOT / 'artifacts/clpu_early_pilot_active.json'
MODEL_REVIEW = ROOT / 'artifacts/clpu_integrated_model_preflight_20261030'
EARLY_DIAGNOSTIC = ROOT / 'artifacts/clpu_early_opportunities_20261031'
EARLY_DIAGNOSTIC_SHA = '73a44b3306c5be53a44cb2b0c983dfa48a788a4ad2093fdd20ee61c404fda605'
MIN_FREE, REPORT_RESERVE = 12 * 2**30, 2 * 2**30
MAX_SECONDS, WORKER_SECONDS, TOTAL_CAP, CELLS = 14400, 3600, 30720, 64
_WORKER_ENTERED = False
_PHASE_ENTERED = False


def settings():
    return {'phase_logical_point_budget':256, 'phase_physical_point_budget':256,
        'phase_expansion_cap':256, 'early_policy_expansions':16,
        'protected_total_point_budget':512, 'reference_total_point_budget':256,
        'max_selected_decisions':512, 'cold_searches':120, 'max_workers':2,
        'cooperative_seconds':MAX_SECONDS, 'worker_seconds':WORKER_SECONDS,
        'total_physical_budget':TOTAL_CAP, 'total_numerical_entry_upper_bound':123240,
        'minimum_free_disk_bytes':MIN_FREE, 'forecast_margin':1.25,
        'report_source_reserve_bytes':REPORT_RESERVE, 'cache_variant':'electrical_equivalent',
        'no_training':True, 'no_historical_answers':True, 'no_cross_cell_cache':True,
        'no_cross_phase_cache':True, 'no_automatic_retry':True,
        'phase_process_model':'one_fresh_exec_child_per_phase; parent never initializes a solver; children run serially',
        'maximum_concurrent_physics_processes':2,
        'phase_order':['EARLY','GUARD'], 'exact_energy_tie':'GUARD',
        'recording':'compact_raw_features_once_per_started_decision; full_search_result_once_per_phase',
        'dispatch':'Hashed context and initial slot order; Latin rotation; adjacent two-worker pairs drain before next pair and context',
        'timing':'Same-run cold full-process and worker/phase timing; no old-run speed comparator',
        'aggregation':'Average seeds within context before averaging eight contexts'}


def source_binding():
    return {key: val for label, (run, digest) in SOURCES.items()
            for key, val in ((label + '_run', run), (label + '_completion_sha256', digest))}


def hash_order(prefix, value):
    return hashlib.sha256((prefix + value).encode()).hexdigest()


def schedule_for(contexts, models):
    require(len(contexts) == 8 and {c['context_id'] for c in contexts} == set(range(8)), 'Eight fixed contexts required')
    require(len({c['scenario_id'] for c in contexts}) == 8 and
        Counter(c['event_id'] for c in contexts) == Counter({30: 2, 20: 2, 6: 2, 3: 2}), 'Fixed training population changed')
    lookup = {(m['method'], m['seed']): m for m in models}
    require(len(models) == len(lookup) == 6 and set(lookup) == {(m,s) for m in ('bc_raw','ppo_raw') for s in SEEDS}, 'Six fixed endpoints required')
    contexts = [{k: c[k] for k in ('context_id','scenario_id','event_id')} for c in contexts]
    ordered = sorted(contexts, key=lambda c: hash_order('clpu-early-context-order-v1|', canonical(c)))
    slots = sorted(SLOTS, key=lambda s: hash_order('clpu-early-arm-order-v1|', s))
    rows = []
    for rank, context in enumerate(ordered):
        for position in range(8):
            slot = slots[(position + rank) % 8]
            seed = next((s for s in SEEDS if slot.endswith(str(s))), 0)
            arm = slot[:-len(str(seed))] if seed else slot
            method = 'bc_raw' if arm == 'protected_bc' else 'ppo_raw' if arm == 'protected_ppo' else 'beam'
            model = lookup.get((method,seed))
            rows.append({**context, 'arm': arm, 'arm_slot': slot, 'seed': seed, 'method': method,
                'phase': 'evaluation', 'group': 'early_intervention', 'representation': 'raw', 'deterministic': True,
                'point_budget':256, 'logical_point_budget':256,
                'physical_point_budget':256 if arm == 'window_reference' else 512,
                'total_logical_point_budget':256 if arm == 'window_reference' else 512,
                'early_policy_expansions':16, 'cold_searches':1 if arm == 'window_reference' else 2,
                'window_variant': 'coverage_v1', 'protected': arm != 'window_reference',
                'cache_variant': 'electrical_equivalent', 'context_order': rank,
                'arm_position': position, 'pair_order': rank * 4 + position // 2,
                'episode_id': len(rows), 'cell': f'episodes/{len(rows):03d}',
                'checkpoint': model['checkpoint'] if model else None,
                'checkpoint_sha256': model['source_sha256'] if model else None})
    return rows


def passed_tests(path):
    cases = list(ET.parse(path).getroot().iter('testcase'))
    require(cases and not any(list(c.iter('failure'))+list(c.iter('error'))+list(c.iter('skipped')) for c in cases), 'All software checks must pass')
    for name in ('test_run_clpu_early_pilot', 'test_report_clpu_early_pilot',
                 'test_nature34_clpu_early', 'test_clpu_early_audit'):
        require(any(name in c.get('classname','') for c in cases), 'Missing component checks: '+name)
    return len(cases)


def source_names():
    names = cache.source_names()
    names += ['scripts/' + name for name in ('run_clpu_early_pilot.py', 'audit_clpu_early_pilot.py',
        'report_clpu_early_pilot.py', 'run_clpu_integrated_pilot.py', 'nature34_clpu_tail_execution.py',
        'nature34_clpu_branch_execution.py', 'audit_nature34_clpu_tail.py', 'audit_nature34_clpu_branch_learning.py')]
    names += ['src/dnr/nature34_clpu_early.py']
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT / 'tests').glob('*clpu_early*.py')]
    # Freeze transitive local utility and fixture imports, including independent
    # replay helpers. Resolve source text only: this does not import or execute it.
    pending, complete = list(set(names + [PROTOCOL])), set()
    while pending:
        name = pending.pop()
        if name in complete:
            continue
        complete.add(name)
        path = ROOT / name
        if path.suffix != '.py' or not path.exists():
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            modules = [a.name for a in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.level == 0 and node.module else []
            for module in modules:
                candidates = [ROOT / 'src' / (module.replace('.','/')+'.py')]
                if '.' not in module:
                    candidates += [ROOT / 'scripts' / (module+'.py'), ROOT / 'tests' / (module+'.py')]
                for candidate in candidates:
                    if candidate.is_file():
                        pending.append(candidate.relative_to(ROOT).as_posix())
    return sorted(complete)


def sealed_sources():
    output = {}
    for label, (name, digest) in SOURCES.items():
        source = ROOT / 'runs' / name
        require(sha(source / 'completion_manifest.json') == digest, 'Source root changed: '+label)
        output[label] = (source, normalize_manifest(read(source / 'completion_manifest.json')))
    return output


def bind_models(run, sources):
    source, manifest = sources['tail']
    require(sha(source / 'model_bindings.json') == manifest['model_bindings.json'], 'Tail model binding changed')
    models = read(source / 'model_bindings.json')
    integrated, im = sources['integrated']
    require(sha(integrated / 'model_bindings.json') == im['model_bindings.json']
            and read(integrated / 'model_bindings.json') == models, 'Integrated model population differs')
    d060, dm = sources['d060']
    require(sha(d060 / 'model_bindings.json') == dm['model_bindings.json'], 'D060 model binding changed')
    target_bindings = read(d060 / 'model_bindings.json')
    learning, lm = sources['d059']
    for m in models:
        name, digest = m['checkpoint'], m['source_sha256']
        require(im[name] == sha(integrated / name) == digest, 'Integrated fixed model bytes differ')
        require(manifest[name] == sha(source / name) == lm[m['source_relative']]
            == sha(learning / m['source_relative']) == digest, 'Frozen model lineage mismatch')
        matched = [v for v in target_bindings if (v['method'],v['seed']) == (m['method'],m['seed'])]
        require(len(matched) == 1 and matched[0]['source_sha256'] == digest
            and dm[matched[0]['checkpoint']] == sha(d060 / matched[0]['checkpoint']) == digest, 'D060 fixed model differs')
        bound_copy(source, run / name, manifest, name)
    bound_copy(source, run / 'model_bindings.json', manifest, 'model_bindings.json')
    review_manifest = cache.verify_manifest(MODEL_REVIEW, required=('summary.json',))
    review = read(MODEL_REVIEW / 'summary.json')
    require(review['status'] == 'PASS' and not review['errors'] and review['new_physical_points'] == review['new_training'] == 0,
            'Independent model source/tensor compatibility preflight failed')
    require([{k:v for k,v in m.items() if k != 'architecture'} for m in review['models']] == models, 'Model preflight endpoints differ')
    for name in (*review_manifest, 'completion_manifest.json'):
        target = run / 'model_preflight' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(MODEL_REVIEW / name, target)
    save(run / 'model_preflight_binding.json', {'path': MODEL_REVIEW.relative_to(ROOT).as_posix(),
        'completion_sha256': sha(MODEL_REVIEW / 'completion_manifest.json'), 'summary_sha256': sha(MODEL_REVIEW / 'summary.json')})
    return models


def verify_early_diagnostic(directory):
    require(sha(directory / 'completion_manifest.json') == EARLY_DIAGNOSTIC_SHA, 'Early diagnostic root changed')
    manifest = cache.verify_manifest(directory,required=('summary.json','all_slots.json','contexts.json','source_bindings.json'))
    summary = read(directory / 'summary.json')
    require(summary['status'] == 'COMPLETE_CAUSAL_PREFIX_DIAGNOSTIC'
            and summary['source_run'] == SOURCES['integrated'][0]
            and summary['source_completion_sha256'] == SOURCES['integrated'][1]
            and summary['contexts'] == 8 and summary['slots'] == 128
            and summary['new_physical_points'] == summary['new_training'] == summary['optimizer_updates'] == 0
            and summary['alternative_actions_executed'] == 0
            and summary['independent_model_argmax_mismatch_positions'] == 0
            and summary['formal_or_tune_or_holdout_access'] is False, 'Early diagnostic incomplete or changed scope')
    return manifest


def bind_early_diagnostic(run):
    manifest = verify_early_diagnostic(EARLY_DIAGNOSTIC)
    for name in (*manifest,'completion_manifest.json'):
        destination = run / 'early_diagnostic' / name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(EARLY_DIAGNOSTIC / name,destination)
    save(run / 'early_diagnostic_binding.json',{'path':EARLY_DIAGNOSTIC.relative_to(ROOT).as_posix(),
        'completion_sha256':EARLY_DIAGNOSTIC_SHA,'summary_sha256':sha(EARLY_DIAGNOSTIC / 'summary.json'),
        'usage':'Opportunity description only; no filtering of eight contexts, six models, K, actions or budgets'})


def directory_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())


def storage_forecast(run, schedule, completed, historical_max):
    measured = [directory_bytes(run / cell) for cell in completed]
    per_cell = max([historical_max, *measured])
    remaining = [s for s in schedule if s['cell'] not in completed]
    written = sum(directory_bytes(run / s['cell']) for s in remaining)
    outstanding = max(0, math.ceil(1.25 * per_cell * len(remaining)) - written)
    return {'historical_max_cell_bytes': historical_max, 'observed_max_cell_bytes': max(measured,default=0),
        'forecast_cell_bytes': per_cell, 'remaining_cells': len(remaining), 'already_written_remaining_bytes': written,
        'remaining_forecast_bytes': outstanding, 'required_free_bytes': outstanding + REPORT_RESERVE + MIN_FREE,
        'free_bytes': shutil.disk_usage(ROOT).free, 'forecast_is_hard_upper_bound': False}


def resource_estimate(sources, tests):
    cache_source = sources['cache'][0]
    tail_source = sources['tail'][0]
    require(sha(cache_source / 'schedule.json') == sources['cache'][1]['schedule.json']
        and sha(tail_source / 'schedule.json') == sources['tail'][1]['schedule.json'], 'Resource forecast source population changed')
    cache_rows = read(cache_source / 'schedule.json')
    tail_rows = [s for s in read(tail_source / 'schedule.json') if s.get('cohort') == 'training_mechanism']
    require(len(cache_rows) == 16 and len(tail_rows) == 32, 'Resource forecast observed-training population changed')
    sizes = [directory_bytes(cache_source / s['cell']) for s in cache_rows]
    tail_sizes = [directory_bytes(tail_source / s['cell']) for s in tail_rows]
    historical_max = 2 * max(sizes + tail_sizes)
    return {'contexts': 8, 'cells': CELLS, 'physical_points': TOTAL_CAP,
        'numerical_entries':120 * 3 + 4 * TOTAL_CAP, 'cold_searches':120, 'workers': 2,
        'minimum_free_bytes': MIN_FREE, 'cooperative_seconds': MAX_SECONDS, 'worker_seconds': WORKER_SECONDS,
        'new_training': 0, 'software_tests': tests, 'free_bytes_at_prepare': shutil.disk_usage(ROOT).free,
        'forecast_margin': 1.25, 'report_source_reserve_bytes': REPORT_RESERVE,
        'historical_max_cell_bytes': historical_max, 'historical_cache_cells':16, 'historical_training_tail_cells':32,
        'cache_max_cell_bytes': max(sizes), 'tail_training_max_cell_bytes':max(tail_sizes),
        'initial_required_free_bytes': math.ceil(CELLS * historical_max * 1.25) + REPORT_RESERVE + MIN_FREE,
        'forecast_is_hard_upper_bound': False}


def prepare(tests_path):
    from dnr.nature34_clpu_branch import BranchConfig
    tests = passed_tests(tests_path)
    sources = sealed_sources()
    source, sm = sources['integrated']
    for name in ('config.json','population.json','selected_scenarios.json','selected_events.json','training_contexts.json','source_cells.json'):
        require(sha(source / name) == sm[name], 'Inherited input changed: '+name)
    contexts = [{k:c[k] for k in ('context_id','scenario_id','event_id')}
                for c in read(source / 'source_cells.json')]
    require({(c['scenario_id'],c['event_id']) for c in contexts}
        == {(c['scenario_id'],c['event_id']) for c in read(source / 'training_contexts.json')}, 'Training population differs')
    cfg = deepcopy(read(source / 'config.json'))
    require(cfg['branch_search'] == asdict(BranchConfig()), 'Original search configuration changed')
    cfg.update(early_study=STUDY, early_pilot=settings())
    budget = resource_estimate(sources, tests)
    require(budget['free_bytes_at_prepare'] >= budget['initial_required_free_bytes'], 'Dynamic storage forecast not met')
    run = create_run('clpu_early_pilot', cfg)
    try:
        for name in (*ROOT_INPUTS,'population.json','selected_scenarios.json','selected_events.json','training_contexts.json'):
            bound_copy(source, run / name, sm, name)
        for name in sm:
            if name.startswith('source_inputs/'):
                bound_copy(source, run / name, sm, name)
        for label, (origin, manifest) in sources.items():
            target = run / 'provenance' / label
            target.mkdir(parents=True)
            shutil.copyfile(origin / 'completion_manifest.json', target / 'source_completion_manifest.json')
            for name in ('model_bindings.json','source_binding.json','terminal_status.json'):
                if name in manifest:
                    bound_copy(origin, target / name, manifest, name)
        models = bind_models(run, sources)
        schedule = schedule_for(contexts, models)
        window, wm = sources['window']
        (run / 'historical').mkdir()
        shutil.copyfile(window / 'completion_manifest.json', run / 'historical/source_completion_manifest.json')
        bound_copy(window, run / 'historical/source_audit_summary.json', wm, 'audit/summary.json')
        source_schedule = read(window / 'schedule.json')
        require(sha(window / 'schedule.json') == wm['schedule.json'], 'Original schedule changed')
        history = []
        for context in contexts:
            record = dict(context)
            for label, variant in (('g0','original'),('g1','coverage_v1')):
                matches = [s for s in source_schedule if s['context_id'] == context['context_id'] and s['window_variant'] == variant]
                require(len(matches) == 1 and all(matches[0][k] == context[k] for k in ('scenario_id','event_id')), 'Historical G0/G1 missing')
                original = matches[0]
                cell, cm = cache.verify_source_cell(window, wm, original)
                prefix = f"historical/c{context['context_id']:02d}/{label}"
                for name in (*cache.HISTORY_FILES,'completion_manifest.json'):
                    target = run / prefix / name
                    target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(cell / name, target)
                record[label] = {'source_cell':original['cell'], 'source_completion_sha256':sha(cell / 'completion_manifest.json'),
                    'source_files':{n:cm[n] for n in (*cache.HISTORY_FILES,'points.jsonl.gz')}, 'historical_directory':prefix}
            history.append(record)
        save(run / 'source_cells.json', history)
        save(run / 'source_binding.json', source_binding())
        cache.bind_scope_review(run); cache.bind_exact_key_review(run)
        bind_early_diagnostic(run)
        save(run / 'resource_budget.json', budget)
        save(run / 'schedule.json', schedule)
        for spec in schedule:
            save(run / spec['cell'] / 'spec.json', spec)
            for phase in phase_names(spec):
                child = phase_spec(spec,phase)
                save(run / child['cell'] / 'spec.json',child)
        shutil.copyfile(tests_path, run / 'software_tests.xml')
        shutil.copyfile(ROOT / PROTOCOL, run / 'protocol.md')
        save(run / 'authorization.json', {'utc':utc(), 'execution_authorized':True, 'user_local_date':'2026-10-31',
            'scope':'One fixed 64-cell early-16 intervention pilot; 120 independent cold searches; six existing actors, no training or K scan',
            'data_handling':'Retain all real outcomes and failures; no adjusted, fabricated or historical numerical answers'})
        names = source_names()
        for name in names:
            for _, manifest in sources.values():
                if 'source/' + name in manifest:
                    require(sha(ROOT / name) == manifest['source/' + name], 'Inherited source changed: '+name)
        save(run / 'physical_scope.json', cache.physical_scope(run,names))
        freeze(run,names)
        save(ACTIVE, {'status':'FROZEN_NOT_STARTED','run':run.name,'scheduled':CELLS,'completed':0,'utc':utc()})
        return run
    except BaseException as exc:
        save(run / 'preparation_error.json', {'error':repr(exc),'traceback':traceback.format_exc(), 'new_physical_points':0,'new_training':0,'utc':utc()})
        raise


def verify_execution(run):
    prepared = read(run / 'prepared.json')
    require(prepared['status'] == 'FROZEN_PREPARED_NO_DSS' and sha(run / 'prephysical_freeze_manifest.json') == prepared['freeze_sha256'], 'Freeze changed')
    cache.verify_manifest(run,'prephysical_freeze_manifest.json',('schedule.json','physical_scope.json','model_bindings.json','model_preflight_binding.json','source_cells.json'))
    for name in prepared['source_files']:
        require(sha(ROOT / name) == sha(run / 'source' / name), 'Executing source changed: '+name)
    require(dependencies() == read(run / 'dependencies.json'), 'Dependencies changed')
    require(sha(Path(sys.executable)) == read(run / 'python_environment.json')['sha256'], 'Interpreter changed')
    cfg = read(run / 'config.json')
    from dnr.nature34_clpu_branch import BranchConfig
    require(cfg['early_study'] == STUDY and cfg['early_pilot'] == settings()
        and cfg['branch_search'] == asdict(BranchConfig()), 'Execution contract changed')
    require(read(run / 'source_binding.json') == source_binding(), 'Scientific source binding changed')
    require(cache.physical_scope(run,prepared['source_files']) == read(run / 'physical_scope.json'), 'Complete physical scope changed')
    require(read(run / 'schedule.json') == schedule_for(read(run / 'source_cells.json'),read(run / 'model_bindings.json')), 'Frozen schedule changed')
    for spec in read(run / 'schedule.json'):
        for phase in phase_names(spec):
            child = phase_spec(spec,phase)
            require(read(run / child['cell'] / 'spec.json') == child, 'Frozen phase configuration changed')
    for folder in ('model_preflight','scope_review'):
        binding = read(run / (folder + '_binding.json'))
        require(sha(run / folder / 'completion_manifest.json') == binding['completion_sha256']
            and sha(run / folder / 'summary.json') == binding['summary_sha256'], 'Independent preflight binding changed')
        cache.verify_manifest(run / folder,required=('summary.json',))
        require(read(run / folder / 'summary.json')['status'] == 'PASS', 'Independent preflight did not pass')
    cache.verify_exact_key_review(run)
    binding = read(run / 'early_diagnostic_binding.json')
    require(binding['path'] == EARLY_DIAGNOSTIC.relative_to(ROOT).as_posix()
            and binding['completion_sha256'] == EARLY_DIAGNOSTIC_SHA
            and binding['summary_sha256'] == sha(run / 'early_diagnostic/summary.json'), 'Early diagnostic binding changed')
    verify_early_diagnostic(run / 'early_diagnostic')


def physical_identity(run,spec,env):
    return {'study':STUDY,'freeze_sha256':sha(run / 'prephysical_freeze_manifest.json'),
        'scenario_id':spec['scenario_id'],'event_id':spec['event_id'],'initial_state':numeric_lists(env.initial.to_dict()),
        'condition_sha256':sha(run / 'condition.json'),'physical_config_sha256':sha(run / 'physical_config.json'),
        'author_static_metadata_sha256':sha(run / 'author_static_metadata.json'),'source_provenance_sha256':sha(run / 'source_provenance.json')}


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def phase_spec(spec, phase):
    require(phase in ('EARLY','GUARD') and (phase != 'GUARD' or spec['protected']), 'Invalid phase')
    return {**spec, 'parent_cell':spec['cell'], 'parent_arm':spec['arm'], 'parent_seed':spec['seed'],
        'cell':spec['cell']+'/'+phase.lower(), 'search_phase':phase,
        'method':spec['method'] if phase == 'EARLY' else 'beam',
        'seed':spec['seed'] if phase == 'EARLY' else 0,
        'checkpoint':spec['checkpoint'] if phase == 'EARLY' else None,
        'checkpoint_sha256':spec['checkpoint_sha256'] if phase == 'EARLY' else None,
        'point_budget':256, 'logical_point_budget':256, 'physical_point_budget':256,
        'total_logical_point_budget':256, 'cold_searches':1}


def phase_names(spec):
    return ('EARLY','GUARD') if spec['protected'] else ('EARLY',)


def oracle_cost(oracle):
    return {'logical_points':oracle.points, 'physical_points':oracle.physical_points,
        'numerical_entries':oracle.factory.stats()['returned']['numerical_solution_entries'],
        'requests':oracle.requests}


def complete_quality(search):
    plan = search.incumbent
    certified = bool(plan and plan.get('certified_complete') is True and plan.get('status') == 'COMPLETE'
                     and plan.get('validation',{}).get('status') == 'COMPLETE')
    return 100 * search.quality if certified else None


def checkpoint(search, phase, kind, phase_choices, previous, previous_choices,
               best_previous, guard_complete, started, computation_started, index):
    local = oracle_cost(search.oracle)
    qualities = [q for q in (best_previous,complete_quality(search)) if q is not None]
    now = time.perf_counter()
    return {'checkpoint_index':index, 'phase':phase, 'kind':kind, 'phase_choices':phase_choices,
        'search_choices':previous_choices+phase_choices,
        **{key:previous.get(key,0)+value for key,value in local.items()},
        'phase_logical_points':local['logical_points'], 'phase_physical_points':local['physical_points'],
        'quality_pct':max(qualities) if qualities else None, 'fully_certified':bool(qualities),
        'guard_established':bool(guard_complete), 'worker_elapsed_seconds':now-started,
        'elapsed_seconds':now-computation_started}


def collect_phase(search, spec, policy, generator, emit, after_step):
    from dnr.nature34_clpu_early import choose_action
    from nature34_clpu_tail_execution import check_deadline
    phase, count, model_count = spec['search_phase'], 0, 0
    timer = {key:0. for key in ('observation_seconds','selection_seconds','search_step_seconds','journal_seconds')}
    while not search.done:
        check_deadline(spec)
        require(search._beam_turn == count, 'Phase expansion counter changed')
        begin = time.perf_counter()
        raw = numeric_lists(search.observation('raw'))
        compact = {key:raw[key] for key in ('representation','candidate_features','global_features','mask','branch_ids')}
        timer['observation_seconds'] += time.perf_counter()-begin
        begin = time.perf_counter()
        chosen = choose_action(search,raw,phase,spec['method'],policy,generator)
        timer['selection_seconds'] += time.perf_counter()-begin
        model_count += int(chosen['policy_used'])
        action = chosen['action']
        identity = {'transition_id':count, 'phase_choice':count, 'phase':phase,
            'method':chosen['method'], 'action':action, 'branch_id':raw['branch_ids'][action],
            'obs_raw_sha256':digest(raw), 'obs_compact_sha256':digest(compact), 'mask_sha256':digest(raw['mask']),
            'checkpoint_sha256':spec['checkpoint_sha256'] if chosen['policy_used'] else None,
            'logprob':chosen['logprob'], 'value':chosen['value'], 'oracle_before':oracle_cost(search.oracle)}
        begin = time.perf_counter()
        emit({**identity,'status':'STARTED','obs_compact':compact})
        timer['journal_seconds'] += time.perf_counter()-begin
        begin = time.perf_counter()
        response = search.step(action)
        timer['search_step_seconds'] += time.perf_counter()-begin
        require(isinstance(response,dict) and type(response.get('done')) is bool
                and math.isfinite(float(response['reward'])), 'Malformed phase response')
        begin = time.perf_counter()
        emit({**identity,'status':'COMPLETED','response':numeric_lists(response),
              'oracle_after':oracle_cost(search.oracle)})
        count += 1
        after_step(count)
        timer['journal_seconds'] += time.perf_counter()-begin
        require(count <= search.config.max_branch_expansions, 'Phase expansion limit exceeded')
    return {'search_decisions':count,'policy_decisions':model_count,'stage_component_seconds':timer}


def sum_oracles(summaries):
    fields = ('logical_points','physical_points','actual_numerical_entries','initialization_numerical_entries',
              'requests','semantic_cache_hits','electrical_reuses')
    require(summaries, 'Missing phase ledgers')
    for row in summaries:
        require(all(type(row.get(key)) is int and row[key] >= 0 for key in fields), 'Invalid phase ledger')
        require(row['physical_points'] <= row['logical_points'] <= 256, 'Closed phase exceeded budget')
        require(row['initialization_numerical_entries'] == 3
                and row['actual_numerical_entries'] == 3+4*row['physical_points'], 'Incomplete numerical ledger')
    return {'variant':'electrical_equivalent', 'scope':'sum_of_independent_cold_phase_ledgers',
        'cold_searches':len(summaries), **{key:sum(row[key] for row in summaries) for key in fields}}


def phase_worker(run,folder):
    """Exactly one oracle/runtime per fresh Python process."""
    global _PHASE_ENTERED
    child_started = time.perf_counter()
    require(not _PHASE_ENTERED, 'A phase process may enter only once')
    _PHASE_ENTERED = True
    from run_nature34_clpu_pilot import make_env
    from dnr.nature34_clpu_branch import BranchConfig,BranchSearch
    from dnr.nature34_clpu_window import WindowBranchSearch
    from dnr.nature34_clpu_equivalent_cache import EquivalenceOracle
    from nature34_clpu_tail_execution import load_fixed_policy,check_deadline
    require(run.is_absolute() and folder.is_absolute() and folder != run and folder.is_relative_to(run), 'Phase escaped run')
    require(not any((folder / name).exists() for name in
        ('phase_started.json','phase_result.json','phase_error.json','points.jsonl.gz')), 'Repeated phase forbidden')
    ps = read(folder / 'spec.json')
    try:
        verify_execution(run)
        parent = read(run / ps['parent_cell'] / 'spec.json')
        require(parent in read(run / 'schedule.json') and ps == phase_spec(parent,ps['search_phase'])
                and folder == run / ps['cell'], 'Unscheduled or changed phase')
        execution = read(folder / 'phase_execution.json')
        require(execution['phase'] == ps['search_phase'] and execution['phase_spec_sha256'] == sha(folder / 'spec.json'),
                'Phase launch contract changed')
        require(execution['task_parent_pid'] != os.getpid(), 'Phase must execute in a fresh child process')
        started = execution['parent_started_monotonic']
        require(type(started) in (int,float) and math.isfinite(started) and child_started >= started,
                'Parent and child monotonic clocks are not comparable')
        require(execution['clock_implementation'] == time.get_clock_info('perf_counter').implementation,
                'Parent and child use different monotonic clocks')
        deadline = execution['deadline_epoch_seconds']
        require(deadline <= read(run / 'execution_started.json')['deadline_epoch_seconds'], 'Child extended global deadline')
        runtime_spec = {**ps,'deadline_utc_epoch':deadline}
        check_deadline(runtime_spec)
        require(shutil.disk_usage(ROOT).free >= MIN_FREE, 'Phase storage floor reached')
        save(folder / 'phase_started.json',{'phase':ps['search_phase'],'pid':os.getpid(),
            'observed_parent_pid':os.getppid(),'task_parent_pid':execution['task_parent_pid'],'utc':utc(),
            'started_monotonic':child_started,'worker_elapsed_seconds':child_started-started,
            'spec_sha256':sha(folder / 'spec.json'),'phase_execution_sha256':sha(folder / 'phase_execution.json')})
        model_path = run / ps['checkpoint'] if ps['checkpoint'] else None
        if model_path:
            require(sha(model_path) == ps['checkpoint_sha256'], 'Fixed child model changed')
        model_started = time.perf_counter()
        if ps['search_phase'] == 'EARLY':
            policy,generator = load_fixed_policy(model_path,runtime_spec)
            model_load_seconds = time.perf_counter()-model_started
        else:
            require(model_path is None and ps['method'] == 'beam' and ps['seed'] == 0, 'Guard must not load a model')
            policy,generator,model_load_seconds = None,None,0.
        phase_compute_started = time.perf_counter()
        save(folder / 'preflight_boundary.json',{'child_started_monotonic':child_started,
            'phase_compute_started_monotonic':phase_compute_started,
            'parent_worker_compute_started_seconds':phase_compute_started-started,
            'child_preflight_elapsed_seconds':phase_compute_started-child_started,
            'model_load_seconds':model_load_seconds,'worker_time_origin':'parent_worker_entry_perf_counter'})
        env = make_env(run,ps)
        save(folder / 'initial_state.json',env.initial.to_dict())
        save(folder / 'physical_identity.json',physical_identity(run,ps,env))
        cfg = BranchConfig(**read(run / 'config.json')['branch_search'])
        scope = read(run / 'physical_scope.json')
        points = []
        with (folder / 'cost_checkpoints.jsonl').open('x',encoding='utf-8') as cost_log:
            with ExitStack() as stack:
                logs = {name:stack.enter_context(gzip.open(folder / (name+'.jsonl.gz'),'xt',encoding='utf-8',compresslevel=3))
                        for name in ('plans','decisions')}
                emit = lambda name,row: append(logs[name],numeric_lists(row))
                oracle = stack.enter_context(EquivalenceOracle(run,folder,env,256,'electrical_equivalent',scope))
                oracle.deadline_monotonic = time.perf_counter()+max(0,deadline-time.time())
                initialization_seconds = time.perf_counter()-phase_compute_started
                require(oracle.points == oracle.physical_points == 0 and not oracle.cache, 'Phase did not begin cold')
                cls = WindowBranchSearch if ps['search_phase'] == 'EARLY' else BranchSearch
                search = cls(env,oracle,cfg,record_callback=lambda row:emit('plans',row))

                def record_cost(count,kind='DECISION',guard_complete=False):
                    row = checkpoint(search,ps['search_phase'],kind,count,{},0,None,
                        guard_complete,started,phase_compute_started,len(points))
                    points.append(row); append(cost_log,row)
                    return row

                record_cost(0,'INITIAL')
                collected = collect_phase(search,runtime_spec,policy,generator,
                    lambda row:emit('decisions',row),record_cost)
                phase_result = numeric_lists(search.result())
            summary = read(folder / 'equivalent_cache_summary.json')
            sum_oracles([summary])
            quality = complete_quality(search)
            record_cost(collected['search_decisions'],'CLOSE',ps['search_phase'] == 'GUARD' and quality is not None)
        if model_path:
            require(sha(model_path) == ps['checkpoint_sha256'], 'Fixed model changed during phase')
        phase_elapsed = time.perf_counter()-phase_compute_started
        phase_result.update(spec=ps,oracle=summary,logical_points=oracle.points,
            actual_physical_points=oracle.physical_points,plan_point_id_semantics='phase_local_logical_point_id',
            quality_pct=quality,fully_certified=quality is not None,**collected,
            runtime_initialization_seconds=initialization_seconds,phase_elapsed_seconds=phase_elapsed,
            model_load_seconds=model_load_seconds,child_preflight_elapsed_seconds=phase_compute_started-child_started,
            child_entry_worker_seconds=child_started-started,
            parent_worker_compute_started_seconds=phase_compute_started-started,
            actual_child_pid=os.getpid(),task_parent_pid=execution['task_parent_pid'])
        save(folder / 'phase_result.json',phase_result)
        save(folder / 'phase_finished.json',{'status':'COMPLETE','phase':ps['search_phase'],'pid':os.getpid(),
            'task_parent_pid':execution['task_parent_pid'],'utc':utc(),'finished_monotonic':time.perf_counter(),
            'worker_elapsed_seconds':time.perf_counter()-started,'phase_elapsed_seconds':phase_elapsed,
            'phase_result_sha256':sha(folder / 'phase_result.json'),
            'oracle_summary_sha256':sha(folder / 'equivalent_cache_summary.json')})
        seal(folder)
        return 0
    except BaseException as exc:
        save(folder / 'phase_error.json',{'status':'STOPPED_RELIABILITY_OR_IMPLEMENTATION_ERROR','spec':ps,
            'actual_child_pid':os.getpid(),'error':repr(exc),'traceback':traceback.format_exc(),
            'partial_evidence_retained':True,'no_automatic_retry':True})
        return 1


def launch_phase(run,cell,ps,started,started_epoch,deadline,dispatch,exits):
    """Parent controller: one child at a time, always wait for real process exit."""
    folder = run / ps['cell']
    require(not (folder / 'phase_execution.json').exists(), 'Phase already dispatched')
    save(folder / 'phase_execution.json',{'phase':ps['search_phase'],'task_parent_pid':os.getpid(),
        'parent_started_monotonic':started,'parent_started_epoch_seconds':started_epoch,
        'deadline_epoch_seconds':deadline,'phase_spec_sha256':sha(folder / 'spec.json'),
        'clock_implementation':time.get_clock_info('perf_counter').implementation})
    env = dict(os.environ)
    env.update({name:'1' for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS')})
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    command = [sys.executable,'-B',str(Path(__file__).resolve()),'phase-worker','--run',str(run),'--cell',str(folder)]
    with (folder / 'stdout.log').open('x',encoding='utf-8') as out, (folder / 'stderr.log').open('x',encoding='utf-8') as err:
        began,epoch = time.perf_counter(),time.time()
        proc = subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=err,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        row = {'phase':ps['search_phase'],'phase_cell':ps['cell'],'task_parent_pid':os.getpid(),
            'launcher_pid':proc.pid,'started_monotonic':began,'started_epoch_seconds':epoch,'utc':utc()}
        append(dispatch,row)
        code = proc.poll()
        while code is None:
            # The child cooperatively enforces the deadline. Never kill a writer.
            time.sleep(.05)
            code = proc.poll()
        ended,ended_epoch = time.perf_counter(),time.time()
    actual = read(folder / 'phase_started.json')['pid'] if (folder / 'phase_started.json').exists() else None
    outcome = {**row,'actual_child_pid':actual,'finished_monotonic':ended,'finished_epoch_seconds':ended_epoch,
        'process_wall_seconds':ended-began,'exit_code':code,
        'phase_completion_sha256':sha(folder / 'completion_manifest.json') if (folder / 'completion_manifest.json').exists() else None}
    exits.append(outcome)
    save(cell / 'phase_process_exits.json',exits)
    require(code == 0 and outcome['phase_completion_sha256'] is not None, 'Phase child did not complete successfully')
    cache.verify_manifest(folder,required=('phase_result.json','phase_started.json','phase_finished.json',
        'preflight_boundary.json','cost_checkpoints.jsonl','spec.json','phase_execution.json'))
    require(actual != os.getpid() and actual == read(folder / 'phase_finished.json')['pid'], 'Child process identity differs')
    require(len({r['actual_child_pid'] for r in exits}) == len(exits), 'Phase processes were not independent')
    return outcome


def combine_checkpoints(local,previous,previous_choices,best_previous,preflight_seconds,index_start):
    result=[]
    for index,row in enumerate(local):
        require(row['checkpoint_index'] == index, 'Child checkpoint omitted')
        values=[value for value in (best_previous,row['quality_pct']) if value is not None]
        current={**row,'checkpoint_index':index_start+index,'search_choices':previous_choices+row['phase_choices'],
            **{key:previous.get(key,0)+row[key] for key in ('logical_points','physical_points','numerical_entries','requests')},
            'quality_pct':max(values) if values else None,'fully_certified':bool(values),
            'elapsed_seconds':row['worker_elapsed_seconds']-preflight_seconds}
        require(current['elapsed_seconds'] >= 0, 'Child certificate predates EARLY compute boundary')
        result.append(current)
    return result


def worker(run,cell):
    """A cold parent task. Never instantiate an environment or physical runtime."""
    global _WORKER_ENTERED
    started,started_epoch = time.perf_counter(),time.time()
    require(not _WORKER_ENTERED, 'Each task requires a cold worker')
    _WORKER_ENTERED = True
    from dnr.nature34_clpu_early import merge_results
    from nature34_clpu_tail_execution import process_memory_evidence
    require(run.is_absolute() and cell.is_absolute() and cell != run and cell.is_relative_to(run), 'Cell escaped')
    require(not any((cell / name).exists() for name in ('worker_started.json','result.json','error.json')), 'Repeated task forbidden')
    spec = read(cell / 'spec.json')
    try:
        verify_execution(run)
        require(spec in read(run / 'schedule.json'), 'Unscheduled task')
        deadline = min(read(run / 'execution_started.json')['deadline_epoch_seconds'],started_epoch+WORKER_SECONDS)
        require(time.time() < deadline and shutil.disk_usage(ROOT).free >= MIN_FREE, 'Worker deadline/storage exhausted')
        save(cell / 'worker_started.json',{'pid':os.getpid(),'utc':utc(),'started_monotonic':started,
            'started_epoch_seconds':started_epoch,'spec_sha256':sha(cell / 'spec.json'),'checkpoint_sha256':spec['checkpoint_sha256']})
        save(cell / 'worker_deadline.json',{'deadline_epoch_seconds':deadline,'worker_seconds':WORKER_SECONDS})
        parent_validation_seconds = time.perf_counter()-started
        checkpoints,phases,phase_results,summaries,phase_processes = [],{},{},[],[]
        previous,previous_choices,best_previous = {},0,None
        preflight_seconds = model_load_seconds = early_block = None
        with (cell / 'cost_checkpoints.jsonl').open('x',encoding='utf-8') as cost_log, \
                (cell / 'phase_dispatch.jsonl').open('x',encoding='utf-8') as dispatch:
            for phase in phase_names(spec):
                require(time.time() < deadline and shutil.disk_usage(ROOT).free >= MIN_FREE, 'Stop before starting next cold phase')
                ps = phase_spec(spec,phase)
                folder = run / ps['cell']
                require(read(folder / 'spec.json') == ps, 'Frozen phase spec changed')
                process = launch_phase(run,cell,ps,started,started_epoch,deadline,dispatch,phase_processes)
                current = read(folder / 'phase_result.json')
                require(current['spec'] == ps and current['actual_child_pid'] == process['actual_child_pid'],
                        'Phase result identity changed')
                summary = read(folder / 'equivalent_cache_summary.json')
                require(summary == current['oracle'], 'Phase oracle ledger changed')
                sum_oracles([summary])
                if phase == 'EARLY':
                    preflight_seconds = current['parent_worker_compute_started_seconds']
                    model_load_seconds = current['model_load_seconds']
                    require(preflight_seconds >= parent_validation_seconds and preflight_seconds >= model_load_seconds,
                            'EARLY model/parent preflight accounting changed')
                else:
                    require(current['model_load_seconds'] == 0., 'Guard loaded a model')
                with (folder / 'cost_checkpoints.jsonl').open(encoding='utf-8') as stream:
                    local = [json.loads(line) for line in stream if line.strip()]
                combined = combine_checkpoints(local,previous,previous_choices,best_previous,preflight_seconds,len(checkpoints))
                for row in combined:
                    checkpoints.append(row);append(cost_log,row)
                require(combined[-1]['kind'] == 'CLOSE', 'Missing normal phase close')
                if phase == 'EARLY':
                    early_block = next(row for row in reversed(combined)
                        if row['kind'] != 'CLOSE' and row['phase_choices'] == min(16,current['search_decisions']))
                fields = ('status','quality_pct','fully_certified','logical_points','search_decisions','policy_decisions',
                          'stop_reason','phase_elapsed_seconds','runtime_initialization_seconds','stage_component_seconds')
                phases[phase] = {'directory':phase.lower(),**{key:current[key] for key in fields},
                    'physical_points':summary['physical_points'],'numerical_entries':summary['actual_numerical_entries'],
                    'requests':summary['requests'],'process_wall_seconds':process['process_wall_seconds'],
                    'actual_child_pid':process['actual_child_pid'],'launcher_pid':process['launcher_pid'],
                    'phase_result_sha256':sha(folder / 'phase_result.json'),
                    'completion_manifest_sha256':sha(folder / 'completion_manifest.json')}
                phase_results[phase],summaries = current,[*summaries,summary]
                previous = {key:combined[-1][key] for key in ('logical_points','physical_points','numerical_entries','requests')}
                previous_choices = combined[-1]['search_choices']
                best_previous = combined[-1]['quality_pct']
        merged = merge_results(phase_results['EARLY'],phase_results.get('GUARD'),protected=spec['protected'])
        combined = sum_oracles(summaries)
        save(cell / 'equivalent_cache_summary.json',combined)
        selected = phase_results[merged['selected_phase']]
        baseline = phase_results.get('GUARD')
        baseline_quality = baseline['quality_pct'] if baseline else None
        gain = (merged['best_plan']['validation']['energy_kwh']-baseline['best_plan']['validation']['energy_kwh']
                if baseline and baseline_quality is not None and merged['best_plan'] else None)
        result = {**merged,'spec':spec,'phases':phases,'oracle':combined,
            'quality_pct':selected['quality_pct'],'fully_certified':selected['fully_certified'],
            'early_quality_pct':phase_results['EARLY']['quality_pct'],'baseline_quality_pct':baseline_quality,
            'gain_over_paid_baseline_kwh':gain,'logical_points':combined['logical_points'],
            'actual_physical_points':combined['physical_points'],'numerical_entries':combined['actual_numerical_entries'],
            'search_decisions':sum(p['search_decisions'] for p in phases.values()),
            'policy_decisions':sum(p['policy_decisions'] for p in phases.values()),
            'stage_decisions':{key:p['search_decisions'] for key,p in phases.items()},
            'stage_component_seconds':{key:p['stage_component_seconds'] for key,p in phases.items()},
            'checkpoints':checkpoints,'early_block_checkpoint':early_block,'final_real_cost':checkpoints[-1],
            'parent_validation_seconds':parent_validation_seconds,'phase_processes':phase_processes,
            'preflight_elapsed_seconds':preflight_seconds,'model_load_seconds':model_load_seconds,
            'runtime_initialization_seconds':sum(p['runtime_initialization_seconds'] for p in phases.values()),
            'computation_elapsed_seconds':time.perf_counter()-started-preflight_seconds,'elapsed_seconds':time.perf_counter()-started,
            'checkpoint_sha256':spec['checkpoint_sha256'],'process_memory':process_memory_evidence(),
            'new_training':0,'optimizer_updates':0,'training_interactions':0,'formal_holdout_numerical_access':False,
            'tune_numerical_access':False,'clpu_holdout_access':False}
        save(cell / 'result.json',result)
        seal(cell)
        return 0
    except BaseException as exc:
        save(cell / 'error.json',{'status':'STOPPED_RELIABILITY_OR_IMPLEMENTATION_ERROR','spec':spec,'error':repr(exc),
            'traceback':traceback.format_exc(),'elapsed_seconds':time.perf_counter()-started,
            'partial_evidence_retained':True,'no_automatic_retry':True})
        return 1

def costs(run, dispatched):
    totals = {'logical_points':0,'physical_points':0,'actual_numerical_entries':0}
    unknown = []
    for spec in dispatched:
        for phase in phase_names(spec):
            folder = run / spec['cell'] / phase.lower()
            # An unstarted phase has no initialized runtime or paid calls. If
            # a started phase lacks a closing ledger, its cost remains unknown.
            if not (folder / 'phase_started.json').exists():
                if (run / spec['cell'] / 'completion_manifest.json').exists():
                    unknown.append(spec['cell']+'/'+phase.lower())
                continue
            try:
                summary = read(folder / 'equivalent_cache_summary.json')
                require(all(type(summary[key]) is int and summary[key] >= 0 for key in totals), 'Invalid phase cost')
                require(summary['physical_points'] <= 256, 'Real phase budget exceeded')
                for key in totals:
                    totals[key] += summary[key]
            except (OSError,KeyError,TypeError,ValueError):
                unknown.append(spec['cell']+'/'+phase.lower())
    return {'known_logical_points':totals['logical_points'],'known_physical_points':totals['physical_points'],
            'known_numerical_entries':totals['actual_numerical_entries'],'unknown_cost_cells':unknown}


def execute(run):
    require(not (run / 'execution_started.json').exists(),'Already started; no automatic repeat')
    verify_execution(run)
    schedule = read(run / 'schedule.json')
    historical_max = read(run / 'resource_budget.json')['historical_max_cell_bytes']
    wall = time.perf_counter()
    save(run / 'execution_started.json',{'utc':utc(),'pid':os.getpid(),'deadline_epoch_seconds':time.time()+MAX_SECONDS})
    completed,dispatched,errors,exits,active = [],[],[],[],{}
    env = dict(os.environ)
    env.update({name:'1' for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS')})
    env['PYTHONDONTWRITEBYTECODE'] = '1'

    def finished(proc,spec,code,began,epoch,drained=False):
        row = {'cell':spec['cell'],'pair_order':spec['pair_order'],'exit_code':code,'launcher_pid':proc.pid,
            'started_epoch_seconds':epoch,'finished_epoch_seconds':time.time(),'process_wall_seconds':time.perf_counter()-began,
            'drained_after_controller_error':drained,'utc':utc()}
        exits.append(row)
        save(run / 'controller_exits.json',exits)
        if code == 0 and (run / spec['cell'] / 'completion_manifest.json').exists():
            completed.append(spec['cell'])
        else:
            errors.append(row)

    def progress():
        row = {'status':'RUNNING' if not errors else 'DRAINING_AFTER_ERROR','run':run.name,'scheduled':CELLS,
            'completed':len(completed),'dispatched':len(dispatched),'active':list(active),'errors':errors,
            'utc':utc(),'elapsed_seconds':time.perf_counter()-wall}
        save(run / 'controller_progress.json',row); save(ACTIVE,row)

    try:
        with (run / 'controller_dispatch.jsonl').open('x',encoding='utf-8') as journal:
            for pair_index in range(32):
                if errors:
                    break
                forecast = storage_forecast(run,schedule,completed,historical_max)
                save(run / 'storage_forecast.json',forecast)
                if forecast['free_bytes'] < forecast['required_free_bytes'] or time.perf_counter()-wall >= MAX_SECONDS:
                    errors.append({'error':'STORAGE_FORECAST_OR_DEADLINE_STOP_NEW_DISPATCH','forecast':forecast}); break
                accounting = costs(run,dispatched)
                if accounting['unknown_cost_cells']:
                    errors.append({'error':'UNKNOWN_COMPLETED_COST_STOP_NEW_DISPATCH',**accounting}); break
                pair = [s for s in schedule if s['pair_order'] == pair_index]
                require(len(pair) == 2,'Invalid fixed pair')
                for spec in pair:
                    cell = run / spec['cell']
                    require(not (cell / 'worker_started.json').exists(),'Repeated dispatch forbidden')
                    out,err = (cell / 'stdout.log').open('x',encoding='utf-8'),(cell / 'stderr.log').open('x',encoding='utf-8')
                    began,epoch = time.perf_counter(),time.time()
                    try:
                        proc = subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'worker','--run',str(run),'--cell',str(cell)],
                            cwd=ROOT,env=env,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    except BaseException:
                        out.close(); err.close(); raise
                    active[spec['cell']] = (proc,spec,out,err,began,epoch)
                    dispatched.append(spec)
                    append(journal,{'cell':spec['cell'],'pair_order':pair_index,'context_order':spec['context_order'],
                        'arm_position':spec['arm_position'],'launcher_pid':proc.pid,'started_epoch_seconds':epoch,'utc':utc()})
                while active:
                    for key,(proc,spec,out,err,began,epoch) in list(active.items()):
                        code = proc.poll()
                        if code is None:
                            continue
                        out.close(); err.close(); del active[key]
                        finished(proc,spec,code,began,epoch)
                    if (shutil.disk_usage(ROOT).free < MIN_FREE or time.perf_counter()-wall >= MAX_SECONDS) and not errors:
                        errors.append({'error':'LOW_DISK_OR_DEADLINE_DRAIN_STARTED_WORKERS'})
                    progress()
                    if active:
                        time.sleep(.1)
    except BaseException as exc:
        errors.append({'error':repr(exc),'traceback':traceback.format_exc()})
    finally:
        while active:
            for key,(proc,spec,out,err,began,epoch) in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                out.close(); err.close(); del active[key]
                finished(proc,spec,code,began,epoch,True)
            try:
                progress()
            except OSError:
                pass
            if active:
                time.sleep(.1)
    accounting = costs(run,dispatched)
    require(accounting['known_physical_points'] <= TOTAL_CAP,'Total physical cap exceeded')
    summary = {'status':'COMPLETE_CELLS_PENDING_AUDIT' if not errors and not accounting['unknown_cost_cells'] and len(completed) == CELLS else 'INCONCLUSIVE',
        'scheduled':CELLS,'completed':completed,'dispatched':dispatched,'exits':exits,'errors':errors,**accounting,
        'wall_seconds':time.perf_counter()-wall,'utc':utc()}
    save(run / 'controller_completion.json',summary)
    save(ACTIVE,{'status':summary['status'],'run':run.name,'scheduled':CELLS,'completed':len(completed),**accounting,'errors':errors,'utc':utc()})
    return 0 if summary['status'] == 'COMPLETE_CELLS_PENDING_AUDIT' else 1


def finalize(run):
    require(not (run / 'completion_manifest.json').exists(),'Run already sealed')
    verify_execution(run)
    audit,report = read(run / 'audit/summary.json'),read(run / 'derived_report/summary.json')
    cache.verify_manifest(run / 'audit',required=('summary.json',))
    cache.verify_manifest(run / 'derived_report',required=('summary.json','all_cells.json','all_pairs.json','context_summary.json','curve_data.json','report.md'))
    require(audit['status'] == 'PASS' and audit['audited_cells'] == CELLS,'Independent audit incomplete')
    controller = read(run / 'controller_completion.json')
    require(controller['status'] == 'COMPLETE_CELLS_PENDING_AUDIT' and not controller['errors']
            and not controller['unknown_cost_cells'] and len(controller['completed']) == CELLS
            and len(controller['exits']) == CELLS and all(row['exit_code'] == 0 for row in controller['exits']),
            'Controller has incomplete or failed tasks')
    require(report['status'] == 'AUDITED_DESCRIPTIVE_EARLY_INTERVENTION_PILOT', 'Unexpected scientific report status')
    require(report['source_bindings']['audit/summary.json'] == sha(run / 'audit/summary.json')
        and report['source_bindings']['controller_completion.json'] == sha(run / 'controller_completion.json'),'Report binding differs')
    terminal = {'status':'COMPLETE_EARLY_PILOT_AUDITED_AND_REPORTED','scientific_status':report['status'],
        'run':run.name,'utc':utc(),'audit_sha256':sha(run / 'audit/summary.json'),
        'report_sha256':sha(run / 'derived_report/summary.json'),'report_manifest_sha256':sha(run / 'derived_report/completion_manifest.json'),
        'physical_points':audit['physical_points'],'numerical_entries':audit['numerical_entries'],
        'new_training':0,'all_valid_failures_preserved':True}
    save(run / 'terminal_status.json',terminal); seal(run)
    save(ACTIVE,{**terminal,'status':'COMPLETE_EARLY_PILOT_SEALED','completion_sha256':sha(run / 'completion_manifest.json')})
    return terminal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=('prepare','worker','phase-worker','execute','finalize'))
    parser.add_argument('--run',type=Path); parser.add_argument('--cell',type=Path); parser.add_argument('--tests',type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(prepare(args.tests.resolve())); return 0
    run = args.run.resolve()
    require(run.parent == ROOT / 'runs','Run outside project')
    if args.command == 'worker':
        return worker(run,args.cell.resolve())
    if args.command == 'phase-worker':
        return phase_worker(run,args.cell.resolve())
    if args.command == 'finalize':
        print(finalize(run)); return 0
    return execute(run)


if __name__ == '__main__':
    raise SystemExit(main())


