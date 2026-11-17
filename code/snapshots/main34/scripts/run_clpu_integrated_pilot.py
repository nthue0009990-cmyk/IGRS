"""Cold 64-cell integration of fixed actors, protected search and equivalent cache.

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

STUDY = 'CLPU_INTEGRATED_PROTECTED_PILOT_V1'
SEEDS = (17, 41, 73)
SOURCES = {
    'cache': ('20261030T123732_313636Z_clpu_equivalent_cache_pilot', 'f3f1bbfa9a4eeecbdec22b55aad4c80550db00f360b3326e6a3acc764694e040'),
    'window': (cache.SOURCE_RUN, cache.SOURCE_SHA),
    'tail': ('20261027T045048_590910Z_clpu_tail_pilot', '3661145543e8d84c8771a6a3d13a24bd7336d1a1ba3eb2699a3efa545d47bb8e'),
    'd059': ('20261024T073617_269327Z_nature34_clpu_branch', '4c3777f938d21f074b40db56c82cd62ca1b956f6aa7d79215832bf192745c56f'),
    'd060': ('20261025T123350_150996Z_d060_main', 'b2f85e5f21d4e1f316c018a7d53428149f750ae32443fda7c3e572c2333ce7a6'),
}
SLOTS = ('window_reference', 'protected_rule', *(f'protected_bc{s}' for s in SEEDS),
         *(f'protected_ppo{s}' for s in SEEDS))
PROTOCOL = 'docs/clpu_integrated_pilot_protocol_v1.md'
ACTIVE = ROOT / 'artifacts/clpu_integrated_pilot_active.json'
MODEL_REVIEW = ROOT / 'artifacts/clpu_integrated_model_preflight_20261030'
MIN_FREE, REPORT_RESERVE = 12 * 2**30, 2 * 2**30
MAX_SECONDS, WORKER_SECONDS, TOTAL_CAP, CELLS = 14400, 3600, 16384, 64
_WORKER_ENTERED = False


def settings():
    return {'logical_point_budget': 256, 'physical_point_budget': 256,
        'window_expansions': 256, 'baseline_expansions': 256, 'continuation_expansions': 256,
        'max_selected_decisions': 768, 'max_workers': 2, 'cooperative_seconds': MAX_SECONDS,
        'worker_seconds': WORKER_SECONDS, 'total_physical_budget': TOTAL_CAP,
        'minimum_free_disk_bytes': MIN_FREE, 'forecast_margin': 1.25,
        'report_source_reserve_bytes': REPORT_RESERVE, 'cache_variant': 'electrical_equivalent',
        'no_training': True, 'no_historical_answers': True, 'no_cross_cell_cache': True,
        'no_automatic_retry': True, 'same_cell_baseline_continuation_cache': True,
        'dispatch': 'Hashed context and initial slot order; Latin rotation; adjacent two-worker pairs drain before next pair and context',
        'timing': 'Same-run cold full-process and worker/computation timing; no old-run speed comparator',
        'aggregation': 'Average seeds within context before averaging eight contexts'}


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
    ordered = sorted(contexts, key=lambda c: hash_order('clpu-integrated-context-order-v1|', canonical(c)))
    slots = sorted(SLOTS, key=lambda s: hash_order('clpu-integrated-arm-order-v1|', s))
    rows = []
    for rank, context in enumerate(ordered):
        for position in range(8):
            slot = slots[(position + rank) % 8]
            seed = next((s for s in SEEDS if slot.endswith(str(s))), 0)
            arm = slot[:-len(str(seed))] if seed else slot
            method = 'bc_raw' if arm == 'protected_bc' else 'ppo_raw' if arm == 'protected_ppo' else 'beam'
            model = lookup.get((method,seed))
            rows.append({**context, 'arm': arm, 'arm_slot': slot, 'seed': seed, 'method': method,
                'phase': 'evaluation', 'group': 'integrated', 'representation': 'raw', 'deterministic': True,
                'point_budget': 256, 'logical_point_budget': 256, 'physical_point_budget': 256,
                'continuation_expansions': 0 if arm == 'window_reference' else 256,
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
    for name in ('test_run_clpu_integrated_pilot', 'test_report_clpu_integrated_pilot',
                 'test_nature34_clpu_integrated', 'test_clpu_integrated_audit'):
        require(any(name in c.get('classname','') for c in cases), 'Missing component checks: '+name)
    return len(cases)


def source_names():
    names = cache.source_names()
    names += ['scripts/' + name for name in ('run_clpu_integrated_pilot.py', 'audit_clpu_integrated_pilot.py',
        'report_clpu_integrated_pilot.py', 'plot_clpu_integrated_pilot.py', 'nature34_clpu_tail_execution.py',
        'nature34_clpu_branch_execution.py', 'audit_nature34_clpu_tail.py', 'audit_nature34_clpu_branch_learning.py')]
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT / 'tests').glob('*clpu_integrated*.py')]
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
    d060, dm = sources['d060']
    require(sha(d060 / 'model_bindings.json') == dm['model_bindings.json'], 'D060 model binding changed')
    target_bindings = read(d060 / 'model_bindings.json')
    learning, lm = sources['d059']
    for m in models:
        name, digest = m['checkpoint'], m['source_sha256']
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
    historical_max = max(sizes + tail_sizes)
    return {'contexts': 8, 'cells': CELLS, 'physical_points': TOTAL_CAP,
        'numerical_entries': CELLS * 3 + 4 * TOTAL_CAP, 'workers': 2,
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
    source, sm = sources['cache']
    for name in ('config.json','population.json','selected_scenarios.json','selected_events.json','training_contexts.json','source_cells.json'):
        require(sha(source / name) == sm[name], 'Inherited input changed: '+name)
    contexts = [{k:c[k] for k in ('context_id','scenario_id','event_id')}
                for c in read(source / 'source_cells.json')]
    require({(c['scenario_id'],c['event_id']) for c in contexts}
        == {(c['scenario_id'],c['event_id']) for c in read(source / 'training_contexts.json')}, 'Training population differs')
    cfg = deepcopy(read(source / 'config.json'))
    require(cfg['branch_search'] == asdict(BranchConfig()), 'Original search configuration changed')
    cfg.update(integrated_study=STUDY, integrated_pilot=settings(),
        tail_search={'continuation_expansions':256, 'enable_exact_failed_prefix':True, 'enable_tail_verification':True})
    budget = resource_estimate(sources, tests)
    require(budget['free_bytes_at_prepare'] >= budget['initial_required_free_bytes'], 'Dynamic storage forecast not met')
    run = create_run('clpu_integrated_pilot', cfg)
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
        save(run / 'resource_budget.json', budget)
        save(run / 'schedule.json', schedule)
        for spec in schedule:
            save(run / spec['cell'] / 'spec.json', spec)
        shutil.copyfile(tests_path, run / 'software_tests.xml')
        shutil.copyfile(ROOT / PROTOCOL, run / 'protocol.md')
        save(run / 'authorization.json', {'utc':utc(), 'execution_authorized':True, 'user_local_date':'2026-10-30',
            'scope':'One fixed 64-cell cached protected-search integration pilot; six existing actors, no training or parameter scan',
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
    require(cfg['integrated_study'] == STUDY and cfg['integrated_pilot'] == settings()
        and cfg['branch_search'] == asdict(BranchConfig()), 'Execution contract changed')
    require(read(run / 'source_binding.json') == source_binding(), 'Scientific source binding changed')
    require(cache.physical_scope(run,prepared['source_files']) == read(run / 'physical_scope.json'), 'Complete physical scope changed')
    require(read(run / 'schedule.json') == schedule_for(read(run / 'source_cells.json'),read(run / 'model_bindings.json')), 'Frozen schedule changed')
    for folder in ('model_preflight','scope_review'):
        binding = read(run / (folder + '_binding.json'))
        require(sha(run / folder / 'completion_manifest.json') == binding['completion_sha256']
            and sha(run / folder / 'summary.json') == binding['summary_sha256'], 'Independent preflight binding changed')
        cache.verify_manifest(run / folder,required=('summary.json',))
        require(read(run / folder / 'summary.json')['status'] == 'PASS', 'Independent preflight did not pass')
    cache.verify_exact_key_review(run)


def physical_identity(run,spec,env):
    return {'study':STUDY,'freeze_sha256':sha(run / 'prephysical_freeze_manifest.json'),
        'scenario_id':spec['scenario_id'],'event_id':spec['event_id'],'initial_state':numeric_lists(env.initial.to_dict()),
        'condition_sha256':sha(run / 'condition.json'),'physical_config_sha256':sha(run / 'physical_config.json'),
        'author_static_metadata_sha256':sha(run / 'author_static_metadata.json'),'source_provenance_sha256':sha(run / 'source_provenance.json')}


def checkpoint(search, count, worker_started, computation_started, protected):
    engine = search.search if protected else search
    best = engine.incumbent
    certified = bool(best and best.get('certified_complete') is True and best.get('validation',{}).get('status') == 'COMPLETE')
    now = time.perf_counter()
    return {'search_choices':count,'logical_points':search.oracle.points,'physical_points':search.oracle.physical_points,
        'numerical_entries':search.oracle.factory.stats()['returned']['numerical_solution_entries'],
        'requests':search.oracle.requests,'quality_pct':100 * search.quality if certified else None,
        'fully_certified':certified,'worker_elapsed_seconds':now-worker_started,
        'elapsed_seconds':now-computation_started,
        'phase':search.phase if protected else 'DONE' if search.done else 'WINDOW'}


def phase_costs(checkpoints, baseline):
    def delta(left,right):
        return {k:right[k]-left.get(k,0) for k in ('logical_points','physical_points','numerical_entries','requests','worker_elapsed_seconds','elapsed_seconds')}
    if not checkpoints:
        return {}
    if baseline is None:
        return {'WINDOW':delta({},checkpoints[-1])}
    return {'BASELINE':delta({},baseline),'AFTER_BASELINE':delta(baseline,checkpoints[-1])}


def collect_window(search,spec,transition,decision,stage,step,after_step,computation_started):
    """Same observation/decision journals as protected arms, direct window policy."""
    from nature34_clpu_branch_execution import _observation
    from nature34_clpu_tail_execution import check_deadline
    count = 0
    timer = {k:0. for k in ('observation_seconds','selection_seconds','search_step_seconds','journal_seconds')}
    stage({'stage':'WINDOW','transition_boundary':0,'oracle':{'requests':search.oracle.requests,'points':search.oracle.points},'utc':utc()})
    while not search.done:
        check_deadline(spec)
        started = time.perf_counter()
        raw, associated = _observation(search,'raw'),_observation(search,'associated')
        require(raw['branch_ids'] == associated['branch_ids'] and raw['mask'] == associated['mask'], 'Window representation changed branches')
        timer['observation_seconds'] += time.perf_counter()-started
        started = time.perf_counter(); action = search.rule_action('beam')
        timer['selection_seconds'] += time.perf_counter()-started
        row = {'transition_id':count,'stage_transition_id':count,'stage':'WINDOW','obs':raw,'obs_raw':raw,
            'obs_associated':associated,'mask':raw['mask'],'action':action,'branch_id':raw['branch_ids'][action],
            'logprob':None,'value':None,'method':'beam','representation':'raw','seed':0,'sampling_seed':0,'checkpoint_sha256':None,
            'oracle_before':{'requests':search.oracle.requests,'points':search.oracle.points}}
        began = time.perf_counter(); decision({**row,'status':'STARTED'})
        step_row = {'transition_id':count,'status':'STARTED','method':'beam','action':action,'branch_id':row['branch_id'],
            'window_branch_ids':raw['branch_ids'],'oracle_before':{**row['oracle_before'],'logical_points':search.oracle.points,'physical_points':search.oracle.physical_points},
            'elapsed_seconds':time.perf_counter()-computation_started}
        step(step_row); timer['journal_seconds'] += time.perf_counter()-began
        began = time.perf_counter(); response = search.step(action)
        timer['search_step_seconds'] += time.perf_counter()-began
        require(isinstance(response,dict) and type(response.get('done')) is bool and math.isfinite(float(response['reward'])), 'Malformed window response')
        row.update(reward=float(response['reward']),done=response['done'],info=numeric_lists({k:v for k,v in response.items() if k not in ('reward','done')}),
            oracle_after={'requests':search.oracle.requests,'points':search.oracle.points})
        began = time.perf_counter(); transition(row)
        decision({k:v for k,v in {**row,'status':'COMPLETED'}.items() if k in ('transition_id','stage_transition_id','stage','status','action','branch_id','reward','done','info','oracle_before','oracle_after')})
        step({**step_row,'status':'COMPLETED','response':numeric_lists(response),'elapsed_seconds':time.perf_counter()-computation_started,
            'oracle_after':{**row['oracle_after'],'logical_points':search.oracle.points,'physical_points':search.oracle.physical_points}})
        count += 1; after_step(count)
        timer['journal_seconds'] += time.perf_counter()-began
        require(count <= 256, 'Window expansion cap exceeded')
    stage({'stage':'DONE','transition_boundary':count,'oracle':{'requests':search.oracle.requests,'points':search.oracle.points},'utc':utc()})
    return {'search_decisions':count,'stage_decisions':{'WINDOW':count},'stage_component_seconds':{'WINDOW':timer}}


def worker(run,cell):
    global _WORKER_ENTERED
    started = time.perf_counter()
    started_epoch = time.time()
    require(not _WORKER_ENTERED, 'Each task requires a cold worker')
    _WORKER_ENTERED = True
    from run_nature34_clpu_pilot import make_env
    from dnr.nature34_clpu_branch import BranchConfig
    from dnr.nature34_clpu_window import WindowBranchSearch
    from dnr.nature34_clpu_integrated import CoverageTailBranchSearch
    from dnr.nature34_clpu_equivalent_cache import EquivalenceOracle
    from nature34_clpu_tail_execution import collect_episode,load_fixed_policy,guard_evidence,process_memory_evidence
    require(cell != run and cell.is_relative_to(run), 'Cell escaped')
    # A refused duplicate must not add an error file to a previously sealed cell.
    require(not any((cell / n).exists() for n in ('worker_started.json','result.json','error.json','points.jsonl.gz')), 'Repeated task forbidden')
    spec = read(cell / 'spec.json')
    baseline_real = None
    try:
        verify_execution(run)
        require(spec in read(run / 'schedule.json'), 'Unscheduled task')
        deadline = min(read(run / 'execution_started.json')['deadline_epoch_seconds'],started_epoch+WORKER_SECONDS)
        require(time.time() < deadline and shutil.disk_usage(ROOT).free >= MIN_FREE, 'Worker deadline/storage exhausted')
        runtime_spec = {**spec,'deadline_utc_epoch':deadline}
        with (cell / 'worker_started.json').open('x',encoding='utf-8') as stream:
            append(stream,{'pid':os.getpid(),'utc':utc(),'spec_sha256':sha(cell / 'spec.json'),'checkpoint_sha256':spec['checkpoint_sha256']})
        save(cell / 'worker_deadline.json',{'deadline_epoch_seconds':deadline,'worker_seconds':WORKER_SECONDS})
        model_started = time.perf_counter()
        model_path = run / spec['checkpoint'] if spec['checkpoint'] else None
        if model_path:
            require(sha(model_path) == spec['checkpoint_sha256'], 'Model changed')
        policy,generator = load_fixed_policy(model_path,runtime_spec)
        model_load_seconds = time.perf_counter()-model_started
        env = make_env(run,spec)
        save(cell / 'initial_state.json',env.initial.to_dict())
        identity = physical_identity(run,spec,env)
        save(cell / 'physical_identity.json',identity)
        preflight_seconds = time.perf_counter()-started
        computation_started = time.perf_counter()
        checkpoints = []
        with ExitStack() as stack:
            logs = {name:stack.enter_context(gzip.open(cell / (name+'.jsonl.gz'),'xt',encoding='utf-8',compresslevel=3))
                    for name in ('plans','transitions','decisions','steps')}
            logs.update({name:stack.enter_context((cell / (name+'.jsonl')).open('x',encoding='utf-8'))
                    for name in ('stage_events','tail_events','cost_checkpoints')})
            emit = lambda name,row: append(logs[name],numeric_lists(row))
            oracle = stack.enter_context(EquivalenceOracle(run,cell,env,256,'electrical_equivalent',read(run / 'physical_scope.json')))
            oracle.deadline_monotonic = time.perf_counter()+max(0,deadline-time.time())
            runtime_initialization_seconds = time.perf_counter()-computation_started
            cfg = BranchConfig(**read(run / 'config.json')['branch_search'])
            if spec['protected']:
                search = CoverageTailBranchSearch(env,oracle,cfg,physics_identity=identity,continuation=True,extra_branch_expansions=256,
                    record_callback=lambda row:emit('plans',row),event_callback=lambda row:emit('tail_events',row))
            else:
                search = WindowBranchSearch(env,oracle,cfg,record_callback=lambda row:emit('plans',row))

            def record_cost(count):
                row = checkpoint(search,count,started,computation_started,spec['protected'])
                checkpoints.append(row); emit('cost_checkpoints',row)

            def save_baseline(current,count):
                nonlocal baseline_real
                require(baseline_real is None,'Baseline overwrite forbidden')
                engine = numeric_lists(current.baseline_checkpoint())
                baseline = engine['result']
                save(cell / 'baseline_result.json',baseline)
                baseline_real = checkpoint(search,count,started,computation_started,True)
                save(cell / 'baseline_checkpoint.json',{'engine_checkpoint':engine,'baseline_result_sha256':sha(cell / 'baseline_result.json'),
                    'transition_boundary':count,'attempt_boundary':len(baseline['attempts']),
                    'oracle_boundary':{'requests':oracle.requests,'points':oracle.points},'real_cost':baseline_real,
                    'elapsed_seconds':time.perf_counter()-started,'physical_identity_sha256':sha(cell / 'physical_identity.json')})

            record_cost(0)
            if spec['protected']:
                def transition(row):
                    emit('transitions',row); record_cost(row['transition_id']+1)
                collected = collect_episode(search,runtime_spec,transition,lambda row:emit('decisions',row),
                    lambda row:emit('stage_events',row),save_baseline,policy,generator)
                require(baseline_real is not None,'Missing baseline boundary')
                result = numeric_lists(search.result())
                protection = guard_evidence(read(cell / 'baseline_result.json'),result)
            else:
                collected = collect_window(search,runtime_spec,lambda row:emit('transitions',row),lambda row:emit('decisions',row),
                    lambda row:emit('stage_events',row),lambda row:emit('steps',row),record_cost,computation_started)
                result = numeric_lists(search.result())
                result.update(logical_points=oracle.points,actual_physical_points=oracle.physical_points,plan_point_id_semantics='logical_point_id')
                protection = None
        if model_path:
            require(sha(model_path) == spec['checkpoint_sha256'],'Fixed model changed during execution')
        summary = read(cell / 'equivalent_cache_summary.json')
        require(summary['physical_points'] <= summary['logical_points'] <= 256,'Cumulative cap exceeded')
        final_cost = checkpoint(search,collected['search_decisions'],started,computation_started,spec['protected'])
        result.update(spec=spec,oracle=summary,checkpoints=checkpoints,**collected,
            counts={k:result[k] for k in ('generated_unique_candidates','assessed_graph_candidates','checked_candidates','frontier_size')},
            baseline_real_cost=baseline_real,final_real_cost=final_cost,phase_costs=phase_costs([final_cost],baseline_real),guard_audit=protection,
            baseline_result_sha256=sha(cell / 'baseline_result.json') if spec['protected'] else None,
            baseline_checkpoint_sha256=sha(cell / 'baseline_checkpoint.json') if spec['protected'] else None,
            preflight_elapsed_seconds=preflight_seconds,model_load_seconds=model_load_seconds,
            runtime_initialization_seconds=runtime_initialization_seconds,
            computation_elapsed_seconds=time.perf_counter()-computation_started,elapsed_seconds=time.perf_counter()-started,
            timing_semantics='Worker time includes imports after entry, preflight/model/environment and computation through journal close; controller process wall includes launch/import/final serialization/seal/exit',
            training_interactions=0,optimizer_updates=0,new_training=0,trajectory_file='transitions.jsonl.gz',
            checkpoint_sha256=spec['checkpoint_sha256'],process_memory=process_memory_evidence(),
            formal_holdout_numerical_access=False,tune_numerical_access=False,clpu_holdout_access=False)
        save(cell / 'result.json',result)
        seal(cell)
        return 0
    except BaseException as exc:
        save(cell / 'error.json',{'status':'STOPPED_RELIABILITY_OR_IMPLEMENTATION_ERROR','spec':spec,'error':repr(exc),
            'traceback':traceback.format_exc(),'elapsed_seconds':time.perf_counter()-started,'partial_evidence_retained':True,'no_automatic_retry':True})
        return 1


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
                accounting = cache.costs(run,dispatched)
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
    accounting = cache.costs(run,dispatched)
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
    require(report['source_bindings']['audit/summary.json'] == sha(run / 'audit/summary.json')
        and report['source_bindings']['controller_completion.json'] == sha(run / 'controller_completion.json'),'Report binding differs')
    terminal = {'status':'COMPLETE_INTEGRATED_PILOT_AUDITED_AND_REPORTED','scientific_status':report['status'],
        'run':run.name,'utc':utc(),'audit_sha256':sha(run / 'audit/summary.json'),
        'report_sha256':sha(run / 'derived_report/summary.json'),'report_manifest_sha256':sha(run / 'derived_report/completion_manifest.json'),
        'physical_points':audit['physical_points'],'numerical_entries':audit['numerical_entries'],
        'new_training':0,'all_valid_failures_preserved':True}
    save(run / 'terminal_status.json',terminal); seal(run)
    save(ACTIVE,{**terminal,'status':'COMPLETE_INTEGRATED_PILOT_SEALED','completion_sha256':sha(run / 'completion_manifest.json')})
    return terminal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=('prepare','worker','execute','finalize'))
    parser.add_argument('--run',type=Path); parser.add_argument('--cell',type=Path); parser.add_argument('--tests',type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(prepare(args.tests.resolve())); return 0
    run = args.run.resolve()
    require(run.parent == ROOT / 'runs','Run outside project')
    if args.command == 'worker':
        return worker(run,args.cell.resolve())
    if args.command == 'finalize':
        print(finalize(run)); return 0
    return execute(run)


if __name__ == '__main__':
    raise SystemExit(main())
