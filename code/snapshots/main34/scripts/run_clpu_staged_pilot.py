"""Common staged-candidate rule/BC pilot, conditional on a sealed opportunity probe."""
from __future__ import annotations

import argparse
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

SOURCE = ROOT / 'runs/20261031T053323_555256Z_clpu_deferred_pilot'
SOURCE_SHA = '4dfecffaee422d8d712e4b8bb2c01d95b22d97eb3721f947866a4861b4c1f56c'
MODELS_SOURCE = ROOT / 'runs/20261031T020246_028867Z_clpu_early_pilot'
MODELS_SHA = 'bed648a067dd82c3c52e0250de25714efcea7cca5b69347fad399e72e9e57747'
STUDY = 'CLPU_STAGED_COMMON_PILOT_V1'
PROTOCOL = 'docs/clpu_staged_pickup_pilot_protocol_v1.md'
ACTIVE = ROOT / 'artifacts/clpu_staged_pilot_active.json'
GRAPH_PROFILE = ROOT / 'artifacts/clpu_staged_pickup_graph_20261031_v2'
GRAPH_PROFILE_SHA = '96227b394dd6a447cdbf06470c9fad6581653c0606caf51afdb22bb566143c32'
SEEDS = (17, 41, 73)
SLOTS = ('staged_window', *(f'staged_bc{s}' for s in SEEDS))
CELLS = 32
MIN_FREE, RESERVE = 12 * 2**30, 2 * 2**30
MAX_SECONDS, WORKER_SECONDS, TOTAL_CAP = 3600, 900, CELLS * 512
_WORKER_ENTERED = False


def settings():
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


def schedule_for(contexts, models):
    require(len(contexts)==8 and {c['context_id'] for c in contexts}==set(range(8)), 'All eight contexts required')
    require(len({c['scenario_id'] for c in contexts})==8, 'Fixed faults changed')
    lookup={(m['method'],m['seed']):m for m in models}
    require(len(models)==len(lookup)==6 and set(lookup)=={(m,s) for m in ('bc_raw','ppo_raw') for s in SEEDS},'Six fixed models required')
    ordered=sorted(contexts,key=lambda c:hashlib.sha256(('staged-context-v1|'+canonical(c)).encode()).hexdigest())
    slots=sorted(SLOTS,key=lambda s:hashlib.sha256(('staged-slots-v1|'+s).encode()).hexdigest())
    rows=[]
    for rank,c in enumerate(ordered):
        for position in range(4):
            slot=slots[(rank+position)%4]
            seed=next((s for s in SEEDS if slot.endswith(str(s))),0)
            arm=slot[:-len(str(seed))] if seed else slot
            method='bc_raw' if arm=='staged_bc' else 'beam'
            model=lookup.get((method,seed))
            rows.append({**{k:c[k] for k in ('context_id','scenario_id','event_id')},
                'cell':f'episodes/{len(rows):03d}', 'episode_id':len(rows), 'pair_order':rank*2+position//2,
                'context_order':rank,'arm_position':position,'arm_slot':slot,
                'mode':'shared','arm':arm,'method':method,'seed':seed,
                'macro_schedule':'staged',
                'phase':'evaluation', 'group':arm, 'protected':True,
                'representation':'raw', 'deterministic':True,
                'checkpoint':model['checkpoint'] if model else None,
                'checkpoint_sha256':model['source_sha256'] if model else None,
                'point_budget':256, 'logical_point_budget':256, 'physical_point_budget':512,
                'total_logical_point_budget':512, 'window_variant':'staged_common_pool_coverage_v1'})
    return rows


def phase_spec(spec, phase):
    require(phase in ('EARLY','GUARD'), 'Unknown phase')
    return {**spec,'parent_cell':spec['cell'],'cell':spec['cell']+'/'+phase,'search_phase':phase,
        'method':spec['method'] if phase=='EARLY' else 'beam',
        'seed':spec['seed'] if phase=='EARLY' else 0,
        'checkpoint':spec['checkpoint'] if phase=='EARLY' else None,
        'checkpoint_sha256':spec['checkpoint_sha256'] if phase=='EARLY' else None,
        'point_budget':256,'logical_point_budget':256,'physical_point_budget':256}


def source_names():
    names=read(SOURCE/'prepared.json')['source_files']
    names += [PROTOCOL,'src/dnr/nature34_clpu_staged_pickup.py','src/dnr/nature34_clpu_staged_control.py',
        'scripts/run_clpu_staged_pilot.py','scripts/audit_clpu_staged_pilot.py',
        'tests/test_nature34_clpu_staged_pickup.py','tests/test_nature34_clpu_staged_control.py',
        'tests/test_run_clpu_staged_pilot.py','tests/test_clpu_staged_audit.py']
    return sorted(set(names))


def validate_tests(path):
    cases=list(ET.parse(path).getroot().iter('testcase'))
    require(cases and not any(list(c.iter('failure'))+list(c.iter('error'))+list(c.iter('skipped')) for c in cases), 'Tests incomplete')
    for part in ('test_nature34_clpu_staged_pickup','test_nature34_clpu_staged_control','test_run_clpu_staged_pilot','test_clpu_staged_audit'):
        require(any(part in c.get('classname','') for c in cases), 'Missing component '+part)
    return len(cases)


def source_binding():
    return {'source_run':SOURCE.name,'source_completion_sha256':SOURCE_SHA,
        'models_run':MODELS_SOURCE.name,'models_completion_sha256':MODELS_SHA}


def bind_models(run):
    require(sha(MODELS_SOURCE/'completion_manifest.json')==MODELS_SHA,'Model source seal changed')
    mm=normalize_manifest(read(MODELS_SOURCE/'completion_manifest.json'))
    models=read(MODELS_SOURCE/'model_bindings.json')
    bound_copy(MODELS_SOURCE,run/'model_bindings.json',mm,'model_bindings.json')
    shutil.copyfile(MODELS_SOURCE/'completion_manifest.json',run/'model_source_completion_manifest.json')
    for item in models:
        require(mm[item['checkpoint']]==item['source_sha256'],'Unbound model')
        bound_copy(MODELS_SOURCE,run/item['checkpoint'],mm,item['checkpoint'])
    for name in mm:
        if name.startswith('model_preflight/') or name=='model_preflight_binding.json':
            bound_copy(MODELS_SOURCE,run/name,mm,name)
    # Parent seals intentionally omit nested completion manifests. Its digest
    # is retained in the separately parent-bound model preflight binding.
    binding=read(run/'model_preflight_binding.json')
    nested=MODELS_SOURCE/'model_preflight/completion_manifest.json'
    require(sha(nested)==binding['completion_sha256'],'Nested model review seal differs')
    shutil.copyfile(nested,run/'model_preflight/completion_manifest.json')
    cache.verify_manifest(run/'model_preflight',required=('summary.json',))
    return models


def partition_schedule(schedule, graph_contexts):
    require(len(schedule)==CELLS and len(graph_contexts)==8
            and {row['context_id'] for row in graph_contexts}==set(range(8)), 'Complete graph-stage population required')
    eligible=[]
    for row in graph_contexts:
        pool=row['generation']
        require(len(pool['candidates'])<=24 and all(math.isfinite(c['graph_energy_kwh']) for c in pool['candidates']), 'Invalid complete graph pool')
        if any(c['graph_energy_kwh']>row['reference_energy_kwh'] for c in pool['candidates']):
            eligible.append(row['context_id'])
    phase1=[s['cell'] for s in schedule if s['context_id'] in eligible]
    phase2=[s['cell'] for s in schedule if s['context_id'] not in eligible]
    return {'schema':'clpu-staged-fixed-partition-v1','original_contexts':8,'scheduled_cells':CELLS,
        'eligible_contexts':sorted(eligible),'phase1_cells':phase1,'phase2_cells':phase2,
        'phase1_rule':'any complete staged graph objective strictly above same-context historical certified reference',
        'phase2_rule':'any audited complete EARLY objective strictly above own complete GUARD',
        'contexts':graph_contexts,'new_physical_points':0,'new_training':0}


def check_graph_profile(path,stage_plan):
    require(sha(path/'completion_manifest.json')==GRAPH_PROFILE_SHA,'Staged graph profile root seal changed')
    entries=cache.verify_manifest(path,required=('all_contexts.json','summary.json','frozen.json'))
    frozen=read(path/'frozen.json');summary=read(path/'summary.json')
    require(frozen['source_completion_sha256']==SOURCE_SHA and summary['contexts']==8
            and summary['new_physical_points']==summary['new_training']==0,'Wrong zero-solver graph profile')
    for name in ('src/dnr/nature34_clpu_staged_pickup.py','src/dnr/nature34_clpu_staged_control.py'):
        require(frozen['source_sha256'][name]==sha(ROOT/name),'Generator/control changed after graph profile')
    profile={row['context_id']:row for row in read(path/'all_contexts.json')}
    require(set(profile)==set(range(8)),'Graph profile missing context')
    for row in stage_plan['contexts']:
        expected=profile[row['context_id']]
        require(all(row[key]==expected[key] for key in ('scenario_id','event_id','reference_energy_kwh',
                'reference_quality_pct','generation','generation_sha256')),'Frozen and freshly generated graph inventory differ')
    require(stage_plan['eligible_contexts']==summary['contexts_with_graph_potential'],'Graph-eligible contexts changed')
    return entries


def build_stage_plan(contexts,schedule,source_manifest):
    from audit_d059_supplement import NoSolverImports,no_solver_loaded
    guard=NoSolverImports();sys.meta_path.insert(0,guard)
    try:
        no_solver_loaded()
        from run_nature34_clpu_pilot import make_env
        from dnr.nature34_clpu_staged_pickup import generate_staged_pickup_candidates
        references={s['context_id']:s for s in read(SOURCE/'schedule.json') if s['arm']=='deferred_window'}
        rows=[]
        for context in sorted(contexts,key=lambda c:c['context_id']):
            spec=references[context['context_id']]
            name=spec['cell']+'/GUARD/phase_result.json'
            require(sha(SOURCE/name)==source_manifest[name],'Reference guard source changed')
            reference=read(SOURCE/name);best=reference['best_plan']
            require(reference['fully_certified'] and best['certified_complete']
                    and best['validation']['status']=='COMPLETE','Reference requires complete certificate')
            generation=generate_staged_pickup_candidates(make_env(SOURCE,spec))
            rows.append({**context,'reference_energy_kwh':best['validation']['energy_kwh'],
                'reference_quality_pct':reference['quality_pct'],'reference_source_file':name,
                'reference_source_sha256':source_manifest[name],'generation':generation,
                'generation_sha256':hashlib.sha256(canonical(generation).encode()).hexdigest()})
        no_solver_loaded()
        return partition_schedule(schedule,rows)
    finally:
        sys.meta_path.remove(guard)


def prepare(tests_path):
    from dnr.nature34_clpu_branch import BranchConfig
    count=validate_tests(tests_path)
    require(sha(SOURCE/'completion_manifest.json')==SOURCE_SHA,'Inherited source seal changed')
    sm=normalize_manifest(read(SOURCE/'completion_manifest.json'))
    for n in ('config.json','schedule.json','training_contexts.json','selected_scenarios.json','selected_events.json','prepared.json','audit/summary.json'):
        require(sha(SOURCE/n)==sm[n],'Inherited input changed: '+n)
    source_audit=read(SOURCE/'audit/summary.json')
    require(source_audit['status']=='PASS' and source_audit['audited_cells']==64 and not source_audit['errors'],'Complete source audit required')
    chosen=[s for s in read(SOURCE/'schedule.json') if s['arm']=='deferred_window']
    contexts=read(SOURCE/'contexts.json')
    require(sha(SOURCE/'contexts.json')==sm['contexts.json'],'Source contexts changed')
    models=read(MODELS_SOURCE/'model_bindings.json')
    schedule=schedule_for(contexts,models)
    stage_plan=build_stage_plan(contexts,schedule,sm)
    graph_files=check_graph_profile(GRAPH_PROFILE,stage_plan)
    cfg=deepcopy(read(SOURCE/'config.json'))
    require(cfg['branch_search']==asdict(BranchConfig()),'Original search changed')
    cfg.update(staged_study=STUDY,staged_pilot=settings())
    rates=[]
    for s in chosen:
        p=SOURCE/s['cell']; result=read(p/'result.json')
        rates.append(sum(f.stat().st_size for f in p.rglob('*') if f.is_file())/max(1,result['actual_physical_points']))
    budget={'historical_max_bytes_per_actual_point':max(rates),'total_physical_cap':TOTAL_CAP,
        'forecast_bytes':math.ceil(max(rates)*TOTAL_CAP*1.25),'minimum_free_bytes':MIN_FREE,'report_reserve_bytes':RESERVE,
        'free_bytes_at_prepare':shutil.disk_usage(ROOT).free,'successful_component_tests':count}
    budget['required_free_bytes']=MIN_FREE+RESERVE+budget['forecast_bytes']
    require(budget['free_bytes_at_prepare']>=budget['required_free_bytes'],'Storage forecast unmet')
    run=create_run('clpu_staged_pilot',cfg)
    try:
        for n in (*ROOT_INPUTS,'population.json','selected_scenarios.json','selected_events.json','training_contexts.json'):
            bound_copy(SOURCE,run/n,sm,n)
        for n in sm:
            if n.startswith('source_inputs/'):
                bound_copy(SOURCE,run/n,sm,n)
        save(run/'contexts.json',contexts)
        save(run/'stage_plan.json',stage_plan)
        for name in graph_files:
            target=run/'graph_profile'/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(GRAPH_PROFILE/name,target)
        shutil.copyfile(GRAPH_PROFILE/'completion_manifest.json',run/'graph_profile/completion_manifest.json')
        save(run/'graph_profile_binding.json',{'path':GRAPH_PROFILE.relative_to(ROOT).as_posix(),'completion_sha256':GRAPH_PROFILE_SHA})
        save(run/'source_binding.json',source_binding())
        shutil.copyfile(SOURCE/'completion_manifest.json',run/'source_completion_manifest.json')
        require(bind_models(run)==models,'Model bindings changed during preparation')
        historical=[]
        for spec in chosen:
            item={k:spec[k] for k in ('context_id','scenario_id','event_id')}
            item['source_cell']=spec['cell']; item['phases']={}
            for phase in ('GUARD',):
                original=spec['cell']+'/'+phase+'/phase_result.json'
                target=f'historical/c{spec["context_id"]:02d}/{phase}/phase_result.json'
                bound_copy(SOURCE,run/target,sm,original)
                item['phases'][phase]={'result_file':target,'source_file':original,'sha256':sm[original]}
            historical.append(item)
        save(run/'historical_bindings.json',historical)
        cache.bind_scope_review(run); cache.bind_exact_key_review(run)
        save(run/'resource_budget.json',budget); save(run/'schedule.json',schedule)
        for spec in schedule:
            save(run/spec['cell']/'spec.json',spec)
            for phase in ('EARLY','GUARD'):
                save(run/spec['cell']/phase/'spec.json',phase_spec(spec,phase))
        shutil.copyfile(tests_path,run/'software_tests.xml')
        shutil.copyfile(ROOT/PROTOCOL,run/'protocol.md')
        save(run/'authorization.json',{'utc':utc(),'execution_authorized':True,
            'scope':'User authorized staged public candidates shared by window and BC with complete Beam; all8 contexts and3 fixed BC seeds frozen, graph-eligible contexts first and audited gain required to continue; no training.',
            'learning':'Frozen BC only ranks current staged candidates; no B learner or trained staging decisions.'})
        names=source_names()
        for n in names:
            if 'source/'+n in sm: require(sha(ROOT/n)==sm['source/'+n],'Inherited code changed: '+n)
        save(run/'physical_scope.json',cache.physical_scope(run,names))
        freeze(run,names)
        save(ACTIVE,{'status':'FROZEN_NOT_STARTED','run':run.name,'scheduled':CELLS,'completed':0,'utc':utc()})
        return run
    except BaseException as exc:
        save(run/'preparation_error.json',{'error':repr(exc),'traceback':traceback.format_exc(),'new_physical_points':0})
        raise


def verify_execution(run):
    require(run.is_absolute() and run.parent==ROOT/'runs','Wrong run location')
    p=read(run/'prepared.json')
    require(p['status']=='FROZEN_PREPARED_NO_DSS' and sha(run/'prephysical_freeze_manifest.json')==p['freeze_sha256'],'Freeze changed')
    cache.verify_manifest(run,'prephysical_freeze_manifest.json',('schedule.json','physical_scope.json','contexts.json','stage_plan.json'))
    for n in p['source_files']: require(sha(ROOT/n)==sha(run/'source'/n),'Executing source changed: '+n)
    require(dependencies()==read(run/'dependencies.json'),'Dependencies changed')
    require(sha(Path(sys.executable))==read(run/'python_environment.json')['sha256'],'Interpreter changed')
    cfg=read(run/'config.json')
    require(cfg['staged_study']==STUDY and cfg['staged_pilot']==settings(),'Deferred contract changed')
    require(read(run/'source_binding.json')==source_binding(),'Source binding changed')
    require(read(run/'schedule.json')==schedule_for(read(run/'contexts.json'),read(run/'model_bindings.json')),'Schedule changed')
    staged=read(run/'stage_plan.json')
    require(staged==partition_schedule(read(run/'schedule.json'),staged['contexts']),'Fixed conditional partition changed')
    require(read(run/'graph_profile_binding.json')=={'path':GRAPH_PROFILE.relative_to(ROOT).as_posix(),'completion_sha256':GRAPH_PROFILE_SHA},'Graph profile binding changed')
    check_graph_profile(run/'graph_profile',staged)
    for model in read(run/'model_bindings.json'):
        require(sha(run/model['checkpoint'])==model['source_sha256'],'Model bytes changed')
    require(cache.physical_scope(run,p['source_files'])==read(run/'physical_scope.json'),'Physical scope changed')
    cache.verify_exact_key_review(run)
    for spec in read(run/'schedule.json'):
        require(read(run/spec['cell']/'spec.json')==spec,'Cell configuration changed')
        for phase in ('EARLY','GUARD'):
            require(read(run/spec['cell']/phase/'spec.json')==phase_spec(spec,phase),'Phase configuration changed')


def costs(oracle):
    return {'logical_points':oracle.points,'physical_points':oracle.physical_points,
        'numerical_entries':4*oracle.physical_points,'requests':oracle.requests}


def quality(search):
    p=search.incumbent
    complete=bool(p and p.get('certified_complete') is True and p.get('status')=='COMPLETE'
        and p.get('validation',{}).get('status')=='COMPLETE')
    return 100*search.quality if complete else None


def compact_proposal(proposal):
    record=numeric_lists(deepcopy(proposal))
    observation=record.pop('observation')
    compact=None if observation is None else {k:observation[k] for k in
        ('representation','candidate_features','global_features','mask','branch_ids')}
    digest=lambda x:hashlib.sha256(canonical(x).encode()).hexdigest()
    return record,compact,digest(observation),digest(compact)


def run_routes(service,env,branch_config,cell,spec,started,deadline,policy=None,generator=None,search_classes=None,expected_generation=None):
    from dnr.nature34_clpu_branch import BranchSearch
    from dnr.nature34_clpu_staged_control import StagedCertificationSearch, StagedMacroController
    classes=search_classes or {'EARLY':StagedCertificationSearch,'GUARD':BranchSearch}
    results,checkpoints={},[]
    previous_logical=previous_requests=previous_choices=0
    previous_quality=None
    with (cell/'cost_checkpoints.jsonl').open('x',encoding='utf-8') as cpstream:
        for phase in ('EARLY','GUARD'):
            require(time.time()<deadline,'Task deadline before phase')
            folder=cell/phase; ps=phase_spec(spec,phase)
            phase_started=time.perf_counter()
            save(folder/'initial_state.json',env.initial.to_dict())
            timers={k:0. for k in ('selection_seconds','search_step_seconds','journal_seconds')}
            with ExitStack() as stack:
                streams={n:stack.enter_context(gzip.open(folder/(n+'.jsonl.gz'),'xt',encoding='utf-8',compresslevel=3)) for n in ('plans','decisions')}
                emit=lambda name,row:append(streams[name],numeric_lists(row))
                oracle=stack.enter_context(service.view(phase))
                require(oracle.points==oracle.physical_points==oracle.requests==0,'Logical view not cold')
                search=classes[phase](env,oracle,branch_config,record_callback=lambda r:emit('plans',r))
                if phase=='EARLY':
                    if search.stage_generation is None:
                        save(folder/'candidate_pool.json',{'generation':None,'generation_sha256':None,
                            'status':'BOOTSTRAP_STOP_NOT_GENERATED','candidate_count':0})
                    require(search.stage_generation is not None,'Paid bootstrap did not complete staged generation')
                    pool=numeric_lists(search.stage_generation)
                    if expected_generation is not None:
                        require(pool==expected_generation,'Actual cold candidate generation differs from frozen pure-graph inventory')
                    save(folder/'candidate_pool.json',{'generation':pool,
                        'generation_sha256':hashlib.sha256(canonical(pool).encode()).hexdigest()})
                control=StagedMacroController(search,method=spec['method'],
                    policy=policy,generator=generator) if phase=='EARLY' else None
                count=0
                def cp(kind):
                    local=costs(oracle); q=quality(search)
                    qs=[x for x in (previous_quality,q) if x is not None]
                    row={'checkpoint_index':len(checkpoints),'phase':phase,'kind':kind,'phase_choices':count,
                        'search_choices':previous_choices+count,'logical_points':previous_logical+oracle.points,
                        'physical_points':service.physical_points,
                        'numerical_entries':service.factory.stats()['returned']['numerical_solution_entries'],
                        'requests':previous_requests+oracle.requests,'phase_logical_points':oracle.points,
                        'phase_physical_points':oracle.physical_points,'phase_numerical_entries':4*oracle.physical_points,
                        'phase_quality_pct':q,'quality_pct':max(qs) if qs else None,'fully_certified':bool(qs),
                        'guard_established':phase=='GUARD' and kind=='CLOSE' and q is not None,
                        'worker_elapsed_seconds':time.perf_counter()-started}
                    checkpoints.append(row); append(cpstream,row)
                cp('INITIAL')
                while not search.done:
                    require(time.time()<deadline,'Task deadline during search')
                    before=time.perf_counter()
                    proposal=control.prepare() if control else None
                    if control and proposal is None:
                        timers['selection_seconds']+=time.perf_counter()-before
                        break
                    action=proposal['action'] if control else search.rule_action('beam')
                    timers['selection_seconds']+=time.perf_counter()-before
                    if control:
                        compact,obs,raw_sha,compact_sha=compact_proposal(proposal)
                        identity={'transition_id':count,'phase_choice':count,'phase':phase,'proposal':compact,
                            'obs_raw_sha256':raw_sha,'obs_compact_sha256':compact_sha,
                            'checkpoint_sha256':spec['checkpoint_sha256'] if proposal['policy_used'] else None,
                            'oracle_before':costs(oracle)}
                    else:
                        identity={'transition_id':count,'phase_choice':count,'phase':phase,'method':'beam','action':action,'oracle_before':costs(oracle)}
                    before=time.perf_counter(); emit('decisions',{**identity,'status':'STARTED',**({'obs_compact':obs} if control else {})})
                    timers['journal_seconds']+=time.perf_counter()-before
                    require(time.time()<deadline,'Task deadline before apply')
                    before=time.perf_counter(); response=control.apply(proposal) if control else search.step(action)
                    timers['search_step_seconds']+=time.perf_counter()-before
                    require(type(response.get('done')) is bool,'Invalid step response')
                    if not control: require(math.isfinite(response['reward']),'Invalid guard reward')
                    before=time.perf_counter(); emit('decisions',{**identity,'status':'COMPLETED','response':response,'oracle_after':costs(oracle)})
                    count+=1; cp('DECISION')
                    timers['journal_seconds']+=time.perf_counter()-before
                    require(count<=(17 if control else 256),'Phase operation limit exceeded')
                result=numeric_lists(control.result() if control else search.result()); q=quality(search)
                require(search.done and search.stop_reason not in ('DEFERRED_RELIABILITY_OR_IMPLEMENTATION_ERROR',),
                    'Unfinished or erroneous EARLY route cannot establish success')
            cp('CLOSE')
            result.update(spec=ps,oracle=oracle.summary(),logical_points=oracle.points,
                actual_physical_points=oracle.physical_points,quality_pct=q,fully_certified=q is not None,
                search_decisions=count,policy_decisions=result.get('policy_decisions',0),
                graph_expansions=result.get('graph_expansions',count),verify_attempts=result.get('verify_attempts'),
                stage_component_seconds=timers,
                phase_elapsed_seconds=time.perf_counter()-phase_started,
                worker_start_seconds=phase_started-started,worker_end_seconds=time.perf_counter()-started)
            save(folder/'phase_result.json',result); results[phase]=result
            previous_logical+=oracle.points; previous_requests+=oracle.requests; previous_choices+=count
            previous_quality=checkpoints[-1]['quality_pct']
    return results,checkpoints


def worker(run,cell):
    global _WORKER_ENTERED
    started=time.perf_counter()
    require(not _WORKER_ENTERED,'A physical worker may enter once'); _WORKER_ENTERED=True
    spec=read(cell/'spec.json')
    require(not (cell/'worker_started.json').exists(),'Repeated worker forbidden')
    save(cell/'worker_started.json',{'pid':os.getpid(),'parent_pid':os.getppid(),'utc':utc(),'started_monotonic':started})
    try:
        verify_execution(run)
        require(spec in read(run/'schedule.json') and cell==run/spec['cell'],'Unscheduled worker')
        ex=read(cell/'execution.json'); deadline=ex['deadline_epoch_seconds']
        require(deadline<=read(run/'execution_started.json')['deadline_epoch_seconds'] and time.time()<deadline,'Invalid task deadline')
        require(shutil.disk_usage(ROOT).free>=MIN_FREE,'Storage floor reached')
        from run_nature34_clpu_pilot import make_env
        from dnr.nature34_clpu_branch import BranchConfig
        from dnr.nature34_clpu_early import merge_results
        from dnr.nature34_clpu_shared_evidence import SharedEvidenceService
        from nature34_clpu_tail_execution import load_fixed_policy
        model_path=run/spec['checkpoint'] if spec['checkpoint'] else None
        if model_path: require(sha(model_path)==spec['checkpoint_sha256'],'Fixed model changed')
        model_started=time.perf_counter()
        policy,generator=load_fixed_policy(model_path,spec)
        model_load_seconds=time.perf_counter()-model_started if model_path else 0.
        env=make_env(run,spec); cfg=BranchConfig(**read(run/'config.json')['branch_search'])
        compute_started=time.perf_counter()
        with SharedEvidenceService(run,cell,env,read(run/'physical_scope.json'),mode=spec['mode'],physical_budget=512,logical_budget=256) as service:
            service.deadline_monotonic=time.perf_counter()+max(0,deadline-time.time())
            init_seconds=time.perf_counter()-compute_started
            expected=next(r['generation'] for r in read(run/'stage_plan.json')['contexts'] if r['context_id']==spec['context_id'])
            phases,cps=run_routes(service,env,cfg,cell,spec,started,deadline,policy,generator,expected_generation=expected)
        if model_path: require(sha(model_path)==spec['checkpoint_sha256'],'Fixed model changed during task')
        merged=merge_results(phases['EARLY'],phases['GUARD'],protected=True)
        guard=phases['GUARD']; selected=phases[merged['selected_phase']]
        finished_tick=time.perf_counter()
        result={**merged,'version':STUDY,'schema':'clpu-staged-protected-result-v1','spec':spec,'study':STUDY,'quality_pct':selected['quality_pct'],
            'fully_certified':selected['fully_certified'],'early_quality_pct':phases['EARLY']['quality_pct'],
            'baseline_quality_pct':guard['quality_pct'],'logical_points':sum(p['logical_points'] for p in phases.values()),
            'actual_physical_points':service.physical_points,
            'numerical_entries':service.factory.stats()['returned']['numerical_solution_entries'],
            'requests':sum(p['oracle']['requests'] for p in phases.values()),
            'search_decisions':sum(p['search_decisions'] for p in phases.values()),
            'policy_decisions':phases['EARLY']['policy_decisions'],'model_load_seconds':model_load_seconds,
            'checkpoint_sha256':spec['checkpoint_sha256'],
            'stage_generation_sha256':hashlib.sha256(canonical(phases['EARLY']['stage_generation']).encode()).hexdigest(),
            'stage_generation_seconds':phases['EARLY']['stage_generation_seconds'],
            'stage_candidate_count':len((phases['EARLY']['stage_generation'] or {}).get('candidates',[])),
            'service':service.summary(),'phase_summary':{k:{name:v[name] for name in ('quality_pct','fully_certified','logical_points','actual_physical_points','search_decisions','policy_decisions','graph_expansions','verify_attempts','phase_elapsed_seconds','stage_component_seconds')} for k,v in phases.items()},
            'checkpoints':cps,'worker_elapsed_seconds':finished_tick-started,
            'preflight_elapsed_seconds':compute_started-started,'runtime_initialization_seconds':init_seconds,
            'computation_elapsed_seconds':finished_tick-compute_started,'new_training':0}
        save(cell/'result.json',result)
        save(cell/'worker_finished.json',{'status':'COMPLETE','pid':os.getpid(),'utc':utc(),
            'result_sha256':sha(cell/'result.json'),'worker_elapsed_seconds':time.perf_counter()-started})
        seal(cell); return 0
    except BaseException as exc:
        save(cell/'error.json',{'status':'STOPPED_RELIABILITY_OR_IMPLEMENTATION_ERROR','error':repr(exc),'traceback':traceback.format_exc(),
            'partial_evidence_retained':True,'no_automatic_retry':True,'pid':os.getpid()})
        return 1


def account(run,dispatched):
    totals={'logical_points':0,'physical_points':0,'numerical_entries':0,'requests':0}; unknown=[]
    for spec in dispatched:
        cell=run/spec['cell']; p=cell/'result.json'
        if p.exists():
            r=read(p); totals['logical_points']+=r['logical_points']; totals['physical_points']+=r['actual_physical_points']; totals['numerical_entries']+=r['numerical_entries'];totals['requests']+=r['requests']
        else:
            unknown.append(spec['cell'])
            for phase in ('EARLY','GUARD'):
                view=cell/phase/'view_summary.json'
                if view.exists():
                    r=read(view);totals['logical_points']+=r['logical_points'];totals['requests']+=r['requests']
            p=cell/'physical/oracle_summary.json'
            if p.exists():
                r=read(p); totals['physical_points']+=r['points']; totals['numerical_entries']+=r['actual_numerical_entries']
            else:
                p=cell/'physical/command_stats.json'
                if p.exists(): totals['numerical_entries']+=read(p)['returned']['numerical_solution_entries']
    return {**totals,'unknown_cost_cells':unknown}


def phase1_gate(run):
    """Check sealed independent stage-one evidence, never infer gain from graph."""
    cache.verify_manifest(run/'audit_stage1',required=('summary.json',))
    audit=read(run/'audit_stage1/summary.json')
    plan=read(run/'stage_plan.json'); controller=read(run/'controller_stage1_completion.json')
    require(audit['status']=='PASS' and audit['stage']=='phase1' and not audit.get('errors',[])
            and audit['audited_cells']==len(plan['phase1_cells']), 'All first-stage cells require independent PASS')
    for key,name in (('freeze_sha256','prephysical_freeze_manifest.json'),
                     ('stage_plan_sha256','stage_plan.json'),
                     ('controller_stage1_completion_sha256','controller_stage1_completion.json')):
        require(audit[key]==sha(run/name),'Stage-one audit binding changed: '+key)
    require(controller['status']=='COMPLETE_STAGE1_PENDING_AUDIT'
            and set(controller['completed'])==set(plan['phase1_cells'])
            and len(controller['exits'])==len(plan['phase1_cells'])
            and all(row['exit_code']==0 for row in controller['exits'])
            and not controller['errors'] and not controller['unknown_cost_cells'], 'Stage one did not truly close')
    checks={row['cell']:row for row in audit['results']}
    require(len(checks)==len(audit['results']) and set(checks)==set(plan['phase1_cells']), 'Stage-one audit population changed')
    positives=[]
    for name in plan['phase1_cells']:
        check=checks[name]; cell=run/name
        require(check['status']=='PASS' and check['cell_completion_sha256']==sha(cell/'completion_manifest.json')
                and check['result_sha256']==sha(cell/'result.json'), 'Audited first-stage evidence changed')
        manifest=cache.verify_manifest(cell,required=('result.json','EARLY/phase_result.json','GUARD/phase_result.json'))
        phases={p:read(cell/p/'phase_result.json') for p in ('EARLY','GUARD')}
        def certified_energy(value):
            best=value.get('best_plan')
            return (best['validation']['energy_kwh'] if value['fully_certified'] and best
                and best.get('certified_complete') and best.get('validation',{}).get('status')=='COMPLETE' else None)
        early,guard=(certified_energy(phases[p]) for p in ('EARLY','GUARD'))
        require(check['early_energy_kwh']==early and check['guard_energy_kwh']==guard
                and check['early_fully_certified']==(early is not None)
                and check['guard_established'] is True and guard is not None, 'Certified gate energies differ')
        if early is not None and early>guard:
            positives.append(name)
    require(audit['gate']['phase2_authorized']==bool(positives)
            and audit['gate']['positive_cells']==positives, 'Independent strict-gain gate differs')
    return {'phase2_authorized':bool(positives),'positive_cells':positives,
        'criterion':'any fully certified EARLY energy strictly above own complete GUARD',
        'audit_stage1_manifest_sha256':sha(run/'audit_stage1/completion_manifest.json'),
        'audit_stage1_summary_sha256':sha(run/'audit_stage1/summary.json'),
        'freeze_sha256':sha(run/'prephysical_freeze_manifest.json'),
        'stage_plan_sha256':sha(run/'stage_plan.json'),
        'controller_stage1_completion_sha256':sha(run/'controller_stage1_completion.json'),
        'phase1_cells':plan['phase1_cells'],'phase2_cells':plan['phase2_cells']}


def _execute_selected(run,stage,selected,previous):
    require(stage in ('phase1','phase2'),'Unknown execution stage')
    prefix='controller_stage1' if stage=='phase1' else 'controller_stage2'
    require(not (run/(prefix+'_started.json')).exists(),'No automatic stage rerun')
    started=read(run/'execution_started.json'); end=started['deadline_epoch_seconds']
    began=time.perf_counter()
    save(run/(prefix+'_started.json'),{'utc':utc(),'stage':stage,'pid':os.getpid(),
        'selected_cells':[s['cell'] for s in selected]})
    dispatched=[];completed=[];exits=[];errors=[];active={};peak=0
    environment=dict(os.environ)
    environment.update({n:'1' for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS')})
    environment['PYTHONDONTWRITEBYTECODE']='1'
    def finish(key,item,code,drained=False):
        spec,proc,out,err,tick,epoch=item
        out.close();err.close();del active[key]
        exits.append({'cell':spec['cell'],'pair_order':spec['pair_order'],'launcher_pid':proc.pid,'exit_code':code,
            'started_epoch_seconds':epoch,'finished_epoch_seconds':time.time(),'process_wall_seconds':time.perf_counter()-tick,
            'drained_after_controller_error':drained,'utc':utc()})
        if code==0:
            try:
                cache.verify_manifest(run/spec['cell'],required=('result.json','worker_finished.json'))
                completed.append(spec['cell'])
            except Exception as exc:
                errors.append({'cell':spec['cell'],'exit_code':code,'seal_error':repr(exc)})
        else:errors.append({'cell':spec['cell'],'exit_code':code})
        save(run/(prefix+'_exits.json'),exits)
    def progress():
        value={'status':'DRAINING_ERROR' if errors else 'RUNNING','stage':stage,'run':run.name,
            'scheduled_total':CELLS,'scheduled_stage':len(selected),'completed_stage':len(completed),
            'completed_prior':len(previous),'active':list(active),'errors':errors,
            'elapsed_seconds':time.perf_counter()-began,'utc':utc()}
        save(run/(prefix+'_progress.json'),value);save(ACTIVE,value)
    try:
        with (run/(prefix+'_dispatch.jsonl')).open('x',encoding='utf-8') as local_journal, \
                (run/'controller_dispatch.jsonl').open('x' if stage=='phase1' else 'a',encoding='utf-8') as journal:
            for offset in range(0,len(selected),2):
                if errors:break
                require(not active,'Drain previous worker pair first')
                require(time.time()<end,'Global controller deadline reached')
                rate=read(run/'resource_budget.json')['historical_max_bytes_per_actual_point']
                for name in previous+completed:
                    directory=run/name;value=read(directory/'result.json')
                    rate=max(rate,sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())/max(1,value['actual_physical_points']))
                remaining=CELLS-len(previous)-len(completed)
                forecast={'free_bytes':shutil.disk_usage(ROOT).free,'remaining_cells':remaining,
                    'required_free_bytes':MIN_FREE+RESERVE+math.ceil(remaining*512*rate*1.25)}
                save(run/'storage_forecast.json',forecast)
                require(forecast['free_bytes']>=forecast['required_free_bytes'],'Remaining storage forecast unmet')
                require(not account(run,dispatched)['unknown_cost_cells'],'Unknown earlier task costs')
                for spec in selected[offset:offset+2]:
                    cell=run/spec['cell'];now=time.time()
                    require(time.time()<end and not (cell/'worker_started.json').exists(),'Expired or repeated cell')
                    save(cell/'execution.json',{'deadline_epoch_seconds':min(end,now+WORKER_SECONDS),'started_epoch_seconds':now,
                        'controller_pid':os.getpid(),'spec_sha256':sha(cell/'spec.json'),'stage':stage})
                    out=(cell/'stdout.log').open('xb');err=(cell/'stderr.log').open('xb');tick=time.perf_counter()
                    try:
                        proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'worker','--run',str(run),'--cell',str(cell)],
                            cwd=ROOT,env=environment,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    except BaseException:
                        out.close();err.close();raise
                    active[spec['cell']]=(spec,proc,out,err,tick,now);dispatched.append(spec);peak=max(peak,len(active))
                    record={'cell':spec['cell'],'pair_order':spec['pair_order'],'launcher_pid':proc.pid,
                        'started_epoch_seconds':now,'stage':stage,'utc':utc()}
                    append(local_journal,record);append(journal,record)
                while active:
                    for key,item in list(active.items()):
                        code=item[1].poll()
                        if code is not None:finish(key,item,code,bool(errors))
                    if (time.time()>=end or shutil.disk_usage(ROOT).free<MIN_FREE) and not errors:
                        errors.append({'stage':'controller','error':'Deadline or storage floor; drain started workers'})
                    progress()
                    if active:time.sleep(.2)
    except BaseException as exc:
        errors.append({'stage':'controller','error':repr(exc),'traceback':traceback.format_exc()})
    finally:
        while active:
            for key,item in list(active.items()):
                code=item[1].poll()
                if code is not None:
                    try:finish(key,item,code,True)
                    except OSError as exc:errors.append({'stage':'drain_exit_record','error':repr(exc)})
            try:progress()
            except OSError:pass
            if active:time.sleep(.2)
    costs=account(run,dispatched)
    if costs['physical_points']>len(selected)*512 or peak>2:
        errors.append({'stage':'accounting','error':'Actual point or concurrency cap exceeded'})
    complete=len(completed)==len(selected) and not errors and not costs['unknown_cost_cells']
    status=('COMPLETE_STAGE1_PENDING_AUDIT' if stage=='phase1' else 'COMPLETE_STAGE2_PENDING_AUDIT') if complete else 'STOPPED_INCOMPLETE'
    result={'status':status,'stage':stage,'scheduled':len(selected),'scheduled_total':CELLS,
        'selected_cells':[s['cell'] for s in selected],'completed':completed,'dispatched':dispatched,
        'exits':exits,'errors':errors,'peak_active':peak,**costs,'wall_seconds':time.perf_counter()-began,'utc':utc()}
    save(run/(prefix+'_completion.json'),result)
    save(ACTIVE,{**result,'run':run.name})
    return result


def execute_stage1(run):
    verify_execution(run)
    require(not (run/'execution_started.json').exists(),'Run already started')
    tick=time.perf_counter();epoch=time.time()
    save(run/'execution_started.json',{'utc':utc(),'pid':os.getpid(),'deadline_epoch_seconds':epoch+MAX_SECONDS,
        'started_epoch_seconds':epoch,'started_monotonic_seconds':tick})
    plan=read(run/'stage_plan.json')
    selected=[s for s in read(run/'schedule.json') if s['cell'] in plan['phase1_cells']]
    result=_execute_selected(run,'phase1',selected,[])
    return 0 if result['status']=='COMPLETE_STAGE1_PENDING_AUDIT' else 1


def execute_stage2(run):
    verify_execution(run)
    require(not (run/'controller_completion.json').exists() and not (run/'phase_gate.json').exists(),'No repeated second-stage decision')
    gate=phase1_gate(run);save(run/'phase_gate.json',gate)
    plan=read(run/'stage_plan.json');schedule=read(run/'schedule.json')
    first=read(run/'controller_stage1_completion.json')
    skipped=[]
    if gate['phase2_authorized']:
        second=_execute_selected(run,'phase2',[s for s in schedule if s['cell'] in plan['phase2_cells']],first['completed'])
        stage_list=[first,second]
        complete=second['status']=='COMPLETE_STAGE2_PENDING_AUDIT'
        status='COMPLETE_CELLS_PENDING_AUDIT' if complete else 'STOPPED_INCOMPLETE'
    else:
        stage_list=[first]
        for spec in schedule:
            if spec['cell'] in plan['phase2_cells']:
                require(not any((run/spec['cell']/name).exists() for name in ('worker_started.json','execution.json','result.json')),
                        'Unexecuted gate slot already dispatched')
                row={'cell':spec['cell'],'context_id':spec['context_id'],'arm':spec['arm'],'seed':spec['seed'],
                    'status':'NOT_RUN_GATE_NO_GAIN','executed':False,'quality_pct':None,'process_wall_seconds':None,
                    'new_physical_points':0,'new_numerical_entries':0,'new_runtime_initializations':0}
                save(run/spec['cell']/'not_run.json',row);skipped.append(row)
        status='COMPLETE_PARTIAL_EXPLORATORY_NO_GAIN'
    dispatched=[s for stage in stage_list for s in stage['dispatched']]
    completed=[s for stage in stage_list for s in stage['completed']]
    exits=[s for stage in stage_list for s in stage['exits']]
    errors=[s for stage in stage_list for s in stage['errors']]
    costs=account(run,dispatched)
    if costs['physical_points']>TOTAL_CAP or costs['unknown_cost_cells']:
        status='STOPPED_INCOMPLETE';errors.append({'stage':'final_accounting','error':'Total cap or unknown costs'})
    result={'status':status,'scheduled':CELLS,'completed':completed,'dispatched':dispatched,'exits':exits,'errors':errors,
        'not_run':skipped,'actual_tasks':len(dispatched),'resolved_slots':len(completed)+len(skipped),
        'stage1_controller_sha256':sha(run/'controller_stage1_completion.json'),
        'stage2_controller_sha256':sha(run/'controller_stage2_completion.json') if len(stage_list)==2 else None,
        'phase_gate_sha256':sha(run/'phase_gate.json'),'stage1_audit_seconds':read(run/'audit_stage1/summary.json')['elapsed_seconds'],
        'all_eight_context_mean_allowed':len(completed)==CELLS and not errors,
        **costs,'wall_seconds':time.perf_counter()-read(run/'execution_started.json')['started_monotonic_seconds'],'utc':utc()}
    save(run/'controller_completion.json',result);save(ACTIVE,{**result,'run':run.name})
    return 0 if status!='STOPPED_INCOMPLETE' else 1


def finalize(run):
    require(not (run/'completion_manifest.json').exists(),'Already sealed')
    verify_execution(run)
    cache.verify_manifest(run/'audit',required=('summary.json',))
    cache.verify_manifest(run/'derived_report',required=('summary.json','all_cells.json','all_pairs.json'))
    audit=read(run/'audit/summary.json'); report=read(run/'derived_report/summary.json')
    controller=read(run/'controller_completion.json');actual=len(controller['completed'])
    require(controller['status'] in ('COMPLETE_CELLS_PENDING_AUDIT','COMPLETE_PARTIAL_EXPLORATORY_NO_GAIN')
            and controller['resolved_slots']==CELLS and not controller['errors'],'Complete conditional disposition required')
    require(audit['status']=='PASS' and audit['audited_cells']==actual and audit['resolved_slots']==CELLS
            and not audit['errors'] and audit['controller_completion_sha256']==sha(run/'controller_completion.json'), 'All actual audits and skipped slots required')
    require(report['status']=='AUDITED_DESCRIPTIVE_STAGED_COMMON_PILOT' and report['scheduled_cells']==CELLS
            and report['executed_cells']==actual,'Complete conditional report required')
    require(report['source_bindings']['audit/summary.json']==sha(run/'audit/summary.json'),'Report audit binding changed')
    require(report['source_bindings']['controller_completion.json']==sha(run/'controller_completion.json'),'Report controller binding changed')
    terminal={'status':('COMPLETE_STAGED_PILOT_AUDITED_AND_REPORTED' if actual==CELLS else 'COMPLETE_STAGED_PARTIAL_EXPLORATORY_AUDITED_AND_REPORTED'),'run':run.name,'utc':utc(),
        'audit_sha256':sha(run/'audit/summary.json'),'report_sha256':sha(run/'derived_report/summary.json'),
        'scheduled_cells':CELLS,'executed_cells':actual,'not_run_cells':CELLS-actual,
        'all_eight_context_mean_allowed':actual==CELLS,
        'physical_points':audit['physical_points'],'numerical_entries':audit['numerical_entries'],'new_training':0}
    save(run/'terminal_status.json',terminal); seal(run)
    save(ACTIVE,{**terminal,'status':('COMPLETE_STAGED_PILOT_SEALED' if actual==CELLS else 'COMPLETE_STAGED_PARTIAL_EXPLORATORY_SEALED'),'completion_sha256':sha(run/'completion_manifest.json')})
    return terminal


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('command',choices=('prepare','worker','execute_stage1','execute_stage2','finalize'))
    p.add_argument('--run',type=Path); p.add_argument('--cell',type=Path); p.add_argument('--tests',type=Path)
    a=p.parse_args()
    if a.command=='prepare': print(prepare(a.tests.resolve()))
    elif a.command=='worker': sys.exit(worker(a.run.resolve(),a.cell.resolve()))
    elif a.command=='execute_stage1': sys.exit(execute_stage1(a.run.resolve()))
    elif a.command=='execute_stage2': sys.exit(execute_stage2(a.run.resolve()))
    else: print(finalize(a.run.resolve()))


