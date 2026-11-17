"""Cold deferred-verification comparison with a complete original Beam guard."""
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

SOURCE = ROOT / 'runs/20261031T045725_491175Z_clpu_shared_rule_pilot'
SOURCE_SHA = '779f0f56987749761884539703a759a4ee3723c2a953d60018a2880b33bfeede'
MODELS_SOURCE = ROOT / 'runs/20261031T020246_028867Z_clpu_early_pilot'
MODELS_SHA = 'bed648a067dd82c3c52e0250de25714efcea7cca5b69347fad399e72e9e57747'
STUDY = 'CLPU_DEFERRED_VERIFICATION_PILOT_V1'
PROTOCOL = 'docs/clpu_deferred_pilot_protocol_v1.md'
ACTIVE = ROOT / 'artifacts/clpu_deferred_pilot_active.json'
SEEDS = (17, 41, 73)
SLOTS = ('interleaved_window','deferred_window', *(f'deferred_bc{s}' for s in SEEDS), *(f'deferred_ppo{s}' for s in SEEDS))
CELLS = 64
MIN_FREE, RESERVE = 12 * 2**30, 2 * 2**30
MAX_SECONDS, WORKER_SECONDS, TOTAL_CAP = 7200, 900, CELLS * 512
_WORKER_ENTERED = False


def settings():
    return {'cells':CELLS, 'contexts':8, 'mode':'shared', 'phases':['EARLY','GUARD'],
        'seeds':list(SEEDS), 'slots':list(SLOTS), 'logical_budget_per_view':256,
        'early_expansion_slots':16, 'early_verification_slots':16, 'maximum_early_operations':33,
        'bootstrap':'original paid initial snapshot and complete WAIT', 'guard_expansion_cap':256,
        'total_physical_budget_per_cell':512, 'total_physical_budget':TOTAL_CAP,
        'maximum_numerical_entries':CELLS*3+4*TOTAL_CAP, 'physical_runtimes_per_cell':1,
        'max_workers':2, 'worker_seconds':WORKER_SECONDS, 'controller_seconds':MAX_SECONDS,
        'minimum_free_disk_bytes':MIN_FREE, 'storage_margin':1.25, 'report_reserve_bytes':RESERVE,
        'phase_order':['EARLY','GUARD'], 'tie_rule':'exact_equal_energy_select_GUARD',
        'same_runtime_both_modes':True, 'new_training':0, 'no_cross_cell_cache':True,
        'no_historical_numerical_answers':True, 'no_automatic_retry':True,
        'no_upper_bound_change':True, 'model_role':'fixed candidate ranking only, not learned EXPAND/VERIFY/YIELD',
        'macro_control':'16 EXPAND then up to16 VERIFY; interleaved rule alternates16 EXPAND/VERIFY; at most1 YIELD',
        'same_optimized_window_for_all_rankers':True, 'no_chosen_plan_targets':True,
        'timing':'cold complete cell process; all initialization, both searches, logging and exit paid',
        'recording':'STARTED proposal with raw observation and stable mapping; COMPLETED exact response; separate logical and real costs',
        'dispatch':'hashed contexts and slots, Latin slot rotation; adjacent pair drained before next pair',
        'aggregation':'mean three seeds within context then equal mean eight contexts; retain all64 tasks'}


def schedule_for(contexts, models):
    require(len(contexts)==8 and {c['context_id'] for c in contexts}==set(range(8)), 'All eight contexts required')
    require(len({c['scenario_id'] for c in contexts})==8, 'Fixed faults changed')
    lookup={(m['method'],m['seed']):m for m in models}
    require(len(models)==len(lookup)==6 and set(lookup)=={(m,s) for m in ('bc_raw','ppo_raw') for s in SEEDS},'Six fixed models required')
    ordered=sorted(contexts,key=lambda c:hashlib.sha256(('deferred-context-v1|'+canonical(c)).encode()).hexdigest())
    slots=sorted(SLOTS,key=lambda s:hashlib.sha256(('deferred-slots-v1|'+s).encode()).hexdigest())
    rows=[]
    for rank,c in enumerate(ordered):
        for position in range(8):
            slot=slots[(rank+position)%8]
            seed=next((s for s in SEEDS if slot.endswith(str(s))),0)
            arm=slot[:-len(str(seed))] if seed else slot
            method='bc_raw' if arm=='deferred_bc' else 'ppo_raw' if arm=='deferred_ppo' else 'beam'
            model=lookup.get((method,seed))
            rows.append({**{k:c[k] for k in ('context_id','scenario_id','event_id')},
                'cell':f'episodes/{len(rows):03d}', 'episode_id':len(rows), 'pair_order':rank*4+position//2,
                'context_order':rank,'arm_position':position,'arm_slot':slot,
                'mode':'shared','arm':arm,'method':method,'seed':seed,
                'macro_schedule':'interleaved' if arm=='interleaved_window' else 'deferred',
                'phase':'evaluation', 'group':arm, 'protected':True,
                'representation':'raw', 'deterministic':True,
                'checkpoint':model['checkpoint'] if model else None,
                'checkpoint_sha256':model['source_sha256'] if model else None,
                'point_budget':256, 'logical_point_budget':256, 'physical_point_budget':512,
                'total_logical_point_budget':512, 'window_variant':'pending_unique_coverage_v1'})
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
    names += [PROTOCOL,'src/dnr/nature34_clpu_deferred.py','src/dnr/nature34_clpu_deferred_control.py',
        'scripts/run_clpu_deferred_pilot.py','scripts/audit_clpu_deferred_pilot.py','scripts/report_clpu_deferred_pilot.py',
        'scripts/plot_clpu_deferred_pilot.py','tests/test_nature34_clpu_deferred.py',
        'tests/test_nature34_clpu_deferred_control.py','tests/test_run_clpu_deferred_pilot.py',
        'tests/test_clpu_deferred_audit.py','tests/test_report_clpu_deferred_pilot.py']
    return sorted(set(names))


def validate_tests(path):
    cases=list(ET.parse(path).getroot().iter('testcase'))
    require(cases and not any(list(c.iter('failure'))+list(c.iter('error'))+list(c.iter('skipped')) for c in cases), 'Tests incomplete')
    for part in ('test_nature34_clpu_deferred_control','test_run_clpu_deferred_pilot','test_clpu_deferred_audit','test_report_clpu_deferred_pilot'):
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


def prepare(tests_path):
    from dnr.nature34_clpu_branch import BranchConfig
    count=validate_tests(tests_path)
    require(sha(SOURCE/'completion_manifest.json')==SOURCE_SHA,'Inherited source seal changed')
    sm=normalize_manifest(read(SOURCE/'completion_manifest.json'))
    for n in ('config.json','schedule.json','training_contexts.json','selected_scenarios.json','selected_events.json','prepared.json'):
        require(sha(SOURCE/n)==sm[n],'Inherited input changed: '+n)
    chosen=[s for s in read(SOURCE/'schedule.json') if s['mode']=='shared']
    contexts=read(SOURCE/'contexts.json')
    require(sha(SOURCE/'contexts.json')==sm['contexts.json'],'Source contexts changed')
    models=read(MODELS_SOURCE/'model_bindings.json')
    schedule=schedule_for(contexts,models)
    cfg=deepcopy(read(SOURCE/'config.json'))
    require(cfg['branch_search']==asdict(BranchConfig()),'Original search changed')
    cfg.update(deferred_study=STUDY,deferred_pilot=settings())
    rates=[]
    for s in chosen:
        p=SOURCE/s['cell']; result=read(p/'result.json')
        rates.append(sum(f.stat().st_size for f in p.rglob('*') if f.is_file())/max(1,result['actual_physical_points']))
    budget={'historical_max_bytes_per_actual_point':max(rates),'total_physical_cap':TOTAL_CAP,
        'forecast_bytes':math.ceil(max(rates)*TOTAL_CAP*1.25),'minimum_free_bytes':MIN_FREE,'report_reserve_bytes':RESERVE,
        'free_bytes_at_prepare':shutil.disk_usage(ROOT).free,'successful_component_tests':count}
    budget['required_free_bytes']=MIN_FREE+RESERVE+budget['forecast_bytes']
    require(budget['free_bytes_at_prepare']>=budget['required_free_bytes'],'Storage forecast unmet')
    run=create_run('clpu_deferred_pilot',cfg)
    try:
        for n in (*ROOT_INPUTS,'population.json','selected_scenarios.json','selected_events.json','training_contexts.json'):
            bound_copy(SOURCE,run/n,sm,n)
        for n in sm:
            if n.startswith('source_inputs/'):
                bound_copy(SOURCE,run/n,sm,n)
        save(run/'contexts.json',contexts)
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
            'scope':'User explicitly approved real deferred verification comparison with equally optimized window and BC, retaining complete Beam; all8 contexts and3 fixed seeds, no training.',
            'learning':'Frozen prior BC/PPO rank current common pending candidates; macro schedule is fixed, not learned.'})
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
    cache.verify_manifest(run,'prephysical_freeze_manifest.json',('schedule.json','physical_scope.json','contexts.json'))
    for n in p['source_files']: require(sha(ROOT/n)==sha(run/'source'/n),'Executing source changed: '+n)
    require(dependencies()==read(run/'dependencies.json'),'Dependencies changed')
    require(sha(Path(sys.executable))==read(run/'python_environment.json')['sha256'],'Interpreter changed')
    cfg=read(run/'config.json')
    require(cfg['deferred_study']==STUDY and cfg['deferred_pilot']==settings(),'Deferred contract changed')
    require(read(run/'source_binding.json')==source_binding(),'Source binding changed')
    require(read(run/'schedule.json')==schedule_for(read(run/'contexts.json'),read(run/'model_bindings.json')),'Schedule changed')
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


def run_routes(service,env,branch_config,cell,spec,started,deadline,policy=None,generator=None,search_classes=None):
    from dnr.nature34_clpu_branch import BranchSearch
    from dnr.nature34_clpu_deferred import DeferredCertificationSearch
    from dnr.nature34_clpu_deferred_control import DeferredMacroController
    classes=search_classes or {'EARLY':DeferredCertificationSearch,'GUARD':BranchSearch}
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
                control=DeferredMacroController(search,schedule=spec['macro_schedule'],method=spec['method'],
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
                    require(count<=(33 if control else 256),'Phase operation limit exceeded')
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
            phases,cps=run_routes(service,env,cfg,cell,spec,started,deadline,policy,generator)
        if model_path: require(sha(model_path)==spec['checkpoint_sha256'],'Fixed model changed during task')
        merged=merge_results(phases['EARLY'],phases['GUARD'],protected=True)
        guard=phases['GUARD']; selected=phases[merged['selected_phase']]
        finished_tick=time.perf_counter()
        result={**merged,'version':STUDY,'schema':'clpu-deferred-protected-result-v1','spec':spec,'study':STUDY,'quality_pct':selected['quality_pct'],
            'fully_certified':selected['fully_certified'],'early_quality_pct':phases['EARLY']['quality_pct'],
            'baseline_quality_pct':guard['quality_pct'],'logical_points':sum(p['logical_points'] for p in phases.values()),
            'actual_physical_points':service.physical_points,
            'numerical_entries':service.factory.stats()['returned']['numerical_solution_entries'],
            'requests':sum(p['oracle']['requests'] for p in phases.values()),
            'search_decisions':sum(p['search_decisions'] for p in phases.values()),
            'policy_decisions':phases['EARLY']['policy_decisions'],'model_load_seconds':model_load_seconds,
            'checkpoint_sha256':spec['checkpoint_sha256'],
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
    totals={'logical_points':0,'physical_points':0,'numerical_entries':0}; unknown=[]
    for spec in dispatched:
        cell=run/spec['cell']; p=cell/'result.json'
        if p.exists():
            r=read(p); totals['logical_points']+=r['logical_points']; totals['physical_points']+=r['actual_physical_points']; totals['numerical_entries']+=r['numerical_entries']
        else:
            unknown.append(spec['cell'])
            for phase in ('EARLY','GUARD'):
                view=cell/phase/'view_summary.json'
                if view.exists(): totals['logical_points']+=read(view)['logical_points']
            p=cell/'physical/oracle_summary.json'
            if p.exists():
                r=read(p); totals['physical_points']+=r['points']; totals['numerical_entries']+=r['actual_numerical_entries']
            else:
                p=cell/'physical/command_stats.json'
                if p.exists(): totals['numerical_entries']+=read(p)['returned']['numerical_solution_entries']
    return {**totals,'unknown_cost_cells':unknown}


def execute(run):
    verify_execution(run); require(not (run/'execution_started.json').exists(),'Run already started')
    schedule=read(run/'schedule.json'); began=time.perf_counter(); end=time.time()+MAX_SECONDS
    save(run/'execution_started.json',{'utc':utc(),'pid':os.getpid(),'deadline_epoch_seconds':end,'started_epoch_seconds':time.time()})
    dispatched=[]; completed=[]; exits=[]; errors=[]; active={}
    environment=dict(os.environ)
    environment.update({n:'1' for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS')})
    environment['PYTHONDONTWRITEBYTECODE']='1'
    def finish(key,item,code,drained=False):
        spec,proc,out,err,tick,epoch=item
        out.close(); err.close(); del active[key]
        exits.append({'cell':spec['cell'],'pair_order':spec['pair_order'],'launcher_pid':proc.pid,'exit_code':code,
            'started_epoch_seconds':epoch,'finished_epoch_seconds':time.time(),'process_wall_seconds':time.perf_counter()-tick,
            'drained_after_controller_error':drained,'utc':utc()})
        if code==0 and (run/spec['cell']/'completion_manifest.json').exists(): completed.append(spec['cell'])
        else: errors.append({'cell':spec['cell'],'exit_code':code,'missing_seal':not (run/spec['cell']/'completion_manifest.json').exists()})
        save(run/'controller_exits.json',exits)

    def progress():
        value={'status':'DRAINING_ERROR' if errors else 'RUNNING','run':run.name,'scheduled':CELLS,'completed':len(completed),
            'dispatched':len(dispatched),'active':list(active),'errors':errors,'elapsed_seconds':time.perf_counter()-began,'utc':utc()}
        save(run/'controller_progress.json',value); save(ACTIVE,value)

    try:
        with (run/'controller_dispatch.jsonl').open('x',encoding='utf-8') as journal:
            for start in range(0,len(schedule),2):
                if errors: break
                require(time.time()<end,'Controller deadline reached')
                rate=read(run/'resource_budget.json')['historical_max_bytes_per_actual_point']
                for name in completed:
                    directory=run/name; value=read(directory/'result.json')
                    rate=max(rate,sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())/max(1,value['actual_physical_points']))
                forecast={'free_bytes':shutil.disk_usage(ROOT).free,'remaining_cells':CELLS-len(completed),
                    'required_free_bytes':MIN_FREE+RESERVE+math.ceil((CELLS-len(completed))*512*rate*1.25)}
                save(run/'storage_forecast.json',forecast)
                require(forecast['free_bytes']>=forecast['required_free_bytes'],'Remaining storage forecast unmet')
                require(not account(run,dispatched)['unknown_cost_cells'],'Unknown prior task costs')
                for spec in schedule[start:start+2]:
                    cell=run/spec['cell']; now=time.time()
                    save(cell/'execution.json',{'deadline_epoch_seconds':min(end,now+WORKER_SECONDS),'started_epoch_seconds':now,
                        'controller_pid':os.getpid(),'spec_sha256':sha(cell/'spec.json')})
                    out=(cell/'stdout.log').open('xb'); err=(cell/'stderr.log').open('xb'); tick=time.perf_counter()
                    try:
                        proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'worker','--run',str(run),'--cell',str(cell)],
                            cwd=ROOT,env=environment,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    except BaseException:
                        out.close(); err.close(); raise
                    active[spec['cell']]=(spec,proc,out,err,tick,now); dispatched.append(spec)
                    append(journal,{'cell':spec['cell'],'pair_order':spec['pair_order'],'launcher_pid':proc.pid,'started_epoch_seconds':now,'utc':utc()})
                while active:
                    for key,item in list(active.items()):
                        code=item[1].poll()
                        if code is not None: finish(key,item,code)
                    if (time.time()>=end or shutil.disk_usage(ROOT).free<MIN_FREE) and not errors:
                        errors.append({'stage':'controller','error':'Deadline or storage floor; drain started workers'})
                    progress()
                    if active: time.sleep(.2)
    except BaseException as exc:
        errors.append({'stage':'controller','error':repr(exc),'traceback':traceback.format_exc()})
    finally:
        while active:
            for key,item in list(active.items()):
                code=item[1].poll()
                if code is not None:
                    try: finish(key,item,code,True)
                    except OSError as exc: errors.append({'stage':'drain_exit_record','error':repr(exc)})
            try: progress()
            except OSError: pass
            if active: time.sleep(.2)
    costs=account(run,dispatched)
    if costs['physical_points']>TOTAL_CAP: errors.append({'stage':'accounting','error':'Real point cap exceeded'})
    status='COMPLETE_CELLS_PENDING_AUDIT' if len(completed)==CELLS and not errors and not costs['unknown_cost_cells'] else 'STOPPED_INCOMPLETE'
    result={'status':status,'scheduled':CELLS,'completed':completed,'dispatched':dispatched,'exits':exits,'errors':errors,
        **costs,'wall_seconds':time.perf_counter()-began,'utc':utc()}
    save(run/'controller_completion.json',result)
    save(ACTIVE,{**{k:result[k] for k in ('status','errors','physical_points','logical_points','numerical_entries','unknown_cost_cells')},
        'run':run.name,'scheduled':CELLS,'completed':len(completed),'utc':utc()})
    return 0 if status=='COMPLETE_CELLS_PENDING_AUDIT' else 1


def finalize(run):
    require(not (run/'completion_manifest.json').exists(),'Already sealed')
    verify_execution(run)
    cache.verify_manifest(run/'audit',required=('summary.json',))
    cache.verify_manifest(run/'derived_report',required=('summary.json','all_cells.json','all_pairs.json'))
    audit=read(run/'audit/summary.json'); report=read(run/'derived_report/summary.json')
    require(audit['status']=='PASS' and audit['audited_cells']==CELLS and not audit['errors'],'All audits required')
    require(report['status']=='AUDITED_DESCRIPTIVE_DEFERRED_VERIFICATION_PILOT' and report['cells']==CELLS,'Complete report required')
    require(report['source_bindings']['audit/summary.json']==sha(run/'audit/summary.json'),'Report audit binding changed')
    terminal={'status':'COMPLETE_DEFERRED_PILOT_AUDITED_AND_REPORTED','run':run.name,'utc':utc(),
        'audit_sha256':sha(run/'audit/summary.json'),'report_sha256':sha(run/'derived_report/summary.json'),
        'physical_points':audit['physical_points'],'numerical_entries':audit['numerical_entries'],'new_training':0}
    save(run/'terminal_status.json',terminal); seal(run)
    save(ACTIVE,{**terminal,'status':'COMPLETE_DEFERRED_PILOT_SEALED','completion_sha256':sha(run/'completion_manifest.json')})
    return terminal


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('command',choices=('prepare','worker','execute','finalize'))
    p.add_argument('--run',type=Path); p.add_argument('--cell',type=Path); p.add_argument('--tests',type=Path)
    a=p.parse_args()
    if a.command=='prepare': print(prepare(a.tests.resolve()))
    elif a.command=='worker': sys.exit(worker(a.run.resolve(),a.cell.resolve()))
    elif a.command=='execute': sys.exit(execute(a.run.resolve()))
    else: print(finalize(a.run.resolve()))
