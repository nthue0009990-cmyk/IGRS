"""Independent P / E / no-free-pruning replay and cold-cost audit.

Each arm follows its actual phase order. Immutable numerical evidence,
certificate, shared-ledger and original-Beam checkers are reused privately.
No production search, selector, runner or solver is imported.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import math
import sys
import time
import traceback
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
sys.dont_write_bytecode=True
import audit_clpu_guard_first_pilot as gf

energy,staged,shared,early,prior,branch,window,old=(gf.energy,gf.staged,gf.shared,gf.early,gf.prior,gf.branch,gf.window,gf.old)
require,close,same=gf.require,gf.close,gf.same
read,save,sha,digest,read_lines=gf.read,gf.save,gf.sha,gf.digest,gf.read_lines
private=gf.private_function
STUDY='CLPU_PUBLIC_MECHANISM_COMMON_PILOT_V1'
SCHEMA='clpu-public-mechanism-protected-result-v1'
PROTOCOL='docs/clpu_public_mechanism_pilot_protocol_v1.md'
SOURCE_RUN,SOURCE_SHA=gf.SOURCE_RUN,gf.SOURCE_SHA
COMPARISON_RUN='20261031T103802_907721Z_clpu_guard_first_pilot'
COMPARISON_SHA='af9dc11957274cb84317e6f227fd94bf0f75df35f207667e26ff466d0678ac02'
ARMS=('P','E','A_prune')
CELLS=24
ABLATION_VERSION='clpu-public-no-free-pruning-v1'
ADDITIONS={PROTOCOL,'src/dnr/nature34_clpu_public_ablation_control.py',
    'scripts/run_clpu_public_mechanism_pilot.py','scripts/audit_clpu_public_mechanism_pilot.py',
    'tests/test_nature34_clpu_public_ablation_control.py','tests/test_run_clpu_public_mechanism_pilot.py',
    'tests/test_clpu_public_mechanism_audit.py'}


def phase_order(arm):
    require(arm in ARMS,'Unknown public mechanism arm')
    return ['EARLY','GUARD'] if arm=='E' else ['GUARD','EARLY']


class NoFreePruneReplay(gf.GuardFirstReplay):
    def refresh(self):
        # This explicitly bypasses only GF's added free queue deletion. The
        # older staged termination checks and own VERIFY skip remain intact.
        return staged.StagedReplay.refresh(self)

    def scientific_snapshot(self):
        value=super().scientific_snapshot()
        require(self.free_pruning==[],'No-prune arm secretly performs free queue deletion')
        value['guard_first']['rule']['prune']='disabled_for_A_prune_only'
        value['guard_first']['rule']['ablation']='free_certified_objective_queue_removal_disabled'
        value['guard_first']['rule']['pruning_cost']='not_applicable_free_removal_disabled'
        value['guard_first']['rule']['exposure']='unchanged_staged_pending_window_with_external_GUARD_features_without_free_removal'
        value['guard_first'].update(ablation_version=ABLATION_VERSION,free_objective_pruning_enabled=False)
        return value


def no_prune_adapter_metadata(replay):
    value=gf.adapter_metadata(replay)
    value.update(dominated_pending='no_free_queue_removal_original_EARLY_own_incumbent_verify_skip_retained',
        ablation_version=ABLATION_VERSION)
    return value


_no_prune_base_result=private(staged.StagedMacroReplay.check_control_result,adapter_metadata=no_prune_adapter_metadata)
_no_prune_gf_result=private(gf.GuardFirstMacroReplay.check_control_result,_result_check=_no_prune_base_result)


class NoFreePruneMacroReplay(gf.GuardFirstMacroReplay):
    def check_control_result(self,result):
        value=deepcopy(result)
        require(value['control'].pop('ablation_version')==ABLATION_VERSION
            and value['control'].pop('free_objective_pruning_enabled') is False,'Ablation metadata missing or enabled')
        return _no_prune_gf_result(self,value)


check_no_prune_records=private(gf.check_early_records,GuardFirstReplay=NoFreePruneReplay,GuardFirstMacroReplay=NoFreePruneMacroReplay)
_p_cell=private(gf.check_cell,STUDY=STUDY,SCHEMA=SCHEMA)
_e_cell=private(energy.check_cell,STUDY=STUDY,SCHEMA=SCHEMA)
_a_cell=private(gf.check_cell,STUDY=STUDY,SCHEMA=SCHEMA,check_early_records=check_no_prune_records)


def check_cell(run,spec,cfg,scope):
    arm=spec['arm'];order=phase_order(arm)
    require(spec['route_arm']==arm and spec['phase_order']==order and spec['method']=='beam' and spec['seed']==0
        and spec['checkpoint'] is spec['checkpoint_sha256'] is None,'Learned selector or substituted route in deterministic comparison')
    checked={'P':_p_cell,'E':_e_cell,'A_prune':_a_cell}[arm](run,spec,cfg,scope)
    cell=run/spec['cell'];result=read(cell/'result.json');phase=read(cell/'EARLY/phase_result.json')
    require(result['phase_order']==order and result['model_load_seconds']==0 and result['policy_decisions']==0,
        'Actual phase order or absent-model cost/participation differs')
    require(result['route_arm']==arm and result['models_used']==0
        and result['external_floor_enabled'] is (arm!='E') and result['free_external_floor_pruning_enabled'] is (arm=='P'),
        'Claimed mechanism differs from the fixed route')
    if arm=='E':
        require('guard_first' not in phase and result['guard_first'] is None and result['intermediate_beam_floor_claimed'] is False,
            'EARLY-first lane received a future GUARD floor')
        free=0
    else:
        free=phase['guard_first']['free_pruned_complete_plans']
        if arm=='A_prune':require(free==0 and phase['guard_first']['free_pruning']==[],'No-prune arm removes pending plans for free')
    skipped=sum(event['verification_status']=='SKIPPED_DOMINATED' for event in phase['deferred_trace'])
    checked.update(route_arm=arm,phase_order=order,models_used=0,free_pruned_complete_plans=free,
        own_incumbent_verify_skips=skipped,free_pruning_enabled=arm=='P',causal_phase_order_verified=True)
    return checked


def context_equalities(checks):
    require(len(checks)==CELLS and {c['context_id'] for c in checks}==set(range(8)),'Incomplete fixed observed population')
    results=[]
    for context in range(8):
        group=[r for r in checks if r['context_id']==context]
        require(len(group)==3 and {r['arm'] for r in group}==set(ARMS),'Missing or duplicate mechanism arm')
        require(all(r['guard_established'] and r['history_verified'] and r['models_used']==0 for r in group),'Incomplete original GUARD or model participation')
        require(len({r['candidate_pool_sha256'] for r in group})==1,'Initial pure-graph candidate pools differ')
        require(len({r['phase_summary']['GUARD']['semantic_signature_sha256'] for r in group})==1
            and len({r['guard_quality_pct'] for r in group})==1,'Original GUARD scientific trace/quality differs')
        # Histories and visibility may differ: stage order and free deletion
        # are the interventions, not nuisance fields to normalize away.
        results.append({'context_id':context,'status':'PASS','tasks':3,'guard_trajectories_equal':True,
            'candidate_pool_sha256':group[0]['candidate_pool_sha256'],'guard_quality_pct':group[0]['guard_quality_pct'],
            'equal_initial_history_or_exposure_claimed':False})
    return results


def check_controller(run,schedule,checks):
    require(len(schedule)==len(checks)==CELLS,'All24 actual tasks required')
    require(not any((run/name).exists() for name in ('phase_gate.json','audit_stage1','controller_stage2_started.json')),
        'Unapproved conditional population or second stage')
    execution=read(run/'controller_stage1_completion.json');controller=read(run/'controller_completion.json')
    require(execution['exits']==read(run/'controller_stage1_exits.json'),'True exit journal changed')
    dispatch=list(read_lines(run/'controller_dispatch.jsonl'))
    require(dispatch==list(read_lines(run/'controller_stage1_dispatch.jsonl')),'Dispatch journal differs')
    names=[s['cell'] for s in schedule]
    require(len(set(names))==CELLS and {c['cell'] for c in checks}==set(names),'Duplicate or omitted task')
    require(execution['status']=='COMPLETE_STAGE1_PENDING_AUDIT' and execution['stage']=='phase1'
        and execution['scheduled']==execution['scheduled_total']==CELLS and execution['selected_cells']==names
        and execution['dispatched']==schedule and len(execution['completed'])==CELLS and set(execution['completed'])==set(names)
        and execution['errors']==execution['unknown_cost_cells']==[] and execution['peak_active']<=2,'Actual controller did not close all tasks')
    require([d['cell'] for d in dispatch]==names and len(execution['exits'])==CELLS,'Retry/omission/order changed')
    exits={e['cell']:e for e in execution['exits']};by_cell={c['cell']:c for c in checks}
    require(set(exits)==set(names),'Exit identity mismatch')
    previous_end=0.
    for start in range(0,CELLS,2):
        ends=[]
        for spec,d in zip(schedule[start:start+2],dispatch[start:start+2]):
            e,c=exits[spec['cell']],by_cell[spec['cell']]
            require(d['stage']==c['execution']['stage']=='phase1' and e['exit_code']==0
                and e['drained_after_controller_error'] is False and e['launcher_pid']==d['launcher_pid']
                and e['pair_order']==d['pair_order']==spec['pair_order'],'Wrong route stage/failed exit or retry')
            begin=window.finite_time(e['started_epoch_seconds'],'process start');end=window.finite_time(e['finished_epoch_seconds'],'process exit')
            elapsed=window.finite_time(e['process_wall_seconds'],'full process time')
            require(begin==d['started_epoch_seconds']==c['execution']['started_epoch_seconds'] and previous_end<=begin<=end
                and elapsed>=c['worker_finished_elapsed_seconds'],'Earlier pair not drained or paid time omitted')
            require(c['worker_parent_pid'] in (d['launcher_pid'],c['execution']['controller_pid'])
                and abs(end-begin-elapsed)<=2.,'Process ancestry or wall clocks differ')
            c['process_wall_seconds']=elapsed;ends.append(end)
        previous_end=max(ends)
    totals={k:sum(c[k] for c in checks) for k in ('logical_points','physical_points','numerical_entries','requests')}
    require(all(execution[k]==v for k,v in totals.items()) and totals['physical_points']<=12288
        and totals['logical_points']<=12288 and totals['numerical_entries']==CELLS*3+4*totals['physical_points'],'Full cold cost does not close')
    require(controller['status']=='COMPLETE_CELLS_PENDING_AUDIT' and controller['scheduled']==controller['resolved_slots']==controller['actual_tasks']==CELLS
        and controller['not_run']==[] and controller['all_eight_context_mean_allowed'] is True
        and controller['errors']==controller['unknown_cost_cells']==[],'Top full-population completion changed')
    require(controller['controller_execution_sha256']==sha(run/'controller_stage1_completion.json'),'Completion not bound to true process records')
    for key in ('completed','dispatched','exits','logical_points','physical_points','numerical_entries','requests'):
        require(controller[key]==execution[key],'Whole-run execution cost altered: '+key)
    require(window.finite_time(controller['wall_seconds'],'controller wall')>=window.finite_time(execution['wall_seconds'],'stage wall')
        >=max(c['process_wall_seconds'] for c in checks),'Controller timer omits actual processes')
    return totals


def protocol_settings():
    value=staged.protocol_settings()
    for key in ('conditional_stages','no_gain_remaining_status'):value.pop(key)
    value.update(cells=24,slots=list(ARMS),seeds=[],models_used=0,execution='single_stage_all24',quality_gate=False,
        total_physical_budget=12288,maximum_numerical_entries=49224,phase_order='explicit per arm',phases=['EARLY','GUARD'],
        arm_phase_orders={arm:phase_order(arm) for arm in ARMS},external_floor={'P':True,'E':False,'A_prune':True},
        free_external_floor_pruning={'P':True,'E':False,'A_prune':False},
        own_incumbent_validation_skip='unchanged original EARLY complete-objective skip in all arms',
        model_role='none; all three arms use the unchanged visible complete-energy selector',
        dispatch='public-mechanism-context-v1 hashed contexts; three rounds, arm=(context_rank+round)%3; adjacent pairs drained',
        aggregation='paired three deterministic arms within each of all eight seen contexts; all24 retained',
        comparison='current cold24 process times only; prior GUARD-first data is source/reference, not a timing denominator',
        internal_execution_journal='controller_stage1_* names retained for one all24 pass only',
        contrast_P_E='whole GUARD-first information/order mechanism, not identical intermediate windows',
        contrast_P_A_prune='free certified-floor batch removal conditional on GUARD-first; original own skip retained')
    return value


def expected_schedule(contexts,models=()):
    require(not models and len(contexts)==8 and {c['context_id'] for c in contexts}==set(range(8))
        and len({c['scenario_id'] for c in contexts})==8,'Changed seen-only deterministic population')
    ordered=sorted(contexts,key=lambda c:prior.digest_text('public-mechanism-context-v1|'+prior.canonical(c)))
    rows=[]
    for turn in range(3):
        for rank,context in enumerate(ordered):
            arm=ARMS[(rank+turn)%3];index=len(rows)
            rows.append({**{k:context[k] for k in ('context_id','scenario_id','event_id')},
                'cell':f'episodes/{index:03d}','episode_id':index,'pair_order':index//2,'context_order':rank,
                'arm_position':turn,'dispatch_round':turn,'arm_slot':arm,'arm':arm,'route_arm':arm,
                'phase_order':phase_order(arm),'external_floor_enabled':arm!='E','free_external_floor_pruning_enabled':arm=='P',
                'mode':'shared','method':'beam','seed':0,'macro_schedule':'staged','phase':'evaluation','group':arm,
                'protected':True,'representation':'raw','deterministic':True,'checkpoint':None,'checkpoint_sha256':None,
                'point_budget':256,'logical_point_budget':256,'physical_point_budget':512,'total_logical_point_budget':512,
                'window_variant':'staged_common_pool_coverage_v1'})
    return rows


def check_comparison(run):
    require(read(run/'comparison_source_binding.json')=={'source_run':COMPARISON_RUN,'source_completion_sha256':COMPARISON_SHA,
        'role':'matched prior candidate/scientific reference; no historical answers or cross-date time denominator'},'Prior attribution changed')
    require(sha(run/'comparison_source_completion_manifest.json')==COMPARISON_SHA,'Previous scientific seal changed')
    sealed=energy.normalized_manifest(read(run/'comparison_source_completion_manifest.json'))
    for name in energy.COMPARISON_FILES:require(sha(run/'comparison_source'/name)==sealed[name],'Previous small evidence changed: '+name)
    summary=read(run/'comparison_source/audit/summary.json')
    require(summary['status']=='PASS' and summary['audited_cells']==summary['resolved_slots']==32
        and summary['complete_protected_cells']==32 and not summary['errors'],'Prior complete public audit missing')
    require(read(run/'comparison_source/contexts.json')==read(run/'contexts.json'),'Original eight contexts changed')
    original=read(run/'comparison_source/stage_plan.json');plan=read(run/'stage_plan.json')
    require(plan=={'schema':'clpu-public-mechanism-fixed-population-v1','original_contexts':8,'scheduled_cells':24,
        'execution':'single_stage_all24','quality_gate':False,'all_cells':[s['cell'] for s in read(run/'schedule.json')],
        'eligible_contexts':original['eligible_contexts'],'eligible_contexts_role':'historical graph diagnostic only; does not gate dispatch',
        'contexts':original['contexts'],'new_physical_points':0,'new_training':0},'Altered candidate generation or conditional cohort')
    staged.check_graph_profile(run,plan)
    return sealed


def check_parent(run):
    run=Path(run).resolve();require(run.parent==ROOT/'runs','Pilot outside project runs')
    prepared=read(run/'prepared.json')
    require(prepared['status']=='FROZEN_PREPARED_NO_DSS' and prepared['freeze_sha256']==sha(run/'prephysical_freeze_manifest.json'),
        'Prephysical freeze changed')
    frozen=shared.checked_manifest(run,'prephysical_freeze_manifest.json')
    source=ROOT/'runs'/SOURCE_RUN
    require(sha(source/'completion_manifest.json')==sha(run/'source_completion_manifest.json')==SOURCE_SHA,'Original science seal changed')
    source_manifest=energy.normalized_manifest(read(run/'source_completion_manifest.json'))
    require(read(run/'source_binding.json')=={'source_run':SOURCE_RUN,'source_completion_sha256':SOURCE_SHA,
        'models_used':0,'historical_answers_loaded':False},'Science/source cache attribution changed')
    for name in ('config.json','schedule.json','prepared.json','audit/summary.json','contexts.json'):
        require(sha(source/name)==source_manifest[name],'Inherited input changed: '+name)
    original_audit=read(source/'audit/summary.json')
    require(original_audit['status']=='PASS' and original_audit['audited_cells']==64 and not original_audit['errors'],'Original complete audit absent')
    previous_manifest=check_comparison(run);inherited=read(run/'comparison_source/prepared.json')['source_files']
    require(prepared['source_files']==sorted(set(inherited)|ADDITIONS),'Required frozen source closure differs')
    for name in prepared['source_files']:
        path=(ROOT/name).resolve()
        require(path.is_relative_to(ROOT) and path!=ROOT and 'source/'+name in frozen and sha(path)==sha(run/'source'/name),
            'Changed frozen source: '+name)
        if name in inherited:require(sha(path)==previous_manifest['source/'+name],'Changed previously sealed source: '+name)
    require(sha(run/'protocol.md')==sha(ROOT/PROTOCOL),'Protocol differs from freeze')
    cfg=read(run/'config.json')
    require(cfg=={**read(source/'config.json'),'staged_study':STUDY,'staged_pilot':protocol_settings()},'Unapproved scientific/configuration change')
    scope=prior.check_frozen_physics(run,prepared,cfg)
    require(read(run/'model_bindings.json')==[] and not (run/'models').exists(),'Learned endpoint included in no-model study')
    contexts=read(source/'contexts.json');require(read(run/'contexts.json')==contexts,'Observed context population changed')
    schedule=read(run/'schedule.json');require(schedule==expected_schedule(contexts),'All24 preselected identity/order differs')
    for spec in schedule:
        require(read(run/spec['cell']/'spec.json')==spec,'Cold cell spec differs')
        for phase in phase_order(spec['arm']):
            require(read(run/spec['cell']/phase/'spec.json')==staged.expected_phase_spec(spec,phase),'Logical phase spec differs')
    chosen=[s for s in read(source/'schedule.json') if s['arm']=='deferred_window'];history=[]
    for spec in chosen:
        original=spec['cell']+'/GUARD/phase_result.json';target=f'historical/c{spec["context_id"]:02d}/GUARD/phase_result.json'
        require(sha(run/target)==source_manifest[original],'Original full GUARD changed')
        history.append({**{k:spec[k] for k in ('context_id','scenario_id','event_id')},'source_cell':spec['cell'],
            'phases':{'GUARD':{'result_file':target,'source_file':original,'sha256':source_manifest[original]}}})
    require(read(run/'historical_bindings.json')==history,'Original GUARD attribution differs')
    cases=list(ET.parse(run/'software_tests.xml').getroot().iter('testcase'))
    require(cases and not any(list(c.iter('failure'))+list(c.iter('error'))+list(c.iter('skipped')) for c in cases),'Passing component evidence missing')
    for part in ('test_nature34_clpu_public_ablation_control','test_run_clpu_public_mechanism_pilot','test_clpu_public_mechanism_audit'):
        require(any(part in c.get('classname','') for c in cases),'Missing component: '+part)
    resource=read(run/'resource_budget.json')
    for key,value in {'total_physical_cap':12288,'minimum_free_bytes':12*2**30,'report_reserve_bytes':2*2**30,
        'successful_component_tests':len(cases)}.items():require(resource[key]==value,'Changed resource cap: '+key)
    rate=max(sum(p.stat().st_size for p in (source/s['cell']).rglob('*') if p.is_file())/
        max(1,read(source/s['cell']/'result.json')['actual_physical_points']) for s in chosen)
    require(resource['historical_max_bytes_per_actual_point']==rate and resource['forecast_bytes']==math.ceil(rate*12288*1.25)
        and resource['required_free_bytes']==14*2**30+resource['forecast_bytes']
        and resource['free_bytes_at_prepare']>=resource['required_free_bytes'],'Incomplete total storage forecast')
    auth=read(run/'authorization.json')
    require(auth['execution_authorized'] is True and auth['models_used']==auth['new_training']==0 and auth['no_automatic_retry'] is True,
        'Missing authorized scope or changed learning/retry policy')
    return cfg,scope,schedule


def source_hashes(run,scope):
    files={'freeze_sha256':'prephysical_freeze_manifest.json','config_sha256':'config.json','schedule_sha256':'schedule.json',
        'source_binding_sha256':'source_binding.json','physical_scope_file_sha256':'physical_scope.json',
        'scope_review_manifest_sha256':'scope_review/completion_manifest.json','exact_key_review_manifest_sha256':'exact_key_review/completion_manifest.json',
        'historical_bindings_sha256':'historical_bindings.json','model_bindings_sha256':'model_bindings.json',
        'stage_plan_sha256':'stage_plan.json','graph_profile_manifest_sha256':'graph_profile/completion_manifest.json',
        'comparison_source_binding_sha256':'comparison_source_binding.json'}
    return {**{key:sha(run/name) for key,name in files.items()},'physical_scope_canonical_sha256':digest(scope),
        'source_deferred_completion_sha256':SOURCE_SHA,'comparison_source_completion_sha256':COMPARISON_SHA,
        'audit_implementation_sha256':sha(Path(__file__)),'core_audit_sha256':sha(Path(staged.__file__)),
        'guard_first_audit_sha256':sha(Path(gf.__file__)),'ranking_audit_sha256':sha(Path(energy.__file__))}


def audit(run,output=None):
    from audit_d059_supplement import NoSolverImports,no_solver_loaded
    run=Path(run).resolve();output=Path(output).resolve() if output else run/'audit'
    require(output.is_relative_to(ROOT) and output!=run and not output.exists(),'Preserve previous audit evidence')
    output.mkdir(parents=True);started=time.perf_counter();checks=[]
    result={'status':'FAIL','source_run':run.name,'scheduled_cells':CELLS,'errors':[],'audited_cells':0,'results':[],
        'new_audit_physical_points':0,'new_training_tasks':0,'models_used':0}
    guard=NoSolverImports();sys.meta_path.insert(0,guard)
    try:
        no_solver_loaded();cfg,scope,schedule=check_parent(run);result.update(source_hashes(run,scope))
        execution=read(run/'execution_started.json')
        require(execution['deadline_epoch_seconds']==execution['started_epoch_seconds']+3600 and execution['execution']=='single_stage_all24',
            'Global execution deadline or one-pass contract changed')
        for spec in schedule:
            checked=check_cell(run,spec,cfg,scope);checks.append(checked)
            save(output/(spec['cell'].replace('/','_')+'.json'),checked)
            save(output/'progress.json',{'status':'RUNNING','audited_cells':len(checks),'scheduled':CELLS,
                'last_cell':spec['cell'],'elapsed_seconds':time.perf_counter()-started,'new_physical_points':0})
        totals=check_controller(run,schedule,checks);equalities=context_equalities(checks);statuses=Counter()
        for row in checks:
            statuses.update(row['physical_statuses']);save(output/(row['cell'].replace('/','_')+'.json'),row)
        result.update(status='PASS',audited_cells=CELLS,resolved_slots=CELLS,actual_tasks=CELLS,not_run=[],results=checks,
            all_eight_context_mean_allowed=True,complete_protected_cells=CELLS,physical_runtimes=CELLS,logical_searches=2*CELLS,
            context_equalities=equalities,physical_statuses=dict(statuses),**totals,
            controller_completion_sha256=sha(run/'controller_completion.json'),controller_execution_sha256=sha(run/'controller_stage1_completion.json'),
            controller_wall_seconds=read(run/'controller_completion.json')['wall_seconds'])
        no_solver_loaded()
    except BaseException as exc:
        result.update(audited_cells=len(checks),results=checks);result['errors'].append({'error':repr(exc),'traceback':traceback.format_exc()})
    finally:sys.meta_path.remove(guard)
    result['elapsed_seconds']=time.perf_counter()-started;save(output/'summary.json',result)
    save(output/'completion_manifest.json',{p.relative_to(output).as_posix():sha(p)
        for p in output.rglob('*') if p.is_file() and p.name!='completion_manifest.json'})
    return result


def preflight(run,output):
    from audit_d059_supplement import NoSolverImports,no_solver_loaded
    run,output=Path(run).resolve(),Path(output).resolve();require(output.is_relative_to(ROOT) and not output.exists(),'Preserve earlier preflight')
    output.mkdir(parents=True);result={'status':'FAIL','errors':[],'new_physical_points':0,'new_training':0}
    guard=NoSolverImports();sys.meta_path.insert(0,guard)
    try:
        no_solver_loaded();_,scope,schedule=check_parent(run);no_solver_loaded()
        result.update(status='PASS',contexts=8,cells=len(schedule),physical_runtimes=CELLS,logical_searches=2*CELLS,models_used=0,**source_hashes(run,scope))
    except BaseException as exc:result['errors'].append({'error':repr(exc),'traceback':traceback.format_exc()})
    finally:sys.meta_path.remove(guard)
    save(output/'summary.json',result);save(output/'completion_manifest.json',{'summary.json':sha(output/'summary.json')})
    return result


main=private(energy.main,audit=audit,preflight=preflight,__doc__=__doc__)
if __name__=='__main__':raise SystemExit(main())
