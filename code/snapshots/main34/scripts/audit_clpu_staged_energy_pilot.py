"""Independent narrow audit of complete-energy staged ranking.

The immutable staged graph, physical, alias, policy and guard checkers remain
authoritative. Only the deterministic visible-plan choice and run population
contract are new. No production engine, selector or runner is imported.
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
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
sys.dont_write_bytecode=True
import audit_clpu_staged_pilot as staged

shared,early,prior,branch,window,old,previous=(staged.shared,staged.early,staged.prior,
    staged.branch,staged.window,staged.old,staged.previous)
require,close,same=staged.require,staged.close,staged.same
read,save,sha,digest,read_lines=staged.read,staged.save,staged.sha,staged.digest,staged.read_lines
STUDY='CLPU_STAGED_ENERGY_COMMON_PILOT_V1'
SCHEMA='clpu-staged-energy-protected-result-v1'
PROTOCOL='docs/clpu_staged_energy_pilot_protocol_v1.md'
COMPARISON_RUN='20261031T085600_616395Z_clpu_staged_pilot'
COMPARISON_SHA='0d288959bece92f5e274ce43e0c2d29520fe15094dbc8663ebcc3a8e3a3ceefb'
SOURCE_RUN,SOURCE_SHA=staged.SOURCE_RUN,staged.SOURCE_SHA
MODELS_RUN,MODELS_SHA=staged.MODELS_RUN,staged.MODELS_SHA
SEEDS=staged.SEEDS
SLOTS=('energy_window','staged_bc17','staged_bc41','staged_bc73')


def visible_energy_choice(nodes):
    require(nodes and len({n['id'] for n in nodes})==len(nodes), 'Empty/duplicated visible plan set')
    require(all(math.isfinite(n['info']['energy_kwh']) for n in nodes), 'Nonfinite complete graph objective')
    return min(range(len(nodes)),key=lambda i:(-nodes[i]['info']['energy_kwh'],nodes[i]['id']))


class EnergyView(staged.StagedView):
    def rule(self,method):
        return visible_energy_choice(self.window()) if method=='beam' else super().rule(method)


class EnergyReplay(staged.StagedReplay):
    def verification_view(self):
        original,aliases=super().verification_view()
        view=object.__new__(EnergyView);view.__dict__=dict(original.__dict__)
        return view,aliases


def selector_metadata():
    version='clpu-staged-visible-complete-energy-selector-v1'
    return {'selector_version':version,'beam_rule':{
        'version':version,'eligible':'current_common_visible_window_and_true_action_mask_only',
        'primary':'maximum_complete_graph_energy_kwh','tie':'exact_equal_energy_then_smallest_branch_id',
        'prefix_depth_priority':False,'future_physical_outcomes':False,
        'fixed_actor_path':'inherited_StagedMacroController_choose_verify_unchanged',
        'guard':'unchanged_original_BranchSearch_beam'}}


class EnergyMacroReplay(staged.StagedMacroReplay):
    def check_control_result(self,result):
        actual=deepcopy(result)
        for key,value in selector_metadata().items():
            require(actual['control'].pop(key)==value,'Complete-energy selector declaration changed')
        require(actual['control'].pop('active_selector')==('complete_graph_energy' if self.method=='beam'
            else 'fixed_raw_actor'), 'Claimed selector differs from executed ranking')
        return super().check_control_result(actual)


_records=FunctionType(staged._check_records.__code__,
    {**staged._check_records.__globals__,'DeferredReplay':EnergyReplay,'MacroReplay':EnergyMacroReplay},
    '_records',staged._check_records.__defaults__,staged._check_records.__closure__)
check_staged_records=FunctionType(staged.check_staged_records.__code__,
    {**staged.check_staged_records.__globals__,'_check_records':_records},
    'check_staged_records',staged.check_staged_records.__defaults__,staged.check_staged_records.__closure__)


def check_cell(run,spec,cfg,scope):
    """Adapt version labels explicitly; retain every old scientific check."""
    def bound_read(path):
        path=Path(path)
        value=read(path)
        if path==run/spec['cell']/'result.json':
            require(value['version']==value['study']==STUDY and value['schema']==SCHEMA,
                'New run is mislabeled or old evidence substituted')
            value={**value,'schema':'clpu-staged-protected-result-v1'}
        return value
    base=FunctionType(staged._check_base_cell.__code__,
        {**staged._check_base_cell.__globals__,'read':bound_read,'STUDY':STUDY,
         'check_staged_records':check_staged_records},'_energy_base_cell',
        staged._check_base_cell.__defaults__,staged._check_base_cell.__closure__)
    outer=FunctionType(staged.check_cell.__code__,{**staged.check_cell.__globals__,'_check_base_cell':base},
        '_energy_cell',staged.check_cell.__defaults__,staged.check_cell.__closure__)
    return outer(run,spec,cfg,scope)


def protocol_settings():
    value=staged.protocol_settings()
    del value['conditional_stages'];del value['no_gain_remaining_status']
    value.update(slots=list(SLOTS),execution='single_stage_all32',
        internal_execution_journal='controller_stage1_* names retained for one all32 pass only',
        quality_gate=False,ranking_change='current visible complete graph_energy_kwh descending then branch_id ascending; no prefix-depth restriction',
        comparison='only current cold32 process times; previous depth-priority run is a mechanism/semantic reference',
        same_optimized_window_for_all_rankers=True)
    return value


expected_schedule=FunctionType(staged.expected_schedule.__code__,{**staged.expected_schedule.__globals__,'SLOTS':SLOTS},
    'expected_schedule',staged.expected_schedule.__defaults__,staged.expected_schedule.__closure__)
expected_phase_spec=staged.expected_phase_spec
ADDITIONS={PROTOCOL,'src/dnr/nature34_clpu_staged_energy_control.py',
    'scripts/run_clpu_staged_energy_pilot.py','scripts/audit_clpu_staged_energy_pilot.py',
    'tests/test_nature34_clpu_staged_energy_control.py','tests/test_run_clpu_staged_energy_pilot.py',
    'tests/test_clpu_staged_energy_audit.py'}
COMPARISON_FILES=('config.json','schedule.json','contexts.json','stage_plan.json','prepared.json',
    'audit/summary.json','terminal_status.json')


def normalized_manifest(rows):
    require(isinstance(rows,dict),'Manifest must be an object')
    result={};aliases=set()
    for name,value in rows.items():
        require(isinstance(name,str) and isinstance(value,str) and len(value)==64
            and all(c in '0123456789abcdef' for c in value),'Malformed source manifest')
        key=name.replace('\\','/')
        require(not Path(key).is_absolute() and not Path(key).drive
            and all(p not in ('','.','..') for p in key.split('/')) and key.casefold() not in aliases,
            'Escaped or aliased source manifest path')
        result[key]=value;aliases.add(key.casefold())
    return result


def check_comparison(run):
    require(read(run/'comparison_source_binding.json')=={'source_run':COMPARISON_RUN,
        'source_completion_sha256':COMPARISON_SHA,
        'role':'matched prior candidate/scientific reference; no historical answers or cross-date time denominator'},
        'Historical scientific reference attribution changed')
    require(sha(run/'comparison_source_completion_manifest.json')==COMPARISON_SHA,'Historical root seal differs')
    sealed=normalized_manifest(read(run/'comparison_source_completion_manifest.json'))
    for name in COMPARISON_FILES:
        require(sha(run/'comparison_source'/name)==sealed[name],'Changed small historical evidence: '+name)
    summary=read(run/'comparison_source/audit/summary.json')
    require(summary['status']=='PASS' and summary['audited_cells']==summary['resolved_slots']==32
        and summary['complete_protected_cells']==32 and not summary['errors'],'Complete prior independent audit absent')
    require(read(run/'comparison_source/contexts.json')==read(run/'contexts.json'),'Training contexts changed')
    original=read(run/'comparison_source/stage_plan.json')
    plan=read(run/'stage_plan.json')
    require(plan['contexts']==original['contexts'],'Previously independently verified graph candidates changed')
    require(plan=={'schema':'clpu-staged-energy-fixed-population-v1','original_contexts':8,'scheduled_cells':32,
        'execution':'single_stage_all32','quality_gate':False,'all_cells':[s['cell'] for s in read(run/'schedule.json')],
        'eligible_contexts':original['eligible_contexts'],
        'eligible_contexts_role':'historical graph diagnostic only; does not gate dispatch',
        'contexts':original['contexts'],'new_physical_points':0,'new_training':0}, 'Conditional gate or changed population introduced')
    staged.check_graph_profile(run,plan)
    return sealed


def check_parent(run):
    run=Path(run).resolve()
    require(run.parent==ROOT/'runs','New pilot outside project run directory')
    prepared=read(run/'prepared.json')
    require(prepared['status']=='FROZEN_PREPARED_NO_DSS' and prepared['freeze_sha256']==sha(run/'prephysical_freeze_manifest.json'),
        'Prephysical freeze changed')
    frozen=shared.checked_manifest(run,'prephysical_freeze_manifest.json')
    require(sha(run/'source_completion_manifest.json')==SOURCE_SHA,'Original science root changed')
    source=ROOT/'runs'/SOURCE_RUN
    require(sha(source/'completion_manifest.json')==SOURCE_SHA,'Original root seal differs')
    source_manifest=normalized_manifest(read(run/'source_completion_manifest.json'))
    require(read(run/'source_binding.json')=={'source_run':SOURCE_RUN,'source_completion_sha256':SOURCE_SHA,
        'models_run':MODELS_RUN,'models_completion_sha256':MODELS_SHA},'Science/model attribution changed')
    for name in ('config.json','schedule.json','prepared.json','audit/summary.json','contexts.json'):
        require(sha(source/name)==source_manifest[name],'Changed inherited input: '+name)
    original_audit=read(source/'audit/summary.json')
    require(original_audit['status']=='PASS' and original_audit['audited_cells']==64 and not original_audit['errors'],
        'Original independent scientific audit absent')
    previous_manifest=check_comparison(run)
    inherited=read(run/'comparison_source/prepared.json')['source_files']
    require(prepared['source_files']==sorted(set(inherited)|ADDITIONS),'Required source closure changed')
    for name in prepared['source_files']:
        path=(ROOT/name).resolve()
        require(path.is_relative_to(ROOT) and path!=ROOT and 'source/'+name in frozen
            and sha(path)==sha(run/'source'/name),'Changed frozen execution source: '+name)
        if name in inherited:
            require(sha(path)==previous_manifest['source/'+name],'Changed previously sealed science: '+name)
    require(sha(run/'protocol.md')==sha(ROOT/PROTOCOL),'Frozen protocol differs')
    cfg=read(run/'config.json')
    require(cfg=={**read(source/'config.json'),'staged_study':STUDY,'staged_pilot':protocol_settings()},
        'Unapproved scientific/configuration change')
    scope=prior.check_frozen_physics(run,prepared,cfg)
    models=staged.check_models(run)
    contexts=read(source/'contexts.json')
    require(read(run/'contexts.json')==contexts,'Original observed training population changed')
    schedule=read(run/'schedule.json')
    require(schedule==expected_schedule(contexts,models),'All32 deterministic population/Latin order changed')
    for spec in schedule:
        require(read(run/spec['cell']/'spec.json')==spec,'Cold task spec differs')
        for phase in shared.VIEWS:
            require(read(run/spec['cell']/phase/'spec.json')==expected_phase_spec(spec,phase),'Logical view configuration differs')
    chosen=[s for s in read(source/'schedule.json') if s['arm']=='deferred_window']
    expected_history=[]
    for spec in chosen:
        original=spec['cell']+'/GUARD/phase_result.json'
        target=f'historical/c{spec["context_id"]:02d}/GUARD/phase_result.json'
        require(sha(run/target)==source_manifest[original],'Original complete GUARD reference changed')
        expected_history.append({**{k:spec[k] for k in ('context_id','scenario_id','event_id')},'source_cell':spec['cell'],
            'phases':{'GUARD':{'result_file':target,'source_file':original,'sha256':source_manifest[original]}}})
    require(read(run/'historical_bindings.json')==expected_history,'Original GUARD attribution differs')
    cases=list(ET.parse(run/'software_tests.xml').getroot().iter('testcase'))
    require(cases and not any(list(c.iter('failure'))+list(c.iter('error'))+list(c.iter('skipped')) for c in cases),
        'Complete passing focused software evidence required')
    for name in ('test_nature34_clpu_staged_energy_control','test_run_clpu_staged_energy_pilot','test_clpu_staged_energy_audit'):
        require(any(name in c.get('classname','') for c in cases),'Missing new focused component: '+name)
    resource=read(run/'resource_budget.json')
    for key,value in {'total_physical_cap':16384,'minimum_free_bytes':12*2**30,'report_reserve_bytes':2*2**30,
        'successful_component_tests':len(cases)}.items():
        require(resource[key]==value,'Resource cap changed: '+key)
    rate=max(sum(p.stat().st_size for p in (source/s['cell']).rglob('*') if p.is_file())/
        max(1,read(source/s['cell']/'result.json')['actual_physical_points']) for s in chosen)
    require(resource['historical_max_bytes_per_actual_point']==rate and resource['forecast_bytes']==math.ceil(rate*16384*1.25)
        and resource['required_free_bytes']==14*2**30+resource['forecast_bytes']
        and resource['free_bytes_at_prepare']>=resource['required_free_bytes'],'Incomplete cold cost/storage forecast')
    require(read(run/'authorization.json')['execution_authorized'] is True,'Missing execution authorization')
    return cfg,scope,schedule


def check_controller(run,schedule,checks):
    require(len(schedule)==len(checks)==32 and {s['context_id'] for s in schedule}==set(range(8)),
        'All eight contexts and 32 new actual tasks are required')
    require(not any((run/name).exists() for name in ('phase_gate.json','audit_stage1','controller_stage2_started.json')),
        'Unapproved conditional gate or second execution stage')
    controller=read(run/'controller_completion.json')
    execution=read(run/'controller_stage1_completion.json')
    require(execution['exits']==read(run/'controller_stage1_exits.json'),'True process exit journal differs')
    dispatch=list(read_lines(run/'controller_dispatch.jsonl'))
    require(dispatch==list(read_lines(run/'controller_stage1_dispatch.jsonl')),'Dispatch journals differ')
    totals=staged.check_stage_controller(execution,schedule,checks,dispatch,'phase1')
    require(controller['status']=='COMPLETE_CELLS_PENDING_AUDIT' and controller['scheduled']==
        controller['resolved_slots']==controller['actual_tasks']==32 and controller['not_run']==[]
        and controller['all_eight_context_mean_allowed'] is True and not controller['errors']
        and not controller['unknown_cost_cells'],'Single full-population controller did not complete')
    require(controller['controller_execution_sha256']==sha(run/'controller_stage1_completion.json'),
        'Whole execution summary not bound to actual dispatch/exit journal')
    for key in ('completed','dispatched','exits','logical_points','physical_points','numerical_entries','requests'):
        require(controller[key]==execution[key],'Run summary alters complete execution: '+key)
    require(window.finite_time(controller['wall_seconds'],'full controller elapsed')>=execution['wall_seconds'],
        'Controller timer omits paid execution')
    return totals


def context_equalities(checks):
    require(len(checks)==32 and {r['context_id'] for r in checks}==set(range(8)), 'Missing complete comparison cohort')
    # Only the rule-arm name changes; these checks retain original complete
    # GUARD and exact first candidate-pool fairness for all four rankers.
    fn=FunctionType(staged.context_equalities.__code__,{**staged.context_equalities.__globals__,'SLOTS':SLOTS},
        'energy_context_equalities',staged.context_equalities.__defaults__,staged.context_equalities.__closure__)
    return fn(checks)


def source_hashes(run,scope):
    hashes=staged.source_hashes(run,scope)
    hashes.update(audit_implementation_sha256=sha(Path(__file__)),core_audit_sha256=sha(Path(staged.__file__)),
        comparison_source_binding_sha256=sha(run/'comparison_source_binding.json'),
        comparison_source_completion_sha256=COMPARISON_SHA)
    return hashes


def audit(run,output=None):
    from audit_d059_supplement import NoSolverImports,no_solver_loaded
    run=Path(run).resolve();output=Path(output).resolve() if output else run/'audit'
    require(output.is_relative_to(ROOT) and output!=run and not output.exists(),'Preserve previous audit evidence')
    output.mkdir(parents=True)
    started=time.perf_counter();checks=[]
    result={'status':'FAIL','source_run':run.name,'scheduled_cells':32,'errors':[],'audited_cells':0,
        'results':[],'new_audit_physical_points':0,'new_training_tasks':0}
    guard=NoSolverImports();sys.meta_path.insert(0,guard)
    try:
        no_solver_loaded();cfg,scope,schedule=check_parent(run);result.update(source_hashes(run,scope))
        execution=read(run/'execution_started.json')
        require(execution['deadline_epoch_seconds']==execution['started_epoch_seconds']+3600,'Global deadline changed')
        for spec in schedule:
            checked=check_cell(run,spec,cfg,scope);checks.append(checked)
            save(output/(spec['cell'].replace('/','_')+'.json'),checked)
            save(output/'progress.json',{'status':'RUNNING','audited_cells':len(checks),'scheduled':32,
                'last_cell':spec['cell'],'elapsed_seconds':time.perf_counter()-started,'new_physical_points':0})
        totals=check_controller(run,schedule,checks)
        equalities=context_equalities(checks)
        statuses=Counter()
        for row in checks:
            statuses.update(row['physical_statuses'])
            save(output/(row['cell'].replace('/','_')+'.json'),row)
        result.update(status='PASS',audited_cells=32,resolved_slots=32,actual_tasks=32,not_run=[],results=checks,
            all_eight_context_mean_allowed=True,complete_protected_cells=32,physical_runtimes=32,logical_searches=64,
            context_equalities=equalities,physical_statuses=dict(statuses),**totals,
            controller_completion_sha256=sha(run/'controller_completion.json'),
            controller_execution_sha256=sha(run/'controller_stage1_completion.json'),
            controller_wall_seconds=read(run/'controller_completion.json')['wall_seconds'])
        no_solver_loaded()
    except BaseException as exc:
        result.update(audited_cells=len(checks),results=checks)
        result['errors'].append({'error':repr(exc),'traceback':traceback.format_exc()})
    finally:sys.meta_path.remove(guard)
    result['elapsed_seconds']=time.perf_counter()-started
    save(output/'summary.json',result)
    save(output/'completion_manifest.json',{p.relative_to(output).as_posix():sha(p)
        for p in output.rglob('*') if p.is_file() and p.name!='completion_manifest.json'})
    return result


def preflight(run,output):
    from audit_d059_supplement import NoSolverImports,no_solver_loaded
    run,output=Path(run).resolve(),Path(output).resolve()
    require(output.is_relative_to(ROOT) and not output.exists(),'Preserve earlier preflight')
    output.mkdir(parents=True)
    result={'status':'FAIL','errors':[],'new_physical_points':0,'new_training':0}
    guard=NoSolverImports();sys.meta_path.insert(0,guard)
    try:
        no_solver_loaded();_,scope,schedule=check_parent(run);no_solver_loaded()
        result.update(status='PASS',contexts=8,cells=len(schedule),physical_runtimes=32,logical_searches=64,
            fixed_BC_models=3,**source_hashes(run,scope))
    except BaseException as exc:result['errors'].append({'error':repr(exc),'traceback':traceback.format_exc()})
    finally:sys.meta_path.remove(guard)
    save(output/'summary.json',result);save(output/'completion_manifest.json',{'summary.json':sha(output/'summary.json')})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path)
    parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    if args.preflight:
        require(args.output is not None,'Preflight requires separate output');result=preflight(args.run,args.output)
    else:result=audit(args.run,args.output)
    print(prior.canonical({k:v for k,v in result.items() if k not in ('results','context_equalities')}),flush=True)
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
