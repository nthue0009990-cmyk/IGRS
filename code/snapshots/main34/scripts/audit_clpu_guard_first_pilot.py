"""Independent causal audit of public complete GUARD before staged search.

GUARD certificates remain in their original logical namespace. An external
floor can affect exposure and safe complete-objective pruning only after the
original search and its evidence view have actually closed.
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import FunctionType,SimpleNamespace
import argparse
import math
import sys
import time
import traceback
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
sys.dont_write_bytecode=True
import audit_clpu_staged_energy_pilot as energy

staged,shared,early,prior,branch,window,old=(energy.staged,energy.shared,energy.early,energy.prior,
    energy.branch,energy.window,energy.old)
require,close,same=energy.require,energy.close,energy.same
read,save,sha,digest,read_lines=energy.read,energy.save,energy.sha,energy.digest,energy.read_lines
VIEWS=('GUARD','EARLY')
STUDY='CLPU_GUARD_FIRST_COMMON_PILOT_V1'
SCHEMA='clpu-guard-first-protected-result-v1'
PROTOCOL='docs/clpu_guard_first_teacher_protocol_v1.md'
COMPARISON_RUN='20261031T092543_308214Z_clpu_staged_energy_pilot'
COMPARISON_SHA='9a2a6f376d11986e5483442457bdb51c5fb58d169a656cbe52c9d908bea4b2cc'
SOURCE_RUN,SOURCE_SHA=energy.SOURCE_RUN,energy.SOURCE_SHA
MODELS_RUN,MODELS_SHA=energy.MODELS_RUN,energy.MODELS_SHA
SLOTS=('guard_first_window','staged_bc17','staged_bc41','staged_bc73')
ADDITIONS={PROTOCOL,'src/dnr/nature34_clpu_guard_first_control.py',
    'scripts/run_clpu_guard_first_pilot.py','scripts/audit_clpu_guard_first_pilot.py',
    'tests/test_nature34_clpu_guard_first_control.py','tests/test_run_clpu_guard_first_pilot.py',
    'tests/test_clpu_guard_first_audit.py'}
FLOOR_VERSION='clpu-public-guard-first-certified-floor-v1'
FLOOR_RULE={
    'phase_order':['GUARD','EARLY'],
    'guard':'unchanged_original_BranchSearch_beam_completed_before_EARLY',
    'floor':'external_GUARD_complete_certificate_not_an_EARLY_logical_certificate',
    'objective':'original_graph_plan_complete_priority_tuple_exact_comparison',
    'prune':'pending_complete_priority_greater_than_or_equal_to_effective_certified_priority',
    'pruning_cost':'zero_VERIFY_choices_zero_oracle_requests',
    'pruning_feasibility':'unknown_unless_already_certified_not_a_FAIL_label',
    'exposure':'unchanged_staged_pending_window_after_public_safe_pruning',
    'feedback':'EARLY_own_validation_history_only_GUARD_answers_available_via_shared_service',
    'final_merge':'strict_energy_improvement_select_EARLY_exact_energy_tie_select_GUARD'}


def private_function(function,**bindings):
    value=FunctionType(function.__code__,{**function.__globals__,**bindings},function.__name__,
        function.__defaults__,function.__closure__)
    value.__kwdefaults__=function.__kwdefaults__
    return value


check_service_ledger=private_function(shared.check_service_ledger,VIEWS=VIEWS)
_service_summaries=private_function(shared.check_service_summaries,VIEWS=VIEWS)
check_times=private_function(shared.check_times,VIEWS=VIEWS)


def check_service_summaries(summary,view_summaries,checked,scope,mode,numerical):
    actual=deepcopy(summary)
    require(actual.pop('guard_first_version')==FLOOR_VERSION and actual.pop('phase_order')==list(VIEWS),
        'Wrong causal service version/order')
    return _service_summaries(actual,view_summaries,checked,scope,mode,numerical)


def check_checkpoints(checkpoints,phase_checks,phase_results):
    # Independently validate the new persistent promise. Only then project
    # its label for the old unchanged cumulative cost/quality/clock checker.
    established=False;close_count=0;projected=deepcopy(checkpoints)
    for actual,value in zip(checkpoints,projected):
        name,kind=actual['phase'],actual['kind']
        require(name in VIEWS,'Unknown search phase')
        if name=='EARLY':
            require(established and close_count==1,'EARLY checkpoint precedes completed GUARD')
        if name=='GUARD':
            require(not established,'GUARD continued or restarted after closure')
            if kind=='CLOSE':
                require(phase_results['GUARD']['done'] is True and phase_results['GUARD']['fully_certified'] is True,
                    'Unfinished GUARD cannot establish a persistent floor')
                established=True;close_count+=1
        require(actual['guard_established'] is established,'Premature or withdrawn certified floor')
        value['guard_established']=name=='GUARD' and kind=='CLOSE' and phase_results['GUARD']['fully_certified']
    require(established and close_count==1,'No completed GUARD boundary')
    return private_function(shared.check_checkpoints,VIEWS=VIEWS)(projected,phase_checks,phase_results)


def check_selection(result,phases):
    require(result['phase_order']==list(VIEWS) and result['intermediate_beam_floor_claimed'] is True
        and result['guard_first']==phases['EARLY']['guard_first'],'Missing explicit causal floor contract')
    # The old final-energy/tie check is unchanged. Its old promise flag is
    # adapted only after the new promise and actual GUARD chronology checks.
    return shared.check_selection({**result,'intermediate_beam_floor_claimed':False},phases)


class GuardFirstReplay(energy.EnergyReplay):
    def __init__(self,*args,guard_result,guard_reference):
        mech=args[0]
        require(guard_result['done'] is True and guard_result['fully_certified'] is True,
            'External floor requires a completed certified GUARD')
        plan=guard_result['best_plan']
        require(plan and plan['certified_complete'] and plan['validation']['status']=='COMPLETE',
            'External floor cannot be a graph-only or partial plan')
        self.guard_floor=deepcopy(plan)
        self.guard_floor_key=mech.plan(plan['actions'])['priority']
        require(math.isclose(-self.guard_floor_key[0],plan['validation']['energy_kwh'],rel_tol=1e-12,abs_tol=1e-10),
            'GUARD objective differs from the current unchanged problem')
        require(guard_reference['phase']=='GUARD' and guard_reference['file']=='GUARD/phase_result.json'
            and len(guard_reference['sha256'])==64,'External floor reference is missing')
        self.guard_reference=deepcopy(guard_reference);self.free_pruning=[]
        super().__init__(*args)

    @property
    def effective_priority(self):
        own=self.m.plan(self.best['actions'])['priority'] if self.best else None
        return own if own is not None and own<self.guard_floor_key else self.guard_floor_key

    @property
    def effective_incumbent(self):
        own=self.m.plan(self.best['actions'])['priority'] if self.best else None
        return self.best if own is not None and own<self.guard_floor_key else self.guard_floor

    def refresh(self):
        if not self.done:
            floor=self.effective_priority
            source='EARLY' if self.effective_incumbent is self.best else 'GUARD'
            before=(self.oracle.used,self.oracle.counts()['requests'],self.actual(),self.verify_attempts)
            for plan,bid in list(self.queue.items()):
                node=self.frontier[bid]
                if node['info']['priority']>=floor:
                    del self.queue[plan];self.skipped.add(plan)
                    self.free_pruning.append({'branch_id':bid,'actions':list(plan),
                        'status':'PRUNED_CERTIFIED_COMPLETE_OBJECTIVE','physical_feasibility':'UNASSESSED',
                        'candidate_priority':deepcopy(node['info']['priority']),'certified_priority':deepcopy(floor),
                        'incumbent_source_phase':source,'completed_verify_attempts':self.verify_attempts,
                        'logical_points':self.oracle.used,'actual_physical_points':self.actual(),
                        'consumed_verification_choice':False})
            require(before==(self.oracle.used,self.oracle.counts()['requests'],self.actual(),self.verify_attempts),
                'Free public pruning consumed hidden evidence/choices')
        super().refresh()

    def verification_view(self):
        view,aliases=super().verification_view()
        view.best=self.effective_incumbent
        return view,aliases

    def scientific_snapshot(self):
        result=super().scientific_snapshot()
        result.update(guard_established=True,complete_guard_still_required=False)
        result['guard_first']={'version':FLOOR_VERSION,'rule':deepcopy(FLOOR_RULE),
            'guard_reference':deepcopy(self.guard_reference),'external_guard_plan':deepcopy(self.guard_floor),
            'external_guard_priority':deepcopy(self.guard_floor_key),
            'effective_incumbent_source_phase':'EARLY' if self.effective_incumbent is self.best else 'GUARD',
            'effective_incumbent_quality_pct':100.*self.effective_incumbent['validation']['energy_kwh']/self.m.denominator,
            'own_best_plan_namespace':'EARLY','free_pruning':deepcopy(self.free_pruning),
            'free_pruned_complete_plans':len(self.free_pruning)}
        return result


def adapter_metadata(replay):
    value=staged.adapter_metadata(replay)
    value.update(guard_first_version=FLOOR_VERSION,
        incumbent_features='best_of_external_completed_GUARD_and_EARLY_own_certificate_by_original_priority',
        dominated_pending='free_public_objective_pruning_before_window_without_VERIFY_or_physical_query')
    return value


_result_check=private_function(staged.StagedMacroReplay.check_control_result,adapter_metadata=adapter_metadata)


class GuardFirstMacroReplay(energy.EnergyMacroReplay):
    def check_control_result(self,result):
        actual=deepcopy(result)
        for key,value in energy.selector_metadata().items():
            require(actual['control'].pop(key)==value,'Complete-energy selector metadata changed')
        require(actual['control'].pop('active_selector')==('complete_graph_energy' if self.method=='beam' else 'fixed_raw_actor'),
            'Claimed rule differs from actual selected action')
        require(actual['control'].pop('guard_first_version')==FLOOR_VERSION
            and actual['control'].pop('handoff_semantics')=='EARLY_termination_after_already_completed_GUARD_no_second_guard_run'
            and actual['control']['guard_established'] is True and actual['control']['complete_guard_still_required'] is False,
            'Premature floor claim or hidden second GUARD')
        # The old controller checker still verifies every unchanged field. Only
        # its explicit old promise labels are projected after the new causal
        # promise has been verified above. Scientific result keeps True floor.
        actual['control'].update(guard_established=False,complete_guard_still_required=True)
        return _result_check(self,actual)


def check_early_records(mech,cfg,spec,result,attempts,decisions,evidence,condition_sha,guard_result,guard_reference,policy=None):
    class BoundReplay(GuardFirstReplay):
        def __init__(self,*args):
            super().__init__(*args,guard_result=guard_result,guard_reference=guard_reference)
    records=private_function(staged._check_records,DeferredReplay=BoundReplay,MacroReplay=GuardFirstMacroReplay)
    checked=records(mech,cfg,spec,result,attempts,decisions,evidence,condition_sha,policy)
    require(checked['graph_expansions']==0 and checked['verify_attempts']<=16 and checked['total_decisions']<=17,
        'Public floor changed the common staged budget')
    checked.update(candidate_pool_sha256=digest(result['stage_generation']),stage_generation_verified=True,
        free_pruned_complete_plans=len(result['guard_first']['free_pruning']),
        external_guard_reference=deepcopy(guard_reference))
    return checked


def check_cell(run,spec,cfg,scope):
    cell=run/spec['cell'];guard_verified=False
    donor=read(cell/'GUARD/phase_result.json')
    reference={'phase':'GUARD','file':'GUARD/phase_result.json','sha256':sha(cell/'GUARD/phase_result.json')}

    def bound_read(path):
        path=Path(path);value=read(path)
        if path==cell/'result.json':
            require(value['version']==value['study']==STUDY and value['schema']==SCHEMA,
                'New GUARD-first result mislabeled or substituted')
            value={**value,'schema':'clpu-staged-protected-result-v1'}
        return value

    def guard_records(*args):
        nonlocal guard_verified
        require(not guard_verified and args[2]=='GUARD','Original complete GUARD must be replayed once first')
        checked=shared.check_phase_records(*args)
        require(args[4]==donor and donor['done'] is True and donor['fully_certified'] is True,
            'Donor is not the independently replayed complete GUARD')
        guard_verified=True
        return checked

    def early_records(mech,config,ps,result,attempts,decisions,evidence,condition_sha,policy=None):
        require(guard_verified,'External floor consumed before independent GUARD replay')
        require(result['guard_first']['guard_reference']==reference
            and result['guard_first']['external_guard_plan']==donor['best_plan'],
            'External floor cites different bytes or another logical namespace')
        return check_early_records(mech,config,ps,result,attempts,decisions,evidence,condition_sha,
            donor,reference,policy)

    facade=SimpleNamespace(**{**vars(shared),'VIEWS':VIEWS,'check_service_ledger':check_service_ledger,
        'check_service_summaries':check_service_summaries,'check_phase_records':guard_records,
        'check_checkpoints':check_checkpoints,'check_selection':check_selection,'check_times':check_times})
    base=private_function(staged._check_base_cell,read=bound_read,STUDY=STUDY,shared=facade,
        check_staged_records=early_records)
    outer=private_function(staged.check_cell,_check_base_cell=base,shared=facade)
    checked=outer(run,spec,cfg,scope)
    actual=read(cell/'EARLY/phase_result.json')
    checked.update(phase_order=list(VIEWS),external_guard_reference=reference,
        causal_guard_first_verified=guard_verified,
        free_pruned_complete_plans=actual['guard_first']['free_pruned_complete_plans'],
        external_floor_namespace_verified=True,pruning_creates_no_physical_label=True)
    return checked


def protocol_settings():
    value=staged.protocol_settings()
    del value['conditional_stages'];del value['no_gain_remaining_status']
    value.update(slots=list(SLOTS),execution='single_stage_all32',
        internal_execution_journal='controller_stage1_* names retained for one all32 pass only',
        quality_gate=False,ranking_change='unchanged complete-energy selector and fixed BC; common external certified-floor pruning',
        comparison='only current cold32 process times; previous EARLY-first energy run is a quality/point reference; no cross-date time ratio',
        same_optimized_window_for_all_rankers=True,phases=list(VIEWS),phase_order=list(VIEWS),
        certified_floor='external complete GUARD certificate; EARLY own best_plan remains local',
        free_pruning='original complete priority >= effective certified priority; zero VERIFY and oracle requests',
        inherited_feedback='EARLY own feedback only; GUARD paid reliable evidence queried through shared service')
    return value


expected_schedule=private_function(staged.expected_schedule,SLOTS=SLOTS)
expected_phase_spec=staged.expected_phase_spec
normalized_manifest=energy.normalized_manifest


def check_comparison(run):
    require(read(run/'comparison_source_binding.json')=={'source_run':COMPARISON_RUN,
        'source_completion_sha256':COMPARISON_SHA,
        'role':'matched prior candidate/scientific reference; no historical answers or cross-date time denominator'},
        'Historical scientific attribution changed')
    require(sha(run/'comparison_source_completion_manifest.json')==COMPARISON_SHA,'Prior energy root seal changed')
    sealed=normalized_manifest(read(run/'comparison_source_completion_manifest.json'))
    for name in energy.COMPARISON_FILES:
        require(sha(run/'comparison_source'/name)==sealed[name],'Changed prior small evidence: '+name)
    summary=read(run/'comparison_source/audit/summary.json')
    require(summary['status']=='PASS' and summary['audited_cells']==summary['resolved_slots']==32
        and summary['complete_protected_cells']==32 and not summary['errors'],'Prior complete independent audit missing')
    require(read(run/'comparison_source/contexts.json')==read(run/'contexts.json'),'Original observed contexts changed')
    original=read(run/'comparison_source/stage_plan.json');plan=read(run/'stage_plan.json')
    require(plan=={'schema':'clpu-guard-first-fixed-population-v1','original_contexts':8,'scheduled_cells':32,
        'execution':'single_stage_all32','quality_gate':False,'all_cells':[s['cell'] for s in read(run/'schedule.json')],
        'eligible_contexts':original['eligible_contexts'],
        'eligible_contexts_role':'historical graph diagnostic only; does not gate dispatch',
        'contexts':original['contexts'],'new_physical_points':0,'new_training':0},'Changed pool/population or hidden gate')
    staged.check_graph_profile(run,plan)
    return sealed


def check_parent(run):
    run=Path(run).resolve()
    require(run.parent==ROOT/'runs','New pilot outside project runs')
    prepared=read(run/'prepared.json')
    require(prepared['status']=='FROZEN_PREPARED_NO_DSS' and prepared['freeze_sha256']==sha(run/'prephysical_freeze_manifest.json'),
        'Prephysical freeze changed')
    frozen=shared.checked_manifest(run,'prephysical_freeze_manifest.json')
    require(sha(run/'source_completion_manifest.json')==SOURCE_SHA,'Original science root changed')
    source=ROOT/'runs'/SOURCE_RUN
    require(sha(source/'completion_manifest.json')==SOURCE_SHA,'Original source seal differs')
    source_manifest=normalized_manifest(read(run/'source_completion_manifest.json'))
    require(read(run/'source_binding.json')=={'source_run':SOURCE_RUN,'source_completion_sha256':SOURCE_SHA,
        'models_run':MODELS_RUN,'models_completion_sha256':MODELS_SHA},'Science/model attribution changed')
    for name in ('config.json','schedule.json','prepared.json','audit/summary.json','contexts.json'):
        require(sha(source/name)==source_manifest[name],'Changed inherited input: '+name)
    original_audit=read(source/'audit/summary.json')
    require(original_audit['status']=='PASS' and original_audit['audited_cells']==64 and not original_audit['errors'],
        'Original independent audit absent')
    previous_manifest=check_comparison(run)
    inherited=read(run/'comparison_source/prepared.json')['source_files']
    require(prepared['source_files']==sorted(set(inherited)|ADDITIONS),'Required source closure changed')
    for name in prepared['source_files']:
        path=(ROOT/name).resolve()
        require(path.is_relative_to(ROOT) and path!=ROOT and 'source/'+name in frozen
            and sha(path)==sha(run/'source'/name),'Changed frozen source: '+name)
        if name in inherited:
            require(sha(path)==previous_manifest['source/'+name],'Changed previously sealed science: '+name)
    require(sha(run/'protocol.md')==sha(ROOT/PROTOCOL),'Frozen protocol differs')
    cfg=read(run/'config.json')
    require(cfg=={**read(source/'config.json'),'staged_study':STUDY,'staged_pilot':protocol_settings()},
        'Unapproved scientific/configuration change')
    scope=prior.check_frozen_physics(run,prepared,cfg)
    models=staged.check_models(run);contexts=read(source/'contexts.json')
    require(read(run/'contexts.json')==contexts,'Observed training population changed')
    schedule=read(run/'schedule.json')
    require(schedule==expected_schedule(contexts,models),'All32 fixed population/Latin order changed')
    for spec in schedule:
        require(read(run/spec['cell']/'spec.json')==spec,'Cold task spec differs')
        for phase in VIEWS:
            require(read(run/spec['cell']/phase/'spec.json')==expected_phase_spec(spec,phase),'Logical view spec differs')
    chosen=[s for s in read(source/'schedule.json') if s['arm']=='deferred_window']
    expected_history=[]
    for spec in chosen:
        original=spec['cell']+'/GUARD/phase_result.json'
        target=f'historical/c{spec["context_id"]:02d}/GUARD/phase_result.json'
        require(sha(run/target)==source_manifest[original],'Original full GUARD changed')
        expected_history.append({**{k:spec[k] for k in ('context_id','scenario_id','event_id')},'source_cell':spec['cell'],
            'phases':{'GUARD':{'result_file':target,'source_file':original,'sha256':source_manifest[original]}}})
    require(read(run/'historical_bindings.json')==expected_history,'Original GUARD attribution differs')
    cases=list(ET.parse(run/'software_tests.xml').getroot().iter('testcase'))
    require(cases and not any(list(c.iter('failure'))+list(c.iter('error'))+list(c.iter('skipped')) for c in cases),
        'Complete passing focused software evidence required')
    for name in ('test_nature34_clpu_guard_first_control','test_run_clpu_guard_first_pilot','test_clpu_guard_first_audit'):
        require(any(name in c.get('classname','') for c in cases),'Missing new component: '+name)
    resource=read(run/'resource_budget.json')
    for key,value in {'total_physical_cap':16384,'minimum_free_bytes':12*2**30,'report_reserve_bytes':2*2**30,
        'successful_component_tests':len(cases)}.items():
        require(resource[key]==value,'Resource cap changed: '+key)
    rate=max(sum(p.stat().st_size for p in (source/s['cell']).rglob('*') if p.is_file())/
        max(1,read(source/s['cell']/'result.json')['actual_physical_points']) for s in chosen)
    require(resource['historical_max_bytes_per_actual_point']==rate and resource['forecast_bytes']==math.ceil(rate*16384*1.25)
        and resource['required_free_bytes']==14*2**30+resource['forecast_bytes']
        and resource['free_bytes_at_prepare']>=resource['required_free_bytes'],'Incomplete cold storage forecast')
    require(read(run/'authorization.json')['execution_authorized'] is True,'Missing authorization')
    return cfg,scope,schedule


check_controller=energy.check_controller
context_equalities=private_function(staged.context_equalities,SLOTS=SLOTS)


def source_hashes(run,scope):
    value=staged.source_hashes(run,scope)
    value.update(audit_implementation_sha256=sha(Path(__file__)),core_audit_sha256=sha(Path(staged.__file__)),
        ranking_audit_sha256=sha(Path(energy.__file__)),
        comparison_source_binding_sha256=sha(run/'comparison_source_binding.json'),
        comparison_source_completion_sha256=COMPARISON_SHA)
    return value


# The lifecycle, physical-import guard, full32 accounting and output seal are
# unchanged; bind the new independent entry points in a private namespace.
audit=private_function(energy.audit,check_parent=check_parent,check_cell=check_cell,
    check_controller=check_controller,context_equalities=context_equalities,source_hashes=source_hashes)
preflight=private_function(energy.preflight,check_parent=check_parent,source_hashes=source_hashes)
main=private_function(energy.main,audit=audit,preflight=preflight,__doc__=__doc__)


if __name__=='__main__':raise SystemExit(main())
