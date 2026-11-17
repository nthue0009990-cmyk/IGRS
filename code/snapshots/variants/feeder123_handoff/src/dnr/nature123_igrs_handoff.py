"""V2: retain generated GUARD holds across its bounded phase handoff.

This revision does not modify GUARD, widen its expansion budget, or seed paths.
Carryover and the unchanged staged generator share sixteen VERIFY opportunities.
The caller still owns one cold oracle, exact-request cache and all physical caps.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import math

from .nature123_igrs import (SearchEnvironment, BranchConfig, BranchSearch, _PhaseView,
    GraphBudgetExhausted, CLPUBudgetExhausted, CLPUReliabilityError)
from .nature123_igrs_search import graph_plan
from .nature123_igrs_staged_pickup import generate_staged_pickup_candidates
from .nature123_igrs_staged_energy_control import complete_energy_action

VERSION='NC123_SPARSE_IGRS_PREFIX_WAIT_HANDOFF_V2'
VERIFY_LIMIT=16


def carryover_candidates(guard):
    """Only actual unverified, profitable prefix-WAIT frontier plans qualify."""
    if not guard.done or guard.incumbent is None:
        raise ValueError('Completed GUARD with certified incumbent required')
    floor=guard.incumbent['validation']['energy_kwh']; unique={}; inventory=[]
    for node in sorted(guard._frontier.values(),key=lambda n:n['id']):
        info=node['info'];actions=tuple(info['actions'])
        reason=('OTHER_ORIGIN' if node['origin']!='prefix_wait' else
            'ALREADY_CHECKED' if actions in guard._checked else
            'NO_STRICT_ENERGY_GAIN' if info['energy_kwh']<=floor else 'ELIGIBLE')
        row={'source_branch_id':node['id'],'source_parent_id':node['parent_id'],
            'source_prefix':list(node['prefix']),'source_origin':node['origin'],
            'actions':list(actions),'graph_energy_kwh':info['energy_kwh'],
            'priority':deepcopy(info['priority']),'eligibility':reason}
        inventory.append(row)
        if reason=='ELIGIBLE':
            item=unique.setdefault(actions,{'actions':list(actions),'priority':deepcopy(info['priority']),
                'graph_energy_kwh':info['energy_kwh'],'source_branch_ids':[],
                'sources':[],'status':'PENDING','physical_feasibility':'UNASSESSED'})
            item['source_branch_ids'].append(node['id']);item['sources'].append(deepcopy(row))
    candidates=sorted(unique.values(),key=lambda n:n['priority'])
    for rank,item in enumerate(candidates):item['rank']=rank
    return {'incumbent_energy_kwh':floor,'inventory':inventory,'candidates':candidates,
        'rule':'prefix_wait AND unchecked_complete_actions AND strict_energy_gain; deduplicate_actions; original_priority',
        'physical_queries_during_selection':0}


def staged_choice(pending, incumbent_info, staged_attempts):
    """The original free priority screen and alternating held-load window."""
    pruned=[r for r in pending if r['info']['priority']>=incumbent_info['priority']]
    eligible=[r for r in pending if r['info']['priority']<incumbent_info['priority']]
    eligible.sort(key=lambda r:r['branch_id'])
    window=eligible
    if staged_attempts%2==1:
        held=[r for r in eligible if r['info']['energy_kwh']>incumbent_info['energy_kwh']
              and r['metadata']['held_out_loads']]
        if held:window=held
    if not window:return None,pruned
    observation={'branches':[{'branch_id':r['branch_id'],'graph_energy_kwh':r['info']['energy_kwh']} for r in window],
        'branch_ids':[r['branch_id'] for r in window],'mask':[True]*len(window)}
    return window[complete_energy_action(observation)],pruned


def run_search(env,oracle,config=None,record_callback=None,event_callback=None):
    if not isinstance(env,SearchEnvironment):raise TypeError('SearchEnvironment required')
    if any(getattr(oracle,k)!=0 for k in ('points','physical_points','requests')):
        raise ValueError('Handoff task requires a cold private oracle')
    cfg=BranchConfig(**config) if isinstance(config,dict) else config
    cfg=cfg or BranchConfig(point_budget=512,max_branch_expansions=32,max_checked_candidates=64)
    if cfg.point_budget!=oracle.budget:raise ValueError('Task logical allowance/config mismatch')
    records=[];stages={};phase='GUARD';baseline=None;best=None;guard=None;stop=None
    emit=event_callback or (lambda row:None)
    handoff={'verify_limit':VERIFY_LIMIT,'verify_used':0,'carryover_verify_used':0,'staged_verify_used':0,
        'selection':None,'staged_generation':None,'staged_candidates':[],'free_pruning':[],
        'extra_wait_validation':False}
    def record(row):
        nonlocal baseline,best
        row=deepcopy(row);row['phase']=phase;records.append(row)
        if phase=='GUARD' and row.get('origin')=='wait_incumbent' and row.get('status')=='COMPLETE':
            baseline=row['validation']['energy_kwh']
        if row.get('certified_complete') is True:
            if best is None or row['validation']['energy_kwh']>best['validation']['energy_kwh']:
                best=deepcopy(row)
        if record_callback:record_callback(deepcopy(row))
    def gained():return best is not None and baseline is not None and best['validation']['energy_kwh']>baseline
    def verify(item,info,origin):
        before=oracle.points;physical_before=oracle.physical_points
        handoff['verify_used']+=1
        key='carryover_verify_used' if origin=='guard_prefix_wait_carryover' else 'staged_verify_used'
        handoff[key]+=1;item['verify_ordinal']=handoff['verify_used'];item['status']='VERIFY_STARTED'
        emit({'event':'HANDOFF_VERIFY_STARTED','phase':phase,'pool':origin,
            'verify_ordinal':handoff['verify_used'],'actions':item['actions']})
        details={'shared_verify_ordinal':handoff['verify_used'],'pool':origin,
            'source_branch_ids':item.get('source_branch_ids',[]),'rank':item.get('rank')}
        try:validation=env.validate_plan(env.initial,item['actions'],oracle)
        except CLPUBudgetExhausted:
            item['status']='BUDGET_EXHAUSTED_PARTIAL'
            record({'actions':item['actions'],'origin':origin,'details':details,
                'status':item['status'],'certified_complete':False,
                'physical_points_before':before,'physical_points_after':oracle.points})
            raise
        if validation['status'] not in ('COMPLETE','FAIL'):
            raise CLPUReliabilityError('Unexpected full-plan status')
        if validation['status']=='COMPLETE' and not math.isclose(validation['energy_kwh'],info['energy_kwh'],rel_tol=1e-12,abs_tol=1e-10):
            raise CLPUReliabilityError('Graph and certified service disagree')
        item.update(status=validation['status'],physical_feasibility=validation['status'],
            logical_points=oracle.points-before,actual_physical_points=oracle.physical_points-physical_before)
        record({'actions':item['actions'],'origin':origin,'details':details,'status':validation['status'],
            'certified_complete':validation['status']=='COMPLETE','validation':validation,
            'physical_points_before':before,'physical_points_after':oracle.points})
    try:
        view=_PhaseView(oracle,env);emit({'event':'PHASE_STARTED','phase':'GUARD','logical_offset':oracle.points})
        guard=BranchSearch(env,view,replace(cfg,point_budget=view.budget),record_callback=record)
        while not guard.done:
            event=guard.step(guard.rule_action('beam'));emit({'event':'SEARCH_STEP','phase':'GUARD',**event})
            if gained():guard._stop('FIRST_COMPLETE_GAIN_OVER_OWN_WAIT');break
        stages['GUARD']=guard.result()
        emit({'event':'PHASE_CLOSED','phase':'GUARD','stop_reason':guard.stop_reason})
        stop=guard.stop_reason
        if not gained() and guard.incumbent is not None and oracle.points<oracle.budget:
            phase='EARLY';emit({'event':'PHASE_STARTED','phase':'EARLY','logical_offset':oracle.points})
            handoff['selection']=carryover_candidates(guard)
            emit({'event':'CARRYOVER_SELECTED','phase':phase,'selection':deepcopy(handoff['selection'])})
            for item in handoff['selection']['candidates']:
                if handoff['verify_used']>=VERIFY_LIMIT or oracle.points>=oracle.budget:break
                info=graph_plan(env,env.initial,item['actions'])
                if info['priority']!=tuple(item['priority']):raise CLPUReliabilityError('Carryover graph identity changed')
                verify(item,info,'guard_prefix_wait_carryover')
                if gained():break
            if not gained() and handoff['verify_used']<VERIFY_LIMIT and oracle.points<oracle.budget:
                generation=generate_staged_pickup_candidates(env);handoff['staged_generation']=deepcopy(generation)
                pending=[]
                for index,metadata in enumerate(generation['candidates']):
                    item={'branch_id':index,'actions':list(metadata['actions']),'rank':index,
                        'status':'PENDING','physical_feasibility':'UNASSESSED'}
                    handoff['staged_candidates'].append(item)
                    pending.append({'branch_id':index,'metadata':metadata,'info':graph_plan(env,env.initial,metadata['actions']),'item':item})
                while pending and handoff['verify_used']<VERIFY_LIMIT and oracle.points<oracle.budget:
                    choice,pruned=staged_choice(pending,graph_plan(env,env.initial,best['actions']),handoff['staged_verify_used'])
                    for row in pruned:
                        row['item']['status']='PRUNED_CERTIFIED_COMPLETE_OBJECTIVE'
                        handoff['free_pruning'].append(deepcopy(row['item']));pending.remove(row)
                    if choice is None:break
                    pending.remove(choice);verify(choice['item'],choice['info'],'staged_pickup')
                    if gained():break
            stop=('FIRST_COMPLETE_GAIN_OVER_OWN_WAIT' if gained() else
                'TASK_LOGICAL_BUDGET_EXHAUSTED' if oracle.points>=oracle.budget else
                'SHARED_VERIFY_LIMIT' if handoff['verify_used']>=VERIFY_LIMIT else 'HANDOFF_AND_STAGED_POOL_EXHAUSTED')
    except (GraphBudgetExhausted,CLPUBudgetExhausted) as exc:
        stop=type(exc).__name__+': '+str(exc)
        if phase=='GUARD' and guard is not None:
            guard._stop(stop);stages['GUARD']=guard.result()
        emit({'event':'BOUNDED_STOP','phase':phase,'reason':stop})
    if phase=='EARLY':
        for item in (handoff['selection'] or {}).get('candidates',[])+handoff['staged_candidates']:
            if item['status']=='PENDING':item['status']='NOT_ATTEMPTED';item['not_attempted_reason']=stop
        stages['EARLY']={'done':True,'stop_reason':stop,'handoff':deepcopy(handoff),
            'best_plan':deepcopy(best) if best and best['phase']=='EARLY' else None}
        emit({'event':'PHASE_CLOSED','phase':'EARLY','stop_reason':stop,'handoff':deepcopy(handoff)})
    simulated=best is not None and not best['validation'].get('physical_certified',False)
    status='FIRST_COMPLETE_GAIN' if gained() else 'NO_GAIN_WITHIN_FIXED_BUDGET'
    if best is None:status='NO_CERTIFIED_COMPLETE_PLAN'
    if simulated:status='SIMULATED_'+status
    return {'version':VERSION,'status':status,'stop_reason':stop,'config':asdict(cfg),'best_plan':deepcopy(best),
        'selected_phase':best['phase'] if best else None,'stages':stages,'handoff':handoff,
        'own_wait_energy_kwh':baseline,'first_complete_gain_found':gained(),
        'physical_certified':best is not None and not simulated,'early_stage_reached':phase=='EARLY',
        'free_pruning_exercised':bool(handoff['free_pruning']),
        'graph_preparation':env.planner.summary(),'external_seed_or_physical_result_used':False,
        'scope':'ONE_DEVELOPMENT_TASK_FIRST_COMPLETE_GAIN_NOT_METHOD_COMPARISON',
        'cache_policy':'CALLER_PRIVATE_EXACT_REQUEST_HASH_NO_EXTERNAL_RESULTS',
        'logical_points':oracle.points,'actual_physical_points':oracle.physical_points,'requests':oracle.requests}
