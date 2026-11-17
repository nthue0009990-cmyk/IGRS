"""Pure whole-load composition repairs from one reliable failed complete plan.

No oracle is accepted. A caller must bind the parent record to its causal paid
history. This module checks its graph/state identity, never grants a certificate.
"""
from copy import deepcopy
import math

from .nature34_clpu_search import graph_plan

VERSION = 'clpu-support-load-composition-repair-v1'
ORIGIN = 'support_load_repair'
CONFIG = {
    'max_generation_calls_per_fork': 2,
    'round1_trigger': 'declared_parent_reliable_FAIL_once',
    'round2_trigger': 'first_reliable_FAIL_of_round1_new_candidate_once',
    'anchor': 'largest_original_eligible_whole_load',
    'round1_supports': 'original_nominal_order_positions_3_and_4_if_energized_at_failed_sample',
    'round1_edit': 'keep_anchor_replace_original_second_blocker_or_add_support_if_only_anchor',
    'round2_edit': 'keep_round1_blocked_pair_add_first_unused_original_position_3_or_later_energized_at_failure',
    'preparation_order': 'nominal_kw_descending_then_original_action',
    'release_orders': ['SMALL_FIRST', 'LARGE_FIRST'],
    'timings': ['NATURAL', 'ONE_TAU'],
    'gap_slots': '1_or_ceil_original_event_tau_over_slot_seconds',
    'variants': 'HOLD_once_then_all_nonempty_release_prefixes_for_both_orders_and_timings',
    'raw_proposal_limits': {'1': 18, '2': 13},
    'horizon': 30, 'slot_seconds': 60, 'sample_seconds': 30,
    'unavailable_horizon': 'retain_record_never_truncate_actions',
    'candidate_identity': 'complete_actions_tuple',
    'objective_pruning': 'none_in_generator;external_certified_floor_applied_by_common_controller',
    'historical_overlap': 'diagnostic_only_not_online_certificate_or_exclusion',
    'pending': 'retain_all;shared_exposure_at_most32_by_original_full_graph_priority_then_stable_id',
    'coverage': 'odd_verify_prefers_held_out_strict_graph_improvements_then_same_priority32;hidden_ids_logged',
    'new_physical_parameters': False,
}


def _prefix(actions):
    last=max((i for i,a in enumerate(actions) if a),default=-1)
    return list(actions[:last+1])


def _eligible(env,route):
    mask=env.initial.command>>10
    for action in route:
        if type(action) is not int or not 11<=action<=19:
            raise ValueError('Original switch-only route required')
        mask ^= 1 << (action-11)
    terminal=(mask<<10)|(env.initial.command&1023)
    target=set(env.energized(terminal))
    rows=[{'action':i+1,'load_name':name,'nominal_kw':float(env.loads[name]['nominal_kw'])}
          for i,name in enumerate(env.static['dispatch_loads'])
          if name in target and name not in env.initial_energized and env.initial.command&(1<<i)]
    return sorted(rows,key=lambda r:(-r['nominal_kw'],r['action']))


def generate_support_repair_candidates(env,parent_record,parent_metadata,base_generation,generation_round):
    if (env.horizon,env.slot_seconds,env.sample_seconds,env.initial.step)!=(30,60,30,0):
        raise ValueError('Original 30/60/30 complete initial-state scope required')
    if type(generation_round) is not int or generation_round not in (1,2):
        raise ValueError('Exactly round1 or round2 is permitted')
    validation=parent_record.get('validation',{})
    samples=validation.get('samples',[])
    if parent_record.get('status')!='FAIL' or validation.get('status')!='FAIL' or not samples or samples[-1].get('status')!='FAIL':
        raise ValueError('Only an actually validated reliable FAIL may trigger repair')
    actions=parent_record.get('actions')
    if actions!=validation.get('actions') or actions!=parent_metadata.get('actions'):
        raise ValueError('Failed plan and generating metadata differ')
    info=graph_plan(env,env.initial,actions)
    failed=validation.get('failed_step')
    if type(failed) is not int or not 0<=failed<30:
        raise ValueError('Invalid complete-plan failed step')
    state=info['states'][failed+1].to_dict();last=samples[-1]
    if state!=last.get('state') or state!=validation.get('failed_state') or last.get('step')!=failed+1:
        raise ValueError('Reliable failure is not the required graph state')
    failure_time=last['time_seconds']
    if failure_time!=validation.get('failed_time_seconds') or failure_time not in (failed*60,failed*60+30,failed*60+60):
        raise ValueError('Failure time is not an original required sample')
    if not validation.get('feedback'):
        raise ValueError('Reliable failure feedback required')
    route=list(parent_metadata['switch_route'])
    route_row=next((r for r in base_generation['routes'] if r.get('selected') and
                   r['route_rank']==parent_metadata['route_rank']),None)
    if route_row is None or route_row['status']!='SELECTED' or route!=route_row['switch_route']:
        raise ValueError('Original shared selected route changed')
    eligible=_eligible(env,route)
    if eligible!=route_row['eligible_whole_loads']:
        raise ValueError('Original eligible whole-load identities changed')
    tau=float(env.event['tau_seconds'])
    if not math.isfinite(tau) or tau<=0 or tau!=base_generation['event_tau_seconds']:
        raise ValueError('Original event time constant changed')
    energized=set(env.energized(state['command']))
    parent_blocked=parent_metadata['blocked_loads']
    known={x['action']:x for x in eligible}
    if any(x!=known.get(x['action']) for x in parent_blocked):
        raise ValueError('Parent blockers are not original eligible whole loads')
    proposals,candidates,seen,sets,selection=[],[],{},[],[]
    if not eligible or eligible[0]['action'] not in {x['action'] for x in parent_blocked}:
        availability='UNAVAILABLE_MAXIMUM_ANCHOR_NOT_BLOCKED'
    else:
        anchor=eligible[0]
        if generation_round==1:
            if parent_metadata.get('origin')!='staged_pickup' or len(parent_blocked) not in (1,2):
                raise ValueError('Round1 requires an original staged failed parent')
            for support in eligible[2:4]:
                enabled=support['load_name'] in energized
                selection.append({'support':deepcopy(support),'eligible_at_failure':enabled})
                if enabled:sets.append([anchor,support])
        else:
            if parent_metadata.get('origin')!=ORIGIN or parent_metadata.get('repair_round')!=1 or len(parent_blocked)!=2:
                raise ValueError('Round2 requires a round1 failed pair; no third generation')
            used={x['action'] for x in parent_blocked}
            for support in eligible[2:]:
                enabled=support['action'] not in used and support['load_name'] in energized
                selection.append({'support':deepcopy(support),'eligible_at_failure':enabled})
                if enabled:
                    sets.append(list(parent_blocked)+[support]);break
        availability='AVAILABLE' if sets else 'UNAVAILABLE_NEW_ENERGIZED_SUPPORT'
    for blocked in sets:
        blocked=sorted(blocked,key=lambda x:(-x['nominal_kw'],x['action']))
        variants=[('HOLD',[],None)]
        for order_name,order in [('SMALL_FIRST',sorted(blocked,key=lambda x:(x['nominal_kw'],x['action']))),('LARGE_FIRST',blocked)]:
            for count in range(1,len(blocked)+1):
                for timing in ('NATURAL','ONE_TAU'):
                    variants.append((order_name,order[:count],timing))
        for order_name,released,timing in variants:
            gap=None if timing is None else (1 if timing=='NATURAL' else math.ceil(tau/env.slot_seconds))
            operations=[x['action'] for x in blocked]+route
            for load in released:operations += [0]*(gap-1)+[load['action']]
            row={'proposal_id':len(proposals),'origin':ORIGIN,'repair_round':generation_round,
                'route_rank':parent_metadata['route_rank'],'switch_route':route,
                'blocked_loads':deepcopy(blocked),'released_loads':deepcopy(released),
                'held_out_loads':[x['load_name'] for x in blocked if x not in released],
                'restore_order':order_name,'timing':timing,'gap_slots':gap,'tau_seconds':tau,
                'operation_prefix':operations,'planned_prefix_length':len(operations),'actions':None,
                'physical_status':'UNVERIFIED','quality_pct':None}
            proposals.append(row)
            if len(operations)>30:
                row['status']='UNAVAILABLE_HORIZON';continue
            actions=tuple(operations+[0]*(30-len(operations)))
            row.update(actions=list(actions),prefix=_prefix(actions))
            try:assessment=graph_plan(env,env.initial,actions)
            except ValueError as exc:
                row.update(status='GRAPH_ILLEGAL',error=str(exc));continue
            row.update(graph_energy_kwh=assessment['energy_kwh'],graph_final_kw=assessment['final_kw'],operations=assessment['operations'])
            if actions in seen:
                row.update(status='DUPLICATE',duplicate_of=seen[actions])
                candidates[seen[actions]]['alias_proposal_ids'].append(row['proposal_id']);continue
            row.update(status='LEGAL_UNIQUE',candidate_id=len(candidates));seen[actions]=row['candidate_id']
            candidates.append({**deepcopy(row),'alias_proposal_ids':[row['proposal_id']]})
    if len(proposals)>CONFIG['raw_proposal_limits'][str(generation_round)]:
        raise AssertionError('Fixed finite repair-generation bound exceeded')
    return {'version':VERSION,'config':deepcopy(CONFIG),'repair_round':generation_round,'availability':availability,
        'parent_actions':list(parent_record['actions']),'parent_failed_step':failed,'parent_failed_time_seconds':failure_time,
        'parent_failed_state':state,'original_eligible_whole_loads':eligible,'support_selection':selection,
        'proposals':proposals,'candidates':candidates,'counts':{'raw_proposals':len(proposals),'unique_legal_candidates':len(candidates)},
        'new_physical_points':0,'new_training':0,'physical_certificates_created':0}
