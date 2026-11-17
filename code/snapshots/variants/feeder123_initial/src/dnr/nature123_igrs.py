"""Isolated sparse NC123 migration of the manuscript's guard-first IGRS.

Original branch selection, staged pool and free screening live in the derived
modules with an exact source/replacement receipt. No hand-built restoration
path, external certificate, saved physical answer, or solver import is used.
The caller supplies the physical oracle and process/resource supervision.
"""
from copy import deepcopy
from dataclasses import asdict, replace
from decimal import Decimal
import hashlib
import json
import math
import time

from .nature34_clpu_environment import CLPUBudgetExhausted, CLPUReliabilityError
from .nature34_clpu_early import merge_results
from .nature123_clpu_environment import RELIABILITY_CHECKS, CONSTRAINT_CHECKS
from .nature123_igrs_branch import BranchConfig, BranchSearch
from .nature123_igrs_guard_first_control import GuardFirstStagedSearch, GuardFirstMacroController


VERSION = 'NC123_SPARSE_IGRS_FIRST_CERTIFIED_GAIN_V1'
SearchConfig = BranchConfig


class GraphBudgetExhausted(RuntimeError):
    """An exact graph calculation was unfinished; no approximate path returned."""


class SparseOptimisticPlanner:
    """Memoized finite-depth Bellman recurrence with original state-path ties.

    Only visited (command, remaining depth) pairs are materialized. The score
    uses exact integer normal demand. Total demand is a safe pruning bound;
    pruning occurs only when even that bound cannot match cumulative service.
    Hitting a resource cap aborts the exact calculation, never changes its rule.
    """
    def __init__(self, graph, *, state_cap=200000, seconds=30.):
        if type(state_cap) is not int or state_cap < 1 or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Positive graph resource caps required')
        self.graph, self.state_cap, self.seconds = graph, state_cap, float(seconds)
        places=max(max(0,-p.as_tuple().exponent) for _,p in graph.loads.values())
        self.scale=10**places
        self.total=int(graph.total_kw*self.scale)
        self.weights={}; self.adjacency={}; self.memo={}; self.elapsed=0.; self.deadline=None
        self.calls=0; self.bound_pruned=0

    def _check(self):
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            raise GraphBudgetExhausted('Exact sparse lookahead cumulative time cap')

    def weight(self, command):
        self._check()
        if command not in self.weights:
            if len(self.weights) >= self.state_cap:
                raise GraphBudgetExhausted('Exact sparse lookahead distinct-command cap')
            self.weights[command]=int(self.graph.metadata(command).nominal_served_kw*self.scale)
        return self.weights[command]

    def successors(self, command):
        if command not in self.adjacency:
            self._check()
            self.adjacency[command]=tuple(sorted({r['destination'] for r in self.graph.successors(command)
                                                 if r['graph_admissible']}))
        return self.adjacency[command]

    def _solve(self, command, depth):
        self._check(); key=(command,depth)
        if key in self.memo:return self.memo[key]
        if len(self.memo) >= self.state_cap:
            raise GraphBudgetExhausted('Exact sparse lookahead memo cap')
        weight=self.weight(command)
        if depth==0: result=(0,weight,0,())
        elif weight==self.total:
            result=(depth*weight,weight,0,(command,)*depth)
        else:
            best=None; bestkey=None
            # Search order can improve exact bound pruning; final ties still
            # use complete command-state paths, just as original JointGraph.
            children=sorted(self.successors(command),key=lambda s:(-self.weight(s),s))
            for child in children:
                addition=self.weight(child)
                if best is not None and addition+(depth-1)*self.total < best[0]:
                    self.bound_pruned+=1; continue
                tail=self._solve(child,depth-1)
                proposal=(addition+tail[0],tail[1],int(child!=command)+tail[2],(child,)+tail[3])
                priority=(-proposal[0],-proposal[1],proposal[2],proposal[3])
                if bestkey is None or priority<bestkey:best,bestkey=proposal,priority
            if best is None:raise ValueError('WAIT should guarantee a graph route')
            result=best
        self.memo[key]=result
        return result

    def solve(self, command, depth):
        if type(depth) is not int or not 0<=depth<=12:raise ValueError('Original depth scope is 0..12')
        if self.elapsed>=self.seconds:raise GraphBudgetExhausted('Exact sparse lookahead cumulative time cap')
        began=time.perf_counter(); self.deadline=began+self.seconds-self.elapsed; self.calls+=1
        try:
            result=self._solve(command,depth)
            return {'status':'COMPLETE','path':[command,*result[3]],'cumulative_weight':result[0],
                    'final_weight':result[1],'operations':result[2],'weight_scale':self.scale,
                    'tie':'lexicographically_smallest_command_state_path','exact_depth_limited':True}
        finally:
            self.elapsed+=time.perf_counter()-began;self.deadline=None

    def summary(self):
        return {'distinct_commands':len(self.weights),'memo_entries':len(self.memo),'calls':self.calls,
                'seconds':self.elapsed,'state_cap':self.state_cap,'seconds_cap':self.seconds,
                'bound_pruned_children':self.bound_pruned,'complete_reachable_space_enumerated':False}


class SearchEnvironment:
    """Compatibility surface; physical certificate strictness remains NC123's."""
    def __init__(self, base_env, *, graph_state_cap=200000, graph_seconds=30, allow_simulated=False):
        self.base=base_env;self.graph=base_env.graph;self.allow_simulated=allow_simulated
        self.static={'nodes':sorted(self.graph.nodes),'bus_edges':self.graph.edges,
            'switch_edges':self.graph.switch_edges,'source_bus':self.graph.source,
            'original_lines':self.graph.original_lines,'dispatch_loads':self.graph.dispatch}
        self.loads=deepcopy(base_env.condition['loads'])
        self.scenario={'scenario_id':'NC123_'+','.join(self.graph.fault_lines),'fault_lines':list(self.graph.fault_lines)}
        self.load_multiplier=float(base_env.condition['load_multiplier'])
        self.initial_energized=self.energized(base_env.initial.command)
        self.planner=SparseOptimisticPlanner(self.graph,state_cap=graph_state_cap,seconds=graph_seconds)

    def __getattr__(self,name):return getattr(self.base,name)
    def energized(self,command):return self.graph.metadata(command).energized_loads

    def optimistic_plan(self,state,depth=6):
        self.base._validate_history(state)
        remaining=self.horizon-state.step
        result=self.planner.solve(state.command,min(depth,remaining,12))
        actions=[]
        for a,b in zip(result['path'],result['path'][1:]):
            diff=a^b
            if diff and diff&(diff-1):raise ValueError('Graph route changed multiple bits')
            actions.append(diff.bit_length())
        return actions+[0]*(remaining-len(actions))

    def _require(self, cert, state, t):
        expected=self.snapshot_request(state,t)['request_sha256']
        kind=cert.get('evidence_kind') if isinstance(cert,dict) else None
        valid=(isinstance(cert,dict) and cert.get('request_sha256')==expected and
            cert.get('point_id') is not None and cert.get('point_id')!='' and
            cert.get('reliable') is True and cert.get('status') in ('PASS','FAIL'))
        if kind=='PHYSICAL':
            checks=cert.get('checks',{})
            valid=valid and self.allow_physical and isinstance(checks,dict) and all(checks.get(k)=='PASS' for k in RELIABILITY_CHECKS)
            valid=valid and all(checks.get(k) in ('PASS','FAIL') for k in CONSTRAINT_CHECKS)
            valid=valid and (all(checks.get(k)=='PASS' for k in CONSTRAINT_CHECKS)==(cert['status']=='PASS'))
        elif kind=='SIMULATED':valid=valid and self.allow_simulated
        else:valid=False
        if not valid:raise CLPUReliabilityError('Unknown, inconsistent, or unbound NC123 snapshot')
        return cert

    def validate_plan(self,start,actions,oracle):
        if len(actions)!=self.horizon-start.step:raise ValueError('Complete remaining horizon required')
        states=self.plan_states(start,actions)
        record={'status':'IN_PROGRESS','actions':list(actions),'states':[s.to_dict() for s in states],
            'samples':[],'energy_kwh':None,'final_fraction':None,'failed_step':None,'feedback':[],
            'physical_certified':False,'normal_demand_denominator_kw':self.total_kw}
        schedule=[('INITIAL',start,start.step*self.slot_seconds,None)]
        schedule += [('SLOT',s,(s.step-1)*self.slot_seconds+off,i)
                     for i,s in enumerate(states[1:]) for off in range(0,self.slot_seconds+1,self.sample_seconds)]
        kinds=set()
        for phase,state,t,index in schedule:
            cert=self._require(oracle.certify(state,t),state,t);kinds.add(cert['evidence_kind'])
            record['samples'].append({'phase':phase,'step':state.step,'time_seconds':t,
                'state':state.to_dict(),'point_id':cert['point_id'],'request_sha256':cert['request_sha256'],
                'status':cert['status'],'evidence_kind':cert['evidence_kind']})
            if cert['status']=='FAIL':
                return {**record,'status':'FAIL','failed_phase':phase,'failed_step':index,
                    'failed_time_seconds':t,'failed_state':state.to_dict(),'feedback':deepcopy(cert.get('feedback',[])),
                    'evidence_kind':cert['evidence_kind']}
        if len(kinds)!=1:raise CLPUReliabilityError('Mixed evidence kinds')
        energy=math.fsum(self.service_kw(s)*self.slot_seconds/3600 for s in states[1:])
        return {**record,'status':'COMPLETE','energy_kwh':energy,
            'final_fraction':self.service_kw(states[-1])/self.total_kw,
            'mean_served_fraction':energy/(self.total_kw*len(actions)*self.slot_seconds/3600),
            'operations':sum(a!=0 for a in actions),'evidence_kind':next(iter(kinds)),
            'physical_certified':kinds=={'PHYSICAL'}}


class _PhaseView:
    """Local counters are derived offsets; the cold oracle is never reset."""
    def __init__(self,oracle,env):
        self.parent,self.env=oracle,env
        self.offset=oracle.points;self.physical_offset=oracle.physical_points;self.request_offset=oracle.requests
        self.budget=oracle.budget-self.offset
        if self.budget<=0:raise CLPUBudgetExhausted('No remaining task logical allowance')
    @property
    def points(self):return self.parent.points-self.offset
    @property
    def physical_points(self):return self.parent.physical_points-self.physical_offset
    @property
    def requests(self):return self.parent.requests-self.request_offset
    def certify(self,state,t):return self.env._require(self.parent.certify(state,t),state,t)


def run_search(env, oracle, config=None, record_callback=None, event_callback=None):
    """Cold guard-first IGRS; stop on the first certified gain over its own WAIT.

    config accepts BranchConfig or its dict. The outer oracle owns the common
    logical/physical cap and all cache receipts. EARLY gets only the remaining
    logical allowance. No saved-path/certificate parameter exists.
    """
    if not isinstance(env,SearchEnvironment):raise TypeError('SearchEnvironment required')
    if any(getattr(oracle,k)!=0 for k in ('points','physical_points','requests')):
        raise ValueError('IGRS task must start with an empty private oracle')
    cfg=BranchConfig(**config) if isinstance(config,dict) else config
    cfg=cfg or BranchConfig(max_branch_expansions=32,max_checked_candidates=64,point_budget=512)
    if cfg.point_budget!=oracle.budget:raise ValueError('Task logical allowance/config mismatch')
    emit=event_callback or (lambda row:None); records=[]; stages={}; baseline=None; phase='GUARD'
    def record(row):
        nonlocal baseline
        row=deepcopy(row);row['phase']=phase;records.append(row)
        if phase=='GUARD' and row.get('origin')=='wait_incumbent' and row.get('status')=='COMPLETE':
            baseline=row['validation']['energy_kwh']
        if record_callback is not None:record_callback(deepcopy(row))
    def gain(search):
        return baseline is not None and search.incumbent is not None and search.incumbent['validation']['energy_kwh']>baseline
    search=None; stop=None; selected='GUARD'
    try:
        view=_PhaseView(oracle,env)
        emit({'event':'PHASE_STARTED','phase':'GUARD','logical_offset':oracle.points})
        search=BranchSearch(env,view,replace(cfg,point_budget=view.budget),record_callback=record)
        while not search.done:
            action=search.rule_action('beam')
            event=search.step(action);emit({'event':'SEARCH_STEP','phase':'GUARD',**event})
            if gain(search):search._stop('FIRST_COMPLETE_GAIN_OVER_OWN_WAIT');break
        stages['GUARD']=search.result()
        emit({'event':'PHASE_CLOSED','phase':'GUARD','stop_reason':search.stop_reason})
        if gain(search):stop='FIRST_COMPLETE_GAIN_OVER_OWN_WAIT'
        elif search.incumbent is None:stop=search.stop_reason
        elif oracle.points>=oracle.budget:stop='TASK_LOGICAL_BUDGET_EXHAUSTED'
        else:
            phase='EARLY';search=None;view=_PhaseView(oracle,env)
            guard_sha=hashlib.sha256(json.dumps(stages['GUARD'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            emit({'event':'PHASE_STARTED','phase':'EARLY','logical_offset':oracle.points,'guard_result_sha256':guard_sha})
            search=GuardFirstStagedSearch(env,view,replace(cfg,point_budget=view.budget),record_callback=record,
                guard_result=stages['GUARD'],guard_reference={'phase':'GUARD','file':'embedded:stages.GUARD','sha256':guard_sha})
            control=GuardFirstMacroController(search,method='beam')
            while not search.done:
                proposal=control.prepare()
                if proposal is None:break
                event=control.apply(proposal);emit({'event':'SEARCH_STEP','phase':'EARLY','proposal':proposal,**event})
                if gain(search):search._stop('FIRST_COMPLETE_GAIN_OVER_OWN_WAIT');break
            stages['EARLY']=control.result()
            selected=merge_results(stages['EARLY'],stages['GUARD'],protected=True)['selected_phase']
            stop='FIRST_COMPLETE_GAIN_OVER_OWN_WAIT' if gain(search) else search.stop_reason
            emit({'event':'PHASE_CLOSED','phase':'EARLY','stop_reason':search.stop_reason})
    except (GraphBudgetExhausted,CLPUBudgetExhausted) as exc:
        stop=type(exc).__name__+': '+str(exc)
        if search is not None:
            search._stop(stop);stages[phase]=search.result()
        emit({'event':'BOUNDED_STOP','phase':phase,'reason':stop})
    complete=[r for r in records if r.get('status')=='COMPLETE' and r.get('certified_complete') is True]
    # Exact energy ties retain GUARD; all alternatives originate in this run.
    best=min(complete,key=lambda r:(-r['validation']['energy_kwh'],r['phase']!='GUARD',
        -r['validation']['final_fraction'],sum(a!=0 for a in r['actions']),tuple(r['actions']))) if complete else None
    selected=best['phase'] if best else None
    improved=best is not None and baseline is not None and best['validation']['energy_kwh']>baseline
    simulated=best is not None and not best['validation'].get('physical_certified',False)
    status='FIRST_COMPLETE_GAIN' if improved else 'NO_GAIN_WITHIN_FIXED_BUDGET'
    if best is None:status='NO_CERTIFIED_COMPLETE_PLAN'
    if simulated:status='SIMULATED_'+status
    return {'version':VERSION,'status':status,'stop_reason':stop,'config':asdict(cfg),
        'selected_phase':selected,'best_plan':deepcopy(best),'stages':stages,'own_wait_energy_kwh':baseline,
        'first_complete_gain_found':improved,'physical_certified':best is not None and not simulated,
        'early_stage_reached':'EARLY' in stages,'free_pruning_exercised':bool(stages.get('EARLY',{}).get('guard_first',{}).get('free_pruned_complete_plans',0)),
        'graph_preparation':env.planner.summary(),'external_seed_or_physical_result_used':False,
        'scope':'ONE_DEVELOPMENT_TASK_FIRST_COMPLETE_GAIN_NOT_METHOD_COMPARISON',
        'cache_policy':'CALLER_PRIVATE_EXACT_REQUEST_HASH_NO_EXTERNAL_RESULTS',
        'logical_points':oracle.points,'actual_physical_points':oracle.physical_points,'requests':oracle.requests}
