"""Cold paired development comparison on Yuan et al.'s non-loop tie-closure branch."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from datetime import timezone, datetime

ROOT=Path(__file__).resolve().parents[1]
EXTERNAL=ROOT/'artifacts/external_seq_20261106/external_yuan.py'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(EXTERNAL.parent)]
from probe_igrs_wait_compression import (BASE,BASE_SHA,read,save,sha,make_env,GuardFirstEvidenceService,
    BranchSearch,BranchConfig,GuardFirstStagedSearch,GuardFirstMacroController,merge_results)
from dnr.nature34_clpu_environment import CLPUEnvironment
from dnr.nature34_clpu_staged_pickup import ORIGIN
from nature34_experiment_support import dependencies
import external_yuan

PROTOCOL=ROOT/'docs/external_common_20261106_protocol.md'


class TieEnv(CLPUEnvironment):
    """A wait commits to ending the switching sequence; all physical data are inherited."""
    def legal_actions(self,state):
        if state.step>=self.horizon:
            return []
        if state.step!=(state.command ^ self.initial.command).bit_count():
            return [0]
        return [a for a in super().legal_actions(state) if a==0 or
                (16<=a<=19 and not state.command & (1<<(a-1)))]

    def optimistic_plan(self,state,depth=6):
        remaining=self.horizon-state.step
        length=min(depth,remaining)
        def suffix(s,k):
            if k==0: return 0,()
            options=[]
            for a in self.legal_actions(s):
                t=self.advance(s,a)
                value,rest=suffix(t,k-1) if a else ((k-1)*self.graph.served_weight(s.command),(0,)*(k-1))
                options.append((-(self.graph.served_weight(t.command)+value),t.command,(a,)+rest))
            cost,_,route=min(options)
            return -cost,route
        return list(suffix(state,length)[1]) + [0]*(remaining-length)


def env_for(spec):
    base=make_env(BASE,spec)
    return TieEnv(base.graph,base.static,base.loads,base.scenario,base.condition,base.event,
                  horizon=base.horizon,slot_seconds=base.slot_seconds,sample_seconds=base.sample_seconds)


def prefixes(env):
    rows=[]
    def visit(p,s):
        rows.append((p,s))
        for a in env.legal_actions(s):
            if a: visit(p+(a,),env.advance(s,a))
    visit((),env.initial)
    return rows


class TiePoolScreen(GuardFirstStagedSearch):
    def _expand_prefix(self,prefix,parent_id):
        assert not prefix and parent_id is None and self.stage_generation is None
        self.stage_generation={'version':'all_contiguous_legal_tie_closures_v1','candidates':[]}
        for p,_ in prefixes(self.env):
            if not p: continue
            full=p+(0,)*(self.horizon-len(p))
            branch_id=self._next_id
            self._add(p,full,ORIGIN,None)
            assert self._next_id==branch_id+1
            row={'candidate_id':branch_id,'held_out_loads':[],'timing':'ONE_PER_MINUTE',
                 'route_rank':branch_id,'actions':list(full),'prefix':list(p)}
            self.stage_branch_metadata[branch_id]=row
            self.stage_generation['candidates'].append(row)


def verify_sources(run):
    f=read(run/'freeze.json')
    assert sha(BASE/'completion_manifest.json')==BASE_SHA
    for name,h in f['input_hashes'].items(): assert sha(BASE/name)==h,name
    for name,h in f['source_hashes'].items(): assert sha(ROOT/name)==h,name
    assert sha(run/'schedule.json')==f['schedule_sha256'] and sha(PROTOCOL)==f['protocol_sha256']


def prepare():
    assert sha(BASE/'completion_manifest.json')==BASE_SHA and dependencies()==read(BASE/'dependencies.json')
    manifest={k.replace('\\','/'):v for k,v in read(BASE/'completion_manifest.json').items()}
    scientific=read(BASE/'prepared.json')['source_files']
    for name in scientific: assert sha(ROOT/name)==manifest['source/'+name],name
    raw_schedule=read(BASE/'schedule.json')
    specs=[s for s in raw_schedule if s['arm']=='P']
    inventory=[]
    selected=[]
    for spec in specs:
        env=env_for(spec)
        states=dict(prefixes(env))
        two=[p for p,s in states.items() if len(p)==2 and
             env.service_kw(s)>env.service_kw(states[p[:1]])>env.service_kw(env.initial)]
        inventory.append({'development_id':spec['context_id'],'prefixes':[list(p) for p in states],
                          'two_positive_closures':two})
        if two and len(selected)<2: selected.append(spec)
    assert [s['context_id'] for s in selected]==[3,6]
    run=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_external_common_20261106')
    run.mkdir()
    schedule=[]
    for i,s in enumerate(selected):
        for arm in [('IGRS','YUAN'),('YUAN','IGRS')][i]:
            schedule.append({'context_id':s['context_id'],'scenario_id':s['scenario_id'],'event_id':s['event_id'],
                'arm':arm,'cell':f'episodes/{len(schedule):03d}'})
    save(run/'schedule.json',schedule)
    save(run/'graph_inventory.json',inventory)
    for name in set(scientific)|{Path(__file__).relative_to(ROOT).as_posix(),PROTOCOL.relative_to(ROOT).as_posix(),
        EXTERNAL.relative_to(ROOT).as_posix(),'scripts/probe_igrs_wait_compression.py'}:
        target=run/'source'/name
        target.parent.mkdir(exist_ok=True,parents=True)
        shutil.copyfile(ROOT/name,target)
    inputs=['config.json','condition.json','load_reference.json','selected_events.json','selected_scenarios.json',
         'author_static_metadata.json','physical_config.json','source_pointer.json','source_provenance.json',
         'command_inventory.json','operating_reference.json','dependencies.json','source_inputs/base_load_properties.json',
         'source_inputs/metadata.json','schedule.json','physical_scope.json']
    for name in inputs: assert sha(BASE/name)==manifest[name],name
    shutil.copyfile(BASE/'dependencies.json',run/'dependencies.json')
    save(run/'freeze.json',{'study':'Yuan no-loop tie-only common adaptation','source_run':BASE.name,
        'source_sha256':BASE_SHA,'input_hashes':{n:manifest[n] for n in inputs},'seed':0,'workers':1,'training':0,
        'time_limit_s':240,'physical_limit':512,'logical_per_view':256,'schedule_sha256':sha(run/'schedule.json'),
        'protocol_sha256':sha(PROTOCOL),'graph_inventory_sha256':sha(run/'graph_inventory.json'),
        'source_hashes':{p.relative_to(run/'source').as_posix():sha(p) for p in (run/'source').rglob('*') if p.is_file()}})
    print(run,flush=True)


def snapshot_metrics(service,view,logical_id,meta):
    record=view._rows[logical_id]
    paid=service._pools['SHARED'][record['electrical_input_sha256']]
    raw=paid['payload']['raw_outputs']
    off=set(paid['payload']['metadata_override']['deenergized_buses'])
    values=[]
    for name,pu in zip(raw['node_names'],raw['node_vmag_pu']):
        bus,phase=name.lower().rsplit('.',1)
        if bus not in off and int(phase) in meta['physical_phase_nodes_by_bus'].get(bus,[]):
            values.append(float(pu))
    assert values and min(values)>0 and len(raw['losses_w_var'])==2
    return min(values),float(raw['losses_w_var'][0])/1000.


def worker(run,index):
    verify_sources(run)
    spec=read(run/'schedule.json')[index]
    cell=run/spec['cell']
    assert read(cell/'spec.json')==spec
    env=env_for(spec)
    cfg=BranchConfig(**read(BASE/'config.json')['branch_search'])
    scope=read(BASE/'physical_scope.json')
    with GuardFirstEvidenceService(BASE,cell,env,scope,mode='shared',physical_budget=512,logical_budget=256) as service:
        service.deadline_monotonic=time.perf_counter()+240.
        with service.view('GUARD') as view:
            guard=BranchSearch(env,view,cfg)
            while not guard.done:guard.step(guard.rule_action('beam'))
            g=guard.result()
            assert g['initial_certificate']['status']=='PASS' and g['best_plan']['certified_complete']
            save(cell/'GUARD/phase_result.json',g)
        with service.view('EARLY') as view:
            if spec['arm']=='IGRS':
                s=TiePoolScreen(env,view,cfg,guard_result=g,guard_reference={'phase':'GUARD','file':'GUARD/phase_result.json',
                    'sha256':sha(cell/'GUARD/phase_result.json')})
                controller=GuardFirstMacroController(s,method='beam')
                while not s.done:
                    proposal=controller.prepare()
                    if proposal is None:break
                    controller.apply(proposal)
                e=controller.result()
                assert s.done and 'ERROR' not in str(s.stop_reason) and e['initial_certificate']['status']=='PASS'
                merge=merge_results(e,g,protected=True)
                selected={'EARLY':e,'GUARD':g}[merge['selected_phase']]['best_plan']
                own=e.get('best_plan')
                extra={'pruned':len(s.free_pruning),'mode':'IGRS screening on tie-only pool',
                       'primary_restricted_plan':selected,'attempts':e['attempts']}
                save(cell/'EARLY/phase_result.json',e)
            else:
                initial=view.certify(env.initial,0)
                assert initial['status']=='PASS' and initial['reliable']
                prefix_evidence=[]
                meta=read(BASE/'source_inputs/metadata.json')
                def extends(p):
                    st=env.plan_states(env.initial,p)[-1]
                    return [a for a in env.legal_actions(st) if a]
                def evaluate(p):
                    states=env.plan_states(env.initial,p)
                    terminal=states[-1]
                    log={'prefix':list(p),'samples':[]}
                    prefix_evidence.append(log)
                    lid=initial['point_id']
                    for s in states[1:]:
                        for dt in (0,30,60):
                            sec=(s.step-1)*60+dt
                            cert=view.certify(s,sec)
                            assert cert['reliable'] and cert['status'] in ('PASS','FAIL')
                            log['samples'].append({'step':s.step,'time_seconds':sec,'logical_id':cert['point_id'],
                                                   'status':cert['status']})
                            if cert['status']=='FAIL':
                                log['status']='FAIL'
                                return {'status':'FAIL'}
                            if s.step==len(p) and dt==0: lid=cert['point_id']
                    minimum,loss=snapshot_metrics(service,view,lid,meta)
                    log.update(status='PASS',cost_logical_id=lid)
                    return {'status':'PASS','terminal_kw':env.service_kw(terminal),
                            'min_voltage_pu':minimum,'loss_kw':loss}
                solve=external_yuan.solve_prefix(extends,evaluate,env.total_kw,horizon=30)
                v=env.validate_plan(env.initial,solve['actions'],view)
                assert v['status'] in ('COMPLETE','FAIL')
                own=({'certified_complete':True,'status':'COMPLETE','actions':solve['actions'],
                      'validation':v} if v['status']=='COMPLETE' else None)
                selected=own if own and own['validation']['energy_kwh']>g['best_plan']['validation']['energy_kwh'] else g['best_plan']
                extra={'mode':'YUAN no-loop trellis','stages':solve,'initial_certificate':initial,
                       'prefix_physical_records':prefix_evidence,'terminal_validation':v}
                save(cell/'EARLY/phase_result.json',extra)
    account=service.summary()
    def metrics(plan):
        if plan is None:return None
        decisions=plan['actions']
        st=env.plan_states(env.initial,decisions)[-1]
        return {'actions':decisions,'operations':sum(a!=0 for a in decisions),
                'terminal_kw':env.service_kw(st),'energy_kwh':plan['validation']['energy_kwh']}
    gstat,ownstat,final=map(metrics,(g['best_plan'],own,selected))
    assert final['energy_kwh']>=gstat['energy_kwh'] and account['actual_numerical_entries']==3+4*account['physical_points']
    save(cell/'result.json',{'spec':spec,'account':account,'guard':gstat,'own_after_guard':ownstat,
        'selected':final,'selected_source':'GUARD' if selected is g['best_plan'] else 'EARLY',
        'external_own_output':ownstat if spec['arm']=='YUAN' else None,
        'sequence_before_wait': [a for a in final['actions'] if a],'zero_training':True})


def execute(run):
    verify_sources(run)
    assert dependencies()==read(BASE/'dependencies.json')
    save(run/'started.json',{'freeze_sha256':sha(run/'freeze.json'),'utc':datetime.now(timezone.utc).isoformat()})
    for i,spec in enumerate(read(run/'schedule.json')):
        cell=run/spec['cell']
        cell.mkdir(parents=True,exist_ok=False)
        save(cell/'spec.json',spec)
        start=time.perf_counter()
        try:
            with (cell/'worker.log').open('x',encoding='utf8') as log:
                subprocess.run([sys.executable,'-B',__file__,'worker',str(run),str(i)],timeout=240,
                               check=True,stdout=log,stderr=subprocess.STDOUT)
            seconds=time.perf_counter()-start
            save(cell/'finished.json',{'cold_seconds':seconds,'result_sha256':sha(cell/'result.json')})
        except Exception:
            save(cell/'failure.json',{'error':traceback.format_exc(),'elapsed_s':time.perf_counter()-start})
            raise
        print(i+1,'/4 finished',flush=True)
    save(run/'finished.json',{'cells':4})


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='worker':worker(Path(sys.argv[2]).resolve(),int(sys.argv[3]))
    elif sys.argv[1]=='execute':execute(Path(sys.argv[2]).resolve())
