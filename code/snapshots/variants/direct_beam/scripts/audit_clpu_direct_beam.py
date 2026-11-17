"""Independent saved-physics, causal-ledger and search replay for P/direct Beam."""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
sys.dont_write_bytecode=True
import run_clpu_direct_beam as runner
import audit_clpu_guard_first_pilot as gf
import clpu_direct_beam_audit_helpers as helper
shared,branch,old=gf.shared,gf.branch,gf.old
read,save,sha,require,same,read_lines=gf.read,gf.save,gf.sha,gf.require,gf.same,gf.read_lines


def mechanics(run,spec):
    scenarios={r['scenario_id']:r for r in read(run/'selected_scenarios.json')}
    events={r['event_id']:r for r in read(run/'selected_events.json')}
    return branch.Mechanics(read(run/'author_static_metadata.json'),read(run/'load_reference.json'),
        scenarios[spec['scenario_id']],read(run/'condition.json'),events[spec['event_id']],read(run/'config.json'))


def preflight(run):
    out=run/'preflight';out.mkdir(exist_ok=False)
    try:
        runner.verify(run)
        require(not (run/'execution_started.json').exists(),'Preflight after execution forbidden')
        contexts=read(run/'contexts.json');source_contexts=read(runner.SOURCE/'contexts.json')
        require(contexts==source_contexts,'Original population changed')
        require(sha(runner.SOURCE/'completion_manifest.json')==runner.SOURCE_SHA,'Prior seal changed')
        sm=runner.prior.base.normalize_manifest(read(runner.SOURCE/'completion_manifest.json'))
        for name in (*runner.prior.base.ROOT_INPUTS,'selected_scenarios.json','selected_events.json','training_contexts.json'):
            require(sha(run/name)==sm[name],'Physical/development input changed: '+name)
        cfg=read(run/'config.json');b=cfg['direct_beam_search'];p=cfg['branch_search']
        require(b['point_budget']==b['max_checked_candidates']==b['max_branch_expansions']==512 and b['max_generated_candidates']==10000,'Merged budget incomplete')
        for key in set(p)-{'point_budget','max_checked_candidates','max_branch_expansions','max_generated_candidates'}:
            require(p[key]==b[key],'Nonresource Beam configuration changed')
        checks=[]
        from run_nature34_clpu_pilot import make_env
        from dnr.nature34_clpu_staged_pickup import generate_staged_pickup_candidates
        for context in sorted(contexts,key=lambda c:c['context_id']):
            spec=next(s for s in read(run/'schedule.json') if s['context_id']==context['context_id'] and s['arm']=='P')
            expected=next(r['generation'] for r in read(run/'candidate_inventory.json') if r['context_id']==context['context_id'])
            actual=runner.numeric_lists(generate_staged_pickup_candidates(make_env(run,spec)))
            independent=gf.staged.independently_generate(mechanics(run,spec))
            gf.staged.check_generation(actual,independent)
            require(actual==expected,'Original candidate generator/input changed')
            checks.append({'context_id':context['context_id'],'candidate_count':len(actual['candidates']),'status':'PASS'})
        save(out/'summary.json',{'status':'PASS','freeze_sha256':sha(run/'prephysical_freeze_manifest.json'),
            'contexts':checks,'shared_actual_budget':512,'shared_total_logical_budget':512,
            'new_physical_queries':0,'new_training':0})
        return 0
    except BaseException as exc:
        save(out/'summary.json',{'status':'FAIL','error':repr(exc),'traceback':traceback.format_exc(),'new_physical_queries':0});raise


def scientific_cell(run,spec):
    cell=run/spec['cell'];cfg=read(run/'config.json');scope=read(run/'physical_scope.json')
    shared.checked_manifest(cell)
    require(not (cell/'error.json').exists(),'Failed task cannot pass')
    r=read(cell/'result.json');finished=read(cell/'worker_finished.json');started=read(cell/'worker_started.json')
    require(read(cell/'spec.json')==r['spec']==spec and r['study']=='CLPU_DIRECT_BEAM_TOTAL_BUDGET_V1','Wrong result identity')
    require(finished['status']=='COMPLETE' and finished['pid']==started['pid'] and finished['result_sha256']==sha(cell/'result.json'),'Unbound worker result')
    mech=mechanics(run,spec)
    points,statuses=shared.check_actual_points(run,cell/'physical',spec,mech)
    commands=old.check_commands(cell/'physical',points,read(run/'command_inventory.json'))
    shared.check_actual_requests(list(read_lines(cell/'physical/requests.jsonl')),points,read(cell/'physical/oracle_summary.json'),commands,statuses)
    order=tuple(spec['phase_order']);budget=spec['point_budget']
    views={n:{'logical_rows':list(read_lines(cell/n/'logical_points.jsonl.gz')),'requests':list(read_lines(cell/n/'logical_requests.jsonl'))} for n in order}
    evidence,_=helper.check_service_ledger(list(read_lines(cell/'service_events.jsonl')),views,points,mech,scope,
        sha(run/'condition.json'),'shared',logical_budget=budget,views_order=order)
    service=read(cell/'service_summary.json');plain=deepcopy(service)
    require(plain.pop('guard_first_version')==gf.FLOOR_VERSION and plain.pop('phase_order')==['GUARD','EARLY'],'Unexpected service semantics')
    helper.check_service_summaries(plain,{n:read(cell/n/'view_summary.json') for n in order},evidence,scope,'shared',commands['numerical_entries'],logical_budget=budget,views_order=order)
    require(r['service']==service,'Top service record differs')
    phases={};checks={}
    for name in order:
        folder=cell/name;p=read(folder/'phase_result.json')
        require(p['spec']==read(folder/'spec.json')==runner.phase_spec(spec,name),'Phase spec differs')
        require(read(folder/'initial_state.json')==mech.initial and p['oracle']==service['views'][name],'Initial state/view differs')
        attempts=list(read_lines(folder/'plans.jsonl.gz'));decisions=list(read_lines(folder/'decisions.jsonl.gz'))
        require(p['attempts']==attempts,'Attempt record omitted/changed')
        if name=='EARLY':
            reference={'phase':'GUARD','file':'GUARD/phase_result.json','sha256':sha(cell/'GUARD/phase_result.json')}
            checked=gf.check_early_records(mech,cfg['branch_search'],p['spec'],p,attempts,decisions,evidence[name],sha(run/'condition.json'),phases['GUARD'],reference)
            expected=next(x['generation'] for x in read(run/'candidate_inventory.json') if x['context_id']==spec['context_id'])
            require(p['stage_generation']==expected,'Changed fixed P candidates')
        else:
            bc=cfg['branch_search'] if spec['arm']=='P' else cfg['direct_beam_search']
            checked=helper.check_phase_records(mech,bc,name,p['spec'],p,attempts,decisions,evidence[name],sha(run/'condition.json'),logical_budget=budget,views_order=order)
        phases[name]=p;checks[name]=checked
    selected='GUARD'
    if spec['arm']=='P':
        e=phases['EARLY'];g=phases['GUARD']
        if e['fully_certified'] and e['best_plan']['validation']['energy_kwh']>g['best_plan']['validation']['energy_kwh']:selected='EARLY'
        gf.check_checkpoints(list(read_lines(cell/'cost_checkpoints.jsonl')),checks,phases)
    else:
        cps=list(read_lines(cell/'cost_checkpoints.jsonl'));boundaries=checks['GUARD']['boundaries']
        require(len(cps)==len(boundaries)+1,'Direct Beam checkpoint count')
        for i,cp in enumerate(cps):
            boundary=boundaries[min(i,len(boundaries)-1)]
            for key in ('phase_choices','logical_points','physical_points','quality_pct'):
                require(cp[key]==boundary[key],'Direct Beam checkpoint changed: '+key)
            require(cp['numerical_entries']==3+4*cp['physical_points'],'Unpaid initialization in checkpoints')
    require(r['checkpoints']==list(read_lines(cell/'cost_checkpoints.jsonl')),'Top checkpoints changed')
    final=phases[selected]
    require(final['fully_certified'] and r['fully_certified'] is True and r['selected_phase']==selected
        and r['best_plan']==final['best_plan'] and r['quality_pct']==final['quality_pct'],'Final certificate/selection changed')
    require(r['energy_kwh']==final['best_plan']['validation']['energy_kwh'] and r['baseline_energy_kwh']==phases['GUARD']['best_plan']['validation']['energy_kwh'],'Energy ledger changed')
    expected={'logical_points':sum(e['logical_points'] for e in evidence.values()),'actual_physical_points':len(points),
        'numerical_entries':commands['numerical_entries'],'requests':sum(e['requests'] for e in evidence.values()),'models_used':0,'new_training':0}
    for key,value in expected.items():require(r[key]==value,'Complete task accounting changed: '+key)
    require(r['logical_points']<=512 and r['numerical_entries']==3+4*len(points),'Resource envelope exceeded')
    require(finished['worker_elapsed_seconds']>=r['worker_elapsed_seconds']>=sum(p['elapsed_seconds'] for p in points.values()),'Physical time omitted')
    return {**{k:spec[k] for k in ('cell','context_id','arm','scenario_id','event_id')},'status':'PASS',
        'energy_kwh':r['energy_kwh'],'quality_pct':r['quality_pct'],**expected,'physical_statuses':dict(statuses),
        'phase_stop_reasons':r['phase_stop_reasons'],'baseline_energy_kwh':r['baseline_energy_kwh'],
        'worker_pid':started['pid'],'worker_elapsed_seconds':finished['worker_elapsed_seconds'],
        'result_sha256':sha(cell/'result.json')}


def audit(run):
    out=run/'audit';out.mkdir(exist_ok=False);tick=time.perf_counter();checks=[];errors=[]
    try:
        runner.verify(run);controller=read(run/'controller_completion.json');schedule=read(run/'schedule.json')
        require(controller['status']=='COMPLETE_PENDING_AUDIT' and not controller['not_run'] and controller['actual_tasks']==16,'Incomplete population retained, cannot report complete study')
        require(controller['exits']==read(run/'controller_exits.json') and len(controller['exits'])==16,'Controller journal mismatch')
        last_end=0.
        for spec,exit_row in zip(schedule,controller['exits']):
            require(exit_row['cell']==spec['cell'] and exit_row['exit_code']==0 and not exit_row['timed_out'],'Wrong/failed/retried task')
            c=scientific_cell(run,spec);execution=read(run/spec['cell']/'execution.json')
            require(c['worker_pid']==exit_row['pid'] and execution['spec_sha256']==sha(run/spec['cell']/'spec.json'),'Cold process not bound')
            require(last_end<=exit_row['started_epoch_seconds']==execution['started_epoch_seconds']<=exit_row['finished_epoch_seconds'],'Overlapped/reordered process')
            require(exit_row['process_wall_seconds']>=c['worker_elapsed_seconds'] and exit_row['process_wall_seconds']<=runner.WORKER_SECONDS+1,'Full cold time invalid')
            last_end=exit_row['finished_epoch_seconds'];c['process_wall_seconds']=exit_row['process_wall_seconds']
            checks.append(c);save(out/(spec['cell'].split('/')[-1]+'.json'),c)
            print(runner.canonical({'audited':len(checks),'cell':spec['cell'],'status':'PASS'}),flush=True)
        require(sum(c['actual_physical_points'] for c in checks)<=8192,'Total physical cap')
    except BaseException as exc:
        errors.append({'error':repr(exc),'traceback':traceback.format_exc()})
    summary={'status':'PASS' if not errors else 'FAIL','audited_cells':len(checks),'scheduled_cells':16,
        'errors':errors,'cells':checks,'elapsed_seconds':time.perf_counter()-tick,'new_physical_queries':0,'new_training':0,
        'freeze_sha256':sha(run/'prephysical_freeze_manifest.json'),'controller_sha256':sha(run/'controller_completion.json')}
    save(out/'summary.json',summary)
    return 0 if not errors else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',type=Path);parser.add_argument('--audit',type=Path);args=parser.parse_args()
    if args.preflight:raise SystemExit(preflight(args.preflight.resolve()))
    if args.audit:raise SystemExit(audit(args.audit.resolve()))
