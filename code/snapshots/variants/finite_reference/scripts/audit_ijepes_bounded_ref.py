"""Independent audit of bounded-family enumeration results, using saved evidence only."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import time
import traceback
import threading

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import audit_clpu_shared_pilot as raw_audit
import clpu_direct_beam_audit_helpers as service_audit

old=raw_audit.old
BASE=ROOT/'runs/20261105T024027_869785Z_clpu_direct_beam'
BASE_HASH='2989962bf0e870540c9fb37a07c8e0d31c7cbf951c0eb20e8177127a8a603932'
CAT=ROOT/'runs/20261106T145040_468073Z_ijepes_bounded_ref_count'
CAT_HASH='f8fdd3c328d7096d41763ad1d73bd69964f51815b80dbf8b35a36fded6b9a179'
FAMILIES=('switch_only','withdrawn_without_reconnection','withdrawn_then_reconnected')
read,read_lines=raw_audit.read,raw_audit.read_lines


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as fp:
        for block in iter(lambda:fp.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save(path, record):
    with path.open('x',encoding='utf8') as fp:
        json.dump(record,fp,allow_nan=False,ensure_ascii=False,indent=2)


def budget_adapter(function, expression, replacement, cap):
    """Keep the archived independent checker byte-pinned except its study-specific cap."""
    module=inspect.getmodule(function)
    expected={'audit_clpu_shared_pilot':'cc1a488d82d96147fbe53a390434a676599f1602628d084e3688ea92d028d9e6',
              'clpu_direct_beam_audit_helpers':'fe149e6f63845e4856634d2bb71c48610c4eff6def78628ce2b07b8f8a666dad'}
    file=Path(module.__file__)
    require(sha(file)==expected[module.__name__],'Independent checker source hash changed')
    source=inspect.getsource(function)
    require(source.count(expression)==1 and type(cap) is int and cap>0,'Unexpected frozen cap expression')
    ns={**function.__globals__,'BOUNDED_PHYSICAL_CAP':cap}
    exec(compile(source.replace(expression,replacement),str(file), 'exec'),ns)
    return ns[function.__name__]


class Replay:
    identity=raw_audit.branch.Mechanics.identity

    def __init__(self,run,spec):
        self.static=read(run/'author_static_metadata.json')
        self.loads=read(run/'load_reference.json')
        self.scenario=next(r for r in read(run/'selected_scenarios.json') if r['scenario_id']==spec['scenario_id'])
        self.event=next(r for r in read(run/'selected_events.json') if r['event_id']==spec['event_id'])
        self.condition=read(run/'condition.json')
        self.names=sorted(self.loads)
        self.horizon,self.slot,self.sample=30,60,30
        self.initial=old.independent_initial(self.static,self.loads,self.scenario)
        self.normal={}

    def states_energy(self,seq):
        states=[self.initial]
        for action in seq+(0,)*25:
            states.append(old.independent_advance(states[-1],action,self.static,self.loads,self.scenario,60))
        for s in states[1:]:
            if s['command'] not in self.normal:
                p=old.independent_projection(self.static,self.loads,self.scenario,s['command'])
                kw=sum(Decimal(str(self.loads[n]['nominal_kw'])) for n in p['energized'])
                self.normal[s['command']]=kw*Decimal(str(self.condition['load_multiplier']))
        return states,sum(self.normal[s['command']] for s in states[1:])


def bounds(rows):
    passing=[r for r in rows if r['status']=='PASS']
    open_=[r for r in rows if r['status']=='UNRESOLVED']
    exact=lambda r:Decimal(r['normal_kw_slot_sum'])
    lower=max(map(exact,passing),default=None)
    upper=max(map(exact,passing+open_),default=None)
    certified=bool(passing) and not open_
    best=[r['actions_first_5'] for r in passing if exact(r)==lower]
    return dict(plans=len(rows),status_counts={s:sum(r['status']==s for r in rows) for s in ('PASS','FAIL','UNRESOLVED')},
        classification=('EMPTY_DOMAIN' if not rows else 'UNRESOLVED' if open_ else
                        'ENUMERATED_BOUNDED_OPTIMUM' if passing else 'NO_FEASIBLE_PLAN_IN_DECLARED_DOMAIN'),
        lower_kwh=None if lower is None else float(lower/60),upper_kwh=None if upper is None else float(upper/60),
        lower_kw_slot_sum=None if lower is None else str(lower),upper_kw_slot_sum=None if upper is None else str(upper),
        best_certified_action_prefixes=best,all_optimal_action_prefixes=best if certified else None,
        unresolved_action_prefixes=[r['actions_first_5'] for r in open_])


def group_summary(rows):
    grouped={'all':bounds(rows)}
    for key in FAMILIES:
        grouped[key]=bounds([r for r in rows if r['family']==key])
    return grouped


def base_rows(catalog):
    result=[]
    for idx,p in enumerate(catalog['sorted_candidate_catalog']):
        prefix=p['actions_first_5']
        flips=prefix.count(catalog['prep_action'])
        require(len(prefix)==5 and flips in (0,1,2),'Malformed declared domain')
        result.append(dict(index=idx,actions_first_5=prefix,normal_kw_slot_sum=p['normal_kw_slot_sum'],
                           normal_kwh=p['normal_kwh'],target_first_pickup_s=p['target_first_pickup_s'],
                           family=FAMILIES[flips],status='UNRESOLVED'))
    return result


def verify_ledgers(run,cell,spec,mech):
    cap=spec['physical_cap']
    check_points=budget_adapter(raw_audit.check_actual_points,'point_id < 512','point_id < BOUNDED_PHYSICAL_CAP',cap)
    check_requests=budget_adapter(raw_audit.check_actual_requests,"summary['point_budget'] == 512",
                                  "summary['point_budget'] == BOUNDED_PHYSICAL_CAP",cap)
    check_events=budget_adapter(service_audit.check_service_ledger,'physical_budget=512',
                                'physical_budget=BOUNDED_PHYSICAL_CAP',cap)
    check_totals=budget_adapter(service_audit.check_service_summaries,"'physical_budget': 512",
                                "'physical_budget': BOUNDED_PHYSICAL_CAP",cap)
    points,statuses=check_points(run,cell/'physical',spec,mech)
    commands=old.check_commands(cell/'physical',points,read(run/'command_inventory.json'))
    require(commands['numerical_entries']<=spec['numerical_cap'],'Numerical-entry cap')
    check_requests(list(read_lines(cell/'physical/requests.jsonl')),points,
                   read(cell/'physical/oracle_summary.json'),commands,statuses)
    scope=read(run/'physical_scope.json')
    views={'GUARD':{'logical_rows':list(read_lines(cell/'GUARD/logical_points.jsonl.gz')),
                    'requests':list(read_lines(cell/'GUARD/logical_requests.jsonl'))}}
    events=list(read_lines(cell/'service_events.jsonl'))
    checked,_=check_events(events,views,points,mech,scope,sha(run/'condition.json'),'shared',
                           logical_budget=spec['logical_cap'],views_order=('GUARD',))
    summary=read(cell/'service_summary.json')
    plain=dict(summary)
    require(plain.pop('guard_first_version')=='clpu-public-guard-first-certified-floor-v1'
            and plain.pop('phase_order')==['GUARD','EARLY'] and list(plain['views'])==['GUARD'],
            'Wrong service protocol or a second actual view')
    check_totals(plain,{'GUARD':read(cell/'GUARD/view_summary.json')},checked,scope,'shared',
                 commands['numerical_entries'],logical_budget=spec['logical_cap'],views_order=('GUARD',))
    require(len(views['GUARD']['requests'])<=spec['requests_cap'],'Request cap')
    return checked['GUARD'],views['GUARD']['requests'],summary,commands


def check_plans(run,cell,catalog,spec):
    mech=Replay(run,spec)
    ledger,requests,account,commands=verify_ledgers(run,cell,spec,mech)
    result=read(cell/'result.json')
    require(result['account']==account and result['spec']==catalog['spec'] and
            result['solver_training_calls']==0,'Task identities or accounts differ')
    logical=ledger['logical']
    initial=read(cell/'initial.json')
    require(initial['state']==mech.initial and initial['time_seconds']==0,'Initial state/time mismatch')
    first=initial['certificate']
    require(first==ledger['logical_rows'][0]['certificate'] and first['point_id']==0 and
            logical[0]['dynamic_state']==mech.initial and logical[0]['time_minutes']==0 and
            requests[0]['logical_point_id']==0,'Missing initial paid evidence')
    rows=base_rows(catalog)
    records=list(read_lines(cell/'plans.jsonl'))
    require(len(rows)==len(records),'Incomplete plan census')
    cursor=1
    id_sha=sha(run/'condition.json')
    hashed={}
    starts=list(read_lines(cell/'plan_starts.jsonl'))
    seen_actual={ledger['logical_rows'][0]['physical_point_id']}
    for row,saved in zip(rows,records,strict=True):
        require(all(row[k]==saved[k] for k in ('index','actions_first_5','normal_kw_slot_sum','normal_kwh')),
                'Plan identity, order or catalog energy differs')
        seq=tuple(row['actions_first_5'])
        states,energy=mech.states_energy(seq)
        require(energy==Decimal(row['normal_kw_slot_sum']) and
                math.isclose(float(energy/60),row['normal_kwh'],abs_tol=1e-11,rel_tol=0),'Energy replay failed')
        samples=saved['samples']
        require(isinstance(samples,list) and len(samples)<=90,'Invalid sample count')
        if samples:
            require(first['status']=='PASS' and saved['first_request']==cursor+1 and
                    saved['last_request']==cursor+len(samples),'Wrong request slice or initial FAIL')
        else:
            require(saved['first_request'] is None and saved['last_request'] is None,'Phantom request')
        if row['index']<len(starts):
            require(starts[row['index']]==dict(index=row['index'],actions_first_5=list(seq),
                    requests_before=cursor,actual_before=len(seen_actual)),'Plan dispatch order/cost')
        else:
            require(not samples,'Candidate with no recorded dispatch')
        failed=None
        for index,lid in enumerate(samples):
            step,off=divmod(index,3)
            state=states[step+1]; seconds=60*step+off*30
            require(type(lid) is int and cursor<len(requests) and requests[cursor]['logical_point_id']==lid,
                    'Candidate mentions a different logical request')
            cursor+=1
            seen_actual.add(ledger['logical_rows'][lid]['physical_point_id'])
            key=(state['command'],tuple(state['pickups']),seconds)
            if key not in hashed:
                hashed[key]=raw_audit.digest(mech.identity(state,seconds,id_sha))
            point=logical[lid]
            require(hashed[key]==point['identity_sha256'],'Sample not certified for current action/history/time')
            if point['status']=='FAIL':
                require(index==len(samples)-1,'Skipped first reliable FAIL')
                constraints=sorted({v['constraint'] for v in point['structured_feedback']['violations']})
                require(constraints,'FAIL missing constraint evidence')
                failed=dict(slot=step,time_seconds=seconds,logical_id=lid,
                            physical_id=ledger['logical_rows'][lid]['physical_point_id'],constraints=constraints,
                            violations=point['structured_feedback']['violations'])
        status='FAIL' if failed else 'PASS' if len(samples)==90 else 'UNRESOLVED'
        require(saved['status']==status and saved['failed_step']==(failed['slot'] if failed else None) and
                saved['failed_time_seconds']==(failed['time_seconds'] if failed else None),'Status/first FAIL differs')
        if failed:
            require(sorted(saved['failure_constraints'])==failed['constraints'],'Reported limiting constraints differ')
        require(math.isclose(saved['certified_kwh'],row['normal_kwh'],abs_tol=1e-8,rel_tol=0)
                if status=='PASS' else saved['certified_kwh'] is None,'Reported certified energy inconsistent')
        if not failed:
            require(saved['failure_constraints']==[],'Spurious failed constraint')
        row.update(status=status,first_fail=failed,checked_slots_samples=len(samples),execution_reason=saved['reason'])
    require(cursor==len(requests),'Unattributed logical requests')
    groups=group_summary(rows)
    require(result['counts']==groups['all']['status_counts'],'Outcome counts differ')
    for key in ('all','switch_only'):
        band=groups[key]
        b=result['bounds'][key]
        require(b['count']==band['plans'] and b['unresolved']==band['status_counts']['UNRESOLVED'] and all(
            b[k]==band[k] for k in ('lower_kw_slot_sum','upper_kw_slot_sum','lower_kwh','upper_kwh')),
            'Bounds disagree with independent classification')
        best=set(map(tuple,band['best_certified_action_prefixes']))
        require(b['best_verified_indices']==[r['index'] for r in rows if tuple(r['actions_first_5']) in best]
                and b['fully_examined']==(b['unresolved']==0)
                and b['certified_optimum']==(band['classification']=='ENUMERATED_BOUNDED_OPTIMUM')
                and b['all_reliably_infeasible']==(band['classification']=='NO_FEASIBLE_PLAN_IN_DECLARED_DOMAIN'),
                'Reported optimality, ties or closure inconsistent')
    require(result['status']!='CASE_COMPLETE' or first['status']=='PASS' and
            groups['all']['status_counts']['UNRESOLVED']==0,'Complete claim with missing decisions')
    return dict(status='CHECKED_SAVED_LEDGER',case_result=result['status'],families=groups,
                requests=cursor,logical_points=ledger['logical_points'],physical_points=ledger['physical_points'],
                semantic_hits=ledger['semantic_cache_hits'],input_reuses=ledger['electrical_reuses'],
                physical_statuses=read(cell/'physical/oracle_summary.json')['statuses'],commands=commands,
                initial_status=first['status'],plans=rows)


def bind_run(run):
    frozen=read(run/'freeze.json')
    require(run.parent==ROOT/'runs' and run.name.endswith('_ijepes_bounded_ref'),'Unexpected run path')
    started=read(run/'started.json')
    require(read(run/'prepared.json')['freeze_sha256']==started['freeze_sha256']==sha(run/'freeze.json'),
            'Freeze changed after preparation/launch')
    require(sha(BASE/'completion_manifest.json')==frozen['source_seal']==BASE_HASH and
            sha(CAT/'completion_manifest.json')==frozen['catalog_seal']==CAT_HASH,'Source/catalog seals')
    orig={n.replace(chr(92),'/'):v for n,v in read(BASE/'completion_manifest.json').items()}
    cat_manifest=read(CAT/'completion_manifest.json')
    required=('author_static_metadata','load_reference','selected_events','selected_scenarios','condition',
              'command_inventory','physical_config','operating_reference','source_inputs/metadata','physical_scope')
    require(all(k+'.json' in frozen['inputs'] for k in required),'Missing essential input binding')
    for n,h in frozen['inputs'].items():
        require(orig[n]==h==sha(run/'inputs'/n)==sha(BASE/n),'Physical inputs changed '+n)
    for n,h in frozen['sources'].items():
        require(h==sha(ROOT/n)==sha(run/'source'/n),'Study code changed '+n)
    for n,h in frozen['gate_files'].items():
        require(sha(ROOT/n)==h,'Changed preflight evidence '+n)
    for n,h in frozen['files'].items():
        require(sha(run/n)==h,'Changed configuration '+n)
    for n,h in frozen['catalogs'].items():
        require(sha(run/n)==h==cat_manifest[Path(n).name],'Changed graph catalog '+n)
    cfg=read(run/'config.json')
    schedule=read(run/'schedule.json')
    require(cfg['schedule']==schedule and [s['context_id'] for s in schedule]==list(range(8)) and
            cfg['audit_limit_s']==started['audit_limit_s']==3600,'Wrong schedule or limits')
    for i,s in enumerate(schedule):
        cat=read(run/s['catalog'])
        require(s['ordinal']==i and s['seed']==0 and s['cell']==f'episodes/c{i:02d}' and
                s['catalog']==f'catalog/c{i:02d}.json' and all(cat['spec'][k]==s[k] for k in cat['spec']),
                'Case binding')
        require((s['plan_count'],s['requests_cap'],s['logical_cap'],s['physical_cap'],s['numerical_cap'])==
            tuple(cat[k] for k in ('plan_count','requests_upper','unique_logical_upper','unique_actual_upper','numerical_entries_upper')),
            'Case caps differ from sealed graph count')
    return schedule


def audit(run,out):
    began=time.monotonic()
    started=read(run/'started.json')
    budget=min(3600.,started['chain_deadline_epoch_seconds']-time.time())
    require(budget>0,'Study chain deadline passed')
    import _thread
    watchdog=threading.Timer(budget,_thread.interrupt_main)
    watchdog.daemon=True
    watchdog.start()
    out.mkdir(exist_ok=False)
    report={'status':'IN_PROGRESS','new_physical_queries':0,'optimization_calls':0,'audit_limit_seconds':budget,
            'source_sha256':sha(Path(__file__)),'cases':[],'errors':[]}
    try:
        schedule=bind_run(run)
        controller=read(run/'controller_result.json')
        exits=controller['exits']
        require(controller['not_dispatched']==schedule[len(exits):],'Unaccounted case dispatches')
        pids=set()
        finish=None
        for i,s in enumerate(schedule):
            cat=read(run/s['catalog'])
            cell=run/s['cell']
            case={'status':'NOT_DISPATCHED','plans':base_rows(cat)}
            try:
                if i<len(exits):
                    stop=exits[i]
                    require(stop==read(cell/'cold_exit.json') and s==read(cell/'spec.json') and
                            stop['context_id']==i and stop['cell']==s['cell'],'Case start/exit evidence')
                    launch=read(cell/'launch.json')
                    require(launch['controller_pid']==started['pid'] and
                            0<launch['deadline']-launch['start']<=1200.00001 and
                            (finish is None or launch['start']>=finish-.001),'Serial launch bounds/order')
                    finish=launch['start']+stop['cold_seconds']
                    require(stop['pid'] not in pids,'Cold process reused')
                    pids.add(stop['pid'])
                    if stop['failure'] or stop['exit_code']!=0 or stop['worker_status']=='STOP_RELIABILITY_ERROR':
                        case['status']='UNVERIFIED_INTERRUPTED_EVIDENCE'
                        case['exit']=stop
                        require(i==len(exits)-1,'Continued after unreliable case')
                    else:
                        proc=read(cell/'worker_started.json')
                        require(proc['pid']==read(cell/'go.json')['worker_pid']==stop['pid'] and
                                proc['parent_pid']==started['pid'],'Worker/parent identity')
                        require(stop['peak_rss_bytes']<=4*2**30 and 0<stop['cold_seconds']<=1200 and
                                stop['result_sha256']==sha(cell/'result.json'),'Invalid cold run or digest')
                        case=check_plans(run/'inputs',cell,cat,s)
                        require(case['case_result']==stop['worker_status'],'Terminal status mismatch')
                else:
                    require(not cell.exists(),'Undispatched case had execution artifacts')
            except Exception:
                case={'status':'INVALID_EVIDENCE','plans':base_rows(cat),'error':traceback.format_exc()}
                report['errors'].append({'context_id':i,'detail':case['error']})
            if 'families' not in case:
                case['families']=group_summary(case['plans'])
            case['context_id']=i
            save(out/f'c{i:02d}.json',case)
            report['cases'].append({k:v for k,v in case.items() if k!='plans'})
            print(i,case['status'],case['families']['all']['status_counts'],flush=True)
        require(sum(r['families']['all']['plans'] for r in report['cases'])==438,'Incomplete total')
        complete=all(c['status']=='CHECKED_SAVED_LEDGER' and c['case_result']=='CASE_COMPLETE' for c in report['cases'])
        require((controller['status']=='COMPLETE_PENDING_INDEPENDENT_AUDIT')==complete or report['errors'],
                'Controller completion inconsistent')
        report['status']='AUDIT_FAILED' if report['errors'] else 'PASS_COMPLETE' if complete else 'PARTIAL_WITH_UNRESOLVED'
        report['totals']={k:sum(c.get(k,0) for c in report['cases']) for k in ('requests','logical_points','physical_points','semantic_hits','input_reuses')}
        report['totals']['numerical_entries_in_fully_audited_cases']=sum(
            c['commands']['numerical_entries'] for c in report['cases'] if 'commands' in c)
        report['totals']['counts']={k:sum(c['families']['all']['status_counts'][k] for c in report['cases'])
                                    for k in ('PASS','FAIL','UNRESOLVED')}
        report['controller_sha256']=sha(run/'controller_result.json')
        report['freeze_sha256']=sha(run/'freeze.json')
    except BaseException:
        report['status']='AUDIT_FAILED'
        report['errors'].append(traceback.format_exc())
        raise
    finally:
        watchdog.cancel()
        report['elapsed_seconds']=time.monotonic()-began
        save(out/'summary.json',report)
        save(out/'manifest.json',{p.name:sha(p) for p in sorted(out.glob('*.json'))})
    return report


def selftest():
    require(sha(CAT/'completion_manifest.json')==CAT_HASH,'Missing sealed population')
    sizes=Counter()
    for i in range(8):
        cat=read(CAT/f'c{i:02d}.json')
        rows=base_rows(cat)
        sizes.update(r['family'] for r in rows)
        require(group_summary(rows)['all']['lower_kwh'] is None,'Missing evidence improperly feasible')
    require([sizes[k] for k in FAMILIES]==[103,215,120],'Declared family sizes')
    probe=[dict(index=i,actions_first_5=[i],normal_kw_slot_sum=s,status=t) for i,(s,t) in enumerate(
        [('60','PASS'),('60.0','PASS'),('60.0000000000001','UNRESOLVED'),('70','FAIL')])]
    b=bounds(probe)
    require(Decimal(b['upper_kw_slot_sum'])>Decimal(b['lower_kw_slot_sum']) and
            b['all_optimal_action_prefixes'] is None and b['best_certified_action_prefixes']==[[0],[1]],'Open bounds')
    probe[2]['status']='FAIL'
    require(bounds(probe)['all_optimal_action_prefixes']==[[0],[1]] and
            bounds([])['classification']=='EMPTY_DOMAIN' and
            bounds([probe[2]])['classification']=='NO_FEASIBLE_PLAN_IN_DECLARED_DOMAIN','Empty/all FAIL/ties')
    for fn,original,new in [(raw_audit.check_actual_points,'point_id < 512','point_id < BOUNDED_PHYSICAL_CAP'),
        (raw_audit.check_actual_requests,"summary['point_budget'] == 512","summary['point_budget'] == BOUNDED_PHYSICAL_CAP"),
        (service_audit.check_service_ledger,'physical_budget=512','physical_budget=BOUNDED_PHYSICAL_CAP'),
        (service_audit.check_service_summaries,"'physical_budget': 512","'physical_budget': BOUNDED_PHYSICAL_CAP")]:
        fn2=budget_adapter(fn,original,new,1222)
        require(fn2.__globals__['BOUNDED_PHYSICAL_CAP']==1222,'Wrong passed budget')
    fixture=BASE/'episodes/001'
    spec=read(fixture/'spec.json')
    require(sha(BASE/'completion_manifest.json')==BASE_HASH and spec['phase_order']==['GUARD'],'Historical single-view fixture')
    spec.update(physical_cap=spec['physical_point_budget'],logical_cap=spec['point_budget'],
                numerical_cap=3+4*spec['physical_point_budget'],requests_cap=100000)
    ledger,requests,account,commands=verify_ledgers(BASE,fixture,spec,Replay(BASE,spec))
    require(len(requests)>len(ledger['logical'])>len(list(read_lines(fixture/'physical/requests.jsonl')))/2 and
            commands['numerical_entries']==3+4*account['physical_points'],'Real paid/alias evidence fixture')
    return dict(status='PASS',family_counts=dict(sizes),new_physical_queries=0,optimization_calls=0,
                bounds_checked=['all tied optima','almost equal unresolved','no known feasible','all fail','empty family'],
                real_evidence_fixture=fixture.relative_to(ROOT).as_posix(),physical_rows_reaudited=account['physical_points'],
                logical_rows_reaudited=len(ledger['logical']),logical_requests_reaudited=len(requests),
                numerical_entries_reaudited=commands['numerical_entries'],
                auditor_sha256=sha(Path(__file__)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path,nargs='?')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.self_test:
        data=selftest()
        if args.output:
            save(args.output,data)
        print(json.dumps(data))
    else:
        if not args.run:
            parser.error('A frozen reference run directory is required')
        run=args.run.resolve()
        data=audit(run,args.output or run/'independent_audit')
        print(data['status'])
        sys.exit(int(data['status']=='AUDIT_FAILED'))
