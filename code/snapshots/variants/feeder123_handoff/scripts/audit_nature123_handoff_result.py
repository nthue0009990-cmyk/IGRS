"""Offline replay of scoped NC123 acceptance and search; no engine is loaded."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dnr.nature123_acceptance_runner import read_json,write_json,sha,seal_run,certificate_valid,read_journal
from dnr.nature123_clpu_environment import CLPUEnvironment, DynamicState
from dnr.nature123_static_state import SparseRestorationGraph
from dnr.nature123_phase_available_physical import certify_phase_available_snapshot


def seal(path):
    m=read_json(path/'completion_manifest.json')
    assert all(sha(path/k)==v for k,v in m.items()),str(path)
    assert set(m)=={p.relative_to(path).as_posix() for p in path.rglob('*')
                    if p.is_file() and p.name!='completion_manifest.json'}


def audit(folder):
    seal(folder)
    freeze=read_json(folder/'freeze.json')
    env=CLPUEnvironment(SparseRestorationGraph(read_json(ROOT/freeze['static_manifest_path']),['l19']),
        model_fingerprint=freeze['model_fingerprint'],condition=freeze['condition'],event=freeze['event'])
    is_search=freeze['schema_version']=='NC123_IGRS_CANDIDATE_HANDOFF_V2'
    work=folder if is_search else folder/'plans/first_batch_160kw_only'
    if not is_search: seal(work)
    for rel,digest in freeze['source_sha256'].items(): assert sha(folder/'source'/rel)==digest,rel
    observed=read_json(work/'supervisor_final.json')
    assert observed['returncode']==0 and observed['termination_reason'] is None
    assert observed['wall_seconds_observed']<=240 and observed['job_peak_committed_bytes']<=4*1024**3
    pid=read_json(work/'worker_identity.json')['pid']
    membership=read_json(work/'worker_job_membership.json')
    assert membership['pid']==pid and membership['actual_worker_in_supervisor_job'] is True
    assert membership['checked_before_release'] is True
    events,errors=read_journal(work/'events.jsonl');assert not errors
    reservations={};entry_reservations={};entry_returns={};completed={}
    logical={};logical_done={}
    for e in events:
        kind=e.get('event')
        if kind=='request_reserved':
            key=e['request_id'];assert key not in reservations
            assert e['request_number']==len(reservations)+1
            reservations[key]=e
        elif kind=='request_completed':
            key=e['point_id'];assert key in reservations and key not in completed
            completed[key]=e
        elif kind=='numerical_entries_reserved':
            token=e['token'];assert e['request_id'] in reservations
            if e['completion']=='NOT_REPORTED':
                assert token not in entry_reservations and token==len(entry_reservations)
                entry_reservations[token]=e
            else:
                assert token in entry_reservations and token not in entry_returns
                assert e['completion']=='API_RETURNED_NOT_CONVERGENCE'
                assert all(e[k]==entry_reservations[token][k] for k in ('count','request_id','operation','phase'))
                entry_returns[token]=e
        elif kind=='logical_request':
            assert e['logical_id']==len(logical)+1; logical[e['logical_id']]=e
        elif kind=='logical_completed':
            assert e['logical_id'] in logical and e['logical_id'] not in logical_done
            logical_done[e['logical_id']]=e
    assert set(entry_reservations)==set(entry_returns)
    points={};values=[]
    for file in sorted((work/'points').glob('*.json')):
        cert=read_json(file);point=cert['point_id'];request=cert['request']
        assert certificate_valid(cert,request,point)
        assert reservations[point]['request_sha256']==request['request_sha256']
        assert completed[point]['result_sha256']==sha(file)
        assert sum(e['count'] for e in entry_returns.values() if e['request_id']==point)==3
        state=DynamicState.from_dict(request['state'])
        assert env.snapshot_request(state,request['time_seconds'])==request
        assert cert['raw_outputs']['algorithm']==0
        checked=certify_phase_available_snapshot(cert['raw_outputs'],cert['metadata'],freeze['physical_config'])
        assert all(checked[k]==cert[k] for k in ('status','reliable','checks')) and cert['reliable'] is True
        element=next(e for e in cert['raw_outputs']['elements'] if e['name'].lower()=='line.l115')
        current=max(math.hypot(*element['currents_real_imag'][2*i:2*i+2])
                    for i,n in enumerate(element['node_order']) if n==1)
        points[point]=cert
        values.append({'point_id':point,'time_seconds':request['time_seconds'],'step':state.step,
                       'status':cert['status'],'l115_a':current})
    assert set(points)==set(reservations)==set(completed)
    initial_energy=env.service_kw(env.initial)*.5

    def prove(actions, refs):
        assert len(actions)==30 and len(refs)==91
        states=env.plan_states(env.initial,actions)
        schedule=[(env.initial,0)]+[(s,(s.step-1)*60+t) for s in states[1:] for t in (0,30,60)]
        selected=[]
        for (state,t),ref in zip(schedule,refs,strict=True):
            cert=points[ref['point_id']]
            assert cert['status']=='PASS' and cert['request']==env.snapshot_request(state,t)
            if 'request_sha256' in ref: assert ref['request_sha256']==cert['request_sha256']
            selected.append(ref['point_id'])
        energy=math.fsum(env.service_kw(s)/60 for s in states[1:])
        peak=max(v['l115_a'] for v in values if v['point_id'] in selected)
        return energy,peak

    if is_search:
        assert set(logical)==set(logical_done)
        for k,row in logical_done.items():
            assert row['request_sha256']==logical[k]['request_sha256']==points[row['point_id']]['request_sha256']
            assert row['cache_hit']==logical[k]['cache_hit']
        result=read_json(folder/'worker_result.json')
        ledger=result['resource_ledger']
        assert ledger['logical_requests']==len(logical)<=512
        assert ledger['physical_requests']==len(points)<=512
        assert ledger['cache_hits']==sum(v['cache_hit'] for v in logical.values())
        assert ledger['counter']['reserved_entries']==sum(e['count'] for e in entry_returns.values())<=1536
        best=result.get('best_plan')
        assert best and best['validation']['status']=='COMPLETE'
        actions=best['actions'];energy,peak=prove(actions,best['validation']['samples'])
        assert math.isclose(energy,best['validation']['energy_kwh'],abs_tol=1e-9)
        # Every complete attempt, including the private WAIT baseline, must replay.
        attempts,errors=read_journal(folder/'attempts.jsonl');assert not errors
        complete_count=0;wait_found=False
        for item in attempts:
            validation=item.get('validation',{})
            if validation.get('status')=='COMPLETE':
                e,_=prove(item['actions'],validation['samples'])
                assert math.isclose(e,validation['energy_kwh'],abs_tol=1e-9)
                complete_count+=1
                if item['actions']==[0]*30: wait_found=True;assert math.isclose(e,initial_energy,abs_tol=1e-9)
        assert wait_found
        handoff=result['handoff']
        assert handoff['verify_limit']==16
        assert handoff['verify_used']==handoff['carryover_verify_used']+handoff['staged_verify_used']<=16
        later=[x for x in attempts if x.get('origin') in ('guard_prefix_wait_carryover','staged_pickup')]
        assert len(later)==handoff['verify_used']
        for item in attempts:
            validation=item.get('validation',{})
            if validation.get('status')=='FAIL':
                assert validation['energy_kwh'] is None
                states=env.plan_states(env.initial,item['actions'])
                schedule=[(env.initial,0)]+[(s,(s.step-1)*60+t) for s in states[1:] for t in (0,30,60)]
                refs=validation['samples'];assert 0<len(refs)<=91
                labels=[]
                for (state,t),ref in zip(schedule,refs):
                    cert=points[ref['point_id']]
                    assert cert['request']==env.snapshot_request(state,t)
                    labels.append(cert['status'])
                assert labels==['PASS']*(len(labels)-1)+['FAIL']
        selection=handoff.get('selection')
        if selection is not None:
            guard=result['stages']['GUARD']
            selected_ids={x['branch_id'] for x in guard['trace']}
            checked_actions={tuple(x['actions']) for x in guard['attempts']}
            floor=guard['best_plan']['validation']['energy_kwh']
            expected={}
            for row in guard['generation_trace']:
                actions_key=tuple(row['actions'])
                if (row['branch_id'] not in selected_ids and row['origin']=='prefix_wait'
                    and actions_key not in checked_actions and row['graph_energy_kwh']>floor):
                    info=expected.setdefault(actions_key,{'ids':[],
                        'priority':(-row['graph_energy_kwh'],-row['graph_final_kw'],row['operations'],actions_key)})
                    info['ids'].append(row['branch_id'])
            ordered=sorted(expected,key=lambda actions_key:expected[actions_key]['priority'])
            assert [tuple(x['actions']) for x in selection['candidates']]==ordered
            for item in selection['candidates']:
                assert item['source_branch_ids']==expected[tuple(item['actions'])]['ids']
            assert [x['actions'] for x in later if x['origin']=='guard_prefix_wait_carryover']==[list(x) for x in ordered[:handoff['carryover_verify_used']]]
        improved_indices=[i for i,x in enumerate(attempts) if x.get('validation',{}).get('status')=='COMPLETE'
                          and x['validation']['energy_kwh']>initial_energy]
        if improved_indices: assert improved_indices==[len(attempts)-1]
        selected_phase=result.get('selected_phase')
        success=energy>initial_energy
        extra={'selected_phase':selected_phase,'search_stop_reason':result.get('stop_reason'),
               'logical_requests':len(logical),'cache_hits':ledger['cache_hits'],
               'complete_attempts_checked':complete_count,'scope_success':success,
               'handoff_candidates':len(selection['candidates']) if selection else 0,
               'handoff_verify_used':handoff['verify_used'],
               'carryover_verify_used':handoff['carryover_verify_used'],
               'staged_verify_used':handoff['staged_verify_used'],
               'selected_origin':best['origin'],'candidate_lineage_verified':True,
               'failed_attempts_checked':sum(x.get('status')=='FAIL' for x in attempts)}
    else:
        saved=read_json(folder/'result.json')['plans'][0]
        assert saved['status']=='COMPLETE' and saved['physical_certified'] is True
        actions=freeze['plans'][0]['actions']
        energy,peak=prove(actions,[{'point_id':'point_%02d'%i} for i in range(91)])
        assert math.isclose(energy,saved['certified_energy_kwh'],abs_tol=1e-9)
        assert len(points)==91 and sum(e['count'] for e in entry_returns.values())==273
        extra={}
    out=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_nc123_scoped_saved_audit')
    out.mkdir()
    report={'status':'PASS_SAVED_RECERTIFICATION','kind':'SEARCH' if is_search else 'PARTIAL_ACCEPTANCE',
        'source_run':str(folder),'source_manifest_sha256':sha(folder/'completion_manifest.json'),
        'actions':actions,'certified_energy_kwh':energy,'own_initial_wait_energy_kwh':initial_energy,
        'gain_kwh':energy-initial_energy,'selected_plan_l115_a_peak_a':peak,'physical_requests':len(points),
        'numerical_entries':sum(e['count'] for e in entry_returns.values()),
        'counts':{s:sum(c['status']==s for c in points.values()) for s in ('PASS','FAIL','UNKNOWN')},
        'supervisor':observed,'worker_job_membership':membership,'new_physical_requests':0,**extra}
    write_json(out/'result.json',report);write_json(out/'all_points.json',values)
    (out/'audit_source.py').write_bytes(Path(__file__).read_bytes());seal_run(out)
    target=ROOT/'artifacts/nature123_handoff_20261107';target.mkdir(exist_ok=True)
    write_json(target/('search_audit_latest.json' if is_search else 'partial_audit_latest.json'),
               {'run':str(out),'source_run':str(folder),'result_sha256':sha(out/'result.json')})
    print(json.dumps({'audit':str(out),**report},ensure_ascii=False),flush=True)


if __name__=='__main__': audit(Path(sys.argv[1]).resolve())


