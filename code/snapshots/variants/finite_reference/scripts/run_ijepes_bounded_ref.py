"""Cold, eight-case offline exhaustive physical reference. No search/solver arm."""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

from run_chen_two_cold import (ROOT, BASE, BASE_SHA, GiB, DISK_FLOOR, RSS_LIMIT, EXPECTED_PYTHON,
    WinJob, peak_rss, memory_state, env_check, require, read, sha, save, rel)

CAT = ROOT/'runs/20261106T145040_468073Z_ijepes_bounded_ref_count'
CAT_SEAL = 'f8fdd3c328d7096d41763ad1d73bd69964f51815b80dbf8b35a36fded6b9a179'
PROTOCOL = ROOT/'docs/proposals/ijepes_bounded_reference_v1.md'
INPUTS = ('config.json', 'condition.json', 'selected_events.json','selected_scenarios.json',
          'author_static_metadata.json','command_inventory.json','physical_config.json',
          'metadata.json','load_reference.json','source_inputs/metadata.json',
          'source_inputs/base_load_properties.json','source_provenance.json','source_pointer.json',
          'dependencies.json','operating_reference.json','physical_scope.json','resource_budget.json',
          'schedule.json','prepared.json')
COLD_S = 1200
CHAIN_S, DISPATCH_S, AUDIT_S = 230*60, 170*60, 60*60
EXTRA = 4*GiB
CAPS = ((56,5041,1792,571),(6,541,346,62),(31,2791,957,286),(126,11341,2931,1222),
        (31,2791,957,286),(31,2791,957,286),(126,11341,2931,1222),(31,2791,957,286))


def size(p):
    return sum(f.stat().st_size for f in p.rglob('*') if f.is_file())


def catalog_schedule():
    require(sha(CAT/'completion_manifest.json')==CAT_SEAL, 'Bound catalog seal differs')
    manifest=read(CAT/'completion_manifest.json')
    for n,h in manifest.items():
        require(sha(CAT/n)==h,'Changed catalog input/record: '+n)
    summary=read(CAT/'summary.json')
    require([r['spec']['context_id'] for r in summary]==list(range(8)), 'Wrong catalog population')
    schedule=[]
    for i,r in enumerate(summary):
        key=f'c{i:02d}'
        cat=read(CAT/(key+'.json'))
        require(cat['spec']==r['spec'] and len(cat['sorted_candidate_catalog'])==r['plan_count'],
                'Catalog candidate/spec mismatch')
        counts=tuple(r[k] for k in ('plan_count','requests_upper','unique_logical_upper','unique_actual_upper'))
        require(counts==CAPS[i] and r['numerical_entries_upper']==3+4*counts[3], 'Resource counts changed')
        schedule.append(dict(**r['spec'],ordinal=i,cell='episodes/'+key,catalog='catalog/'+key+'.json',
                plan_count=counts[0],requests_cap=counts[1],logical_cap=counts[2],physical_cap=counts[3],
                numerical_cap=r['numerical_entries_upper'],seed=0))
    return schedule


def check_gate(path):
    g=read(path)
    require(g['status']=='PASS' and g['cases']==list(range(8)) and g['new_physical_queries']==0 and g['hashes'],
            'Missing passing all-eight zero-flow gate')
    out={rel(path):sha(path)}
    for name,h in g['hashes'].items():
        p=ROOT/name
        require(sha(p)==h,'Changed gate evidence: '+name)
        out[rel(p)]=h
    return out


def storage_proof():
    """Measure native schema bytes; charge conservative row overhead at every occurrence."""
    base_manifest={k.replace(chr(92),'/'):v for k,v in read(BASE/'completion_manifest.json').items()}
    logical_max=request_max=event_max=0
    files=0
    for name,h in base_manifest.items():
        basename=Path(name).name
        if not name.startswith('episodes/') or basename not in {
                'logical_points.jsonl.gz','logical_requests.jsonl','service_events.jsonl'}:
            continue
        f=BASE/name
        require(sha(f)==h,'Storage calibration changed: '+name)
        with (gzip.open(f,'rb') if basename.endswith('.gz') else f.open('rb')) as stream:
            for line in stream:
                if basename=='logical_points.jsonl.gz':
                    obj=json.loads(line)
                    obj['certificate']['feedback']=[]
                    logical_max=max(logical_max,len(json.dumps(obj,ensure_ascii=False).encode('utf8'))+2)
                elif basename=='logical_requests.jsonl':
                    request_max=max(request_max,len(line))
                else:
                    event_max=max(event_max,len(line))
        files+=1
    require(files and 0<logical_max<=64*1024 and 0<request_max<=2048 and 0<event_max<=16*1024,
            'Observed stored record exceeds forecast envelope')
    meta=read(BASE/'metadata.json')
    # One voltage item per modeled node, one line-current item per terminal
    # entry, and at most one terminal total plus per-conductor item per transformer.
    layout=meta['element_layout']
    max_violations=sum(len(v) for v in meta['node_phases'].values()) + sum(
        len(e['node_order']) + (e['terminals'] if e['kind'].lower()=='transformer' else 0)
        for e in layout.values())
    require(max_violations <= 1024 and max(map(len,layout))<=128, 'Device feedback envelope not applicable')
    # At most 438 rejected plans plus eight initial failures produce new
    # distinct FAIL logical rows, because the first reliable violation ends a plan.
    # Account for three copies of feedback: native logical result, candidate
    # certificate and auxiliary result/index, 2KiB per compact violation.
    envelope=dict(logical_without_feedback=11828*64*1024,
        request_logs_and_started_completed_events=39428*3*2048,
        completed_points_and_paid_point_events=(11828+2*4221)*16*1024,
        failed_path_feedback=(438+8)*max_violations*2048*3,
        candidate_compact_sample_records=39428*2048,
        catalog_results_reports_overhead=256*2**20)
    # The worker counts uncompressed UTF8 lengths and preserves the first
    # exceptional line before stopping. A successful full run obeys these caps.
    components=dict(logical_rows=11828*224*1024, actual_start=4221*64*1024,
        ordinary_lines=(3*39428+11828+4221+80)*4*1024, plans=438*8*1024,
        metadata_and_interruption_allowance=128*2**20, audit_reserve=512*2**20)
    require(max(sum(components.values()),sum(envelope.values()))<=EXTRA,
            'Extra output does not fit the 4 GiB reserve')
    actual=math.ceil(read(BASE/'resource_budget.json')['forecast_bytes']*4221/8192)
    return dict(calibration_files=files, max_observed_bytes=dict(logical_without_feedback=logical_max,
            logical_request=request_max,service_event=event_max), feedback_items_bound=max_violations,
            schema_budget=envelope, components=components, modeled_extra_total=sum(components.values()), extra_reserve=EXTRA,
            physical_forecast=actual, required_available=max(math.ceil(21.03*GiB),DISK_FLOOR+EXTRA+actual))


def bind_sources(gate_files):
    base_manifest={n.replace(chr(92),'/'):h for n,h in read(BASE/'completion_manifest.json').items()}
    require(sha(BASE/'completion_manifest.json')==BASE_SHA, 'Changed base archive')
    sources={}
    for n in read(BASE/'prepared.json')['source_files']:
        name=rel(ROOT/n)
        require(sha(ROOT/name)==base_manifest['source/'+name], 'Changed inherited source: '+name)
        sources[name]=sha(ROOT/name)
    countcfg=read(CAT/'config.json')
    for n,h in countcfg['source_sha256'].items():
        require(sha(ROOT/n)==h,'Changed counting/graph implementation: '+n)
        sources[n]=h
    for n in ('scripts/count_ijepes_bounded_reference.py','scripts/worker_ijepes_bounded_ref.py',
              'scripts/run_ijepes_bounded_ref.py','scripts/audit_ijepes_bounded_ref.py',
              'scripts/run_chen_two_cold.py',rel(PROTOCOL)):
        sources[n]=sha(ROOT/n)
    sources.update({n:h for n,h in gate_files.items() if n.startswith(('src/','scripts/','tests/'))})
    return sources, {n:base_manifest[n] for n in INPUTS}


def prepare(gatepath):
    started=time.time()
    env_check()
    gate_files=check_gate(gatepath)
    schedule=catalog_schedule()
    sources,inputs=bind_sources(gate_files)
    budget=storage_proof()
    require(memory_state()['available_physical_bytes']>=RSS_LIMIT,'Insufficient available memory')
    require(shutil.disk_usage(ROOT).free>=budget['required_available'],'Insufficient disk')
    run=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_ijepes_bounded_ref')
    run.mkdir()
    def copy(path,target,digest):
        require(sha(path)==digest,'Changed snapshot input '+str(path))
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
        require(sha(target)==digest,'Snapshot copy differs')
    for n,h in sources.items(): copy(ROOT/n,run/'source'/n,h)
    for n,h in inputs.items(): copy(BASE/n,run/'inputs'/n,h)
    catfiles={}
    for spec in schedule:
        n=spec['catalog'];orig=CAT/Path(n).name
        catfiles[n]=sha(orig);copy(orig,run/n,catfiles[n])
    save(run/'schedule.json',schedule)
    save(run/'config.json',dict(study='IJEPES bounded offline reference',seed=0,workers=1,empty_cache_per_case=True,
        schedule=schedule,solver_arms=0,requests_total=39428,logical_total=11828,physical_total=4221,
        numerical_total=16908,wall_case_s=COLD_S,case_time_sum_s=8*COLD_S,
        dispatch_limit_s=DISPATCH_S,audit_limit_s=AUDIT_S,chain_limit_s=CHAIN_S,
        job_commit_limit=RSS_LIMIT,peak_rss_limit=RSS_LIMIT,disk_floor=DISK_FLOOR,
        extra_quota=EXTRA,old_answers_allowed=False,retry_allowed=False))
    save(run/'storage_preflight.json',budget)
    save(run/'machine.json',dict(cpu=platform.processor(),logical_cpus=os.cpu_count(),platform=platform.platform(),
        memory=memory_state(),disk_available=shutil.disk_usage(ROOT).free,python=sys.version,interpreter=sys.executable,
        venv_sha256=sha(EXPECTED_PYTHON),base_exe_sha256=sha(sys._base_executable)))
    files={n:sha(run/n) for n in ('config.json','schedule.json','machine.json','storage_preflight.json')}
    save(run/'freeze.json',dict(source_run=BASE.name,source_seal=BASE_SHA,catalog_run=CAT.name,catalog_seal=CAT_SEAL,
        sources=sources,inputs=inputs,catalogs=catfiles,files=files,gate=rel(gatepath),gate_files=gate_files))
    save(run/'prepared.json',dict(freeze_sha256=sha(run/'freeze.json'),physical_queries=0,
        preparation_elapsed_seconds=time.time()-started))
    print(run,flush=True)


def verify(run):
    require(run.parent==ROOT/'runs' and run.name.endswith('_ijepes_bounded_ref'),'Wrong reference run')
    env_check()
    require(read(run/'prepared.json')['freeze_sha256']==sha(run/'freeze.json'),'Freeze changed')
    f=read(run/'freeze.json')
    require(sha(BASE/'completion_manifest.json')==f['source_seal']==BASE_SHA and
            sha(CAT/'completion_manifest.json')==f['catalog_seal']==CAT_SEAL, 'Source/count seals changed')
    for n,h in f['sources'].items(): require(sha(ROOT/n)==sha(run/'source'/n)==h,'Changed code '+n)
    for n,h in f['inputs'].items(): require(sha(BASE/n)==sha(run/'inputs'/n)==h,'Changed physical input '+n)
    for n,h in {**f['files'],**f['catalogs']}.items(): require(sha(run/n)==h,'Changed run input '+n)
    require(check_gate(ROOT/f['gate'])==f['gate_files'],'Gate changed')
    machine=read(run/'machine.json')
    require(machine['venv_sha256']==sha(EXPECTED_PYTHON) and machine['base_exe_sha256']==sha(sys._base_executable),
            'Interpreter changed')
    require(read(run/'schedule.json')==catalog_schedule(),'Schedule mismatch')


def worker(run,index):
    spec=read(run/'schedule.json')[index]
    cell=run/spec['cell']
    launch=read(cell/'launch.json')
    while not (cell/'go.json').is_file() and time.perf_counter()<launch['deadline']:
        time.sleep(.02)
    require(read(cell/'go.json')['worker_pid']==os.getpid(),'Worker not bound to controller')
    save(cell/'worker_started.json',dict(pid=os.getpid(),parent_pid=os.getppid(),utc=datetime.now(timezone.utc).isoformat()))
    try:
        verify(run)
        from worker_ijepes_bounded_ref import run_case
        result=run_case(BASE,cell,read(run/spec['catalog']),launch['deadline']-8.)
        require(read(cell/'result.json')==result,'Worker returned different result than it saved')
    except BaseException:
        save(cell/'worker_error.json',dict(traceback=traceback.format_exc()))
        raise


def cold(run,i,s,global_deadline):
    cell=run/s['cell'];cell.mkdir(parents=True,exist_ok=False)
    save(cell/'spec.json',s)
    began=time.perf_counter();deadline=min(global_deadline,began+COLD_S)
    save(cell/'launch.json',dict(start=began,deadline=deadline,utc=datetime.now(timezone.utc).isoformat(),
                              controller_pid=os.getpid()))
    runtime={**os.environ,'__PYVENV_LAUNCHER__':str(EXPECTED_PYTHON),'PYTHONDONTWRITEBYTECODE':'1',
             'PYTHONHASHSEED':'0','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    p=None;job=None;peak=0;failure=None;reason=None
    case_storage=read(run/'storage_preflight.json')
    total_output_cap=case_storage['physical_forecast']+EXTRA
    last_size_check=0.
    try:
        require(memory_state()['available_physical_bytes']>=RSS_LIMIT,'Insufficient available memory')
        job=WinJob()
        with (cell/'stdout.log').open('xb') as out,(cell/'stderr.log').open('xb') as err:
            p=subprocess.Popen([sys._base_executable,'-B','-X','utf8',__file__,'worker',str(run),str(i)],
                cwd=ROOT,env=runtime,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
            job.attach(p)
            save(cell/'go.tmp',dict(worker_pid=p.pid))
            (cell/'go.tmp').rename(cell/'go.json')
            while True:
                done=p.poll() is not None
                peak=max(peak,peak_rss(p))
                require(peak<=RSS_LIMIT and shutil.disk_usage(ROOT).free>=DISK_FLOOR,'Memory or disk floor')
                now=time.perf_counter()
                if done or now-last_size_check>=5:
                    require(size(run)<=total_output_cap,'Run output exceeded reserved disk budget')
                    last_size_check=now
                if time.perf_counter()>=deadline:
                    reason='HARD_TIMEOUT'
                    raise TimeoutError(reason)
                if done:
                    require(p.returncode==0 and (cell/'result.json').is_file(),'Reference worker failed')
                    break
                time.sleep(min(.1,max(0.,deadline-time.perf_counter())))
    except BaseException:
        failure=traceback.format_exc()
        reason=reason or 'RESOURCE_OR_WORKER_ERROR'
    finally:
        if job is not None: job.close()
        if p is not None and p.poll() is None:
            p.kill();p.wait()
    result=read(cell/'result.json') if (cell/'result.json').exists() else {}
    row=dict(context_id=s['context_id'],cell=s['cell'],pid=p.pid if p else None,
             exit_code=p.returncode if p else None,peak_rss_bytes=peak,cold_seconds=time.perf_counter()-began,
             failure=failure,stop_reason=reason,worker_status=result.get('status'),
             result_sha256=sha(cell/'result.json') if result else None)
    save(cell/'cold_exit.json',row)
    return row


def build_dispositions(run,schedule):
    """Count all planned rows, even after an infrastructure failure. No oracle data fed to workers."""
    summary=[]
    for s in schedule:
        cell=run/s['cell']
        cat=read(run/s['catalog'])['sorted_candidate_catalog']
        rows={}
        anomalous=[]
        ledger=cell/'plans.jsonl'
        if ledger.is_file():
            with ledger.open('rb') as stream:
                for raw in stream:
                    try:
                        if not raw.endswith(b'\n'):
                            raise ValueError('unfinished ledger line')
                        row=json.loads(raw)
                        i=row['index']
                        require(type(i) is int and 0<=i<len(cat) and i not in rows and
                            row['actions_first_5']==cat[i]['actions_first_5'] and
                            row['status'] in ('PASS','FAIL','UNRESOLVED'),'Invalid ledger row')
                        rows[i]=row
                    except (ValueError,TypeError,KeyError,RuntimeError) as exc:
                        anomalous.append(str(exc));break
        recorded=[rows[i].get('status','UNRESOLVED') if i in rows else 'UNRESOLVED' for i in range(len(cat))]
        # A start or half-written log is never completion. Even provisional
        # bounds here require intact worker finalization; the external audit
        # independently decodes each physical/alias evidence row before claims.
        exit_path=cell/'cold_exit.json'
        exit_info=read(exit_path) if exit_path.exists() else {}
        trusted_receipt=(exit_info.get('exit_code')==0 and not exit_info.get('failure')
            and not anomalous and exit_info.get('worker_status') in ('CASE_COMPLETE','CASE_TIME_LIMIT'))
        effective=recorded if trusted_receipt else ['UNRESOLVED']*len(cat)
        lows=[cat[i]['normal_kwh'] for i,a in enumerate(effective) if a=='PASS']
        highs=lows+[cat[i]['normal_kwh'] for i,a in enumerate(effective) if a=='UNRESOLVED']
        summary.append(dict(context_id=s['context_id'],worker_finalized=bool(trusted_receipt),
            recorded={k:recorded.count(k) for k in ('PASS','FAIL','UNRESOLVED')},
            conservative_before_independent_audit={k:effective.count(k) for k in ('PASS','FAIL','UNRESOLVED')},
            lower_kwh=max(lows,default=None),upper_kwh=max(highs,default=None),
            journal_errors=anomalous,unresolved_indices=[i for i,a in enumerate(effective) if a=='UNRESOLVED']))
    return summary


def execute(run):
    verify(run)
    schedule=read(run/'schedule.json')
    prepared_s=read(run/'prepared.json')['preparation_elapsed_seconds']
    disk=read(run/'storage_preflight.json')
    require(shutil.disk_usage(ROOT).free>=disk['required_available'],'Disk preflight agreement changed')
    t0=time.time();start=time.perf_counter();deadline=start+DISPATCH_S-prepared_s
    require(deadline>start,'No time left for dispatch')
    save(run/'started.json',dict(pid=os.getpid(),start_epoch_seconds=t0,start_perf_counter=start,
         dispatch_deadline_epoch_seconds=t0+DISPATCH_S-prepared_s,
         chain_deadline_epoch_seconds=t0+CHAIN_S-prepared_s,audit_limit_s=AUDIT_S,
         freeze_sha256=sha(run/'freeze.json')))
    rows=[];stop=None;partial=False
    try:
        for i,s in enumerate(schedule):
            if time.perf_counter()>=deadline:
                stop='DISPATCH_DEADLINE';break
            row=cold(run,i,s,deadline)
            rows.append(row)
            print(f"c{i:02d}",row['worker_status'],row['cold_seconds'],flush=True)
            partial |= row['worker_status']=='CASE_TIME_LIMIT'
            if row['failure'] or row['worker_status'] not in ('CASE_COMPLETE','CASE_TIME_LIMIT'):
                stop=row['stop_reason'] or 'REFERENCE_NOT_CLOSED';break
    except BaseException:
        stop=traceback.format_exc()
        raise
    finally:
        dispositions=build_dispositions(run,schedule)
        save(run/'dispositions_before_audit.json',dispositions)
        status='STOPPED_INCOMPLETE' if len(rows)!=8 or stop else (
               'PARTIAL_PENDING_INDEPENDENT_AUDIT' if partial else 'COMPLETE_PENDING_INDEPENDENT_AUDIT')
        save(run/'controller_result.json',dict(status=status,exits=rows,stop=stop,
            not_dispatched=schedule[len(rows):],cold_elapsed_seconds=time.perf_counter()-start,
            chain_deadline_epoch_seconds=t0+CHAIN_S-prepared_s,audit_limit_s=AUDIT_S))
    return int(bool(stop))


def main():
    p=argparse.ArgumentParser()
    commands=p.add_subparsers(dest='action',required=True)
    commands.add_parser('prepare').add_argument('--preflight',type=Path,required=True)
    commands.add_parser('execute').add_argument('run',type=Path)
    w=commands.add_parser('worker');w.add_argument('run',type=Path);w.add_argument('index',type=int,choices=range(8))
    z=commands.add_parser('selftest');z.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.action=='prepare': prepare(a.preflight.resolve())
    elif a.action=='execute': return execute(a.run.resolve())
    elif a.action=='worker': worker(a.run.resolve(),a.index)
    else:
        begin=time.perf_counter()
        env_check();schedule=catalog_schedule();disk=storage_proof()
        save(a.output,dict(status='PASS',contexts=[s['context_id'] for s in schedule],new_physical_queries=0,
             storage=disk,source_sha256=sha(__file__),elapsed_s=time.perf_counter()-begin))
    return 0


if __name__=='__main__':
    sys.exit(main())
