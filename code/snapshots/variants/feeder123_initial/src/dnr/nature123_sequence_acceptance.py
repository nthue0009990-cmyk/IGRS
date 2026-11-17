"""Fixed 30-minute NC123 v2 sequence acceptance, never a search or method comparison.

No DSS is imported unless the explicit execution gates pass in a supervised
worker. All-wait must certify first. A reliable limit violation stops only that
later plan; missing reliability evidence or resource interruption stops the batch.
"""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback

from .nature123_acceptance_budget import AcceptanceBudget
from .nature123_acceptance_runner import (DurableJournal, canonical_sha, certificate_valid,
    inside, read_json, read_journal, seal_run, sha, supervise, write_json)
from .nature123_clpu_environment import CLPUEnvironment
from .nature123_static_state import SparseRestorationGraph


SCHEMA = 'NC123_SEQUENCE_ACCEPTANCE_V1'
SCOPE = 'FULL_SEQUENCE_ACCEPTANCE_30MIN'
PRESPECIFIED_PLANS = (
    ('all_wait', [0]*30),
    ('legacy_s30c_hold_negative', [18,37,0,18]+[0]*26),
    ('segmented_l26_delay_5min', [22,37,0,0,0,0,22]+[0]*23),
    ('segmented_l26_delay_10min', [22,37]+[0]*9+[22]+[0]*18),
)
REQUIRED = {
    'src/dnr/nature123_sequence_acceptance.py', 'scripts/run_nature123_sequence_acceptance.py',
    'src/dnr/nature123_acceptance_runner.py', 'src/dnr/nature123_acceptance_budget.py',
    'src/dnr/nature123_clpu_physical.py', 'src/dnr/nature123_phase_available_physical.py',
    'src/dnr/nature123_clpu_environment.py', 'src/dnr/nature123_static_state.py',
    'artifacts/nature123_model_v2/model_config.json', 'artifacts/nature123_model_v2/manifest.json',
}


def _positive(value, bound, name, integer=False):
    if (type(value) not in ((int,) if integer else (int, float))
            or not math.isfinite(value) or not 0 < value <= bound):
        raise ValueError('Invalid or enlarged ' + name)


def validate_freeze(project, freeze, *, execute=False):
    """Reconstruct all 91 requests per plan, including repeated boundary samples."""
    if freeze.get('schema_version') != SCHEMA or type(freeze.get('physical_execution_allowed')) is not bool:
        raise ValueError('Explicit sequence schema and Boolean execution gate required')
    budgets = freeze.get('budgets', {})
    per, glob = budgets.get('per_plan', {}), budgets.get('global', {})
    for name, maximum in [('max_requests', 91), ('max_entries', 273), ('memory_bytes', 4*1024**3)]:
        _positive(per.get(name), maximum, name, True)
    for name, maximum in [('seconds', 240), ('poll_seconds', .1)]:
        _positive(per.get(name), maximum, name)
    for name, maximum in [('max_plans', 4), ('max_requests', 364), ('max_entries', 1092)]:
        _positive(glob.get(name), maximum, 'global ' + name, True)
    _positive(glob.get('seconds'), 960, 'global seconds')
    sources = freeze.get('source_sha256', {})
    if not isinstance(sources, dict) or not REQUIRED <= set(sources):
        raise ValueError('Incomplete frozen sequence source inventory')
    for rel, digest in sources.items():
        if sha(inside(project, rel)) != digest:
            raise ValueError('Frozen source mismatch: ' + rel)
    graph_path = freeze.get('static_manifest_path')
    if graph_path not in sources:
        raise ValueError('Static graph must be hash-bound')
    if freeze.get('fault_lines') != ['l19']:
        raise ValueError('Only the frozen l19 development fault is authorized')
    policy = freeze.get('physical_config', {})
    if policy.get('runtime_variant') != 'PHASE_AVAILABLE_V2':
        raise ValueError('Sequence acceptance requires the approved v2 runtime')
    config = read_json(inside(project, 'artifacts/nature123_model_v2/model_config.json'))
    if freeze.get('model_fingerprint') != config.get('model_fingerprint'):
        raise ValueError('Sequence model fingerprint mismatch')
    condition = freeze['condition']
    op = condition.get('operating_config', {})
    if (type(op.get('algorithm')) is not int or op['algorithm'] != 0 or op.get('algorithm_name') != 'Normal'
            or op.get('voltage_policy_pu') != policy.get('voltage_limits_pu')
            or op.get('frequency_hz') != policy.get('frequency_hz')):
        raise ValueError('Only unchanged v2 policy and Normal algorithm are allowed')
    plans = freeze.get('plans')
    if not isinstance(plans, list) or len(plans) != 4 or len(plans) > glob['max_plans']:
        raise ValueError('Exactly the four authorized plans must be retained')
    if [(p.get('plan_id'),p.get('actions')) for p in plans] != list(PRESPECIFIED_PLANS):
        raise ValueError('Prespecified plan identity/order/actions changed')
    if plans[0].get('actions') != [0]*30:
        raise ValueError('The first plan must be the complete all-wait fallback')
    graph = SparseRestorationGraph(read_json(inside(project, graph_path)), ('l19',))
    env = CLPUEnvironment(graph, model_fingerprint=freeze['model_fingerprint'],
                          condition=condition, event=freeze['event'],
                          horizon=30, slot_seconds=60, sample_seconds=30)
    ids = set()
    for plan in plans:
        name, actions = plan.get('plan_id'), plan.get('actions')
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', name) or name in ids:
            raise ValueError('Unique safe plan identity required')
        ids.add(name)
        if not isinstance(plan.get('label'),str) or not plan['label']:
            raise ValueError('Each fixed plan requires a descriptive label')
        if not isinstance(actions, list) or len(actions) != 30:
            raise ValueError('A complete 30-action horizon is required')
        states = env.plan_states(env.initial, actions)
        expected = [env.snapshot_request(env.initial, 0)]
        expected += [env.snapshot_request(state, (state.step-1)*60 + offset)
                     for state in states[1:] for offset in (0, 30, 60)]
        if plan.get('requests') != expected:
            raise ValueError('Plan request/history/sampling replay differs: ' + name)
        if len(expected) != 91:
            raise ValueError('Complete 91-sample schedule required')
        energy = math.fsum(env.service_kw(state)/60 for state in states[1:])
        saved = plan.get('theoretical_nominal_energy_kwh')
        if type(saved) not in (int, float) or not math.isfinite(saved) or not math.isclose(saved, energy, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError('Uncertified theoretical energy differs from action replay')
    if execute:
        if freeze['physical_execution_allowed'] is not True:
            raise ValueError('Sequence physical execution gate is closed')
        if op.get('physical_execution_allowed') is not True or op.get('status') != 'FROZEN_FOR_ACCEPTANCE':
            raise ValueError('Request-level physical policy must be explicitly frozen')
        ready = freeze.get('readiness_checks', {})
        if not ready or any(v != 'PASS' for v in ready.values()):
            raise ValueError('Software/parameter readiness remains incomplete')
        auth = freeze.get('authorization', {})
        if auth.get('scope') != SCOPE or not auth.get('record'):
            raise ValueError('Full-sequence authorization scope must be recorded')
    return plans


def worker(project, run_dir, freeze, plan, *, runtime_factory=None):
    if runtime_factory is None and freeze.get('physical_execution_allowed') is not True:
        raise ValueError('Worker cannot enable a closed sequence freeze')
    run = Path(run_dir)
    journal = DurableJournal(run/'events.jsonl')
    per = freeze['budgets']['per_plan']
    budget = AcceptanceBudget(max_requests=per['max_requests'], max_entries=per['max_entries'],
                              seconds=per['seconds'], emit=journal)
    synthetic = runtime_factory is not None
    current, status, error, count = None, 'UNKNOWN', None, 0
    try:
        if runtime_factory is None:
            from .nature123_phase_available_physical import Nature123PhaseAvailablePhysicalRuntime
            runtime_factory = Nature123PhaseAvailablePhysicalRuntime
        journal({'event':'worker_started', 'pid':os.getpid(), 'ppid':os.getppid(),
                 'test_runtime_injected':synthetic, 'plan_id':plan['plan_id'], 'executable':sys.executable})
        runtime = runtime_factory(project, budget=budget, physical_config=freeze['physical_config'], enable_physical=True)
        for index, request in enumerate(plan['requests']):
            current = 'point_%02d' % index
            journal({'event':'request_started', 'point_id':current, 'request_sha256':request['request_sha256']})
            cert = runtime.evaluate(request, point_id=current)
            path = run/'points'/(current+'.json')
            write_json(path, cert)
            valid = certificate_valid(cert, request, current)
            journal({'event':'request_completed', 'point_id':current,
                     'request_sha256':request['request_sha256'], 'status':cert.get('status'),
                     'reliable_certificate':valid, 'result_sha256':sha(path)})
            count += 1
            if not valid:
                status, error = 'UNKNOWN', 'Unreliable or unbound physical certificate'
                break
            if cert['status'] == 'FAIL':
                status = 'RELIABLE_FAIL'
                break
            current = None
        else:
            status = 'COMPLETE'
    except BaseException as exc:
        status, error = 'UNKNOWN', repr(exc)
        exception = {'event':'worker_exception', 'point_id':current, 'error':error,
                     'traceback':traceback.format_exc(), 'budget':budget.snapshot()}
        write_json(run/'worker_exception.json', exception)
        journal(exception)
    finally:
        write_json(run/'worker_final.json', {'status':status, 'synthetic_test_only':synthetic,
            'physical_certified':status == 'COMPLETE' and not synthetic,
            'energy_kwh':plan['theoretical_nominal_energy_kwh'] if status == 'COMPLETE' and not synthetic else None,
            'theoretical_nominal_energy_kwh':plan['theoretical_nominal_energy_kwh'],
            'theoretical_energy_certified':status == 'COMPLETE' and not synthetic,
            'completed_results':count, 'budget':budget.snapshot(), 'error':error,
            'automatic_retries':0, 'evidence_kind':'SIMULATED' if synthetic else 'PHYSICAL'})
    return 0 if status in ('COMPLETE', 'RELIABLE_FAIL') else 2


def audit_journal(events, requests, per):
    """Sequence-specific budget audit; does not weaken the entry runner's 16/256 cap."""
    expected = {'point_%02d' % i:q['request_sha256'] for i,q in enumerate(requests)}
    started, reserved, completed, entries, errors = {}, {}, {}, {}, []
    for e in events:
        kind = e.get('event')
        if kind in ('request_started', 'request_completed', 'request_reserved'):
            point = e.get('request_id' if kind == 'request_reserved' else 'point_id')
            target = {'request_started':started, 'request_completed':completed, 'request_reserved':reserved}[kind]
            if point not in expected or e.get('request_sha256') != expected.get(point) or point in target:
                errors.append('Duplicate/unbound '+kind)
            if kind == 'request_reserved' and (point not in started or e.get('request_number') != len(reserved)+1):
                errors.append('Request reservation lacks ordered start')
            if kind == 'request_completed' and point not in reserved:
                errors.append('Completion before request reservation')
            target[point] = e
        elif kind == 'numerical_entries_reserved':
            token, count, state, point = e.get('token'), e.get('count'), e.get('completion'), e.get('request_id')
            if (type(token) is not int or token < 0 or type(count) is not int or count < 1
                    or point not in reserved):
                errors.append('Malformed/unbound sequence numerical entry')
                continue
            if state == 'NOT_REPORTED':
                if token in entries or token != len(entries):
                    errors.append('Duplicate/out-of-order numerical reservation')
                else:
                    entries[token] = e
            elif state in ('API_RETURNED_NOT_CONVERGENCE', 'UNKNOWN_NO_REFUND'):
                old = entries.get(token)
                if (old is None or old['completion'] != 'NOT_REPORTED' or
                        any(old.get(k) != e.get(k) for k in ('count','request_id','phase','operation'))):
                    errors.append('Duplicate/unbound numerical completion')
                else:
                    entries[token] = e
            else:
                errors.append('Unknown numerical completion state')
    if len(reserved) > per['max_requests'] or sum(e['count'] for e in entries.values()) > per['max_entries']:
        errors.append('Per-plan budget exceeded')
    for point in completed:
        point_entries = [e for e in entries.values() if e['request_id'] == point]
        if completed[point].get('reliable_certificate') is True and (
                sum(e['count'] for e in point_entries) != 3 or
                any(e['completion'] != 'API_RETURNED_NOT_CONVERGENCE' for e in point_entries)):
            errors.append('Completed request lacks all three charged returned entries')
    if list(started) != list(expected)[:len(started)] or list(reserved) != list(expected)[:len(reserved)]:
        errors.append('Requests were skipped/reordered')
    return started, reserved, completed, entries, errors


def reconcile_plan(run_dir, plan, observed, per):
    run = Path(run_dir)
    events, errors = read_journal(run/'events.jsonl')
    starts, reserved, completions, entries, extra = audit_journal(events, plan['requests'], per)
    errors += extra
    synthetic = any(e.get('test_runtime_injected') is True for e in events if e.get('event') == 'worker_started')
    results = []
    for index, request in enumerate(plan['requests']):
        point = 'point_%02d' % index
        path = run/'points'/(point+'.json')
        status = 'UNKNOWN' if point in starts else 'NOT_RUN'
        detail = 'No bound completed certificate'
        try:
            if path.exists():
                cert = read_json(path)
                record=completions.get(point,{})
                if (not errors and record.get('result_sha256') == sha(path)
                        and record.get('reliable_certificate') is True and record.get('status')==cert.get('status')
                        and certificate_valid(cert,request,point)):
                    status, detail = cert['status'], 'Saved identity, checks and three numerical entries bound'
                else:
                    status = 'UNKNOWN'
        except (ValueError,TypeError,OSError) as exc:
            status, detail = 'UNKNOWN', repr(exc)
        results.append({'index':index,'point_id':point,'status':status,'detail':detail,
            'request_sha256':request['request_sha256'],'time_seconds':request['time_seconds'],
            'step':request['state']['step']})
    statuses = [row['status'] for row in results]
    good_exit = observed.get('returncode') == 0 and observed.get('termination_reason') is None
    status = 'UNKNOWN'
    if not errors and good_exit:
        if statuses == ['PASS']*91:
            status = 'COMPLETE'
        elif 'FAIL' in statuses:
            index = statuses.index('FAIL')
            if statuses == ['PASS']*index + ['FAIL'] + ['NOT_RUN']*(90-index):
                status = 'RELIABLE_FAIL'
    complete = status == 'COMPLETE' and not synthetic
    result = {'plan_id':plan['plan_id'],'label':plan['label'],'status':status,
        'evidence_kind':'SIMULATED' if synthetic else 'PHYSICAL','synthetic_test_only':synthetic,
        'physical_certified':complete,'certified_energy_kwh':plan['theoretical_nominal_energy_kwh'] if complete else None,
        'energy_kwh':plan['theoretical_nominal_energy_kwh'] if complete else None,
        'theoretical_nominal_energy_kwh':plan['theoretical_nominal_energy_kwh'],
        'theoretical_energy_certified':complete,'energy_scope':'NORMAL_DEMAND_NOT_CLPU_SURGE',
        'requests':results,'counts':{s:statuses.count(s) for s in ('PASS','FAIL','UNKNOWN','NOT_RUN')},
        'journal_errors':errors,'reserved_requests':len(reserved),
        'reserved_entries':sum(e['count'] for e in entries.values()),
        'unreported_entries':sum(e['count'] for e in entries.values() if e['completion']=='NOT_REPORTED'),
        'automatic_retries':0,'training':0,'method_comparison':False}
    if synthetic:
        result['status'] = 'SIMULATED_'+status
    write_json(run/'result.json',result)
    return result


def initial_plan_status(plan, reason='NOT_STARTED'):
    return {'plan_id':plan['plan_id'],'label':plan['label'],'status':'NOT_RUN','reason':reason,
            'physical_certified':False,'certified_energy_kwh':None,
            'energy_kwh':None,
            'theoretical_nominal_energy_kwh':plan['theoretical_nominal_energy_kwh'],
            'theoretical_energy_certified':False,
            'counts':{'PASS':0,'FAIL':0,'UNKNOWN':0,'NOT_RUN':91},
            'reserved_requests':0,'reserved_entries':0,
            'requests':[{'index':i,'point_id':'point_%02d'%i,'status':'NOT_RUN',
                'request_sha256':q['request_sha256'],'time_seconds':q['time_seconds'],'step':q['state']['step']}
                for i,q in enumerate(plan['requests'])]}


def run_batch(project, run_dir, freeze, *, launch_plan=None):
    """Cold plans run serially. Test launcher injection cannot produce physical success."""
    root, run = Path(project).resolve(), Path(run_dir)
    per, glob = freeze['budgets']['per_plan'], freeze['budgets']['global']
    synthetic = launch_plan is not None
    if not synthetic and freeze.get('physical_execution_allowed') is not True:
        raise ValueError('Batch cannot enable a closed freeze')
    results = [initial_plan_status(p) for p in freeze['plans']]
    journal = DurableJournal(run/'batch_events.jsonl')
    started = time.monotonic()
    stop = None
    requests_used = entries_used = 0
    resource_counts_known = True
    for index, plan in enumerate(freeze['plans']):
        remaining = glob['seconds']-(time.monotonic()-started)
        if (remaining <= 0 or requests_used+per['max_requests'] > glob['max_requests']
                or entries_used+per['max_entries'] > glob['max_entries']):
            stop = 'GLOBAL_RESOURCE_BOUND'
            break
        plan_run = run/'plans'/plan['plan_id']
        plan_run.mkdir(parents=True)
        write_json(plan_run/'plan.json',plan)
        journal({'event':'plan_started','plan_id':plan['plan_id'],'order':index})
        try:
            if launch_plan is None:
                exe = root/'.venv/Scripts/python.exe'
                release = plan_run/'supervisor_release.json'
                command = [str(exe),str(run/'source/scripts/run_nature123_sequence_acceptance.py'),
                    str(run/'freeze.json'),'--project',str(root),'--execute','--worker',
                    '--plan-id',plan['plan_id'],'--run-dir',str(plan_run),'--release-file',str(release)]
                observed = supervise(command,cwd=plan_run,run_dir=plan_run,seconds=min(per['seconds'],remaining),
                    memory_bytes=per['memory_bytes'],poll_seconds=per['poll_seconds'],
                    release_file=release,identity_file=plan_run/'worker_identity.json')
                result = reconcile_plan(plan_run,plan,observed,per)
            else:
                result = launch_plan(plan,plan_run,min(per['seconds'],remaining))
        except BaseException as exc:
            write_json(plan_run/'launch_or_audit_exception.json',{'error':repr(exc),'traceback':traceback.format_exc()})
            result = initial_plan_status(plan,'LAUNCH_OR_AUDIT_EXCEPTION')
            result.update(status='SIMULATED_UNKNOWN' if synthetic else 'UNKNOWN',error=repr(exc),
                resource_counts_status='UNKNOWN_CONSERVATIVELY_ACCOUNT_FULL_PLAN',
                reserved_requests=None,reserved_entries=None,
                counts={'PASS':0,'FAIL':0,'UNKNOWN':91,'NOT_RUN':0})
            for row in result['requests']:
                row['status']='UNKNOWN'
            write_json(plan_run/'result.json',result)
        results[index] = result
        known = type(result['reserved_requests']) is int and type(result['reserved_entries']) is int
        resource_counts_known &= known
        requests_used += result['reserved_requests'] if known else per['max_requests']
        entries_used += result['reserved_entries'] if known else per['max_entries']
        seal_run(plan_run)
        journal({'event':'plan_finished','plan_id':plan['plan_id'],'status':result['status'],
                 'result_sha256':sha(plan_run/'result.json')})
        complete = result['status'] == ('SIMULATED_COMPLETE' if synthetic else 'COMPLETE')
        failed = result['status'] == ('SIMULATED_RELIABLE_FAIL' if synthetic else 'RELIABLE_FAIL')
        if index == 0 and not complete:
            stop = 'ALL_WAIT_NOT_COMPLETELY_CERTIFIED'
            break
        if not (complete or failed):
            stop = 'RELIABILITY_OR_RESOURCE_STOP'
            break
    for row in results:
        if row['status'] == 'NOT_RUN':
            row['reason'] = stop or 'NOT_STARTED'
    result = {'status':'BATCH_FINISHED' if stop is None else 'BATCH_STOPPED',
        'stop_reason':stop,'plans':results,'reserved_requests':requests_used if resource_counts_known else None,
        'reserved_entries':entries_used if resource_counts_known else None,
        'resource_counts_known':resource_counts_known,'conservative_budget_requests_accounted':requests_used,
        'conservative_budget_entries_accounted':entries_used,
        'wall_seconds_observed':time.monotonic()-started,'evidence_kind':'SIMULATED' if synthetic else 'PHYSICAL',
        'synthetic_test_only':synthetic,'parallel_workers':1,'automatic_retries':0,'cache_reuse':False,
        'training':0,'method_comparison':False,'scope':SCOPE}
    if synthetic:
        result['status'] = 'SIMULATED_'+result['status']
        for row in result['plans']:
            row.update(physical_certified=False,certified_energy_kwh=None,energy_kwh=None,theoretical_energy_certified=False)
    write_json(run/'result.json',result)
    return result


def prepare_run(project,freeze_path,*,execute=False):
    project,freeze_path = Path(project).resolve(),Path(freeze_path).resolve()
    freeze = read_json(freeze_path)
    plans = validate_freeze(project,freeze,execute=execute)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run = project/'runs'/(stamp+('_nc123_sequence_acceptance' if execute else '_nc123_sequence_dryrun'))
    run.mkdir()
    (run/'freeze.json').write_bytes(freeze_path.read_bytes())
    for relative,digest in freeze['source_sha256'].items():
        dest=run/'source'/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(inside(project,relative).read_bytes())
        if sha(dest)!=digest:
            raise ValueError('Snapshot changed while copying '+relative)
    write_json(run/'environment.json',{'python':sys.version,'executable':sys.executable,'seed':freeze.get('seed'),
        'dependencies':sorted((d.metadata.get('Name','UNKNOWN'),d.version) for d in importlib.metadata.distributions()),
        'freeze_sha256':sha(freeze_path),'mode':'EXECUTE' if execute else 'DRY_RUN'})
    if not execute:
        write_json(run/'result.json',{'status':'PASS_SEQUENCE_DRY_RUN_ONLY','plans':[initial_plan_status(p) for p in plans],
            'new_physical_requests':0,'numerical_entries':0,'training':0,'all_91_requests_per_plan_replayed':True})
    return run,freeze


def main(argv=None):
    import argparse
    import json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('freeze',type=Path)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--plan-id',help=argparse.SUPPRESS)
    parser.add_argument('--run-dir',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--release-file',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.worker:
        if not args.execute or not args.plan_id or args.run_dir is None or args.release_file is None:
            parser.error('Worker requires explicit execution, selected plan, and supervisor handshake')
        write_json(args.run_dir/'worker_identity.json',{'pid':os.getpid(),'ppid':os.getppid(),'executable':sys.executable})
        deadline=time.monotonic()+10
        while not args.release_file.exists():
            if time.monotonic()>=deadline:
                raise RuntimeError('Supervisor did not release worker; no physical engine initialized')
            time.sleep(.01)
        freeze=read_json(args.freeze)
        plans=validate_freeze(args.project,freeze,execute=True)
        plan=next((p for p in plans if p['plan_id']==args.plan_id),None)
        if plan is None:
            raise ValueError('Unknown frozen plan identity')
        return worker(args.project,args.run_dir,freeze,plan)
    if Path(sys.executable).resolve()!=(args.project/'.venv/Scripts/python.exe').resolve():
        raise ValueError('Use the project .venv interpreter')
    run,freeze=prepare_run(args.project,args.freeze,execute=args.execute)
    result=run_batch(args.project,run,freeze) if args.execute else read_json(run/'result.json')
    seal_run(run)
    print(json.dumps({'run':str(run),'status':result['status'],
        'plan_statuses':[{k:p[k] for k in ('plan_id','status','certified_energy_kwh')} for p in result['plans']]},ensure_ascii=False))
    return 0 if not args.execute or result['status']=='BATCH_FINISHED' else 2
