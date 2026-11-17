"""Certify one frozen bounded-domain catalog using a fresh original physical service."""
import json
import math
from collections import Counter
from decimal import Decimal
from pathlib import Path
import time


def save_new(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        append(stream, value)


def append(stream, data):
    stream.write(json.dumps(data, separators=(',', ':'), allow_nan=False)+'\n')
    stream.flush()


def require(value, message):
    if not value:
        raise ValueError(message)


class LogSizeCheck:
    """Forward each unmodified ledger row, flushing an oversized row before stopping."""
    def __init__(self, original, kind, measurements):
        self.original, self.kind, self.measurements = original, kind, measurements

    def write(self, line):
        n = len(line.encode('utf8'))+1  # Windows CRLF
        if self.kind == 'service' and '"kind": "ACTUAL_STARTED"' in line[:150]:
            cap = 64 * 1024
        else:
            cap = (224 if self.kind == 'logical' else 4) * 1024
        self.measurements[self.kind] = max(self.measurements.get(self.kind,0),n)
        count = self.original.write(line)
        if n > cap:
            self.original.flush()
            raise OSError(f'Full {self.kind} evidence row retained; size {n} exceeds ceiling {cap}')
        return count

    def __getattr__(self, name):
        return getattr(self.original, name)


class SoftDeadline(Exception):
    pass


def certify_candidate(env, view, entry, index, deadline, upper, *, clock=time.perf_counter):
    """Use the full frozen certificate definition; stop a candidate at its first FAIL."""
    actions = tuple(entry['actions_first_5'])
    require(len(actions)==5 and env.horizon==30 and (env.slot_seconds,env.sample_seconds)==(60,30),
            'Unexpected schedule or sample grid')
    states = env.plan_states(env.initial, actions+(0,)*25)
    normal = math.fsum(env.service_kw(s) * env.slot_seconds/3600 for s in states[1:])
    require(math.isclose(normal, entry['normal_kwh'], abs_tol=1e-8, rel_tol=0) and
            states[-1].command==entry['terminal_command'], 'Catalog disagrees with cold graph replay')
    row = unresolved_row(entry,index,'IN_PROGRESS')
    for slot, state in enumerate(states[1:]):
        for offset in (0,30,60):
            if clock() >= deadline:
                row['reason'] = 'SOFT_DEADLINE'
                return row
            seconds = slot*60+offset
            require(view.requests < upper, 'Maximum sample requests exceeded')
            if row['first_request'] is None:
                row['first_request'] = view.requests+1
            try:
                cert = view.certify(state,seconds)
                require(cert.get('reliable') is True and cert.get('status') in ('PASS','FAIL'),
                        'Unreliable physical or logical evidence')
            except Exception as exc:
                # Retain the completed sample prefix and the failed request number.
                row.update(reason='CERTIFICATION_ERROR', error=repr(exc)[:2048], last_request=view.requests)
                return row
            row['samples'].append(cert['point_id'])
            row['last_request'] = view.requests
            if cert['status']=='FAIL':
                classes=sorted({v['constraint'] for v in cert['feedback']})
                require(classes and set(classes)<= {'voltage','line_thermal','transformer_thermal'},
                        'Reliable failure without a named voltage or thermal constraint')
                row.update(status='FAIL',reason='RELIABLE_FIRST_VIOLATION',failed_step=slot,
                           failed_time_seconds=seconds,failure_constraints=classes)
                return row
    row.update(status='PASS',reason='ALL_90_SAMPLES_PASS',certified_kwh=normal)
    return row


def unresolved_row(entry, index, reason):
    return {'index':index,'actions_first_5':entry['actions_first_5'],
            'normal_kw_slot_sum':entry['normal_kw_slot_sum'],'normal_kwh':entry['normal_kwh'],
            'status':'UNRESOLVED','certified_kwh':None,'samples':[],'failed_step':None,
            'failed_time_seconds':None,'failure_constraints':[],'first_request':None,
            'last_request':None,'reason':reason}


def partial_bounds(rows, prep):
    """Bounds in exact kW-slot-sum units. Physical lower bound requires a PASS."""
    out={}
    for group, family in [('all',rows),('switch_only',[r for r in rows if prep not in r['actions_first_5']])]:
        passed=[r for r in family if r['status']=='PASS']
        remaining=[r for r in family if r['status']=='UNRESOLVED']
        low=max((Decimal(r['normal_kw_slot_sum']) for r in passed),default=None)
        up=max([Decimal(r['normal_kw_slot_sum']) for r in remaining]+([low] if low is not None else []),
               default=None)
        out[group]={'count':len(family),'lower_kw_slot_sum':None if low is None else str(low),
                    'upper_kw_slot_sum':None if up is None else str(up),
                    'lower_kwh':None if low is None else float(low/60),
                    'upper_kwh':None if up is None else float(up/60),
                    'best_verified_indices':[r['index'] for r in passed if Decimal(r['normal_kw_slot_sum'])==low],
                    'unresolved':len(remaining),'fully_examined':not remaining,
                    'certified_optimum':bool(passed and not remaining),
                    'all_reliably_infeasible':bool(family and not passed and not remaining)}
    return out


def run_case(base, cell, catalog, deadline):
    """Entry only for the dedicated isolated cold worker. No imported call starts a runtime."""
    from dnr.chen_joint_domain import ChenEnv
    from dnr.nature34_clpu_guard_first_control import GuardFirstEvidenceService
    from run_nature34_clpu_pilot import make_env

    base, cell = Path(base),Path(cell)
    entries=catalog['sorted_candidate_catalog']
    seqs=[tuple(x['actions_first_5']) for x in entries]
    require(len(entries)==catalog['plan_count'] and seqs==sorted(set(seqs)) and
            catalog['requests_upper']==1+90*len(entries) and
            catalog['numerical_entries_upper']==3+4*catalog['unique_actual_upper'],
            'Catalog or entry ceilings changed')
    env=ChenEnv.from_env(make_env(base,catalog['spec']))
    scope=json.loads((base/'physical_scope.json').read_bytes())
    service=GuardFirstEvidenceService(base,cell,env,scope,mode='shared',
        physical_budget=catalog['unique_actual_upper'],logical_budget=catalog['unique_logical_upper'])
    return run_ledger(env,service,cell,catalog,deadline)


def run_ledger(env,service,cell,catalog,deadline,*,clock=time.perf_counter,measure_streams=True):
    entries=catalog['sorted_candidate_catalog']
    measurements, rows={},[]
    status,reason='CASE_COMPLETE',None
    with (cell/'plans.jsonl').open('x',encoding='utf8') as plans, (cell/'plan_starts.jsonl').open('x',encoding='utf8') as starts:
        try:
            if clock()>=deadline:
                raise SoftDeadline
            with service:
                if measure_streams:
                    service._events=LogSizeCheck(service._events,'service',measurements)
                with service.view('GUARD') as view:
                    if measure_streams:
                        view._logical_log=LogSizeCheck(view._logical_log,'logical',measurements)
                        view._request_log=LogSizeCheck(view._request_log,'requests',measurements)
                    initial=None
                    if clock()<deadline:
                        initial=view.certify(env.initial,0)
                        save_new(cell/'initial.json',{'state':env.initial.to_dict(),'time_seconds':0,'certificate':initial})
                        require(initial.get('reliable') is True and initial['status'] in ('PASS','FAIL'),
                                'Initial evidence unknown or unreliable')
                    if initial is None:
                        status,reason='CASE_TIME_LIMIT','SOFT_DEADLINE'
                    elif initial['status']=='FAIL':
                        status,reason='STOP_INITIAL_FAIL','INITIAL_RELIABLE_FAIL'
                    else:
                        for i,entry in enumerate(entries):
                            if clock()>=deadline:
                                status,reason='CASE_TIME_LIMIT','SOFT_DEADLINE'
                                break
                            append(starts,{'index':i,'actions_first_5':entry['actions_first_5'],
                                           'requests_before':view.requests,'actual_before':service.physical_points})
                            row=certify_candidate(env,view,entry,i,deadline,catalog['requests_upper'],clock=clock)
                            append(plans,row);rows.append(row)
                            if row['status']=='UNRESOLVED':
                                if row['reason']=='SOFT_DEADLINE':
                                    status,reason='CASE_TIME_LIMIT','SOFT_DEADLINE'
                                    break
                                raise RuntimeError(row.get('error',row['reason']))
        except SoftDeadline:
            status,reason='CASE_TIME_LIMIT','SOFT_DEADLINE'
        except Exception as exc:
            status,reason='STOP_RELIABILITY_ERROR',repr(exc)[:2048]
        for i in range(len(rows),len(entries)):
            row=unresolved_row(entries[i],i,status)
            append(plans,row);rows.append(row)

    account=service.summary()
    if status in ('CASE_COMPLETE','CASE_TIME_LIMIT','STOP_INITIAL_FAIL') and account['runtime_initializations']:
        try:
            require(account['physical_cost_complete'] and not account['cross_task_evidence_reuse'] and
                    account['physical_points']<=catalog['unique_actual_upper'] and
                    account['logical_points']<=catalog['unique_logical_upper'] and
                    account['actual_numerical_entries']==3+4*account['physical_points']<=catalog['numerical_entries_upper'],
                    'Case physical or accounting limit mismatched')
        except ValueError as exc:
            status,reason='STOP_RELIABILITY_ERROR',str(exc)
    counts=Counter(r['status'] for r in rows)
    result={'status':status,'reason':reason,'spec':catalog['spec'],
            'counts':{s:counts[s] for s in ('PASS','FAIL','UNRESOLVED')},
            'bounds':partial_bounds(rows,catalog['prep_action']),'account':account,
            'maximum_logged_row_bytes':measurements,'namespace':'GUARD','solver_training_calls':0}
    save_new(cell/'result.json',result)
    return result
