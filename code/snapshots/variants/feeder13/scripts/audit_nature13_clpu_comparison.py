"""Solver-free saved-evidence and exact-search replay for NC13 comparison."""
from collections import Counter
from copy import deepcopy
import gzip
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from dnr.nature13_clpu_evidence import digest, physical_input, file_sha as sha, replay_certificate, save
from dnr.nature13_clpu_environment import DynamicState, CLPUBudgetExhausted


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def need(condition, message):
    if not condition:
        raise ValueError(message)


def gzread(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def gzlines(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def normalized(value):
    # Only measured computation durations are excluded from deterministic replay.
    volatile = {'elapsed_seconds', 'stage_generation_seconds'}
    if isinstance(value, dict):
        return {k: normalized(v) for k,v in value.items() if k not in volatile}
    if isinstance(value, (tuple, list)):
        return [normalized(v) for v in value]
    return value


class ReplayView:
    def __init__(self, service, name):
        self.service, self.name, self.env = service, name, service.env
        self.rows = service.requests[name]
        self.index = self.points = self.requests = self._physical_points = 0
        self.hits = self.reuses = self.cross_reuses = 0
        self.closed = False
        self.budget = 256
        self.statuses = Counter()

    @property
    def physical_points(self):
        return self._physical_points

    def __enter__(self):
        return self

    def certify(self, state, seconds):
        need(self.index < len(self.rows), 'Search replay requested extra evidence')
        row = self.rows[self.index]
        need(row['request_id'] == self.index and row['state'] == state.to_dict()
             and row['time_seconds'] == float(seconds), 'Original search request sequence differs')
        self.index += 1
        self.requests += 1
        if row['kind'] == 'BUDGET':
            need(self.points >= 256 or self.service.physical_count >= 512, 'Unjustified budget stop')
            raise CLPUBudgetExhausted('Saved bounded query stop')
        need(row['kind'] in ('ACTUAL','REUSE','HIT'), 'Error entry in complete task')
        if row['kind'] != 'HIT':
            self.points += 1
            self.statuses[row['certificate']['status']] += 1
        else:
            self.hits += 1
        if row['kind'] == 'ACTUAL':
            self._physical_points += 1
            self.service.physical_count += 1
        if row['kind'] == 'REUSE':
            self.reuses += 1
            self.cross_reuses += int(row['cross_view_reuse'])
        need(self.points == row['logical_points_after'] and self.physical_points == row['physical_points_after'], 'Replay counters differ')
        return deepcopy(row['certificate'])

    def summary(self):
        return {'view': self.name, 'closed': self.closed, 'points': self.points, 'logical_points': self.points,
            'physical_points': self.physical_points, 'requests': self.requests, 'semantic_hits': self.hits,
            'electrical_reuses': self.reuses, 'cross_view_reuses': self.cross_reuses, 'statuses': dict(self.statuses),
            'logical_budget': self.budget}

    def __exit__(self, typ, value, tb):
        self.closed = True
        need(self.index == len(self.rows), 'Unused saved requests after original search replay')
        return False


class ReplayService:
    def __init__(self, env, order, requests):
        self.env, self.phase_order, self.requests = env, order, requests
        self.views, self.physical_count = {}, 0

    def view(self, name):
        need(name == self.phase_order[len(self.views)], 'Replay phase order differs')
        view = ReplayView(self, name)
        self.views[name] = view
        return view


def audit_task(run, spec):
    from prepare_nature13_clpu_static import build_environment
    from dnr.nature13_clpu_physical import check_nature13_constraints, verify_nature13_isolation, MODEL_SCOPE
    from dnr.restoration_ac_audit import audit_ac_snapshot
    from dnr.nature34_operating_physical import verify_unchanged_attributes
    from dnr.nature13_clpu_routes import run_routes
    from audit_nature13_clpu_acceptance import _saved_equal
    cell = run/'tasks'/f"{spec['task_id']:03d}"
    controller = read(cell/'controller_result.json')
    need(controller['returncode'] == 0 and controller['timed_out'] is False, 'Task did not complete')
    for name, h in read(cell/'worker_manifest.json').items():
        need(sha(cell/name) == h, 'Worker file altered: '+name)
    need(read(cell/'spec.json') == spec, 'Task spec differs')
    config = read(run/'design/physical_config.json')
    metadata = read(cell/'metadata.json')
    scope = read(run/'physical_scope.json')
    env = build_environment(ROOT, 'line.'+spec['scenario_id'].removeprefix('nc13_'), spec['event_id'])
    rows = {int(p.name[:4]): gzread(p) for p in sorted((cell/'physical').glob('????.json.gz'))}
    need(sorted(rows) == list(range(len(rows))) and len(rows) <= 512, 'Physical IDs/budget differ')
    classifications = Counter()
    for qid, row in rows.items():
        request = read(cell/'physical'/f'{qid:04d}_request.json')
        need(row['query_id'] == qid and row['scope'] == MODEL_SCOPE and row['algorithm'] == 0, 'Physical scope/ID differs')
        for key in ('action','alpha','algorithm','time_minutes','state_identity'):
            need(row[key] == request[key], 'Actual physical request differs: '+key)
        need(row['fault_lines'] == request['case']['fault_lines'] and row['case'] == request['case']['name']
             and row['clpu_multipliers'] == request['multipliers'], 'Fault/multiplier mismatch')
        merged = {**metadata, **row['metadata_override']}
        raw = row['raw_outputs']
        checks = {'known_constraints': check_nature13_constraints(raw, merged, **config['constraint_tolerances']),
            'isolation': verify_nature13_isolation(raw, merged),
            'ac_audit': audit_ac_snapshot(raw, merged, **config['ac_audit_tolerances'])}
        for name, value in checks.items():
            need(_saved_equal(row[name], value), 'Saved physical classification differs: '+name)
        certificate = replay_certificate(row, request, metadata, config, 0)
        classifications[certificate['status']] += 1
        before, after = row['load_readback']['before'], row['load_readback']['postsolve']
        need(set(before) == set(after) == set(metadata['load_reference']) == set(request['multipliers'])
             and len(after) == 15, 'Full load readback absent')
        verify_unchanged_attributes(before, after)
        for name, ref in metadata['load_reference'].items():
            for key in ('nominal_kw','nominal_kvar'):
                need(math.isclose(before[name][key], ref[key], rel_tol=1e-12, abs_tol=1e-8)
                     and math.isclose(after[name][key], ref[key]*request['multipliers'][name], rel_tol=1e-12, abs_tol=1e-8), 'Load P/Q readback differs')
        generators = {e['name'].lower():e for e in raw['elements'] if e['name'].lower().startswith('generator.')}
        need(set(generators) == {n.lower() for n in metadata['generator_reference']}, 'DER inventory differs')
        for name, ref in metadata['generator_reference'].items():
            point = generators[name.lower()]['operating_setpoint']
            need(math.isclose(point['kw'], .375*ref['available_p_kw'], rel_tol=1e-12, abs_tol=1e-8)
                 and math.isclose(point['kvar'], .375*ref['reference_q_kvar'], rel_tol=1e-12, abs_tol=1e-8), 'DER readback differs')
        need(row['nominal_load_ledger']['denominator_nominal_kw'] == 3466., 'Energy denominator changed')

    events = [json.loads(s) for s in (cell/'numerical_events.jsonl').read_text(encoding='utf-8').splitlines()]
    starts, returns = {}, {}
    for event in events:
        if event['event'] == 'call_started':
            need(event['call_id'] not in starts, 'Repeated numerical ID')
            starts[event['call_id']] = event
        if event['event'] in ('call_returned','call_failed'):
            need(event['call_id'] in starts and event['call_id'] not in returns, 'Bad numerical return binding')
            need(event['counts'] == starts[event['call_id']]['counts']
                 and event['context'] == starts[event['call_id']]['context'], 'Numerical context differs')
            returns[event['call_id']] = event
    need(starts.keys() == returns.keys() and all(e['event']=='call_returned' for e in returns.values()), 'Unfinished numerical entry')
    actual_entries = 0
    for event in starts.values():
        counts = event['counts']
        n = sum(counts[k] for k in ('python_solve_commands','script_solve_commands','text_solve_commands','calcvoltagebases_entries'))
        need(n == counts['numerical_solution_entries'], 'Numerical accounting arithmetic differs')
        if counts['compile_commands']:
            need((counts['script_solve_commands'],counts['calcvoltagebases_entries']) == (1,1), 'Hidden Compile entries differ')
        actual_entries += n
    need(actual_entries == 3+14*len(rows) and actual_entries <= 7171, 'Full initialization/query accounting differs')
    need([e['context']['query_id'] for e in events if e['event']=='query_started'] == list(rows), 'Physical request ledger differs')
    for qid, row in rows.items():
        own = [e for e in returns.values() if e['context'].get('query_id') == qid]
        for key, value in row['actual_call_counts'].items():
            need(sum(e['counts'][key] for e in own) == value, 'Per-point call accounting differs')

    logs, pool, source_views = {}, {}, {}
    physical_seen = 0
    total_logical = 0
    for phase in spec['phase_order']:
        logs[phase] = gzlines(cell/phase/'requests.jsonl.gz')
        semantics = {}
        phase_actual = 0
        for index, entry in enumerate(logs[phase]):
            state = DynamicState(**entry['state'])
            request = env.snapshot_request(state, entry['time_seconds'], algorithm=0)
            semantic = digest(request['state_identity'])
            key = digest({'scope': scope, 'input': physical_input(request)})
            need(entry['request_id'] == index and entry['semantic_sha256'] == semantic
                 and entry['physical_input_sha256'] == key, 'Logical request identity differs')
            kind = entry['kind']
            if kind == 'BUDGET':
                need(semantic not in semantics and (len(semantics)>=256 or physical_seen>=512), 'Unjustified logical/physical budget stop')
                continue
            need(kind in ('ACTUAL','REUSE','HIT'), 'Error request in completed task')
            qid = entry['physical_point_id']
            if kind == 'HIT':
                need(semantic in semantics and entry['certificate'] == semantics[semantic][0]
                     and qid == semantics[semantic][1], 'Semantic hit differs')
            else:
                need(semantic not in semantics and len(semantics)<256, 'Duplicate/new logical budget violation')
                if kind == 'ACTUAL':
                    need(key not in pool and qid == physical_seen, 'Duplicate/noncausal physical query')
                    pool[key] = qid; source_views[qid] = phase
                    physical_seen += 1; phase_actual += 1
                    need(rows[qid]['state_identity'] == request['state_identity'], 'Actual source identity differs')
                else:
                    need(key in pool and pool[key] == qid, 'Reused evidence not previously paid')
                need(entry['source_view'] == source_views[qid]
                     and entry['source_sha256'] == sha(cell/'physical'/f'{qid:04d}.json.gz'), 'Source evidence binding differs')
                need(entry['cross_view_reuse'] == (kind=='REUSE' and source_views[qid]!=phase), 'Cross-view attribution differs')
                certificate = replay_certificate(rows[qid], request, metadata, config, len(semantics))
                need(certificate == entry['certificate'], 'Current-identity certificate differs')
                semantics[semantic] = (certificate, qid)
            need(entry['logical_points_after'] == len(semantics) and entry['physical_points_after']==phase_actual, 'Logical bookkeeping differs')
        total_logical += len(semantics)
    need(physical_seen == len(rows), 'Unaccounted physical results')

    ctx = read(run/'design'/spec['candidate_context_file'])
    replay = ReplayService(env, spec['phase_order'], logs)
    journal = gzlines(cell/'search_journal.jsonl.gz')
    cursor = 0
    def emit(phase, kind, value):
        nonlocal cursor
        need(cursor < len(journal), 'Extra search decision/plan')
        need(normalized(journal[cursor]) == normalized({'phase': phase, 'kind': kind, 'value': value}), 'Original search decision/plan differs')
        cursor += 1
    def write_phase(phase, result):
        need(normalized(result) == normalized(read(cell/phase/'phase_result.json')), 'Original phase replay differs: '+phase)
        return sha(cell/phase/'phase_result.json')
    phases, merged = run_routes(env, replay, read(run/'design/branch_config.json'), spec['arm'],
                               ctx['candidate_generation'], write_phase, emit)
    need(cursor == len(journal), 'Unused saved search journal')
    final = read(cell/'result.json')
    for key, value in merged.items():
        need(normalized(final[key]) == normalized(value), 'Final merge differs: '+key)
    energy = merged['best_plan']['validation']['energy_kwh']
    need(final['energy_kwh'] == energy and final['quality_pct'] == 100*energy/(3466*.5), 'Final quality differs')
    need(final['guard_energy_kwh'] == phases['GUARD']['best_plan']['validation']['energy_kwh'] <= energy, 'Full Beam floor lost')
    need(final['actual_physical_points'] == len(rows) and final['numerical_entries'] == actual_entries
         and final['logical_points'] == total_logical, 'Final cost differs')
    service = read(cell/'service_summary.json')
    need(service == final['service'] and service['closed'] and not service['poisoned'], 'Service did not finish')
    need(service['budget']['queries'] == len(rows) and service['budget']['returned_numerical_entries'] == actual_entries, 'Service budget differs')
    need(controller['full_cold_seconds'] >= final['worker_elapsed_seconds']
         and not controller['timed_out'], 'Complete cold time does not include worker')
    return {'task_id': spec['task_id'], 'context_id': spec['context_id'], 'arm': spec['arm'],
        'status': 'VERIFIED_COMPLETE_TASK', 'energy_kwh': energy, 'quality_pct': final['quality_pct'],
        'guard_energy_kwh': final['guard_energy_kwh'], 'actual_physical_points': len(rows),
        'logical_points': total_logical, 'numerical_entries': actual_entries,
        'full_cold_seconds': controller['full_cold_seconds'], 'physical_statuses': dict(classifications),
        'final_actions': merged['best_plan']['actions'], 'beam_floor_verified': True,
        'new_powerflow_calls': 0, 'original_search_replay': True}


def preflight(run):
    from run_nature13_clpu_comparison import verify, DESIGN_SHA
    from prepare_nature13_clpu_static import build_environment
    from dnr.nature13_clpu_staged_pickup import generate_staged_pickup_candidates
    verify(run)
    need(not (run/'execution_started.json').exists(), 'Already started')
    schedule = read(run/'design/schedule.json')
    need(len(schedule)==18 and len({(r['context_id'],r['arm']) for r in schedule})==18, '18-task balance differs')
    contexts = {}
    for spec in schedule:
        need(spec['phase_order'] == (['EARLY','GUARD'] if spec['arm']=='E' else ['GUARD','EARLY']), 'Phase order differs')
        if spec['context_id'] not in contexts:
            env = build_environment(ROOT, 'line.'+spec['scenario_id'].removeprefix('nc13_'),spec['event_id'])
            expected = read(run/'design'/spec['candidate_context_file'])['candidate_generation']
            need(generate_staged_pickup_candidates(env)==expected, 'Original pool differs')
            contexts[spec['context_id']] = expected['counts']['unique_legal_candidates']
    need(len(contexts)==6 and all(n==12 for n in contexts.values()), 'Six original candidate pools required')
    need(not any(n=='dss' or n.startswith(('dss.','opendssdirect','Environments')) for n in sys.modules), 'Preflight loaded numerical engine')
    return {'status':'PASS','prepared_sha256':sha(run/'prepared.json'), 'design_sha256':DESIGN_SHA,
            'contexts':contexts,'new_powerflow_calls':0,'training_steps':0}


def audit_run(run, output):
    from run_nature13_clpu_comparison import verify
    verify(run)
    for name, h in read(run/'execution_manifest.json').items():
        need(sha(run/name)==h, 'Execution seal differs: '+name)
    output.mkdir(exist_ok=False)
    reports, errors = [], []
    for spec in read(run/'design/schedule.json'):
        cell = run/'tasks'/f"{spec['task_id']:03d}"
        if not (cell/'controller_result.json').exists():
            reports.append({**spec,'status':'NOT_RUN'}); continue
        try:
            record = audit_task(run, spec)
        except BaseException as exc:
            import traceback
            record = {**spec,'status':'AUDIT_ERROR_OR_INCOMPLETE_TASK','error':repr(exc),'traceback':traceback.format_exc()}
            errors.append({'task_id':spec['task_id'],'error':repr(exc)})
        reports.append(record)
        save(output/f"{spec['task_id']:03d}.json",record)
        print(json.dumps({'audit_task':spec['task_id'],'status':record['status']},ensure_ascii=False),flush=True)
    complete = len(reports)==18 and all(r['status']=='VERIFIED_COMPLETE_TASK' for r in reports)
    result = {'status':'VERIFIED_COMPLETE_18' if complete else 'INCOMPLETE_OR_AUDIT_ERRORS',
        'run':run.name,'errors':errors,'tasks':reports,'all_18_verified':complete,
        'new_powerflow_calls':0,'new_training_steps':0,
        'execution_manifest_sha256':sha(run/'execution_manifest.json')}
    need(not any(n=='dss' or n.startswith(('dss.','opendssdirect','Environments')) for n in sys.modules), 'Saved audit loaded numerical engine')
    save(output/'verification.json',result)
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['preflight','audit'])
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();run=args.run.resolve()
    need(run.parent==(ROOT/'runs').resolve(),'Project run required')
    if args.mode=='preflight':
        result=preflight(run);need(not (run/'preflight.json').exists(),'Preflight already exists')
        save(run/'preflight.json',result);print(json.dumps(result))
    else:
        need(args.output is not None,'Separate audit output required')
        result=audit_run(run,args.output.resolve())
        print(json.dumps({'status':result['status'],'errors':result['errors']}))
        raise SystemExit(0 if result['all_18_verified'] else 2)
