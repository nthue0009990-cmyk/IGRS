"""Development mechanism probe. Generate on eight archived IGRS outputs, then
run at most one fresh triggered task, with no archived answer in the worker.
"""
import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from run_nature34_clpu_pilot import make_env, numeric_lists
from dnr.nature34_clpu_branch import BranchConfig, BranchSearch
from dnr.nature34_clpu_early import merge_results
from dnr.nature34_clpu_environment import CLPUBudgetExhausted
from dnr.nature34_clpu_guard_first_control import (
    GuardFirstEvidenceService, GuardFirstMacroController, GuardFirstStagedSearch)
from dnr.nature34_clpu_wait_compression import improving_retimings, choose_retiming

BASE = ROOT / 'runs/20261105T024027_869785Z_clpu_direct_beam'
BASE_SHA = '2989962bf0e870540c9fb37a07c8e0d31c7cbf951c0eb20e8177127a8a603932'
OPERATOR = ROOT / 'src/dnr/nature34_clpu_wait_compression.py'


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as out:
        json.dump(numeric_lists(data), out, ensure_ascii=False, allow_nan=False)


def prepare():
    assert sha(BASE / 'completion_manifest.json') == BASE_SHA
    manifest = {k.replace('\\', '/'): v for k, v in read(BASE / 'completion_manifest.json').items()}
    def bound(name):
        path = BASE / name
        assert sha(path) == manifest[name], name
        return path
    assert read(bound('audit_posthoc_v2/summary.json'))['status'] == 'PASS'
    schedule = [s for s in read(bound('schedule.json')) if s['arm'] == 'P']
    assert [s['context_id'] for s in schedule] == list(range(8))
    bound('config.json')
    for name in ('selected_scenarios.json','selected_events.json','author_static_metadata.json',
                 'load_reference.json','condition.json','physical_scope.json','command_inventory.json',
                 'source_provenance.json','source_pointer.json','physical_config.json','dependencies.json'):
        bound(name)
    # Keep the archived scientific modules and input model binding intact.
    prepared = read(bound('prepared.json'))
    for name in prepared['source_files']:
        assert sha(ROOT / name) == manifest['source/' + name], name
    run = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_igrs_wait_compression_dev')
    run.mkdir()
    shutil.copyfile(__file__, run / 'driver.py')
    shutil.copyfile(OPERATOR, run / 'operator.py')
    shutil.copyfile(BASE / 'dependencies.json', run / 'dependencies.json')
    inventory = []
    for spec in schedule:
        old = read(bound(spec['cell'] + '/result.json'))
        assert old['fully_certified'] and old['spec'] == spec
        case = make_env(BASE, spec)
        candidates = improving_retimings(case, old['best_plan']['actions'])
        inventory.append({'context_id': spec['context_id'], 'source_cell': spec['cell'],
                          'incumbent_kwh': old['energy_kwh'], 'candidates': candidates})
    save(run / 'all8_graph.json', inventory)
    selected = [r['context_id'] for r in inventory if r['candidates']][:1]
    spec = next((s for s in schedule if s['context_id'] in selected), None)
    config = {'scope':'Previously observed development cases; mechanism probe only',
              'parent': BASE.name, 'parent_sha256': BASE_SHA, 'training': 0, 'seed': 0,
              'selection':'first context by original id with positive same-route wait compression',
              'operator':'one original internal or leading wait block per proposal; deterministic energy coverage order',
              'candidate_limit':'at most five fresh retiming choices within unused original 16',
              'logical_per_stage':256,'physical_per_task':512,'worker_seconds':240,
              'budget_partial':'keep fully certified incumbent; numerical/reliability error stops task',
              'end':'one attempt; no retries, retraining, or held-out access', 'spec':spec,
              'inputs':{p:manifest[p] for p in ['config.json','physical_scope.json','schedule.json']},
              'driver_sha256':sha(run/'driver.py'),'operator_sha256':sha(run/'operator.py'),
              'inventory_sha256':sha(run/'all8_graph.json')}
    save(run / 'freeze.json', config)
    print(run, flush=True)
    return run


def worker(run):
    config = read(run / 'freeze.json')
    assert sha(Path(__file__)) == config['driver_sha256'] and sha(OPERATOR) == config['operator_sha256']
    assert all(sha(BASE / p) == h for p,h in config['inputs'].items())
    spec = config['spec']
    cell = run / 'cold'
    cell.mkdir()
    started = time.perf_counter()
    deadline = started + config['worker_seconds']
    env = make_env(BASE, spec)
    bc = BranchConfig(**read(BASE / 'config.json')['branch_search'])
    with GuardFirstEvidenceService(BASE, cell, env, read(BASE/'physical_scope.json'),
            mode='shared', physical_budget=512, logical_budget=256) as service:
        service.deadline_monotonic = deadline
        with service.view('GUARD') as oracle:
            guard = BranchSearch(env, oracle, bc)
            while not guard.done:
                assert time.perf_counter() < deadline
                guard.step(guard.rule_action('beam'))
            guard_result = guard.result()
            assert guard_result['initial_certificate']['status'] == 'PASS'
            save(cell/'GUARD/phase_result.json', guard_result)
        with service.view('EARLY') as oracle:
            early = GuardFirstStagedSearch(env, oracle, bc, guard_result=guard_result,
                guard_reference={'phase':'GUARD','file':'GUARD/phase_result.json',
                                 'sha256':sha(cell/'GUARD/phase_result.json')})
            controller = GuardFirstMacroController(early, method='beam')
            while not early.done:
                assert time.perf_counter() < deadline
                proposal = controller.prepare()
                if proposal is None:
                    break
                controller.apply(proposal)
            assert early.done and early.stop_reason == 'STAGED_VERIFICATION_POOL_EXHAUSTED'
            early_result = controller.result()
            merged = merge_results(early_result, guard_result, protected=True)
            parent = {'EARLY':early_result,'GUARD':guard_result}[merged['selected_phase']]['best_plan']
            best_plan = parent['actions']
            best_energy = parent['validation']['energy_kwh']
            baseline = {'energy_kwh':best_energy, 'actions':best_plan,'queries':service.physical_points,
                        'early_logical':oracle.points,'total_logical':service.summary()['logical_points'],
                        'verify':early.verify_attempts,'elapsed_worker_s':time.perf_counter()-started}
            save(cell/'baseline.json', baseline)
            save(cell/'EARLY/phase_result.json', early_result)
            checked = {tuple(r['actions']) for phase in [guard_result,early_result]
                       for r in phase['attempts'] if r['status'] in ('COMPLETE','FAIL')}
            candidates = improving_retimings(env, best_plan)
            assert len(candidates) + early_result.get('assessed_graph_candidates',0) <= bc.max_generated_candidates
            pending = [r for r in candidates if tuple(r['actions']) not in checked]
            tried = []
            for choice in range(min(5, 16 - early.verify_attempts)):
                pending = [p for p in pending if p['energy_kwh'] > best_energy]
                if not pending:
                    break
                candidate = choose_retiming(pending, tried)
                pending.remove(candidate)
                tried.append(candidate['energy_kwh'])
                start_counts = [service.physical_points, oracle.points]
                try:
                    validation = env.validate_plan(env.initial, candidate['actions'], oracle)
                except CLPUBudgetExhausted:
                    save(cell/f'attempt_{choice}.json', {'candidate':candidate,'status':'PARTIAL_BUDGET',
                         'before':start_counts,'after':[service.physical_points,oracle.points]})
                    break
                assert validation['status'] in ('FAIL','COMPLETE')
                save(cell/f'attempt_{choice}.json', {'candidate':candidate,'status':validation['status'],
                     'validation':validation,'before':start_counts,'after':[service.physical_points,oracle.points]})
                if validation['status'] == 'COMPLETE':
                    assert len(validation['samples']) == 90
                    assert math.isclose(validation['energy_kwh'], candidate['energy_kwh'], abs_tol=1e-9)
                    best_energy, best_plan = validation['energy_kwh'], candidate['actions']
            total_choices = early.verify_attempts + len(tried)
            assert total_choices <= 16 and oracle.points <= 256
    account = service.summary()
    assert account['actual_numerical_entries'] == 3 + 4*account['physical_points']
    assert account['physical_points'] <= 512 and account['logical_points'] <= 512
    save(cell/'result.json', {'baseline':baseline,'energy_kwh':best_energy,'actions':best_plan,
         'quality_pct':100*best_energy/(env.total_kw*env.horizon*env.slot_seconds/3600),
         'candidates':len(candidates),'previously_checked':sum(tuple(r['actions']) in checked for r in candidates),
         'total_verify':total_choices,'account':account,'worker_seconds':time.perf_counter()-started})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','execute','worker'])
    parser.add_argument('run',nargs='?',type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'worker':
        worker(args.run.resolve())
    else:
        run = args.run.resolve()
        assert read(run/'freeze.json')['spec'] is not None
        start = time.perf_counter()
        save(run/'started.json', {'freeze_sha256':sha(run/'freeze.json')})
        try:
            with (run/'worker.log').open('x',encoding='utf-8') as log:
                subprocess.run([sys.executable,'-B',__file__,'worker',str(run)], stdout=log,
                               stderr=subprocess.STDOUT,timeout=240,check=True)
            save(run/'finished.json',{'cold_seconds':time.perf_counter()-start,
                                      'result_sha256':sha(run/'cold/result.json')})
        except Exception:
            save(run/'failure.json',{'details':traceback.format_exc(),'elapsed_s':time.perf_counter()-start})
            raise
