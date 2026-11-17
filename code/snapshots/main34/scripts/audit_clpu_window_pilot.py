"""Independent cold Beam/window pilot audit; no production search or solver.

The window override is reproduced here from the frozen rule, independently of
the production subclass. All remaining graph, plan, request and AC checks use
the existing independent auditors. Every attempted certificate is consumed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
sys.dont_write_bytecode = True
import audit_nature34_clpu_branch as branch
from d060_validation_common import SOURCE_RUN, SOURCE_SHA

old = branch.old
require, close, same = branch.require, branch.close, branch.same
read, save, sha, canonical, read_lines = branch.read, branch.save, branch.sha, branch.canonical, branch.read_lines
GROUPS = {'original': 'G0_original_beam', 'coverage_v1': 'G1_coverage_window_beam'}


class WindowReplay(branch.BranchReplay):
    """Independent alternating FIFO hold-path coverage, ordinary turns intact."""
    def window(self):
        if self.turn % 2 == 1 and self.best is not None:
            energy = self.best['validation']['energy_kwh']
            eligible = sorted((row for row in self.frontier.values()
                if row['origin'] == 'prefix_wait'
                and row['info']['actions'] not in self.checked
                and row['info']['energy_kwh'] > energy), key=lambda row: row['id'])
            if eligible:
                depth, seen, selected = len(eligible[0]['prefix']), set(), []
                for row in eligible:
                    actions = row['info']['actions']
                    if len(row['prefix']) == depth and actions not in seen:
                        selected.append(row)
                        seen.add(actions)
                        if len(selected) == self.cfg['action_window']:
                            break
                return selected
        return super().window()


def manifest(directory, name='completion_manifest.json'):
    directory = Path(directory).resolve()
    rows = read(directory / name)
    for relative, expected in rows.items():
        target = (directory / relative.replace('\\', '/')).resolve()
        require(target.is_relative_to(directory) and target != directory and sha(target) == expected,
                'Changed or outside sealed artifact: ' + relative)
    return rows


def finite_time(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            'Invalid elapsed time: ' + label)
    return value


def checkpoint(replay, oracle, elapsed):
    return {'search_choices': replay.turn, 'physical_points': oracle.used,
            'numerical_entries': 3 + 4 * oracle.used, 'elapsed_seconds': elapsed,
            'quality_pct': 100 * replay.quality if replay.best is not None else None,
            'fully_certified': replay.best is not None}


def physical_seconds(points, start, end):
    return math.fsum(finite_time(points[i]['elapsed_seconds'], 'physical point') for i in range(start, end))


def check_records(mech, branch_cfg, result, attempts, steps, request_rows, points, condition_sha):
    """Replay compact rule-only records, including all reached-quality costs."""
    variant = result['spec']['window_variant']
    require(variant in GROUPS and result['spec']['method'] == 'beam', 'Unsupported pilot rule')
    oracle = branch.PaidRequests(request_rows, points, mech, branch_cfg['point_budget'], condition_sha)
    cls = branch.BranchReplay if variant == 'original' else WindowReplay
    replay = cls(mech, branch_cfg, oracle, attempts)
    same(result['initial_certificate'], replay.initial_certificate, 'initial_certificate')
    elapsed = finite_time(result['initial_elapsed_seconds'], 'bootstrap')
    require(elapsed >= physical_seconds(points, 0, oracle.used), 'Bootstrap time omits paid physical work')
    require(isinstance(result['checkpoints'], list) and result['checkpoints'], 'Missing quality checkpoints')
    checkpoints = [checkpoint(replay, oracle, elapsed)]
    wait = 100 * replay.quality if replay.best is not None else None
    same(result['wait_baseline_quality_pct'], wait, 'wait_baseline_quality_pct')
    iterator, end = iter(steps), object()
    index = 0
    for started in iterator:
        completed = next(iterator, end)
        require(completed is not end, 'Unclosed STARTED step')
        require(not replay.done and started['status'] == 'STARTED'
                and completed['status'] == 'COMPLETED'
                and started['transition_id'] == completed['transition_id'] == index,
                'Step chronology or terminal boundary differs')
        require(started['method'] == 'beam' and started['oracle_before'] == oracle.counts(),
                'Rule or before-query counts differ')
        window = replay.window()
        require(started['window_branch_ids'] == [row['id'] for row in window],
                'Candidate window differs from independent rule')
        action = started['action']
        require(type(action) is int and action == replay.rule('beam')
                and started['branch_id'] == window[action]['id'], 'Beam selected branch differs')
        for key, value in started.items():
            if key not in ('status', 'elapsed_seconds'):
                same(completed[key], value, 'completed_start_binding.' + key)
        before = finite_time(started['elapsed_seconds'], 'step start')
        after = finite_time(completed['elapsed_seconds'], 'step completion')
        require(elapsed <= before <= after, 'Elapsed time reversed or omitted between boundaries')
        before_points = oracle.used
        response = replay.step(action)
        same(completed['response'], response, 'independent_step_response')
        require(completed['oracle_after'] == oracle.counts(), 'After-query counts differ')
        require(after - before >= physical_seconds(points, before_points, oracle.used),
                'Step time omits paid physical work')
        require(index + 1 < len(result['checkpoints']), 'Missing quality checkpoint boundary')
        elapsed = finite_time(result['checkpoints'][index + 1]['elapsed_seconds'], 'quality checkpoint')
        require(elapsed >= after, 'Quality checkpoint precedes completed certificate')
        checkpoints.append(checkpoint(replay, oracle, elapsed))
        index += 1
    require(replay.done and result['stop_reason'] == replay.reason,
            'Unfinished search or invalid upper-bound stop')
    if 'done' in result:
        require(result['done'] is True, 'Completed search falsely marked active')
    require(replay.attempt_index == len(attempts), 'Unconsumed or omitted attempted plans')
    require(oracle.index == len(oracle.rows) and oracle.used == len(points), 'Unbound paid physical queries')
    same(result['best_plan'], replay.best, 'independently_certified_best')
    expected_status = ('INITIAL_FAIL' if replay.reason == 'INITIAL_FAIL' else
                       'CERTIFIED_INCUMBENT' if replay.best is not None else 'NO_CERTIFIED_COMPLETE_PLAN')
    require(result['status'] == expected_status, 'Search status misclassified')
    require(result['search_decisions'] == result['completed_expansions'] == index == replay.turn,
            'Completed branch expansion count differs')
    expected_counts = {'generated_candidates': len(replay.keys), 'graph_assessed_candidates': len(replay.assessed),
        'checked_candidates': len(replay.checked), 'branch_expansions': replay.turn,
        'generation_cap_rejections': replay.rejections}
    same(result['counts'], expected_counts, 'search_counts')
    same(result['checkpoints'], checkpoints, 'quality_checkpoints')
    require(elapsed <= finite_time(result['elapsed_seconds'], 'worker total'),
            'Worker time omits final search work')
    return {'decisions': index, 'generated': len(replay.keys), 'assessed': len(replay.assessed),
            'stop_reason': replay.reason, 'best_plan': replay.best, 'quality_checkpoints': checkpoints,
            'final_quality_pct': 100 * replay.quality if replay.best is not None else None}


def check_population(population, training, schedule):
    faults = {row['scenario_id'] for row in training}
    events = {row['event_id'] for row in training}
    require(len(training) == 32 and len(faults) == 8 and events == {3, 6, 20, 30},
            'Original training population differs')
    faults = sorted(faults, key=lambda value: hashlib.sha256(('route-b-s1-fault-v1|' + value).encode()).hexdigest())
    events = sorted(events, key=lambda value: hashlib.sha256(('route-b-s1-event-v1|' + str(value)).encode()).hexdigest())
    expected = [(fault, events[i % 4]) for i, fault in enumerate(faults)]
    require([(row['scenario_id'], row['event_id']) for row in population['contexts']] == expected,
            'Predeclared S1 context assignment changed')
    require(len(schedule) == 16 and len({row['cell'] for row in schedule}) == 16,
            'Expected sixteen unique cold tasks')
    require({(row['context_id'], row['window_variant']) for row in schedule} ==
            {(i, variant) for i in range(8) for variant in GROUPS}, 'Missing or duplicated paired method')
    for row in schedule:
        i, variant = row['context_id'], row['window_variant']
        require((row['scenario_id'], row['event_id']) == expected[i]
                and row['group'] == GROUPS[variant] and row['method'] == 'beam'
                and row['point_budget'] == 256 and row['seed'] == 0,
                'Task changed method, context or budget')
        require(not row.get('checkpoint') and not row.get('checkpoint_sha256'), 'Rule loaded learned checkpoint')


def check_parent(run):
    run = Path(run).resolve()
    prepared = read(run / 'prepared.json')
    require(sha(run / 'prephysical_freeze_manifest.json') == prepared['freeze_sha256'], 'Pilot freeze changed')
    manifest(run, 'prephysical_freeze_manifest.json')
    for relative in prepared['source_files']:
        require(sha(ROOT / relative) == sha(run / 'source' / relative), 'Executing frozen source changed: ' + relative)
    source = ROOT / 'runs' / SOURCE_RUN
    require(sha(source / 'completion_manifest.json') == SOURCE_SHA, 'Original scientific source seal changed')
    source_manifest = {name.replace('\\', '/'): value for name, value in read(source / 'completion_manifest.json').items()}
    for relative in prepared['source_files']:
        key = 'source/' + relative.replace('\\', '/')
        if key in source_manifest:
            require(sha(ROOT / relative) == source_manifest[key], 'Inherited scientific implementation changed: ' + relative)
    for name in ('config.json', 'training_contexts.json', 'training_scenarios.json', 'selected_events.json'):
        require(sha(source / name) == source_manifest[name], 'Original scientific input changed: ' + name)
    cfg, original = read(run / 'config.json'), read(source / 'config.json')
    require(cfg['branch_search'] == original['branch_search'], 'Original branch configuration changed')
    for key in ('horizon', 'slot_seconds', 'sample_seconds', 'alpha', 'algorithm', 'condition_id'):
        require(cfg[key] == original[key], 'Scientific configuration changed: ' + key)
    for name in ('physical_config.json', 'condition.json', 'operating_reference.json', 'author_static_metadata.json',
                 'load_reference.json', 'command_inventory.json'):
        require(sha(run / name) == source_manifest[name] == sha(source / name), 'Physical source changed: ' + name)
    for name, digest in source_manifest.items():
        if name.startswith('source_inputs/'):
            require(sha(run / name) == digest, 'Physical source input changed: ' + name)
    require(sha(ROOT / 'src/dnr/nature34_clpu_branch.py') ==
            source_manifest['source/src/dnr/nature34_clpu_branch.py'], 'Original Beam implementation changed')
    check_population(read(run / 'population.json'), read(source / 'training_contexts.json'), read(run / 'schedule.json'))
    scenarios = {row['scenario_id']: row for row in read(source / 'training_scenarios.json')}
    events = {row['event_id']: row for row in read(source / 'selected_events.json')}
    contexts = read(run / 'population.json')['contexts']
    wanted_scenarios = {row['scenario_id'] for row in contexts}
    wanted_events = {row['event_id'] for row in contexts}
    selected_scenarios, selected_events = read(run / 'selected_scenarios.json'), read(run / 'selected_events.json')
    require(len(selected_scenarios) == len(wanted_scenarios)
        and {row['scenario_id'] for row in selected_scenarios} == wanted_scenarios
        and all(row == scenarios[row['scenario_id']] for row in selected_scenarios), 'Selected fault details changed')
    require(len(selected_events) == len(wanted_events)
        and {row['event_id'] for row in selected_events} == wanted_events
        and all(row == events[row['event_id']] for row in selected_events), 'Selected event parameters changed')
    binding = read(run / 'source_binding.json')
    require(binding['run'] == SOURCE_RUN and binding['completion_sha256'] == SOURCE_SHA
        and binding['population_sha256'] == sha(run / 'population.json'), 'Source/population binding differs')
    return cfg


def check_cell(run, spec, cfg):
    cell = (run / spec['cell']).resolve()
    require(cell.is_relative_to(run.resolve()) and cell != run.resolve(), 'Cell outside pilot')
    sealed = manifest(cell)
    required = {'spec.json', 'result.json', 'initial_state.json', 'plans.jsonl.gz', 'steps.jsonl.gz',
                'worker_started.json', 'points.jsonl.gz', 'requests.jsonl', 'command_ledger.jsonl',
                'command_stats.json', 'metadata.json', 'oracle_summary.json'}
    require(required <= set(sealed) and not (cell / 'error.json').exists(), 'Missing sealed evidence or retained error')
    result = read(cell / 'result.json')
    require(read(cell / 'spec.json') == result['spec'] == spec, 'Cell spec differs from schedule')
    worker = read(cell / 'worker_started.json')
    require(worker['spec_sha256'] == sha(cell / 'spec.json') and type(worker['pid']) is int and worker['pid'] > 0,
            'Worker identity or started spec binding differs')
    for key in ('training_interactions', 'optimizer_updates', 'new_training'):
        require(result[key] == 0, 'Unapproved learning: ' + key)
    require(result['formal_holdout_numerical_access'] is False and result['tune_numerical_access'] is False,
            'Cross-partition numerical access')
    scenarios = {row['scenario_id']: row for row in read(run / 'selected_scenarios.json')}
    events = {row['event_id']: row for row in read(run / 'selected_events.json')}
    mech = branch.Mechanics(read(run / 'author_static_metadata.json'), read(run / 'load_reference.json'),
        scenarios[spec['scenario_id']], read(run / 'condition.json'), events[spec['event_id']], cfg)
    require(read(cell / 'initial_state.json') == mech.initial, 'Initial state or history changed')
    points, statuses = branch.check_raw_points(run, cell, spec, cfg, mech)
    commands = old.check_commands(cell, points, read(run / 'command_inventory.json'))
    requests = old.check_requests(cell, points, spec['point_budget'])
    oracle = read(cell / 'oracle_summary.json')
    require(result['oracle'] == oracle and oracle['status'] == 'COMPLETE_COUNTERS_PENDING_AUDIT'
        and oracle['points'] == len(points) and oracle['point_budget'] == spec['point_budget']
        and oracle['requests'] == requests['requests'] and oracle['cache_hits'] == requests['cache_hits']
        and oracle['statuses'] == dict(statuses) and oracle['actual_numerical_entries'] == commands['numerical_entries']
        and oracle['maximum_numerical_entries'] == 3 + 4 * spec['point_budget']
        and oracle['point_seconds'] == [row['elapsed_seconds'] for row in points.values()], 'Oracle cost evidence differs')
    require(result['config'] == cfg['branch_search'], 'Branch config differs')
    checked = check_records(mech, cfg['branch_search'], result, list(read_lines(cell / 'plans.jsonl.gz')),
        read_lines(cell / 'steps.jsonl.gz'), list(read_lines(cell / 'requests.jsonl')), points, sha(run / 'condition.json'))
    return {'cell': spec['cell'], 'spec': spec, 'status': result['status'], 'points': len(points),
            'physical_statuses': dict(statuses), 'commands': commands, 'requests': requests, 'branch': checked,
            'cell_completion_sha256': sha(cell / 'completion_manifest.json')}


def check_controller(controller, schedule, checks):
    cells = [row['cell'] for row in schedule]
    require(controller['status'] == 'COMPLETE_CELLS_PENDING_AUDIT' and controller['errors'] == []
        and controller['unknown_cost_cells'] == [] and len(controller['completed']) == 16
        and set(controller['completed']) == set(cells), 'Controller not fully complete')
    exits = controller['exits']
    require(len(exits) == 16 and {row['cell'] for row in exits} == set(cells)
        and all(row['exit_code'] == 0 for row in exits), 'Worker exits missing or failed')
    points = sum(row['points'] for row in checks)
    numerical = sum(row['commands']['numerical_entries'] for row in checks)
    require(controller['known_physical_points'] == points <= 4096
        and controller['known_numerical_entries'] == numerical <= 16432, 'Controller cost totals differ')
    finite_time(controller['wall_seconds'], 'controller total')
    return points, numerical


def audit(run, output=None):
    run = Path(run).resolve()
    require(run.parent == ROOT / 'runs', 'Source run outside project runs')
    output = Path(output).resolve() if output else run / 'audit'
    require(not output.exists(), 'Audit output already exists; preserve previous evidence')
    output.mkdir(parents=True)
    started, checks = time.perf_counter(), []
    result = {'status': 'FAIL', 'source_run': run.name, 'errors': [], 'audited_cells': 0,
              'new_physical_points': 0, 'new_training': 0}
    try:
        cfg = check_parent(run)
        schedule = read(run / 'schedule.json')
        for spec in schedule:
            checked = check_cell(run, spec, cfg)
            checks.append(checked)
            save(output / 'progress.json', {'audited_cells': len(checks), 'scheduled': len(schedule),
                'elapsed_seconds': time.perf_counter() - started})
        points, numerical = check_controller(read(run / 'controller_completion.json'), schedule, checks)
        require('opendssdirect' not in sys.modules and 'dss' not in sys.modules, 'Audit initialized simulator')
        require(all(row['branch']['best_plan'] is not None for row in checks),
                'Missing certified complete plan for protected comparison')
        result.update(status='PASS', physical_points=points, numerical_entries=numerical,
            physical_statuses=dict(sum((Counter(row['physical_statuses']) for row in checks), Counter())),
            schedule_sha256=sha(run / 'schedule.json'), freeze_sha256=sha(run / 'prephysical_freeze_manifest.json'),
            controller_completion_sha256=sha(run / 'controller_completion.json'))
    except BaseException as exc:
        result['errors'].append({'error': repr(exc), 'traceback': traceback.format_exc()})
    result.update(audited_cells=len(checks), results=checks, elapsed_seconds=time.perf_counter() - started)
    save(output / 'summary.json', result)
    save(output / 'source_hashes.json', {str(Path(p).relative_to(ROOT)): sha(p) for p in
        (__file__, branch.__file__, old.__file__, ROOT / 'scripts/audit_nature34_joint_planning.py')})
    save(output / 'completion_manifest.json', {p.relative_to(output).as_posix(): sha(p)
        for p in output.rglob('*') if p.is_file() and p.name != 'completion_manifest.json'})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.run, args.output)
    print(canonical({k: v for k, v in result.items() if k != 'results'}), flush=True)
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
