"""Independent logical/paid evidence audit for exact electrical-input caching.

No production cache, search, environment or solver is imported. Historical
raw-output review is read-only and uses the unchanged physical certificates.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
sys.dont_write_bytecode = True
import audit_clpu_window_pilot as window

branch, old = window.branch, window.old
require, close, same = window.require, window.close, window.same
read, save, sha, canonical, read_lines = window.read, window.save, window.sha, window.canonical, window.read_lines
manifest = window.manifest
SOURCE_RUN = '20261027T121023_469270Z_clpu_window_pilot'
SOURCE_SHA = '9967ea771263b5b9bb45835116dec1c5fd70f8908c17ec2348d90d4a6377fc7e'
ARMS = ('semantic_identity', 'electrical_equivalent')


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def source_cell_binding(source, cell_name, expected_root_sha=SOURCE_SHA):
    source = Path(source).resolve()
    cell = (source / cell_name).resolve()
    require(cell.is_relative_to(source) and cell != source, 'Source cell escaped frozen window run')
    require(sha(source / 'completion_manifest.json') == expected_root_sha, 'Historical root seal changed')
    root_manifest = read(source / 'completion_manifest.json')
    require(root_manifest['audit/summary.json'] == sha(source / 'audit/summary.json'), 'Historical audit is not root-bound')
    audited, result = read(source / 'audit/summary.json'), read(cell / 'result.json')
    require(audited['status'] == 'PASS', 'Historical source audit did not pass')
    matches = [row for row in audited['results'] if row['cell'] == cell_name and row['spec'] == result['spec']]
    require(len(matches) == 1, 'Historical audit cell/spec binding absent or ambiguous')
    cell_sha = sha(cell / 'completion_manifest.json')
    require(matches[0]['cell_completion_sha256'] == cell_sha, 'Historical audited cell seal changed')
    cell_manifest = read(cell / 'completion_manifest.json')
    fields = {'result.json': 'source_result_sha256', 'completion_manifest.json': 'source_completion_sha256',
        'points.jsonl.gz': 'source_points_sha256', 'steps.jsonl.gz': 'source_steps_sha256',
        'plans.jsonl.gz': 'source_plans_sha256', 'requests.jsonl': 'source_requests_sha256'}
    binding = {'source_run': SOURCE_RUN, 'source_cell': cell_name}
    for name, field in fields.items():
        actual = sha(cell / name)
        if name != 'completion_manifest.json':
            require(root_manifest.get(cell_name + '/' + name) == cell_manifest.get(name) == actual,
                    'Root/cell scientific file binding differs: ' + name)
        binding[field] = actual
    return binding


def retimed_feedback(value, minutes):
    if isinstance(value, dict):
        return {key: minutes if key == 'time_minutes' else retimed_feedback(item, minutes)
                for key, item in value.items()}
    if isinstance(value, list):
        return [retimed_feedback(item, minutes) for item in value]
    return value


def candidate_physical_identity(row):
    """Historical opportunity grouping only; each source cell is a fixed scope."""
    identity = row['state_identity']
    require(identity['multipliers'] == row['clpu_multipliers'], 'Historical multiplier identity mismatch')
    value = {key: row[key] for key in ('case', 'fault_lines', 'action', 'alpha', 'algorithm',
        'condition_id', 'clpu_multipliers', 'control_state_semantics')}
    value.update(command=identity['command'], base_condition_sha256=identity['base_condition_sha256'],
                 model_version=identity['model_version'])
    return value


def differences(left, right, path=()):
    """Return every exact raw-field difference, including structural changes."""
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            yield path + ('DICT_KEYS',), sorted(left), sorted(right)
        for key in left.keys() & right.keys():
            yield from differences(left[key], right[key], path + (key,))
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            yield path + ('LIST_LENGTH',), len(left), len(right)
        for index, (a, b) in enumerate(zip(left, right)):
            yield from differences(a, b, path + (index,))
    elif type(left) is not type(right) or left != right:
        yield path, left, right


def disabled_unused_difference(path, left_raw, right_raw):
    """Only fields excluded by the existing AC/isolation/constraint checkers."""
    if len(path) < 3 or path[0] != 'elements' or type(path[1]) is not int:
        return False
    index, field = path[1:3]
    if index >= min(len(left_raw['elements']), len(right_raw['elements'])):
        return False
    left, right = left_raw['elements'][index], right_raw['elements'][index]
    return (left.get('enabled') is right.get('enabled') is False
            and left.get('name') == right.get('name')
            and field in ('voltages_real_imag', 'powers_kw_kvar', 'yprim_real_imag', 'node_order'))


def recheck_raw_certificate(row, metadata, physical_config):
    merged = {**metadata, **row['metadata_override']}
    expected = {'known_constraints': old.check_known_constraints(row['raw_outputs'], merged,
                    **physical_config['constraint_tolerances']),
                'ac_audit': old.audit_ac_snapshot(row['raw_outputs'], merged,
                    **physical_config['ac_audit_tolerances']),
                'isolation': old.verify_isolation(row['raw_outputs'], merged)}
    for key, value in expected.items():
        require(row[key] == value, 'Unchanged independent physical checker disagrees: ' + key)
    require(row['reliable'] is True and row['status'] in ('PASS', 'FAIL'), 'Unreliable historical evidence')
    return expected


def review_historical_outputs(output):
    """Review all 1,121 G1 points, not only one representative conflict pair."""
    output = Path(output).resolve()
    require(not output.exists(), 'Historical review exists; preserve prior evidence')
    output.mkdir(parents=True)
    started = time.perf_counter()
    result = {'status': 'FAIL', 'source_run': SOURCE_RUN, 'source_completion_sha256': SOURCE_SHA,
              'new_physical_points': 0, 'new_training': 0, 'contexts': [], 'errors': []}
    try:
        source = ROOT / 'runs' / SOURCE_RUN
        require(sha(source / 'completion_manifest.json') == SOURCE_SHA, 'Historical root seal changed')
        sealed = read(source / 'completion_manifest.json')
        for name in ('schedule.json', 'physical_config.json', 'source_inputs/metadata.json'):
            require(sha(source / name) == sealed[name], 'Historical review input changed: ' + name)
        metadata, physical = read(source / 'source_inputs/metadata.json'), read(source / 'physical_config.json')
        specs = sorted((s for s in read(source / 'schedule.json') if s['window_variant'] == 'coverage_v1'),
                       key=lambda s: s['context_id'])
        require(len(specs) == 8 and {s['context_id'] for s in specs} == set(range(8)), 'Incomplete source population')
        total, duplicate_total, raw_conflicts = 0, 0, 0
        for spec in specs:
            binding = source_cell_binding(source, spec['cell'])
            representatives, groups, pairs = {}, {}, []
            count = 0
            for row in read_lines(source / spec['cell'] / 'points.jsonl.gz'):
                require(row['point_id'] == count, 'Historical raw chronology changed')
                count += 1
                certificate = recheck_raw_certificate(row, metadata, physical)
                key = digest(candidate_physical_identity(row))
                groups.setdefault(key, []).append(row['point_id'])
                if key not in representatives:
                    representatives[key] = (row, certificate)
                    continue
                original, original_certificate = representatives[key]
                changes = list(differences(original['raw_outputs'], row['raw_outputs']))
                classified = [{'path': list(path), 'left': left, 'right': right,
                    'disabled_unused_field': disabled_unused_difference(path, original['raw_outputs'], row['raw_outputs'])}
                    for path, left, right in changes]
                require(all(c['disabled_unused_field'] for c in classified),
                        f'Certificate-relevant raw difference: context {spec["context_id"]}, point {row["point_id"]}')
                require(certificate == original_certificate, 'Equivalent input certificate differs')
                for field in ('load_readback', 'metadata_override', 'nominal_load_ledger', 'clpu_load_ledger',
                              'generation_ledger', 'measured_enabled_load', 'status', 'reliable'):
                    require(row[field] == original[field], 'Equivalent input scientific field differs: ' + field)
                require(retimed_feedback(original['structured_feedback'], row['time_minutes']) == row['structured_feedback'],
                        'Equivalent input time-corrected feedback differs')
                pairs.append({'representative_point_id': original['point_id'], 'point_id': row['point_id'],
                    'candidate_input_sha256': key, 'raw_equal': not changes, 'differences': classified,
                    'unchanged_certificate_exactly_equal': True, 'time_corrected_feedback_exactly_equal': True})
                raw_conflicts += bool(changes)
            checked = {'context_id': spec['context_id'], 'source_cell': spec['cell'], 'source_binding': binding,
                'physical_points_reviewed': count, 'unique_candidate_inputs': len(groups),
                'duplicate_pairs_reviewed': len(pairs), 'raw_difference_pairs': sum(not p['raw_equal'] for p in pairs),
                'raw_field_differences': sum(len(p['differences']) for p in pairs), 'pairs': pairs,
                'groups': [{'candidate_input_sha256': key, 'point_ids': ids} for key, ids in groups.items()]}
            save(output / f'context_{spec["context_id"]:02d}.json', checked)
            result['contexts'].append({k: v for k, v in checked.items() if k not in ('pairs', 'groups')})
            total += count
            duplicate_total += len(pairs)
            save(output / 'progress.json', {'contexts': len(result['contexts']), 'reviewed_points': total,
                                         'reviewed_duplicate_pairs': duplicate_total, 'new_physical_points': 0})
        require(total == 1121 and duplicate_total == 538, 'Historical population/opportunity count changed')
        result.update(status='PASS', physical_points_reviewed=total, duplicate_pairs_reviewed=duplicate_total,
            raw_difference_pairs=raw_conflicts, all_differences_in_existing_excluded_disabled_cache_fields=True,
            certificate_or_feedback_disagreements=0, constraint_or_audit_tolerances_changed=False,
            limitation='Observed fixed training evidence and frozen quasi-static implementation, not a universal solver-output equality proof')
    except BaseException as exc:
        result['errors'].append({'error': repr(exc), 'traceback': traceback.format_exc()})
    result['elapsed_seconds'] = time.perf_counter() - started
    save(output / 'summary.json', result)
    save(output / 'completion_manifest.json', {p.relative_to(output).as_posix(): sha(p)
        for p in output.rglob('*') if p.is_file() and p.name != 'completion_manifest.json'})
    return result


def electrical_key(scope, mech, identity):
    projection = old.independent_projection(mech.static, mech.loads, mech.scenario, identity['command'])
    return {'version': 'nc34-electrical-input-equivalence-v1', 'scope_sha256': digest(scope),
        'action': list(projection['action']),
        'case': {'name': mech.scenario['scenario_id'], 'fault_lines': mech.scenario['fault_lines']},
        'alpha': .5, 'algorithm': 0, 'condition': mech.condition, 'multipliers': identity['multipliers']}


def check_logical_records(logical_rows, requests, actual_rows, mech, scope, condition_sha, arm, budget=256):
    """Reconstruct dual ledgers; an alias must cite already paid same-cell evidence."""
    require(arm in ARMS and type(budget) is int and budget == 256, 'Changed arm or logical cap')
    logical, physical_keys, first_key, introduced, semantic = {}, {}, {}, set(), {}
    reuse_count, alias_seconds = 0, 0.
    for index, row in enumerate(logical_rows):
        require(row['logical_point_id'] == index and index < budget, 'Logical points not consecutive or over budget')
        state, seconds = row['dynamic_state'], row['time_seconds']
        require(type(state['step']) is int and 0 <= state['step'] <= mech.horizon
                and type(seconds) in (int, float) and math.isfinite(seconds)
                and 0 <= seconds <= mech.horizon * mech.slot and seconds % mech.sample == 0,
                'Logical time/history outside frozen scope')
        identity = mech.identity(state, seconds, condition_sha)
        require(row['semantic_identity'] == identity and row['semantic_identity_sha256'] == digest(identity)
                and digest(identity) not in semantic, 'Forged or duplicated semantic identity')
        key = electrical_key(scope, mech, identity)
        require(row['electrical_input_key'] == key and row['electrical_input_sha256'] == digest(key),
                'Incomplete or approximate electrical equivalence key')
        pid = row['physical_point_id']
        require(type(pid) is int and pid in actual_rows, 'Logical certificate cites absent physical evidence')
        paid = actual_rows[pid]
        paid_key = electrical_key(scope, mech, paid['state_identity'])
        require(paid_key == key, 'Alias reused different physical input')
        evidence = {'raw_outputs': paid['raw_outputs'], 'metadata_override': paid['metadata_override']}
        require(row['physical_evidence_sha256'] == digest(evidence), 'Alias physical raw evidence hash changed')
        if row['reuse_kind'] == 'ACTUAL_PHYSICAL':
            require(pid == len(introduced) and pid not in introduced
                    and paid['state_identity'] == identity and paid['dynamic_state'] == state,
                    'New logical physical point changed identity or order')
            require(arm == 'semantic_identity' or digest(key) not in first_key,
                    'Electrical arm recomputed an already paid exact input')
            introduced.add(pid)
            first_key.setdefault(digest(key), pid)
            physical_keys[pid] = digest(key)
        else:
            require(arm == 'electrical_equivalent' and row['reuse_kind'] == 'ELECTRICAL_EQUIVALENCE'
                    and pid in introduced and first_key.get(digest(key)) == pid,
                    'Future, cross-cell or unapproved electrical reuse')
            reuse_count += 1
        feedback = retimed_feedback(paid['structured_feedback'], seconds / 60.)
        certificate = {'point_id': index, 'status': paid['status'], 'reliable': paid['reliable'],
                       'feedback': feedback['violations']}
        require(certificate['reliable'] is True and certificate['status'] in ('PASS', 'FAIL')
                and row['certificate'] == certificate, 'Alias certificate or current-time feedback differs')
        duration = window.finite_time(row['alias_certification_seconds'], 'alias certification')
        require(row['reuse_kind'] == 'ELECTRICAL_EQUIVALENCE' or duration == 0,
                'Actual point falsely claims alias validation')
        alias_seconds += duration
        logical[index] = {'status': paid['status'], 'reliable': paid['reliable'],
            'clpu_multipliers': identity['multipliers'], 'action': key['action'], 'time_minutes': seconds / 60.,
            'state_identity': identity, 'identity_sha256': digest(identity), 'dynamic_state': state,
            'elapsed_seconds': paid['elapsed_seconds'] if row['reuse_kind'] == 'ACTUAL_PHYSICAL' else 0.,
            'structured_feedback': feedback}
        semantic[digest(identity)] = index
    require(introduced == set(actual_rows) == set(range(len(actual_rows))), 'Unconsumed or nonconsecutive actual points')
    require(logical and len(logical) == len(semantic), 'Empty or duplicated semantic certificate ledger')
    transformed, known = [], {}
    hits = denied = next_logical = 0
    for request_id, row in enumerate(requests, 1):
        require(type(row['request']) is int and row['request'] == request_id
                and type(row['semantic_cache_hit']) is bool and type(row['electrical_reuse']) is bool,
                'Semantic request order or cache flags changed')
        key = row['key']
        base = {'request': request_id, 'key': key, 'cache_hit': row['semantic_cache_hit']}
        if row['semantic_cache_hit']:
            require(key in known and row['logical_point_id'] == known[key]
                    and row['physical_point_id'] == logical_rows[known[key]]['physical_point_id']
                    and row['electrical_reuse'] is False and row['status'] == logical[known[key]]['status'],
                    'Future or false semantic cache hit')
            transformed.append({**base, 'point_id': known[key]})
            hits += 1
        elif row['status'] == 'BUDGET_UNKNOWN':
            require(key not in known and next_logical == budget and row['electrical_reuse'] is False
                    and 'logical_point_id' not in row and 'physical_point_id' not in row,
                    'Premature or malformed semantic budget denial')
            transformed.append({**base, 'status': 'BUDGET_UNKNOWN'})
            denied += 1
        else:
            require(key not in known and next_logical in logical and row['logical_point_id'] == next_logical
                    and key == logical[next_logical]['identity_sha256']
                    and row['physical_point_id'] == logical_rows[next_logical]['physical_point_id']
                    and row['status'] == logical[next_logical]['status']
                    and row['electrical_reuse'] == (logical_rows[next_logical]['reuse_kind'] == 'ELECTRICAL_EQUIVALENCE'),
                    'New semantic certificate and request disagree')
            transformed += [{**base, 'point_id': next_logical, 'status': 'STARTED'},
                            {**base, 'point_id': next_logical, 'status': row['status']}]
            known[key] = next_logical
            next_logical += 1
    require(next_logical == len(logical), 'Unrequested logical evidence')
    return logical, transformed, {'logical_points': len(logical), 'physical_points': len(actual_rows),
        'requests': len(requests), 'semantic_cache_hits': hits, 'electrical_reuses': reuse_count,
        'denied': denied, 'alias_certification_seconds': alias_seconds}


def check_wrapper_summary(summary, result_oracle, costs, logical, scope, arm, numerical_entries):
    require(result_oracle == summary and summary['closed'] is True,
            'Wrapper summary is not closed or differs')
    expected = {'version': 'nc34-semantic-electrical-cache-ledger-v1',
        'status': 'COMPLETE_DUAL_LEDGER_PENDING_AUDIT', 'points': costs['logical_points'],
        'logical_points': costs['logical_points'], 'physical_points': costs['physical_points'],
        'point_budget': 256, 'requests': costs['requests'], 'cache_hits': costs['semantic_cache_hits'],
        'semantic_cache_hits': costs['semantic_cache_hits'], 'electrical_reuses': costs['electrical_reuses'],
        'actual_numerical_entries': numerical_entries, 'initialization_numerical_entries': 3,
        'expected_numerical_entries': 3 + 4 * costs['physical_points'],
        'statuses': dict(Counter(row['status'] for row in logical.values())),
        'variant': arm, 'scope_sha256': digest(scope), 'private_cache_only': True,
        'cross_arm_cache_reuse': False}
    require(numerical_entries == expected['expected_numerical_entries'], 'Actual entries omit initialization or snapshot cost')
    for key, value in expected.items():
        require(type(summary[key]) is type(value) and summary[key] == value,
                'Dual-ledger summary differs: ' + key)
    close(summary['alias_certification_seconds'], costs['alias_certification_seconds'], 'Alias time sum differs')


def actual_prefix_count(logical_rows, logical_count):
    require(type(logical_count) is int and 0 <= logical_count <= len(logical_rows), 'Invalid logical cost boundary')
    ids = {row['physical_point_id'] for row in logical_rows[:logical_count]}
    require(ids == set(range(len(ids))), 'Actual evidence prefix is not causal')
    return len(ids)


def semantic_step_view(steps, logical_rows=None):
    output = []
    for row in steps:
        value = deepcopy(row)
        value.pop('elapsed_seconds', None)
        for label in ('oracle_before', 'oracle_after'):
            if label in value:
                counter = value[label]
                if logical_rows is not None:
                    require(counter['logical_points'] == counter['points']
                            and counter['physical_points'] == actual_prefix_count(logical_rows, counter['points']),
                            'Step hides or fabricates actual physical cost')
                value[label] = {key: counter[key] for key in ('requests', 'points')}
        output.append(value)
    return output


def check_search(run, spec, cfg, result, logical_rows, points, requests, plans, steps, mech):
    """Old independent WindowReplay sees logical identities; paid cost stays separate."""
    view = deepcopy(result)
    original_steps = semantic_step_view(steps, logical_rows)
    replay_steps = []
    for source, normalized in zip(steps, original_steps):
        replay_steps.append({**normalized, 'elapsed_seconds': source['elapsed_seconds']})
    for saved, normalized in zip(result['checkpoints'], view['checkpoints']):
        count = saved['logical_points']
        actual = actual_prefix_count(logical_rows, count)
        require(saved['physical_points'] == actual and saved['numerical_entries'] == 3 + 4 * actual,
                'Checkpoint confuses logical work with paid physical work')
        normalized.pop('logical_points')
        normalized['physical_points'], normalized['numerical_entries'] = count, 3 + 4 * count
    checked = window.check_records(mech, cfg['branch_search'], view, plans, replay_steps, requests,
                                   points, sha(run / 'condition.json'))
    return checked, {'result': {key: result[key] for key in ('status', 'config', 'stop_reason', 'best_plan',
            'initial_certificate', 'wait_baseline_quality_pct', 'completed_expansions', 'search_decisions', 'counts')},
        'plans': plans, 'steps': original_steps, 'requests': requests,
        'checkpoints': [{key: checkpoint[key] for key in ('search_choices', 'physical_points', 'quality_pct', 'fully_certified')}
                        for checkpoint in view['checkpoints']]}


def historical_semantics(run, spec):
    bindings = [row for row in read(run / 'source_cells.json') if row['context_id'] == spec['context_id']]
    require(len(bindings) == 1, 'Missing or repeated historical context binding')
    directory = (run / bindings[0]['historical_directory']).resolve()
    require(directory.is_relative_to(run) and directory != run, 'Historical copy outside run')
    result = read(directory / 'result.json')
    return {'result': {key: result[key] for key in ('status', 'config', 'stop_reason', 'best_plan',
            'initial_certificate', 'wait_baseline_quality_pct', 'completed_expansions', 'search_decisions', 'counts')},
        'plans': list(read_lines(directory / 'plans.jsonl.gz')),
        'steps': semantic_step_view(list(read_lines(directory / 'steps.jsonl.gz'))),
        'requests': list(read_lines(directory / 'requests.jsonl')),
        'checkpoints': [{key: checkpoint[key] for key in ('search_choices', 'physical_points', 'quality_pct', 'fully_certified')}
                        for checkpoint in result['checkpoints']]}


def check_parent(run):
    run = Path(run).resolve()
    require(run.parent == ROOT / 'runs', 'Cache pilot outside project runs')
    prepared = read(run / 'prepared.json')
    require(sha(run / 'prephysical_freeze_manifest.json') == prepared['freeze_sha256'], 'Prephysical freeze changed')
    manifest(run, 'prephysical_freeze_manifest.json')
    source = ROOT / 'runs' / SOURCE_RUN
    require(sha(source / 'completion_manifest.json') == SOURCE_SHA
            and read(run / 'source_binding.json') == {'run': SOURCE_RUN, 'completion_sha256': SOURCE_SHA},
            'Historical source binding changed')
    original_manifest = read(source / 'completion_manifest.json')
    for name in prepared['source_files']:
        require(sha(ROOT / name) == sha(run / 'source' / name), 'Executing frozen source changed: ' + name)
        if 'source/' + name in original_manifest:
            require(sha(ROOT / name) == original_manifest['source/' + name], 'Inherited science changed: ' + name)
    cfg, original = read(run / 'config.json'), read(source / 'config.json')
    require(cfg['equivalent_cache_study'] == 'CLPU_EQUIVALENT_CACHE_PILOT_V1'
            and cfg['branch_search'] == original['branch_search'], 'Study identity or search limits changed')
    for key in ('horizon', 'slot_seconds', 'sample_seconds', 'alpha', 'algorithm', 'condition_id'):
        require(cfg[key] == original[key], 'Inherited scientific configuration changed: ' + key)
    for key, expected in {'arms': list(ARMS), 'logical_point_budget': 256, 'physical_point_budget': 256,
        'max_expansions': 256, 'max_workers': 2, 'cooperative_seconds': 1800,
        'total_physical_budget': 4096, 'minimum_free_disk_bytes': 12 * 2**30,
        'no_training': True, 'no_historical_answers': True, 'no_cross_cell_cache': True, 'no_automatic_retry': True}.items():
        require(cfg['equivalent_cache'][key] == expected, 'Cache protocol parameter changed: ' + key)
    fixed_inputs = ('training_scenarios.json', 'author_static_metadata.json', 'load_reference.json',
        'conditions.json', 'physical_config.json', 'operating_reference.json', 'base_load_properties.json',
        'command_inventory.json', 'source_pointer.json', 'metadata.json', 'adaptation.json',
        'author_input_precheck.json', 'condition.json', 'source_provenance.json', 'event_split.json')
    for name in (*fixed_inputs, 'population.json', 'selected_scenarios.json', 'selected_events.json', 'training_contexts.json'):
        require(sha(run / name) == original_manifest[name] == sha(source / name), 'Inherited physical input changed: ' + name)
    for name, expected in original_manifest.items():
        if name.startswith('source_inputs/'):
            require(sha(run / name) == expected, 'Inherited source input changed: ' + name)
    scope = read(run / 'physical_scope.json')
    needed_inputs = set(fixed_inputs) | {'config.json', 'selected_scenarios.json', 'selected_events.json',
        'training_contexts.json', 'dependencies.json', 'python_environment.json'}
    needed_inputs |= {p.relative_to(run).as_posix() for p in (run / 'source_inputs').rglob('*') if p.is_file()}
    require(scope['scope_version'] == 'CLPU_FULL_FIXED_PHYSICAL_SCOPE_V1'
            and set(scope['input_files']) == needed_inputs
            and scope['input_files'] == {name: sha(run / name) for name in needed_inputs},
            'Complete fixed physical-input scope changed')
    expected_sources = {name: sha(ROOT / name) for name in prepared['source_files'] if name.endswith('.py')}
    require(scope['science_source_sha256'] == expected_sources, 'Complete implementation scope changed')
    for key, value in {'source_provenance': read(run / 'source_provenance.json'),
        'runtime_dependencies': read(run / 'dependencies.json'), 'python_environment': read(run / 'python_environment.json'),
        'physical_config': read(run / 'physical_config.json'), 'alpha': .5, 'algorithm': 0,
        'load_setpoint_precision': '.15g', 'reset_semantics': 'INDEPENDENT_QUASI_STATIC_AUTHOR_RESET_EACH_SNAPSHOT',
        'cache_scope': 'one cold cell only; no historical or cross-arm/cross-context answers'}.items():
        require(scope[key] == value, 'Fixed physical scope binding differs: ' + key)
    require(sha(Path(sys.executable)) == scope['python_environment']['sha256'], 'Audit interpreter differs from frozen runtime')
    review_binding = read(run / 'scope_review_binding.json')
    require(review_binding['path'] == 'artifacts/clpu_equivalent_cache_scope_review_20261030'
            and sha(run / 'scope_review/completion_manifest.json') == review_binding['completion_sha256']
            and sha(run / 'scope_review/summary.json') == review_binding['summary_sha256'], 'Scope review binding changed')
    manifest(run / 'scope_review')
    review = read(run / 'scope_review/summary.json')
    require(review['status'] == 'PASS' and review['new_physical_points'] == review['new_training'] == 0
            and review['physical_points_reviewed'] == 1121 and review['duplicate_pairs_reviewed'] == 538
            and review['certificate_or_feedback_disagreements'] == 0
            and review['constraint_or_audit_tolerances_changed'] is False, 'Full raw scope review has not passed')
    check_exact_key_review(run, review_binding['completion_sha256'])
    population = read(run / 'population.json')['contexts']
    contexts = [{'context_id': i, 'scenario_id': row['scenario_id'], 'event_id': row['event_id']}
                for i, row in enumerate(population)]
    require(len(contexts) == 8, 'Fixed context population changed')
    ordered = sorted(contexts,
        key=lambda row: hashlib.sha256(('clpu-equivalent-cache-context-order-v1|' + canonical(row)).encode()).hexdigest())
    expected_schedule = []
    for rank, context in enumerate(ordered):
        for position, arm in enumerate(ARMS if rank % 2 == 0 else tuple(reversed(ARMS))):
            index = len(expected_schedule)
            expected_schedule.append({**context, 'arm': arm, 'pair_order': rank, 'arm_launch_order': position,
                'method': 'beam', 'window_variant': 'coverage_v1', 'seed': 0, 'logical_point_budget': 256,
                'physical_point_budget': 256, 'max_expansions': 256, 'phase': 'training_cache_mechanism_evaluation',
                'deterministic': True, 'episode_id': index, 'cell': f'episodes/{index:03d}'})
    schedule = read(run / 'schedule.json')
    require(schedule == expected_schedule, 'Paired population/order changed')
    source_cells = read(run / 'source_cells.json')
    require(len(source_cells) == 8 and {row['context_id'] for row in source_cells} == set(range(8)),
            'Historical binding population changed')
    require(sha(run / 'historical/source_completion_manifest.json') == SOURCE_SHA
            and sha(run / 'historical/source_audit_summary.json') == original_manifest['audit/summary.json'],
            'Copied source root/audit binding changed')
    for item in source_cells:
        require(all(item[key] == contexts[item['context_id']][key] for key in ('scenario_id', 'event_id')),
                'Historical context changed')
        bound = source_cell_binding(source, item['source_cell'])
        require(item['source_completion_sha256'] == bound['source_completion_sha256'], 'Historical cell seal changed')
        directory = (run / item['historical_directory']).resolve()
        require(directory.is_relative_to(run) and directory != run, 'Historical directory escaped run')
        require(sha(directory / 'completion_manifest.json') == item['source_completion_sha256'], 'Copied historical cell seal changed')
        cm = read(directory / 'completion_manifest.json')
        require(set(item['source_files']) == {'spec.json', 'result.json', 'steps.jsonl.gz', 'plans.jsonl.gz',
                                            'requests.jsonl', 'points.jsonl.gz'}, 'Historical required file list changed')
        for name, expected in item['source_files'].items():
            require(cm[name] == original_manifest[item['source_cell'] + '/' + name] == expected,
                    'Historical scientific file seal differs: ' + name)
            if name != 'points.jsonl.gz':
                require(sha(directory / name) == expected, 'Copied source data changed: ' + name)
        require(not (directory / 'points.jsonl.gz').exists(), 'Historical raw answers copied into cold run')
        historical_spec = read(directory / 'spec.json')
        require(historical_spec['context_id'] == item['context_id'] and historical_spec['window_variant'] == 'coverage_v1'
                and historical_spec == read(directory / 'result.json')['spec'], 'Historical wrong method/context')
    budget = read(run / 'resource_budget.json')
    for key, expected in {'contexts': 8, 'cells': 16, 'physical_points': 4096, 'numerical_entries': 16432,
                         'workers': 2, 'minimum_free_bytes': 12 * 2**30, 'cooperative_seconds': 1800, 'new_training': 0}.items():
        require(budget[key] == expected, 'Declared resource allocation changed: ' + key)
    return cfg, scope, schedule


def check_exact_key_review(run, prior_review_sha):
    binding = read(run / 'exact_key_review_binding.json')
    directory = run / 'exact_key_review'
    require(binding['path'] == 'artifacts/clpu_equivalent_cache_exact_key_review_20261030'
            and sha(directory / 'completion_manifest.json') == binding['completion_sha256']
            and sha(directory / 'summary.json') == binding['summary_sha256'], 'Final-key review binding changed')
    files = manifest(directory)
    require({'summary.json'} | {f'context_{i:02d}.json' for i in range(8)} <= set(files),
            'Final-key review omits fixed contexts')
    review = read(directory / 'summary.json')
    require(review['status'] == 'PASS' and review['source_completion_sha256'] == SOURCE_SHA
            and review['prior_review_manifest_sha256'] == prior_review_sha
            and review['new_physical_points'] == review['new_training'] == 0
            and review['physical_points_reviewed'] == 1121 and review['duplicate_pairs_reviewed'] == 556
            and review['new_group_merges'] == 18 and review['certificate_or_feedback_disagreements'] == 0
            and review['all_differences_in_existing_excluded_disabled_cache_fields'] is True
            and review['constraint_or_audit_tolerances_changed'] is False,
            'Final exact key lacks complete raw/certificate review')
    return review


def check_cell(run, spec, cfg, scope):
    cell = (run / spec['cell']).resolve()
    require(cell != run and cell.is_relative_to(run), 'Cell escaped pilot')
    sealed = manifest(cell)
    required = {'spec.json', 'result.json', 'initial_state.json', 'plans.jsonl.gz', 'steps.jsonl.gz',
        'worker_started.json', 'points.jsonl.gz', 'requests.jsonl', 'command_ledger.jsonl',
        'command_stats.json', 'metadata.json', 'oracle_summary.json', 'logical_points.jsonl.gz',
        'logical_requests.jsonl', 'equivalent_cache_summary.json'}
    require(required <= set(sealed) and not (cell / 'error.json').exists(), 'Missing or failed cache cell evidence')
    result = read(cell / 'result.json')
    require(read(cell / 'spec.json') == result['spec'] == spec and result['config'] == cfg['branch_search'],
            'Cell scientific identity changed')
    worker = read(cell / 'worker_started.json')
    require(worker['spec_sha256'] == sha(cell / 'spec.json') and type(worker['pid']) is int and worker['pid'] > 0,
            'Worker start/spec binding changed')
    require(all(result[key] == 0 for key in ('training_interactions', 'optimizer_updates', 'new_training'))
            and result['formal_holdout_numerical_access'] is result['tune_numerical_access'] is False,
            'Unapproved training or partition access')
    scenarios = {row['scenario_id']: row for row in read(run / 'selected_scenarios.json')}
    events = {row['event_id']: row for row in read(run / 'selected_events.json')}
    mech = branch.Mechanics(read(run / 'author_static_metadata.json'), read(run / 'load_reference.json'),
        scenarios[spec['scenario_id']], read(run / 'condition.json'), events[spec['event_id']], cfg)
    require(read(cell / 'initial_state.json') == mech.initial, 'Initial restoration history changed')
    points, statuses = branch.check_raw_points(run, cell, {**spec, 'point_budget': 256}, cfg, mech)
    commands = old.check_commands(cell, points, read(run / 'command_inventory.json'))
    actual_requests = old.check_requests(cell, points, 256)
    actual = read(cell / 'oracle_summary.json')
    require(actual['status'] == 'COMPLETE_COUNTERS_PENDING_AUDIT' and actual['points'] == len(points)
            and actual['point_budget'] == 256 and actual['actual_numerical_entries'] == commands['numerical_entries']
            and actual['requests'] == actual_requests['requests'] == len(points)
            and actual['cache_hits'] == actual_requests['cache_hits'] == actual_requests['denied'] == 0
            and actual['statuses'] == dict(statuses) and actual['initialization_numerical_entries'] == 3
            and actual['point_seconds'] == [p['elapsed_seconds'] for p in points.values()],
            'Actual cold oracle ledger differs')
    raw = {row['point_id']: row for row in read_lines(cell / 'points.jsonl.gz')}
    logical_rows = list(read_lines(cell / 'logical_points.jsonl.gz'))
    logical, requests, costs = check_logical_records(logical_rows, list(read_lines(cell / 'logical_requests.jsonl')),
        raw, mech, scope, sha(run / 'condition.json'), spec['arm'])
    summary = read(cell / 'equivalent_cache_summary.json')
    check_wrapper_summary(summary, result['oracle'], costs, logical, scope, spec['arm'], commands['numerical_entries'])
    plans, steps = list(read_lines(cell / 'plans.jsonl.gz')), list(read_lines(cell / 'steps.jsonl.gz'))
    checked, semantics = check_search(run, spec, cfg, result, logical_rows, logical, requests, plans, steps, mech)
    same(semantics, historical_semantics(run, spec), 'complete_historical_G1_semantic_equivalence')
    computation = window.finite_time(result['computation_elapsed_seconds'], 'computation duration')
    initialization = window.finite_time(result['runtime_initialization_seconds'], 'initialization duration')
    preflight = window.finite_time(result['preflight_elapsed_seconds'], 'worker preflight duration')
    elapsed = window.finite_time(result['elapsed_seconds'], 'worker total duration')
    require(computation >= initialization + math.fsum(p['elapsed_seconds'] for p in points.values())
            and elapsed >= computation + preflight and computation >= result['checkpoints'][-1]['elapsed_seconds'],
            'Worker time omits initialization, paid physics, preflight or final computation')
    return {'cell': spec['cell'], 'context_id': spec['context_id'], 'arm': spec['arm'],
        'cell_completion_sha256': sha(cell / 'completion_manifest.json'), 'status': 'PASS',
        'logical_points': len(logical), 'physical_points': len(points), 'numerical_entries': commands['numerical_entries'],
        'semantic_cache_hits': costs['semantic_cache_hits'], 'electrical_reuses': costs['electrical_reuses'],
        'requests': costs['requests'], 'physical_statuses': dict(statuses), 'quality_pct': checked['final_quality_pct'],
        'semantic_signature_sha256': digest(semantics), 'historical_semantic_equivalence': True,
        'worker_elapsed_seconds': elapsed, 'computation_elapsed_seconds': computation,
        'runtime_initialization_seconds': initialization, 'preflight_elapsed_seconds': preflight}, semantics


def check_controller(controller, schedule, checks):
    cells = {s['cell'] for s in schedule}
    require(controller['status'] == 'COMPLETE_CELLS_PENDING_AUDIT' and controller['errors'] == []
            and controller['unknown_cost_cells'] == [] and controller['scheduled'] == len(schedule) == 16
            and len(controller['completed']) == 16 and set(controller['completed']) == cells,
            'Controller did not finish the whole fixed population')
    require(len(controller['exits']) == 16 and {row['cell'] for row in controller['exits']} == cells
            and all(row['exit_code'] == 0 for row in controller['exits']), 'Missing or failed cold worker exit')
    points = sum(row['physical_points'] for row in checks)
    logical = sum(row['logical_points'] for row in checks)
    entries = sum(row['numerical_entries'] for row in checks)
    require(controller['known_physical_points'] == points <= 4096
            and controller['known_logical_points'] == logical <= 4096
            and controller['known_numerical_entries'] == entries == 16 * 3 + 4 * points,
            'Controller dual cost ledger differs')
    for row in checks:
        exit_row = next(item for item in controller['exits'] if item['cell'] == row['cell'])
        require(window.finite_time(exit_row['process_wall_seconds'], 'cold process wall') >= row['worker_elapsed_seconds'],
                'Process wall duration omits worker elapsed time')
        row['process_wall_seconds'] = exit_row['process_wall_seconds']
    window.finite_time(controller['wall_seconds'], 'controller elapsed')
    return logical, points, entries


def audit(run, output=None):
    run = Path(run).resolve()
    output = Path(output).resolve() if output else run / 'audit'
    require(not output.exists(), 'Audit output exists; preserve previous attempt')
    output.mkdir(parents=True)
    started, checks, semantics = time.perf_counter(), [], {}
    result = {'status': 'FAIL', 'source_run': run.name, 'errors': [], 'audited_cells': 0,
              'results': checks, 'pairs': [], 'new_audit_physical_points': 0, 'new_training_tasks': 0}
    from audit_d059_supplement import NoSolverImports, no_solver_loaded
    guard = NoSolverImports()
    sys.meta_path.insert(0, guard)
    try:
        cfg, scope, schedule = check_parent(run)
        for spec in schedule:
            checked, semantic = check_cell(run, spec, cfg, scope)
            checks.append(checked)
            if spec['context_id'] in semantics:
                same(semantic, semantics[spec['context_id']], 'paired_complete_semantic_equivalence')
                result['pairs'].append({'context_id': spec['context_id'], 'semantic_equivalence': True,
                    'historical_equivalence': True, 'status': 'PASS'})
                del semantics[spec['context_id']]
            else:
                semantics[spec['context_id']] = semantic
            save(output / f'cell_{spec["episode_id"]:03d}.json', checked)
            save(output / 'progress.json', {'audited_cells': len(checks), 'scheduled': len(schedule)})
        require(not semantics and len(result['pairs']) == 8, 'Unpaired semantic comparisons')
        logical, points, entries = check_controller(read(run / 'controller_completion.json'), schedule, checks)
        no_solver_loaded()
        result.update(status='PASS', audited_cells=len(checks), logical_points=logical,
            physical_points=points, numerical_entries=entries,
            freeze_sha256=sha(run / 'prephysical_freeze_manifest.json'), schedule_sha256=sha(run / 'schedule.json'),
            source_binding_sha256=sha(run / 'source_binding.json'),
            controller_completion_sha256=sha(run / 'controller_completion.json'),
            source_window_completion_sha256=SOURCE_SHA, physical_scope_sha256=digest(scope),
            scope_review_manifest_sha256=sha(run / 'scope_review/completion_manifest.json'),
            exact_key_review_manifest_sha256=sha(run / 'exact_key_review/completion_manifest.json'))
    except BaseException as exc:
        result['errors'].append({'error': repr(exc), 'traceback': traceback.format_exc()})
    finally:
        sys.meta_path.remove(guard)
    result['elapsed_seconds'] = time.perf_counter() - started
    save(output / 'summary.json', result)
    save(output / 'completion_manifest.json', {p.relative_to(output).as_posix(): sha(p)
        for p in output.rglob('*') if p.is_file() and p.name != 'completion_manifest.json'})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-source', action='store_true')
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    require(bool(args.review_source) != bool(args.run), 'Select exactly one explicit audit operation')
    if args.review_source:
        require(args.output is not None, 'Historical review needs a new output path')
    outcome = review_historical_outputs(args.output) if args.review_source else audit(args.run, args.output)
    print(canonical({key: value for key, value in outcome.items() if key != 'contexts'}), flush=True)
    raise SystemExit(0 if outcome['status'] == 'PASS' else 1)
