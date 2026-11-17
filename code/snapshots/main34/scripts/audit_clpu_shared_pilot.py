"""Independent audit of one runtime and two causally scoped logical searches.

No production shared service, runner, search, or solver is imported. Actual
evidence is checked once in physical/; logical aliases are independent views
of that evidence, never fabricated copies of per-view physical journals.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import math
from pathlib import Path
import sys
import time
import traceback
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
sys.dont_write_bytecode = True

import audit_clpu_early_pilot as early

prior, cache, branch, window, tail, old = early.prior, early.cache, early.branch, early.window, early.tail, early.old
require, close, same = early.require, early.close, early.same
read, save, sha, digest, manifest, read_lines = early.read, early.save, early.sha, early.digest, early.manifest, early.read_lines
SOURCE_RUN = '20261031T020246_028867Z_clpu_early_pilot'
SOURCE_SHA = 'bed648a067dd82c3c52e0250de25714efcea7cca5b69347fad399e72e9e57747'
STUDY = 'CLPU_SHARED_RULE_PILOT_V1'
SERVICE_VERSION = 'clpu-shared-physical-evidence-v1'
VIEWS = ('EARLY', 'GUARD')
MODES = ('private', 'shared')


def checked_manifest(directory, name='completion_manifest.json'):
    """Verify bytes and normalize Windows/POSIX paths without accepting aliases."""
    directory = Path(directory).resolve()
    rows = read(directory / name)
    require(isinstance(rows, dict), 'Manifest must be an object')
    normalized, resolved = {}, set()
    for relative, expected in rows.items():
        require(isinstance(relative, str) and isinstance(expected, str) and len(expected) == 64
                and all(c in '0123456789abcdef' for c in expected), 'Malformed manifest entry')
        key = relative.replace('\\', '/')
        parts = key.split('/')
        require(not Path(key).is_absolute() and not Path(key).drive and not any(p in ('', '.', '..') for p in parts),
                'Absolute, empty, or traversal manifest path')
        target = (directory / key).resolve()
        alias = str(target).casefold()
        require(target.is_relative_to(directory) and target != directory and key not in normalized
                and alias not in resolved and sha(target) == expected, 'Changed, ambiguous, or escaped manifest path')
        normalized[key] = expected
        resolved.add(alias)
    return normalized


def check_actual_points(run, directory, spec, mech):
    """Unchanged independent per-point physics, with per-pool uniqueness later.

Private mode legitimately repeats a semantic identity in another pool. Global
physical point IDs and native call costs remain unique and consecutive.
"""
    metadata, physical = read(directory / 'metadata.json'), read(run / 'physical_config.json')
    require(metadata == read(run / 'source_inputs/metadata.json'), 'Inherited device metadata changed')
    reference = read(run / 'operating_reference.json')
    points, statuses = {}, Counter()
    for row in read_lines(directory / 'points.jsonl.gz'):
        point_id = row['point_id']
        require(type(point_id) is int and point_id == len(points) and point_id < 512,
                'Actual physical point sequence or global cap changed')
        state = row['dynamic_state']
        require(type(state['step']) is int and 0 <= state['step'] <= mech.horizon
                and len(state['pickups']) == len(mech.names) == 68, 'Raw dynamic dimensions differ')
        projection = old.independent_projection(mech.static, mech.loads, mech.scenario, state['command'])
        for name, pickup, initial in zip(mech.names, state['pickups'], mech.initial['pickups']):
            require((pickup == -1) == (initial == -1) and (pickup is not None) == (name in projection['energized']),
                    'Forged steady/unserved history')
            if pickup is not None and pickup != -1:
                require(type(pickup) in (int, float) and math.isfinite(pickup) and pickup >= 0
                        and state['step'] > 0 and pickup <= (state['step'] - 1) * mech.slot
                        and pickup % mech.slot == 0, 'Off-grid/future pickup')
        seconds = row['state_identity']['time_seconds']
        require(seconds % mech.sample == 0 and 0 <= seconds <= mech.horizon * mech.slot, 'Off-grid physical time')
        identity = mech.identity(state, seconds, sha(run / 'condition.json'))
        require(row['state_identity'] == identity and row['identity_sha256'] == digest(identity), 'Raw identity differs')
        require(row['case'] == spec['scenario_id'] and row['fault_lines'] == mech.scenario['fault_lines']
                and row['alpha'] == .5 and row['algorithm'] == 0, 'Raw physical context changed')
        enabled = {}
        for name, layout in metadata['element_layout'].items():
            buses = [b.split('.')[0].lower() for b in layout['buses']]
            value = all(b in projection['powered'] for b in buses)
            if layout['kind'] in ('line', 'transformer'):
                endpoints = list(dict.fromkeys(buses))
                value = value and len(endpoints) == 2 and projection['graph'].has_edge(*endpoints)
            if layout['kind'] == 'load':
                value = name in projection['energized']
            enabled[name] = bool(value)
        require(row['metadata_override']['planned_enabled'] == enabled, 'Device isolation differs')
        checked = old.check_saved_point(row, mech.condition, reference, metadata, physical,
            identity['multipliers'], projection['action'], seconds / 60., projection['energized'])
        require(checked['status'] != 'UNKNOWN', 'Unknown point in completed shared experiment')
        statuses[checked['status']] += 1
        points[point_id] = {k: row[k] for k in ('status', 'reliable', 'clpu_multipliers', 'action', 'time_minutes',
            'state_identity', 'identity_sha256', 'actual_call_counts', 'dynamic_state', 'elapsed_seconds', 'structured_feedback')}
        points[point_id]['physical_evidence_sha256'] = digest({k: row[k] for k in ('raw_outputs', 'metadata_override')})
    require(points, 'Empty actual evidence population')
    return points, statuses


def check_actual_requests(rows, points, summary, commands, statuses):
    """Global native dispatches may repeat identity only through separate pools."""
    require(len(rows) == 2 * len(points), 'Actual dispatch omitted, cached, denied, or unclosed')
    for pid, point in points.items():
        for offset, status in ((0, 'STARTED'), (1, point['status'])):
            require(rows[2 * pid + offset] == {'request': pid + 1, 'key': point['identity_sha256'],
                'cache_hit': False, 'point_id': pid, 'status': status}, 'Native actual request identity/order changed')
    require(summary['status'] == 'COMPLETE_COUNTERS_PENDING_AUDIT' and summary['points'] == len(points)
        and summary['point_budget'] == 512 and summary['actual_numerical_entries'] == commands['numerical_entries']
        and summary['requests'] == len(points) and summary['cache_hits'] == 0
        and summary['statuses'] == dict(statuses) and summary['initialization_numerical_entries'] == 3
        and summary['point_seconds'] == [p['elapsed_seconds'] for p in points.values()], 'Actual service accounting changed')


def check_service_ledger(events, views, points, mech, scope, condition_sha, mode):
    """Rebuild every service event and pool from initially empty dictionaries."""
    require(mode in MODES and list(views) == list(VIEWS), 'Service mode or view population changed')
    pools, source_records, checked, event_index, actual_count = {}, {}, {}, 0, 0

    def event(kind, **fields):
        nonlocal event_index
        require(event_index < len(events), 'Missing service event: ' + kind)
        expected = {'sequence': event_index + 1, 'kind': kind, **fields}
        require(events[event_index] == expected, 'Service chronology/payload differs: ' + kind)
        event_index += 1
        return event_index

    event('SERVICE_STARTED', mode=mode, scope_sha256=digest(scope), physical_budget=512, logical_budget=256)
    event('SERVICE_READY', physical_points=0)
    for name in VIEWS:
        pool = 'SHARED' if mode == 'shared' else name
        evidence_pool = pools.setdefault(pool, {})
        logical_rows, request_rows = views[name]['logical_rows'], views[name]['requests']
        logical, semantic, transformed = {}, {}, []
        hits = aliases = crosses = denied = new_actual = 0
        alias_seconds = 0.
        event('VIEW_STARTED', view=name, pool=pool, physical_points_before=actual_count, logical_budget=256)
        for request_id, request in enumerate(request_rows, 1):
            require(request['request'] == request_id and type(request['request']) is int,
                    'Local request identity is not consecutive')
            key = request['key']
            require(isinstance(key, str) and len(key) == 64 and all(c in '0123456789abcdef' for c in key),
                    'Malformed semantic request hash')
            start_seq = event('REQUEST_STARTED', view=name, pool=pool, request=request_id,
                semantic_identity_sha256=key, logical_points_before=len(logical), physical_points_before=actual_count)
            require(request['service_request_sequence'] == start_seq, 'Request cites wrong service start')
            expected_request = {'request': request_id, 'key': key, 'service_request_sequence': start_seq,
                'semantic_cache_hit': False, 'electrical_reuse': False, 'cross_view_reuse': False}
            base = {'request': request_id, 'key': key, 'cache_hit': key in semantic}
            if key in semantic:
                lid = semantic[key]
                row = logical_rows[lid]
                expected_request.update(semantic_cache_hit=True, status=logical[lid]['status'],
                    logical_point_id=lid, physical_point_id=row['physical_point_id'],
                    source_view=row['source_view'], source_event_sequence=row['source_event_sequence'])
                transformed.append({**base, 'point_id': lid})
                hits += 1
            elif len(logical) == 256:
                expected_request['status'] = 'BUDGET_UNKNOWN'
                transformed.append({**base, 'status': 'BUDGET_UNKNOWN'})
                denied += 1
            else:
                lid = len(logical)
                require(lid < len(logical_rows), 'Request lacks logical evidence')
                row = logical_rows[lid]
                state, seconds = row['dynamic_state'], row['time_seconds']
                require(type(state['step']) is int and 0 <= state['step'] <= mech.horizon
                    and type(seconds) in (int, float) and math.isfinite(seconds)
                    and 0 <= seconds <= mech.horizon * mech.slot and seconds % mech.sample == 0,
                    'Logical state/time outside scope')
                identity = mech.identity(state, seconds, condition_sha)
                require(row['semantic_identity'] == identity and row['semantic_identity_sha256'] == key == digest(identity),
                        'Logical semantic identity differs')
                electrical = cache.electrical_key(scope, mech, identity)
                fingerprint = digest(electrical)
                require(row['electrical_input_key'] == electrical and row['electrical_input_sha256'] == fingerprint,
                        'Incomplete or approximate solver-input equivalence')
                reused = fingerprint in evidence_pool
                if not reused:
                    pid = actual_count
                    require(pid in points, 'Undocumented actual physical dispatch')
                    paid = points[pid]
                    require(paid['state_identity'] == identity and paid['dynamic_state'] == state,
                            'Real dispatch belongs to another semantic query')
                    event('ACTUAL_STARTED', view=name, pool=pool, physical_point_id=pid,
                        semantic_identity_sha256=key, electrical_input_key=electrical, electrical_input_sha256=fingerprint)
                    produced = event('ACTUAL_COMPLETE', view=name, pool=pool, physical_point_id=pid,
                        status=paid['status'], physical_evidence_sha256=paid['physical_evidence_sha256'],
                        electrical_input_sha256=fingerprint)
                    evidence_pool[fingerprint] = (pid, name, produced)
                    source_records[pid] = {'view': name, 'pool': pool, 'sequence': produced, 'key': fingerprint}
                    actual_count += 1
                    new_actual += 1
                pid, source_view, produced = evidence_pool[fingerprint]
                paid = points[pid]
                require(cache.electrical_key(scope, mech, paid['state_identity']) == electrical,
                        'Reused evidence has a different exact physical input')
                reuse_kind = 'ELECTRICAL_EQUIVALENCE' if reused else 'ACTUAL_PHYSICAL'
                crossed = reused and source_view != name
                completion = event('LOGICAL_COMPLETE', view=name, pool=pool, request=request_id,
                    logical_point_id=lid, physical_point_id=pid, source_event_sequence=produced,
                    electrical_input_sha256=fingerprint, reuse_kind=reuse_kind)
                feedback = cache.retimed_feedback(paid['structured_feedback'], seconds / 60.)
                certificate = {'point_id': lid, 'status': paid['status'], 'reliable': paid['reliable'],
                    'feedback': feedback['violations']}
                require(certificate['reliable'] is True and certificate['status'] in ('PASS', 'FAIL'),
                        'Unknown/unreliable evidence used as a certificate')
                duration = window.finite_time(row['alias_certification_seconds'], 'alias certification time')
                require(reused or duration == 0., 'Fresh physical point claims alias work')
                expected_row = {'view': name, 'pool': pool, 'logical_point_id': lid, 'physical_point_id': pid,
                    'service_sequence': completion, 'source_event_sequence': produced, 'source_view': source_view,
                    'semantic_identity': identity, 'semantic_identity_sha256': key,
                    'electrical_input_key': electrical, 'electrical_input_sha256': fingerprint,
                    'dynamic_state': state, 'time_seconds': seconds, 'certificate': certificate,
                    'reuse_kind': reuse_kind, 'cross_view_reuse': crossed,
                    'physical_evidence_sha256': paid['physical_evidence_sha256'], 'alias_certification_seconds': duration}
                require(row == expected_row and produced < completion, 'Logical alias provenance or current-time certificate differs')
                require(not crossed or mode == 'shared', 'Private pool crossed view boundary')
                logical[lid] = {'status': paid['status'], 'reliable': paid['reliable'],
                    'clpu_multipliers': identity['multipliers'], 'action': electrical['action'], 'time_minutes': seconds / 60.,
                    'state_identity': identity, 'identity_sha256': key, 'dynamic_state': state,
                    'elapsed_seconds': 0. if reused else paid['elapsed_seconds'], 'structured_feedback': feedback}
                semantic[key] = lid
                aliases += int(reused)
                crosses += int(crossed)
                alias_seconds += duration
                transformed.extend(({**base, 'point_id': lid, 'status': 'STARTED'},
                    {**base, 'point_id': lid, 'status': paid['status']}))
                expected_request.update(status=paid['status'], electrical_reuse=reused, cross_view_reuse=crossed,
                    logical_point_id=lid, physical_point_id=pid, source_view=source_view, source_event_sequence=produced)
            end_seq = event('REQUEST_COMPLETE', view=name, **expected_request)
            require(request == {**expected_request, 'service_completion_sequence': end_seq}, 'Local/service request differs')
        require(logical and len(logical) == len(logical_rows), 'Unrequested/empty logical points')
        event('VIEW_CLOSED', view=name, status='CLOSED_LOGICAL_VIEW_PENDING_AUDIT',
            physical_points_after=actual_count, logical_points=len(logical))
        checked[name] = {'logical': logical, 'request_rows': transformed, 'logical_rows': logical_rows,
            'logical_points': len(logical), 'physical_points': new_actual, 'requests': len(request_rows),
            'semantic_cache_hits': hits, 'electrical_reuses': aliases, 'cross_view_reuses': crosses,
            'denied': denied, 'alias_certification_seconds': alias_seconds,
            'statuses': dict(Counter(p['status'] for p in logical.values()))}
    event('SERVICE_CLOSED', status='COMPLETE_SHARED_EVIDENCE_PENDING_AUDIT', physical_points=actual_count)
    require(event_index == len(events) and actual_count == len(points) and set(points) == set(range(actual_count)),
            'Extra events or missing/retried/unbound physical evidence')
    return checked, source_records


def check_service_summaries(summary, view_summaries, checked, scope, mode, numerical):
    expected_views = {}
    for name in VIEWS:
        c = checked[name]
        expected = {'version': SERVICE_VERSION, 'view': name, 'pool': 'SHARED' if mode == 'shared' else name,
            'mode': mode, 'scope_sha256': digest(scope), 'status': 'CLOSED_LOGICAL_VIEW_PENDING_AUDIT', 'closed': True,
            'points': c['logical_points'], 'logical_points': c['logical_points'], 'point_budget': 256,
            'physical_points': c['physical_points'], 'requests': c['requests'], 'cache_hits': c['semantic_cache_hits'],
            'semantic_cache_hits': c['semantic_cache_hits'], 'electrical_reuses': c['electrical_reuses'],
            'cross_view_reuses': c['cross_view_reuses'], 'statuses': c['statuses'],
            'alias_certification_seconds': c['alias_certification_seconds'],
            'initialization_numerical_entries_attributed': 0, 'actual_numerical_entries_attributed': 4 * c['physical_points'],
            'expected_numerical_entries_attributed': 4 * c['physical_points'], 'physical_evidence_directory': '../physical'}
        same(view_summaries[name], expected, 'independent_view_summary')
        expected_views[name] = view_summaries[name]
    actual = sum(c['physical_points'] for c in checked.values())
    require(numerical == 3 + 4 * actual, 'One-runtime numerical costs do not close')
    expected = {'version': SERVICE_VERSION, 'mode': mode, 'status': 'COMPLETE_SHARED_EVIDENCE_PENDING_AUDIT',
        'closed': True, 'scope_sha256': digest(scope), 'physical_points': actual, 'physical_budget': 512,
        'logical_budget_per_view': 256, 'logical_points': sum(c['logical_points'] for c in checked.values()),
        'actual_numerical_entries': numerical, 'initialization_numerical_entries': 3,
        'expected_numerical_entries': numerical, 'physical_cost_complete': True, 'runtime_initializations': 1,
        'runtime_initialization_attempts': 1,
        'cross_task_evidence_reuse': False, 'views': expected_views}
    same(summary, expected, 'independent_single_runtime_summary')


def phase_counters(oracle, logical_rows):
    require(0 <= oracle.used <= len(logical_rows), 'Unbound logical checkpoint')
    actual = sum(row['reuse_kind'] == 'ACTUAL_PHYSICAL' for row in logical_rows[:oracle.used])
    return {'logical_points': oracle.used, 'physical_points': actual, 'numerical_entries': 4 * actual,
            'requests': oracle.counts()['requests']}


def check_phase_records(mech, cfg, name, spec, result, attempts, decisions, evidence, condition_sha):
    require(name in VIEWS and spec['method'] == 'beam', 'Unexpected logical search or learned selector')
    oracle = branch.PaidRequests(evidence['request_rows'], evidence['logical'], mech, 256, condition_sha)
    replay = (window.WindowReplay if name == 'EARLY' else branch.BranchReplay)(mech, cfg, oracle, attempts)
    require(len(decisions) % 2 == 0, 'Unclosed rule decision')
    boundaries = []

    def boundary(choice):
        boundaries.append({'phase_choices': choice, **phase_counters(oracle, evidence['logical_rows']),
            'quality_pct': 100 * replay.quality if replay.best is not None else None,
            'fully_certified': replay.best is not None})

    boundary(0)
    for choice in range(len(decisions) // 2):
        start, finish = decisions[2 * choice:2 * choice + 2]
        require(not replay.done and replay.turn == choice, 'Rule query after stop or reset')
        identity = {'transition_id': choice, 'phase_choice': choice, 'phase': name,
            'method': 'beam', 'action': replay.rule('beam'), 'oracle_before': phase_counters(oracle, evidence['logical_rows'])}
        same(start, {**identity, 'status': 'STARTED'}, 'independent_rule_decision')
        response = replay.step(start['action'])
        same(finish, {**identity, 'status': 'COMPLETED', 'response': response,
            'oracle_after': phase_counters(oracle, evidence['logical_rows'])}, 'independent_rule_response')
        boundary(choice + 1)
    require(replay.done and oracle.index == len(oracle.rows) and oracle.used == len(evidence['logical'])
            and replay.attempt_index == len(attempts), 'Unfinished or unbound independent search')
    scientific = tail.baseline_snapshot(replay)
    normalized = {**result, 'actual_physical_points': result['logical_points']}
    for key, value in scientific.items():
        same(normalized[key], value, 'complete_independent_search.' + key)
    require(result['logical_points'] == evidence['logical_points'] and result['actual_physical_points'] == evidence['physical_points']
            and result['search_decisions'] == replay.turn <= cfg['max_branch_expansions'] and result['policy_decisions'] == 0,
            'Logical search budget or attributed actual cost differs')
    require(result['fully_certified'] is (replay.best is not None) and result['quality_pct'] == boundaries[-1]['quality_pct'],
            'Logical search certificate differs')
    return {'phase': name, 'decisions': replay.turn, 'policy_decisions': 0, 'quality_pct': boundaries[-1]['quality_pct'],
        'fully_certified': replay.best is not None, 'stop_reason': replay.reason, 'boundaries': boundaries,
        'scientific_result': scientific, 'semantic_signature_sha256': digest(scientific)}


def protocol_settings():
    return {'cells': 16, 'contexts': 8, 'modes': list(MODES), 'phases': list(VIEWS),
        'method': 'beam', 'seed': 0, 'logical_budget_per_view': 256, 'expansion_cap_per_view': 256,
        'total_physical_budget_per_cell': 512, 'total_physical_budget': 8192,
        'maximum_numerical_entries': 32816, 'physical_runtimes_per_cell': 1,
        'max_workers': 2, 'worker_seconds': 900, 'controller_seconds': 3600,
        'minimum_free_disk_bytes': 12 * 2**30, 'storage_margin': 1.25, 'report_reserve_bytes': 2 * 2**30,
        'phase_order': list(VIEWS), 'tie_rule': 'exact_equal_energy_select_GUARD',
        'same_runtime_both_modes': True, 'new_training': 0, 'no_cross_cell_cache': True,
        'no_historical_numerical_answers': True, 'no_automatic_retry': True,
        'no_upper_bound_change': True, 'no_model_intervention': True,
        'timing': 'cold complete cell process; all initialization, both searches, logging and exit paid',
        'recording': 'rule choices and exact step responses; no redundant unused model feature tensor',
        'dispatch': 'hashed context order; alternating mode order; adjacent pair drained before next pair'}


def expected_schedule(contexts):
    require(len(contexts) == 8 and {r['context_id'] for r in contexts} == set(range(8))
            and len({r['scenario_id'] for r in contexts}) == 8, 'Eight fixed distinct contexts required')
    ordered = sorted(contexts, key=lambda c: prior.digest_text('shared-pilot-v1|' + prior.canonical(c)))
    result = []
    for rank, context in enumerate(ordered):
        for mode in (MODES if rank % 2 == 0 else tuple(reversed(MODES))):
            index = len(result)
            result.append({**context, 'cell': f'episodes/{index:03d}', 'episode_id': index, 'pair_order': rank,
                'mode': mode, 'arm': 'protected_rule', 'method': 'beam', 'seed': 0,
                'phase': 'evaluation', 'group': 'shared_evidence_rule', 'protected': True,
                'representation': 'raw', 'deterministic': True, 'checkpoint': None, 'checkpoint_sha256': None,
                'point_budget': 256, 'logical_point_budget': 256, 'physical_point_budget': 512,
                'total_logical_point_budget': 512, 'window_variant': 'coverage_v1'})
    return result


def expected_phase_spec(spec, name):
    require(name in VIEWS, 'Unknown logical view')
    return {**spec, 'parent_cell': spec['cell'], 'cell': spec['cell'] + '/' + name, 'search_phase': name,
        'point_budget': 256, 'logical_point_budget': 256, 'physical_point_budget': 256}


def check_parent(run):
    run = Path(run).resolve()
    require(run.parent == ROOT / 'runs', 'Shared pilot must be a project run')
    prepared = read(run / 'prepared.json')
    require(prepared['status'] == 'FROZEN_PREPARED_NO_DSS'
            and prepared['freeze_sha256'] == sha(run / 'prephysical_freeze_manifest.json'), 'Freeze changed')
    frozen = checked_manifest(run, 'prephysical_freeze_manifest.json')
    source = ROOT / 'runs' / SOURCE_RUN
    require(sha(source / 'completion_manifest.json') == sha(run / 'source_completion_manifest.json') == SOURCE_SHA,
            'Original completed source binding changed')
    source_seal = {k.replace('\\', '/'): v for k, v in read(run / 'source_completion_manifest.json').items()}
    require(read(run / 'source_binding.json') == {'source_run': SOURCE_RUN, 'source_completion_sha256': SOURCE_SHA},
            'Source identity changed')
    for name in ('config.json', 'schedule.json', 'prepared.json', 'audit/summary.json'):
        require(sha(source / name) == source_seal[name], 'Unbound source scientific input: ' + name)
    source_audit = read(source / 'audit/summary.json')
    require(source_audit['status'] == 'PASS' and source_audit['audited_cells'] == 64 and not source_audit['errors'],
            'Historical source not completely audited')
    additions = {'src/dnr/nature34_clpu_shared_evidence.py', 'docs/clpu_shared_pilot_protocol_v1.md',
        'scripts/run_clpu_shared_pilot.py', 'scripts/audit_clpu_shared_pilot.py', 'scripts/report_clpu_shared_pilot.py',
        'scripts/plot_clpu_shared_pilot.py', 'tests/test_clpu_shared_report_runner.py',
        'tests/test_nature34_clpu_shared_evidence.py', 'tests/test_run_clpu_shared_pilot.py',
        'tests/test_clpu_shared_audit.py', 'tests/test_report_clpu_shared_pilot.py'}
    expected_sources = set(read(source / 'prepared.json')['source_files']) | additions
    require(prepared['source_files'] == sorted(expected_sources), 'Complete original/new source closure changed')
    for name in prepared['source_files']:
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT) and path != ROOT and 'source/' + name in frozen
                and sha(path) == sha(run / 'source' / name), 'Current source not exactly frozen: ' + name)
        if 'source/' + name in source_seal:
            require(sha(path) == source_seal['source/' + name], 'Inherited scientific source changed: ' + name)
    require(sha(run / 'protocol.md') == sha(ROOT / 'docs/clpu_shared_pilot_protocol_v1.md'), 'Protocol differs')
    cfg = read(run / 'config.json')
    same(cfg, {**read(source / 'config.json'), 'shared_study': STUDY, 'shared_pilot': protocol_settings()},
         'complete_shared_scientific_contract')
    scope = prior.check_frozen_physics(run, prepared, cfg)
    chosen = [s for s in read(source / 'schedule.json') if s['arm'] == 'protected_rule']
    contexts = [{k: s[k] for k in ('context_id', 'scenario_id', 'event_id')} for s in chosen]
    require(read(run / 'contexts.json') == contexts, 'Historical training population changed')
    schedule = read(run / 'schedule.json')
    same(schedule, expected_schedule(contexts), 'all_16_shared_rule_cells')
    for spec in schedule:
        require(read(run / spec['cell'] / 'spec.json') == spec, 'Prepared cell differs')
        for name in VIEWS:
            require(read(run / spec['cell'] / name / 'spec.json') == expected_phase_spec(spec, name),
                    'Prepared cold logical view differs')
    expected_history = []
    for spec in chosen:
        row = {k: spec[k] for k in ('context_id', 'scenario_id', 'event_id')}
        row.update(source_cell=spec['cell'], phases={})
        for name in VIEWS:
            original = spec['cell'] + '/' + name.lower() + '/phase_result.json'
            target = f'historical/c{spec["context_id"]:02d}/{name}/phase_result.json'
            require(sha(run / target) == source_seal[original] == sha(source / original), 'Historical search changed')
            row['phases'][name] = {'result_file': target, 'source_file': original, 'sha256': source_seal[original]}
        expected_history.append(row)
    require(read(run / 'historical_bindings.json') == expected_history, 'Historical phase attribution changed')
    cases = list(ET.parse(run / 'software_tests.xml').getroot().iter('testcase'))
    require(cases and not any(list(c.iter('failure')) + list(c.iter('error')) + list(c.iter('skipped')) for c in cases),
            'Passed complete frozen tests required')
    for component in ('test_nature34_clpu_shared_evidence', 'test_run_clpu_shared_pilot',
                      'test_clpu_shared_audit', 'test_report_clpu_shared_pilot'):
        require(any(component in c.get('classname', '') for c in cases), 'Missing component tests: ' + component)
    resource = read(run / 'resource_budget.json')
    for key, value in {'total_physical_cap': 8192, 'minimum_free_bytes': 12 * 2**30,
        'report_reserve_bytes': 2 * 2**30, 'successful_component_tests': len(cases)}.items():
        require(resource[key] == value, 'Resource contract differs: ' + key)
    rate = max(sum(p.stat().st_size for p in (source / s['cell']).rglob('*') if p.is_file()) /
               max(1, read(source / s['cell'] / 'result.json')['actual_physical_points']) for s in chosen)
    require(resource['historical_max_bytes_per_actual_point'] == rate
            and resource['forecast_bytes'] == math.ceil(rate * 8192 * 1.25)
            and resource['required_free_bytes'] == 14 * 2**30 + resource['forecast_bytes']
            and resource['free_bytes_at_prepare'] >= resource['required_free_bytes'], 'Storage forecast incomplete')
    require(read(run / 'authorization.json')['execution_authorized'] is True, 'Execution authorization missing')
    return cfg, scope, schedule


def check_checkpoints(checkpoints, phase_checks, phase_results):
    expected = []
    paid = dict(logical_points=0, physical_points=0, requests=0, choices=0)
    previous_quality = None
    last_time = 0.
    for name in VIEWS:
        checked = phase_checks[name]
        boundaries = checked['boundaries']
        records = [('INITIAL', boundaries[0])]
        records.extend(('DECISION', b) for b in boundaries[1:])
        records.append(('CLOSE', boundaries[-1]))
        for kind, b in records:
            qs = [q for q in (previous_quality, b['quality_pct']) if q is not None]
            expected.append({'checkpoint_index': len(expected), 'phase': name, 'kind': kind,
                'phase_choices': b['phase_choices'], 'search_choices': paid['choices'] + b['phase_choices'],
                'logical_points': paid['logical_points'] + b['logical_points'],
                'physical_points': paid['physical_points'] + b['physical_points'],
                'numerical_entries': 3 + 4 * (paid['physical_points'] + b['physical_points']),
                'requests': paid['requests'] + b['requests'], 'phase_logical_points': b['logical_points'],
                'phase_physical_points': b['physical_points'], 'phase_numerical_entries': 4 * b['physical_points'],
                'phase_quality_pct': b['quality_pct'], 'quality_pct': max(qs) if qs else None,
                'fully_certified': bool(qs), 'guard_established': name == 'GUARD' and kind == 'CLOSE' and b['fully_certified']})
        end = boundaries[-1]
        for k in ('logical_points', 'physical_points', 'requests'):
            paid[k] += end[k]
        paid['choices'] += end['phase_choices']
        previous_quality = expected[-1]['quality_pct']
    require(len(checkpoints) == len(expected), 'Quality/cost boundaries omitted or invented')
    for actual, wanted in zip(checkpoints, expected):
        require(set(actual) == set(wanted) | {'worker_elapsed_seconds'}, 'Checkpoint fields differ')
        same({k: v for k, v in actual.items() if k != 'worker_elapsed_seconds'}, wanted, 'independent_quality_cost_boundary')
        tick = window.finite_time(actual['worker_elapsed_seconds'], 'whole worker checkpoint time')
        phase = phase_results[actual['phase']]
        require(last_time <= tick and phase['worker_start_seconds'] <= tick <= phase['worker_end_seconds'],
                'Checkpoint clock reset, or outside its actual phase')
        last_time = tick
    return expected


def check_selection(result, phases):
    require(result['version'] == result['study'] == STUDY and result['schema'] == 'clpu-shared-protected-result-v1'
            and result['protected'] is True and result['intermediate_beam_floor_claimed'] is False,
            'Protected output contract changed')
    selected = early.protected_choice(phases['EARLY'], phases['GUARD'], True)
    require(result['selected_phase'] == selected['selected_phase'] and result['best_plan'] == selected['best_plan'],
            'Output lost exact primary-energy protection or GUARD tie')
    chosen, guard = phases[selected['selected_phase']], phases['GUARD']
    require(result['status'] == chosen['status'] and result['fully_certified'] == chosen['fully_certified']
            and result['quality_pct'] == chosen['quality_pct'] and result['early_quality_pct'] == phases['EARLY']['quality_pct']
            and result['baseline_quality_pct'] == guard['quality_pct'], 'Final certified quality differs')
    before = early.complete_plan(guard)
    expected = {'guard_established': selected['guard_established'], 'baseline_status': guard['status'],
        'baseline_energy_kwh': before['validation']['energy_kwh'] if before else None,
        'final_energy_kwh': selected['best_plan']['validation']['energy_kwh'] if before else None,
        'nondegradation_verified': selected['guard_established'], 'tie_rule': 'exact_equal_energy_select_guard',
        'scope': 'Fixed 30-minute normal-service objective; computation waiting excluded; final completed output only'}
    require(result['guard_audit'] == expected and selected['guard_established'], 'Complete certified original Beam floor absent')
    return selected


def check_times(result, phases, finished, physical_seconds):
    times = {k: window.finite_time(result[k], k) for k in ('worker_elapsed_seconds', 'preflight_elapsed_seconds',
        'computation_elapsed_seconds', 'runtime_initialization_seconds')}
    close(times['worker_elapsed_seconds'], times['preflight_elapsed_seconds'] + times['computation_elapsed_seconds'],
          'one_worker_clock_decomposition')
    require(times['runtime_initialization_seconds'] <= times['computation_elapsed_seconds'], 'Initialization time excluded')
    cursor = times['preflight_elapsed_seconds'] + times['runtime_initialization_seconds']
    for name in VIEWS:
        p = phases[name]
        start, end, elapsed = [window.finite_time(p[k], k) for k in
                              ('worker_start_seconds', 'worker_end_seconds', 'phase_elapsed_seconds')]
        require(cursor <= start <= end <= times['worker_elapsed_seconds'] and elapsed <= end - start + 1e-8,
                'Logical phases overlap, restart time, or exclude paid work')
        require(set(p['stage_component_seconds']) == {'selection_seconds', 'search_step_seconds', 'journal_seconds'},
                'Stage timing fields changed')
        stage = sum(window.finite_time(v, k) for k, v in p['stage_component_seconds'].items())
        require(stage <= elapsed + 1e-8, 'Stage timers overlap or exceed complete phase time')
        cursor = end
    require(physical_seconds <= times['computation_elapsed_seconds'] + 1e-8, 'Paid physical time omitted')
    require(window.finite_time(finished['worker_elapsed_seconds'], 'worker finished time') >= times['worker_elapsed_seconds']
            >= result['checkpoints'][-1]['worker_elapsed_seconds'], 'Result or final checkpoint omits worker time')


def check_cell(run, spec, cfg, scope):
    cell = (run / spec['cell']).resolve()
    require(cell.is_relative_to(run) and cell != run, 'Escaped cell')
    sealed = checked_manifest(cell)
    required = {'spec.json', 'result.json', 'worker_started.json', 'worker_finished.json', 'execution.json',
        'service_events.jsonl', 'service_summary.json', 'cost_checkpoints.jsonl', 'physical/points.jsonl.gz',
        'physical/requests.jsonl', 'physical/oracle_summary.json', 'physical/command_ledger.jsonl'}
    for name in VIEWS:
        required |= {name + '/' + p for p in ('spec.json', 'phase_result.json', 'initial_state.json', 'plans.jsonl.gz',
            'decisions.jsonl.gz', 'logical_points.jsonl.gz', 'logical_requests.jsonl', 'view_summary.json')}
        require(not (cell / name / 'points.jsonl.gz').exists() and not (cell / name / 'command_ledger.jsonl').exists(),
                'Logical view fabricates a separate physical ledger')
    require(required <= set(sealed) and not (cell / 'error.json').exists(), 'Incomplete/failed task evidence')
    result = read(cell / 'result.json')
    require(read(cell / 'spec.json') == result['spec'] == spec and result['new_training'] == result['policy_decisions'] == 0,
            'Cell identity or learned work changed')
    worker, finished, execution = [read(cell / p) for p in ('worker_started.json', 'worker_finished.json', 'execution.json')]
    require(type(worker['pid']) is int and worker['pid'] > 0 and finished['pid'] == worker['pid']
            and finished['status'] == 'COMPLETE' and finished['result_sha256'] == sha(cell / 'result.json'),
            'Successful single-worker completion not bound')
    global_execution = read(run / 'execution_started.json')
    require(execution['spec_sha256'] == sha(cell / 'spec.json') and execution['controller_pid'] == global_execution['pid']
            and execution['deadline_epoch_seconds'] == min(global_execution['deadline_epoch_seconds'],
                                                           execution['started_epoch_seconds'] + 900), 'Wrong cold-worker deadline')
    scenarios = {r['scenario_id']: r for r in read(run / 'selected_scenarios.json')}
    events = {r['event_id']: r for r in read(run / 'selected_events.json')}
    mech = branch.Mechanics(read(run / 'author_static_metadata.json'), read(run / 'load_reference.json'),
        scenarios[spec['scenario_id']], read(run / 'condition.json'), events[spec['event_id']], cfg)
    points, statuses = check_actual_points(run, cell / 'physical', spec, mech)
    commands = old.check_commands(cell / 'physical', points, read(run / 'command_inventory.json'))
    check_actual_requests(list(read_lines(cell / 'physical/requests.jsonl')), points,
        read(cell / 'physical/oracle_summary.json'), commands, statuses)
    views = {name: {'logical_rows': list(read_lines(cell / name / 'logical_points.jsonl.gz')),
                   'requests': list(read_lines(cell / name / 'logical_requests.jsonl'))} for name in VIEWS}
    evidence, _ = check_service_ledger(list(read_lines(cell / 'service_events.jsonl')), views, points, mech,
                                     scope, sha(run / 'condition.json'), spec['mode'])
    service = read(cell / 'service_summary.json')
    check_service_summaries(service, {name: read(cell / name / 'view_summary.json') for name in VIEWS},
                            evidence, scope, spec['mode'], commands['numerical_entries'])
    require(result['service'] == service, 'Top service accounting differs')
    phases, checks = {}, {}
    history = next(r for r in read(run / 'historical_bindings.json') if r['context_id'] == spec['context_id'])
    for name in VIEWS:
        directory = cell / name
        p = read(directory / 'phase_result.json')
        require(read(directory / 'spec.json') == p['spec'] == expected_phase_spec(spec, name)
                and read(directory / 'initial_state.json') == mech.initial, 'Search identity/initial state changed')
        require(p['oracle'] == service['views'][name], 'Search view summary differs')
        plans = list(read_lines(directory / 'plans.jsonl.gz'))
        require(p['attempts'] == plans, 'Original plan attempt ledger changed')
        checked = check_phase_records(mech, cfg['branch_search'], name, p['spec'], p, plans,
            list(read_lines(directory / 'decisions.jsonl.gz')), evidence[name], sha(run / 'condition.json'))
        previous = read(run / history['phases'][name]['result_file'])
        previous = {**previous, 'actual_physical_points': previous['logical_points']}
        for key, value in checked['scientific_result'].items():
            same(previous[key], value, 'historical_original_search.' + name + '.' + key)
        require(previous['oracle']['requests'] == evidence[name]['requests'], 'Historical query count changed')
        phases[name], checks[name] = p, checked
    chosen = check_selection(result, phases)
    cps = list(read_lines(cell / 'cost_checkpoints.jsonl'))
    require(result['checkpoints'] == cps, 'Top curve omits quality/cost boundaries')
    check_checkpoints(cps, checks, phases)
    for name in VIEWS:
        require(result['phase_summary'][name] == {k: phases[name][k] for k in ('quality_pct', 'fully_certified',
            'logical_points', 'actual_physical_points', 'search_decisions', 'phase_elapsed_seconds', 'stage_component_seconds')},
            'Top phase costs differ')
    require(set(result['phase_summary']) == set(VIEWS), 'Extra/missing phases')
    for key, expected in {'logical_points': service['logical_points'], 'actual_physical_points': len(points),
        'numerical_entries': commands['numerical_entries'], 'requests': sum(e['requests'] for e in evidence.values()),
        'search_decisions': sum(p['search_decisions'] for p in phases.values())}.items():
        require(result[key] == expected, 'Complete task costs differ: ' + key)
    check_times(result, phases, finished, math.fsum(p['elapsed_seconds'] for p in points.values()))
    return {**{k: spec[k] for k in ('cell', 'context_id', 'scenario_id', 'event_id', 'mode', 'arm', 'seed', 'pair_order')},
        'status': 'PASS', 'result_sha256': sha(cell / 'result.json'), 'cell_completion_sha256': sha(cell / 'completion_manifest.json'),
        'quality_pct': result['quality_pct'], 'guard_established': chosen['guard_established'],
        'guard_quality_pct': result['baseline_quality_pct'], 'gain_over_guard_kwh': chosen['gain_over_guard_kwh'],
        'logical_points': service['logical_points'], 'physical_points': len(points), 'numerical_entries': commands['numerical_entries'],
        'requests': result['requests'], 'search_decisions': result['search_decisions'], 'policy_decisions': 0,
        'worker_elapsed_seconds': result['worker_elapsed_seconds'], 'worker_finished_elapsed_seconds': finished['worker_elapsed_seconds'],
        'worker_pid': worker['pid'], 'worker_parent_pid': worker['parent_pid'], 'execution': execution,
        'physical_statuses': dict(statuses), 'history_verified': True, 'quality_checkpoints_verified': True,
        'cross_view_reuses': sum(e['cross_view_reuses'] for e in evidence.values()),
        'phase_summary': {name: {**result['phase_summary'][name], 'requests': evidence[name]['requests'],
            'semantic_signature_sha256': checks[name]['semantic_signature_sha256'],
            'phase_result_sha256': sha(cell / name / 'phase_result.json')} for name in VIEWS},
        'exact_checkpoints': cps}


def check_controller(controller, schedule, checks, dispatch):
    names = [s['cell'] for s in schedule]
    require(len(names) == len(set(names)) == len(checks) == 16 and {c['cell'] for c in checks} == set(names),
            'Incomplete or duplicate audited population')
    require(controller['status'] == 'COMPLETE_CELLS_PENDING_AUDIT' and controller['scheduled'] == 16
            and controller['dispatched'] == schedule and len(controller['completed']) == 16
            and set(controller['completed']) == set(names) and not controller['errors'] and not controller['unknown_cost_cells'],
            'Controller incomplete or uncertain paid cost')
    require([d['cell'] for d in dispatch] == names and len(controller['exits']) == 16, 'Dispatch order changed/retried')
    exits = {e['cell']: e for e in controller['exits']}
    require(set(exits) == set(names), 'Real process exit missing')
    by_cell = {c['cell']: c for c in checks}
    previous_end = 0.
    for start in range(0, 16, 2):
        pair_ends = []
        for spec, d in zip(schedule[start:start+2], dispatch[start:start+2]):
            e, c = exits[spec['cell']], by_cell[spec['cell']]
            require(e['exit_code'] == 0 and e['drained_after_controller_error'] is False
                    and e['launcher_pid'] == d['launcher_pid'] and e['pair_order'] == d['pair_order'] == spec['pair_order'],
                    'Failed, drained, or wrongly attributed cold process')
            begin = window.finite_time(e['started_epoch_seconds'], 'process start')
            end = window.finite_time(e['finished_epoch_seconds'], 'process exit')
            elapsed = window.finite_time(e['process_wall_seconds'], 'full cold process time')
            require(begin == d['started_epoch_seconds'] == c['execution']['started_epoch_seconds']
                    and previous_end <= begin <= end and elapsed >= c['worker_finished_elapsed_seconds'],
                    'Unpaid work, pair overlap, or restart of timing')
            require(c['worker_parent_pid'] in (d['launcher_pid'], c['execution']['controller_pid']),
                    'Actual worker ancestry unbound')
            require(abs((end - begin) - elapsed) <= 2., 'Wall and monotonic process time disagree materially')
            c['process_wall_seconds'] = elapsed
            pair_ends.append(end)
        previous_end = max(pair_ends)
    totals = {key: sum(c[key] for c in checks) for key in ('logical_points', 'physical_points', 'numerical_entries')}
    require(all(controller[k] == v for k, v in totals.items()) and totals['physical_points'] <= 8192
            and totals['logical_points'] <= 8192 and totals['numerical_entries'] == 48 + 4 * totals['physical_points'],
            'All-task budget or single-runtime initialization costs changed')
    wall = window.finite_time(controller['wall_seconds'], 'entire controller time')
    require(wall >= max(c['process_wall_seconds'] for c in checks), 'Controller timing omits running work')
    return totals


def paired_equalities(checks):
    require(len(checks) == 16 and {c['context_id'] for c in checks} == set(range(8)), 'Incomplete paired contexts')
    pairs = []
    for context in range(8):
        rows = [c for c in checks if c['context_id'] == context]
        require(len(rows) == 2 and {r['mode'] for r in rows} == set(MODES), 'Missing paired mode')
        by_mode = {r['mode']: r for r in rows}
        private, shared = by_mode['private'], by_mode['shared']
        require(all(private[k] == shared[k] for k in ('scenario_id', 'event_id', 'quality_pct', 'guard_quality_pct',
            'logical_points', 'requests', 'search_decisions')) and all(r['guard_established'] and r['history_verified'] for r in rows),
            'Sharing changes scientific result, logical budget, or original Beam guarantee')
        for name in VIEWS:
            require(private['phase_summary'][name]['semantic_signature_sha256'] == shared['phase_summary'][name]['semantic_signature_sha256'],
                    'Sharing changes original semantic search trajectory')
        require(shared['physical_points'] <= private['physical_points'] and private['cross_view_reuses'] == 0,
                'Shared mode pays extra actual points or private pool leaks')
        pairs.append({'context_id': context, 'status': 'PASS', 'private_cell': private['cell'], 'shared_cell': shared['cell'],
            'logical_trajectories_equal': True, 'historical_trajectories_equal': True,
            'quality_pct': shared['quality_pct'], 'actual_points_saved': private['physical_points'] - shared['physical_points']})
    return pairs


def source_hashes(run, scope):
    return {'freeze_sha256': sha(run / 'prephysical_freeze_manifest.json'), 'config_sha256': sha(run / 'config.json'),
        'schedule_sha256': sha(run / 'schedule.json'), 'source_binding_sha256': sha(run / 'source_binding.json'),
        'physical_scope_file_sha256': sha(run / 'physical_scope.json'), 'physical_scope_canonical_sha256': digest(scope),
        'scope_review_manifest_sha256': sha(run / 'scope_review/completion_manifest.json'),
        'exact_key_review_manifest_sha256': sha(run / 'exact_key_review/completion_manifest.json'),
        'historical_bindings_sha256': sha(run / 'historical_bindings.json'), 'source_early_completion_sha256': SOURCE_SHA}


def audit(run, output=None):
    from audit_d059_supplement import NoSolverImports, no_solver_loaded
    run = Path(run).resolve()
    output = Path(output).resolve() if output else run / 'audit'
    require(output.is_relative_to(ROOT) and output != run and not output.exists(), 'Preserve earlier audit evidence')
    output.mkdir(parents=True)
    started, checks = time.perf_counter(), []
    result = {'status': 'FAIL', 'source_run': run.name, 'errors': [], 'audited_cells': 0, 'results': [],
              'new_audit_physical_points': 0, 'new_training_tasks': 0}
    guard = NoSolverImports()
    sys.meta_path.insert(0, guard)
    try:
        no_solver_loaded()
        cfg, scope, schedule = check_parent(run)
        result.update(source_hashes(run, scope))
        for spec in schedule:
            checked = check_cell(run, spec, cfg, scope)
            checks.append(checked)
            save(output / (spec['cell'].replace('/', '_') + '.json'), checked)
            save(output / 'progress.json', {'status': 'RUNNING', 'audited_cells': len(checks), 'scheduled': 16,
                'last_cell': spec['cell'], 'elapsed_seconds': time.perf_counter() - started,
                'new_physical_points': 0, 'new_training': 0})
        controller = read(run / 'controller_completion.json')
        require(controller['exits'] == read(run / 'controller_exits.json'), 'Process exit journal differs')
        totals = check_controller(controller, schedule, checks, list(read_lines(run / 'controller_dispatch.jsonl')))
        pairs = paired_equalities(checks)
        no_solver_loaded()
        statuses = Counter()
        for row in checks:
            statuses.update(row['physical_statuses'])
            save(output / (row['cell'].replace('/', '_') + '.json'), row)
        result.update(status='PASS', audited_cells=len(checks), results=checks, paired_equalities=pairs,
            **totals, physical_runtimes=16, logical_searches=32, complete_protected_cells=16, physical_statuses=dict(statuses),
            controller_completion_sha256=sha(run / 'controller_completion.json'),
            controller_dispatch_sha256=sha(run / 'controller_dispatch.jsonl'),
            interpretation='Eight observed training contexts, rule-only paired sharing mechanism test. One paid runtime per cold task; two independent logical searches. No learning, upper-bound stopping, or cross-task numerical answers.')
    except BaseException as exc:
        result.update(audited_cells=len(checks), results=checks)
        result['errors'].append({'error': repr(exc), 'traceback': traceback.format_exc()})
    finally:
        sys.meta_path.remove(guard)
    result['elapsed_seconds'] = time.perf_counter() - started
    save(output / 'summary.json', result)
    save(output / 'completion_manifest.json', {p.relative_to(output).as_posix(): sha(p)
        for p in output.rglob('*') if p.is_file() and p.name != 'completion_manifest.json'})
    return result


def preflight(run, output):
    from audit_d059_supplement import NoSolverImports, no_solver_loaded
    run, output = Path(run).resolve(), Path(output).resolve()
    require(output.is_relative_to(ROOT) and not output.exists(), 'Preserve previous preflight evidence')
    output.mkdir(parents=True)
    result = {'status': 'FAIL', 'source_run': run.name, 'errors': [], 'new_physical_points': 0, 'new_training': 0}
    guard = NoSolverImports()
    sys.meta_path.insert(0, guard)
    try:
        no_solver_loaded()
        _, scope, schedule = check_parent(run)
        no_solver_loaded()
        result.update(status='PASS', contexts=8, cells=len(schedule), logical_searches=32, physical_runtimes=16,
                      **source_hashes(run, scope))
    except BaseException as exc:
        result['errors'].append({'error': repr(exc), 'traceback': traceback.format_exc()})
    finally:
        sys.meta_path.remove(guard)
    save(output / 'summary.json', result)
    save(output / 'completion_manifest.json', {'summary.json': sha(output / 'summary.json')})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        require(args.output is not None, 'Preflight requires an independent output directory')
        result = preflight(args.run, args.output)
    else:
        result = audit(args.run, args.output)
    print(prior.canonical({k: v for k, v in result.items() if k not in ('results', 'paired_equalities')}), flush=True)
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
