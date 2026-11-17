"""Budget-parameterized copies of the saved-evidence auditor; no solver calls.
Only the logical budget and ordered view population become explicit arguments.
Original source SHA256: cc1a488d82d96147fbe53a390434a676599f1602628d084e3688ea92d028d9e6
"""
from audit_clpu_shared_pilot import *


def check_service_ledger(events, views, points, mech, scope, condition_sha, mode, *, logical_budget=256, views_order=('GUARD','EARLY')):
    """Rebuild every service event and pool from initially empty dictionaries."""
    require(mode in MODES and list(views) == list(views_order), 'Service mode or view population changed')
    pools, source_records, checked, event_index, actual_count = {}, {}, {}, 0, 0

    def event(kind, **fields):
        nonlocal event_index
        require(event_index < len(events), 'Missing service event: ' + kind)
        expected = {'sequence': event_index + 1, 'kind': kind, **fields}
        require(events[event_index] == expected, 'Service chronology/payload differs: ' + kind)
        event_index += 1
        return event_index

    event('SERVICE_STARTED', mode=mode, scope_sha256=digest(scope), physical_budget=512, logical_budget=logical_budget)
    event('SERVICE_READY', physical_points=0)
    for name in views_order:
        pool = 'SHARED' if mode == 'shared' else name
        evidence_pool = pools.setdefault(pool, {})
        logical_rows, request_rows = views[name]['logical_rows'], views[name]['requests']
        logical, semantic, transformed = {}, {}, []
        hits = aliases = crosses = denied = new_actual = 0
        alias_seconds = 0.
        event('VIEW_STARTED', view=name, pool=pool, physical_points_before=actual_count, logical_budget=logical_budget)
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
            elif len(logical) == logical_budget:
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


def check_service_summaries(summary, view_summaries, checked, scope, mode, numerical, *, logical_budget=256, views_order=('GUARD','EARLY')):
    expected_views = {}
    for name in views_order:
        c = checked[name]
        expected = {'version': SERVICE_VERSION, 'view': name, 'pool': 'SHARED' if mode == 'shared' else name,
            'mode': mode, 'scope_sha256': digest(scope), 'status': 'CLOSED_LOGICAL_VIEW_PENDING_AUDIT', 'closed': True,
            'points': c['logical_points'], 'logical_points': c['logical_points'], 'point_budget': logical_budget,
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
        'logical_budget_per_view': logical_budget, 'logical_points': sum(c['logical_points'] for c in checked.values()),
        'actual_numerical_entries': numerical, 'initialization_numerical_entries': 3,
        'expected_numerical_entries': numerical, 'physical_cost_complete': True, 'runtime_initializations': 1,
        'runtime_initialization_attempts': 1,
        'cross_task_evidence_reuse': False, 'views': expected_views}
    same(summary, expected, 'independent_single_runtime_summary')


def check_phase_records(mech, cfg, name, spec, result, attempts, decisions, evidence, condition_sha, *, logical_budget=256, views_order=('GUARD','EARLY')):
    require(name in views_order and spec['method'] == 'beam', 'Unexpected logical search or learned selector')
    oracle = branch.PaidRequests(evidence['request_rows'], evidence['logical'], mech, logical_budget, condition_sha)
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
