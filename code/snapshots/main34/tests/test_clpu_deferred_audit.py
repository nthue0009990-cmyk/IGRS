"""Independent deferred graph/query replay; synthetic certificates, zero PF."""
import ast
from copy import deepcopy
from pathlib import Path
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_clpu_deferred_pilot as audit
from dnr.nature34_clpu_branch import BranchConfig
from dnr.nature34_clpu_deferred import DeferredCertificationSearch
from dnr.nature34_clpu_window import WindowBranchSearch
from test_nature34_clpu_branch_audit import LoggedFakeOracle, inputs
from test_clpu_shared_audit import make_service, material
from test_nature34_clpu_equivalent_cache import lines, save


class LogicalActualOracle(LoggedFakeOracle):
    @property
    def physical_points(self):
        return self.points


def production_view(engine):
    representatives = {}
    for node in engine._frontier.values():
        plan = node['info']['actions']
        if plan not in engine.verification_queue:
            continue
        priority = lambda n: (len(n['prefix']), n['origin'] != 'prefix_wait', n['id'])
        if plan not in representatives or priority(node) < priority(representatives[plan]):
            representatives[plan] = node
    view = object.__new__(WindowBranchSearch)
    view.__dict__ = dict(engine.__dict__)
    view._frontier = {n['id']: n for n in representatives.values()}
    view._beam_turn = engine.verify_attempts
    return view


def saved_route(inputs, schedule='deferred', failed=False, budget=45):
    mech, env, cfg = deepcopy(inputs)
    cfg['point_budget'] = budget
    oracle = LogicalActualOracle(env, budget, failed)
    engine = DeferredCertificationSearch(env, oracle, BranchConfig(**cfg))
    chosen, records = set(), []
    operations = ['EXPAND_GRAPH'] * 8 + ['VERIFY_COMPLETE_PLAN'] * 8 if schedule == 'deferred' else ['EXPAND_GRAPH', 'VERIFY_COMPLETE_PLAN'] * 8
    for operation in operations:
        if engine.done: break
        if operation == 'EXPAND_GRAPH':
            eligible = [n for n in engine._frontier.values() if n['id'] not in engine.expanded_nodes
                        and n['info']['actions'] not in chosen]
            if not eligible: continue
            node = min(eligible, key=lambda n: (len(n['prefix']), n['info']['priority'], n['id']))
            chosen.add(node['info']['actions'])
            event = engine.expand(node['id'])
            observation = None
        else:
            view = production_view(engine)
            if not view._window(): continue
            observation = view.observation('raw')
            index = view.rule_action('beam')
            node = view._window()[index]
            event = engine.verify(node['id'])
        records.append({'event': event, 'pending': engine.pending(), 'observation': observation})
    if not engine.done:
        records.append({'event': engine.yield_to_beam(), 'pending': engine.pending(), 'observation': None})
    return {'mech': mech, 'cfg': cfg, 'raw': oracle.raw, 'requests': oracle.rows,
            'logical_rows': [{'reuse_kind': 'ACTUAL_PHYSICAL'} for _ in oracle.raw],
            'result': engine.result(), 'records': records}


def independently_replay(data):
    oracle = audit.branch.PaidRequests(data['requests'], data['raw'], data['mech'], data['cfg']['point_budget'], 'synthetic')
    replay = audit.DeferredReplay(data['mech'], data['cfg'], oracle, data['result']['attempts'], data['logical_rows'])
    chosen = set()
    for row in data['records']:
        event = deepcopy(row['event'])
        event.pop('elapsed_seconds')
        if event['operation'] == 'EXPAND_GRAPH':
            node = replay.fixed_expansion(chosen)
            assert node['id'] == event['branch_id']
            chosen.add(node['info']['actions'])
        elif event['operation'] == 'VERIFY_COMPLETE_PLAN':
            view, aliases = replay.verification_view()
            audit.same(view.observation('raw'), row['observation'], 'independent_pending_observation')
            assert view.window()[view.rule('beam')]['id'] == event['branch_id']
            assert len({n['info']['actions'] for n in view.frontier.values()}) == len(view.frontier)
            assert all(aliases[p] for p in aliases)
        audit.same(event, replay.operate(event['operation'], event['branch_id']), 'independent_macro_event')
        audit.same(row['pending'], replay.pending(), 'independent_pending_inventory')
    return audit.check_deferred_result(replay, data['result'])


@pytest.mark.parametrize('schedule', ['deferred', 'interleaved'])
@pytest.mark.parametrize('failed,budget', [(False, 45), (True, 45), (False, 9), (False, 2)])
def test_independent_graph_query_skip_handoff_and_partial_replay(inputs, schedule, failed, budget):
    data = saved_route(inputs, schedule, failed, budget)
    result = independently_replay(data)
    assert result['logical_points'] <= budget
    assert result['guard_established'] is False
    assert result['total_decisions'] == result['graph_expansions'] + result['verify_attempts'] + int(result['handed_off'])


@pytest.mark.parametrize('field', ['logical_points_after', 'actual_physical_points_after', 'quality_after',
                                  'new_branches', 'verify_attempts', 'pending_complete_plans'])
def test_deferred_event_mutation_cannot_invent_quality_or_hide_work(inputs, field):
    data = saved_route(inputs)
    data['records'][0]['event'][field] += 1
    with pytest.raises(AssertionError): independently_replay(data)


def test_pending_aliases_cannot_fake_certification_or_remove_repair_parents(inputs):
    data = saved_route(inputs)
    data['records'][0]['pending'][0]['verified'] = not data['records'][0]['pending'][0]['verified']
    with pytest.raises(AssertionError): independently_replay(data)


def test_auditor_has_no_production_engine_adapter_or_policy_import():
    tree = ast.parse(Path(audit.__file__).read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        names = [n.name for n in node.names] if isinstance(node, ast.Import) else [node.module or ''] if isinstance(node, ast.ImportFrom) else []
        assert not any(n.startswith('dnr.') or n.startswith('run_clpu_deferred') for n in names)


def saved_controller(inputs, schedule='deferred', failed=False, budget=45):
    from dnr.nature34_clpu_deferred_control import DeferredMacroController
    mech, env, cfg = deepcopy(inputs)
    cfg.update(point_budget=budget, action_window=32)
    oracle = LogicalActualOracle(env, budget, failed)
    engine = DeferredCertificationSearch(env, oracle, BranchConfig(**cfg))
    controller = DeferredMacroController(engine, schedule=schedule)
    records = []
    while True:
        proposal = controller.prepare()
        if proposal is None: break
        records.append({'proposal': deepcopy(proposal), 'response': controller.apply(proposal)})
    return dict(mech=mech, cfg=cfg, raw=oracle.raw, requests=oracle.rows,
                logical_rows=[{'reuse_kind': 'ACTUAL_PHYSICAL'} for _ in oracle.raw],
                result=controller.result(), records=records, schedule=schedule)


def replay_controller(data):
    oracle = audit.branch.PaidRequests(data['requests'], data['raw'], data['mech'], data['cfg']['point_budget'], 'synthetic')
    replay = audit.DeferredReplay(data['mech'], data['cfg'], oracle, data['result']['attempts'], data['logical_rows'])
    macro = audit.MacroReplay(replay, data['schedule'], 'beam')
    for row in data['records']:
        expected = macro.check_proposal(row['proposal'])
        response = deepcopy(row['response']); response.pop('elapsed_seconds')
        audit.same(response, macro.apply(expected), 'independent_real_controller_response')
    return macro.check_control_result(data['result'])


@pytest.mark.parametrize('schedule,failed,budget', [('deferred', False, 45), ('interleaved', False, 45),
    ('deferred', True, 45), ('interleaved', True, 45), ('deferred', False, 2), ('interleaved', False, 9)])
def test_actual_new_controller_full_fixed_schedule_replayed_independently(inputs, schedule, failed, budget):
    result = replay_controller(saved_controller(inputs, schedule, failed, budget))
    assert result['graph_expansions'] <= 16 and result['verify_attempts'] <= 16 and result['total_decisions'] <= 33


@pytest.mark.parametrize('field,value', [('slot_index', 31), ('method', 'ppo_raw'), ('branch_id', -1),
    ('policy_used', True), ('operation', 'YIELD_TO_BEAM')])
def test_macro_schedule_or_action_changes_are_rejected(inputs, field, value):
    data = saved_controller(inputs)
    data['records'][0]['proposal'][field] = value
    with pytest.raises(AssertionError): replay_controller(data)


class IndependentFirstRanker:
    def evaluate(self, compact):
        return next(i for i, v in enumerate(compact['mask']) if v), 0., 0.


def real_routes(make_service, arm, failed=False):
    import run_clpu_deferred_pilot as runner
    from test_run_clpu_deferred_pilot import contexts, models, FakePolicy
    service, mech, cfg = make_service('shared', failed=failed)
    cfg['action_window'] = 32
    spec = {**next(s for s in runner.schedule_for(contexts(), models()) if s['arm'] == arm), 'cell': 'task'}
    for phase in audit.shared.VIEWS:
        save(service.cell / phase / 'spec.json', runner.phase_spec(spec, phase))
    policy = FakePolicy(service.env) if spec['checkpoint'] else None
    with service:
        phases, cps = runner.run_routes(service, service.env, BranchConfig(**cfg), service.cell, spec,
                                       time.perf_counter(), time.time() + 60, policy)
    data = material(service, mech)
    # FakeDelegate has no measured native time. Its synthetic placeholder must
    # not be treated as measured PF latency in the independent timer check.
    for point in data['points'].values(): point['elapsed_seconds'] = 0.
    evidence, _ = audit.shared.check_service_ledger(**data)
    return service, mech, cfg, spec, phases, cps, evidence


def check_routes(saved):
    service, mech, cfg, spec, phases, cps, evidence = saved
    checks = {}
    for name in audit.shared.VIEWS:
        p = phases[name]
        args = (mech, cfg, p['spec'], p, lines(service.cell / name / 'plans.jsonl.gz'),
                lines(service.cell / name / 'decisions.jsonl.gz'), evidence[name], audit.sha(service.parent / 'condition.json'))
        if name == 'EARLY':
            checks[name] = audit.check_deferred_records(*args, IndependentFirstRanker() if spec['checkpoint'] else None)
        else:
            checks[name] = audit.shared.check_phase_records(mech, cfg, name, *args[2:])
    audit.shared.check_checkpoints(cps, checks, phases)
    return checks


@pytest.mark.parametrize('arm', ['interleaved_window', 'deferred_window', 'deferred_bc', 'deferred_ppo'])
def test_actual_runner_compact_journal_both_shared_views_and_all_boundaries(make_service, arm):
    saved = real_routes(make_service, arm)
    checks = check_routes(saved)
    assert checks['EARLY']['policy_decisions'] == (checks['EARLY']['verify_attempts'] if arm in ('deferred_bc','deferred_ppo') else 0)
    assert checks['EARLY']['actual_and_independent_policy_actions_equal']
    assert checks['GUARD']['fully_certified'] and saved[5][-1]['guard_established']


def test_actual_runner_initial_failure_keeps_true_controller_stop_without_prepare(make_service):
    saved = real_routes(make_service, 'deferred_window', failed=True)
    service, mech, cfg, spec, phases, cps, evidence = saved
    p = phases['EARLY']
    assert p['search_decisions'] == 0 and p['stop_reason'] == 'INITIAL_FAIL'
    checked = audit.check_deferred_records(mech, cfg, p['spec'], p, [], [], evidence['EARLY'],
                                           audit.sha(service.parent / 'condition.json'))
    assert checked['decisions'] == 0 and checked['quality_pct'] is None
    assert not cps[-1]['guard_established']


@pytest.mark.parametrize('mutation', ['future_observation', 'hash', 'masked', 'checkpoint', 'response_time', 'skip_slot'])
def test_recorded_compact_hash_mask_causality_and_policy_binding_rejected(make_service, mutation):
    saved = real_routes(make_service, 'deferred_ppo')
    service, mech, cfg, spec, phases, cps, evidence = saved
    decisions = lines(service.cell / 'EARLY/decisions.jsonl.gz')
    i = next(i for i in range(0, len(decisions), 2) if decisions[i]['proposal']['operation'] == 'VERIFY_COMPLETE_PLAN')
    row = decisions[i]
    if mutation == 'future_observation': row['obs_compact']['global_features'][0] += .5; row['obs_compact_sha256'] = audit.digest(row['obs_compact'])
    elif mutation == 'hash': row['obs_compact_sha256'] = '0' * 64
    elif mutation == 'masked': row['obs_compact']['mask'][0] = False; row['obs_compact_sha256'] = audit.digest(row['obs_compact'])
    elif mutation == 'checkpoint': row['checkpoint_sha256'] = 'f' * 64
    elif mutation == 'response_time': decisions[i+1]['response']['elapsed_seconds'] = -1.
    else: row['proposal']['slot_index'] += 1
    # Preserve STARTED/COMPLETED identity: these mutations must be rejected by
    # the independent semantics rather than merely by a mismatched duplicate.
    for key in row:
        if key not in ('status', 'obs_compact'): decisions[i+1][key] = deepcopy(row[key])
    p = phases['EARLY']
    with pytest.raises(AssertionError):
        audit.check_deferred_records(mech, cfg, p['spec'], p, lines(service.cell/'EARLY/plans.jsonl.gz'),
            decisions, evidence['EARLY'], audit.sha(service.parent/'condition.json'), IndependentFirstRanker())


def test_fixed_schedule_settings_and_phase_contract_match_real_runner():
    import run_clpu_deferred_pilot as runner
    from test_run_clpu_deferred_pilot import contexts, models
    assert audit.protocol_settings() == runner.settings()
    schedule = audit.expected_schedule(contexts(), models())
    assert schedule == runner.schedule_for(contexts(), models())
    for spec in schedule:
        for phase in audit.shared.VIEWS:
            assert audit.expected_phase_spec(spec, phase) == runner.phase_spec(spec, phase)


def full_population():
    from test_run_clpu_deferred_pilot import contexts, models
    schedule = audit.expected_schedule(contexts(), models())
    rows, dispatch, exits = [], [], []
    for spec in schedule:
        start = 1000. + spec['pair_order'] * 20
        rows.append({**spec, 'logical_points': 10, 'physical_points': 2, 'numerical_entries': 11,
            'worker_finished_elapsed_seconds': 8., 'worker_parent_pid': 99,
            'execution': {'started_epoch_seconds': start, 'controller_pid': 99},
            'guard_established': True, 'history_verified': True, 'guard_quality_pct': 90.,
            'first_verify_state_sha256': 'common' if spec['macro_schedule'] == 'deferred' else 'interleaved',
            'phase_summary': {'GUARD': {'semantic_signature_sha256': str(spec['context_id'])}}})
        d = {'cell': spec['cell'], 'pair_order': spec['pair_order'], 'launcher_pid': 500 + spec['episode_id'],
             'started_epoch_seconds': start}
        dispatch.append(d)
        exits.append({**d, 'exit_code': 0, 'drained_after_controller_error': False,
                      'finished_epoch_seconds': start + 10., 'process_wall_seconds': 10.})
    c = {'status': 'COMPLETE_CELLS_PENDING_AUDIT', 'scheduled': 64, 'dispatched': schedule,
         'completed': [s['cell'] for s in schedule], 'errors': [], 'unknown_cost_cells': [], 'exits': exits,
         'logical_points': 640, 'physical_points': 128, 'numerical_entries': 704, 'wall_seconds': 640.}
    return c, schedule, rows, dispatch


def test_all64_process_costs_all8_guards_and_common_deferred_exposure_close():
    data = full_population()
    assert audit.check_controller(*data)['numerical_entries'] == 64 * 3 + 4 * 128
    assert len(audit.context_equalities(data[2])) == 8
    assert all(r['process_wall_seconds'] == 10. for r in data[2])


@pytest.mark.parametrize('mutation', ['missing', 'unpaid_init', 'exit_failure', 'retry', 'different_guard', 'different_exposure', 'missing_guard'])
def test_population_process_and_fair_exposure_mutations_rejected(mutation):
    data = full_population()
    c, schedule, rows, dispatch = data
    if mutation == 'missing': rows.pop()
    elif mutation == 'unpaid_init': c['numerical_entries'] -= 3
    elif mutation == 'exit_failure': c['exits'][0]['exit_code'] = 19
    elif mutation == 'retry': dispatch[1] = deepcopy(dispatch[0])
    elif mutation == 'different_guard': rows[0]['phase_summary']['GUARD']['semantic_signature_sha256'] = 'changed'
    elif mutation == 'different_exposure': next(r for r in rows if r['macro_schedule'] == 'deferred')['first_verify_state_sha256'] = 'future'
    else: rows[0]['guard_established'] = False
    with pytest.raises(AssertionError):
        audit.check_controller(*data)
        audit.context_equalities(rows)


def test_actual_six_model_sources_preflight_and_mutated_weight_binding(tmp_path):
    import run_clpu_deferred_pilot as runner
    from audit_d059_supplement import NoSolverImports, no_solver_loaded
    guard = NoSolverImports(); sys.meta_path.insert(0, guard)
    try:
        no_solver_loaded()
        # Exercise the real binder, including the nested manifest that is
        # indirectly bound by model_preflight_binding rather than the root
        # seal's flat file list. A manual copytree would conceal this omission.
        models = runner.bind_models(tmp_path)
        assert (tmp_path / 'model_preflight/completion_manifest.json').is_file()
        assert audit.check_models(tmp_path) == models and len(models) == 6
        no_solver_loaded()
        (tmp_path / models[0]['checkpoint']).write_bytes(b'changed checkpoint')
        with pytest.raises(AssertionError): audit.check_models(tmp_path)
    finally:
        sys.meta_path.remove(guard)
