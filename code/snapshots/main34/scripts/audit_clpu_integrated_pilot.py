"""Independent solver-free replay of cached, Beam-protected fixed-policy search.

The new production integration, cache, search and policy loaders are never
imported. Original independent physics, causal ledgers, graph reconstruction
and Torch actor equations are reused without changing their tolerances.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import math
from pathlib import Path
import sys
import time
import traceback
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
sys.dont_write_bytecode = True

import audit_clpu_equivalent_cache as cache
import audit_nature34_clpu_tail as tail
import audit_nature34_clpu_branch_learning as learning

window, branch, old = cache.window, cache.branch, cache.old
require, close, same = cache.require, cache.close, cache.same
read, save, sha, canonical, read_lines = cache.read, cache.save, cache.sha, cache.canonical, cache.read_lines
digest, manifest = cache.digest, cache.manifest
SEEDS = (17, 41, 73)
LEARNING_RUN = '20261024T073617_269327Z_nature34_clpu_branch'
LEARNING_SHA = '4c3777f938d21f074b40db56c82cd62ca1b956f6aa7d79215832bf192745c56f'
SUPPLEMENT_RUN = '20261025T070549_299770Z_d059_supplement'
SUPPLEMENT_SHA = '879850ffa7b22aa86ea379de535e9bb8caee1db961fe1635c8fe50e56cf19299'
CACHE_RUN = '20261030T123732_313636Z_clpu_equivalent_cache_pilot'
CACHE_SHA = 'f3f1bbfa9a4eeecbdec22b55aad4c80550db00f360b3326e6a3acc764694e040'
TAIL_RUN = '20261027T045048_590910Z_clpu_tail_pilot'
TAIL_SHA = '3661145543e8d84c8771a6a3d13a24bd7336d1a1ba3eb2699a3efa545d47bb8e'
D060_RUN = '20261025T123350_150996Z_d060_main'
D060_SHA = 'b2f85e5f21d4e1f316c018a7d53428149f750ae32443fda7c3e572c2333ce7a6'
STUDY = 'CLPU_INTEGRATED_PROTECTED_PILOT_V1'
ARM_SEEDS = (('window_reference', 0), ('protected_rule', 0),
             *(('protected_bc', seed) for seed in SEEDS),
             *(('protected_ppo', seed) for seed in SEEDS))


class CoverageTailCore(tail.TailCore, window.WindowReplay):
    """Exact failed-prefix filter, then independent coverage window, then Beam.

TailCore.window supplies the VERIFY singleton and retains failed repair
parents when no unfailed frontier exists. Its super call reaches the old
independent coverage algorithm through this MRO; the baseline never uses it.
"""


class IntegratedReplay(tail.TailReplay):
    def finish_baseline(self):
        super().finish_baseline()
        if self.phase in ('CONTINUATION', 'VERIFY'):
            require(type(self.core) is tail.TailCore, 'Baseline core was changed before original Beam finished')
            self.core.__class__ = CoverageTailCore


def model_sources():
    """Bind all six fixed actors to original data/fit/PPO audit evidence."""
    source, supplement = ROOT / 'runs' / LEARNING_RUN, ROOT / 'runs' / SUPPLEMENT_RUN
    require(sha(source / 'completion_manifest.json') == LEARNING_SHA, 'Original learning run seal changed')
    require(sha(supplement / 'completion_manifest.json') == SUPPLEMENT_SHA, 'Original learning supplement changed')
    sm = {name.replace('\\', '/'): value for name, value in read(source / 'completion_manifest.json').items()}
    am = {name.replace('\\', '/'): value for name, value in read(supplement / 'completion_manifest.json').items()}

    def bound(base, sealed, name):
        require(sealed.get(name) == sha(base / name), 'Unbound original learning evidence: ' + name)
        return read(base / name)

    validation = bound(supplement, am, 'learning_evidence_validation.json')
    require(validation['status'] == 'VERIFIED_EXISTING_LEARNING_AUDIT_BINDINGS'
            and len(validation['records']) == 49, 'Missing original fit or PPO-update audits')
    paths = set()
    for row in validation['records']:
        name = row['verification'].replace('\\', '/')
        require(name not in paths and sm[name] == row['sha256'], 'Repeated or changed learning audit binding')
        paths.add(name)
        record = bound(source, sm, name)['output']
        require(record['errors'] == [] and record['status'].startswith('VERIFIED_CLPU_BRANCH_'),
                'Original learning audit failed')
    attribution = bound(supplement, am, 'final_attribution.json')
    require(attribution['status'] == 'VERIFIED_CLPU_BRANCH_LEARNING_ATTRIBUTION'
            and attribution['errors'] == [] and attribution['same_seed_bc_ppo_start'] is True,
            'BC/PPO do not have independently attributed common start')
    endpoints = bound(source, sm, 'final_checkpoints.json')
    bindings = []
    for seed in SEEDS:
        for method, suffix in (('bc_raw', 'start.pt'), ('ppo_raw', 'ppo_round07/after.pt')):
            name = f'learning/seed{seed}/raw/{suffix}'
            require(endpoints[str(seed)][method].replace('\\', '/') == name and sha(source / name) == sm[name],
                    'Fixed seed/model endpoint changed')
            bindings.append({'seed': seed, 'method': method, 'source_relative': name,
                'source_sha256': sm[name], 'checkpoint': f'models/seed{seed}/{method}.pt',
                'learning_audit_binding': 'all 49 sealed source learning audits and final attribution'})
    return bindings, validation, attribution


def check_policy(run, spec, transitions):
    """Only continuation actions are learned; baseline and VERIFY remain rules."""
    if spec['method'] == 'beam':
        require(not spec.get('checkpoint') and not spec.get('checkpoint_sha256'), 'Rule loaded a policy checkpoint')
        return {'policy_decisions': 0, 'checkpoint_sha256': None}
    require(spec['method'] in ('bc_raw', 'ppo_raw') and spec['seed'] in SEEDS
            and spec['representation'] == 'raw' and spec['deterministic'] is True, 'Unapproved learned policy')
    checkpoint = (run / spec['checkpoint']).resolve()
    require(checkpoint.is_relative_to(run) and checkpoint != run and sha(checkpoint) == spec['checkpoint_sha256'],
            'Policy checkpoint escaped run or changed')
    model, saved = learning.load_independent(checkpoint)
    require(saved['seed'] == spec['seed'] and saved['kind'] == 'actor_critic' and saved['representation'] == 'raw',
            'Checkpoint seed/kind/representation mismatch')
    selected = []
    for row in transitions:
        if row['stage'] == 'CONTINUATION':
            require(row['method'] == spec['method'], 'Learned continuation misattributed')
            selected.append({**row, 'transition_id': len(selected)})
        else:
            require(row['stage'] in ('BASELINE', 'VERIFY') and row['checkpoint_sha256'] is None,
                    'Baseline or verifier falsely attributed to a learner')
    return learning.check_policy_transitions(checkpoint, selected, spec)


def source_model_preflight(output):
    """Read-only endpoint and common-start proof, with no simulator imports."""
    from audit_d059_supplement import NoSolverImports, no_solver_loaded
    output = Path(output).resolve()
    require(output.is_relative_to(ROOT) and not output.exists(), 'Preserve previous model preflight')
    output.mkdir(parents=True)
    guard = NoSolverImports()
    sys.meta_path.insert(0, guard)
    result = {'status': 'FAIL', 'errors': [], 'new_physical_points': 0, 'new_training': 0, 'models': []}
    try:
        import torch
        torch.set_num_threads(1)
        bindings, validation, attribution = model_sources()
        source = ROOT / 'runs' / LEARNING_RUN
        sealed = {name.replace('\\', '/'): value for name, value in read(source / 'completion_manifest.json').items()}
        for binding in bindings:
            model, saved = learning.load_independent(source / binding['source_relative'])
            require(saved['seed'] == binding['seed'] and saved['kind'] == 'actor_critic'
                    and saved['architecture'] == {'candidate_dim': 261, 'global_dim': 79,
                        'representation': 'raw', 'hidden': 128}, 'Frozen model architecture changed')
            result['models'].append({**binding, 'architecture': saved['architecture']})
        starts = []
        for seed in SEEDS:
            first = f'learning/seed{seed}/raw/start.pt'
            before = f'learning/seed{seed}/raw/ppo_round00/before.pt'
            require(sha(source / before) == sealed[before], 'Original PPO common-start tensor file changed')
            left, _ = learning.load_independent(source / first)
            right, _ = learning.load_independent(source / before)
            a, b = left.state_dict(), right.state_dict()
            require(a.keys() == b.keys() and all(torch.equal(a[key], b[key]) for key in a),
                    'Same-seed BC/PPO initial tensors differ')
            starts.append({'seed': seed, 'bc_start_sha256': sealed[first],
                           'ppo_round00_before_sha256': sealed[before], 'all_tensors_exactly_equal': True})
        no_solver_loaded()
        result.update(status='PASS', same_seed_starts=starts, learning_audit_records=len(validation['records']),
            source_completion_sha256=LEARNING_SHA, supplement_completion_sha256=SUPPLEMENT_SHA,
            source_attribution_status=attribution['status'],
            interpretation='Common BC teacher/start; PPO has additional on-policy interaction training. Not equal total training data/cost.')
    except BaseException as exc:
        result['errors'].append({'error': repr(exc), 'traceback': traceback.format_exc()})
    finally:
        sys.meta_path.remove(guard)
    save(output / 'summary.json', result)
    save(output / 'completion_manifest.json', {'summary.json': sha(output / 'summary.json')})
    return result


def check_protected_records(mech, cfg, result, baseline, checkpoint, transitions, decisions,
                            request_rows, points, condition_sha, identity, stage_events,
                            cache_hits_reported=True):
    """Independent original tail audit with only the declared coverage adoption."""
    result = deepcopy(result)
    require(result['logical_points'] == len(points), 'Top logical count differs from causal logical ledger')
    result['actual_physical_points'] = result['logical_points']
    oracle = branch.PaidRequests(request_rows, points, mech, cfg['branch_search']['point_budget'], condition_sha)
    info = result['tail']
    replay = IntegratedReplay(mech, cfg, oracle, result['attempts'], identity, True,
                              info['events'], cache_hits_reported)
    spec, decisions = result['spec'], iter(decisions)
    stage_index, stages, previous, count = 0, Counter(), None, 0
    checkpoint_checked = False
    boundaries, quality = [], []

    def quality_boundary(index):
        quality.append({'search_choices': index, 'logical_points': oracle.used, 'requests': oracle.counts()['requests'],
            'quality_pct': 100 * replay.core.quality if replay.core.best is not None else None,
            'fully_certified': replay.core.best is not None, 'phase': replay.phase})

    def stage_entry(index):
        nonlocal previous, stage_index, checkpoint_checked
        if replay.phase == previous:
            return
        if previous == 'BASELINE' or (previous is None and replay.phase != 'BASELINE'):
            require(not checkpoint_checked and replay.baseline is not None, 'Missing/repeated complete baseline seal')
            same(baseline, replay.baseline['result'], 'complete_original_Beam_baseline')
            same(checkpoint['engine_checkpoint'], {**replay.baseline, 'sha256': replay.baseline_sha},
                 'baseline_replay_checkpoint')
            require(checkpoint['transition_boundary'] == index
                    and checkpoint['attempt_boundary'] == len(baseline['attempts'])
                    and checkpoint['oracle_boundary'] == oracle.counts(), 'Paid baseline boundary changed')
            window.finite_time(checkpoint['elapsed_seconds'], 'baseline duration')
            checkpoint_checked = True
        require(stage_index < len(stage_events), 'Missing stage entry journal')
        event = stage_events[stage_index]
        require(set(event) == {'stage', 'transition_boundary', 'oracle', 'utc'}
                and event['stage'] == replay.phase and event['transition_boundary'] == index
                and event['oracle'] == oracle.counts(), 'Stage or cumulative logical boundary differs')
        datetime.fromisoformat(event['utc'])
        boundaries.append({'stage': replay.phase, 'transition_boundary': index, **oracle.counts()})
        stage_index += 1
        previous = replay.phase

    quality_boundary(0)
    for index, row in enumerate(transitions):
        stage_entry(index)
        require(not replay.done, 'Decision after protected search termination')
        phase = replay.phase
        tail.check_transition(replay, row, decisions, spec, index, stages[phase])
        stages[phase] += 1
        count += 1
        quality_boundary(count)
    stage_entry(count)
    require(replay.done and checkpoint_checked and stage_index == len(stage_events), 'Unfinished/omitted stage')
    require(next(decisions, None) is None and replay.event_index == len(replay.events), 'Unclosed decision or tail event')
    require(oracle.index == len(oracle.rows) and oracle.used == len(points), 'Unbound logical requests')
    require(replay.core.attempt_index == len(result['attempts']), 'Unqueried/omitted plan')
    for key, value in tail.baseline_snapshot(replay.core).items():
        same(result[key], value, 'final_core.' + key)
    require(result['search_decisions'] == count and result['stage_decisions'] == dict(stages),
            'Stage attribution count differs')
    require(replay.continued <= cfg['tail_search']['continuation_expansions']
            and replay.core.turn == stages['BASELINE'] + stages['CONTINUATION'] <= 512
            and replay.verified == stages['VERIFY'], 'Cumulative expansion/verification allowance reset')
    replay.check_counters(info['counters'])
    close(info['counters']['step_seconds'], math.fsum(event['duration_seconds'] for event in info['events']
        if event['kind'] == 'SEARCH_STEP'), 'Omitted stage search time')
    expected_tail = {'version': 'protected-beam-tail-v1', 'phase': replay.phase,
        'baseline_result': replay.baseline['result'], 'baseline_sha256': replay.baseline_sha,
        'guarantee_established': replay.baseline['guarantee_established'], 'identity_binding': replay.identity,
        'identity_sha256': replay.identity_sha, 'prefix_certificates': replay.certificates,
        'events': info['events'], 'counters': info['counters'], 'original_config': cfg['branch_search'],
        'additional_expansion_allowance': cfg['tail_search']['continuation_expansions'],
        'accounting': 'Original checks include WAIT/FAIL/partial; certified-prefix reuse is separate; all expansions, screening and time charged. No cross-episode cache.'}
    same(info, expected_tail, 'protected_tail_result')
    for phase, times in result.get('stage_component_seconds', {}).items():
        require(phase in stages and set(times) == {'observation_seconds', 'selection_seconds', 'search_step_seconds', 'journal_seconds'}
                and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in times.values()),
                'Invalid stage component time')
    tail.check_cumulative_caps(result, baseline, cfg)
    established = baseline['status'] == 'CERTIFIED_INCUMBENT'
    gain = tail.check_protection(baseline, result) if established else None
    if not established:
        require(stages['CONTINUATION'] == stages['VERIFY'] == 0 and result['best_plan'] == baseline['best_plan'],
                'Continued without a complete certified baseline')
    return {'decisions': count, 'baseline_decisions': stages['BASELINE'],
        'continuation_decisions': stages['CONTINUATION'], 'verification_decisions': stages['VERIFY'],
        'prefix_certificates': len(replay.certificates), 'repair_only_expansions': replay.repairs,
        'certificate_reuses': replay.reuses, 'stop_reason': replay.core.reason,
        'complete_baseline_replay': True, 'guard_established': established,
        'nondegradation_verified': established, 'gain_over_paid_baseline_kwh': gain,
        'stage_boundaries': boundaries, 'quality_boundaries': quality}


def check_quality_costs(checkpoints, expected, logical_rows, points, result):
    """Check every certified-quality boundary against paid, cumulative costs."""
    require(len(checkpoints) == len(expected) and checkpoints, 'Missing quality/cost boundary')
    fields = {'search_choices', 'logical_points', 'physical_points', 'numerical_entries', 'requests',
              'quality_pct', 'fully_certified', 'worker_elapsed_seconds', 'elapsed_seconds', 'phase'}
    previous_compute = previous_worker = 0.
    previous_actual = previous_requests = 0
    for checkpoint, wanted in zip(checkpoints, expected):
        require(set(checkpoint) == fields, 'Quality/cost checkpoint schema changed')
        for key, value in wanted.items():
            same(checkpoint[key], value, 'certified_checkpoint.' + key)
        actual = cache.actual_prefix_count(logical_rows, checkpoint['logical_points'])
        require(checkpoint['physical_points'] == actual and checkpoint['numerical_entries'] == 3 + 4 * actual,
                'Quality checkpoint omits real physical or initialization cost')
        computation = window.finite_time(checkpoint['elapsed_seconds'], 'checkpoint computation')
        worker = window.finite_time(checkpoint['worker_elapsed_seconds'], 'checkpoint full worker')
        require(computation >= previous_compute and worker >= previous_worker
                and worker >= result['preflight_elapsed_seconds'] + computation
                and checkpoint['requests'] >= previous_requests,
                'Quality checkpoint resets clock/request count or omits worker startup')
        require(computation - previous_compute >= window.physical_seconds(points, previous_actual, actual),
                'Quality interval excludes newly paid physical work')
        previous_compute, previous_worker = computation, worker
        previous_actual, previous_requests = actual, checkpoint['requests']
    require(previous_compute <= result['computation_elapsed_seconds']
            and previous_worker <= result['elapsed_seconds'], 'Final result omits recorded quality cost')
    return checkpoints[-1]['quality_pct']


def check_baseline_cost(checkpoint, checkpoints, logical_rows):
    index = checkpoint['transition_boundary']
    require(type(index) is int and 0 <= index < len(checkpoints), 'Invalid paid baseline boundary')
    saved, boundary = checkpoint['real_cost'], checkpoints[index]
    expected = {key: value for key, value in boundary.items()
                if key not in ('worker_elapsed_seconds', 'elapsed_seconds')}
    for key, value in expected.items():
        same(saved[key], value, 'paid_baseline_cost.' + key)
    require(set(saved) == set(expected) | {'worker_elapsed_seconds', 'elapsed_seconds'},
            'Baseline real cost schema changed')
    require(saved['logical_points'] == checkpoint['oracle_boundary']['points']
            and saved['requests'] == checkpoint['oracle_boundary']['requests']
            and saved['physical_points'] == cache.actual_prefix_count(logical_rows, saved['logical_points']),
            'Baseline cumulative cost was reset or mislabeled')
    # The stage collector seals immediately after the completed quality boundary.
    require(window.finite_time(saved['elapsed_seconds'], 'baseline compute') >= boundary['elapsed_seconds']
            and window.finite_time(saved['worker_elapsed_seconds'], 'baseline worker') >= boundary['worker_elapsed_seconds'],
            'Baseline seal predates its paid certified boundary')
    if 'elapsed_seconds' in checkpoint:
        require(checkpoint['elapsed_seconds'] >= saved['worker_elapsed_seconds'], 'Baseline seal omits paid worker duration')
    return saved


def check_cold_costs(result, points):
    computation = window.finite_time(result['computation_elapsed_seconds'], 'computation duration')
    initialization = window.finite_time(result['runtime_initialization_seconds'], 'initialization duration')
    preflight = window.finite_time(result['preflight_elapsed_seconds'], 'worker preflight duration')
    elapsed = window.finite_time(result['elapsed_seconds'], 'full worker duration')
    require(computation >= initialization + math.fsum(p['elapsed_seconds'] for p in points.values())
            and elapsed >= computation + preflight, 'Worker time excludes physical, initialization or preflight cost')
    model_load = window.finite_time(result['model_load_seconds'], 'model load duration')
    require(preflight >= model_load, 'Cold model loading is uncharged')
    return {'computation_elapsed_seconds': computation, 'worker_elapsed_seconds': elapsed,
            'preflight_elapsed_seconds': preflight, 'model_load_seconds': model_load}


def check_integration(result, baseline, mech, cfg):
    original = cfg['branch_search']
    adopted = (baseline['status'] == 'CERTIFIED_INCUMBENT'
        and baseline['stop_reason'] == 'BRANCH_EXPANSION_CAP'
        and baseline['actual_physical_points'] < original['point_budget']
        and baseline['checked_candidates'] < original['max_checked_candidates'])
    coverage = {'version': 'alternating_unique_hold_coverage_v1',
        'coverage_turn': 'completed_expansions_mod_2_equals_1',
        'eligible': 'prefix_wait_and_unchecked_and_graph_energy_strictly_above_incumbent',
        'coverage_depth': 'prefix_depth_of_oldest_eligible_branch',
        'coverage_order': 'oldest_unique_full_plans_at_coverage_depth',
        'ordinary_window': 'original_BranchSearch_window_unchanged',
        'no_eligible_fallback': 'original_BranchSearch_window_unchanged',
        'scope': 'alternate_turn_candidate_exposure_only_Beam_selection_unchanged',
        'unexposed_frontier': 'preserved_without_mutation',
        'selection_and_physics': 'inherited_unchanged_from_BranchSearch'}
    width = 27 + mech.horizon + 3 * len(mech.names)
    expected = {'version': 'protected-beam-coverage-tail-v1', 'schema_version': 'clpu-integrated-search-result-v1',
        'baseline_engine': 'original_BranchSearch', 'baseline_selector': 'beam',
        'continuation_window_rule': coverage,
        'continuation_order': ['exact_failed_prefix_filter', 'coverage_window', 'configured_selector'],
        'verification_rule': 'unchanged_tail_priority', 'turn_counter': 'cumulative_beam_turn_no_reset',
        'oracle_scope': 'same_private_oracle_baseline_through_verify', 'logical_point_budget': original['point_budget'],
        'baseline_expansion_cap': original['max_branch_expansions'],
        'additional_expansion_cap': cfg['tail_search']['continuation_expansions'],
        'total_expansion_cap': original['max_branch_expansions'] + cfg['tail_search']['continuation_expansions'],
        'cumulative_checked_cap': original['max_checked_candidates'],
        'cumulative_generated_cap': original['max_generated_candidates'], 'continuation_adopted': adopted,
        'adoption_completed_expansions': len(baseline['trace']) if adopted else None,
        'feature_schema': {'source': 'BranchSearch.observation_unchanged', 'raw_candidate_dim': width,
            'associated_candidate_dim': width + 8, 'global_dim': 11 + len(mech.names),
            'action_window_capacity': original['action_window'], 'load_names_in_feature_order': list(mech.names)},
        'nested_counter_semantics': 'Legacy tail, baseline, plan, event, reward and observation point fields are cumulative logical counts; point_id is logical.',
        'physical_counter_semantics': 'Top-level actual_physical_points and the oracle physical ledger count real solves; no counter resets at phase transitions.'}
    same(result['integration'], expected, 'integration_contract')
    require(result['plan_point_id_semantics'] == 'logical_point_id', 'Ambiguous nested point IDs')
    return adopted


def check_source_bindings(run):
    expected = {'cache_run': CACHE_RUN, 'cache_completion_sha256': CACHE_SHA,
        'window_run': cache.SOURCE_RUN, 'window_completion_sha256': cache.SOURCE_SHA,
        'tail_run': TAIL_RUN, 'tail_completion_sha256': TAIL_SHA,
        'd059_run': LEARNING_RUN, 'd059_completion_sha256': LEARNING_SHA,
        'd060_run': D060_RUN, 'd060_completion_sha256': D060_SHA}
    require(read(run / 'source_binding.json') == expected, 'Integrated scientific source chain changed')
    for prefix in ('cache', 'window', 'tail', 'd059', 'd060'):
        source = ROOT / 'runs' / expected[prefix + '_run']
        require(sha(source / 'completion_manifest.json') == expected[prefix + '_completion_sha256'],
                'Original source root seal changed: ' + prefix)
    models, _, _ = model_sources()
    require(read(run / 'model_bindings.json') == models, 'Fixed six model endpoints changed')
    tail_source, d060 = ROOT / 'runs' / TAIL_RUN, ROOT / 'runs' / D060_RUN
    tail_seal = {n.replace('\\', '/'): v for n, v in read(tail_source / 'completion_manifest.json').items()}
    d060_seal = {n.replace('\\', '/'): v for n, v in read(d060 / 'completion_manifest.json').items()}
    require(sha(tail_source / 'model_bindings.json') == tail_seal['model_bindings.json']
            and read(tail_source / 'model_bindings.json') == models, 'Protected-tail model provenance changed')
    d060_bindings = read(d060 / 'model_bindings.json')
    require(sha(d060 / 'model_bindings.json') == d060_seal['model_bindings.json'], 'D060 model provenance changed')
    for binding in models:
        require(sha(run / binding['checkpoint']) == sha(tail_source / binding['checkpoint'])
                == tail_seal[binding['checkpoint']] == binding['source_sha256'], 'Copied model differs from frozen endpoints')
        matching = [r for r in d060_bindings if r['seed'] == binding['seed'] and r['method'] == binding['method']]
        require(len(matching) == 1 and matching[0]['source_sha256'] == binding['source_sha256'],
                'D060 endpoint differs from same-seed fixed actor')
        require(sha(d060 / matching[0]['checkpoint']) == d060_seal[matching[0]['checkpoint']] == binding['source_sha256'],
                'D060 frozen model tensor file changed')
    return models


def check_frozen_physics(run, prepared, cfg):
    source = ROOT / 'runs' / CACHE_RUN
    sealed = {name.replace('\\', '/'): value for name, value in read(source / 'completion_manifest.json').items()}
    original = read(source / 'config.json')
    require(cfg['integrated_study'] == STUDY and cfg['branch_search'] == original['branch_search'],
            'Integrated study or unchanged original Beam settings differ')
    require(cfg['tail_search'] == {'continuation_expansions': 256, 'enable_exact_failed_prefix': True,
                                  'enable_tail_verification': True}, 'Tail settings changed')
    for key in ('horizon', 'slot_seconds', 'sample_seconds', 'alpha', 'algorithm', 'condition_id'):
        require(cfg[key] == original[key], 'Inherited scientific setting changed: ' + key)
    fixed = ('training_scenarios.json', 'author_static_metadata.json', 'load_reference.json', 'conditions.json',
        'physical_config.json', 'operating_reference.json', 'base_load_properties.json', 'command_inventory.json',
        'source_pointer.json', 'metadata.json', 'adaptation.json', 'author_input_precheck.json', 'condition.json',
        'source_provenance.json', 'event_split.json')
    for name in (*fixed, 'population.json', 'selected_scenarios.json', 'selected_events.json', 'training_contexts.json'):
        require(sha(run / name) == sealed[name] == sha(source / name), 'Inherited physical input changed: ' + name)
    for name, expected in sealed.items():
        if name.startswith('source_inputs/'):
            require(sha(run / name) == expected, 'Original solver input changed: ' + name)
    scope = read(run / 'physical_scope.json')
    needed = set(fixed) | {'config.json', 'selected_scenarios.json', 'selected_events.json',
                          'training_contexts.json', 'dependencies.json', 'python_environment.json'}
    needed |= {p.relative_to(run).as_posix() for p in (run / 'source_inputs').rglob('*') if p.is_file()}
    require(scope['scope_version'] == 'CLPU_FULL_FIXED_PHYSICAL_SCOPE_V1'
            and set(scope['input_files']) == needed
            and scope['input_files'] == {name: sha(run / name) for name in needed},
            'Complete fixed solver-input scope changed')
    require(scope['science_source_sha256'] == {name: sha(ROOT / name) for name in prepared['source_files']
                                               if name.endswith('.py')}, 'Complete implementation scope changed')
    for key, value in {'source_provenance': read(run / 'source_provenance.json'),
        'runtime_dependencies': read(run / 'dependencies.json'), 'python_environment': read(run / 'python_environment.json'),
        'physical_config': read(run / 'physical_config.json'), 'alpha': .5, 'algorithm': 0,
        'load_setpoint_precision': '.15g', 'reset_semantics': 'INDEPENDENT_QUASI_STATIC_AUTHOR_RESET_EACH_SNAPSHOT',
        'cache_scope': 'one cold cell only; no historical or cross-arm/cross-context answers'}.items():
        require(scope[key] == value, 'Fixed physical scope changed: ' + key)
    from nature34_experiment_support import dependencies
    require(dependencies() == scope['runtime_dependencies']
            and sha(Path(sys.executable)) == scope['python_environment']['sha256'],
            'Runtime environment differs from freeze')
    binding = read(run / 'scope_review_binding.json')
    require(binding['path'] == 'artifacts/clpu_equivalent_cache_scope_review_20261030'
            and sha(run / 'scope_review/completion_manifest.json') == binding['completion_sha256']
            and sha(run / 'scope_review/summary.json') == binding['summary_sha256'], 'Raw scope review binding changed')
    manifest(run / 'scope_review')
    review = read(run / 'scope_review/summary.json')
    require(review['status'] == 'PASS' and review['new_physical_points'] == review['new_training'] == 0
            and review['physical_points_reviewed'] == 1121 and review['duplicate_pairs_reviewed'] == 538
            and review['certificate_or_feedback_disagreements'] == 0
            and review['constraint_or_audit_tolerances_changed'] is False, 'Complete raw scope review not passed')
    cache.check_exact_key_review(run, binding['completion_sha256'])
    return scope


def expected_schedule(contexts, models):
    require(len(contexts) == 8 and {r['context_id'] for r in contexts} == set(range(8)), 'Fixed eight contexts changed')
    ordered = sorted(contexts, key=lambda r: digest_text('clpu-integrated-context-order-v1|' + canonical(r)))
    slots = {(arm if seed == 0 else arm + str(seed)): (arm, seed) for arm, seed in ARM_SEEDS}
    order = sorted(slots, key=lambda slot: digest_text('clpu-integrated-arm-order-v1|' + slot))
    model = {(r['method'], r['seed']): r for r in models}
    rows = []
    for rank, context in enumerate(ordered):
        for position in range(8):
            slot = order[(position + rank) % 8]
            arm, seed = slots[slot]
            method = 'bc_raw' if arm == 'protected_bc' else 'ppo_raw' if arm == 'protected_ppo' else 'beam'
            binding = model.get((method, seed))
            index = len(rows)
            rows.append({**context, 'arm': arm, 'arm_slot': slot, 'seed': seed, 'method': method,
                'phase': 'evaluation', 'group': 'integrated', 'representation': 'raw', 'deterministic': True,
                'point_budget': 256, 'logical_point_budget': 256, 'physical_point_budget': 256,
                'continuation_expansions': 0 if arm == 'window_reference' else 256,
                'window_variant': 'coverage_v1', 'protected': arm != 'window_reference',
                'cache_variant': 'electrical_equivalent', 'context_order': rank, 'arm_position': position,
                'pair_order': rank * 4 + position // 2, 'episode_id': index, 'cell': f'episodes/{index:03d}',
                'checkpoint': binding['checkpoint'] if binding else None,
                'checkpoint_sha256': binding['source_sha256'] if binding else None})
    return rows


def digest_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


def check_historical_bindings(run, contexts):
    source = ROOT / 'runs' / cache.SOURCE_RUN
    root_seal = read(source / 'completion_manifest.json')
    bindings = read(run / 'source_cells.json')
    require(len(bindings) == 8 and {r['context_id'] for r in bindings} == set(range(8)), 'Missing historical context')
    for context in contexts:
        item = next(r for r in bindings if r['context_id'] == context['context_id'])
        require(all(item[k] == context[k] for k in ('scenario_id', 'event_id')), 'Historical context relabeled')
        for group, variant in (('g0', 'original'), ('g1', 'coverage_v1')):
            record = item[group]
            original = cache.source_cell_binding(source, record['source_cell'])
            require(record['source_completion_sha256'] == original['source_completion_sha256'], 'Historical cell seal changed')
            directory = (run / record['historical_directory']).resolve()
            require(directory.is_relative_to(run) and directory != run, 'Historical path escapes cold run')
            require(sha(directory / 'completion_manifest.json') == record['source_completion_sha256'], 'Copied historical seal changed')
            sealed = read(directory / 'completion_manifest.json')
            require(set(record['source_files']) == {'spec.json', 'result.json', 'steps.jsonl.gz', 'plans.jsonl.gz',
                                                   'requests.jsonl', 'points.jsonl.gz'}, 'Historical scientific file list differs')
            for name, expected in record['source_files'].items():
                require(sealed[name] == root_seal[record['source_cell'] + '/' + name] == expected,
                        'Historical source scientific file changed: ' + name)
                if name != 'points.jsonl.gz':
                    require(sha(directory / name) == expected, 'Copied history changed: ' + name)
            require(not (directory / 'points.jsonl.gz').exists(), 'Historical physical answers copied into cold run')
            spec = read(directory / 'spec.json')
            require(spec == read(directory / 'result.json')['spec'] and spec['window_variant'] == variant
                    and all(spec[k] == context[k] for k in ('context_id', 'scenario_id', 'event_id')),
                    'Wrong historical source method/context')


def historical_directory(run, context_id, group):
    matches = [r for r in read(run / 'source_cells.json') if r['context_id'] == context_id]
    require(len(matches) == 1 and group in ('g0', 'g1'), 'Invalid historical source lookup')
    path = (run / matches[0][group]['historical_directory']).resolve()
    require(path.is_relative_to(run) and path != run, 'Historical source escaped run')
    return path


def historical_semantics(directory):
    result = read(directory / 'result.json')
    return {'result': {key: result[key] for key in ('status', 'config', 'stop_reason', 'best_plan',
            'initial_certificate', 'wait_baseline_quality_pct', 'completed_expansions', 'search_decisions', 'counts')},
        'plans': list(read_lines(directory / 'plans.jsonl.gz')),
        'steps': cache.semantic_step_view(list(read_lines(directory / 'steps.jsonl.gz'))),
        'requests': list(read_lines(directory / 'requests.jsonl')),
        'checkpoints': [{k: c[k] for k in ('search_choices', 'physical_points', 'quality_pct', 'fully_certified')}
                        for c in result['checkpoints']]}


def check_historical_baseline(directory, baseline, checkpoint, request_rows):
    historical = historical_semantics(directory)
    result = historical['result']
    for key in ('status', 'config', 'stop_reason', 'initial_certificate', 'best_plan'):
        same(baseline[key], result[key], 'historical_original_Beam.' + key)
    same(baseline['attempts'], historical['plans'], 'historical_original_Beam.attempts')
    require(len(baseline['trace']) == result['completed_expansions'] == result['search_decisions'],
            'Original Beam historical expansion count differs')
    same(baseline['trace'], [r['response'] for r in historical['steps'] if r['status'] == 'COMPLETED'],
         'historical_original_Beam.complete_trace')
    expected_counts = {'generated_candidates': baseline['generated_unique_candidates'],
        'graph_assessed_candidates': baseline['assessed_graph_candidates'], 'checked_candidates': baseline['checked_candidates'],
        'branch_expansions': len(baseline['trace']), 'generation_cap_rejections': baseline['generation_cap_rejections']}
    same(result['counts'], expected_counts, 'historical_original_Beam.cumulative_counts')
    same([r for r in request_rows if r['request'] <= checkpoint['oracle_boundary']['requests']],
         historical['requests'], 'historical_original_Beam.logical_queries')
    return digest(baseline)


def check_physical_and_logical(run, cell, spec, cfg, scope, mech, result):
    points, statuses = branch.check_raw_points(run, cell, spec, cfg, mech)
    commands = old.check_commands(cell, points, read(run / 'command_inventory.json'))
    requests = old.check_requests(cell, points, 256)
    actual = read(cell / 'oracle_summary.json')
    require(actual['status'] == 'COMPLETE_COUNTERS_PENDING_AUDIT' and actual['points'] == len(points)
            and actual['point_budget'] == 256 and actual['actual_numerical_entries'] == commands['numerical_entries']
            and actual['requests'] == requests['requests'] == len(points)
            and actual['cache_hits'] == requests['cache_hits'] == requests['denied'] == 0
            and actual['statuses'] == dict(statuses) and actual['initialization_numerical_entries'] == 3
            and actual['point_seconds'] == [p['elapsed_seconds'] for p in points.values()],
            'Actual cold oracle ledger changed or hides cost')
    raw = {row['point_id']: row for row in read_lines(cell / 'points.jsonl.gz')}
    logical_rows = list(read_lines(cell / 'logical_points.jsonl.gz'))
    logical, logical_requests, costs = cache.check_logical_records(logical_rows,
        list(read_lines(cell / 'logical_requests.jsonl')), raw, mech, scope, sha(run / 'condition.json'), 'electrical_equivalent')
    cache.check_wrapper_summary(read(cell / 'equivalent_cache_summary.json'), result['oracle'], costs, logical,
                                scope, 'electrical_equivalent', commands['numerical_entries'])
    require(result['actual_physical_points'] == len(points) and result['logical_points'] == len(logical),
            'Result top-level real/logical counters differ')
    return points, statuses, commands, logical_rows, logical, logical_requests, costs


def check_reference_records(run, spec, cfg, result, logical_rows, points, requests, plans, steps, mech,
                            transitions, decisions, stage_events):
    oracle = branch.PaidRequests(requests, points, mech, 256, sha(run / 'condition.json'))
    core = window.WindowReplay(mech, cfg['branch_search'], oracle, plans)

    class Adapter:
        phase = 'WINDOW'

        def __getattr__(self, key):
            return getattr(core, key)

    replay = Adapter()
    expected, stage_counts = [], [oracle.counts()]

    def boundary(index):
        expected.append({'search_choices': index, 'logical_points': oracle.used, 'requests': oracle.counts()['requests'],
            'quality_pct': 100 * core.quality if core.best is not None else None,
            'fully_certified': core.best is not None, 'phase': 'DONE' if core.done else 'WINDOW'})

    boundary(0)
    normalized = cache.semantic_step_view(steps, logical_rows)
    require(len(steps) == 2 * len(transitions), 'Reference steps/transitions differ')
    decisions = iter(decisions)
    last_elapsed = result['checkpoints'][0]['elapsed_seconds']
    for index, row in enumerate(transitions):
        require(not core.done, 'Window action after terminal state')
        first, last = normalized[2 * index:2 * index + 2]
        require(first['status'] == 'STARTED' and last['status'] == 'COMPLETED'
                and first['transition_id'] == last['transition_id'] == index, 'Window step chronology changed')
        for key in ('method', 'action', 'branch_id', 'oracle_before'):
            same(first[key], row[key], 'window_step_transition.' + key)
        require(first['window_branch_ids'] == row['obs_raw']['branch_ids'], 'Window step exposure differs')
        for key, value in first.items():
            if key != 'status':
                same(last[key], value, 'window_step_completion.' + key)
        before = window.finite_time(steps[2 * index]['elapsed_seconds'], 'window STARTED')
        after = window.finite_time(steps[2 * index + 1]['elapsed_seconds'], 'window COMPLETED')
        require(last_elapsed <= before <= after, 'Window step time reversed')
        tail.check_transition(replay, row, decisions, spec, index, index)
        require(after - before >= window.physical_seconds(points, first['oracle_before']['points'], oracle.used),
                'Reference step omits paid physical time')
        same(last['response'], core.trace[-1], 'independent_window_step_response')
        require(last['oracle_after'] == oracle.counts(), 'Window after-step costs differ')
        last_elapsed = window.finite_time(result['checkpoints'][index + 1]['elapsed_seconds'], 'window quality boundary')
        require(last_elapsed >= after, 'Quality recorded before completed reference step')
        boundary(index + 1)
    require(core.done and next(decisions, None) is None and oracle.index == len(oracle.rows)
            and oracle.used == len(points) and core.attempt_index == len(plans), 'Incomplete reference evidence replay')
    require(result['search_decisions'] == core.turn == len(transitions)
            and result['stage_decisions'] == {'WINDOW': core.turn}, 'Reference decision count changed')
    stage_counts.append(oracle.counts())
    require(len(stage_events) == 2, 'Reference stage journal changed')
    for event, phase, count, counters in zip(stage_events, ('WINDOW', 'DONE'), (0, core.turn), stage_counts):
        require(set(event) == {'stage', 'transition_boundary', 'oracle', 'utc'}
                and event['stage'] == phase and event['transition_boundary'] == count and event['oracle'] == counters,
                'Reference cumulative stage boundary changed')
        datetime.fromisoformat(event['utc'])
    view = {**result, 'actual_physical_points': result['logical_points']}
    for key, value in tail.baseline_snapshot(core).items():
        same(view[key], value, 'reference_complete_search.' + key)
    compact = {key: result[key] for key in ('status', 'config', 'stop_reason', 'best_plan', 'initial_certificate')}
    compact.update(wait_baseline_quality_pct=expected[0]['quality_pct'], completed_expansions=core.turn,
        search_decisions=core.turn, counts={'generated_candidates': len(core.keys),
        'graph_assessed_candidates': len(core.assessed), 'checked_candidates': len(core.checked),
        'branch_expansions': core.turn, 'generation_cap_rejections': core.rejections})
    semantic = {'result': compact, 'plans': plans, 'steps': normalized, 'requests': requests,
        'checkpoints': [{'search_choices': row['search_choices'], 'physical_points': row['logical_points'],
                        'quality_pct': row['quality_pct'], 'fully_certified': row['fully_certified']} for row in expected]}
    same(semantic, historical_semantics(historical_directory(run, spec['context_id'], 'g1')),
         'complete_current_WindowBeam_historical_semantics')
    checked = {'decisions': core.turn, 'stop_reason': core.reason, 'final_quality_pct': expected[-1]['quality_pct'],
               'historical_equivalence': True, 'quality_boundaries': expected}
    return checked, expected, digest(semantic)


def check_guard_claim(result, baseline):
    established = baseline['status'] == 'CERTIFIED_INCUMBENT'
    if established:
        tail.check_protection(baseline, result)
        expected = {'guard_established': True, 'baseline_status': baseline['status'],
            'baseline_energy_kwh': float(baseline['best_plan']['validation']['energy_kwh']),
            'final_energy_kwh': float(result['best_plan']['validation']['energy_kwh']),
            'nondegradation_verified': True,
            'scope': 'Fixed 30-minute normal-service planning objective; computation waiting is not in the objective'}
    else:
        require(result['best_plan'] == baseline['best_plan'], 'Uncertified baseline was silently replaced')
        expected = {'guard_established': False, 'baseline_status': baseline['status'],
                    'reason': 'NO_COMPLETE_BASELINE_CERTIFICATE'}
    same(result['guard_audit'], expected, 'guard_claim')
    return established


def check_cell(run, spec, cfg, scope):
    cell = (run / spec['cell']).resolve()
    require(cell != run and cell.is_relative_to(run), 'Integrated cell escaped run')
    sealed = manifest(cell)
    required = {'spec.json', 'result.json', 'initial_state.json', 'plans.jsonl.gz', 'worker_started.json',
        'points.jsonl.gz', 'requests.jsonl', 'command_ledger.jsonl', 'command_stats.json', 'metadata.json',
        'oracle_summary.json', 'logical_points.jsonl.gz', 'logical_requests.jsonl', 'equivalent_cache_summary.json'}
    required |= {'transitions.jsonl.gz', 'decisions.jsonl.gz', 'stage_events.jsonl', 'cost_checkpoints.jsonl'}
    required |= ({'steps.jsonl.gz'} if not spec['protected'] else {
        'baseline_result.json', 'baseline_checkpoint.json', 'physical_identity.json', 'tail_events.jsonl'})
    require(required <= set(sealed) and not (cell / 'error.json').exists(), 'Missing or failed integrated cell evidence')
    result = read(cell / 'result.json')
    require(read(cell / 'spec.json') == result['spec'] == spec, 'Cell changed its frozen scientific identity')
    worker = read(cell / 'worker_started.json')
    require(worker['spec_sha256'] == sha(cell / 'spec.json') and type(worker['pid']) is int and worker['pid'] > 0,
            'Worker start/spec binding changed')
    require(all(result[k] == 0 for k in ('training_interactions', 'optimizer_updates', 'new_training'))
            and all(result[k] is False for k in ('formal_holdout_numerical_access', 'tune_numerical_access', 'clpu_holdout_access')),
            'Unapproved training or partition access')
    scenarios = {row['scenario_id']: row for row in read(run / 'selected_scenarios.json')}
    events = {row['event_id']: row for row in read(run / 'selected_events.json')}
    mech = branch.Mechanics(read(run / 'author_static_metadata.json'), read(run / 'load_reference.json'),
        scenarios[spec['scenario_id']], read(run / 'condition.json'), events[spec['event_id']], cfg)
    require(read(cell / 'initial_state.json') == mech.initial, 'Initial history changed')
    points, statuses, commands, logical_rows, logical, requests, costs = check_physical_and_logical(
        run, cell, spec, cfg, scope, mech, result)
    timing = check_cold_costs(result, points)
    plans = list(read_lines(cell / 'plans.jsonl.gz'))
    require(list(read_lines(cell / 'cost_checkpoints.jsonl')) == result['checkpoints'], 'Immediate quality journal omitted or changed')
    require(result['counts'] == {k: result[k] for k in ('generated_unique_candidates', 'assessed_graph_candidates',
                                                      'checked_candidates', 'frontier_size')}, 'Result count summary differs')
    baseline_cost = baseline_quality = baseline_signature = gain = guard = None
    policy_count, baseline_replayed = 0, False
    if not spec['protected']:
        require(result['config'] == cfg['branch_search'], 'Window reference settings changed')
        checked, expected, signature = check_reference_records(run, spec, cfg, result, logical_rows, logical, requests,
            plans, list(read_lines(cell / 'steps.jsonl.gz')), mech, list(read_lines(cell / 'transitions.jsonl.gz')),
            list(read_lines(cell / 'decisions.jsonl.gz')), list(read_lines(cell / 'stage_events.jsonl')))
        policy = check_policy(run, spec, [])
    else:
        baseline, checkpoint = read(cell / 'baseline_result.json'), read(cell / 'baseline_checkpoint.json')
        require(checkpoint['baseline_result_sha256'] == result['baseline_result_sha256'] == sha(cell / 'baseline_result.json')
                and result['baseline_checkpoint_sha256'] == sha(cell / 'baseline_checkpoint.json')
                and checkpoint['physical_identity_sha256'] == sha(cell / 'physical_identity.json'), 'Immutable Beam seal changed')
        identity = {'study': STUDY, 'freeze_sha256': sha(run / 'prephysical_freeze_manifest.json'),
            'scenario_id': spec['scenario_id'], 'event_id': spec['event_id'], 'initial_state': mech.initial,
            'condition_sha256': sha(run / 'condition.json'), 'physical_config_sha256': sha(run / 'physical_config.json'),
            'author_static_metadata_sha256': sha(run / 'author_static_metadata.json'),
            'source_provenance_sha256': sha(run / 'source_provenance.json')}
        require(read(cell / 'physical_identity.json') == identity, 'Failed-prefix fixed scope changed')
        require(plans == result['attempts'] and list(read_lines(cell / 'tail_events.jsonl')) == result['tail']['events'],
                'Omitted attempted plan or immediate tail event')
        transitions = list(read_lines(cell / 'transitions.jsonl.gz'))
        checked = check_protected_records(mech, cfg, result, baseline, checkpoint, transitions,
            list(read_lines(cell / 'decisions.jsonl.gz')), requests, logical, sha(run / 'condition.json'), identity,
            list(read_lines(cell / 'stage_events.jsonl')))
        check_integration(result, baseline, mech, cfg)
        policy = check_policy(run, spec, transitions)
        policy_count = policy['policy_decisions']
        require(policy_count == (checked['continuation_decisions'] if spec['method'] != 'beam' else 0),
                'Policy decision attribution count differs')
        expected = checked['quality_boundaries']
        baseline_cost = check_baseline_cost(checkpoint, result['checkpoints'], logical_rows)
        same(result['baseline_real_cost'], baseline_cost, 'result_paid_baseline_cost')
        baseline_quality = baseline_cost['quality_pct']
        baseline_signature = check_historical_baseline(historical_directory(run, spec['context_id'], 'g0'),
                                                       baseline, checkpoint, requests)
        guard, gain, baseline_replayed = check_guard_claim(result, baseline), checked['gain_over_paid_baseline_kwh'], True
        signature = digest({'trace': result['trace'], 'attempts': plans, 'best_plan': result['best_plan']})
    quality = check_quality_costs(result['checkpoints'], expected, logical_rows, points, result)
    check_phase_costs(result, baseline_cost)
    return {'cell': spec['cell'], 'context_id': spec['context_id'], 'arm': spec['arm'], 'arm_slot': spec['arm_slot'],
        'seed': spec['seed'], 'method': spec['method'], 'status': 'PASS', 'result_status': result['status'],
        'cell_completion_sha256': sha(cell / 'completion_manifest.json'), 'quality_pct': quality,
        'logical_points': len(logical), 'physical_points': len(points), 'numerical_entries': commands['numerical_entries'],
        'requests': costs['requests'], 'semantic_cache_hits': costs['semantic_cache_hits'],
        'electrical_reuses': costs['electrical_reuses'], 'physical_statuses': dict(statuses),
        'policy_decisions': policy_count, 'policy': policy, 'guard_established': guard,
        'baseline_quality_pct': baseline_quality, 'baseline_real_cost': baseline_cost,
        'gain_over_paid_baseline_kwh': gain, 'quality_checkpoints_verified': True,
        'complete_baseline_replay': baseline_replayed, 'baseline_signature_sha256': baseline_signature,
        'semantic_signature_sha256': signature, 'search': checked, **timing}


def check_controller(controller, schedule, checks, dispatch):
    cells = {r['cell'] for r in schedule}
    require(controller['status'] == 'COMPLETE_CELLS_PENDING_AUDIT' and controller['errors'] == []
            and controller['unknown_cost_cells'] == [] and controller['scheduled'] == len(schedule) == 64
            and len(controller['completed']) == 64 and set(controller['completed']) == cells,
            'Controller did not complete all 64 cold cells')
    exits = controller['exits']
    require(len(exits) == 64 and {r['cell'] for r in exits} == cells
            and all(r['exit_code'] == 0 and r['drained_after_controller_error'] is False for r in exits),
            'Missing, failed or ambiguous cold process exit')
    require(len(checks) == 64 and {r['cell'] for r in checks} == cells, 'Audit population incomplete')
    require([r['cell'] for r in dispatch] == [r['cell'] for r in schedule], 'Cold dispatch order changed or retried')
    ended = {r['cell']: r for r in exits}
    previous_finished, pair_finished = 0., []
    for index, (spec, started) in enumerate(zip(schedule, dispatch)):
        end = ended[spec['cell']]
        if index % 2 == 0:
            if pair_finished:
                previous_finished = max(pair_finished)
            pair_finished = []
        for key in ('cell', 'pair_order', 'context_order', 'arm_position'):
            require(started[key] == spec[key], 'Dispatch does not match frozen Latin schedule')
        require(type(started['launcher_pid']) is int and started['launcher_pid'] > 0
                and end['launcher_pid'] == started['launcher_pid'] and end['pair_order'] == spec['pair_order']
                and end['started_epoch_seconds'] == started['started_epoch_seconds'], 'Process launch/exit binding changed')
        began = window.finite_time(started['started_epoch_seconds'], 'cold launch epoch')
        finish = window.finite_time(end['finished_epoch_seconds'], 'cold exit epoch')
        require(previous_finished <= began <= finish, 'Next pair launched before previous pair fully exited')
        datetime.fromisoformat(started['utc'])
        datetime.fromisoformat(end['utc'])
        pair_finished.append(finish)
    logical, physical, entries = (sum(r[k] for r in checks)
                                  for k in ('logical_points', 'physical_points', 'numerical_entries'))
    require(controller['known_logical_points'] == logical <= 16384
            and controller['known_physical_points'] == physical <= 16384
            and controller['known_numerical_entries'] == entries == 64 * 3 + 4 * physical,
            'Controller omits actual or logical costs')
    for row in checks:
        end = ended[row['cell']]
        require(window.finite_time(end['process_wall_seconds'], 'whole cold process wall') >= row['worker_elapsed_seconds'],
                'Process wall duration omits worker work')
        row['process_wall_seconds'] = end['process_wall_seconds']
    window.finite_time(controller['wall_seconds'], 'controller duration')
    return logical, physical, entries


def check_phase_costs(result, baseline):
    final, last = result['final_real_cost'], result['checkpoints'][-1]
    require(set(final) == set(last), 'Final paid boundary schema differs')
    for key, value in last.items():
        if key not in ('worker_elapsed_seconds', 'elapsed_seconds'):
            same(final[key], value, 'final_paid_boundary.' + key)
    require(final['phase'] == 'DONE', 'Final paid boundary still active')
    for key, total in (('worker_elapsed_seconds', 'elapsed_seconds'), ('elapsed_seconds', 'computation_elapsed_seconds')):
        require(window.finite_time(final[key], 'final paid ' + key) >= last[key]
                and final[key] <= result[total], 'Final cost outside actual worker duration')
    names = ('logical_points', 'physical_points', 'numerical_entries', 'requests', 'worker_elapsed_seconds', 'elapsed_seconds')
    expected = {'WINDOW': {k: final[k] for k in names}} if baseline is None else {
        'BASELINE': {k: baseline[k] for k in names}, 'AFTER_BASELINE': {k: final[k] - baseline[k] for k in names}}
    require(all(v >= 0 for row in expected.values() for v in row.values()), 'Stage cost reset/negative duration')
    same(result['phase_costs'], expected, 'full_cumulative_stage_costs')


def protocol_settings():
    return {'logical_point_budget': 256, 'physical_point_budget': 256,
        'window_expansions': 256, 'baseline_expansions': 256, 'continuation_expansions': 256,
        'max_selected_decisions': 768, 'max_workers': 2, 'cooperative_seconds': 14400,
        'worker_seconds': 3600, 'total_physical_budget': 16384,
        'minimum_free_disk_bytes': 12 * 2**30, 'forecast_margin': 1.25,
        'report_source_reserve_bytes': 2 * 2**30, 'cache_variant': 'electrical_equivalent',
        'no_training': True, 'no_historical_answers': True, 'no_cross_cell_cache': True,
        'no_automatic_retry': True, 'same_cell_baseline_continuation_cache': True,
        'dispatch': 'Hashed context and initial slot order; Latin rotation; adjacent two-worker pairs drain before next pair and context',
        'timing': 'Same-run cold full-process and worker/computation timing; no old-run speed comparator',
        'aggregation': 'Average seeds within context before averaging eight contexts'}


def check_model_preflight(run, models):
    binding, directory = read(run / 'model_preflight_binding.json'), run / 'model_preflight'
    require(binding['path'] == 'artifacts/clpu_integrated_model_preflight_20261030'
            and sha(directory / 'completion_manifest.json') == binding['completion_sha256']
            and sha(directory / 'summary.json') == binding['summary_sha256'], 'Model source preflight binding changed')
    manifest(directory)
    evidence = read(directory / 'summary.json')
    architecture = {'candidate_dim': 261, 'global_dim': 79, 'representation': 'raw', 'hidden': 128}
    require(evidence['status'] == 'PASS' and evidence['errors'] == []
            and evidence['new_physical_points'] == evidence['new_training'] == 0
            and evidence['models'] == [{**model, 'architecture': architecture} for model in models]
            and evidence['source_completion_sha256'] == LEARNING_SHA
            and evidence['supplement_completion_sha256'] == SUPPLEMENT_SHA
            and evidence['learning_audit_records'] == 49, 'Incomplete or changed model source verification')
    starts = evidence['same_seed_starts']
    require(len(starts) == 3 and {r['seed'] for r in starts} == set(SEEDS)
            and all(r['all_tensors_exactly_equal'] is True for r in starts), 'Missing common BC/PPO tensor start proof')
    source = ROOT / 'runs' / LEARNING_RUN
    for row in starts:
        seed = row['seed']
        require(row['bc_start_sha256'] == sha(source / f'learning/seed{seed}/raw/start.pt')
                and row['ppo_round00_before_sha256'] == sha(source / f'learning/seed{seed}/raw/ppo_round00/before.pt'),
                'Preflight tensor source changed')
    require(evidence['interpretation'] == 'Common BC teacher/start; PPO has additional on-policy interaction training. Not equal total training data/cost.',
            'Model relationship overstated')


def check_parent(run):
    run = Path(run).resolve()
    require(run.parent == ROOT / 'runs', 'Integrated pilot outside project runs')
    prepared = read(run / 'prepared.json')
    require(prepared['status'] == 'FROZEN_PREPARED_NO_DSS'
            and sha(run / 'prephysical_freeze_manifest.json') == prepared['freeze_sha256'], 'Integrated freeze changed')
    frozen = manifest(run, 'prephysical_freeze_manifest.json')
    required = {'scripts/audit_clpu_integrated_pilot.py', 'scripts/run_clpu_integrated_pilot.py',
        'scripts/report_clpu_integrated_pilot.py', 'src/dnr/nature34_clpu_integrated.py',
        'scripts/audit_clpu_equivalent_cache.py', 'scripts/audit_nature34_clpu_tail.py',
        'scripts/audit_clpu_window_pilot.py', 'scripts/audit_nature34_clpu_branch_learning.py',
        'docs/clpu_integrated_pilot_protocol_v1.md', 'tests/test_clpu_integrated_audit.py'}
    require(required <= set(prepared['source_files']) and len(prepared['source_files']) == len(set(prepared['source_files'])),
            'Required scientific sources/auditors not frozen')
    for name in prepared['source_files']:
        target = (ROOT / name).resolve()
        require(target.is_relative_to(ROOT) and target != ROOT
                and sha(target) == sha(run / 'source' / name), 'Executing frozen implementation changed: ' + name)
    sources = (('cache', CACHE_RUN, CACHE_SHA), ('window', cache.SOURCE_RUN, cache.SOURCE_SHA),
               ('tail', TAIL_RUN, TAIL_SHA), ('d059', LEARNING_RUN, LEARNING_SHA), ('d060', D060_RUN, D060_SHA))
    for label, origin, expected in sources:
        source = ROOT / 'runs' / origin
        require(sha(source / 'completion_manifest.json') == expected
                and sha(run / f'provenance/{label}/source_completion_manifest.json') == expected,
                'Copied/original source seal changed: ' + label)
        sealed = {n.replace('\\', '/'): v for n, v in read(source / 'completion_manifest.json').items()}
        for name in prepared['source_files']:
            if 'source/' + name in sealed:
                require(sha(ROOT / name) == sealed['source/' + name], 'Inherited scientific source changed: ' + name)
        for name in ('model_bindings.json', 'source_binding.json', 'terminal_status.json'):
            if name in sealed:
                require(sha(run / f'provenance/{label}' / name) == sealed[name], 'Copied attribution changed')
    models = check_source_bindings(run)
    check_model_preflight(run, models)
    cfg = read(run / 'config.json')
    require(cfg['integrated_pilot'] == protocol_settings(), 'Integrated experimental/aggregation protocol changed')
    original = read(ROOT / 'runs' / CACHE_RUN / 'config.json')
    expected_cfg = {**original, 'integrated_study': STUDY, 'integrated_pilot': protocol_settings(),
        'tail_search': {'continuation_expansions': 256, 'enable_exact_failed_prefix': True, 'enable_tail_verification': True}}
    same(cfg, expected_cfg, 'full_frozen_integrated_config')
    scope = check_frozen_physics(run, prepared, cfg)
    contexts = [{'context_id': i, 'scenario_id': r['scenario_id'], 'event_id': r['event_id']}
                for i, r in enumerate(read(run / 'population.json')['contexts'])]
    schedule = read(run / 'schedule.json')
    same(schedule, expected_schedule(contexts, models), 'complete_64_cell_Latin_schedule')
    require(sha(run / 'historical/source_completion_manifest.json') == cache.SOURCE_SHA
            and sha(run / 'historical/source_audit_summary.json') ==
                read(ROOT / 'runs' / cache.SOURCE_RUN / 'completion_manifest.json')['audit/summary.json'],
            'Historical independent audit not source-bound')
    check_historical_bindings(run, contexts)
    cases = list(ET.parse(run / 'software_tests.xml').getroot().iter('testcase'))
    require(cases and not any(list(c.iter('failure')) + list(c.iter('error')) + list(c.iter('skipped')) for c in cases),
            'Frozen component tests incomplete')
    for name in ('test_run_clpu_integrated_pilot', 'test_report_clpu_integrated_pilot',
                 'test_nature34_clpu_integrated', 'test_clpu_integrated_audit'):
        require(any(name in c.get('classname', '') for c in cases), 'Missing component tests: ' + name)
    budget = read(run / 'resource_budget.json')
    for key, value in {'contexts': 8, 'cells': 64, 'physical_points': 16384, 'numerical_entries': 65728,
        'workers': 2, 'minimum_free_bytes': 12 * 2**30, 'cooperative_seconds': 14400, 'worker_seconds': 3600,
        'new_training': 0, 'software_tests': len(cases), 'forecast_margin': 1.25,
        'report_source_reserve_bytes': 2 * 2**30, 'historical_cache_cells': 16, 'historical_training_tail_cells': 32,
        'forecast_is_hard_upper_bound': False}.items():
        require(budget[key] == value, 'Integrated resource contract changed: ' + key)
    sizes = []
    for source_name, cohort, expected in ((CACHE_RUN, None, 16), (TAIL_RUN, 'training_mechanism', 32)):
        source = ROOT / 'runs' / source_name
        rows = [r for r in read(source / 'schedule.json') if cohort is None or r['cohort'] == cohort]
        require(len(rows) == expected, 'Resource forecast cohort changed')
        sizes.append(max(sum(p.stat().st_size for p in (source / r['cell']).rglob('*') if p.is_file()) for r in rows))
    require(budget['cache_max_cell_bytes'] == sizes[0] and budget['tail_training_max_cell_bytes'] == sizes[1]
            and budget['historical_max_cell_bytes'] == max(sizes)
            and budget['initial_required_free_bytes'] == math.ceil(64 * max(sizes) * 1.25) + 14 * 2**30
            and budget['free_bytes_at_prepare'] >= budget['initial_required_free_bytes'], 'Storage forecast or reserve changed')
    require(sha(run / 'protocol.md') == sha(ROOT / 'docs/clpu_integrated_pilot_protocol_v1.md'), 'Frozen prose protocol differs')
    return cfg, scope, schedule


def audit(run, output=None):
    run = Path(run).resolve()
    output = Path(output).resolve() if output else run / 'audit'
    require(output.is_relative_to(ROOT) and output != run and not output.exists(), 'Preserve previous audit evidence')
    output.mkdir(parents=True)
    started, checks = time.perf_counter(), []
    result = {'status': 'FAIL', 'source_run': run.name, 'errors': [], 'audited_cells': 0,
        'results': checks, 'baseline_equality': [], 'new_audit_physical_points': 0, 'new_training_tasks': 0}
    from audit_d059_supplement import NoSolverImports, no_solver_loaded
    guard = NoSolverImports()
    sys.meta_path.insert(0, guard)
    try:
        cfg, scope, schedule = check_parent(run)
        for spec in schedule:
            checked = check_cell(run, spec, cfg, scope)
            checks.append(checked)
            result['audited_cells'] = len(checks)
            save(output / 'progress.json', {'audited_cells': len(checks), 'scheduled': 64,
                                          'elapsed_seconds': time.perf_counter() - started})
        controller = read(run / 'controller_completion.json')
        require(read(run / 'controller_exits.json') == controller['exits']
                and controller['dispatched'] == schedule, 'Controller exit or dispatch journal differs')
        logical, physical, entries = check_controller(controller, schedule, checks,
                                                       list(read_lines(run / 'controller_dispatch.jsonl')))
        for context in range(8):
            rows = [r for r in checks if r['context_id'] == context and r['arm'] != 'window_reference']
            require(len(rows) == 7 and len({r['baseline_signature_sha256'] for r in rows}) == 1,
                    'Independent cold Beam baselines do not coincide')
            result['baseline_equality'].append({'context_id': context, 'cold_paid_baselines': 7,
                'baseline_signature_sha256': rows[0]['baseline_signature_sha256'], 'status': 'PASS'})
        for spec, checked in zip(schedule, checks):
            save(output / f'cell_{spec["episode_id"]:03d}.json', checked)
        no_solver_loaded()
        result.update(status='PASS', logical_points=logical, physical_points=physical, numerical_entries=entries,
            physical_statuses=dict(sum((Counter(r['physical_statuses']) for r in checks), Counter())),
            freeze_sha256=sha(run / 'prephysical_freeze_manifest.json'), schedule_sha256=sha(run / 'schedule.json'),
            source_binding_sha256=sha(run / 'source_binding.json'),
            controller_completion_sha256=sha(run / 'controller_completion.json'),
            controller_dispatch_sha256=sha(run / 'controller_dispatch.jsonl'),
            physical_scope_sha256=digest(scope), model_bindings_sha256=sha(run / 'model_bindings.json'),
            model_preflight_manifest_sha256=sha(run / 'model_preflight/completion_manifest.json'),
            scope_review_manifest_sha256=sha(run / 'scope_review/completion_manifest.json'),
            exact_key_review_manifest_sha256=sha(run / 'exact_key_review/completion_manifest.json'),
            source_window_completion_sha256=cache.SOURCE_SHA, source_cache_completion_sha256=CACHE_SHA,
            source_tail_completion_sha256=TAIL_SHA,
            interpretation='Eight previously observed training contexts; same-source BC/PPO, not equal total training data/cost; no holdout generalization claim.')
    except BaseException as exc:
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
    result = {'status': 'FAIL', 'run': run.name, 'errors': [], 'new_physical_points': 0, 'new_training': 0}
    guard = NoSolverImports()
    sys.meta_path.insert(0, guard)
    try:
        _, scope, schedule = check_parent(run)
        no_solver_loaded()
        result.update(status='PASS', scheduled_cells=len(schedule), contexts=len({s['context_id'] for s in schedule}),
            freeze_sha256=sha(run / 'prephysical_freeze_manifest.json'), schedule_sha256=sha(run / 'schedule.json'),
            source_binding_sha256=sha(run / 'source_binding.json'), physical_scope_sha256=digest(scope),
            model_bindings_sha256=sha(run / 'model_bindings.json'),
            model_preflight_manifest_sha256=sha(run / 'model_preflight/completion_manifest.json'),
            scope_review_manifest_sha256=sha(run / 'scope_review/completion_manifest.json'),
            exact_key_review_manifest_sha256=sha(run / 'exact_key_review/completion_manifest.json'),
            no_solver_imported=True)
    except BaseException as exc:
        result['errors'].append({'error': repr(exc), 'traceback': traceback.format_exc()})
    finally:
        sys.meta_path.remove(guard)
    save(output / 'summary.json', result)
    save(output / 'completion_manifest.json', {'summary.json': sha(output / 'summary.json')})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--model-preflight', action='store_true')
    args = parser.parse_args()
    if args.model_preflight:
        require(args.run is None and args.output is not None and not args.preflight, 'Explicit model preflight output required')
        outcome = source_model_preflight(args.output)
    elif args.preflight:
        require(args.run is not None and args.output is not None, 'Explicit independent preflight paths required')
        outcome = preflight(args.run, args.output)
    else:
        require(args.run is not None, 'Explicit completed run required')
        outcome = audit(args.run, args.output)
    print(canonical({key: value for key, value in outcome.items() if key not in ('results', 'models')}), flush=True)
    raise SystemExit(0 if outcome['status'] == 'PASS' else 1)
