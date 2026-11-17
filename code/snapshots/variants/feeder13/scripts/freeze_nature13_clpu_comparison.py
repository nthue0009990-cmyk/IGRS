"""Freeze the accepted 18-task scientific design; never construct a DSS runtime.

The cold-process/shared-oracle execution bridge is a separately reviewed gate.
This program exposes no execute command and cannot start physical experiments.
"""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
sys.dont_write_bytecode = True
ACCEPTANCE = ROOT / 'runs/20261101T052022_022941Z_nature13_clpu_acceptance'
AUDIT = ROOT / 'artifacts/nature13_clpu_acceptance_resume_20261105/saved_run_audit.json'
PROTOCOL = ROOT / 'docs/nature13_clpu_comparison_protocol_v1.md'
STATIC = ROOT / 'artifacts/nature13_clpu_adaptation_20261101/static'
ACTIVE = ROOT / 'artifacts/nature13_clpu_comparison_design_latest.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def prepare():
    from run_nature13_clpu_acceptance import verify, environment
    from prepare_nature13_clpu_static import load_inputs, make_env
    from dnr.nature13_clpu_branch import BranchConfig
    from dnr.nature13_clpu_staged_pickup import generate_staged_pickup_candidates

    verify(ACCEPTANCE)
    audit = read(AUDIT)
    if audit['status'] != 'VERIFIED_SAVED_ACCEPTANCE' or audit['acceptance_passed'] is not True:
        raise ValueError('Verified acceptance required')
    if audit['execution_manifest_sha256'] != sha(ACCEPTANCE / 'execution_manifest.json'):
        raise ValueError('Acceptance execution binding differs')
    for name, digest in read(ACCEPTANCE / 'execution_manifest.json').items():
        if sha(ACCEPTANCE / name) != digest:
            raise ValueError('Acceptance changed: ' + name)
    if ACTIVE.exists():
        raise FileExistsError('Existing scientific freeze must be inspected, not silently replaced')

    static, loads, condition, scenarios, events, _ = load_inputs()
    contexts = []
    for scenario in scenarios:
        for event in sorted(events, key=lambda r: r['event_id']):
            env = make_env(static, loads, condition, scenario, event)
            pool = generate_staged_pickup_candidates(env)
            cid = scenario['scenario_id'] + '_e' + str(event['event_id'])
            original = read(STATIC / 'contexts' / (cid + '.json'))
            if pool != original['candidate_generation'] or pool['counts']['unique_legal_candidates'] != 12:
                raise ValueError('Original frozen candidate pool differs: ' + cid)
            contexts.append(original)
    if len(contexts) != 6:
        raise ValueError('All six fixed development contexts required')
    arms = ('P', 'E', 'Forward')
    schedule = []
    for turn in range(3):
        for rank, ctx in enumerate(contexts):
            arm = arms[(rank + turn) % 3]
            schedule.append({'task_id': len(schedule), 'context_id': ctx['context_id'],
                'scenario_id': ctx['scenario_id'], 'event_id': ctx['event_id'], 'arm': arm,
                'phase_order': ['EARLY', 'GUARD'] if arm == 'E' else ['GUARD', 'EARLY'],
                'round': turn, 'replicate': 0, 'seed': 0, 'status': 'NOT_RUN',
                'candidate_context_file': 'contexts/' + ctx['context_id'] + '.json'})
    expected = {(c['context_id'], arm) for c in contexts for arm in arms}
    if len(schedule) != 18 or {(s['context_id'], s['arm']) for s in schedule} != expected:
        raise ValueError('Unbalanced fixed schedule')
    if any(n == 'dss' or n.startswith(('dss.', 'opendssdirect', 'Environments')) for n in sys.modules):
        raise RuntimeError('Scientific freeze must remain solver-free')

    run = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_nature13_clpu_comparison_design')
    run.mkdir(exist_ok=False)
    settings = {'study': 'NC13_PUBLIC_COMPARISON_DESIGN_V1',
        'status': 'FROZEN_RESEARCH_DESIGN_NOT_EXECUTED', 'execution_authorized': False,
        'interface_entry_acceptance_passed': True, 'cold_task_execution_bridge_ready': False,
        'tasks': 18, 'contexts': 6, 'arms': list(arms), 'partition': 'development_transfer',
        'formal_test': False, 'training_steps': 0, 'models_used': 0, 'replicates': 1,
        'alpha': .375, 'algorithm': 0, 'horizon': 30, 'slot_seconds': 60, 'sample_seconds': 30,
        'logical_points_per_phase': 256, 'physical_queries_per_task': 512,
        'early_verify_limit': 16, 'workers': 1, 'retry': False,
        'initialization_numerical_entries': 3, 'complete_query_numerical_entries': 14,
        'numerical_entries_per_task': 7171, 'numerical_entries_total': 129078,
        'task_seconds': 240, 'controller_seconds': 3600,
        'minimum_free_bytes': 2 * 2**30,
        'cross_task_evidence_reuse': False, 'same_task_cross_phase_reuse': True,
        'time_accounting': 'child process start through exit; initialization, generation, logs included',
        'new_physical_queries_in_this_freeze': 0,
        'pending_execution_gates': ['NC13 shared evidence bridge', 'cold controller and complete-cost journal',
                                    'independent saved-result auditor', 'pure integration checks', 'execution authorization']}
    statistics = {'primary_comparator': 'E', 'strong_component_comparator': 'Forward',
        'weights': 'all six fixed contexts equal', 'quality': '100 * normal_energy_kwh / (3466 * 0.5)',
        'quality_equality': 'exact full-precision energy equality, checked per paired context',
        'point_saving': '1 - sum(P_actual_queries) / sum(comparator_actual_queries)',
        'time_saving': '1 - sum(P_full_cold_seconds) / sum(comparator_full_cold_seconds)',
        'same_quality_cost_claim_requires': 'all six paired energy values equal and complete paired evidence',
        'mixed_quality': 'report quality-cost tradeoff; do not aggregate as same-quality savings',
        'missing_or_failed_task': 'retain all 18 statuses; six-context aggregate is null if incomplete',
        'inference': 'descriptive paired results only; no significance test or unseen-generalization claim',
        'required_outputs': ['per-context quality, guard energy and final actions', 'actual and logical points',
            'numerical entries', 'complete cold time', 'failures and unknowns', 'wins/ties/losses and worst case'],
        'stopping': 'unknown or reliability defect stops affected work; no replacement or retuning'}
    for name, value in [('settings.json', settings), ('schedule.json', schedule),
            ('statistics.json', statistics), ('branch_config.json', asdict(BranchConfig())),
            ('author_static_metadata.json', static), ('load_reference.json', loads),
            ('condition.json', condition), ('scenarios.json', scenarios), ('selected_events.json', events),
            ('environment.json', environment()), ('physical_config.json', read(ACCEPTANCE / 'physical_config.json'))]:
        save(run / name, value)
    for ctx in contexts:
        save(run / 'contexts' / (ctx['context_id'] + '.json'), ctx)
    shutil.copyfile(PROTOCOL, run / 'protocol.md')
    frozen = read(ACCEPTANCE / 'prepared.json')
    source_hashes = dict(frozen['source_hashes'])
    source_hashes[Path(__file__).relative_to(ROOT).as_posix()] = sha(Path(__file__))
    source_hashes[PROTOCOL.relative_to(ROOT).as_posix()] = sha(PROTOCOL)
    for name, digest in source_hashes.items():
        source = ROOT / name
        if sha(source) != digest:
            raise ValueError('Source mismatch while freezing: ' + name)
        target = run / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for original, relative in frozen['project_input_copies'].items():
        source = ACCEPTANCE / relative
        target = run / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    save(run / 'source_binding.json', {'source_hashes': source_hashes,
        'original_input_hashes': frozen['project_input_hashes'],
        'input_copies': frozen['project_input_copies'],
        'acceptance_run': ACCEPTANCE.name, 'acceptance_audit_sha256': sha(AUDIT),
        'acceptance_execution_sha256': sha(ACCEPTANCE / 'execution_manifest.json'),
        'physical_answers_copied_into_design': False})
    save(run / 'freeze_manifest.json', {p.relative_to(run).as_posix(): sha(p)
        for p in sorted(run.rglob('*')) if p.is_file()})
    save(ACTIVE, {'run': run.name, 'status': settings['status'], 'tasks': 18,
        'execution_authorized': False, 'cold_task_execution_bridge_ready': False,
        'freeze_manifest_sha256': sha(run / 'freeze_manifest.json'), 'new_physical_queries': 0})
    return run


if __name__ == '__main__':
    print(json.dumps({'run': str(prepare()), 'physical_queries': 0, 'execution_started': False}))
