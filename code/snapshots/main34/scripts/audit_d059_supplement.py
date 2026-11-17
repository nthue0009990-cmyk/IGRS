"""Durable, solver-free supplementary verification of the sealed D059 logs.

The original scientific auditor is called unchanged. This module adds only
input binding, atomic per-cell evidence, interruption recovery and a deadline.
The caller owns the process-level deadline; a killed in-flight cell is never
counted as checked. A subsequent explicit invocation reuses only sealed cells.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.abc
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
sys.dont_write_bytecode = True

import audit_nature34_clpu_branch as original

PINNED_SOURCE_COMPLETION_SHA256 = '4c3777f938d21f074b40db56c82cd62ca1b956f6aa7d79215832bf192745c56f'
EXPECTED_EVALUATION_CELLS = 272
FORBIDDEN_MODULES = frozenset(('opendssdirect', 'dss', 'altdss', 'py_dss_interface'))


class AuditDeadlineReached(RuntimeError):
    pass


class NoSolverImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FORBIDDEN_MODULES:
            raise RuntimeError('Supplementary audit attempted a physical-runtime import: ' + fullname)
        return None


def no_solver_loaded():
    original.require(not (FORBIDDEN_MODULES & {n.split('.')[0] for n in sys.modules}),
                     'Supplementary audit initialized physical runtime')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_save(path, value):
    """A partial JSON is never promoted to a completed evidence record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def check_deadline(deadline_epoch):
    if time.time() >= deadline_epoch:
        raise AuditDeadlineReached('Approved supplementary audit deadline reached')


def inside(base, relative):
    original.require(isinstance(relative, str) and relative, 'Empty/non-string manifest path')
    candidate = (base / relative.replace('\\', '/')).resolve()
    original.require(candidate != base and candidate.is_relative_to(base), 'Manifest path escapes declared run')
    return candidate


def normalized_manifest(manifest):
    answer = {}
    for name, digest in manifest.items():
        normalized = name.replace('\\', '/')
        original.require(normalized not in answer, 'Duplicate normalized manifest entry')
        original.require(isinstance(digest, str) and len(digest) == 64, 'Invalid manifest digest')
        answer[normalized] = digest
    return answer


def sealed_hash(run, relative, source_manifest):
    normalized = relative.replace('\\', '/')
    original.require(normalized in source_manifest, 'Unsealed source input: ' + normalized)
    path = inside(run, normalized)
    digest = sha(path)
    original.require(digest == source_manifest[normalized], 'Changed source input: ' + normalized)
    return digest


def auditor_sources():
    return (Path(original.__file__), Path(original.old.__file__),
            ROOT / 'scripts/audit_nature34_joint_planning.py',
            ROOT / 'scripts/audit_nature34_clpu_branch_learning.py', Path(__file__))


def prepare_inputs(source_run):
    """Retain the entire original parent and independent population checks."""
    original.require(source_run.parent == ROOT / 'runs', 'Source run outside project')
    completion_sha = sha(source_run / 'completion_manifest.json')
    original.require(completion_sha == PINNED_SOURCE_COMPLETION_SHA256,
                     'Source completion manifest differs from approved proposal')
    manifest = normalized_manifest(read(source_run / 'completion_manifest.json'))
    cfg = original.check_parent(source_run)
    requested_relative = 'audit_batches/evaluation/cells.json'
    requested_sha = sealed_hash(source_run, requested_relative, manifest)
    values = read(source_run / requested_relative)
    cells = []
    for value in values:
        path = Path(value)
        cell = path.resolve() if path.is_absolute() else (source_run / path).resolve()
        original.require(cell != source_run and cell.is_relative_to(source_run), 'Audit cell outside source run')
        cells.append(cell)
    original.require(len(cells) == EXPECTED_EVALUATION_CELLS and len(set(cells)) == len(cells),
                     'Empty, duplicated or incomplete evaluation audit population')
    schedule = read(source_run / 'schedule.json')
    scheduled = {(source_run / row['cell']).resolve() for row in schedule if row['group'] == 'evaluation'}
    original.require(set(cells) == scheduled, 'Audit batch omits/adds scheduled evaluation cells')
    no_solver_loaded()
    identity = {
        'source_run': str(source_run), 'source_completion_sha256': completion_sha,
        'phase': 'evaluation', 'original_requested_cells_sha256': requested_sha,
        'requested_cells': [p.relative_to(source_run).as_posix() for p in cells],
        'freeze_sha256': sealed_hash(source_run, 'prephysical_freeze_manifest.json', manifest),
        'dependencies_sha256': sealed_hash(source_run, 'dependencies.json', manifest),
        'schedule_sha256': sealed_hash(source_run, 'schedule.json', manifest),
        'python_sha256': sha(Path(sys.executable)),
        'source_hashes': {p.relative_to(ROOT).as_posix(): sha(p) for p in auditor_sources()},
        'scientific_check_implementation': 'unchanged audit_nature34_clpu_branch.check_parent/check_cell',
        'prior_unsealed_progress_reused': False,
    }
    return cfg, cells, manifest, identity


def cell_inputs(source_run, cell, source_manifest):
    """Bind the independent cell seal to the sealed original run, not a new seal."""
    relative = cell.relative_to(source_run).as_posix()
    seal_name = relative + '/completion_manifest.json'
    seal_sha = sealed_hash(source_run, seal_name, source_manifest)
    local_manifest = normalized_manifest(read(cell / 'completion_manifest.json'))
    bindings = {seal_name: seal_sha}
    for name, digest in local_manifest.items():
        inside(cell, name)
        full_name = relative + '/' + name
        original.require(source_manifest.get(full_name) == digest,
                         'Cell seal disagrees with original run seal: ' + full_name)
        bindings[full_name] = digest
    # Learned endpoints are outside the cell and must also be pinned to the
    # immutable original run. The original check_cell validates their policy.
    sealed_hash(source_run, relative + '/spec.json', source_manifest)
    spec = read(cell / 'spec.json')
    if spec.get('checkpoint'):
        checkpoint = inside(source_run, spec['checkpoint'])
        checkpoint_name = checkpoint.relative_to(source_run).as_posix()
        original.require(source_manifest.get(checkpoint_name) == spec['checkpoint_sha256'],
                         'Checkpoint differs from original run seal')
        bindings[checkpoint_name] = spec['checkpoint_sha256']
    return bindings


def verify_bindings(source_run, bindings):
    for name, digest in bindings.items():
        original.require(sha(inside(source_run, name)) == digest, 'Changed audited input: ' + name)


def record_directory(output, index, cell):
    return output / 'per_cell' / f'{index:04d}_{cell.name}'


def load_completed(directory, source_run, index, cell, bindings, identity_sha):
    seal = read(directory / 'seal.json')
    original.require(set(seal) == {'record.json'}, 'Unexpected supplementary cell seal')
    original.require(sha(directory / 'record.json') == seal['record.json'], 'Supplementary result was changed')
    record = read(directory / 'record.json')
    original.require(record['status'] == 'VERIFIED_CLPU_BRANCH_CELL' and record['errors'] == [],
                     'Previous supplementary cell was not verified')
    original.require(record['index'] == index and record['cell'] == cell.relative_to(source_run).as_posix()
                     and record['audit_identity_sha256'] == identity_sha and record['input_sha256'] == bindings,
                     'Supplementary cell order or source binding changed')
    verify_bindings(source_run, bindings)
    return record['check']


def run_audit(source_run: Path, supplement_run: Path, deadline_epoch: float) -> dict:
    source_run, output = Path(source_run).resolve(), Path(supplement_run).resolve()
    original.require(output != source_run and not output.is_relative_to(source_run)
                     and not source_run.is_relative_to(output) and output.is_relative_to(ROOT / 'runs'),
                     'Supplement must be a separate directory under project runs')
    original.require(type(deadline_epoch) in (int, float) and math.isfinite(deadline_epoch), 'Invalid deadline')
    output.mkdir(parents=True, exist_ok=True)
    invocations = output / 'invocations'
    invocations.mkdir(exist_ok=True)
    invocation = invocations / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_' + uuid.uuid4().hex[:8])
    invocation.mkdir()
    began = time.perf_counter()
    result = {'status': 'FAILED_CLPU_BRANCH_CELLS', 'source_run': source_run.name,
              'phase': 'evaluation', 'audit_run': str(output), 'errors': [],
              'new_powerflow_calls': 0, 'new_training_interactions': 0, 'optimizer_updates': 0,
              'formal_holdout_numerical_access': False, 'tune_numerical_access': False,
              'deadline_epoch': deadline_epoch, 'completed': 0, 'cells': EXPECTED_EVALUATION_CELLS}
    checks, reused, guard = [], 0, NoSolverImports()
    atomic_save(invocation / 'started.json', {'started_utc': utc_now(), 'deadline_epoch': deadline_epoch,
                                             'source_run': str(source_run)})
    sys.meta_path.insert(0, guard)
    try:
        no_solver_loaded()
        check_deadline(deadline_epoch)
        # A reliability failure must never be silently retried. A process kill
        # or deadline stop is distinguishable from such a completed failure.
        for old_result in invocations.glob('*/result.json'):
            original.require(read(old_result)['status'] != 'FAILED_CLPU_BRANCH_CELLS',
                             'Earlier reliability failure retained; no automatic retry')
        cfg, cells, manifest, identity = prepare_inputs(source_run)
        identity['deadline_epoch'] = deadline_epoch
        identity_path = output / 'audit_identity.json'
        if identity_path.exists():
            original.require(read(identity_path) == identity, 'Audit identity or approved deadline changed')
        else:
            atomic_save(identity_path, identity)
        identity_sha = sha(identity_path)
        result.update({key: identity[key] for key in ('source_completion_sha256', 'freeze_sha256',
                      'dependencies_sha256', 'schedule_sha256', 'python_sha256')})
        atomic_save(output / 'requested_cells.json', identity['requested_cells'])
        expected_directories = {record_directory(output, i, cell).name for i, cell in enumerate(cells)}
        if (output / 'per_cell').exists():
            original.require(all(p.is_dir() and p.name in expected_directories for p in (output / 'per_cell').iterdir()),
                             'Unexpected supplementary audit population')
        prefix_complete = True
        for index, cell in enumerate(cells):
            check_deadline(deadline_epoch)
            directory = record_directory(output, index, cell)
            bindings = cell_inputs(source_run, cell, manifest)
            if (directory / 'seal.json').exists():
                original.require(prefix_complete, 'Supplementary sealed cells are not a chronological prefix')
                checked = load_completed(directory, source_run, index, cell, bindings, identity_sha)
                reused += 1
            else:
                prefix_complete = False
                # Preserve an unsealed result after interruption. It does not
                # count as passed and cannot be promoted without fresh replay.
                if directory.exists():
                    preserved = directory / ('unsealed_' + uuid.uuid4().hex)
                    preserved.mkdir()
                    for path in list(directory.iterdir()):
                        if path.is_file():
                            os.replace(path, preserved / path.name)
                started = time.perf_counter()
                checked = original.check_cell(source_run, cell, cfg)
                no_solver_loaded()
                verify_bindings(source_run, bindings)
                original.require(checked['cell_completion_sha256'] == bindings[cell.relative_to(source_run).as_posix()
                                 + '/completion_manifest.json'], 'Returned check is bound to a different cell seal')
                record = {'status': 'VERIFIED_CLPU_BRANCH_CELL', 'errors': [], 'index': index,
                          'cell': cell.relative_to(source_run).as_posix(), 'audit_identity_sha256': identity_sha,
                          'input_sha256': bindings, 'check': checked, 'completed_utc': utc_now(),
                          'elapsed_seconds': time.perf_counter() - started,
                          'new_powerflow_calls': 0, 'new_training_interactions': 0, 'optimizer_updates': 0}
                atomic_save(directory / 'record.json', record)
                atomic_save(directory / 'seal.json', {'record.json': sha(directory / 'record.json')})
            checks.append(checked)
            result['completed'] = len(checks)
            atomic_save(output / 'progress.json', {'status': 'AUDITING_EXISTING_LOGS', 'completed': len(checks),
                         'scheduled': len(cells), 'errors': [], 'last_completed_cell': cell.relative_to(source_run).as_posix(),
                         'last_cell_seal_sha256': sha(directory / 'seal.json'), 'reused_sealed_cells': reused,
                         'elapsed_seconds_this_invocation': time.perf_counter() - began, 'updated_utc': utc_now()})
        check_deadline(deadline_epoch)
        no_solver_loaded()
        original.require(sha(source_run / 'completion_manifest.json') == identity['source_completion_sha256'],
                         'Original source completion manifest changed during audit')
        # These aggregate files are produced only after the full ordered
        # population has independently passed. No partial VERIFIED marker.
        atomic_save(output / 'cell_checks.json', checks)
        result.update(status='VERIFIED_CLPU_BRANCH_CELLS', cells=len(checks), episodes=len(checks),
                      physical_points=sum(c['points'] for c in checks),
                      numerical_entries=sum(c['commands']['numerical_entries'] for c in checks),
                      physical_statuses=dict(sum((Counter(c['physical_statuses']) for c in checks), Counter())),
                      cell_checks_sha256=sha(output / 'cell_checks.json'), audit_identity_sha256=identity_sha)
    except BaseException as exc:
        if isinstance(exc, (AuditDeadlineReached, KeyboardInterrupt)):
            result['status'] = 'STOPPED_CLPU_BRANCH_CELLS'
        result['errors'].append({'error': repr(exc), 'traceback': traceback.format_exc()})
    finally:
        sys.meta_path.remove(guard)
    result.update(elapsed_seconds=time.perf_counter() - began, reused_sealed_cells=reused,
                  completed_utc=utc_now(), cell_checks_complete=result['status'] == 'VERIFIED_CLPU_BRANCH_CELLS')
    atomic_save(invocation / 'result.json', result)
    atomic_save(output / 'verification.json', result)
    if result['status'] == 'VERIFIED_CLPU_BRANCH_CELLS':
        roots = ('audit_identity.json', 'requested_cells.json', 'progress.json', 'verification.json',
                 'cell_checks.json', 'per_cell', 'invocations')
        evidence = []
        for name in roots:
            item = output / name
            evidence.extend(item.rglob('*') if item.is_dir() else [item])
        sealed_files = {p.relative_to(output).as_posix(): sha(p) for p in evidence
                        if p.is_file() and not p.name.endswith('.tmp')}
        atomic_save(output / 'audit_completion_manifest.json', sealed_files)
        result['completion_sha256'] = sha(output / 'audit_completion_manifest.json')
        result['audit_completion_manifest'] = 'audit_completion_manifest.json'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--supplement-run', type=Path, required=True)
    parser.add_argument('--deadline-epoch', type=float, required=True)
    args = parser.parse_args()
    result = run_audit(args.source_run, args.supplement_run, args.deadline_epoch)
    print(json.dumps(result, ensure_ascii=True, allow_nan=False), flush=True)
    return 0 if result['status'] == 'VERIFIED_CLPU_BRANCH_CELLS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
