"""Authorized zero-PF application of the sealed post-execution audit adapter.

Original frozen files and original audit/FAIL remain untouched. This wrapper
records the user's approval, then uses the original orchestration with only
the sealed formal-identity graph adapter and explicit posthoc metadata.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import audit_clpu_public_confirmation_posthoc_v2 as adapter

original=adapter.original
ROOT=original.ROOT
require,read,save,sha=original.require,original.read,original.save,original.sha
PROPOSAL='artifacts/public_confirmation_posthoc_v2_proposal_20261031'
PROPOSAL_SHA='8725575a9f0b357d7e1f6a83a54b79bf19b1f58346091165fd39b0ec62884e20'
ADAPTER_SHA='7050025ace22991fc60a46f21a16f762eb12cae4ca3e0cc62f98cbe4676a69de'
ADAPTER_TESTS_SHA='9ffe2a72babce35cf533a4beb26012a2b60d1ecaade57291b38f38b7817435f3'
AUTH_FILE='posthoc_v2_user_authorization.json'
OUTPUT='audit_posthoc_v2'
USER_TEXT='行，应用修复，继续核验这48项记录，保留原失败记录，不改方法、参数和统计规则，不重跑实验或训练。核验通过后生成结果图，尽快告诉我P相对E到底有没有优势，以及离SCI二、三区的目标还差什么。'
SCOPE={'saved_cells':48,'metadata_admission_adapter_only':True,'preserve_original_failed_audit':True,
    'change_method':False,'change_parameters':False,'change_statistics':False,
    'rerun_experiments':False,'new_physical_points':0,'training':False,
    'new_output_directory':OUTPUT,'generate_figures_only_after_complete_audit_PASS':True}


def check_authorization(value):
    expected={'schema','status','run','user_text','utc','scope','proposal_path','proposal_manifest_sha256',
        'original_audit_summary_sha256','original_audit_manifest_sha256','adapter_sha256','adapter_tests_sha256',
        'wrapper_sha256','method_freeze_sha256','physical_freeze_sha256'}
    require(set(value)==expected and value['schema']=='clpu-postexec-explicit-user-authorization-v2'
        and value['status']=='USER_AUTHORIZED_METADATA_ONLY_SAVED_EVIDENCE_REPLAY'
        and value['run']==adapter.TARGET_RUN and value['user_text']==USER_TEXT and value['scope']==SCOPE,
        'Missing or expanded explicit user authorization')
    require(value['proposal_path']==PROPOSAL and value['proposal_manifest_sha256']==PROPOSAL_SHA
        and value['adapter_sha256']==ADAPTER_SHA and value['adapter_tests_sha256']==ADAPTER_TESTS_SHA
        and value['method_freeze_sha256']==adapter.FROZEN_METHOD_SHA
        and value['physical_freeze_sha256']==adapter.FROZEN_PHYSICAL_SHA,'Authorization refers to another adaptation/freeze')
    require(datetime.fromisoformat(value['utc']).tzinfo is not None,'Authorization UTC timestamp missing')


def check_proposal_and_original_failure(run):
    run=Path(run).resolve()
    require(run==ROOT/'runs'/adapter.TARGET_RUN,'Posthoc wrapper is restricted to the authorized existing run')
    proposal=ROOT/PROPOSAL
    require(sha(proposal/'completion_manifest.json')==PROPOSAL_SHA,'Reviewed proposal root changed')
    original.shared.checked_manifest(proposal)
    reviewed=read(proposal/'summary.json')
    require(reviewed['status']=='READY_FOR_USER_REVIEW_NOT_APPLIED' and reviewed['synthetic_tests_passed']==15
        and reviewed['actual_48_cell_replay_started'] is False,'Reviewed proposal/test scope differs')
    require(sha(Path(adapter.__file__))==ADAPTER_SHA
        and sha(ROOT/'tests/test_audit_clpu_public_confirmation_posthoc_v2.py')==ADAPTER_TESTS_SHA,
        'Reviewed immutable adapter/test bytes changed')
    original.shared.checked_manifest(run/'audit')
    require(sha(run/'audit/summary.json')==reviewed['original_audit_failure_sha256']
        and sha(run/'audit/completion_manifest.json')==reviewed['original_audit_failure_manifest_sha256'],
        'Original failed audit changed or was overwritten')
    return reviewed


def prepare_authorization(run):
    run=Path(run).resolve();reviewed=check_proposal_and_original_failure(run)
    require(not (run/AUTH_FILE).exists() and not (run/OUTPUT).exists(),'Preserve previous authorization/replay; no implicit retry')
    value={'schema':'clpu-postexec-explicit-user-authorization-v2',
        'status':'USER_AUTHORIZED_METADATA_ONLY_SAVED_EVIDENCE_REPLAY','run':adapter.TARGET_RUN,
        'user_text':USER_TEXT,'utc':datetime.now(timezone.utc).isoformat(),'scope':deepcopy(SCOPE),
        'proposal_path':PROPOSAL,'proposal_manifest_sha256':PROPOSAL_SHA,
        'original_audit_summary_sha256':reviewed['original_audit_failure_sha256'],
        'original_audit_manifest_sha256':reviewed['original_audit_failure_manifest_sha256'],
        'adapter_sha256':ADAPTER_SHA,'adapter_tests_sha256':ADAPTER_TESTS_SHA,'wrapper_sha256':sha(Path(__file__)),
        'method_freeze_sha256':adapter.FROZEN_METHOD_SHA,'physical_freeze_sha256':adapter.FROZEN_PHYSICAL_SHA}
    check_authorization(value);save(run/AUTH_FILE,value)
    return {'status':value['status'],'authorization_file':str(run/AUTH_FILE),'authorization_sha256':sha(run/AUTH_FILE)}


def audit(run):
    from audit_d059_supplement import NoSolverImports,no_solver_loaded
    run=Path(run).resolve();reviewed=check_proposal_and_original_failure(run)
    value=read(run/AUTH_FILE);check_authorization(value)
    require(value['wrapper_sha256']==sha(Path(__file__))
        and value['original_audit_summary_sha256']==reviewed['original_audit_failure_sha256']
        and value['original_audit_manifest_sha256']==reviewed['original_audit_failure_manifest_sha256'],
        'Authorized wrapper or original failed-audit binding changed')
    require(not (run/OUTPUT).exists(),'Previous posthoc replay exists; no overwrite or automatic retry')
    guard=NoSolverImports();sys.meta_path.insert(0,guard)
    try:
        no_solver_loaded();bound=adapter.bind_frozen_run(run)
        metadata=deepcopy(bound.posthoc_adapter)
        metadata.update(actual_cell_replay_started=True,user_authorization_file=AUTH_FILE,
            user_authorization_sha256=sha(run/AUTH_FILE),proposal_path=PROPOSAL,proposal_manifest_sha256=PROPOSAL_SHA,
            wrapper_implementation_sha256=sha(Path(__file__)),
            wrapper_tests_sha256=sha(ROOT/'tests/test_run_clpu_public_confirmation_posthoc_audit_v2.py'),output_directory=OUTPUT)
        def source_hashes(path,scope):
            source=original.source_hashes(path,scope)
            for name in ('scripts/run_clpu_public_confirmation_posthoc_audit_v2.py',
                'tests/test_run_clpu_public_confirmation_posthoc_audit_v2.py'):
                target=path/OUTPUT/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ROOT/name,target)
            source['source_hashes'][AUTH_FILE]=sha(path/AUTH_FILE)
            source.update(schema='clpu-public-confirmation-postexec-audit-v2',
                audit_route='post_execution_metadata_adapter',original_audit_status='FAIL',
                user_authorization_path=AUTH_FILE,user_authorization_sha256=sha(path/AUTH_FILE),
                audit_implementation_role='frozen_base_orchestration_with_explicit_posthoc_adapter',
                posthoc_adapter=metadata,effective_audit_implementation={
                    'frozen_base_sha256':metadata['frozen_original_auditor_sha256'],
                    'frozen_graph_base_sha256':metadata['frozen_graph_auditor_sha256'],
                    'sealed_adapter_sha256':ADAPTER_SHA,'wrapper_sha256':sha(Path(__file__))})
            return source
        replay=original.private(original.audit,check_cell=bound.check_cell,source_hashes=source_hashes)
        result=replay(run,run/OUTPUT)
        no_solver_loaded()
        require(sha(run/'audit/summary.json')==reviewed['original_audit_failure_sha256']
            and sha(run/'audit/completion_manifest.json')==reviewed['original_audit_failure_manifest_sha256'],
            'Original failed audit was not preserved')
        return result
    finally:sys.meta_path.remove(guard)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('authorize','audit'));parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='authorize':print(prepare_authorization(args.run))
    else:
        result=audit(args.run)
        print({'status':result['status'],'audited_cells':result['audited_cells'],
            'output':str(args.run.resolve()/OUTPUT),'new_physical_points':0})
        raise SystemExit(0 if result['status']=='PASS' else 2)
