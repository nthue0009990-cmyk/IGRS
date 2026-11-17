"""Lightweight authorization gates; no actual run/parameter/result access."""
from copy import deepcopy
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_clpu_public_confirmation_posthoc_audit_v2 as wrapper


def authorization():
    return {'schema':'clpu-postexec-explicit-user-authorization-v2',
        'status':'USER_AUTHORIZED_METADATA_ONLY_SAVED_EVIDENCE_REPLAY','run':wrapper.adapter.TARGET_RUN,
        'user_text':wrapper.USER_TEXT,'utc':'2026-10-31T16:00:00+00:00','scope':deepcopy(wrapper.SCOPE),
        'proposal_path':wrapper.PROPOSAL,'proposal_manifest_sha256':wrapper.PROPOSAL_SHA,
        'original_audit_summary_sha256':'a'*64,'original_audit_manifest_sha256':'b'*64,
        'adapter_sha256':wrapper.ADAPTER_SHA,'adapter_tests_sha256':wrapper.ADAPTER_TESTS_SHA,
        'wrapper_sha256':'c'*64,'method_freeze_sha256':wrapper.adapter.FROZEN_METHOD_SHA,
        'physical_freeze_sha256':wrapper.adapter.FROZEN_PHYSICAL_SHA}


@pytest.mark.parametrize('change',[None,'wording','experiment','training','proposal','scope','missing','timestamp'])
def test_explicit_user_scope_and_immutable_proposal_required(change):
    value=authorization()
    if change=='wording':value['user_text']='vague continuation'
    elif change=='experiment':value['scope']['rerun_experiments']=True
    elif change=='training':value['scope']['training']=True
    elif change=='proposal':value['proposal_manifest_sha256']='0'*64
    elif change=='scope':value['scope']['saved_cells']=47
    elif change=='missing':value.pop('original_audit_summary_sha256')
    elif change=='timestamp':value['utc']='2026-10-31T16:00:00'
    if change:
        with pytest.raises(AssertionError):wrapper.check_authorization(value)
    else:wrapper.check_authorization(value)


def test_output_and_original_names_are_distinct():
    assert wrapper.OUTPUT=='audit_posthoc_v2'
    assert wrapper.AUTH_FILE=='posthoc_v2_user_authorization.json'
    assert wrapper.SCOPE['preserve_original_failed_audit'] is True
    assert wrapper.SCOPE['new_physical_points']==0
