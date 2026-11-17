"""No-solver resource, causal-ledger and direct-Beam replay regression checks."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_clpu_direct_beam as runner
import audit_clpu_direct_beam as audit
import clpu_direct_beam_audit_helpers as helper
from dnr.nature34_clpu_branch import BranchConfig
from dnr.nature34_clpu_guard_first_control import GuardFirstEvidenceService
from dnr.nature34_clpu_environment import CLPUBudgetExhausted
from test_nature34_clpu_branch_audit import inputs
from test_clpu_shared_audit import make_service,material
from test_nature34_clpu_equivalent_cache import lines,save


def single_material(service,mech):
    facade=SimpleNamespace(**{**vars(audit.shared),'VIEWS':('GUARD',)})
    return audit.gf.private_function(material,audit=facade)(service,mech)


def test_resource_merge_preserves_science_and_all_original_contexts():
    old=asdict(BranchConfig());new=runner.beam_config()
    changed={k for k in old if old[k]!=new[k]}
    assert changed=={'point_budget','max_generated_candidates','max_checked_candidates','max_branch_expansions'}
    assert all(new[k]==2*old[k] for k in changed)
    contexts=runner.read(runner.SOURCE/'contexts.json');schedule=runner.schedule_for(contexts)
    assert len(schedule)==16 and len({s['cell'] for s in schedule})==16
    assert all({s['arm'] for s in schedule if s['context_id']==i}=={'P','BEAM'} for i in range(8))
    assert sum(s['arm']=='P' for s in schedule[::2])==4
    assert all(s['physical_point_budget']==s['total_logical_point_budget']==512 for s in schedule)
    assert all(s['checkpoint'] is None and s['seed']==0 for s in schedule)


def test_single_beam_view_uses_512_logical_points_without_an_extra_bootstrap(make_service):
    service,mech,_=make_service(failed=False,long=True)
    service.env.horizon=mech.horizon=513
    service=GuardFirstEvidenceService(service.parent,service.cell,service.env,service.scope,logical_budget=512)
    with service:
        with service.view('GUARD') as view:
            for i in range(512):view.certify(service.env.initial,60*i)
            with pytest.raises(CLPUBudgetExhausted):view.certify(service.env.initial,60*512)
            view.certify(service.env.initial,0)
    data=single_material(service,mech)
    checked,_=helper.check_service_ledger(**data,logical_budget=512,views_order=('GUARD',))
    assert checked['GUARD']['logical_points']==512 and checked['GUARD']['denied']==1
    assert service.physical_points==1 and len(service.views)==1
    plain=deepcopy(service.summary());plain.pop('guard_first_version');plain.pop('phase_order')
    helper.check_service_summaries(plain,{'GUARD':view.summary()},checked,service.scope,'shared',7,logical_budget=512,views_order=('GUARD',))
    with pytest.raises(AssertionError):helper.check_service_ledger(**data,logical_budget=256,views_order=('GUARD',))


@pytest.mark.parametrize('failed',[False,True])
def test_reliable_status_retained_in_single_view(make_service,failed):
    service,mech,_=make_service(failed=failed)
    service.__class__=GuardFirstEvidenceService;service.logical_budget=512
    with service:
        with service.view('GUARD') as view:
            for seconds in (0,30,60):view.certify(service.env.initial,seconds)
    data=single_material(service,mech)
    checked,_=helper.check_service_ledger(**data,logical_budget=512,views_order=('GUARD',))
    assert checked['GUARD']['statuses']=={'FAIL' if failed else 'PASS':3}
    changed=deepcopy(data);changed['events'][0]['logical_budget']=256
    with pytest.raises(AssertionError):helper.check_service_ledger(**changed,logical_budget=512,views_order=('GUARD',))


def test_actual_direct_beam_route_replays_independently_without_solver(make_service):
    service,mech,cfg=make_service(failed=False)
    service.__class__=GuardFirstEvidenceService;service.logical_budget=512
    cfg={**cfg,'point_budget':512,'max_checked_candidates':512,'max_branch_expansions':512,'max_generated_candidates':10000}
    spec={'cell':'task','arm':'BEAM','method':'beam','mode':'shared','point_budget':512}
    save(service.cell/'GUARD/spec.json',runner.phase_spec(spec,'GUARD'))
    with service:
        results,cps=runner.beam_route(service,service.env,BranchConfig(**cfg),service.cell,spec,time.perf_counter(),time.time()+60)
    data=single_material(service,mech)
    evidence,_=helper.check_service_ledger(**data,logical_budget=512,views_order=('GUARD',))
    p=results['GUARD'];attempts=lines(service.cell/'GUARD/plans.jsonl.gz');decisions=lines(service.cell/'GUARD/decisions.jsonl.gz')
    result=helper.check_phase_records(mech,cfg,'GUARD',p['spec'],p,attempts,decisions,evidence['GUARD'],data['condition_sha'],logical_budget=512,views_order=('GUARD',))
    assert result['fully_certified'] and len(cps)==len(result['boundaries'])+1
    changed=deepcopy(p);changed['quality_pct']+=1
    with pytest.raises(AssertionError):helper.check_phase_records(mech,cfg,'GUARD',p['spec'],changed,attempts,decisions,evidence['GUARD'],data['condition_sha'],logical_budget=512,views_order=('GUARD',))
