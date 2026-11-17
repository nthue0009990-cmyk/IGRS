"""Freeze, scheduling and counter integrity tests; no simulator is loaded."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_clpu_equivalent_cache_pilot as runner


def contexts():
    return [{'context_id':i,'scenario_id':f'fault{i}','event_id':(30,20,6,3)[i%4]} for i in range(8)]


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def test_balanced_pair_order_is_fixed_and_full_budget_is_not_reallocated():
    schedule=runner.schedule_for(contexts())
    assert schedule == runner.schedule_for(list(reversed(contexts())))
    assert len(schedule)==16 and sum(s['physical_point_budget'] for s in schedule)==4096
    assert len({s['cell'] for s in schedule})==16
    assert sum(s['arm']==runner.ARMS[0] and s['arm_launch_order']==0 for s in schedule)==4
    for rank in range(8):
        pair=schedule[2*rank:2*rank+2]
        assert {s['arm'] for s in pair}==set(runner.ARMS)
        assert len({s['context_id'] for s in pair})==1
        assert all(s['pair_order']==rank and s['logical_point_budget']==s['max_expansions']==256 for s in pair)


@pytest.mark.parametrize('change',['missing','duplicate_id','duplicate_fault','unbalanced_event'])
def test_no_omission_or_favorable_population_substitution(change):
    rows=contexts()
    if change=='missing':rows.pop()
    elif change=='duplicate_id':rows[1]['context_id']=0
    elif change=='duplicate_fault':rows[1]['scenario_id']=rows[0]['scenario_id']
    else:rows[0]['event_id']=6
    with pytest.raises(ValueError):runner.schedule_for(rows)


def test_windows_manifest_paths_are_normalized_before_required_checks(tmp_path):
    write(tmp_path/'nested/data.json',{'real':'observed'})
    digest=runner.sha(tmp_path/'nested/data.json')
    write(tmp_path/'completion_manifest.json',{'nested\\data.json':digest})
    assert runner.verify_manifest(tmp_path,required=('nested/data.json',))=={'nested/data.json':digest}


@pytest.mark.parametrize('kind',['alias','traversal','wrapper','badsha','changed'])
def test_manifest_cannot_be_replaced_by_self_consistent_or_malformed_aliases(tmp_path,kind):
    write(tmp_path/'x.json',{'x':1});digest=runner.sha(tmp_path/'x.json')
    manifest={'x.json':digest}
    if kind=='alias':manifest['./x.json']=digest
    elif kind=='traversal':manifest={'../x.json':digest}
    elif kind=='wrapper':manifest={'status':'PASS','files':manifest}
    elif kind=='badsha':manifest['x.json']='incorrect'
    else:write(tmp_path/'x.json',{'x':2})
    write(tmp_path/'completion_manifest.json',manifest)
    with pytest.raises(ValueError):runner.verify_manifest(tmp_path)


def historical_fixture(tmp_path):
    spec={'cell':'episodes/000','window_variant':'coverage_v1'}
    cell=tmp_path/spec['cell']
    for name in (*runner.HISTORY_FILES,'points.jsonl.gz'):
        write(cell/name,{'file':name})
    cm={p.name:runner.sha(p) for p in cell.iterdir()}
    write(cell/'completion_manifest.json',cm)
    audit={'status':'PASS','results':[{'cell':spec['cell'],'spec':spec,
        'cell_completion_sha256':runner.sha(cell/'completion_manifest.json')}]}
    write(tmp_path/'audit/summary.json',audit)
    sm={spec['cell']+'/'+name:digest for name,digest in cm.items()}
    sm['audit/summary.json']=runner.sha(tmp_path/'audit/summary.json')
    return spec,sm,cell


def test_historical_nested_seal_is_bound_through_original_root_audit(tmp_path):
    spec,sm,cell=historical_fixture(tmp_path)
    assert runner.verify_source_cell(tmp_path,sm,spec)[0]==cell
    write(cell/'result.json',{'altered':True})
    cm=runner.read(cell/'completion_manifest.json');cm['result.json']=runner.sha(cell/'result.json')
    write(cell/'completion_manifest.json',cm)
    with pytest.raises(ValueError,match='independent audit'):
        runner.verify_source_cell(tmp_path,sm,spec)


def test_accounting_separates_logical_samples_actual_solves_and_missing_work(tmp_path):
    specs=[{'cell':'episodes/000'},{'cell':'episodes/001'}]
    write(tmp_path/'episodes/000/equivalent_cache_summary.json',
        {'logical_points':122,'physical_points':62,'actual_numerical_entries':251})
    assert runner.costs(tmp_path,specs)=={'known_logical_points':122,'known_physical_points':62,
        'known_numerical_entries':251,'unknown_cost_cells':['episodes/001']}


@pytest.mark.parametrize('logical,physical',[(257,200),(61,62),(-1,0),(True,1)])
def test_bad_accounting_is_unknown_not_zero_or_a_success(tmp_path,logical,physical):
    write(tmp_path/'episodes/000/equivalent_cache_summary.json',
        {'logical_points':logical,'physical_points':physical,'actual_numerical_entries':3+4*physical})
    assert runner.costs(tmp_path,[{'cell':'episodes/000'}])['unknown_cost_cells']==['episodes/000']


def test_checkpoint_charges_actual_entries_and_keeps_logical_budget_distinct():
    oracle=SimpleNamespace(points=61,physical_points=1,
        factory=SimpleNamespace(stats=lambda:{'returned':{'numerical_solution_entries':7}}))
    search=SimpleNamespace(oracle=oracle,incumbent={'certified_complete':True,'validation':{'status':'COMPLETE'}},
        _beam_turn=0,quality=.5)
    row=runner.checkpoint(search,runner.time.perf_counter())
    assert row['logical_points']==61 and row['physical_points']==1 and row['numerical_entries']==7
    assert row['fully_certified'] and row['quality_pct']==50


def test_scope_review_requires_zero_solve_pass_and_copies_flat_seal(tmp_path,monkeypatch):
    review=tmp_path/'review';run=tmp_path/'run';run.mkdir()
    write(review/'summary.json',{'status':'PASS','new_physical_points':0,'new_training':0})
    write(review/'completion_manifest.json',{'summary.json':runner.sha(review/'summary.json')})
    monkeypatch.setattr(runner,'REVIEW',review);monkeypatch.setattr(runner,'ROOT',tmp_path)
    binding=runner.bind_scope_review(run)
    assert binding['completion_sha256']==runner.sha(review/'completion_manifest.json')
    assert runner.read(run/'scope_review/summary.json')['status']=='PASS'


def test_passed_test_evidence_requires_all_new_components(tmp_path):
    path=tmp_path/'tests.xml'
    names=('test_run_clpu_equivalent_cache','test_report_clpu_equivalent_cache',
           'test_nature34_clpu_equivalent_cache','test_clpu_equivalent_cache_audit')
    path.write_text('<testsuite>'+''.join(f'<testcase classname="{n}"/>' for n in names)+'</testsuite>')
    assert runner.passed_tests(path)==4
    path.write_text('<testsuite><testcase classname="unrelated"/></testsuite>')
    with pytest.raises(ValueError,match='Missing new component'):runner.passed_tests(path)


def exact_review_fixture(directory, prior_sha='a'*64):
    review={'status':'PASS','source_run':runner.SOURCE_RUN,'source_completion_sha256':runner.SOURCE_SHA,
        'prior_review_manifest_sha256':prior_sha,'new_physical_points':0,'new_training':0,
        'physical_points_reviewed':1121,'duplicate_pairs_reviewed':556,'new_group_merges':18,
        'certificate_or_feedback_disagreements':0,
        'all_differences_in_existing_excluded_disabled_cache_fields':True,
        'constraint_or_audit_tolerances_changed':False,'errors':[]}
    write(directory/'summary.json',review)
    for i in range(8):write(directory/f'context_{i:02d}.json',{'artificial_software_fixture_context':i})
    write(directory/'completion_manifest.json',{p.name:runner.sha(p) for p in directory.iterdir()})
    return review


def test_final_key_review_copies_both_linked_reviews_without_replacing_prior(tmp_path,monkeypatch):
    review=tmp_path/'artifacts/clpu_equivalent_cache_exact_key_review_20261030';run=tmp_path/'run'
    exact_review_fixture(review)
    write(run/'scope_review_binding.json',{'completion_sha256':'a'*64})
    monkeypatch.setattr(runner,'EXACT_REVIEW',review);monkeypatch.setattr(runner,'ROOT',tmp_path)
    result=runner.bind_exact_key_review(run)
    assert result['completion_sha256']==runner.sha(review/'completion_manifest.json')
    assert runner.read(run/'scope_review_binding.json')['completion_sha256']=='a'*64
    assert len(runner.verify_exact_key_review(run))==9


@pytest.mark.parametrize('change',['prior','source','incomplete_contexts','count','physical','feedback','tolerance','errors'])
def test_final_key_review_rejects_unbound_or_incomplete_review(tmp_path,change):
    directory=tmp_path/'review';value=exact_review_fixture(directory)
    if change=='prior':value['prior_review_manifest_sha256']='b'*64
    elif change=='source':value['source_completion_sha256']='b'*64
    elif change=='count':value['duplicate_pairs_reviewed']=538
    elif change=='physical':value['new_physical_points']=1
    elif change=='feedback':value['certificate_or_feedback_disagreements']=1
    elif change=='tolerance':value['constraint_or_audit_tolerances_changed']=True
    elif change=='errors':value['errors']=['unresolved']
    write(directory/'summary.json',value)
    files={p.name:runner.sha(p) for p in directory.iterdir() if p.name!='completion_manifest.json'}
    if change=='incomplete_contexts':files.pop('context_07.json')
    write(directory/'completion_manifest.json',files)
    with pytest.raises(ValueError):runner.validate_exact_review(directory,'a'*64)


def test_actual_sealed_final_key_review_remains_readable_without_solver():
    prior=runner.sha(runner.REVIEW/'completion_manifest.json')
    files=runner.validate_exact_review(runner.EXACT_REVIEW,prior)
    assert {'summary.json',*(f'context_{i:02d}.json' for i in range(8))}<=set(files)


@pytest.mark.parametrize('fail_first',[False,True])
def test_controller_drains_pairs_and_preserves_failed_or_missing_costs(tmp_path,monkeypatch,fail_first):
    run=tmp_path/'run';run.mkdir()
    schedule=runner.schedule_for(contexts())
    write(run/'schedule.json',schedule);write(run/'source_cells.json',contexts())
    for spec in schedule:write(run/spec['cell']/'spec.json',spec)
    monkeypatch.setattr(runner,'verify_execution',lambda run:None)
    monkeypatch.setattr(runner,'ACTIVE',tmp_path/'active.json')
    monkeypatch.setattr(runner.shutil,'disk_usage',lambda root:SimpleNamespace(free=20*2**30))
    monkeypatch.setattr(runner.time,'sleep',lambda _:None)
    launched=[];live=set();maximum=[0]
    class Process:
        def __init__(self,cmd,**kwargs):
            cell=Path(cmd[-1]);self.spec=runner.read(cell/'spec.json');self.pid=100+len(launched)
            # The following pair may begin only after both prior processes were reaped.
            if self.spec['arm_launch_order']==0:assert not live
            live.add(self.pid);maximum[0]=max(maximum[0],len(live));launched.append(self.spec)
            self.code=1 if fail_first and len(launched)==1 else 0
            if self.code==0:
                write(cell/'equivalent_cache_summary.json',
                    {'logical_points':4,'physical_points':2,'actual_numerical_entries':11})
                write(cell/'completion_manifest.json',{'stub':'test-only'})
        def poll(self):
            live.remove(self.pid)
            return self.code
        def wait(self):
            live.discard(self.pid)
            return self.code
    monkeypatch.setattr(runner.subprocess,'Popen',Process)
    code=runner.execute(run)
    result=runner.read(run/'controller_completion.json')
    assert maximum[0]==2 and not live
    if fail_first:
        assert code==1 and len(launched)==2 and len(result['completed'])==1
        assert result['status']=='INCONCLUSIVE' and result['unknown_cost_cells']==[schedule[0]['cell']]
    else:
        assert code==0 and len(launched)==16 and len(result['completed'])==16
        assert result['known_physical_points']==32 and result['known_logical_points']==64
    with pytest.raises(ValueError,match='Already started'):runner.execute(run)
