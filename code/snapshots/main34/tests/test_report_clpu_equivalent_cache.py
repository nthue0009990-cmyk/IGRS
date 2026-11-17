"""Artificial accounting fixtures are software tests, not study results."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import report_clpu_equivalent_cache as report
import run_clpu_equivalent_cache_pilot as runner
from test_run_clpu_equivalent_cache import contexts, write, exact_review_fixture


def fixture(arm='semantic_identity'):
    alias=arm=='electrical_equivalent'
    spec={'cell':'episodes/001' if alias else 'episodes/000','context_id':0,'scenario_id':'fault0',
          'event_id':30,'arm':arm,'pair_order':0,'arm_launch_order':int(alias)}
    plan={'status':'COMPLETE','actions':[0,1],'certified_complete':True,
          'validation':{'status':'COMPLETE','energy_kwh':3}}
    old={'status':'CERTIFIED_INCUMBENT','config':{},'stop_reason':'DONE','best_plan':plan,
         'initial_certificate':{'point_id':0,'status':'PASS','reliable':True,'feedback':[]},
         'wait_baseline_quality_pct':20.,'completed_expansions':1,'search_decisions':1,
         'counts':{'checked_candidates':2},'oracle':{'points':4},
         'checkpoints':[{'search_choices':i,'physical_points':3+i,'quality_pct':20.+10*i,
             'fully_certified':True,'elapsed_seconds':10.+i,'numerical_entries':15+4*i} for i in range(2)]}
    steps=[{'transition_id':0,'status':'COMPLETED','method':'beam','action':0,'branch_id':3,
        'window_branch_ids':[3,4],'oracle_before':{'requests':5,'points':3},
        'oracle_after':{'requests':8,'points':4},'response':{'done':True},'elapsed_seconds':12.}]
    current=deepcopy(old)
    current.update(spec=spec,training_interactions=0,optimizer_updates=0,new_training=0,
        formal_holdout_numerical_access=False,tune_numerical_access=False,
        elapsed_seconds=4.,computation_elapsed_seconds=3.,preflight_elapsed_seconds=1.,
        runtime_initialization_seconds=.1,initial_elapsed_seconds=2.)
    current['oracle']={'variant':arm,'logical_points':4,'physical_points':2 if alias else 4,
        'requests':8,'semantic_cache_hits':4,'electrical_reuses':2 if alias else 0,
        'actual_numerical_entries':11 if alias else 19,'initialization_numerical_entries':3}
    for i,p in enumerate(current['checkpoints']):
        p.update(logical_points=3+i,physical_points=1+i if alias else 3+i,
                 numerical_entries=7+4*i if alias else 15+4*i,elapsed_seconds=2.+i)
    newsteps=deepcopy(steps)
    for name in ('oracle_before','oracle_after'):
        newsteps[0][name].update(logical_points=newsteps[0][name]['points'],physical_points=1 if alias else 3)
    newsteps[0]['elapsed_seconds']=3.
    return spec,current,[plan],newsteps,(old,[deepcopy(plan)],steps)


def row(arm='semantic_identity'):
    s,r,p,t,h=fixture(arm)
    return report.case_row(s,r,5. if arm=='semantic_identity' else 4.5,p,t,h)


def test_physical_reuse_can_pass_only_with_full_search_semantics_unchanged():
    semantic,equivalent=row(),row('electrical_equivalent')
    result=report.pair_rows(semantic,equivalent)
    assert result['equivalence_gate_passed']
    assert result['equivalent_minus_semantic']['logical_points']==0
    assert result['equivalent_minus_semantic']['physical_points']==-2
    assert result['physical_point_saving_fraction']==.5
    assert result['equivalent_minus_semantic']['process_wall_seconds']==-.5


@pytest.mark.parametrize('change',['status','stop','plan','choice','point_count','quality','failure'])
def test_equal_final_quality_alone_cannot_hide_changed_trajectory_or_failure(change):
    s,r,p,t,h=fixture('electrical_equivalent')
    if change=='status':r['status']='OTHER'
    elif change=='stop':r['stop_reason']='OTHER'
    elif change=='plan':p[0]['actions']=[1,0]
    elif change=='choice':t[0]['action']=1
    elif change=='point_count':r['counts']['checked_candidates']=1
    elif change=='quality':r['checkpoints'][0]['quality_pct']=21.
    else:r['initial_certificate']['feedback']=[{'unrecorded':'failure'}]
    modified=report.case_row(s,r,4.5,p,t,h)
    assert not modified['matches_historical_semantics']
    assert not report.pair_rows(row(),modified)['equivalence_gate_passed']


def test_first_failure_location_and_feedback_are_retained():
    plan={'status':'FAIL','actions':[1,0],'validation':{'failed_step':1,'failed_time_seconds':90,
        'failed_state':{'command':9},'samples':[{'point_id':7}], 'feedback':[{'time_minutes':1.5}]}}
    failures=report.first_failures([plan])
    assert failures[0]['point_id']==7 and failures[0]['failed_time_seconds']==90
    assert failures[0]['feedback']==[{'time_minutes':1.5}]


@pytest.mark.parametrize('change',['hidden_actual','false_reuse','semantic_alias','scope','missing_quality','cost_regression','missing_step'])
def test_invalid_dual_ledger_or_hidden_certificate_cost_rejected(change):
    s,r,p,t,h=fixture()
    if change=='hidden_actual':r['oracle']['actual_numerical_entries']=7
    elif change=='false_reuse':r['oracle']['electrical_reuses']=1
    elif change=='semantic_alias':
        r['oracle'].update(physical_points=3,electrical_reuses=1,actual_numerical_entries=15)
    elif change=='scope':r['new_training']=1
    elif change=='missing_quality':r['checkpoints'][0]['fully_certified']=False
    elif change=='cost_regression':r['checkpoints'][1]['physical_points']=2
    else:r['checkpoints'][1]['search_choices']=2
    with pytest.raises(ValueError):report.case_row(s,r,5.,p,t,h)


def test_distribution_is_descriptive_and_eight_sample_p95_is_maximum():
    values=report.distribution([8,1,7,2,6,3,5,4])
    assert values=={'n':8,'mean':4.5,'median':4.5,'p95_nearest_rank':8.,'maximum':8.}
    assert report.distribution([])['mean'] is None


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1,True])
def test_invalid_timing_is_not_silently_dropped(value):
    with pytest.raises(ValueError):report.distribution([1.,value])


def write_lines(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,'wt',encoding='utf-8') as stream:
        for item in rows:stream.write(json.dumps(item)+'\n')


def seal(directory,name='completion_manifest.json'):
    write(directory/name,{p.relative_to(directory).as_posix():report.sha(p)
        for p in directory.rglob('*') if p.is_file() and p.name!=name})


def complete_report_fixture(tmp_path):
    run=tmp_path/'run';run.mkdir()
    schedule=runner.schedule_for(contexts())
    write(run/'schedule.json',schedule)
    write(run/'source_binding.json',{'run':runner.SOURCE_RUN,'completion_sha256':runner.SOURCE_SHA})
    write(run/'physical_scope.json',{'test_only':True})
    write(run/'scope_review_binding.json',{'completion_sha256':'a'*64})
    exact_review_fixture(run/'exact_key_review')
    write(run/'exact_key_review_binding.json',{'path':'artifacts/clpu_equivalent_cache_exact_key_review_20261030',
        'completion_sha256':report.sha(run/'exact_key_review/completion_manifest.json'),
        'summary_sha256':report.sha(run/'exact_key_review/summary.json')})
    write(run/'config.json',{'equivalent_cache_study':runner.STUDY})
    histories=[]
    for context in contexts():
        i=context['context_id'];prefix=f'historical/c{i:02d}';directory=run/prefix
        _,_,_,_,old=fixture()
        write(directory/'result.json',old[0]);write_lines(directory/'plans.jsonl.gz',old[1])
        write_lines(directory/'steps.jsonl.gz',old[2]);seal(directory)
        histories.append({**context,'source_cell':f'episodes/{i:03d}','historical_directory':prefix,
            'source_completion_sha256':report.sha(directory/'completion_manifest.json'),
            'source_files':{p.name:report.sha(p) for p in directory.iterdir() if p.name!='completion_manifest.json'}})
    write(run/'source_cells.json',histories)
    for spec in schedule:write(run/spec['cell']/'spec.json',spec)
    seal(run,'prephysical_freeze_manifest.json')
    write(run/'prepared.json',{'freeze_sha256':report.sha(run/'prephysical_freeze_manifest.json')})
    results=[];exits=[]
    for spec in schedule:
        cell=run/spec['cell'];_,result,plans,steps,_=fixture(spec['arm']);result['spec']=spec
        write(cell/'result.json',result);write(cell/'equivalent_cache_summary.json',result['oracle'])
        write_lines(cell/'plans.jsonl.gz',plans);write_lines(cell/'steps.jsonl.gz',steps);seal(cell)
        results.append({'cell':spec['cell'],'cell_completion_sha256':report.sha(cell/'completion_manifest.json')})
        exits.append({'cell':spec['cell'],'exit_code':0,'process_wall_seconds':5. if spec['arm']==runner.ARMS[0] else 4.5})
    write(run/'controller_completion.json',{'status':'COMPLETE_CELLS_PENDING_AUDIT','errors':[],
        'unknown_cost_cells':[],'completed':[s['cell'] for s in schedule],'scheduled':16,
        'exits':exits,'known_logical_points':64,'known_physical_points':48,'known_numerical_entries':240,'wall_seconds':40.})
    audit={'status':'PASS','errors':[],'new_audit_physical_points':0,'new_training_tasks':0,
        'audited_cells':16,'results':results,'physical_points':48,'numerical_entries':240,
        'source_window_completion_sha256':runner.SOURCE_SHA,
        'exact_key_review_manifest_sha256':report.sha(run/'exact_key_review/completion_manifest.json'),
        **{k:report.sha(run/name) for k,name in [('freeze_sha256','prephysical_freeze_manifest.json'),
            ('schedule_sha256','schedule.json'),('source_binding_sha256','source_binding.json'),
            ('controller_completion_sha256','controller_completion.json')]}}
    write(run/'audit/summary.json',audit);seal(run/'audit')
    return run


def test_full_report_requires_and_retains_all_sixteen_bound_cells(tmp_path):
    run=complete_report_fixture(tmp_path)
    summary=report.report(run)
    assert summary['status']=='EQUIVALENT_WITH_MEASURED_PHYSICAL_REDUCTION'
    assert summary['equivalent_pairs']==8 and summary['physical_points_saved']==16
    assert summary['physical_saving_fraction']==.5 and summary['cells']==16
    assert len(report.read(run/'derived_report/all_cells.json'))==16
    assert len(report.read(run/'derived_report/all_pairs.json'))==8
    assert report.verify_manifest(run/'derived_report',required=('all_cells.json','all_pairs.json','summary.json'))
    with pytest.raises(ValueError,match='New within-run'):report.report(run)


@pytest.mark.parametrize('change',['cell_reseal','controller','omitted_audit_cell','frozen_history','review_audit_binding'])
def test_full_report_rejects_tampering_or_selective_omission(tmp_path,change):
    run=complete_report_fixture(tmp_path)
    if change=='cell_reseal':
        cell=run/'episodes/000';result=report.read(cell/'result.json');result['stop_reason']='CHANGED'
        write(cell/'result.json',result);seal(cell)
    elif change=='controller':
        ctl=report.read(run/'controller_completion.json');ctl['wall_seconds']=0
        write(run/'controller_completion.json',ctl)
    elif change=='omitted_audit_cell':
        audit=report.read(run/'audit/summary.json');audit['results'].pop()
        write(run/'audit/summary.json',audit);seal(run/'audit')
    elif change=='review_audit_binding':
        audit=report.read(run/'audit/summary.json');audit['exact_key_review_manifest_sha256']='b'*64
        write(run/'audit/summary.json',audit);seal(run/'audit')
    else:write(run/'historical/c00/result.json',{'changed':'old evidence'})
    with pytest.raises(ValueError):report.report(run)
