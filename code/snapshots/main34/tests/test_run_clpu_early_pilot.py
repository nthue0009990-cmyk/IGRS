"""Bounded early-driver tests. All oracle calls are synthetic; no solver imports."""
from collections import Counter
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_clpu_early_pilot as runner
from dnr.nature34_clpu_branch import BranchConfig,BranchSearch
from test_nature34_clpu_branch import CountedOracle,environment


def contexts():
    return [{'context_id':i,'scenario_id':f'fault{i}','event_id':(30,20,6,3)[i%4]} for i in range(8)]


def models():
    return [{'seed':seed,'method':method,'checkpoint':f'models/seed{seed}/{method}.pt','source_sha256':str(seed)[0]*64}
            for method in ('bc_raw','ppo_raw') for seed in runner.SEEDS]


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def rows(path):
    op=gzip.open if str(path).endswith('.gz') else open
    with op(path,'rt',encoding='utf-8') as stream:return [json.loads(line) for line in stream if line.strip()]


def test_schedule_has_64_balanced_cells_120_cold_searches_and_full_paid_budget():
    actual=runner.schedule_for(contexts(),models())
    assert actual==runner.schedule_for(list(reversed(contexts())),list(reversed(models())))
    assert len(actual)==64 and sum(s['cold_searches'] for s in actual)==120
    assert sum(s['physical_point_budget'] for s in actual)==30720
    assert Counter(s['arm'] for s in actual)=={'window_reference':8,'protected_rule':8,'protected_bc':24,'protected_ppo':24}
    for slot in runner.SLOTS:
        assert sorted(s['arm_position'] for s in actual if s['arm_slot']==slot)==list(range(8))
    for spec in actual:
        phases=[runner.phase_spec(spec,p) for p in runner.phase_names(spec)]
        assert sum(p['point_budget'] for p in phases)==spec['physical_point_budget']
        assert all(p['point_budget']==p['total_logical_point_budget']==256 for p in phases)
        if spec['protected']:
            assert phases[1]['method']=='beam' and phases[1]['checkpoint'] is None
            assert phases[1]['parent_seed']==spec['seed']


@pytest.mark.parametrize('change',['omit','duplicate_context','event','duplicate_model'])
def test_population_or_model_substitution_is_rejected(change):
    cs,ms=contexts(),models()
    if change=='omit':cs.pop()
    elif change=='duplicate_context':cs[1]['context_id']=0
    elif change=='event':cs[0]['event_id']=6
    else:ms[0]=ms[1]
    with pytest.raises(ValueError):runner.schedule_for(cs,ms)


def test_selector_block_is_closed_loop_and_switches_on_sixteenth_boundary():
    class Search:
        _beam_turn=0;done=False
        config=SimpleNamespace(max_branch_expansions=20)
        oracle=SimpleNamespace(points=0,physical_points=0,requests=0,
            factory=SimpleNamespace(stats=lambda:{'returned':{'numerical_solution_entries':3}}))
        def observation(self,rep):
            return {'representation':rep,'candidate_features':[[float(self._beam_turn)],[float(self._beam_turn+1)]],
                    'global_features':[0.],'branch_ids':[self._beam_turn*2,self._beam_turn*2+1], 'mask':[True,True]}
        def rule_action(self,method):return 0
        def step(self,action):
            self._beam_turn+=1; self.done=self._beam_turn==20
            return {'reward':0.,'done':self.done,'chosen':action}
    class Policy:
        def act(self,obs,mask,**kwargs):return {'action':1,'logprob':0.,'value':0.}
    journal,checkpoints=[],[]
    actual=runner.collect_phase(Search(),{'search_phase':'EARLY','method':'ppo_raw','checkpoint_sha256':'x',
        'deadline_utc_epoch':runner.time.time()+10},Policy(),None,journal.append,checkpoints.append)
    assert actual['policy_decisions']==16 and actual['search_decisions']==20
    assert [r['action'] for r in journal[1::2]]==[1]*16+[0]*4
    assert checkpoints==list(range(1,21))
    for before,after in zip(journal[::2],journal[1::2]):
        assert 'obs_compact' in before and 'obs_compact' not in after
        assert runner.digest(before['obs_compact'])==before['obs_compact_sha256']==after['obs_compact_sha256']
        assert 'observed_failure_history' not in before['obs_compact']


class FakeOracle(CountedOracle):
    failed=False
    unknown_input=False
    initialization_error=False
    instances=[]
    def __init__(self,parent,cell,env,budget,variant,scope):
        if self.initialization_error:
            raise RuntimeError('Synthetic author-runtime initialization failure')
        if any(o.actual_pid == runner.os.getpid() for o in self.instances):
            raise RuntimeError('Only one Nature34 runtime per process is supported')
        super().__init__(env,budget,fail=(lambda s,t:True) if self.failed else None,unknown=self.unknown_input)
        self.cell,self.variant=cell,variant
        self.actual_pid=runner.os.getpid()
        self.factory=SimpleNamespace(stats=lambda:{'returned':{'numerical_solution_entries':3+4*self.points}})
        self.instances.append(self)
    @property
    def physical_points(self):return self.points
    def __enter__(self):return self
    def __exit__(self,*args):
        write(self.cell/'equivalent_cache_summary.json',{'logical_points':self.points,'physical_points':self.points,
            'actual_numerical_entries':3+4*self.points,'initialization_numerical_entries':3,'requests':self.requests,
            'semantic_cache_hits':self.hits,'electrical_reuses':0,'variant':self.variant})


def worker_fixture(tmp_path,monkeypatch,arm):
    import run_nature34_clpu_pilot as pilot
    import dnr.nature34_clpu_equivalent_cache as electrical
    import nature34_clpu_tail_execution as driver
    run=tmp_path/'run';cell=run/'episodes/000'
    spec=next(s for s in runner.schedule_for(contexts(),models()) if s['arm']==arm)
    spec={**spec,'cell':'episodes/000','episode_id':0}
    if spec['checkpoint']:
        model=run/spec['checkpoint'];model.parent.mkdir(parents=True);model.write_bytes(b'fixed-mock')
        spec['checkpoint_sha256']=runner.sha(model)
    for name,value in [('config.json',{'branch_search':asdict(BranchConfig(max_branch_expansions=2))}),
        ('execution_started.json',{'deadline_epoch_seconds':runner.time.time()+100}),('schedule.json',[spec]),
        ('physical_scope.json',{'test':'synthetic'})]:write(run/name,value)
    write(cell/'spec.json',spec)
    for phase in runner.phase_names(spec):
        ps=runner.phase_spec(spec,phase);write(run/ps['cell']/'spec.json',ps)
    monkeypatch.setattr(runner,'verify_execution',lambda r:None)
    monkeypatch.setattr(runner,'physical_identity',lambda *a:{'synthetic_physics':'fixed'})
    monkeypatch.setattr(runner,'_WORKER_ENTERED',False)
    monkeypatch.setattr(pilot,'make_env',lambda *a:environment(horizon=2))
    monkeypatch.setattr(electrical,'EquivalenceOracle',FakeOracle)
    monkeypatch.setattr(FakeOracle,'instances',[])
    class FixedPolicy:
        def act(self,obs,mask,**kwargs):return {'action':len(mask)-1,'logprob':0.,'value':0.}
    monkeypatch.setattr(driver,'load_fixed_policy',lambda p,s:(FixedPolicy(),None) if p else (None,None))
    launched=[]
    class PhaseProcess:
        """Synthetic child boundary only; never invokes Python or DSS."""
        def __init__(self,cmd,**kwargs):
            assert 'phase-worker' in cmd and kwargs['cwd']==runner.ROOT
            folder=Path(cmd[-1])
            self.pid=30000+len(launched)
            actual_pid=20000+len(launched)
            launched.append(self)
            with monkeypatch.context() as child:
                child.setattr(runner,'_PHASE_ENTERED',False)
                child.setattr(runner.os,'getpid',lambda:actual_pid)
                self.code=runner.phase_worker(run,folder)
        def poll(self):return self.code
    monkeypatch.setattr(runner.subprocess,'Popen',PhaseProcess)
    return run,cell,spec


@pytest.mark.parametrize('arm',['window_reference','protected_rule','protected_bc','protected_ppo'])
def test_cold_worker_preserves_independent_guard_and_complete_costs(tmp_path,monkeypatch,arm):
    run,cell,spec=worker_fixture(tmp_path,monkeypatch,arm)
    assert runner.worker(run,cell)==0,runner.read(cell/'error.json') if (cell/'error.json').exists() else ''
    result=runner.read(cell/'result.json')
    assert result['spec']==spec==runner.read(cell/'spec.json')
    assert result['new_training']==result['optimizer_updates']==0
    assert len(FakeOracle.instances)==spec['cold_searches']
    assert len({o.actual_pid for o in FakeOracle.instances})==spec['cold_searches']
    assert all(o.actual_pid!=runner.os.getpid() for o in FakeOracle.instances)
    assert len({id(o.cache) for o in FakeOracle.instances})==spec['cold_searches']
    assert result['oracle']['physical_points']==sum(o.points for o in FakeOracle.instances)
    assert result['numerical_entries']==3*spec['cold_searches']+4*result['actual_physical_points']
    assert result['checkpoints']==rows(cell/'cost_checkpoints.jsonl')
    assert [r['checkpoint_index'] for r in result['checkpoints']]==list(range(len(result['checkpoints'])))
    assert result['final_real_cost']['physical_points']==result['actual_physical_points']
    assert result['early_block_checkpoint']['phase_choices']==min(16,result['phases']['EARLY']['search_decisions'])
    assert result['preflight_elapsed_seconds']>=result['model_load_seconds']
    assert result['elapsed_seconds']>=result['preflight_elapsed_seconds']+result['computation_elapsed_seconds']
    for phase in runner.phase_names(spec):
        folder=cell/phase.lower();p=runner.read(folder/'phase_result.json')
        journal=rows(folder/'decisions.jsonl.gz')
        assert len(journal)==2*p['search_decisions']
        assert runner.cache.verify_manifest(folder,required=('phase_result.json','spec.json'))
        assert all(r['method']=='beam' for r in journal) if phase=='GUARD' else True
    if spec['protected']:
        env=environment(horizon=2);oracle=CountedOracle(env)
        expected=BranchSearch(env,oracle,BranchConfig(max_branch_expansions=2))
        while not expected.done:expected.step(expected.rule_action('beam'))
        actual=runner.read(cell/'guard/phase_result.json')
        assert actual['trace']==expected.result()['trace']
        assert actual['best_plan']==expected.result()['best_plan']
        assert actual['oracle']['logical_points']==oracle.points
        assert result['guard_audit']['nondegradation_verified'] is True
        assert all(not p['guard_established'] for p in result['checkpoints'][:-1])
        assert result['checkpoints'][-1]['guard_established'] is True
        launches=rows(cell/'phase_dispatch.jsonl')
        exits=runner.read(cell/'phase_process_exits.json')
        assert launches[1]['started_monotonic']>=exits[0]['finished_monotonic']
        assert len({row['actual_child_pid'] for row in exits})==2
        assert all(row['actual_child_pid']!=row['launcher_pid'] for row in exits)
        assert result['phase_processes']==exits
    else:assert result['guard_audit'] is None and not (cell/'guard').exists()
    assert runner.costs(run,[spec])['known_physical_points']==result['actual_physical_points']
    assert runner.cache.verify_manifest(cell,required=('result.json','spec.json'))


def test_initial_fail_is_retained_in_both_cold_phases_without_guard_claim(tmp_path,monkeypatch):
    run,cell,spec=worker_fixture(tmp_path,monkeypatch,'protected_rule')
    monkeypatch.setattr(FakeOracle,'failed',True)
    assert runner.worker(run,cell)==0
    result=runner.read(cell/'result.json')
    assert result['status']=='INITIAL_FAIL' and result['quality_pct'] is None
    assert result['oracle']['physical_points']==2 and result['numerical_entries']==14
    assert not result['guard_audit']['guard_established'] and result['search_decisions']==0


def test_unknown_cannot_launch_guard_or_seal_success_and_paid_calls_remain(tmp_path,monkeypatch):
    run,cell,spec=worker_fixture(tmp_path,monkeypatch,'protected_rule')
    monkeypatch.setattr(FakeOracle,'unknown_input',True)
    assert runner.worker(run,cell)==1
    assert not (cell/'completion_manifest.json').exists()
    assert not (cell/'guard/phase_started.json').exists()
    assert runner.costs(run,[spec])['known_physical_points']==1


def test_runtime_initialization_failure_has_unknown_cost_and_never_starts_guard(tmp_path,monkeypatch):
    run,cell,spec=worker_fixture(tmp_path,monkeypatch,'protected_rule')
    monkeypatch.setattr(FakeOracle,'initialization_error',True)
    assert runner.worker(run,cell)==1
    assert (cell/'early/phase_started.json').exists() and (cell/'early/phase_error.json').exists()
    assert not (cell/'guard/phase_execution.json').exists()
    assert runner.costs(run,[spec])['unknown_cost_cells']==[spec['cell']+'/early']
    exits=runner.read(cell/'phase_process_exits.json')
    assert len(exits)==1 and exits[0]['exit_code']==1


def test_same_process_second_runtime_is_forbidden_by_faithful_mock(tmp_path,monkeypatch):
    monkeypatch.setattr(FakeOracle,'instances',[])
    env=environment(horizon=2)
    FakeOracle(tmp_path,tmp_path/'early',env,256,'electrical_equivalent',{})
    with pytest.raises(RuntimeError,match='one Nature34 runtime'):
        FakeOracle(tmp_path,tmp_path/'guard',env,256,'electrical_equivalent',{})


def test_native_exit_failure_after_phase_seal_still_blocks_guard(tmp_path,monkeypatch):
    run,cell,spec=worker_fixture(tmp_path,monkeypatch,'protected_rule')
    normal=runner.subprocess.Popen
    class FailedNativeExit(normal):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.code=19
    monkeypatch.setattr(runner.subprocess,'Popen',FailedNativeExit)
    assert runner.worker(run,cell)==1
    assert (cell/'early/completion_manifest.json').exists()
    assert not (cell/'guard/phase_execution.json').exists()
    assert not (cell/'completion_manifest.json').exists()
    exits=runner.read(cell/'phase_process_exits.json')
    assert exits[0]['exit_code']==19 and exits[0]['phase_completion_sha256'] is not None
    assert runner.costs(run,[spec])['known_physical_points']>0


def test_child_cost_clock_is_converted_without_reset_or_free_guard_points():
    local=[{'checkpoint_index':0,'phase':'GUARD','kind':'INITIAL','phase_choices':0,'search_choices':0,
        'logical_points':61,'physical_points':1,'numerical_entries':7,'requests':91,
        'phase_logical_points':61,'phase_physical_points':1,'quality_pct':20.,'fully_certified':True,
        'guard_established':False,'worker_elapsed_seconds':32.,'elapsed_seconds':2.}]
    previous={'logical_points':200,'physical_points':100,'numerical_entries':403,'requests':1000}
    row=runner.combine_checkpoints(local,previous,180,60.,5.,3)[0]
    assert row['checkpoint_index']==3 and row['search_choices']==180
    assert row['logical_points']==261 and row['physical_points']==101 and row['numerical_entries']==410
    assert row['quality_pct']==60. and row['elapsed_seconds']==27. and row['worker_elapsed_seconds']==32.


def test_duplicate_worker_cannot_rewrite_finished_evidence(tmp_path,monkeypatch):
    run,cell,spec=worker_fixture(tmp_path,monkeypatch,'window_reference')
    write(cell/'worker_started.json',{'finished':True})
    before={p.name:runner.sha(p) for p in cell.iterdir() if p.is_file()}
    with pytest.raises(ValueError,match='Repeated task'):runner.worker(run,cell)
    assert before=={p.name:runner.sha(p) for p in cell.iterdir() if p.is_file()}


def controller_fixture(tmp_path,monkeypatch,fail_at=None,unknown_at=None):
    run=tmp_path/'run';schedule=runner.schedule_for(contexts(),models())
    write(run/'schedule.json',schedule);write(run/'resource_budget.json',{'historical_max_cell_bytes':100})
    for spec in schedule:write(run/spec['cell']/'spec.json',spec)
    monkeypatch.setattr(runner,'verify_execution',lambda r:None)
    monkeypatch.setattr(runner,'ACTIVE',tmp_path/'active.json')
    monkeypatch.setattr(runner,'storage_forecast',lambda *a:{'free_bytes':10**12,'required_free_bytes':0})
    launched=[]
    class Process:
        def __init__(self,cmd,**kwargs):
            self.index=len(launched);self.pid=1000+self.index
            self.cell=Path(cmd[-1]);self.code=1 if self.index==fail_at else 0
            assert all(kwargs['env'][n]=='1' for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'))
            if self.index%2==0 and self.index:assert len(runner.read(run/'controller_exits.json'))==self.index
            launched.append(self)
            write(self.cell/'worker_started.json',{'pid':self.pid})
            spec=runner.read(self.cell/'spec.json')
            for phase in runner.phase_names(spec):
                folder=self.cell/phase.lower()
                write(folder/'phase_started.json',{'pid':self.pid})
                if self.index!=unknown_at:write(folder/'equivalent_cache_summary.json',{'logical_points':2,'physical_points':1,'actual_numerical_entries':7})
            if self.code==0:write(self.cell/'completion_manifest.json',{'mock':True})
        def poll(self):return self.code
    monkeypatch.setattr(runner.subprocess,'Popen',Process)
    return run,launched


def test_controller_drains_pairs_and_accounts_all_120_phase_costs(tmp_path,monkeypatch):
    run,launched=controller_fixture(tmp_path,monkeypatch)
    assert runner.execute(run)==0 and len(launched)==64
    result=runner.read(run/'controller_completion.json')
    assert result['known_physical_points']==120 and result['known_logical_points']==240
    assert result['known_numerical_entries']==840 and len(result['exits'])==64
    dispatch=rows(run/'controller_dispatch.jsonl')
    for pair in range(1,32):
        assert min(r['started_epoch_seconds'] for r in dispatch if r['pair_order']==pair)>=max(
            r['finished_epoch_seconds'] for r in result['exits'] if r['pair_order']==pair-1)
    with pytest.raises(ValueError,match='Already started'):runner.execute(run)


@pytest.mark.parametrize('kind',['failure','unknown'])
def test_controller_stops_after_first_failed_pair_and_retains_evidence(tmp_path,monkeypatch,kind):
    run,launched=controller_fixture(tmp_path,monkeypatch,fail_at=0 if kind=='failure' else None,unknown_at=0 if kind=='unknown' else None)
    assert runner.execute(run)==1 and len(launched)==2
    result=runner.read(run/'controller_completion.json')
    assert result['status']=='INCONCLUSIVE' and len(result['exits'])==2


def test_complete_sources_models_and_old_physical_inputs_are_available_read_only():
    sources=runner.sealed_sources()
    source,manifest=sources['integrated']
    assert runner.sha(source/'source_cells.json')==manifest['source_cells.json']
    assert len(runner.schedule_for(runner.read(source/'source_cells.json'),runner.read(source/'model_bindings.json')))==64
    assert runner.source_binding()['integrated_completion_sha256']=='461826a8e3f9c415821d42b47c81d66acd8c2fa2d01b09f388dd102e9938ba62'


def test_exact_six_integrated_endpoints_and_source_review_are_copied(tmp_path):
    models=runner.bind_models(tmp_path,runner.sealed_sources())
    assert len(models)==6 and all(runner.sha(tmp_path/m['checkpoint'])==m['source_sha256'] for m in models)
    assert runner.read(tmp_path/'model_preflight/summary.json')['status']=='PASS'


def test_required_source_closure_preserves_audit_models_and_independent_helpers():
    names=runner.source_names()
    assert len(names)==len(set(names))
    assert {'scripts/run_clpu_early_pilot.py','scripts/audit_clpu_early_pilot.py',
        'scripts/report_clpu_early_pilot.py','src/dnr/nature34_clpu_early.py',
        'scripts/audit_nature34_clpu_branch.py','tests/test_nature34_clpu_branch.py',runner.PROTOCOL} <= set(names)
    assert not {'README.md','docs/decisions_and_blockers.md','docs/parameter_register.md'} & set(names)
    assert all((runner.ROOT/name).is_file() for name in names)


@pytest.mark.parametrize('kind',['missing','failure','skipped'])
def test_tests_freeze_requires_all_four_successful_components(tmp_path,kind):
    names=['test_run_clpu_early_pilot','test_report_clpu_early_pilot','test_nature34_clpu_early','test_clpu_early_audit']
    xml='<testsuite>'+''.join(f'<testcase classname="{n}"/>' for n in names)+'</testsuite>'
    path=tmp_path/'tests.xml';path.write_text(xml)
    assert runner.passed_tests(path)==4
    if kind=='missing':xml=xml.replace('test_clpu_early_audit','other')
    else:xml=xml.replace('/>',f'><{kind}/></testcase>',1)
    path.write_text(xml)
    with pytest.raises(ValueError):runner.passed_tests(path)


def test_actual_zero_physics_diagnostic_seal_is_checked_and_copied_without_selection(tmp_path):
    runner.bind_early_diagnostic(tmp_path)
    actual=runner.read(tmp_path/'early_diagnostic_binding.json')
    assert actual['completion_sha256']==runner.EARLY_DIAGNOSTIC_SHA
    assert runner.verify_early_diagnostic(tmp_path/'early_diagnostic')
    summary=runner.read(tmp_path/'early_diagnostic/summary.json')
    assert summary['slots']==128 and summary['new_physical_points']==summary['new_training']==0
    (tmp_path/'early_diagnostic/summary.json').write_text('{}')
    with pytest.raises(ValueError):runner.verify_early_diagnostic(tmp_path/'early_diagnostic')


def test_interrupted_actual_call_cost_is_kept_even_before_logical_certificate(tmp_path):
    spec=runner.schedule_for(contexts(),models())[0]
    phase=tmp_path/spec['cell']/'early'
    write(phase/'phase_started.json',{'started':True})
    write(phase/'equivalent_cache_summary.json',{'logical_points':0,'physical_points':1,'actual_numerical_entries':5})
    cost=runner.costs(tmp_path,[spec])
    assert cost['known_physical_points']==1 and cost['known_logical_points']==0
    assert cost['known_numerical_entries']==5 and not cost['unknown_cost_cells']
    (phase/'equivalent_cache_summary.json').unlink()
    assert runner.costs(tmp_path,[spec])['unknown_cost_cells']==[spec['cell']+'/early']


def test_controller_never_starts_physics_when_storage_forecast_fails(tmp_path,monkeypatch):
    run,launched=controller_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(runner,'storage_forecast',lambda *a:{'free_bytes':1,'required_free_bytes':2})
    assert runner.execute(run)==1 and not launched
    assert runner.read(run/'controller_completion.json')['known_physical_points']==0
