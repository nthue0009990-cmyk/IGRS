"""No engine: source parsing, fake-call accounting and isolation evidence."""
from copy import deepcopy
import inspect
from types import SimpleNamespace

import networkx as nx
import pytest

from dnr.nature13_clpu_physical import (
    Nature13Budget, Nature13CLPUPhysicalRuntime, PhysicalBudgetStop, _CountedDSS,
    inventory_nature13_commands, validate_multipliers, plan_isolation,
    verify_nature13_isolation, check_nature13_constraints,
)
from dnr.restoration_constraints import check_known_constraints
from dnr.restoration_isolation import verify_isolation


def fake_counter(tmp_path, limit=3, clock=lambda:0):
    (tmp_path/'IEEE13Nodeckt.dss').write_text('calcv\nSolve\n/*\nSolve\n*/\n',encoding='utf-8')
    (tmp_path/'Loadshapes.dss').write_text('! no numerical entry\n',encoding='utf-8')
    inventories = {n.lower():inventory_nature13_commands(tmp_path/n)
                   for n in ['IEEE13Nodeckt.dss','Loadshapes.dss']}
    invoked=[]
    engine=SimpleNamespace(Text=SimpleNamespace(Command=lambda c:invoked.append(c)),
        Solution=SimpleNamespace(Solve=lambda:invoked.append('python Solve'), SolveNoControl=lambda:invoked.append('BAD')))
    budget=Nature13Budget(max_entries=limit,deadline_monotonic=10,clock=clock)
    return _CountedDSS(engine,budget,[],tmp_path,inventories),budget,invoked


def test_calcv_alias_compile_and_python_solve_reserved_before_invocation(tmp_path):
    wrapped,budget,invoked=fake_counter(tmp_path)
    wrapped.Text.Command('CoMpIlE [IEEE13Nodeckt.dss]')
    assert budget.reserved_entries==budget.returned_entries==2
    wrapped.Solution.Solve()
    assert budget.reserved_entries==3
    with pytest.raises(PhysicalBudgetStop):
        wrapped.Solution.Solve()
    assert len(invoked)==2


@pytest.mark.parametrize('command',['solve; solve','solve\nsolve','redirect missing.dss','so','new x\rsolve'])
def test_unknown_or_multicommand_entry_never_invokes_fake_engine(tmp_path,command):
    wrapped,budget,invoked=fake_counter(tmp_path)
    with pytest.raises(ValueError):wrapped.Text.Command(command)
    assert not invoked and budget.reserved_entries==0


def test_alternative_solution_api_is_rejected_without_engine(tmp_path):
    wrapped,_,invoked=fake_counter(tmp_path)
    with pytest.raises(ValueError):wrapped.Solution.SolveNoControl()
    assert not invoked


def test_deadline_query_cap_and_no_retry_are_preexecution():
    now=[0.]
    budget=Nature13Budget(max_queries=1,max_entries=1,deadline_monotonic=1.,clock=lambda:now[0])
    budget.begin_query('q1')
    with pytest.raises(PhysicalBudgetStop):budget.begin_query('q1')
    with pytest.raises(PhysicalBudgetStop):budget.begin_query('q2')
    now[0]=1.
    with pytest.raises(PhysicalBudgetStop):budget.check()


def test_failed_solve_is_charged_and_not_claimed_returned():
    budget=Nature13Budget(max_entries=1,deadline_monotonic=1.,clock=lambda:0.)
    def fail():raise RuntimeError('fake interruption')
    with pytest.raises(RuntimeError):budget.invoke(fail,'Solution.Solve',(),{}, {'numerical_solution_entries':1})
    assert budget.reserved_entries==1 and budget.returned_entries==0
    assert budget.records[-1]['completion']=='UNKNOWN'


def test_full_fifteen_load_readback_request_required():
    reference={f'Load.{i}':{'nominal_kw':1.,'nominal_kvar':.5,'bus':str(i)} for i in range(15)}
    factors={n:1. for n in reference}
    assert len(validate_multipliers(reference,factors))==15
    for bad in ({**factors,'Load.extra':1.}, {n:v for n,v in factors.items() if n!='Load.0'},
                {**factors,'Load.0':.999}, {**factors,'Load.0':True}):
        with pytest.raises(ValueError):validate_multipliers(reference,bad)


def layouts_fixture():
    def layout(kind,buses):
        return {'kind':kind,'buses':[b+'.1' for b in buses], 'conductors':1,
                'terminals':len(buses),'node_order':[1]*len(buses)}
    return {'Vsource.source':layout('vsource',['s']), 'Line.main':layout('line',['s','a']),
            'Line.stub':layout('line',['b','a']), 'Line.deadstub':layout('line',['a','b']),
            'Load.off':layout('load',['b'])}


def isolation_fixture():
    layouts=layouts_fixture()
    graph=nx.Graph();graph.add_nodes_from(['s','a','b']);graph.add_edge('s','a')
    plan=plan_isolation(layouts,graph,'s',{'load.off':1},{'line.stub','line.deadstub'})
    elements=[]
    for name,layout in layouts.items():
        elements.append({'name':name, **{k:deepcopy(layout[k]) for k in ['buses','node_order','conductors','terminals']},
            'enabled':plan['planned_enabled'][name], 'conductor_open':plan['planned_open_contacts'][name],
            'currents_real_imag':[0.]*(2*layout['terminals'])})
    raw={'converged':True,'graph_nodes':['s','a','b'],'graph_edges':[['s','a']],
         'node_names':['s.1','a.1'],'node_vmag_pu':[1.,1.], 'elements':elements}
    meta={'node_phases':{b:[1] for b in graph},'expected_element_names':list(layouts),
        'source_buses':['s'],'allowed_source_elements':['Vsource.source'],'element_layout':layouts,**plan}
    return raw,meta


def test_single_ended_live_stub_is_kept_dead_stub_and_load_are_disabled():
    raw,meta=isolation_fixture()
    assert meta['planned_enabled']['Line.stub'] is True
    assert meta['planned_enabled']['Line.deadstub'] is False
    assert meta['planned_enabled']['Load.off'] is False
    result=verify_nature13_isolation(raw,meta)
    assert result['status']=='PASS'
    assert result['excluded_node_states']=={'b.1':'STRUCTURALLY_EXCLUDED_NOT_SOLVED'}
    assert 'b.1' not in raw['node_names']


def test_closed_live_dead_bridge_is_not_silently_disabled():
    graph=nx.Graph();graph.add_nodes_from(['s','a','b']);graph.add_edge('s','a')
    with pytest.raises(ValueError,match='live/dead'):
        plan_isolation(layouts_fixture(),graph,'s',{}, {'line.stub'})


@pytest.mark.parametrize('tamper',['contact','dead_voltage','disabled_current','inventory'])
def test_invalid_isolation_cannot_pass(tamper):
    raw,meta=isolation_fixture()
    if tamper=='contact':raw['elements'][2]['conductor_open']=[[False],[False]]
    elif tamper=='dead_voltage':raw['node_names'].append('b.1');raw['node_vmag_pu'].append(.2)
    elif tamper=='disabled_current':raw['elements'][-1]['currents_real_imag']=[.1,0.]
    else:raw['elements'].pop()
    assert verify_nature13_isolation(raw,meta)['status']!='PASS'


def test_constraint_binding_preserves_code_defaults_and_all_other_globals():
    assert check_nature13_constraints.__code__ is check_known_constraints.__code__
    assert inspect.signature(check_nature13_constraints)==inspect.signature(check_known_constraints)
    assert check_nature13_constraints.__defaults__==check_known_constraints.__defaults__
    assert check_nature13_constraints.__kwdefaults__==check_known_constraints.__kwdefaults__
    for key,value in check_known_constraints.__globals__.items():
        if key!='verify_isolation':assert check_nature13_constraints.__globals__[key] is value
    assert check_known_constraints.__globals__['verify_isolation'] is verify_isolation
    raw={'converged':False,'graph_nodes':['s'],'graph_edges':[],'node_names':['s.1'],
         'node_vmag_pu':[1.], 'node_voltages_real_imag':[1.,0.], 'elements':[]}
    metadata={'source_buses':['s'],'node_phases':{'s':[1]}}
    assert check_nature13_constraints(raw,metadata)==check_known_constraints(raw,metadata)


def test_partial_evidence_is_reset_even_when_budget_rejects_new_query():
    runtime=object.__new__(Nature13CLPUPhysicalRuntime)
    runtime.last_query_evidence={'query_id':'old','raw_outputs':{'old':True}}
    runtime.budget=Nature13Budget(deadline_monotonic=1.,clock=lambda:2.)
    with pytest.raises(PhysicalBudgetStop):
        runtime.evaluate_point([1,1,0,0]+[1]*7,{'name':'case','fault_lines':[]},.375,0,{},query_id='new')
    assert runtime.last_query_evidence['query_id']=='new'
    assert runtime.last_query_evidence['request']['action']==[1,1,0,0]+[1]*7
    assert 'raw_outputs' not in runtime.last_query_evidence


def test_raw_survives_postsolve_readback_exception_without_swallow_or_retry(monkeypatch):
    import dnr.nature13_clpu_physical as module
    runtime=object.__new__(Nature13CLPUPhysicalRuntime)
    runtime.budget=Nature13Budget(deadline_monotonic=1.,clock=lambda:0.)
    runtime.events=[]
    reference={f'Load.{i}':{'bus':'a','nominal_kw':1.,'nominal_kvar':.5} for i in range(15)}
    runtime.initial_metadata={'load_reference':reference,'element_layout':{},
                              'excluded_controls':[],'generator_reference':{}}
    runtime.base_load_properties={'before':'original'}
    runtime.physical_config={'solver_convergence_tolerance':1e-8}
    fake_calls=[]
    fake=SimpleNamespace(Text=SimpleNamespace(Command=lambda c:fake_calls.append(c)),
        SwtControls=SimpleNamespace(First=lambda:0),
        Solution=SimpleNamespace(Solve=lambda:fake_calls.append('FAKE_ONLY'),Algorithm=lambda x:None,
                                 Convergence=lambda x:None))
    runtime.core=SimpleNamespace(G_init=nx.Graph([('sourcebus','a')]),SwitchLines=[],
        initialize=lambda:(SimpleNamespace(dss=fake),None,None))
    runtime.fault_edges=lambda lines:[]
    captured={'actual_capture':'unchanged','nonfinite_value':float('nan')}
    runtime.extended_snapshot=lambda graph:captured
    reads=[]
    def readback(dss):
        reads.append(dss)
        if len(reads)==1:return deepcopy(runtime.base_load_properties)
        raise RuntimeError('fake postsolve readback failure')
    monkeypatch.setattr(module,'read_load_properties',readback)
    with pytest.raises(RuntimeError,match='postsolve readback failure'):
        runtime.evaluate_point([1,1,0,0]+[1]*7,{'name':'case','fault_lines':[]},.375,0,
                               {n:1. for n in reference},query_id='failed_capture')
    evidence=runtime.last_query_evidence
    assert evidence['raw_outputs'] is captured
    assert evidence['phase']=='raw_acquired' and evidence['evidence_status']=='PARTIAL_NOT_CERTIFIED'
    assert evidence['load_readback']['before']==runtime.base_load_properties
    assert 'metadata_override' in evidence and len(reads)==2 and runtime.budget.queries==1
