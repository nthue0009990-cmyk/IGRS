"""D019: approved table-defined 69-bus model and physical checks."""
import csv
from pathlib import Path
import numpy as np
import pandapower as pp


def load_tables(directory):
    def read(name):
        with (Path(directory)/name).open(encoding='utf-8', newline='') as f:
            return list(csv.DictReader(f))
    return read('table_A1_nonzero_loads.csv'), read('table_A2_branches.csv'), read('section6_ratings_status.csv')


def build_network(config, case, tables):
    loads, branches, ratings = tables
    assert len(loads)==48 and len(branches)==len(ratings)==73
    assert [int(r['branch_id']) for r in branches]==list(range(1,74))
    assert len(case['open_branches'])==len(set(case['open_branches']))==5
    assert set(case['open_branches']) <= set(range(1,74))
    net=pp.create_empty_network(sn_mva=config['base_mva'], f_hz=50)
    for i in range(1,70):
        pp.create_bus(net, vn_kv=config['base_kv'], index=i, name=f'bus_{i}', min_vm_pu=.95,max_vm_pu=1.05)
    pp.create_ext_grid(net,bus=1,vm_pu=config['slack_vm_pu'],va_degree=config['slack_angle_degree'])
    for r in loads:
        pp.create_load(net,int(r['bus']),p_mw=float(r['p_kw'])*case['scale']/1000,
                       q_mvar=float(r['q_kvar'])*case['scale']/1000,
                       const_z_p_percent=0,const_i_p_percent=0,const_z_q_percent=0,const_i_q_percent=0)
    limits={int(r['branch_id']):float(r['current_limit_a']) for r in ratings}
    for r in branches:
        i=int(r['branch_id'])
        pp.create_line_from_parameters(net,from_bus=int(r['from_bus']),to_bus=int(r['to_bus']),length_km=1,
            r_ohm_per_km=float(r['r_ohm']),x_ohm_per_km=float(r['x_ohm']),c_nf_per_km=0,g_us_per_km=0,
            max_i_ka=limits[i]/1000,index=i,in_service=i not in case['open_branches'],name=f'branch_{i}')
    check_network(net,config,case,tables)
    return net


def check_network(net,config,case,tables):
    loads,branches,ratings=tables
    assert list(net.bus.index)==list(range(1,70)) and net.bus.in_service.all()
    assert len(net.ext_grid)==1 and int(net.ext_grid.iloc[0].bus)==1
    assert net.ext_grid.iloc[0].vm_pu==config['slack_vm_pu'] and net.ext_grid.iloc[0].va_degree==0
    assert (net.bus.vn_kv==12.66).all() and net.sn_mva==100
    assert net.line.in_service.sum()==68
    assert (net.line.c_nf_per_km==0).all() and (net.line.g_us_per_km==0).all()
    assert (net.line.length_km==1).all() and (net.line.parallel==1).all() and (net.line.df==1).all()
    assert (net.load[['const_z_p_percent','const_i_p_percent','const_z_q_percent','const_i_q_percent']]==0).all().all()
    assert len(net.sgen)==len(net.gen)==len(net.shunt)==len(net.trafo)==0
    expected=np.zeros((69,2))
    for r in loads: expected[int(r['bus'])-1]=[float(r['p_kw'])*case['scale']/1000,float(r['q_kvar'])*case['scale']/1000]
    actual=net.load.groupby('bus')[['p_mw','q_mvar']].sum().reindex(range(1,70),fill_value=0).to_numpy()
    np.testing.assert_array_equal(actual,expected)
    adjacency={i:set() for i in range(1,70)}
    for r in branches:
        i=int(r['branch_id']); line=net.line.loc[i]; u,v=int(r['from_bus']),int(r['to_bus'])
        assert (int(line.from_bus),int(line.to_bus))==(u,v)
        assert line.r_ohm_per_km==float(r['r_ohm']) and line.x_ohm_per_km==float(r['x_ohm'])
        limit=400 if i<=9 else 300 if 46<=i<=49 or 52<=i<=64 else 200
        assert line.max_i_ka==limit/1000 and bool(line.in_service)==(i not in case['open_branches'])
        if line.in_service: adjacency[u].add(v); adjacency[v].add(u)
    seen={1}; todo=[1]
    while todo:
        for v in adjacency[todo.pop()]-seen: seen.add(v); todo.append(v)
    assert len(seen)==69, 'Disconnected model'


def branch_physics(v_from_ll_kv, v_to_ll_kv, z_ohm):
    i_ka=(v_from_ll_kv-v_to_ll_kv)/(np.sqrt(3)*z_ohm)
    return np.sqrt(3)*v_from_ll_kv*np.conj(i_ka), -np.sqrt(3)*v_to_ll_kv*np.conj(i_ka), abs(i_ka)


def check_solution(net,config,case,tables):
    assert net.converged
    assert np.isfinite(net.res_bus.to_numpy()).all()
    fields=['p_from_mw','q_from_mvar','p_to_mw','q_to_mvar','pl_mw','ql_mvar','i_from_ka','i_to_ka']
    assert np.isfinite(net.res_line.loc[net.line.in_service,fields].to_numpy()).all()
    assert np.isfinite(net.res_ext_grid[['p_mw','q_mvar']].to_numpy()).all()
    v=net.res_bus.vm_pu.to_numpy()*config['base_kv']*np.exp(1j*np.deg2rad(net.res_bus.va_degree.to_numpy()))
    residual=np.zeros(69,dtype=complex)
    for r in tables[0]: residual[int(r['bus'])-1]+=(float(r['p_kw'])+1j*float(r['q_kvar']))*case['scale']/1000
    ext=complex(net.res_ext_grid.p_mw.sum(),net.res_ext_grid.q_mvar.sum()); residual[0]-=ext
    power_error=0.; current_error=0.; reconstructed_loss=0j
    for r in tables[1]:
        i=int(r['branch_id'])
        if i in case['open_branches']: continue
        a,b=int(r['from_bus'])-1,int(r['to_bus'])-1
        sf,st,ika=branch_physics(v[a],v[b],complex(float(r['r_ohm']),float(r['x_ohm'])))
        actual=net.res_line.loc[i]
        delta=np.array([sf.real-actual.p_from_mw,sf.imag-actual.q_from_mvar,st.real-actual.p_to_mw,st.imag-actual.q_to_mvar])
        power_error=max(power_error,float(np.max(np.abs(delta))))
        current_error=max(current_error,abs(ika-actual.i_from_ka)*1000,abs(ika-actual.i_to_ka)*1000)
        residual[a]+=sf; residual[b]+=st; reconstructed_loss+=sf+st
    balance=float(max(np.max(abs(residual.real)),np.max(abs(residual.imag))))
    loss=float(net.res_line.pl_mw.sum())
    total_error=abs(ext.real-float(net.res_load.p_mw.sum())-loss)
    q_error=abs(ext.imag-float(net.res_load.q_mvar.sum())-float(net.res_line.ql_mvar.sum()))
    assert max(balance,total_error,q_error,power_error,abs(reconstructed_loss.real-loss))<=config['power_residual_abs_tolerance_mw_mvar']
    assert current_error<=config['cross_current_abs_tolerance_a']
    voltage=net.res_bus.vm_pu
    current=net.res_line[['i_from_ka','i_to_ka']].max(axis=1)*1000
    limits=net.line.max_i_ka*1000
    lo,hi=config['voltage_bounds_pu']
    vbad=voltage[(voltage<lo-config['classification_voltage_tolerance_pu'])|(voltage>hi+config['classification_voltage_tolerance_pu'])]
    ibad=current[current>limits+config['classification_current_tolerance_a']]
    metrics = dict(converged=True,topology_valid=True,loss_kw=loss*1000,voltage_compliant=len(vbad)==0,thermal_compliant=len(ibad)==0,
        v_min_pu=float(voltage.min()),v_min_bus=int(voltage.idxmin()),v_max_pu=float(voltage.max()),
        voltage_violating_buses=[int(i) for i in vbad.index],thermal_violating_branches=[int(i) for i in ibad.index],
        max_undervoltage_pu=float(max(0,lo-voltage.min())),max_overvoltage_pu=float(max(0,voltage.max()-hi)),
        max_loading_ratio=float((current/limits).max()),max_current_excess_a=float(max(0,(current-limits).max())),
        max_phasor_power_error_mw_mvar=power_error,max_phasor_current_error_a=float(current_error),
        max_nodal_residual_mw_mvar=balance,total_p_residual_mw=float(total_error),total_q_residual_mvar=float(q_error))
    if 'published_loss_kw' in case:
        metrics.update(published_loss_kw=case['published_loss_kw'],published_loss_delta_kw=loss*1000-case['published_loss_kw'],
            published_loss_rounding_compatible=abs(loss*1000-case['published_loss_kw'])<=config['published_loss_rounding_half_width_kw']+config['published_loss_numeric_tolerance_kw'])
    return metrics
