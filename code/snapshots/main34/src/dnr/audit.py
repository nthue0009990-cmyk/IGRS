"""Stage-0 source and numerical checks. No search, training, or safety claims."""
import re
from pathlib import Path

import networkx as nx
import numpy as np
import pandapower as pp
import pandapower.networks as pn


def parse_matpower_source(path: Path):
    source = path.read_text(encoding="utf-8")
    parsed = {}
    for name in ("bus", "branch"):
        match = re.search(r"mpc\." + name + r"\s*=\s*\[(.*?)\];", source, re.S)
        if match is None:
            raise ValueError(f"Missing matrix: {name}")
        rows = []
        for line in match.group(1).splitlines():
            line = line.split("%", 1)[0].strip().rstrip(";")
            if line:
                rows.append([float(value) for value in line.split()])
        parsed[name] = np.array(rows, dtype=float)
    return parsed


def topology_status(net):
    graph = nx.Graph()
    graph.add_nodes_from(net.bus.index.tolist())
    for line in net.line.itertuples():
        if line.in_service:
            graph.add_edge(line.from_bus, line.to_bus)
    return {
        "bus_count": int(len(net.bus)),
        "closed_branch_count": int(net.line.in_service.sum()),
        "connected": bool(nx.is_connected(graph)),
        "radial": bool(nx.is_tree(graph)),
    }


def verify_source_equivalence(net, source):
    bus, branch = source["bus"], source["branch"]
    np.testing.assert_equal(bus[:, 0], np.arange(1, 34))
    np.testing.assert_equal(len(net.bus), 33)
    np.testing.assert_equal(len(net.line), 37)
    np.testing.assert_equal(net.line.from_bus.to_numpy() + 1, branch[:, 0])
    np.testing.assert_equal(net.line.to_bus.to_numpy() + 1, branch[:, 1])
    np.testing.assert_allclose(net.line.r_ohm_per_km * net.line.length_km, branch[:, 2], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(net.line.x_ohm_per_km * net.line.length_km, branch[:, 3], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(net.line.c_nf_per_km, 0, atol=0)
    np.testing.assert_equal(net.line.in_service.to_numpy(), branch[:, 10].astype(bool))
    loads = net.load.groupby("bus")[["p_mw", "q_mvar"]].sum().reindex(net.bus.index, fill_value=0)
    np.testing.assert_allclose(loads.to_numpy(), bus[:, 2:4] / 1000, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(net.bus.vn_kv, bus[:, 9], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(net.bus.max_vm_pu, bus[:, 11], rtol=0, atol=1e-12)
    np.testing.assert_allclose(net.bus.min_vm_pu, bus[:, 12], rtol=0, atol=1e-12)
    return {"matched": True, "total_load_mw": float(loads.p_mw.sum()), "total_load_mvar": float(loads.q_mvar.sum())}


def solve_base(config, algorithm):
    if config["purpose"] != "source_and_powerflow_validation_only" or config["formal_experiment_enabled"]:
        raise ValueError("Stage-0 runner cannot execute formal optimization experiments")
    if config["load_scale"] != 1.0 or config["pv_enabled"] or config["load_model"] != "constant_PQ":
        raise ValueError("This source-validation runner only supports the unmodified base load")
    if algorithm not in config["powerflow"]["algorithms"]:
        raise ValueError("Algorithm is not declared in the audit configuration")
    net = pn.case33bw()
    np.testing.assert_allclose(net.ext_grid.vm_pu, config["slack_vm_pu"], atol=0, rtol=0)
    status = topology_status(net)
    if not (status["connected"] and status["radial"]):
        raise ValueError("Refusing to solve a disconnected or non-radial audit network")
    options = dict(config["powerflow"])
    options.pop("algorithms")
    pp.runpp(net, algorithm=algorithm, check_connectivity=True, **options)
    if not net.converged:
        raise RuntimeError(f"{algorithm} did not converge")
    if not np.isfinite(net.res_bus[["vm_pu", "va_degree"]].to_numpy()).all():
        raise RuntimeError("Non-finite bus result; do not omit unserved buses")
    for column in ("p_from_mw", "q_from_mvar", "p_to_mw", "q_to_mvar", "i_ka"):
        if not np.isfinite(net.res_line.loc[net.line.in_service, column]).all():
            raise RuntimeError(f"Non-finite active branch result: {column}")
    p_error = float(net.res_ext_grid.p_mw.sum() - net.res_load.p_mw.sum() - net.res_line.pl_mw.sum())
    q_error = float(net.res_ext_grid.q_mvar.sum() - net.res_load.q_mvar.sum() - net.res_line.ql_mvar.sum())
    p_residual = net.res_bus.p_mw.to_numpy().copy()
    q_residual = net.res_bus.q_mvar.to_numpy().copy()
    for branch_id, line in net.line.iterrows():
        if not line.in_service:
            continue
        flow = net.res_line.loc[branch_id]
        p_residual[int(line.from_bus)] += flow.p_from_mw
        p_residual[int(line.to_bus)] += flow.p_to_mw
        q_residual[int(line.from_bus)] += flow.q_from_mvar
        q_residual[int(line.to_bus)] += flow.q_to_mvar
    max_nodal_residual = float(max(np.max(np.abs(p_residual)), np.max(np.abs(q_residual))))
    limits = config["verification"]
    if max(abs(p_error), abs(q_error), max_nodal_residual) > limits["power_balance_abs_tolerance_mw_or_mvar"]:
        raise AssertionError(f"Power balance mismatch: P={p_error}, Q={q_error}, nodal={max_nodal_residual}")
    summary = {
        "algorithm": algorithm, "converged": True, **status,
        "loss_mw": float(net.res_line.pl_mw.sum()),
        "reactive_loss_mvar": float(net.res_line.ql_mvar.sum()),
        "v_min_pu": float(net.res_bus.vm_pu.min()),
        "v_min_bus_original_id": int(net.res_bus.vm_pu.idxmin()) + 1,
        "v_max_pu": float(net.res_bus.vm_pu.max()),
        "power_balance_p_error_mw": p_error, "power_balance_q_error_mvar": q_error,
        "max_nodal_power_residual_mw_or_mvar": max_nodal_residual,
        "thermal_feasibility": "unassessed_missing_source_ratings",
        "fully_feasible": None,
    }
    for name, (lo, hi) in config["voltage_profiles_for_audit_only"].items():
        voltage = net.res_bus.vm_pu
        tol = 1e-8
        summary[name] = {
            "bounds_pu": [lo, hi],
            "violating_bus_count": int(((voltage < lo - tol) | (voltage > hi + tol)).sum()),
            "max_undervoltage_pu": float(max(0, lo - voltage.min())),
            "max_overvoltage_pu": float(max(0, voltage.max() - hi)),
        }
    return net, summary
