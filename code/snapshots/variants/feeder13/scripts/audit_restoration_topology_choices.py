"""Zero-solve structural audit before choosing the RL development network.

Only parses named OpenDSS PD endpoints and literal switch specifications. It does
not import or execute the author's environment, or evaluate electrical states.
"""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import re
import time

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_case(folder, filename):
    initialization = folder / "DSS_Initialize.py"
    tree = ast.parse(initialization.read_text())
    literal = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("sectional_swt", "tie_swt"):
                literal[node.targets[0].id] = ast.literal_eval(node.value)
    source = folder / filename
    text = re.sub(r"/\*.*?\*/", "", source.read_text(), flags=re.S)
    text = "\n".join(re.split(r"!|//", line)[0] for line in text.splitlines())
    starts = list(re.finditer(r"(?im)^\s*new\s+(?:object=)?([a-z]+\.[^\s]+)", text))
    edges = {}
    for index, start in enumerate(starts):
        kind, name = start.group(1).lower().split(".", 1)
        if kind not in ("line", "transformer"):
            continue
        chunk = text[start.end():starts[index + 1].start() if index + 1 < len(starts) else len(text)]
        if kind == "line":
            buses = [re.search(r"\bbus" + str(i) + r"\s*=\s*([^\s]+)", chunk, re.I).group(1) for i in (1, 2)]
        else:
            group = re.search(r"\bbuses\s*=\s*[\[(]([^\])]+)[\])]", chunk, re.I)
            buses = group.group(1).split() if group else re.findall(r"\bbus\s*=\s*([^\s]+)", chunk, re.I)
        assert len(buses) == 2, (name, buses)
        edges[kind + "." + name] = tuple(bus.strip("\"'").split(".")[0].lower() for bus in buses)
    original_lines = {name: edge for name, edge in edges.items() if name.startswith("line.")}
    switch_edges = [edges["line." + item["line"].lower()] for item in literal["sectional_swt"]]
    for tie in literal["tie_swt"]:
        name = "line." + str(tie.get("name", tie["from node"] + tie["to node"])).lower()
        edge = (tie["from node"].lower(), tie["to node"].lower())
        assert name not in edges
        edges[name] = edge
        switch_edges.append(edge)
    graph = nx.Graph()
    graph.add_edges_from(edges.values())
    assert nx.is_connected(graph)
    base = graph.copy()
    base.remove_edges_from(switch_edges[len(literal["sectional_swt"]):])
    assert nx.is_tree(base)
    return graph, original_lines, switch_edges, {str(source.relative_to(ROOT)): sha(source), str(initialization.relative_to(ROOT)): sha(initialization)}


def enumerate_choices(graph, lines, switch_edges, forced_open):
    patterns = list(itertools.product((0, 1), repeat=len(switch_edges)))
    rows = []
    for name, fault in [(None, None)] + sorted(lines.items()):
        valid = []
        for bits in patterns:
            if any(bits[i] for i in forced_open):
                continue
            if fault is not None and any(bit and frozenset(edge) == frozenset(fault) for bit, edge in zip(bits, switch_edges)):
                continue
            candidate = graph.copy()
            if fault is not None:
                candidate.remove_edge(*fault)
            for bit, edge in zip(bits, switch_edges):
                if not bit and candidate.has_edge(*edge):
                    candidate.remove_edge(*edge)
            if nx.is_tree(candidate):
                valid.append(bits)
        rows.append({"fault_line": name, "grid_connected_forest_switch_count": len(valid), "switch_vectors": valid})
    return rows


def main():
    pointer = json.loads((ROOT / "artifacts/restoration_shortlist_sources_latest.json").read_text())
    source = ROOT / pointer["derived_directory"] / "nature_author_source/RL_Training/Environments"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_restoration_topology_choices"
    run = ROOT / "runs" / stamp
    run.mkdir(parents=True)
    started = time.perf_counter()
    results, hashes = {}, {}
    for size, filename, forced in ((13, "IEEE13Nodeckt.dss", [2]), (34, "ieee34Mod1.dss", [])):
        folder = source / f"DSSdirect_{size}bus_loadandswitching"
        graph, lines, switches, inputs = parse_case(folder, filename)
        hashes.update(inputs)
        rows = enumerate_choices(graph, lines, switches, forced)
        n1 = rows[1:]
        results[str(size)] = {"nodes": len(graph), "edges_with_ties": len(graph.edges), "switches": len(switches),
                              "forced_open_switch_indices": forced, "normal_switch_count": rows[0]["grid_connected_forest_switch_count"],
                              "single_line_fault_count": len(n1), "single_fault_choice_histogram": dict(Counter(row["grid_connected_forest_switch_count"] for row in n1)),
                              "single_faults_with_multiple_choices": sum(row["grid_connected_forest_switch_count"] > 1 for row in n1),
                              "rows": rows}
    # Cross-check the static 13 graph against a previously saved physical snapshot.
    physical = ROOT / "runs/20261019T021340_566784Z_nature13_curtailment_validation/metadata.json"
    meta = json.loads(physical.read_text())
    assert set(meta["node_phases"]) == set(parse_case(source / "DSSdirect_13bus_loadandswitching", "IEEE13Nodeckt.dss")[0])
    result = {"status": "STRUCTURAL_AUDIT_ONLY", "run": stamp, "source_commit": pointer["nature_commit"],
              "networks": results, "input_hashes": hashes, "new_powerflow_calls": 0, "rl_training_steps": 0,
              "formal_holdout_numerical_access": False, "physical_feasibility_inferred": False,
              "elapsed_seconds": time.perf_counter() - started}
    (run / "audit.py").write_bytes(Path(__file__).read_bytes())
    (run / "results.json").write_text(json.dumps(result, indent=2))
    (run / "completion_manifest.json").write_text(json.dumps({p.name: sha(p) for p in run.iterdir() if p.is_file()}, indent=2))
    (ROOT / "artifacts/restoration_topology_choices_latest.json").write_text(json.dumps({"run": stamp, "status": result["status"], "completion_sha256": sha(run / "completion_manifest.json")}, indent=2))
    print(json.dumps({**{k: v for k, v in result.items() if k != "networks"}, "networks": {k: {key: value for key, value in data.items() if key != "rows"} for k, data in results.items()}}))


if __name__ == "__main__":
    main()
