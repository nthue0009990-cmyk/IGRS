"""Bind zero-flow software gates and the exact candidate catalogs before dispatch."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAT = ROOT/'runs/20261106T145040_468073Z_ijepes_bounded_ref_count'
GRAPH = ROOT/'runs/20261106T151203_926536Z_bounded_ref_graph_gate'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_bytes())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('gates',nargs='+',type=Path)
    parser.add_argument('--bind',nargs='*',type=Path,default=[])
    args=parser.parse_args()
    assert digest(CAT/'completion_manifest.json')=='f8fdd3c328d7096d41763ad1d73bd69964f51815b80dbf8b35a36fded6b9a179'
    graph=read(GRAPH/'summary.json')
    assert graph['status']=='PASS' and graph['totals']['full_count']==438 and graph['new_physical_queries']==0
    assert digest(ROOT/'scripts/precheck_bounded_ref_catalog.py')==graph['script_sha256']
    receipts=[]
    files=[GRAPH/'summary.json',GRAPH/'precheck.py',Path(__file__),CAT/'completion_manifest.json',
           ROOT/'docs/proposals/ijepes_bounded_reference_v1.md',*args.bind]
    for gate in args.gates:
        cert=read(gate)
        assert cert['status']=='PASS',gate
        receipts.append(gate.resolve().relative_to(ROOT).as_posix())
        files.append(gate)
    files.extend(p for p in (ROOT/'scripts').glob('*bounded_ref*.py'))
    result=dict(status='PASS',cases=list(range(8)),new_physical_queries=0,
                kind='offline reference only; no policy tasks',
                hash_note='Verified plan population and graph-only receipts before any physical dispatch.',
                receipts=receipts,counts=graph['totals'],
                hashes={p.resolve().relative_to(ROOT).as_posix():digest(p) for p in files})
    path=GRAPH/'preflight.json'
    with path.open('x',encoding='utf8') as out:
        json.dump(result,out,indent=2)
    print(path)


if __name__=='__main__':
    main()
