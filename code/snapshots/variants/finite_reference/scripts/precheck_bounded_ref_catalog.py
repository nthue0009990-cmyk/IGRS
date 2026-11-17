"""Independent, graph-only check of the frozen 438-plan offline reference."""
from datetime import datetime, timezone
from decimal import Decimal
from functools import lru_cache
import hashlib
from itertools import product
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
CAT = ROOT/'runs/20261106T145040_468073Z_ijepes_bounded_ref_count'
BASE = ROOT/'runs/20261105T024027_869785Z_clpu_direct_beam'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def inspect(cat, static, loads, scenario, rate):
    ties = [frozenset(x) for x in static['switch_edges']]
    edges = [frozenset(x) for x in static['bus_edges']]
    switches = {e:i for i,e in enumerate(ties)}
    broken = {frozenset(static['original_lines'][n]) for n in scenario['fault_lines']}
    start = sum(1<<i for i,e in enumerate(ties[:5]) if e not in broken)
    ordered = sorted(loads)
    bits = {n:1<<i for i,n in enumerate(ordered)}
    dispatch = static['dispatch_loads']
    watts = {n:Decimal(str(l['nominal_kw'])) for n,l in loads.items()}

    @lru_cache(None)
    def topology(mask):
        parents = {n:n for n in static['nodes']}
        def root(n):
            while parents[n]!=n:
                n=parents[n]
            return n
        for edge in edges:
            i=switches.get(edge)
            if edge in broken:
                if i is not None and mask&(1<<i):return None
                continue
            if i is not None and not mask&(1<<i):continue
            a,b=map(root,edge)
            if a==b:return None
            parents[a]=b
        reached={v for v in parents if root(v)==root(static['source_bus'])}
        return (sum(bits[n] for n in ordered if loads[n]['bus'] in reached),
                sum(1<<i for i,e in enumerate(ties) if mask&(1<<i) and e<=reached))

    @lru_cache(None)
    def power(loadmask):
        return sum((watts[n] for n in ordered if loadmask&bits[n]),Decimal(0))

    on0,live0=topology(start)
    def switch(mask,a):
        new=mask|(1<<(a-11))
        if new==mask:return None
        nxt=topology(new)
        old=topology(mask)
        if nxt is None or old[1]&~nxt[1]:return None
        return new

    gained=0
    last={start}
    for _ in range(2):
        next_masks=set()
        for mask in last:
            for a in range(16,20):
                dest=switch(mask,a)
                if dest is None:continue
                reached=topology(dest)[0]
                if topology(mask)[0]&~reached:continue
                gained |= reached&~on0
                next_masks.add(dest)
        last=next_masks
    names=sorted((n for n in dispatch if bits[n]&gained),key=lambda n:(-watts[n],n))
    name=names[0] if names else None
    prep=1+dispatch.index(name) if name else -1
    assert names==cat['eligible_loads'] and name==cat['controlled_load'] and prep==cat['prep_action']
    allowed=(0,16,17,18,19)+((prep,) if name else ())
    records={tuple(row['actions_first_5']):row for row in cat['sorted_candidate_catalog']}
    assert list(records)==sorted(records) and len(records)==len(cat['sorted_candidate_catalog'])
    found=set()
    for seq in product(allowed, repeat=5):
        if sum(a>=16 for a in seq)>2 or seq.count(prep)>2:continue
        m=start; ready=True; served=on0; acc=Decimal(0); pickup=None
        for t,a in enumerate(seq):
            if a==prep:
                if bool(topology(m)[0]&bits[name])==ready:break
                ready=not ready
            elif a:
                n=switch(m,a)
                if n is None:break
                m=n
            after=topology(m)[0]
            if not ready:after &= ~bits[name]
            if served&~after:break
            if name and bits[name]&(after&~served):pickup=60*t
            served=after
            acc+=power(after)
        else:
            found.add(seq)
            row=records[seq]
            acc=(acc+25*power(served))*rate
            end=(m<<10)|(1023 if ready else (1023^(1<<(prep-1))))
            assert Decimal(row['normal_kw_slot_sum'])==acc and row['terminal_command']==end
            assert row['target_first_pickup_s']==pickup
            assert row['transfer_slots']==[[t,a] for t,a in enumerate(seq) if a>=16]
    assert found==records.keys()
    return dict(context_id=cat['spec']['context_id'], full_count=len(found),
                tie_only=sum(prep not in p for p in found),deferred_no_reconnect=sum(p.count(prep)==1 for p in found),
                deferred_and_reconnected=sum(p.count(prep)==2 for p in found))


def main():
    assert sha(CAT/'completion_manifest.json')=='f8fdd3c328d7096d41763ad1d73bd69964f51815b80dbf8b35a36fded6b9a179'
    assert all(sha(CAT/n)==d for n,d in read(CAT/'completion_manifest.json').items())
    cfg=read(CAT/'config.json')
    assert all(sha(BASE/n)==d for n,d in cfg['inputs'].items())
    run=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_bounded_ref_graph_gate')
    run.mkdir()
    shutil.copyfile(__file__,run/'precheck.py')
    shutil.copyfile(CAT/'dependencies.json',run/'dependencies.json')
    static,loads,scenarios,condition=[read(BASE/(n+'.json')) for n in
        ('author_static_metadata','load_reference','selected_scenarios','condition')]
    rows=[]
    for i in range(8):
        cat=read(CAT/f'c{i:02}.json')
        scenario=next(x for x in scenarios if x['scenario_id']==cat['spec']['scenario_id'])
        rows.append(inspect(cat,static,loads,scenario,Decimal(str(condition['load_multiplier']))))
    totals={k:sum(r[k] for r in rows) for k in ('full_count','tie_only','deferred_no_reconnect','deferred_and_reconnected')}
    assert list(totals.values())==[438,103,215,120]
    out=dict(status='PASS',definition='independent union-find, all strings then full legality; exact nominal energies',
             case_rows=rows,totals=totals,seed=0,inputs=cfg['inputs'],catalog_seal=sha(CAT/'completion_manifest.json'),
             script_sha256=sha(Path(__file__)),new_physical_queries=0,solver_calls=0,python=sys.version)
    (run/'summary.json').write_text(json.dumps(out,indent=2),encoding='utf8')
    print(run,totals)


if __name__=='__main__':
    main()
