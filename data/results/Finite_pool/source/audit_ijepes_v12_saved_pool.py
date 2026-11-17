"""Retrospective bounds on the frozen candidate pools; no numerical solver calls."""
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal as D
from functools import lru_cache
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from dnr.nature34_sequence_catalog import state_metadata

SRC = ROOT/'runs/20261031T145450_585540Z_clpu_public_confirmation'
SEAL = '97d9bfa10d41ace924ab1d94ab2c34be9ffe66351ff6b72090e3c6a22556992f'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    manifest_file = SRC/'completion_manifest.json'
    assert sha(manifest_file) == SEAL
    manifest = {k.replace(chr(92),'/'):v for k,v in json.loads(manifest_file.read_bytes()).items()}
    used = {}
    def checked(name):
        digest = sha(SRC/name)
        assert digest == manifest[name], name
        used[name] = digest
        return json.loads((SRC/name).read_bytes())
    run = ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_ijepes_saved_pool')
    run.mkdir()
    shutil.copyfile(__file__,run/'audit.py')
    module=ROOT/'src/dnr/nature34_sequence_catalog.py'
    assert sha(module)==manifest['source/src/dnr/nature34_sequence_catalog.py']
    shutil.copyfile(module,run/module.name)
    save=lambda name,value: (run/name).write_text(json.dumps(value,indent=2),encoding='utf8')
    checked('dependencies.json')
    shutil.copyfile(SRC/'dependencies.json',run/'dependencies.json')
    save('config.json',dict(source=SRC.name,source_seal=SEAL,definition='fixed generated pool plus starting Beam',
                            scope='post hoc, exact primary-energy comparison, only identical full action strings for failed samples',
                            solver_calls=0,seeds='deterministic',script_sha256=sha(Path(__file__))))
    meta=checked('author_static_metadata.json')
    loads=checked('load_reference.json')
    condition=checked('condition.json')
    scenarios={x['scenario_id']:x for x in checked('selected_scenarios.json')}
    schedule=checked('schedule.json')
    rows=[]; details=[]
    power={n:D(str(v['nominal_kw'])) for n,v in loads.items()}
    multiplier=D(str(condition['load_multiplier']))
    @lru_cache(None)
    def kw(fault,cmd):
        bits=[(cmd>>(10+i))&1 for i in range(9)]+[(cmd>>i)&1 for i in range(10)]
        r=state_metadata(meta,loads,scenarios[fault],bits)
        assert r['forest']
        return sum((power[n] for n in r['energized_load_ids']),D(0))
    for i in range(24):
        pair=[s for s in schedule if s['context_id']==i]
        assert len(pair)==2 and {x['arm'] for x in pair}=={'P','E'}
        failed=set(); feasible=set(); pools=[]; final=[]; selected=[]; attested={}
        for s in pair:
            cell=s['cell']
            fault=s['scenario_id']
            r=checked(cell+'/result.json')
            e=checked(cell+'/EARLY/phase_result.json')
            g=checked(cell+'/GUARD/phase_result.json')
            initial=g['best_plan']['validation']['states'][0]['command']
            def score(seq):
                command=initial; total=D(0)
                assert len(seq)==30
                for a in seq:
                    command=command^(1<<(a-1)) if a else command
                    total+=kw(fault,command)
                return total
            cand={tuple(c['actions']):score(c['actions']) for c in e['stage_generation']['candidates']}
            for c in e['stage_generation']['candidates']:
                assert abs(cand[tuple(c['actions'])]*multiplier/60-D(str(c['graph_energy_kwh'])))<D('0.000001')
            pools.append(cand)
            for ph in (e,g):
                for attempt in ph['attempts']:
                    v=attempt.get('validation') or {}
                    seq=tuple(attempt['actions'])
                    if v.get('status')=='FAIL' and v['samples'][-1]['status']=='FAIL':
                        assert seq==tuple(v['actions'])
                        failed.add(seq)
                        attested[seq]=(cell,ph is g,v['failed_step'],v['samples'][-1]['point_id'])
                    if attempt['certified_complete'] and v.get('status')=='COMPLETE':
                        feasible.add(seq)
            best=tuple(r['best_plan']['actions'])
            assert r['fully_certified'] and best in feasible
            selected.append(best);final.append(score(best))
            assert best in cand or best==tuple(g['best_plan']['actions'])
        assert pools[0]==pools[1] and selected[0]==selected[1] and final[0]==final[1]
        assert not failed.intersection(feasible)
        higher={seq:v for seq,v in pools[0].items() if v>final[0]}
        assert not feasible.intersection(higher)
        open_seq=set(higher)-failed
        upper=max([final[0]]+[higher[a] for a in open_seq])
        gap=(upper-final[0])*multiplier/60
        rows.append(dict(context=i,fault=fault,event=pair[0]['event_id'],pool=len(pools[0]),higher=len(higher),
                         higher_failed=len(higher)-len(open_seq),unresolved_higher=len(open_seq),
                         lower_kwh=str(final[0]*multiplier/60),upper_kwh=str(upper*multiplier/60),
                         gap_kwh=str(gap),closed=not open_seq))
        details.extend(dict(context=i,actions=list(a),source=attested.get(a),status='FAIL' if a in failed else 'OPEN')
                       for a in higher)
    with (run/'all24.csv').open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    save('evidence.json',dict(inputs=used,higher=details))
    counts=dict(contexts=len(rows),closed=sum(r['closed'] for r in rows),
                higher=sum(r['higher'] for r in rows),higher_failed=sum(r['higher_failed'] for r in rows),
                open_cases=[r['context'] for r in rows if not r['closed']],
                max_gap_kwh=str(max(D(r['gap_kwh']) for r in rows)))
    save('summary.json',counts)
    print(run, counts)


if __name__=='__main__':
    main()
