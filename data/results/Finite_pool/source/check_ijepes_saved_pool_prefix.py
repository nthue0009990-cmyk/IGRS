"""Check a post hoc finite-set upper bound against exact saved failing prefixes."""
import csv
from decimal import Decimal as D
import gzip
import json
from pathlib import Path
import shutil
from audit_ijepes_v12_saved_pool import ROOT, SRC, SEAL, sha

RUN=ROOT/'runs/20261106T144935_525067Z_ijepes_saved_pool'


def main():
    manifest_path=SRC/'completion_manifest.json'
    assert sha(manifest_path)==SEAL
    manifest={k.replace(chr(92),'/'):v for k,v in json.loads(manifest_path.read_bytes()).items()}
    def bound(rel):
        path=SRC/rel
        assert sha(path)==manifest[rel],rel
        return path
    def read(rel):
        return json.loads(bound(rel).read_bytes())
    assert read('audit_posthoc_v2/summary.json')['status']=='PASS'
    old={int(x['context']):x for x in csv.DictReader((RUN/'all24.csv').open(encoding='utf8'))}
    schedule=read('schedule.json')
    result=[]; evidence=[]
    for i,base in old.items():
        fails={}; scores={}; selected=None
        for spec in (s for s in schedule if s['context_id']==i):
            assert spec['scenario_id']==base['fault'] and spec['event_id']==int(base['event'])
            cell=spec['cell']
            for phase in ('EARLY','GUARD'):
                filename=cell+'/'+phase+'/logical_points.jsonl.gz'
                with gzip.open(bound(filename),'rt',encoding='utf8') as f:
                    points={p['logical_point_id']:p for line in f if (p:=json.loads(line))['certificate']['status']=='FAIL'}
                phase_result=read(cell+'/'+phase+'/phase_result.json')
                if phase=='EARLY':
                    scores={tuple(c['actions']):D(str(c['graph_energy_kwh'])) for c in phase_result['stage_generation']['candidates']}
                for a in phase_result['attempts']:
                    v=a.get('validation') or {}
                    if v.get('status')!='FAIL':continue
                    sample=v['samples'][-1]; point=points[sample['point_id']]
                    assert v['actions']==a['actions'] and point['certificate']['reliable'] is True
                    assert point['time_seconds']==sample['time_seconds']==v['failed_time_seconds']
                    assert v['failed_state']==point['dynamic_state']==sample['state']
                    assert sample['step']==v['failed_step']+1==point['dynamic_state']['step']
                    assert point['semantic_identity']['scenario_id']==base['fault'] and point['semantic_identity']['event_id']==int(base['event'])
                    prefix=tuple(a['actions'][:sample['step']])
                    fails[prefix]=dict(prefix=prefix,logical_file=filename,logical_id=sample['point_id'],
                                       time_s=sample['time_seconds'],physical_sha=point['physical_evidence_sha256'])
        low=D(base['lower_kwh'])
        # The first-pass audit computed graph scores in exact normal-power units.
        # Take its exact higher-set membership; use recorded graph scores only for upper values.
        high_set={tuple(a['actions']) for a in json.loads((RUN/'evidence.json').read_bytes())['higher'] if a['context']==i}
        assert len(high_set)==int(base['higher'])
        remaining=[]
        for a in sorted(high_set):
            matches=[p for p in fails if a[:len(p)]==p]
            if matches:
                evidence.append(dict(context=i,actions=a,match=fails[min(matches,key=len)]))
            else:
                evidence.append(dict(context=i,actions=a,match=None))
                remaining.append(scores[a])
        upper=max([low]+remaining)
        result.append(dict(context=i,fault=base['fault'],event=base['event'],higher=len(high_set),
                           excluded=len(high_set)-len(remaining),open=len(remaining),lower_kwh=str(low),
                           upper_kwh=str(upper),gap_kwh=str(upper-low),closed=not remaining))
    with (RUN/'prefix_bounds.csv').open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=result[0]);w.writeheader();w.writerows(result)
    out=dict(status='PASS',sources='sealed P/E runs, same case, identical commands up to reliable failing sample',
             scope='post hoc finite generated set plus verified Beam; normal energy only',
             closed=sum(r['closed'] for r in result),higher=sum(r['higher'] for r in result),
             excluded=sum(r['excluded'] for r in result),open=sum(r['open'] for r in result),
             open_cases=[r['context'] for r in result if not r['closed']],new_solves=0)
    for filename,data in [('prefix_evidence.json',evidence),('prefix_summary.json',out)]:
        with (RUN/filename).open('x',encoding='utf8') as f:json.dump(data,f,indent=2)
    shutil.copyfile(__file__,RUN/'check_prefix.py')
    print(out)


if __name__=='__main__':
    main()
