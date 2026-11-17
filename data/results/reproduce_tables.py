"""Reproduce the paired comparisons from paired_outcomes.csv using Python 3."""
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def ratio(a,b):
    return math.fsum(a)/math.fsum(b)


def interval(values):
    ordered=sorted(values)
    def at(p):
        x=(len(ordered)-1)*p
        i=math.floor(x)
        return ordered[i]+(ordered[math.ceil(x)]-ordered[i])*(x-i)
    return [at(.025),at(.975)]


def run():
    with Path(__file__).with_name('paired_outcomes.csv').open(encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==118
    sets=defaultdict(dict)
    for row in rows:
        key=(row['fault'],int(row['event']))
        arms=sets[row['cohort']].setdefault(key,{})
        assert row['arm'] not in arms
        arms[row['arm']]={k:float(row[k]) for k in ['kWh','quality_pct','queries','cold_s','violation_queries']}
    output={}
    for name,pairs in sets.items():
        output[name]={}
        for arm in next(iter(pairs.values())):
            data=[d[arm] for d in pairs.values()]
            output[name][arm]={'n':len(data),'mean_Q_pct':mean(r['quality_pct'] for r in data),
                               'mean_queries':mean(r['queries'] for r in data),'mean_cold_s':mean(r['cold_s'] for r in data)}
    pairs=sets['34_primary'];faults=sorted({f for f,e in pairs});events=sorted({e for f,e in pairs})
    assert len(faults)==4 and len(events)==6 and all(x['P']['kWh']==x['E']['kWh'] for x in pairs.values())
    draws={'queries':[],'cold_s':[]}
    rng=random.Random(20261024)
    for rep in range(10000):
        fc=Counter(faults[rng.randrange(4)] for _ in range(4))
        ec=Counter(events[rng.randrange(6)] for _ in range(6))
        for metric in draws:
            sums={arm:math.fsum(fc[f]*ec[e]*pairs[f,e][arm][metric] for f in faults for e in events) for arm in ('P','E')}
            draws[metric].append(sums['P']/sums['E'])
    output['34_primary']['paired_ratio_95_interval']={k:interval(v) for k,v in draws.items()}
    for k,target in [('queries',[.514,.937]),('cold_s',[.687,.973])]:
        assert [round(x,3) for x in interval(draws[k])]==target
    d=sets['34_direct_dev'];a=sets['34_ablation_dev'];s=sets['34_selector_dev'];t=sets['13_transfer_dev']
    assert [len(data) for data in [d,a,s,t]]==[8,8,4,6]
    assert sum(r['BEAM']['kWh']>r['P']['kWh'] for r in d.values())==1 and sum(r['BEAM']['kWh']==r['P']['kWh'] for r in d.values())==7
    assert all(r['P']['queries']==r['Forward']['queries'] and r['P']['kWh']==r['Forward']['kWh'] for r in [*s.values(),*t.values()])
    assert all(r['P']['kWh']==r['A']['kWh']==r['E']['kWh'] for r in a.values())
    assert sum(r['A']['queries']-r['P']['queries'] for r in a.values())==302
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    run()
