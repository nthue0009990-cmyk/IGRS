"""Replay the finite-pool bounds from archived evidence rows. No solver required."""
import csv
import json
from collections import Counter
from decimal import Decimal as D
from pathlib import Path

BASE = Path(__file__).resolve().parent


def rows(name):
    with (BASE/name).open(encoding='utf8', newline='') as f:
        return list(csv.DictReader(f))


def check():
    pool = {int(r['context']):r for r in rows('all24.csv')}
    after = {int(r['context']):r for r in rows('prefix_bounds.csv')}
    prior = json.loads((BASE/'evidence.json').read_bytes())['higher']
    matched = json.loads((BASE/'prefix_evidence.json').read_bytes())
    summary = json.loads((BASE/'prefix_summary.json').read_bytes())
    scores = {(int(r['context']),tuple(json.loads(r['actions']))):D(r['graph_energy_kwh'])
              for r in rows('higher_energy_index.csv')}
    key = lambda v:(v['context'], tuple(v['actions']))
    assert len(prior)==len(matched)==len(scores)==len({key(r) for r in prior})
    assert {key(r) for r in prior}=={key(r) for r in matched}==set(scores)
    m = {key(r):r['match'] for r in matched}
    for r in prior:
        match = m[key(r)]
        if r['status']=='FAIL':
            assert match and r['source']
        if match:
            prefix = match['prefix']
            assert prefix and r['actions'][:len(prefix)]==prefix and len(match['physical_sha'])==64
    open_ids = []
    excluded = 0
    for i, b in after.items():
        group = [r for r in prior if r['context']==i]
        assert int(b['higher'])==len(group)==int(pool[i]['higher'])
        assert int(pool[i]['higher_failed'])==sum(r['status']=='FAIL' for r in group)
        remaining=[scores[key(r)] for r in group if m[key(r)] is None]
        low=D(pool[i]['lower_kwh'])
        up=max([low]+remaining)
        assert (D(b['lower_kwh']),D(b['upper_kwh']),D(b['gap_kwh']))==(low,up,up-low)
        assert int(b['open'])==len(remaining) and int(b['excluded'])==len(group)-len(remaining)
        assert (b['closed']=='True')==(not remaining)
        if remaining: open_ids.append(i)
        excluded+=len(group)-len(remaining)
    assert set(pool)==set(after)==set(range(24))
    assert len(scores)==summary['higher']==160 and excluded==summary['excluded']==138
    assert summary['open']==22 and 24-len(open_ids)==summary['closed']==19
    assert sorted(open_ids)==sorted(summary['open_cases'])
    print(f'24 contexts; 160 higher-energy case/candidate pairs; '
          f'{sum(r["status"]=="FAIL" for r in prior)} full-sequence failures; '
          f'{excluded} excluded after prefix matching; 22 open pairs in cases {sorted(open_ids)}; '
          f'maximum remaining upper-minus-lower gap {max(D(r["gap_kwh"]) for r in after.values()):.4f} kWh.')


if __name__=='__main__':
    check()
