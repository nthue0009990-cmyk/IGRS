"""Export case gains and paired operating decisions using only the post-audit files."""
from collections import Counter
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_bytes())


def amount(band,end):
    value=band[end+'_kw_slot_sum']
    return None if value is None else Decimal(value)/60


def prefixes(values):
    return None if values is None else '|'.join(' '.join(map(str,seq)) for seq in values)


def write_csv(path,rows):
    with path.open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main():
    run=Path(sys.argv[1]).resolve()
    audited=run/'independent_audit'
    summary=read(audited/'summary.json')
    assert summary['status'] in ('PASS_COMPLETE','PARTIAL_WITH_UNRESOLVED')
    for name,digest in read(audited/'manifest.json').items():
        assert hashlib.sha256((audited/name).read_bytes()).hexdigest()==digest
    out=run/'tables'
    out.mkdir(exist_ok=False)
    cases,plans,comparisons=[],[],[]
    for i in range(8):
        key=f'c{i:02d}'
        data=read(audited/(key+'.json'))
        catalog=read(run/'catalog'/(key+'.json'))
        rows=data['plans']
        band=data['families']
        all_,base=band['all'],band['switch_only']
        L,U=amount(all_,'lower'),amount(all_,'upper')
        bL,bU=amount(base,'lower'),amount(base,'upper')
        feasible_baseline=bL is not None
        closed=all_['classification']==base['classification']=='ENUMERATED_BOUNDED_OPTIMUM'
        dmin=max(Decimal(0),L-bU) if feasible_baseline and L is not None and bU is not None else None
        dmax=U-bL if feasible_baseline and U is not None else None
        if closed:
            assert dmin==dmax and L==U and bL==bU
        bestfamilies=sorted({r['family'] for r in rows if r['status']=='PASS'
                             and Decimal(r['normal_kw_slot_sum'])/60==L})
        row=dict(case=key,fault=catalog['spec']['scenario_id'],event=catalog['spec']['event_id'],
                 controlled_load=catalog['controlled_load'],**all_['status_counts'],
                 baseline_plans=base['plans'],baseline_pass=base['status_counts']['PASS'],
                 baseline_L_kWh=bL,baseline_U_kWh=bU,allowed_L_kWh=L,allowed_U_kWh=U,
                 gain_L_kWh=dmin,gain_U_kWh=dmax,fully_closed=closed,
                 best_certified_families='|'.join(bestfamilies),
                 baseline_optima=prefixes(base['all_optimal_action_prefixes']),
                 allowed_optima=prefixes(all_['all_optimal_action_prefixes']),
                 physical_points=data.get('physical_points'),logical_points=data.get('logical_points'),
                 numerical_entries=data.get('commands',{}).get('numerical_entries'))
        for fam,label in [('withdrawn_without_reconnection','held'),('withdrawn_then_reconnected','reconnected')]:
            row.update({label+'_plans':band[fam]['plans'],label+'_classification':band[fam]['classification'],
                        label+'_L_kWh':amount(band[fam],'lower'),label+'_U_kWh':amount(band[fam],'upper')})
        cases.append(row)
        indexed={tuple(r['actions_first_5']):r for r in rows}
        for r in rows:
            failure=r.get('first_fail') or {}
            p=tuple(r['actions_first_5'])
            value=Decimal(r['normal_kw_slot_sum'])/60
            plans.append(dict(case=key,index=r['index'],actions=' '.join(map(str,p)),family=r['family'],
                              status=r['status'],normal_kWh_if_feasible=value,certified_kWh=value if r['status']=='PASS' else None,
                              scheduled_target_pickup_s=r['target_first_pickup_s'],checked_samples=r.get('checked_slots_samples'),
                              first_failed_slot=failure.get('slot'),first_failed_time_s=failure.get('time_seconds'),
                              limit_types='|'.join(failure.get('constraints',[]))))
            if closed and dmin>0 and r['status']=='PASS' and value==L:
                other=indexed[tuple(a if a>=16 else 0 for a in p)]
                assert other['family']=='switch_only'
                comparisons.append(dict(case=key,winner=p,family=r['family'],kWh=str(value),
                        same_switch_timing_without_prep=other['actions_first_5'],
                        counterfactual_status=other['status'],counterfactual_first_fail=other.get('first_fail')))
    assert len(cases)==8 and len(plans)==438
    write_csv(out/'case_bounds.csv',cases)
    write_csv(out/'all_438_plans.csv',plans)
    (out/'matched_best.json').write_text(json.dumps(comparisons,indent=2),encoding='utf8')
    print('case,base,joint,gain interval,optimal families,pass/fail/unknown,physical,logical,entries')
    for c in cases:
        print(*[c[k] for k in ('case','baseline_L_kWh','allowed_L_kWh','gain_L_kWh','gain_U_kWh','best_certified_families',
                               'PASS','FAIL','UNRESOLVED','physical_points','logical_points','numerical_entries')],sep=',')
    print('family_status',sorted(Counter((r['family'],r['status']) for r in plans).items()))
    print('matched positive cases',[(c['case'],c['family'],c['counterfactual_status']) for c in comparisons])


if __name__=='__main__':
    main()
