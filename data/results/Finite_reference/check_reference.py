"""Check the compact finite reference; uses no external package or simulator."""
from pathlib import Path
from decimal import Decimal
from collections import Counter
import csv
import hashlib
import json

HERE = Path(__file__).resolve().parent
def read_json(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))
def read_csv(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))
def decimal(s):
    return None if not s else Decimal(s)
def actions(s):
    return tuple(map(int, s.split()))
def optima(s):
    return set() if not s else {actions(x) for x in s.split('|')}

manifest = read_json(HERE.parent/'Source_manifest_v13.json')
for row in manifest['added_source_files']:
    path = HERE.parent / row['file']
    assert path.stat().st_size == row['bytes'], row['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], row['file']

audit = read_json(HERE/'independent_audit/summary.json')
assert audit['status'] == 'PASS_COMPLETE' and not audit['errors']
cases = read_csv(HERE/'tables/case_bounds.csv')
plans = read_csv(HERE/'tables/all_438_plans.csv')
assert [c['case'] for c in cases] == [f'c{i:02}' for i in range(8)]
assert len(plans) == 438
assert Counter(p['status'] for p in plans) == {'PASS':398, 'FAIL':40}
families = Counter(p['family'] for p in plans)
assert families == {'switch_only':103, 'withdrawn_without_reconnection':215,
                    'withdrawn_then_reconnected':120}, families

for c in cases:
    group = [p for p in plans if p['case'] == c['case']]
    assert len({actions(p['actions']) for p in group}) == len(group)
    catalog = read_json(HERE/'catalog'/f"{c['case']}.json")
    expected = {tuple(x['actions_first_5']): x for x in catalog['sorted_candidate_catalog']}
    assert set(expected) == {actions(p['actions']) for p in group}
    assert c['fault'] == catalog['spec']['scenario_id']
    assert int(c['event']) == catalog['spec']['event_id']
    for p in group:
        e = expected[actions(p['actions'])]
        assert Decimal(e['normal_kw_slot_sum']) / 60 == Decimal(p['normal_kWh_if_feasible'])
        if p['status'] == 'PASS':
            assert decimal(p['certified_kWh']) == decimal(p['normal_kWh_if_feasible'])
            assert int(p['checked_samples']) == 90
        else:
            assert not p['certified_kWh']
            assert p['first_failed_slot'] and p['first_failed_time_s']
            assert p['limit_types']
    for status in ('PASS','FAIL','UNRESOLVED'):
        assert sum(p['status']==status for p in group) == int(c[status])
    for prefix, family in [('baseline','switch_only'),('allowed',None),
                           ('held','withdrawn_without_reconnection'),
                           ('reconnected','withdrawn_then_reconnected')]:
        subset = [p for p in group if family is None or p['family']==family]
        passed = [p for p in subset if p['status']=='PASS']
        unresolved = [p for p in subset if p['status']=='UNRESOLVED']
        lower = max((Decimal(p['certified_kWh']) for p in passed), default=None)
        possible = [Decimal(p['normal_kWh_if_feasible']) for p in unresolved]
        if lower is not None:
            possible.append(lower)
        upper = max(possible, default=None)
        assert lower == decimal(c[prefix+'_L_kWh']), (c['case'],prefix,'L')
        assert upper == decimal(c[prefix+'_U_kWh']), (c['case'],prefix,'U')
        if prefix in ('baseline','allowed'):
            winners = {actions(p['actions']) for p in passed if Decimal(p['certified_kWh']) == lower}
            assert winners == optima(c[prefix+'_optima'])
        else:
            assert len(subset) == int(c[prefix+'_plans'])
    gain = Decimal(c['allowed_L_kWh']) - Decimal(c['baseline_L_kWh'])
    assert gain == Decimal(c['gain_L_kWh']) == Decimal(c['gain_U_kWh'])
    assert c['fully_closed'] == 'True'
    a = next(x for x in audit['cases'] if x['context_id'] == int(c['case'][1:]))
    assert int(c['physical_points']) == a['physical_points']
    assert int(c['logical_points']) == a['logical_points']
    assert int(c['numerical_entries']) == 3 + 4*int(c['physical_points'])

for label,key in [('physical_points','physical_points'),('logical_points','logical_points'),
                  ('numerical_entries','numerical_entries_in_fully_audited_cases')]:
    assert sum(int(c[label]) for c in cases) == audit['totals'][key]
controller = read_json(HERE/'controller_result.json')
config = read_json(HERE/'config.json')
assert len(controller['exits']) == 8 and not controller['stop'] and not controller['not_dispatched']
assert all(e['exit_code']==0 and e['cold_seconds']<=config['wall_case_s'] for e in controller['exits'])
assert len(read_json(HERE/'tables/matched_best.json')) > 0
print(json.dumps({'status':'COMPACT_REFERENCE_CHECK_PASSED', 'cases':8, 'plans':438,
                  'PASS':398, 'FAIL':40, 'UNRESOLVED':0,
                  'physical_evaluations_in_original_study':3716,
                  'new_physical_evaluations':0, 'new_optimization_calls':0}, indent=2))
