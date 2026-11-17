"""Saved-label/accounting check only; Python standard library, no simulator."""
from pathlib import Path
import hashlib,json,math
r=Path(__file__).resolve().parent
j=lambda n:json.loads((r/n).read_text(encoding='utf-8-sig'))
s=j('result_summary.json'); a=j('saved_audit.json');p=j('delivered_plan.json');links=j('selected_sample_links.json');h=j('handoff_selection.json')
assert s['selected_actions']==p['actions']==[22,37]+[0]*28
assert p['status']=='COMPLETE' and p['certified_complete'] and len(links)==len(p['validation']['samples'])==91
assert all(x['status']=='PASS' for x in links)
for x,y in zip(links,p['validation']['samples']):
 assert all(x[k]==y[k] for k in ['phase','step','time_seconds','point_id','request_sha256','status'])
assert math.isclose(s['energy_kwh'],(3210*60+3370*1740)/3600,abs_tol=1e-9)
assert math.isclose(s['gain_kwh'],s['energy_kwh']-1605,abs_tol=1e-9)
assert math.isclose(s['l115_phase_a_peak_a'],max(x['l115_phase_a_max_terminal_current_a'] for x in links),abs_tol=1e-9)
assert s['logical_requests']==s['physical_requests']+s['cache_hits']==349
assert s['numerical_entries']==3*s['physical_requests']==675
assert s['actual_request_outcomes']=={'PASS':205,'FAIL':20,'UNKNOWN':0}
assert h['verify_used']==h['carryover_verify_used']==14 and h['staged_verify_used']==0
assert len(h['selection']['candidates'])==17 and not s['free_pruning_exercised']
assert p['details']['source_branch_ids']==[85] and p['details']['rank']==13
assert s['prior_version_summary']['certified_energy_kwh']==1605 and s['prior_version_summary']['physical_requests']==140
print('PASS IEEE123 compact saved-data checks: 91 linked samples; 225 requests / 675 entries; no new physical evaluations')
