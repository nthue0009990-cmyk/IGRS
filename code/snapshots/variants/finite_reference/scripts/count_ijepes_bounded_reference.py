"""Offline candidate and input counts for the prespecified bounded reference. No solver."""
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from check_chen_joint_graph_gate import BASE, SEAL, read, sha, save
from dnr.chen_joint_domain import ChenEnv
from dnr.nature34_clpu_equivalent_cache import electrical_input_key, sha_value
from dnr.nature34_joint_sequence import JointGraph

SLOTS = 5


def count(spec, data):
    loads, static = data['load_reference'], data['author_static_metadata']
    scenario = next(s for s in data['selected_scenarios'] if s['scenario_id']==spec['scenario_id'])
    event = next(e for e in data['selected_events'] if e['event_id']==spec['event_id'])
    assert scenario['partition']=='build'
    env = ChenEnv(JointGraph(static, loads, scenario), static, loads, scenario, data['condition'], event)
    controlled = static['dispatch_loads']
    def legal_tie_closures(s):
        for a in env.legal_actions(s):
            if a not in range(16,20) or s.command & (1 << (a-1)):
                continue
            after = env.advance(s,a)
            yield a,after

    # Use exactly the same rule in both original development identities. No outcomes read.
    reachable = set()
    for a,s in legal_tie_closures(env.initial):
        reachable.update(env.energized(s.command))
        for b,t in legal_tie_closures(s):
            reachable.update(env.energized(t.command))
    eligible = sorted((n for n in controlled if n in reachable-env.initial_energized),
                      key=lambda n:(-Decimal(str(loads[n]['nominal_kw'])),n))
    if not eligible:
        # Preserve the full domain including legal tie-only or wait-only plans.
        prep, name, bus = -1, None, None
    else:
        name = eligible[0]
        prep = controlled.index(name)+1
        assert env.initial.command & (1 << (prep-1))
        bus = loads[name]['bus']
    terminals=[]

    def visit(s, seq, flips, closures):
        if len(seq)==SLOTS:
            terminals.append(seq)
            return
        visit(env.advance(s,0),seq+(0,),flips,closures)
        if closures<2:
            for a,t in legal_tie_closures(s):
                visit(t,seq+(a,),flips,closures+1)
        if flips<2 and prep in env.legal_actions(s):
            bus_live=bus in set(env.metadata(s.command)['source_reachable_buses'])
            if (flips==0 and not bus_live) or (flips==1 and bus_live):
                visit(env.advance(s,prep),seq+(prep,),flips+1,closures)

    visit(env.initial,(),0,0)
    # Independent enumeration of all multisets/orderings, then checker/replay.
    controls=(0,16,17,18,19)+((prep,) if eligible else ())
    from itertools import product
    verified=[]
    for seq in product(controls,repeat=SLOTS):
        transfers=[a for a in seq if a>=16]
        if len(transfers)>2 or len(set(transfers))!=len(transfers) or seq.count(prep)>2:
            continue
        s=env.initial; valid=True
        for a in seq:
            if a not in env.legal_actions(s): valid=False; break
            t=env.advance(s,a)
            if a==prep:
                live=bus in env.metadata(s.command)['source_reachable_buses']
                if bool(s.command & (1<<(prep-1)))==live: valid=False; break
            if a>=16:
                if s.command & (1<<(a-1)): valid=False; break
            s=t
        if valid: verified.append(seq)
    assert set(terminals)==set(verified) and len(terminals)==len(set(terminals))
    logical,actual=set(),set()
    def position(s,t):
        key=(s.command,s.pickups,t)
        if key not in logical:
            actual.add(sha_value(electrical_input_key(env,s,t,data['physical_scope'])))
            logical.add(key)
    position(env.initial,0)
    kw_cache={}
    def nominal(cmd):
        if cmd not in kw_cache:
            kw_cache[cmd]=sum(Decimal(str(loads[n]['nominal_kw'])) for n in env.energized(cmd))
        return kw_cache[cmd]
    records=[]
    multiplier=Decimal(str(env.condition['load_multiplier']))
    for seq in sorted(terminals):
        states=env.plan_states(env.initial,seq+(0,)*(30-SLOTS))
        for s in states[1:]:
            for offset in (0,30,60):
                position(s,(s.step-1)*60+offset)
        wt=sum(nominal(s.command) for s in states[1:])*multiplier
        records.append(dict(actions_first_5=seq,terminal_command=states[-1].command,
            target_first_pickup_s=states[-1].pickups[env.load_names.index(name)] if eligible else None,
            normal_kw_slot_sum=str(wt),normal_kwh=float(wt/60),
            transfer_slots=[(i,a) for i,a in enumerate(seq) if a>=16]))
    ntied=set(); w=None
    for i,r in enumerate(records):
        if r['target_first_pickup_s'] is None: continue
        for j in range(i):
            q=records[j]
            if (q['terminal_command']==r['terminal_command'] and
                q['transfer_slots']==r['transfer_slots'] and q['target_first_pickup_s'] is not None and
                q['target_first_pickup_s']!=r['target_first_pickup_s']):
                ntied.add((q['terminal_command'],tuple(q['transfer_slots'])))
                if w is None: w=(q,r)
    return dict(spec=spec,eligible_loads=eligible,controlled_load=name,bus=bus,prep_action=prep,
                plan_count=len(records),switch_only=sum(prep not in x for x in terminals),
                both_action_types=sum(prep in x and any(a>=16 for a in x) for x in terminals),
                paid_defer_transfer_reconnect=sum(x.count(prep)==2 for x in terminals),
                terminal_switch_masks=len({r['terminal_command']>>10 for r in records}),
                switch_orders=sorted({tuple(a for a in seq if a>=16) for seq in terminals}),
                distinct_graph_energies=len({Decimal(r['normal_kw_slot_sum']) for r in records}),
                same_switch_timing_groups_with_different_pickup=len(ntied),timing_witness=w,
                requests_upper=1+90*len(records),unique_logical_upper=len(logical),
                unique_actual_upper=len(actual),numerical_entries_upper=3+4*len(actual),
                sorted_candidate_catalog=records)


def main():
    assert sha(BASE/'completion_manifest.json')==SEAL
    manifest = {k.replace(chr(92),'/'):v for k,v in read(BASE/'completion_manifest.json').items()}
    names=('config','load_reference','author_static_metadata','selected_scenarios','condition',
           'selected_events','schedule','physical_scope','dependencies')
    assert all(sha(BASE/(n+'.json'))==manifest[n+'.json'] for n in names)
    data={n:read(BASE/(n+'.json')) for n in names}
    specs=[{k:s[k] for k in ('context_id','event_id','scenario_id')} for s in data['schedule']
           if s['arm']=='P']
    assert [s['context_id'] for s in specs]==list(range(8))
    run=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_ijepes_bounded_ref_count')
    run.mkdir()
    shutil.copyfile(__file__,run/'count.py')
    source_paths=[Path(__file__),ROOT/'scripts/check_chen_joint_graph_gate.py',
        *(ROOT/'src/dnr'/f for f in ('chen_joint_domain.py','nature34_joint_sequence.py',
         'nature34_free_sequence.py','nature34_clpu_environment.py','nature34_clpu_equivalent_cache.py',
         'clpu_curves.py'))]
    save(run/'config.json',dict(seed=0,slots=SLOTS,case_ids=list(range(8)),base=BASE.name,base_sha=SEAL,
         source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in source_paths},
         inputs={n+'.json':manifest[n+'.json'] for n in names},interpreter=sys.executable,
         python=sys.version,physical_queries=0,optimization_calls=0))
    save(run/'dependencies.json',data['dependencies'])
    summary=[]
    for spec in specs:
        row=count(spec,data)
        save(run/f"c{spec['context_id']:02}.json",row)
        summary.append({k:v for k,v in row.items() if k not in ('sorted_candidate_catalog','timing_witness')})
    save(run/'summary.json',summary)
    save(run/'completion_manifest.json',{p.name:sha(p) for p in sorted(run.iterdir()) if p.is_file()})
    print(run,flush=True)
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
