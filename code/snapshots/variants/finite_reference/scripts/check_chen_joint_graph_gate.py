"""Check the two approved development contexts using graph states only."""
import hashlib
import json
import shutil
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dnr.nature34_free_sequence import enumerate_switch_space
from dnr.nature34_clpu_environment import CLPUEnvironment
from dnr.nature34_joint_sequence import JointGraph

BASE = ROOT / "runs/20261105T024027_869785Z_clpu_direct_beam"
SEAL = "2989962bf0e870540c9fb37a07c8e0d31c7cbf951c0eb20e8177127a8a603932"


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    with path.open("x", encoding="utf-8") as out:
        json.dump(data, out, indent=2, allow_nan=False)


def inspect(spec, inputs):
    static, loads = inputs["author_static_metadata"], inputs["load_reference"]
    scenario = next(x for x in inputs["selected_scenarios"] if x["scenario_id"] == spec["scenario_id"])
    event = next(x for x in inputs["selected_events"] if x["event_id"] == spec["event_id"])
    assert scenario["partition"] == "build" and len(loads) == 68
    graph = JointGraph(static, loads, scenario)
    env = CLPUEnvironment(graph, static, loads, scenario, inputs["condition"], event)
    space = enumerate_switch_space(static, loads, scenario)
    switches = static["switch_edges"]
    rows = {x["mask"]: x for x in space["states"].values() if x["forest"] is True}
    power = {m: set(r["source_reachable_buses"]) for m, r in rows.items()}
    live = {m: sum(1 << i for i, (u, v) in enumerate(switches)
                   if m & (1 << i) and u in power[m] and v in power[m]) for m in rows}
    # Exact reachability of joint commands. All remaining pure-graph successor
    # tests depend on the present command, never the numerical pickup ages.
    start = graph.initial_state
    paths = {start: ()}
    q = deque([start])
    masks = set()
    transfers, ready_transfers, denied = set(), set(), set()
    examples = {}
    while q:
        state = q.popleft()
        path = paths[state]
        m = state >> 10
        masks.add(m)
        if len(path) == env.horizon:
            continue
        for dest in graph.successors(state):
            if state == dest:
                continue
            action = (state ^ dest).bit_length()
            n = dest >> 10
            if live[m] & ~live[n]:
                denied.add((m, n))
                continue
            if dest not in paths:
                paths[dest] = path + (action,)
                q.append(dest)
            if m == n or not power[n] - power[m]:
                continue
            transfers.add((m, n))
            if len(path) + 4 > env.horizon:
                continue
            for j, name in enumerate(static["dispatch_loads"]):
                if loads[name]["bus"] not in power[n] - power[m]:
                    continue
                ready_transfers.add(name)
                if not state & (1 << j):
                    continue
                key = (action, name)
                if key not in examples:
                    examples[key] = (path, action, j + 1, name)

    def replay(seq):
        actions = list(seq) + [0] * (env.horizon - len(seq))
        states = env.plan_states(env.initial, actions)
        trace = []
        for step, (a, s, t) in enumerate(zip(actions, states, states[1:])):
            m, n = s.command >> 10, t.command >> 10
            assert not live[m] & ~live[n] and (s.command ^ t.command).bit_count() <= 1
            trace.append(dict(slot=step, action=a, command=t.command, energized_switch_mask=live[n],
                              newly_served=sorted(env.energized(t.command) - env.energized(s.command))))
        return dict(trace=trace, final_command=states[-1].command,
                    final_loads=sorted(env.energized(states[-1].command)),
                    pickup_s=dict(zip(env.load_names, states[-1].pickups)))

    witnesses = []
    for path, a, prep, name in examples.values():
        # Compare early automatic pickup against paid hold-off, transfer,
        # interior WAIT and paid re-enable. Both end at identical commands.
        early = replay(path + (a,))
        late = replay(path + (prep, a, 0, prep))
        assert early["final_command"] == late["final_command"] and early["final_loads"] == late["final_loads"]
        assert late["pickup_s"][name] > early["pickup_s"][name] >= 0
        # One additional replay isolates delayed pickup at identical switch time.
        intermediate = replay(path + (prep, a, prep))
        assert intermediate["pickup_s"][name] + env.slot_seconds == late["pickup_s"][name]
        prop = inputs["source_inputs/base_load_properties"][name]
        witnesses.append(dict(load=name, bus=loads[name]["bus"], phase_count=prop["phases"],
                              terminals=prop["buses"], transfer=switches[a-11],
                              early=early, late=late, no_interior_wait=intermediate))
    return dict(spec=spec, initial=start, reachable_commands=len(paths), switch_masks=sorted(masks),
                transfer_edges=[dict(before=m, after=n, action=(m^n).bit_length()+10,
                                     new_buses=sorted(power[n]-power[m])) for m,n in sorted(transfers)],
                controllable_loads_on_new_buses=sorted(ready_transfers),
                blocked_switch_transitions=sorted(denied), witnesses=witnesses)


def main():
    run = ROOT / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_chen_zero_graph")
    run.mkdir()
    names = ["author_static_metadata", "load_reference", "selected_scenarios", "selected_events",
             "config", "condition", "schedule", "source_inputs/base_load_properties", "dependencies"]
    manifest = {k.replace(chr(92), "/"): v for k,v in read(BASE / "completion_manifest.json").items()}
    assert sha(BASE / "completion_manifest.json") == SEAL
    inputs = {}
    for n in names:
        p = BASE / (n + ".json")
        assert sha(p) == manifest[n + ".json"]
        inputs[n] = read(p)
    assert (inputs["config"]["horizon"], inputs["config"]["slot_seconds"], inputs["config"]["sample_seconds"]) == (30,60,30)
    specs = [s for s in inputs["schedule"] if s["arm"] == "P" and s["context_id"] in (2,7)]
    assert [s["context_id"] for s in specs] == [2,7]
    source_paths = [Path(__file__), ROOT / "docs/proposals/ijepes_joint_action_comparator_v1.md",
                    *(ROOT / ("src/dnr/"+s+".py") for s in ["nature34_clpu_environment",
                    "nature34_joint_sequence", "nature34_free_sequence","nature34_sequence_catalog","clpu_curves"])]
    for p in source_paths:
        rel = p.relative_to(ROOT)
        if rel.parts[0] == "src":
            assert sha(p) == manifest["source/"+rel.as_posix()]
        dest = run / "source" / rel
        dest.parent.mkdir(exist_ok=True,parents=True)
        shutil.copyfile(p,dest)
    save(run / "config.json",dict(base=str(BASE),seal=SEAL,inputs={n:manifest[n+".json"] for n in names},
                                  specs=specs,seed=0,horizon=30,physical_budget=0,backend_budget=0,
                                  dependencies=inputs["dependencies"],python=sys.version))
    for spec in specs:
        outcome = inspect(spec,inputs)
        save(run / f"c{spec['context_id']:02}.json",outcome)
        print(spec['context_id'], outcome['reachable_commands'], outcome['transfer_edges'],
              outcome['controllable_loads_on_new_buses'], len(outcome['witnesses']), flush=True)
    print(run,flush=True)


if __name__ == "__main__":
    main()
