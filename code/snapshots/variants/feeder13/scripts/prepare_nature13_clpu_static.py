"""Prepare the authorized six NC13 pure graph/history/staged pools; no DSS."""
from __future__ import annotations
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from audit_restoration_topology_choices import parse_case
from dnr.nature13_clpu_graph import JointGraph
from dnr.nature13_clpu_environment import CLPUEnvironment
from dnr.nature13_clpu_staged_pickup import generate_staged_pickup_candidates
from dnr.nature13_clpu_search import graph_plan

OUT = ROOT / 'artifacts/nature13_clpu_adaptation_20261101/static'
EVENTS = ROOT / 'runs/20261024T022803_301538Z_nature34_clpu_calibration/selected_events.json'
EVENTS_SHA = '17910ef862389984022d49761e65641d77fb46102416b114c8a0681cca540eee'
FAULTS = ('line.632633', 'line.632670', 'line.670671')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def load_inputs():
    pointer = read(ROOT / 'artifacts/restoration_shortlist_sources_latest.json')
    folder = ROOT / pointer['derived_directory'] / 'nature_author_source/RL_Training/Environments/DSSdirect_13bus_loadandswitching'
    graph, originals, switches, hashes = parse_case(folder, 'IEEE13Nodeckt.dss')
    old = read(ROOT / 'runs/20261019T022218_959716Z_restoration_topology_choices/results.json')
    if any(old['input_hashes'][name] != value for name, value in hashes.items()):
        raise ValueError('Original source no longer matches saved structural audit')
    literals = {}
    for node in ast.parse((folder / 'DSS_Initialize.py').read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ('dispatch_loads', 'substatn_id'):
                literals[node.targets[0].id] = ast.literal_eval(node.value)
    text = re.sub(r'/\*.*?\*/', '', (folder / 'IEEE13Nodeckt.dss').read_text(encoding='utf-8'), flags=re.S)
    text = '\n'.join(re.split(r'!|//', line)[0] for line in text.splitlines())
    loads = {}
    for match in re.finditer(r'(?im)^\s*new\s+load\.([^\s]+)\s+([^\n]+)', text):
        name, row = 'Load.' + match.group(1), match.group(2)
        def field(key):
            return re.search(r'\b' + key + r'\s*=\s*([^\s]+)', row, re.I).group(1)
        loads[name] = {'bus': field('bus1').split('.')[0].lower(),
                       'nominal_kw': float(field('kw')), 'nominal_kvar': float(field('kvar'))}
    if len(loads) != 15 or sum(row['nominal_kw'] for row in loads.values()) != 3466:
        raise ValueError('Complete original 15-load ledger required')
    static = {'network_id': 'nature13_author_complete_tie1_open', 'source_bus': literals['substatn_id'],
              'nodes': sorted(graph.nodes), 'bus_edges': [list(edge) for edge in graph.edges],
              'original_lines': originals, 'switch_edges': switches,
              'dispatch_loads': literals['dispatch_loads'],
              'all_load_buses': {name: row['bus'] for name, row in loads.items()},
              'dispatch_load_buses': {name: loads[name]['bus'] for name in literals['dispatch_loads']},
              'initial_switch_bits': [1, 1, 0, 0], 'forced_open_switch_indices': [2]}
    condition = {'condition_id': 'author13_nominal_v1', 'load_multiplier': 1.0,
                 'denominator_nominal_kw': 3466.0,
                 'total_demand_kvar': sum(row['nominal_kvar'] for row in loads.values()),
                 'loads': [{'name': name, **row} for name, row in loads.items()]}
    if sha(EVENTS) != EVENTS_SHA:
        raise ValueError('Frozen development event source hash differs')
    events = [row for row in read(EVENTS) if row['event_id'] in (3, 6)]
    if sorted(row['event_id'] for row in events) != [3, 6]:
        raise ValueError('Both authorized original development events are required')
    scenarios = [{'scenario_id': 'nc13_' + fault.replace('line.', ''), 'fault_lines': [fault],
                  'partition': 'build', 'selection_reason': 'prior zero-solve graph witness; not unseen test'} for fault in FAULTS]
    hashes[str(EVENTS.relative_to(ROOT))] = EVENTS_SHA
    return static, loads, condition, scenarios, events, hashes


def make_env(static, loads, condition, scenario, event):
    return CLPUEnvironment(JointGraph(static, loads, scenario), static, loads, scenario, condition, event)


def build_environment(project, fault_line, event_id):
    """Read only the authorized sources and construct a pure in-memory context."""
    if Path(project).resolve() != ROOT.resolve():
        raise ValueError('Environment must use the bound project source inventory')
    static, loads, condition, scenarios, events, _ = load_inputs()
    scenario = next((row for row in scenarios if row['fault_lines'] == [fault_line]), None)
    event = next((row for row in events if row['event_id'] == event_id), None)
    if scenario is None or event is None:
        raise ValueError('Only three authorized development faults and events3/6 are supported')
    return make_env(static, loads, condition, scenario, event)


def main():
    static, loads, condition, scenarios, events, hashes = load_inputs()
    for name, value in [('author_static_metadata.json', static), ('load_reference.json', loads),
                        ('condition.json', condition), ('scenarios.json', scenarios), ('selected_events.json', events)]:
        save(OUT / name, value)
    summary = []
    for scenario in scenarios:
        for event in sorted(events, key=lambda row: row['event_id']):
            env = make_env(static, loads, condition, scenario, event)
            pool = generate_staged_pickup_candidates(env)
            if pool['counts']['unique_legal_candidates'] > 24:
                raise AssertionError('Original pool cap exceeded')
            for candidate in pool['candidates']:
                info = graph_plan(env, env.initial, candidate['actions'])
                for before, after in zip(info['states'], info['states'][1:]):
                    if not env.energized(before.command) <= env.energized(after.command):
                        raise AssertionError('Candidate interrupts an energized load')
            peak = env.advance(env.initial, 11)
            context_id = scenario['scenario_id'] + '_e' + str(event['event_id'])
            value = {'context_id': context_id, 'scenario_id': scenario['scenario_id'], 'event_id': event['event_id'],
                     'graph_summary': env.graph.graph_summary, 'initial_state': env.initial.to_dict(),
                     'initial_command_bits': env.metadata(env.initial.command)['command'],
                     'initial_request': env.snapshot_request(env.initial, 0),
                     'tie2_peak_state': peak.to_dict(), 'tie2_peak_request': env.snapshot_request(peak, 0),
                     'candidate_generation': pool, 'physical_evaluations': 0, 'physical_certificates': 0}
            save(OUT / 'contexts' / (context_id + '.json'), value)
            summary.append({'context_id': context_id, 'fault_lines': scenario['fault_lines'], 'event_id': event['event_id'],
                            'counts': pool['counts'], 'gap_slots': pool['gap_slots'],
                            'reachable_command_states': env.graph.graph_summary['reachable_command_states']})
    summary_status = 'STATIC_NONEMPTY_ORIGINAL_POOLS' if any(row['counts']['unique_legal_candidates'] for row in summary) else 'STOP_ALL_ORIGINAL_POOLS_EMPTY'
    save(OUT / 'summary.json', {'status': summary_status, 'contexts': summary, 'new_physical_queries': 0,
                               'new_training_steps': 0, 'formal_test': False,
                               'physical_feasibility_inferred': False, 'input_hashes': hashes})
    save(OUT / 'interface_schema.json', {'state': '(four switch mask << 7) | seven readiness bits',
        'action_vector': 'four switches then seven logical intent readiness bits; exact supplied request',
        'effective_action_vector': 'derived four switches then seven effective readiness bits; unreachable readiness is zero',
        'command_vector': 'four switches then seven intent readiness bits; off-grid readiness retained',
        'action_ids': {'wait': 0, 'readiness': [1, 7], 'switches': [8, 11], 'forbidden_tie1': 10},
        'load_inventory': 'exact 15 original Load.* identities; normal denominator3466 kW',
        'history': 'pickup=-1 initially steady, None unserved, boundary-seconds on first actual energization',
        'snapshot_request_fields': ['action', 'effective_action', 'case', 'alpha', 'algorithm', 'multipliers', 'time_minutes', 'state_identity'],
        'request_label': 'UNASSESSED only; never a physical certificate', 'physical_queries_permitted': 0})
    print(json.dumps({'status': summary_status, 'contexts': summary}, ensure_ascii=False))


if __name__ == '__main__':
    main()
