"""Pure-logic regression checks, including exhaustive independent tiny graphs."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('nc123_static', ROOT/'src/dnr/nature123_static_state.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
SparseRestorationGraph = module.SparseRestorationGraph


def tiny():
    return {'nodes':['s','a','b','c'], 'source_bus':'s',
            'bus_edges':[['s','a'],['a','b'],['s','b'],['b','c']],
            'switch_names':['ab','sb'], 'switch_edges':[['a','b'],['s','b']],
            'original_lines':{'sa':['s','a'],'ab':['a','b'],'bc':['b','c']},
            'normally_closed':[1,0], 'dispatch_loads':['bload','cload'],
            'loads':{'aload':{'bus':'a','nominal_kw':10},
                     'bload':{'bus':'b','nominal_kw':20},
                     'cload':{'bus':'c','nominal_kw':30}}}


def reference(model, faults, state):
    # Independent set-edge / DFS-cycle implementation, no production helper.
    ns, nl = len(model['switch_names']), len(model['dispatch_loads'])
    switches, ready = state >> nl, state & ((1 << nl)-1)
    fault_edges = {frozenset(model['original_lines'][n]) for n in faults}
    switch_edges = [frozenset(e) for e in model['switch_edges']]
    fault_closed = any(e in fault_edges and switches & (1 << i) for i,e in enumerate(switch_edges))
    adjacency = {n:set() for n in model['nodes']}
    for edge in model['bus_edges']:
        e = frozenset(edge)
        if e in fault_edges or (e in switch_edges and not switches & (1 << switch_edges.index(e))):
            continue
        a,b = edge
        adjacency[a].add(b); adjacency[b].add(a)
    visited, forest = set(), True
    for root in adjacency:
        if root in visited:
            continue
        stack = [(root,None)]
        while stack:
            n,parent = stack.pop()
            if n in visited:
                forest = False
                continue
            visited.add(n)
            stack += [(other,n) for other in adjacency[n] if other != parent]
    reachable, stack = set(), [model['source_bus']]
    while stack:
        n=stack.pop()
        if n in reachable: continue
        reachable.add(n); stack += list(adjacency[n]-reachable)
    energized = {name for name,row in model['loads'].items() if row['bus'] in reachable and
                 (name not in model['dispatch_loads'] or ready & (1 << model['dispatch_loads'].index(name)))}
    return forest and not fault_closed, energized


class SparseTests(unittest.TestCase):
    def test_exhaustive_reference(self):
        m=tiny()
        for faults in [(),('sa',),('ab',),('bc',),('sa','ab')]:
            graph=SparseRestorationGraph(m,faults)
            for state in range(16):
                valid,energized=reference(m,faults,state)
                meta=graph.metadata(state)
                self.assertEqual(meta.graph_admissible,valid)
                self.assertEqual(meta.energized_loads,energized)
                for action in range(5):
                    target=state if not action else state^(1<<(action-1))
                    after_valid,after=reference(m,faults,target)
                    if valid and after_valid and energized <= after:
                        self.assertEqual(graph.transition(state,action),target)
                    else:
                        with self.assertRaises(ValueError): graph.transition(state,action)

    def test_fault_disconnected_readiness_and_delay(self):
        g=SparseRestorationGraph(tiny(),('sa',))
        initial=g.initial_state
        self.assertEqual(g.metadata(initial).energized_loads,frozenset())
        self.assertEqual(g.metadata(initial).nominal_total_kw,60)
        held=g.transition(initial,1)
        transferred=g.transition(held,4)  # close s-b; b load stays held
        self.assertEqual(g.metadata(transferred).held_loads,frozenset({'bload'}))
        self.assertEqual(g.transition(transferred,0),transferred)
        picked=g.transition(transferred,1)
        self.assertEqual(g.metadata(picked).energized_loads, frozenset(tiny()['loads']))
        with self.assertRaises(ValueError): g.transition(picked,1)
        with self.assertRaises(ValueError): g.transition(picked,4)

    def test_fixed_load_no_reinterruption(self):
        g=SparseRestorationGraph(tiny(),('sa',))
        state=g.transition(g.initial_state,4)
        with self.assertRaises(ValueError): g.transition(state,3) # disconnect fixed a

    def test_invalid_domain_and_manifest(self):
        g=SparseRestorationGraph(tiny())
        for bad in [-1,16,True,1.0,'1']:
            with self.assertRaises(ValueError):g.decode(bad)
        for bad in [-1,5,True,1.0,'1']:
            with self.assertRaises(ValueError):g.raw_transition(0,bad)
        with self.assertRaises(ValueError):SparseRestorationGraph(tiny(),('missing',))
        with self.assertRaises(ValueError):SparseRestorationGraph(tiny(),('sa','sa'))
        m=tiny();m['bus_edges'].append(['a','s'])
        with self.assertRaises(ValueError):SparseRestorationGraph(m)

    def test_sparse_cap_and_reconstruction(self):
        g=SparseRestorationGraph(tiny(),('sa',))
        cap=g.explore(g.initial_state,4,2)
        self.assertTrue(cap['state_cap_reached'])
        self.assertEqual(len(cap['parents']),2)
        full=g.explore(g.initial_state,5,100)
        self.assertTrue(full['complete_through_requested_depth'])
        for target in full['parents']:
            states,actions=g.reconstruct(full['parents'],target)
            for before,after,action in zip(states,states[1:],actions):
                self.assertEqual(g.transition(before,action),after)
        self.assertLessEqual(len(g._topologies),4)


if __name__=='__main__':unittest.main()
