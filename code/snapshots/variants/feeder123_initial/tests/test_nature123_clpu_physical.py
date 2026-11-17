"""All tests use fake engines or pure saved data. Never import real DSS."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dnr.nature123_acceptance_budget import AcceptanceBudget, AcceptanceBudgetStop
from dnr.nature123_clpu_environment import _hash
from dnr.nature123_clpu_physical import (SourceBundle, Nature123CLPUPhysicalRuntime,
    proposed_physical_config, phase_connectivity, json_safe_evidence)
from dnr.nature34_operating_physical import read_load_properties

ROOT = Path(__file__).resolve().parents[1]
SCHEDULE = ROOT / 'runs/20261107T113530_810940Z_nc123_acceptance_prepared/schedule.json'


class FakeSolution:
    def __init__(self, owner): self.owner = owner; self.tolerance = None
    def Convergence(self, value=None):
        if value is not None: self.tolerance = value
        return self.tolerance
    def Solve(self): self.owner.calls.append('Solve')
    def Converged(self): return True
    def ControlActionsDone(self): return True
    def ControlIterations(self): return 1


class FakeEngine:
    """Records calls, holds explicitly fake setpoint properties; no solver."""
    def __init__(self, request=None):
        self.calls = []; self.active = None; self.short = None; self.enabled = {}
        self.loads = {} if request is None else deepcopy(request['condition']['loads'])
        self.der = {} if request is None else deepcopy(request['condition']['der_setpoints'])
        owner = self
        class Basic:
            def ClearAll(self): owner.calls.append('ClearAll')
            def AllowForms(self, value): owner.calls.append(('AllowForms', value))
        class Text:
            def Command(self, value):
                owner.calls.append(value)
                parts = value.lower().split()
                if parts[0] == 'edit':
                    name = parts[1]
                    for pair in parts[2:]:
                        key, v = pair.split('=')
                        if key == 'enabled': owner.enabled[name] = v == 'yes'
                        elif key in ('kw', 'kvar'):
                            if name.startswith('load.'):
                                owner.loads[name]['nominal_' + key] = float(v)
                            elif name.startswith('generator.'):
                                owner.der[name][key] = float(v)
        class Circuit:
            def SetActiveElement(self, name): owner.active = name.lower()
        class CktElement:
            def BusNames(self): return [owner.loads[owner.active]['bus'] + '.1']
            def NumConductors(self): return 2
            def NumTerminals(self): return 1
            def NumPhases(self): return 1
            def Close(self, t, c): owner.calls.append(('Close', owner.active, t, c))
            def Enabled(self): return owner.enabled.get(owner.active, True)
        class Loads:
            def AllNames(self): return [n.split('.')[1] for n in owner.loads]
            def Name(self, name): owner.short = name
            def kW(self): return owner.loads['load.' + owner.short]['nominal_kw']
            def kvar(self): return owner.loads['load.' + owner.short]['nominal_kvar']
            def Model(self): return 1
            def IsDelta(self): return False
            def kV(self): return 2.4
        class Generators:
            def Name(self, name): owner.short = name
            def kW(self): return owner.der['generator.' + owner.short]['kw']
            def kvar(self): return owner.der['generator.' + owner.short]['kvar']
        self.Basic, self.Text, self.Circuit, self.CktElement = Basic(), Text(), Circuit(), CktElement()
        self.Loads, self.Generators = Loads(), Generators()
        self.Solution = FakeSolution(self)


class BackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schedule = json.loads(SCHEDULE.read_text(encoding='utf8'))
        cls.config = proposed_physical_config(ROOT)

    def runtime(self, engine=None, budget=None):
        return Nature123CLPUPhysicalRuntime(ROOT, budget=budget or AcceptanceBudget(),
            physical_config=self.config, enable_physical=True,
            engine=engine if engine is not None else FakeEngine())

    def test_gate_precedes_engine_import(self):
        with patch('dnr.nature123_clpu_physical.importlib.import_module', side_effect=AssertionError('imported')):
            with self.assertRaises(PermissionError):
                Nature123CLPUPhysicalRuntime(ROOT, budget=AcceptanceBudget(), physical_config=self.config)

    def test_source_identity_and_literal_expansion(self):
        source = SourceBundle(ROOT)
        self.assertEqual(source.inventory['command_triggered_numerical_solution_entries'], 1)
        self.assertEqual(len(source.commands), 48)
        self.assertEqual(sum('New swtcontrol.' in s for s in source.commands), 22)
        self.assertEqual(sum('New Generator.' in s for s in source.commands), 13)
        self.assertEqual(len(source.line_reference), 133)
        self.assertEqual(source.line_reference['l81']['length'], '0.175')
        self.assertEqual(source.line_reference['l89']['length'], '0.25')

    def test_all_prepared_requests_replay_without_solver(self):
        runtime = self.runtime()
        for row in self.schedule:
            runtime.validate_request(row['request'])
        self.assertEqual(runtime.dss.calls, [])
        self.assertEqual(runtime.evidence_kind, 'SIMULATED')

    def test_rehashed_tampered_clpu_request_rejected(self):
        request = deepcopy(self.schedule[-1]['request'])
        name = next(iter(request['loads']))
        request['loads'][name]['requested_kw'] += 1
        request['request_sha256'] = _hash({k: v for k, v in request.items() if k != 'request_sha256'})
        with self.assertRaisesRegex(ValueError, 'CLPU multipliers'):
            self.runtime().validate_request(request)

    def test_real_scope_rejects_proposal_before_numeric_calls(self):
        runtime = self.runtime(); runtime.mock_engine = False
        with self.assertRaises(PermissionError):
            runtime.evaluate(self.schedule[0]['request'], 'p0')
        self.assertEqual(runtime.dss.calls, [])
        self.assertEqual(runtime.budget.reserved_entries, 0)

    def test_reservation_before_underlying_call_and_no_refund(self):
        budget = AcceptanceBudget(max_entries=1)
        runtime = self.runtime(budget=budget)
        def broken():
            self.assertEqual(budget.reserved_entries, 1)
            raise RuntimeError('fake failure')
        with self.assertRaises(RuntimeError): runtime._numeric('fake', broken)
        self.assertEqual(budget.reserved_entries, 1)
        self.assertEqual(budget.returned_entries, 0)
        with self.assertRaises(AcceptanceBudgetStop):
            runtime._numeric('blocked', lambda: self.fail('underlying call happened'))

    def test_source_initialization_counted_with_fake_engine(self):
        runtime = self.runtime()
        runtime._metadata = lambda: {'fake': True}
        runtime._initialize()
        self.assertEqual(runtime.budget.reserved_entries, 2)
        self.assertEqual(runtime.budget.returned_entries, 2)
        self.assertEqual(runtime.dss.calls.count('Solve'), 1)
        self.assertTrue(any(isinstance(c, str) and c.startswith('Compile [') for c in runtime.dss.calls))

    def test_full_setpoints_isolation_controls_and_postsolve_readback(self):
        request = deepcopy(self.schedule[-1]['request'])
        engine = FakeEngine(request); runtime = self.runtime(engine)
        graph, state = runtime.validate_request(request)
        before = {k.lower(): v for k, v in read_load_properties(engine).items()}
        layouts = {name: {'kind': 'load', 'buses': [r['bus'] + '.1'], 'terminals': 1}
                   for name, r in request['condition']['loads'].items()}
        # These two source phase regulators are respectively powered/dead for
        # the selected l19 witness; expect controls to follow actual plan.
        layouts.update({'transformer.reg1a': {'kind': 'transformer', 'buses': ['150', '150r'], 'terminals': 2},
                        'transformer.reg3a': {'kind': 'transformer', 'buses': ['25', '25r'], 'terminals': 2}})
        refs = {n: {'available_p_kw': r['kw'], 'reference_q_kvar': r['kvar'], 'bus': r['bus']}
                for n, r in request['condition']['der_setpoints'].items()}
        layouts.update({n: {'kind': 'generator', 'buses': [r['bus']], 'terminals': 1} for n, r in refs.items()})
        meta = {'load_reference': before, 'generator_reference': refs, 'element_layout': layouts,
            'excluded_controls': ['regcontrol.creg1a', 'regcontrol.creg3a', 'swtcontrol.swsec1'],
            'regulator_targets': {'regcontrol.creg1a': 'transformer.reg1a', 'regcontrol.creg3a': 'transformer.reg3a'}}
        network, observed = runtime._apply(request, graph, state, meta)
        after = runtime._verify_postsolve(request, meta, observed)
        self.assertEqual(len(after['loads']), 91)
        self.assertEqual(len(after['generators']), 13)
        self.assertTrue(after['controls_enabled']['regcontrol.creg1a'])
        self.assertFalse(after['controls_enabled']['swtcontrol.swsec1'])
        for name, row in request['loads'].items():
            self.assertEqual(after['loads'][name]['nominal_kw'], row['requested_kw'])
            self.assertEqual(meta['planned_enabled'][name], row['enabled'])
        engine.loads[next(iter(engine.loads))]['nominal_kw'] += 1
        with self.assertRaisesRegex(ValueError, 'during solve'):
            runtime._verify_postsolve(request, meta, observed)

    def test_unknown_exception_cannot_be_physical_failure(self):
        runtime = self.runtime()
        runtime._initialize = lambda: (_ for _ in ()).throw(ValueError('fake unsupported metadata'))
        result = runtime.evaluate(self.schedule[0]['request'], 'p0')
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertFalse(result['reliable'])
        self.assertFalse(result['physical_certified'])
        self.assertEqual(result['evidence_kind'], 'SIMULATED')
        with self.assertRaises(AcceptanceBudgetStop): runtime.evaluate(self.schedule[0]['request'], 'p0')

    def test_unregistered_tolerance_rejected(self):
        config = deepcopy(self.config); config['solver_convergence_tolerance'] = 1e-3
        with self.assertRaisesRegex(ValueError, 'Unregistered'):
            Nature123CLPUPhysicalRuntime(ROOT, budget=AcceptanceBudget(), physical_config=config,
                                        enable_physical=True, engine=FakeEngine())

    def test_nonfinite_evidence_is_preserved_without_imputation(self):
        value, paths = json_safe_evidence({'current': [float('nan'), float('inf'), -float('inf'), 0.]})
        self.assertEqual(value['current'], ['NaN', '+Infinity', '-Infinity', 0.])
        self.assertEqual(len(paths), 3)
        json.dumps(value, allow_nan=False)


def element(name, buses, order, phases=3):
    return {'name': name, 'enabled': True, 'buses': buses, 'node_order': order,
            'terminals': len(buses), 'conductors': len(order)//len(buses), 'phases': phases}


class PhaseTests(unittest.TestCase):
    def source(self): return element('vsource.source', ['150', '150.0.0.0'], [1,2,3,0,0,0])
    def test_actual_phase_loss_is_not_bus_connectivity_pass(self):
        raw = {'elements': [self.source(), element('line.x', ['150.1','b.1'], [1,1], 1),
                            element('load.x', ['b.2'], [2], 1)]}
        result = phase_connectivity(raw, {'transformer_ratings': {}})
        self.assertEqual(result['status'], 'FAIL')
        self.assertIn('b.2', result['unpowered_enabled_phase_nodes'])
    def test_delta_requires_all_input_phases(self):
        raw = {'elements': [self.source(), element('line.x', ['150.1','a.1'], [1,1], 1),
            element('transformer.x', ['a','b'], [1,2,3,0,1,2,3,0])]}
        meta = {'transformer_ratings': {'transformer.x': {'windings': [{'connection':'delta'}]*2}}}
        result = phase_connectivity(raw, meta)
        self.assertEqual(result['status'], 'FAIL')
        self.assertIn('b.1', result['unpowered_enabled_phase_nodes'])
        raw['elements'][1] = element('line.x', ['150','a'], [1,2,3,1,2,3])
        self.assertEqual(phase_connectivity(raw, meta)['status'], 'PASS')
    def test_unsupported_or_unattached_phase_unknown(self):
        raw = {'elements': [self.source(), element('load.x', ['150.4'], [4], 1)]}
        self.assertEqual(phase_connectivity(raw, {})['status'], 'UNKNOWN')
        raw['elements'].pop()
        meta = {'physical_phase_nodes_by_bus': {'150':[1,2,3], 'ghost':[1]}, 'deenergized_buses': []}
        self.assertEqual(phase_connectivity(raw, meta)['status'], 'UNKNOWN')


if __name__ == '__main__': unittest.main()
