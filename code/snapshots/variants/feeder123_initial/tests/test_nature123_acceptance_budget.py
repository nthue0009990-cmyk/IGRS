"""Boundary and failure-path tests using a fake clock, never a physical engine."""
import importlib.util
from pathlib import Path
import unittest

p = Path(__file__).resolve().parents[1] / 'src/dnr/nature123_acceptance_budget.py'
spec = importlib.util.spec_from_file_location('budget123', p)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Budget, Stop = module.AcceptanceBudget, module.AcceptanceBudgetStop


class BudgetTests(unittest.TestCase):
    def test_initialization_and_failure_are_charged(self):
        b = Budget(max_entries=3, clock=lambda: 0.)
        a = b.reserve_entry(operation='Compile containing numerical commands', count=2)
        b.finish_entry(a, returned=False)
        b.begin_request('q0', 'a'*64)
        a = b.reserve_entry(operation='Solve', count=1, request_id='q0')
        b.finish_entry(a, returned=True)
        with self.assertRaises(Stop): b.reserve_entry(operation='Solve', count=1, request_id='q0')
        self.assertEqual((b.reserved_entries, b.returned_entries), (3, 1))
        with self.assertRaises(ValueError): b.finish_entry(a, returned=True)

    def test_exact_request_limit_and_no_implicit_retry(self):
        b = Budget(clock=lambda: 0.)
        for i in range(16): b.begin_request(str(i), 'b'*64)
        with self.assertRaises(Stop): b.begin_request('16', 'b'*64)
        with self.assertRaises(Stop): b.begin_request('0', 'b'*64)
        self.assertEqual(len(b.requests), 16)

    def test_deadline_is_pre_call_and_completion_is_not_convergence(self):
        t = [10.]
        b = Budget(seconds=240, clock=lambda: t[0])
        token = b.reserve_entry(operation='Solve', count=1)
        t[0] = 250.
        with self.assertRaises(Stop): b.begin_request('q0', 'c'*64)
        b.finish_entry(token, returned=True)
        self.assertEqual(b.entries[token]['completion'], 'API_RETURNED_NOT_CONVERGENCE')
        self.assertFalse(b.snapshot()['physical_certificate_created'])

    def test_invalid_cost_clock_and_request_binding(self):
        for bad in [True, 0, -1, 1.5]:
            with self.assertRaises(ValueError): Budget(max_entries=bad)
        t = [10.]
        b = Budget(clock=lambda: t[0])
        for bad in [True, 0, -1, 1.5]:
            with self.assertRaises(ValueError): b.reserve_entry(operation='Solve', count=bad)
        with self.assertRaises(ValueError): b.reserve_entry(operation='Solve', count=1, request_id='missing')
        with self.assertRaises(ValueError): b.begin_request('q0', 'not-a-hash')
        t[0] = 9.
        with self.assertRaises(Stop): b.check()
        t[0] = float('nan')
        with self.assertRaises(Stop): b.check()

    def test_log_failure_never_refunds_reservation(self):
        def broken(row): raise OSError('synthetic log failure')
        b = Budget(clock=lambda: 0., emit=broken)
        with self.assertRaises(OSError): b.begin_request('q0', 'd'*64)
        self.assertEqual(len(b.requests), 1)
        with self.assertRaises(OSError): b.reserve_entry(operation='Solve', count=2)
        self.assertEqual(b.reserved_entries, 2)
        self.assertEqual(b.snapshot()['unreported_entries'], 2)


if __name__ == '__main__': unittest.main()
