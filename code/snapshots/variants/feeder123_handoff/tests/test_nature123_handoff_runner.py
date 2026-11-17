"""The handoff revision must not silently change physical accounting or budgets."""
import ast
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


def load(name,filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/filename)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class HandoffRunnerTests(unittest.TestCase):
    def test_accounting_class_identical_to_tested_v1(self):
        nodes=[]
        for filename in ('run_nature123_igrs_acceptance.py','run_nature123_igrs_handoff.py'):
            tree=ast.parse((ROOT/'scripts'/filename).read_text(encoding='utf8'))
            nodes.append(ast.dump(next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PrivateOracle')))
        self.assertEqual(*nodes)

    def test_same_budget_and_distinct_experiment_identity(self):
        old=load('nc123_runner_old','run_nature123_igrs_acceptance.py')
        new=load('nc123_runner_handoff','run_nature123_igrs_handoff.py')
        self.assertEqual(old.LIMITS,new.LIMITS)
        self.assertEqual(old.SEARCH,new.SEARCH)
        self.assertEqual(new.SCOPE,'NC123_IGRS_CANDIDATE_HANDOFF_V2')
        self.assertNotEqual(old.SCOPE,new.SCOPE)


if __name__=='__main__':unittest.main()
