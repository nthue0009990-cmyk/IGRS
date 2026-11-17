"""Static provenance/derivation tests; no physical engine or author imports."""
from pathlib import Path
import importlib.util
import unittest
import re

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('build123',ROOT/'scripts/build_nature123_model_v1.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.original,cls.derived,cls.config,cls.manifest=builder.derive()

    def test_single_sign_and_other_bytes_unchanged(self):
        self.assertEqual([n for n in self.original if self.original[n]!=self.derived[n]],['IEEELineCodes.DSS'])
        offset=self.manifest['delta']['byte_offset'];old=self.original['IEEELineCodes.DSS'];new=self.derived['IEEELineCodes.DSS']
        self.assertEqual(new[offset:offset+12],b'-0.585011253')
        self.assertEqual(new[:offset]+new[offset+1:],old)
        with self.assertRaises(ValueError):builder.correct_c4(new)

    def test_author_geometry_preserved(self):
        master=self.derived['IEEE123Master.dss'].decode('utf-8-sig')
        self.assertEqual(self.derived['IEEE123Master.dss'],self.original['IEEE123Master.dss'])
        for name,length in [('L81','0.175'),('L89','0.25')]:
            line=next(l for l in master.splitlines() if re.match(r'New Line\.'+name+r'\s',l,re.I))
            self.assertEqual(re.search(r'Length=([^\s]+)',line,re.I)[1],length)

    def test_full_rating_partition_without_switch_defaults(self):
        limits=self.config['line_phase_ampacity_A'];ideal=set(self.config['ideal_switch_lines'])
        self.assertEqual(len(limits),125);self.assertEqual(ideal,{f'sw{i}' for i in range(1,9)})
        self.assertFalse(ideal & set(limits))
        self.assertEqual({x:list(limits.values()).count(x) for x in set(limits.values())},{530:65,230:55,175:5})
        self.assertIs(self.config['physical_execution_allowed'],False)
        self.assertEqual(self.config['ideal_switch_hardware_thermal_status'],'NOT_MODELED_NOT_A_PASS')

    def test_include_containment_and_hazards(self):
        audit=builder.audit_sources(self.derived)
        self.assertTrue(any(r['target']=='Loadshape_multload.csv' for r in audit['includes']))
        self.assertTrue(any(r['text'].lower().startswith('calcvoltagebases') for r in audit['hazards']))
        bad=dict(self.derived);bad['IEEE123Master.dss']=b'Redirect ../outside.dss\n'
        with self.assertRaises(ValueError):builder.audit_sources(bad)
        bad['IEEE123Master.dss']=b'Redirect missing.dss\n'
        with self.assertRaises(ValueError):builder.audit_sources(bad)

    def test_fingerprint_recomputes(self):
        config=dict(self.config);digest=config.pop('model_fingerprint')
        expected=builder.sha(builder.canonical({'config':config,'derived_source_sha256':self.manifest['derived_source_sha256']}))
        self.assertEqual(digest,expected)

if __name__=='__main__':unittest.main()
