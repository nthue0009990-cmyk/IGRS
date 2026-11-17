"""Approved policy derivation must not rewrite original electrical parameters."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf8'))
def digest(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class PhaseModelTests(unittest.TestCase):
    def test_all_nine_sources_unchanged(self):
        a=ROOT/'artifacts/nature123_model_v1'; b=ROOT/'artifacts/nature123_model_v2'
        manifest=read(b/'manifest.json')
        self.assertEqual(len(manifest['derived_source_sha256']),9)
        for name,h in manifest['derived_source_sha256'].items():
            self.assertEqual((a/'source'/name).read_bytes(),(b/'source'/name).read_bytes())
            self.assertEqual(hashlib.sha256((b/'source'/name).read_bytes()).hexdigest(),h)

    def test_policy_bound_and_limits_unchanged(self):
        a=read(ROOT/'artifacts/nature123_model_v1/model_config.json')
        b=read(ROOT/'artifacts/nature123_model_v2/model_config.json')
        for key in ('line_phase_ampacity_A','ideal_switch_lines','author_lengths_retained','change','not_modeled'):
            self.assertEqual(a[key],b[key])
        self.assertEqual(b['phase_availability_policy'],'ALL_DECLARED_PHASES_SOURCE_REACHABLE_V1')
        self.assertEqual(b['idle_phase_policy'],'CERTIFIED_DISABLED_ONLY_PHASE_SCOPE_V1')
        self.assertEqual(b['parent_model_fingerprint'],a['model_fingerprint'])

    def test_full_configuration_and_policy_have_new_identity(self):
        b=read(ROOT/'artifacts/nature123_model_v2/model_config.json')
        m=read(ROOT/'artifacts/nature123_model_v2/manifest.json')
        self.assertEqual(digest(b),m['config_sha256'])
        fp=b.pop('model_fingerprint')
        self.assertEqual(digest({'config':b,'derived_source_sha256':m['derived_source_sha256']}),fp)
        b['phase_availability_policy']='unsupported'
        self.assertNotEqual(digest({'config':b,'derived_source_sha256':m['derived_source_sha256']}),fp)


if __name__=='__main__':unittest.main()
