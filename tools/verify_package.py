"""Verify released bytes, without solver or network access."""
from release_common import ROOT,digest,readj,safe
m=readj('manifests/files.json')
for name,entry in m.items():
    p=safe(ROOT,name)
    assert p.is_file() and p.stat().st_size==entry['bytes'],name
    assert digest(p)==entry['sha256'],name
print('PASS release file identities:',len(m),'files;',sum(e['bytes'] for e in m.values()),'bytes')
