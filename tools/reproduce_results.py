"""Recompute processed-result summaries; no raw archives or physical calls."""
from release_common import ROOT
import subprocess
import sys

results = ROOT / 'data/results'
for rel in ['reproduce_tables.py', 'Finite_pool/check_pool.py',
            'Finite_reference/check_reference.py', 'IEEE123/check_saved.py']:
    script = results / rel
    subprocess.run([sys.executable, str(script)], cwd=script.parent, check=True)
print('PASS processed-result checks; no new power-flow evaluations')
