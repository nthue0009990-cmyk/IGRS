from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def readj(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8-sig'))

def safe(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError('Unsafe member path: ' + name)
    return path
