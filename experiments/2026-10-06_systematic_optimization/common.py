"""Shared paths, immutable JSON records and source verification."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
EXP = ROOT.parent
FAIR = EXP/'2026-10-03_fair_comparison'
PANEL = EXP/'2026-10-05_neuron_panel_pilot'/'columns.npy'
MICE = ['MP030', 'MP032', 'MP033', 'MP034']


def read(path):
    return json.loads(path.read_text())


def write(path, obj, replace=False):
    assert replace or not path.exists(), str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def config_id(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


def verify():
    p = read(ROOT/'protocol.json')
    for name, value in p['source_hashes'].items():
        assert digest(PROJECT/name) == value, name
    return p
