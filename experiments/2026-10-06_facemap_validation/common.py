"""Local study I/O and immutable protocol access."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent.parent


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def save(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


PLAN=read(ROOT/'protocol.json')
MICE=PLAN['cohort']


def check_sources():
    lock=read(ROOT/'source_lock.json')
    for path,expected in lock['sha256'].items():
        assert digest(REPO/path)==expected,path


def job_name(job):
    return f"{job['regime']}_{job['mouse']}_{job['role']}_s{job['seed']}"
