"""Verify downloaded holdout recordings against the frozen protocol; reads headers only, never activity or running values."""
import hashlib
import json
from pathlib import Path
import zipfile
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent.parent/'data'/'holdout'


def header(z, name):
    info = z.getinfo(name+'.npy')
    assert info.compress_type==zipfile.ZIP_STORED, name+' must be stored uncompressed'
    with z.open(info) as f:
        version = np.lib.format.read_magic(f)
        shape, _, dtype = np.lib.format._read_array_header(f, version)
    return shape, str(dtype)


def main():
    protocol = json.loads((ROOT/'protocol.json').read_text())
    results, ok = {}, True
    for set_name, s in protocol['sets'].items():
        for r in s['recordings']:
            path = DATA/r['file']
            row = dict(set=set_name, file=r['file'])
            try:
                assert path.exists(), 'missing'
                assert path.stat().st_size==r['uncompressed_bytes'], f"size {path.stat().st_size} != {r['uncompressed_bytes']}"
                crc = 0
                with path.open('rb') as f:
                    for block in iter(lambda: f.read(16*1024*1024), b''): crc = zlib.crc32(block, crc)
                assert crc==r['crc32'], f'crc32 {crc} != {r["crc32"]}'
                with zipfile.ZipFile(path) as z:
                    (neurons, frames), sd = header(z, 'spks')
                    (run_frames,), rd = header(z, 'run')
                assert abs(run_frames-frames)<=1, f'spks frames {frames} vs run {run_frames}'
                assert neurons>=512
                row.update(passed=True, neurons=neurons, frames=frames, run_frames=run_frames, spks_dtype=sd, run_dtype=rd)
            except (AssertionError, KeyError, zipfile.BadZipFile) as error:
                row.update(passed=False, error=repr(error)); ok = False
            results[r['id']] = row
            print(r['id'], 'PASS' if row['passed'] else 'FAIL '+row['error'], flush=True)
    out = dict(passed=ok, protocol_sha256=hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest(),
               values_inspected=False, recordings=results)
    (ROOT/'download_verification.json').write_text(json.dumps(out, indent=2)+'\n')
    print('ALL PASS' if ok else 'FAILURES: do not prepare or fit; report them')


if __name__=='__main__':
    main()
