"""Independent recomputation of scaling selections and scaled test metrics with plain NumPy."""
import hashlib, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent; PARENT = ROOT.parent/'2026-10-07_bounded_hybrid'
read = lambda p: json.loads(p.read_text()); sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
lock, summ, proto = read(ROOT/'selection_lock.json'), read(ROOT/'summary.json'), read(ROOT/'protocol.json')
assert lock['protocol_sha256']==sha(ROOT/'protocol.json')
for p, h in proto['source_sha256'].items(): assert sha(ROOT.parent.parent/p)==h, p
for p, h in lock['delta_sha256'].items(): assert sha(ROOT/'validation_deltas'/p)==h
opts = [tuple(o) for o in proto['options']]; n = 0
def f(b, d, s, m): d = d.astype(np.float64); return np.maximum(b+s*(np.minimum(d, 0) if m=='decrease_only' else d), 0)
for key, c in lock['choices'].items():
    mouse, arm = key.split('_', 1); lower = read(ROOT.parent/'2026-10-06_facemap_validation'/'prepared'/mouse/'metadata.json')['lower']
    grid = []
    for seed in (601, 602, 603):
        with np.load(ROOT/'validation_deltas'/f'{key}_{seed}.npz') as z:
            grid.append([np.mean((f(z['base'], z['delta'], s, m)-z['target'])**2) for s, m in opts])
        with np.load(PARENT/'predictions'/f'{key}_{seed}.npz') as z:
            e = f(z['base'], z['delta'], c['scale'], c['mode'])+lower-z['target']
            r = next(r for r in summ['rows'] if r['mouse']==mouse and r['seed']==seed and r['arm']==arm+'_scaled')
            assert abs(r['mse']-np.mean(e**2))<1e-12 and abs(r['mae']-np.mean(np.abs(e)))<1e-12; n += 1
    np.testing.assert_allclose(grid, c['validation_grid'], rtol=1e-12, atol=1e-14)
    mean = np.mean(grid, 0); assert opts[int(np.flatnonzero(mean<=mean.min()*(1+1e-12))[0])]==(c['scale'], c['mode'])
out = dict(passed=True, selections=len(lock['choices']), test_records=n, hashes_verified=True)
(ROOT/'audit.json').write_text(json.dumps(out, indent=2)+'\n'); print(out)
