"""Independent audit of holdout selections, locks, test metrics, contrasts and gates using plain NumPy."""
import hashlib
import math
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
SET_B, ALL = ('D3', 'D4', 'D7', 'D9'), ('D3', 'D4', 'D7', 'D9', 'TX60_s2')
SEEDS, ALPHAS = (401, 402, 403), (0., .25, .5, .75, 1.)


def metrics(p, y, lower):
    p = np.maximum(np.asarray(p, dtype=np.float64), lower); e = p-y
    q, a = y-lower<=.05, y-lower>=.5
    return dict(mse=np.mean(e**2), mae=np.mean(np.abs(e)), qfm=np.mean(p[q]-lower) if q.any() else None,
                active_mse=np.mean(e[a]**2) if a.any() else None)


def main():
    lock, summary = read(ROOT/'evaluation_lock.json'), read(ROOT/'summary.json')
    assert lock['amendment_sha256']==sha(ROOT/'amendment.json')
    assert read(ROOT/'amendment.json')['protocol_sha256']==sha(ROOT/'protocol.json')
    assert read(ROOT/'source_lock.json')['holdout_py_sha256']==sha(ROOT/'holdout.py')
    for p, h in lock['checkpoints'].items(): assert sha(ROOT/p)==h, p
    for name, h in lock['base_locks'].items(): assert sha(ROOT/'base_locks'/name)==h
    for rid, h in lock['prepared'].items():
        assert sha(ROOT/'prepared'/rid/'receipt.json')==h
        for name, v in read(ROOT/'prepared'/rid/'receipt.json').items(): assert sha(ROOT/'prepared'/rid/name)==v
    fits = 0
    for fit in sorted((ROOT/'fits').iterdir()):
        result, history = read(fit/'result.json'), read(fit/'history.json')
        with np.load(fit/'validation.npz') as z:
            lower = read(ROOT/'prepared'/result['recording']/'metadata.json')['lower']
            if 'family' in result:
                losses = ((np.maximum(z['prediction'].astype(np.float64), lower)-z['target'][None])**2).mean(1)
            else:
                losses = ((z['prediction']-z['target'][None])**2).mean(1)
            np.testing.assert_allclose(losses, [h['validation_mse'] for h in history], rtol=1e-12, atol=1e-12)
            assert int(np.argmin(losses))==result['selected_epoch']
        assert sha(fit/'selected.pt')==result['sha256']; fits += 1
    rows = summary['rows']
    for rid in ALL:
        b = read(ROOT/'base_locks'/f'{rid}.json')
        assert b['alpha']==ALPHAS[int(np.argmin(np.mean(b['validation_grid'], axis=0)))]
        lower = read(ROOT/'prepared'/rid/'metadata.json')['lower']
        for seed in SEEDS:
            with np.load(ROOT/'cache'/f'{rid}_{seed}.npz') as c, np.load(ROOT/'fits'/f'{rid}_attention_bce_{seed+200}'/'validation.npz') as v:
                np.testing.assert_array_equal(v['prediction'][0], c['base_validation'])
            with np.load(ROOT/'predictions'/f'{rid}_{seed}.npz') as z:
                y = z['target']
                base = (1-b['alpha'])*(np.maximum(z['mlp'].astype(np.float64), lower)-lower)+b['alpha']*(np.maximum(z['transformer'].astype(np.float64), lower)-lower)
                np.testing.assert_array_equal(base, z['base'])
                np.testing.assert_array_equal(z['corrected'], np.maximum(base+z['delta'].astype(np.float64), 0)+lower)
                assert np.abs(z['delta']).max()<=.500001
                for arm, value in [('transformer', z['transformer']), ('mlp', z['mlp']), ('blend', base+lower), ('attention_bce', z['corrected'])]:
                    r = next(r for r in rows if r['recording']==rid and r['seed']==seed and r['arm']==arm)
                    m = metrics(value, y, lower)
                    for k, v in m.items():
                        if v is None: assert r[k] is None
                        else: np.testing.assert_allclose(r[k], v, rtol=1e-12, atol=1e-12)
    mean = lambda rid, arm, k: np.mean([r[k] for r in rows if r['recording']==rid and r['arm']==arm])
    for name, c in summary['contrasts'].items():
        a, b = name.split('_vs_')
        g = [1-mean(r, a, 'mse')/mean(r, b, 'mse') for r in SET_B]
        np.testing.assert_allclose(c['mean_gain'], np.mean(g), atol=1e-12)
        assert c['mouse_wins']==sum(v>0 for v in g)
    gates = {}
    for cand in ('blend', 'attention_bce'):
        ok = True
        for parent in ('transformer', 'mlp'):
            g = [1-mean(r, cand, 'mse')/mean(r, parent, 'mse') for r in SET_B]
            mae = [1-mean(r, cand, 'mae')/mean(r, parent, 'mae') for r in SET_B]
            m = {(r['recording'], r['seed'], r['arm']): r['mse'] for r in rows}
            seeds = sum(m[(r, s, cand)]<m[(r, s, parent)] for r in SET_B for s in SEEDS)
            ok &= np.mean(g)>=.05 and sum(v>0 for v in g)>=math.ceil(6/7*len(SET_B)) and seeds>=math.ceil(2/3*3*len(SET_B)) and max(0, -min(g))<=.1 and np.mean(mae)>=0
        if cand=='attention_bce':
            g = [1-mean(r, cand, 'mse')/mean(r, 'blend', 'mse') for r in SET_B]
            q = lambda r, arm: np.mean([x['qfm'] for x in rows if x['recording']==r and x['arm']==arm and x['qfm'] is not None])
            quiet = sum(q(r, cand)<q(r, 'blend') for r in SET_B)
            act = [mean(r, cand, 'active_mse')/mean(r, 'blend', 'active_mse')-1 for r in SET_B]
            ok &= np.mean(g)>=.05 and quiet>=math.ceil(5/7*len(SET_B)) and all(v<=.05 for v in act)
        gates[cand] = bool(ok)
        assert gates[cand]==summary['gates'][cand]['passed'], cand
    out = dict(passed=True, fits_checked=fits, test_records_checked=len(ALL)*len(SEEDS)*4, contrasts=len(summary['contrasts']),
               gates_recomputed=gates, hashes_verified=True)
    (ROOT/'audit.json').write_text(json.dumps(out, indent=2)+'\n'); print('PASS', out)


if __name__=='__main__':
    main()
