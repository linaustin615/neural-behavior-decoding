"""Same-cost single-family pair controls for the holdout blend; frozen rule in control_protocol.json."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
SET_B, SEEDS = ('D3', 'D4', 'D7', 'D9'), (401, 402, 403)
NEXT = {401: 402, 402: 403, 403: 401}


def clip(p, lower): return np.maximum(np.asarray(p, dtype=np.float64), lower)


def main():
    assert not (ROOT/'control_summary.json').exists()
    rows = []
    for rid in SET_B+('TX60_s2',):
        lower = json.loads((ROOT/'prepared'/rid/'metadata.json').read_text())['lower']
        alpha = json.loads((ROOT/'base_locks'/f'{rid}.json').read_text())['alpha']
        pred = {s: dict(np.load(ROOT/'predictions'/f'{rid}_{s}.npz')) for s in SEEDS}
        for s in SEEDS:
            y = pred[s]['target']
            arms = dict(blend=(1-alpha)*clip(pred[s]['mlp'], lower)+alpha*clip(pred[s]['transformer'], lower),
                        transformer_pair=.5*clip(pred[s]['transformer'], lower)+.5*clip(pred[NEXT[s]]['transformer'], lower),
                        mlp_pair=.5*clip(pred[s]['mlp'], lower)+.5*clip(pred[NEXT[s]]['mlp'], lower))
            np.testing.assert_allclose(arms['blend'], pred[s]['base']+lower, rtol=0, atol=1e-12)
            for arm, p in arms.items():
                e = p-y
                rows.append(dict(recording=rid, seed=s, arm=arm, mse=float(np.mean(e**2)), mae=float(np.mean(np.abs(e)))))
    m = {(r['recording'], r['seed'], r['arm']): r for r in rows}
    out = {}
    for ctrl in ('transformer_pair', 'mlp_pair'):
        for recs, tag in ((SET_B, 'set_b'), (('TX60_s2',), 'TX60_s2')):
            g = {r: 1-np.mean([m[(r, s, 'blend')]['mse'] for s in SEEDS])/np.mean([m[(r, s, ctrl)]['mse'] for s in SEEDS]) for r in recs}
            mae = [1-np.mean([m[(r, s, 'blend')]['mae'] for s in SEEDS])/np.mean([m[(r, s, ctrl)]['mae'] for s in SEEDS]) for r in recs]
            seeds = sum(m[(r, s, 'blend')]['mse']<m[(r, s, ctrl)]['mse'] for r in recs for s in SEEDS)
            out[f'blend_vs_{ctrl}_{tag}'] = dict(mean_gain=float(np.mean(list(g.values()))), mouse_wins=int(sum(v>0 for v in g.values())),
                seed_wins=int(seeds), mean_mae_gain=float(np.mean(mae)), per_recording={k: float(v) for k, v in g.items()})
    supported = all(out[f'blend_vs_{c}_set_b']['mean_gain']>0 and out[f'blend_vs_{c}_set_b']['mouse_wins']>=3 and out[f'blend_vs_{c}_set_b']['seed_wins']>=8
                    for c in ('transformer_pair', 'mlp_pair'))
    result = dict(rows=rows, contrasts=out, architecture_mixing_supported=bool(supported))
    (ROOT/'control_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    for k, c in out.items():
        print(f"{k:34s} {100*c['mean_gain']:+6.2f}% wins {c['mouse_wins']} seeds {c['seed_wins']} mae {100*c['mean_mae_gain']:+.2f}%", {r: round(100*v, 2) for r, v in c['per_recording'].items()})
    print('architecture mixing supported:', supported)


if __name__=='__main__':
    main()
