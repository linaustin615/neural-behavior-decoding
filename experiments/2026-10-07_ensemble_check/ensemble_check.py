"""Plain seed-ensemble versus single model on development and holdout test predictions; rule in protocol.json."""
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent
read = lambda p: json.loads(Path(p).read_text())
SEEDS = (401, 402, 403)
NEXT = {401: 402, 402: 403, 403: 401}
DEV = ('TX103', 'TX104', 'TX56', 'TX57', 'TX60', 'TX61', 'VR2')
HOLD = ('D3', 'D4', 'D7', 'D9')


def load_dev(mouse, family):
    lower = read(EXP/'2026-10-06_facemap_validation'/'prepared'/mouse/'metadata.json')['lower']
    out = {}
    for s in SEEDS:
        with np.load(EXP/'2026-10-06_frozen_retrieval'/'predictions'/f'{mouse}_{family}_{s}.npz') as z:
            out[s] = (z['parent'], z['target'])
    return out, lower


def load_hold(rid, family):
    lower = read(EXP/'2026-10-07_holdout_confirmation'/'prepared'/rid/'metadata.json')['lower']
    out = {}
    for s in SEEDS:
        with np.load(EXP/'2026-10-07_holdout_confirmation'/'predictions'/f'{rid}_{s}.npz') as z:
            out[s] = (z[family], z['target'])
    return out, lower


def stats(p, y, lower):
    e = p-y; q, a = y-lower<=.05, y-lower>=.5
    return dict(mse=float(np.mean(e**2)), mae=float(np.mean(np.abs(e))),
                quiet_pred=float(np.mean(p[q]-lower)) if q.any() else None, active_mse=float(np.mean(e[a]**2)) if a.any() else None)


def cohort(names, loader, audited):
    result = {}
    for family in ('transformer', 'mlp'):
        per, slots = {}, 0
        for name in names:
            preds, lower = loader(name, family)
            clip = {s: np.maximum(preds[s][0].astype(np.float64), lower) for s in SEEDS}
            y = preds[SEEDS[0]][1]
            single = {s: stats(clip[s], y, lower) for s in SEEDS}
            for s in SEEDS: np.testing.assert_allclose(single[s]['mse'], audited(name, s, family), rtol=1e-12, atol=1e-12)
            pair = {s: stats(.5*clip[s]+.5*clip[NEXT[s]], y, lower) for s in SEEDS}
            triple = stats(sum(clip.values())/3, y, lower)
            mean = lambda d, k: np.mean([d[s][k] for s in SEEDS if d[s][k] is not None]) if any(d[s][k] is not None for s in SEEDS) else None
            slots += sum(pair[s]['mse']<single[s]['mse'] for s in SEEDS)
            per[name] = dict(pair_gain=float(1-mean(pair, 'mse')/mean(single, 'mse')), triple_gain=float(1-triple['mse']/mean(single, 'mse')),
                             mae_gain=float(1-mean(pair, 'mae')/mean(single, 'mae')),
                             active_change=None if mean(single, 'active_mse') is None else float(mean(pair, 'active_mse')/mean(single, 'active_mse')-1),
                             quiet_pred_change=None if mean(single, 'quiet_pred') is None else float(mean(pair, 'quiet_pred')-mean(single, 'quiet_pred')))
        g = [v['pair_gain'] for v in per.values()]; n = len(names)
        summary = dict(mean_gain=float(np.mean(g)), mouse_wins=int(sum(v>0 for v in g)), slot_wins=int(slots), slots=3*n,
                       max_harm=float(max(0, -min(g))), mean_mae_gain=float(np.mean([v['mae_gain'] for v in per.values()])),
                       triple_mean_gain=float(np.mean([v['triple_gain'] for v in per.values()])))
        summary['gate'] = dict(mean_gain=summary['mean_gain']>=.05, mouse_wins=summary['mouse_wins']>=math.ceil(6/7*n),
                               slot_wins=slots>=math.ceil(2/3*3*n), harm=summary['max_harm']<=.1, mae=summary['mean_mae_gain']>=0)
        summary['passed'] = all(summary['gate'].values())
        result[family] = dict(summary=summary, per_mouse=per)
    return result


def main():
    dev_rows = read(EXP/'2026-10-07_bounded_hybrid'/'summary.json')['rows']
    hold_rows = read(EXP/'2026-10-07_holdout_confirmation'/'summary.json')['rows']
    dev_mse = lambda m, s, f: next(r['mse'] for r in dev_rows if r['mouse']==m and r['seed']==s+200 and r['arm']==f)
    hold_mse = lambda m, s, f: next(r['mse'] for r in hold_rows if r['recording']==m and r['seed']==s and r['arm']==f)
    out = dict(development=cohort(DEV, load_dev, dev_mse), holdout_set_b=cohort(HOLD, load_hold, hold_mse),
               holdout_TX60_s2=cohort(('TX60_s2',), load_hold, hold_mse), inputs_match_audited_rows=True)
    (ROOT/'summary.json').write_text(json.dumps(out, indent=2)+'\n')
    for c, fams in out.items():
        if c=='inputs_match_audited_rows': continue
        for f, r in fams.items():
            s = r['summary']
            print(f"{c:16s} {f:11s} pair {100*s['mean_gain']:+5.2f}% wins {s['mouse_wins']}/{len(r['per_mouse'])} slots {s['slot_wins']}/{s['slots']} harm {100*s['max_harm']:.2f}% mae {100*s['mean_mae_gain']:+.2f}% triple {100*s['triple_mean_gain']:+5.2f}% {'PASS' if s['passed'] else 'FAIL '+str([k for k,v in s['gate'].items() if not v])}")
            print('   ', {m: (round(100*v['pair_gain'], 1), None if v['active_change'] is None else round(100*v['active_change'], 1)) for m, v in r['per_mouse'].items()})


if __name__=='__main__':
    main()
