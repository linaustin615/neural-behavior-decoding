"""Post-hoc audit of fixed averages; no fitting or blend-weight selection."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / '2026-10-04_hybrid_complementarity'
MICE = ['MP030', 'MP032', 'MP033', 'MP034']
SEEDS = [10, 11, 12]


def contrast(rows, control):
    gains = [1-r['mixed']/r[control] for r in rows]
    return dict(mean_relative_gain=float(np.mean(gains)), mouse_gains=gains,
                mouse_wins=sum(g>0 for g in gains),
                paired_seed_wins=sum(a<b for r in rows for a,b in zip(r['mixed_seeds'],r[control+'_seeds'])),
                leave_one_mouse_out=[float(np.mean(np.delete(gains,i))) for i in range(4)])


def main():
    old_protocol=json.loads((OLD/'protocol.json').read_text())
    stages={};later_errors={};checked=0
    for stage,path in [('development',OLD/'development_predictions.npz'),('later',ROOT/'later_predictions.npz')]:
        rows=[]
        with np.load(path) as z:
            for m in MICE:
                ix=slice(*old_protocol['split'][m]['score']) if stage=='development' else slice(None)
                y=z[m+'_target'][ix]
                if stage=='development':
                    a=np.stack([z[f'{m}_attention_mlp_s{s}_left'][ix] for s in SEEDS])
                    b=np.stack([z[f'{m}_attention_mlp_s{s}_right'][ix] for s in SEEDS])
                    saved=np.stack([z[f'{m}_attention_mlp_s{s}_half'][ix] for s in SEEDS])
                else:
                    a=np.stack([z[f'{m}_mlp_s{s}'][ix] for s in SEEDS])
                    b=np.stack([z[f'{m}_attention_s{s}'][ix] for s in SEEDS])
                    saved=np.stack([z[f'{m}_half_s{s}'][ix] for s in SEEDS])
                p={'mlp':a,'attention':b,'mixed':.5*(a+b),
                   'two_mlp':.5*(a+np.roll(a,-1,axis=0)),
                   'two_attention':.5*(b+np.roll(b,-1,axis=0))}
                np.testing.assert_allclose(saved,p['mixed'],rtol=1e-14,atol=1e-14)
                errors={name:(v-y)**2 for name,v in p.items()}
                np.testing.assert_allclose(errors['mixed'],.5*(errors['mlp']+errors['attention'])-.25*(a-b)**2,rtol=1e-11,atol=1e-12)
                row=dict(mouse=m,n=len(y))
                for name,e in errors.items():
                    scores=e.mean(1)
                    manual=[sum((float(v)-float(t))**2 for v,t in zip(q,y))/len(y) for q in p[name]]
                    np.testing.assert_allclose(scores,manual,rtol=1e-12,atol=1e-12);checked+=3
                    row[name+'_seeds']=scores.tolist();row[name]=float(scores.mean())
                rows.append(row)
                if stage=='later':later_errors[m]=errors
        stages[stage]=dict(rows=rows,comparisons={c:contrast(rows,c) for c in ['mlp','attention','two_mlp','two_attention']})
    rng=np.random.default_rng(81218);draws=[]
    for _ in range(2000):
        seeds=rng.integers(0,3,3);gains=[]
        for m in rng.choice(MICE,4):
            e=later_errors[m];n=e['mixed'].shape[1]
            idx=((rng.integers(0,n,(n+99)//100)[:,None]+np.arange(100))%n).ravel()[:n]
            mse={k:float(v[seeds][:,idx].mean()) for k,v in e.items()}
            gains.append([1-mse['mixed']/max(mse[c],1e-15) for c in ['mlp','attention','two_mlp','two_attention']])
        draws.append(np.mean(gains,axis=0))
    intervals={c:np.quantile(np.array(draws)[:,i],[.025,.975]).tolist() for i,c in enumerate(['mlp','attention','two_mlp','two_attention'])}
    files=[Path(__file__),ROOT/'summary.json',ROOT/'later_predictions.npz',ROOT/'protocol.json',
           OLD/'protocol.json',OLD/'development_predictions.npz']
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Post-hoc descriptive context after fixed 50:50 comparator performed well in later replay. No new fitted model,weights,checkpoint choice or primary gate. Two-MLP/two-transformer averages use fixed cyclic seed pairs. Development and later stages use different checkpoint-selection prefixes,so differences do not isolate one cause.',
        new_fits=0,primary_gate_changed=False,stages=stages,descriptive_later_95_intervals=intervals,
        audit=dict(passed=True,independent_scalar_errors=checked,exact_saved_half_predictions=True,ensemble_error_identity=True),
        source_hashes={str(p.relative_to(ROOT.parents[1])):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (ROOT/'ensemble_context.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(later=stages['later']['comparisons'],intervals=intervals,development=stages['development']['comparisons']),indent=2))


if __name__=='__main__':main()
