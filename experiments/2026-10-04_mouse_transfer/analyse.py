"""Evaluate fixed transfer hypotheses; mice and seeds remain paired."""
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent.parent
MICE=['MP030','MP032','MP033','MP034']
SEEDS=[10,11,12]
ARMS=['attention_transfer','attention_scratch','mlp_transfer','mlp_scratch']


def read(p):
    return json.loads(p.read_text())


def write(name,obj):
    p=ROOT/name
    assert not p.exists(),p
    p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def contrast(rows,main,control):
    gains=[1-r['means'][main]['mse']/r['means'][control]['mse'] for r in rows]
    mae=[1-r['means'][main]['mae']/r['means'][control]['mae'] for r in rows]
    result=dict(main=main,control=control,mean_relative_mse_gain=float(np.mean(gains)),mouse_gains=gains,
                mouse_wins=sum(g>0 for g in gains),mean_relative_mae_gain=float(np.mean(mae)),mae_mouse_gains=mae,
                leave_one_mouse_out=[float(np.mean(np.delete(gains,i))) for i in range(4)])
    passes=np.mean(gains)>=.05 and sum(g>0 for g in gains)>=3
    if control in ARMS:
        wins=sum(a<b for r in rows for a,b in zip(r['seed_mse'][main],r['seed_mse'][control]))
        result.update(paired_seed_wins=wins,paired_seed_count=12)
        passes=passes and wins>=8
    result['practical_threshold']=bool(passes)
    return result


def bootstrap(errors):
    rng=np.random.default_rng(81224)
    controls=['attention_scratch','mlp_transfer']
    draws=[]
    for _ in range(2000):
        seeds=rng.integers(0,3,3)
        mouse_gains=[]
        for mouse in rng.integers(0,4,4):
            e=errors[mouse]
            n=e['attention_transfer'].shape[1]
            times=((rng.integers(0,n,(n+99)//100)[:,None]+np.arange(100))%n).ravel()[:n]
            main=e['attention_transfer'][seeds][:,times].mean()
            mouse_gains.append([1-main/e[c][seeds][:,times].mean() for c in controls])
        draws.append(np.mean(mouse_gains,axis=0))
    return {c:np.quantile(np.array(draws)[:,i],[.0125,.9875]).tolist() for i,c in enumerate(controls)}


def main():
    protocol=read(ROOT/'protocol.json')
    source=str(Path(__file__).relative_to(REPO))
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==protocol['input_hashes'][source]
    assert read(ROOT/'audit.json')['passed']
    rows=[]
    errors=[]
    for row in read(ROOT/'results.json')['rows']:
        mouse=row['mouse']
        means={}
        seed_mse={}
        for arm in ARMS:
            scores=[row['scores'][f'{arm}_s{s}'] for s in SEEDS]
            means[arm]={name:float(np.mean([r[name] for r in scores])) for name in ['mse','mae','r2','native_mse']}
            seed_mse[arm]=[s['mse'] for s in scores]
        for label in ['ridge','initial','median']:
            means[label]=row['scores'][label]
        initial_wins=sum(v<means['initial']['mse'] for v in seed_mse['attention_transfer'])
        rows.append(dict(mouse=mouse,n=row['n'],means=means,seed_mse=seed_mse,initial_wins=initial_wins))
        with np.load(ROOT/mouse/'later_predictions.npz') as z:
            y=z['target']
            e={arm:np.stack([(np.maximum(z[f'{arm}_s{s}'].astype(np.float64),row['lower'])-y)**2 for s in SEEDS]) for arm in ARMS}
            for arm in ARMS:
                np.testing.assert_allclose(e[arm].mean(1),seed_mse[arm],rtol=1e-12,atol=1e-12)
            errors.append(e)
    comparisons={c:contrast(rows,'attention_transfer',c) for c in ['attention_scratch','mlp_transfer','mlp_scratch','ridge','initial','median']}
    sharing=comparisons['attention_scratch']['practical_threshold']
    no_ridge_harm=min(comparisons['ridge']['mouse_gains'])>=-.25
    initial_wins=sum(r['initial_wins'] for r in rows)
    utility=sharing and all(comparisons[c]['practical_threshold'] for c in ['mlp_transfer','mlp_scratch','ridge']) and no_ridge_harm and initial_wins>=8
    result=dict(rows=rows,comparisons=comparisons,mlp_transfer=contrast(rows,'mlp_transfer','mlp_scratch'),
                transformer_transfer_gate=sharing,attention_utility_gate=bool(utility),ridge_harm_guard=bool(no_ridge_harm),
                initial_wins=initial_wins,initial_comparisons=12,descriptive_97_5_intervals=bootstrap(errors))
    write('summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':
    main()
