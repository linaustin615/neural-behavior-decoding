import json
from pathlib import Path
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
import architecture_search as q
import base_search as s

ROOT=Path(__file__).resolve().parent
protocol=dict(question='How well do a few population-wide activity modes predict speed?',
              pool_seeds=[101,202,303,404,505,606],components=[1,4,16],alphas=[.1,1,10,100,1000],
              fit='PCA16 fitted only on training activity; ridge on8-bin histories; alpha chosen by validation; common target start31',
              interpretation='Low-dimensional predictability is not proof spatial information is absent; descriptive secondary probe')
(ROOT/'population_probe_protocol.json').write_text(json.dumps(protocol,indent=2))
q.initialize()
torch.set_num_threads(1)
records=[]
for pool in protocol['pool_seeds']:
    raw=s.RAW[q.POOL_IDS[pool]].astype(np.float64)
    end=s.SLICES['train'][1]
    a=(raw-raw[:,:end].mean(1,keepdims=True))/np.maximum(raw[:,:end].std(1,keepdims=True),1e-6)
    pca=PCA(n_components=16,svd_solver='randomized',random_state=10000+pool)
    pca.fit(a[:,:end].T)
    scores=pca.transform(a.T)
    for k in protocol['components']:
        data={}
        for split,(start,stop) in s.SLICES.items():
            windows=np.lib.stride_tricks.sliding_window_view(scores[start:stop,:k],8,axis=0)[24:]
            xx=np.ascontiguousarray(windows.reshape(len(windows),-1))
            yy=(s.SPEED[start+31:stop]-s.SPEED_MEAN)/s.SPEED_STD
            data[split]=(xx,yy)
        x,y=data['train'];xv,yv=data['val'];xt,yt=data['test']
        mean=x.mean(0);std=np.maximum(x.std(0),1e-8)
        x,xv,xt=((z-mean)/std for z in [x,xv,xt])
        best=None
        fits=[]
        for alpha in protocol['alphas']:
            model=Ridge(alpha=alpha,solver='svd').fit(x,y)
            pred=model.predict(xv)
            metrics=s.metrics(pred,yv)
            fits.append(dict(alpha=alpha,**metrics))
            if best is None or metrics['mse']<best[0]:
                best=(metrics['mse'],model,metrics)
        pred=best[1].predict(xt)
        record=dict(pool=pool,components=k,explained_training_neural_variance=float(pca.explained_variance_ratio_[:k].sum()),
                    alpha=best[1].alpha,validation=best[2],test=s.metrics(pred,yt),fits=fits)
        records.append(record)
        np.savez_compressed(ROOT/f'population_probe_p{pool}_k{k}.npz',test_prediction=pred,test_target=yt,
                            coefficients=best[1].coef_,intercept=best[1].intercept_,pca_components=pca.components_[:k])
    print(json.dumps({'pool_done':pool}),flush=True)
summary={str(k):dict(test_mse=float(np.mean([r['test']['mse'] for r in records if r['components']==k])),
                     test_r2=float(np.mean([r['test']['r2'] for r in records if r['components']==k]))) for k in protocol['components']}
(ROOT/'population_probe_results.json').write_text(json.dumps(dict(protocol=protocol,runs=records,summary=summary),indent=2))
print(json.dumps(summary),flush=True)
