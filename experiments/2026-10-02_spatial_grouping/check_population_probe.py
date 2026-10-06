import json
from pathlib import Path
import numpy as np
import torch
import architecture_search as q
import base_search as s

ROOT=Path(__file__).resolve().parent
q.initialize()
torch.set_num_threads(1)
records=json.loads((ROOT/'population_probe_results.json').read_text())['runs']
checks=[]
for pool in [101,202,303,404,505,606]:
    raw=torch.tensor(s.RAW[q.POOL_IDS[pool]],dtype=torch.float64)
    stop=s.SLICES['train'][1]
    a=(raw-raw[:,:stop].mean(1,keepdim=True))/raw[:,:stop].std(1,keepdim=True,unbiased=False).clamp_min(1e-6)
    for row in [r for r in records if r['pool']==pool]:
        k=row['components']
        saved=np.load(ROOT/f'population_probe_p{pool}_k{k}.npz')
        assert all(np.isfinite(saved[name]).all() for name in saved.files)
        components=torch.from_numpy(saved['pca_components'])
        torch.testing.assert_close(components@components.T,torch.eye(k,dtype=torch.float64),atol=1e-10,rtol=1e-10)
        scores=(a.T-a[:,:stop].T.mean(0))@components.T
        data={}
        for split,(start,end) in s.SLICES.items():
            features=scores[start:end].unfold(0,8,1)[24:].reshape(end-start-31,-1)
            target=torch.tensor((s.SPEED[start+31:end]-s.SPEED_MEAN)/s.SPEED_STD,dtype=torch.float64)
            data[split]=(features,target)
        x,y=data['train'];xt,yt=data['test']
        mean=x.mean(0);std=x.std(0,unbiased=False).clamp_min(1e-8)
        x=(x-mean)/std;xt=(xt-mean)/std
        xc=x-x.mean(0);yc=y-y.mean()
        coefficients=torch.linalg.solve(xc.T@xc+row['alpha']*torch.eye(x.shape[1],dtype=torch.float64),xc.T@yc)
        intercept=y.mean()-x.mean(0)@coefficients
        pred=(xt@coefficients+intercept).numpy()
        np.testing.assert_allclose(coefficients.numpy(),saved['coefficients'],atol=1e-8,rtol=1e-6)
        np.testing.assert_allclose(pred,saved['test_prediction'],atol=1e-7,rtol=1e-6)
        checks.append(dict(pool=pool,components=k,max_prediction_error=float(np.max(np.abs(pred-saved['test_prediction']))),
                           max_coefficient_error=float(np.max(np.abs(coefficients.numpy()-saved['coefficients'])))))
result=dict(checks=checks,finite_saved_arrays=True,torch_projection_and_ridge_solve_agree=True,
            note='NumPy/BLAS matmul warnings occurred in sklearn despite finite results; independent Torch projections and regularized normal-equation solutions reproduce all18 saved models')
(ROOT/'population_probe_independent_check.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
