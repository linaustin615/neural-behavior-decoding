"""Exact standardized ridge paths on the same capped training examples."""
import numpy as np
import torch
from common import ROOT, PLAN, read, save, now, digest, check_sources
from fit import mse
import dataset


def solve(x,y,lambdas):
    x=torch.as_tensor(x,dtype=torch.float64)
    y=torch.as_tensor(y,dtype=torch.float64)
    center=x.mean(0)
    scale=x.std(0,unbiased=False).clamp_min(1e-6)
    z=(x-center)/scale
    ym=y.mean()
    target=y-ym
    gram=z@z.T
    values,vectors=torch.linalg.eigh(gram)
    projection=vectors.T@target
    weights=[]
    residuals=[]
    for lam in lambdas:
        penalty=lam*len(x)
        dual=vectors@(projection/(values+penalty))
        residual=float(torch.linalg.vector_norm(gram@dual+penalty*dual-target)/torch.linalg.vector_norm(target).clamp_min(1e-12))
        assert residual<1e-6 and torch.isfinite(dual).all()
        weights.append(z.T@dual)
        residuals.append(residual)
    return dict(center=center.numpy(),scale=scale.numpy(),weights=torch.stack(weights,1).numpy(),mean=float(ym)),residuals


def predict(data,state):
    predictions=[]
    weights=torch.from_numpy(state['weights'])
    for start in range(0,len(data['indices']),128):
        idx=data['indices'][start:start+128]
        x=data['x'][idx].numpy().reshape(len(idx),-1).astype(np.float64)
        x=(x-state['center'])/state['scale']
        p=(torch.from_numpy(x)@weights).numpy()+float(state['mean'])
        if start==0:
            manual=np.einsum('ij,jk->ik',x[:17],state['weights'],optimize=False)+float(state['mean'])
            np.testing.assert_allclose(p[:17],manual,rtol=1e-7,atol=1e-7)
        predictions.append(p)
    result=np.concatenate(predictions)
    assert np.isfinite(result).all()
    return result


def fit(mouse):
    check_sources()
    out=ROOT/'ridge'/mouse
    if (out/'result.json').exists():
        result=read(out/'result.json')
        assert digest(out/'selected.npz')==result['selected_sha256']
        return result
    assert not out.exists(),str(out)
    out.mkdir(parents=True)
    options=[]
    best=None
    for history in PLAN['ridge']['histories']:
        train=dataset.load(mouse,'train',history)
        val=dataset.load(mouse,'validation',history)
        idx=train['indices']
        x=train['x'][idx].numpy().reshape(len(idx),-1)
        state,residuals=solve(x,train['y'][idx],PLAN['ridge']['penalties'])
        predictions=predict(val,state)
        np.savez_compressed(out/f'validation_h{history}.npz',predictions=predictions,target=val['y'])
        for k,(lam,residual) in enumerate(zip(PLAN['ridge']['penalties'],residuals)):
            error=mse(predictions[:,k],val['y'],val['meta']['lower'])
            option=dict(history=history,lam=lam,validation_mse=error,normal_equation_residual=residual)
            options.append(option)
            if best is None or error<best['validation_mse']:
                best=option.copy()
                selected={**state,'weights':state['weights'][:,k:k+1].copy()}
                np.savez(out/'selected.npz',**selected)
        print(mouse,'ridge history',history,'completed',flush=True)
        del x,state,train,val
        dataset.load.cache_clear()
    result=dict(mouse=mouse,selected=best,options=options,solutions=len(options),
        selected_sha256=digest(out/'selected.npz'),completed_utc=now(),test_opened=False)
    save(out/'result.json',result)
    return result
