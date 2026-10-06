"""Open the new test periods only after every checkpoint is locked."""
import numpy as np
import torch
from common import ROOT, PLAN, MICE, read, save, now, digest, check_sources, job_name
from models import Decoder
from fit import predict
import dataset
import ridge


def metrics(prediction,target,lower):
    p=np.asarray(prediction,dtype=np.float64)
    y=np.asarray(target,dtype=np.float64)
    d=np.maximum(p,lower)-y
    variance=float(np.mean((y-y.mean())**2))
    error=float(np.mean(d**2))
    return dict(mse=error,mae=float(np.mean(np.abs(d))),r2=(1-error/variance if variance>0 else None),
        raw_mse=float(np.mean((p-y)**2)),n=len(y))


def lock(jobs):
    assert len(jobs)==PLAN['neural_fit_budget']
    check_sources()
    paths={}
    for job in jobs:
        out=ROOT/'fits'/job_name(job)
        result=read(out/'result.json')
        assert result['job']==job and not result['test_opened']
        assert digest(out/'selected.pt')==result['selected_sha256']
        paths[str((out/'selected.pt').relative_to(ROOT))]=result['selected_sha256']
    for mouse in MICE:
        out=ROOT/'ridge'/mouse
        result=read(out/'result.json')
        assert result['solutions']==24 and not result['test_opened']
        assert digest(out/'selected.npz')==result['selected_sha256']
        paths[str((out/'selected.npz').relative_to(ROOT))]=result['selected_sha256']
    save(ROOT/'evaluation_lock.json',dict(utc=now(),neural_fits=len(jobs),ridge_solutions=168,
        selected_sha256=paths,test_predictions_or_errors_previously_opened=False,
        protocol_sha256=digest(ROOT/'protocol.json')))


def evaluate(jobs):
    lock(jobs)
    rows=[]
    out=ROOT/'test_predictions'
    out.mkdir(exist_ok=False)
    for job in jobs:
        shared=job['regime']=='shared'
        ids=list(range(7)) if shared else [MICE.index(job['mouse'])]
        config=PLAN['architecture_configs'][job['role']]
        net=Decoder(config,job['seed'],None if shared else ids[0])
        folder=ROOT/'fits'/job_name(job)
        result=read(folder/'result.json')
        net.load_state_dict(torch.load(folder/'selected.pt',weights_only=True))
        arrays={}
        for s,i in enumerate(ids):
            mouse=MICE[i]
            d=dataset.load(mouse,'test')
            p=predict(net,d,s if shared else 0)
            arrays[mouse+'_prediction']=p
            arrays[mouse+'_target']=d['y']
            rows.append(dict(regime=job['regime'],mouse=mouse,role=job['role'],seed=job['seed'],
                selected_epoch=result['selected_epoch'],**metrics(p,d['y'],d['meta']['lower'])))
        np.savez_compressed(out/(job_name(job)+'.npz'),**arrays)
    for mouse in MICE:
        folder=ROOT/'ridge'/mouse
        result=read(folder/'result.json')
        with np.load(folder/'selected.npz') as z:
            state={k:z[k] for k in z.files}
        d=dataset.load(mouse,'test',result['selected']['history'])
        p=ridge.predict(d,state)[:,0]
        np.savez_compressed(out/('ridge_'+mouse+'.npz'),prediction=p,target=d['y'])
        rows.append(dict(regime='control',mouse=mouse,role='ridge',seed=None,**metrics(p,d['y'],d['meta']['lower'])))
        for role,value in [('training_mean',0.),('zero_speed',d['meta']['lower'])]:
            rows.append(dict(regime='control',mouse=mouse,role=role,seed=None,
                **metrics(np.full(len(p),value),d['y'],d['meta']['lower'])))
        dataset.load.cache_clear()
    save(ROOT/'test_metrics.json',dict(opened_after_lock=True,rows=rows,completed_utc=now()))
    return rows
