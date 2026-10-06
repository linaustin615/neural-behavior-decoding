import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.cluster import KMeans
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from threadpoolctl import threadpool_limits

import base_search as s
from group_model import GroupCandidate

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
POOLS=[101,202,303]
SEEDS=[10,11]
CONDITIONS=['functional','random','global']
SPLITS={'train':(0,4160),'selection':(4260,4860),'evaluation':(4960,5564)}
CONFIG=dict(n=2048,window=8,width=32,lr=.001,epochs=24,schedule='cosine',temporal='mlp',
            readout='mean',clip=1.,family='latent',latents=16)
RAW=SPEED=POS=ELIGIBLE=LIMITER=None


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(2**20),b''):
            h.update(block)
    return h.hexdigest()


def initialize():
    global RAW,SPEED,POS,ELIGIBLE,LIMITER
    LIMITER=threadpool_limits(limits=2)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    d=np.load(PROJECT/'data/stringer_spontaneous.npy',allow_pickle=True).item()
    RAW=d['sresp'][:,:5564].copy()
    SPEED=d['run'][:5564,0].copy()
    POS=d['xyz'].T.copy()
    del d
    ELIGIBLE=np.flatnonzero(RAW[:,:4160].std(axis=1)>1e-6)


def data(pool):
    ids=np.random.default_rng(pool).choice(ELIGIBLE,2048,replace=False)
    raw=RAW[ids]
    mu=raw[:,:4160].mean(1,keepdims=True)
    sd=np.maximum(raw[:,:4160].std(1,keepdims=True),1e-6)
    mean=float(SPEED[7:4160].mean())
    std=max(float(SPEED[7:4160].std()),1e-6)
    arrays={}
    for name,(start,stop) in SPLITS.items():
        z=(raw[:,start:stop]-mu)/sd
        w=np.lib.stride_tricks.sliding_window_view(z,8,axis=1)
        x=np.ascontiguousarray(w[:,24:,:].transpose(1,0,2),dtype=np.float32)
        y=np.asarray((SPEED[start+31:stop]-mean)/std,dtype=np.float32)
        assert x.shape==(stop-start-31,2048,8) and np.isfinite(x).all()
        arrays[name]=(torch.from_numpy(x),torch.from_numpy(y))
    return ids,arrays,dict(speed_mean=mean,speed_std=std)


def functional_groups(training_activity,pool):
    z=torch.as_tensor(training_activity,dtype=torch.float64)
    z=z-z.mean(1,keepdim=True)
    z=z/torch.linalg.vector_norm(z,dim=1,keepdim=True).clamp_min(1e-12)
    corr=z@z.T
    eigenvalues,eigenvectors=torch.linalg.eigh(corr)
    features=eigenvectors[:,-32:]*eigenvalues[-32:].clamp_min(0).sqrt()
    fit=KMeans(n_clusters=8,n_init=10,max_iter=300,random_state=500000+pool).fit(features.numpy())
    groups=fit.labels_.astype(np.int64)
    distance=torch.sum((features[:,None,:]-torch.as_tensor(fit.cluster_centers_)[None,:,:])**2,dim=2)
    np.testing.assert_array_equal(distance.argmin(1).numpy(),groups)
    assert np.bincount(groups,minlength=8).min()>=2
    return groups,dict(group_sizes=np.bincount(groups).tolist(),
                      rank32_correlation_variance_fraction=float(eigenvalues[-32:].sum()/eigenvalues.sum()),
                      inertia=float(fit.inertia_),iterations=int(fit.n_iter_))


def make_model(seed,condition,groups):
    torch.manual_seed(seed)
    return GroupCandidate(dict(CONFIG,initialization_seed=seed,variant=condition),groups)


def state_hash(model):
    return hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in model.state_dict().values())).hexdigest()


def score(pred,target,meta):
    pred=np.asarray(pred,dtype=np.float64)
    target=np.asarray(target,dtype=np.float64)
    bounded=np.maximum(pred,-meta['speed_mean']/meta['speed_std'])
    mse=float(np.mean((bounded-target)**2))
    return dict(mse=mse,raw_mse=float(np.mean((pred-target)**2)),r2=1-mse/float(np.var(target)))


def prepare():
    assert not (ROOT/'protocol.json').exists()
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        question='Do training-activity-defined groups improve concurrent running-speed decoding beyond same-size randomized groups and unrestricted activity+ID attention?',
        scope='exploratory final bounded batch; no guarantee of significance; no anatomical or causal-connectivity claim',
        pools=POOLS,seeds=SEEDS,conditions=CONDITIONS,new_fits=18,reused_fits=0,
        reuse_reason='checkpoint selection now uses only early validation; previous whole-validation checkpoints are incompatible',
        splits=SPLITS,window=8,target_offset=31,examples=dict(train=4129,selection=569,evaluation=573),
        test_tail='not constructed or evaluated',
        grouping='standardize each training time course to unit L2 norm; eigendecompose cell correlation matrix; top32 scaled eigenvectors; KMeans8 n_init10 seed500000+pool; training activity only, no speed or coordinates',
        random_control='permute functional labels with seed420000+pool; exactly preserve group sizes; one independent assignment per pool',
        architecture='same 81377 parameters,16 summaries,width32,4heads,nonlinear activity embedding,ID vectors,mean readout; first8 summaries masked by groups,last8 unrestricted; global arm all16 unrestricted',
        coordinates='all-zero coordinate inputs in every model; true xyz used only for final visualization',
        normalization='activity and speed training-only; all arms share normalization within pool',
        training=dict(batch=32,epochs=24,min_epochs=12,patience=7,lr=.001,weight_decay=.01,
                      cosine_min_lr=.0001,gradient_clip=1,checkpoint='minimum selection bounded MSE including epoch0'),
        selection_rule='no evaluation examples passed to the training function; after all18 checkpoints are fixed, evaluate once on later interval',
        primary_contrasts=['relative MSE reduction functional vs random','relative MSE reduction functional vs global'],
        promising_gate='>=2% mean relative error reduction against BOTH controls; >=5/6 paired wins each; all3 pool means positive each',
        stronger_exploratory_gate='promising gate plus both conditional97.5% interval lower bounds above0; these are NOT confirmatory significance tests on this repeatedly examined recording',
        bootstrap='4000 paired resamples of3 pool indices,2 seed indices and circular evaluation blocks100; same indices for all arms; 1.25/98.75 percentiles for each of2 contrasts',
        group_stability='secondary: average of1024 sampled distinct-cell pairs per group on later evaluation activity vs999 neuron-label permutations; fixed edges from training groups; descriptive null rank only, no biological independence claim',
        graph_limitations='static simultaneous correlations; no lagged/dynamic or causal model; common running state, population signals and imaging artifacts can explain associations',
        stop_rule='finish exactly18 planned fits and audits; no adaptive extra seeds,pools,hyperparameters or test-tail inspection',
        limitations=['single repeatedly examined recording','overlapping pools and only2 technical seeds',
                     '573 evaluation examples overlap in time windows','new selection/evaluation split does not reset historical data reuse',
                     'fixed architecture and inherited training recipe; not a fully tuned model-family comparison'],
        tasks=[dict(pool=p,seed=seed,condition=c) for p in POOLS for seed in SEEDS for c in CONDITIONS],
        source_hashes={f:digest(ROOT/f) for f in ['run.py','model_snapshot.py','base_search.py','group_model.py']},
        application_hashes={f:digest(PROJECT/f) for f in ['model.py','data.py','train.py']},
        dataset_sha256=digest(PROJECT/'data/stringer_spontaneous.npy'))
    s.write_json(ROOT/'protocol.json',protocol)
    initialize()
    checks=[]
    for pool in POOLS:
        ids,arrays,meta=data(pool)
        groups,info=functional_groups(RAW[ids,:4160],pool)
        random=groups[np.random.default_rng(420000+pool).permutation(2048)]
        np.testing.assert_array_equal(np.bincount(groups),np.bincount(random))
        np.savez_compressed(ROOT/f'groups_{pool}.npz',ids=ids,functional=groups,random=random,positions=POS[ids])
        s.write_json(ROOT/f'groups_{pool}.json',info)
        hashes=[]
        for condition,g in [('functional',groups),('random',random),('global',groups)]:
            net=make_model(10,condition,g)
            hashes.append(state_hash(net))
            assert sum(p.numel() for p in net.parameters())==81377
            if condition=='global':
                assert not net.group_mask.any()
            else:
                np.testing.assert_array_equal(net.group_mask[:8].numpy(),g[None,:]!=np.arange(8)[:,None])
                assert not net.group_mask[8:].any()
            out=net(arrays['train'][0][:2],torch.zeros(2048,3),torch.arange(2048))
            assert out.shape==(2,) and torch.isfinite(out).all()
            out.square().mean().backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
        assert len(set(hashes))==1
        for name,(start,stop) in SPLITS.items():
            x,y=arrays[name]
            np.testing.assert_array_equal(y.numpy(),((SPEED[start+31:stop]-meta['speed_mean'])/meta['speed_std']).astype(np.float32))
            mu=RAW[ids,:4160].mean(1,keepdims=True)
            sd=np.maximum(RAW[ids,:4160].std(1,keepdims=True),1e-6)
            for k in [0,len(x)-1]:
                expected=((RAW[ids,start+k+24:start+k+32]-mu)/sd).astype(np.float32)
                np.testing.assert_array_equal(x[k].numpy(),expected)
        checks.append(dict(pool=pool,**info,passed=True))
        print(json.dumps(checks[-1]),flush=True)
        del arrays
    s.write_json(ROOT/'preflight.json',dict(passed=True,pools=checks,matching_parameter_initialization=True,
        exact_random_group_sizes=True,mask_semantics_and_gradients=True,new_split_boundaries=True))
    print('PROTOCOL_AND_PREFLIGHT_COMPLETE',flush=True)


def verify():
    p=json.loads((ROOT/'protocol.json').read_text())
    for f,h in p['source_hashes'].items():
        assert digest(ROOT/f)==h,f
    for f,h in p['application_hashes'].items():
        assert digest(PROJECT/f)==h,f
    return p


def stem(task):
    return f"p{task['pool']}_s{task['seed']}_{task['condition']}"


def fit(task,x,y,xs,ys,meta,groups,ids):
    name=stem(task)
    path=ROOT/'runs'/f'{name}.json'
    if path.exists():
        return
    start=time.monotonic()
    net=make_model(task['seed'],task['condition'],groups)
    initial_hash=state_hash(net)
    pos=torch.zeros(2048,3)
    token_ids=torch.arange(2048)
    untrained=score(s.predict(net,xs,pos),ys.numpy(),meta)
    best,state,best_epoch,stale=untrained['mse'],deepcopy(net.state_dict()),0,0
    history=[dict(epoch=0,**untrained)]
    loader=DataLoader(TensorDataset(x,y),batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(task['seed']))
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,24,eta_min=.0001)
    for epoch in range(1,25):
        net.train()
        loss_sum=0.
        for xb,yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.mse_loss(net(xb,pos,token_ids),yb)
            assert torch.isfinite(loss)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            loss_sum+=loss.item()*len(yb)
        val=score(s.predict(net,xs,pos),ys.numpy(),meta)
        history.append(dict(epoch=epoch,training_loss=loss_sum/len(x),**val))
        if val['mse']<best:
            best,state,best_epoch,stale=val['mse'],deepcopy(net.state_dict()),epoch,0
        else:
            stale+=1
        scheduler.step()
        s.write_json(ROOT/f"progress_{task['pool']}.json",dict(task=task,epoch=epoch,best_selection=best,seconds=time.monotonic()-start))
        if epoch>=12 and stale>=7:
            break
    net.load_state_dict(state)
    pred=s.predict(net,xs,pos)
    val=score(pred,ys.numpy(),meta)
    assert abs(val['mse']-best)<1e-7
    assert best_epoch==min(history,key=lambda h:h['mse'])['epoch']
    assert all(torch.isfinite(v).all() for v in state.values())
    ck=dict(task=task,state_dict=state,groups=groups,ids=ids,normalization=meta,best_epoch=best_epoch)
    torch.save(ck,path.with_suffix('.pt'))
    np.savez_compressed(path.with_suffix('.npz'),selection_prediction=pred,selection_target=ys.numpy())
    record=dict(task=task,selection=val,untrained_selection=untrained,best_epoch=best_epoch,epochs_run=epoch,
        history=history,initial_state_sha256=initial_hash,parameters=sum(p.numel() for p in net.parameters()),
        normalization=meta,seconds=time.monotonic()-start,evaluation_scored=False)
    s.write_json(path,record)
    print(json.dumps(dict(done=name,selection_mse=best,best_epoch=best_epoch,seconds=round(record['seconds'],1))),flush=True)


def worker(pool):
    initialize()
    protocol=verify()
    ids,arrays,meta=data(pool)
    with np.load(ROOT/f'groups_{pool}.npz') as g:
        np.testing.assert_array_equal(g['ids'],ids)
        groups={c:g['random' if c=='random' else 'functional'] for c in CONDITIONS}
    #the fit function receives only training and checkpoint-selection examples
    x,y=arrays['train']
    xs,ys=arrays['selection']
    del arrays
    for task in protocol['tasks']:
        if task['pool']==pool:
            fit(task,x,y,xs,ys,meta,groups[task['condition']],ids)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','fit'])
    args=parser.parse_args()
    if args.action=='prepare':
        prepare()
    else:
        verify()
        assert json.loads((ROOT/'preflight.json').read_text())['passed']
        start=time.monotonic()
        with ProcessPoolExecutor(max_workers=3) as executor:
            list(executor.map(worker,POOLS))
        s.write_json(ROOT/'execution.json',dict(wall_seconds=time.monotonic()-start,workers=3,threads_per_worker=2))
        print('ALL_18_CHECKPOINTS_FIXED',flush=True)
