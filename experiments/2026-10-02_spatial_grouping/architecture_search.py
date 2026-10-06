import argparse
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
import base_search as s
from architecture_model import ArchitectureCandidate

ROOT=Path(__file__).resolve().parent
s.PROJECT=Path('/Users/austinlin/neuron_transformer')
ELIGIBLE=POOL_IDS=POS_MEAN=POS_STD=None


def initialize():
    global ELIGIBLE,POOL_IDS,POS_MEAN,POS_STD
    s.initialize()
    training_std=s.RAW[:,:s.SLICES['train'][1]].std(axis=1)
    ELIGIBLE=np.flatnonzero(training_std>1e-6)
    POOL_IDS={p:np.random.default_rng(p).choice(ELIGIBLE,2048,replace=False) for p in [101,202,303,404,505,606]}
    POS_MEAN=s.POS[ELIGIBLE].mean(axis=0,keepdims=True)
    POS_STD=np.maximum(s.POS[ELIGIBLE].std(axis=0,keepdims=True),1e-6)
    (ROOT/'runs').mkdir(exist_ok=True)


@lru_cache(maxsize=3)
def get_data(n,pool,split):
    ids=POOL_IDS[pool][:n]
    raw=s.RAW[ids]
    train=raw[:,:s.SLICES['train'][1]]
    mean=train.mean(1,keepdims=True)
    std=np.maximum(train.std(1,keepdims=True),1e-6)
    start,end=s.SLICES[split]
    activity=(raw[:,start:end]-mean)/std
    windows=np.lib.stride_tricks.sliding_window_view(activity,8,axis=1)
    x=np.ascontiguousarray(windows[:,24:,:].transpose(1,0,2),dtype=np.float32)
    y=np.asarray((s.SPEED[start+31:end]-s.SPEED_MEAN)/s.SPEED_STD,dtype=np.float32)
    assert x.shape==(len(y),n,8)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    return torch.from_numpy(x),torch.from_numpy(y)


def coordinates(n,pool,condition):
    ids=POOL_IDS[pool][:n]
    raw=s.POS[ids].copy()
    permutation=np.arange(n)
    rng=np.random.default_rng(910000+pool*31+n)
    if condition=='shuffled':
        permutation=rng.permutation(n)
        raw=raw[permutation]
    elif condition=='within_depth_shuffled':
        for depth in np.unique(raw[:,2]):
            group=np.flatnonzero(raw[:,2]==depth)
            permutation[group]=rng.permutation(group)
        raw=raw[permutation]
        np.testing.assert_array_equal(raw[:,2],s.POS[ids,2])
    pos=np.asarray((raw-POS_MEAN)/POS_STD,dtype=np.float32)
    if condition=='none':
        pos[:]=0
    elif condition=='depth':
        pos[:,:2]=0
    assert condition in ['real','shuffled','none','depth','within_depth_shuffled']
    return torch.from_numpy(pos),permutation


def configuration(n):
    return dict(n=n,window=8,width=32,lr=.001,epochs=24,schedule='cosine',temporal='mlp',readout='mean',clip=1.,family='latent',latents=8)


def run(task):
    n,pool,seed,condition=(task[k] for k in ['n','pool','seed','condition'])
    variant=task['variant']
    stem=f'{variant}_n{n}_p{pool}_s{seed}_{condition}'
    destination=ROOT/'runs'/f'{stem}.json'
    if destination.exists():
        return json.loads(destination.read_text())
    started=time.monotonic()
    print(json.dumps({'started':stem}),flush=True)
    x,y=get_data(n,pool,'train')
    xv,yv=get_data(n,pool,'val')
    xt,yt=get_data(n,pool,'test')
    pos,coordinate_permutation=coordinates(n,pool,condition)
    torch.manual_seed(seed)
    config=dict(configuration(n),variant=variant,initialization_seed=seed,latents=32 if variant=='latent32' else 8)
    model=ArchitectureCandidate(config)
    model.prepare_geometry(pos)
    initial_hash=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(TensorDataset(x,y),batch_size=32,shuffle=True,generator=generator)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,24,eta_min=.0001)
    untrained_val=s.metrics(s.predict(model,xv,pos),yv.numpy())
    untrained_test=s.metrics(s.predict(model,xt,pos),yt.numpy())
    best,best_state,best_epoch,stale=untrained_val['mse'],deepcopy(model.state_dict()),0,0
    history=[{'epoch':0,**untrained_val}]
    token_ids=torch.arange(n)
    for epoch in range(1,25):
        model.train()
        loss_sum=0.
        for xb,yb in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction=model(xb,pos,token_ids)
            loss=nn.functional.mse_loss(prediction,yb)
            assert torch.isfinite(loss)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            loss_sum+=loss.item()*len(yb)
        val=s.metrics(s.predict(model,xv,pos),yv.numpy())
        history.append({'epoch':epoch,'training_loss':loss_sum/len(x),'lr':optimizer.param_groups[0]['lr'],**val})
        if val['mse']<best:
            best,best_state,best_epoch,stale=val['mse'],deepcopy(model.state_dict()),epoch,0
        else:
            stale+=1
        scheduler.step()
        s.write_json(ROOT/f'progress_{variant}_p{pool}_s{seed}_{condition}.json',{'task':task,'epoch':epoch,'best_val':best,'seconds':time.monotonic()-started})
        if epoch>=12 and stale>=7:
            break
    model.load_state_dict(best_state)
    val_prediction=s.predict(model,xv,pos)
    test_prediction=s.predict(model,xt,pos)
    val=s.metrics(val_prediction,yv.numpy())
    test=s.metrics(test_prediction,yt.numpy())
    assert abs(val['mse']-best)<1e-7
    assert best_epoch==min(history,key=lambda h:h['mse'])['epoch']
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
    record={'task':task,'config':config,'validation':val,'test':test,'untrained_validation':untrained_val,'untrained_test':untrained_test,'best_epoch':best_epoch,'epochs_run':epoch,'history':history,'seconds':time.monotonic()-started,'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'graph_info':model.graph_info,'initial_state_sha256':initial_hash,'examples':{'train':len(y),'validation':len(yv),'test':len(yt)},'checkpoint':stem+'.pt','predictions':stem+'.npz'}
    checkpoint={'state_dict':best_state,'config':config,'task':task,'neuron_ids':POOL_IDS[pool][:n].tolist(),'positions':pos,'coordinate_permutation':coordinate_permutation.tolist(),'speed_mean':s.SPEED_MEAN,'speed_std':s.SPEED_STD,'best_epoch':best_epoch}
    torch.save(checkpoint,ROOT/'runs'/record['checkpoint'])
    np.savez_compressed(ROOT/'runs'/record['predictions'],validation_prediction=val_prediction,validation_target=yv.numpy(),test_prediction=test_prediction,test_target=yt.numpy())
    s.write_json(destination,record)
    return record


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('tasks',nargs='?')
    args=parser.parse_args()
    initialize()
    tasks=json.loads(Path(args.tasks).read_text())
    for i,task in enumerate(tasks):
        r=run(task)
        print(json.dumps({'done':i+1,'total':len(tasks),**task,'val_mse':r['validation']['mse'],'test_mse':r['test']['mse'],'best_epoch':r['best_epoch'],'seconds':round(r['seconds'],1)}),flush=True)
    print('SHARD_COMPLETE',flush=True)


if __name__=='__main__':
    main()
