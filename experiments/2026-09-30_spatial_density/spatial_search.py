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

ROOT=Path(__file__).resolve().parent
s.PROJECT=ROOT.parents[1]
ELIGIBLE=POOL_IDS=POS_MEAN=POS_STD=None


def initialize():
    global ELIGIBLE,POOL_IDS,POS_MEAN,POS_STD
    s.initialize()
    training_std=s.RAW[:,:s.SLICES['train'][1]].std(axis=1)
    ELIGIBLE=np.flatnonzero(training_std>1e-6)
    POOL_IDS={p:np.random.default_rng(p).choice(ELIGIBLE,2048,replace=False) for p in [101,202,303]}
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
    stem=f'n{n}_p{pool}_s{seed}_{condition}'
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
    config=configuration(n)
    model=s.Candidate(config)
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
        s.write_json(ROOT/f'progress_{pool}.json',{'task':task,'epoch':epoch,'best_val':best,'seconds':time.monotonic()-started})
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
    record={'task':task,'config':config,'validation':val,'test':test,'untrained_validation':untrained_val,'untrained_test':untrained_test,'best_epoch':best_epoch,'epochs_run':epoch,'history':history,'seconds':time.monotonic()-started,'parameters':sum(p.numel() for p in model.parameters()),'initial_state_sha256':initial_hash,'examples':{'train':len(y),'validation':len(yv),'test':len(yt)},'checkpoint':stem+'.pt','predictions':stem+'.npz'}
    checkpoint={'state_dict':best_state,'config':config,'task':task,'neuron_ids':POOL_IDS[pool][:n].tolist(),'positions':pos,'coordinate_permutation':coordinate_permutation.tolist(),'speed_mean':s.SPEED_MEAN,'speed_std':s.SPEED_STD,'best_epoch':best_epoch}
    torch.save(checkpoint,ROOT/'runs'/record['checkpoint'])
    np.savez_compressed(ROOT/'runs'/record['predictions'],validation_prediction=val_prediction,validation_target=yv.numpy(),test_prediction=test_prediction,test_target=yt.numpy())
    s.write_json(destination,record)
    return record


def audit():
    checks={'eligible_cells':len(ELIGIBLE),'spatial_data':'recorded visual cortex, not whole brain','same_initialization_across_conditions':{},'coordinate_controls':{},'data_checks':{},'gradients_and_shapes':{}}
    for pool in POOL_IDS:
        ids=POOL_IDS[pool]
        assert len(np.unique(ids))==2048
        assert set(ids)<=set(ELIGIBLE)
        for n in [128,512,2048]:
            real,_=coordinates(n,pool,'real')
            shuffled,perm=coordinates(n,pool,'shuffled')
            within,within_perm=coordinates(n,pool,'within_depth_shuffled')
            none,_=coordinates(n,pool,'none')
            depth,_=coordinates(n,pool,'depth')
            assert sorted(perm.tolist())==list(range(n))
            torch.testing.assert_close(real[perm],shuffled,rtol=0,atol=0)
            torch.testing.assert_close(within[:,2],real[:,2],rtol=0,atol=0)
            torch.testing.assert_close(depth[:,2],real[:,2],rtol=0,atol=0)
            assert torch.count_nonzero(none)==0 and torch.count_nonzero(depth[:,:2])==0
            checks['coordinate_controls'][f'{pool}_{n}']={'full_shuffle_moved_fraction':float(np.mean(perm!=np.arange(n))),'within_depth_shuffle_moved_fraction':float(np.mean(within_perm!=np.arange(n))),'multiset_and_depth_preservation':True}
    for split in ['train','val','test']:
        x,y=get_data(128,101,split)
        start,end=s.SLICES[split]
        raw=s.RAW[POOL_IDS[101][:128]]
        train=raw[:,:s.SLICES['train'][1]]
        for j in [0,len(y)//2,len(y)-1]:
            t=start+31+j
            expected=(raw[:,t-7:t+1]-train.mean(1,keepdims=True))/np.maximum(train.std(1,keepdims=True),1e-6)
            np.testing.assert_allclose(expected,x[j].numpy(),rtol=1e-6,atol=1e-6)
            assert abs(y[j].item()-(s.SPEED[t]-s.SPEED_MEAN)/s.SPEED_STD)<1e-6
        checks['data_checks'][split]={'examples':len(y),'target_and_window_alignment':True}
    for condition in ['real','shuffled','none','depth','within_depth_shuffled']:
        torch.manual_seed(10)
        model=s.Candidate(configuration(128))
        checks['same_initialization_across_conditions'][condition]=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
        pos,_=coordinates(128,101,condition)
        x,y=get_data(128,101,'train')
        ids=torch.arange(128)
        model.eval()
        with torch.no_grad():
            pred=model(x[:2],pos,ids)
            one=model(x[:1],pos,ids)
            perm=torch.randperm(128)
            shuffled=model(x[:2,perm],pos[perm],ids[perm])
        assert pred.shape==(2,) and one.shape==(1,)
        torch.testing.assert_close(pred,shuffled,rtol=1e-4,atol=1e-5)
        model.train()
        (model(x[:2],pos,ids)-y[:2]).square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        checks['gradients_and_shapes'][condition]=True
    assert len(set(checks['same_initialization_across_conditions'].values()))==1
    checks['source_hashes_unchanged']={name:hashlib.sha256((s.PROJECT/name).read_bytes()).hexdigest()==value for name,value in json.loads((ROOT/'protocol.json').read_text())['source_hashes'].items()}
    assert all(checks['source_hashes_unchanged'].values())
    s.write_json(ROOT/'integrity_audit.json',checks)
    print(json.dumps(checks,indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('tasks',nargs='?')
    parser.add_argument('--audit',action='store_true')
    args=parser.parse_args()
    initialize()
    if args.audit:
        audit()
        return
    tasks=json.loads(Path(args.tasks).read_text())
    for i,task in enumerate(tasks):
        r=run(task)
        print(json.dumps({'done':i+1,'total':len(tasks),**task,'val_mse':r['validation']['mse'],'test_mse':r['test']['mse'],'best_epoch':r['best_epoch'],'seconds':round(r['seconds'],1)}),flush=True)
    print('SHARD_COMPLETE',flush=True)


if __name__=='__main__':
    main()
