"""Frozen dynamics-pretraining baseline and matched 2x2 decoding experiment."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from threadpoolctl import threadpool_limits

from models import Dynamics,PooledMLP,NEURONS,CONTEXT,PATCH

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
FAIR=ROOT.parent/'2026-10-03_fair_comparison'
BROAD=ROOT.parent/'2026-10-03_broad_screen'
MICE=['MP030','MP032','MP033','MP034']
SEEDS=[10,11,12]
KINDS=['attention','mixer']
ARMS=['attention_scratch','attention_pretrained','mixer_scratch','mixer_pretrained','pooled_mlp']
REGULARIZERS=[.001,.01,.1,1.]


def read(path):return json.loads(path.read_text())


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition,message):
    if not condition:raise RuntimeError(message)


def predict(net,x,task):
    net.eval()
    with torch.no_grad():values=torch.cat([net(x[i:i+64],task) for i in range(0,len(x),64)]).numpy()
    require(np.isfinite(values).all(),'Nonfinite prediction');return values


def mse(pred,y,lower=None):
    pred=np.asarray(pred,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    require(pred.shape==y.shape and np.isfinite(pred).all(),'Invalid metric arrays')
    if lower is not None:pred=np.maximum(pred,lower)
    return float(np.mean((pred-y)**2))


def windows(sequence,horizon=0):
    return np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(sequence,CONTEXT+horizon,axis=0).transpose(0,1,2))


def recover(path,indices):
    old=np.load(path,mmap_mode='r')[:,indices]
    require(np.array_equal(old[1:,:,:-1],old[:-1,:,1:]),'Cached temporal overlap mismatch')
    sequence=np.concatenate([old[0,:,:-1].T,old[:,:,-1]],axis=0)
    return sequence


def check():
    torch.manual_seed(801);x=torch.randn(3,NEURONS,CONTEXT);y=torch.randn(3,NEURONS,PATCH);rows=[]
    for kind in KINDS:
        net=Dynamics(kind,10);net.eval()
        changed=x.clone();changed[:,:,16:]+=torch.randn_like(changed[:,:,16:])*5
        with torch.no_grad():torch.testing.assert_close(net.encode(x)[:,:,:4],net.encode(changed)[:,:,:4],rtol=0,atol=0)
        require(net(x,'forecast').shape==y.shape and net(x).shape==(3,),'Wrong shape')
        initial=deepcopy(net.state_dict());opt=torch.optim.AdamW(net.parameters(),lr=.001)
        net.train();loss=(net(x,'forecast')-y).square().mean();loss.backward()
        watched='population.mix.in_proj_weight' if kind=='attention' else 'population.mix.0.weight'
        require(float(dict(net.named_parameters())[watched].grad.norm())>0,'Missing population gradient')
        require(all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()),'Invalid gradient')
        opt.step();require(not torch.equal(initial[watched],net.state_dict()[watched]),'No update')
        opt.zero_grad(set_to_none=True);(net(x)-torch.arange(3).float()).square().mean().backward()
        require(float(net.speed[-1].weight.grad.norm())>0,'Missing speed gradient')
        clone=Dynamics(kind,10);clone.load_state_dict(net.state_dict());np.testing.assert_array_equal(predict(net,x,'forecast'),predict(clone,x,'forecast'))
        if kind=='mixer':require(not any(isinstance(m,nn.MultiheadAttention) for m in net.modules()),'Attention in mixer')
        rows.append(dict(kind=kind,parameters=sum(p.numel() for p in net.parameters())))
    require(abs(rows[0]['parameters']/rows[1]['parameters']-1)<.01,'Capacity mismatch')
    a,b=Dynamics('attention',10),Dynamics('mixer',10)
    for prefix in ['patch','identity','speed','forecast']:
        for key,v in getattr(a,prefix).state_dict().items():torch.testing.assert_close(v,getattr(b,prefix).state_dict()[key],rtol=0,atol=0)
    raw=np.arange(70*NEURONS).reshape(70,NEURONS)
    w=windows(raw,PATCH);np.testing.assert_array_equal(w[0,:,:CONTEXT],raw[:CONTEXT].T);np.testing.assert_array_equal(w[0,:,CONTEXT:],raw[CONTEXT:CONTEXT+PATCH].T)
    mlp=PooledMLP(10);require(mlp(x).shape==(3,),'MLP shape');mlp(x).sum().backward();require(all(p.grad is None or torch.isfinite(p.grad).all() for p in mlp.parameters()),'MLP gradient')
    rows.append(dict(kind='pooled_mlp',parameters=sum(p.numel() for p in mlp.parameters())))
    write(ROOT/'selfcheck.json',dict(passed=True,models=rows,patch_causality_exact=True,shape_gradients_updates_reload=True,shared_input_output_initialization_exact=True,forecast_window_boundaries=True))


def freeze():
    require(not (ROOT/'protocol.json').exists(),'Already frozen')
    require(read(ROOT/'selfcheck.json')['passed'],'Model checks not passed')
    refs=[FAIR/'protocol.json',BROAD/'protocol.json',BROAD/'results.json']
    for mouse in MICE:
        refs.extend(FAIR/mouse/f for f in ['train_x.npy','train_y.npy','selection_x.npy','selection_y.npy','later_raw.npz','statistics.npz','metadata.json'])
        refs.append(BROAD/mouse/'later_predictions.npz')
    write(ROOT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),mice=MICE,seeds=SEEDS,arms=ARMS,
        scope='Reusable small literature-informed baseline on historically inspected Stringer data;not a novel architecture or independent confirmation',
        budget='24 neural pretraining fits +48 sequence-decoding fits +12 matched pooled-MLP fits;each24epochs,three seeds,four mice. No adaptive grid or extra training.',
        data='128 fixed columns sampled without labels from archived2048-cell pool with seed81203. Recover normalized sequences from cached overlapping8-bin windows,starting at original segment offset24. New32-bin windows use original speed targets[24:]. Forecast next4neural bins only within same segment. Original100-bin split gaps exceed32+4 context/horizon. Existing train-only normalization unchanged.',
        representation='Eight4-bin continuous patches per neuron,width16. One causal temporal block per neuron followed by one population block per patch;retain128x8 representations until readout. No coordinates or cross-mouse ID alignment.',
        models='Attention:2head temporal and population attention. Mixer:causal static8x8 time mixing plus16/30/16 channel MLP;population static128/4/128 mixer. Both use matching residual feature MLPs. Same patch/ID/time embeddings and readouts initialized identically. Fixed-neuron mixer need not be permutation invariant;parameter matching does not match compute.',
        pretraining='Predict next4standardized neural bins from preceding32,continuous MSE,no speed targets. Select epoch0..24 by earlier neural forecast MSE only. No later-segment pretraining. This is terminal next-block forecasting,not full CAPT replication or free-running generation.',
        downstream='Same80-feature readout:mean last-patch latent16 +32 population means +32 population SDs,then LayerNorm/64hidden/GELU/scalar. Pretrained starts from neural-selected checkpoint;scratch from common initial state. All encoder and speed-head parameters fine-tune jointly24epochs. PooledMLP uses same128cells/32bins and amplitude bypass. Each selects epoch0..24 by earlier bounded speed MSE.',
        training='AdamW lr0.001,weight_decay0.01,batch32,clip1,cosine24 eta_min0.0001;identical batch order per seed within stage. Extra24pretraining epochs are explicit budget difference. No MLP/transformer success presumed.',
        baselines='Matched-input ridge speed decoder with loss-normalized lambda.001/.01/.1/1 selected earlier. Forecast:training mean,persistence,per-neuron32lag ridge,and training-PCA16 population32lag ridge;AR lambdas.01/.1/1 selected earlier. Reuse old2048cell8bin ridge/kernel/MLP on exact aligned targets as unequal-input references only.',
        gate='Attention-pretrained must improve equal-weight mean relative speed MSE>=5% vs attention-scratch,mixer-pretrained,matched pooledMLP,and matched ridge;win>=3/4 mice vs each,>=8/12 paired seed comparisons vs each neural comparator;no mouse>25% worse than matched pooledMLP. Separately,forecast utility needs>=5% mean relative gain vs earlier-selected strongest simple forecast baseline and>=3/4 mouse wins. Neither gate proves significance.',
        secondary='Pretraining effect for mixer;difference between attention and mixer pretraining effects;own pre-finetuning epoch0;fixed3seed raw ensembles;forecast persistence/AR/mean and zero speed;selected epochs;leave-one-mouse-out;4000 hierarchical mouse/seed/circular100-bin bootstrap with98.75% descriptive intervals for4primary speed contrasts.',
        stopping='Complete the fixed baseline study and reports. No application edits,publication,hyperparameter extension,new datasets or old evaluation tails.',
        application_hashes=read(BROAD/'protocol.json')['application_hashes'],references={str(p.relative_to(PROJECT)):digest(p) for p in refs},sources={name:digest(ROOT/name) for name in ['run.py','models.py']}))


def verify():
    p=read(ROOT/'protocol.json')
    for name,h in p['sources'].items():require(digest(ROOT/name)==h,'Source changed '+name)
    for field in ['references','application_hashes']:
        for name,h in p[field].items():require(digest(PROJECT/name)==h,'Reference changed '+name)
    return p


def linear_fit(x,y,xv,lam):
    x=torch.as_tensor(x,dtype=torch.float64);y=torch.as_tensor(y,dtype=torch.float64);xv=torch.as_tensor(xv,dtype=torch.float64)
    xm=x.mean(0);ym=y.mean(0);z=x-xm;target=y-ym;n=len(x)
    if x.shape[1]>n:
        gram=z@z.T+n*lam*torch.eye(n,dtype=x.dtype);alpha=torch.linalg.solve(gram,target);weight=z.T@alpha
        require(float((gram@alpha-target).norm()/target.norm().clamp_min(1e-12))<1e-7,'Dual ridge residual')
    else:
        gram=z.T@z+n*lam*torch.eye(x.shape[1],dtype=x.dtype);rhs=z.T@target;weight=torch.linalg.solve(gram,rhs)
        require(float((gram@weight-rhs).norm()/rhs.norm().clamp_min(1e-12))<1e-7,'Primal ridge residual')
    prediction=(xv-xm)@weight+ym
    independent=np.einsum('ij,j...->i...', (xv-xm).numpy(),weight.numpy(),optimize=False)+ym.numpy()
    np.testing.assert_allclose(prediction.numpy(),independent,rtol=1e-8,atol=1e-8)
    return prediction.numpy(),dict(weight=weight.numpy(),center=xm.numpy(),mean=ym.numpy())


def prepare():
    p=verify();require(not (ROOT/'prepared.json').exists(),'Already prepared');indices=np.sort(np.random.default_rng(81203).choice(2048,NEURONS,replace=False));rows=[]
    for mouse in MICE:
        dest=ROOT/mouse;dest.mkdir();np.save(dest/'columns.npy',indices)
        sequences={name:recover(FAIR/mouse/f'{name}_x.npy',indices) for name in ['train','selection']}
        arrays={}
        for name,seq in sequences.items():
            x=windows(seq);y=np.load(FAIR/mouse/f'{name}_y.npy')[24:];require(len(x)==len(y),'Behavior alignment')
            f=windows(seq,PATCH);arrays[name]=(x,y,f[:,:,CONTEXT:]);np.testing.assert_array_equal(f[:,:,:CONTEXT],x[:-PATCH])
            np.save(dest/f'{name}_x.npy',x);np.save(dest/f'{name}_y.npy',y);np.save(dest/f'{name}_future.npy',f[:,:,CONTEXT:])
        x,y,future=arrays['train'];xv,yv,fv=arrays['selection'];meta=read(FAIR/mouse/'metadata.json');lower=-meta['speed_mean']/meta['speed_std']
        speed=[]
        for lam in REGULARIZERS:
            pred,state=linear_fit(x.reshape(len(x),-1),y,xv.reshape(len(xv),-1),lam)
            np.savez_compressed(dest/f'ridge_{lam}.npz',prediction=pred,**state);speed.append(dict(lam=lam,mse=mse(pred,yv,lower)))
        forecast=[dict(kind='mean',mse=mse(np.zeros_like(fv),fv)),dict(kind='persistence',mse=mse(np.repeat(xv[:-PATCH,:,-1:],PATCH,axis=2),fv))]
        torch.manual_seed(81203);activity=torch.from_numpy(sequences['train']).double();center=activity.mean(0);_,_,projection=torch.pca_lowrank(activity-center,q=16,center=False,niter=3)
        projection=projection.numpy();center=center.numpy()
        def pc_features(a):return np.einsum('bnw,nk->bwk',a-center[None,:,None],projection).reshape(len(a),-1)
        xp=pc_features(x[:-PATCH]);xvp=pc_features(xv[:-PATCH])
        for lam in [.01,.1,1.]:
            pred,state=linear_fit(xp,future.reshape(len(future),-1),xvp,lam)
            np.savez_compressed(dest/f'population_ar_{lam}.npz',projection=projection,pca_center=center,prediction=pred.reshape(fv.shape),**state)
            forecast.append(dict(kind='population_ar',lam=lam,mse=mse(pred.reshape(fv.shape),fv)))
            xt=torch.from_numpy(x[:-PATCH].transpose(1,0,2)).double();yt=torch.from_numpy(future.transpose(1,0,2)).double();xtv=torch.from_numpy(xv[:-PATCH].transpose(1,0,2)).double()
            xm=xt.mean(1,keepdim=True);ym=yt.mean(1,keepdim=True);z=xt-xm;target=yt-ym
            gram=z.transpose(1,2)@z+len(future)*lam*torch.eye(CONTEXT,dtype=torch.float64);rhs=z.transpose(1,2)@target;weight=torch.linalg.solve(gram,rhs)
            require(float((gram@weight-rhs).norm()/rhs.norm())<1e-7,'Per-neuron AR residual')
            pred=((xtv-xm)@weight+ym).numpy().transpose(1,0,2)
            alternate=np.einsum('bnw,nwh->bnh',xv[:-PATCH]-xm.numpy().transpose(1,0,2),weight.numpy(),optimize=False)+ym.numpy().transpose(1,0,2)
            np.testing.assert_allclose(pred,alternate,rtol=1e-8,atol=1e-8)
            np.savez_compressed(dest/f'neuron_ar_{lam}.npz',weight=weight.numpy(),center=xm.numpy(),mean=ym.numpy(),prediction=pred)
            forecast.append(dict(kind='neuron_ar',lam=lam,mse=mse(pred,fv)))
        baseline=dict(speed_settings=speed,speed_choice=min(speed,key=lambda r:r['mse']),forecast_settings=forecast,forecast_choice=min(forecast,key=lambda r:r['mse']))
        write(dest/'baselines.json',baseline);rows.append(dict(mouse=mouse,train_n=len(y),selection_n=len(yv),forecast_train_n=len(future),forecast_selection_n=len(fv),**baseline))
        print(json.dumps(rows[-1]),flush=True)
    files={str(f.relative_to(ROOT)):digest(f) for mouse in MICE for f in (ROOT/mouse).iterdir() if f.is_file()}
    write(ROOT/'prepared.json',dict(passed=True,rows=rows,files=files))


def fit_one(mouse,kind,seed,stage,pretrained=False):
    dest=ROOT/mouse;label=f'{kind}_pretrain_s{seed}' if stage=='forecast' else f'{kind}_{"pretrained" if pretrained else "scratch"}_s{seed}' if kind!='pooled_mlp' else f'pooled_mlp_s{seed}'
    out=dest/label;out.mkdir();net=PooledMLP(seed) if kind=='pooled_mlp' else Dynamics(kind,seed)
    if pretrained:net.load_state_dict(torch.load(dest/f'{kind}_pretrain_s{seed}'/'selected.pt',weights_only=True))
    x=torch.from_numpy(np.load(dest/'train_x.npy'));xs=torch.from_numpy(np.load(dest/'selection_x.npy'))
    if stage=='forecast':
        x=x[:-PATCH];xs=xs[:-PATCH];yt=np.load(dest/'train_future.npy');ys=np.load(dest/'selection_future.npy');lower=None
    else:
        yt=np.load(dest/'train_y.npy');ys=np.load(dest/'selection_y.npy');meta=read(FAIR/mouse/'metadata.json');lower=-meta['speed_mean']/meta['speed_std']
    y=torch.from_numpy(yt.astype(np.float32));prediction=predict(net,xs,stage);best=mse(prediction,ys,lower);best_prediction=prediction.copy();chosen=0;history=[dict(epoch=0,selection_mse=best)];predictions=[prediction]
    torch.save(net.state_dict(),out/'initial.pt');torch.save(net.state_dict(),out/'selected.pt');start=time.monotonic()
    gen=torch.Generator().manual_seed(seed);torch.manual_seed(seed+9000)
    loader=DataLoader(TensorDataset(x,y,torch.arange(len(y))),batch_size=32,shuffle=True,generator=gen)
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,24,eta_min=.0001)
    grad_max=0.
    watched=['activity.0.weight','head.2.weight'] if kind=='pooled_mlp' else ['temporal.mix.in_proj_weight','population.mix.in_proj_weight'] if kind=='attention' else ['temporal.weight','population.mix.0.weight']
    gradients={k:0. for k in watched}
    for epoch in range(1,25):
        net.train();total=0.;order=hashlib.sha256()
        for xb,yb,ib in loader:
            order.update(ib.numpy().tobytes());opt.zero_grad(set_to_none=True);loss=(net(xb,stage)-yb).square().mean();require(torch.isfinite(loss),'Nonfinite loss');loss.backward()
            for k in watched:
                g=dict(net.named_parameters())[k].grad;require(g is not None and torch.isfinite(g).all(),'Missing/nonfinite gradient '+k);gradients[k]=max(gradients[k],float(g.norm()))
            norm=nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);grad_max=max(grad_max,float(norm));opt.step();total+=float(loss.detach())*len(yb)
        scheduler.step();prediction=predict(net,xs,stage);score=mse(prediction,ys,lower)
        if score<best:best=score;chosen=epoch;best_prediction=prediction.copy();torch.save(net.state_dict(),out/'selected.pt')
        history.append(dict(epoch=epoch,training_mse=total/len(y),selection_mse=score,batch_order_hash=order.hexdigest()));write(out/'history.json',history)
        if stage!='forecast':predictions.append(prediction)
    require(grad_max>0 and all(v>0 for v in gradients.values()),'No monitored learning')
    net.load_state_dict(torch.load(out/'selected.pt',weights_only=True));selected=predict(net,xs,stage);require(abs(mse(selected,ys,lower)-best)<1e-10,'Reload score differs')
    np.testing.assert_array_equal(selected,best_prediction)
    if stage=='forecast':np.savez_compressed(out/'selection_predictions.npz',selected=selected,initial=predictions[0],target=ys)
    else:
        np.testing.assert_array_equal(selected,predictions[chosen]);np.savez_compressed(out/'selection_predictions.npz',prediction=np.stack(predictions),target=ys)
    result=dict(mouse=mouse,kind=kind,seed=seed,stage=stage,pretrained=pretrained,epochs=24,selected_epoch=chosen,selection_mse=best,initial_mse=history[0]['selection_mse'],grad_max=grad_max,gradients=gradients,selected_reload_exact=True,elapsed_seconds=time.monotonic()-start)
    write(out/'result.json',result);print(json.dumps(result),flush=True)


def pretrain(mouse):
    verify()
    for seed in SEEDS:
        for kind in KINDS:fit_one(mouse,kind,seed,'forecast')
    write(ROOT/mouse/'pretraining_finished.json',dict(complete=True))


def train(mouse):
    verify();require((ROOT/'pretraining_lock.json').exists(),'Pretraining not locked')
    for name,h in read(ROOT/'pretraining_lock.json')['files'].items():require(digest(ROOT/name)==h,'Pretraining checkpoint changed')
    for seed in SEEDS:
        for kind in KINDS:
            for pretrained in [False,True]:fit_one(mouse,kind,seed,'speed',pretrained)
        fit_one(mouse,'pooled_mlp',seed,'speed')
    write(ROOT/mouse/'training_finished.json',dict(complete=True))


def lock(stage):
    verify();filename='pretraining_lock.json' if stage=='forecast' else 'selection_lock.json';require(not (ROOT/filename).exists(),'Already locked');records=[];files={};orders={};checked=0
    for mouse in MICE:
        for path in sorted((ROOT/mouse).glob('*/result.json')):
            r=read(path)
            if r['stage']!=stage:continue
            out=path.parent;history=read(out/'history.json');require(len(history)==25,'Incomplete history')
            order=[a['batch_order_hash'] for a in history[1:]];key=(mouse,r['seed'])
            if key in orders:require(orders[key]==order,'Batch order mismatch')
            orders[key]=order
            meta=read(FAIR/mouse/'metadata.json');lower=None if stage=='forecast' else -meta['speed_mean']/meta['speed_std']
            with np.load(out/'selection_predictions.npz') as saved:
                if stage=='forecast':
                    require(abs(mse(saved['selected'],saved['target'])-r['selection_mse'])<1e-10,'Forecast selected mismatch');checked+=1
                else:
                    errors=[mse(v,saved['target'],lower) for v in saved['prediction']];np.testing.assert_allclose(errors,[h['selection_mse'] for h in history],rtol=1e-10,atol=1e-10);require(int(np.argmin(errors))==r['selected_epoch'],'Selection mismatch');checked+=len(errors)
            require(int(np.argmin([h['selection_mse'] for h in history]))==r['selected_epoch'],'Wrong best epoch')
            records.append(dict(directory=str(out.relative_to(ROOT)),**r));files[str((out/'selected.pt').relative_to(ROOT))]=digest(out/'selected.pt')
    require(len(records)==(24 if stage=='forecast' else 60),'Wrong fit count')
    write(ROOT/filename,dict(locked_utc=datetime.now(timezone.utc).isoformat(),records=records,files=files,predictions_checked=checked,matched_batch_orders=True,later_scored=False))


def baseline_predict(dest,choice,x):
    if choice['kind']=='mean':return np.zeros((len(x),NEURONS,PATCH))
    if choice['kind']=='persistence':return np.repeat(x[:,:,-1:],PATCH,axis=2)
    with np.load(dest/f"{choice['kind']}_{choice['lam']}.npz") as saved:
        if choice['kind']=='neuron_ar':return np.einsum('bnw,nwh->bnh',x-saved['center'].transpose(1,0,2),saved['weight'],optimize=False)+saved['mean'].transpose(1,0,2)
        features=np.einsum('bnw,nk->bwk',x-saved['pca_center'][None,:,None],saved['projection']).reshape(len(x),-1)
        return (np.einsum('ij,jk->ik',features-saved['center'],saved['weight'],optimize=False)+saved['mean']).reshape(len(x),NEURONS,PATCH)


def evaluate():
    p=verify();require(not (ROOT/'results.json').exists(),'Already evaluated')
    locks=[read(ROOT/'pretraining_lock.json'),read(ROOT/'selection_lock.json')]
    for locked in locks:
        for name,h in locked['files'].items():require(digest(ROOT/name)==h,'Checkpoint changed')
    for name,h in read(ROOT/'prepared.json')['files'].items():require(digest(ROOT/name)==h,'Prepared file changed')
    rows=[]
    for mouse in MICE:
        dest=ROOT/mouse;indices=np.load(dest/'columns.npy');meta=read(FAIR/mouse/'metadata.json');lower=-meta['speed_mean']/meta['speed_std']
        with np.load(FAIR/mouse/'later_raw.npz') as raw,np.load(FAIR/mouse/'statistics.npz') as st:
            seq=((raw['activity'][indices,24:]-st['activity_mean'][indices])/st['activity_std'][indices]).T.astype(np.float32)
            y=(raw['speed'][55:]-meta['speed_mean'])/meta['speed_std']
        x=windows(seq);full=windows(seq,PATCH);future=full[:,:,CONTEXT:];require(len(x)==len(y),'Later target length');xt=torch.from_numpy(x);predictions={};records=[];forecast_records=[]
        def add(label,pred,**extra):
            pred=np.asarray(pred,dtype=np.float64);error=mse(pred,y,lower)
            independent=sum((max(float(a),lower)-float(b))**2 for a,b in zip(pred,y))/len(y);require(abs(error-independent)<1e-10*max(1.,error),'Independent speed metric mismatch')
            predictions[label]=pred;records.append(dict(label=label,mse=error,**extra))
        baselines=read(dest/'baselines.json');lam=baselines['speed_choice']['lam']
        with np.load(dest/f'ridge_{lam}.npz') as saved:ridge=np.einsum('ij,j->i',x.reshape(len(x),-1)-saved['center'],saved['weight'],optimize=False)+saved['mean']
        add('ridge',ridge);add('zero',np.full(len(y),lower));add('mean',np.zeros(len(y)))
        with np.load(BROAD/mouse/'later_predictions.npz') as saved:
            np.testing.assert_array_equal(y,saved['target'][24:]);add('old_regression',saved['strong_baseline'][24:])
            for seed in SEEDS:add(f'old_mlp_s{seed}',saved[f'norm_mlp_s{seed}'][24:])
        for kind in ARMS:
            members=[]
            for seed in SEEDS:
                out=dest/f'{kind}_s{seed}';net=PooledMLP(seed) if kind=='pooled_mlp' else Dynamics(kind.split('_')[0],seed)
                net.load_state_dict(torch.load(out/'selected.pt',weights_only=True));pred=predict(net,xt,'speed');add(f'{kind}_s{seed}',pred,epoch=read(out/'result.json')['selected_epoch']);members.append(pred)
                net.load_state_dict(torch.load(out/'initial.pt',weights_only=True));add(f'{kind}_initial_s{seed}',predict(net,xt,'speed'))
            add(f'{kind}_ensemble',np.mean(members,axis=0))
        forecasts={}
        for choice in baselines['forecast_settings']:
            label=choice['kind']+('_'+str(choice['lam']) if 'lam' in choice else '')
            forecasts[label]=baseline_predict(dest,choice,x[:-PATCH]);forecast_records.append(dict(label=label,mse=mse(forecasts[label],future)))
        choice=baselines['forecast_choice'];label=choice['kind']+('_'+str(choice['lam']) if 'lam' in choice else '');forecasts['strong_baseline']=forecasts[label];forecast_records.append(dict(label='strong_baseline',mse=mse(forecasts[label],future)))
        for kind in KINDS:
            for seed in SEEDS:
                net=Dynamics(kind,seed);out=dest/f'{kind}_pretrain_s{seed}';net.load_state_dict(torch.load(out/'selected.pt',weights_only=True));pred=predict(net,xt[:-PATCH],'forecast');forecasts[f'{kind}_s{seed}']=pred
                forecast_records.append(dict(label=f'{kind}_s{seed}',mse=mse(pred,future),epoch=read(out/'result.json')['selected_epoch']))
        np.savez_compressed(dest/'later_predictions.npz',target=y,**predictions)
        np.savez_compressed(dest/'forecast_predictions.npz',target=future,**forecasts)
        rows.append(dict(mouse=mouse,n=len(y),forecast_n=len(future),speed=records,forecast=forecast_records,forecast_baseline=choice))
        print(json.dumps(dict(mouse=mouse,speed={a['label']:a['mse'] for a in records if 'initial' not in a['label']},forecast={a['label']:a['mse'] for a in forecast_records})),flush=True)
    write(ROOT/'results.json',dict(scope=p['scope'],rows=rows));verify()
    write(ROOT/'audit.json',dict(passed=True,pretraining_fits=24,speed_fits=60,pretraining_choices_locked_before_decoding=True,all_choices_locked_before_new_later_scoring=True,selected_checkpoints_reload=True,matched_batch_orders=True,exact_old_target_alignment=True,speed_selection_predictions_checked=1500,forecast_selected_predictions_checked=24,independent_speed_metrics=True,linear_optimality_and_predictions_checked=True,prepared_and_reference_hashes_unchanged=True,application_unchanged=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['check','freeze','prepare','pretrain','train','pretraining_lock','lock','evaluate']);parser.add_argument('--mouse');args=parser.parse_args()
    with threadpool_limits(limits=2):
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        if args.action=='pretrain':pretrain(args.mouse)
        elif args.action=='train':train(args.mouse)
        elif args.action=='pretraining_lock':lock('forecast')
        elif args.action=='lock':lock('speed')
        else:dict(check=check,freeze=freeze,prepare=prepare,evaluate=evaluate)[args.action]()
