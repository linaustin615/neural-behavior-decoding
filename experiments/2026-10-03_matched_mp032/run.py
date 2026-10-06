"""Six fixed-budget MP032 development fits matched to the saved ridge study."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
BASE = ROOT.parent/'2026-10-03_development_baselines'
REPLICATION = ROOT.parent/'2026-10-02_stringer_replication'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def freeze():
    require(not (ROOT/'protocol.json').exists(),'Already frozen')
    baseline=read(BASE/'protocol.json')
    replication=read(REPLICATION/'protocol.json')
    cohort=next(r for r in baseline['cohort'] if r['mouse']=='MP032')
    files=[BASE/'run.py',BASE/'protocol.json',BASE/'results.json',BASE/'MP032_ids.npy',REPLICATION/'run.py',REPLICATION/'protocol.json']
    files += [BASE/f'MP032_f{f}.npz' for f in [1,2,3]]
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='Exploratory matched development diagnosis; no confirmatory p-values',
        cohort=cohort,seeds=[10,11],new_fits=6,epochs=24,
        architecture='Archived unrestricted GroupCandidate:2048 cells,8-bin history,16 global summaries,32 width,4 heads,MLP activity+ID,zero positions;81377 parameters',
        matching='Reuse exact baseline cell IDs, splits and normalization. Baseline targets retained in float64 for scoring; float32 network inputs/training targets. Verify exact normalized values before conversion and archived validation labels.',
        training=dict(batch=32,lr=.001,weight_decay=.01,cosine_min_lr=.0001,gradient_clip=1.,loss='Unbounded normalized-speed MSE',optimizer='AdamW'),
        stopping='All6 fits run exactly24 epochs. Also record shadow stop from inherited minimum12/patience7 rule using bounded validation MSE including epoch0. No adaptive extensions.',
        selections=['minimum bounded validation MSE including epoch0','minimum raw validation MSE including epoch0','inherited early-stopping selection','final epoch24'],
        selection_limit='All displayed validation-selected minima are optimistically development-tuned; no untouched outcome scoring. Train loss uses full eval-mode predictions as well as stochastic batch loss.',
        diagnostics=['Per-epoch train/validation raw and bounded MSE,R2,mean bias,error variance,prediction mean/std,correlation,clipping fraction',
                     'Zero speed,training mean and all4 archived ridge strengths on identical targets',
                     'Attention gradients finite/nonzero and final parameter changes',
                     'Fixed100-bin circular shift of selected validation predictions as a descriptive alignment control only; not a null significance test',
                     'Training90th-percentile speed coverage and training-standardized per-cell mean shifts; descriptive only'],
        archive='All25 epoch checkpoints and train/validation predictions per fit; reload each unique selected/final checkpoint on full validation and audit saved metrics independently',
        stop_rule='Six fixed fits only; no grouping sweep,hyperparameter expansion,other mice,old evaluation-tail scoring or publication',
        application_hashes=replication['application_hashes'],reference_hashes=replication['reference_sources'],
        inherited_hashes={str(f.relative_to(PROJECT)):digest(f) for f in files},source_sha256=digest(Path(__file__)),
        packages={k:version(k) for k in ['numpy','scipy','torch','scikit-learn','threadpoolctl']})
    write(ROOT/'protocol.json',p)


def verify(p):
    require(digest(Path(__file__))==p['source_sha256'],'Runner changed')
    for category in ['application_hashes','reference_hashes','inherited_hashes']:
        for f,h in p[category].items():
            require(digest(PROJECT/f)==h,'Source/artifact changed: '+f)


def stats_score(prediction,target,stats):
    pred=np.asarray(prediction,dtype=np.float64)
    y=np.asarray(target,dtype=np.float64)
    require(pred.shape==y.shape and np.isfinite(pred).all(),'Invalid prediction')
    lower=-float(stats['speed_mean'])/float(stats['speed_std'])
    bounded=np.maximum(pred,lower)
    result={}
    for name,v in [('raw',pred),('bounded',bounded)]:
        error=v-y
        mse=float(np.mean(error**2));bias=float(error.mean());ev=float(np.var(error))
        yp=y-y.mean();vp=v-v.mean()
        denominator=float(np.sqrt(np.sum(yp**2)*np.sum(vp**2)))
        result[name]=dict(mse=mse,r2=1-mse/float(np.var(y)),bias_normalized=bias,
            bias_speed_units=bias*float(stats['speed_std']),bias_squared_fraction=bias**2/mse if mse>0 else 0.,
            error_variance=ev,predicted_speed_mean=float(v.mean()*stats['speed_std']+stats['speed_mean']),
            predicted_speed_std=float(v.std()*stats['speed_std']),correlation=float(np.sum(yp*vp)/denominator) if denominator>1e-12 else None)
        require(abs(mse-(bias*bias+ev))<1e-9*max(1.,mse),'Bias decomposition mismatch')
    result['clipped_fraction']=float(np.mean(pred<lower))
    return result


def selections(history):
    best=history[0]['validation']['bounded']['mse'];best_epoch=0;stale=0
    stop=None;selected=None
    for row in history[1:]:
        value=row['validation']['bounded']['mse']
        if value<best:
            best,best_epoch,stale=value,row['epoch'],0
        else:
            stale+=1
        if row['epoch']>=12 and stale>=7:
            stop,selected=row['epoch'],best_epoch
            break
    if stop is None:
        stop=history[-1]['epoch'];selected=min(history,key=lambda v:v['validation']['bounded']['mse'])['epoch']
    return dict(bounded=min(history,key=lambda v:v['validation']['bounded']['mse'])['epoch'],
                raw=min(history,key=lambda v:v['validation']['raw']['mse'])['epoch'],
                early_stop_selected=selected,early_stop_epoch=stop,final=history[-1]['epoch'])


def predict(net,x):
    net.eval()
    pos=torch.zeros(x.shape[1],3);ids=torch.arange(x.shape[1])
    with torch.no_grad():
        pred=torch.cat([net(x[first:first+128],pos,ids) for first in range(0,len(x),128)]).numpy()
    require(np.isfinite(pred).all(),'Nonfinite network output')
    return pred


def record_fold(activity,speed,split,stats,ids):
    end=split['train'][1];start,stop=split['validation']
    mean_shift=np.abs((activity[:,start:stop].mean(1,dtype=np.float64)-stats['activity_mean'][:,0])/stats['activity_std'][:,0])
    cutoff=float(np.quantile(speed[31:end],.9))
    v=speed[start+31:stop]
    return dict(fold=split['fold'],ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
        training_speed90=cutoff,validation_fraction_above_training90=float(np.mean(v>cutoff)),
        cell_mean_shift_in_training_sd_quantiles=np.quantile(mean_shift,[.5,.9,.99,1]).tolist(),
        cells_mean_shift_over1sd=int(np.sum(mean_shift>1)),target_mean=float(v.mean()),target_std=float(v.std()))


def fit(task,x,y,xv,yv,stats,p,old):
    name=f"f{task['fold']}_s{task['seed']}"
    out=ROOT/name
    out.mkdir()
    net=old.model(task['seed'],'global',np.zeros(2048,dtype=np.int64))
    require(sum(v.numel() for v in net.parameters())==81377,'Wrong model size')
    require(not net.group_mask.any(),'Unrestricted attention unexpectedly masked')
    keys=['readin.in_proj_weight','base.encoder.transformer.self_attn.in_proj_weight']
    initial={k:net.state_dict()[k].clone() for k in keys}
    gradients={k:0. for k in keys}
    initial_hash=old.reference().state_hash(net)
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,p['epochs'],eta_min=.0001)
    loader=DataLoader(TensorDataset(x,y.float(),torch.arange(len(y))),batch_size=32,shuffle=True,
                      generator=torch.Generator().manual_seed(task['seed']))
    pos=torch.zeros(2048,3);ids=torch.arange(2048)
    history=[];train_predictions=[];val_predictions=[];order_hashes=[]
    for epoch in range(p['epochs']+1):
        loss_sum=0.
        if epoch:
            net.train();order=hashlib.sha256()
            for xb,yb,indices in loader:
                order.update(indices.numpy().tobytes())
                optimizer.zero_grad(set_to_none=True)
                loss=nn.functional.mse_loss(net(xb,pos,ids),yb)
                require(torch.isfinite(loss).item(),'Nonfinite loss')
                loss.backward()
                for key,value in net.named_parameters():
                    if key in gradients:
                        require(value.grad is not None and torch.isfinite(value.grad).all(),'Invalid attention gradient')
                        gradients[key]=max(gradients[key],float(value.grad.norm()))
                nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                loss_sum+=loss.item()*len(yb)
            require(all(torch.isfinite(v).all().item() for v in net.parameters()),'Nonfinite parameters')
            order_hashes.append(order.hexdigest());scheduler.step()
        pt=predict(net,x);pv=predict(net,xv)
        history.append(dict(epoch=epoch,training=stats_score(pt,y.numpy(),stats),validation=stats_score(pv,yv.numpy(),stats),
                            batch_training_mse=loss_sum/len(y) if epoch else None))
        train_predictions.append(pt);val_predictions.append(pv)
        torch.save(dict(task=task,epoch=epoch,state_dict=net.state_dict()),out/f'epoch{epoch:02d}.pt')
        write(out/'history.json',history)
        print(json.dumps(dict(task=name,epoch=epoch,train_mse=history[-1]['training']['raw']['mse'],validation_mse=history[-1]['validation']['bounded']['mse'])),flush=True)
    chosen=selections(history)
    for epoch in set(chosen[k] for k in ['bounded','raw','early_stop_selected','final']):
        restored=old.model(task['seed'],'global',np.zeros(2048,dtype=np.int64))
        restored.load_state_dict(torch.load(out/f'epoch{epoch:02d}.pt',map_location='cpu',weights_only=True)['state_dict'])
        np.testing.assert_array_equal(predict(restored,xv),val_predictions[epoch])
    changes={k:float((net.state_dict()[k]-initial[k]).norm()) for k in keys}
    require(all(v>0 for v in gradients.values()) and all(v>0 for v in changes.values()),'Attention did not update')
    np.savez_compressed(out/'predictions.npz',training=np.stack(train_predictions),validation=np.stack(val_predictions),
                        train_target=y.numpy(),validation_target=yv.numpy())
    #audit every saved epoch with an independent scalar MSE expression
    with np.load(out/'predictions.npz') as saved:
        lower=-float(stats['speed_mean'])/float(stats['speed_std'])
        for epoch in range(p['epochs']+1):
            for split,targetkey in [('training','train_target'),('validation','validation_target')]:
                for metric in ['raw','bounded']:
                    values=saved[split][epoch]
                    if metric=='bounded':
                        values=[max(float(v),lower) for v in values]
                    mse=sum((float(v)-float(t))**2 for v,t in zip(values,saved[targetkey]))/len(values)
                    require(abs(mse-history[epoch][split][metric]['mse'])<1e-9*max(1.,mse),'Saved MSE mismatch')
    selected=val_predictions[chosen['bounded']]
    result=dict(task=task,selections=chosen,initial_state_sha256=initial_hash,attention_max_gradient=gradients,
        attention_final_change=changes,batch_order_sha256=order_hashes,
        shifted100_validation=stats_score(np.roll(selected,100),yv.numpy(),stats),
        checks=dict(finite_training=True,full_selected_checkpoint_reloads=True,all_epoch_saved_metrics=True,attention_updates=True))
    write(out/'result.json',result)
    return result


def run():
    p=read(ROOT/'protocol.json');verify(p)
    require(not (ROOT/'started.json').exists(),'Study already started; inspect artifacts before recovery')
    write(ROOT/'started.json',dict(utc=datetime.now(timezone.utc).isoformat()))
    base=module('matched_baseline',BASE/'run.py')
    old=module('matched_replication',REPLICATION/'run.py')
    rec=p['cohort']
    raw=loadmat(PROJECT/rec['file'],variable_names=['Fsp','beh'],simplify_cells=True)
    activity,speed=base.bin_prefix(raw['Fsp'],np.asarray(raw['beh']['runSpeed']).reshape(-1),rec['development_stop'])
    del raw
    ids=np.load(BASE/'MP032_ids.npy');activity=activity[ids]
    outputs=[];diagnostics=[]
    for split in rec['folds']:
        arrays,stats=base.matrices(activity,speed,split)
        with np.load(BASE/f"MP032_f{split['fold']}.npz") as saved:
            for key,value in stats.items():
                np.testing.assert_array_equal(saved[key],value)
            np.testing.assert_array_equal(saved['target'],arrays['validation'][1].numpy())
        x,y=arrays['train'];xv,yv=arrays['validation']
        x=x.float().reshape(len(y),2048,8);xv=xv.float().reshape(len(yv),2048,8)
        del arrays
        diagnostics.append(record_fold(activity,speed,split,stats,ids))
        for seed in p['seeds']:
            outputs.append(fit(dict(fold=split['fold'],seed=seed),x,y,xv,yv,stats,p,old))
        del x,y,xv,yv
    for seed in p['seeds']:
        require(len({r['initial_state_sha256'] for r in outputs if r['task']['seed']==seed})==1,'Seed initialization differs across folds')
    verify(p)
    write(ROOT/'results.json',dict(runs=outputs,fold_diagnostics=diagnostics,confirmation=False))
    write(ROOT/'audit.json',dict(passed=True,fits=6,epochs_per_fit=24,baseline_ids_statistics_targets_matched=True,
        selected_checkpoints_reloaded=True,all_epoch_saved_metrics_verified=True,attention_updated=True,
        application_and_previous_artifacts_unchanged=True,old_evaluation_examples_constructed=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['freeze','run'])
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        {'freeze':freeze,'run':run}[args.action]()
