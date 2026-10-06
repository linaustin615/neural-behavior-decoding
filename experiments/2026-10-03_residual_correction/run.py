"""Chronological out-of-fold residual correction of a frozen ridge decoder."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from scipy.io import loadmat
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
FAIR=ROOT.parent/'2026-10-03_fair_comparison'
BASE=ROOT.parent/'2026-10-03_development_baselines'
SEEDS=[10,11]
FAMILIES=['transformer','pooled_mlp']
PENALTIES=[.1,1.]


def read(path):
    return json.loads(path.read_text())


def write(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value,message):
    if not value:
        raise RuntimeError(message)


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def dependencies():
    fair=module('residual_fair',FAIR/'run.py')
    base,old=fair.dependencies()
    return fair,base,old


def model(fair,old,family,seed):
    net=fair.model(old,family,seed)
    head=net.base.head if family=='transformer' else net.head[-1]
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    return net


def predict(fair,net,x):
    return np.tanh(fair.predict(net,x)).astype(np.float64)


def oof_splits(first,end):
    cuts=[first,first+(end-first)//3,first+2*(end-first)//3,end]
    return [dict(fold=i+1,train=[0,cuts[i]],validation=[cuts[i]+32,cuts[i+1]]) for i in range(3)]


def check():
    fair,base,old=dependencies()
    counts={}
    torch.manual_seed(901)
    x=torch.randn(4,2048,8)
    baseline=torch.tensor([-.2,.3,.4,.7]);target=baseline+torch.tensor([.1,-.2,.3,-.1])
    for family in FAMILIES:
        net=model(fair,old,family,10)
        counts[family]=sum(v.numel() for v in net.parameters())
        np.testing.assert_array_equal(predict(fair,net,x),np.zeros(4))
        opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
        for step in range(3):
            net.train();opt.zero_grad(set_to_none=True)
            delta=torch.tanh(net(x,torch.zeros(2048,3),torch.arange(2048)))
            loss=((baseline+delta-target)**2).mean()+.1*(delta**2).mean()
            loss.backward()
            require(all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()),'Bad synthetic gradient')
            if step:
                watched=[p for name,p in net.named_parameters() if 'identity_embedding' in name or 'in_proj_weight' in name]
                require(all(p.grad is not None and p.grad.norm()>0 for p in watched),'Body fails to receive gradients after zero head opens')
            opt.step()
        delta=predict(fair,net,x)
        require(np.any(delta!=0) and np.max(np.abs(delta))<=1,'Correction does not open or violates bound')
    for split in oof_splits(1000,1600):
        train_last=split['train'][1]-1
        first_input=split['validation'][0]+24
        first_target=split['validation'][0]+31
        require(train_last<first_input<=first_target and first_target<split['validation'][1],'OOF boundary error')
    #a held-out target perturbation must not alter a fitted ridge prediction
    g=torch.Generator().manual_seed(52)
    a=torch.randn(30,5,generator=g,dtype=torch.float64)
    y=torch.randn(30,generator=g,dtype=torch.float64)
    b=torch.randn(7,5,generator=g,dtype=torch.float64)
    pred,_,weights,intercept,_=base.solve(a,y,b,[10.])
    expected=np.einsum('ij,j->i',b.numpy(),weights[:,0],optimize=False)+intercept[0]
    np.testing.assert_allclose(pred[:,0],expected,rtol=1e-12,atol=1e-12)
    original=torch.zeros(7,dtype=torch.float64);changed=original+100
    np.testing.assert_allclose((changed.numpy()-pred[:,0])-(original.numpy()-pred[:,0]),100)
    write(ROOT/'selfcheck.json',dict(passed=True,parameters=counts,zero_correction_exact=True,
          body_gradients_after_head_opens=True,bounded_corrections=True,chronological_boundaries=True,
          independent_ridge_prediction=True))


def freeze():
    require(not (ROOT/'protocol.json').exists(),'Already frozen')
    check()
    old=read(FAIR/'protocol.json');base=read(BASE/'protocol.json')
    cohort=[]
    for rec in old['cohort']:
        first=next(v for v in base['cohort'] if v['mouse']==rec['mouse'])['folds'][0]['train'][1]
        cohort.append(dict(rec,oof=oof_splits(first,rec['train'][1])))
    refs=[FAIR/'run.py',FAIR/'protocol.json',FAIR/'selection_lock.json',FAIR/'results.json',BASE/'run.py',BASE/'protocol.json']
    for rec in cohort:
        refs.extend([BASE/f"{rec['mouse']}_f2.npz",BASE/f"{rec['mouse']}_ids.npy"])
        refs.extend(FAIR/rec['mouse']/f for f in ['train_x.npy','train_y.npy','selection_x.npy','selection_y.npy','statistics.npz','metadata.json','later_raw.npz','later_predictions.npz'])
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='Exploratory historically inspected development data; no confirmatory significance',
        question='Can a bounded transformer correction improve frozen ridge without the prior large generalization failures, and outperform a matched nonattention correction?',
        cohort=cohort,seeds=SEEDS,families=FAMILIES,penalties=PENALTIES,new_neural_fits=32,new_oof_ridge_fits=12,epochs=24,
        baseline='Fixed lambda10 ridge, original full outer-training fit reused. Fixed lambda avoids selecting the OOF predictor hyperparameter using its future targets. Stronger selection-tuned ridge retained as mandatory additional comparison.',
        oof='Three expanding chronological prefix ridge fits per mouse. Earliest prefix is the archived cell-eligibility prefix. Each held-out block starts32 bins after its training end; windows use bins24..31, targets31 onward. Each prefix fits its own activity/target normalization; predictions converted to common outer-training target units. Fixed2048-cell pool eligible using earliest prefix only. No selection/later targets used to fit any OOF ridge.',
        correction='Same transformer/pooledMLP as previous comparison, zero final linear layer, delta=tanh(network_output) bounded to1 outer-training speed SD. No input clipping, temporal attention, new tokenizer or dynamic confidence gate.',
        objective='mean((OOF_ridge_prediction+delta-target)^2)+beta*mean(delta^2), beta0.1/1.0. Network sees only OOF training windows, normalized by outer-training statistics; their target labels are valid correction-training data.',
        training='AdamW lr0.001,weight_decay0.01,batch32,24epochs,cosine eta_min0.0001,gradient clip1; paired seed/batch orders across families/penalties',
        selection='For each seed/penalty choose minimum bounded earlier-selection MSE including epoch0 (exact ridge). Select one penalty per mouse/family by mean selected error across both seeds. Lock all32 fits before later scoring. Ties choose first epoch/penalty.',
        primary='Equal-weight mean per-mouse relative MSE reduction against fixed ridge10; contrast transformer-correction versus pooledMLP-correction. Average individual-seed errors, not ensemble predictions.',
        practical_gate='Transformer mean relative gain>=5% versus ridge10, positive in>=3/4 mice and>=6/8 seeds, no mouse mean more than10% worse than ridge10, and positive mean relative gain versus previously selected ridge. Exploratory gate, not significance.',
        attention_gate='Only call transformer correction specifically promising if practical gate passes and it has positive mean relative gain over pooledMLP correction with>=3/4 positive mouse means.',
        secondary='Per-seed errors, raw errors, zero/training-mean controls, selected-ridge comparison, trained epoch versus exact-ridge epoch0, correction magnitude, same relative-quiet/moving threshold as previous study, prior direct-network scores reused as descriptive context.',
        limitations='OOF correction data are fewer than full training data; OOF base models have shorter histories than the full refitted baseline, creating a residual-distribution mismatch. Historical outcome inspection,4mice,2seeds,one fixed split per mouse. No universal architecture claim.',
        stop='Exactly32 neural fits and12 prefix ridge fits; no outcome-driven extension, temporal-attention addition, old evaluation-tail access or application edits.',
        application_hashes=old['application_hashes'],reference_hashes={str(f.relative_to(PROJECT)):digest(f) for f in refs},source_sha256=digest(Path(__file__)))
    write(ROOT/'protocol.json',p)


def verify():
    p=read(ROOT/'protocol.json')
    require(digest(Path(__file__))==p['source_sha256'],'Runner changed')
    for field in ['application_hashes','reference_hashes']:
        for name,h in p[field].items():
            require(digest(PROJECT/name)==h,'Source changed: '+name)
    return p


def prepare():
    p=verify();fair,base,_=dependencies()
    require(not (ROOT/'prepared.json').exists(),'Already prepared')
    records=[]
    for rec in p['cohort']:
        dest=ROOT/rec['mouse'];dest.mkdir()
        raw=loadmat(PROJECT/rec['file'],variable_names=['Fsp','beh'],simplify_cells=True)
        activity,speed=base.bin_prefix(raw['Fsp'],np.asarray(raw['beh']['runSpeed']).reshape(-1),rec['train'][1])
        del raw
        ids=np.load(BASE/f"{rec['mouse']}_ids.npy")
        activity=activity[ids]
        meta=read(FAIR/rec['mouse']/'metadata.json')
        outer_y=np.load(FAIR/rec['mouse']/'train_y.npy')
        np.testing.assert_array_equal((speed[31:]-meta['speed_mean'])/meta['speed_std'],outer_y)
        predictions=[];indices=[];blocks=[]
        for split in rec['oof']:
            arrays,stats=base.matrices(activity,speed,split)
            x,y=arrays['train'];xv,yv=arrays['validation']
            pred,_,weights,intercept,errors=base.solve(x,y,xv,[10.])
            independent=np.einsum('ij,j->i',xv.numpy(),weights[:,0],optimize=False)+intercept[0]
            np.testing.assert_allclose(pred[:,0],independent,rtol=1e-10,atol=1e-10)
            converted=(pred[:,0]*stats['speed_std']+stats['speed_mean']-meta['speed_mean'])/meta['speed_std']
            idx=np.arange(split['validation'][0],split['validation'][1]-31)
            converted_y=(yv.numpy()*stats['speed_std']+stats['speed_mean']-meta['speed_mean'])/meta['speed_std']
            np.testing.assert_allclose(converted_y,outer_y[idx],rtol=1e-12,atol=1e-12)
            require(split['train'][1]-1<idx[0]+24,'OOF training/input overlap')
            predictions.append(converted);indices.append(idx)
            np.savez_compressed(dest/f"oof{split['fold']}.npz",prediction=converted,target=outer_y[idx],indices=idx,
                                weights=weights,intercept=intercept,**stats)
            blocks.append(dict(**split,n=len(idx),optimality_relative_error=errors[0],independent_prediction_max_error=float(np.max(np.abs(pred[:,0]-independent)))))
            del arrays,x,y,xv,yv
        idx=np.concatenate(indices);prediction=np.concatenate(predictions)
        require(len(np.unique(idx))==len(idx) and np.all(np.diff(idx)>0),'OOF blocks overlap or unordered')
        np.savez_compressed(dest/'oof.npz',indices=idx,prediction=prediction,target=outer_y[idx])
        write(dest/'oof.json',dict(blocks=blocks,n=len(idx),full_train_n=len(outer_y),
              score=fair.score(prediction,outer_y[idx],meta),mean_residual=float(np.mean(outer_y[idx]-prediction))))
        records.append(dict(mouse=rec['mouse'],blocks=blocks,n=len(idx),files={f.name:digest(f) for f in dest.iterdir() if f.is_file()}))
        print(json.dumps(dict(mouse=rec['mouse'],oof_examples=len(idx),ridge_fits=3)),flush=True)
    write(ROOT/'prepared.json',dict(passed=True,records=records,prefix_ridge_fits=12))


def fit(mouse):
    p=verify();fair,_,old=dependencies();dest=ROOT/mouse
    require(mouse in [r['mouse'] for r in p['cohort']],'Unknown mouse')
    prepared=next(r for r in read(ROOT/'prepared.json')['records'] if r['mouse']==mouse)
    for f,h in prepared['files'].items():require(digest(dest/f)==h,'Prepared file changed')
    with np.load(dest/'oof.npz') as saved:
        indices=saved['indices'];ridge=torch.from_numpy(saved['prediction'].astype(np.float32));y=torch.from_numpy(saved['target'].astype(np.float32))
    x=torch.from_numpy(np.load(FAIR/mouse/'train_x.npy',mmap_mode='r')[indices].copy())
    xs=torch.from_numpy(np.load(FAIR/mouse/'selection_x.npy'));ys=np.load(FAIR/mouse/'selection_y.npy')
    with np.load(BASE/f'{mouse}_f2.npz') as saved:
        rs=saved['prediction'][:,3].copy()
        np.testing.assert_array_equal(saved['target'],ys)
    stats=read(FAIR/mouse/'metadata.json')
    for family in FAMILIES:
        for pi,penalty in enumerate(PENALTIES):
            for seed in SEEDS:
                out=dest/f'{family}_p{pi}_s{seed}';out.mkdir();start=time.monotonic()
                net=model(fair,old,family,seed)
                initial=deepcopy(net.state_dict())
                delta=predict(fair,net,xs);np.testing.assert_array_equal(delta,np.zeros(len(xs)))
                best=fair.score(rs,ys,stats)['mse'];state=deepcopy(initial);epoch_best=0
                history=[dict(epoch=0,selection=fair.score(rs+delta,ys,stats),correction_rms=0.)];predictions=[delta]
                loader=DataLoader(TensorDataset(x,ridge,y,torch.arange(len(y))),batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(seed))
                opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
                scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,24,eta_min=.0001)
                gradients={};orders=[]
                for epoch in range(1,25):
                    net.train();loss_sum=0.;order=hashlib.sha256()
                    for xb,rb,yb,ix in loader:
                        order.update(ix.numpy().tobytes());opt.zero_grad(set_to_none=True)
                        correction=torch.tanh(net(xb,torch.zeros(2048,3),torch.arange(2048)))
                        loss=((rb+correction-yb)**2).mean()+penalty*(correction**2).mean()
                        require(torch.isfinite(loss),'Nonfinite loss');loss.backward()
                        nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
                        for name,value in net.named_parameters():
                            if 'in_proj_weight' in name or 'identity_embedding' in name:
                                require(value.grad is not None and torch.isfinite(value.grad).all(),'Invalid body gradient')
                                gradients[name]=max(gradients.get(name,0.),float(value.grad.norm()))
                        opt.step();loss_sum+=float(loss.detach())*len(yb)
                    scheduler.step();orders.append(order.hexdigest())
                    delta=predict(fair,net,xs);metric=fair.score(rs+delta,ys,stats)
                    history.append(dict(epoch=epoch,training_objective=loss_sum/len(y),selection=metric,correction_rms=float(np.sqrt(np.mean(delta**2)))))
                    predictions.append(delta)
                    if metric['mse']<best:
                        best=metric['mse'];state=deepcopy(net.state_dict());epoch_best=epoch
                    write(out/'history.json',history)
                changes={name:float((net.state_dict()[name]-initial[name]).norm()) for name in gradients}
                require(all(v>0 for v in gradients.values()) and all(v>0 for v in changes.values()),'Body never learned')
                torch.save(state,out/'selected.pt')
                net.load_state_dict(torch.load(out/'selected.pt',weights_only=True))
                np.testing.assert_array_equal(predict(fair,net,xs),predictions[epoch_best])
                np.savez_compressed(out/'selection_predictions.npz',correction=np.stack(predictions),ridge=rs,target=ys)
                write(out/'result.json',dict(mouse=mouse,family=family,penalty_index=pi,penalty=penalty,seed=seed,
                     selected_epoch=epoch_best,selection_mse=best,max_gradient=gradients,parameter_changes=changes,
                     batch_order_hashes=orders,checkpoint_reload_exact=True,seconds=time.monotonic()-start))
                print(json.dumps(dict(mouse=mouse,family=family,penalty=penalty,seed=seed,epoch=epoch_best,selection_mse=best,seconds=round(time.monotonic()-start))),flush=True)
    write(dest/'finished.json',dict(fits=8,complete=True))


def lock():
    p=verify();fair,_,_=dependencies();choices=[];files={}
    require(not (ROOT/'selection_lock.json').exists(),'Already locked')
    for rec in p['cohort']:
        mouse=rec['mouse'];dest=ROOT/mouse;stats=read(FAIR/mouse/'metadata.json');orders={}
        require(read(dest/'finished.json')['fits']==8,'Incomplete fits')
        for family in FAMILIES:
            scores=[]
            for pi in range(2):
                vals=[]
                for seed in SEEDS:
                    out=dest/f'{family}_p{pi}_s{seed}';r=read(out/'result.json');vals.append(r['selection_mse'])
                    with np.load(out/'selection_predictions.npz') as saved:
                        np.testing.assert_array_equal(saved['correction'][0],np.zeros(len(saved['target'])))
                        mses=[]
                        lower=-stats['speed_mean']/stats['speed_std']
                        for d in saved['correction']:
                            require(np.max(np.abs(d))<=1,'Correction bound violated')
                            mses.append(sum((max(float(a+b),lower)-float(y))**2 for a,b,y in zip(saved['ridge'],d,saved['target']))/len(d))
                    np.testing.assert_allclose(mses,[h['selection']['mse'] for h in read(out/'history.json')],rtol=1e-12,atol=1e-12)
                    require(int(np.argmin(mses))==r['selected_epoch'],'Wrong selected epoch')
                    if seed in orders:require(orders[seed]==r['batch_order_hashes'],'Unmatched order')
                    orders[seed]=r['batch_order_hashes']
                    for name in ['result.json','selected.pt','selection_predictions.npz']:
                        files[str((out/name).relative_to(ROOT))]=digest(out/name)
                scores.append(float(np.mean(vals)))
            choices.append(dict(mouse=mouse,family=family,penalty_index=int(np.argmin(scores)),selection_means=scores))
    write(ROOT/'selection_lock.json',dict(locked_utc=datetime.now(timezone.utc).isoformat(),choices=choices,files=files,
          all32_fits_complete=True,all800_epoch_errors_independently_checked=True,matched_batch_orders=True,later_scored=False))


def evaluate():
    p=verify();fair,_,old=dependencies();locked=read(ROOT/'selection_lock.json')
    require(not (ROOT/'results.json').exists(),'Already evaluated')
    for f,h in locked['files'].items():require(digest(ROOT/f)==h,'Locked artifact changed')
    rows=[]
    for rec in p['cohort']:
        mouse=rec['mouse'];dest=ROOT/mouse;stats=read(FAIR/mouse/'metadata.json')
        with np.load(FAIR/mouse/'later_raw.npz') as raw,np.load(FAIR/mouse/'statistics.npz') as st:
            z=(raw['activity']-st['activity_mean'])/st['activity_std']
            x64=np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(z,8,axis=1)[:,24:].transpose(1,0,2))
            target=(raw['speed'][31:]-stats['speed_mean'])/stats['speed_std']
            quiet=raw['speed'][31:]<=stats['movement_threshold']
        with np.load(BASE/f'{mouse}_f2.npz') as saved:
            weights=saved['weights'][:,3].copy();intercept=float(saved['intercept'][3])
        #avoid the previously observed BLAS warning and independently verify this new prediction
        ridge=np.einsum('ij,j->i',x64.reshape(len(target),-1),weights,optimize=False)+intercept
        alternate=(torch.from_numpy(x64.reshape(len(target),-1))@torch.from_numpy(weights)+intercept).numpy()
        np.testing.assert_allclose(ridge,alternate,rtol=1e-10,atol=1e-10)
        x=torch.from_numpy(x64.astype(np.float32));predictions={};records=[]
        def add(label,pred,**meta):
            pred=np.asarray(pred,dtype=np.float64);metric=fair.score(pred,target,stats)
            bounded=np.maximum(pred,-stats['speed_mean']/stats['speed_std'])
            metric['quiet_mse']=float(np.mean((bounded[quiet]-target[quiet])**2)) if quiet.any() else None
            metric['moving_mse']=float(np.mean((bounded[~quiet]-target[~quiet])**2)) if (~quiet).any() else None
            predictions[label]=pred;records.append(dict(label=label,**meta,**metric))
        add('ridge10',ridge)
        with np.load(FAIR/mouse/'later_predictions.npz') as saved:
            np.testing.assert_array_equal(saved['target'],target)
            for label in ['ridge','zero','mean','transformer_tuned_s10','transformer_tuned_s11','pooled_mlp_tuned_s10','pooled_mlp_tuned_s11']:
                add('previous_'+label,saved[label])
        for family in FAMILIES:
            choice=next(c for c in locked['choices'] if c['mouse']==mouse and c['family']==family)
            for seed in SEEDS:
                out=dest/f"{family}_p{choice['penalty_index']}_s{seed}"
                net=model(fair,old,family,seed);net.load_state_dict(torch.load(out/'selected.pt',weights_only=True))
                delta=predict(fair,net,x);result=read(out/'result.json')
                if result['selected_epoch']==0:np.testing.assert_array_equal(delta,np.zeros(len(target)))
                add(f'{family}_s{seed}',ridge+delta,epoch=result['selected_epoch'],penalty=result['penalty'],
                    correction_rms=float(np.sqrt(np.mean(delta**2))),correction_max=float(np.max(np.abs(delta))))
                predictions[f'{family}_delta_s{seed}']=delta
        np.savez_compressed(dest/'later_predictions.npz',target=target,quiet=quiet,**predictions)
        with np.load(dest/'later_predictions.npz') as saved:
            lower=-stats['speed_mean']/stats['speed_std']
            for row in records:
                independent=sum((max(float(a),lower)-float(b))**2 for a,b in zip(saved[row['label']],saved['target']))/len(target)
                require(abs(independent-row['mse'])<1e-10*max(1.,row['mse']),'Later MSE mismatch')
        rows.append(dict(mouse=mouse,n=len(target),quiet_n=int(quiet.sum()),moving_n=int((~quiet).sum()),results=records))
        print(json.dumps(rows[-1]),flush=True)
    write(ROOT/'results.json',dict(scope=p['scope'],rows=rows))
    verify()
    write(ROOT/'audit.json',dict(passed=True,neural_fits=32,prefix_ridge_fits=12,
          locked_before_later_scoring=True,zero_correction_exact=True,all800_selection_errors_checked=True,
          selected_checkpoint_reloads=True,paired_batch_orders=True,independent_later_mse=True,
          independent_ridge_predictions=True,application_unchanged=True,old_evaluation_tails_used=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['check','freeze','prepare','fit','lock','evaluate']);parser.add_argument('--mouse');args=parser.parse_args()
    with threadpool_limits(limits=2):
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        if args.action=='fit':fit(args.mouse)
        else:dict(check=check,freeze=freeze,prepare=prepare,lock=lock,evaluate=evaluate)[args.action]()
