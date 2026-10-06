"""Bounded joint MLP/attention hybrid comparison on reused development recordings."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from threadpoolctl import threadpool_limits

from hybrid import Hybrid

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
BROAD=ROOT.parent/'2026-10-03_broad_screen'
FAIR=ROOT.parent/'2026-10-03_fair_comparison'
MICE=['MP030','MP032','MP033','MP034']
SEEDS=[10,11,12]
KINDS=['attention','extra_mlp']


def read(p):return json.loads(p.read_text())


def write(p,value):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(p)


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def require(condition,message):
    if not condition:raise RuntimeError(message)


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def deps():
    fair=module('hybrid_fair',FAIR/'run.py');_,old=fair.dependencies()
    broad=module('hybrid_broad_models',BROAD/'models.py')
    return fair,old,broad


def make(mouse,seed,kind):
    fair,old,broad=deps()
    base=broad.make(old,'norm_mlp',seed,metadata=read(FAIR/mouse/'metadata.json'))
    base.load_state_dict(torch.load(BROAD/mouse/f'norm_mlp_s{seed}'/'initial.pt',weights_only=True))
    return Hybrid(base,kind,seed)


def predict(net,x):
    net.eval()
    with torch.no_grad():
        chunks=[net(x[i:i+64]) for i in range(0,len(x),64)]
        values=[torch.cat([chunk[j] for chunk in chunks]).numpy() for j in range(3)]
    require(all(np.isfinite(v).all() for v in values),'Nonfinite prediction')
    return values


def check():
    torch.manual_seed(59012);x=torch.randn(3,2048,8);y=torch.tensor([.2,.5,-.1]);rows=[]
    for kind in KINDS:
        net=make(MICE[0],10,kind);net.eval()
        with torch.no_grad():expected=net.base(x)[0].numpy()
        out,main,delta=predict(net,x)
        np.testing.assert_array_equal(out,expected);np.testing.assert_array_equal(main,expected)
        require(np.all(delta==0),'Initial correction not zero')
        if kind=='extra_mlp':require(not any(isinstance(m,nn.MultiheadAttention) for m in net.modules()),'Attention in MLP control')
        initial=deepcopy(net.state_dict());opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
        monitored='readin.in_proj_weight' if kind=='attention' else 'body.1.weight'
        grad=0.
        for _ in range(3):
            net.train();opt.zero_grad(set_to_none=True);loss=(net(x)[0]-y).square().mean();loss.backward()
            require(all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()),'Invalid gradient')
            grad=max(grad,float(dict(net.named_parameters())[monitored].grad.norm()))
            nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
        require(grad>0 and not torch.equal(initial[monitored],net.state_dict()[monitored]),'Branch body did not learn')
        require(not torch.equal(initial['base.amplitude.weight'],net.state_dict()['base.amplitude.weight']),'Base did not learn')
        net.eval();permutation=torch.randperm(2048)
        with torch.no_grad():
            torch.testing.assert_close(net(x)[0],net(x[:,permutation],permutation)[0],rtol=1e-5,atol=1e-5)
        clone=make(MICE[0],10,kind);clone.load_state_dict(net.state_dict())
        np.testing.assert_array_equal(predict(net,x)[0],predict(clone,x)[0])
        branch=sum(p.numel() for name,p in net.named_parameters() if not name.startswith('base.'))
        rows.append(dict(kind=kind,total_parameters=sum(p.numel() for p in net.parameters()),branch_parameters=branch,branch_gradient=grad))
    require(abs(rows[0]['branch_parameters']/rows[1]['branch_parameters']-1)<.01,'Branch parameter mismatch')
    write(ROOT/'selfcheck.json',dict(passed=True,models=rows,zero_branch_exact_base=True,branch_and_base_gradients_updates=True,cell_id_permutation=True,reload_exact=True))


def freeze():
    require(not (ROOT/'protocol.json').exists(),'Protocol already frozen');check()
    refs=[BROAD/'models.py',BROAD/'run.py',BROAD/'protocol.json',BROAD/'results.json',FAIR/'run.py']
    for mouse in MICE:
        refs.extend(FAIR/mouse/f for f in ['train_x.npy','train_y.npy','selection_x.npy','selection_y.npy','later_raw.npz','statistics.npz','metadata.json'])
        refs.extend(BROAD/mouse/f for f in ['later_predictions.npz','baselines.json'])
        for seed in SEEDS:
            refs.extend(BROAD/mouse/f'norm_mlp_s{seed}'/f for f in ['initial.pt',f'{mouse}_selected.pt',f'{mouse}_selection.npz','result.json','history.json'])
    write(ROOT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),mice=MICE,seeds=SEEDS,arms=KINDS,
        scope='Exploratory fixed comparison on four historically inspected Stringer recordings,not independent confirmation',
        budget='24 new fits,two hybrids x4mice x3seeds,24epochs each; reuse12 matching normalized-MLP-alone fits,do not refit baselines',
        architecture='Shared normalized32dim activity+ID tokens. Main path is exact prior normalized pooled MLP including population-amplitude bypass. Attention correction has16 learned summaries,cross-attention,one4head transformer with64dim feedforward,and linear readout. Extra-MLP correction has residual32/64/32 token MLP,mean pooling,32/266/1 head. Both final correction heads start exactly zero; all parameters train jointly from the same archived untrained base initialization.',
        difference_from_prior='Full training set,true speed targets,joint end-to-end training,no prefix OOF residual labels,no frozen trained base,no tanh correction bound or extra penalty. This tests one joint hybrid recipe,not post-training augmentation.',
        training='AdamW lr0.001 weight_decay0.01,batch32,gradient clip1,cosine24 eta_min0.0001. Same archived2048cells/eight-bin histories/train scaling/splits. Identical batch order per seed across arms and archived base; dropout masks need not match across different architectures.',
        selection='Bounded earlier-selection MSE chooses epoch0..24 per model/mouse/seed. Lock all24 selections before any new later prediction. No learning-rate/width/penalty search or later-based re-selection.',
        gate='Attention hybrid:>=5% equal-weight mean relative gain vs MLP alone AND extra-MLP hybrid;>=3/4 mouse wins vs each;>=8/12 paired seed wins vs each;no mouse>25% worse than base;positive mean gain vs previously fixed selected regression. Exploratory practical gate,not significance.',
        secondary='Own epoch0 comparison; remove trained correction branch without retraining as a coadaptation-sensitive diagnostic; fixed raw-prediction three-seed ensembles; training75th-percentile quiet/moving subsets; leave-one-mouse-out gains.',
        uncertainty='4000 paired hierarchical mouse/seed/circular100-bin block resamples for attention-vs-base and attention-vs-extraMLP;97.5% intervals per contrast,descriptive conditional only',
        stopping='Complete24 fits and declared reports only. No extra settings or application edits/publication.',
        application_hashes=read(BROAD/'protocol.json')['application_hashes'],references={str(p.relative_to(PROJECT)):digest(p) for p in refs},
        sources={name:digest(ROOT/name) for name in ['run.py','hybrid.py']}))


def verify():
    p=read(ROOT/'protocol.json')
    for name,h in p['sources'].items():require(digest(ROOT/name)==h,'Frozen source changed '+name)
    for field in ['references','application_hashes']:
        for name,h in p[field].items():require(digest(PROJECT/name)==h,'Reference changed '+name)
    return p


def fit(mouse):
    verify();fair,_,_=deps();dest=ROOT/mouse;dest.mkdir()
    x=torch.from_numpy(np.load(FAIR/mouse/'train_x.npy'));y=torch.from_numpy(np.load(FAIR/mouse/'train_y.npy').astype(np.float32))
    xs=torch.from_numpy(np.load(FAIR/mouse/'selection_x.npy'));ys=np.load(FAIR/mouse/'selection_y.npy');stats=read(FAIR/mouse/'metadata.json')
    for kind in KINDS:
        for seed in SEEDS:
            out=dest/f'{kind}_s{seed}';out.mkdir();start=time.monotonic();net=make(mouse,seed,kind)
            initial=predict(net,xs)[0]
            with np.load(BROAD/mouse/f'norm_mlp_s{seed}'/f'{mouse}_selection.npz') as saved:np.testing.assert_array_equal(initial,saved['prediction'][0])
            score=fair.score(initial,ys,stats);best=score['mse'];chosen=0;predictions=[initial];history=[dict(epoch=0,selection=score)]
            torch.save(net.state_dict(),out/'initial.pt');torch.save(net.state_dict(),out/'selected.pt')
            torch.manual_seed(seed+99000);gen=torch.Generator().manual_seed(seed)
            loader=DataLoader(TensorDataset(x,y,torch.arange(len(y))),batch_size=32,shuffle=True,generator=gen)
            opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,24,eta_min=.0001)
            watched=['base.amplitude.weight','readin.in_proj_weight' if kind=='attention' else 'body.1.weight'];gradients={k:0. for k in watched}
            for epoch in range(1,25):
                net.train();total=0.;order=hashlib.sha256()
                for xb,yb,idx in loader:
                    order.update(idx.numpy().tobytes());opt.zero_grad(set_to_none=True);loss=(net(xb)[0]-yb).square().mean();require(torch.isfinite(loss),'Nonfinite loss');loss.backward()
                    for k in watched:
                        g=dict(net.named_parameters())[k].grad;require(g is not None and torch.isfinite(g).all(),'Bad monitored gradient');gradients[k]=max(gradients[k],float(g.norm()))
                    nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step();total+=float(loss.detach())*len(yb)
                sched.step();pred=predict(net,xs)[0];predictions.append(pred);score=fair.score(pred,ys,stats)
                if score['mse']<best:best=score['mse'];chosen=epoch;torch.save(net.state_dict(),out/'selected.pt')
                history.append(dict(epoch=epoch,training_mse=total/len(y),selection=score,batch_order_hash=order.hexdigest()));write(out/'history.json',history)
            require(all(v>0 for v in gradients.values()),'No monitored learning')
            net.load_state_dict(torch.load(out/'selected.pt',weights_only=True));np.testing.assert_array_equal(predict(net,xs)[0],predictions[chosen])
            np.savez_compressed(out/'selection_predictions.npz',prediction=np.stack(predictions),target=ys)
            result=dict(mouse=mouse,kind=kind,seed=seed,epochs=24,selected_epoch=chosen,selection_mse=best,gradients=gradients,selected_reload_exact=True,elapsed_seconds=time.monotonic()-start)
            write(out/'result.json',result);print(json.dumps(result),flush=True)
    write(dest/'finished.json',dict(complete=True,fits=6))


def lock():
    verify();require(not (ROOT/'selection_lock.json').exists(),'Already locked');records=[];files={};checked=0
    for mouse in MICE:
        stats=read(FAIR/mouse/'metadata.json');lower=-stats['speed_mean']/stats['speed_std']
        for seed in SEEDS:
            basehistory=read(BROAD/mouse/f'norm_mlp_s{seed}'/'history.json');orders=[r['batch_order_hash'] for r in basehistory[1:]]
            for kind in KINDS:
                out=ROOT/mouse/f'{kind}_s{seed}';r=read(out/'result.json');h=read(out/'history.json')
                require(r['epochs']==24 and len(h)==25,'Incomplete fit');require([a['batch_order_hash'] for a in h[1:]]==orders,'Batch orders differ')
                with np.load(out/'selection_predictions.npz') as saved:
                    errors=[sum((max(float(a),lower)-float(b))**2 for a,b in zip(v,saved['target']))/len(v) for v in saved['prediction']]
                np.testing.assert_allclose(errors,[v['selection']['mse'] for v in h],rtol=1e-10,atol=1e-10)
                require(int(np.argmin(errors))==r['selected_epoch'],'Checkpoint selection mismatch');checked+=len(errors)
                path=out/'selected.pt';files[str(path.relative_to(PROJECT))]=digest(path);records.append(r)
    write(ROOT/'selection_lock.json',dict(locked_utc=datetime.now(timezone.utc).isoformat(),records=records,files=files,selection_errors_checked=checked,matched_batch_orders=True,later_scored=False))


def evaluate():
    p=verify();fair,_,_=deps();locked=read(ROOT/'selection_lock.json');require(not (ROOT/'results.json').exists(),'Already evaluated')
    for name,h in locked['files'].items():require(digest(PROJECT/name)==h,'Selected checkpoint changed')
    rows=[]
    for mouse in MICE:
        stats=read(FAIR/mouse/'metadata.json');lower=-stats['speed_mean']/stats['speed_std']
        with np.load(FAIR/mouse/'later_raw.npz') as raw,np.load(FAIR/mouse/'statistics.npz') as st:
            z=(raw['activity']-st['activity_mean'])/st['activity_std'];x=np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(z,8,axis=1)[:,24:].transpose(1,0,2)).astype(np.float32)
            y=(raw['speed'][31:]-stats['speed_mean'])/stats['speed_std'];quiet=raw['speed'][31:]<=stats['movement_threshold']
        x=torch.from_numpy(x);predictions={};records=[]
        def add(label,pred,**extra):
            pred=np.asarray(pred,dtype=np.float64);score=fair.score(pred,y,stats);error=(np.maximum(pred,lower)-y)**2
            independent=sum((max(float(a),lower)-float(b))**2 for a,b in zip(pred,y))/len(y);require(abs(independent-score['mse'])<1e-10*max(1.,score['mse']),'MSE mismatch')
            score.update(quiet_mse=float(error[quiet].mean()) if quiet.any() else None,moving_mse=float(error[~quiet].mean()) if (~quiet).any() else None)
            records.append(dict(label=label,**extra,**score));predictions[label]=pred
        with np.load(BROAD/mouse/'later_predictions.npz') as saved:
            np.testing.assert_array_equal(y,saved['target'])
            for label in ['ridge','kernel','strong_baseline','zero','mean']:add(label,saved[label])
            for seed in SEEDS:
                epoch=read(BROAD/mouse/f'norm_mlp_s{seed}'/'result.json')['chosen'][mouse]
                add(f'base_s{seed}',saved[f'norm_mlp_s{seed}'],epoch=epoch);add(f'initial_s{seed}',saved[f'norm_mlp_initial_s{seed}'])
        for kind in KINDS:
            for seed in SEEDS:
                out=ROOT/mouse/f'{kind}_s{seed}';net=make(mouse,seed,kind);net.load_state_dict(torch.load(out/'selected.pt',weights_only=True));pred,main,delta=predict(net,x)
                np.testing.assert_allclose(pred,main+delta,rtol=1e-6,atol=1e-6)
                add(f'{kind}_s{seed}',pred,epoch=read(out/'result.json')['selected_epoch']);add(f'{kind}_branch_off_s{seed}',main)
                predictions[f'{kind}_correction_s{seed}']=delta
        for kind in ['base']+KINDS:add(f'{kind}_ensemble',np.mean([predictions[f'{kind}_s{s}'] for s in SEEDS],axis=0))
        np.savez_compressed(ROOT/mouse/'later_predictions.npz',target=y,quiet=quiet,**predictions)
        rows.append(dict(mouse=mouse,n=len(y),quiet_n=int(quiet.sum()),moving_n=int((~quiet).sum()),records=records))
        print(json.dumps(dict(mouse=mouse,mses={r['label']:r['mse'] for r in records})),flush=True)
    write(ROOT/'results.json',dict(scope=p['scope'],rows=rows));verify()
    write(ROOT/'audit.json',dict(passed=True,new_fits=24,reused_base_fits=12,zero_initial_correction_exact=True,selected_reload_exact=True,all600_selection_errors_checked=True,matched_batch_orders=True,locked_before_new_later_scoring=True,independent_later_errors=True,reused_target_alignment_exact=True,frozen_sources_references_application_unchanged=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['check','freeze','fit','lock','evaluate']);parser.add_argument('--mouse');args=parser.parse_args()
    with threadpool_limits(limits=2):
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        if args.action=='fit':fit(args.mouse)
        else:dict(check=check,freeze=freeze,lock=lock,evaluate=evaluate)[args.action]()
