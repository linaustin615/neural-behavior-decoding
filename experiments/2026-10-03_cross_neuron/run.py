"""Fixed observed-to-hidden neuron reconstruction experiment."""
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from threadpoolctl import threadpool_limits

from models import Reconstruction

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
BASE=ROOT.parent/'2026-10-03_dynamics_baseline'
FAIR=ROOT.parent/'2026-10-03_fair_comparison'
MICE=['MP030','MP032','MP033','MP034']
SEEDS=[10,11,12]
KINDS=['attention','static','mlp']
LAMBDAS=[.0001,.001,.01,.1,1.,10.]


def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');temp.replace(p)
def mse(p,y):
    p=np.asarray(p,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    assert p.shape==y.shape and np.isfinite(p).all()
    return float(np.mean((p-y)**2))
def split(full,source,target):return np.ascontiguousarray(full[:,source]),np.ascontiguousarray(full[:,target,-1:])
def predict(net,x):
    net.eval()
    with torch.no_grad():p=torch.cat([net(x[i:i+64]) for i in range(0,len(x),64)]).numpy()
    assert np.isfinite(p).all();return p


def check():
    rng=np.random.default_rng(81210);partition=rng.permutation(128);source=np.sort(partition[:64]);target=np.sort(partition[64:])
    assert len(set(source)&set(target))==0
    full=rng.normal(size=(3,128,32)).astype(np.float32);changed=full.copy();changed[:,target]+=100
    x,y=split(full,source,target);xx,yy=split(changed,source,target)
    np.testing.assert_array_equal(x,xx);assert not np.array_equal(y,yy)
    x=torch.from_numpy(x);mean=y.mean(0);rows=[]
    for kind in KINDS:
        net=Reconstruction(kind,10,mean);net.eval()
        np.testing.assert_array_equal(net(x).detach().numpy(),np.broadcast_to(mean,(3,64,1)))
        opt=torch.optim.AdamW(net.parameters(),lr=.001)
        for _ in range(2):
            opt.zero_grad(set_to_none=True);loss=(net(x)-torch.from_numpy(y)).square().mean();loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters());opt.step()
        if kind=='attention':watched=['routing.in_proj_weight','activity.0.weight']
        elif kind=='static':watched=['routing.query','routing.value.weight','activity.0.weight']
        else:watched=['body.0.weight']
        assert all(dict(net.named_parameters())[k].grad.norm()>0 for k in watched)
        clone=Reconstruction(kind,10,mean);clone.load_state_dict(net.state_dict());np.testing.assert_array_equal(predict(net,x),predict(clone,x))
        rows.append(dict(kind=kind,parameters=sum(p.numel() for p in net.parameters())))
    a,b=Reconstruction('attention',10,mean),Reconstruction('static',10,mean)
    for key,value in a.state_dict().items():
        if not key.startswith('routing.'):torch.testing.assert_close(value,b.state_dict()[key],rtol=0,atol=0)
    assert abs(rows[0]['parameters']/rows[1]['parameters']-1)<.01
    write(ROOT/'selfcheck.json',dict(passed=True,source_target_disjoint=True,target_perturbation_leaves_input_exactly_unchanged=True,initial_output_exact_mean=True,nonzero_body_gradients_after_head_opens=True,common_initialization_exact=True,reload_exact=True,models=rows))


def freeze():
    assert not (ROOT/'protocol.json').exists() and read(ROOT/'selfcheck.json')['passed']
    partition=np.random.default_rng(81210).permutation(128);source=np.sort(partition[:64]).tolist();target=np.sort(partition[64:]).tolist()
    files=[ROOT/'run.py',ROOT/'models.py',BASE/'protocol.json',BASE/'prepared.json']
    for m in MICE:
        files += [BASE/m/f'{s}_x.npy' for s in ['train','selection']]+[BASE/m/'columns.npy']
        files += [FAIR/m/n for n in ['later_raw.npz','statistics.npz','metadata.json']]
    files += [PROJECT/n for n in ['model.py','data.py','train.py']]
    write(ROOT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        question='Can input-dependent target-specific attention improve reconstruction of hidden-neuron activity from other observed neurons?',
        endpoint='Predict the final-bin standardized activity of64 target neurons from the preceding32-bin histories (including that same final bin) of64 disjoint source neurons. Target-neuron activity is never an input. Training labels are known target neurons;later target values are hidden. This is contemporaneous reconstruction,not future forecasting,causal connectivity,unseen-neuron transfer or behavior decoding.',
        source_columns=source,target_columns=target,partition_seed=81210,
        data='Same archived128-cell subset,chronological train/selection/later intervals and inherited training-only normalization. Fixed label-independent source/target partition,applied to all four mice without cell correspondence across sessions. Source inputs use no time later than the target bin. No running-speed labels or coordinates.',
        models='Attention:shared per-source32/32/16 activity MLP plus sourceID,LayerNorm,targetID queries,two-head cross-attention,residual16/32/16 feature MLP,per-target scalar output. Static:matching model with learned input-independent two-head rank2 source-target routing plus value/output projections. All nonrouting initialization matches;counts within1%. Global MLP:flatten2048 inputs,64hidden,GELU,scalar-per-target output,larger parameter budget. Every output starts at the training target mean.',
        baselines='Training target mean and standardized-input multioutput ridge. Six fixed penalties .0001/.001/.01/.1/1/10,earlier selection. Strong simple baseline chooses mean or ridge on earlier MSE,not later outcomes.',
        feasibility='Before neural fitting,selected simple baseline must beat training-mean selection MSE by>=1% equal-weight average and on>=3/4 mice. Otherwise stop after feasibility report;no conclusion that nonlinear prediction is impossible.',
        training='If feasibility passes:36fits,attention/static/MLP x4mice x3seeds10/11/12,24epochs each. AdamW.001,wd.01,batch32,clip1,cosine24 eta_min.0001. Uniform target-neuron MSE after inherited scaling. Select epoch0..24 by earlier MSE. Lock all36 choices before any new later scoring.',
        gate='Attention>=5% equal-weight mean within-mouse relative MSE gain versus static,MLP,and selected simple baseline;>=3/4 mouse wins each;>=8/12 paired-seed wins versus neural controls;no mouse>25% worse than selected simple baseline. Also beat own mean-output initialization for>=8/12 models. Diagnostic gate,not significance.',
        secondary='Per-neuron MSE and mean-reference skill,initial-checkpoint scores,and one fixed circular shift of source contexts by half the later segment as an alignment-destruction control. Shift does not retrain models and may create distribution shift;not proof of causality.',
        uncertainty='2000 paired hierarchical mouse/seed/circular100-context bootstrap draws,seed81211,descriptive98.3333% intervals for three primary contrasts. Aggregate target-neuron losses within each context;neurons do not count as independent animals.',
        scope='Adaptive new-endpoint exploration on four historically inspected Stringer recordings. Separate this claim from the unsuccessful running-speed studies. No novel architecture or significance claim,no new dataset,no publication or application changes. No grid expansion.',
        hashes={str(p.relative_to(PROJECT)):digest(p) for p in files}))


def verify():
    p=read(ROOT/'protocol.json')
    for name,h in p['hashes'].items():assert digest(PROJECT/name)==h,name
    return p


def prepare():
    p=verify();assert not (ROOT/'prepared.json').exists();rows=[]
    for m in MICE:
        dest=ROOT/m;dest.mkdir()
        x,y=split(np.load(BASE/m/'train_x.npy'),p['source_columns'],p['target_columns'])
        xv,yv=split(np.load(BASE/m/'selection_x.npy'),p['source_columns'],p['target_columns'])
        mean=y.astype(np.float64).mean(0);flat=x.reshape(len(x),-1).astype(np.float64);fv=xv.reshape(len(xv),-1).astype(np.float64)
        center=flat.mean(0);scale=np.maximum(flat.std(0),1e-8);z=torch.from_numpy((flat-center)/scale);target=torch.from_numpy(y.reshape(len(y),-1).astype(np.float64)-mean.ravel())
        dual=z.shape[1]>len(z);gram=z@z.T if dual else z.T@z;rhs=target if dual else z.T@target
        candidates=[dict(kind='mean',mse=mse(np.broadcast_to(mean,yv.shape),yv))]
        for lam in LAMBDAS:
            system=gram+len(x)*lam*torch.eye(len(gram),dtype=torch.float64);solution=torch.linalg.solve(system,rhs)
            weight=z.T@solution if dual else solution
            residual=float((z.T@(z@weight-target)+len(x)*lam*weight).norm()/(z.T@target).norm().clamp_min(1e-12));assert residual<1e-7
            pred=(torch.from_numpy((fv-center)/scale)@weight).numpy()+mean.ravel()
            alternate=np.einsum('ij,jk->ik',(fv-center)/scale,weight.numpy(),optimize=False)+mean.ravel();np.testing.assert_allclose(pred,alternate,rtol=1e-8,atol=1e-8)
            pred=pred.reshape(yv.shape);candidates.append(dict(kind='ridge',lam=lam,mse=mse(pred,yv),normal_equation_residual=residual))
            np.savez_compressed(dest/f'ridge_{lam}.npz',center=center,scale=scale,weight=weight.numpy(),mean=mean,prediction=pred,target=yv)
        for name,a in [('train_x',x),('train_y',y),('selection_x',xv),('selection_y',yv),('target_mean',mean)]:np.save(dest/f'{name}.npy',a)
        row=dict(mouse=m,train_n=len(x),selection_n=len(xv),candidates=candidates,choice=min(candidates,key=lambda c:c['mse']))
        write(dest/'baselines.json',row);rows.append(row);print(m,'baselines prepared',flush=True)
    gains=[1-r['choice']['mse']/r['candidates'][0]['mse'] for r in rows]
    passed=bool(np.mean(gains)>=.01 and sum(g>0 for g in gains)>=3)
    write(ROOT/'prepared.json',dict(rows=rows,feasibility=dict(passed=passed,mean_relative_gain=float(np.mean(gains)),mouse_wins=sum(g>0 for g in gains),gains=gains),hashes={str(f.relative_to(ROOT)):digest(f) for m in MICE for f in (ROOT/m).iterdir() if f.is_file()}))


def train(mouse):
    verify();assert read(ROOT/'prepared.json')['feasibility']['passed'];dest=ROOT/mouse
    x,xv=[torch.from_numpy(np.load(dest/f'{s}_x.npy')) for s in ['train','selection']]
    y=torch.from_numpy(np.load(dest/'train_y.npy'));yv=np.load(dest/'selection_y.npy');mean=np.load(dest/'target_mean.npy')
    for seed in SEEDS:
        for kind in KINDS:
            out=dest/f'{kind}_s{seed}';out.mkdir();net=Reconstruction(kind,seed,mean)
            first=predict(net,xv);best=mse(first,yv);selected=0;history=[dict(epoch=0,selection_mse=best)];start=time.monotonic()
            torch.save(net.state_dict(),out/'initial.pt');torch.save(net.state_dict(),out/'selected.pt')
            torch.manual_seed(seed+9000);generator=torch.Generator().manual_seed(seed)
            loader=DataLoader(TensorDataset(x,y,torch.arange(len(y))),batch_size=32,shuffle=True,generator=generator)
            opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,24,eta_min=.0001)
            key={'attention':'routing.in_proj_weight','static':'routing.query','mlp':'body.0.weight'}[kind];grad_max=0.
            for epoch in range(1,25):
                net.train();order=hashlib.sha256();total=0.
                for xb,yb,ib in loader:
                    order.update(ib.numpy().tobytes());opt.zero_grad(set_to_none=True);loss=(net(xb)-yb).square().mean();assert torch.isfinite(loss);loss.backward()
                    grad_max=max(grad_max,float(dict(net.named_parameters())[key].grad.norm()));nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step();total+=float(loss.detach())*len(xb)
                schedule.step();pred=predict(net,xv);score=mse(pred,yv)
                if score<best:best=score;selected=epoch;torch.save(net.state_dict(),out/'selected.pt')
                history.append(dict(epoch=epoch,training_mse=total/len(y),selection_mse=score,batch_order_hash=order.hexdigest()));write(out/'history.json',history)
            assert grad_max>0;net.load_state_dict(torch.load(out/'selected.pt',weights_only=True));pred=predict(net,xv);assert abs(mse(pred,yv)-best)<1e-10
            np.savez_compressed(out/'selection_predictions.npz',selected=pred,initial=first,target=yv)
            write(out/'result.json',dict(mouse=mouse,kind=kind,seed=seed,epochs=24,selected_epoch=selected,selection_mse=best,initial_mse=history[0]['selection_mse'],grad_max=grad_max,reload_score_matches=True,elapsed_seconds=time.monotonic()-start));print(mouse,kind,seed,'complete',flush=True)
    write(dest/'training_finished.json',dict(complete=True))


def lock():
    verify();assert not (ROOT/'selection_lock.json').exists();records=[];hashes={};orders={}
    for f in sorted(ROOT.glob('*/*/result.json')):
        r=read(f);h=read(f.parent/'history.json');assert len(h)==25 and int(np.argmin([v['selection_mse'] for v in h]))==r['selected_epoch']
        order=[v['batch_order_hash'] for v in h[1:]];key=(r['mouse'],r['seed'])
        if key in orders:assert orders[key]==order
        orders[key]=order
        with np.load(f.parent/'selection_predictions.npz') as saved:assert abs(mse(saved['selected'],saved['target'])-r['selection_mse'])<1e-10
        records.append(r);ck=f.parent/'selected.pt';hashes[str(ck.relative_to(ROOT))]=digest(ck)
    assert len(records)==36 and len(list(ROOT.glob('*/training_finished.json')))==4
    write(ROOT/'selection_lock.json',dict(locked_utc=datetime.now(timezone.utc).isoformat(),records=records,hashes=hashes,matched_batch_orders=True,later_scored=False))


def evaluate():
    p=verify();assert not (ROOT/'results.json').exists();locked=read(ROOT/'selection_lock.json')
    for table in [locked['hashes'],read(ROOT/'prepared.json')['hashes']]:
        for name,h in table.items():assert digest(ROOT/name)==h
    rows=[]
    for m in MICE:
        dest=ROOT/m;indices=np.load(BASE/m/'columns.npy')
        with np.load(FAIR/m/'later_raw.npz') as raw,np.load(FAIR/m/'statistics.npz') as norm:
            seq=((raw['activity'][indices,24:]-norm['activity_mean'][indices])/norm['activity_std'][indices]).T.astype(np.float32)
        full=np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(seq,32,axis=0));x,y=split(full,p['source_columns'],p['target_columns']);xt=torch.from_numpy(x)
        mean=np.load(dest/'target_mean.npy');predictions={'mean':np.broadcast_to(mean,y.shape).copy()};choice=read(dest/'baselines.json')['choice']
        if choice['kind']=='mean':predictions['baseline']=predictions['mean'].copy()
        else:
            with np.load(dest/f"ridge_{choice['lam']}.npz") as fit:
                predictions['baseline']=(np.einsum('ij,jk->ik',(x.reshape(len(x),-1)-fit['center'])/fit['scale'],fit['weight'],optimize=False)+fit['mean'].ravel()).reshape(y.shape)
        records={};per_neuron={}
        shifted=torch.roll(xt,len(x)//2,0)
        for kind in KINDS:
            for seed in SEEDS:
                net=Reconstruction(kind,seed,mean);net.load_state_dict(torch.load(dest/f'{kind}_s{seed}'/'selected.pt',weights_only=True))
                predictions[f'{kind}_s{seed}']=predict(net,xt);predictions[f'{kind}_shifted_s{seed}']=predict(net,shifted)
                net.load_state_dict(torch.load(dest/f'{kind}_s{seed}'/'initial.pt',weights_only=True));predictions[f'{kind}_initial_s{seed}']=predict(net,xt)
        for label,pred in predictions.items():
            records[label]=mse(pred,y);err=np.asarray(pred,dtype=np.float64)-y.astype(np.float64)
            independent=float(np.einsum('ijk,ijk->',err,err,optimize=False)/err.size);assert abs(independent-records[label])<1e-10*max(1.,independent)
            per_neuron[label]=(err**2).mean((0,2)).tolist()
        np.savez_compressed(dest/'later_predictions.npz',target=y,**predictions)
        rows.append(dict(mouse=m,n=len(y),scores=records,per_neuron_mse=per_neuron,baseline_choice=choice));print(m,'scored',flush=True)
    verify();write(ROOT/'results.json',dict(rows=rows));write(ROOT/'audit.json',dict(passed=True,neural_fits=36,epochs_each=24,ridge_settings=24,source_target_disjoint=True,hidden_activity_never_input=True,initial_outputs_training_mean=True,nonzero_routing_or_body_gradients=True,matched_batches=True,selected_reload_scores_match=True,all_choices_locked_before_later_scoring=True,independent_later_metrics=True,frozen_hashes_unchanged=True))


if __name__=='__main__':
    with threadpool_limits(limits=2):
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        if sys.argv[1]=='train':train(sys.argv[2])
        else:{'check':check,'freeze':freeze,'prepare':prepare,'lock':lock,'evaluate':evaluate}[sys.argv[1]]()
