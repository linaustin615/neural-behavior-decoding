"""Supervised-to-supervised restart control for the existing pretraining recipe."""
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

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
BASE=ROOT.parent/'2026-10-03_dynamics_baseline'
FLAT=ROOT.parent/'2026-10-03_neuron_readout'
FAIR=ROOT.parent/'2026-10-03_fair_comparison'
sys.path.insert(0,str(BASE))
from models import Dynamics

MICE=['MP030','MP032','MP033','MP034']
SEEDS=[10,11,12]


def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');temp.replace(p)
def start_path(m,k,s):return BASE/m/f'{k}_scratch_s{s}'/'selected.pt'
def mse(p,y,lower):
    p=np.asarray(p,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    assert p.shape==y.shape and np.isfinite(p).all()
    return float(np.mean((np.maximum(p,lower)-y)**2))
def predict(net,x):
    net.eval()
    with torch.no_grad():p=torch.cat([net(x[i:i+64]) for i in range(0,len(x),64)]).numpy()
    assert np.isfinite(p).all();return p
def make(m,k,s):
    net=Dynamics(k,s);net.load_state_dict(torch.load(start_path(m,k,s),weights_only=True));return net


def freeze():
    assert not (ROOT/'protocol.json').exists()
    hashes=dict(read(FLAT/'protocol.json')['hashes'])
    files=[Path(__file__),BASE/'selection_lock.json',FLAT/'results.json']
    budgets=[]
    for m in MICE:
        n=len(np.load(BASE/m/'train_y.npy'));steps=(n+31)//32;forecast_steps=(n-4+31)//32
        budgets.append(dict(mouse=m,train_windows=n,supervised_steps_per_epoch=steps,forecast_steps_per_epoch=forecast_steps,
            supervised_recipe_searched_updates=48*steps,forecast_recipe_searched_updates=24*(steps+forecast_steps),relative_extra_updates=48*steps/(24*(steps+forecast_steps))-1))
        files.append(FLAT/m/'later_predictions.npz')
        for k in ['attention','mixer']:
            for s in SEEDS:
                files += [start_path(m,k,s),BASE/m/f'{k}_scratch_s{s}'/'result.json',BASE/m/f'{k}_scratch_s{s}'/'history.json',BASE/m/f'{k}_scratch_s{s}'/'selection_predictions.npz']
    hashes.update({str(p.relative_to(PROJECT)):digest(p) for p in files})
    write(ROOT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        question='Does the earlier neural-pretraining advantage survive comparison with another supervised training stage?',
        design='Reuse selected scratch-speed checkpoints for attention/mixer, four mice, seeds10/11/12. Reset AdamW and run24 more speed epochs. Keep exactly the original pooled architecture, inputs, head, lr.001,wd.01,batch32,clip1,cosine24 eta_min.0001. Start dropout and batch generators with the original stage seed. No architecture edits or repeated first-stage fits.',
        selection='Second-stage epoch0..24 selected on earlier bounded speed MSE; epoch0 is the already selected first-stage model. Lock all24 choices before current later scoring. Compare to archived forecast-pretrained and scratch models on identical target endpoints.',
        budget='24 new second-stage fits,24epochs each;24 archived first-stage fits reused. Both full recipes search48epochs with a selected-checkpoint restart and reset optimizer. This is NOT uninterrupted48epoch training or exactly matched update counts. Selection also changes effective trained ancestry, which is recorded per selected model.',
        budget_by_mouse=budgets,
        primary='Forecast-pretrained attention versus supervised-restart attention, and forecast-pretrained mixer versus supervised-restart mixer. Within each architecture, an objective-recipe benefit gate requires>=5% equal-weight mean within-mouse relative gain,>=3/4 mice and>=8/12 paired seeds.',
        secondary='Restart versus one-stage scratch within architecture. Restart attention versus restart mixer, stronger standardized raw ridge and prior matched pooled MLP. Restart-attention utility gate uses>=5% mean gain and>=3/4 mice against all three;>=8/12 paired seeds against neural controls;no mouse>25% worse than MLP.',
        uncertainty='2000 paired hierarchical mouse/seed/circular100-bin bootstrap draws,seed81209,descriptive97.5% intervals for two primary within-architecture contrasts.',
        limits='Adaptive follow-up on reused development recordings. More supervised label exposure and a task-trained speed head differ from forecasting pretraining; compare whole recipes,not an isolated objective. Identical epochs and near-matched searched update counts do not equal compute. No independent significance or novelty claim. No grid expansion.',hashes=hashes))


def verify():
    for name,h in read(ROOT/'protocol.json')['hashes'].items():assert digest(PROJECT/name)==h,name


def train(mouse):
    verify();dest=ROOT/mouse;dest.mkdir()
    x,xv=[torch.from_numpy(np.load(BASE/mouse/f'{s}_x.npy')) for s in ['train','selection']]
    y=torch.from_numpy(np.load(BASE/mouse/'train_y.npy').astype(np.float32));yv=np.load(BASE/mouse/'selection_y.npy')
    meta=read(FAIR/mouse/'metadata.json');lower=-meta['speed_mean']/meta['speed_std']
    for seed in SEEDS:
        for kind in ['attention','mixer']:
            out=dest/f'{kind}_s{seed}';out.mkdir();net=make(mouse,kind,seed)
            first=predict(net,xv);prior=read(BASE/mouse/f'{kind}_scratch_s{seed}'/'result.json')
            with np.load(BASE/mouse/f'{kind}_scratch_s{seed}'/'selection_predictions.npz') as saved:
                np.testing.assert_array_equal(first,saved['prediction'][prior['selected_epoch']]);np.testing.assert_array_equal(yv,saved['target'])
            best=mse(first,yv,lower);selected=0;history=[dict(epoch=0,selection_mse=best)];predictions=[first]
            torch.save(net.state_dict(),out/'selected.pt')
            torch.manual_seed(seed+9000);generator=torch.Generator().manual_seed(seed)
            loader=DataLoader(TensorDataset(x,y,torch.arange(len(y))),batch_size=32,shuffle=True,generator=generator)
            opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,24,eta_min=.0001)
            reference=read(BASE/mouse/f'{kind}_scratch_s{seed}'/'history.json');start=time.monotonic();grad_max=0.
            key='temporal.mix.in_proj_weight' if kind=='attention' else 'temporal.weight'
            for epoch in range(1,25):
                net.train();total=0.;order=hashlib.sha256()
                for xb,yb,ib in loader:
                    order.update(ib.numpy().tobytes());opt.zero_grad(set_to_none=True);loss=(net(xb)-yb).square().mean()
                    assert torch.isfinite(loss);loss.backward();grad_max=max(grad_max,float(dict(net.named_parameters())[key].grad.norm()))
                    nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step();total+=float(loss.detach())*len(xb)
                scheduler.step();p=predict(net,xv);score=mse(p,yv,lower);assert order.hexdigest()==reference[epoch]['batch_order_hash']
                if score<best:best=score;selected=epoch;torch.save(net.state_dict(),out/'selected.pt')
                predictions.append(p);history.append(dict(epoch=epoch,training_mse=total/len(y),selection_mse=score,batch_order_hash=order.hexdigest()));write(out/'history.json',history)
            assert grad_max>0
            net.load_state_dict(torch.load(out/'selected.pt',weights_only=True));np.testing.assert_array_equal(predict(net,xv),predictions[selected])
            np.savez_compressed(out/'selection_predictions.npz',prediction=np.stack(predictions),target=yv)
            write(out/'result.json',dict(mouse=mouse,kind=kind,seed=seed,second_stage_epochs=24,selected_epoch=selected,first_stage_selected_epoch=prior['selected_epoch'],selected_ancestry_updates=(prior['selected_epoch']+selected)*len(loader),selection_mse=best,initial_mse=history[0]['selection_mse'],grad_max=grad_max,initial_exact=True,reload_exact=True,batch_orders_exact=True,elapsed_seconds=time.monotonic()-start))
            print(mouse,kind,seed,'complete',flush=True)
    write(dest/'training_finished.json',dict(complete=True))


def lock():
    verify();assert not (ROOT/'selection_lock.json').exists();records=[];hashes={}
    for p in sorted(ROOT.glob('*/*/result.json')):
        r=read(p);meta=read(FAIR/r['mouse']/'metadata.json');lower=-meta['speed_mean']/meta['speed_std'];history=read(p.parent/'history.json')
        assert len(history)==25
        with np.load(p.parent/'selection_predictions.npz') as saved:errors=[mse(v,saved['target'],lower) for v in saved['prediction']]
        np.testing.assert_allclose(errors,[h['selection_mse'] for h in history],rtol=1e-12,atol=1e-12);assert int(np.argmin(errors))==r['selected_epoch']
        records.append(r);ck=p.parent/'selected.pt';hashes[str(ck.relative_to(ROOT))]=digest(ck)
    assert len(records)==24 and len(list(ROOT.glob('*/training_finished.json')))==4
    write(ROOT/'selection_lock.json',dict(locked_utc=datetime.now(timezone.utc).isoformat(),records=records,hashes=hashes,selection_scores_checked=600,later_scored=False))


def evaluate():
    verify();assert not (ROOT/'results.json').exists();locked=read(ROOT/'selection_lock.json')
    for name,h in locked['hashes'].items():assert digest(ROOT/name)==h
    rows=[]
    for mouse in MICE:
        meta=read(FAIR/mouse/'metadata.json');lower=-meta['speed_mean']/meta['speed_std'];indices=np.load(BASE/mouse/'columns.npy')
        with np.load(FAIR/mouse/'later_raw.npz') as raw,np.load(FAIR/mouse/'statistics.npz') as norm:
            seq=((raw['activity'][indices,24:]-norm['activity_mean'][indices])/norm['activity_std'][indices]).T.astype(np.float32)
            y=(raw['speed'][55:]-meta['speed_mean'])/meta['speed_std']
        x=torch.from_numpy(np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(seq,32,axis=0)));predictions={};scores={}
        for kind in ['attention','mixer']:
            for seed in SEEDS:
                net=make(mouse,kind,seed);net.load_state_dict(torch.load(ROOT/mouse/f'{kind}_s{seed}'/'selected.pt',weights_only=True));predictions[f'{kind}_restart_s{seed}']=predict(net,x)
        with np.load(BASE/mouse/'later_predictions.npz') as saved:
            np.testing.assert_array_equal(y,saved['target'])
            for arm in ['attention_scratch','attention_pretrained','mixer_scratch','mixer_pretrained','pooled_mlp']:
                for seed in SEEDS:predictions[f'{arm}_s{seed}']=saved[f'{arm}_s{seed}']
        with np.load(FLAT/mouse/'later_predictions.npz') as saved:
            np.testing.assert_array_equal(y,saved['target']);predictions['raw_ridge']=saved['raw_ridge']
        for label,p in predictions.items():
            scores[label]=mse(p,y,lower);independent=sum((max(float(a),lower)-float(b))**2 for a,b in zip(p,y))/len(y)
            assert abs(scores[label]-independent)<1e-10*max(1.,independent)
        np.savez_compressed(ROOT/mouse/'later_predictions.npz',target=y,**predictions);rows.append(dict(mouse=mouse,n=len(y),lower=lower,scores=scores));print(mouse,'scored',flush=True)
    verify();write(ROOT/'results.json',dict(rows=rows));write(ROOT/'audit.json',dict(passed=True,new_second_stage_fits=24,first_stage_fits_reused=24,epochs_per_new_fit=24,selection_scores_checked=600,initial_predictions_exactly_match_selected_first_stage=True,batch_orders_exact=True,selected_reload_exact=True,all_choices_locked_before_current_later_scoring=True,exact_target_alignment=True,independent_later_mse=True,frozen_hashes_unchanged=True))


if __name__=='__main__':
    with threadpool_limits(limits=2):
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        if sys.argv[1]=='train':train(sys.argv[2])
        else:{'freeze':freeze,'lock':lock,'evaluate':evaluate}[sys.argv[1]]()
