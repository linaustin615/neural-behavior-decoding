"""Fixed matched training experiment for a neuron-specific speed readout."""
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

from models import NeuronReadout

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
BASE=ROOT.parent/'2026-10-03_dynamics_baseline'
FLAT=ROOT.parent/'2026-10-03_neuron_readout'
FAIR=ROOT.parent/'2026-10-03_fair_comparison'
MICE=['MP030','MP032','MP033','MP034']
SEEDS=[10,11,12]


def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');temp.replace(p)
def checkpoint(m,k,s):return BASE/m/f'{k}_pretrain_s{s}'/'selected.pt'
def mse(p,y,lower):
    p=np.asarray(p,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    assert p.shape==y.shape and np.isfinite(p).all()
    return float(np.mean((np.maximum(p,lower)-y)**2))
def predict(net,x,remove=False):
    net.eval();parts=[]
    with torch.no_grad():
        for i in range(0,len(x),64):
            xb=x[i:i+64]
            parts.append(net.encoder(xb) if remove else net(xb))
    p=torch.cat(parts).numpy();assert np.isfinite(p).all();return p


def check():
    rows=[]
    for kind in ['attention','mixer']:
        net=NeuronReadout(kind,10,checkpoint('MP030',kind,10)).eval()
        torch.manual_seed(81208);x=torch.randn(4,128,32)
        with torch.no_grad():torch.testing.assert_close(net(x),net.encoder(x),rtol=0,atol=0)
        before=net.neuron_head.weight.detach().clone();optimizer=torch.optim.AdamW(net.parameters(),lr=.001)
        loss=(net(x)-torch.arange(4).float()).square().mean();loss.backward()
        assert net.neuron_head.weight.grad.norm()>0
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
        key='temporal.mix.in_proj_weight' if kind=='attention' else 'temporal.weight'
        assert dict(net.encoder.named_parameters())[key].grad.norm()>0
        optimizer.step();assert not torch.equal(before,net.neuron_head.weight)
        clone=NeuronReadout(kind,10,checkpoint('MP030',kind,10));clone.load_state_dict(net.state_dict())
        np.testing.assert_array_equal(predict(net,x),predict(clone,x))
        rows.append(dict(kind=kind,parameters=sum(p.numel() for p in net.parameters()),extra_readout_parameters=sum(p.numel() for p in net.neuron_head.parameters())))
    assert abs(rows[0]['parameters']/rows[1]['parameters']-1)<.01
    write(ROOT/'selfcheck.json',dict(passed=True,models=rows,initial_output_equals_old_head_exactly=True,new_head_and_encoder_gradients_finite_nonzero=True,head_updates=True,reload_exact=True))


def freeze():
    assert not (ROOT/'protocol.json').exists() and read(ROOT/'selfcheck.json')['passed']
    hashes=dict(read(FLAT/'protocol.json')['hashes'])
    files=[ROOT/'run.py',ROOT/'models.py',BASE/'selection_lock.json',FLAT/'protocol.json',FLAT/'results.json']
    for m in MICE:
        files+=[FLAT/m/'later_predictions.npz']
        for k in ['attention','mixer']:
            for s in SEEDS:files+=[BASE/m/f'{k}_pretrained_s{s}'/'history.json']
    hashes.update({str(p.relative_to(PROJECT)):digest(p) for p in files})
    write(ROOT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        question='Does a trainable neuron-specific readout improve the prior end-to-end pretrained attention model and earn an advantage over matched controls?',
        architecture='Keep the exact pretrained encoder and existing nonlinear pooled speed head. Add a zero-initialized linear 2048-to-1 head on separate last-patch neuron representations. Sum both outputs. Exactly 2049 extra parameters; identical design for attention and mixer. Initial predictions exactly equal the prior pretrained condition. No new pretraining.',
        budget='24 new fits: attention/mixer x four mice x seeds10/11/12, each24epochs. Same AdamW lr.001 wd.01, batch32, clip1, cosine24 eta_min.0001 as archived pretrained condition. All encoder and speed-head parameters train jointly; forecast head unused. Same shuffled batches and dropout seed as prior condition.',
        selection='Epoch0..24 by earlier bounded normalized speed MSE. Lock all24 choices before current later scoring. No adaptive recipes or extra epochs.',
        primary='New attention versus archived pretrained attention, new matched mixer, flat random attention with fitted ridge, equally tuned raw ridge, and archived pooled MLP, on identical endpoints.',
        gates='Separate readout-improvement gate: >=5% mean within-mouse gain, >=3/4 mice and >=8/12 paired seeds versus archived pretrained attention. Full attention gate additionally meets those thresholds against every other primary neural control and >=5%/3mice versus raw ridge, with no mouse >25% worse than pooled MLP.',
        secondary='Later epoch-zero comparisons and remove the added readout from the selected jointly trained model. Removal measures branch dependence/coadaptation, not superiority.',
        uncertainty='2000 paired hierarchical mouse/seed/circular100-bin bootstrap draws, seed81208, descriptive99% intervals for five primary contrasts.',
        limitations='Adaptive development on reused recordings, not independent confirmation. More readout capacity is part of the intervention. A neuron-specific decoder is fixed to the sampled neuron identities. No objective or data changes; no claim of novelty, significance or generation.',
        stopping='Complete these24fits and reports with no grid expansion. This closes the bounded readout/attention investigation for this batch.',hashes=hashes))


def verify():
    for name,h in read(ROOT/'protocol.json')['hashes'].items():assert digest(PROJECT/name)==h,name


def train(mouse):
    verify();dest=ROOT/mouse;dest.mkdir()
    x,xv=[torch.from_numpy(np.load(BASE/mouse/f'{s}_x.npy')) for s in ['train','selection']]
    y=torch.from_numpy(np.load(BASE/mouse/'train_y.npy').astype(np.float32));yv=np.load(BASE/mouse/'selection_y.npy')
    meta=read(FAIR/mouse/'metadata.json');lower=-meta['speed_mean']/meta['speed_std']
    for seed in SEEDS:
        for kind in ['attention','mixer']:
            out=dest/f'{kind}_s{seed}';out.mkdir();net=NeuronReadout(kind,seed,checkpoint(mouse,kind,seed))
            first=predict(net,xv);best=mse(first,yv,lower);selected=0;predictions=[first];history=[dict(epoch=0,selection_mse=best)]
            torch.save(net.state_dict(),out/'initial.pt');torch.save(net.state_dict(),out/'selected.pt')
            torch.manual_seed(seed+9000);generator=torch.Generator().manual_seed(seed)
            loader=DataLoader(TensorDataset(x,y,torch.arange(len(y))),batch_size=32,shuffle=True,generator=generator)
            optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,24,eta_min=.0001)
            reference=read(BASE/mouse/f'{kind}_pretrained_s{seed}'/'history.json')
            head_grad=0.;encoder_grad=0.;start=time.monotonic()
            key='temporal.mix.in_proj_weight' if kind=='attention' else 'temporal.weight'
            for epoch in range(1,25):
                net.train();total=0.;order=hashlib.sha256()
                for xb,yb,ib in loader:
                    order.update(ib.numpy().tobytes());optimizer.zero_grad(set_to_none=True)
                    loss=(net(xb)-yb).square().mean();assert torch.isfinite(loss);loss.backward()
                    head_grad=max(head_grad,float(net.neuron_head.weight.grad.norm()))
                    encoder_grad=max(encoder_grad,float(dict(net.encoder.named_parameters())[key].grad.norm()))
                    nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);optimizer.step();total+=float(loss.detach())*len(xb)
                scheduler.step();p=predict(net,xv);score=mse(p,yv,lower)
                assert order.hexdigest()==reference[epoch]['batch_order_hash']
                if score<best:best=score;selected=epoch;torch.save(net.state_dict(),out/'selected.pt')
                predictions.append(p);history.append(dict(epoch=epoch,training_mse=total/len(y),selection_mse=score,batch_order_hash=order.hexdigest()));write(out/'history.json',history)
            assert head_grad>0 and encoder_grad>0
            net.load_state_dict(torch.load(out/'selected.pt',weights_only=True));np.testing.assert_array_equal(predict(net,xv),predictions[selected])
            np.savez_compressed(out/'selection_predictions.npz',prediction=np.stack(predictions),target=yv)
            write(out/'result.json',dict(mouse=mouse,kind=kind,seed=seed,epochs=24,selected_epoch=selected,selection_mse=best,initial_mse=history[0]['selection_mse'],head_grad_max=head_grad,encoder_grad_max=encoder_grad,reload_exact=True,prior_batch_orders_exact=True,elapsed_seconds=time.monotonic()-start))
            print(mouse,kind,seed,'complete',flush=True)
    write(dest/'training_finished.json',dict(complete=True))


def lock():
    verify();assert not (ROOT/'selection_lock.json').exists();records=[];hashes={}
    for path in sorted(ROOT.glob('*/*/result.json')):
        r=read(path);meta=read(FAIR/r['mouse']/'metadata.json');lower=-meta['speed_mean']/meta['speed_std']
        history=read(path.parent/'history.json');assert len(history)==25
        with np.load(path.parent/'selection_predictions.npz') as saved:
            errors=[mse(p,saved['target'],lower) for p in saved['prediction']]
        np.testing.assert_allclose(errors,[h['selection_mse'] for h in history],rtol=1e-12,atol=1e-12)
        assert int(np.argmin(errors))==r['selected_epoch'];records.append(r)
        p=path.parent/'selected.pt';hashes[str(p.relative_to(ROOT))]=digest(p)
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
                out=ROOT/mouse/f'{kind}_s{seed}';net=NeuronReadout(kind,seed,checkpoint(mouse,kind,seed))
                net.load_state_dict(torch.load(out/'selected.pt',weights_only=True))
                predictions[f'{kind}_s{seed}']=predict(net,x);predictions[f'{kind}_removed_s{seed}']=predict(net,x,True)
                net.load_state_dict(torch.load(out/'initial.pt',weights_only=True));predictions[f'{kind}_initial_s{seed}']=predict(net,x)
        with np.load(FLAT/mouse/'later_predictions.npz') as prior:
            np.testing.assert_array_equal(y,prior['target']);predictions['raw_ridge']=prior['raw_ridge']
            for seed in SEEDS:
                for source,target in [('attention_random','random_attention'),('reference_attention_pretrained','prior_attention'),('reference_mixer_pretrained','prior_mixer'),('reference_pooled_mlp','pooled_mlp')]:predictions[f'{target}_s{seed}']=prior[f'{source}_s{seed}']
        for label,p in predictions.items():
            scores[label]=mse(p,y,lower);independent=sum((max(float(a),lower)-float(b))**2 for a,b in zip(p,y))/len(y)
            assert abs(scores[label]-independent)<1e-10*max(1.,independent)
        np.savez_compressed(ROOT/mouse/'later_predictions.npz',target=y,**predictions);rows.append(dict(mouse=mouse,n=len(y),lower=lower,scores=scores));print(mouse,'scored',flush=True)
    verify();write(ROOT/'results.json',dict(rows=rows))
    write(ROOT/'audit.json',dict(passed=True,new_fits=24,epochs_per_fit=24,selection_scores_checked=600,initial_output_equivalence=True,head_and_encoder_gradients=True,prior_batch_orders_exact=True,selected_reload_exact=True,choices_locked_before_current_later_scoring=True,exact_target_alignment=True,independent_later_mse=True,frozen_hashes_unchanged=True))


if __name__=='__main__':
    with threadpool_limits(limits=2):
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        parser=argparse.ArgumentParser();parser.add_argument('action');parser.add_argument('--mouse');args=parser.parse_args()
        if args.action=='train':train(args.mouse)
        else:{'check':check,'freeze':freeze,'lock':lock,'evaluate':evaluate}[args.action]()
