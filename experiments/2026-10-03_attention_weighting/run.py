"""Refit frozen readouts after replacing content-dependent attention with uniform weights."""
import copy
import importlib.util
import sys
import types
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
PREVIOUS=ROOT.parent/'2026-10-03_neuron_readout'
spec=importlib.util.spec_from_file_location('neuron_readout',PREVIOUS/'run.py')
flat=importlib.util.module_from_spec(spec);spec.loader.exec_module(flat)
old=flat.old
BASE,FAIR=old.BASE,old.FAIR
MICE,SEEDS=old.MICE,old.SEEDS
read,write,digest,mse=old.read,old.write,old.digest,old.mse
MODES=['uniform_population','uniform_both']


def uniform_forward(self,query,key,value,attn_mask=None,need_weights=False,**kwargs):
    assert not self.training and key is query and value is query
    width=self.embed_dim
    v=F.linear(value,self.in_proj_weight[2*width:],self.in_proj_bias[2*width:])
    if attn_mask is None:
        mixed=v.mean(1,keepdim=True).expand_as(v)
    else:
        expected=torch.triu(torch.ones(len(attn_mask),len(attn_mask),dtype=torch.bool,device=attn_mask.device),1)
        assert torch.equal(attn_mask,expected)
        mixed=v.cumsum(1)/torch.arange(1,v.shape[1]+1,device=v.device,dtype=v.dtype)[None,:,None]
    return F.linear(mixed,self.out_proj.weight,self.out_proj.bias),None


def check():
    torch.manual_seed(81207)
    layer=torch.nn.MultiheadAttention(16,2,batch_first=True,dropout=0.).eval()
    native=copy.deepcopy(layer)
    with torch.no_grad():
        native.in_proj_weight[:32].zero_();native.in_proj_bias[:32].zero_()
    x=torch.randn(3,8,16)
    mask=torch.triu(torch.ones(8,8,dtype=torch.bool),1)
    for causal in [None,mask]:
        expected=native(x,x,x,attn_mask=causal,need_weights=False)[0]
        actual=uniform_forward(layer,x,x,x,attn_mask=causal)[0]
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    altered=x.clone();altered[:,4:]+=10
    torch.testing.assert_close(uniform_forward(layer,x,x,x,attn_mask=mask)[0][:,:4],uniform_forward(layer,altered,altered,altered,attn_mask=mask)[0][:,:4],rtol=0,atol=0)
    write(ROOT/'selfcheck.json',dict(passed=True,uniform_matches_native_zero_query_key_logits=True,causal_prefix_invariance_exact=True,value_and_output_projections_retained=True))


def freeze():
    assert not (ROOT/'protocol.json').exists() and read(ROOT/'selfcheck.json')['passed']
    hashes=dict(read(PREVIOUS/'protocol.json')['hashes'])
    files=[Path(__file__),PREVIOUS/'protocol.json',PREVIOUS/'results.json']+[PREVIOUS/m/'later_predictions.npz' for m in MICE]
    hashes.update({str(p.relative_to(PROJECT)):digest(p) for p in files})
    write(ROOT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        question='Does input-dependent attention weighting improve neuron-preserving frozen readout performance relative to uniform weighting?',
        design='Archived attention encoders, random/pretrained, four mice and three seeds. Two interventions: uniform population attention with native temporal attention, or uniform population plus causal-prefix temporal attention. Preserve all value/output projections, residual paths, feature MLPs, embeddings and checkpoint tensors. Replace QK-derived weights only during inference. Refit the same standardized 2112-feature ridge readout in every condition. Reuse native predictions from the immediately preceding study.',
        budget='48 new readouts, 288 fixed ridge solves, zero encoder-training updates. Same six penalties, earlier selection, chronological splits and exact targets. Lock all choices before new later inference.',
        primary='Native pretrained attention versus uniform_population and uniform_both pretrained controls. Native-necessity gate requires at least 5% mean relative gain, 3/4 mouse wins and 8/12 paired-seed wins against each.',
        secondary='Same comparisons for random checkpoints; pretrained-versus-random effects within each weighting condition. These are diagnostic controls, not replacements for the primary gate.',
        uncertainty='2000 paired hierarchical mouse/seed/circular100-bin resamples, seed81207, descriptive97.5% intervals for two primary contrasts.',
        limits='An inference intervention on a frozen checkpoint with readout refitting, not training a uniform-attention architecture from scratch. Native pretraining may have shaped other weights through interactions. Equality cannot prove universal uselessness; harm may reflect coadaptation. Same reused development recordings, adaptive follow-up, no independent significance claim. No parameter updates, new datasets or publication.',hashes=hashes))


def verify():
    for name,h in read(ROOT/'protocol.json')['hashes'].items():assert digest(PROJECT/name)==h,name


def features(mouse,state,seed,mode,arrays):
    net=old.Dynamics('attention',seed)
    net.load_state_dict(torch.load(old.checkpoint(mouse,'attention',state,seed),weights_only=True))
    before={k:v.clone() for k,v in net.state_dict().items()}
    net.population.mix.forward=types.MethodType(uniform_forward,net.population.mix)
    if mode=='uniform_both':net.temporal.mix.forward=types.MethodType(uniform_forward,net.temporal.mix)
    net.eval().requires_grad_(False);outputs=[]
    with torch.inference_mode():
        for x in arrays:
            z=torch.cat([net.encode(torch.from_numpy(x[i:i+64]))[:,:,-1].flatten(1) for i in range(0,len(x),64)]).numpy().astype(np.float64)
            assert z.shape==(len(x),2048) and np.isfinite(z).all()
            outputs.append(np.concatenate([z,old.stats(x)],1))
    assert all(torch.equal(before[k],v) for k,v in net.state_dict().items())
    assert all(p.grad is None and not p.requires_grad for p in net.parameters())
    return outputs


def fit(mouse):
    verify();dest=ROOT/mouse;dest.mkdir()
    x,xv=[np.load(BASE/mouse/f'{s}_x.npy') for s in ['train','selection']]
    y,yv=[np.load(BASE/mouse/f'{s}_y.npy').astype(np.float64) for s in ['train','selection']]
    meta=read(FAIR/mouse/'metadata.json');lower=-meta['speed_mean']/meta['speed_std'];records=[]
    for state in ['random','pretrained']:
        for seed in SEEDS:
            for mode in MODES:
                tx,vx=features(mouse,state,seed,mode,[x,xv]);label=f'{state}_{mode}_s{seed}'
                records.append(flat.fit_probe(dest,label,tx,vx,y,yv,lower));print(mouse,label,'complete',flush=True)
    write(dest/'selection.json',dict(mouse=mouse,records=records))


def lock():
    verify();assert not (ROOT/'selection_lock.json').exists();rows=[read(ROOT/m/'selection.json') for m in MICE]
    assert sum(len(r['records']) for r in rows)==48
    write(ROOT/'selection_lock.json',dict(locked_utc=datetime.now(timezone.utc).isoformat(),rows=rows,hashes={str(p.relative_to(ROOT)):digest(p) for p in ROOT.glob('*/*.npz')},later_scored=False))


def evaluate():
    verify();assert not (ROOT/'results.json').exists();locked=read(ROOT/'selection_lock.json')
    for name,h in locked['hashes'].items():assert digest(ROOT/name)==h
    rows=[]
    for row in locked['rows']:
        mouse=row['mouse'];dest=ROOT/mouse;indices=np.load(BASE/mouse/'columns.npy')
        meta=read(FAIR/mouse/'metadata.json');lower=-meta['speed_mean']/meta['speed_std']
        with np.load(FAIR/mouse/'later_raw.npz') as raw,np.load(FAIR/mouse/'statistics.npz') as norm:
            seq=((raw['activity'][indices,24:]-norm['activity_mean'][indices])/norm['activity_std'][indices]).T.astype(np.float32)
            y=(raw['speed'][55:]-meta['speed_mean'])/meta['speed_std']
        x=np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(seq,32,axis=0));predictions={};scores={}
        for record in row['records']:
            label=record['label'];state,rest=label.split('_',1);mode,s=rest.rsplit('_s',1)
            values=features(mouse,state,int(s),mode,[x])[0]
            with np.load(dest/f'{label}.npz') as fitted:
                errors=[mse(p,fitted['target'],lower) for p in fitted['selection_predictions']]
                np.testing.assert_allclose(errors,[v['mse'] for v in record['candidates']],rtol=1e-12,atol=1e-12)
                assert int(np.argmin(errors))==record['selected_index']
                predictions[label]=np.einsum('ij,j->i',(values-fitted['center'])/fitted['scale'],fitted['weight'],optimize=False)+fitted['mean']
        with np.load(PREVIOUS/mouse/'later_predictions.npz') as prior:
            np.testing.assert_array_equal(y,prior['target'])
            for state in ['random','pretrained']:
                for seed in SEEDS:predictions[f'{state}_native_s{seed}']=prior[f'attention_{state}_s{seed}']
        for label,pred in predictions.items():
            scores[label]=mse(pred,y,lower)
            independent=sum((max(float(p),lower)-float(t))**2 for p,t in zip(pred,y))/len(y)
            assert abs(scores[label]-independent)<1e-10*max(1.,independent)
        np.savez_compressed(dest/'later_predictions.npz',target=y,**predictions)
        rows.append(dict(mouse=mouse,n=len(y),lower=lower,scores=scores));print(mouse,'scored',flush=True)
    verify();write(ROOT/'results.json',dict(rows=rows))
    write(ROOT/'audit.json',dict(passed=True,new_readouts=48,ridge_solves_and_selection_scores_checked=288,encoder_updates=0,uniform_weighting_selfcheck=True,encoder_tensors_unchanged=True,gradients_disabled=True,train_only_scaling=True,linear_optimality_and_independent_predictions=True,exact_target_alignment=True,independent_later_metrics=True,choices_locked_before_new_later_scoring=True,frozen_hashes_unchanged=True))


if __name__=='__main__':
    with threadpool_limits(limits=2):
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        if sys.argv[1]=='fit':fit(sys.argv[2])
        else:{'check':check,'freeze':freeze,'lock':lock,'evaluate':evaluate}[sys.argv[1]]()
