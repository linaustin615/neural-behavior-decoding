"""Fixed neuron-preserving readout diagnostic using archived frozen encoders."""
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
PREVIOUS = ROOT.parent / '2026-10-03_frozen_probe'
spec = importlib.util.spec_from_file_location('frozen_probe', PREVIOUS / 'run_probe.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
BASE, FAIR = old.BASE, old.FAIR
MICE, SEEDS, LAMBDAS = old.MICE, old.SEEDS, old.LAMBDAS
read, write, digest, mse = old.read, old.write, old.digest, old.mse


def freeze():
    assert not (ROOT/'protocol.json').exists()
    hashes = dict(read(PREVIOUS/'protocol.json')['hashes'])
    files = [Path(__file__), PREVIOUS/'protocol.json', PREVIOUS/'results.json']
    files += [PREVIOUS/m/'later_predictions.npz' for m in MICE]
    hashes.update({str(p.relative_to(PROJECT)):digest(p) for p in files})
    write(ROOT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        question='Does retaining neuron identity at readout improve decoding, and does learned attention outperform random features and matched controls?',
        design='Frozen archived attention/mixer, random/pretrained, four mice, three seeds. Flatten the last patch of 128x16 representations to 2048 features and append the same 64 population mean/std history values. Compare to archived pooled 16+64 probes. Same ridge objective, train-only feature scaling, six fixed penalties and earlier selection. No encoder training. Four new standardized raw-input ridge controls use the same six penalties. Total 52 readouts and 312 ridge solves.',
        lambda_grid=LAMBDAS,mice=MICE,seeds=SEEDS,
        primary='attention_pretrained versus pooled pretrained attention, flat random attention, flat pretrained mixer, new standardized raw ridge, and archived matched pooled MLP. All predictions refer to identical later targets. Equal-weight mean within-mouse relative MSE changes after averaging seed errors.',
        gate='At least 5% mean relative gain and 3/4 mouse wins against each primary control; at least 8/12 paired-seed wins against neural controls; no mouse more than 25% worse than MLP. Separate pooling-benefit gate: at least 5% gain, 3/4 mice and 8/12 seeds versus pooled pretrained attention.',
        uncertainty='2000 paired mouse/seed/circular100-bin bootstrap draws, seed81206, descriptive99% intervals across five primary contrasts.',
        limits='Adaptive exploratory follow-up on historically inspected recordings. No independent significance claim. Flattening changes both retained information and readout capacity/regularization geometry. It cannot alone attribute improvement uniquely to the averaging operation. Fixed neuron order is required, with no cross-session identity transfer. No extra datasets, encoder fits, publication or application edits.',
        stopping='Finish this fixed diagnostic and report. Use its result to motivate a separate subsequent protocol if warranted; do not change this grid after scoring.',hashes=hashes))


def verify():
    for name,h in read(ROOT/'protocol.json')['hashes'].items():
        assert digest(PROJECT/name)==h,name


def features(mouse,family,state,seed,arrays):
    net = old.Dynamics(family,seed)
    net.load_state_dict(torch.load(old.checkpoint(mouse,family,state,seed),weights_only=True))
    net.eval().requires_grad_(False)
    before = {k:v.clone() for k,v in net.state_dict().items()}
    outputs=[]
    with torch.inference_mode():
        for x in arrays:
            z=torch.cat([net.encode(torch.from_numpy(x[i:i+64]))[:,:,-1].flatten(1) for i in range(0,len(x),64)]).numpy().astype(np.float64)
            assert z.shape==(len(x),2048) and np.isfinite(z).all()
            outputs.append(np.concatenate([z,old.stats(x)],1))
    assert all(torch.equal(before[k],v) for k,v in net.state_dict().items())
    assert all(p.grad is None and not p.requires_grad for p in net.parameters())
    return outputs


def fit_probe(dest,label,x,xv,y,yv,lower):
    center=x.mean(0);scale=np.maximum(x.std(0),1e-8)
    z=torch.from_numpy((x-center)/scale).double();target=torch.from_numpy(y-y.mean()).double()
    dual=z.shape[1]>len(z)
    gram=z@z.T if dual else z.T@z
    rhs=target if dual else z.T@target
    candidates=[];predictions=[];weights=[];residuals=[]
    for lam in LAMBDAS:
        system=gram+len(y)*lam*torch.eye(len(gram),dtype=torch.float64)
        solution=torch.linalg.solve(system,rhs)
        residual=float((system@solution-rhs).norm()/rhs.norm().clamp_min(1e-12))
        assert residual<1e-7
        weight=z.T@solution if dual else solution
        optimality=float((z.T@(z@weight-target)+len(y)*lam*weight).norm()/(z.T@target).norm().clamp_min(1e-12))
        assert optimality<1e-7
        pred=(torch.from_numpy((xv-center)/scale)@weight+y.mean()).numpy()
        alternate=np.einsum('ij,j->i',(xv-center)/scale,weight.numpy(),optimize=False)+y.mean()
        np.testing.assert_allclose(pred,alternate,rtol=1e-8,atol=1e-8)
        candidates.append(dict(lam=lam,mse=mse(pred,yv,lower)));predictions.append(pred);weights.append(weight.numpy());residuals.append(optimality)
    index=int(np.argmin([v['mse'] for v in candidates]))
    np.savez_compressed(dest/f'{label}.npz',center=center,scale=scale,weight=weights[index],mean=y.mean(),selection_predictions=np.stack(predictions),target=yv)
    return dict(label=label,selected_index=index,**candidates[index],candidates=candidates,dual=dual,max_normal_equation_residual=max(residuals))


def fit(mouse):
    verify();dest=ROOT/mouse;dest.mkdir()
    x,xv=[np.load(BASE/mouse/f'{s}_x.npy') for s in ['train','selection']]
    y,yv=[np.load(BASE/mouse/f'{s}_y.npy').astype(np.float64) for s in ['train','selection']]
    meta=read(FAIR/mouse/'metadata.json');lower=-meta['speed_mean']/meta['speed_std']
    records=[fit_probe(dest,'raw_ridge',x.reshape(len(x),-1).astype(np.float64),xv.reshape(len(xv),-1).astype(np.float64),y,yv,lower)]
    for family in ['attention','mixer']:
        for state in ['random','pretrained']:
            for seed in SEEDS:
                tx,vx=features(mouse,family,state,seed,[x,xv])
                label=f'{family}_{state}_s{seed}'
                records.append(fit_probe(dest,label,tx,vx,y,yv,lower))
                print(mouse,label,'fit complete',flush=True)
    write(dest/'selection.json',dict(mouse=mouse,records=records))


def lock():
    verify();assert not (ROOT/'selection_lock.json').exists()
    rows=[read(ROOT/m/'selection.json') for m in MICE]
    assert sum(len(r['records']) for r in rows)==52
    write(ROOT/'selection_lock.json',dict(locked_utc=datetime.now(timezone.utc).isoformat(),rows=rows,
        hashes={str(p.relative_to(ROOT)):digest(p) for p in ROOT.glob('*/*.npz')},later_scored=False))


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
        x=np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(seq,32,axis=0))
        predictions={};scores={}
        for record in row['records']:
            label=record['label']
            if label=='raw_ridge':values=x.reshape(len(x),-1).astype(np.float64)
            else:
                family,state,s=label.split('_');values=features(mouse,family,state,int(s[1:]),[x])[0]
            with np.load(dest/f'{label}.npz') as fitted:
                errors=[mse(p,fitted['target'],lower) for p in fitted['selection_predictions']]
                np.testing.assert_allclose(errors,[v['mse'] for v in record['candidates']],rtol=1e-12,atol=1e-12)
                assert int(np.argmin(errors))==record['selected_index']
                predictions[label]=np.einsum('ij,j->i',(values-fitted['center'])/fitted['scale'],fitted['weight'],optimize=False)+fitted['mean']
        with np.load(PREVIOUS/mouse/'later_predictions.npz') as prior:
            np.testing.assert_array_equal(y,prior['target'])
            for family in ['attention','mixer']:
                for state in ['random','pretrained']:
                    for seed in SEEDS:predictions[f'pooled_{family}_{state}_s{seed}']=prior[f'{family}_{state}_augmented_s{seed}']
            predictions['statistics']=prior['statistics']
        with np.load(BASE/mouse/'later_predictions.npz') as prior:
            np.testing.assert_array_equal(y,prior['target'])
            for arm in ['attention_pretrained','mixer_pretrained','pooled_mlp']:
                for seed in SEEDS:predictions[f'reference_{arm}_s{seed}']=prior[f'{arm}_s{seed}']
            predictions['reference_ridge']=prior['ridge']
        for label,pred in predictions.items():
            scores[label]=mse(pred,y,lower)
            independent=sum((max(float(p),lower)-float(t))**2 for p,t in zip(pred,y))/len(y)
            assert abs(scores[label]-independent)<1e-10*max(1.,independent)
        np.savez_compressed(dest/'later_predictions.npz',target=y,**predictions)
        rows.append(dict(mouse=mouse,n=len(y),lower=lower,scores=scores));print(mouse,'scored',flush=True)
    verify();write(ROOT/'results.json',dict(rows=rows))
    write(ROOT/'audit.json',dict(passed=True,new_readouts=52,ridge_solves_and_selection_errors_checked=312,encoder_training=0,encoder_state_unchanged=True,gradients_disabled=True,train_only_feature_scaling=True,normal_equations_and_independent_predictions_passed=True,all_choices_locked_before_current_later_scoring=True,exact_target_alignment=True,independent_later_mse=True,frozen_hashes_unchanged=True))


if __name__=='__main__':
    with threadpool_limits(limits=2):
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        if sys.argv[1]=='fit':fit(sys.argv[2])
        else:{'freeze':freeze,'lock':lock,'evaluate':evaluate}[sys.argv[1]]()
