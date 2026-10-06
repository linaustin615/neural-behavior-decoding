"""Decisive new-adapter and split/solver checks without fitting real outcomes."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import types
import numpy as np
import torch
from common import ROOT, REPO, PLAN, MICE, save, digest, now
import dataset
import models
import fit
import ridge


def run():
    fit.initialize()
    checks={}
    torch.manual_seed(970)
    x=torch.randn(5,512,32)
    original_x=x.clone()
    comparisons=0
    for role,c in PLAN['architecture_configs'].items():
        parent=models.parent.PopulationDecoder(c,401).eval()
        shared=models.Decoder(c,401).eval()
        with torch.no_grad():
            parent.head[-1].weight.fill_(.02)
            parent.head[-1].bias.fill_(.1)
            shared.head.load_state_dict(parent.head.state_dict())
            for i in range(4):
                torch.testing.assert_close(shared(x,i),parent(x,i),rtol=0,atol=0)
                comparisons+=1
            for i in range(7):
                independent=models.Decoder(c,401,i).eval()
                independent.head.load_state_dict(parent.head.state_dict())
                torch.testing.assert_close(shared(x,i),independent(x,0),rtol=0,atol=0)
                comparisons+=1
            z=shared.encode(x,6)
            changed=x.clone()
            changed[:,:,16:]+=10
            torch.testing.assert_close(shared.encode(changed,6)[:,:16//c['patch']],z[:,:16//c['patch']],rtol=0,atol=0)
        shared.zero_grad(set_to_none=True)
        shared(x,6).square().mean().backward()
        assert shared.readin.grad[6].norm()>0
        assert shared.readin.grad[:6].abs().max()==0
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in shared.parameters())
        assert sum(float(p.grad.norm()) for p in shared.layers.parameters())>0
        clone=models.Decoder(c,401).eval()
        clone.load_state_dict(shared.state_dict())
        torch.testing.assert_close(clone(x,6),shared(x,6),rtol=0,atol=0)
    torch.testing.assert_close(x,original_x,rtol=0,atol=0)
    checks['adapter_parent_and_independent_exact_predictions']=comparisons
    checks['seven_session_gradients_causal_prefix_reload_input_preservation']=True

    rng=np.random.default_rng(891)
    a=rng.normal(size=(1024,1000)).astype(np.float32)
    y=np.sin(np.arange(1000)/15)+2
    root_before=dataset.ROOT
    metadata=[]
    with tempfile.TemporaryDirectory(prefix='facemap_preflight_') as temp:
        temp=Path(temp)
        for k in range(2):
            root=temp/str(k)
            folder=root/'acquired'/'fixture'
            folder.mkdir(parents=True)
            raw=root/'raw.npz'
            aa=a.copy()
            yy=y.copy()
            if k:
                aa[:,600:]+=1000
                yy[600:]+=100
            np.savez(raw,spks=aa,run=yy)
            save(folder/'integrity.json',dict(path=str(raw),uncompressed_sha256=digest(raw)))
            save(folder/'schema.json',dict(timing={'fixture':True}))
            dataset.ROOT=root
            np.testing.assert_array_equal(dataset.mmap_array(raw,'spks'),aa)
            metadata.append(dataset.prepare('fixture'))
            out=root/'prepared'/'fixture'
            panel=np.load(out/'panel.npy')
            np.testing.assert_allclose(np.load(out/'activity_mean.npy'),a[panel,:600].astype(np.float64).mean(1),rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(np.load(out/'activity_std.npy'),a[panel,:600].astype(np.float64).std(1),rtol=1e-12,atol=1e-12)
            for split,(start,end) in dataset.boundaries(1000).items():
                seq=np.load(out/(split+'_seq.npy'))
                target=np.load(out/(split+'_y.npy'))
                for h in (16,32,64):
                    windows=dataset.windows(seq,h)
                    np.testing.assert_array_equal(windows[0],seq[64-h:64].T)
                    np.testing.assert_array_equal(windows[-1],seq[-h:].T)
                    assert len(windows)==len(target)-63
                assert end-start>127
            if k:
                for name in ['panel','activity_mean','activity_std','train_seq','train_y','train_indices']:
                    np.testing.assert_array_equal(np.load(out/(name+'.npy')),np.load(temp/'0'/'prepared'/'fixture'/(name+'.npy')))
        assert metadata[0]['speed_mean']==metadata[1]['speed_mean']
        assert metadata[0]['speed_std']==metadata[1]['speed_std']
        dataset.ROOT=root_before
    checks['synthetic_future_perturbation_cannot_change_training_panel_scaling_or_labels']=True
    checks['mmap_matches_npz_and_three_history_endpoints_match']=True
    idx=dataset.train_indices(16000)
    assert len(idx)==4096 and idx[0]==63 and idx[-1]==15999 and len(np.unique(idx))==4096
    for seed in PLAN['seeds']:
        shared={i:[] for i in range(7)}
        for i,b in fit.orders([135]*7,list(range(7)),seed,1):
            shared[i].extend(b.tolist())
        for i in range(7):
            separate=[]
            for _,b in fit.orders([135],[i],seed,1):
                separate.extend(b.tolist())
            assert separate==shared[i] and sorted(separate)==list(range(135))
    checks['capped_endpoints_and_paired_batch_orders']=True

    x=rng.normal(size=(31,13))
    y=rng.normal(size=31)
    lam=[.0001,.1,10]
    state,residuals=ridge.solve(x,y,lam)
    z=(x-state['center'])/state['scale']
    for k,l in enumerate(lam):
        augmented=np.vstack([z,np.sqrt(l*len(x))*np.eye(x.shape[1])])
        labels=np.r_[y-y.mean(),np.zeros(x.shape[1])]
        expected=np.linalg.lstsq(augmented,labels,rcond=None)[0]
        np.testing.assert_allclose(state['weights'][:,k],expected,rtol=1e-8,atol=1e-8)
    checks['ridge_matches_independent_augmented_least_squares']=True
    checks['ridge_max_residual']=max(residuals)
    save(ROOT/'preflight.json',dict(passed=True,utc=now(),checks=checks,real_neural_fits=0,test_opened=False))
    freeze_sources()


def freeze_sources():
    paths={p for p in ROOT.glob('*.py')}
    paths.add(ROOT/'protocol.json')
    paths.add(ROOT/'acquisition_plan.json')
    pending=list(sys.modules.values())+[models.parent]
    visited=set()
    while pending:
        module=pending.pop()
        if id(module) in visited:
            continue
        visited.add(id(module))
        value=getattr(module,'__file__',None)
        if value:
            p=Path(value).resolve()
            if REPO in p.parents and p.suffix=='.py' and p.is_file():
                paths.add(p)
                pending.extend(v for v in vars(module).values() if isinstance(v,types.ModuleType))
    paths.update(REPO/p for p in ['train.py','model.py','data.py'])
    save(ROOT/'source_lock.json',dict(utc=now(),sha256={str(p.relative_to(REPO)):digest(p) for p in sorted(paths)}))
    print('New-adapter preflight passed;',len(paths),'sources locked',flush=True)


if __name__=='__main__':
    if '--freeze-only' in sys.argv:
        from common import read
        assert read(ROOT/'preflight.json')['passed']
        freeze_sources()
    else:
        run()
