"""Independent scalar, alignment, checkpoint and frozen-source verification."""
import math
import numpy as np
import torch
from common import ROOT, REPO, PLAN, MICE, read, save, now, digest, check_sources, job_name
from models import Decoder
import dataset


def audit(jobs):
    check_sources()
    for p,h in read(ROOT/'pipeline_source_lock.json')['sha256'].items():
        assert digest(REPO/p)==h,p
    for p,h in read(ROOT/'prepared_manifest.json')['sha256'].items():
        assert digest(ROOT/p)==h,p
    lock=read(ROOT/'evaluation_lock.json')
    for p,h in lock['selected_sha256'].items():
        assert digest(ROOT/p)==h,p
    assert read(ROOT/'preflight.json')['passed']
    scalars=predictions=orders=0
    rows=read(ROOT/'test_metrics.json')['rows']
    matched_orders={}
    updates=examples=0
    selected_epochs=[]
    shifts=[]
    for job in jobs:
        shared=job['regime']=='shared'
        ids=list(range(7)) if shared else [MICE.index(job['mouse'])]
        c=PLAN['architecture_configs'][job['role']]
        net=Decoder(c,job['seed'],None if shared else ids[0]).eval()
        path=ROOT/'fits'/job_name(job)
        result=read(path/'result.json')
        state=torch.load(path/'selected.pt',weights_only=True)
        net.load_state_dict(state)
        initial=torch.load(path/'initial.pt',weights_only=True)
        changes={name:float((state[name]-initial[name]).norm()) for name in state if state[name].dtype.is_floating_point and (name=='readin' or name.startswith('layers.') or name.startswith('head.'))}
        shifts.append(dict(job=job,selected_epoch=result['selected_epoch'],readin_change=changes['readin'],
            temporal_change=sum(v for k,v in changes.items() if k.startswith('layers.')),
            head_change=sum(v for k,v in changes.items() if k.startswith('head.'))))
        h=read(path/'history.json')
        assert len(h)==c['epochs']+1 and result['selected_epoch']==int(np.argmin([v['score'] for v in h]))
        assert result['completed_utc']<lock['utc']
        updates+=result['updates']
        examples+=result['examples']
        selected_epochs.append(result['selected_epoch'])
        lengths=[read(ROOT/'prepared'/MICE[i]/'metadata.json')['train_examples'] for i in ids]
        assert result['updates']==c['epochs']*sum(math.ceil(n/64) for n in lengths)
        assert result['examples']==c['epochs']*sum(lengths)
        for epoch in range(1,len(h)):
            for s,i in enumerate(ids):
                key=(MICE[i],job['seed'],epoch)
                value=h[epoch]['order_sha256'][s]
                if key in matched_orders:
                    assert matched_orders[key]==value
                    orders+=1
                else:
                    matched_orders[key]=value
        with np.load(ROOT/'test_predictions'/(job_name(job)+'.npz')) as z:
            for s,i in enumerate(ids):
                m=MICE[i]
                d=dataset.load(m,'test')
                p=z[m+'_prediction']
                target=z[m+'_target']
                np.testing.assert_array_equal(target,d['y'])
                idx=np.unique(np.r_[np.arange(min(64,len(p))),np.arange(max(0,len(p)-64),len(p))])
                with torch.inference_mode():
                    fresh=net(d['x'][idx],s if shared else 0).numpy()
                np.testing.assert_allclose(fresh,p[idx],rtol=1e-6,atol=1e-6)
                predictions+=len(idx)
                row=next(r for r in rows if r['regime']==job['regime'] and r['mouse']==m and r['role']==job['role'] and r['seed']==job['seed'])
                lo=d['meta']['lower']
                error=sum((max(float(a),lo)-float(b))**2 for a,b in zip(p,target))/len(p)
                absolute=sum(abs(max(float(a),lo)-float(b)) for a,b in zip(p,target))/len(p)
                raw=sum((float(a)-float(b))**2 for a,b in zip(p,target))/len(p)
                np.testing.assert_allclose([error,absolute,raw],[row['mse'],row['mae'],row['raw_mse']],rtol=1e-12,atol=1e-12)
                variance=sum((float(v)-float(target.mean()))**2 for v in target)/len(target)
                if variance>0:
                    np.testing.assert_allclose(1-error/variance,row['r2'],rtol=1e-11,atol=1e-11)
                scalars+=4
    raw_checks=0
    for m in MICE:
        out=ROOT/'prepared'/m
        meta=read(out/'metadata.json')
        rawpath=read(ROOT/'acquired'/m/'integrity.json')['path']
        a=dataset.mmap_array(rawpath,'spks')
        run=dataset.mmap_array(rawpath,'run')
        panel=np.load(out/'panel.npy')
        stop=meta['boundaries']['train'][1]
        mean=np.load(out/'activity_mean.npy')
        std=np.load(out/'activity_std.npy')
        prefix=np.asarray(a[panel,:stop],dtype=np.float64)
        np.testing.assert_allclose(prefix.mean(1),mean,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(prefix.std(1),std,rtol=1e-12,atol=1e-12)
        for split,(start,end) in meta['boundaries'].items():
            seq=np.load(out/(split+'_seq.npy'),mmap_mode='r')
            yy=np.load(out/(split+'_y.npy'))
            sample=np.unique(np.r_[np.arange(min(64,len(seq))),np.arange(max(0,len(seq)-64),len(seq))])
            expected=((np.asarray(a[np.ix_(panel,sample+start)],dtype=np.float64).T-mean)/std).astype(np.float32)
            np.testing.assert_array_equal(seq[sample],expected)
            expected_y=(np.abs(np.asarray(run[start:end],dtype=np.float64))-meta['speed_mean'])/meta['speed_std']
            np.testing.assert_array_equal(yy,expected_y)
            raw_checks+=len(sample)
        r=read(ROOT/'ridge'/m/'result.json')
        assert r['completed_utc']<lock['utc'] and len(r['options'])==24
        best=min(r['options'],key=lambda v:(v['validation_mse'],v['history'],v['lam']))
        assert best==r['selected']
        for history in PLAN['ridge']['histories']:
            with np.load(ROOT/'ridge'/m/f'validation_h{history}.npz') as z:
                for k,lam in enumerate(PLAN['ridge']['penalties']):
                    p,y=z['predictions'][:,k],z['target']
                    manual=sum((max(float(a),meta['lower'])-float(b))**2 for a,b in zip(p,y))/len(p)
                    saved=next(v['validation_mse'] for v in r['options'] if v['history']==history and v['lam']==lam)
                    np.testing.assert_allclose(manual,saved,rtol=1e-12,atol=1e-12)
                    scalars+=1
        with np.load(ROOT/'test_predictions'/('ridge_'+m+'.npz')) as z, np.load(ROOT/'ridge'/m/'selected.npz') as state:
            d=dataset.load(m,'test',r['selected']['history'])
            idx=np.arange(min(128,len(d['y'])))
            x=d['x'][idx].numpy().reshape(len(idx),-1).astype(np.float64)
            manual=np.einsum('ij,j->i',(x-state['center'])/state['scale'],state['weights'][:,0],optimize=False)+float(state['mean'])
            np.testing.assert_allclose(manual,z['prediction'][idx],rtol=1e-7,atol=1e-7)
            predictions+=len(idx)
            for role,p in [('ridge',z['prediction']),('training_mean',np.zeros(len(d['y']))),('zero_speed',np.full(len(d['y']),meta['lower']))]:
                row=next(v for v in rows if v['mouse']==m and v['role']==role)
                manual=sum((max(float(a),meta['lower'])-float(b))**2 for a,b in zip(p,z['target']))/len(p)
                np.testing.assert_allclose(manual,row['mse'],rtol=1e-12,atol=1e-12)
                scalars+=1
        dataset.load.cache_clear()
    summary=read(ROOT/'summary.json')
    for contrast in summary['contrasts']:
        effects=[]
        for m in MICE:
            a=[r for r in rows if r['regime']==contrast['regime'] and r['mouse']==m and r['role']=='optimized_attention']
            b=[r for r in rows if r['regime']==('control' if contrast['comparator']=='ridge' else contrast['regime']) and r['mouse']==m and r['role']==contrast['comparator']]
            effect=1-(sum(r['mse'] for r in a)/len(a))/(sum(r['mse'] for r in b)/len(b))
            effects.append(effect)
        np.testing.assert_allclose(sum(effects)/7,contrast['mean_relative_mse_gain'],rtol=1e-12,atol=1e-12)
        assert sum(x>0 for x in effects)==contrast['mouse_wins']
        assert all(contrast['practical_conditions'].values())==contrast['practical_pass']
    save(ROOT/'audit.json',dict(passed=True,utc=now(),neural_fits=len(jobs),ridge_solutions=168,
        independently_recomputed_scalar_errors=scalars,checkpoint_predictions_checked=predictions,
        raw_input_alignment_samples=raw_checks,paired_batch_order_hashes_checked=orders,
        optimizer_updates=updates,training_presentations=examples,selected_epoch_zero=sum(e==0 for e in selected_epochs),
        original_raw_download_integrity_checks_reused=True,all_choices_locked_before_test=True,
        source_prepared_checkpoint_hashes_verified=True,parameter_changes=shifts))
    print('Independent audit passed',flush=True)
