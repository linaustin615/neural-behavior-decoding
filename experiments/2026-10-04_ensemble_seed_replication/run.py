"""Replicate two-model ensemble comparisons with new, fixed training seeds."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
SHARED=ROOT.parent/'2026-10-03_shared_behavior'
sys.path.insert(0,str(SHARED))
spec=importlib.util.spec_from_file_location('shared_reference',SHARED/'run.py')
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
BehaviorDecoder=reference.BehaviorDecoder
MICE=reference.MICE
SEEDS=[13,14,15]
OLD_SEEDS=[10,11,12]
FAMILIES=['attention','mlp']


def read(path):return json.loads(path.read_text())
def write(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def ensemble_errors(a,m,y):
    assert a.shape==m.shape and a.ndim==2 and a.shape[1]==len(y) and len(a)>=2
    pairs=list(itertools.combinations(range(len(a)),2))
    same=((a+m)/2-y)**2
    aa=[];mm=[];cross=[];orientation=[]
    for i,j in pairs:
        aa.append(((a[i]+a[j])/2-y)**2)
        mm.append(((m[i]+m[j])/2-y)**2)
        forward=((a[i]+m[j])/2-y)**2
        reverse=((a[j]+m[i])/2-y)**2
        orientation.append(np.stack([forward,reverse]))
        cross.append((forward+reverse)/2)
    return dict(attention_single=(a-y)**2,mlp_single=(m-y)**2,
        attention_pair=np.stack(aa),mlp_pair=np.stack(mm),mixed_cross=np.stack(cross),
        mixed_same=same,mixed_orientations=np.stack(orientation)),pairs


def check():
    a=np.array([[0.,2.,-1.],[2.,0.,3.],[4.,-2.,1.]])
    m=np.array([[1.,3.,2.],[-1.,1.,0.],[3.,-1.,4.]])
    y=np.array([0.,1.,2.]);errors,pairs=ensemble_errors(a,m,y)
    assert pairs==[(0,1),(0,2),(1,2)]
    manual=[]
    for i,j in pairs:
        for t in range(3):
            manual.append((((a[i,t]+m[j,t])/2-y[t])**2+((a[j,t]+m[i,t])/2-y[t])**2)/2)
    assert abs(errors['mixed_cross'].mean()-np.mean(manual))<1e-12
    assert not np.allclose(errors['mixed_cross'][0],((a[0]+m[1]+a[1]+m[0])/4-y)**2)
    np.testing.assert_allclose(errors['mixed_same'],.5*((a-y)**2+(m-y)**2)-.25*(a-m)**2,atol=1e-12)
    zeros,_=ensemble_errors(np.zeros((3,5)),np.zeros((3,5)),np.zeros(5))
    assert all(np.all(x==0) for x in zeros.values())
    assert errors['mixed_cross'].shape[0]==3 and errors['mixed_orientations'].shape[:2]==(3,2)
    write(ROOT/'selfcheck.json',dict(passed=True,all_distinct_pairs=True,
        independent_scalar_ensemble_errors=True,orientation_errors_not_four_model_predictions=True,
        same_seed_average_identity=True,ties=True,pair_counts_not_orientation_replicates=True))


def freeze():
    assert read(ROOT/'selfcheck.json')['passed'] and not (ROOT/'protocol.json').exists()
    files=[Path(__file__),SHARED/'run.py',SHARED/'models.py',reference.BASE/'models.py',SHARED/'protocol.json']
    files += [PROJECT/n for n in ['train.py','model.py','data.py']]
    for mouse in MICE:
        files += [reference.BASE/mouse/n for n in ['train_x.npy','train_y.npy','selection_x.npy','selection_y.npy','columns.npy']]
        files += [reference.FAIR/mouse/n for n in ['metadata.json','statistics.npz','later_raw.npz']]
        files += [SHARED/mouse/'later_predictions.npz']
    for family in FAMILIES:
        for seed in OLD_SEEDS:
            files += [SHARED/'shared'/f'{family}_s{seed}'/n for n in ['result.json','selected.pt']]
    write(ROOT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        question='Does architecture diversity improve two-model ensembles once initialization pairing is matched, and does the result hold for new training seeds?',
        motivation='Earlier mixed ensemble paired AA/MS with the same seed,whereas AA/AA and MS/MS controls paired different seeds. Common non-temporal initialization and batch plans may matter;shared seed is not assumed to imply higher output correlation. Evaluate all pairings rather than select a favorable pair.',
        scope='Historical four-mouse Stringer cohort. New random initializations are an optimization-reproducibility check,not new animals,new data,or independent scientific confirmation. Architecture and tuning choices were informed by earlier outcomes. Prior fits and raw-data diagnostics are not repeated.',
        budget='Six new shared fits:unchanged AA transformer and MS static-query MLP,seeds13/14/15,24epochs,5688AdamWupdates and179712training windows per fit. Three seed workers,two torch threads each. No extra seeds after outcomes.',
        training='Exact imported shared BehaviorDecoder,initialization,batch generator,predictor and data loader. AdamW lr.001,wd.01,cosine24 eta_min.0001,batch32,clip1. Original equal-mouse batch weighting,totalN/(4*mouseN)*batchN/32. Same order within each seed across both families.',
        selection='One epoch0..24 per family/seed,original equal-mouse earlier bounded MSE divided by frozen original ridge selection denominators. All six choices lock before any new later inference. No ensemble weight,pair or checkpoint combination tuned on later data.',
        selection_denominators=read(SHARED/'protocol.json')['selection_denominators'],
        pairing='For each unordered seed pair(i,j),compare AA_i+AA_j,MS_i+MS_j,and the mean error of two mixed orientations AA_i+MS_j and AA_j+MS_i. Each predictor averages two bounded individual predictions50:50. Average orientation errors,not four predictions. Three matched pair blocks per mouse,12mouse-pair comparisons;not independent replications. Same-seed mixed ensembles are a secondary comparator.',
        primary='New-seed cross-seed mixed ensemble must beat both same-family two-model ensembles:>=2%equal-mouse mean relative MSE gain,>=3/4mouse means,and>=8/12matched mouse-pair wins for each. Independent units remain four animals;no sign-test over pairs or seed expansion.',
        secondary='Old seeds all-pair reanalysis frozen before new results;new single models and same-seed mixed averages;relative change cross versus same pairing;untrained baseline and learning;all individual scores;combined six-seed averages of errors over all15unordered pairs as descriptive context,not extra independent samples. Never pick a winning seed pair.',
        uncertainty='Descriptive paired mouse/seed/circular100-bin bootstrap,2000draws,seed81219,97.5%intervals for two primary contrasts. Seed-level Bayesian bootstrap weights shared across families/mice,products weight distinct pairs;diagonal pairs remain excluded. Conditional on fitted models,historical searches/retraining uncertainty omitted.',
        stop='Complete six fits,lock,later scoring,reports,audit. No new architecture,gate,extra seed,data alteration,application edit or publication in this study.',
        hashes={str(p.relative_to(PROJECT)):digest(p) for p in files}))


def verify():
    protocol=read(ROOT/'protocol.json')
    for rel,h in protocol['hashes'].items():assert digest(PROJECT/rel)==h,rel
    return protocol


def train(seed):
    assert seed in SEEDS
    protocol=verify();data=reference.load_data();sessions=list(range(4))
    lengths=[len(data[s]['x']) for s in sessions];total=sum(lengths)
    for family in FAMILIES:
        out=ROOT/f'{family}_s{seed}';out.mkdir()
        net=BehaviorDecoder(family,seed,sessions)
        def selection():
            p={s:reference.predict(net,data[s]['xv'],s) for s in sessions}
            scores={s:reference.mse(p[s],data[s]['yv'],data[s]['lower']) for s in sessions}
            return p,scores,float(np.mean([scores[s]/protocol['selection_denominators'][MICE[s]] for s in sessions]))
        first,scores,best=selection();predictions={s:[first[s]] for s in sessions}
        assert all(np.array_equal(p,np.zeros_like(p)) for p in first.values())
        history=[dict(epoch=0,selection_score=best,mouse_mse={MICE[s]:v for s,v in scores.items()})]
        torch.save(net.state_dict(),out/'initial.pt');torch.save(net.state_dict(),out/'selected.pt');selected=0
        torch.manual_seed(seed+9000);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,24,eta_min=.0001)
        start=time.monotonic();updates=0;examples=0;max_grad=0.
        for epoch in range(1,25):
            net.train();order=hashlib.sha256();counts=[0]*4;train_sse=[0.]*4
            for s,indices in reference.batches(lengths,sessions,seed,epoch):
                order.update(np.asarray([s],dtype=np.int64).tobytes()+indices.tobytes())
                xb,yb=data[s]['x'][indices],data[s]['y'][indices]
                opt.zero_grad(set_to_none=True);raw=(net(xb,s)-yb).square().mean()
                loss=raw*(total/(4*lengths[s]))*(len(indices)/32)
                assert torch.isfinite(loss);loss.backward()
                norm=nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
                max_grad=max(max_grad,float(norm));opt.step()
                updates+=1;examples+=len(indices);counts[s]+=len(indices);train_sse[s]+=float(raw.detach())*len(indices)
            assert counts==lengths
            schedule.step();p,scores,score=selection()
            if score<best:
                best=score;selected=epoch;torch.save(net.state_dict(),out/'selected.pt')
            for s in sessions:predictions[s].append(p[s])
            history.append(dict(epoch=epoch,selection_score=score,mouse_mse={MICE[s]:v for s,v in scores.items()},
                training_mse={MICE[s]:train_sse[s]/lengths[s] for s in sessions},global_order_hash=order.hexdigest(),
                examples_per_mouse=counts,updates=updates,examples=examples))
            write(out/'history.json',history)
            if epoch%6==0:print(seed,family,'epoch',epoch,'selected',selected,flush=True)
        actual_steps=sorted({int(v['step'].item()) for v in opt.state.values() if 'step' in v})
        assert updates==5688 and examples==179712 and actual_steps==[5688] and max_grad>0
        torch.save(net.state_dict(),out/'final.pt')
        net.load_state_dict(torch.load(out/'selected.pt',weights_only=True));saved={}
        for s in sessions:
            np.testing.assert_array_equal(reference.predict(net,data[s]['xv'],s),predictions[s][selected])
            saved[MICE[s]+'_predictions']=np.stack(predictions[s]);saved[MICE[s]+'_target']=data[s]['yv']
        np.savez_compressed(out/'selection_predictions.npz',**saved)
        write(out/'result.json',dict(family=family,seed=seed,selected_epoch=selected,selection_score=best,
            epochs=24,updates=updates,examples=examples,actual_adam_steps=actual_steps,
            selected_reload_exact=True,initial_predictions_zero=True,max_gradient=max_grad,
            elapsed_seconds=time.monotonic()-start))
        print(seed,family,'complete',flush=True)
    write(ROOT/f'seed{seed}_finished.json',dict(complete=True))


def lock():
    protocol=verify();assert not (ROOT/'selection_lock.json').exists()
    records=[];hashes={};orders={};checked=0
    for family in FAMILIES:
        for seed in SEEDS:
            out=ROOT/f'{family}_s{seed}';r=read(out/'result.json');history=read(out/'history.json')
            assert len(history)==25 and r['updates']==5688 and r['examples']==179712 and r['actual_adam_steps']==[5688]
            scores=[]
            with np.load(out/'selection_predictions.npz') as z:
                for m in MICE:
                    np.testing.assert_array_equal(z[m+'_target'],np.load(reference.BASE/m/'selection_y.npy'))
                    meta=read(reference.FAIR/m/'metadata.json');lower=-meta['speed_mean']/meta['speed_std']
                    values=[reference.mse(p,z[m+'_target'],lower) for p in z[m+'_predictions']]
                    np.testing.assert_allclose(values,[h['mouse_mse'][m] for h in history],rtol=1e-12,atol=1e-12)
                    scores.append(np.array(values)/protocol['selection_denominators'][m]);checked+=25
            joint=np.mean(scores,axis=0)
            np.testing.assert_allclose(joint,[h['selection_score'] for h in history],rtol=1e-12,atol=1e-12)
            assert int(np.argmin(joint))==r['selected_epoch']
            order=[h['global_order_hash'] for h in history[1:]]
            if seed in orders:assert orders[seed]==order
            orders[seed]=order;records.append(r);hashes[str((out/'selected.pt').relative_to(ROOT))]=digest(out/'selected.pt')
    assert checked==600 and len(list(ROOT.glob('seed*_finished.json')))==3
    write(ROOT/'selection_lock.json',dict(locked_utc=datetime.now(timezone.utc).isoformat(),records=records,
        hashes=hashes,selection_scores_checked=checked,matched_batches=True,new_later_scored=False))


def evaluate():
    verify();locked=read(ROOT/'selection_lock.json');assert not (ROOT/'results.json').exists()
    for rel,h in locked['hashes'].items():assert digest(ROOT/rel)==h,rel
    rows=[]
    for session,m in enumerate(MICE):
        meta=read(reference.FAIR/m/'metadata.json');lower=-meta['speed_mean']/meta['speed_std']
        columns=np.load(reference.BASE/m/'columns.npy')
        with np.load(reference.FAIR/m/'later_raw.npz') as raw,np.load(reference.FAIR/m/'statistics.npz') as norm:
            seq=((raw['activity'][columns,24:]-norm['activity_mean'][columns])/norm['activity_std'][columns]).T.astype(np.float32)
            y=(raw['speed'][55:]-meta['speed_mean'])/meta['speed_std']
        x=torch.from_numpy(np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(seq,32,axis=0)))
        predictions={'target':y,'initial':np.zeros(len(y))};scores={}
        for family in FAMILIES:
            for seed in SEEDS:
                net=BehaviorDecoder(family,seed,[0,1,2,3])
                net.load_state_dict(torch.load(ROOT/f'{family}_s{seed}'/'selected.pt',weights_only=True))
                p=reference.predict(net,x,session);predictions[f'{family}_s{seed}']=p
                score=reference.mse(p,y,lower)
                manual=sum((max(float(a),lower)-float(b))**2 for a,b in zip(p,y))/len(y)
                assert abs(score-manual)<1e-12*max(1,score);scores[f'{family}_s{seed}']=score
        with np.load(SHARED/m/'later_predictions.npz') as saved:
            np.testing.assert_array_equal(y,saved['target'])
            for family in FAMILIES:
                for seed in OLD_SEEDS:predictions[f'{family}_s{seed}']=saved[f'shared_{family}_s{seed}']
        out=ROOT/m;out.mkdir();np.savez_compressed(out/'later_predictions.npz',**predictions)
        rows.append(dict(mouse=m,n=len(y),lower=lower,new_scores=scores,initial_mse=reference.mse(predictions['initial'],y,lower)))
        print(m,'scored',flush=True)
    write(ROOT/'results.json',dict(rows=rows));verify()
    write(ROOT/'audit.json',dict(passed=True,new_shared_fits=6,updates_per_fit=5688,examples_per_fit=179712,
        all_choices_locked_before_new_later_inference=True,selection_scores_checked=600,matched_family_batches=True,
        actual_adam_steps=True,exact_selected_reloads=True,exact_target_alignment=True,
        independent_new_prediction_scores=24,frozen_hashes_unchanged=True,old_models_not_refitted=True))


if __name__=='__main__':
    with threadpool_limits(limits=2):
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        if sys.argv[1]=='train':train(int(sys.argv[2]))
        else:{'check':check,'freeze':freeze,'lock':lock,'evaluate':evaluate}[sys.argv[1]]()
