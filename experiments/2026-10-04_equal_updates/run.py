"""Match separate behavior decoders to the archived shared optimization budget."""
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import math
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
spec=importlib.util.spec_from_file_location('archived_shared_runner',SHARED/'run.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
BehaviorDecoder=old.BehaviorDecoder
MICE,SEEDS,KINDS=old.MICE,old.SEEDS,old.KINDS
BASE,FAIR=old.BASE,old.FAIR
read,write,digest,mse,predict=old.read,old.write,old.digest,old.mse,old.predict
BLOCKS=24


def sizes():return [len(np.load(BASE/m/'train_y.npy',mmap_mode='r')) for m in MICE]


def plan(lengths,session,seed):
    #cycle through complete local permutations without resetting at budget-block boundaries
    def stream():
        cycle=1
        while True:
            for s,indices in old.batches(lengths,[session],seed,cycle):yield cycle,indices
            cycle+=1
    source=stream();steps=sum((n+31)//32 for n in lengths)
    for block in range(1,BLOCKS+1):yield block,[next(source) for _ in range(steps)]


def plan_audit(lengths,session,seed):
    blocks=[];cycles={};examples=0
    for block,batches in plan(lengths,session,seed):
        h=hashlib.sha256()
        for cycle,indices in batches:
            assert len(indices)>0 and len(indices)<=32
            assert indices.min()>=0 and indices.max()<lengths[session]
            cycles.setdefault(cycle,[]).append(indices);examples+=len(indices);h.update(indices.tobytes())
        blocks.append(h.hexdigest())
    for cycle,parts in cycles.items():
        seen=np.concatenate(parts)
        assert len(np.unique(seen))==len(seen)
        if cycle<max(cycles):np.testing.assert_array_equal(np.sort(seen),np.arange(lengths[session]))
    return dict(block_hashes=blocks,examples=examples,complete_local_cycles=max(cycles)-1,last_cycle_examples=len(np.concatenate(cycles[max(cycles)])))


def check():
    lengths=sizes();steps=sum((n+31)//32 for n in lengths);checks=[]
    for session,m in enumerate(MICE):
        for seed in SEEDS:checks.append(dict(mouse=m,seed=seed,**plan_audit(lengths,session,seed)))
    parameter=nn.Parameter(torch.zeros(()));opt=torch.optim.AdamW([parameter],lr=.001,weight_decay=.01)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,BLOCKS,eta_min=.0001)
    for block in range(BLOCKS):
        expected=.0001+.0009*(1+math.cos(math.pi*block/BLOCKS))/2
        assert abs(opt.param_groups[0]['lr']-expected)<1e-14
        for _ in range(steps):
            opt.zero_grad(set_to_none=True);(parameter-1).square().backward();opt.step()
        schedule.step()
    assert int(opt.state[parameter]['step'])==BLOCKS*steps
    write(ROOT/'selfcheck.json',dict(passed=True,steps_per_block=steps,updates_per_fit=BLOCKS*steps,
        continuous_local_permutations=True,boundary_and_tail_checks=True,adam_step_counter_and_lr_schedule=True,plans=checks))


def freeze():
    assert not (ROOT/'protocol.json').exists()
    old_protocol=old.verify();assert read(ROOT/'selfcheck.json')['passed']
    lengths=sizes();steps=sum((n+31)//32 for n in lengths)
    files=[ROOT/'run.py',ROOT/'selfcheck.json',SHARED/'protocol.json',SHARED/'selection_lock.json',SHARED/'results.json',SHARED/'summary.json',SHARED/'audit.json']
    files += [PROJECT/name for name in old_protocol['hashes']]
    budget=[]
    for session,m in enumerate(MICE):
        exposure=plan_audit(lengths,session,10)
        budget.append(dict(mouse=m,training_windows=lengths[session],old_updates=24*((lengths[session]+31)//32),matched_updates=BLOCKS*steps,
            matched_examples=exposure['examples'],old_shared_examples=24*sum(lengths),complete_local_cycles=exposure['complete_local_cycles'],last_cycle_examples=exposure['last_cycle_examples']))
        files.append(SHARED/m/'later_predictions.npz')
        for kind in KINDS:
            for seed in SEEDS:
                files += [SHARED/m/f'{kind}_s{seed}'/name for name in ['initial.pt','result.json','selection_predictions.npz']]
    for kind in KINDS:
        for seed in SEEDS:files += [SHARED/'shared'/f'{kind}_s{seed}'/name for name in ['selected.pt','result.json','history.json']]
    write(ROOT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        question='Does the shared behavior-query transformer retain its advantage when separate models search exactly the same number of optimizer updates?',
        design='Reuse six completed shared fits and their locked predictions. Train24new separate fits:attention/static-pooling MLP x4mice x3seeds10/11/12. Same BehaviorDecoder class,128neurons,32bins,training/selection/later intervals and training-only normalization. No application edits,coordinates,reconstruction,newdata or shared retraining.',
        budget=budget,steps_per_block=steps,blocks=BLOCKS,updates_per_fit=BLOCKS*steps,
        optimization='Start from the exact archived separate initialization with fresh AdamW .001,wd.01,clip1,batch32. Repeat shuffled local permutations continuously without resetting at block boundaries. Same seed-based local permutation recipe. Every block has237optimizer steps. Cosine24 eta_min.0001 advances only at block boundaries,matching archived shared per-update LR schedule. Dropout seed=seed+9000. Local loss=MSE*batchN/32 as in the archived separate recipe. Shared loss additionally balanced mice;that difference is retained and disclosed.',
        matching='Exactly5688allocated updates per model and25checkpoint opportunities at the same cumulative steps as shared training. Both new families receive identical local batches. Partial batches make total example counts slightly different. Each separate model repeats its own recording;shared models train on all four. Parameter tying,number of session embeddings,mixed versus local loss scaling and checkpoint selection population differ. Not an isolated causal data-transfer intervention. Selected-checkpoint ancestry can differ despite equal searched budgets. Old24epochseparate predictions are descriptive references with shorter schedules.',
        selection='Separate model:minimum earlier bounded MSE among blocks0..24. Shared:reuse original single jointly selected checkpoint per family/seed. No later labels select checkpoints. Lock all24new choices before any new later inference;shared and historical later scores are already known.',
        primary='Within each family compare archived shared to new equal-update separate. Report both prespecified contrasts;each practical sharing gate requires>=5%equal-mouse mean relative MSE gain,>=3/4mouse wins,>=8/12pairedseed wins. Attention-sharing success is distinct from overall attention superiority.',
        overall_gate='Shared attention must satisfy its sharing gate and the same5%/3mouse/8seed thresholds against new equal-update separate MLP and archived shared MLP. Additionally>=5%and3mouse wins versus ridge,no mouse>25%worse than ridge,>=8/12wins versus own initial. The old shared-MLP contrast already failed;this control cannot erase that evidence or establish a fresh overall win.',
        secondary='New separate versus old short-budget separate within each family;shared attention versus new separate MLP;leave-one-mouse-out gains;initial controls;per-seed selected updates;difference between within-family relative sharing gains. No additional ablation or adaptive fit expansion.',
        uncertainty='2000paired hierarchical mouse/seed/circular100binbootstrap draws,seed81213,descriptive97.5%intervals for the two sharing contrasts. Resample seed identities jointly across mice because shared seeds identify shared fits. Conditional on fitted models,no retraining uncertainty,no correction for historical architecture selection.',
        scope='Known recordings and reused later intervals. No independent significance,unseen-mouse transfer,novel architecture or generation claim. Fixed24new fits then report all outcomes.',
        hashes={str(p.relative_to(PROJECT)):digest(p) for p in sorted(set(files))}))


def verify():
    protocol=read(ROOT/'protocol.json')
    for name,h in protocol['hashes'].items():assert digest(PROJECT/name)==h,name
    return protocol


def train(seed):
    protocol=verify();data=old.load_data();lengths=sizes()
    for session,m in enumerate(MICE):
        dest=ROOT/m;dest.mkdir(exist_ok=True);d=data[session]
        for kind in KINDS:
            out=dest/f'{kind}_s{seed}';out.mkdir();net=BehaviorDecoder(kind,seed,[session])
            archived=torch.load(SHARED/m/f'{kind}_s{seed}'/'initial.pt',weights_only=True)
            for k,v in net.state_dict().items():torch.testing.assert_close(v,archived[k],rtol=0,atol=0)
            first=predict(net,d['xv'],session)
            with np.load(SHARED/m/f'{kind}_s{seed}'/'selection_predictions.npz') as saved:
                np.testing.assert_array_equal(first,saved[m+'_predictions'][0]);np.testing.assert_array_equal(d['yv'],saved[m+'_target'])
            best=mse(first,d['yv'],d['lower']);selected=0;predictions=[first]
            torch.save(net.state_dict(),out/'selected.pt')
            history=[dict(block=0,updates=0,selection_mse=best)]
            torch.manual_seed(seed+9000);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
            schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,BLOCKS,eta_min=.0001)
            start=time.monotonic();updates=0;examples=0;grad_routing=0.;grad_temporal=0.
            for block,batches in plan(lengths,session,seed):
                net.train();h=hashlib.sha256();total=0.;count=0;lr=opt.param_groups[0]['lr']
                for cycle,indices in batches:
                    h.update(indices.tobytes());xb,yb=d['x'][indices],d['y'][indices];opt.zero_grad(set_to_none=True)
                    raw=(net(xb,session)-yb).square().mean();loss=raw*(len(indices)/32)
                    assert torch.isfinite(loss);loss.backward()
                    grad_routing=max(grad_routing,float(net.key_projection.weight.grad.norm()))
                    key='temporal.mix.in_proj_weight' if kind=='attention' else 'temporal.weight'
                    grad_temporal=max(grad_temporal,float(dict(net.named_parameters())[key].grad.norm()))
                    nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
                    updates+=1;examples+=len(indices);count+=len(indices);total+=float(raw.detach())*len(indices)
                schedule.step();p=predict(net,d['xv'],session);score=mse(p,d['yv'],d['lower']);predictions.append(p)
                if score<best:best=score;selected=block;torch.save(net.state_dict(),out/'selected.pt')
                history.append(dict(block=block,updates=updates,examples=examples,selection_mse=score,training_mse=total/count,
                    lr=lr,last_local_cycle=cycle,batch_hash=h.hexdigest()));write(out/'history.json',history)
            adam_steps=sorted(set(int(state['step']) for state in opt.state.values()))
            assert updates==protocol['updates_per_fit'] and adam_steps==[updates]
            assert grad_routing>0 and grad_temporal>0
            torch.save(net.state_dict(),out/'final.pt');net.load_state_dict(torch.load(out/'selected.pt',weights_only=True))
            np.testing.assert_array_equal(predict(net,d['xv'],session),predictions[selected])
            np.savez_compressed(out/'selection_predictions.npz',predictions=np.stack(predictions),target=d['yv'])
            write(out/'result.json',dict(mouse=m,kind=kind,seed=seed,selected_block=selected,selected_updates=selected*protocol['steps_per_block'],
                selection_mse=best,allocated_updates=updates,adam_steps=adam_steps,examples=examples,initial_exact=True,selected_reload_exact=True,
                routing_grad_max=grad_routing,temporal_grad_max=grad_temporal,elapsed_seconds=time.monotonic()-start))
            print(m,kind,seed,'complete',flush=True)
    write(ROOT/f'seed{seed}_finished.json',dict(complete=True))


def lock():
    protocol=verify();assert not (ROOT/'selection_lock.json').exists();records=[];hashes={};checked=0;orders={}
    expected={(m,k,s) for m in MICE for k in KINDS for s in SEEDS}
    for path in sorted(ROOT.glob('*/*/result.json')):
        r=read(path);m=r['mouse'];history=read(path.parent/'history.json');assert len(history)==25
        meta=read(FAIR/m/'metadata.json');lower=-meta['speed_mean']/meta['speed_std']
        with np.load(path.parent/'selection_predictions.npz') as saved:
            np.testing.assert_array_equal(saved['target'],np.load(BASE/m/'selection_y.npy'))
            scores=[mse(p,saved['target'],lower) for p in saved['predictions']];checked+=len(scores)
        np.testing.assert_allclose(scores,[h['selection_mse'] for h in history],rtol=1e-12,atol=1e-12)
        assert int(np.argmin(scores))==r['selected_block'];assert r['allocated_updates']==protocol['updates_per_fit']
        np.testing.assert_array_equal([h['updates'] for h in history],np.arange(25)*protocol['steps_per_block'])
        np.testing.assert_allclose([h['lr'] for h in history[1:]],.0001+.0009*(1+np.cos(np.pi*np.arange(24)/24))/2,rtol=1e-12,atol=1e-14)
        planned=plan_audit(sizes(),MICE.index(m),r['seed'])
        assert planned['block_hashes']==[h['batch_hash'] for h in history[1:]] and planned['examples']==r['examples']
        key=(m,r['seed'])
        if key in orders:assert orders[key]==planned['block_hashes']
        orders[key]=planned['block_hashes'];records.append(r);ck=path.parent/'selected.pt';hashes[str(ck.relative_to(ROOT))]=digest(ck)
    assert {(r['mouse'],r['kind'],r['seed']) for r in records}==expected
    assert len(records)==24 and checked==600 and len(list(ROOT.glob('seed*_finished.json')))==3
    write(ROOT/'selection_lock.json',dict(locked_utc=datetime.now(timezone.utc).isoformat(),records=records,hashes=hashes,
        selection_scores_checked=checked,equal_allocated_updates=True,matched_family_batches=True,matched_lr_schedule=True,new_later_scored=False))


def evaluate():
    verify();assert not (ROOT/'results.json').exists();locked=read(ROOT/'selection_lock.json');rows=[]
    for name,h in locked['hashes'].items():assert digest(ROOT/name)==h
    for session,m in enumerate(MICE):
        meta=read(FAIR/m/'metadata.json');lower=-meta['speed_mean']/meta['speed_std'];indices=np.load(BASE/m/'columns.npy')
        with np.load(FAIR/m/'later_raw.npz') as raw,np.load(FAIR/m/'statistics.npz') as norm:
            seq=((raw['activity'][indices,24:]-norm['activity_mean'][indices])/norm['activity_std'][indices]).T.astype(np.float32)
            y=(raw['speed'][55:]-meta['speed_mean'])/meta['speed_std']
        x=torch.from_numpy(np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(seq,32,axis=0)));predictions={}
        with np.load(SHARED/m/'later_predictions.npz') as saved:
            np.testing.assert_array_equal(y,saved['target'])
            for key in saved.files:
                if key!='target':predictions[key]=saved[key]
        for kind in KINDS:
            for seed in SEEDS:
                net=BehaviorDecoder(kind,seed,[session]);out=ROOT/m/f'{kind}_s{seed}';net.load_state_dict(torch.load(out/'selected.pt',weights_only=True))
                predictions[f'equal_{kind}_s{seed}']=predict(net,x,session)
        scores={}
        for label,p in predictions.items():
            scores[label]=mse(p,y,lower);independent=sum((max(float(a),lower)-float(b))**2 for a,b in zip(p,y))/len(y)
            assert abs(scores[label]-independent)<1e-10*max(1.,independent)
        np.savez_compressed(ROOT/m/'later_predictions.npz',target=y,**predictions);rows.append(dict(mouse=m,n=len(y),lower=lower,scores=scores));print(m,'scored',flush=True)
    verify();write(ROOT/'results.json',dict(rows=rows))
    write(ROOT/'audit.json',dict(passed=True,new_fits=24,shared_fits_reused=6,updates_per_new_fit=5688,selection_scores_checked=600,
        exact_archived_initialization=True,continuous_batch_plan_verified=True,matched_family_batches=True,matched_lr_schedule=True,
        actual_adam_step_counters_verified=True,exact_selected_reload=True,all_new_choices_locked_before_new_later_inference=True,
        exact_later_target_alignment=True,independent_later_metrics=True,source_input_application_and_inherited_hashes_unchanged=True))


if __name__=='__main__':
    with threadpool_limits(limits=2):
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        if sys.argv[1]=='train':train(int(sys.argv[2]))
        else:{'check':check,'freeze':freeze,'lock':lock,'evaluate':evaluate}[sys.argv[1]]()
