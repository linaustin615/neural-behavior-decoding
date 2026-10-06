"""Test stable neuron assignment and information beyond population summaries."""
from datetime import datetime,timezone
import hashlib
import importlib.util
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from models import AnonymousDecoder,BehaviorDecoder,StatisticsMLP,population_stats

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
SHARED=ROOT.parent/'2026-10-03_shared_behavior'
COMP=ROOT.parent/'2026-10-04_attention_components'
spec=importlib.util.spec_from_file_location('shared_information_runner',SHARED/'run.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
assert old.BehaviorDecoder is BehaviorDecoder
BASE,FAIR,MICE,SEEDS=old.BASE,old.FAIR,old.MICE,old.SEEDS
read,write,digest=old.read,old.write,old.digest
ARMS=['anonymous_attention','anonymous_mlp','statistics_mlp']


def reference(temporal,seed):
    return SHARED/'shared'/f'attention_s{seed}' if temporal=='attention' else COMP/f'ma_s{seed}'


def make_net(arm,seed):
    return StatisticsMLP(seed) if arm=='statistics_mlp' else AnonymousDecoder(arm.removeprefix('anonymous_'),seed)


def order(n,session,seed,split,epoch=0,view=0):
    rng=np.random.default_rng(np.random.SeedSequence([81215,seed,{'train':0,'selection':1,'later':2}[split],epoch,session,view]))
    return torch.from_numpy(rng.random((n,128)).argsort(axis=1).astype(np.int64))


def predict(net,x,session,banks=None):
    net.eval();views=[None] if banks is None else banks
    with torch.no_grad():
        result=np.stack([torch.cat([net(x[i:i+64],session,None if bank is None else bank[i:i+64]) for i in range(0,len(x),64)]).numpy() for bank in views])
    assert result.ndim==2 and np.isfinite(result).all();return result


def score(p,y,lower):
    assert p.ndim==2 and p.shape[1:]==y.shape
    return float(np.mean([old.mse(v,y,lower) for v in p]))


def check():
    torch.manual_seed(81215);x=torch.randn(3,128,32);snapshot=x.clone();o=order(3,0,10,'selection')
    np.testing.assert_array_equal(np.sort(o.numpy(),axis=1),np.tile(np.arange(128),(3,1)))
    torch.testing.assert_close(o,order(3,0,10,'selection'),rtol=0,atol=0)
    assert not torch.equal(o,order(3,0,10,'train',epoch=1)) and not torch.equal(order(3,0,10,'train',epoch=1),order(3,0,10,'train',epoch=2))
    shuffled=x.gather(1,o[:,:,None].expand_as(x))
    torch.testing.assert_close(shuffled.sort(dim=1).values,x.sort(dim=1).values,rtol=0,atol=0)
    for b in range(3):torch.testing.assert_close(shuffled[b],x[b,o[b]],rtol=0,atol=0)
    counts={};matched=0
    for temporal in ['attention','mlp']:
        net=AnonymousDecoder(temporal,10);parent=BehaviorDecoder(temporal,10,range(4));parent.kind='attention'
        for seed in SEEDS:
            state=torch.load(reference(temporal,seed)/'selected.pt',weights_only=True);net.load_state_dict(state);parent.load_state_dict(state)
            for s in range(4):np.testing.assert_array_equal(predict(net,x,s)[0],old.predict(parent,x,s));matched+=1
        captured=[];handle=net.head.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
        before={k:v.clone() for k,v in net.state_dict().items()}
        predict(net,x,0);predict(net,x,0,[o]);handle.remove()
        torch.testing.assert_close(captured[0][:,16:],captured[1][:,16:],rtol=0,atol=0)
        assert not torch.equal(captured[0][:,:16],captured[1][:,:16])
        for k,v in net.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
        counts['anonymous_'+temporal]=sum(p.numel() for p in net.parameters())
    for arm in ARMS:
        net=make_net(arm,10);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
        assert torch.equal(net(x,0,o),torch.zeros(3))
        for _ in range(2):
            opt.zero_grad(set_to_none=True);loss=(net(x,0,o)-torch.arange(3).float()).square().mean();loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters());opt.step()
        if arm=='statistics_mlp':assert net.head[1].weight.grad.norm()>0
        else:
            key='temporal.mix.in_proj_weight' if arm=='anonymous_attention' else 'temporal.weight'
            assert net.key_projection.weight.grad.norm()>0 and dict(net.named_parameters())[key].grad.norm()>0
        clone=make_net(arm,10);clone.load_state_dict(net.state_dict());np.testing.assert_array_equal(predict(net,x,0,[o]),predict(clone,x,0,[o]))
        counts[arm]=sum(p.numel() for p in net.parameters())
        if arm=='statistics_mlp':
            np.testing.assert_allclose(predict(net,x,0),predict(net,shuffled,0),rtol=1e-5,atol=1e-6)
    assert max(counts.values())/min(counts.values())<1.01
    torch.testing.assert_close(x,snapshot,rtol=0,atol=0)
    y=np.zeros(2);p=np.array([[1.,-1.],[-1.,1.]])
    assert score(p,y,-2)==1. and np.mean((p.mean(0)-y)**2)==0.
    write(ROOT/'selfcheck.json',dict(passed=True,parameter_counts=counts,trained_native_equivalences=matched,
        exact_whole_history_permutations=True,per_bin_value_multiset_preserved=True,summary_bypass_bitwise_unchanged=True,
        assignment_deterministic_and_changes_by_epoch=True,inputs_and_state_not_mutated=True,
        statistics_model_permutation_invariance=True,new_gradients_and_reloads=True,mean_error_not_ensemble_error=True))


def freeze():
    assert not (ROOT/'protocol.json').exists() and read(ROOT/'selfcheck.json')['passed']
    original=old.verify();files=[ROOT/'run.py',ROOT/'models.py',ROOT/'selfcheck.json',SHARED/'protocol.json',SHARED/'selection_lock.json',COMP/'protocol.json',COMP/'selection_lock.json']
    files += [PROJECT/p for p in original['hashes']]
    for m in MICE:files.append(COMP/m/'later_predictions.npz')
    for temporal in ['attention','mlp']:
        for seed in SEEDS:files += [reference(temporal,seed)/n for n in ['initial.pt','selected.pt','history.json','selection_predictions.npz','result.json']]
    write(ROOT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        question='Does the shared dynamic-query temporal transformer benefit from stable neuron-to-ID assignment and from information beyond per-bin population mean/std?',
        design='Nine new shared fits:anonymous temporal attention,anonymous temporal MLP(both dynamic query),and population-statistics-only MLP x3seeds10/11/12. Four known mice. Reuse native AA and MA from the component study;archived static-query AS/MS remain reference controls. Running speed,128neurons,32bins,chronological intervals,gaps and normalization unchanged.',
        reassignment='For every training window and epoch independently permute all128neuron rows identically across all32bins,after inherited per-neuron normalization and before patch embedding. Fixed ID slots remain in place. Preserve complete histories and the per-bin multiset of values. Compute the population mean/std bypass from original input so its64values are bitwise unchanged. This breaks stable slot-to-neuron assignment but does not guarantee neuron identity cannot be inferred from activity or erase co-firing patterns. It also acts as training augmentation.',
        randomization='Dedicated NumPy SeedSequence[81215,modelseed,splitcode,epoch,session,view] creates per-window permutations. Same banks for attention and MLP. Training uses one fresh assignment/window/epoch;selection/later use four fixed independent banks,split-specific,epoch0. No labels enter orders. Score mean of the four individual squared errors,not error of averaged predictions. Banks are repeated measurements,not independent samples or new animals.',
        statistics_control='64population mean/std history values plus16learned session/global features feed LayerNorm80,MLP80->160->32->1,GELUs,dropout.05. Zero output initialization.18385parameters vs18337attention/18327MLP. Same training/selection recipe. Architecture differs;this is a capacity-close practical summary-only control,not exact shared head weights or exhaustive sufficient-statistic testing.',
        budget='Each new fit:24epochs,5688optimizer updates,179712training examples. Same interleaved batches,AdamW.001/wd.01,clip1,cosine24 eta_min.0001,dropout seed=seed+9000,lossMSE*totalN/(4*mouseN)*batchN/32. Neural models start from exact corresponding native initial tensors. Statistics model starts at normalized0 too. Four-view evaluation raises inference cost,not training updates or checkpoint opportunities.',
        selection='One checkpoint per arm/seed across all four mice,epoch0..24,minimize equal-mouse mean bounded earlier MSE divided by archived earlier ridge MSE. Anonymous model MSE first averages the four order-view errors. Lock all nine choices before new later inference. No architecture choice from later results.',
        selection_denominators=original['selection_denominators'],
        primary=[dict(name='stable_assignment',main='native_attention',control='anonymous_attention'),dict(name='beyond_population_summaries',main='native_attention',control='statistics_mlp')],
        gates='Each primary gate requires>=5%equal-mouse mean relative gain,>=3/4mouse wins,>=8/12pairedseed wins. Combined neuron-information gate requires both. A failed gate does not establish equivalence or that neuron identity is universally useless.',
        secondary='Native MLP versus anonymous MLP and statistics MLP;anonymous attention versus anonymous MLP;anonymous models versus statistics;statistics versus ridge/shared static MLP;new-model initial checks;leave-one-mouse-out;four individual order-view errors and selected epochs. No temporal-order destruction or connectivity interpretation. No new model promoted as a later-data winner.',
        uncertainty='2000paired hierarchical mouse/seed/circular100binbootstrap draws,seed81215,descriptive97.5%intervals for two primary contrasts. Shared seed IDs resampled jointly across mice. Average per-time errors across four fixed order banks before resampling;conditional on these banks and fitted models,omits retraining/randomization-population uncertainty and historical selection.',
        scope='Training-time stable-assignment ablation and new nonlinear shared summary control,not a repeat of old frozen per-mouse linear statistics probes. All four animals historically reused;no independent significance,unseen-neuron/mouse transfer,coordinate,generation or novelty claim. Application unchanged. Fixed nine-fit batch;no adaptive grid expansion.',
        hashes={str(p.relative_to(PROJECT)):digest(p) for p in sorted(set(files))}))


def verify():
    p=read(ROOT/'protocol.json')
    for name,h in p['hashes'].items():assert digest(PROJECT/name)==h,name
    return p


def train(seed):
    protocol=verify();data=old.load_data();lengths=[len(data[s]['x']) for s in range(4)];totalN=sum(lengths)
    selection_banks={s:[order(len(data[s]['xv']),s,seed,'selection',view=v) for v in range(4)] for s in range(4)}
    for arm in ARMS:
        out=ROOT/f'{arm}_s{seed}';out.mkdir();net=make_net(arm,seed);anonymous=arm!='statistics_mlp'
        if anonymous:
            archived=torch.load(reference(net.temporal_kind,seed)/'initial.pt',weights_only=True)
            for k,v in net.state_dict().items():torch.testing.assert_close(v,archived[k],rtol=0,atol=0)
        banks=selection_banks if anonymous else {s:None for s in range(4)}
        def selection():
            predictions={s:predict(net,data[s]['xv'],s,banks[s]) for s in range(4)}
            scores={MICE[s]:score(predictions[s],data[s]['yv'],data[s]['lower']) for s in range(4)}
            aggregate=float(np.mean([scores[m]/protocol['selection_denominators'][m] for m in MICE]))
            return predictions,scores,aggregate
        first,scores,best=selection();selected=0;updates=0;examples=0;grad_body=0.;grad_routing=0.
        with np.load(reference('attention',seed)/'selection_predictions.npz') as saved:
            for s,m in enumerate(MICE):
                for view in first[s]:np.testing.assert_array_equal(view,saved[m+'_predictions'][0])
                np.testing.assert_array_equal(data[s]['yv'],saved[m+'_target'])
        history=[dict(epoch=0,updates=0,selection_score=best,mouse_mse=scores)];predictions={s:[first[s]] for s in range(4)}
        torch.save(net.state_dict(),out/'selected.pt');torch.save(net.state_dict(),out/'initial.pt')
        torch.manual_seed(seed+9000);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,24,eta_min=.0001);start=time.monotonic()
        for epoch in range(1,25):
            training_orders={s:order(lengths[s],s,seed,'train',epoch=epoch) for s in range(4)} if anonymous else None
            net.train();global_order=hashlib.sha256();assignment=hashlib.sha256();totals={m:0. for m in MICE}
            for s,indices in old.batches(lengths,range(4),seed,epoch):
                m=MICE[s];global_order.update(np.asarray([s],dtype=np.int64).tobytes()+indices.tobytes())
                permutation=training_orders[s][indices] if anonymous else None
                if anonymous:assignment.update(permutation.numpy().tobytes())
                opt.zero_grad(set_to_none=True);raw=(net(data[s]['x'][indices],s,permutation)-data[s]['y'][indices]).square().mean()
                loss=raw*(totalN/(4*lengths[s]))*(len(indices)/32);assert torch.isfinite(loss);loss.backward()
                if anonymous:
                    key='temporal.mix.in_proj_weight' if net.temporal_kind=='attention' else 'temporal.weight'
                    grad_body=max(grad_body,float(dict(net.named_parameters())[key].grad.norm()))
                    grad_routing=max(grad_routing,float(net.key_projection.weight.grad.norm()))
                else:grad_body=max(grad_body,float(net.head[1].weight.grad.norm()))
                nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
                updates+=1;examples+=len(indices);totals[m]+=float(raw.detach())*len(indices)
            schedule.step();p,scores,value=selection()
            if value<best:best=value;selected=epoch;torch.save(net.state_dict(),out/'selected.pt')
            for s in range(4):predictions[s].append(p[s])
            history.append(dict(epoch=epoch,updates=updates,examples=examples,selection_score=value,mouse_mse=scores,
                training_mse={m:totals[m]/lengths[s] for s,m in enumerate(MICE)},global_order_hash=global_order.hexdigest(),assignment_hash=assignment.hexdigest() if anonymous else None))
            write(out/'history.json',history)
        adam_steps=sorted(set(int(v['step']) for v in opt.state.values()));assert adam_steps==[5688] and updates==5688 and examples==179712
        assert grad_body>0 and (not anonymous or grad_routing>0)
        torch.save(net.state_dict(),out/'final.pt');net.load_state_dict(torch.load(out/'selected.pt',weights_only=True));saved={}
        for s,m in enumerate(MICE):
            np.testing.assert_array_equal(predict(net,data[s]['xv'],s,banks[s]),predictions[s][selected])
            saved[m+'_predictions']=np.stack(predictions[s]);saved[m+'_target']=data[s]['yv']
        np.savez_compressed(out/'selection_predictions.npz',**saved)
        write(out/'result.json',dict(arm=arm,seed=seed,selected_epoch=selected,selection_score=best,allocated_updates=updates,examples=examples,
            adam_steps=adam_steps,views=4 if anonymous else 1,initial_predictions_exact=True,native_initial_tensors_exact=anonymous,
            selected_reload_exact=True,body_grad_max=grad_body,routing_grad_max=grad_routing if anonymous else None,
            selection_order_hashes={MICE[s]:[hashlib.sha256(o.numpy().tobytes()).hexdigest() for o in banks[s]] for s in range(4)} if anonymous else None,
            elapsed_seconds=time.monotonic()-start))
        print(arm,seed,'complete',flush=True)
    write(ROOT/f'seed{seed}_finished.json',dict(complete=True))


def lock():
    protocol=verify();assert not (ROOT/'selection_lock.json').exists();records=[];hashes={};checked=0;view_scores=0;assignment_hashes={}
    lengths=[len(np.load(BASE/m/'train_y.npy')) for m in MICE]
    for path in sorted(ROOT.glob('*/result.json')):
        r=read(path);history=read(path.parent/'history.json');assert len(history)==25;metrics={}
        with np.load(path.parent/'selection_predictions.npz') as saved:
            for s,m in enumerate(MICE):
                meta=read(FAIR/m/'metadata.json');np.testing.assert_array_equal(saved[m+'_target'],np.load(BASE/m/'selection_y.npy'))
                assert saved[m+'_predictions'].shape[:2]==(25,r['views'])
                metrics[m]=[score(p,saved[m+'_target'],-meta['speed_mean']/meta['speed_std']) for p in saved[m+'_predictions']]
                checked+=25;view_scores+=25*r['views']
                np.testing.assert_allclose(metrics[m],[h['mouse_mse'][m] for h in history],rtol=1e-12,atol=1e-12)
                if r['views']==4:
                    expected=[hashlib.sha256(order(len(saved[m+'_target']),s,r['seed'],'selection',view=v).numpy().tobytes()).hexdigest() for v in range(4)]
                    assert expected==r['selection_order_hashes'][m]
        aggregate=np.mean([np.asarray(metrics[m])/protocol['selection_denominators'][m] for m in MICE],axis=0)
        np.testing.assert_allclose(aggregate,[h['selection_score'] for h in history],rtol=1e-12,atol=1e-12);assert int(np.argmin(aggregate))==r['selected_epoch']
        assert [h['updates'] for h in history]==[i*237 for i in range(25)] and r['adam_steps']==[5688] and r['examples']==179712
        previous=read(reference('attention',r['seed'])/'history.json')
        assert [h['global_order_hash'] for h in history[1:]]==[h['global_order_hash'] for h in previous[1:]]
        if r['views']==4:
            recorded=[h['assignment_hash'] for h in history[1:]]
            if r['seed'] in assignment_hashes:assert assignment_hashes[r['seed']]==recorded
            else:
                expected=[]
                for epoch in range(1,25):
                    banks={s:order(lengths[s],s,r['seed'],'train',epoch=epoch) for s in range(4)};h=hashlib.sha256()
                    for s,indices in old.batches(lengths,range(4),r['seed'],epoch):h.update(banks[s][indices].numpy().tobytes())
                    expected.append(h.hexdigest())
                assert expected==recorded;assignment_hashes[r['seed']]=recorded
        records.append(r);ck=path.parent/'selected.pt';hashes[str(ck.relative_to(ROOT))]=digest(ck)
    assert {(r['arm'],r['seed']) for r in records}=={(a,s) for a in ARMS for s in SEEDS}
    assert checked==900 and view_scores==2700 and len(list(ROOT.glob('seed*_finished.json')))==3
    write(ROOT/'selection_lock.json',dict(locked_utc=datetime.now(timezone.utc).isoformat(),records=records,hashes=hashes,
        mouse_epoch_scores_checked=checked,individual_view_scores_checked=view_scores,matched_batches=True,
        matched_and_regenerated_assignment_hashes=True,equal_allocated_updates_and_examples=True,new_later_scored=False))


def evaluate():
    verify();assert not (ROOT/'results.json').exists();locked=read(ROOT/'selection_lock.json');rows=[]
    for name,h in locked['hashes'].items():assert digest(ROOT/name)==h
    for s,m in enumerate(MICE):
        meta=read(FAIR/m/'metadata.json');lower=-meta['speed_mean']/meta['speed_std'];indices=np.load(BASE/m/'columns.npy')
        with np.load(FAIR/m/'later_raw.npz') as raw,np.load(FAIR/m/'statistics.npz') as norm:
            seq=((raw['activity'][indices,24:]-norm['activity_mean'][indices])/norm['activity_std'][indices]).T.astype(np.float32)
            y=(raw['speed'][55:]-meta['speed_mean'])/meta['speed_std']
        x=torch.from_numpy(np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(seq,32,axis=0)));predictions={};scores={};view_errors={};order_hashes={}
        with np.load(COMP/m/'later_predictions.npz') as saved:
            np.testing.assert_array_equal(y,saved['target']);predictions['raw_ridge']=saved['raw_ridge'][None]
            for seed in SEEDS:
                for label,original in [('native_attention','aa'),('native_mlp','ma'),('static_attention','as'),('static_mlp','ms'),('initial','initial')]:predictions[f'{label}_s{seed}']=saved[f'{original}_s{seed}'][None]
        for arm in ARMS:
            for seed in SEEDS:
                net=make_net(arm,seed);net.load_state_dict(torch.load(ROOT/f'{arm}_s{seed}'/'selected.pt',weights_only=True))
                banks=None if arm=='statistics_mlp' else [order(len(x),s,seed,'later',view=v) for v in range(4)]
                predictions[f'{arm}_s{seed}']=predict(net,x,s,banks)
                if banks is not None:order_hashes[str(seed)]=[hashlib.sha256(b.numpy().tobytes()).hexdigest() for b in banks]
        for label,p in predictions.items():
            scores[label]=score(p,y,lower);individual=[]
            for view in p:
                independent=sum((max(float(a),lower)-float(b))**2 for a,b in zip(view,y))/len(y)
                assert abs(independent-old.mse(view,y,lower))<1e-10*max(1.,independent);individual.append(independent)
            assert abs(scores[label]-sum(individual)/len(individual))<1e-10*max(1.,scores[label]);view_errors[label]=individual
        dest=ROOT/m;dest.mkdir();np.savez_compressed(dest/'later_predictions.npz',target=y,**predictions)
        rows.append(dict(mouse=m,n=len(y),lower=lower,scores=scores,individual_view_errors=view_errors,later_order_hashes=order_hashes));print(m,'scored',flush=True)
    verify();write(ROOT/'results.json',dict(rows=rows));write(ROOT/'audit.json',dict(passed=True,new_shared_fits=9,updates_per_fit=5688,examples_per_fit=179712,
        mouse_epoch_selection_scores_checked=900,individual_view_selection_scores_checked=2700,neural_initial_tensors_exact=True,
        initial_predictions_exact=True,trained_native_forward_equivalences=24,summary_bypass_bitwise_preserved=True,
        matched_batches_and_regenerated_assignments=True,selected_reload_exact=True,actual_adam_counters_verified=True,
        all_new_choices_locked_before_new_later_inference=True,exact_later_targets=True,independent_per_view_metrics=True,
        mean_error_not_ensemble_error=True,frozen_sources_inputs_references_application_unchanged=True))


if __name__=='__main__':
    with threadpool_limits(limits=2):
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        if sys.argv[1]=='train':train(int(sys.argv[2]))
        else:{'check':check,'freeze':freeze,'lock':lock,'evaluate':evaluate}[sys.argv[1]]()
