"""Complete the shared-decoder temporal/query factorial with six new fits."""
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

from models import BehaviorDecoder,FactorialDecoder,VARIANTS

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
SHARED=ROOT.parent/'2026-10-03_shared_behavior'
EQUAL=ROOT.parent/'2026-10-04_equal_updates'
spec=importlib.util.spec_from_file_location('shared_runner_components',SHARED/'run.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
assert old.BehaviorDecoder is BehaviorDecoder
BASE,FAIR=old.BASE,old.FAIR
MICE,SEEDS=old.MICE,old.SEEDS
read,write,digest,mse,predict=old.read,old.write,old.digest,old.mse,old.predict
NEW=['as','ma']


def check():
    torch.manual_seed(81214);x=torch.randn(3,128,32);changed=x+torch.randn_like(x)
    counts={};matched=0
    for variant,(temporal,query) in VARIANTS.items():
        net=FactorialDecoder(variant,10,range(4)).eval();parent=BehaviorDecoder(temporal,10,range(4)).eval()
        for k,v in net.state_dict().items():torch.testing.assert_close(v,parent.state_dict()[k],rtol=0,atol=0)
        assert net.temporal.kind==('attention' if temporal=='attention' else 'mixer')
        with torch.no_grad():
            _,a=net(x,0,return_weights=True);_,b=net(changed,0,return_weights=True)
        if query=='mlp':torch.testing.assert_close(a,b,rtol=0,atol=0)
        else:assert not torch.allclose(a,b,rtol=1e-5,atol=1e-7)
        counts[variant]=sum(p.numel() for p in net.parameters())
        if variant not in NEW:
            for seed in SEEDS:
                state=torch.load(SHARED/'shared'/f'{temporal}_s{seed}'/'selected.pt',weights_only=True)
                net.load_state_dict(state);parent.load_state_dict(state)
                for session in range(4):np.testing.assert_array_equal(predict(net,x,session),predict(parent,x,session));matched+=1
        else:
            net.train();opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
            for _ in range(2):
                opt.zero_grad(set_to_none=True);loss=(net(x,0)-torch.arange(3).float()).square().mean();loss.backward()
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters());opt.step()
            key='temporal.mix.in_proj_weight' if temporal=='attention' else 'temporal.weight'
            assert dict(net.named_parameters())[key].grad.norm()>0 and net.key_projection.weight.grad.norm()>0
            clone=FactorialDecoder(variant,10,range(4));clone.load_state_dict(net.state_dict())
            np.testing.assert_array_equal(predict(net,x,0),predict(clone,x,0))
    assert counts['aa']==counts['as'] and counts['ma']==counts['ms'] and abs(counts['aa']/counts['ms']-1)<.01
    write(ROOT/'selfcheck.json',dict(passed=True,parameter_counts=counts,identical_tensors_for_query_contrast=True,
        temporal_choice_independent_of_query=True,static_keys_input_invariant=True,dynamic_keys_input_dependent=True,
        trained_parent_equivalences=matched,new_variant_gradients_and_reloads=True))


def freeze():
    assert not (ROOT/'protocol.json').exists() and read(ROOT/'selfcheck.json')['passed']
    original=old.verify();files=[ROOT/'models.py',ROOT/'run.py',ROOT/'selfcheck.json',SHARED/'protocol.json',SHARED/'selection_lock.json',SHARED/'summary.json',EQUAL/'protocol.json',EQUAL/'selection_lock.json']
    files += [PROJECT/p for p in original['hashes']]
    for m in MICE:files += [SHARED/m/'later_predictions.npz',EQUAL/m/'later_predictions.npz']
    for kind in ['attention','mlp']:
        for seed in SEEDS:files += [SHARED/'shared'/f'{kind}_s{seed}'/n for n in ['initial.pt','selected.pt','history.json','selection_predictions.npz','result.json']]
    write(ROOT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        question='Which component contributes to the shared running-speed decoder:temporal attention,activity-dependent query pooling,or their combination?',
        variants={k:dict(temporal=v[0],query='dynamic' if v[1]=='attention' else 'static',new=k in NEW) for k,v in VARIANTS.items()},
        design='Complete a2x2shared-training factorial. Reuse AA(temporal attention,dynamic query) and MS(temporal MLP mixer,static query),three seeds each. Six new fits:AS(temporal attention,static query) and MA(temporal MLP mixer,dynamic query),three seeds10/11/12. Four known mice jointly trained;one selected checkpoint per variant/seed. No separate-model fits or application changes.',
        intervention='Construct the exact archived BehaviorDecoder temporal family,then independently set its query-key selector. Within a temporal family all parameter tensors,initialization,capacity and forward operations except routing-key source are identical. Dynamic keys derive from activity-containing tokens;static keys derive only from learned neuron/time/session embeddings. Values always contain activity. Temporal attention vs causal MLP mixing retains the original10parameter difference. Both variants include residual MLPs and population mean/std histories.',
        budget='24epochs x237updates=5688updates and179712example exposures per fit. Same archived interleaved per-mouse batches,AdamW.001,wd.01,batch32,clip1,cosine24 eta_min.0001,dropout seed=seed+9000,equal-mouse training loss weight totalN/(4*mouseN)*batchN/32. All new fits start from corresponding archived shared initial tensors. Allocated budgets and exposure match all four corners;selected epoch can differ.',
        selection='One joint checkpoint per variant/seed,minimum equal-mouse earlier bounded MSE divided by archived earlier raw-ridge MSE,epoch0..24. Same selection rule and denominators as archived corners. Lock all six new choices before new later inference. Archived later scores were already inspected;no fresh confirmation.',
        selection_denominators=original['selection_denominators'],
        primary=[dict(name='query_with_attention_time',main='aa',control='as'),dict(name='query_with_mlp_time',main='ma',control='ms'),
            dict(name='temporal_with_dynamic_query',main='aa',control='ma'),dict(name='temporal_with_static_query',main='as',control='ms')],
        gates='Each simple-effect gate requires>=5%equal-mouse mean relative gain,>=3/4mouse wins,>=8/12pairedseed wins. General query-benefit gate requires both query contrasts;general temporal-attention gate requires both temporal contrasts. A conditional benefit is not a general benefit. Report all contrasts and failures;no gate threshold changes after results.',
        secondary='For each new crossed variant,compare AA,MS,equal-update separate MLP,and raw ridge. A practical candidate gate requires>=5%mean gain and>=3/4mice for all four,>=8/12pairedseeds for neural references,no mouse>25%worse than ridge,and>=8/12wins over own initial. Report both without selecting a later-data winner. Interaction per mouse=(MSE_AS-MSE_AA-MSE_MS+MSE_MA)/MSE_MS;positive means dynamic query reduces error more with temporal attention on this additive error scale. Descriptive only. Include leave-one-mouse-out and selected epochs.',
        uncertainty='2000paired hierarchical mouse/seed/circular100binbootstrap draws,seed81214,descriptive98.75%intervals for four primary contrasts. Shared seed indices resampled jointly across mice;time blocks paired across all corners. Conditional on fitted models,no retraining uncertainty or adjustment for all historical searches.',
        scope='Training-time architecture comparison,not inference-only ablation or connectivity interpretation. Same128neurons,32activity bins,chronological splits,gaps,training-only normalization and running-speed target. Four historically reused animals;no independent significance,novel architecture,unseen-animal transfer,coordinates or generation claim. No adaptive extension.',
        hashes={str(p.relative_to(PROJECT)):digest(p) for p in sorted(set(files))}))


def verify():
    p=read(ROOT/'protocol.json')
    for name,h in p['hashes'].items():assert digest(PROJECT/name)==h,name
    return p


def train(seed):
    protocol=verify();data=old.load_data();lengths=[len(data[s]['x']) for s in range(4)];totalN=sum(lengths)
    for variant in NEW:
        out=ROOT/f'{variant}_s{seed}';out.mkdir();net=FactorialDecoder(variant,seed,range(4))
        parent=SHARED/'shared'/f'{net.temporal_kind}_s{seed}'
        archived=torch.load(parent/'initial.pt',weights_only=True)
        for k,v in net.state_dict().items():torch.testing.assert_close(v,archived[k],rtol=0,atol=0)
        def selection():
            predictions={s:predict(net,data[s]['xv'],s) for s in range(4)}
            scores={MICE[s]:mse(predictions[s],data[s]['yv'],data[s]['lower']) for s in range(4)}
            aggregate=float(np.mean([scores[m]/protocol['selection_denominators'][m] for m in MICE]))
            return predictions,scores,aggregate
        first,scores,best=selection();history=[dict(epoch=0,updates=0,selection_score=best,mouse_mse=scores)]
        predictions={s:[first[s]] for s in range(4)};selected=0;updates=0;examples=0;grad_routing=0.;grad_temporal=0.
        with np.load(parent/'selection_predictions.npz') as saved:
            for s,m in enumerate(MICE):np.testing.assert_array_equal(first[s],saved[m+'_predictions'][0]);np.testing.assert_array_equal(data[s]['yv'],saved[m+'_target'])
        torch.save(net.state_dict(),out/'selected.pt');torch.save(net.state_dict(),out/'initial.pt')
        torch.manual_seed(seed+9000);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,24,eta_min=.0001);start=time.monotonic()
        for epoch in range(1,25):
            net.train();orders={m:hashlib.sha256() for m in MICE};global_order=hashlib.sha256();total={m:0. for m in MICE}
            for s,indices in old.batches(lengths,range(4),seed,epoch):
                m=MICE[s];orders[m].update(indices.tobytes());global_order.update(np.asarray([s],dtype=np.int64).tobytes()+indices.tobytes())
                opt.zero_grad(set_to_none=True);raw=(net(data[s]['x'][indices],s)-data[s]['y'][indices]).square().mean()
                loss=raw*(totalN/(4*lengths[s]))*(len(indices)/32);assert torch.isfinite(loss);loss.backward()
                grad_routing=max(grad_routing,float(net.key_projection.weight.grad.norm()))
                key='temporal.mix.in_proj_weight' if net.temporal_kind=='attention' else 'temporal.weight'
                grad_temporal=max(grad_temporal,float(dict(net.named_parameters())[key].grad.norm()))
                nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
                updates+=1;examples+=len(indices);total[m]+=float(raw.detach())*len(indices)
            schedule.step();pred,scores,score=selection()
            if score<best:best=score;selected=epoch;torch.save(net.state_dict(),out/'selected.pt')
            for s in range(4):predictions[s].append(pred[s])
            history.append(dict(epoch=epoch,updates=updates,examples=examples,selection_score=score,mouse_mse=scores,
                training_mse={m:total[m]/lengths[s] for s,m in enumerate(MICE)},batch_order_hashes={m:h.hexdigest() for m,h in orders.items()},global_order_hash=global_order.hexdigest()))
            write(out/'history.json',history)
        steps=sorted(set(int(v['step']) for v in opt.state.values()));assert steps==[5688] and updates==5688 and examples==179712
        assert grad_routing>0 and grad_temporal>0
        torch.save(net.state_dict(),out/'final.pt');net.load_state_dict(torch.load(out/'selected.pt',weights_only=True));saved={}
        for s,m in enumerate(MICE):
            np.testing.assert_array_equal(predict(net,data[s]['xv'],s),predictions[s][selected]);saved[m+'_predictions']=np.stack(predictions[s]);saved[m+'_target']=data[s]['yv']
        np.savez_compressed(out/'selection_predictions.npz',**saved)
        write(out/'result.json',dict(variant=variant,seed=seed,selected_epoch=selected,selection_score=best,allocated_updates=updates,examples=examples,
            adam_steps=steps,initial_tensors_and_predictions_exact=True,selected_reload_exact=True,routing_grad_max=grad_routing,temporal_grad_max=grad_temporal,elapsed_seconds=time.monotonic()-start))
        print(variant,seed,'complete',flush=True)
    write(ROOT/f'seed{seed}_finished.json',dict(complete=True))


def lock():
    protocol=verify();assert not (ROOT/'selection_lock.json').exists();records=[];hashes={};checked=0
    for path in sorted(ROOT.glob('*/result.json')):
        r=read(path);history=read(path.parent/'history.json');assert len(history)==25;metrics={}
        with np.load(path.parent/'selection_predictions.npz') as saved:
            for m in MICE:
                meta=read(FAIR/m/'metadata.json');np.testing.assert_array_equal(saved[m+'_target'],np.load(BASE/m/'selection_y.npy'))
                metrics[m]=[mse(p,saved[m+'_target'],-meta['speed_mean']/meta['speed_std']) for p in saved[m+'_predictions']];checked+=25
                np.testing.assert_allclose(metrics[m],[h['mouse_mse'][m] for h in history],rtol=1e-12,atol=1e-12)
        scores=np.mean([np.asarray(metrics[m])/protocol['selection_denominators'][m] for m in MICE],axis=0)
        np.testing.assert_allclose(scores,[h['selection_score'] for h in history],rtol=1e-12,atol=1e-12);assert int(np.argmin(scores))==r['selected_epoch']
        assert [h['updates'] for h in history]==[237*i for i in range(25)]
        assert r['examples']==179712 and r['adam_steps']==[5688]
        for kind in ['attention','mlp']:
            previous=read(SHARED/'shared'/f'{kind}_s{r["seed"]}'/'history.json')
            for a,b in zip(history[1:],previous[1:]):assert a['global_order_hash']==b['global_order_hash'] and a['batch_order_hashes']==b['batch_order_hashes']
        records.append(r);ck=path.parent/'selected.pt';hashes[str(ck.relative_to(ROOT))]=digest(ck)
    assert {(r['variant'],r['seed']) for r in records}=={(v,s) for v in NEW for s in SEEDS}
    assert checked==600 and len(list(ROOT.glob('seed*_finished.json')))==3
    write(ROOT/'selection_lock.json',dict(locked_utc=datetime.now(timezone.utc).isoformat(),records=records,hashes=hashes,selection_mouse_epoch_scores_checked=checked,
        batch_orders_match_both_archived_corners=True,equal_allocated_updates_and_examples=True,new_later_scored=False))


def evaluate():
    verify();assert not (ROOT/'results.json').exists();locked=read(ROOT/'selection_lock.json');rows=[]
    for name,h in locked['hashes'].items():assert digest(ROOT/name)==h
    for s,m in enumerate(MICE):
        meta=read(FAIR/m/'metadata.json');lower=-meta['speed_mean']/meta['speed_std'];indices=np.load(BASE/m/'columns.npy')
        with np.load(FAIR/m/'later_raw.npz') as raw,np.load(FAIR/m/'statistics.npz') as norm:
            seq=((raw['activity'][indices,24:]-norm['activity_mean'][indices])/norm['activity_std'][indices]).T.astype(np.float32)
            y=(raw['speed'][55:]-meta['speed_mean'])/meta['speed_std']
        x=torch.from_numpy(np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(seq,32,axis=0)));predictions={}
        with np.load(SHARED/m/'later_predictions.npz') as saved,np.load(EQUAL/m/'later_predictions.npz') as equal:
            np.testing.assert_array_equal(y,saved['target']);np.testing.assert_array_equal(y,equal['target']);predictions['raw_ridge']=saved['raw_ridge']
            for seed in SEEDS:
                predictions[f'aa_s{seed}']=saved[f'shared_attention_s{seed}'];predictions[f'ms_s{seed}']=saved[f'shared_mlp_s{seed}']
                predictions[f'initial_s{seed}']=saved[f'shared_attention_initial_s{seed}'];predictions[f'equal_mlp_s{seed}']=equal[f'equal_mlp_s{seed}']
        for variant in NEW:
            for seed in SEEDS:
                net=FactorialDecoder(variant,seed,range(4));net.load_state_dict(torch.load(ROOT/f'{variant}_s{seed}'/'selected.pt',weights_only=True));predictions[f'{variant}_s{seed}']=predict(net,x,s)
        scores={}
        for label,p in predictions.items():
            scores[label]=mse(p,y,lower);independent=sum((max(float(a),lower)-float(b))**2 for a,b in zip(p,y))/len(y)
            assert abs(scores[label]-independent)<1e-10*max(1.,independent)
        dest=ROOT/m;dest.mkdir();np.savez_compressed(dest/'later_predictions.npz',target=y,**predictions)
        rows.append(dict(mouse=m,n=len(y),lower=lower,scores=scores));print(m,'scored',flush=True)
    verify();write(ROOT/'results.json',dict(rows=rows));write(ROOT/'audit.json',dict(passed=True,new_shared_fits=6,archived_shared_fits_reused=6,
        updates_per_fit=5688,examples_per_fit=179712,selection_mouse_epoch_scores_checked=600,exact_initial_tensors_and_predictions=True,
        new_model_matches_archived_corners=True,query_and_temporal_choices_independent=True,exact_selected_reload=True,
        matched_all_corner_batches=True,actual_adam_step_counters_verified=True,all_new_choices_locked_before_new_later_scoring=True,
        exact_later_target_alignment=True,independent_later_metrics=True,frozen_sources_inputs_references_and_application_unchanged=True))


if __name__=='__main__':
    with threadpool_limits(limits=2):
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        if sys.argv[1]=='train':train(int(sys.argv[2]))
        else:{'check':check,'freeze':freeze,'lock':lock,'evaluate':evaluate}[sys.argv[1]]()
