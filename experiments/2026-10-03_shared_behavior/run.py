"""Frozen factorial study of shared behavior-query attention and static-pooling MLPs."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from models import BehaviorDecoder

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
BASE=ROOT.parent/'2026-10-03_dynamics_baseline'
FLAT=ROOT.parent/'2026-10-03_neuron_readout'
FAIR=ROOT.parent/'2026-10-03_fair_comparison'
MICE=['MP030','MP032','MP033','MP034']
SEEDS=[10,11,12]
KINDS=['attention','mlp']


def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');temp.replace(p)
def mse(p,y,lower):
    p=np.asarray(p,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    assert p.shape==y.shape and np.isfinite(p).all()
    return float(np.mean((np.maximum(p,lower)-y)**2))
def predict(net,x,session,uniform=False):
    net.eval()
    with torch.no_grad():p=torch.cat([net(x[i:i+64],session,uniform) for i in range(0,len(x),64)]).numpy()
    assert np.isfinite(p).all();return p


def batches(lengths,sessions,seed,epoch):
    plans={}
    for session in sessions:
        order=np.random.default_rng(seed*100000+epoch*100+session).permutation(lengths[session])
        plans[session]=[order[i:i+32] for i in range(0,len(order),32)]
    cursors={s:0 for s in sessions};rng=np.random.default_rng(seed*100000+epoch*100+99)
    while True:
        available=[s for s in sessions if cursors[s]<len(plans[s])]
        if not available:break
        s=int(rng.choice(available));indices=plans[s][cursors[s]];cursors[s]+=1
        yield s,indices


def check():
    counts=[];x=torch.randn(3,128,32);changed=x+torch.randn_like(x)
    for kind in KINDS:
        for sessions in [[0],[0,1,2,3]]:
            net=BehaviorDecoder(kind,10,sessions).eval()
            assert torch.equal(net(x,0),torch.zeros(3))
            with torch.no_grad():
                _,a=net(x,0,return_weights=True);_,b=net(changed,0,return_weights=True)
                if kind=='mlp':torch.testing.assert_close(a,b,rtol=0,atol=0)
                else:assert not torch.allclose(a,b,rtol=1e-5,atol=1e-7)
            opt=torch.optim.AdamW(net.parameters(),lr=.001)
            for _ in range(2):
                opt.zero_grad(set_to_none=True);loss=(net(x,0)-torch.arange(3).float()).square().mean();loss.backward()
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters());opt.step()
            assert net.key_projection.weight.grad.norm()>0
            key='temporal.mix.in_proj_weight' if kind=='attention' else 'temporal.weight'
            assert dict(net.named_parameters())[key].grad.norm()>0
            clone=BehaviorDecoder(kind,10,sessions);clone.load_state_dict(net.state_dict());np.testing.assert_array_equal(predict(net,x,0),predict(clone,x,0))
            counts.append(dict(kind=kind,sessions=len(sessions),parameters=sum(p.numel() for p in net.parameters())))
        shared=BehaviorDecoder(kind,10,[0,1,2,3]).eval()
        with torch.no_grad():shared.head[-1].weight.fill_(.01)
        for session in range(4):
            separate=BehaviorDecoder(kind,10,[session]).eval()
            with torch.no_grad():
                separate.head[-1].weight.fill_(.01);torch.testing.assert_close(shared(x,session),separate(x,session),rtol=0,atol=0)
    for n in [1,4]:
        a,b=[v['parameters'] for v in counts if v['sessions']==n];assert abs(a/b-1)<.01
    a,b=BehaviorDecoder('attention',10,[0,1,2,3]),BehaviorDecoder('mlp',10,[0,1,2,3])
    for k,v in a.state_dict().items():
        if not k.startswith('temporal.'):torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
    lengths=[61,84,39,72]
    shared=list(batches(lengths,[0,1,2,3],10,1))
    for s in range(4):
        sa=np.concatenate([idx for session,idx in shared if session==s]);sb=np.concatenate([idx for _,idx in batches(lengths,[s],10,1)])
        np.testing.assert_array_equal(sa,sb);np.testing.assert_array_equal(np.sort(sa),np.arange(lengths[s]))
    write(ROOT/'selfcheck.json',dict(passed=True,models=counts,shared_separate_initial_predictions_exact=True,common_non_temporal_initialization_exact=True,static_weights_input_invariant=True,attention_weights_input_dependent=True,body_and_routing_gradients=True,reload_exact=True,per_mouse_batch_order_shared_separate_exact=True,each_training_example_once_per_epoch=True))


def freeze():
    assert not (ROOT/'protocol.json').exists() and read(ROOT/'selfcheck.json')['passed']
    files=[ROOT/'models.py',ROOT/'run.py',BASE/'models.py',BASE/'protocol.json',FLAT/'protocol.json',FLAT/'selection_lock.json']
    denominators={}
    for m in MICE:
        files += [BASE/m/f'{s}_{a}.npy' for s in ['train','selection'] for a in ['x','y']]
        files += [BASE/m/'columns.npy',BASE/m/'later_predictions.npz',FLAT/m/'later_predictions.npz',FLAT/m/'selection.json',FLAT/m/'raw_ridge.npz']
        files += [FAIR/m/n for n in ['later_raw.npz','statistics.npz','metadata.json']]
        record=next(v for v in read(FLAT/m/'selection.json')['records'] if v['label']=='raw_ridge')
        with np.load(FLAT/m/'raw_ridge.npz') as saved:
            np.testing.assert_array_equal(saved['target'],np.load(BASE/m/'selection_y.npy'))
            meta=read(FAIR/m/'metadata.json');score=mse(saved['selection_predictions'][record['selected_index']],saved['target'],-meta['speed_mean']/meta['speed_std'])
            assert abs(score-record['mse'])<1e-12
        denominators[m]=score
    files += [PROJECT/n for n in ['model.py','data.py','train.py']]
    write(ROOT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        question='Does sharing direct behavioral supervision across recordings help a neuron-time behavior-query transformer more than a matched static-pooling MLP?',
        endpoint='Concurrent running speed only. Same128neuron,32bin contexts and existing chronological train/selection/later splits,training-only normalization and physical-zero lower bound. No neural reconstruction loss,pretraining,coordinates,newdata or old evaluation tails.',
        design='2x2:shared versus separate training,and attention versus MLP/static pooling. Four known recordings,three seeds10/11/12. Shared models keep separate neuron-ID and session embeddings for each recording;all feature and output weights shared. Separate models have one session-specific embedding table and their own feature/output weights. No neuron correspondence across recordings.',
        architecture='Continuous four-bin patches,width16,eight time patches per neuron. Attention uses a causal per-neuron temporal transformer;the learned running-speed query cross-attends to all128x8neuronal-time tokens. No neuron mean before this query. MLP control uses causal static temporal mixing and identical projections with routing keys derived solely from learned identity/time/session embeddings,not activity. Values depend on activity. Both use the same query residual MLP and speed head over16query features plus64population mean/std history features. Shared/nonshared initialization matches for each session;family counts differ<1%. This is a matched static-routing nonlinear control,not a claim to exhaust every MLP.',
        budget='30new model fits:6shared (2families x3seeds) and24separate (2families x4mice x3seeds),24epochs each. Each shared epoch traverses all four training sets;each separate epoch traverses its one set. Every example is used once per epoch. Aggregate exposures/steps match the four separate models. Shared parameters receive updates from more records,so this is not equal updates per model;the matched MLP receives the same pooling opportunity.',
        batches='Fixed per-mouse shuffled batches,interleaved for shared training while preserving within-mouse order. Orders match both families and shared/separate regimes. AdamW.001,wd.01,batch32,clip1,cosine24 eta_min.0001. Shared per-batch loss weight totalN/(4*mouseN)*batchN/32 targets equal-mouse risk;separate uses batchN/32. No selection-data weights enter gradient updates.',
        selection='Shared:one epoch0..24 checkpoint per family/seed,chosen by equal-mouse mean earlier bounded MSE divided by previously selected raw-ridge earlier MSE. Separate:own earlier bounded MSE per mouse. Same weighting choice for both families. All30choices locked before any current later scoring;no per-mouse selection from shared epochs.',
        selection_denominators=denominators,
        primary='Shared attention versus separate attention,shared MLP,and stronger raw ridge. Each requires>=5%equal-weight mean within-mouse gain and>=3/4mouse wins;neural comparisons additionally>=8/12paired seed wins. No mouse>25%worse than raw ridge;>=8/12later wins over own initial predictions.',
        broader_gate='Additionally beat separate MLP,prior pretrained attention,and prior normalized pooled MLP by>=5%mean gain,>=3/4mice and>=8/12seeds each;no mouse>25%worse than prior MLP. Do not claim overall model superiority from the primary gate alone.',
        secondary='Sharing effect within MLP and difference in relative sharing effects between families;selected shared-attention query replaced by uniform pooling at inference;initial predictions;all individual seed scores. Uniform ablation measures dependence/coadaptation,not a separately trained architecture comparison.',
        uncertainty='2000paired hierarchical mouse/seed/circular100-bin bootstrap draws,seed81212,descriptive98.3333%intervals for three primary contrasts. Shared models couple mice;these intervals are conditional on fitted models and omit retraining uncertainty.',
        scope='Supervised pooling across known recordings with later-period evaluation,not unseen-animal transfer. All four animals historically inspected;no independent significance or novel architecture claim. The earlier shared screen was a different limited pooled8-bin prototype;no prior result is overwritten. Fixed batch,no grid expansion or publication.',
        hashes={str(p.relative_to(PROJECT)):digest(p) for p in files}))


def verify():
    p=read(ROOT/'protocol.json')
    for name,h in p['hashes'].items():assert digest(PROJECT/name)==h,name
    return p


def load_data():
    data={}
    for session,m in enumerate(MICE):
        meta=read(FAIR/m/'metadata.json')
        data[session]=dict(x=torch.from_numpy(np.load(BASE/m/'train_x.npy')),y=torch.from_numpy(np.load(BASE/m/'train_y.npy').astype(np.float32)),xv=torch.from_numpy(np.load(BASE/m/'selection_x.npy')),yv=np.load(BASE/m/'selection_y.npy'),lower=-meta['speed_mean']/meta['speed_std'])
    return data


def train(seed):
    protocol=verify();data=load_data();lengths=[len(data[s]['x']) for s in range(4)]
    for group in ['shared']+MICE:
        sessions=list(range(4)) if group=='shared' else [MICE.index(group)]
        dest=ROOT/group;dest.mkdir(exist_ok=True)
        for kind in KINDS:
            out=dest/f'{kind}_s{seed}';out.mkdir();net=BehaviorDecoder(kind,seed,sessions)
            def evaluate_selection():
                pred={s:predict(net,data[s]['xv'],s) for s in sessions}
                scores={s:mse(pred[s],data[s]['yv'],data[s]['lower']) for s in sessions}
                selection=float(np.mean([scores[s]/protocol['selection_denominators'][MICE[s]] for s in sessions])) if group=='shared' else scores[sessions[0]]
                return pred,scores,selection
            first,scores,best=evaluate_selection();predictions={s:[first[s]] for s in sessions};history=[dict(epoch=0,selection_score=best,mouse_mse={MICE[s]:v for s,v in scores.items()})]
            torch.save(net.state_dict(),out/'initial.pt');torch.save(net.state_dict(),out/'selected.pt');selected=0
            torch.manual_seed(seed+9000);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,24,eta_min=.0001)
            grad_routing=0.;grad_temporal=0.;start=time.monotonic();totalN=sum(lengths[s] for s in sessions)
            for epoch in range(1,25):
                net.train();orders={s:hashlib.sha256() for s in sessions};global_order=hashlib.sha256();totals={s:0. for s in sessions}
                for s,indices in batches(lengths,sessions,seed,epoch):
                    orders[s].update(indices.tobytes());global_order.update(np.asarray([s],dtype=np.int64).tobytes()+indices.tobytes())
                    xb,yb=data[s]['x'][indices],data[s]['y'][indices];opt.zero_grad(set_to_none=True)
                    raw=(net(xb,s)-yb).square().mean();loss=raw*(totalN/(len(sessions)*lengths[s]))*(len(indices)/32)
                    assert torch.isfinite(loss);loss.backward();grad_routing=max(grad_routing,float(net.key_projection.weight.grad.norm()))
                    key='temporal.mix.in_proj_weight' if kind=='attention' else 'temporal.weight';grad_temporal=max(grad_temporal,float(dict(net.named_parameters())[key].grad.norm()))
                    nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step();totals[s]+=float(raw.detach())*len(indices)
                schedule.step();pred,scores,score=evaluate_selection()
                if score<best:best=score;selected=epoch;torch.save(net.state_dict(),out/'selected.pt')
                for s in sessions:predictions[s].append(pred[s])
                history.append(dict(epoch=epoch,selection_score=score,mouse_mse={MICE[s]:v for s,v in scores.items()},training_mse={MICE[s]:totals[s]/lengths[s] for s in sessions},batch_order_hashes={MICE[s]:v.hexdigest() for s,v in orders.items()},global_order_hash=global_order.hexdigest()));write(out/'history.json',history)
            assert grad_routing>0 and grad_temporal>0;net.load_state_dict(torch.load(out/'selected.pt',weights_only=True))
            saved={}
            for s in sessions:
                np.testing.assert_array_equal(predict(net,data[s]['xv'],s),predictions[s][selected]);saved[MICE[s]+'_predictions']=np.stack(predictions[s]);saved[MICE[s]+'_target']=data[s]['yv']
            np.savez_compressed(out/'selection_predictions.npz',**saved)
            write(out/'result.json',dict(group=group,kind=kind,seed=seed,sessions=sessions,epochs=24,selected_epoch=selected,selection_score=best,routing_grad_max=grad_routing,temporal_grad_max=grad_temporal,selected_reload_exact=True,elapsed_seconds=time.monotonic()-start));print(group,kind,seed,'complete',flush=True)
    write(ROOT/f'seed{seed}_finished.json',dict(complete=True))


def lock():
    protocol=verify();assert not (ROOT/'selection_lock.json').exists();records=[];hashes={};orders={};globals={};checked=0
    for path in sorted(ROOT.glob('*/*/result.json')):
        r=read(path);history=read(path.parent/'history.json');assert len(history)==25
        metrics={}
        with np.load(path.parent/'selection_predictions.npz') as saved:
            for s in r['sessions']:
                m=MICE[s];meta=read(FAIR/m/'metadata.json');lower=-meta['speed_mean']/meta['speed_std']
                np.testing.assert_array_equal(saved[m+'_target'],np.load(BASE/m/'selection_y.npy'))
                metrics[m]=[mse(p,saved[m+'_target'],lower) for p in saved[m+'_predictions']];checked+=25
                np.testing.assert_allclose(metrics[m],[h['mouse_mse'][m] for h in history],rtol=1e-12,atol=1e-12)
                key=(s,r['seed']);order=[h['batch_order_hashes'][m] for h in history[1:]]
                if key in orders:assert orders[key]==order
                orders[key]=order
            score=np.mean([np.array(v)/protocol['selection_denominators'][m] for m,v in metrics.items()],axis=0) if r['group']=='shared' else next(iter(metrics.values()))
            np.testing.assert_allclose(score,[h['selection_score'] for h in history],rtol=1e-12,atol=1e-12);assert int(np.argmin(score))==r['selected_epoch']
        key=(r['group'],r['seed']);order=[h['global_order_hash'] for h in history[1:]]
        if key in globals:assert globals[key]==order
        globals[key]=order;records.append(r);ck=path.parent/'selected.pt';hashes[str(ck.relative_to(ROOT))]=digest(ck)
    assert len(records)==30 and checked==1200 and len(list(ROOT.glob('seed*_finished.json')))==3
    write(ROOT/'selection_lock.json',dict(locked_utc=datetime.now(timezone.utc).isoformat(),records=records,hashes=hashes,selection_mouse_epoch_scores_checked=checked,matched_per_mouse_and_family_orders=True,later_scored=False))


def evaluate():
    verify();assert not (ROOT/'results.json').exists();locked=read(ROOT/'selection_lock.json')
    for name,h in locked['hashes'].items():assert digest(ROOT/name)==h
    rows=[]
    for session,m in enumerate(MICE):
        meta=read(FAIR/m/'metadata.json');lower=-meta['speed_mean']/meta['speed_std'];indices=np.load(BASE/m/'columns.npy')
        with np.load(FAIR/m/'later_raw.npz') as raw,np.load(FAIR/m/'statistics.npz') as norm:
            seq=((raw['activity'][indices,24:]-norm['activity_mean'][indices])/norm['activity_std'][indices]).T.astype(np.float32)
            y=(raw['speed'][55:]-meta['speed_mean'])/meta['speed_std']
        x=torch.from_numpy(np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(seq,32,axis=0)));predictions={};scores={}
        for group in ['shared',m]:
            regime='shared' if group=='shared' else 'separate';sessions=list(range(4)) if regime=='shared' else [session]
            for kind in KINDS:
                for seed in SEEDS:
                    net=BehaviorDecoder(kind,seed,sessions);out=ROOT/group/f'{kind}_s{seed}';net.load_state_dict(torch.load(out/'selected.pt',weights_only=True))
                    predictions[f'{regime}_{kind}_s{seed}']=predict(net,x,session)
                    if regime=='shared' and kind=='attention':predictions[f'shared_attention_uniform_s{seed}']=predict(net,x,session,True)
                    net.load_state_dict(torch.load(out/'initial.pt',weights_only=True));predictions[f'{regime}_{kind}_initial_s{seed}']=predict(net,x,session)
        with np.load(FLAT/m/'later_predictions.npz') as saved:
            np.testing.assert_array_equal(y,saved['target']);predictions['raw_ridge']=saved['raw_ridge']
            for seed in SEEDS:
                predictions[f'prior_attention_s{seed}']=saved[f'reference_attention_pretrained_s{seed}'];predictions[f'prior_mlp_s{seed}']=saved[f'reference_pooled_mlp_s{seed}']
        predictions['zero']=np.full(len(y),lower)
        for label,pred in predictions.items():
            scores[label]=mse(pred,y,lower);independent=sum((max(float(a),lower)-float(b))**2 for a,b in zip(pred,y))/len(y)
            assert abs(scores[label]-independent)<1e-10*max(1.,independent)
        np.savez_compressed(ROOT/m/'later_predictions.npz',target=y,**predictions);rows.append(dict(mouse=m,n=len(y),lower=lower,scores=scores));print(m,'scored',flush=True)
    verify();write(ROOT/'results.json',dict(rows=rows));write(ROOT/'audit.json',dict(passed=True,new_fits=30,shared_fits=6,separate_fits=24,epochs_per_fit=24,per_mouse_epoch_scores_checked=1200,matched_per_mouse_and_family_batches=True,each_window_once_per_epoch=True,selected_reload_exact=True,single_shared_checkpoint_per_seed=True,initialization_and_routing_checks_passed=True,all_choices_locked_before_current_later_scoring=True,exact_target_alignment=True,independent_later_metrics=True,frozen_hashes_unchanged=True))


if __name__=='__main__':
    with threadpool_limits(limits=2):
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        if sys.argv[1]=='train':train(int(sys.argv[2]))
        else:{'check':check,'freeze':freeze,'lock':lock,'evaluate':evaluate}[sys.argv[1]]()
