"""Leave-one-recording-out pretraining with a fixed small target label budget."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent
REPO = EXP.parent
BASE = EXP / '2026-10-03_dynamics_baseline'
FAIR = EXP / '2026-10-03_fair_comparison'
MODEL = EXP / '2026-10-03_shared_behavior' / 'models.py'
spec = importlib.util.spec_from_file_location('shared_transfer_model', MODEL)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
BehaviorDecoder = module.BehaviorDecoder
MICE = ['MP030', 'MP032', 'MP033', 'MP034']
SEEDS = [10, 11, 12]
FAMILIES = ['attention', 'mlp']
EPOCHS = 24
FIT = (0, 160)
SELECT = (192, 256)
RESET = {'identity', 'session', 'head.3.weight', 'head.3.bias'}
LAMBDAS = [.001, .01, .1, 1., 10.]


def read(p):
    return json.loads(p.read_text())


def write(p, obj, replace=False):
    assert replace or not p.exists(), p
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temp.replace(p)


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def state_digest(state, keys=None):
    h = hashlib.sha256()
    for k in sorted(state if keys is None else keys):
        h.update(k.encode())
        h.update(state[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def mse(p, y, lower):
    p = np.maximum(np.asarray(p, dtype=np.float64), lower)
    assert p.shape == y.shape and np.isfinite(p).all() and np.isfinite(y).all()
    return float(np.mean((p-y)**2))


def predict(net, x, session):
    net.eval()
    with torch.no_grad():
        p = torch.cat([net(x[i:i+64], session) for i in range(0, len(x), 64)]).numpy()
    assert np.isfinite(p).all()
    return p


def batches(lengths, sessions, seed, epoch):
    plans = {}
    for s in sessions:
        order = np.random.default_rng(seed*100000+epoch*100+s).permutation(lengths[s])
        plans[s] = [order[i:i+32] for i in range(0, len(order), 32)]
    cursors = {s: 0 for s in sessions}
    rng = np.random.default_rng(seed*100000+epoch*100+99)
    while True:
        available = [s for s in sessions if cursors[s] < len(plans[s])]
        if not available:
            break
        s = int(rng.choice(available))
        indices = plans[s][cursors[s]]
        cursors[s] += 1
        yield s, indices


def target_model(family, seed, held, source_state=None):
    net = BehaviorDecoder(family, seed, [held])
    fresh = {k: v.clone() for k, v in net.state_dict().items()}
    if source_state is not None:
        state = net.state_dict()
        for key in state:
            if key not in RESET:
                assert state[key].shape == source_state[key].shape
                state[key] = source_state[key].clone()
        net.load_state_dict(state)
        for key, value in net.state_dict().items():
            torch.testing.assert_close(value, fresh[key] if key in RESET else source_state[key], rtol=0, atol=0)
    return net


def check():
    rows = []
    x = torch.randn(4, 128, 32)
    for held in range(4):
        sessions = [s for s in range(4) if s != held]
        for family in FAMILIES:
            source = BehaviorDecoder(family, 10, sessions)
            assert held not in source.sessions
            try:
                source(x, held)
            except ValueError:
                pass
            else:
                raise AssertionError('source accepted the excluded session')
            with torch.no_grad():
                source.patch.weight.add_(.2)
                source.head[3].weight.fill_(.1)
            state = source.state_dict()
            before = state_digest(state)
            transferred = target_model(family, 10, held, state)
            fresh = target_model(family, 10, held)
            np.testing.assert_array_equal(predict(transferred, x, held), np.zeros(4))
            np.testing.assert_array_equal(predict(fresh, x, held), np.zeros(4))
            opt = torch.optim.AdamW(transferred.parameters(), lr=.001)
            for _ in range(2):
                transferred.train()
                opt.zero_grad(set_to_none=True)
                (transferred(x, held)-torch.arange(4.)).square().mean().backward()
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in transferred.parameters())
                opt.step()
            assert transferred.identity.grad.norm() > 0 and transferred.patch.weight.grad.norm() > 0
            assert state_digest(state) == before
            rows.append(dict(held=MICE[held], family=family, source_parameters=sum(p.numel() for p in source.parameters()),
                             target_parameters=sum(p.numel() for p in transferred.parameters())))
    sequence = np.arange(200*128, dtype=np.float64).reshape(200,128)
    windows = np.lib.stride_tricks.sliding_window_view(sequence,32,axis=0)
    recovered = np.concatenate([windows[0,:,:-1], windows[:160,:,-1].T],axis=1)
    np.testing.assert_array_equal(recovered, sequence[:191].T)
    assert SELECT[0]-31 > FIT[1]-1
    write(ROOT/'selfcheck.json', dict(passed=True, rows=rows, heldout_session_rejected=True,
        shared_weights_copy_exact=True, fresh_target_embeddings_and_output_layer=True, equal_initial_output_zero=True,
        source_state_not_mutated=True, body_and_identity_gradients=True, unique_prefix_normalization_bins=191,
        disjoint_fit_selection_contexts=True))


def freeze():
    assert read(ROOT/'selfcheck.json')['passed']
    paths = [Path(__file__), ROOT/'analyse.py', MODEL, BASE/'models.py'] + [REPO/n for n in ['train.py','model.py','data.py']]
    for mouse in MICE:
        paths += [BASE/mouse/f'{stage}_{a}.npy' for stage in ['train','selection'] for a in ['x','y']]
        paths += [BASE/mouse/'columns.npy'] + [FAIR/mouse/n for n in ['statistics.npz','metadata.json','later_raw.npz']]
    write(ROOT/'protocol.json', dict(frozen_utc=datetime.now(timezone.utc).isoformat(), mice=MICE,seeds=SEEDS,
        question='Does source-mouse pretraining improve held-out-mouse speed decoding with224 target labels, and does transformer transfer beat matched MLP transfer and target-only controls?',
        design='4 held-out mice x3 seeds; pretrain each family on the other3 mice only; adapt transferred and freshly initialized versions of both families to the held-out mouse; deterministic target-only ridge plus target mean/median references',
        architecture='exact archived shared BehaviorDecoder,128neurons x32bins; temporal transformer+dynamic behavior query versus temporal MLP+static query; no coordinates or cross-mouse neuron identity matching',
        budget='24 source fits (4folds x3seeds x2families),48 target fits (4folds x3seeds x4arms),20 ridge candidates; source24epochs over all source training windows, target24epochs x5batches=120updates; no seed/architecture/budget expansion',
        target_budget=dict(fit=list(FIT), selection=list(SELECT), labeled_windows=224, gap=32),
        target_normalization='undo cached full-training normalization using archived statistics; compute native activity means/SDs on191 unique bins covered by160 fit windows only; SD<1e-6 becomes1; speed mean/SD from160 fit labels only, SD<1e-6 becomes1; apply frozen prefix normalization to target selection and later raw data',
        panel_limitation='retain the fixed128-cell panel inherited from prior unlabeled full-training eligibility preprocessing; no new target neuron selection. This is a limited behavior-label test, not an untouched-target-data or novel-cohort claim; cache inversion has float32 rounding',
        source_normalization='original per-source training-only normalization; original full source training/selection intervals; held-out mouse absent from source data loader, source sessions, source selection scores and source loss',
        transfer='copy all selected source shared parameters except neuron-ID table,session embedding and final scalar output layer; keep these fresh and identical to target-only control. All target parameters fine-tune; zero initial normalized output for every target arm',
        optimizer='AdamW lr.001 wd.01 batch32 clip1 cosine24 eta_min.0001; equal-mouse source training loss weights totalN/(3*mouseN)*batchN/32; target standard MSE; fresh optimizer for adaptation; same target batches and dropout RNG seed across arms',
        selection='source one joint epoch0..24 by equal-mouse normalized source selection MSE only; target epoch0..24 by64 target selection labels only; ridge penalty .001/.01/.1/1/10 by the same64 labels; lock all choices before any new later inference',
        primary='transfer attention versus target-only attention: >=5% mean within-mouse relative MSE gain, >=3/4 mouse wins, >=8/12 paired seed wins',
        utility='add comparisons with transfer MLP,target-only MLP,and target-only ridge: each>=5% mean gain and>=3/4mouse wins; neural controls>=8/12pairedseedwins; no mouse>25%worse thanridge; >=8/12wins versus owninitial',
        secondary='MAE,R2,all per-mouse/seed scores,MLP transfer effect,training-median reference,leave-one-mouse-out means; same target label/update/selection budgets,not equal total source+target compute',
        uncertainty='2000 paired mouse/seed/circular100-bin bootstrap draws,seed81224; shared seed draw acrossfolds; descriptive97.5% intervals for attention transfer versus target-only attention and transfer MLP; conditional on fitted models,source-overlapping folds and historical method selection are not independent',
        stopping='complete fixed experiment and report failures; no additional transfer recipe,adaptation fraction or architecture search; application unchanged and no publication',
        hypotheses_scope='one fixed224-label budget tests relative usefulness at that budget,not a learning curve or quantified number of labels saved; all animals historically inspected,not independent significance',
        input_hashes={str(p.relative_to(REPO)):digest(p) for p in paths}))
    print('Frozen four-fold transfer study:24 source fits,48 target fits,224 target labels',flush=True)


def verify(prepared=True):
    protocol = read(ROOT/'protocol.json')
    for name,value in protocol['input_hashes'].items():
        assert digest(REPO/name)==value,name
    if prepared:
        for name,value in read(ROOT/'prepared.json')['hashes'].items():
            assert digest(ROOT/name)==value,name
    return protocol


def prepare():
    verify(False)
    hashes = {}
    records = []
    for mouse in MICE:
        out = ROOT/mouse
        out.mkdir()
        cols = np.load(BASE/mouse/'columns.npy')
        meta = read(FAIR/mouse/'metadata.json')
        with np.load(FAIR/mouse/'statistics.npz') as norm:
            old_mean=norm['activity_mean'][cols]
            old_std=norm['activity_std'][cols]
        cached=np.load(BASE/mouse/'train_x.npy',mmap_mode='r')
        target=np.load(BASE/mouse/'train_y.npy',mmap_mode='r')
        raw_fit=cached[FIT[0]:FIT[1]].astype(np.float64)*old_std+old_mean
        unique=np.concatenate([raw_fit[0,:,:-1],raw_fit[:,:,-1].T],axis=1)
        mean=unique.mean(1,keepdims=True)
        std=unique.std(1,keepdims=True)
        constant=int(np.sum(std<1e-6))
        std=np.where(std<1e-6,1.,std)
        raw_y=target[FIT[0]:FIT[1]]*meta['speed_std']+meta['speed_mean']
        ym=float(raw_y.mean())
        ys=float(raw_y.std())
        ys=ys if ys>=1e-6 else 1.
        raw_val=cached[SELECT[0]:SELECT[1]].astype(np.float64)*old_std+old_mean
        yv=(target[SELECT[0]:SELECT[1]]*meta['speed_std']+meta['speed_mean']-ym)/ys
        x=((raw_fit-mean)/std).astype(np.float32)
        xv=((raw_val-mean)/std).astype(np.float32)
        y=(raw_y-ym)/ys
        assert x.shape==(160,128,32) and xv.shape==(64,128,32)
        assert all(np.isfinite(a).all() for a in [x,xv,y,yv])
        np.testing.assert_allclose(((unique-mean)/std).mean(1),0,atol=1e-9)
        np.savez_compressed(out/'normalization.npz',mean=mean,std=std,speed_mean=ym,speed_std=ys,columns=cols)
        np.savez_compressed(out/'target_data.npz',x=x,y=y,xv=xv,yv=yv)
        xf=x.reshape(160,-1).astype(np.float64)
        vf=xv.reshape(64,-1).astype(np.float64)
        center=xf.mean(0)
        z=torch.from_numpy(xf-center)
        target_tensor=torch.from_numpy(y-y.mean())
        gram=z@z.T
        weights=[]
        predictions=[]
        values=[]
        for lam in LAMBDAS:
            matrix=gram+len(y)*lam*torch.eye(len(y),dtype=z.dtype)
            alpha=torch.linalg.solve(matrix,target_tensor)
            assert float((matrix@alpha-target_tensor).norm()/target_tensor.norm().clamp_min(1e-15))<1e-7
            weight=(z.T@alpha).numpy()
            pred=np.einsum('ij,j->i',vf-center,weight,optimize=False)+y.mean()
            independent=(torch.from_numpy(vf-center)@torch.from_numpy(weight)).numpy()+y.mean()
            np.testing.assert_allclose(pred,independent,rtol=1e-9,atol=1e-9)
            weights.append(weight)
            predictions.append(pred)
            values.append(mse(pred,yv,-ym/ys))
        chosen=int(np.argmin(values))
        np.savez_compressed(out/'ridge.npz',weights=np.stack(weights),center=center,mean=float(y.mean()),
                            selection_predictions=np.stack(predictions),target=yv)
        record=dict(mouse=mouse,fit_n=160,selection_n=64,unique_activity_bins=191,prefix_constant_cells=constant,
                    speed_mean=ym,speed_std=ys,lower=-ym/ys,training_median=float(np.median(y)),
                    ridge_index=chosen,ridge_lambda=LAMBDAS[chosen],ridge_selection_mse=values)
        records.append(record)
        for name in ['normalization.npz','target_data.npz','ridge.npz']:
            hashes[str((out/name).relative_to(ROOT))]=digest(out/name)
    write(ROOT/'prepared.json',dict(records=records,hashes=hashes))
    print(json.dumps(records,indent=2),flush=True)


def source_data(held):
    data={}
    for s,mouse in enumerate(MICE):
        if s==held:
            continue
        meta=read(FAIR/mouse/'metadata.json')
        data[s]=dict(x=torch.from_numpy(np.load(BASE/mouse/'train_x.npy')),
                     y=torch.from_numpy(np.load(BASE/mouse/'train_y.npy').astype(np.float32)),
                     xv=torch.from_numpy(np.load(BASE/mouse/'selection_x.npy')),
                     yv=np.load(BASE/mouse/'selection_y.npy'),lower=-meta['speed_mean']/meta['speed_std'])
    assert held not in data and len(data)==3
    return data


def train_fit(net,data,sessions,seed,out,stage,held,family,mode):
    out.mkdir(parents=True)
    lengths={s:len(data[s]['x']) for s in sessions}
    total=sum(lengths.values())
    def selection():
        predictions={s:predict(net,data[s]['xv'],s) for s in sessions}
        scores={s:mse(predictions[s],data[s]['yv'],data[s]['lower']) for s in sessions}
        return predictions,scores,float(np.mean(list(scores.values())))
    first,scores,best=selection()
    assert all(np.array_equal(p,np.zeros_like(p)) for p in first.values())
    history=[dict(epoch=0,score=best,mouse_mse={MICE[s]:v for s,v in scores.items()})]
    bank={s:[first[s]] for s in sessions}
    initial_hash=state_digest(net.state_dict())
    torch.save(net.state_dict(),out/'initial.pt')
    torch.save(net.state_dict(),out/'selected.pt')
    selected=0
    torch.manual_seed(seed+9000)
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,24,eta_min=.0001)
    updates=0
    exposures=0
    start=time.monotonic()
    for epoch in range(1,25):
        net.train()
        order=hashlib.sha256()
        counts={s:0 for s in sessions}
        for s,idx in batches(lengths,sessions,seed,epoch):
            assert (s!=held if stage=='source' else s==held)
            order.update(np.asarray([s],dtype=np.int64).tobytes()+idx.tobytes())
            opt.zero_grad(set_to_none=True)
            loss=(net(data[s]['x'][idx],s)-data[s]['y'][idx]).square().mean()
            loss=loss*(total/(len(sessions)*lengths[s]))*(len(idx)/32)
            assert torch.isfinite(loss)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            counts[s]+=len(idx)
            updates+=1
            exposures+=len(idx)
        assert counts==lengths
        schedule.step()
        pred,scores,score=selection()
        if score<best:
            best=score
            selected=epoch
            torch.save(net.state_dict(),out/'selected.pt')
        for s in sessions:
            bank[s].append(pred[s])
        history.append(dict(epoch=epoch,score=score,mouse_mse={MICE[s]:v for s,v in scores.items()},
                            order_hash=order.hexdigest(),updates=updates,examples=exposures))
        write(out/'history.json',history,replace=True)
        if epoch%6==0:
            print(seed,MICE[held],stage,family,mode,'epoch',epoch,'selected',selected,flush=True)
    steps=sorted({int(v['step'].item()) for v in opt.state.values() if 'step' in v})
    expected=24*sum((n+31)//32 for n in lengths.values())
    assert updates==expected and steps==[updates] and exposures==24*total
    net.load_state_dict(torch.load(out/'selected.pt',weights_only=True))
    saved={}
    for s in sessions:
        np.testing.assert_array_equal(predict(net,data[s]['xv'],s),bank[s][selected])
        saved[MICE[s]+'_predictions']=np.stack(bank[s])
        saved[MICE[s]+'_target']=data[s]['yv']
    np.savez_compressed(out/'selection_predictions.npz',**saved)
    record=dict(stage=stage,held=MICE[held],held_index=held,family=family,mode=mode,seed=seed,sessions=sessions,
                selected_epoch=selected,selection_score=best,updates=updates,examples=exposures,actual_adam_steps=steps,
                epochs=24,initial_hash=initial_hash,selected_reload_exact=True,elapsed_seconds=time.monotonic()-start,
                parameters=sum(p.numel() for p in net.parameters()))
    write(out/'result.json',record)
    return {k:v.clone() for k,v in net.state_dict().items()}


def worker(seed):
    verify()
    assert seed in SEEDS
    prep={r['mouse']:r for r in read(ROOT/'prepared.json')['records']}
    for held,mouse in enumerate(MICE):
        sources=source_data(held)
        with np.load(ROOT/mouse/'target_data.npz') as z:
            target={held:dict(x=torch.from_numpy(z['x'].copy()),y=torch.from_numpy(z['y'].astype(np.float32)),
                              xv=torch.from_numpy(z['xv'].copy()),yv=z['yv'].copy(),lower=prep[mouse]['lower'])}
        for family in FAMILIES:
            source_out=ROOT/mouse/f's{seed}'/('source_'+family)
            source=BehaviorDecoder(family,seed,list(sources))
            selected=train_fit(source,sources,list(sources),seed,source_out,'source',held,family,'source')
            source_hash=state_digest(selected)
            for mode in ['transfer','scratch']:
                net=target_model(family,seed,held,selected if mode=='transfer' else None)
                initial=net.state_dict()
                out=ROOT/mouse/f's{seed}'/(family+'_'+mode)
                shared_keys=[k for k in initial if k not in RESET]
                provenance=dict(source_checkpoint=str((source_out/'selected.pt').relative_to(ROOT)),
                    source_sha256=digest(source_out/'selected.pt'),copied_shared_weights=mode=='transfer',
                    source_shared_hash=state_digest(selected,shared_keys),target_shared_hash=state_digest(initial,shared_keys),
                    reset_hash=state_digest(initial,RESET))
                if mode=='transfer':
                    assert provenance['source_shared_hash']==provenance['target_shared_hash']
                train_fit(net,target,[held],seed,out,'target',held,family,mode)
                write(out/'provenance.json',provenance)
                assert state_digest(selected)==source_hash
        print('COMPLETED FOLD',seed,mouse,flush=True)
    write(ROOT/f'seed{seed}_finished.json',dict(complete=True))


def lock():
    verify()
    records=[]
    hashes={}
    orders={}
    checked=0
    prepared={r['mouse']:r for r in read(ROOT/'prepared.json')['records']}
    for path in sorted(ROOT.glob('MP*/s*/*/result.json')):
        record=read(path)
        out=path.parent
        history=read(out/'history.json')
        assert len(history)==25
        assert record['actual_adam_steps']==[record['updates']]
        if record['stage']=='target':
            assert record['updates']==120 and record['examples']==3840
            assert record['sessions']==[record['held_index']]
            provenance=read(out/'provenance.json')
            assert digest(ROOT/provenance['source_checkpoint'])==provenance['source_sha256']
            source=torch.load(ROOT/provenance['source_checkpoint'],weights_only=True)
            expected=target_model(record['family'],record['seed'],record['held_index'],source if record['mode']=='transfer' else None)
            assert state_digest(expected.state_dict())==record['initial_hash']
        else:
            assert record['held_index'] not in record['sessions'] and len(record['sessions'])==3
        values=[]
        with np.load(out/'selection_predictions.npz') as saved:
            for s in record['sessions']:
                mouse=MICE[s]
                if record['stage']=='source':
                    truth=np.load(BASE/mouse/'selection_y.npy')
                    meta=read(FAIR/mouse/'metadata.json')
                    lower=-meta['speed_mean']/meta['speed_std']
                else:
                    with np.load(ROOT/mouse/'target_data.npz') as z:
                        truth=z['yv'].copy()
                    lower=prepared[mouse]['lower']
                np.testing.assert_array_equal(saved[mouse+'_target'],truth)
                scores=[mse(v,truth,lower) for v in saved[mouse+'_predictions']]
                np.testing.assert_allclose(scores,[h['mouse_mse'][mouse] for h in history],rtol=1e-12,atol=1e-12)
                values.append(scores)
                checked+=25
        joint=np.mean(values,axis=0)
        np.testing.assert_allclose(joint,[h['score'] for h in history],rtol=1e-12,atol=1e-12)
        assert int(np.argmin(joint))==record['selected_epoch']
        key=(record['held'],record['seed'],record['stage'])
        order=[h['order_hash'] for h in history[1:]]
        if key in orders:
            assert orders[key]==order
        orders[key]=order
        records.append(dict(**record,directory=str(out.relative_to(ROOT))))
        for name in ['selected.pt','result.json','selection_predictions.npz','history.json']:
            hashes[str((out/name).relative_to(ROOT))]=digest(out/name)
    assert len(records)==72 and checked==3000
    assert all(read(ROOT/f'seed{s}_finished.json')['complete'] for s in SEEDS)
    write(ROOT/'selection_lock.json',dict(locked_utc=datetime.now(timezone.utc).isoformat(),records=records,
        hashes=hashes,selection_scores_checked=checked,matched_orders=True,later_scored=False))
    print('Locked all72 fits;3000 selection scores and exact transfer provenance checked',flush=True)


def evaluate():
    verify()
    lock=read(ROOT/'selection_lock.json')
    assert not (ROOT/'results.json').exists()
    for name,value in lock['hashes'].items():
        assert digest(ROOT/name)==value,name
    prep={r['mouse']:r for r in read(ROOT/'prepared.json')['records']}
    rows=[]
    checked=0
    for held,mouse in enumerate(MICE):
        with np.load(ROOT/mouse/'normalization.npz') as norm,np.load(FAIR/mouse/'later_raw.npz') as raw:
            seq=((raw['activity'][norm['columns'],24:]-norm['mean'])/norm['std']).T.astype(np.float32)
            y=(raw['speed'][55:]-float(norm['speed_mean']))/float(norm['speed_std'])
            speed_std=float(norm['speed_std'])
        x=np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(seq,32,axis=0))
        assert x.shape==(len(y),128,32) and np.isfinite(x).all() and np.isfinite(y).all()
        xt=torch.from_numpy(x)
        lower=prep[mouse]['lower']
        predictions={'initial':np.zeros(len(y)),'median':np.full(len(y),prep[mouse]['training_median'])}
        for record in lock['records']:
            if record['stage']!='target' or record['held']!=mouse:
                continue
            net=BehaviorDecoder(record['family'],record['seed'],[held])
            net.load_state_dict(torch.load(ROOT/record['directory']/'selected.pt',weights_only=True))
            label=f"{record['family']}_{record['mode']}_s{record['seed']}"
            predictions[label]=predict(net,xt,held)
        with np.load(ROOT/mouse/'ridge.npz') as ridge:
            features=x.reshape(len(x),-1).astype(np.float64)-ridge['center']
            weight=ridge['weights'][prep[mouse]['ridge_index']]
            predictions['ridge']=np.einsum('ij,j->i',features,weight,optimize=False)+float(ridge['mean'])
            independent=(torch.from_numpy(features)@torch.from_numpy(weight)).numpy()+float(ridge['mean'])
            np.testing.assert_allclose(predictions['ridge'],independent,rtol=1e-8,atol=1e-8)
        scores={}
        for name,values in predictions.items():
            bounded=np.maximum(values.astype(np.float64),lower)
            score=mse(values,y,lower)
            manual=sum((max(float(a),lower)-float(b))**2 for a,b in zip(values,y))/len(y)
            assert abs(manual-score)<1e-12*max(1,score)
            scores[name]=dict(mse=score,mae=float(np.mean(np.abs(bounded-y))),
                              r2=1-score/float(np.var(y)),native_mse=score*speed_std**2)
            checked+=1
        np.savez_compressed(ROOT/mouse/'later_predictions.npz',target=y,**predictions)
        rows.append(dict(mouse=mouse,n=len(y),lower=lower,scores=scores))
        print(mouse,'later scoring complete',flush=True)
    verify()
    write(ROOT/'results.json',dict(rows=rows))
    write(ROOT/'audit.json',dict(passed=True,source_fits=24,target_fits=48,ridge_candidates=20,
        source_excludes_heldout_mouse=True,source_selection_excludes_heldout_mouse=True,
        target_labels_fit=160,target_labels_selection=64,target_normalization_fit_prefix_only=True,
        fresh_target_embeddings_and_final_output=True,source_shared_weight_copy_exact=True,
        matched_target_updates=120,matched_target_examples=3840,matched_batch_orders=True,
        selection_scores_checked=3000,initial_predictions_zero=True,selected_reload_exact=True,
        actual_adam_steps_checked=True,all_choices_locked_before_later_scoring=True,
        scalar_later_mse_checks=checked,frozen_input_source_application_hashes_unchanged=True))
    write(ROOT/'environment.json',dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,platform=platform.platform()))


if __name__=='__main__':
    with threadpool_limits(limits=2):
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        if sys.argv[1:] == ['check']:
            check()
        elif sys.argv[1:] == ['freeze']:
            freeze()
        elif sys.argv[1:] == ['prepare']:
            prepare()
        elif len(sys.argv)==3 and sys.argv[1]=='worker':
            worker(int(sys.argv[2]))
        elif sys.argv[1:]==['lock']:
            lock()
        elif sys.argv[1:]==['evaluate']:
            evaluate()
        else:
            raise SystemExit('usage: run.py check|freeze|prepare|worker SEED|lock|evaluate')
