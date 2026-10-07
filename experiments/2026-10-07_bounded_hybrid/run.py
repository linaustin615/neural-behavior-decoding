"""Validation-selected two-model base with bounded, movement-supervised correction."""
import argparse
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'2026-10-06_frozen_retrieval'))
import run as prior
from decoding.config import Recipe
from decoding.model import PopulationDecoder

SEEDS = (601,602,603)
ARMS = ('attention_bce','mlp_bce','attention_mse')
ALPHAS = (0.,.25,.5,.75,1.)
EPOCHS = 24


class Residual(nn.Module):
    def __init__(self,arm,seed,mouse,prevalence):
        super().__init__()
        family = 'mlp' if arm=='mlp_bce' else 'attention'
        self.encoder = PopulationDecoder(Recipe(family=family,width=16,depth=1,patch=4,dropout=0.),seed,mouse)
        self.encoder.head = nn.Identity()
        torch.manual_seed(seed+5000)
        self.head = nn.Sequential(nn.LayerNorm(81),nn.Linear(81,64),nn.GELU(),nn.Linear(64,2))
        nn.init.zeros_(self.head[-1].weight)
        with torch.no_grad():
            self.head[-1].bias.copy_(torch.tensor([0.,math.log(prevalence/(1-prevalence))]))

    def forward(self,x,base):
        tokens = self.encoder.encode(x)
        features = torch.cat([tokens[:,-1],x.mean(1),x.std(1,unbiased=False),base[:,None]],dim=1)
        raw,logit = self.head(features).unbind(1)
        delta = .5*torch.tanh(raw)*(1-torch.sigmoid(logit))
        return delta,logit


def model(mouse,arm,seed):
    d = prior.examples(mouse,'train')
    p = float(np.clip(np.mean(d.y[d.indices]-d.metadata['lower']>.05),1e-4,1-1e-4))
    return Residual(arm,seed,prior.MICE.index(mouse),p)


def check():
    for path,value in prior.read(ROOT/'protocol.json')['source_sha256'].items():
        assert prior.digest(prior.REPO/path)==value,path


def prepare():
    prior.check_lock()
    assert not (ROOT/'protocol.json').exists()
    sources = [Path(__file__),Path(prior.__file__),prior.REPO/'decoding/model.py',prior.REPO/'decoding/data.py',prior.REPO/'decoding/config.py']
    prior.save(ROOT/'protocol.json',dict(utc=prior.utc(),mice=prior.MICE,seeds=SEEDS,arms=ARMS,
        fits=63,epochs=24,lr=.001,weight_decay=.001,batch=64,alpha_grid=ALPHAS,
        primary_goal='combined decoder beats BOTH original transformer and MLP; isolated attention advantage is secondary',
        base='clip original transformer/MLP speed at physical zero in float64; validation-select alpha per mouse across three seeds for (1-alpha)*MLP+alpha*transformer',
        mapping='new601/602/603 paired with original401/402/403',
        correction='trainable raw512x32 activity encoder width16 one block patch4; concatenate last token, population mean/std and base speed; head emits residual and movement logit; delta=.5*tanh(residual)*(1-sigmoid(logit)); final=max(base+delta,0)',
        initialization='zero residual gives exactly the base prediction at epoch0; class-logit bias from training moving prevalence',
        loss='speed MSE + BCE(weight1) for *_bce + .1*mean(1[speed>=.5]*delta^2); BCE labels speed>.05; identical protection penalty in all arms',
        selection='minimum full validation speed MSE including epoch0; fixed24epochs AdamW cosine to.0001 clipgrad1; paired batches; no test labels in training or inference',
        barrier='lock baseline alphas before fits; lock all63 selected checkpoints before scoring',
        gate='for each arm versus each original parent: >=5%mean relative MSE gain,>=6/7mouse wins,>=14/21seed wins,maxmouseharm<=10%,meanMAEgain>=0; also >=5%gain vs blend, quiet wins>=5/7 against blend and no mouse activeMSE more than5%worse than blend',
        active='observed speed>=.5 trainingSD; quiet<=.05; missing active/quiet slice not a win; no test-based threshold changes',
        limits='exploratory reused animals; no independent significance; full parent weights frozen; not exhaustive; no automatic extensions',
        source_sha256={str(p.relative_to(prior.REPO)):prior.digest(p) for p in sources}))
    cache = ROOT/'cache'
    cache.mkdir(exist_ok=False)
    choices = {}
    for mouse in prior.MICE:
        scores = []
        records = {}
        for seed in SEEDS:
            arrays = {}
            for family in ('transformer','mlp'):
                net = prior.network(mouse,family,seed-200)
                for split in ('train','validation'):
                    d = prior.examples(mouse,split)
                    _,p = prior.features(net,d)
                    arrays[family+'_'+split] = np.maximum(p.astype(float),d.metadata['lower'])-d.metadata['lower']
            y = d.y[d.indices]-d.metadata['lower']
            scores.append([float(np.mean(((1-a)*arrays['mlp_validation']+a*arrays['transformer_validation']-y)**2)) for a in ALPHAS])
            records[seed] = arrays
        alpha = ALPHAS[int(np.argmin(np.mean(scores,axis=0)))]
        choices[mouse] = dict(alpha=alpha,validation_grid=scores)
        for seed,arrays in records.items():
            for split in ('train','validation'):
                arrays['base_'+split] = (1-alpha)*arrays['mlp_'+split]+alpha*arrays['transformer_'+split]
            np.savez(cache/f'{mouse}_{seed}.npz',**arrays)
        print(mouse,'base selected',alpha,flush=True)
    prior.save(ROOT/'base_lock.json',dict(utc=prior.utc(),choices=choices,protocol_sha256=prior.digest(ROOT/'protocol.json'),
        cache_sha256={p.name:prior.digest(p) for p in cache.glob('*.npz')}))


def predict(net,data,base):
    net.eval()
    delta,logits = [],[]
    with torch.inference_mode():
        for start in range(0,len(data.indices),128):
            x = torch.from_numpy(np.array(data.x[data.indices[start:start+128]],copy=True))
            b = torch.tensor(base[start:start+128],dtype=torch.float32)
            d,l = net(x,b)
            delta.append(d.numpy()); logits.append(l.numpy())
    delta,logits = np.concatenate(delta),np.concatenate(logits)
    assert np.isfinite(delta).all() and np.abs(delta).max()<=.500001
    return np.maximum(base+delta.astype(float),0),delta,logits


def fit(mouse,arm,seed):
    check()
    out = ROOT/'fits'/f'{mouse}_{arm}_{seed}'
    if (out/'result.json').exists():
        assert prior.digest(out/'selected.pt')==prior.read(out/'result.json')['sha256']
        return
    out.mkdir(parents=True,exist_ok=False)
    path = ROOT/'cache'/f'{mouse}_{seed}.npz'
    assert prior.digest(path)==prior.read(ROOT/'base_lock.json')['cache_sha256'][path.name]
    with np.load(path) as c: tb,vb = c['base_train'],c['base_validation']
    train,val = prior.examples(mouse,'train'),prior.examples(mouse,'validation')
    lower = train.metadata['lower']
    target = val.y[val.indices]-lower
    net = model(mouse,arm,seed)
    initial = predict(net,val,vb)[0]
    np.testing.assert_array_equal(initial,vb)
    best = float(np.mean((initial-target)**2)); chosen = 0
    torch.save(net.state_dict(),out/'initial.pt'); torch.save(net.state_dict(),out/'selected.pt')
    history = [dict(epoch=0,validation_mse=best)]
    predictions = [initial]
    opt = torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS,eta_min=.0001)
    started = time.monotonic()
    for epoch in range(1,EPOCHS+1):
        net.train()
        order = np.random.default_rng(seed*100000+epoch*100+prior.MICE.index(mouse)).permutation(len(train.indices))
        sums = np.zeros(3)
        for start in range(0,len(order),64):
            local = order[start:start+64]; idx = train.indices[local]
            x = torch.from_numpy(np.array(train.x[idx],copy=True))
            b = torch.tensor(tb[local],dtype=torch.float32)
            y = torch.tensor(train.y[idx]-lower,dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            delta,logits = net(x,b)
            mse = ((b+delta).clamp_min(0)-y).square().mean()
            bce = F.binary_cross_entropy_with_logits(logits,(y>.05).float())
            protection = ((y>=.5).float()*delta.square()).mean()
            loss = (mse+(bce if arm.endswith('bce') else 0)+.1*protection)*(len(idx)/64)
            assert torch.isfinite(loss)
            loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True); opt.step()
            sums += np.array([float(mse.detach()),float(bce.detach()),float(protection.detach())])*len(idx)
        sched.step()
        p = predict(net,val,vb)[0]; score = float(np.mean((p-target)**2))
        assert np.isfinite(score)
        if score<best:
            best,chosen = score,epoch
            torch.save(net.state_dict(),out/'selected.pt')
        predictions.append(p)
        history.append(dict(epoch=epoch,validation_mse=score,training_mse=sums[0]/len(order),training_bce=sums[1]/len(order),active_penalty=sums[2]/len(order)))
        prior.save(out/'history.json',history)
    net.load_state_dict(torch.load(out/'selected.pt',weights_only=True))
    np.testing.assert_array_equal(predict(net,val,vb)[0],predictions[chosen])
    np.savez_compressed(out/'validation.npz',prediction=np.stack(predictions),target=target)
    prior.save(out/'result.json',dict(mouse=mouse,arm=arm,seed=seed,selected_epoch=chosen,validation_mse=best,
        sha256=prior.digest(out/'selected.pt'),parameters=sum(p.numel() for p in net.parameters()),
        epochs=EPOCHS,seconds=time.monotonic()-started,selected_reload_exact=True,test_scored=False))
    print(mouse,arm,seed,'complete',chosen,flush=True)


def lock():
    check()
    paths = list((ROOT/'fits').glob('*/selected.pt'))
    assert len(paths)==63 and len(list((ROOT/'fits').glob('*/result.json')))==63
    prior.save(ROOT/'evaluation_lock.json',dict(utc=prior.utc(),protocol_sha256=prior.digest(ROOT/'protocol.json'),
        base_lock_sha256=prior.digest(ROOT/'base_lock.json'),sha256={str(p.relative_to(ROOT)):prior.digest(p) for p in paths}))


def evaluate():
    check()
    locked = prior.read(ROOT/'evaluation_lock.json')
    assert locked['protocol_sha256']==prior.digest(ROOT/'protocol.json')
    assert locked['base_lock_sha256']==prior.digest(ROOT/'base_lock.json')
    for path,h in locked['sha256'].items(): assert prior.digest(ROOT/path)==h
    out = ROOT/'predictions'; out.mkdir(exist_ok=False)
    choices = prior.read(ROOT/'base_lock.json')['choices']
    rows = []
    for mouse in prior.MICE:
        data = prior.examples(mouse,'test'); lower = data.metadata['lower']; y = data.y[data.indices]
        for seed in SEEDS:
            parents = {}
            for family in ('transformer','mlp'):
                with np.load(prior.ROOT/'predictions'/f'{mouse}_{family}_{seed-200}.npz') as saved:
                    np.testing.assert_array_equal(y,saved['target'])
                    parents[family] = np.maximum(saved['parent'].astype(float),lower)-lower
                    rows.append(dict(mouse=mouse,seed=seed,arm=family,**prior.metrics(saved['parent'],y,lower)))
            a = choices[mouse]['alpha']; base = (1-a)*parents['mlp']+a*parents['transformer']
            rows.append(dict(mouse=mouse,seed=seed,arm='blend',**prior.metrics(base+lower,y,lower)))
            for arm in ARMS:
                net = model(mouse,arm,seed)
                net.load_state_dict(torch.load(ROOT/'fits'/f'{mouse}_{arm}_{seed}'/'selected.pt',weights_only=True))
                p,delta,logits = predict(net,data,base)
                row = dict(mouse=mouse,seed=seed,arm=arm,**prior.metrics(p+lower,y,lower))
                rows.append(row)
                np.savez_compressed(out/f'{mouse}_{arm}_{seed}.npz',prediction=p+lower,target=y,base=base,delta=delta,logits=logits)
        for arm,value in [('zero',lower),('mean',0.)]:
            rows.append(dict(mouse=mouse,seed=None,arm=arm,**prior.metrics(np.full(len(y),value),y,lower)))
        print(mouse,'scored',flush=True)
    pairs = [(a,b) for a in ARMS for b in ('transformer','mlp','blend','zero')]
    pairs += [('attention_bce','mlp_bce'),('attention_bce','attention_mse')]
    contrasts = {a+'_vs_'+b:prior.contrast(rows,a,b) for a,b in pairs}
    gates = {}
    for arm in ARMS:
        checks = {}
        for parent in ('transformer','mlp'):
            c = contrasts[arm+'_vs_'+parent]
            seed_wins = sum(next(r['mse'] for r in rows if r['mouse']==m and r['seed']==s and r['arm']==arm)<next(r['mse'] for r in rows if r['mouse']==m and r['seed']==s and r['arm']==parent) for m in prior.MICE for s in SEEDS)
            checks[parent] = dict(mean_gain=c['mean_gain']>=.05,mouse_wins=c['mouse_wins']>=6,seed_wins=seed_wins>=14,harm=c['max_harm']<=.1,mae=c['mean_mae_gain']>=0)
            c['seed_wins'] = seed_wins
        c = contrasts[arm+'_vs_blend']
        active_harms = {}
        for mouse in prior.MICE:
            av = [r['active_mse'] for r in rows if r['mouse']==mouse and r['arm']==arm]
            bv = [r['active_mse'] for r in rows if r['mouse']==mouse and r['arm']=='blend']
            active_harms[mouse] = float(np.mean(av)/np.mean(bv)-1) if all(v is not None for v in av+bv) and np.mean(bv)>0 else None
        checks['blend'] = dict(mean_gain=c['mean_gain']>=.05,quiet=c['quiet_wins']>=5,
            active_protection=all(v is not None and v<=.05 for v in active_harms.values()))
        gates[arm] = dict(passed=all(v for group in checks.values() for v in group.values()),checks=checks,active_harm=active_harms)
    prior.save(ROOT/'summary.json',dict(utc=prior.utc(),fits=63,exploratory=True,rows=rows,contrasts=contrasts,gates=gates))
    print(gates,flush=True)


def smoke():
    x = torch.randn(4,512,32); base = torch.tensor([0.,.1,1.,3.])
    models = {a:Residual(a,601,0,.4) for a in ARMS}
    for a,net in models.items():
        delta,logit = net(x,base)
        assert torch.equal(delta,torch.zeros_like(delta))
        loss = ((base+delta)-torch.tensor([0.,0.,2.,2.])).square().mean()+F.binary_cross_entropy_with_logits(logit,torch.tensor([0.,0.,1.,1.]))
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    for k,v in models['attention_bce'].state_dict().items(): assert torch.equal(v,models['attention_mse'].state_dict()[k])
    prior.save(ROOT/'smoke.json',dict(passed=True,checks=['exact zero correction initialization','finite gradients','matched loss-control initial state']))
    print('Smoke passed')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','fit','lock','evaluate','smoke'])
    parser.add_argument('--mouse',choices=prior.MICE)
    args = parser.parse_args(); torch.set_num_threads(2)
    if args.stage=='fit':
        assert args.mouse
        for seed in SEEDS:
            for arm in ARMS: fit(args.mouse,arm,seed)
    else: {'prepare':prepare,'lock':lock,'evaluate':evaluate,'smoke':smoke}[args.stage]()
