"""Matched attention/MLP movement gates and positive speed heads on frozen MLP features."""
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

ARMS = ('attention_bce','mlp_bce','attention_mse','mlp_mse')
SEEDS = (501,502,503)
EPOCHS = 24


class Decoder(nn.Module):
    def __init__(self,arm,seed,mouse,prevalence,conditional_mean):
        super().__init__()
        family = 'attention' if arm.startswith('attention') else 'mlp'
        self.gate = PopulationDecoder(Recipe(family=family,width=16,depth=1,patch=4,dropout=0.),seed,mouse)
        nn.init.constant_(self.gate.head[-1].bias,math.log(prevalence/(1-prevalence)))
        torch.manual_seed(seed+5000)
        self.speed = nn.Sequential(nn.LayerNorm(128),nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))
        nn.init.zeros_(self.speed[-1].weight)
        nn.init.constant_(self.speed[-1].bias,conditional_mean+math.log(-math.expm1(-conditional_mean)))

    def forward(self,x,features):
        logits = self.gate(x)
        speed = F.softplus(self.speed(features).squeeze(-1))
        return logits.sigmoid()*speed,logits,speed


def name(mouse,arm,seed):
    return f'{mouse}_{arm}_{seed}'


def settings(mouse):
    d = prior.examples(mouse,'train')
    y = d.y[d.indices]-d.metadata['lower']
    p = float(np.clip(np.mean(y>.05),1e-4,1-1e-4))
    return p,float(y.mean()/p)


def model(mouse,arm,seed):
    return Decoder(arm,seed,prior.MICE.index(mouse),*settings(mouse))


def check():
    p = prior.read(ROOT/'protocol.json')
    for file,value in p['sources'].items():
        assert prior.digest(prior.REPO/file)==value,file


def prepare():
    prior.check_lock()
    assert not (ROOT/'protocol.json').exists()
    sources = [Path(__file__),prior.REPO/'decoding/model.py',prior.REPO/'decoding/config.py',
        prior.REPO/'decoding/data.py',Path(prior.__file__),prior.ROOT/'protocol.json']
    prior.save(ROOT/'protocol.json',dict(utc=prior.utc(),mice=prior.MICE,seeds=SEEDS,arms=ARMS,
        fits=84,epochs=EPOCHS,updates_per_fit=24*64,batch_size=64,lr=.001,weight_decay=.001,
        question='Does a trainable temporal-attention movement gate improve an MLP speed decoder, and does explicit quiet/active supervision matter?',
        gate='512x32 activity; trainable population readin width16, patch4, one causal attention or temporalMLP block; final token plus population mean/std histories; scalar logit',
        speed='frozen selected MLP checkpoint features128, new LayerNorm128/Linear64/GELU/Linear1/softplus head; gradients through new speed head and raw-activity gate only',
        seed_mapping='new seeds501/502/503 use pretrained MLP401/402/403 respectively; cached features from training and validation separately',
        target='nonnegative running in training-SD units; moving iff speed>0.05; unweighted BCE weight1 for *_bce, weight0 for *_mse',
        loss='mean squared speed error of sigmoid(gate)*softplus(speedhead), plus BCE when enabled; no class resampling, label input, or conditional masking',
        initialization='all start near training mean: gate prior=train movement prevalence clipped1e-4; conditional speed=train mean/prior; common tensors paired across arms',
        optimizer='AdamW, cosine to0.1lr over24epochs, clip grad norm1, identical batches/order per seed/mouse; select validation speed MSE including epoch0; no early stop',
        primary='attention_bce vs mlp_bce; same feature backbone, target, updates, losses, initialization of common components; parameter counts recorded',
        secondary=['attention_bce vs attention_mse','mlp_bce vs mlp_mse','each arm vs original selected MLP and transformer','zero-speed and training-mean references'],
        gates='primary: >=5%mean relative MSE gain,>=6/7mouse wins,>=14/21paired seed wins,maxharm<=10%,nonnegative MAE gain; useful repair additionally beats own pretrained MLP baseline by>=5%,>=5/7wins,maxharm<=10%,QFM improves>=5/7; original failures unchanged',
        barrier='all84 checkpoints locked before new test scoring; no architecture/loss/epoch/grid changes after scoring',
        inference='gate receives only activity, speed head receives only neural features; true movement labels used for training/slice evaluation only',
        limitations='exploratory reused recordings and pretrained validation-selected MLP features; not fresh confirmation, not full end-to-end speed backbone training, no universal architecture conclusion; no automatic extension',
        sources={str(p.relative_to(prior.REPO)):prior.digest(p) for p in sources}))
    cache = ROOT/'cache'
    cache.mkdir(exist_ok=False)
    receipts = {}
    for mouse in prior.MICE:
        for seed in SEEDS:
            backbone_seed = seed-100
            net = prior.network(mouse,'mlp',backbone_seed)
            arrays = {}
            for split in ('train','validation'):
                data = prior.examples(mouse,split)
                features,head = prior.features(net,data)
                arrays[split+'_features'] = features
                arrays[split+'_parent'] = head
            path = cache/f'{mouse}_{seed}.npz'
            np.savez(path,**arrays)
            receipts[path.name] = prior.digest(path)
        print(mouse,'frozen MLP feature cache complete',flush=True)
    prior.save(ROOT/'cache_lock.json',dict(utc=prior.utc(),sha256=receipts))


def predict(net,data,features):
    net.eval()
    outputs = [[],[],[]]
    with torch.inference_mode():
        for start in range(0,len(data.indices),128):
            idx = data.indices[start:start+128]
            x = torch.from_numpy(np.array(data.x[idx],copy=True))
            values = net(x,torch.from_numpy(features[start:start+128]))
            for out,value in zip(outputs,values): out.append(value.numpy())
    return [np.concatenate(v) for v in outputs]


def fit(mouse,arm,seed):
    check()
    out = ROOT/'fits'/name(mouse,arm,seed)
    if (out/'result.json').exists():
        r = prior.read(out/'result.json')
        assert prior.digest(out/'selected.pt')==r['sha256']
        return
    out.mkdir(parents=True,exist_ok=False)
    path = ROOT/'cache'/f'{mouse}_{seed}.npz'
    assert prior.digest(path)==prior.read(ROOT/'cache_lock.json')['sha256'][path.name]
    with np.load(path) as cache:
        train_features,val_features = cache['train_features'],cache['validation_features']
    train,val = prior.examples(mouse,'train'),prior.examples(mouse,'validation')
    lower = train.metadata['lower']
    net = model(mouse,arm,seed)
    torch.save(net.state_dict(),out/'initial.pt')
    torch.save(net.state_dict(),out/'selected.pt')
    target = val.y[val.indices]-lower
    initial = predict(net,val,val_features)[0]
    best = float(np.mean((initial.astype(float)-target)**2))
    chosen = 0
    history = [dict(epoch=0,validation_mse=best)]
    val_predictions = [initial]
    opt = torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS,eta_min=.0001)
    start_time = time.monotonic()
    for epoch in range(1,EPOCHS+1):
        net.train()
        order = np.random.default_rng(seed*100000+epoch*100+prior.MICE.index(mouse)).permutation(len(train.indices))
        losses = np.zeros(2)
        for start in range(0,len(order),64):
            local = order[start:start+64]
            idx = train.indices[local]
            x = torch.from_numpy(np.array(train.x[idx],copy=True))
            y = torch.tensor(train.y[idx]-lower,dtype=torch.float32)
            f = torch.from_numpy(train_features[local])
            opt.zero_grad(set_to_none=True)
            p,logits,_ = net(x,f)
            speed_loss = (p-y).square().mean()
            bce = F.binary_cross_entropy_with_logits(logits,(y>.05).float())
            loss = (speed_loss+(bce if arm.endswith('bce') else 0))*(len(idx)/64)
            assert torch.isfinite(loss)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            losses += [float(speed_loss.detach())*len(idx),float(bce.detach())*len(idx)]
        sched.step()
        p = predict(net,val,val_features)[0]
        error = float(np.mean((p.astype(float)-target)**2))
        assert np.isfinite(error)
        if error<best:
            best,chosen = error,epoch
            torch.save(net.state_dict(),out/'selected.pt')
        history.append(dict(epoch=epoch,validation_mse=error,training_mse=losses[0]/len(order),training_bce=losses[1]/len(order)))
        val_predictions.append(p)
        prior.save(out/'history.json',history)
    net.load_state_dict(torch.load(out/'selected.pt',weights_only=True))
    np.testing.assert_array_equal(predict(net,val,val_features)[0],val_predictions[chosen])
    assert chosen==int(np.argmin([h['validation_mse'] for h in history]))
    np.savez_compressed(out/'validation.npz',prediction=np.stack(val_predictions),target=target)
    prior.save(out/'result.json',dict(mouse=mouse,arm=arm,seed=seed,selected_epoch=chosen,
        validation_mse=best,sha256=prior.digest(out/'selected.pt'),seconds=time.monotonic()-start_time,
        parameters=sum(p.numel() for p in net.parameters()),initial_sha256=prior.digest(out/'initial.pt'),
        selected_reload_exact=True,epochs=EPOCHS,updates=EPOCHS*math.ceil(len(train.indices)/64),test_scored=False))
    print(name(mouse,arm,seed),'complete; selected epoch',chosen,flush=True)


def train_all():
    for mouse in prior.MICE:
        for seed in SEEDS:
            for arm in ARMS: fit(mouse,arm,seed)
    check()
    paths = {str(p.relative_to(ROOT)):prior.digest(p) for p in sorted((ROOT/'fits').glob('*/selected.pt'))}
    assert len(paths)==84
    prior.save(ROOT/'evaluation_lock.json',dict(utc=prior.utc(),protocol_sha256=prior.digest(ROOT/'protocol.json'),sha256=paths))


def evaluate():
    check()
    lock = prior.read(ROOT/'evaluation_lock.json')
    assert lock['protocol_sha256']==prior.digest(ROOT/'protocol.json')
    for name_,expected in lock['sha256'].items(): assert prior.digest(ROOT/name_)==expected
    out = ROOT/'predictions'
    out.mkdir(exist_ok=False)
    rows = []
    for mouse in prior.MICE:
        data = prior.examples(mouse,'test')
        lower = data.metadata['lower']
        y = data.y[data.indices]
        for seed in SEEDS:
            features,parent = prior.features(prior.network(mouse,'mlp',seed-100),data)
            for family in ('mlp','transformer'):
                path = prior.ROOT/'predictions'/f'{mouse}_{family}_{seed-100}.npz'
                with np.load(path) as saved:
                    np.testing.assert_array_equal(y,saved['target'])
                    if family=='mlp': np.testing.assert_array_equal(parent,saved['parent'])
                    rows.append(dict(mouse=mouse,seed=seed,arm=family+'_parent',**prior.metrics(saved['parent'],y,lower)))
            for arm in ARMS:
                net = model(mouse,arm,seed)
                folder = ROOT/'fits'/name(mouse,arm,seed)
                net.load_state_dict(torch.load(folder/'selected.pt',weights_only=True))
                speed,logits,conditional = predict(net,data,features)
                p = speed.astype(float)+lower
                probability = torch.from_numpy(logits).sigmoid().numpy()
                move = (y-lower>.05).astype(float)
                rows.append(dict(mouse=mouse,seed=seed,arm=arm,**prior.metrics(p,y,lower),
                    brier=float(np.mean((probability-move)**2)),mean_probability=float(probability.mean())))
                np.savez_compressed(out/(name(mouse,arm,seed)+'.npz'),prediction=p,target=y,
                    probability=probability,conditional_speed=conditional)
        for arm,value in [('zero',lower),('mean',0.)]:
            rows.append(dict(mouse=mouse,seed=None,arm=arm,**prior.metrics(np.full(len(y),value),y,lower)))
        prior.save(ROOT/'test_progress.json',rows)
        print(mouse,'test scoring complete',flush=True)
    pairs = [('attention_bce','mlp_bce'),('attention_bce','attention_mse'),('mlp_bce','mlp_mse')]
    pairs += [(arm,c) for arm in ARMS for c in ('mlp_parent','transformer_parent','zero')]
    contrasts = {a+'_vs_'+b:prior.contrast(rows,a,b) for a,b in pairs}
    primary = contrasts['attention_bce_vs_mlp_bce']
    wins = sum(next(r['mse'] for r in rows if r['mouse']==m and r['seed']==s and r['arm']=='attention_bce') <
               next(r['mse'] for r in rows if r['mouse']==m and r['seed']==s and r['arm']=='mlp_bce') for m in prior.MICE for s in SEEDS)
    checks = dict(mean_gain=primary['mean_gain']>=.05,mouse_wins=primary['mouse_wins']>=6,
        seed_wins=wins>=14,harm=primary['max_harm']<=.1,mae=primary['mean_mae_gain']>=0)
    repairs = {}
    for arm in ARMS:
        c = contrasts[arm+'_vs_mlp_parent']
        cc = dict(mean_gain=c['mean_gain']>=.05,mouse_wins=c['mouse_wins']>=5,harm=c['max_harm']<=.1,quiet=c['quiet_wins']>=5)
        repairs[arm] = dict(passed=all(cc.values()),checks=cc)
    prior.save(ROOT/'summary.json',dict(utc=prior.utc(),exploratory=True,new_fits=84,rows=rows,
        contrasts=contrasts,primary_gate=dict(passed=all(checks.values()),checks=checks,seed_wins=wins),repair_gates=repairs))
    print('Primary:',checks,'repairs:',repairs,flush=True)


def smoke():
    x = torch.randn(4,512,32)
    features = torch.randn(4,128)
    models = {arm:Decoder(arm,501,0,.4,2.) for arm in ARMS}
    for arm,net in models.items():
        p,logits,speed = net(x,features)
        assert p.shape==(4,) and (p>=0).all()
        torch.testing.assert_close(p,torch.full((4,),.8))
        loss = (p-torch.tensor([0.,1.,0.,2.])).square().mean()+F.binary_cross_entropy_with_logits(logits,torch.tensor([0.,1.,0.,1.]))
        loss.backward()
        assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in net.parameters())
    for a,b in [('attention_bce','attention_mse'),('mlp_bce','mlp_mse')]:
        for k,v in models[a].state_dict().items(): torch.testing.assert_close(v,models[b].state_dict()[k],rtol=0,atol=0)
    for k,v in models['attention_bce'].speed.state_dict().items():
        torch.testing.assert_close(v,models['mlp_bce'].speed.state_dict()[k],rtol=0,atol=0)
    prior.save(ROOT/'smoke.json',dict(passed=True,checks=['shape','nonnegative speed','training mean initialization','finite gradients','loss-pair exact initialization','speed heads exactly matched']))
    print('Smoke checks passed',flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['smoke','prepare','train','evaluate'])
    args = parser.parse_args()
    torch.set_num_threads(2)
    {'smoke':smoke,'prepare':prepare,'train':train_all,'evaluate':evaluate}[args.stage]()
