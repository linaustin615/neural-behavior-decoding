"""Bounded development comparison; later outcomes are scored only after selection locks."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from scipy.io import loadmat
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
BASE = ROOT.parent/'2026-10-03_development_baselines'
OLD = ROOT.parent/'2026-10-02_stringer_replication'
SEEDS = [10, 11]
RECIPES = [dict(lr=lr, weight_decay=wd) for lr in [.001, .0003] for wd in [.01, .1]]
FAMILIES = ['transformer', 'pooled_mlp']


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def dependencies():
    return module('fair_baseline', BASE/'run.py'), module('fair_archive', OLD/'run.py')


class PooledMLP(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = deepcopy(tokenizer)
        self.mix = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, 64), nn.GELU(),
                                 nn.Dropout(.05), nn.Linear(64, 32))
        self.head = nn.Sequential(nn.Linear(32, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, x, positions, ids):
        tokens = self.tokenizer(x, positions, ids)
        pooled = (tokens+self.mix(tokens)).mean(1)
        return self.head(pooled).squeeze(-1)


def model(old, family, seed):
    original = old.model(seed, 'global', np.zeros(2048, dtype=np.int64))
    if family == 'transformer':
        return original
    require(family == 'pooled_mlp', 'Unknown model')
    torch.manual_seed(seed+900000)
    return PooledMLP(original.base.encoder.tokenizer)


def predict(net, x):
    net.eval()
    with torch.no_grad():
        result = torch.cat([net(x[i:i+64], torch.zeros(2048, 3), torch.arange(2048))
                            for i in range(0, len(x), 64)]).numpy()
    require(np.isfinite(result).all(), 'Nonfinite predictions')
    return result


def score(prediction, y, stats):
    pred = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64)
    require(pred.shape == target.shape and np.isfinite(pred).all(), 'Invalid prediction')
    bounded = np.maximum(pred, -stats['speed_mean']/stats['speed_std'])
    mse = float(np.mean((bounded-target)**2))
    return dict(mse=mse, raw_mse=float(np.mean((pred-target)**2)),
                r2=1-mse/float(np.var(target)),
                predicted_speed_std=float(bounded.std()*stats['speed_std']),
                rmse_speed_units=float(np.sqrt(mse)*stats['speed_std']))


def selfcheck():
    _, old = dependencies()
    a, b = model(old, 'transformer', 10), model(old, 'pooled_mlp', 10)
    for key, value in a.base.encoder.tokenizer.state_dict().items():
        torch.testing.assert_close(value, b.tokenizer.state_dict()[key], rtol=0, atol=0)
    counts = {f: sum(p.numel() for p in n.parameters()) for f, n in zip(FAMILIES, [a, b])}
    require(abs(counts['transformer']-counts['pooled_mlp'])/counts['transformer'] < .01,
            'Capacity mismatch above1%')
    require(not any(isinstance(m, nn.MultiheadAttention) for m in b.modules()), 'Attention in control')
    x = torch.randn(3, 2048, 8)
    ids = torch.arange(2048)
    permutation = torch.randperm(2048)
    for net in [a, b]:
        net.eval()
        original = net(x, torch.zeros(2048, 3), ids)
        permuted = net(x[:, permutation], torch.zeros(2048, 3), ids[permutation])
        torch.testing.assert_close(original, permuted, rtol=2e-5, atol=2e-6)
        net.train()
        opt = torch.optim.AdamW(net.parameters(), lr=.001)
        state = deepcopy(net.state_dict())
        loss = nn.functional.mse_loss(net(x, torch.zeros(2048, 3), ids), torch.tensor([0., 1., -1.]))
        loss.backward()
        require(all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()), 'Bad gradient')
        require(all(p.grad is not None and p.grad.norm()>0 for n,p in net.named_parameters()
                    if 'identity_embedding' in n), 'ID embedding disconnected')
        opt.step()
        require(any(not torch.equal(state[k],v) for k,v in net.state_dict().items()), 'No update')
    stats = dict(speed_mean=1., speed_std=2.)
    s = score(np.array([-2., .5]), np.array([-.5, 1.]), stats)
    require(s['mse'] == .125 and s['raw_mse'] == 1.25, 'Scoring failed')
    write(ROOT/'selfcheck.json', dict(passed=True, parameters=counts, identical_initial_tokenizers=True,
          no_attention_control=True, joint_cell_id_permutation_invariant=True, gradients_and_updates=True))


def freeze():
    require(not (ROOT/'protocol.json').exists(), 'Already frozen')
    base = read(BASE/'protocol.json')
    selfcheck()
    records = []
    for rec in base['cohort']:
        records.append(dict(mouse=rec['mouse'], file=rec['file'], development_stop=rec['development_stop'],
            train=rec['folds'][1]['train'], selection=rec['folds'][1]['validation'],
            evaluation=rec['folds'][2]['validation']))
    refs = [BASE/'run.py', BASE/'protocol.json', OLD/'run.py', OLD/'protocol.json']
    refs += [BASE/f"{r['mouse']}{suffix}" for r in records for suffix in ['_ids.npy', '_f2.npz']]
    p = dict(created_utc=datetime.now(timezone.utc).isoformat(), cohort=records,
        scope='Exploratory development, historically inspected recordings; no confirmatory p-values',
        question='Does modest tuning enable transformer superiority over a similarly sized nonattention network and ridge across four mice?',
        seeds=SEEDS, recipes=RECIPES, families=FAMILIES, fits=64, epochs=24,
        parameters=read(ROOT/'selfcheck.json')['parameters'],
        representation='Identical2048 IDs, eight activity bins, training-only normalization, additive nonlinear activity+ID tokenizer initialized identically per seed; zero positions. Transformer inherited16 summaries; control shared residual token MLP then uniform mean pooling and nonlinear head. Counts within1%; not an isolated attention-only ablation.',
        training='AdamW,batch32,gradient norm clip1,unbounded normalized MSE,cosine schedule ending at10% initial lr,24 epochs,no early stop; same batch orders per seed',
        selection='For each seed/recipe choose minimum bounded selection MSE including epoch0. For each mouse/family choose one recipe by mean selection MSE across BOTH seeds. Ties choose first epoch/recipe. Lock all choices before any later scoring.',
        ridge='Reuse4 archived fold2 ridge solutions .01,.1,1,10 with exactly same training IDs/normalization. Select lambda by selection interval only; no new ridge fits.',
        primary='Per-mouse mean of two selected-seed bounded later MSE divided by selected-ridge later MSE; also pooledMLP contrast and zero/training-mean baselines. Mouse is biological unit, not seeds or blocks.',
        secondary='Fixed default recipe0 vs tuned; selected-family epoch0; raw MSE; quiet/moving errors using training speed75th percentile as a threshold; apply training-feature min/max guard to selected checkpoints and ridge as a fixed inference-only robustness probe. No choice based on these outcomes.',
        exclusions='No old evaluation-tail examples, coordinate/grouping sweep, data/seed replacement, extra epochs, grid expansion or publication.',
        limits='One new split arrangement per mouse; all previously inspected; two technical seeds; finite tuning grid, approximate parameter matching, architecture differences beyond attention; no universal model-family verdict.',
        application_hashes=base['application_hashes'], reference_hashes={str(f.relative_to(PROJECT)):digest(f) for f in refs},
        source_sha256=digest(Path(__file__)))
    write(ROOT/'protocol.json', p)


def verify():
    p = read(ROOT/'protocol.json')
    require(digest(Path(__file__)) == p['source_sha256'], 'Runner changed')
    for category in ['application_hashes', 'reference_hashes']:
        for name, expected in p[category].items():
            require(digest(PROJECT/name) == expected, 'Source changed: '+name)
    return p


def prepare():
    p = verify()
    require(not (ROOT/'prepared.json').exists(), 'Already prepared')
    base, _ = dependencies()
    rows = []
    for rec in p['cohort']:
        dest = ROOT/rec['mouse']
        dest.mkdir()
        raw = loadmat(PROJECT/rec['file'], variable_names=['Fsp', 'beh'], simplify_cells=True)
        activity, speed = base.bin_prefix(raw['Fsp'], np.asarray(raw['beh']['runSpeed']).reshape(-1), rec['development_stop'])
        del raw
        ids = np.load(BASE/f"{rec['mouse']}_ids.npy")
        activity = activity[ids]
        arrays, stats = base.matrices(activity, speed, dict(train=rec['train'], validation=rec['selection']))
        with np.load(BASE/f"{rec['mouse']}_f2.npz") as saved:
            for name, value in stats.items():
                np.testing.assert_array_equal(value, saved[name])
            np.testing.assert_array_equal(arrays['validation'][1].numpy(), saved['target'])
        for source, name in [('train', 'train'), ('validation', 'selection')]:
            x, y = arrays[source]
            np.save(dest/f'{name}_x.npy', x.numpy().reshape(-1, 2048, 8).astype(np.float32))
            np.save(dest/f'{name}_y.npy', y.numpy())
        x = arrays['train'][0].numpy().reshape(-1, 2048, 8).astype(np.float32)
        np.savez(dest/'bounds.npz', low=x.min(0), high=x.max(0))
        #archive only the later raw segment; fitting never constructs its examples
        start, stop = rec['evaluation']
        require(rec['train'][1]+100 <= rec['selection'][0] and rec['selection'][1]+100 <= start,
                'Chronological gap failed')
        np.savez(dest/'later_raw.npz', activity=activity[:, start:stop], speed=speed[start:stop])
        np.savez(dest/'statistics.npz', **stats)
        write(dest/'metadata.json', dict(mouse=rec['mouse'], speed_mean=stats['speed_mean'], speed_std=stats['speed_std'],
              movement_threshold=float(np.quantile(speed[31:rec['train'][1]], .75)),
              train_n=len(arrays['train'][1]), selection_n=len(arrays['validation'][1]),
              evaluation_n=stop-start-31))
        rows.append(dict(mouse=rec['mouse'], exact_archived_statistics_and_targets=True,
                         files={f.name:digest(f) for f in dest.iterdir() if f.is_file()}))
        print('prepared '+rec['mouse'], flush=True)
    write(ROOT/'prepared.json', dict(passed=True, records=rows))


def fit(mouse):
    p = verify()
    require(mouse in [r['mouse'] for r in p['cohort']], 'Unknown mouse')
    _, old = dependencies()
    dest = ROOT/mouse
    prepared = next(r for r in read(ROOT/'prepared.json')['records'] if r['mouse']==mouse)
    for name,h in prepared['files'].items():
        require(digest(dest/name)==h, 'Prepared file changed')
    x = torch.from_numpy(np.load(dest/'train_x.npy'))
    y = torch.from_numpy(np.load(dest/'train_y.npy').astype(np.float32))
    xs = torch.from_numpy(np.load(dest/'selection_x.npy'))
    ys = np.load(dest/'selection_y.npy')
    stats = read(dest/'metadata.json')
    for family in FAMILIES:
        for recipe_id, recipe in enumerate(RECIPES):
            for seed in SEEDS:
                out = dest/f'{family}_r{recipe_id}_s{seed}'
                out.mkdir()
                start = time.monotonic()
                net = model(old, family, seed)
                initial = deepcopy(net.state_dict())
                torch.save(initial, out/'initial.pt')
                before = predict(net, xs)
                best = score(before, ys, stats)['mse']
                chosen = deepcopy(initial)
                selected_epoch = 0
                history = [dict(epoch=0, selection=score(before, ys, stats))]
                predictions = [before]
                generator = torch.Generator().manual_seed(seed)
                loader = DataLoader(TensorDataset(x,y,torch.arange(len(y))),batch_size=32,shuffle=True,generator=generator)
                opt = torch.optim.AdamW(net.parameters(), **recipe)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,24,eta_min=recipe['lr']*.1)
                gradients = {}; orders = []
                for epoch in range(1,25):
                    net.train(); total=0.; order=hashlib.sha256()
                    for xb,yb,idx in loader:
                        order.update(idx.numpy().tobytes())
                        opt.zero_grad(set_to_none=True)
                        loss = nn.functional.mse_loss(net(xb,torch.zeros(2048,3),torch.arange(2048)),yb)
                        require(torch.isfinite(loss), 'Nonfinite loss')
                        loss.backward()
                        nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
                        for name,par in net.named_parameters():
                            if 'in_proj_weight' in name or 'identity_embedding' in name:
                                require(par.grad is not None and torch.isfinite(par.grad).all(), 'Missing gradient')
                                gradients[name]=max(gradients.get(name,0.),float(par.grad.norm()))
                        opt.step(); total+=float(loss.detach())*len(yb)
                    scheduler.step(); orders.append(order.hexdigest())
                    prediction = predict(net,xs)
                    metric = score(prediction,ys,stats)
                    history.append(dict(epoch=epoch,training_batch_mse=total/len(y),selection=metric))
                    predictions.append(prediction)
                    if metric['mse'] < best:
                        best=metric['mse']; selected_epoch=epoch; chosen=deepcopy(net.state_dict())
                    write(out/'history.json',history)
                require(all(v>0 for v in gradients.values()), 'Zero monitored gradients')
                changes={k:float((net.state_dict()[k]-initial[k]).norm()) for k in gradients}
                require(all(v>0 for v in changes.values()), 'No monitored parameter change')
                torch.save(chosen,out/'selected.pt')
                net.load_state_dict(torch.load(out/'selected.pt',weights_only=True))
                np.testing.assert_array_equal(predict(net,xs),predictions[selected_epoch])
                training = score(predict(net,x),np.load(dest/'train_y.npy'),stats)
                np.savez_compressed(out/'selection_predictions.npz',predictions=np.stack(predictions),target=ys)
                write(out/'result.json',dict(mouse=mouse,family=family,recipe_id=recipe_id,seed=seed,
                      selected_epoch=selected_epoch,selection_mse=best,training=training,
                      max_gradient=gradients,parameter_changes=changes,batch_order_hashes=orders,
                      selected_reload_exact=True,elapsed_seconds=time.monotonic()-start))
                print(json.dumps(dict(mouse=mouse,family=family,recipe=recipe_id,seed=seed,
                      epoch=selected_epoch,selection_mse=best,seconds=round(time.monotonic()-start))),flush=True)
    write(dest/'finished.json',dict(fits=16,complete=True))


def lock():
    p=verify()
    require(not (ROOT/'selection_lock.json').exists(),'Already locked')
    choices=[]; files={}
    for rec in p['cohort']:
        dest=ROOT/rec['mouse']
        require(read(dest/'finished.json')['fits']==16,'Incomplete fits')
        orders={}
        for family in FAMILIES:
            means=[]
            for rid in range(4):
                values=[]
                for seed in SEEDS:
                    out=dest/f'{family}_r{rid}_s{seed}'
                    r=read(out/'result.json'); values.append(r['selection_mse'])
                    if seed in orders:
                        require(orders[seed]==r['batch_order_hashes'],'Unequal batch orders')
                    orders[seed]=r['batch_order_hashes']
                    with np.load(out/'selection_predictions.npz') as saved:
                        metrics=[score(v,saved['target'],read(dest/'metadata.json'))['mse'] for v in saved['predictions']]
                    require(int(np.argmin(metrics))==r['selected_epoch'],'Selection mismatch')
                    np.testing.assert_allclose(metrics,[v['selection']['mse'] for v in read(out/'history.json')],rtol=1e-12,atol=1e-12)
                    for filename in ['selected.pt','initial.pt','result.json','selection_predictions.npz']:
                        files[str((out/filename).relative_to(ROOT))]=digest(out/filename)
                means.append(float(np.mean(values)))
            choices.append(dict(mouse=rec['mouse'],family=family,recipe_id=int(np.argmin(means)),selection_means=means))
        with np.load(BASE/f"{rec['mouse']}_f2.npz") as saved:
            scores=[score(saved['prediction'][:,i],saved['target'],read(dest/'metadata.json'))['mse'] for i in range(4)]
        choices.append(dict(mouse=rec['mouse'],family='ridge',lambda_index=int(np.argmin(scores)),selection_mses=scores))
    write(ROOT/'selection_lock.json',dict(locked_utc=datetime.now(timezone.utc).isoformat(),choices=choices,files=files,
          fits=64,matched_batch_orders=True,all_epoch_selection_scores_checked=True,later_scored=False))


def evaluate():
    p=verify(); locked=read(ROOT/'selection_lock.json')
    require(not (ROOT/'results.json').exists(),'Already evaluated')
    for name,h in locked['files'].items():
        require(digest(ROOT/name)==h,'Locked artifact changed')
    base,old=dependencies(); rows=[]
    for rec in p['cohort']:
        dest=ROOT/rec['mouse']; stats=read(dest/'metadata.json')
        with np.load(dest/'later_raw.npz') as raw, np.load(dest/'statistics.npz') as st:
            z=(raw['activity']-st['activity_mean'])/st['activity_std']
            x64=np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(z,8,axis=1)[:,24:].transpose(1,0,2))
            y=(raw['speed'][31:]-stats['speed_mean'])/stats['speed_std']
            physical_y=raw['speed'][31:].copy()
        x=torch.from_numpy(x64.astype(np.float32))
        with np.load(dest/'bounds.npz') as bounds:
            guarded=np.clip(x.numpy(),bounds['low'],bounds['high'])
            guarded64=np.clip(x64,bounds['low'],bounds['high'])
        quiet=physical_y<=stats['movement_threshold']
        predictions={}; records=[]
        def add(label,pred,**meta):
            metric=score(pred,y,stats)
            bounded=np.maximum(pred,-stats['speed_mean']/stats['speed_std'])
            metric['quiet_mse']=float(np.mean((bounded[quiet]-y[quiet])**2)) if quiet.any() else None
            metric['moving_mse']=float(np.mean((bounded[~quiet]-y[~quiet])**2)) if (~quiet).any() else None
            predictions[label]=pred
            records.append(dict(label=label,**meta,**metric))
        add('zero',np.full(len(y),-stats['speed_mean']/stats['speed_std']))
        add('mean',np.zeros(len(y)))
        ridge_choice=next(c for c in locked['choices'] if c['mouse']==rec['mouse'] and c['family']=='ridge')
        idx=ridge_choice['lambda_index']
        with np.load(BASE/f"{rec['mouse']}_f2.npz") as saved:
            ridge=x64.reshape(len(y),-1)@saved['weights'][:,idx]+saved['intercept'][idx]
            rg=guarded64.reshape(len(y),-1)@saved['weights'][:,idx]+saved['intercept'][idx]
        add('ridge',ridge,regularization=[.01,.1,1.,10.][idx])
        add('ridge_guard',rg,regularization=[.01,.1,1.,10.][idx])
        for family in FAMILIES:
            choice=next(c for c in locked['choices'] if c['mouse']==rec['mouse'] and c['family']==family)
            rid=choice['recipe_id']
            for seed in SEEDS:
                for kind,recipe_id in [('tuned',rid),('default',0)]:
                    out=dest/f'{family}_r{recipe_id}_s{seed}'
                    net=model(old,family,seed)
                    net.load_state_dict(torch.load(out/'selected.pt',weights_only=True))
                    label=f'{family}_{kind}_s{seed}'
                    add(label,predict(net,x),recipe_id=recipe_id,epoch=read(out/'result.json')['selected_epoch'])
                    if kind=='tuned':
                        add(label+'_guard',predict(net,torch.from_numpy(guarded)),recipe_id=recipe_id)
                net.load_state_dict(torch.load(out/'initial.pt',weights_only=True))
                add(f'{family}_untrained_s{seed}',predict(net,x))
        np.savez_compressed(dest/'later_predictions.npz',target=y,quiet=quiet,**predictions)
        with np.load(dest/'later_predictions.npz') as saved:
            for row in records:
                pred=np.maximum(saved[row['label']].astype(np.float64),-stats['speed_mean']/stats['speed_std'])
                independent=sum((float(a)-float(b))**2 for a,b in zip(pred,saved['target']))/len(y)
                require(abs(independent-row['mse'])<1e-10*max(1.,row['mse']),'Later MSE mismatch')
        rows.append(dict(mouse=rec['mouse'],n=len(y),quiet_n=int(quiet.sum()),moving_n=int((~quiet).sum()),
             movement_threshold=stats['movement_threshold'],out_of_range_fraction=float(np.mean(x.numpy()!=guarded)),results=records))
        print(json.dumps(rows[-1]),flush=True)
    write(ROOT/'results.json',dict(scope=p['scope'],rows=rows,selection_lock_sha256=digest(ROOT/'selection_lock.json')))
    verify()
    write(ROOT/'audit.json',dict(passed=True,neural_fits=64,selection_locked_before_evaluation=True,
          selection_reload_exact=True,all_epoch_selection_mse_checked=True,matched_batch_orders=True,
          saved_later_mse_independently_checked=True,application_unchanged=True,old_evaluation_tails_used=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['check','freeze','prepare','fit','lock','evaluate'])
    parser.add_argument('--mouse')
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        if args.action=='fit':
            fit(args.mouse)
        else:
            dict(check=selfcheck,freeze=freeze,prepare=prepare,lock=lock,evaluate=evaluate)[args.action]()
