"""Frozen encoder retrieval pilot; no neural training, no test-based selection."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
SOURCE = ROOT.parent / '2026-10-06_facemap_validation'
sys.path.insert(0, str(REPO))
from decoding.config import MICE, SEEDS, recipe
from decoding.data import load
from decoding.model import PopulationDecoder

TAUS = (.01, .03, .1, .3, 1.)
ROLES = {'transformer': 'optimized_attention', 'mlp': 'optimized_population_mlp'}


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def job(mouse, family, seed):
    return f'independent_{mouse}_{ROLES[family]}_s{seed}'


def checkpoint(mouse, family, seed):
    return SOURCE / 'fits' / job(mouse, family, seed) / 'selected.pt'


def freeze():
    assert not (ROOT/'protocol.json').exists(), 'existing protocol cannot be overwritten'
    files = [Path(__file__), REPO/'decoding/model.py', REPO/'decoding/config.py',
             REPO/'decoding/data.py', SOURCE/'protocol.json', SOURCE/'evaluation_lock.json']
    source_lock = read(SOURCE/'evaluation_lock.json')['selected_sha256']
    for mouse in MICE:
        folder = SOURCE/'prepared'/mouse
        metadata = read(folder/'metadata.json')
        for name, expected in metadata['sha256'].items():
            assert digest(folder/name) == expected, (mouse, name)
        files.extend([folder/'metadata.json', *sorted(folder.glob('*.npy'))])
        spans = metadata['boundaries']
        assert spans['validation'][0] - spans['train'][1] >= 64
        assert spans['test'][0] - spans['validation'][1] >= 64
        for family in ROLES:
            for seed in SEEDS:
                p = checkpoint(mouse, family, seed)
                assert digest(p) == source_lock[str(p.relative_to(SOURCE))]
                files.extend([p, SOURCE/'test_predictions'/(job(mouse, family, seed)+'.npz')])
    save(ROOT/'protocol.json', dict(
        frozen_utc=utc(), status='exploratory; all seven test intervals previously examined',
        question='Does training-label retrieval improve frozen transformer features and outperform identical MLP/PCA retrieval?',
        mice=MICE, seeds=SEEDS, neural_fits=0, checkpoints=42, temperatures=TAUS,
        features='last encoder token + mean and population SD across neurons for each of 32 frames',
        normalization='subtract selected-training-feature mean, divide population SD floored at 1e-6, then L2 normalize; zero vectors remain zero',
        memory='all original capped training examples (up to4096), unique original indices, labels from training only',
        similarity='cosine; softmax(similarity/temperature); label weighted mean; float32 attention with float64 metrics',
        pca='flatten 512x32 standardized-activity windows; center on selected training windows; torch.pca_lowrank q72 niter2 seed1701, first64 components; standardize scores on training only; same retrieval',
        selection='one temperature per mouse/family minimizing mean validation clipped MSE over three seeds; PCA once per mouse; earliest listed temperature breaks exact ties',
        barrier='all21 mouse/family choices locked before any new test scoring; immutable protocol and input hashes',
        overlap='no training queries evaluated; chronological gaps exceed64 frames, disjoint train/validation/test windows',
        quiet='observed speed <= physical zero +0.05 training SD; QFM=mean clipped predicted speed above physical zero; no quiet frames yields null and no quiet win',
        primary_gate={'mean_relative_mse_gain_vs_parent_min': .05, 'mouse_wins_min': 5,
                      'max_mouse_relative_harm': .10, 'quiet_mouse_wins_min': 5,
                      'mean_relative_gain_vs_pca_strictly_positive': True},
        aggregation='average seed metrics within mouse first, then equal-mouse mean relative gains; no seed/window significance claims',
        secondary='same gate for MLP; transformer retrieval versus MLP retrieval, parent MLP, ridge and zero; MAE and active MSE descriptive',
        scope='Phase0 only; no automatic end-to-end training, no fresh confirmation, no changes to historical gates',
        sha256={str(p.relative_to(REPO)): digest(p) for p in files}))


def check_lock():
    protocol = read(ROOT/'protocol.json')
    for name, expected in protocol['sha256'].items():
        assert digest(REPO/name) == expected, name


def examples(mouse, split):
    return load(SOURCE/'prepared'/mouse, split)


def network(mouse, family, seed):
    net = PopulationDecoder(recipe(family), seed, MICE.index(mouse)).eval()
    net.load_state_dict(torch.load(checkpoint(mouse, family, seed), map_location='cpu', weights_only=True))
    return net


@torch.inference_mode()
def features(net, data):
    result, heads = [], []
    for start in range(0, len(data.indices), 64):
        x = torch.from_numpy(np.array(data.x[data.indices[start:start+64]], copy=True))
        tokens = net.encode(x)
        f = torch.cat([tokens[:, -1], x.mean(1), x.std(1, unbiased=False)], dim=1)
        result.append(f.numpy())
        heads.append(net.head(f).squeeze(-1).numpy())
    return np.concatenate(result), np.concatenate(heads)


def scaling(train):
    return train.mean(0, dtype=np.float64), np.maximum(train.std(0, dtype=np.float64), 1e-6)


def normalized(value, mean, scale):
    return F.normalize(torch.from_numpy(((value-mean)/scale).astype(np.float32)), dim=1)


def retrieve(query, bank, labels, taus):
    labels = torch.as_tensor(labels, dtype=torch.float32)
    result = [[] for _ in taus]
    for start in range(0, len(query), 256):
        similarity = query[start:start+256] @ bank.T
        for out, tau in zip(result, taus):
            out.append((torch.softmax(similarity/tau, dim=1) @ labels).numpy())
    predictions = [np.concatenate(out) for out in result]
    for p in predictions:
        assert np.isfinite(p).all()
        assert p.min() >= float(labels.min())-1e-5 and p.max() <= float(labels.max())+1e-5
    return predictions


def metrics(prediction, target, lower):
    p = np.maximum(np.asarray(prediction, dtype=np.float64), lower)
    y = np.asarray(target, dtype=np.float64)
    error = p-y
    quiet = y-lower <= .05
    active = y-lower >= .5
    variance = np.mean((y-y.mean())**2)
    return dict(mse=float(np.mean(error**2)), mae=float(np.mean(np.abs(error))),
                r2=float(1-np.mean(error**2)/variance) if variance else None,
                qfm=float(np.mean(p[quiet]-lower)) if quiet.any() else None,
                quiet_n=int(quiet.sum()), n=len(y),
                active_mse=float(np.mean(error[active]**2)) if active.any() else None)


def flat(data):
    return np.asarray(data.x[data.indices], dtype=np.float32).reshape(len(data.indices), -1)


def select():
    check_lock()
    assert not (ROOT/'selection_lock.json').exists()
    cache = ROOT/'cache'
    cache.mkdir(exist_ok=True)
    selections = {}
    for mouse in MICE:
        train, val = examples(mouse, 'train'), examples(mouse, 'validation')
        ybank, yval, lower = train.y[train.indices], val.y[val.indices], train.metadata['lower']
        selections[mouse] = {}
        for family in ROLES:
            scores = []
            for seed in SEEDS:
                net = network(mouse, family, seed)
                bank, _ = features(net, train)
                query, _ = features(net, val)
                mean, scale = scaling(bank)
                bn = normalized(bank, mean, scale)
                predictions = retrieve(normalized(query, mean, scale), bn, ybank, TAUS)
                scores.append([metrics(p, yval, lower)['mse'] for p in predictions])
                np.savez(cache/f'{mouse}_{family}_{seed}.npz', bank=bn.numpy(), mean=mean, scale=scale)
            index = int(np.argmin(np.mean(scores, axis=0)))
            selections[mouse][family] = dict(tau=TAUS[index], validation_mse_by_seed=scores)
        x = torch.from_numpy(flat(train))
        center = x.mean(0)
        x -= center
        torch.manual_seed(1701)
        _, _, basis = torch.pca_lowrank(x, q=72, center=False, niter=2)
        basis = basis[:, :64].contiguous()
        bank = (x@basis).numpy()
        query = ((torch.from_numpy(flat(val))-center)@basis).numpy()
        mean, scale = scaling(bank)
        bn = normalized(bank, mean, scale)
        predictions = retrieve(normalized(query, mean, scale), bn, ybank, TAUS)
        scores = [metrics(p, yval, lower)['mse'] for p in predictions]
        selections[mouse]['pca'] = dict(tau=TAUS[int(np.argmin(scores))], validation_mse=scores)
        np.savez(cache/f'{mouse}_pca.npz', center=center.numpy(), basis=basis.numpy(),
                 bank=bn.numpy(), mean=mean, scale=scale)
        save(ROOT/'selection_progress.json', selections)
        print(mouse, 'validation selection complete', flush=True)
    save(ROOT/'selection_lock.json', dict(utc=utc(), protocol_sha256=digest(ROOT/'protocol.json'),
         choices=selections, cache_sha256={p.name:digest(p) for p in sorted(cache.glob('*.npz'))}))


def contrast(rows, candidate, control):
    per_mouse = []
    for mouse in MICE:
        a = [r for r in rows if r['mouse']==mouse and r['arm']==candidate]
        b = [r for r in rows if r['mouse']==mouse and r['arm']==control]
        am, bm = np.mean([r['mse'] for r in a]), np.mean([r['mse'] for r in b])
        aq = [r['qfm'] for r in a if r['qfm'] is not None]
        bq = [r['qfm'] for r in b if r['qfm'] is not None]
        per_mouse.append(dict(mouse=mouse, candidate_mse=float(am), control_mse=float(bm),
                              gain=float(1-am/bm), quiet_win=bool(aq and bq and np.mean(aq)<np.mean(bq)),
                              mae_gain=float(1-np.mean([r['mae'] for r in a])/np.mean([r['mae'] for r in b]))))
    gains = [r['gain'] for r in per_mouse]
    return dict(mean_gain=float(np.mean(gains)), mouse_wins=sum(g>0 for g in gains),
                max_harm=float(max(0, -min(gains))), quiet_wins=sum(r['quiet_win'] for r in per_mouse),
                mean_mae_gain=float(np.mean([r['mae_gain'] for r in per_mouse])), per_mouse=per_mouse)


def evaluate():
    check_lock()
    lock = read(ROOT/'selection_lock.json')
    assert lock['protocol_sha256']==digest(ROOT/'protocol.json')
    for name, expected in lock['cache_sha256'].items():
        assert digest(ROOT/'cache'/name)==expected
    out = ROOT/'predictions'
    out.mkdir(exist_ok=False)
    rows, reloads = [], []
    for mouse in MICE:
        train, test = examples(mouse, 'train'), examples(mouse, 'test')
        ybank, target, lower = train.y[train.indices], test.y[test.indices], train.metadata['lower']
        for family in ROLES:
            for seed in SEEDS:
                net = network(mouse, family, seed)
                query, head = features(net, test)
                with np.load(SOURCE/'test_predictions'/(job(mouse,family,seed)+'.npz')) as old:
                    np.testing.assert_array_equal(target, old[mouse+'_target'])
                    discrepancy = float(np.max(np.abs(head-old[mouse+'_prediction'])))
                    np.testing.assert_allclose(head, old[mouse+'_prediction'], atol=2e-6, rtol=2e-6)
                    parent = old[mouse+'_prediction'].copy()
                reloads.append(dict(mouse=mouse, family=family, seed=seed, max_prediction_difference=discrepancy))
                with np.load(ROOT/'cache'/f'{mouse}_{family}_{seed}.npz') as c:
                    p = retrieve(normalized(query,c['mean'],c['scale']), torch.from_numpy(c['bank']),
                                 ybank, [lock['choices'][mouse][family]['tau']])[0]
                np.savez_compressed(out/f'{mouse}_{family}_{seed}.npz', prediction=p, parent=parent, target=target)
                for arm, prediction in [(family+'_retrieval',p),(family+'_parent',parent)]:
                    rows.append(dict(mouse=mouse, arm=arm, seed=seed, **metrics(prediction,target,lower)))
        with np.load(ROOT/'cache'/f'{mouse}_pca.npz') as c:
            query = ((torch.from_numpy(flat(test))-torch.from_numpy(c['center']))@torch.from_numpy(c['basis'])).numpy()
            p = retrieve(normalized(query,c['mean'],c['scale']),torch.from_numpy(c['bank']),
                         ybank,[lock['choices'][mouse]['pca']['tau']])[0]
        np.savez_compressed(out/f'{mouse}_pca.npz', prediction=p, target=target)
        rows.append(dict(mouse=mouse,arm='pca',seed=None,**metrics(p,target,lower)))
        with np.load(SOURCE/'test_predictions'/f'ridge_{mouse}.npz') as old:
            np.testing.assert_array_equal(target, old['target'])
            rows.append(dict(mouse=mouse,arm='ridge',seed=None,**metrics(old['prediction'],target,lower)))
        for name, value in [('zero',lower),('mean',0.)]:
            rows.append(dict(mouse=mouse,arm=name,seed=None,**metrics(np.full(len(target),value),target,lower)))
        save(ROOT/'test_progress.json', rows)
        print(mouse, 'locked test scoring complete', flush=True)
    pairs = [('transformer_retrieval',c) for c in ('transformer_parent','mlp_retrieval','mlp_parent','pca','ridge','zero')]
    pairs += [('mlp_retrieval',c) for c in ('mlp_parent','pca')]
    contrasts = {a+'_vs_'+b:contrast(rows,a,b) for a,b in pairs}
    gates = {}
    for family in ROLES:
        c = contrasts[f'{family}_retrieval_vs_{family}_parent']
        checks = dict(mean_gain=c['mean_gain']>=.05, mouse_wins=c['mouse_wins']>=5,
                      harm=c['max_harm']<=.1, quiet_wins=c['quiet_wins']>=5,
                      pca_gain=contrasts[f'{family}_retrieval_vs_pca']['mean_gain']>0)
        gates[family] = dict(passed=all(checks.values()), checks=checks)
    save(ROOT/'summary.json',dict(completed_utc=utc(),exploratory=True,neural_fits=0, rows=rows,
                                  contrasts=contrasts,gates=gates,reload_checks=reloads))
    print(json.dumps(gates),flush=True)


def selftest():
    bank = F.normalize(torch.tensor([[1.,0.],[0.,1.],[-1.,0.]]),dim=1)
    query = torch.tensor([[1.,0.],[0.,0.]])
    labels = np.array([0.,2.,5.])
    result = retrieve(query,bank,labels,[.1,1.])
    for tau,p in zip([.1,1.],result):
        logits = query.double().numpy()@bank.double().numpy().T/tau
        weights = np.exp(logits-logits.max(1,keepdims=True))
        reference = weights@labels/weights.sum(1)
        np.testing.assert_allclose(p,reference,atol=1e-6)
    np.testing.assert_allclose(result[0][1],labels.mean(),atol=1e-6)
    m = metrics(np.array([-2.,1.]),np.array([-1.,0.]),-1.)
    assert m['mse']==.5 and m['qfm']==0 and m['quiet_n']==1
    assert metrics([1.],[1.],-1.)['qfm'] is None
    save(ROOT/'selftest.json',dict(passed=True,utc=utc(),checks=['independent NumPy attention formula',
         'zero-query uniform weights','convex bounds','physical-zero clipping','missing quiet frames']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['selftest','freeze','select','evaluate'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    {'selftest':selftest,'freeze':freeze,'select':select,'evaluate':evaluate}[args.stage]()
