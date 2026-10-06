"""Diagnostic linear probes of archived frozen neural encoders; no encoder training."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / '2026-10-03_dynamics_baseline'
FAIR = ROOT.parent / '2026-10-03_fair_comparison'
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(BASE))
from models import Dynamics

MICE = ['MP030', 'MP032', 'MP033', 'MP034']
SEEDS = [10, 11, 12]
LAMBDAS = [0.0001, 0.001, 0.01, 0.1, 1., 10.]
CONTROLS = ['attention_random_augmented', 'mixer_pretrained_augmented', 'statistics', 'ridge']


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint(mouse, family, state, seed):
    return BASE / mouse / f'{family}_pretrain_s{seed}' / ('selected.pt' if state == 'pretrained' else 'initial.pt')


def freeze():
    assert not (ROOT / 'protocol.json').exists()
    files = [Path(__file__), BASE / 'models.py', BASE / 'protocol.json', BASE / 'results.json']
    for mouse in MICE:
        files += [BASE / mouse / f'{split}_{name}.npy' for split in ['train', 'selection'] for name in ['x', 'y']]
        files += [BASE / mouse / 'columns.npy', BASE / mouse / 'later_predictions.npz']
        files += [FAIR / mouse / name for name in ['later_raw.npz', 'statistics.npz', 'metadata.json']]
        files += [checkpoint(mouse, f, s, seed) for f in ['attention', 'mixer'] for s in ['random', 'pretrained'] for seed in SEEDS]
    files += [PROJECT / name for name in ['model.py', 'data.py', 'train.py']]
    write(ROOT / 'protocol.json', dict(
        created_utc=datetime.now(timezone.utc).isoformat(), mice=MICE, seeds=SEEDS,
        question='Does neural pretraining improve linearly accessible running-speed information in a fixed encoder, and does attention add value over a mixer?',
        design='Reuse archived initial and neural-selected checkpoints. No encoder updates, new neural training, new data, or architecture changes. Four mice, three seeds, attention/mixer, random/pretrained. Fit a ridge probe on pooled last-patch 16-dimensional features alone (secondary), or those features plus 64 population mean/std history features (primary). Statistics-only 64-dimensional ridge is a mandatory control. No nonlinear speed head or LayerNorm is used by the probes.',
        fitting='Feature centering/scaling uses training examples only; SD floor 1e-8. Unpenalized intercept, loss-normalized ridge. Fit all six fixed lambdas, select by earlier bounded speed MSE. Freeze all 100 probe choices before new later inference. Same chronological splits, contexts, selected neurons, targets and speed lower bound as the completed baseline.',
        lambdas=LAMBDAS, fits=100, linear_solves=600,
        primary='attention_pretrained_augmented versus attention_random_augmented, mixer_pretrained_augmented, statistics-only and archived matched ridge. Equal-weight mean within-mouse relative MSE gains; average seed errors first. Archived fine-tuned models/MLP are context, not a causal estimate of freezing.',
        gate='At least 5% average gain and at least 3/4 mouse wins versus each primary control; at least 8/12 paired-seed wins versus each neural control; no mouse more than 25% worse than statistics-only. Diagnostic decision rule, not significance.',
        uncertainty='2000 paired hierarchical mouse/seed/circular-100-bin bootstrap draws, seed 81205, descriptive 98.75% intervals for four contrasts.',
        limits='Historically reused data and an adaptively motivated follow-up; no independent significance claim. Linear probing tests accessible information in this pooled representation, not all information in the network. Frozen linear and prior fine-tuned nonlinear heads differ. No equal-total-update control is tested here. No selective mice, grid expansion, or additional fit budget.',
        hashes={str(p.relative_to(PROJECT)): digest(p) for p in files}))


def verify():
    for name, h in read(ROOT / 'protocol.json')['hashes'].items():
        assert digest(PROJECT / name) == h, name


def stats(x):
    return np.concatenate([x.mean(1), x.std(1)], 1).astype(np.float64)


def features(mouse, family, state, seed, arrays):
    net = Dynamics(family, seed)
    net.load_state_dict(torch.load(checkpoint(mouse, family, state, seed), weights_only=True))
    net.eval().requires_grad_(False)
    before = {k: v.clone() for k, v in net.state_dict().items()}
    result = []
    with torch.inference_mode():
        for x in arrays:
            z = torch.cat([net.encode(torch.from_numpy(x[i:i+64]))[:, :, -1].mean(1) for i in range(0, len(x), 64)]).numpy().astype(np.float64)
            assert z.shape == (len(x), 16) and np.isfinite(z).all()
            result.append(z)
    assert all(torch.equal(before[k], v) for k, v in net.state_dict().items())
    assert all(p.grad is None and not p.requires_grad for p in net.parameters())
    return result


def mse(pred, target, lower):
    assert pred.shape == target.shape and np.isfinite(pred).all()
    return float(np.mean((np.maximum(pred, lower) - target) ** 2))


def fit_probe(dest, label, x, xv, y, yv, lower):
    center = x.mean(0)
    scale = np.maximum(x.std(0), 1e-8)
    a = torch.from_numpy((x-center)/scale).double()
    target = torch.from_numpy(y-y.mean()).double()
    gram = a.T @ a
    rhs = a.T @ target
    candidates = []
    predictions = []
    weights = []
    for lam in LAMBDAS:
        system = gram + len(y)*lam*torch.eye(a.shape[1], dtype=torch.float64)
        weight = torch.linalg.solve(system, rhs)
        assert float((system@weight-rhs).norm()/rhs.norm().clamp_min(1e-12)) < 1e-7
        pred = ((torch.from_numpy((xv-center)/scale) @ weight) + y.mean()).numpy()
        independent = np.einsum('ij,j->i', (xv-center)/scale, weight.numpy(), optimize=False) + y.mean()
        np.testing.assert_allclose(pred, independent, rtol=1e-9, atol=1e-9)
        candidates.append(dict(lam=lam, mse=mse(pred, yv, lower)))
        predictions.append(pred)
        weights.append(weight.numpy())
    index = int(np.argmin([v['mse'] for v in candidates]))
    np.savez_compressed(dest / f'{label}.npz', center=center, scale=scale, weight=weights[index], mean=y.mean(), selection_predictions=np.stack(predictions), target=yv)
    return dict(label=label, selected_index=index, **candidates[index], candidates=candidates)


def fit():
    verify()
    assert not (ROOT / 'selection_lock.json').exists()
    rows = []
    for mouse in MICE:
        dest = ROOT / mouse
        dest.mkdir()
        x, xv = [np.load(BASE / mouse / f'{name}_x.npy') for name in ['train', 'selection']]
        y, yv = [np.load(BASE / mouse / f'{name}_y.npy').astype(np.float64) for name in ['train', 'selection']]
        meta = read(FAIR / mouse / 'metadata.json')
        lower = -meta['speed_mean']/meta['speed_std']
        a, av = stats(x), stats(xv)
        records = [fit_probe(dest, 'statistics', a, av, y, yv, lower)]
        for family in ['attention', 'mixer']:
            for state in ['random', 'pretrained']:
                for seed in SEEDS:
                    z, zv = features(mouse, family, state, seed, [x, xv])
                    for mode in ['latent', 'augmented']:
                        tx, vx = (z, zv) if mode == 'latent' else (np.concatenate([z,a],1), np.concatenate([zv,av],1))
                        records.append(fit_probe(dest, f'{family}_{state}_{mode}_s{seed}', tx, vx, y, yv, lower))
        rows.append(dict(mouse=mouse, records=records))
        write(dest / 'selection.json', rows[-1])
        print(mouse, '25 probe selections complete', flush=True)
    assert sum(len(r['records']) for r in rows) == 100
    write(ROOT / 'selection_lock.json', dict(locked_utc=datetime.now(timezone.utc).isoformat(), rows=rows,
        hashes={str(p.relative_to(ROOT)): digest(p) for p in ROOT.glob('*/*.npz')}, later_scored=False))


def evaluate():
    verify()
    assert not (ROOT / 'results.json').exists()
    lock = read(ROOT / 'selection_lock.json')
    for name, h in lock['hashes'].items():
        assert digest(ROOT / name) == h
    rows = []
    for locked in lock['rows']:
        mouse = locked['mouse']; dest = ROOT / mouse
        indices = np.load(BASE / mouse / 'columns.npy')
        meta = read(FAIR / mouse / 'metadata.json'); lower = -meta['speed_mean']/meta['speed_std']
        with np.load(FAIR / mouse / 'later_raw.npz') as raw, np.load(FAIR / mouse / 'statistics.npz') as normal:
            sequence = ((raw['activity'][indices,24:]-normal['activity_mean'][indices])/normal['activity_std'][indices]).T.astype(np.float32)
            y = (raw['speed'][55:]-meta['speed_mean'])/meta['speed_std']
        x = np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(sequence,32,axis=0))
        a = stats(x); values = {'statistics':a}
        for family in ['attention','mixer']:
            for state in ['random','pretrained']:
                for seed in SEEDS:
                    z = features(mouse,family,state,seed,[x])[0]
                    values[f'{family}_{state}_latent_s{seed}'] = z
                    values[f'{family}_{state}_augmented_s{seed}'] = np.concatenate([z,a],1)
        predictions = {}; scores = {}
        for record in locked['records']:
            label = record['label']
            with np.load(dest / f'{label}.npz') as fit:
                errors = [mse(v,fit['target'],lower) for v in fit['selection_predictions']]
                np.testing.assert_allclose(errors,[c['mse'] for c in record['candidates']],rtol=1e-12,atol=1e-12)
                assert int(np.argmin(errors)) == record['selected_index']
                predictions[label] = np.einsum('ij,j->i',(values[label]-fit['center'])/fit['scale'],fit['weight'],optimize=False)+fit['mean']
        with np.load(BASE / mouse / 'later_predictions.npz') as old:
            np.testing.assert_array_equal(y,old['target'])
            for key in ['ridge']+[f'{arm}_s{s}' for arm in ['attention_scratch','attention_pretrained','mixer_pretrained','pooled_mlp'] for s in SEEDS]:
                predictions['reference_'+key] = old[key]
        for label,pred in predictions.items():
            scores[label] = mse(pred,y,lower)
            direct = sum((max(float(p),lower)-float(t))**2 for p,t in zip(pred,y))/len(y)
            assert abs(scores[label]-direct) < 1e-10*max(1.,direct)
        np.savez_compressed(dest / 'later_predictions.npz',target=y,**predictions)
        rows.append(dict(mouse=mouse,n=len(y),lower=lower,scores=scores))
        print(mouse,'later scoring complete',flush=True)
    verify()
    write(ROOT / 'results.json',dict(rows=rows))
    write(ROOT / 'audit.json',dict(passed=True,encoder_updates=0,probes=100,linear_solves_and_selection_scores_checked=600,checkpoint_state_unchanged=True,gradients_disabled=True,linear_optimality_and_independent_prediction_checks=True,all_choices_locked_before_later_inference=True,exact_archived_target_alignment=True,independent_later_metrics=True,source_checkpoint_input_and_application_hashes_unchanged=True))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        {'freeze':freeze,'fit':fit,'evaluate':evaluate}[sys.argv[1]]()
