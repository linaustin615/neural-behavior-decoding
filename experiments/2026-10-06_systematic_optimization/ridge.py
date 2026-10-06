"""Chronologically tune a strong standardized linear-regression baseline."""
import numpy as np
import torch
import data
from common import ROOT, MICE, read, write, digest, now
from fit import mse

LAMBDAS = [10.**i for i in range(-4, 4)]


def fit_path(x, y, xv, lambdas):
    x = torch.as_tensor(x.reshape(len(x), -1), dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    xv = torch.as_tensor(xv.reshape(len(xv), -1), dtype=torch.float64)
    center = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(1e-6)
    x = (x-center)/scale
    xv = (xv-center)/scale
    mean = y.mean()
    target = y-mean
    gram = x@x.T
    values, vectors = torch.linalg.eigh(gram)
    projection = vectors.T@target
    predictions, states, checks = [], [], []
    for lam in lambdas:
        penalty = lam*len(x)
        dual = vectors@(projection/(values+penalty))
        residual = float(torch.linalg.vector_norm(gram@dual+penalty*dual-target)/torch.linalg.vector_norm(target).clamp_min(1e-12))
        assert residual < 1e-6 and torch.isfinite(dual).all()
        weight = x.T@dual
        p = (xv@weight+mean).numpy()
        independent = np.einsum('ij,j->i', xv.numpy(), weight.numpy(), optimize=False)+float(mean)
        np.testing.assert_allclose(p, independent, rtol=1e-7, atol=1e-7)
        assert np.isfinite(p).all()
        predictions.append(p)
        states.append(dict(center=center.numpy(), scale=scale.numpy(), weight=weight.numpy(), mean=mean.numpy()))
        checks.append(residual)
    return predictions, states, checks


def search():
    rows = []
    for history in [16, 32, 64]:
        for fold in ['fold0', 'fold1']:
            dataset = data.load(fold, history)
            for s, mouse in enumerate(MICE):
                d = dataset[s]
                predictions, _, checks = fit_path(d['x'].numpy(), d['y'].numpy(), d['xv'].numpy(), LAMBDAS)
                out = ROOT/'ridge_development'/f'{mouse}_{fold}_h{history}.npz'
                out.parent.mkdir(exist_ok=True)
                np.savez_compressed(out, predictions=np.stack(predictions), target=d['yv'])
                for lam, p, residual in zip(LAMBDAS, predictions, checks):
                    error = mse(p, d['yv'], d['lower'])
                    rows.append(dict(mouse=mouse, fold=fold, history=history, lam=lam, mse=error,
                        score=error/d['denominator'], normal_equation_residual=residual))
            del dataset
    selected = {}
    for mouse in MICE:
        options = []
        for h in [16, 32, 64]:
            for lam in LAMBDAS:
                subset = [r for r in rows if r['mouse']==mouse and r['history']==h and r['lam']==lam]
                assert len(subset) == 2
                options.append(dict(history=h, lam=lam, score=float(np.mean([r['score'] for r in subset]))))
        selected[mouse] = min(options, key=lambda v:(v['score'], v['history'], v['lam']))
    assert len(rows) == 192
    write(ROOT/'ridge_selection.json', dict(locked_utc=now(), rows=rows, selected=selected,
        development_solutions=192, all_normal_equations_and_predictions_checked=True,
        prediction_hashes={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'ridge_development').glob('*.npz')}))
    return dict(ridge_complete=True, development_solutions=192)


def final_fit():
    selection = read(ROOT/'ridge_selection.json')
    rows = []
    for s, mouse in enumerate(MICE):
        c = selection['selected'][mouse]
        d = data.load('full', c['history'])[s]
        predictions, states, checks = fit_path(d['x'].numpy(), d['y'].numpy(), d['xv'].numpy(), [c['lam']])
        out = ROOT/'ridge_final'/f'{mouse}.npz'
        out.parent.mkdir(exist_ok=True)
        np.savez_compressed(out, **states[0], selection_prediction=predictions[0], selection_target=d['yv'])
        rows.append(dict(mouse=mouse, config=c, normal_equation_residual=checks[0], path=str(out.relative_to(ROOT)), sha256=digest(out)))
    write(ROOT/'ridge_final_lock.json', dict(locked_utc=now(), rows=rows, final_solutions=4, later_scored=False))


def predict(x, state):
    features = x.reshape(len(x), -1).astype(np.float64)
    z = (features-state['center'])/state['scale']
    result = np.einsum('ij,j->i', z, state['weight'], optimize=False)+float(state['mean'])
    assert np.isfinite(result).all()
    return result
