"""Chronological feasibility test for decoding concurrent speed differences."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent
REPO = EXP.parent
BASE = EXP / '2026-10-03_dynamics_baseline'
MICE = ['MP030', 'MP032', 'MP033', 'MP034']
LAMBDAS = [.001, .01, .1, 1., 10.]
ARMS = ['aligned', 'misaligned']
torch.set_num_threads(2)
torch.set_num_interop_threads(1)


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def features(x):
    assert x.ndim == 3 and x.shape[1:] == (128, 32)
    return np.ascontiguousarray(x.reshape(len(x), -1), dtype=np.float64)


def arrays(mouse, segment, start, stop):
    x = np.load(BASE / mouse / f'{segment}_x.npy', mmap_mode='r')
    y = np.load(BASE / mouse / f'{segment}_y.npy', mmap_mode='r')
    assert len(x) == len(y) and 1 <= start < stop <= len(y)
    target = np.asarray(y[start:stop] - y[start-1:stop-1], dtype=np.float64)
    z = features(x[start:stop])
    assert z.shape == (len(target), 4096) and np.isfinite(z).all() and np.isfinite(target).all()
    return z, target


def scores(prediction, target):
    error = prediction - target
    return dict(mse=float(np.mean(error ** 2)), mae=float(np.mean(np.abs(error))),
                signed_error=float(np.mean(error)))


def selfcheck():
    x = np.arange(4 * 128 * 32, dtype=np.float64).reshape(4, 128, 32)
    y = np.array([3., 5., 4., 9.])
    np.testing.assert_array_equal(np.diff(y), [2., -1., 5.])
    z = features(x[1:])
    assert z.shape == (3, 4096)
    np.testing.assert_array_equal(z[0].reshape(128, 32), x[1])
    a = torch.tensor([[1., 2.], [2., 1.], [4., 3.], [3., 6.]], dtype=torch.float64)
    b = torch.tensor([1., -2., 3., 2.], dtype=torch.float64)
    a = a - a.mean(0)
    b = b - b.mean()
    lam = .1
    eig, u = torch.linalg.eigh(a @ a.T)
    dual = u @ ((u.T @ b) / (eig.clamp_min(0) + len(a) * lam))
    weight = a.T @ dual
    primal = torch.linalg.solve(a.T @ a + len(a) * lam * torch.eye(a.shape[1], dtype=a.dtype), a.T @ b)
    torch.testing.assert_close(weight, primal, rtol=1e-12, atol=1e-12)
    return dict(passed=True, signed_consecutive_targets=True, current_neural_window_alignment=True,
                neural_only_feature_function=True, spectral_ridge_matches_primal=True)


def freeze():
    selfcheck()
    bounds = {}
    paths = [Path(__file__)] + [REPO / f for f in ['model.py', 'train.py', 'data.py']]
    for mouse in MICE:
        nt = len(np.load(BASE / mouse / 'train_y.npy', mmap_mode='r'))
        nv = len(np.load(BASE / mouse / 'selection_y.npy', mmap_mode='r'))
        half = nv // 2
        bounds[mouse] = dict(train=[1, nt], tune=[1, half], score=[half + 32, nv])
        assert half + 32 < nv and (half + 32) - 31 > half - 1
        paths += [BASE / mouse / f'{segment}_{kind}.npy' for segment in ['train', 'selection'] for kind in ['x', 'y']]
    write(ROOT / 'selfcheck.json', selfcheck())
    write(ROOT / 'protocol.json', dict(frozen_utc=datetime.now(timezone.utc).isoformat(), mice=MICE,
        question='Does a simple neural-only decoder predict concurrent one-bin speed change well enough to justify a speed-change supervision experiment?',
        data='unchanged128 neurons x32 neural bins; full raw flattened4096 features; original training-only normalization; no behavioral input or coordinates',
        target='signed difference speed[t]-speed[t-1] in training-standardized speed units; current neural history through t is available; concurrent decoding, not forecasting or physical acceleration',
        bounds=bounds, first_sample='omit first target in each independently stored segment; never difference across segment gaps',
        fit='per-mouse ridge on complete original training prefix, training-only feature and target centering, objective mean squared error +lambda*weight norm squared; unpenalized intercept',
        lambdas=LAMBDAS, selection='choose each arm lambda by first development half MSE; ties use first grid entry; no refit; lock choices before scoring second half after32-window gap',
        control='fit identical ridge candidates with training change targets circularly shifted by floor(train_count/2); select its lambda separately against true tune labels; one fixed control, not a permutation significance test',
        references=['zero predicted change', 'training mean change'],
        budget='4 shared eigendecompositions; 4mice x2target alignments x5regularizers =40 candidate ridge solutions; zero neural fits and zero neural inference',
        practical_gate='aligned ridge >=5% equal-mouse mean relative MSE gain vszero and >=3/4 mouse wins, no mouse >10% harm; positive mean gain vs misaligned control with >=3/4 mouse wins',
        secondary='report MAE, all per-mouse errors, selected penalties, prediction-target correlation, raw denominators, leave-one-mouse-out means; MAE is not a gate',
        stopping='if gate fails, do not launch the neural auxiliary-loss experiment or expand features/lags/regularizers to rescue it; if gate passes, use it only as exploratory feasibility, not transformer superiority',
        limitations='historically reused cohort and earlier model-development data; current new ridge weights never fit score labels, but this is not independent confirmation; four mice, not windows, are replication units',
        final_evaluation='do not open later predictions or raw final evaluation recordings in this study',
        input_hashes={str(p.relative_to(REPO)): digest(p) for p in paths}))
    print(json.dumps(dict(frozen=True, bounds=bounds, candidate_fits=40), indent=2), flush=True)


def verify():
    p = read(ROOT / 'protocol.json')
    for name, value in p['input_hashes'].items():
        assert digest(REPO / name) == value, name
    return p


def fit_all():
    protocol = verify()
    assert not (ROOT / 'selection_lock.json').exists()
    records = []
    for mouse in MICE:
        start_time = time.monotonic()
        out = ROOT / mouse
        out.mkdir()
        bounds = protocol['bounds'][mouse]
        x, y = arrays(mouse, 'train', *bounds['train'])
        xv, yv = arrays(mouse, 'selection', *bounds['tune'])
        center = x.mean(0)
        z = torch.from_numpy(x - center)
        zv = torch.from_numpy(xv - center)
        targets = np.column_stack([y, np.roll(y, len(y) // 2)])
        target_mean = targets.mean(0)
        target = torch.from_numpy(targets - target_mean)
        gram = z @ z.T
        eigenvalues, u = torch.linalg.eigh(gram)
        assert float(eigenvalues.min()) > -1e-8 * max(float(eigenvalues.max()), 1.)
        eigenvalues = eigenvalues.clamp_min(0)
        projected = u.T @ target
        candidate_weights, candidate_predictions, residuals = [], [], []
        for lam in LAMBDAS:
            alpha = u @ (projected / (eigenvalues[:, None] + len(z) * lam))
            residual = gram @ alpha + len(z) * lam * alpha - target
            relative_residual = float(residual.norm() / target.norm().clamp_min(1e-15))
            assert relative_residual < 1e-7
            weight = z.T @ alpha
            pred = zv @ weight + torch.from_numpy(target_mean)
            independent = np.einsum('ij,jk->ik', xv-center, weight.numpy(), optimize=False) + target_mean
            np.testing.assert_allclose(pred.numpy(), independent, rtol=1e-9, atol=1e-10)
            candidate_weights.append(weight.numpy())
            candidate_predictions.append(pred.numpy())
            residuals.append(relative_residual)
        weights = np.stack(candidate_weights)
        predictions = np.stack(candidate_predictions)
        choices = {}
        for i, arm in enumerate(ARMS):
            values = [scores(predictions[k, :, i], yv) for k in range(len(LAMBDAS))]
            selected = int(np.argmin([v['mse'] for v in values]))
            choices[arm] = dict(index=selected, regularizer=LAMBDAS[selected], tune_scores=values)
            direct_alpha = torch.linalg.solve(gram + len(z) * LAMBDAS[selected] * torch.eye(len(z), dtype=z.dtype), target[:, i])
            np.testing.assert_allclose((z.T @ direct_alpha).numpy(), weights[selected, :, i], rtol=1e-7, atol=1e-9)
        np.savez_compressed(out / 'fit.npz', center=center, target_mean=target_mean, weights=weights,
                            tune_predictions=predictions, tune_target=yv)
        record = dict(mouse=mouse, train_n=len(y), tune_n=len(yv), features=z.shape[1], target_shift=len(y)//2,
                      choices=choices, train_mean=float(y.mean()), train_change_variance=float(np.var(y)),
                      solve_relative_residuals=residuals, elapsed_seconds=time.monotonic()-start_time,
                      fit_sha256=digest(out / 'fit.npz'))
        write(out / 'selection.json', record)
        records.append(record)
        print(json.dumps(dict(mouse=mouse, complete=True, elapsed=record['elapsed_seconds'],
                              penalties={a:v['regularizer'] for a,v in choices.items()})), flush=True)
        del gram, u, z, zv
    verify()
    write(ROOT / 'selection_lock.json', dict(locked_utc=datetime.now(timezone.utc).isoformat(), records=records,
        protocol_sha256=digest(ROOT / 'protocol.json'), source_sha256=digest(Path(__file__)),
        score_labels_used_for_selection=False, candidates=40))
    print('All 40 ridge candidates complete; choices locked before scoring', flush=True)


def relative_gain(a, b):
    assert b > 0
    return 1 - a / b


def summarize(rows, reference):
    gains = [relative_gain(r['scores']['aligned']['mse'], r['scores'][reference]['mse']) for r in rows]
    mae = [relative_gain(r['scores']['aligned']['mae'], r['scores'][reference]['mae']) for r in rows]
    return dict(reference=reference, mean_relative_mse_gain=float(np.mean(gains)), mouse_gains=gains,
                mouse_wins=sum(g > 0 for g in gains), worst_mouse_gain=min(gains),
                mean_relative_mae_gain=float(np.mean(mae)), mae_mouse_gains=mae,
                leave_one_mouse_out=[float(np.mean(np.delete(gains, i))) for i in range(4)])


def evaluate():
    protocol = verify()
    lock = read(ROOT / 'selection_lock.json')
    assert lock['protocol_sha256'] == digest(ROOT / 'protocol.json')
    assert lock['source_sha256'] == digest(Path(__file__))
    assert not (ROOT / 'results.json').exists()
    rows = []
    checks = 0
    for record in lock['records']:
        mouse = record['mouse']
        out = ROOT / mouse
        assert digest(out / 'fit.npz') == record['fit_sha256']
        x, y = arrays(mouse, 'selection', *protocol['bounds'][mouse]['score'])
        with np.load(out / 'fit.npz') as fitted:
            centered = x - fitted['center']
            predictions = dict(zero=np.zeros_like(y), training_mean=np.full_like(y, record['train_mean']))
            for i, arm in enumerate(ARMS):
                weight = fitted['weights'][record['choices'][arm]['index'], :, i]
                pred = np.einsum('ij,j->i', centered, weight, optimize=False) + fitted['target_mean'][i]
                independent = (torch.from_numpy(centered) @ torch.from_numpy(weight)).numpy() + fitted['target_mean'][i]
                np.testing.assert_allclose(pred, independent, rtol=1e-9, atol=1e-10)
                predictions[arm] = pred
        measured = {name: scores(pred, y) for name, pred in predictions.items()}
        for name, pred in predictions.items():
            for metric, power in [('mse', 2), ('mae', 1)]:
                manual = sum(abs(float(a)-float(b)) ** power for a,b in zip(pred,y))/len(y)
                assert abs(manual-measured[name][metric]) < 1e-12*max(1, manual)
                checks += 1
        correlations = {name: float(np.corrcoef(pred,y)[0,1]) if np.var(pred)>1e-15 and np.var(y)>1e-15 else None
                        for name,pred in predictions.items()}
        np.savez_compressed(out / 'score_predictions.npz', target=y, **predictions)
        rows.append(dict(mouse=mouse, n=len(y), scores=measured, correlations=correlations,
                         selected_penalties={a:v['regularizer'] for a,v in record['choices'].items()}))
    contrasts = {name: summarize(rows,name) for name in ['zero','training_mean','misaligned']}
    zero, control = contrasts['zero'], contrasts['misaligned']
    gate = bool(zero['mean_relative_mse_gain'] >= .05 and zero['mouse_wins'] >= 3 and zero['worst_mouse_gain'] >= -.10
                and control['mean_relative_mse_gain'] > 0 and control['mouse_wins'] >= 3)
    verify()
    write(ROOT / 'results.json', dict(rows=rows, comparisons=contrasts, feasibility_gate=gate,
                                      supports_next_neural_pilot=gate))
    write(ROOT / 'audit.json', dict(passed=True, candidate_fits=40, spectral_decompositions=4,
        direct_selected_solve_checks=8, independent_tune_prediction_checks=40, independent_score_prediction_checks=8,
        independent_scalar_metrics=checks, source_input_application_hashes_unchanged=True,
        choices_locked_before_score=True, score_labels_not_used_for_fit_or_selection=True,
        neural_only_features=True, targets_signed_no_clipping=True, first_samples_dropped=True,
        boundary_gap_windows=32, final_evaluation_archives_opened=False,
        selection_lock_sha256=digest(ROOT / 'selection_lock.json')))
    write(ROOT / 'environment.json', dict(python=sys.version, numpy=np.__version__, torch=torch.__version__,
                                          platform=platform.platform(), torch_threads=torch.get_num_threads()))
    print(json.dumps(dict(comparisons=contrasts, feasibility_gate=gate), indent=2), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['freeze']:
        freeze()
    elif sys.argv[1:] == ['fit']:
        fit_all()
    elif sys.argv[1:] == ['evaluate']:
        evaluate()
    else:
        raise SystemExit('usage: run.py freeze|fit|evaluate')
