"""Test a fixed two-coefficient calibration using archived model outputs."""
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent
REPO = EXP.parent
OLD = EXP / '2026-10-03_shared_behavior'
NEW = EXP / '2026-10-04_ensemble_seed_replication'
ROBUST = EXP / '2026-10-04_behavior_robustness'
BASE = EXP / '2026-10-03_dynamics_baseline'
MICE = ['MP030', 'MP032', 'MP033', 'MP034']
SEEDS = list(range(10, 16))
FAMILIES = ['attention', 'mlp']
PAIRS = list(itertools.combinations(range(6), 2))


def read(path):
    return json.loads(path.read_text())


def write(name, value):
    path = ROOT / name
    assert not path.exists(), path
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def directory(family, seed):
    return (OLD / 'shared' if seed < 13 else NEW) / f'{family}_s{seed}'


def pair_predictions(members):
    return np.stack([(members[i] + members[j]) / 2 for i, j in PAIRS])


def fit(x, y):
    assert x.shape == y.shape and x.ndim == 1 and np.isfinite(x).all() and np.isfinite(y).all()
    xm, ym = float(x.mean()), float(y.mean())
    xc, yc = x - xm, y - ym
    variance = float(np.mean(xc ** 2))
    covariance = float(np.mean(xc * yc))
    slope = max(0., covariance / variance) if variance > 1e-15 else 1.
    intercept = ym - slope * xm
    residual = slope * x + intercept - y
    assert abs(float(residual.mean())) < 1e-10
    gradient = float(np.mean(residual * xc))
    assert abs(gradient) < 1e-10 if slope > 0 else gradient >= -1e-10
    assert np.mean(residual ** 2) <= np.mean((x - y) ** 2) + 1e-12
    return dict(slope=slope, intercept=intercept, input_variance=variance,
                calibration_mse_before=float(np.mean((x - y) ** 2)),
                calibration_mse_after_unbounded=float(np.mean(residual ** 2)))


def apply(prediction, coefficients, lower):
    return np.maximum(np.array([c['slope'] for c in coefficients])[:, None] * prediction +
                      np.array([c['intercept'] for c in coefficients])[:, None], lower)


def score(p, y):
    error = p - y
    return dict(mse=float(np.mean(error ** 2)), mae=float(np.mean(np.abs(error))),
                pair_mse=np.mean(error ** 2, axis=1).tolist(), pair_mae=np.mean(np.abs(error), axis=1).tolist())


def gain(a, b):
    return 1 - a / b if b > 1e-15 else None


def correlation(x, y):
    xc, yc = x - x.mean(), y - y.mean()
    denominator = float(np.sqrt(np.sum(xc ** 2) * np.sum(yc ** 2)))
    return float(np.sum(xc * yc) / denominator) if denominator > 1e-15 else None


def diagnostics(p, y, cuts):
    masks = dict(low=y <= cuts['median'], typical=(y > cuts['median']) & (y <= cuts['q90']), high=y > cuts['q90'])
    change = np.diff(y)
    masks['rapid_rise'] = np.r_[False, change > cuts['change_q90']]
    masks['rapid_fall'] = np.r_[False, change < -cuts['change_q90']]
    output = {}
    for label, mask in masks.items():
        n = int(mask.sum())
        residual = p[:, mask] - y[mask]
        output[label] = dict(n=n, supported=n >= 30,
                             signed_error=float(residual.mean()) if n else None,
                             mean_pair_bias_squared=float(np.mean(residual.mean(1) ** 2)) if n else None,
                             mse=float(np.mean(residual ** 2)) if n else None)
    dp = np.diff(p, axis=1)
    reference = float(np.mean(change ** 2))
    output['consecutive_changes'] = dict(n=len(change), mse=float(np.mean((dp - change) ** 2)),
        no_change_mse=reference, gain_vs_no_change=gain(float(np.mean((dp - change) ** 2)), reference),
        mean_pair_correlation=float(np.mean([correlation(q, change) for q in dp])) if np.var(change) > 1e-15 and np.all(np.var(dp, axis=1) > 1e-15) else None,
        rms_prediction_to_target=float(np.sqrt(np.mean(dp ** 2) / reference)) if reference > 1e-15 else None)
    return output


def selfcheck():
    x = np.arange(8, dtype=np.float64)
    c = fit(x, 2 * x + 3)
    assert c['slope'] == 2. and c['intercept'] == 3.
    c = fit(x, -x)
    assert c['slope'] == 0. and c['intercept'] == -3.5
    c = fit(np.ones(8), x)
    assert c['slope'] == 1. and c['intercept'] == 2.5
    np.testing.assert_array_equal(apply(np.array([[-2., 3.]]), [dict(slope=1., intercept=-1.)], 0.), [[0., 2.]])
    z = pair_predictions(np.arange(12, dtype=np.float64).reshape(6, 2))
    np.testing.assert_array_equal(z[0], [1., 2.])
    assert z.shape == (15, 2)
    assert score(np.array([[1., 1.], [-1., -1.]]), np.zeros(2))['mse'] == 1.
    return dict(passed=True, exact_linear_recovery=True, nonnegative_slope=True, constant_predictor=True,
                physical_floor=True, distinct_seed_pairs=True, average_error_not_error_of_average=True)


def freeze():
    cuts = read(ROBUST / 'protocol.json')['thresholds']
    paths = [Path(__file__), ROBUST / 'protocol.json', ROBUST / 'results.json']
    paths += [REPO / p for p in ['model.py', 'train.py', 'data.py']]
    bounds = {}
    for mouse in MICE:
        target = BASE / mouse / 'selection_y.npy'
        n = len(np.load(target, mmap_mode='r'))
        half = n // 2
        bounds[mouse] = dict(calibration=[0, half], development_check=[half + 32, n])
        assert half + 32 < n
        paths += [target, NEW / mouse / 'later_predictions.npz']
    for family in FAMILIES:
        for seed in SEEDS:
            paths += [directory(family, seed) / f for f in ['result.json', 'selection_predictions.npz']]
    write('selfcheck.json', selfcheck())
    write('protocol.json', dict(frozen_utc=datetime.now(timezone.utc).isoformat(), mice=MICE, seeds=SEEDS,
        question='Can earlier-fitted output scale and offset reduce the shared transformer ensemble errors without changing its neural representation?',
        hypotheses=['stable amplitude/offset error is repairable by two scalar coefficients',
                    'signed errors during fast increases/decreases can describe smoothing, but cannot establish physical latency'],
        families=FAMILIES, seed_pairs=[[SEEDS[i], SEEDS[j]] for i, j in PAIRS], bounds=bounds, thresholds=cuts,
        base_checkpoints='use the original full-development selected epoch in each result.json; no checkpoint reselection',
        calibration='one nonnegative-slope least-squares affine map per mouse/family/pair, fitted on first half of development targets only; zero predictor variance uses slope1 and mean residual offset; clip output at physical zero',
        evaluation='lock all120 coefficient pairs before scoring later-development remainder or archived later intervals; no refit or calibrate/identity selection',
        primary='calibrated transformer versus its unchanged pair: >=5% equal-mouse relative MSE gain, >=3/4 mouse wins, >=40/60 matched pair wins, no mouse >10% MSE harm, nonnegative mean relative MAE gain',
        secondary='apply identical correction to MLP; report same gate for MLP and calibrated-transformer versus calibrated-MLP comparison without reusing gate as an attention-utility claim',
        diagnostics='signed and squared errors at frozen relative-speed cutoffs and rapid rises/falls; first target omitted only for consecutive-change calculations; report each pair prediction-change error versus zero predicted change',
        independence='historically reused animals, architecture and later labels; base epochs saw the full development interval, so its remainder is NOT a pristine holdout even though calibrators did not fit its labels; pairs/windows are dependent',
        stopping='one120-fit analytic calibration experiment; no model fits/inference, search, timing shift, new seeds, application edit or publication',
        input_hashes={str(p.relative_to(REPO)): digest(p) for p in paths}))
    print(json.dumps(dict(frozen=True, bounds=bounds, analytic_fits=120, base_fits=0)))


def verify():
    p = read(ROOT / 'protocol.json')
    for name, value in p['input_hashes'].items():
        assert digest(REPO / name) == value, name
    return p


def development(mouse, family, lower):
    members = []
    expected = np.load(BASE / mouse / 'selection_y.npy')
    epochs = []
    for seed in SEEDS:
        folder = directory(family, seed)
        epoch = read(folder / 'result.json')['selected_epoch']
        with np.load(folder / 'selection_predictions.npz') as z:
            np.testing.assert_array_equal(z[mouse + '_target'], expected)
            members.append(np.maximum(z[mouse + '_predictions'][epoch].astype(np.float64), lower))
        epochs.append(epoch)
    return pair_predictions(np.stack(members)), expected, epochs


def calibrate():
    p = verify()
    selfcheck()
    rows = []
    for mouse in MICE:
        start, stop = p['bounds'][mouse]['calibration']
        for family in FAMILIES:
            predictions, y, epochs = development(mouse, family, p['thresholds'][mouse]['lower'])
            coefficients = [fit(q[start:stop], y[start:stop]) for q in predictions]
            rows.append(dict(mouse=mouse, family=family, epochs=epochs, n=stop-start, coefficients=coefficients))
    assert sum(len(r['coefficients']) for r in rows) == 120
    write('calibration_lock.json', dict(locked_utc=datetime.now(timezone.utc).isoformat(), rows=rows,
                                      protocol_sha256=digest(ROOT / 'protocol.json'), source_sha256=digest(Path(__file__)),
                                      scored_development_remainder=False, scored_later=False))
    print('Locked 120 affine calibrators; no evaluation scores computed by this stage')


def compare(rows, a, b):
    gains = [gain(r['scores'][a]['mse'], r['scores'][b]['mse']) for r in rows]
    mae = [gain(r['scores'][a]['mae'], r['scores'][b]['mae']) for r in rows]
    wins = sum(x < y for r in rows for x, y in zip(r['scores'][a]['pair_mse'], r['scores'][b]['pair_mse']))
    return dict(mean_relative_mse_gain=float(np.mean(gains)), mouse_gains=gains, mouse_wins=sum(g > 0 for g in gains),
                matched_pair_wins=wins, matched_pair_count=60, mean_relative_mae_gain=float(np.mean(mae)),
                mae_mouse_gains=mae, leave_one_mouse_out=[float(np.mean(np.delete(gains, i))) for i in range(4)],
                practical_calibration_gate=bool(np.mean(gains) >= .05 and sum(g > 0 for g in gains) >= 3 and wins >= 40 and min(gains) >= -.10 and np.mean(mae) >= 0) if a == b + '_calibrated' else None)


def evaluate():
    p = verify()
    locked = read(ROOT / 'calibration_lock.json')
    assert locked['protocol_sha256'] == digest(ROOT / 'protocol.json')
    assert locked['source_sha256'] == digest(Path(__file__))
    coeffs = {(r['mouse'], r['family']): r['coefficients'] for r in locked['rows']}
    prior = {r['mouse']: r for r in read(ROBUST / 'results.json')['rows']}
    scalar_checks = 0
    output = {}
    for stage in ['development_check', 'later']:
        rows = []
        for mouse in MICE:
            cuts = p['thresholds'][mouse]
            pred = {}
            for family in FAMILIES:
                if stage == 'development_check':
                    q, y, _ = development(mouse, family, cuts['lower'])
                    start, stop = p['bounds'][mouse][stage]
                    q, y = q[:, start:stop], y[start:stop]
                else:
                    with np.load(NEW / mouse / 'later_predictions.npz') as z:
                        y = z['target'].astype(np.float64)
                        q = pair_predictions(np.stack([np.maximum(z[f'{family}_s{s}'].astype(np.float64), cuts['lower']) for s in SEEDS]))
                pred[family] = q
                pred[family + '_calibrated'] = apply(q, coeffs[(mouse, family)], cuts['lower'])
                assert np.isfinite(pred[family + '_calibrated']).all()
            scores = {name: score(q, y) for name, q in pred.items()}
            for name, q in pred.items():
                for metric, power in [('mse', 2), ('mae', 1)]:
                    manual = [sum(abs(float(a) - float(b)) ** power for a, b in zip(pair, y)) / len(y) for pair in q]
                    np.testing.assert_allclose(manual, scores[name]['pair_' + metric], rtol=1e-12, atol=1e-14)
                    scalar_checks += len(manual)
            if stage == 'later':
                for family in FAMILIES:
                    np.testing.assert_allclose(scores[family]['pair_mse'], prior[mouse]['scores']['all'][family]['pair_mse'], rtol=1e-12, atol=1e-14)
            rows.append(dict(mouse=mouse, n=len(y), scores=scores,
                             diagnostics={name: diagnostics(q, y, cuts) for name, q in pred.items()}))
        comparisons = {a + '_vs_' + b: compare(rows, a, b) for a, b in [
            ('attention_calibrated', 'attention'), ('mlp_calibrated', 'mlp'),
            ('attention_calibrated', 'mlp_calibrated'), ('attention_calibrated', 'mlp')]}
        output[stage] = dict(rows=rows, comparisons=comparisons)
    verify()
    write('results.json', output)
    write('audit.json', dict(passed=True, analytic_fits=120, base_model_fits=0, model_inferences=0,
        metric_scalar_checks=scalar_checks, archived_pair_mse_matches=120,
        calibration_first_development_half_only=True, all_coefficients_locked_before_scoring=True,
        original_selected_epochs_unchanged=True, exact_development_target_alignment=True,
        matched_family_and_pair_budgets=True, input_source_application_hashes_unchanged=True,
        calibration_lock_sha256=digest(ROOT / 'calibration_lock.json')))
    write('environment.json', dict(python=sys.version, numpy=np.__version__, platform=platform.platform()))
    print(json.dumps({s: v['comparisons'] for s, v in output.items()}, indent=2))


if __name__ == '__main__':
    command = sys.argv[1:]
    if command == ['freeze']:
        freeze()
    elif command == ['calibrate']:
        calibrate()
    elif command == ['evaluate']:
        evaluate()
    else:
        raise SystemExit('usage: run.py freeze|calibrate|evaluate')
