"""Fixed, descriptive slices of archived ensemble errors; no fits or inference."""
import hashlib
import itertools
import json
import platform
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
EXP = ROOT.parent
REPLICATION = EXP / '2026-10-04_ensemble_seed_replication'
MICE = ['MP030', 'MP032', 'MP033', 'MP034']
SEEDS = list(range(10, 16))
PAIRS = list(itertools.combinations(range(len(SEEDS)), 2))
SPEED_STATES = ['low_speed', 'typical_speed', 'high_speed']
CHANGE_STATES = ['ordinary_change', 'rapid_change']
QUARTERS = ['quarter_1', 'quarter_2', 'quarter_3', 'quarter_4']
MIN_SUPPORT = 30


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    assert not path.exists(), f'refuse to replace {path}'
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return str(path.relative_to(REPO))


def thresholds(y):
    assert y.ndim == 1 and len(y) > 1 and np.isfinite(y).all()
    return dict(median=float(np.quantile(y, .5)), q90=float(np.quantile(y, .9)),
                change_q90=float(np.quantile(np.abs(np.diff(y)), .9)))


def masks(y, cuts):
    n = len(y)
    changes = np.r_[False, np.abs(np.diff(y)) > cuts['change_q90']]
    eligible = np.arange(n) > 0
    result = dict(all=np.ones(n, dtype=bool), low_speed=y <= cuts['median'],
                  typical_speed=(y > cuts['median']) & (y <= cuts['q90']),
                  high_speed=y > cuts['q90'], ordinary_change=eligible & ~changes,
                  rapid_change=eligible & changes)
    for label, indices in zip(QUARTERS, np.array_split(np.arange(n), 4)):
        result[label] = np.isin(np.arange(n), indices)
    return result


def family_predictions(z, family, lower):
    bounded = np.stack([np.maximum(z[f'{family}_s{s}'].astype(np.float64), lower) for s in SEEDS])
    return np.stack([(bounded[i] + bounded[j]) / 2 for i, j in PAIRS])


def metrics(predictions, y, mask):
    count = int(mask.sum())
    if not count:
        return dict(n=0, mse=None, mae=None, signed_bias=None, pair_mse=[], pair_mae=[])
    residual = predictions[:, mask] - y[mask]
    pair_mse = np.mean(residual ** 2, axis=1)
    pair_mae = np.mean(np.abs(residual), axis=1)
    pair_bias = np.mean(residual, axis=1)
    bias_squared = float(np.mean(pair_bias ** 2))
    centered = float(np.mean((residual - pair_bias[:, None]) ** 2))
    mse = float(pair_mse.mean())
    assert np.isclose(mse, bias_squared + centered, rtol=1e-12, atol=1e-14)
    target_variance = float(np.var(y[mask]))
    return dict(n=count, mse=mse, mae=float(pair_mae.mean()), signed_bias=float(pair_bias.mean()),
                mean_pair_bias_squared=bias_squared, centered_residual_variance=centered,
                r2=(1 - mse / target_variance) if target_variance > 1e-15 else None,
                target_variance=target_variance, pair_mse=pair_mse.tolist(), pair_mae=pair_mae.tolist())


def contrast(a, b, metric):
    return 1 - a[metric] / b[metric] if a[metric] is not None and b[metric] is not None and b[metric] > 1e-15 else None


def selfcheck():
    y = np.array([0., 1., 2., 3., 6.])
    slices = masks(y, dict(median=1., q90=3., change_q90=1.))
    assert np.array_equal(slices['low_speed'], [1, 1, 0, 0, 0])
    assert np.array_equal(slices['typical_speed'], [0, 0, 1, 1, 0])
    assert np.array_equal(slices['high_speed'], [0, 0, 0, 0, 1])
    assert np.array_equal(slices['rapid_change'], [0, 0, 0, 0, 1])
    assert np.array_equal(slices['ordinary_change'], [0, 1, 1, 1, 0])
    z = {f'attention_s{s}': np.array([-2., float(s - 10)]) for s in SEEDS}
    predictions = family_predictions(z, 'attention', -1.)
    assert predictions.shape == (15, 2)
    np.testing.assert_array_equal(predictions[:, 0], -np.ones(15))
    np.testing.assert_array_equal(predictions[0], [-1., .5])
    result = metrics(np.array([[1., 1.], [-1., -1.]]), np.zeros(2), np.ones(2, bool))
    assert result['mse'] == 1. and result['signed_bias'] == 0. and result['mean_pair_bias_squared'] == 1.
    assert metrics(np.zeros((1, 2)), np.zeros(2), np.zeros(2, bool))['mse'] is None
    return dict(passed=True, boundary_ties=True, first_change_excluded=True,
                individual_clipping_before_averaging=True, pair_errors_not_all_seed_prediction=True,
                average_squared_bias_not_squared_average_bias=True, empty_stratum_defined=True)


def freeze():
    assert not (ROOT / 'protocol.json').exists()
    checks = selfcheck()
    files = [Path(__file__), REPLICATION / 'combined_context.json', REPLICATION / 'attention_context.json']
    files += [REPO / f for f in ['train.py', 'model.py', 'data.py']]
    cuts = {}
    for mouse in MICE:
        train = EXP / '2026-10-03_dynamics_baseline' / mouse / 'train_y.npy'
        meta = EXP / '2026-10-03_fair_comparison' / mouse / 'metadata.json'
        y = np.load(train)
        info = read(meta)
        cuts[mouse] = dict(**thresholds(y), train_n=len(y), lower=-info['speed_mean'] / info['speed_std'],
                           speed_mean=info['speed_mean'], speed_std=info['speed_std'])
        files += [train, meta, REPLICATION / mouse / 'later_predictions.npz',
                  EXP / '2026-10-03_shared_behavior' / mouse / 'later_predictions.npz']
    protocol = dict(
        question='Does the archived transformer-pair advantage persist across speed, speed change, time, and MAE?',
        status='exploratory descriptive diagnosis on historically reused recordings; no new scientific gate',
        frozen_utc=datetime.now(timezone.utc).isoformat(), mice=MICE, seeds=SEEDS,
        pairs=[[SEEDS[i], SEEDS[j]] for i, j in PAIRS], thresholds=cuts, minimum_support=MIN_SUPPORT,
        aggregation='clip each seed at physical zero, average two predictions, score each pair, average pair errors within mouse, then average relative gains equally across mice',
        speed_states='<=training median; >median and <=training90th percentile; >training90th percentile',
        changes='rapid means absolute consecutive target change >training90th percentile; omit first later sample only from change slices',
        time='four contiguous near-equal quarters; also omit each quarter in turn',
        metrics=['MSE', 'MAE', 'whole-interval R2', 'pairwise bias squared plus centered residual variance',
                 'each speed stratum contribution to whole-interval relative MSE gain'],
        references=['matched static-query temporal MLP pairs', 'archived raw ridge',
                    'archived training mean (normalized zero)', 'training median', 'physical zero speed'],
        support='show every count and cell; summaries exclude cells under30 windows and always show eligible mice; no fill-in or threshold changes',
        constraints=['no fitting, inference, architecture changes, seed/weight/pair selection, or application edits',
                     'no confidence intervals or p-values on post-hoc slices; no independent confirmation',
                     'seed pairs, quarters and adjacent windows are dependent; four animals remain the replication units',
                     'relative speed is not verified rest/motion; rapid-change labels use evaluation targets for diagnosis only',
                     'R2 against each interval mean is descriptive, not a deployable target-mean predictor',
                     'ridge comparison is archived context, not matched architecture or compute'],
        input_hashes={relative(p): digest(p) for p in files})
    write(ROOT / 'selfcheck.json', checks)
    write(ROOT / 'protocol.json', protocol)
    print(json.dumps(dict(frozen=True, files=len(files), thresholds=cuts), indent=2))


def run():
    assert not (ROOT / 'results.json').exists()
    protocol = read(ROOT / 'protocol.json')
    for name, value in protocol['input_hashes'].items():
        assert digest(REPO / name) == value, name
    selfcheck()
    old = {row['mouse']: row for row in read(REPLICATION / 'combined_context.json')['rows']}
    rows = []
    scalar_checks = 0
    all_labels = ['all'] + SPEED_STATES + CHANGE_STATES + QUARTERS
    for mouse in MICE:
        cuts = protocol['thresholds'][mouse]
        with np.load(REPLICATION / mouse / 'later_predictions.npz') as z:
            y = z['target'].astype(np.float64)
            predictions = {family: family_predictions(z, family, cuts['lower']) for family in ['attention', 'mlp']}
        with np.load(EXP / '2026-10-03_shared_behavior' / mouse / 'later_predictions.npz') as z:
            np.testing.assert_array_equal(y, z['target'])
            predictions['ridge'] = np.maximum(z['raw_ridge'].astype(np.float64), cuts['lower'])[None, :]
        predictions.update(training_mean=np.zeros((1, len(y))),
                           training_median=np.full((1, len(y)), max(cuts['median'], cuts['lower'])),
                           zero_speed=np.full((1, len(y)), cuts['lower']))
        assert y.ndim == 1 and np.isfinite(y).all()
        assert all(p.ndim == 2 and p.shape[1] == len(y) and np.isfinite(p).all() for p in predictions.values())
        slices = masks(y, cuts)
        for group in [SPEED_STATES, QUARTERS]:
            np.testing.assert_array_equal(sum(slices[s].astype(int) for s in group), np.ones(len(y), int))
        np.testing.assert_array_equal(sum(slices[s].astype(int) for s in CHANGE_STATES), (np.arange(len(y)) > 0).astype(int))
        scores = {label: {name: metrics(p, y, slices[label]) for name, p in predictions.items()} for label in all_labels}
        for family in ['attention', 'mlp']:
            np.testing.assert_allclose(scores['all'][family]['pair_mse'], old[mouse][family + '_pair_pair_mse'], rtol=1e-12, atol=1e-14)
        contrasts = {}
        for label in all_labels:
            s = scores[label]
            contrasts[label] = dict(n=int(slices[label].sum()), supported=bool(slices[label].sum() >= MIN_SUPPORT),
                                   mse_gain=contrast(s['attention'], s['mlp'], 'mse'),
                                   mae_gain=contrast(s['attention'], s['mlp'], 'mae'))
            for name, p in predictions.items():
                indices = np.flatnonzero(slices[label]).tolist()
                if not indices:
                    continue
                for metric, power in [('mse', 2), ('mae', 1)]:
                    manual = [sum(abs(float(pred[t]) - float(y[t])) ** power for t in indices) / len(indices) for pred in p]
                    np.testing.assert_allclose(manual, s[name]['pair_' + metric], rtol=1e-12, atol=1e-14)
                    scalar_checks += len(manual)
        contributions = {}
        denominator = scores['all']['mlp']['mse']
        for label in SPEED_STATES:
            n = int(slices[label].sum())
            contributions[label] = (n / len(y) * (scores[label]['mlp']['mse'] - scores[label]['attention']['mse']) / denominator) if n else 0.
        assert abs(sum(contributions.values()) - contrasts['all']['mse_gain']) < 1e-12
        omitted_quarters = {}
        for label in QUARTERS:
            a = metrics(predictions['attention'], y, ~slices[label])
            b = metrics(predictions['mlp'], y, ~slices[label])
            omitted_quarters[label] = {k: contrast(a, b, k) for k in ['mse', 'mae']}
        baseline_gains = {name: {metric: contrast(scores['all']['attention'], scores['all'][name], metric)
                                for metric in ['mse', 'mae']} for name in predictions if name != 'attention'}
        decomposition = {name: (scores['all']['mlp'][name] - scores['all']['attention'][name]) / denominator
                         for name in ['mean_pair_bias_squared', 'centered_residual_variance']}
        assert abs(sum(decomposition.values()) - contrasts['all']['mse_gain']) < 1e-12
        rows.append(dict(mouse=mouse, n=len(y), thresholds=cuts, scores=scores, contrasts=contrasts,
                         speed_contributions=contributions, omit_quarter_gains=omitted_quarters,
                         baseline_gains=baseline_gains, gain_decomposition=decomposition))
    summaries = {}
    for label in all_labels:
        eligible = [r for r in rows if r['contrasts'][label]['supported']]
        summaries[label] = dict(eligible_mice=[r['mouse'] for r in eligible], **{
            metric: dict(mean_relative_gain=float(np.mean([r['contrasts'][label][metric + '_gain'] for r in eligible])) if eligible else None,
                         mouse_wins=sum(r['contrasts'][label][metric + '_gain'] > 0 for r in eligible))
            for metric in ['mse', 'mae']})
    contributions = {label: float(np.mean([r['speed_contributions'][label] for r in rows])) for label in SPEED_STATES}
    baseline_summary = {name: {metric: dict(mean_relative_gain=float(np.mean([r['baseline_gains'][name][metric] for r in rows])),
                                            mouse_wins=sum(r['baseline_gains'][name][metric] > 0 for r in rows))
                               for metric in ['mse', 'mae']} for name in rows[0]['baseline_gains']}
    omissions = {label: {metric: float(np.mean([r['omit_quarter_gains'][label][metric] for r in rows]))
                         for metric in ['mse', 'mae']} for label in QUARTERS}
    write(ROOT / 'results.json', dict(rows=rows, summaries=summaries, speed_contributions=contributions,
                                      baseline_summary=baseline_summary, omit_quarter_gains=omissions))
    for name, value in protocol['input_hashes'].items():
        assert digest(REPO / name) == value, name
    write(ROOT / 'audit.json', dict(passed=True, independent_scalar_scores=scalar_checks,
                                    archived_pair_mse_matches=120, target_alignment=True,
                                    partitions_exact=True, error_and_gain_decompositions=True,
                                    input_hashes_unchanged=True, no_fits=True, no_model_inference=True))
    write(ROOT / 'environment.json', dict(python=sys.version, numpy=np.__version__, platform=platform.platform()))
    print(json.dumps(dict(summaries=summaries, speed_contributions=contributions,
                          baseline_summary=baseline_summary, omit_quarter_gains=omissions), indent=2))


if __name__ == '__main__':
    if sys.argv[1:] == ['freeze']:
        freeze()
    elif sys.argv[1:] == ['run']:
        run()
    else:
        raise SystemExit('usage: analyse.py freeze|run')
