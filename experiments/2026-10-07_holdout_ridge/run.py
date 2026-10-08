"""Post hoc ridge comparison using frozen holdout inputs and saved neural predictions."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
sys.path.insert(0, str(REPO))
from decoding.config import RIDGE_HISTORIES, RIDGE_PENALTIES, WARMUP
from decoding.data import load
from decoding.ridge import solve, predict
from decoding.train import digest, mse

OLD = ROOT.parent/'2026-10-07_holdout_confirmation'
MICE = ('D3', 'D4', 'D7', 'D9')
SEEDS = (401, 402, 403)


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    with path.open('x') as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write('\n')


def utc():
    return datetime.now(timezone.utc).isoformat()


def receipt(paths):
    return {str(p.relative_to(REPO)): digest(p) for p in sorted(set(paths))}


def verify(hashes):
    for name, expected in hashes.items():
        assert digest(REPO/name) == expected, name


def freeze():
    paths = [Path(__file__), OLD/'summary.json', OLD/'protocol.json', OLD/'amendment.json']
    paths += list((REPO/'decoding').glob('*.py'))
    for mouse in MICE:
        paths += list((OLD/'prepared'/mouse).glob('*'))
        paths += [OLD/'base_locks'/f'{mouse}.json']
        paths += [OLD/'predictions'/f'{mouse}_{seed}.npz' for seed in SEEDS]
    save(ROOT/'protocol.json', dict(created_utc=utc(), question='Does the saved transformer/MLP blend beat tuned ridge on identical holdout frames?',
        status='Post hoc comparison on four now-examined mice, not fresh confirmation', mice=MICE, seeds=SEEDS,
        inputs='Reuse exact prepared panels, training cap4096, chronological splits/gaps, target scaling and frame alignment',
        ridge=dict(histories=RIDGE_HISTORIES, penalties=RIDGE_PENALTIES, objective='mean squared loss + penalty * squared standardized-feature weights; unpenalized intercept',
                   selection='minimum clipped validation MSE; first option on exact ties; train only, no train+validation refit',
                   primary='history and penalty selected on validation', matched_history='history32; penalty selected on validation'),
        evaluation='All four ridge selections locked before new test prediction; neural predictions reused without retraining or reselection',
        endpoint='Per-mouse seed-mean MSE, then equal-mouse mean relative reduction versus deterministic ridge; do not average neural predictions across seeds',
        secondary='MAE, R2, quiet/active MSE, quiet mean predicted speed; physical zero clipping for every model',
        decision='Report magnitude and four mouse directions; no new pass gate, significance claim, or tuning after scoring',
        limits='Ridge history search includes twice the neural history; matched32 also reported. Two neural models versus one ridge, not equal compute. Existing two-frame alignment amendment inherited.',
        source_and_input_sha256=receipt(paths)))


def smoke():
    rng = np.random.default_rng(9127)
    x = rng.normal(size=(29, 47))
    x[:, -1] = 2.
    y = rng.normal(size=29)
    penalties = (.001, .1, 10.)
    state = solve(x, y, penalties)
    z = (x-x.mean(0))/np.maximum(x.std(0), 1e-6)
    errors = []
    for i, penalty in enumerate(penalties):
        gram = np.einsum('ni,nj->ij', z, z, optimize=False)
        rhs = np.einsum('ni,n->i', z, y-y.mean(), optimize=False)
        ref = np.linalg.solve(gram+penalty*len(x)*np.eye(x.shape[1]), rhs)
        np.testing.assert_allclose(state['weights'][:, i], ref, rtol=1e-8, atol=1e-9)
        errors.append(float(np.max(np.abs(state['weights'][:, i]-ref))))
    save(ROOT/'smoke.json', dict(independent_primal_solve=True, constant_feature=True, max_weight_errors=errors))


def fit():
    protocol = read(ROOT/'protocol.json')
    verify(protocol['source_and_input_sha256'])
    assert not (ROOT/'selection_lock.json').exists()
    for mouse in MICE:
        out = ROOT/'fits'/mouse
        out.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        options = []
        for history in RIDGE_HISTORIES:
            train = load(OLD/'prepared'/mouse, 'train', history)
            validation = load(OLD/'prepared'/mouse, 'validation', history)
            x = train.x[train.indices].reshape(len(train.indices), -1)
            state = solve(x, train.y[train.indices], RIDGE_PENALTIES)
            del x
            values = predict(validation, state)
            np.savez(out/f'state_h{history}.npz', **state)
            np.savez_compressed(out/f'validation_h{history}.npz', prediction=values, target=validation.y)
            for index, penalty in enumerate(RIDGE_PENALTIES):
                options.append(dict(history=history, penalty=penalty, column=index,
                    validation_mse=mse(values[:, index], validation.y, train.metadata['lower'])))
            print(mouse, 'history', history, 'complete', round(time.monotonic()-started, 1), 'seconds', flush=True)
            del train, validation, state, values
        save(out/'result.json', dict(mouse=mouse, options=options,
            selected=min(options, key=lambda r: r['validation_mse']),
            matched32=min((r for r in options if r['history']==32), key=lambda r: r['validation_mse']),
            seconds=time.monotonic()-started, test_predictions_computed=False))
    verify(protocol['source_and_input_sha256'])
    paths = list((ROOT/'fits').glob('*/*'))
    save(ROOT/'selection_lock.json', dict(utc=utc(), protocol_sha256=digest(ROOT/'protocol.json'),
        selections={m: read(ROOT/'fits'/m/'result.json') for m in MICE}, artifacts=receipt(paths)))


def metrics(p, y, lower):
    p = np.maximum(np.asarray(p, dtype=np.float64), lower)
    error = p-y
    physical = y-lower
    quiet, active = physical<=.05, physical>=.5
    return dict(mse=float(np.mean(error**2)), mae=float(np.mean(np.abs(error))),
        r2=float(1-np.mean(error**2)/np.var(y)),
        quiet_mse=float(np.mean(error[quiet]**2)) if quiet.any() else None,
        active_mse=float(np.mean(error[active]**2)) if active.any() else None,
        qfm=float(np.mean(p[quiet]-lower)) if quiet.any() else None)


def aggregate(rows):
    results = {}
    for baseline in ('ridge', 'ridge32'):
        for candidate in ('blend', 'transformer', 'mlp'):
            per = []
            for mouse in MICE:
                a = [r for r in rows if r['mouse']==mouse and r['model']==candidate]
                b = next(r for r in rows if r['mouse']==mouse and r['model']==baseline)
                am = float(np.mean([r['mse'] for r in a]))
                per.append(dict(mouse=mouse, candidate_mse=am, ridge_mse=b['mse'], gain=1-am/b['mse'],
                    mae_gain=1-float(np.mean([r['mae'] for r in a]))/b['mae'],
                    seed_wins=sum(r['mse']<b['mse'] for r in a)))
            results[candidate+'_vs_'+baseline] = dict(per_mouse=per,
                mean_gain=float(np.mean([r['gain'] for r in per])), mouse_wins=sum(r['gain']>0 for r in per),
                mean_mae_gain=float(np.mean([r['mae_gain'] for r in per])), seed_wins=sum(r['seed_wins'] for r in per))
    return results


def evaluate():
    protocol = read(ROOT/'protocol.json')
    lock = read(ROOT/'selection_lock.json')
    assert digest(ROOT/'protocol.json')==lock['protocol_sha256']
    verify(protocol['source_and_input_sha256']); verify(lock['artifacts'])
    out = ROOT/'predictions'
    out.mkdir(exist_ok=False)
    rows = []
    for mouse in MICE:
        result = lock['selections'][mouse]
        predictions = {}
        for model, choice in [('ridge', result['selected']), ('ridge32', result['matched32'])]:
            test = load(OLD/'prepared'/mouse, 'test', choice['history'])
            with np.load(ROOT/'fits'/mouse/f"state_h{choice['history']}.npz") as arrays:
                state = dict(arrays)
            state['weights'] = state['weights'][:, choice['column']:choice['column']+1]
            predictions[model] = predict(test, state)[:, 0]
        y, lower = test.y, test.metadata['lower']
        for seed in SEEDS:
            with np.load(OLD/'predictions'/f'{mouse}_{seed}.npz') as old:
                np.testing.assert_array_equal(y, old['target'])
                for model, p in [('blend', old['base']+lower), ('transformer', old['transformer']), ('mlp', old['mlp'])]:
                    predictions[f'{model}_{seed}'] = p.copy()
                    rows.append(dict(mouse=mouse, model=model, seed=seed, **metrics(p, y, lower)))
        predictions.update(zero=np.full(len(y), lower), mean=np.zeros(len(y)))
        for model in ('ridge', 'ridge32', 'zero', 'mean'):
            rows.append(dict(mouse=mouse, model=model, seed=None, **metrics(predictions[model], y, lower)))
        np.savez_compressed(out/f'{mouse}.npz', target=y, lower=lower,
            frames=np.arange(test.metadata['boundaries']['test'][0]+WARMUP, test.metadata['boundaries']['test'][1]), **predictions)
    save(ROOT/'summary.json', dict(utc=utc(), status=protocol['status'], rows=rows, contrasts=aggregate(rows),
        selection_lock_sha256=digest(ROOT/'selection_lock.json')))
    print(json.dumps(read(ROOT/'summary.json')['contrasts'], indent=2), flush=True)


def audit():
    protocol = read(ROOT/'protocol.json'); lock = read(ROOT/'selection_lock.json')
    verify(protocol['source_and_input_sha256']); verify(lock['artifacts'])
    summary = read(ROOT/'summary.json')
    original = read(OLD/'summary.json')['rows']
    checked, prediction_error, selection_count = 0, 0., 0
    for mouse in MICE:
        meta = read(OLD/'prepared'/mouse/'metadata.json')
        lower = meta['lower']
        options = lock['selections'][mouse]['options']
        for option in options:
            with np.load(ROOT/'fits'/mouse/f"validation_h{option['history']}.npz") as a:
                err = np.clip(a['prediction'][:, option['column']], lower, None)-a['target']
                value = float(np.dot(err, err)/len(err))
                np.testing.assert_allclose(value, option['validation_mse'], rtol=1e-12, atol=1e-12)
        for label in ('selected', 'matched32'):
            candidates = options if label=='selected' else [o for o in options if o['history']==32]
            chosen = sorted(candidates, key=lambda o: o['validation_mse'])[0]
            assert chosen==lock['selections'][mouse][label]
            selection_count += 1
        with np.load(ROOT/'predictions'/f'{mouse}.npz') as a:
            y = a['target']; n = len(y)
            np.testing.assert_array_equal(y, np.load(OLD/'prepared'/mouse/'test_y.npy')[WARMUP:])
            np.testing.assert_array_equal(a['frames'], np.arange(meta['boundaries']['test'][0]+WARMUP, meta['boundaries']['test'][1]))
            for model, label in [('ridge', 'selected'), ('ridge32', 'matched32')]:
                choice = lock['selections'][mouse][label]
                test = load(OLD/'prepared'/mouse, 'test', choice['history'])
                idx = np.linspace(0, n-1, 17, dtype=int)
                with np.load(ROOT/'fits'/mouse/f"state_h{choice['history']}.npz") as state:
                    x = (test.x[idx].reshape(len(idx), -1).astype(float)-state['center'])/state['scale']
                    ref = np.einsum('ij,j->i', x, state['weights'][:, choice['column']], optimize=False)+state['mean']
                np.testing.assert_allclose(ref, a[model][idx], rtol=1e-9, atol=1e-9)
                prediction_error = max(prediction_error, float(np.max(np.abs(ref-a[model][idx]))))
            for row in [r for r in summary['rows'] if r['mouse']==mouse]:
                key = row['model'] if row['seed'] is None else f"{row['model']}_{row['seed']}"
                p = np.clip(a[key], lower, None); e = p-y
                assert np.isfinite(p).all() and np.isfinite(y).all()
                quiet = y-lower<=.05; active = y-lower>=.5
                reference = dict(mse=np.dot(e, e)/n, mae=sum(np.abs(e))/n,
                    r2=1-np.dot(e, e)/np.dot(y-y.mean(), y-y.mean()),
                    quiet_mse=np.dot(e[quiet], e[quiet])/quiet.sum() if quiet.any() else None,
                    active_mse=np.dot(e[active], e[active])/active.sum() if active.any() else None,
                    qfm=sum(p[quiet]-lower)/quiet.sum() if quiet.any() else None)
                for name, value in reference.items():
                    if value is None: assert row[name] is None
                    else: np.testing.assert_allclose(row[name], value, rtol=1e-11, atol=1e-12)
                if row['seed'] is not None:
                    old = next(r for r in original if r['recording']==mouse and r['arm']==row['model'] and r['seed']==row['seed'])
                    for name in ('mse', 'mae', 'r2', 'qfm', 'active_mse'):
                        np.testing.assert_allclose(row[name], old[name], rtol=1e-11, atol=1e-12)
                checked += 1
    for name, contrast in summary['contrasts'].items():
        candidate, baseline = name.split('_vs_')
        gains, maes, wins = [], [], 0
        for mouse in MICE:
            with np.load(ROOT/'predictions'/f'{mouse}.npz') as a:
                y, lower = a['target'], float(a['lower'])
                ref = np.clip(a[baseline], lower, None)-y
                errs = [np.clip(a[f'{candidate}_{s}'], lower, None)-y for s in SEEDS]
                mses = [np.dot(e, e)/len(e) for e in errs]
                gains.append(1-sum(mses)/3/(np.dot(ref, ref)/len(ref)))
                maes.append(1-sum(np.abs(e).mean() for e in errs)/3/np.abs(ref).mean())
                wins += sum(v<np.dot(ref, ref)/len(ref) for v in mses)
        np.testing.assert_allclose(contrast['mean_gain'], sum(gains)/4, atol=1e-12)
        np.testing.assert_allclose(contrast['mean_mae_gain'], sum(maes)/4, atol=1e-12)
        assert contrast['mouse_wins']==sum(g>0 for g in gains) and contrast['seed_wins']==wins
    save(ROOT/'audit.json', dict(passed=True, utc=utc(), independent_metric_records=checked,
        selections_checked=selection_count, validation_options_checked=96, contrasts_checked=6,
        max_independent_prediction_error=prediction_error, inputs_and_sources_unchanged=True,
        archived_neural_metrics_match=True, identical_target_frames=True,
        limits='Independent primal solver smoke; full-size normal equations checked by reused solver; sampled saved-checkpoint inference and all scored metrics checked independently'))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'smoke', 'fit', 'evaluate', 'audit'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    globals()[args.stage]()
