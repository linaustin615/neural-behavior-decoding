"""Repair only audit-side float64 clipping; frozen fit/evaluation and scores unchanged."""
from run import ROOT, OLD, MICE, SEEDS, WARMUP, read, save, utc, verify, digest, load, np


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
                p = np.clip(a[key].astype(np.float64), lower, None); e = p-y
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
                errs = [np.clip(a[f'{candidate}_{s}'].astype(np.float64), lower, None)-y for s in SEEDS]
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
    audit()
