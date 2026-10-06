"""Separate additional-seed replication from previously observed seed results."""
import itertools

import numpy as np

import run

GROUPS = ['baseline', 'free_dynamic', 'free_static', 'dynamic', 'static', 'prior_mlp']
SUBSETS = dict(original=[10, 11, 12], additional=[13, 14, 15], all=list(range(10, 16)))
CONTRASTS = dict(dynamic_vs_baseline=('dynamic', 'baseline'),
    dynamic_vs_free_dynamic=('dynamic', 'free_dynamic'), dynamic_vs_static=('dynamic', 'static'),
    static_vs_baseline=('static', 'baseline'), static_vs_free_static=('static', 'free_static'),
    dynamic_vs_prior_mlp=('dynamic', 'prior_mlp'), static_vs_prior_mlp=('static', 'prior_mlp'))


def metrics(raw, target, lower):
    assert raw.shape[1:] == target.shape and np.isfinite(raw).all() and np.isfinite(target).all()
    bounded = np.maximum(raw.astype(np.float64), lower)
    pairs = np.stack([(bounded[i] + bounded[j]) / 2 for i, j in itertools.combinations(range(len(raw)), 2)])
    single_error, pair_error = bounded - target, pairs - target
    result = dict(mse=float(np.mean(pair_error ** 2)), mae=float(np.mean(np.abs(pair_error))),
        pair_mse=np.mean(pair_error ** 2, 1).tolist(), pair_mae=np.mean(np.abs(pair_error), 1).tolist(),
        single_mse=np.mean(single_error ** 2, 1).tolist(), single_mae=np.mean(np.abs(single_error), 1).tolist())
    result['r2'] = 1 - result['mse'] / float(np.var(target)) if np.var(target) > 0 else None
    return result


def summarize(rows, seeds):
    summaries = {}
    for name, (a, b) in CONTRASTS.items():
        gains = [1 - r['scores'][a]['mse'] / r['scores'][b]['mse'] for r in rows]
        mae = [1 - r['scores'][a]['mae'] / r['scores'][b]['mae'] for r in rows]
        single_a = np.array([r['scores'][a]['single_mse'] for r in rows])
        single_b = np.array([r['scores'][b]['single_mse'] for r in rows])
        pair_a = np.array([r['scores'][a]['pair_mse'] for r in rows])
        pair_b = np.array([r['scores'][b]['pair_mse'] for r in rows])
        s = dict(mean_mse_gain=float(np.mean(gains)), mouse_mse_gains=gains, mouse_wins=sum(v > 0 for v in gains),
            mean_mae_gain=float(np.mean(mae)), mouse_mae_gains=mae, seed_wins=int(np.sum(single_a < single_b)),
            pair_wins=int(np.sum(pair_a < pair_b)),
            single_mean_mse_gain=float(np.mean(1 - single_a.mean(1) / single_b.mean(1))),
            leave_one_mouse_out=[float(np.mean(gains[:i] + gains[i + 1:])) for i in range(4)],
            individual_seeds=[dict(seed=seed, mean_mse_gain=float(np.mean(1 - single_a[:, i] / single_b[:, i])),
                mouse_wins=int(np.sum(single_a[:, i] < single_b[:, i]))) for i, seed in enumerate(seeds)])
        s['contrast_passed'] = s['mean_mse_gain'] >= .05 and s['mouse_wins'] >= 3 and s['seed_wins'] >= int(np.ceil(8 * len(seeds) / 3))
        summaries[name] = s
    gates = {}
    for variant in ['dynamic', 'static']:
        base = summaries[variant + '_vs_baseline']
        gates[variant + '_no_mouse_over10pct_harm'] = min(base['mouse_mse_gains']) >= -.1
        gates[variant + '_utility'] = (base['contrast_passed'] and summaries[variant + '_vs_free_' + variant]['contrast_passed']
            and gates[variant + '_no_mouse_over10pct_harm'])
    gates['dynamic_routing'] = gates['dynamic_utility'] and summaries['dynamic_vs_static']['contrast_passed']
    return summaries, gates


def independent_check(raw, target, lower, result):
    counts = 0
    for power, key in [(2, 'mse'), (1, 'mae')]:
        singles = [sum(abs(max(float(p), lower) - float(t)) ** power for p, t in zip(row, target)) / len(target) for row in raw]
        pairs = [sum(abs((max(float(a), lower) + max(float(b), lower)) / 2 - float(t)) ** power
            for a, b, t in zip(raw[i], raw[j], target)) / len(target) for i, j in itertools.combinations(range(len(raw)), 2)]
        np.testing.assert_allclose(singles, result['single_' + key], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(pairs, result['pair_' + key], rtol=1e-12, atol=1e-14)
        assert np.isclose(np.mean(pairs), result[key], rtol=1e-12, atol=1e-14)
        counts += len(singles) + len(pairs)
    return counts


def main():
    locked = run.verify_lock()
    evaluation = run.read(run.ROOT / 'evaluation.json')
    for name, value in evaluation['prediction_hashes'].items():
        assert run.digest(run.ROOT / name) == value
    results, scalar_checks, archive_checks = {}, 0, 0
    for subset, seeds in SUBSETS.items():
        periods = {}
        for period in ['selection', 'later']:
            rows = []
            for mouse in run.MICE:
                meta = run.read(run.reference.FAIR / mouse / 'metadata.json')
                lower = -meta['speed_mean'] / meta['speed_std']
                scores = {}
                if period == 'later':
                    with np.load(run.ROOT / mouse / 'later_predictions.npz') as z:
                        target = z['target']
                        for group in GROUPS:
                            raw = np.stack([z[f'{group}_s{s}'] for s in seeds])
                            scores[group] = metrics(raw, target, lower)
                            scalar_checks += independent_check(raw, target, lower, scores[group])
                        for directory, mapping, archive_seeds in [
                            (run.REPLICATION, dict(baseline='attention', prior_mlp='mlp'), seeds),
                            (run.FREE, dict(free_dynamic='dynamic', free_static='static'), seeds),
                            (run.PRIOR, dict(dynamic='interleaved_dynamic'), [s for s in seeds if s < 13]),
                        ]:
                            with np.load(directory / mouse / 'later_predictions.npz') as old:
                                np.testing.assert_array_equal(target, old['target'])
                                for a, b in mapping.items():
                                    for seed in archive_seeds:
                                        np.testing.assert_array_equal(z[f'{a}_s{seed}'], old[f'{b}_s{seed}'])
                                        archive_checks += 1
                else:
                    target = np.load(run.reference.BASE / mouse / 'selection_y.npy')
                    for group in GROUPS:
                        predictions = []
                        for seed in seeds:
                            if group == 'baseline':
                                path = run.parent_dir(seed)
                            elif group == 'prior_mlp':
                                path = run.parent_dir(seed).parent / f'mlp_s{seed}'
                            elif group.startswith('free_'):
                                path = run.FREE / f'{group[5:]}_s{seed}'
                            else:
                                path = run.fit_dir(group, seed)
                            prediction, saved_target = run.old.selected_validation(path, mouse)
                            np.testing.assert_array_equal(target, saved_target)
                            predictions.append(prediction)
                        scores[group] = metrics(np.stack(predictions), target, lower)
                median = float(np.median(np.load(run.reference.BASE / mouse / 'train_y.npy')))
                controls = {name: dict(mse=float(np.mean((value - target) ** 2)), mae=float(np.mean(np.abs(value - target))))
                    for name, value in [('training_mean', 0.), ('training_median', median)]}
                rows.append(dict(mouse=mouse, n=len(target), lower=lower, scores=scores, controls=controls))
            summaries, gates = summarize(rows, seeds)
            periods[period] = dict(rows=rows, summaries=summaries, gates=gates,
                single_wins_vs_initial={g: sum(sum(x < r['controls']['training_mean']['mse'] for x in r['scores'][g]['single_mse']) for r in rows) for g in GROUPS})
        results[subset] = dict(seeds=seeds, pairs_per_mouse=len(list(itertools.combinations(seeds, 2))), **periods)
    primary = results['additional']['later']['gates']
    aggregate = results['all']['later']['gates']
    decisions = {name: primary[name] and aggregate[name] for name in ['dynamic_utility', 'static_utility', 'dynamic_routing']}
    run.write(run.ROOT / 'results.json', dict(subsets=results, decisions=decisions,
        primary_subset='additional', all_six_is_a_guard_not_rescue=True,
        independent_animal_significance=False, new_fits=9, baseline_fits_repeated=0))
    run.write(run.ROOT / 'audit.json', dict(passed=True, new_fits=9, reused_interleaved_fits=3,
        scalar_errors_independently_checked=scalar_checks, archived_prediction_arrays_exact=archive_checks,
        selection_scores_checked=locked['selection_scores_checked'], new_later_predictions=evaluation['new_predictions'],
        frozen_files=len(run.verify()['hashes']), locked_artifacts=len(locked['hashes']),
        prediction_hashes_checked=4, actual_steps_per_fit=5688, examples_per_fit=179712,
        main_application_unchanged=True, old_studies_unchanged=True))
    print({name: results['additional']['later']['summaries'][name] for name in ['dynamic_vs_baseline', 'dynamic_vs_static', 'static_vs_baseline']}, flush=True)
    print('Final decisions:', decisions, flush=True)


if __name__ == '__main__':
    main()
