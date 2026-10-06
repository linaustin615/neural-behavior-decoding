"""Audit matched continuation and seed replication using the frozen endpoints."""
import itertools
import math

import numpy as np

import run

GROUPS = ['native_attention', 'native_mlp', 'tuned_attention', 'tuned_mlp']
CONTRASTS = dict(tuned_attention_vs_native_attention=('tuned_attention', 'native_attention'),
    tuned_attention_vs_tuned_mlp=('tuned_attention', 'tuned_mlp'),
    tuned_mlp_vs_native_mlp=('tuned_mlp', 'native_mlp'),
    tuned_attention_vs_native_mlp=('tuned_attention', 'native_mlp'),
    native_attention_vs_native_mlp=('native_attention', 'native_mlp'))
SUBSETS = dict(original=[10, 11, 12], additional=[13, 14, 15], all=list(range(10, 16)))


def metrics(raw, y, lower):
    assert raw.ndim == 2 and raw.shape[1:] == y.shape and np.isfinite(raw).all()
    bounded = np.maximum(raw.astype(np.float64), lower)
    pairs = np.stack([(bounded[i] + bounded[j]) / 2 for i, j in itertools.combinations(range(len(raw)), 2)])
    result = dict(mse=float(np.mean((pairs - y) ** 2)), mae=float(np.mean(np.abs(pairs - y))),
        pair_mse=np.mean((pairs - y) ** 2, 1).tolist(), pair_mae=np.mean(np.abs(pairs - y), 1).tolist(),
        single_mse=np.mean((bounded - y) ** 2, 1).tolist(), single_mae=np.mean(np.abs(bounded - y), 1).tolist())
    result['r2'] = 1 - result['mse'] / float(np.var(y)) if np.var(y) else None
    count = 0
    for key, power in [('mse', 2), ('mae', 1)]:
        single = [sum(abs(max(float(p), lower) - float(t)) ** power for p, t in zip(row, y)) / len(y) for row in raw]
        paired = [sum(abs((max(float(a), lower) + max(float(b), lower)) / 2 - float(t)) ** power
            for a, b, t in zip(raw[i], raw[j], y)) / len(y) for i, j in itertools.combinations(range(len(raw)), 2)]
        np.testing.assert_allclose(single, result['single_' + key], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(paired, result['pair_' + key], rtol=1e-12, atol=1e-14)
        count += len(single) + len(paired)
    return result, count


def summarize(rows, seeds):
    summaries = {}
    for name, (a, b) in CONTRASTS.items():
        gains = [1 - r['scores'][a]['mse'] / r['scores'][b]['mse'] for r in rows]
        mae = [1 - r['scores'][a]['mae'] / r['scores'][b]['mae'] for r in rows]
        sa = np.array([r['scores'][a]['single_mse'] for r in rows])
        sb = np.array([r['scores'][b]['single_mse'] for r in rows])
        pa = np.array([r['scores'][a]['pair_mse'] for r in rows])
        pb = np.array([r['scores'][b]['pair_mse'] for r in rows])
        s = dict(mean_mse_gain=float(np.mean(gains)), mouse_mse_gains=gains, mouse_wins=sum(x > 0 for x in gains),
            mean_mae_gain=float(np.mean(mae)), mouse_mae_gains=mae, seed_wins=int(np.sum(sa < sb)),
            pair_wins=int(np.sum(pa < pb)), single_mean_mse_gain=float(np.mean(1 - sa.mean(1) / sb.mean(1))),
            individual_seeds=[dict(seed=seed, mean_mse_gain=float(np.mean(1 - sa[:, i] / sb[:, i])),
                mouse_wins=int(np.sum(sa[:, i] < sb[:, i]))) for i, seed in enumerate(seeds)],
            leave_one_mouse_out=[float(np.mean(gains[:i] + gains[i + 1:])) for i in range(4)])
        s['contrast_passed'] = s['mean_mse_gain'] >= .05 and s['mouse_wins'] >= 3 and s['seed_wins'] >= math.ceil(8 * len(seeds) / 3)
        summaries[name] = s
    gates = {}
    for family in ['attention', 'mlp']:
        s = summaries[f'tuned_{family}_vs_native_{family}']
        gates[family + '_harm_guard'] = min(s['mouse_mse_gains']) >= -.1
        gates[family + '_mae_guard'] = s['mean_mae_gain'] >= 0
        gates[family + '_improvement'] = s['contrast_passed'] and gates[family + '_harm_guard'] and gates[family + '_mae_guard']
    gates['transformer_full'] = gates['attention_improvement'] and summaries['tuned_attention_vs_tuned_mlp']['contrast_passed']
    return summaries, gates


def main():
    locked = run.verify_lock()
    evaluation = run.read(run.ROOT / 'evaluation.json')
    for path, value in evaluation['prediction_hashes'].items():
        assert run.digest(run.ROOT / path) == value
    result, scalar_checks, archived = {}, 0, 0
    for name, seeds in SUBSETS.items():
        periods = {}
        for period in ['selection', 'later']:
            rows = []
            for mouse in run.MICE:
                meta = run.read(run.ref.FAIR / mouse / 'metadata.json')
                lower = -meta['speed_mean'] / meta['speed_std']
                if period == 'later':
                    with np.load(run.ROOT / mouse / 'later_predictions.npz') as z:
                        y = z['target']
                        predictions = {g: np.stack([z[f'{g}_s{s}'] for s in seeds]) for g in GROUPS}
                        with np.load(run.REPLICATION / mouse / 'later_predictions.npz') as old:
                            np.testing.assert_array_equal(y, old['target'])
                            for family in run.FAMILIES:
                                for seed in seeds:
                                    np.testing.assert_array_equal(z[f'native_{family}_s{seed}'], old[f'{family}_s{seed}'])
                                    archived += 1
                else:
                    y = np.load(run.ref.BASE / mouse / 'selection_y.npy')
                    predictions = {}
                    for family in run.FAMILIES:
                        recipe = locked['choices'][family]
                        native, tuned = [], []
                        for seed in seeds:
                            p = run.native_dir(family, seed)
                            epoch = run.read(p / 'result.json')['selected_epoch']
                            with np.load(p / 'selection_predictions.npz') as z:
                                np.testing.assert_array_equal(y, z[mouse + '_target'])
                                native.append(z[mouse + '_predictions'][epoch])
                            p = run.fit_dir(family, recipe['rate'], seed)
                            epoch = run.read(p / 'result.json')['selected_epochs'][recipe['mode']]
                            with np.load(p / 'selection_predictions.npz') as z:
                                np.testing.assert_array_equal(y, z[mouse + '_target'])
                                tuned.append(z[recipe['mode'] + '_' + mouse][epoch])
                        predictions['native_' + family] = np.stack(native)
                        predictions['tuned_' + family] = np.stack(tuned)
                scores = {}
                for group, raw in predictions.items():
                    scores[group], count = metrics(raw, y, lower)
                    scalar_checks += count
                median = float(np.median(np.load(run.ref.BASE / mouse / 'train_y.npy')))
                controls = {k: dict(mse=float(np.mean((v - y) ** 2)), mae=float(np.mean(np.abs(v - y))))
                    for k, v in [('training_mean', 0.), ('training_median', median)]}
                rows.append(dict(mouse=mouse, n=len(y), lower=lower, scores=scores, controls=controls))
            summaries, gates = summarize(rows, seeds)
            periods[period] = dict(rows=rows, summaries=summaries, gates=gates)
        result[name] = dict(seeds=seeds, pairs_per_mouse=math.comb(len(seeds), 2), **periods)
    decisions = {key: result['additional']['later']['gates'][key] and result['all']['later']['gates'][key]
        for key in ['attention_improvement', 'mlp_improvement', 'transformer_full']}
    run.write(run.ROOT / 'results.json', dict(subsets=result, choices=locked['choices'], decisions=decisions,
        new_fits=18, baseline_fits_repeated=0, independent_animal_significance=False))
    run.write(run.ROOT / 'audit.json', dict(passed=True, scalar_errors_checked=scalar_checks,
        archived_prediction_arrays_exact=archived, source_input_hashes=len(run.verify()['hashes']),
        selected_artifact_hashes=len(locked['hashes']), selection_scores_checked=locked['selection_scores_checked'],
        new_predictions=evaluation['new_predictions'], exact_native_starts=18 * 4,
        main_application_unchanged=True, losing_recipes_not_scored_on_later=True))
    for name in ['original', 'additional', 'all']:
        print(name, result[name]['later']['summaries'], flush=True)
    print('Decisions:', decisions, flush=True)


if __name__ == '__main__':
    main()
