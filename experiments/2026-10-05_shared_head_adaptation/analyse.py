"""Evaluate selected residual heads with the same fixed practical criteria."""
import importlib.util
import math

import numpy as np

import run

spec = importlib.util.spec_from_file_location('head_metrics', run.FINETUNING / 'analyse.py')
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)


def main():
    lock = run.verify_lock()
    evaluation = run.read(run.ROOT / 'evaluation.json')
    for name, value in evaluation['prediction_hashes'].items():
        assert run.digest(run.ROOT / name) == value
    for family in run.FAMILIES:
        scores = np.mean([run.read(run.ROOT / f'{family}_s{s}' / 'result.json')['joint_scores'] for s in [10, 11, 12]], axis=0)
        index = int(np.argmin(scores))
        assert lock['choices'][family]['index'] == index
        assert lock['choices'][family]['config'] == run.CONFIGS[index]
    result, count, checks = {}, 0, 0
    for subset, seeds in shared.SUBSETS.items():
        periods = {}
        for period in ['selection', 'later']:
            rows = []
            for mouse in run.MICE:
                meta = run.read(run.ref.FAIR / mouse / 'metadata.json')
                lower = -meta['speed_mean'] / meta['speed_std']
                if period == 'later':
                    with np.load(run.ROOT / mouse / 'later_predictions.npz') as z:
                        y = z['target']
                        predictions = {g: np.stack([z[f'{g}_s{s}'] for s in seeds]) for g in shared.GROUPS}
                else:
                    y = np.load(run.ref.BASE / mouse / 'selection_y.npy')
                    predictions = {}
                    for family in run.FAMILIES:
                        native, adapted = [], []
                        for seed in seeds:
                            with np.load(run.ROOT / f'{family}_s{seed}' / 'selection_predictions.npz') as z:
                                np.testing.assert_array_equal(y, z[mouse + '_target'])
                                native.append(z[mouse + '_native'])
                                index = lock['choices'][family]['index'] if seed < 13 else 0
                                adapted.append(z[mouse + '_predictions'][index])
                        predictions['native_' + family] = np.stack(native)
                        predictions['tuned_' + family] = np.stack(adapted)
                scores = {}
                for group, pred in predictions.items():
                    scores[group], checked = shared.metrics(pred, y, lower)
                    count += checked
                median = float(np.median(np.load(run.ref.BASE / mouse / 'train_y.npy')))
                controls = {k: dict(mse=float(np.mean((v - y) ** 2)), mae=float(np.mean(np.abs(v - y))))
                    for k, v in [('training_mean', 0.), ('training_median', median)]}
                rows.append(dict(mouse=mouse, n=len(y), lower=lower, scores=scores, controls=controls))
            summaries, gates = shared.summarize(rows, seeds)
            for name, summary in summaries.items():
                a, b = name.split('_vs_')
                gains = [1 - row['scores'][a]['mse'] / row['scores'][b]['mse'] for row in rows]
                wins = sum(sum(x < y for x, y in zip(row['scores'][a]['single_mse'], row['scores'][b]['single_mse'])) for row in rows)
                np.testing.assert_allclose(summary['mean_mse_gain'], sum(gains) / 4, rtol=1e-12, atol=1e-14)
                assert summary['seed_wins'] == wins
                assert summary['contrast_passed'] == (sum(gains) / 4 >= .05 and sum(g > 0 for g in gains) >= 3 and wins >= math.ceil(8 * len(seeds) / 3))
                checks += 3
            for family in run.FAMILIES:
                contrast = summaries[f'tuned_{family}_vs_native_{family}']
                expected = contrast['contrast_passed'] and min(contrast['mouse_mse_gains']) >= -.1 and contrast['mean_mae_gain'] >= 0
                assert gates[family + '_improvement'] == expected
                checks += 1
            assert gates['transformer_full'] == (gates['attention_improvement'] and summaries['tuned_attention_vs_tuned_mlp']['contrast_passed'])
            checks += 1
            periods[period] = dict(rows=rows, summaries=summaries, gates=gates)
        result[subset] = dict(seeds=seeds, pairs_per_mouse=math.comb(len(seeds), 2), **periods)
    decisions = {k: result['additional']['later']['gates'][k] and result['all']['later']['gates'][k]
        for k in ['attention_improvement', 'mlp_improvement', 'transformer_full']}
    run.write(run.ROOT / 'results.json', dict(subsets=result, choices=lock['choices'], decisions=decisions,
        neural_fits=0, independent_animal_significance=False))
    run.write(run.ROOT / 'audit.json', dict(passed=True, scalar_errors_checked=count,
        aggregate_gate_checks=checks, native_later_feature_prediction_checks=48,
        selected_artifact_hashes=len(lock['hashes']), frozen_source_input_hashes=len(run.verify()['hashes']),
        selection_scores_checked=lock['selection_scores_checked'], recipe_selection_independently_checked=True,
        main_application_unchanged=True))
    for subset in ['additional', 'all']:
        print(subset, result[subset]['later']['summaries'], flush=True)
    print('Decisions:', decisions, flush=True)


if __name__ == '__main__':
    main()
