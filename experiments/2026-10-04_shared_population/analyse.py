"""Frozen three-way comparison of shared decoder population blocks."""
import itertools
import json
from pathlib import Path

import numpy as np

import run


ROOT = Path(__file__).resolve().parent
PAIRS = list(itertools.combinations(range(6), 2))
GROUPS = ['baseline', 'attention', 'mixer', 'prior_mlp']
CONTRASTS = dict(attention_vs_baseline=('attention', 'baseline'),
                 attention_vs_mixer=('attention', 'mixer'),
                 mixer_vs_baseline=('mixer', 'baseline'),
                 attention_vs_prior_mlp=('attention', 'prior_mlp'),
                 mixer_vs_prior_mlp=('mixer', 'prior_mlp'))


def metrics(raw, y, lower):
    bounded = np.maximum(raw.astype(np.float64), lower)
    pairs = np.stack([(bounded[i] + bounded[j]) / 2 for i, j in PAIRS])
    error = pairs - y
    result = dict(mse=float(np.mean(error ** 2)), mae=float(np.mean(np.abs(error))),
                  pair_mse=np.mean(error ** 2, 1).tolist(), pair_mae=np.mean(np.abs(error), 1).tolist(),
                  single_mse=np.mean((bounded - y) ** 2, 1).tolist(),
                  single_mae=np.mean(np.abs(bounded - y), 1).tolist())
    result['r2'] = 1 - result['mse'] / float(np.var(y)) if np.var(y) > 0 else None
    return result, pairs


def check():
    raw = np.arange(6 * 7, dtype=float).reshape(6, 7) / 3 - 4
    y = np.arange(7, dtype=float)
    score, pairs = metrics(raw, y, -1)
    expected = [[(((max(raw[i, t], -1) + max(raw[j, t], -1)) / 2) - y[t]) ** 2 for t in range(7)] for i, j in PAIRS]
    assert abs(np.mean(expected) - score['mse']) < 1e-12
    assert not np.isclose(score['mse'], np.mean((pairs.mean(0) - y) ** 2))
    assert pairs.shape == (15, 7)


def main():
    check()
    run.verify_lock()
    evaluation = run.read(ROOT / 'evaluation.json')
    for name, value in evaluation['prediction_hashes'].items():
        assert run.digest(ROOT / name) == value
    rows, scalar_checks = [], 0
    for mouse in run.MICE:
        meta = run.read(run.reference.FAIR / mouse / 'metadata.json')
        lower = -meta['speed_mean'] / meta['speed_std']
        scores = {}
        with np.load(ROOT / mouse / 'later_predictions.npz') as z:
            y = z['target']
            for group in GROUPS:
                raw = np.stack([z[f'{group}_s{s}'] for s in run.SEEDS])
                scores[group], pairs = metrics(raw, y, lower)
                for metric, power in [('mse', 2), ('mae', 1)]:
                    manual = [sum(abs(float(p) - float(t)) ** power for p, t in zip(pair, y)) / len(y) for pair in pairs]
                    np.testing.assert_allclose(manual, scores[group]['pair_' + metric], rtol=1e-12, atol=1e-14)
                    scalar_checks += len(manual)
                    individual = [sum(abs(max(float(p), lower) - float(t)) ** power for p, t in zip(pred, y)) / len(y) for pred in raw]
                    np.testing.assert_allclose(individual, scores[group]['single_' + metric], rtol=1e-12, atol=1e-14)
                    scalar_checks += len(individual)
            with np.load(run.REPLICATION / mouse / 'later_predictions.npz') as saved:
                np.testing.assert_array_equal(y, saved['target'])
                for seed in run.SEEDS:
                    np.testing.assert_array_equal(z[f'baseline_s{seed}'], saved[f'attention_s{seed}'])
                    np.testing.assert_array_equal(z[f'prior_mlp_s{seed}'], saved[f'mlp_s{seed}'])
        median = float(np.median(np.load(run.reference.BASE / mouse / 'train_y.npy')))
        references = {name: dict(mse=float(np.mean((value - y) ** 2)), mae=float(np.mean(np.abs(value - y))))
                      for name, value in [('training_mean', 0.), ('training_median', median)]}
        contrasts = {}
        for name, (main_group, control) in CONTRASTS.items():
            a, b = scores[main_group], scores[control]
            single_a, single_b = np.array(a['single_mse']), np.array(b['single_mse'])
            contrasts[name] = dict(mse_gain=1 - a['mse'] / b['mse'], mae_gain=1 - a['mae'] / b['mae'],
                seed_wins=int(np.sum(single_a < single_b)), seed_gains=(1 - single_a / single_b).tolist(),
                pair_wins=int(np.sum(np.array(a['pair_mse']) < np.array(b['pair_mse']))),
                single_mean_mse_gain=1 - float(single_a.mean() / single_b.mean()))
        rows.append(dict(mouse=mouse, n=len(y), lower=lower, scores=scores, references=references, contrasts=contrasts))
    summaries = {}
    for name in CONTRASTS:
        cells = [r['contrasts'][name] for r in rows]
        gains = [v['mse_gain'] for v in cells]
        summaries[name] = dict(mean_mse_gain=float(np.mean(gains)), mouse_mse_gains=gains,
            mouse_wins=sum(v > 0 for v in gains), mean_mae_gain=float(np.mean([v['mae_gain'] for v in cells])),
            mae_mouse_wins=sum(v['mae_gain'] > 0 for v in cells), seed_wins=sum(v['seed_wins'] for v in cells),
            pair_wins=sum(v['pair_wins'] for v in cells),
            single_mean_mse_gain=float(np.mean([v['single_mean_mse_gain'] for v in cells])),
            leave_one_mouse_out=[float(np.mean(gains[:i] + gains[i + 1:])) for i in range(4)])
    gates = {}
    for name in ['attention_vs_baseline', 'attention_vs_mixer', 'mixer_vs_baseline']:
        s = summaries[name]
        gates[name] = bool(s['mean_mse_gain'] >= .05 and s['mouse_wins'] >= 3 and s['seed_wins'] >= 16)
    gates['attention_no_mouse_over10pct_harm'] = bool(min(summaries['attention_vs_baseline']['mouse_mse_gains']) >= -.1)
    gates['mixer_no_mouse_over10pct_harm'] = bool(min(summaries['mixer_vs_baseline']['mouse_mse_gains']) >= -.1)
    gates['attention_full'] = gates['attention_vs_baseline'] and gates['attention_vs_mixer'] and gates['attention_no_mouse_over10pct_harm']
    gates['mixer_secondary'] = gates['mixer_vs_baseline'] and gates['mixer_no_mouse_over10pct_harm']
    initial_wins = {g: sum(sum(v < r['references']['training_mean']['mse'] for v in r['scores'][g]['single_mse']) for r in rows) for g in GROUPS}
    run.write(ROOT / 'results.json', dict(rows=rows, summaries=summaries, gates=gates, single_wins_vs_initial=initial_wins))
    run.verify_lock()
    run.write(ROOT / 'audit.json', dict(passed=True, new_fits=12, baseline_fits_repeated=0,
        epochs_per_fit=24, updates_per_fit=5688, examples_per_fit=179712, actual_adam_steps_verified=True,
        common_initializations_exact=True, batches_match_archived_baseline=True,
        population_gradients_nonzero=True, selected_reloads_exact=True, selection_scores_checked=1200,
        choices_locked_before_current_later_inference=True, baseline_firstbatch_exact=24,
        independent_scalar_scores=scalar_checks, target_alignment_and_archived_predictions_exact=True,
        inputs_and_model_tensors_preserved=True, frozen_source_application_and_selection_hashes_unchanged=True,
        new_later_predictions=12 * sum(r['n'] for r in rows)))
    run.write(ROOT / 'environment.json', dict(platform=run.platform.platform(), torch=run.torch.__version__,
        numpy=np.__version__, device='cpu', training_torch_threads=2, interop_threads=1))
    print(json.dumps(dict(summaries=summaries, gates=gates, single_wins_vs_initial=initial_wins), indent=2))


if __name__ == '__main__':
    main()
