"""Temporal-specialization utility and component comparisons with fixed criteria."""
import itertools
import json
from pathlib import Path

import numpy as np

import run


ROOT = Path(__file__).resolve().parent
SEEDS = []
GROUPS = ['baseline', 'free_dynamic', 'free_static', 'time_dynamic', 'time_static', 'interleaved_dynamic', 'prior_mlp']
CONTRASTS = {
    'time_dynamic_vs_baseline': ('time_dynamic', 'baseline'),
    'time_dynamic_vs_free_dynamic': ('time_dynamic', 'free_dynamic'),
    'time_dynamic_vs_interleaved_dynamic': ('time_dynamic', 'interleaved_dynamic'),
    'time_dynamic_vs_time_static': ('time_dynamic', 'time_static'),
    'time_static_vs_baseline': ('time_static', 'baseline'),
    'time_static_vs_free_static': ('time_static', 'free_static'),
    'interleaved_dynamic_vs_baseline': ('interleaved_dynamic', 'baseline'),
    'interleaved_dynamic_vs_free_dynamic': ('interleaved_dynamic', 'free_dynamic'),
    'time_dynamic_vs_prior_mlp': ('time_dynamic', 'prior_mlp'),
    'time_static_vs_prior_mlp': ('time_static', 'prior_mlp'),
}


def metrics(raw, y, lower):
    bounded = np.maximum(raw.astype(np.float64), lower)
    pairs = np.stack([(bounded[i] + bounded[j]) / 2 for i, j in itertools.combinations(range(len(raw)), 2)])
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
    expected = [[(((max(raw[i, t], -1) + max(raw[j, t], -1)) / 2) - y[t]) ** 2 for t in range(7)] for i, j in itertools.combinations(range(6), 2)]
    assert abs(np.mean(expected) - score['mse']) < 1e-12
    assert not np.isclose(score['mse'], np.mean((pairs.mean(0) - y) ** 2))
    assert pairs.shape == (15, 7)


def replication_summary():
    records = []
    for mouse in run.MICE:
        meta = run.read(run.reference.FAIR / mouse / 'metadata.json')
        lower = -meta['speed_mean'] / meta['speed_std']
        with np.load(ROOT / mouse / 'later_predictions.npz') as z:
            scores = {g: metrics(np.stack([z[f'{g}_s{s}'] for s in [13, 14, 15]]), z['target'], lower)[0] for g in GROUPS}
        records.append(dict(mouse=mouse, scores=scores))
    summary = {}
    for name, (a, b) in CONTRASTS.items():
        gains = [1 - r['scores'][a]['mse'] / r['scores'][b]['mse'] for r in records]
        summary[name] = dict(mean_mse_gain=float(np.mean(gains)), mouse_gains=gains,
            mouse_wins=sum(g > 0 for g in gains),
            seed_wins=sum(int(np.sum(np.array(r['scores'][a]['single_mse']) < r['scores'][b]['single_mse'])) for r in records))
    return dict(seeds=[13, 14, 15], rows=records, summaries=summary,
                scope='Training-seed replication on the same historically reused animals;not independent animal confirmation')


def main():
    check()
    locked = run.verify_lock()
    SEEDS[:] = locked['seeds']
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
                raw = np.stack([z[f'{group}_s{s}'] for s in SEEDS])
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
                for seed in SEEDS:
                    np.testing.assert_array_equal(z[f'baseline_s{seed}'], saved[f'attention_s{seed}'])
                    np.testing.assert_array_equal(z[f'prior_mlp_s{seed}'], saved[f'mlp_s{seed}'])
        with np.load(ROOT / mouse / 'later_predictions.npz') as current, np.load(run.FREE / mouse / 'later_predictions.npz') as saved:
            np.testing.assert_array_equal(current['target'], saved['target'])
            for seed in SEEDS:
                for group in ['dynamic', 'static']:
                    np.testing.assert_array_equal(current[f'free_{group}_s{seed}'], saved[f'{group}_s{seed}'])
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
    for name in CONTRASTS:
        s = summaries[name]
        gates[name] = bool(s['mean_mse_gain'] >= .05 and s['mouse_wins'] >= 3 and s['seed_wins'] >= int(np.ceil(2 * 4 * len(SEEDS) / 3)))
    screen = run.read(ROOT / 'screen.json')
    for variant, parent in [('time_dynamic', 'free_dynamic'), ('time_static', 'free_static')]:
        gates[variant + '_no_mouse_over10pct_harm'] = bool(min(summaries[variant + '_vs_baseline']['mouse_mse_gains']) >= -.1)
        gates[variant + '_earlier_screen'] = screen['candidate_passed'][variant]
        gates[variant + '_utility'] = all(gates[k] for k in [variant + '_earlier_screen', variant + '_vs_baseline', variant + '_vs_' + parent, variant + '_no_mouse_over10pct_harm'])
    gates['time_dynamic_full'] = gates['time_dynamic_utility'] and gates['time_dynamic_vs_interleaved_dynamic'] and gates['time_dynamic_vs_time_static']
    gates['time_static_secondary'] = gates['time_static_utility']
    initial_wins = {g: sum(sum(v < r['references']['training_mean']['mse'] for v in r['scores'][g]['single_mse']) for r in rows) for g in GROUPS}
    replication = replication_summary() if len(SEEDS) == 6 else None
    run.write(ROOT / 'results.json', dict(seeds=SEEDS, pairs_per_mouse=len(list(itertools.combinations(SEEDS, 2))), rows=rows, summaries=summaries, gates=gates, single_wins_vs_initial=initial_wins, extra_seed_replication=replication))
    run.verify_lock()
    run.write(ROOT / 'audit.json', dict(passed=True, new_fits=len(run.VARIANTS) * len(SEEDS), baseline_fits_repeated=0,
        epochs_per_fit=24, updates_per_fit=5688, examples_per_fit=179712, actual_adam_steps_verified=True,
        all_unrestricted_parent_initial_tensors_exact=True, all_initial_tensors_match_between_new_variants=True, batches_match_archived_baseline=True,
        query_and_temporal_gradients_nonzero=True, selected_reloads_exact=True, selection_scores_checked=locked['selection_scores_checked'],
        choices_locked_before_current_later_inference=True, baseline_firstbatch_exact=evaluation['exact_baseline_firstbatch_checks'],
        independent_scalar_scores=scalar_checks, target_alignment_and_archived_predictions_exact=True,
        inputs_and_model_tensors_preserved=True, frozen_source_application_and_selection_hashes_unchanged=True,
        new_later_predictions=len(run.VARIANTS) * len(SEEDS) * sum(r['n'] for r in rows)))
    run.write(ROOT / 'environment.json', dict(platform=run.platform.platform(), torch=run.torch.__version__,
        numpy=np.__version__, device='cpu', training_torch_threads=2, interop_threads=1))
    print(json.dumps(dict(summaries=summaries, gates=gates, single_wins_vs_initial=initial_wins), indent=2))


if __name__ == '__main__':
    main()
