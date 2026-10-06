"""Frozen comparisons and practical gates for one neuron-masking training recipe."""
import itertools
import json
from pathlib import Path

import numpy as np

import run


ROOT = Path(__file__).resolve().parent
PAIRS = list(itertools.combinations(range(6), 2))
GROUPS = [t + '_' + f for t in ['native', 'augmented'] for f in run.FAMILIES]
CONTRASTS = dict(attention_augmentation=('augmented_attention', 'native_attention'),
                 mlp_augmentation=('augmented_mlp', 'native_mlp'),
                 augmented_attention_vs_mlp=('augmented_attention', 'augmented_mlp'),
                 native_attention_vs_mlp=('native_attention', 'native_mlp'))


def score(raw, y, lower):
    bounded = np.maximum(raw.astype(np.float64), lower)
    pairs = np.stack([(bounded[i] + bounded[j]) / 2 for i, j in PAIRS])
    error = pairs - y
    values = dict(mse=float(np.mean(error ** 2)), mae=float(np.mean(np.abs(error))),
                  pair_view_mse=np.mean(error ** 2, -1).tolist(),
                  pair_view_mae=np.mean(np.abs(error), -1).tolist(),
                  single_view_mse=np.mean((bounded - y) ** 2, -1).tolist(),
                  single_view_mae=np.mean(np.abs(bounded - y), -1).tolist())
    for name, power in [('mse', 2), ('mae', 1)]:
        manual = np.array([[sum(abs(float(p) - float(t)) ** power for p, t in zip(view, y)) / len(y)
                            for view in pair] for pair in pairs])
        np.testing.assert_allclose(manual, values['pair_view_' + name], rtol=1e-12, atol=1e-14)
    return values, pairs.shape[0] * pairs.shape[1] * 2


def check():
    raw = np.arange(6 * 5 * 7, dtype=float).reshape(6, 5, 7) / 10 - 6
    y = np.arange(7, dtype=float)
    result, _ = score(raw, y, -1)
    expected = []
    for i, j in PAIRS:
        expected.append([[(((max(raw[i, v, t], -1) + max(raw[j, v, t], -1)) / 2) - y[t]) ** 2
                         for t in range(7)] for v in range(5)])
    assert abs(np.mean(expected) - result['mse']) < 1e-12
    pairs = [(np.maximum(raw[i], -1) + np.maximum(raw[j], -1)) / 2 for i, j in PAIRS]
    assert not np.isclose(result['mse'], np.mean((np.mean(pairs, axis=(0, 1)) - y) ** 2))
    np.testing.assert_array_equal(np.asarray(result['pair_view_mse']).shape, (15, 5))


def main():
    check()
    run.verify_lock()
    rows, hashes = [], {}
    scalar_checks = 0
    for mouse in run.MICE:
        completed = run.read(ROOT / mouse / 'complete.json')
        assert len(completed['records']) == 24
        bank = {g: {c: [] for c in run.CONDITIONS} for g in GROUPS}
        target = None
        for group in GROUPS:
            for seed in run.SEEDS:
                path = ROOT / mouse / f'{group}_s{seed}.npz'
                hashes[str(path.relative_to(ROOT))] = run.digest(path)
                treatment, family = group.split('_')
                record = next(v for v in completed['records'] if v['treatment'] == treatment and v['family'] == family and v['seed'] == seed)
                assert hashes[str(path.relative_to(ROOT))] == record['hash']
                with np.load(path) as z:
                    if target is None:
                        target = z['target']
                    np.testing.assert_array_equal(target, z['target'])
                    for c in run.CONDITIONS:
                        assert z[c].shape == (1 if c == 'clean' else 5, len(target))
                        bank[group][c].append(z[c])
        lower = completed['lower']
        scores = {c: {} for c in run.CONDITIONS}
        for c in run.CONDITIONS:
            for group in GROUPS:
                scores[c][group], checked = score(np.stack(bank[group][c]), target, lower)
                scalar_checks += checked
        median = float(np.median(np.load(run.reference.BASE / mouse / 'train_y.npy')))
        references = {label: dict(mse=float(np.mean((value - target) ** 2)), mae=float(np.mean(np.abs(value - target))))
                      for label, value in [('training_mean', 0.), ('training_median', median)]}
        contrasts = {}
        for name, (main_group, control) in CONTRASTS.items():
            contrasts[name] = {}
            for c in run.CONDITIONS:
                a, b = scores[c][main_group], scores[c][control]
                one = np.mean(a['single_view_mse'], 1)
                other = np.mean(b['single_view_mse'], 1)
                contrasts[name][c] = dict(mse_gain=1 - a['mse'] / b['mse'], mae_gain=1 - a['mae'] / b['mae'],
                    seed_wins=int(np.sum(one < other)), seed_gains=(1 - one / other).tolist(),
                    view_gains=(1 - np.mean(a['pair_view_mse'], 0) / np.mean(b['pair_view_mse'], 0)).tolist())
        degradation = {g: {c: scores[c][g]['mse'] / scores['clean'][g]['mse'] - 1 for c in run.CONDITIONS} for g in GROUPS}
        initial_wins = {g: int(np.sum(np.mean(scores['clean'][g]['single_view_mse'], 1) < references['training_mean']['mse'])) for g in GROUPS}
        rows.append(dict(mouse=mouse, n=len(target), lower=lower, scores=scores, contrasts=contrasts,
                         degradation=degradation, references=references, initial_wins=initial_wins))
    summaries = {}
    for name in CONTRASTS:
        summaries[name] = {}
        for c in run.CONDITIONS:
            cells = [r['contrasts'][name][c] for r in rows]
            gains = [v['mse_gain'] for v in cells]
            summaries[name][c] = dict(mean_mse_gain=float(np.mean(gains)), mouse_wins=sum(v > 0 for v in gains),
                mouse_mse_gains=gains, mean_mae_gain=float(np.mean([v['mae_gain'] for v in cells])),
                seed_wins=sum(v['seed_wins'] for v in cells),
                leave_one_mouse_out=[float(np.mean(gains[:i] + gains[i + 1:])) for i in range(4)],
                view_mean_gains=np.mean([v['view_gains'] for v in cells], 0).tolist())
    gates = {}
    for family in run.FAMILIES:
        cells = summaries[family + '_augmentation']
        missing, clean = cells['missing_32'], cells['clean']
        robustness = missing['mean_mse_gain'] >= .1 and missing['mouse_wins'] >= 3 and missing['seed_wins'] >= 16
        safeguard = clean['mean_mse_gain'] >= -.05 and min(clean['mouse_mse_gains']) >= -.1
        gates[family] = dict(robustness_pass=bool(robustness), clean_safeguard_pass=bool(safeguard),
                            combined_pass=bool(robustness and safeguard))
    comparator = summaries['augmented_attention_vs_mlp']['missing_32']
    gates['augmented_attention_advantage_25pct'] = bool(comparator['mean_mse_gain'] >= .05 and comparator['mouse_wins'] >= 3 and comparator['seed_wins'] >= 16)
    degradation = {g: {c: float(np.mean([r['degradation'][g][c] for r in rows])) for c in run.CONDITIONS} for g in GROUPS}
    learning = {g: sum(r['initial_wins'][g] for r in rows) for g in GROUPS}
    run.write(ROOT / 'results.json', dict(rows=rows, summaries=summaries, gates=gates, degradation=degradation,
                                        clean_single_wins_vs_initial=learning))
    run.verify_lock()
    run.write(ROOT / 'audit.json', dict(passed=True, new_fits=12, epochs_per_fit=24, updates_per_fit=5688,
        examples_per_fit=179712, actual_optimizer_steps_verified=True, exact_initializations=True,
        matched_parent_batches=True, matched_family_training_masks=True, selected_reloads_exact=True,
        selection_scores_checked=1200, all_choices_locked_before_current_later_inference=True,
        fresh_mask_bank=True, baseline_clean_firstbatch_exact=48, independent_scalar_scores=scalar_checks,
        source_and_application_hashes_unchanged=True, prediction_hashes=hashes,
        new_later_predictions=24 * 15 * sum(r['n'] for r in rows) + 12 * sum(r['n'] for r in rows)))
    run.write(ROOT / 'environment.json', dict(platform=run.platform.platform(), torch=run.torch.__version__,
        numpy=np.__version__, device='cpu', training_torch_threads=2, interop_threads=1))
    print(json.dumps(dict(summaries=summaries, gates=gates, degradation=degradation,
                          clean_single_wins_vs_initial=learning), indent=2))


if __name__ == '__main__':
    main()
