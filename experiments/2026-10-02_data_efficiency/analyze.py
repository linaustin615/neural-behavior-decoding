import csv
import json
from pathlib import Path

import numpy as np
import torch

import pilot as p


def main():
    protocol = p.verify_frozen()
    p.initialize()
    errors = np.empty((3, 3, 4, 1273))
    raw_errors = np.empty_like(errors)
    records = []
    audit = []
    max_reload = 0.
    for fi, fraction in enumerate(p.FRACTIONS):
        x, y, xv, yv, meta = p.data(fraction)
        for si, seed in enumerate(p.SEEDS):
            hashes = []
            for ci, condition in enumerate(p.CONDITIONS):
                task = dict(fraction=fraction, seed=seed, condition=condition)
                name = p.stem(task)
                path = p.ROOT/'runs'/f'{name}.json'
                r = json.loads(path.read_text())
                assert r['task'] == task
                assert r['normalization'] == meta
                assert r['parameters'] == 81377
                hashes.append(r['initial_state_sha256'])
                with np.load(path.with_suffix('.npz')) as a:
                    assert sorted(a.files) == ['validation_prediction', 'validation_target']
                    np.testing.assert_array_equal(a['validation_target'], yv.numpy())
                    pred = a['validation_prediction'].copy()
                physical = pred.astype(np.float64)*meta['speed_std']+meta['speed_mean']
                target = p.SPEED[4291:5564].astype(np.float64)
                errors[fi, si, ci] = (np.maximum(physical, 0)-target)**2
                raw_errors[fi, si, ci] = (physical-target)**2
                assert abs(errors[fi, si, ci].mean()-r['validation']['mse']) < 1e-5
                assert r['best_epoch'] == min(r['history'], key=lambda h:h['mse'])['epoch']
                ck = torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=False)
                assert all(torch.isfinite(v).all() for v in ck['state_dict'].values())
                np.testing.assert_array_equal(ck['neuron_ids'], p.IDS)
                np.testing.assert_array_equal(ck['positions'], p.POSITIONS[condition])
                if not r['reused_from']:
                    net = p.model(seed)
                    net.load_state_dict(ck['state_dict'])
                    full = seed == 12 and condition == 'real'
                    indices = np.arange(1273) if full else np.linspace(0, 1272, 64, dtype=int)
                    reloaded = p.s.predict(net, xv[indices], p.POSITIONS[condition])
                    discrepancy = float(np.max(np.abs(reloaded-pred[indices])))
                    assert discrepancy < 2e-5, (name, discrepancy)
                    max_reload = max(max_reload, discrepancy)
                    audit.append(dict(run=name, examples=len(indices), max_difference=discrepancy))
                constant = float(np.mean((meta['speed_mean']-target)**2))
                r['constant_baseline_mse'] = constant
                records.append(r)
            assert len(set(hashes)) == 1
        del x, y, xv, yv
    assert np.isfinite(errors).all()
    means = errors.mean(axis=(1, 3))
    raw_means = raw_errors.mean(axis=(1, 3))
    none_relative = 1-means[:, 0]/means[:, 1]
    shuffle_relative = 1-means[:, 0]/means[:, 2:].mean(axis=1)
    per_seed = errors.mean(axis=-1)
    seed_none = (1-per_seed[:, :, 0]/per_seed[:, :, 1]).mean(axis=0)
    seed_shuffle = (1-per_seed[:, :, 0]/per_seed[:, :, 2:].mean(axis=2)).mean(axis=0)
    promising = (none_relative.mean() >= .02 and shuffle_relative.mean() > 0
                 and (none_relative > 0).sum() >= 2 and (shuffle_relative > 0).sum() >= 2
                 and (seed_none > 0).sum() >= 2 and (seed_shuffle > 0).sum() >= 2)
    low_data = (none_relative[:2].mean() > none_relative[2]
                and shuffle_relative[:2].mean() > shuffle_relative[2])
    negative = bool((none_relative <= 0).all())
    decision = 'promising' if promising else ('negative' if negative else 'inconclusive')
    rng = np.random.default_rng(20261002)
    boot = []
    for _ in range(2000):
        seeds = rng.integers(0, 3, 3)
        starts = rng.integers(0, 1273, 13)
        times = ((starts[:, None]+np.arange(100)) % 1273).ravel()[:1273]
        m = np.take(np.take(errors, seeds, axis=1), times, axis=3).mean(axis=(1, 3))
        boot.append([(1-m[:, 0]/m[:, 1]).mean(), (1-m[:, 0]/m[:, 2:].mean(1)).mean()])
    intervals = np.quantile(boot, [.025, .975], axis=0).T.tolist()
    rows = []
    for i, fraction in enumerate(p.FRACTIONS):
        rows.append(dict(fraction=fraction, examples=4160*fraction//100-31,
                         **{c:float(means[i,j]) for j,c in enumerate(p.CONDITIONS)},
                         advantage_vs_none=float(none_relative[i]),
                         advantage_vs_shuffle=float(shuffle_relative[i])))
    result = dict(decision=decision, promising_gate=bool(promising),
                  low_data_interaction=bool(low_data), negative_gate=negative,
                  mean_relative_advantage_vs_none=float(none_relative.mean()),
                  mean_relative_advantage_vs_shuffle=float(shuffle_relative.mean()),
                  conditional_intervals=dict(vs_none=intervals[0], vs_shuffle=intervals[1]),
                  per_seed_curve_advantage=dict(vs_none=seed_none.tolist(), vs_shuffle=seed_shuffle.tolist()),
                  rows=rows, raw_mse_means=raw_means.tolist(),
                  fits=36, new_fits=sum(not r['reused_from'] for r in records),
                  reused_fits=sum(bool(r['reused_from']) for r in records),
                  beat_untrained=sum(r['validation']['mse'] < r['untrained_validation']['mse'] for r in records),
                  beat_constant=sum(r['validation']['mse'] < r['constant_baseline_mse'] for r in records),
                  selected_budget_limit=sum(r['best_epoch'] == 24 for r in records),
                  new_fit_seconds=sum(r['seconds'] for r in records if not r['reused_from']),
                  maximum_reload_difference=max_reload, test_evaluations=0)
    p.s.write_json(p.ROOT/'results.json', result)
    p.s.write_json(p.ROOT/'completion_checks.json', dict(passed=True,
        new_checkpoint_reloads=audit, reused_checkpoint_reloads='prior completed audits retained; compatibility checked before reuse',
        all_saved_validation_metrics_recomputed=True, matching_initial_states=True,
        finite_parameters=True, unchanged_frozen_sources=True, unchanged_application=True,
        only_validation_arrays_saved=True, test_evaluations=0))
    with (p.ROOT/'per_run.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['fraction','seed','condition','validation_mse','raw_mse','r2','best_epoch','epochs_run','reused','beats_constant','beats_untrained'])
        for r in records:
            writer.writerow([r['task'][k] for k in ['fraction','seed','condition']]+[
                r['validation'][k] for k in ['mse','raw_mse','r2']]+[
                r['best_epoch'],r['epochs_run'],bool(r['reused_from']),
                r['validation']['mse']<r['constant_baseline_mse'],
                r['validation']['mse']<r['untrained_validation']['mse']])
    report = [
        '# Coordinate data-efficiency pilot', '',
        f'Frozen pilot decision: **{decision.upper()}**. This is an exploratory validation comparison, not independent confirmation.', '',
        'All 36 planned records are complete: 30 new fits and six compatible full-data fits reused without rerunning their completed diagnostics. No test examples were evaluated.', '',
        '| Training prefix | Examples | Correct coordinates | No coordinates | Shuffle 1 | Shuffle 2 | Correct vs none | Correct vs average shuffle |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        report.append(f"| {r['fraction']}% | {r['examples']} | {r['real']:.5f} | {r['none']:.5f} | {r['shuffle1']:.5f} | {r['shuffle2']:.5f} | {r['advantage_vs_none']:+.2%} | {r['advantage_vs_shuffle']:+.2%} |")
    report += ['', 'Errors are mean validation MSE across three training seeds, in the original dataset speed units squared. Lower is better. Positive relative advantage favors correct coordinates. Physical units have not been independently calibrated. Predictions are floored at zero speed for the primary metric; raw errors are also archived.', '',
        f"The equally weighted learning-curve advantage is {none_relative.mean():+.2%} versus no coordinates and {shuffle_relative.mean():+.2%} versus the average of the two fixed shuffles.", '',
        f"Conditional 95% descriptive intervals: [{intervals[0][0]:+.2%}, {intervals[0][1]:+.2%}] versus none; [{intervals[1][0]:+.2%}, {intervals[1][1]:+.2%}] versus shuffled. These use 2,000 paired seed/block resamples, three seeds and circular 100-bin validation blocks. Validation also selected checkpoints; these are not confirmatory confidence intervals or biological replication.", '',
        f"Positive seed-level curve effects: {(seed_none>0).sum()}/3 versus none and {(seed_shuffle>0).sum()}/3 versus average shuffled. Positive fractions: {(none_relative>0).sum()}/3 and {(shuffle_relative>0).sum()}/3, respectively.", '',
        f"Frozen promising gate (at least 2% mean benefit over none, positive shuffled benefit and consistency across fractions/seeds): **{bool(promising)}**. Lower-data advantage larger than full-data advantage against both controls: **{bool(low_data)}**. The latter is a descriptive interaction check and requires the promising gate before claiming support for the low-data hypothesis.", '',
        '## Design and limitations', '',
        'One MP019 recording, one fixed pool of 2,048 cells, optimizer seeds 10/11/12, and four coordinate conditions. All models retain learned neuron IDs, nonlinear activity embeddings and the same unrestricted 16-summary architecture (81,377 parameters). The no-coordinate arm retains a coordinate MLP receiving zeros, hence a learned common offset. Both shuffled assignments are fixed across time, fractions and seeds.', '',
        'Training prefixes are [0,416), [0,1248), and [0,4160). Windows have eight bins and targets begin at prefix index 31. Validation uses [4260,5564), with targets [4291,5564). Activity and target normalization use only the respective training prefix. Cell eligibility is decided using the smallest prefix; coordinates are standardized using that eligible population. Raw xyz rows and ID/activity alignment are unchanged between conditions.', '',
        'All arms inherit AdamW lr 0.001, weight decay 0.01, batch 32, gradient clip 1, and a 24-epoch cosine schedule to 0.0001. Early stopping requires at least 12 epochs and seven stale validation checks. Best validation checkpoints include epoch zero. No new hyperparameter search was performed: this compares the fixed inherited recipe, not independently optimized model families.', '',
        'Longer prefixes change behavioral coverage, temporal distance to validation, normalization estimates and optimizer-update count as well as data quantity. Overlapping windows are not independent observations. One pool and three technical seeds cannot establish across-animal generalization. This validation segment was already examined in prior work and selects checkpoints here. The test tail is untouched by this pilot, but is already used historically.', '',
        'The 2% cutoff is a predeclared provisional prioritization threshold, not an established biological effect size. No adaptive extra fits or posthoc model selection are authorized by this pilot.', '',
        '## Verification', '',
        f"{result['beat_untrained']}/36 models beat their untrained validation error; {result['beat_constant']}/36 beat their training-mean constant. {result['selected_budget_limit']}/36 selected epoch 24; convergence is not established by the fixed budget.", '',
        f'All saved validation metrics were independently recomputed against original speed values. All 30 new checkpoints were reloaded: three on complete validation sets, the other 27 on 64 examples. Maximum prediction discrepancy was {max_reload:.3g}. Reused fits retain their prior completed audits; exact neuron IDs, coordinates, normalization scalars, initialization hashes and configuration were checked for compatibility.', '',
        'Frozen implementation hashes, application hashes, equal parameter counts, paired initial states, finite saved parameters and minimum-validation checkpoint selection passed. New prefix window/target boundaries were checked before fitting. Application code was not changed.', '',
        '## Reproduction', '',
        'Run `python3 -B pilot.py run` to resume missing fits; completed records are skipped. Run `python3 -B analyze.py` to regenerate this report and audit new checkpoints. The immutable protocol includes dataset/source hashes and all 36 tasks. The copied model files preserve the original implementation.']
    (p.ROOT/'report.md').write_text('\n'.join(report)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for j, label in enumerate(['Correct', 'None', 'Shuffle 1', 'Shuffle 2']):
        axes[0].plot(p.FRACTIONS, means[:, j], marker='o', label=label)
    axes[0].set(xlabel='Training prefix (%)', ylabel='Validation MSE (dataset speed units squared)', title='Fixed decoder; 3 seeds, 1 neuron pool')
    axes[0].legend()
    axes[1].plot(p.FRACTIONS, none_relative*100, marker='o', label='vs none')
    axes[1].plot(p.FRACTIONS, shuffle_relative*100, marker='o', label='vs mean shuffled')
    axes[1].axhline(0, color='gray', linewidth=1)
    axes[1].set(xlabel='Training prefix (%)', ylabel='Correct-coordinate error reduction (%)', title='Positive favors correct coordinates')
    axes[1].legend()
    fig.suptitle('Exploratory validation learning curve; no test evaluation')
    fig.tight_layout()
    fig.savefig(p.ROOT/'learning_curve.png', dpi=160)
    plt.close(fig)
    manifest = {str(path.relative_to(p.ROOT)):p.digest(path) for path in p.ROOT.rglob('*')
                if path.is_file() and path.name != 'manifest.json' and path.suffix != '.log'
                and '__pycache__' not in path.parts}
    p.s.write_json(p.ROOT/'manifest.json', manifest)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
