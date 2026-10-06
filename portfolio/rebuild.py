"""Verify the portable results bundle and rebuild tables/figures without training."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np


HERE = Path(__file__).resolve().parent
T = 'optimized_attention'
MLP = 'optimized_population_mlp'
SMALL = 'default_attention'
LABELS = {T: 'Selected transformer', MLP: 'Selected population MLP',
          SMALL: 'Small transformer', 'optimized_local_mlp': 'Selected local MLP',
          'default_local_mlp': 'Default local MLP', 'matched_population_mlp': 'Matched population MLP',
          'ridge': 'Ridge', 'zero_speed': 'Zero speed', 'training_mean': 'Training mean'}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-11)


def score(prediction, target, lower):
    raw = np.asarray(prediction, dtype=np.float64)
    p = np.maximum(raw, lower)
    require(raw.shape == target.shape and np.isfinite(raw).all(), 'Invalid prediction shape or values')
    e = p - target
    variance = float(np.mean((target - target.mean()) ** 2))
    require(variance > 0, 'R2 is undefined for a constant target')
    mse = float(np.mean(e ** 2))
    return dict(n=len(target), mse=mse, mae=float(np.mean(np.abs(e))),
                r2=1 - mse / variance, raw_mse=float(np.mean((raw-target) ** 2)))


def csv_write(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sign_p(wins, losses):
    n = wins + losses
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2 ** n)


def build(data, out, archive_root=None, figures=True):
    out.mkdir(parents=True, exist_ok=True)
    checksums = read(data / 'checksums.json')
    for file, expected in checksums.items():
        require(hashlib.sha256((data / file).read_bytes()).hexdigest() == expected, f'Bundle checksum mismatch: {file}')
    source_checks = 0
    if archive_root is not None:
        for file, expected in read(data / 'source_provenance.json')['files'].items():
            require(hashlib.sha256((archive_root / file).read_bytes()).hexdigest() == expected, f'Archive changed: {file}')
            source_checks += 1
    bundle = read(data / 'bundle.json')
    reference = read(data / 'reference.json')
    require(bundle['schema_version'] == 1, 'Unsupported bundle schema')
    records, predictions = [], {}
    for case in bundle['cases']:
        cohort, mouse = case['cohort'], case['mouse']
        with np.load(data / case['file'], allow_pickle=False) as z:
            y = z['target'].astype(np.float64)
            require(y.ndim == 1 and len(y) == case['n'] and np.isfinite(y).all(), 'Invalid target')
            require(np.all(y >= case['lower'] - 1e-7), 'Target below physical zero')
            predictions[(cohort, mouse)] = {key: z[key].copy() for key in z.files}
            for role, p in [('ridge', z['ridge']), ('zero_speed', np.full_like(y, case['lower'])), ('training_mean', np.zeros_like(y))]:
                records.append(dict(cohort=cohort, regime='control', mouse=mouse, role=role, seed=None,
                                    **score(p, y, case['lower'])))
            for regime in case['regimes']:
                for role in case['roles']:
                    for seed in case['seeds']:
                        records.append(dict(cohort=cohort, regime=regime, mouse=mouse, role=role, seed=seed,
                                            **score(z[f'{regime}__{role}__{seed}'], y, case['lower'])))

    scalar_checks = 0
    index = {(r['cohort'], r['regime'], r['mouse'], r['role'], r['seed']): r for r in records}
    require(len(index) == len(records), 'Duplicate cases')
    for r in reference['separate_metrics']:
        actual = index[('separate', r['regime'], r['mouse'], r['role'], r['seed'])]
        for key in ('mse', 'mae', 'r2', 'raw_mse', 'n'):
            close(actual[key], r[key])
            scalar_checks += 1
    for r in reference['development']['all']['rows']:
        for role, values in r['scores'].items():
            rows = [index[('development', 'control', r['mouse'], role, None)]] if role == 'ridge' else [
                index[('development', 'shared', r['mouse'], role, seed)] for seed in reference['development']['all']['seeds']]
            for key in ('mse', 'mae', 'r2'):
                close(np.mean([v[key] for v in rows]), values[key])
                scalar_checks += 1
            if role != 'ridge':
                for key in ('mse', 'mae'):
                    close([v[key] for v in rows], values['single_' + key])
                    scalar_checks += len(rows)

    def group(cohort, regime, mouse, role, seeds):
        if role in ('ridge', 'zero_speed', 'training_mean'):
            return [index[(cohort, 'control', mouse, role, None)] for _ in seeds]
        return [index[(cohort, regime, mouse, role, seed)] for seed in seeds]

    contrasts, effects = [], []
    for cohort, regime, subset, seeds, controls, expected in [
        *[('development', 'shared', subset, v['seeds'], list(v['contrasts']), v['contrasts'])
          for subset, v in reference['development'].items()],
        *[('separate', regime, 'all', [401, 402, 403], [MLP, SMALL, 'ridge'],
           {r['comparator']: r for r in reference['separate']['contrasts'] if r['regime'] == regime})
          for regime in ('independent', 'shared')],
    ]:
        mice = [c['mouse'] for c in bundle['cases'] if c['cohort'] == cohort]
        for control in controls:
            mse_gains, mae_gains, wins = [], [], 0
            for mouse in mice:
                a = group(cohort, regime, mouse, T, seeds)
                b = group(cohort, regime, mouse, control, seeds)
                mse_gain = 1 - np.mean([v['mse'] for v in a]) / np.mean([v['mse'] for v in b])
                mae_gain = 1 - np.mean([v['mae'] for v in a]) / np.mean([v['mae'] for v in b])
                mse_gains.append(float(mse_gain))
                mae_gains.append(float(mae_gain))
                wins += sum(x['mse'] < y['mse'] for x, y in zip(a, b))
                effects.append(dict(cohort=cohort, regime=regime, subset=subset, mouse=mouse,
                                    comparator=control, relative_mse_gain=float(mse_gain), relative_mae_gain=float(mae_gain)))
            mouse_wins = sum(v > 0 for v in mse_gains)
            if cohort == 'separate':
                gate = reference['separate_practical_gate']
                passed = np.mean(mse_gains) >= gate['mean_relative_mse_gain_min'] and mouse_wins >= gate['mouse_wins_min'] and wins >= gate['individual_seed_wins_min'] and min(mse_gains) >= -gate['max_mouse_relative_harm'] and np.mean(mae_gains) >= gate['mean_relative_mae_gain_min']
                expected_values = [expected[control]['mean_relative_mse_gain'], expected[control]['mean_relative_mae_gain'], expected[control]['mouse_wins'], expected[control]['individual_seed_wins']]
                expected_pass = expected[control]['practical_pass']
            else:
                passed = np.mean(mse_gains) >= .05 and mouse_wins >= 3 and wins >= math.ceil(8*len(seeds)/3) and min(mse_gains) >= -.1 and np.mean(mae_gains) >= 0
                expected_values = [expected[control]['mean_mse_gain'], expected[control]['mean_mae_gain'], expected[control]['mouse_wins'], expected[control]['seed_wins']]
                expected_pass = expected[control]['practical_gain_passed']
            close([np.mean(mse_gains), np.mean(mae_gains), mouse_wins, wins], expected_values)
            require(bool(passed) == expected_pass, 'Practical gate changed during export')
            scalar_checks += 5
            p = sign_p(mouse_wins, sum(v < 0 for v in mse_gains)) if cohort == 'separate' and regime == 'independent' else None
            contrasts.append(dict(cohort=cohort, regime=regime, subset=subset, comparator=control,
                                  mean_relative_mse_gain=float(np.mean(mse_gains)), mean_relative_mae_gain=float(np.mean(mae_gains)),
                                  mouse_wins=mouse_wins, mice=len(mice), seed_wins=wins, seed_cases=len(mice)*len(seeds),
                                  worst_mouse_gain=min(mse_gains), practical_pass=bool(passed), two_sided_sign_p=p, holm_p=None))
    primary = [r for r in contrasts if r['two_sided_sign_p'] is not None]
    previous = 0.0
    for rank, r in enumerate(sorted(primary, key=lambda v: v['two_sided_sign_p'])):
        r['holm_p'] = max(previous, min(1.0, (len(primary)-rank)*r['two_sided_sign_p']))
        previous = r['holm_p']
        expected = next(v for v in reference['separate']['contrasts'] if v['regime'] == 'independent' and v['comparator'] == r['comparator'])
        close([r['two_sided_sign_p'], r['holm_p']], [expected['two_sided_sign_p'], expected['holm_p']])
        scalar_checks += 2

    mouse_metrics = []
    for case in bundle['cases']:
        for regime in case['regimes']:
            zero = index[(case['cohort'], 'control', case['mouse'], 'zero_speed', None)]['mse']
            mean = index[(case['cohort'], 'control', case['mouse'], 'training_mean', None)]['mse']
            for role in case['roles'] + ['ridge', 'zero_speed', 'training_mean']:
                rows = group(case['cohort'], regime, case['mouse'], role, case['seeds'])
                mse = float(np.mean([r['mse'] for r in rows]))
                mouse_metrics.append(dict(cohort=case['cohort'], regime=regime, mouse=case['mouse'], role=role,
                                          mse=mse, mae=float(np.mean([r['mae'] for r in rows])),
                                          r2=float(np.mean([r['r2'] for r in rows])),
                                          mse_over_zero=mse/zero, mse_over_training_mean=mse/mean))

    csv_write(out / 'per_seed_metrics.csv', records)
    csv_write(out / 'per_mouse_metrics.csv', mouse_metrics)
    csv_write(out / 'per_mouse_effects.csv', effects)
    csv_write(out / 'contrasts.csv', contrasts)
    write(out / 'comparisons.json', contrasts)
    lines = ['# Reproduced results', '',
             'Recomputed from the portable saved predictions. Positive gain means lower transformer error; negative gain means higher error. Each mouse receives equal weight after averaging individual-seed errors. These are not ensemble scores.', '',
             '| Cohort / fitting | Comparator | MSE gain | MAE gain | Mouse wins | Seed wins | Practical gate | Adjusted p |',
             '| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |']
    for r in contrasts:
        if r['subset'] != 'all':
            continue
        p = f"{r['holm_p']:.3f}" if r['holm_p'] is not None else 'Not tested'
        lines.append(f"| {r['cohort']} / {r['regime']} | {LABELS[r['comparator']]} | {100*r['mean_relative_mse_gain']:+.2f}% | {100*r['mean_relative_mae_gain']:+.2f}% | {r['mouse_wins']}/{r['mice']} | {r['seed_wins']}/{r['seed_cases']} | {'PASS' if r['practical_pass'] else 'FAIL'} | {p} |")
    lines += ['', 'The development optimization criterion additionally required both seed subsets to pass. Its overall result remains **FAIL**, even though the pooled small-transformer comparison passes. A practical pass on reused development mice is not independent significance.', '',
              '## Separate cohort: absolute accuracy', '',
              '| Mouse | Transformer R² | MLP R² | Ridge R² | Transformer MSE / zero speed |',
              '| --- | ---: | ---: | ---: | ---: |']
    for case in [c for c in bundle['cases'] if c['cohort'] == 'separate']:
        values = {r['role']: r for r in mouse_metrics if r['cohort'] == 'separate' and r['regime'] == 'independent' and r['mouse'] == case['mouse']}
        lines.append(f"| {case['mouse']} | {values[T]['r2']:.3f} | {values[MLP]['r2']:.3f} | {values['ridge']['r2']:.3f} | {values[T]['mse_over_zero']:.2f}× |")
    lines += ['', 'R² compares with the test-mean oracle reference; zero speed is a deployable constant control. Below-zero R² means worse error than that test-mean reference. Test-mean labels are never supplied to the models.', '',
              'All seven mice are included. Fitting is within each new recording; weights are trained anew. This is recipe validation, not zero-shot transfer. Separate releases were not harmonized in physical time or publisher preprocessing.', '',
              'All per-seed values, all six development finalists, both separate-cohort fitting regimes and both development seed subsets are retained in the CSV files. No new hypothesis test or model selection is introduced.']
    (out / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    if figures:
        plot(out, bundle, reference, predictions, contrasts, effects, mouse_metrics)
    verification = dict(passed=True, prediction_metric_rows=len(records),
                        archived_scalar_values_matched=scalar_checks, compact_files_hashed=len(checksums),
                        archive_source_hashes_checked=source_checks, cohorts=2, development_mice=4, separate_mice=7,
                        scientific_gates_changed=False, new_model_fits=0, new_model_inferences=0,
                        scope='Packaging/reproduction checks only; original training and data audits are not rerun',
                        numpy_version=np.__version__, figures_generated=figures)
    write(out / 'verification.json', verification)
    print(f"PASS: {len(records)} metric rows; {scalar_checks} archived values; no training or model inference")
    return verification


def plot(out, bundle, reference, predictions, contrasts, effects, mouse_metrics):
    with tempfile.TemporaryDirectory(prefix='neuron-portfolio-mpl-') as cache:
        os.environ.setdefault('MPLCONFIGDIR', cache)
        os.environ.setdefault('XDG_CACHE_HOME', cache)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                             'axes.spines.right': False, 'figure.facecolor': '#fcfcfa', 'axes.facecolor': '#fcfcfa',
                             'savefig.facecolor': '#fcfcfa', 'pdf.fonttype': 42})
        colors = {T: '#2065a8', MLP: '#c57924', SMALL: '#805ba6', 'ridge': '#56756c'}

        def save(fig, name):
            fig.savefig(out / f'{name}.png', dpi=180, bbox_inches='tight')
            fig.savefig(out / f'{name}.pdf', bbox_inches='tight', metadata={'CreationDate': None})
            plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), gridspec_kw={'width_ratios': [1.1, 1.1, 1]})
        for ax, cohort, regime, title in [
            (axes[0], 'development', 'shared', 'Development: 4 reused mice\n6 training seeds; shared fits'),
            (axes[1], 'separate', 'independent', 'Separate cohort: 7 mice\n3 seeds; independent fits')]:
            for y, role in enumerate([MLP, SMALL, 'ridge']):
                row = next(r for r in contrasts if r['cohort'] == cohort and r['regime'] == regime and r['subset'] == 'all' and r['comparator'] == role)
                vals = [100*r['relative_mse_gain'] for r in effects if r['cohort'] == cohort and r['regime'] == regime and r['subset'] == 'all' and r['comparator'] == role]
                ax.scatter(vals, y + np.linspace(-.10, .10, len(vals)), s=24, color='#999999', alpha=.8)
                ax.scatter(100*row['mean_relative_mse_gain'], y, s=85, marker='D', color=colors[T], zorder=4)
                ax.text(.99, y-.24, f"{100*row['mean_relative_mse_gain']:+.1f}%  ({row['mouse_wins']}/{row['mice']} mice)", transform=ax.get_yaxis_transform(), va='center', ha='right', fontsize=9)
            ax.axvline(0, color='#777777', linewidth=.8)
            ax.set_yticks(range(3), ['vs population MLP', 'vs small transformer', 'vs ridge'])
            ax.set_ylim(2.55, -.55)
            ax.set_xlim(-25, 85)
            ax.set_title(title, loc='left', fontweight='bold', fontsize=11)
            ax.set_xlabel('Transformer relative MSE gain (%)\nPositive favors transformer')
            ax.grid(axis='x', alpha=.15)
        roles = [T, MLP, 'optimized_local_mlp']
        times = [1000*next(r['median_across_seed_medians'] for r in reference['timing']['summaries'] if r['role'] == role and r['batch'] == 64) for role in roles]
        axes[2].barh(range(3), times, color=[colors[T], colors[MLP], '#929292'], height=.5)
        axes[2].set_yticks(range(3), ['Population T', 'Population MLP', 'Local MLP'])
        axes[2].invert_yaxis()
        axes[2].set_xlim(0, 105)
        for i, time in enumerate(times):
            axes[2].text(time + 1.5, i, f'{time:.2f}', va='center')
        axes[2].set_title('Population compression is cheaper\nSaved CPU benchmark; batch 64', loc='left', fontweight='bold', fontsize=11)
        axes[2].set_xlabel('Milliseconds per forward batch\nSame measured benefit for both families')
        fig.suptitle('Neural behavior decoding: strong controls change the conclusion', fontsize=16, x=.03, ha='left', y=1.04)
        fig.subplots_adjust(wspace=.85, bottom=.22)
        fig.text(.03, -.03, 'Gray dots = individual mice; diamonds = equal-mouse means, not confidence intervals. Separate-cohort gates all FAIL (Holm p = 1).\nDevelopment is exploratory. CPU timings exclude preprocessing and I/O; quality is not equal across architectures. Releases are not time-harmonized.', fontsize=9)
        save(fig, 'overview')

        mice = [c['mouse'] for c in bundle['cases'] if c['cohort'] == 'separate']
        fig, ax = plt.subplots(figsize=(10, 5.2))
        for offset, role in zip([-.24, -.08, .08, .24], [T, MLP, SMALL, 'ridge']):
            values = [next(r['mse_over_zero'] for r in mouse_metrics if r['cohort'] == 'separate' and r['regime'] == 'independent' and r['mouse'] == mouse and r['role'] == role) for mouse in mice]
            ax.scatter(np.arange(len(mice)) + offset, values, color=colors[role], s=55, label=LABELS[role], zorder=3)
        ax.axhline(1, color='#333333', ls='--', linewidth=1)
        ax.set_yscale('log')
        ax.set_xticks(range(len(mice)), mice)
        ax.set_ylabel('Model MSE / zero-speed MSE (log scale)')
        ax.set_ylim(.13, 55)
        ax.set_title('Relative model wins can coexist with failure against zero speed', loc='left', fontweight='bold', pad=18)
        ax.text(.01, .97, 'Above 1: zero speed is better', transform=ax.transAxes, va='top')
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.22), ncol=2, frameon=False)
        ax.grid(axis='y', alpha=.15)
        fig.subplots_adjust(bottom=.34, top=.86)
        fig.text(.08, .015, 'All seven separate mice; average individual-seed MSE. No mice or seeds excluded.\nTX104 and TX61 fail for every tested recipe and ridge. Low-running test periods are an observed condition; the cause is not established.', fontsize=9)
        save(fig, 'zero_speed_control')

        fig, axes = plt.subplots(7, 1, figsize=(12, 15))
        for ax, case in zip(axes, [c for c in bundle['cases'] if c['cohort'] == 'separate']):
            z = predictions[('separate', case['mouse'])]
            x = np.arange(case['n'])
            ax.plot(x, z['target']-case['lower'], color='#222222', lw=.7, alpha=.85, label='Observed')
            for role in (T, MLP):
                p = np.stack([np.maximum(z[f'independent__{role}__{seed}'].astype(np.float64), case['lower']) - case['lower'] for seed in case['seeds']])
                ax.fill_between(x, p.min(0), p.max(0), color=colors[role], alpha=.10, linewidth=0)
                ax.plot(x, p.mean(0), color=colors[role], lw=.7, alpha=.9, label=LABELS[role])
            ax.set_xlim(0, len(x)-1)
            ax.set_ylim(bottom=0)
            ax.set_title(case['mouse'], loc='left', fontsize=11, fontweight='bold', pad=3)
            ax.grid(alpha=.12)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(.99, .968), ncol=3, frameon=False, fontsize=8)
        axes[-1].set_xlabel('Native-frame offset from the first scored test target (each full interval)')
        fig.supylabel('Running above physical zero, measured in each mouse’s training-target SD', x=.035)
        fig.suptitle('All seven test recordings: observed running and fixed-recipe predictions', fontsize=15, y=.995)
        fig.tight_layout(rect=(.045, .05, 1, .985), h_pad=1.3)
        fig.text(.075, .012, 'Lines show means of three independently trained predictions; shading shows their range, not a confidence interval. No smoothing or window selection.\nFigures illustrate saved predictions; reported primary scores average individual-seed errors, not ensemble errors. Vertical scales differ by mouse.\nNative frames are not converted to seconds. These traces do not demonstrate forecasting, causality, or generated neural activity.', fontsize=9)
        save(fig, 'all_mouse_traces')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=HERE / 'data')
    parser.add_argument('--output', type=Path, default=HERE / 'results')
    parser.add_argument('--archive-root', type=Path, help='Optionally verify original local source receipts')
    parser.add_argument('--no-figures', action='store_true')
    args = parser.parse_args()
    build(args.data.resolve(), args.output.resolve(), args.archive_root.resolve() if args.archive_root else None, not args.no_figures)
