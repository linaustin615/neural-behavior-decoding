"""Render the completed descriptive robustness diagnostic."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np


ROOT = Path(__file__).resolve().parent
result = json.loads((ROOT / 'results.json').read_text())
protocol = json.loads((ROOT / 'protocol.json').read_text())
audit = json.loads((ROOT / 'audit.json').read_text())
assert audit['passed']
rows = result['rows']


def pct(value):
    return 'undefined' if value is None else f'{100 * value:+.1f}%'


def num(value):
    return 'undefined' if value is None else f'{value:.6f}'


def table(headers, records):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in records])


def save(name, content):
    path = ROOT / name
    path.write_text(content.strip() + '\n')


lines = [
    '# Behavior and time robustness — 2026-10-04',
    'The six-seed transformer-pair comparison retains its overall advantage over matched MLP pairs, including absolute error, but the benefit is not uniform across behavior or time. These are descriptive slices of reused recordings, not independent confirmation or a new passed scientific gate.',
    '## Fixed analysis',
    'Reuse six archived shared-transformer and six matched shared-MLP predictions (seeds 10–15), four mice and 2,218 later windows. Enumerate all 15 unordered two-model pairs per family. Clip each individual output at physical zero before averaging; average pair errors within each mouse, then average relative gains equally across mice. This scores the expected performance of a uniformly selected two-model pair, not a six-model ensemble or a chosen best pair. Model pairings, time windows and slices are dependent; the experiment still contains only four animals.',
    'Freeze analysis source, 22 input/source hashes, thresholds and reporting rules in protocol.json before computing these slices. Global outcomes were already known. Relative speed cutoffs use each mouse’s training median and 90th percentile. Rapid changes exceed the training 90th percentile of absolute consecutive target differences. The first later sample has no preceding target inside this interval and is excluded only from change slices. Quarters are four contiguous near-equal index blocks. These labels do not establish physical rest/movement or acceleration units.',
    'Show all cells; require at least 30 windows per mouse and slice for a slice-level summary. This is a reporting threshold, not evidence that 30 adjacent windows are statistically independent. No fitting, model inference, new seeds, checkpoint selection, hyperparameter search, application changes, confidence intervals or p-values.',
    '## Whole-interval scores',
    'Positive gain means lower transformer error. MSE is squared error; MAE is absolute error. Targets are normalized using each mouse’s training speed statistics. R² compares error with that later interval’s target variance; it does not make the later target mean an available deployment baseline.',
    table(['Mouse', 'N', 'Transformer MSE', 'MLP MSE', 'MSE gain', 'Transformer MAE', 'MLP MAE', 'MAE gain', 'Transformer R²'],
          [[r['mouse'], r['n'], num(r['scores']['all']['attention']['mse']), num(r['scores']['all']['mlp']['mse']),
            pct(r['contrasts']['all']['mse_gain']), num(r['scores']['all']['attention']['mae']), num(r['scores']['all']['mlp']['mae']),
            pct(r['contrasts']['all']['mae_gain']), f"{r['scores']['all']['attention']['r2']:.3f}"] for r in rows]),
    'Equal-mouse mean improvement:19.3% MSE (3/4 mice),9.3% MAE (4/4 mice). The MSE value reproduces the prior study exactly; MAE and the following breakdowns are the new diagnostics.',
    '## Speed and change slices',
    'Each cell gives count, MSE gain and MAE gain. An asterisk marks fewer than 30 windows; those cells remain visible but are excluded from slice summaries. Undefined means no windows. “Typical” only means between the training median and 90th percentile.',
    table(['Mouse', 'Low speed: N; MSE; MAE', 'Typical speed: N; MSE; MAE', 'High speed: N; MSE; MAE', 'Ordinary change: N; MSE; MAE', 'Rapid change: N; MSE; MAE'],
          [[r['mouse']] + [f"{r['contrasts'][s]['n']}{'' if r['contrasts'][s]['supported'] else '*'}; {pct(r['contrasts'][s]['mse_gain'])}; {pct(r['contrasts'][s]['mae_gain'])}"
                          for s in ['low_speed', 'typical_speed', 'high_speed', 'ordinary_change', 'rapid_change']] for r in rows]),
    table(['Slice', 'Eligible mice', 'Mean MSE gain', 'MSE mouse wins', 'Mean MAE gain', 'MAE mouse wins'],
          [[s, ', '.join(result['summaries'][s]['eligible_mice']), pct(result['summaries'][s]['mse']['mean_relative_gain']),
            result['summaries'][s]['mse']['mouse_wins'], pct(result['summaries'][s]['mae']['mean_relative_gain']),
            result['summaries'][s]['mae']['mouse_wins']] for s in ['low_speed', 'typical_speed', 'high_speed', 'ordinary_change', 'rapid_change']]),
    'Only MP033 meets the reporting threshold for either high speed or rapid changes, so neither slice supports an across-mouse conclusion. Its transformer errors are 24.8% worse at high speed and 2.7% worse during rapid changes. MP030’s entire MSE advantage depends on 10 high-speed windows (1.4% of its726 windows): it loses in both lower-speed slices and wins strongly in those 10 windows. MP034 also loses at high speed, with only 16 windows. Do not pool the sparse cells into a claim about fast running.',
    'Low-speed MSE is worse in 3/4 mice even though low-speed MAE is better in 3/4. This is consistent with a tradeoff involving larger errors, not uniform improvement. This analysis does not establish the cause of those errors.',
    '## Contributions to the original MSE endpoint',
    'For each mouse and speed slice: count / total count × (MLP slice MSE − transformer slice MSE) / whole-interval MLP MSE. These contributions add exactly to that mouse’s original relative gain. Their equal-mouse means also add to 19.3%. This avoids confusing the average of slice-relative ratios with a decomposition of the original endpoint.',
    table(['Mouse', 'Low contribution (pp)', 'Typical contribution (pp)', 'High contribution (pp)', 'Total gain (pp)'],
          [[r['mouse']] + [f"{100 * r['speed_contributions'][s]:+.2f}" for s in ['low_speed', 'typical_speed', 'high_speed']] +
           [f"{100 * r['contrasts']['all']['mse_gain']:+.2f}"] for r in rows] +
          [['Equal-mouse mean'] + [f"{100 * result['speed_contributions'][s]:+.2f}" for s in ['low_speed', 'typical_speed', 'high_speed']] + ['+19.30']]),
    'Sparse cells are included in this exact accounting of all windows, but their slice ratios are not promoted to supported comparisons. Low-speed cells contribute+1.31 percentage points overall despite their negative mean slice-relative gain, because the two summaries use different denominators and weights.',
    '## Time stability',
    table(['Mouse', 'Quarter 1: N; MSE; MAE', 'Quarter 2: N; MSE; MAE', 'Quarter 3: N; MSE; MAE', 'Quarter 4: N; MSE; MAE'],
          [[r['mouse']] + [f"{r['contrasts'][s]['n']}; {pct(r['contrasts'][s]['mse_gain'])}; {pct(r['contrasts'][s]['mae_gain'])}"
                          for s in ['quarter_1', 'quarter_2', 'quarter_3', 'quarter_4']] for r in rows]),
    table(['Omitted quarter, in every mouse', 'Remaining mean MSE gain', 'Remaining mean MAE gain'],
          [[s, pct(v['mse']), pct(v['mae'])] for s, v in result['omit_quarter_gains'].items()]),
    'Removing any one corresponding quarter leaves positive aggregate MSE improvement (7.6%–25.9%) and MAE improvement (5.9%–12.9%). That supports aggregate time robustness, not a win in every interval. MP030 quarter 2 has−242.5% relative MSE gain because transformer MSE .015347 exceeds a small MLP MSE .004481; this dominates the negative mean quarter 2 slice ratio. There is no common clock alignment across mice, and the16 mouse-quarter blocks are not 16 independent animals.',
    '## Simple references and practical limits',
    'These references reuse archived ridge predictions and three constants determined without later labels: the original training mean (normalized zero), the median of the eligible training targets, and physical zero speed. All use the same target alignment and nonnegative-speed floor. The ridge comparison is contextual, with different model/training budgets; it is not a matched capacity or compute claim.',
    table(['Reference', 'Mean MSE gain', 'MSE mouse wins', 'Mean MAE gain', 'MAE mouse wins'],
          [[name, pct(v['mse']['mean_relative_gain']), v['mse']['mouse_wins'], pct(v['mae']['mean_relative_gain']), v['mae']['mouse_wins']]
           for name, v in result['baseline_summary'].items()]),
    table(['Mouse', 'Training-median MSE', 'Transformer MSE gain vs median', 'Training-median MAE', 'Transformer MAE gain vs median'],
          [[r['mouse'], num(r['scores']['all']['training_median']['mse']), pct(r['baseline_gains']['training_median']['mse']),
            num(r['scores']['all']['training_median']['mae']), pct(r['baseline_gains']['training_median']['mae'])] for r in rows]),
    'MP032’s38.9% MSE win over MLP is only a2.0% win over a constant training-median speed; its later R² is−.121. Both neural models struggle on that interval despite their relative ranking. The training-median predictor also has lower MAE than the transformer on MP030 and MP032. Conversely, transformer R² is .604 and .816 on MP033 and MP034, and their MSE gains over the training median are 64.6% and 90.0%. Useful learned decoding and weak individual intervals coexist.',
    '## Bias and variable residuals',
    'For each pair separately, MSE = squared mean signed residual + mean squared residual after subtracting that pair’s mean residual. Average those two components over pairs. This is an algebraic decomposition, not the statistical model bias–variance decomposition and not an uncertainty estimate.',
    table(['Mouse', 'Bias-squared contribution to gain (pp)', 'Centered-residual contribution (pp)', 'Transformer signed bias', 'MLP signed bias'],
          [[r['mouse'], f"{100 * r['gain_decomposition']['mean_pair_bias_squared']:+.2f}",
            f"{100 * r['gain_decomposition']['centered_residual_variance']:+.2f}",
            num(r['scores']['all']['attention']['signed_bias']), num(r['scores']['all']['mlp']['signed_bias'])] for r in rows]),
    'Mean contributions are+4.8 percentage points from squared-bias reduction and+14.5 from centered residuals. MP034’s MSE gain comes from lower bias despite worse centered residuals; the same mechanism does not explain all four mice.',
    '## Thresholds and verification',
    table(['Mouse', 'Train N', 'Median (normalized)', 'Speed90th percentile (normalized)', 'Change90th percentile (normalized)'],
          [[r['mouse'], r['thresholds']['train_n'], num(r['thresholds']['median']), num(r['thresholds']['q90']), num(r['thresholds']['change_q90'])] for r in rows]),
    f"All {audit['independent_scalar_scores']:,} scalar MSE/MAE checks pass. All 120 whole-interval pair MSEs reproduce the completed archive. Exact target alignment, disjoint and exhaustive speed/time partitions, first-change exclusion, boundary-tie rules, per-output clipping, error averaging, bias decomposition, contribution sums and all 22 frozen hashes pass. The synthetic check distinguishes average pair error from error of an averaged prediction. Main application files remain byte-for-byte unchanged. Zero new training fits and zero new model inferences.",
    '## Decision',
    'Retain the shared transformer and a fixed two-transformer ensemble as experimental baselines, with matched MLP pairs, ridge and training-median references. The mixed MLP/transformer hybrid gate remains failed. The present evidence supports a promising overall decoding result on these four recordings; it does not support reliable improvement at every speed, on every interval, or a generally superior attention mechanism.',
    'Do not add another architecture or retune to these slices. Before claiming robust running-behavior superiority, evaluate the frozen comparison on recordings or intervals not used in architecture selection that contain enough fast running and speed changes across multiple animals. All available local cohorts have historical reuse, so this diagnostic cannot supply that independent evidence. If further method development uses these same data, label it exploratory and retain the constant-speed and state-specific failure checks. Generation remains deferred. No new fitting sweep, application edit or publication is queued.',
    'Artifacts: [protocol](protocol.json), [results](results.json), [audit](audit.json), [figure](robustness.png), [PDF figure](robustness.pdf). Exact per-pair scores, all reference slice errors, raw denominators and support counts are in results.json.'
]
save('report.md', '\n\n'.join(lines))
save('ASSESSMENT.md', '''# Behavior robustness assessment — 2026-10-04

Complete: no new training or inference; reuse all six seeds, all 15 unordered two-model pairs per family, four mice and 2,218 later windows. Thresholds and analysis source were fixed before scoring these slices; global results and the recordings were already known.

- The transformer pairs retain **19.3% lower mean relative MSE** than matched MLP pairs (3/4 mice) and show **9.3% lower MAE** (4/4). Dropping any one corresponding recording quarter leaves positive mean improvements.
- **The gain is uneven.** MP030’s squared-error advantage depends on 10 high-speed windows. At low speed the transformer has worse MSE in 3/4 mice. Only MP033 has at least 30 high-speed or rapid-change windows, and the transformer loses both slices there.
- **Relative wins can overstate usefulness.** MP032 beats the MLP by 38.9% MSE but improves only 2.0% over a constant training-median predictor and has negative later R². That constant also beats transformer MAE on MP030 and MP032.
- Learned decoding remains useful in the more variable intervals: transformer R² is .604/.816 on MP033/MP034. Against archived ridge, transformer pairs improve mean relative MSE 46.9% and MAE 39.8%, winning all four mice; that contextual comparison does not match model/compute budgets.

Keep the shared transformer as the leading experimental baseline. Do not claim general superiority, robust fast-running decoding, independent significance, or a rescued hybrid. The next scientific requirement is a frozen comparison on data unused in method selection, with enough fast running and changes across animals. Further reuse of these recordings remains exploratory.

All 2,652 scalar metric checks and 120 archived pair-MSE matches passed. Source/input/application hashes are unchanged. No further jobs or application edits are queued. [Full report](report.md) · [Figure](robustness.png) · [Protocol](protocol.json)
''')

fig, (ax, heat) = plt.subplots(2, 1, figsize=(13, 9), gridspec_kw={'height_ratios': [1, 1.7]})
locations = np.arange(4)
ax.bar(locations - .18, [100 * r['contrasts']['all']['mse_gain'] for r in rows], width=.36,
       color='#286a9a', label='Squared error (MSE)')
ax.bar(locations + .18, [100 * r['contrasts']['all']['mae_gain'] for r in rows], width=.36,
       color='#78b0ae', label='Absolute error (MAE)')
ax.axhline(0, color='black', linewidth=.7)
ax.set_xticks(locations, [r['mouse'] for r in rows])
ax.set_ylabel('Error reduction vs MLP (%)')
ax.set_title('Overall gains persist, but vary by mouse and error metric', loc='left', fontsize=13)
ax.legend(frameon=False, loc='upper right')
ax.spines[['top', 'right']].set_visible(False)
labels = ['low_speed', 'typical_speed', 'high_speed', 'rapid_change', 'quarter_1', 'quarter_2', 'quarter_3', 'quarter_4']
values = np.array([[100 * r['contrasts'][s]['mse_gain'] if r['contrasts'][s]['supported'] else np.nan for s in labels] for r in rows])
cmap = plt.get_cmap('RdYlGn').copy()
cmap.set_bad('#e0e0e0')
plot = heat.imshow(values, cmap=cmap, norm=SymLogNorm(linthresh=10, vmin=-250, vmax=250), aspect='auto')
heat.set_xticks(np.arange(len(labels)), ['Low\nspeed', 'Typical\nspeed', 'High\nspeed', 'Rapid\nchange', 'Quarter 1', 'Quarter 2', 'Quarter 3', 'Quarter 4'])
heat.set_yticks(np.arange(4), [r['mouse'] for r in rows])
heat.set_title('MSE gain by fixed slice; gray cells have fewer than 30 windows', loc='left', fontsize=13, pad=14)
for i, row in enumerate(rows):
    for j, label in enumerate(labels):
        cell = row['contrasts'][label]
        text = f"{100 * cell['mse_gain']:+.1f}%\nn={cell['n']}" if cell['supported'] else f"sparse\nn={cell['n']}"
        heat.text(j, i, text, ha='center', va='center', fontsize=9,
                  color='white' if cell['supported'] and abs(values[i, j]) > 90 else 'black')
bar = fig.colorbar(plot, ax=heat, fraction=.025, pad=.025, ticks=[-250, -100, -25, 0, 25, 100, 250])
bar.set_ticklabels(['−250', '−100', '−25', '0', '25', '100', '250'])
bar.set_label('MSE reduction (%)\nSymmetric log color scale')
fig.suptitle('Archived transformer vs matched MLP pairs: behavior robustness', fontsize=16, x=.06, ha='left')
fig.text(.06, .02, 'Six seeds; mean errors of all 15 two-model pairs per family; four reused recordings.\n'
         'Positive values favor transformers. Slices/windows/pairs are dependent; this is descriptive, not independent confirmation.', fontsize=10)
fig.subplots_adjust(left=.07, right=.95, top=.90, bottom=.13, hspace=.50)
for name in ['robustness.png', 'robustness.pdf']:
    fig.savefig(ROOT / name, dpi=160, bbox_inches='tight')
plt.close(fig)
print('Wrote report.md, ASSESSMENT.md, robustness.png and robustness.pdf')
