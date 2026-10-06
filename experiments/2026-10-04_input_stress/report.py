"""Render completed frozen stress tests without changing numerical analysis."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
result = json.loads((ROOT / 'results.json').read_text())
audit = json.loads((ROOT / 'audit.json').read_text())
assert audit['passed']
rows, summaries = result['rows'], result['summaries']
LABELS = dict(clean='Original inputs', missing_16='12.5% missing neurons',
              missing_32='25% missing neurons', missing_64='50% missing neurons',
              order_all='Reorder old patches: both paths',
              order_tokens='Reorder old patches: token path only', all_mean='All neurons at training means')


def pct(v):
    return f'{100 * v:+.1f}%'


def table(headers, records):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, r)) + ' |' for r in records])


lines = [
    '# Saved decoder input stress tests — 2026-10-04',
    'Three new checks: missing recorded neurons, dependence on the order of earlier activity, and CPU inference cost. These are frozen-model diagnostics on previously examined recordings. No model was retrained, selected or modified.',
    '## Fixed design',
    'Four mice, six seeds (10–15), both shared model families, and 2,218 later windows. Replace 16, 32 or 64 of 128 complete neuron histories with normalized zero, their training means. Use three fixed random masks per mouse, nested across missingness levels and identical for all seeds and architectures. A mask stays fixed throughout a recording. IDs and all 128 input slots remain. This tests a specific mean-imputation policy, not actual removal, biological silencing, or a decoder trained for missing neurons.',
    'For the time test, derange the oldest seven four-bin patches and preserve the newest four-bin patch. Apply the same three fixed orders to every neuron and window; retain within-patch time order and synchronous population values. First change both input paths. Then repeat with the exact original 64 population mean/std features restored immediately before the unchanged output head. The second condition isolates perturbation of the learned token pathway while preserving the summary shortcut. It still changes the inputs associated with learned lag positions and creates artificial histories.',
    'An all-training-means control keeps learned neuron, time and session embeddings but removes all variable activity. Clean predictions are reused; only the first 64 windows per mouse/model are recomputed to verify exact checkpoint/data compatibility. Inputs, trained tensors and old studies remain unchanged.',
    'For every condition, clip each model output at physical zero, average each of the 15 distinct two-model seed pairs, and average their errors over the three perturbation views. Average within-mouse relative changes equally across the four mice. This is expected performance across pairs/views, not a 6-model ensemble or an average over corrupted views. No pair/mask is selected. Repeated seeds, masks and adjacent windows are dependent; there are four historically reused animals. No new success threshold, confidence interval or significance test is introduced.',
    '## Aggregate comparison',
    'Positive error increase means damage relative to that same family on clean inputs. Positive transformer gain means lower error than the matched MLP under the same perturbation.',
    table(['Condition', 'Transformer MSE increase', 'MLP MSE increase', 'Transformer MSE gain vs MLP', 'Mouse wins', 'Transformer MAE gain vs MLP'],
          [[LABELS[c], pct(s['attention']['mse_increase']), pct(s['mlp']['mse_increase']),
            pct(s['mean_mse_gain']), f"{s['mse_mouse_wins']}/4", pct(s['mean_mae_gain'])] for c, s in summaries.items()]),
    '## Per-mouse effects',
]
for c in LABELS:
    lines += [f'### {LABELS[c]}',
              table(['Mouse', 'Transformer MSE', 'MLP MSE', 'Transformer MSE change vs clean', 'MLP MSE change vs clean', 'Transformer gain vs MLP', 'Transformer gain vs training median'],
                    [[r['mouse'], f"{r['scores'][c]['attention']['mse']:.6f}", f"{r['scores'][c]['mlp']['mse']:.6f}",
                      pct(r['contrasts'][c]['attention']['mse_increase']), pct(r['contrasts'][c]['mlp']['mse_increase']),
                      pct(r['contrasts'][c]['mse_gain']), pct(r['contrasts'][c]['attention']['mse_gain_vs_median'])] for r in rows])]
lines += [
    '## Variation across fixed perturbations',
    'Each value below is an equal-mouse mean MSE increase over clean inputs, after averaging pair errors. The three entries correspond to the three prespecified view indices, not independent experimental repeats or confidence limits. All per-pair/per-view scores are preserved in results.json.',
    table(['Condition', 'Transformer views', 'MLP views'],
          [[LABELS[c], ', '.join(pct(v) for v in s['attention']['view_mean_mse_increases']),
            ', '.join(pct(v) for v in s['mlp']['view_mean_mse_increases'])] for c, s in summaries.items() if c not in ['clean', 'all_mean']]),
    '## Dependence on historical order',
    table(['Condition', 'Mouse', 'Transformer mean absolute output change', 'MLP mean absolute output change'],
          [[LABELS[c], r['mouse'], f"{r['scores'][c]['attention']['mean_abs_prediction_change']:.6f}",
            f"{r['scores'][c]['mlp']['mean_abs_prediction_change']:.6f}"] for c in ['order_all', 'order_tokens'] for r in rows]),
    'Output displacements use training-standardized speed units. Error increases show sensitivity of these fitted decoders to this perturbation. They do not prove the original temporal order is indispensable after retraining, establish a biological mechanism, or distinguish a reliance on smooth histories from learned dynamics. Error reductions under corruption are diagnostic observations, not permission to deploy a corruption chosen using these later outcomes.',
    '## CPU cost',
    table(['Batch size', 'Transformer median forward ms', 'MLP median forward ms', 'Median matched-seed ratio'],
          [[batch, f"{t['attention_median_ms']:.3f}", f"{t['mlp_median_ms']:.3f}", f"{t['median_paired_ratio']:.2f}x"] for batch, t in result['timing'].items()]),
    'Measured sequentially after both stress workers finished, CPU only, two torch threads and one interop thread. All 12 saved models used the first 64 MP030 windows. Three warmups and ten timed repetitions per model/batch, with independently randomized interleaving each round. Table entries are medians across the six per-model medians. Individual measurements, 10th/90th percentiles and parameter counts are in benchmark.json. Transformer/MLP have 18,337/18,327 parameters. Timing covers a model forward only: no preprocessing, I/O, training, device transfer or accelerator comparison. Two-model ensembles require two forwards; no physical real-time claim is made because acquisition timing is not established here.',
    '## Verification and scope',
    f"All {audit['new_stress_predictions']:,} new stress predictions completed with zero training fits. All 48 clean first-batch checks exactly matched saved predictions. All {audit['independent_scalar_metrics']:,} independent scalar MSE/MAE checks passed, along with target alignment, input/model preservation and frozen hashes. Synthetic checks verify nested masks, untouched retained neurons, whole-patch permutation, latest-patch preservation, the exact population-summary bypass, hook cleanup, physical-zero clipping before pair averaging, and averaging errors rather than perturbed predictions.",
    'The experiment addresses input sensitivity and local runtime. It does not fix poor new-mouse transfer, uneven fast-running performance, the dependence on stable neuron IDs, or reuse of the cohort. It does not revisit coordinates or establish cross-neuron causal connectivity. No architecture or main application change follows automatically from the stress scores.',
    'Artifacts: [protocol](protocol.json), [perturbation plans](plans.json), [numeric results](results.json), [timing](benchmark.json), [audit](audit.json), [figure](stress.png), [PDF](stress.pdf), [assessment](ASSESSMENT.md).'
]
(ROOT / 'report.md').write_text('\n\n'.join(lines) + '\n')

fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
for family, label, color in [('attention', 'Transformer', '#286a9a'), ('mlp', 'Matched MLP', '#be6c28')]:
    values = [100 * summaries[c][family]['mse_increase'] for c in ['clean', 'missing_16', 'missing_32', 'missing_64']]
    axes[0].plot([0, 12.5, 25, 50], values, marker='o', label=label, color=color)
axes[0].set(xlabel='Neuron histories replaced by training means (%)', ylabel='Mean relative MSE increase (%)', title='Missing recorded neurons')
axes[0].set_yscale('symlog', linthresh=10)
axes[0].legend(frameon=False)
for offset, family, label, color in [(-.18, 'attention', 'Transformer', '#286a9a'), (.18, 'mlp', 'Matched MLP', '#be6c28')]:
    axes[1].bar(np.arange(2) + offset, [100 * summaries[c][family]['mse_increase'] for c in ['order_all', 'order_tokens']], width=.36, color=color)
    axes[2].bar(np.arange(2) + offset, [result['timing'][b][family + '_median_ms'] for b in ['1', '64']], width=.36, color=color)
axes[1].set(xticks=[0, 1], xticklabels=['Both input paths', 'Token path only'], ylabel='Mean relative MSE increase (%)', title='Reordered earlier activity')
axes[2].set(xticks=[0, 1], xticklabels=['Batch 1', 'Batch 64'], ylabel='Milliseconds / model forward', title='CPU inference cost')
for ax in axes:
    ax.axhline(0, color='black', linewidth=.6)
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Saved shared decoder: three fixed practical checks', x=.05, ha='left')
fig.text(.05, .015, 'Four reused mice, six seeds, all 15 pairs, three perturbation views; no retraining or independent significance.\n'
         'Missing-neuron chart uses a symmetric log scale. Timing is local CPU forward cost only.', fontsize=9)
fig.tight_layout(rect=[0, .10, 1, .93])
for name in ['stress.png', 'stress.pdf']:
    fig.savefig(ROOT / name, dpi=160, bbox_inches='tight')
plt.close(fig)
print('Wrote report.md, stress.png and stress.pdf')
