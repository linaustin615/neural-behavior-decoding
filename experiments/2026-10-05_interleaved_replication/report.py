"""Render the fixed interleaved replication results without selecting seeds."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
read = lambda name: json.loads((ROOT / name).read_text())
r, audit, lock = read('results.json'), read('audit.json'), read('selection_lock.json')
LABELS = dict(baseline='Original transformer', free_dynamic='Unrestricted dynamic readout',
    free_static='Unrestricted static readout', dynamic='Interleaved dynamic',
    static='Interleaved static', prior_mlp='Archived shared MLP')
label = lambda key: ' vs '.join(LABELS[g] for g in key.split('_vs_'))
pct = lambda x: f'{100*x:+.1f}%'
passed = lambda x: 'PASS' if x else 'FAIL'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
        ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


lines = [
    '# Focused interleaved replication — 2026-10-05',
    'Completed nine new fits: exact interleaved dynamic model on seeds 13–15, and a matched static readout on seeds 10–15. Reused dynamic seeds 10–12 and all archived comparators. No completed fit or archived later-period inference was repeated.',
    table(['Final practical decision', 'Result'], [[k, passed(v)] for k, v in r['decisions'].items()]),
    '## Fixed design and interpretation',
    'Each of four queries reads patch positions (0,4), (1,5), (2,6), or (3,7) across all 128 neurons. Patches contain four activity bins; each query reads 256 of the 1,024 source tokens. The dynamic model inherits the exact previous interleaved forward. The new static model changes only readout key input to learned neuron/time/session embeddings. Values remain activity-dependent. Both retain the original causal temporal attention, four queries, 64 population mean/std history features and the same output head. Both have 21,553 parameters and identical starting tensors.',
    'Static readout is a temporal transformer with fixed-per-recording readout weights, not an all-MLP model. Dynamic-vs-static tests the readout weighting contribution under this training recipe. It does not isolate all attention or establish biological connectivity. Causal token states can contain earlier activity; masks separate readout positions, not raw histories. The global statistics shortcut sees full history.',
    'Each new fit receives 24 epochs, 5,688 AdamW updates and 179,712 training presentations with the original batches, loss weights, optimizer, dropout seed, preprocessing and joint earlier checkpoint selection. The archived training function is reused without source edits. All 12 interleaved selections, including three reused models, locked before current later inference. Earlier selection scores below are descriptive and optimistic because they selected checkpoints.',
    '## Replication criteria',
    'Additional seeds 13–15 are the primary replication. Utility requires at least 5% mean relative MSE improvement, at least three of four mouse wins and eight of twelve individual paired-seed wins versus BOTH the original one-query transformer and the matching unrestricted four-query model. No mouse may be more than 10% worse than the original transformer. Dynamic routing additionally requires those improvement thresholds against interleaved static readout. Static utility is secondary.',
    'Retention also requires the same utility criteria across all six seeds, with sixteen of twenty-four individual seed wins. The all-six aggregate and original seeds cannot rescue a failed additional-seed replication. Dynamic routing must pass in both additional and all-six comparisons. These are frozen practical development criteria, not statistical significance tests.',
    'Bound each individual prediction at physical zero, then average each distinct two-model pair. Average all pair errors within a mouse and relative improvements equally across four mice. Each three-seed subset contains three pairs per mouse; all-six contains fifteen. This reports average pair performance, not a selected pair or a single six-model ensemble. Individual seed wins use single models. Mice, seeds and overlapping pairs are not interchangeable independent samples.',
]

for subset, title in [('additional', 'Primary: additional seeds 13–15'), ('original', 'Original seeds 10–12, previously observed dynamic models'), ('all', 'All six seeds, secondary consistency guard')]:
    part = r['subsets'][subset]
    p = part['later']
    n = 4 * len(part['seeds'])
    lines += [
        '## ' + title,
        table(['Later comparison', 'Mean MSE gain', 'Mouse wins', 'Single-seed wins', 'Mean MAE gain', 'Contrast'],
            [[label(k), pct(v['mean_mse_gain']), f"{v['mouse_wins']}/4", f"{v['seed_wins']}/{n}", pct(v['mean_mae_gain']), passed(v['contrast_passed'])] for k, v in p['summaries'].items()]),
        table(['Subset requirement', 'Result'], [[k, passed(v)] for k, v in p['gates'].items()]),
        table(['Mouse', 'N', 'Baseline MSE', 'Dynamic MSE', 'Static MSE', 'Dynamic gain', 'Static gain'],
            [[v['mouse'], v['n']] + [f"{v['scores'][g]['mse']:.6f}" for g in ['baseline', 'dynamic', 'static']] +
             [pct(1 - v['scores'][g]['mse'] / v['scores']['baseline']['mse']) for g in ['dynamic', 'static']] for v in p['rows']]),
        table(['Comparison', 'Single-model mean MSE gain', 'Leave-one-mouse-out pair gains'],
            [[label(k), pct(p['summaries'][k]['single_mean_mse_gain']), ', '.join(pct(v) for v in p['summaries'][k]['leave_one_mouse_out'])]
             for k in ['dynamic_vs_baseline', 'static_vs_baseline', 'dynamic_vs_static']]),
    ]

all_results = r['subsets']['all']['later']
lines += [
    '## Individual seeds and reference predictors',
    'Individual-seed relative gains average equally over mice. This differs from the primary pair-error aggregation. No seed is selected or discarded after outcomes.',
    table(['Comparison', 'Seed', 'Mean individual MSE gain', 'Mouse wins'],
        [[label(k), v['seed'], pct(v['mean_mse_gain']), f"{v['mouse_wins']}/4"]
         for k in ['dynamic_vs_baseline', 'static_vs_baseline', 'dynamic_vs_static'] for v in all_results['summaries'][k]['individual_seeds']]),
    table(['Mouse', 'Training-mean MSE', 'Training-median MSE', 'Baseline R²', 'Dynamic R²', 'Static R²'],
        [[v['mouse'], f"{v['controls']['training_mean']['mse']:.6f}", f"{v['controls']['training_median']['mse']:.6f}"] +
         [f"{v['scores'][g]['r2']:.3f}" for g in ['baseline', 'dynamic', 'static']] for v in all_results['rows']]),
    table(['Model', 'Single-model wins over initial training-mean output'],
        [[LABELS[g], f'{v}/24'] for g, v in all_results['single_wins_vs_initial'].items()]),
    'MSE/MAE use training-standardized speed. Later variance supplies only the descriptive R² denominator; a predictor using the later mean would not be available at training. The archived MLP changes temporal processing and readout; it is context, not the matched static-readout control.',
    '## Earlier checkpoint-selection period',
    'These scores are descriptive and did not trigger extra fitting or replace the frozen later replication gate.',
    table(['Seed subset', 'Comparison', 'Mean selection MSE gain', 'Mouse wins', 'Single-seed wins'],
        [[subset, label(k), pct(v['mean_mse_gain']), f"{v['mouse_wins']}/4", f"{v['seed_wins']}/{4*len(part['seeds'])}"]
         for subset, part in r['subsets'].items() for k, v in part['selection']['summaries'].items()
         if k in ['dynamic_vs_baseline', 'static_vs_baseline', 'dynamic_vs_static']]),
    '## Verification and limits',
    table(['Variant', 'Seed', 'Reused', 'Selected epoch', 'Fit seconds'],
        [[v['variant'], v['seed'], v['reused'], v['record']['selected_epoch'], f"{v['record']['elapsed_seconds']:.1f}"] for v in lock['records']]),
    f"Preflight checked exact archived dynamic-wrapper predictions, matching initial weights and masks, invariant static readout weights with activity-dependent predictions, valid gradients and reloads. Training checked every batch order, optimizer steps, counts and selected-checkpoint reloads. Locking checked {audit['selection_scores_checked']} selection scores. Evaluation created {audit['new_later_predictions']:,} new predictions and reused all archived predictions without inference. Analysis independently checked {audit['scalar_errors_independently_checked']} scalar errors and {audit['archived_prediction_arrays_exact']} archived arrays across the reported subsets. Frozen inputs, sources, application files, checkpoints and prediction hashes were verified. See review.json for the final independent aggregate/gate audit.",
    'This follow-up was motivated by a favorable secondary result after substantial prior architecture exploration. All four Stringer mice and later periods were historically reused. Seeds 13–15 are additional for this architecture, while their comparator outcomes were already known. Replication can establish training-seed robustness conditional on these recordings; it cannot establish independent animal-level significance, unseen-mouse transfer or architectural novelty. No p-values are calculated. No post-result seed expansion, alternative masks, optimizer tuning, main application migration, publication, generation or reconstruction resumption occurred.',
    'Artifacts: [assessment](ASSESSMENT.md), [protocol](protocol.json), [results](results.json), [selection lock](selection_lock.json), [audit](audit.json), [review](review.json), [figure](interleaved.png), [PDF](interleaved.pdf).',
]
(ROOT / 'report.md').write_text('\n\n'.join(lines) + '\n')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
keys = ['dynamic_vs_baseline', 'static_vs_baseline', 'dynamic_vs_static']
for offset, subset, title, color in [(-.24, 'original', 'Original seeds 10–12', '#92a8b8'),
    (0, 'additional', 'Additional seeds 13–15', '#216c9c'), (.24, 'all', 'All six seeds', '#d08a45')]:
    axes[0].bar(np.arange(3) + offset, [100*r['subsets'][subset]['later']['summaries'][k]['mean_mse_gain'] for k in keys],
        width=.24, label=title, color=color)
axes[0].set(xticks=range(3), xticklabels=['Dynamic\nvs baseline', 'Static\nvs baseline', 'Dynamic\nvs static'], title='Average two-model pair performance', ylabel='Mean relative MSE reduction (%)')
axes[0].legend(frameon=False, fontsize=8)
for offset, key, title, color in [(-.18, 'dynamic_vs_baseline', 'Dynamic', '#216c9c'), (.18, 'static_vs_baseline', 'Static', '#d08a45')]:
    rows = all_results['summaries'][key]['individual_seeds']
    axes[1].bar(np.arange(6) + offset, [100*v['mean_mse_gain'] for v in rows], width=.36, color=color, label=title)
axes[1].axvline(2.5, color='#888888', linestyle='--', linewidth=.8)
axes[1].set(xticks=range(6), xticklabels=range(10, 16), title='Individual models versus baseline', xlabel='Training seed', ylabel='Mean relative MSE reduction (%)')
axes[1].legend(frameon=False, fontsize=9)
for ax in axes:
    ax.axhline(0, color='black', linewidth=.8)
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Interleaved readout: seed replication and static control', x=.06, ha='left')
fig.text(.06, .02, 'Additional seeds are the primary replication; original and pooled results cannot rescue failure.\nFour historically reused mice. Positive values favor the named candidate; seeds are not independent animals.', fontsize=9)
fig.tight_layout(rect=[0, .11, 1, .94])
for suffix in ['png', 'pdf']:
    fig.savefig(ROOT / f'interleaved.{suffix}', dpi=160, bbox_inches='tight')
plt.close(fig)
print('Report and figures saved')
