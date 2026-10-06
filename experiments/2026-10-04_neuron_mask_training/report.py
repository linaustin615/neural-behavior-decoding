"""Render the completed fixed masking recipe comparison."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
result = json.loads((ROOT / 'results.json').read_text())
audit = json.loads((ROOT / 'audit.json').read_text())
locked = json.loads((ROOT / 'selection_lock.json').read_text())
assert audit['passed']
rows, summaries, gates = result['rows'], result['summaries'], result['gates']
CONDITIONS = ['clean', 'missing_16', 'missing_32', 'missing_64']
LABELS = dict(clean='Intact', missing_16='12.5% missing', missing_32='25% missing', missing_64='50% missing')
GROUPS = ['native_attention', 'augmented_attention', 'native_mlp', 'augmented_mlp']


def pct(v):
    return f'{100 * v:+.1f}%'


def passed(v):
    return 'PASS' if v else 'FAIL'


def table(headers, records):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, r)) + ' |' for r in records])


lines = [
    '# Whole-neuron masking during training — 2026-10-04',
    f"Transformer combined practical gate: **{passed(gates['attention']['combined_pass'])}**. MLP combined practical gate: **{passed(gates['mlp']['combined_pass'])}**. The original shared transformer is the accepted baseline; this study tests one augmentation candidate. Practical gates are not independent statistical confirmation.",
    '## Fixed comparison',
    'Twelve new shared fits: unchanged temporal transformer and matched static-query temporal MLP, six seeds 10–15. Each uses 24 epochs, 5,688 AdamW updates and 179,712 training-example exposures. All corresponding native fits are reused, with exact matched initial tensors and example orders. The architecture, 128-neuron panel, 32-bin histories, training-only normalization, target, optimizer, loss weighting and clean checkpoint-selection rule are unchanged. Both models retain population mean/std histories and recording-specific neuron-ID/session embeddings. No coordinates or observed behavior are supplied as inputs.',
    'Each training example independently has probability 0.5 of receiving a uniformly sampled missing set of exactly 32 neurons. Replace those entire histories with normalized zero (their training means), otherwise keep the example intact. Masks are redrawn every batch/epoch and identical across families at matched seed/epoch/batch. There is no inverse-probability scaling, new missingness indicator, token removal or change to IDs. Both neural-token and summary paths see the imputed activity. The independent NumPy mask generator does not consume the torch dropout RNG. This deliberately changes the available training information; counts of presentations and optimizer steps remain matched.',
    'Select one checkpoint from epochs 0–24 using the original intact-validation criterion. Lock all 12 choices before current later inference. No missing-data validation chooses the epoch, and no pair, rate, weight or architecture is selected using later scores.',
    'Evaluate all models on intact inputs and a new bank of five fixed nested masks per mouse at 16/32/64 missing histories. A mask remains fixed over a recording and is shared by every model. The new bank avoids selecting the exact masks that motivated the study, but it does not create new animals or repair historical reuse of these recordings. Existing native clean predictions are reused, with 48 exact first-batch reproduction checks; native missing-input predictions are newly computed for the new mask bank.',
    'For each condition, bound individual speed outputs at physical zero, average each of all 15 distinct two-model pairs, and average pair errors across the five mask views. Then average within-mouse relative gains equally over four mice. This is expected pair performance, not a six-model prediction or an ensemble of corrupted views. Paired-seed wins use single-model errors averaged over views. Masks, pairs, seeds and overlapping windows are dependent; four mice remain the cohort units.',
    '## Prespecified practical decision',
    'At 25% missingness, augmentation must improve mean relative MSE by at least 10%, help at least three mice and win at least 16 of 24 paired single-seed comparisons. The intact-input safeguard allows at most 5% mean MSE harm and at most 10% harm for any mouse. Both conditions must pass. These are chosen engineering tolerances, not a statistical noninferiority test. The same gate is applied to the MLP as a secondary comparison.',
    table(['Family', 'Missing-input robustness', 'Intact-input safeguard', 'Combined'],
          [[f, passed(gates[f]['robustness_pass']), passed(gates[f]['clean_safeguard_pass']), passed(gates[f]['combined_pass'])] for f in ['attention', 'mlp']]),
    '## Augmentation versus corresponding native model',
    'Positive values mean augmentation reduces error on the same evaluation condition. This compares absolute performance under matched missingness, rather than rewarding a smaller degradation caused by already-poor intact performance.',
]
for f in ['attention', 'mlp']:
    cells = summaries[f + '_augmentation']
    lines += [f'### {f}',
              table(['Inputs', 'Mean MSE gain', 'Mouse wins', 'Paired-seed wins', 'Mean MAE gain', 'Leave-one-mouse-out mean MSE gains'],
                    [[LABELS[c], pct(cells[c]['mean_mse_gain']), f"{cells[c]['mouse_wins']}/4", f"{cells[c]['seed_wins']}/24",
                      pct(cells[c]['mean_mae_gain']), ', '.join(pct(v) for v in cells[c]['leave_one_mouse_out'])] for c in CONDITIONS])]
lines += [
    '## Does the augmented transformer beat the equally augmented MLP?',
    table(['Inputs', 'Transformer MSE gain', 'Mouse wins', 'Paired-seed wins', 'Transformer MAE gain'],
          [[LABELS[c], pct(summaries['augmented_attention_vs_mlp'][c]['mean_mse_gain']),
            f"{summaries['augmented_attention_vs_mlp'][c]['mouse_wins']}/4", f"{summaries['augmented_attention_vs_mlp'][c]['seed_wins']}/24",
            pct(summaries['augmented_attention_vs_mlp'][c]['mean_mae_gain'])] for c in CONDITIONS]),
    f"The secondary architecture gate at 25% missingness requires at least 5% mean MSE gain, three mouse wins and 16 paired-seed wins: **{passed(gates['augmented_attention_advantage_25pct'])}**. Improving a transformer relative to itself is not evidence of an advantage over an equally treated MLP.",
    'For context, the native-model comparison below uses this study’s new mask bank. Its corrupted-input values need not equal the previous three-mask stress test.',
    table(['Inputs', 'Native transformer MSE gain vs native MLP', 'Mouse wins'],
          [[LABELS[c], pct(summaries['native_attention_vs_mlp'][c]['mean_mse_gain']),
            f"{summaries['native_attention_vs_mlp'][c]['mouse_wins']}/4"] for c in CONDITIONS]),
    '## Individual mice',
]
for c in CONDITIONS:
    lines += [f'### {LABELS[c]}',
              table(['Mouse', 'Native transformer MSE', 'Augmented transformer MSE', 'Native MLP MSE', 'Augmented MLP MSE', 'Transformer augmentation gain', 'MLP augmentation gain'],
                    [[r['mouse']] + [f"{r['scores'][c][g]['mse']:.6f}" for g in GROUPS] +
                     [pct(r['contrasts']['attention_augmentation'][c]['mse_gain']), pct(r['contrasts']['mlp_augmentation'][c]['mse_gain'])] for r in rows])]
lines += [
    '## Missingness degradation relative to each model’s intact score',
    table(['Model', '12.5% missing MSE increase', '25% missing MSE increase', '50% missing MSE increase'],
          [[g] + [pct(result['degradation'][g][c]) for c in CONDITIONS[1:]] for g in GROUPS]),
    '## Variation across missing-neuron views',
    'For each of the five fixed view indices, the following values average within-mouse treatment gains after averaging pair errors. They are descriptive sensitivity checks, not confidence limits or independent replications.',
    table(['Family', 'Missingness', 'Five view-specific mean MSE gains'],
          [[f, LABELS[c], ', '.join(pct(v) for v in summaries[f + '_augmentation'][c]['view_mean_gains'])]
           for f in ['attention', 'mlp'] for c in CONDITIONS[1:]]),
    '## Simple references and learning',
    table(['Mouse', 'Training-mean MSE', 'Training-median MSE', 'Augmented transformer intact MSE', 'Augmented transformer 25% missing MSE'],
          [[r['mouse'], f"{r['references']['training_mean']['mse']:.6f}", f"{r['references']['training_median']['mse']:.6f}",
            f"{r['scores']['clean']['augmented_attention']['mse']:.6f}", f"{r['scores']['missing_32']['augmented_attention']['mse']:.6f}"] for r in rows]),
    table(['Model', 'Intact single-model wins over initial normalized-zero output'],
          [[g, f"{result['clean_single_wins_vs_initial'][g]}/24"] for g in GROUPS]),
    'Initial output is normalized zero (the training speed mean), not physical zero speed. Constant references do not observe later labels. They are contextual checks rather than equal-capacity or equal-training-budget architectures.',
    '## Training records',
    table(['Family', 'Seed', 'Selected epoch', 'Masked examples', 'Total presentations', 'Training seconds'],
          [[r['family'], r['seed'], r['selected_epoch'], r['masked_examples'], r['examples'], f"{r['elapsed_seconds']:.1f}"] for r in locked['records']]),
    'All checkpoints use intact validation. Stochastic augmentation means approximately half, rather than exactly half, of the training presentations are masked. The same totals and mask hashes must agree between transformer and MLP for each seed.',
    '## Verification and limitations',
    f"All 12 fits completed; actual Adam counters verify {audit['updates_per_fit']:,} updates per fit. All {audit['selection_scores_checked']:,} mouse/epoch selection scores, exact parent initializations, matched example orders, matched family mask hashes, selected-model reloads, 48 archived clean-output checks, {audit['independent_scalar_scores']:,} scalar MSE/MAE checks and frozen source/input/application hashes pass. There were {audit['new_later_predictions']:,} new later predictions, plus the first-batch compatibility checks. Full raw prediction arrays and pair/view/single scores are retained.",
    'This is one augmentation recipe selected in response to an observed weakness on four historically reused recordings. It tests synthetic mean-imputed missing observations, not neuron death or unseen-mouse transfer. Training uses changing per-example masks while evaluation holds each missing panel fixed over time. No explicit missingness signal distinguishes imputed zero from observed mean activity. The clean safeguard allows small degradation by design; passing it does not mean identical predictions or zero cost. No independent significance, universal transformer superiority or causal-connectivity claim follows.',
    'No masking-rate, loss, checkpoint criterion or architecture search is extended after these results. Main application files and old studies remain unchanged. See the assessment for the decision about this candidate; the original shared transformer baseline is preserved.',
    'Artifacts: [protocol](protocol.json), [selection lock](selection_lock.json), [evaluation masks](evaluation_plans.json), [numeric results](results.json), [audit](audit.json), [figure](mask_training.png), [PDF](mask_training.pdf), [assessment](ASSESSMENT.md).'
]
(ROOT / 'report.md').write_text('\n\n'.join(lines) + '\n')

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
for offset, f, label, color in [(-.18, 'attention', 'Transformer', '#286a9a'), (.18, 'mlp', 'MLP', '#be6c28')]:
    axes[0].bar(np.arange(4) + offset, [100 * summaries[f + '_augmentation'][c]['mean_mse_gain'] for c in CONDITIONS], width=.36, label=label, color=color)
    axes[1].bar(np.arange(4) + offset, [100 * r['contrasts'][f + '_augmentation']['missing_32']['mse_gain'] for r in rows], width=.36, color=color)
axes[0].set(xticks=np.arange(4), xticklabels=['Intact', '12.5%', '25%', '50%'], xlabel='Missing neuron histories', title='Mean augmentation benefit')
axes[1].set(xticks=np.arange(4), xticklabels=[r['mouse'] for r in rows], title='Benefit at 25% missing, by mouse')
for ax in axes:
    ax.axhline(0, color='black', linewidth=.8)
    ax.set_ylabel('MSE reduction versus native model (%)')
    ax.spines[['top', 'right']].set_visible(False)
axes[0].legend(frameon=False)
fig.suptitle('One fixed neuron-masking training recipe', x=.06, ha='left')
fig.text(.06, .02, 'Four reused mice, six matched seeds, all 15 model pairs, five fixed missing panels per mouse.\n'
         'Positive values favor augmentation. Practical criteria are not independent statistical confirmation.', fontsize=9)
fig.tight_layout(rect=[0, .10, 1, .93])
for name in ['mask_training.png', 'mask_training.pdf']:
    fig.savefig(ROOT / name, dpi=160, bbox_inches='tight')
plt.close(fig)
print('Wrote report.md, mask_training.png and mask_training.pdf')
