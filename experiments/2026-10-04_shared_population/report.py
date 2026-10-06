"""Render the completed shared population-block comparison."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
result = json.loads((ROOT / 'results.json').read_text())
protocol = json.loads((ROOT / 'protocol.json').read_text())
audit = json.loads((ROOT / 'audit.json').read_text())
locked = json.loads((ROOT / 'selection_lock.json').read_text())
assert audit['passed']
rows, summaries, gates = result['rows'], result['summaries'], result['gates']
LABELS = dict(baseline='Original shared transformer', attention='+ population attention',
              mixer='+ population mixer', prior_mlp='Archived shared temporal MLP')
CONTRASTS = dict(attention_vs_baseline='Population attention vs baseline',
                 attention_vs_mixer='Population attention vs population mixer',
                 mixer_vs_baseline='Population mixer vs baseline',
                 attention_vs_prior_mlp='Population attention vs archived temporal MLP',
                 mixer_vs_prior_mlp='Population mixer vs archived temporal MLP')


def pct(value):
    return f'{100 * value:+.1f}%'


def passed(value):
    return 'PASS' if value else 'FAIL'


def table(headers, records):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, r)) + ' |' for r in records])


lines = [
    '# Population block on the shared baseline — 2026-10-04',
    f"Population-attention full practical gate: **{passed(gates['attention_full'])}**. Population-mixer secondary gate: **{passed(gates['mixer_secondary'])}**. These are descriptive development-cohort comparisons, not independent statistical significance.",
    '## Question and architecture',
    'Does explicit neuron-to-neuron attention add running-speed decoding value to the accepted shared temporal transformer? The three primary arms are the unchanged shared transformer, that same decoder with one population-attention block, and that same decoder with one nonattention population-mixing block. The last arm still has the original temporal and behavior-query attention; it is not an all-MLP network.',
    'The baseline processes 128 neuron histories, each with eight four-bin patches at width 16. The new block sees the latest temporal state from every neuron: a batch-by-128-by-16 tensor. Each state can already summarize its neuron’s 32-bin history. The block replaces only these latest states. The earlier seven states per neuron, the behavior query over all 1,024 tokens, and the 64 population mean/std history features remain. No future observations or behavior values enter the input. Different examples and recordings are never mixed together.',
    'The attention block uses two-head self-attention across the 128 neurons. The control applies a learned 128→4→128 GELU MLP across neuron slots separately for each feature. Both use the same residual structure, LayerNorm and 16→32→16 GELU feedforward branch. Total parameter counts are 20,561 and 20,629, versus 18,337 for baseline. Added-block counts are 2,224 and 2,292. Common parent tensors and new norm/feedforward tensors start identically. The control is one fixed low-rank nonlinear mixer; parameter matching does not exhaust nonattention alternatives or equalize operation counts. Its neuron-slot mixing is shared across mice without a claim that same-index cells are biological counterparts.',
    '## Fixed training and selection',
    'Twelve new fits, two additions × six seeds 10–15. Each receives 24 epochs, 5,688 AdamW updates and 179,712 training presentations. Exact archived baseline initial tensors, data, batches, training-only normalization, optimizer settings, loss weights and intact-validation selection rule are reused. Six completed native transformer fits are reused, with no baseline retraining. New models start from initial tensors, not from trained baseline checkpoints. Added operations consume different dropout random draws, so dropout probability matches but individual dropout masks are not claimed identical.',
    'Select one checkpoint at the lowest original joint validation score among epochs 0–24. All 12 choices are locked before current later inference. Do not choose a pair, epoch combination, population placement, rank or model using later outcomes. Main application files, earlier studies, coordinates and the failed neuron-masking recipe are unchanged.',
    '## Prespecified comparisons',
    'For each arm, clip each single model output at physical zero, average every distinct pair of seeds, and average errors over all 15 pairs within each mouse. Then average within-mouse relative error gains equally across the four mice. This is expected two-model-pair performance, not a selected pair or a six-model ensemble. Seed wins use individual models; pairs, seeds and overlapping windows are not independent animals.',
    'The attention addition must beat both baseline and population mixer by at least 5% mean relative MSE, win at least three of four mouse averages, and win at least 16 of 24 paired single-seed comparisons for each contrast. It must also avoid more than 10% harm on any mouse versus baseline. All conditions are required. The mixer gets the same baseline comparison and harm guard as a secondary question. These are engineering criteria, not significance thresholds.',
    table(['Comparison', 'Mean MSE gain', 'Mouse wins', 'Paired-seed wins', 'Mean MAE gain', 'Practical contrast'],
          [[CONTRASTS[k], pct(summaries[k]['mean_mse_gain']), f"{summaries[k]['mouse_wins']}/4",
            f"{summaries[k]['seed_wins']}/24", pct(summaries[k]['mean_mae_gain']), passed(gates[k])]
           for k in ['attention_vs_baseline', 'attention_vs_mixer', 'mixer_vs_baseline']]),
    table(['Requirement', 'Result'], [[name, passed(gates[name])] for name in
          ['attention_no_mouse_over10pct_harm', 'mixer_no_mouse_over10pct_harm', 'attention_full', 'mixer_secondary']]),
    '## Individual mice',
    table(['Mouse', 'N', 'Baseline MSE', '+ attention MSE', '+ mixer MSE', 'Attention gain vs baseline', 'Attention gain vs mixer', 'Mixer gain vs baseline'],
          [[r['mouse'], r['n']] + [f"{r['scores'][g]['mse']:.6f}" for g in ['baseline', 'attention', 'mixer']] +
           [pct(r['contrasts'][k]['mse_gain']) for k in ['attention_vs_baseline', 'attention_vs_mixer', 'mixer_vs_baseline']] for r in rows]),
    table(['Mouse', 'Baseline MAE', '+ attention MAE', '+ mixer MAE', 'Baseline R²', '+ attention R²', '+ mixer R²'],
          [[r['mouse']] + [f"{r['scores'][g]['mae']:.6f}" for g in ['baseline', 'attention', 'mixer']] +
           [f"{r['scores'][g]['r2']:.3f}" for g in ['baseline', 'attention', 'mixer']] for r in rows]),
    'MSE/MAE use training-standardized speed units. R² uses the later target variance as a descriptive reference; the later target mean is not an available training/deployment baseline.',
    '## Single models and sensitivity to mice',
    table(['Comparison', 'Single-model mean MSE gain', 'Pair wins', 'Leave-one-mouse-out mean pair MSE gains'],
          [[CONTRASTS[k], pct(v['single_mean_mse_gain']), f"{v['pair_wins']}/60", ', '.join(pct(x) for x in v['leave_one_mouse_out'])]
           for k, v in summaries.items() if k in ['attention_vs_baseline', 'attention_vs_mixer', 'mixer_vs_baseline']]),
    'The leave-one-out entries omit MP030, MP032, MP033 and MP034 in that order. These are sensitivity summaries, not a license to remove an unfavorable recording. Raw individual-seed and pair scores are in results.json; no favorable seed or pair is chosen.',
    '## Contextual controls',
    table(['Comparison', 'Mean MSE gain', 'Mouse wins', 'Mean MAE gain'],
          [[CONTRASTS[k], pct(summaries[k]['mean_mse_gain']), f"{summaries[k]['mouse_wins']}/4", pct(summaries[k]['mean_mae_gain'])]
           for k in ['attention_vs_prior_mlp', 'mixer_vs_prior_mlp']]),
    'The archived shared temporal-MLP decoder is a contextual reference. It differs in temporal mixing, behavior-query routing and parameter count. The capacity-close population-mixer arm is the direct control for the new population block.',
    table(['Mouse', 'Training-mean MSE', 'Training-median MSE', '+ attention MSE', '+ mixer MSE'],
          [[r['mouse'], f"{r['references']['training_mean']['mse']:.6f}", f"{r['references']['training_median']['mse']:.6f}",
            f"{r['scores']['attention']['mse']:.6f}", f"{r['scores']['mixer']['mse']:.6f}"] for r in rows]),
    table(['Model', 'Single-model wins over initial constant output'], [[LABELS[g], f'{v}/24'] for g, v in result['single_wins_vs_initial'].items()]),
    'Initial output is normalized zero, the training speed mean. Learning versus this reference is separate from beating an already useful trained baseline.',
    '## Training records',
    table(['Added block', 'Seed', 'Selected epoch', 'Updates', 'Presentations', 'Population gradient max', 'Fit seconds'],
          [[r['variant'], r['seed'], r['selected_epoch'], r['updates'], r['examples'], f"{r['population_gradient_max']:.4f}", f"{r['elapsed_seconds']:.1f}"] for r in locked['records']]),
    'Fit times are observed wall times while workers share the machine, not a controlled compute-efficiency benchmark. Equal update budgets do not imply equal floating-point work or optimum tuning for every architecture.',
    '## Verification and limits',
    f"All 12 new fits completed their allocated budgets. Preflight reproduced 48 trained-parent predictions exactly with the addition bypassed, verified preserved earlier states, cross-neuron influence, no cross-example mixing, finite gradients and exact reloads. All {audit['selection_scores_checked']:,} validation mouse/epoch scores, matched batch orders and actual optimizer counters pass. Later evaluation completed {audit['new_later_predictions']:,} new predictions, 24 exact baseline first-batch checks and {audit['independent_scalar_scores']:,} independent scalar pair/single MSE/MAE checks. Inputs, checkpoint tensors, source files and main application hashes remain unchanged.",
    'This tests one latest-state population block, not every neuron-attention design or mixing at all time steps. Extra norm/feedforward layers accompany both additions; the direct attention-versus-mixer comparison is needed to distinguish attention from generic extra processing. Attention weights are not biological connections and any predictive benefit would not establish a causal co-firing mechanism. Four historically reused mice and repeated seeds cannot provide new independent animal-level confirmation. No additional width, placement, rank, seed or loss search is appended after outcomes.',
    'Artifacts: [assessment](ASSESSMENT.md), [protocol](protocol.json), [model](models.py), [selection lock](selection_lock.json), [results](results.json), [audit](audit.json), [figure](population.png), [PDF](population.pdf).'
]
(ROOT / 'report.md').write_text('\n\n'.join(lines) + '\n')

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
contrasts = ['attention_vs_baseline', 'attention_vs_mixer', 'mixer_vs_baseline']
axes[0].bar(np.arange(3), [100 * summaries[k]['mean_mse_gain'] for k in contrasts], color=['#286a9a', '#5c8dad', '#be6c28'])
axes[0].set(xticks=np.arange(3), xticklabels=['Attention\nvs baseline', 'Attention\nvs mixer', 'Mixer\nvs baseline'], title='Mean change in squared error')
for offset, key, label, color in [(-.18, 'attention_vs_baseline', 'Attention addition', '#286a9a'), (.18, 'mixer_vs_baseline', 'Mixer addition', '#be6c28')]:
    axes[1].bar(np.arange(4) + offset, [100 * r['contrasts'][key]['mse_gain'] for r in rows], width=.36, label=label, color=color)
axes[1].set(xticks=np.arange(4), xticklabels=[r['mouse'] for r in rows], title='Effect versus baseline by mouse')
axes[1].legend(frameon=False)
for ax in axes:
    ax.axhline(0, color='black', linewidth=.8)
    ax.set_ylabel('MSE reduction (%)')
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Explicit neuron mixing on the shared transformer baseline', x=.06, ha='left')
fig.text(.06, .02, 'Four reused mice, six matched seeds, mean errors across all 15 two-model pairs.\n'
         'Positive values favor the named addition. Practical criteria are not independent statistical confirmation.', fontsize=9)
fig.tight_layout(rect=[0, .10, 1, .93])
for name in ['population.png', 'population.pdf']:
    fig.savefig(ROOT / name, dpi=160, bbox_inches='tight')
plt.close(fig)
print('Wrote report.md, population.png and population.pdf')
