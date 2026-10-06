"""Report the complete fixed shared-model continuation study."""
import json
from pathlib import Path
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/neuron_finetuning_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
read = lambda name: json.loads((ROOT / name).read_text())
r, audit, review = read('results.json'), read('audit.json'), read('review.json')
lock, recipes = read('selection_lock.json'), read('recipe_lock.json')
LABELS = dict(native_attention='Original transformer', native_mlp='Original MLP',
    tuned_attention='Fine-tuned transformer', tuned_mlp='Fine-tuned MLP')
label = lambda key: ' vs '.join(LABELS[g] for g in key.split('_vs_'))
pct = lambda x: f'{100*x:+.1f}%'
passed = lambda x: 'PASS' if x else 'FAIL'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
        ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


lines = [
    '# Shared decoder fine-tuning — 2026-10-05',
    'Completed eighteen fine-tuning fits without changing either architecture. Twelve initial search fits used seeds 10–12; six replication fits used independently chosen family recipes on seeds 13–15. No original model was retrained from scratch.',
    table(['Final practical decision', 'Result'], [[k, passed(v)] for k, v in r['decisions'].items()]),
    '## Frozen search and matched controls',
    'Each fit starts from its own archived selected shared checkpoint. Two initial learning rates, 1e-4 and 3e-5, use eight additional epochs, AdamW with fresh optimizer state, weight decay .01, clipping 1 and cosine decay to one tenth of the starting rate. Each receives 1,896 updates and 59,904 training presentations. Batches are the original deterministic generator at epoch indices 25–32; dropout seed is seed+19000. Data, normalization, loss weights, neuron panel, temporal context, target and architecture are unchanged.',
    'The ordinary model and a running average of floating model tensors are evaluated after each epoch. The average includes the starting checkpoint and every epoch-end checkpoint so far; fixed masks remain unchanged. This is a single network with averaged weights, not prediction ensembling or averaging separately initialized models. It adds no gradient fits. The same options are available to both families.',
    'Each mode selects one joint epoch 0–8 using the original equal-mouse earlier bounded MSE divided by fixed ridge denominators. Epoch zero is the unchanged native checkpoint. One rate/mode per family is chosen by mean selection score over seeds 10–12. That recipe is fixed for seeds 13–15, which may select their own epoch by the same earlier rule. All choices lock before current later inference. Losing recipes are never scored on the later period.',
    table(['Family', 'Selected rate', 'Selected mode', 'Earlier search score'],
        [[k, v['rate'], v['mode'], f"{v['score']:.6f}"] for k, v in r['choices'].items()]),
    table(['Family', 'Rate', 'Mode', 'Earlier search score'],
        [[family, v['rate'], v['mode'], f"{v['score']:.6f}"] for family, records in recipes['candidates'].items() for v in records]),
    '## Required replication',
    'On additional seeds 13–15, the tuned transformer must improve at least 5% in equal-mouse mean relative pair MSE, win at least three mouse means and eight of twelve individual seed comparisons against BOTH the unchanged transformer and the equally tuned MLP. No mouse may have more than 10% MSE harm versus the original transformer, and mean MAE must not worsen. The same requirements apply to all-six results, with sixteen of twenty-four seed wins. Original or pooled scores cannot rescue failed additional-seed replication. MLP improvement over its own native model is secondary, using the same own-family criteria.',
    'Individual outputs are bounded at physical zero before averaging each two-model pair. Errors are averaged across all distinct pairs within each mouse; relative changes are then averaged equally over four mice. Each three-seed subset has three pairs; all-six has fifteen. Seed wins use individual models, not selected pairs. A passed practical gate would still require a separately frozen additional-seed check before calling the optimization effect robust; it would not create independent animal-level significance.',
]
for subset, title in [('additional', 'Additional seeds 13–15: primary'), ('original', 'Original seeds 10–12: search subset'), ('all', 'All six seeds: consistency guard')]:
    part = r['subsets'][subset]
    block = part['later']
    lines += [
        '## ' + title,
        table(['Comparison', 'Mean MSE gain', 'Mouse wins', 'Single-seed wins', 'Mean MAE gain', 'Contrast'],
            [[label(k), pct(v['mean_mse_gain']), f"{v['mouse_wins']}/4", f"{v['seed_wins']}/{4*len(part['seeds'])}",
              pct(v['mean_mae_gain']), passed(v['contrast_passed'])] for k, v in block['summaries'].items()]),
        table(['Requirement', 'Result'], [[k, passed(v)] for k, v in block['gates'].items()]),
        table(['Mouse', 'N', 'Original transformer MSE', 'Tuned transformer MSE', 'Original MLP MSE', 'Tuned MLP MSE'],
            [[v['mouse'], v['n']] + [f"{v['scores'][g]['mse']:.6f}" for g in ['native_attention', 'tuned_attention', 'native_mlp', 'tuned_mlp']] for v in block['rows']]),
        table(['Comparison', 'Single-model mean gain', 'Leave-one-mouse-out pair gains'],
            [[label(k), pct(v['single_mean_mse_gain']), ', '.join(pct(x) for x in v['leave_one_mouse_out'])] for k, v in block['summaries'].items() if k in ['tuned_attention_vs_native_attention', 'tuned_attention_vs_tuned_mlp']]),
    ]
all_results = r['subsets']['all']['later']
lines += [
    '## Individual seeds and earlier results',
    table(['Comparison', 'Seed', 'Individual mean MSE gain', 'Mouse wins'],
        [[label(k), v['seed'], pct(v['mean_mse_gain']), f"{v['mouse_wins']}/4"] for k in ['tuned_attention_vs_native_attention', 'tuned_attention_vs_tuned_mlp'] for v in all_results['summaries'][k]['individual_seeds']]),
    table(['Subset', 'Earlier comparison', 'Mean MSE gain', 'Mouse wins', 'Seed wins'],
        [[subset, label(k), pct(v['mean_mse_gain']), v['mouse_wins'], v['seed_wins']] for subset, part in r['subsets'].items()
         for k, v in part['selection']['summaries'].items() if k in ['tuned_attention_vs_native_attention', 'tuned_mlp_vs_native_mlp']]),
    'Earlier scores are optimistic because they selected starting checkpoints, recipes and continuation checkpoints. An earlier gain alone is not evidence of later improvement.',
    table(['Mouse', 'Original transformer R²', 'Tuned transformer R²', 'Training-mean MSE', 'Training-median MSE'],
        [[v['mouse'], f"{v['scores']['native_attention']['r2']:.3f}", f"{v['scores']['tuned_attention']['r2']:.3f}",
          f"{v['controls']['training_mean']['mse']:.6f}", f"{v['controls']['training_median']['mse']:.6f}"] for v in all_results['rows']]),
    'MSE and MAE are based on training-standardized speed. Later variance is the descriptive R² denominator; predicting the later mean would not be an available trained baseline.',
    '## Training record and checks',
    table(['Family', 'Seed', 'Rate', 'Native epoch', 'Plain epoch', 'Averaged epoch', 'Fit seconds'],
        [[v['family'], v['seed'], v['rate'], v['starting_epoch'], v['selected_epochs']['plain'], v['selected_epochs']['averaged'], f"{v['elapsed_seconds']:.1f}"] for v in lock['records']]),
    f"All starts reproduce the native selected validation outputs exactly ({audit['exact_native_starts']} mouse/model checks). Parameter averaging matches an independently calculated mean and preserves fixed buffers and the training model. All updates and training counts are checked, batches match across recipes/families, and selected reloads are exact. Locking checked {audit['selection_scores_checked']} selection scores. Evaluation produced {audit['new_predictions']:,} new predictions with exact targets and preserved inputs/models. Analysis independently checked {audit['scalar_errors_checked']} scalar errors and {audit['archived_prediction_arrays_exact']} archived prediction arrays. Final review passed {review['aggregate_gate_accounting_checks']} aggregate/gate/accounting checks, {review['source_input_hashes']} source/input hashes and {review['selected_artifact_hashes']} selected-artifact hashes. No main application file changed.",
    'All four mice and later periods were historically searched. Seeds 13–15 test recipe transfer across existing random initializations, not unseen animals. Neither a favorable result nor a practical gate supplies independent significance. A failed recipe does not prove that every fine-tuning method or architecture is inadequate. No learning rates, averaging rules, epochs or seeds were appended to this study after results.',
    'Artifacts: [assessment](ASSESSMENT.md), [protocol](protocol.json), [recipe lock](recipe_lock.json), [selection lock](selection_lock.json), [results](results.json), [audit](audit.json), [review](review.json), [PNG](finetuning.png), [PDF](finetuning.pdf).',
]
(ROOT / 'report.md').write_text('\n\n'.join(lines) + '\n')
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
keys = ['tuned_attention_vs_native_attention', 'tuned_attention_vs_tuned_mlp', 'tuned_mlp_vs_native_mlp']
for offset, subset, name, color in [(-.24, 'original', 'Search seeds 10–12', '#9db2c0'), (0, 'additional', 'Additional seeds 13–15', '#216c9c'), (.24, 'all', 'All six seeds', '#c88b48')]:
    axes[0].bar(np.arange(3)+offset, [100*r['subsets'][subset]['later']['summaries'][k]['mean_mse_gain'] for k in keys], width=.24, label=name, color=color)
axes[0].set(xticks=range(3), xticklabels=['Tuned transformer\nvs original', 'Tuned transformer\nvs tuned MLP', 'Tuned MLP\nvs original MLP'], title='Mean relative pair-MSE reduction', ylabel='MSE reduction (%)')
axes[0].legend(frameon=False, fontsize=8)
for offset, group, name, color in [(-.18, 'tuned_attention', 'Transformer', '#216c9c'), (.18, 'tuned_mlp', 'MLP', '#c88b48')]:
    native = group.replace('tuned_', 'native_')
    rows = all_results['rows']
    axes[1].bar(np.arange(4)+offset, [100*(1-v['scores'][group]['mse']/v['scores'][native]['mse']) for v in rows], width=.36, label=name, color=color)
axes[1].set(xticks=range(4), xticklabels=[v['mouse'] for v in rows], title='All-six gain versus own original model', ylabel='MSE reduction (%)')
axes[1].legend(frameon=False)
for ax in axes:
    ax.axhline(0, color='black', linewidth=.8)
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Shared decoder: fixed fine-tuning and weight averaging', x=.06, ha='left')
fig.text(.06, .02, 'Positive favors fine-tuning. Four historically reused mice; seeds and model pairs are not independent animals.\nRecipes chosen on earlier data only; failed additional-seed replication cannot be rescued by pooled results.', fontsize=9)
fig.tight_layout(rect=[0, .11, 1, .94])
for suffix in ['png', 'pdf']:
    fig.savefig(ROOT / f'finetuning.{suffix}', dpi=160, bbox_inches='tight')
plt.close(fig)
print('Report and figures saved', flush=True)
