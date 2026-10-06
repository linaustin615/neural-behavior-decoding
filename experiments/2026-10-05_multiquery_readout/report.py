"""Render the completed multiquery readout screen."""
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
screen = json.loads((ROOT / 'screen.json').read_text())
nseeds = len(result['seeds'])
pair_count = result['pairs_per_mouse']
LABELS = dict(baseline='Original one-query transformer', dynamic='Four dynamic queries',
              static='Four static queries', prior_mlp='Archived shared temporal MLP')
CONTRASTS = dict(dynamic_vs_baseline='Four dynamic queries vs baseline',
                 dynamic_vs_static='Four dynamic queries vs four static queries',
                 static_vs_baseline='Four static queries vs baseline',
                 dynamic_vs_prior_mlp='Four dynamic queries vs archived MLP',
                 static_vs_prior_mlp='Four static queries vs archived MLP')


def pct(value):
    return f'{100 * value:+.1f}%'


def passed(value):
    return 'PASS' if value else 'FAIL'


def table(headers, records):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, r)) + ' |' for r in records])


primary = ['dynamic_vs_baseline', 'dynamic_vs_static', 'static_vs_baseline']
lines = [
    '# Multiquery readout on the shared transformer — 2026-10-05',
    f"Dynamic-query full practical gate: **{passed(gates['dynamic_full'])}**. Static-query secondary gate: **{passed(gates['static_secondary'])}**. These are descriptive development-cohort results, not independent statistical significance.",
    '## Question',
    'Does a single 16-value query summary restrict useful neuron-specific information? Replace it with four learned summaries (64 values), preserve the temporal transformer and population mean/std histories, and compare against an equally sized static-pooling control. This is a hypothesis about the readout recipe; additional head capacity and changed normalization are bundled with query count.',
    'The idea draws on learned pooling seed vectors in [Set Transformer, section 3.2](https://proceedings.mlr.press/v97/lee19d/lee19d.pdf). We adapt this mechanism to a scalar behavioral decoder, concatenate summaries and use an MLP head; we do not implement the full paper architecture or claim a novel invention. The paper does not establish that this will help these recordings.',
    '## Architecture and matching',
    'Both models retain 128 neurons × 32 activity bins, eight four-bin patches, width 16 and causal temporal attention within each neuron. Each of four learned queries independently reads all 1,024 neuron-time tokens through shared two-head projections. Concatenate four resulting 16-value representations and the original 64 population mean/std features; LayerNorm and a 128→64→1 GELU head predict speed. There is no extra population-attention block or query-to-query attention.',
    'Dynamic pooling computes keys from activity-dependent temporal states. Static pooling computes keys from learned neuron-ID/time/session embeddings; its values still carry activity. Both retain temporal attention, so the static arm is not an all-MLP decoder. Both have 21,553 parameters and identical initial tensors, versus 18,337 in the archived one-query baseline. Common nonhead tensors and the first query exactly match baseline initialization. The larger head is initialized from a fixed seed; all models start at normalized zero speed. Original dropout probabilities match, but changed query count changes draws relative to baseline. Equal updates do not equalize compute.',
    '## Fixed staged budget',
    'Stage 1 trains both variants with seeds 10, 11 and 12: six fits. Every fit gets 24 epochs, 5,688 AdamW updates and 179,712 presentations, using the original shared data, batches, loss weighting, optimizer and training-only preprocessing. Original baseline fits are reused. Select one joint checkpoint from epochs 0–24 by the original validation rule with fixed historical ridge denominators.',
    'Each variant must improve earlier-validation mean relative MSE by at least 5%, win three of four mice and eight of twelve single paired-seed comparisons, and avoid more than 10% harm to any mouse. If either passes, fit both variants on seeds 13–15, for at most twelve fits. Otherwise stop training. This screen reuses checkpoint-selection data and is optimistic, not a separate untouched validation set.',
    table(['Earlier screen vs baseline', 'Mean MSE gain', 'Mouse wins', 'Seed wins', 'Decision'],
          [[LABELS[k], pct(v['mean_mse_gain']), f"{v['mouse_wins']}/4", f"{v['seed_wins']}/12", passed(v['passed'])] for k, v in screen['summaries'].items()]),
    f"Extra-seed replication triggered: **{'yes' if screen['promote'] else 'no'}**. Completed **{audit['new_fits']} new fits**, seeds {result['seeds']}. No old fit repeated. All checkpoint choices and the screen decision were locked before current later inference.",
    '## Later-period comparisons',
    f"For each model, bound outputs at physical zero before averaging each distinct pair of seeds. Average pair errors within a mouse, then relative error changes equally over four mice. There are {pair_count} pairs per mouse in this study; comparisons use the same seed subset. These are average two-model-ensemble errors, not a selected pair or an ensemble of every seed.",
    f"Dynamic adoption requires its own earlier screen to pass, then at least 5% mean MSE improvement, three mouse wins and {int(np.ceil(2 * 4 * nseeds / 3))}/{4 * nseeds} single paired-seed wins against BOTH baseline and static control, plus no mouse more than 10% worse than baseline. Static adoption requires its earlier screen and the same later baseline comparison/harm guard. Failed screening cannot be rescued by later outcomes. A study stopped at screening still reports its later diagnostic transparently.",
    table(['Comparison', 'Mean MSE gain', 'Mouse wins', 'Paired-seed wins', 'Mean MAE gain', 'Later contrast'],
          [[CONTRASTS[k], pct(summaries[k]['mean_mse_gain']), f"{summaries[k]['mouse_wins']}/4", f"{summaries[k]['seed_wins']}/{4*nseeds}", pct(summaries[k]['mean_mae_gain']), passed(gates[k])] for k in primary]),
    table(['Requirement', 'Result'], [[k, passed(v)] for k, v in gates.items() if k not in primary]),
    '## Individual mice',
    table(['Mouse', 'N', 'Baseline MSE', 'Dynamic MSE', 'Static MSE', 'Dynamic gain vs baseline', 'Static gain vs baseline'],
          [[r['mouse'], r['n']] + [f"{r['scores'][g]['mse']:.6f}" for g in ['baseline', 'dynamic', 'static']] + [pct(r['contrasts'][k]['mse_gain']) for k in ['dynamic_vs_baseline', 'static_vs_baseline']] for r in rows]),
    table(['Mouse', 'Baseline MAE', 'Dynamic MAE', 'Static MAE', 'Baseline R²', 'Dynamic R²', 'Static R²'],
          [[r['mouse']] + [f"{r['scores'][g]['mae']:.6f}" for g in ['baseline', 'dynamic', 'static']] + [f"{r['scores'][g]['r2']:.3f}" for g in ['baseline', 'dynamic', 'static']] for r in rows]),
    'MSE/MAE use training-standardized speed. R² uses the later target variance descriptively; the later target mean is not an available training/deployment baseline.',
    '## Seed and mouse sensitivity',
    table(['Comparison', 'Single-model mean MSE gain', 'Pair wins', 'Leave-one-mouse-out pair gains'],
          [[CONTRASTS[k], pct(summaries[k]['single_mean_mse_gain']), f"{summaries[k]['pair_wins']}/{4*pair_count}", ', '.join(pct(x) for x in summaries[k]['leave_one_mouse_out'])] for k in primary]),
    'Leave-one-out entries omit MP030, MP032, MP033 and MP034, respectively. They do not authorize removing an unfavorable mouse. Seeds, overlapping pairs and overlapping time windows are not independent animals.',
    '## Context and learning',
    table(['Comparison', 'Mean MSE gain', 'Mouse wins', 'Mean MAE gain'],
          [[CONTRASTS[k], pct(summaries[k]['mean_mse_gain']), f"{summaries[k]['mouse_wins']}/4", pct(summaries[k]['mean_mae_gain'])] for k in ['dynamic_vs_prior_mlp', 'static_vs_prior_mlp']]),
    'The archived shared MLP differs in temporal processing, query routing and capacity; it is a contextual baseline. The four-static-query arm is the direct control for activity-dependent readout at the new capacity.',
    table(['Mouse', 'Training-mean MSE', 'Training-median MSE', 'Dynamic MSE', 'Static MSE'],
          [[r['mouse'], f"{r['references']['training_mean']['mse']:.6f}", f"{r['references']['training_median']['mse']:.6f}", f"{r['scores']['dynamic']['mse']:.6f}", f"{r['scores']['static']['mse']:.6f}"] for r in rows]),
    table(['Model', 'Single-model wins over initial training-mean prediction'], [[LABELS[k], f'{v}/{4*nseeds}'] for k,v in result['single_wins_vs_initial'].items()]),
    '## Training and verification',
    table(['Variant', 'Seed', 'Selected epoch', 'Updates', 'Presentations', 'Query gradient max', 'Fit seconds'],
          [[r['variant'], r['seed'], r['selected_epoch'], r['updates'], r['examples'], f"{r['query_gradient_max']:.4f}", f"{r['elapsed_seconds']:.1f}"] for r in locked['records']]),
    'Fit times include contention among workers; these are not controlled inference or efficiency benchmarks.',
    f"Preflight passed 36 exact trained-parent comparisons in one-query mode, equal new-variant tensors, multiquery shapes and weight sums, static/dynamic routing behavior, distinct initial queries, gradients to every query and the temporal path, input/example isolation and exact reloads. Training passed matched orders, actual optimizer counts and selected reloads. Locking verified {audit['selection_scores_checked']} validation scores. Later scoring produced {audit['new_later_predictions']:,} new predictions, {audit['baseline_firstbatch_exact']} exact baseline first-batch comparisons and {audit['independent_scalar_scores']} independently calculated scalar metrics. Frozen numerical sources, inputs, checkpoints and application files remain unchanged.",
    'Four historically reused Stringer mice remain the evidence limit. A successful engineering gate would support this fixed development recipe, not independent significance, causal connectivity, general superiority over all MLPs or realistic neural generation. No query-count, width, placement, loss, optimization or seed grid is appended after outcomes.',
    'Artifacts: [assessment](ASSESSMENT.md), [protocol](protocol.json), [earlier screen](screen.json), [selection lock](selection_lock.json), [results](results.json), [audit](audit.json), [review](review.json), [figure](readout.png), [PDF](readout.pdf).'
]
if result['extra_seed_replication'] is not None:
    replication = result['extra_seed_replication']
    lines.insert(-1, '## Seeds 13–15 separately\n\n' + table(['Comparison', 'Mean MSE gain', 'Mouse wins', 'Seed wins'],
        [[CONTRASTS[k], pct(v['mean_mse_gain']), f"{v['mouse_wins']}/4", f"{v['seed_wins']}/12"] for k,v in replication['summaries'].items()]) +
        '\n\nNew training seeds on the same reused animals; this is not independent animal confirmation.')
context = json.loads((ROOT / 'readout_context.json').read_text())
lines.insert(-1, '## Supplementary one-query static comparison\n\n'
    'Declared after the positive static validation screen, before current later scoring. This uses only matching seeds 10–12 and their three pairs; it does not alter the frozen primary criteria. All predictions are reused. Comparing static four-query with static one-query narrows the routing confound, but the wider head and normalization remain bundled with query count.\n\n' +
    table(['Comparison vs one-query static', 'Mean MSE gain', 'Mouse wins', 'Seed wins', 'Mean MAE gain'],
        [[k, pct(v['mean_mse_gain']), f"{v['mouse_wins']}/4", f"{v['seed_wins']}/12", pct(v['mean_mae_gain'])] for k,v in context['summaries'].items()]))
diversity = json.loads((ROOT / 'readout_diagnostic.json').read_text())
records = []
for group in ['baseline', 'dynamic', 'static']:
    selected = [r for r in diversity['rows'] if r['variant'] == group]
    variations = [r['mean_pair_weight_total_variation'] for r in selected if r['mean_pair_weight_total_variation'] is not None]
    cosines = [v for r in selected for v in r['centered_query_feature_cosines'] if v is not None]
    ranks = [r['centered_neural_feature_effective_rank'] for r in selected]
    records.append([LABELS[group], f'{np.mean(variations):.4f}' if variations else '—',
                    f'{np.median(cosines):.4f}' if cosines else '—', f'{np.mean(ranks):.2f}'])
lines.insert(-1, '## Earlier-data query diversity\n\n'
    'Fixed while first-stage training was incomplete: first 64 selection windows per mouse, all selected active seeds. Total variation compares query-pair pooling distributions (0 identical, 1 disjoint); feature cosine compares their vectors after centering each feature over examples. Effective rank is the participation ratio of centered neural-only head features. These are descriptive finite-sample redundancy measures; even highly similar queries may preserve useful small differences. They neither select a model nor prove why error changed.\n\n' +
    table(['Model', 'Mean pair weight total variation', 'Median centered feature cosine', 'Mean feature effective rank'], records))
(ROOT/'report.md').write_text('\n\n'.join(lines)+'\n')
fig, axes = plt.subplots(1,2,figsize=(12,4.8))
axes[0].bar(range(3), [100*summaries[k]['mean_mse_gain'] for k in primary],color=['#286a9a','#5c8dad','#be6c28'])
axes[0].set(xticks=range(3),xticklabels=['Dynamic\nvs baseline','Dynamic\nvs static','Static\nvs baseline'],title='Later mean relative error change')
for offset,key,label,color in [(-.18,'dynamic_vs_baseline','Four dynamic queries','#286a9a'),(.18,'static_vs_baseline','Four static queries','#be6c28')]:
    axes[1].bar(np.arange(4)+offset,[100*r['contrasts'][key]['mse_gain'] for r in rows],width=.36,label=label,color=color)
axes[1].set(xticks=range(4),xticklabels=[r['mouse'] for r in rows],title='Effect versus one-query baseline')
axes[1].legend(frameon=False)
for ax in axes:
    ax.axhline(0,color='black',linewidth=.8)
    ax.set_ylabel('MSE reduction (%)')
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('Four pooling queries on the shared temporal transformer',x=.06,ha='left')
fig.text(.06,.02,f'Four reused mice, {nseeds} matched seeds, all {pair_count} two-model pairs per mouse.\n'
         'Positive values favor the named candidate; failed earlier screening cannot be rescued by later outcomes.',fontsize=9)
fig.tight_layout(rect=[0,.10,1,.93])
for name in ['readout.png','readout.pdf']:
    fig.savefig(ROOT/name,dpi=160,bbox_inches='tight')
plt.close(fig)
print('Wrote report.md, readout.png and readout.pdf')
