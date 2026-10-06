from pathlib import Path
import json
import os
import re
import numpy as np

ROOT = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(ROOT / '.matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

r = json.loads((ROOT / 'results.json').read_text())
variants = r['protocol']['variants']
labels = {'baseline': 'Original: 8 summaries + IDs', 'no_id': 'Remove neuron IDs', 'latent32': 'Increase to 32 summaries', 'local': 'Add spatial neighbor layer'}
colors = {'real': '#187b5b', 'none': '#64748b', 'shuffled': '#c46d26'}
fig, ax = plt.subplots(figsize=(11, 5.4), constrained_layout=True)
x = np.arange(4)
for offset, condition in enumerate(['real', 'none', 'shuffled']):
    means = [r['groups'][v][condition]['test_mse'] for v in variants]
    sd = [r['groups'][v][condition]['test_sd'] for v in variants]
    bars = ax.bar(x + (offset - 1) * .24, means, .23, yerr=sd, capsize=3, label={'real': 'Correct coordinates', 'none': 'No coordinates', 'shuffled': 'Shuffled coordinates'}[condition], color=colors[condition])
    ax.bar_label(bars, labels=[f'{value:.3f}' for value in means], padding=4, fontsize=9)
ax.set_xticks(x, [labels[v].replace(': ', ':\n').replace(' spatial ', '\nspatial ') for v in variants], fontsize=9)
ax.set_ylabel('Test MSE (lower is better)')
ax.set_title('Architecture tests on the same visual-cortex recording')
ax.legend(frameon=False, loc='upper right', fontsize=9)
ax.spines[['top', 'right']].set_visible(False)
fig.text(.5, -.12, '12 matched repeats per condition: 6 overlapping neuron samples × 2 training seeds. Error bars: repeat SD.\nExploratory results on a repeatedly examined recording; these are not independent animals.', ha='center', fontsize=9)
fig.savefig(ROOT / 'comparison.png', dpi=180, bbox_inches='tight')
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), constrained_layout=True)
for ax, variant in zip(axes, variants[1:]):
    h = r['hypotheses'][variant]
    values = [h['interactions'][c] for c in ['none', 'shuffled']] + [h['absolute_real_improvement']]
    for index, value in enumerate(values):
        low, high = value['interval95']
        ax.plot([low, high], [index, index], color='#334155', linewidth=2)
        ax.scatter(value['mean'], index, color='#2563eb', zorder=3)
    ax.axvline(0, color='#94a3b8', linewidth=1)
    ax.set_yticks(range(3), ['Extra spatial benefit\nvs no coordinates', 'Extra spatial benefit\nvs shuffled coordinates', 'Absolute improvement\nwith correct coordinates'], fontsize=8)
    ax.set_ylim(2.6, -.6)
    ax.set_title(labels[variant], fontsize=10)
    ax.set_xlabel('MSE improvement; positive favors change', fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Does each change make correct coordinates more useful?', fontsize=13)
fig.text(.5, -.14, 'Bars: exploratory 95% intervals from paired pool/seed/time-block resampling.\nTwo training seeds and one recording limit uncertainty estimates; no correction for multiple comparisons.', ha='center', fontsize=9)
fig.savefig(ROOT / 'interactions.png', dpi=180, bbox_inches='tight')
plt.close(fig)

lines = ['# Tests of three architectural explanations for weak spatial benefit', '', 'This study completed 108 new fits and reused 36 verified baseline fits: 144 models, covering four architectures × three coordinate conditions × six neuron samples × two training seeds. Every planned fit is included. These are exploratory comparisons on one repeatedly examined Stringer MP019 visual-cortex recording, not independent biological replications.', '', '## Main results', '', '| Architecture | Correct coordinates | No coordinates | Shuffled coordinates | Correct-coordinate test R² |', '|---|---:|---:|---:|---:|']
for v in variants:
    g = r['groups'][v]
    lines.append(f"| {labels[v]} | {g['real']['test_mse']:.5f} | {g['none']['test_mse']:.5f} | {g['shuffled']['test_mse']:.5f} | {g['real']['test_r2']:.3f} |")
lines += ['', 'Values are mean normalized test MSE; lower is better. These are means of individual models, not ensembles. The original baseline here uses seeds10 and11 on all six pools; its mean differs from the earlier 18-repeat study, which also included seed12.', '', '## Does geometry help within each architecture?', '', '| Architecture | Control | Test advantage of real coordinates | Relative error reduction | Paired wins | Positive pool means | Exploratory 95% interval |', '|---|---|---:|---:|---:|---:|---|']
for v in variants:
    for c in ['none', 'shuffled']:
        a = r['contrasts'][v][c]
        lo, hi = a['interval95']
        lines.append(f"| {labels[v]} | {c} | {a['mean']:+.5f} | {a['relative_error_reduction_percent']:+.2f}% | {a['positive_pairs']}/12 | {a['positive_pool_means']}/6 | [{lo:+.5f}, {hi:+.5f}] |")
lines += ['', 'Positive advantages mean correct coordinates lower error. Shuffled coordinates remain fixed within a fit and can act as arbitrary identifiers. In the local architecture, shuffling also scrambles neighbor connections; absent coordinates instead use the mean of all other cells in the added branch, avoiding arbitrary tied-distance neighbors.', '', '## Architectural hypotheses', '', 'The interaction is the change in spatial advantage relative to the original architecture. Improving overall speed prediction alone does not establish a spatial improvement. A larger coordinate advantage obtained by damaging the controls also does not establish a useful model improvement.', '']
for v in variants[1:]:
    h = r['hypotheses'][v]
    lines += [f"### {labels[v]}", '', '| Quantity | Mean MSE advantage | Positive pool means | Exploratory 95% interval |', '|---|---:|---:|---|']
    for label, a in [('Extra spatial benefit versus no coordinates', h['interactions']['none']), ('Extra spatial benefit versus shuffled coordinates', h['interactions']['shuffled']), ('Absolute improvement over original with real coordinates', h['absolute_real_improvement'])]:
        lo, hi = a['interval95']
        lines.append(f"| {label} | {a['mean']:+.5f} | {a['positive_pool_means']}/6 | [{lo:+.5f}, {hi:+.5f}] |")
    lines += ['', f"Both mean spatial effects and architecture interactions positive: **{h['mean_spatial_and_interaction_positive']}**. Also improves absolute real-coordinate performance: **{h['practical_mean_support']}**. Both spatial effects and interactions have positive interval lower bounds and positive means in every pool: **{h['robust_exploratory_support']}**.", '']
lines += ['## Validation and raw-error checks', '', '| Architecture | Real validation MSE | None validation MSE | Shuffled validation MSE | Raw test advantage vs none | Raw test advantage vs shuffled |', '|---|---:|---:|---:|---:|---:|']
for v in variants:
    g = r['groups'][v]
    lines.append(f"| {labels[v]} | {g['real']['validation_mse']:.5f} | {g['none']['validation_mse']:.5f} | {g['shuffled']['validation_mse']:.5f} | {r['contrasts'][v]['none']['test_raw_advantage']:+.5f} | {r['contrasts'][v]['shuffled']['test_raw_advantage']:+.5f} |")
lines += ['', '## Per-sample spatial effects', '', 'Each value averages seeds10 and11; positive means real coordinates improve test MSE.', '', '| Architecture | Sample | None minus real | Shuffled minus real |', '|---|---:|---:|---:|']
for v in variants:
    for pool in r['protocol']['pool_seeds']:
        lines.append(f"| {labels[v]} | {pool} | {r['contrasts'][v]['none']['pool_means'][str(pool)]:+.5f} | {r['contrasts'][v]['shuffled']['pool_means'][str(pool)]:+.5f} |")
lines += ['', '## What was changed', '', '- Baseline: nonlinear activity and position embeddings plus neuron-ID embedding,8 learned summary queries,width32,one transformer layer and mean readout.', '- No IDs: omit and freeze the ID embedding. The model retains81,121 stored parameters but only15,585 trainable parameters; the baseline has81,121 trainable parameters. The comparison tests removing cell-specific memory, not a parameter-count-matched alternative.', '-32 summaries: retain all shared initial weights and the original8 queries; append24 independently seeded queries. Width remains32. Total81,889 trainable parameters.', '- Local branch: for each cell, find8 nearest neighbors in separately standardized xyz, exclude itself, and average their8-bin activity histories using normalized Gaussian distance weights. Feed its own history and neighbor-minus-own history through a16→32→32 GELU MLP and add0.1 times its output to the token before global read-in. Total82,721 trainable parameters. Standardized xyz is a modeling choice, not calibrated physical distance; the layer does not reproduce measured connectivity or molecular cell types.', '', 'Every change is tested separately; combinations are not evaluated. This budget tests these implementations, not every possible spatial architecture.', '', '## Training and evaluation', '', 'All conditions use2,048 cells, the same6 pools101/202/303/404/505/606 and seeds10/11, the same8-bin history and chronological splits, and identical batch ordering within a matched repeat. Raw splits are[0,4160),[4260,5564),[5664,7018); targets begin31 bins into each split, yielding4,129 training,1,273 validation and1,323 test examples. Per-cell activity and speed scaling use training data; coordinate scaling is fixed across all eligible cells.', '', 'Training uses AdamW(lr0.001,weight decay0.01),batch32,gradient clipping1,and24-epoch cosine scheduling to0.0001. Early stopping requires at least12 epochs and7 stale validations. Checkpoints minimize validation MSE including the untrained model; test error never chooses a checkpoint. Training uses raw standardized MSE; evaluation floors negative physical-speed predictions atzero for every condition. Raw errors are also reported. Speed units are the dataset’s units, not verified cm/s.', '', 'Uncertainty uses2,000 crossed resamples of six pool blocks and two shared seed blocks, plus paired circular100-bin time blocks. Intervals describe this finite experiment conditionally; overlapping pools,two seeds,one animal,repeated test inspection and multiple comparisons limit inference. No pristine test-set or confirmatory significance claim is made.', '', '## Integrity checks', '', f"All {r['audit']['trained_beat_own_untrained_test']}/144 selected models beat their own untrained test error; {r['audit']['best_at_last_allowed_epoch']}/144 selected epoch24. Shared initial weights, coordinate conditions, label/window alignment inherited from the frozen runner, and36 reused fits were checked. Model outputs pass batch-size and joint-neuron-permutation checks; gradients are finite for all trainable parameters. Neighbor indices, weights and aggregation were checked independently with NumPy. Saved predictions reproduce every metric, and each selected checkpoint minimizes its logged validation loss.", '', 'The identity-absorption check folds the trained position contribution into fixed neuron-ID embeddings and removes coordinate input. Its maximum prediction discrepancy is below0.000001 on complete validation and test segments. This verifies representational redundancy for known cells; it does not show training automatically learns the same weights or that geometry cannot improve data efficiency.', '', 'completion_audit.json verifies all checkpoint tensors and neuron/coordinate assignments and reloads the full validation/test predictions for the worst coordinate-versus-none pair within each new architecture,including all three conditions. Local-branch parameters are checked for actual changes during training. Application source and dataset hashes are preserved.', '', '## Artifacts', '', '- protocol.json and frozen_implementation.json: pre-run design and hashes', '- results.json and per_run.csv: all results, per-pool contrasts and interactions', '- comparison.png and interactions.png: performance and uncertainty figures', '- runs/: every model, training curve and saved predictions', '- architecture_model.py and architecture_search.py: isolated experimental models and runner', '- analyze_architecture.py,report_architecture.py,check_architecture.py,verify_completion.py: analysis and integrity checks', '', 'The experiment uses the project’s data/stringer_spontaneous.npy. Run architecture_search.py with a shard JSON to execute its frozen schedule; existing completed records are reused. Analyze and render after all runs finish. No production learning code was edited.', '']
text = '\n'.join(lines)
text += f"\n## Simple fixed baselines\n\nAlways predicting the average training speed gives test MSE {r['simple_baselines']['constant_training_mean']['test']['mse']:.5f}; always predicting zero speed gives {r['simple_baselines']['constant_zero_speed']['test']['mse']:.5f}. {r['audit']['beat_constant_training_mean_test']}/144 selected models beat the training-mean baseline; {r['audit']['beat_constant_zero_speed_test']}/144 beat zero speed. These supplemental checks do not change the frozen architectural hypotheses.\n"
text = re.sub(r'(?<=[a-z])(?=\d)', ' ', text)
text = re.sub(r',(?=[A-Za-z])', ', ', text)
text = text.replace('-32 summaries:', '- 32 summaries:')
if (ROOT / 'interpretation.md').exists():
    text = text.replace('## Main results', (ROOT / 'interpretation.md').read_text().strip() + '\n\n## Main results', 1)
if (ROOT / 'summary_diversity.json').exists():
    diversity = json.loads((ROOT / 'summary_diversity.json').read_text())
    text += '\n## Summary-attention inspection\n\n' + diversity['scope'] + '\n\n'
    for variant, values in diversity['means'].items():
        text += f"{labels[variant]}: mean pairwise attention cosine {values['mean_attention_cosine']:.4f}; mean attention participation rank {values['attention_participation_rank']:.3f}.\n\n"
(ROOT / 'report.md').write_text(text)
print('Wrote report.md, comparison.png, interactions.png')
