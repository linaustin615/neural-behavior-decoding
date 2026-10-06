"""Independently review the frozen population-context query comparison."""
import math
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/neuron_finetuning_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import run

ROOT = run.ROOT
PRIMARY = ['context_mlp', 'large_mlp', 'population_attention', 'ridge512']
NAMES = {
    'context_mlp': 'Matched context MLP',
    'large_mlp': '512-neuron local MLP',
    'population_attention': 'Population transformer parent',
    'ridge512': '512-neuron ridge',
    'native_attention': 'Original 128-neuron transformer (secondary)',
}
stage = 2 if (ROOT / 'stage2_results.json').exists() else 1
subsets = {}
audits = []
locks = []
checks = 0
for s in range(1, stage + 1):
    locks.append(run.verify_lock(s))
    subsets.update(run.read(ROOT / f'stage{s}_results.json')['subsets'])
    audits.append(run.read(ROOT / f'stage{s}_audit.json'))
    for name, value in run.read(ROOT / f'stage{s}_evaluation.json')['prediction_hashes'].items():
        assert run.digest(ROOT / name) == value

for name, part in subsets.items():
    seeds, rows = part['seeds'], part['rows']
    assert seeds == {'stage1': [10, 11, 12], 'additional': [13, 14, 15], 'all': list(range(10, 16))}[name]
    assert [r['mouse'] for r in rows] == run.MICE
    primary = []
    for contrast, value in part['summaries'].items():
        control = contrast[len('context_attention_vs_'):]
        effects = [1 - r['scores']['context_attention']['mse'] / r['scores'][control]['mse'] for r in rows]
        mae = [1 - r['scores']['context_attention']['mae'] / r['scores'][control]['mae'] for r in rows]
        wins = sum(sum(a < b for a, b in zip(r['scores']['context_attention']['single_mse'], r['scores'][control]['single_mse'])) for r in rows)
        np.testing.assert_allclose([value['mean_mse_gain'], value['mean_mae_gain']], [sum(effects) / 4, sum(mae) / 4], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(value['mouse_mse_gains'], effects, rtol=1e-12, atol=1e-14)
        assert value['mouse_wins'] == sum(x > 0 for x in effects) and value['seed_wins'] == wins
        passed = sum(effects) / 4 >= .05 and sum(x > 0 for x in effects) >= 3 and wins >= math.ceil(8 * len(seeds) / 3) and min(effects) >= -.1 and sum(mae) / 4 >= 0
        assert value['full_pass'] == passed
        if control in PRIMARY:
            primary.append(passed)
        checks += 5
        for i, seed in enumerate(seeds):
            effects_i = [1 - r['scores']['context_attention']['single_mse'][i] / r['scores'][control]['single_mse'][i] for r in rows]
            record = value['individual_seeds'][i]
            assert record['seed'] == seed and record['mouse_wins'] == sum(x > 0 for x in effects_i)
            np.testing.assert_allclose(record['mean_mse_gain'], sum(effects_i) / 4, rtol=1e-12, atol=1e-14)
            checks += 2
    assert len(primary) == 4
    initial = sum(sum(x < r['initial_mse'] for x in r['scores']['context_attention']['single_mse']) for r in rows)
    assert part['initial_wins'] == initial
    assert part['primary_pass'] == (all(primary) and initial >= math.ceil(8 * len(seeds) / 3))
    checks += 2

assert subsets['stage1']['primary_pass'] == (stage == 2)
full = stage == 2 and all(value['primary_pass'] for value in subsets.values())
assert run.read(ROOT / f'stage{stage}_results.json')['full_replication_passed'] == full
fits = len(list(ROOT.glob('*_s*/result.json')))
assert fits == (6 if stage == 1 else 18)
for name, value in run.read(run.LARGE / 'prepared.json')['hashes'].items():
    assert run.digest(run.LARGE / name) == value

changes = []
for path in sorted(ROOT.glob('*_s*/result.json')):
    record = run.read(path)
    folder, family = path.parent, record['family']
    states = {kind: torch.load(folder / f'{kind}.pt', weights_only=True) for kind in ['initial', 'final', 'selected']}
    keys = ['head.3.weight']
    if family in ['attention', 'mlp']:
        keys += ['context_projection.weight', 'context.readin', 'context.temporal.mix.in_proj_weight' if family == 'attention' else 'context.temporal.weight']
    elif family == 'old_mlp':
        keys += ['temporal.weight']
    elif family == 'pop_parent':
        keys += ['readin', 'temporal.mix.in_proj_weight']
    else:
        raise ValueError(family)
    for key in keys:
        change = float((states['final'][key] - states['initial'][key]).norm())
        selected_change = float((states['selected'][key] - states['initial'][key]).norm())
        assert change > 0 and np.isfinite(change)
        assert np.isfinite(selected_change) and (selected_change > 0) == (record['selected_epoch'] > 0)
        changes.append(dict(family=family, seed=record['seed'], parameter=key, final_change=change, selected_change=selected_change))
        checks += 2

review = dict(passed=True, aggregate_and_parameter_checks=checks, new_neural_fits=fits, full_replication_passed=full,
    selection_scores_checked=sum(value['selection_scores_checked'] for value in locks),
    scalar_errors_checked=sum(value['scalar_errors_checked'] for value in audits),
    new_predictions=sum(value['new_predictions'] for value in audits),
    frozen_source_hashes=len(run.verify()['hashes']), selected_artifact_hashes=sum(len(value['hashes']) for value in locks),
    main_application_unchanged=True, parameter_changes=changes)
run.write(ROOT / 'review.json', review)


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


pct = lambda x: f'{100*x:+.2f}%'
verdict = lambda x: 'PASS' if x else 'FAIL'
lines = ['# Population context for neuron readout',
    f"Full practical replication gate: **{verdict(full)}**. {fits} new fits completed. " + ('Stage one failed; conditional replication was not launched.' if stage == 1 else 'Stage one passed and triggered the twelve predefined replication fits.'),
    'The hybrid retains the complete stronger 512-neuron local MLP decoder. A second encoder learns 16 population signals per time bin, forms eight four-bin temporal patches, and applies causal temporal attention. Its final state changes the behavior query through a zero-initialized 16-by-16 projection. The query then reads local activity values using static learned neuron/time/session keys. The population context enters before query projection and also reaches the query residual; this study does not isolate those two paths.',
    'The matched context-MLP control has the same input-dependent query mechanism, but replaces population temporal self-attention with a causal MLP mixer. Both models have activity-dependent gating; the comparison isolates population temporal attention, not all possible attention or gating. They have 79,383 and 79,373 parameters, with 49 common initial tensors exactly matched. Both are trained from scratch. No parent checkpoint initializes a scored fit; trained parent weights were used only in discarded preflight equivalence checks.',
    'Other primary controls are the stronger 512-neuron local MLP, the standalone population transformer and 512-neuron ridge. The original 128-neuron transformer is secondary. The hybrid has greater capacity and compute than either neural parent; the similarly sized context-MLP control addresses that difference. The design is an integration hypothesis, not a demonstrated novel architecture or causal neuron-interaction model.',
    'Each new fit uses the original 24 epochs, 5,688 AdamW updates, 179,712 example presentations, earlier checkpoint rule, training batches, optimizer, scheduler and loss weighting. The fixed first stage fits two hybrid families on seeds 10–12. Only a full pass permits both hybrids and both neural parents on seeds 13–15: at most 18 fits. Completed parent predictions are reused where available; no old fit is repeated.',
    'The hybrid transformer must beat every primary control by at least 5% equal-mouse mean relative pair-MSE, win at least three of four mouse means and two-thirds of single-seed comparisons, avoid more than 10% MSE harm in any mouse, and have nonnegative mean relative MAE gain. Two-thirds of single models must beat their initial training-mean predictor. First-stage, additional-seed and all-six-seed gates must pass separately; a pooled result cannot rescue failure.',
    'Individual normalized predictions are bounded at physical zero before averaging each distinct two-seed pair. Pair errors are averaged within mouse, then relative effects are averaged equally over four mice. Single models determine seed consistency. No favorable pair, seed or mouse is chosen. Repeating a deterministic ridge prediction does not create independent fits.',
]
for name, part in subsets.items():
    lines += ['## ' + name,
        f"Subset gate: **{verdict(part['primary_pass'])}**. Initial-predictor wins: {part['initial_wins']}/{4*len(part['seeds'])}.",
        table(['Hybrid transformer versus', 'Mean MSE gain', 'Mouse wins', 'Individual wins', 'Mean MAE gain', 'Contrast'],
            [[NAMES[key[len('context_attention_vs_'):]], pct(value['mean_mse_gain']), f"{value['mouse_wins']}/4", f"{value['seed_wins']}/{4*len(part['seeds'])}", pct(value['mean_mae_gain']), verdict(value['full_pass'])] for key, value in part['summaries'].items()]),
        table(['Mouse', 'Hybrid T MSE', 'Context MLP MSE', 'Local MLP MSE', 'Population T MSE', 'Ridge MSE', 'Hybrid T R²'],
            [[row['mouse']] + [f"{row['scores'][key]['mse']:.6f}" for key in ['context_attention'] + PRIMARY] + [f"{row['scores']['context_attention']['r2']:.3f}"] for row in part['rows']]),
        table(['Control', 'Mouse MSE gains (MP030/032/033/034)', 'Leave-one-mouse-out means'],
            [[NAMES[key[len('context_attention_vs_'):]], ', '.join(pct(x) for x in value['mouse_mse_gains']), ', '.join(pct(x) for x in value['leave_one_mouse_out'])] for key, value in part['summaries'].items()]),
        table(['Control', 'Individual seed', 'Mean MSE gain', 'Mouse wins'],
            [[NAMES[key[len('context_attention_vs_'):]], value['seed'], pct(value['mean_mse_gain']), f"{value['mouse_wins']}/4"] for key, summary in part['summaries'].items() for value in summary['individual_seeds']]),
    ]
lines += ['## Verification and limitations',
    f"Preflight passed six exact trained-parent checks with zero or disabled context, input-dependent readout checks, legacy-predictor compatibility, branch learning, finite context/temporal gradients, common initialization, reloads and input preservation. Selection locks checked {review['selection_scores_checked']} earlier scores. Evaluation produced {review['new_predictions']} new predictions; {review['scalar_errors_checked']} scalar errors were independently checked. Review checked {checks} aggregate/gate/parameter conditions, {review['frozen_source_hashes']} frozen source hashes, {review['selected_artifact_hashes']} selected artifacts and eight prepared arrays. Selected/final parameter changes are recorded, including the initially zero context projection. Parameter movement is a computational check, not proof of useful generalization.",
    'All fits use the same four historically searched mice, recording-specific cell identities and previously reused chronological evaluation periods. New seeds and model pairs are not independent animals. Any practical pass here is a development result, not independent statistical significance, unseen-mouse transfer, causal interaction evidence or coordinate utility. No additional query scales, projection ranks, layers, seeds or relaxed gates are appended to this study.',
    'Main application files and previous experiments remain unchanged. Artifacts: [assessment](ASSESSMENT.md), [protocol](protocol.json), [review](review.json), [status](STATUS.json), [model](models.py), [figure](context_query.png).',
]
(ROOT / 'report.md').write_text('\n\n'.join(lines) + '\n')

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
keys = ['context_attention_vs_' + key for key in PRIMARY]
labels = ['Context MLP', 'Local MLP', 'Population T', 'Ridge512']
width = .7 / len(subsets)
for i, (name, part) in enumerate(subsets.items()):
    axes[0].bar(np.arange(4) + (i - (len(subsets) - 1) / 2) * width, [100 * part['summaries'][key]['mean_mse_gain'] for key in keys], width=width, label=name)
axes[0].set(xticks=range(4), xticklabels=labels, ylabel='MSE reduction (%)', title='Context transformer mean gains')
axes[0].legend(frameon=False)
part = subsets['additional'] if 'additional' in subsets else subsets['stage1']
for i, key in enumerate(keys):
    axes[1].bar(np.arange(4) + (i - 1.5) * .19, [100 * x for x in part['summaries'][key]['mouse_mse_gains']], width=.19, label=labels[i])
axes[1].set(xticks=range(4), xticklabels=run.MICE, ylabel='MSE reduction (%)', title='Latest tested seed group by mouse')
axes[1].legend(frameon=False, fontsize=8)
for ax in axes:
    ax.axhline(0, color='black', linewidth=.8)
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Population context guides the per-neuron MLP readout', x=.06, ha='left')
fig.text(.06, .02, 'Positive values favor the hybrid transformer. Four historically searched mice; no independent animal confirmation.\nAll four primary controls and every required seed subset must pass.', fontsize=9)
fig.tight_layout(rect=[0, .11, 1, .94])
for suffix in ['png', 'pdf']:
    fig.savefig(ROOT / f'context_query.{suffix}', dpi=160, bbox_inches='tight')
plt.close(fig)
print('Context-query review, report and figures saved', flush=True)
