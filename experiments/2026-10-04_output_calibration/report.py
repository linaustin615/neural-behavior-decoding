"""Report the locked calibration test and label post-hoc distribution context."""
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent
protocol = json.loads((ROOT / 'protocol.json').read_text())
result = json.loads((ROOT / 'results.json').read_text())
locked = json.loads((ROOT / 'calibration_lock.json').read_text())
audit = json.loads((ROOT / 'audit.json').read_text())
assert audit['passed']


def pct(value):
    return 'undefined' if value is None else f'{100 * value:+.1f}%'


def number(value):
    return 'undefined' if value is None else f'{value:.4f}'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows])


context = []
for mouse in protocol['mice']:
    dev = np.load(EXP / '2026-10-03_dynamics_baseline' / mouse / 'selection_y.npy')
    with np.load(EXP / '2026-10-04_ensemble_seed_replication' / mouse / 'later_predictions.npz') as z:
        later = z['target'].copy()
    cuts = protocol['thresholds'][mouse]
    for label, y in [('calibration', dev[slice(*protocol['bounds'][mouse]['calibration'])]),
                     ('development_check', dev[slice(*protocol['bounds'][mouse]['development_check'])]), ('later', later)]:
        context.append(dict(mouse=mouse, stage=label, n=len(y), mean=float(y.mean()), std=float(y.std()),
                            low_speed_fraction=float(np.mean(y <= cuts['median'])), high_speed_fraction=float(np.mean(y > cuts['q90']))))
description = dict(status='post-hoc description after calibration outcome; no fitting or rule changes', rows=context)
path = ROOT / 'distribution_context.json'
if path.exists():
    assert json.loads(path.read_text()) == description
else:
    path.write_text(json.dumps(description, indent=2) + '\n')

lines = [
    '# Output calibration and error direction — 2026-10-04',
    'The fixed affine correction fails. It increases later transformer mean relative MSE by113.2% and MAE by85.5%, worsening all four mouse-average MSEs. The identical MLP correction also fails. Keep the original models; do not apply this correction or tune it using these outcomes.',
    '## Question and locked procedure',
    'Test whether a stable scale/offset error explains the archived decoder’s weaknesses. Preserve all original selected checkpoints, six seeds10–15, both model families and all15 unordered two-model pairs. Predictions first receive the original nonnegative-speed floor, then pair averaging. Fit one affine map per mouse/family/pair: corrected prediction = max(physical-zero floor, slope × prediction + offset). Minimize ordinary unbounded squared error with nonnegative slope; for effectively constant predictors use slope1 and a mean-residual offset. There is no regularization, slope cap, correction-strength search or per-case choice between corrected and original.',
    'Fit only the first half of each development interval:337/277/196/196 windows for MP030/032/033/034. Leave32 windows before the later-development check:306/246/165/165 windows. Lock all120 coefficient pairs before computing either development-check or later scores. The final later intervals contain726/605/443/444 windows. No coefficients are refit after either outcome. This uses zero network fits and zero model inferences.',
    'This is exploratory historical replay. The base checkpoint was originally selected using the full development interval, including its remainder. Therefore the development check is disjoint from calibrator fitting but is not a pristine model holdout. Later recordings and prior architecture choices were also historically inspected. Seed pairs and adjacent windows are dependent; four mice remain the replication units.',
    'Calibration practical gate, fixed before fitting: at least5% mean relative MSE improvement versus the unchanged same-family pair, at least3/4 mouse wins, at least40/60 matched mouse-pair wins, no mouse over10% worse, and no mean relative MAE harm. The transformer gate is primary; MLP calibration uses the same rule as a control. No p-values or significance claim. Comparisons between calibrated families do not create a new attention-utility gate.',
    '## Results',
    table(['Stage', 'Correction vs original', 'Mean MSE gain', 'Mouse wins', 'Pair wins /60', 'Mean MAE gain', 'Calibration gate'],
          [[stage, family, pct(result[stage]['comparisons'][family + '_calibrated_vs_' + family]['mean_relative_mse_gain']),
            result[stage]['comparisons'][family + '_calibrated_vs_' + family]['mouse_wins'],
            result[stage]['comparisons'][family + '_calibrated_vs_' + family]['matched_pair_wins'],
            pct(result[stage]['comparisons'][family + '_calibrated_vs_' + family]['mean_relative_mae_gain']),
            'PASS' if result[stage]['comparisons'][family + '_calibrated_vs_' + family]['practical_calibration_gate'] else 'FAIL']
           for stage in ['development_check', 'later'] for family in ['attention', 'mlp']]),
    table(['Mouse', 'Original transformer MSE', 'Corrected transformer MSE', 'Transformer correction gain', 'MLP correction gain'],
          [[r['mouse'], number(r['scores']['attention']['mse']), number(r['scores']['attention_calibrated']['mse']),
            pct(result['later']['comparisons']['attention_calibrated_vs_attention']['mouse_gains'][i]),
            pct(result['later']['comparisons']['mlp_calibrated_vs_mlp']['mouse_gains'][i])]
           for i, r in enumerate(result['later']['rows'])]),
    'Transformer correction already worsens the subsequent development check by8.9% mean relative MSE (1/4 mouse wins) and21.1% MAE. On later data it wins only5/60 dependent mouse-pair comparisons and0/4 mouse means; even omitting any mouse leaves a negative mean gain. The later calibrated-transformer versus calibrated-MLP comparison has2.9% lower mean relative MSE but2.8% higher MAE. Both calibrated families are worse than their respective originals; this comparison does not rescue the correction.',
    '## What the coefficients did',
    table(['Mouse', 'Family', 'Mean slope', 'Slope range across pairs', 'Mean offset', 'Fit MSE before', 'Fit MSE after, before floor'],
          [[r['mouse'], r['family'], number(np.mean([c['slope'] for c in r['coefficients']])),
            number(min(c['slope'] for c in r['coefficients'])) + ' to ' + number(max(c['slope'] for c in r['coefficients'])),
            number(np.mean([c['intercept'] for c in r['coefficients']])),
            number(np.mean([c['calibration_mse_before'] for c in r['coefficients']])),
            number(np.mean([c['calibration_mse_after_unbounded'] for c in r['coefficients']]))] for r in locked['rows']]),
    'Every fit reduces calibration loss as expected, but that does not predict later improvement. MP033’s transformer correction shrinks outputs to about31% of their variation and adds an average .640 normalized offset. This severely raises predictions in the later low-speed interval. MP032 slopes range .471–8.162, exposing instability when calibration predictions have little variation. These observations reject this unregularized earlier-fit affine recipe, not every possible calibration method.',
    '## Error direction and speed changes',
    'The following diagnostics were specified before scoring. Positive signed error means speed overestimation; negative means underestimation. Relative-speed thresholds and rapid-change cutoffs come from the prior training-only protocol. Counts under30 remain descriptive sparse cells, not supported multi-animal state comparisons.',
    table(['Mouse', 'Low: N; signed error', 'Typical: N; signed error', 'High: N; signed error', 'Rapid rise: N; signed error', 'Rapid fall: N; signed error'],
          [[r['mouse']] + [f"{r['diagnostics']['attention'][s]['n']}; {number(r['diagnostics']['attention'][s]['signed_error'])}"
                          for s in ['low', 'typical', 'high', 'rapid_rise', 'rapid_fall']] for r in result['later']['rows']]),
    'The original transformer overestimates the low-speed average in every mouse and underestimates the high-speed average wherever high-speed windows exist. MP033 has43 high-speed windows with mean signed error−1.238 training-standardized units; this is the only high-speed slice with at least30 windows. Every directional rapid-rise/fall slice contains fewer than30 windows, so those signs cannot establish a general timing mechanism.',
    'For consecutive-change scoring, compare prediction[t] − prediction[t−1] with target[t] − target[t−1]. A zero-change reference always predicts a speed difference of zero; it is a reference for this secondary difference metric, not a new speed decoder with access to previous measured speed. First samples are omitted. No measured speed is supplied to either neural predictor.',
    table(['Mouse', 'Transformer change-MSE gain vs zero', 'MLP change-MSE gain vs zero', 'Transformer change correlation', 'Transformer change RMS / target RMS'],
          [[r['mouse'], pct(r['diagnostics']['attention']['consecutive_changes']['gain_vs_no_change']),
            pct(r['diagnostics']['mlp']['consecutive_changes']['gain_vs_no_change']),
            number(r['diagnostics']['attention']['consecutive_changes']['mean_pair_correlation']),
            number(r['diagnostics']['attention']['consecutive_changes']['rms_prediction_to_target'])] for r in result['later']['rows']]),
    'Neither original family beats zero-change MSE on MP030/032/033; both do on MP034. The MLP change error is lower than transformer change error on3/4 mice, despite losing the overall speed-MSE comparison. Thus good speed-level decoding does not establish precise tracking of moment-to-moment changes. MP030 transformer changes are more variable than true changes, whereas the other three are less variable: a universal smoothing or fixed-delay explanation is unsupported. Differencing can also emphasize measurement noise; this diagnostic cannot isolate neural information limits from label noise, representation limits or learning objectives.',
    '## Distribution context, added after the outcome',
    'This table was added to explain the failed correction, not to select or change it. Speed is in each mouse’s original training-standardized units; thresholds remain fixed. All raw descriptive values are in distribution_context.json.',
    table(['Mouse', 'Stage', 'Mean target speed', 'Target SD', 'Low-speed fraction', 'High-speed fraction'],
          [[r['mouse'], r['stage'], number(r['mean']), number(r['std']), f"{100 * r['low_speed_fraction']:.1f}%", f"{100 * r['high_speed_fraction']:.1f}%"] for r in context]),
    'MP033’s low-speed fraction changes from16.3% during calibration to66.4% later; MP034 changes from26.0% to84.0%. Mean target speeds also fall substantially. These observed distribution changes are consistent with calibration failing to transfer, but do not prove that distribution change is the only cause. Model selection reuse, small calibration samples and unrestricted slopes also limit the test.',
    '## Verification and next decision',
    f"All{audit['metric_scalar_checks']} independently accumulated scalar MSE/MAE checks and120 original pair-MSE matches pass. Calibration target alignment, frozen original epochs, all120 analytical optimality checks, coefficient-lock hashes and all{len(protocol['input_hashes'])} source/input/application hashes pass. Synthetic checks cover exact slope recovery, negative covariance, constant predictions, output flooring and pair-error averaging. Base models and main application files remain unchanged.",
    'Keep the unchanged shared transformer baseline and reject this correction. The useful next training hypothesis is whether explicit supervision of speed changes improves temporal tracking while preserving speed accuracy. Before a neural sweep, test whether a simple change decoder can beat zero change on earlier chronological data. Any later transformer test should apply the identical objective change to the MLP, retain speed MSE as the primary endpoint, freeze its budget and weight before scoring, and preserve constant-speed controls. This is a proposed hypothesis, not a proven fix or a queued fit. Current data remain exploratory; independent superiority still requires data unused in method development.',
    'Artifacts: [protocol](protocol.json), [coefficient lock](calibration_lock.json), [results](results.json), [audit](audit.json), [distribution context](distribution_context.json).'
]

text = '\n\n'.join(lines)
for a, b in [('by113', 'by 113'), ('by85', 'by 85'), ('seeds10', 'seeds 10'), ('all15', 'all 15'), ('slope1', 'slope 1'),
             ('interval:337', 'interval: 337'), ('Leave32', 'Leave 32'), ('contain726', 'contain 726'), ('all120', 'all 120'),
             ('least5', 'least 5'), ('least3', 'least 3'), ('least40', 'least 40'), ('over10', 'over 10'), ('by8.9', 'by 8.9'),
             ('and21', 'and 21'), ('only5/', 'only 5/'), ('and0/', 'and 0/'), ('has2.9', 'has 2.9'), ('but2.8', 'but 2.8'),
             ('about31', 'about 31'), ('under30', 'under 30'), ('has43', 'has 43'), ('least30', 'least 30'), ('than30', 'than 30'),
             ('on3/4', 'on 3/4'), ('from16.3', 'from 16.3'), ('to66.4', 'to 66.4'), ('from26', 'from 26'), ('to84', 'to 84'),
             ('and120', 'and 120'), ('All960', 'All 960'), ('all38', 'all 38')]:
    text = text.replace(a, b)
(ROOT / 'report.md').write_text(text + '\n')
(ROOT / 'ASSESSMENT.md').write_text('''# Output calibration assessment — 2026-10-04

Complete. A fixed scale-and-offset correction did **not** repair the errors: transformer mean relative later MSE increased **113.2%**, worsening all four mice; MAE increased **85.5%**. Identical MLP calibration also failed, increasing MSE **81.0%**. Both predefined calibration gates fail. Keep the original model outputs.

Fit 120 two-coefficient rules using only the first development half, with a 32-window gap before the development check. Preserve all six original selected checkpoints per family and all 15 unordered pairs. No network training or model inference. All corrections locked before scoring; no refitting or search after results. The base checkpoints previously saw full development labels, and these recordings were historically reused, so this is exploratory replay.

The unchanged transformer overestimates low speeds and underestimates high-speed peaks on average. Both transformer and MLP fail to beat a zero-change reference for consecutive speed changes on three of four mice. The MLP tracks changes better on three mice despite worse overall speed error. These are different endpoints; this does not undo the established speed-level result or establish a physical delay.

Post-hoc distribution context helps explain why fixed correction transfers poorly: low-speed prevalence changes from 16% to 66% on MP033 and from 26% to 84% on MP034 between calibration and later scoring. This is observed drift, not proof of a single causal explanation. Small calibration samples and unrestricted slopes can also matter.

Next bounded hypothesis: explicit speed-change supervision might improve tracking. First test a simple change decoder on earlier data; only then consider a fixed transformer/MLP objective comparison, keeping speed error primary. No such fit or application edit is queued. Independent superiority still needs data unused in method selection.

All 960 scalar metric checks, 120 archived pair-MSE matches, analytical calibration checks and frozen hashes passed. Application and prior experiment artifacts remain unchanged. [Full report](report.md) · [Protocol](protocol.json)
''')
print('Wrote report.md and ASSESSMENT.md; verified distribution context')
