"""Summarize the fixed development-only complementarity pilot."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def main():
    summary, results, protocol, locked = [read(n) for n in ['summary.json', 'results.json', 'protocol.json', 'locked_rules.json']]
    assert read('audit.json')['passed'] and read('numerical_audit.json')['passed']
    rows = summary['rows']['score']
    names = {
        'attention_mlp_left': 'MLP', 'attention_mlp_right': 'Transformer',
        'attention_mlp_calibration_parent': 'Earlier-chosen parent',
        'attention_mlp_half': '50:50 hybrid', 'attention_mlp_global': 'Global blend',
        'attention_mlp_session': 'Per-mouse blend', 'attention_mlp_gate': 'Gated hybrid',
        'mlp_mlp_gate': 'Two-MLP gate', 'mlp_mlp_global': 'Two-MLP global blend',
        'mlp_mlp_session': 'Two-MLP per-mouse blend'}
    columns = ['attention_mlp_left', 'attention_mlp_right', 'attention_mlp_half',
               'attention_mlp_global', 'attention_mlp_session', 'attention_mlp_gate', 'mlp_mlp_gate']
    lines = ['# Hybrid complementarity: completed pilot', '',
        'The transformer and MLP make complementary errors, but the tested earlier-fitted routing rules do not consistently outperform both parents or the two-MLP controls. Both primary practical gates fail. This supports a possible opportunity, not an established advantage for a hybrid architecture.', '',
        'No base model was trained again. Six archived shared experts (AA transformer and MS static-query MLP, three seeds each) supply every epoch’s development predictions. Six small ten-coefficient gates and analytic scalar weights were fitted under one fixed protocol. The original final later evaluation archives were not read.', '',
        '## Temporal replay', '',
        '| Mouse | Select expert epoch | Fit blending rules | Score rules |',
        '|---|---|---|---|']
    for m, bounds in protocol['split'].items():
        lines.append('| ' + m + ' | ' + ' | '.join(f'[{a}, {b}) ({b-a} windows)' for a, b in bounds.values()) + ' |')
    lines += ['', 'Indices refer to the existing development interval. Checkpoints are reselected using only its first third, ignoring the original full-interval choices and ridge denominators. Each boundary discards 32 windows before the next block, preventing overlap between 32-bin contexts. The middle block fits the blend; all six rules lock before final-block errors are computed. There are 544 calibration and 544 scoring windows across four recordings, shared across the seed comparisons.', '',
        'Experts were trained on the earlier training interval with a fixed schedule. This replay avoids using final-block labels in its current checkpoint and blending choices. Historical model and research decisions have already used these recordings, including the development interval; this is not a pristine test or independent confirmation.', '',
        '## Score-block errors', '',
        '| Mouse | ' + ' | '.join(names[k] for k in columns) + ' |',
        '|---|' + '---:|' * len(columns)]
    for r in rows:
        lines.append('| ' + r['mouse'] + ' | ' + ' | '.join(f'{r[k]:.6f}' for k in columns) + ' |')
    lines += ['', 'Entries are bounded, training-normalized MSE, averaged over individual seed errors. They do not average seed predictions. Gains below average within-mouse relative MSE changes, with equal mouse weight; they differ from percentage changes in pooled or mean absolute MSE.', '',
        '| Comparison | Mean relative gain | Mouse wins | Paired-seed wins | Preset threshold |',
        '|---|---:|---:|---:|---|']
    for c in summary['comparisons'].values():
        lines.append(f"| {names[c['main']]} vs {names[c['control']]} | {100*c['mean_relative_gain']:.2f}% | {c['mouse_wins']}/4 | {c['paired_seed_wins']}/12 | {'Pass' if c['threshold_pass'] else 'Fail'} |")
    lines += ['', 'Positive gains favor the first model. Each threshold requires at least 2% mean relative gain, three mouse wins and eight paired-seed wins. The global-blend primary gate requires passing against both parents. The routing gate requires passing against both parents, 50:50, global and per-mouse blending, and the gated two-MLP control. Both combined gates fail. No method was promoted after inspecting final-block outcomes.', '',
        'The gated hybrid improves on the single MLP by 14.8% (three mice, nine paired runs), but its mean relative error is 55.5% higher than the transformer, 7.3% higher than the two-MLP gate, 1.8% higher than the per-mouse constant blend, and 0.15% higher than simple 50:50 averaging. It improves on the global learned blend by 10.8%, with four mouse wins. Improvement against one comparator does not establish the full hybrid hypothesis.', '',
        'The transformer comparison is sensitive to MP032, where the transformer’s MSE is 0.000824 and the hybrid’s is 0.003407. The hybrid wins the other three mouse means, but loses by 313.4% on this small-error recording. The table exposes the absolute scale; counting wins alone would hide this failure. The training objective weights normalized absolute MSE equally by mouse, whereas the reported relative-error endpoint strongly weights small-error comparisons. These objectives are not identical, and no retrospective reweighting is performed.', '',
        '## Complementary errors and unattainable reference rules', '',
        '| Mouse | Mean signed-error correlation | MLP better windows | Transformer better windows | Hard oracle gain | Convex oracle gain |',
        '|---|---:|---:|---:|---:|---:|']
    for r, oracle in zip(rows, summary['oracle_headroom']):
        diagnostics = [d for d in results['diagnostics'] if d['stage'] == 'score' and d['pair'] == 'attention_mlp' and d['mouse'] == r['mouse']]
        means = {k: np.mean([d[k] for d in diagnostics]) for k in ['signed_error_correlation', 'left_better_fraction', 'right_better_fraction']}
        lines.append(f"| {r['mouse']} | {means['signed_error_correlation']:.3f} | {100*means['left_better_fraction']:.1f}% | {100*means['right_better_fraction']:.1f}% | {100*oracle['hard_oracle_gain_vs_best_parent']:.1f}% | {100*oracle['convex_oracle_gain_vs_best_parent']:.1f}% |")
    hard = np.mean([o['hard_oracle_gain_vs_best_parent'] for o in summary['oracle_headroom']])
    convex = np.mean([o['convex_oracle_gain_vs_best_parent'] for o in summary['oracle_headroom']])
    mlp_hard = np.mean([1-r['mlp_mlp_hard_oracle']/min(r['mlp_mlp_left'],r['mlp_mlp_right']) for r in rows])
    lines += ['', f'The hard oracle chooses the less wrong expert separately for every window using its true speed; the convex oracle also chooses the perfect weight using that answer. Their gains are {100*hard:.1f}% and {100*convex:.1f}% against the better parent mean for each mouse. They are hindsight lower bounds on error, not usable predictors, and do not establish that the expert preference is learnable from available inputs. Ties explain window fractions that sum to less than 100%.', '',
        f'Two MLP seeds also have oracle headroom ({100*mlp_hard:.1f}% hard-oracle mean gain over their better parent). Error diversity is therefore not specific to mixing architectures. The two-MLP gate beating the tested hybrid is the relevant practical control.', '',
        '## What the gate learns', '',
        'The gate uses both model predictions, their absolute disagreement, three label-free activity summaries, and session indicators. Only middle-block feature means/stds and labels enter fitting. It outputs a weight between zero and one; the hybrid stays between the two parent predictions. The gate has ten coefficients and a fixed penalty, one initialization and one optimization budget. No architecture, feature or regularization sweep was run.', '',
        '| Pair | Seed | Global right-expert weight | Per-mouse weights MP030/032/033/034 | Optimizer steps |',
        '|---|---:|---:|---|---:|']
    for name, rule in locked['records'].items():
        pair, seed = name.rsplit('_s', 1)
        lines.append(f"| {pair} | {seed} | {rule['global_alpha']:.3f} | " + ', '.join(f'{x:.3f}' for x in rule['session_alpha']) + f" | {rule['iterations']} |")
    lines += ['', 'The transformer’s role varies across mice and seeds, and earlier weights do not consistently transfer to the scoring block. The tested dynamic rule fails to improve on the simple per-mouse scalar rule. This does not rule out all gates, but adding a more complex gate is not supported merely by oracle headroom.', '',
        '## Verification and limits', '',
        'Synthetic checks verified exact scalar solutions and endpoints, ties, a recoverable known mixture, finite-difference gate gradients, convex bounds, target isolation for checkpoint selection, and context gaps. Real archives align exactly with development targets. All six gate optimizers converged; 432 saved MSEs match independent scalar calculations. Frozen source, input and application hashes remain unchanged.', '',
        'NumPy emitted floating-point warnings inside matrix multiplication during fitting despite finite results. Before scoring, an independent calculation without BLAS reproduced every saved gate prediction to within 4.5e-16, calibration objectives to within 2.3e-16, and found maximum gradient below 8.5e-8 at each solution. Original locked coefficients were retained. This checks the numerical result without claiming the warning’s library-level cause was diagnosed; see numerical_audit.json.', '',
        'Scoring blocks contain only 99–193 windows per mouse and retain temporal autocorrelation. Seeds and the cyclic two-MLP pairs are not independent animals. Four historically examined recordings and a new split do not create fresh significance. Checkpoint selection uses a smaller prefix than previous studies, so these MSEs are not directly comparable with earlier full-selection/later-evaluation tables.', '',
        'Keep the existing standalone baselines and preserve this negative routing result. The evidence supports complementary errors, but neither robust two-parent superiority nor an attention-specific ensemble advantage. No new base-model fit, final-tail evaluation, application edit, publication or expanded gate search is queued.', '',
        'See [assessment](ASSESSMENT.md), [protocol](protocol.json), [locked rules](locked_rules.json), [numeric summary](summary.json), [audit](audit.json), [numerical audit](numerical_audit.json), and [runner](run.py).']
    (ROOT / 'report.md').write_text('\n'.join(lines) + '\n')
    writeup = '''# Hybrid complementarity: completed

The transformer and MLP make complementary errors, but the tested hybrid does not consistently outperform both parents or the two-MLP control. Both predefined practical gates fail.

No base-model training was repeated. We reused every-epoch development predictions from six shared models, reselected epochs on the first third, fitted blending rules on the middle third, and scored the last third after locking all choices. Each boundary has a 32-window gap. Four mice and three seeds; 544 scoring windows in total. Six small gates were fitted, with no parameter search. The original final later interval was not opened.

| Result on the scoring block | Finding |
|---|---|
| Gated hybrid versus one MLP | 14.8% lower mean relative error; 3/4 mice, 9/12 paired runs |
| Gated hybrid versus transformer | 55.5% higher mean relative error; wins 3/4 mice but loses heavily on low-error MP032 |
| Gated hybrid versus two-MLP gate | 7.3% higher mean relative error; 1/4 mouse wins |
| Gated hybrid versus simple 50:50 | 0.15% higher mean relative error |
| Gated hybrid versus per-mouse constant blend | 1.8% higher mean relative error |
| Perfect hindsight choice of expert | 27.6% lower error than the better parent per mouse; not deployable |

The gate does improve on a global learned blending weight, but does not beat simpler and stronger alternatives consistently. Its input-dependent weighting has not demonstrated a reliable advantage beyond ordinary ensembling. The oracle result establishes headroom only; it cannot show that a real gate can identify which model will be right without knowing the target.

The failure is not absence of complementary predictions. It is failure of this fixed routing recipe to turn their differences into dependable improvement across future blocks and strong controls. This result concerns one small gate, not every possible hybrid.

Numerical, temporal-separation and archive-alignment checks passed. Matrix-multiplication warnings were investigated before scoring: independent non-BLAS calculations reproduced the locked outputs/objectives and verified small gradients; no fit was replaced. Application files and earlier studies are unchanged.

This is an exploratory development replay on historically examined recordings, not independent statistical confirmation. We should keep the standalone baselines and avoid promoting this hybrid. No additional architecture search or training job is queued.

See [full report](report.md), [summary](summary.json), [protocol](protocol.json), [audit](audit.json), and [numerical checks](numerical_audit.json).
'''
    (ROOT / 'ASSESSMENT.md').write_text(writeup)


if __name__ == '__main__':
    main()
