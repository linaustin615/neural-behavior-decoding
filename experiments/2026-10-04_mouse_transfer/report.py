"""Render the completed fixed-budget mouse-transfer experiment."""
import json
from pathlib import Path

import numpy as np


ROOT=Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT/name).read_text())


def pct(value):
    return f'{100*value:+.1f}%'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])


def main():
    summary=read('summary.json')
    audit=read('audit.json')
    assert audit['passed']
    locked=read('selection_lock.json')
    prep=read('prepared.json')
    assert read('pre_evaluation_lock.json')['passed'] and read('prefix_audit.json')['passed']
    c=summary['comparisons']
    transfer=c['attention_scratch']
    matched=c['mlp_transfer']
    utility=summary['attention_utility_gate']
    arms=['attention_transfer','attention_scratch','mlp_transfer','mlp_scratch','ridge','initial','median']
    rows=summary['rows']
    headline=('The practical transformer transfer gate passes.' if summary['transformer_transfer_gate'] else
              'The practical transformer transfer gate fails.')
    headline+=' '+('The broader limited-label utility gate also passes.' if utility else
                   'The broader limited-label utility gate fails.')
    lines=[
        '# Held-out-mouse transfer — 2026-10-04',
        headline+' These are fixed-budget exploratory results on historically inspected recordings, not independent significance.',
        '## Frozen question',
        'Does a network trained on three source mice help decode running speed for an excluded fourth mouse when only224 target behavior labels are available? Rotate through all four mice with seeds10/11/12. Compare transferred and freshly initialized transformer and MLP models, plus target-only ridge and constant references. This tests performance at one fixed label budget; it does not measure a learning curve or quantify how many labels transfer saves.',
        'The model is the exact archived BehaviorDecoder:128 neurons,32 neural bins, eight four-bin patches, a temporal block and a behavior-query readout. The transformer uses temporal attention and activity-dependent query keys. The matched MLP uses static causal time mixing and input-independent query keys with activity-dependent values. Both use the same additional population summaries. This compares two full architectures, not an isolated causal contribution of attention.',
        '## Target exclusion and label budget',
        'Each source model trains and selects a checkpoint using only the three other mice. Its session list, batch iterator and validation risk exclude the held-out mouse. Source training uses full original source training intervals and source-only development selection. No final evaluation interval is used for source training.',
        'For the target mouse, fit on its first160 archived training windows and select on64 windows at indices[192,256). Leave32 windows between them so neural contexts do not overlap. All target arms receive the same224 labels, the same batches,24 searched epochs and120 optimizer updates. Initial normalized speed predictions are exactly zero for every target neural model. The held-out mouse is excluded from pretraining but is explicitly supervised during adaptation; this is not zero-shot transfer.',
        'Undo cached normalization using archived statistics, then calculate new target activity mean/SD from the191 unique neural bins covered by the fitting windows. Calculate speed mean/SD from the160 fitting labels only. Apply those fixed prefix statistics to target selection and later raw data. A SD below1e-6 is replaced with1. Float32 cache reconstruction introduces rounding. The fixed128-neuron panel is inherited from prior unlabeled full-training eligibility preprocessing; the study limits target behavioral labels but does not claim that all target neural-data preprocessing was untouched.',
        'Copy all selected source shared parameters except the neuron-ID table, session embedding and final scalar output layer. Those remain freshly initialized, exactly matching the target-only controls. No neuron identity is assumed to correspond across mice. Fine-tune all target parameters with a new optimizer; retain the transferred hidden readout and feature weights.',
        table(['Mouse','Fit labels','Selection labels','Unique normalization bins','Target speed mean','Target speed SD','Prefix-constant cells','Ridge lambda'],
              [[r['mouse'],r['fit_n'],r['selection_n'],r['unique_activity_bins'],f"{r['speed_mean']:.5f}",f"{r['speed_std']:.5f}",r['prefix_constant_cells'],r['ridge_lambda']] for r in prep['records']]),
        '## Training and selection',
        'Complete24 source fits and48 target fits, with24 epochs each. Source batches target equal-mouse risk. AdamW uses lr.001,weight decay.01,batch32,gradient clipping1 and a24-epoch cosine schedule ending at.0001. The transfer and scratch arms have equal target-data, adaptation-update and target-checkpoint-selection budgets. Transfer additionally uses source data and computation; total training compute is not matched.',
        'Select one source epoch from0–24 by equal-mouse mean source validation MSE in source training-standardized units. Select target epochs from0–24 by bounded speed MSE on the64 target validation labels. Target-only ridge uses the same160 fit/64 selection labels, with penalties.001/.01/.1/1/10, training-only feature centering and an unpenalized intercept. All72 neural choices and ridge choices are locked before any new later inference. The zero baseline means the target fitting-prefix mean speed; the median baseline is the fitting-prefix median. All speed outputs are floored at physical zero.',
        '## Practical gates',
        'Primary transfer gate: transferred transformer versus target-only transformer needs at least5% equal-mouse mean relative MSE gain, at least3/4 mouse wins and at least8/12 paired seed wins. Broader utility additionally requires those neural thresholds against transferred MLP and target-only MLP, at least5%/3 mouse wins against ridge, no mouse more than25% worse than ridge, and at least8/12 wins over initial outputs. These are practical gates, not tests of independent significance.',
        table(['Control for transferred transformer','Mean MSE gain','Mouse wins /4','Seed wins /12','Mean MAE gain','Practical comparison'],
              [[name,pct(v['mean_relative_mse_gain']),v['mouse_wins'],v.get('paired_seed_wins','not applicable'),pct(v['mean_relative_mae_gain']),
                ('PASS' if v['practical_threshold'] else 'FAIL') if name in ['attention_scratch','mlp_transfer','mlp_scratch','ridge'] else 'secondary context'] for name,v in c.items()]),
        f"Transformer transfer gate: **{'PASS' if summary['transformer_transfer_gate'] else 'FAIL'}**. Broader utility gate: **{'PASS' if utility else 'FAIL'}**. Ridge harm guard: **{'PASS' if summary['ridge_harm_guard'] else 'FAIL'}**. Wins versus initial output: **{summary['initial_wins']}/12**.",
        '## Per-mouse results',
        'MSE below uses each target’s fitting-prefix speed normalization, so its scale differs from prior studies and across mice. Compare within each mouse; the headline effect averages four relative gains equally. Average errors across the three model seeds; do not average their predictions into an ensemble.',
        table(['Mouse','N']+arms,[[r['mouse'],r['n']]+[f"{r['means'][a]['mse']:.6f}" for a in arms] for r in rows]),
        table(['Mouse','Transfer transformer vs scratch transformer','Transfer transformer vs transfer MLP','Transfer MLP vs scratch MLP','Transfer transformer R²'],
              [[r['mouse'],pct(transfer['mouse_gains'][i]),pct(matched['mouse_gains'][i]),pct(summary['mlp_transfer']['mouse_gains'][i]),f"{r['means']['attention_transfer']['r2']:.3f}"] for i,r in enumerate(rows)]),
        'Matched MLP transfer effect: '+pct(summary['mlp_transfer']['mean_relative_mse_gain'])+f" mean relative MSE gain, {summary['mlp_transfer']['mouse_wins']}/4 mouse wins and {summary['mlp_transfer']['paired_seed_wins']}/12 paired seed wins. This determines whether any benefit is shared by the nonlinear control rather than uniquely supported for the transformer.",
        'Native-unit MSE, MAE, R² and every seed score are preserved in results.json. The target constant-speed controls matter particularly for quiet recordings; beating a weak neural control alone does not establish useful decoding.',
        'In this completed run, transfer improves transformer MSE17.2% and20.4% on MP033/MP034, while slightly worsening MP030/MP032 (1.7% and1.5%). Only6/12 transferred-transformer models beat their initial training-mean prediction. Three target transformer runs select epoch0; retain those outcomes rather than force a trained checkpoint. Transformer R² across MP030/032/033/034 is−2.208/−.094/.033/.302. These limited-label models are weak on several recordings, even when a relative comparison improves.',
        'This does not revise the earlier full-data, known-mouse decoder result. The present experiment uses224 target labels and excludes that mouse from pretraining; the earlier shared decoder had full supervised training on each mouse. Its six-seed pair advantage and the failed transfer test address different settings. Neither result should be substituted for the other.',
        '## Uncertainty and sensitivity',
        table(['Comparison','Descriptive97.5% interval','Leave-one-mouse-out mean gain range'],
              [[name,' to '.join(pct(v) for v in summary['descriptive_97_5_intervals'][name]),
                pct(min(c[name]['leave_one_mouse_out']))+' to '+pct(max(c[name]['leave_one_mouse_out']))] for name in ['attention_scratch','mlp_transfer']]),
        'The2000 bootstrap draws resample mice, shared seed identities and circular100-bin time blocks, keeping each comparison paired. They are conditional on fitted models and do not account for retraining uncertainty, overlapping source folds or prior method selection. All four recordings have historical reuse. Neither seed counts nor correlated time windows create additional independent animals; these intervals do not convert the experiment into independent confirmation.',
        '## Checkpoints and budgets',
        table(['Held mouse','Seed','Stage','Family','Mode','Selected epoch','Updates','Examples'],
              [[r['held'],r['seed'],r['stage'],r['family'],r['mode'],r['selected_epoch'],r['updates'],r['examples']] for r in locked['records']]),
        '## Verification and decision',
        f"All{audit['selection_scores_checked']} selection scores reproduce from saved predictions, all selected reloads match exactly, and all{audit['scalar_later_mse_checks']} later MSEs pass independent scalar checks. Actual Adam step counters, target batch matching, source-mouse exclusion, exact shared-weight copies, fresh target embeddings/output layers, target-only prefix statistics, normalization boundaries and frozen source/input/application hashes pass. The target speed head starts at zero; a selected epoch0 is retained honestly if it beats trained checkpoints on the small validation block.",
        'An independent cache-space calculation confirms prefix normalization, with input differences under1.3e-6 from float32 rounding and matching normalized labels. A separate pre-evaluation lock binds preparation metadata, all neural choices and independently checked ridge argmins; its hashes remain unchanged after scoring.',
        'Execution note: evaluation completed all four mice, wrote results.json and a passed audit, then exited with NameError because the final environment-metadata write referenced platform without importing it. Environment metadata was recovered separately. No model fit, inference or numerical score was rerun or changed. The frozen numerical source is preserved; evaluate.py supplies the missing metadata-only import for a clean reproduction. See metadata_recovery.json. This is a reporting-stage recovery, not a passed end-to-end exit from the original evaluator.',
        'Retain the frozen outcome and its gate status. Do not choose another target prefix, transfer subset, learning rate, seed set or objective after seeing these scores. This is a one-budget transfer test using historical data, not a novel architecture claim or proof of universal superiority. Main application edits and publication remain paused. No further training is queued.',
        'Artifacts: [protocol](protocol.json), [selection lock](selection_lock.json), [summary](summary.json), [all results](results.json), [audit](audit.json), [assessment](ASSESSMENT.md).'
    ]
    text='\n\n'.join(lines)
    import re
    text=re.sub(r'(?<=[a-z])(?=\d)', ' ', text)
    (ROOT/'report.md').write_text(text+'\n')
    assessment=f'''# Held-out-mouse transfer assessment — 2026-10-04

Complete: 24 source fits, 48 target fits and 20 ridge candidates; four held-out mice and three seeds. Each target contributes 160 fit labels plus 64 checkpoint-selection labels. Source training excludes that mouse. Target normalization, fresh neuron embeddings and output-layer reset preserve the fixed adaptation comparison.

**Transformer transfer gate: {'PASS' if summary['transformer_transfer_gate'] else 'FAIL'}. Broader utility gate: {'PASS' if utility else 'FAIL'}.**

- Transferred transformer versus target-only transformer: **{pct(transfer['mean_relative_mse_gain'])}** mean relative MSE gain; {transfer['mouse_wins']}/4 mice and {transfer['paired_seed_wins']}/12 paired seeds win.
- Versus transferred MLP: **{pct(matched['mean_relative_mse_gain'])}**; {matched['mouse_wins']}/4 mice and {matched['paired_seed_wins']}/12 paired seeds win.
- Versus target-only MLP: **{pct(c['mlp_scratch']['mean_relative_mse_gain'])}**; {c['mlp_scratch']['mouse_wins']}/4 mice win.
- Versus target-only ridge: **{pct(c['ridge']['mean_relative_mse_gain'])}**; {c['ridge']['mouse_wins']}/4 mice win. Ridge harm guard {'passes' if summary['ridge_harm_guard'] else 'fails'}.
- MLP transfer versus its own scratch control: **{pct(summary['mlp_transfer']['mean_relative_mse_gain'])}**; {summary['mlp_transfer']['mouse_wins']}/4 mice win.

Positive percentages mean improvement. This uses one fixed 224-label target budget, not a learning curve or a quantified reduction in labels required. Transfer adds source training; total compute is not matched. The fixed neuron panels inherit earlier unlabeled preprocessing, and every animal was historically examined. Results are exploratory, not independent significance or an isolated attention-mechanism claim.

All required checks passed. No application files changed and no further fits are queued. Preserve the outcome without tuning another target prefix or transfer recipe to these scores. [Full report](report.md) · [Protocol](protocol.json) · [Detailed results](summary.json)

Execution note: the evaluator completed numerical scoring and its audit, then failed on a missing import in the final environment-metadata write. That metadata was recovered separately without repeating training or inference; the original evaluator exit code is recorded as1.
'''
    (ROOT/'ASSESSMENT.md').write_text(assessment)
    print(headline)


if __name__=='__main__':
    main()
