# Failure review and proposed next steps

Publication is on hold until the user's evidence requirement is met. This document proposes development work; it does not amend the completed protocol, establish a new significant result, or authorize another training sweep. Generation remains part 2.

## What needs attention

| Issue | What is established | Action |
|---|---|---|
| Quiet periods reward trivial predictions | MP032 mean speed falls from 0.714 in training to 0.086 during selection and 0.047 in evaluation. Its seed-10 untrained predictions become exactly zero after clipping. | Include an explicit zero-speed baseline alongside the training-mean baseline. Report raw and bounded errors and prediction clipping rates. Keep quiet periods in the results. |
| Clipping and selection interact | Inspecting saved histories shows that raw MSE would choose a trained epoch for five of the eight epoch-0 selections, but all three MP032 seed-11 models would still choose epoch 0. | Treat output constraints/selection as one hypothesis, not a proven cure. Do not ban epoch 0 or retroactively change the metric. No alternative later predictions were computed or checkpoints reselected. |
| Single selection interval is unrepresentative | MP030 and MP034 also have much quieter selection intervals than their training/evaluation intervals. | Use multiple chronological development folds with gaps; inspect each fold separately. Avoid random adjacent-frame splits. Refit normalization and models using only each fold's training prefix. |
| Model complexity is not justified yet | Training losses fall, but generalization fails on MP032. The current 24-fit batch has no matched linear baseline. Older ridge results use different pools/tasks and cannot substitute. | Fit a regularized linear speed decoder with the same cells, history and target. If it works where the transformer fails, focus on the transformer recipe. If both fail, investigate temporal coverage and the neural-to-speed relationship first. |
| Static grouping can hurt | Grouping is based on simultaneous neural correlations, not speed relevance; MP033 is substantially worse than unrestricted attention. | Establish a stable unrestricted decoder first. Further grouping mechanisms require a separate hypothesis and fair controls; no new clustering sweep now. |
| Significance and independence | Four mice cannot produce two-sided p<0.05 under the frozen exact sign test; two of four effects currently favor grouping. | Choose the scientific claim and independent sampling unit before a confirmation study. More seeds/windows do not create more mice. A different test is not justified merely because it could give a smaller p-value. |
| Fresh confirmation is scarce | The archived release catalog lists seven spontaneous-recording mouse IDs: MP019, MP027, MP028, MP030, MP032, MP033, MP034. The older pilot examined the first three; the new batch examined the other four. | Do not call any of these mouse identities completely unexamined. Additional MP030 sessions can test session repeatability, but do not add independent mice. Staying within this exact release limits a genuinely new-animal confirmation. |

Existing file-integrity, shape, normalization, window-alignment and checkpoint checks passed. There is currently no evidence requiring another blanket data-validation pass or a wholesale code rewrite. The physical timing/internal acquisition-boundary limitation remains documented; it has not been established as the cause of these failures.

## First proposed experiment: development baselines only

Recommended initial claim, pending the user's preference: **neural activity plus cell identity supports reliable concurrent speed decoding beyond simple heuristics**. This is distinct from proving that grouping improves decoding, that IDs themselves are necessary, or that coordinates add information. A linear baseline can establish decodability; it cannot by itself establish a transformer advantage.

Use only the pre-evaluation portion of each of the four current recordings for this development pilot. The old evaluation tails stay closed during development, but are already inspected and cannot become fresh confirmation later.

1. Use three expanding chronological training prefixes with subsequent validation blocks: boundaries at 40%, 60%, 80% and 100% of the available pre-evaluation interval. Training stops 50 bins before each internal boundary; validation starts 50 bins after it and stops 50 bins before the next boundary, except at the final endpoint. Retain the inherited eight-bin history/31-bin target offset. These are repeated development views of each mouse, not independent biological replicates.
2. Select one 2,048-cell pool per mouse from cells eligible in its earliest training prefix only, with seed 101. Recompute preprocessing from the source recording for this purpose; the existing pool was chosen using a later prefix and must not silently become a training-only choice for earlier folds. Fit means/stds separately within each fold's training prefix. If eligibility or target variance is inadequate, record the failure without substituting a more favorable split.
3. Compare zero speed, the training-target mean, and a regularized linear model using the same eight-bin neural histories. Proposed objective: mean squared training error plus lambda times squared weights, with an unpenalized intercept; proposed lambda grid 0.01, 0.1, 1, 10 after training-only scaling. Record the solver convention explicitly so regularization strength does not change accidentally with sample count.
4. Budget: four mice × three folds × four fixed ridge strengths = **48 small linear fits, zero new transformer fits**. Report every candidate; any later recipe choice from these development results is tuning, not held-out confirmation. No adaptive grid expansion in this pilot.
5. Report per-fold MSE, raw/bounded predictions, constant and zero-baseline errors, target mean/variance, selected regularization and prediction variability. Report per-fold failures and the full regularization grid before choosing a common development recipe. Any running-versus-rest subgroup metric requires a physically justified or training-only threshold fixed before scoring; do not invent a favorable threshold afterward. No significance claim from the 12 correlated folds.

Decision branches:

- Linear decoding works consistently, transformer does not: next propose a bounded unrestricted-transformer study of optimization and output/selection rules against that baseline.
- Both fail in the quiet/shifted periods: prioritize temporal coverage and behavior-state generalization; adding architectural complexity is not supported.
- Both work but grouping still loses: the baseline decoder is usable; the grouping hypothesis remains unsupported.

These branches guide the next experiment; they do not retroactively change the failed replication gate.

## Confirmation requirement

Specify one primary comparison, minimum useful effect, significance level, multiplicity handling and population claim before confirmation. Under a two-sided exact sign test, six independent positive effects is the arithmetic minimum to get p=0.03125 for one comparison; this is not a sample-size/power recommendation. With two contrasts and Holm correction, six uniformly positive effects per contrast would still be insufficient to pass both at 5%. The actual needed sample size depends on the expected effect consistency and required power.

A within-recording claim could use a carefully justified temporal null/uncertainty model, but would not establish improvement across animals. Repeated model tuning on the existing evaluation intervals cannot support a fresh confirmatory p-value. New data must be reserved before tuning; a two-day deadline cannot guarantee a significant result.

## Status

No new fits or functional code edits were made during this review. No source datasets were revalidated. Only previously saved histories were inspected for the raw-versus-bounded selection question, and the archived catalog/older protocol were compared to establish which mouse identities have already been examined. The next teaching step, if proceeding with the recommended claim, is an experiment-local baseline/fold runner, leaving application `train.py` unchanged.
