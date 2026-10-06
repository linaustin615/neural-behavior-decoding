# Evidence and claim boundaries

This is a retrospective presentation of completed work, not a new protocol or a search for a favorable endpoint. The primary results and gate decisions remain unchanged. Exact machine-readable support is bundled in [reference.json](data/reference.json); [source receipts](data/source_provenance.json) identify the original files and hashes.

| Claim | Verdict | Evidence and scope |
| --- | --- | --- |
| “Neural activity contains useful information about running in some recordings.” | Supported within the tested setting | Separate transformer mean-seed R²: TX103 0.471, TX57 0.123. Five other mice below 0.01. [All mice and controls](results/per_mouse_metrics.csv). This is association, not a causal account of behavior. |
| “The selected transformer reliably beats the MLP on separate mice.” | Unsupported | +2.33% mean MSE gain, 3/7 mouse wins, 11/21 seed wins, −1.20% MAE gain. Practical gate fails; Holm p = 1. [Fixed independent-fit comparison](results/RESULTS.md). |
| “The transformer reliably beats ridge on separate mice.” | Unsupported | +8.17% mean MSE gain, 3/7 mouse wins; both primary gates fail. A large development advantage did not become consistent separate-cohort evidence. |
| “The optimized transformer is a validated improvement over its smaller reference.” | Unsupported | Development +13.01% pooled mean gain, but the required first seed subset failed. Separate −0.22% mean gain. [All subsets](results/contrasts.csv). |
| “Population compression is substantially cheaper in the measured implementation.” | Supported engineering result | Batch-64 CPU medians: population T 5.61 ms, population MLP 5.82 ms, local MLP 84.97 ms. Not an attention-specific advantage, equal-accuracy claim or deployment latency measurement. [Timing source](../experiments/2026-10-06_systematic_optimization/cost_results.json). |
| “Correct coordinates were proved useless.” | Unsupported | Correct versus omitted coordinates: +0.50% MSE improvement, 7/12 paired wins, only 2/6 positive pool means; uncertainty crosses zero. One historically examined recording; overlapping neuron pools. [Coordinate report](../experiments/2026-10-02_coordinate16/report.md). |
| “The earlier additive tokenizer has an ID/coordinate redundancy for fixed known cells.” | Supported algebra and numerical check | Adding the coordinate-embedding difference to the corresponding free ID vector preserves the token sum. Max archived prediction discrepancy 1.67e-6. Does not extend automatically to new cells or all architectures. |
| “The trained model uses a feature, therefore it improves learning.” | Invalid inference | Coordinate deletion at evaluation raised error 6.35%, yet retraining gain was small and uncertain. Perturbation tests reliance; controlled retraining tests incremental value. |
| “Larger cell panels helped the archived development controls.” | Descriptive support | 128→512 cells: MLP pair-ensemble MSE gain 30.35%, ridge 32.24%, both 4/4 mouse means. Posthoc, reused mice; panel size changes capacity and summaries. Different endpoint from the current primary individual-seed score. [Archived context](data/reference.json). |
| “Zero speed can be a decisive baseline.” | Supported here | On TX104/TX61, every independent neural recipe and ridge loses to zero; transformer errors 20.01×/3.99× zero. This remains true despite its favorable average relative comparison with MLP/ridge. |
| “Behavioral shift caused the failure.” | Unresolved | Lower test running mean and variance are measured; bias explains only part of error. No experiment isolates causes. Release differences and sampling support also remain confounded. [Archived failure context](../experiments/2026-10-06_facemap_validation/failure_context.json). |
| “These weights transfer to unseen mice.” | Not tested by the separate-cohort result | Each independent recipe is trained anew on that recording; shared fitting includes each mouse's training prefix. This is fixed-recipe validation, not zero-shot transfer. |
| “We searched transformers exhaustively.” | False as a universal claim | The declared 54-structure grid per population family was exhausted, but screening budgets, width bounds, retained structures and architecture families were finite. |
| “We have an independently replicated significant neural-transformer improvement.” | Unsupported | All three primary new-cohort tests have adjusted p = 1. New seeds are technical replications, not new animals. Prior adaptive searching limits development claims. |
| “We created realistic neurons that generate mouse behavior.” | Not achieved | Earlier conditional reconstruction used measured context/behavior and nearly flat predicted activity. No validated neural simulator or autonomous behavior generator. |

## Why some positive numbers are not headline claims

The seven-mouse 2.33% average is not statistically or practically validated. The old 39.38% gain over ridge is a development result and disappears as a consistent advantage on the new cohort. The 13.01% gain over the small transformer fails the combined required seed-subset criterion. None of these is discarded; all appear alongside their failed conditions.

The coordinate uncertainty intervals describe one recording conditional on its analysis and overlapping pools. They are not population-level biological confidence intervals. The seed range in the trace figure is also not a biological confidence interval.

The strongest positive claims are narrower: some neural-to-running decoding occurred; an implementation choice reduced measured forward cost; and a concrete representational redundancy explains why feature reliance need not imply incremental predictive value. A transparent benchmark can demonstrate research skill without presenting these as a statistically significant new biological discovery.

## Source trail

The primary development records are [protocol](../experiments/2026-10-06_systematic_optimization/protocol.json), [selection](../experiments/2026-10-06_systematic_optimization/final_selection.json) and [results](../experiments/2026-10-06_systematic_optimization/results.json). The separate-cohort records are [protocol v2](../experiments/2026-10-06_facemap_validation/protocol.json), [summary](../experiments/2026-10-06_facemap_validation/summary.json) and [assessment](../experiments/2026-10-06_facemap_validation/ASSESSMENT.md). Original reports and audits remain in place. The portable bundle contains the numerical references needed for reproduction without those directories.
