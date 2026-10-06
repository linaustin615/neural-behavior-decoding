# Broad architecture screen: completed assessment

Both transformer finalists failed the frozen practical gate. The useful positive signal came from a nonattention control, the normalized pooled MLP. These results do not establish transformer superiority or statistical significance.

## What was tested

All twelve proposed directions were represented by eleven concrete model/training prototypes and fixed ensemble comparisons. The screen used40 individual fits plus one joint four-session fit, one seed and12 supervised epochs. Earlier selection alone promoted token normalization and low-rank neuron IDs. Their preassigned nonattention controls and unchanged transformer/pooled-MLP references were evaluated with three seeds and24 epochs, followed by later-period scoring.

The completed budget was89 distinct new neural fits,1,740 supervised model-epochs and32 reconstruction model-epochs, plus16 reused reference fits. Eight screening fits continued into refinement without restarting. A new training-only PCA/RBF kernel baseline evaluated nine settings per mouse. Baseline selection chose ridge on MP030/033/034 and kernel regression on MP032 using earlier data; it did not choose whichever baseline won later.

## Locked later results

Percentages average within-mouse relative error changes after averaging the three individual seed errors. They are not percentages of pooled raw error or ensemble scores.

| Finalist | Error versus selected regression | Error versus assigned nonattention control | Mouse wins versus regression | Seed wins versus regression | Gate |
|---|---:|---:|---:|---:|---|
| Token normalization | 4.9% higher | 25.4% higher | 1/4 | 5/12 | failed |
| Low-rank IDs | 222.6% higher | 298.6% higher | 2/4 | 7/12 | failed |

Normalization reduced large errors on MP030/032 relative to the unchanged transformer, but worsened MP033 and slightly worsened MP034. Its mean relative gain against that transformer was32.1%. This is evidence that the recipe can reduce particular failures, not that attention is necessary. The recipe bundles token LayerNorm with a population-amplitude bypass, so this experiment does not isolate LayerNorm alone.

Low-rank IDs retained wins on MP033/034 but failed badly on MP030/032. It should not replace the current model based on these results. Temporal attention, recurrence, convolutions, multiscale history, masking, graph messages, robust weighting and shared sessions did not advance from this bounded screen. One seed and12 epochs cannot rule out their entire families.

Both finalists' descriptive98.75% bootstrap intervals cross zero against regression and their assigned controls. Leaving out individual mice also exposes instability. The reports retain all failures and epoch-zero selections.

## The promising secondary finding

The normalized pooled MLP had15.5% lower mean relative error than the selected regression baseline, won on3/4 mice and9/12 seeds, and beat its own untrained output on12/12 comparisons. Against ridge alone, its mean gain was12.9%. Its mouse gains against the selected baseline were10.7%,21.9%,-23.5%,52.8% for MP030/032/033/034. MP032 still favored the constant zero-speed predictor over this MLP.

The low-rank pooled MLP also improved mean relative error by9.9%, with3/4 mouse wins. These are useful secondary control results, not newly promoted finalists or confirmatory discoveries. The normalized MLP's apparent advantage deserves independent verification before adoption or publication claims.

Fixed three-seed averaging gave the normalized transformer a7.0% mean gain against regression, but the normalized MLP ensemble gained19.4%. Averaging therefore did not establish a transformer-specific advantage, and ensemble scores do not replace the individual-seed gate after seeing outcomes.

## Decision and verification

Stop this architecture sweep. Preserve the normalized pooled MLP as the strongest new benchmark candidate and retain the original transformer as a comparator because it still performs well on particular recordings. Independent confirmation of a frozen comparison is the next scientific requirement. More tuning on these same four historically inspected recordings cannot create that independence; optimizer seeds and overlapping windows are not additional animals. The other previously inspected Stringer recordings are not pristine confirmation either.

All89 new fits passed checkpoint reload checks. The completion audit independently checked1,868 saved selection predictions, reconstructed the original shortlist, checked matching batch orders, confirmed every expected run and verified frozen source/reference/preparation/checkpoint hashes. All72 final model choices were locked before current later scoring; later errors and kernel predictions passed independent numerical checks. Reports include4,000 paired hierarchical block-bootstrap resamples per finalist, individual seed scores, epoch-zero comparisons, quiet/moving subsets and fixed ensembles.

Application files remain unchanged. No publication action was taken. No jobs or further fits remain scheduled.

See [full report](report.md), [numeric summary](summary.json), [protocol](protocol.json), [evaluation audit](audit.json), [completion audit](completion_audit.json) and [chart](comparison.png).
