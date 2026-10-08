# Confidence-gating outcome

**The learned correction retains a substantial quiet-period effect, but the new confidence gate does not improve it overall.** The unchanged transformer–MLP blend remains the reference result. No models were trained and no earlier locked file or outcome was changed.

Across the four previously held-out mice, confidence-gated downward correction reduces mean relative quiet-frame predicted speed by52.59% versus the blend, with improvements on4/4 mice. It reduces mean relative total MSE by0.62%, winning3/4 mice. However, D3 total MSE is1.09%worse and active MSE is3.26%worse; D4 active MSE is1.30%worse. This fails the new prospective1%harm safeguards. The unchanged correction already reduced quiet predicted speed by53.08% and total MSE by1.16%, so this is not a new improvement over that correction.

Quiet-frame predicted speed, in training-SD units, averaged over three seeds:

| Mouse | Blend | Existing correction | Confidence-gated correction |
| --- | ---: | ---: | ---: |
| D3 | 0.087748 | 0.047183 | 0.047554 |
| D4 | 0.078299 | 0.041076 | 0.042659 |
| D7 | 0.069371 | 0.027024 | 0.024938 |
| D9 | 0.056662 | 0.024074 | 0.025512 |

The percentage is a reduction in mean predicted speed during observed quiet frames, not a53%reduction in overall prediction error or in a false-positive classification rate. Quiet frames can have small nonzero speed. Restricting corrections to decreases makes non-increasing quiet predictions a structural property; the meaningful test is the trade-off with total and active-period errors.

## Controls and development results

- Matched downward-only correction achieves0.68%lower total MSE on the four former holdout mice, compared with0.62%for confidence gating. Confidence gating provides no overall advantage over this simpler control.
- On the original seven development mice, gated correction averages1.55%lower MSE than the blend, but improves only2/7mouse means, lowers quiet predictions on4/7 and has worst active harm2.12%. Its prospective refinement gate also fails.
- Validation disables correction on TX103, TX56 and TX61. It favors25%strength on TX104/TX57/TX60 and75%on VR2. The earlier TX61 correction helped total MSE while harming activity; disabling it preserves the blend but sacrifices that improvement.
- TX60's later session is descriptive only:0.10%mean MSE improvement and1.98%quiet prediction reduction. It does not add an independent mouse.

## Decision and portfolio framing

Keep the ensemble as the main successful result. Describe the existing learned correction as an optional research refinement that suppresses false movement during quiet periods, with modest total-error benefit and an active-period trade-off. Do not claim this new confidence threshold fixes that trade-off or earns a new confirmation result. Neither cohort passes the targeted refinement criteria, and neither meets the historical5%overall improvement-over-blend requirement. No further threshold search is queued.

This follow-up explicitly reuses already examined development and former holdout recordings. It is not blind and does not change the completed pre-registered blend result. Choices were nevertheless restricted to validation and all24 recording/arm selections were locked before this follow-up scoring.

All24 validation selections and144 metric records passed independent arithmetic checks, along with saved prediction reconstruction, separate cohort aggregates, gate decisions and unchanged input/source/checkpoint hashes. See [full tables](REPORT.md), [protocol](protocol.json), [selection lock](selection_lock.json), [summary](summary.json), and [audit](audit.json).
