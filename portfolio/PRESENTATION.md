# Presenting the project

## Short pitch

I built a reproducible benchmark for decoding mouse running speed from neural activity. It compares ridge regression, an MLP, a transformer and their combinations. On four sensorimotor mice, a validation-selected MLP–transformer ensemble reduced mean relative test MSE by 25.1% versus tuned ridge and about 5% versus either standalone neural model. Matched controls show that averaging is the dependable ingredient; a learned correction helps quiet predictions but introduces an active-period trade-off. Every comparison, failed trial and limitation remains traceable.

The neural-parent holdout comparison was pre-registered. The ridge comparison and same-family ensemble checks were post hoc. Each model was fitted within its recording; this is not zero-shot transfer.

## Three-minute walkthrough

| Time | Show | Explain |
| --- | --- | --- |
| 0:00–0:30 | README GIF | Recorded neurons and illustrative speed-driven mice; synchronized mix, corrected mix and ridge. The excerpt is chosen using observed behavior only. |
| 0:30–1:10 | Five-model chart | Same four mice and test frames; 25.1% lower MSE than ridge. Standalone neural models already help; combining them adds about 5%. |
| 1:10–1:40 | Architecture diagram | Activity feeds parallel MLP and transformer paths; validation chooses the mixing weight. Coordinates are for the 3D view only. |
| 1:40–2:10 | Correction trade-off | Quiet predictions drop 53%, but overall MSE improves just 1.16%; running-period error can increase. Keep the plain mix as the default. |
| 2:10–2:35 | Same-family ensemble controls | Two models of one family perform similarly. This is a practical ensembling result, not an invented uniquely superior transformer. |
| 2:35–3:00 | Research report and archive | Earlier seven-mouse results include failures against zero speed. Show honest cohort boundaries, locked selections and reproducible figures. |

## Portfolio bullets

- Implemented and compared population transformers, strong MLPs and validation-tuned ridge for within-recording neural-to-behavior decoding.
- Evaluated a combined decoder on four sensorimotor mice: 25.1% lower mean relative MSE than ridge in a post hoc comparison, and approximately 5% lower than each neural parent in the pre-registered holdout study.
- Preserved the complete experiment history, paired seeds, simple controls, validation locks and independent numerical audits; tested same-cost ensembles and correction trade-offs.
- Built an offline 3D replay and reproducible figures that connect recorded activity to predictions without presenting illustrations as biological simulation.

## Questions to be ready for

**Why use both models if an MLP is strong?** The mix beats either single parent here. However, same-family ensembles perform similarly at matched model count, so the evidence does not require two different architectures.

**Is correction better?** Slightly on overall average, but it missed the declared improvement requirement and worsens running-period error on some mice. We show it as experimental rather than promoting the largest headline number.

**Does the model generalize to an unseen mouse?** The training recipe was tested on previously unused mice, but weights were fitted separately within each recording. Pretrained-weight transfer was not demonstrated.

**Why are earlier results worse?** Different recordings and releases give different outcomes. The seven-mouse visual cohort includes low-running failures, whereas the four sensorimotor mice support the current result. The contribution of data shift, coverage, timing and modeling limitations was not fully isolated.

**What did you invent?** The implemented benchmark, controlled evaluation, audit trail and synthesis are the project contribution. Neither ensembling nor the basic transformer/MLP architecture is claimed as a new research invention.

**Can I reproduce it?** `python3 portfolio/build_current.py` rebuilds the current figures from bundled audited records. `python3 portfolio/rebuild.py` reconstructs the earlier303-row evidence from saved predictions. Retraining needs the original publisher recordings; see the training guide.

## Avoid these claims

“25% better because of attention,” “53% lower overall error from correction,” “works on any mouse,” “statistically significant,” “exhaustively searched every transformer,” and “generated realistic neural behavior” are not supported. Also avoid describing the entire project as unsuccessful: the ensemble result is a useful, measured improvement with clear limits.
