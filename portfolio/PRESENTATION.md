# GitHub research-portfolio presentation

## Positioning

**Title:** Neural Behavior Decoding: What Survives Strong Baselines?

**One line:** A reproducible study of transformers, MLPs and ridge regression for mouse running-speed decoding, showing where architectural gains hold, where they disappear, and what separate-recording validation reveals.

**Short pitch:** I investigated whether transformers could extract useful behavioral relationships from population neural activity beyond strong simpler models. The recent work packages covered 572 neural fits, with a fixed-recipe evaluation on seven separate mice after extensive development. Population compression reduced measured CPU forward cost by about 15× relative to the local-neuron implementation, but both MLP and transformer benefited. The transformer did not establish reliable superiority, and two recordings exposed failures against zero speed. The deliverable is an auditable comparison, a concrete account of feature redundancy, and a portable bundle that rebuilds all finalist metrics and figures without training.

This is a draft the project author can adapt. Keep descriptions of personal contribution accurate, including tool assistance if asked. The published datasets were collected by the original experimental teams.

## Project bullets

- Built a controlled mouse-behavior decoding benchmark with chronological splits, training-only normalization, locked checkpoints, strong linear/nonlinear baselines, and per-mouse evaluation across development and separate cohorts.
- Compared population transformer and MLP decoders through a fixed structural/recipe search; measured approximately 15× lower batch-forward CPU time than the local-neuron MLP implementation, while documenting quality tradeoffs.
- Tested fixed recipes on seven separate mice; identified two recordings where all tested model families lost to zero speed, and preserved the failed superiority result rather than reporting only a favorable average.
- Packaged all finalist predictions, 303 metric rows, source hashes and complete per-mouse plots for reproduction without raw neural data or retraining.

Do not compress these into “achieved a 15× faster and more accurate transformer,” “proved coordinates do not matter,” “state of the art,” or “statistically significant improvement.” Those descriptions are not supported.

## Three-minute walkthrough

| Time | Show | Explain |
| --- | --- | --- |
| 0:00–0:30 | Root README | The question: does attention add reliable decoding value beyond identity, activity and strong baselines? |
| 0:30–1:10 | Overview figure | Development transformer-versus-ridge gains looked strong, but the tuned MLP was stronger; population compression reduced runtime for both families. |
| 1:10–1:50 | Zero-speed figure | Separate validation changes the interpretation: a relative win can still be a poor prediction. Keep every mouse, including the two failures. |
| 1:50–2:20 | Architecture diagram and coordinate explanation | Explain the population read-in and the earlier exact ID/coordinate compensation. Separate representation, optimization and added information. |
| 2:20–3:00 | `python3 portfolio/rebuild.py` and results table | Reproduce the evidence, identify the failed gates, and end on the narrower unresolved reliability question. |

## Questions a reviewer is likely to ask

**Why keep a transformer if the MLP is strong?** It remains a useful controlled reference and a plausible family for richer tasks. This benchmark does not establish a reason to prefer it for the current task. Keeping it is different from claiming it won.

**Was the search exhaustive?** It exhausted a declared 54-structure grid per population family, followed by bounded tuning. It did not cover every width, architecture, objective or training scale. Screening may miss slow learners.

**Does seven-mouse validation mean zero-shot transfer?** No. The recipes were fixed before new outcomes, then trained separately within each recording. Shared fitting is also reported descriptively. No unseen-mouse weights transfer is claimed from these results.

**Why is a positive 2.33% mean not enough?** It wins only three of seven mice, worsens MAE, misses the fixed practical threshold and fails the animal-level significance test. Selecting the mean alone would hide contradictory evidence.

**Is low running definitely the cause?** No. Lower test running mean and variance are measured, and every family fails the zero baseline on two mice. Activity changes, sampling support, preprocessing and coverage were not separated experimentally.

**What is original here?** The specific controlled evaluation, implemented comparison, preserved audit trail and synthesis of failure modes. Population transformers and neural decoding are established research areas. This is not a claim that their basic architecture was invented here.

**Can a reader retrain it immediately?** The portfolio reproduces metrics and figures from saved predictions. The clean `decoding/` package supports independent-recording fits after obtaining the large publisher recordings. It has synthetic pipeline tests and archived-checkpoint compatibility checks; the cleanup did not rerun the full real-data cohort or historical search. See [the training guide](../docs/TRAINING.md).

## Release scope

The release presents a controlled benchmark with its failed superiority gates intact. It includes the clean core, teaching guide, compact predictions, figures, offline viewer, and all historical small source/result records. The [experiment index](../research/EXPERIMENTS.md) distinguishes evaluated work from screening and the reconstruction study stopped before evaluation.

Original code uses MIT; dataset-derived assets retain their publishers' CC BY-NC 4.0 terms. Raw recordings, prepared arrays and checkpoints remain outside Git. Release and cleanup do not create a statistically significant transformer advantage.
