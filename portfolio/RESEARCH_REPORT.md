# Neural behavior decoding: what survives strong controls?

## Research question and outcome

Can a transformer use neuron identities and activity histories to predict a mouse's running speed more reliably than an equally informed MLP or ridge regression? Earlier work also asked whether measured neuron coordinates add information beyond identity and activity.

The project produced useful within-recording decoding in some cases and a substantially cheaper population representation. It did **not** establish a reliable transformer advantage over strong nonlinear controls. The separate-cohort experiment also exposed failures against a zero-speed predictor that relative model rankings alone would conceal.

This is a research-engineering case study: defining fair comparisons, searching a bounded space, preserving unsuccessful results, validating fixed recipes on separate recordings, and making the evidence reproducible. Statistical significance, novelty, deployment readiness and causal biological interpretation are not claimed.

## How the investigation developed

| Phase | Question | Durable lesson |
| --- | --- | --- |
| Spatial representations | Do correct positions help beyond activity and identity? | Small, inconsistent retraining gains; test-time reliance does not establish incremental value. |
| Activity relationships | Do functional grouping, attention and learned neuron readouts improve behavior decoding? | Correlation structure can persist without producing dependable predictive gains. Attention maps are not synaptic evidence. |
| Fair baselines and hybrids | Can pretraining, residual correction, mixed MLP/attention paths or calibration rescue performance? | Several attractive averages failed seed/mouse consistency or stronger controls. Different optimization budgets and ensembles require explicit comparisons. |
| Population representations | Can a short sequence of population states replace many neuron-time tokens? | Both transformer and MLP population models substantially reduce measured forward cost. Attention does not uniquely explain that benefit. |
| Systematic optimization | What survives a larger fixed structural/recipe search? | A better development transformer still lost to the independently tuned population MLP. The improvement-over-small-transformer consistency criterion failed. |
| Separate recordings | Do fixed recipes remain useful on seven separate mice? | No primary superiority gate passed; absolute prediction was weak on five mice, with two failures against zero speed. |

The [54-folder catalog](data/ARCHIVE.md) includes feasibility work, syntheses and reused fits. It is not a biological sample count. The stopped cross-neuron reconstruction study completed fitting but was halted before its later evaluation when the objective returned to behavior; it is not scored as a failed evaluated experiment. Generation remains a deferred objective.

## The evaluated architecture

The development-selected population transformer receives 512 identified neurons × 32 activity bins per example. It predicts running at the final input time. That is concurrent decoding with a history, not forecasting future behavior.

```mermaid
flowchart LR
    A[512 neurons x 32 activity bins] --> B[Recording-specific linear read-in]
    B --> C[64 population features per bin]
    C --> D[Four-bin patches: 8 temporal tokens]
    D --> E[Two causal temporal-attention blocks]
    E --> F[Final temporal token]
    A --> G[Population mean and SD histories]
    F --> H[Concatenate and predict running speed]
    G --> H
```

Neuron identity is represented by the neuron-specific read-in weights; coordinates are absent. Time and session embeddings are added before temporal processing. Each attention block contains both attention and a feed-forward MLP, with normalization and residual connections. Thus “transformer versus MLP” means comparing the full decoder families, not claiming a transformer has no MLP inside it.

The strongest population-MLP control keeps the population read-in, population statistics and prediction head, with learned causal temporal mixing and channel MLPs in place of temporal self-attention. Independent tuning selected a different patch size and dropout. That comparison addresses the best tested recipes within each family, not the isolated causal contribution of attention. A matched-settings MLP is also retained in development results; the transformer averaged 3.46% higher MSE against it.

| Recipe | Width / depth | History / patch | Dropout | Learning rate / weight decay | Epoch cap |
| --- | --- | --- | --- | --- | --- |
| Selected population transformer | 64 / 2 | 32 / 4 | 0 | 0.001 / 0.001 | 48 |
| Selected population MLP | 64 / 2 | 32 / 2 | 0.1 | 0.001 / 0.001 | 48 |
| Small population transformer | 16 / 1 | 32 / 4 | 0.05 | 0.001 / 0.01 | 24 |

The four-session development transformer has 223,809 parameters; the selected population MLP has 216,029. Recording-specific components make parameter counts depend on the number of sessions; these counts must not be presented as universal independent-fit counts. The old local-neuron MLP has 42,903 parameters but much higher measured forward cost. Parameter count alone does not predict runtime.

## Bounded optimization, not a universal optimum

The systematic study screened 54 transformer and 54 population-MLP structures: widths 16/32/64, depths 1/2/3, histories 16/32/64 and patches 2/4. Each was screened for 12 epochs on two chronological development folds. The top two structures per family received longer, predeclared recipe tuning; local-MLP controls were included. The study completed 440 new neural fits. Six finalist roles were compared with seeds 201–206, with separate requirements for each three-seed subset as well as their pooled summary.

The winning transformer width sits at the upper search boundary. Twenty-eight transformer screening fits selected the short epoch cap, so slower learners may have been screened out. Only two structures per family received longer recipe tuning. Local-MLP structure, larger widths, alternative objectives and all possible architectures were not exhaustively optimized. No final transformer selected the 48-epoch cap, but that does not prove global convergence.

The separate continuous-search work package contains 60 additional neural fits, and new-recording validation adds 72, for **572 new fits across these three non-overlapping work packages**. Earlier fits are not included in this conservative subtotal. Analytic ridge solutions, reused checkpoints and technical seeds are not counted as new animals or added to the neural-fit subtotal.

## What the comparisons show

Primary scores average each trained model's MSE over test targets, then average those errors over seeds within mouse. Relative gains are computed per mouse and finally averaged with equal mouse weight:

\[
g_m = 1 - \frac{\operatorname{mean}_s\mathrm{MSE}_{T,m,s}}{\operatorname{mean}_s\mathrm{MSE}_{C,m,s}},\qquad
G = \operatorname{mean}_m g_m.
\]

Positive gain favors the transformer. Averaging predictions before scoring is an ensemble endpoint and is not substituted for this primary score. MSE is measured in each recording's training-standardized target units; raw MSE values must not be pooled across recordings as though they had one common physical scale.

| Transformer comparison | Development: four reused mice, shared fits | Separate: seven mice, independent fits |
| --- | --- | --- |
| Versus selected population MLP | −10.28% mean MSE gain; 0/4 mouse wins | +2.33%; 3/7 mouse wins; MAE 1.20% worse |
| Versus smaller transformer | +13.01%; 3/4 wins; required first seed subset failed | −0.22%; 4/7 wins |
| Versus ridge | +39.38%; 4/4 wins; practical development criteria passed | +8.17%; 3/7 wins; primary gate failed |

Every separate-cohort primary contrast has Holm-adjusted two-sided animal sign-test **p = 1** and fails its practical gate. This does not prove equivalence or that transformers cannot work. It shows that the proposed advantage did not satisfy the prespecified evidence requirements here. Seven independent mice permit a smaller minimum exact sign-test p-value than four, but sample count cannot rescue inconsistent directions of effect.

Jointly fitting the seven new recordings was a secondary descriptive analysis because pooled training couples the animals. Its transformer averaged **1.50% higher MSE** than the selected MLP and won only 2/7 mouse means. The favorable independent-fit average cannot replace this outcome; neither can be chosen after looking at results as a rescue claim.

The separate practical rule required at least 5% mean relative MSE gain, at least 6/7 mouse wins, at least 14/21 individual-seed wins, no mouse more than 10% worse, and nonnegative mean MAE gain. A validated advantage also required the adjusted significance test and beating the training-mean control in every mouse mean. The archived thresholds were not changed for this presentation.

See [all results](results/RESULTS.md), [per-seed metrics](results/per_seed_metrics.csv), [per-mouse metrics](results/per_mouse_metrics.csv) and [every contrast](results/contrasts.csv).

## Three findings beyond the leaderboard

**Coordinates and learned identities can be redundant on known cells.** In the earlier additive tokenizer, a token was activity embedding + free ID embedding + coordinate embedding. For a fixed cell, replace its ID vector by `old_ID + position(real) - position(changed)`. The sum, and hence all downstream predictions, remains unchanged for any activity input. This is an algebraic property of that representation, not a biological assertion. Archived checks across twelve trained models recovered predictions within 1.67e-6. Removing coordinates only at inference raised error 6.35%, yet retraining without them changed mean error by only about 0.50%, with uncertainty intervals crossing zero. Reliance and added predictive information are different questions. Coordinates can still supply a useful prior for unseen cells or under constrained identity embeddings; that was not proved by the fixed-cell experiment.

**Population compression improved measured computational cost for both model families.** On the saved two-thread CPU benchmark, median batch-64 forwards took 5.61 ms for the selected population transformer, 5.82 ms for the population MLP, and 84.97 ms for the selected local MLP. That is roughly 15× less forward time for either population model. Batch-one timings were 0.371, 0.334 and 1.752 ms respectively. The benchmark used six seeds, randomized interleaved trials, 30 timed forwards and five warmups. It excludes preprocessing, I/O and GPU execution. It is not a claim of equal accuracy: the transformer harmed MP030 relative to the selected local MLP.

**Absolute controls exposed failure that relative gains obscure.** The separate transformer beat ridge on average, yet on TX104 and TX61 every tested neural recipe and ridge lost to zero speed. Transformer MSE was 20.01× and 3.99× that constant's error. Test target SD was only 0.125 and 0.270 of training SD, and test mean running shifted downward by 0.894 and 0.448 training SD. Posthoc squared prediction bias accounted for 37.9% and 34.0% of transformer MSE, leaving substantial fluctuation error too. Low-running periods are an observed failure context; causal responsibility of behavior shift, activity shift, preprocessing, sampling rate or training coverage has not been isolated.

![Failures against the zero-speed control](results/zero_speed_control.png)

The [full seven-mouse traces](results/all_mouse_traces.png) show every test interval and every mouse. Lines average three trained predictions for illustration; shading shows the seed range, not a confidence interval. The numerical results score individual models. Vertical scales differ, and native frames are not labeled as seconds.

An additional archived, posthoc panel-size comparison favored 512 rather than 128 cells in all four development mice: approximately 30.35% lower MLP pair-ensemble MSE and 32.24% lower ridge MSE. This older endpoint differs from the newer individual-seed primary endpoint. More cells also change model capacity, population summaries and regularization behavior; it does not isolate an attention benefit or provide new confirmation.

## What was validated—and what was not

The seven separate visual-cortex recordings were selected by an outcome-independent rule: the earliest recording for each of TX103, TX104, TX56, TX57, TX60, TX61 and VR2 in Facemap v2. Published IDs, dates and ages support non-overlap with the old MP cohort; direct identity confirmation from the authors was not obtained. The cohort remains a small convenience sample from one laboratory and brain region, not a random sample of all mice or behaviors.

Each independent fit used 512 random training-eligible cells, at most 4,096 evenly spaced training targets, chronological 60/20/20 partitions with 64-frame gaps, and a common 63-frame warmup. Activity and target scaling used training data only. Validation selected checkpoints, including epoch zero as a control; all final checkpoints were locked before test predictions. Three seeds were used per recipe. Ridge selected from three histories and eight penalties using validation data. No separate-cohort architecture search occurred.

Some released activity/run arrays differed by one terminal sample. A documented **pre-fit** protocol amendment kept the common valid prefix, following the author's same-index convention. No shift, interpolation or outcome-based alignment was selected. Raw ball-sensor synchronization was not independently reconstructed. This explicit convention is not proof of physical alignment.

The old release used nominal 1.2-second analysis bins; the new protocol retained 32 native frames and did not verify a physical seconds conversion. Facemap v2 also changed publisher deconvolution/motion correction. Consequently this is validation of fixed recipes on a separate release, **not a controlled test of behavior shift alone with identical physical input histories**. These differences could contribute to performance changes; their contribution is unmeasured. The absolute-zero-control failures occur within the separate cohort and remain valid regardless of a cross-release explanation.

Earlier audits checked checkpoint reloads, chronology, training-only scaling, batch-order matching and saved metrics. The present package adds a separate portability check for exported results; it does not redo those training audits. The archive has a long history of adaptive searching. No old test interval is relabeled as untouched confirmation. Further tuning on either current cohort would be development.

## A sensible stopping point and future research question

The current defensible endpoint is a benchmark and failure-analysis portfolio with three retained reference families: population transformer, strong population MLP and tuned ridge. A public claim of transformer superiority remains on hold. More trials on the same tails would spend compute without creating independent evidence.

If new scientific work is later authorized, the narrower question is: **can a fixed decoder remain useful across transitions between quiet and active running periods?** Physical time support and preprocessing should first be harmonized on development recordings, followed by a preregistered comparison of one justified remedy against the same MLP, ridge, training-mean and zero-speed controls. Absolute error, quiet-period false movement and per-mouse consistency should be fixed before scoring genuinely unused data. The quiet threshold, minimum useful effect and any recipe-selection rule must also be chosen on development data. Existing simple scale/offset calibration and several robustness approaches already failed; they should not be presented as untested breakthroughs or automatically rerun. No such new experiment is launched here.

## Data and related work

- **Original recordings:** [Stringer et al., “Spontaneous behaviors drive multidimensional, brainwide activity” (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6525101/) and its [public data](https://figshare.com/articles/Recordings_of_ten_thousand_neurons_in_visual_cortex_during_spontaneous_behaviors/6163622). This project reuses published activity and behavior rather than collecting experiments.
- **Separate cohort:** [“Facemap: a framework for modeling neural activity based on orofacial tracking”](https://pmc.ncbi.nlm.nih.gov/articles/PMC10774130/), the [version-2 dataset](https://janelia.figshare.com/articles/dataset/Facemap_a_framework_for_modeling_neural_activity_based_on_orofacial_tracking/23712957/2), and [pinned author plotting code](https://github.com/MouseLand/facemap/blob/f8b5b518efcde8b4aae7727283bf017b23641c7f/paper/fig4.py). Our endpoint predicts running from neural activity; it is not a reproduction of the paper's principal modeling task.
- **Neural Data Transformers:** [Ye and Pandarinath (2021)](https://arxiv.org/abs/2108.01210) study transformer representations of neural population dynamics. Neural transformers predate this project.
- **POYO:** [Azabou et al. (2023), author project and paper](https://poyo-brain.github.io/) use population read-in, latent attention and behavior queries for multi-session neural decoding. Population compression and behavior-oriented attention are established ideas, not novelty claims here.
- **POYO+:** [Author project and ICLR 2025 paper](https://poyo-plus.github.io/) extend multi-session, multi-task decoding to calcium recordings. The existence of successful neural-transformer work does not establish success on this particular running-speed benchmark, and our failures do not refute that literature.

These sources motivate and bound the project. Their datasets, input modalities, targets, training scales and splits differ. We did not reproduce these published systems head-to-head, and their reported performance is not placed on our leaderboard.
