# Dynamics baseline: completed assessment

The baseline is implemented and fully evaluated. Neural pretraining improved average running-speed decoding for both attention and the nonattention mixer. Attention did not establish an overall advantage over the matched controls. Both preset scientific gates failed; the implementation and evaluation checks passed.

## What we built

The model keeps separate neuron and time dimensions: 128 neurons, each with eight four-bin activity patches. A temporal block processes each neuron's history, then a population block exchanges information across neurons. A neural forecasting head predicts the next four activity bins. A separate speed head combines the final neural representation with population activity statistics.

We compared attention and a similarly sized nonattention mixer, each trained from scratch and after neural forecasting pretraining. Ridge regression and a normalized pooled MLP use the same neurons, histories and target endpoints. All models train separately per mouse. This is a small literature-informed baseline, not a novel architecture, a full published-model reproduction, or a test of coordinates.

## Results on the later interval

Percentages below average relative MSE changes within each mouse, giving all four mice equal weight. Neural scores average three individual seed errors; ensembles do not replace them.

| Comparison | Result |
|---|---|
| Pretrained attention vs scratch attention | 16.8% lower error; 4/4 mouse-average wins, but only 7/12 paired-seed wins |
| Pretrained mixer vs scratch mixer | 16.1% lower error; 3/4 mouse-average wins |
| Pretrained attention vs pretrained mixer | 14.7% higher error; 1/4 mouse wins, 5/12 seed wins |
| Pretrained attention vs matched pooled MLP | 0.5% higher error; 3/4 mouse wins, 9/12 seed wins; 103.3% worse on MP032 |
| Pretrained attention vs matched ridge | 22.8% lower error; 3/4 mouse wins |
| Attention neural forecasting vs selected simple forecast baseline | 1.75% lower error; 4/4 mouse wins |
| Mixer neural forecasting vs selected simple forecast baseline | 1.43% lower error; 4/4 mouse wins |

Attention's pretraining gains by mouse were 6.9%, 49.6%, 7.9% and 2.8% for MP030, MP032, MP033 and MP034. Its aggregate gain therefore depends substantially on MP032, though every leave-one-mouse-out mean remains positive. On MP034, pretrained attention beat the pretrained mixer by 43.0%; that is a conditional advantage, not consistent superiority across recordings.

The speed gate required at least 5% average improvement against every primary comparator, at least three mouse wins, at least eight paired-seed wins against each neural comparator, and no mouse more than 25% worse than the MLP. It failed several conditions. Neural forecasting improved consistently but fell below its separate 5% threshold.

All four descriptive 98.75% bootstrap intervals for the primary speed contrasts cross zero. These are four historically inspected mice, not fresh independent confirmation. More seeds and overlapping windows do not create more independent animals. We cannot claim statistical significance.

## What this establishes

All 60 supervised fits beat their own epoch-zero checkpoint on the later interval. Learning occurred. The pretraining recipe has a promising aggregate effect, but its benefit was almost the same for attention and the mixer: their relative improvements differ by only 0.69 percentage points. These results do not establish an attention-specific pretraining benefit.

Pretrained models also receive 24 extra training epochs. The scratch comparison measures the entire pretraining recipe, not the neural objective independently of additional optimization. No equal-total-update supervised control or frozen-backbone probe was included. Forecasting errors remain close to simple baselines, so this is not evidence of realistic generated neural dynamics or recreated behavior.

The strongest defensible next foundation is to retain both sequence models and the simpler controls in this harness. A future representation-quality question could be isolated with a frozen-backbone readout and an equal-total-update supervised control, under a new protocol. No extra sweep is queued, and these results should not be used to select a favorable mouse retrospectively.

## Budget and verification

Completed 24 neural-pretraining fits plus 60 supervised fits: four mice, seeds 10/11/12, 24 epochs per fit. All 24 pretraining choices were locked before supervised fitting; all 60 supervised choices were locked before this study's later evaluation. Training used no selection or later examples. Context and future targets stay inside chronological segments, with the inherited split gaps exceeding the context plus forecast horizon.

Architecture causality, shapes, gradients, updates, checkpoint reloads, common initialization, window indexing, linear solves and predictions passed their checks. All 1,500 saved speed-selection predictions and 24 selected forecast predictions were verified. Batch orders matched within each stage. Later targets matched the archived endpoints exactly; speed metrics and all 60 forecasting metrics were independently recomputed. Frozen sources, inherited data and application hashes remained unchanged. All four workers exited successfully; no jobs remain running or scheduled.

The 128-neuron subset and one-block models limit scope. The 32-bin history is nominally about 38 seconds, and the forecast about 4.8 seconds. Old 2,048-neuron/eight-bin results are unequal-input references only. Main application files were not changed or published.

Start with [models.py](models.py) for the architecture, [report.md](report.md) for all scores, [protocol.json](protocol.json) for the frozen design, [audit.json](audit.json) and [forecast_metric_audit.json](forecast_metric_audit.json) for verification, and [comparison.png](comparison.png) for the comparison chart.
