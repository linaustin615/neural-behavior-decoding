# Defensible project conclusion after the final batch

Training-only activity groups show persistent co-activity structure in later data from this recording. The within-group correlations are modest in absolute size (0.039–0.047), but exceed matched random reassignment averages (about 0.005) in all three neuron pools. This supports an association structure worth visualizing. It does not establish synapses, causal influence, or behavior-specific relationships.

The activity-plus-ID decoder learns a useful concurrent running-speed signal. Every one of the 18 final models beats its untrained version and the training-mean constant on the later evaluation interval. The unrestricted reference averages R² 0.782; activity grouping averages R² 0.788. This is within-recording decoding, not forecasting or a generative model of movement.

Adding explicit activity groups is not yet a dependable improvement. Their mean error is 2.58% lower than the unrestricted reference, with positive pool means in 3/3 pools, but only 4/6 paired wins and a wide conditional interval crossing zero. Against random groups, the 7.97% mean advantage is driven by pool 101; only 2/6 runs and 1/3 pool means favor activity grouping. Both frozen gates fail. Do not report the average gains without this inconsistency.

The most defensible description is:

> We decoded concurrent running speed from neural activity and neuron identity, and identified activity-defined groups whose co-activity persisted in a later interval of the same recording. Adding these groups to the decoder did not establish a reliable improvement over matched controls. Coordinates were used to visualize the population and were not inputs to this final comparison.

The prior coordinate and reconstruction studies also remain negative or inconclusive for their intended stronger claims. These experiments do not establish useful added anatomical information, independent-animal generalization, realistic artificial neural activity, or a system that generates running behavior.

Stop the current-recording model search here. Preserve the unrestricted activity-plus-ID decoder as the reference and use the learned groups as an exploratory visualization. A stronger predictive or biological claim requires a justified frozen comparison on genuinely unused recordings. Changing random seeds or making another plot of the same recording would not supply that confirmation.

All 18 final fits, saved predictions, checkpoints, verification results, source hashes, protocol, numerical summaries and plots are archived. No application source was changed and no further jobs are scheduled.
