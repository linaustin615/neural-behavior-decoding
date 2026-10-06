# Assessment

Separate validation is complete: seven visual-cortex mice,72neural fits and168ridge solutions. **No reliable transformer advantage was established.**

The independently fitted selected transformer averages2.33% lower MSE than the selected MLP, but wins only3/7mouse means and11/21seed comparisons, has1.20% worse MAE, and fails every required consistency/significance criterion (Holm-adjusted p=1). It averages8.17% lower MSE than ridge but wins only3/7mice; that contrast also fails. The selected larger transformer is0.22% worse than its smaller reference. Shared fitting is descriptive and puts the transformer1.50% behind the MLP.

Absolute accuracy is weak on five mice (selected-transformer R²<0.01); only TX103/TX57 exceed0.1 (0.471/0.123). Zero speed beats all neural recipes and ridge on TX104/TX61; transformer errors are20.01×/3.99× zero. Those periods have much lower running means and variability than training. These are observed failure conditions, not a proven causal explanation. The original four-mouse ridge advantage did not become consistent on this separate cohort.

Keep the frozen transformer, strong MLP, smaller transformer and ridge as benchmark recipes. The next useful question is reliability across changing running regimes; the current result does not justify a transformer-superiority claim or publication premised on one. Any new tuning on these seven mice is development. No additional experiments are queued.

Audit and visual review passed; application code and completed earlier experiments are unchanged. See report.md for all results, the source-supported one-frame indexing amendment, inherited publisher preprocessing and statistical limits.
