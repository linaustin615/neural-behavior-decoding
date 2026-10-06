# Isolated attention components: complete

Neither tested combination supplied a validated replacement. The earlier-selected temporal-attention/static-readout model passed the development screen but had **26.66% higher later MSE than the512-neuron MLP**, with only1/4mouse means and1/12individual comparisons favoring it. MAE was33.85%worse. The full first-stage gate failed, so no additional seed fits were launched.

Earlier mean selected scores were0.5763for temporal attention/static readout,0.7329for temporalMLP/dynamic readout,0.7815for the full-attention parent and0.6442for the all-MLP parent. The first variant therefore advanced under the frozen rule. Its later ranking reversed against the MLP. This demonstrates a selection-to-later performance reversal for this comparison; it does not identify the cause or justify selecting a different model using later outcomes. The losing variant was never later-scored.

The selected candidate improved5.92%mean MSE over the original128-neuron transformer but won only6/12individual comparisons, worsened mean MAE8.06%, and exceeded the10%harm guard onMP034. Against512-neuron ridge it improved19.57%mean MSE, with3/4mice and8/12individual wins, but MP032worsened11.66%. Against the full-attention512parent it was3.75%worse. All12candidate runs across mice/seeds beat the initial training-mean predictor. Learning occurred, but the required superiority did not.

Six new fits completed, each with24epochs,5688updates and179712presentations. Six trained-parent forwards and six initial-state comparisons matched exactly. All batch/update/reload checks,600earlier errors,6654new later predictions,240independently recomputed scalar errors,42aggregate/gate checks,195source hashes and30selected artifacts passed. All workers and the final pipeline exited0; the figure was inspected. Main application files are unchanged.

This study is closed with unchanged gates and no extra candidates or seeds. The separately frozen population-token fallback was triggered: it combines neurons into learned population signals before applying temporal attention. That alternative was specified before this study's outcomes and cannot change this failed result. Neither study supplies independent animal validation from these four historically searched mice.

See [report](report.md), [protocol](protocol.json), [candidate selection](candidate_lock.json), [results](stage1_results.json), [review](review.json), and [figure](components.png).
