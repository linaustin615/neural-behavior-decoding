# Larger neuron panel: complete

The expanded transformer improved over its 128-neuron parent by **8.19% mean relative MSE**, but lost to the equally informed 512-neuron MLP by **22.34%**, with every mouse average favoring the MLP. It failed the frozen first-stage gate, so no additional seed fits were launched. The bigger input panel is useful; these results do not support the bigger transformer over the stronger matched control.

Against the 128-neuron transformer, three mouse averages favored the larger model, but only six of twelve individual comparisons did. MP032 worsened by10.97%, exceeding the10% harm guard. Against the512-neuron MLP it won only2/12individual comparisons and worsened MAE by18.67%. Against512-neuron ridge it improved17.38%mean MSE and20.12%MAE, with3/4mice and8/12individual wins, but MP032 worsened23.99%, failing the harm guard. No threshold was relaxed.

Six new fits completed: transformer and MLP on seeds10–12, each with24epochs,5688updates and179712training presentations. Only neuron count changed from the accepted architecture; the nested panel contains every original cell. Both families received the same expanded inputs and training budget. The stronger ridge was selected on earlier data in the separate pilot. Original128-cell predictions were reused.

All workers and the final pipeline exited0. New input shapes, common initial tensors, finite gradients, additional ID learning and reload checks passed. Verification covered600selection errors,13308new neural predictions,2218ridge predictions,240independently recomputed scalar errors,42aggregate/gate checks,153frozen source hashes,30selected artifacts andeight prepared arrays. The figure was visually inspected. A status-file overwrite error stopped the controller before any training began; it was fixed without changing the protocol or repeating a fit, and the initial traceback is retained.

This experiment is closed. It used four historically searched mice and does not establish independent significance. The next separate question is which attention component explains the larger model's loss: temporal attention, activity-dependent readout, or their combination. The two missing512-cell factorial combinations can isolate these alternatives while reusing all completed controls; they cannot change this failed result. Main application files remain unchanged.

See [report](report.md), [protocol](protocol.json), [results](stage1_results.json), [review](review.json), and [figure](larger_panel.png).
