# Shared fine-tuning assessment — complete

The fixed fine-tuning search did not improve the transformer. Every transformer fit selected epoch zero under both ordinary and averaged-weight selection. The unchanged model remained the best earlier-validation choice; no losing continuation was inspected on later data. This is evidence against these continuation recipes, not a proof that all optimization methods fail.

All18fits completed:two rates1e-4/3e-5 × two families × seeds10–12,then selected rate1e-4/ordinary checkpoints for both families on13–15. Each added8epochs/1896updates/59904presentations from the original selected checkpoint. Original weights, architecture, preprocessing and labels were preserved; optimizer state was reset. All starts and selected reloads matched exactly. Weight averaging included the starting point and epoch-end weights.

Both families selected the unchanged checkpoint throughout initial search. On additional seeds, only MLP seed14 selected a trained checkpoint (epoch2); the other MLP and every transformer remained unchanged. MLP later MSE improved4.35% across additional seeds (4/4mouse means but4/12single wins) and2.05% across allsix (4/4mice but4/24single wins). Its improvement gate failed. Transformer improvement and full fine-tuning gates fail because the transformer has zero improvement over itself.

The original transformer still beats the equally fine-tuned MLP:25.64% mean MSE gain on13–15 (4/4mice,11/12seed wins,19.56%MAE gain) and17.68% on allsix (3/4mice,18/24seed wins,7.77%MAE gain). This supports retaining the original as the strongest candidate; it is not a successful new fine-tuning recipe. Additional training did not establish an architecture limit or a single cause of errors.

Audit:1296selection metrics,72exact native starting predictions,26616new later predictions,2112independent scalar errors,96archived prediction arrays. Independent review passed743aggregate/gate/accounting checks,94source/input hashes,90selected-artifact hashes and4prediction hashes. All workers/stages exited0,figures inspected. No application files or old study sources changed. No extra rates/epochs were appended. Allfour mice were historically searched; no independent significance is claimed.

The pre-frozen residual-readout fallback was executed separately. Continuous search remains authorized, but this specific fine-tuning study is closed. Details: [report](report.md), [protocol](protocol.json), [results](results.json), [review](review.json), [figure](finetuning.png).
