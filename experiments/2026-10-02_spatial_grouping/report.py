import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parent
r=json.loads((ROOT/'results.json').read_text())
p=json.loads((ROOT/'probe_results.json').read_text())
population=json.loads((ROOT/'population_probe_results.json').read_text())
synthetic=json.loads((ROOT/'synthetic_results.json').read_text())
followup=json.loads((ROOT/'synthetic_fixed_results.json').read_text())
labels={'global':'Unrestricted16 summaries','spatial':'Spatial patches + global summaries',
        'random':'Random groups + global summaries','depth_random':'Depth-matched random groups + global summaries'}
lines=['# Spatially anchored summaries: completed exploratory study','',
       '**This design did not establish a dependable spatial advantage.** Spatial patches reduced mean test error by only0.32% versus random groups and0.52% versus depth-matched random groups, while having3.88% higher error than unrestricted summaries. All three paired intervals included zero. The grouped mechanism was active, but creating distinct summaries did not produce a reliable decoding benefit.', '',
       'All48 planned main fits completed:2,048 neurons, six neuron samples, two optimization seeds, four grouping conditions. Each model has81,377 trainable parameters. All conditions retain neuron IDs and the same true-coordinate token embeddings. This isolates the added effect of spatial grouping; it does not remove every source of geometry. The task decodes concurrent running speed from an8-bin history that includes the current bin; it is not future-speed forecasting.',
       '', '| Grouping | Validation MSE | Test MSE | Test R² |','|---|---:|---:|---:|']
for v,g in r['groups'].items():
    lines.append(f"| {labels[v]} | {g['validation_mse']:.6f} | {g['test_mse']:.6f} | {g['test_r2']:.4f} |")
lines+=['','MSE uses standardized speed; lower is better. Predictions are bounded at zero speed for the primary metric, matching previous experiments. Raw unbounded metrics are retained in results.json. The original8-summary reference had test MSE'+f" {r['previous8_summary_reference']['test_mse']:.6f}.",
        '', '| Spatial comparison | MSE improvement | Relative change | Spatial wins | Positive sample means | Conditional95% interval |',
        '|---|---:|---:|---:|---:|---|']
for c,x in r['contrasts'].items():
    lo,hi=x['interval95']
    lines.append(f"| vs {labels[c]} | {x['mean_mse_improvement']:+.6f} | {x['relative_improvement_percent']:+.2f}% | {x['positive_pairs']}/12 | {x['positive_pool_means']}/6 | [{lo:+.6f}, {hi:+.6f}] |")
lines+=['','Positive improvement means spatial patches reduce error. Intervals use2,000 paired crossed sample/seed bootstraps with circular time blocks of100 bins. These are conditional uncertainty summaries within this recording, not confidence intervals across animals.',
        '',f"Frozen candidate gate passed: **{r['candidate_gate']}**. Stronger within-recording gate passed: **{r['stronger_within_recording_gate']}**. Exact criteria were frozen before training in protocol.json. Failing a gate does not prove a zero spatial effect.",
        '', '## Mechanism checks','',
        '| Grouping | Attention participation rank | Encoded-feature participation rank | Test MSE increase when local read-in is replaced by global mean |',
        '|---|---:|---:|---:|']
for v,a in p['audit_summary'].items():
    lines.append(f"| {labels[v]} | {a['attention_rank']:.4f} | {a['encoded_rank']:.4f} | {a['local_ablation_test_mse_increase']:+.6f} |")
lines+=['','Attention ranks are computed separately within attention heads on64 validation examples per fit; feature ranks concern the16 summary vectors, not the full network. Disjoint local masks create distinct attention supports by construction. Ablation changes the input distribution of later layers: sensitivity shows reliance on summaries, but is not evidence that true anatomy outperforms random grouping.',
        '', '## Output-layer probe','',
        'A posthoc probe freezes each of the48 trained encoders and fits a regularized linear decoder to either the32 averaged features or all512 separate-summary features. Five regularization strengths are selected by validation, using training-only scaling. This adds480 coefficients to the separate-summary decoder and reuses the validation set already used to select encoders.',
        '', '| Grouping | Refit averaged-feature test MSE | Separate-feature test MSE | Separate-feature wins | Paired improvement interval |',
        '|---|---:|---:|---:|---|']
for v,g in p['groups'].items():
    d=p['separate_minus_mean_probe'][v];lo,hi=d['interval95']
    lines.append(f"| {labels[v]} | {g['mean']['test_mse']:.6f} | {g['separate']['test_mse']:.6f} | {d['positive_pairs']}/12 | [{lo:+.6f}, {hi:+.6f}] |")
lines+=['','This is a diagnostic of recoverable information, not a replacement end-to-end training experiment. All96 chosen heads and480 regularization fits are reported. No head is selected using its test score.',
        '', '## Population-wide activity probe','',
        'PCA is fitted only on training activity. Linear speed decoders use8-bin histories of the first1,4, or16 principal components, with validation-selected regularization. Six neuron samples and five regularization strengths give90 fits and18 selected models.',
        '', '| Principal components | Mean test MSE | Mean test R² |','|---|---:|---:|']
for k,g in population['summary'].items():
    lines.append(f"| {k} | {g['test_mse']:.6f} | {g['test_r2']:.4f} |")
lines+=['','The first principal component alone is insufficient in this test. This does not rule out other low-dimensional, supervised, or nonlinear summaries. NumPy/BLAS issued matmul warnings despite finite outputs; independent Torch projections and regularized linear solves reproduced all18 models. See population_probe_independent_check.json.',
        '', '## Artificial regional-interaction check','',
        'The target is a product of differences between four known regional signals. This is a stronger and different nonlinear task than real speed regression. Six initial fits use two seeds and three grouping arms; all use the same initialization within seed.',
        '', '| Seed | Grouping | Original test R² | Epochs run | Full50-epoch follow-up R² |','|---|---|---:|---:|---:|']
for x in synthetic['runs']:
    rr=[f for f in followup['runs'] if (f['seed'],f['variant'])==(x['seed'],x['variant'])]
    value=f"{rr[0]['test_r2']:.4f}" if rr else 'Already ran50'
    lines.append(f"| {x['seed']} | {x['variant']} | {x['test_r2']:.4f} | {x['epochs_run']} | {value} |")
lines+=['','The original frozen positive-control gate, spatial R² above0.75 in both seeds, failed. A separate follow-up reran only the three early-stopped fits for all50 epochs. Learning improved, but spatial runs still fell below that threshold and random groups performed better. This flags optimization/readout limitations of this particular architecture; it prevents treating a real-data null result as a general rejection of spatial interactions. No real-data budget or model changed in response.',
        '', '## Data and integrity','',
        '- All11,983 coordinate rows match their ROI metadata exactly: horizontal coordinates are twice stat.med, and depth is -90 minus30 times the imaging-plane index. Physical units were not independently verified. Nine depth values identify imaging planes, not nine cortical layers.',
        '- Patches use recursive threshold splits on depth, then x, then y. Random controls preserve group sizes; depth-matched controls preserve exact group-by-imaging-plane counts. Group definitions use no activity or speed labels.',
        '- The48 fits share initial parameters and batch order within matched repeats. Attention supports, finite gradients, joint neuron-permutation invariance, training-only normalization and chronological window boundaries passed preflight checks.',
        f"- All48 saved checkpoints were reloaded on64 validation examples;{p['full_checkpoint_reloads']} prespecified checkpoints were checked on full validation and test splits. Maximum sampled reload error:{p['max_reload_64_validation_error']:.3g}.",
        f"- All saved metrics were recomputed from predictions. {r['learning_check']['beat_untrained']}/48 models beat their untrained test error;{r['learning_check']['beat_train_mean']}/48 beat the training-mean baseline. {r['learning_check']['final_epoch_selected']}/48 selected the final allowed epoch.",
        '- This is one repeatedly examined mouse recording. Neuron samples overlap, seeds are technical repeats, and the test period has informed earlier work. No independent-animal or pristine confirmatory claim is warranted.',
        '- Application files model.py, data.py and train.py retain their original hashes. Experimental code and checkpoints are isolated here.',
        '', '## Reproduction','',
        'With the existing data/stringer_spontaneous.npy file, run group_search.py on shard_0.json through shard_3.json in a clean copy to refit. Completed records are reused. Run mechanism_audit.py and readout_probe.py after the main fits, then analyze.py, summarize_probes.py, report.py and plot_summary.py. Each training worker uses two CPU threads. Population and synthetic probes have separate scripts and protocols. Source hashes, dataset hash, environment and final artifact manifest are included.']
report=re.sub(r'(?<=[A-Za-z])(?=[0-9])',' ','\n'.join(lines)+'\n')
report=re.sub(r'(?<=[:;])(?=\S)',' ',report).replace('1,4, or 16','1, 4, or 16')
(ROOT/'report.md').write_text(report)
print('report.md written')
