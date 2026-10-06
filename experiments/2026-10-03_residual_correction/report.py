"""Report the residual comparison against its frozen exploratory gates."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
FAIR=ROOT.parent/'2026-10-03_fair_comparison'


def read(path):
    return json.loads(path.read_text())


def main():
    p=read(ROOT/'protocol.json');results=read(ROOT/'results.json');locked=read(ROOT/'selection_lock.json')
    rows=[]
    for mouse in results['rows']:
        v={r['label']:r for r in mouse['results']}
        row=dict(mouse=mouse['mouse'],ridge10=v['ridge10']['mse'],selected_ridge=v['previous_ridge']['mse'],
                 zero=v['previous_zero']['mse'],mean=v['previous_mean']['mse'])
        for family in p['families']:
            values=[v[f'{family}_s{s}'] for s in p['seeds']]
            mse=float(np.mean([r['mse'] for r in values]));row[family]=mse
            row[family+'_seed_mses']=[r['mse'] for r in values]
            row[family+'_epochs']=[r['epoch'] for r in values]
            row[family+'_penalty']=values[0]['penalty']
            row[family+'_correction_rms']=[r['correction_rms'] for r in values]
            row[family+'_gain_vs_ridge10']=1-mse/row['ridge10']
            row[family+'_gain_vs_selected_ridge']=1-mse/row['selected_ridge']
            row[family+'_seed_wins']=sum(r['mse']<row['ridge10'] for r in values)
        row['transformer_gain_vs_mlp']=1-row['transformer']/row['pooled_mlp']
        rows.append(row)
    summary={}
    for family in p['families']:
        s=dict(mean_relative_gain_vs_ridge10=float(np.mean([r[family+'_gain_vs_ridge10'] for r in rows])),
               mean_relative_gain_vs_selected_ridge=float(np.mean([r[family+'_gain_vs_selected_ridge'] for r in rows])),
               mouse_wins=sum(r[family]<r['ridge10'] for r in rows),seed_wins=sum(r[family+'_seed_wins'] for r in rows),
               worst_mouse_relative_harm=float(max(-r[family+'_gain_vs_ridge10'] for r in rows)),
               epoch0_selections=sum(e==0 for r in rows for e in r[family+'_epochs']),
               mice_beating_both_constants=sum(r[family]<min(r['zero'],r['mean']) for r in rows))
        s['practical_gate']=bool(s['mean_relative_gain_vs_ridge10']>=.05 and s['mouse_wins']>=3 and s['seed_wins']>=6 and s['worst_mouse_relative_harm']<=.1 and s['mean_relative_gain_vs_selected_ridge']>0)
        summary[family]=s
    contrast=dict(mean_relative_gain=float(np.mean([r['transformer_gain_vs_mlp'] for r in rows])),mouse_wins=sum(r['transformer']<r['pooled_mlp'] for r in rows))
    contrast['attention_gate']=bool(summary['transformer']['practical_gate'] and contrast['mean_relative_gain']>0 and contrast['mouse_wins']>=3)
    summary['transformer_vs_pooled_mlp']=contrast
    calibration=read(ROOT/'calibration_results.json')['records']
    for family in ['constant','affine']:
        for row in rows:
            row[family]=next(r['mse'] for r in calibration if r['mouse']==row['mouse'] and r['family']==family)
        summary[family]=dict(mean_relative_gain_vs_ridge10=float(np.mean([1-r[family]/r['ridge10'] for r in rows])),
                            mice_beating_ridge10=sum(r[family]<r['ridge10'] for r in rows),
                            transformer_mean_gain_vs_this_control=float(np.mean([1-r['transformer']/r[family] for r in rows])))
    (ROOT/'summary.json').write_text(json.dumps(dict(per_mouse=rows,summary=summary),indent=2)+'\n')
    with (ROOT/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Frozen ridge plus learned correction','',
        'Completed32 correction fits (four mice × two families × two penalties × two seeds), plus12 chronological prefix ridge fits. No changes to the application or prior artifacts. This is historically inspected development data, not independent confirmation.','',
        '## Primary result','',
        '| Mouse | Fixed ridge10 | Previously selected ridge | Ridge + transformer | Ridge + pooled MLP | Zero speed |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['mouse']} | {r['ridge10']:.6f} | {r['selected_ridge']:.6f} | {r['transformer']:.6f} | {r['pooled_mlp']:.6f} | {r['zero']:.6f} |")
    lines+=['','Lower MSE is better within each mouse. Neural values average the two individual seed errors, not ensemble predictions. Different mice have different training-speed normalization.','',
        'Frozen exploratory gates and summaries:','', '```json',json.dumps(summary,indent=2),'```','',
        'The practical gate requires>=5% mean relative error reduction against ridge10,>=3/4 positive mouse means,>=6/8 seed wins, no mouse mean more than10% worse, and positive mean relative gain versus selection-tuned ridge. The attention gate additionally requires positive mean gain over pooledMLP corrections and>=3/4 mouse wins. These are development criteria, not significance tests.','',
        '## What was trained','',
        '- Frozen baseline:lambda10 ridge from the same complete outer-training prefix as the previous comparison. A fixed lambda prevents its OOF choice from depending on future residual-target labels. Selection-tuned ridge remains a mandatory stronger comparison.',
        '- Correction:the same81,377-parameter transformer or81,025-parameter attention-free pooled MLP, final linear layer initialized to zero. Output is `ridge + tanh(network(activity,IDs))`, bounding the correction to one outer-training speed SD.',
        '- Loss:raw combined prediction MSE plus0.1 or1.0 times mean squared correction. AdamW lr0.001,weight decay0.01,24epochs,cosine schedule,batch32,gradient clip1.',
        '- Correction targets:errors from three ridge models fitted only on preceding prefixes. Each prefix fits its own normalization; ridge predictions and labels are then expressed in the common outer-training units.',
        '- Each OOF block begins32 bins after its ridge fit ends. First input uses block bins24..31 and first target is bin31. No predictor uses that block’s target labels to fit its coefficients. Neuron eligibility uses only the earliest prefix.',
        '- Earlier selection picks epoch per seed/penalty including epoch0, then one penalty per mouse/family by mean selected error across both seeds. All32 choices lock before later scoring.',
        '- No temporal attention, token normalization or input clipping was added; this isolates the residual-design experiment from those additional hypotheses.','',
        '## Effective correction-training data','',
        '| Mouse | Full training examples | Out-of-fold correction examples | OOF mean residual |',
        '|---|---:|---:|---:|']
    for r in rows:
        o=read(ROOT/r['mouse']/'oof.json')
        lines.append(f"| {r['mouse']} | {o['full_train_n']} | {o['n']} | {o['mean_residual']:.6f} |")
    lines+=['','Correction training has fewer labeled examples than the direct models. Prefix ridge fits also have shorter training histories than the final frozen ridge. Their error distributions can differ; this is a limitation of this chronological stacking procedure, not a proven property of all residual architectures.','',
        '## Selected models and learning','',
        '| Mouse | Correction | Penalty | Seed10 /11 epochs | Seed10 /11 later MSE | Seed10 /11 correction RMS |',
        '|---|---|---:|---|---|---|']
    for r in rows:
        for family in p['families']:
            errors=' / '.join(f'{v:.6f}' for v in r[family+'_seed_mses'])
            rms=' / '.join(f'{v:.6f}' for v in r[family+'_correction_rms'])
            lines.append(f"| {r['mouse']} | {family} | {r[family+'_penalty']} | {r[family+'_epochs']} | {errors} | {rms} |")
    lines+=['','Epoch0 means exactly the fixed ridge prediction, not learned neural decoding. Every training run nonetheless received nonzero body gradients and updated its monitored weights after the zero head opened.','',
        '## Relative-quiet and moving errors','',
        'Threshold is the same training75th-percentile speed used previously; these are relative behavioral subsets, not verified physical rest. Whole-period MSE is primary.','',
        '| Mouse | Quiet / moving samples | Model | Quiet MSE | Moving MSE |','|---|---|---|---:|---:|']
    for mouse in results['rows']:
        v={r['label']:r for r in mouse['results']}
        for family in ['ridge10','previous_ridge','previous_zero']+p['families']:
            parts=[v[family]] if family in v else [v[f'{family}_s{s}'] for s in p['seeds']]
            values=['n/a' if parts[0][key] is None else f"{np.mean([r[key] for r in parts]):.6f}" for key in ['quiet_mse','moving_mse']]
            lines.append(f"| {mouse['mouse']} | {mouse['quiet_n']} / {mouse['moving_n']} | {family} | {values[0]} | {values[1]} |")
    lines+=['','## Supplementary calibration controls','',
        'These controls were declared and selected after neural fitting began but before any later scoring. The original frozen gates are unchanged. Constant and affine corrections use the same OOF ridge errors, penalize correction size with0.1/1.0, and include zero correction. Coefficients fit OOF data only; earlier selection chooses the candidate. Final corrections are bounded to[-1,1]. Affine fitting solves the unbounded penalized problem before clipping, rather than an optimal constrained problem.','',
        '| Mouse | Fixed ridge10 | Constant correction | Affine correction | Transformer correction | Pooled MLP correction |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['mouse']} | {r['ridge10']:.6f} | {r['constant']:.6f} | {r['affine']:.6f} | {r['transformer']:.6f} | {r['pooled_mlp']:.6f} |")
    lines+=['','See [supplement plan](calibration_plan.json), [locked calibration choices](calibration_lock.json) and [calibration results](calibration_results.json). Neither calibrator was fit or reselected using the later labels.','',
        '## Verification','',
        'Synthetic checks verified exact zero correction, finite gradients entering the body after the head opens, output bounds and chronological boundaries. Each new OOF ridge fit passed its normal-equation residual and non-BLAS prediction checks. Targets were matched to archived normalization. All32 selected checkpoints reproduced complete selection predictions; all800 selection-epoch errors were independently recomputed with scalar arithmetic. Batch orders matched across families/penalties. Later ridge predictions were checked against an independent computation; all later saved errors were independently recomputed. Prior artifacts and application hashes stayed unchanged.',
        '', 'See [assessment](ASSESSMENT.md), [protocol](protocol.json), [selection lock](selection_lock.json), [audit](audit.json), [numeric summary](summary.json), [comparison chart](comparison.png), and [later traces](later_traces.png).']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    fig,ax=plt.subplots(figsize=(10,5));xx=np.arange(4);width=.18
    for i,(key,label,color) in enumerate([('ridge10','Fixed ridge10','gray'),('selected_ridge','Selected ridge','tab:green'),('transformer','Ridge + transformer','tab:blue'),('pooled_mlp','Ridge + pooled MLP','tab:orange')]):
        ax.bar(xx+(i-1.5)*width,[r[key]/r['ridge10'] for r in rows],width,label=label,color=color)
    ax.axhline(1,color='black',linewidth=.7);ax.set_xticks(xx,[r['mouse'] for r in rows]);ax.set_ylabel('Later MSE / fixed ridge10 MSE')
    ax.set_title('Bounded corrections selected before later development scoring');ax.legend();fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(4,1,figsize=(13,12))
    for ax,r in zip(axes,rows):
        stats=read(FAIR/r['mouse']/'metadata.json')
        with np.load(ROOT/r['mouse']/'later_predictions.npz') as saved:
            ax.plot(saved['target']*stats['speed_std']+stats['speed_mean'],color='black',linewidth=1,label='Observed')
            for key,label,color in [('ridge10','Ridge10','tab:green'),('transformer_s10','Ridge + transformer seed10','tab:blue'),('transformer_s11','Ridge + transformer seed11','tab:cyan'),('pooled_mlp_s10','Ridge + pooled MLP seed10','tab:orange')]:
                ax.plot(np.maximum(saved[key]*stats['speed_std']+stats['speed_mean'],0),label=label,color=color,alpha=.7,linewidth=.8)
        ax.set_title(r['mouse']);ax.set_ylabel('Supplied speed units')
    axes[0].legend(ncol=2);axes[-1].set_xlabel('Later development sample');fig.tight_layout();fig.savefig(ROOT/'later_traces.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(per_mouse=rows,summary=summary),indent=2))


if __name__=='__main__':
    main()
