"""Summarize the frozen comparison without any outcome-based model selection."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT/name).read_text())


def main():
    results=read('results.json')
    locked=read('selection_lock.json')
    p=read('protocol.json')
    summary=[]
    for mouse in results['rows']:
        values={r['label']:r for r in mouse['results']}
        row=dict(mouse=mouse['mouse'],zero=values['zero']['mse'],mean=values['mean']['mse'],ridge=values['ridge']['mse'],
                 ridge_guard=values['ridge_guard']['mse'],ridge_lambda=values['ridge']['regularization'])
        for family in p['families']:
            for kind in ['tuned','default','untrained']:
                row[family+'_'+kind]=float(np.mean([values[f'{family}_{kind}_s{s}']['mse'] for s in p['seeds']]))
            row[family+'_guard']=float(np.mean([values[f'{family}_tuned_s{s}_guard']['mse'] for s in p['seeds']]))
            choice=next(c for c in locked['choices'] if c['mouse']==mouse['mouse'] and c['family']==family)
            row[family+'_recipe']=choice['recipe_id']
            row[family+'_epochs']=[values[f'{family}_tuned_s{s}']['epoch'] for s in p['seeds']]
            row[family+'_gain_vs_ridge']=1-row[family+'_tuned']/row['ridge']
            row[family+'_gain_vs_default']=1-row[family+'_tuned']/row[family+'_default']
            row[family+'_beats_both_constants']=row[family+'_tuned']<min(row['zero'],row['mean'])
        row['transformer_gain_vs_pooled_mlp']=1-row['transformer_tuned']/row['pooled_mlp_tuned']
        summary.append(row)
    aggregate={}
    for family in p['families']:
        aggregate[family]=dict(mean_relative_gain_vs_ridge=float(np.mean([r[family+'_gain_vs_ridge'] for r in summary])),
            mice_beating_ridge=sum(r[family+'_tuned']<r['ridge'] for r in summary),
            mice_beating_both_constants=sum(r[family+'_beats_both_constants'] for r in summary),
            mean_relative_gain_vs_default=float(np.mean([r[family+'_gain_vs_default'] for r in summary])),
            epoch0_selected=sum(e==0 for r in summary for e in r[family+'_epochs']))
    aggregate['transformer_vs_pooled_mlp']=dict(mice_won=sum(r['transformer_tuned']<r['pooled_mlp_tuned'] for r in summary),
        mean_relative_gain=float(np.mean([r['transformer_gain_vs_pooled_mlp'] for r in summary])))
    (ROOT/'summary.json').write_text(json.dumps(dict(per_mouse=summary,aggregate=aggregate),indent=2)+'\n')
    with (ROOT/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]))
        writer.writeheader();writer.writerows(summary)
    lines=['# Four-mouse model comparison','',
        'Exploratory development study completed after64 neural fits. Four settings per family, two seeds, four mice. All64 fits and checkpoint/recipe choices finished before later outcomes were scored. Historical inspection of these recordings means this is not independent confirmation.','',
        '## Design','',
        '- Transformer:81,377 parameters; nonlinear activity+ID tokens,16 cross-attention summaries, summary self-attention.',
        '- Pooled MLP:81,025 parameters; exactly matched initial tokenizer, residual shared token MLP, uniform mean pooling, nonlinear head. No attention. This is a comparable-capacity nonlinear control, not a pure attention-only ablation.',
        '- Both receive the same2,048 neurons, eight-bin histories, training normalization,24 epochs, batch order per seed, four learning-rate/weight-decay choices.',
        '- Four recipes:lr0.001/0.0003 crossed with weight decay0.01/0.1. Cosine schedule, gradient clipping1, raw normalized MSE training. No grid extension.',
        '- Select epochs using the earlier selection interval, including epoch0. Choose one recipe per mouse/family by mean selection MSE across both seeds. Ridge lambda is selected on that same interval from0.01/0.1/1/10.',
        '- Later interval uses the earlier training normalization; no refit, no later-label model selection. Old evaluation tails remain excluded.',
        '- Neural results average the errors of the two individual seeds; they are not ensemble predictions.','',
        '## Primary later-period errors','',
        'Lower is better within each row. Different mice have different training-speed normalization, so raw errors should not be averaged across rows.','',
        '| Mouse | Zero speed | Training mean | Selected ridge | Tuned transformer | Tuned pooled MLP |',
        '|---|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['mouse']} | {r['zero']:.6f} | {r['mean']:.6f} | {r['ridge']:.6f} | {r['transformer_tuned']:.6f} | {r['pooled_mlp_tuned']:.6f} |")
    lines+=['','Seed-level later errors (to expose variability hidden by averages):','',
        '| Mouse | Model | Seed10 MSE | Seed11 MSE | Seed10 / seed11 beat own untrained model? |',
        '|---|---|---:|---:|---|']
    for mouse in results['rows']:
        vals={r['label']:r for r in mouse['results']}
        for family in p['families']:
            trained=[vals[f'{family}_tuned_s{s}']['mse'] for s in p['seeds']]
            untrained=[vals[f'{family}_untrained_s{s}']['mse'] for s in p['seeds']]
            flags=['yes' if a<b else 'no' for a,b in zip(trained,untrained)]
            lines.append(f"| {mouse['mouse']} | {family} | {trained[0]:.6f} | {trained[1]:.6f} | {' / '.join(flags)} |")
    lines+=['','Equal-weight mouse summaries (descriptive, not significance):','', '```json',json.dumps(aggregate,indent=2),'```','',
        '## Did tuning improve the inherited recipe?','',
        'The default is recipe0:lr0.001,weight decay0.01, with checkpoints selected on the same earlier interval.','',
        '| Mouse | Transformer default → tuned | Pooled MLP default → tuned | Transformer selected epochs | Pooled MLP selected epochs |',
        '|---|---:|---:|---|---|']
    for r in summary:
        lines.append(f"| {r['mouse']} | {r['transformer_default']:.6f} → {r['transformer_tuned']:.6f} | {r['pooled_mlp_default']:.6f} → {r['pooled_mlp_tuned']:.6f} | {r['transformer_epochs']} | {r['pooled_mlp_epochs']} |")
    lines+=['','Selected recipes and ridge lambdas:','', '```json',json.dumps(locked['choices'],indent=2),'```','',
        '## Prespecified range guard','',
        'Inference-only bounds come from training feature minima/maxima. This secondary probe was declared before these fits, but was motivated by the previous inspected MP032 failure. It was not used to select a checkpoint/recipe and is not a validated preprocessing improvement.','',
        '| Mouse | Ridge original → guard | Transformer original → guard | Pooled MLP original → guard |',
        '|---|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['mouse']} | {r['ridge']:.6f} → {r['ridge_guard']:.6f} | {r['transformer_tuned']:.6f} → {r['transformer_guard']:.6f} | {r['pooled_mlp_tuned']:.6f} → {r['pooled_mlp_guard']:.6f} |")
    lines+=['','## Quiet and moving periods','',
        'The threshold is each mouse’s training75th-percentile speed. “Quiet” means at or below that relative threshold, not verified physical rest. Both subsets and sample counts are reported; whole-period MSE remains primary.','',
        '| Mouse | Quiet / moving samples | Model | Quiet MSE | Moving MSE |',
        '|---|---|---|---:|---:|']
    for mouse in results['rows']:
        vals={r['label']:r for r in mouse['results']}
        for family in ['zero','ridge','transformer','pooled_mlp']:
            parts=[vals[family]] if family in ['zero','ridge'] else [vals[f'{family}_tuned_s{s}'] for s in p['seeds']]
            means=[]
            for key in ['quiet_mse','moving_mse']:
                means.append('n/a' if parts[0][key] is None else f"{np.mean([v[key] for v in parts]):.6f}")
            lines.append(f"| {mouse['mouse']} | {mouse['quiet_n']} / {mouse['moving_n']} | {family} | {means[0]} | {means[1]} |")
    lines+=['','## Limits and checks','',
        'This tests two finite model configurations and a bounded training search. It does not exhaust transformer architectures, temporal attention, data coverage, optimization or preprocessing. The nonlinear control differs in more than attention, so a difference cannot be attributed uniquely to attention.',
        '', 'There are four animals and two technical seeds; later periods remain historically examined development data. No confirmatory p-values or publication claim. Coordinates, grouping and generation are outside this study.',
        '', 'New control passed initial-tokenizer matching, parameter-count matching within1%, joint cell/ID permutation invariance, and gradient/update checks. Every selected checkpoint reproduced full selection predictions. All epoch selection errors and locked choices were independently checked. Batch orders match across families/recipes. Later saved MSE was independently recomputed. Application and inherited artifacts remain unchanged.',
        '', 'NumPy BLAS emitted warnings during ridge prediction despite finite outputs. A separate [numerical audit](numerical_audit.json) reproduced all new ridge and guarded-ridge predictions using Torch float64 and a non-BLAS einsum calculation; maximum discrepancy6.7e-15.',
        '', 'See [interpretation](ASSESSMENT.md), [protocol](protocol.json), [selection lock](selection_lock.json), [audit](audit.json), [summary](summary.json), [comparison chart](comparison.png), and [full later traces](later_traces.png).']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    fig,ax=plt.subplots(figsize=(10,5))
    xx=np.arange(4);width=.19
    for i,(key,label,color) in enumerate([('zero','Zero speed','gray'),('ridge','Selected ridge','tab:green'),('transformer_tuned','Tuned transformer','tab:blue'),('pooled_mlp_tuned','Tuned pooled MLP','tab:orange')]):
        ax.bar(xx+(i-1.5)*width,[r[key]/r['ridge'] for r in summary],width,label=label,color=color)
    ax.axhline(1,color='black',linewidth=.7);ax.set_yscale('log')
    ax.set_ylim(.2,8);ax.set_yticks([.25,.5,1,2,4,8],['0.25','0.5','1','2','4','8'])
    ax.set_xticks(xx,[r['mouse'] for r in summary]);ax.set_ylabel('Later MSE / selected ridge MSE (log scale)')
    ax.set_title('Earlier-interval selection; later development evaluation')
    ax.legend();fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(4,1,figsize=(13,12))
    for ax,r in zip(axes,summary):
        stats=read(r['mouse']+'/metadata.json')
        with np.load(ROOT/r['mouse']/'later_predictions.npz') as saved:
            actual=saved['target']*stats['speed_std']+stats['speed_mean']
            ax.plot(actual,color='black',linewidth=1,label='Observed')
            for key,label,color in [('ridge','Ridge','tab:green'),('transformer_tuned_s10','Transformer seed10','tab:blue'),('transformer_tuned_s11','Transformer seed11','tab:cyan'),('pooled_mlp_tuned_s10','Pooled MLP seed10','tab:orange')]:
                ax.plot(np.maximum(saved[key]*stats['speed_std']+stats['speed_mean'],0),alpha=.7,linewidth=.8,label=label,color=color)
        ax.set_title(r['mouse']);ax.set_ylabel('Supplied speed units')
    axes[0].legend(ncol=3);axes[-1].set_xlabel('Later development sample')
    fig.tight_layout();fig.savefig(ROOT/'later_traces.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(per_mouse=summary,aggregate=aggregate),indent=2))


if __name__=='__main__':
    main()
