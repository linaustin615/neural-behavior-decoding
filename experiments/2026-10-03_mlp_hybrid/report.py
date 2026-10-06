"""Summarize the locked hybrid experiment without changing any selection."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
FAIR=ROOT.parent/'2026-10-03_fair_comparison'
SEEDS=[10,11,12]
KINDS=['base','attention','extra_mlp']


def read(p):return json.loads(p.read_text())


def intervals(results):
    data=[]
    for row in results['rows']:
        mouse=row['mouse'];meta=read(FAIR/mouse/'metadata.json');lower=-meta['speed_mean']/meta['speed_std']
        with np.load(ROOT/mouse/'later_predictions.npz') as saved:
            data.append({kind:np.stack([(np.maximum(saved[f'{kind}_s{s}'],lower)-saved['target'])**2 for s in SEEDS]) for kind in KINDS})
    rng=np.random.default_rng(590310);draws=np.empty((4000,2))
    for iteration in range(4000):
        gains=[]
        for mouse in rng.integers(0,4,4):
            d=data[mouse];n=d['base'].shape[1];seeds=rng.integers(0,3,3)
            starts=rng.integers(0,n,(n+99)//100);indices=((starts[:,None]+np.arange(100)[None])%n).ravel()[:n]
            errors={kind:float(values[seeds][:,indices].mean()) for kind,values in d.items()}
            gains.append([1-errors['attention']/max(errors[c],1e-15) for c in ['base','extra_mlp']])
        draws[iteration]=np.mean(gains,axis=0)
    return dict(resamples=4000,block_bins=100,interval_percent=97.5,versus_base=np.quantile(draws[:,0],[.0125,.9875]).tolist(),versus_extra_mlp=np.quantile(draws[:,1],[.0125,.9875]).tolist(),interpretation='Descriptive conditional intervals on4previously inspected mice,not confirmatory significance')


def main():
    results=read(ROOT/'results.json');p=read(ROOT/'protocol.json');locked=read(ROOT/'selection_lock.json');audit=read(ROOT/'audit.json');rows=[]
    for row in results['rows']:
        by={v['label']:v for v in row['records']};a=dict(mouse=row['mouse'],regression=by['strong_baseline']['mse'],ridge=by['ridge']['mse'],zero=by['zero']['mse'])
        for kind in KINDS:
            a[kind+'_seeds']=[by[f'{kind}_s{s}']['mse'] for s in SEEDS]
            a[kind]=float(np.mean(a[kind+'_seeds']));a[kind+'_epochs']=[by[f'{kind}_s{s}']['epoch'] for s in SEEDS]
            a[kind+'_ensemble']=by[f'{kind}_ensemble']['mse']
            a[kind+'_initial_wins']=sum(by[f'{kind}_s{s}']['mse']<by[f'initial_s{s}']['mse'] for s in SEEDS)
            if kind!='base':
                a[kind+'_branch_off']=float(np.mean([by[f'{kind}_branch_off_s{s}']['mse'] for s in SEEDS]))
                a[kind+'_branch_helpful_seeds']=sum(by[f'{kind}_s{s}']['mse']<by[f'{kind}_branch_off_s{s}']['mse'] for s in SEEDS)
        rows.append(a)
    summaries={}
    for kind in ['attention','extra_mlp']:
        comparators=['base','extra_mlp'] if kind=='attention' else ['base']
        s={}
        for control in comparators:
            gain=[1-r[kind]/r[control] for r in rows]
            s['versus_'+control]=dict(mean_relative_gain=float(np.mean(gain)),mouse_wins=sum(g>0 for g in gain),paired_seed_wins=sum(a<b for r in rows for a,b in zip(r[kind+'_seeds'],r[control+'_seeds'])),worst_mouse_harm=float(max(-g for g in gain)),leave_one_mouse_out=[float(np.mean([g for j,g in enumerate(gain) if j!=i])) for i in range(4)])
        s['mean_relative_gain_vs_regression']=float(np.mean([1-r[kind]/r['regression'] for r in rows]))
        s['initial_wins']=sum(r[kind+'_initial_wins'] for r in rows);s['epoch0_selections']=sum(e==0 for r in rows for e in r[kind+'_epochs'])
        s['branch_helpful_seeds']=sum(r[kind+'_branch_helpful_seeds'] for r in rows)
        s['ensemble_mean_relative_gain_vs_base_ensemble']=float(np.mean([1-r[kind+'_ensemble']/r['base_ensemble'] for r in rows]))
        summaries[kind]=s
    a=summaries['attention'];a['gate']=bool(all(a['versus_'+c]['mean_relative_gain']>=.05 and a['versus_'+c]['mouse_wins']>=3 and a['versus_'+c]['paired_seed_wins']>=8 for c in ['base','extra_mlp']) and a['versus_base']['worst_mouse_harm']<=.25 and a['mean_relative_gain_vs_regression']>0)
    a['uncertainty']=intervals(results)
    summary=dict(per_mouse=rows,comparisons=summaries)
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (ROOT/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Normalized MLP with a parallel correction branch','',
        'Completed24 new fits:two hybrid variants ×four mice ×three seeds,24epochs each. Twelve exact MLP-alone baselines are reused. This is further exploratory development on historically inspected Stringer recordings,not independent confirmation.','',
        '## Design and checks','',
        p['architecture'],'',p['difference_from_prior'],'',
        'The attention branch has13,313 parameters; the extra-MLP branch has13,301. Total sizes are94,418/94,406 versus81,105 for the base alone. Both hybrids start from exactly the same archived untrained base weights. Their initial predictions exactly match the archived baseline predictions for every mouse/seed. All parameters train jointly for24epochs; the baseline receives the same24epoch budget. Same batch orders do not imply identical dropout masks across architectures.','',
        'Synthetic checks verified finite gradients,branch-body learning after the zero head opens,base updates,cell/ID permutation invariance,and exact checkpoint reload. Every real fit passed its selected-checkpoint reload; all600 new epoch-selection errors were independently checked. All24 checkpoints were locked before new later scoring. Frozen sources,application and inherited artifacts are unchanged.','',
        '## Primary later-period comparison','',
        '| Mouse | Selected regression | Zero speed | MLP alone | MLP + attention | MLP + extra MLP |','|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['mouse']} | {r['regression']:.6f} | {r['zero']:.6f} | {r['base']:.6f} | {r['attention']:.6f} | {r['extra_mlp']:.6f} |")
    lines+=['','Entries are bounded normalized MSE averaged across individual seed errors. Aggregate gains average within-mouse relative gains,equally weighting the four mice. Regression is the earlier-selected ridge/kernel baseline from the broad study; it is not reselected from later outcomes.','',
        'Frozen practical gate: '+p['gate'],'','```json',json.dumps(summaries,indent=2),'```','',
        'The bootstrap resamples mice,seeds and paired circular100-bin blocks4000 times. The97.5% intervals for each of two declared contrasts are descriptive; historical model selection,four biological units and overlapping windows prevent a confirmatory interpretation. Seeds are not extra animals.','',
        '## Seeds and fixed ensembles','',
        '| Mouse | Model | Seed10 /11 /12 MSE | Earlier-selected epochs | Three-seed ensemble MSE |','|---|---|---|---|---:|']
    for r in rows:
        for kind in KINDS:lines.append(f"| {r['mouse']} | {kind} | "+' / '.join(f'{v:.6f}' for v in r[kind+'_seeds'])+f" | {r[kind+'_epochs']} | {r[kind+'_ensemble']:.6f} |")
    lines+=['','Ensemble weights are fixed equal raw-prediction weights. They do not replace the individual-seed primary gate. Epoch0 is the common untrained MLP with zero correction,not a pretrained MLP.','',
        '## Removing the learned correction','',
        '| Mouse | Hybrid | Full model MSE | Correction removed MSE | Seeds helped by correction |','|---|---|---:|---:|---:|']
    for r in rows:
        for kind in ['attention','extra_mlp']:lines.append(f"| {r['mouse']} | {kind} | {r[kind]:.6f} | {r[kind+'_branch_off']:.6f} | {r[kind+'_branch_helpful_seeds']}/3 |")
    lines+=['','Removing a branch after joint training also breaks coadaptation between branches. This diagnostic shows reliance on that branch; it does not independently establish an attention-specific advantage. The separately trained capacity-matched extra-MLP hybrid is the relevant primary control.','',
        '## Quiet and moving subsets','',
        'Threshold is the training75th-percentile speed,not independently verified physical rest. All examples remain in primary MSE.','',
        '| Mouse | Quiet / moving samples | Model | Quiet MSE | Moving MSE |','|---|---|---|---:|---:|']
    for row in results['rows']:
        by={v['label']:v for v in row['records']}
        for kind in ['strong_baseline','zero']+KINDS:
            parts=[by[kind]] if kind in by else [by[f'{kind}_s{s}'] for s in SEEDS]
            vals=['n/a' if parts[0][key] is None else f"{np.mean([r[key] for r in parts]):.6f}" for key in ['quiet_mse','moving_mse']]
            lines.append(f"| {row['mouse']} | {row['quiet_n']} / {row['moving_n']} | {kind} | {vals[0]} | {vals[1]} |")
    lines+=['','## Scope','',
        'This tests one jointly trained hybrid recipe at one fixed budget. It does not test frozen pretrained-MLP augmentation or all possible mixtures. Widths,initialization and optimization remain potential limitations even with matched parameter counts. Equal parameter counts and training epochs do not imply equal computation. No correction penalty or width search was added after outcomes. No raw-file validation or old fitted baselines were repeated. No application edits or publication occurred.','',
        'See [assessment](ASSESSMENT.md), [protocol](protocol.json), [selection lock](selection_lock.json), [audit](audit.json), [numeric summary](summary.json), and [chart](comparison.png).']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    fig,ax=plt.subplots(figsize=(10,5));xx=np.arange(4);width=.24
    for i,kind in enumerate(KINDS):ax.bar(xx+(i-1)*width,[r[kind]/r['base'] for r in rows],width,label=kind)
    ax.axhline(1,color='black',linewidth=.8);ax.set_xticks(xx,[r['mouse'] for r in rows]);ax.set_ylabel('Later MSE / MLP-alone MSE');ax.set_title('Joint hybrid comparison: three seeds per mouse');ax.legend();fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
