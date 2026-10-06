"""Report the broad screen, fixed refinement, and descriptive paired uncertainty."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
FAIR=ROOT.parent/'2026-10-03_fair_comparison'


def read(path):return json.loads(path.read_text())


def bootstrap(finalist,control,results,seeds):
    data=[]
    for mouse in results['rows']:
        stats=read(FAIR/mouse['mouse']/'metadata.json');lower=-stats['speed_mean']/stats['speed_std']
        with np.load(ROOT/mouse['mouse']/'later_predictions.npz') as saved:
            y=saved['target'];a=np.stack([(np.maximum(saved[f'{finalist}_s{s}'],lower)-y)**2 for s in seeds]);b=np.stack([(np.maximum(saved[f'{control}_s{s}'],lower)-y)**2 for s in seeds]);base=(np.maximum(saved['strong_baseline'],lower)-y)**2
        data.append((a,b,base))
    rng=np.random.default_rng(59102);draws=np.empty((4000,2))
    for iteration in range(4000):
        gains=[]
        for mouse in rng.integers(0,4,4):
            a,b,base=data[mouse];n=a.shape[1];chosen_seeds=rng.integers(0,3,3)
            starts=rng.integers(0,n,(n+99)//100);indices=((starts[:,None]+np.arange(100)[None])%n).ravel()[:n]
            ma=float(a[chosen_seeds][:,indices].mean());mb=float(b[chosen_seeds][:,indices].mean());mr=float(base[indices].mean())
            gains.append([1-ma/max(mr,1e-15),1-ma/max(mb,1e-15)])
        draws[iteration]=np.mean(gains,axis=0)
    return dict(resamples=4000,block_bins=100,interval_percent=98.75,
                versus_baseline=np.quantile(draws[:,0],[.00625,.99375]).tolist(),
                versus_control=np.quantile(draws[:,1],[.00625,.99375]).tolist(),
                interpretation='Descriptive conditional intervals with4mice; historical data reuse and architecture selection prevent confirmatory significance claims')


def main():
    p=read(ROOT/'protocol.json');short=read(ROOT/'shortlist.json');results=read(ROOT/'results.json');lock=read(ROOT/'final_lock.json');audit=read(ROOT/'completion_audit.json')
    rows=[]
    for mouse in results['rows']:
        v={r['label']:r for r in mouse['results']}
        row=dict(mouse=mouse['mouse'],strong_kind=mouse['strong_kind'],ridge=v['ridge']['mse'],kernel=v['kernel']['mse'],baseline=v['strong_baseline']['mse'],zero=v['zero']['mse'],mean=v['mean']['mse'])
        for family in short['refine_families']:
            row[family]=float(np.mean([v[f'{family}_s{s}']['mse'] for s in p['refinement_seeds']]))
            row[family+'_seeds']=[v[f'{family}_s{s}']['mse'] for s in p['refinement_seeds']]
            row[family+'_epochs']=[v[f'{family}_s{s}']['epoch'] for s in p['refinement_seeds']]
            row[family+'_ensemble']=v[f'{family}_ensemble']['mse']
            row[family+'_learned_wins']=sum(v[f'{family}_s{s}']['mse']<v[f'{family}_initial_s{s}']['mse'] for s in p['refinement_seeds'])
        rows.append(row)
    summary={}
    for family in short['finalists']:
        control=short['controls'][family]
        gains=[1-r[family]/r['baseline'] for r in rows];cgains=[1-r[family]/r[control] for r in rows]
        s=dict(control=control,mean_relative_gain_vs_baseline=float(np.mean(gains)),mean_relative_gain_vs_control=float(np.mean(cgains)),
               mouse_wins_vs_baseline=sum(g>0 for g in gains),mouse_wins_vs_control=sum(g>0 for g in cgains),
               seed_wins_vs_baseline=sum(v<r['baseline'] for r in rows for v in r[family+'_seeds']),
               worst_mouse_harm=float(max(-g for g in gains)),learned_seed_wins=sum(r[family+'_learned_wins'] for r in rows),
               epoch0_selections=sum(e==0 for r in rows for e in r[family+'_epochs']),
               leave_one_mouse_out_gain_vs_baseline=[float(np.mean([g for j,g in enumerate(gains) if j!=i])) for i in range(4)],
               uncertainty=bootstrap(family,control,results,p['refinement_seeds']))
        s['gate']=bool(s['mean_relative_gain_vs_baseline']>=.05 and s['mean_relative_gain_vs_control']>=.05 and s['mouse_wins_vs_baseline']>=3 and s['mouse_wins_vs_control']>=3 and s['seed_wins_vs_baseline']>=8 and s['worst_mouse_harm']<=.25 and s['learned_seed_wins']>=8)
        summary[family]=s
    (ROOT/'summary.json').write_text(json.dumps(dict(per_mouse=rows,finalists=summary),indent=2)+'\n')
    with (ROOT/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    counts=read(ROOT/'selfcheck.json')['models'];params={r['kind']:r['parameters'] for r in counts}
    lines=['# Broad screen followed by fixed refinement','',
        'All listed design directions were represented by concrete prototypes or ensemble comparisons. This is exploratory development on previously inspected recordings; later refinement is separate from screening within this study but is not fresh independent confirmation.','',
        f"Completed budget: {audit['counts']['new_fits']} distinct new neural fits, {audit['counts']['supervised_epochs']} supervised model-epochs and {audit['counts']['reconstruction_epochs']} reconstruction model-epochs. Continued screening runs count once. Refinement additionally reuses16 completed reference fits. Kernel regression evaluated9 settings per mouse.",'',
        '## Screening','',
        'Ten individual variants on four mice plus one shared four-session fit:41 screening fits,seed10,12 supervised epochs. Masked variants also receive8 training-only reconstruction epochs. Existing default-model predictions supply reference scores without repeated training. Graphs and PCA/kernel preprocessing use training examples only.','',
        '| Rank | Prototype | Parameters | Mean selection MSE / selected baseline | Mice selecting trained checkpoint | Promoted |',
        '|---:|---|---:|---:|---:|---|']
    for i,r in enumerate(short['ranking']):
        lines.append(f"| {i+1} | {r['kind']} | {params[r['kind']]} | {r['mean_ratio']:.4f} | {r['trained_mice']}/4 | {'yes' if r['kind'] in short['finalists'] else 'no'} |")
    lines+=['','The selected baseline per mouse is ridge or PCA/RBF kernel ridge, chosen by earlier-selection MSE. Ranking and the maximum of two finalists were fixed before any current later scoring. This small screen can miss slower learners; nonpromotion is not a universal negative result.','',
        '## Scope of the prototypes','']
    for name,description in p['candidate_details'].items():lines.append(f'- **{name}:** {description}')
    lines+=['- **Ensembles:** fixed two-seed transformer averages in screening, then fixed three-seed raw-prediction averages for every refined family. No learned model-selection gate.',
        '', 'The shared model uses labeled training prefixes from four known sessions, with session-specific IDs and per-mouse checkpoint selection. It tests whether pooling training data helps those sessions, not zero-shot transfer to a new mouse. Recurrent/graph/pretraining prototypes are not full LFADS, STNDT or NDT2 replications. Widths, parameter counts and pretraining budgets differ; matched controls in refinement narrow interpretation but do not erase every architectural difference.',
        '', '## Later refinement','',
        'Finalists and preassigned controls run24 supervised epochs with seeds10/11/12. Compatible seed10 screen runs continue from exact optimizer/scheduler/random-generator state. Completed24epoch default reference seeds10/11 are reused; only seed12 references are newly fit. Earlier selection still chooses checkpoints. No later labels choose a family, setting, checkpoint or replacement finalist.',
        '', '| Mouse | Selected baseline | Ridge MSE | Kernel MSE | Zero MSE | '+' | '.join(short['refine_families'])+' |',
        '|---|---|---:|---:|---:|'+ '---:|'*len(short['refine_families'])]
    for r in rows:
        lines.append(f"| {r['mouse']} | {r['strong_kind']} | {r['ridge']:.6f} | {r['kernel']:.6f} | {r['zero']:.6f} | "+' | '.join(f"{r[f]:.6f}" for f in short['refine_families'])+' |')
    lines+=['','Neural scores above average individual-seed errors, not ensemble predictions. Compare raw normalized MSE within a mouse; cross-mouse summaries average relative errors.','',
        'Frozen exploratory gate:>=5% mean relative gain versus both selected baseline and assigned control;>=3/4 mouse wins versus each;>=8/12 seed wins against baseline;no mouse>25% worse than baseline;>=8/12 beats own epoch0. Gate outcomes do not establish significance.','',
        '```json',json.dumps(summary,indent=2),'```','',
        'Intervals resample mice, seeds and paired circular100-bin blocks4000 times. The98.75% intervals are descriptive and conditional on this inspected cohort/pipeline; neither the small biological sample nor historical tuning disappears through bootstrapping. Leave-one-mouse-out gains expose dependence on one animal.','',
        '## Seed variability and ensembles','',
        '| Mouse | Model | Seed10 /11 /12 MSE | Selected epochs | Three-seed ensemble MSE |',
        '|---|---|---|---|---:|']
    for r in rows:
        for family in short['refine_families']:
            values=' / '.join(f'{v:.6f}' for v in r[family+'_seeds'])
            lines.append(f"| {r['mouse']} | {family} | {values} | {r[family+'_epochs']} | {r[family+'_ensemble']:.6f} |")
    lines+=['','Epoch0 is retained as a failure indicator. For masked pretraining it means after reconstruction training but before behavioral fine-tuning. Ensemble weights are fixed and are not tuned on later outcomes.','',
        '## Quiet and moving periods','',
        'Threshold is each mouse’s training75th-percentile speed, not verified physical rest. All examples remain in primary MSE.','',
        '| Mouse | Quiet / moving samples | Model | Quiet MSE | Moving MSE |','|---|---|---|---:|---:|']
    for mouse in results['rows']:
        v={r['label']:r for r in mouse['results']}
        for family in ['strong_baseline','zero']+short['finalists']:
            parts=[v[family]] if family in v else [v[f'{family}_s{s}'] for s in p['refinement_seeds']]
            values=['n/a' if parts[0][key] is None else f"{np.mean([r[key] for r in parts]):.6f}" for key in ['quiet_mse','moving_mse']]
            lines.append(f"| {mouse['mouse']} | {mouse['quiet_n']} / {mouse['moving_n']} | {family} | {values[0]} | {values[1]} |")
    lines+=['','## Evidence and limits','',
        'Model shape/gradient/update/checkpoint checks and exact optimizer/scheduler/RNG-resume tests passed before screening. Training-derived graphs exclude self edges. Kernel fits passed numerical solve and independent prediction checks. All selected model checkpoints reproduce earlier predictions. Every final selection was locked before later scoring; selection and later errors were independently recomputed. Application and inherited artifacts remain unchanged.',
        '', 'One-seed12epoch screening is deliberately coarse. Its omissions cannot establish that an entire model family is ineffective. Kernel bandwidth/regularization grids are finite. Four previously examined mice and overlapping time windows do not supply independent confirmation; more seeds quantify optimizer variability, not more biological replication.',
        '', 'See [assessment](ASSESSMENT.md), [protocol](protocol.json), [shortlist](shortlist.json), [final selection lock](final_lock.json), [evaluation audit](audit.json), [completion audit](completion_audit.json), [numeric summary](summary.json), and [comparison chart](comparison.png).']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    fig,ax=plt.subplots(figsize=(12,5));xx=np.arange(4);families=['plain','pooled_mlp']+short['finalists']+list(short['controls'].values());families=list(dict.fromkeys(families));width=.8/len(families)
    for i,family in enumerate(families):ax.bar(xx+(i-(len(families)-1)/2)*width,[r[family]/r['baseline'] for r in rows],width,label=family)
    ax.axhline(1,color='black',linewidth=.8);ax.set_xticks(xx,[r['mouse'] for r in rows]);ax.set_yscale('log');ax.set_yticks([.25,.5,1,2,4,8],labels=['0.25','0.5','1','2','4','8']);ax.set_ylabel('Later MSE / selected baseline MSE (log scale)');ax.set_title('Three-seed refinement after broad screening');ax.legend(ncol=3);fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(per_mouse=rows,finalists=summary),indent=2))


if __name__=='__main__':main()
