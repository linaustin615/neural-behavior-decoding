"""Report forecasting and the fixed two-by-two downstream comparison separately."""
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
ARMS=['attention_scratch','attention_pretrained','mixer_scratch','mixer_pretrained','pooled_mlp']
CONTROLS=['attention_scratch','mixer_pretrained','pooled_mlp','ridge']


def read(path):return json.loads(path.read_text())


def uncertainty(results):
    data=[]
    for row in results['rows']:
        meta=read(FAIR/row['mouse']/'metadata.json');lower=-meta['speed_mean']/meta['speed_std']
        with np.load(ROOT/row['mouse']/'later_predictions.npz') as saved:
            errors={kind:np.stack([(np.maximum(saved[f'{kind}_s{s}'],lower)-saved['target'])**2 for s in SEEDS]) for kind in ARMS}
            errors['ridge']=np.repeat(((np.maximum(saved['ridge'],lower)-saved['target'])**2)[None],3,axis=0)
        data.append(errors)
    rng=np.random.default_rng(81204);draws=np.empty((4000,4))
    for i in range(4000):
        values=[]
        for mouse in rng.integers(0,4,4):
            d=data[mouse];n=d['ridge'].shape[1];seeds=rng.integers(0,3,3);starts=rng.integers(0,n,(n+99)//100)
            indices=((starts[:,None]+np.arange(100)[None])%n).ravel()[:n]
            errors={k:float(v[seeds][:,indices].mean()) for k,v in d.items()}
            values.append([1-errors['attention_pretrained']/max(errors[k],1e-15) for k in CONTROLS])
        draws[i]=np.mean(values,axis=0)
    return dict(resamples=4000,block_bins=100,interval_percent=98.75,contrasts={k:np.quantile(draws[:,j],[.00625,.99375]).tolist() for j,k in enumerate(CONTROLS)},interpretation='Conditional descriptive intervals on four historically inspected mice, not confirmatory significance')


def main():
    results=read(ROOT/'results.json');protocol=read(ROOT/'protocol.json');rows=[];forecast=[]
    for row in results['rows']:
        v={r['label']:r for r in row['speed']};a=dict(mouse=row['mouse'],ridge=v['ridge']['mse'],zero=v['zero']['mse'],mean=v['mean']['mse'],old_regression=v['old_regression']['mse'],old_mlp=float(np.mean([v[f'old_mlp_s{s}']['mse'] for s in SEEDS])))
        for arm in ARMS:
            a[arm+'_seeds']=[v[f'{arm}_s{s}']['mse'] for s in SEEDS];a[arm]=float(np.mean(a[arm+'_seeds']))
            a[arm+'_epochs']=[v[f'{arm}_s{s}']['epoch'] for s in SEEDS];a[arm+'_ensemble']=v[f'{arm}_ensemble']['mse']
            a[arm+'_initial_wins']=sum(v[f'{arm}_s{s}']['mse']<v[f'{arm}_initial_s{s}']['mse'] for s in SEEDS)
        rows.append(a)
        f={r['label']:r for r in row['forecast']};b=dict(mouse=row['mouse'],baseline=f['strong_baseline']['mse'],baseline_choice=row['forecast_baseline'],mean=f['mean']['mse'],persistence=f['persistence']['mse'])
        for kind in ['attention','mixer']:
            b[kind+'_seeds']=[f[f'{kind}_s{s}']['mse'] for s in SEEDS];b[kind]=float(np.mean(b[kind+'_seeds']));b[kind+'_epochs']=[f[f'{kind}_s{s}']['epoch'] for s in SEEDS]
        forecast.append(b)
    contrasts={}
    for control in CONTROLS:
        gains=[1-r['attention_pretrained']/r[control] for r in rows]
        contrasts[control]=dict(mean_relative_gain=float(np.mean(gains)),mouse_wins=sum(g>0 for g in gains),worst_mouse_harm=float(max(-g for g in gains)),leave_one_mouse_out=[float(np.mean([g for j,g in enumerate(gains) if j!=i])) for i in range(4)])
        if control!='ridge':contrasts[control]['paired_seed_wins']=sum(a<b for r in rows for a,b in zip(r['attention_pretrained_seeds'],r[control+'_seeds']))
    gate=all(c['mean_relative_gain']>=.05 and c['mouse_wins']>=3 and c.get('paired_seed_wins',8)>=8 for c in contrasts.values()) and contrasts['pooled_mlp']['worst_mouse_harm']<=.25
    pretraining={kind:dict(mean_relative_speed_gain=float(np.mean([1-r[kind+'_pretrained']/r[kind+'_scratch'] for r in rows])),mouse_wins=sum(r[kind+'_pretrained']<r[kind+'_scratch'] for r in rows)) for kind in ['attention','mixer']}
    interaction=float(np.mean([(1-r['attention_pretrained']/r['attention_scratch'])-(1-r['mixer_pretrained']/r['mixer_scratch']) for r in rows]))
    forecast_summary={}
    for kind in ['attention','mixer']:
        gains=[1-r[kind]/r['baseline'] for r in forecast]
        forecast_summary[kind]=dict(mean_relative_gain=float(np.mean(gains)),mouse_wins=sum(g>0 for g in gains),gate=bool(np.mean(gains)>=.05 and sum(g>0 for g in gains)>=3))
    summary=dict(speed=rows,forecast=forecast,attention_pretrained_contrasts=contrasts,speed_gate=bool(gate),pretraining_effects=pretraining,attention_minus_mixer_pretraining_gain=interaction,forecast_summary=forecast_summary,uncertainty=uncertainty(results))
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (ROOT/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Dynamics pretraining baseline','',
        'Completed 24 neural forecasting pretraining fits, 48 sequence speed-decoding fits and 12 matched pooled-MLP fits. Each fit ran 24 epochs; three seeds and four mice. This is a reusable small literature-informed baseline, not a novel architecture or independent confirmation.','',
        '## Representation and controls','',protocol['data'],'',protocol['representation'],'',protocol['models'],'',
        'The attention/mixer sequence models have 12,181/12,239 parameters; the pooled-MLP comparator has 19,297. The mixer has no attention. Both sequence models preserve 128-neuron × 8-patch representations through temporal and population blocks. A synthetic intervention verified exactly that later input patches cannot alter earlier representations. Shared patch/ID embeddings and forecast/speed readout initializations match between sequence families.','',
        'Pretraining predicts the next four standardized activity bins from 32 past bins. Under the inherited nominal 1.2-second binning, this is about 38 seconds of context and 4.8 seconds of future activity; these time scales differ from many published benchmarks. It uses no running-speed labels and never trains on selection/later examples. Earlier neural MSE selects its checkpoint before downstream training. The same supervised head is then fine-tuned together with the encoder for 24 epochs in both pretrained and scratch conditions. No frozen-backbone probe was added in this initial baseline.','',
        'Pretrained models receive 24 additional neural-training epochs. The with/without comparison measures that entire recipe; it does not isolate objective choice from extra optimization. Attention-pretrained versus mixer-pretrained has the same stage budgets. Parameter/epoch matching does not imply equal computation.','',
        '## Neural forecasting: separate endpoint','',
        '| Mouse | Selected simple baseline | Baseline MSE | Mean MSE | Persistence MSE | Attention MSE | Mixer MSE |','|---|---|---:|---:|---:|---:|---:|']
    for r in forecast:lines.append(f"| {r['mouse']} | {r['baseline_choice']['kind']} {r['baseline_choice'].get('lam','')} | {r['baseline']:.6f} | {r['mean']:.6f} | {r['persistence']:.6f} | {r['attention']:.6f} | {r['mixer']:.6f} |")
    lines+=['','Neural MSE averages standardized activity errors across neurons and all four forecast bins. Baseline family/regularization is selected on the earlier interval, not later outcomes. Simple controls include per-neuron AR32 and training-PCA16 population AR32. Forecast utility gate requires >=5% mean relative gain and >=3/4 mouse wins against the selected simple baseline. A forecasting win alone is not evidence of improved running-speed decoding.','',
        '```json',json.dumps(forecast_summary,indent=2),'```','',
        '## Running speed: primary endpoint','',
        '| Mouse | Matched ridge | Zero | Attention scratch | Attention pretrained | Mixer scratch | Mixer pretrained | Matched pooled MLP |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['mouse']} | {r['ridge']:.6f} | {r['zero']:.6f} | "+' | '.join(f'{r[k]:.6f}' for k in ARMS)+' |')
    lines+=['','Entries are bounded normalized speed MSE averaged across individual seed errors. Aggregate comparisons average within-mouse relative gains, equally weighting mice.','',protocol['gate'],'',
        '```json',json.dumps(dict(gate=gate,contrasts=contrasts,pretraining_effects=pretraining,attention_minus_mixer_pretraining_gain=interaction,uncertainty=summary['uncertainty']),indent=2),'```','',
        'The difference in pretraining effects is descriptive: positive means a larger relative benefit from the pretraining recipe for attention than for the mixer. No retrospective architecture or epoch replacement was made. Bootstrap resamples mice, seeds and paired circular 100-bin blocks 4,000 times; 98.75% intervals for four primary contrasts remain descriptive because the cohort and previous analyses are historically reused.','',
        '## Individual seeds and ensembles','',
        '| Mouse | Arm | Seed10 /11 /12 MSE | Selected epochs | Wins over own epoch0 | Fixed ensemble MSE |','|---|---|---|---|---:|---:|']
    for r in rows:
        for arm in ARMS:lines.append(f"| {r['mouse']} | {arm} | "+' / '.join(f'{v:.6f}' for v in r[arm+'_seeds'])+f" | {r[arm+'_epochs']} | {r[arm+'_initial_wins']}/3 | {r[arm+'_ensemble']:.6f} |")
    lines+=['','For pretrained arms, epoch 0 is after neural pretraining and before speed fine-tuning; it is not an entirely untrained model. Ensemble predictions are equal-weight raw averages; their scores do not replace the primary individual-seed gate.','',
        '## Earlier full-population references','',
        '| Mouse | Previous selected regression | Previous normalized MLP |','|---|---:|---:|']
    for r in rows:lines.append(f"| {r['mouse']} | {r['old_regression']:.6f} | {r['old_mlp']:.6f} |")
    lines+=['','These predictions are aligned exactly to the new target endpoints but use 2,048 neurons and 8 bins, versus 128 neurons and 32 bins here, and have slightly more training endpoints. They are contextual references, not equal-input primary controls. Historical scores should not be compared without this target alignment.','',
        '## Verification and scope','',
        'New architecture shape/gradient/update/reload checks, exact patch-level causality,matching common initialization, and synthetic context/forecast indexing passed. Cached overlapping histories were checked when constructing the new representation. Ridge/AR fits passed numerical optimality and independent predictions. All 24 pretraining choices were locked before speed fitting;all 60 speed choices were locked before current later scoring. Selected checkpoints reload exactly; all 1,500 speed selection predictions and 24 selected neural predictions were checked. Batch orders match within each training stage. Later target alignment is exact and speed metrics were independently recomputed. Application and inherited artifacts are unchanged.','',
        'This compact one-block design and 128-cell subset are feasibility choices. Models are trained separately per mouse; this baseline does not test cross-session pooling or transfer. It is not a reproduction of the size, data volume, training schedule or full objectives of CAPT, CalM or POYO+. Stringer signals here are inherited binned deconvolved activity; they are not raw fluorescence or discrete spike counts. The design uses continuous MSE rather than assuming a Poisson count likelihood. A failure cannot rule out larger or differently trained versions; a success still needs independent confirmation. No old evaluation tails, new datasets, application edits or publication were used.','',
        'Literature design references: [CAPT continuous-patch forecasting](https://arxiv.org/html/2607.23258v2), [POYO+ calcium decoding](https://proceedings.iclr.cc/paper_files/paper/2025/file/953390c834451505703c9da45de634d8-Paper-Conference.pdf), and [NDT2 neural pretraining](https://proceedings.neurips.cc/paper_files/paper/2023/file/fe51de4e7baf52e743b679e3bdba7905-Paper-Conference.pdf).','',
        'See [assessment](ASSESSMENT.md), [protocol](protocol.json), [models](models.py), [runner](run.py), [audit](audit.json), [numeric summary](summary.json), and [chart](comparison.png).']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(13,5));xx=np.arange(4);width=.15
    for i,arm in enumerate(ARMS):axes[0].bar(xx+(i-2)*width,[r[arm]/r['ridge'] for r in rows],width,label=arm)
    axes[0].axhline(1,color='black',linewidth=.8);axes[0].set_xticks(xx,[r['mouse'] for r in rows]);axes[0].set_ylabel('Speed MSE / matched ridge MSE');axes[0].set_title('Three-seed running-speed decoding');axes[0].legend(fontsize=7)
    for i,kind in enumerate(['attention','mixer']):axes[1].bar(xx+(i-.5)*.32,[r[kind]/r['baseline'] for r in forecast],.32,label=kind)
    axes[1].axhline(1,color='black',linewidth=.8);axes[1].set_xticks(xx,[r['mouse'] for r in forecast]);axes[1].set_ylabel('Forecast MSE / selected simple baseline MSE');axes[1].set_title('Next-block neural forecasting');axes[1].legend();fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
