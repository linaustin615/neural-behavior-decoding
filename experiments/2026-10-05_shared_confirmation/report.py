"""Report previously unused training-seed confirmation honestly."""
import json
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/neuron_finetuning_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
read=lambda name:json.loads((ROOT/name).read_text())
r,audit,review=read('results.json'),read('audit.json'),read('review.json')
names=dict(native_attention='Original transformer',native_mlp='Original MLP',tuned_attention='Continued transformer',tuned_mlp='Fine-tuned MLP',ridge='Raw ridge')
label=lambda key:' vs '.join(names[x] for x in key.split('_vs_'))
pct=lambda x:f'{100*x:+.1f}%'
passed=lambda x:'PASS' if x else 'FAIL'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])


lines=[
    '# Original shared transformer: unused-seed confirmation',
    f"Combined practical confirmation: **{passed(r['practical_confirmation_passed'])}**. This is training-seed reproducibility on four historically reused mice, not independent animal-level significance.",
    'The preceding fixed fine-tuning and residual-readout studies retained the unchanged transformer. This study asks a different question: whether that strongest existing candidate beats the matched original MLP, an equally fine-tuned MLP and the archived raw ridge on previously unused random training seeds. It does not convert either failed modification gate into a success or claim a novel architecture.',
    '## Frozen design',
    'Six native fits use the exact original shared transformer/MLP, seeds16–18,24epochs/5688updates/179712presentations each. Six continuations use the already selected1e-4/plain recipe,8epochs/1896updates/59904presentations each. Original archived trainer sources are imported unchanged. There is no new architecture, input, loss, tuning grid or neuron panel. Each family has the same added training opportunity; continuation includes the native checkpoint option. Averaged-weight snapshots computed by the inherited runner are ineligible for selection in this study and are never later-scored.',
    'Native and continuation checkpoints are chosen on the original earlier joint bounded-MSE criterion. All12choices lock before current later scoring. Predictions for seeds10–15 and raw ridge are reused exactly. The primary candidate is the native transformer; a favorable continuation result cannot rescue its failure.',
    'New seeds16–18 must beat EACH comparator by at least5%mean relative pair MSE, at least3/4mouse means and8/12individual seed wins. Against EACH comparator, no mouse may have more than10%MSE harm and mean MAE must not worsen. At least8/12individual wins over the initial training-mean predictor are also required. The same requirements apply to all nine seeds, with24/36individual wins. Pooled scores cannot rescue failed new-seed replication. Ridge is a deterministic comparator broadcast across seeds; those repeated comparisons are consistency checks, not independent animals or new regression fits.',
    'Each individual output is bounded at physical zero before averaging every distinct two-seed pair. Average allpair errors within each mouse, then relative effects equally over four mice. The new subset has3pairs per mouse; allnine has36. No seed or pair is selected. Errors use training-standardized speed.',
]
for subset in ['new','all']:
    part=r['subsets'][subset];n=len(part['seeds'])
    lines += [
        '## '+('Previously unused seeds16–18' if subset=='new' else 'All nine seeds10–18'),
        f"Subset practical gate: **{passed(part['primary_pass'])}**. Wins over initial training-mean prediction: {part['initial_wins']}/{4*n}.",
        table(['Comparison','Mean MSE gain','Mouse wins','Single-seed wins','Mean MAE gain','Full contrast'],
            [[label(k),pct(v['mean_mse_gain']),f"{v['mouse_wins']}/4",f"{v['seed_wins']}/{4*n}",pct(v['mean_mae_gain']),passed(v['full_pass'])] for k,v in part['summaries'].items()]),
        table(['Mouse','N','Transformer MSE','MLP MSE','Tuned MLP MSE','Ridge MSE','Transformer R²'],
            [[v['mouse'],v['n']]+[f"{v['scores'][g]['mse']:.6f}" for g in ['native_attention','native_mlp','tuned_mlp','ridge']]+[f"{v['scores']['native_attention']['r2']:.3f}"] for v in part['rows']]),
        table(['Comparison','Single-model mean MSE gain','Leave-one-mouse-out gains'],
            [[label(k),pct(v['single_mean_mse_gain']),', '.join(pct(x) for x in v['leave_one_mouse_out'])] for k,v in part['summaries'].items() if k.startswith('native_attention')]),
    ]
lines += [
    '## Individual new seeds',
    table(['Comparison','Seed','Mean individual MSE gain','Mouse wins'],
        [[label(k),v['seed'],pct(v['mean_mse_gain']),f"{v['mouse_wins']}/4"] for k,s in r['subsets']['new']['summaries'].items() if k.startswith('native_attention') for v in s['individual_seeds']]),
    '## Verification and limits',
    f"All12fits completed their assigned updates/presentations; native and continuation batch orders match across families. Continuation starts reproduce native predictions exactly and preserve the same source weights; selected reloads are exact. Locking checked {audit['selection_scores_checked']} selection metrics. Evaluation produced {audit['new_predictions']:,} new predictions, with exact archived target/prediction alignment and unchanged inputs/models. Analysis independently checked {audit['scalar_errors_checked']} scalar errors. Final review passed {review['aggregate_gate_training_checks']} aggregate/gate/training checks, {review['source_input_hashes']} source/input hashes and {review['selected_artifact_hashes']} selected-artifact hashes. Main application files are unchanged.",
    'The candidate and dataset have already undergone extensive development searches. New initialization seeds test optimization reproducibility conditional on these recordings, not new animals, new sessions or pristine independent validation. No statistical significance, biological connectivity, new architectural invention or unseen-mouse transfer is established. This fixed study ends regardless of its result; no extra seeds, altered gates or new candidate selection are appended.',
    'Artifacts: [assessment](ASSESSMENT.md), [protocol](protocol.json), [selection lock](selection_lock.json), [results](results.json), [audit](audit.json), [review](review.json), [PNG](confirmation.png), [PDF](confirmation.pdf).',
]
(ROOT/'report.md').write_text('\n\n'.join(lines)+'\n')
fig,axes=plt.subplots(1,2,figsize=(12,4.7));keys=['native_attention_vs_native_mlp','native_attention_vs_tuned_mlp','native_attention_vs_ridge']
for offset,subset,name,color in [(-.18,'new','New seeds16–18','#216c9c'),(.18,'all','All nine seeds','#c88b48')]:
    axes[0].bar(np.arange(3)+offset,[100*r['subsets'][subset]['summaries'][k]['mean_mse_gain'] for k in keys],width=.36,label=name,color=color)
axes[0].set(xticks=range(3),xticklabels=['vs original MLP','vs tuned MLP','vs raw ridge'],title='Original transformer mean MSE reduction',ylabel='MSE reduction (%)');axes[0].legend(frameon=False)
for shift,key,name,color in [(-.24,keys[0],'vs MLP','#216c9c'),(0,keys[1],'vs tuned MLP','#619cb8'),(.24,keys[2],'vs ridge','#c88b48')]:
    axes[1].bar(np.arange(4)+shift,[100*v for v in r['subsets']['new']['summaries'][key]['mouse_mse_gains']],width=.24,label=name,color=color)
axes[1].set(xticks=range(4),xticklabels=[v['mouse'] for v in r['subsets']['new']['rows']],title='New-seed effects by mouse',ylabel='MSE reduction (%)');axes[1].legend(frameon=False,fontsize=8)
for ax in axes:ax.axhline(0,color='black',linewidth=.8);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Retained shared transformer: previously unused training seeds',x=.06,ha='left')
fig.text(.06,.02,'Four historically reused mice; new seeds are not new animals. Positive values favor the original transformer.\nAll-nine results cannot rescue a failed new-seed comparison.',fontsize=9)
fig.tight_layout(rect=[0,.11,1,.94])
for suffix in ['png','pdf']:fig.savefig(ROOT/f'confirmation.{suffix}',dpi=160,bbox_inches='tight')
plt.close(fig);print('Confirmation report and figures saved',flush=True)
