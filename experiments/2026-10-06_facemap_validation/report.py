"""Render the bounded validation outcome without changing its criteria."""
import numpy as np
from common import ROOT, PLAN, MICE, read, save, now, digest

LABELS={'optimized_attention':'Selected transformer','optimized_population_mlp':'Selected MLP','default_attention':'Smaller transformer','ridge':'Tuned ridge'}


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s=read(ROOT/'summary.json')
    a=read(ROOT/'audit.json')
    rows=read(ROOT/'test_metrics.json')['rows']
    primary=[r for r in s['contrasts'] if r['regime']=='independent']
    lines=['# Separate Stringer/Facemap validation','',
        'Completed the fixed comparison on seven visual-cortex mouse IDs, using one recording per mouse. The primary models were fitted independently per mouse; a separately trained shared model is a descriptive secondary check.',
        '', '72 neural fits and 168 ridge solutions; three fixed training seeds (401–403), 512 training-selected cells per mouse, at most 4096 evenly spaced training targets, all eligible validation/test targets. The old architecture search was not reopened.', '',
        '| Transformer compared with | Mean relative MSE reduction | Mouse wins | Seed wins | Holm-adjusted sign p | Practical gate | Validated advantage |',
        '|---|---:|---:|---:|---:|---|---|']
    for r in primary:
        lines.append(f"| {LABELS[r['comparator']]} | {100*r['mean_relative_mse_gain']:+.2f}% | {r['mouse_wins']}/7 | {r['individual_seed_wins']}/21 | {r['holm_p']:.5f} | {'PASS' if r['practical_pass'] else 'FAIL'} | {'YES' if r['validated_transformer_advantage'] else 'NO'} |")
    lines+=['','Positive reduction favors the transformer; negative values mean higher error. Average seed errors within each mouse before computing relative effects, then weight mice equally. These are not ensemble errors. Seeds and frames do not increase the biological sample size.',
        '', 'Two-sided exact sign tests test the direction of mouse-level differences, with Holm correction across all three prespecified contrasts. The shared models have no inferential p-values. Statistical and practical criteria are separate; neither is relaxed after outcomes.',
        '', '| Mouse | Selected transformer R² | Selected MLP R² | Smaller transformer R² | Ridge R² |', '|---|---:|---:|---:|---:|']
    for m in MICE:
        values=[]
        for role in LABELS:
            subset=[r for r in rows if r['mouse']==m and r['role']==role and r['regime']==('control' if role=='ridge' else 'independent')]
            values.append(float(np.mean([r['r2'] for r in subset])) if all(r['r2'] is not None for r in subset) else float('nan'))
        lines.append('| '+m+' | '+' | '.join(f'{v:.3f}' for v in values)+' |')
    lines+=['','Shared-fit secondary comparisons:']
    for r in s['contrasts']:
        if r['regime']=='shared':
            lines.append(f"- Versus {LABELS[r['comparator']]}: {100*r['mean_relative_mse_gain']:+.2f}% mean relative MSE reduction, {r['mouse_wins']}/7 mouse means, {r['individual_seed_wins']}/21 seed comparisons; practical gate {'PASS' if r['practical_pass'] else 'FAIL'}.")
    lines+=['','Learning versus training-mean control:']
    for regime,v in s['learning'].items():
        for role,q in v.items():
            lines.append(f"- {regime}, {LABELS[role]}: {q['mouse_means_beating_training_mean']}/7 mouse means and {q['individuals_beating_training_mean']}/21 individual fits beat the untrained predictor.")
    lines+=['','Method and limits:',
        '- Chronological 60/20/20 partitions, 64-frame gaps and common 63-frame warmup. Normalization and neuron eligibility use training only. Validation chooses checkpoints, including epoch0; every choice was locked before test prediction.',
        '- Native-frame activity predicts concurrent absolute running values. Physical speed/time units and original sensor alignment are not independently reconstructed; this is not forecasting. No camera inputs, past behavior, coordinates or outcome-selected lags.',
        '- A pre-fit amendment allows only a one-frame activity/run length mismatch and uses their common index prefix, following the authors’ direct neural-index lookup. Original protocol and failure logs are retained. Per-recording discrepancies are recorded in acquired schema files.',
        '- The selected architectures and optimizer recipes came entirely from the closed older four-mouse study. New fits learn new neuron-specific read-ins; this is not zero-shot transfer. Shared weights couple animals in the secondary comparison.',
        '- The Facemap v2 release updates deconvolution and some motion correction, so this is a new recording/context validation, not identical-preprocessing replication. IDs/dates and reported ages support separation from the old MP cohort; no direct author identity confirmation was obtained.',
        '- Seven animals from one lab, one fixed neuron panel each, three seeds and a fixed training-example cap limit breadth. The independently tuned transformer/MLP recipes differ in patching and dropout, so the comparison does not isolate attention causally.',
        '- No further model/seed/recording choices were made from these test outcomes. No universal optimality, causal neural-connection, generation or publication claim follows automatically.',
        '', f"Audit passed: {a['independently_recomputed_scalar_errors']} independent scalar-error checks, {a['checkpoint_predictions_checked']} checkpoint predictions, {a['raw_input_alignment_samples']} raw-input samples and {a['paired_batch_order_hashes_checked']} paired training-order checks. {a['optimizer_updates']} updates and {a['training_presentations']} training presentations; {a['selected_epoch_zero']} selected epoch0 checkpoints. Download member CRC/size checks are reused, not rerun; whole-archive publisher MD5 was not verified because only selected members were downloaded.",
        '', 'Artifacts: [protocol](protocol.json), [original protocol](protocol_v1.json), [summary](summary.json), [per-run metrics](per_run.csv), [audit](audit.json), [figure](comparison.png), [evaluation lock](evaluation_lock.json).',
        '', 'Sources: [publisher v2 data](https://janelia.figshare.com/articles/dataset/Facemap_a_framework_for_modeling_neural_activity_based_on_orofacial_tracking/23712957/2), [Facemap paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10774130/), [pinned author indexing](https://github.com/MouseLand/facemap/blob/f8b5b518efcde8b4aae7727283bf017b23641c7f/paper/fig4.py).','']
    (ROOT/'report.md').write_text('\n'.join(lines))
    mlp=primary[0]
    assessment=['# Assessment','',f"On seven separate visual-cortex mouse IDs, the selected transformer has {abs(100*mlp['mean_relative_mse_gain']):.2f}% {'lower' if mlp['mean_relative_mse_gain']>0 else 'higher'} mean relative test MSE than the selected MLP, winning {mlp['mouse_wins']}/7 mouse means (Holm-adjusted sign p={mlp['holm_p']:.5f}). Its prespecified practical gate {'passes' if mlp['practical_pass'] else 'fails'}.",'',
        'Read report.md for every comparator, mouse, learning control and limitation. A failed transformer advantage does not mean the transformer failed to learn; inspect the training-mean comparisons and R² values. Shared-model outcomes are descriptive. No old study or application code was changed.','']
    (ROOT/'ASSESSMENT.md').write_text('\n'.join(assessment))
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True,layout='constrained')
    colors=['#136f93','#df8a19','#4d8062']
    x=np.arange(7)
    for ax,regime in zip(axes,['independent','shared']):
        for k,r in enumerate([q for q in s['contrasts'] if q['regime']==regime]):
            values=[100*q['relative_mse_gain'] for q in r['per_mouse']]
            ax.bar(x+(k-1)*.25,values,width=.24,color=colors[k],label=LABELS[r['comparator']])
        ax.axhline(0,color='#333',lw=.8)
        ax.set_xticks(x,MICE,rotation=35,ha='right')
        ax.set_title('Independent fits (primary)' if regime=='independent' else 'Shared fits (descriptive)')
        ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Transformer MSE reduction versus comparator (%)\npositive favors transformer')
    axes[1].legend(frameon=False,fontsize=9)
    fig.suptitle('Separate Stringer/Facemap recordings · 7 mice · 3 seeds')
    fig.savefig(ROOT/'comparison.png',dpi=180)
    fig.savefig(ROOT/'comparison.pdf')
    plt.close(fig)
    save(ROOT/'manifest.json',dict(utc=now(),sha256={str(p.relative_to(ROOT)):digest(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name not in ['manifest.json','pipeline.log','STATUS.json'] and '__pycache__' not in p.parts}))
