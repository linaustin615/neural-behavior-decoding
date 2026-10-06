"""Assemble the completed continuous-search outcomes without refitting models."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
EXP=ROOT.parent


def read(path):return json.loads(path.read_text())


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


studies=[
    ('2026-10-05_shared_finetuning','Shared fine-tuning',18),
    ('2026-10-05_shared_head_adaptation','Regularized frozen-encoder heads',0),
    ('2026-10-05_shared_confirmation','Original transformer, additional seeds',12),
    ('2026-10-05_static_readout_replication','Static readout replication',6),
    ('2026-10-05_neuron_panel_pilot','Nested neuron-panel development pilot',0),
    ('2026-10-05_larger_panel','512-neuron transformer and MLP',6),
    ('2026-10-06_large_panel_components','512-neuron attention components',None),
    ('2026-10-06_population_tokens','Population-state temporal tokens',None),
    ('2026-10-06_context_query','Population-context neuron readout',None),
]
records=[];hashes={}
for folder,title,fits in studies:
    path=EXP/folder;status=read(path/'STATUS.json');assert status['status']=='complete',folder
    not_run=status.get('phase')=='not_run'
    review=read(path/'review.json') if (path/'review.json').exists() else {}
    if fits is None:fits=0 if not_run else review['new_neural_fits']
    if not_run:outcome='Not run: preceding study passed'
    elif folder.endswith('neuron_panel_pilot'):outcome='Development screen passed; no later test in this pilot'
    elif review.get('full_replication_passed'):outcome='Full practical replication passed'
    else:outcome='Full practical improvement/confirmation gate failed'
    records.append(dict(folder=folder,title=title,new_neural_fits=fits,outcome=outcome,full_practical_pass=bool(review.get('full_replication_passed')),report=f'../{folder}/report.md'))
    for name in ['STATUS.json','protocol.json','review.json','report.md','ASSESSMENT.md']:
        p=path/name
        if p.exists():hashes[str(p.relative_to(EXP.parent))]=sha(p)
context=read(ROOT/'panel_context.json');passed=[r for r in records if r['full_practical_pass']]
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),new_neural_fits=sum(r['new_neural_fits'] for r in records),
    new_analytic_solutions=216,analytic_breakdown={'regularized_residual_heads':192,'512_neuron_ridge_pilot':24},
    full_practical_candidates=[r['folder'] for r in passed],independent_animal_significance=False,
    historical_mice=['MP030','MP032','MP033','MP034'],main_application_unchanged=True,studies=records,source_hashes=hashes)
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# Continuous architecture search: completed batch',
    ('A candidate passed the full predefined practical replication gates: '+', '.join(r['title'] for r in passed)+'.' if passed else '**No candidate in this batch passed the full predefined practical replication gates.**'),
    f"The batch completed {summary['new_neural_fits']} new neural fits and 216 new analytic ridge solutions. Previously completed fits and saved predictions were reused. Computational checks passed, but a successful training run was not counted as a scientific pass. No gate was relaxed, no unfavorable seed removed and no failed subset rescued by a pooled average.",
]
study_table=['| Study | New neural fits | Outcome |','| --- | --- | --- |']
for r in records:study_table.append(f"| [{r['title']}]({r['report']}) | {r['new_neural_fits']} | {r['outcome']} |")
lines.append('\n'.join(study_table))
population=read(EXP/'2026-10-06_population_tokens'/'stage1_results.json')['subsets']['stage1']['summaries']
context_root=EXP/'2026-10-06_context_query'
context_stage=2 if (context_root/'stage2_results.json').exists() else 1
context_results=read(context_root/f'stage{context_stage}_results.json')['subsets']
context_latest=context_results['additional' if context_stage==2 else 'stage1']
lines += ['## Strongest new transformer evidence',
    f"The population-state transformer reduced mean relative pair-MSE by {100*population['population_attention_vs_native_attention']['mean_mse_gain']:.2f}% against the original 128-neuron transformer and {100*population['population_attention_vs_ridge512']['mean_mse_gain']:.2f}% against 512-neuron ridge, improving all four mouse means against both. These contrasts passed their practical gates. However, it remained {abs(100*population['population_attention_vs_large_mlp']['mean_mse_gain']):.2f}% worse than the stronger 512-neuron local MLP, and its matched population-MLP comparison won only two mouse means. The full gate failed.",
    'The final hybrid uses that population context to change how the local MLP reads neuron activity. Its similarly sized control replaces population temporal attention with an MLP while retaining input-dependent gating. The latest required seed group gave the following results; these are not interchangeable with pooled results.',
]
hybrid_rows=['| Hybrid transformer versus | Mean MSE gain | Mouse wins | Single-model wins | Contrast |','| --- | --- | --- | --- | --- |']
for key,name in [('context_mlp','Matched context MLP'),('large_mlp','512-neuron local MLP'),('population_attention','Population transformer parent'),('ridge512','512-neuron ridge')]:
    v=context_latest['summaries']['context_attention_vs_'+key]
    hybrid_rows.append(f"| {name} | {100*v['mean_mse_gain']:+.2f}% | {v['mouse_wins']}/4 | {v['seed_wins']}/{4*len(context_latest['seeds'])} | {'PASS' if v['full_pass'] else 'FAIL'} |")
lines.append('\n'.join(hybrid_rows))
lines += ['', '## The clearest new input finding',
    f"In a post-hoc descriptive comparison of already scored later predictions, moving from 128 to 512 neurons reduced matched MLP mean relative MSE by {100*context['contrasts']['mlp']['mean_mse_gain']:.2f}% and ridge MSE by {100*context['contrasts']['ridge']['mean_mse_gain']:.2f}%, with both improving in all four mice. Corresponding MAE gains were {100*context['contrasts']['mlp']['mean_mae_gain']:.2f}% and {100*context['contrasts']['ridge']['mean_mae_gain']:.2f}%. MLP individual comparisons favored 512 cells in 9/12 cases. [Exact effects and provenance](panel_context.json).",
    'That supports expanding the input panel in this development cohort. It does not isolate additional neural information from model capacity, population-summary changes or regularization. It does not establish an attention advantage or independent significance.',
    'The full-attention 512-neuron transformer improved 8.19% over its 128-neuron parent but was 22.34% worse than the equally informed 512-neuron MLP, losing all four mouse means. The earlier original-transformer confirmation retained an 18.28% pooled advantage over the 128-neuron MLP across nine seeds, but its three newest seeds failed consistency. These comparisons use different panels and seed sets; neither should be substituted for the other.',
    '## Validation scope and next evidence',
    'All current behavioral results reuse four historically searched mice. Seeds, overlapping windows and model pairs are not new animals. Repeated architectural searching on these periods cannot produce fresh independent confirmation. Any practical pass above remains a development result; no independent animal-level significance is claimed.',
    'A separate Stringer oriented-stimulus release has six different named mice paired with running arrays in the authors’ first-six-recording analysis. Only public metadata, identities, shapes and publisher hashes were checked. Response files were not downloaded; trial timing, response alignment and animal non-overlap remain to be verified. This is a possible future confirmation cohort, not completed validation. See [feasibility record](../2026-10-06_stringer_oriented_feasibility/feasibility.json) and [publisher catalog](https://api.figshare.com/v2/articles/8279387).',
    'A source-only timing follow-up also found that the released trial summaries cannot silently replace the original continuous bins: 32 ordinary trials span roughly 56 seconds. The public preprocessing helper normalizes across all stimulus trials and would need a train-only replacement. Actual response fields and the aggregation of running values remain unverified. See [alignment notes and primary sources](oriented_alignment_notes.json). No new animal data were scored.',
    'Application train.py, model.py and data.py remain unchanged. Nothing was published. Generation and the stopped reconstruction study remain paused. Every study above is closed under its own stopping rule; failed grids are not extended.',
]
(ROOT/'REPORT.md').write_text('\n\n'.join(lines)+'\n')
(ROOT/'STATUS.json').write_text(json.dumps(dict(status='complete',new_neural_fits=summary['new_neural_fits'],new_analytic_solutions=216,full_practical_pass=bool(passed),independent_animal_significance=False),indent=2)+'\n')
print('Search summary saved:',summary['new_neural_fits'],'new neural fits; practical candidates:',summary['full_practical_candidates'])
