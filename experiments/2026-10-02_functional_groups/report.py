import json

import numpy as np

import run as r


def main():
    result=json.loads((r.ROOT/'results.json').read_text())
    protocol=r.verify()
    checks=json.loads((r.ROOT/'completion_checks.json').read_text())
    assert checks['passed']
    rows=result['evaluations']
    lines=['# Final activity-grouping comparison','',
        f"Decision: **{result['decision'].upper()}**.",'',
        'All 18 prespecified fits completed. This is a new, bounded exploratory comparison on the existing MP019 recording. It does not establish independent statistical confirmation, measured connections, or a neural simulator.', '',
        '| Model | Mean later-evaluation MSE | Mean R² |',
        '|---|---:|---:|']
    for condition in r.CONDITIONS:
        selected=[x for x in rows if x['task']['condition']==condition]
        lines.append(f"| {condition} | {result['mean_evaluation_mse'][condition]:.6f} | {np.mean([x['evaluation']['r2'] for x in selected]):.5f} |")
    lines += ['', 'MSE uses training-standardized speed; lower is better. Primary predictions are floored at zero physical speed. The target is concurrent running speed, not future speed. `global` is unrestricted attention across activity-plus-ID tokens. Coordinates are zero in every arm.', '',
        '| Functional groups versus… | Relative error reduction | Matched wins | Positive pool means | Conditional 97.5% interval |',
        '|---|---:|---:|---:|---:|']
    for name,c in result['contrasts'].items():
        lo,hi=c['conditional_97_5_interval']
        lines.append(f"| {name} | {c['relative_improvement']:+.2%} | {c['pair_wins']}/6 | {c['positive_pools']}/3 | [{lo:+.2%}, {hi:+.2%}] |")
    lines += ['',f"Frozen promising gate: **{result['promising_gate']}**. Stronger exploratory gate: **{result['stronger_exploratory_gate']}**.",'',
        'The promising gate requires at least 2% mean error reduction against BOTH controls, at least 5/6 paired wins each, and positive means in all three pools each. The stronger exploratory gate additionally requires both conditional interval lower bounds above zero. Both outcomes are reported without adding runs or changing criteria.', '',
        'Intervals use 4,000 paired crossed resamples of three pools, two training seeds and circular blocks of 100 evaluation examples. The 1.25th/98.75th percentiles allow for the two planned contrasts, but do not correct the historical project-wide search or provide biological replication. The 573 evaluation examples are temporally dependent. Pool samples overlap and the two seeds are technical repeats. These intervals are descriptive and conditional, not confirmatory significance tests.', '',
        '## Do the relationships persist later?', '',
        '| Pool | Training within-group correlation | Later within-group correlation | Mean randomized later correlation | Descriptive upper-tail rank |',
        '|---|---:|---:|---:|---:|']
    for row in result['group_stability']:
        e=row['evaluation']
        lines.append(f"| {row['pool']} | {row['train']['mean_within_group_correlation']:.5f} | {e['mean_within_group_correlation']:.5f} | {e['mean_randomized_correlation']:.5f} | {e['descriptive_upper_tail_rank']:.3f} |")
    lines += ['', 'The pair list is fixed from training-defined groups: 1,024 distinct-cell pairs sampled within each of eight groups, with equal group weighting. The same edge pattern is reassigned through 999 neuron permutations for a descriptive null comparison. Persistent co-activity can reflect shared behavior, common population signals or recording artifacts. It is not evidence of synapses, direction of influence or causation. Persistence and incremental predictive value are separate endpoints.', '',
        '## Frozen design', '',
        'Three fixed 2,048-cell pools (101, 202, 303), two training seeds (10, 11), and three conditions give 18 fits. Each training time course is centered and normalized to unit L2 norm. The first 32 scaled eigenvectors of the training-only neuron correlation matrix feed eight-cluster KMeans with ten fixed-seed initializations. No running-speed labels or coordinates enter grouping. This is one static, simultaneous co-activity construction; it does not test lagged or state-dependent relationships.', '',
        'Functional groups restrict the first eight learned summaries, one per group; the remaining eight summaries attend to all neurons. Random groups permute cell membership while exactly preserving the functional group sizes. The global baseline leaves all sixteen summaries unrestricted. Every model has 81,377 parameters, activity embeddings, unrestricted neuron IDs, the same initialization within each pool/seed, and zero coordinate inputs. The retained coordinate network supplies only a learned common offset.', '',
        'Training is [0,4160), checkpoint selection [4260,4860), and later evaluation [4960,5564). An eight-bin window and offset 31 produce 4,129 training, 569 selection and 573 evaluation examples. Normalization uses training only. All 18 checkpoints were fixed before the later interval was scored. This clean separation within the new run does not make historically examined data pristine. No examples from the old test tail were evaluated.', '',
        'All models use AdamW lr 0.001, weight decay 0.01, batch 32, cosine decay to 0.0001 over 24 epochs, gradient clip 1, and early stopping after at least 12 epochs and seven stale selection evaluations. Selection includes untrained epoch zero. The inherited recipe is fixed across conditions; this is not a comparison of fully optimized model families. Archived whole-validation-selected checkpoints were incompatible with the new selection rule, so all 18 fits were new.', '',
        '## Verification and limits', '',
        f"{result['beat_untrained']}/18 models beat their untrained later-evaluation error; {result['beat_constant']}/18 beat the training-mean constant. {result['selected_budget_limit']}/18 selected the maximum epoch 24. A successful training run is not itself a passed scientific gate.", '',
        f"All 18 checkpoints were reloaded on the complete selection interval before their later evaluation. Maximum selection-prediction difference: {result['selection_reload_max_error']:.3g}. Independent Torch arithmetic reproduced every evaluation MSE. Matching initialization, equal parameter counts, group masks, cell alignment, finite gradients/checkpoints, new split/window boundaries and unchanged application/source hashes passed.", '',
        'KMeans emitted NumPy/BLAS matmul warnings while choosing initial centers. The resulting cluster labels were independently verified against nearest-center distances using Torch, and all groups, model tensors and predictions were finite. Correlation eigendecomposition uses Torch float64. The grouping is not claimed to be a globally optimal clustering.', '',
        'The top-32 correlation representation retains only about 11% of total standardized activity variance. Therefore a failure of these groups is not a rejection of every possible activity relationship. Likewise, beating randomized groups alone would not justify replacing the unrestricted reference model.', '',
        'True xyz positions appear only in the visualization. Group colors represent training-defined activity clusters; they do not designate anatomy or measured connections.', '',
        '## Artifacts', '',
        '`protocol.json` fixes the question, budget and decision rules. `groups_*.npz` and `groups_*.json` store cell identities, positions, labels and grouping metadata. `runs/` holds all checkpoints, selection histories and separate later-evaluation predictions. `results.json`, `per_run.csv`, `completion_checks.json`, `comparison.png` and `groups_in_space.png` summarize outcomes. Application code was not edited. The handoff records the final result; no further fitting is scheduled.']
    (r.ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    figures(result)
    manifest={str(p.relative_to(r.ROOT)):r.digest(p) for p in r.ROOT.rglob('*')
        if p.is_file() and p.name!='manifest.json' and p.suffix!='.log' and '__pycache__' not in p.parts}
    r.s.write_json(r.ROOT/'manifest.json',manifest)
    print('REPORT_AND_ARCHIVE_COMPLETE')


def figures(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for pool in r.POOLS:
        for seed in r.SEEDS:
            values=[next(x['evaluation']['mse'] for x in result['evaluations']
                if x['task']==dict(pool=pool,seed=seed,condition=c)) for c in r.CONDITIONS]
            axes[0].plot(range(3),values,marker='o',alpha=.75,label=f'{pool}/{seed}')
    axes[0].set(xticks=range(3),xticklabels=['Activity groups','Random groups','Unrestricted'],ylabel='Later-evaluation MSE',title='Six matched runs')
    axes[0].legend(title='Pool / seed',fontsize=7)
    for i,name in enumerate(['random','global']):
        c=result['contrasts'][name]
        lo,hi=np.array(c['conditional_97_5_interval'])*100
        point=c['relative_improvement']*100
        axes[1].plot([i,i],[lo,hi],color='tab:blue')
        axes[1].plot(i,point,'o',color='tab:blue')
    axes[1].axhline(0,color='gray',linewidth=1)
    axes[1].axhline(2,color='gray',linestyle='--',linewidth=1)
    axes[1].set(xticks=[0,1],xticklabels=['vs random','vs unrestricted'],ylabel='Functional-group error reduction (%)',title='Conditional descriptive 97.5% intervals')
    fig.suptitle('Final bounded comparison; one repeatedly examined recording')
    fig.tight_layout()
    fig.savefig(r.ROOT/'comparison.png',dpi=160)
    plt.close(fig)
    fig=plt.figure(figsize=(8,7))
    ax=fig.add_subplot(111,projection='3d')
    with np.load(r.ROOT/'groups_101.npz') as g:
        p=g['positions']
        ax.scatter(p[:,0],p[:,1],p[:,2],c=g['functional'],cmap='tab10',s=7,alpha=.65)
    ax.set(xlabel='x (dataset units)',ylabel='y (dataset units)',zlabel='z (dataset units)',
           title='Coordinates for display only\nColors show training-activity groups; pool 101')
    fig.tight_layout()
    fig.savefig(r.ROOT/'groups_in_space.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    main()
