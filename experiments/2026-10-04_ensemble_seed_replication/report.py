"""Report fresh-seed ensemble results with matched initialization pairings."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def read(name):return json.loads((ROOT/name).read_text())


def main():
    new,old,combined=[read(n) for n in ['summary.json','old_pairing_context.json','combined_context.json']]
    attention=read('attention_context.json')['results']
    assert read('audit.json')['passed'] and read('ensemble_audit.json')['passed']
    passed=new['overall_gate']
    status='PASS' if passed else 'FAIL'
    lines=['# Ensemble seed replication: completed','',
        f'The new-seed architecture-diversity practical gate is **{status}**. Six new shared models were trained with fixed seeds 13/14/15 and the unchanged original recipe. The study compares two-model ensembles using matched seed pairs and averages errors over all allowed pairings; it does not choose a favorable pair.', '',
        '## Why this test was needed','',
        'The previous fixed mixed average combined transformer and MLP models with the same seed, while the same-family controls used different seeds. Shared non-temporal initial weights and batch plans could affect error diversity. That possibility did not establish the direction or size of a bias, so we first checked every pairing in the saved models and then repeated the recipe with new seeds.', '',
        'For each unordered seed pair (i, j), the same-family ensembles are the averages of A_i with A_j and M_i with M_j. The mixed condition averages the errors of two predictions: (A_i+M_j)/2 and (A_j+M_i)/2. Each prediction uses exactly two models. We never average all four predictions. Bounded individual speed predictions are averaged before scoring.', '',
        'Three seed pairs produce twelve mouse-by-pair comparisons. These comparisons overlap in fitted models and are not twelve independent animals. The experimental replication units remain the same four historically examined recordings. Model counts and allocated training updates are matched; elapsed time and operation counts are not matched between attention and MLP models.', '',
        '## Primary new-seed results','',
        '| Cross-seed mixed compared with | Mean relative gain | Mouse wins | Mouse-pair wins | Practical threshold | Descriptive 97.5% interval |',
        '|---|---:|---:|---:|---|---|']
    names={'attention_pair':'Two transformers','mlp_pair':'Two MLPs','mixed_same':'Same-seed mixed average',
           'attention_single':'Single transformer','mlp_single':'Single MLP'}
    for control in ['attention_pair','mlp_pair']:
        c=new['comparisons'][control];lo,hi=new['descriptive_97_5_intervals'][control]
        lines.append(f"| {names[control]} | {100*c['mean_relative_gain']:.2f}% | {c['mouse_wins']}/4 | {c['mouse_pair_wins']}/12 | {'Pass' if c['threshold_pass'] else 'Fail'} | {100*lo:.1f}% to {100*hi:.1f}% |")
    lines += ['', 'Positive gains favor the mixed ensemble. Each comparison must achieve at least 2% mean within-mouse relative error reduction, three mouse wins and eight matched mouse-pair wins; both comparisons must pass. Percentages average mouse-level ratios after averaging ensemble errors, not predictions, across seed pairs.', '',
        '| Mouse | Single transformer | Single MLP | Two transformers | Two MLPs | Cross-seed mixed | Same-seed mixed |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in new['rows']:
        lines.append('| '+r['mouse']+' | '+' | '.join(f'{r[k]:.6f}' for k in ['attention_single','mlp_single','attention_pair','mlp_pair','mixed_cross','mixed_same'])+' |')
    lines += ['', 'Entries are bounded MSE in training-normalized speed units. Single-model entries average individual seed errors. Ensemble entries average the errors of two-model predictors; they are not predictions from an ensemble of all six models.', '',
        '| Mouse | Gain vs two transformers | Gain vs two MLPs |',
        '|---|---:|---:|']
    for i,r in enumerate(new['rows']):
        lines.append(f"| {r['mouse']} | {100*new['comparisons']['attention_pair']['mouse_gains'][i]:.2f}% | {100*new['comparisons']['mlp_pair']['mouse_gains'][i]:.2f}% |")
    lines += ['', '## Old, new and combined seed sets','',
        '| Seed set | Mixed vs two transformers | Mixed vs two MLPs | Mixed vs same-seed mixture |',
        '|---|---:|---:|---:|']
    for label,record in [('Original 10/11/12',old),('New 13/14/15',new),('Combined 10–15 (descriptive)',combined)]:
        lines.append('| '+label+' | '+' | '.join(f"{100*record['comparisons'][k]['mean_relative_gain']:.2f}% ({record['comparisons'][k]['mouse_wins']}/4 mice)" for k in ['attention_pair','mlp_pair','mixed_same'])+' |')
    lines += ['', 'The old all-pair result was recorded before the new fits completed. It does not rescue the mixed-architecture hypothesis: the mixed condition is 4.2% worse than two transformers and 4.8% better than two MLPs, with two mouse wins in each case. The combined analysis uses all fifteen unordered pairs from six seeds, but is secondary and does not turn pair counts into independent evidence. No seed was added after results.', '',
        '| New mixed ensemble compared with | Mean relative gain | Mouse wins |',
        '|---|---:|---:|']
    for name in ['attention_single','mlp_single','mixed_same']:
        c=new['comparisons'][name]
        lines.append(f"| {names[name]} | {100*c['mean_relative_gain']:.2f}% | {c['mouse_wins']}/4 |")
    lines += ['', '## Secondary finding: the transformer itself','',
        'The new batch supports a more useful direction than mixed-architecture routing: both single transformers and transformer pairs outperform their MLP counterparts on all four mouse means. These are secondary contrasts of the same predefined models. They do not replace the failed primary hybrid gate.', '',
        '| Seed set | Transformer pair vs MLP pair | Mouse wins | Matched mouse-pair wins | Descriptive 97.5% interval | Single transformer vs single MLP |',
        '|---|---:|---:|---:|---|---:|']
    for cohort,label in [('old','Original 10/11/12'),('new','New 13/14/15'),('combined','Combined 10–15')]:
        pair=attention[cohort]['attention_pair'];single=attention[cohort]['attention_single'];lo,hi=pair['descriptive_97_5_interval']
        lines.append(f"| {label} | {100*pair['mean_relative_gain']:.2f}% | {pair['mouse_wins']}/4 | {pair['paired_comparison_wins']}/{pair['paired_comparison_count']} | {100*lo:.1f}% to {100*hi:.1f}% | {100*single['mean_relative_gain']:.2f}% |")
    lines += ['', 'The new transformer-pair gain is 28.7%, with four mouse wins and eleven matched mouse-pair wins. All six seeds together give a 19.3% mean pair gain and three mouse wins; single transformers give an 18.3% mean gain, also three mouse wins. Combined pair gains are 34.4%, 38.9%, −4.4% and 8.3% for MP030/032/033/034. Leaving any one mouse out keeps the combined pair gain positive at 12.8%–27.2%. MP033 remains a counterexample to uniform superiority.', '',
        'The old three seeds favored transformer pairs on only two mice, and every descriptive interval still crosses zero. The additional initializations strengthen the practical transformer signal but do not establish independent significance. Seeds and seed pairs cannot substitute for more independent recordings.', '',
        'The architecture comparison bundles temporal attention and dynamic query routing against the temporal-MLP/static-query control. It is not an isolated causal test of one attention component and says nothing about coordinates, neural generation or biological connectivity. The earlier controlled component study retains its narrower interpretation.', '',
        'Retain the shared transformer and a fixed transformer-pair ensemble as leading experimental baselines, alongside MLP controls. The new mixed ensemble is 12.6% worse than two transformers on average and helps only one mouse relative to that control, so adding the MLP has not justified itself in this batch. See [secondary attention context](attention_context.json).', '',
        '## Pairing and prediction diversity','',
        '| Seed set | Mouse | Same-seed residual correlation | Cross-seed residual correlation |',
        '|---|---|---:|---:|']
    for label,record in [('Old',old),('New',new)]:
        for r in record['rows']:
            lines.append(f"| {label} | {r['mouse']} | {r['mean_same_seed_error_correlation']:.3f} | {r['mean_cross_seed_error_correlation']:.3f} |")
    lines += ['', 'The arithmetic check is exact: averaged squared error of a 50:50 pair equals the average individual error minus one quarter of the mean squared prediction disagreement. Across all pairings, the average single-model errors are held fixed, so the same-seed/cross-seed MSE difference is explained exactly by their disagreement term. This describes ensemble behavior; it does not establish biological connectivity or prove that seed labels themselves cause a particular error correlation.', '',
        '## Training, selection and learning','',
        '| Family | Seed | Selected epoch | Updates searched | Training examples |',
        '|---|---:|---:|---:|---:|']
    for r in read('selection_lock.json')['records']:
        lines.append(f"| {r['family']} | {r['seed']} | {r['selected_epoch']} | {r['updates']} | {r['examples']} |")
    lines += ['', f"New transformer fits beat their untrained output on {new['learning_vs_initial']['attention']}/12 mouse-seed comparisons; new MLP fits on {new['learning_vs_initial']['mlp']}/12. These learning checks are distinct from the ensemble-superiority gate.", '',
        'The architecture, 128-neuron/32-bin inputs, optimizer, 24-epoch schedule, balanced-mouse loss, batch generator and joint checkpoint selection are inherited unchanged. Each new fit completes 5,688 AdamW updates and 179,712 training windows. Earlier selection chooses one epoch per family/seed; all six choices lock before any new later inference. Different selected epochs are not different allocated search budgets.', '',
        '## Verification and interpretation','',
        'New arithmetic checks cover all distinct seed pairs, scalar ensemble errors, orientation-error averaging, ties and the averaging identity. All 600 mouse-by-epoch selection scores were independently recomputed, exact selected checkpoint reloads passed, batch orders match across families, and actual Adam counters confirm the allocated update budget. New later targets match the archived targets exactly and all 24 new single-model MSEs match independent scalar calculations.', '',
        'Ensemble auditing independently verifies 36 new mouse-pair scores and 180 combined mouse-pair scores, with both mixed orientations included before each comparison. Fixed source, input, reference and application hashes match. Earlier neural fits, raw-data diagnostics and application code were not changed.', '',
        'The descriptive intervals use 2,000 paired draws of mice and circular 100-bin time blocks. Shared Dirichlet weights over seed identities induce product weights on distinct pairs, preserving exclusion of duplicate-model pairs. They condition on the fitted models, ignore historical search and retraining uncertainty, and do not provide independent animal-level confirmation. Four existing recordings and fresh initializations cannot establish significance for a new animal population.', '',
        'This is a fixed initialization-reproducibility check of an existing recipe. It is not a new architecture, an adaptive seed search, or a biological experiment. Retain the original single-model and same-family ensemble controls when describing the mixed-model result. No pair was selected from later performance, and no further seed or architecture grid is queued.', '',
        'See [assessment](ASSESSMENT.md), [numeric summary](summary.json), [old pair context](old_pairing_context.json), [combined context](combined_context.json), [protocol](protocol.json), [training audit](audit.json), [ensemble audit](ensemble_audit.json), [runner](run.py), and [analysis](analyse.py).']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    ca,cm=new['comparisons']['attention_pair'],new['comparisons']['mlp_pair']
    final=['# Ensemble seed replication: completed','',
        'The new initializations strengthen the shared transformer as an experimental baseline. Two-transformer ensembles have 28.7% lower mean relative error than two-MLP ensembles across the new seeds, winning all four mouse means and 11/12 matched mouse-pair comparisons. Across all six seeds the gain is 19.3%, with three mouse wins; single transformers gain 18.3%, also on three mice. These are secondary comparisons, not independent statistical confirmation.', '',
        f'The fixed mixed-architecture gate **{status}**. Six new shared models used seeds 13/14/15 with the original architecture and training recipe; no old fit was repeated.', '',
        '| Cross-seed mixed ensemble compared with | Mean relative gain | Mouse wins | Matched mouse-pair wins |',
        '|---|---:|---:|---:|',
        f"| Two transformers | {100*ca['mean_relative_gain']:.2f}% | {ca['mouse_wins']}/4 | {ca['mouse_pair_wins']}/12 |",
        f"| Two MLPs | {100*cm['mean_relative_gain']:.2f}% | {cm['mouse_wins']}/4 | {cm['mouse_pair_wins']}/12 |", '',
        'Positive means lower mixed-ensemble error. Each predictor uses exactly two models. For a given seed pair, the two mixed orientations contribute averaged errors, not an average of four predictions. The comparisons overlap in seeds and remain four-animal evidence.', '',
        'The previous comparison used same-seed mixed models but different-seed same-family models. Matching all seed pairs did not rescue the original result: old mixed pairs were 4.2% worse than two transformers and 4.8% better than two MLPs, winning only two mice against either. The new-seed batch was fixed before those new results, so it checks optimization reproducibility without tuning toward a preferred answer.', '',
        'All six fits, locked later evaluation, numerical checks and reports are complete. Each fit received 24 epochs, 5,688 updates and 179,712 training windows. Checkpoint reloads, 600 selection scores, matched batches, Adam counters, target alignment and independent ensemble metrics passed. Application code is unchanged.', '',
        'The combined six-seed analysis is descriptive. These are historically reused recordings, not new animals or untouched data; new seeds do not establish independent significance. Every descriptive interval crosses zero, and MP033 still favors the two-MLP ensemble by 4.4% in the combined comparison. No additional seed, gate, architecture change or publication is queued.', '',
        'The practical direction is to retain the shared transformer and its fixed two-model ensemble as leading benchmarks, while keeping the matched MLP and mixed-average controls. Adding an MLP does not improve the transformer ensemble consistently. This remains evidence about the whole tested architecture, not proof that a particular attention mechanism is uniquely responsible.', '',
        'See [full report](report.md), [new-seed results](summary.json), [attention context](attention_context.json), [combined context](combined_context.json), and [frozen protocol](protocol.json).']
    (ROOT/'ASSESSMENT.md').write_text('\n'.join(final)+'\n')


if __name__=='__main__':main()
