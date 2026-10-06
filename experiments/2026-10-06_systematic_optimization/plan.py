"""Fixed structural grid, refinement recipes and promotion rules."""
import itertools
from common import ROOT, PROJECT, EXP, FAIR, PANEL, MICE, read, write, digest, now, config_id


def baseline(family):
    return dict(family=family, width=16, depth=1, history=32, patch=4, dropout=.05, lr=.001, weight_decay=.01, epochs=24)


def recipes():
    return [dict(lr=lr, weight_decay=wd, dropout=drop, epochs=48)
        for lr, wd, drop in itertools.product([.0003, .001], [.001, .01], [0., .1])] + [dict(lr=.001, weight_decay=.01, dropout=.05, epochs=24)]


def screen_configs():
    configs = []
    for family, width, depth, history, patch in itertools.product(['attention', 'mlp'], [16, 32, 64], [1, 2, 3], [16, 32, 64], [2, 4]):
        configs.append(dict(baseline(family), width=width, depth=depth, history=history, patch=patch))
    assert len(configs) == 108 and len({config_id(c) for c in configs}) == 108
    return configs


def jobs(configs, splits, seeds, screen=False):
    return [dict(config=c, split=split, seed=seed, epochs=12 if screen else c['epochs'])
        for c in configs for split in splits for seed in seeds]


def freeze():
    assert read(ROOT/'selfcheck.json')['passed'] and read(ROOT/'prepared.json')['passed'] and read(ROOT/'ridge_selfcheck.json')['passed']
    source = [ROOT/name for name in ['common.py', 'models.py', 'data.py', 'fit.py', 'plan.py', 'pipeline.py', 'evaluate.py', 'ridge.py', 'check.py', 'extra_checks.py', 'selfcheck.json', 'ridge_selfcheck.json', 'prepared.json']]
    source += [PROJECT/name for name in ['train.py', 'model.py', 'data.py']]
    source += [PANEL, EXP/'2026-10-05_larger_panel'/'models.py', EXP/'2026-10-03_shared_behavior'/'models.py', EXP/'2026-10-03_dynamics_baseline'/'models.py']
    source += [FAIR/mouse/name for mouse in MICE for name in ['train_x.npy', 'train_y.npy', 'selection_x.npy', 'selection_y.npy', 'statistics.npz', 'metadata.json', 'later_raw.npz']]
    configs = screen_configs()
    write(ROOT/'screen_jobs.json', jobs(configs, ['fold0', 'fold1'], [101], screen=True))
    source.append(ROOT/'screen_jobs.json')
    write(ROOT/'protocol.json', dict(created_utc=now(),
        question='Find a carefully optimized population transformer and compare it with independently tuned population MLP, strong local MLP and ridge; transformer superiority is an outcome, not an entry requirement.',
        authorization='User explicitly requested more exhaustive optimization and use of available budget after accepting a three-family comparative project. This is a new optimization study, not an extension or reinterpretation of previous failed gates.',
        scope='Local CPU, three workers with two threads each. No external paid compute. Application code remains unchanged. All four mice and earlier/later periods have historical reuse; no independent animal significance.',
        structural_grid=dict(width=[16,32,64], depth=[1,2,3], history=[16,32,64], patch=[2,4], families=['attention','mlp'], neurons=512, attention_heads='2 atwidth16;4 otherwise', feedforward='2xwidth', output='last token plus2xhistory population mean/std features'),
        screening='108 configurations x2 chronological folds xseed101 =216 fits. Each12epochs of a24epoch cosine schedule; select epoch0..12. All grid combinations tested. Screening is limited-budget selection, not proof eliminated models cannot win after more training.',
        refinement='For each population family promote its top2 structural configurations by mean fold score, tie by config hash. Evaluate all9 predeclared recipes on each. Also include its historical baseline configuration if absent. LocalMLP independently receives the same9 recipes. Each configuration uses2folds xseeds102,103. Maximum188 additional fits. No recipe grid changed after screening.',
        recipes=recipes(),
        final='Select lowest mean score over2folds x2seeds independently for populationT,populationMLP,localMLP. Freeze choices before opening original selection outcomes for these fits. Train each winner plus fixed original populationT and localMLP recipes and the MLP matched to the winning transformer configuration, deduplicating identical configs. Seeds201..206; maximum36final fits. Original selection segment chooses epoch only; no architecture selection there. Later scoring only after all selections are locked. Six seeds are not six animals.',
        maximum_new_neural_fits=440,
        ridge='Search histories16/32/64 and8regularizers10^-4..10^3 on both chronological folds, using exact train-only feature standardization and an unpenalized intercept. Choose history/penalty per mouse by mean fold score; fit4final regressions.192development solutions plus4final; no later tuning.',
        data='Recover fixed512cell sequences from existing cached windows. Fold0 train first55%, validate after64bin gap until75%;fold1 train first75%, validate after64bin gap toend. Each segment requires64bins before first target, so every history uses identical endpoints. Fit-only neuron and target scaling recomputed independently per fold. Fullfit usesoriginaltraining segment and originalselection segment with the same64bin warmup. Existing neuronpanel and historical full-prefix unlabeled eligibility inherited and disclosed.',
        objective='Equal-mouse training normalizedMSE, batch64, AdamW, clip1, cosine to10%lr. Selection uses mean across mice of physical-zero-boundedMSE divided by max(validationMSE of fitting-label-mean prediction,.05) in fold-normalized units. Hyperparameter rank is mean score overbothfolds(andbothrefinementseeds). These are development metrics, not new holdouts.',
        endpoints='Later common endpoints begin at originalraw index87:24cached-sequence offset+63history warmup. All current models/ridge see the same endpoints. Primary metrics average individual-seed MSE/MAE within mouse, then equal-mouse relative effects; all distinct2seedpair errors secondary. Report raw tables,R2,individualseeds,leave-one-mouse-out and learning versus untrainedmean.',
        comparisons='Primary optimization benefit: selectedT versus fixeddefaultpopulationT. Separately report versus independently optimizedMLPs,matched-configMLP,ridge andfixedlocalMLP. Practical gain criterion remains>=5%meanrelativeMSE,>=3/4mousemeans,>=two-thirds singles,no mouse>10%harm,nonnegativeMAEgain and>=two-thirds winsversusinitial. Check201-203,204-206,all6separately; pooledcannotrescue. If winner equalsbaseline, optimization gainiszero,notapass. Previous study failures stayfailures; projectvaliditydoesnotrequireTtobeoverallwinner.',
        stopping='Complete this finite grid, refinement and fixedfinal comparison regardless of rankings. No appended configurations,seeds,mouse exclusions or threshold changes. This exhausts the listed grid,not all transformer architectures or all hyperparameter combinations.',
        source_hashes={str(p.relative_to(PROJECT)):digest(p) for p in source}))


if __name__ == '__main__':
    freeze()
