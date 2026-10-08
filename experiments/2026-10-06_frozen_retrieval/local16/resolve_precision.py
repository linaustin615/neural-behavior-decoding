"""Document and resolve clipping-roundoff ties without changing selected recipes."""
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import run
import local16


def main():
    root = local16.OUT
    assert not (root/'precision_resolution.json').exists()
    original = run.read(root/'selection_lock_v1.json')
    validated = {}
    for mouse in run.MICE:
        validated[mouse] = {}
        for family in ('transformer','mlp'):
            choice = original['choices'][mouse][family]
            losses = []
            for seed in run.SEEDS:
                d,p,head = local16.predictions(mouse,family,seed,'validation',[choice['tau']])
                y,lower = d.y[d.indices],d.metadata['lower']
                losses.append([run.metrics(local16.blend(head,p[0],a,lower),y,lower)['mse'] for a in local16.ALPHAS])
            selected = local16.ALPHAS[int(np.argmin(np.mean(losses,axis=0)))]
            assert selected==choice['alpha'], 'stop if a numerical correction changes a scientific choice'
            validated[mouse][family] = dict(alpha_unchanged=selected,corrected_validation_mse_grid=losses)
    summary = run.read(root/'summary_v1.json')
    max_change = 0.
    for mouse in run.MICE:
        lower = run.examples(mouse,'train').metadata['lower']
        for family in ('transformer','mlp'):
            alpha = original['choices'][mouse][family]['alpha']
            for seed in run.SEEDS:
                name = f'{mouse}_{family}_{seed}.npz'
                with np.load(root/'predictions_v1'/name) as old:
                    arrays = {k:old[k] for k in old.files}
                corrected = local16.blend(arrays['parent'],arrays['prediction'],alpha,lower)
                max_change = max(max_change,float(np.max(np.abs(corrected-arrays['hybrid']))))
                arrays['hybrid'] = corrected
                np.savez_compressed(root/'predictions'/name,**arrays)
                row = next(r for r in summary['rows'] if r['mouse']==mouse and r['arm']==family+'_hybrid' and r['seed']==seed)
                row.update(run.metrics(corrected,arrays['target'],lower))
                if alpha==0:
                    parent = next(r for r in summary['rows'] if r['mouse']==mouse and r['arm']==family+'_parent' and r['seed']==seed)
                    assert row['mse']==parent['mse'] and row['qfm']==parent['qfm']
    for name in summary['contrasts']:
        a,b = name.split('_vs_')
        summary['contrasts'][name] = run.contrast(summary['rows'],a,b)
    for arm in summary['gates']:
        family = arm.split('_')[0]
        c = summary['contrasts'][arm+'_vs_'+family+'_parent']
        checks = dict(mean_gain=c['mean_gain']>=.05,mouse_wins=c['mouse_wins']>=5,harm=c['max_harm']<=.1,
            quiet_wins=c['quiet_wins']>=5,pca_gain=summary['contrasts'][arm+'_vs_pca_local16']['mean_gain']>0)
        summary['gates'][arm] = dict(passed=all(checks.values()),checks=checks)
    resolution = dict(utc=run.utc(),reason='float32 clipping rounded the physical-zero floor before float64 scoring, causing microscopic non-ties at alpha0',
        correction='clip both components in float64 before blending; alpha0 exactly preserves parent metrics',
        outcomes_already_examined=True,all_validation_choices_unchanged=True,max_prediction_change=max_change,
        old_protocol_sha256=run.digest(root/'protocol_v1.json'),old_source_sha256=run.digest(root/'source_v1.py'),
        new_source_sha256=run.digest(local16.__file__),validation_checks=validated,
        scientific_changes='none; same features, k, temperatures, alphas, data, comparisons and gates; retain all v1 records')
    run.save(root/'precision_resolution.json',resolution)
    protocol = run.read(root/'protocol_v1.json')
    protocol.update(source_sha256=run.digest(local16.__file__),numeric_revision='float64 blend clipping; see precision_resolution.json',
        amended_utc=run.utc(),original_protocol_sha256=run.digest(root/'protocol_v1.json'))
    run.save(root/'protocol.json',protocol)
    original['protocol_sha256'] = run.digest(root/'protocol.json')
    original['numeric_amendment_utc'] = run.utc()
    original['choice_note'] = 'all original selections unchanged; corrected validation grids in precision_resolution.json'
    run.save(root/'selection_lock.json',original)
    summary['precision_resolution'] = 'precision_resolution.json; all recipe choices unchanged'
    run.save(root/'summary.json',summary)
    print('Precision correction complete; all alpha choices unchanged; max change',max_change)


if __name__=='__main__':
    torch.set_num_threads(4)
    main()
