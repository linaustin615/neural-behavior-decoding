"""Fresh-seed practical confirmation, distinct from modification utility."""
import math
import numpy as np
import run

shared = run.module('confirmation_metrics', run.FT / 'analyse.py')
GROUPS = ['native_attention', 'native_mlp', 'tuned_attention', 'tuned_mlp', 'ridge']
PRIMARY = ['native_attention_vs_native_mlp', 'native_attention_vs_tuned_mlp', 'native_attention_vs_ridge']
CONTRASTS = PRIMARY + ['tuned_attention_vs_native_attention', 'tuned_attention_vs_tuned_mlp']


def main():
    locked = run.verify_lock()
    evaluation = run.read(run.ROOT / 'evaluation.json')
    for name, value in evaluation['prediction_hashes'].items():
        assert run.digest(run.ROOT / name) == value
    subsets, count, checks = {}, 0, 0
    for name, seeds in [('new', run.SEEDS), ('all', run.ALL_SEEDS)]:
        rows = []
        for mouse in run.MICE:
            meta = run.read(run.ref.FAIR / mouse / 'metadata.json')
            lower = -meta['speed_mean'] / meta['speed_std']
            scores = {}
            with np.load(run.ROOT / mouse / 'later_predictions.npz') as z:
                y = z['target']
                for group in GROUPS:
                    raw = np.stack([z['ridge'] if group == 'ridge' else z[f'{group}_s{s}'] for s in seeds])
                    scores[group], n = shared.metrics(raw, y, lower)
                    count += n
                with np.load(run.FT / mouse / 'later_predictions.npz') as old:
                    np.testing.assert_array_equal(y, old['target'])
                    for key in old.files:
                        np.testing.assert_array_equal(z[key], old[key])
                        checks += 1
                with np.load(run.ft.SHARED / mouse / 'later_predictions.npz') as old:
                    np.testing.assert_array_equal(y, old['target'])
                    np.testing.assert_array_equal(z['ridge'], old['raw_ridge'])
                    checks += 2
            median = float(np.median(np.load(run.ref.BASE / mouse / 'train_y.npy')))
            controls = {k: dict(mse=float(np.mean((value-y)**2)),mae=float(np.mean(abs(value-y))))
                for k,value in [('training_mean',0.),('training_median',median)]}
            rows.append(dict(mouse=mouse,n=len(y),scores=scores,controls=controls))
        summaries = {}
        for contrast in CONTRASTS:
            a,b=contrast.split('_vs_')
            mse=[1-r['scores'][a]['mse']/r['scores'][b]['mse'] for r in rows]
            mae=[1-r['scores'][a]['mae']/r['scores'][b]['mae'] for r in rows]
            sa=np.array([r['scores'][a]['single_mse'] for r in rows]);sb=np.array([r['scores'][b]['single_mse'] for r in rows])
            s=dict(mean_mse_gain=float(np.mean(mse)),mean_mae_gain=float(np.mean(mae)),mouse_mse_gains=mse,mouse_mae_gains=mae,
                mouse_wins=sum(v>0 for v in mse),seed_wins=int(np.sum(sa<sb)),
                single_mean_mse_gain=float(np.mean(1-sa.mean(1)/sb.mean(1))),
                leave_one_mouse_out=[float(np.mean(mse[:i]+mse[i+1:])) for i in range(4)],
                individual_seeds=[dict(seed=seed,mean_mse_gain=float(np.mean(1-sa[:,i]/sb[:,i])),mouse_wins=int(np.sum(sa[:,i]<sb[:,i]))) for i,seed in enumerate(seeds)])
            s['mse_consistency_pass']=s['mean_mse_gain']>=.05 and s['mouse_wins']>=3 and s['seed_wins']>=math.ceil(8*len(seeds)/3)
            s['harm_guard']=min(mse)>=-.1
            s['mae_guard']=s['mean_mae_gain']>=0
            s['full_pass']=s['mse_consistency_pass'] and s['harm_guard'] and s['mae_guard']
            independent_wins=sum(sum(x<y for x,y in zip(r['scores'][a]['single_mse'],r['scores'][b]['single_mse'])) for r in rows)
            assert independent_wins==s['seed_wins']
            assert s['full_pass']==(sum(mse)/4>=.05 and sum(x>0 for x in mse)>=3 and independent_wins>=math.ceil(8*len(seeds)/3) and all(x>=-.1 for x in mse) and sum(mae)/4>=0)
            checks+=2
            summaries[contrast]=s
        initial_wins=sum(sum(x<r['controls']['training_mean']['mse'] for x in r['scores']['native_attention']['single_mse']) for r in rows)
        gate=all(summaries[k]['full_pass'] for k in PRIMARY) and initial_wins>=math.ceil(8*len(seeds)/3)
        subsets[name]=dict(seeds=seeds,rows=rows,summaries=summaries,pairs_per_mouse=math.comb(len(seeds),2),
            initial_wins=initial_wins,primary_pass=gate)
    decision=subsets['new']['primary_pass'] and subsets['all']['primary_pass']
    run.write(run.ROOT/'results.json',dict(subsets=subsets,practical_confirmation_passed=decision,
        independent_animal_significance=False,new_neural_fits=12,old_fits_repeated=0,
        candidate='original shared transformer; neither failed modification study is rescued'))
    run.write(run.ROOT/'audit.json',dict(passed=True,scalar_errors_checked=count,aggregate_archive_checks=checks,
        source_input_hashes=len(run.verify()['hashes']),selected_artifact_hashes=len(locked['hashes']),
        selection_scores_checked=locked['selection_scores_checked'],new_predictions=evaluation['new_predictions'],
        new_native_fits=6,new_continuations=6,main_application_unchanged=True))
    for subset in ['new','all']:
        print(subset,subsets[subset]['summaries'],flush=True)
    print('Practical confirmation:',decision,flush=True)


if __name__=='__main__':
    main()
