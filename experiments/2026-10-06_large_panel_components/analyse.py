"""Apply the fixed component-selection follow-up gates."""
import math
import sys
import numpy as np
import run

metrics=run.module('panel_component_metrics',run.large.previous.FT/'analyse.py').metrics


def main(stage):
    locked=run.verify_lock(stage);ev=run.read(run.ROOT/f'stage{stage}_evaluation.json')
    for name,value in ev['prediction_hashes'].items():assert run.digest(run.ROOT/name)==value
    groups=['candidate','large_mlp','native_attention','ridge512']+(['large_attention'] if stage==1 else [])
    subsets=[('stage1',[10,11,12])] if stage==1 else [('additional',[13,14,15]),('all',run.SEEDS)]
    results={};count=checks=0
    for subset,seeds in subsets:
        rows=[]
        for mouse in run.MICE:
            meta=run.read(run.ref.FAIR/mouse/'metadata.json');lower=-meta['speed_mean']/meta['speed_std'];scores={}
            with np.load(run.ROOT/mouse/f'stage{stage}_predictions.npz') as z:
                y=z['target']
                for group in groups:
                    raw=np.stack([z['ridge512'] if group=='ridge512' else z[f'{group}_s{s}'] for s in seeds])
                    scores[group],n=metrics(raw,y,lower);count+=n
                with np.load(run.LARGE/mouse/'stage1_predictions.npz') as old:
                    for key in old.files:np.testing.assert_array_equal(z[key],old[key]);checks+=1
            rows.append(dict(mouse=mouse,n=len(y),scores=scores,initial_mse=float(np.mean(y**2))))
        summaries={}
        for control in groups[1:]:
            mse=[1-r['scores']['candidate']['mse']/r['scores'][control]['mse'] for r in rows]
            mae=[1-r['scores']['candidate']['mae']/r['scores'][control]['mae'] for r in rows]
            a=np.array([r['scores']['candidate']['single_mse'] for r in rows]);b=np.array([r['scores'][control]['single_mse'] for r in rows])
            wins=int(np.sum(a<b))
            gate=np.mean(mse)>=.05 and sum(x>0 for x in mse)>=3 and wins>=math.ceil(8*len(seeds)/3) and min(mse)>=-.1 and np.mean(mae)>=0
            summaries['candidate_vs_'+control]=dict(mean_mse_gain=float(np.mean(mse)),mean_mae_gain=float(np.mean(mae)),mouse_mse_gains=mse,mouse_mae_gains=mae,
                mouse_wins=sum(x>0 for x in mse),seed_wins=wins,full_pass=bool(gate),
                leave_one_mouse_out=[float(np.mean(mse[:i]+mse[i+1:])) for i in range(4)],
                individual_seeds=[dict(seed=seed,mean_mse_gain=float(np.mean(1-a[:,i]/b[:,i])),mouse_wins=int(np.sum(a[:,i]<b[:,i]))) for i,seed in enumerate(seeds)])
        initial=sum(sum(x<r['initial_mse'] for x in r['scores']['candidate']['single_mse']) for r in rows)
        primary=all(summaries['candidate_vs_'+c]['full_pass'] for c in ['native_attention','large_mlp','ridge512']) and initial>=math.ceil(8*len(seeds)/3)
        results[subset]=dict(seeds=seeds,rows=rows,summaries=summaries,initial_wins=initial,primary_pass=primary)
    prior=run.read(run.ROOT/'stage1_results.json')['stage_passed'] if stage==2 else True
    passed=prior and all(v['primary_pass'] for v in results.values())
    run.write(run.ROOT/f'stage{stage}_results.json',dict(subsets=results,stage_passed=passed,full_replication_passed=stage==2 and passed,selected_variant=run.chosen(),independent_animal_significance=False))
    run.write(run.ROOT/f'stage{stage}_audit.json',dict(passed=True,scalar_errors_checked=count,archived_arrays_checked=checks,selection_scores_checked=locked['selection_scores_checked'],
        frozen_source_hashes=len(run.verify()['hashes']),selected_artifacts=len(locked['hashes']),new_predictions=ev['new_predictions']))
    for name,part in results.items():print(name,part['summaries'],flush=True)
    print('Stage',stage,'pass',passed,flush=True)


if __name__=='__main__':main(int(sys.argv[1]))
