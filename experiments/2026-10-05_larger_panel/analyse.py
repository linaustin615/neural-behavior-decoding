"""Frozen pair-ensemble and single-fit criteria for the larger neuron panel."""
import math
import sys
import numpy as np
import run

metrics=run.module('larger_panel_metrics',run.previous.FT/'analyse.py').metrics
GROUPS=['large_attention','large_mlp','native_attention','native_mlp','ridge512']
CONTROLS=['native_attention','large_mlp','ridge512','native_mlp']


def main(stage):
    locked=run.verify_lock(stage);ev=run.read(run.ROOT/f'stage{stage}_evaluation.json')
    for name,value in ev['prediction_hashes'].items():assert run.digest(run.ROOT/name)==value,name
    subsets=[('stage1',[10,11,12])] if stage==1 else [('additional',[13,14,15]),('all',run.SEEDS)]
    results={};count=0;checks=0
    for subset,seeds in subsets:
        rows=[]
        for mouse in run.MICE:
            meta=run.read(run.ref.FAIR/mouse/'metadata.json');lower=-meta['speed_mean']/meta['speed_std'];scores={}
            with np.load(run.ROOT/mouse/f'stage{stage}_predictions.npz') as z:
                y=z['target']
                for group in GROUPS:
                    raw=np.stack([z['ridge512'] if group=='ridge512' else z[f'{group}_s{s}'] for s in seeds])
                    scores[group],n=metrics(raw,y,lower);count+=n
                with np.load(run.CONFIRMATION/mouse/'later_predictions.npz') as old:
                    for key in old.files:np.testing.assert_array_equal(z[key],old[key]);checks+=1
            rows.append(dict(mouse=mouse,n=len(y),scores=scores,initial_mse=float(np.mean(y**2))))
        summaries={}
        for control in CONTROLS:
            mse=[1-v['scores']['large_attention']['mse']/v['scores'][control]['mse'] for v in rows]
            mae=[1-v['scores']['large_attention']['mae']/v['scores'][control]['mae'] for v in rows]
            a=np.array([v['scores']['large_attention']['single_mse'] for v in rows]);b=np.array([v['scores'][control]['single_mse'] for v in rows]);wins=int(np.sum(a<b))
            passed=np.mean(mse)>=.05 and sum(x>0 for x in mse)>=3 and wins>=math.ceil(8*len(seeds)/3) and min(mse)>=-.1 and np.mean(mae)>=0
            summaries['large_attention_vs_'+control]=dict(mean_mse_gain=float(np.mean(mse)),mean_mae_gain=float(np.mean(mae)),
                mouse_mse_gains=mse,mouse_mae_gains=mae,mouse_wins=sum(x>0 for x in mse),seed_wins=wins,full_pass=bool(passed),
                leave_one_mouse_out=[float(np.mean(mse[:i]+mse[i+1:])) for i in range(4)],
                individual_seeds=[dict(seed=seed,mean_mse_gain=float(np.mean(1-a[:,i]/b[:,i])),mouse_wins=int(np.sum(a[:,i]<b[:,i]))) for i,seed in enumerate(seeds)])
        initial=sum(sum(x<v['initial_mse'] for x in v['scores']['large_attention']['single_mse']) for v in rows)
        primary=all(summaries['large_attention_vs_'+c]['full_pass'] for c in CONTROLS[:3]) and initial>=math.ceil(8*len(seeds)/3)
        results[subset]=dict(seeds=seeds,rows=rows,summaries=summaries,initial_wins=initial,primary_pass=primary,pairs_per_mouse=math.comb(len(seeds),2))
    prior=run.read(run.ROOT/'stage1_results.json')['subsets']['stage1']['primary_pass'] if stage==2 else True
    passed=prior and all(v['primary_pass'] for v in results.values())
    run.write(run.ROOT/f'stage{stage}_results.json',dict(subsets=results,stage_passed=passed,full_replication_passed=stage==2 and passed,independent_animal_significance=False))
    run.write(run.ROOT/f'stage{stage}_audit.json',dict(passed=True,scalar_errors_checked=count,archive_checks=checks,selection_scores_checked=locked['selection_scores_checked'],
        frozen_source_hashes=len(run.verify()['hashes']),selected_artifacts=len(locked['hashes']),new_predictions=ev['new_neural_predictions']))
    for name,part in results.items():print(name,part['summaries'],flush=True)
    print('Stage',stage,'pass',passed,'replication',stage==2 and passed,flush=True)


if __name__=='__main__':main(int(sys.argv[1]))
