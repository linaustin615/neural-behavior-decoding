"""Fixed static-readout replication and consistency checks."""
import math
import numpy as np
import run

metrics=run.previous.shared.metrics if hasattr(run.previous,'shared') else run.module('static_metrics',run.previous.FT/'analyse.py').metrics
GROUPS=['static','native_attention','native_mlp','tuned_mlp','ridge']
CONTROLS=['native_attention','native_mlp','ridge','tuned_mlp']


def main():
    locked=run.verify_lock();evaluation=run.read(run.ROOT/'evaluation.json')
    for name,value in evaluation['prediction_hashes'].items():assert run.digest(run.ROOT/name)==value
    result={};count=checks=0
    for subset,seeds in [('latest',[16,17,18]),('additional',run.SEEDS),('all',run.ALL_SEEDS)]:
        rows=[]
        for mouse in run.MICE:
            meta=run.read(run.ref.FAIR/mouse/'metadata.json');lower=-meta['speed_mean']/meta['speed_std'];scores={}
            with np.load(run.ROOT/mouse/'later_predictions.npz') as z:
                y=z['target']
                for group in GROUPS:
                    raw=np.stack([z['ridge'] if group=='ridge' else z[f'{group}_s{s}'] for s in seeds])
                    scores[group],n=metrics(raw,y,lower);count+=n
                with np.load(run.CONFIRMATION/mouse/'later_predictions.npz') as old:
                    for key in old.files:np.testing.assert_array_equal(z[key],old[key]);checks+=1
                with np.load(run.COMPONENTS/mouse/'later_predictions.npz') as old:
                    np.testing.assert_array_equal(y,old['target'])
                    for seed in [10,11,12]:np.testing.assert_array_equal(z[f'static_s{seed}'],old[f'as_s{seed}']);checks+=1
            rows.append(dict(mouse=mouse,n=len(y),scores=scores,initial_mse=float(np.mean(y**2))))
        summaries={}
        for control in CONTROLS:
            mse=[1-v['scores']['static']['mse']/v['scores'][control]['mse'] for v in rows]
            mae=[1-v['scores']['static']['mae']/v['scores'][control]['mae'] for v in rows]
            sa=np.array([v['scores']['static']['single_mse'] for v in rows]);sb=np.array([v['scores'][control]['single_mse'] for v in rows])
            wins=int(np.sum(sa<sb))
            s=dict(mean_mse_gain=float(np.mean(mse)),mean_mae_gain=float(np.mean(mae)),mouse_mse_gains=mse,mouse_mae_gains=mae,
                mouse_wins=sum(x>0 for x in mse),seed_wins=wins,leave_one_mouse_out=[float(np.mean(mse[:i]+mse[i+1:])) for i in range(4)],
                individual_seeds=[dict(seed=seed,mean_mse_gain=float(np.mean(1-sa[:,i]/sb[:,i])),mouse_wins=int(np.sum(sa[:,i]<sb[:,i]))) for i,seed in enumerate(seeds)])
            s['full_pass']=s['mean_mse_gain']>=.05 and s['mouse_wins']>=3 and wins>=math.ceil(8*len(seeds)/3) and min(mse)>=-.1 and s['mean_mae_gain']>=0
            independent=sum(sum(x<y for x,y in zip(v['scores']['static']['single_mse'],v['scores'][control]['single_mse'])) for v in rows)
            assert wins==independent
            assert s['full_pass']==(sum(mse)/4>=.05 and sum(x>0 for x in mse)>=3 and independent>=math.ceil(8*len(seeds)/3) and all(x>=-.1 for x in mse) and sum(mae)/4>=0);checks+=2
            summaries['static_vs_'+control]=s
        initial=sum(sum(x<v['initial_mse'] for x in v['scores']['static']['single_mse']) for v in rows)
        gate=all(summaries['static_vs_'+c]['full_pass'] for c in CONTROLS[:3]) and initial>=math.ceil(8*len(seeds)/3)
        result[subset]=dict(seeds=seeds,rows=rows,summaries=summaries,initial_wins=initial,primary_pass=gate,pairs_per_mouse=math.comb(len(seeds),2))
    decision=all(part['primary_pass'] for part in result.values())
    run.write(run.ROOT/'results.json',dict(subsets=result,practical_gate_passed=decision,new_fits=6,reused_static_fits=3,independent_animal_significance=False))
    run.write(run.ROOT/'audit.json',dict(passed=True,scalar_errors_checked=count,aggregate_archive_checks=checks,
        selection_scores_checked=locked['selection_scores_checked'],frozen_source_hashes=len(run.verify()['hashes']),
        selected_artifact_hashes=len(locked['hashes']),new_predictions=evaluation['new_predictions'],main_application_unchanged=True))
    for name,part in result.items():print(name,part['summaries'],flush=True)
    print('Static replication gate:',decision,flush=True)


if __name__=='__main__':main()
