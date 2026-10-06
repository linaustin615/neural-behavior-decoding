import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import numpy as np
import base_search as s

ROOT=Path(__file__).resolve().parent
POOLS=[101,202,303,404,505,606]
SEEDS=[10,11,12]
CONDITIONS=['real','none','shuffled']


def summary(values):
    a=np.asarray(values,dtype=float)
    return {'mean':float(a.mean()),'sd':float(a.std(ddof=1)) if len(a)>1 else 0.,'count':len(a),'values':a.tolist()}


def main():
    tasks=json.loads((ROOT/'all_tasks.json').read_text())
    old_tasks=json.loads((ROOT/'reused_tasks.json').read_text())
    key=lambda t:(t['pool'],t['seed'],t['condition'])
    records=[json.loads(p.read_text()) for p in (ROOT/'runs').glob('*.json')]
    assert len(records)==len(tasks)==54
    assert set(key(r['task']) for r in records)==set(map(key,tasks))
    by={key(r['task']):r for r in records}
    old_pairs=sorted(set((t['pool'],t['seed']) for t in old_tasks))
    all_pairs=[(p,z) for p in POOLS for z in SEEDS]
    groups={'previous_trials':old_pairs,'new_trials':[a for a in all_pairs if a not in old_pairs],'fresh_seed_on_previous_pools':[(p,12) for p in POOLS[:3]],'new_neuron_pools':[(p,z) for p in POOLS[3:] for z in SEEDS],'combined':all_pairs}
    result={'new_fits':36,'reused_fits':18,'all_fits':54,'paired_comparisons':18,'groups':{},'per_pool':{},'conditional_intervals':{},'checks':{}}
    for group,pairs in groups.items():
        result['groups'][group]={'pairs':[list(a) for a in pairs],'conditions':{},'contrasts':{}}
        for c in CONDITIONS:
            result['groups'][group]['conditions'][c]={split:{metric:summary([by[p,z,c][split][metric] for p,z in pairs]) for metric in ['mse','raw_mse','r2','rmse_speed_units','mae_speed_units']} for split in ['validation','test']}
        for control in ['none','shuffled']:
            comparison={}
            for split in ['validation','test']:
                delta=np.array([by[p,z,control][split]['mse']-by[p,z,'real'][split]['mse'] for p,z in pairs])
                baseline=np.array([by[p,z,control][split]['mse'] for p,z in pairs])
                raw=np.array([by[p,z,control][split]['raw_mse']-by[p,z,'real'][split]['raw_mse'] for p,z in pairs])
                comparison[split]={'advantage':summary(delta),'raw_advantage':summary(raw),'wins':int((delta>0).sum()),'ties':int((delta==0).sum()),'relative_mean_mse_reduction_percent':100*float(delta.mean()/baseline.mean())}
            result['groups'][group]['contrasts'][control]=comparison
    for pool in POOLS:
        result['per_pool'][pool]={'previously_used_pool':pool in POOLS[:3],'conditions':{},'contrasts':{}}
        for c in CONDITIONS:
            result['per_pool'][pool]['conditions'][c]={split:summary([by[pool,z,c][split]['mse'] for z in SEEDS]) for split in ['validation','test']}
        for control in ['none','shuffled']:
            result['per_pool'][pool]['contrasts'][control]={split:summary([by[pool,z,control][split]['mse']-by[pool,z,'real'][split]['mse'] for z in SEEDS]) for split in ['validation','test']}
    errors={}
    targets={}
    floor=-2.927973651537748/7.422715803805266
    for split in ['validation','test']:
        for c in CONDITIONS:
            cube=[]
            for p in POOLS:
                rows=[]
                for z in SEEDS:
                    record=by[p,z,c]
                    data=np.load(ROOT/'runs'/record['predictions'])
                    y=data[split+'_target'].astype(float)
                    prediction=data[split+'_prediction'].astype(float)
                    if split in targets:np.testing.assert_array_equal(y,targets[split])
                    else:targets[split]=y.copy()
                    loss=(np.maximum(prediction,floor)-y)**2
                    assert abs(float(loss.mean())-record[split]['mse'])<1e-7
                    rows.append(loss)
                cube.append(rows)
            errors[split,c]=np.array(cube)
    rng=np.random.default_rng(6061213)
    repetitions=2000
    pool_draws=rng.integers(6,size=(repetitions,6))
    seed_draws=rng.integers(3,size=(repetitions,3))
    pool_weights=np.stack([(pool_draws==i).sum(1)/6 for i in range(6)],axis=1)
    seed_weights=np.stack([(seed_draws==i).sum(1)/3 for i in range(3)],axis=1)
    weights=pool_weights[:,:,None]*seed_weights[:,None,:]
    for split in ['validation','test']:
        length=len(targets[split])
        starts=rng.integers(length,size=(repetitions,math.ceil(length/100)))
        time_indices=((starts[:,:,None]+np.arange(100))%length).reshape(repetitions,-1)[:,:length]
        result['conditional_intervals'][split]={}
        for control in ['none','shuffled']:
            difference=errors[split,control]-errors[split,'real']
            curves=np.einsum('bps,pst->bt',weights,difference,optimize=False)
            draws=np.take_along_axis(curves,time_indices,axis=1).mean(1)
            entry={'mean':float(difference.mean()),'percentile_interval95':np.quantile(draws,[.025,.975]).tolist(),'pool_only_direction_count':sum(result['per_pool'][p]['contrasts'][control][split]['mean']>0 for p in POOLS)}
            result['conditional_intervals'][split][control]=entry
            assert abs(entry['mean']-result['groups']['combined']['contrasts'][control][split]['advantage']['mean'])<1e-7
    result['interval_method']={'resamples':2000,'time_block_bins':100,'description':'Crossed resampling of 6 pool blocks and 3 optimizer-seed blocks, plus circular temporal blocks, shared within each coordinate comparison. Pools overlap; all trials use one recording. Intervals are exploratory and conditional on this study, not independent-animal confidence intervals.'}
    result['consistency_gates']={}
    for split in ['validation','test']:
        result['consistency_gates'][split]={}
        for control in ['none','shuffled']:
            mean=result['groups']['combined']['contrasts'][control][split]['advantage']['mean']
            consistent=all(result['per_pool'][p]['contrasts'][control][split]['mean']>0 for p in POOLS)
            result['consistency_gates'][split][control]={'positive_overall_and_each_pool':bool(mean>0 and consistent),'conditional_interval_excludes_zero_in_positive_direction':result['conditional_intervals'][split][control]['percentile_interval95'][0]>0}
        result['consistency_gates'][split]['true_geometry_over_both_controls']=all(result['consistency_gates'][split][c]['positive_overall_and_each_pool'] for c in ['none','shuffled'])
    for p,z in all_pairs:
        rows=[by[p,z,c] for c in CONDITIONS]
        assert len(set(r['initial_state_sha256'] for r in rows))==1
        assert len(set(r['parameters'] for r in rows))==1
        for row in rows:
            assert row['best_epoch']==min(row['history'],key=lambda v:v['mse'])['epoch']
    protocol=json.loads((ROOT/'protocol.json').read_text())
    project=ROOT.parents[1]
    unchanged={name:hashlib.sha256((project/name).read_bytes()).hexdigest()==digest for name,digest in protocol['source_hashes'].items()}
    assert all(unchanged.values())
    result['checks']={'all54_tasks_present_once':True,'matched_initial_weights_and_parameter_counts':True,'selected_checkpoints_minimize_validation_loss':True,'prediction_arrays_reproduce_all_recorded_mse':True,'application_files_unchanged':unchanged,'test_improved_over_untrained':sum(r['test']['mse']<r['untrained_test']['mse'] for r in records),'last_epoch_selected':[r['task'] for r in records if r['best_epoch']==24],'new_fit_seconds':sum(r['seconds'] for r in records if (r['task']['pool'],r['task']['seed']) not in old_pairs)}
    result['compatibility_audit']=json.loads((ROOT/'compatibility_audit.json').read_text())
    s.write_json(ROOT/'results.json',result)
    csv_rows=[{**r['task'],'is_reused':(r['task']['pool'],r['task']['seed']) in old_pairs,'validation_mse':r['validation']['mse'],'test_mse':r['test']['mse'],'raw_validation_mse':r['validation']['raw_mse'],'raw_test_mse':r['test']['raw_mse'],'test_r2':r['test']['r2'],'best_epoch':r['best_epoch'],'epochs_run':r['epochs_run'],'fit_seconds':r['seconds']} for r in sorted(records,key=lambda r:key(r['task']))]
    with (ROOT/'per_run.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    print(json.dumps({'combined_test':{c:result['groups']['combined']['conditions'][c]['test']['mse'] for c in CONDITIONS},'new_trial_contrasts':result['groups']['new_trials']['contrasts'],'test_intervals':result['conditional_intervals']['test'],'consistency':result['consistency_gates']['test'],'checks':result['checks']},indent=2))


if __name__=='__main__':
    main()
