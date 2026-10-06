import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
import numpy as np
import base_search as s

ROOT=Path(__file__).resolve().parent
COUNTS=[128,512,2048]
POOLS=[101,202,303]
SEEDS=[10,11]


def describe(values):
    a=np.asarray(values,dtype=float)
    return {'mean':float(a.mean()),'sd':float(a.std(ddof=1)) if len(a)>1 else 0.,'values':a.tolist(),'positive_count':int((a>0).sum()),'count':len(a)}


def main():
    expected=json.loads((ROOT/'tasks.json').read_text())
    records=[json.loads(p.read_text()) for p in (ROOT/'runs').glob('*.json')]
    key=lambda t:(t['n'],t['pool'],t['seed'],t['condition'])
    assert len(records)==len(expected)==66
    assert set(key(r['task']) for r in records)==set(key(t) for t in expected)
    by={key(r['task']):r for r in records}
    results={'counts':COUNTS,'pool_seeds':POOLS,'training_seeds':SEEDS,'runs':len(records),'summary':{},'paired':{},'density_interaction':{},'integrity':{}}
    for n in COUNTS:
        conditions=['real','shuffled','none']+(['depth','within_depth_shuffled'] if n==2048 else [])
        results['summary'][n]={}
        for condition in conditions:
            group=[by[n,p,z,condition] for p in POOLS for z in SEEDS]
            summary={'n':n,'condition':condition,'fits':len(group),'mean_fit_seconds':statistics.mean(r['seconds'] for r in group)}
            for split in ['validation','test']:
                summary[split]={metric:describe([r[split][metric] for r in group]) for metric in ['mse','raw_mse','r2','rmse_speed_units','mae_speed_units']}
            summary['pool_means']={p:{split:statistics.mean(by[n,p,z,condition][split]['mse'] for z in SEEDS) for split in ['validation','test']} for p in POOLS}
            results['summary'][n][condition]=summary
        results['paired'][n]={}
        for control in [c for c in conditions if c!='real']:
            result={}
            for split in ['validation','test']:
                diffs=[by[n,p,z,control][split]['mse']-by[n,p,z,'real'][split]['mse'] for p in POOLS for z in SEEDS]
                raw=[by[n,p,z,control][split]['raw_mse']-by[n,p,z,'real'][split]['raw_mse'] for p in POOLS for z in SEEDS]
                result[split]={'mse_advantage':describe(diffs),'raw_mse_advantage':describe(raw),'pool_mean_advantage':{p:statistics.mean(by[n,p,z,control][split]['mse']-by[n,p,z,'real'][split]['mse'] for z in SEEDS) for p in POOLS}}
            results['paired'][n][control]=result
    for control in ['shuffled','none']:
        results['density_interaction'][control]={}
        for split in ['validation','test']:
            high=results['paired'][2048][control][split]['mse_advantage']['values']
            low=results['paired'][128][control][split]['mse_advantage']['values']
            delta=np.asarray(high)-low
            results['density_interaction'][control][split]={'mse_advantage_change':describe(delta),'pool_mean_change':{p:float(delta[i*2:i*2+2].mean()) for i,p in enumerate(POOLS)}}
    correct2048=all(results['paired'][2048][c]['test']['mse_advantage']['mean']>0 and all(v>0 for v in results['paired'][2048][c]['test']['pool_mean_advantage'].values()) for c in ['shuffled','none'])
    growing=correct2048 and all(all(v>0 for v in results['density_interaction'][c]['test']['pool_mean_change'].values()) for c in ['shuffled','none'])
    results['gates']={'consistent_spatial_advantage_at_2048':correct2048,'increasing_spatial_advantage_with_density':growing,'interpretation':'predeclared exploratory gates within one recording; a pass is not independent biological replication'}
    floor=-2.927973651537748/7.422715803805266
    error_differences={}
    for n in COUNTS:
        for control in ['shuffled','none']:
            differences=[]
            for pool in POOLS:
                for seed in SEEDS:
                    a=by[n,pool,seed,'real'];b=by[n,pool,seed,control]
                    real=np.load(ROOT/'runs'/a['predictions'])
                    counter=np.load(ROOT/'runs'/b['predictions'])
                    y=real['test_target'].astype(float)
                    np.testing.assert_array_equal(y,counter['test_target'])
                    e_real=(np.maximum(real['test_prediction'].astype(float),floor)-y)**2
                    e_counter=(np.maximum(counter['test_prediction'].astype(float),floor)-y)**2
                    differences.append(e_counter-e_real)
            error_differences[n,control]=np.mean(differences,axis=0)
    total=len(next(iter(error_differences.values())))
    rng=np.random.default_rng(280913)
    starts=rng.integers(total,size=(2000,math.ceil(total/100)))
    indices=((starts[:,:,None]+np.arange(100))%total).reshape(2000,-1)[:,:total]
    results['temporal_block_intervals']={'block_bins':100,'resamples':2000,'interpretation':'conditional on one recording and the fitted models; not uncertainty across animals','contrasts':{},'interactions':{}}
    for (n,control),difference in error_differences.items():
        resampled=difference[indices].mean(1)
        value={'mean':float(difference.mean()),'interval95':np.quantile(resampled,[.025,.975]).tolist()}
        assert abs(value['mean']-results['paired'][n][control]['test']['mse_advantage']['mean'])<1e-7
        results['temporal_block_intervals']['contrasts'][f'{n}_{control}']=value
    for control in ['shuffled','none']:
        diff=error_differences[2048,control]-error_differences[128,control]
        results['temporal_block_intervals']['interactions'][control]={'mean':float(diff.mean()),'interval95':np.quantile(diff[indices].mean(1),[.025,.975]).tolist()}
    for n in COUNTS:
        for pool in POOLS:
            for seed in SEEDS:
                group=[r for r in records if (r['task']['n'],r['task']['pool'],r['task']['seed'])==(n,pool,seed)]
                assert len(set(r['initial_state_sha256'] for r in group))==1
                assert len(set(r['parameters'] for r in group))==1
                for r in group:
                    assert r['best_epoch']==min(r['history'],key=lambda row:row['mse'])['epoch']
    protocol=json.loads((ROOT/'protocol.json').read_text())
    project=ROOT.parents[1]
    unchanged={name:hashlib.sha256((project/name).read_bytes()).hexdigest()==value for name,value in protocol['source_hashes'].items()}
    assert all(unchanged.values())
    results['integrity']={'all66_tasks_completed':True,'same_initialization_and_parameter_count_in_each_matched_comparison':True,'checkpoints_selected_only_by_validation':True,'source_files_unchanged':unchanged,'test_improved_over_own_untrained':sum(r['test']['mse']<r['untrained_test']['mse'] for r in records),'last_epoch_selected':[r['task'] for r in records if r['best_epoch']==24],'best_epochs':describe([r['best_epoch'] for r in records]),'total_fit_seconds':sum(r['seconds'] for r in records)}
    synthetic=json.loads((ROOT/'synthetic_results.json').read_text())
    syn={r['condition']:r['test']['r2'] for r in synthetic['runs']}
    results['synthetic_positive_control']={'test_r2':syn,'pass':all(syn['real']-syn[c]>.1 for c in ['shuffled','none']),'limitation':'single synthetic seed, large known effect; not evidence of spatial benefit in this neural recording'}
    s.write_json(ROOT/'results.json',results)
    csv_rows=[]
    for r in sorted(records,key=lambda r:key(r['task'])):
        csv_rows.append({**r['task'],'validation_mse':r['validation']['mse'],'test_mse':r['test']['mse'],'raw_validation_mse':r['validation']['raw_mse'],'raw_test_mse':r['test']['raw_mse'],'test_r2':r['test']['r2'],'best_epoch':r['best_epoch'],'epochs_run':r['epochs_run'],'fit_seconds':r['seconds'],'parameters':r['parameters']})
    with (ROOT/'per_run.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    print(json.dumps({'gates':results['gates'],'primary':{n:{c:round(results['summary'][n][c]['test']['mse']['mean'],5) for c in ['real','shuffled','none']} for n in COUNTS},'interactions':results['temporal_block_intervals']['interactions'],'integrity':results['integrity']},indent=2))


if __name__=='__main__':
    main()
