import json
import math
from pathlib import Path
import numpy as np
import torch
import search as s


def block_interval(difference, block=100, repeats=2000):
    rng=np.random.default_rng(834)
    n=len(difference)
    starts=rng.integers(n,size=(repeats,math.ceil(n/block)))
    indices=(starts[:,:,None]+np.arange(block))%n
    samples=difference[indices.reshape(repeats,-1)[:,:n]].mean(1)
    return {'mean_mse_advantage':float(difference.mean()),'interval_95':np.quantile(samples,[.025,.975]).tolist(),'block_bins':block,'resamples':repeats,'interpretation':'conditional exploratory interval for this recording and these fitted models; not independent-animal uncertainty'}


def main():
    root=s.ROOT
    freeze=json.loads((root/'frozen_selection.json').read_text())
    assert freeze['test_evaluated_before_freeze'] is False
    s.initialize()
    records=[json.loads(p.read_text()) for p in (root/'runs').glob('*.json')]
    groups={}
    predictions={}
    test_targets=None
    example_count=None
    for name in freeze['evaluate_names']:
        chosen=sorted([r for r in records if r['name']==name],key=lambda r:r['seed'] if r['seed'] is not None else -1)
        assert len(chosen)==freeze['expected_runs'][name]
        vp,tp=[],[]
        rows=[]
        for r in chosen:
            cfg=r['config']
            x,y,pos=s.get_data(cfg['n'],cfg['window'],'test')
            if test_targets is None:
                test_targets=y.numpy().copy()
                example_count=len(y)
            else:
                np.testing.assert_array_equal(test_targets,y.numpy())
            cached=np.load(root/'runs'/r['predictions'])
            validation_prediction=cached['validation_prediction']
            validation_target=cached['validation_target']
            if r['kind']=='neural':
                state=torch.load(root/'runs'/r['checkpoint'],map_location='cpu',weights_only=True)
                assert state['config']==cfg and state['neuron_ids']==s.IDS[:cfg['n']].tolist()
                assert state['speed_mean']==s.SPEED_MEAN and state['speed_std']==s.SPEED_STD
                model=s.Candidate(cfg)
                model.load_state_dict(state['state_dict'],strict=True)
                xv,yv,_=s.get_data(cfg['n'],cfg['window'],'val')
                restored=s.predict(model,xv,pos)
                np.testing.assert_allclose(restored,validation_prediction,rtol=1e-5,atol=1e-6)
                prediction=s.predict(model,x,pos)
            else:
                weights=np.load(root/'runs'/r['checkpoint'])
                prediction=(x.flatten(1).double() @ torch.from_numpy(weights['coef'])+float(weights['intercept'])).numpy()
                assert np.isfinite(prediction).all()
            vp.append(validation_prediction)
            tp.append(prediction)
            rows.append({'seed':r['seed'],'best_epoch':r.get('best_epoch'),'checkpoint':r['checkpoint'],'validation':s.metrics(validation_prediction,validation_target),'test':s.metrics(prediction,test_targets)})
        v=np.asarray(vp,dtype=np.float64)
        t=np.asarray(tp,dtype=np.float64)
        ensemble_val=v.mean(0)
        ensemble_test=t.mean(0)
        predictions[name]=ensemble_test
        result={'runs':rows,'count':len(rows),'parameters':chosen[0]['parameters'],'config':chosen[0]['config'],'mean_single_model':{},'ensemble':{'validation':s.metrics(ensemble_val,validation_target),'test':s.metrics(ensemble_test,test_targets)}}
        for split in ['validation','test']:
            result['mean_single_model'][split]={key:float(np.mean([r[split][key] for r in rows])) for key in rows[0][split]}
            result['mean_single_model'][split]['mse_sample_std']=float(np.std([r[split]['mse'] for r in rows],ddof=1)) if len(rows)>1 else 0.
        groups[name]=result
        np.savez_compressed(root/f'final_{name}_predictions.npz',seeds=np.array([r['seed'] if r['seed'] is not None else -1 for r in rows]),validation_predictions=v,validation_targets=validation_target,test_predictions=t,test_targets=test_targets)
        print(json.dumps({'name':name,'validation_mean_mse':result['mean_single_model']['validation']['mse'],'test_mean_mse':result['mean_single_model']['test']['mse'],'test_mean_r2':result['mean_single_model']['test']['r2'],'ensemble_test_mse':result['ensemble']['test']['mse']}),flush=True)
    best=freeze['selected_neural']
    floor=-s.SPEED_MEAN/s.SPEED_STD
    errors={name:(np.maximum(p,floor)-test_targets)**2 for name,p in predictions.items()}
    intervals={}
    for baseline in freeze['bootstrap_baselines']:
        intervals[baseline]=block_interval(errors[baseline]-errors[best])
    y_speed=test_targets*s.SPEED_STD+s.SPEED_MEAN
    train_speed=s.SPEED[s.COMMON_START:s.SLICES['train'][1]]
    threshold=float(np.quantile(train_speed[train_speed>0],.9))
    masks={'stationary':y_speed<1e-5,'moving':y_speed>=1e-5,'high_speed_above_training_positive_p90':y_speed>=threshold}
    strata={label:{'count':int(mask.sum()),'metrics':{name:s.metrics(p[mask],test_targets[mask]) for name,p in predictions.items()}} for label,mask in masks.items() if mask.sum()>1}
    constants={}
    for split in ['val','test']:
        _,y,_=s.get_data(128,8,split)
        constants[split]={'training_mean':s.metrics(np.zeros(len(y)),y.numpy()),'zero_speed':s.metrics(np.full(len(y),floor),y.numpy())}
    result={'selection':freeze,'test_examples':example_count,'speed_scaler':{'mean':s.SPEED_MEAN,'std':s.SPEED_STD},'groups':groups,'constant_baselines':constants,'ensemble_block_bootstrap':intervals,'ensemble_error_by_speed':strata,'high_speed_threshold':threshold}
    s.write_json(root/'final_results.json',result)
    selected=groups[best]
    best_seed=freeze['selected_single_seed']
    selected_row=next(r for r in selected['runs'] if r['seed']==best_seed)
    state=torch.load(root/'runs'/selected_row['checkpoint'],map_location='cpu',weights_only=True)
    ids=s.IDS[:selected['config']['n']]
    raw=s.RAW[ids,:s.SLICES['train'][1]]
    coordinates=s.POS[ids]
    state['preprocessing']={'activity_mean':torch.from_numpy(raw.mean(1,keepdims=True)),'activity_std':torch.from_numpy(np.maximum(raw.std(1,keepdims=True),1e-6)),'normalized_positions':torch.tensor((coordinates-coordinates.mean(0,keepdims=True))/np.maximum(coordinates.std(0,keepdims=True),1e-6),dtype=torch.float32),'input_order':'batch, selected_neuron, oldest_to_latest_time','speed_floor':0.,'speed_units':'arbitrary dataset units'}
    torch.save(state,root/'best_model.pt')
    print('FINAL_EVALUATION_COMPLETE',flush=True)


if __name__=='__main__':
    main()
