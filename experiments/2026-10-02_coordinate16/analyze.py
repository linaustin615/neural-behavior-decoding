import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = Path('/Users/austinlin/neuron_transformer')
protocol = json.loads((ROOT/'protocol.json').read_text())
conditions, pools, seeds = (protocol[k] for k in ['conditions', 'pool_seeds', 'optimizer_seeds'])
rows = [json.loads(p.read_text()) for p in sorted((ROOT/'runs').glob('*.json'))]
lookup = {(r['task'].get('condition','real'),r['task']['pool'],r['task']['seed']):r for r in rows}
assert len(rows) == len(lookup) == 36
floor = -2.927973651537748 / 7.422715803805266
errors, raw_errors, targets = {}, {}, {}
for split in ['validation','test']:
    e, raw = np.empty((3,6,2,1273 if split=='validation' else 1323)), []
    raw = np.empty_like(e)
    for ci,c in enumerate(conditions):
        for pi,p in enumerate(pools):
            for si,s in enumerate(seeds):
                r = lookup[c,p,s]
                with np.load(ROOT/'runs'/r['predictions']) as saved:
                    pred = saved[split+'_prediction'].astype(np.float64)
                    target = saved[split+'_target'].astype(np.float64)
                assert np.isfinite(pred).all() and np.isfinite(target).all()
                if split not in targets: targets[split] = target
                np.testing.assert_array_equal(target,targets[split])
                e[ci,pi,si] = (np.maximum(pred,floor)-target)**2
                raw[ci,pi,si] = (pred-target)**2
                assert abs(e[ci,pi,si].mean()-r[split]['mse']) < 1e-10
                assert abs(raw[ci,pi,si].mean()-r[split]['raw_mse']) < 1e-10
                assert abs(1-e[ci,pi,si].mean()/target.var()-r[split]['r2']) < 1e-10
                assert r['best_epoch']==min(r['history'],key=lambda h:h['mse'])['epoch']
                assert abs(r['validation']['mse']-min(h['mse'] for h in r['history'])) < 1e-10
                assert r['parameters']==r['trainable_parameters']==81377
    errors[split], raw_errors[split] = e, raw
for p in pools:
    for s in seeds:
        matched = [lookup[c,p,s] for c in conditions]
        assert len({r['initial_state_sha256'] for r in matched})==1
        assert all(r['config']==matched[0]['config'] for r in matched)
        assert all(r['examples']==matched[0]['examples'] for r in matched)
for name,digest in json.loads((ROOT/'frozen_implementation.json').read_text())['sha256'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
for name,digest in protocol['source_hashes'].items():
    assert hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==digest,name
rng = np.random.default_rng(20261002)
boot = np.empty((2000,3))
length = errors['test'].shape[-1]
for b in range(2000):
    pp = rng.integers(0,len(pools),len(pools))
    ss = rng.integers(0,len(seeds),len(seeds))
    starts = rng.integers(0,length,int(np.ceil(length/100)))
    tt = ((starts[:,None]+np.arange(100))%length).ravel()[:length]
    boot[b] = errors['test'][:,pp][:,:,ss].mean((1,2))[:,tt].mean(-1)
result = dict(protocol=protocol,conditions={},contrasts={})
for ci,c in enumerate(conditions):
    result['conditions'][c] = {}
    for split in ['validation','test']:
        result['conditions'][c][split] = dict(mse=float(errors[split][ci].mean()),raw_mse=float(raw_errors[split][ci].mean()),r2=float(1-errors[split][ci].mean()/targets[split].var()),pair_mse=errors[split][ci].mean(-1).tolist())
for ci,c in enumerate(conditions[1:],1):
    delta = (errors['test'][ci]-errors['test'][0]).mean(-1)
    pool_means = delta.mean(1)
    result['contrasts'][c] = dict(mean_mse_improvement=float(delta.mean()),relative_improvement_percent=100*float(delta.mean()/errors['test'][ci].mean()),positive_pairs=int((delta>0).sum()),pairs=12,positive_pool_means=int((pool_means>0).sum()),pool_means=dict(zip(map(str,pools),map(float,pool_means))),interval95=np.quantile(boot[:,ci]-boot[:,0],[.025,.975]).tolist(),raw_mean_mse_improvement=float((raw_errors['test'][ci]-raw_errors['test'][0]).mean()),validation_mean_improvement=float((errors['validation'][ci]-errors['validation'][0]).mean()))
result['candidate_gate'] = all(c['mean_mse_improvement']>0 and c['positive_pairs']>=8 for c in result['contrasts'].values())
result['stronger_within_recording_gate'] = all(c['interval95'][0]>0 and c['positive_pool_means']>=5 for c in result['contrasts'].values())
result['learning_check'] = dict(beat_untrained=sum(r['test']['mse']<r['untrained_test']['mse'] for r in rows),train_mean_baseline_mse=float(np.mean(targets['test']**2)),beat_train_mean=int(sum(r['test']['mse']<np.mean(targets['test']**2) for r in rows)),final_epoch_selected=sum(r['best_epoch']==24 for r in rows),new_fit_seconds=sum(r['seconds'] for r in rows if r['task'].get('condition','real')!='real'))
reliance = json.loads((ROOT/'reliance_results.json').read_text())
assert reliance['complete'] and len(reliance['runs'])==12
result['reliance'] = {}
for c in ['none','shuffled']:
    summary = {}
    for split,short in [('validation','val'),('test','test')]:
        items = [r['conditions'][c][short] for r in reliance['runs']]
        for r,item in zip(reliance['runs'],items):
            saved = np.load(ROOT/f"reliance_p{r['pool']}_s{r['seed']}_{c}.npz")
            np.testing.assert_array_equal(saved[split+'_target'],targets[split])
            for state in ['original','perturbed','compensated']:
                pred = saved[split+'_'+state].astype(np.float64)
                assert np.isfinite(pred).all()
                mse = np.mean((np.maximum(pred,floor)-targets[split])**2)
                assert abs(mse-item[state]['mse'])<1e-10
            np.testing.assert_allclose(saved[split+'_original'],saved[split+'_compensated'],atol=3e-6,rtol=1e-5)
        original = float(np.mean([i['original']['mse'] for i in items]))
        changed = float(np.mean([i['perturbed']['mse'] for i in items]))
        summary[split] = dict(original_mse=original,perturbed_mse=changed,mse_increase=changed-original,percent_increase=100*(changed-original)/original,positive_increases=sum(i['perturbed']['mse']>i['original']['mse'] for i in items),max_compensated_prediction_error=max(i['compensated_max_prediction_error'] for i in items))
    result['reliance'][c] = summary
checks = [json.loads(f.read_text()) for f in (ROOT/'audits').glob('*.json')]
assert len(checks)==36
result['audit'] = dict(records=36,metrics_recomputed=True,checkpoint_selection_correct=True,finite_predictions=True,matched_initial_weights_and_configuration=True,frozen_sources_and_application_hashes_match=True,checkpoint_reloads=36,full_validation_and_test_reloads=sum(a['reload']['test']['examples']==1323 for a in checks),max_reload_prediction_error=max(v['max_prediction_error'] for a in checks for v in a['reload'].values()),reliance_metrics_recomputed=True)
(ROOT/'results.json').write_text(json.dumps(result,indent=2))
(ROOT/'completion_audit.json').write_text(json.dumps(result['audit'],indent=2))
with (ROOT/'per_run.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['condition','pool','seed','validation_mse','test_mse','test_r2','best_epoch','epochs_run','seconds'])
    writer.writeheader()
    for r in rows:
        writer.writerow(dict(condition=r['task'].get('condition','real'),pool=r['task']['pool'],seed=r['task']['seed'],validation_mse=r['validation']['mse'],test_mse=r['test']['mse'],test_r2=r['test']['r2'],best_epoch=r['best_epoch'],epochs_run=r['epochs_run'],seconds=r['seconds']))
print(json.dumps({k:v for k,v in result.items() if k!='protocol'},indent=2))
