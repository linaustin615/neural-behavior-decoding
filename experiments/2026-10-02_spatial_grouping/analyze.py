import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
protocol = json.loads((ROOT/'protocol.json').read_text())
variants, pools, seeds = (protocol[k] for k in ['variants','pool_seeds','optimizer_seeds'])
rows = [json.loads(p.read_text()) for p in sorted((ROOT/'runs').glob('*.json'))]
assert len(rows) == 48, len(rows)
lookup = {(r['task']['variant'],r['task']['pool'],r['task']['seed']):r for r in rows}
assert len(lookup) == 48
floor = -2.927973651537748 / 7.422715803805266
errors, raw_errors = {}, {}
audit = dict(records=48, metrics_recomputed=True, checkpoint_selection_correct=True,
             finite_predictions=True, matched_weights=True, matched_parameter_counts=True)
for split in ['validation','test']:
    target0 = None
    error, raw = [], []
    for variant in variants:
        pe, pr = [], []
        for pool in pools:
            se, sr = [], []
            for seed in seeds:
                record = lookup[variant,pool,seed]
                data = np.load(ROOT/'runs'/record['predictions'])
                pred = data[split+'_prediction'].astype(np.float64)
                target = data[split+'_target'].astype(np.float64)
                assert np.isfinite(pred).all()
                if target0 is None:
                    target0 = target
                np.testing.assert_array_equal(target,target0)
                e = (np.maximum(pred,floor)-target)**2
                er = (pred-target)**2
                assert abs(e.mean()-record[split]['mse']) < 1e-10
                assert abs(er.mean()-record[split]['raw_mse']) < 1e-10
                assert record['best_epoch'] == min(record['history'],key=lambda h:h['mse'])['epoch']
                se.append(e);sr.append(er)
            pe.append(se);pr.append(sr)
        error.append(pe);raw.append(pr)
    errors[split]=np.array(error);raw_errors[split]=np.array(raw)
for pool in pools:
    for seed in seeds:
        matched=[lookup[v,pool,seed] for v in variants]
        assert len({r['initial_state_sha256'] for r in matched})==1
        assert len({r['parameters'] for r in matched})==1
        assert len({r['trainable_parameters'] for r in matched})==1
for name,digest in json.loads((ROOT/'frozen_implementation.json').read_text())['sha256'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
for name,digest in protocol['source_hashes'].items():
    assert hashlib.sha256((Path('/Users/austinlin/neuron_transformer')/name).read_bytes()).hexdigest()==digest,name
audit['frozen_sources_and_application_hashes_match']=True
rng=np.random.default_rng(20261002)
boot=np.empty((2000,len(variants)))
length=errors['test'].shape[-1]
for b in range(2000):
    pp=rng.integers(0,len(pools),len(pools))
    ss=rng.integers(0,len(seeds),len(seeds))
    starts=rng.integers(0,length,int(np.ceil(length/100)))
    tt=((starts[:,None]+np.arange(100))%length).ravel()[:length]
    boot[b]=errors['test'][:,pp][:,:,ss].mean((1,2))[:,tt].mean(-1)
result=dict(protocol=protocol,groups={},contrasts={},audit=audit)
for vi,v in enumerate(variants):
    group={}
    for split in ['validation','test']:
        group[split+'_mse']=float(errors[split][vi].mean())
        group[split+'_raw_mse']=float(raw_errors[split][vi].mean())
        group[split+'_r2']=float(np.mean([lookup[v,p,s][split]['r2'] for p in pools for s in seeds]))
        group[split+'_pair_mse']=errors[split][vi].mean(-1).tolist()
    group['parameters']=lookup[v,pools[0],seeds[0]]['parameters']
    result['groups'][v]=group
sp=variants.index('spatial')
for control in ['random','depth_random','global']:
    ci=variants.index(control)
    d=(errors['test'][ci]-errors['test'][sp]).mean(-1)
    pool_means=d.mean(1)
    contrast=dict(mean_mse_improvement=float(d.mean()),relative_improvement_percent=100*float(d.mean()/errors['test'][ci].mean()),
                  positive_pairs=int((d>0).sum()),pairs=12,positive_pool_means=int((pool_means>0).sum()),
                  pool_means={str(p):float(m) for p,m in zip(pools,pool_means)},
                  interval95=np.quantile(boot[:,ci]-boot[:,sp],[.025,.975]).tolist(),
                  raw_mean_mse_improvement=float((raw_errors['test'][ci]-raw_errors['test'][sp]).mean()),
                  validation_mean_improvement=float((errors['validation'][ci]-errors['validation'][sp]).mean()))
    result['contrasts'][control]=contrast
result['candidate_gate']=all(c['mean_mse_improvement']>0 and c['positive_pairs']>=8 for c in result['contrasts'].values())
result['stronger_within_recording_gate']=all(c['interval95'][0]>0 and c['positive_pool_means']>=5 for c in result['contrasts'].values())
result['learning_check']=dict(beat_untrained=sum(r['test']['mse']<r['untrained_test']['mse'] for r in rows),
                             beat_train_mean=sum(r['test']['mse']<2.4612878299 for r in rows),
                             final_epoch_selected=sum(r['best_epoch']==24 for r in rows),
                             total_fit_seconds=sum(r['seconds'] for r in rows))
old=[]
for pool in pools:
    for seed in seeds:
        old.append(json.loads((ROOT/'baseline_reference'/f'baseline_n2048_p{pool}_s{seed}_real.json').read_text()))
result['previous8_summary_reference']=dict(test_mse=float(np.mean([r['test']['mse'] for r in old])),
                                          validation_mse=float(np.mean([r['validation']['mse'] for r in old])),
                                          note='Reused exploratory comparison;8 queries versus16 and256 fewer parameters')
(ROOT/'results.json').write_text(json.dumps(result,indent=2))
(ROOT/'integrity_audit.json').write_text(json.dumps(audit,indent=2))
with (ROOT/'per_run.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['variant','pool','seed','val_mse','test_mse','test_r2','best_epoch','seconds'])
    writer.writeheader()
    for r in rows:
        writer.writerow(dict(variant=r['task']['variant'],pool=r['task']['pool'],seed=r['task']['seed'],
                             val_mse=r['validation']['mse'],test_mse=r['test']['mse'],test_r2=r['test']['r2'],
                             best_epoch=r['best_epoch'],seconds=r['seconds']))
print(json.dumps(result,indent=2))
