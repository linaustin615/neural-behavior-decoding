"""Compare all ensemble pairings without treating pairs as new animals."""
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('replication',ROOT/'run.py')
run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
LABELS=['attention_single','mlp_single','attention_pair','mlp_pair','mixed_cross','mixed_same']


def contrast(rows,control):
    gains=[1-r['mixed_cross']/max(r[control],1e-15) for r in rows]
    result=dict(control=control,mean_relative_gain=float(np.mean(gains)),mouse_gains=gains,
        mouse_wins=sum(g>0 for g in gains),leave_one_mouse_out=[float(np.mean(np.delete(gains,i))) for i in range(4)])
    if control in ['attention_pair','mlp_pair']:
        wins=sum(a<b for r in rows for a,b in zip(r['mixed_cross_pair_mse'],r[control+'_pair_mse']))
        count=sum(len(r['mixed_cross_pair_mse']) for r in rows)
        result.update(mouse_pair_wins=wins,mouse_pair_comparisons=count)
        if count==12:result['threshold_pass']=bool(np.mean(gains)>=.02 and sum(g>0 for g in gains)>=3 and wins>=8)
    return result


def bootstrap(errors,seeds):
    rng=np.random.default_rng(81219);pairs=list(itertools.combinations(range(len(seeds)),2));draws=[]
    for _ in range(2000):
        weights=rng.dirichlet(np.ones(len(seeds)))
        pair_weights=np.array([weights[i]*weights[j] for i,j in pairs]);pair_weights/=pair_weights.sum()
        changes=[]
        for mouse in rng.integers(0,4,4):
            e=errors[mouse];n=e['mixed_cross'].shape[1]
            times=((rng.integers(0,n,(n+99)//100)[:,None]+np.arange(100))%n).ravel()[:n]
            mse={name:float(np.sum(v[:,times].mean(1)*pair_weights)) for name,v in e.items()}
            changes.append([1-mse['mixed_cross']/max(mse[c],1e-15) for c in ['attention_pair','mlp_pair']])
        draws.append(np.mean(changes,axis=0))
    return {name:np.quantile(np.array(draws)[:,i],[.0125,.9875]).tolist() for i,name in enumerate(['attention_pair','mlp_pair'])}


def analyse(cohort,seeds):
    rows=[];errors=[];checked=0
    for m in run.MICE:
        meta=run.read(run.reference.FAIR/m/'metadata.json');lower=-meta['speed_mean']/meta['speed_std']
        path=(run.SHARED/m/'later_predictions.npz') if cohort=='old' else ROOT/m/'later_predictions.npz'
        with np.load(path) as z:
            y=z['target']
            prefix='shared_' if cohort=='old' else ''
            a=np.stack([np.maximum(z[f'{prefix}attention_s{s}'].astype(np.float64),lower) for s in seeds])
            b=np.stack([np.maximum(z[f'{prefix}mlp_s{s}'].astype(np.float64),lower) for s in seeds])
        e,pairs=run.ensemble_errors(a,b,y)
        row=dict(mouse=m,n=len(y),seeds=seeds,seed_pairs=[[seeds[i],seeds[j]] for i,j in pairs])
        for name in LABELS:
            row[name]=float(e[name].mean());row[name+'_pair_mse']=e[name].mean(1).tolist()
        for p,(i,j) in enumerate(pairs):
            formulas={'attention_pair':[(a[i]+a[j])/2],'mlp_pair':[(b[i]+b[j])/2],
                      'mixed_cross':[(a[i]+b[j])/2,(a[j]+b[i])/2]}
            for name,values in formulas.items():
                manual=np.mean([sum((float(v)-float(t))**2 for v,t in zip(q,y))/len(y) for q in values])
                assert abs(manual-row[name+'_pair_mse'][p])<1e-12*max(1,manual);checked+=1
        same_disagreement=float(np.mean((a-b)**2))
        cross_disagreement=float(np.mean([(a[i]-b[j])**2 for i in range(len(seeds)) for j in range(len(seeds)) if i!=j]))
        difference=(same_disagreement-cross_disagreement)/4
        assert abs(row['mixed_cross']-row['mixed_same']-difference)<1e-12
        correlations=[]
        for i in range(len(seeds)):
            for j in range(len(seeds)):
                ea,eb=a[i]-y,b[j]-y
                value=float(np.corrcoef(ea,eb)[0,1]) if ea.std()>1e-12 and eb.std()>1e-12 else None
                correlations.append(dict(attention_seed=seeds[i],mlp_seed=seeds[j],same_seed=i==j,correlation=value))
        row.update(same_seed_disagreement=same_disagreement,cross_seed_disagreement=cross_disagreement,
                   mean_same_seed_error_correlation=float(np.mean([c['correlation'] for c in correlations if c['same_seed'] and c['correlation'] is not None])),
                   mean_cross_seed_error_correlation=float(np.mean([c['correlation'] for c in correlations if not c['same_seed'] and c['correlation'] is not None])))
        row['all_cross_orientation_mse']=e['mixed_orientations'].mean(2).tolist()
        initial=float(np.mean((np.maximum(np.zeros(len(y)),lower)-y)**2))
        row['initial_mse']=initial
        row['attention_initial_wins']=int(np.sum(e['attention_single'].mean(1)<initial))
        row['mlp_initial_wins']=int(np.sum(e['mlp_single'].mean(1)<initial))
        errors.append({name:e[name] for name in ['mixed_cross','attention_pair','mlp_pair']});rows.append(row)
    comparisons={name:contrast(rows,name) for name in ['attention_pair','mlp_pair','mixed_same','attention_single','mlp_single']}
    result=dict(cohort=cohort,seeds=seeds,rows=rows,comparisons=comparisons,
        independent_pair_score_checks=checked,pairing_error_identity=True,
        learning_vs_initial={family:sum(r[family+'_initial_wins'] for r in rows) for family in ['attention','mlp']})
    if cohort=='new':
        result['overall_gate']=all(comparisons[c]['threshold_pass'] for c in ['attention_pair','mlp_pair'])
        result['descriptive_97_5_intervals']=bootstrap(errors,seeds)
    return result


def main():
    run.verify()
    analysis_lock=run.read(ROOT/'analysis_lock.json')
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==analysis_lock['source_sha256']
    if sys.argv[1]=='old':
        assert not (ROOT/'old_pairing_context.json').exists()
        result=analyse('old',run.OLD_SEEDS)
        run.write(ROOT/'old_pairing_context.json',result)
        print(json.dumps(result['comparisons'],indent=2))
    else:
        assert run.read(ROOT/'audit.json')['passed'] and not (ROOT/'summary.json').exists()
        result=analyse('new',run.SEEDS)
        combined=analyse('combined',run.OLD_SEEDS+run.SEEDS)
        run.write(ROOT/'summary.json',result);run.write(ROOT/'combined_context.json',combined)
        run.write(ROOT/'ensemble_audit.json',dict(passed=True,new_pair_metrics=result['independent_pair_score_checks'],
            combined_pair_metrics=combined['independent_pair_score_checks'],pair_orientation_error_averaging=True,
            error_difference_identity=True,old_and_new_pairs_not_independent_animals=True,
            seed_level_bootstrap_keeps_diagonal_pairs_excluded=True))
        print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
