"""Separate risk weighting and neuron information, then lock a later replay."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import eigh
from scipy.linalg.blas import dsyrk
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
OLD = ROOT.parent / '2026-10-04_hybrid_complementarity'
spec = importlib.util.spec_from_file_location('previous_pilot', OLD / 'run.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
MICE, SEEDS, PAIRS = old.MICE, old.SEEDS, old.PAIRS
CONFIGS = ['absolute_coarse', 'balanced_coarse', 'absolute_neuron', 'balanced_neuron']


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def patch_features(x):
    assert x.shape[1:] == (128, 32)
    return x.reshape(len(x), 128, 4, 8).mean(-1, dtype=np.float64).reshape(len(x), 512)


def fit_pca(x):
    features = patch_features(x)
    mean, std = features.mean(0), np.maximum(features.std(0), 1e-6)
    z = (features - mean) / std
    upper = dsyrk(1 / len(z), np.asfortranarray(z.T), lower=0)
    covariance = np.triu(upper) + np.triu(upper, 1).T
    values, vectors = eigh(covariance, subset_by_index=[508, 511], check_finite=True)
    values, vectors = values[::-1].copy(), vectors[:, ::-1].copy()
    for i in range(4):
        if vectors[np.argmax(np.abs(vectors[:, i])), i] < 0:
            vectors[:, i] *= -1
    return dict(mean=mean, std=std, vectors=vectors, values=values,
                explained_fraction=values / np.trace(covariance))


def project(x, pca):
    z = (patch_features(x) - pca['mean']) / pca['std']
    return np.einsum('ni,ij->nj', z, pca['vectors'], optimize=False) / np.sqrt(np.maximum(pca['values'], 1e-8))


def objective(theta, f, a, b, y, w, scale):
    g = expit(np.sum(f * theta, axis=1))
    residual = a + g * (b - a) - y
    value = np.sum(w * residual ** 2) / scale + .01 * np.sum(theta[1:] ** 2)
    factors = 2 * w * residual * (b - a) * g * (1 - g) / scale
    grad = np.sum(f * factors[:, None], axis=0)
    grad[1:] += .02 * theta[1:]
    return float(value), grad


def matrix(raw, session, mean, std):
    return np.column_stack([np.ones(len(raw)), np.clip((raw - mean) / std, -5, 5),
                            *[session == s for s in [1, 2, 3]]])


def fit_rule(a, b, y, raw, session, balanced):
    denominator = np.array([max(float(np.mean(((a[session == s]-y[session == s])**2 +
                                (b[session == s]-y[session == s])**2) / 2)), 1e-4) for s in range(4)])
    w = np.empty(len(y))
    for s in range(4):
        mask = session == s
        w[mask] = 1 / (4 * mask.sum() * (denominator[s] if balanced else 1))
    alpha = old.weight(a, b, y, w)
    mean, std = raw.mean(0), np.maximum(raw.std(0), 1e-6)
    f = matrix(raw, session, mean, std)
    theta = np.zeros(f.shape[1])
    start = np.clip(alpha, .01, .99)
    theta[0] = np.log(start / (1 - start))
    scale = max(float(np.sum(w * (a - y) ** 2)), 1e-8)
    result = minimize(objective, theta, args=(f, a, b, y, w, scale), jac=True,
                      method='L-BFGS-B', options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
    assert result.success, result.message
    value, gradient = objective(result.x, f, a, b, y, w, scale)
    assert np.isfinite(result.x).all() and max(abs(gradient)) < 1e-5
    return dict(theta=result.x.tolist(), feature_mean=mean.tolist(), feature_std=std.tolist(),
                global_alpha=alpha, denominators=denominator.tolist(), balanced=balanced,
                objective=value, max_gradient=float(max(abs(gradient))), iterations=int(result.nit),
                optimizer_success=bool(result.success), scale=scale)


def predict(a, b, raw, session, rule):
    f = matrix(raw, session, np.array(rule['feature_mean']), np.array(rule['feature_std']))
    g = expit(np.sum(f * np.array(rule['theta']), axis=1))
    p = a + g * (b - a)
    assert np.isfinite(p).all() and np.all(p >= np.minimum(a, b)-1e-12) and np.all(p <= np.maximum(a, b)+1e-12)
    return p, g


def features(a, b, neural, pc, session, config):
    raw = np.column_stack([a, b, np.abs(b-a), neural])
    if config.endswith('neuron'):
        blocked = np.zeros((len(a), 16))
        blocked[:, session*4:session*4+4] = pc
        raw = np.column_stack([raw, blocked])
    return raw


def labels(pair, seed):
    i = SEEDS.index(seed)
    return f'mlp_s{seed}', (f'attention_s{seed}' if pair == 'attention_mlp' else f'mlp_s{SEEDS[(i+1)%3]}')


def check():
    rng = np.random.default_rng(81217)
    for dimensions in [10, 26]:
        f = rng.normal(size=(80, dimensions)); f[:, 0] = 1
        a, b, y = rng.normal(size=(3, 80))
        theta = rng.normal(size=dimensions) * .1
        w = rng.uniform(.1, 2, 80) / 80
        value, grad = objective(theta, f, a, b, y, w, .7)
        numeric = []
        for i in range(dimensions):
            delta = np.zeros(dimensions); delta[i] = 1e-6
            numeric.append((objective(theta+delta, f, a, b, y, w, .7)[0] - objective(theta-delta, f, a, b, y, w, .7)[0]) / 2e-6)
        np.testing.assert_allclose(grad, numeric, rtol=1e-6, atol=1e-8)
    x = rng.normal(size=(96, 128, 32))
    pca = fit_pca(x); projected = project(x, pca)
    np.testing.assert_allclose(projected.mean(0), 0, atol=1e-12)
    np.testing.assert_allclose(projected.std(0), 1, atol=1e-12)
    assert not np.allclose(projected, project(x[:, ::-1], pca))
    before = {k:v.copy() for k,v in pca.items()}
    project(x+100, pca)
    for k in pca: np.testing.assert_array_equal(pca[k], before[k])
    a, b = rng.normal(size=(2, 80)); y = .65*a+.35*b
    sessions = np.repeat(np.arange(4), 20); raw = rng.normal(size=(80, 6))
    for balanced in [False, True]:
        rule = fit_rule(a,b,y,raw,sessions,balanced)
        out, _ = predict(a,b,raw,sessions,rule)
        assert abs(rule['global_alpha']-.35)<1e-12 and np.mean((out-y)**2)<1e-10
    write(ROOT/'selfcheck.json', dict(passed=True, gate_gradient_checks_both_sizes=True,
        known_blend_recovery_both_losses=True, training_pca_unit_variance=True,
        fixed_pca_unchanged_by_new_inputs=True, neuron_assignment_changes_features=True))


def freeze():
    assert read(ROOT/'selfcheck.json')['passed'] and not (ROOT/'protocol.json').exists()
    files = [Path(__file__), OLD/'run.py', OLD/'protocol.json', OLD/'locked_rules.json',
             OLD/'development_predictions.npz', OLD/'summary.json']
    for m in MICE:
        files += [old.BASE/m/name for name in ['train_x.npy','selection_x.npy','selection_y.npy','columns.npy']]
        files += [old.FAIR/m/name for name in ['metadata.json','statistics.npz','later_raw.npz']]
        files += [old.SHARED/m/'later_predictions.npz']
    for kind in ['attention','mlp']:
        for seed in SEEDS:
            files += [old.SHARED/'shared'/f'{kind}_s{seed}'/name for name in ['selection_predictions.npz','result.json']]
    files += [PROJECT/name for name in ['train.py','model.py','data.py']]
    write(ROOT/'protocol.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
      question='Does balancing session risk or supplying neuron-specific gate features improve time-stable hybrid routing?',
      authorization='User requested deeper exploration with sequential steps after completed complementarity pilot. New bounded follow-up, not a revision of its failed gates.',
      diagnosis='Prior MP032 contributes2.65%of unpenalized calibration error despite large later relative harm. Previous gate has predictions and population summaries only. These motivate separate loss/feature ablations; neither is assumed causal.',
      scope='Exploratory, historically reused four-mouse cohort. Later archive keys/shapes inspected before freeze; no current follow-up numerical later scores used. No independent significance,unseen-animal claim,publication or application edits.',
      screen='2x2 absolute versus risk-balanced loss x coarse versus neuron-specific features. Same AA/MS expert epochs and three development blocks as prior pilot; reuse absolute_coarse predictions and six rules exactly,do not refit them. Eighteen new gate fits for the other three configs,two pair types,three seeds. No subsequent added configs.',
      loss='Fixed sigmoid gate and L2=.01 as pilot. Balanced weights per mouse1/(4*N*D),D=max(mean of two parent calibration MSEs,1e-4); normalize total loss by weighted left-expert calibration MSE. Absolute weights omit D. Denominators and feature scales use calibration only.',
      features='Coarse:6original continuous features+3session indicators+intercept=10coefficients. Neuron:append four PCs per mouse as16session-specific columns=26coefficients. PCs fit only on original training inputs,using512fixed-neuron/four-eight-bin-mean features,training-only standardization,top4covariance eigenvectors,unit training PC variance. No cross-mouse PCA correspondence assumed;no label used for PCA.',
      optimization='One scalar-blend initialization,L-BFGS-B300iterations,ftol1e-12,gtol1e-8,clip standardized gate features+-5. Non-BLAS objective/gradient reductions. No coefficient,feature-count or penalty grid.',
      selection='Select one config per pair type by lowest equal-mouse mean score-block MSE divided by its earlier calibration mean parent MSE (average seeds),floor1e-4. Preset CONFIGS order breaks ties. Select regardless of whether screen gates pass,then perform one locked later replay. Do not select using any later result.',
      final_fit='Reuse original full-development-selected AA/MS archived expert checkpoints and their predictions. Fit selected hybrid gate,matched-config two-MLP gate,and independently selected two-MLP gate on all development predictions;maximum9new gate fits,cache duplicate configs. Also fit original unweighted global/session scalar and parent-choice controls. Development labels serve earlier expert checkpoint selection and meta fitting;possible selection optimism is a limitation. All parameters locked before opening current later targets for scoring.',
      final_evaluation='Reuse archived later predictions and exact neural inputs derived by original shared runner recipe. No base training/inference. Score both parents,half average,global/session blends,calibration-selected parent,and matched/tuned two-MLP controls. One later evaluation;no tuning or new model after outcomes.',
      gate='Selected gated hybrid must beat each parent,half average,unweighted global/per-session blends,matched and independently selected two-MLP gates:each>=2%equal-mouse mean relative MSE gain,>=3/4mouse wins,>=8/12paired-seed wins. Report all failures and per-mouse harms. Same development screen thresholds are descriptive,not confirmation.',
      reporting='Per-mouse and seed MSEs,relative gains,leave-one-mouse-out means,calibration risks,feature ablations,gate weights. Descriptive paired mouse/seed/circular100-bin bootstrap,2000draws,seed81217,95%intervals for final comparisons versus MLP,transformer,tuned two-MLP;not multiplicity-adjusted significance.',
      budget='18new screening gates plus at most9final gates,4unsupervised training-only PCA fits,zero base-model fits/inferences. Stop after locked later comparison and assessment,no automatic grid expansion.',
      methodological_sources=['https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html','https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html'],
      hashes={str(p.relative_to(PROJECT)):digest(p) for p in files}))


def verify():
    protocol=read(ROOT/'protocol.json')
    for name,expected in protocol['hashes'].items():assert digest(PROJECT/name)==expected,name
    return protocol


def pcas():
    result={}
    for m in MICE:
        with np.load(ROOT/f'{m}_pca.npz') as z:result[m]={k:z[k] for k in z.files}
    return result


def prepare():
    verify();assert not (ROOT/'features.npz').exists()
    cache={};records=[]
    for m in MICE:
        train=np.load(old.BASE/m/'train_x.npy',mmap_mode='r')
        pca=fit_pca(train);np.savez_compressed(ROOT/f'{m}_pca.npz',**pca)
        cache[m]=project(np.load(old.BASE/m/'selection_x.npy',mmap_mode='r'),pca)
        assert cache[m].shape==(len(np.load(old.BASE/m/'selection_y.npy')),4)
        records.append(dict(mouse=m,training_windows=len(train),explained_fraction=pca['explained_fraction'].tolist()))
    np.savez_compressed(ROOT/'features.npz',**cache)
    write(ROOT/'preparation.json',dict(passed=True,pcas=records,feature_sha256=digest(ROOT/'features.npz'),
          pca_hashes={m:digest(ROOT/f'{m}_pca.npz') for m in MICE}))


def development(epochs):
    allpred,y,neural=old.load_archives()
    selected={k:{m:v[m][epochs[k]] for m in MICE} for k,v in allpred.items()}
    with np.load(ROOT/'features.npz') as z:pc={m:z[m] for m in MICE}
    return selected,y,neural,pc


def fit_group(pred,y,neural,pc,pair,seed,config,bounds):
    left,right=labels(pair,seed);rows=[]
    for session,m in enumerate(MICE):
        a,b=pred[left][m],pred[right][m];idx=slice(*bounds[m])
        raw=features(a,b,neural[m],pc[m],session,config)
        rows.append((a[idx],b[idx],y[m][idx],raw[idx],np.full(len(y[m][idx]),session)))
    rule=fit_rule(*[np.concatenate([r[j] for r in rows]) for j in range(5)],config.startswith('balanced'))
    return dict(pair=pair,seed=seed,config=config,left=left,right=right,**rule)


def predict_group(pred,neural,pc,rule):
    outputs={}
    for session,m in enumerate(MICE):
        a,b=pred[rule['left']][m],pred[rule['right']][m]
        raw=features(a,b,neural[m],pc[m],session,rule['config'])
        outputs[m]=predict(a,b,raw,np.full(len(a),session),rule)
    return outputs


def screen_fit():
    verify();assert not (ROOT/'screen_lock.json').exists()
    previous=read(OLD/'locked_rules.json');bounds=read(OLD/'protocol.json')['split']
    pred,y,neural,pc=development(previous['epochs']);records={};cache={}
    with np.load(OLD/'development_predictions.npz') as z:
        for pair in PAIRS:
            for seed in SEEDS:
                key=f'{pair}_s{seed}';records[key+'_absolute_coarse']=previous['records'][key]
                for m in MICE:
                    cache[f'{m}_{key}_absolute_coarse']=z[f'{m}_{key}_gate']
    for config in CONFIGS[1:]:
        for pair in PAIRS:
            for seed in SEEDS:
                key=f'{pair}_s{seed}_{config}'
                rule=fit_group(pred,y,neural,pc,pair,seed,config,{m:bounds[m]['calibration'] for m in MICE})
                records[key]=rule
                for m,(p,g) in predict_group(pred,neural,pc,rule).items():cache[m+'_'+key]=p
    np.savez_compressed(ROOT/'screen_predictions.npz',**cache)
    write(ROOT/'screen_lock.json',dict(locked_utc=datetime.now(timezone.utc).isoformat(),
        epochs=previous['epochs'],records=records,new_fits=18,reused_fits=6,
        score_errors_read=False,prediction_sha256=digest(ROOT/'screen_predictions.npz')))


def scores(p,y):
    v=float(np.mean((p-y)**2))
    independent=sum((float(a)-float(b))**2 for a,b in zip(p,y))/len(y)
    assert abs(v-independent)<1e-12*max(1,v)
    return v


def screen_score():
    verify();assert not (ROOT/'screen_selection.json').exists()
    lock=read(ROOT/'screen_lock.json');bounds=read(OLD/'protocol.json')['split']
    assert digest(ROOT/'screen_predictions.npz')==lock['prediction_sha256']
    pred,y,_,_=development(lock['epochs']);results={};selected={}
    with np.load(ROOT/'screen_predictions.npz') as z:
        for pair in PAIRS:
            risks={}
            for m in MICE:
                cal=slice(*bounds[m]['calibration']);terms=[]
                for seed in SEEDS:
                    for label in labels(pair,seed):terms.append(np.mean((pred[label][m][cal]-y[m][cal])**2))
                risks[m]=max(float(np.mean(terms)),1e-4)
            records=[]
            for config in CONFIGS:
                rows=[]
                for m in MICE:
                    idx=slice(*bounds[m]['score']);values=[scores(z[f'{m}_{pair}_s{seed}_{config}'][idx],y[m][idx]) for seed in SEEDS]
                    rows.append(dict(mouse=m,mse=float(np.mean(values)),seed_mse=values,selection_denominator=risks[m]))
                records.append(dict(config=config,rows=rows,selection_score=float(np.mean([r['mse']/risks[r['mouse']] for r in rows]))))
            selected[pair]=min(records,key=lambda r:r['selection_score'])['config'];results[pair]=records
    write(ROOT/'screen_selection.json',dict(selected=selected,results=results,
        selected_utc=datetime.now(timezone.utc).isoformat(),later_scores_read=False,
        screen_lock_sha256=digest(ROOT/'screen_lock.json')))
    print(json.dumps(dict(selected=selected,scores={pair:{r['config']:r['selection_score'] for r in records} for pair,records in results.items()}),indent=2))


def final_fit():
    verify();assert not (ROOT/'final_lock.json').exists()
    selection=read(ROOT/'screen_selection.json');epochs={}
    for kind in ['attention','mlp']:
        for seed in SEEDS:epochs[f'{kind}_s{seed}']=read(old.SHARED/'shared'/f'{kind}_s{seed}'/'result.json')['selected_epoch']
    pred,y,neural,pc=development(epochs);records={};simple={};configs=selection['selected']
    groups=[('hybrid','attention_mlp',configs['attention_mlp']),
            ('matched_two_mlp','mlp_mlp',configs['attention_mlp']),
            ('tuned_two_mlp','mlp_mlp',configs['mlp_mlp'])]
    for name,pair,config in groups:
        for seed in SEEDS:
            key=f'{pair}_s{seed}_{config}'
            if key not in records:records[key]=fit_group(pred,y,neural,pc,pair,seed,config,{m:[0,len(y[m])] for m in MICE})
    for pair in PAIRS:
        for seed in SEEDS:
            left,right=labels(pair,seed);a=np.concatenate([pred[left][m] for m in MICE]);b=np.concatenate([pred[right][m] for m in MICE])
            target=np.concatenate([y[m] for m in MICE]);w=np.concatenate([np.full(len(y[m]),1/(4*len(y[m]))) for m in MICE])
            simple[f'{pair}_s{seed}']=dict(global_alpha=old.weight(a,b,target,w),
                session_alpha=[old.weight(pred[left][m],pred[right][m],y[m],np.ones(len(y[m]))) for m in MICE],
                parent=[int(np.mean((pred[right][m]-y[m])**2)<np.mean((pred[left][m]-y[m])**2)) for m in MICE])
    write(ROOT/'final_lock.json',dict(locked_utc=datetime.now(timezone.utc).isoformat(),
        epochs=epochs,groups=groups,records=records,simple=simple,new_fits=len(records),
        later_scores_read=False,selection_sha256=digest(ROOT/'screen_selection.json'),
        preparation_sha256=digest(ROOT/'preparation.json'),protocol_sha256=digest(ROOT/'protocol.json')))
    print(json.dumps(dict(final_gate_fits=len(records),groups=groups,epochs=epochs),indent=2))


def load_later():
    pca=pcas();pred={f'{kind}_s{seed}':{} for kind in ['attention','mlp'] for seed in SEEDS};y={};neural={};pc={}
    for m in MICE:
        meta=read(old.FAIR/m/'metadata.json');columns=np.load(old.BASE/m/'columns.npy');lower=-meta['speed_mean']/meta['speed_std']
        with np.load(old.FAIR/m/'later_raw.npz') as raw,np.load(old.FAIR/m/'statistics.npz') as norm:
            seq=((raw['activity'][columns,24:]-norm['activity_mean'][columns])/norm['activity_std'][columns]).T.astype(np.float32)
            target=(raw['speed'][55:]-meta['speed_mean'])/meta['speed_std']
        x=np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(seq,32,axis=0))
        with np.load(old.SHARED/m/'later_predictions.npz') as z:
            np.testing.assert_array_equal(target,z['target']);y[m]=z['target']
            for kind in ['attention','mlp']:
                for seed in SEEDS:pred[f'{kind}_s{seed}'][m]=np.maximum(z[f'shared_{kind}_s{seed}'].astype(np.float64),lower)
        assert len(x)==len(y[m])
        neural[m]=np.column_stack([x.mean((1,2),dtype=np.float64),x.std(axis=1,dtype=np.float64).mean(1),np.abs(np.diff(x,axis=2)).mean((1,2),dtype=np.float64)])
        pc[m]=project(x,pca[m])
    return pred,y,neural,pc


def contrast(rows,main,control):
    gains=[1-r[main]/max(r[control],1e-15) for r in rows]
    wins=sum(a<b for r in rows for a,b in zip(r[main+'_seeds'],r[control+'_seeds']))
    return dict(main=main,control=control,mean_relative_gain=float(np.mean(gains)),mouse_gains=gains,
        mouse_wins=sum(g>0 for g in gains),paired_seed_wins=wins,
        leave_one_mouse_out=[float(np.mean(np.delete(gains,i))) for i in range(4)],
        threshold_pass=bool(np.mean(gains)>=.02 and sum(g>0 for g in gains)>=3 and wins>=8))


def final_score():
    verify();assert not (ROOT/'summary.json').exists()
    lock=read(ROOT/'final_lock.json');prep=read(ROOT/'preparation.json')
    for name,key in [('screen_selection.json','selection_sha256'),('preparation.json','preparation_sha256'),('protocol.json','protocol_sha256')]:assert digest(ROOT/name)==lock[key]
    for m,h in prep['pca_hashes'].items():assert digest(ROOT/f'{m}_pca.npz')==h
    pred,y,neural,pc=load_later();cache={m+'_target':y[m] for m in MICE};rows=[]
    for name,pair,config in lock['groups']:
        for seed in SEEDS:
            rule=lock['records'][f'{pair}_s{seed}_{config}']
            for m,(p,g) in predict_group(pred,neural,pc,rule).items():
                cache[f'{m}_{name}_s{seed}']=p;cache[f'{m}_{name}_weight_s{seed}']=g
    for session,m in enumerate(MICE):
        for seed in SEEDS:
            a,b=pred[f'mlp_s{seed}'][m],pred[f'attention_s{seed}'][m];simple=lock['simple'][f'attention_mlp_s{seed}']
            for label,value in {'mlp':a,'attention':b,'half':.5*(a+b),'global':a+simple['global_alpha']*(b-a),
                    'session':a+simple['session_alpha'][session]*(b-a),'parent':a+simple['parent'][session]*(b-a)}.items():cache[f'{m}_{label}_s{seed}']=value
        row=dict(mouse=m,n=len(y[m]))
        for label in ['hybrid','matched_two_mlp','tuned_two_mlp','mlp','attention','half','global','session','parent']:
            values=[scores(cache[f'{m}_{label}_s{seed}'],y[m]) for seed in SEEDS]
            row[label+'_seeds']=values;row[label]=float(np.mean(values))
        rows.append(row)
    comparisons={name:contrast(rows,'hybrid',name) for name in ['mlp','attention','half','global','session','matched_two_mlp','tuned_two_mlp','parent']}
    overall=all(c['threshold_pass'] for k,c in comparisons.items() if k!='parent')
    errors={m:{name:np.stack([(cache[f'{m}_{name}_s{seed}']-y[m])**2 for seed in SEEDS]) for name in ['hybrid','mlp','attention','tuned_two_mlp']} for m in MICE}
    rng=np.random.default_rng(81217);draws=[]
    for _ in range(2000):
        sampled_seeds=rng.integers(0,3,3);gains=[]
        for m in rng.choice(MICE,4):
            n=len(y[m]);idx=((rng.integers(0,n,(n+99)//100)[:,None]+np.arange(100))%n).ravel()[:n]
            mse={name:float(error[sampled_seeds][:,idx].mean()) for name,error in errors[m].items()}
            gains.append([1-mse['hybrid']/max(mse[c],1e-15) for c in ['mlp','attention','tuned_two_mlp']])
        draws.append(np.mean(gains,axis=0))
    intervals={name:np.quantile(np.array(draws)[:,i],[.025,.975]).tolist() for i,name in enumerate(['mlp','attention','tuned_two_mlp'])}
    np.savez_compressed(ROOT/'later_predictions.npz',**cache)
    summary=dict(rows=rows,comparisons=comparisons,overall_gate=overall,descriptive_95_intervals=intervals,
        selected_configs=read(ROOT/'screen_selection.json')['selected'])
    write(ROOT/'summary.json',summary);verify()
    write(ROOT/'audit.json',dict(passed=True,new_screen_gates=18,new_final_gates=lock['new_fits'],
        prior_gates_reused=6,training_only_pca_fits=4,base_fits=0,base_inferences=0,
        exact_later_target_alignment=True,independent_final_scalar_metrics=108,
        selection_locked_before_final_fits=True,final_rules_locked_before_current_later_scoring=True,
        frozen_hashes_unchanged=True,application_unchanged=True,independent_confirmation=False))
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):
        {'check':check,'freeze':freeze,'prepare':prepare,'screen_fit':screen_fit,'screen_score':screen_score,
         'final_fit':final_fit,'final_score':final_score}[sys.argv[1]]()
