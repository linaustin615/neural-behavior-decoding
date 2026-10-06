"""Prespecified animal-level comparisons and descriptive shared results."""
import csv
import io
import math
import numpy as np
from common import ROOT, PLAN, MICE, read, save, now
from evaluate import metrics


def sign_p(wins,losses):
    n=wins+losses
    if not n:
        return 1.
    return min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n)


def holm(pvalues):
    order=np.argsort(pvalues,kind='stable')
    adjusted=np.empty(len(order))
    largest=0.
    for rank,i in enumerate(order):
        largest=max(largest,min(1.,(len(order)-rank)*pvalues[i]))
        adjusted[i]=largest
    return adjusted.tolist()


def compare(rows,regime,other):
    effects=[]
    maes=[]
    details=[]
    seed_wins=0
    for mouse in MICE:
        a=sorted([r for r in rows if r['regime']==regime and r['mouse']==mouse and r['role']=='optimized_attention'],key=lambda r:r['seed'])
        b=[r for r in rows if r['regime']==('control' if other=='ridge' else regime) and r['mouse']==mouse and r['role']==other]
        if other=='ridge':
            assert len(b)==1
            b=b*3
        else:
            b=sorted(b,key=lambda r:r['seed'])
        assert len(a)==len(b)==3
        am,bm=np.mean([r['mse'] for r in a]),np.mean([r['mse'] for r in b])
        assert bm>0
        effect=float(1-am/bm)
        ma=float(1-np.mean([r['mae'] for r in a])/np.mean([r['mae'] for r in b]))
        wins=sum(x['mse']<y['mse'] for x,y in zip(a,b))
        seed_wins+=wins
        effects.append(effect)
        maes.append(ma)
        details.append(dict(mouse=mouse,relative_mse_gain=effect,relative_mae_gain=ma,
            transformer_mse=float(am),comparator_mse=float(bm),seed_wins=wins))
    gate=PLAN['practical_gate']
    conditions=dict(mean_mse=float(np.mean(effects))>=gate['mean_relative_mse_gain_min'],
        mouse_wins=sum(v>0 for v in effects)>=gate['mouse_wins_min'],
        seed_wins=seed_wins>=gate['individual_seed_wins_min'],
        harm=min(effects)>=-gate['max_mouse_relative_harm'],
        mae=float(np.mean(maes))>=gate['mean_relative_mae_gain_min'])
    result=dict(regime=regime,comparator=other,mean_relative_mse_gain=float(np.mean(effects)),
        mean_relative_mae_gain=float(np.mean(maes)),mouse_wins=sum(v>0 for v in effects),
        mouse_losses=sum(v<0 for v in effects),individual_seed_wins=seed_wins,individual_seed_total=21,
        worst_mouse_gain=min(effects),per_mouse=details,practical_conditions=conditions,
        practical_pass=all(conditions.values()),leave_one_mouse_out_mean_gain=[float(np.mean(np.delete(effects,i))) for i in range(7)])
    if regime=='independent':
        result['two_sided_sign_p']=sign_p(result['mouse_wins'],result['mouse_losses'])
    return result


def analyze():
    rows=read(ROOT/'test_metrics.json')['rows']
    contrasts=[]
    for regime in ['independent','shared']:
        for other in ['optimized_population_mlp','default_attention','ridge']:
            contrasts.append(compare(rows,regime,other))
    primary=[r for r in contrasts if r['regime']=='independent']
    adjusted=holm([r['two_sided_sign_p'] for r in primary])
    learning={}
    for regime in ['independent','shared']:
        learning[regime]={}
        for role in PLAN['architecture_configs']:
            means=[]
            single=0
            for mouse in MICE:
                subset=[r for r in rows if r['regime']==regime and r['mouse']==mouse and r['role']==role]
                baseline=next(r['mse'] for r in rows if r['mouse']==mouse and r['role']=='training_mean')
                means.append(float(np.mean([r['mse'] for r in subset]))<baseline)
                single+=sum(r['mse']<baseline for r in subset)
            learning[regime][role]=dict(mouse_means_beating_training_mean=sum(means),individuals_beating_training_mean=single)
    for r,p in zip(primary,adjusted):
        r['holm_p']=p
        r['significant_two_sided']=p<.05
        r['direction']='transformer' if r['mouse_wins']>r['mouse_losses'] else 'comparator' if r['mouse_wins']<r['mouse_losses'] else 'tied'
        r['validated_transformer_advantage']=bool(p<.05 and r['practical_pass'] and learning['independent']['optimized_attention']['mouse_means_beating_training_mean']==7)
    ensembles=[]
    for regime in ['independent','shared']:
        for role in PLAN['architecture_configs']:
            for mouse in MICE:
                predictions=[]
                for seed in PLAN['seeds']:
                    name=f'{regime}_{"all" if regime=="shared" else mouse}_{role}_s{seed}.npz'
                    with np.load(ROOT/'test_predictions'/name) as z:
                        predictions.append(z[mouse+'_prediction'])
                        target=z[mouse+'_target']
                meta=read(ROOT/'prepared'/mouse/'metadata.json')
                ensembles.append(dict(regime=regime,role=role,mouse=mouse,
                    **metrics(np.mean(np.stack(predictions).astype(np.float64),0),target,meta['lower'])))
    result=dict(utc=now(),contrasts=contrasts,learning=learning,ensembles=ensembles,
        independent_animals=7,training_seeds_per_recipe=3,shared_inferential_tests=False,
        inference_caveat='Independent fits prevent shared-training coupling; seven same-lab convenience-sampled animals, distinct identity inferred from publisher IDs/dates, not author correspondence. Exact sign tests concern direction across animals, not the magnitude of mean benefit.')
    save(ROOT/'summary.json',result)
    fields=['regime','mouse','role','seed','selected_epoch','mse','mae','r2','raw_mse','n']
    with (ROOT/'per_run.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fields)
        writer.writeheader()
        writer.writerows(rows)
    return result
