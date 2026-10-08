"""Frozen-score inactivity gating; validation-only choices, no neural training."""
import argparse
import importlib.util
import math
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
B = ROOT.parent/'2026-10-07_bounded_hybrid'
H = ROOT.parent/'2026-10-07_holdout_confirmation'
spec = importlib.util.spec_from_file_location('holdout_source',H/'holdout.py')
source = importlib.util.module_from_spec(spec); spec.loader.exec_module(source)
bounded,prior = source.bounded,source.prior
SEEDS = (401,402,403)
COHORTS = {'development':list(prior.MICE),'previous_holdout':list(source.SET_B),'later_session':list(source.DESCRIPTIVE)}
RECORDINGS = [r for group in COHORTS.values() for r in group]
THRESHOLDS = (.05,.1,.2,.3,.5)
SCALES = (.25,.5,.75,1.)
OPTIONS = [(0.,0.)]+[(s,t) for s in SCALES for t in THRESHOLDS]
DOWN = [(0.,1.)]+[(s,1.) for s in SCALES]


def probability(logits):
    return np.exp(-np.logaddexp(0.,-np.asarray(logits,dtype=float)))


def apply(base,delta,logits,scale,threshold):
    base = np.asarray(base,dtype=float)
    change = scale*np.minimum(np.asarray(delta,dtype=float),0.)*(probability(logits)<=threshold)
    return np.maximum(base+change,0.)


def folder(rid): return B if rid in prior.MICE else H
def fitpath(rid,seed): return folder(rid)/'fits'/f'{rid}_attention_bce_{seed+200}'
def testpath(rid,seed):
    return B/'predictions'/f'{rid}_attention_bce_{seed+200}.npz' if rid in prior.MICE else H/'predictions'/f'{rid}_{seed}.npz'
def cachepath(rid,seed): return folder(rid)/'cache'/f'{rid}_{seed+200 if rid in prior.MICE else seed}.npz'
def datapath(rid): return (prior.SOURCE if rid in prior.MICE else H)/'prepared'/rid


def freeze():
    assert not (ROOT/'protocol.json').exists()
    sources = [Path(__file__),H/'holdout.py',B/'run.py',Path(prior.__file__),prior.REPO/'decoding/model.py',prior.REPO/'decoding/data.py',prior.REPO/'decoding/config.py']
    files = list(sources)
    for study in (B,H): files.extend(study/n for n in ('protocol.json','evaluation_lock.json','summary.json','audit.json'))
    for rid in RECORDINGS:
        files.append(datapath(rid)/'metadata.json')
        files.extend(datapath(rid)/name for name in ('validation_seq.npy','validation_y.npy','train_seq.npy','train_y.npy','train_indices.npy'))
        for seed in SEEDS:
            files.extend([fitpath(rid,seed)/'selected.pt',fitpath(rid,seed)/'result.json',fitpath(rid,seed)/'validation.npz',cachepath(rid,seed),testpath(rid,seed)])
    prior.save(ROOT/'protocol.json',dict(utc=prior.utc(),question='Can conservative confidence-gated downward correction improve quiet predictions without sacrificing running accuracy?',
        disclosure='exploratory follow-up on already examined recordings; former holdout is no longer untouched; no changes to completed confirmation',
        cohorts=COHORTS,seeds=SEEDS,new_fits=0,gated_options=OPTIONS,downward_control_options=DOWN,
        formula='max(base + scale*min(delta,0)*1[sigmoid(movement_logit)<=threshold],0); the score is not assumed calibrated',
        selection='per recording/arm across3seeds: minimize mean full validation MSE among candidates with mean active MSE<=1.01*base and each seed active MSE<=1.05*base; include scale0; ties within1e-12 favor smaller scale then lower threshold',
        barrier='all24 arm/recording choices locked before follow-up test scoring; reconstruct validation only using locked models',
        quiet='speed<=.05 trainingSD; active speed>=.5; missing slices fail benefit/protection checks, cannot be counted as wins',
        comparisons=['gated vs blend','gated vs original correction','gated vs downward-only control','downward-only vs blend','original correction vs blend'],
        refinement_criteria='each cohort separately: mean relative quiet false-movement reduction>=5%, quiet wins>=ceil(.75*N),mean relative totalMSEgain>=0,maxmouseMSEharm<=1%, every mouse activeMSEharm<=1%; practical exploratory criterion, not a new confirmation',
        original_requirement='also report >=5%mean MSEgain vs blend and activeharm<=5% for continuity; no replacement of historical FAIL',
        diagnostic='fraction of all/quiet/active queries changed, movement-score gate coverage, active MSE and quiet QFM',
        limitations='same validation used for checkpoint and gate selection; repeated test use; no neural retraining, new animals or automatic grid extension',
        sha256={str(p.relative_to(prior.REPO)):prior.digest(p) for p in files}))


def check():
    for name,value in prior.read(ROOT/'protocol.json')['sha256'].items():
        assert prior.digest(prior.REPO/name)==value,name


def score(p,y):
    quiet,active = y<=.05,y>=.5
    return dict(mse=float(np.mean((p-y)**2)),mae=float(np.mean(np.abs(p-y))),
        qfm=float(np.mean(p[quiet])) if quiet.any() else None,
        active_mse=float(np.mean((p[active]-y[active])**2)) if active.any() else None,
        quiet_n=int(quiet.sum()),active_n=int(active.sum()),n=len(y))


def select():
    check(); assert not (ROOT/'selection_lock.json').exists()
    out = ROOT/'validation'; out.mkdir(exist_ok=False)
    choices = {}
    for rid in RECORDINGS:
        records = []
        for seed in SEEDS:
            d = prior.examples(rid,'validation') if rid in prior.MICE else source.data(rid,'validation')
            y = d.y[d.indices]-d.metadata['lower']
            net = bounded.model(rid,'attention_bce',seed+200) if rid in prior.MICE else source.correction_model(rid,seed+200)
            fit = fitpath(rid,seed)
            net.load_state_dict(torch.load(fit/'selected.pt',weights_only=True))
            with np.load(cachepath(rid,seed)) as z: base = z['base_validation']
            p,delta,logits = bounded.predict(net,d,base)
            with np.load(fit/'validation.npz') as z:
                np.testing.assert_array_equal(y,z['target'])
                np.testing.assert_array_equal(p,z['prediction'][prior.read(fit/'result.json')['selected_epoch']])
            arrays = dict(base=base,delta=delta,logits=logits,target=y)
            np.savez_compressed(out/f'{rid}_{seed}.npz',**arrays); records.append(arrays)
        choices[rid] = {}
        base_scores = [score(z['base'],z['target']) for z in records]
        for arm,options in [('gated',OPTIONS),('downward',DOWN)]:
            grid = []
            for scale,threshold in options:
                scores = [score(apply(z['base'],z['delta'],z['logits'],scale,threshold),z['target']) for z in records]
                av,bv = [s['active_mse'] for s in scores],[s['active_mse'] for s in base_scores]
                feasible = scale==0 or (all(v is not None for v in av+bv) and np.mean(av)<=1.01*np.mean(bv) and all(a<=1.05*b for a,b in zip(av,bv)))
                grid.append(dict(scale=scale,threshold=threshold,feasible=bool(feasible),mean_mse=float(np.mean([s['mse'] for s in scores])),by_seed=scores))
            best = min(g['mean_mse'] for g in grid if g['feasible'])
            choice = next(g for g in grid if g['feasible'] and g['mean_mse']<=best+1e-12*max(1,abs(best)))
            choices[rid][arm] = dict(scale=choice['scale'],threshold=choice['threshold'],grid=grid,base_scores=base_scores)
        print(rid,{a:(c['scale'],c['threshold']) for a,c in choices[rid].items()},flush=True)
    prior.save(ROOT/'selection_lock.json',dict(utc=prior.utc(),protocol_sha256=prior.digest(ROOT/'protocol.json'),choices=choices,
        validation_sha256={p.name:prior.digest(p) for p in sorted(out.glob('*.npz'))}))


def contrast(rows,rids,a,b):
    per = []
    for rid in rids:
        aa = [r for r in rows if r['recording']==rid and r['arm']==a]
        bb = [r for r in rows if r['recording']==rid and r['arm']==b]
        mean = lambda rr,k: float(np.mean([r[k] for r in rr])) if all(r[k] is not None for r in rr) else None
        am,bm = mean(aa,'mse'),mean(bb,'mse'); aq,bq = mean(aa,'qfm'),mean(bb,'qfm'); av,bv = mean(aa,'active_mse'),mean(bb,'active_mse')
        per.append(dict(recording=rid,gain=1-am/bm,quiet_gain=1-aq/bq if aq is not None and bq is not None and bq>0 else None,
            quiet_win=aq is not None and bq is not None and aq<bq,active_harm=av/bv-1 if av is not None and bv is not None and bv>0 else None,
            candidate_mse=am,control_mse=bm,candidate_qfm=aq,control_qfm=bq))
    return dict(mean_gain=float(np.mean([r['gain'] for r in per])),wins=sum(r['gain']>0 for r in per),max_harm=max(0.,-min(r['gain'] for r in per)),
        quiet_wins=sum(r['quiet_win'] for r in per),mean_quiet_gain=float(np.mean([r['quiet_gain'] for r in per])) if all(r['quiet_gain'] is not None for r in per) else None,
        max_active_harm=max(r['active_harm'] for r in per) if all(r['active_harm'] is not None for r in per) else None,per_recording=per)


def evaluate():
    check(); lock = prior.read(ROOT/'selection_lock.json')
    assert lock['protocol_sha256']==prior.digest(ROOT/'protocol.json')
    for name,h in lock['validation_sha256'].items(): assert prior.digest(ROOT/'validation'/name)==h
    out = ROOT/'predictions'; out.mkdir(exist_ok=False); rows = []
    for rid in RECORDINGS:
        lower = prior.read(datapath(rid)/'metadata.json')['lower']
        for seed in SEEDS:
            with np.load(testpath(rid,seed)) as z: arrays = {k:z[k] for k in z.files}
            y = arrays['target']-lower; base,delta,logits = arrays['base'],arrays['delta'],arrays['logits']
            old = arrays['prediction'] if rid in prior.MICE else arrays['corrected']
            np.testing.assert_array_equal(np.maximum(base+delta.astype(float),0)+lower,old)
            preds = dict(blend=base,original=np.maximum(base+delta.astype(float),0))
            for arm in ('gated','downward'):
                c = lock['choices'][rid][arm]
                preds[arm] = apply(base,delta,logits,c['scale'],c['threshold'])
                assert np.isfinite(preds[arm]).all() and (preds[arm]>=0).all() and (preds[arm]<=base).all()
            for arm,p in preds.items():
                changed = p!=base; quiet,active = y<=.05,y>=.5
                rows.append(dict(recording=rid,seed=seed,arm=arm,**score(p,y),changed_fraction=float(changed.mean()),
                    quiet_changed=float(changed[quiet].mean()) if quiet.any() else None,active_changed=float(changed[active].mean()) if active.any() else None))
            np.savez_compressed(out/f'{rid}_{seed}.npz',target=y,**preds)
    comparisons = {}; gates = {}
    for cohort,rids in COHORTS.items():
        comparisons[cohort] = {a+'_vs_'+b:contrast(rows,rids,a,b) for a,b in [('gated','blend'),('gated','original'),('gated','downward'),('downward','blend'),('original','blend')]}
        c = comparisons[cohort]['gated_vs_blend']
        checks = dict(quiet_gain=c['mean_quiet_gain'] is not None and c['mean_quiet_gain']>=.05,
            quiet_wins=c['quiet_wins']>=math.ceil(.75*len(rids)),total_mean=c['mean_gain']>=0,total_harm=c['max_harm']<=.01,
            active_protection=c['max_active_harm'] is not None and c['max_active_harm']<=.01)
        gates[cohort] = dict(passed=all(checks.values()) if cohort!='later_session' else None,checks=checks,
            original_gain_threshold_met=c['mean_gain']>=.05)
    prior.save(ROOT/'summary.json',dict(utc=prior.utc(),exploratory=True,new_fits=0,rows=rows,comparisons=comparisons,gates=gates))
    print(gates,flush=True)


def smoke():
    b=np.array([0.,1.,2.,3.]); d=np.array([-.5,-.4,.3,-.2]); l=np.array([-10.,0.,-10.,10.])
    np.testing.assert_array_equal(apply(b,d,l,0,.1),b)
    np.testing.assert_allclose(apply(b,d,l,1,.1),[0.,1.,2.,3.])
    np.testing.assert_allclose(apply(b,d,l,1,.5),[0.,.6,2.,3.])
    np.testing.assert_allclose(apply(b,d,l,.5,1),[0.,.8,2.,2.9])
    assert np.isfinite(probability(np.array([-1000.,1000.]))).all()
    prior.save(ROOT/'smoke.json',dict(passed=True,checks=['disabled exact identity','threshold boundary','positive delta excluded','zero clipping','scaled negative correction','stable probability']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('stage',choices=['smoke','freeze','select','evaluate'])
    torch.set_num_threads(2)
    {'smoke':smoke,'freeze':freeze,'select':select,'evaluate':evaluate}[parser.parse_args().stage]()
