"""Replicate the existing one-query static-readout temporal transformer."""
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
EXP=ROOT.parent
COMPONENTS=EXP/'2026-10-04_attention_components'
CONFIRMATION=EXP/'2026-10-05_shared_confirmation'


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    loaded=importlib.util.module_from_spec(spec);spec.loader.exec_module(loaded);return loaded


previous=module('static_replication_reference',CONFIRMATION/'run.py')
models=module('static_replication_models',COMPONENTS/'models.py')
ref,read,write,digest=previous.ref,previous.read,previous.write,previous.digest
MICE=ref.MICE
SEEDS=list(range(13,19))
ALL_SEEDS=list(range(10,19))


def native_dir(seed):
    return previous.ft.native_dir('attention',seed) if seed<16 else previous.native_dir('attention',seed)


def fit_dir(seed):
    return COMPONENTS/f'as_s{seed}' if seed<13 else ROOT/f'static_s{seed}'


def make_model(family,seed,sessions):
    assert family=='static'
    return models.FactorialDecoder('as',seed,sessions)


def freeze():
    old=read(CONFIRMATION/'protocol.json')
    paths=[ROOT/n for n in ['run.py','analyse.py']]+[COMPONENTS/n for n in ['models.py','selfcheck.json','protocol.json','summary.json']]
    paths += [CONFIRMATION/n for n in ['run.py','analyse.py','protocol.json','results.json','review.json']]
    for seed in ALL_SEEDS:
        paths += [native_dir(seed)/n for n in ['initial.pt','selected.pt','history.json','result.json','selection_predictions.npz']]
        if seed<13:paths += [fit_dir(seed)/n for n in ['initial.pt','selected.pt','history.json','result.json','selection_predictions.npz']]
    for mouse in MICE:paths += [p/mouse/'later_predictions.npz' for p in [COMPONENTS,CONFIRMATION]]
    assert read(COMPONENTS/'selfcheck.json')['static_keys_input_invariant']
    for seed in SEEDS:
        net=make_model('static',seed,range(4))
        state=torch.load(native_dir(seed)/'initial.pt',weights_only=True)
        assert all(torch.equal(v,state[k]) for k,v in net.state_dict().items())
        assert net.temporal.kind=='attention' and net.kind=='mlp'
    write(ROOT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        authorization='Continuous search requested. New fixed replication of the previously tested one-query static variant; prior factorial study remains unchanged.',
        question='Does removing activity-dependent readout routing, while retaining temporal attention, produce a more reliable decoder across additional training seeds?',
        motivation='Original component study found3.4%average single-model gain vsdynamic transformer but failed fullutility. Native transformer now has positive mean gains onnew16-18 but inadequate individualseedconsistency. Simpler activity-independent readout is a plausible variance-control hypothesis,not a verified cause.',
        architecture='Exact archived FactorialDecoder(as):128neurons,32bins,causal temporalattention,single learned query with static neuron/time/session keys and activity-dependent values,originalpopulationstatistics/head. No newparameters;initial tensors matchnativeT exactly. Retainoriginalsource andcompletedpreflight.',
        budget='Six newfits on13-18,24epochs5688updates179712presentations each. Reuse as10-12 and allnine dynamicT/MLP/ridge comparators. Import original shared replication trainer with only model factory and family label configured. No fine-tuning orhyperparam search.',
        selection_denominators=old['selection_denominators'],
        selection='One joint earlier-selected epoch0-24 using original boundedMSE/ridge denominators. Lockallchoices before currentlaterinference. No newpermouse selection,rates,heads,widths orseedextensions.',
        primary='Candidate staticT must beat nativeT,nativeMLP andrawridge by>=5%equal-mousemeanpairMSE,>=3/4mousemeans,>=ceil(2/3*4*nseeds)individualseedwins;against EACH no mouse>10%harm andnonnegative meanMAEgain. Require in latest16-18,alladditional13-18 andallnine10-18 separately;pooledoutcomescannotrescuefailedlatestseeds. Atleasttwo-thirds ofsinglesmust beatinitialtrainingmean. TunedMLP comparisonreportedsecondary.',
        aggregation='Same physicalzeroindividualbounding beforealltwo-seedpair averaging,withinmouseerror averages,equalmouse relativeeffects. Reportlatest3,additional6,all9;scores,singles,MAE,R2,leave-one-out. Deterministicridgebroadcastis consistencyreference,notnewindependentfits.',
        scope='Allbaseline outcomes and originalAS results alreadyknown. This testsunrunarchitecture/seedcombinations onhistoricallysearchedmice,not independent animal significance. No general attention,novelty,biologicalconnectivity orunseen-mouseclaim.',
        stop='Completefixed6fits,lock,evaluation,audit/report;do notappendseed/modelgrid orrelaxgates. No mainapplication,publishing,generationorreconstruction edits.',
        hashes={**old['hashes'],**{str(p.relative_to(PROJECT)):digest(p) for p in paths}}))


def verify():
    p=read(ROOT/'protocol.json')
    for name,value in p['hashes'].items():assert digest(PROJECT/name)==value,name
    return p


def train(seed):
    assert seed in SEEDS
    runner=previous.native
    runner.ROOT,runner.SEEDS,runner.FAMILIES=ROOT,SEEDS,['static']
    runner.BehaviorDecoder,runner.verify=make_model,verify
    runner.train(seed)
    state=torch.load(fit_dir(seed)/'initial.pt',weights_only=True)
    native=torch.load(native_dir(seed)/'initial.pt',weights_only=True)
    assert all(torch.equal(v,native[k]) for k,v in state.items())


def lock():
    p=verify();hashes={};checked=0;records=[]
    assert all(read(ROOT/f'seed{s}_finished.json')['complete'] for s in SEEDS)
    for seed in ALL_SEEDS:
        path=fit_dir(seed);record=read(path/'result.json');history=read(path/'history.json')
        assert len(history)==25 and record['examples']==179712
        assert record.get('updates',record.get('allocated_updates'))==5688
        assert record.get('actual_adam_steps',record.get('adam_steps'))==[5688]
        assert record['selected_reload_exact']
        with np.load(path/'selection_predictions.npz') as z:
            scores=[]
            for mouse in MICE:
                y=np.load(ref.BASE/mouse/'selection_y.npy');meta=read(ref.FAIR/mouse/'metadata.json')
                np.testing.assert_array_equal(y,z[mouse+'_target'])
                values=[ref.mse(v,y,-meta['speed_mean']/meta['speed_std']) for v in z[mouse+'_predictions']]
                np.testing.assert_allclose(values,[h['mouse_mse'][mouse] for h in history],rtol=1e-12,atol=1e-12)
                scores.append(np.array(values)/p['selection_denominators'][mouse]);checked+=len(values)
            joint=np.mean(scores,axis=0)
            np.testing.assert_allclose(joint,[h['selection_score'] for h in history],rtol=1e-12,atol=1e-12)
            assert int(np.argmin(joint))==record['selected_epoch']
        assert [h['global_order_hash'] for h in history[1:]]==[h['global_order_hash'] for h in read(native_dir(seed)/'history.json')[1:]]
        records.append(dict(seed=seed,reused=seed<13,record=record))
        for name in ['initial.pt','selected.pt','result.json','history.json','selection_predictions.npz']:
            hashes[str((path/name).relative_to(PROJECT))]=digest(path/name)
    write(ROOT/'selection_lock.json',dict(records=records,hashes=hashes,selection_scores_checked=checked,
        locked_utc=datetime.now(timezone.utc).isoformat(),new_later_scored=False))


def verify_lock():
    verify();locked=read(ROOT/'selection_lock.json')
    for name,value in locked['hashes'].items():assert digest(PROJECT/name)==value,name
    return locked


def evaluate():
    verify_lock();hashes={};count=0
    for s,mouse in enumerate(MICE):
        meta=read(ref.FAIR/mouse/'metadata.json');cols=np.load(ref.BASE/mouse/'columns.npy')
        with np.load(ref.FAIR/mouse/'later_raw.npz') as raw,np.load(ref.FAIR/mouse/'statistics.npz') as norm:
            seq=((raw['activity'][cols,24:]-norm['activity_mean'][cols])/norm['activity_std'][cols]).T.astype(np.float32)
            y=(raw['speed'][55:]-meta['speed_mean'])/meta['speed_std']
        x=torch.from_numpy(np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(seq,32,axis=0)))
        original=x.clone()
        with np.load(CONFIRMATION/mouse/'later_predictions.npz') as z:
            np.testing.assert_array_equal(y,z['target']);pred={k:z[k] for k in z.files}
        with np.load(COMPONENTS/mouse/'later_predictions.npz') as z:
            np.testing.assert_array_equal(y,z['target'])
            for seed in [10,11,12]:pred[f'static_s{seed}']=z[f'as_s{seed}']
        for seed in SEEDS:
            net=make_model('static',seed,range(4));state=torch.load(fit_dir(seed)/'selected.pt',weights_only=True);net.load_state_dict(state)
            pred[f'static_s{seed}']=ref.predict(net,x,s);count+=len(y)
            assert all(torch.equal(v,state[k]) for k,v in net.state_dict().items())
        assert torch.equal(original,x)
        out=ROOT/mouse;out.mkdir();path=out/'later_predictions.npz';np.savez_compressed(path,**pred);hashes[str(path.relative_to(ROOT))]=digest(path)
        print(mouse,'scored',flush=True)
    verify_lock();write(ROOT/'evaluation.json',dict(complete=True,prediction_hashes=hashes,new_predictions=count,
        native_and_original_static_predictions_reused=True,inputs_models_unchanged=True))


if __name__=='__main__':
    with threadpool_limits(limits=2):
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        if sys.argv[1]=='train':
            for seed in map(int,sys.argv[2:]):train(seed)
        else:{'freeze':freeze,'lock':lock,'evaluate':evaluate}[sys.argv[1]]()
