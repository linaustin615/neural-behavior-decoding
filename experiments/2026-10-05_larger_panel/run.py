"""Fixed larger-panel transformer comparison, with conditional seed replication."""
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
PILOT=EXP/'2026-10-05_neuron_panel_pilot'
CONFIRMATION=EXP/'2026-10-05_shared_confirmation'
MICE=['MP030','MP032','MP033','MP034']
SEEDS=list(range(10,16))
FAMILIES=['attention','mlp']


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    loaded=importlib.util.module_from_spec(spec);spec.loader.exec_module(loaded);return loaded


previous=module('larger_panel_previous',CONFIRMATION/'run.py')
pilot=module('larger_panel_pilot',PILOT/'run.py')
models=module('larger_panel_models',ROOT/'models.py')
ref,read,write,digest=previous.ref,previous.read,previous.write,previous.digest


def selfcheck():
    torch.manual_seed(101);x=torch.randn(2,512,32);before=x.clone()
    nets={k:models.BehaviorDecoder(k,10,range(4)) for k in FAMILIES}
    common=set(nets['attention'].state_dict())&set(nets['mlp'].state_dict())
    shared=[k for k in common if not k.startswith('temporal.')]
    assert all(torch.equal(nets['attention'].state_dict()[k],nets['mlp'].state_dict()[k]) for k in shared)
    records=[]
    for family,net in nets.items():
        net.eval();p,w=net(x,0,return_weights=True)
        assert p.shape==(2,) and w.shape==(2,2,1,4096) and torch.equal(p,torch.zeros_like(p))
        torch.testing.assert_close(w.sum(-1),torch.ones_like(w.sum(-1)))
        changed=x.clone();changed[:,128:]+=torch.randn_like(changed[:,128:])*2
        p2,w2=net(changed,0,return_weights=True)
        assert (torch.equal(w,w2))==(family=='mlp')
        opt=torch.optim.AdamW(net.parameters(),lr=.001)
        for _ in range(2):
            opt.zero_grad(set_to_none=True);loss=(net(x,0)-torch.tensor([1.,-1.])).square().mean();loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
            opt.step()
        assert net.identity.grad[0,128:].abs().sum()>0
        state={k:v.clone() for k,v in net.state_dict().items()}
        clone=models.BehaviorDecoder(family,99,range(4));clone.load_state_dict(state);clone.eval()
        with torch.inference_mode():torch.testing.assert_close(net(x,0),clone(x,0),rtol=0,atol=0)
        assert torch.equal(x,before) and all(torch.equal(v,state[k]) for k,v in net.state_dict().items())
        records.append(dict(family=family,parameters=sum(v.numel() for v in net.parameters()),new_neuron_id_gradients=True,reload_exact=True))
    write(ROOT/'selfcheck.json',dict(passed=True,records=records,common_initial_tensors=len(shared),weights_shape=[2,2,1,4096],input_unchanged=True))


def freeze():
    assert not (ROOT/'protocol.json').exists()
    old=read(CONFIRMATION/'protocol.json');p=pilot.verify()
    assert read(PILOT/'results.json')['development_advance'] and read(ROOT/'selfcheck.json')['passed']
    paths=[ROOT/n for n in ['run.py','models.py','analyse.py','selfcheck.json']]
    paths += [PILOT/n for n in ['protocol.json','results.json','columns.npy']]
    paths += [PILOT/m/n for m in MICE for n in ['ridge.npz','result.json']]
    paths += [CONFIRMATION/m/'later_predictions.npz' for m in MICE]
    write(ROOT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        question='Does the original shared transformer benefit from a larger nested neuron panel enough to outperform its128-cell parent and equally informed controls?',
        motivation='Separate development-only ridge pilot found31.4%lower earlier MSE with512cells and4/4mouse gains. This suggests an input-information limit worth testing, not proof that attention will exploit it better.',
        architecture='Exact original BehaviorDecoder with only NEURONS=512. Same32-bin history,8four-bin patches,width16,causal temporalattention,dynamic singlequery and64populationstatistics. Matched MLP changes only the original temporal/readout choices.42913versus42903parameters; extra learned cellIDs shared fairly across both families.',
        data='Reuse pilot nested512panel, original training-only normalization, identical chronological train/selection/later targets and100-bin gaps. Original128cells contained exactly. No cell ranking or extra panel search.',
        budget='Stage1:6new neural fits,attention/MLP xseeds10-12,24epochs5688updates179712presentations each. If and only if fullstage1latergate passes, stage2 adds same6fits on13-15 with identical recipe. Maximum12new neural fits. Reuse all128-cell predictions and four frozen512ridge fits. No hyperparameter sweep.',
        selection_denominators=old['selection_denominators'],
        training='Exact archived ensemble replication trainer, with only model factory,512data loader,output root and allowed seeds configured. AdamW.001,wd.01,cosine24to.0001,clip1,batch32,original equal-mouse trainingweights andbatchseeds. Bigger models have more compute; no equal-FLOP claim.',
        selection='Original joint boundedMSE divided by frozen128ridge denominators, epoch0-24 allowed. Eachstage locks bothfamilies before its newlaterinference; larger ridge penalty alreadylocked by pilot onearlierdata. Stage1is exploratoryscreen, stage2is additional training-seed replication on sameanimals.',
        primary='512attention mustbeat128nativeattention,512MLP,and512rawridge:>=5%equal-mouse mean relative two-model-pairMSEgain,>=3/4mousemeans,>=two-thirds individual seedwins, no mouse>10%MSEharm andnonnegative meanrelativeMAEgain versus EACH. Atleasttwo-thirds ofindividualmodels beatinitialtrainingmean. Requirestage1,additional13-15,andall6separately. Pooledcannotrescueeitherstage. Native128MLP secondary.',
        aggregation='Bound eachprediction at physicalzero,averageeverydistinct2seedpair,thenaveragepairerrorswithinmouse,thenrelativeeffects equallyover4mice. Individualwins computedfromsinglemodelerrors. Reportallerrors,MAE,R2,perseed,permouse,leave-one-mouse-out. No bestpair/modelselection onlaterlabels.',
        stopping='Afailedstage1stopsbeforestage2. Otherwisefinishfixedstage2andreportregardlessofoutcome. No extra cells,seeds,trainingrecipe orrelaxedthresholds afterresults. Anyexpansionbenefit includesinputcontent,IDcapacity andsummarychanges,not uniquelyattention.',
        scope='Historically searchedfourmice,known128comparators anddevelopmentpilot informedthishypothesis. No newanimalorindependentsignificance claim. No originalapp,publishing,generation orreconstruction edits.',
        hashes={**old['hashes'],**p['hashes'],**{str(path.relative_to(PROJECT)):digest(path) for path in paths}}))


def verify():
    p=read(ROOT/'protocol.json')
    for name,value in p['hashes'].items():assert digest(PROJECT/name)==value,name
    return p


def prepare():
    verify();panel=np.load(PILOT/'columns.npy');hashes={}
    for mouse in MICE:
        out=ROOT/mouse;out.mkdir()
        oldslots=np.searchsorted(panel,np.load(ref.BASE/mouse/'columns.npy'))
        for split in ['train','selection']:
            x=pilot.recover(ref.FAIR/mouse/f'{split}_x.npy',panel)
            np.testing.assert_array_equal(x[:,oldslots],np.load(ref.BASE/mouse/f'{split}_x.npy'))
            path=out/f'{split}_x.npy';np.save(path,x);hashes[str(path.relative_to(ROOT))]=digest(path)
    write(ROOT/'prepared.json',dict(complete=True,hashes=hashes,original128columns_exact=True))


def load_data():
    data={}
    for s,mouse in enumerate(MICE):
        meta=read(ref.FAIR/mouse/'metadata.json')
        data[s]=dict(x=torch.from_numpy(np.load(ROOT/mouse/'train_x.npy')),y=torch.from_numpy(np.load(ref.BASE/mouse/'train_y.npy').astype(np.float32)),
            xv=torch.from_numpy(np.load(ROOT/mouse/'selection_x.npy')),yv=np.load(ref.BASE/mouse/'selection_y.npy'),lower=-meta['speed_mean']/meta['speed_std'])
    return data


def train(seed):
    assert seed in SEEDS
    if seed>=13:assert read(ROOT/'stage1_results.json')['subsets']['stage1']['primary_pass']
    verify()
    for name,value in read(ROOT/'prepared.json')['hashes'].items():assert digest(ROOT/name)==value,name
    runner=previous.native
    runner.ROOT,runner.SEEDS,runner.FAMILIES=ROOT,SEEDS,FAMILIES
    runner.BehaviorDecoder,runner.verify=models.BehaviorDecoder,verify
    runner.reference.load_data=load_data
    runner.train(seed)


def seeds_for(stage):return [10,11,12] if stage==1 else [13,14,15]


def lock(stage):
    p=verify();seeds=seeds_for(stage);records=[];hashes={};count=0
    assert all(read(ROOT/f'seed{s}_finished.json')['complete'] for s in seeds)
    for seed in seeds:
        for family in FAMILIES:
            path=ROOT/f'{family}_s{seed}';record=read(path/'result.json');history=read(path/'history.json')
            assert record['updates']==5688 and record['examples']==179712 and record['actual_adam_steps']==[5688] and record['selected_reload_exact']
            with np.load(path/'selection_predictions.npz') as z:
                scores=[]
                for mouse in MICE:
                    y=np.load(ref.BASE/mouse/'selection_y.npy');meta=read(ref.FAIR/mouse/'metadata.json')
                    np.testing.assert_array_equal(y,z[mouse+'_target'])
                    values=[ref.mse(v,y,-meta['speed_mean']/meta['speed_std']) for v in z[mouse+'_predictions']]
                    np.testing.assert_allclose(values,[h['mouse_mse'][mouse] for h in history],rtol=1e-12,atol=1e-12)
                    scores.append(np.array(values)/p['selection_denominators'][mouse]);count+=len(values)
                joint=np.mean(scores,axis=0)
                np.testing.assert_allclose(joint,[h['selection_score'] for h in history],rtol=1e-12,atol=1e-12)
                assert int(np.argmin(joint))==record['selected_epoch']
            native=previous.ft.native_dir(family,seed)
            assert [h['global_order_hash'] for h in history[1:]]==[h['global_order_hash'] for h in read(native/'history.json')[1:]]
            for name in ['initial.pt','selected.pt','result.json','history.json','selection_predictions.npz']:hashes[str((path/name).relative_to(ROOT))]=digest(path/name)
            records.append(dict(seed=seed,family=family,record=record))
    write(ROOT/f'stage{stage}_lock.json',dict(locked_utc=datetime.now(timezone.utc).isoformat(),records=records,hashes=hashes,selection_scores_checked=count,later_scored=False))


def verify_lock(stage):
    verify();locked=read(ROOT/f'stage{stage}_lock.json')
    for name,value in locked['hashes'].items():assert digest(ROOT/name)==value,name
    return locked


def evaluate(stage):
    verify_lock(stage);panel=np.load(PILOT/'columns.npy');seeds=seeds_for(stage);hashes={};count=0
    for s,mouse in enumerate(MICE):
        meta=read(ref.FAIR/mouse/'metadata.json')
        with np.load(ref.FAIR/mouse/'later_raw.npz') as raw,np.load(ref.FAIR/mouse/'statistics.npz') as norm:
            seq=((raw['activity'][panel,24:]-norm['activity_mean'][panel])/norm['activity_std'][panel]).T.astype(np.float32)
            y=(raw['speed'][55:]-meta['speed_mean'])/meta['speed_std']
        x=np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(seq,32,axis=0));tx=torch.from_numpy(x);before=tx.clone()
        archive=CONFIRMATION/mouse/'later_predictions.npz' if stage==1 else ROOT/mouse/'stage1_predictions.npz'
        with np.load(archive) as old:
            np.testing.assert_array_equal(y,old['target']);pred={k:old[k] for k in old.files}
        if stage==1:
            with np.load(PILOT/mouse/'ridge.npz') as ridge:
                flat=x.reshape(len(x),-1).astype(np.float64)
                pred['ridge512']=np.einsum('ij,j->i',(flat-ridge['center'])/ridge['scale'],ridge['weight'],optimize=False)+ridge['mean']
            assert np.isfinite(pred['ridge512']).all()
        for seed in seeds:
            for family in FAMILIES:
                net=models.BehaviorDecoder(family,seed,range(4));state=torch.load(ROOT/f'{family}_s{seed}'/'selected.pt',weights_only=True);net.load_state_dict(state)
                pred[f'large_{family}_s{seed}']=ref.predict(net,tx,s);count+=len(y)
                assert all(torch.equal(v,state[k]) for k,v in net.state_dict().items())
        assert torch.equal(before,tx)
        path=ROOT/mouse/f'stage{stage}_predictions.npz';np.savez_compressed(path,**pred);hashes[str(path.relative_to(ROOT))]=digest(path)
        print('stage',stage,mouse,'scored',flush=True)
    verify_lock(stage);write(ROOT/f'stage{stage}_evaluation.json',dict(complete=True,new_neural_predictions=count,prediction_hashes=hashes,new_ridge_predictions=2218 if stage==1 else 0,inputs_states_unchanged=True))


if __name__=='__main__':
    with threadpool_limits(limits=2):
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        if sys.argv[1]=='train':train(int(sys.argv[2]))
        elif sys.argv[1] in ['lock','evaluate']:{'lock':lock,'evaluate':evaluate}[sys.argv[1]](int(sys.argv[2]))
        else:{'selfcheck':selfcheck,'freeze':freeze,'prepare':prepare}[sys.argv[1]]()
