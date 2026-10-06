"""Fixed population-first temporal attention test, conditional on prior failure."""
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
LARGE=EXP/'2026-10-05_larger_panel'
COMP=EXP/'2026-10-06_large_panel_components'
MICE=['MP030','MP032','MP033','MP034']
SEEDS=list(range(10,16))


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    loaded=importlib.util.module_from_spec(spec);spec.loader.exec_module(loaded);return loaded


large=module('population_parent_run',LARGE/'run.py')
models=module('population_token_models',ROOT/'models.py')
ref,read,write,digest=large.ref,large.read,large.write,large.digest


def make_model(family,seed,sessions):
    return large.models.BehaviorDecoder('mlp',seed,sessions) if family=='old_mlp' else models.PopulationDecoder(family,seed,sessions)


def selfcheck():
    torch.manual_seed(616);x=torch.randn(3,512,32);before=x.clone()
    a=make_model('attention',10,range(4));b=make_model('mlp',10,range(4));rows=[]
    common=[k for k in a.state_dict() if not k.startswith('temporal.')]
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items() if k in common)
    for family,net in [('attention',a),('mlp',b)]:
        net.eval();initial=net(x,0);assert torch.equal(initial,torch.zeros(3))
        changed=x.clone();changed[:,:,16:]+=torch.randn_like(changed[:,:,16:])*3
        with torch.inference_mode():
            z=net.encode(x,0);z2=net.encode(changed,0)
            assert z.shape==(3,8,16)
            torch.testing.assert_close(z[:,:4],z2[:,:4],rtol=0,atol=0)
            assert not torch.equal(z[:,4:],z2[:,4:])
            pop=x.transpose(1,2)@net.readin[0]
            independent=torch.einsum('bnt,nw->btw',x,net.readin[0])
            torch.testing.assert_close(pop,independent,rtol=1e-5,atol=1e-5)
        opt=torch.optim.AdamW(net.parameters(),lr=.001)
        for _ in range(2):
            opt.zero_grad(set_to_none=True);loss=(net(x,0)-torch.tensor([1.,-.5,.2])).square().mean();loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
            opt.step()
        assert net.readin.grad[0].abs().sum()>0 and net.readin.grad[1:].abs().sum()==0
        state={k:v.clone() for k,v in net.state_dict().items()};clone=make_model(family,999,range(4));clone.load_state_dict(state);clone.eval()
        with torch.inference_mode():torch.testing.assert_close(net(x,0),clone(x,0),rtol=0,atol=0)
        assert torch.equal(x,before) and all(torch.equal(v,state[k]) for k,v in net.state_dict().items())
        rows.append(dict(family=family,parameters=sum(p.numel() for p in net.parameters()),causal_patch_prefix_exact=True,readin_gradient=True,other_session_gradients_zero=True,reload_exact=True))
    write(ROOT/'selfcheck.json',dict(passed=True,rows=rows,common_initial_tensors=len(common),population_projection_independently_checked=True,inputs_unchanged=True))


def freeze():
    assert not (ROOT/'protocol.json').exists()
    old=large.verify();assert read(ROOT/'selfcheck.json')['passed']
    paths=[ROOT/n for n in ['models.py','run.py','analyse.py','selfcheck.json']]
    paths += [LARGE/n for n in ['protocol.json','stage1_results.json','review.json','prepared.json']]
    paths += [LARGE/m/'stage1_predictions.npz' for m in MICE]
    paths += [COMP/'protocol.json']
    write(ROOT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        question='Can temporal attention help when its tokens represent joint population states rather than one neuron at a time?',
        motivation='Existing512-neuron model loses to itsmatchedMLP. Per-neuron temporalattention onlysees localhistories untilfinalreadout. A learned linearpopulationprojection beforetemporalattention exposes jointpatterns earlier. This is a distinct architecture hypothesis,notanidentifiedcause ofpreviousfailure.',
        provenance='NDT1 uses full-population time tokens; see https://arxiv.org/abs/2108.01210 and https://proceedings.neurips.cc/paper_files/paper/2023/file/fe51de4e7baf52e743b679e3bdba7905-Paper-Conference.pdf . This compactsupervisedcontinuous-inputdecoder isnot a reproduction ornoveltyclaim. Earlierbroad_temporal had nonlinearquerypooling,2048cells,8bins andseparatefits; it didnot testthissharedlinearreadin32binrecipe.',
        architecture='Per-recording learned512x16linearreadin withstd1/sqrt512 transforms each of32bins into16populationvalues. Flatten each4bin patch to64values thenLinear64to16. Addtime/sessionembeddings,onecausal8tokenAxisBlock,lasttoken plus64originalpopulationmean/stdfeatures into original80to64to1head. No neuron-queryreadout. MatchedMLP onlyreplaces temporalattention withoriginalcausalMLPmixer. Commonstarts exact;headzero atinitialtrainingmean.',
        data='Exactlytheexisting512panel/32bininputs/targets/trainnormalization/chronologicalsplits. Reusepreparedarrays,no newcells ordata selection.',
        trigger='Freezebeforecomponent-studyoutcomes. Launchonly afterthepreceding large-panel-components studycompletesandfailsfullreplication; ifitpasses, recordnotrun. Do notmodifythisrecipebasedonthatoutcome.',
        budget='Stage1 sixnewfits:populationattention/MLP x10-12,24epochs5688updates179712presentations each. If andonlyiffullstage1gate passes, trainpopulationattention,matchedpopulationMLP,andstrongerold512MLP x13-15 (nineadditionalfits). Maximum15newfits. Reuseallold128and512parent/ridgepredictions; nohyperparametersearch.',
        selection_denominators=old['selection_denominators'],
        training='Samearchivedtrainer/batches/optimizer/scheduler/clip/lossweights/earlierselection. Modeltokenizationandreadoutchange; no equalFLOPclaim. Allstagechoiceslockedbeforelaterinference.',
        primary='Populationattention mustbeatmatchedpopulationMLP,old512MLP,128nativeT,and512ridge by>=5%meanrelativepairMSEgain,>=3/4mousemeans,>=two-thirds individualseedwins,no mouse>10%MSEharm,andnonnegative meanrelativeMAEgain againstEACH. Atleasttwo-thirds singlesbeatinitialtrainingmean. Requirestage1,additional13-15,andall6separately; no pooledrescue.',
        aggregation='Bound individual predictions atphysicalzero,averageeachdistinct2seedpair,meanpairerrorswithinmouse,equalmouse relativeeffects. Singlesforconsistency. Ridgebroadcastnotindependentfits. ReportMAE,R2,everyseed/mouse,andleave-one-mouse-out.',
        scope='Adaptivehypothesis onhistoricallysearchedfourmice; notindependentanimalsignificance orcausalconnectivity. Populationmixing islearnedlinearprojection,notitselfcross-neuronattention. Itsattentioncomparespopulationstatesovertime. Mainapp,publishing,generationandreconstructionunchanged.',
        stop='Earliercomponentpasscancelsthisstudy. Otherwisefinishfixedstage1,thenconditionalstage2onlyiffullpass. No model/width/seedextension orrelaxed gates afterresults.',
        hashes={**old['hashes'],**{str(p.relative_to(PROJECT)):digest(p) for p in paths}}))


def verify():
    p=read(ROOT/'protocol.json')
    for name,value in p['hashes'].items():assert digest(PROJECT/name)==value,name
    return p


def seeds_for(stage):return [10,11,12] if stage==1 else [13,14,15]


def families_for(stage):return ['attention','mlp']+(['old_mlp'] if stage==2 else [])


def train(seed):
    assert seed in SEEDS and not read(COMP/'review.json')['full_replication_passed'];stage=1 if seed<13 else 2
    if stage==2:assert read(ROOT/'stage1_results.json')['stage_passed']
    verify()
    for name,value in read(LARGE/'prepared.json')['hashes'].items():assert digest(LARGE/name)==value
    runner=large.previous.native
    runner.ROOT,runner.SEEDS,runner.FAMILIES=ROOT,SEEDS,families_for(stage)
    runner.BehaviorDecoder,runner.verify=make_model,verify
    runner.reference.load_data=large.load_data
    runner.train(seed)


def lock(stage):
    p=verify();hashes={};records=[];count=0
    for seed in seeds_for(stage):
        assert read(ROOT/f'seed{seed}_finished.json')['complete']
        for family in families_for(stage):
            path=ROOT/f'{family}_s{seed}';r=read(path/'result.json');history=read(path/'history.json')
            assert r['updates']==5688 and r['examples']==179712 and r['actual_adam_steps']==[5688] and r['selected_reload_exact']
            with np.load(path/'selection_predictions.npz') as z:
                scores=[]
                for mouse in MICE:
                    y=np.load(ref.BASE/mouse/'selection_y.npy');meta=read(ref.FAIR/mouse/'metadata.json');np.testing.assert_array_equal(y,z[mouse+'_target'])
                    values=[ref.mse(v,y,-meta['speed_mean']/meta['speed_std']) for v in z[mouse+'_predictions']]
                    np.testing.assert_allclose(values,[h['mouse_mse'][mouse] for h in history],rtol=1e-12,atol=1e-12)
                    scores.append(np.array(values)/p['selection_denominators'][mouse]);count+=len(values)
                joint=np.mean(scores,axis=0);np.testing.assert_allclose(joint,[h['selection_score'] for h in history],rtol=1e-12,atol=1e-12)
                assert int(np.argmin(joint))==r['selected_epoch']
            native=large.previous.ft.native_dir('attention',seed)
            assert [h['global_order_hash'] for h in history[1:]]==[h['global_order_hash'] for h in read(native/'history.json')[1:]]
            for name in ['initial.pt','selected.pt','history.json','result.json','selection_predictions.npz']:hashes[str((path/name).relative_to(ROOT))]=digest(path/name)
            records.append(dict(seed=seed,family=family,record=r))
    write(ROOT/f'stage{stage}_lock.json',dict(locked_utc=datetime.now(timezone.utc).isoformat(),hashes=hashes,records=records,selection_scores_checked=count,later_scored=False))


def verify_lock(stage):
    verify();locked=read(ROOT/f'stage{stage}_lock.json')
    for name,value in locked['hashes'].items():assert digest(ROOT/name)==value,name
    return locked


def evaluate(stage):
    verify_lock(stage);panel=np.load(large.PILOT/'columns.npy');hashes={};count=0
    for s,mouse in enumerate(MICE):
        meta=read(ref.FAIR/mouse/'metadata.json')
        with np.load(ref.FAIR/mouse/'later_raw.npz') as raw,np.load(ref.FAIR/mouse/'statistics.npz') as norm:
            seq=((raw['activity'][panel,24:]-norm['activity_mean'][panel])/norm['activity_std'][panel]).T.astype(np.float32)
            y=(raw['speed'][55:]-meta['speed_mean'])/meta['speed_std']
        x=torch.from_numpy(np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(seq,32,axis=0)));before=x.clone()
        source=LARGE/mouse/'stage1_predictions.npz' if stage==1 else ROOT/mouse/'stage1_predictions.npz'
        with np.load(source) as z:
            np.testing.assert_array_equal(y,z['target']);pred={k:z[k] for k in z.files}
        for seed in seeds_for(stage):
            for family in families_for(stage):
                net=make_model(family,seed,range(4));state=torch.load(ROOT/f'{family}_s{seed}'/'selected.pt',weights_only=True);net.load_state_dict(state)
                key=f'large_mlp_s{seed}' if family=='old_mlp' else f'population_{family}_s{seed}'
                pred[key]=ref.predict(net,x,s);count+=len(y)
                assert all(torch.equal(v,state[k]) for k,v in net.state_dict().items())
        assert torch.equal(before,x)
        out=ROOT/mouse;out.mkdir(exist_ok=True);path=out/f'stage{stage}_predictions.npz';np.savez_compressed(path,**pred);hashes[str(path.relative_to(ROOT))]=digest(path)
        print(stage,mouse,'scored',flush=True)
    verify_lock(stage);write(ROOT/f'stage{stage}_evaluation.json',dict(complete=True,new_predictions=count,prediction_hashes=hashes,inputs_states_unchanged=True))


if __name__=='__main__':
    with threadpool_limits(limits=2):
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        if sys.argv[1]=='train':train(int(sys.argv[2]))
        elif sys.argv[1] in ['lock','evaluate']:{'lock':lock,'evaluate':evaluate}[sys.argv[1]](int(sys.argv[2]))
        else:{'selfcheck':selfcheck,'freeze':freeze}[sys.argv[1]]()
