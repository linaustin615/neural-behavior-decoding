"""Fixed global-context query experiment with matched gated-MLP controls."""
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
POP=EXP/'2026-10-06_population_tokens'
LARGE=EXP/'2026-10-05_larger_panel'
MICE=['MP030','MP032','MP033','MP034']
SEEDS=list(range(10,16))


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    loaded=importlib.util.module_from_spec(spec);spec.loader.exec_module(loaded);return loaded


population=module('context_reference_runner',POP/'run.py')
models=module('context_query_models',ROOT/'models.py')
large=population.large
ref,read,write,digest=population.ref,population.read,population.write,population.digest


def make_model(family,seed,sessions):
    if family=='old_mlp':return large.models.BehaviorDecoder('mlp',seed,sessions)
    if family=='pop_parent':return population.models.PopulationDecoder('attention',seed,sessions)
    return models.ContextQueryDecoder(family,seed,sessions)


def selfcheck():
    x=torch.from_numpy(np.load(LARGE/MICE[0]/'selection_x.npy')[:3]);before=x.clone();rows=[]
    for seed in [10,11,12]:
        parent=large.models.BehaviorDecoder('mlp',seed,range(4));state=torch.load(LARGE/f'mlp_s{seed}'/'selected.pt',weights_only=True);parent.load_state_dict(state);parent.eval()
        for family in ['attention','mlp']:
            net=make_model(family,seed,range(4));missing=net.load_state_dict(state,strict=False)
            assert not missing.unexpected_keys and all(k.startswith(('context.','context_projection.')) for k in missing.missing_keys)
            net.eval()
            np.testing.assert_array_equal(ref.predict(net,x,0),ref.predict(parent,x,0))
            with torch.no_grad():
                net.context_projection.weight.copy_(torch.eye(16)*.1)
                changed=x+torch.randn_like(x)*.2
                active,weights=net(x,0,return_weights=True);_,other=net(changed,0,return_weights=True)
                assert weights.shape==(3,2,1,4096) and not torch.equal(weights,other)
                disabled,w0=net(x,0,return_weights=True,context_enabled=False)
                _,w1=net(changed,0,return_weights=True,context_enabled=False)
                assert torch.equal(w0,w1)
                torch.testing.assert_close(disabled,parent(x,0),rtol=0,atol=0)
            rows.append(dict(seed=seed,family=family,zero_projection_exact_trained_parent=True,disabled_context_exact_parent=True,active_query_input_dependent=True,legacy_predictor_compatible=True))
    records=[];a=make_model('attention',10,range(4));b=make_model('mlp',10,range(4))
    common=[k for k in a.state_dict() if not k.startswith('context.temporal.')]
    assert all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in common)
    for family,net in [('attention',a),('mlp',b)]:
        net.eval();assert torch.equal(net(x,0),torch.zeros(3));opt=torch.optim.AdamW(net.parameters(),lr=.001)
        for _ in range(4):
            opt.zero_grad(set_to_none=True);loss=(net(x,0)-torch.tensor([1.,-.5,.2])).square().mean();loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
            opt.step()
        assert net.context_projection.weight.grad.abs().sum()>0 and net.context.readin.grad.abs().sum()>0
        temporal=net.context.temporal.mix.in_proj_weight if family=='attention' else net.context.temporal.weight
        assert temporal.grad.abs().sum()>0
        state={k:v.clone() for k,v in net.state_dict().items()};clone=make_model(family,999,range(4));clone.load_state_dict(state);clone.eval()
        with torch.no_grad():torch.testing.assert_close(ref.predict(net,x,0),ref.predict(clone,x,0),rtol=0,atol=0)
        records.append(dict(family=family,parameters=sum(p.numel() for p in net.parameters()),zero_branch_learns=True,context_and_temporal_gradients=True,reload_exact=True))
    assert torch.equal(x,before)
    write(ROOT/'selfcheck.json',dict(passed=True,parent_checks=rows,records=records,common_initial_tensors=len(common),input_unchanged=True))


def freeze():
    assert not (ROOT/'protocol.json').exists()
    old=read(POP/'protocol.json');check=read(ROOT/'selfcheck.json');assert check['passed']
    paths=[ROOT/n for n in ['models.py','run.py','analyse.py','selfcheck.json']]
    paths += [POP/n for n in ['models.py','run.py','protocol.json','stage1_results.json','review.json']]
    paths += [POP/m/'stage1_predictions.npz' for m in MICE]
    for seed in [10,11,12]:paths += [LARGE/f'mlp_s{seed}'/n for n in ['initial.pt','selected.pt','history.json','result.json']]
    write(ROOT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        question='Can population temporal context guide the stronger per-neuron MLP readout, improving both parent architectures and equally capable MLP controls?',
        motivation='Population-tokenT beat128T by24.5%and512ridge29.1%,butlost1.9%to512localMLP. Parent mouseeffects differ. This motivates integrating population context into neuron selection rather than assuming either standalone bottleneck is sufficient. Complementarity is a hypothesis,not a proved ensemble gain.',
        architecture='Keep entire original512-neuron MLP decoder. Add a population encoder (same16signals/8patches, constructor seed+500), remove its unused head. A zero-initialized16x16bias-free projection adds its last temporal state to the behavior query BEFORE query projection,readout and residual path. Neuron/time/session keys remain static; activity-dependent values stay local. Query therefore depends on input population context afterlearning. Same80featurehead and64stats. Train allparameters fromscratch; not prediction averaging or warm-starting.',
        control='Matched gated-MLP model has identical localdecoder,projection,capacity and readout, but its population-context temporalblock is the originalMLPmixer. Both have input-dependent query gating; the control removes population temporalself-attention,not all gating. Common initialtensors exact,parameters differby10. Also compare stronger512localMLP,populationTparent,and512ridge.128T secondary.',
        parameter_counts={v['family']:v['parameters'] for v in check['records']},
        data='Exact existing512panel/32binarrays,train-onlynormalization andchronologicaltargets/gaps. No newdata/cellselection.',
        budget='Stage1sixfits:contextT/gatedMLP x10-12. Onlyif fullstage1gatepasses,stage2 contextT/gatedMLP/old512MLP/populationTparent x13-15 (12additionalfits). Maximum18newfits. Each24epochs5688updates179712presentations. Previousfits/predictionsreused; no repeatedfit.',
        selection_denominators=old['selection_denominators'],
        training='Original archivedtrainer,optimizer,scheduler,clip,batches,equal-mouseweighting andearlier epoch0-24selection. Allstagechoiceslocked beforelaterinference. Largerhybridcapacity iscontrolled bymatchedgatedMLP; no equalcomputeclaimversussingleparents.',
        primary='ContextT mustbeatmatchedgatedMLP,old512MLP,populationTparent,and512ridge by>=5%equal-mousemeanrelativepairMSE,>=3/4mousemeans,>=two-thirds individualseedwins,no mouse>10%MSEharm,andnonnegative meanrelativeMAE againstEACH. Atleasttwo-thirds beatinitialtrainingmean. Requirestage1,additional13-15,andall6separately.128Tsecondary. No pooledrescue.',
        aggregation='Physicalzero bounding perindividual beforeeverydistinct2seedpair average;meanpairerrorwithinmouse,equalmouse relativeeffects;singlemodelsforconsistency. No favorablepairselection. Deterministicridgebroadcastnotindependentfits.',
        scope='Adaptive development onfourhistoricallysearchedmice, notindependentanimal significance,novelty,causalconnectivity orcoordinateutility. Awin mustsurvivematchedglobalcontextMLP,notjustweakerparents. Mainapp,publishing,generation,reconstruction unchanged.',
        stop='Fixedstage1thenconditionalstage2onlyiffullpass. No extraqueryscales,projectionranks,seeds,ensembles orrelaxedgates afteroutcomes.',
        hashes={**old['hashes'],**{str(p.relative_to(PROJECT)):digest(p) for p in paths}}))


def verify():
    p=read(ROOT/'protocol.json')
    for name,value in p['hashes'].items():assert digest(PROJECT/name)==value,name
    return p


def seeds_for(stage):return [10,11,12] if stage==1 else [13,14,15]


def families_for(stage):return ['attention','mlp']+(['old_mlp','pop_parent'] if stage==2 else [])


def train(seed):
    assert seed in SEEDS;stage=1 if seed<13 else 2
    if stage==2:assert read(ROOT/'stage1_results.json')['stage_passed']
    verify()
    for name,value in read(LARGE/'prepared.json')['hashes'].items():assert digest(LARGE/name)==value
    runner=large.previous.native
    runner.ROOT,runner.SEEDS,runner.FAMILIES=ROOT,SEEDS,families_for(stage)
    runner.BehaviorDecoder,runner.verify=make_model,verify
    runner.reference.load_data=large.load_data
    runner.train(seed)


def configure_lock_runner():
    population.ROOT=ROOT;population.verify=verify;population.families_for=families_for


def lock(stage):
    configure_lock_runner();population.lock(stage)


def verify_lock(stage):
    configure_lock_runner();return population.verify_lock(stage)


def evaluate(stage):
    verify_lock(stage);panel=np.load(large.PILOT/'columns.npy');hashes={};count=0
    for s,mouse in enumerate(MICE):
        meta=read(ref.FAIR/mouse/'metadata.json')
        with np.load(ref.FAIR/mouse/'later_raw.npz') as raw,np.load(ref.FAIR/mouse/'statistics.npz') as norm:
            seq=((raw['activity'][panel,24:]-norm['activity_mean'][panel])/norm['activity_std'][panel]).T.astype(np.float32)
            y=(raw['speed'][55:]-meta['speed_mean'])/meta['speed_std']
        x=torch.from_numpy(np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(seq,32,axis=0)));before=x.clone()
        source=POP/mouse/'stage1_predictions.npz' if stage==1 else ROOT/mouse/'stage1_predictions.npz'
        with np.load(source) as z:
            np.testing.assert_array_equal(y,z['target']);pred={k:z[k] for k in z.files}
        for seed in seeds_for(stage):
            for family in families_for(stage):
                net=make_model(family,seed,range(4));state=torch.load(ROOT/f'{family}_s{seed}'/'selected.pt',weights_only=True);net.load_state_dict(state)
                prefix={'old_mlp':'large_mlp','pop_parent':'population_attention'}.get(family,'context_'+family)
                pred[f'{prefix}_s{seed}']=ref.predict(net,x,s);count+=len(y)
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
