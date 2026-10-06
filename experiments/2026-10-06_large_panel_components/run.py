"""Identify whether either isolated attention component helps the larger panel."""
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
COMP=EXP/'2026-10-04_attention_components'
MICE=['MP030','MP032','MP033','MP034']
SEEDS=list(range(10,16))
VARIANTS=['as','ma']


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    loaded=importlib.util.module_from_spec(spec);spec.loader.exec_module(loaded);return loaded


large=module('large_components_parent',LARGE/'run.py')
models=module('large_components_factorial',COMP/'models.py')
models.archived.NEURONS=512
ref,read,write,digest=large.ref,large.read,large.write,large.digest


def make_model(family,seed,sessions):return models.FactorialDecoder('ms' if family=='mlp' else family,seed,sessions)


def selfcheck():
    x=torch.from_numpy(np.load(LARGE/MICE[0]/'selection_x.npy')[:3]);before=x.clone();rows=[]
    for seed in [10,11,12]:
        for variant,parent in [('aa','attention'),('ms','mlp')]:
            a=make_model(variant,seed,range(4));b=large.models.BehaviorDecoder(parent,seed,range(4))
            state=torch.load(LARGE/f'{parent}_s{seed}'/'selected.pt',weights_only=True)
            a.load_state_dict(state);b.load_state_dict(state);a.eval();b.eval()
            with torch.inference_mode():torch.testing.assert_close(a(x,0),b(x,0),rtol=0,atol=0)
            rows.append(dict(seed=seed,variant=variant,trained_forward_exact=True))
        for variant,parent in [('as','attention'),('ma','mlp')]:
            net=make_model(variant,seed,range(4));state=torch.load(LARGE/f'{parent}_s{seed}'/'initial.pt',weights_only=True)
            assert all(torch.equal(v,state[k]) for k,v in net.state_dict().items())
            assert net.temporal.kind==('attention' if variant=='as' else 'mixer')
            assert net.kind==('mlp' if variant=='as' else 'attention')
    assert torch.equal(x,before)
    write(ROOT/'selfcheck.json',dict(passed=True,trained_forward_checks=rows,new_initial_matches=6,input_unchanged=True))


def freeze():
    assert not (ROOT/'protocol.json').exists()
    old=large.verify();assert read(ROOT/'selfcheck.json')['passed']
    paths=[ROOT/n for n in ['run.py','analyse.py','selfcheck.json']]+[COMP/'models.py']
    paths += [LARGE/n for n in ['protocol.json','stage1_results.json','review.json','prepared.json']]
    for seed in [10,11,12]:
        for family in ['attention','mlp']:paths += [LARGE/f'{family}_s{seed}'/n for n in ['initial.pt','selected.pt','history.json','result.json','selection_predictions.npz']]
    paths += [LARGE/m/'stage1_predictions.npz' for m in MICE]
    write(ROOT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        question='With512neurons, does isolating temporal attention or dynamic readout recover a useful advantage over the stronger all-MLP control?',
        motivation='Completed512T improves8.2%versus128T but loses22.3%to equally informed512MLP acrossallfourmice. The two missing factorial combinations identify whether either attention mechanism helps when used alone. Existing128factorial outcomes do not resolve this panel interaction.',
        variants={'as':'causal temporal attention, static behavior-query keys, activity-dependent values','ma':'causal temporal MLP, activity-dependent behavior-query keys and values'},
        architecture='Exact archived FactorialDecoder with NEURONS=512; all other dimensions, single query, IDs, summaries and head unchanged. AS42913params, MA/MS42903params. AS and MA match their corresponding trained-parent initial tensors exactly.',
        budget='Stage1:6new fits,AS/MA x10-12,24epochs each. Select one variant globally by mean earlier selected score over3seeds, tie orderASthenMA. Advance to later scoring only if its earlier score is strictly below BOTH archived512AA and512MS scores. Only selected variant later-scored. If fullstage1latergate passes, train selectedvariant and512MS on13-15 (6morefits). Maximum12new neural fits; zero repeated fits.',
        selection_denominators=old['selection_denominators'],
        training='Same archived trainer,24epochs5688updates179712presentations, optimizer/scheduler/batches/equal-mouseweights. Reuse exactlyprepared512arrays. No newcellpanel,preprocessing,lr,width,regularization orloss search.',
        primary='Selected variant mustbeat128nativeT,512MS,and512ridge by>=5%equal-mousemeanrelativepairMSEgain,>=3/4mousemeans,>=two-thirds individualseedwins,no mouse>10%MSEharm,andnonnegative meanrelativeMAEgain versusEACH. Atleasttwo-thirds beatinitialtrainingmean. Requirestage1,additional13-15 andall6separately. No pooledrescue. Archived512AA is secondary in stage1 only.',
        aggregation='Physical-zero bounding ofindividualpredictions beforeeverydistinct2seedpair average,thenmeanpairerrorswithinmouse,thenrelativeeffects equallyover4mice. Singlesforconsistency; noselectedpair. Ridgebroadcastisnotnewreplication.',
        limits='Adaptive development on historically searchedfourmice, not independentanimal significance. AS-MS isolates temporalattention understaticreadout; MA-MS isolates input-dependent readout withsame temporalMLP. Neither implies every transformer ornovelarchitecture superiority. Candidatechosenonlyusingearlierlabels.',
        stop='Ifearliercandidate fails tobeatbothparents, stopwithoutnewlaterpredictions. Ifstage1laterfails, do nottrainadditionalfits. Otherwise finish fixedstage2. No extra candidates,seeds orrelaxed gates. Mainapp,publishing,generation,reconstruction unchanged.',
        hashes={**old['hashes'],**{str(p.relative_to(PROJECT)):digest(p) for p in paths}}))


def verify():
    p=read(ROOT/'protocol.json')
    for name,value in p['hashes'].items():assert digest(PROJECT/name)==value,name
    return p


def chosen():return read(ROOT/'candidate_lock.json')['selected_variant']


def seeds_for(stage):return [10,11,12] if stage==1 else [13,14,15]


def families_for(stage):return VARIANTS if stage==1 else [chosen(),'mlp']


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
    if stage==1:
        scores={k:float(np.mean([read(ROOT/f'{k}_s{s}'/'result.json')['selection_score'] for s in [10,11,12]])) for k in VARIANTS}
        parents={k:float(np.mean([read(LARGE/f'{k}_s{s}'/'result.json')['selection_score'] for s in [10,11,12]])) for k in ['attention','mlp']}
        selected=min(VARIANTS,key=lambda k:scores[k]);advance=scores[selected]<min(parents.values())
        write(ROOT/'candidate_lock.json',dict(locked_utc=datetime.now(timezone.utc).isoformat(),selected_variant=selected,earlier_advance=advance,
            variant_scores=scores,parent_scores=parents,tie_order=VARIANTS,later_scored=False))


def verify_lock(stage):
    verify();locked=read(ROOT/f'stage{stage}_lock.json')
    for name,value in locked['hashes'].items():assert digest(ROOT/name)==value,name
    selected=read(ROOT/'candidate_lock.json')
    scores={k:float(np.mean([read(ROOT/f'{k}_s{s}'/'result.json')['selection_score'] for s in [10,11,12]])) for k in VARIANTS}
    parents={k:float(np.mean([read(LARGE/f'{k}_s{s}'/'result.json')['selection_score'] for s in [10,11,12]])) for k in ['attention','mlp']}
    winner=min(VARIANTS,key=lambda k:scores[k])
    assert selected['variant_scores']==scores and selected['parent_scores']==parents
    assert selected['selected_variant']==winner and selected['earlier_advance']==(scores[winner]<min(parents.values()))
    return locked


def evaluate(stage):
    verify_lock(stage);assert read(ROOT/'candidate_lock.json')['earlier_advance'];variant=chosen()
    panel=np.load(large.PILOT/'columns.npy');hashes={};count=0
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
            for family in [variant]+(['mlp'] if stage==2 else []):
                net=make_model(family,seed,range(4));state=torch.load(ROOT/f'{family}_s{seed}'/'selected.pt',weights_only=True);net.load_state_dict(state)
                key=f'candidate_s{seed}' if family==variant else f'large_mlp_s{seed}'
                pred[key]=ref.predict(net,x,s);count+=len(y)
                assert all(torch.equal(v,state[k]) for k,v in net.state_dict().items())
        assert torch.equal(before,x)
        out=ROOT/mouse;out.mkdir(exist_ok=True);path=out/f'stage{stage}_predictions.npz';np.savez_compressed(path,**pred);hashes[str(path.relative_to(ROOT))]=digest(path)
        print(stage,mouse,'scored',flush=True)
    verify_lock(stage);write(ROOT/f'stage{stage}_evaluation.json',dict(complete=True,new_predictions=count,prediction_hashes=hashes,losing_variant_later_scored=False))


if __name__=='__main__':
    with threadpool_limits(limits=2):
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        if sys.argv[1]=='train':train(int(sys.argv[2]))
        elif sys.argv[1] in ['lock','evaluate']:{'lock':lock,'evaluate':evaluate}[sys.argv[1]](int(sys.argv[2]))
        else:{'selfcheck':selfcheck,'freeze':freeze}[sys.argv[1]]()
