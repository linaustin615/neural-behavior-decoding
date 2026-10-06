import csv
import json

import numpy as np
import torch

import run as r


def group_stability(pool,ids,groups):
    rng=np.random.default_rng(880000+pool)
    us,vs=[],[]
    for g in range(8):
        cells=np.flatnonzero(groups==g)
        first=rng.integers(0,len(cells),1024)
        offset=rng.integers(1,len(cells),1024)
        us.extend(cells[first])
        vs.extend(cells[(first+offset)%len(cells)])
    u,v=np.array(us),np.array(vs)
    assert np.all(u!=v)
    values={}
    for name,(start,stop) in [('train',(0,4160)),('evaluation',(4991,5564))]:
        z=torch.as_tensor(r.RAW[ids,start:stop],dtype=torch.float64)
        z=z-z.mean(1,keepdim=True)
        z=z/torch.linalg.vector_norm(z,dim=1,keepdim=True).clamp_min(1e-12)
        c=(z@z.T).numpy()
        observed=float(c[u,v].mean())
        if name=='evaluation':
            null=[]
            for _ in range(999):
                permutation=rng.permutation(2048)
                null.append(float(c[permutation[u],permutation[v]].mean()))
            values[name]=dict(mean_within_group_correlation=observed,
                mean_randomized_correlation=float(np.mean(null)),
                randomized_95_interval=np.quantile(null,[.025,.975]).tolist(),
                descriptive_upper_tail_rank=float((1+sum(x>=observed for x in null))/1000),
                interpretation='fixed-population descriptive association; not a synapse or biological significance test')
        else:
            values[name]=dict(mean_within_group_correlation=observed)
    return dict(pool=pool,pairs_per_group=1024,**values)


def main():
    assert not (r.ROOT/'results.json').exists(), 'read the completed report instead of repeating evaluation'
    protocol=r.verify()
    records=[json.loads((r.ROOT/'runs'/f'{r.stem(t)}.json').read_text()) for t in protocol['tasks']]
    assert len(records)==18 and all(not x['evaluation_scored'] for x in records)
    r.initialize()
    error=np.empty((3,2,3,573))
    raw_error=np.empty_like(error)
    evaluations=[]
    stability=[]
    reload_error=0.
    for pi,pool in enumerate(r.POOLS):
        ids,arrays,meta=r.data(pool)
        xs,ys=arrays['selection']
        xe,ye=arrays['evaluation']
        with np.load(r.ROOT/f'groups_{pool}.npz') as g:
            np.testing.assert_array_equal(ids,g['ids'])
            group={c:g['random' if c=='random' else 'functional'] for c in r.CONDITIONS}
        stability.append(group_stability(pool,ids,group['functional']))
        for si,seed in enumerate(r.SEEDS):
            initial=[]
            for ci,condition in enumerate(r.CONDITIONS):
                task=dict(pool=pool,seed=seed,condition=condition)
                name=r.stem(task)
                path=r.ROOT/'runs'/f'{name}.json'
                record=json.loads(path.read_text())
                initial.append(record['initial_state_sha256'])
                ck=torch.load(path.with_suffix('.pt'),map_location='cpu',weights_only=False)
                np.testing.assert_array_equal(ck['ids'],ids)
                np.testing.assert_array_equal(ck['groups'],group[condition])
                assert ck['normalization']==meta and record['normalization']==meta
                assert record['parameters']==81377
                assert record['best_epoch']==min(record['history'],key=lambda h:h['mse'])['epoch']
                net=r.make_model(seed,condition,group[condition])
                assert r.state_hash(net)==record['initial_state_sha256']
                pos=torch.zeros(2048,3)
                untrained=r.score(r.s.predict(net,xe,pos),ye.numpy(),meta)
                net.load_state_dict(ck['state_dict'])
                assert all(torch.isfinite(v).all() for v in net.state_dict().values())
                selected=r.s.predict(net,xs,pos)
                with np.load(path.with_suffix('.npz')) as old:
                    np.testing.assert_array_equal(old['selection_target'],ys.numpy())
                    discrepancy=float(np.max(np.abs(selected-old['selection_prediction'])))
                    assert discrepancy<2e-5
                    reload_error=max(reload_error,discrepancy)
                assert abs(r.score(selected,ys.numpy(),meta)['mse']-record['selection']['mse'])<1e-7
                pred=r.s.predict(net,xe,pos)
                target=ye.numpy().astype(np.float64)
                bounded=np.maximum(pred.astype(np.float64),-meta['speed_mean']/meta['speed_std'])
                error[pi,si,ci]=(bounded-target)**2
                raw_error[pi,si,ci]=(pred.astype(np.float64)-target)**2
                metric=r.score(pred,target,meta)
                independent=float(torch.mean((torch.as_tensor(bounded)-torch.as_tensor(target))**2))
                assert abs(independent-metric['mse'])<1e-12
                np.savez_compressed(r.ROOT/'runs'/f'{name}_evaluation.npz',prediction=pred,target=target,
                    time_index=np.arange(4991,5564))
                evaluations.append(dict(task=task,selection=record['selection'],evaluation=metric,
                    untrained_evaluation=untrained,constant_mse=float(np.mean(target**2)),
                    best_epoch=record['best_epoch'],epochs_run=record['epochs_run'],seconds=record['seconds']))
            assert len(set(initial))==1
        del arrays
    means=error.mean(axis=(0,1,3))
    paired=error.mean(axis=3)
    contrasts={}
    for ci,condition in [(1,'random'),(2,'global')]:
        contrast=paired[:,:,ci]-paired[:,:,0]
        contrasts[condition]=dict(relative_improvement=float(1-means[0]/means[ci]),
            mse_difference=float(means[ci]-means[0]),pair_wins=int((contrast>0).sum()),
            pool_mean_differences=contrast.mean(1).tolist(),positive_pools=int((contrast.mean(1)>0).sum()))
    rng=np.random.default_rng(20261004)
    boot=[]
    for _ in range(4000):
        pools=rng.integers(0,3,3)
        seeds=rng.integers(0,2,2)
        times=((rng.integers(0,573,6)[:,None]+np.arange(100))%573).ravel()[:573]
        m=np.take(np.take(np.take(error,pools,axis=0),seeds,axis=1),times,axis=3).mean(axis=(0,1,3))
        boot.append([1-m[0]/m[1],1-m[0]/m[2]])
    intervals=np.quantile(boot,[.0125,.9875],axis=0).T
    for i,c in enumerate(['random','global']):
        contrasts[c]['conditional_97_5_interval']=intervals[i].tolist()
    promising=all(v['relative_improvement']>=.02 and v['pair_wins']>=5 and v['positive_pools']==3 for v in contrasts.values())
    stronger=promising and bool((intervals[:,0]>0).all())
    raw_means=raw_error.mean(axis=(0,1,3))
    result=dict(mean_evaluation_mse=dict(zip(r.CONDITIONS,means.tolist())),
        mean_raw_mse=dict(zip(r.CONDITIONS,raw_means.tolist())),contrasts=contrasts,
        promising_gate=bool(promising),stronger_exploratory_gate=bool(stronger),
        decision='promising exploratory evidence' if promising else 'added predictive benefit not established',
        group_stability=stability,evaluations=evaluations,
        beat_untrained=sum(x['evaluation']['mse']<x['untrained_evaluation']['mse'] for x in evaluations),
        beat_constant=sum(x['evaluation']['mse']<x['constant_mse'] for x in evaluations),
        selected_budget_limit=sum(x['best_epoch']==24 for x in evaluations),
        selection_reload_max_error=reload_error,test_evaluations=0)
    r.s.write_json(r.ROOT/'results.json',result)
    r.s.write_json(r.ROOT/'completion_checks.json',dict(passed=True,
        all_18_checkpoints_fixed_before_evaluation=True,full_selection_reload_all18=True,
        independent_evaluation_mse_recomputed=True,finite_checkpoints=True,equal_parameter_counts=True,
        paired_initial_states=True,cell_order_and_masks_checked=True,application_and_sources_unchanged=True,
        selection_reload_max_error=reload_error,test_evaluations=0))
    with (r.ROOT/'per_run.csv').open('w') as f:
        writer=csv.writer(f)
        writer.writerow(['pool','seed','condition','selection_mse','evaluation_mse','evaluation_raw_mse','evaluation_r2','untrained_mse','constant_mse','best_epoch'])
        for x in evaluations:
            writer.writerow([x['task'][k] for k in ['pool','seed','condition']]+[
                x['selection']['mse'],x['evaluation']['mse'],x['evaluation']['raw_mse'],x['evaluation']['r2'],
                x['untrained_evaluation']['mse'],x['constant_mse'],x['best_epoch']])
    print(json.dumps({k:v for k,v in result.items() if k not in ['evaluations','group_stability']},indent=2))


if __name__=='__main__':
    main()
