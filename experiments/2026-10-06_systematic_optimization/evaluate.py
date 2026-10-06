"""Locked later-period comparison; no architecture choices are made here."""
import itertools
import math
import numpy as np
import torch
from common import ROOT, FAIR, PANEL, MICE, read, write, digest, config_id, verify
import data
import fit
import ridge
from models import make_model


def metrics(pred, target, lower):
    bounded = np.maximum(np.asarray(pred, dtype=np.float64), lower)
    y = np.asarray(target, dtype=np.float64)
    assert bounded.ndim == 2 and bounded.shape[1] == len(y) and np.isfinite(bounded).all()
    single_mse, single_mae, pair_mse, pair_mae = [], [], [], []
    checks = 0
    for family, items in [('single', list(bounded)), ('pair', [(bounded[i]+bounded[j])/2 for i,j in itertools.combinations(range(len(bounded)), 2)])]:
        for p in items:
            error = p-y
            mse, mae = float(np.mean(error**2)), float(np.mean(np.abs(error)))
            manual_mse = sum((float(a)-float(b))**2 for a,b in zip(p,y))/len(y)
            manual_mae = sum(abs(float(a)-float(b)) for a,b in zip(p,y))/len(y)
            np.testing.assert_allclose([mse,mae], [manual_mse,manual_mae], rtol=1e-12, atol=1e-12)
            if family == 'single':
                single_mse.append(mse)
                single_mae.append(mae)
            else:
                pair_mse.append(mse)
                pair_mae.append(mae)
            checks += 2
    variance = float(np.mean((y-y.mean())**2))
    return dict(mse=float(np.mean(single_mse)), mae=float(np.mean(single_mae)), r2=1-float(np.mean(single_mse))/variance,
        single_mse=single_mse, single_mae=single_mae, pair_mse=float(np.mean(pair_mse)), pair_mae=float(np.mean(pair_mae))), checks


def contrast(rows, control):
    gains = [1-r['scores']['optimized_attention']['mse']/r['scores'][control]['mse'] for r in rows]
    mae = [1-r['scores']['optimized_attention']['mae']/r['scores'][control]['mae'] for r in rows]
    a = np.array([r['scores']['optimized_attention']['single_mse'] for r in rows])
    b = np.array([r['scores'][control]['single_mse'] for r in rows])
    wins = int((a<b).sum())
    n = a.shape[1]
    gate = np.mean(gains)>=.05 and sum(v>0 for v in gains)>=3 and wins>=math.ceil(8*n/3) and min(gains)>=-.1 and np.mean(mae)>=0
    return dict(mean_mse_gain=float(np.mean(gains)), mean_mae_gain=float(np.mean(mae)), mouse_mse_gains=gains,
        mouse_mae_gains=mae, mouse_wins=sum(v>0 for v in gains), seed_wins=wins, practical_gain_passed=bool(gate),
        pair_mean_mse_gain=float(np.mean([1-r['scores']['optimized_attention']['pair_mse']/r['scores'][control]['pair_mse'] for r in rows])),
        leave_one_mouse_out=[float(np.mean(gains[:i]+gains[i+1:])) for i in range(4)])


def main():
    protocol = verify()
    lock = read(ROOT/'evaluation_lock.json')
    for name,value in lock['hashes'].items():
        assert digest(ROOT/name)==value, name
    selected = read(ROOT/'final_selection.json')
    roles = selected['roles']
    configs = {config_id(c):c for c in roles.values()}
    panel = np.load(PANEL)
    prediction_hashes = {}
    prediction_count = 0
    for session, mouse in enumerate(MICE):
        prepared = ROOT/'prepared'/mouse/'full'
        meta = read(prepared/'metadata.json')
        with np.load(FAIR/mouse/'later_raw.npz') as raw, np.load(FAIR/mouse/'statistics.npz') as old:
            sequence = ((raw['activity'][panel,24:]-old['activity_mean'][panel])/old['activity_std'][panel]).T
            target = ((raw['speed'][87:]-meta['original_speed_mean'])/meta['original_speed_std']-meta['speed_mean'])/meta['speed_std']
            physical_target = raw['speed'][87:]
        sequence = ((sequence-np.load(prepared/'activity_mean.npy'))/np.load(prepared/'activity_std.npy')).astype(np.float32)
        cache = {history:torch.from_numpy(data.windows(sequence,history)) for history in [16,32,64]}
        assert all(len(x)==len(target) for x in cache.values())
        predictions = dict(target=target, physical_target=physical_target)
        for cid,c in configs.items():
            x = cache[c['history']]
            before = x.clone()
            for seed in range(201,207):
                job = dict(config=c, split='full', seed=seed, epochs=c['epochs'])
                path = ROOT/'fits'/fit.job_id(job)/'selected.pt'
                state = torch.load(path,weights_only=True)
                net = make_model(c,seed)
                net.load_state_dict(state)
                p = fit.predict(net,x,session)
                assert all(torch.equal(value,state[key]) for key,value in net.state_dict().items())
                predictions[f'{cid}_s{seed}'] = p
                prediction_count += len(p)
            assert torch.equal(x,before)
        rec = next(r for r in read(ROOT/'ridge_final_lock.json')['rows'] if r['mouse']==mouse)
        assert digest(ROOT/rec['path'])==rec['sha256']
        with np.load(ROOT/rec['path']) as state:
            predictions['ridge'] = ridge.predict(cache[rec['config']['history']].numpy(),state)
        prediction_count += len(target)
        out = ROOT/'later'/mouse
        out.mkdir(parents=True)
        path = out/'predictions.npz'
        np.savez_compressed(path,**predictions)
        prediction_hashes[str(path.relative_to(ROOT))] = digest(path)
        print(mouse,'locked later comparison scored',flush=True)
    groups = list(roles)+['ridge']
    subsets = {}
    scalar_checks = 0
    for name,seeds in [('first',[201,202,203]),('second',[204,205,206]),('all',list(range(201,207)))]:
        rows = []
        for mouse in MICE:
            meta = read(ROOT/'prepared'/mouse/'full'/'metadata.json')
            scores = {}
            with np.load(ROOT/'later'/mouse/'predictions.npz') as z:
                y = z['target']
                for group in groups:
                    raw = np.stack([z['ridge'] if group=='ridge' else z[f'{config_id(roles[group])}_s{seed}'] for seed in seeds])
                    scores[group],n = metrics(raw,y,meta['lower'])
                    scalar_checks += n
                initial_mse = float(np.mean(y**2))
            rows.append(dict(mouse=mouse,n=len(y),scores=scores,initial_mse=initial_mse,
                physical_speed_scale=meta['speed_std']*meta['original_speed_std']))
        contrasts = {group:contrast(rows,group) for group in groups if group!='optimized_attention'}
        initial_wins = sum(sum(x<row['initial_mse'] for x in row['scores']['optimized_attention']['single_mse']) for row in rows)
        subsets[name] = dict(seeds=seeds,rows=rows,contrasts=contrasts,initial_wins=initial_wins,
            optimization_gate=contrasts['default_attention']['practical_gain_passed'] and initial_wins>=math.ceil(8*len(seeds)/3))
    write(ROOT/'results.json',dict(subsets=subsets,optimization_gain_passed=all(v['optimization_gate'] for v in subsets.values()),
        independent_animal_significance=False,primary_aggregation='Mean of individual-seed errors,not ensemble predictions;pair errors secondary',roles=roles))
    for name,value in lock['hashes'].items():
        assert digest(ROOT/name)==value
    verify()
    write(ROOT/'evaluation_audit.json',dict(passed=True,scalar_errors_checked=scalar_checks,new_predictions=prediction_count,
        prediction_hashes=prediction_hashes,selected_artifacts_checked=len(lock['hashes']),source_hashes_checked=len(protocol['source_hashes']),
        states_and_inputs_unchanged=True,common_targets_start_at_original_raw_index=87,distinct_animals=4))
    print('Optimization practical gain:',all(v['optimization_gate'] for v in subsets.values()),flush=True)


if __name__ == '__main__':
    fit.initialize_worker()
    main()
