"""Check saved artifacts without rerunning fits or changing selections."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
FAIR=ROOT.parent/'2026-10-03_fair_comparison'


def read(path):return json.loads(path.read_text())


def main():
    protocol=read(ROOT/'protocol.json');short=read(ROOT/'shortlist.json');lock=read(ROOT/'final_lock.json')
    for paths in [protocol['application_hashes'],protocol['reference_hashes'],lock['files']]:
        for name,expected in paths.items():assert hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==expected,name
    for name,expected in protocol['sources'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    for name,expected in read(ROOT/'prepared.json')['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    assert protocol['created_utc']<short['locked_utc']<lock['locked_utc']
    assert len(short['finalists'])==2
    expected_runs={('shared' if kind=='shared' else mouse,kind,10):12 for kind in protocol['screen_candidates'] for mouse in protocol['mice']}
    for mouse in protocol['mice']:
        for kind in short['refine_families']:
            for seed in protocol['refinement_seeds']:
                if kind in ['plain','pooled_mlp'] and seed in [10,11]:continue
                expected_runs[('shared' if kind=='shared' else mouse,kind,seed)]=24
    counts=dict(new_fits=0,supervised_epochs=0,reconstruction_epochs=0,selection_predictions_checked=0)
    orders={};screen_rows={kind:[] for kind in protocol['screen_candidates']}
    for mouse in protocol['mice']:
        for family in ['transformer','pooled_mlp']:
            for seed in [10,11]:
                r=read(FAIR/mouse/f'{family}_r0_s{seed}'/'result.json')
                key=(mouse,seed)
                if key in orders:assert orders[key]==r['batch_order_hashes']
                orders[key]=r['batch_order_hashes']
    for path in sorted(ROOT.glob('*/*/result.json')):
        result=read(path);out=path.parent;history=read(out/'history.json')
        assert expected_runs.pop((result['mouse'],result['kind'],result['seed']))==result['epochs']
        assert result['selected_reload_exact'] and result['grad_max']>0
        assert len(history)==result['epochs']+1
        counts['new_fits']+=1;counts['supervised_epochs']+=result['epochs']
        if (out/'pretraining.json').exists():counts['reconstruction_epochs']+=read(out/'pretraining.json')['epochs']
        key=(result['mouse'],result['seed']);current=[r['batch_order_hash'] for r in history[1:]]
        if key in orders:assert current==orders[key][:len(current)],str(path)
        else:orders[key]=current
        for mouse,chosen in result['chosen'].items():
            stats=read(FAIR/mouse/'metadata.json');lower=-stats['speed_mean']/stats['speed_std']
            with np.load(out/f'{mouse}_selection.npz') as saved:
                errors=np.mean((np.maximum(saved['prediction'].astype(np.float64),lower)-saved['target'])**2,axis=1)
            expected=[r['selection'][mouse]['mse'] for r in history]
            np.testing.assert_allclose(errors,expected,rtol=1e-10,atol=1e-10)
            assert int(np.argmin(errors))==chosen
            counts['selection_predictions_checked']+=len(errors)
            if result['seed']==10 and result['kind'] in screen_rows:
                epoch=int(np.argmin(errors[:13]));baseline=read(ROOT/mouse/'baselines.json')['strong_selection_mse']
                screen_rows[result['kind']].append(dict(mouse=mouse,epoch=epoch,ratio=float(errors[epoch]/baseline)))
    assert not expected_runs,expected_runs
    reconstructed=[]
    for kind,rows in screen_rows.items():
        assert len(rows)==4,kind
        reconstructed.append(dict(kind=kind,mean_ratio=float(np.mean([r['ratio'] for r in rows])),trained_mice=sum(r['epoch']>0 for r in rows)))
    reconstructed.sort(key=lambda r:(r['mean_ratio'],r['kind']))
    for a,b in zip(reconstructed,short['ranking']):
        assert a['kind']==b['kind'] and a['trained_mice']==b['trained_mice']
        assert abs(a['mean_ratio']-b['mean_ratio'])<1e-9
    eligible=[r['kind'] for r in reconstructed if r['trained_mice']>=3][:2]
    eligible+=[r['kind'] for r in reconstructed if r['kind'] not in eligible][:2-len(eligible)]
    assert eligible==short['finalists']
    output=dict(passed=True,counts=counts,matched_batch_orders=True,screen_ranking_independently_reconstructed=True,all_new_epoch_scores_checked=True,frozen_sources_and_references_unchanged=True,selected_checkpoints_unchanged=True)
    (ROOT/'completion_audit.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
