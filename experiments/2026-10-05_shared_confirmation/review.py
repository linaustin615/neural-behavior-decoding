"""Check fresh-seed gates, training orders and artifact provenance."""
import math
import numpy as np
import run


def main():
    p=run.verify();locked=run.verify_lock();r=run.read(run.ROOT/'results.json');checks=0
    for seed in run.SEEDS:
        a=run.read(run.native_dir('attention',seed)/'history.json')
        b=run.read(run.native_dir('mlp',seed)/'history.json')
        assert [v['global_order_hash'] for v in a[1:]]==[v['global_order_hash'] for v in b[1:]]
        a=run.read(run.continuation_dir('attention',seed)/'result.json')
        b=run.read(run.continuation_dir('mlp',seed)/'result.json')
        assert a['order_hashes']==b['order_hashes']
        for rec in [a,b]:
            assert rec['rate']=='lr1e4' and rec['epochs']==8
            assert rec['initial_tensors_and_predictions_exact'] and rec['selected_reload_exact']
            assert rec['starting_checkpoint_sha256']==run.digest(run.native_dir(rec['family'],seed)/'selected.pt')
        checks+=6
    primary=['native_attention_vs_native_mlp','native_attention_vs_tuned_mlp','native_attention_vs_ridge']
    for name,part in r['subsets'].items():
        seeds=part['seeds'];rows=part['rows']
        assert seeds==(run.SEEDS if name=='new' else run.ALL_SEEDS)
        assert part['pairs_per_mouse']==math.comb(len(seeds),2)
        passed=[]
        for contrast,s in part['summaries'].items():
            a,b=contrast.split('_vs_')
            gains=[1-row['scores'][a]['mse']/row['scores'][b]['mse'] for row in rows]
            mae=[1-row['scores'][a]['mae']/row['scores'][b]['mae'] for row in rows]
            wins=sum(sum(x<y for x,y in zip(row['scores'][a]['single_mse'],row['scores'][b]['single_mse'])) for row in rows)
            expected=dict(mean_mse_gain=sum(gains)/4,mean_mae_gain=sum(mae)/4,mouse_mse_gains=gains,mouse_mae_gains=mae,
                seed_wins=wins,mouse_wins=sum(g>0 for g in gains),leave_one_mouse_out=[sum(gains[j] for j in range(4) if j!=i)/3 for i in range(4)])
            for key,val in expected.items():
                np.testing.assert_allclose(s[key],val,rtol=1e-12,atol=1e-14);checks+=1
            gate=sum(gains)/4>=.05 and sum(g>0 for g in gains)>=3 and wins>=math.ceil(8*len(seeds)/3) and min(gains)>=-.1 and sum(mae)/4>=0
            assert s['full_pass']==gate;checks+=1
            if contrast in primary:passed.append(gate)
            for i,seed in enumerate(seeds):
                values=[1-row['scores'][a]['single_mse'][i]/row['scores'][b]['single_mse'][i] for row in rows]
                v=s['individual_seeds'][i]
                assert v['seed']==seed and v['mouse_wins']==sum(x>0 for x in values)
                np.testing.assert_allclose(v['mean_mse_gain'],sum(values)/4,rtol=1e-12,atol=1e-14);checks+=2
        initial=sum(sum(x<row['controls']['training_mean']['mse'] for x in row['scores']['native_attention']['single_mse']) for row in rows)
        assert initial==part['initial_wins']
        assert part['primary_pass']==(all(passed) and initial>=math.ceil(8*len(seeds)/3));checks+=2
    assert r['practical_confirmation_passed']==(r['subsets']['new']['primary_pass'] and r['subsets']['all']['primary_pass'])
    assert len(locked['records'])==12
    for stage in ['native','continuation']:
        assert len(list((run.ROOT/stage).glob('*_s*/result.json')))==6
    for path,value in run.read(run.ROOT/'evaluation.json')['prediction_hashes'].items():assert run.digest(run.ROOT/path)==value
    run.write(run.ROOT/'review.json',dict(passed=True,aggregate_gate_training_checks=checks,
        source_input_hashes=len(p['hashes']),selected_artifact_hashes=len(locked['hashes']),
        new_seeds=[16,17,18],new_native_fits=6,new_continuations=6,old_fits_repeated=0,
        main_application_unchanged=True))
    print('Confirmation review passed:',checks,flush=True)


if __name__=='__main__':main()
