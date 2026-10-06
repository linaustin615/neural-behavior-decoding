"""Predeclared secondary inference timing; never used for model selection."""
import os
import platform
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from common import ROOT, read, write, digest, config_id
from models import make_model
import data
import fit


def main():
    plan = read(ROOT/'cost_plan.json')
    assert digest(ROOT/'cost.py')==plan['source_sha256']
    lock = read(ROOT/'evaluation_lock.json')
    for name,value in lock['hashes'].items():
        assert digest(ROOT/name)==value
    selection = read(ROOT/'final_selection.json')
    configs = {config_id(c):c for c in selection['roles'].values()}
    seq = np.load(ROOT/'prepared'/'MP030'/'full'/'validation_seq.npy')
    inputs = {h:torch.from_numpy(data.windows(seq,h)[:64]) for h in [16,32,64]}
    torch.set_num_threads(2)
    records = []
    rng = np.random.default_rng(20261006)
    with threadpool_limits(limits=2), torch.inference_mode():
        for seed in range(201,207):
            nets, states = {}, {}
            for cid,c in configs.items():
                job = dict(config=c,split='full',seed=seed,epochs=c['epochs'])
                states[cid] = torch.load(ROOT/'fits'/fit.job_id(job)/'selected.pt',weights_only=True)
                nets[cid] = make_model(c,seed).eval()
                nets[cid].load_state_dict(states[cid])
            for batch in [1,64]:
                expected = {}
                for cid,c in configs.items():
                    x = inputs[c['history']][:batch]
                    for _ in range(5):
                        expected[cid] = nets[cid](x,0)
                durations = {cid:[] for cid in configs}
                for _ in range(30):
                    for cid in rng.permutation(list(configs)):
                        x = inputs[configs[cid]['history']][:batch]
                        start = time.perf_counter()
                        pred = nets[cid](x,0)
                        durations[cid].append(time.perf_counter()-start)
                        torch.testing.assert_close(pred,expected[cid],rtol=0,atol=0)
                for cid,values in durations.items():
                    assert len(values)==30 and min(values)>0
                    records.append(dict(config_id=cid,seed=seed,batch=batch,median_seconds=float(np.median(values)),measurements_seconds=values))
            for cid,net in nets.items():
                assert all(torch.equal(v,states[cid][k]) for k,v in net.state_dict().items())
    summaries = []
    for role,c in selection['roles'].items():
        cid = config_id(c)
        for batch in [1,64]:
            values = [r['median_seconds'] for r in records if r['config_id']==cid and r['batch']==batch]
            assert len(values)==6
            summaries.append(dict(role=role,config_id=cid,batch=batch,median_across_seed_medians=float(np.median(values))))
    write(ROOT/'cost_results.json',dict(passed=True,records=records,summaries=summaries,selection_unchanged=True,
        scope='Secondary CPU neural-forward timing on development inputs; no preprocessing,I/O,GPU,physical real-time or model-selection claim.',
        environment=dict(machine=platform.machine(),os=platform.platform(),logical_cpus=os.cpu_count(),torch_version=torch.__version__,torch_threads=2),
        plan_sha256=digest(ROOT/'cost_plan.json')))
    print('Secondary interleaved inference timings complete',flush=True)


if __name__=='__main__':
    main()
