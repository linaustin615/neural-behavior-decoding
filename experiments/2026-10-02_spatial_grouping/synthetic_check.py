from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from threadpoolctl import threadpool_limits
from group_model import GroupCandidate, grouping

ROOT = Path(__file__).resolve().parent
threadpool_limits(limits=1)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
protocol = dict(purpose='Can the exact grouped architecture learn a known interaction between regional signals?',
                n=256, train_examples=2048, val_examples=512, test_examples=512,
                seeds=[20,21], variants=['global','spatial','random'], max_epochs=50,
                patience=10, min_epochs=20, data_seed=721,
                target='(regional_drive0-regional_drive7)*(regional_drive2-regional_drive5)/2; normalize by train mean/std',
                activity='Each neuron shares its region signal with independent Gaussian noise SD.5 at each of8 time bins',
                gate='Both spatial seeds testR2>.75 and beat their untrained predictions; other arms need not fail',
                limitation='Artificial strong group interaction; not evidence of anatomical benefit in real data')
(ROOT/'synthetic_protocol.json').write_text(json.dumps(protocol,indent=2))
rng = np.random.default_rng(721)
pos = rng.uniform(-1, 1, (256, 3))
group, _ = grouping(pos, 0, 'spatial')
drive = rng.normal(size=(3072,8)).astype(np.float32)
x = drive[:,group,None] + rng.normal(0,.5,(3072,256,8)).astype(np.float32)
y = (drive[:,0]-drive[:,7])*(drive[:,2]-drive[:,5])/2
y = (y-y[:2048].mean())/y[:2048].std()
x, y = torch.from_numpy(x), torch.from_numpy(y)
positions = torch.tensor(pos, dtype=torch.float32)
ids = torch.arange(256)


def predict(model, values):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(values[i:i+64], positions, ids) for i in range(0,len(values),64)])


results = []
for seed in protocol['seeds']:
    for variant in protocol['variants']:
        started = time.monotonic()
        g, _ = grouping(pos, 0, variant)
        config = dict(n=256,window=8,width=32,family='latent',temporal='mlp',readout='mean',
                      latents=16,variant=variant,initialization_seed=seed)
        torch.manual_seed(seed)
        model = GroupCandidate(config,g)
        initial_hash=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
        initial = float(((predict(model,x[2560:])-y[2560:])**2).mean())
        best = float(((predict(model,x[2048:2560])-y[2048:2560])**2).mean())
        best_state, best_epoch, stale = deepcopy(model.state_dict()), 0, 0
        loader=DataLoader(TensorDataset(x[:2048],y[:2048]),batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(seed))
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,50,eta_min=.0001)
        history=[]
        for epoch in range(1,51):
            model.train()
            for xb,yb in loader:
                opt.zero_grad(set_to_none=True)
                loss=((model(xb,positions,ids)-yb)**2).mean()
                assert torch.isfinite(loss)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                opt.step()
            val=float(((predict(model,x[2048:2560])-y[2048:2560])**2).mean())
            history.append(dict(epoch=epoch,val_mse=val))
            if val<best:
                best,best_state,best_epoch,stale=val,deepcopy(model.state_dict()),epoch,0
            else:
                stale+=1
            scheduler.step()
            if epoch>=20 and stale>=10:
                break
        model.load_state_dict(best_state)
        pred=predict(model,x[2560:])
        mse=float(((pred-y[2560:])**2).mean())
        r2=1-mse/float(y[2560:].var(unbiased=False))
        record=dict(seed=seed,variant=variant,initial_state_sha256=initial_hash,untrained_test_mse=initial,
                    val_mse=best,test_mse=mse,test_r2=r2,best_epoch=best_epoch,epochs_run=epoch,
                    history=history,seconds=time.monotonic()-started)
        results.append(record)
        torch.save(dict(config=config,state_dict=best_state,groups=g.tolist(),positions=positions),ROOT/f'synthetic_{variant}_{seed}.pt')
        np.savez_compressed(ROOT/f'synthetic_{variant}_{seed}.npz',prediction=pred.numpy(),target=y[2560:].numpy())
        (ROOT/'synthetic_results.json').write_text(json.dumps(dict(protocol=protocol,runs=results,complete=len(results)==6),indent=2))
        print(json.dumps({k:v for k,v in record.items() if k!='history'}),flush=True)
assert all(len({r['initial_state_sha256'] for r in results if r['seed']==seed})==1 for seed in protocol['seeds'])
print('SYNTHETIC_COMPLETE',flush=True)
