"""Fixed top16 retrieval and validation-selected convex hybrids; no neural fits."""
import argparse
from pathlib import Path

import numpy as np
import torch

import run

OUT = Path(__file__).resolve().parent/'local16'
ALPHAS = (0., .25, .5, .75, 1.)


def check():
    run.check_lock()
    lock = run.read(run.ROOT/'selection_lock.json')
    for name, expected in lock['cache_sha256'].items():
        assert run.digest(run.ROOT/'cache'/name)==expected
    if (OUT/'protocol.json').exists():
        assert run.read(OUT/'protocol.json')['source_sha256']==run.digest(__file__)


def retrieve(query, bank, labels, taus):
    labels = torch.as_tensor(labels,dtype=torch.float32)
    result = [[] for _ in taus]
    for start in range(0,len(query),256):
        values, indices = (query[start:start+256]@bank.T).topk(16,dim=1)
        for out,tau in zip(result,taus):
            out.append((torch.softmax(values/tau,dim=1)*labels[indices]).sum(1).numpy())
    result = [np.concatenate(p) for p in result]
    for p in result:
        assert np.isfinite(p).all() and p.min()>=float(labels.min())-1e-5 and p.max()<=float(labels.max())+1e-5
    return result


def predictions(mouse,family,seed,split,taus):
    data = run.examples(mouse,split)
    train = run.examples(mouse,'train')
    suffix = f'{mouse}_{family}'+(f'_{seed}' if seed else '')
    with np.load(run.ROOT/'cache'/(suffix+'.npz')) as c:
        if seed:
            f,head = run.features(run.network(mouse,family,seed),data)
        else:
            f = ((torch.from_numpy(run.flat(data))-torch.from_numpy(c['center']))@torch.from_numpy(c['basis'])).numpy()
            head = None
        p = retrieve(run.normalized(f,c['mean'],c['scale']),torch.from_numpy(c['bank']),train.y[train.indices],taus)
    return data,p,head


def blend(parent,local,alpha,lower):
    parent = np.maximum(np.asarray(parent,dtype=np.float64),lower)
    local = np.maximum(np.asarray(local,dtype=np.float64),lower)
    return (1-alpha)*parent+alpha*local


def select():
    check()
    OUT.mkdir(exist_ok=False)
    run.save(OUT/'protocol.json',dict(utc=run.utc(),source_sha256=run.digest(__file__),
        parent_selection_sha256=run.digest(run.ROOT/'selection_lock.json'),
        status='new exploratory follow-up on already examined mice; original FAIL unchanged',
        k=16,taus=run.TAUS,alphas=ALPHAS,neural_fits=0,
        features='unchanged frozen transformer, MLP and PCA64 bank and scaling from Phase0',
        selection='per mouse/family: select top16 temperature by mean validation clipped MSE across seeds; then select alpha with that temperature by same validation objective; ascending-grid ties; PCA once per mouse',
        hybrid='(1-alpha)*clipped original head + alpha*clipped same-encoder top16 retrieval; alpha0 retains parent; both families tested, no test-based advancement',
        barrier='all local temperatures and hybrid weights locked before any new test scoring',
        gates='each neural local/hybrid: >=5% equal-mouse relative MSE gain vs parent, >=5/7 wins, maxharm<=10%, QFM lower>=5/7, positive mean gain vs PCA top16; MAE and active MSE descriptive',
        limits='no new animals, causal claim, dynamic gate, or validated superiority; no grid extension after results'))
    choices = {}
    for mouse in run.MICE:
        choices[mouse] = {}
        for family in ('transformer','mlp','pca'):
            scores, cache = [], []
            for seed in (run.SEEDS if family!='pca' else (None,)):
                data,p,head = predictions(mouse,family,seed,'validation',run.TAUS)
                y,lower = data.y[data.indices],data.metadata['lower']
                scores.append([run.metrics(v,y,lower)['mse'] for v in p])
                cache.append((p,head,y,lower))
            index = int(np.argmin(np.mean(scores,axis=0)))
            choice = dict(tau=run.TAUS[index],validation_mse_grid=scores)
            if family!='pca':
                losses = [[run.metrics(blend(head,p[index],a,lo),y,lo)['mse'] for a in ALPHAS] for p,head,y,lo in cache]
                choice.update(alpha=ALPHAS[int(np.argmin(np.mean(losses,axis=0)))],blend_validation_mse_grid=losses)
            choices[mouse][family] = choice
        run.save(OUT/'selection_progress.json',choices)
        print(mouse,'top16 and hybrid validation locked',flush=True)
    run.save(OUT/'selection_lock.json',dict(utc=run.utc(),protocol_sha256=run.digest(OUT/'protocol.json'),choices=choices))


def evaluate():
    check()
    lock = run.read(OUT/'selection_lock.json')
    assert lock['protocol_sha256']==run.digest(OUT/'protocol.json')
    out = OUT/'predictions'
    out.mkdir(exist_ok=False)
    rows = run.read(run.ROOT/'summary.json')['rows'].copy()
    differences = []
    for mouse in run.MICE:
        for family in ('transformer','mlp','pca'):
            choice = lock['choices'][mouse][family]
            for seed in (run.SEEDS if family!='pca' else (None,)):
                data,p,head = predictions(mouse,family,seed,'test',[choice['tau']])
                p = p[0]
                y,lower = data.y[data.indices],data.metadata['lower']
                suffix = f'{mouse}_{family}'+(f'_{seed}' if seed else '')
                arrays = dict(prediction=p,target=y)
                rows.append(dict(mouse=mouse,arm=family+'_local16',seed=seed,**run.metrics(p,y,lower)))
                if seed:
                    with np.load(run.ROOT/'predictions'/(suffix+'.npz')) as old:
                        np.testing.assert_array_equal(y,old['target'])
                        np.testing.assert_allclose(head,old['parent'],atol=2e-6,rtol=2e-6)
                        differences.append(float(np.max(np.abs(head-old['parent']))))
                    hybrid = blend(head,p,choice['alpha'],lower)
                    arrays.update(parent=head,hybrid=hybrid)
                    rows.append(dict(mouse=mouse,arm=family+'_hybrid',seed=seed,**run.metrics(hybrid,y,lower)))
                np.savez_compressed(out/(suffix+'.npz'),**arrays)
        run.save(OUT/'test_progress.json',rows)
        print(mouse,'top16 and hybrid scoring complete',flush=True)
    pairs = []
    for family in ('transformer','mlp'):
        for kind in ('local16','hybrid'):
            pairs += [(family+'_'+kind,c) for c in (family+'_parent',family+'_retrieval','pca_local16','zero')]
        pairs.append((family+'_hybrid',family+'_local16'))
    pairs += [('transformer_'+kind,'mlp_'+kind) for kind in ('local16','hybrid')]
    contrasts = {a+'_vs_'+b:run.contrast(rows,a,b) for a,b in pairs}
    gates = {}
    for family in ('transformer','mlp'):
        for kind in ('local16','hybrid'):
            arm = family+'_'+kind
            c = contrasts[arm+'_vs_'+family+'_parent']
            checks = dict(mean_gain=c['mean_gain']>=.05,mouse_wins=c['mouse_wins']>=5,
                harm=c['max_harm']<=.1,quiet_wins=c['quiet_wins']>=5,pca_gain=contrasts[arm+'_vs_pca_local16']['mean_gain']>0)
            gates[arm] = dict(passed=all(checks.values()),checks=checks)
    run.save(OUT/'summary.json',dict(utc=run.utc(),rows=rows,contrasts=contrasts,gates=gates,
        exploratory=True,neural_fits=0,max_parent_prediction_difference=max(differences)))
    print(gates,flush=True)


def selftest():
    rng = np.random.default_rng(81)
    bank = rng.normal(size=(33,5)); bank/=np.linalg.norm(bank,axis=1,keepdims=True)
    query = rng.normal(size=(7,5)); query/=np.linalg.norm(query,axis=1,keepdims=True)
    labels = rng.uniform(-1,4,33)
    actual = retrieve(torch.tensor(query,dtype=torch.float32),torch.tensor(bank,dtype=torch.float32),labels,[.03,.3])
    similarity = np.einsum('ik,jk->ij',query,bank,optimize=False)
    idx = np.argsort(-similarity,axis=1)[:,:16]
    for tau,p in zip((.03,.3),actual):
        logits = np.take_along_axis(similarity,idx,axis=1)/tau
        w = np.exp(logits-logits.max(1,keepdims=True))
        expected = (w*labels[idx]).sum(1)/w.sum(1)
        np.testing.assert_allclose(p,expected,atol=3e-6,rtol=3e-6)
    np.testing.assert_array_equal(blend(np.array([-3.,2.]),np.array([1.,4.]),0.,-1),[-1.,2.])
    np.testing.assert_array_equal(blend(np.array([-3.,2.]),np.array([1.,4.]),1.,-1),[1.,4.])
    print('PASS independent top16 arithmetic and clipped blend endpoints')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['selftest','select','evaluate'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    {'selftest':selftest,'select':select,'evaluate':evaluate}[args.stage]()
