"""Check saved pilot metrics and retrieval arithmetic independently of the runner."""
import json
from pathlib import Path

import numpy as np
import torch

import run


def main():
    torch.set_num_threads(4)
    run.check_lock()
    root = Path(__file__).resolve().parent
    summary = run.read(root/'summary.json')
    lock = run.read(root/'selection_lock.json')
    assert lock['protocol_sha256'] == run.digest(root/'protocol.json')
    for name, expected in lock['cache_sha256'].items():
        assert run.digest(root/'cache'/name) == expected
    metric_checks, references = 0, []
    for mouse in run.MICE:
        data = run.examples(mouse, 'test')
        train = run.examples(mouse, 'train')
        labels = train.y[train.indices]
        lower = data.metadata['lower']
        for family in ('transformer', 'mlp', 'pca'):
            for seed in (run.SEEDS if family != 'pca' else (None,)):
                suffix = f'{mouse}_{family}' + (f'_{seed}' if seed else '')
                saved = np.load(root/'predictions'/(suffix+'.npz'))
                np.testing.assert_array_equal(saved['target'], data.y[data.indices])
                for arm, field in ([(family+'_retrieval','prediction'),(family+'_parent','parent')]
                                   if seed else [('pca','prediction')]):
                    row = next(r for r in summary['rows'] if r['mouse']==mouse and r['arm']==arm and r['seed']==seed)
                    p = np.maximum(saved[field].astype(np.float64),lower)
                    y = saved['target']
                    error = p-y
                    np.testing.assert_allclose(row['mse'], np.dot(error,error)/len(y), rtol=1e-12)
                    np.testing.assert_allclose(row['mae'], np.abs(error).sum()/len(y), rtol=1e-12)
                    quiet = y <= lower+.05
                    assert row['quiet_n'] == int(quiet.sum())
                    if quiet.any():
                        np.testing.assert_allclose(row['qfm'],np.mean(p[quiet]-lower),rtol=1e-12)
                    else:
                        assert row['qfm'] is None
                    metric_checks += 1
                with np.load(root/'cache'/(suffix+'.npz')) as cache:
                    small = run.examples(mouse,'test')
                    small.indices = small.indices[:16]
                    if seed:
                        features, _ = run.features(run.network(mouse,family,seed),small)
                    else:
                        features = ((torch.from_numpy(run.flat(small))-torch.from_numpy(cache['center'])) @ torch.from_numpy(cache['basis'])).numpy()
                    query = (features-cache['mean'])/cache['scale']
                    query /= np.maximum(np.linalg.norm(query,axis=1,keepdims=True),1e-12)
                    bank = cache['bank'].astype(np.float64)
                    logits = np.einsum('ik,jk->ij',query,bank,optimize=False) / lock['choices'][mouse][family]['tau']
                    weights = np.exp(logits-logits.max(1,keepdims=True))
                    expected = np.einsum('ij,j->i',weights,labels,optimize=False)/weights.sum(1)
                    assert np.isfinite(expected).all()
                    difference = float(np.max(np.abs(expected-saved['prediction'][:16])))
                    np.testing.assert_allclose(expected,saved['prediction'][:16],atol=1e-5,rtol=1e-5)
                    references.append(dict(mouse=mouse,family=family,seed=seed,max_difference=difference))
                saved.close()
    for name, result in summary['contrasts'].items():
        candidate, control = name.split('_vs_')
        gains = []
        for mouse in run.MICE:
            a = [r['mse'] for r in summary['rows'] if r['mouse']==mouse and r['arm']==candidate]
            b = [r['mse'] for r in summary['rows'] if r['mouse']==mouse and r['arm']==control]
            gains.append(1-sum(a)/len(a)/(sum(b)/len(b)))
        np.testing.assert_allclose(result['mean_gain'],sum(gains)/len(gains),atol=1e-12)
        assert result['mouse_wins']==sum(g>0 for g in gains)
    run.save(root/'audit.json',dict(passed=True,utc=run.utc(),metric_records=metric_checks,
        reference_queries=16*len(references),independent_float64_retrieval=references,
        contrasts_checked=len(summary['contrasts']),source_and_cache_hashes_unchanged=True,
        arithmetic_note='initial BLAS matmul reference emitted runtime warnings despite matching finite outputs; final independent reference uses direct einsum reductions without BLAS and passed without warnings'))
    print('PASS:',metric_checks,'metric records;',16*len(references),'reference predictions')


if __name__=='__main__':
    main()
