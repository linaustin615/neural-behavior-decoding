"""Independently verify saved local-retrieval results and render their report."""
import numpy as np
import torch

import run
import local16

ROOT = local16.OUT


def audit():
    local16.check()
    lock = run.read(ROOT/'selection_lock.json')
    assert lock['protocol_sha256']==run.digest(ROOT/'protocol.json')
    summary = run.read(ROOT/'summary.json')
    records, references, maximum = 0,0,0.
    for mouse in run.MICE:
        train = run.examples(mouse,'train')
        labels = train.y[train.indices]
        lower = train.metadata['lower']
        for family in ('transformer','mlp','pca'):
            choice = lock['choices'][mouse][family]
            index = int(np.argmin(np.asarray(choice['validation_mse_grid']).mean(0)))
            assert choice['tau']==run.TAUS[index]
            if family!='pca':
                index = int(np.argmin(np.asarray(choice['blend_validation_mse_grid']).mean(0)))
                assert choice['alpha']==local16.ALPHAS[index]
            for seed in (run.SEEDS if family!='pca' else (None,)):
                suffix = f'{mouse}_{family}'+(f'_{seed}' if seed else '')
                with np.load(ROOT/'predictions'/(suffix+'.npz')) as saved, np.load(run.ROOT/'cache'/(suffix+'.npz')) as cache:
                    data = run.examples(mouse,'test')
                    np.testing.assert_array_equal(saved['target'],data.y[data.indices])
                    arms = [(family+'_local16','prediction')]
                    if seed:
                        arms.append((family+'_hybrid','hybrid'))
                        a = choice['alpha']
                        np.testing.assert_array_equal(saved['hybrid'],(1-a)*np.maximum(saved['parent'].astype(float),lower)+a*np.maximum(saved['prediction'].astype(float),lower))
                    for arm,field in arms:
                        row = next(r for r in summary['rows'] if r['mouse']==mouse and r['arm']==arm and r['seed']==seed)
                        p = np.maximum(saved[field].astype(float),lower)
                        y = saved['target']
                        error = p-y
                        np.testing.assert_allclose(row['mse'],np.square(error).sum()/len(y),atol=1e-12)
                        np.testing.assert_allclose(row['mae'],np.abs(error).sum()/len(y),atol=1e-12)
                        quiet = y-lower<=.05
                        if quiet.any(): np.testing.assert_allclose(row['qfm'],(p[quiet]-lower).mean(),atol=1e-12)
                        else: assert row['qfm'] is None
                        records += 1
                    data.indices = data.indices[:8]
                    if seed: features,_ = run.features(run.network(mouse,family,seed),data)
                    else: features = ((torch.from_numpy(run.flat(data))-torch.from_numpy(cache['center']))@torch.from_numpy(cache['basis'])).numpy()
                    query = (features-cache['mean'])/cache['scale']
                    query /= np.maximum(np.linalg.norm(query,axis=1,keepdims=True),1e-12)
                    similarity = np.einsum('ik,jk->ij',query,cache['bank'].astype(float),optimize=False)
                    idx = np.argsort(-similarity,axis=1)[:,:16]
                    logits = np.take_along_axis(similarity,idx,axis=1)/choice['tau']
                    weights = np.exp(logits-logits.max(1,keepdims=True))
                    expected = (weights*labels[idx]).sum(1)/weights.sum(1)
                    np.testing.assert_allclose(expected,saved['prediction'][:8],atol=1e-5,rtol=1e-5)
                    maximum = max(maximum,float(np.max(np.abs(expected-saved['prediction'][:8]))))
                    references += len(expected)
    for name,c in summary['contrasts'].items():
        a,b = name.split('_vs_')
        gains = []
        for mouse in run.MICE:
            ae = [r['mse'] for r in summary['rows'] if r['mouse']==mouse and r['arm']==a]
            be = [r['mse'] for r in summary['rows'] if r['mouse']==mouse and r['arm']==b]
            gains.append(1-(sum(ae)/len(ae))/(sum(be)/len(be)))
        np.testing.assert_allclose(c['mean_gain'],sum(gains)/len(gains),atol=1e-12)
        assert c['mouse_wins']==sum(g>0 for g in gains)
        np.testing.assert_allclose(c['max_harm'],max(0,-min(gains)),atol=1e-12)
    run.save(ROOT/'audit.json',dict(passed=True,utc=run.utc(),metrics_checked=records,
        independent_float64_top16_predictions=references,max_difference=maximum,
        selection_argmins_verified=True,blend_formula_verified=True,
        contrasts_checked=len(summary['contrasts']),locked_sources_and_cache_verified=True))
    print('PASS:',records,'metric records;',references,'independent predictions',flush=True)


def report():
    s = run.read(ROOT/'summary.json')
    choices = run.read(ROOT/'selection_lock.json')['choices']
    lines = ['# Fixed top16 retrieval and convex hybrids','',
        'Exploratory follow-up using the same seven already examined mice and42 frozen checkpoints. '
        'No neural fits or new PCA fits. The neighbor count was fixed at16 before scoring; '
        'temperatures and hybrid weights were selected only on validation, then locked for every mouse. '
        'Previous failed gates remain unchanged.','',
        'Hybrids combine the clipped original head with same-encoder local retrieval: '
        '(1−alpha) × parent + alpha × retrieval. Alpha is selected from0/.25/.5/.75/1, '
        'one choice per mouse/family across the three seeds. Alpha0 retains the original parent.','',
        '## Results','',
        'Positive gain means lower MSE. Average seed errors within mouse first, then average relative gains equally across mice.','',
        '| Comparison | Mean MSE gain | Mouse wins | Worst harm | Quiet wins | MAE gain |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name,c in s['contrasts'].items():
        lines.append(f"| {name.replace('_vs_',' versus ').replace('_',' ')} | {100*c['mean_gain']:+.2f}% | {c['mouse_wins']}/7 | {100*c['max_harm']:.2f}% | {c['quiet_wins']}/7 | {100*c['mean_mae_gain']:+.2f}% |")
    lines += ['', '## Frozen feasibility criteria','',
        'Require≥5%mean MSE gain versus parent,≥5/7mouse wins,≤10%worst harm,≥5/7quiet false-movement wins, '
        'and positive mean MSE gain versus PCA top16. Passing this development gate would not establish independent superiority.','']
    for name,g in s['gates'].items():
        lines.append(f"- {name}: **{'PASS' if g['passed'] else 'FAIL'}**; "+'; '.join(k+': '+('pass' if v else 'fail') for k,v in g['checks'].items()))
    lines += ['', '## Locked validation selections','',
        '| Mouse | Transformer tau | Transformer alpha | MLP tau | MLP alpha | PCA tau |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for mouse,c in choices.items():
        lines.append(f"| {mouse} | {c['transformer']['tau']} | {c['transformer']['alpha']} | {c['mlp']['tau']} | {c['mlp']['alpha']} | {c['pca']['tau']} |")
    lines += ['', '## Per-mouse MSE','',
        '| Mouse | T parent | T local16 | T hybrid | MLP parent | MLP local16 | MLP hybrid | PCA local16 | Zero |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for mouse in run.MICE:
        values = []
        for arm in ('transformer_parent','transformer_local16','transformer_hybrid','mlp_parent','mlp_local16','mlp_hybrid','pca_local16','zero'):
            errors = [r['mse'] for r in s['rows'] if r['mouse']==mouse and r['arm']==arm]
            values.append(f'{np.mean(errors):.6f}')
        lines.append('| '+mouse+' | '+' | '.join(values)+' |')
    a = run.read(ROOT/'audit.json')
    lines += ['', '## Checks and limitations','',
        f"Independent checks passed for{a['metrics_checked']}metric records, {a['independent_float64_top16_predictions']} "
        f"float64 reference predictions, all{a['contrasts_checked']}contrast aggregates, validation selection argmins, "
        'blend arithmetic and source/cache locks. Original heads reproduce exactly: maximum difference '
        f"{s['max_parent_prediction_difference']:.3g}. Detailed records and predictions are saved locally.",'',
        'This is an adaptive research follow-up, not new confirmation. Local retrieval may reduce diffuse averaging '
        'while increasing variance or harming active-state accuracy. The seed runs do not add biological replicates. '
        'A documented numerical correction clips blend components in float64 so alpha0 exactly preserves parent metrics. '
        'All validation-selected weights were verified unchanged; original code, protocol, predictions and summary are retained '
        'alongside `precision_resolution.json`. No learned routing, end-to-end retrieval or new architecture was tested. A failed consistency or harm criterion '
        'cannot be rescued by average improvement. Grid-boundary optima do not trigger extensions.','',
        'Local commands: `python3 experiments/2026-10-06_frozen_retrieval/local16.py selftest`, then '
        '`select`, then `evaluate`; `python3 experiments/2026-10-06_frozen_retrieval/local16_report.py` '
        'checks results and renders this report. Completed output directories are protected against overwrite.','']
    (ROOT/'REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':
    torch.set_num_threads(4)
    audit()
    report()
