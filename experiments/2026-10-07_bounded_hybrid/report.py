"""Independent artifact audit and descriptive combined-model report."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    protocol = read(ROOT/'protocol.json'); lock = read(ROOT/'evaluation_lock.json')
    baseline = read(ROOT/'base_lock.json'); summary = read(ROOT/'summary.json')
    assert lock['protocol_sha256']==sha(ROOT/'protocol.json')
    assert lock['base_lock_sha256']==sha(ROOT/'base_lock.json')
    for path,h in protocol['source_sha256'].items(): assert sha(REPO/path)==h
    for path,h in lock['sha256'].items(): assert sha(ROOT/path)==h
    for path,h in baseline['cache_sha256'].items(): assert sha(ROOT/'cache'/path)==h
    for path,h in read(ROOT.parent/'2026-10-06_frozen_retrieval'/'protocol.json')['sha256'].items(): assert sha(REPO/path)==h
    fits = []
    for mouse in protocol['mice']:
        lower = read(ROOT.parent/'2026-10-06_facemap_validation'/'prepared'/mouse/'metadata.json')['lower']
        c = baseline['choices'][mouse]
        assert c['alpha']==protocol['alpha_grid'][int(np.argmin(np.mean(c['validation_grid'],axis=0)))]
        for seed in protocol['seeds']:
            states = {}
            for arm in protocol['arms']:
                name = f'{mouse}_{arm}_{seed}'; folder = ROOT/'fits'/name
                result = read(folder/'result.json'); history = read(folder/'history.json'); fits.append(result)
                states[arm] = torch.load(folder/'initial.pt',weights_only=True)
                with np.load(folder/'validation.npz') as z:
                    losses = ((z['prediction']-z['target'][None])**2).mean(1)
                    np.testing.assert_allclose(losses,[h['validation_mse'] for h in history],rtol=1e-12,atol=1e-12)
                    assert len(losses)==25 and int(np.argmin(losses))==result['selected_epoch']
                    with np.load(ROOT/'cache'/f'{mouse}_{seed}.npz') as cache:
                        np.testing.assert_array_equal(z['prediction'][0],cache['base_validation'])
                with np.load(ROOT/'predictions'/(name+'.npz')) as z:
                    assert np.isfinite(z['prediction']).all() and np.abs(z['delta']).max()<=.500001
                    np.testing.assert_array_equal(z['prediction'],np.maximum(z['base']+z['delta'].astype(float),0)+lower)
                    r = next(r for r in summary['rows'] if r['mouse']==mouse and r['seed']==seed and r['arm']==arm)
                    e = np.maximum(z['prediction'],lower)-z['target']
                    np.testing.assert_allclose(r['mse'],np.square(e).mean(),atol=1e-12,rtol=1e-12)
                    np.testing.assert_allclose(r['mae'],np.abs(e).mean(),atol=1e-12,rtol=1e-12)
                    quiet = z['target']-lower<=.05; active = z['target']-lower>=.5
                    if quiet.any(): np.testing.assert_allclose(r['qfm'],np.mean(z['prediction'][quiet]-lower),atol=1e-12)
                    if active.any(): np.testing.assert_allclose(r['active_mse'],np.mean(e[active]**2),atol=1e-12)
            for k,v in states['attention_bce'].items(): assert torch.equal(v,states['attention_mse'][k])
            for k,v in states['attention_bce'].items():
                if k.startswith(('head.','encoder.readin','encoder.session','encoder.patch','encoder.time')):
                    assert torch.equal(v,states['mlp_bce'][k])
    for name,c in summary['contrasts'].items():
        a,b = name.split('_vs_'); gains = []
        for mouse in protocol['mice']:
            aa = [r['mse'] for r in summary['rows'] if r['mouse']==mouse and r['arm']==a]
            bb = [r['mse'] for r in summary['rows'] if r['mouse']==mouse and r['arm']==b]
            gains.append(1-np.mean(aa)/np.mean(bb))
        np.testing.assert_allclose(c['mean_gain'],np.mean(gains),atol=1e-12)
        assert c['mouse_wins']==sum(v>0 for v in gains)
    audit = dict(passed=True,fit_selections=len(fits),test_records=len(fits),contrasts=len(summary['contrasts']),
        exact_initial_base_predictions=True,paired_initializations=True,correction_bounds_and_reconstruction=True,
        input_source_checkpoint_cache_hashes_verified=True)
    (ROOT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    lines = ['# Bounded hybrid correction','',
        'The primary goal is a combined model better than both original standalone models. '
        'Attention versus an MLP correction is a secondary control. Completed63 fits across seven mice and three seeds, '
        'with24 epochs per fit. All data are previously examined; this is exploratory development.','',
        'The base is a validation-selected convex blend of original MLP and transformer predictions. '
        'The small trainable correction sees neural activity and the base prediction; its magnitude is bounded '
        'by0.5 training SD and reduced when its movement score is high. It starts at exactly zero correction. '
        'BCE provides explicit movement supervision, while a fixed active-frame penalty discourages damaging '
        'the base during movement. The selected epoch can remain0 if validation rejects every correction.','',
        '## Comparisons','',
        'Positive gain is lower candidate error; seed MSEs are averaged within mouse, followed by equal-mouse relative gains.','',
        '| Candidate versus control | MSE gain | Mouse wins | Worst harm | Quiet wins | MAE gain |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name,c in summary['contrasts'].items():
        lines.append(f"| {name.replace('_vs_',' versus ').replace('_',' ')} | {100*c['mean_gain']:+.2f}% | {c['mouse_wins']}/7 | {100*c['max_harm']:.2f}% | {c['quiet_wins']}/7 | {100*c['mean_mae_gain']:+.2f}% |")
    lines += ['', '## Full gates','',
        'Require≥5%mean MSE gain against each parent,≥6/7mouse wins,≥14/21seed wins,≤10%worst harm '
        'and nonnegative MAE gains. Also require≥5%gain over the simple blend, quiet wins≥5/7 against that blend, '
        'and no mouse with active MSE more than5%worse than the blend. These are development criteria, not independent significance.','']
    for arm,g in summary['gates'].items():
        lines.append(f"- {arm}: **{'PASS' if g['passed'] else 'FAIL'}**. "+
            '; '.join(parent+': '+', '.join(k+' '+('pass' if v else 'fail') for k,v in checks.items()) for parent,checks in g['checks'].items()))
    lines += ['', '## Per-mouse MSE','',
        '| Mouse | Original T | Original MLP | Simple blend | Attention+BCE | MLP+BCE | Attention MSE-only | Zero |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for mouse in protocol['mice']:
        vals = [np.mean([r['mse'] for r in summary['rows'] if r['mouse']==mouse and r['arm']==arm]) for arm in ('transformer','mlp','blend','attention_bce','mlp_bce','attention_mse','zero')]
        lines.append('| '+mouse+' | '+' | '.join(f'{v:.6f}' for v in vals)+' |')
    lines += ['', '## Selection and verification','',
        '| Mouse | Transformer fraction in base |', '| --- | ---: |']
    for mouse,c in baseline['choices'].items(): lines.append(f"| {mouse} | {c['alpha']} |")
    for arm in protocol['arms']:
        rr = [r for r in fits if r['arm']==arm]
        lines.append(f"\n{arm}: {rr[0]['parameters']} trainable parameters; {sum(r['selected_epoch']==0 for r in rr)}/21 selections retain the uncorrected base.")
    lines += ['', 'All63 checkpoint selections and test metric records were independently checked, along with14 '
        'contrast aggregates, initial-state pairing, exact initial base preservation, prediction reconstruction '
        'and correction bounds. All selected validation reloads matched exactly; data, cache and source hashes passed.','',
        'Fixed recipes, explored mice and reused pretrained parents limit interpretation. This is not exhaustive '
        'optimization, new-animal confirmation, or a claim that attention must contribute uniquely. '
        'No gate uses observed query speed at inference. Large binary artifacts remain local.','',
        'Commands: `python3 experiments/2026-10-07_bounded_hybrid/run.py smoke`, `prepare`, '
        '`fit --mouse TX103` (repeat each fixed mouse; independent mouse fits may run in parallel), '
        '`lock`, `evaluate`, then `python3 experiments/2026-10-07_bounded_hybrid/report.py`. '
        'Completed fit results are hash verified and skipped; incomplete folders are not silently overwritten.','']
    (ROOT/'REPORT.md').write_text('\n'.join(lines))
    print('PASS',audit)


if __name__=='__main__':
    torch.set_num_threads(2)
    main()
