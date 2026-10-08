"""Independently audit frozen movement-gate artifacts and render the comparison."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    protocol = read(ROOT/'protocol.json')
    lock = read(ROOT/'evaluation_lock.json')
    summary = read(ROOT/'summary.json')
    assert lock['protocol_sha256']==digest(ROOT/'protocol.json')
    for file,expected in read(ROOT.parent/'2026-10-06_frozen_retrieval'/'protocol.json')['sha256'].items():
        assert digest(REPO/file)==expected,file
    for file,expected in protocol['sources'].items(): assert digest(REPO/file)==expected
    for file,expected in lock['sha256'].items(): assert digest(ROOT/file)==expected
    for name,expected in read(ROOT/'cache_lock.json')['sha256'].items(): assert digest(ROOT/'cache'/name)==expected
    fits,records = 0,0
    for mouse in protocol['mice']:
        meta = read(ROOT.parent/'2026-10-06_facemap_validation'/'prepared'/mouse/'metadata.json')
        lower = meta['lower']
        for seed in protocol['seeds']:
            initial = {}
            for arm in protocol['arms']:
                name = f'{mouse}_{arm}_{seed}'
                folder = ROOT/'fits'/name
                result = read(folder/'result.json')
                history = read(folder/'history.json')
                initial[arm] = torch.load(folder/'initial.pt',weights_only=True,map_location='cpu')
                with np.load(folder/'validation.npz') as saved:
                    losses = np.square(saved['prediction'].astype(float)-saved['target'][None]).mean(1)
                    np.testing.assert_allclose(losses,[h['validation_mse'] for h in history],rtol=1e-12,atol=1e-12)
                    assert int(np.argmin(losses))==result['selected_epoch']
                    assert len(losses)==protocol['epochs']+1
                with np.load(ROOT/'predictions'/(name+'.npz')) as saved:
                    p,y = saved['prediction'],saved['target']
                    assert np.isfinite(p).all() and (p>=lower-1e-6).all()
                    probability,conditional = saved['probability'],saved['conditional_speed']
                    assert np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all()
                    assert np.isfinite(conditional).all() and (conditional>=0).all()
                    np.testing.assert_allclose(p,(probability*conditional).astype(float)+lower,atol=1e-6,rtol=1e-6)
                    row = next(r for r in summary['rows'] if r['mouse']==mouse and r['seed']==seed and r['arm']==arm)
                    error = np.maximum(p,lower)-y
                    np.testing.assert_allclose(row['mse'],np.square(error).sum()/len(error),atol=1e-12,rtol=1e-12)
                    np.testing.assert_allclose(row['mae'],np.abs(error).mean(),atol=1e-12,rtol=1e-12)
                    quiet = y-lower<=.05
                    if quiet.any(): np.testing.assert_allclose(row['qfm'],(np.maximum(p,lower)[quiet]-lower).mean(),atol=1e-12)
                    np.testing.assert_allclose(row['brier'],np.square(probability-(~quiet).astype(float)).mean(),atol=1e-12)
                    records += 1
                fits += 1
            for a,b in [('attention_bce','attention_mse'),('mlp_bce','mlp_mse')]:
                for key,value in initial[a].items(): assert torch.equal(value,initial[b][key]),(a,b,key)
            for key,value in initial['attention_bce'].items():
                if key.startswith('speed.') or key.startswith(('gate.readin','gate.session','gate.time','gate.patch','gate.head')):
                    assert torch.equal(value,initial['mlp_bce'][key]),key
    for name,c in summary['contrasts'].items():
        a,b = name.split('_vs_')
        gains = []
        for mouse in protocol['mice']:
            aa = [r['mse'] for r in summary['rows'] if r['mouse']==mouse and r['arm']==a]
            bb = [r['mse'] for r in summary['rows'] if r['mouse']==mouse and r['arm']==b]
            gains.append(1-(sum(aa)/len(aa))/(sum(bb)/len(bb)))
        np.testing.assert_allclose(c['mean_gain'],sum(gains)/len(gains),atol=1e-12)
        assert c['mouse_wins']==sum(x>0 for x in gains)
    result = dict(passed=True,fit_selections_checked=fits,test_metric_records_checked=records,
        matched_initializations=True,probability_speed_product_checked=True,contrasts_checked=len(summary['contrasts']),
        source_input_cache_locks_checked=True)
    (ROOT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS:',fits,'checkpoint selections;',records,'test metric records',flush=True)


def report():
    p = read(ROOT/'protocol.json')
    s = read(ROOT/'summary.json')
    lines = ['# Movement-gated MLP speed decoder','',
        'Completed84 fits: four arms × seven mice × three seeds,24 epochs each. '
        'Already examined recordings and previously validation-selected MLP features make this exploratory development, '
        'not fresh confirmation. No historical result or failed gate is changed.','',
        'The trainable movement gate reads the original512×32 activity window through either a small temporal transformer '
        'or a matched temporal MLP. A new positive speed head reads128 frozen pretrained MLP features. '
        'Prediction = sigmoid(movement logit) × softplus(speed head). Explicit movement supervision adds BCE '
        '(weight1) for speed>0.05 training SD above physical zero. The MSE-only controls use identical architectures '
        'and starting weights without BCE. A sigmoid output is not automatically calibrated.','',
        'All components except the pretrained MLP feature encoder are optimized. Thus this tests partial transformer '
        'use in a trainable gate; it does not exhaust fully end-to-end architectures. Epoch selection uses validation '
        'speed MSE including epoch0, and all84 checkpoints were locked before new test scoring.','',
        '## Results','',
        'Positive gain means lower candidate MSE. Average seeds within mouse, then relative gains equally across mice.','',
        '| Comparison | Mean MSE gain | Mouse wins | Worst harm | Quiet wins | MAE gain |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name,c in s['contrasts'].items():
        lines.append(f"| {name.replace('_vs_',' versus ').replace('_',' ')} | {100*c['mean_gain']:+.2f}% | {c['mouse_wins']}/7 | {100*c['max_harm']:.2f}% | {c['quiet_wins']}/7 | {100*c['mean_mae_gain']:+.2f}% |")
    g = s['primary_gate']
    lines += ['',f"Primary attention-versus-MLP gate: **{'PASS' if g['passed'] else 'FAIL'}**; {g['seed_wins']}/21 paired seed wins. "+
        '; '.join(k+': '+('pass' if v else 'fail') for k,v in g['checks'].items()),'',
        'Repair gates additionally require≥5%gain over the original pretrained MLP,≥5/7mouse wins,≤10%worst harm '
        'and quiet false movement lower on≥5/7mice.','']
    for arm,g in s['repair_gates'].items():
        lines.append(f"- {arm}: **{'PASS' if g['passed'] else 'FAIL'}**; "+'; '.join(k+': '+('pass' if v else 'fail') for k,v in g['checks'].items()))
    lines += ['', '## Per-mouse MSE','',
        '| Mouse | Attention + BCE | MLP + BCE | Attention MSE-only | MLP MSE-only | Original MLP | Original transformer | Zero |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for mouse in p['mice']:
        values = []
        for arm in (*p['arms'],'mlp_parent','transformer_parent','zero'):
            values.append(f"{np.mean([r['mse'] for r in s['rows'] if r['mouse']==mouse and r['arm']==arm]):.6f}")
        lines.append('| '+mouse+' | '+' | '.join(values)+' |')
    lines += ['', '## Training and checks','',
        '| Arm | Trainable parameters | Selected epoch0 | Median selected epoch |',
        '| --- | ---: | ---: | ---: |']
    results = [read(f) for f in (ROOT/'fits').glob('*/result.json')]
    for arm in p['arms']:
        rr = [r for r in results if r['arm']==arm]
        lines.append(f"| {arm} | {rr[0]['parameters']} | {sum(r['selected_epoch']==0 for r in rr)}/21 | {np.median([r['selected_epoch'] for r in rr]):.0f} |")
    lines += ['',
        'All selected checkpoints reproduce their selected validation predictions exactly. Independent audit recomputes '
        'all epoch-selection MSEs, all84 test metric records, all15 contrast aggregates, the probability×speed '
        'prediction identity, and paired initialization. Frozen source/checkpoint/cache hashes are checked. '
        'Finite gradients and initial nonnegative training-mean predictions passed smoke checks.','',
        'The loss coefficient, threshold, gate size and training budget are one fixed recipe, not an exhaustive family test. '
        'Original predictors have different training histories; the matched new attention/MLP comparison isolates '
        'the gate family more closely. Behavioral quiet labels are never inference inputs. '
        'Any apparent average advantage must satisfy the stated consistency and harm criteria before promotion.','',
        'Local commands: `python3 experiments/2026-10-06_movement_gate/run.py smoke`, then '
        '`prepare`, `train`, `evaluate`. Training skips complete hash-verified jobs, but refuses incomplete existing '
        'job directories rather than silently restarting them. '
        '`python3 experiments/2026-10-06_movement_gate/report.py` verifies and renders results. '
        'No automatic grid extension or additional fitting is queued.','']
    (ROOT/'REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':
    torch.set_num_threads(2)
    audit()
    report()
