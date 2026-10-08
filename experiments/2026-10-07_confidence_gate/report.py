"""Independent prediction, selection and metric checks for inactivity gating."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def reference(z,scale,threshold):
    logits = z['logits'].astype(float)
    assert np.isfinite(logits).all()
    p = .5*(1+np.tanh(logits/2))
    correction = np.where((p<=threshold)&(z['delta']<0),z['delta'].astype(float)*scale,0.)
    return np.clip(z['base']+correction,0,None)


def main():
    p = read(ROOT/'protocol.json'); lock = read(ROOT/'selection_lock.json'); s = read(ROOT/'summary.json')
    assert lock['protocol_sha256']==sha(ROOT/'protocol.json')
    for path,h in p['sha256'].items(): assert sha(REPO/path)==h
    for path,h in lock['validation_sha256'].items(): assert sha(ROOT/'validation'/path)==h
    selections=metrics=0
    for rid,choices in lock['choices'].items():
        for arm,c in choices.items():
            options = p['gated_options'] if arm=='gated' else p['downward_control_options']
            grids = []
            for scale,threshold in options:
                losses=[]; active_losses=[]; base_active=[]
                for seed in p['seeds']:
                    with np.load(ROOT/'validation'/f'{rid}_{seed}.npz') as z:
                        v=reference(z,scale,threshold); y=z['target']; active=y>=.5
                        losses.append(float(np.mean((v-y)**2)))
                        active_losses.append(float(np.mean((v[active]-y[active])**2)) if active.any() else None)
                        base_active.append(float(np.mean((z['base'][active]-y[active])**2)) if active.any() else None)
                feasible = scale==0 or (all(v is not None for v in active_losses+base_active) and np.mean(active_losses)<=1.01*np.mean(base_active) and all(a<=1.05*b for a,b in zip(active_losses,base_active)))
                grids.append((bool(feasible),float(np.mean(losses))))
            for computed,saved in zip(grids,c['grid']):
                assert computed[0]==saved['feasible']
                np.testing.assert_allclose(computed[1],saved['mean_mse'],atol=1e-12)
            best=min(loss for ok,loss in grids if ok)
            index=next(i for i,(ok,loss) in enumerate(grids) if ok and loss<=best+1e-12*max(1,abs(best)))
            assert tuple(options[index])==(c['scale'],c['threshold'])
            selections+=1
        for seed in p['seeds']:
            dev=rid in p['cohorts']['development']
            source=ROOT.parent/('2026-10-07_bounded_hybrid' if dev else '2026-10-07_holdout_confirmation')/'predictions'
            path=source/(f'{rid}_attention_bce_{seed+200}.npz' if dev else f'{rid}_{seed}.npz')
            with np.load(path) as original,np.load(ROOT/'predictions'/f'{rid}_{seed}.npz') as saved:
                for arm,c in choices.items():
                    np.testing.assert_allclose(saved[arm],reference(original,c['scale'],c['threshold']),atol=1e-12)
                for arm in ('blend','original','gated','downward'):
                    prediction,y=saved[arm],saved['target']; e=prediction-y
                    row=next(r for r in s['rows'] if r['recording']==rid and r['seed']==seed and r['arm']==arm)
                    np.testing.assert_allclose(row['mse'],np.square(e).mean(),atol=1e-12)
                    np.testing.assert_allclose(row['mae'],np.abs(e).mean(),atol=1e-12)
                    quiet,active=y<=.05,y>=.5
                    if quiet.any(): np.testing.assert_allclose(row['qfm'],prediction[quiet].mean(),atol=1e-12)
                    if active.any(): np.testing.assert_allclose(row['active_mse'],np.square(e[active]).mean(),atol=1e-12)
                    metrics+=1
    for cohort,comparisons in s['comparisons'].items():
        for name,c in comparisons.items():
            a,b=name.split('_vs_'); gains=[]
            for rid in p['cohorts'][cohort]:
                aa=[r['mse'] for r in s['rows'] if r['recording']==rid and r['arm']==a]
                bb=[r['mse'] for r in s['rows'] if r['recording']==rid and r['arm']==b]
                gains.append(1-np.mean(aa)/np.mean(bb))
            np.testing.assert_allclose(c['mean_gain'],np.mean(gains),atol=1e-12)
            assert c['wins']==sum(v>0 for v in gains)
        c=comparisons['gated_vs_blend']; g=s['gates'][cohort]
        checks=dict(quiet_gain=c['mean_quiet_gain'] is not None and c['mean_quiet_gain']>=.05,
            quiet_wins=c['quiet_wins']>=math.ceil(.75*len(p['cohorts'][cohort])),total_mean=c['mean_gain']>=0,
            total_harm=c['max_harm']<=.01,active_protection=c['max_active_harm'] is not None and c['max_active_harm']<=.01)
        assert checks==g['checks']
        assert g['passed']==(all(checks.values()) if cohort!='later_session' else None)
    audit=dict(passed=True,validation_selections=selections,test_records=metrics,
        independent_tanh_probability_reference=True,prediction_and_metric_checks=True,source_and_input_hashes_unchanged=True,
        cohorts_separate=True,refinement_gate_decisions_checked=True)
    (ROOT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    lines=['# Confidence-gated downward correction','',
        'Exploratory follow-up using existing attention+BCE correction checkpoints. No retraining. '
        'The four previously held-out mice have now been examined and tuned on; their results below are development evidence, '
        'not a new confirmation. Original holdout results and locked artifacts are unchanged.','',
        'The new rule applies a scaled negative correction only when the movement score is below a validation-selected '
        'threshold. It can never increase the blend prediction. A matched downward-only control tests whether confidence '
        'gating adds anything beyond restricting the correction direction. Both choices use validation total MSE, subject '
        'to≤1%mean and≤5%per-seed validation active-MSE harm; disabling correction is allowed. Scores are not assumed calibrated.','',
        '## Results','',
        'Positive MSE gain is better. Quiet gain means reduced mean predicted speed on observed quiet frames; '
        'it is not a classification error rate. Seed metrics are averaged within mouse before equal-mouse relative gains.','',
        '| Cohort | Comparison | MSE gain | Wins | Quiet gain | Quiet wins | Worst active harm |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    fmt=lambda x: 'n/a' if x is None else f'{100*x:+.2f}%'
    for cohort,cs in s['comparisons'].items():
        n=len(p['cohorts'][cohort])
        for name,c in cs.items():
            lines.append(f"| {cohort} | {name.replace('_vs_',' vs ')} | {fmt(c['mean_gain'])} | {c['wins']}/{n} | {fmt(c['mean_quiet_gain'])} | {c['quiet_wins']}/{n} | {fmt(c['max_active_harm'])} |")
    lines+=['','## Prospective refinement checks','',
        'The new targeted criterion requires≥5%quiet prediction reduction, quiet wins on≥75%of mice, '
        'nonnegative mean total-MSE gain, no mouse total-MSE harm>1%, and no active-MSE harm>1%. '
        'This deliberately evaluates a quiet-period refinement; it does not replace the historical5%overall-gain criterion.','']
    for cohort,g in s['gates'].items():
        verdict='descriptive only' if g['passed'] is None else 'PASS' if g['passed'] else 'FAIL'
        lines.append(f"- {cohort}: **{verdict}**; "+'; '.join(k+': '+('pass' if v else 'fail') for k,v in g['checks'].items())+f". Original5%overall-gain threshold: {'met' if g['original_gain_threshold_met'] else 'not met'}.")
    lines+=['','## Locked choices','',
        '| Recording | Gated scale | Movement-score threshold | Downward-only scale |',
        '| --- | ---: | ---: | ---: |']
    for rid,c in lock['choices'].items(): lines.append(f"| {rid} | {c['gated']['scale']} | {c['gated']['threshold']} | {c['downward']['scale']} |")
    lines+=['','## Verification and limits','',
        f"Independent reference arithmetic verified all{selections}validation selections and{metrics}test metric records, "
        'along with all cohort contrast means and refinement decisions. Original sources, inputs and selected-checkpoint '
        'hashes were unchanged. Selected validation predictions reproduced archived selected epochs exactly.','',
        'Validation has already selected the parent checkpoints and is reused for gate selection; apparent validation '
        'protection need not transfer. Hard score thresholds may miss or fragment movement. Downward-only correction '
        'guarantees no increase in quiet predicted speed, not better overall decoding. The control is important for '
        'distinguishing threshold benefit from simply reducing correction strength. No further grid extension is queued.','',
        'Reproduction requires local checkpoints and arrays: `python3 experiments/2026-10-07_confidence_gate/run.py` '
        'with `smoke`, `freeze`, `select`, `evaluate`, then `python3 experiments/2026-10-07_confidence_gate/report.py`. '
        'Do not rerun completed stages; locks and outputs are protected.','']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)); print('PASS',audit)


if __name__=='__main__': main()
