import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
CONDITIONS = ['none', 'real', 'shuffle1', 'shuffle2']
LAMBDAS = [.001, .01, .1, 1., 10.]
LIMITER = threadpool_limits(limits=2)
torch.set_num_threads(2)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def solve(x, y, regularization, prior=None):
    #fit a per-neuron ID coefficient vector around the shared prior
    x, y = torch.as_tensor(x, dtype=torch.float64), torch.as_tensor(y, dtype=torch.float64)
    if prior is None:
        prior = torch.zeros((x.shape[1], y.shape[1]), dtype=torch.float64)
    else:
        prior = torch.as_tensor(prior, dtype=torch.float64)
    penalty = torch.eye(x.shape[1], dtype=torch.float64)*regularization
    penalty[0, 0] = 0
    a = x.T@x/len(x)+penalty
    b = x.T@(y-x@prior)/len(x)
    residual = torch.linalg.solve(a, b)
    assert torch.isfinite(residual).all()
    assert torch.max(torch.abs(a@residual-b)).item() < 1e-8
    return (prior+residual).numpy()


def weights(query, donor):
    d2 = np.square(query[:, None, :]-donor[None, :, :]).sum(axis=2)
    near = np.argsort(d2, axis=1, kind='stable')[:, :32]
    local_d2 = np.take_along_axis(d2, near, axis=1)
    bandwidth = np.maximum(local_d2[:, -1:], 1e-8)
    local = np.exp(-.5*local_d2/bandwidth)
    local /= local.sum(axis=1, keepdims=True)
    w = np.zeros_like(d2)
    np.put_along_axis(w, near, local, axis=1)
    np.testing.assert_allclose(w.sum(axis=1), 1)
    return w


def evaluate(prediction, truth):
    assert prediction.shape == truth.shape and np.isfinite(prediction).all()
    error = np.mean((prediction-truth)**2, axis=0)
    variance = np.var(truth, axis=0)
    r2 = 1-error/np.maximum(variance, 1e-12)
    pc, tc = prediction-prediction.mean(0), truth-truth.mean(0)
    correlation = np.sum(pc*tc, axis=0)/np.maximum(np.sqrt(np.sum(pc**2, axis=0)*np.sum(tc**2, axis=0)), 1e-12)
    pv = np.var(prediction, axis=0)
    lag_true = np.sum(tc[1:]*tc[:-1], axis=0)/np.maximum(np.sqrt(np.sum(tc[1:]**2, axis=0)*np.sum(tc[:-1]**2, axis=0)), 1e-12)
    lag_pred = np.sum(pc[1:]*pc[:-1], axis=0)/np.maximum(np.sqrt(np.sum(pc[1:]**2, axis=0)*np.sum(pc[:-1]**2, axis=0)), 1e-12)
    cov_true, cov_pred = tc.T@tc/len(tc), pc.T@pc/len(pc)
    return dict(mean_mse=float(error.mean()), mean_r2=float(r2.mean()), median_r2=float(np.median(r2)),
                positive_r2_cells=int((r2>0).sum()), mean_correlation=float(correlation.mean()),
                median_variance_ratio=float(np.median(pv/np.maximum(variance, 1e-12))),
                mean_lag1_error=float(np.mean(np.abs(lag_true-lag_pred))),
                relative_covariance_error=float(np.linalg.norm(cov_pred-cov_true)/np.linalg.norm(cov_true)),
                per_cell_mse=error.tolist(), per_cell_r2=r2.tolist())


def design(series):
    windows = np.lib.stride_tricks.sliding_window_view(series, 8, axis=0)
    return windows[24:].reshape(len(series)-31, -1)


def prepare():
    assert not (ROOT/'protocol.json').exists()
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        question='Can a simple shared spatial prior improve few-shot reconstruction of new neurons beyond activity, running speed, and per-neuron ID coefficients?',
        scope='deterministic conditional-mean reconstruction; not a stochastic generator or a causal locomotion model',
        seed=20261003, reference_cells=512, donor_cells=1024, tuning_cells=128, evaluation_cells=128,
        disjoint_roles=True, eligible='nonconstant activity in calibration [0,416)',
        shared_training=[0,4160], calibration=[0,416], evaluation=[4260,5564],
        examples=dict(shared=4129, calibration=385, evaluation=1273),
        test='never construct or evaluate time bins >=5564', window=8, target_offset=31,
        features='reference PCA16 x 8-bin history plus speed x 8-bin history, training-standardized; intercept',
        shared_learning='donor-specific ridge coefficients from shared training, mean-loss penalty 0.01',
        spatial_prior='weighted average coefficients of 32 closest donor coordinates; Gaussian weights with per-query 32nd distance bandwidth',
        none_prior='mean donor coefficients; preserves donor transfer without geometry',
        coordinate_conditions=CONDITIONS,
        shuffles='fixed independent permutations of donor/tuning/evaluation coordinate assignments together; seed10101/20202; activity and ID assignment fixed',
        adaptation='per-target ID-specific coefficient residual fitted on calibration labels; intercept unpenalized',
        adaptation_lambdas=LAMBDAS,
        selection='each condition independently chooses lambda by mean standardized MSE on 128 tuning cells at validation times; 128 evaluation cells never select a model',
        baselines=['calibration mean', 'speed-history-only ridge with target-specific coefficients',
                   'reference activity plus speed ridge with target-specific coefficients and zero shared prior'],
        normalization='reference/donor activity uses shared training; tuning/evaluation activity uses their calibration only; no later target activity used in preprocessing',
        primary='relative improvement in mean per-cell calibration-standardized evaluation MSE',
        promising_gate='real at least 2% better than none; real better than each shuffle and independent activity+speed ridge; real mean R2 positive; >=80/128 cells beat none',
        secondary='per-cell correlation, predicted/actual variance, lag1 autocorrelation error, population covariance error; conditional-mean predictions are not full generative samples',
        uncertainty='2000 paired circular100-time-block resamples, conditional on this one population; cells not treated as independent animals',
        budget='one fixed population; 20 adaptation fits plus five speed-only and five independent ridge fits; no gradient training or adaptive expansion',
        limitation='different target cells but same previously examined recording and validation time interval; tuning and evaluation cells share population activity; no independent confirmation',
        application_hashes={f:sha(PROJECT/f) for f in ['model.py','data.py','train.py']},
        implementation_sha256=sha(Path(__file__)), dataset_sha256=sha(PROJECT/'data/stringer_spontaneous.npy'))
    write(ROOT/'protocol.json', protocol)
    print('PROTOCOL_FROZEN', flush=True)


def main():
    started = time.monotonic()
    protocol = json.loads((ROOT/'protocol.json').read_text())
    assert sha(Path(__file__)) == protocol['implementation_sha256']
    recording = np.load(PROJECT/'data/stringer_spontaneous.npy', allow_pickle=True).item()
    raw = recording['sresp'][:, :5564].copy()
    speed = recording['run'][:5564, 0].copy()
    xyz = recording['xyz'].T.copy()
    del recording
    eligible = np.flatnonzero(raw[:, :416].std(axis=1)>1e-6)
    selected = np.random.default_rng(20261003).choice(eligible, 1792, replace=False)
    roles = dict(reference=selected[:512], donor=selected[512:1536], tuning=selected[1536:1664], evaluation=selected[1664:])
    assert len(set(selected)) == 1792
    write(ROOT/'cell_roles.json', {k:v.tolist() for k,v in roles.items()})
    ref = raw[roles['reference']].T
    ref_mean, ref_std = ref[:4160].mean(0), np.maximum(ref[:4160].std(0), 1e-6)
    ref = (ref-ref_mean)/ref_std
    pca = PCA(n_components=16, svd_solver='randomized', random_state=20261003)
    pca.fit(ref[:4160])
    pc = pca.transform(ref)
    speed_mean, speed_std = speed[:4160].mean(), max(speed[:4160].std(), 1e-6)
    signals = np.column_stack([pc, (speed-speed_mean)/speed_std])
    ftrain, fval = design(signals[:4160]), design(signals[4260:5564])
    fm, fs = ftrain.mean(0), np.maximum(ftrain.std(0), 1e-6)
    x = np.column_stack([np.ones(len(ftrain)), (ftrain-fm)/fs])
    xv = np.column_stack([np.ones(len(fval)), (fval-fm)/fs])
    xc = x[:385]
    speed_columns = [0]+list(range(129,137))
    assert x.shape == (4129,137) and xv.shape == (1273,137)
    np.testing.assert_allclose(ftrain[0], signals[24:32].T.reshape(-1), rtol=0, atol=0)
    np.testing.assert_allclose(fval[0], signals[4284:4292].T.reshape(-1), rtol=0, atol=0)
    ys, scaling = {}, {}
    for role in ['donor','tuning','evaluation']:
        a = raw[roles[role]].T
        stop = 4160 if role=='donor' else 416
        mu, sd = a[:stop].mean(0), np.maximum(a[:stop].std(0), 1e-6)
        ys[role] = (a-mu)/sd
        scaling[role] = (mu, sd)
    #evaluation cells provide only calibration labels until model selection is complete
    donor_coefficients = solve(x, ys['donor'][31:4160], .01)
    pos_mu = xyz[eligible].mean(0)
    pos_sd = np.maximum(xyz[eligible].std(0), 1e-6)
    positions = (xyz[selected[512:]]-pos_mu)/pos_sd
    prior = {}
    for condition in CONDITIONS:
        assigned = positions.copy()
        if condition.startswith('shuffle'):
            seed = 10101 if condition=='shuffle1' else 20202
            assigned = assigned[np.random.default_rng(seed).permutation(len(assigned))]
        if condition=='none':
            w = np.full((256,1024), 1/1024)
        else:
            w = weights(assigned[1024:], assigned[:1024])
        prior[condition] = donor_coefficients@w.T
    tuning_results, chosen = {}, {}
    for condition in CONDITIONS+['independent', 'speed_only']:
        xx, vv = (xc[:, speed_columns], xv[:, speed_columns]) if condition=='speed_only' else (xc, xv)
        base = prior[condition][:,:128] if condition in CONDITIONS else None
        records = []
        for lam in LAMBDAS:
            coefficients = solve(xx, ys['tuning'][31:416], lam, base)
            mse = float(np.mean((vv@coefficients-ys['tuning'][4291:5564])**2))
            records.append(dict(regularization=lam, tuning_mse=mse))
        chosen[condition] = min(records, key=lambda r:r['tuning_mse'])['regularization']
        tuning_results[condition] = records
    write(ROOT/'selection.json', dict(chosen=chosen, tuning_results=tuning_results))
    truth = ys['evaluation'][4291:5564]
    predictions, coefficients = {}, {}
    for condition in CONDITIONS+['independent','speed_only']:
        xx, vv = (xc[:, speed_columns], xv[:, speed_columns]) if condition=='speed_only' else (xc, xv)
        base = prior[condition][:,128:] if condition in CONDITIONS else None
        coefficients[condition] = solve(xx, ys['evaluation'][31:416], chosen[condition], base)
        predictions[condition] = vv@coefficients[condition]
    predictions['constant'] = np.broadcast_to(ys['evaluation'][31:416].mean(0), truth.shape).copy()
    metrics = {c:evaluate(pred, truth) for c,pred in predictions.items()}
    mse = {c:m['mean_mse'] for c,m in metrics.items()}
    contrasts = {c:1-mse['real']/mse[c] for c in ['none','shuffle1','shuffle2','independent','speed_only','constant']}
    wins = int((np.array(metrics['real']['per_cell_mse'])<np.array(metrics['none']['per_cell_mse'])).sum())
    promising = (contrasts['none']>=.02 and contrasts['shuffle1']>0 and contrasts['shuffle2']>0
                 and contrasts['independent']>0 and metrics['real']['mean_r2']>0 and wins>=80)
    errors = {c:np.mean((v-truth)**2, axis=1) for c,v in predictions.items()}
    rng = np.random.default_rng(20261003)
    boot = {c:[] for c in contrasts}
    for _ in range(2000):
        indices = ((rng.integers(0,1273,13)[:,None]+np.arange(100))%1273).ravel()[:1273]
        real = errors['real'][indices].mean()
        for c in boot:
            boot[c].append(1-real/errors[c][indices].mean())
    intervals = {c:np.quantile(v,[.025,.975]).tolist() for c,v in boot.items()}
    #zero-calibration output is an exploratory conditional mean, not a sampled neural process
    zero_calibration = xv@prior['real'][:,128:]
    metrics['real_zero_calibration'] = evaluate(zero_calibration, truth)
    predictions['real_zero_calibration'] = zero_calibration
    arrays = dict(truth=truth, raw_truth=raw[roles['evaluation'],4291:5564].T,
                  speed=speed[4291:5564], time_index=np.arange(4291,5564), **predictions)
    np.savez_compressed(ROOT/'predictions.npz', **arrays)
    checkpoint = dict(donor_coefficients=donor_coefficients, reference_ids=roles['reference'],
        reference_mean=ref_mean, reference_std=ref_std, pca_components=pca.components_, pca_mean=pca.mean_,
        speed_mean=speed_mean, speed_std=speed_std, feature_mean=fm, feature_std=fs,
        position_mean=pos_mu, position_std=pos_sd, donor_positions=positions[:1024],
        evaluation_positions=positions[1152:], calibration_mean=scaling['evaluation'][0],
        calibration_std=scaling['evaluation'][1], evaluation_ids=roles['evaluation'],
        validation_design=xv, **{'coef_'+c:v for c,v in coefficients.items()})
    np.savez_compressed(ROOT/'checkpoint.npz', **checkpoint)
    #verify saved inference and an independent least-squares solution for the chosen spatial fit
    with np.load(ROOT/'checkpoint.npz') as ck:
        for c in coefficients:
            xx = ck['validation_design'][:,speed_columns] if c=='speed_only' else ck['validation_design']
            np.testing.assert_allclose(xx@ck['coef_'+c], predictions[c], rtol=1e-12, atol=1e-12)
        z = ck['validation_design']@(ck['donor_coefficients']@weights(ck['evaluation_positions'],ck['donor_positions']).T)
        np.testing.assert_allclose(z, zero_calibration, rtol=1e-12, atol=1e-12)
    penalty = np.eye(137)*np.sqrt(chosen['real']*len(xc))
    penalty[0,0]=0
    base = prior['real'][:,128:]
    aug_x = np.vstack([xc,penalty])
    aug_y = np.vstack([ys['evaluation'][31:416]-xc@base,np.zeros((137,128))])
    independent = base+np.linalg.lstsq(aug_x,aug_y,rcond=None)[0]
    numerical_error = float(np.max(np.abs(xv@independent-predictions['real'])))
    assert numerical_error < 1e-7
    for name, expected in protocol['application_hashes'].items():
        assert sha(PROJECT/name)==expected
    result = dict(promising_gate=bool(promising), decision='promising' if promising else 'not established',
        metrics=metrics, real_relative_improvement=contrasts, conditional_intervals=intervals,
        real_vs_none_cell_wins=wins, cells=128, selected_regularization=chosen,
        seconds=time.monotonic()-started, test_evaluations=0,
        interpretation='reconstruction conditional on observed reference activity and speed; coordinate contribution evaluated by retrained controls; no behavioral or causal simulation claim')
    write(ROOT/'results.json',result)
    write(ROOT/'checks.json',dict(passed=True, disjoint_cell_roles=True, target_calibration_only=True,
        tuning_cells_separate_from_evaluation_cells=True, exact_window_boundaries=True,
        finite_predictions=True, checkpoint_prediction_reload=True, independent_solve_max_error=numerical_error,
        unchanged_application=True, no_test_evaluation=True))
    report(result, protocol)
    print(json.dumps({k:v for k,v in result.items() if k!='metrics'},indent=2),flush=True)


def report(result, protocol):
    lines = ['# Few-shot neuron reconstruction prototype','',
        f"Frozen spatial gate: **{result['decision'].upper()}**. This is a deterministic reconstruction prototype conditioned on measured activity and running speed, not a self-running neural generator.", '',
        '| Model | Mean standardized MSE | Mean cell R² | Cells with positive R² | Mean correlation | Median predicted/actual variance |',
        '|---|---:|---:|---:|---:|---:|']
    for c,m in result['metrics'].items():
        lines.append(f"| {c} | {m['mean_mse']:.5f} | {m['mean_r2']:.5f} | {m['positive_r2_cells']}/128 | {m['mean_correlation']:.4f} | {m['median_variance_ratio']:.4f} |")
    lines += ['', 'Each cell is normalized using its calibration period only. R² compares against variation in that cell during evaluation, so it is not the same as improvement over the calibration-mean baseline. Correlation alone does not establish accurate amplitude or realistic dynamics.', '',
        '## Does geometry help?', '',
        '| Comparator | Correct-coordinate relative MSE reduction | Conditional descriptive 95% interval |',
        '|---|---:|---:|']
    for c,v in result['real_relative_improvement'].items():
        lo,hi=result['conditional_intervals'][c]
        lines.append(f'| {c} | {v:+.2%} | [{lo:+.2%}, {hi:+.2%}] |')
    lines += ['',f"Correct coordinates beat no coordinates in {result['real_vs_none_cell_wins']}/128 cells. The frozen gate requires >=2% mean benefit over none, improvement over both shuffles and independent ridge, positive mean cell R², and at least 80 cell wins. Passing would prioritize replication, not establish independent anatomical evidence.",'',
        'The intervals use 2,000 paired circular resamples of 100-bin evaluation blocks. They condition on this single cell population and recording; correlated cells are not counted as independent animals. No extra seeds or neuron pools were selected after seeing results.', '',
        '## What was built', '',
        '512 reference neurons provide observed context. PCA16 fitted on training times summarizes their activity. Eight-bin histories of those components and running speed form 136 input features plus an intercept. 1,024 different donor cells teach per-cell regression coefficients from the full training period. A spatial prior averages the coefficients of 32 nearby donor cells. The no-coordinate prior averages all donor cells, while two fixed coordinate permutations control the cell-location assignment.', '',
        '128 tuning cells choose the adaptation regularization separately for each condition. Another 128 evaluation cells provide only a short calibration period for fitting their per-neuron ID coefficients. All four cell roles are disjoint. The ID representation here is an explicit per-cell coefficient vector rather than a transformer embedding. Coordinates specify the shared prior; the fitted ID coefficients capture cell-specific corrections.', '',
        'The independent baseline has the same activity and speed features and learns each target’s coefficients without a donor prior. Speed-only uses only the eight speed inputs. Both get the same calibration labels and their own tuning-cell-selected regularization. The constant predicts each target’s calibration mean. All four geometry conditions have the same number of fitted target coefficients.', '',
        'Training is [0,4160), target calibration [0,416), and evaluation [4260,5564). Window eight and target offset 31 give 4,129 donor examples, 385 calibration examples and 1,273 evaluation examples. This is contemporaneous reconstruction: current reference activity and speed are available, while current target activity is withheld. All preprocessing uses the permitted training/calibration data. No test-tail examples were evaluated.', '',
        'Zero-calibration output is a secondary probe using the spatial prior without target coefficient adaptation. It is expressed in target-calibration-standardized units for scoring; therefore it is not a demonstration of zero-observation generation in original activity units. Its output is a conditional mean and contains no modeled residual noise.', '',
        '## Limits and interpretation', '',
        'One recording and one population; the time interval was previously examined for other project tasks. Evaluation cells are held out from shared learning and hyperparameter selection, but they do supply calibration labels. This is few-shot transfer, not strict unseen-cell zero-shot transfer or independent-animal evidence. True coordinates are standardized soma positions, not measured connectivity or receptive fields.', '',
        'A coordinate advantage here could arise from spatial recording artifacts or shared signals. Reconstructing activity associated with running does not demonstrate that artificial neurons cause running. A small predicted variance or inaccurate autocorrelation/covariance would also prevent calling the outputs realistic samples even if mean prediction error improves.', '',
        'This bounded prototype uses closed-form linear regression with a spatial smoothing prior. It does not test every neural generator. No extra model sweep follows automatically from these results.', '',
        '## Verification and artifacts', '',
        'Cell-role disjointness, window indices, saved coefficient reload predictions, finite outputs, and unchanged application hashes passed. An independent augmented least-squares fit reproduced the selected spatial predictions. Full metrics, per-cell scores, selected hyperparameters and time-block intervals are saved. Completed earlier diagnostics were not rerun.', '',
        'Files: `protocol.json`, `cell_roles.json`, `selection.json`, `checkpoint.npz`, `predictions.npz`, `results.json`, `checks.json`, and `reconstruction.png`. Run `python3 -B prototype.py run` only in a fresh experiment copy to refit; completed output is protected.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with np.load(ROOT/'predictions.npz') as a:
        truth=a['truth']
        fig,axes=plt.subplots(4,1,figsize=(12,10))
        axes[0].plot(a['time_index'],a['speed'],color='black')
        axes[0].set(ylabel='Recorded speed',title='Few-shot reconstruction: three cells fixed by split order, not selected by score')
        for j,ax in enumerate(axes[1:]):
            ax.plot(a['time_index'][:250],truth[:250,j],label='Recorded',alpha=.7)
            ax.plot(a['time_index'][:250],a['real'][:250,j],label='Real-coordinate prior')
            ax.plot(a['time_index'][:250],a['independent'][:250,j],label='Independent ridge',alpha=.75)
            ax.set(ylabel=f'Cell {j} activity (z)',xlabel='Recording time bin')
        axes[1].legend(loc='upper right')
        fig.tight_layout()
        fig.savefig(ROOT/'reconstruction.png',dpi=140)
        plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','run'])
    args=parser.parse_args()
    if args.action=='prepare':
        prepare()
    else:
        assert not (ROOT/'results.json').exists(), 'completed experiment; read report instead of rerunning'
        main()
