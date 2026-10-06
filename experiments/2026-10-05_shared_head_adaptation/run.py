"""Train-only regularized residual readouts on frozen shared-model features."""
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
EXP = ROOT.parent
FINETUNING = EXP / '2026-10-05_shared_finetuning'
spec = importlib.util.spec_from_file_location('head_reference', FINETUNING / 'run.py')
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
ref, Model = parent.ref, parent.Model
MICE, FAMILIES, SEEDS = parent.MICE, parent.FAMILIES, parent.SEEDS
native_dir, read, write, digest = parent.native_dir, parent.read, parent.write, parent.digest
REPLICATION = parent.REPLICATION
PENALTIES = [.1, 1., 10., 100.]
CONFIGS = [dict(kind='unchanged', penalty=None)] + [dict(kind=k, penalty=p) for k in ['affine', 'features'] for p in PENALTIES]


def design(features, prediction, kind):
    return prediction[:, None].astype(np.float64) if kind == 'affine' else np.column_stack([features, prediction]).astype(np.float64)


def fit(features, prediction, target, config):
    if config['kind'] == 'unchanged':
        return dict(**config, mean=[], scale=[], beta=[])
    x = design(features, prediction, config['kind'])
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-6, scale, 1.)
    z = np.column_stack([(x - mean) / scale, np.ones(len(x))])
    residual = target.astype(np.float64) - prediction
    matrix = np.einsum('ni,nj->ij', z, z, optimize=False) / len(z) + config['penalty'] * np.eye(z.shape[1])
    rhs = np.einsum('ni,n->i', z, residual, optimize=False) / len(z)
    beta = np.linalg.solve(matrix, rhs)
    np.testing.assert_allclose(np.einsum('ij,j->i', matrix, beta, optimize=False), rhs, rtol=1e-9, atol=1e-10)
    return dict(**config, mean=mean.tolist(), scale=scale.tolist(), beta=beta.tolist())


def predict_head(features, prediction, head):
    if head['kind'] == 'unchanged':
        return prediction.astype(np.float64)
    x = design(features, prediction, head['kind'])
    z = np.column_stack([(x - head['mean']) / head['scale'], np.ones(len(x))])
    return prediction + np.einsum('ij,j->i', z, np.asarray(head['beta']), optimize=False)


def features(net, x, session):
    captured = []
    hook = net.head.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().numpy().copy()))
    try:
        prediction = ref.predict(net, x, session)
    finally:
        hook.remove()
    values = np.concatenate(captured)
    assert values.shape == (len(x), 80) and np.isfinite(values).all()
    return values, prediction


def check():
    rng = np.random.default_rng(91602)
    x = rng.normal(size=(128, 80))
    x[:, 2] = 1
    p = rng.normal(size=128)
    target = p + .3 * x[:, 0] + .1
    for config in CONFIGS:
        h = fit(x, p, target, config)
        output = predict_head(x, p, h)
        assert output.shape == p.shape and np.isfinite(output).all()
        if config['kind'] == 'unchanged':
            np.testing.assert_array_equal(output, p)
        if config['kind'] == 'features' and config['penalty'] == .1:
            assert np.mean((output - target) ** 2) < np.mean((p - target) ** 2)
        zero = fit(x, p, p, config)
        np.testing.assert_allclose(predict_head(x, p, zero), p, rtol=0, atol=1e-14)
    net = Model('attention', 10, range(4))
    net.load_state_dict(torch.load(native_dir('attention', 10) / 'selected.pt', weights_only=True))
    tx = torch.from_numpy(rng.normal(size=(5, 128, 32)).astype(np.float32))
    state = {k: v.clone() for k, v in net.state_dict().items()}
    f, p = features(net, tx, 0)
    np.testing.assert_array_equal(p, ref.predict(net, tx, 0))
    assert all(torch.equal(v, net.state_dict()[k]) for k, v in state.items())
    write(ROOT / 'selfcheck.json', dict(passed=True, zero_correction_identity=True, ridge_equations_checked=True,
        constant_features_finite=True, synthetic_residual_learned=True, exact_80_feature_capture=True, encoder_unchanged=True))


def freeze():
    assert read(ROOT / 'selfcheck.json')['passed']
    paths = [ROOT / n for n in ['run.py', 'analyse.py', 'selfcheck.json']] + [FINETUNING / n for n in ['run.py', 'analyse.py', 'protocol.json']]
    inherited = read(FINETUNING / 'protocol.json')['hashes']
    write(ROOT / 'protocol.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        authorization='User requested continuous search. This fallback is fixed before seeing fine-tuning later results; execute only if that study fails transformer_full.',
        question='Does a strongly regularized recording-specific readout improve the frozen original shared transformer beyond its original head and equally adapted MLP?',
        design='Original shared encoder and nonlinear head remain frozen. Fit a linear residual correction independently within each recording, using either native scalar prediction alone (affine) or80existing head-input features plus native prediction (features). Add correction to native raw normalized prediction, then physical-zero bound. No new neural encoder,coordinates ornewinput.',
        fit='Training-data-only feature means/scales and residual targets. Solve normalized MSE +lambda*sum(beta^2), including intercept penalty. lambda=.1,1,10,100; unchanged zero correction option. No validation labels in regression or scaling. No gradient fits.',
        selection='Seeds10-12:fit all nine configurations per family/seed/mouse; choose one global config per family minimizing original joint earlier score averaged overthree seeds. Ties choose listed order, unchanged first. Then fit ONLY that config on13-15. Lock all coefficients andselection beforelater scoring. No later score oflosingconfigurations.',
        budget='192 nonzero analytic fits for initial seeds (2families*3seeds*4mice*8configs), up to24chosen fits onadditional seeds. Feature extraction from original checkpoints is new work; previously completed fits are not repeated. No data-model retraining.',
        primary='Same practical criteria as fine-tuning:additional13-15 adapted transformer>=5% meanpairMSEgain,>=3/4mice,>=8/12singleseedwins versus BOTH native transformer and equallyadaptedMLP;no mouse>10%harm and nonnegative meanMAEgain versusnative. Require sameall-six with16/24wins. Originalsubset cannotrescuefailure. AdaptedMLP own-family utilitysecondary.',
        controls='Both affine andfeature options availableequally tobothfamilies; report selectedkind. A selectedaffine correction iscalibration,notnovelarchitecture. A featurehead improvement alone doesnotprove whichfeaturesmatter orindividual-neuroninteractions. Extra per-recording parameters andtraining-onlyfitting differfrom olddevelopment-fit unregularizedaffinecalibration.',
        selection_denominators=read(FINETUNING / 'protocol.json')['selection_denominators'],
        aggregation='Same original/additional/allseed subsets,physicalzeroindividualbounding beforealltwo-modelpairaveraging,equal-mouserelativeeffects,MAE/singles/leave-one-out. Fourhistoricallyreusedanimals;noindependentsignificance.',
        stop='Finishfixed search/replication/lock/evaluation/audit/report. Noextraalphas/features/methodsafteroutcomes. Passingcandidatewouldneedseparateadditional-seedconfirmation. Mainapplication,publication,generationandreconstructionremainunchanged.',
        hashes={**inherited, **{str(p.relative_to(PROJECT)): digest(p) for p in paths}}))


def verify():
    p = read(ROOT / 'protocol.json')
    for name, value in p['hashes'].items():
        assert digest(PROJECT / name) == value, name
    return p


def prepare(seed):
    protocol = verify()
    assert not read(FINETUNING / 'results.json')['decisions']['transformer_full'], 'fallback only after the frozen fine-tuning gate fails'
    data = ref.load_data()
    chosen = read(ROOT / 'recipe_lock.json')['choices'] if seed >= 13 else None
    for family in FAMILIES:
        net = Model(family, seed, range(4))
        state = torch.load(native_dir(family, seed) / 'selected.pt', weights_only=True)
        net.load_state_dict(state)
        directory = ROOT / f'{family}_s{seed}'
        directory.mkdir()
        configs = CONFIGS if chosen is None else [chosen[family]['config']]
        records, saved = [], {}
        archive = np.load(native_dir(family, seed) / 'selection_predictions.npz')
        epoch = read(native_dir(family, seed) / 'result.json')['selected_epoch']
        for s, mouse in enumerate(MICE):
            f, p = features(net, data[s]['x'], s)
            fv, pv = features(net, data[s]['xv'], s)
            np.testing.assert_array_equal(pv, archive[mouse + '_predictions'][epoch])
            saved[mouse + '_native'] = pv
            saved[mouse + '_target'] = data[s]['yv']
            heads, scores, predictions = [], [], []
            for config in configs:
                head = fit(f, p, data[s]['y'].numpy(), config)
                out = predict_head(fv, pv, head)
                assert np.isfinite(out).all()
                heads.append(head)
                predictions.append(out)
                scores.append(ref.mse(out, data[s]['yv'], data[s]['lower']))
            records.append(dict(mouse=mouse, heads=heads, selection_mse=scores, training_n=len(p)))
            saved[mouse + '_predictions'] = np.stack(predictions)
        archive.close()
        assert all(torch.equal(v, net.state_dict()[k]) for k, v in state.items())
        joint = np.mean([np.array(r['selection_mse']) / protocol['selection_denominators'][r['mouse']] for r in records], axis=0)
        write(directory / 'result.json', dict(family=family, seed=seed, configs=configs, rows=records, joint_scores=joint.tolist(), encoder_unchanged=True))
        np.savez_compressed(directory / 'selection_predictions.npz', **saved)
        print(family, seed, 'prepared', flush=True)
    write(ROOT / f'seed{seed}_finished.json', dict(complete=True))


def choose():
    verify()
    choices, hashes, scores = {}, {}, {}
    for family in FAMILIES:
        values = np.mean([read(ROOT / f'{family}_s{s}' / 'result.json')['joint_scores'] for s in [10, 11, 12]], axis=0)
        index = int(np.argmin(values))
        choices[family] = dict(index=index, config=CONFIGS[index], score=float(values[index]))
        scores[family] = values.tolist()
        for seed in [10, 11, 12]:
            for name in ['result.json', 'selection_predictions.npz']:
                path = ROOT / f'{family}_s{seed}' / name
                hashes[str(path.relative_to(ROOT))] = digest(path)
    write(ROOT / 'recipe_lock.json', dict(choices=choices, scores=scores, hashes=hashes, later_scored=False))
    print('Recipes:', choices, flush=True)


def lock():
    p = verify()
    recipes = read(ROOT / 'recipe_lock.json')
    hashes, checks = {}, 0
    for name, value in recipes['hashes'].items():
        assert digest(ROOT / name) == value
    for family in FAMILIES:
        for seed in SEEDS:
            directory = ROOT / f'{family}_s{seed}'
            record = read(directory / 'result.json')
            if seed >= 13:
                assert record['configs'] == [recipes['choices'][family]['config']]
            with np.load(directory / 'selection_predictions.npz') as z:
                scores = []
                for row in record['rows']:
                    mouse = row['mouse']
                    meta = read(ref.FAIR / mouse / 'metadata.json')
                    np.testing.assert_array_equal(z[mouse + '_target'], np.load(ref.BASE / mouse / 'selection_y.npy'))
                    values = [ref.mse(pred, z[mouse + '_target'], -meta['speed_mean'] / meta['speed_std']) for pred in z[mouse + '_predictions']]
                    np.testing.assert_allclose(values, row['selection_mse'], rtol=1e-12, atol=1e-12)
                    scores.append(np.array(values) / p['selection_denominators'][mouse])
                    checks += len(values)
                np.testing.assert_allclose(np.mean(scores, axis=0), record['joint_scores'], rtol=1e-12, atol=1e-12)
            for name in ['result.json', 'selection_predictions.npz']:
                hashes[str((directory / name).relative_to(ROOT))] = digest(directory / name)
    write(ROOT / 'selection_lock.json', dict(choices=recipes['choices'], hashes=hashes,
        recipe_sha256=digest(ROOT / 'recipe_lock.json'), selection_scores_checked=checks, later_scored=False))


def verify_lock():
    verify()
    locked = read(ROOT / 'selection_lock.json')
    assert digest(ROOT / 'recipe_lock.json') == locked['recipe_sha256']
    for name, value in locked['hashes'].items():
        assert digest(ROOT / name) == value
    return locked


def evaluate():
    locked = verify_lock()
    hashes, checks = {}, 0
    for s, mouse in enumerate(MICE):
        meta = read(ref.FAIR / mouse / 'metadata.json')
        columns = np.load(ref.BASE / mouse / 'columns.npy')
        with np.load(ref.FAIR / mouse / 'later_raw.npz') as raw, np.load(ref.FAIR / mouse / 'statistics.npz') as norm:
            seq = ((raw['activity'][columns, 24:] - norm['activity_mean'][columns]) / norm['activity_std'][columns]).T.astype(np.float32)
            target = (raw['speed'][55:] - meta['speed_mean']) / meta['speed_std']
        x = torch.from_numpy(np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(seq, 32, axis=0)))
        saved = dict(target=target)
        with np.load(REPLICATION / mouse / 'later_predictions.npz') as old:
            np.testing.assert_array_equal(target, old['target'])
            for family in FAMILIES:
                for seed in SEEDS:
                    net = Model(family, seed, range(4))
                    state = torch.load(native_dir(family, seed) / 'selected.pt', weights_only=True)
                    net.load_state_dict(state)
                    f, pred = features(net, x, s)
                    np.testing.assert_array_equal(pred, old[f'{family}_s{seed}'])
                    checks += 1
                    record = read(ROOT / f'{family}_s{seed}' / 'result.json')
                    index = locked['choices'][family]['index'] if seed < 13 else 0
                    head = record['rows'][s]['heads'][index]
                    assert record['rows'][s]['mouse'] == mouse
                    saved[f'native_{family}_s{seed}'] = old[f'{family}_s{seed}']
                    saved[f'tuned_{family}_s{seed}'] = predict_head(f, pred, head)
                    assert all(torch.equal(v, net.state_dict()[k]) for k, v in state.items())
        directory = ROOT / mouse
        directory.mkdir()
        path = directory / 'later_predictions.npz'
        np.savez_compressed(path, **saved)
        hashes[str(path.relative_to(ROOT))] = digest(path)
        print(mouse, 'scored', flush=True)
    verify_lock()
    write(ROOT / 'evaluation.json', dict(complete=True, prediction_hashes=hashes,
        exact_native_predictions_during_new_feature_extraction=checks, neural_fits=0))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        if sys.argv[1] == 'prepare':
            for seed in map(int, sys.argv[2:]):
                prepare(seed)
        else:
            {'check': check, 'freeze': freeze, 'choose': choose, 'lock': lock, 'evaluate': evaluate}[sys.argv[1]]()
