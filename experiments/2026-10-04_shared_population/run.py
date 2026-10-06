"""Fixed population-block comparison on the accepted shared behavior baseline."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
EXP = ROOT.parent
SHARED = EXP / '2026-10-03_shared_behavior'
REPLICATION = EXP / '2026-10-04_ensemble_seed_replication'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


new_models = module('shared_population_models', ROOT / 'models.py')
sys.path.insert(0, str(SHARED))
reference = module('shared_population_reference', SHARED / 'run.py')
PopulationDecoder = new_models.PopulationDecoder
MICE = reference.MICE
SEEDS = list(range(10, 16))
VARIANTS = ['attention', 'mixer']


def read(path):
    return json.loads(path.read_text())


def write(path, value, replace=False):
    assert replace or not path.exists(), f'refuse to replace {path}'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def parent_dir(seed):
    return (SHARED / 'shared' if seed < 13 else REPLICATION) / f'attention_s{seed}'


def check():
    torch.manual_seed(93200)
    x = torch.randn(3, 128, 32)
    original = x.clone()
    counts, extra, equivalences = {}, {}, 0
    for variant in VARIANTS:
        for seed in SEEDS:
            net = PopulationDecoder(variant, seed, [0, 1, 2, 3])
            initial = torch.load(parent_dir(seed) / 'initial.pt', weights_only=True)
            for k, v in initial.items():
                assert torch.equal(net.state_dict()[k], v), (variant, seed, k)
            assert all(k in initial or k.startswith('population.') for k in net.state_dict())
            trained = torch.load(parent_dir(seed) / 'selected.pt', weights_only=True)
            incompatible = net.load_state_dict(trained, strict=False)
            assert not incompatible.unexpected_keys and all(k.startswith('population.') for k in incompatible.missing_keys)
            parent = reference.BehaviorDecoder('attention', seed, [0, 1, 2, 3])
            parent.load_state_dict(trained)
            net.population_enabled = False
            for session in range(4):
                np.testing.assert_array_equal(reference.predict(net, x, session), reference.predict(parent, x, session))
                equivalences += 1
        net = PopulationDecoder(variant, 10, [0, 1, 2, 3]).eval()
        counts[variant] = sum(p.numel() for p in net.parameters())
        extra[variant] = sum(p.numel() for p in net.population.parameters())
        z = torch.randn(2, 128, 8, 16)
        changed = z.clone()
        changed[0, 1, -1, 0] += 4
        with torch.no_grad():
            a, b = net.mix_latest(z), net.mix_latest(changed)
        assert torch.equal(a[:, :, :-1], z[:, :, :-1])
        assert not torch.allclose(a[0, 0, -1], b[0, 0, -1], atol=1e-7, rtol=1e-7)
        assert torch.equal(a[1], b[1])
        np.testing.assert_array_equal(reference.predict(net, x, 0), np.zeros(3, dtype=np.float32))
        opt = torch.optim.AdamW(net.parameters(), lr=.001)
        for step in range(3):
            net.train()
            opt.zero_grad(set_to_none=True)
            loss = (net(x, 0) - torch.arange(3, dtype=torch.float32)).square().mean()
            loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
            opt.step()
        key = 'mix.in_proj_weight' if variant == 'attention' else 'mix.0.weight'
        assert dict(net.population.named_parameters())[key].grad.norm() > 0
        assert net.temporal.mix.in_proj_weight.grad.norm() > 0
        assert net.key_projection.weight.grad.norm() > 0
        clone = PopulationDecoder(variant, 10, [0, 1, 2, 3])
        clone.load_state_dict(net.state_dict())
        np.testing.assert_array_equal(reference.predict(net, x, 0), reference.predict(clone, x, 0))
        assert torch.equal(x, original)
    a, b = [PopulationDecoder(v, 10, range(4)).state_dict() for v in VARIANTS]
    for k, v in a.items():
        if not k.startswith('population.mix.'):
            assert torch.equal(v, b[k]), k
    assert abs(counts['attention'] / counts['mixer'] - 1) < .01
    write(ROOT / 'selfcheck.json', dict(passed=True, parameter_counts=counts, added_parameters=extra,
        archived_base_parameters_exact=True, trained_parent_equivalences=equivalences,
        common_population_norm_and_ff_initialization_exact=True, earliest7patches_preserved=True,
        latest_state_cross_neuron_effect=True, no_cross_example_mixing=True,
        finite_gradients_in_population_temporal_and_query=True, reload_exact=True,
        input_preserved=True, zero_initial_predictions=True))


def freeze():
    checks = read(ROOT / 'selfcheck.json')
    assert checks['passed']
    files = [ROOT / n for n in ['models.py', 'run.py', 'analyse.py', 'selfcheck.json']]
    files += [SHARED / n for n in ['models.py', 'run.py', 'protocol.json']]
    files += [reference.BASE / 'models.py'] + [PROJECT / n for n in ['train.py', 'model.py', 'data.py']]
    for seed in SEEDS:
        files += [parent_dir(seed) / n for n in ['initial.pt', 'selected.pt', 'history.json', 'result.json']]
    for mouse in MICE:
        files += [reference.BASE / mouse / n for n in ['train_x.npy', 'train_y.npy', 'selection_x.npy', 'selection_y.npy', 'columns.npy']]
        files += [reference.FAIR / mouse / n for n in ['later_raw.npz', 'statistics.npz', 'metadata.json']]
        files += [REPLICATION / mouse / 'later_predictions.npz']
    write(ROOT / 'protocol.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        question='Does adding explicit neuron-to-neuron attention to the shared temporal transformer improve concurrent running-speed decoding beyond both the accepted baseline and a matched nonattention population mixer?',
        scope='Four historically reused Stringer recordings,128stable neurons x32historybins. Intact inputs only. No coordinates,masking,pretraining,behavior inputs,new labels or main application edits. Older population-block experiments used separate models/different readouts;this isolates the addition to the shared behavior-query baseline.',
        architecture='Construct exact original shared temporal-attention/dynamic-query decoder. After temporal processing,apply one population AxisBlock to each neuron latest time state (B,128,16),which can already contain its32-bin history. Replace only the last of8time states for each neuron;keep the earlier7states and unchanged behavior query over all1024tokens. The64population mean/std features are unchanged. No communication across examples/sessions or future input.',
        variants='population attention:two-head width16 self-attention across128latest neuron states,plus residuals,LayerNorm and16->32->16GELU feedforward. Population mixer:same residual/norm/feedforward recipe;instead mix neuron slots with128->4->128GELU MLP separately perfeature. Both retain the baseline temporal/query attention;the mixer variant is NOT an all-MLP decoder. Reuse archived unmodified shared transformer as third arm.',
        matching='Common parent tensors exactly equal archived initializations. Both new population blocks share exact norm/feedforward initial tensors;new routing tensors necessarily differ. Same dropout probability. Added operations consume different random draws,so internal dropout realizations are not claimed identical. Parameter counts close,not FLOPs or wall-time matched. Mixer weights share neuron-slot positions across sessions;neuron correspondence across mice is not assumed. This is one low-rank nonlinear mixer control,not every possible nonattention architecture.',
        parameter_counts=checks['parameter_counts'], added_parameters=checks['added_parameters'], baseline_parameters=18337,
        budget='12new fits:two additions xseeds10-15. 24epochs/5688AdamWupdates/179712example exposures each. Three workers runseedpairs10+13,11+14,12+15,two torchthreads each. Baseline6fits reused. Same parent batch order,lossweights,AdamWlr.001/wd.01,clip1,cosine24 eta_min.0001. Fresh frommatchedinitialization,not continuedtrainedcheckpoints.',
        selection='Original clean validation rule:selectepoch0..24 minimizing equal-mouse boundedMSE dividedby originalfixedridge denominators. Same24checkpointopportunitiesperarm. All12choices lockbefore currentlaterinference. No pair/epoch/modelselection from later outcomes.',
        selection_denominators=read(SHARED / 'protocol.json')['selection_denominators'],
        primary='Population attention must beat BOTH native baseline and population mixer:>=5%equal-mouse mean relative MSEgain,>=3/4mousewins,>=16/24paired-single-seedwins for eachcontrast. Additionally no mousemean >10%MSEharm versus native baseline. All required for adoption. Practical criteria,not statistical significance.',
        aggregation='Primary mean error of all15distinct two-model seedpairs perarm. Bound each individual output at physicalzero before50:50pair averaging. Averagepairerrorswithinmouse thenrelativegains equally overfourmice. Pairedseedwins useindividualmodels. Report singles,pairs,MSE,MAE,rawdenominators,leave-one-mouse-out and initial/training-median references. No bestpairselection.',
        secondary='Population mixer versus native baseline using same5%/3mouse/16seed thresholds and10%harmguard;MAE for allcontrasts,allrawpermouseandseedresults,initial-learning checks. Archived matched temporalMLP cleanpredictions are context only. Population weights are not causal connectivity or proof of co-firing mechanism.',
        uncertainty='Descriptive comparisons on reused cohort;no new pvalues or independentconfirmation. Sharedtraining couplesmice;seeds,pairs,overlappingwindows are notnewanimals. Parameter-matched control doesnot equal exhaustive optimizer/architecture search.',
        stop='Finish12fits,lock,evaluate,analyse,technical audit,report,handoff. No extension ofpopulationplacement/rank/depth/width/seed grid or changedgateafteroutcomes. Retainoriginalbaselineiffullgatefails;no publication,generation ormainapplicationmigration.',
        hashes={str(p.relative_to(PROJECT)): digest(p) for p in files}))


def verify():
    protocol = read(ROOT / 'protocol.json')
    for name, value in protocol['hashes'].items():
        assert digest(PROJECT / name) == value, name
    return protocol


def train(seed):
    assert seed in SEEDS
    protocol = verify()
    data = reference.load_data()
    sessions = list(range(4))
    lengths = [len(data[s]['x']) for s in sessions]
    total = sum(lengths)
    parent_history = read(parent_dir(seed) / 'history.json')
    for variant in VARIANTS:
        out = ROOT / f'{variant}_s{seed}'
        out.mkdir()
        net = PopulationDecoder(variant, seed, sessions)
        initial = torch.load(parent_dir(seed) / 'initial.pt', weights_only=True)
        for k, v in initial.items():
            assert torch.equal(net.state_dict()[k], v)
        def selection():
            p = {s: reference.predict(net, data[s]['xv'], s) for s in sessions}
            scores = {s: reference.mse(p[s], data[s]['yv'], data[s]['lower']) for s in sessions}
            joint = float(np.mean([scores[s] / protocol['selection_denominators'][MICE[s]] for s in sessions]))
            return p, scores, joint
        first, scores, best = selection()
        assert all(np.array_equal(p, np.zeros_like(p)) for p in first.values())
        assert best == parent_history[0]['selection_score']
        predictions = {s: [first[s]] for s in sessions}
        history = [dict(epoch=0, selection_score=best, mouse_mse={MICE[s]: v for s, v in scores.items()})]
        torch.save(net.state_dict(), out / 'initial.pt')
        torch.save(net.state_dict(), out / 'selected.pt')
        selected = 0
        torch.manual_seed(seed + 9000)
        opt = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.01)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 24, eta_min=.0001)
        start = time.monotonic()
        updates = examples = 0
        max_grad = pop_grad = 0.
        for epoch in range(1, 25):
            net.train()
            order = hashlib.sha256()
            counts, train_sse = [0] * 4, [0.] * 4
            for s, indices in reference.batches(lengths, sessions, seed, epoch):
                order.update(np.asarray([s], dtype=np.int64).tobytes() + indices.tobytes())
                xb, yb = data[s]['x'][indices], data[s]['y'][indices]
                opt.zero_grad(set_to_none=True)
                raw = (net(xb, s) - yb).square().mean()
                loss = raw * (total / (4 * lengths[s])) * (len(indices) / 32)
                assert torch.isfinite(loss)
                loss.backward()
                key = 'mix.in_proj_weight' if variant == 'attention' else 'mix.0.weight'
                pop_grad = max(pop_grad, float(dict(net.population.named_parameters())[key].grad.norm()))
                norm = nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
                max_grad = max(max_grad, float(norm))
                opt.step()
                updates += 1
                examples += len(indices)
                counts[s] += len(indices)
                train_sse[s] += float(raw.detach()) * len(indices)
            assert counts == lengths and order.hexdigest() == parent_history[epoch]['global_order_hash']
            schedule.step()
            p, scores, score = selection()
            if score < best:
                best, selected = score, epoch
                torch.save(net.state_dict(), out / 'selected.pt')
            for s in sessions:
                predictions[s].append(p[s])
            history.append(dict(epoch=epoch, selection_score=score, mouse_mse={MICE[s]: v for s, v in scores.items()},
                training_mse={MICE[s]: train_sse[s] / lengths[s] for s in sessions}, global_order_hash=order.hexdigest(),
                examples_per_mouse=counts, updates=updates, examples=examples))
            write(out / 'history.json', history, replace=True)
            if epoch % 6 == 0:
                print(seed, variant, 'epoch', epoch, 'selected', selected, flush=True)
        actual_steps = sorted({int(v['step'].item()) for v in opt.state.values() if 'step' in v})
        assert updates == 5688 and examples == 179712 and actual_steps == [5688] and pop_grad > 0
        torch.save(net.state_dict(), out / 'final.pt')
        net.load_state_dict(torch.load(out / 'selected.pt', weights_only=True))
        saved = {}
        for s in sessions:
            np.testing.assert_array_equal(reference.predict(net, data[s]['xv'], s), predictions[s][selected])
            saved[MICE[s] + '_predictions'] = np.stack(predictions[s])
            saved[MICE[s] + '_target'] = data[s]['yv']
        np.savez_compressed(out / 'selection_predictions.npz', **saved)
        write(out / 'result.json', dict(variant=variant, seed=seed, selected_epoch=selected, selection_score=best,
            epochs=24, updates=updates, examples=examples, actual_adam_steps=actual_steps,
            selected_reload_exact=True, common_initial_tensors_exact=True, initial_predictions_zero=True,
            max_gradient=max_grad, population_gradient_max=pop_grad, elapsed_seconds=time.monotonic() - start))
        print(seed, variant, 'complete', flush=True)
    write(ROOT / f'seed{seed}_finished.json', dict(complete=True))


def lock():
    protocol = verify()
    records, hashes, orders = [], {}, {}
    checked = 0
    for seed in SEEDS:
        assert read(ROOT / f'seed{seed}_finished.json')['complete']
        for variant in VARIANTS:
            out = ROOT / f'{variant}_s{seed}'
            record, history = read(out / 'result.json'), read(out / 'history.json')
            assert len(history) == 25 and record['updates'] == 5688 and record['examples'] == 179712
            assert record['actual_adam_steps'] == [5688] and record['population_gradient_max'] > 0
            scores = []
            with np.load(out / 'selection_predictions.npz') as z:
                for mouse in MICE:
                    np.testing.assert_array_equal(z[mouse + '_target'], np.load(reference.BASE / mouse / 'selection_y.npy'))
                    meta = read(reference.FAIR / mouse / 'metadata.json')
                    values = [reference.mse(p, z[mouse + '_target'], -meta['speed_mean'] / meta['speed_std']) for p in z[mouse + '_predictions']]
                    np.testing.assert_allclose(values, [h['mouse_mse'][mouse] for h in history], rtol=1e-12, atol=1e-12)
                    scores.append(np.array(values) / protocol['selection_denominators'][mouse])
                    checked += len(values)
            joint = np.mean(scores, axis=0)
            np.testing.assert_allclose(joint, [h['selection_score'] for h in history], rtol=1e-12, atol=1e-12)
            assert int(np.argmin(joint)) == record['selected_epoch']
            order = [h['global_order_hash'] for h in history[1:]]
            assert order == [h['global_order_hash'] for h in read(parent_dir(seed) / 'history.json')[1:]]
            if seed in orders:
                assert orders[seed] == order
            orders[seed] = order
            records.append(record)
            for name in ['selected.pt', 'history.json', 'result.json', 'selection_predictions.npz']:
                hashes[str((out / name).relative_to(ROOT))] = digest(out / name)
    assert checked == 1200
    write(ROOT / 'selection_lock.json', dict(locked_utc=datetime.now(timezone.utc).isoformat(), records=records,
        hashes=hashes, selection_scores_checked=checked, batches_match_native=True, new_later_scored=False))


def verify_lock():
    verify()
    locked = read(ROOT / 'selection_lock.json')
    for name, value in locked['hashes'].items():
        assert digest(ROOT / name) == value, name
    return locked


def evaluate():
    verify_lock()
    hashes = {}
    for session, mouse in enumerate(MICE):
        out = ROOT / mouse
        out.mkdir()
        meta = read(reference.FAIR / mouse / 'metadata.json')
        columns = np.load(reference.BASE / mouse / 'columns.npy')
        with np.load(reference.FAIR / mouse / 'later_raw.npz') as raw, np.load(reference.FAIR / mouse / 'statistics.npz') as norm:
            seq = ((raw['activity'][columns, 24:] - norm['activity_mean'][columns]) / norm['activity_std'][columns]).T.astype(np.float32)
            y = (raw['speed'][55:] - meta['speed_mean']) / meta['speed_std']
        x = torch.from_numpy(np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(seq, 32, axis=0)))
        original = x.clone()
        predictions = dict(target=y)
        with np.load(REPLICATION / mouse / 'later_predictions.npz') as saved:
            np.testing.assert_array_equal(y, saved['target'])
            for seed in SEEDS:
                predictions[f'baseline_s{seed}'] = saved[f'attention_s{seed}']
                predictions[f'prior_mlp_s{seed}'] = saved[f'mlp_s{seed}']
                parent = reference.BehaviorDecoder('attention', seed, range(4))
                parent.load_state_dict(torch.load(parent_dir(seed) / 'selected.pt', weights_only=True))
                np.testing.assert_array_equal(reference.predict(parent, x[:64], session), saved[f'attention_s{seed}'][:64])
                for variant in VARIANTS:
                    net = PopulationDecoder(variant, seed, range(4))
                    state = torch.load(ROOT / f'{variant}_s{seed}' / 'selected.pt', weights_only=True)
                    net.load_state_dict(state)
                    predictions[f'{variant}_s{seed}'] = reference.predict(net, x, session)
                    for k, v in net.state_dict().items():
                        assert torch.equal(v, state[k])
        assert torch.equal(x, original)
        path = out / 'later_predictions.npz'
        np.savez_compressed(path, **predictions)
        hashes[str(path.relative_to(ROOT))] = digest(path)
        print(mouse, 'scored', flush=True)
    verify_lock()
    write(ROOT / 'evaluation.json', dict(complete=True, prediction_hashes=hashes,
        exact_baseline_firstbatch_checks=24, exact_target_alignment=True, model_and_input_unchanged=True))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        if sys.argv[1] == 'train':
            for seed in map(int, sys.argv[2:]):
                train(seed)
        else:
            {'check': check, 'freeze': freeze, 'lock': lock, 'evaluate': evaluate}[sys.argv[1]]()
