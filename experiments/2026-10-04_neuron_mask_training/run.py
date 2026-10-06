"""Train one fixed whole-neuron masking recipe against archived matched controls."""
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


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


replication = module('mask_replication', EXP / '2026-10-04_ensemble_seed_replication' / 'run.py')
reference = replication.reference
stress = module('mask_stress', EXP / '2026-10-04_input_stress' / 'run.py')
BehaviorDecoder = reference.BehaviorDecoder
MICE = reference.MICE
SEEDS = list(range(10, 16))
FAMILIES = ['attention', 'mlp']
CONDITIONS = ['clean', 'missing_16', 'missing_32', 'missing_64']
VIEWS = 5


def read(path):
    return json.loads(path.read_text())


def write(path, value, replace=False):
    assert replace or not path.exists(), f'refuse to replace {path}'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def parent_dir(family, seed):
    return stress.checkpoint(family, seed).parent


def make_mask(batch, seed, epoch, step):
    rng = np.random.default_rng(np.random.SeedSequence([92410, seed, epoch, step]))
    chosen = rng.random(batch) < .5
    cells = np.argsort(rng.random((batch, 128)), axis=1)[:, :32]
    mask = np.zeros((batch, 128), dtype=bool)
    mask[np.arange(batch)[:, None], cells] = chosen[:, None]
    return mask


def augment(x, seed, epoch, step):
    mask = make_mask(len(x), seed, epoch, step)
    return x.masked_fill(torch.from_numpy(mask)[:, :, None], 0), mask


def check():
    x = torch.arange(32 * 128 * 32, dtype=torch.float32).reshape(32, 128, 32) + 1
    before = x.clone()
    torch_state = torch.get_rng_state().clone()
    changed, mask = augment(x, 10, 1, 0)
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert torch.equal(x, before) and changed.shape == x.shape
    assert set(mask.sum(1).tolist()) == {0, 32}
    np.testing.assert_array_equal(mask, make_mask(32, 10, 1, 0))
    assert not np.array_equal(mask, make_mask(32, 10, 2, 0))
    assert torch.count_nonzero(changed[torch.from_numpy(mask)]) == 0
    assert torch.equal(changed[torch.from_numpy(~mask)], x[torch.from_numpy(~mask)])
    assert np.isin(make_mask(1, 10, 1, 0).sum(1), [0, 32]).all()
    init_checks = 0
    for seed in SEEDS:
        for family in FAMILIES:
            net = BehaviorDecoder(family, seed, [0, 1, 2, 3])
            saved = torch.load(parent_dir(family, seed) / 'initial.pt', weights_only=True)
            for k, v in net.state_dict().items():
                assert torch.equal(v, saved[k]), (family, seed, k)
            init_checks += 1
    for family in FAMILIES:
        net = BehaviorDecoder(family, 10, [0, 1, 2, 3])
        xb = torch.randn(4, 128, 32)
        yb = torch.arange(4, dtype=torch.float32)
        opt = torch.optim.AdamW(net.parameters(), lr=.001)
        for step in range(2):
            masked, _ = augment(xb, 10, 1, step)
            opt.zero_grad(set_to_none=True)
            loss = (net(masked, 0) - yb).square().mean()
            loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
            opt.step()
        assert net.patch.weight.grad.norm() > 0
        key = 'temporal.mix.in_proj_weight' if family == 'attention' else 'temporal.weight'
        assert dict(net.named_parameters())[key].grad.norm() > 0
    write(ROOT / 'selfcheck.json', dict(passed=True, source_input_preserved=True,
        whole_histories_zeroed=True, exactly32_neurons_per_masked_example=True,
        clean_examples_present=True, masks_deterministic_and_change_by_epoch=True,
        torch_dropout_rng_unchanged=True, partial_batch_supported=True,
        exact_archived_initializations=init_checks, finite_gradients_and_body_learning=True))


def freeze():
    assert read(ROOT / 'selfcheck.json')['passed']
    files = [ROOT / 'run.py', ROOT / 'analyse.py', ROOT / 'selfcheck.json',
        EXP / '2026-10-04_ensemble_seed_replication' / 'run.py',
        EXP / '2026-10-04_input_stress' / 'run.py',
        reference.ROOT / 'run.py', reference.ROOT / 'models.py', reference.ROOT / 'protocol.json', reference.BASE / 'models.py']
    files += [PROJECT / n for n in ['train.py', 'model.py', 'data.py']]
    for family in FAMILIES:
        for seed in SEEDS:
            files += [parent_dir(family, seed) / n for n in ['initial.pt', 'selected.pt', 'history.json', 'result.json']]
    plans = {}
    for session, mouse in enumerate(MICE):
        files += [reference.BASE / mouse / n for n in ['train_x.npy', 'train_y.npy', 'selection_x.npy', 'selection_y.npy', 'columns.npy']]
        files += [reference.FAIR / mouse / n for n in ['later_raw.npz', 'statistics.npz', 'metadata.json']]
        files += [stress.REPLICATION / mouse / 'later_predictions.npz']
        rng = np.random.default_rng(92490 + session)
        plans[mouse] = [dict(neurons=rng.permutation(128).tolist()) for _ in range(VIEWS)]
    write(ROOT / 'evaluation_plans.json', plans)
    files += [ROOT / 'evaluation_plans.json']
    write(ROOT / 'protocol.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        question='Does one fixed whole-neuron masking training recipe improve the shared transformer under missing recordings without materially harming intact-panel speed decoding,and does the same treatment help the MLP?',
        baseline='User retains the original shared transformer as the baseline. New augmentation is a candidate,not an automatic baseline replacement. Same four historically reused Stringer mice and archived chronological partitions.',
        design='12new shared fits:unchanged transformer and matched static-query MLP,seeds10-15. Corresponding12native fits and clean later predictions reused. All exact initial tensors match archived parents. No architecture,indicator,loss,normalization,neuron-panel or target changes.',
        augmentation='Independently for each training example,probability.5 of replacing exactly32/128whole histories by normalized0(trainingmeans);otherwise clean. Uniform random cell subsets,redrawn per batch/epoch. PreserveIDslots,no rescaling,no missingness indicator. Both token and population-summary inputs see the corruption. Same masks for both families at matched seed/epoch/batch;NumPy RNG does not consume torch dropout RNG. This changes training information content intentionally.',
        budget='24epochs,5688AdamWupdates,179712training examples per fit. Exact parent batch generator,loss weights,optimizerlr.001/wd.01,clip1,cosine24 eta_min.0001. Three workers,two torch threads each;seeds10+13,11+14,12+15. No additional seeds or masking recipes after results.',
        selection='Original clean-only joint selection rule,epoch0..24:equal-mouse mean bounded MSE divided by fixed original ridge selection denominators. No missing-data validation in checkpoint selection. Lock all12selected checkpoints and verify1200mouse/epoch scores before current later inference. No ensemble checkpoint,pair or mask selection.',
        selection_denominators=read(reference.ROOT / 'protocol.json')['selection_denominators'],
        evaluation='Clean and exactly16/32/64missing histories. Five new fixed nested masks per mouse,identical across families,seeds,treatments and all laterwindows;no masks copied from the prior stress bank. Both augmented and archived native checkpoints scored under these new masks. Native clean predictions reused with first64windows verified. New masks are a new perturbation bank,not new animals or independent confirmation.',
        primary='Augmented transformer versus native transformer at25%missingness:>=10%equal-mouse mean relative MSE gain,>=3/4mouse wins,>=16/24paired-single-seed wins. Clean safeguard:equal-mouse mean relative MSE harm<=5%,no mouse mean harm>10%. Both must pass to support this robustness recipe. Practical tolerances,not statistical noninferiority/significance.',
        secondary='Apply identical practical gate to MLP. Compare augmented transformer with augmented MLP atclean/12.5%/25%/50%missingness. Transformer advantage gate at25%:>=5%mean MSEgain,>=3/4mice,>=16/24pairedseeds. ShowMSE/MAE,allmice/singles/pairs/views,clean-relative degradation,training-median controls,initialprediction learning,view ranges and leave-one-mouse-out means. No change to primary thresholds based on secondary outcomes.',
        aggregation='Bound individual predictions at physicalzero,averageeach of all15distinct two-model pairs,and average errors overfive fixedviews. Relative treatment gains use native score on samecondition. Equal weight permouse. Single-seedwins averageerrors overviews. Views/pairs/windows/seeds aredependent;fourmicearecohortunits.',
        uncertainty='Descriptive fixed-cohort comparisons only. No new pvalues or independent significance. Augmentation choice motivated by prior later-data stress results;new mask bank does not repair historical cohort reuse. No claims about actual neuron loss or unseen-animal transfer.',
        stop='Finish12fits,lock,evaluate,analyse,audit,report,handoff. If either primary robustness orclean safeguard fails,retain originalbaseline;do not tune maskrate/selection/loss or launch anothergrid. Mainapplicationunchanged,no publication or generation.',
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
    for family in FAMILIES:
        out = ROOT / f'{family}_s{seed}'
        out.mkdir()
        net = BehaviorDecoder(family, seed, sessions)
        initial = torch.load(parent_dir(family, seed) / 'initial.pt', weights_only=True)
        for k, v in net.state_dict().items():
            assert torch.equal(v, initial[k])
        parent_history = read(parent_dir(family, seed) / 'history.json')
        def selection():
            p = {s: reference.predict(net, data[s]['xv'], s) for s in sessions}
            scores = {s: reference.mse(p[s], data[s]['yv'], data[s]['lower']) for s in sessions}
            joint = float(np.mean([scores[s] / protocol['selection_denominators'][MICE[s]] for s in sessions]))
            return p, scores, joint
        first, scores, best = selection()
        assert all(np.array_equal(p, np.zeros_like(p)) for p in first.values())
        np.testing.assert_allclose(best, parent_history[0]['selection_score'], rtol=0, atol=0)
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
        max_grad = 0.
        for epoch in range(1, 25):
            net.train()
            order, mask_hash = hashlib.sha256(), hashlib.sha256()
            counts, masked_counts, train_sse = [0] * 4, [0] * 4, [0.] * 4
            for step, (s, indices) in enumerate(reference.batches(lengths, sessions, seed, epoch)):
                header = np.asarray([s], dtype=np.int64).tobytes() + indices.tobytes()
                order.update(header)
                xb, yb = data[s]['x'][indices], data[s]['y'][indices]
                changed, mask = augment(xb, seed, epoch, step)
                mask_hash.update(header + mask.tobytes())
                masked_counts[s] += int(mask.any(1).sum())
                opt.zero_grad(set_to_none=True)
                raw = (net(changed, s) - yb).square().mean()
                loss = raw * (total / (4 * lengths[s])) * (len(indices) / 32)
                assert torch.isfinite(loss)
                loss.backward()
                norm = nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
                max_grad = max(max_grad, float(norm))
                opt.step()
                updates += 1
                examples += len(indices)
                counts[s] += len(indices)
                train_sse[s] += float(raw.detach()) * len(indices)
            assert counts == lengths
            assert order.hexdigest() == parent_history[epoch]['global_order_hash']
            schedule.step()
            p, scores, score = selection()
            if score < best:
                best, selected = score, epoch
                torch.save(net.state_dict(), out / 'selected.pt')
            for s in sessions:
                predictions[s].append(p[s])
            history.append(dict(epoch=epoch, selection_score=score, mouse_mse={MICE[s]: v for s, v in scores.items()},
                training_mse={MICE[s]: train_sse[s] / lengths[s] for s in sessions},
                global_order_hash=order.hexdigest(), mask_hash=mask_hash.hexdigest(),
                examples_per_mouse=counts, masked_examples_per_mouse=masked_counts, updates=updates, examples=examples))
            write(out / 'history.json', history, replace=True)
            if epoch % 6 == 0:
                print(seed, family, 'epoch', epoch, 'selected', selected, flush=True)
        actual_steps = sorted({int(v['step'].item()) for v in opt.state.values() if 'step' in v})
        assert updates == 5688 and examples == 179712 and actual_steps == [5688] and max_grad > 0
        torch.save(net.state_dict(), out / 'final.pt')
        net.load_state_dict(torch.load(out / 'selected.pt', weights_only=True))
        saved = {}
        for s in sessions:
            np.testing.assert_array_equal(reference.predict(net, data[s]['xv'], s), predictions[s][selected])
            saved[MICE[s] + '_predictions'] = np.stack(predictions[s])
            saved[MICE[s] + '_target'] = data[s]['yv']
        np.savez_compressed(out / 'selection_predictions.npz', **saved)
        write(out / 'result.json', dict(family=family, seed=seed, selected_epoch=selected, selection_score=best,
            epochs=24, updates=updates, examples=examples, actual_adam_steps=actual_steps,
            selected_reload_exact=True, initial_tensors_exact=True, initial_predictions_zero=True,
            max_gradient=max_grad, elapsed_seconds=time.monotonic() - start,
            masked_examples=sum(sum(h['masked_examples_per_mouse']) for h in history[1:])))
        print(seed, family, 'complete', flush=True)
    write(ROOT / f'seed{seed}_finished.json', dict(complete=True))


def lock():
    protocol = verify()
    records, hashes, plans = [], {}, {}
    checked = 0
    for seed in SEEDS:
        assert read(ROOT / f'seed{seed}_finished.json')['complete']
        for family in FAMILIES:
            out = ROOT / f'{family}_s{seed}'
            record, history = read(out / 'result.json'), read(out / 'history.json')
            assert len(history) == 25 and record['updates'] == 5688 and record['examples'] == 179712
            assert record['actual_adam_steps'] == [5688]
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
            order = [(h['global_order_hash'], h['mask_hash'], h['masked_examples_per_mouse']) for h in history[1:]]
            if seed in plans:
                assert plans[seed] == order
            plans[seed] = order
            records.append(record)
            for name in ['selected.pt', 'history.json', 'result.json', 'selection_predictions.npz']:
                hashes[str((out / name).relative_to(ROOT))] = digest(out / name)
    assert checked == 1200
    write(ROOT / 'selection_lock.json', dict(locked_utc=datetime.now(timezone.utc).isoformat(), records=records,
        hashes=hashes, selection_scores_checked=checked, matched_batches_and_masks=True, new_later_scored=False))


def verify_lock():
    verify()
    locked = read(ROOT / 'selection_lock.json')
    for name, value in locked['hashes'].items():
        assert digest(ROOT / name) == value, name
    return locked


def evaluate(session):
    verify_lock()
    mouse = MICE[session]
    out = ROOT / mouse
    out.mkdir()
    x, y, lower = stress.load_later(mouse)
    original = x.clone()
    plans = read(ROOT / 'evaluation_plans.json')[mouse]
    records = []
    with np.load(stress.REPLICATION / mouse / 'later_predictions.npz') as saved:
        np.testing.assert_array_equal(y, saved['target'])
        for seed in SEEDS:
            for family in FAMILIES:
                for treatment in ['native', 'augmented']:
                    path = parent_dir(family, seed) / 'selected.pt' if treatment == 'native' else ROOT / f'{family}_s{seed}' / 'selected.pt'
                    net = BehaviorDecoder(family, seed, [0, 1, 2, 3])
                    state = torch.load(path, weights_only=True)
                    net.load_state_dict(state)
                    if treatment == 'native':
                        clean = saved[f'{family}_s{seed}']
                        np.testing.assert_array_equal(reference.predict(net, x[:64], session), clean[:64])
                    else:
                        clean = reference.predict(net, x, session)
                    predictions = dict(target=y, clean=clean[None])
                    for condition in CONDITIONS[1:]:
                        predictions[condition] = np.stack([reference.predict(net, stress.perturb(x, condition, plan), session) for plan in plans])
                    assert torch.equal(x, original)
                    for k, v in net.state_dict().items():
                        assert torch.equal(v, state[k])
                    dest = out / f'{treatment}_{family}_s{seed}.npz'
                    np.savez_compressed(dest, **predictions)
                    records.append(dict(treatment=treatment, family=family, seed=seed, hash=digest(dest)))
            print(mouse, 'seed', seed, 'scored', flush=True)
    verify_lock()
    write(out / 'complete.json', dict(mouse=mouse, n=len(y), lower=lower, records=records,
        original_input_and_checkpoint_tensors_preserved=True, exact_target_alignment=True))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        if sys.argv[1] in ['train', 'evaluate']:
            for value in map(int, sys.argv[2:]):
                {'train': train, 'evaluate': evaluate}[sys.argv[1]](value)
        else:
            {'check': check, 'freeze': freeze, 'lock': lock}[sys.argv[1]]()
