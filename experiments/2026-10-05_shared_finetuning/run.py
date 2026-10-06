"""Finite, matched search for gentle continuation of the shared decoders."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
from pathlib import Path
import json
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
sys.path.insert(0, str(SHARED))
spec = importlib.util.spec_from_file_location('finetuning_reference', SHARED / 'run.py')
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
Model = ref.BehaviorDecoder
MICE = ref.MICE
FAMILIES = ['attention', 'mlp']
SEEDS = list(range(10, 16))
SEARCH_SEEDS = [10, 11, 12]
RATES = {'lr1e4': .0001, 'lr3e5': .00003}
MODES = ['plain', 'averaged']
EPOCHS = 8


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


def native_dir(family, seed):
    return (SHARED / 'shared' if seed < 13 else REPLICATION) / f'{family}_s{seed}'


def fit_dir(family, rate, seed):
    return ROOT / f'{family}_{rate}_s{seed}'


@torch.no_grad()
def update_average(averaged, current, count):
    for name, dest in averaged.state_dict().items():
        source = current.state_dict()[name]
        if dest.is_floating_point():
            dest.add_((source - dest) / count)
        else:
            assert torch.equal(dest, source), name


def check():
    torch.manual_seed(91601)
    x = torch.randn(3, 128, 32)
    original = x.clone()
    starts = 0
    for family in FAMILIES:
        net = Model(family, 10, range(4))
        state = torch.load(native_dir(family, 10) / 'selected.pt', weights_only=True)
        net.load_state_dict(state)
        averaged = deepcopy(net)
        for session in range(4):
            np.testing.assert_array_equal(ref.predict(net, x, session), ref.predict(averaged, x, session))
            starts += 1
        states = [{k: v.clone() for k, v in net.state_dict().items()}]
        opt = torch.optim.AdamW(net.parameters(), lr=.0001, weight_decay=.01)
        for count in [2, 3]:
            net.train()
            opt.zero_grad(set_to_none=True)
            (net(x, 0) - torch.arange(3)).square().mean().backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
            assert net.key_projection.weight.grad.norm() > 0
            nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            states.append({k: v.clone() for k, v in net.state_dict().items()})
            before = {k: v.clone() for k, v in net.state_dict().items()}
            update_average(averaged, net, count)
            assert all(torch.equal(v, net.state_dict()[k]) for k, v in before.items())
            for k, v in averaged.state_dict().items():
                if v.is_floating_point():
                    torch.testing.assert_close(v, torch.stack([s[k] for s in states]).mean(0), rtol=2e-6, atol=2e-7)
                else:
                    assert torch.equal(v, state[k])
        clone = deepcopy(averaged)
        np.testing.assert_array_equal(ref.predict(averaged, x, 0), ref.predict(clone, x, 0))
        assert torch.equal(x, original)
    write(ROOT / 'selfcheck.json', dict(passed=True, exact_starting_model_matches=starts,
        running_average_matches_explicit_parameter_mean=True, buffers_preserved=True,
        averaging_does_not_mutate_training_model=True, finite_nonzero_gradients=True,
        reload_exact=True, input_preserved=True))


def freeze():
    assert read(ROOT / 'selfcheck.json')['passed']
    files = [ROOT / n for n in ['run.py', 'analyse.py', 'selfcheck.json']]
    files += [SHARED / n for n in ['run.py', 'models.py', 'protocol.json']]
    files += [ref.BASE / 'models.py'] + [PROJECT / n for n in ['train.py', 'model.py', 'data.py']]
    for family in FAMILIES:
        for seed in SEEDS:
            files += [native_dir(family, seed) / n for n in ['selected.pt', 'result.json', 'history.json', 'selection_predictions.npz']]
    for mouse in MICE:
        files += [ref.BASE / mouse / n for n in ['train_x.npy', 'train_y.npy', 'selection_x.npy', 'selection_y.npy', 'columns.npy']]
        files += [ref.FAIR / mouse / n for n in ['metadata.json', 'statistics.npz', 'later_raw.npz']]
        files += [REPLICATION / mouse / 'later_predictions.npz']
    write(ROOT / 'protocol.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        authorization='User requested sustained searching without breaks. Fixed study of previously proposed shared-model fine-tuning; no claim that a passing model is guaranteed.',
        question='Can gentle supervised continuation, with or without averaging weights along the path, improve the accepted shared transformer robustly beyond an equally tuned MLP?',
        architecture='Exact archived BehaviorDecoder for attention and matched MLP; no architecture, input, neuron, history, target or preprocessing changes.',
        search='Seeds10-12:two learning rates .0001/.00003 x two families x three seeds =12new fits. Each produces ordinary and running-average checkpoint sequences. Equal-average floating tensors over starting checkpoint plus epoch-end checkpoints so far; fixed buffers unchanged. Select best epoch0..8 separately for each mode, then one rate/mode per family by mean joint earlier score over three seeds; ties use listed rate/mode order. No later data enters this choice.',
        replication='Always train BOTH families on13-15 with their selected rate:6additional fits,18total. Recipe/mode fixed from10-12; select own joint epoch0..8 using earlier validation. Score only selected recipe on later data; do not score losing recipes there.',
        training='Start at each archived selected shared checkpoint,not random initialization or final epoch. Reset AdamW; retain wd.01,clip1,batch32,equal-mouse loss weights,original batches for epoch indices25-32 and seed-specific dropout seed+19000. Eight epochs,1896updates,59904presentations perfit. Cosine schedule from chosen rate to one tenth over8epochs. Both families same added budget. Starting checkpoint remains epoch0 option. Base selected ancestry can differ.',
        selection_denominators=read(SHARED / 'protocol.json')['selection_denominators'],
        selection='Original joint equal-mouse bounded MSE/frozen ridge denominator. Earlier data and baseline checkpoint choices historically reused; this is exploratory optimization, not nested independent validation. All recipe choices and18fit checkpoint histories lock before current later inference.',
        primary='On additional13-15, tuned transformer must improve >=5% equal-mouse mean pair MSE,>=3/4mouse wins,>=8/12individual paired-seed wins versus BOTH native transformer and equally tuned MLP. No mouse>10%harm versus native transformer and nonnegative mean MAE gain versus native. Same criteria required on all10-15 with>=16/24seed wins. Original subset cannot rescue failed replication. Tuned MLP improvement over native MLP is secondary with same own-family criteria.',
        aggregation='Bound each individual output atphysical zero before averaging each distincttwo-model pair. Average errors over allpairs withinmouse,then relativechanges equallyover4mice. Report original10-12,additional13-15,all10-15; singles,allpairs,MAE,R2,leave-one-mouse-out and allperseed effects. No pair/seed selection afteroutcomes.',
        stop='Complete18fits,recipe choice,selection lock,later evaluation,independent audit andreport regardlessofoutcomes. No extra rates/epochs/modes appended afteroutcomes. A passing practical candidate needs a separately frozen additional-seed confirmation before being called robust; same four mice cannot provide independent animal-level significance.',
        limits='Four historically searched Stringer mice. No unseen-animal,biological-connectivity,significance or novel-architecture claim. Repeated later evaluations remain development evidence. No main application migration,publication,generation orstoppedreconstruction resumption.',
        hashes={str(p.relative_to(PROJECT)): digest(p) for p in files}))
    write(ROOT / 'environment.json', dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
        numpy=np.__version__, device='cpu', torch_threads=2, interop_threads=1))


def verify():
    protocol = read(ROOT / 'protocol.json')
    for name, value in protocol['hashes'].items():
        assert digest(PROJECT / name) == value, name
    return protocol


def choices():
    chosen = read(ROOT / 'recipe_lock.json')
    for name, value in chosen['hashes'].items():
        assert digest(ROOT / name) == value, name
    return chosen['choices']


def train(seed):
    protocol = verify()
    assert seed in SEEDS
    selected = choices() if seed not in SEARCH_SEEDS else None
    data = ref.load_data()
    lengths = [len(data[s]['x']) for s in range(4)]
    total = sum(lengths)
    for family in FAMILIES:
        rates = list(RATES) if seed in SEARCH_SEEDS else [selected[family]['rate']]
        native = native_dir(family, seed)
        native_record = read(native / 'result.json')
        for rate in rates:
            out = fit_dir(family, rate, seed)
            out.mkdir()
            net = Model(family, seed, range(4))
            state = torch.load(native / 'selected.pt', weights_only=True)
            net.load_state_dict(state)
            assert all(torch.equal(v, state[k]) for k, v in net.state_dict().items())
            averaged = deepcopy(net)

            def selection(model):
                pred = {s: ref.predict(model, data[s]['xv'], s) for s in range(4)}
                scores = {MICE[s]: ref.mse(pred[s], data[s]['yv'], data[s]['lower']) for s in range(4)}
                joint = float(np.mean([scores[m] / protocol['selection_denominators'][m] for m in MICE]))
                return pred, scores, joint

            first, scores, initial = selection(net)
            assert initial == native_record['selection_score']
            with np.load(native / 'selection_predictions.npz') as z:
                for s, mouse in enumerate(MICE):
                    np.testing.assert_array_equal(first[s], z[mouse + '_predictions'][native_record['selected_epoch']])
            predictions = {mode: {s: [first[s]] for s in range(4)} for mode in MODES}
            best = {mode: initial for mode in MODES}
            epochs = {mode: 0 for mode in MODES}
            histories = {mode: [dict(epoch=0, selection_score=initial, mouse_mse=scores)] for mode in MODES}
            for mode in MODES:
                torch.save(net.state_dict(), out / f'{mode}_selected.pt')
            torch.manual_seed(seed + 19000)
            opt = torch.optim.AdamW(net.parameters(), lr=RATES[rate], weight_decay=.01)
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS, eta_min=RATES[rate] / 10)
            start = time.monotonic()
            updates = examples = 0
            max_grad = 0.
            orders = []
            for epoch in range(1, EPOCHS + 1):
                net.train()
                order = hashlib.sha256()
                counts = [0] * 4
                sse = [0.] * 4
                for s, indices in ref.batches(lengths, list(range(4)), seed, epoch + 24):
                    order.update(np.asarray([s], dtype=np.int64).tobytes() + indices.tobytes())
                    opt.zero_grad(set_to_none=True)
                    raw = (net(data[s]['x'][indices], s) - data[s]['y'][indices]).square().mean()
                    loss = raw * (total / (4 * lengths[s])) * (len(indices) / 32)
                    assert torch.isfinite(loss)
                    loss.backward()
                    norm = nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
                    max_grad = max(max_grad, float(norm))
                    opt.step()
                    updates += 1
                    examples += len(indices)
                    counts[s] += len(indices)
                    sse[s] += float(raw.detach()) * len(indices)
                assert counts == lengths
                orders.append(order.hexdigest())
                schedule.step()
                update_average(averaged, net, epoch + 1)
                for mode, model in [('plain', net), ('averaged', averaged)]:
                    pred, scores, score = selection(model)
                    if score < best[mode]:
                        best[mode], epochs[mode] = score, epoch
                        torch.save(model.state_dict(), out / f'{mode}_selected.pt')
                    for s in range(4):
                        predictions[mode][s].append(pred[s])
                    histories[mode].append(dict(epoch=epoch, selection_score=score, mouse_mse=scores,
                        training_mse={MICE[s]: sse[s] / lengths[s] for s in range(4)},
                        order_hash=orders[-1], updates=updates, examples=examples))
                write(out / 'history.json', histories, replace=True)
                if epoch % 4 == 0:
                    print(seed, family, rate, epoch, flush=True)
            steps = sorted({int(v['step'].item()) for v in opt.state.values() if 'step' in v})
            assert updates == 1896 and examples == 59904 and steps == [1896] and max_grad > 0
            saved = {}
            for mode in MODES:
                net.load_state_dict(torch.load(out / f'{mode}_selected.pt', weights_only=True))
                for s, mouse in enumerate(MICE):
                    np.testing.assert_array_equal(ref.predict(net, data[s]['xv'], s), predictions[mode][s][epochs[mode]])
                    saved[f'{mode}_{mouse}'] = np.stack(predictions[mode][s])
                    saved[mouse + '_target'] = data[s]['yv']
            np.savez_compressed(out / 'selection_predictions.npz', **saved)
            write(out / 'result.json', dict(family=family, rate=rate, seed=seed, epochs=EPOCHS,
                selected_epochs=epochs, selection_scores=best, initial_score=initial,
                starting_epoch=native_record['selected_epoch'], starting_checkpoint_sha256=digest(native / 'selected.pt'),
                updates=updates, examples=examples, actual_adam_steps=steps, order_hashes=orders,
                initial_tensors_and_predictions_exact=True, selected_reload_exact=True,
                max_gradient=max_grad, elapsed_seconds=time.monotonic() - start))
            print(seed, family, rate, 'complete', flush=True)
    write(ROOT / f'seed{seed}_finished.json', dict(complete=True))


def choose():
    verify()
    assert all(read(ROOT / f'seed{s}_finished.json')['complete'] for s in SEARCH_SEEDS)
    candidates, picked, hashes = {}, {}, {}
    for family in FAMILIES:
        scores = []
        for rate in RATES:
            records = [read(fit_dir(family, rate, s) / 'result.json') for s in SEARCH_SEEDS]
            for mode in MODES:
                score = float(np.mean([r['selection_scores'][mode] for r in records]))
                scores.append(dict(rate=rate, mode=mode, score=score))
            for seed in SEARCH_SEEDS:
                out = fit_dir(family, rate, seed)
                for name in ['result.json', 'history.json', 'selection_predictions.npz'] + [f'{m}_selected.pt' for m in MODES]:
                    hashes[str((out / name).relative_to(ROOT))] = digest(out / name)
        candidates[family] = scores
        picked[family] = min(scores, key=lambda s: s['score'])
    write(ROOT / 'recipe_lock.json', dict(locked_utc=datetime.now(timezone.utc).isoformat(),
        candidates=candidates, choices=picked, hashes=hashes, later_scored=False))
    print('Selected recipes:', picked, flush=True)


def lock():
    protocol = verify()
    selected = choices()
    assert all(read(ROOT / f'seed{s}_finished.json')['complete'] for s in SEEDS)
    hashes, records, orders = {}, [], {}
    checked = 0
    for out in sorted(ROOT.glob('*_s*')):
        if not out.is_dir():
            continue
        record, history = read(out / 'result.json'), read(out / 'history.json')
        assert record['epochs'] == 8 and record['updates'] == 1896 and record['examples'] == 59904
        assert record['actual_adam_steps'] == [1896] and record['selected_reload_exact']
        seed = record['seed']
        if seed in orders:
            assert orders[seed] == record['order_hashes']
        orders[seed] = record['order_hashes']
        with np.load(out / 'selection_predictions.npz') as z:
            for mode in MODES:
                scores = []
                for mouse in MICE:
                    meta = read(ref.FAIR / mouse / 'metadata.json')
                    target = np.load(ref.BASE / mouse / 'selection_y.npy')
                    np.testing.assert_array_equal(target, z[mouse + '_target'])
                    values = [ref.mse(p, target, -meta['speed_mean'] / meta['speed_std']) for p in z[f'{mode}_{mouse}']]
                    np.testing.assert_allclose(values, [h['mouse_mse'][mouse] for h in history[mode]], rtol=1e-12, atol=1e-12)
                    scores.append(np.array(values) / protocol['selection_denominators'][mouse])
                    checked += len(values)
                joint = np.mean(scores, axis=0)
                np.testing.assert_allclose(joint, [h['selection_score'] for h in history[mode]], rtol=1e-12, atol=1e-12)
                assert int(np.argmin(joint)) == record['selected_epochs'][mode]
                assert np.isclose(joint.min(), record['selection_scores'][mode], rtol=1e-12, atol=1e-12)
        records.append(record)
        for name in ['result.json', 'history.json', 'selection_predictions.npz'] + [f'{m}_selected.pt' for m in MODES]:
            hashes[str((out / name).relative_to(ROOT))] = digest(out / name)
    assert len(records) == 18 and checked == 18 * 2 * 9 * 4
    write(ROOT / 'selection_lock.json', dict(locked_utc=datetime.now(timezone.utc).isoformat(), choices=selected,
        recipe_sha256=digest(ROOT / 'recipe_lock.json'), records=records, hashes=hashes,
        selection_scores_checked=checked, matched_batch_orders=True, new_later_scored=False))


def verify_lock():
    verify()
    locked = read(ROOT / 'selection_lock.json')
    assert digest(ROOT / 'recipe_lock.json') == locked['recipe_sha256']
    for name, value in locked['hashes'].items():
        assert digest(ROOT / name) == value, name
    return locked


def evaluate():
    locked = verify_lock()
    hashes, count = {}, 0
    for s, mouse in enumerate(MICE):
        meta = read(ref.FAIR / mouse / 'metadata.json')
        columns = np.load(ref.BASE / mouse / 'columns.npy')
        with np.load(ref.FAIR / mouse / 'later_raw.npz') as raw, np.load(ref.FAIR / mouse / 'statistics.npz') as norm:
            seq = ((raw['activity'][columns, 24:] - norm['activity_mean'][columns]) / norm['activity_std'][columns]).T.astype(np.float32)
            y = (raw['speed'][55:] - meta['speed_mean']) / meta['speed_std']
        x = torch.from_numpy(np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(seq, 32, axis=0)))
        original = x.clone()
        pred = dict(target=y)
        with np.load(REPLICATION / mouse / 'later_predictions.npz') as z:
            np.testing.assert_array_equal(y, z['target'])
            for family in FAMILIES:
                recipe = locked['choices'][family]
                for seed in SEEDS:
                    pred[f'native_{family}_s{seed}'] = z[f'{family}_s{seed}']
                    state = torch.load(fit_dir(family, recipe['rate'], seed) / (recipe['mode'] + '_selected.pt'), weights_only=True)
                    net = Model(family, seed, range(4))
                    net.load_state_dict(state)
                    pred[f'tuned_{family}_s{seed}'] = ref.predict(net, x, s)
                    assert all(torch.equal(v, state[k]) for k, v in net.state_dict().items())
                    count += len(y)
        assert torch.equal(original, x)
        out = ROOT / mouse
        out.mkdir()
        path = out / 'later_predictions.npz'
        np.savez_compressed(path, **pred)
        hashes[str(path.relative_to(ROOT))] = digest(path)
        print(mouse, 'scored', flush=True)
    verify_lock()
    write(ROOT / 'evaluation.json', dict(complete=True, prediction_hashes=hashes, new_predictions=count,
        target_alignment_exact=True, input_model_preserved=True, archived_predictions_reused=True,
        losing_recipes_not_scored_on_later=True))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        if sys.argv[1] == 'train':
            for seed in map(int, sys.argv[2:]):
                train(seed)
        else:
            {'check': check, 'freeze': freeze, 'choose': choose, 'lock': lock, 'evaluate': evaluate}[sys.argv[1]]()
