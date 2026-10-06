"""Fixed multiquery readout screen with conditional seed replication."""
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


new_models = module('multiquery_models', ROOT / 'models.py')
sys.path.insert(0, str(SHARED))
reference = module('multiquery_reference', SHARED / 'run.py')
ReadoutDecoder = new_models.ReadoutDecoder
MICE = reference.MICE
SEEDS = list(range(10, 16))
VARIANTS = ['dynamic', 'static']
SCREEN_SEEDS = [10, 11, 12]
COMPONENTS = EXP / '2026-10-04_attention_components'


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
    torch.manual_seed(81250)
    x = torch.randn(3, 128, 32)
    original = x.clone()
    matches = 0
    counts = {}
    for variant in VARIANTS:
        for seed in SEEDS:
            net = ReadoutDecoder(variant, seed, range(4))
            initial = torch.load(parent_dir(seed) / 'initial.pt', weights_only=True)
            for k, v in initial.items():
                if k != 'behavior_query' and not k.startswith('head.'):
                    assert torch.equal(net.state_dict()[k], v), k
            assert torch.equal(net.behavior_query[:1], initial['behavior_query'])
            if variant == 'static' and seed > 12:
                continue
            one = ReadoutDecoder(variant, seed, range(4), queries=1)
            path = parent_dir(seed) if variant == 'dynamic' else COMPONENTS / f'as_s{seed}'
            state = torch.load(path / 'selected.pt', weights_only=True)
            one.load_state_dict(state)
            base = reference.BehaviorDecoder('attention', seed, range(4))
            base.kind = one.kind
            base.load_state_dict(state)
            for session in range(4):
                np.testing.assert_array_equal(reference.predict(one, x, session), reference.predict(base, x, session))
                matches += 1
        net = ReadoutDecoder(variant, 10, range(4)).eval()
        counts[variant] = sum(p.numel() for p in net.parameters())
        changed = x.clone()
        changed[0] += torch.randn_like(changed[0]) * 2
        with torch.no_grad():
            p, a = net(x, 0, return_weights=True)
            _, b = net(changed, 0, return_weights=True)
        assert a.shape == (3, 2, 4, 1024) and torch.equal(p, torch.zeros(3))
        torch.testing.assert_close(a.sum(-1), torch.ones(3, 2, 4))
        assert torch.equal(a[1:], b[1:])
        if variant == 'static':
            assert torch.equal(a, b)
        else:
            assert not torch.allclose(a[0], b[0], atol=1e-7, rtol=1e-5)
        assert not torch.allclose(a[:, :, 0], a[:, :, 1], atol=1e-7, rtol=1e-5)
        opt = torch.optim.AdamW(net.parameters(), lr=.001)
        for step in range(3):
            net.train()
            opt.zero_grad(set_to_none=True)
            loss = (net(x, 0) - torch.arange(3, dtype=torch.float32)).square().mean()
            loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
            opt.step()
        assert bool(torch.all(net.behavior_query.grad.norm(dim=1) > 0))
        assert net.key_projection.weight.grad.norm() > 0
        assert net.temporal.mix.in_proj_weight.grad.norm() > 0
        clone = ReadoutDecoder(variant, 10, range(4))
        clone.load_state_dict(net.state_dict())
        np.testing.assert_array_equal(reference.predict(net, x, 0), reference.predict(clone, x, 0))
        with torch.no_grad():
            torch.testing.assert_close(net(x, 0)[1:], net(changed, 0)[1:], rtol=0, atol=0)
        assert torch.equal(x, original)
    a, b = [ReadoutDecoder(v, 10, range(4)).state_dict() for v in VARIANTS]
    assert all(torch.equal(v, b[k]) for k, v in a.items())
    assert counts['dynamic'] == counts['static'] == 21553
    write(ROOT / 'selfcheck.json', dict(passed=True, parameter_counts=counts,
        trained_single_query_parent_equivalences=matches, common_body_and_first_query_initials_exact=True,
        both_variants_all_initial_tensors_exact=True, multiquery_shapes_and_weight_sums=True,
        four_distinct_initial_queries=True, static_weights_input_invariant=True,
        dynamic_weights_input_dependent=True, no_cross_example_mixing=True,
        all_four_queries_and_temporal_gradients_nonzero=True, reload_exact=True, input_preserved=True,
        initial_predictions_zero=True))


def freeze():
    assert read(ROOT / 'selfcheck.json')['passed']
    files = [ROOT / n for n in ['models.py', 'run.py', 'analyse.py', 'selfcheck.json']]
    files += [SHARED / n for n in ['models.py', 'run.py', 'protocol.json']]
    files += [reference.BASE / 'models.py'] + [PROJECT / n for n in ['train.py', 'model.py', 'data.py']]
    for seed in SEEDS:
        files += [parent_dir(seed) / n for n in ['initial.pt', 'selected.pt', 'history.json', 'result.json', 'selection_predictions.npz']]
    for mouse in MICE:
        files += [reference.BASE / mouse / n for n in ['train_x.npy', 'train_y.npy', 'selection_x.npy', 'selection_y.npy', 'columns.npy']]
        files += [reference.FAIR / mouse / n for n in ['later_raw.npz', 'statistics.npz', 'metadata.json']]
        files += [REPLICATION / mouse / 'later_predictions.npz']
    write(ROOT / 'protocol.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        question='Does preserving four learned neuron-time summaries instead of one improve running-speed decoding on the shared temporal transformer, and does activity-dependent readout routing beat an identical-capacity static-routing control?',
        hypothesis='The original behavior query compresses1024neuron-time tokens to16numbers plus64population statistics. Four distinct queries retain64neural features and expose them jointly to the head. This is an unverified readout bottleneck hypothesis;old frozen-encoder flattening tests do not test this shared end-to-end recipe.',
        architecture='Retain128neurons x32bins,8four-bin patches,width16,original causal temporal attention,IDs/session embeddings. Replace one learned query with four. Each independently reads all1024tokens through shared two-head projections and query FF. Concatenate four16-vectors,append the unchanged64mean/std features,and use LayerNorm128,Linear128->64,GELU,Linear64->1. No extra population block or query-query attention.',
        control='Dynamic uses activity-dependent keys;static uses learned ID/time/session keys. Both use activity-dependent values and exactly equal21553parameters/all initial tensors. Both retain temporal attention,so static is not allMLP. Original one-query transformer18337parameters is reused. Wider head and changed normalization accompany query count;this is a readout-recipe comparison,not an isolated query-count effect or exhaustive MLP search.',
        initialization='All inherited nonhead body tensors and first query equal archived parent initial tensors. Three new query vectors and resized head use fixedseed+300 initialization. Both variants have exactly identical tensors and initialzero outputs. Different query count changes dropout draws vsparent;dropout probabilities and update budgets match,not masks/FLOPs.',
        budget='Stage1:6newfits,two variants xseeds10/11/12,24epochs/5688updates/179712presentations each. Reuse archived baseline fits. Screen using earlier selection only. If either variant passes fixed baseline screen,fit BOTH variants for seeds13/14/15 (6more,max12newfits). Otherwise stop training. No count,width,loss,LR,seed or placement extension.',
        training='Original shared batches,lossweights and training-only preprocessing. AdamWlr.001/wd.01,batch32,clip1,cosine24 eta_min.0001. Common initial body,not trained checkpoint continuation. Both arms intact inputs only;no coordinates,masking,pretraining,behavior inputs or application edits.',
        selection='One joint checkpoint perseed/variant,minimum original equal-mouse bounded validation MSE dividedby fixed historicalridge denominators overepochs0..24. Screening uses chosen earlier predictions;optimistic selection,not newindependent validation. Allfinal checkpointchoices andscreen decisionlock beforecurrentlaterinference.',
        selection_denominators=read(SHARED / 'protocol.json')['selection_denominators'],
        screen='For eachvariant vsbaseline onearlierselection:>=5%equal-mouse meanpairMSEgain,>=3/4mousewins,>=8/12singlepairedseedwins,no mouse>10%harm. Promote bothvariants toextra seeds iffEITHERpasses. Finaloveralladoption also requires itsownscreen pass;screenfailure cannotberescuedbylateroutcomes.',
        primary='Dynamic fullgate requires ownscreenpass AND later>=5%meanpairMSEgain,>=3/4mice,>=ceil(2/3*4*nseeds)singlepairedseedwins versusBOTHbaselineandstatic,andno mouse>10%harmvsbaseline. Staticsecondarygate requires ownscreenpass andsamebaseline thresholds/harmguard. Ifpromoted,reportseeds13-15 separately as training-seed replication,notnewanimals.',
        aggregation='Physicalzero boundeachsingleoutput beforepairaveraging. AverageMSE overall distincttwo-seedpairs withinmouse,thenrelativechanges equallyover4mice. Compareexact same seedsubsetbetweenarms. Stage1uses3pairs;expandeduses15. Singles,MAE,rawdenominators,leave-one-out,trainingmean/medianandarchivedMLPcontextreported. No bestpairselection.',
        scope='Four historically reused Stringer recordings;seeds,pairs andoverlappingwindows arenotindependentanimals. No independent significance claim or pvalues. Data-driven screen andhistoricalsearch remain. Newstudytests different readout fromfailedpopulationblock;do not reopenstoppedreconstruction orfailedgrids.',
        stop='Complete authorizedscreen,conditionalreplication,lock,laterdiagnostic/confirmation,analysis,audit,report,handoff. Retainoriginalbaselineunlessfullpresetgatepasses. No publication,generation ormainapplicationmigration.',
        sources=[dict(title='Set Transformer, section3.2 learned pooling seed vectors', url='https://proceedings.mlr.press/v97/lee19d/lee19d.pdf', use='Architecture inspiration only;not a claim that this is a full Set Transformer implementation or that multiple summaries improve this dataset.')],
        hashes={str(p.relative_to(PROJECT)): digest(p) for p in files}))


def selected_validation(path, mouse):
    record = read(path / 'result.json')
    history = read(path / 'history.json')
    assert int(np.argmin([h['selection_score'] for h in history])) == record['selected_epoch']
    with np.load(path / 'selection_predictions.npz') as z:
        y = z[mouse + '_target']
        np.testing.assert_array_equal(y, np.load(reference.BASE / mouse / 'selection_y.npy'))
        return z[mouse + '_predictions'][record['selected_epoch']], y


def screen():
    verify()
    rows = []
    hashes = {}
    for seed in SCREEN_SEEDS:
        assert read(ROOT / f'seed{seed}_finished.json')['complete']
        for variant in VARIANTS:
            out = ROOT / f'{variant}_s{seed}'
            r = read(out / 'result.json')
            assert r['updates'] == 5688 and r['examples'] == 179712 and r['actual_adam_steps'] == [5688]
            for name in ['selected.pt', 'history.json', 'result.json', 'selection_predictions.npz']:
                hashes[str((out / name).relative_to(ROOT))] = digest(out / name)
    for mouse in MICE:
        meta = read(reference.FAIR / mouse / 'metadata.json')
        lower = -meta['speed_mean'] / meta['speed_std']
        scores = {}
        for group in ['baseline'] + VARIANTS:
            predictions = []
            for seed in SCREEN_SEEDS:
                path = parent_dir(seed) if group == 'baseline' else ROOT / f'{group}_s{seed}'
                p, y = selected_validation(path, mouse)
                predictions.append(np.maximum(p.astype(np.float64), lower))
            singles = np.array([np.mean((p - y) ** 2) for p in predictions])
            pair_mse = [float(np.mean(((predictions[i] + predictions[j]) / 2 - y) ** 2)) for i, j in [(0, 1), (0, 2), (1, 2)]]
            scores[group] = dict(mse=float(np.mean(pair_mse)), single_mse=singles.tolist(), pair_mse=pair_mse)
        contrasts = {}
        for variant in VARIANTS:
            a, b = scores[variant], scores['baseline']
            contrasts[variant] = dict(gain=1 - a['mse'] / b['mse'], seed_wins=int(np.sum(np.array(a['single_mse']) < b['single_mse'])))
        rows.append(dict(mouse=mouse, scores=scores, contrasts=contrasts))
    summaries = {}
    for variant in VARIANTS:
        gains = [r['contrasts'][variant]['gain'] for r in rows]
        s = dict(mean_mse_gain=float(np.mean(gains)), mouse_gains=gains, mouse_wins=sum(g > 0 for g in gains),
                 seed_wins=sum(r['contrasts'][variant]['seed_wins'] for r in rows))
        s['passed'] = s['mean_mse_gain'] >= .05 and s['mouse_wins'] >= 3 and s['seed_wins'] >= 8 and min(gains) >= -.1
        summaries[variant] = s
    write(ROOT / 'screen.json', dict(created_utc=datetime.now(timezone.utc).isoformat(), rows=rows, summaries=summaries,
        promote=any(s['passed'] for s in summaries.values()), locked_stage1_artifacts=hashes, current_later_predictions_created=False))
    print(json.dumps(summaries, indent=2), flush=True)


def active_seeds():
    s = read(ROOT / 'screen.json')
    for name, value in s['locked_stage1_artifacts'].items():
        assert digest(ROOT / name) == value
    return SEEDS if s['promote'] else SCREEN_SEEDS


def verify():
    protocol = read(ROOT / 'protocol.json')
    for name, value in protocol['hashes'].items():
        assert digest(PROJECT / name) == value, name
    return protocol


def train(seed):
    assert seed in SEEDS
    protocol = verify()
    if seed not in SCREEN_SEEDS:
        assert read(ROOT / 'screen.json')['promote'], 'screen did not authorize replication'
    data = reference.load_data()
    sessions = list(range(4))
    lengths = [len(data[s]['x']) for s in sessions]
    total = sum(lengths)
    parent_history = read(parent_dir(seed) / 'history.json')
    for variant in VARIANTS:
        out = ROOT / f'{variant}_s{seed}'
        out.mkdir()
        net = ReadoutDecoder(variant, seed, sessions)
        initial = torch.load(parent_dir(seed) / 'initial.pt', weights_only=True)
        for k, v in initial.items():
            if k != 'behavior_query' and not k.startswith('head.'):
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
        max_grad = query_grad = temporal_grad = 0.
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
                query_grad = max(query_grad, float(net.behavior_query.grad.norm()))
                temporal_grad = max(temporal_grad, float(net.temporal.mix.in_proj_weight.grad.norm()))
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
        assert updates == 5688 and examples == 179712 and actual_steps == [5688] and query_grad > 0 and temporal_grad > 0
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
            selected_reload_exact=True, common_body_initial_tensors_exact=True, initial_predictions_zero=True,
            max_gradient=max_grad, query_gradient_max=query_grad, temporal_gradient_max=temporal_grad, elapsed_seconds=time.monotonic() - start))
        print(seed, variant, 'complete', flush=True)
    write(ROOT / f'seed{seed}_finished.json', dict(complete=True))


def lock():
    protocol = verify()
    seeds = active_seeds()
    records, hashes, orders = [], {}, {}
    checked = 0
    for seed in seeds:
        assert read(ROOT / f'seed{seed}_finished.json')['complete']
        for variant in VARIANTS:
            out = ROOT / f'{variant}_s{seed}'
            record, history = read(out / 'result.json'), read(out / 'history.json')
            assert len(history) == 25 and record['updates'] == 5688 and record['examples'] == 179712
            assert record['actual_adam_steps'] == [5688] and record['query_gradient_max'] > 0 and record['temporal_gradient_max'] > 0
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
    assert checked == len(seeds) * 2 * 25 * 4
    write(ROOT / 'selection_lock.json', dict(locked_utc=datetime.now(timezone.utc).isoformat(), records=records, seeds=seeds, screen_sha256=digest(ROOT / 'screen.json'),
        hashes=hashes, selection_scores_checked=checked, batches_match_native=True, new_later_scored=False))


def verify_lock():
    verify()
    locked = read(ROOT / 'selection_lock.json')
    assert digest(ROOT / 'screen.json') == locked['screen_sha256']
    for name, value in locked['hashes'].items():
        assert digest(ROOT / name) == value, name
    return locked


def evaluate():
    locked = verify_lock()
    seeds = locked['seeds']
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
            for seed in seeds:
                predictions[f'baseline_s{seed}'] = saved[f'attention_s{seed}']
                predictions[f'prior_mlp_s{seed}'] = saved[f'mlp_s{seed}']
                parent = reference.BehaviorDecoder('attention', seed, range(4))
                parent.load_state_dict(torch.load(parent_dir(seed) / 'selected.pt', weights_only=True))
                np.testing.assert_array_equal(reference.predict(parent, x[:64], session), saved[f'attention_s{seed}'][:64])
                for variant in VARIANTS:
                    net = ReadoutDecoder(variant, seed, range(4))
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
        exact_baseline_firstbatch_checks=4 * len(seeds), exact_target_alignment=True, model_and_input_unchanged=True))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        if sys.argv[1] == 'train':
            for seed in map(int, sys.argv[2:]):
                train(seed)
        else:
            {'check': check, 'freeze': freeze, 'screen': screen, 'lock': lock, 'evaluate': evaluate}[sys.argv[1]]()
