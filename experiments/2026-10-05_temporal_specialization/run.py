"""Fixed temporal-specialization screen with matched routing controls."""
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
SpecializedDecoder = new_models.SpecializedDecoder
MICE = reference.MICE
SEEDS = list(range(10, 16))
VARIANTS = ['time_dynamic', 'time_static', 'interleaved_dynamic']
FREE = EXP / '2026-10-05_multiquery_readout'
PARENTS = new_models.VARIANTS
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
    torch.manual_seed(91520)
    x = torch.randn(3, 128, 32)
    original = x.clone()
    matches = 0
    counts = {}
    for variant in VARIANTS:
        for seed in SEEDS:
            net = SpecializedDecoder(variant, seed, range(4))
            path = FREE / f'{PARENTS[variant]}_s{seed}'
            initial = torch.load(path / 'initial.pt', weights_only=True)
            assert all(torch.equal(net.state_dict()[k], v) for k, v in initial.items())
            state = torch.load(path / 'selected.pt', weights_only=True)
            missing = net.load_state_dict(state, strict=False)
            assert missing.missing_keys == ['allowed'] and not missing.unexpected_keys
            net.mask_enabled = False
            parent = new_models.ReadoutDecoder(PARENTS[variant], seed, range(4))
            parent.load_state_dict(state)
            for session in range(4):
                np.testing.assert_array_equal(reference.predict(net, x, session), reference.predict(parent, x, session))
                matches += 1
        net = SpecializedDecoder(variant, 10, range(4)).eval()
        counts[variant] = sum(p.numel() for p in net.parameters())
        assert net.allowed.shape == (4, 1024)
        assert torch.equal(net.allowed.sum(1), torch.full((4,), 256))
        assert torch.equal(net.allowed.sum(0), torch.ones(1024, dtype=torch.int64))
        mapping = net.allowed.reshape(4, 128, 8)
        expected = [(q, q + 4) if variant == 'interleaved_dynamic' else (2 * q, 2 * q + 1) for q in range(4)]
        for q, times in enumerate(expected):
            for n in range(128):
                assert torch.where(mapping[q, n])[0].tolist() == list(times)
        changed = x.clone()
        changed[0] += 2 * torch.randn_like(changed[0])
        with torch.no_grad():
            pred, weights = net(x, 0, return_weights=True)
            _, other = net(changed, 0, return_weights=True)
            _, uniform = net(x, 0, uniform=True, return_weights=True)
        assert torch.equal(pred, torch.zeros(3))
        forbidden = (~net.allowed)[None, None].expand_as(weights)
        assert torch.count_nonzero(weights[forbidden]) == 0
        assert torch.count_nonzero(uniform[forbidden]) == 0
        torch.testing.assert_close(weights.sum(-1), torch.ones(3, 2, 4))
        assert torch.equal(uniform[~forbidden], torch.full_like(uniform[~forbidden], 1 / 256))
        assert torch.equal(weights[1:], other[1:])
        if variant == 'time_static':
            assert torch.equal(weights, other)
        else:
            assert not torch.allclose(weights[0], other[0], rtol=1e-5, atol=1e-7)
        if variant != 'interleaved_dynamic':
            captured = []
            hook = net.head.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
            late = x.clone()
            late[:, :, 28:] += 3
            with torch.no_grad():
                net(x, 0)
                net(late, 0)
            hook.remove()
            assert torch.equal(captured[0][:, :48], captured[1][:, :48])
        opt = torch.optim.AdamW(net.parameters(), lr=.001)
        for step in range(3):
            net.train()
            opt.zero_grad(set_to_none=True)
            loss = (net(x, 0) - torch.arange(3, dtype=torch.float32)).square().mean()
            loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
            opt.step()
        assert bool(torch.all(net.behavior_query.grad.norm(dim=1) > 0))
        assert net.temporal.mix.in_proj_weight.grad.norm() > 0
        clone = SpecializedDecoder(variant, 10, range(4))
        clone.load_state_dict(net.state_dict())
        np.testing.assert_array_equal(reference.predict(net, x, 0), reference.predict(clone, x, 0))
        assert torch.equal(x, original)
    states = [SpecializedDecoder(v, 10, range(4)).state_dict() for v in VARIANTS]
    assert all(torch.equal(value, s[k]) for k, value in states[0].items() if k != 'allowed' for s in states[1:])
    assert all(n == 21553 for n in counts.values())
    write(ROOT / 'selfcheck.json', dict(passed=True, parameter_counts=counts, mask_disabled_trained_parent_equivalences=matches,
        all_parent_initial_tensors_exact=True, all_new_variant_parameter_initializations_exact=True,
        mask_neuron_major_mapping_exact=True, each_query_256_tokens_each_token_once=True,
        forbidden_weights_exact_zero=True, uniform_weights_respect_mask=True,
        no_cross_example_mixing=True, later_input_does_not_change_earlier_query_features=True,
        dynamic_static_routing_behaviors_correct=True, gradients_to_all_four_queries_and_temporal=True,
        reload_exact=True, input_preserved=True, zero_initial_predictions=True))


def freeze():
    assert read(ROOT / 'selfcheck.json')['passed']
    files = [ROOT / n for n in ['models.py', 'run.py', 'analyse.py', 'diagnose.py', 'selfcheck.json']]
    files += [SHARED / n for n in ['models.py', 'run.py', 'protocol.json']]
    files += [reference.BASE / 'models.py', FREE / 'models.py']
    files += [PROJECT / n for n in ['train.py', 'model.py', 'data.py']]
    for seed in SEEDS:
        files += [parent_dir(seed) / n for n in ['initial.pt', 'selected.pt', 'history.json', 'result.json', 'selection_predictions.npz']]
        for group in ['dynamic', 'static']:
            files += [FREE / f'{group}_s{seed}' / n for n in ['initial.pt', 'selected.pt', 'history.json', 'result.json', 'selection_predictions.npz']]
    for mouse in MICE:
        files += [reference.BASE / mouse / n for n in ['train_x.npy', 'train_y.npy', 'selection_x.npy', 'selection_y.npy', 'columns.npy']]
        files += [reference.FAIR / mouse / n for n in ['later_raw.npz', 'statistics.npz', 'metadata.json']]
        files += [REPLICATION / mouse / 'later_predictions.npz', FREE / mouse / 'later_predictions.npz']
    files += [FREE / 'readout_diagnostic.json', FREE / 'results.json']
    write(ROOT / 'protocol.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        question='Does giving four readout queries explicit temporal roles improve later running-speed decoding beyond the original transformer and unconstrained four-query models, and does contiguous timing beat a balanced interleaved restriction?',
        authorization='User explicitly accepted the proposed specialization hypothesis with okay continue after the completed unrestricted multiquery study. New fixed question; no old study or source is modified.',
        variants='time_dynamic and time_static:query0 reads patches0/1,query1 reads2/3,query2 reads4/5,query3 reads6/7,each across128neurons. interleaved_dynamic control reads0/4,1/5,2/6,3/7. Four queries each access256tokens;each of1024tokens belongs toone query. Dynamic keys useactivity;static keys useID/time/session;both values useactivity.',
        architecture='Exact archived four-query decoder except fixed pre-softmax readout mask. All three have21553parameters;all tensors match archived unrestricted initial tensors,including resizedhead andfourqueries. Retain causal temporal encoder,32bins,128neurons,64populationstatistics and128->64->1head. Same dense operator shapes anddropout draws. No extra loss,newinputs,coordinates,pretraining orper-sessionroutingselection.',
        masking_limit='Disjoint readout source positions,not disjoint raw histories:causal temporal states can summarize earlier patches. Latest branch canindirectly containolderactivity;statistics shortcut still seesfullhistory. Maskchanges support andsoftmax normalization. Interleaved control distinguishes one contiguous assignment fromone equallysized noncontiguous assignment;not allspecialization designs.',
        budget='Stage1:9newfits,threevariants xseeds10/11/12. Each24epochs,5688updates,179712presentations. Reuse alloriginalandunrestrictedfits. Ifeither contiguous candidate passes fixed earlier utilityscreen,trainALLthreevariants on13/14/15 (9more,max18). Otherwise stoptraining. No alternate partitions,querycounts,regularizers,losses orseeds afteroutcomes.',
        training='Original shared batches,lossweights,AdamWlr.001/wd.01,batch32,clip1,cosine24 eta_min.0001,training-only normalization. Fresh fromexact archivedunrestrictedinitialization,notfinetunedcheckpoint. Everyvariant samebudget;equalupdates andparameters arenotuniversal optimality claims.',
        selection='One joint selected epoch0..24 using original equal-mouse bounded validationMSE/fixedridge denominators. Earlier screenreusesselectiondata andisoptimistic,notanindependentholdout. Allfinalchoicesandscreendecisionlock beforecurrentlaterinference.',
        selection_denominators=read(SHARED / 'protocol.json')['selection_denominators'],
        screen='For EACH contiguous candidate,beat BOTH originalone-querybaseline and itsmatchingunrestrictedfour-queryparent by>=5%equal-mouse meanpairMSEgain,>=3/4mice and>=8/12singlepairedseedwins. Also no mouse>10%harmvsoriginalbaseline. Promoteallthreevariants iff eithercontiguouscandidate passes. Interleavedcontrol cannottriggerreplication.',
        primary='time_dynamic_full requiresownscreenpass thenlater>=5%meanpairMSEgain,>=3mice,>=ceil(2/3*4*nseeds)singlepairedseedwins against baseline,free_dynamic,interleaved_dynamic ANDtime_static,plusnomouse>10%harmvsbaseline. Separate utilitygate againstbaseline/free_dynamic. time_staticsecondaryrequiresownscreenpass andsameutilitythresholdsagainstbaseline/free_static plusbaselineharmguard. No interleavedstatic arm,so staticutilitydoesnotestablishcontiguity-specificbenefit.',
        aggregation='Boundeachindividualoutputatphysicalzero beforepairaveraging. Averageall3or15distincttwo-seedpairerrorswithinmouse,thenrelativechanges equallyover4mice. Sameactiveseedsubset for allcomparators. ReportMSE,MAE,singles,pairs,rawdenominators,leave-one-out,initial/mediancontrols;extraseeds13-15separately ifexpanded. Pairs/seeds/windowsnotindependentanimals.',
        diagnostic='Afterlock,onfirst64earlierselectionwindows/mouse:measurecenteredqueryfeaturecosines andparticipation-ratioeffective rank;compare savedmatchingunrestricteddiagnosticrowswithoutrepeatingoldinference. Verifyzero forbiddenweights. Disjoint-weightTV=1isforcedbymaskandNOTevidenceofusefulspecialization. Comparefeatures,notjustweightdiversity. Descriptive,nocausalattributionorselection.',
        limits='FourhistoricallyreusedStringermice;newseedsarenotnewanimals. No newpvalues,independentsignificance,biologicalconnectivity,unseenmousetransferorgenerationclaims. Redundancymaypersistbecausecausalstatesandpopulationstatisticscarrysharedhistory.',
        stop='Finishscreen,conditionalreplication,lockedlaterscoring,diagnostic,analysis,reviewandreport. Keeporiginalbaselineunlesspresetgatespass. No architecturegridextension,mainapplicationmigration,publication,generation orstoppedreconstructionresumption.',
        hashes={str(p.relative_to(PROJECT)): digest(p) for p in files}))


def score_summary(rows, contrast):
    values = [r['contrasts'][contrast] for r in rows]
    gains = [v['gain'] for v in values]
    summary = dict(mean_mse_gain=float(np.mean(gains)), mouse_gains=gains,
        mouse_wins=sum(g > 0 for g in gains), seed_wins=sum(v['seed_wins'] for v in values))
    summary['contrast_passed'] = summary['mean_mse_gain'] >= .05 and summary['mouse_wins'] >= 3 and summary['seed_wins'] >= 8
    return summary


def screen():
    verify()
    hashes, rows = {}, []
    for seed in SCREEN_SEEDS:
        assert read(ROOT / f'seed{seed}_finished.json')['complete']
        for variant in VARIANTS:
            out = ROOT / f'{variant}_s{seed}'
            record = read(out / 'result.json')
            assert record['updates'] == 5688 and record['examples'] == 179712 and record['actual_adam_steps'] == [5688]
            for name in ['selected.pt', 'history.json', 'result.json', 'selection_predictions.npz']:
                hashes[str((out / name).relative_to(ROOT))] = digest(out / name)
    groups = ['baseline', 'free_dynamic', 'free_static'] + VARIANTS
    contrasts = {f'{variant}_vs_{control}': (variant, control) for variant in ['time_dynamic', 'time_static']
                 for control in ['baseline', 'free_' + PARENTS[variant]]}
    for mouse in MICE:
        meta = read(reference.FAIR / mouse / 'metadata.json')
        lower = -meta['speed_mean'] / meta['speed_std']
        scores = {}
        for group in groups:
            predictions = []
            for seed in SCREEN_SEEDS:
                path = parent_dir(seed) if group == 'baseline' else FREE / f'{group[5:]}_s{seed}' if group.startswith('free_') else ROOT / f'{group}_s{seed}'
                pred, y = selected_validation(path, mouse)
                predictions.append(np.maximum(pred.astype(np.float64), lower))
            singles = [float(np.mean((p - y) ** 2)) for p in predictions]
            pairs = [float(np.mean(((predictions[i] + predictions[j]) / 2 - y) ** 2)) for i, j in [(0, 1), (0, 2), (1, 2)]]
            scores[group] = dict(mse=float(np.mean(pairs)), single_mse=singles, pair_mse=pairs)
        effects = {name: dict(gain=1 - scores[a]['mse'] / scores[b]['mse'],
            seed_wins=int(np.sum(np.array(scores[a]['single_mse']) < scores[b]['single_mse']))) for name, (a, b) in contrasts.items()}
        rows.append(dict(mouse=mouse, scores=scores, contrasts=effects))
    summaries = {name: score_summary(rows, name) for name in contrasts}
    passed = {v: summaries[v + '_vs_baseline']['contrast_passed'] and summaries[v + '_vs_free_' + PARENTS[v]]['contrast_passed'] and
              min(summaries[v + '_vs_baseline']['mouse_gains']) >= -.1 for v in ['time_dynamic', 'time_static']}
    write(ROOT / 'screen.json', dict(created_utc=datetime.now(timezone.utc).isoformat(), rows=rows, summaries=summaries,
        candidate_passed=passed, promote=any(passed.values()), locked_stage1_artifacts=hashes, current_later_predictions_created=False))
    print(json.dumps(dict(summaries=summaries, candidate_passed=passed, promote=any(passed.values())), indent=2), flush=True)


def selected_validation(path, mouse):
    record = read(path / 'result.json')
    history = read(path / 'history.json')
    assert int(np.argmin([h['selection_score'] for h in history])) == record['selected_epoch']
    with np.load(path / 'selection_predictions.npz') as z:
        y = z[mouse + '_target']
        np.testing.assert_array_equal(y, np.load(reference.BASE / mouse / 'selection_y.npy'))
        return z[mouse + '_predictions'][record['selected_epoch']], y


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
        net = SpecializedDecoder(variant, seed, sessions)
        initial = torch.load(FREE / f'{PARENTS[variant]}_s{seed}' / 'initial.pt', weights_only=True)
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
            selected_reload_exact=True, all_parent_initial_tensors_exact=True, initial_predictions_zero=True,
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
    assert checked == len(seeds) * len(VARIANTS) * 25 * 4
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
                with np.load(FREE / mouse / 'later_predictions.npz') as free:
                    np.testing.assert_array_equal(y, free['target'])
                    for group in ['dynamic', 'static']:
                        predictions[f'free_{group}_s{seed}'] = free[f'{group}_s{seed}']
                for variant in VARIANTS:
                    net = SpecializedDecoder(variant, seed, range(4))
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
