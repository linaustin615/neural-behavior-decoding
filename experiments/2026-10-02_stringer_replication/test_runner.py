"""Synthetic regression checks; never read or fit the Stringer recordings."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import savemat
import torch
from threadpoolctl import threadpool_limits

import run as r


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = r.verify_protocol()
        torch.set_num_threads(2)
        cls.limiter = threadpool_limits(limits=2)

    @classmethod
    def tearDownClass(cls):
        cls.limiter.restore_original_limits()

    def test_mat_loader_and_triples(self):
        with tempfile.TemporaryDirectory(prefix='stringer-loader-') as directory:
            root = Path(directory)
            raw = np.arange(2*20, dtype=np.float32).reshape(2, 20)
            speed = np.arange(20, dtype=np.float64)
            raw[0, 0] = -1e-6
            savemat(root/'synthetic.mat', dict(Fsp=raw, beh=dict(runSpeed=speed), med=np.zeros((2, 3))))
            record = dict(file='synthetic.mat', native_samples=20, binned_samples=6, splits={'selection':[0, 4]})
            with patch.object(r, 'PROJECT', root):
                a, y, positions = r.load_recording(record)
                all_a, all_y, _ = r.load_recording(record, evaluation=True)
            expected = np.stack([raw[:, i:i+3].mean(1) for i in range(0, 18, 3)], axis=1)
            np.testing.assert_array_equal(a, expected[:, :4])
            np.testing.assert_array_equal(all_a, expected)
            np.testing.assert_array_equal(y, [1, 4, 7, 10])
            np.testing.assert_array_equal(all_y, [1, 4, 7, 10, 13, 16])
            self.assertEqual(positions.shape, (2, 3))
            for tail in (0, 1, 2):
                aa, yy = r.bin_triples(raw[:, :18+tail], speed[:18+tail])
                np.testing.assert_array_equal(aa, expected)
                np.testing.assert_array_equal(yy, all_y)
            negative, _ = r.bin_triples(np.full((2, 6), -1e-6), np.arange(6))
            self.assertTrue((negative < 0).all())

    def test_training_only_normalization_and_lazy_windows(self):
        rng = np.random.default_rng(17)
        a, y = rng.normal(size=(70, 250)).astype(np.float32), rng.uniform(size=250)
        a[0, :100] = 1
        stats = r.normalize_training(a, y, 100, count=64)
        changed, changed_y = a.copy(), y.copy()
        changed[:, 100:], changed_y[100:] = 1e10, -1e10
        later = r.normalize_training(changed, changed_y, 100, count=64)
        for key in stats:
            np.testing.assert_array_equal(stats[key], later[key])
        self.assertNotIn(0, stats['ids'])
        selected = a[stats['ids']]
        for start, stop in [(0, 100), (120, 175), (195, 250)]:
            windows = r.Windows(selected, y, start, stop, stats)
            self.assertEqual(len(windows), stop-start-31)
            for i in (0, len(windows)-1):
                x, target, index = windows[i]
                t = start+i+31
                expected = ((selected[:, t-7:t+1]-stats['activity_mean'])/stats['activity_std']).astype(np.float32)
                np.testing.assert_array_equal(x.numpy(), expected)
                self.assertEqual(target.item(), np.float32((y[t]-stats['speed_mean'])/stats['speed_std']))
                self.assertEqual(index, i)
            with self.assertRaises(IndexError):
                windows[-1]
        with self.assertRaises(RuntimeError):
            r.normalize_training(a, np.zeros(250), 100, count=64)
        with self.assertRaises(RuntimeError):
            r.normalize_training(np.ones_like(a), y, 100, count=64)
        with self.assertRaises(RuntimeError):
            r.Windows(selected, y, 0, 31, stats)

    def test_metrics_and_mouse_weighting(self):
        metrics = r.score([-3, 1], [0, 2], dict(speed_mean=2, speed_std=2))
        self.assertEqual(metrics['mse'], 1)
        self.assertEqual(metrics['raw_mse'], 5)
        self.assertEqual(metrics['rmse_speed_units'], 2)
        self.assertEqual(r.score([0, 1], [1, 1], dict(speed_mean=0, speed_std=1))['r2'], None)
        rows = []
        for mouse, f, c in zip(['A', 'B', 'C', 'D'], [8, 90, 450, 2000], [10, 100, 500, 2000]):
            for seed in (10, 11):
                for condition in ('functional', 'random', 'global'):
                    rows.append(dict(task=dict(mouse=mouse, seed=seed, condition=condition),
                        evaluation=dict(mse=f if condition == 'functional' else c),
                        untrained_evaluation=dict(mse=10000), constant_evaluation=dict(mse=10000)))
        summary = r.summarize(rows)
        self.assertAlmostEqual(summary['contrasts']['global']['mean_benefit'], .1)
        self.assertFalse(summary['practical_gate_passed'])
        self.assertEqual(r.sign_test([1, 2, 3, 4]), .125)
        self.assertEqual(r.sign_test([0, 0, 0, 0]), 1)
        self.assertEqual(r.sign_test([1, -1, 0, 0]), 1)
        for row in rows:
            if row['task']['condition'] == 'functional':
                row['evaluation']['mse'] = 1
        summary = r.summarize(rows)
        self.assertTrue(summary['practical_gate_passed'])
        self.assertEqual(summary['contrasts']['global']['holm_p'], .25)
        rows[0]['untrained_evaluation']['mse'] = .5
        self.assertFalse(r.summarize(rows)['practical_gate_passed'])

    def test_prepare_uses_only_training_for_groups(self):
        with tempfile.TemporaryDirectory(prefix='stringer-prepare-') as directory, redirect_stdout(io.StringIO()):
            work = Path(directory)
            rng = np.random.default_rng(912)
            activity = rng.normal(size=(2050, 81)).astype(np.float32)
            speed = rng.uniform(size=81)
            positions = np.repeat(np.arange(2050)[:, None], 3, axis=1)
            record = deepcopy(self.protocol['cohort'][0])
            record['splits'] = dict(train=[0, 36], selection=[45, 81], evaluation=[90, 126])
            record['examples'] = dict(train=5, selection=5, evaluation=5)
            p = dict(self.protocol, cohort=[record])
            groups = np.repeat(np.arange(8), 256)
            stats = r.normalize_training(activity, speed, 36)
            with patch.object(r, 'load_recording', return_value=(activity, speed, positions)), \
                 patch.object(r.reference(), 'functional_groups', return_value=(groups, {})) as grouping:
                r.prepare(p, work)
            np.testing.assert_array_equal(grouping.call_args.args[0], activity[stats['ids'], :36])
            self.assertEqual(grouping.call_args.args[1], 101)
            r.verify_prepared(work)
            with np.load(work/(record['mouse']+'.npz')) as saved:
                np.testing.assert_array_equal(saved['positions'], positions[saved['ids']])
                np.testing.assert_array_equal(saved['activity'], activity[saved['ids']])
                self.assertEqual(saved['activity'].shape[1], 81)
            with patch.object(r, 'load_recording') as load:
                r.prepare(p, work)
                load.assert_not_called()
            with patch.object(r, 'provenance', return_value={}):
                with self.assertRaisesRegex(RuntimeError, 'changed after preparation'):
                    r.verify_prepared(work)

    def test_checkpoint_pipeline_and_evaluation_guards(self):
        #six tiny synthetic fits exercise both seeds and all masks; cloned records only test orchestration
        with tempfile.TemporaryDirectory(prefix='stringer-pipeline-') as directory, redirect_stdout(io.StringIO()):
            work = Path(directory)
            (work/'runs').mkdir()
            p = deepcopy(self.protocol)
            splits = dict(train=[0, 36], selection=[45, 81], evaluation=[90, 126])
            rng = np.random.default_rng(312)
            activity = rng.normal(size=(2050, 126)).astype(np.float32)
            speed = rng.uniform(0, 3, size=126)
            stats = r.normalize_training(activity, speed, 36)
            selected = activity[stats['ids']]
            groups = np.repeat(np.arange(8), 256)
            arrays = dict(stats, activity=selected[:, :81], speed=speed[:81], positions=np.zeros((2048, 3)),
                          functional=groups, random=groups[rng.permutation(2048)])
            r.check_models(arrays)
            prepared_files = {}
            for record in p['cohort']:
                record['splits'], record['examples'] = splits, {s: 5 for s in splits}
                file = work/(record['mouse']+'.npz')
                r.save_npz(file, **arrays)
                prepared_files[file.name] = r.digest(file)
            r.write_json(work/'prepared.json', dict(provenance=r.provenance(), files=prepared_files))
            with patch.object(r, 'load_recording') as load:
                with self.assertRaisesRegex(RuntimeError, 'locked checkpoints'):
                    r.evaluate(p, work)
                load.assert_not_called()
            train = r.Windows(selected, speed, *splits['train'], stats)
            selection = r.Windows(selected, speed, *splits['selection'], stats)
            settings = dict(p['training'], epochs=1, min_epochs=1)
            completed = {}
            for task in p['tasks'][:6]:
                path = work/'runs'/(r.stem(task)+'.json')
                result = r.fit_one(task, train, selection, stats, r.groups_for(arrays, task['condition']), path, settings)
                self.assertTrue(result['full_selection_reload'])
                self.assertEqual(len(result['batch_order_sha256']), 1)
                completed[task['seed'], task['condition']] = result
            with self.assertRaisesRegex(RuntimeError, 'Missing completed fit'):
                r.lock_checkpoints(p, work)
            self.assertFalse((work/'checkpoints_locked.json').exists())
            for task in p['tasks'][6:]:
                source = completed[task['seed'], task['condition']]
                cloned = deepcopy(source)
                cloned['task'] = task
                cloned['files'] = {}
                for oldname in source['files']:
                    suffix = Path(oldname).suffix
                    new = work/'runs'/(r.stem(task)+suffix)
                    if suffix == '.pt':
                        ck = torch.load(work/'runs'/oldname, weights_only=True)
                        ck['task'] = task
                        torch.save(ck, new)
                    else:
                        new.write_bytes((work/'runs'/oldname).read_bytes())
                    cloned['files'][new.name] = r.digest(new)
                r.write_json(work/'runs'/(r.stem(task)+'.json'), cloned)
            r.lock_checkpoints(p, work)
            r.verify_lock(p, work)
            with patch.object(r, 'fit_one') as fitting:
                r.fit(p, work)
                fitting.assert_not_called()
            tampered = work/'runs'/(r.stem(p['tasks'][0])+'.pt')
            original = tampered.read_bytes()
            tampered.write_bytes(original+b'x')
            with patch.object(r, 'load_recording') as load:
                with self.assertRaisesRegex(RuntimeError, 'artifact changed'):
                    r.evaluate(p, work)
                load.assert_not_called()
            tampered.write_bytes(original)
            with patch.object(r, 'load_recording', return_value=(activity, speed, np.zeros((2050, 3)))) as load:
                r.evaluate(p, work)
                self.assertEqual(load.call_count, 4)
            results = r.read_json(work/'results.json')
            self.assertEqual(len(results['runs']), 24)
            self.assertTrue((work/'manifest.json').exists())
            self.assertTrue(r.read_json(work/'results_audit.json')['passed'])
            altered = deepcopy(results)
            altered['summary']['contrasts']['global']['mean_benefit'] += .1
            r.write_json(work/'results.json', altered)
            with self.assertRaisesRegex(RuntimeError, 'Equal-mouse mean mismatch'):
                r.audit_saved_results(p, work)
            r.write_json(work/'results.json', results)
            with patch.object(r, 'load_recording') as load:
                with self.assertRaisesRegex(RuntimeError, 'already started'):
                    r.evaluate(p, work)
                load.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, 'Cannot fit'):
                r.fit(p, work)


if __name__ == '__main__':
    unittest.main(verbosity=2)
