"""Small synthetic checks; these do not establish a scientific advantage."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from decoding.config import recipe
from decoding.data import load, prepare, read_array, windows
from decoding.evaluate import evaluate, metrics
from decoding.model import PopulationDecoder
from decoding.ridge import fit as fit_ridge, solve
from decoding.train import fit


class DecodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        rng = np.random.default_rng(17)
        cls.activity = rng.normal(size=(520, 1000)).astype(np.float32)
        cls.running = np.abs(cls.activity[:8].mean(0)+.8)
        cls.raw = cls.root/'raw.npz'
        np.savez(cls.raw, spks=cls.activity, run=cls.running)
        cls.prepared = cls.root/'prepared'
        prepare(cls.raw, cls.prepared, 'TX103', cap=65)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_npz_mapping_and_input_rejection(self):
        np.testing.assert_array_equal(read_array(self.raw, 'spks'), self.activity)
        compressed = self.root/'compressed.npz'
        np.savez_compressed(compressed, spks=self.activity)
        with self.assertRaises(ValueError):
            read_array(compressed, 'spks')
        with self.assertRaises(FileExistsError):
            prepare(self.raw, self.prepared, 'TX103')

    def test_future_does_not_change_training(self):
        changed = self.activity.copy()
        changed[:, 600:] += 900
        run = self.running.copy()
        run[600:] += 800
        path, output = self.root/'future.npz', self.root/'future'
        np.savez(path, spks=changed, run=run)
        first = json.loads((self.prepared/'metadata.json').read_text())
        second = prepare(path, output, 'TX103', cap=65)
        for key in ('speed_mean', 'speed_std', 'lower'):
            self.assertEqual(first[key], second[key])
        for name in ('panel', 'activity_mean', 'activity_std', 'train_seq', 'train_y', 'train_indices'):
            np.testing.assert_array_equal(np.load(self.prepared/(name+'.npy')), np.load(output/(name+'.npy')))

    def test_common_target_and_window_alignment(self):
        for split in ('train', 'validation', 'test'):
            seq = np.load(self.prepared/(split+'_seq.npy'))
            y = np.load(self.prepared/(split+'_y.npy'))
            for history in (16, 32, 64):
                data = load(self.prepared, split, history)
                np.testing.assert_array_equal(data.x[0], seq[64-history:64].T)
                np.testing.assert_array_equal(data.x[-1], seq[-history:].T)
                np.testing.assert_array_equal(data.y, y[63:])
                self.assertFalse(data.x.flags.writeable)

    def test_causal_tokens_and_trainable_readin(self):
        for name in ('transformer', 'mlp', 'small_transformer'):
            net = PopulationDecoder(recipe(name), seed=401).eval()
            x = torch.randn(3, 512, 32)
            future = x.clone()
            future[:, :, 16:] += 50
            early = 16//net.config.patch
            torch.testing.assert_close(net.encode(x)[:, :early], net.encode(future)[:, :early], rtol=0, atol=0)
            torch.testing.assert_close(net(x), torch.zeros(3), rtol=0, atol=0)
            with torch.no_grad():
                net.head[-1].weight.fill_(.01)
            net(x).square().sum().backward()
            self.assertGreater(float(net.readin.grad.abs().sum()), 0)
            with self.assertRaises(ValueError):
                net(torch.zeros(3, 511, 32))

    def test_ridge_matches_augmented_least_squares(self):
        rng = np.random.default_rng(20)
        x, y = rng.normal(size=(25, 9)), rng.normal(size=25)
        penalty = .1
        state = solve(x, y, [penalty])
        z = (x-state['center'])/state['scale']
        augmented = np.vstack([z, np.sqrt(penalty*len(x))*np.eye(x.shape[1])])
        target = np.concatenate([y-y.mean(), np.zeros(x.shape[1])])
        expected = np.linalg.lstsq(augmented, target, rcond=None)[0]
        np.testing.assert_allclose(state['weights'][:, 0], expected, rtol=1e-10, atol=1e-10)

    def test_metrics_include_physical_zero_and_constant_targets(self):
        result = metrics(np.array([-2., 2.]), np.array([-1., 1.]), -1.)
        self.assertEqual(result['mse'], .5)
        self.assertEqual(result['mae'], .5)
        self.assertEqual(result['r2'], .5)
        self.assertIsNone(metrics(np.ones(3), np.ones(3), -1.)['r2'])
        with self.assertRaises(ValueError):
            metrics(np.ones(2), np.ones(3), -1.)

    def test_training_lock_evaluation_and_tamper_rejection(self):
        fits = []
        for name in ('transformer', 'mlp'):
            folder = self.root/name
            result = fit(self.prepared, folder, name, epochs=2)
            self.assertFalse(result['test_opened'])
            self.assertFalse(result['fixed_recipe'])
            self.assertEqual(result['updates'], 4)
            history = json.loads((folder/'history.json').read_text())
            self.assertEqual(result['selected_epoch'], int(np.argmin([r['validation_mse'] for r in history])))
            initial = torch.load(folder/'initial.pt', weights_only=True)
            self.assertEqual(float(initial['head.3.weight'].abs().sum()), 0)
            fits.append(folder)
        folder = self.root/'ridge'
        result = fit_ridge(self.prepared, folder)
        self.assertEqual(len(result['options']), 24)
        fits.append(folder)
        output = self.root/'evaluation'
        scores = evaluate(self.prepared, fits, output)
        self.assertEqual(len(scores), 5)
        self.assertTrue((output/'lock.json').exists())
        with self.assertRaises(FileExistsError):
            evaluate(self.prepared, fits, output)
        with self.assertRaises(ValueError):
            evaluate(self.prepared, [fits[0], fits[0]], self.root/'duplicate')
        checkpoint = fits[0]/'selected.pt'
        with checkpoint.open('ab') as file:
            file.write(b'changed')
        with self.assertRaises(ValueError):
            evaluate(self.prepared, fits, self.root/'tampered')
        self.assertFalse((self.root/'tampered').exists())


if __name__ == '__main__':
    unittest.main()
