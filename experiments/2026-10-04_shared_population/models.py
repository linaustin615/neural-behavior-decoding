"""Add one population block to the shared behavior decoder's latest time states."""
import importlib.util
from pathlib import Path

import torch


source = Path(__file__).resolve().parent.parent / '2026-10-03_shared_behavior' / 'models.py'
spec = importlib.util.spec_from_file_location('population_parent', source)
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
BehaviorDecoder = parent.BehaviorDecoder
WIDTH, TIMES, PATCH = parent.WIDTH, parent.TIMES, parent.PATCH


class PopulationDecoder(BehaviorDecoder):
    def __init__(self, variant, seed, sessions):
        assert variant in ['attention', 'mixer']
        super().__init__('attention', seed, sessions)
        self.variant = variant
        self.population_enabled = True
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 300)
            self.population = parent.axes.AxisBlock(variant, 'population')

    def mix_latest(self, z):
        latest = self.population(z[:, :, -1].contiguous())
        return torch.cat([z[:, :, :-1], latest[:, :, None]], dim=2)

    def forward(self, x, session, uniform=False, return_weights=False):
        if session not in self.sessions:
            raise ValueError('Session is absent from this model')
        slot = self.sessions.index(session)
        b, n, t = x.shape
        assert n == 128 and t == 32
        position = self.identity[slot, :, None] + self.time[None] + self.session[slot]
        z = self.patch(x.reshape(b, n, TIMES, PATCH)) + position[None]
        z = self.temporal(z.reshape(b * n, TIMES, WIDTH)).reshape(b, n, TIMES, WIDTH)
        if self.population_enabled:
            z = self.mix_latest(z) #retain the earlier time states
        z = z.reshape(b, n * TIMES, WIDTH)
        source = self.source_norm(z)
        static = self.source_norm(position.reshape(1, n * TIMES, WIDTH)).expand(b, -1, -1)
        query = (self.behavior_query + self.session[slot])[None].expand(b, -1, -1)
        q = self.query_projection(query).reshape(b, 1, 2, WIDTH // 2).transpose(1, 2)
        key = source if self.kind == 'attention' else static
        k = self.key_projection(key).reshape(b, n * TIMES, 2, WIDTH // 2).transpose(1, 2)
        v = self.value_projection(source).reshape(b, n * TIMES, 2, WIDTH // 2).transpose(1, 2)
        weights = (q @ k.transpose(-1, -2) / (WIDTH // 2) ** .5).softmax(-1)
        if uniform:
            weights = torch.full_like(weights, 1 / (n * TIMES))
        mixed = (self.drop(weights) @ v).transpose(1, 2).reshape(b, 1, WIDTH)
        query = query + self.drop(self.route_output(mixed))
        query = query + self.drop(self.query_ff(self.query_norm(query)))
        features = torch.cat([query[:, 0], x.mean(1), x.std(1, unbiased=False)], 1)
        prediction = self.head(features).squeeze(-1)
        return (prediction, weights) if return_weights else prediction
