"""Test several learned pooling queries on the shared temporal decoder."""
import importlib.util
from pathlib import Path

import torch
from torch import nn


source = Path(__file__).resolve().parent.parent / '2026-10-03_shared_behavior' / 'models.py'
spec = importlib.util.spec_from_file_location('multiquery_parent', source)
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
BehaviorDecoder = parent.BehaviorDecoder
WIDTH, TIMES, PATCH = parent.WIDTH, parent.TIMES, parent.PATCH


class ReadoutDecoder(BehaviorDecoder):
    def __init__(self, variant, seed, sessions, queries=4):
        assert variant in ['dynamic', 'static'] and queries in [1, 4]
        super().__init__('attention', seed, sessions)
        self.variant, self.queries = variant, queries
        self.kind = 'attention' if variant == 'dynamic' else 'mlp'
        if queries > 1:
            first_query = self.behavior_query.detach().clone()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 300)
                extra = torch.randn(queries - 1, WIDTH) * .02
                self.behavior_query = nn.Parameter(torch.cat([first_query, extra]))
                size = queries * WIDTH + 64
                self.head = nn.Sequential(nn.LayerNorm(size), nn.Linear(size, 64), nn.GELU(), nn.Linear(64, 1))
                nn.init.zeros_(self.head[-1].weight)
                nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, session, uniform=False, return_weights=False):
        if session not in self.sessions:
            raise ValueError('Session is absent from this model')
        slot = self.sessions.index(session)
        b, n, t = x.shape
        assert n == 128 and t == 32
        position = self.identity[slot, :, None] + self.time[None] + self.session[slot]
        z = self.patch(x.reshape(b, n, TIMES, PATCH)) + position[None]
        z = self.temporal(z.reshape(b * n, TIMES, WIDTH)).reshape(b, n * TIMES, WIDTH)
        source = self.source_norm(z)
        static = self.source_norm(position.reshape(1, n * TIMES, WIDTH)).expand(b, -1, -1)
        query = (self.behavior_query + self.session[slot])[None].expand(b, -1, -1)
        q = self.query_projection(query).reshape(b, self.queries, 2, WIDTH // 2).transpose(1, 2)
        key = source if self.kind == 'attention' else static
        k = self.key_projection(key).reshape(b, n * TIMES, 2, WIDTH // 2).transpose(1, 2)
        v = self.value_projection(source).reshape(b, n * TIMES, 2, WIDTH // 2).transpose(1, 2)
        weights = (q @ k.transpose(-1, -2) / (WIDTH // 2) ** .5).softmax(-1)
        if uniform:
            weights = torch.full_like(weights, 1 / (n * TIMES))
        mixed = (self.drop(weights) @ v).transpose(1, 2).reshape(b, self.queries, WIDTH)
        query = query + self.drop(self.route_output(mixed))
        query = query + self.drop(self.query_ff(self.query_norm(query)))
        features = torch.cat([query.reshape(b, self.queries * WIDTH), x.mean(1), x.std(1, unbiased=False)], 1)
        prediction = self.head(features).squeeze(-1)
        return (prediction, weights) if return_weights else prediction
