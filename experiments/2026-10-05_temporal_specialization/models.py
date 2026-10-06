"""Assign four readout queries fixed temporal roles without adding parameters."""
import importlib.util
from pathlib import Path

import torch

source = Path(__file__).resolve().parent.parent / '2026-10-05_multiquery_readout' / 'models.py'
spec = importlib.util.spec_from_file_location('specialized_readout_parent', source)
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
ReadoutDecoder = parent.ReadoutDecoder
WIDTH, TIMES, PATCH = parent.WIDTH, parent.TIMES, parent.PATCH
VARIANTS = {'time_dynamic': 'dynamic', 'time_static': 'static', 'interleaved_dynamic': 'dynamic'}


class SpecializedDecoder(ReadoutDecoder):
    def __init__(self, variant, seed, sessions):
        assert variant in VARIANTS
        super().__init__(VARIANTS[variant], seed, sessions)
        self.variant = variant
        self.mask_enabled = True
        times = torch.arange(TIMES).repeat(128)
        owner = times % 4 if variant == 'interleaved_dynamic' else times // 2
        allowed = torch.arange(4)[:, None] == owner[None]
        self.register_buffer('allowed', allowed)

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
        logits = q @ k.transpose(-1, -2) / (WIDTH // 2) ** .5
        if self.mask_enabled:
            logits = logits.masked_fill(~self.allowed[None, None], -torch.inf)
        weights = logits.softmax(-1)
        if uniform:
            if self.mask_enabled:
                weights = self.allowed.to(weights.dtype)[None, None].expand_as(weights) / 256
            else:
                weights = torch.full_like(weights, 1 / (n * TIMES))
        mixed = (self.drop(weights) @ v).transpose(1, 2).reshape(b, self.queries, WIDTH)
        query = query + self.drop(self.route_output(mixed))
        query = query + self.drop(self.query_ff(self.query_norm(query)))
        features = torch.cat([query.reshape(b, self.queries * WIDTH), x.mean(1), x.std(1, unbiased=False)], 1)
        prediction = self.head(features).squeeze(-1)
        return (prediction, weights) if return_weights else prediction
