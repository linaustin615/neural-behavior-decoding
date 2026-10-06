import numpy as np
import torch
from torch import nn
import base_search as s


def spatial_groups(raw_positions):
    groups = np.zeros(len(raw_positions), dtype=np.int64)
    boundaries = []
    for level, axis in enumerate([2, 0, 1]):
        following = np.empty_like(groups)
        for group in range(2 ** level):
            indices = np.flatnonzero(groups == group)
            values, counts = np.unique(raw_positions[indices, axis], return_counts=True)
            assert len(values) > 1
            cut = int(np.argmin(np.abs(np.cumsum(counts)[:-1] - len(indices) / 2)))
            threshold = float((values[cut] + values[cut + 1]) / 2)
            following[indices] = 2 * group + (raw_positions[indices, axis] > threshold)
            boundaries.append(dict(level=level, group=group, axis=axis, threshold=threshold))
        groups = following
    assert set(groups) == set(range(8))
    return groups, boundaries


def grouping(raw_positions, pool, variant):
    original, boundaries = spatial_groups(raw_positions)
    groups = original.copy()
    rng = np.random.default_rng(420000 + pool)
    if variant == 'random':
        groups = groups[rng.permutation(len(groups))]
    elif variant == 'depth_random':
        for depth in np.unique(raw_positions[:, 2]):
            indices = np.flatnonzero(raw_positions[:, 2] == depth)
            groups[indices] = original[rng.permutation(indices)]
    assert variant in ['global', 'spatial', 'random', 'depth_random']
    np.testing.assert_array_equal(np.bincount(groups), np.bincount(original))
    return groups, boundaries


class GroupCandidate(s.Candidate):
    def __init__(self, config, groups):
        super().__init__(dict(config, latents=8))
        self.config = dict(config)
        generator = torch.Generator().manual_seed(700000 + config['initialization_seed'])
        extra = torch.randn(1, 8, config['width'], generator=generator) * .02
        self.latents = nn.Parameter(torch.cat([self.latents.detach(), extra], dim=1))
        self.register_buffer('group_mask', None, persistent=False)
        self.set_groups(groups, config['variant'])

    def set_groups(self, groups, variant):
        groups = torch.as_tensor(groups, dtype=torch.long)
        mask = torch.zeros(16, len(groups), dtype=torch.bool)
        if variant != 'global':
            mask[:8] = groups[None, :] != torch.arange(8)[:, None]
        assert (~mask).any(dim=1).all()
        self.group_mask = mask

    def read_tokens(self, activity, positions, token_ids, weights=False):
        tokens = self.base.encoder.tokenizer(activity, positions, token_ids)
        queries = self.latents.expand(activity.shape[0], -1, -1)
        latent, attention = self.readin(queries, tokens, tokens, attn_mask=self.group_mask,
                                       need_weights=weights, average_attn_weights=False)
        return latent, queries, attention

    def forward(self, activity, positions, token_ids, ablate_local=False):
        latent, queries, _ = self.read_tokens(activity, positions, token_ids)
        if ablate_local:
            latent = torch.cat([latent[:, 8:].mean(1, keepdim=True).expand(-1, 8, -1), latent[:, 8:]], dim=1)
        encoded = self.base.encoder.transformer(latent + queries)
        return self.base.head(encoded.mean(dim=1)).squeeze(-1)
