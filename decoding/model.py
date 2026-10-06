"""The population transformer and its feed-forward temporal-MLP control."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import MICE, NEURONS, Recipe


class TemporalBlock(nn.Module):
    def __init__(self, family, width, times, dropout):
        super().__init__()
        if family not in ('attention', 'mlp'):
            raise ValueError('Expected attention or mlp.')
        self.family = family
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2*width), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(2*width, width))
        self.drop = nn.Dropout(dropout)
        if family == 'attention':
            self.mix = nn.MultiheadAttention(width, 2 if width == 16 else 4,
                                             dropout=dropout, batch_first=True)
        else:
            self.weight = nn.Parameter(torch.randn(times, times) / math.sqrt(times))
            self.bias = nn.Parameter(torch.zeros(times))
            self.mix = nn.Sequential(nn.Linear(width, 2*width-2), nn.GELU(),
                                     nn.Linear(2*width-2, width))
        self.register_buffer('future', torch.triu(torch.ones(times, times, dtype=torch.bool), 1))

    def forward(self, x):
        z = self.norm1(x)
        if self.family == 'attention':
            z = self.mix(z, z, z, attn_mask=self.future, need_weights=False)[0]
        else:
            weights = self.weight.masked_fill(self.future, 0) #prevent access to later tokens
            z = F.linear(z.transpose(1, 2), weights, self.bias).transpose(1, 2)
            z = self.mix(z)
        x = x + self.drop(z)
        return x + self.drop(self.ff(self.norm2(x)))


class PopulationDecoder(nn.Module):
    def __init__(self, config: Recipe, seed=401, mouse_index=0):
        super().__init__()
        if mouse_index is not None and mouse_index not in range(len(MICE)):
            raise ValueError('Mouse index must be 0..6, or None for shared fitting.')
        width, history, patch = config.width, config.history, config.patch
        if patch < 1 or history < 1 or history > 64 or width < 4 or history % patch or width % 4 or config.depth < 1:
            raise ValueError('Invalid patch, width or depth.')
        self.config = config
        times = history // patch
        torch.manual_seed(seed)
        #preserve the original four-session initialization before the seven-mouse extension
        self.readin = nn.Parameter(torch.randn(4, NEURONS, width) / math.sqrt(NEURONS))
        self.session = nn.Parameter(torch.randn(4, width) * .02)
        self.time = nn.Parameter(torch.randn(times, width) * .02)
        self.patch = nn.Linear(patch*width, width)
        self.layers = nn.ModuleList()
        for i in range(config.depth):
            torch.manual_seed(seed + 100 + i)
            self.layers.append(TemporalBlock(config.family, width, times, config.dropout))
        torch.manual_seed(seed + 200)
        self.head = nn.Sequential(nn.LayerNorm(width + 2*history), nn.Linear(width + 2*history, 64),
                                 nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.head[-1].weight) #start at the training-mean predictor
        nn.init.zeros_(self.head[-1].bias)
        torch.manual_seed(seed + 3000)
        extra = torch.randn(3, NEURONS, width) / math.sqrt(NEURONS)
        extra_session = torch.randn(3, width) * .02
        readin = torch.cat([self.readin.detach(), extra])
        session = torch.cat([self.session.detach(), extra_session])
        if mouse_index is not None:
            readin = readin[mouse_index:mouse_index+1].clone()
            session = session[mouse_index:mouse_index+1].clone()
        self.readin = nn.Parameter(readin)
        self.session = nn.Parameter(session)

    def encode(self, x, session=0):
        c = self.config
        if x.ndim != 3 or tuple(x.shape[1:]) != (NEURONS, c.history):
            raise ValueError(f'Expected [batch, {NEURONS}, {c.history}], got {tuple(x.shape)}.')
        if session not in range(len(self.session)):
            raise ValueError('Unknown model session.')
        z = x.transpose(1, 2) @ self.readin[session] #combine neurons into population features
        z = self.patch(z.reshape(len(x), c.history // c.patch, c.patch*c.width))
        z = z + self.time[None] + self.session[session]
        for layer in self.layers:
            z = layer(z)
        return z

    def forward(self, x, session=0):
        tokens = self.encode(x, session)
        features = torch.cat([tokens[:, -1], x.mean(1), x.std(1, unbiased=False)], dim=1)
        return self.head(features).squeeze(-1)
