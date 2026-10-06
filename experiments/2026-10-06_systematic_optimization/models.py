"""Configurable population decoders and the established local-MLP control."""
import importlib.util
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


class TemporalBlock(nn.Module):
    def __init__(self, kind, width, times, dropout):
        super().__init__()
        self.kind = kind
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2*width), nn.GELU(), nn.Dropout(dropout), nn.Linear(2*width, width))
        self.drop = nn.Dropout(dropout)
        if kind == 'attention':
            self.mix = nn.MultiheadAttention(width, 2 if width == 16 else 4, dropout=dropout, batch_first=True)
        else:
            self.weight = nn.Parameter(torch.randn(times, times) / math.sqrt(times))
            self.bias = nn.Parameter(torch.zeros(times))
            self.mix = nn.Sequential(nn.Linear(width, 2*width-2), nn.GELU(), nn.Linear(2*width-2, width))
        self.register_buffer('future', torch.triu(torch.ones(times, times, dtype=torch.bool), 1))

    def forward(self, x):
        z = self.norm1(x)
        if self.kind == 'attention':
            z = self.mix(z, z, z, attn_mask=self.future, need_weights=False)[0]
        else:
            z = self.mix(F.linear(z.transpose(1, 2), self.weight.masked_fill(self.future, 0), self.bias).transpose(1, 2))
        x = x + self.drop(z)
        return x + self.drop(self.ff(self.norm2(x)))


class PopulationDecoder(nn.Module):
    def __init__(self, config, seed):
        super().__init__()
        self.config = dict(config)
        width, history, patch = config['width'], config['history'], config['patch']
        assert history % patch == 0
        times = history // patch
        torch.manual_seed(seed)
        self.readin = nn.Parameter(torch.randn(4, 512, width) / math.sqrt(512))
        self.session = nn.Parameter(torch.randn(4, width) * .02)
        self.time = nn.Parameter(torch.randn(times, width) * .02)
        self.patch = nn.Linear(patch*width, width)
        self.layers = nn.ModuleList()
        for i in range(config['depth']):
            torch.manual_seed(seed + 100 + i)
            self.layers.append(TemporalBlock(config['family'], width, times, config['dropout']))
        torch.manual_seed(seed + 200)
        self.head = nn.Sequential(nn.LayerNorm(width+2*history), nn.Linear(width+2*history, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def encode(self, x, session):
        c = self.config
        assert x.ndim == 3 and x.shape[1:] == (512, c['history']) and session in range(4)
        z = x.transpose(1, 2) @ self.readin[session]
        z = self.patch(z.reshape(len(x), c['history']//c['patch'], c['patch']*c['width']))
        z = z + self.time[None] + self.session[session]
        for layer in self.layers:
            z = layer(z)
        return z

    def forward(self, x, session):
        z = self.encode(x, session)
        return self.head(torch.cat([z[:, -1], x.mean(1), x.std(1, unbiased=False)], 1)).squeeze(-1)


source = Path(__file__).resolve().parent.parent/'2026-10-05_larger_panel'/'models.py'
spec = importlib.util.spec_from_file_location('optimization_local_parent', source)
local = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local)


def make_model(config, seed):
    if config['family'] != 'local_mlp':
        return PopulationDecoder(config, seed)
    assert (config['width'], config['depth'], config['history'], config['patch']) == (16, 1, 32, 4)
    net = local.BehaviorDecoder('mlp', seed, range(4))
    for module in net.modules():
        if isinstance(module, nn.Dropout):
            module.p = config['dropout']
    return net
