"""Preserve the existing speed head and add a neuron-specific linear readout."""
from pathlib import Path
import importlib.util

import torch
from torch import nn

source=Path(__file__).resolve().parent.parent/'2026-10-03_dynamics_baseline'/'models.py'
spec=importlib.util.spec_from_file_location('dynamics_models',source)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)


class NeuronReadout(nn.Module):
    def __init__(self,kind,seed,checkpoint):
        super().__init__()
        self.encoder=base.Dynamics(kind,seed)
        self.encoder.load_state_dict(torch.load(checkpoint,weights_only=True))
        self.neuron_head=nn.Linear(base.NEURONS*base.WIDTH,1)
        nn.init.zeros_(self.neuron_head.weight)
        nn.init.zeros_(self.neuron_head.bias)

    def forward(self,x):
        z=self.encoder.encode(x)[:,:,-1]
        pooled=torch.cat([z.mean(1),x.mean(1),x.std(1,unbiased=False)],1)
        return self.encoder.speed(pooled).squeeze(-1)+self.neuron_head(z.flatten(1)).squeeze(-1)
