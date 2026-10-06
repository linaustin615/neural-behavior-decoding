"""Learn population signals before comparing their temporal histories."""
import importlib.util
from pathlib import Path

import torch
from torch import nn

source=Path(__file__).resolve().parent.parent/'2026-10-03_dynamics_baseline'/'models.py'
spec=importlib.util.spec_from_file_location('population_token_axes',source)
axes=importlib.util.module_from_spec(spec);spec.loader.exec_module(axes)
NEURONS=512
WIDTH=16
TIMES=8
PATCH=4


class PopulationDecoder(nn.Module):
    def __init__(self,kind,seed,sessions):
        super().__init__();self.kind=kind;self.sessions=tuple(sessions)
        torch.manual_seed(seed)
        self.readin=nn.Parameter((torch.randn(4,NEURONS,WIDTH)/NEURONS**.5)[list(sessions)].clone())
        self.session=nn.Parameter(torch.randn(4,WIDTH)[list(sessions)]*.02)
        self.time=nn.Parameter(torch.randn(TIMES,WIDTH)*.02)
        self.patch=nn.Linear(PATCH*WIDTH,WIDTH)
        torch.manual_seed(seed+100)
        self.temporal=axes.AxisBlock('attention' if kind=='attention' else 'mixer','time')
        torch.manual_seed(seed+200)
        self.head=nn.Sequential(nn.LayerNorm(WIDTH+64),nn.Linear(WIDTH+64,64),nn.GELU(),nn.Linear(64,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def encode(self,x,session):
        if session not in self.sessions:raise ValueError('Session absent from this model')
        if x.ndim!=3 or x.shape[1:]!=(NEURONS,32):raise ValueError('Expected batch x512neurons x32bins')
        slot=self.sessions.index(session)
        population=x.transpose(1,2)@self.readin[slot]
        z=self.patch(population.reshape(len(x),TIMES,PATCH*WIDTH))+self.time[None]+self.session[slot]
        return self.temporal(z)

    def forward(self,x,session):
        z=self.encode(x,session)
        features=torch.cat([z[:,-1],x.mean(1),x.std(1,unbiased=False)],1)
        return self.head(features).squeeze(-1)
