"""Test stable neuron assignment while preserving the population-statistics path."""
import importlib.util
from pathlib import Path

import torch
from torch import nn

source=Path(__file__).resolve().parent.parent/'2026-10-03_shared_behavior'/'models.py'
spec=importlib.util.spec_from_file_location('behavior_information_parent',source)
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
BehaviorDecoder=parent.BehaviorDecoder
WIDTH,TIMES,PATCH=parent.WIDTH,parent.TIMES,parent.PATCH


def population_stats(x):return torch.cat([x.mean(1),x.std(1,unbiased=False)],1)


class AnonymousDecoder(BehaviorDecoder):
    def __init__(self,temporal,seed):
        super().__init__(temporal,seed,range(4))
        self.kind='attention'
        self.temporal_kind=temporal

    def forward(self,x,session,neuron_order=None):
        if session not in self.sessions:raise ValueError('Session is absent from this model')
        slot=self.sessions.index(session);b,n,t=x.shape
        encoder_x=x if neuron_order is None else x.gather(1,neuron_order[:,:,None].expand(-1,-1,t))
        position=self.identity[slot,:,None]+self.time[None]+self.session[slot]
        z=self.patch(encoder_x.reshape(b,n,TIMES,PATCH))+position[None]
        z=self.temporal(z.reshape(b*n,TIMES,WIDTH)).reshape(b,n*TIMES,WIDTH)
        source=self.source_norm(z)
        query=(self.behavior_query+self.session[slot])[None].expand(b,-1,-1)
        q=self.query_projection(query).reshape(b,1,2,WIDTH//2).transpose(1,2)
        k=self.key_projection(source).reshape(b,n*TIMES,2,WIDTH//2).transpose(1,2)
        v=self.value_projection(source).reshape(b,n*TIMES,2,WIDTH//2).transpose(1,2)
        weights=(q@k.transpose(-1,-2)/(WIDTH//2)**.5).softmax(-1)
        mixed=(self.drop(weights)@v).transpose(1,2).reshape(b,1,WIDTH)
        query=query+self.drop(self.route_output(mixed))
        query=query+self.drop(self.query_ff(self.query_norm(query)))
        #compute the summary path from the original input to preserve its exact values
        features=torch.cat([query[:,0],population_stats(x)],1)
        return self.head(features).squeeze(-1)


class StatisticsMLP(nn.Module):
    def __init__(self,seed):
        super().__init__();torch.manual_seed(seed)
        self.session=nn.Parameter(torch.randn(4,16)*.02)
        self.query=nn.Parameter(torch.randn(1,16)*.02)
        torch.manual_seed(seed+200)
        self.head=nn.Sequential(nn.LayerNorm(80),nn.Linear(80,160),nn.GELU(),nn.Dropout(.05),
            nn.Linear(160,32),nn.GELU(),nn.Linear(32,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,x,session,neuron_order=None):
        if not 0<=session<4:raise ValueError('Session is absent from this model')
        context=(self.query+self.session[session]).expand(len(x),-1)
        return self.head(torch.cat([context,population_stats(x)],1)).squeeze(-1)
