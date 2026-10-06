"""Behavior-query decoding with shared or session-specific training."""
import importlib.util
from pathlib import Path

import torch
from torch import nn

source=Path(__file__).resolve().parent.parent/'2026-10-03_dynamics_baseline'/'models.py'
spec=importlib.util.spec_from_file_location('dynamics_axes',source)
axes=importlib.util.module_from_spec(spec);spec.loader.exec_module(axes)
WIDTH=16
NEURONS=128
TIMES=8
PATCH=4


class BehaviorDecoder(nn.Module):
    def __init__(self,kind,seed,sessions):
        super().__init__()
        self.kind=kind
        self.sessions=tuple(sessions)
        torch.manual_seed(seed)
        self.patch=nn.Linear(PATCH,WIDTH)
        ids=torch.randn(4,NEURONS,WIDTH)*.02
        tags=torch.randn(4,WIDTH)*.02
        self.identity=nn.Parameter(ids[list(sessions)].clone())
        self.session=nn.Parameter(tags[list(sessions)].clone())
        self.time=nn.Parameter(torch.randn(TIMES,WIDTH)*.02)
        self.behavior_query=nn.Parameter(torch.randn(1,WIDTH)*.02)
        torch.manual_seed(seed+100)
        self.temporal=axes.AxisBlock('attention' if kind=='attention' else 'mixer','time')
        torch.manual_seed(seed+200)
        self.source_norm=nn.LayerNorm(WIDTH)
        self.query_projection=nn.Linear(WIDTH,WIDTH)
        self.key_projection=nn.Linear(WIDTH,WIDTH)
        self.value_projection=nn.Linear(WIDTH,WIDTH)
        self.route_output=nn.Linear(WIDTH,WIDTH)
        self.query_norm=nn.LayerNorm(WIDTH)
        self.query_ff=nn.Sequential(nn.Linear(WIDTH,32),nn.GELU(),nn.Dropout(.05),nn.Linear(32,WIDTH))
        self.drop=nn.Dropout(.05)
        self.head=nn.Sequential(nn.LayerNorm(WIDTH+64),nn.Linear(WIDTH+64,64),nn.GELU(),nn.Linear(64,1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self,x,session,uniform=False,return_weights=False):
        if session not in self.sessions:raise ValueError('Session is absent from this model')
        slot=self.sessions.index(session)
        b,n,t=x.shape
        position=self.identity[slot,:,None]+self.time[None]+self.session[slot]
        z=self.patch(x.reshape(b,n,TIMES,PATCH))+position[None]
        z=self.temporal(z.reshape(b*n,TIMES,WIDTH)).reshape(b,n*TIMES,WIDTH)
        source=self.source_norm(z)
        static=self.source_norm(position.reshape(1,n*TIMES,WIDTH)).expand(b,-1,-1)
        query=(self.behavior_query+self.session[slot])[None].expand(b,-1,-1)
        q=self.query_projection(query).reshape(b,1,2,WIDTH//2).transpose(1,2)
        key=source if self.kind=='attention' else static
        k=self.key_projection(key).reshape(b,n*TIMES,2,WIDTH//2).transpose(1,2)
        v=self.value_projection(source).reshape(b,n*TIMES,2,WIDTH//2).transpose(1,2)
        weights=(q@k.transpose(-1,-2)/(WIDTH//2)**.5).softmax(-1)
        if uniform:weights=torch.full_like(weights,1/(n*TIMES))
        mixed=(self.drop(weights)@v).transpose(1,2).reshape(b,1,WIDTH)
        query=query+self.drop(self.route_output(mixed))
        query=query+self.drop(self.query_ff(self.query_norm(query)))
        features=torch.cat([query[:,0],x.mean(1),x.std(1,unbiased=False)],1)
        prediction=self.head(features).squeeze(-1)
        return (prediction,weights) if return_weights else prediction
