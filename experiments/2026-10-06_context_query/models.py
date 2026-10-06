"""Condition neuron readout on a learned population trajectory."""
import importlib.util
from pathlib import Path

import torch
from torch import nn

EXP=Path(__file__).resolve().parent.parent


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    loaded=importlib.util.module_from_spec(spec);spec.loader.exec_module(loaded);return loaded


local=module('context_local_decoder',EXP/'2026-10-05_larger_panel'/'models.py')
population=module('context_population_decoder',EXP/'2026-10-06_population_tokens'/'models.py')
WIDTH=16
TIMES=8
PATCH=4


class ContextQueryDecoder(local.BehaviorDecoder):
    def __init__(self,context_kind,seed,sessions):
        super().__init__('mlp',seed,sessions)
        self.context_kind=context_kind
        self.context=population.PopulationDecoder(context_kind,seed+500,sessions)
        del self.context.head
        self.context_projection=nn.Linear(WIDTH,WIDTH,bias=False)
        nn.init.zeros_(self.context_projection.weight)

    def forward(self,x,session,uniform=False,return_weights=False,context_enabled=True):
        if session not in self.sessions:raise ValueError('Session absent from this model')
        slot=self.sessions.index(session);b,n,t=x.shape
        position=self.identity[slot,:,None]+self.time[None]+self.session[slot]
        z=self.patch(x.reshape(b,n,TIMES,PATCH))+position[None]
        z=self.temporal(z.reshape(b*n,TIMES,WIDTH)).reshape(b,n*TIMES,WIDTH)
        source=self.source_norm(z)
        static=self.source_norm(position.reshape(1,n*TIMES,WIDTH)).expand(b,-1,-1)
        query=(self.behavior_query+self.session[slot])[None].expand(b,-1,-1)
        if context_enabled:query=query+self.context_projection(self.context.encode(x,session)[:,-1])[:,None]
        q=self.query_projection(query).reshape(b,1,2,WIDTH//2).transpose(1,2)
        k=self.key_projection(static).reshape(b,n*TIMES,2,WIDTH//2).transpose(1,2)
        v=self.value_projection(source).reshape(b,n*TIMES,2,WIDTH//2).transpose(1,2)
        weights=(q@k.transpose(-1,-2)/(WIDTH//2)**.5).softmax(-1)
        if uniform:weights=torch.full_like(weights,1/(n*TIMES))
        mixed=(self.drop(weights)@v).transpose(1,2).reshape(b,1,WIDTH)
        query=query+self.drop(self.route_output(mixed))
        query=query+self.drop(self.query_ff(self.query_norm(query)))
        features=torch.cat([query[:,0],x.mean(1),x.std(1,unbiased=False)],1)
        prediction=self.head(features).squeeze(-1)
        return (prediction,weights) if return_weights else prediction
