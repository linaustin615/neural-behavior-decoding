"""Target-specific attention versus fixed routing for missing-neuron activity."""
import math
import torch
from torch import nn

CELLS=64
WIDTH=16
CONTEXT=32
OUTPUT=1


class StaticRouting(nn.Module):
    def __init__(self):
        super().__init__()
        self.query=nn.Parameter(torch.randn(2,CELLS,2)*.1)
        self.key=nn.Parameter(torch.randn(2,CELLS,2)*.1)
        self.value=nn.Linear(WIDTH,WIDTH)
        self.output=nn.Linear(WIDTH,WIDTH)

    def forward(self,source,query):
        b=source.shape[0]
        values=self.value(source).reshape(b,CELLS,2,WIDTH//2).transpose(1,2)
        weights=(self.query@self.key.transpose(-1,-2)/math.sqrt(2)).softmax(-1)
        z=(weights[None]@values).transpose(1,2).reshape(b,CELLS,WIDTH)
        return self.output(z)


class Reconstruction(nn.Module):
    def __init__(self,kind,seed,mean):
        super().__init__()
        self.kind=kind
        self.register_buffer('target_mean',torch.as_tensor(mean,dtype=torch.float32))
        torch.manual_seed(seed)
        if kind=='mlp':
            self.body=nn.Sequential(nn.Linear(CELLS*CONTEXT,64),nn.GELU(),nn.Dropout(.05))
            self.output=nn.Linear(64,CELLS*OUTPUT)
        else:
            self.activity=nn.Sequential(nn.Linear(CONTEXT,32),nn.GELU(),nn.Linear(32,WIDTH))
            self.source_id=nn.Parameter(torch.randn(CELLS,WIDTH)*.02)
            self.target_id=nn.Parameter(torch.randn(CELLS,WIDTH)*.02)
            self.source_norm=nn.LayerNorm(WIDTH)
            torch.manual_seed(seed+100)
            self.routing=nn.MultiheadAttention(WIDTH,2,dropout=.05,batch_first=True) if kind=='attention' else StaticRouting()
            torch.manual_seed(seed+200)
            self.norm=nn.LayerNorm(WIDTH)
            self.ff=nn.Sequential(nn.Linear(WIDTH,32),nn.GELU(),nn.Dropout(.05),nn.Linear(32,WIDTH))
            self.drop=nn.Dropout(.05)
            self.output=nn.Linear(WIDTH,OUTPUT)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self,x):
        if self.kind=='mlp':
            pred=self.output(self.body(x.flatten(1))).reshape(len(x),CELLS,OUTPUT)
        else:
            source=self.source_norm(self.activity(x)+self.source_id[None])
            query=self.target_id[None].expand(len(x),-1,-1)
            if self.kind=='attention':mixed=self.routing(query,source,source,need_weights=False)[0]
            else:mixed=self.routing(source,query)
            z=query+self.drop(mixed)
            z=z+self.drop(self.ff(self.norm(z)))
            pred=self.output(z)
        return pred+self.target_mean[None]
