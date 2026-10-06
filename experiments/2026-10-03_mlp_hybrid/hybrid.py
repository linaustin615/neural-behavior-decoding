"""Normalized pooled MLP with a zero-initialized parallel correction branch."""
import torch
from torch import nn


class Hybrid(nn.Module):
    def __init__(self,base,kind,seed):
        super().__init__()
        self.base=base
        self.kind=kind
        torch.manual_seed(seed+51000)
        if kind=='attention':
            self.queries=nn.Parameter(torch.randn(1,16,32)*.02)
            self.readin=nn.MultiheadAttention(32,4,batch_first=True)
            self.body=nn.TransformerEncoderLayer(32,4,64,dropout=.05,batch_first=True)
            self.head=nn.Linear(32,1)
        elif kind=='extra_mlp':
            self.body=nn.Sequential(nn.LayerNorm(32),nn.Linear(32,64),nn.GELU(),nn.Dropout(.05),nn.Linear(64,32))
            self.head=nn.Sequential(nn.Linear(32,266),nn.GELU(),nn.Linear(266,1))
        else:
            raise ValueError(kind)
        output=self.head if kind=='attention' else self.head[-1]
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def forward(self,x,ids=None):
        if ids is None:ids=torch.arange(x.shape[1],device=x.device)
        tok=self.base.original.base.encoder.tokenizer
        tokens=self.base.norm(tok(x,torch.zeros(x.shape[1],3,device=x.device),ids))
        pooled=(tokens+self.base.mix(tokens)).mean(1)
        amplitude=torch.cat([x.mean(1),x.std(1,unbiased=False)],-1)
        main=self.base.original.base.head(pooled).squeeze(-1)+self.base.amplitude(amplitude).squeeze(-1)
        if self.kind=='attention':
            query=self.queries.expand(len(x),-1,-1)
            z,_=self.readin(query,tokens,tokens,need_weights=False)
            z=self.body(z+query).mean(1)
        else:
            z=(tokens+self.body(tokens)).mean(1)
        correction=self.head(z).squeeze(-1)
        return main+correction,main,correction
