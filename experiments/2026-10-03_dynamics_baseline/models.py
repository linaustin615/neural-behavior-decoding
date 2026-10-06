"""Small continuous-patch models retaining the neuron and time axes."""
import math
import torch
from torch import nn
from torch.nn import functional as F

NEURONS=128
CONTEXT=32
PATCH=4
WIDTH=16
TIMES=CONTEXT//PATCH


class AxisBlock(nn.Module):
    def __init__(self,kind,axis):
        super().__init__()
        self.kind,self.axis=kind,axis
        self.norm1=nn.LayerNorm(WIDTH)
        self.norm2=nn.LayerNorm(WIDTH)
        self.ff=nn.Sequential(nn.Linear(WIDTH,32),nn.GELU(),nn.Dropout(.05),nn.Linear(32,WIDTH))
        self.drop=nn.Dropout(.05)
        if kind=='attention':
            self.mix=nn.MultiheadAttention(WIDTH,2,dropout=.05,batch_first=True)
        elif axis=='time':
            self.weight=nn.Parameter(torch.randn(TIMES,TIMES)/math.sqrt(TIMES))
            self.bias=nn.Parameter(torch.zeros(TIMES))
            self.mix=nn.Sequential(nn.Linear(WIDTH,30),nn.GELU(),nn.Linear(30,WIDTH))
        else:
            self.mix=nn.Sequential(nn.Linear(NEURONS,4),nn.GELU(),nn.Linear(4,NEURONS))
        self.register_buffer('future',torch.triu(torch.ones(TIMES,TIMES,dtype=torch.bool),1))

    def forward(self,x):
        z=self.norm1(x)
        if self.kind=='attention':
            z=self.mix(z,z,z,attn_mask=self.future if self.axis=='time' else None,need_weights=False)[0]
        elif self.axis=='time':
            z=F.linear(z.transpose(1,2),self.weight.masked_fill(self.future,0),self.bias).transpose(1,2)
            z=self.mix(z)
        else:
            z=self.mix(z.transpose(1,2)).transpose(1,2)
        x=x+self.drop(z)
        return x+self.drop(self.ff(self.norm2(x)))


class Dynamics(nn.Module):
    def __init__(self,kind,seed):
        super().__init__()
        torch.manual_seed(seed)
        self.patch=nn.Linear(PATCH,WIDTH)
        self.identity=nn.Embedding(NEURONS,WIDTH)
        nn.init.normal_(self.identity.weight,std=.02)
        self.position=nn.Parameter(torch.randn(1,1,TIMES,WIDTH)*.02)
        torch.manual_seed(seed+100)
        self.temporal=AxisBlock(kind,'time')
        self.population=AxisBlock(kind,'population')
        torch.manual_seed(seed+200)
        self.forecast=nn.Linear(WIDTH,PATCH)
        self.speed=nn.Sequential(nn.LayerNorm(WIDTH+2*CONTEXT),nn.Linear(WIDTH+2*CONTEXT,64),nn.GELU(),nn.Linear(64,1))

    def encode(self,x):
        b,n,t=x.shape
        z=self.patch(x.reshape(b,n,TIMES,PATCH))+self.identity.weight[None,:,None]+self.position
        z=self.temporal(z.reshape(b*n,TIMES,WIDTH)).reshape(b,n,TIMES,WIDTH)
        z=z.transpose(1,2).reshape(b*TIMES,n,WIDTH)
        z=self.population(z).reshape(b,TIMES,n,WIDTH).transpose(1,2)
        return z

    def forward(self,x,task='speed'):
        z=self.encode(x)[:,:,-1]
        if task=='forecast':return self.forecast(z)
        features=torch.cat([z.mean(1),x.mean(1),x.std(1,unbiased=False)],1)
        return self.speed(features).squeeze(-1)


class PooledMLP(nn.Module):
    def __init__(self,seed):
        super().__init__()
        torch.manual_seed(seed)
        self.activity=nn.Sequential(nn.Linear(CONTEXT,32),nn.GELU(),nn.Linear(32,32))
        self.identity=nn.Embedding(NEURONS,32)
        nn.init.normal_(self.identity.weight,std=.02)
        self.norm=nn.LayerNorm(32)
        self.mix=nn.Sequential(nn.LayerNorm(32),nn.Linear(32,64),nn.GELU(),nn.Dropout(.05),nn.Linear(64,32))
        self.head=nn.Sequential(nn.Linear(32,256),nn.GELU(),nn.Linear(256,1))
        self.amplitude=nn.Linear(2*CONTEXT,1,bias=False)
        nn.init.zeros_(self.amplitude.weight)

    def forward(self,x,task='speed'):
        z=self.norm(self.activity(x)+self.identity.weight[None])
        return self.head((z+self.mix(z)).mean(1)).squeeze(-1)+self.amplitude(torch.cat([x.mean(1),x.std(1,unbiased=False)],1)).squeeze(-1)
