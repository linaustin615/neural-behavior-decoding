"""Concrete, bounded prototypes for the broad architecture screen."""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F

SINGLE=['time_attn','token_norm','state_head','masked_pretrain','lowrank_id',
        'population_gru','population_conv','multiscale','block_robust','functional_graph']
CONTROLS=dict(time_attn='time_mean',token_norm='norm_mlp',state_head='state_mlp',
              masked_pretrain='masked_mlp',lowrank_id='lowrank_mlp',population_gru='time_linear',
              population_conv='time_linear',multiscale='time_attn',block_robust='plain',
              functional_graph='graph_random',shared='lowrank_id')


class Temporal(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        self.activity=nn.Linear(1,16);self.identity=nn.Embedding(2048,16)
        nn.init.normal_(self.identity.weight,std=.02)
        self.queries=nn.Parameter(torch.randn(1,4,16)*.02)
        self.readin=nn.MultiheadAttention(16,4,batch_first=True)
        self.time=nn.Parameter(torch.randn(1,8,16)*.02)
        if kind in ['time_attn','multiscale']:
            self.temporal=nn.TransformerEncoderLayer(16,4,64,dropout=.05,batch_first=True)
        elif kind=='population_gru':
            self.temporal=nn.GRU(16,16,batch_first=True)
        elif kind=='population_conv':
            self.temporal=nn.Sequential(nn.Conv1d(16,32,3,padding=1),nn.GELU(),nn.Conv1d(32,16,3,padding=1),nn.GELU())
        elif kind=='time_mean':
            self.temporal=nn.Sequential(nn.Linear(16,64),nn.GELU(),nn.Linear(64,16))
        self.head=nn.Linear(128 if kind=='time_linear' else 16,1)

    def forward(self,x,subject=None):
        b,n,t=x.shape
        token=self.activity(x.transpose(1,2).reshape(b*t,n,1))+self.identity.weight[None]
        query=self.queries.expand(b*t,-1,-1)
        z,_=self.readin(query,token,token,need_weights=False)
        z=(z+query).mean(1).reshape(b,t,16)+self.time
        if self.kind=='population_gru':z=self.temporal(z)[0][:,-1]
        elif self.kind=='population_conv':z=self.temporal(z.transpose(1,2))[:,:,-1]
        elif self.kind=='time_linear':z=z.flatten(1)
        elif self.kind=='time_mean':z=self.temporal(z).mean(1)
        else:
            if self.kind=='multiscale':
                z=torch.cat([z,z.reshape(b,4,2,16).mean(2),z.reshape(b,2,4,16).mean(2)],1)
            z=self.temporal(z).mean(1)
        return self.head(z).squeeze(-1),None


class Population(nn.Module):
    def __init__(self,original,kind,graph=None,metadata=None):
        super().__init__();self.kind=kind;self.original=original
        tok=self.original.base.encoder.tokenizer
        self.mlp=kind in ['pooled_mlp','norm_mlp','state_mlp','masked_mlp','lowrank_mlp']
        if self.mlp:
            del self.original.readin
            del self.original.latents
            del self.original.base.encoder.transformer
            self.mix=nn.Sequential(nn.LayerNorm(32),nn.Linear(32,64),nn.GELU(),nn.Dropout(.05),nn.Linear(64,32))
            self.original.base.head=nn.Sequential(nn.Linear(32,256),nn.GELU(),nn.Linear(256,1))
        if kind in ['lowrank_id','lowrank_mlp','shared']:
            embedding=nn.Embedding(8192 if kind=='shared' else 2048,8)
            nn.init.normal_(embedding.weight,std=.02)
            tok.identity_embedding=nn.Sequential(embedding,nn.Linear(8,32,bias=False))
        if kind=='shared':self.session=nn.Embedding(4,32)
        if kind in ['token_norm','norm_mlp']:
            self.norm=nn.LayerNorm(32)
            self.amplitude=nn.Linear(16,1,bias=False)
            nn.init.zeros_(self.amplitude.weight)
        if kind in ['state_head','state_mlp']:
            self.state=nn.Linear(32,3)
            self.register_buffer('target_offset',torch.tensor(float(metadata['speed_mean']/metadata['speed_std'])))
        if kind in ['masked_pretrain','masked_mlp']:
            self.mask_token=nn.Parameter(torch.randn(32)*.02)
            self.reconstruct=nn.Sequential(nn.Linear(64,32),nn.GELU(),nn.Linear(32,8))
        if kind in ['functional_graph','graph_random']:
            self.register_buffer('neighbors',torch.as_tensor(graph,dtype=torch.long))
            self.message=nn.Sequential(nn.Linear(16,32),nn.GELU(),nn.Linear(32,32))
            self.gate=nn.Linear(16,1)

    def encode(self,x,subject=None,mask=None):
        ids=torch.arange(2048,device=x.device)
        if self.kind=='shared':
            if subject is None:raise ValueError('Shared model needs session labels')
            ids=ids[None]+2048*subject[:,None]
        tokenizer=self.original.base.encoder.tokenizer
        tokens=tokenizer(x,torch.zeros(2048,3,device=x.device),ids)
        if self.kind=='shared':tokens=tokens+self.session(subject)[:,None]
        if mask is not None:tokens=tokens+mask[:,:,None]*self.mask_token
        if self.kind in ['functional_graph','graph_random']:
            neighbor=x[:,self.neighbors].mean(2)
            pair=torch.cat([x,neighbor],-1)
            tokens=tokens+.1*torch.sigmoid(self.gate(pair))*self.message(pair)
        if self.kind in ['token_norm','norm_mlp']:tokens=self.norm(tokens)
        if self.mlp:return (tokens+self.mix(tokens)).mean(1)
        query=self.original.latents.expand(len(x),-1,-1)
        latent,_=self.original.readin(query,tokens,tokens,need_weights=False)
        return self.original.base.encoder.transformer(latent+query).mean(1)

    def forward(self,x,subject=None):
        z=self.encode(x,subject)
        if self.kind in ['state_head','state_mlp']:
            heads=self.state(z);gate=torch.sigmoid(heads[:,0])
            pred=(1-gate)*F.softplus(heads[:,1])+gate*F.softplus(heads[:,2])-self.target_offset
            return pred,heads[:,0]
        pred=self.original.base.head(z).squeeze(-1)
        if self.kind in ['token_norm','norm_mlp']:
            features=torch.cat([x.mean(1),x.std(1,unbiased=False)],-1)
            pred=pred+self.amplitude(features).squeeze(-1)
        return pred,None

    def pretrain_loss(self,x):
        mask=torch.rand(x.shape[:2],device=x.device)<.2
        masked=x.masked_fill(mask[:,:,None],0)
        z=self.encode(masked,mask=mask)
        ids=self.original.base.encoder.tokenizer.identity_embedding(torch.arange(2048,device=x.device))
        combined=torch.cat([z[:,None].expand(-1,2048,-1),ids[None].expand(len(x),-1,-1)],-1)
        predicted=self.reconstruct(combined)
        return ((predicted-x).square()*mask[:,:,None]).sum()/(mask.sum().clamp_min(1)*8)


def make(old,kind,seed,graph=None,metadata=None):
    torch.manual_seed(seed)
    if kind in ['time_attn','population_gru','population_conv','multiscale','time_mean','time_linear']:
        return Temporal(kind)
    original=old.model(seed,'global',torch.zeros(2048,dtype=torch.long).numpy())
    return Population(original,kind,graph,metadata)
