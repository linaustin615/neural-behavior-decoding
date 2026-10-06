"""Preserve frozen backbones while adapting recording-table size."""
import importlib.util
import math
import torch
from torch import nn
from common import ROOT

path=ROOT.parent/'2026-10-06_systematic_optimization'/'models.py'
spec=importlib.util.spec_from_file_location('frozen_population_parent',path)
parent=importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)


class Decoder(parent.PopulationDecoder):
    def __init__(self,config,seed,mouse=None):
        super().__init__(config,seed)
        torch.manual_seed(seed+3000)
        extra=torch.randn(3,512,config['width'])/math.sqrt(512)
        extra_session=torch.randn(3,config['width'])*.02
        readin=torch.cat([self.readin.detach(),extra])
        session=torch.cat([self.session.detach(),extra_session])
        if mouse is not None:
            assert mouse in range(7)
            readin=readin[mouse:mouse+1].clone()
            session=session[mouse:mouse+1].clone()
        self.readin=nn.Parameter(readin)
        self.session=nn.Parameter(session)

    def encode(self,x,session):
        c=self.config
        assert x.ndim==3 and x.shape[1:]==(512,c['history']) and session in range(len(self.session))
        z=x.transpose(1,2)@self.readin[session]
        z=self.patch(z.reshape(len(x),c['history']//c['patch'],c['patch']*c['width']))
        z=z+self.time[None]+self.session[session]
        for layer in self.layers:
            z=layer(z)
        return z
