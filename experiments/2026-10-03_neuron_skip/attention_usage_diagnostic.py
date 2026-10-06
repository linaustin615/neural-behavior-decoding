import hashlib,importlib.util,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits
root=Path(__file__).resolve().parent
base=root.parent/'2026-10-03_dynamics_baseline'
spec=importlib.util.spec_from_file_location('dyn',base/'models.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
rows=[]
with threadpool_limits(limits=2):
 torch.set_num_threads(2)
 for mouse in ['MP030','MP032','MP033','MP034']:
  data=np.load(base/mouse/'train_x.npy',mmap_mode='r');x=torch.from_numpy(np.array(data[np.linspace(0,len(data)-1,32,dtype=int)]))
  for seed in [10,11,12]:
   for state in ['random','pretrained','prior_finetuned','neuron_readout_finetuned']:
    if state in ['random','pretrained']:path=base/mouse/f'attention_pretrain_s{seed}'/('initial.pt' if state=='random' else 'selected.pt')
    elif state=='prior_finetuned':path=base/mouse/f'attention_pretrained_s{seed}'/'selected.pt'
    else:path=root/mouse/f'attention_s{seed}'/'selected.pt'
    weights=torch.load(path,weights_only=True)
    if state=='neuron_readout_finetuned':weights={k[len('encoder.'):]:v for k,v in weights.items() if k.startswith('encoder.')}
    net=mod.Dynamics('attention',seed);net.load_state_dict(weights);net.eval();metrics={};handles=[]
    for axis in ['temporal','population']:
     def hook(layer,args,kwargs,axis=axis):
      z=args[0];b,t,d=z.shape;h=layer.num_heads;dh=d//h
      q,k,_=F.linear(z,layer.in_proj_weight,layer.in_proj_bias).chunk(3,-1)
      q=q.reshape(b,t,h,dh).transpose(1,2);k=k.reshape(b,t,h,dh).transpose(1,2)
      scores=q@k.transpose(-1,-2)/(dh**.5)
      mask=kwargs.get('attn_mask')
      if mask is not None:scores=scores.masked_fill(mask,float('-inf'))
      probabilities=scores.softmax(-1)
      support=torch.ones((t,t),dtype=z.dtype) if mask is None else (~mask).to(z.dtype)
      uniform=support/support.sum(-1,keepdim=True)
      entropy=-(probabilities*probabilities.clamp_min(1e-30).log()).sum(-1)
      denom=support.sum(-1).log();valid=denom>0
      norm=(entropy[:,:,valid]/denom[valid]).mean()
      tv=(probabilities-uniform).abs().sum(-1).mul(.5).mean()
      metrics[axis]=dict(normalized_entropy=float(norm),total_variation_from_uniform=float(tv))
     handles.append(getattr(net,axis).mix.register_forward_pre_hook(hook,with_kwargs=True))
    with torch.inference_mode():net.encode(x)
    for handle in handles:handle.remove()
    assert all(torch.equal(weights[k],v) for k,v in net.state_dict().items())
    rows.append(dict(mouse=mouse,seed=seed,state=state,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),metrics=metrics))
summary={}
for state in ['random','pretrained','prior_finetuned','neuron_readout_finetuned']:
 summary[state]={axis:{metric:float(np.mean([r['metrics'][axis][metric] for r in rows if r['state']==state])) for metric in ['normalized_entropy','total_variation_from_uniform']} for axis in ['temporal','population']}
(root/'attention_usage.json').write_text(json.dumps(dict(scope='Post-hoc training-only audit; no causal or significance claim',contexts_per_mouse=32,checkpoint_states_checked=48,encoder_state_unchanged=True,summary=summary,rows=rows),indent=2)+'\n')
print(json.dumps(summary,indent=2))
