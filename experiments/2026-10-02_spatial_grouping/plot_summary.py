import json
import os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
r=json.loads((ROOT/'results.json').read_text())
p=json.loads((ROOT/'probe_results.json').read_text())
variants=r['protocol']['variants']
labels=['Unrestricted','Spatial patches','Random groups','Depth-matched\nrandom groups']
colors=['#566573','#2471A3','#C97722','#875A9D']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(12,8.2),layout='constrained')
ax=axes[0,0]
for i,v in enumerate(variants):
    values=np.array(r['groups'][v]['test_pair_mse']).flatten()
    ax.scatter(i+np.linspace(-.16,.16,len(values)),values,s=28,color=colors[i],alpha=.7)
    ax.plot([i-.24,i+.24],[values.mean()]*2,color=colors[i],lw=3)
ax.set(xticks=range(4),xticklabels=labels,ylabel='Test MSE (lower is better)',title='A. Main comparison: 12 matched fits per model')
ax.tick_params(axis='x',labelsize=9)
ax=axes[0,1]
for i,c in enumerate(['global','random','depth_random']):
    d=r['contrasts'][c]
    lo,hi=d['interval95'];m=d['mean_mse_improvement']
    ax.errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='o',color=colors[variants.index(c)],capsize=4)
    ax.scatter(list(d['pool_means'].values()),[i]*6,color=colors[variants.index(c)],alpha=.4,s=18)
ax.axvline(0,color='#777777',ls='--',lw=1)
ax.set(yticks=range(3),yticklabels=['vs unrestricted','vs random','vs depth-matched random'],
       xlabel='Control MSE minus spatial MSE\nPositive values favor spatial patches',title='B. Paired effects and conditional 95% intervals')
ax=axes[1,0]
ax.bar(range(4),[p['audit_summary'][v]['attention_rank'] for v in variants],color=colors)
ax.set(xticks=range(4),xticklabels=labels,ylabel='Attention participation rank',title='C. Grouping creates distinct attention patterns')
ax.tick_params(axis='x',labelsize=9)
ax=axes[1,1]
x=np.arange(4)
series=[('Original head',[r['groups'][v]['test_mse'] for v in variants],'#AAB7B8'),
        ('Refit averaged summaries',[p['groups'][v]['mean']['test_mse'] for v in variants],'#73A9CF'),
        ('Refit separate summaries',[p['groups'][v]['separate']['test_mse'] for v in variants],'#1F618D')]
for i,(label,values,color) in enumerate(series):
    ax.bar(x+(i-1)*.24,values,width=.23,label=label,color=color)
ax.set(xticks=x,xticklabels=labels,ylabel='Mean test MSE',title='D. Frozen-encoder output-layer probe')
ax.tick_params(axis='x',labelsize=9)
ax.set_ylim(0,.53)
ax.legend(fontsize=8,loc='upper left')
fig.suptitle('Spatial grouping in one mouse recording',fontsize=17)
fig.supxlabel('Six overlapping neuron samples × two training seeds; exploratory results on an already examined test period',fontsize=10)
fig.savefig(ROOT/'comparison.png',dpi=180)
fig.savefig(ROOT/'comparison.pdf')
plt.close(fig)
