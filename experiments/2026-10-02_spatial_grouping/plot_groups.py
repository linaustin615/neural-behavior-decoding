from pathlib import Path
import os
import numpy as np
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import architecture_search as q
import base_search as s
from group_model import grouping

ROOT=Path(__file__).resolve().parent
q.initialize()
raw=s.POS[q.POOL_IDS[101]]
spatial,boundaries=grouping(raw,101,'spatial')
threshold=boundaries[0]['threshold']
fig,axes=plt.subplots(2,3,figsize=(11,7),layout='constrained',sharex=True,sharey=True)
for col,(variant,label) in enumerate([('spatial','Spatial patches'),('random','Random groups'),('depth_random','Depth-matched random groups')]):
    groups,_=grouping(raw,101,variant)
    for row in range(2):
        ix=(raw[:,2]<=threshold) if row==0 else (raw[:,2]>threshold)
        axes[row,col].scatter(raw[ix,0],raw[ix,1],c=groups[ix],cmap='tab10',vmin=0,vmax=9,s=7,alpha=.8,rasterized=True)
        axes[row,col].set_aspect('equal')
        axes[row,col].set_title(label if row==0 else '')
        axes[row,col].set_xlabel('Recorded coordinate 1')
        axes[row,col].set_ylabel(('Deeper imaging planes\n' if row==0 else 'Shallower imaging planes\n')+'Recorded coordinate 2')
fig.suptitle('The same 2,048 neurons assigned to eight summary groups',fontsize=16)
fig.supxlabel('Colors identify groups. Both random controls preserve group sizes; the depth control also preserves group-by-plane counts.',fontsize=9)
fig.savefig(ROOT/'group_assignments.png',dpi=180)
plt.close(fig)
