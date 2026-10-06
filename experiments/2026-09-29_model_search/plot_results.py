import json
import os
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(root/'.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

r=json.loads((root/'final_results.json').read_text())
names=['original_constant','combo_cosine','latent_n256','latent_n512','latent_n512_no_positions','ridge_512_w8']
labels=['Current\n128 neurons','MLP + readout\n128 neurons','Latent + xyz\n256 neurons','Latent + xyz\n512 neurons','Selected latent\n512 neurons','Tuned ridge\n512 neurons']
fig,axes=plt.subplots(1,2,figsize=(13,4.8),sharey=True,constrained_layout=True)
colors=['#84919f','#84919f','#619fbc','#3c91aa','#007f73','#ab895f']
for ax,split,title in zip(axes,['validation','test'],['Validation: used to select models','Reserved test segment: evaluated after selection']):
    means=[r['groups'][n]['mean_single_model'][split]['mse'] for n in names]
    std=[r['groups'][n]['mean_single_model'][split]['mse_sample_std'] for n in names]
    ax.bar(np.arange(len(names)),means,yerr=std,capsize=3,color=colors,alpha=.8)
    for i,name in enumerate(names):
        values=[v[split]['mse'] for v in r['groups'][name]['runs']]
        ax.scatter(i+np.linspace(-.12,.12,len(values)),values,s=16,c='black',zorder=4)
    ax.set_xticks(np.arange(len(names)),labels,fontsize=8)
    ax.set_title(title,fontsize=11)
    ax.grid(axis='y',alpha=.18)
    ax.set_axisbelow(True)
axes[0].set_ylabel('Normalized MSE (lower is better)')
fig.suptitle('Five seeds per neural model; bars show means, whiskers show sample SD',fontsize=12)
fig.savefig(root/'comparison.png',dpi=180)
plt.close(fig)
mean,std=r['speed_scaler']['mean'],r['speed_scaler']['std']
selected=r['selection']['selected_neural']
data=np.load(root/f'final_{selected}_predictions.npz')
true=data['test_targets']*std+mean
prediction=np.maximum(data['test_predictions'].mean(0)*std+mean,0)
old=np.load(root/'final_original_constant_predictions.npz')
original=np.maximum(old['test_predictions'].mean(0)*std+mean,0)
fig=plt.figure(figsize=(12,8),constrained_layout=True)
grid=fig.add_gridspec(2,2)
ax=fig.add_subplot(grid[0,:])
ax.plot(true,color='#202830',lw=1.3,label='Recorded speed')
ax.plot(original,color='#b1b8bf',lw=1,alpha=.7,label='Current model ensemble')
ax.plot(prediction,color='#007f73',lw=1,label='Selected model ensemble')
ax.set_xlabel('Test time bin (entire evaluated segment)')
ax.set_ylabel('Speed (dataset units)')
ax.legend(frameon=False,ncol=3,fontsize=9)
ax.set_title('Predictions from five-model ensembles; training and selection use earlier time segments')
ax=fig.add_subplot(grid[1,0])
ax.scatter(true,prediction,s=7,alpha=.35,color='#007f73',rasterized=True)
limit=float(max(true.max(),prediction.max()))
ax.plot([0,limit],[0,limit],'--',color='#65737e',lw=1)
ax.set_xlabel('Recorded speed')
ax.set_ylabel('Predicted speed')
ax.set_title('Large speeds still produce large errors')
ax=fig.add_subplot(grid[1,1])
masks=[true<1e-5,(true>=1e-5)&(true<5),(true>=5)&(true<15),true>=15]
original_errors=[np.abs(original[m]-true[m]).mean() for m in masks]
selected_errors=[np.abs(prediction[m]-true[m]).mean() for m in masks]
x=np.arange(4)
ax.bar(x-.18,original_errors,width=.36,color='#a4afb9',label='Current ensemble')
ax.bar(x+.18,selected_errors,width=.36,color='#007f73',label='Selected ensemble')
ax.set_xticks(x,[f'Zero\n(n={masks[0].sum()})',f'0–5\n(n={masks[1].sum()})',f'5–15\n(n={masks[2].sum()})',f'15+\n(n={masks[3].sum()})'])
ax.set_ylabel('Mean absolute error (dataset units)')
ax.set_xlabel('Recorded speed group')
ax.legend(frameon=False,fontsize=9)
fig.savefig(root/'predictions.png',dpi=180)
plt.close(fig)
print('Saved comparison.png and predictions.png')
