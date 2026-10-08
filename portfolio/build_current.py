"""Build the current presentation from audited aggregate records; no training or inference."""
import csv
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/neural-presentation-mpl')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/neural-presentation-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results'
MODELS = ['ridge', 'mlp', 'transformer', 'blend', 'corrected']
LABELS = ['Ridge regression', 'MLP', 'Transformer', 'MLP + transformer', 'Mix + correction*']
COLORS = ['#a8b3c7', '#f4c67a', '#81b4f5', '#6fe3c4', '#f0a882']
BG, FG, MUTED, GRID = '#0b1420', '#edf3fa', '#a6b6c8', '#26384a'


def setup():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
        'figure.facecolor': BG, 'axes.facecolor': BG, 'savefig.facecolor': BG,
        'text.color': FG, 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': FG,
        'axes.edgecolor': GRID, 'svg.fonttype': 'none', 'pdf.fonttype': 42})


def save(fig, name):
    for extension in ('png', 'svg'):
        fig.savefig(OUT/f'{name}.{extension}', dpi=170)
    plt.close(fig)


def clean(ax):
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, pad=10)
    ax.set_axisbelow(True)


def heading(fig, eyebrow, title, subtitle):
    fig.text(.055, .95, eyebrow, color=COLORS[3], fontsize=10, weight='bold')
    fig.text(.055, .895, title, fontsize=23, weight='bold')
    fig.text(.055, .85, subtitle, color=MUTED, fontsize=10)


def main():
    setup(); OUT.mkdir(exist_ok=True)
    source = json.loads((ROOT/'data/current_reference.json').read_text())
    mice = source['cohort']; rows = source['rows']
    means = {}
    for model in MODELS:
        for mouse in mice:
            subset = [r for r in rows if r['model']==model and r['mouse']==mouse]
            assert len(subset)==(1 if model=='ridge' else 3)
            means[model, mouse] = {k: float(np.mean([r[k] for r in subset])) for k in ('mse', 'mae', 'r2', 'qfm', 'active_mse')}
    ratios = np.array([[means[model, m]['mse']/means['ridge', m]['mse'] for m in mice] for model in MODELS])
    figures = {}
    fig = plt.figure(figsize=(13, 7.4))
    heading(fig, 'FOUR SENSORIMOTOR MICE  /  SAVED TEST RESULTS', 'What each model contributes',
            'Same neurons and test frames. Ridge tuning uses validation only. Lower error is better.')
    ax = fig.add_axes([.21, .24, .36, .52]); clean(ax)
    y = np.arange(5)
    ax.barh(y, ratios.mean(1), height=.56, color=COLORS, alpha=.9)
    for i, values in enumerate(ratios):
        ax.scatter(values, i+np.array([-.13, -.045, .045, .13]), s=25, color=BG, edgecolors=FG, linewidth=.7, zorder=3)
        gain = 100*(1-values.mean())
        ax.text(1.055, i, 'reference' if i==0 else f'{gain:.1f}% less', va='center', fontsize=10)
    ax.set_yticks(y, LABELS); ax.invert_yaxis(); ax.set_xlim(0, 1.34)
    ax.set_xticks([0, .25, .5, .75, 1]); ax.xaxis.grid(True, color=GRID)
    ax.set_xlabel('MSE relative to tuned ridge  ·  dots = individual mice', fontsize=9, labelpad=14)
    ax.set_title('Average error relative to ridge', loc='left', fontsize=12, pad=18)
    ax = fig.add_axes([.69, .24, .265, .52]); clean(ax)
    values = np.array([[means[model, m]['r2'] for m in mice] for model in MODELS])
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('mint', ['#142332', '#23565a', '#459b8a'])
    ax.imshow(values, vmin=0, vmax=1, cmap=cmap, aspect='auto')
    for i in range(5):
        for j in range(4): ax.text(j, i, f'{values[i,j]:.3f}', ha='center', va='center', fontsize=10)
    ax.set_xticks(range(4), mice); ax.set_yticks([])
    ax.set_title('R² by mouse · higher is better', loc='left', fontsize=12, pad=18)
    ax.set_xlabel('Same model order as left panel', fontsize=9, labelpad=14)
    fig.text(.055, .115, '*Correction is experimental: only 1.16% lower MSE than the mix; it missed its frozen improvement gate.', color=COLORS[4], fontsize=10)
    fig.text(.055, .065, 'Seed-mean errors per mouse; mice weighted equally. Dots are four mice, not confidence intervals.\nMix vs parents: pre-registered holdout study. Ridge comparison and this synthesis: post hoc. One lab; no transfer claim.', color=MUTED, fontsize=9, linespacing=1.6)
    save(fig, 'current_models')
    figures['mse_gain_vs_ridge_percent'] = {model: float(100*(1-ratios[i].mean())) for i, model in enumerate(MODELS)}

    fig, axes = plt.subplots(1, 3, figsize=(13, 6.3))
    fig.subplots_adjust(left=.09, right=.97, bottom=.24, top=.72, wspace=.45)
    heading(fig, 'CORRECTION ABLATION  /  SAME FOUR MICE', 'Quieter predictions come with a cost',
            'Original learned correction versus the plain mix. Positive values mean a reduction; zero means no change.')
    for ax, key, title in zip(axes, ['mse', 'qfm', 'active_mse'], ['Overall MSE', 'Mean quiet predicted speed', 'Active-period MSE']):
        gains = np.array([100*(1-means['corrected', m][key]/means['blend', m][key]) for m in mice])
        clean(ax); ax.axvline(0, color=MUTED, lw=1)
        ax.barh(range(4), gains, color=[COLORS[3] if g>=0 else COLORS[4] for g in gains], height=.53)
        ax.set_yticks(range(4), mice); ax.invert_yaxis()
        span = max(gains.max()-gains.min(), 2)
        ax.set_xlim(min(gains.min(), 0)-span*.48, max(gains.max(), 0)+span*.4)
        for i, g in enumerate(gains):
            ax.text(g+(.08*span if g>=0 else -.08*span), i, f'{g:+.1f}%', ha='left' if g>=0 else 'right', va='center', fontsize=10)
        ax.set_title(title, fontsize=12, loc='left', pad=16)
        ax.set_xlabel('Reduction versus mix (%)', fontsize=9)
        figures[key+'_correction_gain_percent'] = dict(zip(mice, gains.tolist()))
    fig.text(.055, .12, '53.08% lower mean quiet predicted speed ≠ 53.08% lower overall error.', fontsize=12, color=COLORS[4], weight='bold')
    fig.text(.055, .057, 'Quiet: observed speed ≤0.05 training SD. Active: ≥0.5 SD. Quiet suppression is not a classification false-positive rate.\nOverall gain: 1.16%; active MSE can worsen by 3.29%. Later confidence gating did not resolve this trade-off.', color=MUTED, fontsize=9, linespacing=1.6)
    save(fig, 'correction_tradeoff')

    fig = plt.figure(figsize=(13, 6.4)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 13); ax.set_ylim(0, 6.4); ax.axis('off')
    heading(fig, 'MODEL DESIGN  /  ASSOCIATION, NOT CAUSAL SIMULATION', 'Two decoders, one running-speed estimate',
            'The mix is the main result. The small correction is an optional research branch.')
    def box(x, y, w, h, title, detail, color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.12,rounding_size=0.14',facecolor='#132333',edgecolor=color,lw=1.3))
        ax.text(x+.14,y+h-.3,title,color=color,weight='bold',fontsize=12,va='top')
        ax.text(x+.14,y+h-.69,detail,color=FG,fontsize=10,va='top',linespacing=1.5)
    def arrow(a,b,color=MUTED,style='-'):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,color=color,lw=1.4,linestyle=style,connectionstyle='arc3,rad=0'))
    box(.65,2.2,2.4,1.4,'Recorded activity','512 neurons\n32 native-frame history',COLORS[0])
    box(3.75,3.25,2.5,1.2,'Transformer','Population read-in\nTemporal attention + MLP',COLORS[2])
    box(3.75,1.4,2.5,1.2,'Population MLP','Patch representation\nNonlinear speed readout',COLORS[1])
    box(7.05,2.2,2.35,1.4,'Weighted mix','α × transformer\n+ (1 − α) × MLP',COLORS[3])
    box(10.05,2.2,2.25,1.4,'Predicted speed','Same-frame decoding\nClipped at physical zero',COLORS[3])
    arrow((3.18,3.1),(3.62,3.75)); arrow((3.18,2.6),(3.62,2.0))
    arrow((6.38,3.75),(6.92,3.1)); arrow((6.38,2.0),(6.92,2.6)); arrow((9.53,2.9),(9.92,2.9),COLORS[3])
    ax.text(7.1,1.65,'Parents clipped at zero before mixing\nα chosen using validation only',fontsize=9,color=MUTED,linespacing=1.5)
    ax.text(.65,.78,'OPTIONAL CORRECTION',fontsize=10,color=COLORS[4],weight='bold')
    ax.text(3.45,.78,'Activity + mix → small attention branch → bounded, movement-gated adjustment',fontsize=10,color=FG)
    ax.text(.65,.3,'Coordinates appear only in the 3D illustration. Ridge is the separately tuned linear control. Every model is fitted within its recording.',fontsize=9,color=MUTED)
    save(fig, 'combined_architecture')

    with (OUT/'current_models.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['mouse', 'model', 'mse', 'mae', 'r2', 'qfm', 'active_mse', 'mse_gain_vs_ridge'])
        writer.writeheader()
        for i, model in enumerate(MODELS):
            for j, mouse in enumerate(mice): writer.writerow(dict(mouse=mouse, model=model, **means[model,mouse], mse_gain_vs_ridge=1-ratios[i,j]))
    (OUT/'current_figures.json').write_text(json.dumps(figures, indent=2)+'\n')
    print(json.dumps(figures, indent=2))


if __name__=='__main__':
    main()
