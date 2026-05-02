"""Generate publication-ready figures from Indian-Data results."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import spearmanr
from pathlib import Path

OUT = Path('figures')
OUT.mkdir(exist_ok=True)

# Publication style
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times', 'Times New Roman', 'DejaVu Serif']
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['axes.titlesize'] = 11
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'

df = pd.read_csv('results_per_doc.csv')
summary = json.load(open('summary_results.json'))

# ============================================================================
# Figure 1: Threshold sweep curves
# ============================================================================
# We don't have the sweep stored in summary_results.json; reconstruct from notebook output
# Values from your run: τ=0.40 P=0.671 R=0.958 F1=0.783 ... etc.
sweep = {
    0.40: (0.671, 0.958, 0.783),
    0.45: (0.585, 0.923, 0.708),
    0.50: (0.505, 0.883, 0.632),
    0.55: (0.428, 0.816, 0.550),
    0.60: (0.361, 0.741, 0.474),
    0.65: (0.298, 0.655, 0.396),
    0.70: (0.244, 0.563, 0.327),
    0.75: (0.194, 0.453, 0.257),
    0.80: (0.162, 0.376, 0.212),
}
ts = list(sweep.keys())
ps = [sweep[t][0] for t in ts]
rs = [sweep[t][1] for t in ts]
f1s = [sweep[t][2] for t in ts]

fig, ax = plt.subplots(figsize=(5.0, 3.5))
ax.plot(ts, ps, 'o-', label='Precision', color='#1f77b4', lw=1.4, ms=4)
ax.plot(ts, rs, 's-', label='Recall',    color='#2ca02c', lw=1.4, ms=4)
ax.plot(ts, f1s, '^-', label='F1',       color='#d62728', lw=1.6, ms=5)
ax.axvline(0.45, color='gray', linestyle='--', lw=0.8, alpha=0.7)
ax.text(0.452, 0.15, r'selected $\tau = 0.45$', rotation=90, fontsize=8, color='gray', va='bottom')
ax.set_xlabel(r'Similarity threshold $\tau$')
ax.set_ylabel('Score')
ax.set_xticks(ts)
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.25, linestyle=':')
ax.legend(loc='upper right', frameon=True, framealpha=0.9)
fig.tight_layout()
fig.savefig(OUT / 'fig_threshold_sweep.pdf')
fig.savefig(OUT / 'fig_threshold_sweep.png')
plt.close()
print('Saved: fig_threshold_sweep')

# ============================================================================
# Figure 2: Correlation matrix heatmap
# ============================================================================
metrics = ['BLEU', 'ROUGE-L', 'OrigIntent_F1', 'SIM_F1', 'LLM_Judge']
labels  = ['BLEU', 'ROUGE-L', 'Orig.Intent', 'SIM (ours)', 'LLM-Judge']
M = np.zeros((5, 5))
for i, m1 in enumerate(metrics):
    for j, m2 in enumerate(metrics):
        rho, _ = spearmanr(df[m1].values, df[m2].values)
        M[i, j] = rho

fig, ax = plt.subplots(figsize=(4.7, 4.2))
im = ax.imshow(M, vmin=-1, vmax=1, cmap='RdBu_r', aspect='equal')
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.set_xticklabels(labels, rotation=35, ha='right')
ax.set_yticklabels(labels)
for i in range(5):
    for j in range(5):
        v = M[i, j]
        color = 'white' if abs(v) > 0.5 else 'black'
        ax.text(j, i, f'{v:+.2f}', ha='center', va='center', fontsize=9, color=color)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Spearman ρ', fontsize=9)
fig.tight_layout()
fig.savefig(OUT / 'fig_correlation_matrix.pdf')
fig.savefig(OUT / 'fig_correlation_matrix.png')
plt.close()
print('Saved: fig_correlation_matrix')

# ============================================================================
# Figure 3: Per-category SIM vs OrigIntent comparison (grouped bars)
# ============================================================================
cats = sorted(df['category'].unique())
orig_means = [df[df['category']==c]['OrigIntent_F1'].mean() for c in cats]
sim_means  = [df[df['category']==c]['SIM_F1'].mean() for c in cats]
counts     = [len(df[df['category']==c]) for c in cats]

x = np.arange(len(cats))
w = 0.35

fig, ax = plt.subplots(figsize=(5.5, 3.5))
b1 = ax.bar(x - w/2, orig_means, w, label='Original Intent (Mullick et al., 2022)', color='#a6cee3', edgecolor='black', lw=0.5)
b2 = ax.bar(x + w/2, sim_means, w, label='SIM (ours)', color='#1f77b4', edgecolor='black', lw=0.5)
for i, (o, s) in enumerate(zip(orig_means, sim_means)):
    ax.text(x[i] - w/2, o + 0.015, f'{o:.2f}', ha='center', va='bottom', fontsize=8)
    ax.text(x[i] + w/2, s + 0.015, f'{s:.2f}', ha='center', va='bottom', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([f'{c}\n(n={n})' for c, n in zip(cats, counts)])
ax.set_ylabel('F1')
ax.set_ylim(0, 0.95)
ax.legend(loc='upper left', frameon=True, framealpha=0.9)
ax.grid(True, axis='y', alpha=0.25, linestyle=':')
fig.tight_layout()
fig.savefig(OUT / 'fig_category_comparison.pdf')
fig.savefig(OUT / 'fig_category_comparison.png')
plt.close()
print('Saved: fig_category_comparison')

# ============================================================================
# Figure 4: Scatter — SIM_F1 vs LLM-Judge with category coloring
# ============================================================================
fig, ax = plt.subplots(figsize=(5.0, 3.6))
colors = {'Murder': '#e41a1c', 'Robbery': '#377eb8', 'Land Dispute': '#4daf4a', 'Corruption': '#984ea3'}
for c in cats:
    sub = df[df['category']==c]
    # add small jitter to LLM_Judge (integer values) for visibility
    jitter = np.random.RandomState(0).uniform(-0.08, 0.08, size=len(sub))
    ax.scatter(sub['SIM_F1'].values, sub['LLM_Judge'].values + jitter,
               label=c, color=colors[c], s=22, alpha=0.75, edgecolor='black', lw=0.4)
rho, p = spearmanr(df['SIM_F1'].values, df['LLM_Judge'].values)
ax.set_xlabel('SIM F1')
ax.set_ylabel('LLM-as-Judge score')
ax.set_yticks([3, 4, 5])
ax.text(0.05, 0.95, fr'Spearman $\rho = +{rho:.3f}$ (p = {p:.3f})',
        transform=ax.transAxes, fontsize=9, va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))
ax.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=8)
ax.grid(True, alpha=0.25, linestyle=':')
fig.tight_layout()
fig.savefig(OUT / 'fig_sim_vs_judge.pdf')
fig.savefig(OUT / 'fig_sim_vs_judge.png')
plt.close()
print('Saved: fig_sim_vs_judge')

print('\nAll figures saved to figures/')
