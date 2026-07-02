#!/usr/bin/env python3
"""Re-run analysis/plotting from saved data."""

import json
import os
import sys
import numpy as np
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture

OUT_DIR = '/data1/caoshuaihu.csh/workspace/dahd_speculative_decoding/results/multi_task_difficulty/'
TASK_NAMES = ['code', 'math', 'writing', 'translation']
K = 5

# Load data
task_metrics = {}
task_summaries = {}
task_confidences = {}

for task_name in TASK_NAMES:
    task_dir = os.path.join(OUT_DIR, task_name)
    metrics = []
    with open(os.path.join(task_dir, 'per_token_metrics.jsonl')) as f:
        for line in f:
            metrics.append(json.loads(line))
    task_metrics[task_name] = metrics
    with open(os.path.join(task_dir, 'per_step_summary.json')) as f:
        task_summaries[task_name] = json.load(f)
    task_confidences[task_name] = [m['confidence'] for m in metrics]

print(f"Loaded data for {len(task_metrics)} tasks")
for t in TASK_NAMES:
    print(f"  {t}: {len(task_metrics[t])} tokens, {len(task_summaries[t])} prompts")

colors = {'code': '#2196F3', 'math': '#FF5722', 'writing': '#4CAF50', 'translation': '#9C27B0'}
markers = {'code': 'o', 'math': 's', 'writing': '^', 'translation': 'D'}

# Plot 1: difficulty_distributions.png
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for idx, task_name in enumerate(TASK_NAMES):
    ax = axes[idx]
    confidences = np.array(task_confidences[task_name])
    ax.hist(confidences, bins=30, density=True, alpha=0.6, color=colors[task_name],
           edgecolor='black', linewidth=0.5, label=f'n={len(confidences)}')
    confs_2d = confidences.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=42).fit(confs_2d)
    x_plot = np.linspace(0, 1, 200).reshape(-1, 1)
    pdf = np.exp(gmm.score_samples(x_plot))
    ax.plot(x_plot, pdf, 'r-', linewidth=2, label='GMM (2-comp)')
    for i in range(2):
        mean = gmm.means_[i, 0]
        var = gmm.covariances_[i, 0, 0]
        weight = gmm.weights_[i]
        comp_pdf = weight * (1/np.sqrt(2*np.pi*var)) * np.exp(-0.5*(x_plot.flatten()-mean)**2/var)
        ax.plot(x_plot, comp_pdf, '--', linewidth=1.5, alpha=0.7,
               label=f'Comp {i+1}: \u03bc={mean:.2f}, w={weight:.2f}')
    ax.set_xlabel('Draft Confidence')
    ax.set_ylabel('Density')
    ax.set_title(f'{task_name.upper()}\nmean={confidences.mean():.3f}, easy={float((confidences>0.8).mean()):.1%}')
    ax.legend(fontsize=8)
    ax.set_xlim(-0.05, 1.05)
plt.suptitle('Token Difficulty Distribution by Task Type (Confidence)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'difficulty_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved difficulty_distributions.png")

# Plot 2: task_comparison.png
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
for task_name in TASK_NAMES:
    confs = np.array(task_confidences[task_name])
    ax.hist(confs, bins=30, density=True, alpha=0.3, color=colors[task_name],
           label=f'{task_name} (\u03bc={confs.mean():.3f})')
    kde = gaussian_kde(confs, bw_method=0.1)
    x_kde = np.linspace(0, 1, 200)
    ax.plot(x_kde, kde(x_kde), color=colors[task_name], linewidth=2)
ax.set_xlabel('Draft Confidence')
ax.set_ylabel('Density')
ax.set_title('Confidence Distribution Comparison')
ax.legend()
ax.set_xlim(-0.05, 1.05)

ax = axes[1]
task_acceptance_rates = []
task_labels = []
for task_name in TASK_NAMES:
    rates = [s['overall_acceptance_rate'] for s in task_summaries[task_name]]
    if rates:
        task_acceptance_rates.append(rates)
        task_labels.append(task_name)
bp = ax.boxplot(task_acceptance_rates, patch_artist=True)
ax.set_xticklabels(task_labels)
for i, (patch, tname) in enumerate(zip(bp['boxes'], task_labels)):
    patch.set_facecolor(colors.get(tname, 'gray'))
    patch.set_alpha(0.6)
ax.set_ylabel('Per-Prompt Acceptance Rate')
ax.set_title('Acceptance Rate Distribution by Task')
ax.set_ylim(0, 1.05)
for i, rates in enumerate(task_acceptance_rates):
    ax.scatter(i+1, np.mean(rates), marker='D', color='red', s=50, zorder=3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'task_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved task_comparison.png")

# Plot 3: per_position_by_task.png
fig, ax = plt.subplots(figsize=(10, 6))
for task_name in TASK_NAMES:
    metrics = task_metrics[task_name]
    position_accepted = defaultdict(list)
    for m in metrics:
        pos = m['position']
        if pos < K:
            position_accepted[pos].append(1 if m['accepted'] else 0)
    positions = list(range(K))
    rates = [np.mean(position_accepted[k]) if position_accepted[k] else 0 for k in positions]
    counts = [len(position_accepted[k]) for k in positions]
    ax.plot(positions, rates, marker=markers[task_name], color=colors[task_name],
           linewidth=2, markersize=8, label=f'{task_name} (n={sum(counts)})')
    ci_low, ci_high = [], []
    for k in positions:
        n, p = counts[k], rates[k]
        se = np.sqrt(p*(1-p)/n) if n > 0 else 0
        ci_low.append(max(0, p-1.96*se))
        ci_high.append(min(1, p+1.96*se))
    ax.fill_between(positions, ci_low, ci_high, color=colors[task_name], alpha=0.1)
ax.set_xlabel('Draft Position (k)', fontsize=12)
ax.set_ylabel('Acceptance Rate', fontsize=12)
ax.set_title('Per-Position Acceptance Rate by Task Type', fontsize=14)
ax.set_xticks(range(K))
ax.set_xticklabels([f'k={i+1}' for i in range(K)])
ax.set_ylim(0, 1.05)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'per_position_by_task.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved per_position_by_task.png")

# Plot 4: acceptance_length_by_task.png
fig, ax = plt.subplots(figsize=(10, 6))
width = 0.2
x = np.arange(K + 1)
for i, task_name in enumerate(TASK_NAMES):
    lengths = []
    for summ in task_summaries[task_name]:
        for step in summ['step_details']:
            lengths.append(step['n_accepted'])
    length_counts = np.zeros(K + 1)
    for l in lengths:
        if l <= K:
            length_counts[l] += 1
    length_dist = length_counts / length_counts.sum()
    ax.bar(x + i*width, length_dist, width, alpha=0.8, color=colors[task_name],
          label=f'{task_name} (\u03bc={np.mean(lengths):.2f})')
ax.set_xlabel('Acceptance Length (tokens accepted per step)', fontsize=12)
ax.set_ylabel('Proportion', fontsize=12)
ax.set_title('Acceptance Length Distribution by Task Type', fontsize=14)
ax.set_xticks(x + width*1.5)
ax.set_xticklabels([str(i) for i in range(K + 1)])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'acceptance_length_by_task.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved acceptance_length_by_task.png")

print("\nAll plots regenerated successfully!")
