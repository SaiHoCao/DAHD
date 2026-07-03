#!/usr/bin/env python3
"""Regenerate the e2e comparison chart with correct labels."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(SCRIPT_DIR, 'e2e_comparison_v2.json')) as f:
    data = json.load(f)

summary = data['summary']
methods = ['vanilla', 'eagle3', 'parallel', 'dahd']
labels = ['vanilla', 'eagle3', 'parallel\n(gumiho)', 'dahd']
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']

tps = [summary[m]['avg_tokens_per_sec'] for m in methods]
speedup = [summary[m]['speedup_vs_vanilla'] for m in methods]
acc = [summary[m]['avg_accepted_per_step'] for m in methods]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. Tokens/sec
bars = axes[0].bar(labels, tps, color=colors, alpha=0.85)
axes[0].set_ylabel('Tokens per second')
axes[0].set_title('Tokens/sec')
axes[0].set_ylim(0, max(tps) * 1.15)

# 2. Speedup vs Vanilla AR
bars = axes[1].bar(labels, speedup, color=colors, alpha=0.85)
axes[1].axhline(y=1.0, color='red', linestyle='--', label='baseline', alpha=0.7)
axes[1].set_ylabel('Speedup (x)')
axes[1].set_title('Speedup vs Vanilla AR')
axes[1].set_ylim(0, max(speedup) * 1.15)
axes[1].legend()

# 3. Avg Draft Acceptance per Step
acc_methods = ['eagle3', 'parallel', 'dahd']
acc_labels = ['eagle3', 'parallel\n(gumiho)', 'dahd']
acc_colors = ['#2ca02c', '#ff7f0e', '#9467bd']
acc_vals = [summary[m]['avg_accepted_per_step'] for m in acc_methods]
bars = axes[2].bar(acc_labels, acc_vals, color=acc_colors, alpha=0.85)
axes[2].set_ylabel('Accepted tokens')
axes[2].set_title('Avg Draft Acceptance per Step')
axes[2].set_ylim(0, max(acc_vals) * 1.15)

plt.tight_layout()
out_path = os.path.join(SCRIPT_DIR, 'e2e_comparison_chart.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Chart saved to {out_path}')
