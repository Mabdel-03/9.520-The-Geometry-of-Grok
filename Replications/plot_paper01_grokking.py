#!/usr/bin/env python
"""
Visualize Paper 01 grokking results
Power et al. (2022) - The Original Grokking Paper
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load data
data_file = Path('01_power_et_al_2022_openai_grok/logs/training_history.json')
with open(data_file) as f:
    data = json.load(f)

epochs = np.array(data['epoch'])
train_acc = np.array([x if x is not None else 0 for x in data['train_acc']])
test_acc = np.array([min(x, 1.0) if x is not None else 0 for x in data['test_acc']])  # Cap at 100%
train_loss = np.array([x if x is not None else np.nan for x in data['train_loss']])
test_loss = np.array([x if x is not None else np.nan for x in data['test_loss']])

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Paper 01: Power et al. (2022) - Grokking: The Original Paper\n' +
             'Modular Addition (x+y mod 97)',
             fontsize=14, fontweight='bold')

# Plot 1: Accuracy over time
ax = axes[0, 0]
ax.plot(epochs, train_acc, 'b-', label='Train Accuracy', linewidth=2, alpha=0.8)
ax.plot(epochs, test_acc, 'r-', label='Test Accuracy', linewidth=2, alpha=0.8)

# Find when train hits 100%
train_100_idx = np.where(train_acc >= 0.99)[0]
if len(train_100_idx) > 0:
    train_100_epoch = epochs[train_100_idx[0]]
    ax.axvline(train_100_epoch, color='blue', linestyle='--', alpha=0.5, 
               label=f'Train→100% (step {train_100_epoch})')

ax.set_xlabel('Training Step', fontsize=11)
ax.set_ylabel('Accuracy', fontsize=11)
ax.set_title('Accuracy: Delayed Generalization (Grokking)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])

# Plot 2: Loss curves (log scale)
ax = axes[0, 1]
ax.semilogy(epochs, train_loss, 'b-', label='Train Loss', linewidth=2, alpha=0.8)
ax.semilogy(epochs, test_loss, 'r-', label='Test Loss', linewidth=2, alpha=0.8)
ax.set_xlabel('Training Step', fontsize=11)
ax.set_ylabel('Loss (log scale)', fontsize=11)
ax.set_title('Loss Curves', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')

# Plot 3: Generalization gap
ax = axes[1, 0]
gap = train_acc - test_acc
ax.plot(epochs, gap, 'purple', linewidth=2, alpha=0.8)
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.fill_between(epochs, 0, gap, where=(gap > 0), alpha=0.3, color='orange', label='Generalization gap')
ax.set_xlabel('Training Step', fontsize=11)
ax.set_ylabel('Train Acc - Test Acc', fontsize=11)
ax.set_title('Generalization Gap Over Time', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 4: Comparison to original paper
ax = axes[1, 1]
ax.plot(epochs, train_acc, 'b-', label='Train', linewidth=2.5, alpha=0.7)
ax.plot(epochs, test_acc, 'r-', label='Test', linewidth=2.5, alpha=0.7)
ax.set_xlabel('Training Step', fontsize=11)
ax.set_ylabel('Accuracy', fontsize=11)
ax.set_title('Replication of Original Grokking Paper', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])

# Add text box with metrics
train_max = np.nanmax(train_acc)
test_max = np.nanmax(test_acc)
test_final = test_acc[-1]

textstr = f'''Results:
Train: {train_max:.1%}
Test (max): {test_max:.1%}
Test (final): {test_final:.1%}
Steps: {epochs[-1]:,}

Original Paper:
Expected ~99% test
after grokking
'''
ax.text(0.02, 0.55, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Save
output_file = Path('analysis_results/paper_01_grokking.png')
output_file.parent.mkdir(exist_ok=True)
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Plot saved to: {output_file}")

plt.close()

print("\n" + "="*80)
print("PAPER 01 VISUALIZATION COMPLETE")
print("="*80)

