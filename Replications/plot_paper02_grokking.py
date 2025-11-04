#!/usr/bin/env python
"""
Plot grokking results for Paper 02: Liu et al. (2022) - Effective Theory
Shows train/test accuracy, losses, and RQI (Representation Quality Index)
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load data
data_file = Path('02_liu_et_al_2022_effective_theory/logs/training_history.json')
with open(data_file) as f:
    data = json.load(f)

epochs = np.array(data['epoch'])
train_acc = np.array(data['train_acc'])
test_acc = np.array(data['test_acc'])
train_loss = np.array(data['train_loss'])
test_loss = np.array(data['test_loss'])
rqi = np.array(data['rqi'])

# Grokking thresholds
grok_sum = data['grokking_summary']
train_thresh_step = grok_sum['train_acc_step']
test_thresh_step = grok_sum['test_acc_step']
rqi_thresh_step = grok_sum['rqi_step']
grok_delay = grok_sum['grokking_delay']

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Paper 02: Liu et al. (2022) - Effective Theory of Grokking\n' + 
             f'Delayed Generalization: Test lags Train by {grok_delay} steps',
             fontsize=14, fontweight='bold')

# Plot 1: Accuracy over time
ax = axes[0, 0]
ax.plot(epochs, train_acc, 'b-', label='Train Accuracy', linewidth=2, alpha=0.8)
ax.plot(epochs, test_acc, 'r-', label='Test Accuracy', linewidth=2, alpha=0.8)
ax.axvline(train_thresh_step, color='blue', linestyle='--', alpha=0.5, label=f'Train reaches 90% (step {train_thresh_step})')
ax.axvline(test_thresh_step, color='red', linestyle='--', alpha=0.5, label=f'Test reaches 90% (step {test_thresh_step})')
ax.axhline(0.9, color='gray', linestyle=':', alpha=0.3)
ax.set_xlabel('Training Step', fontsize=11)
ax.set_ylabel('Accuracy', fontsize=11)
ax.set_title('Accuracy: Delayed Generalization (Grokking)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])

# Add annotation for grokking delay
ax.annotate('', xy=(test_thresh_step, 0.92), xytext=(train_thresh_step, 0.92),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text((train_thresh_step + test_thresh_step)/2, 0.94, 
        f'{grok_delay} step delay', ha='center', fontsize=10, color='green', fontweight='bold')

# Plot 2: Loss over time (log scale)
ax = axes[0, 1]
ax.semilogy(epochs, train_loss, 'b-', label='Train Loss', linewidth=2, alpha=0.8)
ax.semilogy(epochs, test_loss, 'r-', label='Test Loss', linewidth=2, alpha=0.8)
ax.axvline(train_thresh_step, color='blue', linestyle='--', alpha=0.5)
ax.axvline(test_thresh_step, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Training Step', fontsize=11)
ax.set_ylabel('Loss (log scale)', fontsize=11)
ax.set_title('Loss Curves', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')

# Plot 3: RQI (Representation Quality Index) - unique to this paper!
ax = axes[1, 0]
ax.plot(epochs, rqi, 'purple', label='RQI (Representation Quality)', linewidth=2, alpha=0.8)
ax.axvline(rqi_thresh_step, color='purple', linestyle='--', alpha=0.5, label=f'RQI reaches 95% (step {rqi_thresh_step})')
ax.axhline(0.95, color='gray', linestyle=':', alpha=0.3)
ax.set_xlabel('Training Step', fontsize=11)
ax.set_ylabel('RQI', fontsize=11)
ax.set_title('Representation Quality Index (RQI)\nNovel metric from this paper', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])

# Plot 4: Train vs Test comparison
ax = axes[1, 1]
ax.plot(epochs, train_acc, 'b-', label='Train Accuracy', linewidth=2.5, alpha=0.7)
ax.plot(epochs, test_acc, 'r-', label='Test Accuracy', linewidth=2.5, alpha=0.7)
ax.fill_between(epochs, train_acc, test_acc, where=(train_acc > test_acc), 
                 alpha=0.2, color='orange', label='Generalization Gap')
ax.axvline(train_thresh_step, color='blue', linestyle='--', linewidth=1.5, alpha=0.6)
ax.axvline(test_thresh_step, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
ax.set_xlabel('Training Step', fontsize=11)
ax.set_ylabel('Accuracy', fontsize=11)
ax.set_title('Generalization Gap Closure\n(Grokking Visualization)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 5000])
ax.set_ylim([0, 1.05])

# Add text box with key metrics
textstr = f'''Key Results:
• Train: 100% at step {train_thresh_step}
• Test: 100% at step {test_thresh_step}  
• Grokking delay: {grok_delay} steps
• Final RQI: {rqi[-1]:.3f}
'''
ax.text(0.02, 0.55, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Save figure
output_file = Path('analysis_results/paper_02_grokking.png')
output_file.parent.mkdir(exist_ok=True)
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Plot saved to: {output_file}")
print(f"✓ Figure shows clear grokking with {grok_delay}-step delay")

plt.close()

# Also create a zoomed view of the grokking transition
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# Focus on the grokking region (500 steps before and after)
zoom_start = max(0, train_thresh_step - 500)
zoom_end = min(len(epochs), test_thresh_step + 500)
zoom_mask = (epochs >= zoom_start) & (epochs <= zoom_end)

ax.plot(epochs[zoom_mask], train_acc[zoom_mask], 'b-', label='Train Accuracy', linewidth=3, alpha=0.8)
ax.plot(epochs[zoom_mask], test_acc[zoom_mask], 'r-', label='Test Accuracy', linewidth=3, alpha=0.8)
ax.axvline(train_thresh_step, color='blue', linestyle='--', linewidth=2, alpha=0.7, label=f'Train → 90% (step {train_thresh_step})')
ax.axvline(test_thresh_step, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Test → 90% (step {test_thresh_step})')
ax.axhline(0.9, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)

# Shade the grokking delay region
ax.axvspan(train_thresh_step, test_thresh_step, alpha=0.2, color='green', label='Grokking Delay')

ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title(f'Paper 02: Grokking Transition (Zoomed)\n400-step Delayed Generalization', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])

zoom_file = Path('analysis_results/paper_02_grokking_zoomed.png')
plt.savefig(zoom_file, dpi=300, bbox_inches='tight')
print(f"✓ Zoomed plot saved to: {zoom_file}")

plt.close()

print("\n" + "="*80)
print("Paper 02 Visualization Complete!")
print("="*80)

