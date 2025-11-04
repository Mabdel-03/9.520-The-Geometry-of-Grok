#!/usr/bin/env python3
"""
Enhanced plotting to highlight grokking transitions
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Focus on Paper 03 which shows clear grokking
print("Analyzing Paper 03 - Nanda et al. (2023)")
print("="*80)

json_file = "03_nanda_et_al_2023_progress_measures/logs/training_history.json"

with open(json_file, 'r') as f:
    data = json.load(f)

epochs = np.array(data['epoch'])
train_loss = np.array(data['train_loss'])
test_loss = np.array(data['test_loss'])
train_acc = np.array(data['train_acc'])
test_acc = np.array(data['test_acc'])

# Find grokking transitions (large jumps in test accuracy)
diffs = np.diff(test_acc)
large_jumps = np.where(diffs > 0.1)[0]

print(f"\nFound {len(large_jumps)} major grokking transitions:")
for idx in large_jumps:
    print(f"  Epoch {epochs[idx]:>6} → {epochs[idx+1]:>6}: "
          f"Test Acc {test_acc[idx]:.1%} → {test_acc[idx+1]:.1%} "
          f"(+{diffs[idx]:.1%})")

# Create enhanced visualization
fig = plt.figure(figsize=(18, 12))

# Plot 1: Full training history with transitions marked
ax1 = plt.subplot(3, 2, 1)
ax1.plot(epochs, train_acc, label='Train Accuracy', color='blue', linewidth=2, alpha=0.7)
ax1.plot(epochs, test_acc, label='Test Accuracy', color='red', linewidth=2.5, alpha=0.9)
for idx in large_jumps:
    ax1.axvline(epochs[idx], color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax1.annotate(f'+{diffs[idx]:.0%}', 
                xy=(epochs[idx], test_acc[idx]), 
                xytext=(10, 10), textcoords='offset points',
                fontsize=8, color='green',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.set_title('Full Training History\n(Green lines = Grokking transitions)', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1.05])

# Plot 2: Loss curves (log scale)
ax2 = plt.subplot(3, 2, 2)
ax2.plot(epochs, train_loss, label='Train Loss', color='blue', linewidth=2, alpha=0.7)
ax2.plot(epochs, test_loss, label='Test Loss', color='red', linewidth=2, alpha=0.7)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss (log scale)')
ax2.set_title('Loss Curves', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

# Plot 3: Zoomed view of first major grokking (epochs 4000-6000)
if len(large_jumps) > 0:
    # Find the transition around epoch 5000
    transition_idx = large_jumps[large_jumps < 100][0] if any(large_jumps < 100) else large_jumps[0]
    start_idx = max(0, transition_idx - 20)
    end_idx = min(len(epochs), transition_idx + 30)
    
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(epochs[start_idx:end_idx], test_acc[start_idx:end_idx], 
            marker='o', markersize=6, color='red', linewidth=2, label='Test Accuracy')
    ax3.axvline(epochs[transition_idx], color='green', linestyle='--', linewidth=2, label='Grokking!')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Test Accuracy')
    ax3.set_title(f'Zoomed: First Major Transition\n(Epoch {epochs[transition_idx]})', 
                 fontsize=11, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.05])

# Plot 4: Zoomed view of last major grokking (epochs 37000-40000)
if len(large_jumps) > 0:
    # Last transition
    transition_idx = large_jumps[-1]
    start_idx = max(0, transition_idx - 20)
    end_idx = min(len(epochs), transition_idx + 20)
    
    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(epochs[start_idx:end_idx], test_acc[start_idx:end_idx], 
            marker='o', markersize=6, color='red', linewidth=2, label='Test Accuracy')
    ax4.axvline(epochs[transition_idx], color='green', linestyle='--', linewidth=2, 
               label=f'Grokking! (+{diffs[transition_idx]:.0%})')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Test Accuracy')
    ax4.set_title(f'Zoomed: Final Transition\n(Epoch {epochs[transition_idx]})', 
                 fontsize=11, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0.5, 1.05])

# Plot 5: Train vs Test Accuracy Gap
ax5 = plt.subplot(3, 2, 5)
acc_gap = train_acc - test_acc
ax5.plot(epochs, acc_gap, color='purple', linewidth=2)
ax5.fill_between(epochs, 0, acc_gap, alpha=0.3, color='purple')
ax5.set_xlabel('Epoch')
ax5.set_ylabel('Train Acc - Test Acc')
ax5.set_title('Generalization Gap\n(Large gap = overfitting, Small gap = grokked)', 
             fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.axhline(0, color='black', linestyle='-', linewidth=1)

# Plot 6: Rate of change in test accuracy (shows when grokking happens)
ax6 = plt.subplot(3, 2, 6)
test_acc_change = np.concatenate([[0], diffs])  # Prepend 0 for first epoch
ax6.bar(epochs, test_acc_change, width=100, color='green', alpha=0.6)
ax6.set_xlabel('Epoch')
ax6.set_ylabel('Change in Test Accuracy')
ax6.set_title('Grokking Events\n(Spikes = sudden improvement)', fontsize=11, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')
ax6.axhline(0.1, color='red', linestyle='--', linewidth=1, label='Major transition threshold')
ax6.legend(fontsize=9)

plt.suptitle('Paper 03: Nanda et al. (2023) - Detailed Grokking Analysis', 
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

output_dir = Path("analysis_results")
output_dir.mkdir(exist_ok=True)
save_path = output_dir / "paper_03_grokking_detailed.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved detailed plot: {save_path}")
print("\nThis plot clearly shows multiple grokking transitions!")
print("Note the green vertical lines and the spike plot in the bottom right.")

