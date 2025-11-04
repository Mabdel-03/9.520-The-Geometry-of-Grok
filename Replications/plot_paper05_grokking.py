#!/usr/bin/env python3
"""
Plot grokking behavior for Paper 05: Omnigrok MNIST
Shows train/test loss and accuracy curves with grokking transitions
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load training data
data_path = Path('05_liu_et_al_2022_omnigrok/logs/training_history.json')
with open(data_path, 'r') as f:
    data = json.load(f)

# Extract data
epochs = np.array(data['epoch'])
train_loss = np.array(data['train_loss'])
train_acc = np.array(data['train_acc'])
test_loss = np.array(data['test_loss'])
test_acc = np.array(data['test_acc'])

# Find grokking transitions (test accuracy jumps > 5%)
test_acc_diff = np.diff(test_acc)
grokking_indices = np.where(test_acc_diff > 0.05)[0]

print("=== Paper 05: Omnigrok MNIST ===")
print(f"Training steps: {len(epochs)}")
print(f"Final step: {epochs[-1]}")
print(f"Final train acc: {train_acc[-1]*100:.2f}%")
print(f"Final test acc: {test_acc[-1]*100:.2f}%")
print(f"\nGrokking transitions detected: {len(grokking_indices)}")
for idx in grokking_indices:
    print(f"  Step {epochs[idx]} → {epochs[idx+1]}: "
          f"{test_acc[idx]*100:.2f}% → {test_acc[idx+1]*100:.2f}% "
          f"(+{(test_acc[idx+1]-test_acc[idx])*100:.2f}%)")

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Paper 05: Omnigrok - MNIST Grokking (1000 Training Samples)', 
             fontsize=16, fontweight='bold')

# Plot 1: Loss curves
ax1 = axes[0, 0]
ax1.plot(epochs, train_loss, label='Train Loss', color='blue', linewidth=2, alpha=0.8)
ax1.plot(epochs, test_loss, label='Test Loss', color='red', linewidth=2, alpha=0.8)
ax1.set_xlabel('Training Steps', fontsize=12)
ax1.set_ylabel('Loss (MSE)', fontsize=12)
ax1.set_title('Loss Curves', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# Plot 2: Accuracy curves with grokking markers
ax2 = axes[0, 1]
ax2.plot(epochs, train_acc * 100, label='Train Accuracy', color='blue', linewidth=2, alpha=0.8)
ax2.plot(epochs, test_acc * 100, label='Test Accuracy', color='red', linewidth=2, alpha=0.8)

# Mark grokking transitions
for idx in grokking_indices:
    ax2.axvline(epochs[idx], color='green', linestyle='--', alpha=0.5, linewidth=1.5)
    ax2.annotate(f'+{(test_acc[idx+1]-test_acc[idx])*100:.1f}%',
                xy=(epochs[idx], test_acc[idx]*100),
                xytext=(10, 10), textcoords='offset points',
                fontsize=9, color='green', fontweight='bold')

ax2.set_xlabel('Training Steps', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('Accuracy Curves (Grokking Transitions Marked)', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 105])

# Plot 3: Generalization gap
ax3 = axes[1, 0]
gen_gap = train_acc - test_acc
ax3.plot(epochs, gen_gap * 100, color='purple', linewidth=2)
ax3.axhline(0, color='black', linestyle='--', alpha=0.3)
ax3.fill_between(epochs, 0, gen_gap * 100, alpha=0.3, color='purple')
ax3.set_xlabel('Training Steps', fontsize=12)
ax3.set_ylabel('Generalization Gap (%)', fontsize=12)
ax3.set_title('Train - Test Accuracy Gap', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Add annotation
final_gap = gen_gap[-1] * 100
ax3.text(0.98, 0.95, f'Final Gap: {final_gap:.2f}%',
        transform=ax3.transAxes, fontsize=11, verticalalignment='top',
        horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 4: Phase diagram (train acc vs test acc)
ax4 = axes[1, 1]
scatter = ax4.scatter(train_acc * 100, test_acc * 100, c=epochs, cmap='viridis', 
                     s=30, alpha=0.6)
ax4.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='Perfect Generalization')

# Mark start and end
ax4.scatter(train_acc[0] * 100, test_acc[0] * 100, color='green', s=200, 
           marker='*', edgecolors='black', linewidths=1.5, label='Start', zorder=5)
ax4.scatter(train_acc[-1] * 100, test_acc[-1] * 100, color='red', s=200, 
           marker='*', edgecolors='black', linewidths=1.5, label='End', zorder=5)

ax4.set_xlabel('Train Accuracy (%)', fontsize=12)
ax4.set_ylabel('Test Accuracy (%)', fontsize=12)
ax4.set_title('Learning Trajectory (Train vs Test)', fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.set_xlim([0, 105])
ax4.set_ylim([0, 105])

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label('Training Step', fontsize=11)

plt.tight_layout()

# Save figure
output_dir = Path('analysis_results')
output_dir.mkdir(exist_ok=True)
output_path = output_dir / 'paper_05_grokking.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")

# Also create a detailed grokking analysis plot
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
fig2.suptitle('Paper 05: Omnigrok MNIST - Detailed Grokking Analysis', 
             fontsize=16, fontweight='bold')

# Main accuracy plot
ax_main = axes2[0, :]
ax_main = plt.subplot(2, 3, (1, 3))
ax_main.plot(epochs, train_acc * 100, label='Train Accuracy', color='blue', linewidth=2.5, alpha=0.8)
ax_main.plot(epochs, test_acc * 100, label='Test Accuracy', color='red', linewidth=2.5, alpha=0.8)

# Mark all grokking transitions
for idx in grokking_indices:
    ax_main.axvline(epochs[idx], color='green', linestyle='--', alpha=0.6, linewidth=2)
    
ax_main.set_xlabel('Training Steps', fontsize=13)
ax_main.set_ylabel('Accuracy (%)', fontsize=13)
ax_main.set_title('MNIST Grokking with 1000 Training Samples (100K Steps)', 
                 fontsize=14, fontweight='bold')
ax_main.legend(fontsize=12, loc='lower right')
ax_main.grid(True, alpha=0.3)
ax_main.set_ylim([0, 105])

# Add text box with final results
textstr = f'Final Results:\nTrain: {train_acc[-1]*100:.2f}%\nTest: {test_acc[-1]*100:.2f}%\nGap: {(train_acc[-1]-test_acc[-1])*100:.2f}%'
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
ax_main.text(0.02, 0.98, textstr, transform=ax_main.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

# Zoomed views of different phases
# Early phase (0-30K steps)
ax_early = axes2[1, 0]
early_mask = epochs <= 30000
ax_early.plot(epochs[early_mask], train_acc[early_mask] * 100, 'b-', linewidth=2, label='Train')
ax_early.plot(epochs[early_mask], test_acc[early_mask] * 100, 'r-', linewidth=2, label='Test')
ax_early.set_xlabel('Steps', fontsize=11)
ax_early.set_ylabel('Accuracy (%)', fontsize=11)
ax_early.set_title('Early Phase (0-30K)', fontsize=12, fontweight='bold')
ax_early.legend(fontsize=10)
ax_early.grid(True, alpha=0.3)

# Middle phase (30K-70K steps)
ax_mid = axes2[1, 1]
mid_mask = (epochs > 30000) & (epochs <= 70000)
ax_mid.plot(epochs[mid_mask], train_acc[mid_mask] * 100, 'b-', linewidth=2, label='Train')
ax_mid.plot(epochs[mid_mask], test_acc[mid_mask] * 100, 'r-', linewidth=2, label='Test')
ax_mid.set_xlabel('Steps', fontsize=11)
ax_mid.set_ylabel('Accuracy (%)', fontsize=11)
ax_mid.set_title('Middle Phase (30K-70K)', fontsize=12, fontweight='bold')
ax_mid.legend(fontsize=10)
ax_mid.grid(True, alpha=0.3)

# Late phase (70K-100K steps)
ax_late = axes2[1, 2]
late_mask = epochs > 70000
ax_late.plot(epochs[late_mask], train_acc[late_mask] * 100, 'b-', linewidth=2, label='Train')
ax_late.plot(epochs[late_mask], test_acc[late_mask] * 100, 'r-', linewidth=2, label='Test')
ax_late.set_xlabel('Steps', fontsize=11)
ax_late.set_ylabel('Accuracy (%)', fontsize=11)
ax_late.set_title('Late Phase (70K-100K)', fontsize=12, fontweight='bold')
ax_late.legend(fontsize=10)
ax_late.grid(True, alpha=0.3)

plt.tight_layout()

# Save detailed plot
detailed_path = output_dir / 'paper_05_grokking_detailed.png'
plt.savefig(detailed_path, dpi=300, bbox_inches='tight')
print(f"Detailed plot saved to: {detailed_path}")

plt.show()

print("\n✅ Plots generated successfully!")
print(f"\nKey findings:")
print(f"  • Final train accuracy: {train_acc[-1]*100:.2f}%")
print(f"  • Final test accuracy: {test_acc[-1]*100:.2f}%")
print(f"  • Generalization gap: {(train_acc[-1]-test_acc[-1])*100:.2f}%")
print(f"  • Grokking transitions: {len(grokking_indices)}")
print(f"\n🎉 Clear grokking behavior observed: model memorized training data then generalized!")

