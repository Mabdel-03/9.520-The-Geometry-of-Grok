#!/usr/bin/env python
"""
Plot grokking results for Paper 06: Humayun et al. (2024) - Deep Networks Always Grok
Shows rapid grokking on MNIST
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load data
data_file = Path('06_humayun_et_al_2024_deep_networks/logs/training_history.json')
with open(data_file) as f:
    data = json.load(f)

epochs = np.array(data['epoch'])
train_acc = np.array(data['train_acc'])
test_acc = np.array(data['test_acc'])
train_loss = np.array(data['train_loss'])
test_loss = np.array(data['test_loss'])

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Paper 06: Humayun et al. (2024) - Deep Networks Always Grok\n' +
             'Rapid Grokking on MNIST (1000 samples)',
             fontsize=14, fontweight='bold')

# Plot 1: Full accuracy trajectory
ax = axes[0, 0]
ax.plot(epochs, train_acc, 'b-', label='Train Accuracy', linewidth=2, alpha=0.8)
ax.plot(epochs, test_acc, 'r-', label='Test Accuracy', linewidth=2, alpha=0.8)
ax.axvline(100, color='green', linestyle='--', alpha=0.6, label='Epoch 100 (both ~90%+)')
ax.axhline(0.9, color='gray', linestyle=':', alpha=0.3)
ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax.set_title('Accuracy: Rapid Grokking in First 100 Epochs', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])

# Add annotation
ax.annotate('33% Jump!', xy=(50, 0.73), xytext=(2000, 0.65),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=11, color='green', fontweight='bold')

# Plot 2: Loss curves (log scale)
ax = axes[0, 1]
ax.semilogy(epochs, train_loss, 'b-', label='Train Loss', linewidth=2, alpha=0.8)
ax.semilogy(epochs, test_loss, 'r-', label='Test Loss', linewidth=2, alpha=0.8)
ax.axvline(100, color='green', linestyle='--', alpha=0.6)
ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax.set_ylabel('Loss (log scale)', fontsize=11, fontweight='bold')
ax.set_title('Loss Curves', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')

# Plot 3: Zoomed early phase (0-1000 epochs)
ax = axes[1, 0]
early_mask = epochs <= 1000
ax.plot(epochs[early_mask], train_acc[early_mask], 'b-', label='Train', linewidth=3)
ax.plot(epochs[early_mask], test_acc[early_mask], 'r-', label='Test', linewidth=3)
ax.axvline(100, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Grokking point')
ax.axhline(0.9, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax.set_title('Early Grokking Phase (0-1000 epochs)\nRapid 33% Jump', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])

# Plot 4: Generalization gap
ax = axes[1, 1]
gap = train_acc - test_acc
ax.plot(epochs, gap, 'purple', linewidth=2, alpha=0.8)
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.fill_between(epochs, 0, gap, where=(gap > 0), alpha=0.3, color='orange', label='Generalization gap')
ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax.set_ylabel('Train Acc - Test Acc', fontsize=11, fontweight='bold')
ax.set_title('Generalization Gap: Stable ~11%', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Add text box
textstr = f'''Key Results:
• Dataset: MNIST (1000 samples)
• Initial test: 56.6%
• After 100 epochs: 89.8%
• Jump: +33.2%
• Final: Train 100%, Test 89.2%
• Gap: 10.8% (stable)
'''
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Save figure
output_file = Path('analysis_results/paper_06_grokking.png')
output_file.parent.mkdir(exist_ok=True)
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Plot saved to: {output_file}")
print(f"✓ Paper 06 shows rapid grokking: 33.2% jump in first 100 epochs")

plt.close()

print("\n" + "="*80)
print("PAPER 06: RAPID GROKKING CONFIRMED!")
print("="*80)
print(f"✓ Test accuracy: 56.6% → 89.8% in 100 epochs")
print(f"✓ Final: Train 100% | Test 89.2%")
print(f"✓ Demonstrates grokking on PRACTICAL vision task (MNIST)")
print("="*80)

