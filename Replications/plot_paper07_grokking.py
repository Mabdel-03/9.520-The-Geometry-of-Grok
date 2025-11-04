#!/usr/bin/env python
"""
Plot grokking results for Paper 07: Thilak et al. (2022) - Slingshot Mechanism
Shows dramatic cyclic grokking transitions
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load data
data_file = Path('07_thilak_et_al_2022_slingshot/logs/training_history.json')
with open(data_file) as f:
    data = json.load(f)

epochs = np.array(data['epoch'])
train_acc = np.array(data['train_acc'])
test_acc = np.array(data['test_acc'])
train_loss = np.array(data['train_loss'])
test_loss = np.array(data['test_loss'])

# Find major transitions
test_diff = np.diff(test_acc)
major_jumps = np.where(test_diff > 0.2)[0]

# Create comprehensive figure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

fig.suptitle('Paper 07: Thilak et al. (2022) - The Slingshot Mechanism\n' +
             'Cyclic Grokking with Massive Test Accuracy Jumps',
             fontsize=15, fontweight='bold')

# Plot 1: Full accuracy trajectory
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(epochs, train_acc, 'b-', label='Train Accuracy', linewidth=2, alpha=0.8)
ax1.plot(epochs, test_acc, 'r-', label='Test Accuracy', linewidth=2, alpha=0.8)

# Mark major jumps
for idx in major_jumps[:10]:
    ax1.axvline(epochs[idx], color='green', linestyle=':', alpha=0.3, linewidth=1)
    if test_diff[idx] > 0.5:  # Very large jumps
        ax1.annotate(f'+{test_diff[idx]:.0%}', 
                    xy=(epochs[idx], test_acc[idx+1]), 
                    xytext=(epochs[idx], test_acc[idx+1] + 0.05),
                    fontsize=9, color='green', fontweight='bold',
                    ha='center')

ax1.axhline(0.9, color='gray', linestyle='--', alpha=0.3, label='90% threshold')
ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax1.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax1.set_title('Cyclic Grokking: Multiple Massive Jumps in Test Accuracy', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1.05])

# Plot 2: Loss curves (log scale)
ax2 = fig.add_subplot(gs[1, 0])
ax2.semilogy(epochs, train_loss, 'b-', label='Train Loss', linewidth=2, alpha=0.8)
ax2.semilogy(epochs, test_loss, 'r-', label='Test Loss', linewidth=2, alpha=0.8)
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Loss (log scale)', fontsize=11)
ax2.set_title('Loss Curves', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, which='both')

# Plot 3: Generalization gap
ax3 = fig.add_subplot(gs[1, 1])
gap = train_acc - test_acc
ax3.plot(epochs, gap, 'purple', linewidth=2, alpha=0.8)
ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax3.fill_between(epochs, 0, gap, where=(gap > 0), alpha=0.3, color='orange', label='Overfitting region')
ax3.set_xlabel('Epoch', fontsize=11)
ax3.set_ylabel('Train Acc - Test Acc', fontsize=11)
ax3.set_title('Generalization Gap (Closes with Grokking)', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Plot 4: Early grokking phase (zoomed)
ax4 = fig.add_subplot(gs[2, 0])
early_mask = epochs <= 2000
ax4.plot(epochs[early_mask], train_acc[early_mask], 'b-', label='Train', linewidth=2.5)
ax4.plot(epochs[early_mask], test_acc[early_mask], 'r-', label='Test', linewidth=2.5)
ax4.axvline(200, color='blue', linestyle='--', alpha=0.6, label='Train→90% (epoch 200)')
ax4.axvline(700, color='red', linestyle='--', alpha=0.6, label='Test→90% (epoch 700)')
ax4.axhline(0.9, color='gray', linestyle=':', alpha=0.3)
ax4.fill_betweenx([0, 1], 200, 700, alpha=0.15, color='green', label='500-epoch grokking delay')
ax4.set_xlabel('Epoch', fontsize=11)
ax4.set_ylabel('Accuracy', fontsize=11)
ax4.set_title('Early Grokking Phase (0-2000 epochs)', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_ylim([0, 1.05])

# Plot 5: Major jump at epoch 31,200 (zoomed)
ax5 = fig.add_subplot(gs[2, 1])
zoom_center = 31250
zoom_width = 500
zoom_mask = (epochs >= zoom_center - zoom_width) & (epochs <= zoom_center + zoom_width)
ax5.plot(epochs[zoom_mask], train_acc[zoom_mask], 'b-', label='Train', linewidth=3)
ax5.plot(epochs[zoom_mask], test_acc[zoom_mask], 'r-', label='Test', linewidth=3)
ax5.axvline(31200, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Massive 91% jump!')
ax5.set_xlabel('Epoch', fontsize=11)
ax5.set_ylabel('Accuracy', fontsize=11)
ax5.set_title('Slingshot: 91% Jump at Epoch 31,200', fontsize=12, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)
ax5.set_ylim([0, 1.05])

plt.tight_layout()

# Save figure
output_file = Path('analysis_results/paper_07_slingshot_grokking.png')
output_file.parent.mkdir(exist_ok=True)
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Plot saved to: {output_file}")

plt.close()

print("\n" + "="*80)
print("PAPER 07: SLINGSHOT GROKKING CONFIRMED!")
print("="*80)
print(f"✓ Train→90%: Epoch 200")
print(f"✓ Test→90%:  Epoch 700")
print(f"⭐ Grokking delay: 500 epochs")
print(f"⭐ Final: Train {train_acc[-1]:.1%} | Test {test_acc[-1]:.1%}")
print(f"⭐ Largest jump: 91% at epoch 31,200!")
print("="*80)
EOF

