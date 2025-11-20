"""
Compare weight_decay=1.0 vs weight_decay=0.0 experiments
Determine if Slingshot mechanism requires regularization or occurs independently
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print("="*80)
print("PAPER 7: WEIGHT DECAY COMPARISON")
print("="*80)
print()

# Load both experiments
wd1_path = Path('../results_backup_wd1.0/logs/training_history.json')
wd0_path = Path('../results/logs/training_history.json')

if not wd1_path.exists():
    print("❌ ERROR: weight_decay=1.0 results not found!")
    print(f"   Expected: {wd1_path}")
    exit(1)

if not wd0_path.exists():
    print("❌ ERROR: weight_decay=0.0 results not found!")
    print(f"   Expected: {wd0_path}")
    print("   Run the exact replication first!")
    exit(1)

print("✅ Loading weight_decay=1.0 results...")
with open(wd1_path, 'r') as f:
    wd1_history = json.load(f)

print("✅ Loading weight_decay=0.0 results...")
with open(wd0_path, 'r') as f:
    wd0_history = json.load(f)

print()

# Extract data
wd1_epochs = np.array(wd1_history['epoch'])
wd1_train = np.array(wd1_history['train_acc']) * 100
wd1_test = np.array(wd1_history['test_acc']) * 100
wd1_norm = np.array(wd1_history['last_layer_norm'])

wd0_epochs = np.array(wd0_history['epoch'])
wd0_train = np.array(wd0_history['train_acc']) * 100
wd0_test = np.array(wd0_history['test_acc']) * 100
wd0_norm = np.array(wd0_history['last_layer_norm'])

# Analysis
print("EXPERIMENT COMPARISON")
print("-"*80)
print()

print("1. FINAL PERFORMANCE")
print(f"   WD=1.0: Train={wd1_train[-1]:.2f}%, Test={wd1_test[-1]:.2f}%")
print(f"   WD=0.0: Train={wd0_train[-1]:.2f}%, Test={wd0_test[-1]:.2f}%")
print()

print("2. GROKKING DETECTION (Test reaches 90%)")
wd1_grok_idx = np.where(wd1_test >= 90.0)[0]
wd0_grok_idx = np.where(wd0_test >= 90.0)[0]

if len(wd1_grok_idx) > 0:
    print(f"   WD=1.0: First at epoch {wd1_epochs[wd1_grok_idx[0]]}")
else:
    print(f"   WD=1.0: Never reached 90%")

if len(wd0_grok_idx) > 0:
    print(f"   WD=0.0: First at epoch {wd0_epochs[wd0_grok_idx[0]]}")
else:
    print(f"   WD=0.0: Never reached 90%")
print()

print("3. MAJOR TEST ACCURACY JUMPS (>20%)")
def count_major_jumps(test_acc, threshold=20.0):
    jumps = []
    for i in range(1, len(test_acc)):
        jump = test_acc[i] - test_acc[i-1]
        if abs(jump) > threshold:
            jumps.append(jump)
    return jumps

wd1_jumps = count_major_jumps(wd1_test)
wd0_jumps = count_major_jumps(wd0_test)

print(f"   WD=1.0: {len(wd1_jumps)} jumps, largest={max(wd1_jumps) if wd1_jumps else 0:.1f}%")
print(f"   WD=0.0: {len(wd0_jumps)} jumps, largest={max(wd0_jumps) if wd0_jumps else 0:.1f}%")
print()

print("4. LAST LAYER NORM BEHAVIOR")
print(f"   WD=1.0: Range={wd1_norm.min():.2f}-{wd1_norm.max():.2f}, Mean={wd1_norm.mean():.2f}, Std={wd1_norm.std():.2f}")
print(f"   WD=0.0: Range={wd0_norm.min():.2f}-{wd0_norm.max():.2f}, Mean={wd0_norm.mean():.2f}, Std={wd0_norm.std():.2f}")
print()

print("5. TEST ACCURACY VOLATILITY")
wd1_volatility = np.std(wd1_test)
wd0_volatility = np.std(wd0_test)
print(f"   WD=1.0: Std={wd1_volatility:.2f}%")
print(f"   WD=0.0: Std={wd0_volatility:.2f}%")
print()

# Create comparison visualization
fig, axes = plt.subplots(3, 2, figsize=(16, 12))

# Row 1: Test Accuracy
ax = axes[0, 0]
ax.plot(wd1_epochs, wd1_test, linewidth=1.5, color='#E63946', alpha=0.8, label='WD=1.0')
ax.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
ax.set_title('Test Accuracy: WD=1.0 (Regularization)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 105)

ax = axes[0, 1]
ax.plot(wd0_epochs, wd0_test, linewidth=1.5, color='#2A9D8F', alpha=0.8, label='WD=0.0')
ax.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
ax.set_title('Test Accuracy: WD=0.0 (NO Regularization)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 105)

# Row 2: Last Layer Norm
ax = axes[1, 0]
ax.plot(wd1_epochs, wd1_norm, linewidth=1.5, color='#E63946', alpha=0.8)
ax.set_ylabel('Last Layer Norm', fontsize=11, fontweight='bold')
ax.set_title('Weight Norm: WD=1.0', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.plot(wd0_epochs, wd0_norm, linewidth=1.5, color='#2A9D8F', alpha=0.8)
ax.set_ylabel('Last Layer Norm', fontsize=11, fontweight='bold')
ax.set_title('Weight Norm: WD=0.0', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Row 3: Direct Comparison
ax = axes[2, 0]
ax.plot(wd1_epochs, wd1_test, linewidth=1.5, color='#E63946', alpha=0.7, label='WD=1.0')
ax.plot(wd0_epochs, wd0_test, linewidth=1.5, color='#2A9D8F', alpha=0.7, label='WD=0.0')
ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
ax.set_title('Direct Comparison: Test Accuracy', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 105)

ax = axes[2, 1]
ax.plot(wd1_epochs, wd1_norm, linewidth=1.5, color='#E63946', alpha=0.7, label='WD=1.0')
ax.plot(wd0_epochs, wd0_norm, linewidth=1.5, color='#2A9D8F', alpha=0.7, label='WD=0.0')
ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax.set_ylabel('Last Layer Norm', fontsize=11, fontweight='bold')
ax.set_title('Direct Comparison: Weight Norm', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/weight_decay_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved visualization: ../results/weight_decay_comparison.png")
print()

# Final verdict
print("="*80)
print("SLINGSHOT MECHANISM VERIFICATION")
print("="*80)
print()

both_grok = len(wd1_grok_idx) > 0 and len(wd0_grok_idx) > 0
wd0_cyclic = len(wd0_jumps) > 5
wd0_strong_norm_cycles = wd0_norm.std() > 0.5

print("KEY QUESTION: Does Slingshot occur WITHOUT weight decay?")
print()

if both_grok and wd0_cyclic:
    print("✅ PAPER'S CLAIM VALIDATED")
    print("   Both experiments achieve grokking")
    print("   WD=0.0 shows cyclic behavior")
    print("   → Slingshot mechanism operates independently of regularization")
elif both_grok and not wd0_cyclic:
    print("⚠️ PARTIAL VALIDATION")
    print("   Both achieve grokking, but WD=0.0 lacks strong cycles")
    print("   → Regularization may enhance but not cause Slingshot")
elif len(wd1_grok_idx) > 0 and len(wd0_grok_idx) == 0:
    print("❌ PAPER'S CLAIM NOT VALIDATED")
    print("   Only WD=1.0 achieves grokking")
    print("   → Grokking appears to require regularization in our setup")
else:
    print("❌ NEITHER EXPERIMENT GROKS")
    print("   Need to debug implementation")

print()

if wd0_strong_norm_cycles:
    print(f"✅ Last layer norm shows cycles in WD=0.0 (std={wd0_norm.std():.2f})")
    print("   → Supports Slingshot mechanism")
else:
    print(f"⚠️ Weak norm cycles in WD=0.0 (std={wd0_norm.std():.2f})")
    print("   → Slingshot mechanism less clear")

print()
print("="*80)

# Summary statistics
print()
print("SUMMARY TABLE")
print("-"*80)
print(f"{'Metric':<30} {'WD=1.0':<20} {'WD=0.0':<20}")
print("-"*80)
print(f"{'Final Test Accuracy':<30} {wd1_test[-1]:>18.2f}% {wd0_test[-1]:>18.2f}%")
print(f"{'Grokking Achieved (>90%)':<30} {('Yes' if len(wd1_grok_idx)>0 else 'No'):>20} {('Yes' if len(wd0_grok_idx)>0 else 'No'):>20}")
print(f"{'Major Jumps (>20%)':<30} {len(wd1_jumps):>20d} {len(wd0_jumps):>20d}")
print(f"{'Test Acc Volatility (std)':<30} {wd1_volatility:>18.2f}% {wd0_volatility:>18.2f}%")
print(f"{'Norm Mean':<30} {wd1_norm.mean():>20.2f} {wd0_norm.mean():>20.2f}")
print(f"{'Norm Std Dev':<30} {wd1_norm.std():>20.2f} {wd0_norm.std():>20.2f}")
print("-"*80)

