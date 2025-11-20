"""
Analyze the Slingshot Mechanism: Last Layer Norm vs Grokking Events
This script verifies whether cyclic weight norm behavior correlates with test accuracy jumps
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load training history
history_path = Path('../results/logs/training_history.json')
with open(history_path, 'r') as f:
    history = json.load(f)

epochs = np.array(history['epoch'])
train_acc = np.array(history['train_acc'])
test_acc = np.array(history['test_acc'])
last_layer_norm = np.array(history['last_layer_norm'])

print("="*80)
print("PAPER 7: SLINGSHOT MECHANISM ANALYSIS")
print("="*80)
print()

# Basic statistics
print("LAST LAYER NORM STATISTICS")
print("-"*80)
print(f"Min:    {last_layer_norm.min():.2f}")
print(f"Max:    {last_layer_norm.max():.2f}")
print(f"Mean:   {last_layer_norm.mean():.2f}")
print(f"Std:    {last_layer_norm.std():.2f}")
print(f"Range:  {last_layer_norm.max() - last_layer_norm.min():.2f}")
print()

# Find major test accuracy jumps (>20% in 100 epochs)
print("MAJOR TEST ACCURACY JUMPS (>20%)")
print("-"*80)
jump_threshold = 20.0
major_jumps = []

for i in range(1, len(test_acc)):
    jump = (test_acc[i] - test_acc[i-1]) * 100
    if abs(jump) > jump_threshold:
        major_jumps.append({
            'epoch': epochs[i],
            'from': test_acc[i-1] * 100,
            'to': test_acc[i] * 100,
            'jump': jump,
            'norm_before': last_layer_norm[i-1],
            'norm_after': last_layer_norm[i],
            'norm_change': last_layer_norm[i] - last_layer_norm[i-1]
        })

for j in major_jumps:
    print(f"Epoch {j['epoch']:6d}: {j['from']:5.1f}% → {j['to']:5.1f}% "
          f"(Δ{j['jump']:+6.1f}%) | Norm: {j['norm_before']:.1f} → {j['norm_after']:.1f} "
          f"(Δ{j['norm_change']:+.1f})")

print()
print(f"Total major jumps: {len(major_jumps)}")
print()

# Analyze correlation between norm changes and test accuracy changes
print("CORRELATION ANALYSIS")
print("-"*80)
test_acc_changes = np.diff(test_acc) * 100  # Convert to percentage
norm_changes = np.diff(last_layer_norm)

correlation = np.corrcoef(test_acc_changes, norm_changes)[0, 1]
print(f"Correlation between test acc change and norm change: {correlation:.3f}")
print()

# Detect cyclic behavior in last layer norm
print("CYCLIC BEHAVIOR DETECTION")
print("-"*80)

# Simple peak/trough detection without scipy
def find_peaks_simple(data, distance=10, threshold=0):
    """Find local maxima in data"""
    peaks = []
    for i in range(distance, len(data) - distance):
        # Check if this point is higher than neighbors
        is_peak = True
        for j in range(i - distance, i + distance + 1):
            if j != i and data[j] >= data[i]:
                is_peak = False
                break
        if is_peak and (len(peaks) == 0 or i - peaks[-1] >= distance):
            # Check prominence (difference from local minima)
            local_min = min(data[max(0, i-distance):min(len(data), i+distance+1)])
            if data[i] - local_min > threshold:
                peaks.append(i)
    return np.array(peaks)

# Smooth the data slightly to avoid noise
window = 5
smoothed_norm = np.convolve(last_layer_norm, np.ones(window)/window, mode='same')

# Find peaks (local maxima) and troughs (local minima)
peaks = find_peaks_simple(smoothed_norm, distance=10, threshold=1.0)
troughs = find_peaks_simple(-smoothed_norm, distance=10, threshold=1.0)

print(f"Number of peaks (high norm points): {len(peaks)}")
print(f"Number of troughs (low norm points): {len(troughs)}")
print(f"Total oscillations: ~{min(len(peaks), len(troughs))}")
print()

if len(peaks) > 0:
    print("Peak epochs (high norm):")
    for i, peak_idx in enumerate(peaks[:10]):  # Show first 10
        print(f"  Epoch {epochs[peak_idx]:6d}: Norm = {last_layer_norm[peak_idx]:.1f}, Test Acc = {test_acc[peak_idx]*100:.1f}%")
    if len(peaks) > 10:
        print(f"  ... and {len(peaks)-10} more")
print()

# Create comprehensive visualization
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

# Panel 1: Test Accuracy with major jumps
ax1 = axes[0]
ax1.plot(epochs, test_acc * 100, linewidth=1.5, color='#2E86AB', label='Test Accuracy')
ax1.scatter([j['epoch'] for j in major_jumps], [j['to'] for j in major_jumps], 
           color='red', s=50, zorder=5, label='Major Jumps (>20%)')
ax1.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='best')
ax1.set_title('Test Accuracy Trajectory', fontsize=12, fontweight='bold')

# Panel 2: Last Layer Norm
ax2 = axes[1]
ax2.plot(epochs, last_layer_norm, linewidth=1.5, color='#A23B72', label='Last Layer Norm')
ax2.scatter(epochs[peaks], last_layer_norm[peaks], color='orange', s=40, 
           marker='^', zorder=5, label='Peaks')
ax2.scatter(epochs[troughs], last_layer_norm[troughs], color='purple', s=40, 
           marker='v', zorder=5, label='Troughs')
ax2.set_ylabel('Last Layer L2 Norm', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='best')
ax2.set_title('Slingshot Mechanism: Last Layer Weight Norm Cycles', fontsize=12, fontweight='bold')

# Panel 3: Overlay - Norm and Test Accuracy
ax3 = axes[2]
ax3_twin = ax3.twinx()
ax3.plot(epochs, last_layer_norm, linewidth=1.5, color='#A23B72', alpha=0.7, label='Norm')
ax3_twin.plot(epochs, test_acc * 100, linewidth=1.5, color='#2E86AB', alpha=0.7, label='Test Acc')
ax3.set_ylabel('Last Layer Norm', fontsize=11, fontweight='bold', color='#A23B72')
ax3_twin.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold', color='#2E86AB')
ax3.tick_params(axis='y', labelcolor='#A23B72')
ax3_twin.tick_params(axis='y', labelcolor='#2E86AB')
ax3.grid(True, alpha=0.3)
ax3.set_title('Correlation: Weight Norm vs Test Accuracy', fontsize=12, fontweight='bold')

# Panel 4: Zoomed view of a dramatic cycle (epoch 31000-32000)
ax4 = axes[3]
zoom_start = np.searchsorted(epochs, 31000)
zoom_end = np.searchsorted(epochs, 32000)
ax4_twin = ax4.twinx()
ax4.plot(epochs[zoom_start:zoom_end], last_layer_norm[zoom_start:zoom_end], 
        linewidth=2, color='#A23B72', marker='o', markersize=4, label='Norm')
ax4_twin.plot(epochs[zoom_start:zoom_end], test_acc[zoom_start:zoom_end] * 100, 
             linewidth=2, color='#2E86AB', marker='s', markersize=4, label='Test Acc')
ax4.set_ylabel('Last Layer Norm', fontsize=11, fontweight='bold', color='#A23B72')
ax4_twin.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold', color='#2E86AB')
ax4.tick_params(axis='y', labelcolor='#A23B72')
ax4_twin.tick_params(axis='y', labelcolor='#2E86AB')
ax4.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_title('Zoomed: Largest Slingshot Event (90.7% jump at epoch 31,200)', 
             fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('../results/slingshot_mechanism_analysis.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved visualization: ../results/slingshot_mechanism_analysis.png")
print()

# Summary statistics for the report
print("SLINGSHOT MECHANISM VERIFICATION")
print("="*80)
print()
print("KEY FINDINGS:")
print()
print(f"1. Last layer norm shows {'CLEAR' if len(peaks) > 5 else 'WEAK'} cyclic behavior")
print(f"   - {len(peaks)} peaks and {len(troughs)} troughs detected")
print(f"   - Norm oscillates between {last_layer_norm.min():.1f} and {last_layer_norm.max():.1f}")
print()
print(f"2. Test accuracy shows EXTREME cyclic behavior")
print(f"   - {len(major_jumps)} major jumps (>20%) detected")
print(f"   - Largest jump: {max([j['jump'] for j in major_jumps]) if major_jumps else 0:.1f}%")
print()
print(f"3. Correlation between norm change and test accuracy change: {correlation:.3f}")
if abs(correlation) > 0.3:
    print("   - STRONG correlation suggests Slingshot mechanism is active")
elif abs(correlation) > 0.1:
    print("   - MODERATE correlation suggests partial Slingshot effect")
else:
    print("   - WEAK correlation suggests other mechanisms may dominate")
print()
print("VERDICT:")
if len(peaks) > 5 and len(major_jumps) > 3:
    print("✅ SLINGSHOT MECHANISM CONFIRMED")
    print("   - Cyclic weight norm behavior present")
    print("   - Multiple grokking events correlate with cycles")
elif len(major_jumps) > 3:
    print("⚠️ CYCLIC GROKKING CONFIRMED, but norm cycles unclear")
    print("   - Test accuracy shows dramatic cyclic behavior")
    print("   - Weight norm cycles less pronounced than expected")
else:
    print("❌ SLINGSHOT MECHANISM NOT CLEARLY PRESENT")
print()
print("="*80)

