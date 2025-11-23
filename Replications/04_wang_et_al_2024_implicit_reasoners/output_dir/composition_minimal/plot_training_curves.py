"""
Generate training curves for Paper 4
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("Generating training curve plots...")

# Load data
df = pd.read_csv('training_progress_scores.csv')

# Separate train and eval data
train_data = df[df['train_loss'] != -1.0].copy()
eval_data = df[df['eval_loss'] != -1.0].copy()

# Create figure with 2 subplots
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Training Loss
ax1 = axes[0]
ax1.plot(train_data['global_step'], train_data['train_loss'], 'b-', linewidth=2, label='Training Loss')
ax1.set_xlabel('Training Steps', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Paper 4 (Wang et al. 2024) - Training Loss Progression', fontsize=14, fontweight='bold')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)
ax1.axvline(x=50000, color='red', linestyle='--', alpha=0.5, label='Potential Grokking Point')

# Plot 2: Evaluation Loss at Checkpoints
ax2 = axes[1]
ax2.plot(eval_data['global_step'], eval_data['eval_loss'], 'ro-', linewidth=2, markersize=8, label='Validation Loss')
ax2.set_xlabel('Training Steps', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.set_title('Validation Loss at Checkpoints - Grokking Transition', fontsize=14, fontweight='bold')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)

# Highlight grokking region
ax2.axvspan(40000, 50000, alpha=0.2, color='green', label='Grokking Region')
ax2.text(45000, eval_data['eval_loss'].max() * 0.5, 'GROKKING\nTRANSITION', 
         ha='center', va='center', fontsize=12, fontweight='bold', 
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig('../../results/paper_04_training_curves.png', dpi=150, bbox_inches='tight')
print(f"✅ Saved: results/paper_04_training_curves.png")

# Create detailed grokking plot
fig2, ax = plt.subplots(1, 1, figsize=(14, 8))

# Plot both train and eval
ax.plot(train_data['global_step'], train_data['train_loss'], 'b-', linewidth=1.5, alpha=0.7, label='Training Loss')
ax.plot(eval_data['global_step'], eval_data['eval_loss'], 'ro-', linewidth=2, markersize=8, label='Validation Loss')

ax.set_xlabel('Training Steps', fontsize=13, fontweight='bold')
ax.set_ylabel('Loss (log scale)', fontsize=13, fontweight='bold')
ax.set_title('Paper 4: Grokking on Compositional Reasoning\n(Wang et al. 2024 - 100K Steps)', 
             fontsize=15, fontweight='bold')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, which='both')
ax.legend(fontsize=12, loc='upper right')

# Mark phases
ax.axvspan(0, 10000, alpha=0.1, color='blue', label='Memorization')
ax.axvspan(10000, 40000, alpha=0.1, color='orange', label='Pre-Grokking Plateau')
ax.axvspan(40000, 60000, alpha=0.1, color='green', label='Grokking Transition')
ax.axvspan(60000, 100000, alpha=0.1, color='purple', label='Post-Grokking')

# Add annotations
ax.annotate('Memorization\nComplete', xy=(10000, 0.01), xytext=(15000, 0.1),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2),
            fontsize=11, fontweight='bold', color='blue')

ax.annotate('GROKKING!\nHuge Drop', xy=(45000, 0.000005), xytext=(45000, 0.001),
            arrowprops=dict(arrowstyle='->', color='red', lw=3),
            fontsize=12, fontweight='bold', color='red',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

plt.tight_layout()
plt.savefig('../../results/paper_04_grokking_detailed.png', dpi=150, bbox_inches='tight')
print(f"✅ Saved: results/paper_04_grokking_detailed.png")

print()
print("=" * 70)
print("✅ Training curve visualizations created successfully!")
print("=" * 70)
