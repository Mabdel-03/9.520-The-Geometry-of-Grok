#!/usr/bin/env python3
"""
Simple script to plot grokking results from training_history.json files
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paper names
papers = {
    "03": "Nanda et al. (2023) - Progress Measures",
    "07": "Thilak et al. (2022) - Slingshot",
    "08": "Doshi et al. (2024) - Modular Polynomials",
    "09": "Levi et al. (2023) - Linear Estimators",
}

# Create output directory
output_dir = Path("analysis_results")
output_dir.mkdir(exist_ok=True)

print("="*80)
print("Plotting Grokking Results")
print("="*80)

# Plot each paper
for paper_num, paper_name in papers.items():
    print(f"\nProcessing Paper {paper_num}: {paper_name}")
    
    # Find JSON file
    json_files = list(Path(".").glob(f"{paper_num}_*/logs/training_history.json"))
    
    if not json_files:
        print(f"  No training_history.json found, skipping...")
        continue
    
    json_file = json_files[0]
    print(f"  Loading: {json_file}")
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract data
        epochs = np.array(data['epoch'])
        train_loss = np.array(data['train_loss'])
        test_loss = np.array(data['test_loss'])
        train_acc = np.array(data['train_acc'])
        test_acc = np.array(data['test_acc'])
        
        print(f"  Found {len(epochs)} epochs of data")
        print(f"  Final Train Acc: {train_acc[-1]:.4f}, Test Acc: {test_acc[-1]:.4f}")
        
        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Loss curves
        ax = axes[0]
        ax.plot(epochs, train_loss, label='Train Loss', color='blue', linewidth=2, alpha=0.8)
        ax.plot(epochs, test_loss, label='Test Loss', color='red', linewidth=2, alpha=0.8)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'Loss Curves\n{paper_name}', fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Plot 2: Accuracy curves  
        ax = axes[1]
        ax.plot(epochs, train_acc, label='Train Accuracy', color='blue', linewidth=2, alpha=0.8)
        ax.plot(epochs, test_acc, label='Test Accuracy', color='red', linewidth=2, alpha=0.8)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'Accuracy Curves\n{paper_name}', fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        # Highlight grokking transition if visible
        if len(test_acc) > 10:
            # Find where test acc starts improving significantly
            test_acc_diff = np.diff(test_acc)
            if np.max(test_acc_diff) > 0.1:  # Significant jump
                grok_idx = np.argmax(test_acc_diff)
                ax.axvline(epochs[grok_idx], color='green', linestyle='--', 
                          label=f'Grokking Transition (epoch {epochs[grok_idx]})', alpha=0.7)
                ax.legend(fontsize=10)
        
        plt.tight_layout()
        save_path = output_dir / f"paper_{paper_num}_results.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {save_path}")
        
    except Exception as e:
        print(f"  Error: {e}")
        continue

print("\n" + "="*80)
print("Plotting Complete!")
print(f"Plots saved to: {output_dir}/")
print("="*80)

# Create a comparison plot
print("\nCreating comparison plot...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, (paper_num, paper_name) in enumerate(papers.items()):
    json_files = list(Path(".").glob(f"{paper_num}_*/logs/training_history.json"))
    
    if not json_files:
        continue
    
    try:
        with open(json_files[0], 'r') as f:
            data = json.load(f)
        
        epochs = np.array(data['epoch'])
        train_loss = np.array(data['train_loss'])
        test_loss = np.array(data['test_loss'])
        train_acc = np.array(data['train_acc'])
        test_acc = np.array(data['test_acc'])
        
        ax = axes[idx]
        
        # Plot both loss and accuracy on same plot (dual y-axis)
        ax2 = ax.twinx()
        
        # Losses (log scale) on left axis
        l1 = ax.plot(epochs, train_loss, label='Train Loss', color='blue', linewidth=1.5, alpha=0.7)
        l2 = ax.plot(epochs, test_loss, label='Test Loss', color='red', linewidth=1.5, alpha=0.7)
        ax.set_ylabel('Loss (log scale)', fontsize=10, color='blue')
        ax.set_yscale('log')
        ax.tick_params(axis='y', labelcolor='blue')
        
        # Accuracies on right axis
        l3 = ax2.plot(epochs, train_acc, label='Train Acc', color='cyan', linewidth=1.5, 
                     alpha=0.6, linestyle='--')
        l4 = ax2.plot(epochs, test_acc, label='Test Acc', color='orange', linewidth=2, alpha=0.8)
        ax2.set_ylabel('Accuracy', fontsize=10, color='orange')
        ax2.set_ylim([0, 1.05])
        ax2.tick_params(axis='y', labelcolor='orange')
        
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_title(f'Paper {paper_num}: {paper_name.split("-")[0].strip()}', fontsize=11)
        
        # Combined legend
        lns = l1 + l2 + l3 + l4
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='center right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
    except Exception as e:
        axes[idx].text(0.5, 0.5, f'Paper {paper_num}\nNo Data', 
                      ha='center', va='center', transform=axes[idx].transAxes, fontsize=12)

plt.suptitle('Grokking Comparison: All Papers', fontsize=16, y=0.995)
plt.tight_layout()
comparison_path = output_dir / "all_papers_comparison.png"
plt.savefig(comparison_path, dpi=200, bbox_inches='tight')
plt.close()

print(f"✓ Saved comparison: {comparison_path}")
print("\nAll done!")

