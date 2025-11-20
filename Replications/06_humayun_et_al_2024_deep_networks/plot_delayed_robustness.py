"""
Visualization script for delayed robustness in Deep Networks Always Grok
Humayun et al. (2024)

Plots clean accuracy and adversarial accuracy over training to demonstrate
delayed robustness phenomenon.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def plot_delayed_robustness(history_path, output_path, title="Delayed Robustness"):
    """
    Plot clean and adversarial accuracies over training
    
    Args:
        history_path: Path to training_history.json
        output_path: Path to save plot
        title: Plot title
    """
    # Load training history
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = np.array(history['epoch'])
    train_acc = np.array(history['train_acc'])
    test_acc = np.array(history['test_acc'])
    
    # Extract adversarial accuracies
    adv_keys = [k for k in history.keys() if k.startswith('adv_acc_eps_')]
    epsilons = []
    adv_accs = {}
    
    for key in sorted(adv_keys):
        eps = float(key.split('_')[-1])
        epsilons.append(eps)
        adv_accs[eps] = np.array(history[key])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Clean accuracies
    ax1.plot(epochs, train_acc * 100, 'b-', linewidth=2, label='Train Accuracy', alpha=0.8)
    ax1.plot(epochs, test_acc * 100, 'r-', linewidth=2, label='Test Accuracy (Clean)', alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title(f'{title}\nClean Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Right plot: Adversarial accuracies
    ax2.plot(epochs, test_acc * 100, 'k-', linewidth=2.5, label='Clean (ε=0)', alpha=0.9)
    
    # Color map for different epsilon values
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(epsilons)))
    
    for eps, color in zip(epsilons, colors):
        ax2.plot(epochs, adv_accs[eps] * 100, '-', linewidth=2, 
                color=color, label=f'ε={eps:.2f}', alpha=0.8)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax2.set_title(f'{title}\nDelayed Robustness', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    # Add annotation about delayed robustness
    ax2.text(0.05, 0.05, 'Adversarial accuracy improves\nafter clean accuracy plateaus',
            transform=ax2.transAxes, fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    # Print summary statistics
    print(f"\n{title} Summary:")
    print(f"Final Train Accuracy: {train_acc[-1]*100:.2f}%")
    print(f"Final Test Accuracy (Clean): {test_acc[-1]*100:.2f}%")
    for eps in epsilons:
        print(f"Final Adversarial Accuracy (ε={eps:.2f}): {adv_accs[eps][-1]*100:.2f}%")
    
    # Detect grokking
    if len(epochs) > 1:
        test_diff = np.diff(test_acc * 100)
        max_jump_idx = np.argmax(test_diff)
        max_jump = test_diff[max_jump_idx]
        if max_jump > 5:
            print(f"\nGrokking detected at epoch {epochs[max_jump_idx+1]}: {max_jump:.2f}% jump")
    
    return fig


def plot_comparison(history_paths, labels, output_path, title="Delayed Robustness Comparison"):
    """
    Compare delayed robustness across multiple experiments
    
    Args:
        history_paths: List of paths to training_history.json files
        labels: List of labels for each experiment
        output_path: Path to save plot
        title: Plot title
    """
    fig, axes = plt.subplots(len(history_paths), 2, figsize=(16, 6*len(history_paths)))
    
    if len(history_paths) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (path, label) in enumerate(zip(history_paths, labels)):
        # Load training history
        with open(path, 'r') as f:
            history = json.load(f)
        
        epochs = np.array(history['epoch'])
        train_acc = np.array(history['train_acc'])
        test_acc = np.array(history['test_acc'])
        
        # Extract adversarial accuracies
        adv_keys = [k for k in history.keys() if k.startswith('adv_acc_eps_')]
        epsilons = []
        adv_accs = {}
        
        for key in sorted(adv_keys):
            eps = float(key.split('_')[-1])
            epsilons.append(eps)
            adv_accs[eps] = np.array(history[key])
        
        # Left plot: Clean accuracies
        axes[idx, 0].plot(epochs, train_acc * 100, 'b-', linewidth=2, label='Train', alpha=0.8)
        axes[idx, 0].plot(epochs, test_acc * 100, 'r-', linewidth=2, label='Test', alpha=0.8)
        axes[idx, 0].set_xlabel('Epoch', fontsize=11)
        axes[idx, 0].set_ylabel('Accuracy (%)', fontsize=11)
        axes[idx, 0].set_title(f'{label}: Clean Accuracy', fontsize=12, fontweight='bold')
        axes[idx, 0].legend(fontsize=10)
        axes[idx, 0].grid(True, alpha=0.3)
        axes[idx, 0].set_xscale('log')
        
        # Right plot: Adversarial accuracies
        axes[idx, 1].plot(epochs, test_acc * 100, 'k-', linewidth=2.5, label='Clean', alpha=0.9)
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(epsilons)))
        for eps, color in zip(epsilons, colors):
            axes[idx, 1].plot(epochs, adv_accs[eps] * 100, '-', linewidth=2, 
                            color=color, label=f'ε={eps:.2f}', alpha=0.8)
        
        axes[idx, 1].set_xlabel('Epoch', fontsize=11)
        axes[idx, 1].set_ylabel('Test Accuracy (%)', fontsize=11)
        axes[idx, 1].set_title(f'{label}: Delayed Robustness', fontsize=12, fontweight='bold')
        axes[idx, 1].legend(fontsize=9, loc='best')
        axes[idx, 1].grid(True, alpha=0.3)
        axes[idx, 1].set_xscale('log')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.001)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {output_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Plot delayed robustness results')
    parser.add_argument('--history', type=str, required=True, 
                       help='Path to training_history.json')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for plot')
    parser.add_argument('--title', type=str, default='Delayed Robustness',
                       help='Plot title')
    
    args = parser.parse_args()
    
    plot_delayed_robustness(args.history, args.output, args.title)


if __name__ == '__main__':
    main()

