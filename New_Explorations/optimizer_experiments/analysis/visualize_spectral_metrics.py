"""
Visualization script for spectral metrics from grokking experiments
Creates comprehensive plots comparing different optimizers and weight decay values
"""

import argparse
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_experiment_data(experiment_dir: Path) -> Dict:
    """Load training history and spectral metrics from experiment directory."""
    data = {}
    
    # Load training history
    history_path = experiment_dir / 'training_history.json'
    if history_path.exists():
        with open(history_path, 'r') as f:
            data['history'] = json.load(f)
    
    # Load spectral metrics
    spectral_path = experiment_dir / 'spectral_metrics.h5'
    if spectral_path.exists():
        with h5py.File(spectral_path, 'r') as f:
            data['spectral'] = {}
            for key in f.keys():
                data['spectral'][key] = f[key][:]
    
    # Load config
    config_path = experiment_dir / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            data['config'] = json.load(f)
    
    return data


def plot_training_curves(data: Dict, save_path: Path):
    """Plot training and test loss/accuracy curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    history = data['history']
    
    # Train loss
    axes[0, 0].plot(history['epoch'], history['train_loss'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Train Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Test loss
    axes[0, 1].plot(history['epoch'], history['test_loss'], 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Test Loss')
    axes[0, 1].set_title('Test Loss')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Train accuracy
    axes[1, 0].plot(history['epoch'], history['train_acc'], 'b-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Train Accuracy')
    axes[1, 0].set_title('Training Accuracy')
    axes[1, 0].set_ylim([0, 1.05])
    axes[1, 0].grid(True, alpha=0.3)
    
    # Test accuracy
    axes[1, 1].plot(history['epoch'], history['test_acc'], 'r-', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Test Accuracy')
    axes[1, 1].set_title('Test Accuracy (Grokking)')
    axes[1, 1].set_ylim([0, 1.05])
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add grokking marker if it occurred
    train_acc = np.array(history['train_acc'])
    test_acc = np.array(history['test_acc'])
    if len(train_acc) > 0 and len(test_acc) > 0:
        # Detect grokking: train acc high, test acc suddenly jumps
        train_high = train_acc > 0.95
        test_high = test_acc > 0.9
        if train_high.any() and test_high.any():
            train_grok_idx = np.where(train_high)[0][0]
            test_grok_idx = np.where(test_high)[0][0]
            if test_grok_idx > train_grok_idx:
                grok_epoch = history['epoch'][test_grok_idx]
                axes[1, 1].axvline(grok_epoch, color='green', linestyle='--', 
                                   linewidth=2, alpha=0.7, label=f'Grokking at epoch {grok_epoch}')
                axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {save_path}")


def plot_spectral_metrics(data: Dict, save_path: Path):
    """Plot key spectral metrics over training."""
    if 'spectral' not in data or not data['spectral']:
        print("No spectral metrics found, skipping spectral plot")
        return
    
    spectral = data['spectral']
    epochs = spectral['epoch']
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    # 1. Eigengap
    if 'eigengap' in spectral:
        axes[0, 0].plot(epochs, spectral['eigengap'], 'purple', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Eigengap')
        axes[0, 0].set_title('Eigengap (λ₁ - λ₂)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
    
    # 2. Top eigenvalue energy ratio
    if 'top_eigenvalue_energy_ratio' in spectral:
        axes[0, 1].plot(epochs, spectral['top_eigenvalue_energy_ratio'], 
                       'darkblue', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Energy Ratio')
        axes[0, 1].set_title('Energy in Top Eigenvector')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1.05])
    
    # 3. Spectral radius
    if 'spectral_radius' in spectral:
        axes[1, 0].plot(epochs, spectral['spectral_radius'], 'red', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Spectral Radius')
        axes[1, 0].set_title('Spectral Radius (λ_max)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
    
    # 4. Trace
    if 'trace' in spectral:
        axes[1, 1].plot(epochs, np.abs(spectral['trace']), 'green', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('|Trace|')
        axes[1, 1].set_title('Trace of GOP')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
    
    # 5. Spectral radius to trace ratio
    if 'spectral_radius_to_trace_ratio' in spectral:
        axes[2, 0].plot(epochs, spectral['spectral_radius_to_trace_ratio'], 
                       'orange', linewidth=2)
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('Ratio')
        axes[2, 0].set_title('Spectral Radius / Trace')
        axes[2, 0].grid(True, alpha=0.3)
    
    # 6. Effective rank
    if 'effective_rank' in spectral:
        axes[2, 1].plot(epochs, spectral['effective_rank'], 'brown', linewidth=2)
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('Effective Rank')
        axes[2, 1].set_title('Effective Rank of GOP')
        axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved spectral metrics to {save_path}")


def plot_top_k_eigenvalues(data: Dict, save_path: Path, k: int = 10):
    """Plot evolution of top-k eigenvalues."""
    if 'spectral' not in data or not data['spectral']:
        return
    
    spectral = data['spectral']
    epochs = spectral['epoch']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot top k eigenvalues
    colors = plt.cm.viridis(np.linspace(0, 1, k))
    for i in range(k):
        key = f'eigenvalue_{i+1}'
        if key in spectral:
            ax.plot(epochs, spectral[key], color=colors[i], 
                   linewidth=2, label=f'λ{i+1}', alpha=0.8)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Eigenvalue')
    ax.set_title(f'Top {k} Eigenvalues Evolution')
    ax.set_yscale('log')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved top-k eigenvalues to {save_path}")


def plot_comparison(experiments: Dict[str, Dict], save_path: Path, metric: str = 'test_acc'):
    """Compare multiple experiments on a single metric."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    
    for i, (name, data) in enumerate(experiments.items()):
        if 'history' in data and metric in data['history']:
            epochs = data['history']['epoch']
            values = data['history'][metric]
            ax.plot(epochs, values, color=colors[i], linewidth=2, 
                   label=name, alpha=0.8)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'Comparison: {metric.replace("_", " ").title()}')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    if 'loss' in metric:
        ax.set_yscale('log')
    elif 'acc' in metric:
        ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize grokking experiment results')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing experiment results')
    parser.add_argument('--experiment', type=str, default=None,
                       help='Specific experiment to visualize (or None for all)')
    parser.add_argument('--compare', action='store_true',
                       help='Create comparison plots across experiments')
    parser.add_argument('--output_dir', type=str, default='./plots',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.experiment:
        # Visualize single experiment
        exp_dir = results_dir / args.experiment
        if not exp_dir.exists():
            print(f"Experiment directory not found: {exp_dir}")
            return
        
        print(f"Loading experiment: {args.experiment}")
        data = load_experiment_data(exp_dir)
        
        exp_output_dir = output_dir / args.experiment
        exp_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create plots
        plot_training_curves(data, exp_output_dir / 'training_curves.png')
        plot_spectral_metrics(data, exp_output_dir / 'spectral_metrics.png')
        plot_top_k_eigenvalues(data, exp_output_dir / 'top_eigenvalues.png')
        
        print(f"Plots saved to {exp_output_dir}")
    
    elif args.compare:
        # Compare all experiments
        print("Loading all experiments for comparison...")
        experiments = {}
        
        for exp_dir in results_dir.iterdir():
            if exp_dir.is_dir() and (exp_dir / 'training_history.json').exists():
                print(f"  Loading: {exp_dir.name}")
                experiments[exp_dir.name] = load_experiment_data(exp_dir)
        
        if not experiments:
            print("No experiments found!")
            return
        
        print(f"\nCreating comparison plots for {len(experiments)} experiments...")
        
        # Create comparison plots for key metrics
        for metric in ['train_acc', 'test_acc', 'train_loss', 'test_loss']:
            plot_comparison(experiments, output_dir / f'comparison_{metric}.png', metric)
        
        print(f"Comparison plots saved to {output_dir}")
    
    else:
        # Visualize all experiments individually
        print("Visualizing all experiments...")
        
        for exp_dir in results_dir.iterdir():
            if exp_dir.is_dir() and (exp_dir / 'training_history.json').exists():
                print(f"\nProcessing: {exp_dir.name}")
                data = load_experiment_data(exp_dir)
                
                exp_output_dir = output_dir / exp_dir.name
                exp_output_dir.mkdir(parents=True, exist_ok=True)
                
                plot_training_curves(data, exp_output_dir / 'training_curves.png')
                plot_spectral_metrics(data, exp_output_dir / 'spectral_metrics.png')
                plot_top_k_eigenvalues(data, exp_output_dir / 'top_eigenvalues.png')
        
        print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()

