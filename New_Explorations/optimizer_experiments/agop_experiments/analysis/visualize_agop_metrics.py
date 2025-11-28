"""
Visualization script for AGOP metrics across grokking experiments

This script loads AGOP metrics from multiple experiments and creates comprehensive
visualizations to analyze grokking mechanisms. Adapted from notebook Cells 7-9.

Generated Plots (single experiment):
    - training_curves.png: Train/test accuracy over epochs
    - agop_basic_metrics.png: Frobenius norm, spectral radius, trace
    - agop_collapse_metrics.png: Eigengap, variation collapse ratio
    - agop_subspace_similarity.png: Top-k subspace stability
    - combined_grokking_agop.png: 2x2 grid of test acc + AGOP metrics
    - comprehensive_timeline.png: 5-panel aligned timeline with grokking annotation
    - aligned_test_acc_*.png: Dual-axis plots showing test acc vs each AGOP metric

Usage:
    python visualize_agop_metrics.py --results_dir ./results/agop_experiments/nanda/nanda_adamw_wd1.0_seed42
    python visualize_agop_metrics.py --results_dir ./results/agop_experiments --compare_optimizers
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
import h5py
from pathlib import Path
from typing import Dict, List, Tuple
import seaborn as sns

sns.set_style("whitegrid")


def smooth_series(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply moving average smoothing"""
    smoothed = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        smoothed[i] = np.mean(x[start:i + 1])
    return smoothed


def load_experiment(exp_dir: Path) -> Dict:
    """Load experiment data from directory"""
    data = {}
    
    # Load config
    config_path = exp_dir / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            data['config'] = json.load(f)
    
    # Load training history
    history_path = exp_dir / 'training_history.json'
    if history_path.exists():
        with open(history_path, 'r') as f:
            data['history'] = json.load(f)
    
    # Load AGOP metrics
    agop_path = exp_dir / 'agop_metrics.h5'
    if agop_path.exists():
        data['agop'] = {}
        with h5py.File(agop_path, 'r') as f:
            for key in f.keys():
                data['agop'][key] = f[key][:]
    
    return data


def plot_training_curves(experiments: Dict[str, Dict], save_path: Path):
    """Plot training and test accuracy curves"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for name, exp in experiments.items():
        history = exp['history']
        epochs = np.array(history['epoch'])
        
        axes[0].plot(epochs, history['train_acc'], label=name, alpha=0.7)
        axes[1].plot(epochs, history['test_acc'], label=name, alpha=0.7)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Train Accuracy')
    axes[0].set_title('Training Accuracy')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Test Accuracy')
    axes[1].set_title('Test Accuracy (Grokking)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved training curves to {save_path}")
    plt.close()


def plot_agop_basic_metrics(experiments: Dict[str, Dict], save_path: Path):
    """Plot basic AGOP metrics: Frobenius, Spectral Radius, Trace"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for name, exp in experiments.items():
        if 'agop' not in exp or not exp['agop']:
            continue
        
        agop = exp['agop']
        epochs = agop.get('epoch', np.arange(len(agop.get('agop_frobenius', []))))
        
        if 'agop_frobenius' in agop:
            frobenius_smooth = smooth_series(agop['agop_frobenius'])
            axes[0].plot(epochs, frobenius_smooth, label=name, alpha=0.8)
        
        if 'agop_spectral_radius' in agop:
            spectral_smooth = smooth_series(agop['agop_spectral_radius'])
            axes[1].plot(epochs, spectral_smooth, label=name, alpha=0.8)
        
        if 'agop_trace' in agop:
            trace_smooth = smooth_series(agop['agop_trace'])
            axes[2].plot(epochs, trace_smooth, label=name, alpha=0.8)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Frobenius Norm')
    axes[0].set_title('AGOP Frobenius Norm (smoothed)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Spectral Radius (λ₁)')
    axes[1].set_title('AGOP Spectral Radius (smoothed)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Trace (Σλᵢ)')
    axes[2].set_title('AGOP Trace (smoothed)')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved AGOP basic metrics to {save_path}")
    plt.close()


def plot_agop_collapse_metrics(experiments: Dict[str, Dict], save_path: Path):
    """Plot collapse metrics: Eigengap and Variation Collapse Ratio"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for name, exp in experiments.items():
        if 'agop' not in exp or not exp['agop']:
            continue
        
        agop = exp['agop']
        epochs = agop.get('epoch', np.arange(len(agop.get('agop_eigengap', []))))
        
        if 'agop_eigengap' in agop:
            eigengap_smooth = smooth_series(agop['agop_eigengap'])
            axes[0].plot(epochs, eigengap_smooth, label=name, alpha=0.8)
        
        if 'agop_variation_collapse_ratio' in agop:
            vcr_smooth = smooth_series(agop['agop_variation_collapse_ratio'])
            axes[1].plot(epochs, vcr_smooth, label=name, alpha=0.8)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Eigengap (λ₁ - λ₂)')
    axes[0].set_title('AGOP Eigengap (smoothed)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('VCR (λ₁ / Σλᵢ)')
    axes[1].set_title('Variation Collapse Ratio (smoothed)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved AGOP collapse metrics to {save_path}")
    plt.close()


def plot_agop_subspace_similarity(experiments: Dict[str, Dict], save_path: Path):
    """Plot top-k subspace similarity over time"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, exp in experiments.items():
        if 'agop' not in exp or not exp['agop']:
            continue
        
        agop = exp['agop']
        if 'agop_topk_subspace_similarity' not in agop:
            continue
        
        epochs = agop.get('epoch', np.arange(len(agop['agop_topk_subspace_similarity'])))
        similarity = agop['agop_topk_subspace_similarity']
        
        # Filter out NaN values
        valid_mask = ~np.isnan(similarity)
        if valid_mask.sum() > 0:
            ax.plot(epochs[valid_mask], similarity[valid_mask], label=name, alpha=0.8)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Subspace Similarity')
    ax.set_title('Top-k Subspace Stability')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved subspace similarity to {save_path}")
    plt.close()


def plot_combined_grokking_agop(exp_data: Dict, save_path: Path):
    """Create combined plot showing test accuracy and AGOP metrics together"""
    if 'agop' not in exp_data or not exp_data['agop']:
        print("No AGOP data available for combined plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    history = exp_data['history']
    agop = exp_data['agop']
    
    train_epochs = np.array(history['epoch'])
    agop_epochs = agop.get('epoch', train_epochs)
    
    # Plot 1: Test accuracy (grokking indicator)
    axes[0, 0].plot(train_epochs, history['test_acc'], 'b-', linewidth=2)
    axes[0, 0].axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='Grokking threshold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Test Accuracy')
    axes[0, 0].set_title('Test Accuracy (Grokking)')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Plot 2: Variation Collapse Ratio
    if 'agop_variation_collapse_ratio' in agop:
        vcr = smooth_series(agop['agop_variation_collapse_ratio'])
        axes[0, 1].plot(agop_epochs, vcr, 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('VCR (λ₁ / Σλᵢ)')
        axes[0, 1].set_title('Variation Collapse Ratio')
        axes[0, 1].grid(alpha=0.3)
    
    # Plot 3: Eigengap
    if 'agop_eigengap' in agop:
        eigengap = smooth_series(agop['agop_eigengap'])
        axes[1, 0].plot(agop_epochs, eigengap, 'r-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Eigengap (λ₁ - λ₂)')
        axes[1, 0].set_title('AGOP Eigengap')
        axes[1, 0].grid(alpha=0.3)
    
    # Plot 4: Spectral radius and Trace
    if 'agop_spectral_radius' in agop and 'agop_trace' in agop:
        spectral = smooth_series(agop['agop_spectral_radius'])
        trace = smooth_series(agop['agop_trace'])
        axes[1, 1].plot(agop_epochs, spectral, 'orange', linewidth=2, label='Spectral Radius (λ₁)')
        axes[1, 1].plot(agop_epochs, trace, 'purple', linewidth=2, label='Trace (Σλᵢ)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_title('Spectral Radius vs Trace')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined grokking-AGOP plot to {save_path}")
    plt.close()


def detect_and_annotate_grokking(ax, epochs, test_acc, threshold=0.95, window=10):
    """
    Detect grokking transition and annotate on plot.
    
    Args:
        ax: Matplotlib axis object
        epochs: Array of epoch numbers
        test_acc: Array of test accuracies
        threshold: Accuracy threshold for grokking (default: 0.95)
        window: Number of epochs to average for stability (default: 10)
        
    Returns:
        grok_epoch: Epoch where grokking occurred (None if no grokking)
    """
    # Find first epoch where test acc crosses threshold and stays above
    grok_epoch = None
    epochs_arr = np.array(epochs)
    test_acc_arr = np.array(test_acc)
    
    for i in range(len(test_acc_arr) - window):
        if np.mean(test_acc_arr[i:i+window]) > threshold:
            grok_epoch = epochs_arr[i]
            break
    
    if grok_epoch is not None:
        ax.axvline(x=grok_epoch, color='red', linestyle=':', linewidth=2, 
                   alpha=0.7, label=f'Grokking @ epoch {grok_epoch}')
        ax.text(grok_epoch, 0.5, f'Grok\n@{grok_epoch}', 
                rotation=90, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    return grok_epoch


def plot_aligned_dual_axis(exp_data: Dict, metric_name: str, 
                           metric_label: str, save_path: Path, window: int = 5):
    """
    Create dual-axis plot with test accuracy and AGOP metric aligned.
    Perfect for seeing correlation between grokking and AGOP changes.
    
    Args:
        exp_data: Experiment data dictionary
        metric_name: Name of AGOP metric to plot (e.g., 'agop_variation_collapse_ratio')
        metric_label: Label for metric (e.g., 'VCR (λ₁ / Σλᵢ)')
        save_path: Path to save figure
        window: Smoothing window size
    """
    if 'agop' not in exp_data or not exp_data['agop']:
        print("No AGOP data available")
        return
    
    history = exp_data['history']
    agop = exp_data['agop']
    
    if metric_name not in agop:
        print(f"Metric {metric_name} not found")
        return
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Test accuracy on left axis
    train_epochs = np.array(history['epoch'])
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Test Accuracy', color='blue', fontsize=12)
    line1 = ax1.plot(train_epochs, history['test_acc'], 'b-', linewidth=2, label='Test Accuracy')
    ax1.axhline(y=0.95, color='blue', linestyle='--', alpha=0.3, label='Grokking threshold')
    
    # Detect and annotate grokking
    grok_epoch = detect_and_annotate_grokking(ax1, train_epochs, history['test_acc'])
    
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim([0, 1.05])
    ax1.grid(alpha=0.3)
    
    # AGOP metric on right axis
    ax2 = ax1.twinx()
    agop_epochs = agop.get('epoch', train_epochs)
    metric_values = smooth_series(agop[metric_name], window=window)
    line2 = ax2.plot(agop_epochs, metric_values, 'r-', linewidth=2, label=metric_label)
    ax2.set_ylabel(metric_label, color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Mark grokking epoch on metric plot
    if grok_epoch is not None and grok_epoch in agop_epochs:
        idx = np.where(agop_epochs == grok_epoch)[0]
        if len(idx) > 0:
            ax2.plot(grok_epoch, metric_values[idx[0]], 'ro', markersize=10, 
                    label=f'Metric @ grokking')
    
    # Title and combined legend
    plt.title(f'Test Accuracy vs {metric_label}', fontsize=14, pad=20)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved dual-axis plot to {save_path}")
    plt.close()


def plot_comprehensive_timeline(exp_data: Dict, save_path: Path, window: int = 5):
    """
    Create comprehensive timeline showing all key metrics aligned.
    Useful for publication or detailed analysis.
    
    Args:
        exp_data: Experiment data dictionary
        save_path: Path to save figure
        window: Smoothing window size
    """
    if 'agop' not in exp_data or not exp_data['agop']:
        print("No AGOP data available")
        return
    
    history = exp_data['history']
    agop = exp_data['agop']
    
    train_epochs = np.array(history['epoch'])
    agop_epochs = agop.get('epoch', train_epochs)
    
    # Create figure with shared x-axis
    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
    
    # Plot 1: Accuracy
    axes[0].plot(train_epochs, history['train_acc'], 'b-', label='Train Acc', alpha=0.7)
    axes[0].plot(train_epochs, history['test_acc'], 'r-', label='Test Acc', linewidth=2)
    axes[0].axhline(y=0.95, color='gray', linestyle='--', alpha=0.3)
    
    # Detect and annotate grokking
    grok_epoch = detect_and_annotate_grokking(axes[0], train_epochs, history['test_acc'])
    
    axes[0].set_ylabel('Accuracy', fontsize=11)
    axes[0].set_title('Training Progress and AGOP Evolution', fontsize=13, fontweight='bold')
    axes[0].legend(loc='lower right')
    axes[0].grid(alpha=0.3)
    
    # Mark grokking epoch on all subsequent plots
    if grok_epoch is not None:
        for ax in axes[1:]:
            ax.axvline(x=grok_epoch, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
    
    # Plot 2: Variation Collapse Ratio
    if 'agop_variation_collapse_ratio' in agop:
        vcr = smooth_series(agop['agop_variation_collapse_ratio'], window)
        axes[1].plot(agop_epochs, vcr, 'g-', linewidth=2)
        axes[1].set_ylabel('VCR (λ₁/Σλᵢ)', fontsize=11)
        axes[1].set_title('Variation Collapse Ratio', fontsize=11)
        axes[1].grid(alpha=0.3)
    
    # Plot 3: Eigengap
    if 'agop_eigengap' in agop:
        eigengap = smooth_series(agop['agop_eigengap'], window)
        axes[2].plot(agop_epochs, eigengap, 'purple', linewidth=2)
        axes[2].set_ylabel('Eigengap (λ₁-λ₂)', fontsize=11)
        axes[2].set_title('AGOP Eigengap', fontsize=11)
        axes[2].grid(alpha=0.3)
    
    # Plot 4: Spectral Radius vs Trace
    if 'agop_spectral_radius' in agop and 'agop_trace' in agop:
        spectral = smooth_series(agop['agop_spectral_radius'], window)
        trace = smooth_series(agop['agop_trace'], window)
        axes[3].plot(agop_epochs, spectral, 'orange', linewidth=2, label='λ₁ (Spectral Radius)')
        axes[3].plot(agop_epochs, trace, 'brown', linewidth=2, label='Σλᵢ (Trace)')
        axes[3].set_ylabel('Value', fontsize=11)
        axes[3].set_title('Spectral Radius vs Trace', fontsize=11)
        axes[3].legend()
        axes[3].grid(alpha=0.3)
    
    # Plot 5: Subspace Similarity
    if 'agop_topk_subspace_similarity' in agop:
        similarity = agop['agop_topk_subspace_similarity']
        valid_mask = ~np.isnan(similarity)
        if valid_mask.sum() > 0:
            axes[4].plot(agop_epochs[valid_mask], similarity[valid_mask], 
                        'teal', linewidth=2)
            axes[4].set_ylabel('Subspace Similarity', fontsize=11)
            axes[4].set_title('Top-k Subspace Stability', fontsize=11)
            axes[4].grid(alpha=0.3)
    
    axes[4].set_xlabel('Epoch', fontsize=12)
    
    # Add text annotation about grokking
    if grok_epoch is not None:
        fig.text(0.99, 0.01, f'Grokking detected at epoch {grok_epoch}', 
                ha='right', va='bottom', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved comprehensive timeline to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize AGOP metrics from grokking experiments')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing experiment results')
    parser.add_argument('--experiment_pattern', type=str, default='*',
                       help='Pattern to match experiment directories (e.g., "nanda_adamw*")')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save plots (default: results_dir/plots)')
    parser.add_argument('--compare_optimizers', action='store_true',
                       help='Compare different optimizers')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} does not exist")
        return
    
    # Set output directory
    if args.output_dir is None:
        output_dir = results_dir / 'plots'
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find experiment directories
    if args.compare_optimizers:
        exp_dirs = list(results_dir.glob(args.experiment_pattern))
    else:
        # If results_dir is itself an experiment dir
        if (results_dir / 'config.json').exists():
            exp_dirs = [results_dir]
        else:
            exp_dirs = list(results_dir.glob(args.experiment_pattern))
    
    if not exp_dirs:
        print(f"No experiment directories found matching pattern: {args.experiment_pattern}")
        return
    
    print(f"Found {len(exp_dirs)} experiment(s)")
    
    # Load experiments
    experiments = {}
    for exp_dir in exp_dirs:
        if not exp_dir.is_dir():
            continue
        name = exp_dir.name
        print(f"Loading {name}...")
        experiments[name] = load_experiment(exp_dir)
    
    if not experiments:
        print("No valid experiments loaded")
        return
    
    print(f"\nGenerating visualizations...")
    
    # Generate plots
    plot_training_curves(experiments, output_dir / 'training_curves.png')
    plot_agop_basic_metrics(experiments, output_dir / 'agop_basic_metrics.png')
    plot_agop_collapse_metrics(experiments, output_dir / 'agop_collapse_metrics.png')
    plot_agop_subspace_similarity(experiments, output_dir / 'agop_subspace_similarity.png')
    
    # If single experiment, also create enhanced visualizations
    if len(experiments) == 1:
        exp_name = list(experiments.keys())[0]
        exp_data = experiments[exp_name]
        
        # Original combined plot
        plot_combined_grokking_agop(exp_data, output_dir / 'combined_grokking_agop.png')
        
        # NEW: Comprehensive timeline
        plot_comprehensive_timeline(exp_data, output_dir / 'comprehensive_timeline.png')
        
        # NEW: Dual-axis plots for key metrics
        plot_aligned_dual_axis(
            exp_data, 
            'agop_variation_collapse_ratio', 
            'VCR (λ₁ / Σλᵢ)',
            output_dir / 'aligned_test_acc_vcr.png'
        )
        plot_aligned_dual_axis(
            exp_data, 
            'agop_eigengap', 
            'Eigengap (λ₁ - λ₂)',
            output_dir / 'aligned_test_acc_eigengap.png'
        )
        plot_aligned_dual_axis(
            exp_data, 
            'agop_trace', 
            'Trace (Σλᵢ)',
            output_dir / 'aligned_test_acc_trace.png'
        )
        plot_aligned_dual_axis(
            exp_data, 
            'agop_spectral_radius', 
            'Spectral Radius (λ₁)',
            output_dir / 'aligned_test_acc_spectral.png'
        )
    
    print(f"\n{'='*80}")
    print(f"Visualization complete! Plots saved to {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

