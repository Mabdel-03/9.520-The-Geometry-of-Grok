"""
Visualization Tools for Gradient Outer Product Analysis

Provides tools to visualize GOP evolution during training and
identify grokking transitions.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_experiment_metrics(results_dir: str) -> dict:
    """Load all metrics from an experiment."""
    results_path = Path(results_dir)
    metrics_file = results_path / "metrics.h5"
    
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
    
    metrics = {}
    with h5py.File(metrics_file, 'r') as f:
        for key in f.keys():
            metrics[key] = f[key][:]
    
    return metrics


def plot_training_curves(metrics: dict, save_path: Optional[str] = None):
    """Plot training and test loss/accuracy curves."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Get epochs (x-axis)
    n_epochs = len(metrics.get('train_loss', []))
    epochs = np.arange(n_epochs)
    
    # Plot train loss
    if 'train_loss' in metrics:
        axes[0, 0].plot(epochs, metrics['train_loss'], label='Train Loss', linewidth=1)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
    
    # Plot test loss
    if 'test_loss' in metrics:
        axes[0, 1].plot(epochs, metrics['test_loss'], label='Test Loss', color='orange', linewidth=1)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Test Loss')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
    
    # Plot train accuracy
    if 'train_acc' in metrics:
        axes[1, 0].plot(epochs, metrics['train_acc'], label='Train Accuracy', linewidth=1)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Training Accuracy')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
    
    # Plot test accuracy
    if 'test_acc' in metrics:
        axes[1, 1].plot(epochs, metrics['test_acc'], label='Test Accuracy', color='orange', linewidth=1)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Test Accuracy (Grokking)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        # Highlight grokking point if visible
        test_acc = metrics['test_acc']
        if len(test_acc) > 0 and np.max(test_acc) > 0.9:
            grok_idx = np.where(test_acc > 0.9)[0][0] if np.any(test_acc > 0.9) else None
            if grok_idx:
                axes[1, 1].axvline(grok_idx, color='red', linestyle='--', alpha=0.5, label=f'Grokking at epoch {grok_idx}')
                axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved training curves to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_gop_metrics(metrics: dict, save_path: Optional[str] = None):
    """Plot GOP-specific metrics evolution."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    n_epochs = len(metrics.get('train_loss', []))
    epochs = np.arange(n_epochs)
    
    gop_metric_names = [
        ('gop_trace', 'GOP Trace'),
        ('gop_frobenius_norm', 'GOP Frobenius Norm'),
        ('gop_spectral_norm', 'GOP Spectral Norm'),
        ('gop_rank', 'GOP Effective Rank'),
        ('gop_condition_number', 'GOP Condition Number'),
        ('gop_top_k_cumulative_variance', 'Top-K Eigenvalue Variance')
    ]
    
    for idx, (metric_key, title) in enumerate(gop_metric_names):
        ax = axes[idx // 3, idx % 3]
        
        if metric_key in metrics:
            data = metrics[metric_key]
            ax.plot(epochs, data, linewidth=1.5, color='blue')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
            # Use log scale for some metrics
            if metric_key in ['gop_trace', 'gop_frobenius_norm', 'gop_condition_number']:
                ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved GOP metrics to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_eigenvalue_spectrum(results_dir: str, epochs: List[int], save_path: Optional[str] = None):
    """
    Plot eigenvalue spectrum at different epochs.
    
    Args:
        results_dir: Path to results directory
        epochs: List of epochs to plot
        save_path: Optional save path
    """
    results_path = Path(results_dir)
    gop_full_file = results_path / "gop_full.h5"
    
    if not gop_full_file.exists():
        logger.warning(f"GOP full file not found: {gop_full_file}")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    with h5py.File(gop_full_file, 'r') as f:
        for epoch in epochs:
            epoch_key = f'epoch_{epoch}'
            if epoch_key in f and 'eigenvalues' in f[epoch_key]:
                eigenvalues = f[epoch_key]['eigenvalues'][:]
                # Plot eigenvalue spectrum
                ax.semilogy(eigenvalues, label=f'Epoch {epoch}', alpha=0.7)
    
    ax.set_xlabel('Eigenvalue Index')
    ax.set_ylabel('Eigenvalue Magnitude')
    ax.set_title('Eigenvalue Spectrum Evolution')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved eigenvalue spectrum to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_top_eigenvalues_evolution(metrics: dict, top_n: int = 10, save_path: Optional[str] = None):
    """Plot evolution of top-N eigenvalues over training."""
    if 'eigenvalues_top_k' not in metrics:
        logger.warning("No top-k eigenvalues found in metrics")
        return
    
    eigenvalues_over_time = metrics['eigenvalues_top_k']  # Shape: (n_epochs, k)
    n_epochs = eigenvalues_over_time.shape[0]
    epochs = np.arange(n_epochs)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i in range(min(top_n, eigenvalues_over_time.shape[1])):
        ax.plot(epochs, eigenvalues_over_time[:, i], label=f'λ_{i+1}', alpha=0.7)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Eigenvalue')
    ax.set_title(f'Evolution of Top-{top_n} Eigenvalues')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved top eigenvalues evolution to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_summary_plot(results_dir: str, save_path: Optional[str] = None):
    """Create comprehensive summary plot combining all metrics."""
    metrics = load_experiment_metrics(results_dir)
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    n_epochs = len(metrics.get('train_loss', []))
    epochs = np.arange(n_epochs)
    
    # Row 1: Loss and Accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    if 'train_loss' in metrics and 'test_loss' in metrics:
        ax1.plot(epochs, metrics['train_loss'], label='Train', alpha=0.7)
        ax1.plot(epochs, metrics['test_loss'], label='Test', alpha=0.7)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Dynamics')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    if 'train_acc' in metrics and 'test_acc' in metrics:
        ax2.plot(epochs, metrics['train_acc'], label='Train', alpha=0.7)
        ax2.plot(epochs, metrics['test_acc'], label='Test', alpha=0.7)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy (Grokking)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[0, 2])
    if 'eigenvalues_top_k' in metrics:
        eigs = metrics['eigenvalues_top_k']
        for i in range(min(5, eigs.shape[1])):
            ax3.plot(epochs, eigs[:, i], label=f'λ_{i+1}', alpha=0.7)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Eigenvalue')
        ax3.set_title('Top-5 Eigenvalues')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Row 2: GOP Norms
    ax4 = fig.add_subplot(gs[1, 0])
    if 'gop_trace' in metrics:
        ax4.plot(epochs, metrics['gop_trace'], color='purple', linewidth=1.5)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Trace')
        ax4.set_title('GOP Trace')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[1, 1])
    if 'gop_frobenius_norm' in metrics:
        ax5.plot(epochs, metrics['gop_frobenius_norm'], color='green', linewidth=1.5)
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Frobenius Norm')
        ax5.set_title('GOP Frobenius Norm')
        ax5.set_yscale('log')
        ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(gs[1, 2])
    if 'gop_spectral_norm' in metrics:
        ax6.plot(epochs, metrics['gop_spectral_norm'], color='red', linewidth=1.5)
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Spectral Norm')
        ax6.set_title('GOP Spectral Norm')
        ax6.set_yscale('log')
        ax6.grid(True, alpha=0.3)
    
    # Row 3: GOP Properties
    ax7 = fig.add_subplot(gs[2, 0])
    if 'gop_rank' in metrics:
        ax7.plot(epochs, metrics['gop_rank'], color='brown', linewidth=1.5)
        ax7.set_xlabel('Epoch')
        ax7.set_ylabel('Rank')
        ax7.set_title('GOP Effective Rank')
        ax7.grid(True, alpha=0.3)
    
    ax8 = fig.add_subplot(gs[2, 1])
    if 'gop_condition_number' in metrics:
        ax8.plot(epochs, metrics['gop_condition_number'], color='orange', linewidth=1.5)
        ax8.set_xlabel('Epoch')
        ax8.set_ylabel('Condition Number')
        ax8.set_title('GOP Condition Number')
        ax8.set_yscale('log')
        ax8.grid(True, alpha=0.3)
    
    ax9 = fig.add_subplot(gs[2, 2])
    if 'gop_top_k_cumulative_variance' in metrics:
        ax9.plot(epochs, metrics['gop_top_k_cumulative_variance'], color='teal', linewidth=1.5)
        ax9.set_xlabel('Epoch')
        ax9.set_ylabel('Cumulative Variance')
        ax9.set_title('Top-K Eigenvalue Explained Variance')
        ax9.grid(True, alpha=0.3)
    
    plt.suptitle('Gradient Outer Product Analysis', fontsize=16, y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved summary plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def detect_grokking_point(metrics: dict) -> Optional[int]:
    """
    Automatically detect grokking transition point.
    
    Grokking is identified as when test accuracy jumps above threshold
    after remaining low for extended period.
    
    Args:
        metrics: Metrics dictionary
        
    Returns:
        Epoch number of grokking transition or None
    """
    if 'test_acc' not in metrics:
        return None
    
    test_acc = metrics['test_acc']
    threshold = 0.9
    
    # Find first epoch where test acc exceeds threshold
    above_threshold = np.where(test_acc > threshold)[0]
    
    if len(above_threshold) > 0:
        grok_epoch = above_threshold[0]
        logger.info(f"Grokking detected at epoch {grok_epoch}")
        return int(grok_epoch)
    
    return None


def main():
    parser = argparse.ArgumentParser(description='Visualize GOP analysis results')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Path to results directory')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save plots (default: results_dir/plots)')
    parser.add_argument('--epochs', type=int, nargs='+', default=None,
                       help='Specific epochs to plot eigenvalue spectra')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return
    
    # Create save directory
    save_dir = Path(args.save_dir) if args.save_dir else results_dir / 'plots'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metrics
    logger.info(f"Loading metrics from {results_dir}")
    metrics = load_experiment_metrics(str(results_dir))
    
    # Detect grokking
    grok_epoch = detect_grokking_point(metrics)
    if grok_epoch:
        logger.info(f"Grokking detected at epoch {grok_epoch}")
    
    # Create plots
    logger.info("Creating training curves...")
    plot_training_curves(metrics, save_path=str(save_dir / 'training_curves.png'))
    
    logger.info("Creating GOP metrics plot...")
    plot_gop_metrics(metrics, save_path=str(save_dir / 'gop_metrics.png'))
    
    logger.info("Creating summary plot...")
    create_summary_plot(str(results_dir), save_path=str(save_dir / 'summary.png'))
    
    # Plot eigenvalue evolution
    if 'eigenvalues_top_k' in metrics:
        logger.info("Creating top eigenvalues evolution plot...")
        plot_top_eigenvalues_evolution(metrics, top_n=10, save_path=str(save_dir / 'top_eigenvalues.png'))
    
    # Plot eigenvalue spectrum at key epochs
    if args.epochs:
        logger.info(f"Plotting eigenvalue spectra at epochs: {args.epochs}")
        plot_eigenvalue_spectrum(str(results_dir), args.epochs, save_path=str(save_dir / 'eigenvalue_spectrum.png'))
    elif grok_epoch:
        # Plot spectrum before, during, and after grokking
        key_epochs = [
            max(0, grok_epoch - 1000),
            grok_epoch,
            min(len(metrics['test_acc']) - 1, grok_epoch + 1000)
        ]
        plot_eigenvalue_spectrum(str(results_dir), key_epochs, save_path=str(save_dir / 'eigenvalue_spectrum_grokking.png'))
    
    logger.info(f"All plots saved to {save_dir}")


if __name__ == '__main__':
    main()


from typing import Optional, List

