"""
Compare GOP metrics across multiple experiments

Allows overlaying and comparing grokking dynamics across different
papers, datasets, and hyperparameters.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_experiment_summary(results_dir: str) -> Dict:
    """Load summary data from an experiment."""
    results_path = Path(results_dir)
    metrics_file = results_path / "metrics.h5"
    
    summary = {
        'name': results_path.name,
        'path': str(results_path)
    }
    
    if metrics_file.exists():
        with h5py.File(metrics_file, 'r') as f:
            # Load key metrics
            for key in ['train_loss', 'test_loss', 'train_acc', 'test_acc',
                       'gop_trace', 'gop_spectral_norm', 'gop_rank']:
                if key in f:
                    summary[key] = f[key][:]
    
    return summary


def compare_test_accuracy(experiments: List[Dict], save_path: Optional[str] = None):
    """Compare test accuracy curves (grokking) across experiments."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for exp in experiments:
        if 'test_acc' in exp:
            n_epochs = len(exp['test_acc'])
            epochs = np.arange(n_epochs)
            ax.plot(epochs, exp['test_acc'], label=exp['name'], alpha=0.7, linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Grokking Comparison Across Experiments')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved comparison plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def compare_gop_metrics(experiments: List[Dict], metric_name: str, save_path: Optional[str] = None):
    """Compare a specific GOP metric across experiments."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for exp in experiments:
        if metric_name in exp:
            n_epochs = len(exp[metric_name])
            epochs = np.arange(n_epochs)
            ax.plot(epochs, exp[metric_name], label=exp['name'], alpha=0.7, linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric_name.replace('gop_', '').replace('_', ' ').title())
    ax.set_title(f'{metric_name.replace("gop_", "GOP ").replace("_", " ").title()} Comparison')
    
    # Use log scale for certain metrics
    if metric_name in ['gop_trace', 'gop_frobenius_norm', 'gop_condition_number']:
        ax.set_yscale('log')
    
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare GOP analysis across experiments')
    parser.add_argument('--experiments', type=str, nargs='+', required=True,
                       help='List of experiment result directories')
    parser.add_argument('--save_dir', type=str, default='./comparison_plots',
                       help='Directory to save comparison plots')
    parser.add_argument('--metrics', type=str, nargs='*',
                       default=['test_acc', 'gop_trace', 'gop_spectral_norm', 'gop_rank'],
                       help='Metrics to compare')
    
    args = parser.parse_args()
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all experiments
    logger.info(f"Loading {len(args.experiments)} experiments...")
    experiment_data = []
    for exp_path in args.experiments:
        try:
            exp_data = load_experiment_summary(exp_path)
            experiment_data.append(exp_data)
            logger.info(f"Loaded: {exp_data['name']}")
        except Exception as e:
            logger.error(f"Failed to load {exp_path}: {e}")
    
    if not experiment_data:
        logger.error("No experiments loaded successfully")
        return
    
    # Create comparison plots
    logger.info("Creating comparison plots...")
    
    # Test accuracy comparison (grokking)
    compare_test_accuracy(experiment_data, save_path=str(save_dir / 'test_acc_comparison.png'))
    
    # Compare requested GOP metrics
    for metric in args.metrics:
        if metric != 'test_acc':  # Already plotted
            try:
                compare_gop_metrics(experiment_data, metric, save_path=str(save_dir / f'{metric}_comparison.png'))
            except Exception as e:
                logger.warning(f"Could not plot {metric}: {e}")
    
    logger.info(f"All comparison plots saved to {save_dir}")


if __name__ == '__main__':
    main()

