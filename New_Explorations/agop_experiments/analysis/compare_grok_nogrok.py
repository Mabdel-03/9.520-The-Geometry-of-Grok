"""
Compare AGOP metrics between grokking and non-grokking runs

This script analyzes differences in AGOP metrics between runs that grok
(test acc > 95%) and runs that don't grok, to identify predictive signatures.

Usage:
    python compare_grok_nogrok.py --results_dir ./results/agop_experiments/nanda
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


def load_experiment(exp_dir: Path) -> Dict:
    """Load experiment data"""
    data = {}
    
    config_path = exp_dir / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            data['config'] = json.load(f)
    
    history_path = exp_dir / 'training_history.json'
    if history_path.exists():
        with open(history_path, 'r') as f:
            data['history'] = json.load(f)
    
    agop_path = exp_dir / 'agop_metrics.h5'
    if agop_path.exists():
        data['agop'] = {}
        with h5py.File(agop_path, 'r') as f:
            for key in f.keys():
                data['agop'][key] = f[key][:]
    
    return data


def classify_grokking(exp_data: Dict, threshold: float = 0.95) -> bool:
    """Determine if experiment achieved grokking"""
    history = exp_data.get('history', {})
    test_acc = history.get('test_acc', [])
    
    if not test_acc:
        return False
    
    final_acc = test_acc[-1]
    return final_acc > threshold


def find_grokking_epoch(exp_data: Dict, threshold: float = 0.95) -> int:
    """Find epoch where grokking occurred (test acc > threshold)"""
    history = exp_data.get('history', {})
    test_acc = history.get('test_acc', [])
    epochs = history.get('epoch', list(range(len(test_acc))))
    
    for i, acc in enumerate(test_acc):
        if acc > threshold:
            return epochs[i]
    
    return -1


def compute_metric_statistics(grok_experiments: List[Dict], nogrok_experiments: List[Dict],
                              metric_name: str) -> Dict:
    """Compute statistics for a metric across grok vs no-grok experiments"""
    stats = {
        'grok_mean': [],
        'grok_std': [],
        'nogrok_mean': [],
        'nogrok_std': [],
        'epochs': None
    }
    
    # Collect grok data
    grok_data = []
    for exp in grok_experiments:
        if 'agop' in exp and metric_name in exp['agop']:
            grok_data.append(exp['agop'][metric_name])
            if stats['epochs'] is None and 'epoch' in exp['agop']:
                stats['epochs'] = exp['agop']['epoch']
    
    # Collect no-grok data
    nogrok_data = []
    for exp in nogrok_experiments:
        if 'agop' in exp and metric_name in exp['agop']:
            nogrok_data.append(exp['agop'][metric_name])
    
    # Compute statistics
    if grok_data:
        # Ensure all arrays have same length
        min_len = min(len(d) for d in grok_data)
        grok_data_aligned = np.array([d[:min_len] for d in grok_data])
        stats['grok_mean'] = np.mean(grok_data_aligned, axis=0)
        stats['grok_std'] = np.std(grok_data_aligned, axis=0)
    
    if nogrok_data:
        min_len = min(len(d) for d in nogrok_data)
        nogrok_data_aligned = np.array([d[:min_len] for d in nogrok_data])
        stats['nogrok_mean'] = np.mean(nogrok_data_aligned, axis=0)
        stats['nogrok_std'] = np.std(nogrok_data_aligned, axis=0)
    
    return stats


def plot_comparison(grok_experiments: List[Dict], nogrok_experiments: List[Dict],
                   metric_name: str, ylabel: str, title: str, save_path: Path):
    """Plot comparison of a metric between grok and no-grok experiments"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    stats = compute_metric_statistics(grok_experiments, nogrok_experiments, metric_name)
    
    if len(stats['grok_mean']) > 0:
        epochs = stats['epochs'][:len(stats['grok_mean'])] if stats['epochs'] is not None else np.arange(len(stats['grok_mean']))
        ax.plot(epochs, stats['grok_mean'], 'g-', linewidth=2, label='Grokking (mean)')
        ax.fill_between(epochs,
                       stats['grok_mean'] - stats['grok_std'],
                       stats['grok_mean'] + stats['grok_std'],
                       alpha=0.3, color='g')
    
    if len(stats['nogrok_mean']) > 0:
        epochs = stats['epochs'][:len(stats['nogrok_mean'])] if stats['epochs'] is not None else np.arange(len(stats['nogrok_mean']))
        ax.plot(epochs, stats['nogrok_mean'], 'r-', linewidth=2, label='No Grokking (mean)')
        ax.fill_between(epochs,
                       stats['nogrok_mean'] - stats['nogrok_std'],
                       stats['nogrok_mean'] + stats['nogrok_std'],
                       alpha=0.3, color='r')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {save_path}")
    plt.close()


def create_summary_report(grok_experiments: List[Dict], nogrok_experiments: List[Dict],
                         output_path: Path):
    """Create text summary report comparing grok vs no-grok"""
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("AGOP Metrics: Grokking vs Non-Grokking Comparison\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Grokking experiments: {len(grok_experiments)}\n")
        f.write(f"Non-grokking experiments: {len(nogrok_experiments)}\n\n")
        
        # Grokking experiments details
        f.write("Grokking Experiments:\n")
        f.write("-" * 80 + "\n")
        for i, exp in enumerate(grok_experiments):
            config = exp.get('config', {})
            history = exp.get('history', {})
            grok_epoch = find_grokking_epoch(exp)
            final_acc = history.get('test_acc', [0])[-1] if history.get('test_acc') else 0
            
            f.write(f"{i+1}. Optimizer: {config.get('optimizer', 'unknown')}, "
                   f"LR: {config.get('lr', 'unknown')}, "
                   f"WD: {config.get('weight_decay', 'unknown')}\n")
            f.write(f"   Grokking at epoch: {grok_epoch}, Final test acc: {final_acc:.4f}\n")
        
        f.write("\nNon-Grokking Experiments:\n")
        f.write("-" * 80 + "\n")
        for i, exp in enumerate(nogrok_experiments):
            config = exp.get('config', {})
            history = exp.get('history', {})
            final_acc = history.get('test_acc', [0])[-1] if history.get('test_acc') else 0
            
            f.write(f"{i+1}. Optimizer: {config.get('optimizer', 'unknown')}, "
                   f"LR: {config.get('lr', 'unknown')}, "
                   f"WD: {config.get('weight_decay', 'unknown')}\n")
            f.write(f"   Final test acc: {final_acc:.4f}\n")
        
        # Metric comparisons at final epoch
        f.write("\n" + "="*80 + "\n")
        f.write("AGOP Metrics at Final Epoch:\n")
        f.write("="*80 + "\n\n")
        
        metrics_to_compare = [
            'agop_trace', 'agop_spectral_radius', 'agop_eigengap',
            'agop_variation_collapse_ratio', 'agop_top_eigenvalue_energy'
        ]
        
        for metric in metrics_to_compare:
            # Get final values
            grok_finals = []
            for exp in grok_experiments:
                if 'agop' in exp and metric in exp['agop']:
                    grok_finals.append(exp['agop'][metric][-1])
            
            nogrok_finals = []
            for exp in nogrok_experiments:
                if 'agop' in exp and metric in exp['agop']:
                    nogrok_finals.append(exp['agop'][metric][-1])
            
            if grok_finals or nogrok_finals:
                f.write(f"{metric}:\n")
                if grok_finals:
                    f.write(f"  Grokking:     mean={np.mean(grok_finals):.4e}, std={np.std(grok_finals):.4e}\n")
                if nogrok_finals:
                    f.write(f"  Non-grokking: mean={np.mean(nogrok_finals):.4e}, std={np.std(nogrok_finals):.4e}\n")
                
                if grok_finals and nogrok_finals:
                    diff = np.mean(grok_finals) - np.mean(nogrok_finals)
                    f.write(f"  Difference:   {diff:.4e}\n")
                f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("End of Report\n")
        f.write("="*80 + "\n")
    
    print(f"Saved summary report to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare AGOP metrics between grokking and non-grokking runs')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing experiment results')
    parser.add_argument('--experiment_pattern', type=str, default='*',
                       help='Pattern to match experiment directories')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save analysis (default: results_dir/analysis)')
    parser.add_argument('--grok_threshold', type=float, default=0.95,
                       help='Test accuracy threshold for grokking')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} does not exist")
        return
    
    # Set output directory
    if args.output_dir is None:
        output_dir = results_dir / 'analysis'
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find experiment directories
    exp_dirs = list(results_dir.glob(args.experiment_pattern))
    if not exp_dirs:
        print(f"No experiment directories found")
        return
    
    print(f"Found {len(exp_dirs)} experiment(s)")
    
    # Load and classify experiments
    grok_experiments = []
    nogrok_experiments = []
    
    for exp_dir in exp_dirs:
        if not exp_dir.is_dir():
            continue
        
        print(f"Loading {exp_dir.name}...")
        exp_data = load_experiment(exp_dir)
        
        if classify_grokking(exp_data, args.grok_threshold):
            grok_experiments.append(exp_data)
            print(f"  -> Grokking")
        else:
            nogrok_experiments.append(exp_data)
            print(f"  -> No grokking")
    
    print(f"\nGrokking: {len(grok_experiments)}, Non-grokking: {len(nogrok_experiments)}")
    
    if len(grok_experiments) == 0 and len(nogrok_experiments) == 0:
        print("No experiments to compare")
        return
    
    # Generate comparison plots
    print("\nGenerating comparison plots...")
    
    metrics_to_plot = [
        ('agop_variation_collapse_ratio', 'VCR (λ₁ / Σλᵢ)', 'Variation Collapse Ratio'),
        ('agop_eigengap', 'Eigengap (λ₁ - λ₂)', 'AGOP Eigengap'),
        ('agop_trace', 'Trace (Σλᵢ)', 'AGOP Trace'),
        ('agop_spectral_radius', 'Spectral Radius (λ₁)', 'AGOP Spectral Radius'),
        ('agop_top_eigenvalue_energy', 'Energy Ratio', 'Top Eigenvalue Energy Concentration'),
    ]
    
    for metric_name, ylabel, title in metrics_to_plot:
        save_path = output_dir / f'comparison_{metric_name}.png'
        plot_comparison(grok_experiments, nogrok_experiments, metric_name, 
                       ylabel, f'{title}: Grokking vs Non-Grokking', save_path)
    
    # Generate summary report
    create_summary_report(grok_experiments, nogrok_experiments, 
                         output_dir / 'comparison_summary.txt')
    
    print(f"\n{'='*80}")
    print(f"Comparison analysis complete! Results saved to {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

