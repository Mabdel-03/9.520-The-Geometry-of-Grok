"""
Master visualization script for Paper 06 adversarial robustness results
Creates publication-quality plots for all experiments
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Add the 06_humayun directory to path
sys.path.insert(0, str(Path(__file__).parent / '06_humayun_et_al_2024_deep_networks'))
from plot_delayed_robustness import plot_delayed_robustness, plot_comparison


def main():
    base_dir = Path(__file__).parent / '06_humayun_et_al_2024_deep_networks' / 'results'
    output_dir = Path(__file__).parent / 'analysis_results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Individual experiment plots
    experiments = [
        ('mnist_mlp_adv', 'MNIST + MLP'),
        ('cifar10_cnn', 'CIFAR-10 + CNN'),
        ('imagenette_resnet', 'Imagenette + ResNet-18')
    ]
    
    available_experiments = []
    history_paths = []
    labels = []
    
    for exp_dir, label in experiments:
        history_path = base_dir / exp_dir / 'training_history.json'
        if history_path.exists():
            print(f"\nProcessing {label}...")
            output_path = output_dir / f'paper_06_{exp_dir}_delayed_robustness.png'
            plot_delayed_robustness(history_path, output_path, title=label)
            available_experiments.append(exp_dir)
            history_paths.append(history_path)
            labels.append(label)
        else:
            print(f"\n{label}: Experiment not yet complete (history file not found)")
    
    # Create comparison plot if multiple experiments are complete
    if len(available_experiments) >= 2:
        print("\nCreating comparison plot...")
        comparison_path = output_dir / 'paper_06_all_experiments_comparison.png'
        plot_comparison(history_paths, labels, comparison_path, 
                       title="Deep Networks Always Grok: Delayed Robustness Across Datasets")
    
    print(f"\n{'='*60}")
    print("Paper 06 Visualization Complete!")
    print(f"{'='*60}")
    print(f"Completed experiments: {', '.join(available_experiments)}")
    print(f"Plots saved to: {output_dir}")


if __name__ == '__main__':
    main()

