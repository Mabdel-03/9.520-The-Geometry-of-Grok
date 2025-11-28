#!/usr/bin/env python3
"""
Comprehensive AGOP Results Checker

Analyzes all AGOP experiments and creates detailed status tables.
"""

import json
from pathlib import Path
import h5py
import numpy as np
from typing import Dict, List, Tuple


def check_experiment(exp_dir: Path) -> Dict:
    """Check status and results of a single experiment."""
    status = {
        'name': exp_dir.name,
        'has_config': (exp_dir / 'config.json').exists(),
        'has_history': (exp_dir / 'training_history.json').exists(),
        'has_agop': (exp_dir / 'agop_metrics.h5').exists(),
        'final_train_acc': None,
        'final_test_acc': None,
        'grokked': False,
        'grok_epoch': None,
        'n_epochs_completed': 0,
        'agop_samples': 0,
    }
    
    # Read training history
    history_path = exp_dir / 'training_history.json'
    if history_path.exists():
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            if history['epoch']:
                status['n_epochs_completed'] = len(history['epoch'])
                status['final_train_acc'] = history['train_acc'][-1]
                status['final_test_acc'] = history['test_acc'][-1]
                
                # Check for grokking (>90% test acc)
                if status['final_test_acc'] > 0.9:
                    status['grokked'] = True
                    for i, acc in enumerate(history['test_acc']):
                        if acc > 0.9:
                            status['grok_epoch'] = history['epoch'][i]
                            break
        except Exception as e:
            status['error'] = f"Error reading history: {e}"
    
    # Read AGOP metrics
    agop_path = exp_dir / 'agop_metrics.h5'
    if agop_path.exists():
        try:
            with h5py.File(agop_path, 'r') as f:
                if 'epoch' in f:
                    status['agop_samples'] = len(f['epoch'][:])
        except Exception as e:
            status['agop_error'] = f"Error reading AGOP: {e}"
    
    return status


def analyze_dataset(dataset_dir: Path) -> Tuple[List[Dict], Dict]:
    """Analyze all experiments for a dataset."""
    experiments = []
    
    for exp_dir in sorted(dataset_dir.iterdir()):
        if exp_dir.is_dir():
            status = check_experiment(exp_dir)
            experiments.append(status)
    
    # Summary stats
    summary = {
        'total': len(experiments),
        'has_config': sum(1 for e in experiments if e['has_config']),
        'has_history': sum(1 for e in experiments if e['has_history']),
        'has_agop': sum(1 for e in experiments if e['has_agop']),
        'complete': sum(1 for e in experiments if e['has_history'] and e['has_agop']),
        'grokked': sum(1 for e in experiments if e['grokked']),
    }
    
    return experiments, summary


def main():
    results_dir = Path('/om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/agop_experiments/results')
    
    if not results_dir.exists():
        print("Results directory not found!")
        return
    
    print("="*100)
    print(" "*30 + "AGOP EXPERIMENTS - COMPREHENSIVE STATUS")
    print("="*100)
    
    datasets = ['nanda', 'softmax', 'mnist', 'composition']
    all_summaries = {}
    
    for dataset in datasets:
        dataset_dir = results_dir / dataset
        if not dataset_dir.exists():
            print(f"\n❌ {dataset.upper()}: Directory not found")
            continue
        
        experiments, summary = analyze_dataset(dataset_dir)
        all_summaries[dataset] = summary
        
        print(f"\n{'='*100}")
        print(f"📁 {dataset.upper()}")
        print(f"{'='*100}")
        print(f"Total experiments: {summary['total']}")
        print(f"Complete (history + AGOP): {summary['complete']}/{summary['total']}")
        print(f"Grokked: {summary['grokked']}/{summary['complete']}")
        print(f"-"*100)
        
        # Print table header
        print(f"{'Experiment':<60} {'Epochs':>8} {'Train Acc':>10} {'Test Acc':>10} {'AGOP':>6} {'Status':>12}")
        print(f"{'-'*60} {'-'*8} {'-'*10} {'-'*10} {'-'*6} {'-'*12}")
        
        for exp in experiments:
            name = exp['name'][:58]
            epochs = exp['n_epochs_completed'] if exp['n_epochs_completed'] > 0 else '-'
            train_acc = f"{exp['final_train_acc']:.4f}" if exp['final_train_acc'] is not None else 'N/A'
            test_acc = f"{exp['final_test_acc']:.4f}" if exp['final_test_acc'] is not None else 'N/A'
            agop = '✓' if exp['has_agop'] else '✗'
            
            if exp['grokked']:
                status = f"✅ @{exp['grok_epoch']}"
            elif exp['has_history'] and exp['has_agop']:
                status = "✓ Complete"
            elif exp['has_history']:
                status = "⚠️ No AGOP"
            elif exp['has_config']:
                status = "⏳ Running"
            else:
                status = "❌ Missing"
            
            print(f"{name:<60} {str(epochs):>8} {train_acc:>10} {test_acc:>10} {agop:>6} {status:>12}")
    
    # Overall summary
    print(f"\n{'='*100}")
    print("OVERALL SUMMARY")
    print(f"{'='*100}")
    
    total_expected = 72  # 24 + 24 + 12 + 12
    total_complete = sum(s['complete'] for s in all_summaries.values())
    total_grokked = sum(s['grokked'] for s in all_summaries.values())
    
    print(f"Expected experiments: {total_expected}")
    print(f"Complete (with AGOP): {total_complete} ({total_complete/total_expected*100:.1f}%)")
    print(f"Grokked: {total_grokked}")
    print(f"Still running/queued: {total_expected - total_complete}")
    
    print(f"\n{'='*100}")


if __name__ == "__main__":
    main()

