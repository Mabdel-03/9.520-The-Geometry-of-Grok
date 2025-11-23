#!/usr/bin/env python3
"""
Quick script to check status of all experiments
Shows which experiments have been run and their final accuracies
"""

import json
from pathlib import Path
import argparse
from typing import Dict, List, Tuple


def check_experiment_status(exp_dir: Path) -> Dict:
    """Check status of a single experiment."""
    status = {
        'name': exp_dir.name,
        'exists': exp_dir.exists(),
        'completed': False,
        'final_train_acc': None,
        'final_test_acc': None,
        'grokked': False,
        'grok_epoch': None,
    }
    
    if not exp_dir.exists():
        return status
    
    # Check if training history exists
    history_path = exp_dir / 'training_history.json'
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        if history['epoch']:
            status['completed'] = True
            status['final_train_acc'] = history['train_acc'][-1]
            status['final_test_acc'] = history['test_acc'][-1]
            
            # Detect grokking
            if status['final_test_acc'] > 0.9:
                status['grokked'] = True
                # Find when test acc first exceeded 0.9
                for i, acc in enumerate(history['test_acc']):
                    if acc > 0.9:
                        status['grok_epoch'] = history['epoch'][i]
                        break
    
    return status


def main():
    parser = argparse.ArgumentParser(description='Check status of all experiments')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Directory containing results')
    parser.add_argument('--paper', type=str, default=None,
                       choices=['paper03_nanda', 'paper05_omnigrok', 'paper04_wang'],
                       help='Specific paper to check (or None for all)')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("No experiments have been run yet.")
        return
    
    # Determine which papers to check
    if args.paper:
        paper_dirs = [results_dir / args.paper]
    else:
        paper_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    
    print("="*80)
    print("EXPERIMENT STATUS REPORT")
    print("="*80)
    
    for paper_dir in sorted(paper_dirs):
        if not paper_dir.exists():
            continue
        
        print(f"\n📁 {paper_dir.name}")
        print("-"*80)
        
        # Find all experiments
        experiments = [d for d in paper_dir.iterdir() if d.is_dir()]
        
        if not experiments:
            print("  No experiments found.")
            continue
        
        # Check status of each
        statuses = []
        for exp_dir in sorted(experiments):
            status = check_experiment_status(exp_dir)
            statuses.append(status)
        
        # Print summary table
        completed = [s for s in statuses if s['completed']]
        grokked = [s for s in statuses if s['grokked']]
        
        print(f"  Total experiments: {len(statuses)}")
        print(f"  Completed: {len(completed)}")
        print(f"  Grokked: {len(grokked)}")
        print()
        
        # Print details
        print(f"  {'Experiment':<40} {'Status':<12} {'Train Acc':<12} {'Test Acc':<12} {'Grok Epoch':<12}")
        print(f"  {'-'*40} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        
        for status in sorted(statuses, key=lambda x: x['name']):
            name = status['name'][:38]
            
            if status['completed']:
                status_str = "✓ Complete"
                train_acc = f"{status['final_train_acc']:.4f}" if status['final_train_acc'] is not None else "N/A"
                test_acc = f"{status['final_test_acc']:.4f}" if status['final_test_acc'] is not None else "N/A"
                grok_str = str(status['grok_epoch']) if status['grokked'] else "No"
            else:
                status_str = "✗ Incomplete"
                train_acc = "N/A"
                test_acc = "N/A"
                grok_str = "N/A"
            
            print(f"  {name:<40} {status_str:<12} {train_acc:<12} {test_acc:<12} {grok_str:<12}")
    
    print("\n" + "="*80)
    print("Legend:")
    print("  ✓ Complete  : Experiment finished successfully")
    print("  ✗ Incomplete: Experiment not yet run or crashed")
    print("  Grok Epoch  : First epoch when test accuracy exceeded 90%")
    print("="*80)


if __name__ == "__main__":
    main()

