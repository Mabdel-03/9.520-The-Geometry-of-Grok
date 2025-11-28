"""
Test script to validate the analysis infrastructure

Tests:
1. Can load experiments from both datasets
2. Can generate summary tables
3. Can classify grokking
4. Can create basic plots
5. Analysis utilities work correctly
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

# Import analysis utilities
sys.path.append(str(Path(__file__).parent))
from analysis_utils import (
    load_all_experiments, generate_summary_table, classify_grokking,
    compute_time_to_grok, statistical_comparison, filter_experiments,
    smooth_series, compute_correlation
)

def test_load_experiments():
    """Test loading experiments"""
    print("="*80)
    print("TEST 1: Loading Experiments")
    print("="*80)
    
    nanda_dir = Path(__file__).parent.parent / 'results' / 'nanda'
    softmax_dir = Path(__file__).parent.parent / 'results' / 'softmax'
    
    try:
        # Load Nanda
        print("\nLoading Nanda experiments...")
        nanda_exps = load_all_experiments(nanda_dir)
        print(f"✓ Loaded {len(nanda_exps)} Nanda experiments")
        
        # Load Softmax
        print("\nLoading Softmax experiments...")
        softmax_exps = load_all_experiments(softmax_dir)
        print(f"✓ Loaded {len(softmax_exps)} Softmax experiments")
        
        # Check data structure
        if nanda_exps:
            first_exp = list(nanda_exps.values())[0]
            assert 'config' in first_exp, "Missing 'config' in experiment data"
            assert 'history' in first_exp, "Missing 'history' in experiment data"
            print("✓ Experiment data structure is valid")
        
        return nanda_exps, softmax_exps
    
    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_summary_table(experiments):
    """Test generating summary tables"""
    print("\n" + "="*80)
    print("TEST 2: Generating Summary Tables")
    print("="*80)
    
    try:
        df = generate_summary_table(experiments)
        print(f"✓ Generated summary table with {len(df)} rows")
        
        # Check required columns
        required_cols = ['experiment', 'architecture', 'optimizer', 'weight_decay', 
                        'final_test_acc', 'grokked', 'grok_epoch']
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"
        print(f"✓ All required columns present")
        
        # Print statistics
        print(f"\nSummary statistics:")
        print(f"  Total experiments: {len(df)}")
        print(f"  Grokked: {df['grokked'].sum()}")
        print(f"  Architectures: {df['architecture'].unique().tolist()}")
        print(f"  Optimizers: {df['optimizer'].unique().tolist()}")
        
        return df
    
    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_grokking_classification(experiments):
    """Test grokking classification"""
    print("\n" + "="*80)
    print("TEST 3: Grokking Classification")
    print("="*80)
    
    try:
        grok_count = 0
        nogrok_count = 0
        
        for exp_name, exp_data in experiments.items():
            grokked = classify_grokking(exp_data)
            if grokked:
                grok_count += 1
                grok_epoch = compute_time_to_grok(exp_data)
                assert grok_epoch > 0, f"Grokked but no grok epoch found for {exp_name}"
            else:
                nogrok_count += 1
        
        print(f"✓ Classified {len(experiments)} experiments")
        print(f"  Grokked: {grok_count}")
        print(f"  No grokking: {nogrok_count}")
        
        return grok_count, nogrok_count
    
    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_statistical_comparison(experiments):
    """Test statistical comparison functions"""
    print("\n" + "="*80)
    print("TEST 4: Statistical Comparisons")
    print("="*80)
    
    try:
        # Separate by optimizer
        adamw_exps = filter_experiments(experiments, optimizer='adamw')
        muon_exps = filter_experiments(experiments, optimizer='muon')
        
        # Get final accuracies
        adamw_acc = [exp['history']['test_acc'][-1] for exp in adamw_exps.values() 
                     if 'test_acc' in exp['history'] and exp['history']['test_acc']]
        muon_acc = [exp['history']['test_acc'][-1] for exp in muon_exps.values() 
                    if 'test_acc' in exp['history'] and exp['history']['test_acc']]
        
        if adamw_acc and muon_acc:
            stats = statistical_comparison(adamw_acc, muon_acc, ('AdamW', 'Muon'))
            print(f"✓ Statistical comparison successful")
            print(f"  AdamW mean: {stats['group1_mean']:.4f}")
            print(f"  Muon mean: {stats['group2_mean']:.4f}")
            print(f"  P-value: {stats['p_value']:.4f}")
        else:
            print("⚠ Not enough data for comparison")
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_utility_functions():
    """Test utility functions"""
    print("\n" + "="*80)
    print("TEST 5: Utility Functions")
    print("="*80)
    
    try:
        # Test smoothing
        test_data = np.random.randn(100)
        smoothed = smooth_series(test_data, window=5)
        assert len(smoothed) == len(test_data), "Smoothing changed length"
        print("✓ Smoothing function works")
        
        # Test correlation
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        corr, p_val = compute_correlation(x, y, method='pearson')
        assert abs(corr - 1.0) < 0.01, "Perfect correlation not detected"
        print(f"✓ Correlation function works (r={corr:.3f})")
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_plot_generation(experiments):
    """Test basic plot generation"""
    print("\n" + "="*80)
    print("TEST 6: Plot Generation")
    print("="*80)
    
    try:
        figures_dir = Path(__file__).parent / 'figures' / 'test'
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Test simple plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot test accuracy curves
        count = 0
        for exp_name, exp_data in list(experiments.items())[:3]:  # Just first 3
            history = exp_data.get('history', {})
            if 'test_acc' in history and history['test_acc']:
                epochs = np.array(history.get('epoch', range(len(history['test_acc']))))
                test_acc = np.array(history['test_acc'])
                ax.plot(epochs, test_acc, label=exp_name, alpha=0.7)
                count += 1
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Test Accuracy')
        ax.set_title('Test Plot')
        ax.legend()
        ax.grid(alpha=0.3)
        
        save_path = figures_dir / 'test_plot.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        assert save_path.exists(), "Plot file not created"
        print(f"✓ Generated test plot: {save_path}")
        print(f"  Plotted {count} experiments")
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "ANALYSIS INFRASTRUCTURE TEST" + " "*30 + "║")
    print("╚" + "="*78 + "╝")
    print()
    
    try:
        # Test 1: Load experiments
        nanda_exps, softmax_exps = test_load_experiments()
        
        # Test 2: Summary tables
        nanda_df = test_summary_table(nanda_exps)
        softmax_df = test_summary_table(softmax_exps)
        
        # Test 3: Grokking classification
        test_grokking_classification(nanda_exps)
        test_grokking_classification(softmax_exps)
        
        # Test 4: Statistical comparison
        test_statistical_comparison(nanda_exps)
        
        # Test 5: Utility functions
        test_utility_functions()
        
        # Test 6: Plot generation
        test_plot_generation(nanda_exps)
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print("✓ ALL TESTS PASSED!")
        print()
        print("Analysis infrastructure is ready for use.")
        print()
        print("Next steps:")
        print("  1. Open Jupyter: jupyter notebook")
        print("  2. Run analyze_nanda_experiments.ipynb")
        print("  3. Run analyze_softmax_experiments.ipynb")
        print("  4. Run cross_dataset_comparison.ipynb")
        print("="*80)
        
        return 0
    
    except Exception as e:
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"✗ TESTS FAILED")
        print(f"Error: {e}")
        print("="*80)
        return 1


if __name__ == '__main__':
    exit(main())

