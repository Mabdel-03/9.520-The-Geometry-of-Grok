"""
Basic functionality test for analysis infrastructure

Tests core data loading and analysis functions without plotting dependencies.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Import analysis utilities
sys.path.append(str(Path(__file__).parent))
from analysis_utils import (
    load_all_experiments, generate_summary_table, classify_grokking,
    compute_time_to_grok, filter_experiments, smooth_series, HAS_SCIPY
)

# Import statistical functions only if available
if HAS_SCIPY:
    from analysis_utils import statistical_comparison, compute_correlation

def test_load_experiments():
    """Test loading experiments"""
    print("="*80)
    print("TEST 1: Loading Experiments")
    print("="*80)
    
    nanda_dir = Path(__file__).parent.parent / 'results' / 'nanda'
    softmax_dir = Path(__file__).parent.parent / 'results' / 'softmax'
    
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


def test_summary_table(experiments, dataset_name):
    """Test generating summary tables"""
    print("\n" + "="*80)
    print(f"TEST 2: Generating Summary Table for {dataset_name}")
    print("="*80)
    
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
    grok_rate = df['grokked'].sum() / len(df) * 100
    print(f"  Grokking rate: {grok_rate:.1f}%")
    print(f"  Architectures: {df['architecture'].unique().tolist()}")
    print(f"  Optimizers: {df['optimizer'].unique().tolist()}")
    print(f"  Mean final test acc: {df['final_test_acc'].mean():.4f}")
    
    return df


def test_grokking_classification(experiments, dataset_name):
    """Test grokking classification"""
    print("\n" + "="*80)
    print(f"TEST 3: Grokking Classification for {dataset_name}")
    print("="*80)
    
    grok_count = 0
    nogrok_count = 0
    grok_details = []
    
    for exp_name, exp_data in experiments.items():
        grokked = classify_grokking(exp_data)
        if grokked:
            grok_count += 1
            grok_epoch = compute_time_to_grok(exp_data)
            assert grok_epoch > 0, f"Grokked but no grok epoch found for {exp_name}"
            grok_details.append((exp_name, grok_epoch))
        else:
            nogrok_count += 1
    
    print(f"✓ Classified {len(experiments)} experiments")
    print(f"  Grokked: {grok_count}")
    print(f"  No grokking: {nogrok_count}")
    
    if grok_details:
        print(f"\nGrokking experiments:")
        for name, epoch in sorted(grok_details, key=lambda x: x[1])[:5]:  # Show first 5
            print(f"  - {name}: epoch {epoch}")
        if len(grok_details) > 5:
            print(f"  ... and {len(grok_details) - 5} more")
    
    return grok_count, nogrok_count


def test_statistical_comparison(experiments, dataset_name):
    """Test statistical comparison functions"""
    print("\n" + "="*80)
    print(f"TEST 4: Statistical Comparisons for {dataset_name}")
    print("="*80)
    
    if not HAS_SCIPY:
        print("⚠ Scipy not available - skipping statistical tests")
        print("  Install scipy for statistical analysis: pip install scipy")
        return
    
    # Separate by optimizer
    adamw_exps = filter_experiments(experiments, optimizer='adamw')
    muon_exps = filter_experiments(experiments, optimizer='muon')
    sgd_exps = filter_experiments(experiments, optimizer='sgd')
    
    print(f"\nExperiments by optimizer:")
    print(f"  AdamW: {len(adamw_exps)}")
    print(f"  Muon: {len(muon_exps)}")
    print(f"  SGD: {len(sgd_exps)}")
    
    # Get final accuracies
    adamw_acc = [exp['history']['test_acc'][-1] for exp in adamw_exps.values() 
                 if 'test_acc' in exp['history'] and exp['history']['test_acc']]
    muon_acc = [exp['history']['test_acc'][-1] for exp in muon_exps.values() 
                if 'test_acc' in exp['history'] and exp['history']['test_acc']]
    
    if adamw_acc and muon_acc:
        stats = statistical_comparison(adamw_acc, muon_acc, ('AdamW', 'Muon'))
        print(f"\n✓ Statistical comparison successful")
        print(f"  AdamW: {stats['group1_mean']:.4f} ± {stats['group1_std']:.4f} (n={stats['group1_n']})")
        print(f"  Muon:  {stats['group2_mean']:.4f} ± {stats['group2_std']:.4f} (n={stats['group2_n']})")
        print(f"  P-value: {stats['p_value']:.4f}")
        print(f"  {stats['interpretation']}")
    else:
        print("⚠ Not enough data for comparison")


def test_utility_functions():
    """Test utility functions"""
    print("\n" + "="*80)
    print("TEST 5: Utility Functions")
    print("="*80)
    
    # Test smoothing
    test_data = np.random.randn(100)
    smoothed = smooth_series(test_data, window=5)
    assert len(smoothed) == len(test_data), "Smoothing changed length"
    print("✓ Smoothing function works")
    
    # Test correlation (if scipy available)
    if HAS_SCIPY:
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        corr, p_val = compute_correlation(x, y, method='pearson')
        assert abs(corr - 1.0) < 0.01, "Perfect correlation not detected"
        print(f"✓ Correlation function works (r={corr:.3f}, p={p_val:.4e})")
    else:
        print("⚠ Skipping correlation test (scipy not available)")
    
    # Test filtering
    test_exps = {
        'exp1': {'config': {'optimizer': 'adamw', 'architecture': 'mlp'}, 'history': {}},
        'exp2': {'config': {'optimizer': 'muon', 'architecture': 'mlp'}, 'history': {}},
        'exp3': {'config': {'optimizer': 'adamw', 'architecture': 'transformer'}, 'history': {}},
    }
    filtered = filter_experiments(test_exps, optimizer='adamw')
    assert len(filtered) == 2, "Filtering didn't work correctly"
    print(f"✓ Filtering function works")


def test_notebooks_exist():
    """Test that notebooks were created"""
    print("\n" + "="*80)
    print("TEST 6: Notebook Files")
    print("="*80)
    
    analysis_dir = Path(__file__).parent
    
    notebooks = [
        'analyze_nanda_experiments.ipynb',
        'analyze_softmax_experiments.ipynb',
        'cross_dataset_comparison.ipynb'
    ]
    
    for notebook in notebooks:
        nb_path = analysis_dir / notebook
        if nb_path.exists():
            print(f"✓ {notebook} exists")
            # Check file size
            size_kb = nb_path.stat().st_size / 1024
            print(f"  Size: {size_kb:.1f} KB")
        else:
            raise FileNotFoundError(f"Notebook not found: {notebook}")


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
        nanda_df = test_summary_table(nanda_exps, "Nanda")
        softmax_df = test_summary_table(softmax_exps, "Softmax")
        
        # Test 3: Grokking classification
        test_grokking_classification(nanda_exps, "Nanda")
        test_grokking_classification(softmax_exps, "Softmax")
        
        # Test 4: Statistical comparison
        test_statistical_comparison(nanda_exps, "Nanda")
        test_statistical_comparison(softmax_exps, "Softmax")
        
        # Test 5: Utility functions
        test_utility_functions()
        
        # Test 6: Notebooks exist
        test_notebooks_exist()
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print("✓ ALL TESTS PASSED!")
        print()
        print("Analysis infrastructure is ready for use.")
        print()
        print("Dataset Summary:")
        print(f"  Nanda:   {len(nanda_df)} experiments, {nanda_df['grokked'].sum()} grokked ({nanda_df['grokked'].sum()/len(nanda_df)*100:.1f}%)")
        print(f"  Softmax: {len(softmax_df)} experiments, {softmax_df['grokked'].sum()} grokked ({softmax_df['grokked'].sum()/len(softmax_df)*100:.1f}%)")
        print()
        print("Next steps:")
        print("  1. Open Jupyter: jupyter notebook")
        print("  2. Navigate to analysis/ directory")
        print("  3. Run the following notebooks:")
        print("     - analyze_nanda_experiments.ipynb")
        print("     - analyze_softmax_experiments.ipynb")
        print("     - cross_dataset_comparison.ipynb")
        print()
        print("Note: Notebooks require matplotlib, seaborn, and h5py packages.")
        print("      Install with: pip install matplotlib seaborn h5py nbformat")
        print("="*80)
        
        return 0
    
    except Exception as e:
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"✗ TESTS FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("="*80)
        return 1


if __name__ == '__main__':
    exit(main())

