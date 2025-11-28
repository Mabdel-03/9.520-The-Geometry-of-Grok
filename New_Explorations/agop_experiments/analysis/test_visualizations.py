"""
Test script to demonstrate enhanced visualization functions with synthetic data.

This creates fake experiment data and generates all visualization plots.
Useful for testing the visualization pipeline before real experiments complete.

Usage:
    python test_visualizations.py
"""

import numpy as np
import json
import h5py
from pathlib import Path
import tempfile
import shutil

def create_synthetic_experiment(output_dir: Path, grokking: bool = True):
    """
    Create synthetic experiment data that mimics grokking behavior.
    
    Args:
        output_dir: Directory to save synthetic data
        grokking: If True, creates grokking pattern; if False, no grokking
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_epochs = 10000
    agop_freq = 100
    
    # Create training history
    epochs = np.arange(0, n_epochs, 100)
    n_points = len(epochs)
    
    if grokking:
        # Grokking pattern: quick train fit, delayed test generalization
        train_acc = 1.0 - 0.1 * np.exp(-epochs / 1000)
        
        # Test accuracy: stays low then suddenly increases around epoch 5000
        test_acc = np.zeros_like(epochs, dtype=float)
        grok_point = 5000
        for i, e in enumerate(epochs):
            if e < grok_point:
                test_acc[i] = 0.1 + 0.05 * np.random.randn()
            else:
                test_acc[i] = 0.98 - 0.1 * np.exp(-(e - grok_point) / 1000)
        test_acc = np.clip(test_acc, 0, 1)
    else:
        # No grokking: both train and test plateau low
        train_acc = 0.3 + 0.1 * (1.0 - np.exp(-epochs / 1000))
        test_acc = 0.2 + 0.05 * (1.0 - np.exp(-epochs / 1000))
    
    train_loss = -np.log(train_acc + 0.01)
    test_loss = -np.log(test_acc + 0.01)
    
    history = {
        'epoch': epochs.tolist(),
        'train_acc': train_acc.tolist(),
        'train_loss': train_loss.tolist(),
        'test_acc': test_acc.tolist(),
        'test_loss': test_loss.tolist(),
    }
    
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Create AGOP metrics
    agop_epochs = epochs[::1]  # Every epoch for AGOP
    n_agop = len(agop_epochs)
    
    if grokking:
        # AGOP patterns during grokking
        # Variation collapse ratio increases at grokking
        vcr = 0.2 + 0.6 * (1.0 / (1.0 + np.exp(-(agop_epochs - 5000) / 500)))
        
        # Eigengap increases
        eigengap = 0.1 + 0.5 * (1.0 / (1.0 + np.exp(-(agop_epochs - 5000) / 500)))
        
        # Trace decreases
        trace = 10.0 * np.exp(-agop_epochs / 3000) + 1.0
        
        # Spectral radius (top eigenvalue)
        spectral = vcr * trace
    else:
        # No clear pattern
        vcr = 0.3 + 0.1 * np.random.randn(n_agop) * 0.05
        eigengap = 0.2 + 0.1 * np.random.randn(n_agop) * 0.05
        trace = 5.0 + np.random.randn(n_agop) * 0.5
        spectral = vcr * trace
    
    # Other metrics
    frobenius = np.sqrt(trace**2 + spectral**2)
    subspace_sim = 0.9 + 0.1 * np.random.randn(n_agop) * 0.05
    subspace_sim[0] = np.nan  # First epoch has no previous
    subspace_sim = np.clip(subspace_sim, 0, 1)
    
    # Top eigenvalues
    eigenvalue_1 = spectral
    eigenvalue_2 = spectral - eigengap
    eigenvalue_3 = eigenvalue_2 * 0.8
    
    top_eigenvalue_energy = eigenvalue_1 / trace
    top5_energy = (eigenvalue_1 + eigenvalue_2 + eigenvalue_3) / trace
    top10_energy = top5_energy * 1.1
    
    # Save AGOP metrics to HDF5
    with h5py.File(output_dir / 'agop_metrics.h5', 'w') as f:
        f.create_dataset('epoch', data=agop_epochs, compression='gzip')
        f.create_dataset('agop_frobenius', data=frobenius, compression='gzip')
        f.create_dataset('agop_spectral_radius', data=spectral, compression='gzip')
        f.create_dataset('agop_trace', data=trace, compression='gzip')
        f.create_dataset('agop_eigengap', data=eigengap, compression='gzip')
        f.create_dataset('agop_variation_collapse_ratio', data=vcr, compression='gzip')
        f.create_dataset('agop_topk_subspace_similarity', data=subspace_sim, compression='gzip')
        f.create_dataset('agop_top_eigenvalue_energy', data=top_eigenvalue_energy, compression='gzip')
        f.create_dataset('agop_top5_energy_ratio', data=top5_energy, compression='gzip')
        f.create_dataset('agop_top10_energy_ratio', data=top10_energy, compression='gzip')
        f.create_dataset('agop_eigenvalue_1', data=eigenvalue_1, compression='gzip')
        f.create_dataset('agop_eigenvalue_2', data=eigenvalue_2, compression='gzip')
        f.create_dataset('agop_eigenvalue_3', data=eigenvalue_3, compression='gzip')
    
    # Create config
    config = {
        'optimizer': 'adamw' if grokking else 'sgd',
        'lr': 0.001,
        'weight_decay': 1.0 if grokking else 0.0,
        'n_epochs': n_epochs,
        'agop_freq': agop_freq,
        'synthetic': True,
        'grokking': grokking,
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created synthetic {'grokking' if grokking else 'non-grokking'} experiment in {output_dir}")


def main():
    print("="*80)
    print("Testing Enhanced AGOP Visualizations with Synthetic Data")
    print("="*80)
    
    # Create temporary directory for test data
    temp_dir = Path(tempfile.mkdtemp(prefix='agop_viz_test_'))
    print(f"\nCreating synthetic data in: {temp_dir}")
    
    try:
        # Create synthetic grokking experiment
        grok_dir = temp_dir / 'synthetic_grokking'
        create_synthetic_experiment(grok_dir, grokking=True)
        
        # Create synthetic non-grokking experiment
        nogrok_dir = temp_dir / 'synthetic_no_grokking'
        create_synthetic_experiment(nogrok_dir, grokking=False)
        
        # Test visualization script
        print("\n" + "="*80)
        print("Testing visualizations...")
        print("="*80)
        
        import subprocess
        import sys
        
        # Test single experiment visualization
        print("\n1. Testing single experiment (grokking)...")
        result = subprocess.run([
            sys.executable, 'visualize_agop_metrics.py',
            '--results_dir', str(grok_dir)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Grokking visualization successful!")
            print(f"  Check plots in: {grok_dir / 'plots'}")
        else:
            print("✗ Visualization failed:")
            print(result.stderr)
        
        # Test comparison visualization
        print("\n2. Testing comparison (grokking vs non-grokking)...")
        result = subprocess.run([
            sys.executable, 'visualize_agop_metrics.py',
            '--results_dir', str(temp_dir),
            '--compare_optimizers'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Comparison visualization successful!")
            print(f"  Check plots in: {temp_dir / 'plots'}")
        else:
            print("✗ Visualization failed:")
            print(result.stderr)
        
        print("\n" + "="*80)
        print("Test complete!")
        print("="*80)
        print(f"\nGenerated plots location: {temp_dir}")
        print("Inspect the plots to verify the enhanced visualizations.")
        print("\nTo keep the test data, copy it before exiting:")
        print(f"  cp -r {temp_dir} ./test_output")
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    # Uncomment to auto-cleanup:
    # shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()

