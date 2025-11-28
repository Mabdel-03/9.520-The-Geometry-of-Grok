"""
Comprehensive Test for One-Hot AGOP Implementation

Tests all datasets, both architectures (MLP and Transformer), and AGOP computation.
This verifies the complete pipeline works before running full experiments.

Usage:
    python test_onehot_complete.py
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add paths
sys.path.insert(0, '/om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/agop_experiments/core')

from onehot_datasets import (
    create_onehot_modular_dataset,
    create_onehot_mnist_dataset,
    create_onehot_composition_dataset
)
from onehot_models import (
    ModularArithmeticMLP,
    OneHotReLUTransformer,
    OneHotStandardTransformer,
    CompositionMLP,
    MNISTModel
)
from agop_utils import InputGradientAGOPTracker


def test_dataset_and_model_and_agop(dataset_name, create_dataset_fn, model, dataset_args):
    """Test complete pipeline for one dataset"""
    print(f"\n{'='*80}")
    print(f"Testing: {dataset_name}")
    print(f"{'='*80}")
    
    # Create dataset
    print(f"1. Creating one-hot dataset...")
    train_X, train_y, test_X, test_y = create_dataset_fn(**dataset_args)
    print(f"   Train: {train_X.shape}, Test: {test_X.shape}")
    print(f"   Input dtype: {train_X.dtype}")
    print(f"   Input is float: {train_X.dtype in [torch.float32, torch.float64]}")
    
    # Test model forward
    print(f"2. Testing model forward pass...")
    model.eval()
    with torch.no_grad():
        logits = model(train_X[:10])
    print(f"   Input shape: {train_X[:10].shape}")
    print(f"   Output shape: {logits.shape}")
    
    # Test AGOP computation
    print(f"3. Testing AGOP computation...")
    tracker = InputGradientAGOPTracker(top_k=4, subsample_size=50, device='cpu', agop_device='cpu')
    criterion = nn.CrossEntropyLoss()
    
    agop = tracker.compute_input_agop(model, train_X[:100], train_y[:100], criterion)
    
    if agop is not None:
        print(f"   ✓ AGOP computed successfully!")
        print(f"   AGOP shape: {agop.shape}")
        print(f"   AGOP trace: {torch.trace(agop).item():.6f}")
        
        # Test metrics
        history = {}
        metrics = tracker.compute_agop_metrics(history, agop)
        print(f"   Metrics computed: {len(metrics)} metrics")
        print(f"     Eigengap: {metrics.get('agop_eigengap', 0):.6f}")
        print(f"     VCR: {metrics.get('agop_variation_collapse_ratio', 0):.6f}")
        return True
    else:
        print(f"   ✗ AGOP computation failed!")
        return False


def main():
    print("="*80)
    print("COMPREHENSIVE ONE-HOT AGOP PIPELINE TEST")
    print("="*80)
    
    device = 'cpu'
    results = {}
    
    # Test 1: Nanda MLP
    results['nanda_mlp'] = test_dataset_and_model_and_agop(
        dataset_name="Nanda Modular Addition (MLP)",
        create_dataset_fn=create_onehot_modular_dataset,
        model=ModularArithmeticMLP(p=97, hidden_dim=128),
        dataset_args={'p': 97, 'operation': 'add', 'train_fraction': 0.3, 'device': device}
    )
    
    # Test 2: Nanda ReLU Transformer
    results['nanda_transformer'] = test_dataset_and_model_and_agop(
        dataset_name="Nanda Modular Addition (ReLU Transformer)",
        create_dataset_fn=create_onehot_modular_dataset,
        model=OneHotReLUTransformer(p=97, d_model=128, n_heads=4, d_mlp=512),
        dataset_args={'p': 97, 'operation': 'add', 'train_fraction': 0.3, 'device': device}
    )
    
    # Test 3: Softmax MLP
    results['softmax_mlp'] = test_dataset_and_model_and_agop(
        dataset_name="Softmax Modular Addition (MLP)",
        create_dataset_fn=create_onehot_modular_dataset,
        model=ModularArithmeticMLP(p=97, hidden_dim=128),
        dataset_args={'p': 97, 'operation': 'add', 'train_fraction': 0.5, 'device': device}
    )
    
    # Test 4: Softmax Standard Transformer
    results['softmax_transformer'] = test_dataset_and_model_and_agop(
        dataset_name="Softmax Modular Addition (Standard Transformer)",
        create_dataset_fn=create_onehot_modular_dataset,
        model=OneHotStandardTransformer(p=97, d_model=128, n_heads=4, n_layers=2),
        dataset_args={'p': 97, 'operation': 'add', 'train_fraction': 0.5, 'device': device}
    )
    
    # Test 5: MNIST
    results['mnist'] = test_dataset_and_model_and_agop(
        dataset_name="MNIST (MLP)",
        create_dataset_fn=create_onehot_mnist_dataset,
        model=MNISTModel(input_dim=784, hidden_dim=200, output_dim=10, depth=3),
        dataset_args={'train_points': 200, 'device': device}
    )
    
    # Test 6: Composition
    results['composition'] = test_dataset_and_model_and_agop(
        dataset_name="Composition (MLP)",
        create_dataset_fn=create_onehot_composition_dataset,
        model=CompositionMLP(vocab_size=50, seq_len=10, hidden_dim=256, n_layers=2),
        dataset_args={'vocab_size': 50, 'seq_len': 10, 'n_facts': 200, 'train_fraction': 0.3, 'device': device}
    )
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:8s} {name}")
    
    total = len(results)
    passed = sum(results.values())
    
    print("\n" + "="*80)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("="*80)
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED!")
        print("  One-hot encoding + AGOP pipeline working for all datasets and architectures!")
        print("  Ready to run full experiments.")
        return True
    else:
        print(f"\n✗ {total - passed} tests failed")
        print("  Check errors above and fix issues")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

