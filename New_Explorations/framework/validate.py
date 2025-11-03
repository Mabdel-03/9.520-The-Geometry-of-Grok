"""
Validation and Testing for GOP Framework

Provides unit tests and validation for GOP computation correctness.
"""

import torch
import torch.nn as nn
import numpy as np
import tempfile
import shutil
from pathlib import Path
import logging

from .gop_tracker import GOPTracker
from .gop_metrics import GOPMetrics
from .storage import HDF5Storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TinyModel(nn.Module):
    """Tiny model for testing (100 parameters)."""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(5, 10)
        self.layer2 = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.layer2(torch.relu(self.layer1(x)))


def test_gop_computation():
    """Test that GOP is computed correctly."""
    logger.info("Testing GOP computation...")
    
    # Create tiny model
    model = TinyModel()
    
    # Create dummy data
    x = torch.randn(3, 5)
    y = torch.randint(0, 2, (3,))
    
    # Forward and backward
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, y)
    loss.backward()
    
    # Extract gradients manually
    manual_grads = []
    for param in model.parameters():
        if param.grad is not None:
            manual_grads.append(param.grad.flatten())
    manual_grad_vector = torch.cat(manual_grads).cpu().numpy()
    
    # Compute GOP manually
    manual_gop = np.outer(manual_grad_vector, manual_grad_vector)
    
    # Compute GOP using tracker
    tracker = GOPTracker(model, compute_full=True, compute_per_layer=False)
    tracked_gop = tracker.compute_full_gop()
    
    # Compare
    difference = np.abs(manual_gop - tracked_gop).max()
    
    if difference < 1e-6:
        logger.info(f"✓ GOP computation test PASSED (max difference: {difference:.2e})")
        return True
    else:
        logger.error(f"✗ GOP computation test FAILED (max difference: {difference:.2e})")
        return False


def test_gop_metrics():
    """Test that metrics are computed correctly."""
    logger.info("Testing GOP metrics...")
    
    # Create a simple symmetric matrix
    A = np.random.randn(10, 10)
    gop = A @ A.T  # Symmetric positive semidefinite
    
    # Compute metrics
    metrics_computer = GOPMetrics(top_k=5)
    metrics = metrics_computer.compute_all_metrics(gop, compute_eigenvectors=True)
    
    # Validate trace
    trace_direct = np.trace(gop)
    trace_computed = metrics['trace']
    
    if np.abs(trace_direct - trace_computed) < 1e-6:
        logger.info(f"✓ Trace test PASSED")
    else:
        logger.error(f"✗ Trace test FAILED: {trace_direct} vs {trace_computed}")
        return False
    
    # Validate eigenvalues (sum should equal trace)
    eigenvalue_sum = np.sum(metrics['eigenvalues_full'])
    if np.abs(eigenvalue_sum - trace_direct) < 1e-3:
        logger.info(f"✓ Eigenvalue sum test PASSED")
    else:
        logger.error(f"✗ Eigenvalue sum test FAILED")
        return False
    
    # Validate Frobenius norm
    frob_direct = np.linalg.norm(gop, 'fro')
    frob_computed = metrics['frobenius_norm']
    
    if np.abs(frob_direct - frob_computed) < 1e-6:
        logger.info(f"✓ Frobenius norm test PASSED")
    else:
        logger.error(f"✗ Frobenius norm test FAILED")
        return False
    
    logger.info("✓ All GOP metrics tests PASSED")
    return True


def test_storage():
    """Test HDF5 storage functionality."""
    logger.info("Testing HDF5 storage...")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Initialize storage
        storage = HDF5Storage(
            output_dir=temp_dir,
            compression="gzip",
            compression_level=4
        )
        
        # Create dummy metrics
        gop_metrics = {
            'trace': 100.0,
            'frobenius_norm': 50.0,
            'spectral_norm': 25.0,
            'rank': 10,
            'condition_number': 5.0,
            'eigenvalues_top_k': np.array([10.0, 5.0, 2.0, 1.0, 0.5])
        }
        
        # Save metrics
        storage.save_epoch_metrics(
            epoch=0,
            train_loss=1.0,
            test_loss=2.0,
            train_acc=0.5,
            test_acc=0.3,
            gop_metrics=gop_metrics
        )
        
        # Save GOP
        dummy_gop = np.random.randn(10, 10)
        dummy_gop = dummy_gop @ dummy_gop.T  # Make symmetric
        
        storage.save_epoch_gop_full(
            epoch=0,
            gop_matrix=dummy_gop,
            eigenvalues=np.array([5.0, 3.0, 1.0])
        )
        
        # Load back
        metrics = storage.load_metrics()
        
        # Validate
        if 'train_loss' in metrics and metrics['train_loss'][0] == 1.0:
            logger.info("✓ Storage test PASSED")
            return True
        else:
            logger.error("✗ Storage test FAILED")
            return False
            
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_memory_estimation():
    """Test memory estimation."""
    logger.info("Testing memory estimation...")
    
    model = TinyModel()
    tracker = GOPTracker(model)
    
    mem_estimate = tracker.estimate_memory_usage()
    
    logger.info(f"Memory estimate for tiny model: {mem_estimate}")
    
    # Should be very small for tiny model
    if mem_estimate['full_gop_gb'] < 0.001:  # Less than 1 MB
        logger.info("✓ Memory estimation test PASSED")
        return True
    else:
        logger.error("✗ Memory estimation test FAILED")
        return False


def run_all_tests():
    """Run all validation tests."""
    logger.info("="*60)
    logger.info("Running GOP Framework Validation Tests")
    logger.info("="*60)
    
    tests = [
        ("GOP Computation", test_gop_computation),
        ("GOP Metrics", test_gop_metrics),
        ("HDF5 Storage", test_storage),
        ("Memory Estimation", test_memory_estimation)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Test: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Test Summary")
    logger.info(f"{'='*60}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 All tests passed! Framework is ready to use.")
        return True
    else:
        logger.error(f"\n❌ {total - passed} test(s) failed. Please fix before using.")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)

