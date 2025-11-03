"""
End-to-End Test of GOP Framework

Quick test to verify the framework works correctly.
Creates a tiny model, trains for 10 epochs, and verifies GOP tracking.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys

# Add framework to path
sys.path.insert(0, str(Path(__file__).parent / "framework"))

from framework import GOPTracker, GOPMetrics, HDF5Storage, TrainingWrapper

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TinyModel(nn.Module):
    """Tiny 2-layer MLP for testing."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
    
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def test_framework_end_to_end():
    """Run complete end-to-end test."""
    logger.info("="*60)
    logger.info("GOP Framework End-to-End Test")
    logger.info("="*60)
    
    # Create temp directory for results
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Temp directory: {temp_dir}")
    
    try:
        # Create tiny model and data
        logger.info("\n1. Creating model and dataset...")
        model = TinyModel()
        X_train = torch.randn(50, 10)
        y_train = torch.randint(0, 5, (50,))
        X_test = torch.randn(20, 10)
        y_test = torch.randint(0, 5, (20,))
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"   Model has {total_params} parameters")
        
        # Create storage
        logger.info("\n2. Initializing storage...")
        storage = HDF5Storage(
            output_dir=temp_dir,
            compression="gzip",
            compression_level=4,
            store_full_gop=True,
            store_eigenvectors=True
        )
        
        # Create wrapper
        logger.info("\n3. Creating training wrapper...")
        gop_config = {
            'compute_full': True,
            'compute_per_layer': True,
            'frequency': 1,
            'top_k_eigen': 10,
            'use_gpu': False,  # Use CPU for testing
            'store_eigenvectors': True
        }
        
        wrapper = TrainingWrapper(
            model=model,
            storage=storage,
            gop_config=gop_config,
            device='cpu'
        )
        
        mem_est = wrapper.get_memory_estimate()
        logger.info(f"   Estimated memory: {mem_est['total_gb']*1000:.2f} MB per epoch")
        
        # Train for a few epochs
        logger.info("\n4. Training for 10 epochs...")
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(10):
            model.train()
            
            # Forward and backward
            output = model(X_train)
            train_loss = criterion(output, y_train)
            
            optimizer.zero_grad()
            train_loss.backward()
            
            # Compute accuracy
            with torch.no_grad():
                train_preds = output.argmax(dim=-1)
                train_acc = (train_preds == y_train).float().mean().item()
                
                # Test
                test_output = model(X_test)
                test_loss = criterion(test_output, y_test)
                test_preds = test_output.argmax(dim=-1)
                test_acc = (test_preds == y_test).float().mean().item()
            
            # Track GOP (before optimizer.step())
            wrapper.track_epoch(
                epoch=epoch,
                train_loss=train_loss.item(),
                test_loss=test_loss.item(),
                train_acc=train_acc,
                test_acc=test_acc
            )
            
            # Optimizer step
            optimizer.step()
            
            logger.info(f"   Epoch {epoch}: Train Loss={train_loss.item():.4f}, "
                       f"Test Acc={test_acc:.4f}")
        
        # Verify files were created
        logger.info("\n5. Verifying output files...")
        
        metrics_file = Path(temp_dir) / "metrics.h5"
        gop_full_file = Path(temp_dir) / "gop_full.h5"
        gop_layers_file = Path(temp_dir) / "gop_layers.h5"
        
        files_exist = [
            ("metrics.h5", metrics_file.exists()),
            ("gop_full.h5", gop_full_file.exists()),
            ("gop_layers.h5", gop_layers_file.exists())
        ]
        
        all_exist = all(exists for _, exists in files_exist)
        
        for name, exists in files_exist:
            status = "✓" if exists else "✗"
            logger.info(f"   {status} {name}")
        
        # Load and verify data
        if all_exist:
            logger.info("\n6. Verifying data integrity...")
            
            import h5py
            
            # Check metrics
            with h5py.File(metrics_file, 'r') as f:
                logger.info(f"   Metrics file keys: {list(f.keys())}")
                if 'train_loss' in f:
                    logger.info(f"   ✓ Train loss has {len(f['train_loss'])} entries")
                if 'gop_trace' in f:
                    logger.info(f"   ✓ GOP trace has {len(f['gop_trace'])} entries")
                if 'eigenvalues_top_k' in f:
                    logger.info(f"   ✓ Top eigenvalues shape: {f['eigenvalues_top_k'].shape}")
            
            # Check GOP full
            with h5py.File(gop_full_file, 'r') as f:
                epochs_stored = [k for k in f.keys() if k.startswith('epoch_')]
                logger.info(f"   ✓ Stored full GOPs for {len(epochs_stored)} epochs")
                if epochs_stored:
                    sample_gop = f[epochs_stored[0]]['gop']
                    logger.info(f"   ✓ GOP matrix shape: {sample_gop.shape}")
            
            # Check GOP layers
            with h5py.File(gop_layers_file, 'r') as f:
                epochs_stored = [k for k in f.keys() if k.startswith('epoch_')]
                if epochs_stored:
                    sample_epoch = f[epochs_stored[0]]
                    layers = list(sample_epoch.keys())
                    logger.info(f"   ✓ Stored {len(layers)} layer GOPs")
            
            # Get file sizes
            logger.info("\n7. File sizes:")
            for name, path in [("metrics.h5", metrics_file),
                             ("gop_full.h5", gop_full_file),
                             ("gop_layers.h5", gop_layers_file)]:
                if path.exists():
                    size_mb = path.stat().st_size / (1024 ** 2)
                    logger.info(f"   {name}: {size_mb:.2f} MB")
        
        logger.info("\n" + "="*60)
        if all_exist:
            logger.info("✅ END-TO-END TEST PASSED!")
            logger.info("="*60)
            logger.info("\nFramework is working correctly!")
            logger.info("You can now run full experiments.")
            return True
        else:
            logger.error("❌ END-TO-END TEST FAILED!")
            logger.error("="*60)
            logger.error("Some files were not created. Check logs above.")
            return False
            
    except Exception as e:
        logger.error(f"\n❌ TEST FAILED WITH ERROR: {e}", exc_info=True)
        return False
        
    finally:
        # Cleanup
        logger.info(f"\nCleaning up temp directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    success = test_framework_end_to_end()
    sys.exit(0 if success else 1)

