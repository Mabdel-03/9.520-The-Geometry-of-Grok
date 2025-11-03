"""
Training Wrapper: Instrumentation for GOP tracking during training

This module provides a wrapper that instruments existing training loops
to compute and save GOP data at each epoch.
"""

import torch
import logging
from typing import Optional, Callable, Dict
from .gop_tracker import GOPTracker
from .gop_metrics import GOPMetrics
from .storage import HDF5Storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingWrapper:
    """
    Wraps training loops to add GOP tracking functionality.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        storage: HDF5Storage,
        gop_config: Dict,
        device: str = 'cuda'
    ):
        """
        Initialize training wrapper.
        
        Args:
            model: PyTorch model
            storage: HDF5Storage instance
            gop_config: Configuration dict for GOP tracking
            device: Device to use
        """
        self.model = model
        self.storage = storage
        self.device = device
        
        # Initialize GOP tracker
        self.gop_tracker = GOPTracker(
            model=model,
            compute_full=gop_config.get('compute_full', True),
            compute_per_layer=gop_config.get('compute_per_layer', True),
            use_gpu_for_gop=gop_config.get('use_gpu', True)
        )
        
        # Initialize metrics computer
        self.gop_metrics = GOPMetrics(
            top_k=gop_config.get('top_k_eigen', 100),
            rank_threshold=gop_config.get('rank_threshold', 1e-6)
        )
        
        self.tracking_frequency = gop_config.get('frequency', 1)
        self.compute_eigenvectors = gop_config.get('store_eigenvectors', True)
        
        logger.info(f"Initialized training wrapper with GOP tracking every {self.tracking_frequency} epoch(s)")
    
    def track_epoch(
        self,
        epoch: int,
        train_loss: float,
        test_loss: float,
        train_acc: float,
        test_acc: float
    ):
        """
        Compute and save GOP data for an epoch.
        
        Should be called after backward() but before optimizer.step().
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss
            test_loss: Test loss
            train_acc: Training accuracy
            test_acc: Test accuracy
        """
        if epoch % self.tracking_frequency != 0:
            return
        
        logger.info(f"Tracking GOP data for epoch {epoch}")
        
        try:
            # Compute GOPs
            gop_data = self.gop_tracker.compute_all()
            
            # Compute metrics for full GOP
            full_gop_metrics = None
            if gop_data['full_gop'] is not None:
                full_gop_metrics = self.gop_metrics.compute_all_metrics(
                    gop_data['full_gop'],
                    compute_eigenvectors=self.compute_eigenvectors
                )
                
                # Save full GOP
                self.storage.save_epoch_gop_full(
                    epoch=epoch,
                    gop_matrix=gop_data['full_gop'],
                    eigenvalues=full_gop_metrics.get('eigenvalues_full'),
                    eigenvectors=full_gop_metrics.get('eigenvectors_top_k')
                )
            
            # Compute metrics for layer GOPs
            layer_metrics = None
            if gop_data['layer_gops'] is not None:
                layer_metrics = self.gop_metrics.compute_layer_metrics(
                    gop_data['layer_gops'],
                    compute_eigenvectors=False  # Usually too large for per-layer
                )
                
                # Save layer GOPs
                self.storage.save_epoch_gop_layers(
                    epoch=epoch,
                    layer_gops=gop_data['layer_gops'],
                    layer_metrics=layer_metrics
                )
            
            # Save scalar metrics
            if full_gop_metrics is not None:
                self.storage.save_epoch_metrics(
                    epoch=epoch,
                    train_loss=train_loss,
                    test_loss=test_loss,
                    train_acc=train_acc,
                    test_acc=test_acc,
                    gop_metrics=full_gop_metrics
                )
            
            logger.info(f"Successfully saved GOP data for epoch {epoch}")
            
        except Exception as e:
            logger.error(f"Error tracking GOP for epoch {epoch}: {e}")
            raise
    
    def get_memory_estimate(self) -> Dict[str, float]:
        """Get estimated memory usage for GOP computation."""
        return self.gop_tracker.estimate_memory_usage()

