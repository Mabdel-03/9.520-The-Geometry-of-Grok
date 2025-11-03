"""
HDF5 Storage: Efficient compressed storage for GOP data

This module handles storage of:
- Scalar time series (loss, accuracy, GOP metrics)
- Full GOP matrices per epoch
- Per-layer GOP matrices
- Eigenvalues and eigenvectors

Uses HDF5 with compression for efficient storage of large matrices.
"""

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Union
import logging
import time
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HDF5Storage:
    """
    Manages HDF5 storage for GOP analysis data.
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        compression: str = "gzip",
        compression_level: int = 6,
        store_full_gop: bool = True,
        store_eigenvectors: bool = True,
        float_precision: str = "float32"
    ):
        """
        Initialize HDF5 storage manager.
        
        Args:
            output_dir: Directory to store HDF5 files
            compression: Compression algorithm ("gzip", "lzf", or None)
            compression_level: Compression level (0-9 for gzip)
            store_full_gop: Whether to store full GOP matrices
            store_eigenvectors: Whether to store eigenvectors
            float_precision: Float precision ("float32" or "float64")
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.compression = compression
        self.compression_level = compression_level if compression == "gzip" else None
        self.store_full_gop = store_full_gop
        self.store_eigenvectors = store_eigenvectors
        self.float_dtype = np.float32 if float_precision == "float32" else np.float64
        
        # File paths
        self.metrics_file = self.output_dir / "metrics.h5"
        self.gop_full_file = self.output_dir / "gop_full.h5"
        self.gop_layers_file = self.output_dir / "gop_layers.h5"
        
        # Initialize files
        self._init_files()
        
        logger.info(f"Initialized HDF5 storage at {self.output_dir}")
        logger.info(f"Compression: {compression} (level {compression_level})")
    
    def _init_files(self):
        """Initialize HDF5 files with metadata."""
        # Create metrics file
        with h5py.File(self.metrics_file, 'a') as f:
            f.attrs['created'] = time.strftime('%Y-%m-%d %H:%M:%S')
            f.attrs['compression'] = self.compression or 'none'
            if self.compression_level:
                f.attrs['compression_level'] = self.compression_level
        
        # Create GOP files if needed
        if self.store_full_gop:
            with h5py.File(self.gop_full_file, 'a') as f:
                f.attrs['created'] = time.strftime('%Y-%m-%d %H:%M:%S')
                f.attrs['compression'] = self.compression or 'none'
        
        with h5py.File(self.gop_layers_file, 'a') as f:
            f.attrs['created'] = time.strftime('%Y-%m-%d %H:%M:%S')
            f.attrs['compression'] = self.compression or 'none'
    
    def save_epoch_metrics(
        self,
        epoch: int,
        train_loss: float,
        test_loss: float,
        train_acc: float,
        test_acc: float,
        gop_metrics: Dict[str, any]
    ):
        """
        Save scalar metrics for an epoch.
        
        Args:
            epoch: Epoch number
            train_loss: Training loss
            test_loss: Test loss
            train_acc: Training accuracy
            test_acc: Test accuracy
            gop_metrics: Dictionary of GOP metrics
        """
        with h5py.File(self.metrics_file, 'a') as f:
            # Create or extend datasets
            for name, value in [
                ('train_loss', train_loss),
                ('test_loss', test_loss),
                ('train_acc', train_acc),
                ('test_acc', test_acc)
            ]:
                if name not in f:
                    f.create_dataset(
                        name,
                        shape=(0,),
                        maxshape=(None,),
                        dtype=np.float32,
                        compression=self.compression,
                        compression_opts=self.compression_level
                    )
                
                # Append value
                dset = f[name]
                dset.resize((epoch + 1,))
                dset[epoch] = value
            
            # Save GOP scalar metrics
            scalar_metrics = [
                'trace', 'frobenius_norm', 'spectral_norm', 'nuclear_norm',
                'rank', 'condition_number', 'determinant',
                'eigenvalue_max', 'eigenvalue_min', 'eigenvalue_mean', 'eigenvalue_std',
                'top_k_cumulative_variance'
            ]
            
            for metric_name in scalar_metrics:
                if metric_name in gop_metrics:
                    dset_name = f'gop_{metric_name}'
                    if dset_name not in f:
                        f.create_dataset(
                            dset_name,
                            shape=(0,),
                            maxshape=(None,),
                            dtype=np.float32,
                            compression=self.compression,
                            compression_opts=self.compression_level
                        )
                    
                    dset = f[dset_name]
                    dset.resize((epoch + 1,))
                    dset[epoch] = gop_metrics[metric_name]
            
            # Save top-k eigenvalues as 2D array
            if 'eigenvalues_top_k' in gop_metrics:
                eigs = gop_metrics['eigenvalues_top_k']
                if eigs is not None and len(eigs) > 0:
                    dset_name = 'eigenvalues_top_k'
                    if dset_name not in f:
                        k = len(eigs)
                        f.create_dataset(
                            dset_name,
                            shape=(0, k),
                            maxshape=(None, k),
                            dtype=np.float32,
                            compression=self.compression,
                            compression_opts=self.compression_level
                        )
                    
                    dset = f[dset_name]
                    dset.resize((epoch + 1, dset.shape[1]))
                    dset[epoch, :] = eigs[:dset.shape[1]]  # Truncate if needed
    
    def save_epoch_gop_full(
        self,
        epoch: int,
        gop_matrix: np.ndarray,
        eigenvalues: Optional[np.ndarray] = None,
        eigenvectors: Optional[np.ndarray] = None
    ):
        """
        Save full GOP matrix for an epoch.
        
        Args:
            epoch: Epoch number
            gop_matrix: Full GOP matrix
            eigenvalues: Full eigenvalue spectrum
            eigenvectors: Top-k eigenvectors
        """
        if not self.store_full_gop:
            return
        
        with h5py.File(self.gop_full_file, 'a') as f:
            epoch_group = f.create_group(f'epoch_{epoch}')
            
            # Store GOP matrix
            epoch_group.create_dataset(
                'gop',
                data=gop_matrix.astype(self.float_dtype),
                compression=self.compression,
                compression_opts=self.compression_level
            )
            
            # Store full eigenvalues
            if eigenvalues is not None:
                epoch_group.create_dataset(
                    'eigenvalues',
                    data=eigenvalues.astype(np.float64),  # Higher precision for eigenvalues
                    compression=self.compression,
                    compression_opts=self.compression_level
                )
            
            # Store top-k eigenvectors
            if eigenvectors is not None and self.store_eigenvectors:
                epoch_group.create_dataset(
                    'eigenvectors_top_k',
                    data=eigenvectors.astype(self.float_dtype),
                    compression=self.compression,
                    compression_opts=self.compression_level
                )
            
            epoch_group.attrs['epoch'] = epoch
            epoch_group.attrs['shape'] = gop_matrix.shape
            epoch_group.attrs['dtype'] = str(self.float_dtype)
    
    def save_epoch_gop_layers(
        self,
        epoch: int,
        layer_gops: Dict[str, np.ndarray],
        layer_metrics: Dict[str, Dict[str, any]]
    ):
        """
        Save per-layer GOP matrices and metrics for an epoch.
        
        Args:
            epoch: Epoch number
            layer_gops: Dictionary mapping layer names to GOP matrices
            layer_metrics: Dictionary mapping layer names to their metrics
        """
        with h5py.File(self.gop_layers_file, 'a') as f:
            epoch_group = f.require_group(f'epoch_{epoch}')
            
            for layer_name, gop_matrix in layer_gops.items():
                layer_group = epoch_group.create_group(layer_name)
                
                # Store GOP matrix
                layer_group.create_dataset(
                    'gop',
                    data=gop_matrix.astype(self.float_dtype),
                    compression=self.compression,
                    compression_opts=self.compression_level
                )
                
                # Store metrics as attributes (small scalars)
                if layer_name in layer_metrics:
                    metrics = layer_metrics[layer_name]
                    for metric_name, value in metrics.items():
                        if metric_name not in ['eigenvalues_full', 'eigenvalues_top_k', 'eigenvectors_top_k']:
                            layer_group.attrs[metric_name] = value
                    
                    # Store eigenvalues as datasets
                    if 'eigenvalues_top_k' in metrics and metrics['eigenvalues_top_k'] is not None:
                        layer_group.create_dataset(
                            'eigenvalues_top_k',
                            data=metrics['eigenvalues_top_k'].astype(np.float32),
                            compression=self.compression,
                            compression_opts=self.compression_level
                        )
    
    def save_config(self, config: Dict):
        """
        Save experiment configuration.
        
        Args:
            config: Configuration dictionary
        """
        config_file = self.output_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved configuration to {config_file}")
    
    def load_metrics(self) -> Dict[str, np.ndarray]:
        """
        Load all scalar metrics.
        
        Returns:
            Dictionary mapping metric names to arrays
        """
        metrics = {}
        with h5py.File(self.metrics_file, 'r') as f:
            for key in f.keys():
                metrics[key] = f[key][:]
        return metrics
    
    def load_epoch_gop_full(self, epoch: int) -> Dict[str, np.ndarray]:
        """
        Load full GOP data for a specific epoch.
        
        Args:
            epoch: Epoch number
            
        Returns:
            Dictionary containing GOP matrix, eigenvalues, eigenvectors
        """
        data = {}
        with h5py.File(self.gop_full_file, 'r') as f:
            epoch_group = f[f'epoch_{epoch}']
            data['gop'] = epoch_group['gop'][:]
            if 'eigenvalues' in epoch_group:
                data['eigenvalues'] = epoch_group['eigenvalues'][:]
            if 'eigenvectors_top_k' in epoch_group:
                data['eigenvectors_top_k'] = epoch_group['eigenvectors_top_k'][:]
        return data
    
    def load_epoch_gop_layers(self, epoch: int) -> Dict[str, Dict[str, any]]:
        """
        Load per-layer GOP data for a specific epoch.
        
        Args:
            epoch: Epoch number
            
        Returns:
            Dictionary mapping layer names to their GOP data and metrics
        """
        layers = {}
        with h5py.File(self.gop_layers_file, 'r') as f:
            epoch_group = f[f'epoch_{epoch}']
            for layer_name in epoch_group.keys():
                layer_group = epoch_group[layer_name]
                layers[layer_name] = {
                    'gop': layer_group['gop'][:],
                    'metrics': dict(layer_group.attrs)
                }
                if 'eigenvalues_top_k' in layer_group:
                    layers[layer_name]['eigenvalues_top_k'] = layer_group['eigenvalues_top_k'][:]
        return layers
    
    def get_stored_epochs(self) -> List[int]:
        """
        Get list of epochs that have been stored.
        
        Returns:
            List of epoch numbers
        """
        epochs = []
        if self.gop_full_file.exists():
            with h5py.File(self.gop_full_file, 'r') as f:
                epochs = [int(key.split('_')[1]) for key in f.keys() if key.startswith('epoch_')]
        return sorted(epochs)
    
    def get_file_sizes(self) -> Dict[str, float]:
        """
        Get sizes of storage files in GB.
        
        Returns:
            Dictionary mapping file names to sizes in GB
        """
        sizes = {}
        for name, path in [
            ('metrics', self.metrics_file),
            ('gop_full', self.gop_full_file),
            ('gop_layers', self.gop_layers_file)
        ]:
            if path.exists():
                sizes[name] = path.stat().st_size / (1024 ** 3)  # Convert to GB
        sizes['total'] = sum(sizes.values())
        return sizes
    
    def close(self):
        """Close all file handles (if any are open)."""
        # HDF5 files are opened and closed within each method,
        # so no persistent handles to close
        logger.info("HDF5 storage closed")

