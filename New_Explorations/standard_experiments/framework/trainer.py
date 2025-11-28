"""
Base Training Framework with AGOP-based Spectral Metrics Tracking
Supports multiple optimizers (Muon, Adam, SGD) and comprehensive metric logging

AGOP = Average Gradient Outer Product (Beaglehole et al.)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import h5py
from pathlib import Path
from typing import Dict, Optional, Callable, List, Any
import time
from tqdm import tqdm
import warnings

try:
    from .spectral_metrics import SpectralMetricsComputer
    from .muon_official import Muon, MuonW  # Official Muon from modded-nanogpt
except ImportError:
    # Handle absolute imports when not run as package
    import sys
    from pathlib import Path
    framework_path = Path(__file__).parent
    if str(framework_path) not in sys.path:
        sys.path.insert(0, str(framework_path))
    from spectral_metrics import SpectralMetricsComputer
    from muon_official import Muon, MuonW  # Official Muon from modded-nanogpt


class GrokkingTrainer:
    """
    Unified trainer for grokking experiments with AGOP-based spectral analysis.
    
    Features:
    - Multiple optimizer support (Muon, MuonW, Adam, AdamW, SGD)
    - Comprehensive AGOP spectral metrics tracking (Beaglehole et al.)
    - Memory-efficient per-sample gradient computation
    - Flexible data loading
    - Checkpointing and logging
    - HDF5 storage for efficient metric storage
    
    AGOP Metrics Tracked:
    - Eigengap (λ₁ - λ₂): Gradient alignment measure
    - Energy in top eigenvector (λ₁/Σλᵢ): Neural collapse indicator
    - Trace (Σλᵢ = E[||∇L||²]): Average squared gradient norm
    - Spectral radius to trace ratio: Concentration measure
    - Effective rank: Dimensionality of gradient space
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        test_data: torch.Tensor,
        test_labels: torch.Tensor,
        optimizer_name: str = 'adamw',
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: Optional[int] = None,  # None = full batch
        n_epochs: int = 40000,
        device: str = 'cuda',
        # Loss function
        use_mse_loss: bool = False,  # For MNIST Omnigrok experiments
        # AGOP spectral metrics settings
        compute_spectral_metrics: bool = True,
        spectral_metrics_freq: int = 100,  # How often to compute AGOP (expensive!)
        spectral_top_k: int = 20,
        compute_full_spectrum: bool = False,
        compute_per_layer: bool = False,
        agop_subsample_size: Optional[int] = None,  # Subsample data for AGOP (memory)
        # Logging settings
        log_freq: int = 100,
        save_dir: str = './results',
        experiment_name: str = 'experiment',
        # Checkpointing
        checkpoint_freq: int = 1000,
        save_checkpoints: bool = True,
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            train_data, train_labels: Training dataset
            test_data, test_labels: Test dataset
            optimizer_name: One of ['muon', 'muonw', 'adam', 'adamw', 'sgd']
            lr: Learning rate
            weight_decay: Weight decay coefficient
            batch_size: Batch size (None for full batch)
            n_epochs: Number of training epochs
            device: Device to train on
            compute_spectral_metrics: Whether to compute AGOP spectral metrics
            spectral_metrics_freq: How often to compute AGOP (every N epochs, expensive!)
            spectral_top_k: Number of top eigenvalues to track
            compute_full_spectrum: Whether to compute full eigenvalue spectrum
            compute_per_layer: Whether to compute per-layer AGOP metrics
            agop_subsample_size: If set, subsample this many examples for AGOP computation
            log_freq: How often to print training progress
            save_dir: Directory to save results
            experiment_name: Name of experiment (for saving)
            checkpoint_freq: How often to save checkpoints
            save_checkpoints: Whether to save model checkpoints
        """
        self.model = model.to(device)
        self.device = device
        
        # Move data to device
        self.train_data = train_data.to(device)
        self.train_labels = train_labels.to(device)
        self.test_data = test_data.to(device)
        self.test_labels = test_labels.to(device)
        
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.log_freq = log_freq
        self.checkpoint_freq = checkpoint_freq
        self.save_checkpoints = save_checkpoints
        
        # Create optimizer
        self.optimizer_name = optimizer_name.lower()
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = self._create_optimizer()
        
        # Loss function
        self.use_mse_loss = use_mse_loss
        if use_mse_loss:
            self.criterion = nn.MSELoss()
            # Store one-hot lookup - will be indexed and moved to device as needed
            self.one_hots = torch.eye(10, dtype=torch.float64)
            # Move to same device as data for efficiency
            if torch.cuda.is_available() and 'cuda' in str(device):
                try:
                    self.one_hots = self.one_hots.to(device)
                except:
                    # If CUDA fails, keep on CPU and move during forward pass
                    pass
            print("Using MSE loss with one-hot targets (Omnigrok setup)")
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.one_hots = None
        
        # AGOP spectral metrics computer
        self.compute_spectral_metrics = compute_spectral_metrics
        self.spectral_metrics_freq = spectral_metrics_freq
        self.compute_per_layer = compute_per_layer
        self.agop_subsample_size = agop_subsample_size
        
        if compute_spectral_metrics:
            self.spectral_computer = SpectralMetricsComputer(
                top_k=spectral_top_k,
                compute_full_spectrum=compute_full_spectrum,
                subsample_size=agop_subsample_size,
                device=device,
                agop_device='cpu'  # Accumulate AGOP on CPU to save GPU memory
            )
            print(f"AGOP tracking enabled: freq={spectral_metrics_freq}, top_k={spectral_top_k}")
            if agop_subsample_size:
                print(f"  Using {agop_subsample_size} subsampled examples for AGOP")
        else:
            self.spectral_computer = None
        
        # Setup save directories
        self.save_dir = Path(save_dir) / experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize history
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'lr': [],
        }
        
        # Spectral metrics history (separate for efficiency)
        self.spectral_history = {
            'epoch': [],
        }
        
        # Per-layer history (if enabled)
        if compute_per_layer:
            self.layer_history = {}
        
        # Save configuration
        self._save_config()
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on name."""
        params = self.model.parameters()
        
        if self.optimizer_name == 'muon':
            return Muon(
                params,
                lr=self.lr,
                weight_decay=0.0,  # No weight decay for base Muon
                momentum=0.95,  # Official Muon default
                use_nesterov=True
            )
        elif self.optimizer_name == 'muonw':
            return MuonW(
                params,
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=0.95,  # Official Muon default
                use_nesterov=True
            )
        elif self.optimizer_name == 'adam':
            return optim.Adam(
                params,
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name == 'adamw':
            return optim.AdamW(
                params,
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name == 'sgd':
            return optim.SGD(
                params,
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
    
    def _save_config(self):
        """Save experiment configuration."""
        config = {
            'optimizer': self.optimizer_name,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'n_epochs': self.n_epochs,
            'model_params': sum(p.numel() for p in self.model.parameters()),
            'train_size': len(self.train_data),
            'test_size': len(self.test_data),
            'device': str(self.device),
            'compute_spectral_metrics': self.compute_spectral_metrics,
            'spectral_metrics_freq': self.spectral_metrics_freq,
        }
        
        with open(self.save_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
    
    def train(self):
        """Main training loop."""
        print(f"Starting training: {self.n_epochs} epochs")
        print(f"Optimizer: {self.optimizer_name}, LR: {self.lr}, Weight Decay: {self.weight_decay}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        print(f"Train size: {len(self.train_data)}, Test size: {len(self.test_data)}")
        print(f"Save directory: {self.save_dir}")
        
        start_time = time.time()
        
        for epoch in tqdm(range(self.n_epochs), desc="Training"):
            # Training step
            train_loss, train_acc = self._train_epoch()
            
            # Evaluation step (every log_freq epochs)
            if epoch % self.log_freq == 0 or epoch == self.n_epochs - 1:
                test_loss, test_acc = self._evaluate()
                
                # Log metrics
                self.history['epoch'].append(epoch)
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
                self.history['test_loss'].append(test_loss)
                self.history['test_acc'].append(test_acc)
                self.history['lr'].append(self.lr)
                
                # Compute spectral metrics
                if (self.compute_spectral_metrics and 
                    epoch % self.spectral_metrics_freq == 0):
                    self._compute_and_log_spectral_metrics(epoch)
                
                # Print progress
                if epoch % (self.log_freq * 10) == 0:
                    print(f"\nEpoch {epoch}/{self.n_epochs}")
                    print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
                    print(f"  Test:  Loss={test_loss:.4f}, Acc={test_acc:.4f}")
            
            # Save checkpoint
            if (self.save_checkpoints and 
                epoch % self.checkpoint_freq == 0 and 
                epoch > 0):
                self._save_checkpoint(epoch)
        
        # Final save
        end_time = time.time()
        print(f"\nTraining completed in {end_time - start_time:.2f} seconds")
        self._save_final_results()
        
        return self.history
    
    def _train_epoch(self) -> tuple:
        """Train for one epoch."""
        self.model.train()
        
        if self.batch_size is None:
            # Full batch training
            self.optimizer.zero_grad()
            logits = self.model(self.train_data)
            
            # Compute loss (MSE or CrossEntropy)
            if self.use_mse_loss:
                # Index on same device as one_hots, then ensure same device as logits
                if self.one_hots.device != self.train_labels.device:
                    labels_cpu = self.train_labels.cpu()
                    targets = self.one_hots[labels_cpu].to(logits.device)
                else:
                    targets = self.one_hots[self.train_labels]
                    if targets.device != logits.device:
                        targets = targets.to(logits.device)
                loss = self.criterion(logits, targets)
            else:
                loss = self.criterion(logits, self.train_labels)
            
            loss.backward()
            self.optimizer.step()
            
            # Compute accuracy
            preds = logits.argmax(dim=-1)
            acc = (preds == self.train_labels).float().mean().item()
            
            return loss.item(), acc
        else:
            # Mini-batch training
            dataset = TensorDataset(self.train_data, self.train_labels)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            total_loss = 0.0
            correct = 0
            total = 0
            
            for batch_data, batch_labels in loader:
                self.optimizer.zero_grad()
                logits = self.model(batch_data)
                
                # Compute loss (MSE or CrossEntropy)
                if self.use_mse_loss:
                    if self.one_hots.device != batch_labels.device:
                        labels_cpu = batch_labels.cpu()
                        targets = self.one_hots[labels_cpu].to(logits.device)
                    else:
                        targets = self.one_hots[batch_labels]
                        if targets.device != logits.device:
                            targets = targets.to(logits.device)
                    loss = self.criterion(logits, targets)
                else:
                    loss = self.criterion(logits, batch_labels)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item() * len(batch_data)
                preds = logits.argmax(dim=-1)
                correct += (preds == batch_labels).sum().item()
                total += len(batch_data)
            
            avg_loss = total_loss / total
            avg_acc = correct / total
            
            return avg_loss, avg_acc
    
    @torch.no_grad()
    def _evaluate(self) -> tuple:
        """Evaluate on test set."""
        self.model.eval()
        
        logits = self.model(self.test_data)
        
        # Compute loss (MSE or CrossEntropy)
        if self.use_mse_loss:
            if self.one_hots.device != self.test_labels.device:
                labels_cpu = self.test_labels.cpu()
                targets = self.one_hots[labels_cpu].to(logits.device)
            else:
                targets = self.one_hots[self.test_labels]
                if targets.device != logits.device:
                    targets = targets.to(logits.device)
            loss = self.criterion(logits, targets)
        else:
            loss = self.criterion(logits, self.test_labels)
        
        preds = logits.argmax(dim=-1)
        acc = (preds == self.test_labels).float().mean().item()
        
        return loss.item(), acc
    
    def _compute_and_log_spectral_metrics(self, epoch: int):
        """Compute and log AGOP-based spectral metrics."""
        if self.spectral_computer is None:
            return
        
        try:
            # Compute AGOP (expensive! processes each sample individually)
            print(f"    Computing AGOP at epoch {epoch}...", end=' ', flush=True)
            agop_start = time.time()
            
            agop_result = self.spectral_computer.compute_agop(
                model=self.model,
                data=self.train_data,
                labels=self.train_labels,
                criterion=self.criterion,
                return_full_matrix=False
            )
            
            agop_time = time.time() - agop_start
            
            if agop_result is not None:
                print(f"Done ({agop_time:.1f}s, {agop_result['n_samples']} samples)")
                
                # Compute metrics from AGOP
                metrics = self.spectral_computer.compute_metrics_from_agop_result(agop_result)
                
                # Log to history
                if not self.spectral_history['epoch']:  # First time
                    # Initialize metric keys
                    for key in metrics.keys():
                        if key != 'eigenvalues_full' and key != 'eigenvectors_top_k':
                            self.spectral_history[key] = []
                
                self.spectral_history['epoch'].append(epoch)
                for key, value in metrics.items():
                    if key != 'eigenvalues_full' and key != 'eigenvectors_top_k':
                        if isinstance(value, (int, float)):
                            self.spectral_history[key].append(value)
                
                # Print key metrics
                print(f"      Trace (E[||∇L||²]): {metrics['trace']:.4e}")
                print(f"      Eigengap: {metrics['eigengap']:.4e}")
                print(f"      Top eigenvalue energy: {metrics['top_eigenvalue_energy_ratio']:.4f}")
                
                # Compute per-layer AGOP metrics if enabled
                if self.compute_per_layer:
                    print(f"    Computing per-layer AGOP...", end=' ', flush=True)
                    layer_start = time.time()
                    
                    layer_metrics = self.spectral_computer.compute_per_layer_agop(
                        model=self.model,
                        data=self.train_data,
                        labels=self.train_labels,
                        criterion=self.criterion
                    )
                    
                    print(f"Done ({time.time() - layer_start:.1f}s)")
                    
                    for layer_name, layer_metric_dict in layer_metrics.items():
                        if layer_name not in self.layer_history:
                            self.layer_history[layer_name] = {'epoch': []}
                            for key in layer_metric_dict.keys():
                                if key != 'eigenvalues_full':
                                    self.layer_history[layer_name][key] = []
                        
                        self.layer_history[layer_name]['epoch'].append(epoch)
                        for key, value in layer_metric_dict.items():
                            if key != 'eigenvalues_full' and isinstance(value, (int, float)):
                                self.layer_history[layer_name][key].append(value)
            else:
                print(f"Failed")
        
        except Exception as e:
            warnings.warn(f"Error computing AGOP metrics at epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint_path = self.save_dir / 'checkpoints' / f'epoch_{epoch}.pt'
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else None,
            'test_loss': self.history['test_loss'][-1] if self.history['test_loss'] else None,
        }, checkpoint_path)
    
    def _save_final_results(self):
        """Save all results to disk."""
        # Save training history as JSON
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Save spectral metrics as HDF5 (more efficient for large arrays)
        if self.spectral_history['epoch']:
            spectral_path = self.save_dir / 'spectral_metrics.h5'
            with h5py.File(spectral_path, 'w') as f:
                for key, values in self.spectral_history.items():
                    f.create_dataset(key, data=np.array(values), compression='gzip')
        
        # Save per-layer metrics
        if self.compute_per_layer and hasattr(self, 'layer_history'):
            layer_path = self.save_dir / 'layer_metrics.h5'
            with h5py.File(layer_path, 'w') as f:
                for layer_name, layer_data in self.layer_history.items():
                    grp = f.create_group(layer_name.replace('.', '_'))
                    for key, values in layer_data.items():
                        grp.create_dataset(key, data=np.array(values), compression='gzip')
        
        print(f"Results saved to {self.save_dir}")

