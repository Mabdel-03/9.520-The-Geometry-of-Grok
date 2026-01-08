"""
Input-Gradient AGOP Tracker for Grokking Analysis

This module implements tractable AGOP (Average Gradient Outer Product) computation
using input gradients rather than parameter gradients, making it computationally 
feasible for large models.

Based on: "Average gradient outer product as a mechanism for deep neural collapse"
by Beaglehole et al., adapted from Group1_Grokking_Code_Base.ipynb

AGOP_input = (1/N) Σᵢ (∇_x f(xᵢ) ⊗ ∇_x f(xᵢ))

Key metrics tracked:
- Frobenius norm: ||AGOP||_F
- Spectral radius: λ₁ (largest eigenvalue)
- Trace: Σλᵢ (total variance)
- Eigengap: λ₁ - λ₂ (gradient alignment)
- Top-k subspace similarity: Stability of gradient directions
- Variation collapse ratio: λ₁/Σλᵢ (concentration measure)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
import warnings


class InputGradientAGOPTracker:
    """
    Computes AGOP metrics using input gradients (tractable for large models).
    
    Instead of computing gradients w.r.t. parameters (expensive for large models),
    we compute gradients w.r.t. inputs, which is much more tractable since
    input_dim << param_dim (e.g., 194 vs 100K for modular arithmetic).
    
    This analyzes the geometry of the input space and how the model's sensitivity
    to inputs evolves during training.
    """
    
    def __init__(
        self,
        top_k: int = 4,
        subsample_size: Optional[int] = None,
        device: str = 'cuda',
        agop_device: str = 'cpu',  # Accumulate on CPU to save GPU memory
        use_mse_loss: bool = False,  # For MNIST Omnigrok experiments
    ):
        """
        Args:
            top_k: Number of top eigenvectors to track for subspace similarity
            subsample_size: If set, randomly sample this many examples for AGOP
            device: Device for model computation (GPU)
            agop_device: Device for AGOP accumulation (CPU recommended)
            use_mse_loss: Whether to use MSE loss with one-hot targets (Omnigrok)
        """
        self.top_k = top_k
        self.subsample_size = subsample_size
        self.device = device
        self.agop_device = agop_device
        self.use_mse_loss = use_mse_loss
        
        # For MSE loss (Omnigrok MNIST)
        if use_mse_loss:
            self.one_hots = torch.eye(10, dtype=torch.float32)
    
    def compute_input_agop(
        self,
        model: nn.Module,
        data: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module,
    ) -> Optional[torch.Tensor]:
        """
        Compute Average Gradient Outer Product using input gradients.
        
        AGOP_input = (1/N) Σᵢ (∇_x f(xᵢ) ⊗ ∇_x f(xᵢ))
        
        This is much more tractable than parameter-gradient AGOP since
        input_dim << param_dim.
        
        Args:
            model: PyTorch model
            data: Input data (N, d_input)
            labels: Labels (N,)
            criterion: Loss function
            
        Returns:
            AGOP matrix (d_input, d_input) or None if computation fails
        """
        try:
            # Determine samples to use
            n_total = len(data)
            if self.subsample_size is not None and self.subsample_size < n_total:
                indices = torch.randperm(n_total)[:self.subsample_size]
                data_subset = data[indices]
                labels_subset = labels[indices]
                n_samples = self.subsample_size
            else:
                data_subset = data
                labels_subset = labels
                n_samples = n_total
            
            # Get input dimension
            if len(data_subset.shape) == 2:
                d_input = data_subset.shape[1]
            else:
                # Flatten for images
                d_input = np.prod(data_subset.shape[1:])
            
            # Initialize AGOP on CPU (to save GPU memory)
            agop = torch.zeros(d_input, d_input, dtype=torch.float32, device=self.agop_device)
            
            model.eval()  # Use eval mode to disable dropout
            
            # Process each sample
            for i in range(n_samples):
                # Get single sample
                x_i = data_subset[i:i+1].clone().detach().to(self.device)
                y_i = labels_subset[i:i+1].to(self.device)
                
                # Ensure input is float (required for gradient computation)
                if x_i.dtype not in [torch.float32, torch.float64, torch.float16]:
                    x_i = x_i.float()
                
                # Enable gradient tracking
                x_i.requires_grad_(True)
                
                # Forward pass
                logits = model(x_i)
                
                # Choose scalar function to differentiate
                # Use mean logit for true class (more meaningful than loss)
                f_scalar = logits[torch.arange(y_i.size(0)), y_i].mean()
                
                # Compute ∇_x f(x)
                grads = torch.autograd.grad(
                    outputs=f_scalar,
                    inputs=x_i,
                    retain_graph=False,
                    create_graph=False
                )[0]  # shape: [1, d_input] or [1, C, H, W]
                
                # Flatten gradient
                grad_flat = grads.detach().cpu().flatten()  # [d_input]
                
                # Accumulate outer product (on CPU)
                agop.add_(torch.outer(grad_flat, grad_flat))
                
                # Cleanup
                del x_i, y_i, logits, grads, grad_flat
                if i % 100 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Average over samples
            agop.div_(n_samples)
            
            return agop
            
        except Exception as e:
            warnings.warn(f"Error computing input-gradient AGOP: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def compute_agop_metrics(
        self,
        history: Dict,
        epoch_agop: torch.Tensor,
        k: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Compute comprehensive AGOP metrics from an AGOP matrix.
        
        This function extracts the key metrics that characterize the AGOP
        and tracks them over training. Adapted from notebook Cell 4.
        
        Args:
            history: Dictionary to update with metrics
            epoch_agop: AGOP matrix for this epoch (d, d)
            k: Number of top eigenvectors for subspace similarity (default: self.top_k)
            
        Returns:
            Dictionary of computed metrics
        """
        if k is None:
            k = self.top_k
        
        metrics = {}
        
        try:
            # Compute eigendecomposition
            eigvals, eigvecs = torch.linalg.eigh(epoch_agop)
            
            # Sort in descending order
            eigvals = eigvals.flip(0)
            eigvecs = eigvecs.flip(1)
            
            lambda1 = eigvals[0].item()
            lambda2 = eigvals[1].item() if len(eigvals) > 1 else 0.0
            
            # Top-k eigenvectors for subspace tracking
            U_k = eigvecs[:, :k].cpu()
            
            # ---------- Store basic scalars ----------
            metrics['agop_frobenius'] = torch.norm(epoch_agop, p='fro').item()
            metrics['agop_spectral_radius'] = lambda1
            metrics['agop_trace'] = eigvals.sum().item()
            metrics['agop_eigengap'] = lambda1 - lambda2
            
            # ---------- Variation collapse ratio ----------
            if metrics['agop_trace'] > 1e-10:
                metrics['agop_variation_collapse_ratio'] = lambda1 / metrics['agop_trace']
            else:
                metrics['agop_variation_collapse_ratio'] = 0.0
            
            # ---------- Top-k subspace similarity ----------
            # Initialize history keys if first call
            if 'agop_topk_subspace_prev' not in history:
                history['agop_topk_subspace_prev'] = None
            
            if history['agop_topk_subspace_prev'] is None:
                metrics['agop_topk_subspace_similarity'] = float('nan')
            else:
                U_prev = history['agop_topk_subspace_prev']
                M = U_k.T @ U_prev
                svals = torch.linalg.svdvals(M)
                metrics['agop_topk_subspace_similarity'] = svals.mean().item()
            
            # Update previous subspace
            history['agop_topk_subspace_prev'] = U_k
            
            # ---------- Energy concentration metrics ----------
            total_energy = eigvals.sum().item()
            if total_energy > 1e-10:
                # Top eigenvalue energy ratio
                metrics['agop_top_eigenvalue_energy'] = lambda1 / total_energy
                
                # Top-5 energy ratio
                top5_energy = eigvals[:min(5, len(eigvals))].sum().item()
                metrics['agop_top5_energy_ratio'] = top5_energy / total_energy
                
                # Top-10 energy ratio
                top10_energy = eigvals[:min(10, len(eigvals))].sum().item()
                metrics['agop_top10_energy_ratio'] = top10_energy / total_energy
            else:
                metrics['agop_top_eigenvalue_energy'] = 0.0
                metrics['agop_top5_energy_ratio'] = 0.0
                metrics['agop_top10_energy_ratio'] = 0.0
            
            # ---------- Store top eigenvalues ----------
            for i in range(min(10, len(eigvals))):
                metrics[f'agop_eigenvalue_{i+1}'] = eigvals[i].item()
            
            # Update history with all metrics
            for key, value in metrics.items():
                if key not in history:
                    history[key] = []
                history[key].append(value)
            
            return metrics
            
        except Exception as e:
            warnings.warn(f"Error computing AGOP metrics: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def smooth_series(self, x: np.ndarray, window: int = 5) -> np.ndarray:
        """
        Apply moving average smoothing to a time series.
        
        Args:
            x: Input array
            window: Window size for moving average
            
        Returns:
            Smoothed array
        """
        smoothed = np.zeros_like(x, dtype=float)
        for i in range(len(x)):
            start = max(0, i - window + 1)
            smoothed[i] = np.mean(x[start:i + 1])
        return smoothed


def test_input_agop():
    """Test the input-gradient AGOP computation on a toy example."""
    print("="*80)
    print("Testing Input-Gradient AGOP Tracker")
    print("="*80)
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    # Create toy dataset
    n_samples = 50
    x = torch.randn(n_samples, 10)
    y = torch.randint(0, 5, (n_samples,))
    
    criterion = nn.CrossEntropyLoss()
    
    # Compute input-gradient AGOP
    print(f"\nComputing input-gradient AGOP for {n_samples} samples...")
    tracker = InputGradientAGOPTracker(
        top_k=4,
        subsample_size=None,
        agop_device='cpu'
    )
    
    agop = tracker.compute_input_agop(
        model=model,
        data=x,
        labels=y,
        criterion=criterion
    )
    
    if agop is not None:
        print(f"✓ Input-gradient AGOP computed successfully")
        print(f"  AGOP shape: {agop.shape} (expected: [10, 10])")
        print(f"  AGOP trace: {torch.trace(agop).item():.6f}")
        
        # Compute metrics
        history = {}
        metrics = tracker.compute_agop_metrics(history, agop)
        
        print("\n" + "="*80)
        print("AGOP Metrics:")
        print("="*80)
        for key, value in sorted(metrics.items()):
            if not key.startswith('agop_eigenvalue'):
                print(f"  {key:.<50} {value:.6f}")
        
        print("\nTop 5 eigenvalues:")
        for i in range(5):
            key = f'agop_eigenvalue_{i+1}'
            if key in metrics:
                print(f"  λ_{i+1}: {metrics[key]:.6f}")
        
        print("\n" + "="*80)
        print("✓ Input-gradient AGOP test passed!")
        print("="*80)
        
    else:
        print("✗ Failed to compute input-gradient AGOP")


if __name__ == "__main__":
    test_input_agop()

