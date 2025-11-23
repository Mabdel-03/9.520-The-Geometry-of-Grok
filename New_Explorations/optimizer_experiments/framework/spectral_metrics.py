"""
Spectral Metrics Computation for Average Gradient Outer Product (AGOP) Analysis
Computes eigengap, top-k subspace, energy concentration, spectral radius, trace, etc.

Based on: "Average gradient outer product as a mechanism for deep neural collapse"
by Beaglehole et al.

AGOP = (1/N) Σᵢ (∇L(xᵢ) ⊗ ∇L(xᵢ))
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, List
import warnings


class SpectralMetricsComputer:
    """
    Computes comprehensive spectral metrics from Average Gradient Outer Product (AGOP).
    
    AGOP = (1/N) Σᵢ (∇L(xᵢ) ⊗ ∇L(xᵢ))
    
    Metrics tracked:
    - Eigengap: λ₁ - λ₂ (gradient alignment measure)
    - Top-k subspace: Energy in top-k eigenvectors
    - Energy concentration in top eigenvector: λ₁/Σλᵢ (neural collapse indicator)
    - Spectral radius: λ_max (maximum variance direction)
    - Trace: Σλᵢ = E[||∇L||²] (total gradient variance)
    - Spectral radius to trace ratio: λ_max/Σλᵢ
    - Effective rank: Participation ratio of eigenvalues
    
    Memory-efficient implementation:
    - Accumulates AGOP on CPU
    - Computes only top-k eigenvalues by default
    - Supports subsampling for very large datasets
    """
    
    def __init__(
        self,
        top_k: int = 20,
        compute_full_spectrum: bool = False,
        subsample_size: Optional[int] = None,
        device: str = 'cuda',
        agop_device: str = 'cpu'  # Where to accumulate AGOP (CPU to save GPU memory)
    ):
        """
        Args:
            top_k: Number of top eigenvalues/eigenvectors to track
            compute_full_spectrum: Whether to compute all eigenvalues (expensive!)
            subsample_size: If set, randomly sample this many examples for AGOP
            device: Device for model computation (GPU)
            agop_device: Device for AGOP accumulation (CPU recommended for memory)
        """
        self.top_k = top_k
        self.compute_full_spectrum = compute_full_spectrum
        self.subsample_size = subsample_size
        self.device = device
        self.agop_device = agop_device
        
    def compute_agop(
        self,
        model: torch.nn.Module,
        data: torch.Tensor,
        labels: torch.Tensor,
        criterion: torch.nn.Module,
        return_full_matrix: bool = False
    ) -> Optional[Dict]:
        """
        Compute Average Gradient Outer Product (AGOP) in a memory-efficient manner.
        
        AGOP = (1/N) Σᵢ (∇L(xᵢ) ⊗ ∇L(xᵢ))
        
        Memory-efficient strategy:
        1. Accumulate AGOP on CPU (cheaper memory)
        2. Process samples one at a time or in small batches
        3. Return only eigenvalues/eigenvectors, not full matrix
        
        Args:
            model: PyTorch model
            data: Input data (N, ...)
            labels: Labels (N,)
            criterion: Loss function
            return_full_matrix: If True, return full AGOP matrix (memory intensive!)
            
        Returns:
            Dictionary with:
                - 'eigenvalues': Top-k eigenvalues (or all if compute_full_spectrum)
                - 'trace': Trace of AGOP (sum of all eigenvalues = E[||∇L||²])
                - 'n_params': Number of parameters
                - 'n_samples': Number of samples used
                - 'agop_matrix': Full matrix (only if return_full_matrix=True)
        """
        try:
            # Determine number of samples to use
            n_total = len(data)
            if self.subsample_size is not None and self.subsample_size < n_total:
                # Random subsampling
                indices = torch.randperm(n_total)[:self.subsample_size]
                data_subset = data[indices]
                labels_subset = labels[indices]
                n_samples = self.subsample_size
            else:
                data_subset = data
                labels_subset = labels
                n_samples = n_total
            
            # Get number of parameters
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Initialize AGOP on CPU (to save GPU memory)
            agop = torch.zeros(n_params, n_params, dtype=torch.float32, device=self.agop_device)
            
            # Process each sample individually
            model.train()
            for i in range(n_samples):
                # Get single sample
                x_i = data_subset[i:i+1].to(self.device)
                y_i = labels_subset[i:i+1].to(self.device)
                
                # Zero gradients
                model.zero_grad()
                
                # Forward pass
                output = model(x_i)
                loss_i = criterion(output, y_i)
                
                # Backward pass
                loss_i.backward()
                
                # Collect gradient vector (move to CPU immediately to save GPU memory)
                grad_list = []
                for param in model.parameters():
                    if param.requires_grad and param.grad is not None:
                        grad_list.append(param.grad.detach().cpu().view(-1))
                
                if not grad_list:
                    continue
                
                grad_i = torch.cat(grad_list)
                
                # Accumulate outer product (on CPU)
                # This is the memory bottleneck, but unavoidable for full AGOP
                agop.add_(torch.outer(grad_i, grad_i))
                
                # Clear GPU memory
                del output, loss_i, x_i, y_i
                if i % 100 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Average over samples
            agop.div_(n_samples)
            
            # Compute trace (can be done before eigendecomposition)
            trace = torch.trace(agop).item()
            
            # Prepare return dictionary
            result = {
                'trace': trace,
                'n_params': n_params,
                'n_samples': n_samples,
            }
            
            # Compute eigenvalues
            if self.compute_full_spectrum or n_params <= 5000:
                # Full eigendecomposition (expensive!)
                eigenvalues, eigenvectors = torch.linalg.eigh(agop)
                eigenvalues = eigenvalues.flip(0)  # Descending order
                eigenvectors = eigenvectors.flip(1)
                result['eigenvalues'] = eigenvalues.numpy()
                result['eigenvectors'] = eigenvectors[:, :self.top_k].numpy()
            else:
                # Top-k eigenvalues only (more efficient)
                # Use power iteration or Lanczos for top-k
                # For now, compute full spectrum but this could be optimized
                eigenvalues, eigenvectors = torch.linalg.eigh(agop)
                eigenvalues = eigenvalues.flip(0)
                eigenvectors = eigenvectors.flip(1)
                result['eigenvalues'] = eigenvalues[:max(self.top_k, 100)].numpy()  # Store top-100 for analysis
                result['eigenvectors'] = eigenvectors[:, :self.top_k].numpy()
            
            # Optionally return full matrix
            if return_full_matrix:
                result['agop_matrix'] = agop
            
            return result
            
        except Exception as e:
            warnings.warn(f"Error computing AGOP: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def compute_metrics_from_agop_result(
        self,
        agop_result: Dict
    ) -> Dict[str, float]:
        """
        Compute all spectral metrics from AGOP result dictionary.
        
        Args:
            agop_result: Dictionary from compute_agop() containing eigenvalues and trace
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        try:
            eigenvalues_np = agop_result['eigenvalues']
            trace = agop_result['trace']
            n_samples = agop_result['n_samples']
            
            # Basic statistics
            metrics['trace'] = float(eigenvalues_np.sum())
            metrics['frobenius_norm'] = float(np.sqrt((eigenvalues_np ** 2).sum()))
            
            # Spectral radius (largest eigenvalue magnitude)
            metrics['spectral_radius'] = float(np.abs(eigenvalues_np[0]))
            
            # Top eigenvalues
            for i in range(min(self.top_k, len(eigenvalues_np))):
                metrics[f'eigenvalue_{i+1}'] = float(eigenvalues_np[i])
            
            # Eigengap: difference between largest and second-largest eigenvalue
            if len(eigenvalues_np) >= 2:
                metrics['eigengap'] = float(eigenvalues_np[0] - eigenvalues_np[1])
                metrics['eigengap_ratio'] = float(
                    eigenvalues_np[0] / (eigenvalues_np[1] + 1e-10)
                )
            else:
                metrics['eigengap'] = 0.0
                metrics['eigengap_ratio'] = 1.0
            
            # Energy concentration in top eigenvector
            total_energy = eigenvalues_np.sum()
            if total_energy > 0:
                metrics['top_eigenvalue_energy_ratio'] = float(
                    eigenvalues_np[0] / total_energy
                )
            else:
                metrics['top_eigenvalue_energy_ratio'] = 0.0
            
            # Energy in top-k subspace
            for k in [5, 10, 20, 50]:
                if k <= len(eigenvalues_np):
                    top_k_energy = eigenvalues_np[:k].sum()
                    if total_energy > 0:
                        metrics[f'top_{k}_energy_ratio'] = float(
                            top_k_energy / total_energy
                        )
                    else:
                        metrics[f'top_{k}_energy_ratio'] = 0.0
            
            # Spectral radius to trace ratio
            if metrics['trace'] != 0:
                metrics['spectral_radius_to_trace_ratio'] = float(
                    metrics['spectral_radius'] / abs(metrics['trace'])
                )
            else:
                metrics['spectral_radius_to_trace_ratio'] = 0.0
            
            # Effective rank (participation ratio) - measures dimensionality of gradient space
            # This indicates how many principal gradient directions are active
            if total_energy > 0:
                normalized_eigs = eigenvalues_np / total_energy
                # Filter out numerical zeros
                normalized_eigs = normalized_eigs[normalized_eigs > 1e-10]
                if len(normalized_eigs) > 0:
                    # Entropy-based effective rank
                    metrics['effective_rank'] = float(
                        np.exp(-np.sum(normalized_eigs * np.log(normalized_eigs + 1e-10)))
                    )
                    # Alternative: inverse participation ratio
                    metrics['inverse_participation_ratio'] = float(
                        1.0 / np.sum(normalized_eigs ** 2)
                    )
                else:
                    metrics['effective_rank'] = 0.0
                    metrics['inverse_participation_ratio'] = 0.0
            else:
                metrics['effective_rank'] = 0.0
                metrics['inverse_participation_ratio'] = 0.0
            
            # Condition number (ratio of largest to smallest non-zero eigenvalue)
            # High condition number indicates ill-conditioned optimization
            nonzero_eigs = eigenvalues_np[np.abs(eigenvalues_np) > 1e-10]
            if len(nonzero_eigs) >= 2:
                metrics['condition_number'] = float(
                    np.abs(nonzero_eigs[0]) / np.abs(nonzero_eigs[-1])
                )
            else:
                metrics['condition_number'] = 1.0
            
            # Store full spectrum if requested
            if self.compute_full_spectrum:
                metrics['eigenvalues_full'] = eigenvalues_np.tolist()
                
        except Exception as e:
            warnings.warn(f"Error computing spectral metrics: {e}")
            import traceback
            traceback.print_exc()
            # Return default metrics
            metrics = {
                'trace': 0.0,
                'spectral_radius': 0.0,
                'eigengap': 0.0,
                'top_eigenvalue_energy_ratio': 0.0,
                'spectral_radius_to_trace_ratio': 0.0,
                'effective_rank': 0.0,
                'condition_number': 1.0,
                'n_samples_used': 0
            }
        
        return metrics
    
    def compute_metrics(
        self, 
        gop: torch.Tensor,
        compute_eigenvectors: bool = False
    ) -> Dict[str, float]:
        """
        Legacy method for backward compatibility.
        Computes metrics from a pre-computed GOP/AGOP matrix.
        
        NOTE: For proper AGOP analysis, use compute_agop() instead.
        
        Args:
            gop: Gradient outer product matrix (M x M)
            compute_eigenvectors: Whether to return eigenvectors
            
        Returns:
            Dictionary with all metrics
        """
        warnings.warn(
            "compute_metrics() from matrix is deprecated. "
            "Use compute_agop() for proper AGOP computation.",
            DeprecationWarning
        )
        
        # Compute eigendecomposition
        gop_cpu = gop.detach().cpu()
        eigenvalues, eigenvectors = torch.linalg.eigh(gop_cpu)
        eigenvalues = eigenvalues.flip(0)
        
        # Create fake agop_result for compatibility
        agop_result = {
            'eigenvalues': eigenvalues.numpy(),
            'trace': torch.trace(gop_cpu).item(),
            'n_samples': 1  # Unknown for pre-computed matrix
        }
        
        return self.compute_metrics_from_agop_result(agop_result)
    
    def compute_per_layer_agop(
        self,
        model: torch.nn.Module,
        data: torch.Tensor,
        labels: torch.Tensor,
        criterion: torch.nn.Module
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute AGOP-based spectral metrics for each layer separately.
        
        For each layer: AGOP_layer = (1/N) Σᵢ (∇_layer L(xᵢ) ⊗ ∇_layer L(xᵢ))
        
        Args:
            model: PyTorch model
            data: Input data
            labels: Labels
            criterion: Loss function
            
        Returns:
            Dictionary mapping layer names to their spectral metrics
        """
        layer_metrics = {}
        
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
        
        # Initialize AGOP for each layer
        layer_agops = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                n_params = param.numel()
                layer_agops[name] = torch.zeros(
                    n_params, n_params, 
                    dtype=torch.float32, 
                    device=self.agop_device
                )
        
        # Accumulate per-sample gradients
        model.train()
        for i in range(n_samples):
            x_i = data_subset[i:i+1].to(self.device)
            y_i = labels_subset[i:i+1].to(self.device)
            
            model.zero_grad()
            output = model(x_i)
            loss_i = criterion(output, y_i)
            loss_i.backward()
            
            # Accumulate for each layer
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_i = param.grad.detach().cpu().view(-1)
                    layer_agops[name].add_(torch.outer(grad_i, grad_i))
            
            # Clean up
            del output, loss_i, x_i, y_i
            if i % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Average and compute metrics for each layer
        for name, agop in layer_agops.items():
            agop.div_(n_samples)
            
            # Compute eigenvalues
            eigenvalues, _ = torch.linalg.eigh(agop)
            eigenvalues = eigenvalues.flip(0)
            
            # Create result dict
            agop_result = {
                'eigenvalues': eigenvalues.numpy(),
                'trace': torch.trace(agop).item(),
                'n_samples': n_samples
            }
            
            # Compute metrics
            layer_metrics[name] = self.compute_metrics_from_agop_result(agop_result)
        
        return layer_metrics


def test_spectral_metrics():
    """Test the AGOP computation on a toy example."""
    print("="*80)
    print("Testing AGOP-based SpectralMetricsComputer")
    print("="*80)
    
    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 2)
    )
    
    # Create toy dataset
    n_samples = 20
    x = torch.randn(n_samples, 10)
    y = torch.randint(0, 2, (n_samples,))
    
    criterion = torch.nn.CrossEntropyLoss()
    
    # Compute AGOP
    print(f"\nComputing AGOP for {n_samples} samples...")
    computer = SpectralMetricsComputer(
        top_k=5, 
        compute_full_spectrum=False,
        subsample_size=None,
        agop_device='cpu'
    )
    
    agop_result = computer.compute_agop(
        model=model,
        data=x,
        labels=y,
        criterion=criterion,
        return_full_matrix=False
    )
    
    if agop_result is not None:
        print(f"✓ AGOP computed successfully")
        print(f"  Number of parameters: {agop_result['n_params']}")
        print(f"  Samples used: {agop_result['n_samples']}")
        print(f"  Trace (E[||∇L||²]): {agop_result['trace']:.6f}")
        print(f"  Top eigenvalues: {agop_result['eigenvalues'][:5]}")
        
        # Compute metrics
        metrics = computer.compute_metrics_from_agop_result(agop_result)
        
        print("\n" + "="*80)
        print("Spectral Metrics (from AGOP):")
        print("="*80)
        for key, value in sorted(metrics.items()):
            if key != 'eigenvalues_full':
                if isinstance(value, float):
                    print(f"  {key:.<40} {value:.6f}")
                else:
                    print(f"  {key:.<40} {value}")
        
        print("\n" + "="*80)
        print("✓ AGOP SpectralMetricsComputer test passed!")
        print("="*80)
        
        # Test with subsampling
        print("\nTesting with subsampling (10 samples)...")
        computer_sub = SpectralMetricsComputer(
            top_k=5,
            subsample_size=10
        )
        agop_result_sub = computer_sub.compute_agop(model, x, y, criterion)
        if agop_result_sub:
            print(f"✓ Subsampled AGOP: used {agop_result_sub['n_samples']} samples")
        
    else:
        print("✗ Failed to compute AGOP")


if __name__ == "__main__":
    test_spectral_metrics()

