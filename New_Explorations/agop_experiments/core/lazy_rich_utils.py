"""
Lazy-to-Rich Training Dynamics Tracker

Implements metrics from Kumar et al. (2024) "Grokking as the Transition from 
Lazy to Rich Training Dynamics" to track when neural networks transition from
lazy training (fixed features, kernel regression) to rich training (feature learning).

Key metrics:
- Weight Norm Evolution: ||θₜ||₂ (total and per-layer)
- NTK Distance: ||Kₜ - K₀||_F / ||K₀||_F (normalized distance from initialization)
- Feature Kernel Distance: How hidden representations change from initialization

Reference: https://arxiv.org/abs/2310.06110
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, List, Tuple, Callable
import warnings
from functools import partial


def compute_weight_norms(model: nn.Module) -> Dict[str, float]:
    """
    Compute L2 norms of model weights (total and per-layer).
    
    This is cheap to compute (O(p) where p = num params) and can be
    logged every epoch. Kumar et al. observed weight norm increases
    during the lazy→rich transition.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with 'total' and 'layer_{i}' norms
    """
    norms = {}
    total_sq = 0.0
    
    for i, (name, param) in enumerate(model.named_parameters()):
        if param.requires_grad:
            layer_norm = param.data.norm(2).item()
            # Clean up the name for readability
            clean_name = name.replace('.', '_')
            norms[f'layer_{clean_name}'] = layer_norm
            total_sq += layer_norm ** 2
    
    norms['total'] = np.sqrt(total_sq)
    return norms


def compute_ntk_subsample(
    model: nn.Module,
    data: torch.Tensor,
    n_subsample: int = 200,
    device: str = 'cuda',
    output_device: str = 'cpu'
) -> torch.Tensor:
    """
    Compute the Neural Tangent Kernel (NTK) on a subsample of data.
    
    The NTK measures how changes in parameters affect outputs:
        K_θ(x, x') = <∇_θ f(x; θ), ∇_θ f(x'; θ)>
    
    In the lazy regime, K_t ≈ K_0 (kernel doesn't change).
    In the rich regime, K_t evolves significantly (feature learning).
    
    For multi-output networks, we sum over output dimensions:
        K_θ(x, x') = Σ_c <∇_θ f_c(x), ∇_θ f_c(x')>
    
    Args:
        model: PyTorch model
        data: Input data tensor [N, d_input]
        n_subsample: Number of points to use (for tractability)
        device: Device for model computation
        output_device: Device for NTK storage (CPU recommended)
        
    Returns:
        NTK Gram matrix [n_subsample, n_subsample]
    """
    model.eval()
    
    # Subsample data
    n_total = len(data)
    if n_subsample < n_total:
        indices = torch.randperm(n_total)[:n_subsample]
        data_sub = data[indices].to(device)
    else:
        data_sub = data.to(device)
        n_subsample = n_total
    
    # Get number of parameters
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    
    # Compute Jacobians for each data point
    # J[i] = ∇_θ f(x_i) with shape [n_outputs, n_params]
    jacobians = []
    
    for i in range(n_subsample):
        x_i = data_sub[i:i+1]
        
        # Ensure float for gradient computation
        if x_i.dtype not in [torch.float32, torch.float64, torch.float16]:
            x_i = x_i.float()
        
        # Forward pass
        output = model(x_i)  # [1, n_outputs]
        n_outputs = output.shape[-1]
        
        # Compute gradient for each output
        grads_per_output = []
        for c in range(n_outputs):
            model.zero_grad()
            output_c = model(x_i)[0, c]  # Recompute to get fresh graph
            output_c.backward(retain_graph=(c < n_outputs - 1))
            
            # Collect gradients
            grad_flat = torch.cat([p.grad.flatten() for p in params])
            grads_per_output.append(grad_flat.detach())
        
        # Stack gradients: [n_outputs, n_params]
        jacobian_i = torch.stack(grads_per_output, dim=0)
        jacobians.append(jacobian_i.to(output_device))
        
        # Cleanup
        if i % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Compute NTK: K[i,j] = Σ_c <J_c(x_i), J_c(x_j)>
    # This is equivalent to tr(J_i @ J_j^T)
    ntk = torch.zeros(n_subsample, n_subsample, device=output_device)
    
    for i in range(n_subsample):
        for j in range(i, n_subsample):
            # Inner product over all outputs and params
            k_ij = (jacobians[i] * jacobians[j]).sum().item()
            ntk[i, j] = k_ij
            ntk[j, i] = k_ij
    
    return ntk


def compute_ntk_efficient(
    model: nn.Module,
    data: torch.Tensor,
    n_subsample: int = 200,
    device: str = 'cuda',
    output_device: str = 'cpu'
) -> torch.Tensor:
    """
    Compute NTK more efficiently using vectorized operations.
    
    Uses the fact that K = J @ J^T where J is the stacked Jacobian matrix.
    We compute J row by row and accumulate K incrementally.
    
    Args:
        model: PyTorch model
        data: Input data tensor [N, d_input]
        n_subsample: Number of points to use
        device: Device for model computation
        output_device: Device for NTK storage
        
    Returns:
        NTK Gram matrix [n_subsample, n_subsample]
    """
    model.eval()
    
    # Subsample data
    n_total = len(data)
    if n_subsample < n_total:
        indices = torch.randperm(n_total)[:n_subsample]
        data_sub = data[indices].to(device)
    else:
        data_sub = data.to(device)
        n_subsample = n_total
    
    # Get parameter info
    params = [p for p in model.parameters() if p.requires_grad]
    
    # First pass: determine output dimension
    with torch.no_grad():
        test_out = model(data_sub[0:1].float() if data_sub.dtype not in [torch.float32, torch.float64] else data_sub[0:1])
        n_outputs = test_out.shape[-1]
    
    # Compute Jacobian rows and accumulate NTK
    jacobian_rows = []  # Store [n_subsample, n_outputs * n_params] flattened
    
    for i in range(n_subsample):
        x_i = data_sub[i:i+1]
        if x_i.dtype not in [torch.float32, torch.float64, torch.float16]:
            x_i = x_i.float()
        
        row_grads = []
        for c in range(n_outputs):
            model.zero_grad()
            output = model(x_i)
            output[0, c].backward()
            
            grad_flat = torch.cat([p.grad.flatten() for p in params])
            row_grads.append(grad_flat)
        
        # Concatenate all output gradients: [n_outputs * n_params]
        jacobian_row = torch.cat(row_grads).to(output_device)
        jacobian_rows.append(jacobian_row)
        
        if i % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Stack into matrix [n_subsample, n_outputs * n_params]
    J = torch.stack(jacobian_rows, dim=0)
    
    # NTK = J @ J^T
    ntk = J @ J.T
    
    return ntk


def compute_feature_kernel(
    model: nn.Module,
    data: torch.Tensor,
    layer_name: Optional[str] = None,
    n_subsample: Optional[int] = None,
    device: str = 'cuda',
    output_device: str = 'cpu'
) -> torch.Tensor:
    """
    Compute the feature kernel from hidden layer activations.
    
    The feature kernel is:
        K_features(x, x') = <h(x), h(x')>
    
    where h(x) are the hidden layer activations. This is a cheaper proxy
    for feature learning than the full NTK.
    
    Args:
        model: PyTorch model
        data: Input data tensor [N, d_input]
        layer_name: Name of layer to hook (if None, uses last layer before output)
        n_subsample: Number of points to use (None = use all)
        device: Device for computation
        output_device: Device for kernel storage
        
    Returns:
        Feature kernel Gram matrix [n, n]
    """
    model.eval()
    
    # Subsample if needed
    n_total = len(data)
    if n_subsample is not None and n_subsample < n_total:
        indices = torch.randperm(n_total)[:n_subsample]
        data_sub = data[indices].to(device)
        n = n_subsample
    else:
        data_sub = data.to(device)
        n = n_total
    
    if data_sub.dtype not in [torch.float32, torch.float64, torch.float16]:
        data_sub = data_sub.float()
    
    # Set up hook to capture activations
    activations = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    # Find the layer to hook
    hooks = []
    target_layer = None
    
    if layer_name is not None:
        # User specified a layer
        for name, module in model.named_modules():
            if name == layer_name:
                target_layer = name
                hooks.append(module.register_forward_hook(hook_fn(name)))
                break
    else:
        # Find the last linear layer before output (heuristic)
        linear_layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers.append((name, module))
        
        if len(linear_layers) >= 2:
            # Use second-to-last linear layer (before output projection)
            target_layer = linear_layers[-2][0]
            hooks.append(linear_layers[-2][1].register_forward_hook(hook_fn(target_layer)))
        elif len(linear_layers) == 1:
            # Use the only linear layer's input
            target_layer = linear_layers[0][0]
            hooks.append(linear_layers[0][1].register_forward_hook(hook_fn(target_layer)))
    
    if not hooks:
        warnings.warn("Could not find layer to hook. Using model output as features.")
        with torch.no_grad():
            features = model(data_sub).to(output_device)
    else:
        # Forward pass to capture activations
        with torch.no_grad():
            _ = model(data_sub)
        
        # Get features
        features = activations[target_layer]
        
        # Flatten if needed (e.g., for transformer outputs with sequence dim)
        if len(features.shape) > 2:
            features = features.view(features.shape[0], -1)
        
        features = features.to(output_device)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
    
    # Compute kernel: K[i,j] = <h_i, h_j>
    kernel = features @ features.T
    
    return kernel


class LazyRichTracker:
    """
    Tracks lazy-to-rich training dynamics metrics.
    
    Stores initial NTK (K₀) and feature kernel, then computes distances
    during training to detect the transition from lazy to rich regime.
    
    Attributes:
        ntk_0: Initial NTK [n, n]
        feature_kernel_0: Initial feature kernel [n, n]
        ntk_0_norm: ||K₀||_F for normalization
        feature_kernel_0_norm: ||K_features(0)||_F for normalization
    """
    
    def __init__(
        self,
        n_subsample: int = 200,
        device: str = 'cuda',
        output_device: str = 'cpu',
        feature_layer: Optional[str] = None,
        use_efficient_ntk: bool = True
    ):
        """
        Args:
            n_subsample: Number of points for NTK/kernel computation
            device: Device for model computation
            output_device: Device for storing kernels
            feature_layer: Layer name for feature kernel (None = auto-detect)
            use_efficient_ntk: Use efficient vectorized NTK computation
        """
        self.n_subsample = n_subsample
        self.device = device
        self.output_device = output_device
        self.feature_layer = feature_layer
        self.use_efficient_ntk = use_efficient_ntk
        
        # Will be set during initialization
        self.ntk_0: Optional[torch.Tensor] = None
        self.feature_kernel_0: Optional[torch.Tensor] = None
        self.ntk_0_norm: float = 0.0
        self.feature_kernel_0_norm: float = 0.0
        self.weight_norms_0: Optional[Dict[str, float]] = None
        
        # Subsample indices (for consistent comparison)
        self.subsample_indices: Optional[torch.Tensor] = None
        
    def initialize(
        self,
        model: nn.Module,
        data: torch.Tensor,
        compute_ntk: bool = True,
        compute_feature_kernel: bool = True
    ):
        """
        Store initial kernels before training begins.
        
        Args:
            model: Model at initialization
            data: Training data
            compute_ntk: Whether to compute NTK (expensive)
            compute_feature_kernel: Whether to compute feature kernel (cheaper)
        """
        print("Initializing LazyRichTracker...")
        
        # Set consistent subsample indices
        n_total = len(data)
        if self.n_subsample < n_total:
            self.subsample_indices = torch.randperm(n_total)[:self.n_subsample]
        else:
            self.subsample_indices = torch.arange(n_total)
        
        # Store initial weight norms
        self.weight_norms_0 = compute_weight_norms(model)
        print(f"  Initial weight norm: {self.weight_norms_0['total']:.4f}")
        
        # Compute and store initial NTK
        if compute_ntk:
            print(f"  Computing initial NTK (n={len(self.subsample_indices)})...", end=' ', flush=True)
            data_sub = data[self.subsample_indices]
            
            if self.use_efficient_ntk:
                self.ntk_0 = compute_ntk_efficient(
                    model, data_sub, n_subsample=len(data_sub),
                    device=self.device, output_device=self.output_device
                )
            else:
                self.ntk_0 = compute_ntk_subsample(
                    model, data_sub, n_subsample=len(data_sub),
                    device=self.device, output_device=self.output_device
                )
            
            self.ntk_0_norm = torch.norm(self.ntk_0, p='fro').item()
            print(f"||K₀||_F = {self.ntk_0_norm:.4e}")
        
        # Compute and store initial feature kernel
        if compute_feature_kernel:
            print(f"  Computing initial feature kernel...", end=' ', flush=True)
            data_sub = data[self.subsample_indices]
            
            self.feature_kernel_0 = compute_feature_kernel(
                model, data_sub, layer_name=self.feature_layer,
                n_subsample=None,  # Already subsampled
                device=self.device, output_device=self.output_device
            )
            
            self.feature_kernel_0_norm = torch.norm(self.feature_kernel_0, p='fro').item()
            print(f"||K_features(0)||_F = {self.feature_kernel_0_norm:.4e}")
        
        print("  Initialization complete.")
    
    def compute_metrics(
        self,
        model: nn.Module,
        data: torch.Tensor,
        history: Dict,
        compute_ntk: bool = True,
        compute_feature_kernel_dist: bool = True
    ) -> Dict[str, float]:
        """
        Compute lazy-rich metrics at current epoch.
        
        Args:
            model: Current model state
            data: Training data
            history: Dictionary to append metrics to
            compute_ntk: Whether to compute NTK distance
            compute_feature_kernel_dist: Whether to compute feature kernel distance
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = {}
        
        # Get subsampled data (use consistent indices)
        if self.subsample_indices is not None:
            data_sub = data[self.subsample_indices]
        else:
            data_sub = data
        
        # Weight norms (always compute - it's cheap)
        weight_norms = compute_weight_norms(model)
        metrics['weight_norm_total'] = weight_norms['total']
        
        # Relative weight change from initialization
        if self.weight_norms_0 is not None:
            metrics['weight_norm_change'] = (
                weight_norms['total'] - self.weight_norms_0['total']
            ) / (self.weight_norms_0['total'] + 1e-10)
        
        # Store per-layer norms
        for key, value in weight_norms.items():
            if key != 'total':
                metrics[f'weight_norm_{key}'] = value
        
        # NTK distance
        if compute_ntk and self.ntk_0 is not None:
            if self.use_efficient_ntk:
                ntk_t = compute_ntk_efficient(
                    model, data_sub, n_subsample=len(data_sub),
                    device=self.device, output_device=self.output_device
                )
            else:
                ntk_t = compute_ntk_subsample(
                    model, data_sub, n_subsample=len(data_sub),
                    device=self.device, output_device=self.output_device
                )
            
            # Normalized distance: ||K_t - K_0||_F / ||K_0||_F
            ntk_diff = ntk_t - self.ntk_0
            ntk_distance = torch.norm(ntk_diff, p='fro').item() / (self.ntk_0_norm + 1e-10)
            metrics['ntk_distance'] = ntk_distance
            
            # Also track NTK norm evolution
            metrics['ntk_norm'] = torch.norm(ntk_t, p='fro').item()
        
        # Feature kernel distance
        if compute_feature_kernel_dist and self.feature_kernel_0 is not None:
            feature_kernel_t = compute_feature_kernel(
                model, data_sub, layer_name=self.feature_layer,
                n_subsample=None,
                device=self.device, output_device=self.output_device
            )
            
            # Normalized distance
            fk_diff = feature_kernel_t - self.feature_kernel_0
            fk_distance = torch.norm(fk_diff, p='fro').item() / (self.feature_kernel_0_norm + 1e-10)
            metrics['feature_kernel_distance'] = fk_distance
            
            # Track feature kernel norm
            metrics['feature_kernel_norm'] = torch.norm(feature_kernel_t, p='fro').item()
        
        # Append to history
        for key, value in metrics.items():
            if key not in history:
                history[key] = []
            history[key].append(value)
        
        return metrics


def detect_lazy_rich_transition(
    ntk_distances: List[float],
    epochs: List[int],
    threshold: float = 0.1,
    window: int = 5
) -> Optional[int]:
    """
    Detect the epoch where lazy→rich transition occurs.
    
    Looks for the point where NTK distance starts increasing rapidly,
    indicating the network is leaving the lazy regime.
    
    Args:
        ntk_distances: List of NTK distances over training
        epochs: Corresponding epoch numbers
        threshold: Threshold for detecting significant change
        window: Smoothing window size
        
    Returns:
        Transition epoch or None if not detected
    """
    if len(ntk_distances) < window * 2:
        return None
    
    # Smooth the distances
    distances = np.array(ntk_distances)
    smoothed = np.convolve(distances, np.ones(window)/window, mode='valid')
    
    # Compute rate of change
    diffs = np.diff(smoothed)
    
    # Find first point where change exceeds threshold
    for i, diff in enumerate(diffs):
        if diff > threshold:
            # Map back to original epoch
            epoch_idx = i + window // 2
            if epoch_idx < len(epochs):
                return epochs[epoch_idx]
    
    return None


def test_lazy_rich_tracker():
    """Test the LazyRichTracker on a toy example."""
    print("="*80)
    print("Testing Lazy-Rich Tracker")
    print("="*80)
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    # Create toy data
    n_samples = 50
    x = torch.randn(n_samples, 10)
    
    print("\n1. Testing weight norm computation...")
    norms = compute_weight_norms(model)
    print(f"   Total norm: {norms['total']:.4f}")
    print(f"   Number of layers tracked: {len(norms) - 1}")
    
    print("\n2. Testing feature kernel computation...")
    fk = compute_feature_kernel(model, x, n_subsample=20)
    print(f"   Feature kernel shape: {fk.shape}")
    print(f"   Feature kernel norm: {torch.norm(fk, p='fro').item():.4f}")
    
    print("\n3. Testing NTK computation (efficient)...")
    ntk = compute_ntk_efficient(model, x, n_subsample=20)
    print(f"   NTK shape: {ntk.shape}")
    print(f"   NTK norm: {torch.norm(ntk, p='fro').item():.4f}")
    
    print("\n4. Testing LazyRichTracker initialization...")
    tracker = LazyRichTracker(n_subsample=20, device='cpu', output_device='cpu')
    tracker.initialize(model, x, compute_ntk=True, compute_feature_kernel=True)
    
    print("\n5. Testing metric computation...")
    # Simulate some training (modify weights slightly)
    with torch.no_grad():
        for p in model.parameters():
            p.data += 0.1 * torch.randn_like(p)
    
    history = {}
    metrics = tracker.compute_metrics(model, x, history)
    
    print(f"   Weight norm change: {metrics.get('weight_norm_change', 0):.4f}")
    print(f"   NTK distance: {metrics.get('ntk_distance', 0):.4f}")
    print(f"   Feature kernel distance: {metrics.get('feature_kernel_distance', 0):.4f}")
    
    print("\n" + "="*80)
    print("✓ All Lazy-Rich Tracker tests passed!")
    print("="*80)


if __name__ == "__main__":
    test_lazy_rich_tracker()

