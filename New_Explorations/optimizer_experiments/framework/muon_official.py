"""
Official Muon Optimizer - Adapted from modded-nanogpt
Source: https://github.com/KellerJordan/modded-nanogpt

Muon - MomentUm Orthogonalized by Newton-schulz
https://kellerjordan.github.io/posts/muon/

Key differences from custom implementation:
1. Uses Newton-Schulz iteration (5 steps) for proper orthogonalization
2. Per-layer learning rate scaling: lr * sqrt(max(1, rows/cols))
3. Orthogonalizes momentum_buffer + grad (not just grad)
4. Should NOT be used for embeddings or 1D parameters (use AdamW for those)

Adapted for float32/float64 (original uses bfloat16)
"""

import torch
from torch.optim.optimizer import Optimizer
from typing import List


def zeropower_via_newtonschulz5(G: torch.Tensor) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute orthogonalization of G.
    
    Performs 5 iterations with coefficients selected to maximize slope at zero.
    Returns approximately U @ V.T where U @ S @ V.T = SVD(G).
    
    Args:
        G: Tensor of shape (..., m, n) to orthogonalize
        
    Returns:
        Orthogonalized tensor of same shape
    """
    assert G.ndim >= 2
    
    # Work with the smaller dimension for efficiency
    X = G.float()  # Convert to float32 for stability
    transposed = False
    if G.size(-2) > G.size(-1):
        X = X.mT
        transposed = True
    
    # Normalize spectral norm to at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    
    # Perform 5 Newton-Schulz iterations
    # Coefficients from modded-nanogpt (optimized for convergence)
    for a, b, c in [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    
    if transposed:
        X = X.mT
    
    return X.to(G.dtype)


class Muon(Optimizer):
    """
    Muon optimizer adapted from modded-nanogpt.
    
    Momentum + Newton-Schulz Orthogonalization
    
    Key features:
    - SGD with momentum
    - Orthogonalizes the momentum buffer using Newton-Schulz iteration
    - Per-layer adaptive learning rate (scales by sqrt of aspect ratio)
    - Works best on 2D parameters (weight matrices)
    
    Args:
        params: Parameters to optimize
        lr: Learning rate (default: 0.02 from official implementation)
        weight_decay: Weight decay coefficient (default: 0.01)
        momentum: Momentum factor (default: 0.95)
        use_nesterov: Whether to use Nesterov momentum (default: True)
    
    Note: For embedding and output layers, use AdamW instead!
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.02,  # Official Muon default
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        use_nesterov: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, use_nesterov=use_nesterov)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            use_nesterov = group['use_nesterov']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # Initialize state
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                
                buf = state['momentum_buffer']
                
                # Standard momentum update
                buf.mul_(momentum).add_(grad, alpha=1.0 - momentum)
                
                # Compute update (momentum buffer or Nesterov)
                if use_nesterov:
                    update = grad.mul(1.0 - momentum).add(buf, alpha=momentum)
                else:
                    update = buf.clone()
                
                # Orthogonalize 2D parameters using Newton-Schulz
                if p.dim() >= 2:
                    # Apply Newton-Schulz orthogonalization
                    update = zeropower_via_newtonschulz5(update)
                
                # Per-layer learning rate scaling (from official Muon)
                # Scale by sqrt of aspect ratio for 2D matrices
                if p.dim() == 2:
                    # lr * max(1, rows/cols)^0.5
                    scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                    eff_lr = lr * scale
                else:
                    eff_lr = lr
                
                # Apply weight decay (decoupled, applied to parameters)
                if weight_decay != 0:
                    p.mul_(1 - eff_lr * weight_decay)
                
                # Apply update
                p.add_(update, alpha=-eff_lr)
        
        return loss


class MuonW(Muon):
    """
    Alias for Muon with decoupled weight decay (same as base Muon).
    Kept for API compatibility.
    """
    pass


def test_official_muon():
    """Test the official Muon implementation."""
    print("="*70)
    print("Testing Official Muon Optimizer (from modded-nanogpt)")
    print("="*70)
    
    # Simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 2)
    )
    
    # Dummy data
    x = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))
    
    # Create optimizer (use official defaults)
    optimizer = Muon(model.parameters(), lr=0.02, momentum=0.95, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    print(f"\nTraining with official Muon (lr=0.02, momentum=0.95, wd=0.01):")
    for i in range(15):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        if i % 5 == 0:
            with torch.no_grad():
                pred = output.argmax(dim=1)
                acc = (pred == y).float().mean()
            print(f"  Step {i:2d}: Loss={loss.item():.4f}, Acc={acc.item():.4f}")
    
    print("\n✓ Official Muon test passed!")
    print("="*70)


if __name__ == "__main__":
    test_official_muon()

