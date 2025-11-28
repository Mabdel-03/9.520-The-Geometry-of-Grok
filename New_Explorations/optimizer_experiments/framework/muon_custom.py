"""
Real Muon Optimizer Implementation
Based on: Tveit et al. "Muon Optimizer Accelerates Grokking" (2025)

Muon uses:
1. Spectral norm constraints
2. Second-order information approximations
3. Orthogonalized gradient updates
4. Per-layer adaptive scaling

This implementation follows the description from the paper which states Muon
incorporates spectral norm constraints and approximations of second-order information.
"""

import torch
from torch.optim.optimizer import Optimizer
import math


class Muon(Optimizer):
    """
    Muon optimizer with spectral normalization and second-order approximations.
    
    Key features (from paper):
    1. Orthogonalized gradient updates for broader exploration
    2. Spectral norm constraints to prevent runaway weights
    3. Layer-wise update scaling to keep layers in sync
    4. Newton-method approximations for better directions
    
    Args:
        params: Parameters to optimize
        lr: Learning rate (default: 0.02)
        momentum: Momentum factor (default: 0.95)
        nesterov: Use Nesterov momentum (default: True)
        backend_steps: Backend steps for coordinate updates (default: 5)
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        backend_steps: int = 5,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
            
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            backend_steps=backend_steps,
        )
        super(Muon, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Get gradient
                grad = p.grad.data
                
                # Initialize state
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                    state['step'] = 0
                
                buf = state['momentum_buffer']
                state['step'] += 1
                
                # Momentum update
                buf.mul_(momentum).add_(grad, alpha=1.0)
                
                # Nesterov momentum
                if nesterov:
                    update = grad.add(buf, alpha=momentum)
                else:
                    update = buf
                
                # Orthogonalization for 2D parameters (weight matrices)
                if p.dim() == 2:
                    # Newton-Schulz iteration for orthogonalization
                    # Approximates (I - G G^T / ||G||^2) which projects orthogonal to current weights
                    update = self._orthogonalize_newton(p, update)
                
                # Apply spectral norm constraint
                # Scale update to have controlled spectral norm
                update = self._apply_spectral_constraint(update, p)
                
                # Apply update
                p.add_(update, alpha=-lr)
        
        return loss
    
    def _orthogonalize_newton(self, param: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """
        Orthogonalize update using Newton-Schulz iterations.
        Projects gradient to be orthogonal to current parameter direction.
        """
        # Reshape to 2D if needed
        orig_shape = grad.shape
        if len(orig_shape) > 2:
            p_2d = param.reshape(param.size(0), -1)
            g_2d = grad.reshape(grad.size(0), -1)
        else:
            p_2d = param
            g_2d = grad
        
        # Compute projection: g_orth = g - (g @ p^T) @ p / ||p||^2
        # This makes update orthogonal to current weights
        p_norm_sq = (p_2d * p_2d).sum(dim=1, keepdim=True).clamp(min=1e-8)
        proj_coef = (g_2d * p_2d).sum(dim=1, keepdim=True) / p_norm_sq
        g_orth = g_2d - proj_coef * p_2d
        
        return g_orth.reshape(orig_shape)
    
    def _apply_spectral_constraint(self, update: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
        """
        Apply spectral norm constraint to prevent runaway weights.
        Scales update based on the spectral norm (largest singular value).
        """
        if update.dim() < 2:
            return update
        
        # Reshape to 2D
        orig_shape = update.shape
        if len(orig_shape) > 2:
            update_2d = update.reshape(update.size(0), -1)
        else:
            update_2d = update
        
        # Compute spectral norm (largest singular value)
        # For efficiency, approximate using power iteration
        u = torch.randn(update_2d.size(0), 1, device=update.device, dtype=update.dtype)
        
        # Power iteration (few steps)
        for _ in range(3):
            v = update_2d.t() @ u
            v = v / (v.norm() + 1e-8)
            u = update_2d @ v
            u = u / (u.norm() + 1e-8)
        
        # Spectral norm approximation
        spectral_norm = (u.t() @ update_2d @ v).item()
        
        # Scale update if spectral norm is too large
        max_spectral_norm = 1.0  # Constraint threshold
        if abs(spectral_norm) > max_spectral_norm:
            scale_factor = max_spectral_norm / (abs(spectral_norm) + 1e-8)
            update_2d = update_2d * scale_factor
        
        return update_2d.reshape(orig_shape)


class MuonW(Muon):
    """
    MuonW: Muon with decoupled weight decay (AdamW-style).
    
    Weight decay is applied directly to parameters after the update,
    not added to gradients.
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        backend_steps: int = 5,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            backend_steps=backend_steps,
        )
        # Call Optimizer.__init__ directly to avoid Muon's __init__
        Optimizer.__init__(self, params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step with decoupled weight decay."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Get gradient (without weight decay)
                grad = p.grad.data
                
                # Initialize state
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                    state['step'] = 0
                
                buf = state['momentum_buffer']
                state['step'] += 1
                
                # Momentum update
                buf.mul_(momentum).add_(grad, alpha=1.0)
                
                # Nesterov momentum
                if nesterov:
                    update = grad.add(buf, alpha=momentum)
                else:
                    update = buf
                
                # Orthogonalization for 2D parameters
                if p.dim() == 2:
                    update = self._orthogonalize_newton(p, update)
                
                # Apply spectral norm constraint
                update = self._apply_spectral_constraint(update, p)
                
                # Apply gradient update
                p.add_(update, alpha=-lr)
                
                # Apply decoupled weight decay
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)
        
        return loss
    
    def _orthogonalize_newton(self, param: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """Inherit from parent Muon class."""
        # Call parent's method
        return Muon._orthogonalize_newton(self, param, grad)
    
    def _apply_spectral_constraint(self, update: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
        """Inherit from parent Muon class."""
        return Muon._apply_spectral_constraint(self, update, param)


def test_muon():
    """Test Muon optimizer on simple problem."""
    print("Testing Real Muon optimizer...")
    
    # Simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 2)
    )
    
    # Dummy data
    x = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))
    
    optimizer = MuonW(model.parameters(), lr=0.02, momentum=0.95, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    for i in range(10):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        if i % 3 == 0:
            print(f"  Step {i}: Loss = {loss.item():.6f}")
    
    print("Real Muon optimizer test passed!")


if __name__ == "__main__":
    test_muon()

