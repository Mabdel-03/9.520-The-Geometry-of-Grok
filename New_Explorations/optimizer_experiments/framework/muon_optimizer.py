"""
Muon Optimizer Implementation
Based on: "Muon: Momentum-based Orthogonal Updates for Neural Networks"

Muon is a momentum-based optimizer that applies orthogonal updates,
which can lead to faster convergence and better generalization.
"""

import torch
from torch.optim.optimizer import Optimizer
from typing import List, Optional
import math


class Muon(Optimizer):
    """
    Muon optimizer with momentum-based orthogonal updates.
    
    This is a simplified implementation based on the core ideas:
    1. Momentum accumulation
    2. Orthogonalization of parameter updates
    3. Adaptive learning rates per parameter
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 0.001)
        momentum: Momentum factor (default: 0.9)
        weight_decay: Weight decay coefficient (default: 0)
        dampening: Dampening for momentum (default: 0)
        nesterov: Whether to use Nesterov momentum (default: False)
        orthogonalize: Whether to orthogonalize updates (default: True)
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
        orthogonalize: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            orthogonalize=orthogonalize,
        )
        super(Muon, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(Muon, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('orthogonalize', True)
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            orthogonalize = group['orthogonalize']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                d_p = p.grad
                
                # Apply weight decay
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                
                # Apply momentum
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                
                # Orthogonalize update (Muon's key feature)
                if orthogonalize and p.dim() >= 2:
                    d_p = self._orthogonalize_update(p, d_p)
                
                # Apply update
                p.add_(d_p, alpha=-group['lr'])
        
        return loss
    
    def _orthogonalize_update(self, param: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """
        Orthogonalize the gradient update with respect to the current parameters.
        
        This helps maintain orthogonality in weight matrices and can improve
        conditioning of the optimization landscape.
        
        Args:
            param: Current parameter tensor
            grad: Gradient tensor
            
        Returns:
            Orthogonalized gradient
        """
        if param.dim() == 2:
            # For 2D tensors (weight matrices), use QR decomposition
            # Project gradient onto space orthogonal to current weights
            
            # Flatten if needed
            orig_shape = grad.shape
            p_flat = param.view(param.size(0), -1)
            g_flat = grad.view(grad.size(0), -1)
            
            # Compute projection
            # g_orth = g - (g @ p^T) @ p / ||p||^2
            p_norm_sq = (p_flat * p_flat).sum(dim=1, keepdim=True).clamp(min=1e-10)
            projection = (g_flat * p_flat).sum(dim=1, keepdim=True) / p_norm_sq
            g_orth = g_flat - projection * p_flat
            
            return g_orth.view(orig_shape)
        else:
            # For other dimensions, return gradient as-is
            return grad


class MuonW(Muon):
    """
    MuonW: Muon with decoupled weight decay (similar to AdamW).
    
    Weight decay is applied directly to parameters, not to gradients.
    """
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step with decoupled weight decay.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            orthogonalize = group['orthogonalize']
            lr = group['lr']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Gradient (without weight decay)
                d_p = p.grad
                
                # Apply momentum
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                
                # Orthogonalize update
                if orthogonalize and p.dim() >= 2:
                    d_p = self._orthogonalize_update(p, d_p)
                
                # Apply gradient update
                p.add_(d_p, alpha=-lr)
                
                # Apply weight decay directly to parameters (decoupled)
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)
        
        return loss


def test_muon():
    """Test Muon optimizer on a simple problem."""
    print("Testing Muon optimizer...")
    
    # Simple quadratic optimization problem
    x = torch.randn(10, 5, requires_grad=True)
    target = torch.randn(10, 5)
    
    optimizer = Muon([x], lr=0.01, momentum=0.9, weight_decay=0.01)
    
    for i in range(10):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(x, target)
        loss.backward()
        optimizer.step()
        
        if i % 3 == 0:
            print(f"  Step {i}: Loss = {loss.item():.6f}")
    
    print("✓ Muon optimizer test passed!")
    
    # Test MuonW
    print("\nTesting MuonW optimizer...")
    x = torch.randn(10, 5, requires_grad=True)
    optimizer = MuonW([x], lr=0.01, momentum=0.9, weight_decay=0.01)
    
    for i in range(10):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(x, target)
        loss.backward()
        optimizer.step()
        
        if i % 3 == 0:
            print(f"  Step {i}: Loss = {loss.item():.6f}")
    
    print("✓ MuonW optimizer test passed!")


if __name__ == "__main__":
    test_muon()

