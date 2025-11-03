"""
GOP Tracker: Gradient Outer Product Computation

This module handles the computation of gradient outer products for neural networks.
It supports both full-model GOPs and per-layer GOPs.
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, List
import logging
from collections import OrderedDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GOPTracker:
    """
    Tracks and computes Gradient Outer Products during training.
    
    The gradient outer product is defined as G ⊗ G^T where G is the 
    flattened gradient vector of all (or per-layer) parameters.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        compute_full: bool = True,
        compute_per_layer: bool = True,
        use_gpu_for_gop: bool = True,
        chunk_size: Optional[int] = None
    ):
        """
        Initialize GOP tracker.
        
        Args:
            model: PyTorch model to track
            compute_full: Whether to compute full model GOP
            compute_per_layer: Whether to compute per-layer GOPs
            use_gpu_for_gop: Whether to use GPU for GOP computation
            chunk_size: Chunk size for memory-efficient GOP computation
        """
        self.model = model
        self.compute_full = compute_full
        self.compute_per_layer = compute_per_layer
        self.use_gpu_for_gop = use_gpu_for_gop
        self.chunk_size = chunk_size
        
        # Count total parameters
        self.total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Tracking model with {self.total_params:,} parameters")
        
        # Store layer information
        self.layer_info = self._get_layer_info()
        logger.info(f"Identified {len(self.layer_info)} layers")
        
    def _get_layer_info(self) -> OrderedDict:
        """
        Get information about each layer in the model.
        
        Returns:
            OrderedDict mapping layer names to parameter info
        """
        layer_info = OrderedDict()
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Extract layer name (e.g., "layer1.weight" -> "layer1")
                layer_name = name.split('.')[0] if '.' in name else name
                
                if layer_name not in layer_info:
                    layer_info[layer_name] = {
                        'params': [],
                        'total_params': 0
                    }
                
                layer_info[layer_name]['params'].append((name, param))
                layer_info[layer_name]['total_params'] += param.numel()
        
        return layer_info
    
    def _extract_gradients(
        self,
        parameters: List[Tuple[str, torch.nn.Parameter]]
    ) -> torch.Tensor:
        """
        Extract and flatten gradients from parameters.
        
        Args:
            parameters: List of (name, parameter) tuples
            
        Returns:
            Flattened gradient vector
        """
        gradients = []
        
        for name, param in parameters:
            if param.grad is not None:
                gradients.append(param.grad.detach().flatten())
            else:
                # If no gradient, use zeros
                logger.warning(f"Parameter {name} has no gradient, using zeros")
                gradients.append(torch.zeros(param.numel(), device=param.device))
        
        if not gradients:
            raise ValueError("No gradients found in parameters")
        
        return torch.cat(gradients)
    
    def _compute_gop_gpu(self, gradient: torch.Tensor) -> torch.Tensor:
        """
        Compute GOP on GPU using torch.
        
        Args:
            gradient: Gradient vector (M,)
            
        Returns:
            GOP matrix (M, M)
        """
        # Ensure gradient is on GPU
        if not gradient.is_cuda and torch.cuda.is_available():
            gradient = gradient.cuda()
        
        # Compute outer product: g ⊗ g^T
        gop = torch.outer(gradient, gradient)
        
        return gop
    
    def _compute_gop_chunked(
        self,
        gradient: torch.Tensor,
        chunk_size: int
    ) -> torch.Tensor:
        """
        Compute GOP in chunks to save memory.
        
        Args:
            gradient: Gradient vector (M,)
            chunk_size: Size of chunks
            
        Returns:
            GOP matrix (M, M)
        """
        M = gradient.shape[0]
        device = gradient.device
        
        # Initialize GOP matrix
        gop = torch.zeros((M, M), device=device, dtype=gradient.dtype)
        
        # Compute in chunks
        for i in range(0, M, chunk_size):
            end_i = min(i + chunk_size, M)
            chunk_i = gradient[i:end_i]
            
            for j in range(0, M, chunk_size):
                end_j = min(j + chunk_size, M)
                chunk_j = gradient[j:end_j]
                
                gop[i:end_i, j:end_j] = torch.outer(chunk_i, chunk_j)
        
        return gop
    
    def compute_full_gop(self) -> Optional[np.ndarray]:
        """
        Compute full model gradient outer product.
        
        Returns:
            GOP matrix as numpy array, or None if computation disabled
        """
        if not self.compute_full:
            return None
        
        # Extract all gradients
        all_params = [(name, param) for name, param in self.model.named_parameters() 
                      if param.requires_grad]
        
        gradient = self._extract_gradients(all_params)
        
        logger.debug(f"Computing full GOP for {gradient.shape[0]:,} parameters")
        
        # Compute GOP
        if self.chunk_size is not None:
            gop = self._compute_gop_chunked(gradient, self.chunk_size)
        elif self.use_gpu_for_gop and torch.cuda.is_available():
            gop = self._compute_gop_gpu(gradient)
        else:
            # CPU computation
            gradient_np = gradient.cpu().numpy()
            gop_np = np.outer(gradient_np, gradient_np)
            return gop_np
        
        # Convert to numpy
        gop_np = gop.cpu().numpy()
        
        return gop_np
    
    def compute_layer_gops(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Compute per-layer gradient outer products.
        
        Returns:
            Dictionary mapping layer names to GOP matrices, or None if disabled
        """
        if not self.compute_per_layer:
            return None
        
        layer_gops = {}
        
        for layer_name, layer_data in self.layer_info.items():
            try:
                # Extract gradients for this layer
                gradient = self._extract_gradients(layer_data['params'])
                
                logger.debug(f"Computing GOP for layer {layer_name} "
                           f"({gradient.shape[0]:,} parameters)")
                
                # Compute GOP
                if self.use_gpu_for_gop and torch.cuda.is_available():
                    gop = self._compute_gop_gpu(gradient)
                    gop_np = gop.cpu().numpy()
                else:
                    gradient_np = gradient.cpu().numpy()
                    gop_np = np.outer(gradient_np, gradient_np)
                
                layer_gops[layer_name] = gop_np
                
            except Exception as e:
                logger.error(f"Error computing GOP for layer {layer_name}: {e}")
                continue
        
        return layer_gops
    
    def compute_all(self) -> Dict[str, any]:
        """
        Compute all GOP data (full and per-layer).
        
        Returns:
            Dictionary containing:
                - 'full_gop': Full model GOP matrix or None
                - 'layer_gops': Dict of per-layer GOP matrices or None
                - 'metadata': Information about computation
        """
        result = {
            'full_gop': None,
            'layer_gops': None,
            'metadata': {
                'total_params': self.total_params,
                'num_layers': len(self.layer_info),
                'layer_names': list(self.layer_info.keys())
            }
        }
        
        # Compute full GOP
        if self.compute_full:
            try:
                result['full_gop'] = self.compute_full_gop()
                if result['full_gop'] is not None:
                    logger.info(f"Computed full GOP: shape {result['full_gop'].shape}")
            except Exception as e:
                logger.error(f"Error computing full GOP: {e}")
        
        # Compute per-layer GOPs
        if self.compute_per_layer:
            try:
                result['layer_gops'] = self.compute_layer_gops()
                if result['layer_gops'] is not None:
                    logger.info(f"Computed {len(result['layer_gops'])} layer GOPs")
            except Exception as e:
                logger.error(f"Error computing layer GOPs: {e}")
        
        return result
    
    def estimate_memory_usage(self) -> Dict[str, float]:
        """
        Estimate memory usage for GOP computation.
        
        Returns:
            Dictionary with memory estimates in GB
        """
        bytes_per_float32 = 4
        bytes_per_gb = 1024 ** 3
        
        full_gop_memory = (self.total_params ** 2 * bytes_per_float32) / bytes_per_gb
        
        layer_gop_memory = 0
        for layer_data in self.layer_info.values():
            n_params = layer_data['total_params']
            layer_gop_memory += (n_params ** 2 * bytes_per_float32) / bytes_per_gb
        
        return {
            'full_gop_gb': full_gop_memory,
            'layer_gops_gb': layer_gop_memory,
            'total_gb': full_gop_memory + layer_gop_memory
        }
    
    def get_layer_sizes(self) -> Dict[str, int]:
        """
        Get the number of parameters in each layer.
        
        Returns:
            Dictionary mapping layer names to parameter counts
        """
        return {name: data['total_params'] 
                for name, data in self.layer_info.items()}

