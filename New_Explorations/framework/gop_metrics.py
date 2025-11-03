"""
GOP Metrics: Comprehensive Metric Computation for Gradient Outer Products

This module computes various metrics from GOP matrices including:
- Eigenvalues and eigenvectors
- Trace, norms, rank, condition number
- All metrics required for analyzing grokking dynamics
"""

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import eigsh
from typing import Dict, Optional, Tuple
import logging
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GOPMetrics:
    """
    Computes comprehensive metrics from Gradient Outer Product matrices.
    """
    
    def __init__(
        self,
        top_k: int = 100,
        rank_threshold: float = 1e-6,
        use_sparse: bool = False
    ):
        """
        Initialize GOP metrics computer.
        
        Args:
            top_k: Number of top eigenvalues/eigenvectors to keep
            rank_threshold: Threshold for determining effective rank
            use_sparse: Whether to use sparse eigenvalue computation for large matrices
        """
        self.top_k = top_k
        self.rank_threshold = rank_threshold
        self.use_sparse = use_sparse
    
    def compute_eigenvalues(
        self,
        gop_matrix: np.ndarray,
        compute_vectors: bool = False,
        top_k_only: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Compute eigenvalues (and optionally eigenvectors) of GOP matrix.
        
        Since GOP = G ⊗ G^T is symmetric and positive semi-definite,
        we use specialized symmetric eigenvalue solvers.
        
        Args:
            gop_matrix: GOP matrix (M, M)
            compute_vectors: Whether to compute eigenvectors
            top_k_only: Whether to compute only top-k eigenvalues
            
        Returns:
            Tuple of (eigenvalues, eigenvectors)
            - eigenvalues: Sorted in descending order
            - eigenvectors: None if compute_vectors=False, otherwise matrix of eigenvectors
        """
        M = gop_matrix.shape[0]
        
        # For very large matrices or if requested, use sparse computation
        if (self.use_sparse or M > 10000) and top_k_only:
            logger.debug(f"Using sparse eigenvalue computation for matrix size {M}")
            try:
                # Compute top-k eigenvalues using sparse method
                k = min(self.top_k, M - 2)  # scipy requires k < M - 1
                
                if compute_vectors:
                    eigenvalues, eigenvectors = eigsh(
                        gop_matrix,
                        k=k,
                        which='LA',  # Largest Algebraic
                        return_eigenvectors=True
                    )
                else:
                    eigenvalues = eigsh(
                        gop_matrix,
                        k=k,
                        which='LA',
                        return_eigenvectors=False
                    )
                    eigenvectors = None
                
                # Sort in descending order
                idx = np.argsort(eigenvalues)[::-1]
                eigenvalues = eigenvalues[idx]
                if eigenvectors is not None:
                    eigenvectors = eigenvectors[:, idx]
                
                return eigenvalues, eigenvectors
                
            except Exception as e:
                logger.warning(f"Sparse eigenvalue computation failed: {e}, falling back to dense")
        
        # Dense computation
        logger.debug(f"Using dense eigenvalue computation for matrix size {M}")
        
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                
                if compute_vectors:
                    # Compute all eigenvalues and eigenvectors
                    eigenvalues, eigenvectors = linalg.eigh(gop_matrix)
                else:
                    # Compute only eigenvalues
                    eigenvalues = linalg.eigvalsh(gop_matrix)
                    eigenvectors = None
                
                # Sort in descending order
                idx = np.argsort(eigenvalues)[::-1]
                eigenvalues = eigenvalues[idx]
                if eigenvectors is not None:
                    eigenvectors = eigenvectors[:, idx]
                
                # If top_k_only, keep only top-k
                if top_k_only:
                    k = min(self.top_k, len(eigenvalues))
                    eigenvalues = eigenvalues[:k]
                    if eigenvectors is not None:
                        eigenvectors = eigenvectors[:, :k]
                
                return eigenvalues, eigenvectors
                
        except Exception as e:
            logger.error(f"Eigenvalue computation failed: {e}")
            # Return empty arrays as fallback
            return np.array([]), None
    
    def compute_trace(self, gop_matrix: np.ndarray) -> float:
        """
        Compute trace of GOP matrix.
        
        Trace = sum of diagonal elements = sum of eigenvalues
        
        Args:
            gop_matrix: GOP matrix (M, M)
            
        Returns:
            Trace value
        """
        return np.trace(gop_matrix)
    
    def compute_frobenius_norm(self, gop_matrix: np.ndarray) -> float:
        """
        Compute Frobenius norm of GOP matrix.
        
        Frobenius norm = sqrt(sum of squared elements)
        
        Args:
            gop_matrix: GOP matrix (M, M)
            
        Returns:
            Frobenius norm
        """
        return np.linalg.norm(gop_matrix, 'fro')
    
    def compute_spectral_norm(
        self,
        gop_matrix: Optional[np.ndarray] = None,
        eigenvalues: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute spectral norm (largest singular value = largest eigenvalue for symmetric matrix).
        
        Args:
            gop_matrix: GOP matrix (M, M)
            eigenvalues: Pre-computed eigenvalues (if available)
            
        Returns:
            Spectral norm
        """
        if eigenvalues is not None and len(eigenvalues) > 0:
            return float(np.abs(eigenvalues[0]))
        
        if gop_matrix is not None:
            # For symmetric matrix, spectral norm = largest absolute eigenvalue
            eigenvalues, _ = self.compute_eigenvalues(gop_matrix, top_k_only=True)
            if len(eigenvalues) > 0:
                return float(np.abs(eigenvalues[0]))
        
        return 0.0
    
    def compute_nuclear_norm(
        self,
        gop_matrix: Optional[np.ndarray] = None,
        eigenvalues: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute nuclear norm (sum of singular values = sum of absolute eigenvalues).
        
        Args:
            gop_matrix: GOP matrix (M, M)
            eigenvalues: Pre-computed eigenvalues (if available)
            
        Returns:
            Nuclear norm
        """
        if eigenvalues is not None:
            return float(np.sum(np.abs(eigenvalues)))
        
        if gop_matrix is not None:
            eigenvalues, _ = self.compute_eigenvalues(gop_matrix, top_k_only=False)
            return float(np.sum(np.abs(eigenvalues)))
        
        return 0.0
    
    def compute_rank(
        self,
        gop_matrix: Optional[np.ndarray] = None,
        eigenvalues: Optional[np.ndarray] = None
    ) -> int:
        """
        Compute effective rank (number of eigenvalues above threshold).
        
        Args:
            gop_matrix: GOP matrix (M, M)
            eigenvalues: Pre-computed eigenvalues (if available)
            
        Returns:
            Effective rank
        """
        if eigenvalues is None:
            if gop_matrix is not None:
                eigenvalues, _ = self.compute_eigenvalues(gop_matrix, top_k_only=False)
            else:
                return 0
        
        # Count eigenvalues above threshold
        max_eigenvalue = np.max(np.abs(eigenvalues)) if len(eigenvalues) > 0 else 0
        threshold = max_eigenvalue * self.rank_threshold
        rank = np.sum(np.abs(eigenvalues) > threshold)
        
        return int(rank)
    
    def compute_condition_number(
        self,
        gop_matrix: Optional[np.ndarray] = None,
        eigenvalues: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute condition number (ratio of largest to smallest non-zero eigenvalue).
        
        Args:
            gop_matrix: GOP matrix (M, M)
            eigenvalues: Pre-computed eigenvalues (if available)
            
        Returns:
            Condition number
        """
        if eigenvalues is None:
            if gop_matrix is not None:
                eigenvalues, _ = self.compute_eigenvalues(gop_matrix, top_k_only=False)
            else:
                return np.inf
        
        if len(eigenvalues) == 0:
            return np.inf
        
        # Get non-zero eigenvalues
        abs_eigenvalues = np.abs(eigenvalues)
        max_eigenvalue = np.max(abs_eigenvalues)
        threshold = max_eigenvalue * self.rank_threshold
        nonzero_eigenvalues = abs_eigenvalues[abs_eigenvalues > threshold]
        
        if len(nonzero_eigenvalues) == 0:
            return np.inf
        
        min_nonzero = np.min(nonzero_eigenvalues)
        max_eigenvalue = np.max(nonzero_eigenvalues)
        
        if min_nonzero > 0:
            return float(max_eigenvalue / min_nonzero)
        else:
            return np.inf
    
    def compute_determinant(
        self,
        gop_matrix: Optional[np.ndarray] = None,
        eigenvalues: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute determinant (product of eigenvalues).
        
        Note: For large matrices, determinant can be numerically unstable.
        
        Args:
            gop_matrix: GOP matrix (M, M)
            eigenvalues: Pre-computed eigenvalues (if available)
            
        Returns:
            Determinant
        """
        if eigenvalues is not None:
            # Use log-sum-exp trick for numerical stability
            log_abs_eigenvalues = np.log(np.abs(eigenvalues) + 1e-100)
            log_det = np.sum(log_abs_eigenvalues)
            # Check for overflow
            if log_det > 700:  # exp(700) is near max float
                logger.warning("Determinant computation overflow")
                return np.inf
            return float(np.exp(log_det))
        
        if gop_matrix is not None:
            try:
                sign, logdet = np.linalg.slogdet(gop_matrix)
                if logdet > 700:
                    return np.inf
                return float(sign * np.exp(logdet))
            except:
                eigenvalues, _ = self.compute_eigenvalues(gop_matrix, top_k_only=False)
                return self.compute_determinant(eigenvalues=eigenvalues)
        
        return 0.0
    
    def compute_all_metrics(
        self,
        gop_matrix: np.ndarray,
        compute_eigenvectors: bool = True
    ) -> Dict[str, any]:
        """
        Compute all available metrics from a GOP matrix.
        
        Args:
            gop_matrix: GOP matrix (M, M)
            compute_eigenvectors: Whether to compute eigenvectors
            
        Returns:
            Dictionary containing all computed metrics
        """
        logger.debug(f"Computing all metrics for GOP matrix of shape {gop_matrix.shape}")
        
        # Compute eigenvalues first (used by many metrics)
        eigenvalues_full, eigenvectors_full = self.compute_eigenvalues(
            gop_matrix,
            compute_vectors=False,
            top_k_only=False
        )
        
        eigenvalues_top_k, eigenvectors_top_k = self.compute_eigenvalues(
            gop_matrix,
            compute_vectors=compute_eigenvectors,
            top_k_only=True
        )
        
        # Compute all metrics
        metrics = {
            # Eigenvalue data
            'eigenvalues_full': eigenvalues_full,
            'eigenvalues_top_k': eigenvalues_top_k,
            'eigenvectors_top_k': eigenvectors_top_k,
            
            # Scalar metrics
            'trace': self.compute_trace(gop_matrix),
            'frobenius_norm': self.compute_frobenius_norm(gop_matrix),
            'spectral_norm': self.compute_spectral_norm(eigenvalues=eigenvalues_top_k),
            'nuclear_norm': self.compute_nuclear_norm(eigenvalues=eigenvalues_full),
            'rank': self.compute_rank(eigenvalues=eigenvalues_full),
            'condition_number': self.compute_condition_number(eigenvalues=eigenvalues_full),
            'determinant': self.compute_determinant(eigenvalues=eigenvalues_full),
            
            # Additional statistics
            'eigenvalue_max': float(eigenvalues_full[0]) if len(eigenvalues_full) > 0 else 0.0,
            'eigenvalue_min': float(eigenvalues_full[-1]) if len(eigenvalues_full) > 0 else 0.0,
            'eigenvalue_mean': float(np.mean(eigenvalues_full)) if len(eigenvalues_full) > 0 else 0.0,
            'eigenvalue_std': float(np.std(eigenvalues_full)) if len(eigenvalues_full) > 0 else 0.0,
            
            # Top eigenvalue cumulative explained variance
            'top_k_cumulative_variance': (
                float(np.sum(eigenvalues_top_k) / np.sum(eigenvalues_full))
                if len(eigenvalues_full) > 0 and np.sum(eigenvalues_full) > 0
                else 0.0
            ),
        }
        
        logger.debug(f"Computed metrics: trace={metrics['trace']:.2e}, "
                    f"rank={metrics['rank']}, condition={metrics['condition_number']:.2e}")
        
        return metrics
    
    def compute_layer_metrics(
        self,
        layer_gops: Dict[str, np.ndarray],
        compute_eigenvectors: bool = False
    ) -> Dict[str, Dict[str, any]]:
        """
        Compute metrics for all layer GOPs.
        
        Args:
            layer_gops: Dictionary mapping layer names to GOP matrices
            compute_eigenvectors: Whether to compute eigenvectors
            
        Returns:
            Dictionary mapping layer names to their metrics
        """
        layer_metrics = {}
        
        for layer_name, gop_matrix in layer_gops.items():
            try:
                metrics = self.compute_all_metrics(gop_matrix, compute_eigenvectors)
                layer_metrics[layer_name] = metrics
            except Exception as e:
                logger.error(f"Error computing metrics for layer {layer_name}: {e}")
                continue
        
        return layer_metrics

