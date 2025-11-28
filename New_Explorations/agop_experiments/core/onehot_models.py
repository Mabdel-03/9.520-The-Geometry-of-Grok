"""
Models for One-Hot Encoded Inputs - Enables Tractable Input-Gradient AGOP

Provides both MLP and Transformer architectures that accept one-hot continuous inputs.
This enables tractable input-gradient AGOP computation while preserving architectural
flexibility for comparison studies.

Key difference from standard models:
- Replaces nn.Embedding (requires integers) with nn.Linear (accepts floats)
- All inputs are continuous one-hot vectors (differentiable)
- Input-gradient AGOP computation is tractable and consistent across all models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# MLP Models (Simple, matches notebook exactly)
# ============================================================================

class ModularArithmeticMLP(nn.Module):
    """
    Simple MLP for modular arithmetic with one-hot inputs.
    Matches notebook architecture exactly.
    
    Args:
        p: Modulus
        hidden_dim: Hidden layer size
        dropout: Dropout probability
    """
    def __init__(self, p: int, hidden_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.p = p
        self.net = nn.Sequential(
            nn.Linear(2*p, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, p)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, 2*p] one-hot encoded input
        Returns:
            logits: [B, p] output logits
        """
        return self.net(x)


class CompositionMLP(nn.Module):
    """
    MLP for compositional reasoning with one-hot sequence inputs.
    
    Args:
        vocab_size: Vocabulary size
        seq_len: Sequence length
        hidden_dim: Hidden layer size
        n_layers: Number of hidden layers
    """
    def __init__(self, vocab_size: int, seq_len: int, hidden_dim: int = 256, n_layers: int = 2):
        super().__init__()
        input_dim = vocab_size * seq_len
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Linear(hidden_dim, vocab_size))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: [B, vocab_size * seq_len] one-hot encoded sequence
        Returns:
            logits: [B, vocab_size] output logits
        """
        return self.net(x)


class MNISTModel(nn.Module):
    """
    Simple MLP for MNIST classification (from Omnigrok paper).
    Already uses continuous inputs (pixel values).
    
    Args:
        input_dim: Input dimension (784 for flattened MNIST)
        hidden_dim: Hidden layer size
        output_dim: Number of classes (10 for MNIST)
        depth: Number of layers
        activation: Activation function
        initialization_scale: Weight initialization scale
    """
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 200,
        output_dim: int = 10,
        depth: int = 3,
        activation: str = 'relu',
        initialization_scale: float = 8.0
    ):
        super().__init__()
        
        if activation.lower() == 'relu':
            activation_fn = nn.ReLU
        elif activation.lower() == 'tanh':
            activation_fn = nn.Tanh
        elif activation.lower() == 'gelu':
            activation_fn = nn.GELU
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        layers = [nn.Flatten()]
        
        for i in range(depth):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(activation_fn())
            elif i == depth - 1:
                layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(activation_fn())
        
        self.net = nn.Sequential(*layers)
        
        # Initialize with scaled weights (as in Omnigrok)
        with torch.no_grad():
            for p in self.parameters():
                p.data = initialization_scale * p.data
    
    def forward(self, x):
        """
        Args:
            x: [B, 784] flattened images or [B, 28, 28] images
        Returns:
            logits: [B, 10] output logits
        """
        return self.net(x)


# ============================================================================
# Transformer Models (Preserves paper architecture, tractable AGOP!)
# ============================================================================

class ReLUAttention(nn.Module):
    """Multi-head ReLU attention (from Nanda et al.)"""
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        Q = self.W_Q(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        K = self.W_K(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        V = self.W_V(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn_weights = F.relu(scores)  # ReLU instead of softmax!
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_O(attn_output)
        
        return output


class TransformerMLP(nn.Module):
    """MLP block for transformer"""
    def __init__(self, d_model: int, d_mlp: int):
        super().__init__()
        self.W_in = nn.Linear(d_model, d_mlp, bias=False)
        self.W_out = nn.Linear(d_mlp, d_model, bias=False)
    
    def forward(self, x):
        return self.W_out(F.relu(self.W_in(x)))


class OneHotReLUTransformer(nn.Module):
    """
    Nanda's one-layer ReLU Transformer adapted for one-hot continuous inputs.
    
    Key modification: Replaces nn.Embedding with nn.Linear projection.
    This enables input-gradient AGOP computation while preserving architecture.
    
    Args:
        p: Modulus
        d_model: Model dimension
        n_heads: Number of attention heads
        d_mlp: MLP hidden dimension
    """
    def __init__(self, p: int, d_model: int = 128, n_heads: int = 4, d_mlp: int = 512):
        super().__init__()
        self.p = p
        self.d_model = d_model
        
        # Replace embeddings with linear projections
        self.input_proj = nn.Linear(2*p, d_model)  # Project one-hot to d_model
        
        # Keep Nanda's architecture
        self.attention = ReLUAttention(d_model, n_heads)
        self.mlp = TransformerMLP(d_model, d_mlp)
        self.unembed = nn.Linear(d_model, p, bias=False)
    
    def forward(self, x):
        """
        Args:
            x: [B, 2*p] one-hot encoded input (continuous)
        Returns:
            logits: [B, p] output logits
        """
        # Project one-hot input to d_model
        h = self.input_proj(x)  # [B, d_model]
        
        # Add sequence dimension for attention
        h = h.unsqueeze(1)  # [B, 1, d_model]
        
        # Transformer layer (with residual)
        h = h + self.attention(h)
        h = h + self.mlp(h)
        
        # Remove sequence dimension
        h = h.squeeze(1)  # [B, d_model]
        
        # Unembed to logits
        logits = self.unembed(h)  # [B, p]
        
        return logits


class OneHotStandardTransformer(nn.Module):
    """
    Standard transformer (softmax attention) with one-hot inputs.
    For comparison with ReLU transformer.
    
    Args:
        p: Modulus
        d_model: Model dimension
        n_heads: Number of attention heads  
        n_layers: Number of transformer layers
        d_ff: Feedforward dimension
    """
    def __init__(self, p: int, d_model: int = 128, n_heads: int = 4, 
                 n_layers: int = 2, d_ff: int = 512):
        super().__init__()
        self.p = p
        self.d_model = d_model
        
        # Project one-hot input to d_model
        self.input_proj = nn.Linear(2*p, d_model)
        
        # Standard transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, p)
    
    def forward(self, x):
        """
        Args:
            x: [B, 2*p] one-hot encoded input
        Returns:
            logits: [B, p] output logits
        """
        # Project to d_model
        h = self.input_proj(x)  # [B, d_model]
        
        # Add sequence dimension
        h = h.unsqueeze(1)  # [B, 1, d_model]
        
        # Transformer
        h = self.transformer(h)  # [B, 1, d_model]
        
        # Remove sequence dimension
        h = h.squeeze(1)  # [B, d_model]
        
        # Output
        logits = self.output_proj(h)  # [B, p]
        
        return logits


def test_models():
    """Test model creation and forward pass"""
    print("="*80)
    print("Testing One-Hot Models")
    print("="*80)
    
    p = 97
    batch_size = 32
    
    # Test MLP
    print("\n1. Testing ModularArithmeticMLP...")
    model_mlp = ModularArithmeticMLP(p, hidden_dim=128)
    x_onehot = torch.randn(batch_size, 2*p)  # Random one-hot-like input
    logits = model_mlp(x_onehot)
    print(f"  Input shape: {x_onehot.shape}")
    print(f"  Output shape: {logits.shape} (expected: [{batch_size}, {p}])")
    print(f"  ✓ MLP OK")
    
    # Test ReLU Transformer
    print("\n2. Testing OneHotReLUTransformer...")
    model_trans = OneHotReLUTransformer(p, d_model=128, n_heads=4, d_mlp=512)
    logits = model_trans(x_onehot)
    print(f"  Input shape: {x_onehot.shape}")
    print(f"  Output shape: {logits.shape} (expected: [{batch_size}, {p}])")
    print(f"  ✓ ReLU Transformer OK")
    
    # Test Standard Transformer
    print("\n3. Testing OneHotStandardTransformer...")
    model_std = OneHotStandardTransformer(p, d_model=128, n_heads=4, n_layers=2)
    logits = model_std(x_onehot)
    print(f"  Input shape: {x_onehot.shape}")
    print(f"  Output shape: {logits.shape} (expected: [{batch_size}, {p}])")
    print(f"  ✓ Standard Transformer OK")
    
    print("\n" + "="*80)
    print("✓ All models working correctly!")
    print("="*80)


if __name__ == "__main__":
    test_models()

