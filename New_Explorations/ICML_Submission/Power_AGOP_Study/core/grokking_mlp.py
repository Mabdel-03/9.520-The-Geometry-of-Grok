"""
Grokking MLP - Simple MLP Baseline for Grokking Experiments

A minimal 3-layer MLP with no inductive biases, serving as a baseline
to compare against the transformer architecture.

Architecture:
- Input: Concatenated embeddings [emb_a; emb_b] (dim 2*d) or one-hot (dim 2*p)
- Hidden: 512 -> 512 (ReLU activation)
- Output: p logits

Supports both:
1. Discrete token input with learned embeddings
2. One-hot continuous input (for AGOP computation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GrokkingMLP(nn.Module):
    """
    3-layer MLP for modular arithmetic experiments.
    
    Supports two input modes:
    1. Discrete tokens: a, b as integers embedded via learned embeddings
    2. One-hot: a, b as one-hot vectors concatenated (for gradient computation)
    
    Args:
        p: Modulus for modular arithmetic (determines output size)
        d_embed: Embedding dimension for discrete tokens
        d_hidden: Hidden layer dimension (used for both hidden layers)
        use_layernorm: Whether to use LayerNorm after hidden layers
        dropout: Dropout probability
        input_type: 'discrete' for token embeddings, 'onehot' for one-hot input
    """
    
    def __init__(
        self,
        p: int = 97,
        d_embed: int = 128,
        d_hidden: int = 512,
        use_layernorm: bool = False,
        dropout: float = 0.0,
        input_type: str = 'discrete'
    ):
        super().__init__()
        self.p = p
        self.d_embed = d_embed
        self.d_hidden = d_hidden
        self.use_layernorm = use_layernorm
        self.input_type = input_type
        
        # Input dimension depends on input type
        if input_type == 'discrete':
            # Embedding layer for discrete tokens
            self.embedding = nn.Embedding(p, d_embed)
            input_dim = 2 * d_embed  # Concatenated embeddings [emb_a; emb_b]
        else:  # 'onehot'
            self.embedding = None
            input_dim = 2 * p  # Concatenated one-hot vectors
        
        # MLP layers: input -> 512 -> 512 -> p
        self.fc1 = nn.Linear(input_dim, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.fc3 = nn.Linear(d_hidden, p)
        
        # Optional LayerNorm
        if use_layernorm:
            self.ln1 = nn.LayerNorm(d_hidden)
            self.ln2 = nn.LayerNorm(d_hidden)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small random values"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: For discrete input: [batch_size, 2] integer tokens (a, b)
               For one-hot input: [batch_size, 2*p] concatenated one-hot vectors
               
        Returns:
            logits: [batch_size, p] output logits
        """
        if self.input_type == 'discrete':
            # x is [B, 2] with tokens a, b
            a_emb = self.embedding(x[:, 0])  # [B, d_embed]
            b_emb = self.embedding(x[:, 1])  # [B, d_embed]
            h = torch.cat([a_emb, b_emb], dim=1)  # [B, 2*d_embed]
        else:
            # x is already [B, 2*p] one-hot
            h = x
        
        # First hidden layer
        h = self.fc1(h)
        if self.use_layernorm:
            h = self.ln1(h)
        h = F.relu(h)
        h = self.dropout(h)
        
        # Second hidden layer
        h = self.fc2(h)
        if self.use_layernorm:
            h = self.ln2(h)
        h = F.relu(h)
        h = self.dropout(h)
        
        # Output layer
        logits = self.fc3(h)  # [B, p]
        
        return logits
    
    def forward_onehot(self, x_onehot: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for one-hot encoded input (for AGOP computation).
        
        If model uses discrete embeddings, convert one-hot to tokens first,
        then use embeddings. This allows gradient computation w.r.t. inputs.
        
        Args:
            x_onehot: [batch_size, 2*p] one-hot encoded input
            
        Returns:
            logits: [batch_size, p] output logits
        """
        if self.input_type == 'onehot':
            # Directly use one-hot input
            return self.forward(x_onehot)
        else:
            # For discrete mode, we need to convert but keep gradients flowing
            # Use soft embedding lookup via matrix multiplication
            # x_onehot[:, :p] @ embedding.weight gives embedded 'a'
            # x_onehot[:, p:] @ embedding.weight gives embedded 'b'
            
            a_emb = x_onehot[:, :self.p] @ self.embedding.weight  # [B, d_embed]
            b_emb = x_onehot[:, self.p:] @ self.embedding.weight  # [B, d_embed]
            h = torch.cat([a_emb, b_emb], dim=1)  # [B, 2*d_embed]
            
            # First hidden layer
            h = self.fc1(h)
            if self.use_layernorm:
                h = self.ln1(h)
            h = F.relu(h)
            h = self.dropout(h)
            
            # Second hidden layer
            h = self.fc2(h)
            if self.use_layernorm:
                h = self.ln2(h)
            h = F.relu(h)
            h = self.dropout(h)
            
            # Output layer
            logits = self.fc3(h)
            
            return logits
    
    def get_config(self) -> dict:
        """Return model configuration"""
        return {
            'p': self.p,
            'd_embed': self.d_embed,
            'd_hidden': self.d_hidden,
            'use_layernorm': self.use_layernorm,
            'input_type': self.input_type,
        }


def test_grokking_mlp():
    """Test GrokkingMLP with sample inputs"""
    print("=" * 80)
    print("Testing GrokkingMLP")
    print("=" * 80)
    
    p = 97
    batch_size = 32
    
    # Test discrete input mode
    print("\n1. Testing discrete token input mode...")
    model_discrete = GrokkingMLP(
        p=p,
        d_embed=128,
        d_hidden=512,
        use_layernorm=False,
        input_type='discrete'
    )
    
    n_params = sum(param.numel() for param in model_discrete.parameters())
    print(f"   Model created with {n_params:,} parameters")
    print(f"   Config: {model_discrete.get_config()}")
    
    # Create discrete input [a, b]
    tokens = torch.stack([
        torch.randint(0, p, (batch_size,)),
        torch.randint(0, p, (batch_size,))
    ], dim=1)  # [B, 2]
    
    logits = model_discrete(tokens)
    print(f"   Input tokens shape: {tokens.shape}")
    print(f"   Output logits shape: {logits.shape}")
    assert logits.shape == (batch_size, p), "Discrete output shape mismatch!"
    print("   ✓ Discrete token test passed")
    
    # Test one-hot forward for AGOP
    print("\n2. Testing one-hot forward pass (for AGOP)...")
    x_onehot = torch.zeros(batch_size, 2 * p)
    for i in range(batch_size):
        x_onehot[i, tokens[i, 0]] = 1.0
        x_onehot[i, p + tokens[i, 1]] = 1.0
    
    x_onehot.requires_grad_(True)
    logits_onehot = model_discrete.forward_onehot(x_onehot)
    print(f"   Input one-hot shape: {x_onehot.shape}")
    print(f"   Output logits shape: {logits_onehot.shape}")
    print("   ✓ One-hot forward test passed")
    
    # Test gradient flow
    print("\n3. Testing gradient flow...")
    loss = logits_onehot.sum()
    loss.backward()
    print(f"   Gradient shape: {x_onehot.grad.shape}")
    print(f"   Gradient non-zero: {(x_onehot.grad != 0).any()}")
    print("   ✓ Gradient flow test passed")
    
    # Test one-hot input mode
    print("\n4. Testing native one-hot input mode...")
    model_onehot = GrokkingMLP(
        p=p,
        d_embed=128,
        d_hidden=512,
        use_layernorm=False,
        input_type='onehot'
    )
    
    x_onehot_new = torch.zeros(batch_size, 2 * p)
    for i in range(batch_size):
        x_onehot_new[i, tokens[i, 0]] = 1.0
        x_onehot_new[i, p + tokens[i, 1]] = 1.0
    
    logits_native = model_onehot(x_onehot_new)
    print(f"   Output logits shape: {logits_native.shape}")
    assert logits_native.shape == (batch_size, p), "One-hot mode output shape mismatch!"
    print("   ✓ Native one-hot mode test passed")
    
    # Test with LayerNorm
    print("\n5. Testing with LayerNorm enabled...")
    model_ln = GrokkingMLP(
        p=p,
        d_embed=128,
        d_hidden=512,
        use_layernorm=True,
        input_type='discrete'
    )
    
    logits_ln = model_ln(tokens)
    print(f"   Output logits shape: {logits_ln.shape}")
    print("   ✓ LayerNorm mode test passed")
    
    print("\n" + "=" * 80)
    print("✓ All GrokkingMLP tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    test_grokking_mlp()

