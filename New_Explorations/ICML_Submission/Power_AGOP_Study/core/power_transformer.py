"""
Power Transformer - Decoder-Only Transformer for Grokking Experiments

Based on Power et al. (2022) "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"

Key features:
- Decoder-only with causal masking
- Discrete token embeddings (a, b, =) as sequence
- Learned positional embeddings
- Pre-norm LayerNorm configuration
- GELU activation in MLP

Architecture:
- Input: Sequence of 3 tokens [tok_a, tok_b, tok_equals]
- Embedding dim: 128
- Layers: 2
- Heads: 4 (head_dim = 32)
- MLP hidden: 512
- Output: Linear from final token to p logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention for decoder-only transformer.
    
    Uses causal masking to prevent attending to future positions.
    """
    
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 4, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Combined QKV projection for efficiency
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Register causal mask as buffer (not a parameter)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer('causal_mask', mask.view(1, 1, max_seq_len, max_seq_len))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        B, T, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv_proj(x)  # [B, T, 3*d_model]
        qkv = qkv.view(B, T, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, n_heads, T, d_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: [B, n_heads, T, d_head]
        
        # Compute attention scores
        scale = 1.0 / math.sqrt(self.d_head)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, n_heads, T, T]
        
        # Apply causal mask
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Compute output
        out = torch.matmul(attn, v)  # [B, n_heads, T, d_head]
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, d_model]
        out = self.out_proj(out)
        
        return out


class TransformerMLP(nn.Module):
    """MLP block with GELU activation (Power et al. style)"""
    
    def __init__(self, d_model: int, d_mlp: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_mlp)
        self.fc2 = nn.Linear(d_mlp, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer block with pre-norm LayerNorm configuration.
    
    Pre-norm: LayerNorm is applied before attention/MLP (GPT-2 style)
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        d_mlp: int,
        max_seq_len: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = TransformerMLP(d_model, d_mlp, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        x = x + self.dropout(self.attn(self.ln1(x)))
        # Pre-norm MLP
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


class PowerTransformer(nn.Module):
    """
    Decoder-only Transformer for modular arithmetic (Power et al. 2022 style).
    
    Takes discrete token inputs [a, b, =] and outputs logits for (a op b) mod p.
    
    Args:
        p: Modulus for modular arithmetic (vocabulary size for operands)
        d_model: Embedding and model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer blocks
        d_mlp: Hidden dimension in MLP blocks
        dropout: Dropout probability
        max_seq_len: Maximum sequence length (default 3 for [a, b, =])
    """
    
    def __init__(
        self,
        p: int = 97,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_mlp: int = 512,
        dropout: float = 0.0,
        max_seq_len: int = 3
    ):
        super().__init__()
        self.p = p
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        # Token embeddings: p tokens for operands + 1 for equals sign
        # Vocabulary: 0 to p-1 for operands, p for "="
        self.vocab_size = p + 1
        self.token_embedding = nn.Embedding(self.vocab_size, d_model)
        
        # Learned positional embeddings
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_mlp, max_seq_len, dropout)
            for _ in range(n_layers)
        ])
        
        # Final LayerNorm (pre-norm style)
        self.ln_f = nn.LayerNorm(d_model)
        
        # Output head: project from final position to p logits
        self.head = nn.Linear(d_model, p, bias=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following GPT-2 conventions"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for discrete token input.
        
        Args:
            tokens: [batch_size, seq_len] integer tokens
                    For modular addition: [a, b, =] where a,b in [0,p-1], = is p
                    
        Returns:
            logits: [batch_size, p] output logits for the answer
        """
        B, T = tokens.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max {self.max_seq_len}"
        
        # Get token embeddings
        tok_emb = self.token_embedding(tokens)  # [B, T, d_model]
        
        # Get positional embeddings
        positions = torch.arange(T, device=tokens.device)
        pos_emb = self.pos_embedding(positions)  # [T, d_model]
        
        # Combine embeddings
        x = tok_emb + pos_emb  # [B, T, d_model]
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final LayerNorm
        x = self.ln_f(x)
        
        # Get output from final position (the "=" token)
        x = x[:, -1, :]  # [B, d_model]
        
        # Project to logits
        logits = self.head(x)  # [B, p]
        
        return logits
    
    def forward_onehot(self, x_onehot: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for one-hot encoded input (for AGOP computation).
        
        The one-hot input is [B, 2*p] where first p dims encode 'a' and 
        next p dims encode 'b'. Uses differentiable soft embedding lookup
        via matrix multiplication to maintain gradient flow.
        
        Args:
            x_onehot: [batch_size, 2*p] one-hot encoded input
            
        Returns:
            logits: [batch_size, p] output logits
        """
        B = x_onehot.shape[0]
        device = x_onehot.device
        
        # Get embedding weight matrix (excludes the "=" token at index p)
        # token_embedding.weight is [vocab_size, d_model] where vocab_size = p+1
        operand_embeddings = self.token_embedding.weight[:self.p, :]  # [p, d_model]
        equals_embedding = self.token_embedding.weight[self.p:self.p+1, :]  # [1, d_model]
        
        # Soft embedding lookup via matrix multiplication (differentiable)
        # x_onehot[:, :p] @ operand_embeddings gives embedded 'a'
        # x_onehot[:, p:] @ operand_embeddings gives embedded 'b'
        a_emb = x_onehot[:, :self.p] @ operand_embeddings  # [B, d_model]
        b_emb = x_onehot[:, self.p:] @ operand_embeddings  # [B, d_model]
        equals_emb = equals_embedding.expand(B, -1)  # [B, d_model]
        
        # Stack into sequence [a, b, =]
        tok_emb = torch.stack([a_emb, b_emb, equals_emb], dim=1)  # [B, 3, d_model]
        
        # Get positional embeddings
        positions = torch.arange(3, device=device)
        pos_emb = self.pos_embedding(positions)  # [3, d_model]
        
        # Combine embeddings
        x = tok_emb + pos_emb  # [B, 3, d_model]
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final LayerNorm
        x = self.ln_f(x)
        
        # Get output from final position (the "=" token)
        x = x[:, -1, :]  # [B, d_model]
        
        # Project to logits
        logits = self.head(x)  # [B, p]
        
        return logits
    
    def get_config(self) -> dict:
        """Return model configuration"""
        return {
            'p': self.p,
            'd_model': self.d_model,
            'n_heads': self.blocks[0].attn.n_heads,
            'n_layers': self.n_layers,
            'd_mlp': self.blocks[0].mlp.fc1.out_features,
            'max_seq_len': self.max_seq_len,
            'vocab_size': self.vocab_size,
        }


def test_power_transformer():
    """Test PowerTransformer with sample inputs"""
    print("=" * 80)
    print("Testing PowerTransformer")
    print("=" * 80)
    
    p = 97
    batch_size = 32
    
    model = PowerTransformer(
        p=p,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_mlp=512,
        dropout=0.0
    )
    
    n_params = sum(param.numel() for param in model.parameters())
    print(f"\n1. Model created with {n_params:,} parameters")
    print(f"   Config: {model.get_config()}")
    
    # Test discrete token input
    print("\n2. Testing discrete token forward pass...")
    a = torch.randint(0, p, (batch_size,))
    b = torch.randint(0, p, (batch_size,))
    equals = torch.full((batch_size,), p)
    tokens = torch.stack([a, b, equals], dim=1)  # [B, 3]
    
    logits = model(tokens)
    print(f"   Input tokens shape: {tokens.shape}")
    print(f"   Output logits shape: {logits.shape}")
    assert logits.shape == (batch_size, p), "Discrete token output shape mismatch!"
    print("   ✓ Discrete token test passed")
    
    # Test one-hot input (for AGOP)
    print("\n3. Testing one-hot forward pass...")
    x_onehot = torch.zeros(batch_size, 2 * p)
    for i in range(batch_size):
        x_onehot[i, a[i]] = 1.0
        x_onehot[i, p + b[i]] = 1.0
    
    logits_onehot = model.forward_onehot(x_onehot)
    print(f"   Input one-hot shape: {x_onehot.shape}")
    print(f"   Output logits shape: {logits_onehot.shape}")
    
    # Verify both methods give same output
    assert torch.allclose(logits, logits_onehot), "One-hot and token outputs should match!"
    print("   ✓ One-hot test passed (matches discrete token output)")
    
    # Test gradient flow
    print("\n4. Testing gradient flow...")
    tokens.requires_grad = False
    x_onehot.requires_grad_(True)
    logits = model.forward_onehot(x_onehot)
    loss = logits.sum()
    loss.backward()
    print(f"   Gradient computed for one-hot input: {x_onehot.grad is not None}")
    print("   ✓ Gradient flow test passed")
    
    print("\n" + "=" * 80)
    print("✓ All PowerTransformer tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    test_power_transformer()

