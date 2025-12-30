"""
Unified Transformer Model for ICML 16_Runs Experiments

A configurable transformer architecture supporting:
- Softmax vs ReLU attention
- LayerNorm on/off toggle
- Variable modulus (97 or 113)

This enables systematic study of how architectural choices affect grokking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SoftmaxAttention(nn.Module):
    """Standard multi-head softmax attention"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        Q = self.W_Q(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        K = self.W_K(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        V = self.W_V(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        
        Q = Q.transpose(1, 2)  # [B, n_heads, seq_len, d_head]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_O(attn_output)
        
        return output


class ReLUAttention(nn.Module):
    """Multi-head ReLU attention (from Nanda et al.)"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
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
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_O(attn_output)
        
        return output


class TransformerMLP(nn.Module):
    """MLP block for transformer with GELU activation"""
    def __init__(self, d_model: int, d_mlp: int, dropout: float = 0.0):
        super().__init__()
        self.W_in = nn.Linear(d_model, d_mlp)
        self.W_out = nn.Linear(d_mlp, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.W_in(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.W_out(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer block with configurable attention and LayerNorm.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_mlp: MLP hidden dimension
        attention_type: 'softmax' or 'relu'
        use_layernorm: Whether to use LayerNorm
        dropout: Dropout probability
    """
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        d_mlp: int,
        attention_type: str = 'softmax',
        use_layernorm: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        self.use_layernorm = use_layernorm
        
        # Choose attention type
        if attention_type == 'softmax':
            self.attention = SoftmaxAttention(d_model, n_heads, dropout)
        elif attention_type == 'relu':
            self.attention = ReLUAttention(d_model, n_heads, dropout)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        self.mlp = TransformerMLP(d_model, d_mlp, dropout)
        
        # Optional LayerNorm
        if use_layernorm:
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Attention with residual
        if self.use_layernorm:
            x = x + self.dropout(self.attention(self.ln1(x)))
            x = x + self.dropout(self.mlp(self.ln2(x)))
        else:
            x = x + self.dropout(self.attention(x))
            x = x + self.dropout(self.mlp(x))
        
        return x


class UnifiedTransformer(nn.Module):
    """
    Unified Transformer for ICML 16_Runs experiments.
    
    Supports configurable:
    - Modulus (determines input/output dimensions)
    - Attention type (softmax or relu)
    - LayerNorm (on or off)
    
    Args:
        p: Modulus for modular arithmetic
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_mlp: MLP hidden dimension
        attention_type: 'softmax' or 'relu'
        use_layernorm: Whether to use LayerNorm
        dropout: Dropout probability
    """
    def __init__(
        self,
        p: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 1,
        d_mlp: int = 512,
        attention_type: str = 'softmax',
        use_layernorm: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        self.p = p
        self.d_model = d_model
        self.attention_type = attention_type
        self.use_layernorm = use_layernorm
        
        # Input projection: one-hot [B, 2*p] -> [B, d_model]
        self.input_proj = nn.Linear(2 * p, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_mlp=d_mlp,
                attention_type=attention_type,
                use_layernorm=use_layernorm,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        
        # Optional final LayerNorm
        if use_layernorm:
            self.final_ln = nn.LayerNorm(d_model)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, p)
    
    def forward(self, x):
        """
        Args:
            x: [B, 2*p] one-hot encoded input (a, b concatenated)
        Returns:
            logits: [B, p] output logits for (a + b) mod p
        """
        # Project to model dimension
        h = self.input_proj(x)  # [B, d_model]
        
        # Add sequence dimension for attention
        h = h.unsqueeze(1)  # [B, 1, d_model]
        
        # Transformer blocks
        for block in self.blocks:
            h = block(h)
        
        # Remove sequence dimension
        h = h.squeeze(1)  # [B, d_model]
        
        # Final LayerNorm if enabled
        if self.use_layernorm:
            h = self.final_ln(h)
        
        # Output projection
        logits = self.output_proj(h)  # [B, p]
        
        return logits
    
    def get_config(self):
        """Return model configuration as dict"""
        return {
            'p': self.p,
            'd_model': self.d_model,
            'n_heads': self.blocks[0].attention.n_heads,
            'n_layers': len(self.blocks),
            'd_mlp': self.blocks[0].mlp.W_in.out_features,
            'attention_type': self.attention_type,
            'use_layernorm': self.use_layernorm,
        }


def test_unified_transformer():
    """Test UnifiedTransformer with all configurations"""
    print("=" * 80)
    print("Testing UnifiedTransformer - All Configurations")
    print("=" * 80)
    
    batch_size = 32
    
    configs = [
        {'p': 97, 'attention_type': 'softmax', 'use_layernorm': True},
        {'p': 97, 'attention_type': 'softmax', 'use_layernorm': False},
        {'p': 97, 'attention_type': 'relu', 'use_layernorm': True},
        {'p': 97, 'attention_type': 'relu', 'use_layernorm': False},
        {'p': 113, 'attention_type': 'softmax', 'use_layernorm': True},
        {'p': 113, 'attention_type': 'softmax', 'use_layernorm': False},
        {'p': 113, 'attention_type': 'relu', 'use_layernorm': True},
        {'p': 113, 'attention_type': 'relu', 'use_layernorm': False},
    ]
    
    for i, cfg in enumerate(configs, 1):
        p = cfg['p']
        model = UnifiedTransformer(
            p=p,
            d_model=128,
            n_heads=4,
            n_layers=1,
            d_mlp=512,
            attention_type=cfg['attention_type'],
            use_layernorm=cfg['use_layernorm']
        )
        
        x = torch.randn(batch_size, 2 * p)
        logits = model(x)
        
        n_params = sum(p.numel() for p in model.parameters())
        
        print(f"{i}. p={p}, attn={cfg['attention_type']}, LN={cfg['use_layernorm']}")
        print(f"   Input: {x.shape} -> Output: {logits.shape}")
        print(f"   Parameters: {n_params:,}")
        print()
    
    print("=" * 80)
    print("All configurations passed!")
    print("=" * 80)


if __name__ == "__main__":
    test_unified_transformer()

