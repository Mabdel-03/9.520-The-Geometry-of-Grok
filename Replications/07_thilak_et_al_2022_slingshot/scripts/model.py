"""
Transformer model for Slingshot Mechanism experiments
Thilak et al. (2022) - The Slingshot Mechanism

Based on standard transformer architecture for modular arithmetic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention"""
    def __init__(self, d_model, n_heads):
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
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_O(attn_output)
        
        return output


class TransformerBlock(nn.Module):
    """Standard transformer block"""
    def __init__(self, d_model, n_heads, d_mlp):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.ReLU(),
            nn.Linear(d_mlp, d_model)
        )
        
    def forward(self, x):
        x = x + self.attention(x)
        x = x + self.mlp(x)
        return x


class ModularArithmeticTransformer(nn.Module):
    """
    Transformer for modular arithmetic tasks
    Focus on tracking last-layer weight norms for Slingshot analysis
    """
    def __init__(self, p, d_model=128, n_heads=4, n_layers=2, d_mlp=512):
        super().__init__()
        self.p = p
        self.d_model = d_model
        
        # Embeddings
        self.token_embed = nn.Embedding(p + 1, d_model)  # +1 for special tokens
        self.pos_embed = nn.Embedding(3, d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_mlp)
            for _ in range(n_layers)
        ])
        
        # Output projection (last layer - track this for Slingshot)
        self.output_layer = nn.Linear(d_model, p, bias=True)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling"""
        # Initialize embeddings
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        
        # Initialize transformer layers
        for layer in self.layers:
            for p in layer.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        
        # Initialize output layer
        nn.init.normal_(self.output_layer.weight, std=0.02)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Clip to valid range
        x_clipped = torch.clamp(x, 0, self.p)
        
        # Embeddings
        token_embeds = self.token_embed(x_clipped)
        pos_ids = torch.arange(3, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.pos_embed(pos_ids)
        
        h = token_embeds + pos_embeds
        
        # Apply transformer layers
        for layer in self.layers:
            h = layer(h)
        
        # Read from last position
        h_out = h[:, -1, :]
        logits = self.output_layer(h_out)
        
        return logits
    
    def get_last_layer_norm(self):
        """Get L2 norm of last layer weights for Slingshot tracking"""
        return torch.norm(self.output_layer.weight).item()
    
    def get_all_parameter_norms(self):
        """Get norms of all parameter groups"""
        norms = {}
        norms['embeddings'] = (
            torch.norm(self.token_embed.weight).item() +
            torch.norm(self.pos_embed.weight).item()
        )
        norms['transformers'] = sum(
            torch.norm(p).item() for layer in self.layers for p in layer.parameters()
        )
        norms['output'] = (
            torch.norm(self.output_layer.weight).item() +
            torch.norm(self.output_layer.bias).item()
        )
        return norms


def create_modular_addition_dataset(p, train_fraction=0.5, device='cpu'):
    """Create dataset for modular addition"""
    all_pairs = []
    all_labels = []
    
    for a in range(p):
        for b in range(p):
            all_pairs.append([a, b, p])  # p as special "=" token
            all_labels.append((a + b) % p)
    
    all_pairs = torch.tensor(all_pairs, device=device)
    all_labels = torch.tensor(all_labels, device=device)
    
    n_total = len(all_pairs)
    n_train = int(n_total * train_fraction)
    
    perm = torch.randperm(n_total, device=device)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    
    return (all_pairs[train_idx], all_labels[train_idx],
            all_pairs[test_idx], all_labels[test_idx])

