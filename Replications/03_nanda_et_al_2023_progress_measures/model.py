"""
1-Layer ReLU Transformer for Modular Addition
Based on Nanda et al. (2023) - Progress Measures for Grokking via Mechanistic Interpretability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ReLUAttention(nn.Module):
    """Multi-head attention with ReLU activation instead of softmax"""
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
        
        # Project to Q, K, V
        Q = self.W_Q(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        K = self.W_K(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        V = self.W_V(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        
        # Transpose for attention: (batch, n_heads, seq_len, d_head)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        # Apply ReLU instead of softmax
        attn_weights = F.relu(scores)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_O(attn_output)
        
        return output


class MLP(nn.Module):
    """MLP block with ReLU activation"""
    def __init__(self, d_model, d_mlp):
        super().__init__()
        self.W_in = nn.Linear(d_model, d_mlp, bias=False)
        self.W_out = nn.Linear(d_mlp, d_model, bias=False)
        
    def forward(self, x):
        return self.W_out(F.relu(self.W_in(x)))


class OneLayerReLUTransformer(nn.Module):
    """
    One-layer ReLU Transformer for modular addition
    Architecture from Nanda et al. (2023)
    """
    def __init__(self, p, d_model=128, n_heads=4, d_mlp=512):
        super().__init__()
        self.p = p  # Modulus (e.g., 113)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_mlp = d_mlp
        
        # Token embeddings for numbers 0 to p-1, plus special tokens
        # Token format: "a b =" where we read output from position 2 (after =)
        self.token_embed = nn.Embedding(p, d_model)
        self.pos_embed = nn.Embedding(3, d_model)  # 3 positions
        
        # Transformer layer
        self.attention = ReLUAttention(d_model, n_heads)
        self.mlp = MLP(d_model, d_mlp)
        
        # Output projection (no LayerNorm as specified in paper)
        self.unembed = nn.Linear(d_model, p, bias=False)
        
    def forward(self, x):
        """
        x: (batch_size, 3) containing [a, b, equals_token]
        where equals_token is represented as p (special token)
        """
        batch_size = x.shape[0]
        
        # For the equals token, we use modulo to wrap it to valid range
        x_clipped = torch.clamp(x, 0, self.p - 1)
        
        # Embeddings
        token_embeds = self.token_embed(x_clipped)  # (batch, 3, d_model)
        pos_ids = torch.arange(3, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.pos_embed(pos_ids)
        
        h = token_embeds + pos_embeds
        
        # Transformer layer
        h = h + self.attention(h)
        h = h + self.mlp(h)
        
        # Read output from position 2 (after =)
        h_out = h[:, 2, :]  # (batch, d_model)
        logits = self.unembed(h_out)  # (batch, p)
        
        return logits


def create_modular_addition_dataset(p, train_fraction=0.3, device='cpu'):
    """
    Create train/test split for modular addition
    
    Args:
        p: modulus (prime number)
        train_fraction: fraction of data for training
        device: torch device
        
    Returns:
        (train_data, train_labels, test_data, test_labels)
    """
    # Generate all possible pairs
    all_pairs = []
    all_labels = []
    
    for a in range(p):
        for b in range(p):
            all_pairs.append([a, b, p])  # p represents the "=" token
            all_labels.append((a + b) % p)
    
    # Convert to tensors
    all_pairs = torch.tensor(all_pairs, device=device)
    all_labels = torch.tensor(all_labels, device=device)
    
    # Random split
    n_total = len(all_pairs)
    n_train = int(n_total * train_fraction)
    
    perm = torch.randperm(n_total, device=device)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    
    train_data = all_pairs[train_idx]
    train_labels = all_labels[train_idx]
    test_data = all_pairs[test_idx]
    test_labels = all_labels[test_idx]
    
    return train_data, train_labels, test_data, test_labels

