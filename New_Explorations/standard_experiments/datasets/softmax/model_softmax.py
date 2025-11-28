"""
Standard Softmax Transformer for Modular Addition
Compatible with Muon optimizer (uses softmax attention, LayerNorm)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class StandardTransformer(nn.Module):
    """
    Standard Transformer for modular addition.
    
    Architecture compatible with Muon optimizer:
    - Softmax attention (standard multi-head attention)
    - LayerNorm for normalization
    - Residual connections
    - GELU activation in FFN
    - Learned positional embeddings
    
    This contrasts with Nanda's ReLU Transformer which uses:
    - ReLU attention (non-standard)
    - No normalization layers
    - No residual connections
    """
    
    def __init__(
        self,
        p: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.p = p
        self.d_model = d_model
        
        # Token embeddings: 0 to p-1 (numbers) + special token for "="
        self.token_embed = nn.Embedding(p + 1, d_model)
        self.pos_embed = nn.Embedding(3, d_model)  # 3 positions: a, b, =
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',  # GELU activation
            batch_first=True,
            norm_first=False  # Post-norm (LayerNorm after attention)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, p)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (batch_size, 3) containing [a, b, equals_token]
               where a, b are in range [0, p-1]
               and equals_token can be represented as p
        
        Returns:
            logits: (batch_size, p) - predictions for (a + b) mod p
        """
        batch_size = x.shape[0]
        
        # Clip values to valid range
        x_clipped = torch.clamp(x, 0, self.p)
        
        # Token embeddings
        token_emb = self.token_embed(x_clipped)  # (batch, 3, d_model)
        
        # Positional embeddings
        pos_ids = torch.arange(3, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embed(pos_ids)  # (batch, 3, d_model)
        
        # Combine embeddings
        embeddings = token_emb + pos_emb
        
        # Transformer (with softmax attention + LayerNorm)
        transformer_out = self.transformer(embeddings)  # (batch, 3, d_model)
        
        # Read from position 2 (after "=")
        final_repr = transformer_out[:, 2, :]  # (batch, d_model)
        
        # Project to output logits
        logits = self.output_proj(final_repr)  # (batch, p)
        
        return logits


def create_modular_addition_dataset(p, train_fraction, device='cuda'):
    """
    Create modular addition dataset: (a + b) mod p
    
    Args:
        p: Modulus (e.g., 97 or 113)
        train_fraction: Fraction of data for training (e.g., 0.5 or 0.8)
        device: Device to place tensors on
        
    Returns:
        train_data, train_labels, test_data, test_labels
    """
    # Generate all possible pairs
    all_pairs = []
    all_labels = []
    
    for a in range(p):
        for b in range(p):
            # Input: [a, b, p] where p represents "="
            all_pairs.append([a, b, p])
            # Label: (a + b) mod p
            all_labels.append((a + b) % p)
    
    # Convert to tensors
    all_pairs = torch.tensor(all_pairs, dtype=torch.long)
    all_labels = torch.tensor(all_labels, dtype=torch.long)
    
    # Shuffle
    perm = torch.randperm(len(all_pairs))
    all_pairs = all_pairs[perm]
    all_labels = all_labels[perm]
    
    # Split
    n_train = int(len(all_pairs) * train_fraction)
    
    train_data = all_pairs[:n_train].to(device)
    train_labels = all_labels[:n_train].to(device)
    test_data = all_pairs[n_train:].to(device)
    test_labels = all_labels[n_train:].to(device)
    
    return train_data, train_labels, test_data, test_labels


if __name__ == "__main__":
    # Test the model
    print("Testing Standard Softmax Transformer...")
    
    p = 97
    model = StandardTransformer(p, d_model=128, n_heads=4, n_layers=2)
    
    # Test forward pass
    batch_size = 32
    x = torch.randint(0, p, (batch_size, 3))
    x[:, 2] = p  # Set third position to "=" token
    
    logits = model(x)
    print(f"Model output shape: {logits.shape}")
    print(f"Expected: ({batch_size}, {p})")
    
    # Test dataset creation
    train_data, train_labels, test_data, test_labels = create_modular_addition_dataset(
        p=97, train_fraction=0.5, device='cpu'
    )
    print(f"\nDataset created:")
    print(f"  Train size: {len(train_data)}")
    print(f"  Test size: {len(test_data)}")
    print(f"  Total: {len(train_data) + len(test_data)} (should be {97*97})")
    
    print("\n✓ Softmax Transformer test passed!")

