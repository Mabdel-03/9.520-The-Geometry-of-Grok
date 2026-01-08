"""
Dataset Loaders for Power AGOP Grokking Experiments

Provides modular arithmetic datasets in two formats:
1. Discrete tokens: Integer indices for (a, b) pairs
2. One-hot continuous: Float vectors for gradient computation (AGOP)

Based on Power et al. (2022) "Grokking" setup:
- Modular addition: (a + b) mod p
- p = 97 (prime)
- 50/50 train/test split
- Fixed random seed for reproducibility
"""

import torch
import numpy as np
from typing import Tuple, Optional


def create_modular_dataset_discrete(
    p: int = 97,
    operation: str = 'add',
    train_fraction: float = 0.5,
    seed: int = 42,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create modular arithmetic dataset with discrete token representation.
    
    Returns integer pairs (a, b) as input and (a op b) mod p as labels.
    
    Args:
        p: Modulus (prime number)
        operation: 'add', 'sub', 'mul', or 'div'
        train_fraction: Fraction of data for training (default 0.5 = 50%)
        seed: Random seed for reproducibility
        device: Device to place tensors on
        
    Returns:
        train_data: [N_train, 2] integer tokens (a, b)
        train_labels: [N_train] targets
        test_data: [N_test, 2] integer tokens
        test_labels: [N_test] targets
    """
    # Set seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    all_pairs = []
    all_labels = []
    
    # Helper for modular division (Fermat's little theorem)
    def mod_inverse(b, p):
        return pow(b, p - 2, p)
    
    # Generate all pairs
    for a in range(p):
        for b in range(p):
            # Skip invalid division by zero
            if operation == 'div' and b == 0:
                continue
            
            all_pairs.append([a, b])
            
            # Compute label based on operation
            if operation == 'add':
                label = (a + b) % p
            elif operation == 'sub':
                label = (a - b) % p
            elif operation == 'mul':
                label = (a * b) % p
            elif operation == 'div':
                label = (a * mod_inverse(b, p)) % p
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            all_labels.append(label)
    
    # Convert to tensors
    X = torch.tensor(all_pairs, dtype=torch.long, device=device)
    y = torch.tensor(all_labels, dtype=torch.long, device=device)
    
    # Shuffle with fixed seed
    n_total = len(X)
    perm = torch.randperm(n_total, device=device)
    X = X[perm]
    y = y[perm]
    
    # Split train/test
    n_train = int(n_total * train_fraction)
    
    train_data = X[:n_train]
    train_labels = y[:n_train]
    test_data = X[n_train:]
    test_labels = y[n_train:]
    
    return train_data, train_labels, test_data, test_labels


def create_modular_dataset_onehot(
    p: int = 97,
    operation: str = 'add',
    train_fraction: float = 0.5,
    seed: int = 42,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create modular arithmetic dataset with one-hot encoded representation.
    
    Returns concatenated one-hot vectors [onehot(a); onehot(b)] of size 2*p.
    This enables gradient computation w.r.t. inputs for AGOP analysis.
    
    Args:
        p: Modulus (prime number)
        operation: 'add', 'sub', 'mul', or 'div'
        train_fraction: Fraction of data for training (default 0.5 = 50%)
        seed: Random seed for reproducibility
        device: Device to place tensors on
        
    Returns:
        train_data: [N_train, 2*p] one-hot encoded (float32)
        train_labels: [N_train] targets (long)
        test_data: [N_test, 2*p] one-hot encoded (float32)
        test_labels: [N_test] targets (long)
    """
    # Set seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    all_data = []
    all_labels = []
    
    # Helper for modular division
    def mod_inverse(b, p):
        return pow(b, p - 2, p)
    
    # Generate all pairs
    for a in range(p):
        for b in range(p):
            # Skip invalid division by zero
            if operation == 'div' and b == 0:
                continue
            
            # One-hot encode: [onehot(a); onehot(b)]
            v = np.zeros(2 * p, dtype=np.float32)
            v[a] = 1.0        # First p dims: one-hot for a
            v[p + b] = 1.0    # Next p dims: one-hot for b
            all_data.append(v)
            
            # Compute label
            if operation == 'add':
                label = (a + b) % p
            elif operation == 'sub':
                label = (a - b) % p
            elif operation == 'mul':
                label = (a * b) % p
            elif operation == 'div':
                label = (a * mod_inverse(b, p)) % p
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            all_labels.append(label)
    
    # Convert to tensors
    X = torch.tensor(np.array(all_data), dtype=torch.float32, device=device)
    y = torch.tensor(all_labels, dtype=torch.long, device=device)
    
    # Shuffle with same seed as discrete version
    n_total = len(X)
    perm = torch.randperm(n_total, device=device)
    X = X[perm]
    y = y[perm]
    
    # Split train/test
    n_train = int(n_total * train_fraction)
    
    train_data = X[:n_train]
    train_labels = y[:n_train]
    test_data = X[n_train:]
    test_labels = y[n_train:]
    
    return train_data, train_labels, test_data, test_labels


def discrete_to_onehot(tokens: torch.Tensor, p: int) -> torch.Tensor:
    """
    Convert discrete token pairs to one-hot representation.
    
    Args:
        tokens: [batch_size, 2] integer tokens (a, b)
        p: Modulus (vocabulary size)
        
    Returns:
        onehot: [batch_size, 2*p] one-hot encoded
    """
    batch_size = tokens.shape[0]
    device = tokens.device
    
    onehot = torch.zeros(batch_size, 2 * p, dtype=torch.float32, device=device)
    
    # Set one-hot for a (first p dimensions)
    onehot.scatter_(1, tokens[:, 0:1], 1.0)
    
    # Set one-hot for b (next p dimensions, offset by p)
    onehot.scatter_(1, tokens[:, 1:2] + p, 1.0)
    
    return onehot


def onehot_to_discrete(onehot: torch.Tensor, p: int) -> torch.Tensor:
    """
    Convert one-hot representation to discrete token pairs.
    
    Args:
        onehot: [batch_size, 2*p] one-hot encoded
        p: Modulus
        
    Returns:
        tokens: [batch_size, 2] integer tokens (a, b)
    """
    a = onehot[:, :p].argmax(dim=1)
    b = onehot[:, p:].argmax(dim=1)
    return torch.stack([a, b], dim=1)


def create_transformer_tokens(
    discrete_data: torch.Tensor,
    p: int
) -> torch.Tensor:
    """
    Convert discrete pairs to transformer sequence format [a, b, =].
    
    Args:
        discrete_data: [batch_size, 2] integer tokens (a, b)
        p: Modulus (= token is p)
        
    Returns:
        tokens: [batch_size, 3] sequence tokens [a, b, =]
    """
    batch_size = discrete_data.shape[0]
    device = discrete_data.device
    
    # Create equals token (index p in vocabulary)
    equals = torch.full((batch_size, 1), p, dtype=torch.long, device=device)
    
    # Concatenate: [a, b, =]
    tokens = torch.cat([discrete_data, equals], dim=1)
    
    return tokens


class ModularArithmeticDataset:
    """
    Unified dataset class that provides both discrete and one-hot representations.
    
    Useful for training with discrete tokens but computing AGOP with one-hot.
    """
    
    def __init__(
        self,
        p: int = 97,
        operation: str = 'add',
        train_fraction: float = 0.5,
        seed: int = 42,
        device: str = 'cpu'
    ):
        self.p = p
        self.operation = operation
        self.seed = seed
        self.device = device
        
        # Load discrete version
        self.train_discrete, self.train_labels, self.test_discrete, self.test_labels = \
            create_modular_dataset_discrete(p, operation, train_fraction, seed, device)
        
        # Create one-hot versions
        self.train_onehot = discrete_to_onehot(self.train_discrete, p)
        self.test_onehot = discrete_to_onehot(self.test_discrete, p)
        
        # Create transformer sequence versions
        self.train_tokens = create_transformer_tokens(self.train_discrete, p)
        self.test_tokens = create_transformer_tokens(self.test_discrete, p)
    
    def get_train_data(self, format: str = 'discrete'):
        """
        Get training data in specified format.
        
        Args:
            format: 'discrete', 'onehot', or 'transformer'
            
        Returns:
            data, labels tuple
        """
        if format == 'discrete':
            return self.train_discrete, self.train_labels
        elif format == 'onehot':
            return self.train_onehot, self.train_labels
        elif format == 'transformer':
            return self.train_tokens, self.train_labels
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def get_test_data(self, format: str = 'discrete'):
        """Get test data in specified format."""
        if format == 'discrete':
            return self.test_discrete, self.test_labels
        elif format == 'onehot':
            return self.test_onehot, self.test_labels
        elif format == 'transformer':
            return self.test_tokens, self.test_labels
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def __repr__(self):
        return (f"ModularArithmeticDataset(p={self.p}, op='{self.operation}', "
                f"train={len(self.train_labels)}, test={len(self.test_labels)})")


def test_datasets():
    """Test dataset creation and conversions"""
    print("=" * 80)
    print("Testing Dataset Loaders")
    print("=" * 80)
    
    p = 97
    
    # Test discrete dataset
    print("\n1. Testing discrete token dataset...")
    train_d, train_l, test_d, test_l = create_modular_dataset_discrete(
        p=p, operation='add', train_fraction=0.5, seed=42
    )
    print(f"   Train: data shape={train_d.shape}, labels shape={train_l.shape}")
    print(f"   Test:  data shape={test_d.shape}, labels shape={test_l.shape}")
    print(f"   Total: {len(train_l) + len(test_l)} = {p}² = {p**2}")
    assert len(train_l) + len(test_l) == p ** 2
    print("   ✓ Discrete dataset test passed")
    
    # Test one-hot dataset
    print("\n2. Testing one-hot dataset...")
    train_oh, train_l2, test_oh, test_l2 = create_modular_dataset_onehot(
        p=p, operation='add', train_fraction=0.5, seed=42
    )
    print(f"   Train: data shape={train_oh.shape}, labels shape={train_l2.shape}")
    print(f"   Test:  data shape={test_oh.shape}, labels shape={test_l2.shape}")
    print(f"   One-hot sum (should be 2.0): {train_oh[0].sum().item()}")
    assert train_oh[0].sum().item() == 2.0
    print("   ✓ One-hot dataset test passed")
    
    # Test same split (same random seed)
    print("\n3. Testing reproducibility (same split with same seed)...")
    assert torch.equal(train_l, train_l2), "Labels should match!"
    print("   ✓ Same seed produces same split")
    
    # Test conversion functions
    print("\n4. Testing conversion functions...")
    onehot_conv = discrete_to_onehot(train_d[:10], p)
    discrete_conv = onehot_to_discrete(onehot_conv, p)
    assert torch.equal(train_d[:10], discrete_conv)
    print(f"   discrete -> onehot -> discrete: match={torch.equal(train_d[:10], discrete_conv)}")
    print("   ✓ Conversion test passed")
    
    # Test transformer tokens
    print("\n5. Testing transformer token format...")
    tokens = create_transformer_tokens(train_d[:5], p)
    print(f"   Input shape: {train_d[:5].shape} -> Token shape: {tokens.shape}")
    print(f"   Sample tokens: {tokens[0].tolist()}")
    assert tokens.shape == (5, 3)
    assert tokens[0, 2].item() == p  # Equals token
    print("   ✓ Transformer token test passed")
    
    # Test unified dataset class
    print("\n6. Testing ModularArithmeticDataset class...")
    dataset = ModularArithmeticDataset(p=p, operation='add', seed=42)
    print(f"   {dataset}")
    
    d_data, d_labels = dataset.get_train_data('discrete')
    oh_data, oh_labels = dataset.get_train_data('onehot')
    tf_data, tf_labels = dataset.get_train_data('transformer')
    
    print(f"   Discrete:    {d_data.shape}")
    print(f"   One-hot:     {oh_data.shape}")
    print(f"   Transformer: {tf_data.shape}")
    print("   ✓ Unified dataset class test passed")
    
    print("\n" + "=" * 80)
    print("✓ All dataset tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    test_datasets()

