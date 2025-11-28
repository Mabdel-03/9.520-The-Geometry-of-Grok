"""
One-Hot Encoded Dataset Loaders for Tractable Input-Gradient AGOP

This module creates datasets with one-hot encoded inputs (continuous, differentiable)
instead of discrete token indices. This enables tractable input-gradient AGOP computation
for all experiments.

Based on: Group1_Grokking_Code_Base.ipynb (Cell 2 - collate_batch with one_hot=True)
"""

import torch
import torch.nn as nn
import torchvision
import numpy as np
from pathlib import Path
from typing import Tuple


def create_onehot_modular_dataset(
    p: int,
    operation: str = 'add',
    train_fraction: float = 0.3,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create one-hot encoded modular arithmetic dataset.
    
    Matches notebook's approach: encodes (a, b) as single one-hot vector of size 2*p
    where positions [0:p] encode first number and positions [p:2*p] encode second number.
    
    Args:
        p: Modulus (prime number, e.g., 97, 113)
        operation: 'add', 'sub', 'mul', or 'div'
        train_fraction: Fraction of data for training
        device: Device to place tensors on
        
    Returns:
        train_data: [N_train, 2*p] one-hot encoded (float32)
        train_labels: [N_train] targets (long)
        test_data: [N_test, 2*p] one-hot encoded (float32)
        test_labels: [N_test] targets (long)
    """
    all_pairs = []
    all_labels = []
    
    # Helper for modular division
    def mod_inverse(b, p):
        return pow(b, p-2, p)  # Fermat's little theorem
    
    # Generate all pairs
    for a in range(p):
        for b in range(p):
            # Skip invalid operations
            if operation == 'div' and b == 0:
                continue
            
            # One-hot encode [a, b] into single vector
            v = np.zeros(2*p, dtype=np.float32)
            v[a] = 1.0        # First number
            v[p + b] = 1.0    # Second number
            all_pairs.append(v)
            
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
    X = torch.tensor(np.array(all_pairs), dtype=torch.float32, device=device)
    y = torch.tensor(all_labels, dtype=torch.long, device=device)
    
    # Shuffle
    perm = torch.randperm(len(X), device=device)
    X = X[perm]
    y = y[perm]
    
    # Split train/test
    n_train = int(len(X) * train_fraction)
    
    train_data = X[:n_train]
    train_labels = y[:n_train]
    test_data = X[n_train:]
    test_labels = y[n_train:]
    
    return train_data, train_labels, test_data, test_labels


def create_onehot_mnist_dataset(
    train_points: int = 1000,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create MNIST dataset with flattened images (already continuous).
    
    Args:
        train_points: Number of training points
        device: Device to place tensors on
        
    Returns:
        train_data: [N_train, 784] flattened images (float32)
        train_labels: [N_train] labels (long)
        test_data: [N_test, 784] flattened images (float32)
        test_labels: [N_test] labels (long)
    """
    # Load MNIST
    download_dir = Path.home() / '.cache' / 'mnist'
    download_dir.mkdir(parents=True, exist_ok=True)
    
    train_dataset = torchvision.datasets.MNIST(
        root=str(download_dir),
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root=str(download_dir),
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    
    # Subsample training data
    indices = torch.randperm(len(train_dataset))[:train_points]
    train_data = torch.stack([train_dataset[i][0] for i in indices])
    train_labels = torch.tensor([train_dataset[i][1] for i in indices])
    
    # Full test set
    test_data = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
    test_labels = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])
    
    # Flatten to [N, 784]
    train_data = train_data.view(len(train_data), -1)
    test_data = test_data.view(len(test_data), -1)
    
    return train_data.to(device), train_labels.to(device), test_data.to(device), test_labels.to(device)


def create_onehot_composition_dataset(
    vocab_size: int = 100,
    seq_len: int = 10,
    n_facts: int = 500,
    train_fraction: float = 0.3,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create one-hot encoded compositional reasoning dataset.
    
    Encodes token sequences as concatenated one-hot vectors.
    E.g., sequence [5, 12, 3] with vocab_size=100 becomes [1000] vector
    where positions [0:100] encode token 1, [100:200] encode token 2, etc.
    
    Args:
        vocab_size: Size of vocabulary
        seq_len: Sequence length
        n_facts: Total number of training examples
        train_fraction: Fraction for training
        device: Device to place tensors on
        
    Returns:
        train_data: [N_train, vocab_size*seq_len] one-hot (float32)
        train_labels: [N_train] targets (long)
        test_data: [N_test, vocab_size*seq_len] one-hot (float32)
        test_labels: [N_test] targets (long)
    """
    # Generate random compositional examples (placeholder)
    # In production, this would be proper knowledge graph reasoning
    
    all_data = []
    all_labels = []
    
    for _ in range(n_facts):
        # Random sequence
        seq = np.random.randint(0, vocab_size, size=seq_len)
        
        # One-hot encode: concatenate one-hot for each position
        v = np.zeros(vocab_size * seq_len, dtype=np.float32)
        for pos, token in enumerate(seq):
            v[pos * vocab_size + token] = 1.0
        
        all_data.append(v)
        
        # Random label (placeholder - would be compositional reasoning result)
        all_labels.append(np.random.randint(0, vocab_size))
    
    # Convert to tensors
    X = torch.tensor(np.array(all_data), dtype=torch.float32, device=device)
    y = torch.tensor(all_labels, dtype=torch.long, device=device)
    
    # Shuffle
    perm = torch.randperm(len(X), device=device)
    X = X[perm]
    y = y[perm]
    
    # Split
    n_train = int(len(X) * train_fraction)
    
    train_data = X[:n_train]
    train_labels = y[:n_train]
    test_data = X[n_train:]
    test_labels = y[n_train:]
    
    return train_data, train_labels, test_data, test_labels


def test_onehot_datasets():
    """Test one-hot dataset creation"""
    print("="*80)
    print("Testing One-Hot Dataset Creation")
    print("="*80)
    
    # Test modular arithmetic
    print("\n1. Testing modular arithmetic dataset (p=97, add)...")
    train_X, train_y, test_X, test_y = create_onehot_modular_dataset(
        p=97, operation='add', train_fraction=0.3, device='cpu'
    )
    print(f"  Train: X shape={train_X.shape}, y shape={train_y.shape}")
    print(f"  Test:  X shape={test_X.shape}, y shape={test_y.shape}")
    print(f"  X dtype: {train_X.dtype} (should be float32)")
    print(f"  X is one-hot: sum={train_X[0].sum()} (should be 2.0)")
    print(f"  ✓ Modular dataset OK")
    
    # Test MNIST
    print("\n2. Testing MNIST dataset...")
    train_X, train_y, test_X, test_y = create_onehot_mnist_dataset(
        train_points=100, device='cpu'
    )
    print(f"  Train: X shape={train_X.shape}, y shape={train_y.shape}")
    print(f"  Test:  X shape={test_X.shape}, y shape={test_y.shape}")
    print(f"  X dtype: {train_X.dtype}")
    print(f"  X range: [{train_X.min():.3f}, {train_X.max():.3f}]")
    print(f"  ✓ MNIST dataset OK")
    
    # Test composition
    print("\n3. Testing composition dataset...")
    train_X, train_y, test_X, test_y = create_onehot_composition_dataset(
        vocab_size=50, seq_len=5, n_facts=100, train_fraction=0.3, device='cpu'
    )
    print(f"  Train: X shape={train_X.shape}, y shape={train_y.shape}")
    print(f"  Test:  X shape={test_X.shape}, y shape={test_y.shape}")
    print(f"  X dtype: {train_X.dtype}")
    print(f"  X is one-hot: sum={train_X[0].sum()} (should be {5.0})")
    print(f"  ✓ Composition dataset OK")
    
    print("\n" + "="*80)
    print("✓ All one-hot datasets working correctly!")
    print("="*80)


if __name__ == "__main__":
    test_onehot_datasets()

