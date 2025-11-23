"""
Training script for Paper 5: Liu et al. (2022) - Omnigrok
MNIST Grokking with different optimizers and spectral metrics tracking
"""

import sys
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torchvision
import numpy as np
import random

# Add framework path
sys.path.insert(0, str(Path(__file__).parent.parent / 'framework'))
from trainer import GrokkingTrainer


class MNISTModel(nn.Module):
    """
    Simple MLP for MNIST classification (from Omnigrok paper)
    
    Architecture:
    - depth: 3 layers
    - width: 200 hidden units
    - activation: ReLU
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
        
        # Select activation function
        if activation.lower() == 'relu':
            activation_fn = nn.ReLU
        elif activation.lower() == 'tanh':
            activation_fn = nn.Tanh
        elif activation.lower() == 'gelu':
            activation_fn = nn.GELU
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build layers
        layers = [nn.Flatten()]
        
        for i in range(depth):
            if i == 0:
                # Input layer
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(activation_fn())
            elif i == depth - 1:
                # Output layer
                layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                # Hidden layers
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(activation_fn())
        
        self.net = nn.Sequential(*layers)
        
        # Initialize with scaled weights (as in Omnigrok)
        with torch.no_grad():
            for p in self.parameters():
                p.data = initialization_scale * p.data
    
    def forward(self, x):
        return self.net(x)


def create_mnist_datasets(train_points: int = 1000, device: str = 'cuda'):
    """
    Create MNIST training and test datasets.
    
    Args:
        train_points: Number of training points (subset of full training set)
        device: Device to place tensors on
        
    Returns:
        train_data, train_labels, test_data, test_labels
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
    
    # Subset training data
    train_subset = torch.utils.data.Subset(train_dataset, range(train_points))
    
    # Convert to tensors
    train_data = []
    train_labels = []
    for img, label in train_subset:
        train_data.append(img)
        train_labels.append(label)
    
    test_data = []
    test_labels = []
    for img, label in test_dataset:
        test_data.append(img)
        test_labels.append(label)
    
    # Stack and move to device
    train_data = torch.stack(train_data).to(device)
    train_labels = torch.tensor(train_labels).to(device)
    test_data = torch.stack(test_data).to(device)
    test_labels = torch.tensor(test_labels).to(device)
    
    return train_data, train_labels, test_data, test_labels


def main():
    parser = argparse.ArgumentParser(description='Train MNIST model with different optimizers')
    
    # Model hyperparameters (from Omnigrok paper)
    parser.add_argument('--train_points', type=int, default=1000,
                       help='Number of training points')
    parser.add_argument('--hidden_dim', type=int, default=200,
                       help='Hidden layer dimension')
    parser.add_argument('--depth', type=int, default=3,
                       help='Number of layers')
    parser.add_argument('--activation', type=str, default='relu',
                       choices=['relu', 'tanh', 'gelu'],
                       help='Activation function')
    parser.add_argument('--init_scale', type=float, default=8.0,
                       help='Weight initialization scale')
    
    # Optimizer settings
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['muon', 'muonw', 'adam', 'adamw', 'sgd'],
                       help='Optimizer to use')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    
    # Training settings
    parser.add_argument('--n_epochs', type=int, default=100000,
                       help='Number of optimization steps')
    parser.add_argument('--batch_size', type=int, default=200,
                       help='Batch size (None=full batch)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    
    # Spectral metrics settings
    parser.add_argument('--spectral_metrics', action='store_true', default=False,
                       help='Compute spectral metrics (requires large memory!)')
    parser.add_argument('--spectral_freq', type=int, default=100,
                       help='Frequency of spectral metrics computation')
    parser.add_argument('--spectral_top_k', type=int, default=20,
                       help='Number of top eigenvalues to track')
    parser.add_argument('--compute_per_layer', action='store_true', default=False,
                       help='Compute per-layer spectral metrics')
    parser.add_argument('--agop_subsample_size', type=int, default=None,
                       help='Subsample size for AGOP computation (reduces memory)')
    
    # Logging settings
    parser.add_argument('--log_freq', type=int, default=100, help='Logging frequency')
    parser.add_argument('--checkpoint_freq', type=int, default=5000,
                       help='Checkpoint frequency')
    parser.add_argument('--save_dir', type=str, default='./results/paper05_omnigrok',
                       help='Directory to save results')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Auto-generate experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = f'mnist_{args.optimizer}_lr{args.lr}_wd{args.weight_decay}_n{args.train_points}'
    
    print("="*80)
    print("Paper 5: Liu et al. (2022) - Omnigrok MNIST Grokking")
    print("="*80)
    print(f"Configuration:")
    print(f"  Training points: {args.train_points}")
    print(f"  Architecture: depth={args.depth}, width={args.hidden_dim}, activation={args.activation}")
    print(f"  Initialization scale: {args.init_scale}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Optimization steps: {args.n_epochs}")
    print(f"  Spectral metrics: {args.spectral_metrics} (freq={args.spectral_freq})")
    print(f"  Seed: {args.seed}")
    print("="*80)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    print(f"\nCreating MNIST dataset ({args.train_points} training points)...")
    train_data, train_labels, test_data, test_labels = create_mnist_datasets(
        args.train_points, device
    )
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    
    # Create model
    print(f"\nCreating model...")
    model = MNISTModel(
        input_dim=784,
        hidden_dim=args.hidden_dim,
        output_dim=10,
        depth=args.depth,
        activation=args.activation,
        initialization_scale=args.init_scale
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {n_params:,} parameters")
    
    # Create trainer
    print(f"\nInitializing trainer...")
    trainer = GrokkingTrainer(
        model=model,
        train_data=train_data,
        train_labels=train_labels,
        test_data=test_data,
        test_labels=test_labels,
        optimizer_name=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        device=str(device),
        compute_spectral_metrics=args.spectral_metrics,
        spectral_metrics_freq=args.spectral_freq,
        spectral_top_k=args.spectral_top_k,
        compute_per_layer=args.compute_per_layer,
        agop_subsample_size=args.agop_subsample_size,
        log_freq=args.log_freq,
        save_dir=args.save_dir,
        experiment_name=args.experiment_name,
        checkpoint_freq=args.checkpoint_freq,
    )
    
    # Train
    print(f"\n{'='*80}")
    print("Starting training...")
    print(f"{'='*80}\n")
    
    history = trainer.train()
    
    print(f"\n{'='*80}")
    print("Training complete!")
    print(f"{'='*80}")
    print(f"Final train accuracy: {history['train_acc'][-1]:.4f}")
    print(f"Final test accuracy: {history['test_acc'][-1]:.4f}")
    print(f"Results saved to: {trainer.save_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

