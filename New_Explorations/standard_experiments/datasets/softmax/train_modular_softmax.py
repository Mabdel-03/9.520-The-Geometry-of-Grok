"""
Training script for Modular Addition with Standard Softmax Transformer
Compatible with Muon optimizer (as in Tveit et al. 2025)
"""

import sys
from pathlib import Path
import argparse
import torch
import random
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'framework'))

from model_softmax import StandardTransformer, create_modular_addition_dataset
from trainer import GrokkingTrainer


def main():
    parser = argparse.ArgumentParser(description='Train Softmax Transformer with different optimizers')
    
    # Model hyperparameters
    parser.add_argument('--p', type=int, default=97, help='Modulus for modular addition')
    parser.add_argument('--train_fraction', type=float, default=0.5, help='Fraction of data for training')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--d_ff', type=int, default=512, help='FFN dimension')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    
    # Optimizer settings
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['muon', 'muonw', 'adam', 'adamw', 'sgd'],
                       help='Optimizer to use')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (default: optimizer-specific)')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
    
    # Training settings
    parser.add_argument('--n_epochs', type=int, default=50000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (None=full batch)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    
    # Logging settings
    parser.add_argument('--log_freq', type=int, default=100, help='Logging frequency')
    parser.add_argument('--checkpoint_freq', type=int, default=1000, help='Checkpoint frequency')
    parser.add_argument('--save_dir', type=str, default='./results/paper03_softmax',
                       help='Directory to save results')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Set optimizer-specific learning rates if not provided
    if args.lr is None:
        lr_defaults = {
            'adamw': 1e-3,
            'adam': 1e-3,
            'muon': 0.02,   # Official Muon default from modded-nanogpt
            'muonw': 0.02,  # Official Muon default
            'sgd': 1e-2     # 10× higher for SGD
        }
        args.lr = lr_defaults[args.optimizer]
    
    # Set random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Auto-generate experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = f'softmax_{args.optimizer}_p{args.p}_wd{args.weight_decay}'
    
    print("="*80)
    print("Modular Addition - Standard Softmax Transformer (Muon-compatible)")
    print("="*80)
    print(f"Configuration:")
    print(f"  Modulus p: {args.p}")
    print(f"  Train fraction: {args.train_fraction}")
    print(f"  Architecture: Softmax Transformer")
    print(f"    - d_model={args.d_model}, n_heads={args.n_heads}, n_layers={args.n_layers}")
    print(f"    - Softmax attention + LayerNorm + Residuals")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Epochs: {args.n_epochs}")
    print(f"  Seed: {args.seed}")
    print("="*80)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    print(f"\nCreating modular addition dataset (p={args.p})...")
    train_data, train_labels, test_data, test_labels = create_modular_addition_dataset(
        args.p, args.train_fraction, device
    )
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    
    # Create model
    print(f"\nCreating Standard Softmax Transformer...")
    model = StandardTransformer(
        p=args.p,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout
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
        compute_spectral_metrics=False,  # Disable AGOP for now
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
    
    # Check for grokking
    if history['test_acc'][-1] > 0.95 and history['train_acc'][-1] > 0.99:
        # Find grokking point
        test_acc = history['test_acc']
        epochs = history['epoch']
        for i, acc in enumerate(test_acc):
            if acc > 0.95:
                print(f"🎉 GROKKING at epoch {epochs[i]}!")
                break
    
    print(f"Results saved to: {trainer.save_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

