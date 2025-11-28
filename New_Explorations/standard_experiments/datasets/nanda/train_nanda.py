"""
Training script for Paper 3: Nanda et al. (2023) - Progress Measures for Grokking
Modular Addition with different optimizers and spectral metrics tracking
"""

import sys
from pathlib import Path
import argparse
import torch

# Add paths
repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root / 'Replications' / '03_nanda_et_al_2023_progress_measures' / 'scripts'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'framework'))

from model import OneLayerReLUTransformer, create_modular_addition_dataset
from trainer import GrokkingTrainer


def main():
    parser = argparse.ArgumentParser(description='Train Nanda model with different optimizers')
    
    # Model hyperparameters (from paper)
    parser.add_argument('--p', type=int, default=113, help='Modulus for modular addition')
    parser.add_argument('--train_fraction', type=float, default=0.3, help='Fraction of data for training')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--d_mlp', type=int, default=512, help='MLP hidden dimension')
    
    # Optimizer settings
    parser.add_argument('--optimizer', type=str, default='adamw', 
                       choices=['muon', 'muonw', 'adam', 'adamw', 'sgd'],
                       help='Optimizer to use')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1.0, help='Weight decay')
    
    # Training settings
    parser.add_argument('--n_epochs', type=int, default=40000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (None=full batch)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
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
    parser.add_argument('--checkpoint_freq', type=int, default=1000, help='Checkpoint frequency')
    parser.add_argument('--save_dir', type=str, default='./results/paper03_nanda',
                       help='Directory to save results')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Auto-generate experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = f'nanda_{args.optimizer}_lr{args.lr}_wd{args.weight_decay}'
    
    print("="*80)
    print("Paper 3: Nanda et al. (2023) - Modular Addition Grokking")
    print("="*80)
    print(f"Configuration:")
    print(f"  Modulus p: {args.p}")
    print(f"  Train fraction: {args.train_fraction}")
    print(f"  Model: d_model={args.d_model}, n_heads={args.n_heads}, d_mlp={args.d_mlp}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Epochs: {args.n_epochs}")
    print(f"  Spectral metrics: {args.spectral_metrics} (freq={args.spectral_freq})")
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
    print(f"\nCreating model...")
    model = OneLayerReLUTransformer(
        args.p, 
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp
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

