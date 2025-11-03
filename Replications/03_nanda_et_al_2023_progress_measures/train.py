"""
Training script for 1-Layer ReLU Transformer on Modular Addition
Replicating Nanda et al. (2023) - Progress Measures for Grokking
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import argparse
from pathlib import Path

from model import OneLayerReLUTransformer, create_modular_addition_dataset


def train_model(
    p=113,
    train_fraction=0.3,
    d_model=128,
    n_heads=4,
    d_mlp=512,
    lr=0.001,
    weight_decay=1.0,
    n_epochs=40000,
    log_interval=100,
    save_dir='./checkpoints',
    device='cuda'
):
    """
    Train the model and track grokking phenomenon
    """
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logs directory
    log_dir = Path('./logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    print(f"Creating modular addition dataset (p={p}, train_fraction={train_fraction})")
    train_data, train_labels, test_data, test_labels = create_modular_addition_dataset(
        p, train_fraction, device
    )
    
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    
    # Create model
    model = OneLayerReLUTransformer(p, d_model, n_heads, d_mlp).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    # Training loop
    print("Starting training...")
    for epoch in range(n_epochs):
        model.train()
        
        # Forward pass on full training set (full batch as in paper)
        logits = model(train_data)
        train_loss = criterion(logits, train_labels)
        
        # Backward pass
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        # Compute training accuracy
        train_preds = logits.argmax(dim=-1)
        train_acc = (train_preds == train_labels).float().mean().item()
        
        # Evaluate on test set
        if epoch % log_interval == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                test_logits = model(test_data)
                test_loss = criterion(test_logits, test_labels)
                test_preds = test_logits.argmax(dim=-1)
                test_acc = (test_preds == test_labels).float().mean().item()
            
            # Log metrics
            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss.item())
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss.item())
            history['test_acc'].append(test_acc)
            
            print(f"Epoch {epoch:5d} | Train Loss: {train_loss.item():.4f} | "
                  f"Train Acc: {train_acc:.4f} | Test Loss: {test_loss.item():.4f} | "
                  f"Test Acc: {test_acc:.4f}")
            
            # Save checkpoint
            if epoch % 1000 == 0 or epoch == n_epochs - 1:
                checkpoint_path = save_dir / f'checkpoint_epoch_{epoch}.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss.item(),
                    'test_loss': test_loss.item(),
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                }, checkpoint_path)
    
    # Save training history
    history_path = log_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining complete! History saved to {history_path}")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train 1-Layer ReLU Transformer on Modular Addition')
    parser.add_argument('--p', type=int, default=113, help='Modulus (prime number)')
    parser.add_argument('--train_fraction', type=float, default=0.3, help='Fraction of data for training')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--d_mlp', type=int, default=512, help='MLP hidden dimension')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1.0, help='Weight decay')
    parser.add_argument('--n_epochs', type=int, default=40000, help='Number of epochs')
    parser.add_argument('--log_interval', type=int, default=100, help='Logging interval')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Train model
    model, history = train_model(
        p=args.p,
        train_fraction=args.train_fraction,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp,
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_epochs=args.n_epochs,
        log_interval=args.log_interval,
        save_dir=args.save_dir,
        device=args.device
    )


if __name__ == '__main__':
    main()

