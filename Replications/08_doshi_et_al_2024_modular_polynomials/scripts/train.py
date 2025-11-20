"""
Training script for Modular Polynomials with Power Activation
Doshi et al. (2024) - Grokking Modular Polynomials
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import argparse
from pathlib import Path

from model import (
    ModularAdditionMLP,
    ModularMultiplicationMLP,
    create_modular_addition_dataset,
    create_modular_multiplication_dataset
)


def train_model(
    task='addition',
    p=97,
    num_terms=2,
    hidden_dim=500,
    power=2,
    train_fraction=0.5,
    lr=0.005,
    weight_decay=5.0,
    n_epochs=50000,
    log_interval=100,
    save_dir='./checkpoints',
    device='cuda',
    seed=42
):
    """
    Train modular polynomial model
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create directories
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path('./logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    print(f"Creating modular {task} dataset (p={p})")
    if task == 'addition':
        train_data, train_labels, test_data, test_labels = create_modular_addition_dataset(
            p, num_terms, train_fraction
        )
        model = ModularAdditionMLP(p, num_terms, hidden_dim, power).to(device)
    elif task == 'multiplication':
        train_data, train_labels, test_data, test_labels = create_modular_multiplication_dataset(
            p, train_fraction
        )
        model = ModularMultiplicationMLP(p, hidden_dim).to(device)
    else:
        raise ValueError(f"Unknown task: {task}")
    
    train_data = train_data.to(device)
    train_labels = train_labels.to(device)
    test_data = test_data.to(device)
    test_labels = test_labels.to(device)
    
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    print(f"Model: {task}, Hidden dim: {hidden_dim}, Power: {power}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Optimizer (Adam with MSE loss as in paper)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()  # Paper uses MSE loss
    
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
        
        # Forward pass (full batch)
        logits = model(train_data)
        
        # Convert labels to one-hot for MSE loss
        train_labels_onehot = torch.zeros_like(logits)
        train_labels_onehot.scatter_(1, train_labels.unsqueeze(1), 1.0)
        
        train_loss = criterion(logits, train_labels_onehot)
        
        # Backward pass
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        # Compute accuracy
        train_preds = logits.argmax(dim=-1)
        train_acc = (train_preds == train_labels).float().mean().item()
        
        # Evaluate
        if epoch % log_interval == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                test_logits = model(test_data)
                
                test_labels_onehot = torch.zeros_like(test_logits)
                test_labels_onehot.scatter_(1, test_labels.unsqueeze(1), 1.0)
                
                test_loss = criterion(test_logits, test_labels_onehot)
                test_preds = test_logits.argmax(dim=-1)
                test_acc = (test_preds == test_labels).float().mean().item()
            
            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss.item())
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss.item())
            history['test_acc'].append(test_acc)
            
            print(f"Epoch {epoch:5d} | Train Loss: {train_loss.item():.4f} | "
                  f"Train Acc: {train_acc:.4f} | Test Loss: {test_loss.item():.4f} | "
                  f"Test Acc: {test_acc:.4f}")
            
            # Save checkpoints
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
    
    # Save history
    history_path = log_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining complete! History saved to {history_path}")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train modular polynomial models')
    parser.add_argument('--task', type=str, default='addition', choices=['addition', 'multiplication'],
                       help='Task type')
    parser.add_argument('--p', type=int, default=97, help='Modulus (prime)')
    parser.add_argument('--num_terms', type=int, default=2, help='Number of terms (for addition)')
    parser.add_argument('--hidden_dim', type=int, default=500, help='Hidden dimension (N)')
    parser.add_argument('--power', type=int, default=2, help='Power for activation function')
    parser.add_argument('--train_fraction', type=float, default=0.5, help='Training fraction')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5.0, help='Weight decay')
    parser.add_argument('--n_epochs', type=int, default=50000, help='Number of epochs')
    parser.add_argument('--log_interval', type=int, default=100, help='Logging interval')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Save directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    train_model(
        task=args.task,
        p=args.p,
        num_terms=args.num_terms,
        hidden_dim=args.hidden_dim,
        power=args.power,
        train_fraction=args.train_fraction,
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_epochs=args.n_epochs,
        log_interval=args.log_interval,
        save_dir=args.save_dir,
        device=args.device,
        seed=args.seed
    )


if __name__ == '__main__':
    main()

