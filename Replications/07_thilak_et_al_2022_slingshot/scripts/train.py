"""
Training script for Slingshot Mechanism experiments
Thilak et al. (2022)

Focus on tracking last-layer weight norms and optimizer dynamics
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import argparse
from pathlib import Path

from model import ModularArithmeticTransformer, create_modular_addition_dataset


def train_model(
    p=97,
    train_fraction=0.5,
    d_model=128,
    n_heads=4,
    n_layers=2,
    d_mlp=512,
    optimizer_name='adam',
    lr=0.001,
    weight_decay=0.0,  # Slingshot can occur even without weight decay
    n_epochs=50000,
    log_interval=50,
    save_dir='./checkpoints',
    device='cuda',
    seed=42
):
    """
    Train model and track Slingshot dynamics
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
    print(f"Creating modular addition dataset (p={p}, train_fraction={train_fraction})")
    train_data, train_labels, test_data, test_labels = create_modular_addition_dataset(
        p, train_fraction, device
    )
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    
    # Create model
    model = ModularArithmeticTransformer(p, d_model, n_heads, n_layers, d_mlp).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Optimizer
    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    print(f"Optimizer: {optimizer_name}, lr={lr}, weight_decay={weight_decay}")
    
    criterion = nn.CrossEntropyLoss()
    
    # Training history with Slingshot-specific metrics
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'last_layer_norm': [],  # Track last layer weight norm (Slingshot indicator)
        'embedding_norm': [],
        'transformer_norm': [],
        'output_norm': []
    }
    
    # Training loop
    print("Starting training...")
    print("Tracking last-layer weight norms for Slingshot detection...")
    
    for epoch in range(n_epochs):
        model.train()
        
        # Full batch gradient descent
        logits = model(train_data)
        train_loss = criterion(logits, train_labels)
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        train_preds = logits.argmax(dim=-1)
        train_acc = (train_preds == train_labels).float().mean().item()
        
        # Log metrics
        if epoch % log_interval == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                test_logits = model(test_data)
                test_loss = criterion(test_logits, test_labels)
                test_preds = test_logits.argmax(dim=-1)
                test_acc = (test_preds == test_labels).float().mean().item()
                
                # Track weight norms (Slingshot indicators)
                last_layer_norm = model.get_last_layer_norm()
                param_norms = model.get_all_parameter_norms()
            
            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss.item())
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss.item())
            history['test_acc'].append(test_acc)
            history['last_layer_norm'].append(last_layer_norm)
            history['embedding_norm'].append(param_norms['embeddings'])
            history['transformer_norm'].append(param_norms['transformers'])
            history['output_norm'].append(param_norms['output'])
            
            print(f"Epoch {epoch:5d} | Train Loss: {train_loss.item():.4f} | "
                  f"Train Acc: {train_acc:.4f} | Test Loss: {test_loss.item():.4f} | "
                  f"Test Acc: {test_acc:.4f} | Last Layer Norm: {last_layer_norm:.2f}")
            
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
                    'last_layer_norm': last_layer_norm,
                }, checkpoint_path)
    
    # Save training history
    history_path = log_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining complete! History saved to {history_path}")
    print("\nAnalyze 'last_layer_norm' for cyclic behavior indicating Slingshot mechanism")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train model to observe Slingshot mechanism')
    parser.add_argument('--p', type=int, default=97, help='Modulus (prime)')
    parser.add_argument('--train_fraction', type=float, default=0.5, help='Training fraction')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--d_mlp', type=int, default=512, help='MLP dimension')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'],
                       help='Optimizer (Adam family for Slingshot)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='Weight decay (Slingshot can occur even with 0)')
    parser.add_argument('--n_epochs', type=int, default=50000, help='Number of epochs')
    parser.add_argument('--log_interval', type=int, default=50, help='Logging interval')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Save directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    train_model(
        p=args.p,
        train_fraction=args.train_fraction,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_mlp=args.d_mlp,
        optimizer_name=args.optimizer,
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

