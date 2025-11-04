"""
Effective Theory of Representation Learning - Simplified Training
Liu et al. (2022)

Simplified version without Sacred framework, saves to training_history.json
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from pathlib import Path
import argparse


class ToyModel(nn.Module):
    """
    Simple encoder-decoder model for toy representation learning
    Learns sparse one-hot representations
    """
    def __init__(self, p, d_hidden=200):
        super().__init__()
        self.p = p
        
        # Encoder: maps input to representation
        self.encoder = nn.Linear(p, d_hidden, bias=False)
        
        # Decoder: maps representation back to output
        self.decoder = nn.Linear(d_hidden, p, bias=False)
        
    def forward(self, x):
        """
        x: one-hot encoded input (batch_size, p)
        """
        h = self.encoder(x)
        y = self.decoder(h)
        return y, h


def create_modular_addition_data(p, train_fraction=0.45):
    """Create modular addition dataset"""
    all_data = []
    all_labels = []
    
    for a in range(p):
        for b in range(p):
            # Input: concatenate one-hot vectors for a and b
            input_vec = torch.zeros(2 * p)
            input_vec[a] = 1.0
            input_vec[p + b] = 1.0
            
            # Output: one-hot for (a + b) mod p
            output_vec = torch.zeros(p)
            output_vec[(a + b) % p] = 1.0
            
            all_data.append(input_vec)
            all_labels.append(output_vec)
    
    all_data = torch.stack(all_data)
    all_labels = torch.stack(all_labels)
    
    # Split into train/test
    n_total = len(all_data)
    n_train = int(n_total * train_fraction)
    
    perm = torch.randperm(n_total)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    
    return (all_data[train_idx], all_labels[train_idx],
            all_data[test_idx], all_labels[test_idx])


def compute_accuracy(model, data, labels, device):
    """Compute accuracy on dataset"""
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        labels = labels.to(device)
        
        outputs, _ = model(data)
        preds = torch.argmax(outputs, dim=1)
        targets = torch.argmax(labels, dim=1)
        acc = (preds == targets).float().mean().item()
    
    model.train()
    return acc


def train_model(
    p=10,
    d_hidden=200,
    train_fraction=0.45,
    train_steps=50000,
    encoder_lr=1e-3,
    decoder_lr=1e-3,
    encoder_weight_decay=1.0,
    decoder_weight_decay=1.0,
    log_interval=100,
    device='cuda',
    seed=0
):
    """Train toy model for representation learning"""
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    print(f"Creating modular addition dataset (p={p}, train_fraction={train_fraction})")
    train_data, train_labels, test_data, test_labels = create_modular_addition_data(
        p, train_fraction
    )
    
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    
    # Create model
    model = ToyModel(2 * p, d_hidden).to(device)  # Input is 2p dimensional (two one-hot vectors)
    
    # Create separate optimizers for encoder and decoder
    encoder_params = model.encoder.parameters()
    decoder_params = model.decoder.parameters()
    
    encoder_optimizer = optim.AdamW(encoder_params, lr=encoder_lr, weight_decay=encoder_weight_decay)
    decoder_optimizer = optim.AdamW(decoder_params, lr=decoder_lr, weight_decay=decoder_weight_decay)
    
    criterion = nn.MSELoss()
    
    # Training history
    history = {
        'step': [],
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    # Move data to device
    train_data = train_data.to(device)
    train_labels = train_labels.to(device)
    test_data = test_data.to(device)
    test_labels = test_labels.to(device)
    
    # Training loop
    print("Starting training...")
    model.train()
    
    for step in range(train_steps):
        # Full batch training
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        
        outputs, representations = model(train_data)
        loss = criterion(outputs, train_labels)
        
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        
        # Log metrics
        if step % log_interval == 0 or step == train_steps - 1:
            train_loss = loss.item()
            train_acc = compute_accuracy(model, train_data, train_labels, device)
            
            with torch.no_grad():
                test_outputs, _ = model(test_data)
                test_loss = criterion(test_outputs, test_labels).item()
            
            test_acc = compute_accuracy(model, test_data, test_labels, device)
            
            history['step'].append(step)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            
            print(f"Step {step:5d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    
    # Save training history
    log_dir = Path('./logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    history_path = log_dir / 'training_history.json'
    
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining complete! History saved to {history_path}")
    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=int, default=10, help='Modulus (small prime)')
    parser.add_argument('--d_hidden', type=int, default=200, help='Hidden dimension')
    parser.add_argument('--train_fraction', type=float, default=0.45, help='Training fraction')
    parser.add_argument('--train_steps', type=int, default=50000, help='Training steps')
    parser.add_argument('--encoder_lr', type=float, default=1e-3, help='Encoder learning rate')
    parser.add_argument('--decoder_lr', type=float, default=1e-3, help='Decoder learning rate')
    parser.add_argument('--encoder_wd', type=float, default=1.0, help='Encoder weight decay')
    parser.add_argument('--decoder_wd', type=float, default=1.0, help='Decoder weight decay')
    parser.add_argument('--log_interval', type=int, default=100, help='Logging interval')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    
    args = parser.parse_args()
    
    model, history = train_model(
        p=args.p,
        d_hidden=args.d_hidden,
        train_fraction=args.train_fraction,
        train_steps=args.train_steps,
        encoder_lr=args.encoder_lr,
        decoder_lr=args.decoder_lr,
        encoder_weight_decay=args.encoder_wd,
        decoder_weight_decay=args.decoder_wd,
        log_interval=args.log_interval,
        device=args.device,
        seed=args.seed
    )

