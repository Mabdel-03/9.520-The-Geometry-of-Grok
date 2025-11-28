"""
Training script for MNIST Omnigrok with Input-Gradient AGOP tracking

Trains a simple MLP on MNIST with limited data to induce grokking, while tracking
input-gradient AGOP metrics. This provides comparison between perceptual (images)
and symbolic (modular arithmetic) grokking tasks.

Key features:
- Uses MSE loss with one-hot targets (Omnigrok setup)
- Limited training data to induce grokking
- Tracks input-gradient AGOP on flattened 784-dim images
"""

import sys
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
import json
import h5py
import time
from tqdm import tqdm

# Add paths
sys.path.insert(0, '/om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/standard_experiments/framework')
sys.path.insert(0, '/om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/agop_experiments/core')

from muon_official import Muon
from agop_utils import InputGradientAGOPTracker
from onehot_datasets import create_onehot_mnist_dataset
from onehot_models import MNISTModel


class MNISTModel(nn.Module):
    """Simple MLP for MNIST (from Omnigrok paper)"""
    def __init__(self, input_dim=784, hidden_dim=200, output_dim=10, depth=3, 
                 activation='relu', initialization_scale=8.0):
        super().__init__()
        if activation.lower() == 'relu':
            activation_fn = nn.ReLU
        elif activation.lower() == 'tanh':
            activation_fn = nn.Tanh
        elif activation.lower() == 'gelu':
            activation_fn = nn.GELU
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        layers = [nn.Flatten()]
        for i in range(depth):
            if i == 0:
                layers.extend([nn.Linear(input_dim, hidden_dim), activation_fn()])
            elif i == depth - 1:
                layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                layers.extend([nn.Linear(hidden_dim, hidden_dim), activation_fn()])
        
        self.net = nn.Sequential(*layers)
        
        with torch.no_grad():
            for p in self.parameters():
                p.data = initialization_scale * p.data
    
    def forward(self, x):
        return self.net(x)


# MNIST dataset loader moved to onehot_datasets.py for consistency


def train_with_agop_tracking(
    model, train_data, train_labels, test_data, test_labels,
    optimizer, n_epochs, device, agop_tracker, agop_freq, log_freq, save_dir,
):
    """Training loop with input-gradient AGOP tracking for MNIST (MSE loss)"""
    criterion = nn.MSELoss()
    one_hots = torch.eye(10, dtype=torch.float32, device=device)
    
    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    
    print(f"Starting training for {n_epochs} epochs (MSE loss)")
    print(f"AGOP tracking enabled: computing every {agop_freq} epochs")
    start_time = time.time()
    
    for epoch in tqdm(range(n_epochs), desc="Training"):
        model.train()
        optimizer.zero_grad()
        logits = model(train_data)
        targets = one_hots[train_labels]
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        train_loss = loss.item()
        preds = logits.argmax(dim=-1)
        train_acc = (preds == train_labels).float().mean().item()
        
        if epoch % log_freq == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                test_logits = model(test_data)
                test_targets = one_hots[test_labels]
                test_loss = criterion(test_logits, test_targets).item()
                test_preds = test_logits.argmax(dim=-1)
                test_acc = (test_preds == test_labels).float().mean().item()
            
            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            
            if epoch % agop_freq == 0:
                print(f"\n  Computing AGOP at epoch {epoch}...", end=' ', flush=True)
                agop_start = time.time()
                # Use CrossEntropyLoss for AGOP computation (more meaningful gradients)
                ce_criterion = nn.CrossEntropyLoss()
                agop = agop_tracker.compute_input_agop(model, train_data, train_labels, ce_criterion)
                
                if agop is not None:
                    metrics = agop_tracker.compute_agop_metrics(history, agop)
                    agop_time = time.time() - agop_start
                    print(f"Done ({agop_time:.1f}s)")
                    print(f"    Trace: {metrics.get('agop_trace', 0):.4e}, "
                          f"Eigengap: {metrics.get('agop_eigengap', 0):.4e}, "
                          f"VCR: {metrics.get('agop_variation_collapse_ratio', 0):.4f}")
                else:
                    print("Failed")
            
            if epoch % (log_freq * 10) == 0:
                print(f"\nEpoch {epoch}/{n_epochs}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
    
    print(f"\nTraining completed in {time.time() - start_time:.2f} seconds")
    
    # Save results
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    history_json = {k: v for k, v in history.items() if k != 'agop_topk_subspace_prev'}
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history_json, f, indent=2)
    
    agop_keys = [k for k in history.keys() if k.startswith('agop_') and k != 'agop_topk_subspace_prev']
    if agop_keys:
        with h5py.File(save_dir / 'agop_metrics.h5', 'w') as f:
            for key in agop_keys:
                f.create_dataset(key, data=np.array(history[key]), compression='gzip')
            agop_epochs = history['epoch'][::agop_freq] if agop_freq > 1 else history['epoch']
            f.create_dataset('epoch', data=np.array(agop_epochs[:len(history[agop_keys[0]])]), compression='gzip')
    
    print(f"Results saved to {save_dir}")
    return history


def main():
    parser = argparse.ArgumentParser(description='Train MNIST MLP with Input-Gradient AGOP tracking')
    parser.add_argument('--train_points', type=int, default=1000, help='Number of training points')
    parser.add_argument('--hidden_dim', type=int, default=200, help='Hidden dimension')
    parser.add_argument('--depth', type=int, default=3, help='Network depth')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'tanh', 'gelu'])
    parser.add_argument('--init_scale', type=float, default=8.0, help='Initialization scale')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['muon', 'adam', 'adamw', 'sgd'])
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--n_epochs', type=int, default=50000, help='Number of epochs')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--agop_freq', type=int, default=100, help='AGOP computation frequency')
    parser.add_argument('--agop_subsample', type=int, default=500, help='Subsample size for AGOP (images are 784-dim)')
    parser.add_argument('--agop_top_k', type=int, default=20, help='Top eigenvectors to track')
    parser.add_argument('--log_freq', type=int, default=100, help='Logging frequency')
    parser.add_argument('--save_dir', type=str, default='./results/agop_experiments/mnist')
    parser.add_argument('--experiment_name', type=str, default=None)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    if args.experiment_name is None:
        args.experiment_name = f'mnist_{args.optimizer}_n{args.train_points}_lr{args.lr}_wd{args.weight_decay}_seed{args.seed}'
    
    print("="*80)
    print("MNIST Omnigrok with Input-Gradient AGOP")
    print("="*80)
    print(f"Training points: {args.train_points}, Hidden dim: {args.hidden_dim}, Depth: {args.depth}")
    print(f"Optimizer: {args.optimizer}, lr={args.lr}, wd={args.weight_decay}")
    print(f"AGOP: freq={args.agop_freq}, top_k={args.agop_top_k}, subsample={args.agop_subsample}")
    print("="*80)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_data, train_labels, test_data, test_labels = create_onehot_mnist_dataset(args.train_points, device)
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    print(f"Input dim: {train_data.shape[1]} (784 pixels, flattened)")
    print(f"Input dtype: {train_data.dtype} (continuous for AGOP)")
    
    model = MNISTModel(input_dim=784, hidden_dim=args.hidden_dim, output_dim=10, 
                      depth=args.depth, activation=args.activation, 
                      initialization_scale=args.init_scale).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    if args.optimizer == 'muon':
        optimizer = Muon(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.95, use_nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    
    agop_tracker = InputGradientAGOPTracker(top_k=args.agop_top_k, subsample_size=args.agop_subsample,
                                            device=str(device), agop_device='cpu', use_mse_loss=False)
    
    save_dir = Path(args.save_dir) / args.experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    history = train_with_agop_tracking(model, train_data, train_labels, test_data, test_labels,
                                      optimizer, args.n_epochs, device, agop_tracker,
                                      args.agop_freq, args.log_freq, save_dir)
    
    print(f"\n{'='*80}")
    print(f"Final: Train Acc={history['train_acc'][-1]:.4f}, Test Acc={history['test_acc'][-1]:.4f}")
    if history['test_acc'][-1] > 0.95:
        for i, acc in enumerate(history['test_acc']):
            if acc > 0.95:
                print(f"🎉 GROKKING at epoch {history['epoch'][i]}!")
                break
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

