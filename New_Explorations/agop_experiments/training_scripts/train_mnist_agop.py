"""
Training script for MNIST Omnigrok with Input-Gradient AGOP and Lazy-Rich tracking

Trains a simple MLP on MNIST with limited data to induce grokking, while tracking:
1. Input-gradient AGOP metrics (gradient geometry in input space)
2. Lazy-Rich dynamics (NTK evolution, weight norms, feature kernels)

Key features:
- Uses MSE loss with one-hot targets (Omnigrok setup)
- Limited training data to induce grokking
- Tracks input-gradient AGOP on flattened 784-dim images
- Tracks NTK distance from initialization (lazy→rich transition)
- Tracks weight norm evolution

Reference for Lazy-Rich metrics: Kumar et al. (2024) 
"Grokking as the Transition from Lazy to Rich Training Dynamics"
https://arxiv.org/abs/2310.06110
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
from lazy_rich_utils import LazyRichTracker, compute_weight_norms
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


def train_with_tracking(
    model, train_data, train_labels, test_data, test_labels,
    optimizer, n_epochs, device, 
    agop_tracker, lazy_rich_tracker,
    agop_freq, log_freq, save_dir,
    compute_ntk=True, compute_feature_kernel=True
):
    """Training loop with input-gradient AGOP and Lazy-Rich tracking for MNIST (MSE loss)"""
    criterion = nn.MSELoss()
    one_hots = torch.eye(10, dtype=torch.float32, device=device)
    
    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    lazy_rich_history = {}  # Separate history for lazy-rich metrics (different frequency possible)
    
    print(f"Starting training for {n_epochs} epochs (MSE loss)")
    print(f"AGOP tracking enabled: computing every {agop_freq} epochs")
    print(f"Lazy-Rich tracking: NTK={compute_ntk}, FeatureKernel={compute_feature_kernel}")
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
            
            # Always track weight norms (cheap)
            weight_norms = compute_weight_norms(model)
            if 'weight_norm_total' not in history:
                history['weight_norm_total'] = []
            history['weight_norm_total'].append(weight_norms['total'])
            
            if epoch % agop_freq == 0:
                print(f"\n  Epoch {epoch}: Computing metrics...", flush=True)
                
                # Compute AGOP
                print(f"    AGOP...", end=' ', flush=True)
                agop_start = time.time()
                # Use CrossEntropyLoss for AGOP computation (more meaningful gradients)
                ce_criterion = nn.CrossEntropyLoss()
                agop = agop_tracker.compute_input_agop(model, train_data, train_labels, ce_criterion)
                
                if agop is not None:
                    metrics = agop_tracker.compute_agop_metrics(history, agop)
                    agop_time = time.time() - agop_start
                    print(f"Done ({agop_time:.1f}s)")
                    print(f"      Trace: {metrics.get('agop_trace', 0):.4e}, "
                          f"Eigengap: {metrics.get('agop_eigengap', 0):.4e}, "
                          f"VCR: {metrics.get('agop_variation_collapse_ratio', 0):.4f}")
                else:
                    print("Failed")
                
                # Compute Lazy-Rich metrics
                print(f"    Lazy-Rich metrics...", end=' ', flush=True)
                lr_start = time.time()
                
                # Track epoch for lazy-rich metrics
                if 'epoch' not in lazy_rich_history:
                    lazy_rich_history['epoch'] = []
                lazy_rich_history['epoch'].append(epoch)
                
                lr_metrics = lazy_rich_tracker.compute_metrics(
                    model, train_data, lazy_rich_history,
                    compute_ntk=compute_ntk,
                    compute_feature_kernel_dist=compute_feature_kernel
                )
                lr_time = time.time() - lr_start
                print(f"Done ({lr_time:.1f}s)")
                
                if 'ntk_distance' in lr_metrics:
                    print(f"      NTK Distance: {lr_metrics['ntk_distance']:.4e}, "
                          f"Weight Norm: {lr_metrics['weight_norm_total']:.4f}")
                if 'feature_kernel_distance' in lr_metrics:
                    print(f"      Feature Kernel Distance: {lr_metrics['feature_kernel_distance']:.4e}")
            
            if epoch % (log_freq * 10) == 0:
                print(f"\nEpoch {epoch}/{n_epochs}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
    
    print(f"\nTraining completed in {time.time() - start_time:.2f} seconds")
    
    # Save results
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training history (basic metrics + weight norms)
    history_json = {k: v for k, v in history.items() 
                   if k != 'agop_topk_subspace_prev' and not k.startswith('agop_')}
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history_json, f, indent=2)
    
    # Save AGOP metrics
    agop_keys = [k for k in history.keys() if k.startswith('agop_') and k != 'agop_topk_subspace_prev']
    if agop_keys:
        with h5py.File(save_dir / 'agop_metrics.h5', 'w') as f:
            for key in agop_keys:
                f.create_dataset(key, data=np.array(history[key]), compression='gzip')
            # Compute AGOP epochs
            agop_epochs = [e for e in history['epoch'] if e % agop_freq == 0]
            f.create_dataset('epoch', data=np.array(agop_epochs[:len(history[agop_keys[0]])]), compression='gzip')
    
    # Save Lazy-Rich metrics
    lr_keys = [k for k in lazy_rich_history.keys() if k != 'epoch']
    if lr_keys:
        with h5py.File(save_dir / 'lazy_rich_metrics.h5', 'w') as f:
            for key in lr_keys:
                data = lazy_rich_history[key]
                f.create_dataset(key, data=np.array(data), compression='gzip')
            f.create_dataset('epoch', data=np.array(lazy_rich_history['epoch']), compression='gzip')
    
    print(f"Results saved to {save_dir}")
    return history, lazy_rich_history


def main():
    parser = argparse.ArgumentParser(description='Train MNIST MLP with AGOP and Lazy-Rich tracking')
    
    # Model arguments
    parser.add_argument('--train_points', type=int, default=1000, help='Number of training points')
    parser.add_argument('--hidden_dim', type=int, default=200, help='Hidden dimension')
    parser.add_argument('--depth', type=int, default=3, help='Network depth')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'tanh', 'gelu'])
    parser.add_argument('--init_scale', type=float, default=8.0, help='Initialization scale')
    
    # Optimizer arguments
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['muon', 'adam', 'adamw', 'sgd'])
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
    
    # Training arguments
    parser.add_argument('--n_epochs', type=int, default=50000, help='Number of epochs')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log_freq', type=int, default=100, help='Logging frequency')
    
    # AGOP arguments
    parser.add_argument('--agop_freq', type=int, default=100, help='AGOP computation frequency')
    parser.add_argument('--agop_subsample', type=int, default=500, help='Subsample size for AGOP')
    parser.add_argument('--agop_top_k', type=int, default=20, help='Top eigenvectors to track')
    
    # Lazy-Rich arguments
    parser.add_argument('--ntk_subsample', type=int, default=200, help='Subsample size for NTK computation')
    parser.add_argument('--compute_ntk', action='store_true', default=True, help='Compute NTK distance')
    parser.add_argument('--no_ntk', action='store_true', help='Disable NTK computation (expensive)')
    parser.add_argument('--compute_feature_kernel', action='store_true', default=True, help='Compute feature kernel')
    parser.add_argument('--no_feature_kernel', action='store_true', help='Disable feature kernel computation')
    
    # Output arguments
    parser.add_argument('--save_dir', type=str, default='./results/agop_experiments/mnist')
    parser.add_argument('--experiment_name', type=str, default=None)
    
    args = parser.parse_args()
    
    # Handle negation flags
    compute_ntk = args.compute_ntk and not args.no_ntk
    compute_feature_kernel = args.compute_feature_kernel and not args.no_feature_kernel
    
    # Set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    if args.experiment_name is None:
        args.experiment_name = f'mnist_{args.optimizer}_wd{args.weight_decay}_seed{args.seed}'
    
    print("="*80)
    print("MNIST Omnigrok with AGOP + Lazy-Rich Tracking")
    print("="*80)
    print(f"Training points: {args.train_points}, Hidden dim: {args.hidden_dim}, Depth: {args.depth}")
    print(f"Optimizer: {args.optimizer}, lr={args.lr}, wd={args.weight_decay}")
    print(f"AGOP: freq={args.agop_freq}, top_k={args.agop_top_k}, subsample={args.agop_subsample}")
    print(f"Lazy-Rich: NTK={compute_ntk} (n={args.ntk_subsample}), FeatureKernel={compute_feature_kernel}")
    print("="*80)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_data, train_labels, test_data, test_labels = create_onehot_mnist_dataset(args.train_points, device)
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    print(f"Input dim: {train_data.shape[1]} (784 pixels, flattened)")
    print(f"Input dtype: {train_data.dtype} (continuous for AGOP)")
    
    # Create model
    model = MNISTModel(input_dim=784, hidden_dim=args.hidden_dim, output_dim=10, 
                      depth=args.depth, activation=args.activation, 
                      initialization_scale=args.init_scale).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create optimizer
    if args.optimizer == 'muon':
        optimizer = Muon(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.95, use_nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    
    # Create AGOP tracker
    agop_tracker = InputGradientAGOPTracker(
        top_k=args.agop_top_k, 
        subsample_size=args.agop_subsample,
        device=str(device), 
        agop_device='cpu', 
        use_mse_loss=False
    )
    
    # Create Lazy-Rich tracker and initialize with K_0
    lazy_rich_tracker = LazyRichTracker(
        n_subsample=args.ntk_subsample,
        device=str(device),
        output_device='cpu',
        feature_layer=None,  # Auto-detect
        use_efficient_ntk=True
    )
    
    # Initialize lazy-rich tracker (stores K_0 before any training)
    print("\nInitializing Lazy-Rich tracker (storing K_0)...")
    lazy_rich_tracker.initialize(
        model, train_data,
        compute_ntk=compute_ntk,
        compute_fk=compute_feature_kernel
    )
    
    # Save directory setup
    save_dir = Path(args.save_dir) / args.experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = vars(args).copy()
    config['compute_ntk'] = compute_ntk
    config['compute_feature_kernel'] = compute_feature_kernel
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Train
    history, lazy_rich_history = train_with_tracking(
        model, train_data, train_labels, test_data, test_labels,
        optimizer, args.n_epochs, device,
        agop_tracker, lazy_rich_tracker,
        args.agop_freq, args.log_freq, save_dir,
        compute_ntk=compute_ntk,
        compute_feature_kernel=compute_feature_kernel
    )
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Final: Train Acc={history['train_acc'][-1]:.4f}, Test Acc={history['test_acc'][-1]:.4f}")
    
    if history['test_acc'][-1] > 0.95:
        for i, acc in enumerate(history['test_acc']):
            if acc > 0.95:
                print(f"GROKKING detected at epoch {history['epoch'][i]}!")
                break
    
    # Print lazy-rich summary
    if 'ntk_distance' in lazy_rich_history and len(lazy_rich_history['ntk_distance']) > 0:
        print(f"Final NTK Distance: {lazy_rich_history['ntk_distance'][-1]:.4e}")
        print(f"Max NTK Distance: {max(lazy_rich_history['ntk_distance']):.4e}")
    
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
