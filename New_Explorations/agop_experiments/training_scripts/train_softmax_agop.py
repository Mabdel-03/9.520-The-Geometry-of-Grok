"""
Training script for Softmax Transformer modular addition with Input-Gradient AGOP tracking

Trains a standard softmax transformer on modular addition while tracking
input-gradient AGOP metrics. This allows comparison with Nanda's ReLU transformer.

Key differences from Nanda:
- Standard softmax attention (vs ReLU attention)
- LayerNorm (vs no normalization)
- Residual connections (vs no residuals)
"""

import sys
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
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
from onehot_datasets import create_onehot_modular_dataset
from onehot_models import ModularArithmeticMLP, OneHotStandardTransformer


def train_with_agop_tracking(
    model, train_data, train_labels, test_data, test_labels,
    optimizer, n_epochs, device, agop_tracker, agop_freq, log_freq, save_dir,
):
    """Training loop with input-gradient AGOP tracking."""
    criterion = nn.CrossEntropyLoss()
    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    
    print(f"Starting training for {n_epochs} epochs")
    print(f"AGOP tracking enabled: computing every {agop_freq} epochs")
    start_time = time.time()
    
    for epoch in tqdm(range(n_epochs), desc="Training"):
        model.train()
        optimizer.zero_grad()
        logits = model(train_data)
        loss = criterion(logits, train_labels)
        loss.backward()
        optimizer.step()
        
        train_loss = loss.item()
        preds = logits.argmax(dim=-1)
        train_acc = (preds == train_labels).float().mean().item()
        
        if epoch % log_freq == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                test_logits = model(test_data)
                test_loss = criterion(test_logits, test_labels).item()
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
                agop = agop_tracker.compute_input_agop(model, train_data, train_labels, criterion)
                
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
    parser = argparse.ArgumentParser(description='Train Softmax Transformer with Input-Gradient AGOP tracking')
    parser.add_argument('--p', type=int, default=97, help='Modulus for modular addition')
    parser.add_argument('--train_fraction', type=float, default=0.5, help='Fraction of data for training')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--d_ff', type=int, default=512, help='FFN dimension')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--architecture', type=str, default='mlp',
                       choices=['mlp', 'transformer'],
                       help='Model architecture (both support tractable AGOP)')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['muon', 'adam', 'adamw', 'sgd'])
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--n_epochs', type=int, default=50000, help='Number of epochs')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--agop_freq', type=int, default=100, help='AGOP computation frequency')
    parser.add_argument('--agop_subsample', type=int, default=None, help='Subsample size for AGOP')
    parser.add_argument('--agop_top_k', type=int, default=20, help='Top eigenvectors to track')
    parser.add_argument('--log_freq', type=int, default=100, help='Logging frequency')
    parser.add_argument('--save_dir', type=str, default='./results/agop_experiments/softmax')
    parser.add_argument('--experiment_name', type=str, default=None)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    if args.experiment_name is None:
        args.experiment_name = f'softmax_{args.architecture}_{args.optimizer}_p{args.p}_lr{args.lr}_wd{args.weight_decay}_seed{args.seed}'
    
    print("="*80)
    print("Softmax Modular Addition with Tractable Input-Gradient AGOP (One-Hot)")
    print("="*80)
    print(f"Modulus p={args.p}, Train fraction={args.train_fraction}")
    print(f"Architecture: {args.architecture.upper()}")
    print(f"Optimizer: {args.optimizer}, lr={args.lr}, wd={args.weight_decay}")
    print(f"AGOP: freq={args.agop_freq}, top_k={args.agop_top_k}")
    print("="*80)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create one-hot encoded dataset
    train_data, train_labels, test_data, test_labels = create_onehot_modular_dataset(
        p=args.p, operation='add', train_fraction=args.train_fraction, device=device
    )
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    print(f"Input dim: {train_data.shape[1]} (2*p={2*args.p}) - One-hot encoded")
    print(f"Input dtype: {train_data.dtype} (continuous for AGOP)")
    
    # Create model based on architecture choice
    if args.architecture == 'mlp':
        model = ModularArithmeticMLP(p=args.p, hidden_dim=args.d_model).to(device)
        print(f"Model: MLP with {sum(p.numel() for p in model.parameters()):,} parameters")
    elif args.architecture == 'transformer':
        model = OneHotStandardTransformer(p=args.p, d_model=args.d_model, 
                                         n_heads=args.n_heads, n_layers=args.n_layers, 
                                         d_ff=args.d_ff).to(device)
        print(f"Model: Standard Transformer with {sum(p.numel() for p in model.parameters()):,} parameters")
    else:
        raise ValueError(f"Unknown architecture: {args.architecture}")
    
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

