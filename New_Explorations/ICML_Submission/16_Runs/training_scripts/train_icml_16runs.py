"""
Unified Training Script for ICML 16_Runs Experiments

Trains the UnifiedTransformer on modular addition while tracking:
1. Training metrics (loss, accuracy)
2. Input-gradient AGOP metrics (gradient geometry)
3. Lazy-Rich dynamics (NTK evolution, weight norms)

Supports all 16 configuration combinations:
- Modulus: 97, 113
- Attention: softmax, relu
- LayerNorm: on, off
- Optimizer: adam, muon

Reference for Lazy-Rich metrics: Kumar et al. (2024)
"Grokking as the Transition from Lazy to Rich Training Dynamics"
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

# Add paths for imports
SCRIPT_DIR = Path(__file__).parent.absolute()
CORE_DIR = SCRIPT_DIR.parent / 'core'
FRAMEWORK_DIR = Path('/om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/standard_experiments/framework')

sys.path.insert(0, str(CORE_DIR))
sys.path.insert(0, str(FRAMEWORK_DIR))

from muon_official import Muon
from agop_utils import InputGradientAGOPTracker
from lazy_rich_utils import LazyRichTracker, compute_weight_norms
from onehot_datasets import create_onehot_modular_dataset
from unified_transformer import UnifiedTransformer


def train_with_tracking(
    model, train_data, train_labels, test_data, test_labels,
    optimizer, n_epochs, device,
    agop_tracker, lazy_rich_tracker,
    agop_freq, log_freq, save_dir,
    compute_ntk=True, compute_feature_kernel=True
):
    """Training loop with AGOP and Lazy-Rich tracking."""
    criterion = nn.CrossEntropyLoss()
    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    lazy_rich_history = {}
    
    print(f"Starting training for {n_epochs} epochs")
    print(f"AGOP tracking: computing every {agop_freq} epochs")
    print(f"Lazy-Rich tracking: NTK={compute_ntk}, FeatureKernel={compute_feature_kernel}")
    start_time = time.time()
    
    for epoch in tqdm(range(n_epochs), desc="Training"):
        # Training step
        model.train()
        optimizer.zero_grad()
        logits = model(train_data)
        loss = criterion(logits, train_labels)
        loss.backward()
        optimizer.step()
        
        train_loss = loss.item()
        preds = logits.argmax(dim=-1)
        train_acc = (preds == train_labels).float().mean().item()
        
        # Evaluation (every log_freq epochs)
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
            
            # Always track weight norms (cheap)
            weight_norms = compute_weight_norms(model)
            if 'weight_norm_total' not in history:
                history['weight_norm_total'] = []
            history['weight_norm_total'].append(weight_norms['total'])
            
            # Compute AGOP and Lazy-Rich metrics at agop_freq intervals
            if epoch % agop_freq == 0:
                print(f"\n  Epoch {epoch}: Computing metrics...", flush=True)
                
                # Compute AGOP
                print(f"    AGOP...", end=' ', flush=True)
                agop_start = time.time()
                agop = agop_tracker.compute_input_agop(model, train_data, train_labels, criterion)
                
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
    
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds ({elapsed_time/3600:.2f} hours)")
    
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
    parser = argparse.ArgumentParser(description='Train UnifiedTransformer for ICML 16_Runs experiments')
    
    # Dataset arguments
    parser.add_argument('--modulus', '-p', type=int, default=97, choices=[97, 113],
                       help='Modulus for modular addition (97 or 113)')
    parser.add_argument('--train_fraction', type=float, default=0.5, 
                       help='Fraction of data for training')
    
    # Model arguments
    parser.add_argument('--attention_type', type=str, default='softmax', 
                       choices=['softmax', 'relu'],
                       help='Attention type: softmax or relu')
    parser.add_argument('--use_layernorm', action='store_true', default=False,
                       help='Enable LayerNorm in transformer')
    parser.add_argument('--no_layernorm', action='store_true',
                       help='Disable LayerNorm (for explicit control)')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of transformer layers')
    parser.add_argument('--d_mlp', type=int, default=512, help='MLP hidden dimension')
    
    # Optimizer arguments
    parser.add_argument('--optimizer', type=str, default='adam', 
                       choices=['adam', 'adamw', 'muon', 'sgd'],
                       help='Optimizer to use')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    
    # Training arguments
    parser.add_argument('--n_epochs', type=int, default=50000, help='Number of epochs')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log_freq', type=int, default=100, help='Logging frequency')
    
    # AGOP arguments
    parser.add_argument('--agop_freq', type=int, default=100, help='AGOP computation frequency')
    parser.add_argument('--agop_subsample', type=int, default=None, help='Subsample size for AGOP')
    parser.add_argument('--agop_top_k', type=int, default=20, help='Top eigenvectors to track')
    
    # Lazy-Rich arguments
    parser.add_argument('--ntk_subsample', type=int, default=200, 
                       help='Subsample size for NTK computation')
    parser.add_argument('--compute_ntk', action='store_true', default=True, 
                       help='Compute NTK distance')
    parser.add_argument('--no_ntk', action='store_true', help='Disable NTK computation')
    parser.add_argument('--compute_feature_kernel', action='store_true', default=True, 
                       help='Compute feature kernel')
    parser.add_argument('--no_feature_kernel', action='store_true', 
                       help='Disable feature kernel computation')
    
    # Output arguments
    parser.add_argument('--save_dir', type=str, default='./results',
                       help='Base directory for saving results')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Override auto-generated experiment name')
    
    args = parser.parse_args()
    
    # Handle negation flags
    use_layernorm = args.use_layernorm and not args.no_layernorm
    compute_ntk = args.compute_ntk and not args.no_ntk
    compute_feature_kernel = args.compute_feature_kernel and not args.no_feature_kernel
    
    # Set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    # Generate experiment name
    ln_str = 'ln' if use_layernorm else 'noln'
    if args.experiment_name is None:
        base_config = f'p{args.modulus}_{args.attention_type}_{ln_str}_{args.optimizer}'
        args.experiment_name = f'{base_config}/wd{args.weight_decay}_seed{args.seed}'
    
    print("=" * 80)
    print("ICML 16_Runs: UnifiedTransformer Training")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Modulus: p={args.modulus}")
    print(f"  Attention: {args.attention_type}")
    print(f"  LayerNorm: {use_layernorm}")
    print(f"  Optimizer: {args.optimizer}, lr={args.lr}, wd={args.weight_decay}")
    print(f"  Epochs: {args.n_epochs}")
    print(f"Tracking:")
    print(f"  AGOP: freq={args.agop_freq}, top_k={args.agop_top_k}")
    print(f"  Lazy-Rich: NTK={compute_ntk} (n={args.ntk_subsample}), FeatureKernel={compute_feature_kernel}")
    print(f"Output: {args.save_dir}/{args.experiment_name}")
    print("=" * 80)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create one-hot encoded dataset
    train_data, train_labels, test_data, test_labels = create_onehot_modular_dataset(
        p=args.modulus, operation='add', train_fraction=args.train_fraction, device=device
    )
    print(f"Dataset: modular addition mod {args.modulus}")
    print(f"  Train size: {len(train_data)}, Test size: {len(test_data)}")
    print(f"  Input dim: {train_data.shape[1]} (2*p={2*args.modulus}) - One-hot encoded")
    
    # Create model
    model = UnifiedTransformer(
        p=args.modulus,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_mlp=args.d_mlp,
        attention_type=args.attention_type,
        use_layernorm=use_layernorm,
        dropout=0.0
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: UnifiedTransformer with {n_params:,} parameters")
    
    # Create optimizer
    if args.optimizer == 'muon':
        optimizer = Muon(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, 
                        momentum=0.95, use_nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, 
                             momentum=0.9)
    
    # Create AGOP tracker
    agop_tracker = InputGradientAGOPTracker(
        top_k=args.agop_top_k, 
        subsample_size=args.agop_subsample, 
        device=str(device), 
        agop_device='cpu', 
        use_mse_loss=False
    )
    
    # Create Lazy-Rich tracker
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
    config['use_layernorm'] = use_layernorm
    config['compute_ntk'] = compute_ntk
    config['compute_feature_kernel'] = compute_feature_kernel
    config['n_params'] = n_params
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
    print("Training Summary")
    print(f"{'='*80}")
    print(f"Final: Train Acc={history['train_acc'][-1]:.4f}, Test Acc={history['test_acc'][-1]:.4f}")
    
    # Check for grokking
    grokked = False
    grok_epoch = None
    if history['test_acc'][-1] > 0.95:
        for i, acc in enumerate(history['test_acc']):
            if acc > 0.95:
                grokked = True
                grok_epoch = history['epoch'][i]
                print(f"GROKKING detected at epoch {grok_epoch}!")
                break
    
    if not grokked:
        print("No grokking detected (test acc < 95%)")
    
    # Print lazy-rich summary
    if 'ntk_distance' in lazy_rich_history and len(lazy_rich_history['ntk_distance']) > 0:
        print(f"NTK Distance: final={lazy_rich_history['ntk_distance'][-1]:.4e}, "
              f"max={max(lazy_rich_history['ntk_distance']):.4e}")
    
    print(f"Results saved to: {save_dir}")
    print(f"{'='*80}")
    
    # Return summary for potential programmatic use
    return {
        'grokked': grokked,
        'grok_epoch': grok_epoch,
        'final_train_acc': history['train_acc'][-1],
        'final_test_acc': history['test_acc'][-1],
        'save_dir': str(save_dir)
    }


if __name__ == "__main__":
    main()

