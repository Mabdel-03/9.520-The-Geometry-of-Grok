"""
Training Script for Power AGOP Grokking Experiments

Trains PowerTransformer or GrokkingMLP on modular addition while tracking:
1. Training metrics (loss, accuracy)
2. Input-gradient AGOP metrics (gradient geometry, VCR)

Based on Power et al. (2022) experimental setup with AGOP analysis.

Supports:
- Architectures: transformer, mlp
- Optimizers: adamw, muon
- Input types: discrete (token embeddings), onehot (continuous)
- Weight decay sweep: 0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0
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
from power_transformer import PowerTransformer
from grokking_mlp import GrokkingMLP
from datasets import ModularArithmeticDataset, create_transformer_tokens


def train_with_agop_tracking(
    model: nn.Module,
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    test_data: torch.Tensor,
    test_labels: torch.Tensor,
    train_onehot: torch.Tensor,
    optimizer: optim.Optimizer,
    n_epochs: int,
    device: torch.device,
    agop_tracker: InputGradientAGOPTracker,
    agop_model: nn.Module,
    agop_freq: int,
    log_freq: int,
    save_dir: Path,
    model_type: str,
    input_type: str = 'discrete',
):
    """
    Training loop with AGOP tracking.
    
    Args:
        model: Model to train
        train_data: Training inputs (discrete tokens or one-hot)
        train_labels: Training labels
        test_data: Test inputs
        test_labels: Test labels
        train_onehot: One-hot version of training data (for AGOP)
        agop_model: Wrapped model that uses forward_onehot for AGOP computation
        optimizer: Optimizer instance
        n_epochs: Number of training epochs
        device: Torch device
        agop_tracker: AGOP tracking instance
        agop_freq: Frequency of AGOP computation
        log_freq: Frequency of logging
        save_dir: Directory to save results
        model_type: 'transformer' or 'mlp'
        input_type: 'discrete' or 'onehot' - determines which forward method to call
    """
    criterion = nn.CrossEntropyLoss()
    
    # Define forward function based on input type
    if input_type == 'onehot':
        def forward_fn(m, x):
            return m.forward_onehot(x)
    else:
        def forward_fn(m, x):
            return m(x)
    history = {
        'epoch': [], 
        'train_loss': [], 
        'train_acc': [], 
        'test_loss': [], 
        'test_acc': [],
        'weight_norm_total': []
    }
    
    print(f"Starting training for {n_epochs} epochs")
    print(f"AGOP tracking: computing every {agop_freq} epochs")
    start_time = time.time()
    
    for epoch in tqdm(range(n_epochs), desc="Training"):
        # Training step
        model.train()
        optimizer.zero_grad()
        logits = forward_fn(model, train_data)
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
                test_logits = forward_fn(model, test_data)
                test_loss = criterion(test_logits, test_labels).item()
                test_preds = test_logits.argmax(dim=-1)
                test_acc = (test_preds == test_labels).float().mean().item()
            
            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            
            # Track weight norms (cheap)
            weight_norm = sum(p.data.norm(2).item() ** 2 for p in model.parameters()) ** 0.5
            history['weight_norm_total'].append(weight_norm)
            
            # Compute AGOP metrics at agop_freq intervals
            if epoch % agop_freq == 0:
                print(f"\n  Epoch {epoch}: Computing AGOP...", end=' ', flush=True)
                agop_start = time.time()
                
                # Use forward_onehot for AGOP computation via agop_model wrapper
                agop = agop_tracker.compute_input_agop(
                    agop_model, train_onehot, train_labels, criterion
                )
                
                if agop is not None:
                    metrics = agop_tracker.compute_agop_metrics(history, agop)
                    agop_time = time.time() - agop_start
                    print(f"Done ({agop_time:.1f}s)")
                    print(f"      Trace: {metrics.get('agop_trace', 0):.4e}, "
                          f"Eigengap: {metrics.get('agop_eigengap', 0):.4e}, "
                          f"VCR: {metrics.get('agop_variation_collapse_ratio', 0):.4f}")
                else:
                    print("Failed")
            
            if epoch % (log_freq * 10) == 0:
                print(f"\nEpoch {epoch}/{n_epochs}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
    
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds ({elapsed_time/3600:.2f} hours)")
    
    # Save results
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training history (basic metrics)
    history_json = {k: v for k, v in history.items() 
                   if k != 'agop_topk_subspace_prev' and not k.startswith('agop_')}
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history_json, f, indent=2)
    
    # Save AGOP metrics to HDF5
    agop_keys = [k for k in history.keys() if k.startswith('agop_') and k != 'agop_topk_subspace_prev']
    if agop_keys:
        with h5py.File(save_dir / 'agop_metrics.h5', 'w') as f:
            for key in agop_keys:
                data = history[key]
                # Eigenvectors are stored as lists of lists, need special handling
                if 'eigenvector' in key:
                    # Convert list of lists to 2D numpy array (n_epochs, d_input)
                    f.create_dataset(key, data=np.array(data), compression='gzip')
                else:
                    f.create_dataset(key, data=np.array(data), compression='gzip')
            # Store epochs at which AGOP was computed
            agop_epochs = [e for e in history['epoch'] if e % agop_freq == 0]
            # Find a non-eigenvector key to get the count
            scalar_keys = [k for k in agop_keys if 'eigenvector' not in k]
            if scalar_keys:
                f.create_dataset('epoch', data=np.array(agop_epochs[:len(history[scalar_keys[0]])]), compression='gzip')
    
    print(f"Results saved to {save_dir}")
    return history


class OneHotForwardWrapper(nn.Module):
    """
    Wrapper to make model use forward_onehot for AGOP computation.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        return self.model.forward_onehot(x)
    
    def parameters(self):
        return self.model.parameters()
    
    def named_parameters(self):
        return self.model.named_parameters()
    
    def eval(self):
        return self.model.eval()
    
    def train(self, mode=True):
        return self.model.train(mode)


def main():
    parser = argparse.ArgumentParser(description='Train for Power AGOP experiments')
    
    # Dataset arguments
    parser.add_argument('--modulus', '-p', type=int, default=97,
                       help='Modulus for modular arithmetic')
    parser.add_argument('--operation', type=str, default='add',
                       choices=['add', 'sub', 'mul', 'div', 'cubic', 'quadratic', 'symmetric_cubic', 'mixed_poly', 'pure_cubic', 'pure_mul'],
                       help='Modular operation: add (x+y), sub (x-y), mul (x*y), div (x/y), cubic (x^3+xy), quadratic (a^2+b), symmetric_cubic (a^3+b^3), mixed_poly (a^2+ab+b^2), pure_cubic (x^3), pure_mul (x*y)')
    parser.add_argument('--train_fraction', type=float, default=0.5, 
                       help='Fraction of data for training (0.5 means 50 percent)')
    
    # Architecture arguments
    parser.add_argument('--architecture', type=str, default='transformer',
                       choices=['transformer', 'mlp'],
                       help='Model architecture')
    parser.add_argument('--input_type', type=str, default='discrete',
                       choices=['discrete', 'onehot'],
                       help='Input representation type')
    parser.add_argument('--d_model', type=int, default=128, 
                       help='Model/embedding dimension')
    parser.add_argument('--n_heads', type=int, default=4, 
                       help='Number of attention heads (transformer only)')
    parser.add_argument('--n_layers', type=int, default=2, 
                       help='Number of transformer layers')
    parser.add_argument('--d_mlp', type=int, default=512, 
                       help='MLP hidden dimension')
    
    # Optimizer arguments
    parser.add_argument('--optimizer', type=str, default='adamw', 
                       choices=['adamw', 'muon'],
                       help='Optimizer to use')
    parser.add_argument('--lr', type=float, default=0.001, 
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, 
                       help='Weight decay')
    
    # Training arguments
    parser.add_argument('--n_epochs', type=int, default=50000, 
                       help='Number of epochs')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed')
    parser.add_argument('--log_freq', type=int, default=100, 
                       help='Logging frequency')
    
    # AGOP arguments
    parser.add_argument('--agop_freq', type=int, default=100, 
                       help='AGOP computation frequency')
    parser.add_argument('--agop_subsample', type=int, default=None, 
                       help='Subsample size for AGOP (None = use all)')
    parser.add_argument('--agop_top_k', type=int, default=20, 
                       help='Top eigenvectors to track')
    
    # Output arguments
    parser.add_argument('--save_dir', type=str, default='./results',
                       help='Base directory for saving results')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Override auto-generated experiment name')
    
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    # Generate experiment name
    if args.experiment_name is None:
        args.experiment_name = (
            f'{args.architecture}_{args.input_type}_{args.optimizer}/'
            f'wd{args.weight_decay}_seed{args.seed}'
        )
    
    # Get operation description for logging
    op_descriptions = {
        'add': 'x + y',
        'sub': 'x - y', 
        'mul': 'x * y',
        'div': 'x / y',
        'cubic': 'x³ + xy',
        'quadratic': 'a² + b',
        'symmetric_cubic': 'a³ + b³',
        'mixed_poly': 'a² + ab + b²',
        'pure_cubic': 'x³',
        'pure_mul': 'x * y',
    }
    op_desc = op_descriptions.get(args.operation, args.operation)
    
    print("=" * 80)
    print("Power AGOP Study: Grokking Experiments")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Task: ({op_desc}) mod {args.modulus}")
    print(f"  Operation: {args.operation}")
    print(f"  Modulus: p={args.modulus}")
    print(f"  Architecture: {args.architecture}")
    print(f"  Input type: {args.input_type}")
    print(f"  Optimizer: {args.optimizer}, lr={args.lr}, wd={args.weight_decay}")
    print(f"  Epochs: {args.n_epochs}")
    print(f"Tracking:")
    print(f"  AGOP: freq={args.agop_freq}, top_k={args.agop_top_k}")
    print(f"Output: {args.save_dir}/{args.experiment_name}")
    print("=" * 80)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = ModularArithmeticDataset(
        p=args.modulus,
        operation=args.operation,
        train_fraction=args.train_fraction,
        seed=args.seed,
        device=device
    )
    print(f"Dataset: {dataset}")
    
    # Get data in appropriate format
    if args.architecture == 'transformer':
        if args.input_type == 'discrete':
            train_data, train_labels = dataset.get_train_data('transformer')
            test_data, test_labels = dataset.get_test_data('transformer')
        else:  # onehot
            train_data, train_labels = dataset.get_train_data('onehot')
            test_data, test_labels = dataset.get_test_data('onehot')
    else:  # mlp
        if args.input_type == 'discrete':
            train_data, train_labels = dataset.get_train_data('discrete')
            test_data, test_labels = dataset.get_test_data('discrete')
        else:  # onehot
            train_data, train_labels = dataset.get_train_data('onehot')
            test_data, test_labels = dataset.get_test_data('onehot')
    
    # Get one-hot data for AGOP computation
    train_onehot, _ = dataset.get_train_data('onehot')
    
    print(f"  Train size: {len(train_data)}, Test size: {len(test_data)}")
    print(f"  Train data shape: {train_data.shape}")
    
    # Create model
    if args.architecture == 'transformer':
        model = PowerTransformer(
            p=args.modulus,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_mlp=args.d_mlp,
            dropout=0.0
        ).to(device)
    else:  # mlp
        model = GrokkingMLP(
            p=args.modulus,
            d_embed=args.d_model,
            d_hidden=args.d_mlp,
            use_layernorm=False,
            dropout=0.0,
            input_type=args.input_type
        ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.architecture} with {n_params:,} parameters")
    
    # Create optimizer
    if args.optimizer == 'muon':
        optimizer = Muon(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay,
            momentum=0.95, 
            use_nesterov=True
        )
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
    
    # Create AGOP tracker
    # Wrap model for one-hot forward if using discrete input
    agop_model = OneHotForwardWrapper(model)
    
    agop_tracker = InputGradientAGOPTracker(
        top_k=args.agop_top_k,
        subsample_size=args.agop_subsample,
        device=str(device),
        agop_device='cpu',
        use_mse_loss=False
    )
    
    # Save directory setup
    save_dir = Path(args.save_dir) / args.experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = vars(args).copy()
    config['n_params'] = n_params
    config['model_config'] = model.get_config()
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Train with AGOP tracking
    history = train_with_agop_tracking(
        model=model,
        train_data=train_data,
        train_labels=train_labels,
        test_data=test_data,
        test_labels=test_labels,
        train_onehot=train_onehot,
        optimizer=optimizer,
        n_epochs=args.n_epochs,
        device=device,
        agop_tracker=agop_tracker,
        agop_model=agop_model,
        agop_freq=args.agop_freq,
        log_freq=args.log_freq,
        save_dir=save_dir,
        model_type=args.architecture,
        input_type=args.input_type,
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
    
    # Print VCR summary if available
    if 'agop_variation_collapse_ratio' in history:
        vcr = history['agop_variation_collapse_ratio']
        print(f"VCR: final={vcr[-1]:.4f}, max={max(vcr):.4f}")
    
    print(f"Results saved to: {save_dir}")
    print(f"{'='*80}")
    
    return {
        'grokked': grokked,
        'grok_epoch': grok_epoch,
        'final_train_acc': history['train_acc'][-1],
        'final_test_acc': history['test_acc'][-1],
        'save_dir': str(save_dir)
    }


if __name__ == "__main__":
    main()

