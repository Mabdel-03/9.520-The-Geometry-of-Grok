#!/usr/bin/env python
"""
Proper replication of Paper 02 using train_add (as used in the paper's figures)
Liu et al. (2022) - Effective Theory
"""

import sys
from pathlib import Path
import json

# Add toy directory to path
sys.path.insert(0, str(Path(__file__).parent / 'toy'))

from train_add import train_add

def extract_and_save_results(result_dict, output_file):
    """Extract metrics from train_add result and save to training_history.json"""
    
    # Extract the full training curves (logged at every step!)
    train_accs = result_dict['acc_train']
    test_accs = result_dict['acc_test']
    train_losses = result_dict['loss_train']
    test_losses = result_dict['loss_test']
    rqis = result_dict['rqi']
    
    # Convert numpy arrays to lists
    if hasattr(train_accs, 'tolist'):
        train_accs = train_accs.tolist()
    if hasattr(test_accs, 'tolist'):
        test_accs = test_accs.tolist()
    if hasattr(train_losses, 'tolist'):
        train_losses = train_losses.tolist()
    if hasattr(test_losses, 'tolist'):
        test_losses = test_losses.tolist()
    if hasattr(rqis, 'tolist'):
        rqis = rqis.tolist()
    
    history = {
        'epoch': list(range(len(train_accs))),
        'train_loss': train_losses,
        'train_acc': train_accs,
        'test_loss': test_losses,
        'test_acc': test_accs,
        'rqi': rqis,
        'grokking_summary': {
            'train_acc_step': int(result_dict['iter_train']),
            'test_acc_step': int(result_dict['iter_test']),
            'rqi_step': int(result_dict['iter_rqi']),
            'grokking_delay': int(result_dict['iter_test']) - int(result_dict['iter_train']),
            'grokking_detected': int(result_dict['iter_test']) > int(result_dict['iter_train'])
        }
    }
    
    # Save to JSON
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    return history


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=int, default=10, help='Number of symbols')
    parser.add_argument('--reprs_dim', type=int, default=1, help='Representation dimension')
    parser.add_argument('--train_num', type=int, default=45, help='Number of training samples')
    parser.add_argument('--steps', type=int, default=5000, help='Training steps')
    parser.add_argument('--eta_reprs', type=float, default=1e-3, help='LR for representations')
    parser.add_argument('--eta_dec', type=float, default=1e-4, help='LR for decoder')
    parser.add_argument('--weight_decay_reprs', type=float, default=0.0, help='WD for representations')
    parser.add_argument('--weight_decay_dec', type=float, default=0.0, help='WD for decoder')
    parser.add_argument('--seed', type=int, default=58, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Paper 02: Liu et al. (2022) - Effective Theory")
    print("Using train_add (as in the paper)")
    print("="*80)
    print(f"Parameters:")
    print(f"  p={args.p}, train_num={args.train_num}/{args.p*(args.p+1)//2}")
    print(f"  steps={args.steps}, reprs_dim={args.reprs_dim}")
    print(f"  eta_reprs={args.eta_reprs}, eta_dec={args.eta_dec}")
    print(f"  weight_decay: reprs={args.weight_decay_reprs}, dec={args.weight_decay_dec}")
    print("="*80)
    
    # Run training
    result = train_add(
        p=args.p,
        reprs_dim=args.reprs_dim,
        train_num=args.train_num,
        steps=args.steps,
        eta_reprs=args.eta_reprs,
        eta_dec=args.eta_dec,
        weight_decay_reprs=args.weight_decay_reprs,
        weight_decay_dec=args.weight_decay_dec,
        seed=args.seed,
        device=args.device,
        loss_type="MSE"
    )
    
    print("="*80)
    print("✅ Training complete!")
    
    # Extract the integer values directly
    iter_train = int(result['iter_train'])
    iter_test = int(result['iter_test'])
    iter_rqi = int(result['iter_rqi'])
    
    print()
    print("🎯 Grokking Analysis:")
    print(f"  Train acc threshold (0.9) reached: step {iter_train}")
    print(f"  Test acc threshold (0.9) reached:  step {iter_test}")
    print(f"  RQI threshold (0.95) reached:      step {iter_rqi}")
    print()
    
    if iter_test > iter_train:
        delay = iter_test - iter_train
        print(f"  ⭐⭐⭐ GROKKING DETECTED! ⭐⭐⭐")
        print(f"  Test accuracy lagged train by {delay} steps")
        print(f"  This is DELAYED GENERALIZATION - the hallmark of grokking!")
    else:
        print(f"  ⚠️ No grokking (test reached threshold before/with train)")
    
    print("="*80)
    
    # Extract and save metrics
    output_file = Path(__file__).parent / 'logs' / 'training_history.json'
    history = extract_and_save_results(result, output_file)
    
    print(f"✓ Results saved to {output_file}")
    print("="*80)

