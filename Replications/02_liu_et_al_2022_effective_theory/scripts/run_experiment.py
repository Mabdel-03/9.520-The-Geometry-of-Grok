#!/usr/bin/env python
"""
Unified experiment runner for Liu et al. (2022) - Effective Theory
Runs experiments and saves results in organized structure.
"""

import sys
import json
import argparse
from pathlib import Path

# Add toy directory to path for train_add
sys.path.insert(0, str(Path(__file__).parent / 'toy'))

from train_add import train_add


def save_toy_model_results(result_dict, output_dir):
    """Save toy model experiment results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract metrics
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
    
    # Save full training history
    history = {
        'step': list(range(len(train_accs))),
        'train_loss': train_losses,
        'train_acc': train_accs,
        'test_loss': test_losses,
        'test_acc': test_accs,
        'rqi': rqis
    }
    
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save summary metrics
    iter_train = int(result_dict['iter_train'])
    iter_test = int(result_dict['iter_test'])
    iter_rqi = int(result_dict['iter_rqi'])
    grokking_delay = iter_test - iter_train
    
    metrics = {
        'train_acc_90_step': iter_train,
        'test_acc_90_step': iter_test,
        'rqi_95_step': iter_rqi,
        'grokking_delay': grokking_delay,
        'grokking_detected': iter_test > iter_train,
        'final_train_acc': float(train_accs[-1]),
        'final_test_acc': float(test_accs[-1]),
        'final_rqi': float(rqis[-1])
    }
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save human-readable summary
    with open(output_dir / 'metrics.txt', 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Experiment 1: Toy Model - Grokking Results\n")
        f.write("=" * 60 + "\n\n")
        f.write("Grokking Milestones:\n")
        f.write(f"  Train accuracy (90%):  Step {iter_train}\n")
        f.write(f"  Test accuracy (90%):   Step {iter_test}\n")
        f.write(f"  RQI threshold (95%):   Step {iter_rqi}\n")
        f.write(f"  Grokking delay:        {grokking_delay} steps\n\n")
        
        f.write("Final Performance:\n")
        f.write(f"  Train accuracy:  {train_accs[-1]:.4f}\n")
        f.write(f"  Test accuracy:   {test_accs[-1]:.4f}\n")
        f.write(f"  RQI:             {rqis[-1]:.4f}\n\n")
        
        if iter_test > iter_train:
            f.write("✅ GROKKING CONFIRMED!\n")
            f.write(f"   Test accuracy lagged train by {grokking_delay} steps.\n")
        else:
            f.write("⚠️  No grokking detected.\n")
        
        f.write("=" * 60 + "\n")
    
    print(f"\n✅ Results saved to {output_dir}")
    print(f"   - training_history.json (full training curves)")
    print(f"   - metrics.json (summary statistics)")
    print(f"   - metrics.txt (human-readable summary)")
    
    return metrics


def run_toy_model(args):
    """Run the main toy model experiment."""
    print("=" * 80)
    print("Experiment 1: Toy Model (Modular Addition)")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Modulus (p):          {args.p}")
    print(f"  Training samples:     {args.train_num}")
    print(f"  Representation dim:   {args.reprs_dim}")
    print(f"  Training steps:       {args.steps}")
    print(f"  LR (representations): {args.eta_reprs}")
    print(f"  LR (decoder):         {args.eta_dec}")
    print(f"  Weight decay:         {args.weight_decay_reprs}, {args.weight_decay_dec}")
    print(f"  Seed:                 {args.seed}")
    print("=" * 80)
    
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
        loss_type=args.loss_type
    )
    
    # Save results
    output_dir = Path(args.output_dir) / 'experiment_1_toy_model'
    metrics = save_toy_model_results(result, output_dir)
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Train accuracy (90%): Step {metrics['train_acc_90_step']}")
    print(f"Test accuracy (90%):  Step {metrics['test_acc_90_step']}")
    print(f"RQI threshold (95%):  Step {metrics['rqi_95_step']}")
    print(f"Grokking delay:       {metrics['grokking_delay']} steps")
    print(f"Final performance:    {metrics['final_train_acc']:.1%} train, {metrics['final_test_acc']:.1%} test")
    
    if metrics['grokking_detected']:
        print("\n✅ GROKKING CONFIRMED!")
    else:
        print("\n⚠️  No grokking detected")
    
    print("=" * 80)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Run Liu et al. (2022) experiments'
    )
    
    # Experiment selection
    parser.add_argument('--experiment', type=str, default='toy_model',
                        choices=['toy_model', 'phase_diagram'],
                        help='Which experiment to run')
    
    # Toy model parameters
    parser.add_argument('--p', type=int, default=10,
                        help='Modulus for modular addition')
    parser.add_argument('--reprs_dim', type=int, default=1,
                        help='Dimension of internal representation')
    parser.add_argument('--train_num', type=int, default=45,
                        help='Number of training samples')
    parser.add_argument('--steps', type=int, default=5000,
                        help='Number of training steps')
    parser.add_argument('--eta_reprs', type=float, default=1e-3,
                        help='Learning rate for representations')
    parser.add_argument('--eta_dec', type=float, default=1e-4,
                        help='Learning rate for decoder')
    parser.add_argument('--weight_decay_reprs', type=float, default=0.0,
                        help='Weight decay for representations')
    parser.add_argument('--weight_decay_dec', type=float, default=0.0,
                        help='Weight decay for decoder')
    parser.add_argument('--seed', type=int, default=58,
                        help='Random seed')
    parser.add_argument('--loss_type', type=str, default='MSE',
                        choices=['MSE', 'CE'],
                        help='Loss function type')
    
    # General parameters
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    if args.experiment == 'toy_model':
        run_toy_model(args)
    elif args.experiment == 'phase_diagram':
        print("Phase diagram experiment not yet implemented.")
        print("Use the original notebooks in phase_diagram_plot/ for phase diagram generation.")
        sys.exit(1)
    
    print("\n✅ Experiment complete!")


if __name__ == '__main__':
    main()

