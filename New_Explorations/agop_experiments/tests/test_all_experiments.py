"""
Comprehensive Test Script for AGOP Experiments

This script runs mini versions of all experiments to verify they work correctly
before running full-scale experiments.

Usage:
    python test_all_experiments.py [--optimizer OPTIMIZER] [--dataset DATASET]
    python test_all_experiments.py --all  # Run all tests
"""

import subprocess
import sys
from pathlib import Path
import json
import argparse
import time
from datetime import datetime

# Test configurations
TEST_CONFIGS = {
    'nanda': {
        'script': 'train_nanda_agop.py',
        'args': {
            'p': 97,  # Smaller modulus for faster testing
            'train_fraction': 0.3,
            'n_epochs': 200,
            'agop_freq': 50,
            'log_freq': 10,
            'd_model': 128,
            'n_heads': 4,
            'd_mlp': 512,
        },
    },
    'softmax': {
        'script': 'train_softmax_agop.py',
        'args': {
            'p': 97,
            'train_fraction': 0.5,
            'n_epochs': 200,
            'agop_freq': 50,
            'log_freq': 10,
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 2,
            'd_ff': 512,
        },
    },
    'mnist': {
        'script': 'train_mnist_agop.py',
        'args': {
            'train_points': 500,  # Reduced from 1000
            'hidden_dim': 200,
            'depth': 3,
            'n_epochs': 200,
            'agop_freq': 50,
            'agop_subsample': 250,  # Subsample for speed
            'log_freq': 10,
        },
    },
    'composition': {
        'script': 'train_composition_agop.py',
        'args': {
            'n_entities': 50,
            'n_facts': 200,  # Reduced from 500
            'train_fraction': 0.3,
            'n_epochs': 200,
            'agop_freq': 50,
            'log_freq': 10,
        },
    },
}

OPTIMIZERS = ['adamw', 'muon', 'sgd']

OPTIMIZER_SETTINGS = {
    'adamw': {'lr': 0.001, 'weight_decay': 1.0},
    'muon': {'lr': 0.001, 'weight_decay': 1.0},
    'sgd': {'lr': 0.01, 'weight_decay': 1.0},
}


def run_test(dataset: str, optimizer: str, device: str = 'cuda', seed: int = 42) -> dict:
    """
    Run a single test experiment.
    
    Returns:
        dict with 'success', 'time', 'output_dir', 'error' keys
    """
    if dataset not in TEST_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    config = TEST_CONFIGS[dataset]
    script = config['script']
    args = config['args'].copy()
    
    # Add optimizer settings
    opt_settings = OPTIMIZER_SETTINGS.get(optimizer, {'lr': 0.001, 'weight_decay': 1.0})
    args.update(opt_settings)
    
    # Add common args
    args['optimizer'] = optimizer
    args['device'] = 'cpu'  # Use CPU for testing to avoid CUDA compatibility issues
    args['seed'] = seed
    args['save_dir'] = f'./test_results/test_{dataset}'
    args['experiment_name'] = f'test_{dataset}_{optimizer}_seed{seed}'
    
    # Build command
    cmd = [sys.executable, script]
    for key, value in args.items():
        cmd.extend([f'--{key}', str(value)])
    
    print(f"\n{'='*80}")
    print(f"Testing: {dataset.upper()} with {optimizer.upper()}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd[:10])}...")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✓ SUCCESS in {elapsed:.1f}s")
            return {
                'success': True,
                'time': elapsed,
                'output_dir': Path(args['save_dir']) / args['experiment_name'],
                'error': None,
            }
        else:
            print(f"✗ FAILED in {elapsed:.1f}s")
            print(f"Error:\n{result.stderr[-500:]}")  # Last 500 chars
            return {
                'success': False,
                'time': elapsed,
                'output_dir': None,
                'error': result.stderr,
            }
    
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"✗ TIMEOUT after {elapsed:.1f}s")
        return {
            'success': False,
            'time': elapsed,
            'output_dir': None,
            'error': 'Timeout after 10 minutes',
        }
    
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ EXCEPTION: {e}")
        return {
            'success': False,
            'time': elapsed,
            'output_dir': None,
            'error': str(e),
        }


def verify_outputs(output_dir: Path) -> dict:
    """
    Verify that expected output files exist.
    
    Returns:
        dict with verification results
    """
    if not output_dir.exists():
        return {'success': False, 'message': 'Output directory does not exist'}
    
    expected_files = ['config.json', 'training_history.json', 'agop_metrics.h5']
    missing_files = []
    
    for filename in expected_files:
        if not (output_dir / filename).exists():
            missing_files.append(filename)
    
    if missing_files:
        return {
            'success': False,
            'message': f'Missing files: {missing_files}',
        }
    
    # Check that files have content
    try:
        with open(output_dir / 'training_history.json') as f:
            history = json.load(f)
            n_epochs = len(history.get('epoch', []))
        
        return {
            'success': True,
            'message': f'All files present, {n_epochs} epochs logged',
        }
    
    except Exception as e:
        return {
            'success': False,
            'message': f'Error reading files: {e}',
        }


def run_all_tests(datasets=None, optimizers=None, device='cpu', seed=42):
    """Run all test combinations."""
    if datasets is None:
        datasets = list(TEST_CONFIGS.keys())
    if optimizers is None:
        optimizers = OPTIMIZERS
    
    results = {}
    total_tests = len(datasets) * len(optimizers)
    current_test = 0
    
    print("\n" + "="*80)
    print(f"AGOP EXPERIMENTS COMPREHENSIVE TEST SUITE")
    print("="*80)
    print(f"Datasets: {datasets}")
    print(f"Optimizers: {optimizers}")
    print(f"Total tests: {total_tests}")
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print("="*80)
    
    start_time = time.time()
    
    for dataset in datasets:
        results[dataset] = {}
        
        for optimizer in optimizers:
            current_test += 1
            print(f"\nTest {current_test}/{total_tests}")
            
            result = run_test(dataset, optimizer, device, seed)
            results[dataset][optimizer] = result
            
            # Verify outputs if successful
            if result['success'] and result['output_dir']:
                verification = verify_outputs(result['output_dir'])
                result['verification'] = verification
                
                if verification['success']:
                    print(f"  ✓ Outputs verified: {verification['message']}")
                else:
                    print(f"  ✗ Output verification failed: {verification['message']}")
    
    total_time = time.time() - start_time
    
    # Generate summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    successes = 0
    failures = 0
    
    for dataset, opt_results in results.items():
        print(f"\n{dataset.upper()}:")
        for optimizer, result in opt_results.items():
            status = "✓ PASS" if result['success'] else "✗ FAIL"
            print(f"  {optimizer:8s}: {status:8s} ({result['time']:.1f}s)")
            if result['success']:
                successes += 1
                if 'verification' in result and not result['verification']['success']:
                    print(f"           WARNING: {result['verification']['message']}")
            else:
                failures += 1
                if result['error']:
                    error_preview = result['error'][:100].replace('\n', ' ')
                    print(f"           Error: {error_preview}...")
    
    print("\n" + "="*80)
    print(f"RESULTS: {successes}/{total_tests} tests passed")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average time per test: {total_time/total_tests:.1f}s")
    print("="*80)
    
    # Save results
    results_file = Path('test_results') / f'test_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        # Convert Path objects to strings for JSON serialization
        results_serializable = {}
        for dataset, opt_results in results.items():
            results_serializable[dataset] = {}
            for optimizer, result in opt_results.items():
                result_copy = result.copy()
                if result_copy['output_dir']:
                    result_copy['output_dir'] = str(result_copy['output_dir'])
                results_serializable[dataset][optimizer] = result_copy
        
        json.dump({
            'summary': {
                'total_tests': total_tests,
                'successes': successes,
                'failures': failures,
                'total_time': total_time,
            },
            'results': results_serializable,
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    return successes == total_tests


def main():
    parser = argparse.ArgumentParser(description='Test AGOP experiments')
    parser.add_argument('--dataset', type=str, choices=list(TEST_CONFIGS.keys()) + ['all'],
                       default='all', help='Dataset to test')
    parser.add_argument('--optimizer', type=str, choices=OPTIMIZERS + ['all'],
                       default='all', help='Optimizer to test')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu for testing to avoid CUDA issues)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    
    args = parser.parse_args()
    
    # Determine what to test
    if args.all or args.dataset == 'all':
        datasets = list(TEST_CONFIGS.keys())
    else:
        datasets = [args.dataset]
    
    if args.all or args.optimizer == 'all':
        optimizers = OPTIMIZERS
    else:
        optimizers = [args.optimizer]
    
    # Run tests
    success = run_all_tests(datasets, optimizers, args.device, args.seed)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

