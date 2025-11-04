#!/usr/bin/env python
"""
Wrapper for Paper 01 (Power et al. 2022) that captures training metrics
and saves them in the standard training_history.json format.
"""

import json
import os
import sys
from pathlib import Path
import pandas as pd
import glob

# Add the grok module to path
sys.path.insert(0, str(Path(__file__).parent))

import grok


def convert_csv_to_json(logdir):
    """Convert PyTorch Lightning CSV logs to training_history.json format"""
    # Find the metrics CSV file
    csv_files = glob.glob(os.path.join(logdir, "*/metrics.csv"))
    
    if not csv_files:
        csv_files = glob.glob(os.path.join(logdir, "lightning_logs/*/metrics.csv"))
    
    if not csv_files:
        print(f"Warning: No metrics.csv found in {logdir}")
        return None
    
    # Use the most recent version
    csv_file = sorted(csv_files)[-1]
    print(f"Reading metrics from: {csv_file}")
    
    df = pd.read_csv(csv_file)
    
    # Initialize output dictionary
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    # Group by step (multiple rows per step due to train/val logging)
    for step, group in df.groupby('step'):
        # Get the last row for this step (most complete)
        row = group.iloc[-1]
        
        history['epoch'].append(int(row['epoch']) if 'epoch' in row and not pd.isna(row['epoch']) else int(step))
        
        # Extract metrics (handle different possible column names)
        history['train_loss'].append(float(row['train_loss']) if 'train_loss' in row and not pd.isna(row['train_loss']) else None)
        history['train_acc'].append(float(row['train_accuracy']) if 'train_accuracy' in row and not pd.isna(row['train_accuracy']) else None)
        history['test_loss'].append(float(row['val_loss']) if 'val_loss' in row and not pd.isna(row['val_loss']) else None)
        history['test_acc'].append(float(row['val_accuracy']) if 'val_accuracy' in row and not pd.isna(row['val_accuracy']) else None)
    
    # Remove None values (keep lists aligned)
    return history


if __name__ == '__main__':
    parser = grok.training.add_args()
    parser.set_defaults(logdir=os.path.abspath("./logs"))
    hparams = parser.parse_args()
    hparams.datadir = os.path.abspath(hparams.datadir)
    hparams.logdir = os.path.abspath(hparams.logdir)
    
    # Create logs directory
    os.makedirs(hparams.logdir, exist_ok=True)
    log_file = os.path.join(hparams.logdir, 'training_history.json')
    
    print("=" * 80)
    print("Paper 01: Power et al. (2022) - OpenAI Grok")
    print("=" * 80)
    print(f"Logging directory: {hparams.logdir}")
    print(f"Hyperparameters:")
    print(f"  - Operation: {hparams.math_operator}")
    print(f"  - Training data: {hparams.train_data_pct*100}%")
    print(f"  - Weight decay: {hparams.weight_decay}")
    print(f"  - Learning rate: {hparams.max_lr}")
    print(f"  - Max steps: {hparams.max_steps}")
    print("=" * 80)
    
    # Train model
    result = grok.training.train(hparams)
    
    print("=" * 80)
    print("Training complete! Converting logs to standard format...")
    
    # Convert CSV logs to JSON
    history = convert_csv_to_json(hparams.logdir)
    
    if history:
        with open(log_file, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"✓ Metrics saved to: {log_file}")
        print(f"✓ Total epochs logged: {len(history['epoch'])}")
    else:
        print("✗ Failed to convert metrics")
    
    print("=" * 80)

