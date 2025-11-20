#!/usr/bin/env python
"""
Fixed converter for Paper 01 PyTorch Lightning CSV logs
Properly handles train/val metrics on separate rows
"""

import json
import os
import sys
import pandas as pd
import glob
from pathlib import Path
import numpy as np


def convert_csv_to_json(logdir):
    """Convert PyTorch Lightning CSV logs to training_history.json format"""
    csv_files = glob.glob(os.path.join(logdir, "*/metrics.csv"))
    
    if not csv_files:
        csv_files = glob.glob(os.path.join(logdir, "lightning_logs/*/metrics.csv"))
    
    if not csv_files:
        print(f"Error: No metrics.csv found in {logdir}")
        return None
    
    csv_file = sorted(csv_files)[-1]
    print(f"Reading metrics from: {csv_file}")
    
    df = pd.read_csv(csv_file)
    print(f"Found {len(df)} rows in CSV")
    
    # Group by epoch and aggregate
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    # Get unique epochs
    unique_epochs = df['epoch'].dropna().unique()
    unique_epochs.sort()
    
    for epoch in unique_epochs:
        epoch_rows = df[df['epoch'] == epoch]
        
        history['epoch'].append(int(epoch))
        
        # Extract train metrics (use full_train_acc if available, else train_accuracy)
        if 'full_train_acc' in epoch_rows.columns:
            train_acc_vals = epoch_rows['full_train_acc'].dropna()
            train_acc = train_acc_vals.iloc[-1] if len(train_acc_vals) > 0 else None
        elif 'train_accuracy' in epoch_rows.columns:
            train_acc_vals = epoch_rows['train_accuracy'].dropna()
            train_acc = train_acc_vals.iloc[-1] if len(train_acc_vals) > 0 else None
        else:
            train_acc = None
            
        if 'full_train_loss' in epoch_rows.columns:
            train_loss_vals = epoch_rows['full_train_loss'].dropna()
            train_loss = train_loss_vals.iloc[-1] if len(train_loss_vals) > 0 else None
        elif 'train_loss' in epoch_rows.columns:
            train_loss_vals = epoch_rows['train_loss'].dropna()
            train_loss = train_loss_vals.iloc[-1] if len(train_loss_vals) > 0 else None
        else:
            train_loss = None
        
        # Extract val/test metrics
        if 'val_accuracy' in epoch_rows.columns:
            val_acc_vals = epoch_rows['val_accuracy'].dropna()
            test_acc = val_acc_vals.iloc[-1] if len(val_acc_vals) > 0 else None
        else:
            test_acc = None
            
        if 'val_loss' in epoch_rows.columns:
            val_loss_vals = epoch_rows['val_loss'].dropna()
            test_loss = val_loss_vals.iloc[-1] if len(val_loss_vals) > 0 else None
        else:
            test_loss = None
        
        history['train_acc'].append(float(train_acc) if train_acc is not None else None)
        history['train_loss'].append(float(train_loss) if train_loss is not None else None)
        history['test_acc'].append(float(test_acc) if test_acc is not None else None)
        history['test_loss'].append(float(test_loss) if test_loss is not None else None)
    
    return history


if __name__ == '__main__':
    logdir = sys.argv[1] if len(sys.argv) > 1 else "./logs"
    logdir = os.path.abspath(logdir)
    output_file = os.path.join(logdir, 'training_history.json')
    
    print("="*80)
    print("Paper 01 Fixed Log Converter")
    print("="*80)
    print(f"Log directory: {logdir}")
    
    history = convert_csv_to_json(logdir)
    
    if history and len(history['epoch']) > 0:
        with open(output_file, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"✓ Converted {len(history['epoch'])} epochs to: {output_file}")
        
        # Print summary with None handling
        train_accs = [x for x in history['train_acc'] if x is not None]
        test_accs = [x for x in history['test_acc'] if x is not None]
        
        if train_accs:
            print(f"✓ Train accuracy: {min(train_accs):.2%} to {max(train_accs):.2%}")
        if test_accs:
            print(f"✓ Test accuracy: {min(test_accs):.2%} to {max(test_accs):.2%}")
    else:
        print("✗ No data found or conversion failed")
    
    print("="*80)

