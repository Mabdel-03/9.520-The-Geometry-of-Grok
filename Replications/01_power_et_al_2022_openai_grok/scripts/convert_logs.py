#!/usr/bin/env python
"""
Convert PyTorch Lightning CSV logs from Paper 01 to training_history.json format
Run this after training completes.
"""

import json
import os
import sys
import pandas as pd
import glob
from pathlib import Path


def convert_csv_to_json(logdir):
    """Convert PyTorch Lightning CSV logs to training_history.json format"""
    # Find the metrics CSV file
    csv_files = glob.glob(os.path.join(logdir, "version_*/metrics.csv"))
    
    if not csv_files:
        csv_files = glob.glob(os.path.join(logdir, "lightning_logs/version_*/metrics.csv"))
    
    if not csv_files:
        print(f"Error: No metrics.csv found in {logdir}")
        return None
    
    # Use the most recent version
    csv_file = sorted(csv_files)[-1]
    print(f"Reading metrics from: {csv_file}")
    
    df = pd.read_csv(csv_file)
    print(f"Found {len(df)} rows in CSV")
    print(f"Columns: {df.columns.tolist()}")
    
    # Initialize output dictionary
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    # Group by epoch
    for epoch, group in df.groupby('epoch'):
        if pd.isna(epoch):
            continue
            
        # Get the last row for this epoch (most complete)
        row = group.iloc[-1]
        
        history['epoch'].append(int(epoch))
        
        # Extract metrics (handle different possible column names)
        train_loss = row['train_loss'] if 'train_loss' in row and not pd.isna(row['train_loss']) else None
        train_acc = row['train_accuracy'] if 'train_accuracy' in row and not pd.isna(row['train_accuracy']) else None
        test_loss = row['val_loss'] if 'val_loss' in row and not pd.isna(row['val_loss']) else None
        test_acc = row['val_accuracy'] if 'val_accuracy' in row and not pd.isna(row['val_accuracy']) else None
        
        history['train_loss'].append(float(train_loss) if train_loss is not None else None)
        history['train_acc'].append(float(train_acc) if train_acc is not None else None)
        history['test_loss'].append(float(test_loss) if test_loss is not None else None)
        history['test_acc'].append(float(test_acc) if test_acc is not None else None)
    
    return history


if __name__ == '__main__':
    if len(sys.argv) > 1:
        logdir = sys.argv[1]
    else:
        logdir = "./logs"
    
    logdir = os.path.abspath(logdir)
    output_file = os.path.join(logdir, 'training_history.json')
    
    print("=" * 80)
    print("Paper 01 Log Converter")
    print("=" * 80)
    print(f"Log directory: {logdir}")
    
    history = convert_csv_to_json(logdir)
    
    if history and len(history['epoch']) > 0:
        with open(output_file, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"✓ Converted {len(history['epoch'])} epochs to: {output_file}")
        
        # Print summary
        non_none_train = [x for x in history['train_acc'] if x is not None]
        non_none_test = [x for x in history['test_acc'] if x is not None]
        if non_none_train:
            print(f"✓ Final train accuracy: {non_none_train[-1]:.2%}")
        if non_none_test:
            print(f"✓ Final test accuracy: {non_none_test[-1]:.2%}")
    else:
        print("✗ No data found or conversion failed")
    
    print("=" * 80)

