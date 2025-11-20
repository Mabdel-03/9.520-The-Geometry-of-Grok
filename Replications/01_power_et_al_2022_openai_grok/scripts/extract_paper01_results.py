#!/usr/bin/env python
"""
Extract Paper 01 results from PyTorch Lightning CSV logs
Handles the complex format where train and validation are on separate rows
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path


def extract_metrics(csv_file):
    """Extract and merge train/val metrics from PyTorch Lightning CSV"""
    
    df = pd.read_csv(csv_file)
    
    print(f"Total rows in CSV: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    
    # Get dataset sizes (for normalizing accuracy counts)
    train_size = df['len_train_ds'].dropna().iloc[0]
    val_size = df['len_val_ds'].dropna().iloc[0]
    print(f"Dataset sizes: Train={train_size:.0f}, Validation={val_size:.0f}")
    
    # Group by step to merge train and val rows
    # PyTorch Lightning logs train and val on separate rows
    merged_data = []
    
    for step in df['step'].dropna().unique():
        step_rows = df[df['step'] == step]
        
        # Get values, preferring non-null
        epoch = step_rows['epoch'].dropna().iloc[0] if len(step_rows['epoch'].dropna()) > 0 else step
        
        # Train metrics - full_train_acc is stored as 0-100 percentage, convert to 0-1
        full_train_rows = step_rows['full_train_acc'].dropna()
        if len(full_train_rows) > 0:
            train_acc = full_train_rows.iloc[0] / 100.0  # Convert percentage to fraction
        else:
            train_acc_rows = step_rows['train_accuracy'].dropna()
            train_acc = train_acc_rows.iloc[0] / train_size if len(train_acc_rows) > 0 else None
        
        train_loss_rows = step_rows['train_loss'].dropna()
        train_loss = train_loss_rows.iloc[0] if len(train_loss_rows) > 0 else None
        
        # Use full_train_loss if available
        if train_loss is None:
            full_train_loss_rows = step_rows['full_train_loss'].dropna()
            train_loss = full_train_loss_rows.iloc[0] if len(full_train_loss_rows) > 0 else None
        
        # Validation metrics (stored as 0-100 percentage, convert to 0-1)
        val_acc_rows = step_rows['val_accuracy'].dropna()
        val_acc = val_acc_rows.iloc[0] / 100.0 if len(val_acc_rows) > 0 else None
        
        val_loss_rows = step_rows['val_loss'].dropna()
        val_loss = val_loss_rows.iloc[0] if len(val_loss_rows) > 0 else None
        
        merged_data.append({
            'step': int(step),
            'epoch': int(epoch) if not pd.isna(epoch) else int(step),
            'train_acc': train_acc,
            'train_loss': train_loss,
            'test_acc': val_acc,  # Use val as test
            'test_loss': val_loss
        })
    
    # Convert to clean format
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    for row in merged_data:
        # Only include rows that have at least test accuracy
        if row['test_acc'] is not None:
            history['epoch'].append(row['epoch'])
            history['train_loss'].append(float(row['train_loss']) if row['train_loss'] is not None else None)
            history['train_acc'].append(float(row['train_acc']) if row['train_acc'] is not None else None)
            history['test_loss'].append(float(row['test_loss']) if row['test_loss'] is not None else None)
            history['test_acc'].append(float(row['test_acc']))
    
    return history


if __name__ == '__main__':
    print("="*80)
    print("PAPER 01: EXTRACTING RESULTS FROM PYTORCH LIGHTNING LOGS")
    print("="*80)
    
    # Find the latest version directory
    import glob
    version_dirs = glob.glob('logs/lightning_logs/version_*/metrics.csv')
    if not version_dirs:
        print("ERROR: No metrics.csv found!")
        exit(1)
    csv_file = Path(sorted(version_dirs)[-1])
    output_file = Path('logs/training_history.json')
    
    print(f"\nReading: {csv_file}")
    history = extract_metrics(csv_file)
    
    print(f"\nExtracted {len(history['epoch'])} epochs with complete data")
    print(f"Epoch range: {history['epoch'][0]} to {history['epoch'][-1]}")
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✓ Saved to: {output_file}")
    
    # Print summary statistics
    train_acc_clean = [x for x in history['train_acc'] if x is not None]
    test_acc_clean = [x for x in history['test_acc'] if x is not None]
    
    if train_acc_clean and test_acc_clean:
        print(f"\nSummary:")
        print(f"  Train accuracy: {min(train_acc_clean):.2%} → {max(train_acc_clean):.2%}")
        print(f"  Test accuracy: {min(test_acc_clean):.2%} → {max(test_acc_clean):.2%}")
        print(f"  Final test accuracy: {test_acc_clean[-1]:.2%}")
        
        # Check for grokking
        if max(train_acc_clean) > 0.90 and max(test_acc_clean) > 0.70:
            print(f"\n✅ GROKKING DETECTED!")
            print(f"   Train reached high accuracy")
            print(f"   Test improved to {max(test_acc_clean):.1%}")
    
    print("="*80)

