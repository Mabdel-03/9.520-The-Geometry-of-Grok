#!/usr/bin/env python
"""
Wrapper for Paper 01 (Power et al. 2022) that captures training metrics
and saves them in the standard training_history.json format.
"""

import json
import os
import sys
from pathlib import Path

# Add the grok module to path
sys.path.insert(0, str(Path(__file__).parent))

import grok
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

class MetricsLogger(Callback):
    """Custom callback to log training metrics"""
    def __init__(self, log_file):
        super().__init__()
        self.log_file = log_file
        self.metrics = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': []
        }
    
    def on_validation_end(self, trainer, pl_module):
        """Called at the end of validation"""
        # Get metrics from trainer
        metrics = trainer.callback_metrics
        
        self.metrics['epoch'].append(trainer.current_epoch)
        
        # Extract train metrics
        if 'train_loss' in metrics:
            self.metrics['train_loss'].append(float(metrics['train_loss']))
        if 'train_accuracy' in metrics:
            self.metrics['train_acc'].append(float(metrics['train_accuracy']))
        
        # Extract validation/test metrics  
        if 'val_loss' in metrics:
            self.metrics['test_loss'].append(float(metrics['val_loss']))
        if 'val_accuracy' in metrics:
            self.metrics['test_acc'].append(float(metrics['val_accuracy']))
        
        # Save after each logging step
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def on_train_end(self, trainer, pl_module):
        """Save final metrics"""
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)


if __name__ == '__main__':
    parser = grok.training.add_args()
    parser.set_defaults(logdir=os.path.abspath("./logs"))
    hparams = parser.parse_args()
    hparams.datadir = os.path.abspath(hparams.datadir)
    hparams.logdir = os.path.abspath(hparams.logdir)
    
    # Create logs directory
    os.makedirs(hparams.logdir, exist_ok=True)
    log_file = os.path.join(hparams.logdir, 'training_history.json')
    
    print(f"Logging directory: {hparams.logdir}")
    print(f"Metrics will be saved to: {log_file}")
    print(f"Hyperparameters: {hparams}")
    
    # Train with custom logger
    result = grok.training.train(hparams)
    print(f"Training complete! Results saved to {result}")

