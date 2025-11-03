"""
Wrapped Training Script for Thilak et al. (2022) - Slingshot Mechanism
Adds GOP tracking to observe Slingshot dynamics
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import logging

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "framework"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "Replications/07_thilak_et_al_2022_slingshot"))

from framework import TrainingWrapper, HDF5Storage, ExperimentConfig
from model import ModularArithmeticTransformer, create_modular_addition_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_with_gop_tracking(config_path: str):
    """Train model with GOP tracking."""
    config = ExperimentConfig(config_path=config_path)
    logger.info(f"Starting experiment: {config.experiment_name}")
    
    training_params = config.training
    device = torch.device(training_params['device'] if torch.cuda.is_available() else 'cpu')
    
    # Create dataset
    train_data, train_labels, test_data, test_labels = create_modular_addition_dataset(
        p=training_params['p'],
        train_fraction=training_params['train_fraction'],
        device=device
    )
    
    # Create model
    model = ModularArithmeticTransformer(
        p=training_params['p'],
        d_model=training_params['d_model'],
        n_heads=training_params['n_heads'],
        n_layers=training_params['n_layers'],
        d_mlp=training_params['d_mlp']
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create optimizer
    if training_params['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=training_params['lr'], 
                              weight_decay=training_params['weight_decay'])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=training_params['lr'],
                               weight_decay=training_params['weight_decay'])
    
    criterion = nn.CrossEntropyLoss()
    
    # Initialize storage and wrapper
    storage = HDF5Storage(
        output_dir=config.storage['output_dir'],
        compression=config.storage['compression'],
        compression_level=config.storage['compression_level'],
        store_full_gop=config.storage['store_full_gop'],
        store_eigenvectors=config.gop_tracking['store_eigenvectors'],
        float_precision=config.storage['float_precision']
    )
    storage.save_config(config.to_dict())
    
    wrapper = TrainingWrapper(model, storage, config.gop_tracking, str(device))
    
    # Training loop
    logger.info(f"Starting training for {training_params['epochs']} epochs")
    
    for epoch in range(training_params['epochs']):
        model.train()
        logits = model(train_data)
        train_loss = criterion(logits, train_labels)
        
        optimizer.zero_grad()
        train_loss.backward()
        
        with torch.no_grad():
            train_preds = logits.argmax(dim=-1)
            train_acc = (train_preds == train_labels).float().mean().item()
        
        if epoch % training_params['log_interval'] == 0:
            model.eval()
            with torch.no_grad():
                test_logits = model(test_data)
                test_loss = criterion(test_logits, test_labels)
                test_preds = test_logits.argmax(dim=-1)
                test_acc = (test_preds == test_labels).float().mean().item()
                
                # Track last layer norm for Slingshot
                last_layer_norm = model.get_last_layer_norm()
            
            logger.info(f"Epoch {epoch:5d} | Train: {train_loss.item():.4f}/{train_acc:.4f} | "
                       f"Test: {test_loss.item():.4f}/{test_acc:.4f} | "
                       f"Last Layer Norm: {last_layer_norm:.2f}")
            
            wrapper.track_epoch(epoch, train_loss.item(), test_loss.item(), train_acc, test_acc)
        
        optimizer.step()
    
    logger.info("Training complete!")


def main():
    parser = argparse.ArgumentParser(description='Train with GOP tracking - Slingshot')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    args = parser.parse_args()
    train_with_gop_tracking(args.config)


if __name__ == '__main__':
    main()

