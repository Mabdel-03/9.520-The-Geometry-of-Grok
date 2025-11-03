"""
Wrapped Training Script for Doshi et al. (2024) - Modular Polynomials
Adds GOP tracking to power activation MLP
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "framework"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "Replications/08_doshi_et_al_2024_modular_polynomials"))

from framework import TrainingWrapper, HDF5Storage, ExperimentConfig
from model import ModularAdditionMLP, create_modular_addition_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_with_gop_tracking(config_path: str):
    config = ExperimentConfig(config_path=config_path)
    training_params = config.training
    device = torch.device(training_params['device'] if torch.cuda.is_available() else 'cpu')
    
    # Create dataset
    train_data, train_labels, test_data, test_labels = create_modular_addition_dataset(
        p=training_params['p'],
        num_terms=training_params['num_terms'],
        train_fraction=training_params['train_fraction']
    )
    train_data, train_labels = train_data.to(device), train_labels.to(device)
    test_data, test_labels = test_data.to(device), test_labels.to(device)
    
    # Create model
    model = ModularAdditionMLP(
        p=training_params['p'],
        num_terms=training_params['num_terms'],
        hidden_dim=training_params['hidden_dim'],
        power=training_params['power']
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    optimizer = optim.Adam(model.parameters(), lr=training_params['lr'],
                          weight_decay=training_params['weight_decay'])
    criterion = nn.MSELoss()
    
    storage = HDF5Storage(**config.storage)
    storage.save_config(config.to_dict())
    wrapper = TrainingWrapper(model, storage, config.gop_tracking, str(device))
    
    for epoch in range(training_params['epochs']):
        model.train()
        logits = model(train_data)
        
        train_labels_onehot = torch.zeros_like(logits)
        train_labels_onehot.scatter_(1, train_labels.unsqueeze(1), 1.0)
        train_loss = criterion(logits, train_labels_onehot)
        
        optimizer.zero_grad()
        train_loss.backward()
        
        with torch.no_grad():
            train_acc = (logits.argmax(dim=-1) == train_labels).float().mean().item()
        
        if epoch % training_params['log_interval'] == 0:
            model.eval()
            with torch.no_grad():
                test_logits = model(test_data)
                test_labels_onehot = torch.zeros_like(test_logits)
                test_labels_onehot.scatter_(1, test_labels.unsqueeze(1), 1.0)
                test_loss = criterion(test_logits, test_labels_onehot)
                test_acc = (test_logits.argmax(dim=-1) == test_labels).float().mean().item()
            
            logger.info(f"Epoch {epoch:5d} | Train: {train_loss.item():.4f}/{train_acc:.4f} | "
                       f"Test: {test_loss.item():.4f}/{test_acc:.4f}")
            
            wrapper.track_epoch(epoch, train_loss.item(), test_loss.item(), train_acc, test_acc)
        
        optimizer.step()
    
    logger.info("Training complete!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    train_with_gop_tracking(args.config)


if __name__ == '__main__':
    main()

