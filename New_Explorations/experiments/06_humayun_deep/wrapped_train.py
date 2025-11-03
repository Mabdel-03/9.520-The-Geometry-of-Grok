"""
Wrapped Training Script for Humayun et al. (2024) - Deep Networks Always Grok
Adds GOP tracking to MNIST MLP experiments
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "framework"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "Replications/06_humayun_et_al_2024_deep_networks"))

from framework import TrainingWrapper, HDF5Storage, ExperimentConfig
from models import get_model
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_with_gop_tracking(config_path: str):
    config = ExperimentConfig(config_path=config_path)
    training_params = config.training
    device = torch.device(training_params['device'] if torch.cuda.is_available() else 'cpu')
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Reduce training size
    if training_params['train_size'] < len(train_dataset):
        indices = np.random.choice(len(train_dataset), training_params['train_size'], replace=False)
        train_dataset = Subset(train_dataset, indices)
    
    train_loader = DataLoader(train_dataset, batch_size=training_params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=training_params['batch_size'], shuffle=False)
    
    logger.info(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    
    # Create model
    model = get_model(training_params['model_name'], training_params['dataset']).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    optimizer = optim.Adam(model.parameters(), lr=training_params['lr'], 
                          weight_decay=training_params['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    
    # Initialize storage and wrapper
    storage = HDF5Storage(**config.storage)
    storage.save_config(config.to_dict())
    wrapper = TrainingWrapper(model, storage, config.gop_tracking, str(device))
    
    # Training loop
    for epoch in range(training_params['epochs']):
        model.train()
        train_loss_sum = 0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        train_loss = train_loss_sum / len(train_loader)
        train_acc = train_correct / train_total
        
        if epoch % training_params['log_interval'] == 0:
            # Evaluate on test set
            model.eval()
            test_loss_sum = 0
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    test_loss_sum += loss.item()
                    _, predicted = outputs.max(1)
                    test_total += targets.size(0)
                    test_correct += predicted.eq(targets).sum().item()
            
            test_loss = test_loss_sum / len(test_loader)
            test_acc = test_correct / test_total
            
            logger.info(f"Epoch {epoch:5d} | Train: {train_loss:.4f}/{train_acc:.4f} | "
                       f"Test: {test_loss:.4f}/{test_acc:.4f}")
            
            # For GOP tracking, need gradients on a batch
            # Recompute on first batch
            model.train()
            inputs, targets = next(iter(train_loader))
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Track GOP
            wrapper.track_epoch(epoch, train_loss, test_loss, train_acc, test_acc)
    
    logger.info("Training complete!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    train_with_gop_tracking(args.config)


if __name__ == '__main__':
    main()

