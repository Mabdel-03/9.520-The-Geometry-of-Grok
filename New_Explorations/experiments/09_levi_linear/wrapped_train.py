"""
Wrapped Training Script for Levi et al. (2023) - Linear Estimators
Adds GOP tracking to teacher-student setup
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "Replications" / "09_levi_et_al_2023_linear_estimators"))

from framework import TrainingWrapper, HDF5Storage, ExperimentConfig
from model import TeacherStudentDataset, get_student_model, compute_accuracy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_with_gop_tracking(config_path: str):
    config = ExperimentConfig(config_path=config_path)
    training_params = config.training
    device = torch.device(training_params['device'] if torch.cuda.is_available() else 'cpu')
    
    # Create dataset
    dataset = TeacherStudentDataset(
        d_in=training_params['d_in'],
        d_out=training_params['d_out'],
        n_train=training_params['n_train'],
        n_test=training_params['n_test']
    ).to(device)
    
    X_train, Y_train = dataset.get_train_data()
    X_test, Y_test = dataset.get_test_data()
    
    # Create model
    model = get_student_model(
        architecture=training_params['architecture'],
        d_in=training_params['d_in'],
        d_out=training_params['d_out'],
        d_h=training_params.get('d_h')
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    optimizer = optim.SGD(model.parameters(), lr=training_params['lr'],
                         weight_decay=training_params['weight_decay'])
    criterion = nn.MSELoss()
    
    storage = HDF5Storage(**config.storage)
    storage.save_config(config.to_dict())
    wrapper = TrainingWrapper(model, storage, config.gop_tracking, str(device))
    
    for epoch in range(training_params['epochs']):
        model.train()
        Y_pred_train = model(X_train)
        train_loss = criterion(Y_pred_train, Y_train)
        
        optimizer.zero_grad()
        train_loss.backward()
        
        train_acc = compute_accuracy(Y_pred_train, Y_train, training_params['accuracy_threshold'])
        
        if epoch % training_params['log_interval'] == 0:
            model.eval()
            with torch.no_grad():
                Y_pred_test = model(X_test)
                test_loss = criterion(Y_pred_test, Y_test)
                test_acc = compute_accuracy(Y_pred_test, Y_test, training_params['accuracy_threshold'])
            
            logger.info(f"Epoch {epoch:6d} | Train: {train_loss.item():.6f}/{train_acc:.4f} | "
                       f"Test: {test_loss.item():.6f}/{test_acc:.4f}")
            
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

