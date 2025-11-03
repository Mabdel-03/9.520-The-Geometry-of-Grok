"""
Wrapped Training Script for Nanda et al. (2023) with GOP Tracking

This script wraps the original Nanda training code to add GOP tracking.
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
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "Replications/03_nanda_et_al_2023_progress_measures"))

from framework import TrainingWrapper, HDF5Storage, ExperimentConfig
from model import OneLayerReLUTransformer, create_modular_addition_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_with_gop_tracking(config_path: str):
    """
    Train model with GOP tracking.
    
    Args:
        config_path: Path to configuration YAML file
    """
    # Load configuration
    config = ExperimentConfig(config_path=config_path)
    logger.info(f"Starting experiment: {config.experiment_name}")
    
    # Get training parameters
    training_params = config.training
    device = torch.device(training_params['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create dataset
    logger.info(f"Creating dataset: p={training_params['p']}, train_fraction={training_params['train_fraction']}")
    train_data, train_labels, test_data, test_labels = create_modular_addition_dataset(
        p=training_params['p'],
        train_fraction=training_params['train_fraction'],
        device=device
    )
    logger.info(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    
    # Create model
    model = OneLayerReLUTransformer(
        p=training_params['p'],
        d_model=training_params['d_model'],
        n_heads=training_params['n_heads'],
        d_mlp=training_params['d_mlp']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created with {total_params:,} parameters")
    
    # Create optimizer and loss
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_params['lr'],
        weight_decay=training_params['weight_decay']
    )
    criterion = nn.CrossEntropyLoss()
    
    # Initialize storage
    storage = HDF5Storage(
        output_dir=config.storage['output_dir'],
        compression=config.storage['compression'],
        compression_level=config.storage['compression_level'],
        store_full_gop=config.storage['store_full_gop'],
        store_eigenvectors=config.gop_tracking['store_eigenvectors'],
        float_precision=config.storage['float_precision']
    )
    
    # Save configuration
    storage.save_config(config.to_dict())
    
    # Initialize training wrapper
    wrapper = TrainingWrapper(
        model=model,
        storage=storage,
        gop_config=config.gop_tracking,
        device=str(device)
    )
    
    # Estimate memory usage
    memory_est = wrapper.get_memory_estimate()
    logger.info(f"Estimated GOP memory usage: {memory_est['total_gb']:.2f} GB per epoch")
    
    # Training loop
    logger.info(f"Starting training for {training_params['epochs']} epochs")
    
    for epoch in range(training_params['epochs']):
        model.train()
        
        # Forward pass (full batch)
        logits = model(train_data)
        train_loss = criterion(logits, train_labels)
        
        # Backward pass
        optimizer.zero_grad()
        train_loss.backward()
        
        # Compute training accuracy
        with torch.no_grad():
            train_preds = logits.argmax(dim=-1)
            train_acc = (train_preds == train_labels).float().mean().item()
        
        # Evaluate on test set
        if epoch % training_params['log_interval'] == 0 or epoch == training_params['epochs'] - 1:
            model.eval()
            with torch.no_grad():
                test_logits = model(test_data)
                test_loss = criterion(test_logits, test_labels)
                test_preds = test_logits.argmax(dim=-1)
                test_acc = (test_preds == test_labels).float().mean().item()
            
            logger.info(
                f"Epoch {epoch:5d} | Train Loss: {train_loss.item():.4f} | "
                f"Train Acc: {train_acc:.4f} | Test Loss: {test_loss.item():.4f} | "
                f"Test Acc: {test_acc:.4f}"
            )
            
            # Track GOP (before optimizer.step()!)
            try:
                wrapper.track_epoch(
                    epoch=epoch,
                    train_loss=train_loss.item(),
                    test_loss=test_loss.item(),
                    train_acc=train_acc,
                    test_acc=test_acc
                )
            except Exception as e:
                logger.error(f"Error tracking GOP at epoch {epoch}: {e}")
                # Continue training even if GOP tracking fails
        
        # Optimizer step
        optimizer.step()
        
        # Periodic storage size check
        if epoch % 1000 == 0 and epoch > 0:
            sizes = storage.get_file_sizes()
            logger.info(f"Storage usage: {sizes['total']:.2f} GB")
    
    logger.info("Training complete!")
    logger.info(f"Results saved to: {config.storage['output_dir']}")
    
    # Final storage info
    final_sizes = storage.get_file_sizes()
    logger.info(f"Final storage usage:")
    for name, size in final_sizes.items():
        logger.info(f"  {name}: {size:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description='Train with GOP tracking')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    
    args = parser.parse_args()
    
    try:
        train_with_gop_tracking(args.config)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

