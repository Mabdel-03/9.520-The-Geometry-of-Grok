"""
Template for Creating Wrapped Training Scripts

Copy this template and adapt for your specific experiment.
Follow the TODO markers to customize.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import logging

# Add framework path
sys.path.insert(0, str(Path(__file__).parent.parent / "framework"))

# TODO: Add path to your replication directory
# sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Replications/XX_your_paper"))

from framework import TrainingWrapper, HDF5Storage, ExperimentConfig

# TODO: Import your model and dataset functions
# from model import YourModel, create_dataset

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
    
    # TODO: Create your dataset
    # train_data, train_labels, test_data, test_labels = create_dataset(...)
    # Move to device
    # train_data = train_data.to(device)
    # ...
    
    # TODO: Create your model
    # model = YourModel(
    #     param1=training_params['param1'],
    #     param2=training_params['param2'],
    #     ...
    # ).to(device)
    
    # Log model size
    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # logger.info(f"Model created with {total_params:,} parameters")
    
    # TODO: Create optimizer and loss function
    # optimizer = optim.AdamW(model.parameters(), lr=training_params['lr'],
    #                         weight_decay=training_params['weight_decay'])
    # criterion = nn.CrossEntropyLoss()  # or MSELoss, etc.
    
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
        model=None,  # TODO: Replace with your model
        storage=storage,
        gop_config=config.gop_tracking,
        device=str(device)
    )
    
    # Estimate memory usage
    memory_est = wrapper.get_memory_estimate()
    logger.info(f"Estimated GOP memory: {memory_est['total_gb']:.2f} GB per epoch")
    
    # Training loop
    logger.info(f"Starting training for {training_params['epochs']} epochs")
    
    for epoch in range(training_params['epochs']):
        # TODO: Your training step
        # model.train()
        # logits = model(train_data)
        # train_loss = criterion(logits, train_labels)
        
        # optimizer.zero_grad()
        # train_loss.backward()
        
        # Compute training accuracy
        # with torch.no_grad():
        #     train_preds = logits.argmax(dim=-1)
        #     train_acc = (train_preds == train_labels).float().mean().item()
        
        # Evaluate on test set
        if epoch % training_params['log_interval'] == 0:
            # TODO: Your evaluation step
            # model.eval()
            # with torch.no_grad():
            #     test_logits = model(test_data)
            #     test_loss = criterion(test_logits, test_labels)
            #     test_preds = test_logits.argmax(dim=-1)
            #     test_acc = (test_preds == test_labels).float().mean().item()
            
            # TODO: Log progress
            # logger.info(f"Epoch {epoch} | Train: {train_loss:.4f}/{train_acc:.4f} | "
            #            f"Test: {test_loss:.4f}/{test_acc:.4f}")
            
            # CRITICAL: Track GOP BEFORE optimizer.step()
            # wrapper.track_epoch(
            #     epoch=epoch,
            #     train_loss=train_loss.item(),
            #     test_loss=test_loss.item(),
            #     train_acc=train_acc,
            #     test_acc=test_acc
            # )
            pass
        
        # TODO: Optimizer step
        # optimizer.step()
    
    logger.info("Training complete!")
    logger.info(f"Results saved to: {config.storage['output_dir']}")


def main():
    parser = argparse.ArgumentParser(description='Train with GOP tracking')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--dry_run', action='store_true',
                       help='Estimate storage without training')
    
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("Dry run mode - estimating requirements")
        # TODO: Load config and estimate storage
        config = ExperimentConfig(config_path=args.config)
        logger.info(f"Experiment: {config.experiment_name}")
        # Estimate and print
        return
    
    try:
        train_with_gop_tracking(args.config)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

