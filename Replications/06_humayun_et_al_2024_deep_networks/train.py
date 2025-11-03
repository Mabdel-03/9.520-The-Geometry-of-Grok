"""
Training script for Deep Networks Always Grok experiments
Humayun et al. (2024)

Supports MNIST, CIFAR-10, CIFAR-100 with various architectures
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
import argparse
from pathlib import Path

from models import get_model


def get_dataset(dataset_name, train_size=None, data_dir='./data'):
    """
    Load dataset with optional size reduction for grokking
    
    Args:
        dataset_name: 'mnist', 'cifar10', 'cifar100'
        train_size: Number of training samples (None = full dataset)
        data_dir: Directory to store datasets
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = torchvision.datasets.MNIST(root=data_dir, train=True, 
                                                    download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False, 
                                                   download=True, transform=transform)
    
    elif dataset_name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                                      download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                                     download=True, transform=transform_test)
    
    elif dataset_name == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True,
                                                       download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False,
                                                      download=True, transform=transform_test)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Reduce training size if specified (for grokking experiments)
    if train_size is not None and train_size < len(train_dataset):
        indices = np.random.choice(len(train_dataset), train_size, replace=False)
        train_dataset = Subset(train_dataset, indices)
    
    return train_dataset, test_dataset


def train_model(
    model_name='mlp',
    dataset='mnist',
    train_size=1000,
    batch_size=200,
    lr=0.001,
    weight_decay=0.01,
    n_epochs=100000,
    log_interval=100,
    save_dir='./checkpoints',
    data_dir='./data',
    device='cuda',
    seed=42
):
    """
    Train model and observe grokking
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create directories
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path('./logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading {dataset} dataset (train_size={train_size})")
    train_dataset, test_dataset = get_dataset(dataset, train_size, data_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    
    # Create model
    model = get_model(model_name, dataset).to(device)
    print(f"Model: {model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    # Training loop
    print("Starting training...")
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Evaluate on test set
        if epoch % log_interval == 0 or epoch == n_epochs - 1:
            model.eval()
            test_loss = 0
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    test_total += targets.size(0)
                    test_correct += predicted.eq(targets).sum().item()
            
            test_loss /= len(test_loader)
            test_acc = test_correct / test_total
            
            # Log metrics
            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            
            print(f"Epoch {epoch:5d} | Train Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} | "
                  f"Test Acc: {test_acc:.4f}")
            
            # Save checkpoint
            if epoch % 1000 == 0 or epoch == n_epochs - 1:
                checkpoint_path = save_dir / f'checkpoint_epoch_{epoch}.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                }, checkpoint_path)
    
    # Save training history
    history_path = log_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining complete! History saved to {history_path}")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train models for grokking experiments')
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'cnn', 'resnet18'],
                       help='Model architecture')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10', 'cifar100'],
                       help='Dataset')
    parser.add_argument('--train_size', type=int, default=1000, help='Training set size')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--n_epochs', type=int, default=100000, help='Number of epochs')
    parser.add_argument('--log_interval', type=int, default=100, help='Logging interval')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    train_model(
        model_name=args.model,
        dataset=args.dataset,
        train_size=args.train_size,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_epochs=args.n_epochs,
        log_interval=args.log_interval,
        save_dir=args.save_dir,
        data_dir=args.data_dir,
        device=args.device,
        seed=args.seed
    )


if __name__ == '__main__':
    main()

