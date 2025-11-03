"""
Training script for Linear Estimators (Teacher-Student)
Levi et al. (2023) - Grokking in Linear Estimators

Demonstrates grokking in a solvable linear model
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import argparse
from pathlib import Path

from model import TeacherStudentDataset, get_student_model, compute_accuracy


def train_model(
    architecture='1layer',
    d_in=1000,
    d_out=1,
    d_h=500,
    n_train=500,
    n_test=10000,
    lr=0.01,
    weight_decay=0.01,
    n_epochs=100000,
    log_interval=100,
    accuracy_threshold=1e-3,
    save_dir='./checkpoints',
    device='cuda',
    seed=42
):
    """
    Train student model in teacher-student setup
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
    
    # Create dataset
    print(f"Creating teacher-student dataset")
    print(f"d_in={d_in}, d_out={d_out}, n_train={n_train}, n_test={n_test}")
    print(f"Lambda (d_in/n_train) = {d_in/n_train:.3f}")
    
    dataset = TeacherStudentDataset(d_in, d_out, n_train, n_test, seed=seed).to(device)
    X_train, Y_train = dataset.get_train_data()
    X_test, Y_test = dataset.get_test_data()
    
    # Create student model
    model = get_student_model(architecture, d_in, d_out, d_h).to(device)
    print(f"Architecture: {architecture}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Optimizer (gradient flow: full batch GD with small lr)
    # Paper uses gradient flow limit, we approximate with small lr
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    # Training history
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'generalization_loss': []
    }
    
    # Training loop
    print("Starting training...")
    print(f"Accuracy threshold: {accuracy_threshold}")
    
    for epoch in range(n_epochs):
        model.train()
        
        # Forward pass (full batch)
        Y_pred_train = model(X_train)
        train_loss = criterion(Y_pred_train, Y_train)
        
        # Backward pass
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        # Compute train accuracy
        train_acc = compute_accuracy(Y_pred_train, Y_train, accuracy_threshold)
        
        # Evaluate
        if epoch % log_interval == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                Y_pred_test = model(X_test)
                test_loss = criterion(Y_pred_test, Y_test)
                test_acc = compute_accuracy(Y_pred_test, Y_test, accuracy_threshold)
                
                # Generalization loss (difference between test and train loss)
                gen_loss = test_loss.item() - train_loss.item()
            
            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss.item())
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss.item())
            history['test_acc'].append(test_acc)
            history['generalization_loss'].append(gen_loss)
            
            print(f"Epoch {epoch:6d} | Train Loss: {train_loss.item():.6f} | "
                  f"Train Acc: {train_acc:.4f} | Test Loss: {test_loss.item():.6f} | "
                  f"Test Acc: {test_acc:.4f} | Gen Loss: {gen_loss:.6f}")
            
            # Save checkpoints
            if epoch % 1000 == 0 or epoch == n_epochs - 1:
                checkpoint_path = save_dir / f'checkpoint_epoch_{epoch}.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'teacher_matrix': dataset.T,
                    'train_loss': train_loss.item(),
                    'test_loss': test_loss.item(),
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                }, checkpoint_path)
    
    # Save history
    history_path = log_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining complete! History saved to {history_path}")
    print("\nNote: Grokking appears as delayed jump in test_acc")
    print("This is due to accuracy crossing threshold, not true 'understanding'")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train linear estimator (teacher-student)')
    parser.add_argument('--architecture', type=str, default='1layer',
                       choices=['1layer', '2layer_linear', '2layer_nonlinear'],
                       help='Student architecture')
    parser.add_argument('--d_in', type=int, default=1000, help='Input dimension')
    parser.add_argument('--d_out', type=int, default=1, help='Output dimension')
    parser.add_argument('--d_h', type=int, default=500, help='Hidden dimension (for 2-layer)')
    parser.add_argument('--n_train', type=int, default=500, help='Training samples')
    parser.add_argument('--n_test', type=int, default=10000, help='Test samples')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay (gamma)')
    parser.add_argument('--n_epochs', type=int, default=100000, help='Number of epochs')
    parser.add_argument('--log_interval', type=int, default=100, help='Logging interval')
    parser.add_argument('--accuracy_threshold', type=float, default=1e-3,
                       help='Accuracy threshold epsilon')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Save directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    train_model(
        architecture=args.architecture,
        d_in=args.d_in,
        d_out=args.d_out,
        d_h=args.d_h,
        n_train=args.n_train,
        n_test=args.n_test,
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_epochs=args.n_epochs,
        log_interval=args.log_interval,
        accuracy_threshold=args.accuracy_threshold,
        save_dir=args.save_dir,
        device=args.device,
        seed=args.seed
    )


if __name__ == '__main__':
    main()

