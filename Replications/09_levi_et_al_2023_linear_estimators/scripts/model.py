"""
Linear Estimators for Grokking - Teacher-Student Framework
Levi et al. (2023) - Grokking in Linear Estimators

A solvable model demonstrating grokking in linear networks
"""

import torch
import torch.nn as nn
import numpy as np


class LinearStudent(nn.Module):
    """
    1-layer linear network (student)
    S: d_in x d_out matrix
    """
    def __init__(self, d_in, d_out):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out, bias=False)
        
        # Initialize as in paper: N(0, 1/(2*d_in*d_out))
        nn.init.normal_(self.linear.weight, mean=0, std=1/(2*d_in*d_out)**0.5)
    
    def forward(self, x):
        return self.linear(x)


class TwoLayerLinearStudent(nn.Module):
    """
    2-layer linear network: S = S0 @ S1
    S0: d_in x d_h
    S1: d_h x d_out
    """
    def __init__(self, d_in, d_h, d_out):
        super().__init__()
        self.layer1 = nn.Linear(d_in, d_h, bias=False)
        self.layer2 = nn.Linear(d_h, d_out, bias=False)
        
        # Initialize
        nn.init.normal_(self.layer1.weight, mean=0, std=1/(2*d_in*d_h)**0.5)
        nn.init.normal_(self.layer2.weight, mean=0, std=1/(2*d_h*d_out)**0.5)
    
    def forward(self, x):
        h = self.layer1(x)
        return self.layer2(h)


class TwoLayerNonlinearStudent(nn.Module):
    """
    2-layer network with tanh activation
    """
    def __init__(self, d_in, d_h, d_out):
        super().__init__()
        self.layer1 = nn.Linear(d_in, d_h, bias=False)
        self.layer2 = nn.Linear(d_h, d_out, bias=False)
        
        nn.init.normal_(self.layer1.weight, mean=0, std=1/(2*d_in*d_h)**0.5)
        nn.init.normal_(self.layer2.weight, mean=0, std=1/(2*d_h*d_out)**0.5)
    
    def forward(self, x):
        h = torch.tanh(self.layer1(x))
        return self.layer2(h)


class TeacherStudentDataset:
    """
    Gaussian dataset with linear teacher
    x_i ~ N(0, I)
    y_i = T^T @ x_i where T is the teacher matrix
    """
    def __init__(self, d_in, d_out, n_train, n_test, seed=42):
        self.d_in = d_in
        self.d_out = d_out
        self.n_train = n_train
        self.n_test = n_test
        
        # Set seed for reproducibility
        rng = np.random.RandomState(seed)
        
        # Create teacher matrix T: d_in x d_out
        # Initialized as N(0, 1/(2*d_in*d_out))
        self.T = torch.tensor(
            rng.normal(0, 1/(2*d_in*d_out)**0.5, size=(d_in, d_out)),
            dtype=torch.float32
        )
        
        # Generate training data
        self.X_train = torch.tensor(
            rng.normal(0, 1, size=(n_train, d_in)),
            dtype=torch.float32
        )
        self.Y_train = torch.matmul(self.X_train, self.T)
        
        # Generate test data
        self.X_test = torch.tensor(
            rng.normal(0, 1, size=(n_test, d_in)),
            dtype=torch.float32
        )
        self.Y_test = torch.matmul(self.X_test, self.T)
        
    def to(self, device):
        """Move data to device"""
        self.T = self.T.to(device)
        self.X_train = self.X_train.to(device)
        self.Y_train = self.Y_train.to(device)
        self.X_test = self.X_test.to(device)
        self.Y_test = self.Y_test.to(device)
        return self
    
    def get_train_data(self):
        return self.X_train, self.Y_train
    
    def get_test_data(self):
        return self.X_test, self.Y_test


def compute_accuracy(predictions, targets, epsilon=1e-3):
    """
    Compute accuracy with threshold epsilon
    Sample is correct if MSE per output < epsilon
    
    As in paper: A = Erf(sqrt(epsilon / (2*L)))
    where L is the loss
    """
    # MSE per sample
    mse_per_sample = ((predictions - targets) ** 2).mean(dim=1)
    
    # Count samples with error < epsilon
    correct = (mse_per_sample < epsilon).float().sum().item()
    accuracy = correct / len(predictions)
    
    return accuracy


def get_student_model(architecture, d_in, d_out, d_h=None):
    """
    Factory function to create student models
    
    Args:
        architecture: '1layer', '2layer_linear', '2layer_nonlinear'
        d_in: input dimension
        d_out: output dimension
        d_h: hidden dimension (for 2-layer models)
    """
    if architecture == '1layer':
        return LinearStudent(d_in, d_out)
    elif architecture == '2layer_linear':
        if d_h is None:
            raise ValueError("d_h must be specified for 2-layer models")
        return TwoLayerLinearStudent(d_in, d_h, d_out)
    elif architecture == '2layer_nonlinear':
        if d_h is None:
            raise ValueError("d_h must be specified for 2-layer models")
        return TwoLayerNonlinearStudent(d_in, d_h, d_out)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

