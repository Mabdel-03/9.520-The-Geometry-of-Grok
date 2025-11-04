"""
2-Layer MLP with Power Activation for Modular Polynomials
Doshi et al. (2024) - Grokking Modular Polynomials

Architecture: f(x) = W * phi(U^(1)*x_1 + ... + U^(S)*x_S)
where phi(x) = x^S (element-wise power activation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PowerActivationMLP(nn.Module):
    """
    2-layer MLP with power activation for modular polynomials
    
    For modular addition: f(n1, n2, ..., nS) = (c1*n1 + c2*n2 + ... + cS*nS) mod p
    For modular multiplication: f(n1, n2) = (n1^a * n2^b) mod p
    """
    def __init__(self, p, num_inputs=2, hidden_dim=500, power=2):
        """
        Args:
            p: modulus (prime)
            num_inputs: number of input numbers (S in paper)
            hidden_dim: width of hidden layer (N in paper)
            power: power for activation function (S in paper, confusingly same letter)
        """
        super().__init__()
        self.p = p
        self.num_inputs = num_inputs
        self.hidden_dim = hidden_dim
        self.power = power
        
        # Embedding matrices U^(1), ..., U^(S) for each input
        # Each U^(i): p x N (one-hot p-dim input -> N-dim hidden)
        self.embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(p, hidden_dim) / (p * hidden_dim) ** 0.5)
            for _ in range(num_inputs)
        ])
        
        # Output matrix W: N x p (N-dim hidden -> p-dim output)
        self.output_weight = nn.Parameter(torch.randn(hidden_dim, p) / hidden_dim ** 0.5)
        
    def power_activation(self, x):
        """Element-wise power activation: phi(x) = x^power"""
        return torch.pow(x, self.power)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, num_inputs) containing integers in [0, p-1]
        
        Returns:
            logits: (batch_size, p)
        """
        batch_size = x.shape[0]
        
        # Convert to one-hot and embed each input
        hidden = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        for i in range(self.num_inputs):
            # One-hot encode: (batch_size, p)
            one_hot = F.one_hot(x[:, i], num_classes=self.p).float()
            # Embed: (batch_size, N)
            embedded = torch.matmul(one_hot, self.embeddings[i])
            hidden = hidden + embedded
        
        # Apply power activation
        hidden = self.power_activation(hidden)
        
        # Output projection
        logits = torch.matmul(hidden, self.output_weight)
        
        return logits


class ModularAdditionMLP(PowerActivationMLP):
    """Specialized for modular addition with multiple terms"""
    def __init__(self, p, num_terms=2, hidden_dim=500, power=2):
        super().__init__(p, num_inputs=num_terms, hidden_dim=hidden_dim, power=power)


class ModularMultiplicationMLP(PowerActivationMLP):
    """
    Specialized for modular multiplication
    f(n1, n2) = (n1^a * n2^b) mod p
    """
    def __init__(self, p, hidden_dim=500):
        # For multiplication, we use power=1 and handle exponentiation differently
        super().__init__(p, num_inputs=2, hidden_dim=hidden_dim, power=1)
        
        # For multiplication, the network learns to compute in log space
        # This is a simplified version; full paper uses more sophisticated approach
        self.log_scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        """
        x: (batch_size, 2) containing [n1, n2]
        Returns: logits for (n1 * n2) mod p
        """
        # Use parent class forward for basic structure
        # In practice, multiplication requires learning discrete log/exp maps
        return super().forward(x)


def create_modular_addition_dataset(p, num_terms=2, train_fraction=0.5, coefficients=None):
    """
    Create dataset for modular addition: (c1*n1 + c2*n2 + ...) mod p
    
    Args:
        p: modulus
        num_terms: number of terms to add
        train_fraction: fraction for training
        coefficients: list of coefficients [c1, c2, ...], default all 1s
    """
    if coefficients is None:
        coefficients = [1] * num_terms
    
    # Generate all possible combinations
    import itertools
    all_combinations = list(itertools.product(range(p), repeat=num_terms))
    
    data = torch.tensor(all_combinations, dtype=torch.long)
    
    # Compute labels: (c1*n1 + c2*n2 + ...) mod p
    labels = torch.zeros(len(data), dtype=torch.long)
    for i, c in enumerate(coefficients):
        labels += c * data[:, i]
    labels = labels % p
    
    # Train/test split
    n_total = len(data)
    n_train = int(n_total * train_fraction)
    perm = torch.randperm(n_total)
    
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    
    return data[train_idx], labels[train_idx], data[test_idx], labels[test_idx]


def create_modular_multiplication_dataset(p, train_fraction=0.5, exponents=(1, 1)):
    """
    Create dataset for modular multiplication: (n1^a * n2^b) mod p
    
    Args:
        p: modulus (should be prime)
        train_fraction: fraction for training
        exponents: (a, b) for n1^a * n2^b
    """
    a, b = exponents
    
    # Generate all pairs
    data = []
    labels = []
    
    for n1 in range(p):
        for n2 in range(p):
            data.append([n1, n2])
            # Compute (n1^a * n2^b) mod p
            result = (pow(n1, a, p) * pow(n2, b, p)) % p
            labels.append(result)
    
    data = torch.tensor(data, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    
    # Train/test split
    n_total = len(data)
    n_train = int(n_total * train_fraction)
    perm = torch.randperm(n_total)
    
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    
    return data[train_idx], labels[train_idx], data[test_idx], labels[test_idx]

