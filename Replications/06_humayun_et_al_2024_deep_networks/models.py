"""
Models for Deep Networks Always Grok
Humayun et al. (2024)

Implements various architectures for MNIST, CIFAR-10, CIFAR-100
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Multi-layer perceptron with ReLU activations
    Used for MNIST experiments
    """
    def __init__(self, input_dim=784, hidden_dim=200, num_layers=4, num_classes=10):
        super().__init__()
        
        layers = []
        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        return self.network(x)


class SimpleCNN(nn.Module):
    """
    Simple CNN for CIFAR experiments
    5 convolutional layers + 2 linear layers
    """
    def __init__(self, num_classes=10, input_channels=3):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # Conv block 1
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Conv block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Conv block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class BasicBlock(nn.Module):
    """Basic ResNet block without batch normalization"""
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )
    
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18NoBN(nn.Module):
    """
    ResNet-18 without batch normalization
    Adapted for grokking experiments
    """
    def __init__(self, num_classes=10, width=16):
        super().__init__()
        self.in_planes = width
        
        self.conv1 = nn.Conv2d(3, width, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(BasicBlock, width, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, width * 2, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, width * 4, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, width * 8, 2, stride=2)
        self.linear = nn.Linear(width * 8 * BasicBlock.expansion, num_classes)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def get_model(model_name, dataset='mnist', **kwargs):
    """
    Factory function to get model by name
    
    Args:
        model_name: 'mlp', 'cnn', 'resnet18'
        dataset: 'mnist', 'cifar10', 'cifar100'
    """
    if model_name == 'mlp':
        if dataset == 'mnist':
            return MLP(input_dim=784, hidden_dim=kwargs.get('hidden_dim', 200),
                      num_layers=kwargs.get('num_layers', 4), num_classes=10)
        else:
            raise ValueError(f"MLP not typically used for {dataset}")
    
    elif model_name == 'cnn':
        num_classes = 10 if dataset in ['mnist', 'cifar10'] else 100
        input_channels = 1 if dataset == 'mnist' else 3
        return SimpleCNN(num_classes=num_classes, input_channels=input_channels)
    
    elif model_name == 'resnet18':
        num_classes = 10 if dataset in ['mnist', 'cifar10'] else 100
        width = kwargs.get('width', 16)
        return ResNet18NoBN(num_classes=num_classes, width=width)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")

