#!/usr/bin/env python
"""
Modified MNIST Grokking experiment that saves training_history.json
Based on Paper 05: Liu et al. (2022) - Omnigrok
"""

from collections import defaultdict
from itertools import islice
import random
import time
from pathlib import Path
import math
import json

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torchvision

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def compute_accuracy(network, dataset, device, N=2000, batch_size=50):
    """Computes accuracy of `network` on `dataset`."""
    with torch.no_grad():
        N = min(len(dataset), N)
        batch_size = min(batch_size, N)
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        correct = 0
        total = 0
        for x, labels in islice(dataset_loader, N // batch_size):
            logits = network(x.to(device))
            predicted_labels = torch.argmax(logits, dim=1)
            correct += torch.sum(predicted_labels == labels.to(device))
            total += x.size(0)
        return (correct / total).item()

def compute_loss(network, dataset, loss_function, device, N=2000, batch_size=50):
    """Computes mean loss of `network` on `dataset`."""
    with torch.no_grad():
        N = min(len(dataset), N)
        batch_size = min(batch_size, N)
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        loss_fn = loss_function_dict[loss_function](reduction='sum')
        one_hots = torch.eye(10, 10).to(device)
        total = 0
        points = 0
        for x, labels in islice(dataset_loader, N // batch_size):
            y = network(x.to(device))
            if loss_function == 'CrossEntropy':
                total += loss_fn(y, labels.to(device)).item()
            elif loss_function == 'MSE':
                total += loss_fn(y, one_hots[labels]).item()
            points += len(labels)
        return total / points

optimizer_dict = {
    'AdamW': torch.optim.AdamW,
    'Adam': torch.optim.Adam,
    'SGD': torch.optim.SGD
}

activation_dict = {
    'ReLU': nn.ReLU,
    'Tanh': nn.Tanh,
    'Sigmoid': nn.Sigmoid,
    'GELU': nn.GELU
}

loss_function_dict = {
    'MSE': nn.MSELoss,
    'CrossEntropy': nn.CrossEntropyLoss
}

# Hyperparameters
train_points = 1000
optimization_steps = 100000
batch_size = 200
loss_function = 'MSE'
optimizer_name = 'AdamW'
weight_decay = 0.01
lr = 1e-3
initialization_scale = 8.0
download_directory = "."

depth = 3
width = 200
activation = 'ReLU'

log_freq = math.ceil(optimization_steps / 150)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.float64
seed = 0

torch.set_default_dtype(dtype)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)

print(f"Device: {device}")
print(f"Training points: {train_points}")
print(f"Optimization steps: {optimization_steps}")
print(f"Weight decay: {weight_decay}")

# Load dataset
train = torchvision.datasets.MNIST(root=download_directory, train=True, 
    transform=torchvision.transforms.ToTensor(), download=True)
test = torchvision.datasets.MNIST(root=download_directory, train=False, 
    transform=torchvision.transforms.ToTensor(), download=True)
train = torch.utils.data.Subset(train, range(train_points))
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

activation_fn = activation_dict[activation]

# Create model
layers = [nn.Flatten()]
for i in range(depth):
    if i == 0:
        layers.append(nn.Linear(784, width))
        layers.append(activation_fn())
    elif i == depth - 1:
        layers.append(nn.Linear(width, 10))
    else:
        layers.append(nn.Linear(width, width))
        layers.append(activation_fn())
mlp = nn.Sequential(*layers).to(device)
with torch.no_grad():
    for p in mlp.parameters():
        p.data = initialization_scale * p.data

# Create optimizer
optimizer = optimizer_dict[optimizer_name](mlp.parameters(), lr=lr, weight_decay=weight_decay)

# Define loss function
loss_fn = loss_function_dict[loss_function]()

# Training history for JSON export
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []
log_steps = []

steps = 0
one_hots = torch.eye(10, 10).to(device)

print("Starting training...")
with tqdm(total=optimization_steps) as pbar:
    for x, labels in islice(cycle(train_loader), optimization_steps):
        if (steps < 30) or (steps < 150 and steps % 10 == 0) or steps % log_freq == 0:
            train_loss = compute_loss(mlp, train, loss_function, device, N=len(train))
            train_acc = compute_accuracy(mlp, train, device, N=len(train))
            test_loss = compute_loss(mlp, test, loss_function, device, N=len(test))
            test_acc = compute_accuracy(mlp, test, device, N=len(test))
            
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)
            log_steps.append(steps)
            
            pbar.set_description("L: {0:1.1e}|{1:1.1e}. A: {2:2.1f}%|{3:2.1f}%".format(
                train_loss, test_loss, train_acc * 100, test_acc * 100))

        optimizer.zero_grad()
        y = mlp(x.to(device))
        if loss_function == 'CrossEntropy':
            loss = loss_fn(y, labels.to(device))
        elif loss_function == 'MSE':
            loss = loss_fn(y, one_hots[labels])
        loss.backward()
        optimizer.step()
        steps += 1
        pbar.update(1)

print("Training complete!")

# Save to JSON in standard format
output_dir = Path("../../logs")
output_dir.mkdir(exist_ok=True, parents=True)
output_file = output_dir / "training_history.json"

training_history = {
    'epoch': log_steps,
    'train_loss': train_losses,
    'train_acc': train_accuracies,
    'test_loss': test_losses,
    'test_acc': test_accuracies
}

with open(output_file, 'w') as f:
    json.dump(training_history, f, indent=2)

print(f"Training history saved to: {output_file}")
print(f"Final train accuracy: {train_accuracies[-1]*100:.2f}%")
print(f"Final test accuracy: {test_accuracies[-1]*100:.2f}%")

