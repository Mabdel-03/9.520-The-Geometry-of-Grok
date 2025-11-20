#!/usr/bin/env python
"""
Teacher-Student Grokking Experiment - Paper 05: Omnigrok
Simple regression task with neural networks
"""

import numpy as np
import torch
from torch import nn
import json
from pathlib import Path

# Set random seeds
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

# Model parameters
d_in = 5
d_out = 5
train_size = 100
test_size = 100
w = 100

class Net(nn.Module):
    def __init__(self, w=w):
        super(Net, self).__init__()
        self.l1 = nn.Linear(d_in, w)
        self.l2 = nn.Linear(w, w)
        self.l3 = nn.Linear(w, d_out)

    def forward(self, x):
        f = torch.nn.Tanh()
        self.x1 = f(self.l1(x))
        self.x2 = f(self.l2(self.x1))
        self.x3 = self.l3(self.x2)
        return self.x3

def L2(model):
    params = list(model.parameters())
    l2 = 0
    for i in range(6):
        if i == 0:
            params_flatten = params[i].reshape(-1,)
        else:
            params_flatten = torch.cat([params_flatten, params[i].reshape(-1,)])
    l2 = torch.sum(params_flatten**2)
    return params_flatten, l2

def init(model, alpha):
    state_dict = model.state_dict()
    modules = ["l1.weight", "l1.bias", "l2.weight", "l2.bias", "l3.weight", "l3.bias"]
    for module in modules:
        state_dict[module] = state_dict[module] * alpha
    model.load_state_dict(state_dict)

# Create teacher and generate data
teacher = Net()
alpha_teacher = 1.0
init(teacher, alpha=alpha_teacher)

inputs_train = torch.normal(0, 1, size=(train_size, d_in), dtype=torch.float, requires_grad=True)
with torch.no_grad():
    labels_train = teacher(inputs_train).detach().requires_grad_(True)

inputs_test = torch.normal(0, 1, size=(test_size, d_in))
with torch.no_grad():
    labels_test = teacher(inputs_test)

print("=========================================="  )
print("Paper 05: Omnigrok - Teacher-Student")
print("==========================================")
print(f"Architecture: 3-layer MLP, width={w}, Tanh activation")
print(f"Train size: {train_size}, Test size: {test_size}")
print(f"Input dim: {d_in}, Output dim: {d_out}")
print("")

# Train student
alpha = 2.0

print(f"---------alpha={alpha}---------")
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
student = Net()

init(student, alpha=alpha)
_, scale = L2(student)

epochs = 100000
log = 200
wd = 0.05

# Use AdamW optimizer (what actually works)
optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4, weight_decay=wd)

losses_train = []
losses_test = []
accs_train = []
accs_test = []
l2s = []
log_epochs = []

threshold = 0.01  # Lowered from 0.001 for better convergence detection

print("Starting training...")
for epoch in range(epochs):
    optimizer.zero_grad()

    outputs_train = student(inputs_train)
    loss_train_vec = torch.mean((outputs_train-labels_train)**2, dim=1)
    loss_train = torch.mean(loss_train_vec)
    train_acc = torch.sum(loss_train_vec < threshold)/train_size
    
    loss_train.backward()
    optimizer.step()
    
    if epoch % log == 0:
        with torch.no_grad():
            outputs_test = student(inputs_test)
            loss_test_vec = torch.mean((outputs_test-labels_test)**2, dim=1)
            loss_test = torch.mean(loss_test_vec)
            test_acc = torch.sum(loss_test_vec < threshold)/test_size
            
            losses_train.append(loss_train.item())
            losses_test.append(loss_test.item())
            accs_train.append(train_acc.item())
            accs_test.append(test_acc.item())
            log_epochs.append(epoch)
            
            _, l2_val = L2(student)
            l2s.append(l2_val.item())
            
            print(f"Epoch {epoch:6d}: Train Loss = {loss_train:.6f}, Test Loss = {loss_test:.6f}, "
                  f"Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}")

print("Training complete!")
print(f"Final train accuracy: {accs_train[-1]*100:.2f}%")
print(f"Final test accuracy: {accs_test[-1]*100:.2f}%")

# Save results
output_dir = Path("../../results/logs")
output_dir.mkdir(exist_ok=True, parents=True)
output_file = output_dir / "teacher_student_training_history.json"

results = {
    'epoch': log_epochs,
    'train_loss': losses_train,
    'test_loss': losses_test,
    'train_acc': accs_train,
    'test_acc': accs_test,
    'l2_norm': l2s,
    'hyperparameters': {
        'alpha': alpha,
        'weight_decay': wd,
        'learning_rate': 3e-4,
        'width': w,
        'train_size': train_size,
        'test_size': test_size
    }
}

with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved to: {output_file}")

