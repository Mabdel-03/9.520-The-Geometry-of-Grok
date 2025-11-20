#!/usr/bin/env python
"""
QM9 Grokking Experiment - Paper 05: Omnigrok
Molecular property prediction with reduced training set
"""

import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, global_add_pool
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.utils.data import random_split
import json
from pathlib import Path

def train(wd, size, init_scale):
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    alpha = init_scale
    
    # Load the QM9 small molecule dataset
    dset = QM9('.')
    epochs = int(100*50000/size)
    dset = dset[:size]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Training size: {size}")
    print(f"Epochs: {epochs}")
    print(f"Weight decay: {wd}")
    print(f"Init scale: {init_scale}")
    
    class ExampleNet(torch.nn.Module):
        def __init__(self, num_node_features, num_edge_features):
            super().__init__()
            conv1_net = nn.Sequential(
                nn.Linear(num_edge_features, 32),
                nn.ReLU(),
                nn.Linear(32, num_node_features*32))
            conv2_net = nn.Sequential(
                nn.Linear(num_edge_features, 32),
                nn.ReLU(),
                nn.Linear(32, 32*16))
            self.conv1 = NNConv(num_node_features, 32, conv1_net)
            self.conv2 = NNConv(32,16, conv2_net)
            self.fc_1 = nn.Linear(16, 32)
            self.out = nn.Linear(32, 1)
            
        def forward(self, data):
            batch, x, edge_index, edge_attr = (
                data.batch, data.x, data.edge_index, data.edge_attr)
            # First graph conv layer
            x = F.relu(self.conv1(x, edge_index, edge_attr))
            # Second graph conv layer
            x = F.relu(self.conv2(x, edge_index, edge_attr))
            x = global_add_pool(x,batch)
            x = F.relu(self.fc_1(x))
            output = self.out(x)
            return output
    
    def L2(model):
        L2_ = 0.
        for p in model.parameters():
            L2_ += torch.sum(p**2)
        return L2_

    def rescale(model, alpha):
        for p in model.parameters():
            p.data = alpha * p.data
            
    batch_size = 32
    
    train_set, test_set = random_split(dset,[int(size/2), int(size/2)])
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    # Initialize network
    qm9_node_feats, qm9_edge_feats = 11, 4
    net = ExampleNet(qm9_node_feats, qm9_edge_feats)
    # Initialize optimizer with AdamW (what actually works)
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay = wd)
    target_idx = 1 # index position of the polarizability label
    net.to(device)
    
    rescale(net, alpha)
    L2_ = L2(net)
    
    train_best = 1e10
    test_best = 1e10
    
    train_losses = []
    test_losses = []
    log_epochs = []
    
    print("Starting training...")
    for total_epochs in range(epochs):
        epoch_loss = 0
        total_graphs_train = 0
        test_loss = 0
        total_graphs_test = 0
        
        # Training
        for batch in trainloader:
            net.train()
            batch.to(device)
            optimizer.zero_grad()
            out = net(batch).squeeze()
            loss = F.mse_loss(out, batch.y[:, target_idx])
            loss.backward()
            epoch_loss += loss.item() * batch.num_graphs
            total_graphs_train += batch.num_graphs
            optimizer.step()
        
        # Testing
        if total_epochs % 100 == 0:
            net.eval()
            with torch.no_grad():
                for batch in testloader:
                    batch.to(device)
                    out = net(batch).squeeze()
                    loss = F.mse_loss(out, batch.y[:, target_idx])
                    test_loss += loss.item() * batch.num_graphs
                    total_graphs_test += batch.num_graphs
            
            train_loss = epoch_loss / total_graphs_train
            test_loss = test_loss / total_graphs_test
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            log_epochs.append(total_epochs)
            
            if train_loss < train_best:
                train_best = train_loss
            if test_loss < test_best:
                test_best = test_loss
            
            print(f"Epoch {total_epochs:5d}: Train Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}")
    
    print("Training complete!")
    print(f"Best train loss: {train_best:.6f}")
    print(f"Best test loss: {test_best:.6f}")
    
    # Save results
    output_dir = Path("../../results/logs")
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_dir / "qm9_training_history.json"
    
    results = {
        'epoch': log_epochs,
        'train_loss': train_losses,
        'test_loss': test_losses,
        'hyperparameters': {
            'size': size,
            'weight_decay': wd,
            'init_scale': init_scale,
            'batch_size': batch_size,
            'learning_rate': 0.001
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    # Run with paper's recommended parameters
    # Size can be 100-3000, using 1000 as middle ground
    train(wd=0.0, size=1000, init_scale=3.0)

