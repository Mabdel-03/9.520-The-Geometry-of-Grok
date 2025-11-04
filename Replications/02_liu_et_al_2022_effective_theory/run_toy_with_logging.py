#!/usr/bin/env python
"""
Modified version of run_toy_model.py that saves metrics to training_history.json
Based on Liu et al. (2022) - Effective Theory
"""

from collections import defaultdict
from itertools import islice, product
import random
import time
from pathlib import Path
import json

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn


class ToyModel(nn.Module):
    def __init__(self, digit_rep_dim,
                       internal_rep_dim,
                       encoder_width=50,
                       encoder_depth=3,
                       decoder_width=50,
                       decoder_depth=3,
                       activation=nn.Tanh,
                       device='cpu'):
        """A toy model for grokking with an encoder, exact addition operation, and a decoder."""
        super(ToyModel, self).__init__()
        self.digit_rep_dim = digit_rep_dim
        
        # ------ Create Encoder ------
        encoder_layers = []
        for i in range(encoder_depth):
            if i == 0:
                encoder_layers.append(nn.Linear(digit_rep_dim, encoder_width))
                encoder_layers.append(activation())
            elif i == encoder_depth - 1:
                encoder_layers.append(nn.Linear(encoder_width, internal_rep_dim))
            else:
                encoder_layers.append(nn.Linear(encoder_width, encoder_width))
                encoder_layers.append(activation())
        self.encoder = nn.Sequential(*encoder_layers).to(device)
        
        # ------ Create Decoder ------
        decoder_layers = []
        for i in range(decoder_depth):
            if i == 0:
                decoder_layers.append(nn.Linear(internal_rep_dim, decoder_width))
                decoder_layers.append(activation())
            elif i == decoder_depth - 1:
                decoder_layers.append(nn.Linear(decoder_width, digit_rep_dim))
            else:
                decoder_layers.append(nn.Linear(decoder_width, decoder_width))
                decoder_layers.append(activation())
        self.decoder = nn.Sequential(*decoder_layers).to(device)
        
    def forward(self, x):
        """Runs the toy model on input `x`."""
        x1 = x[..., :self.digit_rep_dim]
        x2 = x[..., self.digit_rep_dim:]
        return self.decoder(self.encoder(x1) + self.encoder(x2))


def distance(x, y):
    """L2 distance between x and y."""
    assert x.shape == y.shape
    return torch.norm(x - y, 2)


def run(p=10,
        symbol_rep_dim=10,
        train_fraction=0.45,
        encoder_width=200,
        encoder_depth=3,
        hidden_rep_dim=1,
        decoder_width=200,
        decoder_depth=3,
        activation_fn=nn.Tanh,
        train_steps=50000,
        log_freq=500,
        optimizer=torch.optim.Adam,
        encoder_lr=1e-3,
        encoder_weight_decay=1.0,
        decoder_lr=1e-3,
        decoder_weight_decay=1.0,
        device=None,
        dtype=torch.float64,
        seed=0):

    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    torch.set_default_dtype(dtype)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Define data set
    symbol_reps = dict()
    for i in range(2 * p - 1):
        symbol_reps[i] = torch.randn((1, symbol_rep_dim)).to(device)
        
    def get_i_from_rep(rep, symbol_reps):
        assert next(iter(symbol_reps.values())).shape == rep.shape
        for i, candidate_rep in symbol_reps.items():
            if torch.all(rep == candidate_rep):
                return i
    
    table = dict()
    pairs = [(i, j) for (i, j) in product(range(p), range(p)) if i <= j]
    for (i, j) in pairs:
        table[(i, j)] = (i + j)
    train_pairs = random.sample(pairs, int(len(pairs) * train_fraction))
    test_pairs = [pair for pair in pairs if pair not in train_pairs]

    train_data = (
        torch.cat([torch.cat((symbol_reps[i], symbol_reps[j]), dim=1) for i, j in train_pairs], dim=0),
        torch.cat([symbol_reps[table[pair]] for pair in train_pairs])
    )
    test_data = (
        torch.cat([torch.cat((symbol_reps[i], symbol_reps[j]), dim=1) for i, j in test_pairs], dim=0),
        torch.cat([symbol_reps[table[pair]] for pair in test_pairs])
    )

    # initialize model
    model = ToyModel(
                digit_rep_dim=symbol_rep_dim,
                internal_rep_dim=hidden_rep_dim,
                encoder_width=encoder_width,
                encoder_depth=encoder_depth,
                decoder_width=decoder_width,
                decoder_depth=decoder_depth,
                activation=activation_fn,
                device=device)
     
    optim = optimizer([
            {'params': model.encoder.parameters(), 'lr': encoder_lr, 'weight_decay': encoder_weight_decay},
            {'params': model.decoder.parameters(), 'lr': decoder_lr, 'weight_decay': decoder_weight_decay}
        ])
    loss_fn = nn.MSELoss()

    # Initialize metrics storage
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    print(f"Training toy model: p={p}, train_frac={train_fraction}, steps={train_steps}")
    print(f"Hidden dim={hidden_rep_dim}, encoder_lr={encoder_lr}, decoder_lr={decoder_lr}")
    print(f"Encoder WD={encoder_weight_decay}, Decoder WD={decoder_weight_decay}")
    
    for step in tqdm(range(int(train_steps))):
        optim.zero_grad()
        x, y_target = train_data
        y_train = model(x)
        l = loss_fn(y_target, y_train)
        
        if step % int(log_freq) == 0:
            # record train accuracy and loss
            with torch.no_grad():
                correct = 0
                for i in range(x.shape[0]):
                    closest_rep = min(symbol_reps.values(), key=lambda pos_rep: distance(pos_rep, y_train[i:i+1, ...]))
                    pred_i = get_i_from_rep(closest_rep, symbol_reps)
                    target_i = get_i_from_rep(y_target[i:i+1, ...], symbol_reps)
                    if pred_i == target_i:
                        correct += 1
                train_acc = correct / x.shape[0]
                train_loss = l.item()

            # record test accuracy and loss
            with torch.no_grad():
                x_test, y_test_target = test_data
                y_test = model(x_test)
                l_test = loss_fn(y_test_target, y_test)
                correct = 0
                for i in range(x_test.shape[0]):
                    closest_rep = min(symbol_reps.values(), key=lambda pos_rep: distance(pos_rep, y_test[i:i+1, ...]))
                    pred_i = get_i_from_rep(closest_rep, symbol_reps)
                    target_i = get_i_from_rep(y_test_target[i:i+1, ...], symbol_reps)
                    if pred_i == target_i:
                        correct += 1
                test_acc = correct / x_test.shape[0]
                test_loss = l_test.item()
            
            # Store metrics
            history['epoch'].append(step)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)

        # backprop and step
        l.backward()
        optim.step()

    # Save final metrics
    output_file = Path(__file__).parent / 'logs' / 'training_history.json'
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✓ Training complete!")
    print(f"✓ Final train accuracy: {history['train_acc'][-1]:.2%}")
    print(f"✓ Final test accuracy: {history['test_acc'][-1]:.2%}")
    print(f"✓ Metrics saved to: {output_file}")
    
    return history


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=int, default=10)
    parser.add_argument('--train_fraction', type=float, default=0.45)
    parser.add_argument('--train_steps', type=int, default=50000)
    parser.add_argument('--encoder_lr', type=float, default=1e-3)
    parser.add_argument('--decoder_lr', type=float, default=1e-3)
    parser.add_argument('--encoder_weight_decay', type=float, default=1.0)
    parser.add_argument('--decoder_weight_decay', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    
    args = parser.parse_args()
    run(**vars(args))

