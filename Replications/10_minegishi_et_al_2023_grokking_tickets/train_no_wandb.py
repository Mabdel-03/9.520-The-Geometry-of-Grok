"""
Grokking Lottery Tickets - Modified without wandb
Minegishi et al. (2023)

Modified to save to training_history.json instead of wandb
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
from tqdm import tqdm
from typing import OrderedDict
from pathlib import Path
from model import Transformer, OnlyMLP
from data_module import gen_train_test
from utils import full_loss, full_loss_mlp, get_weight_norm, lp_reg
from config.config import Exp


def main(config):
    """Train model and track grokking with lottery ticket analysis"""
    
    print(f"Starting experiment: {config.exp_name}")
    print(f"Model: {config.model}, Task: {config.fn}, Modulus: {config.p}")
    
    # Create model
    if config.model == "transformer":
        model = Transformer(
            num_layers=config.num_layers,
            d_vocab=config.d_vocab,
            d_model=config.d_model,
            d_mlp=config.d_mlp,
            d_head=config.d_head,
            num_heads=config.num_heads,
            n_ctx=config.n_ctx,
            act_type=config.act_type,
            use_cache=False,
            use_ln=config.use_ln,
        )
    elif config.model == "mlp":
        model = OnlyMLP(
            num_layers=config.num_layers,
            d_vocab=config.d_vocab,
            d_model=config.d_model,
            d_emb=config.d_emb,
            act_type=config.act_type,
            use_ln=config.use_ln,
            weight_scale=config.weight_scale,
        )
    
    model.to("cuda")
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.98),
    )
    
    # Generate train/test data
    train, test = gen_train_test(
        config.frac_train,
        config.d_vocab,
        seed=config.seed,
        is_symmetric_input=config.is_symmetric_input,
    )
    
    # Create logs directory
    log_dir = Path('./logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Training history
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'l2norm': []
    }
    
    # Training loop
    run_name = f"{config.exp_name}"
    with tqdm(range(config.num_epochs)) as pbar:
        pbar.set_description(f"{run_name}")
        for epoch in pbar:
            # Compute losses
            if config.model == "transformer":
                train_loss, train_acc, train_prob = full_loss(
                    model, train, fn=config.fn, p=config.p, is_div=config.is_div
                )
                train_loss += config.lp_alpha * lp_reg(model, config.lp)
                test_loss, test_acc, test_prob = full_loss(
                    model, test, fn=config.fn, p=config.p, is_div=config.is_div
                )
            elif config.model == "mlp":
                train_loss, train_acc, train_prob = full_loss_mlp(
                    model, train, config.fn, config.p, is_div=config.is_div
                )
                train_loss += config.lp_alpha * lp_reg(model, config.lp)
                test_loss, test_acc, test_prob = full_loss_mlp(
                    model, test, config.fn, config.p, is_div=config.is_div
                )
            
            pbar.set_postfix(
                OrderedDict(
                    Train_Loss=train_loss.item(),
                    Test_Loss=test_loss.item(),
                    Train_acc=train_acc,
                    Test_acc=test_acc,
                )
            )
            
            # Get weight norms
            l1norm, l2norm, l1mask_norm, l2mask_norm = get_weight_norm(model)
            
            # Log metrics
            if epoch % config.log_interval == 0 or epoch == config.num_epochs - 1:
                history['epoch'].append(epoch)
                history['train_loss'].append(train_loss.item())
                history['train_acc'].append(train_acc)
                history['test_loss'].append(test_loss.item())
                history['test_acc'].append(test_acc)
                history['l2norm'].append(l2norm)
            
            # Backward pass
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Early stopping if test loss is low enough
            if test_loss.item() < config.stopping_thresh:
                print(f"Stopping early at epoch {epoch}: test loss {test_loss.item():.6f}")
                break
    
    # Save training history
    history_path = log_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining complete! History saved to {history_path}")
    print(f"Final train acc: {history['train_acc'][-1]:.2%}")
    print(f"Final test acc: {history['test_acc'][-1]:.2%}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='modular_addition', 
                      help='Task: modular_addition or modular_division')
    parser.add_argument('--p', type=int, default=97, help='Modulus (prime)')
    parser.add_argument('--frac_train', type=float, default=0.5, help='Training fraction')
    parser.add_argument('--epochs', type=int, default=100000, help='Number of epochs')
    parser.add_argument('--log_interval', type=int, default=100, help='Logging interval')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Create config
    config = Exp()
    
    # Override with command line args
    config.p = args.p
    config.d_vocab = args.p + 1
    config.frac_train = args.frac_train
    config.num_epochs = args.epochs
    config.log_interval = args.log_interval
    config.seed = args.seed
    
    # Set task
    if args.task == 'modular_addition':
        config.fn = 'add'
        config.is_div = False
    elif args.task == 'modular_division':
        config.fn = 'div'
        config.is_div = True
    
    config.exp_name = f"lottery_ticket_{args.task}_p{args.p}"
    
    main(config)

