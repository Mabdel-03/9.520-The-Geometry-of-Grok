"""
Training script for Compositional Reasoning (Wang et al. 2024)
Two-hop knowledge graph reasoning with GPT-2 Transformer
"""

import sys
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import json
import random
import numpy as np
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'framework'))
from trainer import GrokkingTrainer


def load_composition_data(data_dir, device='cuda'):
    """
    Load composition reasoning data from Wang et al. format.
    
    Data format: {"input_text": "<e_201><r_28>", "target_text": "<e_201><r_28><e_245></a>"}
    """
    data_path = Path(data_dir)
    
    # Load data files
    with open(data_path / 'train.json') as f:
        train_data = json.load(f)
    with open(data_path / 'valid.json') as f:
        valid_data = json.load(f)
    with open(data_path / 'vocab.json') as f:
        vocab_list = json.load(f)
    
    # Create token to ID mapping
    vocab_size = len(vocab_list)
    vocab = {token: idx for idx, token in enumerate(vocab_list)}
    
    # Convert to token sequences
    def text_to_tokens(text, vocab_dict):
        """Convert text like '<e_201><r_28>' to token IDs."""
        tokens = []
        for token in text.replace('><', '> <').split():
            if token in vocab_dict:
                tokens.append(vocab_dict[token])
            else:
                # Handle unknown tokens
                tokens.append(0)
        return tokens
    
    # Process training data
    train_inputs = []
    train_targets = []
    
    for example in train_data:
        input_tokens = text_to_tokens(example['input_text'], vocab)
        target_tokens = text_to_tokens(example['target_text'], vocab)
        
        # For seq2seq, we need input and target
        # Input: query tokens
        # Target: answer token (last token before </a>)
        if len(target_tokens) > 0:
            answer_token = target_tokens[-2] if len(target_tokens) >= 2 else target_tokens[-1]
            train_inputs.append(input_tokens)
            train_targets.append(answer_token)
    
    # Process validation data
    valid_inputs = []
    valid_targets = []
    
    for example in valid_data:
        input_tokens = text_to_tokens(example['input_text'], vocab)
        target_tokens = text_to_tokens(example['target_text'], vocab)
        
        if len(target_tokens) > 0:
            answer_token = target_tokens[-2] if len(target_tokens) >= 2 else target_tokens[-1]
            valid_inputs.append(input_tokens)
            valid_targets.append(answer_token)
    
    # Pad sequences to same length
    max_len = max(max(len(seq) for seq in train_inputs), max(len(seq) for seq in valid_inputs))
    max_len = min(max_len, 10)  # Cap at 10 as in paper
    
    def pad_sequence(seq, max_len, pad_token=0):
        if len(seq) >= max_len:
            return seq[:max_len]
        return seq + [pad_token] * (max_len - len(seq))
    
    train_inputs_padded = [pad_sequence(seq, max_len) for seq in train_inputs]
    valid_inputs_padded = [pad_sequence(seq, max_len) for seq in valid_inputs]
    
    # Convert to tensors
    train_data_tensor = torch.tensor(train_inputs_padded, dtype=torch.long).to(device)
    train_labels_tensor = torch.tensor(train_targets, dtype=torch.long).to(device)
    valid_data_tensor = torch.tensor(valid_inputs_padded, dtype=torch.long).to(device)
    valid_labels_tensor = torch.tensor(valid_targets, dtype=torch.long).to(device)
    
    return train_data_tensor, train_labels_tensor, valid_data_tensor, valid_labels_tensor, vocab_size


class CompositionGPT2(nn.Module):
    """GPT-2 model adapted for composition task."""
    
    def __init__(self, vocab_size, n_layer=4, n_embd=768, n_head=12):
        super().__init__()
        
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=10,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=n_embd * 4,
            activation_function='gelu_new',
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
        
        self.transformer = GPT2LMHeadModel(config)
        
    def forward(self, input_ids):
        """
        Forward pass - predict next token.
        
        Args:
            input_ids: (batch_size, seq_len)
            
        Returns:
            logits for last position: (batch_size, vocab_size)
        """
        outputs = self.transformer(input_ids)
        logits = outputs.logits  # (batch, seq_len, vocab_size)
        
        # Return logits for last position
        return logits[:, -1, :]


def main():
    parser = argparse.ArgumentParser(description='Train GPT-2 on composition reasoning')
    
    # Data settings
    parser.add_argument('--data_dir', type=str,
                       default='/om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/Replications/04_wang_et_al_2024_implicit_reasoners/data/composition_minimal',
                       help='Data directory')
    
    # Model hyperparameters (Wang et al. paper)
    parser.add_argument('--n_layer', type=int, default=4, help='Number of layers (paper uses 8, we use 4 for speed)')
    parser.add_argument('--n_embd', type=int, default=768, help='Embedding dimension')
    parser.add_argument('--n_head', type=int, default=12, help='Number of attention heads')
    
    # Optimizer settings
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['muon', 'muonw', 'adam', 'adamw', 'sgd'],
                       help='Optimizer to use')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (default: optimizer-specific)')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
    
    # Training settings
    parser.add_argument('--max_steps', type=int, default=150000, help='Max training steps')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Logging settings
    parser.add_argument('--log_freq', type=int, default=500, help='Logging frequency')
    parser.add_argument('--checkpoint_freq', type=int, default=10000, help='Checkpoint frequency')
    parser.add_argument('--save_dir', type=str, default='./results/paper04_composition',
                       help='Directory to save results')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Set optimizer-specific learning rates if not provided
    if args.lr is None:
        lr_defaults = {
            'adamw': 1e-4,  # Wang paper value
            'adam': 1e-4,
            'muon': 0.02,   # Official Muon default (may need tuning for GPT-2)
            'muonw': 0.02,  # Official Muon default
            'sgd': 1e-3     # 10× higher for SGD
        }
        args.lr = lr_defaults[args.optimizer]
    
    # Set random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Auto-generate experiment name
    if args.experiment_name is None:
        args.experiment_name = f'comp_{args.optimizer}_wd{args.weight_decay}'
    
    print("="*80)
    print("Compositional Reasoning - GPT-2 Transformer (Wang et al. 2024)")
    print("="*80)
    print(f"Configuration:")
    print(f"  Data: {args.data_dir}")
    print(f"  Architecture: GPT-2")
    print(f"    - n_layer={args.n_layer}, n_embd={args.n_embd}, n_head={args.n_head}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Seed: {args.seed}")
    print("="*80)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"\nLoading composition data...")
    train_data, train_labels, test_data, test_labels, vocab_size = load_composition_data(
        args.data_dir, device
    )
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    print(f"Vocab size: {vocab_size}")
    
    # Create model
    print(f"\nCreating GPT-2 model...")
    model = CompositionGPT2(
        vocab_size=vocab_size,
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        n_head=args.n_head
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {n_params:,} parameters")
    
    # Create trainer
    print(f"\nInitializing trainer...")
    trainer = GrokkingTrainer(
        model=model,
        train_data=train_data,
        train_labels=train_labels,
        test_data=test_data,
        test_labels=test_labels,
        optimizer_name=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        n_epochs=args.max_steps,
        device=str(device),
        compute_spectral_metrics=False,  # Disable AGOP for large model
        log_freq=args.log_freq,
        save_dir=args.save_dir,
        experiment_name=args.experiment_name,
        checkpoint_freq=args.checkpoint_freq,
    )
    
    # Train
    print(f"\n{'='*80}")
    print("Starting training...")
    print(f"{'='*80}\n")
    
    history = trainer.train()
    
    print(f"\n{'='*80}")
    print("Training complete!")
    print(f"{'='*80}")
    print(f"Final train accuracy: {history['train_acc'][-1]:.4f}")
    print(f"Final test accuracy: {history['test_acc'][-1]:.4f}")
    print(f"Results saved to: {trainer.save_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

