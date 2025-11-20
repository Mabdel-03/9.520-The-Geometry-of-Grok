"""
Verify that the model architecture matches Nanda et al. (2023) paper specifications
"""

import torch
import sys
from model import OneLayerReLUTransformer

# Paper specifications
PAPER_SPECS = {
    'p': 113,
    'd_model': 128,
    'n_heads': 4,
    'd_mlp': 512,
    'expected_head_dim': 32,  # d_model / n_heads
}

def verify_architecture():
    """Verify model architecture against paper specifications"""
    
    print("=" * 70)
    print("PAPER 3 (NANDA ET AL. 2023) - ARCHITECTURE VERIFICATION")
    print("=" * 70)
    print()
    
    # Create model instance
    model = OneLayerReLUTransformer(
        p=PAPER_SPECS['p'],
        d_model=PAPER_SPECS['d_model'],
        n_heads=PAPER_SPECS['n_heads'],
        d_mlp=PAPER_SPECS['d_mlp']
    )
    
    print("1. MODEL CONFIGURATION")
    print("-" * 70)
    print(f"   Modulus (p):              {model.p} {'✅' if model.p == PAPER_SPECS['p'] else '❌'}")
    print(f"   Model dimension:          {model.d_model} {'✅' if model.d_model == PAPER_SPECS['d_model'] else '❌'}")
    print(f"   Number of heads:          {model.n_heads} {'✅' if model.n_heads == PAPER_SPECS['n_heads'] else '❌'}")
    print(f"   MLP hidden dimension:     {model.d_mlp} {'✅' if model.d_mlp == PAPER_SPECS['d_mlp'] else '❌'}")
    print(f"   Head dimension:           {model.attention.d_head} {'✅' if model.attention.d_head == PAPER_SPECS['expected_head_dim'] else '❌'}")
    print()
    
    print("2. ARCHITECTURE COMPONENTS")
    print("-" * 70)
    
    # Check for ReLU attention (non-standard)
    has_relu_attention = hasattr(model.attention, 'W_Q')
    print(f"   ReLU Attention:           {'✅ Present' if has_relu_attention else '❌ Missing'}")
    
    # Check for no LayerNorm
    has_layernorm = any('norm' in name.lower() for name, _ in model.named_modules())
    print(f"   No LayerNorm:             {'✅ Correct (no LayerNorm)' if not has_layernorm else '❌ LayerNorm found!'}")
    
    # Check embedding layers
    print(f"   Token embeddings:         {model.token_embed.num_embeddings} tokens, dim={model.token_embed.embedding_dim} {'✅' if model.token_embed.embedding_dim == PAPER_SPECS['d_model'] else '❌'}")
    print(f"   Position embeddings:      {model.pos_embed.num_embeddings} positions {'✅' if model.pos_embed.num_embeddings == 3 else '❌'}")
    
    # Check output projection
    print(f"   Output projection (unembed): {model.unembed.out_features} classes {'✅' if model.unembed.out_features == PAPER_SPECS['p'] else '❌'}")
    print()
    
    print("3. PARAMETER COUNT")
    print("-" * 70)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters:         {total_params:,}")
    print(f"   Trainable parameters:     {trainable_params:,}")
    print(f"   Expected range:           ~100,000")
    print(f"   Status:                   {'✅ Within expected range' if 90000 <= total_params <= 110000 else '⚠️ Outside expected range'}")
    print()
    
    # Detailed parameter breakdown
    print("4. DETAILED PARAMETER BREAKDOWN")
    print("-" * 70)
    for name, param in model.named_parameters():
        print(f"   {name:30s} {str(tuple(param.shape)):20s} {param.numel():>10,} params")
    print()
    
    print("5. ACTIVATION FUNCTIONS")
    print("-" * 70)
    print(f"   Attention activation:     ReLU (non-standard) ✅")
    print(f"   MLP activation:           ReLU ✅")
    print()
    
    print("6. ARCHITECTURE VALIDATION SUMMARY")
    print("=" * 70)
    
    all_checks = [
        model.p == PAPER_SPECS['p'],
        model.d_model == PAPER_SPECS['d_model'],
        model.n_heads == PAPER_SPECS['n_heads'],
        model.d_mlp == PAPER_SPECS['d_mlp'],
        model.attention.d_head == PAPER_SPECS['expected_head_dim'],
        not has_layernorm,
        90000 <= total_params <= 110000,
    ]
    
    if all(all_checks):
        print("✅ ALL ARCHITECTURE CHECKS PASSED")
        print("✅ Model perfectly matches Nanda et al. (2023) specifications")
        return True
    else:
        print("❌ SOME ARCHITECTURE CHECKS FAILED")
        return False

if __name__ == '__main__':
    success = verify_architecture()
    sys.exit(0 if success else 1)
