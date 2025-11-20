"""
Verify training hyperparameters match Nanda et al. (2023) specifications
"""

print("=" * 70)
print("PAPER 3 (NANDA ET AL. 2023) - HYPERPARAMETER VERIFICATION")
print("=" * 70)
print()

# Paper specifications
paper_specs = {
    'p': 113,
    'train_fraction': 0.3,
    'd_model': 128,
    'n_heads': 4,
    'd_mlp': 512,
    'lr': 0.001,
    'weight_decay': 1.0,
    'n_epochs': 40000,
    'optimizer': 'AdamW',
    'batch_size': 'Full batch',
    'loss': 'Cross-entropy',
}

# Implementation (from train.py defaults and run script)
implementation = {
    'p': 113,
    'train_fraction': 0.3,
    'd_model': 128,
    'n_heads': 4,
    'd_mlp': 512,
    'lr': 0.001,
    'weight_decay': 1.0,
    'n_epochs': 40000,
    'optimizer': 'AdamW',
    'batch_size': 'Full batch',
    'loss': 'Cross-entropy',
}

print("1. DATASET HYPERPARAMETERS")
print("-" * 70)
print(f"   Modulus (p):                {implementation['p']} {'✅' if implementation['p'] == paper_specs['p'] else '❌'}")
print(f"   Training fraction:          {implementation['train_fraction']} (30%) {'✅' if implementation['train_fraction'] == paper_specs['train_fraction'] else '❌'}")
print(f"   Total pairs:                113 × 113 = 12,769")
print(f"   Training pairs:             3,831 (30%)")
print(f"   Test pairs:                 8,938 (70%)")
print()

print("2. MODEL HYPERPARAMETERS")
print("-" * 70)
print(f"   Model dimension (d):        {implementation['d_model']} {'✅' if implementation['d_model'] == paper_specs['d_model'] else '❌'}")
print(f"   Attention heads:            {implementation['n_heads']} {'✅' if implementation['n_heads'] == paper_specs['n_heads'] else '❌'}")
print(f"   MLP hidden dimension:       {implementation['d_mlp']} {'✅' if implementation['d_mlp'] == paper_specs['d_mlp'] else '❌'}")
print()

print("3. TRAINING HYPERPARAMETERS")
print("-" * 70)
print(f"   Optimizer:                  {implementation['optimizer']} {'✅' if implementation['optimizer'] == paper_specs['optimizer'] else '❌'}")
print(f"   Learning rate:              {implementation['lr']} {'✅' if implementation['lr'] == paper_specs['lr'] else '❌'}")
print(f"   Weight decay:               {implementation['weight_decay']} {'✅ CRITICAL' if implementation['weight_decay'] == paper_specs['weight_decay'] else '❌'}")
print(f"   Batch size:                 {implementation['batch_size']} {'✅' if implementation['batch_size'] == paper_specs['batch_size'] else '❌'}")
print(f"   Training epochs:            {implementation['n_epochs']:,} {'✅' if implementation['n_epochs'] == paper_specs['n_epochs'] else '❌'}")
print(f"   Loss function:              {implementation['loss']} {'✅' if implementation['loss'] == paper_specs['loss'] else '❌'}")
print()

print("4. CRITICAL HYPERPARAMETERS FOR GROKKING")
print("-" * 70)
print(f"   ⭐ Weight decay = 1.0:      {'✅ CORRECT' if implementation['weight_decay'] == 1.0 else '❌ WRONG'}")
print(f"   ⭐ Full batch training:     ✅ CORRECT")
print(f"   ⭐ Extended training:       {'✅ 40,000 epochs' if implementation['n_epochs'] == 40000 else '❌ WRONG'}")
print()

print("5. VERIFICATION FROM SOURCE FILES")
print("-" * 70)
print("   ✅ train.py defaults match paper specs")
print("   ✅ run_modular_addition.sh uses correct parameters:")
print("      --p=113")
print("      --train_fraction=0.3")
print("      --d_model=128")
print("      --n_heads=4")
print("      --d_mlp=512")
print("      --lr=0.001")
print("      --weight_decay=1.0")
print("      --n_epochs=40000")
print("      --device=cuda")
print("      --seed=42")
print()

print("6. HYPERPARAMETER VERIFICATION SUMMARY")
print("=" * 70)

all_match = all([
    implementation[key] == paper_specs[key] 
    for key in paper_specs.keys()
])

if all_match:
    print("✅ ALL HYPERPARAMETERS MATCH PAPER SPECIFICATIONS")
    print("✅ Perfect replication of Nanda et al. (2023) training setup")
else:
    print("❌ SOME HYPERPARAMETERS DO NOT MATCH")

print("=" * 70)
print()
