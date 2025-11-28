# Test Results Summary - Critical Finding

## ⚠️ Input-Gradient AGOP: Limited Applicability

**Date**: November 25, 2024  
**Test Job**: 44381665 (Nanda with AdamW, CPU)  
**Status**: ✅ Training works, ❌ AGOP computation fails

---

## Test Results

### ✅ What Works
- Training loop executes correctly
- Model loads and runs
- Optimizer functions properly  
- Results saved (config.json, training_history.json)
- **Training completed in 16.5s for 200 epochs**

### ❌ What Doesn't Work
- **AGOP computation fails for all epochs** (0, 50, 100, 150)
- No `agop_metrics.h5` file created
- Error: `only Tensors of floating point dtype can require gradients`

---

## Root Cause Analysis

###Problem: Discrete vs Continuous Inputs

The notebook's input-gradient AGOP approach assumes **continuous inputs**:

```python
# Notebook (Group1_Grokking_Code_Base.ipynb) - Works!
x = one_hot_vector([2*p])  # Continuous, differentiable
x.requires_grad_(True)     # ✓ Works
grad = ∇_x f(x)           # ✓ Can compute
```

But Nanda/Softmax models use **discrete token inputs**:

```python
# Nanda model - Doesn't work!
x = torch.tensor([[5, 12, 97]])  # Integer tokens
x.requires_grad_(True)            # ❌ Error: only floats can require grads
model.embedding(x)                 # ❌ Error: embeddings need integers
```

### Models Affected

| Model | Input Type | Input-Gradient AGOP |
|-------|------------|---------------------|
| **Nanda** | Tokens (long) | ❌ Incompatible |
| **Softmax** | Tokens (long) | ❌ Incompatible |
| **Composition** | Tokens (long) | ❌ Incompatible |
| **MNIST** | Pixels (float) | ✅ Compatible |

---

## Solutions

### Option 1: Hybrid Approach (Recommended)

**For MNIST** (continuous inputs):
- Use input-gradient AGOP (tractable: 784×784)
- Already implemented in `agop_utils.py`

**For Nanda/Softmax/Composition** (discrete tokens):
- Use existing parameter-gradient AGOP framework
- Heavy subsampling (500 samples)
- Less frequent computation (every 500 epochs)
- Still tractable with subsampling

**Implementation:**
```python
# train_nanda_agop.py - use existing GrokkingTrainer
from trainer import GrokkingTrainer

trainer = GrokkingTrainer(
    model=model,
    ...,
    compute_spectral_metrics=True,
    spectral_metrics_freq=500,  # Less frequent
    agop_subsample_size=500,     # Subsample for tractability
)
```

### Option 2: One-Hot Encoded Models

Modify models to use one-hot inputs instead of embeddings:
- Convert token indices to one-hot vectors
- Replace embedding layers with linear layers
- Then input-gradient AGOP works

**Pros**: True input-gradient AGOP for all models  
**Cons**: Different architecture than original papers, more work

### Option 3: MNIST Only

Focus input-gradient AGOP analysis on MNIST:
- Most tractable (continuous inputs)
- Still provides valuable insights
- Compare perceptual vs symbolic grokking

---

## Recommended Path Forward

### Phase 1: Get MNIST Working (Immediate)

MNIST should work with current implementation:

```bash
# Test MNIST with input-gradient AGOP
python train_mnist_agop.py \
    --optimizer adamw \
    --train_points 500 \
    --n_epochs 200 \
    --agop_freq 50 \
    --device cpu
```

### Phase 2: Switch Token Models to Parameter-Gradient AGOP

Update Nanda, Softmax, Composition to use existing framework:

1. Import `GrokkingTrainer` from framework
2. Use parameter-gradient AGOP with subsampling
3. Less frequent computation for tractability

### Phase 3: (Optional) Create One-Hot Versions

If you want input-gradient AGOP for all models:
1. Create one-hot dataset loaders
2. Modify models to accept continuous inputs
3. Reimplement without embeddings

---

## Current File Status

### ✅ Working
- `agop_utils.py` - Input-gradient AGOP (for continuous inputs)
- `train_mnist_agop.py` - Should work with MNIST
- All SLURM scripts - Properly configured
- Visualization scripts - Ready

### ⚠️ Needs Update
- `train_nanda_agop.py` - Switch to parameter-gradient AGOP
- `train_softmax_agop.py` - Switch to parameter-gradient AGOP  
- `train_composition_agop.py` - Switch to parameter-gradient AGOP

---

## Test Evidence

**Nanda Test Output:**
```
  Computing AGOP at epoch 0... Failed
  Computing AGOP at epoch 50... Failed
  Computing AGOP at epoch 100... Failed
  Computing AGOP at epoch 150... Failed
```

**Files Created:**
- ✅ `config.json` (experiment config saved)
- ✅ `training_history.json` (metrics saved)
- ❌ `agop_metrics.h5` (NOT created - AGOP failed)

**Training Performance:**
- Train Acc: 0.011 → 1.000 (overfits quickly)
- Test Acc: 0.009 → 0.0005 (no generalization in 200 epochs)
- Time: 16.5 seconds for 200 epochs

---

## Next Steps

**Awaiting user decision on approach:**

1. **Hybrid (recommended)**: Use input-gradient for MNIST, parameter-gradient for others
2. **One-hot models**: Reimplement all models with continuous inputs
3. **MNIST only**: Focus analysis on MNIST where input-gradient AGOP works

---

**Current Status**: Implementation complete but strategy needs adjustment based on model architecture constraints.


