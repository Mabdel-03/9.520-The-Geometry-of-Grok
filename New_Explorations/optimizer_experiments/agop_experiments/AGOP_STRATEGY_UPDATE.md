# AGOP Strategy Update - Critical Issue Found

## Problem Discovered During Testing

The original plan was to use **input-gradient AGOP** for all experiments (tractable approach from notebook). However, testing revealed a fundamental incompatibility:

### Input Format Mismatch

| Model | Input Type | AGOP Compatibility |
|-------|------------|-------------------|
| **Notebook MLP** | One-hot vectors (float) | ✅ Input-gradient AGOP works |
| **Nanda Transformer** | Token indices (long) | ❌ Cannot compute input gradients |
| **Softmax Transformer** | Token indices (long) | ❌ Cannot compute input gradients |
| **MNIST MLP** | Pixel values (float) | ✅ Input-gradient AGOP works |
| **Composition Transformer** | Token indices (long) | ❌ Cannot compute input gradients |

### Why Token-Based Models Fail

```python
# Nanda data format
x = torch.tensor([[5, 12, 97]])  # [a, b, equals_token] - INTEGERS
x.requires_grad_(True)  # ❌ ERROR: only floats can require gradients

# Embedding layers
embedding(x)  # ❌ ERROR: embeddings require integer indices, not floats
```

**Root cause**: Embeddings require integer indices, but gradient computation requires float tensors. These requirements are mutually exclusive.

---

## Updated Strategy

### Option 1: Use Existing Parameter-Gradient AGOP (Recommended)

The existing framework (`framework/spectral_metrics.py`) already handles this correctly with heavy subsampling:

**For Nanda/Softmax/Composition:**
- ✅ Use parameter-gradient AGOP (already implemented)
- ✅ Subsample aggressively (e.g., 500 samples)
- ✅ Compute less frequently (e.g., every 500 epochs)
- ✅ Track only top-20 eigenvalues

**For MNIST:**
- ✅ Use input-gradient AGOP (tractable, 784-dim)
- ✅ Implemented in `agop_utils.py`

### Option 2: Modify Model to Accept Continuous Inputs

Create one-hot encoded versions of the models (matches notebook):
- Convert token indices to one-hot vectors
- Remove embedding layers
- Use linear layer instead

**Pros**: True input-gradient AGOP  
**Cons**: Different architecture than original papers

---

## Recommendation: Hybrid Approach

Use the **existing framework** for embedding-based models:

```python
# train_nanda_agop.py
from trainer import GrokkingTrainer  # Use existing framework

trainer = GrokkingTrainer(
    model=model,
    train_data=train_data,
    train_labels=train_labels,
    test_data=test_data,
    test_labels=test_labels,
    optimizer_name=args.optimizer,
    lr=args.lr,
    weight_decay=args.weight_decay,
    n_epochs=args.n_epochs,
    device=device,
    compute_spectral_metrics=True,  # Enable parameter-gradient AGOP
    spectral_metrics_freq=500,       # Less frequent (expensive)
    agop_subsample_size=500,         # Subsample heavily
    spectral_top_k=20,
)
```

This reuses the proven implementation from your previous experiments.

---

## Action Items

### Immediate (Testing)

1. ✅ MNIST test with input-gradient AGOP - Should work!
2. ⏳ Nanda test with parameter-gradient AGOP - Use existing framework
3. ⏳ Update training scripts to use hybrid approach

### Alternative (If you want pure input-gradient AGOP)

Create one-hot encoded dataset loaders:
- Transform `[5, 12, 97]` → one-hot matrix (p×3 dimensions)
- Modify model to accept continuous one-hot inputs
- Remove embedding layers

---

## Files That Need Updates

If using existing framework (recommended):

1. `train_nanda_agop.py` - Import and use `GrokkingTrainer`
2. `train_softmax_agop.py` - Import and use `GrokkingTrainer`
3. `train_composition_agop.py` - Import and use `GrokkingTrainer`
4. `train_mnist_agop.py` - Keep current input-gradient AGOP ✓

---

## Current Test Status

**Job 44381665** (Nanda test):
- ✅ Training works
- ✅ Model loads correctly
- ✅ Optimization runs
- ❌ Input-gradient AGOP fails (expected - token inputs)
- ⏳ Need to switch to parameter-gradient AGOP

**Next Step**: Either:
1. Accept that only MNIST uses input-gradient AGOP
2. Switch Nanda/Softmax/Composition to use existing `GrokkingTrainer` framework
3. Create one-hot encoded versions of models

Which approach would you prefer?


