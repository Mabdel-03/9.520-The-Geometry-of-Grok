# AGOP Quick Reference Card

## What is AGOP?

**Average Gradient Outer Product** = (1/N) Σᵢ (∇L(xᵢ) ⊗ ∇L(xᵢ))

From: Beaglehole et al., "Average gradient outer product as a mechanism for deep neural collapse"

## Key Metrics

| Metric | Formula | Physical Meaning | Neural Collapse |
|--------|---------|-----------------|-----------------|
| **Trace** | Σλᵢ | E[\\|∇L\\|²] - Average squared gradient norm | Decreases |
| **Spectral Radius** | λ₁ | Maximum variance direction | Large value |
| **Eigengap** | λ₁ - λ₂ | Gradient alignment measure | **Increases** |
| **Top Eigenvalue Energy** | λ₁/Σλᵢ | Concentration in top direction | **Increases** |
| **Effective Rank** | exp(-Σpᵢlog pᵢ) | Dimensionality of gradient space | **Decreases** |

## Grokking Hypothesis

| Phase | Trace | Eigengap | Effective Rank | Top Energy |
|-------|-------|----------|----------------|------------|
| Memorization (Early) | High | Small | High | Low |
| **Transition (Grokking)** | ↓ | **↑** | **↓** | **↑** |
| Generalization (Late) | Low | Large | Low | High |

**Key Prediction**: Eigengap and top eigenvalue energy should **increase during grokking**

## Usage in Framework

```python
trainer = GrokkingTrainer(
    model=model,
    train_data=train_data,
    train_labels=train_labels,
    test_data=test_data,
    test_labels=test_labels,
    
    # Standard params
    optimizer_name='adamw',
    lr=0.001,
    weight_decay=1.0,
    n_epochs=40000,
    
    # AGOP settings (IMPORTANT!)
    compute_spectral_metrics=True,
    spectral_metrics_freq=100,      # Every 100 epochs (EXPENSIVE!)
    spectral_top_k=20,              # Track top-20 eigenvalues
    agop_subsample_size=1000,       # Use 1000 samples (saves time)
    compute_per_layer=False,        # Usually disable (very expensive)
)
```

## Recommended Settings

### Small Models (<100K params)
```python
spectral_metrics_freq=100
agop_subsample_size=None  # Use all data
```

### Medium Models (100K-500K params)
```python
spectral_metrics_freq=200
agop_subsample_size=1000
```

### Large Models (>500K params)
```python
spectral_metrics_freq=500
agop_subsample_size=500
compute_spectral_metrics=False  # Or very sparse
```

## Reading Results

```python
import h5py
import matplotlib.pyplot as plt

# Load metrics
with h5py.File('results/my_exp/spectral_metrics.h5', 'r') as f:
    epochs = f['epoch'][:]
    eigengap = f['eigengap'][:]
    top_energy = f['top_eigenvalue_energy_ratio'][:]
    trace = f['trace'][:]

# Load training history
import json
with open('results/my_exp/training_history.json') as f:
    history = json.load(f)

# Plot: Does eigengap predict grokking?
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Test accuracy (grokking)
ax1.plot(history['epoch'], history['test_acc'], 'b-')
ax1.set_ylabel('Test Accuracy')
ax1.set_title('Grokking Behavior')

# Eigengap (should increase during grokking!)
ax2.plot(epochs, eigengap, 'r-')
ax2.set_ylabel('Eigengap (λ₁ - λ₂)')
ax2.set_xlabel('Epoch')
ax2.set_yscale('log')
ax2.set_title('Gradient Alignment')

plt.tight_layout()
plt.savefig('grokking_vs_eigengap.png')
```

## What to Look For

### Evidence of Neural Collapse During Grokking:

✅ **Eigengap increases** when test accuracy improves  
✅ **Top eigenvalue energy ratio increases** (→1)  
✅ **Effective rank decreases** (fewer active directions)  
✅ **Trace decreases** (smaller gradients overall)  

### Optimizer Comparison:

- **Does Muon show different eigengap trajectory than Adam?**
- **Does SGD have higher effective rank (more exploration)?**
- **Does weight decay increase eigengap faster?**

## Computational Cost

| Setting | Time per AGOP | Frequency | Overhead |
|---------|---------------|-----------|----------|
| Full (3800 samples) | ~60s | Every 100 epochs | ~0.6s/epoch |
| Subsampled (1000) | ~20s | Every 100 epochs | ~0.2s/epoch |
| Subsampled (500) | ~10s | Every 100 epochs | ~0.1s/epoch |

**Recommendation**: Use subsampling! Still captures key behavior.

## Memory Requirements

- **AGOP matrix**: ~40 GB for 100K params (on CPU)
- **Stored metrics**: ~10 MB per experiment
- **Eigenvalues only**: Minimal storage

**Note**: Matrix stays on CPU, not GPU

## Quick Diagnostics

```python
# Check if AGOP is working
with h5py.File('results/my_exp/spectral_metrics.h5', 'r') as f:
    print("Epochs with AGOP:", f['epoch'][:])
    print("Trace values:", f['trace'][:])
    
    # Sanity checks
    assert len(f['epoch'][:]) > 0, "No AGOP computed!"
    assert all(f['trace'][:] > 0), "Trace should be positive"
    assert all(f['eigengap'][:] >= -1e-6), "Eigengap should be non-negative"
```

## Common Issues

**Issue**: "AGOP computation too slow"  
**Solution**: Increase `spectral_metrics_freq` or reduce `agop_subsample_size`

**Issue**: "Out of memory"  
**Solution**: AGOP uses CPU memory. Check CPU RAM, not GPU memory.

**Issue**: "Eigengap is always zero"  
**Solution**: Model might not be training. Check train/test accuracy first.

**Issue**: "Metrics look random"  
**Solution**: Compute AGOP more frequently to see smooth trends.

## Key Differences from Previous Implementation

| Aspect | OLD (Wrong) | NEW (Correct) |
|--------|-------------|---------------|
| Computation | G ⊗ G where G = mean gradient | (1/N) Σᵢ (Gᵢ ⊗ Gᵢ) |
| Trace | \\|mean gradient\\|² | E[\\|∇L\\|²] |
| Interpretation | Mean direction only | Gradient variance structure |
| Memory | Small (one outer product) | Large (need M×M matrix) |

## Research Questions

1. **Does eigengap predict grokking?** Plot eigengap vs test accuracy
2. **Do optimizers differ in AGOP structure?** Compare eigenspectra
3. **Does weight decay induce collapse?** Higher wd → larger eigengap?
4. **Are there early warning signs?** Does eigengap increase before test accuracy?
5. **Is grokking a form of late neural collapse?** Similar AGOP dynamics?

---

**Read**: `AGOP_IMPLEMENTATION.md` for full details  
**Test**: Run with `spectral_metrics_freq=10` on small model first  
**Analyze**: Look for eigengap increase during grokking phase  

