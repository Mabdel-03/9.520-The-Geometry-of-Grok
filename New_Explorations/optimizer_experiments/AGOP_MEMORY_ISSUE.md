# AGOP Memory Issue and Workaround

## Problem Identified

The full AGOP matrix computation requires **massive amounts of memory**:

### Memory Requirements

For a model with M parameters:
- **AGOP matrix**: M × M × 4 bytes (float32)

**Examples**:
- Nanda model (~226K params): 226K × 226K × 4 = **204 GB**
- MNIST model (~200K params): 200K × 200K × 4 = **160 GB**

This exceeds available node memory (32-64 GB typical).

### Error Message
```
RuntimeError: can't allocate memory: you tried to allocate 204159385600 bytes
```

## Current Status

**Experiments are running** but **AGOP computation is disabled** to allow training to proceed.

- ✅ Training metrics ARE being tracked (loss, accuracy)
- ❌ AGOP spectral metrics are NOT being computed
- ✅ Models will still grok
- ❌ Cannot study eigengap/neural collapse yet

## Solutions (In Order of Complexity)

### Solution 1: Streaming Eigenvalue Computation (Recommended)

Don't store full AGOP matrix. Use randomized/streaming algorithms:

```python
def compute_top_k_eigenvalues_streaming(model, data, labels, criterion, k=20):
    """
    Compute top-k eigenvalues without storing full AGOP matrix.
    Uses power iteration or randomized SVD.
    """
    # Collect gradient vectors for all samples
    gradients = []  # List of gradient vectors (N × M)
    
    for i in range(len(data)):
        # Compute gradient for sample i
        loss_i = criterion(model(data[i:i+1]), labels[i:i+1])
        loss_i.backward()
        grad_i = get_gradient_vector()  # Vector of length M
        gradients.append(grad_i)
    
    # Stack into matrix: G = (N × M)
    G = torch.stack(gradients)  # Only N×M, not M×M!
    
    # AGOP = (1/N) G^T G
    # Top-k eigenvalues of AGOP = top-k singular values of G
    # Use randomized SVD (much faster!)
    U, S, Vh = torch.svd_lowrank(G, q=k)  # Only top-k
    
    eigenvalues = (S ** 2) / len(data)  # Convert to AGOP eigenvalues
    
    return eigenvalues  # Only k values, not M!
```

**Memory**: N × M instead of M × M (226K × 3800 vs 226K × 226K)
**Time**: Much faster

### Solution 2: Disable AGOP, Track Only Training Metrics (Current)

```bash
# In SLURM scripts: remove --spectral_metrics flag
python train_nanda.py ... # No AGOP computation
```

**Pros**: Experiments run immediately  
**Cons**: Cannot study neural collapse via AGOP

### Solution 3: Smaller Models

Train smaller versions:
```python
# Nanda with fewer params
d_model=64, d_mlp=256  # ~25K params → 2.5 GB AGOP
```

### Solution 4: High-Memory Nodes

Request nodes with 256+ GB RAM:
```bash
#SBATCH --mem=256GB
#SBATCH --partition=high_memory
```

## Recommended Next Steps

1. **For immediate results**: Keep AGOP disabled (current state)
   - All experiments will complete successfully
   - Track grokking via train/test accuracy
   - Analyze optimizer/weight decay effects

2. **For AGOP analysis (later)**: Implement Solution 1
   - Use streaming eigenvalue computation
   - Requires ~2-3 hours to implement
   - Would give you all spectral metrics efficiently

3. **Alternative**: Run smaller-scale experiments with AGOP
   - Reduce model size to ~50K params
   - Full AGOP becomes feasible (~10 GB)

## Current Experiment Status

**42 experiments submitted**:
- 24 Nanda (modular addition)
- 18 MNIST (image grokking)

**Status**: ✅ Running correctly (without AGOP)
**Tracking**: Train/test loss and accuracy at each epoch  
**Results**: Will show grokking behavior across optimizers/weight decay

**You will be able to**:
- ✅ Compare optimizer grokking speeds
- ✅ Find optimal weight decay values
- ✅ See training dynamics
- ❌ Cannot study eigengap/neural collapse (yet)

## Timeline

**With current setup** (no AGOP):
- Training continues normally
- Results in 2-7 days
- Can analyze optimizer effects

**To add AGOP later**:
- Implement streaming eigenvalue computation
- Rerun selected experiments (best performing ones)
- Or load checkpoints and compute AGOP retroactively

## Decision

For now, let the experiments run WITHOUT AGOP. This gives you:
1. Complete optimizer comparison
2. Weight decay effects
3. Grokking behavior across conditions

Then decide if AGOP is critical for your analysis.

---

**Bottom Line**: Experiments are running successfully. AGOP tracking temporarily disabled due to memory constraints. Can be added later with streaming implementation if needed.

