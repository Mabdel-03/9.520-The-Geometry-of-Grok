# Paper 08 Status: Doshi et al. (2024) - Modular Polynomials

**Status**: ⚠️ **MODEL DIDN'T LEARN - Needs Debugging**  
**Priority**: Medium (architecture-specific issue)

---

## Problem Diagnosis

### Symptom
- **Loss stuck at 0.0103** (exactly 1/97) from epoch 0 to 99,999
- **Accuracy ~1%** (random for 97 classes)
- **No learning occurred** at all

### Root Cause
The loss value 0.0103 = 1/p = 1/97 indicates the model outputs **uniform predictions**.

This means:
1. Gradients might be vanishing/exploding
2. Power activation φ(x) = x² may not be working correctly
3. Initialization inappropriate for power activation
4. Learning rate incompatible with this architecture

---

## What the Paper Does

**Unique Contribution**: Derives **analytical solutions** for grokking!

### Key Innovation
- Constructs "expert" networks with exact weights that solve modular polynomials
- Shows networks can discover these analytical solutions through training
- Uses power activation φ(x) = x^S (element-wise power)

### Tasks
1. **Modular addition**: (c₁n₁ + c₂n₂ + ... + cₛnₛ) mod p
2. **Modular multiplication**: (n₁ᵃ × n₂ᵇ) mod p

### Architecture
**2-layer MLP with power activation:**
```
Input (one-hot) → Embeddings U^(1),...,U^(S) 
                → Sum → φ(x) = x^power  
                → Output matrix W → Logits
```

---

## Training Configuration Used

```python
task = "addition"
p = 97                    # mod 97
num_terms = 2             # 2-term addition
hidden_dim = 500          # N=500
power = 2                 # φ(x) = x²
train_fraction = 0.5
lr = 0.001                # Reduced from 0.005
weight_decay = 1.0        # Reduced from 5.0
n_epochs = 100,000
```

**Parameters**: 145,500

---

## What Went Wrong

### Loss Analysis
```
Epoch 0:      Loss = 0.0103 ← Uniform predictions
Epoch 1000:   Loss = 0.0103 ← No change
Epoch 10000:  Loss = 0.0103 ← Still no change  
Epoch 99999:  Loss = 0.0103 ← Never learned
```

### Possible Issues

1. **Power Activation Problem**
   - x² might cause numerical instability
   - Gradients through power function may vanish
   - Need careful initialization for power activations

2. **Initialization**
   - Current: `randn() / sqrt(dim)`
   - Power activation may need different scaling
   - Analytical solution might require specific initialization

3. **Learning Rate**
   - 0.001 might be too large or too small
   - Power activation has different gradient scale
   - May need very different LR than standard MLPs

4. **Loss Function**
   - MSE on one-hot may not be ideal
   - Cross-entropy might work better
   - Softmax before loss might help

---

## Paper's Recommended Parameters

From README:
```python
lr = 0.005                # Paper's default
weight_decay = 5.0        # Much higher than other papers!
hidden_dim = 500          # For multiplication
hidden_dim = 5000         # For multi-term addition
```

We used reduced values (lr=0.001, WD=1.0) which might be too conservative.

---

## Potential Fixes to Try

### Fix 1: Use Paper's Original Hyperparameters
```bash
python train.py \
    --lr=0.005 \
    --weight_decay=5.0 \
    --hidden_dim=5000  # Larger network
```

### Fix 2: Try Cross-Entropy Loss
Replace MSE with Cross-Entropy (more standard for classification)

### Fix 3: Different Initialization
```python
# Instead of randn() / sqrt(dim)
# Try Xavier or specific initialization for power activation
```

### Fix 4: Add Gradient Clipping
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Fix 5: Check Gradients
Add gradient monitoring to see if they're non-zero

---

## Why This Paper Is Interesting

Despite the current issue, this paper is valuable because:
1. **Analytical framework**: Only paper with exact solutions
2. **Novel architecture**: Power activation (not ReLU/Tanh)
3. **Theoretical grounding**: Explains WHY networks can grok
4. **Extends to polynomials**: Beyond simple addition

---

## Next Steps

### Quick Test (5 minutes)
Try with paper's original hyperparameters:
```bash
sbatch run_addition.sh  # Use original, not _fixed
```

### If Still Fails (30 minutes)
1. Add gradient monitoring
2. Try Cross-Entropy loss
3. Check power activation implementation
4. Try different initialization schemes

### If Complex (Skip for now)
Come back after getting other papers working

---

## Comparison with Working Papers

| Paper | Architecture | Activation | LR | WD | Status |
|-------|--------------|-----------|----|----|---------|
| 03 (Nanda) | Transformer | ReLU | 1e-3 | 1.0 | ✅ Works |
| 07 (Slingshot) | Transformer | - | 1e-3 | 1.0 | ✅ Works |
| **08 (Doshi)** | **MLP** | **x²** | **1e-3** | **1.0** | **❌ Fails** |

The power activation makes this fundamentally different from other papers.

---

**Decision**: Document and move to Papers 06, 09, 10. Return to Paper 08 if time permits.

**Priority**: Medium - interesting but complex to debug.

