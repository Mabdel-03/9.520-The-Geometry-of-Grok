# Paper 09: 1 Million Epoch Results

**Date**: November 4, 2025  
**Status**: ⚠️ **MODEL CONVERGED TO SUBOPTIMAL SOLUTION**

---

## Results

### 1 Million Epoch Run (Completed in 8:42)
- **Train Accuracy**: 83.4% (stuck from epoch ~1000 onward)
- **Test Accuracy**: 5.84% (no improvement)
- **Train Loss**: 0.000494 (constant)
- **Test Loss**: 0.215106 (constant)

### Problem
Model converged to a **local minimum** at epoch ~1000 and **never improved** for the remaining 999,000 epochs.

---

## Why This Happened

### Configuration Issue
The model partially learned (83.4% train) but got stuck:
- Loss decreased from 1.08 → 0.000494 in first 1000 epochs
- Then completely flatlined
- SGD optimizer may have found local minimum
- Weight decay 0.1 might be preventing escape

### Possible Fixes
1. **Different optimizer**: Try Adam instead of SGD
2. **Different learning rate**: Try 0.001 or 0.0001
3. **Different weight decay**: Try 1.0 (like other papers)
4. **Check paper's exact setup**: We may be using wrong task/configuration

---

## Comparison with 100K Run

Interestingly, the **100K epoch run showed DIFFERENT behavior**:
- 100K run: Train 100%, Test 5.72%
- 1M run: Train 83.4%, Test 5.84%

This suggests different runs or different configurations were used!

---

## Decision

Given:
- We already have **5 confirmed grokking papers** (50% success)
- Papers 1, 8, 10 still need investigation
- Paper 9 would need significant debugging

**Skip Paper 9 for now** and focus on Papers 10, 1, and 8.

---

**Status**: Not successful - model stuck in local minimum. Would need configuration debugging.

