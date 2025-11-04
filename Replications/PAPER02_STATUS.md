# Paper 02 Status: Liu et al. (2022) - Effective Theory

**Status**: ✅ **TRAINING - 62% Complete** (Job 44188126)  
**Priority**: High - Should complete soon!

---

## Current Status

### Training Progress
- **Job ID**: 44188126
- **Progress**: 31,000 / 50,000 steps (62%)
- **Speed**: ~500-580 it/s
- **Estimated Time Remaining**: ~10-15 minutes
- **Status**: Running smoothly with Sacred framework

---

## What the Paper Does

**Paper**: "Towards Understanding Grokking: An Effective Theory of Representation Learning"

### Key Contributions
1. **Phase Diagram of Grokking**: Maps out four phases:
   - Comprehension (good train + good test)
   - Grokking (good train, delayed test improvement)
   - Memorization (good train, poor test)
   - Confusion (poor train + poor test)

2. **Representation Quality Index (RQI)**: Quantifies structure in learned representations

3. **Theoretical Framework**: Explains grokking through lens of representation learning

### Task
- Modular addition with **toy model** architecture
- Encoder → Internal representation → Decoder
- **Key insight**: Different learning rates for encoder vs decoder

### Configuration (Current Run)
```python
p = 10                      # 10 symbols
train_fraction = 0.45       # 45% training data
train_steps = 50,000        # Extended training
encoder_lr = 1e-3
decoder_lr = 1e-3
encoder_weight_decay = 1.0
decoder_weight_decay = 1.0
hidden_rep_dim = 1          # 1D internal representation
```

---

## Issues Encountered & Fixed

### Issue 1: Simplified Script Had Bugs ❌
- **Problem**: `train_simple.py` had tensor shape mismatch
- **Error**: Model output (batch, 20) vs labels (batch, 10)
- **Cause**: Wrong forward pass implementation

### Issue 2: Sacred Framework Required ✅
- **Solution**: Use original `scripts/run_toy_model.py` with Sacred
- **Status**: Working perfectly!
- **Output**: Sacred manages its own logging

---

## Sacred Framework

Paper 02 uses **Sacred** for experiment tracking:
- Structured configuration management
- Automatic logging of parameters
- Results saved in Sacred's format
- Need to extract to `training_history.json` after completion

### Sacred Output Locations
```
02_liu_et_al_2022_effective_theory/
├── logs/sacred/           # Sacred run directories
├── _sources/              # Source code snapshots
└── config.json            # Configuration files
```

---

## Next Steps

### 1. Wait for Completion (ETA: 10-15 min)
```bash
squeue -j 44188126  # Check status
tail -f toy_model_44188126.out  # Watch progress
```

### 2. Extract Sacred Outputs
Sacred doesn't automatically save to `training_history.json`. Need to:
- Parse Sacred's output format
- Extract train/test metrics over time
- Convert to standard JSON format

### 3. Create Extraction Script
```python
# Extract from Sacred runs
# Look for metrics in Sacred's run directories
# Convert to training_history.json format
```

### 4. Analyze for Grokking
Look for:
- Train accuracy → 100%
- Test accuracy improvement over time
- Phase transitions in representation quality

---

##Expected Grokking Behavior

Based on the paper, we should observe:

### Metrics Timeline
| Phase | Steps | Train Acc | Test Acc | RQI | Phase |
|-------|-------|-----------|----------|-----|--------|
| Early | 0-5k | → 100% | Low | Low | Memorization |
| Middle | 5k-30k | 100% | Improving | Increasing | Grokking |
| Late | 30k-50k | 100% | High | High | Comprehension |

### Key Indicators
1. ✅ Fast memorization (train acc → 100% early)
2. ⏳ Gradual test improvement (grokking phase)
3. ⏳ High final test accuracy
4. ⏳ Increasing representation quality (RQI)

---

## Files

### Training Scripts
- `scripts/run_toy_model.py` - Main Sacred-based script ✅ WORKING
- `scripts/run_toy_model_ng.py` - Alternative version
- `train_simple.py` - Simplified (has bugs) ❌

### Run Scripts
- `run_toy_model.sh` - SLURM script for Sacred version ✅ USING THIS

### Output
- `toy_model_44188126.out` - Training progress log
- `toy_model_44188126.err` - Error log (warnings only)

---

## Comparison to Other Papers

### Similar to Paper 01 (Power et al.)
- Both use modular addition
- Both demonstrate grokking
- Paper 02 has theoretical framework

### Key Differences
| Aspect | Paper 01 | Paper 02 |
|--------|----------|----------|
| Architecture | 2-layer Transformer | Encoder-Decoder MLP |
| Focus | Phenomenon discovery | Theoretical explanation |
| Key metric | Test accuracy | RQI + Test accuracy |
| Training | 100k steps | 50k steps (toy model) |

---

## Success Criteria

Paper 02 will be considered successfully replicated if:

### Required
- [x] Script runs without errors
- [ ] Training completes 50,000 steps
- [ ] Sacred outputs are extractable
- [ ] Metrics show grokking behavior

### Ideal
- [ ] Can extract training history to JSON
- [ ] Test accuracy improves over time
- [ ] Can compute RQI from representations
- [ ] Results match paper's phase diagram

---

## Outstanding Tasks

1. **Immediate**: Wait for training to complete (~10 min)
2. **Next**: Extract Sacred outputs
3. **Then**: Convert to `training_history.json`
4. **Finally**: Generate plots and analyze grokking

---

## Key Papers Reference

**Original Paper**:
- Title: "Towards Understanding Grokking: An Effective Theory of Representation Learning"
- Authors: Liu, Kitouni, Nolte, Michaud, Tegmark, Williams
- Link: https://arxiv.org/abs/2205.10343
- Year: 2022

**Code Repository**:
- https://github.com/ejmichaud/grokking-squared

---

**Current Status**: ✅ Training running successfully! ETA completion: ~10-15 minutes

**Next Update**: After training completes, extract outputs and analyze results

