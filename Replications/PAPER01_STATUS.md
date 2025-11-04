# Paper 01 Status: Power et al. (2022) - OpenAI Grok

**Status**: ⚠️ **BLOCKED - PyTorch Lightning 2.0 Incompatibility**  
**Priority**: Medium (come back after checking other papers)

---

## Issues Found

### 1. Argument Errors (FIXED)
- ✅ Removed `--val_every=500` (doesn't exist in codebase)
- ✅ Changed `--math_operator=x+y` to `--math_operator="+"`

### 2. PyTorch Lightning 2.0 Incompatibilities (PARTIALLY FIXED)

#### Fixed:
- ✅ `self.hparams = hparams` → `save_hyperparameters()`
- ✅ `flush_logs_every_n_steps` removed
- ✅ `min_steps` removed
- ✅ `gpus=[X]` → `accelerator="gpu"` + `devices=[X]`

#### Still Needed:
- ❌ `training_epoch_end()` → `on_train_epoch_end()`
- ❌ `validation_epoch_end()` → `on_validation_epoch_end()`  
- ❌ `test_epoch_end()` → `on_test_epoch_end()`

### 3. Root Cause
The OpenAI grok repository was written for PyTorch Lightning 1.x but the conda environment has 2.x. This requires significant code migration.

---

## What the Paper Does

**Task**: Modular addition (x + y mod 97)  
**Architecture**: 2-layer Transformer (4 heads, d_model=128, ReLU)  
**Training**: 100,000 steps with AdamW  
**Key hyperparameters**:
- Training fraction: 50%
- Weight decay: 1.0 (critical!)
- Learning rate: 1e-3

**Expected grokking**: Test accuracy jumps from ~10% to ~99% between 10k-50k steps

---

## Solutions

### Option A: Complete PyTorch Lightning Migration
Migrate all epoch_end methods:
- `training_epoch_end` → `on_train_epoch_end`
- `validation_epoch_end` → `on_validation_epoch_end`
- `test_epoch_end` → `on_test_epoch_end`

Estimated time: 30-60 minutes

### Option B: Create Separate Environment
Create a new conda environment with PyTorch Lightning 1.x specifically for Paper 01:
```bash
conda create -n grok_paper01 python=3.8
conda activate grok_paper01  
pip install pytorch-lightning==1.9.0
```

### Option C: Use Docker/Singularity
Run Paper 01 in an isolated container with correct dependencies.

---

## Files Modified

1. `run_paper01_with_logging.sh` - Fixed arguments
2. `train_with_logging.py` - Added CSV to JSON converter
3. `grok/training.py` - Partial PL 2.0 migration
4. `convert_logs.py` - Created log converter script
5. `run_paper01_simple.sh` - Created simplified runner

---

## Next Steps

**After reviewing other papers:**
1. Choose migration strategy (A, B, or C)
2. Complete PyTorch Lightning migration
3. Submit long training run (100k steps, ~4-6 hours)
4. Generate grokking visualizations

---

## Expected Output

Once working, should generate:
- `logs/training_history.json` with 100k steps of metrics
- Clear grokking transition visible in plots
- Train accuracy: 100%
- Test accuracy: ~99%
- Grokking point: ~10k-50k steps

---

**Decision**: Move to Paper 02, return to Paper 01 after understanding patterns across all papers.

