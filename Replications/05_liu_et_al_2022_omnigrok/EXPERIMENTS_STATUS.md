# Paper 05: Omnigrok - Experiments Status

**Generated**: November 20, 2025
**Status**: Experiments Running

---

## Summary

All 5 runnable experiments have been submitted. IMDb requires manual dataset download.

### Corrections Applied

1. **Optimizer**: Changed from `AdamW` to `Adam` in all experiments (matches paper specification)
2. **IMDb Architecture**: Fixed `hidden_dim` from 256 to 128 (matches paper specification)

---

## Experiment Status

### ✅ Running/Completed Experiments

| Experiment | Job ID | Status | Architecture | Expected Time |
|------------|--------|--------|--------------|---------------|
| **MNIST** (corrected) | 44339203 | Running | 3-layer MLP, width=200, ReLU | 2-4 hours |
| **Teacher-Student** | 44339204 | Running | 3-layer MLP, width=100, Tanh | 4-6 hours |
| **MNIST Representation** | 44339205 | Running | 3-layer MLP, width=200, ReLU | 2-4 hours |
| **QM9 Molecules** | 44339199 | Pending | 2-layer GCNN | 8-12 hours |
| **Modular Addition** | 44339201 | Pending | 1-layer Transformer | 6-8 hours |

### ⚠️ Requires Manual Setup

| Experiment | Status | Issue | Solution |
|------------|--------|-------|----------|
| **IMDb Sentiment** | Not Run | Missing dataset | Download IMDB Dataset.csv from Kaggle |

**IMDb Dataset Download Instructions:**
1. Visit: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
2. Download `IMDB Dataset.csv`
3. Place in: `/om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/Replications/05_liu_et_al_2022_omnigrok/imdb/grokking/`
4. Run: `sbatch scripts/run_imdb.sh`

---

## File Corrections Made

### Optimizer Changes (AdamW → Adam)

1. ✅ `mnist/grokking/mnist_grokking_logged.py` - Line 84
2. ✅ `mnist/grokking/mnist-grokking.ipynb` - Cell 7
3. ✅ `imdb/grokking/imdb-grokking` - Line 387
4. ✅ `qm9/grokking/qm9-grokking.ipynb` - Cell 0
5. ✅ `teacher-student/grokking/regression_grokking.ipynb` - Cell 2
6. ✅ `mod-addition/grokking/modular-addition-grokking.ipynb` - Cells 13 & 21

### Architecture Corrections

1. ✅ `imdb/grokking/imdb-grokking` - `hidden_dim` 256 → 128 (Line 369)

---

## Monitoring Jobs

Check job status:
```bash
squeue -u mabdel03 | grep paper05
```

Check logs:
```bash
ls -lt results/logs/
tail -f results/logs/mnist_corrected_*.out
```

---

## Expected Results

All experiments should demonstrate grokking:

1. **MNIST**: Smooth grokking, 100% train, ~89% test (previously observed)
2. **IMDb**: Subtle grokking, longer training
3. **QM9**: Clear grokking with small training sets
4. **Teacher-Student**: Regression task grokking (Figure 2 in paper)
5. **Modular Addition**: Sharp grokking transitions (Figure 6 & 8 in paper)
6. **MNIST Representation**: Representation learning dynamics (Figure 7)

---

## Next Steps

1. **Wait for experiments to complete** (~4-12 hours)
2. **Analyze results** using visualization scripts
3. **Download and run IMDb** experiment
4. **Create comprehensive results document** comparing all 6 experiments
5. **Verify each demonstrates grokking** as per paper specifications

---

## Results Files

Experiments will generate:
- `results/logs/training_history.json` (MNIST)
- `results/logs/qm9_training_history.json` (QM9)
- `results/logs/teacher_student_training_history.json` (Teacher-Student)
- Plotting outputs in individual experiment directories

