# 🚀 ALL AGOP EXPERIMENTS SUBMITTED

**Date:** November 26, 2024  
**Status:** 72 jobs submitted and running/queued

---

## Submission Summary

### Job Details

| Dataset | Architectures | Optimizers | Weight Decays | Total Jobs | Job ID |
|---------|--------------|------------|---------------|------------|---------|
| **Nanda** | MLP, Transformer | AdamW, Muon, SGD | 0.1, 1.0, 5.0, 10.0 | 24 | 44382007 |
| **Softmax** | MLP, Transformer | AdamW, Muon, SGD | 0.01, 0.1, 0.5, 1.0 | 24 | 44382008 |
| **MNIST** | MLP | AdamW, Muon, SGD | 0.01, 0.1, 0.5, 1.0 | 12 | 44382009 |
| **Composition** | MLP | AdamW, Muon, SGD | 0.01, 0.1, 0.5, 1.0 | 12 | 44382010 |
| **TOTAL** | | | | **72** | |

---

## Experiment Matrix

### Nanda (24 jobs)
```
MLP × AdamW × [0.1, 1.0, 5.0, 10.0]     = 4 jobs
MLP × Muon × [0.1, 1.0, 5.0, 10.0]      = 4 jobs
MLP × SGD × [0.1, 1.0, 5.0, 10.0]       = 4 jobs
Transformer × AdamW × [0.1, 1.0, 5.0, 10.0] = 4 jobs
Transformer × Muon × [0.1, 1.0, 5.0, 10.0]  = 4 jobs
Transformer × SGD × [0.1, 1.0, 5.0, 10.0]   = 4 jobs
```

### Softmax (24 jobs)
```
MLP × AdamW × [0.01, 0.1, 0.5, 1.0]     = 4 jobs
MLP × Muon × [0.01, 0.1, 0.5, 1.0]      = 4 jobs
MLP × SGD × [0.01, 0.1, 0.5, 1.0]       = 4 jobs
Transformer × AdamW × [0.01, 0.1, 0.5, 1.0] = 4 jobs
Transformer × Muon × [0.01, 0.1, 0.5, 1.0]  = 4 jobs
Transformer × SGD × [0.01, 0.1, 0.5, 1.0]   = 4 jobs
```

### MNIST (12 jobs)
```
MLP × AdamW × [0.01, 0.1, 0.5, 1.0]     = 4 jobs
MLP × Muon × [0.01, 0.1, 0.5, 1.0]      = 4 jobs
MLP × SGD × [0.01, 0.1, 0.5, 1.0]       = 4 jobs
```

### Composition (12 jobs)
```
MLP × AdamW × [0.01, 0.1, 0.5, 1.0]     = 4 jobs
MLP × Muon × [0.01, 0.1, 0.5, 1.0]      = 4 jobs
MLP × SGD × [0.01, 0.1, 0.5, 1.0]       = 4 jobs
```

---

## Experiment Configurations

### Common Settings
- **Epochs:** 40,000 (Nanda), 50,000 (Softmax, MNIST), 100,000 (Composition)
- **AGOP frequency:** Every 100 epochs
- **Top-k eigenvalues:** 20
- **Seed:** 42
- **Device:** CUDA

### Dataset-Specific
**Nanda:**
- Modulus p=113
- Train fraction: 30%
- Input dim: 226 (2*113) one-hot
- AGOP matrix: 226×226

**Softmax:**
- Modulus p=97
- Train fraction: 50%
- Input dim: 194 (2*97) one-hot
- AGOP matrix: 194×194

**MNIST:**
- Training points: 1000
- Input dim: 784 (flattened pixels)
- AGOP matrix: 784×784
- Subsampled: 500 examples for AGOP

**Composition:**
- Entities: 50
- Input dim: 1000 (vocab×seq_len)
- AGOP matrix: 500×500

---

## Expected Outputs

Each job will create:
```
results/{dataset}/{experiment_name}/
├── config.json
├── training_history.json
└── agop_metrics.h5
```

**Total expected files:** 72 × 3 = 216 files

**Total expected storage:** ~10-15 GB

---

## Monitoring

### Check Job Status
```bash
# All jobs
squeue -u $USER

# Count running
squeue -u $USER | grep agop | wc -l

# Watch live
watch -n 30 'squeue -u $USER'
```

### Check Logs
```bash
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/agop_experiments/slurm_scripts

# Latest Nanda log
tail -f logs/nanda_agop_44382007_*.out

# Check for errors
tail -f logs/nanda_agop_44382007_*.err
```

### Check Results
```bash
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/agop_experiments

# Count completed experiments
find results/ -name "agop_metrics.h5" | wc -l

# List recent completions
find results/ -name "agop_metrics.h5" -mmin -60
```

---

## Expected Timeline

- **Shortest jobs:** MNIST (~6-12 hours at 50K epochs)
- **Medium jobs:** Nanda/Softmax (~12-24 hours at 40-50K epochs)
- **Longest jobs:** Composition (~24-48 hours at 100K epochs)

**Estimated completion:** 2-3 days for all 72 jobs

---

## Analysis Workflow

Once experiments complete:

### 1. Check Individual Experiments
```bash
cd analysis/
python visualize_agop_metrics.py \
    --results_dir ../results/nanda/nanda_mlp_adamw_wd1.0_seed42
```

### 2. Compare Architectures
```bash
python visualize_agop_metrics.py \
    --results_dir ../results/nanda \
    --experiment_pattern "nanda_*_adamw_wd1.0_*" \
    --compare_optimizers
```

### 3. Compare Grokking vs Non-Grokking
```bash
python compare_grok_nogrok.py \
    --results_dir ../results/nanda
```

### 4. Cross-Dataset Analysis
Compare symbolic (Nanda) vs perceptual (MNIST) grokking patterns

---

## Research Questions to Answer

With 72 experiments, you can analyze:

1. **Architecture effect:** MLP vs Transformer with same one-hot inputs
2. **Optimizer effect:** AdamW vs Muon vs SGD
3. **Regularization effect:** Weight decay sweep
4. **Task effect:** Symbolic (modular) vs Perceptual (MNIST)
5. **AGOP signatures:** Which metrics predict grokking?

---

## Expected Grokking Patterns

Based on literature and initial tests:

**Likely to grok:**
- AdamW with high weight decay (1.0, 5.0, 10.0)
- Possibly Muon with moderate weight decay

**Likely NOT to grok:**
- SGD (especially low weight decay)
- Low weight decay across all optimizers

**Compare AGOP metrics** between these conditions!

---

**Status:** ✅ All 72 jobs submitted  
**Monitor:** `squeue -u $USER`  
**Next:** Wait 2-3 days, then analyze results!


