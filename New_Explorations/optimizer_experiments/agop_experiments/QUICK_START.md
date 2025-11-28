# 🚀 AGOP Experiments - Quick Start Guide

## ⚡ TL;DR - Run This Now

```bash
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/optimizer_experiments/agop_experiments

# Submit comprehensive test (12 tests, ~1-2 hours)
sbatch test_agop_slurm.sh

# Monitor progress
squeue -u $USER
tail -f test_results/test_agop_*.out
```

---

## ✅ Everything is Ready!

### What's Been Fixed & Configured

1. ✅ **Muon optimizer imports** - Now uses `muon_official` (updated implementation)
2. ✅ **Conda environment** - All scripts use `/om/scratch/Tue/mabdel03/9.520/conda_envs/grok_exp`
3. ✅ **Syntax verified** - All 7 scripts compile correctly
4. ✅ **Test infrastructure** - Comprehensive 12-test suite created
5. ✅ **Visualizations** - 9 plots per experiment, with grokking detection

---

## 📊 Complete Test → Run → Analyze Workflow

### Step 1: Quick Test (Start Here!)

```bash
# Submit test job
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/optimizer_experiments/agop_experiments
sbatch test_agop_slurm.sh
```

**What this does:**
- Tests 4 datasets (Nanda, Softmax, MNIST, Composition)
- Tests 3 optimizers per dataset (AdamW, Muon, SGD)
- Runs 200 epochs per test (quick)
- Total: 12 tests in ~1-2 hours

**Expected output:**
```
Submitted batch job 123456
```

**Monitor:**
```bash
# Check job status
squeue -u $USER

# Watch output live
tail -f test_results/test_agop_*.out

# Check for errors
tail -f test_results/test_agop_*.err
```

### Step 2: Verify Tests Passed

```bash
# Check final summary in output file
tail -50 test_results/test_agop_*.out
```

**Look for:**
```
================================================================================
RESULTS: 12/12 tests passed
================================================================================
```

**If all 12 tests pass:** ✅ Proceed to Step 3  
**If any fail:** 🔧 Check error logs and fix issues

### Step 3: Submit Full Experiments

```bash
cd slurm_scripts/

# Submit ALL experiments (48 jobs)
./run_all_agop.sh

# OR submit by dataset:
sbatch run_nanda_agop.sh      # 12 jobs (3 optimizers × 4 weight decays)
sbatch run_softmax_agop.sh    # 12 jobs
sbatch run_mnist_agop.sh      # 12 jobs
# sbatch run_composition_agop.sh  # 12 jobs (optional, run last)
```

**Monitor:**
```bash
squeue -u $USER | wc -l  # Count running jobs
watch -n 30 'squeue -u $USER'  # Auto-refresh every 30s
```

### Step 4: Generate Visualizations

```bash
cd analysis/

# Once an experiment completes, visualize it
python visualize_agop_metrics.py \
    --results_dir ../results/agop_experiments/nanda/nanda_adamw_wd1.0_seed42

# Compare all optimizers for a dataset
python visualize_agop_metrics.py \
    --results_dir ../results/agop_experiments/nanda \
    --experiment_pattern "nanda_*_wd1.0_*" \
    --compare_optimizers

# Analyze grokking vs non-grokking
python compare_grok_nogrok.py \
    --results_dir ../results/agop_experiments/nanda
```

**Generates 9 plots including:**
- Training curves
- AGOP metrics evolution
- Comprehensive timeline with grokking detection
- Dual-axis aligned plots

---

## 🎯 What Each Test Does

| Test # | Dataset | Optimizer | Purpose |
|--------|---------|-----------|---------|
| 1-3 | Nanda | AdamW/Muon/SGD | Verify ReLU transformer + AGOP |
| 4-6 | Softmax | AdamW/Muon/SGD | Verify standard transformer + AGOP |
| 7-9 | MNIST | AdamW/Muon/SGD | Verify MLP + AGOP on images |
| 10-12 | Composition | AdamW/Muon/SGD | Verify sequence model |

**Each test runs 200 epochs and computes AGOP every 50 epochs**

---

## 📁 File Locations

### Your Working Directory
```
/om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/optimizer_experiments/agop_experiments/
```

### Conda Environment
```
/om/scratch/Tue/mabdel03/9.520/conda_envs/grok_exp
```

### Test Results (After Running)
```
test_results/
├── test_agop_*.out           # SLURM output
├── test_agop_*.err           # SLURM errors
├── test_summary_*.json       # JSON summary
└── test_{dataset}_{opt}_seed42/  # Individual test outputs
```

### Full Experiment Results (After Running)
```
results/agop_experiments/
├── nanda/{experiment_name}/
├── softmax/{experiment_name}/
├── mnist/{experiment_name}/
└── composition/{experiment_name}/
```

---

## 🔍 Pre-Flight Checklist

Before submitting jobs, verify:

```bash
# 1. Check conda environment exists
ls -la /om/scratch/Tue/mabdel03/9.520/conda_envs/grok_exp/bin/python
# Expected: Should show python executable

# 2. Verify scripts are executable
ls -la test_agop_slurm.sh slurm_scripts/*.sh
# Expected: Should show -rwxr-xr-x permissions

# 3. Check syntax (optional)
python verify_imports.py
# Expected: 7/7 scripts verified

# 4. Check you're in right directory
pwd
# Expected: .../agop_experiments
```

---

## ⏱️ Timeline Estimates

### Test Suite (`test_agop_slurm.sh`)
- **Time:** 1-2 hours
- **Jobs:** 1 job (12 tests internally)
- **Epochs:** 200 per test
- **Purpose:** Verify everything works

### Full Experiments (Nanda only)
- **Time:** 24-48 hours
- **Jobs:** 12 jobs (3 optimizers × 4 weight decays)
- **Epochs:** 40,000 per job
- **Purpose:** Full grokking analysis

### All Datasets
- **Time:** 3-5 days
- **Jobs:** 48 jobs total
- **Storage:** ~5GB
- **Purpose:** Complete AGOP analysis

---

## 📈 What You'll Learn

After experiments complete and visualizations are generated:

### Research Questions Answered:
1. **Does variation collapse ratio increase before grokking?**
2. **Does eigengap expand during the grokking transition?**
3. **Are there different AGOP patterns for different optimizers?**
4. **Do symbolic tasks (modular) show different AGOP than perceptual (MNIST)?**
5. **Which AGOP metric best predicts grokking onset?**

### Visualizations Generated (per experiment):
- Comprehensive 5-panel timeline with grokking detection
- 4 dual-axis plots showing test accuracy vs AGOP metrics
- Comparison plots across optimizers
- Statistical analysis of grokking vs non-grokking

---

## 🐛 Quick Troubleshooting

### Test Job Fails
```bash
# Check error log
cat test_results/test_agop_*.err

# Verify environment
srun --pty --gres=gpu:1 bash
/om/scratch/Tue/mabdel03/9.520/conda_envs/grok_exp/bin/python --version
```

### Import Errors
```bash
# Should already be fixed, but verify:
python verify_imports.py
```

### Out of Memory
```bash
# For MNIST tests, reduce subsample:
# Edit test_all_experiments.py:
# 'agop_subsample': 100,  # Instead of 250
```

---

## 📚 Documentation

| File | Purpose |
|------|---------|
| **READY_TO_RUN.md** | This guide - start here! |
| **README.md** | Complete documentation |
| **TEST_REPORT.md** | What was fixed and why |
| **IMPLEMENTATION_COMPLETE.md** | Technical summary |
| **ENHANCED_VISUALIZATIONS.md** | Visualization features |
| **analysis/VISUALIZATION_GUIDE.md** | How to interpret plots |

---

## 🎉 You're Ready!

Everything is configured and ready to run. Just submit:

```bash
sbatch test_agop_slurm.sh
```

Once tests pass, submit full experiments:

```bash
cd slurm_scripts/
./run_all_agop.sh
```

Good luck with your AGOP analysis! 🚀


