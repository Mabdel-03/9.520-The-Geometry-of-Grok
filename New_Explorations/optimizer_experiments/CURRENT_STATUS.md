# Experiments Status - Live Report

**Generated**: November 23, 2025, 02:00 EST  
**Total Experiments**: 42 (24 Nanda + 18 MNIST)

---

## ✅ SUCCESS! Experiments Running

### Paper 3 (Nanda) - Modular Addition

**Status**: 20/24 completed, 4 still running

#### 🏆 Grokking Observed (6 experiments)

| Experiment | Train Acc | Test Acc | Grok Epoch | Speed |
|------------|-----------|----------|------------|-------|
| **nanda_adamw_wd1.0** | 100% | **100%** | 7,500 | ⭐⭐⭐ |
| **nanda_adamw_wd2.0** | 100% | **100%** | 2,600 | ⭐⭐⭐⭐ |
| **nanda_adamw_wd5.0** | 100% | **98.4%** | 900 | ⭐⭐⭐⭐⭐ |
| **nanda_adamw_wd0.5** | 100% | **99.98%** | 12,600 | ⭐⭐ |
| **nanda_adamw_wd10.0** | 99.3% | **95.9%** | 400 | ⭐⭐⭐⭐⭐ |
| **nanda_sgd_wd0.01** | 100% | **99.5%** | 36,800 | ⭐ |

**Key Findings**:
- ✅ **AdamW performs best** - 5/6 grokking experiments
- ✅ **Weight decay 2.0-5.0** shows fastest grokking (epochs 900-2600)
- ✅ **Higher weight decay → faster grokking** (but may reduce final accuracy)
- ⚠️ **Muon not working well** - Low accuracy across all weight decays
- ⚠️ **SGD groksvery slowly** - Only 1 success at epoch 36,800

#### ❌ No Grokking (14 experiments)

**AdamW**:
- wd=0.0, 0.01: No grokking (insufficient regularization)

**Muon**:
- All weight decays: Low accuracy (~1%) - Optimizer may need different hyperparameters

**SGD**:
- Most weight decays: Low accuracy or no grokking

### Paper 5 (MNIST) - Image Grokking

**Status**: 18 experiments running (started more recently)

Results will be available in ~12-24 hours per experiment.

---

## 🎯 Initial Insights

### 1. AdamW is the Winner
**Best configurations**:
- Weight decay 1.0: Perfect grokking, classic dynamics
- Weight decay 2.0: Faster grokking, still perfect
- Weight decay 5.0: FASTEST grokking (epoch 900), high accuracy

### 2. Weight Decay Effect Clear
- **Too low (0.0, 0.01)**: No grokking
- **Optimal (0.5-2.0)**: Perfect grokking
- **High (5.0-10.0)**: Faster but may sacrifice final accuracy

### 3. Optimizer Comparison
- **AdamW**: ⭐⭐⭐⭐⭐ Excellent
- **SGD**: ⭐⭐ Slow but works
- **Muon**: ⚠️ Not working (needs investigation)

---

## 🔍 Monitoring

### Watch Progress Live
```bash
# Monitor dashboard (updates every minute)
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/optimizer_experiments
watch -n 60 './MONITORING_DASHBOARD.sh'
```

### Check Specific Jobs
```bash
# See all jobs
squeue -u $USER

# Watch latest output
tail -f slurm_scripts/logs/nanda_*.out
tail -f slurm_scripts/logs/mnist_*.out

# Check for errors
tail -f slurm_scripts/logs/*.err
```

### Check Results
```bash
./check_status.py --results_dir /om/scratch/Tue/mabdel03/9.520/results/optimizer_experiments
```

---

## 📂 Results Location

All results saved to:
```
/om/scratch/Tue/mabdel03/9.520/results/optimizer_experiments/
├── paper03_nanda/
│   ├── nanda_adamw_wd1.0/     ✅ GROKKED!
│   ├── nanda_adamw_wd2.0/     ✅ GROKKED!
│   ├── nanda_adamw_wd5.0/     ✅ GROKKED!
│   └── ... (24 total)
└── paper05_omnigrok/
    └── ... (18 total, running)
```

Accessible via symlink:
```
/om2/.../optimizer_experiments/results/ → /om/scratch/.../results/optimizer_experiments/
```

---

## 🎨 Next Steps (After All Complete)

### 1. Visualize Results
```bash
cd analysis
python visualize_spectral_metrics.py \
    --results_dir /om/scratch/Tue/mabdel03/9.520/results/optimizer_experiments/paper03_nanda \
    --compare \
    --output_dir plots/nanda
```

### 2. Analyze Grokking Patterns
Compare grokking epochs across optimizers and weight decay values

### 3. Investigate Muon
Why did Muon fail? Needs different learning rate or hyperparameters?

---

## ⚠️ Known Issues

### AGOP Disabled
- Full AGOP matrix needs 204 GB RAM (too large)
- Currently tracking only training metrics (loss/accuracy)
- Can implement streaming eigenvalue computation later if needed
- See: `AGOP_MEMORY_ISSUE.md`

### Muon Performance
- Low accuracy across all weight decays
- May need:
  - Different learning rate (try 10× higher?)
  - Different momentum settings
  - Architecture-specific tuning

---

## 📊 Performance Stats

**Training Speed**:
- Nanda: ~170-180 epochs/second
- Total training time: ~230-240 seconds per experiment (< 4 minutes!)
- Much faster than expected (full batch training is efficient)

**Storage**:
- Current: 8.8 GB in scratch
- Will grow as more experiments complete
- No quota issues (unlimited scratch)

---

## ✨ Summary

**Status**: ✅ **20/24 Nanda experiments complete, 6 show grokking!**

**Best Result**: AdamW with weight decay 1.0-5.0 shows consistent grokking

**Running**: 4 Nanda + 18 MNIST experiments still in progress

**Next**: Wait for MNIST results (12-24 hours), then analyze all data

---

## 🎯 Quick Commands

```bash
# Check status
./check_status.py --results_dir /om/scratch/Tue/mabdel03/9.520/results/optimizer_experiments

# Monitor jobs
squeue -u $USER | grep grok

# Watch live
./MONITORING_DASHBOARD.sh

# View results
ls -lh /om/scratch/Tue/mabdel03/9.520/results/optimizer_experiments/paper03_nanda/
```

---

**Experiments are running successfully! Early results look very promising!** 🎉🚀

*Monitor with*: `./MONITORING_DASHBOARD.sh` or `watch -n 60 './MONITORING_DASHBOARD.sh'`

