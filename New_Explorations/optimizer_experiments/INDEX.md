# Optimizer Comparison Framework - Complete Index

## 📁 Directory Structure

```
optimizer_experiments/
│
├── 📘 Documentation (Start Here!)
│   ├── README.md                    # Comprehensive guide (15KB)
│   ├── QUICK_START.md               # Step-by-step tutorial
│   ├── EXPERIMENT_SUMMARY.md        # High-level overview
│   ├── SETUP_COMPLETE.md            # Setup verification
│   └── INDEX.md                     # This file
│
├── 🔧 Core Framework
│   └── framework/
│       ├── __init__.py              # Module exports
│       ├── spectral_metrics.py      # ALL requested metrics computation
│       ├── muon_optimizer.py        # Muon & MuonW optimizer
│       └── trainer.py               # Unified training framework
│
├── 🧪 Experiments
│   ├── paper03_nanda/               # Nanda et al. - Modular Addition
│   │   └── train_nanda.py           # Full implementation
│   │
│   ├── paper05_omnigrok/            # Liu et al. - MNIST Grokking
│   │   └── train_mnist.py           # Full implementation
│   │
│   └── paper04_wang/                # Wang et al. - Compositional Reasoning
│       ├── README_PAPER4.md         # Status and options
│       └── train_composition_placeholder.py  # Placeholder
│
├── ⚙️ Configuration
│   └── configs/
│       └── weight_decay_sweep.yaml  # Experiment configurations
│
├── 🖥️ SLURM Scripts
│   └── slurm_scripts/
│       ├── run_nanda_single.sh      # Single Nanda job
│       ├── run_mnist_single.sh      # Single MNIST job
│       ├── run_all_nanda.sh         # All 24 Nanda experiments
│       └── run_all_mnist.sh         # All 18 MNIST experiments
│
├── 📊 Analysis
│   └── analysis/
│       └── visualize_spectral_metrics.py  # Comprehensive visualization
│
├── 🛠️ Utilities
│   ├── check_status.py              # Monitor experiment progress
│   └── requirements.txt             # Python dependencies
│
└── 📂 Results (Created at Runtime)
    └── results/
        ├── paper03_nanda/
        │   ├── nanda_muonw_wd0.0/
        │   ├── nanda_muonw_wd0.01/
        │   ├── nanda_adamw_wd1.0/
        │   └── ... (24 total)
        │
        └── paper05_omnigrok/
            ├── mnist_muonw_wd0.0/
            ├── mnist_adamw_wd0.01/
            └── ... (18 total)
```

## 📚 Quick Reference

### Getting Started (5 minutes)
1. Read: `QUICK_START.md`
2. Install: `pip install -r requirements.txt`
3. Test: `cd paper03_nanda && python train_nanda.py --n_epochs 1000`

### Running Experiments (1 minute to submit)
```bash
cd slurm_scripts
bash run_all_nanda.sh    # 24 jobs
bash run_all_mnist.sh    # 18 jobs
```

### Monitoring Progress (anytime)
```bash
./check_status.py        # See completion status
squeue -u $USER          # See SLURM queue
```

### Analyzing Results (after completion)
```bash
cd analysis
python visualize_spectral_metrics.py --results_dir ../results/paper03_nanda --compare
```

## 🎯 What Each File Does

### Documentation

| File | Purpose | Read When |
|------|---------|-----------|
| `README.md` | Complete reference manual | Need details |
| `QUICK_START.md` | Step-by-step guide | First time |
| `EXPERIMENT_SUMMARY.md` | High-level overview | Want big picture |
| `SETUP_COMPLETE.md` | Verify setup | Checking status |
| `INDEX.md` | This file | Finding things |

### Framework Code

| File | Purpose | Edit When |
|------|---------|-----------|
| `framework/spectral_metrics.py` | Computes eigengap, trace, ratio, etc. | Adding metrics |
| `framework/muon_optimizer.py` | Muon optimizer implementation | Modifying Muon |
| `framework/trainer.py` | Main training loop + metrics | Changing training |
| `framework/__init__.py` | Module exports | Adding modules |

### Experiment Scripts

| File | Purpose | Use For |
|------|---------|---------|
| `paper03_nanda/train_nanda.py` | Modular addition experiments | Algorithmic tasks |
| `paper05_omnigrok/train_mnist.py` | MNIST grokking experiments | Visual tasks |
| `paper04_wang/train_composition_placeholder.py` | Placeholder | Future work |

### SLURM Scripts

| File | Purpose | Run When |
|------|---------|----------|
| `slurm_scripts/run_nanda_single.sh` | Single Nanda job | Testing |
| `slurm_scripts/run_mnist_single.sh` | Single MNIST job | Testing |
| `slurm_scripts/run_all_nanda.sh` | All Nanda combinations | Full experiment |
| `slurm_scripts/run_all_mnist.sh` | All MNIST combinations | Full experiment |

### Analysis Tools

| File | Purpose | Use For |
|------|---------|---------|
| `analysis/visualize_spectral_metrics.py` | Create all plots | Visualization |
| `check_status.py` | Monitor experiments | Tracking progress |

## 🔍 Find What You Need

### "I want to..."

**...start quickly**
→ Read `QUICK_START.md`

**...understand everything**
→ Read `README.md`

**...run experiments**
→ Use `slurm_scripts/run_all_*.sh`

**...check progress**
→ Run `./check_status.py`

**...analyze results**
→ Use `analysis/visualize_spectral_metrics.py`

**...modify metrics**
→ Edit `framework/spectral_metrics.py`

**...add optimizer**
→ Edit `framework/trainer.py`

**...change weight decay values**
→ Edit `configs/weight_decay_sweep.yaml`

**...understand Paper 4 status**
→ Read `paper04_wang/README_PAPER4.md`

## 📊 Metrics Tracked

### Training Metrics (Standard)
- Train/Test Loss
- Train/Test Accuracy
- Learning Rate

### Spectral Metrics (Novel)
1. **Eigengap**: λ₁ - λ₂
2. **Top-k Energy**: Energy in top-k eigenvectors
3. **Top Eigenvector Energy**: λ₁ / Σλᵢ
4. **Spectral Radius**: λ_max
5. **Trace**: Σλᵢ
6. **Spectral Radius to Trace Ratio**: λ_max / Σλᵢ
7. **Effective Rank**: Participation ratio
8. **Condition Number**: λ_max / λ_min
9. **Top-20 Eigenvalues**: Individual tracking

All metrics saved in HDF5 format for efficient storage.

## 🧪 Experiment Matrix

### Paper 3: Nanda (24 experiments)
- Optimizers: MuonW, AdamW, SGD
- Weight Decay: 0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0
- Architecture: 1-layer ReLU Transformer
- Task: Modular addition (a+b) mod 113

### Paper 5: MNIST (18 experiments)
- Optimizers: MuonW, AdamW, SGD
- Weight Decay: 0.0, 0.001, 0.01, 0.1, 0.5, 1.0
- Architecture: 3-layer MLP
- Task: MNIST digit classification (1000 samples)

**Total: 42 comprehensive experiments**

## 🕐 Expected Timelines

| Task | Time |
|------|------|
| Install dependencies | 5 min |
| Quick test run | 10 min |
| Single Nanda experiment | 6-12 hours |
| Single MNIST experiment | 12-24 hours |
| All Nanda experiments | 2-5 days (parallel) |
| All MNIST experiments | 4-7 days (parallel) |
| Analysis and visualization | 1-2 hours |

## 💾 Storage Requirements

| Component | Size |
|-----------|------|
| Framework code | ~100 KB |
| Single Nanda result | ~500 MB |
| Single MNIST result | ~1 GB |
| All Nanda results | ~12 GB |
| All MNIST results | ~18 GB |
| **Total** | **~30 GB** |

## 🎓 Research Workflow

1. **Setup** (Day 0)
   - Install: `pip install -r requirements.txt`
   - Test: Run quick test experiment
   - Verify: Check results

2. **Submit** (Day 1)
   - Run: `bash slurm_scripts/run_all_nanda.sh`
   - Run: `bash slurm_scripts/run_all_mnist.sh`
   - Monitor: `./check_status.py`

3. **Monitor** (Days 2-7)
   - Check: `squeue -u $USER`
   - Track: `./check_status.py`
   - Review: Early results as they complete

4. **Analyze** (Days 8-10)
   - Visualize: Run visualization scripts
   - Compare: Cross-experiment analysis
   - Investigate: Custom analysis in Python

5. **Report** (Day 10+)
   - Document findings
   - Create figures
   - Write conclusions

## 🚀 Quick Commands

```bash
# Navigate
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/optimizer_experiments

# Install
pip install -r requirements.txt

# Test
cd paper03_nanda && python train_nanda.py --n_epochs 1000

# Submit all
cd slurm_scripts && bash run_all_nanda.sh && bash run_all_mnist.sh

# Check status
./check_status.py

# Analyze
cd analysis && python visualize_spectral_metrics.py --results_dir ../results/paper03_nanda --compare
```

## 📞 Need Help?

1. **Getting Started**: Read `QUICK_START.md`
2. **Understanding Code**: Check docstrings in Python files
3. **Troubleshooting**: See "Troubleshooting" section in `README.md`
4. **Examples**: Look at `/Replications` for reference
5. **Configuration**: Check `configs/weight_decay_sweep.yaml`

## ✅ Checklist

Before running experiments:
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Quick test run successful
- [ ] SLURM scripts executable (`chmod +x slurm_scripts/*.sh`)
- [ ] Results directory writable
- [ ] Sufficient disk space (~30 GB)

Before analyzing:
- [ ] At least one experiment completed
- [ ] Results files exist (check with `./check_status.py`)
- [ ] Visualization dependencies installed

## 🎉 You're Ready!

Everything is set up and documented. To begin:

```bash
# Start here
cat QUICK_START.md
```

**Happy Grokking!** 🧠✨

---

*Complete framework with 42 experiments ready to run*  
*All requested metrics implemented and tested*  
*Full documentation and analysis tools included*

