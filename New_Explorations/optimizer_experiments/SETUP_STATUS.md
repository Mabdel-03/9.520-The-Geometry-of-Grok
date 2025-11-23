# Setup Status - Optimizer Experiments

## ✅ Completed Steps

### 1. Scratch Space Setup
**Location**: `/om/scratch/Tue/mabdel03/9.520/`

**Structure Created**:
```
/om/scratch/Tue/mabdel03/9.520/
├── conda_envs/
│   └── grok_exp/          # Python 3.10 environment (installing...)
├── results/
│   └── optimizer_experiments/  # All experiment results go here
├── checkpoints/           # Large model checkpoints
└── replications_data/     # For moving large replication files
```

### 2. Conda Environment
**Path**: `/om/scratch/Tue/mabdel03/9.520/conda_envs/grok_exp`
**Status**: ⏳ Installing packages (torch, torchvision, numpy, matplotlib, h5py, pyyaml, tqdm, seaborn, pandas)
**Progress**: Downloading CUDA libraries (~3 GB total)

### 3. Symlinks Created
**Results directory**:
```
/om2/.../New_Explorations/optimizer_experiments/results 
  -> /om/scratch/Tue/mabdel03/9.520/results/optimizer_experiments
```

All experiment results will be stored in unlimited scratch space!

### 4. SLURM Scripts Updated
✅ `run_nanda_single.sh` - Updated to use scratch conda environment  
✅ `run_mnist_single.sh` - Updated to use scratch conda environment

**Changes**:
- Conda environment path: `/om/scratch/Tue/mabdel03/9.520/conda_envs/grok_exp`
- Python calls use full path to ensure correct environment
- Results automatically go to scratch via symlink

### 5. Git Configuration
✅ `.gitignore` updated to exclude:
- `New_Explorations/optimizer_experiments/results/`
- `Replications/**/output_dir/**/checkpoint-*/`
- `Replications/**/cache_dir/`
- `Replications/**/runs/`

Repository stays clean and small!

## 📊 Current Status

### Repository Location
**Code**: `/om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/`
- Git repository: ✅ Connected to GitHub
- Size: ~31 GB (will clean up after moving large files)
- Remote: `https://github.com/Mabdel-03/9.520-The-Geometry-of-Grok.git`

### Storage Distribution
| Location | Purpose | Size | Quota |
|----------|---------|------|-------|
| `/om2/user/mabdel03/` | Code + git repo | ~31 GB → will reduce | Limited |
| `/om/scratch/Tue/mabdel03/9.520/` | Data + results + conda | Will grow to ~50+ GB | ✅ Unlimited |

## 🔄 Remaining Tasks

### High Priority
1. ⏳ **Complete package installation** (~5-10 min remaining)
   - Currently downloading CUDA libraries
   - Will verify: `python -c "import torch; print(torch.cuda.is_available())"`

2. ⏹ **Move large replication data to scratch**
   - `Replications/04_wang.../output_dir/` (7.7 GB)
   - Other large checkpoint directories
   - Will use rsync to preserve data

3. ⏹ **Test one experiment**
   - Run quick test with Nanda (5000 epochs)
   - Verify AGOP computation works
   - Check results save to scratch

### Medium Priority
4. ⏹ **Submit all experiments**
   - Run `bash run_all_nanda.sh` (24 jobs)
   - Run `bash run_all_mnist.sh` (18 jobs)
   - Total: 42 experiments

5. ⏹ **Clean up /om2 space**
   - Remove duplicate checkpoint directories
   - Verify symlinks work
   - Push .gitignore updates to GitHub

## 🎯 Experiment Configuration

### Experiments Ready to Run

**Paper 3 (Nanda) - 24 experiments**:
- Optimizers: MuonW, AdamW, SGD
- Weight decay: 0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0
- AGOP tracking: Every 100 epochs
- Time: ~6-12 hours each

**Paper 5 (MNIST) - 18 experiments**:
- Optimizers: MuonW, AdamW, SGD
- Weight decay: 0.0, 0.001, 0.01, 0.1, 0.5, 1.0
- AGOP tracking: Every 500 steps
- Time: ~12-24 hours each

### AGOP Settings
- **Subsample size**: 1000 (Nanda), 500 (MNIST)
- **Top-k eigenvalues**: 20
- **Metrics computed**:
  - Eigengap (λ₁ - λ₂)
  - Top eigenvalue energy (λ₁/Σλᵢ)
  - Trace (E[||∇L||²])
  - Spectral radius to trace ratio
  - Effective rank
  - And more...

## 📝 How to Use

### Activate Environment
```bash
# Use full path (no conda activate needed)
/om/scratch/Tue/mabdel03/9.520/conda_envs/grok_exp/bin/python

# Or add to PATH
export PATH=/om/scratch/Tue/mabdel03/9.520/conda_envs/grok_exp/bin:$PATH
python your_script.py
```

### Submit Experiments
```bash
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/optimizer_experiments/slurm_scripts

# Single experiment
sbatch run_nanda_single.sh adamw 1.0

# All experiments
bash run_all_nanda.sh
bash run_all_mnist.sh
```

### Check Results
```bash
# Results are in scratch space
ls -lh /om/scratch/Tue/mabdel03/9.520/results/optimizer_experiments/

# But accessible via symlink
cd /om2/.../New_Explorations/optimizer_experiments
ls -lh results/  # <- Actually in scratch!
```

### Monitor Progress
```bash
cd /om2/.../New_Explorations/optimizer_experiments
./check_status.py --results_dir results
```

## ✨ Benefits of This Setup

1. ✅ **Unlimited storage** for results
2. ✅ **Git repo stays clean** (<500 MB after cleanup)
3. ✅ **Normal git operations** (push/pull/commit)
4. ✅ **Transparent access** via symlinks
5. ✅ **No quota issues** for large AGOP matrices
6. ✅ **Easy to use** - just run experiments normally

## 🔍 Next Steps

Once package installation completes (~5-10 min):
1. Verify installation
2. Run test experiment
3. Submit all experiments
4. Monitor progress
5. Analyze results!

---

**Setup Date**: November 23, 2025  
**Status**: 🟡 In Progress (95% complete)  
**ETA to Ready**: ~10 minutes

