# Ready to Run Experiments! 🚀

## ✅ Setup Complete

All systems are configured and ready to run the optimizer comparison experiments!

### What's Been Set Up

1. **✅ Conda Environment** - `/om/scratch/Tue/mabdel03/9.520/conda_envs/grok_exp`
   - Python 3.10
   - PyTorch 2.9.1 with CUDA 12.8 support
   - All dependencies installed (numpy, matplotlib, h5py, etc.)

2. **✅ Storage Configuration**
   - Results directory: symlink to `/om/scratch/Tue/mabdel03/9.520/results/optimizer_experiments/`
   - Unlimited storage space for all experiments
   - Git repo stays clean and small

3. **✅ SLURM Scripts Updated**
   - All scripts use scratch conda environment
   - Paths configured correctly
   - Ready to submit

4. **✅ Git Configuration**
   - `.gitignore` updated to exclude large files
   - Repository ready for version control

---

## 🎯 Run All Experiments (Recommended)

### Submit Everything at Once

```bash
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/optimizer_experiments/slurm_scripts

# Create logs directory
mkdir -p logs

# Submit all Nanda experiments (24 jobs)
bash run_all_nanda.sh

# Submit all MNIST experiments (18 jobs)  
bash run_all_mnist.sh

# Check submitted jobs
squeue -u $USER
```

**Total**: 42 experiments will run in parallel!

---

## 🧪 Or Test First (Conservative Approach)

### Run One Test Experiment

```bash
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/optimizer_experiments/slurm_scripts

mkdir -p logs

# Submit one Nanda experiment as a test
sbatch run_nanda_single.sh adamw 1.0

# Monitor the job
squeue -u $USER
watch -n 10 'squeue -u $USER'

# Check output (after job starts)
tail -f logs/nanda_*.out
```

### If Test Succeeds, Submit All

```bash
# Once the test completes successfully
bash run_all_nanda.sh
bash run_all_mnist.sh
```

---

## 📊 Monitor Progress

### Check Job Status

```bash
# View your jobs
squeue -u $USER

# Count running/pending jobs
squeue -u $USER | grep grok | wc -l

# Check specific job details
scontrol show job <JOB_ID>
```

### Check Experiment Results

```bash
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/optimizer_experiments

# Check status of all experiments
./check_status.py --results_dir results

# Check specific paper
./check_status.py --results_dir results --paper paper03_nanda

# List results in scratch
ls -lh /om/scratch/Tue/mabdel03/9.520/results/optimizer_experiments/
```

### Monitor Real-Time Output

```bash
cd slurm_scripts

# Watch latest Nanda output
tail -f logs/nanda_*.out

# Watch latest MNIST output  
tail -f logs/mnist_*.out

# Watch for errors
tail -f logs/*.err
```

---

## 📁 Where Things Are

### Code (Git Repo)
```
/om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/
```

### Conda Environment
```
/om/scratch/Tue/mabdel03/9.520/conda_envs/grok_exp/
```

### Results (Unlimited Storage)
```
/om/scratch/Tue/mabdel03/9.520/results/optimizer_experiments/
  ├── paper03_nanda/
  │   ├── nanda_muonw_wd0.0/
  │   ├── nanda_muonw_wd0.01/
  │   ├── nanda_adamw_wd1.0/
  │   └── ... (24 experiments)
  └── paper05_omnigrok/
      └── ... (18 experiments)
```

### Access via Symlink
```
/om2/.../New_Explorations/optimizer_experiments/results/
  → /om/scratch/Tue/mabdel03/9.520/results/optimizer_experiments/
```

---

## ⏱️ Expected Timeline

### Individual Experiments
- **Nanda (Paper 3)**: 6-12 hours each
- **MNIST (Paper 5)**: 12-24 hours each

### All Experiments (Parallel)
- **Nanda (24 jobs)**: 2-5 days total
- **MNIST (18 jobs)**: 4-7 days total
- **Combined**: ~7-10 days for everything

Jobs run in parallel, so total time depends on cluster availability.

---

## 🎨 After Experiments Complete

### Visualize Results

```bash
cd /om2/.../New_Explorations/optimizer_experiments/analysis

# Visualize single experiment
python visualize_spectral_metrics.py \
    --results_dir ../results/paper03_nanda \
    --experiment nanda_adamw_wd1.0 \
    --output_dir plots

# Compare all experiments
python visualize_spectral_metrics.py \
    --results_dir ../results/paper03_nanda \
    --compare \
    --output_dir plots/comparisons
```

### Analyze AGOP Metrics

```python
import h5py
import matplotlib.pyplot as plt

# Load metrics
with h5py.File('results/paper03_nanda/nanda_adamw_wd1.0/spectral_metrics.h5', 'r') as f:
    epochs = f['epoch'][:]
    eigengap = f['eigengap'][:]
    top_energy = f['top_eigenvalue_energy_ratio'][:]
    trace = f['trace'][:]

# Plot
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(epochs, eigengap)
plt.xlabel('Epoch')
plt.ylabel('Eigengap (λ₁ - λ₂)')
plt.yscale('log')
plt.title('Gradient Alignment')

plt.subplot(2, 2, 2)
plt.plot(epochs, top_energy)
plt.xlabel('Epoch')
plt.ylabel('λ₁/Σλᵢ')
plt.title('Energy in Top Eigenvector')

# ... more plots ...
plt.tight_layout()
plt.show()
```

---

## 🐛 Troubleshooting

### Job Fails to Start
```bash
# Check error log
cat slurm_scripts/logs/nanda_*.err

# Verify conda environment
/om/scratch/Tue/mabdel03/9.520/conda_envs/grok_exp/bin/python --version
```

### Out of Memory
- AGOP uses CPU memory (not GPU)
- Check if `agop_subsample_size` is set in training scripts
- Default: 1000 for Nanda, 500 for MNIST

### Jobs Pending Forever
```bash
# Check partition status
sinfo

# Check your job priority
sprio -u $USER
```

### No Results Appearing
```bash
# Check if symlink is correct
ls -la /om2/.../optimizer_experiments/results

# Check scratch space
ls -la /om/scratch/Tue/mabdel03/9.520/results/optimizer_experiments/
```

---

## 🎯 Quick Command Reference

```bash
# Submit all experiments
cd slurm_scripts && bash run_all_nanda.sh && bash run_all_mnist.sh

# Check status
squeue -u $USER

# Monitor experiments
./check_status.py

# Cancel jobs
scancel <JOB_ID>          # Cancel one
scancel -u $USER          # Cancel all

# Check results
ls -lh /om/scratch/Tue/mabdel03/9.520/results/optimizer_experiments/

# Visualize
cd analysis && python visualize_spectral_metrics.py --compare
```

---

## 📚 Documentation

- **AGOP Details**: `AGOP_IMPLEMENTATION.md`
- **Quick Reference**: `AGOP_QUICK_REFERENCE.md`
- **Full README**: `README.md`
- **Setup Status**: `SETUP_STATUS.md`

---

## ✨ You're All Set!

**Everything is configured and ready to go.**

**Recommended**: Start with `bash run_all_nanda.sh && bash run_all_mnist.sh` to run all 42 experiments in parallel.

The framework will:
- ✅ Track all AGOP metrics (eigengap, trace, energy ratios, etc.)
- ✅ Save results to unlimited scratch space
- ✅ Generate comprehensive logs
- ✅ Create checkpoints every 1000 epochs
- ✅ Allow easy analysis and visualization

**Good luck with your experiments!** 🚀🧠

---

*Setup completed: November 23, 2025*  
*Ready to run: 42 optimizer comparison experiments with AGOP tracking*

