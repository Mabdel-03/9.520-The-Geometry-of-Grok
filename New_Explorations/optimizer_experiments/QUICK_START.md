# Quick Start Guide

## Installation (One-Time Setup)

```bash
# Navigate to the directory
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/optimizer_experiments

# Install dependencies
pip install -r requirements.txt

# Make scripts executable (already done, but just in case)
chmod +x check_status.py
chmod +x slurm_scripts/*.sh
```

## Running Experiments

### Option 1: Test Run (Single Experiment)

Test the framework with a quick run:

```bash
# Test Nanda experiment (small version)
cd paper03_nanda
python train_nanda.py \
    --optimizer adamw \
    --weight_decay 1.0 \
    --n_epochs 5000 \
    --spectral_freq 100 \
    --experiment_name test_run

# Check results
ls -lh ../results/paper03_nanda/test_run/
```

### Option 2: Single Full Experiment via SLURM

```bash
cd slurm_scripts

# Create logs directory
mkdir -p logs

# Submit a single experiment
sbatch run_nanda_single.sh adamw 1.0

# Check job status
squeue -u $USER

# Monitor output
tail -f logs/nanda_*.out
```

### Option 3: Full Experiment Suite (Recommended)

```bash
cd slurm_scripts

# Submit all Nanda experiments (24 jobs)
# Tests 3 optimizers × 8 weight decay values
bash run_all_nanda.sh

# Submit all MNIST experiments (18 jobs)
# Tests 3 optimizers × 6 weight decay values
bash run_all_mnist.sh

# Check all job statuses
squeue -u $USER | grep grok
```

## Monitoring Progress

### Check Experiment Status

```bash
# Check status of all experiments
./check_status.py --results_dir results

# Check specific paper
./check_status.py --results_dir results --paper paper03_nanda
```

### Monitor SLURM Jobs

```bash
# View queue
squeue -u $USER

# View detailed job info
scontrol show job <JOB_ID>

# Cancel a job
scancel <JOB_ID>

# Cancel all your jobs
scancel -u $USER
```

### Monitor Logs

```bash
# Watch output in real-time
tail -f slurm_scripts/logs/nanda_*.out

# Check errors
tail -f slurm_scripts/logs/nanda_*.err

# View all recent logs
ls -lt slurm_scripts/logs/ | head -20
```

## Analyzing Results

### Quick Visualization

```bash
# Visualize single experiment
python analysis/visualize_spectral_metrics.py \
    --results_dir results/paper03_nanda \
    --experiment nanda_adamw_wd1.0 \
    --output_dir plots

# Compare all Nanda experiments
python analysis/visualize_spectral_metrics.py \
    --results_dir results/paper03_nanda \
    --compare \
    --output_dir plots/nanda_comparison
```

### Custom Analysis in Python

```python
import json
import h5py
import matplotlib.pyplot as plt

# Load experiment
exp_dir = 'results/paper03_nanda/nanda_adamw_wd1.0'

# Load training history
with open(f'{exp_dir}/training_history.json') as f:
    history = json.load(f)

# Load spectral metrics
with h5py.File(f'{exp_dir}/spectral_metrics.h5', 'r') as f:
    epochs = f['epoch'][:]
    eigengap = f['eigengap'][:]
    ratio = f['spectral_radius_to_trace_ratio'][:]

# Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history['epoch'], history['test_acc'])
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.title('Grokking Curve')

plt.subplot(1, 2, 2)
plt.plot(epochs, eigengap)
plt.xlabel('Epoch')
plt.ylabel('Eigengap')
plt.title('Eigengap Evolution')
plt.yscale('log')

plt.tight_layout()
plt.savefig('my_analysis.png')
```

## Expected Timelines

### Nanda (Modular Addition)
- **Training time**: 6-12 hours per experiment
- **Grokking**: Usually occurs between epochs 10,000-30,000
- **Total for all experiments**: 2-5 days (with parallel jobs)

### MNIST (Omnigrok)
- **Training time**: 12-24 hours per experiment
- **Grokking**: Usually occurs between steps 20,000-60,000
- **Total for all experiments**: 4-7 days (with parallel jobs)

## Typical Workflow

1. **Submit Jobs** (Day 1)
   ```bash
   cd slurm_scripts
   bash run_all_nanda.sh
   bash run_all_mnist.sh
   ```

2. **Monitor** (Days 1-7)
   ```bash
   # Check daily
   ./check_status.py --results_dir results
   squeue -u $USER
   ```

3. **Analyze** (After completion)
   ```bash
   # Generate all plots
   python analysis/visualize_spectral_metrics.py \
       --results_dir results/paper03_nanda \
       --output_dir plots/nanda
   
   python analysis/visualize_spectral_metrics.py \
       --results_dir results/paper05_omnigrok \
       --output_dir plots/mnist
   
   # Create comparisons
   python analysis/visualize_spectral_metrics.py \
       --results_dir results/paper03_nanda \
       --compare \
       --output_dir plots/comparisons
   ```

4. **Investigate** (Custom analysis)
   - Load HDF5 files with h5py
   - Create custom plots
   - Test hypotheses about grokking

## Troubleshooting

### "Out of Memory"
```bash
# Reduce spectral metrics frequency
python train_nanda.py ... --spectral_freq 500

# Or reduce batch size (for MNIST)
python train_mnist.py ... --batch_size 100
```

### "Job Failed"
```bash
# Check error log
cat slurm_scripts/logs/nanda_*_<JOB_ID>.err

# Common issues:
# 1. CUDA out of memory → reduce batch size or spectral_freq
# 2. Module not found → activate correct conda environment
# 3. File not found → check paths in SLURM scripts
```

### "No Grokking Observed"
```bash
# Run longer
python train_nanda.py ... --n_epochs 60000

# Or try different weight decay
python train_nanda.py ... --weight_decay 0.5
```

### "Results Not Appearing"
```bash
# Check save directory
ls -lh results/paper03_nanda/

# Check permissions
chmod -R u+w results/

# Verify paths in scripts
head -20 paper03_nanda/train_nanda.py
```

## Next Steps

After completing the experiments:

1. **Compare optimizers**: Which optimizer groks fastest?
2. **Analyze weight decay**: What's the optimal weight decay for each optimizer?
3. **Study spectral metrics**: Do eigengap/ratio predict grokking?
4. **Cross-dataset comparison**: Do Nanda and MNIST show similar patterns?
5. **Write up results**: Document findings in a report or paper

## Need Help?

- **README.md**: Full documentation
- **framework/*.py**: Code documentation
- **Check existing replications**: See `/Replications` for reference implementations

Happy experimenting! 🧪

