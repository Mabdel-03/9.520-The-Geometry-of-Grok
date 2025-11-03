# GOP Analysis Framework - Usage Guide

This guide shows how to use the Gradient Outer Product Analysis Framework to track grokking dynamics across all 10 paper replications.

## Quick Start

### 1. Install Dependencies

```bash
cd New_Explorations
pip install -r requirements.txt
```

### 2. Run a Simple Experiment (Linear Estimators - Smallest Model)

This is the easiest to start with due to small model size (~1K parameters):

```bash
cd experiments/09_levi_linear
python wrapped_train.py --config ../../configs/09_levi_linear.yaml
```

### 3. Visualize Results

```bash
cd ../../analysis
python visualize_gop.py --results_dir ../results/09_levi_linear
```

## Running on HPC

### Step 1: Choose an Experiment

Start with smaller models first to test the framework:
- **Easiest:** `09_levi_linear` (~1K params, fast)
- **Small:** `08_doshi_polynomials` (~50K params)
- **Medium:** `03_nanda_progress` (~100K params)
- **Large:** `07_thilak_slingshot` (~150K params)

### Step 2: Submit SLURM Job

```bash
cd experiments/09_levi_linear
chmod +x run_gop_analysis.sh

# Edit SLURM parameters for your cluster
nano run_gop_analysis.sh

# Submit
sbatch run_gop_analysis.sh
```

### Step 3: Monitor Progress

```bash
# Check job status
squeue -u $USER

# Watch logs
tail -f logs/gop_*.out

# Check storage usage
du -sh ../../results/09_levi_linear/
```

## Understanding the Framework

### How It Works

1. **GOP Tracker** computes gradients after `loss.backward()`
2. **Before optimizer.step()**, gradients are:
   - Flattened into vector G
   - Outer product formed: G ⊗ G^T
   - Eigenvalues and metrics computed
3. **Storage** saves to HDF5:
   - Full GOP matrices in `gop_full.h5`
   - Per-layer GOPs in `gop_layers.h5`
   - Scalar metrics in `metrics.h5`

### What Gets Tracked

**Every Epoch:**
- Train/test loss and accuracy
- Full GOP matrix (M × M for M parameters)
- Per-layer GOP matrices
- GOP metrics:
  - All eigenvalues
  - Top-k eigenvalues and eigenvectors
  - Trace, Frobenius norm, spectral norm
  - Rank, condition number, determinant

## Storage Requirements

### Estimates Per Epoch

| Experiment | Parameters | Full GOP | Compressed | Per-Layer Total |
|-----------|-----------|----------|------------|-----------------|
| Levi (linear) | 1K | 4 MB | ~400 KB | ~100 KB |
| Doshi (poly) | 50K | 10 GB | ~1 GB | ~500 MB |
| Nanda (transformer) | 100K | 40 GB | ~4 GB | ~1 GB |
| Thilak (slingshot) | 150K | 90 GB | ~9 GB | ~2 GB |

### Total Storage (Full Training)

For 40,000 epochs with compression:
- **Levi:** ~16 GB
- **Doshi:** ~40 TB
- **Nanda:** ~160 TB
- **Thilak:** ~360 TB

**Important:** These are theoretical maxima. In practice:
- Compression reduces by ~10x
- Consider sampling every N epochs for full GOPs
- Metrics only storage is < 1 GB per experiment

## Configuration Options

### Adjusting Storage Frequency

Edit config YAML to track less frequently:

```yaml
gop_tracking:
  frequency: 10  # Track every 10 epochs instead of every epoch
```

### Disable Full GOP Storage

For large models, store only metrics:

```yaml
storage:
  store_full_gop: false  # Only store metrics, not full matrices
```

### Increase Compression

```yaml
storage:
  compression_level: 9  # Maximum compression (slower but smaller)
```

## Creating Wrapped Scripts for Other Experiments

### Template Pattern

All wrapped training scripts follow this pattern:

```python
# 1. Import framework
from framework import TrainingWrapper, HDF5Storage, ExperimentConfig

# 2. Import original model code
from model import YourModel, create_dataset

# 3. Create training function
def train_with_gop_tracking(config_path):
    config = ExperimentConfig(config_path)
    # ... create model, optimizer
    
    storage = HDF5Storage(**config.storage)
    wrapper = TrainingWrapper(model, storage, config.gop_tracking)
    
    for epoch in range(n_epochs):
        # ... forward pass
        loss.backward()
        
        # CRITICAL: Track BEFORE optimizer.step()
        wrapper.track_epoch(epoch, train_loss, test_loss, train_acc, test_acc)
        
        optimizer.step()
```

### For Papers 1, 2, 4, 5, 10 (Cloned Repos)

These require slight adaptation since they have their own training scripts:

1. Copy their original `train.py` to `wrapped_train.py`
2. Add framework imports
3. Insert `wrapper.track_epoch()` call in the training loop
4. Ensure it's called AFTER backward, BEFORE optimizer.step()

## Analysis Tools

### Basic Visualization

```bash
python analysis/visualize_gop.py --results_dir results/03_nanda_progress
```

Creates:
- `plots/training_curves.png`: Loss and accuracy over time
- `plots/gop_metrics.png`: GOP metrics evolution
- `plots/top_eigenvalues.png`: Top eigenvalues over time
- `plots/summary.png`: Comprehensive overview

### Compare Multiple Experiments

```bash
python analysis/compare_experiments.py \
    --experiments results/03_nanda_progress results/07_thilak_slingshot \
    --save_dir comparison_plots/
```

### Custom Analysis

Load HDF5 data directly in Python:

```python
import h5py
import numpy as np

# Load metrics
with h5py.File('results/03_nanda_progress/metrics.h5', 'r') as f:
    test_acc = f['test_acc'][:]
    gop_trace = f['gop_trace'][:]
    eigenvalues_top_k = f['eigenvalues_top_k'][:]

# Load full GOP for specific epoch
with h5py.File('results/03_nanda_progress/gop_full.h5', 'r') as f:
    epoch_10000 = f['epoch_10000']
    gop_matrix = epoch_10000['gop'][:]
    eigenvalues = epoch_10000['eigenvalues'][:]

# Your custom analysis here
...
```

## Troubleshooting

### Out of Memory

**Symptom:** CUDA out of memory during GOP computation

**Solutions:**
1. Reduce model size or use CPU for GOP: `use_gpu: false` in config
2. Use chunked computation (modify `gop_tracker.py`)
3. Compute only per-layer GOPs

### Storage Filling Up

**Symptom:** Disk quota exceeded

**Solutions:**
1. Set `store_full_gop: false` to save only metrics
2. Increase `frequency` to track less often
3. Increase compression level to 9
4. Use sampling strategy (implement custom frequency function)

### Slow Eigenvalue Computation

**Symptom:** Training takes too long per epoch

**Solutions:**
1. Compute only top-k eigenvalues: modify `gop_metrics.py` to use sparse solvers
2. Move eigenvalue computation to separate process
3. Reduce tracking frequency

## Best Practices

1. **Start small:** Test on linear estimators (09) first
2. **Monitor storage:** Check file sizes regularly
3. **Checkpoint often:** Default is every 1000 epochs
4. **Use compression:** gzip level 6 is good balance
5. **Plan storage:** Calculate total requirements before running

## Expected Timeline

| Experiment | Parameters | Training Time | Storage (compressed) |
|-----------|-----------|---------------|---------------------|
| Levi (100K epochs) | 1K | ~6 hours | ~16 GB |
| Doshi (50K epochs) | 50K | ~48 hours | ~50 GB |
| Nanda (40K epochs) | 100K | ~72 hours | ~160 GB |

Times assume 1 GPU, may vary based on hardware.

## Next Steps

After running basic experiments:
1. Compare GOP dynamics across different datasets
2. Analyze eigenvalue evolution during grokking transition
3. Correlate GOP metrics with generalization performance
4. Investigate per-layer contributions to grokking

## Support

For issues:
1. Check logs in `experiments/XX/logs/`
2. Verify config syntax
3. Ensure sufficient storage space
4. Check GPU memory availability

