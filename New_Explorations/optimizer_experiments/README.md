# Optimizer Comparison Experiments for Grokking

This directory contains a comprehensive framework for studying grokking behavior across different **datasets**, **architectures**, **optimizers**, and **weight decay** values, with detailed **spectral metrics** tracking.

## Overview

We systematically investigate grokking using implementations from:
- **Paper 3**: Nanda et al. (2023) - Modular addition with 1-layer ReLU Transformer
- **Paper 4**: Wang et al. (2024) - Compositional reasoning with GPT-2 style models
- **Paper 5**: Liu et al. (2022) - MNIST grokking with MLPs (Omnigrok)

For each paper, we test:
- **3 Optimizers**: Muon, Adam (AdamW), SGD
- **Multiple weight decay values**: 0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0

## Spectral Metrics Tracked

At each epoch (configurable frequency), we compute:

### Core Metrics
1. **Eigengap**: Gap between largest and second-largest eigenvalue (λ₁ - λ₂)
2. **Top-k subspace**: Energy concentration in top-k eigenvectors
3. **Energy in top eigenvector**: λ₁ / Σλᵢ
4. **Spectral radius**: Largest eigenvalue λ_max
5. **Trace**: Sum of all eigenvalues Σλᵢ
6. **Spectral radius to trace ratio**: λ_max / Σλᵢ

### Additional Metrics
- **Effective rank**: Participation ratio of eigenvalues
- **Condition number**: λ_max / λ_min
- **Top-k eigenvalues**: Individual tracking of top 10-20 eigenvalues
- **Frobenius norm**: ||GOP||_F

## Directory Structure

```
optimizer_experiments/
├── README.md                          # This file
├── framework/                         # Core training and metrics code
│   ├── __init__.py
│   ├── spectral_metrics.py           # Spectral metrics computation
│   ├── muon_optimizer.py             # Muon optimizer implementation
│   └── trainer.py                    # Unified training framework
├── paper03_nanda/                     # Nanda et al. experiments
│   └── train_nanda.py
├── paper04_wang/                      # Wang et al. experiments (future)
│   └── train_composition.py
├── paper05_omnigrok/                  # Liu et al. MNIST experiments
│   └── train_mnist.py
├── configs/                           # Configuration files
│   └── weight_decay_sweep.yaml
├── slurm_scripts/                     # SLURM submission scripts
│   ├── run_nanda_single.sh
│   ├── run_mnist_single.sh
│   ├── run_all_nanda.sh
│   └── run_all_mnist.sh
├── analysis/                          # Analysis and visualization
│   └── visualize_spectral_metrics.py
└── results/                           # Experiment results (created at runtime)
    ├── paper03_nanda/
    │   ├── nanda_adamw_wd1.0/
    │   │   ├── config.json
    │   │   ├── training_history.json
    │   │   ├── spectral_metrics.h5
    │   │   └── checkpoints/
    │   └── ...
    └── paper05_omnigrok/
        └── ...
```

## Quick Start

### 1. Installation

```bash
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/optimizer_experiments

# Install dependencies (if not already installed)
pip install torch torchvision numpy matplotlib h5py pyyaml tqdm seaborn
```

### 2. Run a Single Experiment

#### Paper 3: Nanda (Modular Addition)

```bash
cd paper03_nanda
python train_nanda.py \
    --optimizer adamw \
    --weight_decay 1.0 \
    --lr 0.001 \
    --n_epochs 40000 \
    --spectral_metrics \
    --spectral_freq 100
```

#### Paper 5: Omnigrok (MNIST)

```bash
cd paper05_omnigrok
python train_mnist.py \
    --optimizer adamw \
    --weight_decay 0.01 \
    --lr 0.001 \
    --n_epochs 100000 \
    --train_points 1000 \
    --spectral_metrics \
    --spectral_freq 100
```

### 3. Submit SLURM Jobs

#### Single Job

```bash
cd slurm_scripts
sbatch run_nanda_single.sh adamw 1.0
sbatch run_mnist_single.sh adamw 0.01
```

#### All Combinations (Recommended)

```bash
cd slurm_scripts

# Submit all Nanda experiments (3 optimizers × 8 weight decays = 24 jobs)
bash run_all_nanda.sh

# Submit all MNIST experiments (3 optimizers × 6 weight decays = 18 jobs)
bash run_all_mnist.sh
```

Monitor jobs:
```bash
squeue -u $USER
```

### 4. Visualize Results

```bash
cd analysis

# Visualize single experiment
python visualize_spectral_metrics.py \
    --results_dir ../results/paper03_nanda \
    --experiment nanda_adamw_wd1.0 \
    --output_dir ./plots

# Compare all experiments
python visualize_spectral_metrics.py \
    --results_dir ../results/paper03_nanda \
    --compare \
    --output_dir ./plots/comparisons

# Visualize all experiments
python visualize_spectral_metrics.py \
    --results_dir ../results/paper03_nanda \
    --output_dir ./plots
```

## Experiment Details

### Paper 3: Nanda et al. - Modular Addition

**Task**: Learn $(a + b) \mod 113$ with 30% of data

**Architecture**:
- 1-layer ReLU Transformer
- d_model = 128, n_heads = 4, d_mlp = 512
- ~100K parameters

**Training**:
- Full batch gradient descent
- 40,000 epochs
- Default: AdamW with lr=0.001, wd=1.0

**Expected Behavior**:
- Train accuracy reaches ~100% by epoch 1,000
- Test accuracy groks around epochs 10,000-30,000 (varies with optimizer/wd)

### Paper 5: Liu et al. - MNIST Grokking

**Task**: MNIST digit classification with 1,000 training points

**Architecture**:
- 3-layer MLP
- Input: 784, Hidden: 200, Output: 10
- ~160K parameters

**Training**:
- Mini-batch (batch_size=200)
- 100,000 steps
- Default: AdamW with lr=0.001, wd=0.01

**Expected Behavior**:
- Train accuracy reaches ~100% by step 1,000
- Test accuracy groks around steps 20,000-60,000

## Optimizers

### 1. MuonW (Momentum-based Orthogonal Updates)
- Novel optimizer with orthogonalization of weight updates
- Often faster convergence than Adam
- May show different grokking behavior

### 2. AdamW (Adam with Weight Decay)
- Standard choice for grokking experiments
- Decoupled weight decay
- Baseline for comparison

### 3. SGD (Stochastic Gradient Descent)
- With momentum (0.9)
- Often requires higher learning rate
- May grok slower but potentially better generalization

## Weight Decay Values

We test a range of weight decay values to understand regularization's role in grokking:

- **0.0**: No regularization (may not grok)
- **0.001, 0.01**: Light regularization
- **0.1, 0.5**: Medium regularization
- **1.0**: Standard for Nanda et al.
- **2.0, 5.0, 10.0**: Heavy regularization (may slow grokking)

## Output Files

Each experiment creates:

```
results/{paper}/{experiment_name}/
├── config.json                    # Experiment configuration
├── training_history.json          # Train/test loss and accuracy
├── spectral_metrics.h5           # All spectral metrics (HDF5 format)
├── layer_metrics.h5              # Per-layer metrics (if enabled)
└── checkpoints/                   # Model checkpoints
    ├── epoch_1000.pt
    ├── epoch_2000.pt
    └── ...
```

### File Formats

**training_history.json**:
```json
{
  "epoch": [0, 100, 200, ...],
  "train_loss": [2.3, 1.8, 1.2, ...],
  "train_acc": [0.1, 0.3, 0.6, ...],
  "test_loss": [2.4, 2.1, 1.9, ...],
  "test_acc": [0.09, 0.11, 0.12, ...]
}
```

**spectral_metrics.h5** (HDF5 datasets):
- `epoch`: Array of epochs when metrics were computed
- `eigengap`: Eigengap values
- `spectral_radius`: Spectral radius values
- `trace`: Trace values
- `top_eigenvalue_energy_ratio`: Energy in top eigenvector
- `spectral_radius_to_trace_ratio`: Ratio values
- `eigenvalue_1`, `eigenvalue_2`, ..., `eigenvalue_20`: Top eigenvalues
- And more...

## Computational Considerations

### Storage

Approximate storage per experiment:
- **Nanda (100K params)**: ~500 MB (with spectral metrics every 100 epochs)
- **MNIST (160K params)**: ~1 GB
- **Total for all experiments**: ~50-100 GB

### Compute Time

Approximate time per experiment (with spectral metrics):
- **Nanda**: 6-12 hours on 1 GPU
- **MNIST**: 12-24 hours on 1 GPU

### Memory Requirements

- **Nanda**: 8-16 GB GPU memory
- **MNIST**: 16-32 GB GPU memory
- Increase `spectral_freq` if running out of memory

## Analysis Workflow

### 1. Compare Optimizers

```bash
# After running all experiments
python analysis/visualize_spectral_metrics.py \
    --results_dir results/paper03_nanda \
    --compare \
    --output_dir plots/nanda_comparison
```

This creates:
- `comparison_train_acc.png`: Compare training accuracy
- `comparison_test_acc.png`: Compare test accuracy (grokking)
- `comparison_train_loss.png`
- `comparison_test_loss.png`

### 2. Analyze Spectral Evolution

For each experiment, you get:
- `training_curves.png`: Train/test loss and accuracy
- `spectral_metrics.png`: 6-panel plot with eigengap, energy, spectral radius, trace, ratio, effective rank
- `top_eigenvalues.png`: Evolution of top-10 eigenvalues

### 3. Custom Analysis

Load data in Python:

```python
import json
import h5py
import numpy as np

# Load training history
with open('results/paper03_nanda/nanda_adamw_wd1.0/training_history.json') as f:
    history = json.load(f)

# Load spectral metrics
with h5py.File('results/paper03_nanda/nanda_adamw_wd1.0/spectral_metrics.h5', 'r') as f:
    epochs = f['epoch'][:]
    eigengap = f['eigengap'][:]
    spectral_radius = f['spectral_radius'][:]
    trace = f['trace'][:]
    ratio = f['spectral_radius_to_trace_ratio'][:]

# Your analysis here...
```

## Troubleshooting

### Out of Memory

```bash
# Reduce spectral metrics frequency
python train_nanda.py ... --spectral_freq 500

# Disable per-layer metrics
python train_nanda.py ... --no-compute_per_layer
```

### Slow Training

```bash
# Reduce number of epochs
python train_nanda.py ... --n_epochs 20000

# Increase log frequency
python train_nanda.py ... --log_freq 500
```

### Jobs Failing

```bash
# Check logs
cat slurm_scripts/logs/nanda_*.err

# Reduce time limit or request more memory in SLURM scripts
```

## Research Questions to Investigate

1. **Does Muon grok faster than AdamW?**
   - Compare test accuracy curves
   - Measure epochs to 90% test accuracy

2. **How does weight decay affect grokking time?**
   - Plot grokking epoch vs weight decay
   - Identify optimal weight decay for each optimizer

3. **Are spectral metrics predictive of grokking?**
   - Correlate eigengap/ratio changes with test accuracy jumps
   - Identify early indicators of grokking

4. **Do different datasets show different spectral signatures?**
   - Compare Nanda vs MNIST spectral evolution
   - Look for universal patterns

5. **What is the relationship between eigengap and generalization?**
   - Plot eigengap vs test accuracy
   - Test hypothesis: larger eigengap → better generalization

## References

- **Paper 3**: Nanda et al. (2023). Progress Measures for Grokking via Mechanistic Interpretability. [arXiv:2301.05217](https://arxiv.org/abs/2301.05217)
- **Paper 4**: Wang et al. (2024). Grokked Transformers are Implicit Reasoners. [arXiv:2405.15071](https://arxiv.org/abs/2405.15071)
- **Paper 5**: Liu et al. (2022). Omnigrok: Grokking Beyond Algorithmic Data. [arXiv:2210.01117](https://arxiv.org/abs/2210.01117)

## Contributing

To add a new optimizer:
1. Add optimizer to `framework/trainer.py` in `_create_optimizer()`
2. Update `configs/weight_decay_sweep.yaml`
3. Create new SLURM script if needed

To add a new dataset/paper:
1. Create new directory `paper{XX}_{name}/`
2. Implement `train_{name}.py` using `GrokkingTrainer`
3. Create corresponding SLURM scripts
4. Update this README

## Contact

For questions or issues, please refer to the main project repository.
