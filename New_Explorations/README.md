# Gradient Outer Product Analysis Framework

This directory contains a unified framework for analyzing gradient outer products (GOPs) across all 10 grokking paper replications.

## Overview

The framework instruments existing training code from `Replications/` to compute and store:
- **Train/test loss and accuracy** at every epoch
- **Full gradient outer product matrices** (G ⊗ G^T)
- **Per-layer GOP matrices** for each model layer
- **GOP metrics**: eigenvalues, trace, norms, rank, condition number
- **Storage**: All data stored in compressed HDF5 format

## Quick Start

### 1. Install Dependencies

```bash
cd New_Explorations
pip install -r requirements.txt
```

### 2. Run an Experiment

```bash
# Example: Run GOP analysis on Nanda et al. (smallest model)
cd experiments/03_nanda_progress
python wrapped_train.py --config ../../configs/03_nanda_progress.yaml
```

### 3. Analyze Results

```bash
cd analysis
python visualize_gop.py --experiment 03_nanda_progress
```

## Directory Structure

```
New_Explorations/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── framework/                   # Core GOP tracking modules
│   ├── gop_tracker.py          # Gradient and GOP computation
│   ├── gop_metrics.py          # Eigenvalue and metric calculations
│   ├── storage.py              # HDF5 storage management
│   ├── wrapper.py              # Training loop instrumentation
│   └── config.py               # Configuration management
├── configs/                     # Per-experiment YAML configs
│   ├── 01_power_grok.yaml
│   ├── 02_liu_effective.yaml
│   └── ... (10 configs total)
├── experiments/                 # Wrapped training scripts
│   ├── 01_power_et_al/
│   │   ├── wrapped_train.py
│   │   └── run_gop_analysis.sh
│   └── ... (10 experiment dirs)
├── analysis/                    # Analysis and visualization tools
│   ├── visualize_gop.py
│   ├── compare_experiments.py
│   └── detect_grokking.py
└── results/                     # Output data (created at runtime)
    ├── 01_power_grok/
    │   ├── metrics.h5          # Scalar time series
    │   ├── gop_full.h5         # Full GOP matrices
    │   ├── gop_layers.h5       # Per-layer GOPs
    │   └── config.yaml         # Config snapshot
    └── ... (results per experiment)
```

## Framework Components

### GOP Tracker (`framework/gop_tracker.py`)

Computes gradient outer products:
- Full model gradient: `G = flatten([grad(p) for p in model.parameters()])`
- Outer product: `GOP = G ⊗ G^T` (M × M matrix for M parameters)
- Per-layer: Separate GOP for each layer

### GOP Metrics (`framework/gop_metrics.py`)

Computes comprehensive metrics:
- Full eigenvalue spectrum
- Top-k eigenvalues and eigenvectors
- Trace, Frobenius norm, spectral norm
- Rank, condition number, determinant

### HDF5 Storage (`framework/storage.py`)

Efficient compressed storage:
- `metrics.h5`: Scalars (loss, accuracy, GOP metrics) vs epoch
- `gop_full.h5`: Full GOP matrices per epoch with compression
- `gop_layers.h5`: Per-layer GOP matrices

### Training Wrapper (`framework/wrapper.py`)

Minimal instrumentation of existing training code:
- Hooks into training loop after backward pass
- Computes GOP before optimizer step
- Saves data incrementally to avoid memory issues

## Computational Considerations

### Storage Requirements

| Model       | Parameters | GOP Size/Epoch | Compressed | Total (40K epochs) |
|-------------|-----------|----------------|------------|-------------------|
| Nanda       | ~100K     | 40 GB          | ~4 GB      | 160 TB            |
| Power       | ~500K     | 1 TB           | ~100 GB    | 4 PB              |
| Wang (large)| ~85M      | 28 PB          | -          | Infeasible        |

**Notes:**
- For large models, store only metrics + top-k eigenvalues
- Use aggressive compression (gzip level 6-9)
- Per-layer GOPs are more storage-friendly
- Consider sampling frequency for full GOPs

### Compute Time

- GOP formation: O(M²) for M parameters
- Eigenvalue decomposition: O(M³)
- Estimate: ~10-1000 seconds per epoch depending on model size

## Running on HPC

Each experiment has a SLURM script:

```bash
cd experiments/03_nanda_progress
sbatch run_gop_analysis.sh
```

Adjust SLURM parameters for your cluster:
- Memory: 64-256 GB depending on model size
- Time: 24-72 hours
- GPU: 1 GPU recommended

## Configuration

Each experiment has a YAML config specifying:
- Model location and architecture
- Training hyperparameters
- GOP tracking settings (frequency, compression, storage options)
- Computational limits

Example config:
```yaml
experiment:
  name: "nanda_progress_modular_addition"
  
gop_tracking:
  enabled: true
  compute_full: true
  compute_per_layer: true
  frequency: 1  # every epoch
  
storage:
  compression: "gzip"
  compression_level: 6
  store_full_gop: true
```

## Analysis Tools

### Visualize GOP Evolution

```bash
python analysis/visualize_gop.py --experiment 03_nanda_progress
```

Creates plots:
- Eigenvalue spectrum evolution over epochs
- GOP metrics (trace, norms) vs epoch
- Correlation with train/test loss

### Compare Experiments

```bash
python analysis/compare_experiments.py --experiments 03_nanda_progress 07_thilak_slingshot
```

Overlays GOP metrics from multiple experiments.

### Detect Grokking

```bash
python analysis/detect_grokking.py --experiment 03_nanda_progress
```

Automatically identifies grokking transitions using GOP metrics.

## Safety Features

1. **Storage monitoring**: Warns if disk usage exceeds threshold
2. **Checkpointing**: Saves intermediate results every N epochs
3. **Graceful degradation**: Falls back to metrics-only if GOP too large
4. **Dry-run mode**: Estimates storage/compute before running
5. **Configuration validation**: Checks feasibility before execution

## Validation

Before running full experiments, validate on small scale:

```bash
# Run for 100 epochs on smallest model
python experiments/03_nanda_progress/wrapped_train.py \
    --config configs/03_nanda_progress.yaml \
    --epochs 100
```

## Expected Outputs

After training completes, each experiment produces:
- `results/{experiment}/metrics.h5`: Time series of all scalars
- `results/{experiment}/gop_full.h5`: Full GOP matrices (~TBs)
- `results/{experiment}/gop_layers.h5`: Per-layer GOPs (~GBs)
- Plots in `results/{experiment}/plots/`

## Troubleshooting

**Out of memory during GOP computation:**
- Reduce batch size
- Enable per-layer only mode
- Increase storage frequency (e.g., every 10 epochs)

**Storage filling up:**
- Increase compression level
- Store only metrics + top-k eigenvalues
- Use sampling strategy

**Slow eigenvalue decomposition:**
- Use only top-k eigenvalues (`top_k_eigen: 100`)
- Move eigenvalue computation to CPU
- Enable parallel eigenvalue solver

## Citation

If you use this framework, please cite the original grokking papers (see `../Replications/README.md`) and this framework.

## Contact

For issues or questions about the GOP analysis framework, see the main project repository.

