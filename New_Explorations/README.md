# Grokking Experiments - New Explorations

Clean, organized experiments for analyzing grokking phenomena with and without AGOP tracking.

---

## Directory Structure

```
New_Explorations/
├── standard_experiments/     # Experiments WITHOUT AGOP tracking
│   ├── datasets/             # Dataset-specific implementations
│   ├── framework/            # Shared training framework
│   └── slurm_scripts/        # Batch job submission
│
├── agop_experiments/         # Experiments WITH AGOP tracking (tractable)
│   ├── core/                 # AGOP implementation (one-hot encoding)
│   ├── training_scripts/     # Training with AGOP metrics
│   ├── analysis/             # Visualization and comparison tools
│   ├── tests/                # Verification tests
│   └── slurm_scripts/        # AGOP batch jobs
│
└── docs/                     # Consolidated documentation
```

---

## Quick Start

### Standard Experiments (No AGOP)

For running experiments from papers without AGOP overhead:

```bash
cd standard_experiments/

# See available datasets
ls datasets/

# Run experiment (see standard_experiments/README.md)
```

### AGOP Experiments (Tractable Input-Gradient Tracking)

For mechanistic analysis with tractable AGOP metrics:

```bash
cd agop_experiments/

# Quick test (verify setup)
python tests/test_onehot_complete.py

# Run single experiment
python training_scripts/train_nanda_agop.py \
    --architecture mlp \
    --optimizer adamw \
    --n_epochs 40000

# Submit batch jobs
cd slurm_scripts/
sbatch run_nanda_agop.sh
```

**See [`agop_experiments/README.md`](agop_experiments/README.md) for complete guide.**

---

## Key Differences

| Feature | Standard Experiments | AGOP Experiments |
|---------|---------------------|------------------|
| **Purpose** | Replicate papers, test optimizers | Mechanistic analysis of grokking |
| **Architecture** | Original (transformers with embeddings) | One-hot encoded (MLP or Transformer) |
| **Inputs** | Discrete tokens | Continuous one-hot vectors |
| **AGOP Tracking** | No (or expensive parameter-gradient) | Yes (tractable input-gradient) |
| **Metrics** | Train/test accuracy, loss | + 19 AGOP metrics (eigengap, VCR, etc.) |
| **Tractability** | N/A | Matrices <5 MB, compute in seconds |

---

## Datasets Available

Both experiment types support:

1. **Nanda** - Modular addition (p=113, ReLU transformer)
2. **Softmax** - Modular addition (p=97, standard transformer)
3. **MNIST** - Image classification (Omnigrok setup)
4. **Composition** - Compositional reasoning (placeholder)

---

## Documentation

- **This file** - Navigation and overview
- [`standard_experiments/README.md`](standard_experiments/README.md) - Standard experiments guide
- [`agop_experiments/README.md`](agop_experiments/README.md) - Complete AGOP guide
- [`docs/`](docs/) - Implementation history, troubleshooting, technical details

---

## Recommended Workflow

### 1. Start with AGOP Experiments (Most Valuable)

The AGOP experiments provide mechanistic insights into grokking:

```bash
cd agop_experiments/
python tests/test_onehot_complete.py  # Verify setup (6/6 tests)
sbatch tests/test_quick_train.sh       # Short training test
sbatch slurm_scripts/run_nanda_agop.sh  # Full experiment
```

### 2. Analyze Results

```bash
cd agop_experiments/analysis/
python visualize_agop_metrics.py --results_dir ../results/...
python compare_grok_nogrok.py --results_dir ../results/...
```

### 3. Compare to Standard Experiments (Optional)

Run baseline experiments without AGOP overhead if needed for comparison.

---

## Research Questions

The AGOP experiments enable investigation of:

1. **When does grokking happen?** (test accuracy transition)
2. **What predicts grokking?** (AGOP metric signatures)
3. **Do architectures matter?** (MLP vs Transformer comparison)
4. **Symbolic vs perceptual?** (Nanda vs MNIST comparison)

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- See `requirements.txt` for full list

**Conda environment:** `/om/scratch/Tue/mabdel03/9.520/conda_envs/grok_exp`

---

## Support

- **Troubleshooting:** See [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)
- **AGOP Guide:** See [`docs/AGOP_GUIDE.md`](docs/AGOP_GUIDE.md)
- **Implementation History:** See [`docs/IMPLEMENTATION_HISTORY.md`](docs/IMPLEMENTATION_HISTORY.md)

---

**Status:** Fully implemented and tested  
**Last Updated:** November 25, 2024
