# Standard Grokking Experiments

Original paper implementations WITHOUT AGOP tracking overhead.

---

## Overview

This directory contains standard grokking experiments from various papers, using the original model architectures with discrete token inputs and embeddings.

**Use this for:**
- Replicating paper results
- Comparing optimizers without analysis overhead
- Baseline experiments

**For mechanistic analysis, use [`../agop_experiments/`](../agop_experiments/) instead.**

---

## Datasets

### 1. Nanda ([`datasets/nanda/`](datasets/nanda/))
- **Paper:** Nanda et al. (2023) - Progress Measures for Grokking
- **Task:** Modular addition (a + b) mod 113
- **Architecture:** 1-layer ReLU Transformer
- **Key feature:** No LayerNorm, ReLU attention

### 2. Softmax ([`datasets/softmax/`](datasets/softmax/))
- **Task:** Modular addition (a + b) mod 97
- **Architecture:** Standard Softmax Transformer
- **Key feature:** LayerNorm, softmax attention (Muon-compatible)

### 3. MNIST ([`datasets/mnist/`](datasets/mnist/))
- **Paper:** Liu et al. (2022) - Omnigrok
- **Task:** Image classification with limited data
- **Architecture:** 3-layer MLP
- **Key feature:** MSE loss with one-hot targets

### 4. Composition ([`datasets/composition/`](datasets/composition/))
- **Paper:** Wang et al. (2024) - Implicit Reasoners
- **Task:** Compositional reasoning (A→B, B→C ⇒ A→C)
- **Architecture:** Transformer
- **Status:** Placeholder implementation

---

## Quick Start

```bash
cd standard_experiments/

# Run Nanda experiment
cd datasets/nanda/
python train_nanda.py --optimizer adamw --weight_decay 1.0 --n_epochs 40000

# Run MNIST experiment  
cd datasets/mnist/
python train_mnist.py --optimizer adamw --train_points 1000 --n_epochs 50000
```

---

## Framework

Shared training framework in [`framework/`](framework/):
- `trainer.py` - Base training loop
- `muon_official.py` - Muon optimizer
- `spectral_metrics.py` - Optional parameter-gradient metrics (expensive)

---

## SLURM Batch Jobs

Submit multiple experiments:

```bash
cd slurm_scripts/

# Run all Nanda experiments (optimizer × weight_decay sweep)
sbatch run_all_nanda.sh

# Run single experiment
sbatch run_nanda_single.sh adamw 1.0
```

---

## Conda Environment

All scripts use:
```
/om/scratch/Tue/mabdel03/9.520/conda_envs/grok_exp
```

---

## Results

Experiment outputs saved to: [`results/`](results/)

Structure:
```
results/
└── paper03_nanda/
    └── nanda_adamw_wd1.0/
        ├── config.json
        ├── training_history.json
        └── checkpoints/
```

---

## When to Use This vs AGOP Experiments

**Use standard_experiments when:**
- Replicating exact paper architectures
- Want fastest training (no AGOP overhead)
- Comparing optimizers only
- Do not need mechanistic analysis

**Use agop_experiments when:**
- Want to understand WHY grokking happens
- Need tractable AGOP metrics
- Comparing mechanisms across datasets
- Doing deep analysis

---

**For most research: Use [`../agop_experiments/`](../agop_experiments/) - it provides much more insight.**
