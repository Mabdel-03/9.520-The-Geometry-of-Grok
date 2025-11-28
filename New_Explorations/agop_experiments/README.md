# AGOP-Tracking Experiments for Grokking Analysis

This directory contains experiments that track **Input-Gradient AGOP (Average Gradient Outer Product)** metrics to analyze grokking mechanisms across different tasks and optimizers.

## Overview

**Goal**: Understand what distinguishes grokking from non-grokking by tracking tractable AGOP metrics during training.

**Key Insight**: Instead of computing AGOP over parameters (expensive: 100K+ dimensions), we compute it over **inputs** (tractable: ~200-800 dimensions). This analyzes how the model's sensitivity to inputs evolves during grokking.

## What is Input-Gradient AGOP?

**AGOP** = (1/N) Σᵢ (∇L(xᵢ) ⊗ ∇L(xᵢ))

Instead of gradients w.r.t. parameters ∇_θ L, we use gradients w.r.t. inputs ∇_x f(x):
- **Much smaller matrices**: e.g., 194×194 for modular arithmetic (p=97) vs 100K×100K for parameters
- **Computationally tractable**: Full eigendecomposition is feasible
- **Interpretable**: Measures how input sensitivity evolves during learning

## Metrics Tracked

From the AGOP matrix, we compute:

1. **Frobenius Norm** ||AGOP||_F - Overall magnitude
2. **Spectral Radius** (λ₁) - Largest eigenvalue, max variance direction
3. **Trace** (Σλᵢ) - Total variance = E[||∇L||²]
4. **Eigengap** (λ₁ - λ₂) - Gradient alignment measure
5. **Variation Collapse Ratio** (λ₁/Σλᵢ) - Concentration measure
6. **Top-k Subspace Similarity** - Stability of gradient directions

## Datasets

Experiments run on 4 datasets (in order):

1. **Nanda** - Modular addition with ReLU transformer (cleanest grokking)
2. **Softmax** - Modular addition with standard transformer (architecture comparison)
3. **MNIST** - Image classification with MLP (perceptual vs symbolic)
4. **Composition** - Compositional reasoning (placeholder, run last)

## Directory Structure

```
agop_experiments/
├── README.md                    # This file
├── agop_utils.py               # Input-gradient AGOP tracker
├── train_nanda_agop.py         # Training script for Nanda
├── train_softmax_agop.py       # Training script for Softmax
├── train_mnist_agop.py         # Training script for MNIST
├── train_composition_agop.py   # Training script for Composition
├── analysis/
│   ├── visualize_agop_metrics.py   # Visualization
│   └── compare_grok_nogrok.py      # Compare conditions
├── configs/
│   ├── nanda_agop.yaml
│   ├── softmax_agop.yaml
│   ├── mnist_agop.yaml
│   └── composition_agop.yaml
└── slurm_scripts/
    ├── run_nanda_agop.sh
    ├── run_softmax_agop.sh
    ├── run_mnist_agop.sh
    ├── run_composition_agop.sh
    └── run_all_agop.sh
```

## Quick Start

### 1. Run Single Experiment (local)

```bash
# Nanda modular addition with AdamW
python train_nanda_agop.py \
    --optimizer adamw \
    --weight_decay 1.0 \
    --lr 0.001 \
    --p 113 \
    --n_epochs 40000 \
    --agop_freq 100 \
    --device cuda

# MNIST with Muon
python train_mnist_agop.py \
    --optimizer muon \
    --weight_decay 0.1 \
    --train_points 1000 \
    --n_epochs 50000 \
    --agop_freq 100 \
    --agop_subsample 500 \
    --device cuda
```

### 2. Run Batch Experiments (SLURM)

```bash
cd slurm_scripts/

# Submit all experiments (36 jobs total: 12 per dataset × 3 datasets)
./run_all_agop.sh

# Or submit individual datasets
sbatch run_nanda_agop.sh      # 12 jobs: 3 optimizers × 4 weight decays
sbatch run_softmax_agop.sh    # 12 jobs
sbatch run_mnist_agop.sh      # 12 jobs

# Monitor
squeue -u $USER
```

### 3. Analyze Results

```bash
cd analysis/

# Visualize single experiment (creates 9 plots!)
python visualize_agop_metrics.py \
    --results_dir ../results/agop_experiments/nanda/nanda_adamw_wd1.0_seed42

# Compare multiple optimizers
python visualize_agop_metrics.py \
    --results_dir ../results/agop_experiments/nanda \
    --experiment_pattern "nanda_*_wd1.0_*" \
    --compare_optimizers

# Compare grokking vs non-grokking
python compare_grok_nogrok.py \
    --results_dir ../results/agop_experiments/nanda \
    --experiment_pattern "nanda_*"
```

#### Generated Visualizations (Single Experiment)

The visualization script now creates **9 comprehensive plots**:

1. **`training_curves.png`** - Standard train/test accuracy curves
2. **`agop_basic_metrics.png`** - Frobenius norm, spectral radius, trace (3 panels)
3. **`agop_collapse_metrics.png`** - Eigengap and variation collapse ratio (2 panels)
4. **`agop_subspace_similarity.png`** - Top-k subspace stability over time
5. **`combined_grokking_agop.png`** - 2×2 grid: test acc + key AGOP metrics
6. **`comprehensive_timeline.png`** - **NEW!** 5-panel aligned timeline with grokking detection
7. **`aligned_test_acc_vcr.png`** - **NEW!** Dual-axis: test acc + variation collapse ratio
8. **`aligned_test_acc_eigengap.png`** - **NEW!** Dual-axis: test acc + eigengap
9. **`aligned_test_acc_trace.png`** - **NEW!** Dual-axis: test acc + trace
10. **`aligned_test_acc_spectral.png`** - **NEW!** Dual-axis: test acc + spectral radius

**Key Features:**
- 🎯 **Automatic grokking detection** and annotation on all plots
- 📊 **Dual-axis plots** show exact correlation between test accuracy and AGOP metrics
- 📈 **Comprehensive timeline** displays all metrics on shared x-axis with grokking marker
- 🔍 **Smoothing** applied to reduce noise (configurable window size)

## Experiment Matrix

Each dataset runs with:
- **Optimizers**: AdamW, Muon, SGD
- **Weight Decays**: 0.01, 0.1, 0.5, 1.0 (or 0.1, 1.0, 5.0, 10.0 for Nanda)
- **Total**: 12 runs per dataset

### Expected Outcomes

- **AdamW** with high weight decay: **Groks** (test acc → 99%)
- **SGD** with high weight decay: **Doesn't grok** (test acc stays low)
- **Muon**: **Varies** (interesting intermediate case)

## Key Questions to Answer

1. **Does eigengap collapse before grokking?**
   - Hypothesis: Eigengap should increase during grokking (gradient alignment)

2. **Does variation collapse ratio predict grokking?**
   - Hypothesis: VCR increases as gradients concentrate in top eigenvector

3. **Are there different AGOP patterns for different optimizers?**
   - Compare AdamW (groks) vs SGD (doesn't grok) vs Muon (?)

4. **Do symbolic and perceptual tasks show different AGOP dynamics?**
   - Compare modular arithmetic (Nanda/Softmax) vs MNIST

## Results Storage

Results are saved as:
```
results/agop_experiments/{dataset}/{experiment_name}/
├── config.json              # Experiment configuration
├── training_history.json    # Train/test acc and loss
└── agop_metrics.h5         # AGOP metrics (HDF5 for efficiency)
```

AGOP metrics in HDF5 include:
- `epoch`: Epochs when AGOP was computed
- `agop_frobenius`: Frobenius norm
- `agop_spectral_radius`: λ₁
- `agop_trace`: Σλᵢ
- `agop_eigengap`: λ₁ - λ₂
- `agop_variation_collapse_ratio`: λ₁/Σλᵢ
- `agop_topk_subspace_similarity`: Subspace stability
- `agop_eigenvalue_1` through `agop_eigenvalue_10`: Top eigenvalues

## Implementation Notes

### Memory Efficiency

Input-gradient AGOP is much more tractable than parameter-gradient AGOP:

| Dataset | Input Dim | AGOP Size | Memory |
|---------|-----------|-----------|--------|
| Nanda (p=113) | 226 | 226×226 | ~400KB |
| Softmax (p=97) | 3 | 3×3 | ~72B |
| MNIST | 784 | 784×784 | ~4.7MB |

Compare to parameter-gradient AGOP for typical models: ~40GB+

### Subsampling

For large datasets or frequent AGOP computation, use `--agop_subsample`:
```bash
python train_mnist_agop.py --agop_subsample 500  # Use 500 samples for AGOP
```

## Visualization Examples

The analysis scripts generate:

1. **Training curves**: Train/test accuracy over time
2. **AGOP basic metrics**: Frobenius norm, spectral radius, trace
3. **Collapse metrics**: Eigengap, variation collapse ratio
4. **Subspace stability**: Top-k subspace similarity
5. **Combined plots**: Test accuracy + AGOP metrics side-by-side
6. **Comparison plots**: Grokking vs non-grokking conditions

## References

- **AGOP Theory**: Beaglehole et al. "Average gradient outer product as a mechanism for deep neural collapse"
- **Grokking**: Power et al. "Grokking: Generalization beyond overfitting on small algorithmic datasets"
- **Notebook Implementation**: `Group1_Grokking_Code_Base.ipynb` (Cells 4-9)

## Future Work

### Phase 2: Parameter-Gradient AGOP

After completing input-gradient AGOP analysis, optionally add parameter-gradient AGOP:
- Use existing `framework/spectral_metrics.py`
- Subsample heavily for tractability
- Compare input-space vs parameter-space geometry

### Additional Analyses

- Correlation between AGOP metrics and grokking time
- Critical epochs for AGOP transitions
- Layer-wise AGOP (if feasible)
- Relationship to neural collapse

## Troubleshooting

### CUDA Out of Memory

Reduce AGOP frequency or subsample:
```bash
python train_nanda_agop.py --agop_freq 500 --agop_subsample 1000
```

### Missing Dependencies

Ensure you have:
```bash
pip install torch torchvision matplotlib h5py tqdm pyyaml seaborn
```

### SLURM Job Failures

Check logs in `slurm_scripts/logs/`:
```bash
tail slurm_scripts/logs/nanda_agop_*.err
```

## Contact

For questions or issues, refer to the main optimizer experiments documentation at:
`/om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/optimizer_experiments/`

---

## Current Status

**Experiments Completed**: 60/72 ✅  
**AGOP Metrics Available**: 60/72 ✅  

### Dataset Status:
- **Nanda**: 24/24 directories (modular addition with ReLU transformer)
- **Softmax**: 24/24 directories (modular addition with standard transformer)
- **MNIST**: 12/12 directories (image classification with MLP)
- **Composition**: 12/12 directories (compositional reasoning)

### Key Findings:
- 🎉 **Major Discovery**: Muon optimizer successfully groks on Softmax Transformer with one-hot inputs!
- ✅ **Transformers** grok ~2.4× better than MLPs across datasets
- ✅ **AdamW** most reliable optimizer for grokking (75% success rate on transformers)
- ✅ **Softmax dataset** shows highest grokking rate (47%)

For detailed results, see:
- `COMPREHENSIVE_RESULTS_REPORT.md` - Full analysis of all experiments
- `AGOP_RESULTS_SUMMARY.md` - Quick summary of key findings
- `analysis/` directory - Jupyter notebooks with visualizations and statistical analysis

---

**Status**: Experiments complete, analysis in progress ✓  
**Last Updated**: November 2025  
**Maintainer**: Course project team

