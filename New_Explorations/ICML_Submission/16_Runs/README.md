# ICML 16_Runs Experiments

Systematic experiments studying grokking dynamics across architectural and optimizer variations.

## Experiment Design

**16 base configurations** (2×2×2×2) across **6 weight decay values** = **96 total experiments**

### Factors

| Factor | Values |
|--------|--------|
| Modulus | 97, 113 |
| Activation | Softmax, ReLU |
| LayerNorm | On, Off |
| Optimizer | Adam, Muon |
| Weight Decay | 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0 |

### Naming Convention

```
p{modulus}_{attention}_{ln|noln}_{optimizer}/wd{value}_seed42/
```

Example: `p97_softmax_ln_adam/wd0.01_seed42/`

## Directory Structure

```
16_Runs/
├── README.md                    # This file
├── core/
│   ├── unified_transformer.py  # Configurable transformer model
│   ├── agop_utils.py           # Input-gradient AGOP tracker (symlink)
│   ├── lazy_rich_utils.py      # Lazy-rich dynamics tracker (symlink)
│   └── onehot_datasets.py      # One-hot encoded datasets (symlink)
├── configs/
│   └── icml_16runs.yaml        # Full experiment configuration
├── training_scripts/
│   └── train_icml_16runs.py    # Unified training script
├── slurm_scripts/
│   ├── run_16runs_sweep.sh     # Main SLURM array job (96 tasks)
│   └── logs/                   # SLURM job logs
├── results/
│   ├── p97_softmax_ln_adam/    # Base config directory
│   │   ├── wd1e-05_seed42/     # Weight decay run
│   │   ├── wd0.0001_seed42/
│   │   ├── wd0.001_seed42/
│   │   ├── wd0.01_seed42/
│   │   ├── wd0.1_seed42/
│   │   └── wd1.0_seed42/
│   ├── p97_softmax_ln_muon/
│   ├── ... (16 base config directories)
│   └── README.md
└── analysis/
    ├── analyze_16runs.ipynb    # Comprehensive analysis notebook
    └── figures/
```

## Tracked Metrics

### Training Metrics (`training_history.json`)
- `epoch`: Training epoch
- `train_loss`, `test_loss`: Cross-entropy loss
- `train_acc`, `test_acc`: Classification accuracy
- `weight_norm_total`: Total L2 norm of parameters

### AGOP Metrics (`agop_metrics.h5`)
- `agop_frobenius`: Frobenius norm ||AGOP||_F
- `agop_spectral_radius`: Largest eigenvalue λ₁
- `agop_trace`: Sum of eigenvalues Σλᵢ
- `agop_eigengap`: Gap between top eigenvalues (λ₁ - λ₂)
- `agop_variation_collapse_ratio`: λ₁/Σλᵢ
- `agop_eigenvalue_1` ... `agop_eigenvalue_10`: Top 10 eigenvalues

### Lazy-Rich Metrics (`lazy_rich_metrics.h5`)
- `ntk_distance`: Normalized NTK distance from initialization ||Kₜ - K₀||_F / ||K₀||_F
- `weight_norm_total`: Total weight norm
- `feature_kernel_distance`: Hidden representation kernel distance

## Quick Start

### Run Single Experiment (Local)

```bash
python training_scripts/train_icml_16runs.py \
    --modulus 97 \
    --attention_type softmax \
    --use_layernorm \
    --optimizer adam \
    --weight_decay 0.01 \
    --lr 0.001 \
    --n_epochs 50000 \
    --device cuda
```

### Run Full Sweep (SLURM)

```bash
cd slurm_scripts/
sbatch run_16runs_sweep.sh
```

This submits 96 jobs (16 configs × 6 weight decays).

### Monitor Jobs

```bash
squeue -u $USER
tail -f slurm_scripts/logs/icml_16runs_*.out
```

## Key Research Questions

1. **Modulus Effect**: Does p=97 vs p=113 affect grokking dynamics?
2. **Attention Type**: How does ReLU vs Softmax attention affect feature learning?
3. **LayerNorm**: Does LayerNorm accelerate or hinder grokking?
4. **Optimizer**: Adam vs Muon differences in lazy-to-rich transition?
5. **Weight Decay**: Critical threshold for grokking across configurations?

## Model Architecture

The `UnifiedTransformer` supports all configuration combinations:

```python
from core.unified_transformer import UnifiedTransformer

model = UnifiedTransformer(
    p=97,                      # Modulus
    d_model=128,               # Model dimension
    n_heads=4,                 # Attention heads
    n_layers=1,                # Transformer layers
    d_mlp=512,                 # MLP hidden dim
    attention_type='softmax',  # 'softmax' or 'relu'
    use_layernorm=True,        # LayerNorm on/off
)
```

## References

- **AGOP Theory**: Beaglehole et al. "Average gradient outer product as a mechanism for deep neural collapse"
- **Grokking**: Power et al. "Grokking: Generalization beyond overfitting on small algorithmic datasets"
- **Lazy-Rich Dynamics**: Kumar et al. (2024) "Grokking as the Transition from Lazy to Rich Training Dynamics"
- **ReLU Attention**: Nanda et al. (2023) "Progress measures for grokking via mechanistic interpretability"

## Status

- **Setup**: Complete
- **Experiments**: Pending
- **Analysis**: Pending

---

Last Updated: December 2024

