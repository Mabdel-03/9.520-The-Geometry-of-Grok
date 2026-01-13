# Power AGOP Study: Grokking and Input-Gradient Sensitivity

This study investigates whether **grokking** (delayed generalization after overfitting) corresponds to **concentration of input-gradient sensitivity**, measured via the **Average Gradient Outer Product (AGOP)** eigenspectrum analysis.

Based on [Power et al. (2022)](https://arxiv.org/abs/2201.02177) "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets".

---

## Overview

### Research Question
Does grokking correspond to a concentration of the model's gradient sensitivity in a low-dimensional subspace of the input space?

### Key Metric: Variation Collapse Ratio (VCR)
$$\text{VCR} = \frac{\lambda_1}{\sum_i \lambda_i}$$

Where $\lambda_i$ are the eigenvalues of the AGOP matrix. A higher VCR indicates that gradient sensitivity is concentrated along fewer directions.

---

## Tasks

This study includes two modular arithmetic tasks:

| Task | Operation | Description | Expected Behavior |
|------|-----------|-------------|-------------------|
| **Addition** | $f(a, b) = (a + b) \mod 97$ | Original Power et al. task | Grokking expected |
| **Cubic Polynomial** | $f(a, b) = (a^3 + ab) \mod 97$ | More complex polynomial | Grokking unlikely |

The cubic polynomial task serves as a **negative control** to test whether VCR spikes are specifically associated with grokking, or occur in all training scenarios.

---

## Task 1: Modular Addition

**Operation:** $f(a, b) = (a + b) \mod p$

**Modulus:** $p = 97$ (prime)

### Dataset
| Property | Value |
|----------|-------|
| Total examples | $97^2 = 9409$ |
| Train split | 50% (4,704 examples) |
| Test split | 50% (4,705 examples) |
| Split method | Random (seed=42) |
| Output | 97-dimensional logits + cross-entropy loss |

### Input Representations
1. **Discrete tokens** (default): Integers $a, b \in \{0, 1, \ldots, 96\}$ embedded via learned embedding layer
2. **Continuous one-hot** (ablation): 97-dimensional one-hot vectors for $a$ and $b$, concatenated to form 194-dimensional input

---

## Architectures

### Transformer (Decoder-Only)

Based on the Power et al. (2022) architecture with modern conventions.

| Component | Configuration |
|-----------|---------------|
| Type | Decoder-only with causal masking |
| Input format | Sequence of 3 tokens: `[tok_a, tok_b, tok_equals]` |
| Embedding dimension | 128 |
| Layers | 2 |
| Attention heads | 4 (head dimension = 32) |
| MLP hidden dimension | 512 (4× embedding dim) |
| Activation | GELU |
| Positional encoding | Learned embeddings |
| Normalization | Pre-norm LayerNorm (GPT-2 style) |
| Output | Linear projection from final token → 97 logits |
| **Total parameters** | ~421,000 |

**Key Design Choices:**
- Pre-norm configuration places LayerNorm before attention and MLP blocks
- Causal masking prevents attending to future positions
- Final token (the "=" position) is used for prediction
- Combined QKV projection for efficiency

### MLP (3-Layer)

A minimal architecture with no inductive biases, serving as a baseline.

| Component | Configuration |
|-----------|---------------|
| Input format | Concatenated embeddings `[emb_a; emb_b]` (dim 256) |
| Architecture | `256 → 512 → 512 → 97` |
| Activation | ReLU |
| Normalization | None (baseline) or LayerNorm (ablation) |
| Dropout | None |
| **Total parameters** | ~300,000 |

**Rationale:** If VCR spikes occur in both transformer and MLP, this strengthens the claim that gradient concentration is an invariant property of grokking, independent of architectural inductive biases.

---

## Task 2: Cubic Polynomial (Negative Control)

**Operation:** $f(a, b) = (a^3 + ab) \mod p$

**Modulus:** $p = 97$ (prime)

### Dataset
Same structure as modular addition:
| Property | Value |
|----------|-------|
| Total examples | $97^2 = 9409$ |
| Train split | 50% (4,704 examples) |
| Test split | 50% (4,705 examples) |
| Split method | Random (seed=42) |

### Motivation
The cubic polynomial is more complex than simple addition:
- Involves higher-order polynomial terms ($a^3$)
- Includes a cross-term ($ab$)
- Less symmetric structure than basic operations

**Hypothesis:** This task is unlikely to exhibit grokking with the current architectures and hyperparameters, providing a control condition to validate that VCR spikes are specific to successful generalization.

### Expected Outcomes
- **No grokking:** Models may memorize training data but fail to generalize
- **No VCR spike:** If VCR spikes correlate with grokking, they should be absent here
- **Comparison:** Analyzing AGOP behavior in grokking vs. non-grokking scenarios

---

## Experimental Factors

### Weight Decay (Primary Factor)
The main experimental variable, systematically varied to produce both grokking and non-grokking outcomes.

| Value | Regime | Expected Behavior |
|-------|--------|-------------------|
| 0 | No regularization | Non-grokking baseline (pure memorization) |
| 1e-4 | Minimal | Late/rare grokking |
| 1e-3 | Productive | Reliable grokking |
| **1e-2** | **Sweet spot** | Fast, reliable grokking |
| 1e-1 | Productive | Even faster grokking |
| 1.0 | Strong | Testing upper limit (potential instability) |

### Optimizers

1. **AdamW**
   - Learning rate: 0.001
   - Betas: (0.9, 0.999)
   - Epsilon: 1e-8
   - Weight decay: Applied per experiment

2. **Muon**
   - Learning rate: 0.001
   - Momentum: 0.95
   - Nesterov: True
   - Newton-Schulz orthogonalization
   - Weight decay: Applied per experiment

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 50,000 |
| Batch size | Full batch (all training examples) |
| Learning rate | 0.001 |
| Device | CUDA |
| Random seed | 42 |

### Logging
- **Metrics logged:** Train loss, train accuracy, test loss, test accuracy, weight norm
- **Logging frequency:** Every 100 epochs

---

## AGOP Tracking

The Average Gradient Outer Product captures how the model's output sensitivity varies with respect to input directions.

### Configuration
| Parameter | Value |
|-----------|-------|
| Computation frequency | Every 100 epochs |
| Top eigenvalues tracked | 20 |
| Input representation | One-hot (for differentiability) |

### Metrics Computed

**Primary:**
- `agop_variation_collapse_ratio` (VCR): $\lambda_1 / \sum_i \lambda_i$

**Secondary:**
- `agop_trace`: $\sum_i \lambda_i$ (total gradient variance)
- `agop_eigengap`: $\lambda_1 - \lambda_2$ (gradient alignment)
- `agop_spectral_radius`: $\lambda_1$ (largest eigenvalue)
- `agop_frobenius`: $\|AGOP\|_F$ (Frobenius norm)

**Energy Concentration:**
- `agop_top5_energy_ratio`: $\sum_{i=1}^{5} \lambda_i / \sum \lambda_i$
- `agop_top10_energy_ratio`: $\sum_{i=1}^{10} \lambda_i / \sum \lambda_i$

---

## Experiment Matrix

**Total: 96 experiments** = 2 (tasks) × 2 (architectures) × 2 (optimizers) × 2 (input types) × 6 (weight decays)

- **48 experiments** for modular addition (original)
- **48 experiments** for cubic polynomial (negative control)

### SLURM Array Task Mapping

```
task_id = arch_idx × 24 + opt_idx × 12 + input_idx × 6 + wd_idx
```

| Task Range | Architecture | Optimizer | Input Type |
|------------|--------------|-----------|------------|
| 0-5 | Transformer | AdamW | Discrete |
| 6-11 | Transformer | AdamW | One-hot |
| 12-17 | Transformer | Muon | Discrete |
| 18-23 | Transformer | Muon | One-hot |
| 24-29 | MLP | AdamW | Discrete |
| 30-35 | MLP | AdamW | One-hot |
| 36-41 | MLP | Muon | Discrete |
| 42-47 | MLP | Muon | One-hot |

Within each range, weight decay varies: `[0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]`

---

## Directory Structure

```
Power_AGOP_Study/
├── README.md                    # This file
├── configs/
│   ├── power_agop_sweep.yaml   # Addition experiment configuration
│   └── cubic_agop_sweep.yaml   # Cubic polynomial experiment configuration
├── core/
│   ├── __init__.py
│   ├── power_transformer.py    # Decoder-only transformer implementation
│   ├── grokking_mlp.py         # MLP baseline implementation
│   ├── datasets.py             # Modular arithmetic dataset (add, cubic, etc.)
│   ├── agop_utils.py           # AGOP computation utilities
│   └── lazy_rich_utils.py      # Training dynamics utilities
├── training_scripts/
│   └── train_power_agop.py     # Main training script (supports --operation)
├── slurm_scripts/
│   ├── run_power_sweep.sh      # SLURM array job for addition (48 experiments)
│   ├── run_cubic_sweep.sh      # SLURM array job for cubic (48 experiments)
│   └── logs/                   # Job output logs
├── results/                    # Addition experiment outputs
│   └── {arch}_{input}_{opt}/
│       └── wd{weight_decay}_seed{seed}/
│           ├── config.json
│           ├── training_history.json
│           └── agop_metrics.h5
└── results_cubic/              # Cubic polynomial experiment outputs
    └── {arch}_{input}_{opt}/
        └── wd{weight_decay}_seed{seed}/
            ├── config.json
            ├── training_history.json
            └── agop_metrics.h5
```

---

## Running Experiments

### Submit All Addition Experiments (48 jobs)
```bash
cd slurm_scripts
sbatch run_power_sweep.sh
```

### Submit All Cubic Polynomial Experiments (48 jobs)
```bash
cd slurm_scripts
sbatch run_cubic_sweep.sh
```

### Submit Specific Tasks
```bash
# Run only transformer + adamw + discrete experiments (tasks 0-5)
sbatch --array=0-5 run_power_sweep.sh   # Addition
sbatch --array=0-5 run_cubic_sweep.sh   # Cubic

# Run a single experiment
sbatch --array=3 run_power_sweep.sh  # transformer_discrete_adamw, wd=1e-2, addition
sbatch --array=3 run_cubic_sweep.sh  # transformer_discrete_adamw, wd=1e-2, cubic
```

### Run Single Experiment Manually
```bash
# Addition (default)
python training_scripts/train_power_agop.py \
    --architecture transformer \
    --input_type discrete \
    --optimizer adamw \
    --weight_decay 0.01 \
    --n_epochs 50000 \
    --seed 42

# Cubic polynomial
python training_scripts/train_power_agop.py \
    --operation cubic \
    --architecture transformer \
    --input_type discrete \
    --optimizer adamw \
    --weight_decay 0.01 \
    --n_epochs 50000 \
    --seed 42 \
    --save_dir ./results_cubic
```

---

## Output Files

### `training_history.json`
Contains per-epoch metrics:
```json
{
  "epoch": [0, 100, 200, ...],
  "train_loss": [...],
  "train_acc": [...],
  "test_loss": [...],
  "test_acc": [...],
  "weight_norm_total": [...]
}
```

### `agop_metrics.h5`
HDF5 file containing AGOP eigenvalues and metrics at each checkpoint:
- `epochs`: Array of epoch numbers
- `eigenvalues`: [n_checkpoints, top_k] array of eigenvalues
- `vcr`: Variation collapse ratio over time
- `trace`, `eigengap`, etc.

---

## Experiment Status

### Modular Addition Experiments (Complete)
- **Job ID:** 44540373
- **Status:** Completed
- **Results:** Available in `results/`

### Cubic Polynomial Experiments (Running)
- **Job ID:** 44559114
- **Status:** In progress (started January 12, 2026)
- **Results:** Saving to `results_cubic/`

---

## Early Results (Preliminary)

### Modular Addition Results

From completed experiments on transformer + AdamW + discrete:

| Weight Decay | VCR at ~33K epochs | Observation |
|--------------|-------------------|-------------|
| 0 | ~0.13 | Low concentration |
| 1e-4 | ~0.13 | Similar to wd=0 |
| 1e-3 | ~0.30 | Moderate concentration |
| **1e-2** | **~0.53** | **Highest concentration** |
| 1e-1 | ~0.26 | Decreasing |
| 1.0 | ~0.24 | Stable |

**Key Finding:** Weight decay = 1e-2 produces the highest VCR, suggesting an optimal regularization strength for gradient concentration.

**Muon vs AdamW:** Muon optimizer shows consistently lower VCR (~0.04) compared to AdamW (~0.13-0.53), indicating fundamentally different gradient geometry.

### Cubic Polynomial Results (Preliminary)

Early observations from running experiments (epoch ~300):

| Configuration | Train Acc | Test Acc | VCR | Observation |
|--------------|-----------|----------|-----|-------------|
| transformer_discrete_adamw, wd=0 | ~1.0% | ~1.9% | ~0.05-0.06 | Low accuracy, low VCR |

**Early Indication:** As hypothesized, the cubic polynomial task shows:
- Very low test accuracy (near random chance)
- Low, stable VCR without the spike seen in addition experiments
- This supports the hypothesis that VCR concentration is specific to grokking

---

## Comparing Addition vs. Cubic Experiments

The key analysis will compare AGOP metrics between the two tasks:

| Metric | Addition (Grokking) | Cubic (No Grokking) |
|--------|---------------------|---------------------|
| Final Test Accuracy | Expected: >95% | Expected: ~50% (random) |
| VCR Spike | Expected: Yes | Expected: No |
| AGOP Eigenspectrum | Concentrated | Diffuse |

**Key Questions:**
1. Does VCR spike occur only when grokking occurs?
2. What is the AGOP eigenspectrum like for non-grokking tasks?
3. Is there a relationship between task complexity and VCR behavior?

---

## References

1. Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). Grokking: Generalization beyond overfitting on small algorithmic datasets. *arXiv preprint arXiv:2201.02177*.

2. Nanda, N., Chan, L., Liberum, T., Smith, J., & Steinhardt, J. (2023). Progress measures for grokking via mechanistic interpretability. *arXiv preprint arXiv:2301.05217*.

---

## License

MIT License

