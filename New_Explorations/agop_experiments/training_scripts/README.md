# Training Scripts

This directory contains training scripts for grokking experiments with integrated AGOP (Average Gradient Outer Product) and Lazy-Rich dynamics tracking. Each script trains models on a specific dataset while recording comprehensive spectral and kernel-based metrics for analyzing the grokking phenomenon.

---

## Overview

All training scripts share a common experimental framework:

1. **Data preparation**: One-hot encoded inputs for tractable input-gradient AGOP computation
2. **Model instantiation**: Architecture-specific models (MLP or Transformer variants)
3. **Training loop**: Full-batch gradient descent with periodic metric logging
4. **Metric tracking**: AGOP metrics + Lazy-Rich dynamics (NTK distance, weight norms)
5. **Result serialization**: JSON for training history, HDF5 for metric tensors

---

## Experiment Specifications

### 1. train_nanda_agop.py

Modular addition experiment following Nanda et al. (2023) with AGOP and Lazy-Rich tracking.

#### Dataset

| Parameter | Value | Description |
|-----------|-------|-------------|
| Task | Modular Addition | (a + b) mod p |
| Modulus (p) | 113 | Prime number |
| Input Encoding | One-hot | Concatenated: [one_hot(a), one_hot(b)] |
| Input Dimension | 226 | 2 × p |
| Output Classes | 113 | p possible sums |
| Total Examples | 12,769 | p² |
| Train Fraction | 0.3 | ~3,830 training examples |
| Test Fraction | 0.7 | ~8,939 test examples |

#### Architectures

**MLP (ModularArithmeticMLP)**

| Layer | Dimensions | Activation |
|-------|------------|------------|
| Input | 226 → 128 | ReLU |
| Output | 128 → 113 | None |
| Total Parameters | ~43,500 | — |

**ReLU Transformer (OneHotReLUTransformer)**

| Component | Specification |
|-----------|---------------|
| d_model | 128 |
| n_heads | 4 |
| d_head | 32 |
| d_mlp | 512 |
| Attention Type | ReLU (not softmax) |
| Layers | 1 |
| Residual Connections | Yes |
| Layer Normalization | No |
| Total Parameters | ~200,000 |

#### Training Configuration

| Parameter | Value |
|-----------|-------|
| Loss Function | CrossEntropyLoss |
| Batch Size | Full batch |
| Training Epochs | 40,000 |
| Logging Frequency | Every 100 epochs |
| AGOP Computation Frequency | Every 100 epochs |
| Random Seed | 42 |

#### Optimizer Configurations

| Optimizer | Learning Rate | Weight Decay Values | Momentum | Additional |
|-----------|---------------|---------------------|----------|------------|
| AdamW | 0.001 | 0.1, 1.0, 5.0, 10.0 | β₁=0.9, β₂=0.999 | ε=1e-8 |
| Muon | 0.001 | 0.1, 1.0, 5.0, 10.0 | 0.95 | Nesterov=True |
| SGD | 0.01 | 0.1, 1.0, 5.0, 10.0 | 0.9 | — |

#### AGOP Tracking Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| top_k | 20 | Top eigenvectors for subspace tracking |
| subsample | None | Use all training data |
| agop_device | cpu | AGOP accumulation device |

#### Lazy-Rich Tracking Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| ntk_subsample | 200 | Subsample size for NTK computation |
| compute_ntk | True | Track NTK distance from initialization |
| compute_feature_kernel | True | Track hidden representation changes |

---

### 2. train_softmax_agop.py

Standard softmax transformer on modular addition for architecture comparison.

#### Dataset

| Parameter | Value | Description |
|-----------|-------|-------------|
| Task | Modular Addition | (a + b) mod p |
| Modulus (p) | 97 | Prime number |
| Input Encoding | One-hot | Concatenated: [one_hot(a), one_hot(b)] |
| Input Dimension | 194 | 2 × p |
| Output Classes | 97 | p possible sums |
| Total Examples | 9,409 | p² |
| Train Fraction | 0.5 | ~4,704 training examples |
| Test Fraction | 0.5 | ~4,705 test examples |

#### Architectures

**MLP (ModularArithmeticMLP)**

| Layer | Dimensions | Activation |
|-------|------------|------------|
| Input | 194 → 128 | ReLU |
| Output | 128 → 97 | None |
| Total Parameters | ~37,500 | — |

**Standard Transformer (OneHotStandardTransformer)**

| Component | Specification |
|-----------|---------------|
| d_model | 128 |
| n_heads | 4 |
| n_layers | 2 |
| d_ff | 512 |
| Attention Type | Softmax |
| Activation | GELU |
| Dropout | 0.0 |
| Layer Normalization | Yes (post-norm) |
| Residual Connections | Yes |
| Total Parameters | ~350,000 |

#### Training Configuration

| Parameter | Value |
|-----------|-------|
| Loss Function | CrossEntropyLoss |
| Batch Size | Full batch |
| Training Epochs | 50,000 |
| Logging Frequency | Every 100 epochs |
| AGOP Computation Frequency | Every 100 epochs |
| Random Seed | 42 |

#### Optimizer Configurations

| Optimizer | Learning Rate | Weight Decay Values | Momentum | Additional |
|-----------|---------------|---------------------|----------|------------|
| AdamW | 0.001 | 0.01, 0.1, 0.5, 1.0 | β₁=0.9, β₂=0.999 | ε=1e-8 |
| Muon | 0.001 | 0.01, 0.1, 0.5, 1.0 | 0.95 | Nesterov=True |
| SGD | 0.01 | 0.01, 0.1, 0.5, 1.0 | 0.9 | — |

---

### 3. train_mnist_agop.py

MNIST classification following the Omnigrok setup (Liu et al., 2022) with MSE loss.

#### Dataset

| Parameter | Value | Description |
|-----------|-------|-------------|
| Task | Image Classification | Digit recognition (0-9) |
| Input Format | Flattened grayscale | 28×28 → 784 |
| Input Dimension | 784 | Continuous pixel values [0, 1] |
| Output Classes | 10 | Digits 0-9 |
| Training Points | 1,000 | Subsampled from 60,000 |
| Test Points | 10,000 | Full MNIST test set |
| Normalization | ToTensor | Scales to [0, 1] |

#### Architecture (MNISTModel)

| Layer | Dimensions | Activation |
|-------|------------|------------|
| Flatten | 28×28 → 784 | — |
| Hidden 1 | 784 → 200 | ReLU |
| Hidden 2 | 200 → 200 | ReLU |
| Output | 200 → 10 | None |

| Property | Value |
|----------|-------|
| Depth | 3 layers |
| Hidden Dimension | 200 |
| Initialization Scale | 8.0 |
| Total Parameters | ~199,210 |

**Initialization**: All weights scaled by factor of 8.0 (Omnigrok prescription for inducing grokking).

#### Training Configuration

| Parameter | Value |
|-----------|-------|
| Loss Function | MSELoss |
| Target Encoding | One-hot (10-dim) |
| Batch Size | Full batch |
| Training Epochs | 50,000 |
| Logging Frequency | Every 100 epochs |
| AGOP Computation Frequency | Every 100 epochs |
| Random Seed | 42 |

**Note**: MSE loss with one-hot targets is used per the Omnigrok paper, which demonstrated this setup reliably induces grokking on image classification tasks.

#### Optimizer Configurations

| Optimizer | Learning Rate | Weight Decay Values | Momentum | Additional |
|-----------|---------------|---------------------|----------|------------|
| AdamW | 0.001 | 0.01, 0.1, 0.5, 1.0 | β₁=0.9, β₂=0.999 | ε=1e-8 |
| Muon | 0.001 | 0.01, 0.1, 0.5, 1.0 | 0.95 | Nesterov=True |
| SGD | 0.01 | 0.01, 0.1, 0.5, 1.0 | 0.9 | — |

#### AGOP Tracking Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| top_k | 20 | Top eigenvectors for subspace tracking |
| subsample | 500 | Reduced for memory efficiency (784-dim inputs) |
| agop_device | cpu | AGOP accumulation device |

---

### 4. train_composition_agop.py

Compositional reasoning experiment (placeholder implementation).

#### Dataset

| Parameter | Value | Description |
|-----------|-------|-------------|
| Task | Compositional Reasoning | Sequence → class prediction |
| Vocabulary Size | 100 | Token vocabulary |
| Sequence Length | 10 | Tokens per sequence |
| Input Encoding | One-hot | Concatenated per-position one-hots |
| Input Dimension | 1,000 | vocab_size × seq_len |
| Output Classes | 100 | vocab_size |
| Total Facts | 500 | Generated examples |
| Train Fraction | 0.3 | ~150 training examples |
| Test Fraction | 0.7 | ~350 test examples |

**Note**: This is a simplified placeholder implementation. Full compositional reasoning requires proper knowledge graph generation following Wang et al.

#### Architecture (CompositionMLP)

| Layer | Dimensions | Activation |
|-------|------------|------------|
| Input | 1000 → 128 | ReLU |
| Hidden | 128 → 128 | ReLU |
| Output | 128 → 100 | None |

| Property | Value |
|----------|-------|
| n_layers | 2 |
| Hidden Dimension | 128 |
| Total Parameters | ~145,000 |

#### Training Configuration

| Parameter | Value |
|-----------|-------|
| Loss Function | CrossEntropyLoss |
| Batch Size | Full batch |
| Training Epochs | 100,000 |
| Logging Frequency | Every 100 epochs |
| AGOP Computation Frequency | Every 100 epochs |
| Random Seed | 42 |

#### Optimizer Configurations

| Optimizer | Learning Rate | Weight Decay Values | Momentum | Additional |
|-----------|---------------|---------------------|----------|------------|
| AdamW | 0.0001 | 0.01, 0.1, 0.5, 1.0 | β₁=0.9, β₂=0.999 | ε=1e-8 |
| Muon | 0.0001 | 0.01, 0.1, 0.5, 1.0 | 0.95 | Nesterov=True |
| SGD | 0.001 | 0.01, 0.1, 0.5, 1.0 | 0.9 | — |

---

## Metrics Tracked

### AGOP Metrics (Input-Gradient)

All scripts compute the following from the input-gradient AGOP matrix:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Frobenius Norm | ‖AGOP‖_F | Overall gradient magnitude |
| Spectral Radius | λ₁ | Maximum variance direction |
| Trace | Σλᵢ | Total variance = E[‖∇L‖²] |
| Eigengap | λ₁ - λ₂ | Gradient alignment strength |
| Variation Collapse Ratio | λ₁ / Σλᵢ | Concentration in top direction |
| Top-k Subspace Similarity | mean(svd(Uₖᵀ U'ₖ)) | Stability of gradient directions |
| Top Eigenvalues | λ₁, λ₂, ..., λ₁₀ | Spectrum shape |

### Lazy-Rich Metrics (Kumar et al., 2024)

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| NTK Distance | ‖Kₜ - K₀‖_F / ‖K₀‖_F | Kernel drift from initialization |
| NTK Norm | ‖Kₜ‖_F | Current kernel magnitude |
| Weight Norm Total | ‖θₜ‖₂ | Total L2 norm of parameters |
| Weight Norm Change | (‖θₜ‖ - ‖θ₀‖) / ‖θ₀‖ | Relative weight change |
| Feature Kernel Distance | ‖Kₕ,ₜ - Kₕ,₀‖_F / ‖Kₕ,₀‖_F | Hidden representation change |

### Training Metrics

| Metric | Description |
|--------|-------------|
| train_loss | Training loss per epoch |
| train_acc | Training accuracy per epoch |
| test_loss | Test loss per evaluation |
| test_acc | Test accuracy per evaluation |
| weight_norm_total | Total weight norm |

---

## Output Files

Each experiment produces the following files in its result directory:

| File | Format | Contents |
|------|--------|----------|
| `config.json` | JSON | Complete experiment configuration |
| `training_history.json` | JSON | Epoch, loss, accuracy, weight norms |
| `agop_metrics.h5` | HDF5 | AGOP eigenvalues and derived metrics |
| `lazy_rich_metrics.h5` | HDF5 | NTK distance, feature kernel distance |

---

## Usage Examples

### Single Experiment (Local)

```bash
# Nanda with AdamW
python train_nanda_agop.py \
    --architecture transformer \
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
    --agop_subsample 500 \
    --device cuda
```

### Command-Line Arguments

All scripts support the following argument categories:

**Dataset Arguments**
- `--p`: Modulus for modular arithmetic (Nanda/Softmax)
- `--train_fraction`: Fraction of data for training
- `--train_points`: Number of training points (MNIST)

**Model Arguments**
- `--architecture`: Model type (`mlp` or `transformer`)
- `--d_model`: Model dimension
- `--n_heads`: Number of attention heads
- `--hidden_dim`: MLP hidden dimension
- `--depth`: Network depth

**Optimizer Arguments**
- `--optimizer`: Optimizer choice (`adamw`, `muon`, `sgd`, `adam`)
- `--lr`: Learning rate
- `--weight_decay`: Weight decay coefficient

**Training Arguments**
- `--n_epochs`: Total training epochs
- `--log_freq`: Logging frequency
- `--device`: Compute device (`cuda` or `cpu`)
- `--seed`: Random seed

**AGOP Arguments**
- `--agop_freq`: AGOP computation frequency
- `--agop_subsample`: Subsample size for AGOP
- `--agop_top_k`: Number of top eigenvectors to track

**Lazy-Rich Arguments**
- `--ntk_subsample`: Subsample size for NTK
- `--compute_ntk`: Enable NTK tracking
- `--no_ntk`: Disable NTK tracking
- `--compute_feature_kernel`: Enable feature kernel tracking

**Output Arguments**
- `--save_dir`: Results directory
- `--experiment_name`: Custom experiment name

---

## Dependencies

```python
# Core
torch >= 1.9.0
numpy >= 1.20.0

# Data & I/O
torchvision  # For MNIST
h5py >= 3.0.0  # For metric storage

# Progress
tqdm >= 4.60.0

# Custom modules (from framework)
muon_official  # Muon optimizer implementation
```

---

## References

1. **Grokking**: Power, A., et al. (2022). "Grokking: Generalization beyond overfitting on small algorithmic datasets." arXiv:2201.02177
2. **Nanda Setup**: Nanda, N., et al. (2023). "Progress measures for grokking via mechanistic interpretability." ICLR 2023
3. **Omnigrok (MNIST)**: Liu, Z., et al. (2022). "Omnigrok: Grokking Beyond Algorithmic Data." ICLR 2023
4. **AGOP Theory**: Beaglehole, D., et al. (2023). "Average gradient outer product as a mechanism for deep neural collapse." arXiv:2310.02672
5. **Lazy-Rich Dynamics**: Kumar, A., et al. (2024). "Grokking as the Transition from Lazy to Rich Training Dynamics." arXiv:2310.06110
6. **Muon Optimizer**: Jordan, K. (2024). "Muon: An optimizer for hidden layers in neural networks."

---

*Last Updated: December 2024*

