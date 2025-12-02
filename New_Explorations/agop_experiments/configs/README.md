# Configuration Files

This directory contains YAML configuration files that define default parameters for each experiment type. These files serve as reference specifications and can be used to configure batch experiments via SLURM scripts.

---

## Configuration Structure

Each configuration file follows a consistent structure with the following sections:

```yaml
dataset:       # Dataset-specific parameters
model:         # Architecture specifications
training:      # Training hyperparameters
optimizers:    # List of optimizer configurations with weight decay sweeps
agop:          # AGOP tracking parameters
save:          # Output directory configuration
```

---

## Configuration Files

### nanda_agop.yaml

Configuration for Nanda et al. modular addition experiments with ReLU transformer.

#### Dataset Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| name | `nanda_modular_addition` | Dataset identifier |
| p | 113 | Modulus (prime number) |
| train_fraction | 0.3 | 30% training, 70% test |

#### Model Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| type | `OneLayerReLUTransformer` | Architecture class |
| d_model | 128 | Embedding dimension |
| n_heads | 4 | Attention heads |
| d_mlp | 512 | MLP hidden dimension |

#### Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| n_epochs | 40,000 | Total training epochs |
| log_freq | 100 | Logging frequency |
| device | `cuda` | Compute device |
| seed | 42 | Random seed |

#### Optimizer Sweep

| Optimizer | Learning Rate | Weight Decays |
|-----------|---------------|---------------|
| AdamW | 0.001 | 0.1, 1.0, 5.0, 10.0 |
| Muon | 0.001 | 0.1, 1.0, 5.0, 10.0 |
| SGD | 0.01 | 0.1, 1.0, 5.0, 10.0 |

**Total configurations**: 3 optimizers × 4 weight decays = 12 experiments

---

### softmax_agop.yaml

Configuration for standard softmax transformer on modular addition.

#### Dataset Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| name | `softmax_modular_addition` | Dataset identifier |
| p | 97 | Modulus (prime number) |
| train_fraction | 0.5 | 50% training, 50% test |

#### Model Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| type | `StandardTransformer` | Architecture class |
| d_model | 128 | Embedding dimension |
| n_heads | 4 | Attention heads |
| n_layers | 2 | Transformer layers |
| d_ff | 512 | Feedforward dimension |
| dropout | 0.0 | No dropout |

#### Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| n_epochs | 50,000 | Total training epochs |
| log_freq | 100 | Logging frequency |
| device | `cuda` | Compute device |
| seed | 42 | Random seed |

#### Optimizer Sweep

| Optimizer | Learning Rate | Weight Decays |
|-----------|---------------|---------------|
| AdamW | 0.001 | 0.01, 0.1, 0.5, 1.0 |
| Muon | 0.001 | 0.01, 0.1, 0.5, 1.0 |
| SGD | 0.01 | 0.01, 0.1, 0.5, 1.0 |

**Total configurations**: 3 optimizers × 4 weight decays = 12 experiments

---

### mnist_agop.yaml

Configuration for MNIST Omnigrok experiments with MSE loss.

#### Dataset Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| name | `mnist_omnigrok` | Dataset identifier |
| train_points | 1,000 | Subsampled training set |
| input_dim | 784 | Flattened 28×28 images |

#### Model Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| type | `MNISTModel` | Architecture class |
| hidden_dim | 200 | Hidden layer size |
| output_dim | 10 | Number of classes |
| depth | 3 | Network depth |
| activation | `relu` | Activation function |
| initialization_scale | 8.0 | Weight scaling factor |

#### Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| n_epochs | 50,000 | Total training epochs |
| log_freq | 100 | Logging frequency |
| device | `cuda` | Compute device |
| seed | 42 | Random seed |
| use_mse_loss | true | MSE with one-hot targets |

#### AGOP Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| enabled | true | Enable AGOP tracking |
| freq | 100 | Computation frequency |
| top_k | 20 | Top eigenvectors to track |
| subsample | 500 | Reduced for 784-dim inputs |

#### Optimizer Sweep

| Optimizer | Learning Rate | Weight Decays |
|-----------|---------------|---------------|
| AdamW | 0.001 | 0.01, 0.1, 0.5, 1.0 |
| Muon | 0.001 | 0.01, 0.1, 0.5, 1.0 |
| SGD | 0.01 | 0.01, 0.1, 0.5, 1.0 |

**Total configurations**: 3 optimizers × 4 weight decays = 12 experiments

---

### composition_agop.yaml

Configuration for compositional reasoning experiments (placeholder implementation).

#### Dataset Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| name | `composition_reasoning` | Dataset identifier |
| n_entities | 50 | Number of entities |
| n_facts | 500 | Total facts/examples |
| train_fraction | 0.3 | 30% training |
| vocab_size | 100 | Vocabulary size |

#### Model Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| type | `SimpleCompositionTransformer` | Architecture class |
| d_model | 128 | Model dimension |
| n_heads | 4 | Attention heads |
| n_layers | 2 | Transformer layers |
| max_seq_len | 10 | Maximum sequence length |

#### Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| n_epochs | 100,000 | Total training epochs |
| log_freq | 100 | Logging frequency |
| device | `cuda` | Compute device |
| seed | 42 | Random seed |

#### AGOP Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| enabled | false | Disabled (needs adaptation) |
| freq | 100 | — |
| top_k | 20 | — |

**Note**: AGOP tracking is disabled for composition experiments as sequence models require additional adaptation. Run this dataset last after completing other experiments.

#### Optimizer Sweep

| Optimizer | Learning Rate | Weight Decays |
|-----------|---------------|---------------|
| AdamW | 0.0001 | 0.01, 0.1, 0.5, 1.0 |
| Muon | 0.0001 | 0.01, 0.1, 0.5, 1.0 |
| SGD | 0.001 | 0.01, 0.1, 0.5, 1.0 |

**Total configurations**: 3 optimizers × 4 weight decays = 12 experiments

---

## AGOP Configuration Section

Common AGOP parameters across all configurations:

| Parameter | Type | Description |
|-----------|------|-------------|
| enabled | bool | Whether to compute AGOP metrics |
| freq | int | Epochs between AGOP computations |
| top_k | int | Number of top eigenvectors for subspace tracking |
| subsample | int/null | Subsample size (null = use all data) |

**Memory considerations**:
- For high-dimensional inputs (MNIST: 784), use `subsample: 500` to reduce memory
- For modular arithmetic (≤226 dims), `subsample: null` is tractable

---

## Experiment Matrix Summary

| Dataset | Architectures | Optimizers | Weight Decays | Total Experiments |
|---------|---------------|------------|---------------|-------------------|
| Nanda | 2 (MLP, Transformer) | 3 | 4 | 24 |
| Softmax | 2 (MLP, Transformer) | 3 | 4 | 24 |
| MNIST | 1 (MLP) | 3 | 4 | 12 |
| Composition | 1 (MLP) | 3 | 4 | 12 |
| **Total** | — | — | — | **72** |

---

## Usage

### Loading Configuration in Python

```python
import yaml
from pathlib import Path

config_path = Path('configs/nanda_agop.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Access parameters
p = config['dataset']['p']
n_epochs = config['training']['n_epochs']
optimizers = config['optimizers']
```

### Using with Training Scripts

Configurations are typically consumed by SLURM scripts that iterate over optimizer and weight decay combinations:

```bash
# From slurm_scripts/run_nanda_full_sweep.sh
for OPTIMIZER in adamw muon sgd; do
    for WD in 0.1 1.0 5.0 10.0; do
        python train_nanda_agop.py \
            --optimizer $OPTIMIZER \
            --weight_decay $WD \
            --p 113 \
            --n_epochs 40000
    done
done
```

---

## Parameter Selection Rationale

### Modulus Values (p)

| Dataset | p | Rationale |
|---------|---|-----------|
| Nanda | 113 | Standard benchmark from original grokking papers |
| Softmax | 97 | Slightly smaller for computational efficiency |

### Learning Rates

| Optimizer | LR | Rationale |
|-----------|-----|-----------|
| AdamW | 0.001 | Standard for transformer training |
| Muon | 0.001 | Matched to AdamW for fair comparison |
| SGD | 0.01 | Higher LR needed for SGD convergence |

### Weight Decay Ranges

| Dataset | Range | Rationale |
|---------|-------|-----------|
| Nanda | 0.1–10.0 | Higher WD needed for grokking |
| Softmax/MNIST | 0.01–1.0 | Standard regularization range |

### Training Epochs

| Dataset | Epochs | Rationale |
|---------|--------|-----------|
| Nanda | 40,000 | Sufficient for grokking observation |
| Softmax | 50,000 | Slightly longer for architecture comparison |
| MNIST | 50,000 | Matches Omnigrok paper |
| Composition | 100,000 | Longer horizon for complex reasoning |

---

## References

- Nanda, N., et al. (2023). "Progress measures for grokking via mechanistic interpretability."
- Liu, Z., et al. (2022). "Omnigrok: Grokking Beyond Algorithmic Data."
- Power, A., et al. (2022). "Grokking: Generalization beyond overfitting."

---

*Last Updated: December 2024*

