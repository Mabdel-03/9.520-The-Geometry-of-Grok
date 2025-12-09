# Core Modules

This directory contains the foundational utilities for AGOP-tracking grokking experiments. These modules implement tractable input-gradient AGOP computation, Lazy-Rich training dynamics tracking, one-hot encoded datasets, and model architectures designed for differentiable inputs.

---

## Module Overview

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| `agop_utils.py` | Input-gradient AGOP computation | `InputGradientAGOPTracker` |
| `lazy_rich_utils.py` | NTK and weight norm tracking | `LazyRichTracker`, `compute_ntk_efficient` |
| `onehot_datasets.py` | Dataset creation with one-hot encoding | `create_onehot_modular_dataset`, etc. |
| `onehot_models.py` | Model architectures for continuous inputs | `ModularArithmeticMLP`, `OneHotReLUTransformer`, etc. |

---

## agop_utils.py

Implements tractable AGOP computation using input gradients rather than parameter gradients.

### Theoretical Background

The Average Gradient Outer Product (AGOP) is defined as:

```
AGOP = (1/N) Σᵢ (∇f(xᵢ) ⊗ ∇f(xᵢ))
```

**Key insight**: Instead of computing gradients with respect to parameters (∇_θ, which yields matrices of size ~100K × 100K), we compute gradients with respect to inputs (∇_x), yielding tractable matrices:

| Dataset | Input Dimension | AGOP Matrix Size | Memory |
|---------|-----------------|------------------|--------|
| Nanda (p=113) | 226 | 226 × 226 | ~400 KB |
| Softmax (p=97) | 194 | 194 × 194 | ~300 KB |
| MNIST | 784 | 784 × 784 | ~4.7 MB |

Compare to parameter-gradient AGOP: ~40+ GB for typical models.

### Class: InputGradientAGOPTracker

```python
class InputGradientAGOPTracker:
    def __init__(
        self,
        top_k: int = 4,              # Top eigenvectors for subspace tracking
        subsample_size: int = None,   # Random subsample (None = use all)
        device: str = 'cuda',         # Model computation device
        agop_device: str = 'cpu',     # AGOP accumulation device
        use_mse_loss: bool = False    # For Omnigrok experiments
    )
```

#### Methods

**`compute_input_agop(model, data, labels, criterion)`**

Computes the input-gradient AGOP matrix.

| Parameter | Type | Description |
|-----------|------|-------------|
| model | nn.Module | PyTorch model |
| data | Tensor | Input data [N, d_input] |
| labels | Tensor | Labels [N] |
| criterion | nn.Module | Loss function |
| **Returns** | Tensor | AGOP matrix [d_input, d_input] |

**`compute_agop_metrics(history, epoch_agop, k=None)`**

Extracts comprehensive metrics from an AGOP matrix.

| Metric | Key | Formula |
|--------|-----|---------|
| Frobenius Norm | `agop_frobenius` | ‖AGOP‖_F |
| Spectral Radius | `agop_spectral_radius` | λ₁ |
| Trace | `agop_trace` | Σλᵢ |
| Eigengap | `agop_eigengap` | λ₁ - λ₂ |
| Variation Collapse Ratio | `agop_variation_collapse_ratio` | λ₁ / Σλᵢ |
| Top-k Subspace Similarity | `agop_topk_subspace_similarity` | Alignment with previous epoch |
| Top Eigenvalue Energy | `agop_top_eigenvalue_energy` | λ₁ / Σλᵢ |
| Top-5 Energy Ratio | `agop_top5_energy_ratio` | Σλ₁₋₅ / Σλᵢ |
| Top-10 Energy Ratio | `agop_top10_energy_ratio` | Σλ₁₋₁₀ / Σλᵢ |
| Individual Eigenvalues | `agop_eigenvalue_{1-10}` | λ₁, λ₂, ..., λ₁₀ |

### Usage Example

```python
from agop_utils import InputGradientAGOPTracker

tracker = InputGradientAGOPTracker(
    top_k=20,
    subsample_size=500,
    device='cuda',
    agop_device='cpu'
)

# Compute AGOP
agop_matrix = tracker.compute_input_agop(model, train_data, train_labels, criterion)

# Extract metrics
history = {}
metrics = tracker.compute_agop_metrics(history, agop_matrix)
print(f"Eigengap: {metrics['agop_eigengap']:.4e}")
print(f"VCR: {metrics['agop_variation_collapse_ratio']:.4f}")
```

---

## lazy_rich_utils.py

Implements Lazy-Rich training dynamics tracking based on Kumar et al. (2024).

### Theoretical Background

In the **lazy regime**, the Neural Tangent Kernel (NTK) remains approximately constant—the network behaves like a kernel machine with fixed features. In the **rich regime**, the NTK evolves significantly, indicating active feature learning.

**Key hypothesis**: Grokking coincides with the transition from lazy to rich training dynamics.

### Class: LazyRichTracker

```python
class LazyRichTracker:
    def __init__(
        self,
        n_subsample: int = 200,       # Points for NTK computation
        device: str = 'cuda',          # Model computation device
        output_device: str = 'cpu',    # Kernel storage device
        feature_layer: str = None,     # Layer for feature kernel (auto-detect)
        use_efficient_ntk: bool = True # Vectorized NTK computation
    )
```

#### Methods

**`initialize(model, data, compute_ntk=True, compute_fk=True)`**

Stores initial kernels K₀ before training begins.

| Parameter | Type | Description |
|-----------|------|-------------|
| model | nn.Module | Model at initialization |
| data | Tensor | Training data |
| compute_ntk | bool | Store initial NTK |
| compute_fk | bool | Store initial feature kernel |

**`compute_metrics(model, data, history, compute_ntk=True, compute_feature_kernel_dist=True)`**

Computes lazy-rich metrics at current epoch.

| Metric | Key | Formula |
|--------|-----|---------|
| NTK Distance | `ntk_distance` | ‖Kₜ - K₀‖_F / ‖K₀‖_F |
| NTK Norm | `ntk_norm` | ‖Kₜ‖_F |
| Weight Norm Total | `weight_norm_total` | ‖θₜ‖₂ |
| Weight Norm Change | `weight_norm_change` | (‖θₜ‖ - ‖θ₀‖) / ‖θ₀‖ |
| Feature Kernel Distance | `feature_kernel_distance` | ‖Kₕ,ₜ - Kₕ,₀‖_F / ‖Kₕ,₀‖_F |
| Feature Kernel Norm | `feature_kernel_norm` | ‖Kₕ,ₜ‖_F |

### Standalone Functions

**`compute_weight_norms(model)`**

Returns dictionary with total and per-layer L2 norms. O(p) complexity—suitable for every epoch.

**`compute_ntk_efficient(model, data, n_subsample, device, output_device)`**

Vectorized NTK computation: K = J @ J^T where J is the stacked Jacobian.

**`compute_feature_kernel(model, data, layer_name, n_subsample, device, output_device)`**

Computes feature kernel from hidden layer activations using forward hooks.

**`detect_lazy_rich_transition(ntk_distances, epochs, threshold, window)`**

Automatically detects the epoch where NTK distance begins increasing rapidly.

### Usage Example

```python
from lazy_rich_utils import LazyRichTracker, compute_weight_norms

# Initialize before training
tracker = LazyRichTracker(n_subsample=200, device='cuda')
tracker.initialize(model, train_data, compute_ntk=True, compute_fk=True)

# During training
lazy_rich_history = {}
for epoch in range(n_epochs):
    # ... training step ...
    
    if epoch % 100 == 0:
        metrics = tracker.compute_metrics(
            model, train_data, lazy_rich_history,
            compute_ntk=True,
            compute_feature_kernel_dist=True
        )
        print(f"NTK Distance: {metrics['ntk_distance']:.4e}")
```

---

## onehot_datasets.py

Creates datasets with one-hot encoded inputs for tractable input-gradient AGOP computation.

### Design Rationale

Standard datasets use discrete token indices (integers), which are not differentiable. One-hot encoding converts inputs to continuous vectors, enabling gradient computation with respect to inputs.

### Functions

**`create_onehot_modular_dataset(p, operation, train_fraction, device)`**

Creates modular arithmetic dataset with one-hot encoded pairs.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| p | int | — | Modulus (prime) |
| operation | str | 'add' | 'add', 'sub', 'mul', or 'div' |
| train_fraction | float | 0.3 | Training set fraction |
| device | str | 'cpu' | Tensor device |

**Returns**: `(train_data, train_labels, test_data, test_labels)`
- `train_data`: [N_train, 2p] one-hot encoded (float32)
- `train_labels`: [N_train] targets (long)

**Encoding scheme**: Input (a, b) → vector of size 2p where positions [0:p] encode a and [p:2p] encode b.

---

**`create_onehot_mnist_dataset(train_points, device)`**

Creates MNIST dataset with flattened continuous pixel values.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| train_points | int | 1000 | Number of training examples |
| device | str | 'cpu' | Tensor device |

**Returns**: `(train_data, train_labels, test_data, test_labels)`
- `train_data`: [N_train, 784] flattened images (float32)
- `test_data`: [10000, 784] full test set

---

**`create_onehot_composition_dataset(vocab_size, seq_len, n_facts, train_fraction, device)`**

Creates compositional reasoning dataset with concatenated one-hot tokens.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| vocab_size | int | 100 | Vocabulary size |
| seq_len | int | 10 | Sequence length |
| n_facts | int | 500 | Total examples |
| train_fraction | float | 0.3 | Training set fraction |

**Encoding scheme**: Sequence [t₁, t₂, ...] → vector of size vocab_size × seq_len where each position contains one-hot encoding of corresponding token.

### Usage Example

```python
from onehot_datasets import create_onehot_modular_dataset

train_X, train_y, test_X, test_y = create_onehot_modular_dataset(
    p=113,
    operation='add',
    train_fraction=0.3,
    device='cuda'
)

print(f"Training data shape: {train_X.shape}")  # [~3830, 226]
print(f"Input dtype: {train_X.dtype}")          # torch.float32
print(f"One-hot check: {train_X[0].sum()}")     # 2.0 (two positions set)
```

---

## onehot_models.py

Model architectures accepting continuous one-hot inputs instead of discrete token indices.

### Design Rationale

Standard transformer models use `nn.Embedding`, which requires integer indices and is not differentiable with respect to inputs. These models replace embeddings with `nn.Linear` projections, enabling input-gradient AGOP computation while preserving architectural properties.

### Class: ModularArithmeticMLP

Simple MLP for modular arithmetic tasks.

```python
class ModularArithmeticMLP(nn.Module):
    def __init__(
        self,
        p: int,                  # Modulus
        hidden_dim: int = 128,   # Hidden layer size
        dropout: float = 0.0     # Dropout probability
    )
```

**Architecture**:
```
Input [B, 2p] → Linear(2p, hidden_dim) → ReLU → Dropout → Linear(hidden_dim, p) → Output [B, p]
```

---

### Class: MNISTModel

MLP for MNIST following Omnigrok paper specifications.

```python
class MNISTModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 200,
        output_dim: int = 10,
        depth: int = 3,
        activation: str = 'relu',        # 'relu', 'tanh', or 'gelu'
        initialization_scale: float = 8.0
    )
```

**Architecture**:
```
Flatten → [Linear → Activation] × (depth-1) → Linear → Output
```

**Initialization**: All parameters scaled by `initialization_scale` (default 8.0) to induce grokking.

---

### Class: OneHotReLUTransformer

Nanda-style one-layer ReLU transformer adapted for one-hot inputs.

```python
class OneHotReLUTransformer(nn.Module):
    def __init__(
        self,
        p: int,                  # Modulus
        d_model: int = 128,      # Model dimension
        n_heads: int = 4,        # Attention heads
        d_mlp: int = 512         # MLP hidden dimension
    )
```

**Architecture**:
```
Input [B, 2p] → Linear(2p, d_model) → [unsqueeze] → 
ReLU Attention + Residual → MLP + Residual → [squeeze] → 
Linear(d_model, p) → Output [B, p]
```

**Key differences from standard transformer**:
- ReLU attention instead of softmax
- No layer normalization
- Single layer

**ReLU Attention** (from `ReLUAttention` class):
```python
scores = Q @ K^T / sqrt(d_head)
attn_weights = ReLU(scores)  # Not softmax!
output = attn_weights @ V
```

---

### Class: OneHotStandardTransformer

Standard softmax transformer for architecture comparison.

```python
class OneHotStandardTransformer(nn.Module):
    def __init__(
        self,
        p: int,                  # Modulus
        d_model: int = 128,      # Model dimension
        n_heads: int = 4,        # Attention heads
        n_layers: int = 2,       # Transformer layers
        d_ff: int = 512          # Feedforward dimension
    )
```

**Architecture**:
```
Input [B, 2p] → Linear(2p, d_model) → [unsqueeze] → 
TransformerEncoder (n_layers) → [squeeze] → 
Linear(d_model, p) → Output [B, p]
```

Uses PyTorch's `nn.TransformerEncoderLayer` with:
- Softmax attention
- GELU activation
- Post-norm layer normalization
- Dropout = 0.0

---

### Class: CompositionMLP

MLP for compositional reasoning with sequence inputs.

```python
class CompositionMLP(nn.Module):
    def __init__(
        self,
        vocab_size: int,         # Vocabulary size
        seq_len: int,            # Sequence length
        hidden_dim: int = 256,   # Hidden dimension
        n_layers: int = 2        # Number of hidden layers
    )
```

**Architecture**:
```
Input [B, vocab_size × seq_len] → Linear → ReLU → 
[Linear → ReLU] × (n_layers-1) → Linear(hidden_dim, vocab_size) → Output [B, vocab_size]
```

---

### Usage Example

```python
from onehot_models import ModularArithmeticMLP, OneHotReLUTransformer

# Create MLP
mlp = ModularArithmeticMLP(p=113, hidden_dim=128)
print(f"MLP parameters: {sum(p.numel() for p in mlp.parameters()):,}")

# Create Transformer
transformer = OneHotReLUTransformer(p=113, d_model=128, n_heads=4, d_mlp=512)
print(f"Transformer parameters: {sum(p.numel() for p in transformer.parameters()):,}")

# Forward pass (one-hot inputs)
x = torch.randn(32, 226)  # [batch_size, 2*p]
logits_mlp = mlp(x)       # [32, 113]
logits_trans = transformer(x)  # [32, 113]
```

---

## Testing

Each module includes a test function that can be run directly:

```bash
# Test AGOP utilities
python agop_utils.py

# Test Lazy-Rich tracker
python lazy_rich_utils.py

# Test datasets
python onehot_datasets.py

# Test models
python onehot_models.py
```

All tests validate:
- Correct tensor shapes
- Proper data types (float32 for continuous inputs)
- Expected metric ranges
- Forward pass compatibility

---

## References

1. **AGOP Theory**: Beaglehole, D., et al. (2023). "Average gradient outer product as a mechanism for deep neural collapse."
2. **Lazy-Rich Dynamics**: Kumar, A., et al. (2024). "Grokking as the Transition from Lazy to Rich Training Dynamics." arXiv:2310.06110
3. **ReLU Transformer**: Nanda, N., et al. (2023). "Progress measures for grokking via mechanistic interpretability."
4. **Omnigrok**: Liu, Z., et al. (2022). "Omnigrok: Grokking Beyond Algorithmic Data."

---

*Last Updated: December 2024*







