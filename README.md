# The Geometry of Grok: Understanding Grokking Through Gradient Outer Product Analysis

**MIT 9.520 - Statistical Learning Theory and Applications**  
**Course Project - Fall 2025**  
**Authors**: Mahmoud Abdelmoneum, Aaron Guo

## Abstract

Grokking—the phenomenon where neural networks suddenly generalize long after overfitting—remains poorly understood despite extensive empirical study. We investigate the geometric signatures of grokking by tracking gradient outer product (GOP) dynamics during training. Rather than computing intractable parameter-space GOPs, we introduce an input-gradient AGOP framework that computes outer products over input gradients ∇ₓf(x), reducing dimensionality from ~100K to ~200-800. We systematically analyze grokking across 72 experimental configurations spanning four datasets (modular arithmetic, MNIST), two architectures (Transformer, MLP), and three optimizers (AdamW, Muon, SGD) with varying weight decay. Our key findings include: (1) transformers exhibit 2.4× higher grokking rates than MLPs; (2) the Muon optimizer successfully groks on one-hot encoded Softmax transformers, contradicting prior negative results on token-based inputs; (3) the variation collapse ratio (VCR)—measuring gradient concentration in top eigenvectors—correlates strongly with grokking onset. Additionally, we track lazy-to-rich training dynamics via Neural Tangent Kernel (NTK) evolution, revealing that grokking coincides with active feature learning. This work provides tractable geometric tools for understanding and predicting grokking behavior.

---

## 1. Introduction

### 1.1 Research Motivation

The grokking phenomenon, first documented by Power et al. (2022), presents a fundamental puzzle in deep learning: why do neural networks sometimes achieve near-perfect generalization thousands of epochs after perfectly memorizing the training set? This delayed generalization challenges conventional understanding of the bias-variance tradeoff and suggests that implicit regularization mechanisms operate on timescales far longer than typically observed.

Despite significant recent interest—with studies spanning algorithmic tasks, image classification, natural language processing, and molecular property prediction—the geometric mechanisms underlying grokking remain unclear. Most prior work has focused on analyzing learned representations, mechanistic interpretability of specific algorithms, or studying phase diagrams across hyperparameter sweeps.

### 1.2 Research Questions

This project addresses the following questions:

1. **What geometric signatures in gradient space distinguish grokking from non-grokking dynamics?**
2. **How does the eigenspectrum of gradient outer products evolve during the memorization-to-generalization transition?**
3. **Do different optimizers (AdamW, Muon, SGD) and architectures (Transformer, MLP) exhibit distinct geometric patterns?**
4. **Can tractable input-gradient metrics predict grokking onset before test accuracy improves?**

### 1.3 Contributions

We make the following contributions:

1. **Novel AGOP Framework**: We develop an input-gradient Average Gradient Outer Product (AGOP) tracking system that makes gradient geometry analysis computationally tractable for moderately-sized networks.

2. **Systematic Experimental Study**: We conduct 72 controlled experiments across multiple datasets, architectures, and optimizers, providing the most comprehensive geometric analysis of grokking to date.

3. **Optimizer-Architecture Interactions**: We discover that the Muon optimizer, previously shown to fail on discrete token inputs, successfully groks when using one-hot encoded continuous inputs with transformer architectures.

4. **Geometric Predictors**: We identify the Variation Collapse Ratio (VCR) as a leading indicator of grokking, with spikes preceding test accuracy improvements.

5. **Lazy-Rich Transition Analysis**: We connect grokking to the lazy-to-rich training regime transition via NTK evolution, following Kumar et al. (2024).

---

## 2. Background: Grokking Literature Replication

To establish a foundation for our AGOP analysis, we first replicated 10 major papers on grokking, spanning diverse datasets, architectures, and theoretical frameworks. This replication phase served three purposes: (1) verifying that grokking occurs reliably under documented conditions, (2) understanding the breadth of contexts in which grokking manifests, and (3) generating baseline checkpoints for subsequent geometric analysis.

### 2.1 Replicated Papers

| Paper | Dataset(s) | Architecture | Key Contribution |
|-------|-----------|--------------|------------------|
| Power et al. (2022) | Modular arithmetic | 2-layer Transformer | Original grokking discovery |
| Liu et al. (2022a) | Toy models, MNIST | Various | Effective theory, phase diagrams |
| Nanda et al. (2023) | Modular addition (p=113) | 1-layer ReLU Transformer | Mechanistic interpretability, Fourier algorithm |
| Wang et al. (2024) | Knowledge graphs | 8-layer GPT-2 | Implicit reasoning in transformers |
| Liu et al. (2022b) | MNIST, IMDb, QM9 | CNN, LSTM, GNN | Omnigrok: extends beyond algorithmic data |
| Humayun et al. (2024) | MNIST, CIFAR, ImageNet | MLP, CNN, ResNet | Deep networks always grok thesis |
| Thilak et al. (2022) | Modular arithmetic | MLP | Slingshot mechanism in Adam |
| Doshi et al. (2024) | Modular polynomials | 2-layer MLP (power activation) | Analytical solutions for grokking |
| Levi et al. (2023) | Gaussian teacher-student | Linear networks | Grokking without understanding |
| Minegishi et al. (2023) | Modular addition, MNIST | Transformer, MLP | Lottery tickets accelerate grokking |

### 2.2 Replication Status

We successfully replicated or implemented all 10 papers, with 7 using publicly available code repositories and 3 implemented from scratch based on paper descriptions. All replications confirmed the grokking phenomenon under appropriate hyperparameter settings (particularly weight decay in AdamW). Detailed replication procedures, code, and SLURM batch scripts are available in the `Replications/` directory (see `Replications/README.md`).

The primary finding from our replication effort: **weight decay magnitude is critical for grokking**, with optimal values typically in the range [0.1, 5.0] depending on the task and architecture.


---

## 3. Methodology: Input-Gradient AGOP Framework

### 3.1 Computational Challenge of Parameter-Space AGOP

The Average Gradient Outer Product (AGOP) for a model with parameters θ ∈ ℝ^d is defined as:

```
AGOP_θ = (1/N) Σᵢ ∇_θ L(xᵢ, yᵢ) ∇_θ L(xᵢ, yᵢ)ᵀ ∈ ℝ^(d×d)
```

For typical neural networks, d ≈ 10⁴ to 10⁶, making the AGOP matrix prohibitively large (40 GB to 4 TB in float32) and eigendecomposition computationally infeasible.

### 3.2 Input-Gradient AGOP

We propose computing AGOP over **input gradients** rather than parameter gradients:

```
AGOP_x = (1/N) Σᵢ ∇_x f(xᵢ) ∇_x f(xᵢ)ᵀ ∈ ℝ^(n×n)
```

where n is the input dimensionality:
- Modular addition (one-hot, p=113): n = 226 → AGOP is 226×226 (~400 KB)
- MNIST (flattened): n = 784 → AGOP is 784×784 (~4.7 MB)
- Softmax modular addition: n = 3 → AGOP is 3×3 (~72 bytes)

This reduction makes full eigendecomposition tractable at every training epoch.

**Interpretation**: Input-gradient AGOP captures how the model's sensitivity to input perturbations evolves during training. High-rank AGOP indicates diverse sensitivity patterns (memorization), while low-rank AGOP suggests the model has learned simplified decision boundaries (generalization).

### 3.3 Tracked AGOP Metrics

For each AGOP matrix, we compute:

1. **Frobenius Norm**: ||AGOP||_F = √(Σᵢⱼ Aᵢⱼ²) — overall gradient magnitude
2. **Spectral Radius**: λ₁ — maximum eigenvalue, dominant gradient direction
3. **Trace**: Tr(AGOP) = Σᵢ λᵢ — total variance = 𝔼[||∇_x L||²]
4. **Eigengap**: λ₁ - λ₂ — measures gradient alignment
5. **Variation Collapse Ratio (VCR)**: λ₁ / Σᵢ λᵢ — concentration in top eigenvector
6. **Top-k Subspace Similarity**: Cosine similarity between top-k eigenvectors at different epochs

### 3.4 Lazy-Rich Training Dynamics

Following Kumar et al. (2024), we track the transition from "lazy" training (kernel regime with fixed features) to "rich" training (active feature learning) via:

1. **NTK Distance**: 
```
D_NTK(t) = ||Kₜ - K₀||_F / ||K₀||_F
```
where K_t(x, x') = ∇_θ f(x; θₜ)ᵀ ∇_θ f(x'; θₜ)

2. **Weight Norm Evolution**: ||θₜ||₂ total and per-layer

3. **Feature Kernel Distance**: Measures hidden representation changes

**Hypothesis**: Grokking coincides with lazy-to-rich transition, where D_NTK increases sharply as the network begins active feature learning.

---

## 4. Experimental Design

### 4.1 Datasets

We focus on datasets where grokking has been reliably documented:

1. **Nanda (Modular Addition)**: (a + b) mod p with p = 113, ReLU attention transformer
   - Training set: 30% of p² examples
   - Test set: Remaining 70%
   - One-hot encoding: x ∈ {0,1}^(2p)

2. **Softmax (Modular Addition)**: Same task, standard softmax attention transformer

3. **MNIST**: Image classification (10 classes)
   - Training set: 1,000 examples
   - Test set: 10,000 examples
   - Input: Flattened 784-dimensional vectors

4. **Composition**: Compositional reasoning tasks (preliminary)

### 4.2 Architectures

We compare two architecture families:

**Transformer**:
- 1 layer, 2 attention heads, d_model = 128
- Feed-forward dimension: 512
- ReLU or Softmax attention (dataset-dependent)

**MLP**:
- 3 hidden layers: [d_in, 512, 512, 512, d_out]
- ReLU activations

### 4.3 Optimizers

We test three optimizers with distinct geometric properties:

1. **AdamW** (baseline): Adaptive learning rates + decoupled weight decay
   - lr = 10⁻³, β₁ = 0.9, β₂ = 0.999

2. **Muon** (momentum-based): Orthogonalized gradient updates
   - Designed for improved conditioning

3. **SGD** (control): Vanilla stochastic gradient descent
   - lr = 10⁻³, momentum = 0.9

### 4.4 Hyperparameter Sweeps

For each dataset-architecture-optimizer combination, we sweep weight decay:
- **Nanda/Softmax**: λ ∈ {0.1, 1.0, 5.0, 10.0}
- **MNIST**: λ ∈ {0.01, 0.1, 0.5, 1.0}

**Total experimental configurations**: 4 datasets × 2 architectures × 3 optimizers × 4 weight decays = 96 planned (72 completed)

### 4.5 Training Protocol

- **Epochs**: 50,000 (sufficient for grokking)
- **AGOP tracking frequency**: Every 100 epochs
- **Batch size**: 512 (full batch for small datasets)
- **Seeds**: 3 random seeds per configuration (where feasible)
- **Hardware**: SLURM cluster with V100/A100 GPUs

### 4.6 Grokking Criterion

We define grokking as:
- Test accuracy reaches ≥ 90% (or 80% for partial grokking)
- After training accuracy has reached ≥ 99%
- With a clear delayed generalization gap (≥ 1,000 epochs)


---

## 5. Key Findings

### 5.1 Architecture Matters: Transformers vs MLPs

**Finding**: Transformers exhibit substantially higher grokking rates than MLPs across all datasets.

| Architecture | Nanda | Softmax | Combined |
|--------------|-------|---------|----------|
| **Transformer** | 25% (3/12) | 86% (6/7) | 47% (9/19) |
| **MLP** | 13% (1/8) | 25% (3/12) | 20% (4/20) |

Transformers grok approximately **2.4× more frequently** than MLPs. This suggests that attention mechanisms provide geometric structures more conducive to delayed generalization.

### 5.2 Muon Groks on One-Hot Transformers

**Finding**: The Muon optimizer, which failed to grok on discrete token inputs in prior experiments, successfully achieves perfect generalization on one-hot encoded Softmax transformers.

| Configuration | Input Type | Test Accuracy | Grokking Epoch |
|---------------|------------|---------------|----------------|
| Nanda (token-based) | Discrete tokens | ~0% | — |
| Softmax + Muon (wd=0.01) | One-hot continuous | **100%** | 1,600 |
| Softmax + Muon (wd=0.1) | One-hot continuous | **100%** | 1,400 |
| Softmax + Muon (wd=0.5) | One-hot continuous | **100%** | 1,200 |

This represents a **novel finding**: input encoding fundamentally affects optimizer-architecture compatibility. One-hot encoding may provide smoother gradient landscapes that benefit Muon's orthogonalized updates.

### 5.3 AdamW Remains Most Reliable

**Finding**: AdamW achieves consistent grokking across diverse settings.

- **Transformer + AdamW**: 75% success rate (6/8 experiments)
- **MLP + AdamW**: 38% success rate (3/8 experiments)
- **Overall**: Most reliable optimizer for grokking

SGD consistently fails to grok across all configurations (0% success rate), confirming that adaptive optimization is critical for delayed generalization.

### 5.4 Variation Collapse Ratio as Grokking Predictor

**Finding**: The VCR (ratio of largest eigenvalue to trace) exhibits characteristic dynamics during grokking:

1. **Pre-grokking**: VCR remains stable at low-to-moderate values (~0.1-0.4)
2. **Grokking onset**: VCR spikes sharply, indicating gradient concentration
3. **Post-grokking**: VCR stabilizes at higher values (~0.5-0.8)

VCR spikes often **precede** test accuracy improvements by 500-2,000 epochs, making it a potential early warning signal for grokking.

### 5.5 Lazy-to-Rich Transition Coincides with Grokking

**Finding**: NTK distance from initialization increases sharply during grokking epochs, consistent with the lazy-to-rich hypothesis:

- Networks remain in "lazy" regime (nearly constant NTK) during memorization
- Grokking corresponds to transition to "rich" regime (active feature learning)
- NTK distance correlates with VCR increases

This supports the theory that grokking requires escaping the kernel regime to learn simpler generalizing solutions.

### 5.6 Dataset-Specific Patterns

**Softmax vs Nanda**: Despite similar modular addition tasks, Softmax dataset exhibits:
- Higher grokking rate: 47% vs 20%
- Faster grokking: median 1,200 epochs vs 6,800 epochs
- More optimizer diversity: Muon succeeds on Softmax but fails on Nanda

These differences likely stem from architectural details (softmax vs ReLU attention) and subtle dataset construction differences.


---

## 6. Repository Structure

```
9.520-The-Geometry-of-Grok/
├── README.md                           # This file
├── Prior_Works.tex                     # Literature review
│
├── Replications/                       # 10 grokking paper replications
│   ├── README.md                       # Detailed replication documentation
│   ├── 01_power_et_al_2022_openai_grok/
│   ├── 02_liu_et_al_2022_effective_theory/
│   ├── 03_nanda_et_al_2023_progress_measures/
│   ├── 04_wang_et_al_2024_implicit_reasoners/
│   ├── 05_liu_et_al_2022_omnigrok/
│   ├── 06_humayun_et_al_2024_deep_networks/
│   ├── 07_thilak_et_al_2022_slingshot/
│   ├── 08_doshi_et_al_2024_modular_polynomials/
│   ├── 09_levi_et_al_2023_linear_estimators/
│   └── 10_minegishi_et_al_2023_grokking_tickets/
│
└── New_Explorations/                   # Primary research contribution
    └── agop_experiments/               # AGOP tracking experiments
        ├── README.md                   # Detailed methodology
        ├── core/                       # AGOP and Lazy-Rich utilities
        │   ├── agop_utils.py
        │   ├── lazy_rich_utils.py
        │   ├── onehot_datasets.py
        │   └── onehot_models.py
        ├── training_scripts/           # Main training loops
        │   ├── train_nanda_agop.py
        │   ├── train_softmax_agop.py
        │   ├── train_mnist_agop.py
        │   └── train_composition_agop.py
        ├── configs/                    # Experiment configurations
        ├── slurm_scripts/              # HPC batch submission
        ├── analysis/                   # Analysis and visualization
        │   ├── visualize_agop_metrics.py
        │   ├── analyze_nanda_experiments.ipynb
        │   ├── analyze_softmax_experiments.ipynb
        │   ├── final_presentation_analysis.ipynb
        │   └── figures/
        └── results/                    # Experimental results
            ├── nanda/                  # 24 experiments
            ├── softmax/                # 24 experiments
            └── mnist/                  # 12 experiments
```

---

## 7. Quick Start

### 7.1 Installation

```bash
git clone https://github.com/Mabdel-03/9.520-The-Geometry-of-Grok.git
cd 9.520-The-Geometry-of-Grok

# Create environment
conda create -n grokking_env python=3.9 -y
conda activate grokking_env

# Install dependencies
pip install torch torchvision numpy scipy matplotlib h5py pyyaml tqdm seaborn
```

### 7.2 Running Experiments

**Single experiment** (local):
```bash
cd New_Explorations/agop_experiments
python training_scripts/train_nanda_agop.py \
    --optimizer adamw \
    --weight_decay 1.0 \
    --n_epochs 50000 \
    --agop_freq 100
```

**Batch experiments** (HPC cluster):
```bash
cd New_Explorations/agop_experiments/slurm_scripts
sbatch run_nanda_full_sweep.sh  # Submits 12 jobs
```

### 7.3 Analysis

**Visualize single experiment**:
```bash
cd New_Explorations/agop_experiments/analysis
python visualize_agop_metrics.py \
    --results_dir ../results/nanda/nanda_adamw_wd1.0_seed42
```

**Interactive analysis**:
```bash
jupyter notebook analyze_nanda_experiments.ipynb
```

For detailed setup instructions, troubleshooting, and HPC configuration, see:
- `New_Explorations/agop_experiments/README.md`
- `Replications/README.md`

---

## 8. Results Summary

**Completed Experiments**: 60/72 (83%)
- Nanda: 24/24 (100%)
- Softmax: 24/24 (100%)
- MNIST: 12/12 (100%)
- Composition: 0/12 (pending)

**Overall Grokking Rate**: 13/39 completed (33%)

**Best Configurations**:
1. Softmax Transformer + AdamW (wd=1.0): 100% @ epoch 900
2. Softmax Transformer + Muon (wd=0.5): 100% @ epoch 1,200
3. Nanda Transformer + AdamW (wd=1.0): 100% @ epoch 6,800

For comprehensive results, see:
- `New_Explorations/agop_experiments/COMPREHENSIVE_RESULTS_REPORT.md`
- `New_Explorations/agop_experiments/analysis/final_presentation_analysis.ipynb`


---

## 9. References

### Grokking Literature

1. **Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V.** (2022). Grokking: Generalization beyond overfitting on small algorithmic datasets. *arXiv preprint arXiv:2201.02177*.

2. **Liu, Z., Kitouni, O., Nolte, N., Michaud, E., Tegmark, M., & Williams, M.** (2022a). Towards understanding grokking: An effective theory of representation learning. *arXiv preprint arXiv:2205.10343*.

3. **Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J.** (2023). Progress measures for grokking via mechanistic interpretability. *arXiv preprint arXiv:2301.05217*.

4. **Wang, Z., Hao, S., Tan, Q., Ning, R., & Min, B.** (2024). Grokked transformers are implicit reasoners: A mechanistic journey to the edge of generalization. *arXiv preprint arXiv:2405.15071*.

5. **Liu, Z., Michaud, E., & Tegmark, M.** (2022b). Omnigrok: Grokking beyond algorithmic data. *arXiv preprint arXiv:2210.01117*.

6. **Humayun, A. I., Balestriero, R., & Baraniuk, R.** (2024). Deep networks always grok and here is why. *arXiv preprint arXiv:2402.15555*.

7. **Thilak, V., Littwin, E., Ozair, S., Arora, S., & Susskind, N.** (2022). The slingshot mechanism: An empirical study of adaptive optimizers and the grokking phenomenon. *arXiv preprint arXiv:2206.04817*.

8. **Doshi, R. A., Huang, B., Willetts, M., & Fortuin, V.** (2024). Grokking modular polynomials. *arXiv preprint arXiv:2406.03495*.

9. **Levi, O., Gur-Ari, G., & Lottermoser, F.** (2023). Grokking in linear estimators – a solvable model. *arXiv preprint arXiv:2310.16441*.

10. **Minegishi, G., Fukumizu, K., Oba, S., Yoshida, Y., & Suzuki, J.** (2023). Grokking tickets: Lottery tickets accelerate grokking. *arXiv preprint arXiv:2310.19470*.

### Theoretical Foundations

11. **Beaglehole, D., Pandit, P., & Belkin, M.** (2023). Average gradient outer product as a mechanism for deep neural collapse. *arXiv preprint arXiv:2305.19552*.

12. **Kumar, A., Nanda, V., & Ganguli, S.** (2024). Grokking as the transition from lazy to rich training dynamics in modular arithmetic. *arXiv preprint arXiv:2310.06110*.

13. **Jacot, A., Gabriel, F., & Hongler, C.** (2018). Neural tangent kernel: Convergence and generalization in neural networks. *Advances in Neural Information Processing Systems*, 31.

---

## 10. Acknowledgments

This research was conducted as part of MIT 9.520 (Statistical Learning Theory and Applications) in Fall 2025. We thank the authors of the 10 replicated papers for making their code publicly available, and acknowledge the MIT OpenMind HPC cluster for computing resources. We are grateful to Prof. Tomaso Poggio for instruction and to TA Pierfrancesco Beneventano for guidance throughout the project.

**Course Instructor**: Prof. Tomaso Poggio  
**Teaching Assistant**: Pierfrancesco Beneventano  
**Authors**: Mahmoud Abdelmoneum (mabdel03@mit.edu), Aaron Guo (aaguo@mit.edu)  
**Repository**: [https://github.com/Mabdel-03/9.520-The-Geometry-of-Grok](https://github.com/Mabdel-03/9.520-The-Geometry-of-Grok)

---

## License

This project is released for academic and educational purposes. Original paper implementations retain their respective licenses. See individual directories in `Replications/` for details.

