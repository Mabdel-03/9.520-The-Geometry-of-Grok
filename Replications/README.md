# Grokking Papers Replications

This directory contains replications of 10 major papers on the grokking phenomenon in neural networks. Each subdirectory includes code, SLURM batch scripts for HPC execution, and documentation.

## Overview

The goal is to replicate these papers to:
1. Observe grokking across different datasets, architectures, and optimizers
2. Analyze gradient outer product dynamics during the transition from memorization to generalization
3. Understand the geometry of learned functions before and after grokking

## Papers Replicated

### 1. Power et al. (2022) - Original Grokking Paper
**Directory:** `01_power_et_al_2022_openai_grok/`
- **Paper:** [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177)
- **Datasets:** Modular arithmetic, permutation groups
- **Architecture:** 2-layer transformer
- **Status:** ✅ Cloned from GitHub

### 2. Liu et al. (2022) - Effective Theory
**Directory:** `02_liu_et_al_2022_effective_theory/`
- **Paper:** [Towards Understanding Grokking: An Effective Theory of Representation Learning](https://arxiv.org/abs/2205.10343)
- **Datasets:** Toy models, modular addition, S_3, MNIST
- **Key Contribution:** Phase diagrams, Representation Quality Index
- **Status:** ✅ Cloned from GitHub

### 3. Nanda et al. (2023) - Mechanistic Interpretability
**Directory:** `03_nanda_et_al_2023_progress_measures/`
- **Paper:** [Progress Measures for Grokking via Mechanistic Interpretability](https://arxiv.org/abs/2301.05217)
- **Dataset:** Modular addition (mod 113)
- **Architecture:** 1-layer ReLU transformer
- **Key Contribution:** Fourier multiplication algorithm discovery
- **Status:** ✅ Implemented from scratch

### 4. Wang et al. (2024) - Implicit Reasoning
**Directory:** `04_wang_et_al_2024_implicit_reasoners/`
- **Paper:** [Grokked Transformers are Implicit Reasoners](https://arxiv.org/abs/2405.15071)
- **Datasets:** Knowledge graph reasoning (composition, comparison)
- **Architecture:** GPT-2 style transformer (8 layers)
- **Status:** ✅ Cloned from GitHub

### 5. Liu et al. (2022) - Omnigrok
**Directory:** `05_liu_et_al_2022_omnigrok/`
- **Paper:** [Omnigrok: Grokking Beyond Algorithmic Data](https://arxiv.org/abs/2210.01117)
- **Datasets:** MNIST, IMDb, QM9, modular addition
- **Key Contribution:** LU mechanism, extends grokking to diverse domains
- **Status:** ✅ Cloned from GitHub

### 6. Humayun et al. (2024) - Deep Networks Always Grok
**Directory:** `06_humayun_et_al_2024_deep_networks/`
- **Paper:** [Deep Networks Always Grok and Here is Why](https://arxiv.org/abs/2402.15555)
- **Datasets:** MNIST, CIFAR-10/100, Imagenette, Shakespeare
- **Architectures:** MLP, CNN, ResNet-18, GPT
- **Key Contribution:** Delayed adversarial robustness, local complexity
- **Status:** ✅ Implemented from scratch

### 7. Thilak et al. (2022) - Slingshot Mechanism
**Directory:** `07_thilak_et_al_2022_slingshot/`
- **Paper:** [The Slingshot Mechanism](https://arxiv.org/abs/2206.04817)
- **Dataset:** Modular arithmetic
- **Key Contribution:** Cyclic dynamics in Adam optimizers
- **Status:** ✅ Implemented from scratch

### 8. Doshi et al. (2024) - Modular Polynomials
**Directory:** `08_doshi_et_al_2024_modular_polynomials/`
- **Paper:** [Grokking Modular Polynomials](https://arxiv.org/abs/2406.03495)
- **Datasets:** Modular addition/multiplication with power activation
- **Architecture:** 2-layer MLP with $\phi(x) = x^S$
- **Key Contribution:** Analytical solutions for grokking
- **Status:** ✅ Implemented from scratch

### 9. Levi et al. (2023) - Linear Estimators
**Directory:** `09_levi_et_al_2023_linear_estimators/`
- **Paper:** [Grokking in Linear Estimators](https://arxiv.org/abs/2310.16441)
- **Dataset:** Gaussian teacher-student
- **Architecture:** Linear networks (1-layer, 2-layer)
- **Key Contribution:** Grokking without understanding (threshold artifact)
- **Status:** ✅ Implemented from scratch

### 10. Minegishi et al. (2023) - Grokking Tickets
**Directory:** `10_minegishi_et_al_2023_grokking_tickets/`
- **Paper:** [Grokking Tickets: Lottery Tickets Accelerate Grokking](https://arxiv.org/abs/2310.19470)
- **Datasets:** Modular addition, MNIST
- **Key Contribution:** Sparse subnetworks accelerate grokking
- **Status:** ✅ Cloned from GitHub

## Directory Structure

Each paper's directory contains:
```
XX_paper_name/
├── README.md or REPLICATION_README.md  # Setup and usage instructions
├── requirements.txt                     # Python dependencies
├── run_*.sh                            # SLURM batch scripts
├── train.py or similar                 # Training scripts
├── model.py or similar                 # Model definitions
├── logs/                               # Training logs (created at runtime)
└── checkpoints/                        # Model checkpoints (created at runtime)
```

## Running Experiments on HPC

### General Workflow

1. **Navigate to paper directory:**
   ```bash
   cd Replications/XX_paper_name/
   ```

2. **Make SLURM script executable:**
   ```bash
   chmod +x run_*.sh
   ```

3. **Modify SLURM parameters if needed:**
   Edit the `#SBATCH` directives in the `.sh` file for your cluster's configuration:
   - Partition names
   - Time limits
   - Memory requirements
   - GPU types

4. **Submit job:**
   ```bash
   sbatch run_*.sh
   ```

5. **Monitor job:**
   ```bash
   squeue -u $USER
   tail -f logs/job_name_*.out
   ```

### Example: Running All Papers

```bash
cd /path/to/9.520-The-Geometry-of-Grok/Replications

# Submit all main experiments
for dir in */; do
    cd "$dir"
    if [ -f run_*.sh ]; then
        chmod +x run_*.sh
        sbatch run_*.sh
    fi
    cd ..
done
```

## Common Module Loads (Adjust for Your HPC)

Most SLURM scripts assume:
```bash
module load python/3.9
module load cuda/11.8
```

Modify these in each `run_*.sh` script according to your cluster's available modules.

## Dataset Coverage

### Algorithmic Tasks
- Modular addition, subtraction, division, multiplication
- Permutation groups (S_3, S_5)
- Modular polynomials

### Image Classification
- MNIST (full and reduced)
- CIFAR-10, CIFAR-100
- Imagenette

### Natural Language
- IMDb sentiment analysis
- Shakespeare text generation

### Structured Data
- QM9 molecular properties
- Knowledge graph reasoning

### Synthetic
- Gaussian teacher-student setups

## For Gradient Outer Product Analysis

All implementations save model checkpoints regularly. To analyze gradient outer products:

1. **Load checkpoints** from different epochs (pre-grokking, during, post-grokking)
2. **Compute gradients** on train/test batches
3. **Form outer products** $\nabla \theta \nabla \theta^T$
4. **Track evolution** of eigenvalues, eigenvectors, or other metrics

Example analysis script structure:
```python
import torch

checkpoint = torch.load('checkpoints/checkpoint_epoch_10000.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Compute gradients
for x, y in data_loader:
    logits = model(x)
    loss = criterion(logits, y)
    grads = torch.autograd.grad(loss, model.parameters())
    
    # Form outer product and analyze
    ...
```

## Key Hyperparameters Across Papers

| Paper | Weight Decay | Optimizer | Learning Rate | Critical for Grokking? |
|-------|--------------|-----------|---------------|------------------------|
| 1. Power et al. | 1.0 | AdamW | 1e-3 | Yes (weight decay) |
| 2. Liu et al. | 0-20 | Adam/AdamW | 1e-3 | Yes (weight decay) |
| 3. Nanda et al. | 1.0 | AdamW | 1e-3 | Yes (weight decay) |
| 4. Wang et al. | 0.1 | AdamW | 1e-4 | Yes (weight decay) |
| 5. Liu et al. | 0.01-1.0 | Adam | 1e-3 | Varies by dataset |
| 6. Humayun et al. | 0-0.01 | Adam | 1e-3 | No (occurs anyway) |
| 7. Thilak et al. | 0.0 | Adam | 1e-3 | No (Slingshot without WD) |
| 8. Doshi et al. | 5.0 | Adam | 5e-3 | Yes (high weight decay) |
| 9. Levi et al. | 0.01 | SGD | 1e-2 | Yes (weight decay) |
| 10. Minegishi et al. | 0.0-1.0 | Adam/AdamW | 1e-3 | No (with good tickets) |

## Notes

- **Relative paths:** All code uses relative paths for portability between local and HPC environments
- **GPU requirements:** Most experiments run on single GPU; some (Wang et al.) may benefit from multi-GPU
- **Training time:** Varies from hours (linear estimators) to days (knowledge graph reasoning)
- **Reproducibility:** Random seeds are set but GPU non-determinism may cause minor variations

## Citation

If you use these replications, please cite the original papers (see individual READMEs) and this repository.

## Contact

For questions about this replication suite, refer to individual paper READMEs or the main project repository.

