# Grokking in Linear Estimators: A Solvable Model that Groks without Understanding

**Authors:** Noam Levi, Alon Beck, Yohai Bar-Sinai

**Paper:** [arXiv:2310.16441](https://arxiv.org/abs/2310.16441)

## Summary

This paper demonstrates that grokking can occur even in simple linear networks performing linear tasks in a teacher-student setup with Gaussian inputs. The authors derive exact analytical solutions for training dynamics in terms of covariance matrices. The key insight is that grokking does not necessarily represent "understanding" but is simply an artifact of the accuracy metric crossing a threshold.

## Dataset

**Synthetic Gaussian Teacher-Student:**
- **Inputs:** x_i ~ N(0, I_{d_in × d_in})
- **Teacher:** Generates labels y_i = T^T x_i where T ∈ ℝ^{d_in × d_out}
- **Training samples:** N_tr varied
- **Test samples:** N_gen independent samples
- **Key ratio:** λ = d_in/N_tr (controls underdetermined vs overdetermined regime)

## Model Architectures

### 1-Layer Linear Network
- Student matrix S ∈ ℝ^{d_in × d_out}, no biases
- Direct mapping from input to output

### 2-Layer Linear Network
- S = S_0 S_1 with S_0 ∈ ℝ^{d_in × d_h}, S_1 ∈ ℝ^{d_h × d_out}

### 2-Layer with Nonlinearity
- Same as 2-layer linear but with tanh activation

## Training Hyperparameters

- **Optimizer:** Full batch gradient descent (GD) in gradient flow limit
- **Learning rate:** Small (0.01 typical), approximating continuous gradient flow
- **Weight decay:** γ ∈ [10^{-5}, 1]
- **Loss function:** MSE: L = (1/N_tr d_out) Σ_i ||(S-T)^T x_i||^2
- **Accuracy threshold:** ε (sample correct if error < ε), typically 10^{-3} or 10^{-6}
- **Training:** Up to 10^5 epochs

## Setup and Installation

```bash
pip install -r requirements.txt
```

## Running Experiments

### 1-Layer Linear (SLURM)

```bash
chmod +x run_linear_1layer.sh
sbatch run_linear_1layer.sh
```

### 1-Layer Linear (Local)

```bash
python train.py \
    --architecture=1layer \
    --d_in=1000 \
    --d_out=1 \
    --n_train=500 \
    --n_test=10000 \
    --lr=0.01 \
    --weight_decay=0.01 \
    --n_epochs=100000 \
    --accuracy_threshold=1e-3 \
    --device=cuda
```

### 2-Layer Linear

```bash
python train.py \
    --architecture=2layer_linear \
    --d_in=1000 \
    --d_h=500 \
    --d_out=1 \
    --n_train=500 \
    --weight_decay=0.01 \
    --device=cuda
```

### Exploring Different Regimes

**Overdetermined (λ < 1):**
```bash
python train.py --d_in=500 --n_train=1000  # λ = 0.5
```

**Underdetermined (λ > 1):**
```bash
python train.py --d_in=1000 --n_train=500  # λ = 2.0
```

## Expected Results

- **Loss decreases smoothly** while **accuracy jumps suddenly** (when loss crosses threshold)
- Grokking time depends on:
  - Weight decay γ
  - Data ratio λ = d_in/N_tr
  - Accuracy threshold ε
- **Key insight:** The "grokking" is not a phase transition in learning but an artifact of discrete accuracy measurement

## Output Files

- `logs/training_history.json`: Loss, accuracy, generalization loss over time
- `checkpoints/checkpoint_epoch_*.pt`: Includes student weights and teacher matrix

## Theoretical Analysis

This implementation provides empirical validation of the paper's analytical predictions. You can compare:
1. Training dynamics to theoretical covariance matrix evolution
2. Grokking time to analytical predictions
3. Effect of hyperparameters on grokking onset

## Citation

```bibtex
@article{levi2023grokking,
  title={Grokking in Linear Estimators: A Solvable Model that Groks without Understanding},
  author={Levi, Noam and Beck, Alon and Bar-Sinai, Yohai},
  journal={arXiv preprint arXiv:2310.16441},
  year={2023}
}
```
