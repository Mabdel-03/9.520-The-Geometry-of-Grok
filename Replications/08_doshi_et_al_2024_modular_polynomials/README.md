# Grokking Modular Polynomials

**Authors:** Darshil Doshi, Tianyu He, Aritra Das, Andrey Gromov

**Paper:** [arXiv:2406.03495](https://arxiv.org/abs/2406.03495)

## Summary

This paper extends analytical solutions for grokking to modular multiplication and modular addition with multiple terms. The authors construct "expert" networks that can solve arbitrary modular polynomials by combining solutions for elementary operations. They provide exact analytical solutions for network weights that achieve perfect generalization.

## Datasets

- **Modular addition:** $(c_1n_1 + c_2n_2 + \cdots + c_Sn_S) \mod p$
- **Modular multiplication:** $(n_1^a \times n_2^b) \mod p$ where $a, b \in \mathbb{Z}_p \setminus \{0\}$
- **Primes used:** $p = 11, 23, 97$
- **Train/test split:** Typically 50/50

## Model Architecture

**2-layer MLP with power activation:**
- **Input:** One-hot encoded numbers
- **First layer:** Embedding matrices $U^{(1)}, \ldots, U^{(S)} \in \mathbb{R}^{N \times p}$
- **Activation:** $\phi(x) = x^S$ (element-wise power)
- **Second layer:** Output matrix $W \in \mathbb{R}^{N \times p}$
- **Output:** $f(e_{n_1}, \ldots, e_{n_S}) = W \cdot \phi(U^{(1)}e_{n_1} + \cdots + U^{(S)}e_{n_S})$

## Training Hyperparameters

- **Optimizer:** Adam
- **Learning rate:** 0.005
- **Weight decay:** 5.0
- **Loss function:** MSE loss
- **Network width:** $N = 500$ (multiplication), $N = 5000$ (addition with many terms)

## Setup and Installation

```bash
pip install -r requirements.txt
```

## Running Experiments

### Modular Addition (SLURM)

```bash
chmod +x run_addition.sh
sbatch run_addition.sh
```

### Modular Addition (Local)

```bash
python train.py \
    --task=addition \
    --p=97 \
    --num_terms=2 \
    --hidden_dim=500 \
    --power=2 \
    --train_fraction=0.5 \
    --lr=0.005 \
    --weight_decay=5.0 \
    --n_epochs=50000 \
    --device=cuda
```

### Modular Multiplication

```bash
python train.py \
    --task=multiplication \
    --p=97 \
    --hidden_dim=500 \
    --train_fraction=0.5 \
    --lr=0.005 \
    --weight_decay=5.0 \
    --n_epochs=50000 \
    --device=cuda
```

### Addition with Multiple Terms

```bash
python train.py \
    --task=addition \
    --p=97 \
    --num_terms=5 \
    --hidden_dim=5000 \
    --power=2 \
    --device=cuda
```

## Expected Results

- Networks learn to solve modular polynomials through grokking
- Trained networks' weights should approximate the analytical solutions described in the paper
- Higher weight decay (5.0) accelerates grokking compared to standard values

## Output Files

- `logs/training_history.json`: Training and test metrics
- `checkpoints/checkpoint_epoch_*.pt`: Model checkpoints

## Citation

```bibtex
@article{doshi2024grokking,
  title={Grokking Modular Polynomials},
  author={Doshi, Darshil and He, Tianyu and Das, Aritra and Gromov, Andrey},
  journal={arXiv preprint arXiv:2406.03495},
  year={2024}
}
```

